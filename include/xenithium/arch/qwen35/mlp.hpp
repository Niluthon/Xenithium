// Xenithium · qwen35 dense SwiGLU MLP on the cards.
//
// TP by intermediate column over the two cards: each card holds gate | up rows [2 I/2][H] as one W4
// matrix (DenseStore: gate_proj's slice, then up_proj's) and down's k half [H][I/2]. Per step and
// layer, on each card: gu = x . Wgu^T (w4a16_gemm_rows) -> h = swiglu(gu) -> part = h . Wd^T ->
// out = bf16(part_0 + part_1) through the mailbox (pair_reduce). Both cards end with the same bf16
// out; the caller adds it to the residual (rms_norm's bf16 add). Four launches per layer, or three with swiglu in
// the gate|up epilogue (set_fused, bit for bit). One card (TP1): the whole matrices, out = bf16(part) without a wire.
#pragma once

#include <cstdint>
#include <vector>

#include "xenithium/arch/qwen35/arch.hpp"
#include "xenithium/kernels/gemm/mlp.hpp"
#include "xenithium/kernels/lora.hpp"
#include "xenithium/kernels/rows.hpp"
#include "xenithium/kernels/comm/wire.hpp"
#include "xenithium/load/dense_store.hpp"
#include "xenithium/load/lora.hpp"
#include "xenithium/rt/fabric.hpp"

namespace xenithium::arch::qwen35 {

struct MlpWeights {
  load::W4Weights gate_up;  // [2 ipc][H]
  load::W4Weights down;     // [H][ipc]
};

class Mlp {
 public:
  static constexpr int kRows = kernels::kStepRows32;
  using bf16 = sycl::ext::oneapi::bfloat16;

  struct Card {
    float *gu = nullptr;    // [kRows][2 ipc]
    bf16 *h = nullptr;      // [kRows][ipc]
    float *part = nullptr;  // [kRows][H]
    bf16 *out = nullptr;    // [kRows][H]
    float *silu = nullptr;  // kernels:swiglu_table (s(g) by g's bf16 bits) for the fused decode gate|up
    kernels::PairWire wire;
  };
  struct Events {
    sycl::event gu, silu, down, reduce;
  };
  // Many rows (prefill): chunk-sized scratch and mailbox per card.
  struct Rows {
    float *gu = nullptr;    // [rows_max][2 ipc] (allocated by the first unfused run_rows)
    bf16 *h = nullptr;      // [rows_max][ipc]
    float *part = nullptr;  // [rows_max][H]
    kernels::PairRowsWire wire;
  };

  // rows: the rows its step buffers hold (<= kRows; the model's step / call rows)
  Mlp(Fabric &fab, const Config &cfg, int rows = kRows);
  Mlp(const Mlp &) = delete;
  Mlp &operator=(const Mlp &) = delete;
  ~Mlp();

  // The layer's slices on card c from the store (layers.L.mlp.{gate_proj,down_proj}, the checkpoint's packed name).
  MlpWeights weights(const load::DenseStore &dense, int card, int layer) const;
  // One layer on every card: x[c] bf16 [rows][H] (rows finite), w[c] the card's slices; out in card(c).out.
  // Submits on both cards and returns; the mailbox parity alternates per call. lora (null: no adapter): the layer's
  // gate | up | down deltas (lora->layer(c, layer)) added to the fp32 gu / part before swiglu / the reduce; the gate |
  // up delta forces the gate | up -> swiglu path (the fused epilogue has no fp32 gu).
  void run(int rows, const std::vector<const bf16 *> &x, const std::vector<MlpWeights> &w, int gu_ksplit = 0, int dn_ksplit = 0,
           const load::LoraAdapter *lora = nullptr, int layer = 0);
  // Prefill: allocate the many-row scratch for up to rows_max rows (once). h (optional, one per card): the caller's
  // [rows_max][ipc] bf16 for h (the model's shared stage scratch; not released here). wire (optional, one per card): the
  // caller's rows wire for the reduce (the model's mixer wire, its parity passed to run_rows), no own mailbox.
  void reserve_rows(int rows_max, const std::vector<bf16 *> &h = {}, const std::vector<const kernels::PairRowsWire *> &wire = {});
  // m rows (<= rows_max) on every card: gu = w4a16_gemm_pf -> swiglu -> down w4a16_gemm_pf -> pair_reduce_rows into
  // out[c] bf16 [m][H]. The same rounding points as run(); the GEMMs sum k in another order (fp32 only).
  // reduce false: leave the fp32 partial in rows_part(c) for the caller's split sum; out unused.
  // fused = gate|up with swiglu in its epilogue (w4a16_gemm_pf_swiglu: same bits, no fp32 gu); send (reduce
  // false only): null, or per card the split's send - down's peer rows first, their copy on the side queue beside the
  // own rows (the same path as gdn_rows'); the copies' events in sent(c).
  // gu_ready (fused only): h already holds the gate|up + swiglu of every row (gate_up_rows in row parts).
  // parity: the reduce's mailbox parity with reserve_rows' caller wire (the caller alternates it), else unused.
  void run_rows(int m, const std::vector<const bf16 *> &x, const std::vector<MlpWeights> &w, const std::vector<bf16 *> &out, bool reduce = true,
                bool fused = true, const kernels::PairSplitSend *const *send = nullptr, bool gu_ready = false, int parity = 0,
                const load::LoraAdapter *lora = nullptr, int layer = 0);
  // Card c's fused gate|up + swiglu of rows [lo, lo + rows) of x (bf16 [.][H], x pointing at row 0) into h. lora (null:
  // no adapter): the gate | up delta (lora->layer(c, layer)) added to the fp32 gu before swiglu (forces the gate | up ->
  // swiglu path).
  void gate_up_rows(int c, int lo, int rows, const bf16 *x, const MlpWeights &w, const kernels::LoraLayer *ll = nullptr,
                    kernels::LoraScratch *ls = nullptr);
  const sycl::event &sent(int c) const { return sent_[static_cast<size_t>(c)]; }
  const float *rows_part(int c) const { return rows_[static_cast<size_t>(c)].part; }
  void wait();
  const Card &card(int c) const { return cards_[static_cast<size_t>(c)]; }
  const Events &events(int c) const { return ev_[static_cast<size_t>(c)]; }
  int hidden() const { return hidden_; }
  int ipc() const { return ipc_; }
  uint32_t wire_error(int c);
  void set_wire_bf16(bool on) {  // the decode sum's bf16 wire (kernels:PairWire:bf16)
    for (Card &k : cards_) k.wire.bf16 = on;
  }
  void set_wire_i8(bool on) {  // its int8 wire (kernels:PairWire:i8)
    for (Card &k : cards_) k.wire.i8 = on;
  }
  // run() as gate|up with swiglu in its epilogue (kernels::w4a16_gemm_rows_swiglu, the same bits; three launches per
  // layer). false = the gate|up -> swiglu -> down.
  void set_fused(bool on) { fused_ = on; }
  // the decode wire's parity (step graphs: recorded state)
  int parity() const { return parity_; }
  void set_parity(int p) { parity_ = p & 1; }

 private:
  Fabric &fab_;
  Config cfg_;
  int hidden_ = 0, ipc_ = 0, parity_ = 0, rows_max_ = 0, rows_parity_ = 0, step_rows_ = kRows;
  bool fused_ = true;
  bool h_ext_ = false;     // rows_[c].h is the caller's (reserve_rows)
  bool wire_ext_ = false;  // the reduce uses the caller's wire (reserve_rows)
  std::vector<const kernels::PairRowsWire *> ext_wire_;
  std::vector<Card> cards_;
  std::vector<Rows> rows_;
  std::vector<Events> ev_;
  std::vector<sycl::event> sent_;
};

}  // namespace xenithium::arch::qwen35
