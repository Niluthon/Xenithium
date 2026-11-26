#include "xenithium/arch/qwen35/mlp.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "xenithium/kernels/gemm/w4a16_gemm.hpp"

namespace xenithium::arch::qwen35 {

Mlp::Mlp(Fabric &fab, const Config &cfg, int rows) : fab_(fab), cfg_(cfg), step_rows_(rows) {
  if (rows < 1 || rows > kRows) throw std::invalid_argument("qwen35 mlp: rows 1..kRows");
  const int C = static_cast<int>(fab.size());
  if (C > 2) throw std::runtime_error("qwen35 mlp: the pair wire pairs exactly two cards");
  if (cfg.inter % (C * cfg.w_group) || cfg.hidden % 64) throw std::runtime_error("qwen35 mlp: intermediate / hidden do not split");
  hidden_ = cfg.hidden;
  ipc_ = cfg.inter / C;
  cards_.resize(static_cast<size_t>(C));
  ev_.resize(static_cast<size_t>(C));
  for (int c = 0; c < C; ++c) {
    Card &k = cards_[static_cast<size_t>(c)];
    auto dev = [&](size_t bytes) { return fab.alloc(static_cast<size_t>(c), std::max<size_t>(bytes, 4096)); };
    k.gu = static_cast<float *>(dev(static_cast<size_t>(step_rows_) * 2 * ipc_ * 4));
    k.h = static_cast<bf16 *>(dev(static_cast<size_t>(step_rows_) * ipc_ * 2));
    k.part = static_cast<float *>(dev(static_cast<size_t>(step_rows_) * hidden_ * 4));
    k.out = static_cast<bf16 *>(dev(static_cast<size_t>(step_rows_) * hidden_ * 2));
    k.silu = static_cast<float *>(dev(static_cast<size_t>(kernels::kSiluTabSize) * 4));
    if (C > 1) {
      k.wire.inbox = static_cast<uint32_t *>(fab.alloc_mailbox(static_cast<size_t>(c), kernels::pair_wire_words(hidden_) * 4));
      k.wire.error = static_cast<uint32_t *>(dev(64));
    }
  }
  for (int c = 0; c < C; ++c) {
    Card &k = cards_[static_cast<size_t>(c)];
    auto &q = fab.compute(static_cast<size_t>(c));
    // Dead rows of the GEMM inputs must hold finite values: zero them once.
    q.memset(k.gu, 0, static_cast<size_t>(step_rows_) * 2 * ipc_ * 4);
    q.memset(k.h, 0, static_cast<size_t>(step_rows_) * ipc_ * 2);
    q.memset(k.part, 0, static_cast<size_t>(step_rows_) * hidden_ * 4);
    kernels::swiglu_table(q, k.silu);
    if (C > 1) {
      k.wire.peer_inbox = cards_[static_cast<size_t>((c + 1) % C)].wire.inbox;
      kernels::pair_wire_init(q, hidden_, k.wire.inbox);
      q.memset(k.wire.error, 0, 64);
    }
    q.wait();
  }
}

Mlp::~Mlp() {
  for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).wait();
  for (Card &k : cards_)
    for (void *p : {(void *)k.gu, (void *)k.h, (void *)k.part, (void *)k.out, (void *)k.silu, (void *)k.wire.inbox, (void *)k.wire.error})
      if (p) fab_.release(p);
  for (Rows &r : rows_)
    for (void *p : {(void *)r.gu, h_ext_ ? nullptr : (void *)r.h, (void *)r.part, wire_ext_ ? nullptr : (void *)r.wire.inbox,
                    wire_ext_ ? nullptr : (void *)r.wire.error})
      if (p) fab_.release(p);
}

void Mlp::reserve_rows(int rows_max, const std::vector<bf16 *> &h, const std::vector<const kernels::PairRowsWire *> &wire) {
  if (rows_max <= rows_max_) return;
  if (rows_max_ > 0) throw std::logic_error("qwen35 mlp: reserve_rows once");
  const int C = static_cast<int>(cards_.size());
  if (!h.empty() && static_cast<int>(h.size()) != C) throw std::invalid_argument("qwen35 mlp: reserve_rows takes one h per card");
  if (!wire.empty() && static_cast<int>(wire.size()) != C) throw std::invalid_argument("qwen35 mlp: reserve_rows takes one wire per card");
  h_ext_ = !h.empty();
  wire_ext_ = !wire.empty();
  ext_wire_ = wire;
  rows_.resize(static_cast<size_t>(C));
  for (int c = 0; c < C; ++c) {
    Rows &r = rows_[static_cast<size_t>(c)];
    auto dev = [&](size_t bytes) { return fab_.alloc(static_cast<size_t>(c), std::max<size_t>(bytes, 4096)); };
    r.h = h_ext_ ? h[static_cast<size_t>(c)] : static_cast<bf16 *>(dev(static_cast<size_t>(rows_max) * ipc_ * 2));  // gu: on the first unfused call
    r.part = static_cast<float *>(dev(static_cast<size_t>(rows_max) * hidden_ * 4));
    if (C > 1 && wire_ext_) {
      if (wire[static_cast<size_t>(c)]->rows_max < rows_max || wire[static_cast<size_t>(c)]->n_max < hidden_)
        throw std::invalid_argument("qwen35 mlp: the caller's wire is short of rows_max rows");
      r.wire = {};  // run_rows reads the caller's wire at each call (its bf16 flag stays the MLP's own: fp32)
    } else if (C > 1) {
      r.wire.inbox = static_cast<uint32_t *>(fab_.alloc_mailbox(static_cast<size_t>(c), kernels::pair_rows_wire_words(rows_max, hidden_) * 4));
      r.wire.error = static_cast<uint32_t *>(dev(64));
      r.wire.rows_max = rows_max;
      r.wire.n_max = hidden_;
    }
  }
  for (int c = 0; c < C; ++c) {
    Rows &r = rows_[static_cast<size_t>(c)];
    auto &q = fab_.compute(static_cast<size_t>(c));
    if (C > 1 && !wire_ext_) {
      r.wire.peer_inbox = rows_[static_cast<size_t>((c + 1) % C)].wire.inbox;
      kernels::pair_rows_wire_init(q, rows_max, hidden_, r.wire.inbox);
      q.memset(r.wire.error, 0, 64);
    }
    q.wait();
  }
  rows_max_ = rows_max;
}

void Mlp::gate_up_rows(int c, int lo, int rows, const bf16 *x, const MlpWeights &w, const kernels::LoraLayer *ll, kernels::LoraScratch *ls) {
  if (lo < 0 || rows < 1 || lo + rows > rows_max_) throw std::invalid_argument("qwen35 mlp: gate_up_rows past reserve_rows");
  Rows &r = rows_[static_cast<size_t>(c)];
  auto &q = fab_.compute(static_cast<size_t>(c));
  const bool gu_lora = ll && (ll->gate.present() || ll->up.present());
  if (gu_lora) {  // the gate | up delta needs the fp32 gu: the gate | up -> swiglu path (the same bits)
    if (!r.gu) r.gu = static_cast<float *>(fab_.alloc(static_cast<size_t>(c), std::max<size_t>(static_cast<size_t>(rows_max_) * 2 * ipc_ * 4, 4096)));
    ev_[static_cast<size_t>(c)].gu =
        kernels::w4a16_gemm_pf(q, rows, x + static_cast<size_t>(lo) * hidden_, hidden_, w.gate_up.w, w.gate_up.scales,
                               r.gu + static_cast<size_t>(lo) * 2 * ipc_, 2 * ipc_, 2 * ipc_, hidden_);
    if (ll->gate.present())
      ev_[static_cast<size_t>(c)].gu = kernels::lora_apply(q, rows, x + static_cast<size_t>(lo) * hidden_, ll->gate, r.gu + static_cast<size_t>(lo) * 2 * ipc_, 2 * ipc_, *ls, 0);
    if (ll->up.present())  // out_off = ipc: the base pointer, the offset lands the delta on the up section
      ev_[static_cast<size_t>(c)].gu = kernels::lora_apply(q, rows, x + static_cast<size_t>(lo) * hidden_, ll->up, r.gu + static_cast<size_t>(lo) * 2 * ipc_, 2 * ipc_, *ls, 0);
    ev_[static_cast<size_t>(c)].silu = kernels::swiglu(q, rows, ipc_, r.gu + static_cast<size_t>(lo) * 2 * ipc_, r.h + static_cast<size_t>(lo) * ipc_);
    kernels::w4a16_pf_a8_input_clear(q);
    return;
  }
  kernels::w4a16_pf_set_h_codes_only(true);  // h's only reader is down (run_rows); on W4A8 it takes the codes
  ev_[static_cast<size_t>(c)].gu = ev_[static_cast<size_t>(c)].silu =
      kernels::w4a16_gemm_pf_swiglu(q, rows, x + static_cast<size_t>(lo) * hidden_, hidden_, w.gate_up.w, w.gate_up.scales,
                                    r.h + static_cast<size_t>(lo) * ipc_, ipc_, ipc_, hidden_);
  kernels::w4a16_pf_set_h_codes_only(false);
}

void Mlp::run_rows(int m, const std::vector<const bf16 *> &x, const std::vector<MlpWeights> &w, const std::vector<bf16 *> &out, bool reduce, bool fused,
                   const kernels::PairSplitSend *const *send, bool gu_ready, int parity, const load::LoraAdapter *lora, int layer) {
  const size_t C = cards_.size();
  if (m < 1 || m > rows_max_) throw std::invalid_argument("qwen35 mlp: run_rows past reserve_rows");
  if (x.size() != C || w.size() != C || out.size() != C) throw std::invalid_argument("qwen35 mlp: one input, weight set and output per card");
  if (send && (reduce || C != 2)) throw std::invalid_argument("qwen35 mlp: send is the split sum's (two cards, reduce false)");
  if (gu_ready && !fused) throw std::invalid_argument("qwen35 mlp: gu_ready is the fused path's");
  sent_.assign(C, sycl::event{});
  for (size_t c = 0; c < C; ++c) {
    Rows &r = rows_[c];
    auto &q = fab_.compute(c);
    const kernels::LoraLayer *ll = lora ? &lora->layer(static_cast<int>(c), layer) : nullptr;
    kernels::LoraScratch *ls = lora ? &lora->scratch(static_cast<int>(c)) : nullptr;
    const bool gu_lora = ll && (ll->gate.present() || ll->up.present());
    const bool dn_lora = ll && ll->down.present();
    if ((!fused || gu_lora) && !r.gu)
      r.gu = static_cast<float *>(fab_.alloc(c, std::max<size_t>(static_cast<size_t>(rows_max_) * 2 * ipc_ * 4, 4096)));  // the unfused path
    if (gu_ready) {
      // h filled by gate_up_rows (its gate | up delta applied there)
    } else if (fused && !gu_lora) {
      // h's only reader is down below; on W4A8 it takes the codes (a split down starts at row m / 2: codes are found
      // from a row that is a multiple of 4)
      const bool codes_only = !send || (m / 2) % 4 == 0;
      kernels::w4a16_pf_set_h_codes_only(codes_only);
      ev_[c].gu = ev_[c].silu = kernels::w4a16_gemm_pf_swiglu(q, m, x[c], hidden_, w[c].gate_up.w, w[c].gate_up.scales, r.h, ipc_, ipc_, hidden_);
      kernels::w4a16_pf_set_h_codes_only(false);
    } else {
      ev_[c].gu = kernels::w4a16_gemm_pf(q, m, x[c], hidden_, w[c].gate_up.w, w[c].gate_up.scales, r.gu, 2 * ipc_, 2 * ipc_, hidden_);
      if (ll && ll->gate.present()) ev_[c].gu = kernels::lora_apply(q, m, x[c], ll->gate, r.gu, 2 * ipc_, *ls, 0);
      if (ll && ll->up.present()) ev_[c].gu = kernels::lora_apply(q, m, x[c], ll->up, r.gu, 2 * ipc_, *ls, 0);  // out_off = ipc
      ev_[c].silu = kernels::swiglu(q, m, ipc_, r.gu, r.h);
      kernels::w4a16_pf_a8_input_clear(q);  // h rewritten without codes (drops a W4A8 GLU's registered h codes)
    }
    auto down = [&](int lo, int rows) {
      return kernels::w4a16_gemm_pf(q, rows, r.h + static_cast<size_t>(lo) * ipc_, ipc_, w[c].down.w, w[c].down.scales, r.part + static_cast<size_t>(lo) * hidden_,
                                    hidden_, hidden_, ipc_);
    };
    if (send && send[c] && !dn_lora) {  // the peer's rows first, their copy beside the own rows (a LoRA down: the plain sum)
      const kernels::PairSplitSend &sd = *send[c];
      int lo = 0, hi = 0;
      kernels::pair_split_owner_rows(1 - static_cast<int>(c), m, &lo, &hi);
      if (hi > lo) down(lo, hi - lo);
      sent_[c] = kernels::pair_split_send(q, sd.side, static_cast<int>(c), m, hidden_, r.part, *sd.sp, sd.seq);
      kernels::pair_split_owner_rows(static_cast<int>(c), m, &lo, &hi);
      if (hi > lo) ev_[c].down = down(lo, hi - lo);
    } else {
      ev_[c].down = down(0, m);
    }
    if (dn_lora) ev_[c].down = kernels::lora_apply(q, m, r.h, ll->down, r.part, hidden_, *ls, 0);  // the down delta before the sum
    if (reduce) {
      kernels::PairRowsWire xw{};
      if (wire_ext_ && C > 1) {
        xw = *ext_wire_[c];
        xw.bf16 = false;  // the MLP's prefill sums stay fp32
      }
      ev_[c].reduce = kernels::pair_reduce_rows(q, m, hidden_, r.part, C > 1 ? (wire_ext_ ? &xw : &r.wire) : nullptr, wire_ext_ ? parity : rows_parity_, out[c]);
    }
  }
  if (reduce && !wire_ext_) rows_parity_ ^= 1;
}

MlpWeights Mlp::weights(const load::DenseStore &dense, int card, int layer) const {
  const std::string P = cfg_.prefix + "layers." + std::to_string(layer) + ".mlp.";
  MlpWeights w;
  w.gate_up = dense.at(card, cfg_.packed_name(P + "gate_proj")).w4();
  w.down = dense.at(card, cfg_.packed_name(P + "down_proj")).w4();
  return w;
}

void Mlp::run(int rows, const std::vector<const bf16 *> &x, const std::vector<MlpWeights> &w, int gu_ksplit, int dn_ksplit,
              const load::LoraAdapter *lora, int layer) {
  const size_t C = cards_.size();
  if (rows < 1 || rows > step_rows_) throw std::invalid_argument("qwen35 mlp: rows");
  if (x.size() != C || w.size() != C) throw std::invalid_argument("qwen35 mlp: one input and one weight set per card");
  for (size_t c = 0; c < C; ++c)
    if (w[c].gate_up.n != 2 * ipc_ || w[c].gate_up.k != hidden_ || w[c].down.n != hidden_ || w[c].down.k != ipc_)
      throw std::invalid_argument("qwen35 mlp: weight slices do not match the shape");
  for (size_t c = 0; c < C; ++c) {
    Card &k = cards_[c];
    auto &q = fab_.compute(c);
    const kernels::LoraLayer *ll = lora ? &lora->layer(static_cast<int>(c), layer) : nullptr;
    kernels::LoraScratch *ls = lora ? &lora->scratch(static_cast<int>(c)) : nullptr;
    const bool gu_lora = ll && (ll->gate.present() || ll->up.present());
    if (fused_ && !gu_lora) {  // swiglu in the gate|up epilogue (same bits); a LoRA gate | up needs the fp32 gu
      ev_[c].gu = ev_[c].silu = kernels::w4a16_gemm_rows_swiglu(q, rows, x[c], w[c].gate_up.w, w[c].gate_up.scales, k.silu, k.h, ipc_, hidden_, gu_ksplit);
    } else {
      ev_[c].gu = kernels::w4a16_gemm_rows(q, rows, x[c], w[c].gate_up.w, w[c].gate_up.scales, k.gu, 2 * ipc_, hidden_, gu_ksplit);
      if (ll && ll->gate.present()) ev_[c].gu = kernels::lora_apply(q, rows, x[c], ll->gate, k.gu, 2 * ipc_, *ls, 0);
      if (ll && ll->up.present()) ev_[c].gu = kernels::lora_apply(q, rows, x[c], ll->up, k.gu, 2 * ipc_, *ls, 0);  // out_off = ipc
      ev_[c].silu = kernels::swiglu(q, rows, ipc_, k.gu, k.h);
    }
    ev_[c].down = kernels::w4a16_gemm_rows(q, rows, k.h, w[c].down.w, w[c].down.scales, k.part, hidden_, ipc_, dn_ksplit);
    if (ll && ll->down.present()) ev_[c].down = kernels::lora_apply(q, rows, k.h, ll->down, k.part, hidden_, *ls, 0);  // the down delta before the reduce
    ev_[c].reduce = kernels::pair_reduce(q, rows, hidden_, k.part, C > 1 ? &k.wire : nullptr, parity_, k.out);
  }
  parity_ ^= 1;
}

void Mlp::wait() {
  for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).wait_and_throw();
}

uint32_t Mlp::wire_error(int c) {
  uint32_t e = 0, f = 0;
  if (cards_[static_cast<size_t>(c)].wire.error) fab_.compute(static_cast<size_t>(c)).memcpy(&e, cards_[static_cast<size_t>(c)].wire.error, 4).wait();
  if (!rows_.empty() && rows_[static_cast<size_t>(c)].wire.error)
    fab_.compute(static_cast<size_t>(c)).memcpy(&f, rows_[static_cast<size_t>(c)].wire.error, 4).wait();
  return e | f;
}

}  // namespace xenithium::arch::qwen35
