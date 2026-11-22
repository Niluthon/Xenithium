// Xenithium · qwen38 step ends on the cards: embedding in, final mixer + lm_head + greedy out.
//
// Per card (TP over the vocabulary, everything else replicated):
//   embed:     res[row][s] = embed_tokens[id]     (table in pinned host, read over PCIe)
//   final mix: combine (optional pending block) + norm, down, silu, up, gate mix -> out[row]
//   head:      logits = out . lm_head_slice^T (int8 W8A16), greedy pick across both cards
// Both cards compute the replicated parts on their own copy, so they end with identical
// residuals and the same tokens without any exchange except the pick's two words per row.
#pragma once

#include <cstdint>
#include <vector>

#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/kernels/residual/hc.hpp"
#include "xenithium/kernels/head/head.hpp"
#include "xenithium/load/dense_store.hpp"
#include "xenithium/rt/fabric.hpp"

namespace xenithium::arch::qwen38 {

class Head {
 public:
  // Rows the head's buffers and the model's per-row arrays hold (64, kernels::kMaxRows). What a step may carry is the
  // model's step_rows_cap() (32 unless ModelOptions::step_rows_cap).
  static constexpr int kRows = kernels::kMaxRows;
  using bf16 = sycl::ext::oneapi::bfloat16;

  struct Card {
    kernels::HcState st;     // residual + down output, double-buffered (res[i]: [kRows][hc * hidden])
    bf16 *xn = nullptr;      // [kRows][hc * hidden]
    bf16 *s = nullptr;       // [kRows][lowrank]
    float *gate = nullptr;   // [kRows][hc * hidden]
    bf16 *out = nullptr;     // [kRows][hidden]
    float *logits = nullptr; // [kRows][vocab / cards]
    int32_t *tokens = nullptr;
    uint64_t *local_keys = nullptr;
    void *scratch = nullptr;
    kernels::PickWire wire;
  };
  struct Events {  // the last launch of each stage, per card (profiling)
    sycl::event embed, norm, down, mix, lm, chunks, pick;  // mix = the mixer's last launch
  };

  Head(Fabric &fab, const load::DenseStore &dense, const Config &cfg);
  Head(const Head &) = delete;
  Head &operator=(const Head &) = delete;
  ~Head();

  // ids: host array of `rows` token ids for the next embed (written into a pinned host buffer
  // that the embed launches read over PCIe, as the PLE rows: no copy, no wait; call it only while no
  // embed of an earlier step is still queued, i.e. after that step's wait).
  void set_ids(const int32_t *ids, int rows);
  // Each stage submits on both cards and returns without waiting.
  void embed(int rows);
  // Final mixer. block (per card, empty = none pending) is the last block's output, combined
  // with the injection logits the last layer mixer left in the card's state.
  void final_mix(int rows, const std::vector<bf16 *> &block = {}, kernels::HcPath path = kernels::HcPath::three);
  // A step's rows by owner (kernels/comm/wire.hpp; four cards): the final mixer on each card's own rows
  // (kernels::owner_rows) of the state, block and out. The caller shares out (lm_pick reads every row on every card) and
  // the residual (kernels::owner_share).
  void final_mix_own(int rows, const std::vector<bf16 *> &block);
  kernels::HcState &state(int c) { return cards_[static_cast<size_t>(c)].st; }
  bf16 *residual(int c) const { return cards_[static_cast<size_t>(c)].st.residual(); }
  void lm_pick(int rows);
  // lm_head + pick on other inputs (the MTP drafter's final mixer output): in[c] bf16 [kRows][hidden]
  // (all rows finite), tokens[c] device int32 [rows], logits[c] fp32 [kRows][logits_floats() / kRows]
  // (the caller's: the target's logits stay readable for host sampling).
  // w: other lm_head weights per card (the same vocab slice; the drafter's int2 draft head); null = lm_head.
  void lm_pick(int rows, const std::vector<const bf16 *> &in, const std::vector<int32_t *> &tokens, const std::vector<float *> &logits,
               const std::vector<kernels::DenseWeights> *w = nullptr);
  size_t logits_floats() const { return static_cast<size_t>(kRows) * vpc_; }
  // The embedding table card c gathers from (pinned host).
  const bf16 *embed_table(int c) const { return embed_[static_cast<size_t>(c)]; }
  void wait();
  std::vector<int32_t> tokens(int card, int rows);

  const Card &card(int c) const { return cards_[static_cast<size_t>(c)]; }
  // The final mixer's and lm_head's weights on card c (the many-row head of prompt logprobs).
  const kernels::HcWeights &final_weights(int c) const { return mix_[static_cast<size_t>(c)]; }
  const kernels::DenseWeights &lm_weights(int c) const { return lm_[static_cast<size_t>(c)]; }
  const Events &events(int c) const { return ev_[static_cast<size_t>(c)]; }
  const kernels::HcShape &hc_shape() const { return hc_; }
  int vocab_per_card() const { return vpc_; }
  int lora_ld() const { return lora_ld_; }
  kernels::HcScratch scratch(int c) const {
    const Card &k = cards_[static_cast<size_t>(c)];
    return {k.xn, k.s, k.gate};
  }
  uint32_t wire_error(int c);

 private:
  Fabric &fab_;
  kernels::HcShape hc_;
  int vpc_ = 0, lora_ld_ = 0;
  std::vector<Card> cards_;
  bool host_pick_ = false;
  int32_t *ids_host_ = nullptr;  // the next embed's token ids, pinned host [kPickMaxRows]
  std::vector<Events> ev_;
  // weights per card
  std::vector<const bf16 *> embed_;
  std::vector<kernels::HcWeights> mix_;
  std::vector<kernels::DenseWeights> lm_;
};

}  // namespace xenithium::arch::qwen38
