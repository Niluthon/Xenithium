#pragma once
#include <memory>
#include "xenithium/arch/qwen35/trained_qsa.hpp"
#include "xenithium/kernels/attention/gqa.hpp"
#include "xenithium/rt/fabric.hpp"

namespace xenithium::arch::qwen35::trained_qsa {
inline constexpr int kMaxContext = 262144;
// Per-rank workspace is shared in compute-queue order. Each covered layer owns
// its pooled history, open-block tail and transient rows until accept().
class Device {
 public:
  Device(Fabric &fab, std::span<const Layer> weights, int slots, int capacity, int prefill_rows, int step_rows, bool xmx_prefill = false);
  static size_t bytes_per_rank(size_t layers, int hidden, int slots, int capacity, int prefill_rows, int step_rows, bool xmx_prefill = false);
  ~Device();
  // thw_row0 >= 0: the rows' M-RoPE positions start at that row of set_rows_thw's rows; -1: position + the slot's delta
  kernels::GqaTrainedPrefill prefill(int card, const kernels::gqa_bf16 *x, int rows, int slot, int pos, int bank_layer = 0, int fa_rows = 0, int thw_row0 = -1);
  // M-RoPE (image input): the next prefill call's (t, h, w) per row (host, rows x 3; null clears), a slot's delta (on
  // the card's compute queue, before the work that reads it), and mrope_section's h / w frequencies
  void set_rows_thw(int card, const int32_t *thw, int rows);
  void set_delta(int card, int slot, int delta);
  void set_mrope(int sec_h, int sec_w);
  void stage(int card, const kernels::gqa_bf16 *x, int rows, int kv_heads, kernels::GqaStep &step, int bank_layer = 0, int model_layer = -1,
             uint32_t track=~uint32_t(0),bool sparse=true,uint32_t dense_rows=0);
  void accept(int slot, int pos, int row0, int keep, bool tree_batch=false, uint64_t path=0);
  // Submission only on the compute queues. The prefix proof is owned by Model.
  size_t copy_aligned_prefix(int dst, int src, int pos);
  struct Selection { int eligible, pages; std::vector<int32_t> ids; };
  std::vector<int32_t> read_logical_pages(int card, int row, int pages);
  void check_selector_reference(int card, int row, const Selection &picked);
  // Diagnostic readback only, after a stage that actually scored this row.
  Selection read_selection(int card, int row);
 private:
  kernels::GqaTrainedPrefill prefill_plan(int card,int rows,int slot,int pos,int bank,int fa_rows,bool skip_subtiles=false);
  struct Impl;
  std::unique_ptr<Impl> p_;
};
}
