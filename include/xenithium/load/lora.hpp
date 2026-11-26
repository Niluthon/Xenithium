// Xenithium · a PEFT LoRA adapter (adapter_model.safetensors + adapter_config.json) loaded onto the cards at
// startup. The adapter's A/B are read (F32, converted to bf16 at load), sharded per card by the module's
// placement class, packed in the dense bf16 layout and uploaded; the per-(card, layer) LoraLayer bundles the
// packed pointers the inference path hands to kernels::lora_apply.
//
// Placement (the standard column / row-parallel split, C = the card count):
//   column-parallel (q/k/v, gate/up): A replicated [r16][k], B row-split [n/C][r16]; the delta lands in the
//     card's output slice (out_off the section offset, n the card's slice). No cross-card traffic.
//   row-parallel (o, down): A col-split [r16][k/C], B replicated [n][r16]; the partial (B·(A·x_card)) is added
//     to the card's fp32 part buffer (out_off 0, n the full output) before the existing pair_reduce. No new wire.
// r is padded to r16 = ceil(r/16)·16 (the dense GEMM body needs n % 16 == 0). scale = lora_alpha / r.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "xenithium/kernels/lora.hpp"

namespace xenithium {
class Fabric;

namespace load {

// The base model's geometry the adapter is validated against (from the arch's Config).
struct LoraGeom {
  int cards = 1;
  int layers = 0;
  int hidden = 0;
  int q_heads = 0, kv_heads = 0, head_dim = 0;
  bool attn_output_gate = false;
  int inter = 0;
  std::vector<bool> full_attention;  // per layer: true = GQA (q/k/v/o), false = GDN
  std::string prefix = "model.language_model.";  // the base module path prefix (the adapter name is "base_model.model." + it)
};

// The loaded adapter: the per-(card, layer) LoraLayer bundles, the per-card scratch, and the adapter's scale.
class LoraAdapter {
 public:
  LoraAdapter(Fabric &fab, const std::string &adapter_dir, const LoraGeom &geom, int threads);
  ~LoraAdapter();
  LoraAdapter(const LoraAdapter &) = delete;
  LoraAdapter &operator=(const LoraAdapter &) = delete;

  int cards() const { return C_; }
  int layers() const { return geom_.layers; }
  int r16() const { return r16_; }
  int n_max() const { return n_max_; }  // the largest module output on a card (sizes the scratch delta)
  const std::string &name() const { return name_; }
  float scale() const { return scale_; }
  const kernels::LoraLayer &layer(int card, int layer) const { return layers_.at(static_cast<size_t>(card) * geom_.layers + layer); }

  // The per-card scratch (mid + delta), sized to m_max rows; called once after load, before any step.
  void reserve_scratch(Fabric &fab, int m_max);
  kernels::LoraScratch &scratch(int card) { return sc_.at(static_cast<size_t>(card)); }
  kernels::LoraScratch &scratch(int card) const { return sc_.at(static_cast<size_t>(card)); }

 private:
  Fabric *fab_ = nullptr;
  LoraGeom geom_;
  int C_ = 0;
  int r16_ = 0;
  int n_max_ = 0;
  float scale_ = 1.f;
  std::string name_;
  std::vector<kernels::LoraLayer> layers_;  // [card * layers + layer]
  mutable std::vector<kernels::LoraScratch> sc_;    // [card]
  std::vector<void *> owned_;               // every device allocation (released in the destructor)
};

}  // namespace xenithium::load
}  // namespace xenithium
