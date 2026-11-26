// Xenithium · the PEFT LoRA adapter loader (load/lora.hpp). Reads the adapter's A/B (F32, converted to
// bf16 at load), validates them against the base geometry, shards per card by the module's placement
// class, packs in the dense bf16 layout and uploads; the per-(card, layer) LoraLayer bundles the packed
// pointers the inference path hands to kernels::lora_apply.
#include "xenithium/load/lora.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/rows.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/rt/fabric.hpp"
#include "xenithium/util/json.hpp"

namespace xenithium::load {

namespace {

// f32 -> bf16 (1-8-7), round-to-nearest-even (dense_store.cpp's float_to_bf16).
uint16_t f32_to_bf16(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return static_cast<uint16_t>((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
}

// Read an F32 tensor's elements from its shard (pread) and convert to bf16 (row-major, numel elements).
void read_f32_bf16(const model::Checkpoint &ck, const model::TensorInfo &ti, std::vector<uint16_t> &out) {
  if (ti.dtype != model::DType::f32) throw std::runtime_error("lora: " + ti.name + " is not F32 (the adapter's A/B must be float32)");
  const size_t n = static_cast<size_t>(ti.numel());
  out.assign(n, 0);
  std::vector<float> f32(n);
  const int fd = ::open(ck.shards()[ti.shard].path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("lora: cannot open " + ck.shards()[ti.shard].path);
  size_t got = 0;
  const size_t want = ti.bytes;
  while (got < want) {
    const ssize_t r = ::pread(fd, reinterpret_cast<char *>(f32.data()) + got, want - got, static_cast<off_t>(ti.offset + got));
    if (r <= 0) break;
    got += static_cast<size_t>(r);
  }
  ::close(fd);
  if (got != want) throw std::runtime_error("lora: short read of " + ti.name);
  for (size_t i = 0; i < n; ++i) out[i] = f32_to_bf16(f32[i]);
}

// Pad a [rows][cols] bf16 matrix to [rows_pad][cols_pad] (zero the extra rows / cols), row-major.
void pad2(const std::vector<uint16_t> &src, int rows, int cols, int rows_pad, int cols_pad, std::vector<uint16_t> &out) {
  out.assign(static_cast<size_t>(rows_pad) * cols_pad, 0);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) out[static_cast<size_t>(r) * cols_pad + c] = src[static_cast<size_t>(r) * cols + c];
}
// Rows [r0, r0 + nrows) of a [rows][cols] row-major matrix -> [nrows][cols].
void rows_slice(const std::vector<uint16_t> &src, int cols, int r0, int nrows, std::vector<uint16_t> &out) {
  out.assign(static_cast<size_t>(nrows) * cols, 0);
  for (int r = 0; r < nrows; ++r)
    for (int c = 0; c < cols; ++c) out[static_cast<size_t>(r) * cols + c] = src[static_cast<size_t>(r0 + r) * cols + c];
}
// Cols [c0, c0 + ncols) of a [rows][cols] row-major matrix -> [rows][ncols].
void cols_slice(const std::vector<uint16_t> &src, int rows, int cols, int c0, int ncols, std::vector<uint16_t> &out) {
  out.assign(static_cast<size_t>(rows) * ncols, 0);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < ncols; ++c) out[static_cast<size_t>(r) * ncols + c] = src[static_cast<size_t>(r) * cols + (c0 + c)];
}

// The module's placement and full dims.
enum class Mod { Q, K, V, O, Gate, Up, Down };
struct ModInfo {
  Mod mod;
  bool col;      // true = column-parallel, false = row-parallel
  int full_n;    // B's output dim (full)
  int full_k;    // A's input dim (full)
  int out_off;   // the offset of this module's B rows within the target buffer (column-parallel)
};

const char *mod_name(Mod m) {
  switch (m) {
    case Mod::Q: return "q_proj";
    case Mod::K: return "k_proj";
    case Mod::V: return "v_proj";
    case Mod::O: return "o_proj";
    case Mod::Gate: return "gate_proj";
    case Mod::Up: return "up_proj";
    case Mod::Down: return "down_proj";
  }
  return "";
}

// Map a module name to its ModInfo (false when unsupported). The out_off is the module's B rows' offset within
// the per-card target buffer (the qkv / gu row is the card's slice, so the section offsets are per-card).
bool mod_info(const std::string &name, const LoraGeom &g, ModInfo &out) {
  const int C = g.cards;
  const int q_n = g.q_heads * (g.attn_output_gate ? 2 : 1) * g.head_dim;  // the q section (query | gate), full
  const int k_n = g.kv_heads * g.head_dim;                                // the k / v section, full
  const int q_hd = g.q_heads * g.head_dim;                                // the attention output (query-only)
  if (name == "q_proj") { out = {Mod::Q, true, q_n, g.hidden, 0}; return true; }
  if (name == "k_proj") { out = {Mod::K, true, k_n, g.hidden, q_n / C}; return true; }  // the per-card q section width
  if (name == "v_proj") { out = {Mod::V, true, k_n, g.hidden, (q_n + k_n) / C}; return true; }
  if (name == "o_proj") { out = {Mod::O, false, g.hidden, q_hd, 0}; return true; }
  if (name == "gate_proj") { out = {Mod::Gate, true, g.inter, g.hidden, 0}; return true; }
  if (name == "up_proj") { out = {Mod::Up, true, g.inter, g.hidden, g.inter / C}; return true; }  // the per-card gate width
  if (name == "down_proj") { out = {Mod::Down, false, g.hidden, g.inter, 0}; return true; }
  return false;
}

// Parse an adapter tensor name -> (layer, module, is_a). False when not a LoRA tensor.
bool parse_name(const std::string &name, const LoraGeom &g, int &layer, std::string &module, bool &is_a) {
  const std::string pre = "base_model.model." + g.prefix + "layers.";
  if (name.rfind(pre, 0) != 0) return false;
  const std::string rest = name.substr(pre.size());  // "{l}.{module_path}.lora_{A,B}.weight"
  const auto dot = rest.find('.');
  if (dot == std::string::npos || dot == 0) return false;
  int l = 0;
  try { l = std::stoi(rest.substr(0, dot)); } catch (...) { return false; }
  layer = l;
  std::string mid = rest.substr(dot + 1);  // "{module_path}.lora_{A,B}.weight"
  const std::string la = ".lora_A.weight", lb = ".lora_B.weight";
  if (mid.size() > la.size() && mid.compare(mid.size() - la.size(), la.size(), la) == 0) {
    is_a = true;
    module = mid.substr(0, mid.size() - la.size());
  } else if (mid.size() > lb.size() && mid.compare(mid.size() - lb.size(), lb.size(), lb) == 0) {
    is_a = false;
    module = mid.substr(0, mid.size() - lb.size());
  } else {
    return false;
  }
  const auto d2 = module.rfind('.');
  if (d2 != std::string::npos) module = module.substr(d2 + 1);  // "mlp.gate_proj" -> "gate_proj"
  return true;
}

// A module's packed A/B sizes and the LoraModule's k/n, for a card count.
void module_sizes(const ModInfo &mi, int C, int r16, int &k, int &n, size_t &a_bytes, size_t &b_bytes) {
  if (mi.col) {
    k = mi.full_k;
    n = mi.full_n / C;  // B row-split over the card's output slice
  } else {
    k = mi.full_k / C;  // A col-split over the card's input slice
    n = mi.full_n;
  }
  a_bytes = static_cast<size_t>(r16) * k * 2;
  b_bytes = static_cast<size_t>(n) * r16 * 2;
}

// The LoraModule of a layer's bundle for a module.
kernels::LoraModule *layer_module(kernels::LoraLayer &ll, Mod m) {
  switch (m) {
    case Mod::Q: return &ll.q;
    case Mod::K: return &ll.k;
    case Mod::V: return &ll.v;
    case Mod::O: return &ll.o;
    case Mod::Gate: return &ll.gate;
    case Mod::Up: return &ll.up;
    case Mod::Down: return &ll.down;
  }
  return nullptr;
}

}  // namespace

LoraAdapter::LoraAdapter(Fabric &fab, const std::string &adapter_dir, const LoraGeom &geom, int threads)
    : fab_(&fab), geom_(geom), C_(geom.cards) {
  (void)threads;  // the adapter is small (a few hundred small tensors); no threads needed
  try {
    // 1. The config: r, lora_alpha (the scale), and the adapter's base-model name.
    std::ifstream f(adapter_dir + "/adapter_config.json");
    if (!f) throw std::runtime_error("lora: cannot open " + adapter_dir + "/adapter_config.json");
    std::ostringstream ss;
    ss << f.rdbuf();
    json::Value cfg;
    std::string err;
    if (!json::parse(ss.str(), cfg, &err)) throw std::runtime_error("lora: adapter_config.json: " + err);
    if (const std::string *pt = cfg.get_string("peft_type"); !pt || *pt != "LORA") throw std::runtime_error("lora: peft_type must be LORA");
    const json::Value *rv = cfg.get("r");
    if (!rv || !rv->is_number() || rv->num < 1) throw std::runtime_error("lora: adapter_config.json has no positive integer r");
    const int r = static_cast<int>(rv->num);
    const json::Value *av = cfg.get("lora_alpha");
    if (!av || !av->is_number() || av->num <= 0) throw std::runtime_error("lora: adapter_config.json has no positive lora_alpha");
    if (const json::Value *d = cfg.get("use_dora"); d && d->is_bool() && d->b) throw std::runtime_error("lora: use_dora is not supported");
    if (const json::Value *rs = cfg.get("use_rslora"); rs && rs->is_bool() && rs->b) throw std::runtime_error("lora: use_rslora is not supported");
    if (const std::string *bn = cfg.get_string("base_model_name_or_path")) name_ = *bn;
    scale_ = static_cast<float>(av->num) / static_cast<float>(r);
    r16_ = (r + 15) / 16 * 16;

    // 2. The adapter's tensors (the safetensors reader; no index cross-check, an adapter has none).
    const model::Checkpoint ck = model::Checkpoint::open(adapter_dir, false);

    // 3. Parse and validate every tensor -> (layer, module) -> {A, B, info}.
    struct Entry {
      const model::TensorInfo *a = nullptr;
      const model::TensorInfo *b = nullptr;
      ModInfo info{};
    };
    std::unordered_map<int, std::unordered_map<std::string, Entry>> entries;
    for (const model::TensorInfo &ti : ck.tensors()) {
      int layer;
      std::string module;
      bool is_a;
      if (!parse_name(ti.name, geom_, layer, module, is_a)) continue;  // a side table, not a LoRA tensor
      if (layer < 0 || layer >= geom_.layers)
        throw std::runtime_error("lora: " + ti.name + " names layer " + std::to_string(layer) + " past the model's " + std::to_string(geom_.layers));
      ModInfo mi;
      if (!mod_info(module, geom_, mi)) throw std::runtime_error("lora: unsupported module " + module + " (" + ti.name + ")");
      if ((mi.mod == Mod::Q || mi.mod == Mod::K || mi.mod == Mod::V || mi.mod == Mod::O) && !geom_.full_attention[layer])
        throw std::runtime_error("lora: " + module + " on layer " + std::to_string(layer) + " is not a full-attention (GQA) layer");
      Entry &e = entries[layer][module];
      if (is_a) {
        if (ti.dim(0) != r || ti.dim(1) != mi.full_k)
          throw std::runtime_error("lora: " + ti.name + " is [" + std::to_string(ti.dim(0)) + ", " + std::to_string(ti.dim(1)) + "], expected [r=" + std::to_string(r) + ", k=" + std::to_string(mi.full_k) + "]");
        e.a = &ti;
      } else {
        if (ti.dim(0) != mi.full_n || ti.dim(1) != r)
          throw std::runtime_error("lora: " + ti.name + " is [" + std::to_string(ti.dim(0)) + ", " + std::to_string(ti.dim(1)) + "], expected [n=" + std::to_string(mi.full_n) + ", r=" + std::to_string(r) + "]");
        e.b = &ti;
      }
      e.info = mi;
    }
    if (entries.empty()) throw std::runtime_error("lora: no LoRA tensors found in " + adapter_dir);
    for (auto &[l, mods] : entries)
      for (auto &[mn, e] : mods)
        if (!e.a || !e.b) throw std::runtime_error("lora: layer " + std::to_string(l) + " " + mn + " has " + (e.a ? "an A but not the B" : "a B but not the A"));

    // 4. The per-card buffer layout: one buffer per card, every module's packed A/B in it (the layout is the
    //    same on every card: the replicated A/B are whole, the split ones are the card's slice).
    const Mod order[] = {Mod::Q, Mod::K, Mod::V, Mod::O, Mod::Gate, Mod::Up, Mod::Down};
    struct Slot {
      int layer;
      Mod mod;
      const model::TensorInfo *a, *b;
      ModInfo info;
      size_t a_off, b_off;
      int k, n;
    };
    std::vector<Slot> slots;
    size_t off = 0;
    n_max_ = 0;
    for (int l = 0; l < geom_.layers; ++l) {
      auto lit = entries.find(l);
      if (lit == entries.end()) continue;
      for (Mod m : order) {
        auto mit = lit->second.find(mod_name(m));
        if (mit == lit->second.end()) continue;
        const Entry &e = mit->second;
        if (e.info.col && e.info.full_n % C_ != 0)
          throw std::runtime_error(std::string("lora: ") + mod_name(m) + " output " + std::to_string(e.info.full_n) + " does not split over " + std::to_string(C_) + " cards");
        if (!e.info.col && e.info.full_k % C_ != 0)
          throw std::runtime_error(std::string("lora: ") + mod_name(m) + " input " + std::to_string(e.info.full_k) + " does not split over " + std::to_string(C_) + " cards");
        Slot s;
        s.layer = l;
        s.mod = m;
        s.a = e.a;
        s.b = e.b;
        s.info = e.info;
        size_t a_bytes, b_bytes;
        module_sizes(e.info, C_, r16_, s.k, s.n, a_bytes, b_bytes);
        if (s.k % 16 != 0 || s.n % 16 != 0)
          throw std::runtime_error(std::string("lora: ") + mod_name(m) + " on " + std::to_string(C_) + " cards does not pack (a slice is not a multiple of 16)");
        s.a_off = off;
        off += a_bytes;
        s.b_off = off;
        off += b_bytes;
        n_max_ = std::max(n_max_, s.n);
        slots.push_back(s);
      }
    }
    const size_t total = off;
    if (total == 0) throw std::runtime_error("lora: no LoRA modules to load");

    // 5. Allocate one device buffer per card; fill it through a host staging buffer. Read each module's
    //    A/B once (F32 -> bf16, padded to r16), then shard and pack every card's slice into the staging
    //    buffer and ship it to the card.
    layers_.assign(static_cast<size_t>(C_) * geom_.layers, kernels::LoraLayer{});
    std::vector<uint8_t *> bases(C_);
    for (int c = 0; c < C_; ++c) {
      bases[c] = static_cast<uint8_t *>(fab_->alloc(static_cast<size_t>(c), total));
      owned_.push_back(bases[c]);
    }
    struct SlotData {
      std::vector<uint16_t> a_pad, b_pad;  // the padded A [r16][full_k] and B [full_n][r16]
      int full_k, full_n;
    };
    std::vector<SlotData> data(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
      const Slot &s = slots[i];
      std::vector<uint16_t> a_bits, b_bits;
      read_f32_bf16(ck, *s.a, a_bits);  // [r][full_k]
      read_f32_bf16(ck, *s.b, b_bits);  // [full_n][r]
      data[i].full_k = s.info.full_k;
      data[i].full_n = s.info.full_n;
      pad2(a_bits, r, s.info.full_k, r16_, s.info.full_k, data[i].a_pad);
      pad2(b_bits, s.info.full_n, r, s.info.full_n, r16_, data[i].b_pad);
    }
    std::vector<uint8_t> host(total);  // the staging buffer (the pack writes every byte, no clear needed)
    for (int c = 0; c < C_; ++c) {
      for (size_t i = 0; i < slots.size(); ++i) {
        const Slot &s = slots[i];
        const SlotData &d = data[i];
        std::vector<uint16_t> a_host, b_host;
        if (s.info.col) {
          a_host = d.a_pad;  // A replicated [r16][full_k]
          rows_slice(d.b_pad, r16_, c * (d.full_n / C_), d.full_n / C_, b_host);  // B row-split [full_n/C][r16]
        } else {
          cols_slice(d.a_pad, r16_, d.full_k, c * (d.full_k / C_), d.full_k / C_, a_host);  // A col-split [r16][full_k/C]
          b_host = d.b_pad;  // B replicated [full_n][r16]
        }
        kernels::pack_dense_bf16(a_host.data(), r16_, s.k, reinterpret_cast<uint16_t *>(host.data() + s.a_off));
        kernels::pack_dense_bf16(b_host.data(), s.n, r16_, reinterpret_cast<uint16_t *>(host.data() + s.b_off));
      }
      fab_->copy(static_cast<size_t>(c)).copy(bases[c], host.data(), total, true);
      for (size_t i = 0; i < slots.size(); ++i) {
        const Slot &s = slots[i];
        kernels::LoraModule &lm = *layer_module(layers_[static_cast<size_t>(c) * geom_.layers + s.layer], s.mod);
        lm.a = reinterpret_cast<const uint16_t *>(bases[c] + s.a_off);
        lm.b = reinterpret_cast<const uint16_t *>(bases[c] + s.b_off);
        lm.r16 = r16_;
        lm.k = s.k;
        lm.n = s.n;
        lm.out_off = s.info.col ? s.info.out_off : 0;
        lm.scale = scale_;
      }
    }
    for (int c = 0; c < C_; ++c) fab_->copy(static_cast<size_t>(c)).wait();
  } catch (...) {
    for (auto *p : owned_) fab_->release(p);
    owned_.clear();
    throw;
  }
}

LoraAdapter::~LoraAdapter() {
  for (auto *p : owned_) fab_->release(p);
}

void LoraAdapter::reserve_scratch(Fabric &fab, int m_max) {
  if (m_max < 1) throw std::invalid_argument("lora: m_max >= 1");
  sc_.assign(C_, kernels::LoraScratch{});
  const int rows = std::max(kernels::row_shape(m_max), m_max);  // decode runs at its row shape (<= 64); a prefill chunk runs at m rows
  for (int c = 0; c < C_; ++c) {
    auto &sc = sc_[static_cast<size_t>(c)];
    sc.r16 = r16_;
    sc.n_max = n_max_;
    sc.mid_f32 = static_cast<float *>(fab.alloc(static_cast<size_t>(c), static_cast<size_t>(rows) * r16_ * 4));
    sc.mid_b16 = static_cast<kernels::bf16 *>(fab.alloc(static_cast<size_t>(c), static_cast<size_t>(rows) * r16_ * 2));
    sc.delta = static_cast<float *>(fab.alloc(static_cast<size_t>(c), static_cast<size_t>(rows) * n_max_ * 4));
    owned_.push_back(sc.mid_f32);
    owned_.push_back(sc.mid_b16);
    owned_.push_back(sc.delta);
  }
}

}  // namespace xenithium::load
