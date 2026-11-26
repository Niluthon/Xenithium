// LoRA adapter gate (GPU): the PEFT loader (adapter_config parse, name -> placement mapping, per-card
// column / row sharding, F32 -> bf16, pack, upload) and the delta kernels (lora_apply) against a host
// reference. One card runs the kernel reference and the zero-adapter invariance; two cards also exercise
// the column / row sharding. The reference matches the kernel's rounding points (the mid is rounded to
// bf16 once, as the engine's activations), so the bound is a small relative tolerance: the DPAS
// accumulation order differs from the host's sequential sum (two GEMMs, mid then delta).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <unistd.h>

#include "xenithium/kernels/lora.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/load/lora.hpp"
#include "xenithium/rt/fabric.hpp"
#include "xenithium/rt/l0.hpp"

namespace fs = std::filesystem;
namespace k = xenithium::kernels;
namespace load = xenithium::load;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace {

int g_fail = 0, g_pass = 0;
void expect(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  ok ? ++g_pass : ++g_fail;
}
bool throws(const std::function<void()> &f, std::string *msg = nullptr) {
  try {
    f();
  } catch (const std::exception &e) {
    if (msg) *msg = e.what();
    return true;
  }
  return false;
}

// A safetensors file: header (padded with `pad` spaces) + data.
void write_shard(const fs::path &p, const std::string &header, const std::string &data, int pad = 0) {
  std::string h = header + std::string(pad, ' ');
  std::ofstream o(p, std::ios::binary);
  uint64_t n = h.size();
  for (int i = 0; i < 8; ++i) o.put(static_cast<char>((n >> (8 * i)) & 0xff));
  o << h << data;
}

// A deterministic value in [-0.5, 0.5) (the adapter's F32, converted to bf16 at load).
float val(uint32_t seed, size_t i) {
  uint32_t x = seed * 2654435761u + static_cast<uint32_t>(i) * 40503u;
  x ^= x >> 15;
  x *= 2246822519u;
  x ^= x >> 13;
  return static_cast<float>(x & 0xFFFF) / 65535.f - 0.5f;
}

// The test geometry (small, packs on 1 and 2 cards): 2 layers (0 = GQA, 1 = GDN), hidden 64, q / k / v
// sections 32 each (no output gate), inter 64, r 16 (r16 16), alpha 32 (scale 2).
struct Geom {
  int layers = 2, hidden = 64, q_heads = 2, kv_heads = 2, head_dim = 16, inter = 64;
  bool gate = false;
  int q_n() const { return q_heads * (gate ? 2 : 1) * head_dim; }  // 32
  int k_n() const { return kv_heads * head_dim; }                  // 32
  int q_hd() const { return q_heads * head_dim; }                  // 32
};

struct Mod {
  const char *name;
  bool col;
  int full_n, full_k;
  int mpath;  // 0 self_attn (q/k/v/o), 1 mlp (gate/up/down)
};
std::vector<Mod> all_mods(const Geom &g) {
  return {{"q_proj", true, g.q_n(), g.hidden, 0}, {"k_proj", true, g.k_n(), g.hidden, 0}, {"v_proj", true, g.k_n(), g.hidden, 0},
          {"o_proj", false, g.hidden, g.q_hd(), 0}, {"gate_proj", true, g.inter, g.hidden, 1},
          {"up_proj", true, g.inter, g.hidden, 1}, {"down_proj", false, g.hidden, g.inter, 1}};
}
bool is_gqa_mod(const Mod &m) { return m.mpath == 0; }
uint32_t mod_seed(const Mod &m) {
  uint32_t h = 2166136261u;
  for (const char *p = m.name; *p; ++p) h = (h ^ static_cast<uint8_t>(*p)) * 16777619u;
  return h;
}
// The module's B rows' offset within the per-card target buffer (column-parallel), and the target's row stride.
int out_off(const Mod &m, const Geom &g, int C) {
  if (!m.col) return 0;
  if (m.name[0] == 'k') return g.q_n() / C;
  if (m.name[0] == 'v') return (g.q_n() + g.k_n()) / C;
  if (m.name[0] == 'u') return g.inter / C;
  return 0;  // q, gate
}
int out_ld(const Mod &m, const Geom &g, int C) {
  if (m.mpath == 0 && m.name[0] != 'o') return (g.q_n() + 2 * g.k_n()) / C;  // the qkv row
  if (m.mpath == 1 && m.col) return 2 * g.inter / C;                          // the gu row
  return g.hidden;                                                            // o, down
}

// The adapter's F32 -> bf16 (the loader's f32_to_bf16, round to nearest even).
uint16_t abits(uint32_t seed, size_t i, bool zero) { return zero ? 0u : k::bf16_bits(val(seed, i)); }

// The full (padded to r16) A [r16][full_k] and B [full_n][r16] of a module, as the loader holds them.
std::vector<uint16_t> full_A(const Mod &m, int l, int r16, bool zero) {
  std::vector<uint16_t> a(static_cast<size_t>(r16) * m.full_k);
  for (size_t i = 0; i < a.size(); ++i) a[i] = abits(mod_seed(m) + static_cast<uint32_t>(l) * 97u, i, zero);
  return a;
}
std::vector<uint16_t> full_B(const Mod &m, int l, int r16, bool zero) {
  std::vector<uint16_t> b(static_cast<size_t>(m.full_n) * r16);
  for (size_t i = 0; i < b.size(); ++i) b[i] = abits(mod_seed(m) + static_cast<uint32_t>(l) * 97u + 7u, i, zero);
  return b;
}

// Write adapter_config.json + adapter_model.safetensors. gqa_layers: the layers that carry q/k/v/o.
void write_adapter(const fs::path &dir, const Geom &g, int r, int alpha, const std::vector<int> &gqa_layers, bool zero) {
  fs::remove_all(dir);
  fs::create_directories(dir);
  std::ofstream(dir / "adapter_config.json")
      << R"({"peft_type":"LORA","r":)" << r << R"(,"lora_alpha":)" << alpha << R"(,"base_model_name_or_path":"test-base","use_dora":false,"use_rslora":false})";
  std::string hdr = "{";
  std::string data;
  size_t off = 0;
  bool first = true;
  auto add = [&](const std::string &name, const std::vector<int64_t> &shape, const std::vector<float> &f32) {
    const size_t bytes = f32.size() * 4;
    if (!first) hdr += ",";
    first = false;
    hdr += "\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[";
    for (size_t i = 0; i < shape.size(); ++i) hdr += (i ? "," : "") + std::to_string(shape[i]);
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + bytes) + "]}";
    for (float v : f32) {
      char b[4];
      std::memcpy(b, &v, 4);
      data.append(b, 4);
    }
    off += bytes;
  };
  for (int l = 0; l < g.layers; ++l) {
    const bool gqa = std::find(gqa_layers.begin(), gqa_layers.end(), l) != gqa_layers.end();
    for (const Mod &m : all_mods(g)) {
      if (is_gqa_mod(m) && !gqa) continue;  // GDN layers carry only the MLP modules
      const std::string mpath = std::string(m.mpath == 0 ? "self_attn." : "mlp.") + m.name;
      const std::string base = "base_model.model.model.language_model.layers." + std::to_string(l) + "." + mpath;
      std::vector<float> a(static_cast<size_t>(r) * m.full_k), b(static_cast<size_t>(m.full_n) * r);
      for (size_t i = 0; i < a.size(); ++i) a[i] = zero ? 0.f : val(mod_seed(m) + static_cast<uint32_t>(l) * 97u, i);
      for (size_t i = 0; i < b.size(); ++i) b[i] = zero ? 0.f : val(mod_seed(m) + static_cast<uint32_t>(l) * 97u + 7u, i);
      add(base + ".lora_A.weight", {r, m.full_k}, a);
      add(base + ".lora_B.weight", {m.full_n, r}, b);
    }
  }
  hdr += "}";
  write_shard(dir / "adapter_model.safetensors", hdr, data, 3);
}

// A card's A / B slice of the full (padded) matrices, and the card's input slice.
std::vector<uint16_t> card_a(const std::vector<uint16_t> &A, int r16, int full_k, int c, int C, bool col, int &k) {
  k = col ? full_k : full_k / C;
  if (col) return A;  // replicated
  const int c0 = c * (full_k / C);
  std::vector<uint16_t> o(static_cast<size_t>(r16) * k);
  for (int j = 0; j < r16; ++j)
    for (int kk = 0; kk < k; ++kk) o[static_cast<size_t>(j) * k + kk] = A[static_cast<size_t>(j) * full_k + c0 + kk];
  return o;
}
std::vector<uint16_t> card_b(const std::vector<uint16_t> &B, int r16, int full_n, int c, int C, bool col, int &n) {
  n = col ? full_n / C : full_n;
  if (col) {
    const int r0 = c * (full_n / C);
    std::vector<uint16_t> o(static_cast<size_t>(n) * r16);
    for (int cc = 0; cc < n; ++cc)
      for (int j = 0; j < r16; ++j) o[static_cast<size_t>(cc) * r16 + j] = B[static_cast<size_t>(r0 + cc) * r16 + j];
    return o;
  }
  return B;  // replicated
}
std::vector<uint16_t> card_x(const std::vector<uint16_t> &x, int m, int full_k, int c, int C, bool col, int &k) {
  k = col ? full_k : full_k / C;
  const int c0 = col ? 0 : c * (full_k / C);
  std::vector<uint16_t> o(static_cast<size_t>(m) * k);
  for (int r = 0; r < m; ++r)
    for (int kk = 0; kk < k; ++kk) o[static_cast<size_t>(r) * k + kk] = x[static_cast<size_t>(r) * 64 + c0 + kk];
  return o;
}

// The host reference: out[r][out_off + c] += scale * (B . bf16(A . x)), the mid rounded to bf16 once.
void lora_ref(const std::vector<uint16_t> &A, int r16, int k, const std::vector<uint16_t> &B, int n, const std::vector<uint16_t> &x, int m,
              float scale, int out_off, int out_ld, std::vector<float> &out) {
  std::vector<float> mid_b(static_cast<size_t>(m) * r16);
  for (int r = 0; r < m; ++r)
    for (int j = 0; j < r16; ++j) {
      float mid = 0;
      for (int kk = 0; kk < k; ++kk) mid += k::bf16_value(A[static_cast<size_t>(j) * k + kk]) * k::bf16_value(x[static_cast<size_t>(r) * k + kk]);
      mid_b[static_cast<size_t>(r) * r16 + j] = k::bf16_value(k::bf16_bits(mid));
    }
  for (int r = 0; r < m; ++r)
    for (int c2 = 0; c2 < n; ++c2) {
      float d = 0;
      for (int j = 0; j < r16; ++j) d += mid_b[static_cast<size_t>(r) * r16 + j] * k::bf16_value(B[static_cast<size_t>(c2) * r16 + j]);
      out[static_cast<size_t>(r) * out_ld + out_off + c2] += scale * d;
    }
}

// A fp32 comparison with the "same rounding points" bound: the worst relative deviation (of max(|ref|, the
// row's rms)) is small and at most 1 % of the elements are off an absolute + relative floor.
bool fp32_cmp(const std::vector<float> &got, const std::vector<float> &ref, size_t row, double &worst, size_t &off) {
  worst = 0;
  off = 0;
  if (!row) row = got.size();
  for (size_t r0 = 0; r0 < got.size(); r0 += row) {
    double ss = 0;
    for (size_t i = r0; i < r0 + row && i < got.size(); ++i) ss += static_cast<double>(ref[i]) * ref[i];
    const double rms = std::sqrt(ss / static_cast<double>(row));
    for (size_t i = r0; i < r0 + row && i < got.size(); ++i) {
      const double d = std::fabs(got[i] - ref[i]);
      worst = std::max(worst, d / std::max<double>({static_cast<double>(std::fabs(ref[i])), rms, 1e-30}));
      off += d > 2e-3 + 4e-3 * std::max<double>(static_cast<double>(std::fabs(ref[i])), rms);
    }
  }
  return worst <= 0.02 && off * 100 <= got.size();
}

fs::path tmpdir() {
  const char *t = std::getenv("TMPDIR");
  return fs::path(t ? t : "/tmp") / ("xen-test-lora-" + std::to_string(::getpid()));
}

load::LoraGeom geom_of(const Geom &g, int C, const std::vector<int> &gqa_layers) {
  load::LoraGeom lg;
  lg.cards = C;
  lg.layers = g.layers;
  lg.hidden = g.hidden;
  lg.q_heads = g.q_heads;
  lg.kv_heads = g.kv_heads;
  lg.head_dim = g.head_dim;
  lg.attn_output_gate = g.gate;
  lg.inter = g.inter;
  lg.full_attention.assign(g.layers, false);
  for (int l : gqa_layers) lg.full_attention[l] = true;
  lg.prefix = "model.language_model.";
  return lg;
}

}  // namespace

int main() {
  try {
    const auto loader = xenithium::l0::Loader::open();
    const int C = std::min(2, static_cast<int>(loader.size()));
    std::vector<size_t> ranks(C);
    for (int c = 0; c < C; ++c) ranks[c] = c;
    xenithium::Fabric fab(loader, ranks, false);
    const Geom g;
    const std::vector<int> gqa_layers = {0};
    const load::LoraGeom lg = geom_of(g, C, gqa_layers);
    const fs::path base = tmpdir();
    int dir = 0;

    // ---- adapter_config rejects (throw before any GPU work) --------------------------------------------
    {
      struct Bad {
        const char *what, *cfg;
      };
      for (const Bad &b : {Bad{"no peft_type", R"({"r":16,"lora_alpha":32})"},
                           Bad{"peft_type not LORA", R"({"peft_type":"ADALORA","r":16,"lora_alpha":32})"},
                           Bad{"no r", R"({"peft_type":"LORA","lora_alpha":32})"},
                           Bad{"r = 0", R"({"peft_type":"LORA","r":0,"lora_alpha":32})"},
                           Bad{"no lora_alpha", R"({"peft_type":"LORA","r":16})"},
                           Bad{"use_dora", R"({"peft_type":"LORA","r":16,"lora_alpha":32,"use_dora":true})"},
                           Bad{"use_rslora", R"({"peft_type":"LORA","r":16,"lora_alpha":32,"use_rslora":true})"}}) {
        const fs::path d = base / (std::to_string(dir++));
        fs::create_directories(d);
        std::ofstream(d / "adapter_config.json") << b.cfg;
        std::string msg;
        const bool t = throws([&] { load::LoraAdapter ad(fab, d.string(), lg, 1); }, &msg);
        expect(t, std::string("config: ") + b.what + " throws (" + msg + ")");
      }
    }

    // ---- name -> placement mapping and shape rejects (throw before the data is read) --------------------
    {
      const std::string cfg = R"({"peft_type":"LORA","r":16,"lora_alpha":32,"base_model_name_or_path":"t"})";
      const std::string P = "base_model.model.model.language_model.layers.";
      auto write_one = [&](const fs::path &d, const std::vector<std::tuple<std::string, std::vector<int64_t>>> &tensors) {
        fs::remove_all(d);
        fs::create_directories(d);
        std::ofstream(d / "adapter_config.json") << cfg;
        std::string hdr = "{";
        std::string data;
        size_t off = 0;
        bool first = true;
        for (const auto &[name, shape] : tensors) {
          size_t numel = 1;
          for (auto s : shape) numel *= static_cast<size_t>(s);
          const size_t bytes = numel * 4;
          if (!first) hdr += ",";
          first = false;
          hdr += "\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[";
          for (size_t i = 0; i < shape.size(); ++i) hdr += (i ? "," : "") + std::to_string(shape[i]);
          hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + bytes) + "]}";
          data.append(bytes, '\0');
          off += bytes;
        }
        hdr += "}";
        write_shard(d / "adapter_model.safetensors", hdr, data, 3);
      };
      struct Bad {
        const char *what;
        std::vector<std::tuple<std::string, std::vector<int64_t>>> t;
      };
      for (const Bad &b : {Bad{"A shape [r+1][k]", {{P + "0.self_attn.q_proj.lora_A.weight", {17, 64}}, {P + "0.self_attn.q_proj.lora_B.weight", {32, 16}}}},
                           Bad{"B shape [n][r+1]", {{P + "0.self_attn.q_proj.lora_A.weight", {16, 64}}, {P + "0.self_attn.q_proj.lora_B.weight", {32, 17}}}},
                           Bad{"unsupported module", {{P + "0.self_attn.router.lora_A.weight", {16, 64}}, {P + "0.self_attn.router.lora_B.weight", {32, 16}}}},
                           Bad{"layer past the model", {{P + "99.self_attn.q_proj.lora_A.weight", {16, 64}}, {P + "99.self_attn.q_proj.lora_B.weight", {32, 16}}}},
                           Bad{"q_proj on a GDN layer", {{P + "1.self_attn.q_proj.lora_A.weight", {16, 64}}, {P + "1.self_attn.q_proj.lora_B.weight", {32, 16}}}},
                           Bad{"A without B", {{P + "0.self_attn.q_proj.lora_A.weight", {16, 64}}}}}) {
        const fs::path d = base / (std::to_string(dir++));
        write_one(d, b.t);
        std::string msg;
        const bool t = throws([&] { load::LoraAdapter ad(fab, d.string(), lg, 1); }, &msg);
        expect(t, std::string("adapter: ") + b.what + " throws (" + msg + ")");
      }
    }

    // ---- the integration: load the adapter, lora_apply vs the host reference (decode + prefill) ---------
    {
      const int r = 16, alpha = 32, r16 = 16;
      const fs::path d = base / (std::to_string(dir++));
      write_adapter(d, g, r, alpha, gqa_layers, false);
      load::LoraAdapter ad(fab, d.string(), lg, 1);
      expect(ad.cards() == C && ad.layers() == g.layers && ad.r16() == r16, "adapter: cards, layers, r16");
      expect(ad.scale() == 2.0f, "adapter: scale = lora_alpha / r = 2.0");
      ad.reserve_scratch(fab, 128);  // m up to 128 (decode 4, prefill 100)
      for (int c = 0; c < C; ++c) {  // the loaded layer bundles: layer 0 (GQA) carries every module, layer 1 (GDN) only the MLP
        const k::LoraLayer &gqa = ad.layer(c, 0);
        expect(gqa.q.present() && gqa.k.present() && gqa.v.present() && gqa.o.present() && gqa.gate.present() && gqa.up.present() && gqa.down.present(),
               "adapter: the GQA layer carries every module (q/k/v/o/gate/up/down)");
        const k::LoraLayer &gdn = ad.layer(c, 1);
        expect(!gdn.q.present() && !gdn.k.present() && !gdn.v.present() && !gdn.o.present(), "adapter: the GDN layer carries no GQA module");
        expect(gdn.gate.present() && gdn.up.present() && gdn.down.present(), "adapter: the GDN layer carries the MLP modules");
      }
      for (const int m : {4, 100}) {
        std::vector<uint16_t> x(static_cast<size_t>(m) * 64);
        for (size_t i = 0; i < x.size(); ++i) x[i] = k::bf16_bits(val(0xABCDu, i));
        for (int c = 0; c < C; ++c) {
          for (int l = 0; l < g.layers; ++l) {
            const k::LoraLayer &ll = ad.layer(c, l);
            for (const Mod &md : all_mods(g)) {
              if (is_gqa_mod(md) && l != 0) continue;  // GDN layers carry no GQA modules
              const k::LoraModule *lm =
                  md.name[0] == 'q' ? &ll.q : md.name[0] == 'k' ? &ll.k : md.name[0] == 'v' ? &ll.v : md.name[0] == 'o' ? &ll.o
                                                                                               : md.name[0] == 'g' ? &ll.gate : md.name[0] == 'u' ? &ll.up : &ll.down;
              if (!lm->present()) continue;
              int k_, n_;
              const std::vector<uint16_t> A = card_a(full_A(md, l, r16, false), r16, md.full_k, c, C, md.col, k_);
              const std::vector<uint16_t> B = card_b(full_B(md, l, r16, false), r16, md.full_n, c, C, md.col, n_);
              const std::vector<uint16_t> xc = card_x(x, m, md.full_k, c, C, md.col, k_);
              const int ooff = out_off(md, g, C), old = out_ld(md, g, C);
              std::vector<float> ref(static_cast<size_t>(m) * old, 0.f);
              lora_ref(A, r16, k_, B, n_, xc, m, 2.0f, ooff, old, ref);
              auto *d_x = static_cast<bf16 *>(fab.alloc(c, xc.size() * 2));
              auto *d_out = static_cast<float *>(fab.alloc(c, ref.size() * 4));
              std::vector<float> zero(ref.size(), 0.f);
              fab.copy(c).copy(d_x, xc.data(), xc.size() * 2, true);
              fab.copy(c).copy(d_out, zero.data(), zero.size() * 4, true);
              k::lora_apply(fab.compute(c), m, d_x, *lm, d_out, old, ad.scratch(c), 0);
              fab.compute(c).wait_and_throw();
              std::vector<float> got(ref.size());
              fab.copy(c).copy(got.data(), d_out, got.size() * 4, true);
              fab.release(d_x);
              fab.release(d_out);
              double worst;
              size_t off;
              const bool ok = fp32_cmp(got, ref, old, worst, off);
              char what[160];
              std::snprintf(what, sizeof what, "lora %s layer %d card %d m=%d: == host reference (worst %.1e, %zu off)", md.name, l, c, m, worst, off);
              expect(ok, what);
            }
          }
        }
      }
    }

    // ---- zero-adapter invariance: the delta is exactly 0, the output bit-identical ----------------------
    {
      const int r = 16, alpha = 32;
      const fs::path d = base / (std::to_string(dir++));
      write_adapter(d, g, r, alpha, gqa_layers, true);  // all-zero A / B
      load::LoraAdapter ad(fab, d.string(), lg, 1);
      ad.reserve_scratch(fab, 16);
      const int m = 4, c = 0, l = 0;
      const k::LoraModule &lm = ad.layer(c, l).q;
      std::vector<uint16_t> x(static_cast<size_t>(m) * lm.k);
      for (size_t i = 0; i < x.size(); ++i) x[i] = k::bf16_bits(val(0x1234u, i));
      const int old = out_ld(all_mods(g)[0], g, C);  // q
      std::vector<float> init(static_cast<size_t>(m) * old);
      for (size_t i = 0; i < init.size(); ++i) init[i] = val(0x5678u, i);
      auto *d_x = static_cast<bf16 *>(fab.alloc(c, x.size() * 2));
      auto *d_out = static_cast<float *>(fab.alloc(c, init.size() * 4));
      fab.copy(c).copy(d_x, x.data(), x.size() * 2, true);
      fab.copy(c).copy(d_out, init.data(), init.size() * 4, true);
      k::lora_apply(fab.compute(c), m, d_x, lm, d_out, old, ad.scratch(c), 0);
      fab.compute(c).wait_and_throw();
      std::vector<float> got(init.size());
      fab.copy(c).copy(got.data(), d_out, got.size() * 4, true);
      fab.release(d_x);
      fab.release(d_out);
      expect(got == init, "zero adapter: lora_apply leaves the output bit-identical (the delta is exactly 0)");
    }

    fs::remove_all(base);
  } catch (const std::exception &e) {
    std::printf("[FAIL] exception: %s\n", e.what());
    return 1;
  }
  std::printf("%s: %d passed, %d failed\n", g_fail ? "FAILED" : "OK", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
