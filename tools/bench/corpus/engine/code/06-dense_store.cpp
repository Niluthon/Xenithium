#include "xenithium/load/dense_store.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <future>
#include <stdexcept>
#include <thread>

#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"

namespace xenithium::load {

using model::DenseSpec;

namespace {

using clk = std::chrono::steady_clock;
double secs(clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); }
constexpr size_t kAlign = 4096;
size_t align_up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

// f16 (1-5-10) -> f32, exact (subnormals m x 2^-14 normalised through the exponent).
float f16_to_float(uint16_t x) {
  uint32_t f;
  const uint32_t e = (x >> 10) & 0x1Fu, m = x & 0x3FFu;
  if (e == 0) {
    if (m) {
      const int p = 31 - __builtin_clz(m);  // top bit position (0..9): value 2^(p - 24) x 1.f, biased exponent 103 + p
      f = ((x & 0x8000u) << 16) | ((103u + p) << 23) | ((m << (9 - p)) - 512u) << 14;
    } else
      f = (x & 0x8000u) << 16;
  } else if (e == 0x1Fu)
    f = ((x & 0x8000u) << 16) | 0xFFu << 23 | m << 13;
  else
    f = ((x & 0x8000u) << 16) | (e - 15 + 127) << 23 | m << 13;
  float r;
  std::memcpy(&r, &f, 4);
  return r;
}
// f32 -> bf16 (1-8-7), round-to-nearest-even.
uint16_t float_to_bf16(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return static_cast<uint16_t>((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
}
// f16 -> bf16 (exact through f32, then RNE).
uint16_t f16_to_bf16(uint16_t x) { return float_to_bf16(f16_to_float(x)); }
float bf16_to_float(uint16_t x) {
  const uint32_t u = static_cast<uint32_t>(x) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t float_to_f16(float f) {  // round to nearest even (the compiler's conversion); callers check exactness
  const _Float16 h = static_cast<_Float16>(f);
  uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}

void run_tasks(std::vector<std::function<void()>> &tasks, int threads) {
  std::atomic<size_t> next{0};
  std::vector<std::string> err(static_cast<size_t>(threads));
  std::vector<std::thread> th;
  for (int t = 0; t < threads; ++t)
    th.emplace_back([&, t] {
      try {
        for (size_t i; (i = next.fetch_add(1)) < tasks.size();) tasks[i]();
      } catch (const std::exception &e) {
        err[static_cast<size_t>(t)] = e.what();
        next = tasks.size();
      }
    });
  for (auto &x : th) x.join();
  for (const auto &e : err)
    if (!e.empty()) throw std::runtime_error(e);
}

std::vector<l0::Queue::CopyOp> chunked(void *dst, const void *src, size_t bytes) {
  constexpr size_t kChunk = 64ull << 20;
  std::vector<l0::Queue::CopyOp> ops;
  for (size_t off = 0; off < bytes; off += kChunk)
    ops.push_back({static_cast<uint8_t *>(dst) + off, static_cast<const uint8_t *>(src) + off, std::min(kChunk, bytes - off)});
  return ops;
}

// w4, two orientations (DenseSpec:W4Layout): ct = the module's weight_packed I32 [n][k/8] with weight_scale
// F16 [n][k/128] beside it; gptq = qweight I32 [k/8][n] with scales F16 [k/128][n] and qzeros I32 [k/128][n/8]
// (symmetric: every word 0x77777777, checked on the data).
using W4L = DenseSpec::W4Layout;
constexpr uint32_t kGptqSymZeros = 0x77777777u;
std::string w4_sibling(const std::string &packed_name, W4L layout, const char *which) {
  const std::string suffix = layout == W4L::gptq ? ".qweight" : layout == W4L::mlx ? ".weight" : ".weight_packed";
  if (packed_name.size() <= suffix.size() || packed_name.compare(packed_name.size() - suffix.size(), suffix.size(), suffix) != 0)
    throw std::runtime_error("dense store: w4 tensor " + packed_name + " is not a *" + suffix);
  return packed_name.substr(0, packed_name.size() - suffix.size()) + which;
}
std::string w4_scale_name(const std::string &packed_name, W4L layout) {
  return w4_sibling(packed_name, layout, layout == W4L::gptq || layout == W4L::mlx ? ".scales" : ".weight_scale");
}
// gptq: qzeros; ct_zp: weight_zero_point; affine: the group biases
std::string w4_zeros_name(const std::string &packed_name, W4L layout = W4L::gptq) {
  return w4_sibling(packed_name, layout, layout == W4L::gptq ? ".qzeros" : layout == W4L::mlx ? ".biases" : ".weight_zero_point");
}
// The logical (n, k) of a packed tensor in its layout.
int64_t w4_n(const model::TensorInfo &t, W4L layout) { return layout == W4L::gptq ? t.dim(1) : t.dim(0); }
int64_t w4_k(const model::TensorInfo &t, W4L layout) { return layout == W4L::mlx ? t.dim(1) * 2 : (layout == W4L::gptq ? t.dim(0) : t.dim(1)) * 8; }
void w4_check(const model::Checkpoint &ck, const std::string &name, const model::TensorInfo &t, int64_t n, int64_t k, W4L layout) {
  if (layout == W4L::mlx) {  // affine, group 64: codes U8 [n][k/2], scales and biases BF16 [n][k/64]
    const int64_t G = kernels::kW4AffGroup;
    if (t.dtype != model::DType::u8 || t.shape.size() != 2) throw std::runtime_error("dense store: " + name + " must be U8 2-D");
    if (k % kernels::kW4Group || n % 16) throw std::runtime_error("dense store: " + name + " k is not a multiple of 128 or n of 16");
    for (const std::string &sib : {w4_scale_name(name, layout), w4_zeros_name(name, layout)}) {
      const model::TensorInfo *s = ck.find(sib);
      if (!s || s->dtype != model::DType::bf16 || s->shape != std::vector<int64_t>{n, k / G})
        throw std::runtime_error("dense store: " + sib + " missing or not BF16 [n, k/64]");
    }
    return;
  }
  if (t.dtype != model::DType::i32 || t.shape.size() != 2) throw std::runtime_error("dense store: " + name + " must be I32 2-D");
  if (k % kernels::kW4Group) throw std::runtime_error("dense store: " + name + " k is not a multiple of 128");
  if (layout == W4L::ct_zp) {  // compressed-tensors asymmetric, group 32: scales bf16 / f16 [n][k/32], zero points I32 [n/8][k/32]
    const int64_t G = kernels::kW4ZpGroup;
    const model::TensorInfo *s = ck.find(w4_scale_name(name, layout)), *z = ck.find(w4_zeros_name(name, layout));
    if (!s || (s->dtype != model::DType::bf16 && s->dtype != model::DType::f16) || s->shape != std::vector<int64_t>{n, k / G})
      throw std::runtime_error("dense store: " + w4_scale_name(name, layout) + " missing or not BF16 / F16 [n, k/32]");
    if (n % 16 || !z || z->dtype != model::DType::i32 || z->shape != std::vector<int64_t>{n / 8, k / G})
      throw std::runtime_error("dense store: " + w4_zeros_name(name, layout) + " missing or not I32 [n/8, k/32] (n a multiple of 16)");
    return;
  }
  const model::TensorInfo *s = ck.find(w4_scale_name(name, layout));
  const std::vector<int64_t> want = layout == W4L::gptq ? std::vector<int64_t>{k / kernels::kW4Group, n} : std::vector<int64_t>{n, k / kernels::kW4Group};
  if (!s || s->dtype != model::DType::f16 || s->shape != want)
    throw std::runtime_error("dense store: " + w4_scale_name(name, layout) + " missing or not F16 " + (layout == W4L::gptq ? "[k/128, n]" : "[n, k/128]"));
  if (layout == W4L::gptq) {
    const model::TensorInfo *z = ck.find(w4_zeros_name(name));
    if (!z || z->dtype != model::DType::i32 || z->shape != std::vector<int64_t>{k / kernels::kW4Group, n / 8})
      throw std::runtime_error("dense store: " + w4_zeros_name(name) + " missing or not I32 [k/128, n/8]");
  }
}

// One tensor's slice on one card: where its rows come from and where it goes.
struct Item {
  size_t spec = 0;
  int card = 0;
  const model::TensorInfo *t = nullptr;
  std::vector<std::pair<int64_t, int64_t>> src_rows;  // source row ranges, in slice order
  std::vector<const model::TensorInfo *> src_t;  // the tensor of each range (appends)
  std::vector<const model::TensorInfo *> src_s;  // w4: the scale tensor of each range
  std::vector<const model::TensorInfo *> src_z;  // w4 gptq: the qzeros tensor of each range; ct_zp: weight_zero_point
  int64_t G = 128;  // w4: k per scale (ct_zp: 32)
  size_t zero_off = 0;  // w4 ct_zp: the zero-point nibbles [k/64][rows_pad] within the item; affine: the
                                                      // bf16 biases [k/64][rows_pad]
  int64_t col0 = 0;  // first source column (cols split; w4: in k elements)
  int64_t rows = 0, cols = 0, rows_pad = 0;  // slice; rows padded to 16 for matrices (w4: cols = k)
  size_t esize = 0;  // source element size
  size_t off = 0, bytes = 0, scale_off = 0;  // within the group's block
  size_t payload = 0;
};

}  // namespace

W4Weights DeviceTensor::w4() const {
  if (store != DenseSpec::Store::w4) throw std::runtime_error("w4(): not a w4 tensor");
  W4Weights w;
  w.w = ptr;
  w.scales = reinterpret_cast<const sycl::half *>(scale16);
  w.n = static_cast<int>((rows + 15) / 16 * 16);
  w.k = static_cast<int>(cols);
  w.zeros = zeros;
  w.biases = biases;
  w.group = group;
  return w;
}

kernels::DenseWeights DeviceTensor::dense() const {
  kernels::DenseWeights w;
  if (store == DenseSpec::Store::raw) throw std::runtime_error("dense(): raw tensor");
  if (store == DenseSpec::Store::w4) throw std::runtime_error("dense(): w4 tensor (use w4())");
  w.w = ptr;
  w.scale = scale;
  w.kind = store == DenseSpec::Store::i8 ? kernels::DenseKind::i8 : kernels::DenseKind::bf16;
  w.n = static_cast<int>((rows + 15) / 16 * 16);
  w.k = static_cast<int>(cols);
  return w;
}

std::vector<std::pair<int64_t, int64_t>> slice_rows(int64_t d0, const std::vector<int64_t> &sections, int card, int cards) {
  std::vector<int64_t> secs = sections.empty() ? std::vector<int64_t>{d0} : sections;
  int64_t sum = 0, off = 0;
  for (int64_t s : secs) sum += s;
  if (sum != d0) throw std::runtime_error("slice_rows: sections do not add up to the leading dimension");
  std::vector<std::pair<int64_t, int64_t>> out;
  for (int64_t s : secs) {
    if (s % cards) throw std::runtime_error("slice_rows: a section does not split over the cards");
    out.push_back({off + s / cards * card, off + s / cards * (card + 1)});
    off += s;
  }
  return out;
}

DenseStore::DenseStore(Fabric &fabric) : fab_(fabric), cards_(static_cast<int>(fabric.size())), t_(fabric.size()) {}

DenseStore::~DenseStore() {
  for (const uint8_t *w : zp_registered_) kernels::w4_clear_zero_points(w);
  for (const uint8_t *w : aff_registered_) kernels::w4_clear_affine(w);
  for (int c = 0; c < cards_; ++c) fab_.copy(c).wait();
  for (void *p : dev_blocks_) fab_.release(p);
  for (void *p : host_blocks_) fab_.release(p);
}

const DeviceTensor &DenseStore::at(int card, const std::string &name) const {
  const auto it = t_.at(static_cast<size_t>(card)).find(name);
  if (it == t_[static_cast<size_t>(card)].end()) throw std::runtime_error("dense store: no tensor " + name);
  return it->second;
}

void DenseStore::max_i8(int card, size_t *bytes, int64_t *rows, const std::vector<std::string> &skip) const {
  *bytes = 0, *rows = 0;
  for (const auto &kv : t_.at(static_cast<size_t>(card))) {
    const DeviceTensor &t = kv.second;
    if (t.store != model::DenseSpec::Store::i8) continue;
    if (std::any_of(skip.begin(), skip.end(), [&](const std::string &s) { return kv.first.find(s) != std::string::npos; })) continue;
    *bytes = std::max(*bytes, static_cast<size_t>(t.rows * t.cols));
    *rows = std::max(*rows, t.rows);
  }
}

void DenseStore::load(const model::Checkpoint &ck, const std::vector<DenseSpec> &plan, int threads) {
  const auto t_all = clk::now();
  stats_ = {};
  stats_.card_bytes.assign(static_cast<size_t>(cards_), {});
  stats_.card_alloc.assign(static_cast<size_t>(cards_), 0);
  const int C = cards_;

  // ---- items: every (spec, card) slice, grouped by layer; host tensors once ----
  std::map<int, std::vector<Item>> groups;
  for (size_t si = 0; si < plan.size(); ++si) {
    const DenseSpec &sp = plan[si];
    const model::TensorInfo &t = ck.at(sp.name);
    const bool w4 = sp.store == DenseSpec::Store::w4;  // packed int4 source
    const DenseSpec::Store est = sp.store;
    const bool w4r = est == DenseSpec::Store::w4r;  // BF16 source, int4 by RTN at load
    const bool w4d = sp.store == DenseSpec::Store::w4d;  // GPTQ int4 source, dequantised to dense at load
    const W4L layout = sp.w4_layout;
    const int64_t d0 = w4 || w4d ? w4_n(t, layout) : t.dim(0);
    const int64_t cols_all = w4 || w4d ? w4_k(t, layout) : t.shape.empty() ? 1 : t.numel() / std::max<int64_t>(1, d0);
    const size_t es = model::dtype_size(t.dtype);
    if (w4 || w4d) w4_check(ck, sp.name, t, d0, cols_all, layout);
    // A matrix source is bf16, or f16 (converted to bf16 at load); raw tensors keep their bytes.
    if (!w4 && !w4d && sp.store != DenseSpec::Store::raw && t.dtype != model::DType::bf16 && t.dtype != model::DType::f16)
      throw std::runtime_error("dense store: " + sp.name + " must be bf16 (or f16, converted at load) to be stored as a matrix");
    const int ncards = sp.split == DenseSpec::Split::host ? 1 : C;
    for (int c = 0; c < ncards; ++c) {
      Item it;
      it.spec = si;
      it.card = c;
      it.t = &t;
      it.esize = es;
      switch (sp.split) {
        case DenseSpec::Split::rows:
          if (!sp.row_ranges.empty()) {  // a row subset as one matrix, then its contiguous 1/C
            if (!sp.sections.empty() || !sp.append.empty()) throw std::runtime_error("dense store: " + sp.name + ": row_ranges take no sections / append");
            it.src_rows = model::slice_row_ranges(d0, sp.row_ranges, c, C);
          } else {
            it.src_rows = slice_rows(d0, sp.sections, c, C);
          }
          it.cols = cols_all;
          break;
        case DenseSpec::Split::cols:
          if (cols_all % C || ((w4 || w4r || w4d) && cols_all % (C * kernels::kW4Group)))
            throw std::runtime_error("dense store: " + sp.name + " columns do not split over the cards" + (w4 || w4r || w4d ? " by whole 128-k groups" : ""));
          it.src_rows = {{0, d0}};
          it.cols = cols_all / C;
          it.col0 = it.cols * c;
          break;
        default:
          it.src_rows = {{0, d0}};
          it.cols = cols_all;
      }
      it.src_t.assign(it.src_rows.size(), &t);
      if (w4 || w4d) it.src_s.assign(it.src_rows.size(), &ck.at(w4_scale_name(sp.name, layout)));
      if ((w4 || w4d) && layout == W4L::gptq) it.src_z.assign(it.src_rows.size(), &ck.at(w4_zeros_name(sp.name)));
      if (layout == W4L::ct_zp) {
        if (!w4 || sp.w4_group != kernels::kW4ZpGroup) throw std::runtime_error("dense store: " + sp.name + ": ct_zp needs Store::w4 with group 32");
        it.src_z.assign(it.src_rows.size(), &ck.at(w4_zeros_name(sp.name, layout)));
        it.G = kernels::kW4ZpGroup;
      }
      if (layout == W4L::mlx) {
        if (!w4 || sp.w4_group != kernels::kW4AffGroup) throw std::runtime_error("dense store: " + sp.name + ": mlx needs Store::w4 with group 64");
        it.src_z.assign(it.src_rows.size(), &ck.at(w4_zeros_name(sp.name, layout)));
        it.G = kernels::kW4AffGroup;
      }
      if (!sp.append.empty()) {
        if ((sp.split != DenseSpec::Split::replicate && sp.split != DenseSpec::Split::rows) || sp.store == DenseSpec::Store::raw)
          throw std::runtime_error("dense store: " + sp.name + ": appended rows need a replicated or row-split matrix");
        for (const auto &an : sp.append) {
          const model::TensorInfo &a = ck.at(an);
          const int64_t a_n = w4 || w4d ? w4_n(a, layout) : a.dim(0);
          if (a.dtype != t.dtype || ((w4 || w4d) ? w4_k(a, layout) : a.numel() / std::max<int64_t>(1, a.dim(0))) != cols_all)
            throw std::runtime_error("dense store: " + an + " does not match " + sp.name + "'s columns and dtype");
          if (w4 || w4d) {
            w4_check(ck, an, a, a_n, cols_all, layout);
            it.src_s.push_back(&ck.at(w4_scale_name(an, layout)));
            if (layout == W4L::gptq) it.src_z.push_back(&ck.at(w4_zeros_name(an)));
            if (layout == W4L::ct_zp || layout == W4L::mlx) it.src_z.push_back(&ck.at(w4_zeros_name(an, layout)));
          }
          if (sp.split == DenseSpec::Split::rows && sp.append_heads > 0 && sp.append_heads < C) {
            // fewer heads than cards: this card's head whole
            if (C % sp.append_heads || a_n % sp.append_heads) throw std::runtime_error("dense store: " + an + ": heads do not divide over the cards");
            const int64_t hr = a_n / sp.append_heads, h = static_cast<int64_t>(c) * sp.append_heads / C;
            it.src_rows.push_back({h * hr, (h + 1) * hr});
          } else if (sp.split == DenseSpec::Split::rows) {
            for (const auto &r : slice_rows(a_n, {}, c, C)) it.src_rows.push_back(r);  // its own 1/C
          } else {
            it.src_rows.push_back({0, a_n});
          }
          it.src_t.push_back(&a);
        }
      }
      for (const auto &r : it.src_rows) it.rows += r.second - r.first;
      const bool mat = sp.store != DenseSpec::Store::raw;
      if (mat && it.cols % 16) throw std::runtime_error("dense store: " + sp.name + " k is not a multiple of 16");
      it.rows_pad = mat ? (it.rows + 15) / 16 * 16 : it.rows;
      if (w4 || w4r) {
        if (it.cols % kernels::kW4Group) throw std::runtime_error("dense store: " + sp.name + " k is not a multiple of 128");
        it.scale_off = align_up(static_cast<size_t>(it.rows_pad * it.cols / 2));
        it.bytes = it.scale_off + align_up(static_cast<size_t>(it.rows_pad * (it.cols / it.G)) * 2);
        it.payload = static_cast<size_t>(it.rows * it.cols / 2) + static_cast<size_t>(it.rows * (it.cols / it.G)) * 2;
        if (sp.w4_layout == W4L::ct_zp) {  // + the zero points, two 4-bit values per byte
          it.zero_off = it.bytes;
          it.bytes += align_up(static_cast<size_t>(it.rows_pad * (it.cols / it.G) / 2));
          it.payload += static_cast<size_t>(it.rows * (it.cols / it.G) / 2);
        }
        if (sp.w4_layout == W4L::mlx) {  // + the bf16 biases
          it.zero_off = it.bytes;
          it.bytes += align_up(static_cast<size_t>(it.rows_pad * (it.cols / it.G)) * 2);
          it.payload += static_cast<size_t>(it.rows * (it.cols / it.G)) * 2;
        }
      } else if (est == DenseSpec::Store::i8 || (w4d && sp.dense_bytes == 1)) {
        it.bytes = align_up(static_cast<size_t>(it.rows_pad * it.cols)) + align_up(static_cast<size_t>(it.rows_pad) * 4);
        it.scale_off = align_up(static_cast<size_t>(it.rows_pad * it.cols));
        it.payload = static_cast<size_t>(it.rows * it.cols) + static_cast<size_t>(it.rows) * 4;
      } else if (est == DenseSpec::Store::bf16 || w4d) {
        it.bytes = align_up(static_cast<size_t>(it.rows_pad * it.cols) * 2);
        it.payload = static_cast<size_t>(it.rows * it.cols) * 2;
      } else {
        it.bytes = align_up(static_cast<size_t>(it.rows * it.cols) * es);
        it.payload = static_cast<size_t>(it.rows * it.cols) * es;
      }
      if (w4d && it.cols % kernels::kW4Group) throw std::runtime_error("dense store: " + sp.name + " k is not a multiple of 128");
      groups[sp.split == DenseSpec::Split::host ? -2 : sp.layer].push_back(std::move(it));
    }
  }

  // ---- blocks: one VRAM block per (card, group); host tensors in one pinned block ----
  std::map<int, std::vector<uint8_t *>> blocks;  // group -> [card] block
  size_t max_group = 0;
  for (auto &[g, items] : groups) {
    std::vector<size_t> size(static_cast<size_t>(C), 0);
    for (auto &it : items) {
      it.off = size[static_cast<size_t>(it.card)];
      size[static_cast<size_t>(it.card)] += it.bytes;
    }
    blocks[g].assign(static_cast<size_t>(C), nullptr);
    for (int c = 0; c < C; ++c) {
      if (!size[c]) continue;
      if (g == -2) {
        blocks[g][c] = static_cast<uint8_t *>(fab_.alloc_host(size[c]));
        host_blocks_.push_back(blocks[g][c]);
        stats_.host_bytes += size[c];
      } else {
        blocks[g][c] = static_cast<uint8_t *>(fab_.alloc(static_cast<size_t>(c), size[c]));
        dev_blocks_.push_back(blocks[g][c]);
        stats_.card_alloc[c] += size[c];
        max_group = std::max(max_group, size[c]);
      }
    }
  }
  std::vector<uint8_t *> stage(static_cast<size_t>(C) * 2, nullptr);
  for (auto &p : stage) p = static_cast<uint8_t *>(fab_.alloc_host(std::max<size_t>(max_group, kAlign)));

  // ---- pipeline over groups ----
  model::BufferPool pool;
  const model::BulkOptions ropt = pool.options();
  std::vector<int> order;
  for (const auto &kv : groups) order.push_back(kv.first);
  auto read_group = [&](int g) {
    std::vector<const model::TensorInfo *> ts;
    for (const auto &it : groups[g]) {
      for (const auto *t : it.src_t)
        if (std::find(ts.begin(), ts.end(), t) == ts.end()) ts.push_back(t);
      for (const auto *t : it.src_s)
        if (std::find(ts.begin(), ts.end(), t) == ts.end()) ts.push_back(t);
      for (const auto *t : it.src_z)
        if (std::find(ts.begin(), ts.end(), t) == ts.end()) ts.push_back(t);
    }
    return model::bulk_read(ck, ts, ropt);
  };
  std::vector<int> inflight(static_cast<size_t>(C), -1);
  try {
    // one reader thread, groups in order, read_ahead_ groups past the one packed
    model::SeqReader reader(static_cast<int>(order.size()), read_ahead_, [&](int gi) { return read_group(order[static_cast<size_t>(gi)]); },
                            reads_done_);
    for (size_t gi = 0; gi < order.size(); ++gi) {
      const int g = order[gi];
      auto t0 = clk::now();
      model::BulkBuffer buf = reader.take(static_cast<int>(gi));
      stats_.read_wait_s += secs(t0);
      stats_.read_bytes += buf.stats().read_bytes;
      const int slot = static_cast<int>(gi % 2);
      const bool host = g == -2;
      t0 = clk::now();
      for (int c = 0; c < C; ++c)
        if (inflight[c] == slot) {
          fab_.copy(c).wait();
          inflight[c] = -1;
        }
      stats_.h2d_wait_s += secs(t0);

      t0 = clk::now();
      // Destination of each item: the host block directly, or this slot's staging.
      auto dst = [&](const Item &it) -> uint8_t * {
        return (host ? blocks[g][0] : stage[static_cast<size_t>(it.card) * 2 + slot]) + it.off;
      };
      // Phase A (per item, row chunks): gather the slice; int8: quantise; raw: copy.
      std::vector<std::vector<uint16_t>> wide(groups[g].size());
      std::vector<std::vector<int8_t>> codes(groups[g].size());
      std::vector<std::vector<uint32_t>> ctw(groups[g].size());  // w4 ct: the slice's words [rows_pad][k/8]; gptq: [k/8][rows_pad]
      std::vector<std::vector<uint16_t>> cts(groups[g].size());  // w4 ct: its scales [rows_pad][k/G] (fp16 bits); gptq: [k/128][rows_pad]
      std::vector<std::vector<uint8_t>> ctz(groups[g].size());  // w4 ct_zp: its zero points [rows_pad][k/32], one per byte
      std::vector<std::vector<uint16_t>> ctb(groups[g].size());  // w4 affine: its bf16 biases [rows_pad][k/64] (scales in cts, bf16 bits)
      std::atomic<int64_t> inexact_scales{0};  // w4 ct_zp: bf16 scales fp16 cannot hold exactly
      std::atomic<int64_t> bad_zeros{0};  // w4 gptq: qzeros words that are not the symmetric 0x77777777
      std::vector<std::function<void()>> tasks;
      constexpr int64_t kChunkRows = 2048;
      for (size_t ii = 0; ii < groups[g].size(); ++ii) {
        const Item &it = groups[g][ii];
        const DenseSpec &sp = plan[it.spec];
        const bool w4 = sp.store == DenseSpec::Store::w4;
        const DenseSpec::Store est = sp.store;
        const bool w4r = est == DenseSpec::Store::w4r;
        const bool w4d = sp.store == DenseSpec::Store::w4d;
        const bool gptq = (sp.store == DenseSpec::Store::w4 || w4d) && sp.w4_layout == W4L::gptq;
        const int64_t src_cols = w4 || w4d ? w4_k(*it.t, sp.w4_layout) : it.t->numel() / std::max<int64_t>(1, it.t->dim(0));
        // slice row r -> its source row's bytes (appended tensors follow the first)
        auto src_row = [&it, &buf, src_cols](int64_t r) -> const uint8_t * {
          for (size_t i = 0; i < it.src_rows.size(); ++i) {
            const auto &rg = it.src_rows[i];
            if (r < rg.second - rg.first)
              return buf.data(*it.src_t[i]) + (static_cast<size_t>(rg.first + r) * src_cols + it.col0) * it.esize;
            r -= rg.second - rg.first;
          }
          throw std::logic_error("dense store: slice row out of range");
        };
        const bool f16src = it.t->dtype == model::DType::f16;
        const bool target_bf16 = est == DenseSpec::Store::bf16 || (w4d && sp.dense_bytes == 2);
        const bool target_i8 = est == DenseSpec::Store::i8 || (w4d && sp.dense_bytes == 1);
        if (target_bf16) wide[ii].assign(static_cast<size_t>(it.rows_pad * it.cols), 0);
        if (target_i8) codes[ii].assign(static_cast<size_t>(it.rows_pad * it.cols), 0);
        if (w4 || w4r || w4d) {
          // Pad rows (gptq: pad columns): code 8 (zero) under scale 0.
          ctw[ii].assign(static_cast<size_t>(it.rows_pad * (it.cols / 8)), 0x88888888u);
          cts[ii].assign(static_cast<size_t>(it.rows_pad * (it.cols / it.G)), 0);
          if (sp.w4_layout == W4L::ct_zp) ctz[ii].assign(static_cast<size_t>(it.rows_pad * (it.cols / it.G)), 8);
          if (sp.w4_layout == W4L::mlx) ctb[ii].assign(static_cast<size_t>(it.rows_pad * (it.cols / it.G)), 0);  // pad rows: scale 0, bias 0
        }
        // gptq: slice rows [r0, r1) (a range inside one source's n) -> their columns of every word row / scale row /
        // zeros word of this card's k range, gathered into the [k/8][rows_pad] slice.
        auto gather_gptq = [&it, &buf, &ctw, &cts, &bad_zeros, ii](int64_t r0, int64_t r1) {
          const int64_t words = it.cols / 8, gs = it.cols / kernels::kW4Group, w0 = it.col0 / 8, g0 = it.col0 / kernels::kW4Group;
          int64_t base = 0;
          for (size_t i = 0; i < it.src_rows.size(); ++i) {
            const auto &rg = it.src_rows[i];
            const int64_t len = rg.second - rg.first;
            const int64_t a = std::max(r0, base), b = std::min(r1, base + len);
            if (a < b) {
              const int64_t n_src = it.src_t[i]->dim(1), sa = rg.first + (a - base);
              const auto *pw = reinterpret_cast<const uint32_t *>(buf.data(*it.src_t[i]));
              const auto *ps = reinterpret_cast<const uint16_t *>(buf.data(*it.src_s[i]));
              const auto *pz = reinterpret_cast<const uint32_t *>(buf.data(*it.src_z[i]));
              for (int64_t wr = 0; wr < words; ++wr)
                std::memcpy(&ctw[ii][static_cast<size_t>(wr * it.rows_pad + a)], pw + static_cast<size_t>(w0 + wr) * n_src + sa, static_cast<size_t>(b - a) * 4);
              for (int64_t gr = 0; gr < gs; ++gr) {
                std::memcpy(&cts[ii][static_cast<size_t>(gr * it.rows_pad + a)], ps + static_cast<size_t>(g0 + gr) * n_src + sa, static_cast<size_t>(b - a) * 2);
                // zeros: the words covering columns [sa, sa + b - a) of this group row (8 columns per word; ranges are 16-aligned)
                int64_t bad = 0;
                for (int64_t zc = sa / 8; zc < (sa + (b - a) + 7) / 8; ++zc) bad += pz[static_cast<size_t>(g0 + gr) * (n_src / 8) + zc] != kGptqSymZeros;
                if (bad) bad_zeros += bad;
              }
            }
            base += len;
          }
        };
        // w4: slice row r -> (its words at col0, its scales at col0 / 128) in the source tensors
        auto src_w4 = [&it, &buf, src_cols](int64_t r, const uint8_t **words, const uint8_t **scales) {
          for (size_t i = 0; i < it.src_rows.size(); ++i) {
            const auto &rg = it.src_rows[i];
            if (r < rg.second - rg.first) {
              const size_t sr = static_cast<size_t>(rg.first + r);
              *words = buf.data(*it.src_t[i]) + (sr * (src_cols / 8) + it.col0 / 8) * 4;
              *scales = buf.data(*it.src_s[i]) + (sr * (src_cols / kernels::kW4Group) + it.col0 / kernels::kW4Group) * 2;
              return;
            }
            r -= rg.second - rg.first;
          }
          throw std::logic_error("dense store: slice row out of range");
        };
        for (int64_t r0 = 0; r0 < it.rows; r0 += kChunkRows) {
          const int64_t r1 = std::min(it.rows, r0 + kChunkRows);
          tasks.push_back([&, ii, r0, r1, src_row, src_w4, gather_gptq, gptq, f16src, src_cols] {  // loop locals by value: the task runs later
            const Item &it2 = groups[g][ii];
            const DenseSpec &sp2 = plan[it2.spec];
            const bool w4d2 = sp2.store == DenseSpec::Store::w4d;
            if (gptq && !w4d2) {
              gather_gptq(r0, r1);
              return;
            }
            if (w4d2) {  // GPTQ int4 source: gather the slice, dequantise each row to the dense target
              gather_gptq(r0, r1);
              thread_local std::vector<uint16_t> tmp;
              if (tmp.size() < static_cast<size_t>(it2.cols)) tmp.resize(it2.cols);
              const int64_t gs = it2.cols / kernels::kW4Group;
              for (int64_t r = r0; r < r1; ++r) {
                for (int64_t gr = 0; gr < gs; ++gr) {
                  const float sc = f16_to_float(cts[ii][static_cast<size_t>(gr * it2.rows_pad + r)]);
                  for (int64_t j = 0; j < 16; ++j) {
                    const uint32_t w = ctw[ii][static_cast<size_t>((gr * 16 + j) * it2.rows_pad + r)];
                    for (int b = 0; b < 8; ++b)
                      tmp[static_cast<size_t>(gr * 128 + j * 8 + b)] = float_to_bf16(sc * (static_cast<int>((w >> (4 * b)) & 15u) - 8));
                  }
                }
                if (sp2.dense_bytes == 1) {
                  float sc;
                  kernels::quantize_i8_rows(tmp.data(), 1, static_cast<int>(it2.cols),
                                            &codes[ii][static_cast<size_t>(r * it2.cols)], &sc, 1);
                  std::memcpy(dst(it2) + it2.scale_off + static_cast<size_t>(r) * 4, &sc, 4);
                } else {
                  std::memcpy(&wide[ii][static_cast<size_t>(r * it2.cols)], tmp.data(), static_cast<size_t>(it2.cols) * 2);
                }
              }
              return;
            }
            if (sp2.store == DenseSpec::Store::w4 && sp2.w4_layout == W4L::mlx) {
              // codes: the U8 [n][k/2] row bytes are the ct words; scales and biases stay bf16 (exact; fp16 cannot hold the
              // smallest scales)
              const int64_t words = it2.cols / 8, gs = it2.cols / it2.G, src_gs = src_cols / it2.G, g0 = it2.col0 / it2.G;
              for (int64_t r = r0; r < r1; ++r) {
                int64_t rr = r;
                size_t i = 0;
                while (rr >= it2.src_rows[i].second - it2.src_rows[i].first) {
                  rr -= it2.src_rows[i].second - it2.src_rows[i].first;
                  ++i;
                }
                const size_t sr = static_cast<size_t>(it2.src_rows[i].first + rr);
                std::memcpy(&ctw[ii][static_cast<size_t>(r * words)], buf.data(*it2.src_t[i]) + sr * (src_cols / 2) + it2.col0 / 2,
                            static_cast<size_t>(words) * 4);
                std::memcpy(&cts[ii][static_cast<size_t>(r * gs)], reinterpret_cast<const uint16_t *>(buf.data(*it2.src_s[i])) + sr * src_gs + g0,
                            static_cast<size_t>(gs) * 2);
                std::memcpy(&ctb[ii][static_cast<size_t>(r * gs)], reinterpret_cast<const uint16_t *>(buf.data(*it2.src_z[i])) + sr * src_gs + g0,
                            static_cast<size_t>(gs) * 2);
              }
              return;
            }
            if (sp2.store == DenseSpec::Store::w4 && sp2.w4_layout == W4L::ct_zp) {
              // words as ct; scales to fp16 (bf16 sources must convert exactly); zero points from the [n/8][k/32] words
              const int64_t words = it2.cols / 8, gs = it2.cols / it2.G, src_gs = src_cols / it2.G, g0 = it2.col0 / it2.G;
              int64_t inexact = 0;
              for (int64_t r = r0; r < r1; ++r) {
                int64_t rr = r;
                size_t i = 0;
                while (rr >= it2.src_rows[i].second - it2.src_rows[i].first) {
                  rr -= it2.src_rows[i].second - it2.src_rows[i].first;
                  ++i;
                }
                const size_t sr = static_cast<size_t>(it2.src_rows[i].first + rr);
                std::memcpy(&ctw[ii][static_cast<size_t>(r * words)], buf.data(*it2.src_t[i]) + (sr * (src_cols / 8) + it2.col0 / 8) * 4,
                            static_cast<size_t>(words) * 4);
                const auto *ps = reinterpret_cast<const uint16_t *>(buf.data(*it2.src_s[i])) + sr * src_gs + g0;
                const bool bf = it2.src_s[i]->dtype == model::DType::bf16;
                for (int64_t gr = 0; gr < gs; ++gr) {
                  uint16_t h = ps[gr];
                  if (bf) {
                    const float v = bf16_to_float(ps[gr]);
                    h = float_to_f16(v);
                    inexact += f16_to_float(h) != v;
                  }
                  cts[ii][static_cast<size_t>(r * gs + gr)] = h;
                }
                const auto *pz = reinterpret_cast<const uint32_t *>(buf.data(*it2.src_z[i])) + (sr / 8) * src_gs + g0;
                for (int64_t gr = 0; gr < gs; ++gr) ctz[ii][static_cast<size_t>(r * gs + gr)] = static_cast<uint8_t>((pz[gr] >> (4 * (sr % 8))) & 15u);
              }
              if (inexact) inexact_scales += inexact;
              return;
            }
            if (sp2.store == DenseSpec::Store::w4) {
              const int64_t words = it2.cols / 8, gs = it2.cols / kernels::kW4Group;
              for (int64_t r = r0; r < r1; ++r) {
                const uint8_t *w, *s;
                src_w4(r, &w, &s);
                std::memcpy(&ctw[ii][static_cast<size_t>(r * words)], w, static_cast<size_t>(words) * 4);
                std::memcpy(&cts[ii][static_cast<size_t>(r * gs)], s, static_cast<size_t>(gs) * 2);
              }
              return;
            }
            if (sp2.store == DenseSpec::Store::w4r) {  // RTN int4 into the ct words / scales of the slice
              thread_local std::vector<uint16_t> tmp;
              for (int64_t r = r0; r < r1; ++r) {
                uint16_t const *w = reinterpret_cast<const uint16_t *>(src_row(r));
                if (f16src) {
                  if (tmp.size() < static_cast<size_t>(it2.cols)) tmp.resize(it2.cols);
                  for (int64_t c2 = 0; c2 < it2.cols; ++c2) tmp[c2] = f16_to_bf16(w[c2]);
                  w = tmp.data();
                }
                kernels::quantize_w4_row(w, it2.cols, &ctw[ii][static_cast<size_t>(r * (it2.cols / 8))],
                                         &cts[ii][static_cast<size_t>(r * (it2.cols / kernels::kW4Group))], sp2.w4_mse);
              }
              return;
            }
            for (int64_t r = r0; r < r1; ++r) {
              const uint8_t *row = src_row(r);
              if (sp2.store == DenseSpec::Store::raw) {
                if (f16src) {  // an f16 source: the card reads bf16
                  auto *o = reinterpret_cast<uint16_t *>(dst(it2) + static_cast<size_t>(r * it2.cols) * it2.esize);
                  const auto *s = reinterpret_cast<const uint16_t *>(row);
                  for (int64_t c2 = 0; c2 < it2.cols; ++c2) o[c2] = f16_to_bf16(s[c2]);
                } else
                  std::memcpy(dst(it2) + static_cast<size_t>(r * it2.cols) * it2.esize, row, static_cast<size_t>(it2.cols) * it2.esize);
              } else if (sp2.store == DenseSpec::Store::bf16) {
                if (f16src) {
                  auto *o = &wide[ii][static_cast<size_t>(r * it2.cols)];
                  const auto *s = reinterpret_cast<const uint16_t *>(row);
                  for (int64_t c2 = 0; c2 < it2.cols; ++c2) o[c2] = f16_to_bf16(s[c2]);
                } else
                  std::memcpy(&wide[ii][static_cast<size_t>(r * it2.cols)], row, static_cast<size_t>(it2.cols) * 2);
              } else {  // i8
                uint16_t const *w = reinterpret_cast<const uint16_t *>(row);
                if (f16src) {
                  thread_local std::vector<uint16_t> tmp;
                  if (tmp.size() < static_cast<size_t>(it2.cols)) tmp.resize(it2.cols);
                  for (int64_t c2 = 0; c2 < it2.cols; ++c2) tmp[c2] = f16_to_bf16(w[c2]);
                  w = tmp.data();
                }
                float sc;
                kernels::quantize_i8_rows(w, 1, static_cast<int>(it2.cols),
                                          &codes[ii][static_cast<size_t>(r * it2.cols)], &sc, 1);
                std::memcpy(dst(it2) + it2.scale_off + static_cast<size_t>(r) * 4, &sc, 4);
              }
            }
          });
        }
        if ((est == DenseSpec::Store::i8 || (w4d && sp.dense_bytes == 1)) && it.rows_pad > it.rows) {
          const float one = 1.f;
          for (int64_t r = it.rows; r < it.rows_pad; ++r) std::memcpy(dst(it) + it.scale_off + static_cast<size_t>(r) * 4, &one, 4);
        }
      }
      run_tasks(tasks, threads);
      if (inexact_scales)
        throw std::runtime_error("dense store: " + std::to_string(inexact_scales.load()) + " bf16 weight scales in layer group " + std::to_string(g) +
                                 " are not exact in fp16 (the W4 kernels read fp16 scales)");
      if (bad_zeros) throw std::runtime_error("dense store: " + std::to_string(bad_zeros.load()) + " qzeros words in layer group " + std::to_string(g) +
                                              " are not the symmetric 0x77777777 (asymmetric GPTQ is not served)");
      // Phase B: pack matrices into DPAS tiles (row chunks of 16-multiples in parallel).
      tasks.clear();
      for (size_t ii = 0; ii < groups[g].size(); ++ii) {
        const Item &it = groups[g][ii];
        const DenseSpec &sp = plan[it.spec];
        if (sp.store == DenseSpec::Store::raw) continue;
        if (sp.store == DenseSpec::Store::w4 || sp.store == DenseSpec::Store::w4r)
          tasks.push_back([&, ii] {
            const Item &it2 = groups[g][ii];
            if (plan[it2.spec].store == DenseSpec::Store::w4 && plan[it2.spec].w4_layout == W4L::gptq) {  // already [k/128][n_pad]
              std::memcpy(dst(it2) + it2.scale_off, cts[ii].data(), cts[ii].size() * 2);
            } else if (plan[it2.spec].w4_layout == W4L::ct_zp) {  // [rows_pad][k/32] -> [k/32][rows_pad]; zero points -> nibbles
              const int64_t n = it2.rows_pad, gs = it2.cols / it2.G;
              auto *o = reinterpret_cast<uint16_t *>(dst(it2) + it2.scale_off);
              for (int64_t r = 0; r < n; ++r)
                for (int64_t gr = 0; gr < gs; ++gr) o[static_cast<size_t>(gr * n + r)] = cts[ii][static_cast<size_t>(r * gs + gr)];
              kernels::pack_w4_zeros(ctz[ii].data(), static_cast<int>(n), static_cast<int>(it2.cols), dst(it2) + it2.zero_off);
            } else if (plan[it2.spec].w4_layout == W4L::mlx) {  // scales and biases [rows_pad][k/64] -> [k/64][rows_pad]
              const int64_t n = it2.rows_pad, gs = it2.cols / it2.G;
              auto *os = reinterpret_cast<uint16_t *>(dst(it2) + it2.scale_off);
              auto *ob = reinterpret_cast<uint16_t *>(dst(it2) + it2.zero_off);
              for (int64_t r = 0; r < n; ++r)
                for (int64_t gr = 0; gr < gs; ++gr) {
                  os[static_cast<size_t>(gr * n + r)] = cts[ii][static_cast<size_t>(r * gs + gr)];
                  ob[static_cast<size_t>(gr * n + r)] = ctb[ii][static_cast<size_t>(r * gs + gr)];
                }
            } else
              kernels::ct_to_gptq_scales(cts[ii].data(), static_cast<int>(it2.rows_pad), static_cast<int>(it2.cols),
                                         reinterpret_cast<uint16_t *>(dst(it2) + it2.scale_off));
          });
        for (int64_t r0 = 0; r0 < it.rows_pad; r0 += kChunkRows)
          tasks.push_back([&, ii, r0] {
            const Item &it2 = groups[g][ii];
            const DenseSpec &sp3 = plan[it2.spec];
            const bool w4d3 = sp3.store == DenseSpec::Store::w4d;
            const int n = static_cast<int>(it2.rows_pad), kk = static_cast<int>(it2.cols);
            const int r1 = static_cast<int>(std::min<int64_t>(it2.rows_pad, r0 + kChunkRows));
            if (sp3.store == DenseSpec::Store::w4 && sp3.w4_layout == W4L::gptq)
              kernels::pack_w4_gptq_rows(ctw[ii].data(), n, kk, static_cast<int>(r0), r1, dst(it2));
            else if (sp3.store == DenseSpec::Store::w4 || sp3.store == DenseSpec::Store::w4r)
              kernels::pack_w4_ct_rows(ctw[ii].data(), n, kk, static_cast<int>(r0), r1, dst(it2));
            else if (sp3.store == DenseSpec::Store::bf16 || (w4d3 && sp3.dense_bytes == 2))
              kernels::pack_dense_bf16_rows(wide[ii].data(), n, kk, static_cast<int>(r0), r1, reinterpret_cast<uint16_t *>(dst(it2)));
            else
              kernels::pack_dense_i8_rows(codes[ii].data(), n, kk, static_cast<int>(r0), r1, reinterpret_cast<int8_t *>(dst(it2)));
          });
      }
      run_tasks(tasks, threads);
      stats_.pack_s += secs(t0);

      // Record tensors; ship the staging slot to each card.
      for (const Item &it : groups[g]) {
        const DenseSpec &sp = plan[it.spec];
        DeviceTensor dt;
        dt.store = sp.store == DenseSpec::Store::w4r ? DenseSpec::Store::w4
                 : sp.store == DenseSpec::Store::w4d ? (sp.dense_bytes == 1 ? DenseSpec::Store::i8 : DenseSpec::Store::bf16)
                 : sp.store;  // w4r is held as a w4 tensor; w4d as its dense target
        dt.ptr = blocks[g][static_cast<size_t>(it.card)] + it.off;
        dt.scale = dt.store == DenseSpec::Store::i8 ? reinterpret_cast<const float *>(dt.ptr + it.scale_off) : nullptr;
        dt.scale16 = dt.store == DenseSpec::Store::w4 ? reinterpret_cast<const uint16_t *>(dt.ptr + it.scale_off) : nullptr;
        if (sp.store == DenseSpec::Store::w4 && sp.w4_layout == W4L::ct_zp) {
          dt.zeros = dt.ptr + it.zero_off;
          dt.group = static_cast<int>(it.G);
          kernels::w4_set_zero_points(dt.ptr, dt.zeros);
          zp_registered_.push_back(dt.ptr);
        }
        if (sp.store == DenseSpec::Store::w4 && sp.w4_layout == W4L::mlx) {
          dt.biases = reinterpret_cast<const uint16_t *>(dt.ptr + it.zero_off);
          dt.group = static_cast<int>(it.G);
          kernels::w4_set_affine(dt.ptr, dt.biases);
          aff_registered_.push_back(dt.ptr);
        }
        dt.rows = it.rows;
        dt.cols = it.cols;
        dt.bytes = it.payload;
        if (host)
          for (int c = 0; c < C; ++c) t_[static_cast<size_t>(c)][sp.key.empty() ? sp.name : sp.key] = dt;
        else {
          t_[static_cast<size_t>(it.card)][sp.key.empty() ? sp.name : sp.key] = dt;
          stats_.card_bytes[static_cast<size_t>(it.card)][sp.comp] += it.payload;
        }
      }
      if (!host) {
        t0 = clk::now();
        for (int c = 0; c < C; ++c) {
          size_t bytes = 0;
          for (const Item &it : groups[g])
            if (it.card == c) bytes = std::max(bytes, it.off + it.bytes);
          if (!bytes) continue;
          fab_.copy(c).copy_batch(chunked(blocks[g][static_cast<size_t>(c)], stage[static_cast<size_t>(c) * 2 + slot], bytes), false);
          inflight[c] = slot;
        }
        stats_.h2d_wait_s += secs(t0);
      }
      { model::BulkBuffer gone = std::move(buf); }
    }
    const auto t0 = clk::now();
    for (int c = 0; c < C; ++c) fab_.copy(c).wait();
    stats_.h2d_wait_s += secs(t0);
  } catch (...) {
    for (int c = 0; c < C; ++c) fab_.copy(c).wait();
    for (auto *p : stage) fab_.release(p);
    throw;
  }
  for (auto *p : stage) fab_.release(p);
  stats_.total_s = secs(t_all);
}

}  // namespace xenithium:load
