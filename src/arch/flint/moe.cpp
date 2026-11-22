#include "xenithium/arch/flint/moe.hpp"

#include <algorithm>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>

#include "xenithium/kernels/gemm/mlp.hpp"
#include "xenithium/model/bulk_read.hpp"

namespace xenithium::arch::flint {

namespace k = xenithium::kernels;

MoeSet::MoeSet(Fabric &fab, const model::Checkpoint &ck, const load::DenseStore &dense, const qwen35::Config &cfg, int threads) : fab_(fab) {
  const auto t0 = std::chrono::steady_clock::now();
  const int C = static_cast<int>(fab.size()), H = cfg.hidden, E = cfg.experts, I = cfg.moe_inter, SI = cfg.shared_inter;
  shape_ = k::MoeAffShape{H, I, E, cfg.topk, SI, C, 0};
  if (!cfg.flint || static_cast<int>(cfg.moe.size()) != cfg.layers) throw std::invalid_argument("flint moe: not an arch flint configuration");
  if (I % (128 * C) || SI % (128 * C) || H % 128) throw std::runtime_error("flint moe: the intermediate columns do not split into 128s per card");
  const k::MoeAffLayout lay = k::moe_aff_layout(H, I / C), slay = k::moe_aff_layout(H, SI / C);
  cards_.resize(static_cast<size_t>(C));
  ev_.resize(static_cast<size_t>(C));
  layers_.resize(static_cast<size_t>(cfg.layers));
  // per card: scratch, wire, out, silu table
  std::vector<float *> silu(static_cast<size_t>(C));
  for (int c = 0; c < C; ++c) {
    Card &cd = cards_[static_cast<size_t>(c)];
    auto dev = [&](size_t b) {
      void *p = fab.alloc(static_cast<size_t>(c), std::max<size_t>(b, 4096));
      cd.mem.push_back(p);
      return p;
    };
    k::MoeAffShape s = shape_;
    s.card = c;
    auto &q = fab.compute(static_cast<size_t>(c));
    cd.sc.logits = static_cast<float *>(dev(static_cast<size_t>(k::kStepRows16) * E * 4));
    cd.sc.route = static_cast<k::MoeRoute *>(dev(sizeof(k::MoeRoute)));
    cd.sc.h = static_cast<bf16 *>(dev(k::moe_aff_h_elems(s) * 2));
    cd.sc.y = static_cast<float *>(dev(k::moe_aff_y_elems(s) * 4));
    cd.sc.partial = static_cast<float *>(dev(static_cast<size_t>(k::kStepRows16) * H * 4));
    cd.out = static_cast<bf16 *>(dev(static_cast<size_t>(k::kStepRows16) * H * 2));
    q.memset(cd.sc.h, 0, k::moe_aff_h_elems(s) * 2);  // rows past a step's live rows feed the 4 / 8-row tiles: finite
    silu[static_cast<size_t>(c)] = static_cast<float *>(dev(k::kSiluTabSize * 4));
    k::swiglu_table(q, silu[static_cast<size_t>(c)]);
    if (C > 1) {
      cd.wire.inbox = static_cast<uint32_t *>(fab.alloc_kernel_mailbox(k::pair_wire_words(H) * 4));
      cd.mem.push_back(cd.wire.inbox);
    }
    cd.wire.error = static_cast<uint32_t *>(dev(64));
    q.memset(cd.wire.error, 0, 64);
    q.wait();
  }
  for (int c = 0; c < C && C > 1; ++c) {
    cards_[static_cast<size_t>(c)].wire.peer_inbox = cards_[static_cast<size_t>(1 - c)].wire.inbox;
    k::pair_wire_init(fab.compute(static_cast<size_t>(c)), H, cards_[static_cast<size_t>(c)].wire.inbox).wait();
  }
  // the experts, layer by layer: read the layer's stacked tensors, pack every (expert, card) blob on `threads` threads,
  // upload per card
  const int nt = std::max(1, threads);
  for (int l = 0; l < cfg.layers; ++l) {
    if (!cfg.moe[static_cast<size_t>(l)]) continue;
    const std::string P = cfg.prefix + "layers." + std::to_string(l) + ".mlp.";
    std::vector<const model::TensorInfo *> ts;
    for (const char *mat : {"switch_mlp.gate_proj", "switch_mlp.up_proj", "switch_mlp.down_proj", "shared_expert.gate_proj", "shared_expert.up_proj",
                            "shared_expert.down_proj"})
      for (const char *suf : {".weight", ".scales", ".biases"}) ts.push_back(&ck.at(P + mat + suf));
    // shapes: stacked [E][n][...] (codes U8 [n][k/2], scales / biases BF16 [n][k/64])
    auto want = [&](const std::string &n, std::vector<int64_t> shape) {
      if (ck.at(P + n).shape != shape) throw std::runtime_error("flint moe: unexpected shape of " + P + n);
    };
    want("switch_mlp.gate_proj.weight", {E, I, H / 2});
    want("switch_mlp.up_proj.weight", {E, I, H / 2});
    want("switch_mlp.down_proj.weight", {E, H, I / 2});
    want("switch_mlp.down_proj.scales", {E, H, I / 64});
    want("shared_expert.gate_proj.weight", {SI, H / 2});
    want("shared_expert.down_proj.weight", {H, SI / 2});
    model::BulkOptions bo;
    bo.threads = nt;
    const auto buf = model::bulk_read(ck, ts, bo);
    auto u8 = [&](const std::string &n) { return buf.data(ck.at(P + n)); };
    auto u16 = [&](const std::string &n) { return reinterpret_cast<const uint16_t *>(buf.data(ck.at(P + n))); };
    auto src_of = [&](int e) {
      const size_t gw = static_cast<size_t>(I) * H / 2, gs = static_cast<size_t>(I) * H / 64, dw = static_cast<size_t>(H) * I / 2, ds = static_cast<size_t>(H) * I / 64;
      k::MoeAffSrc s;
      s.gate_w = u8("switch_mlp.gate_proj.weight") + e * gw, s.up_w = u8("switch_mlp.up_proj.weight") + e * gw;
      s.down_w = u8("switch_mlp.down_proj.weight") + e * dw;
      s.gate_s = u16("switch_mlp.gate_proj.scales") + e * gs, s.gate_b = u16("switch_mlp.gate_proj.biases") + e * gs;
      s.up_s = u16("switch_mlp.up_proj.scales") + e * gs, s.up_b = u16("switch_mlp.up_proj.biases") + e * gs;
      s.down_s = u16("switch_mlp.down_proj.scales") + e * ds, s.down_b = u16("switch_mlp.down_proj.biases") + e * ds;
      s.inter = I;
      return s;
    };
    const k::MoeAffSrc ssrc{u8("shared_expert.gate_proj.weight"), u8("shared_expert.up_proj.weight"), u8("shared_expert.down_proj.weight"),
                            u16("shared_expert.gate_proj.scales"), u16("shared_expert.gate_proj.biases"), u16("shared_expert.up_proj.scales"),
                            u16("shared_expert.up_proj.biases"), u16("shared_expert.down_proj.scales"), u16("shared_expert.down_proj.biases"), SI};
    Layer &L = layers_[static_cast<size_t>(l)];
    L.on = true;
    L.w.resize(static_cast<size_t>(C));
    std::vector<uint8_t> host(lay.bytes * E + slay.bytes);
    for (int c = 0; c < C; ++c) {
      std::atomic<int> next{0};
      std::vector<std::thread> pool;
      for (int t = 0; t < nt; ++t)
        pool.emplace_back([&, c] {
          for (int e = next++; e <= E; e = next++) {
            if (e < E) k::moe_aff_pack(H, C, c, src_of(e), lay, host.data() + static_cast<size_t>(e) * lay.bytes);
            else k::moe_aff_pack(H, C, c, ssrc, slay, host.data() + static_cast<size_t>(E) * lay.bytes);
          }
        });
      for (auto &th : pool) th.join();
      Card &cd = cards_[static_cast<size_t>(c)];
      auto &q = fab.compute(static_cast<size_t>(c));
      auto *blob = static_cast<uint8_t *>(fab.alloc(static_cast<size_t>(c), host.size()));
      cd.mem.push_back(blob);
      cd.bytes += host.size();
      q.memcpy(blob, host.data(), host.size()).wait();
      std::vector<const uint8_t *> table(static_cast<size_t>(E));
      for (int e = 0; e < E; ++e) table[static_cast<size_t>(e)] = blob + static_cast<size_t>(e) * lay.bytes;
      auto *dt = static_cast<const uint8_t **>(fab.alloc(static_cast<size_t>(c), std::max<size_t>(static_cast<size_t>(E) * sizeof(void *), 4096)));
      cd.mem.push_back(dt);
      q.memcpy(dt, table.data(), static_cast<size_t>(E) * sizeof(void *)).wait();
      k::MoeAffWeights &w = L.w[static_cast<size_t>(c)];
      w.blobs = dt;
      w.blob = lay;
      w.shared = blob + static_cast<size_t>(E) * lay.bytes;
      w.shared_lay = slay;
      w.silu_tab = silu[static_cast<size_t>(c)];
      w.router = dense.at(c, P + "gate.weight").dense();
      if (w.router.kind != k::DenseKind::bf16 || w.router.n != E || w.router.k != H) throw std::runtime_error("flint moe: the router must be bf16 [experts][hidden]");
    }
  }
  load_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

MoeSet::~MoeSet() {
  for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).wait();
  for (Card &cd : cards_)
    for (void *p : cd.mem) fab_.release(p);
}

void MoeSet::set_wire_bf16(bool on) {
  for (Card &cd : cards_) cd.wire.bf16 = on;
}

void MoeSet::run(int layer, int rows, const std::vector<const bf16 *> &x) {
  if (!has(layer)) throw std::invalid_argument("flint moe: layer " + std::to_string(layer) + " is not an MoE layer");
  const size_t C = cards_.size();
  if (x.size() != C) throw std::invalid_argument("flint moe: one input per card");
  const Layer &L = layers_[static_cast<size_t>(layer)];
  for (size_t c = 0; c < C; ++c) {
    k::MoeAffShape s = shape_;
    s.card = static_cast<int>(c);
    Card &cd = cards_[c];
    ev_[c] = k::moe_aff_decode(fab_.compute(c), s, rows, x[c], L.w[c], cd.sc, C > 1 ? &cd.wire : nullptr, parity_, cd.out);
  }
  parity_ ^= 1;
}

void MoeSet::reserve_rows(int rows_max) {
  if (rows_max < 1) throw std::invalid_argument("flint moe: reserve_rows of no rows");
  for (size_t c = 0; c < cards_.size(); ++c) {
    Card &cd = cards_[c];
    if (cd.rows.rows_max >= rows_max) continue;
    k::MoeAffShape s = shape_;
    s.card = static_cast<int>(c);
    const auto z = k::moe_aff_rows_sizes(s, rows_max);
    size_t total = 0;
    auto dev = [&](size_t b) {
      b = std::max<size_t>(b, 4096);
      void *p = fab_.alloc(c, b);
      cd.mem.push_back(p);
      total += b;
      return p;
    };
    k::MoeAffRowsScratch &r = cd.rows;
    r.rows_max = rows_max;
    r.logits = static_cast<float *>(dev(z.logits * 4));
    r.ids = static_cast<int32_t *>(dev(z.ids * 4));
    r.topw = static_cast<float *>(dev(z.ids * 4));
    r.slot = static_cast<int32_t *>(dev(z.ids * 4));
    r.row = static_cast<int32_t *>(dev(z.row * 4));
    r.seg = static_cast<int32_t *>(dev(z.seg * 4));
    r.seg_n = static_cast<int32_t *>(dev(64));
    r.ax = static_cast<float *>(dev(z.ax * 4));
    r.ah = static_cast<float *>(dev(z.ah * 4));
    r.lds_x = (rows_max + 31) / 32 * 32;
    r.lds_h = (rows_max * shape_.topk + 31) / 32 * 32;
    r.h = static_cast<bf16 *>(dev(z.h * 2));
    r.y = static_cast<float *>(dev(z.y * 4));
    r.sh = static_cast<bf16 *>(dev(z.sh * 2));
    r.sy = static_cast<float *>(dev(z.sy * 4));
    r.part = static_cast<float *>(dev(z.part * 4));
    auto &q = fab_.compute(c);
    q.memset(r.ax, 0, z.ax * 4);  // the row sums' padding rows are read by the last tiles
    q.memset(r.ah, 0, z.ah * 4);
    q.wait();
    cd.rows_bytes = total;
  }
}

void MoeSet::run_rows(int layer, int m, const std::vector<const bf16 *> &x, const std::vector<const k::PairRowsWire *> &wire, int parity,
                      const std::vector<bf16 *> &out) {
  if (!has(layer)) throw std::invalid_argument("flint moe: layer " + std::to_string(layer) + " is not an MoE layer");
  const size_t C = cards_.size();
  if (x.size() != C || wire.size() != C || out.size() != C) throw std::invalid_argument("flint moe: one input, wire and output per card");
  const Layer &L = layers_[static_cast<size_t>(layer)];
  for (size_t c = 0; c < C; ++c) {
    k::MoeAffShape s = shape_;
    s.card = static_cast<int>(c);
    Card &cd = cards_[c];
    if (cd.rows.rows_max < m) throw std::logic_error("flint moe: run_rows past reserve_rows");
    ev_[c] = k::moe_aff_rows(fab_.compute(c), s, m, x[c], L.w[c], cd.rows, wire[c], parity, out[c]);
  }
}

uint32_t MoeSet::wire_error(int c) {
  uint32_t e = 0;
  fab_.compute(static_cast<size_t>(c)).memcpy(&e, cards_[static_cast<size_t>(c)].wire.error, 4).wait();
  return e;
}

}  // namespace xenithium::arch::flint
