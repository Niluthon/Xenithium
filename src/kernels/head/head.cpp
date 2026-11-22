#include "xenithium/kernels/head/head.hpp"
#include "xenithium/kernels/rows.hpp"
#include "xenithium/util/gumbel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include <sycl/ext/intel/esimd.hpp>

namespace xenithium::kernels {
namespace {

constexpr int kPickItems = 256;

// Order-preserving 16-bit image of a bf16 (bits b): -0 -> +0, NaN -> lowest.
inline uint32_t ordered_bf16(uint16_t b) {
  if ((b & 0x7fff) > 0x7f80) return 0;  // NaN never wins
  if (b == 0x8000) b = 0;  // -0 == +0
  return (b & 0x8000) ? static_cast<uint32_t>(~b & 0xffff) : static_cast<uint32_t>(b | 0x8000);
}

inline uint64_t make_key(float logit, uint32_t global_index) {
  const head_bf16 h(logit);
  uint16_t b;
  std::memcpy(&b, &h, 2);
  return (static_cast<uint64_t>(ordered_bf16(b)) << 32) | (0xFFFFFFFEu - global_index);
}

class EmbedGather;
class EmbedGatherExt;
class PickChunks;
template <bool WIRE>
class PickMerge;

// One ESIMD work-item per row. The wire words cross PCIe and the receiving card's caches are
// not coherent with peer writes (a cached load can return a stale key), so every mailbox access
// is an uncached L1/L3 load or store, with a fence inside the spin so the load is not hoisted.
template <bool WIRE>
sycl::event merge(sycl::queue &q, const uint64_t *keys, int rows, int chunks, PickWire w, int32_t *tokens, uint64_t *local_keys) {
  namespace esimd = sycl::ext::intel::esimd;
  return q.parallel_for<PickMerge<WIRE>>(sycl::nd_range<1>(kPickMaxRows, kPickMaxRows), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    const int r = static_cast<int>(it.get_local_id(0));
    int parity = 0;
    if constexpr (WIRE) parity = w.step[0] & 1;
    if (r < rows) {
      uint64_t best = 0;
      for (int c0 = 0; c0 < chunks; c0 += 8) {
        uint64_t v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = keys[static_cast<size_t>(r) * chunks + (c0 + i < chunks ? c0 + i : chunks - 1)];
#pragma unroll
        for (int i = 0; i < 8; ++i) best = v[i] > best ? v[i] : best;
      }
      if (local_keys) local_keys[r] = best;
      if constexpr (WIRE) {
        constexpr auto uc = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>,
                                              esimd::cache_hint_L2<esimd::cache_hint::uncached>, esimd::alignment<8>};
        // one hop: store best into the peer's region, spin on this card's, keep the max; on four cards a second hop
        // with card ^ 2 (its own region)
        auto hop = [&](uint32_t *peer, size_t region) SYCL_ESIMD_FUNCTION {
          const size_t slot = region + (static_cast<size_t>(parity) * kPickMaxRows + r) * 2;
          esimd::simd<uint32_t, 2> mine;
          mine[0] = static_cast<uint32_t>(best >> 32);
          mine[1] = static_cast<uint32_t>(best);
          esimd::block_store<uint32_t, 2>(peer + slot, mine, uc);
          esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
          esimd::simd<uint32_t, 2> got = esimd::block_load<uint32_t, 2>(w.inbox + slot, uc);
          bool ok = got[0] != kMailEmpty && got[1] != kMailEmpty;
          if (!ok) {
            const uint64_t t0 = esimd::rdtsc();
            while (!ok) {
              esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
              got = esimd::block_load<uint32_t, 2>(w.inbox + slot, uc);
              ok = got[0] != kMailEmpty && got[1] != kMailEmpty;
              if (!ok && esimd::rdtsc() - t0 > w.wait_cycles) break;
            }
          }
          if (ok) {
            const uint64_t other = (static_cast<uint64_t>(static_cast<uint32_t>(got[0])) << 32) | static_cast<uint32_t>(got[1]);
            best = other > best ? other : best;
            esimd::block_store<uint32_t, 2>(w.inbox + slot, esimd::simd<uint32_t, 2>(kMailEmpty), uc);
          } else {
            w.error[0] = 1u;  // every writer stores the same value
          }
        };
        hop(w.peer_inbox, 0);
        if (w.peer2_inbox) hop(w.peer2_inbox, kPickWireWords);
      }
      tokens[r] = static_cast<int32_t>(0xFFFFFFFEu - static_cast<uint32_t>(best));
    }
    if constexpr (WIRE) {
      esimd::barrier();
      if (r == 0) w.step[0] = w.step[0] + 1;
    }
  });
}

}  // namespace

sycl::event embed_gather(sycl::queue &q, const head_bf16 *table, const int32_t *ids, int rows, int hidden, int streams, head_bf16 *res) {
  return q.parallel_for<EmbedGather>(sycl::range<1>(static_cast<size_t>(rows) * hidden), [=](sycl::id<1> id) {
    const int r = static_cast<int>(id[0]) / hidden, c = static_cast<int>(id[0]) % hidden;
    const head_bf16 v = table[static_cast<size_t>(ids[r]) * hidden + c];
    for (int s = 0; s < streams; ++s) res[(static_cast<size_t>(r) * streams + s) * hidden + c] = v;
  });
}

sycl::event embed_gather_ext(sycl::queue &q, const head_bf16 *table, const int32_t *ids, int rows, int hidden, int streams, head_bf16 *res, const head_bf16 *extra,
                             int32_t base) {
  if (!extra) return embed_gather(q, table, ids, rows, hidden, streams, res);
  return q.parallel_for<EmbedGatherExt>(sycl::range<1>(static_cast<size_t>(rows) * hidden), [=](sycl::id<1> id) {
    const int r = static_cast<int>(id[0]) / hidden, c = static_cast<int>(id[0]) % hidden;
    const int32_t t = ids[r];
    const head_bf16 v = t >= base ? extra[static_cast<size_t>(t - base) * hidden + c] : table[static_cast<size_t>(t) * hidden + c];
    for (int s = 0; s < streams; ++s) res[(static_cast<size_t>(r) * streams + s) * hidden + c] = v;
  });
}

int pick_chunks(int n) { return (n + kPickChunk - 1) / kPickChunk; }
size_t pick_scratch_bytes(int n) { return static_cast<size_t>(kPickMaxRows) * pick_chunks(n) * sizeof(uint64_t); }

sycl::event greedy_pick(sycl::queue &q, const float *logits, int ld, int rows, int n, int col_base, void *scratch, const PickWire *wire,
                        int32_t *tokens, sycl::event *chunks_event, uint64_t *local_keys) {
  if (rows < 1 || rows > kPickMaxRows || n < 1 || ld < n) throw std::invalid_argument("greedy_pick: shape");
  const int chunks = pick_chunks(n);
  auto *keys = static_cast<uint64_t *>(scratch);
  sycl::event ce = q.parallel_for<PickChunks>(sycl::nd_range<2>({static_cast<size_t>(rows), static_cast<size_t>(chunks) * kPickItems}, {1, kPickItems}), [=](sycl::nd_item<2> it) {
    const int r = static_cast<int>(it.get_group(0)), ch = static_cast<int>(it.get_group(1)), lid = static_cast<int>(it.get_local_id(1));  // a (row, chunk) grid
    const int c0 = ch * kPickChunk, c1 = sycl::min(n, c0 + kPickChunk);
    uint64_t best = 0;
    for (int c = c0 + lid; c < c1; c += kPickItems) {
      const uint64_t k = make_key(logits[static_cast<size_t>(r) * ld + c], static_cast<uint32_t>(col_base + c));
      best = k > best ? k : best;
    }
    best = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<uint64_t>());
    if (lid == 0) keys[static_cast<size_t>(r) * chunks + ch] = best;
  });
  if (chunks_event) *chunks_event = ce;
  return wire ? merge<true>(q, keys, rows, chunks, *wire, tokens, local_keys) : merge<false>(q, keys, rows, chunks, PickWire{}, tokens, local_keys);
}

uint64_t pick_key(float logit, int64_t global_index) { return make_key(logit, static_cast<uint32_t>(global_index)); }

void draft_gumbel(sycl::queue &q, float *logits, int ld, int rows, int n, int col_base, int sub_k, int sub_s0, const int32_t *seg, int seg_stride,
                  int seg_rows, int slot_fixed, const DraftNoiseRec *noise, int pass) {
  if (rows < 1 || n < 1 || ld < n || (seg ? !seg_rows_ok(rows, seg_rows, seg_stride) || seg_rows == 0 : slot_fixed < 0)) throw std::invalid_argument("draft_gumbel: shape");
  constexpr int kItems = 256;
  const int per_row = (n + kItems - 1) / kItems;
  q.parallel_for<class DraftGumbel>(sycl::nd_range<1>(static_cast<size_t>(rows) * per_row * kItems, kItems), [=](sycl::nd_item<1> it) {
    const int g = static_cast<int>(it.get_group(0)), r = g / per_row;
    const int c = (g % per_row) * kItems + static_cast<int>(it.get_local_id(0));
    const int slot = seg ? seg_row(seg, seg_stride, seg_rows, r).slot : slot_fixed;  // any layout
    const DraftNoiseRec rec = noise[slot];
    if (!(rec.inv_t > 0.0f) || c >= n) return;
    const int cv = col_base + c;
    const int32_t tok = sub_k > 0 && cv >= sub_k ? cv - sub_k + sub_s0 : cv;
    float &x = logits[static_cast<size_t>(r) * ld + c];
    x = x * rec.inv_t + util::gumbel(rec.key, rec.pos + static_cast<uint64_t>(pass), tok);
  });
}

// ---- the two-stage lm_head's shortlist ----
namespace {

constexpr int kHsTile = 16, kHsSuper = 16, kHsNP = 32;  // columns per tile, tiles per super tile, keys per lane (<= 512 / 16)

inline uint32_t float_key(float v) {  // order-preserving image of an fp32 (NaN above +inf: a NaN tile is rescored)
  const uint32_t u = sycl::bit_cast<uint32_t>(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exact top-T of the 16 x kHsNP keys a sub-group holds (lane l, slot j = key j * 16 + l in index order; key 0 = none),
// ascending index into out[0..T): the T-th largest key by a 2-bit-per-step threshold search on packed 10-bit counts, then
// an ordered compaction (keys above it, then equal keys from the lowest index).
template <class Sg, class Id>
inline void sg_topk(const Sg &sg, const uint32_t (&kv)[kHsNP], int T, Id id, int32_t *out) {
  uint32_t mx = 0, mn = 0xFFFFFFFFu;
#pragma unroll
  for (int j = 0; j < kHsNP; ++j) mx = sycl::max(mx, kv[j]), mn = kv[j] ? sycl::min(mn, kv[j]) : mn;
  mx = sycl::reduce_over_group(sg, mx, sycl::maximum<uint32_t>());
  mn = sycl::reduce_over_group(sg, mn, sycl::minimum<uint32_t>());
  const uint32_t diff = mx ^ mn;
  int left = diff ? 32 - sycl::clz(diff) : 0;
  uint32_t base = left == 32 ? 0u : (mx >> left) << left;  // count(key >= base) >= T (base <= every valid key)
  while (left > 0) {
    const int w = left >= 2 ? 2 : 1, shift = left - w;
    uint32_t pk = 0;  // count(key >= base | d << shift) for d = 1..3, 10 bits each (16 x 32 keys <= 512)
#pragma unroll
    for (int d = 1; d < 4; ++d) {
      if (d >= (1 << w)) break;
      const uint32_t c = base | (static_cast<uint32_t>(d) << shift);
      uint32_t n = 0;
#pragma unroll
      for (int j = 0; j < kHsNP; ++j) n += kv[j] >= c;
      pk |= n << (10 * (d - 1));
    }
    pk = sycl::reduce_over_group(sg, pk, sycl::plus<uint32_t>());
    int best = 0;
    for (int d = (1 << w) - 1; d >= 1; --d)
      if (static_cast<int>((pk >> (10 * (d - 1))) & 0x3FFu) >= T) {
        best = d;
        break;
      }
    base |= static_cast<uint32_t>(best) << shift;
    left = shift;
  }
  const uint32_t th = base;  // the T-th largest key
  int gt = 0;
#pragma unroll
  for (int j = 0; j < kHsNP; ++j) gt += kv[j] > th;
  const int need = T - sycl::reduce_over_group(sg, gt, sycl::plus<int>());
  int o = 0, eq_seen = 0;
#pragma unroll
  for (int j = 0; j < kHsNP; ++j) {
    const bool isgt = kv[j] > th, iseq = kv[j] == th;
    const int eqb = sycl::exclusive_scan_over_group(sg, static_cast<int>(iseq), sycl::plus<int>());
    const bool take = isgt || (iseq && eq_seen + eqb < need);
    const int pos = sycl::exclusive_scan_over_group(sg, static_cast<int>(take), sycl::plus<int>());
    if (take) out[o + pos] = id(j);
    o += sycl::reduce_over_group(sg, static_cast<int>(take), sycl::plus<int>());
    eq_seen += sycl::reduce_over_group(sg, static_cast<int>(iseq), sycl::plus<int>());
  }
}

template <bool BF16>
class HeadTileMax;
template <bool BF16>
class HeadTopTiles;
class HeadCandGather;

// the key of a tile maximum for the sampling candidates (head_cand_tiles): the order-preserving
// image of the bf16 rounding (the host sampler compares the rounded values: bf16(max) == max of the rounded values, since
// rounding is monotone), -0 == +0, NaN never (key 0 = none). Ties then fall to the lowest tile = the lowest token ids,
// the host's tie rule.
inline uint32_t tile_key_bf16(float v) {
  const head_bf16 h(v);
  return ordered_bf16(sycl::bit_cast<uint16_t>(h));
}

// One part of a row: columns [0, n) of logits (stride ld), its top T tiles at tiles[r * ostr + i] as tile + tbase.
template <bool BF16>
sycl::event select_part(sycl::queue &q, const float *logits, int ld, int rows, int n, int T, void *scratch, int32_t *tiles, int ostr, int tbase,
                        sycl::event *max_event) {
  const int NT = n / kHsTile, NS = (NT + kHsSuper - 1) / kHsSuper;
  if (rows < 1 || rows > kPickMaxRows || n % kHsTile || ld < n || T < 1 || T > kHeadTilesMax || NS > kHsNP * 16 || NS < T)
    throw std::invalid_argument("head_select_tiles: shape");
  auto *keys = static_cast<uint32_t *>(scratch);  // [rows][NT]
  uint32_t *skeys = keys + static_cast<size_t>(kPickMaxRows) * NT;  // [rows][NS]
  const size_t ntp = static_cast<size_t>(NS) * kHsSuper, total = static_cast<size_t>(rows) * ntp, wg = 256;
  sycl::event me = q.parallel_for<HeadTileMax<BF16>>(sycl::nd_range<1>((total + wg - 1) / wg * wg, wg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
    const size_t i = it.get_global_id(0);
    if (i >= total) return;  // whole sub-groups (total % 16 == 0)
    const size_t r = i / ntp, t = i % ntp;
    uint32_t key = 0;
    if (t < static_cast<size_t>(NT)) {
      const sycl::float4 *p = reinterpret_cast<const sycl::float4 *>(logits + r * ld + t * kHsTile);
      const sycl::float4 m = sycl::fmax(sycl::fmax(p[0], p[1]), sycl::fmax(p[2], p[3]));
      const float mx = sycl::fmax(sycl::fmax(m.x(), m.y()), sycl::fmax(m.z(), m.w()));
      if constexpr (BF16) key = tile_key_bf16(mx);
      else key = float_key(mx);
      keys[r * NT + t] = key;
    }
    const auto sg = it.get_sub_group();
    const uint32_t sk = sycl::reduce_over_group(sg, key, sycl::maximum<uint32_t>());
    if (sg.get_local_linear_id() == 0) skeys[r * NS + t / kHsSuper] = sk;
  });
  if (max_event) *max_event = me;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<int32_t, 1> ss(sycl::range<1>(kHeadTilesMax), h);
    h.parallel_for<HeadTopTiles<BF16>>(sycl::nd_range<1>(static_cast<size_t>(rows) * 16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
      const int r = static_cast<int>(it.get_group(0)), ln = static_cast<int>(it.get_local_id(0));
      const auto sg = it.get_sub_group();
      uint32_t kv[kHsNP];
#pragma unroll
      for (int j = 0; j < kHsNP; ++j) {
        const int s = j * 16 + ln;
        kv[j] = s < NS ? skeys[static_cast<size_t>(r) * NS + s] : 0u;
      }
      sg_topk(sg, kv, T, [&](int j) { return j * 16 + ln; }, &ss[0]);
      sycl::group_barrier(it.get_group());
#pragma unroll
      for (int j = 0; j < kHsNP; ++j) {  // slot j < T: super tile ss[j]'s tile ln (ascending tiles in index order)
        const int t = j < T ? ss[j] * kHsSuper + ln : NT;
        kv[j] = t < NT ? keys[static_cast<size_t>(r) * NT + t] : 0u;
      }
      sg_topk(sg, kv, T, [&](int j) { return ss[j] * kHsSuper + ln + tbase; }, tiles + static_cast<size_t>(r) * ostr);
    });
  });
}

// A row wider than one sub-group's 512 super tiles (TP1's whole vocabulary: 970) in head_select_parts(n) equal parts, each
// its own top T: the row's list is the parts' lists one after the other (P x T tiles, ascending).
template <bool BF16>
sycl::event select_tiles(sycl::queue &q, const float *logits, int ld, int rows, int n, int T, void *scratch, int32_t *tiles,
                         sycl::event *max_event) {
  const int P = head_select_parts(n), w = n / P;
  sycl::event e;
  for (int p = 0; p < P; ++p)  // one scratch: the parts run in order on the queue
    e = select_part<BF16>(q, logits + static_cast<size_t>(p) * w, ld, rows, w, T, scratch, tiles + static_cast<size_t>(p) * T, P * T, p * (w / kHsTile),
                          p == 0 ? max_event : nullptr);
  return e;
}

}  // namespace

int head_select_parts(int n) {
  const int ns = (n / kHsTile + kHsSuper - 1) / kHsSuper, P = (ns + kHsNP * 16 - 1) / (kHsNP * 16);
  if (P < 1 || n % (P * kHsTile)) throw std::invalid_argument("head_select_parts: the row does not split into whole tiles per part");
  return P;
}

size_t head_select_scratch_bytes(int n) {
  const size_t nt = static_cast<size_t>(n / kHsTile), ns = (nt + kHsSuper - 1) / kHsSuper;
  return static_cast<size_t>(kPickMaxRows) * (nt + ns) * 4;
}

sycl::event head_select_tiles(sycl::queue &q, const float *logits, int ld, int rows, int n, int T, void *scratch, int32_t *tiles, sycl::event *max_event) {
  return select_tiles<false>(q, logits, ld, rows, n, T, scratch, tiles, max_event);
}

sycl::event head_cand_tiles(sycl::queue &q, const float *logits, int ld, int rows, int n, int T, void *scratch, int32_t *tiles, float *vals,
                            sycl::event *select_event) {
  const sycl::event se = select_tiles<true>(q, logits, ld, rows, n, T, scratch, tiles, nullptr);
  if (select_event) *select_event = se;
  const int PT = head_select_parts(n) * T;  // tiles per row
  const size_t total = static_cast<size_t>(rows) * PT * kHsTile;
  return q.parallel_for<HeadCandGather>(sycl::range<1>(total), [=](sycl::id<1> id) {
    const size_t i = id[0], rt = i / kHsTile, j = i % kHsTile, r = rt / static_cast<size_t>(PT);
    vals[i] = logits[r * ld + static_cast<size_t>(tiles[rt]) * kHsTile + j];
  });
}

void head_select_ref(const float *row, int n, int T, int32_t *tiles) {
  const int P = head_select_parts(n), w = n / P, NT = w / kHsTile;
  for (int p = 0; p < P; ++p) {  // each part its own top T (head_select_parts), its tiles at p x T
    const float *pr = row + static_cast<size_t>(p) * w;
    std::vector<std::pair<uint32_t, int>> km(static_cast<size_t>(NT));
    for (int t = 0; t < NT; ++t) {
      float m = pr[t * kHsTile];
      for (int j = 1; j < kHsTile; ++j) m = std::fmax(m, pr[t * kHsTile + j]);
      uint32_t u;
      std::memcpy(&u, &m, 4);
      km[static_cast<size_t>(t)] = {(u & 0x80000000u) ? ~u : (u | 0x80000000u), t};
    }
    std::stable_sort(km.begin(), km.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    std::vector<int32_t> v(static_cast<size_t>(T));
    for (int i = 0; i < T; ++i) v[static_cast<size_t>(i)] = km[static_cast<size_t>(i)].second + p * NT;
    std::sort(v.begin(), v.end());
    std::copy(v.begin(), v.end(), tiles + static_cast<size_t>(p) * T);
  }
}

void head_cand_ref(const float *row, int n, int T, int32_t *tiles) {
  const int P = head_select_parts(n), w = n / P, NT = w / kHsTile;
  for (int p = 0; p < P; ++p) {  // each part its own top T (head_select_parts), its tiles at p x T
    const float *pr = row + static_cast<size_t>(p) * w;
    std::vector<std::pair<uint32_t, int>> km(static_cast<size_t>(NT));
    for (int t = 0; t < NT; ++t) {
      float m = pr[t * kHsTile];
      for (int j = 1; j < kHsTile; ++j) m = std::fmax(m, pr[t * kHsTile + j]);
      const head_bf16 h(m);
      uint16_t b;
      std::memcpy(&b, &h, 2);
      km[static_cast<size_t>(t)] = {ordered_bf16(b), t};
    }
    std::stable_sort(km.begin(), km.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    std::vector<int32_t> v(static_cast<size_t>(T));
    for (int i = 0; i < T; ++i) v[static_cast<size_t>(i)] = km[static_cast<size_t>(i)].second + p * NT;
    std::sort(v.begin(), v.end());
    std::copy(v.begin(), v.end(), tiles + static_cast<size_t>(p) * T);
  }
}

int64_t pick_ref(const float *row, int64_t n, int64_t col_base) {
  // Independent of the key: compare the bf16-rounded values as floats (-0 == +0 for free).
  int64_t best = -1;
  float bv = 0.f;
  for (int64_t i = 0; i < n; ++i) {
    const float v = static_cast<float>(head_bf16(row[i]));
    if (std::isnan(v)) continue;
    if (best < 0 || v > bv) {
      best = i;
      bv = v;
    }
  }
  return best < 0 ? col_base : col_base + best;
}

}  // namespace xenithium:kernels
