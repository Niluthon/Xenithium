// Xenithium · qwen38 prefix cache: snapshots of a slot's sequence state, restore, fork, trim.
//
// The state of a slot outside its KV pages (model.hpp, "the prefix cache"): per GDN layer S and both conv windows, per
// QSA layer the raw indexer-key ring (position % 16), the PLE position ring, the drafter's indexer ring; on the host the
// position, the GDN parity and the PLE n-gram context. Every copy runs on the compute queue of its card, in order with
// the calls around it: a save gathers the pieces into a device staging buffer and copies it to pinned host memory in one
// transfer (nothing waits); a restore copies host -> staging -> the pieces. A fork copies the source's KV pages below the
// snapshot (every QSA layer's K / V / scales / compressed keys and the drafter's) into the destination's pages with
// kernels:kv_copy_pages, then restores.
#include <cstdio>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include "model_internal.hpp"
#include "xenithium/kernels/attention/kv_copy.hpp"
#include "xenithium/kernels/attention/qsa_ref.hpp"

namespace xenithium::arch::qwen38 {

namespace k = xenithium::kernels;

std::vector<std::pair<void *, size_t>> Model::snap_pieces(int c, int slot) const {
  const Card &K = *cards_.at(static_cast<size_t>(c));
  const size_t s = static_cast<size_t>(slot);
  std::vector<std::pair<void *, size_t>> v;
  for (int l : gs_.layers) {
    const k::GdnState &st = K.gst[static_cast<size_t>(l)];
    v.push_back({st.S + s * st.slot_S, st.slot_S * sizeof(float)});
    for (int pr = 0; pr < 2; ++pr) v.push_back({st.win[pr] + s * st.slot_win, st.slot_win * 2});
  }
  const size_t ring = static_cast<size_t>(k::kQsaRing) * k::kQsaIdxDim;
  for (int l : qs_.layers) v.push_back({K.qc[static_cast<size_t>(l)].ring + s * ring, ring * 2});
  v.push_back({K.ple_st.ring + s * K.ple_st.slot_elems, K.ple_st.slot_elems * 2});
  if (mtp_) v.push_back({mtp_->cards[static_cast<size_t>(c)].qc.ring + s * ring, ring * 2});
  if (dflash_)  // the DFlash drafter's context rings (per slot, not paged: a fork restores them)
    for (const auto &pc : dflash_->ring_pieces(c, slot)) v.push_back(pc);
  return v;
}

std::vector<size_t> Model::snap_piece_bytes(int c) const {
  std::vector<size_t> v;
  for (const auto &p : snap_pieces(c, 0)) v.push_back(p.second);
  return v;
}

size_t Model::snap_bytes(int c) const {
  size_t b = 0;
  for (const auto &p : snap_pieces(c, 0)) b += p.second;
  return b;
}

void Model::snap_copy(int slot, SnapRec &r, bool save) {
  for (int c = 0; c < nc_; ++c) {
    auto &q = fab_.compute(c);
    char *stage = static_cast<char *>(snap_stage_[c]);
    const auto pieces = snap_pieces(c, slot);
    size_t total = 0;
    for (const auto &p : pieces) total += p.second;
    if (!save) q.memcpy(stage, r.host[c], total);
    size_t off = 0;
    for (const auto &[dev, n] : pieces) {
      if (save) q.memcpy(stage + off, dev, n);
      else q.memcpy(dev, stage + off, n);
      off += n;
    }
    if (save) q.memcpy(r.host[c], stage, total);
  }
}

int Model::snap_save(int slot) {
  if (snap_cap_ <= 0) return -1;
  if (slot < 0 || slot >= slots()) throw std::invalid_argument("qwen38 model: no such slot");
  if (pf_run_slot() == slot) throw std::logic_error("qwen38 model: a snapshot with the slot's prefill call pending");
  const Seq &sq = seqs_[static_cast<size_t>(slot)];
  if (!sq.pf_committed || sq.pending) throw std::logic_error("qwen38 model: a snapshot is taken right after a prefill call");
  const bool pct = [] { const char *e = std::getenv("XEN_PC_TRACE"); return e && *e == '1'; }();
  int id = -1;
  for (size_t i = 0; i < snaps_.size() && id < 0; ++i)
    if (!snaps_[i].used) id = static_cast<int>(i);
  if (id < 0) {
    if (static_cast<int>(snaps_.size()) >= snap_cap_) return -1;
    snaps_.emplace_back();
    id = static_cast<int>(snaps_.size()) - 1;
    if (pct) std::fprintf(stderr, "SNAPSAVE slot %d grew the store to %d ids\n", slot, static_cast<int>(snaps_.size()));
  }
  SnapRec &r = snaps_[static_cast<size_t>(id)];
  for (int c = 0; c < nc_; ++c)
    if (!r.host[c]) r.host[c] = fab_.alloc_host(std::max<size_t>(snap_bytes(c), 4096));
  r.pos = sq.pos;
  r.gdn_par = sq.gdn_par;
  r.ple = sq.ple_ctx;
  r.used = true;
  if (pct) std::fprintf(stderr, "SNAPSAVE slot %d -> id %d pos %lld (store %d ids)\n", slot, id, static_cast<long long>(r.pos), static_cast<int>(snaps_.size()));
  if (dflash_) dflash_->settle(slot, std::numeric_limits<int64_t>::max());  // its deferred rows into the rings first
  snap_copy(slot, r, true);
  return id;
}

int Model::snap_prepin(int n) {
  int done = 0;
  while (done < n && static_cast<int>(snaps_.size()) < snap_cap_) {
    snaps_.emplace_back();  // unused: snap_save picks it before growing the store
    for (int c = 0; c < nc_; ++c) snaps_.back().host[c] = fab_.alloc_host(std::max<size_t>(snap_bytes(c), 4096));
    ++done;
  }
  if (done > 0 && static_cast<int>(snaps_.size()) == snap_cap_)
    std::fprintf(stderr, "qwen38 model: prefix cache: all %d snapshot ids' host buffers pinned ahead of use\n", snap_cap_);
  return done;
}

void Model::snap_sync() {
  for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();
}

void Model::snap_free(int id) {
  if (id < 0 || id >= static_cast<int>(snaps_.size()) || !snaps_[static_cast<size_t>(id)].used) throw std::invalid_argument("qwen38 model: no such snapshot");
  snaps_[static_cast<size_t>(id)].used = false;  // the host buffers stay; a later copy queues behind the last (in order)
}

void Model::snap_restore(int slot, int id) {
  if (slot < 0 || slot >= slots()) throw std::invalid_argument("qwen38 model: no such slot");
  if (id < 0 || id >= static_cast<int>(snaps_.size()) || !snaps_[static_cast<size_t>(id)].used) throw std::invalid_argument("qwen38 model: no such snapshot");
  if (pf_run_slot() == slot) throw std::logic_error("qwen38 model: a restore with the slot's prefill call pending");
  SnapRec &r = snaps_[static_cast<size_t>(id)];
  if (kv_held_tokens(slot) < r.pos) throw std::logic_error("qwen38 model: the slot holds no pages for the snapshot's positions");
  if (dflash_) dflash_->settle(slot, 0);  // the slot's deferred rows are replaced with the snapshot's rings
  snap_copy(slot, r, false);
  Seq &sq = seqs_[static_cast<size_t>(slot)];
  sq = Seq{};
  sq.pos = r.pos;
  sq.gdn_par = r.gdn_par;
  sq.last_rows = 0;  // as after a prefill call: committed, nothing to replay
  sq.ple_ctx = r.ple;
  sq.pf_committed = true;
  batch_slots_.erase(std::remove(batch_slots_.begin(), batch_slots_.end(), slot), batch_slots_.end());
}

void Model::snap_fork(int dst, int src, int id) {
  if (dst == src || dst < 0 || src < 0 || dst >= slots() || src >= slots()) throw std::invalid_argument("qwen38 model: a fork takes two slots");
  if (id < 0 || id >= static_cast<int>(snaps_.size()) || !snaps_[static_cast<size_t>(id)].used) throw std::invalid_argument("qwen38 model: no such snapshot");
  const int n = model::KvPages::pages_for(snaps_[static_cast<size_t>(id)].pos);
  if (kvp_.held_pages(src) < n || kvp_.held_pages(dst) < n) throw std::logic_error("qwen38 model: a fork needs the snapshot's pages on both slots");
  std::vector<int32_t> pairs(2 * static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    pairs[static_cast<size_t>(i)] = kvp_.page(src, i);
    pairs[static_cast<size_t>(n + i)] = kvp_.page(dst, i);
  }
  for (int c = 0; c < nc_; ++c) {
    auto &q = fab_.compute(c);
    q.memcpy(fork_pages_[c], pairs.data(), pairs.size() * 4);
    std::vector<k::KvPageBuf> bufs;
    auto add = [&](const k::QsaCache &cc, const k::QsaShape &sh) {
      bufs.push_back({cc.k, k::qsa_page_kv_bytes(sh)});
      bufs.push_back({cc.v, k::qsa_page_v_bytes(sh)});
      if (cc.ks && cc.vs) {
        bufs.push_back({cc.ks, k::qsa_page_scale_floats(sh) * 4});
        bufs.push_back({cc.vs, k::qsa_page_vscale_floats(sh) * 4});
      }
      bufs.push_back({cc.kc, k::qsa_page_kc_elems() * 2});
    };
    const Card &K = *cards_[static_cast<size_t>(c)];
    for (int l : qs_.layers) add(K.qc[static_cast<size_t>(l)], qs_.shape_of(c));
    if (mtp_) {  // the drafter's layer: the target's shape with its own KV dtype (drafter.cpp)
      auto sh = qs_.shape_of(c);
      sh.kv = opt_.mtp_kv;
      add(mtp_->cards[static_cast<size_t>(c)].qc, sh);
    }
    k::kv_copy_pages(q, bufs, fork_pages_[c], fork_pages_[c] + n, n);
  }
  snap_restore(dst, id);
  for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();  // the pairs live in pageable host memory here
}

bool Model::live_ok(int slot) const {
  if (slot < 0 || slot >= slots()) throw std::invalid_argument("qwen38 model: no such slot");
  const Seq &sq = seqs_[static_cast<size_t>(slot)];
  if (sq.broken || sq.pending || pf_run_slot() == slot) return false;
  // a prefill call replays up to 4 pending GDN rows (kSegAccept), as a chain (not a tree's branch)
  return sq.pf_committed || (sq.last_rows <= k::kGdnRows && (sq.last_rows == 0 || sq.path == 0));
}

void Model::kv_trim(int slot, int64_t tokens) {
  if (slot < 0 || slot >= slots()) throw std::invalid_argument("qwen38 model: no such slot");
  kvp_.trim(slot, tokens);
  kv_upload();
}

}  // namespace xenithium::arch::qwen38
