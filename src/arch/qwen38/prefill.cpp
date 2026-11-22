// Xenithium · qwen38 prefill: a prompt's tokens through the model LAYER-MAJOR, in chunks.
//
// Within a chunk of m rows (positions pos .. pos + m - 1), layer l runs on every row before layer
// l + 1 starts. Causality holds without any per-row ordering trick: every state a layer keeps is
// per layer (GDN S / window / pending log, the QSA caches, the PLE ring), and within a layer the
// rows still run in position order. The chunk's multi-stream state and the block outputs live in
// chunk-sized buffers (Card::Pf); a 4-row slice of them is a view at row offset 4 s.
//
// The components run on many-row kernels; against the same tokens in 4-row step()s the numerics differ by
// fp32 summation order:
//   HC mixers: hc_mixer_rows (pf_dense) over the whole chunk.
//   MoE: moe_rows over the whole chunk (expert TP; EP2 keeps the decode kernels per slice).
//   GDN: gdn_rows over the whole chunk (commits every row: the next decode step replays none).
//   QSA: qsa_rows: projections + two-card sum over the chunk (stage A), prep over the chunk and
//        score / select / attn / merge per 64-row batch (stage B; bit-identical to stage A).
//   PLE block: ple_rows over the whole chunk.
#include <algorithm>
#include <functional>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "xenithium/arch/common/vision.hpp"
#include "xenithium/kernels/head/head.hpp"
#include "xenithium/kernels/head/logprob.hpp"
#include "xenithium/kernels/gemm/pf_dense.hpp"
#include "model_internal.hpp"
#include "xenithium/util/env.hpp"

namespace xenithium::arch::qwen38 {

namespace k = xenithium::kernels;
using bf16 = sycl::ext::oneapi::bfloat16;

void Model::prefill(const int32_t *tokens, int n, const std::function<void(int, int)> &on_rows, int32_t next_after) {
  if (pf_run_) throw std::logic_error("qwen38 model: prefill while a split prefill call is pending");
  const Seq &sq = seqs_[static_cast<size_t>(slot_)];
  if (sq.pending) throw std::logic_error("qwen38 model: prefill while a verify waits for its commit");
  if (!sq.pf_committed && sq.last_rows > k::kGdnRows)  // the many-row GDN replays up to 4 pending rows (live_ok)
    throw std::logic_error("qwen38 model: a prefill after a step that kept more than 4 rows (step one row first)");
  if (!sq.pf_committed && sq.last_rows > 0 && sq.path != 0)  // the many-row GDN replays a chain
    throw std::logic_error("qwen38 model: a prefill after a tree step that kept a branch (step one row first)");
  if (n < 1) throw std::invalid_argument("qwen38 model: prefill of no tokens");
  if (sq.pos + n > cap_) throw std::runtime_error("qwen38 model: the prompt outgrew the KV capacity");
  for (int r0 = 0; r0 < n; r0 += opt_.prefill_chunk) {
    const int m = std::min(opt_.prefill_chunk, n - r0);
    pf_start(tokens + r0, m, r0, on_rows, r0 + m < n ? tokens[r0 + m] : next_after);
    pf_layers(cfg_.layers);
    pf_finish();
  }
}

// See model.hpp. The request rides pf_lp_ through the chunks of one prefill() call.
bool Model::pf_logprob_rows() const {
  for (int c = 0; c < nc_; ++c)
    if (cards_[static_cast<size_t>(c)]->pf.lp_rows <= 0) return false;
  return true;
}

void Model::prefill_logprobs(const int32_t *tokens, int n, int32_t next_after, const int32_t *targets, int top, std::vector<float> &rec, std::vector<float> &xt) {
  if (!pf_logprob_rows()) throw std::logic_error("qwen38 model: prefill_logprobs needs the many-row head (expert TP)");
  if (top < 0 || top > k::kLpMaxTop) throw std::invalid_argument("qwen38 model: prefill_logprobs top");
  rec.assign(static_cast<size_t>(nc_) * n * k::kLpRec, 0.f);  // a record per card
  xt.assign(static_cast<size_t>(n), std::numeric_limits<float>::infinity());
  const PfLp lp{targets, n, top, rec.data(), xt.data()};
  pf_lp_ = &lp;
  try {
    prefill(tokens, n, {}, next_after);
  } catch (...) {
    pf_lp_ = nullptr;
    throw;
  }
  pf_lp_ = nullptr;
}

void Model::prefill_begin(const int32_t *tokens, int n, int32_t next_after) {
  if (pf_run_) throw std::logic_error("qwen38 model: prefill_begin while a split prefill call is pending");
  const Seq &sq = seqs_[static_cast<size_t>(slot_)];
  if (sq.pending) throw std::logic_error("qwen38 model: prefill while a verify waits for its commit");
  if (!sq.pf_committed && sq.last_rows > k::kGdnRows)  // the many-row GDN replays up to 4 pending rows (live_ok)
    throw std::logic_error("qwen38 model: a prefill after a step that kept more than 4 rows (step one row first)");
  if (!sq.pf_committed && sq.last_rows > 0 && sq.path != 0)  // the many-row GDN replays a chain
    throw std::logic_error("qwen38 model: a prefill after a tree step that kept a branch (step one row first)");
  if (n < 1 || n > opt_.prefill_chunk) throw std::invalid_argument("qwen38 model: prefill_begin takes 1..prefill_chunk tokens");
  if (sq.pos + n > cap_) throw std::runtime_error("qwen38 model: the prompt outgrew the KV capacity");
  pf_start(tokens, n, 0, {}, next_after);
}

bool Model::prefill_advance(int layers) {
  if (!pf_run_) throw std::logic_error("qwen38 model: prefill_advance without a pending prefill call");
  pf_layers(std::max(1, layers));
  if (pf_run_->l < cfg_.layers) {
    // the part's device time lands in its wall time (the engine's cost model times the parts); the
    // last part waits in the head
    for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();
    return false;
  }
  pf_finish();
  return true;
}

void Model::prefill_abort() {
  if (pf_run_) seqs_[static_cast<size_t>(pf_run_->slot)].broken = true;  // never resumed live
  pf_run_.reset();
}

std::vector<int32_t> Model::draft_after_prefill(int32_t first, int depth, int seqs) {
  if (!dflash_ && !mtp_) throw std::logic_error("qwen38 model: no drafter loaded (ModelOptions::mtp_dir / dflash_dir)");
  const bool mtp_drafting = mtp_drafts(seqs);
  // A block drafter drafts at most block - 1 tokens per pass (DFlash 2: block 8 -> 7). The caller's cap is the step's
  // rows (XEN_SEG_ROWS - 1, 15 at the 16-row default), which is a DSpine b16 setting: asking a block-8 drafter for 15
  // drafts overruns its block layout, and an 8K prompt then resets the engine (CCS job timeouts + a CAT error, all four
  // cards, "UR_RESULT_ERROR_DEVICE_LOST"). Clamp to whichever is smaller.
  if (dflash_ && !mtp_drafting) depth = std::min(depth, dflash_->max_depth());
  if (mtp_) {
    Mtp &M = *mtp_;
    if (M.last_rows < 1) throw std::logic_error("qwen38 model: draft_after_prefill without a drafted prefill");
    const int last = M.last_rows - 1;
    // pass 1 ran with the pick; the emitted token differs (sampled): its last row again, with the token (the drafter KV
    // must hold it whichever drafter drafts; KV only when DFlash drafts)
    if (M.ids_host[last] != first) {
      for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();  // the pinned source: the prefill's pending copy reads it
      M.ids_host[last] = first;
      for (int c = 0; c < nc_; ++c) fab_.compute(c).memcpy(M.cards[static_cast<size_t>(c)].ids, M.ids_host, static_cast<size_t>(M.last_rows) * 4);
      mtp_pass(0, M.last_rows, per_card([&](int c) -> const bf16 * { return head_->residual(c); }),
               per_card([&](int c) -> const int32_t * { return M.cards[static_cast<size_t>(c)].ids; }), M.p1_segs, M.p1_seg_rows, !mtp_drafting);
    }
    if (mtp_drafting) return draft_chain(last, depth);  // the prompt is in the DFlash context too (the prefill put it there)
  }
  // the prompt is in the drafter's context; the block starts at the position after it
  const Seq &sq = seqs_[static_cast<size_t>(slot_)];
  if (!sq.pf_committed) throw std::logic_error("qwen38 model: draft_after_prefill follows a prefill");
  return dflash_drafts({arch::DFlash::Seg{slot_, sq.pos, sq.pos, first}}, 0, depth, seqs)[0];  // the step's sequences
}

namespace {
const char *pf_prof_env() {
  static const char *e = util::env_value("XEN_PF_PROFILE");  // unset, empty or "0" = off
  return e;
}
// XEN_PF_PROFILE=1: both cards drained after each phase, host wall per phase summed over the
// layers and printed per chunk (a diagnostic: the waits serialise the cards' queues).
// XEN_PF_PROFILE=2: no waits; a barrier per phase on card 0's in-order queue, the
// device time between consecutive barriers summed per phase (what the chunk costs the card).
// XEN_PF_PROFILE=3: as 2, but a one-item stamp kernel per phase writes card 0's
// GT-clock counter instead of a barrier, and the split sums' wait kernels write their own spans; no
// barriers and no per-kernel events, so the side queue's copies keep their overlap. Cycles -> ms by a
// calibration over the call (the first and last stamps' events on each card).
bool pf_mode(const char *v) { return pf_prof_env() != nullptr && std::string(pf_prof_env()) == v; }
bool pf_prof() { return pf_prof_env() != nullptr && !pf_mode("2") && !pf_mode("3"); }
bool pf_dprof() { return pf_mode("2"); }
bool pf_sprof() { return pf_mode("3"); }
bool pf_eprof() { return pf_prof() || pf_dprof(); }  // the modes that collect events and barriers
// In the event modes on profiling queues, card 0's GDN / QSA launches go into the call's stage traces
// (PfRun::gdn_trace / qsa_trace), printed per chunk by pf_finish
template <class Trace>
Trace *stage_trace(const Fabric &fab, int c, Trace &trace) {
  return pf_eprof() && fab.profiling() && c == 0 ? &trace : nullptr;
}
constexpr int kPfStampWords = 32768;  // per card per call (mode 3)
constexpr int kPfSpanShared = 3;      // PfRun::SplitSpan::kind of a shared rows' wait (0 - 2: the GDN / QSA / MoE sums)
}  // namespace

void Model::pf_mark(int i) {
  PfRun &R = *pf_run_;
  if (pf_dprof()) R.dmarks.push_back({i, fab_.compute(0).ext_oneapi_submit_barrier()});
  if (pf_sprof() && R.stamp_n[0] < kPfStampWords) {
    k::stamp(fab_.compute(0), pf_stamps_[0] + R.stamp_n[0]);
    R.smarks.push_back({i, R.stamp_n[0]++});
    {  // host submit time per phase (no waits; a phase whose host time exceeds its device time starves the cards)
      const auto now = std::chrono::steady_clock::now();
      if (R.smarks.size() > 1) R.ph[i] += std::chrono::duration<double, std::milli>(now - R.tick).count();
      R.tick = now;
    }
  }
  if (!pf_prof()) return;
  for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();
  const auto now = std::chrono::steady_clock::now();
  R.ph[i] += std::chrono::duration<double, std::milli>(now - R.tick).count();
  R.tick = now;
}

// trace (diagnostic): card 0's rows after a module
void Model::pf_tr(const std::string &name, const bf16 *dev, int width) {
  if (!pf_trace_) return;
  if (!pf_trace_->points.empty()) {  // the point = the name after "L<l>."
    const size_t dot = name.size() > 1 && name[0] == 'L' ? name.find('.') : std::string::npos;
    const std::string point = dot == std::string::npos ? name : name.substr(dot + 1);
    if (std::find(pf_trace_->points.begin(), pf_trace_->points.end(), point) == pf_trace_->points.end()) return;
  }
  for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();
  const int rows = std::min(pf_trace_->rows, pf_run_->m);
  std::vector<uint16_t> h(static_cast<size_t>(rows) * width);
  fab_.compute(0).memcpy(h.data(), dev, h.size() * 2).wait();
  pf_trace_->names.push_back(name);
  pf_trace_->data.push_back(std::move(h));
  pf_trace_->widths.push_back(width);
}

void Model::pf_start(const int32_t *tokens, int m, int row_base, const std::function<void(int, int)> &on_rows, int32_t next_after) {
  kv_need(slot_, seqs_[static_cast<size_t>(slot_)].pos + m);  // the chunk's KV pages (target + drafter)
  pf_run_ = std::make_unique<PfRun>();
  PfRun &R = *pf_run_;
  R.tokens.assign(tokens, tokens + m);
  R.m = m;
  R.row_base = row_base;
  R.slices = (m + 3) / 4;
  R.next_after = next_after;
  R.on_rows = on_rows;
  // The chunk runs on the current slot: its state pointers are chosen here on the host (a prefill is
  // never recorded), and the slice records name the slot for the kernels that read it on the device.
  R.slot = slot_;
  Seq &sq = seqs_[static_cast<size_t>(R.slot)];
  R.pos0 = sq.pos;
  R.gp0 = sq.gdn_par;  // the slot's GDN parity: its previous step wrote window / log [gp0 ^ 1]
  if (const int lanes = pf_lane_count(m); lanes > 0) {  // the rows by owner, in lanes
    // A lane ends where a call may end and keep every bit: at a position that is a multiple of kPfLaneAlign, at least
    // kPfOwnerMinRows rows from the lane's start and from the call's end (a live resume's call starts at any position,
    // so the cuts are placed by position, not by row).
    std::vector<int> cuts{0};
    for (int j = 1; j < lanes; ++j) {
      const int64_t pos = (R.pos0 + static_cast<int64_t>(m) * j / lanes) / kPfLaneAlign * kPfLaneAlign;
      const int r = static_cast<int>(pos - R.pos0);
      if (r - cuts.back() >= kPfOwnerMinRows && m - r >= kPfOwnerMinRows) cuts.push_back(r);
    }
    cuts.push_back(m);
    int w0 = 0;
    for (size_t j = 0; j + 1 < cuts.size(); ++j) {
      PfRun::Lane ln;
      ln.r0 = cuts[j];
      ln.m = cuts[j + 1] - cuts[j];
      ln.w0 = w0;
      ln.gp0 = R.gp0 ^ static_cast<int>(j & 1);
      w0 += (ln.m + 3) / 4;
      R.lanes.push_back(ln);
    }
    R.moe_lanes = static_cast<int>(m < kPfMoeTwoCallRows ? R.lanes.size() : (R.lanes.size() + 1) / 2);
  }
  const int32_t *tk = R.tokens.data();

  // PLE rows of the whole chunk (host): the n-gram context advances past the chunk (committed).
  // A calibration layer that is not the PLE layer skips them (an NVMe gather per sample).
  if (!pf_skip_ple_) {
    std::vector<int64_t> toks(static_cast<size_t>(m));
    const int32_t vocab = cfg_.vocab;  // an image (video) position's PLE n-grams see the image (video) token
    for (int r = 0; r < m; ++r) toks[static_cast<size_t>(r)] = vis_ && tk[r] >= vocab ? vis_->pad_token(tk[r]) : tk[r];  // video: <|video_pad|>
    std::vector<int64_t> ids(static_cast<size_t>(m) * ple_tab_->heads());
    std::vector<uint8_t> codes(ids.size() * ple_tab_->row_bytes());
    ple_tab_->ids(toks.data(), m, sq.ple_ctx, ids.data());
    ple_tab_->read(ids.data(), m, codes.data());
    ple_tab_->dequant(codes.data(), m, pf_ple_);
  }
  // Slice records (rows.hpp): position of the slice's row 0, the GDN accept count (the previous
  // slice's rows), the slot.
  pf_words_.assign(static_cast<size_t>(R.slices) * k::kSegRec, 0);
  for (int s = 0; s < R.slices; ++s) {
    int32_t *w = pf_words_.data() + static_cast<size_t>(s) * k::kSegRec;
    w[k::kSegPos] = static_cast<int32_t>(R.pos0 + 4 * s);
    w[k::kSegAccept] = s == 0 ? sq.last_rows : 4;
    w[k::kSegParity] = sq.gdn_par;
    w[k::kSegSlot] = R.slot;
  }
  // Each lane's slice records, from the lane's own first row (the lanes' GDN / QSA calls read these; the call's records
  // above stay as they are for the drafter's pass): a later lane starts as a next call would, with nothing pending to
  // replay and the GDN parity the lanes before it flipped.
  const size_t lane_words_at = pf_words_.size();
  for (const PfRun::Lane &ln : R.lanes)
    for (int s = 0; s < (ln.m + 3) / 4; ++s) {
      pf_words_.resize(pf_words_.size() + k::kSegRec, 0);
      int32_t *w = pf_words_.data() + pf_words_.size() - k::kSegRec;
      w[k::kSegPos] = static_cast<int32_t>(R.pos0 + ln.r0 + 4 * s);
      w[k::kSegAccept] = s > 0 ? 4 : ln.r0 == 0 ? sq.last_rows : 0;
      w[k::kSegParity] = ln.gp0;
      w[k::kSegSlot] = R.slot;
    }
  R.cs.resize(static_cast<size_t>(nc_));  // the chunk's HC state per card, indices as the head's
  const int32_t *dev_ids = tk;
  if (vis_) {  // image ids -> image rows (the images this chunk or the drafter's next token names encoded first)
    if (next_after >= cfg_.vocab) vis_map(&next_after, 1, vis_tmp_);
    dev_ids = vis_map(tk, m, vis_ids_);
  }
  for (int c = 0; c < nc_; ++c) {
    auto &q = fab_.compute(c);
    Card &K = *cards_[static_cast<size_t>(c)];
    q.memcpy(K.pf.words, pf_words_.data(), lane_words_at * 4);
    if (pf_words_.size() > lane_words_at) q.memcpy(K.pf.lane_words, pf_words_.data() + lane_words_at, (pf_words_.size() - lane_words_at) * 4);
    q.memcpy(K.pf.ids, dev_ids, static_cast<size_t>(m) * 4);
    const k::HcState &hst = head_->state(c);
    k::HcState &st = R.cs[static_cast<size_t>(c)];
    st = hst;
    R.start_cur[c] = hst.cur;
    R.start_lcur[c] = hst.lcur;
    for (int i = 0; i < 2; ++i) {
      st.res[i] = K.pf.res[i];
      st.lora[i] = K.pf.lora[i];
    }
    embed_rows(q, c, K.pf.ids, m, hs_.shape.streams, st.residual());
  }
  if (pf_dprof()) R.dmarks.push_back({-1, fab_.compute(0).ext_oneapi_submit_barrier()});
  if (pf_sprof()) {
    for (int c = 0; c < nc_; ++c) {
      if (!pf_stamps_[c]) pf_stamps_[c] = static_cast<uint64_t *>(fab_.alloc(static_cast<size_t>(c), kPfStampWords * 8));
      fab_.compute(c).memset(pf_stamps_[c], 0, kPfStampWords * 8);  // a wait that never ran reads 0
      R.cal[c][0] = k::stamp(fab_.compute(c), pf_stamps_[c] + R.stamp_n[c]);
      R.cal_at[c][0] = R.stamp_n[c]++;
    }
    R.smarks.push_back({-1, R.cal_at[0][0]});
  }
  R.t_sub0 = std::chrono::steady_clock::now();
  R.tick = R.t_sub0;
  pf_mark(4);
  R.ph[4] = 0;
  pf_tr("embed.res", R.cs[0].residual(), hs_.shape.width());
}

// Finish a chunk's two-card sum after both cards' split-mode kernels. add: the MoE's shared
// term (MoeRowsScratch sy x sg).
template <class PartOf, class OutOf>
// sent: the cards' copies already sent (pair_split_send by the producer: their events).
void Model::pf_split_sum(PartOf part_of, OutOf out_of, bool add, int kind, const sycl::event *sent) {
  const int m = pf_run_->m;
  if (nc_ < 2) throw std::logic_error("qwen38 model: the split sum needs two or four cards");
  sycl::queue *q[kMaxCards], *side[kMaxCards];
  const float *part[kMaxCards];
  bf16 *out[kMaxCards];
  k::PairSplit sp[kMaxCards];
  k::PairSplitAdd ad[kMaxCards];
  for (int c = 0; c < nc_; ++c) {
    Card &K = *cards_[static_cast<size_t>(c)];
    q[c] = &fab_.compute(c);
    side[c] = sent ? &*K.pf.side : nullptr;  // under kPairSplitSendMinRows: (compute queue)
    part[c] = part_of(K);
    out[c] = out_of(K);
    sp[c] = K.pf.split;
    if (add) sp[c].bf16 = false;  // the MoE sums stay fp32 (its short calls' wire has no bf16 form: equal bits per call size)
    ad[c] = k::PairSplitAdd{K.pf.moe.sy, K.pf.moe.sg};
  }
  sycl::event prev[kMaxCards];
  if (pf_eprof())
    for (int c = 0; c < nc_; ++c) prev[c] = q[c]->ext_oneapi_submit_barrier();
  PfRun &R = *pf_run_;
  const bool spans = pf_sprof() && R.stamp_n[0] + 4 <= kPfStampWords && R.stamp_n[nc_ - 1] + 4 <= kPfStampWords;
  if (spans) {
    PfRun::SplitSpan s;
    s.kind = kind;
    for (int c = 0; c < nc_; ++c) {
      s.at[c] = R.stamp_n[c];
      sp[c].wait_span = pf_stamps_[c] + R.stamp_n[c];
      R.stamp_n[c] += 4;
    }
    R.split_span.push_back(s);
  }
  pf_split_seq_ = k::pair_split_next_seq(pf_split_seq_);
  k::PairSplitEvents ev;
  // With the producers' sends the consumer mixer takes the peer's rows late (pf_mix_rows), so the back copy runs beside
  // the card's own rows (from kPairSplitDeferMinRows).
  const bool defer = sent && m >= k::kPairSplitDeferMinRows;
  if (sent) {
    ev = k::pair_split_finish(q, side, m, cfg_.hidden, part, out, sp, pf_split_seq_, add ? ad : nullptr, defer ? k::SplitRows::deferred : k::SplitRows::all);
    for (int c = 0; c < nc_; ++c) ev.copy[c] = sent[c];
  } else {
    ev = k::pair_split_rows(q, m, cfg_.hidden, part, out, sp, pf_split_seq_, add ? ad : nullptr, side);
  }
  if (pf_eprof()) {
    PfRun::SplitRec rec{kind, ev, {}};
    for (int c = 0; c < nc_; ++c) rec.prev[c] = prev[c];
    pf_run_->split_ev.push_back(rec);
  }
  if (defer) {
    pf_run_->pend_seq = pf_split_seq_;
    pf_run_->pend_rec = pf_eprof() ? static_cast<int>(pf_run_->split_ev.size()) - 1 : -1;
    for (int c = 0; c < nc_; ++c) pf_run_->pend_span[c] = spans ? sp[c].wait_span : nullptr;
  }
}

void Model::pf_split_wait() {
  PfRun &R = *pf_run_;
  if (!R.pend_seq) return;
  for (int c = 0; c < nc_; ++c) {
    k::PairSplit sp = cards_[static_cast<size_t>(c)]->pf.split;
    sp.wait_span = R.pend_span[c];
    const sycl::event e = k::pair_split_wait(fab_.compute(c), sp, R.pend_seq);
    if (R.pend_rec >= 0) R.split_ev[static_cast<size_t>(R.pend_rec)].ev.done[c] = e;
  }
  R.pend_seq = 0;
  R.pend_rec = -1;
}

void Model::pf_mix_rows(int c, const bf16 *block, const k::HcWeights &w, bf16 *out) {
  PfRun &R = *pf_run_;
  auto &q = fab_.compute(c);
  Card &K = *cards_[static_cast<size_t>(c)];
  k::HcState &st = R.cs[static_cast<size_t>(c)];
  if (!R.pend_seq || !block) {
    k::hc_mixer_rows(q, hs_.shape, R.m, st, block, w, K.pf.hc, out);
    return;
  }
  // block's own rows are this card's sum (ordered on q); the peer's arrive by its back copy
  int lo = 0, hi = 0;
  k::pair_split_owner_rows(c, R.m, &lo, &hi, nc_);
  k::hc_mixer_rows_part(q, hs_.shape, lo, hi, st, block, w, K.pf.hc, out);
  const sycl::event e = k::pair_split_wait(q, K.pf.split, R.pend_seq);
  if (R.pend_rec >= 0) R.split_ev[static_cast<size_t>(R.pend_rec)].ev.done[c] = e;
  if (lo > 0) k::hc_mixer_rows_part(q, hs_.shape, 0, lo, st, block, w, K.pf.hc, out);
  if (hi < R.m) k::hc_mixer_rows_part(q, hs_.shape, hi, R.m, st, block, w, K.pf.hc, out);
  k::hc_mixer_rows_flip(st, w);
  if (c == nc_ - 1) {  // every card waited
    R.pend_seq = 0;
    R.pend_rec = -1;
  }
}

// ---- a prefill call's rows by owner ---------------------------------------------------------------------------------
// A row belongs to one card (kernels::pair_split_owner_rows inside its lane): only that card keeps the row's streams
// (res, lora) and runs its mixers, instead of every card holding every row's streams and running every mixer on every
// row. A mixer's output is a module's input, which every card needs for its partial, so the owner copies it to the
// other cards (pf_lane_share); a module's sum is read by the owner's next mixer only, so the partials go to the owner
// and nothing comes back (SplitRows::owner). After the last layer the owners share the chunk's state (pf_owner_state)
// and the head and the drafters find every row on every card. Each row runs the same kernels on the same values: the
// same bits.
//
// Lanes: the copies take as long as they take on the cards' links, so they must not stand between two kernels. The
// call's rows therefore go through each mixer and each GDN / QSA as several calls, the lanes (a lane's rows continue
// the previous lane's exactly as a next prefill call would), and a card's queue alternates the lanes: while one lane's
// inputs or partials are on the link, the card computes the others. A lane's sum is queued right before the mixer that
// reads it. The MoE is cheaper per row the more rows a call has (rows per expert tile), so one MoE call spans several
// lanes (pf_lanes_moe) and sends each lane's partial as its own. The lane counts by call size: model.hpp kPf*.
namespace {
struct RowRange {
  int lo = 0, hi = 0;
  int rows() const { return hi - lo; }
};
// Card `card`'s rows of the lane [r0, r0 + m), as rows of the call.
RowRange lane_own_rows(int card, int cards, int r0, int m) {
  int lo = 0, hi = 0;
  k::pair_split_owner_rows(card, m, &lo, &hi, cards);
  return {r0 + lo, r0 + hi};
}
}  // namespace

int Model::pf_lane_count(int m) const {
  // the diagnostics that read a card's whole chunk state mid-call keep every row on every card
  const bool whole_state = pf_trace_ || calib_cap_ || calib_hook_ || pf_eprof();
  // Four cards only: two cards would run this path with the same bits, but a mixer row is saved once there, not three
  // times, against the same copies.
  if (nc_ != kMaxCards || !ms_.shape.tp || whole_state || m < kPfOwnerMinRows) return 0;
  return m < kPfTwoLaneRows ? 1 : m < kPfFourLaneRows ? 2 : kPfLanesMax;
}

template <class BufOf>
Model::PfShared Model::pf_shared(BufOf buf_of, size_t row_bytes) {
  PfShared set;
  set.row_bytes = row_bytes;
  for (int c = 0; c < nc_; ++c) set.bufs[c] = buf_of(*cards_[static_cast<size_t>(c)]);
  return set;
}

void Model::pf_lane_share(int lane, std::span<const PfShared> sets, std::span<const sycl::event> written) {
  PfRun::Lane &ln = pf_run_->lanes[static_cast<size_t>(lane)];
  std::vector<PfShared> at_lane(sets.begin(), sets.end());  // the lane's rows of each buffer
  std::vector<k::SplitShare> shares;
  for (PfShared &set : at_lane) {
    for (int c = 0; c < nc_; ++c) set.bufs[c] = static_cast<std::byte *>(set.bufs[c]) + static_cast<size_t>(ln.r0) * set.row_bytes;
    shares.push_back({set.bufs, set.row_bytes});
  }
  ln.share_seq = pf_split_seq_ = k::pair_split_next_seq(pf_split_seq_);
  for (int c = 0; c < nc_; ++c) {
    Card &K = *cards_[static_cast<size_t>(c)];
    const sycl::event *after = written.empty() ? nullptr : &written[static_cast<size_t>(c)];
    k::pair_split_share(fab_.compute(c), &*K.pf.side, c, ln.m, shares, K.pf.lane[lane], ln.share_seq, after);
  }
}

void Model::pf_lane_wait(int lane) {
  PfRun &R = *pf_run_;
  const PfRun::Lane &ln = R.lanes[static_cast<size_t>(lane)];
  const bool spans = pf_sprof() && R.stamp_n[0] + 4 <= kPfStampWords && R.stamp_n[nc_ - 1] + 4 <= kPfStampWords;
  PfRun::SplitSpan span;
  span.kind = kPfSpanShared;
  for (int c = 0; c < nc_; ++c) {
    k::PairSplit sp = cards_[static_cast<size_t>(c)]->pf.lane[lane];
    if (spans) {
      span.at[c] = R.stamp_n[c];
      sp.wait_span = pf_stamps_[c] + R.stamp_n[c];
      R.stamp_n[c] += 4;
    }
    k::pair_split_wait(fab_.compute(c), sp, ln.share_seq);
  }
  if (spans) R.split_span.push_back(span);
}

void Model::pf_lane_sum(int lane) {
  PfRun &R = *pf_run_;
  PfRun::Lane &ln = R.lanes[static_cast<size_t>(lane)];
  if (ln.pend_kind < 0) return;
  const bool moe = ln.pend_kind == 2;
  const size_t at = static_cast<size_t>(ln.r0) * cfg_.hidden;
  sycl::queue *q[kMaxCards], *side[kMaxCards];
  const float *part[kMaxCards];
  bf16 *out[kMaxCards];
  k::PairSplit sp[kMaxCards];
  k::PairSplitAdd add[kMaxCards];
  const bool spans = pf_sprof() && R.stamp_n[0] + 4 <= kPfStampWords && R.stamp_n[nc_ - 1] + 4 <= kPfStampWords;
  PfRun::SplitSpan span;
  span.kind = ln.pend_kind;
  for (int c = 0; c < nc_; ++c) {
    Card &K = *cards_[static_cast<size_t>(c)];
    q[c] = &fab_.compute(c);
    side[c] = &*K.pf.side;
    part[c] = (moe ? K.pf.moe.part : ln.pend_kind == 1 ? K.pf.qsa.partial : K.pf.gdn.partial) + at;
    out[c] = (moe ? K.pf.ym : K.pf.ya) + at;
    sp[c] = K.pf.lane[lane];
    if (moe) {  // the MoE sums stay fp32; its shared expert's term joins the sum
      sp[c].bf16 = false;
      add[c] = k::PairSplitAdd{K.pf.moe.sy + at, K.pf.moe.sg + ln.r0};
    }
    if (spans) {
      span.at[c] = R.stamp_n[c];
      sp[c].wait_span = pf_stamps_[c] + R.stamp_n[c];
      R.stamp_n[c] += 4;
    }
  }
  if (spans) R.split_span.push_back(span);
  k::pair_split_finish(q, side, ln.m, cfg_.hidden, part, out, sp, ln.pend_seq, moe ? add : nullptr, k::SplitRows::owner);
  ln.pend_kind = -1;
}

void Model::pf_lane_mixers(int layer, PfMixer which) {
  PfRun &R = *pf_run_;
  const int H = cfg_.hidden, W = hs_.shape.width();
  const bool attention = which == PfMixer::attention;
  // the block the mixer combines: the previous module's sum (none before the first layer; the PLE block combined its own)
  const bool has_block = !attention || (layer > 0 && layer != ps_.layer);
  const int feature = attention && dflash_ ? df_feat_at_[static_cast<size_t>(layer)] : -1;  // a drafter feature
  auto weights_of = [&](int c) -> const k::HcWeights & { return attention ? hs_.attn[c][layer] : hs_.mlp[c][layer]; };
  for (int j = 0; j < static_cast<int>(R.lanes.size()); ++j) {
    const PfRun::Lane &ln = R.lanes[static_cast<size_t>(j)];
    pf_lane_sum(j);
    sycl::event written[kMaxCards];  // per card: its last kernel that writes the rows shared below
    for (int c = 0; c < nc_; ++c) {
      Card &K = *cards_[static_cast<size_t>(c)];
      const RowRange own = lane_own_rows(c, nc_, ln.r0, ln.m);
      const k::HcState &st = R.cs[static_cast<size_t>(c)];
      const bf16 *block = has_block ? (attention ? K.pf.ym : K.pf.ya) : nullptr;
      written[c] = k::hc_mixer_rows_part(fab_.compute(c), hs_.shape, own.lo, own.hi, st, block, weights_of(c), K.pf.hc, attention ? K.pf.xa : K.pf.xm).up;
      // the feature = the streams' mean: the card's own rows of the whole mean into pf.ya (free until this layer's
      // GDN / QSA sum), shared below; pf_lane_attention then takes each card's columns of it
      if (feature >= 0)
        written[c] = k::df_mean_cols(fab_.compute(c), own.rows(), st.residual() + static_cast<size_t>(own.lo) * W, hs_.shape.streams, H, 0,
                                     K.pf.ya + static_cast<size_t>(own.lo) * H, H, 0, H);
    }
    const size_t row_bytes = static_cast<size_t>(H) * sizeof(bf16);
    const PfShared sets[] = {attention ? pf_shared([](Card &K) { return K.pf.xa; }, row_bytes) : pf_shared([](Card &K) { return K.pf.xm; }, row_bytes),
                             pf_shared([](Card &K) { return K.pf.ya; }, row_bytes)};
    pf_lane_share(j, std::span(sets).first(feature >= 0 ? 2 : 1), std::span<const sycl::event>(written, static_cast<size_t>(nc_)));
  }
  for (int c = 0; c < nc_; ++c) k::hc_mixer_rows_flip(R.cs[static_cast<size_t>(c)], weights_of(c));
}

void Model::pf_lane_attention(int lane, int layer) {
  PfRun &R = *pf_run_;
  PfRun::Lane &ln = R.lanes[static_cast<size_t>(lane)];
  const int H = cfg_.hidden;
  const size_t at = static_cast<size_t>(ln.r0) * H, slot = static_cast<size_t>(R.slot);
  pf_lane_wait(lane);
  if (dflash_ && df_feat_at_[static_cast<size_t>(layer)] >= 0)  // each card's columns of the feature, from every row's mean
    for (int c = 0; c < nc_; ++c)
      dflash_->capture(c, fab_.compute(c), df_feat_at_[static_cast<size_t>(layer)], cards_[static_cast<size_t>(c)]->pf.ya + at, ln.m,
                       dflash_->prefill_features(c) + static_cast<size_t>(ln.r0) * dflash_->feat_cols());
  ln.pend_seq = pf_split_seq_ = k::pair_split_next_seq(pf_split_seq_);
  ln.pend_kind = cfg_.full_attention[layer] ? 1 : 0;
  // The QSA indexer by owner rows on four cards: each card scores and selects its own rows of the lane and shares their
  // selection (sel, sel_n), instead of every card selecting every row (the same bits: a row's top-512 depends only on
  // its own scores). Only where a select runs: the lane's last row past 512 complete blocks.
  const bool sel_owner = cfg_.full_attention[layer] && nc_ == kMaxCards && (R.pos0 + ln.r0 + ln.m) / k::kQsaCompress > k::kQsaTopBlocks;
  const uint32_t sel_seq = sel_owner ? (pf_split_seq_ = k::pair_split_next_seq(pf_split_seq_)) : 0;
  k::PairSplitSend send[kMaxCards];
  for (int c = 0; c < nc_; ++c) {
    Card &K = *cards_[static_cast<size_t>(c)];
    k::PairRowsWire pw = K.pf.pw;
    pw.split = true;
    send[c] = {&*K.pf.side, &K.pf.lane[lane], ln.pend_seq};
    pw.send = &send[c];
    const int32_t *words = K.pf.lane_words + static_cast<size_t>(ln.w0) * k::kSegRec;  // the lane's first slice record
    if (!cfg_.full_attention[layer]) {
      // GDN over the lane: replays the previous step's pending rows (the call's first lane only), commits through its last row
      const k::GdnState &g = K.gst[static_cast<size_t>(layer)];
      k::GdnRowsState rs;
      rs.S = g.S + slot * g.slot_S;
      rs.win_in = g.win[ln.gp0 ^ 1] + slot * g.slot_win;
      rs.win_out = g.win[ln.gp0] + slot * g.slot_win;  // what the slot's next parity (gp0 ^ 1) reads
      rs.log_y = g.y[ln.gp0 ^ 1] + slot * g.slot_y;
      rs.log_ab = g.ab[ln.gp0 ^ 1] + slot * g.slot_ab;
      rs.accept = words + k::kSegAccept;
      rs.pos = opt_.gdn_scan ? static_cast<int>(R.pos0 + ln.r0) : -1;
      k::GdnRowsScratch sc = K.pf.gdn;
      sc.partial += at;  // the lane's own rows of the partial: its sum runs after the other lanes' modules
      sc.trace = stage_trace(fab_, c, R.gdn_trace);
      k::gdn_rows(fab_.compute(c), gs_.shape_of(c), ln.m, K.pf.xa + at, gs_.w[c][layer], rs, sc, wire_of(pw), pf_pair_parity_, K.pf.ya + at);
    } else {
      k::QsaRowsScratch sc = K.pf.qsa;
      sc.partial += at;
      sc.trace = stage_trace(fab_, c, R.qsa_trace);
      std::function<void(const sycl::event *)> between;
      if (sel_owner) {
        // the lane's rows of sel / sel_n (a peer's share lands there while this card may still read the lane before)
        const size_t sel_at = static_cast<size_t>(ln.r0);
        sc.sel += sel_at * k::kQsaTopBlocks;
        sc.sel_n += sel_at;
        const RowRange own = lane_own_rows(c, nc_, 0, ln.m);
        sc.sel_lo = own.lo / k::kQsaRows * k::kQsaRows;
        sc.sel_hi = std::min(ln.m, (own.hi + k::kQsaRows - 1) / k::kQsaRows * k::kQsaRows);
        between = [this, c, lane, sel_at, sel_seq, m = ln.m](const sycl::event *after) {
          Card &Kc = *cards_[static_cast<size_t>(c)];
          const PfShared sets[] = {pf_shared([&](Card &X) { return X.pf.qsa.sel + sel_at * k::kQsaTopBlocks; }, k::kQsaTopBlocks * sizeof(int32_t)),
                                   pf_shared([&](Card &X) { return X.pf.qsa.sel_n + sel_at; }, sizeof(int32_t))};
          const k::SplitShare shares[] = {{sets[0].bufs, sets[0].row_bytes}, {sets[1].bufs, sets[1].row_bytes}};
          k::pair_split_share(fab_.compute(c), &*Kc.pf.side, c, m, shares, Kc.pf.sel[lane], sel_seq, after);
          k::pair_split_wait(fab_.compute(c), Kc.pf.sel[lane], sel_seq);
        };
        sc.between = &between;
      }
      k::qsa_rows(fab_.compute(c), qs_.shape_of(c), ln.m, K.pf.xa + at, qs_.w[c][layer], K.qc[static_cast<size_t>(layer)], K.rope, words, k::kSegRec,
                  R.pos0 + ln.r0, K.qsc, sc, wire_of(pw), pf_pair_parity_, K.pf.ya + at);
    }
  }
  pf_pair_parity_ ^= 1;
}

void Model::pf_lanes_moe(int lane0, int count, int layer) {
  PfRun &R = *pf_run_;
  const int per = ms_.shape.per_card(), r0 = R.lanes[static_cast<size_t>(lane0)].r0;
  const size_t at = static_cast<size_t>(r0) * cfg_.hidden;
  int m = 0;
  for (int j = lane0; j < lane0 + count; ++j) {
    PfRun::Lane &ln = R.lanes[static_cast<size_t>(j)];
    pf_lane_wait(j);
    ln.pend_seq = pf_split_seq_ = k::pair_split_next_seq(pf_split_seq_);
    ln.pend_kind = 2;
    m += ln.m;
  }
  k::PairSplitSend send[kMaxCards][kPfLanesMax];
  k::PairSplit fp32_split[kMaxCards][kPfLanesMax];
  k::MoeSplitPart parts[kMaxCards][kPfLanesMax];
  for (int c = 0; c < nc_; ++c) {
    Card &K = *cards_[static_cast<size_t>(c)];
    k::MoeWeights w = ms_.w[c][layer];
    w.blobs = K.tab + static_cast<size_t>(layer) * per;
    if (K.cold) w.cold = K.cold + static_cast<size_t>(layer) * per;  // cold-expert staging
    k::MoeRowsWire mw = K.pf.mw;
    mw.split = true;
    mw.shared_own_rows = true;  // a lane's sum adds the shared term on its own rows only
    for (int i = 0; i < count; ++i) {
      const PfRun::Lane &ln = R.lanes[static_cast<size_t>(lane0 + i)];
      fp32_split[c][i] = K.pf.lane[lane0 + i];
      fp32_split[c][i].bf16 = false;  // fp32 MoE sums
      send[c][i] = {&*K.pf.side, &fp32_split[c][i], ln.pend_seq};
      parts[c][i] = {ln.r0 - r0, ln.m, &send[c][i]};
    }
    mw.parts = std::span<const k::MoeSplitPart>(parts[c], static_cast<size_t>(count));
    k::MoeRowsScratch sc = K.pf.moe;  // what the lanes' sums read later keeps its own rows: the partial and the shared term
    sc.part += at;
    sc.sy += at;
    sc.sg += r0;
    k::moe_rows(fab_.compute(c), ms_.shape_of(c), m, K.pf.xm + at, w, sc, wire_of(mw), pf_moe_parity_, K.pf.ym + at, opt_.prefill_moe_a8);
  }
  pf_moe_parity_ ^= 1;
}

void Model::pf_owner_ple(int layer) {
  PfRun &R = *pf_run_;
  // The block's conv reads the rows before a row (gated by their residual streams) and leaves its history in the slot's
  // ring on every card: every card needs every row's streams here. Before the first layer they are the embeddings
  // (every card has them); later the owners share the chunk's state first.
  if (layer > 0) pf_owner_state();
  for (int c = 0; c < nc_; ++c) {
    auto &q = fab_.compute(c);
    Card &K = *cards_[static_cast<size_t>(c)];
    k::HcState &st = R.cs[static_cast<size_t>(c)];
    if (layer > 0) k::hc_combine(q, hs_.shape, R.m, st, K.pf.ym);
    k::PleState ps = K.ple_st;
    ps.ring += static_cast<size_t>(R.slot) * ps.slot_elems;
    k::ple_rows(q, ps_.shape, R.m, R.pos0, reinterpret_cast<const bf16 *>(pf_ple_), ps_.w[c], ps, K.pf.ple, st.residual());
  }
  pf_mark(1);
}

// Every card gets every row of the chunk's state: the lanes' pending sums, then the owners' rows of the residual
// streams, the injection logits and that sum to the other cards.
void Model::pf_owner_state() {
  PfRun &R = *pf_run_;
  const k::HcState &st = R.cs[0];  // the cards' indices and strides agree (each mixer flips every card alike)
  const int cur = st.cur, lcur = st.lcur;
  const PfShared sets[] = {
      pf_shared([&](Card &K) { return K.pf.res[cur]; }, static_cast<size_t>(hs_.shape.width()) * sizeof(bf16)),
      pf_shared([&](Card &K) { return K.pf.lora[lcur]; }, static_cast<size_t>(st.lora_ld) * sizeof(float)),
      pf_shared([](Card &K) { return K.pf.ym; }, static_cast<size_t>(cfg_.hidden) * sizeof(bf16)),  // the last layer's pending sum
  };
  for (int j = 0; j < static_cast<int>(R.lanes.size()); ++j) {
    pf_lane_sum(j);
    pf_lane_share(j, sets);
  }
  for (int j = 0; j < static_cast<int>(R.lanes.size()); ++j) pf_lane_wait(j);
}

void Model::pf_layers_owner(int count) {
  PfRun &R = *pf_run_;
  const int L = cfg_.layers, lanes = static_cast<int>(R.lanes.size());
  const int l_end = std::min(L, R.l + count);
  for (; R.l < l_end; ++R.l) {
    const int l = R.l;
    if (l == ps_.layer) pf_owner_ple(l);
    pf_lane_mixers(l, PfMixer::attention);
    pf_mark(0);
    for (int j = 0; j < lanes; ++j) pf_lane_attention(j, l);
    pf_mark(cfg_.full_attention[l] ? 2 : 5);
    pf_lane_mixers(l, PfMixer::mlp);
    pf_mark(0);
    for (int j = 0; j < lanes; j += R.moe_lanes) pf_lanes_moe(j, std::min(R.moe_lanes, lanes - j), l);
    pf_mark(3);
  }
  if (R.l == L) {
    pf_owner_state();
    if (pf_sprof()) R.sub_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - R.t_sub0).count();
  }
}

void Model::pf_layers(int count) {
  PfRun &R = *pf_run_;
  if (!R.lanes.empty()) {
    pf_layers_owner(count);
    return;
  }
  const int L = cfg_.layers, H = cfg_.hidden, W = hs_.shape.width(), per = ms_.shape.per_card();
  const int m = R.m, slices = R.slices;
  auto slice_rows = [&](int s) { return std::min(4, m - 4 * s); };
  const size_t slot = static_cast<size_t>(R.slot);
  const int64_t pos0 = R.pos0;
  const int gp0 = R.gp0;
  auto &cs = R.cs;
  auto rows_at = [&](bf16 *p, int s) { return p + static_cast<size_t>(s) * 4 * H; };
  const bool prof = pf_prof();
  // The chunk's two-card sums (GDN, QSA, MoE) split over the cards from kPfSplitMinRows rows (pair_split_rows: copies
  // instead of mailbox words; the same bits). The producers (GDN / QSA out projection, MoE combine) finish the peer's
  // rows first and send them on the card's side queue, so the copy runs beside their own rows (and the shared expert);
  // pf_split_sum finishes. TP1 has no two-card sum (no split, no sends); TP4 splits as TP2 (each card owns a quarter of
  // the rows; the thresholds are TP2's).
  const bool split = nc_ > 1 && m >= kPfSplitMinRows, send_ok = nc_ > 1 && m >= k::kPairSplitSendMinRows;
  const int l_end = std::min(L, R.l + count);
  for (; R.l < l_end; ++R.l) {
    const int l = R.l;
    // attention mixer over the chunk (PLE first on its layer: the pending combine, then the block per slice)
    for (int c = 0; c < nc_; ++c) {
      auto &q = fab_.compute(c);
      Card &K = *cards_[static_cast<size_t>(c)];
      k::HcState &st = cs[static_cast<size_t>(c)];
      const bf16 *blk = l > 0 ? K.pf.ym : nullptr;
      if (l == ps_.layer) {
        if (c == 0) pf_split_wait();  // the combine and PLE's conv read every row
        if (blk) {
          k::hc_combine(q, hs_.shape, m, st, blk);
          blk = nullptr;
        }
        k::PleState ps = K.ple_st;
        ps.ring += slot * ps.slot_elems;
        k::ple_rows(q, ps_.shape, m, pos0, reinterpret_cast<const bf16 *>(pf_ple_), ps_.w[c], ps, K.pf.ple, st.residual());
      }
    }
    if (l == ps_.layer) pf_mark(1);
    if (pf_trace_) {
      const std::string L_ = "L" + std::to_string(l) + ".";
      if (l == ps_.layer) pf_tr(L_ + "ple.res", cs[0].residual(), W);
      if (l > 0 && l != ps_.layer) pf_tr(L_ + "ym_in", cards_[0]->pf.ym, H);
    }
    for (int c = 0; c < nc_; ++c) {
      Card &K = *cards_[static_cast<size_t>(c)];
      const bf16 *blk = l > 0 && l != ps_.layer ? K.pf.ym : nullptr;
      pf_mix_rows(c, blk, hs_.attn[c][l], K.pf.xa);
      if (dflash_ && df_feat_at_[static_cast<size_t>(l)] >= 0)  // a drafter feature of every chunk row
        dflash_->capture_mean(c, fab_.compute(c), df_feat_at_[static_cast<size_t>(l)], cs[static_cast<size_t>(c)].residual(), hs_.shape.streams, m,
                              dflash_->prefill_features(c));
    }
    pf_mark(0);
    if (pf_trace_) {
      pf_tr("L" + std::to_string(l) + ".attn_mix.res", cs[0].residual(), W);
      pf_tr("L" + std::to_string(l) + ".xa", cards_[0]->pf.xa, H);
    }
    sycl::event sent[kMaxCards];  // the split's peer copies, sent by the producers
    k::PairSplitSend send[kMaxCards];
    if (!cfg_.full_attention[l]) {
      // GDN over the chunk: replays the previous step's pending rows, commits through the last row
      const int pnext = gp0 ^ 1;  // the slot's next decode step's parity: it reads win[pnext ^ 1]
      for (int c = 0; c < nc_; ++c) {
        Card &K = *cards_[static_cast<size_t>(c)];
        const k::GdnState &g = K.gst[static_cast<size_t>(l)];
        k::GdnRowsState rs;
        rs.S = g.S + slot * g.slot_S;
        rs.win_in = g.win[gp0 ^ 1] + slot * g.slot_win;
        rs.win_out = g.win[pnext ^ 1] + slot * g.slot_win;
        rs.log_y = g.y[gp0 ^ 1] + slot * g.slot_y;
        rs.log_ab = g.ab[gp0 ^ 1] + slot * g.slot_ab;
        rs.accept = K.pf.words + k::kSegAccept;  // slice 0's accept = the previous step's rows
        rs.pos = opt_.gdn_scan ? static_cast<int>(R.pos0) : -1;  // chunks at absolute multiples of 64
        k::PairRowsWire pw = K.pf.pw;
        pw.split = split;
        if (send_ok) {
          send[c] = {&*K.pf.side, &K.pf.split, k::pair_split_next_seq(pf_split_seq_)};
          pw.send = &send[c];
        }
        k::GdnRowsScratch gsc = K.pf.gdn;
        gsc.trace = stage_trace(fab_, c, R.gdn_trace);
        sent[c] = k::gdn_rows(fab_.compute(c), gs_.shape_of(c), m, K.pf.xa, gs_.w[c][l], rs, gsc, wire_of(pw), pf_pair_parity_, K.pf.ya).send;
      }
      if (split) pf_split_sum([&](Card &K) { return K.pf.gdn.partial; }, [&](Card &K) { return K.pf.ya; }, false, 0, send_ok ? sent : nullptr);
      pf_pair_parity_ ^= 1;
    } else {
      // QSA over the chunk: projections + two-card sum, prep, then 64-row batches of the middle
      for (int c = 0; c < nc_; ++c) {
        Card &K = *cards_[static_cast<size_t>(c)];
        k::PairRowsWire pw = K.pf.pw;
        pw.split = split;
        if (send_ok) {
          send[c] = {&*K.pf.side, &K.pf.split, k::pair_split_next_seq(pf_split_seq_)};
          pw.send = &send[c];
        }
        k::QsaRowsScratch qrs = K.pf.qsa;
        qrs.trace = stage_trace(fab_, c, R.qsa_trace);
        sent[c] = k::qsa_rows(fab_.compute(c), qs_.shape_of(c), m, K.pf.xa, qs_.w[c][l], K.qc[static_cast<size_t>(l)], K.rope, K.pf.words, k::kSegRec, pos0, K.qsc,
                    qrs, wire_of(pw), pf_pair_parity_, K.pf.ya).send;
      }
      if (split) pf_split_sum([&](Card &K) { return K.pf.qsa.partial; }, [&](Card &K) { return K.pf.ya; }, false, 1, send_ok ? sent : nullptr);
      pf_pair_parity_ ^= 1;
    }
    pf_mark(cfg_.full_attention[l] ? 2 : 5);
    pf_tr("L" + std::to_string(l) + (cfg_.full_attention[l] ? ".qsa.ya" : ".gdn.ya"), cards_[0]->pf.ya, H);
    // MLP mixer over the chunk, then the MoE per 4-row slice (decode kernels)
    for (int c = 0; c < nc_; ++c) {
      Card &K = *cards_[static_cast<size_t>(c)];
      pf_mix_rows(c, K.pf.ya, hs_.mlp[c][l], K.pf.xm);
    }
    pf_mark(0);
    if (pf_trace_) {
      pf_tr("L" + std::to_string(l) + ".mlp_mix.res", cs[0].residual(), W);
      pf_tr("L" + std::to_string(l) + ".xm", cards_[0]->pf.xm, H);
    }
    if (calib_cap_ && l == calib_layer_) {  // the MoE's input, the residual, the injection logits (card 0)
      for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();
      if (pf_run_->pend_seq) pf_split_wait();
      calib_capture_mix(m, cs[0], cards_[0]->pf.xm);
    }
    if (ms_.shape.tp) {
      for (int c = 0; c < nc_; ++c) {
        Card &K = *cards_[static_cast<size_t>(c)];
        k::MoeWeights w = ms_.w[c][l];
        w.blobs = K.tab + static_cast<size_t>(l) * per;
        if (K.cold) w.cold = K.cold + static_cast<size_t>(l) * per;  // cold-expert staging
        k::MoeRowsWire mw = K.pf.mw;
        mw.split = split;
        k::PairSplit msp = K.pf.split;
        msp.bf16 = false;  // fp32 MoE sums (pf_split_sum)
        if (send_ok) {
          send[c] = {&*K.pf.side, &msp, k::pair_split_next_seq(pf_split_seq_)};
          mw.send = &send[c];
        }
        const auto me = k::moe_rows(fab_.compute(c), ms_.shape_of(c), m, K.pf.xm, w, K.pf.moe, wire_of(mw), pf_moe_parity_, K.pf.ym, opt_.prefill_moe_a8);
        if (prof || pf_dprof()) (c == 0 ? R.moe_ev : R.moe_ev1).push_back(me);
        sent[c] = me.send;
      }
      if (split) pf_split_sum([&](Card &K) { return K.pf.moe.part; }, [&](Card &K) { return K.pf.ym; }, true, 2, send_ok ? sent : nullptr);
      pf_moe_parity_ ^= 1;
      // the routing (card 0's; both cards route the same rows identically)
      if (pf_trace_) pf_tr("L" + std::to_string(l) + ".moe.ids", reinterpret_cast<const bf16 *>(cards_[0]->pf.moe.ids), 2 * cfg_.topk);
      if ((calib_cap_ || calib_hook_) && l == calib_layer_) {  // routing + the shared term; the hook's MoE output
        pf_split_wait();
        for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();
        calib_capture_moe(l, m);
      }
    } else {
    for (int s = 0; s < slices; ++s) {
      const int rows = slice_rows(s);
      for (int c = 0; c < nc_; ++c) {
        auto &q = fab_.compute(c);
        Card &K = *cards_[static_cast<size_t>(c)];
        k::MoeWeights w = ms_.w[c][l];
        w.blobs = K.tab + static_cast<size_t>(l) * per;
        k::MoeScratch sc = K.msc;
        sc.route = K.msc.route + l;
        k::moe_decode(q, ms_.shape_of(c), rows, rows_at(K.pf.xm, s), w, sc, wire_of(K.mw), moe_parity_, rows_at(K.pf.ym, s), opt_.moe_gu_ksplit, opt_.moe_dn_ksplit);
      }
      moe_parity_ ^= 1;
    }
    }
    pf_mark(3);
    pf_tr("L" + std::to_string(l) + ".moe.ym", cards_[0]->pf.ym, H);
  }
  pf_split_wait();  // the head, the drafter and decode steps between the parts read every row
  if ((pf_dprof() || pf_sprof()) && R.l == L) R.sub_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - R.t_sub0).count();
}

void Model::pf_finish() {
  PfRun &R = *pf_run_;
  slot_ = R.slot;  // the drafter's pass 1 names it (decode steps between the parts do not move it)
  const int H = cfg_.hidden, W = hs_.shape.width();
  const int m = R.m, slices = R.slices, row_base = R.row_base;
  const int32_t next_after = R.next_after;
  const int32_t *tokens = R.tokens.data();
  const auto &on_rows = R.on_rows;
  auto slice_rows = [&](int s) { return std::min(4, m - 4 * s); };
  Seq &sq = seqs_[static_cast<size_t>(R.slot)];
  const int64_t pos0 = R.pos0;
  const int gp0 = R.gp0;
  auto &cs = R.cs;
  const bool prof = pf_prof(), dprof = pf_dprof();
  // A 4-row slice of the chunk state: res rows at 4 s; lora rows at the current row stride (the
  // many-row mixers write [m][down.n] contiguous).
  auto view = [&](const k::HcState &st, int s) {
    k::HcState v = st;
    for (int i = 0; i < 2; ++i) {
      v.res[i] = st.res[i] + static_cast<size_t>(s) * 4 * W;
      v.lora[i] = st.lora[i] + static_cast<size_t>(s) * 4 * st.lora_ld;
    }
    return v;
  };
  auto rows_at = [&](bf16 *p, int s) { return p + static_cast<size_t>(s) * 4 * H; };
  // The head's indices: the chunk's, shifted by what decode steps between the call's parts flipped
  // (none for an unsplit call).
  int shift_cur[kMaxCards], shift_lcur[kMaxCards];
  for (int c = 0; c < nc_; ++c) {
    shift_cur[c] = head_->state(c).cur ^ R.start_cur[c];
    shift_lcur[c] = head_->state(c).lcur ^ R.start_lcur[c];
  }

  // Head: the last slice (or every slice with on_rows / the slices drafter) on the head's own state:
  // its rows are copied in under the chunk's indices, then the final mixer / lm_head / pick run as in
  // a step. The drafter's pass 1 (its prompt KV): slices = after each slice's head on the head's
  // hidden rows; rows = once over the chunk after the head (mtp_prefill_rows).
  const MtpPrefill dmode = !mtp_ ? MtpPrefill::off : (mtp_pf_ == MtpPrefill::rows && !ms_.shape.tp ? MtpPrefill::slices : mtp_pf_);
  const bool draft_slices = dmode == MtpPrefill::slices;
  auto next_of = [&](int i, const std::vector<int32_t> &last_picks) {  // row i's next token (last_picks: the last slice's)
    const int32_t nx = i + 1 < m ? tokens[i + 1] : next_after;
    return nx >= 0 ? nx : last_picks.at(static_cast<size_t>(i % 4));
  };
  std::vector<int32_t> last_picks;
  if (pf_lp_) pf_head_rows();  // every row's logprob records; the last slice's head below as always
  for (int s = on_rows || draft_slices ? 0 : slices - 1; s < slices; ++s) {
    const int rows = slice_rows(s);
    for (int c = 0; c < nc_; ++c) {
      auto &q = fab_.compute(c);
      k::HcState &hst = head_->state(c);
      const k::HcState v = view(cs[static_cast<size_t>(c)], s);
      hst.cur = v.cur ^ shift_cur[c];
      hst.lcur = v.lcur ^ shift_lcur[c];
      hst.lora_ld = v.lora_ld;
      q.memcpy(hst.res[hst.cur], v.res[v.cur], static_cast<size_t>(k::kDecodeRows) * W * 2);
      q.memcpy(hst.lora[hst.lcur], v.lora[v.lcur], static_cast<size_t>(k::kDecodeRows) * v.lora_ld * 4);
    }
    head_->final_mix(rows, per_card([&](int c) { return rows_at(cards_[static_cast<size_t>(c)]->pf.ym, s); }));
    head_->lm_pick(rows);
    head_->wait();
    if (on_rows) on_rows(row_base + 4 * s, rows);
    if (s == slices - 1 && next_after < 0) last_picks = head_->tokens(0, rows);
    if (draft_slices) {
      Mtp &M = *mtp_;
      std::vector<int32_t> picks;
      for (int r = 0; r < rows; ++r) {
        const int i = 4 * s + r;
        int32_t nx = i + 1 < m ? tokens[i + 1] : next_after;
        if (nx < 0) {
          if (picks.empty()) picks = head_->tokens(0, rows);
          nx = picks[static_cast<size_t>(r)];
        }
        M.ids_host[r] = nx;
      }
      if (vis_) {  // a prompt's next token may be an image position
        const int32_t *mp = vis_map(M.ids_host, rows, vis_tmp_);
        for (int r = 0; r < rows; ++r) M.ids_host[r] = mp[r];
      }
      sq.run_pos = pos0 + 4 * s;
      sq.run_rows = rows;
      for (int c = 0; c < nc_; ++c) fab_.compute(c).memcpy(M.cards[static_cast<size_t>(c)].ids, M.ids_host, static_cast<size_t>(rows) * 4);
      M.p1_segs = {{R.slot, sq.run_pos}};
      M.p1_seg_rows = rows;
      mtp_pass(0, rows, per_card([&](int c) -> const bf16 * { return head_->residual(c); }), per_card([&](int c) -> const int32_t * { return M.cards[static_cast<size_t>(c)].ids; }), M.p1_segs, rows);
      M.last_rows = rows;
      if (mtp_pf_picks_) {
        const auto d = draft_picks(rows);
        mtp_pf_picks_->insert(mtp_pf_picks_->end(), d.begin(), d.end());
      }
    }
  }
  pf_mark(4);
  if (dmode == MtpPrefill::rows) {
    std::vector<int32_t> next(static_cast<size_t>(m));
    for (int i = 0; i < m; ++i) next[static_cast<size_t>(i)] = next_of(i, last_picks);
    sq.run_pos = pos0 + 4 * (slices - 1);
    sq.run_rows = slice_rows(slices - 1);
    mtp_prefill_rows(m, pos0, cs, next);
    pf_mark(6);
  }
  if (dflash_) {  // every row of the chunk into the drafter's context (its features need no next token);
    // the chunk's bf16 rows (pf.xa / pf.xm) and pair wire are free after the layers and the head
    std::vector<arch::DFlash::Lend> lend(static_cast<size_t>(nc_));
    k::PairRowsWire pw[kMaxCards];
    for (int c = 0; c < nc_; ++c) {
      Card &K = *cards_[static_cast<size_t>(c)];
      pw[c] = K.pf.pw;
      pw[c].split = false;
      pw[c].send = nullptr;
      lend[static_cast<size_t>(c)] = {K.pf.xa, K.pf.xm, df_part_[c], df_ykv_[c], wire_of(pw[c])};
    }
    dflash_->prefill_ctx(m, pos0, R.slot, lend, pf_pair_parity_);
    pf_pair_parity_ ^= 1;
    pf_mark(6);
  }
  for (int mc = 0; mc < nc_ && (prof || dprof); ++mc) {
    const auto &mev = mc == 0 ? R.moe_ev : R.moe_ev1;
    if (mev.empty()) continue;
    fab_.compute(mc).wait();
    double t[7] = {};
    auto ms = [](const sycl::event &a, const sycl::event &b) {
      return static_cast<double>(b.get_profiling_info<sycl::info::event_profiling::command_end>() - a.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1e6;
    };
    for (const auto &e : mev) {
      t[0] += ms(e.router, e.router);
      t[1] += ms(e.route, e.route);
      t[2] += ms(e.gate_up, e.gate_up);
      t[3] += ms(e.down, e.down);
      t[4] += ms(e.combine, e.combine);
      t[5] += ms(e.topk, e.topk);
      t[6] += ms(e.shared_gu, e.shared_down);
    }
    std::printf("  moe card %d kernels: router %.1f, topk %.1f, group %.1f, gate_up %.1f, down %.1f, shared %.1f, combine %.1f ms\n", mc, t[0], t[5], t[1], t[2], t[3], t[6], t[4]);
  }
  // The split sums per card and kind, device clock of that card (each card's own clock: no cross-card times). pre: the
  // barrier after the partial's producer (its end = the producer's end), copy: the partial's half to the peer, wait:
  // from the copy's end to the peer's flag (the flag store + skew), sum, back: the rows' copy to the peer, tail: from
  // back's end to the peer's rows' flag. The producers send the copy before their own rows, so gap is negative (the
  // overlap) and all = the exposed part. XEN_PF_PROFILE=2's barriers can stop the side queue's copy from running beside
  // the kernels on some runs: time the overlap without the profile. A deferred sum's wait follows the consumer's own
  // rows, so its tail and all include them.
  if ((prof || dprof) && !R.split_ev.empty()) {
    for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();
    auto st = [](const sycl::event &e) { return e.get_profiling_info<sycl::info::event_profiling::command_start>(); };
    auto en = [](const sycl::event &e) { return e.get_profiling_info<sycl::info::event_profiling::command_end>(); };
    auto d = [](uint64_t a, uint64_t b) { return static_cast<double>(static_cast<int64_t>(b - a)) / 1e6; };
    static const char *names[3] = {"gdn", "qsa", "moe"};
    for (int kind = 0; kind < 3; ++kind)
      for (int c = 0; c < nc_; ++c) {
        double t[7] = {};
        int n = 0;
        for (const auto &x : R.split_ev) {
          if (x.kind != kind) continue;
          const auto &e = x.ev;
          ++n;
          t[0] += d(en(x.prev[c]), st(e.copy[c]));      // gap: producer end -> copy start
          t[1] += d(st(e.copy[c]), en(e.copy[c]));      // copy
          t[2] += d(en(e.copy[c]), en(e.recv[c]));      // flag + wait for the peer's half
          t[3] += d(st(e.sum[c]), en(e.sum[c]));        // sum
          t[4] += d(st(e.back[c]), en(e.back[c]));      // back copy
          t[5] += d(en(e.back[c]), en(e.done[c]));      // flag + wait for the peer's rows
          t[6] += d(en(x.prev[c]), en(e.done[c]));      // all
        }
        if (n)
          std::printf("  split %s card %d, %d sums: gap %.2f, copy %.2f, wait %.2f, sum %.2f, back %.2f, tail %.2f = %.2f ms (%.3f per sum)\n", names[kind], c, n,
                      t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[6] / n);
      }
  }
  if (pf_sprof() && R.smarks.size() > 1) {
    for (int c = 0; c < nc_; ++c) {
      R.cal[c][1] = k::stamp(fab_.compute(c), pf_stamps_[c] + R.stamp_n[c]);
      R.cal_at[c][1] = R.stamp_n[c]++;
    }
    std::vector<uint64_t> st[kMaxCards];
    double ns_per[kMaxCards] = {};
    for (int c = 0; c < nc_; ++c) {
      fab_.compute(c).wait();
      st[c].resize(static_cast<size_t>(R.stamp_n[c]));
      fab_.compute(c).memcpy(st[c].data(), pf_stamps_[c], st[c].size() * 8).wait();
      auto end_of = [](const sycl::event &e) { return e.get_profiling_info<sycl::info::event_profiling::command_end>(); };
      const double ticks = static_cast<double>(st[c][static_cast<size_t>(R.cal_at[c][1])] - st[c][static_cast<size_t>(R.cal_at[c][0])]);
      if (fab_.profiling()) {
        const double ns = static_cast<double>(end_of(R.cal[c][1]) - end_of(R.cal[c][0]));
        ns_per[c] = ticks > 0 ? ns / ticks : 0;
      } else {
        ns_per[c] = 1.0 / 2.8;  // queues without event profiling (the server's): the nominal 2.8 GHz GT clock
      }
    }
    auto ms0 = [&](uint64_t a, uint64_t b) { return static_cast<double>(static_cast<int64_t>(b - a)) * ns_per[0] / 1e6; };
    double d[7] = {};
    for (size_t i = 1; i < R.smarks.size(); ++i)
      d[R.smarks[i].first] += ms0(st[0][static_cast<size_t>(R.smarks[i - 1].second)], st[0][static_cast<size_t>(R.smarks[i].second)]);
    const double span = ms0(st[0][static_cast<size_t>(R.smarks.front().second)], st[0][static_cast<size_t>(R.smarks.back().second)]);
    std::printf("prefill chunk %d rows at %lld, card 0 device time (stamps, GT %.2f / %.2f GHz): hc %.2f, ple %.2f, qsa %.2f, gdn %.2f, moe %.2f, head %.2f, drafter %.2f = "
                "%.2f ms; host submit of the layers %.2f ms\n",
                m, static_cast<long long>(pos0), ns_per[0] > 0 ? 1.0 / ns_per[0] : 0.0, ns_per[1] > 0 ? 1.0 / ns_per[1] : 0.0, d[0], d[1], d[2], d[5], d[3], d[4], d[6],
                span, R.sub_ms);
    std::printf("prefill chunk %d rows, host submit per phase: hc %.2f, ple %.2f, qsa %.2f, gdn %.2f, moe %.2f, head (to the end) %.2f ms\n", m, R.ph[0], R.ph[1],
                R.ph[2], R.ph[5], R.ph[3], R.ph[4]);
    // The split sums' waits, each measured by the wait kernel itself (entry -> the flag seen): recv =
    // the peer's partial, rows = the peer's finished rows (deferred: after the consumer's own rows).
    static const char *names[4] = {"gdn", "qsa", "moe", "rows shared"};  // (the last: kPfSpanShared, pf_lane_wait)
    for (int kind = 0; kind < 4; ++kind)
      for (int c = 0; c < nc_; ++c) {
        double recv = 0, rows = 0;
        int n = 0, spun = 0;
        for (const auto &s : R.split_span) {
          if (s.kind != kind) continue;
          const uint64_t *w = st[c].data() + s.at[c];
          const double a = static_cast<double>(w[1] - w[0]) * ns_per[c] / 1e6, b = static_cast<double>(w[3] - w[2]) * ns_per[c] / 1e6;
          recv += a;
          rows += b;
          spun += (a > 0.002) + (b > 0.002);
          ++n;
        }
        if (n)
          std::printf("  split %s card %d, %d sums (stamps): wait recv %.3f, wait rows %.3f ms (%.4f per sum); %d of %d waits > 2 us\n", names[kind], c, n, recv, rows,
                      (recv + rows) / n, spun, 2 * n);
      }
  }
  if (dprof && R.dmarks.size() > 1) {
    fab_.compute(0).wait();
    double d[7] = {};
    auto end_of = [](const sycl::event &e) { return e.get_profiling_info<sycl::info::event_profiling::command_end>(); };
    for (size_t i = 1; i < R.dmarks.size(); ++i) d[R.dmarks[i].first] += static_cast<double>(end_of(R.dmarks[i].second) - end_of(R.dmarks[i - 1].second)) / 1e6;
    const double span = static_cast<double>(end_of(R.dmarks.back().second) - end_of(R.dmarks.front().second)) / 1e6;
    std::printf("prefill chunk %d rows at %lld, card 0 device time: hc %.2f, ple %.2f, qsa %.2f, gdn %.2f, moe %.2f, head %.2f, drafter %.2f = %.2f ms; host submit of the layers %.2f ms\n", m,
                static_cast<long long>(pos0), d[0], d[1], d[2], d[5], d[3], d[4], d[6], span, R.sub_ms);
  }
  if (prof)
    std::printf("prefill chunk %d rows at %lld: hc %.1f, ple %.1f, qsa %.1f, gdn %.1f, moe %.1f, head %.1f, drafter %.1f ms\n", m, static_cast<long long>(pos0), R.ph[0], R.ph[1],
                R.ph[2], R.ph[5], R.ph[3], R.ph[4], R.ph[6]);
  if (!R.qsa_trace.ev[0].empty() || !R.gdn_trace.ev[0].empty()) {  // card 0's GDN / QSA kernel ms by stage
    fab_.compute(0).wait();
    std::string line = "prefill chunk " + std::to_string(m) + " rows at " + std::to_string(pos0) + ", card 0 kernel ms by stage: QSA";
    char b[64];
    for (int st = 0; st < static_cast<int>(k::QsaStage::kCount); ++st) {
      std::snprintf(b, sizeof b, " %s %.2f,", k::qsa_stage_name(static_cast<k::QsaStage>(st)), R.qsa_trace.ms(st));
      line += b;
    }
    line += " GDN";
    for (int st = 0; st < static_cast<int>(k::GdnStage::kCount); ++st) {
      std::snprintf(b, sizeof b, " %s %.2f,", k::gdn_stage_name(static_cast<k::GdnStage>(st)), R.gdn_trace.ms(st));
      line += b;
    }
    line.pop_back();
    std::printf("%s\n", line.c_str());
  }
  // A chunk flips the HC state indices as one step does (each mixer once), so it advances the step
  // parity by one: recorded step graphs key on the parity and check the indices against it. Relative
  // to the parity now (decode steps may have run between the call's parts).
  step_parity_ ^= 1;
  const size_t gdn_commits = R.lanes.empty() ? 1 : R.lanes.size();  // each lane commits its rows as a call does
  sq.gdn_par = gp0 ^ static_cast<int>(gdn_commits & 1);
  sq.pos += m;
  sq.last_rows = 0;  // every GDN layer committed its rows: the next step replays nothing
  sq.pf_committed = true;  // a snapshot may be taken now
  pf_run_.reset();
}

int Model::pf_run_slot() const { return pf_run_ ? pf_run_->slot : -1; }

// The many-row head of a prompt-logprobs prefill, for the pending call's chunk.
// Its lm_head keeps bf16 activations under prefill_dense_a8: the records must equal the head per
// 4-row slice (decode numerics) bit for bit.
// The final mixer over the chunk on a COPY of its HC state (it writes the other res / lora buffers,
// which nothing reads after the layers: the last slice's head copies the current ones, the drafter's
// combine rewrites the other), its output in pf.xa (dead after the layers). Then per group of lp_rows
// rows: lm_head on pf_dense into pf.moe.y as bf16 (groups of as many rows as y holds, 105
// at 512-row chunks: 5 lm_head passes per chunk; the logits rounded once, as logprob_stats rounds
// them), the targets' logits from their owning cards to the host and back to both cards, the records.
// pf.xa / moe.y / hc scratch are reused by the drafter's pass afterwards (in order on each queue).
void Model::pf_head_rows() {
  PfRun &R = *pf_run_;
  const PfLp &L = *pf_lp_;
  const int m = R.m, H = cfg_.hidden, V = head_->vocab_per_card();
  if (R.row_base + m > L.n) throw std::logic_error("qwen38 model: prefill_logprobs rows");
  for (int c = 0; c < nc_; ++c) {
    Card &K = *cards_[static_cast<size_t>(c)];
    k::HcState v = R.cs[static_cast<size_t>(c)];
    k::hc_mixer_rows(fab_.compute(c), hs_.shape, m, v, K.pf.ym, head_->final_weights(c), K.pf.hc, K.pf.xa);
  }
  const int G = std::min(cards_[0]->pf.lp_rows, cards_[static_cast<size_t>(nc_ - 1)]->pf.lp_rows);
  std::vector<int32_t> tg(static_cast<size_t>(m));
  std::vector<std::vector<float>> own(static_cast<size_t>(nc_), std::vector<float>(static_cast<size_t>(m)));
  for (int i = 0; i < m; ++i) tg[static_cast<size_t>(i)] = L.targets[R.row_base + i];
  for (int g0 = 0; g0 < m; g0 += G) {
    const int gr = std::min(G, m - g0);
    for (int c = 0; c < nc_; ++c) {
      auto &q = fab_.compute(c);
      Card &K = *cards_[static_cast<size_t>(c)];
      auto *lg = reinterpret_cast<bf16 *>(K.pf.moe.y);
      q.memcpy(K.pf.lp_tgt, tg.data() + g0, static_cast<size_t>(gr) * 4);
      k::pf_dense(q, gr, head_->lm_weights(c), K.pf.xa + static_cast<size_t>(g0) * H, H, k::PfOut{nullptr, lg, V});
      k::logprob_targets(q, lg, V, gr, V, c * V, K.pf.lp_tgt, K.pf.lp_xt);
    }
    // the copies to pageable host memory block the host until their queue drains: both cards' heads are queued first,
    // so they run together
    for (int c = 0; c < nc_; ++c) fab_.compute(c).memcpy(own[c].data() + g0, cards_[static_cast<size_t>(c)]->pf.lp_xt, static_cast<size_t>(gr) * 4);
    for (int c = 0; c < nc_; ++c) fab_.compute(c).wait_and_throw();
    float *xt = L.xt + R.row_base + g0;
    for (int i = 0; i < gr; ++i) {
      const int32_t t = tg[static_cast<size_t>(g0 + i)];
      if (t >= nc_ * V) throw std::invalid_argument("qwen38 model: logprob target outside the vocabulary");
      xt[i] = t < 0 ? std::numeric_limits<float>::infinity() : own[t / V][static_cast<size_t>(g0 + i)];
    }
    for (int c = 0; c < nc_; ++c) {
      auto &q = fab_.compute(c);
      Card &K = *cards_[static_cast<size_t>(c)];
      q.memcpy(K.pf.lp_xt, xt, static_cast<size_t>(gr) * 4);
      k::logprob_stats_rows(q, reinterpret_cast<const bf16 *>(K.pf.moe.y), V, gr, V, c * V, L.top, K.pf.lp_xt, K.pf.lp_rec);
    }
    for (int c = 0; c < nc_; ++c)
      fab_.compute(c).memcpy(L.rec + (static_cast<size_t>(c) * L.n + R.row_base + g0) * k::kLpRec, cards_[static_cast<size_t>(c)]->pf.lp_rec,
                             static_cast<size_t>(gr) * k::kLpRec * 4);
  }
  for (int c = 0; c < nc_; ++c) fab_.compute(c).wait_and_throw();
}

// ---- calibration (the drafter's) --------------------------------------------------
// A CalibState's first m rows into the chunk's HC state (residual, the injection logits at the lora
// columns [lowrank, lowrank + S) under prev_layer's MLP mixer stride) and pf.ym, both cards.
void Model::calib_put_state(std::vector<k::HcState> &cs, const CalibState &in, int m, int prev_layer) {
  const int H = cfg_.hidden, W = hs_.shape.width(), S = hs_.shape.streams;
  for (int c = 0; c < nc_; ++c) {
    auto &q = fab_.compute(c);
    Card &K = *cards_[static_cast<size_t>(c)];
    k::HcState &st = cs[static_cast<size_t>(c)];
    st.lora_ld = hs_.mlp[c][prev_layer].down.n;  // what that layer's MLP mixer leaves
    q.memcpy(st.residual(), in.res.data(), static_cast<size_t>(m) * W * 2);
    q.memcpy(K.pf.ym, in.ym.data(), static_cast<size_t>(m) * H * 2);
    q.memcpy(calib_tmp_[c], in.inj.data(), static_cast<size_t>(m) * S * 4);
    float *lora = st.lora[st.lcur];
    const float *tmp = calib_tmp_[c];
    const int ld = st.lora_ld, col = hs_.shape.lowrank;
    q.parallel_for(sycl::range<1>(static_cast<size_t>(m) * S), [=](sycl::id<1> i) {
      lora[(i / S) * static_cast<size_t>(ld) + col + i % S] = tmp[i];
    });
    q.wait();
  }
}

// Card 0's state after a block: its residual, the injection logits, pf.ym (the pending MoE output).
void Model::calib_get_state(const k::HcState &st, int m, CalibState &out) {
  const int H = cfg_.hidden, W = hs_.shape.width(), S = hs_.shape.streams;
  auto &q = fab_.compute(0);
  out.m = m;
  out.res.resize(static_cast<size_t>(m) * W);
  out.inj.resize(static_cast<size_t>(m) * S);
  out.ym.resize(static_cast<size_t>(m) * H);
  const float *lora = st.lora[st.lcur];
  float *tmp = calib_tmp_[0];
  const int ld = st.lora_ld, col = hs_.shape.lowrank;
  q.parallel_for(sycl::range<1>(static_cast<size_t>(m) * S), [=](sycl::id<1> i) {
    tmp[i] = lora[(i / S) * static_cast<size_t>(ld) + col + i % S];
  });
  q.memcpy(out.res.data(), st.residual(), out.res.size() * 2);
  q.memcpy(out.inj.data(), tmp, out.inj.size() * 4);
  q.memcpy(out.ym.data(), cards_[0]->pf.ym, out.ym.size() * 2);
  q.wait();
}

// After the MLP mixer (both cards drained): the MoE's input, the residual, the injection logits.
void Model::calib_capture_mix(int m, const k::HcState &st, const bf16 *xm) {
  const int H = cfg_.hidden, W = hs_.shape.width(), Si = hs_.shape.streams;
  const size_t S = static_cast<size_t>(Si);
  auto &q = fab_.compute(0);
  CalibCapture &cp = *calib_cap_;
  cp.xm.resize(static_cast<size_t>(m) * H);
  cp.res.resize(static_cast<size_t>(m) * W);
  cp.inj.resize(static_cast<size_t>(m) * S);
  const float *lora = st.lora[st.lcur];
  const int ld = st.lora_ld, col = hs_.shape.lowrank;
  float *tmp = calib_tmp_[0];
  q.parallel_for(sycl::range<1>(static_cast<size_t>(m) * S), [=](sycl::id<1> i) {
    tmp[i] = lora[(i / Si) * static_cast<size_t>(ld) + col + i % Si];
  });
  q.memcpy(cp.xm.data(), xm, cp.xm.size() * 2);
  q.memcpy(cp.res.data(), st.residual(), cp.res.size() * 2);
  q.memcpy(cp.inj.data(), tmp, cp.inj.size() * 4);
  q.wait();
}

// After the MoE (both cards drained): routing + the shared term (card 0's moe_rows scratch); the
// hook's output replaces pf.ym on both cards.
void Model::calib_capture_moe(int layer, int m) {
  const int H = cfg_.hidden;
  CalibCapture &cp = *calib_cap_;
  const auto &mr = cards_[0]->pf.moe;
  auto &q = fab_.compute(0);
  const size_t K = static_cast<size_t>(cfg_.topk);
  cp.ids.resize(static_cast<size_t>(m) * K);
  cp.topw.resize(cp.ids.size());
  cp.sg.resize(static_cast<size_t>(m));
  cp.sy.resize(static_cast<size_t>(m) * H);
  q.memcpy(cp.ids.data(), mr.ids, cp.ids.size() * 4);
  q.memcpy(cp.topw.data(), mr.topw, cp.topw.size() * 4);
  q.memcpy(cp.sg.data(), mr.sg, cp.sg.size() * 4);
  q.memcpy(cp.sy.data(), mr.sy, cp.sy.size() * 4);
  q.wait();
  if (calib_hook_) {
    std::vector<uint16_t> ym;
    (*calib_hook_)(layer, cp, ym);
    if (ym.size() != static_cast<size_t>(m) * H) throw std::runtime_error("qwen38 model: calib hook wrote the wrong size");
    for (int c = 0; c < nc_; ++c) fab_.compute(c).memcpy(cards_[static_cast<size_t>(c)]->pf.ym, ym.data(), ym.size() * 2).wait();
  }
}

void Model::calib_layer(const int32_t *tokens, int m, int layer, const CalibState *in, CalibState *out, CalibCapture *cap,
                        const CalibMoeHook *hook) {
  if (!ms_.shape.tp) throw std::logic_error("qwen38 model: calib_layer needs expert TP (the many-row MoE)");
  if (m < 1 || m > opt_.prefill_chunk) throw std::invalid_argument("qwen38 model: calib_layer takes 1..prefill_chunk rows");
  if (layer < 0 || layer >= cfg_.layers) throw std::invalid_argument("qwen38 model: calib_layer: no such layer");
  if ((layer == 0) != (in == nullptr)) throw std::invalid_argument("qwen38 model: calib_layer: layer 0 starts from the embedding, others from `in`");
  if (in && (in->m != m || in->res.size() != static_cast<size_t>(m) * hs_.shape.width() ||
             in->inj.size() != static_cast<size_t>(m) * hs_.shape.streams || in->ym.size() != static_cast<size_t>(m) * cfg_.hidden))
    throw std::invalid_argument("qwen38 model: calib_layer: input state sizes");
  if (pf_run_) throw std::logic_error("qwen38 model: calib_layer while a prefill call is pending");
  use_slot(0);
  begin_sequence();
  pf_skip_ple_ = layer != ps_.layer;
  CalibCapture local;
  try {
    pf_start(tokens, m, 0, {}, -1);
    pf_skip_ple_ = false;
    PfRun &R = *pf_run_;
    if (in) {
      calib_put_state(R.cs, *in, m, layer - 1);
      R.l = layer;
    }
    calib_layer_ = layer;
    calib_cap_ = cap ? cap : (hook ? &local : nullptr);
    calib_hook_ = hook;
    pf_layers(1);
    for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();
    if (out) calib_get_state(R.cs[0], m, *out);
  } catch (...) {
    pf_skip_ple_ = false;
    calib_cap_ = nullptr;
    calib_hook_ = nullptr;
    calib_layer_ = -1;
    pf_run_.reset();
    throw;
  }
  calib_cap_ = nullptr;
  calib_hook_ = nullptr;
  calib_layer_ = -1;
  pf_run_.reset();
}

void Model::calib_final_state(const int32_t *tokens, int m, CalibState &out) {
  if (!ms_.shape.tp) throw std::logic_error("qwen38 model: calib_final_state needs expert TP (the many-row MoE)");
  if (m < 1 || m > opt_.prefill_chunk) throw std::invalid_argument("qwen38 model: calib_final_state takes 1..prefill_chunk rows");
  if (pf_run_) throw std::logic_error("qwen38 model: calib_final_state while a prefill call is pending");
  use_slot(0);
  begin_sequence();
  try {
    pf_start(tokens, m, 0, {}, -1);
    PfRun &R = *pf_run_;
    pf_layers(cfg_.layers);
    for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();
    calib_get_state(R.cs[0], m, out);
  } catch (...) {
    pf_run_.reset();
    throw;
  }
  pf_run_.reset();
}

// The drafter's layer (calib_layer_ == layers marks it for mtp_prefill_rows' capture points).
void Model::calib_mtp(const int32_t *tokens, int m, const CalibState &in, const int32_t *next, CalibState *out, CalibCapture *cap,
                      const CalibMoeHook *hook, std::vector<uint16_t> *head_in) {
  if (!mtp_) throw std::logic_error("qwen38 model: calib_mtp needs a drafter (ModelOptions::mtp_dir)");
  if (!ms_.shape.tp) throw std::logic_error("qwen38 model: calib_mtp needs expert TP (the many-row pass)");
  if (m < 1 || m > opt_.prefill_chunk) throw std::invalid_argument("qwen38 model: calib_mtp takes 1..prefill_chunk rows");
  if (in.m < m || in.res.size() != static_cast<size_t>(in.m) * hs_.shape.width() || in.inj.size() != static_cast<size_t>(in.m) * hs_.shape.streams ||
      in.ym.size() != static_cast<size_t>(in.m) * cfg_.hidden)
    throw std::invalid_argument("qwen38 model: calib_mtp: input state sizes");
  if (pf_run_) throw std::logic_error("qwen38 model: calib_mtp while a prefill call is pending");
  use_slot(0);
  begin_sequence();
  pf_skip_ple_ = true;
  CalibCapture local;
  auto clear = [&] {
    pf_skip_ple_ = false;
    calib_cap_ = nullptr;
    calib_hook_ = nullptr;
    calib_layer_ = -1;
    pf_run_.reset();
  };
  try {
    pf_start(tokens, m, 0, {}, -1);  // the chunk's records (positions, slot 0); the embedding is unused
    pf_skip_ple_ = false;
    PfRun &R = *pf_run_;
    calib_put_state(R.cs, in, m, cfg_.layers - 1);
    R.l = cfg_.layers;
    calib_layer_ = cfg_.layers;
    calib_cap_ = cap ? cap : (hook ? &local : nullptr);
    calib_hook_ = hook;
    std::vector<k::HcState> ds;
    mtp_prefill_rows(m, R.pos0, R.cs, std::vector<int32_t>(next, next + m), &ds);
    for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();
    if (out) calib_get_state(ds[0], m, *out);
    if (head_in) {  // the final mixer over every row (combines the pending MoE output)
      for (int c = 0; c < nc_; ++c) {
        Card &K = *cards_[static_cast<size_t>(c)];
        k::hc_mixer_rows(fab_.compute(c), hs_.shape, m, ds[static_cast<size_t>(c)], K.pf.ym, mtp_->cards[static_cast<size_t>(c)].fin, K.pf.hc, K.pf.xa);
      }
      head_in->resize(static_cast<size_t>(m) * cfg_.hidden);
      fab_.compute(1).wait();
      fab_.compute(0).memcpy(head_in->data(), cards_[0]->pf.xa, head_in->size() * 2).wait();
    }
  } catch (...) {
    clear();
    throw;
  }
  clear();
}

// The head on a CalibState after the last layer (pf_head_rows' path: final mixer over the rows, lm_head per group into
// pf.moe.y as bf16).
void Model::calib_logits(const int32_t *tokens, int m, const CalibState &in, const LogitSink &sink) {
  if (!ms_.shape.tp || !pf_logprob_rows()) throw std::logic_error("qwen38 model: calib_logits needs expert TP (the many-row head)");
  if (m < 1 || m > opt_.prefill_chunk) throw std::invalid_argument("qwen38 model: calib_logits takes 1..prefill_chunk rows");
  if (in.m < m || in.res.size() != static_cast<size_t>(in.m) * hs_.shape.width() || in.inj.size() != static_cast<size_t>(in.m) * hs_.shape.streams ||
      in.ym.size() != static_cast<size_t>(in.m) * cfg_.hidden)
    throw std::invalid_argument("qwen38 model: calib_logits: input state sizes");
  if (pf_run_) throw std::logic_error("qwen38 model: calib_logits while a prefill call is pending");
  use_slot(0);
  begin_sequence();
  pf_skip_ple_ = true;
  try {
    pf_start(tokens, m, 0, {}, -1);
    pf_skip_ple_ = false;
    PfRun &R = *pf_run_;
    calib_put_state(R.cs, in, m, cfg_.layers - 1);
    const int H = cfg_.hidden, V = head_->vocab_per_card();
    for (int c = 0; c < nc_; ++c) {
      Card &K = *cards_[static_cast<size_t>(c)];
      k::HcState v = R.cs[static_cast<size_t>(c)];
      k::hc_mixer_rows(fab_.compute(c), hs_.shape, m, v, K.pf.ym, head_->final_weights(c), K.pf.hc, K.pf.xa);
    }
    const int G = std::min(cards_[0]->pf.lp_rows, cards_[static_cast<size_t>(nc_ - 1)]->pf.lp_rows);
    std::vector<uint16_t> half[kMaxCards], joined;
    for (int g0 = 0; g0 < m; g0 += G) {
      const int gr = std::min(G, m - g0);
      for (int c = 0; c < nc_; ++c) {
        auto &q = fab_.compute(c);
        Card &K = *cards_[static_cast<size_t>(c)];
        auto *lg = reinterpret_cast<bf16 *>(K.pf.moe.y);
        k::pf_dense(q, gr, head_->lm_weights(c), K.pf.xa + static_cast<size_t>(g0) * H, H, k::PfOut{nullptr, lg, V});
        half[c].resize(static_cast<size_t>(gr) * V);
        q.memcpy(half[c].data(), lg, half[c].size() * 2);
      }
      for (int c = 0; c < nc_; ++c) fab_.compute(c).wait_and_throw();
      joined.resize(static_cast<size_t>(gr) * nc_ * V);
      for (int r = 0; r < gr; ++r)
        for (int c = 0; c < nc_; ++c)
          std::memcpy(joined.data() + (static_cast<size_t>(r) * nc_ + c) * V, half[c].data() + static_cast<size_t>(r) * V, static_cast<size_t>(V) * 2);
      sink(g0, gr, joined.data());
    }
  } catch (...) {
    pf_skip_ple_ = false;
    pf_run_.reset();
    throw;
  }
  pf_run_.reset();
}

// Image input (set_vision).
const int32_t *Model::vis_map(const int32_t *tokens, int m, std::vector<int32_t> &buf) {
  if (!vis_) return tokens;
  buf.resize(static_cast<size_t>(m));
  vis_->prepare(tokens, m, buf.data());
  return buf.data();
}

void Model::embed_rows(sycl::queue &q, int c, const int32_t *ids, int rows, int streams, bf16 *out) {
  if (!vis_) {
    k::embed_gather(q, head_->embed_table(c), ids, rows, cfg_.hidden, streams, out);
    return;
  }
  k::embed_gather_ext(q, head_->embed_table(c), ids, rows, cfg_.hidden, streams, out, static_cast<const bf16 *>(vis_->pool(c)), static_cast<int32_t>(vis_->vocab()));
}

}  // namespace xenithium::arch::qwen38
