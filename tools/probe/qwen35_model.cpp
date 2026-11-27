// xenithium-probe model35: the whole qwen35 model on both cards, one load for every run.
// xenithium-probe model35 --model DIR [--tp 1|2] [--ppl a.bin,b.bin] [--rows 1,4] [--generate N] [--at P] [--context N]
// [--kv-bytes 1|2] [--lp-out DIR] [--lp-ref F,F] [--pf-chunk N]
// [--pf-slices 0|1] [--pf-attn-slices 0|1] [--pf-attn 0|1|2|3] [--pf-p-lo 0|1] [--pf-w4dq 0|1] [--pf-split 0|1] [--pf-ab CASES] [--pf-reps N] [--pf-lens N,N]
// --ppl teacher-forced perplexity on token files (int32 n_chunks, then per chunk int32 length + ids: the
// qwen38 gate's sets; the vocabulary is the same; the probe counts the combining marks the two
// pre-tokenizer regexes treat differently and prints samples of the decoded text). Each chunk is a fresh sequence; a
// step feeds R tokens and row j's logits score token p0 + j + 1 (the chunk's last row scores its own
// greedy pick, as the reference gate does). Logits rounded to bf16, log-softmax in double. Gates: finite,
// device pick == host argmax, both cards' residuals identical after every chunk, every row count's
// per-chunk nll == the first's bit for bit, no mailbox timeout.
// --generate N greedy, 1 row per step, after chunk 0 of the first token file (or a fixed prompt): the decoded text,
// host ms per step; then the history is filled with corpus tokens in 4-row steps up to --at P (default
// 8,192) and 1-row and 4-row steps are timed with the per-module GPU profile (card 0).
// --rows pf the prefill path: each chunk of a token file through Model:prefill (layer-major, chunks of
// --pf-chunk rows, default 512) with the head on every 4-row slice for scoring; the gates above apply,
// plus, with --pf-slices 1 (every component on its decode kernels per 4-row slice), per-chunk nll == the
// 4-row steps' bit for bit; with the many-row kernels (default) judge it by ppl / --lp-ref KL against a
// rows-4 dump. Then per token file: prefill of chunk 0 timed (median of 3, head on the last slice only) and
// the continuation gate: 16 greedy tokens after a prefill == after 4-row steps over the same prompt
// (8-row with --rows 8). --pf-checks 0 skips this part: ~50 s of an 8K dump run.
// --pf-calls R,R.. prefill calls of R rows each at --at (slot 0 prefilled there first; the position then advances by
// every call): host wall median over --pf-reps rounds per --pf-ab case (cases alternate per round; with
// --pf-calls the --pf-ab section at position 0 is skipped), then one profiled call per R (first case), and the
// decode steps of --step-shapes' 1xR shapes (R <= the segment rows) at the same position for comparison.
// What a 9-63-row call (a 16-token draft block's verify) costs on the many-row path.
// --rows 8 8-row teacher-forced steps (GDN's 8-row log without a drafter): the 1-row bits at ~6x the speed.
// --lp-ab "name:key=v...." the --ppl section once per case on one load, dumps in --lp-out/name; keys
// are the --step-opts keys and the --pf-ab keys, each case starting from the command line's options.
// --lp-out DIR per --ppl file and row count, the top-32 log-probs of every scored row as an XLP1 dump
// (DIR/<set>.rowsR.xlp, the `probe model --lp-out` format), for `xenithium-kl compare` or a later --lp-ref.
// --kv-pool N one KV pool of N tokens shared by the slots (0 = fixed per slot, -1 = auto from the free
// VRAM); --kv-scramble 1 hands its pages out shuffled: every gate above must give the same bits.
// --lp-ref F.. reference dumps (`xenithium-kl dump, or an --lp-out of this probe), one per --ppl file in order:
// KL(ref || this) per position bucket on the reference's top k + a tail bucket, exact on this side
// (the full bf16 logits row), plus each engine's ppl and top-1 agreement.
#include <algorithm>
#include <chrono>
#include <functional>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <future>
#include <thread>
#include <sstream>
#include <string>
#include <vector>

#include "xenithium/arch/qwen35/dflash.hpp"
#include "xenithium/arch/qwen35/model.hpp"
#include "xenithium/eval/logprobs.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"  // w4a16_pf_set_a8_shape (--pf-ab key a8s)
#include "xenithium/rt/fabric.hpp"
#include "xenithium/text/tokenizer.hpp"

#include "kv.hpp"
#include "probe.hpp"  // the run-wide flags

// --head-w4 / --pf-a8, --pf-a8-keep into a qwen35 model's options (every qwen35 model command that loads one)
void q35_apply_r26_options(xenithium::arch::qwen35::ModelOptions &mo) {
  mo.head_w4 = g_head_w4 != 0;
  mo.head_rescore = g_head_rescore;
  mo.draft_head = g_draft_head;
  mo.prefill_w4_a8 = g_pf_a8 != 0;
  mo.prefill_a8_keep = g_pf_a8_keep;
  if (mo.head_rescore > 0 || mo.draft_head >= 0)
    std::printf("model35: round-28 switches: two-stage head %s (%d tiles per card), draft rows %s\n", mo.head_rescore > 0 ? "on" : "off",
                mo.head_rescore, mo.draft_head < 0 ? "as the target" : mo.draft_head ? "int4" : "int8");
  if (mo.head_w4 || mo.prefill_w4_a8)
    std::printf("model35: round-26 switches: lm_head %s, prefill activations %s%s%s\n", mo.head_w4 ? "int4 (lm_head.qweight)" : "as served",
                mo.prefill_w4_a8 ? "int8 (W4A8)" : "bf16 (W4A16)", mo.prefill_a8_keep.empty() ? "" : ", W4A16 kept: ",
                mo.prefill_a8_keep.c_str());
}
void dflash_check(xenithium::arch::qwen35::Model &m, const std::string &target_dir, const std::vector<int32_t> &prompt, int n, int dense_bytes,
                  const std::function<void(bool, const std::string &)> &check);  // qwen35_dflash.cpp

namespace {
namespace q35 = xenithium::arch::qwen35;
namespace k = xenithium::kernels;

bool read_chunks(const std::string &path, std::vector<std::vector<int32_t>> &chunks) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  int32_t n = 0;
  if (!f.read(reinterpret_cast<char *>(&n), 4) || n <= 0 || n > 100000) return false;
  chunks.resize(static_cast<size_t>(n));
  for (auto &c : chunks) {
    int32_t len = 0;
    if (!f.read(reinterpret_cast<char *>(&len), 4) || len <= 1 || len > (1 << 24)) return false;
    c.resize(static_cast<size_t>(len));
    if (!f.read(reinterpret_cast<char *>(c.data()), static_cast<std::streamsize>(len) * 4)) return false;
  }
  return true;
}
double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
// Combining marks (\p{M}) in UTF-8 text, the main blocks: U+0300-036F, 0483-0489, 0591-05BD, 0610-061A, 064B-065F,
// 0900-0903.. (Indic: 0900-0DFF counted whole), 1AB0-1AFF, 1DC0-1DFF, 20D0-20FF, FE20-FE2F. A rough count is enough:
// the gate wants zero.
int marks(const std::string &s) {
  int n = 0;
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    uint32_t cp = c;
    size_t len = 1;
    if (c >= 0xF0 && i + 3 < s.size()) cp = ((c & 7u) << 18) | ((s[i + 1] & 63u) << 12) | ((s[i + 2] & 63u) << 6) | (s[i + 3] & 63u), len = 4;
    else if (c >= 0xE0 && i + 2 < s.size()) cp = ((c & 15u) << 12) | ((s[i + 1] & 63u) << 6) | (s[i + 2] & 63u), len = 3;
    else if (c >= 0xC0 && i + 1 < s.size()) cp = ((c & 31u) << 6) | (s[i + 1] & 63u), len = 2;
    n += (cp >= 0x300 && cp <= 0x36F) || (cp >= 0x483 && cp <= 0x489) || (cp >= 0x591 && cp <= 0x5BD) || (cp >= 0x610 && cp <= 0x61A) ||
         (cp >= 0x64B && cp <= 0x65F) || (cp >= 0x900 && cp <= 0xDFF) || (cp >= 0x1AB0 && cp <= 0x1AFF) || (cp >= 0x1DC0 && cp <= 0x1DFF) ||
         (cp >= 0x20D0 && cp <= 0x20FF) || (cp >= 0xFE20 && cp <= 0xFE2F);
    i += len;
  }
  return n;
}
void print_profile(const q35::StepProfile &p) {
  std::printf("      GPU span %.2f ms | embed %.3f norms %.3f GDN %.3f (qkvz %.3f core %.3f out %.3f reduce %.3f) attention %.3f (qkv %.3f prep %.3f "
              "attn %.3f merge %.3f o %.3f reduce %.3f) MLP %.3f (gate|up %.3f swiglu %.3f down %.3f reduce %.3f) head %.3f ms\n",
              p.span / 1e3, p.embed / 1e3, p.norms / 1e3, p.gdn / 1e3, p.gdn_parts[0] / 1e3, p.gdn_parts[1] / 1e3, p.gdn_parts[2] / 1e3, p.gdn_parts[3] / 1e3,
              p.attn / 1e3, p.attn_parts[0] / 1e3, p.attn_parts[1] / 1e3, p.attn_parts[2] / 1e3, p.attn_parts[3] / 1e3, p.attn_parts[4] / 1e3,
              p.attn_parts[5] / 1e3, p.mlp / 1e3, p.mlp_parts[0] / 1e3, p.mlp_parts[1] / 1e3, p.mlp_parts[2] / 1e3, p.mlp_parts[3] / 1e3, p.head / 1e3);
  if (p.moe > 0)  // arch flint
    std::printf("      MoE %.3f ms (router %.3f route %.3f gate_up %.3f down %.3f combine + exchange %.3f)\n", p.moe / 1e3, p.moe_parts[0] / 1e3,
                p.moe_parts[1] / 1e3, p.moe_parts[2] / 1e3, p.moe_parts[3] / 1e3, p.moe_parts[4] / 1e3);
  const double sum = p.embed + p.norms + p.gdn + p.attn + p.mlp + p.moe + p.head;
  std::printf("      launch sum %.2f ms, gaps %.2f ms\n", sum / 1e3, (p.span - sum) / 1e3);
}

std::string rl(int R) { return R == 0 ? std::string("pf") : std::to_string(R); }

// decode step switches "key=v,key=v" (--step-opts, --step-ab cases) into o
bool parse_step_opts(const std::string &list, q35::StepOptions &o) {
  std::stringstream kv(list);
  std::string t;
  while (std::getline(kv, t, ',')) {
    const size_t eq = t.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = t.substr(0, eq);
    const int v = std::stoi(t.substr(eq + 1));
    if (key == "b16") o.wire_bf16 = v != 0;
    else if (key == "wi8") o.wire_i8 = v != 0;  // The int8 wire (rules over b16)
    else if (key == "da8") o.w4_a8 = v != 0;  // The decode int4 GEMMs on int8 activations
    else if (key == "w4i") o.w4_items = v;  // the W4 GEMMs' auto k split target (512)
    else if (key == "w4g") o.w4_items_gu = v;  // gate|up's own target (0 = by the cards: TP1 16384)
    else if (key == "swf") o.fused_swiglu = v != 0;  // swiglu in the decode gate|up epilogue
    else if (key == "head") o.head = v;  // 0 int8, 1 int4, 2 two-stage (rescore)
    else if (key == "graphs") o.graphs = v != 0;  // arch flint: step graphs (1) or eager (0)
    else if (key == "ht") o.head_tiles = v;  // the two-stage head's tiles per card
    else if (key == "aqk") o.attn_qk = v;  // Decode attention q.k (0 bf16, 1 int8, 2 int8 hi + lo)
    else if (key == "aplo") o.attn_p_lo = v != 0;  // Decode attention P hi + lo (1) or hi only (0)
    else if (key == "apf") o.attn_pf = v;  // Decode attention L2 prefetch (-1 auto, 0, 1)
    else if (key == "agrf") o.attn_grf = v;  // Decode attention register file (0 auto, 128, 256)
    else {
      std::fprintf(stderr, "model35: step option %s unknown\n", key.c_str());
      return false;
    }
  }
  return true;
}

// the per-stage GPU split of a profiled prefill (card 0)
static void print_pf_parts(const q35::PfProfile &pp) {
  std::printf("    card 0 GPU ms: attention qkv %.1f prep %.1f attn %.1f o %.1f reduce %.1f | MLP gate|up %.1f swiglu %.1f down %.1f reduce %.1f | "
              "GDN qkvz %.1f out %.1f reduce %.1f | split sums %.1f\n", pp.attn_parts[0], pp.attn_parts[1], pp.attn_parts[2], pp.attn_parts[3], pp.attn_parts[4],
              pp.mlp_parts[0], pp.mlp_parts[1], pp.mlp_parts[2], pp.mlp_parts[3], pp.gdn_parts[0], pp.gdn_parts[1], pp.gdn_parts[2], pp.split);
  double moe = 0;
  for (double v : pp.moe_parts) moe += v;
  if (moe > 0)  // arch flint
    std::printf("    card 0 GPU ms: MoE router %.1f route %.1f gate_up %.1f down %.1f combine %.1f\n", pp.moe_parts[0], pp.moe_parts[1], pp.moe_parts[2],
                pp.moe_parts[3], pp.moe_parts[4]);
}

// onward: one --pf-ab key (prefill options, settable on a loaded model) into o; false = not a prefill key
bool pf_key(const std::string &key, const std::string &val, q35::ModelOptions &o) {
  if (key == "keep") {  // prefill_a8_keep (items joined by '+'; "-" = none)
    o.prefill_a8_keep = val == "-" ? "" : val;
    return true;
  }
  const int v = std::stoi(val);
  if (key == "attn") o.prefill_attn = v;
  else if (key == "rows") o.prefill_attn_rows = v;
  else if (key == "plo") o.prefill_attn_p_lo = v != 0;
  else if (key == "qk") o.prefill_attn_qk = v;  // int8 q.k (0 / 1 / 2)
  else if (key == "sel") o.prefill_select = v;  // Page-selected prefill attention at v pages (load with --pf-select N)
  else if (key == "dq") o.prefill_w4_dequant = v != 0;
  else if (key == "a8") o.prefill_w4_a8 = v != 0;
  else if (key == "a8b") o.prefill_a8_body = v;  // The W4A8 GEMM body (-1 auto, 0 slm = direct = SLM-free; 2: SLM-free on n >= 6,144; 3 = direct up to 128 rows, slm above = auto at TP2)
  else if (key == "split") o.prefill_split = v != 0;
  else if (key == "short") o.prefill_short = v;  // Prefill calls of <= N rows on the decode GEMMs
  else if (key == "pwire") o.prefill_wire_bf16 = v != 0;  // The many-row two-card sums on the bf16 wire
  else if (key == "tier") o.rope_tier_reset = v;  // The YaRN tier reset starts sequences in (-1 = the last)
  else if (key == "send") o.prefill_send = v != 0;
  else if (key == "defer") o.prefill_defer = v != 0;
  else if (key == "fuse") o.prefill_fused_swiglu = v != 0;
  else if (key == "embed") o.prefill_embed_chunk = v != 0;
  else if (key == "chunk") o.prefill_chunk_run = v;
  else return false;
  return true;
}

}  // namespace

int cmd_model35(const std::string &model, const std::string &ppl, const std::string &rows_arg, int generate, int64_t at, int64_t context, int kv_bytes,
                int threads, const std::string &lp_out, const std::string &lp_ref, int pf_chunk, bool pf_slices, bool pf_attn_slices, int pf_attn, bool pf_p_lo, bool pf_w4dq, bool pf_split,
                const std::string &pf_ab, int pf_reps, const std::vector<int> &pf_lens, int spec, int depth, int mtp_kv_bytes, int mtp_dense_bytes,
                int mtp_vocab, const std::string &mtp_dump, int seqs, int seq_tokens, const std::string &seq_depths, const std::string &step_ab,
                const std::string &step_shapes, int step_reps, int profile_card, const std::string &step_opts, long long kv_pool, bool kv_scramble) {
  namespace ev = xenithium::eval;
  std::vector<std::string> files;
  std::vector<std::vector<std::vector<int32_t>>> sets;
  {
    std::stringstream ss(ppl);
    std::string f;
    while (std::getline(ss, f, ','))
      if (!f.empty()) {
        sets.emplace_back();
        if (!read_chunks(f, sets.back())) {
          std::fprintf(stderr, "model35: cannot read token file %s\n", f.c_str());
          return 2;
        }
        files.push_back(f.substr(f.find_last_of('/') + 1));
      }
  }
  std::vector<ev::LpSet> refs;  // --lp-ref, one per token file, checked against the token files before the load
  {
    std::stringstream ss(lp_ref);
    std::string f;
    while (std::getline(ss, f, ','))
      if (!f.empty()) {
        refs.emplace_back();
        std::string err;
        if (!ev::read_lp(f, refs.back(), &err)) {
          std::fprintf(stderr, "model35: %s\n", err.c_str());
          return 2;
        }
      }
    if (!refs.empty() && refs.size() != sets.size()) {
      std::fprintf(stderr, "model35: --lp-ref takes one dump per --ppl file (%zu vs %zu)\n", refs.size(), sets.size());
      return 2;
    }
    for (size_t i = 0; i < refs.size(); ++i) {
      bool ok = refs[i].chunks.size() == sets[i].size();
      for (size_t c = 0; ok && c < sets[i].size(); ++c) {
        ok &= refs[i].chunks[c].size() == sets[i][c].size();
        for (size_t j = 0; ok && j + 1 < sets[i][c].size(); ++j) ok &= refs[i].chunks[c][j].target == sets[i][c][j + 1];
      }
      if (!ok) {
        std::fprintf(stderr, "model35: --lp-ref dump %zu does not line up with %s (chunks, rows or targets differ)\n", i, files[i].c_str());
        return 2;
      }
    }
  }
  std::vector<int> row_list;
  {
    std::stringstream ss(rows_arg);
    std::string t;
    while (std::getline(ss, t, ','))
      if (!t.empty()) row_list.push_back(t == "pf" ? 0 : std::stoi(t));
  }
  bool rows8 = false;  // teacher-forced scoring in 8-row steps (same bits as 1 row, ~3x faster than 4 rows)
  for (int R : row_list) {
    if (R < 0 || R > 8 || (R > q35::Model::kRows && R != 8)) {
      std::fprintf(stderr, "model35: --rows takes 1..4, 8 or pf\n");
      return 2;
    }
    rows8 |= R == 8;
  }
  if (at <= 0) at = 8192;
  int64_t longest = 0;
  for (const auto &s : sets)
    for (const auto &c : s) longest = std::max<int64_t>(longest, static_cast<int64_t>(c.size()));
  // --step-ab shapes SxR (S sequences x R rows each)
  std::vector<std::pair<int, int>> shapes;
  std::vector<char> shape_tree;  // SxRt = each segment a draft tree (side leaves at rows 1 / 3 / 5, then a chain)
  {
    std::stringstream ss(step_shapes.empty() ? std::string("1x1,1x4,2x4") : step_shapes);
    std::string t;
    while (std::getline(ss, t, ','))
      if (!t.empty()) {
        const bool tr = t.back() == 't';
        if (tr) t.pop_back();
        const size_t x = t.find('x');
        const int S = x == std::string::npos ? 1 : std::stoi(t.substr(0, x)), R = std::stoi(x == std::string::npos ? t : t.substr(x + 1));
        const int rmax = g_dflash.empty() ? q35::Model::kRows : g_dflash_wide > 0 ? 16 : 8;  // 16 with --dflash-wide
        if (S < 1 || R < 1 || R > rmax || S * R > xenithium::kernels::kStepRows32 || (tr && (R < 2 || g_dflash.empty()))) {  // 16
          std::fprintf(stderr, "model35: --step-shapes SxR[t], R 1..4 (1..8 with --dflash, 16 with --dflash-wide), S x R <= 32, t (a tree) with --dflash\n");
          return 2;
        }
        shapes.push_back({S, R});
        shape_tree.push_back(tr ? 1 : 0);
      }
  }
  int step_slots = 1;
  if (!step_ab.empty())
    for (const auto &sh : shapes) step_slots = std::max(step_slots, sh.first);
  int64_t pf_calls_tokens = 0;  // --pf-calls: the positions the calls advance through
  std::vector<int> pf_calls;
  {
    std::stringstream ss(g_pf_calls);
    std::string t;
    while (std::getline(ss, t, ','))
      if (!t.empty()) pf_calls.push_back(std::stoi(t)), pf_calls_tokens += pf_calls.back();
    pf_calls_tokens *= static_cast<int64_t>(std::max(1, pf_reps) + 1) * (std::count(pf_ab.begin(), pf_ab.end(), ';') + 1);
    if (!pf_calls.empty()) pf_calls_tokens += at + 64 * (std::max(1, pf_reps) + 2) * 8 + 1024;
  }
  const int64_t need = std::max<int64_t>({pf_calls_tokens, longest + 8 + (spec > 0 ? spec + 8 : 0) + (seqs > 0 ? seq_tokens + 8 : 0), generate > 0 ? at + 4 * generate + 1024 : 0,
                                          step_ab.empty() ? 0 : at + 1024 + static_cast<int64_t>(shapes.size()) * (std::count(step_ab.begin(), step_ab.end(), ';') + 1) * std::max(1, step_reps) * 64 + 512});
  q35::ModelOptions mo;
  // 32-row steps when --step-rows 32 or a step shape holds 17-32 rows (the model's default is 16)
  int max_shape_rows = 0;
  for (const auto &sh : shapes) max_shape_rows = std::max(max_shape_rows, sh.first * sh.second);
  if (g_step_rows == 32 || (!step_ab.empty() && max_shape_rows > 16)) mo.step_rows = 32;
  mo.context = context > 0 ? std::max(context, need) : std::max<int64_t>(8192, need);
  mo.kv = probe_kv(kv_bytes);  // --kv-v4 1: int8 K + int4 V
  mo.attn_select = g_attn_select;  // --attn-select N the caches keep key bounds, steps past N pages select
  mo.prefill_select = g_pf_select;  // --pf-select N prefill calls past N pages select (--pf-ab key sel switches it per case)
  mo.threads = threads;
  if (pf_chunk > 0) mo.prefill_chunk = pf_chunk;
  mo.prefill_slices = pf_slices;
  mo.prefill_attn_slices = pf_attn_slices;
  mo.prefill_attn = pf_attn;
  mo.prefill_attn_p_lo = pf_p_lo;
  mo.prefill_w4_dequant = pf_w4dq;
  mo.prefill_split = pf_split;
  mo.rows8 = rows8;
  q35_apply_r26_options(mo);
  if (seqs > 1) mo.slots = seqs;
  if (step_slots > mo.slots) {
    if (seqs > 1) {
      std::fprintf(stderr, "model35: --step-ab shapes need more slots than --seqs\n");
      return 2;
    }
    mo.slots = step_slots;
  }
  mo.profile_card = profile_card;
  mo.kv_pool_tokens = kv_pool;
  mo.kv_pool_scramble = kv_scramble;
  mo.kv_lend = g_kv_lend, mo.kv_lend_keep = g_kv_lend_keep;  // --kv-lend 1 (needs --kv-pool)
  if (g_lend_check) mo.slots = std::max(mo.slots, 2);
  if (!g_pf_merge.empty()) mo.slots = std::max(mo.slots, static_cast<int>(std::count(g_pf_merge.begin(), g_pf_merge.end(), ',')) + 1);
  if (g_kv_lend && kv_pool > 0)  // the slots the asked pool needs (a pool holds at most slots x context)
    mo.slots = std::max(mo.slots, static_cast<int>(std::min<long long>(16, (kv_pool + mo.context - 1) / mo.context)));
  // --pf-ab too (a prefill A/B with --dflash times the drafter's context passes per chunk)
  if ((spec > 0 || g_dflash_check > 0 || !seq_depths.empty() || !step_ab.empty() || !g_pf_calls.empty() || !pf_ab.empty()) && !g_dflash.empty()) {  // the DFlash 2 drafter (step A/B: 8-row shapes)
    mo.dflash_dir = g_dflash;
    mo.dflash_dense_bytes = g_dflash_bytes;
    mo.dflash_profile = g_dflash_prof;
    if (g_dflash_defer >= 0) mo.dflash_ctx_defer = g_dflash_defer;
    if (g_dflash_cut >= 0) mo.dflash_conf_cut = g_dflash_cut;
    mo.dflash_tree = g_dflash_tree, mo.dflash_tree_cut = g_dflash_tree_cut;
    mo.dflash_tree_seqs = g_dflash_tree_seqs, mo.dflash_tree_temp = g_dflash_tree_temp;
    mo.seg_rows = g_dflash_wide > 0 ? 16 : 0, mo.dflash_tree_wide = g_dflash_tree_wide;
    mo.mtp_vocab = mtp_vocab;
    mo.mtp_from = g_mtp_from;  // both drafters
    mo.mtp_dense_bytes = mtp_dense_bytes;
    mo.mtp_kv = mtp_kv_bytes == 1 ? xenithium::kernels::QsaKv::i8 : xenithium::kernels::QsaKv::bf16;
  } else if (spec > 0) {  // the MTP drafter
    mo.mtp = true;
    mo.mtp_dense_bytes = mtp_dense_bytes;
    mo.mtp_kv = mtp_kv_bytes == 1 ? xenithium::kernels::QsaKv::i8 : xenithium::kernels::QsaKv::bf16;
    mo.mtp_vocab = mtp_vocab;
  }
  if (g_mtp_defer >= 0) mo.mtp_kv_defer = g_mtp_defer;
  const auto loader = xenithium::l0::Loader::open();
  if (g_tp != 1 && g_tp != 2) throw std::runtime_error("model35: --tp 1 or 2");
  xenithium::Fabric fab(loader, tp_devices("model35", false), g_queue_prof);
  if (!g_queue_prof) std::printf("model35: --queue-prof 0: no event profiling on the queues (walls as served; profiled lines read 0)\n");
  mo.loader = &loader;
  q35::Model m(fab, model, mo);
  const int C = m.cards();
  // "a / b" per card (TP2) or "a" (TP1), GiB
  auto per_card = [&](auto per) {
    char b[64];
    if (C > 1) std::snprintf(b, sizeof b, "%.2f / %.2f GiB per card", per(0) / 1073741824.0, per(1) / 1073741824.0);
    else std::snprintf(b, sizeof b, "%.2f GiB on one card (TP1)", per(0) / 1073741824.0);
    return std::string(b);
  };
  if (kv_pool < 0) m.kv_fit(1.5);  // auto (no warm-up here: the kernels' first-use VRAM stays free)
  std::printf("model35: KV %s, %lld tokens (%s)%s\n", m.kv_shared() ? "pool shared by the slots" : "fixed per slot",
              static_cast<long long>(m.kv_pool_tokens()), per_card([&](int c) { return static_cast<double>(m.kv_bytes(c)); }).c_str(),
              kv_scramble ? ", pages handed out shuffled" : "");
  if (g_kv_lend) std::printf("model35: KV lending %s\n", m.kv_lend_note().c_str());
  q35::StepOptions base_step;  // --step-opts, the decode step switches of every section
  if (!parse_step_opts(step_opts, base_step)) return 2;
  m.set_step_options(base_step);
  if (!step_opts.empty()) std::printf("model35: step options %s\n", step_opts.c_str());
  const auto &cfg = m.config();
  if (C > 1)
    std::printf("model35: two-card sums in prefill: %s\n",
                pf_split ? ("split over the cards (pair_split_rows) from " + std::to_string(q35::kPfSplitMinRows) + " rows").c_str() : "chunk mailbox");
  else std::printf("model35: TP1 (one card): every matrix whole, no two-card sums\n");
  pf_attn = m.options().prefill_attn;  // as the model runs it (arch llama pins attention 0)
  std::printf("model35: prefill %s, chunk %d rows\n", pf_slices ? "on decode kernels per 4-row slice (stage 1)" : pf_attn_slices ? "many-row MLP + GDN, attention on 4-row decode slices" : pf_attn == 4 ? "many-row kernels (MLP, GDN, attention: oneDNN GQA, f16)" : pf_attn == 3 ? (pf_p_lo ? "many-row kernels (MLP, GDN, attention: fa kernel)" : "many-row kernels (MLP, GDN, attention: fa kernel, P hi only)") : pf_attn == 2 ? "many-row kernels (MLP, GDN, attention: staged kernel)" : pf_attn == 1 ? "many-row kernels (MLP, GDN, attention: round-12 block kernel)" : "many-row kernels (MLP, GDN, attention: decode kernel per 64-row batch)", mo.prefill_chunk);
  if (pf_w4dq) std::printf("model35: prefill W4 GEMMs dequantised (bf16 w x s, whole-k DPAS sums: a numerics switch)\n");
  std::printf("model35: %s loaded in %.1f s (dense weights %s), context %lld, KV %s\n", model.c_str(), m.load_seconds(),
              per_card([&](int c) { return static_cast<double>(m.card_bytes(c)); }).c_str(), static_cast<long long>(m.capacity()), probe_kv_name(mo.kv));
  xenithium::text::Tokenizer tok;
  std::string terr;
  const bool have_tok = tok.load(model, terr);
  if (!have_tok) std::printf("model35: tokenizer not loaded (%s): no decoded text, no token-file check\n", terr.c_str());
  int rc = 0;
  auto check = [&](bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) rc = 1;
  };
  const int64_t V = cfg.vocab;
  std::vector<float> lg;

  // --lp-ab "name:key=v...." runs the --ppl section once per case on this load (dumps in --lp-out/name). Keys:
  // the decode step options (--step-opts) and the prefill options (--pf-ab); each case starts from the command line's.
  struct LpCase {
    std::string name;
    q35::StepOptions so;
    q35::ModelOptions po;
  };
  std::vector<LpCase> lp_cases;
  {
    std::stringstream cs(g_lp_ab);
    std::string item;
    while (std::getline(cs, item, ';')) {
      if (item.empty()) continue;
      LpCase c{item.substr(0, item.find(':')), base_step, m.options()};
      if (item.find(':') != std::string::npos) {
        std::stringstream kv(item.substr(item.find(':') + 1));
        std::string t;
        while (std::getline(kv, t, ',')) {
          const size_t eq = t.find('=');
          if (eq == std::string::npos) continue;
          if (!pf_key(t.substr(0, eq), t.substr(eq + 1), c.po) && !parse_step_opts(t, c.so)) return 2;
        }
      }
      lp_cases.push_back(c);
    }
  }
  const bool lp_ab = !lp_cases.empty();
  if (!lp_ab) lp_cases.push_back({"", base_step, m.options()});
  const q35::ModelOptions pf_keep = m.options();
  for (const auto &lc : lp_cases) {
  std::string lp_dir = lp_out;
  if (lp_ab) {
    m.set_step_options(lc.so);
    m.set_prefill_options(lc.po);
    if (!lp_out.empty()) {
      lp_dir = lp_out + "/" + lc.name;
      std::filesystem::create_directories(lp_dir);
    }
    std::printf("model35: --lp-ab case %s\n", lc.name.c_str());
  }
  for (size_t si = 0; si < sets.size(); ++si) {
    const auto &chunks = sets[si];
    if (have_tok) {
      int mk = 0;
      for (const auto &c : chunks) mk += marks(tok.decode(c, false));
      for (size_t ci = 0; ci < chunks.size() && ci < 2; ++ci) {  // what the set is
        // tokens [p, p + 160) decoded, newlines shown as \n: at the start and around every 2,048 positions
        for (size_t p = 0; p < chunks[ci].size(); p += 2048) {
          const size_t q0 = p >= 64 ? p - 64 : 0, q1 = std::min(chunks[ci].size(), q0 + 160);
          std::string t = tok.decode(std::vector<int32_t>(chunks[ci].begin() + q0, chunks[ci].begin() + q1), false), e;
          for (char ch : t) e += ch == '\n' ? std::string("\\n") : std::string(1, ch);
          std::printf("  %s chunk %zu (%zu tokens) at %zu: %s\n", files[si].c_str(), ci, chunks[ci].size(), q0, e.c_str());
        }
      }
      // a note, not a gate: the sets are id files of the qwen38 tokenizer; a scorer only needs both engines fed the same ids
      std::printf("  %s: %d combining-mark code points in the decoded text (%s)\n", files[si].c_str(), mk,
                  mk ? "the qwen35 pre-tokenizer may split those spans differently: ids are not its canonical encoding there"
                     : "the qwen38 and qwen35 pre-tokenizers agree on it: the ids are this model's own encoding");
    }
    std::vector<std::vector<double>> nll_by_rows;
    for (int R : row_list) {
      double nll_all = 0;
      int64_t n_all = 0, hits_all = 0, pick_diff = 0, disagree = 0;
      std::vector<double> step_ms, chunk_nll;
      constexpr int kB = 6;
      const int bounds[kB + 1] = {0, 512, 1024, 2048, 4096, 6144, 1 << 30};
      double b_nll[kB] = {};
      int64_t b_n[kB] = {};
      constexpr int kLpK = 32;
      ev::LpSet dump;
      dump.k = kLpK;
      const bool want_dump = !lp_out.empty(), want_kl = si < refs.size();
      ev::KlStats kls({128, 512, 2048, 8192, 32768});
      std::vector<float> lgb[2];  // logits of the step being scored and of the next one
      int cur = 0;
      std::future<void> pending;
      const auto t0 = std::chrono::steady_clock::now();
      for (size_t ci = 0; ci < chunks.size(); ++ci) {
        const auto &tk = chunks[ci];
        const int len = static_cast<int>(tk.size());
        m.reset();
        if (want_dump) dump.chunks.emplace_back();
        double nll = 0;
        int n = 0, hits = 0;
        // the host scoring runs off the step path. Each call fetches its rows' logits (one of two buffers) and
        // device picks in order, then a worker scores the rows in parallel while the next step runs; the results fold in
        // row order, so every sum, dump row and KL row is the serial scorer's.
        struct RowOut {
          int64_t best = 0;
          double tl = 0;
          ev::LpRow rec;
          ev::KlRow kl;
        };
        auto drain = [&] {
          if (pending.valid()) pending.get();
        };
        auto score = [&](int p0, int rows) {
          std::vector<float> &buf = lgb[cur];
          cur ^= 1;
          m.logits(rows, buf);
          auto picks = m.picks(0, rows);
          drain();
          pending = std::async(std::launch::async, [&, p0, rows, picks = std::move(picks)] {
            std::vector<RowOut> out(static_cast<size_t>(rows));
            auto one = [&](int j) {
              const float *l = buf.data() + static_cast<size_t>(j) * V;
              RowOut &o = out[static_cast<size_t>(j)];
              int64_t best = 0;
              for (int64_t i = 1; i < V; ++i)
                if (l[i] > l[best]) best = i;
              o.best = best;
              const int t = p0 + j + 1;
              const int64_t target = t < len ? tk[static_cast<size_t>(t)] : best;
              double se = 0;
              const double mx = l[best];
              for (int64_t i = 0; i < V; ++i) se += std::exp(static_cast<double>(l[i]) - mx);
              const double lse = mx + std::log(se);
              o.tl = static_cast<double>(l[target]) - lse;
              if (want_dump) o.rec = ev::top_logprobs(l, V, kLpK, static_cast<int32_t>(target));
              if (want_kl) {
                const auto &ref = refs[si].chunks[ci][static_cast<size_t>(p0 + j)];
                auto test_lp = [&](int32_t id, bool &exact) {
                  exact = true;
                  return static_cast<double>(l[id]) - lse;
                };
                o.kl = ev::kl_coarse(ref, test_lp, static_cast<int32_t>(best));
              }
            };
            std::vector<std::thread> th;
            for (int j = 1; j < rows; ++j) th.emplace_back(one, j);
            one(0);
            for (auto &x : th) x.join();
            for (int j = 0; j < rows; ++j) {
              RowOut &o = out[static_cast<size_t>(j)];
              const int t = p0 + j + 1;
              pick_diff += picks[static_cast<size_t>(j)] != o.best;
              nll -= o.tl;
              if (want_dump) dump.chunks.back().push_back(std::move(o.rec));
              if (want_kl) kls.add(p0 + j, o.kl, refs[si].chunks[ci][static_cast<size_t>(p0 + j)].target_lp, o.tl);
              for (int b = 0; b < kB; ++b)
                if (p0 + j >= bounds[b] && p0 + j < bounds[b + 1]) b_nll[b] -= o.tl, ++b_n[b];
              ++n;
              hits += t < len && o.best == tk[static_cast<size_t>(t)];
            }
          });
        };
        if (R == 0) {
          const auto ts = std::chrono::steady_clock::now();
          m.prefill(tk.data(), len, [&](int64_t p0, int rows) { score(static_cast<int>(p0), rows); });
          drain();
          step_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
        } else {
          for (int p0 = 0; p0 < len; p0 += R) {
            const int rows = std::min(R, len - p0);
            const auto ts = std::chrono::steady_clock::now();
            m.step(&tk[static_cast<size_t>(p0)], rows);
            step_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
            score(p0, rows);
            m.accept(rows);
          }
          drain();
        }
        if (C > 1) disagree += m.residual(0, 1) != m.residual(1, 1);
        chunk_nll.push_back(nll);
        if (chunks.size() <= 8) std::printf("  %s rows %s chunk %zu: %d tokens scored, ppl %.4f, greedy next-token hits %d/%d\n", files[si].c_str(), rl(R).c_str(), ci, n,
                                            std::exp(nll / n), hits, n);
        nll_all += nll;
        n_all += n;
        hits_all += hits;
      }
      const double ppl_v = std::exp(nll_all / static_cast<double>(n_all));
      const uint32_t we = m.wire_errors();
      std::printf("ppl %s (rows %s): %.4f over %lld tokens (%zu chunks, %.1f s, step median %.2f ms host), greedy hits %lld/%lld; device pick != host argmax "
                  "in %lld rows; cards disagree after %lld chunks; wire errors %u\n",
                  files[si].c_str(), rl(R).c_str(), ppl_v, static_cast<long long>(n_all), chunks.size(),
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), median(step_ms), static_cast<long long>(hits_all),
                  static_cast<long long>(n_all), static_cast<long long>(pick_diff), static_cast<long long>(disagree), we);
      {
        std::string bl = "  ppl by position:";
        for (int b = 0; b < kB; ++b)
          if (b_n[b]) {
            char t[96];
            std::snprintf(t, sizeof t, " [%d, %s) %.3f (%lld)", bounds[b], b + 1 < kB ? std::to_string(bounds[b + 1]).c_str() : "end",
                          std::exp(b_nll[b] / b_n[b]), static_cast<long long>(b_n[b]));
            bl += t;
          }
        std::printf("%s\n", bl.c_str());
      }
      char line[256];
      std::snprintf(line, sizeof line, "%s rows %s: ppl %.4f finite, device picks == host argmax, both cards identical, no mailbox timeout", files[si].c_str(),
                    rl(R).c_str(), ppl_v);
      check(std::isfinite(ppl_v) && pick_diff == 0 && disagree == 0 && we == 0, line);
      nll_by_rows.push_back(chunk_nll);
      if (want_dump) {
        const std::string stem = files[si].substr(0, files[si].find_last_of('.'));
        const std::string path = lp_dir + "/" + stem + ".rows" + rl(R) + ".xlp";
        std::string err;
        check(ev::write_lp(path, dump, &err), "log-probs (top " + std::to_string(kLpK) + ") -> " + path + (err.empty() ? "" : ": " + err));
      }
      if (want_kl)
        std::printf("KL(reference || xenithium) %s rows %s, ref top %d + tail bucket (a lower bound on the full-vocab KL); "
                    "ppl columns use each engine's own last-row target\n%s",
                    files[si].c_str(), rl(R).c_str(), refs[si].k, kls.table().c_str());
    }
    for (size_t i = 1; i < nll_by_rows.size(); ++i) {
      const std::string what = files[si] + ": per-chunk nll at rows " + rl(row_list[i]) + " == rows " + rl(row_list[0]) + " bit for bit";
      if (row_list[i] != 0 && row_list[0] != 0) check(nll_by_rows[i] == nll_by_rows[0], what);
      else if (pf_slices) check(nll_by_rows[i] == nll_by_rows[0], what + " (prefill on decode slices)");
      else std::printf("note: %s: %s (many-row prefill: fp32 summation order; judge by ppl / KL)\n", what.c_str(), nll_by_rows[i] == nll_by_rows[0] ? "yes" : "no");
    }
  }
  }
  if (lp_ab) {
    m.set_step_options(base_step);
    m.set_prefill_options(pf_keep);
  }

  // ---- prefill A/B on one load: --pf-ab "name:key=val...;name..." ----
  // keys: a8s (the W4A8 GEMM shape, w4a16_pf_set_a8_shape: -1 by m, 0 = 32 x 2 R 8, 5 = 24 x 4 HALF), attn (0 batch / 1 block / 2 staged / 3 fa), rows (staged / fa rows per work-group), plo (fa: P lo half 1 / 0), qk (fa: q.k 0 bf16 / 1 int8 q / 2 int8 hi + lo q), sel (page-selected prefill at N pages, 0 dense; load with --pf-select), dq (prefill W4 GEMMs dequantised 1 / exact 0), short, split, send, defer, fuse (swiglu in
  // gate|up), embed (one gather per chunk), chunk (rows per chunk at run time, <= --pf-chunk). Each case starts from the
  // command line's options. Per token file (its first chunk): --pf-reps rounds over the cases in turn (alternating per
  // case, the same load), median ms; the last slice's logits vs the first case's (bits, max |diff|, KL of the last row).
  struct PfCase {
    std::string name;
    q35::ModelOptions o;
    int a8s = -1;  // key a8s: the W4A8 GEMM shape (w4a16_pf_set_a8_shape: -1 by m, 0 = 32 x 2, 5 = 24 x 4 HALF...)
  };
  std::vector<PfCase> pf_cases;
  {
    std::stringstream cs(pf_ab.empty() ? std::string("base") : pf_ab);
    std::string item;
    while (std::getline(cs, item, ';')) {
      if (item.empty()) continue;
      PfCase c{item, m.options()};
      const size_t colon = item.find(':');
      c.name = item.substr(0, colon);
      if (colon != std::string::npos) {
        std::stringstream kv(item.substr(colon + 1));
        std::string t;
        while (std::getline(kv, t, ',')) {
          const size_t eq = t.find('=');
          if (eq == std::string::npos) continue;
          const std::string key = t.substr(0, eq);
          if (key == "a8s") {  // a kernel-wide bench setting, set before each of the case's calls
            c.a8s = std::stoi(t.substr(eq + 1));
            continue;
          }

          if (!pf_key(key, t.substr(eq + 1), c.o)) {
            std::fprintf(stderr, "model35: --pf-ab key %s unknown\n", key.c_str());
            return 2;
          }
        }
      }
      pf_cases.push_back(c);
    }
  }
  // ---- --lend-check 1 P2): the kept oneDNN layouts in the pool's free pages. The same prompt's
  // logits with the layouts lent and filled, given back under KV pressure (the per-call repack), lent again: equal bits;
  // the counts at each state; a prefill's ms kept vs repacked. ----
  if (g_lend_check && !sets.empty()) {
    const auto &tk = sets[0][0];
    const int len = static_cast<int>(std::min<size_t>(tk.size(), 4096)), last = (len - 1) % 4 + 1;
    auto run = [&](std::vector<float> &lg) {
      m.use_slot(0);
      m.reset();
      const auto ts = std::chrono::steady_clock::now();
      m.prefill(tk.data(), len);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
      m.logits(last, lg);
      return ms;
    };
    std::printf("KV lending %s\n", m.kv_lend_note().c_str());
    auto st = m.lend_stats();
    check(st.on && st.lent > 0, "KV lending: on, " + std::to_string(st.lent) + " of " + std::to_string(st.items) + " wide matrices have a region in the pool's free pages");
    const int lent0 = st.lent;
    std::vector<float> a, b, c2, d2, e2;
    const double t_fill = run(a);  // the layouts repacked into their regions
    st = m.lend_stats();
    check(st.filled == st.lent, "KV lending: after one prefill every lent layout is filled (" + std::to_string(st.filled) + " of " + std::to_string(st.lent) + ")");
    double t_kept = 1e30;
    for (int r = 0; r < 3; ++r) t_kept = std::min(t_kept, run(b));
    // KV pressure: slot 1 takes every page the pool has left
    m.use_slot(0);
    m.reset();
    const int64_t all = std::min<int64_t>(m.kv_free_tokens(), m.capacity());
    bool got = m.kv_reserve(1, all);
    if (m.lend_stats().lent == lent0) {  // (a pool of several contexts: slot 0 takes its share too, then lets go)
      got = got && m.kv_reserve(0, std::min<int64_t>(m.kv_free_tokens(), m.capacity()));
      m.reset();
    }
    st = m.lend_stats();
    check(got && st.lent < lent0 && st.raises > 0,
          "KV lending: a reservation of " + std::to_string(all) + " tokens gets the lent pages (" + std::to_string(st.lent) + " layouts stay, limit " +
              std::to_string(st.limit_tokens) + " tokens)");
    const int lent1 = st.lent;
    bool room = m.kv_reserve(0, len + 64);
    if (!room) {  // (a pool of one context: give the prompt its pages back)
      m.kv_release(1);
      m.kv_reserve(1, std::max<int64_t>(all - len - 4096, 64));
      room = m.kv_reserve(0, len + 64);
    }
    double t_repack = 1e30;
    for (int r = 0; r < 3 && room; ++r) t_repack = std::min(t_repack, run(c2));
    st = m.lend_stats();
    check(room && st.lent <= lent1, "KV lending: the prompt again with the layouts given back (" + std::to_string(st.lent) + " lent)");
    // the pressure gone: lent again while idle, filled by the idle work
    m.kv_release(1);
    m.use_slot(0);
    m.reset();
    int calls = 0;
    while (m.lend_idle(16) > 0 && calls < 1000) ++calls;
    st = m.lend_stats();
    check(st.lent == lent0 && st.filled == st.lent && st.relends > 0,
          "KV lending: lent again while idle (" + std::to_string(st.lent) + " of " + std::to_string(st.items) + ", " + std::to_string(st.filled) + " filled by " +
              std::to_string(calls) + " idle calls)");
    const double t_again = run(d2);
    check(a == b && a == c2 && a == d2 && !a.empty(), "KV lending: the prompt's last-slice logits are the same bits filled / kept / given back / lent again");
    std::printf("KV lending, %d-token prefill (native xenithium, %s): filling %.1f ms, kept %.1f ms, given back (per-call repack) %.1f ms, lent again %.1f ms\n", len,
                m.cards() == 1 ? "one B70" : "two B70s", t_fill, t_kept, t_repack, t_again);
    check(m.wire_errors() == 0, "KV lending: no mailbox timeout");
  }
  if (!pf_ab.empty() && pf_calls.empty()) {
    struct Case {
      std::string name;
      q35::ModelOptions o;
    };
    std::vector<Case> cases;
    std::stringstream cs(pf_ab);
    std::string item;
    while (std::getline(cs, item, ';')) {
      if (item.empty()) continue;
      Case c{item, m.options()};
      const size_t colon = item.find(':');
      c.name = item.substr(0, colon);
      if (colon != std::string::npos) {
        std::stringstream kv(item.substr(colon + 1));
        std::string t;
        while (std::getline(kv, t, ',')) {
          const size_t eq = t.find('=');
          if (eq == std::string::npos) continue;
          const std::string key = t.substr(0, eq);
          if (key == "a8s") continue;  // set per case below
          if (!pf_key(key, t.substr(eq + 1), c.o)) {
            std::fprintf(stderr, "model35: --pf-ab key %s unknown\n", key.c_str());
            return 2;
          }
        }
      }
      cases.push_back(c);
    }
    const q35::ModelOptions keep = m.options();
    // --pf-lens: prefixes of each file's first chunk (prompt lengths), else the whole chunk
    std::vector<std::pair<size_t, int>> runs;
    for (size_t si = 0; si < sets.size(); ++si) {
      if (pf_lens.empty()) runs.push_back({si, static_cast<int>(sets[si][0].size())});
      for (int L : pf_lens)
        if (L <= static_cast<int>(sets[si][0].size())) runs.push_back({si, L});
    }
    for (const auto &[si, len] : runs) {
      const auto &tk = sets[si][0];
      const int last = (len - 1) % 4 + 1;
      std::vector<std::vector<double>> ms(cases.size());
      std::vector<std::vector<float>> lgs(cases.size());
      for (int rep = 0; rep < std::max(1, pf_reps); ++rep)
        for (size_t ci = 0; ci < cases.size(); ++ci) {
          m.set_prefill_options(cases[ci].o);
          xenithium::kernels::w4a16_pf_set_a8_shape(pf_cases[ci].a8s);
          m.reset();
          const auto ts = std::chrono::steady_clock::now();
          m.prefill(tk.data(), len);
          ms[ci].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
          if (rep == 0) m.logits(last, lgs[ci]);
        }
      std::printf("prefill A/B %s chunk 0 (%d tokens), %d rounds (native xenithium, %s), last-slice logits vs case 0:\n", files[si].c_str(), len,
                  std::max(1, pf_reps), m.cards() == 1 ? "one B70" : "two B70s");
      std::printf("  %-28s %10s %9s %8s %9s %10s %10s\n", "case", "median ms", "tok/s", "vs 0 %", "bits", "max |d|", "KL last");
      for (size_t ci = 0; ci < cases.size(); ++ci) {
        const double md = median(ms[ci]), m0 = median(ms[0]);
        const auto &a = lgs[0], &b = lgs[ci];
        double mx = 0;
        for (size_t i = 0; i < a.size(); ++i) mx = std::max(mx, static_cast<double>(std::fabs(a[i] - b[i])));
        // KL(case 0 || case) of the last row
        const float *la = a.data() + static_cast<size_t>(last - 1) * V, *lb = b.data() + static_cast<size_t>(last - 1) * V;
        double ma = la[0], mb = lb[0];
        for (int64_t i = 1; i < V; ++i) ma = std::max(ma, static_cast<double>(la[i])), mb = std::max(mb, static_cast<double>(lb[i]));
        double sa = 0, sb = 0;
        for (int64_t i = 0; i < V; ++i) sa += std::exp(la[i] - ma), sb += std::exp(lb[i] - mb);
        const double za = ma + std::log(sa), zb = mb + std::log(sb);
        double kl = 0;
        for (int64_t i = 0; i < V; ++i) {
          const double pa = la[i] - za;
          kl += std::exp(pa) * (pa - (lb[i] - zb));
        }
        std::printf("  %-28s %10.1f %9.0f %+8.2f %9s %10.4g %10.3g\n", cases[ci].name.c_str(), md, len / (md / 1e3), (m0 / md - 1) * 100,
                    a == b ? "same" : "differ", mx, kl);
      }
      if (g_pf_ab_prof)  // --pf-ab-prof 1: one profiled prefill per case (both cards synced per stage)
        for (size_t ci = 0; ci < cases.size(); ++ci) {
          m.set_prefill_options(cases[ci].o);
          xenithium::kernels::w4a16_pf_set_a8_shape(pf_cases[ci].a8s);
          m.set_profile(true);
          m.reset();
          m.prefill(tk.data(), len);
          m.set_profile(false);
          const auto &pp = m.pf_profile();
          std::printf("  %s profiled (synced per stage) %.1f ms: embed %.1f norms %.1f GDN %.1f attention %.1f MLP %.1f head %.1f ms\n", cases[ci].name.c_str(),
                      pp.total, pp.embed, pp.norms, pp.gdn, pp.attn, pp.mlp, pp.head);
          print_pf_parts(pp);
        }
    }
    m.set_prefill_options(keep);
    xenithium::kernels::w4a16_pf_set_a8_shape(-1);
    if (g_kv_lend) {
      const auto st = m.lend_stats();
      std::printf("prefill A/B: KV lending at the end: %d of %d layouts lent, %d filled, KV limit %lld tokens, %llu given back\n", st.lent, st.items, st.filled,
                  static_cast<long long>(st.limit_tokens), static_cast<unsigned long long>(st.given_back));
    }
    check(m.wire_errors() == 0, "prefill A/B: no mailbox timeout");
  }

  // ---- --pf-calls: prefill calls of R rows at --at (what a 9-63-row call costs on the many-row path) ----
  if (!pf_calls.empty()) {
    std::vector<int32_t> corpus;
    for (const auto &s : sets)
      for (const auto &c : s) corpus.insert(corpus.end(), c.begin(), c.end());
    if (corpus.empty()) corpus = {760, 3365, 314, 279};
    std::vector<int32_t> fill;
    while (static_cast<int64_t>(fill.size()) < at) fill.insert(fill.end(), corpus.begin(), corpus.end());
    fill.resize(static_cast<size_t>(at));
    const q35::ModelOptions keep = m.options();
    m.use_slot(0);
    m.reset();
    m.prefill(fill.data(), static_cast<int>(fill.size()));
    size_t cur = 0;
    std::vector<int32_t> tk;
    auto next_tokens = [&](int R) {
      tk.resize(static_cast<size_t>(R));
      for (auto &t : tk) t = corpus[cur++ % corpus.size()];
    };
    const int reps = std::max(1, pf_reps);
    // the decode steps first (1xR shapes the model's segments take), so both run near the same position
    std::vector<std::pair<int, double>> dec;
    for (const auto &[S, R] : shapes) {
      if (S != 1 || R > m.seg_rows_max()) continue;
      std::vector<double> w;
      for (int i = 0; i < 8 * reps; ++i) {
        next_tokens(R);
        std::vector<q35::Model::SeqRows> segs{q35::Model::SeqRows{0, tk.data(), R}};
        const auto ts = std::chrono::steady_clock::now();
        m.verify_batch(segs);
        w.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
        m.accept_batch(std::vector<int>{1});
      }
      dec.push_back({R, median(w)});
    }
    std::vector<std::vector<std::vector<double>>> ms(pf_cases.size(), std::vector<std::vector<double>>(pf_calls.size()));
    const int64_t p0 = m.position(0);
    for (int rep = 0; rep < reps + 1; ++rep)  // = warm-up (each R's first call builds its shapes)
      for (size_t ri = 0; ri < pf_calls.size(); ++ri)
        for (size_t ci = 0; ci < pf_cases.size(); ++ci) {
          m.set_prefill_options(pf_cases[ci].o);
          xenithium::kernels::w4a16_pf_set_a8_shape(pf_cases[ci].a8s);
          next_tokens(pf_calls[ri]);
          const auto ts = std::chrono::steady_clock::now();
          m.prefill(tk.data(), pf_calls[ri]);
          if (rep > 0) ms[ci][ri].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
        }
    std::printf("\nprefill calls at position %lld -> %lld (native xenithium, %s), wall ms per call, median of %d:\n", static_cast<long long>(p0),
                static_cast<long long>(m.position(0)), m.cards() == 1 ? "one B70" : "two B70s", reps);
    std::printf("  | rows |");
    for (const auto &c : pf_cases) std::printf(" %s |", c.name.c_str());
    std::printf("\n  | --- |");
    for (size_t ci = 0; ci < pf_cases.size(); ++ci) std::printf(" --- |");
    std::printf("\n");
    for (size_t ri = 0; ri < pf_calls.size(); ++ri) {
      std::printf("  | %d |", pf_calls[ri]);
      for (size_t ci = 0; ci < pf_cases.size(); ++ci) std::printf(" %.3f |", median(ms[ci][ri]));
      std::printf("\n");
    }
    for (const auto &[R, w] : dec) std::printf("  decode step 1x%d (verify_batch, accept 1) near position %lld: %.3f ms\n", R, static_cast<long long>(p0), w);
    m.set_prefill_options(pf_cases[0].o);
    xenithium::kernels::w4a16_pf_set_a8_shape(pf_cases[0].a8s);
    for (int R : pf_calls) {  // one profiled call per R (both cards synced per stage)
      next_tokens(R);
      m.set_profile(true);
      m.prefill(tk.data(), R);
      m.set_profile(false);
      const auto &pp = m.pf_profile();
      std::printf("  %s %d rows profiled (synced per stage) %.2f ms: embed %.2f norms %.2f GDN %.2f attention %.2f MLP %.2f head %.2f ms\n",
                  pf_cases[0].name.c_str(), R, pp.total, pp.embed, pp.norms, pp.gdn, pp.attn, pp.mlp, pp.head);
      print_pf_parts(pp);
    }
    m.set_prefill_options(keep);
    xenithium::kernels::w4a16_pf_set_a8_shape(-1);
    check(m.wire_errors() == 0, "prefill calls: no mailbox timeout");
  }

  // ---- one prefill call for several slots (--pf-merge N,N..) ----
  // Slot i holds --at + 7 i tokens of history (so the slots stand at different positions, off the 64-row scan grid),
  // then its prompt call of N_i rows: alone (Model:prefill_begin of the slot) and as a member of one call for all
  // slots. Gate: every slot's first pick, logits row, first drafts, the keys of its pages and 6 greedy 1-row steps
  // (tokens and logits) are equal bit for bit; the merged call in parts of 16 layers likewise. Then the walls.
  if (!g_pf_merge.empty()) {
    std::vector<int> ns;
    {
      std::stringstream ss(g_pf_merge);
      std::string t;
      while (std::getline(ss, t, ','))
        if (!t.empty()) ns.push_back(std::stoi(t));
    }
    const int K = static_cast<int>(ns.size());
    if (K < 2 || K > m.prefill_merge_max() || K > m.slots()) {
      std::fprintf(stderr, "model35: --pf-merge takes 2..%d row counts (this load merges up to %d sequences in %d rows)\n", std::min(m.slots(), m.prefill_merge_max()),
                   m.prefill_merge_max(), m.prefill_merge_rows());
      return 2;
    }
    std::vector<int32_t> corpus;
    for (const auto &s : sets)
      for (const auto &c : s) corpus.insert(corpus.end(), c.begin(), c.end());
    if (corpus.empty()) corpus = {760, 3365, 314, 279};
    size_t cur = 0;
    auto take = [&](size_t n) {
      std::vector<int32_t> v(n);
      for (auto &t : v) t = corpus[cur++ % corpus.size()];
      return v;
    };
    std::vector<std::vector<int32_t>> hist(static_cast<size_t>(K)), prm(static_cast<size_t>(K));
    const int64_t base = g_pf_merge_at >= 0 ? g_pf_merge_at : at;  // (0: new prompts, every slot at position 0)
    for (int i = 0; i < K; ++i) hist[static_cast<size_t>(i)] = take(static_cast<size_t>(base + (base > 0 ? 7 * i : 0))), prm[static_cast<size_t>(i)] = take(static_cast<size_t>(ns[static_cast<size_t>(i)]));
    const q35::ModelOptions keep = m.options();
    if (!pf_cases.empty()) {
      m.set_prefill_options(pf_cases[0].o);
      xenithium::kernels::w4a16_pf_set_a8_shape(pf_cases[0].a8s);
    }
    const int L = cfg.layers, T = 6;
    int attn_layer = 0;
    while (attn_layer < L && !cfg.full_attention[static_cast<size_t>(attn_layer)]) ++attn_layer;
    struct Out {
      int32_t pick = -1;
      std::vector<float> lg, step_lg;
      std::vector<int32_t> drafts, toks;
      std::vector<int8_t> keys;
      std::vector<float> ks;
    };
    auto history = [&](int i) {
      m.use_slot(i);
      m.reset();
      if (!hist[static_cast<size_t>(i)].empty()) m.prefill(hist[static_cast<size_t>(i)].data(), static_cast<int>(hist[static_cast<size_t>(i)].size()));
    };
    auto first = [&](int i, Out &o) {  // the slot's first token and its logits (before any slot's drafts or steps: they reuse the head)
      m.use_slot(i);
      const int last = m.prefill_last_row();
      o.pick = m.picks(0, last + 1)[static_cast<size_t>(last)];
      std::vector<float> all;
      m.logits(last + 1, all);
      o.lg.assign(all.begin() + static_cast<std::ptrdiff_t>(static_cast<int64_t>(last) * V), all.begin() + static_cast<std::ptrdiff_t>(static_cast<int64_t>(last + 1) * V));
    };
    auto rest = [&](int i, Out &o) {
      m.use_slot(i);
      if (m.draft_depth_max() > 0) o.drafts = m.draft_after_prefill(o.pick, std::min(3, m.draft_depth_max()));
      int32_t tk = o.pick;
      std::vector<float> one;
      for (int t = 0; t < T; ++t) {
        m.verify_batch(std::vector<q35::Model::SeqRows>{q35::Model::SeqRows{i, &tk, 1}});
        tk = m.picks(0, 1)[0];
        m.logits(1, one);
        o.step_lg.insert(o.step_lg.end(), one.begin(), one.end());
        o.toks.push_back(tk);
        m.accept_batch(std::vector<int>{1});
      }
      if (m.options().kv == xenithium::kernels::QsaKv::i8) {
        const int pages = static_cast<int>((m.position(i) + 63) / 64);
        std::vector<int8_t> kk;
        std::vector<float> ss;
        for (int c = 0; c < m.cards(); ++c) {
          const int hkv = m.read_keys(i, attn_layer, c, pages, kk, ss);
          // (the last page's rows past the position are whatever the pool page held before: not the sequence's)
          for (int h = 0; h < hkv; ++h)
            for (int64_t t = m.position(i) - static_cast<int64_t>(pages - 1) * 64; t < 64; ++t) {
              const size_t row = (static_cast<size_t>(pages - 1) * hkv + h) * 64 + static_cast<size_t>(t);
              std::fill(kk.begin() + static_cast<long>(row * 256), kk.begin() + static_cast<long>((row + 1) * 256), int8_t{0});
              ss[row] = 0;
            }
          o.keys.insert(o.keys.end(), kk.begin(), kk.end());
          o.ks.insert(o.ks.end(), ss.begin(), ss.end());
        }
      }
    };
    auto run_all = [&](int layers) {
      while (!m.prefill_advance(layers)) {
      }
    };
    using clk = std::chrono::steady_clock;
    auto ms_since = [](clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); };
    const int reps = std::max(1, pf_reps);
    std::vector<Out> ref(static_cast<size_t>(K)), got(static_cast<size_t>(K)), parts(static_cast<size_t>(K));
    std::vector<double> w_alone, w_merged;
    for (int rep = 0; rep < reps + 1; ++rep) {  // rep 0 = warm-up and the gate's outputs
      double alone = 0;
      for (int i = 0; i < K; ++i) {
        history(i);
        const auto t0 = clk::now();
        m.prefill_begin(prm[static_cast<size_t>(i)].data(), ns[static_cast<size_t>(i)], -1);
        run_all(L);
        alone += ms_since(t0);
        if (rep == 0) first(i, ref[static_cast<size_t>(i)]), rest(i, ref[static_cast<size_t>(i)]);
      }
      for (int i = 0; i < K; ++i) history(i);
      std::vector<q35::Model::PrefillSeg> segs(static_cast<size_t>(K));
      for (int i = 0; i < K; ++i) segs[static_cast<size_t>(i)] = q35::Model::PrefillSeg{i, prm[static_cast<size_t>(i)].data(), ns[static_cast<size_t>(i)], -1};
      const auto t0 = clk::now();
      m.prefill_begin(segs.data(), K);
      run_all(L);
      const double merged = ms_since(t0);
      if (rep == 0) {
        for (int i = 0; i < K; ++i) first(i, got[static_cast<size_t>(i)]);
        for (int i = 0; i < K; ++i) rest(i, got[static_cast<size_t>(i)]);
        for (int i = 0; i < K; ++i) history(i);
        m.prefill_begin(segs.data(), K);
        run_all(16);
        for (int i = 0; i < K; ++i) first(i, parts[static_cast<size_t>(i)]);
        for (int i = K - 1; i >= 0; --i) rest(i, parts[static_cast<size_t>(i)]);  // the first drafts in another order
      } else {
        w_alone.push_back(alone), w_merged.push_back(merged);
      }
    }
    auto same = [](const Out &a, const Out &b) {
      auto eq = [](const std::vector<float> &x, const std::vector<float> &y) { return x.size() == y.size() && (x.empty() || std::memcmp(x.data(), y.data(), x.size() * 4) == 0); };
      std::string d;
      if (a.pick != b.pick) d += " pick";
      if (!eq(a.lg, b.lg)) d += " logits";
      if (a.drafts != b.drafts) d += " drafts";
      if (a.toks != b.toks) d += " step-tokens";
      if (!eq(a.step_lg, b.step_lg)) d += " step-logits";
      if (a.keys != b.keys || !eq(a.ks, b.ks)) d += " keys";
      return d;
    };
    std::string rows_text;
    for (int i = 0; i < K; ++i) rows_text += (i ? " + " : "") + std::to_string(ns[static_cast<size_t>(i)]);
    for (int i = 0; i < K; ++i) {
      const std::string d1 = same(ref[static_cast<size_t>(i)], got[static_cast<size_t>(i)]), d2 = same(ref[static_cast<size_t>(i)], parts[static_cast<size_t>(i)]);
      check(d1.empty() && d2.empty(), "merged prefill call (" + rows_text + " rows): slot " + std::to_string(i) + " at " +
                                          std::to_string(hist[static_cast<size_t>(i)].size()) + " == its own call bit for bit: first pick, logits, " +
                                          std::to_string(ref[static_cast<size_t>(i)].drafts.size()) + " first drafts, " + std::to_string(T) + " steps, " +
                                          std::to_string(ref[static_cast<size_t>(i)].keys.size()) + " key bytes" + (d1.empty() ? "" : " (differs:" + d1 + ")") +
                                          (d2.empty() ? "; in parts of 16 layers too" : " (in parts differs:" + d2 + ")"));
    }
    const double a = median(w_alone), g = median(w_merged);
    std::printf("merged prefill call, %s rows at position %lld: alone %.2f ms in all, one call %.2f ms (%+.1f %%), median of %d; rope delta %lld%s\n", rows_text.c_str(), static_cast<long long>(base), a, g,
                100.0 * (g / a - 1.0), reps, static_cast<long long>(m.rope_delta(0)), m.rope_delta(0) ? " (a YaRN tier: the segments' own rope rows)" : "");
    m.set_prefill_options(keep);
    xenithium::kernels::w4a16_pf_set_a8_shape(-1);
    check(m.wire_errors() == 0, "merged prefill call: no mailbox timeout");
  }

  // ---- prefill: timing with the head on the last slice, then the continuation gate ----
  if (g_pf_checks && std::find(row_list.begin(), row_list.end(), 0) != row_list.end()) {
    const int SR = m.seg_rows_max();  // the decode side in 8-row steps with --rows 8 (same bits as 4)
    for (size_t si = 0; si < sets.size(); ++si) {
      const auto &tk = sets[si][0];
      const int len = static_cast<int>(tk.size());
      std::vector<double> ms;
      for (int rep = 0; rep < 3; ++rep) {
        m.reset();
        const auto ts = std::chrono::steady_clock::now();
        m.prefill(tk.data(), len);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
      }
      std::printf("prefill %s chunk 0: %d tokens in chunks of %d rows: %.1f ms (median of 3, head on the last slice) = %.0f tokens/s\n", files[si].c_str(), len,
                  m.prefill_chunk(), median(ms), len / (median(ms) / 1e3));
      {  // one more, profiled (both cards synced after every stage)
        m.reset();
        m.set_profile(true);
        m.prefill(tk.data(), len);
        m.set_profile(false);
        const auto &pp = m.pf_profile();
        std::printf("  profiled prefill (synced per stage) %.1f ms: embed %.1f norms %.1f GDN %.1f attention %.1f MLP %.1f head %.1f ms\n", pp.total, pp.embed,
                    pp.norms, pp.gdn, pp.attn, pp.mlp, pp.head);
        print_pf_parts(pp);
      }
      const int gen = 16;
      std::vector<int32_t> a, b;
      for (int path = 0; path < 2; ++path) {
        m.reset();
        int last = 0;
        if (path == 0) {
          m.prefill(tk.data(), len);
          last = (len - 1) % 4 + 1;
        } else {
          for (int p0 = 0; p0 < len; p0 += SR) {
            last = std::min(SR, len - p0);
            m.step(&tk[static_cast<size_t>(p0)], last);
            m.accept(last);
          }
        }
        auto &out = path == 0 ? a : b;
        int32_t next = m.picks(0, last).back();
        for (int i = 0; i < gen; ++i) {
          out.push_back(next);
          m.step(&next, 1);
          m.accept(1);
          next = m.picks(0, 1)[0];
        }
      }
      const std::string what = files[si] + ": 16 greedy tokens after a prefill == after " + std::to_string(SR) + "-row steps over the same " + std::to_string(len) + "-token prompt";
      if (pf_slices) {
        check(a == b, what);
      } else {
        int same = 0;
        while (same < gen && a[static_cast<size_t>(same)] == b[static_cast<size_t>(same)]) ++same;
        std::printf("note: %s: first %d of %d identical (many-row prefill: token identity is not its gate)\n", what.c_str(), same, gen);
      }
    }
    check(m.wire_errors() == 0, "prefill: no mailbox timeout");
  }

  if (generate > 0) {
    std::vector<int32_t> prompt;
    if (!sets.empty()) prompt = sets[0][0];
    else if (have_tok) prompt = tok.encode("The history of the printing press begins");
    else prompt = {760, 3365, 314, 279};
    m.reset();
    for (size_t p0 = 0; p0 < prompt.size(); p0 += 4) {
      const int rows = static_cast<int>(std::min<size_t>(4, prompt.size() - p0));
      m.step(&prompt[p0], rows);
      m.accept(rows);
    }
    std::vector<int32_t> out;
    // the last prompt step's last row holds the first pick
    int32_t next = m.picks(0, static_cast<int>((prompt.size() - 1) % 4 + 1)).back();
    std::vector<double> ms;
    for (int i = 0; i < generate; ++i) {
      out.push_back(next);
      const auto ts = std::chrono::steady_clock::now();
      m.step(&next, 1);
      ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
      m.accept(1);
      next = m.picks(0, 1)[0];
    }
    std::printf("\ngenerate: %d greedy tokens after a %zu-token prompt, host %.2f ms per step (median)\n", generate, prompt.size(), median(ms));
    if (have_tok) std::printf("--- prompt ---\n%s\n--- continuation ---\n%s\n--- end ---\n", tok.decode(prompt, false).c_str(), tok.decode(out, false).c_str());
    // the timed steps at position `at`: fill with corpus tokens in 4-row steps
    std::vector<int32_t> corpus;
    for (const auto &s : sets)
      for (const auto &c : s) corpus.insert(corpus.end(), c.begin(), c.end());
    if (corpus.empty()) corpus = prompt;
    size_t cur = 0;
    const auto tf = std::chrono::steady_clock::now();
    while (m.position() + 4 <= at) {
      int32_t t4[4];
      for (auto &t : t4) t = corpus[cur++ % corpus.size()];
      m.step(t4, 4);
      m.accept(4);
    }
    while (m.position() < at) {
      int32_t t = corpus[cur++ % corpus.size()];
      m.step(&t, 1);
      m.accept(1);
    }
    std::printf("\ngenerate: history filled to position %lld in %.1f s\n", static_cast<long long>(m.position()),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tf).count());
    for (const int R : {1, 4}) {
      std::vector<double> span, host;
      q35::StepProfile last;
      m.set_profile(true);
      for (int i = 0; i < generate; ++i) {
        int32_t t4[4];
        for (auto &t : t4) t = corpus[cur++ % corpus.size()];
        const auto ts = std::chrono::steady_clock::now();
        m.step(t4, R);
        host.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
        span.push_back(m.profile().span / 1e3);
        if (i == generate / 2) last = m.profile();
        m.accept(1);  // the verify shape: keep one row, the position moves by one
      }
      m.set_profile(false);
      std::printf("  %d row(s) at position %lld: GPU span median %.2f ms, host %.2f ms per step (%d steps, profiled = eager)\n", R,
                  static_cast<long long>(m.position()), median(span), median(host), generate);
      print_profile(last);
    }
    check(m.wire_errors() == 0, "generate: no mailbox timeout");
  }
  // ----: decode step A/B (--step-ab "name[:key=v..].."): every slot prefilled with corpus tokens to --at, then per
  // shape (--step-shapes SxR) and round (--step-reps) each case runs --generate steps (accept 1 row per segment), host wall
  // (--step-reset 1: the shape's slots are prefilled to --at again before every shape, so shapes compare at one position)
  // median; cases alternate per round. Keys: b16 (the bf16 wire), w4i (the W4 GEMMs' k split target). Then card 0 and card 1 profiles of the first case per
  // shape. ----
  if (!step_ab.empty()) {
    struct Case {
      std::string name;
      q35::StepOptions o;
    };
    std::vector<Case> cases;
    std::stringstream cs(step_ab);
    std::string item;
    while (std::getline(cs, item, ';')) {
      if (item.empty()) continue;
      Case c{item, q35::StepOptions{}};
      const size_t colon = item.find(':');
      c.name = item.substr(0, colon);
      if (colon != std::string::npos && !parse_step_opts(item.substr(colon + 1), c.o)) return 2;
      cases.push_back(c);
    }
    std::vector<int32_t> corpus;
    for (const auto &s : sets)
      for (const auto &c : s) corpus.insert(corpus.end(), c.begin(), c.end());
    if (corpus.empty()) corpus = {760, 3365, 314, 279};
    std::vector<int32_t> fill;
    while (static_cast<int64_t>(fill.size()) < at) fill.insert(fill.end(), corpus.begin(), corpus.end());
    fill.resize(static_cast<size_t>(at));
    const auto tf = std::chrono::steady_clock::now();
    // A fresh served prompt: the trained-QSA index starts at token 0 (Model::note_prompt, as the batch engine does), so
    // past kTrainedQsaShort the steps take the trained sparse walk that serving takes, not the dense fallback.
    auto fill_slot = [&](int sl) {
      m.use_slot(sl);
      m.reset();
      m.note_prompt(sl, static_cast<int64_t>(fill.size()), 0);
      m.prefill(fill.data(), static_cast<int>(fill.size()));
    };
    for (int sl = 0; sl < m.slots(); ++sl) fill_slot(sl);
    m.use_slot(0);
    std::printf("\nstep A/B: %d slot(s) prefilled to position %lld in %.1f s (native xenithium, %s)\n", m.slots(), static_cast<long long>(at),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tf).count(), m.cards() == 1 ? "one B70" : "two B70s");
    const int N = generate > 0 ? generate : 32;
    size_t cur = 0;
    bool run_tree = false;  // The shape's segments are trees (shape_tree)
    int tree_par[xenithium::kernels::kStepRows32] = {-1};
    for (int r = 1; r < xenithium::kernels::kStepRows32; ++r) tree_par[r] = r <= 6 ? (r % 2 ? r - 1 : r - 2) : r - 1;
    long sparse_steps = 0, all_steps = 0;  // verify steps on the trained sparse walk (route check)
    auto run = [&](int S, int R, std::vector<double> *wall, std::vector<double> *sub, int count) {
      std::vector<int32_t> tk(static_cast<size_t>(S * R));
      std::vector<q35::Model::SeqRows> segs;
      for (int i = 0; i < S; ++i) segs.push_back(q35::Model::SeqRows{i, nullptr, R, run_tree ? tree_par : nullptr});
      for (int i = 0; i < count; ++i) {
        for (auto &t : tk) t = corpus[cur++ % corpus.size()];
        for (int j = 0; j < S; ++j) segs[static_cast<size_t>(j)].tokens = tk.data() + j * R;
        const auto ts = std::chrono::steady_clock::now();
        m.verify_batch(segs);
        sparse_steps += m.step_trained_sparse() ? 1 : 0, all_steps += 1;
        if (wall) wall->push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
        if (sub) sub->push_back(m.submit_ms());
        m.accept_batch(std::vector<int>(static_cast<size_t>(S), 1));
      }
    };
    for (size_t si = 0; si < shapes.size(); ++si) {
      const auto [S, R] = shapes[si];
      run_tree = shape_tree[si] != 0;
      if (g_step_reset && si > 0) {  // every shape from --at: a shape's walls do not carry the positions of the shapes before it
        for (int sl = 0; sl < S; ++sl) fill_slot(sl);
        m.use_slot(0);
      }
      std::vector<std::vector<double>> wall(cases.size()), sub(cases.size());
      sparse_steps = all_steps = 0;
      const int64_t p_start = m.position(0);
      for (int rep = 0; rep < std::max(1, step_reps); ++rep)
        for (size_t ci = 0; ci < cases.size(); ++ci) {
          m.set_step_options(cases[ci].o);
          run(S, R, &wall[ci], &sub[ci], N);
        }
      std::printf("  (slot 0 position %lld -> %lld)\n", static_cast<long long>(p_start), static_cast<long long>(m.position(0)));
      m.set_step_options(base_step);
      std::printf("step A/B %dx%d%s (%d rows) near position %lld, %d rounds x %d steps:\n", S, R, run_tree ? "t" : "", S * R, static_cast<long long>(m.position(0)),
                  std::max(1, step_reps), N);
      if (m.trained_qsa())
        std::printf("  trained-QSA route: %ld of %ld steps sparse%s\n", sparse_steps, all_steps,
                    sparse_steps < all_steps && at > q35::Model::trained_qsa_short() ? " (WARNING: dense steps past the short threshold)" : "");
      std::printf("  | case | wall ms per step | submit ms |\n  | --- | --- | --- |\n");
      for (size_t ci = 0; ci < cases.size(); ++ci) std::printf("  | %s | %.3f | %.3f |\n", cases[ci].name.c_str(), median(wall[ci]), median(sub[ci]));
      for (int card = 0; card < C; ++card) {
        m.set_step_options(cases[0].o);
        q35::ModelOptions po = m.options();
        (void)po;
        m.set_profile(true);
        m.set_profile_card(card);
        std::vector<double> span;
        q35::StepProfile last;
        for (int i = 0; i < N; ++i) {
          run(S, R, nullptr, nullptr, 1);
          span.push_back(m.profile().span / 1e3);
          if (i == N / 2) last = m.profile();
        }
        m.set_profile(false);
        m.set_step_options(base_step);
        std::printf("  profile card %d (%s, eager, profiled): GPU span median %.2f ms\n", card, cases[0].name.c_str(), median(span));
        print_profile(last);
      }
      m.set_profile_card(profile_card);
    }
    check(m.wire_errors() == 0, "step A/B: no mailbox timeout");
  }
  // ---- speculative greedy decoding with the MTP drafter: --spec N tokens after every chunk of the first token
  // file as a prompt (prefilled), MTP-depth. Gate: the emitted tokens == plain greedy 1-row decoding after the same
  // prefill, token for token (4 rows == 1 row bit for bit, rejected rows leave no state). Reports the accepted length
  // per verify step, the step and draft times and tokens per second against the plain run. ----
  if (g_dflash_check > 0) {  // the DFlash drafter vs its host reference
    if (sets.empty() || sets[0].empty()) throw std::runtime_error("model35 --dflash-check: needs a --ppl token file (its first chunk is the prompt)");
    dflash_check(m, model, sets[0][0], g_dflash_check, g_dflash_bytes, check);
    check(m.wire_errors() == 0, "dflash-check: no mailbox timeout");
  }
  if (spec > 0) {
    if (!m.has_drafter()) throw std::runtime_error("model35: no drafter");
    if (m.dflash())
      std::printf("\nspec: DFlash 2 drafter %s, %s (matrices %s, context ring %d tokens, block %d, draft vocabulary %d%s), depth %d\n",
                  m.dflash()->dir().c_str(), per_card([&](int c) { return static_cast<double>(m.dflash()->card_bytes(c)); }).c_str(),
                  g_dflash_bytes == 2 ? "bf16" : g_dflash_bytes == 4 ? "int4 (RTN; fc, k|v int8)" : "int8 per row", m.dflash()->ring_tokens(), m.dflash()->config().block, m.dflash()->draft_vocab(),
                  m.dflash()->draft_vocab() ? "" : " = the target's lm_head", depth);
    else
      std::printf("\nspec: MTP drafter %s (matrices %s, KV %s, draft vocabulary %d%s), depth %d\n",
                  per_card([&](int c) { return static_cast<double>(m.drafter()->card_bytes(c)); }).c_str(), mtp_dense_bytes == 2 ? "bf16" : "int8 per row", mtp_kv_bytes == 1 ? "int8" : "bf16",
                  m.drafter()->draft_vocab(), m.drafter()->draft_vocab() ? "" : " = the target's lm_head", depth);
    std::vector<std::vector<int32_t>> prompts;
    if (!sets.empty())
      for (const auto &c : sets[0]) prompts.push_back(c);
    if (prompts.empty()) prompts.push_back(have_tok ? tok.encode("The history of the printing press begins") : std::vector<int32_t>{760, 3365, 314, 279});
    // --mtp-dump FILE: teacher-forced drafter outputs over the first prompt S (n tokens) for a host reference.
    // A: every position by 1-row target steps, each followed by a depth-1 draft (position p: target hidden h_p, token
    // S[p+1]); B: prefill S[0, m) then the same from m - 1 (the drafter KV of the prompt from the prefill path).
    // Layout: int32 n, H, m; int32 S[n]; A: u16 h[n-1][H], u16 out[n-1][H], int32 tok[n-1]; B: u16 out[n-m][H], int32 tok[n-m].
    if (!mtp_dump.empty() && m.drafter()) {
      const auto &S = prompts[0];
      const int n = static_cast<int>(S.size()), H = m.config().hidden, mp = n / 2 / 4 * 4;
      std::vector<uint16_t> ha, oa, ob;
      std::vector<int32_t> ta, tb;
      m.reset();
      for (int p = 0; p + 1 < n; ++p) {
        m.step(&S[static_cast<size_t>(p)], 1);
        m.accept(1);
        const auto h = m.final_hidden(0, 1);
        ha.insert(ha.end(), h.begin(), h.end());
        ta.push_back(m.draft(&S[static_cast<size_t>(p + 1)], 1, 1, 1)[0]);
        const auto o = m.drafter()->read_hidden(0, 1);
        oa.insert(oa.end(), o.begin(), o.end());
      }
      m.reset();
      m.prefill(S.data(), mp, {}, S[static_cast<size_t>(mp)]);
      tb.push_back(m.draft_after_prefill(S[static_cast<size_t>(mp)], 1)[0]);
      auto o = m.drafter()->read_hidden(0, 1);
      ob.insert(ob.end(), o.begin(), o.end());
      for (int p = mp; p + 1 < n; ++p) {
        m.step(&S[static_cast<size_t>(p)], 1);
        m.accept(1);
        tb.push_back(m.draft(&S[static_cast<size_t>(p + 1)], 1, 1, 1)[0]);
        o = m.drafter()->read_hidden(0, 1);
        ob.insert(ob.end(), o.begin(), o.end());
      }
      FILE *f = std::fopen(mtp_dump.c_str(), "wb");
      if (!f) throw std::runtime_error("model35: cannot write " + mtp_dump);
      const int32_t hdr[3] = {n, H, mp};
      std::fwrite(hdr, 4, 3, f);
      std::fwrite(S.data(), 4, S.size(), f);
      std::fwrite(ha.data(), 2, ha.size(), f);
      std::fwrite(oa.data(), 2, oa.size(), f);
      std::fwrite(ta.data(), 4, ta.size(), f);
      std::fwrite(ob.data(), 2, ob.size(), f);
      std::fwrite(tb.data(), 4, tb.size(), f);
      std::fclose(f);
      int same = 0;
      for (int p = mp - 1; p + 1 < n; ++p) same += ta[static_cast<size_t>(p)] == tb[static_cast<size_t>(p - mp + 1)];
      std::printf("mtp-dump: %d positions -> %s; drafts after a prefill of %d == drafts by steps at %d of %d positions\n", n - 1, mtp_dump.c_str(), mp, same,
                  n - mp);
    }
    if (m.dflash()) m.dflash()->clear_stages();
    FILE *conf_f = g_draft_conf ? std::fopen(g_draft_conf_path.c_str(), "w") : nullptr;
    int64_t steps_all = 0, toks_all = 0;
    double plain_ms_all = 0, spec_ms_all = 0, verify_ms_all = 0, draft_ms_all = 0;
    std::vector<int> hist(static_cast<size_t>(std::max(depth, m.seg_rows_max() - 1)) + 2, 0);
    int64_t wide_steps_all = 0;  // 16-row verifies (--dflash-wide)
    double wide_ms_all = 0;
    bool same_all = true;
    int64_t bound_pages = 0, bound_errors = 0;  // --attn-select: the page-bounds audit
    for (size_t pi = 0; pi < prompts.size(); ++pi) {
      const auto &prompt = prompts[pi];
      // plain: prefill, then 1-row greedy steps
      m.reset();
      m.prefill(prompt.data(), static_cast<int>(prompt.size()));
      const int lr = static_cast<int>((prompt.size() - 1) % 4);
      const int32_t first = m.picks(0, lr + 1)[static_cast<size_t>(lr)];
      std::vector<int32_t> ref{first};
      auto t0 = std::chrono::steady_clock::now();
      while (static_cast<int>(ref.size()) < spec) {
        m.step(&ref.back(), 1);
        m.accept(1);
        ref.push_back(m.picks(0, 1)[0]);
      }
      const double plain_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      // speculative: prefill (drafter KV of the prompt), drafts after it, then verify steps of 1 + depth rows
      m.reset();
      m.prefill(prompt.data(), static_cast<int>(prompt.size()));
      const int32_t first2 = m.picks(0, lr + 1)[static_cast<size_t>(lr)];
      std::vector<int32_t> out{first2};
      t0 = std::chrono::steady_clock::now();
      std::vector<int32_t> drafts = m.draft_after_prefill(first2, depth);
      // (audit C2): --draft-conf FILE: per verify step "n d c_0.. c_{d-1}" (accepted rows, drafts, each draft's
      // softmax probability among its block row's top-16 logits), for the dynamic-depth bound
      std::vector<float> conf;
      auto read_conf = [&] {
        conf.clear();
        if (!g_draft_conf || !m.dflash() || drafts.empty()) return;
        const auto keys = m.dflash()->read_keys(static_cast<int>(drafts.size()) + 1);
        for (size_t j = 0; j < drafts.size(); ++j) {
          const uint64_t *kr = keys.data() + (j + 1) * k::kDfTopK;
          double mx = -1e30, se = 0, sel = -1e30;
          for (int b = 0; b < k::kDfTopK; ++b) mx = std::max(mx, static_cast<double>(k::df_key_value(kr[b])));
          for (int b = 0; b < k::kDfTopK; ++b) {
            const double v = k::df_key_value(kr[b]);
            se += std::exp(v - mx);
            int64_t t = k::df_key_index(kr[b]);
            const int sk = m.dflash()->draft_vocab() ? m.dflash()->draft_vocab() - 128 : 0;
            if (sk > 0 && t >= sk) t = t - sk + m.dflash()->draft_vocab_s0();
            if (t == drafts[j]) sel = v;
          }
          conf.push_back(sel > -1e29 ? static_cast<float>(std::exp(sel - mx) / se) : 0.f);
        }
      };
      read_conf();
      double vms = 0, dms = m.draft_ms();
      int64_t steps = 0;
      while (static_cast<int>(out.size()) < spec) {
        std::vector<int32_t> rows_t{out.back()};
        rows_t.insert(rows_t.end(), drafts.begin(), drafts.end());
        const int R = static_cast<int>(rows_t.size());
        // The drafts are a tree (DFlashOptions:tree): verify it, accept the longest matching path
        const auto tpar = m.draft_parents(m.slot());  // empty: a chain (chain-shaped trees too)
        const bool tree = R > 1 && tpar.size() >= static_cast<size_t>(R);
        const auto tv = std::chrono::steady_clock::now();
        if (tree) m.step_tree(rows_t.data(), tpar.data(), R);
        else m.step(rows_t.data(), R);
        auto pk = m.picks(0, R);
        vms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv).count();
        int n = 1;
        if (tree) {
          const auto &par = tpar;
          int path[k::kTreeRows] = {0};
          for (int at = 0;;) {  // the child of the last accepted row whose token is that row's pick
            int nx = -1;
            for (int r = at + 1; r < R && nx < 0; ++r)
              if (par[static_cast<size_t>(r)] == at && rows_t[static_cast<size_t>(r)] == pk[static_cast<size_t>(at)]) nx = r;
            if (nx < 0) break;
            path[n++] = at = nx;
          }
          m.accept_tree(path, n);
          std::vector<int32_t> pp(pk.size());
          for (int r = 0; r < R; ++r) pp[static_cast<size_t>(r)] = pk[static_cast<size_t>(path[std::min(r, n - 1)])];
          pk = pp;
        } else {
          while (n < R && rows_t[static_cast<size_t>(n)] == pk[static_cast<size_t>(n - 1)]) ++n;
          m.accept(n);
        }
        for (int r = 0; r < n; ++r) out.push_back(pk[static_cast<size_t>(r)]);
        ++steps;
        ++hist[static_cast<size_t>(n)];
        if (conf_f) {
          std::fprintf(conf_f, "%d %zu", n, conf.size());
          for (float c : conf) std::fprintf(conf_f, " %.5f", c);
          std::fprintf(conf_f, "\n");
        }
        if (R > k::kStepRows8) ++wide_steps_all, wide_ms_all += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv).count();
        // --dflash-wide N: the next pass drafts DSpine's whole block (16 rows) when this verify
        // accepted N or more drafts, else the usual depth (the engine's rule, StepModelAdapter:wide_from)
        const int wdepth = g_dflash_wide > 0 && m.seg_rows_max() > k::kStepRows8 && n - 1 >= g_dflash_wide ? m.seg_rows_max() - 1 : depth;
        drafts = m.draft(pk.data(), R, n, wdepth);
        dms += m.draft_ms();
        read_conf();
      }
      const double spec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      out.resize(static_cast<size_t>(spec));
      const bool same = out == ref && first2 == first;
      same_all &= same;
      if (g_attn_select > 0) {  // every whole page the spec run's steps (chains and trees) wrote and left
        std::vector<int> pages;
        for (int p = static_cast<int>(prompt.size() / 64); p < static_cast<int>(m.position() / 64); ++p) pages.push_back(p);
        bound_pages += static_cast<int64_t>(pages.size());
        if (!pages.empty()) bound_errors += m.page_bound_errors(0, pages);
      }
      const int64_t gen = spec - 1;  // tokens decoded after the prefill's pick
      std::printf("  prompt %zu (%zu tokens): %lld verify steps for %lld tokens = %.2f per step; plain %.2f ms per token, spec %.2f (verify %.2f + "
                  "drafts %.2f ms per step) -> %.2fx; %s\n",
                  pi, prompt.size(), static_cast<long long>(steps), static_cast<long long>(gen), static_cast<double>(gen) / std::max<int64_t>(steps, 1),
                  plain_ms / gen, spec_ms / gen, vms / std::max<int64_t>(steps, 1), dms / std::max<int64_t>(steps, 1), plain_ms / spec_ms,
                  same ? "== plain greedy" : "DIFFERS from plain greedy");
      if (have_tok && pi == 0) std::printf("    --- continuation ---\n%s\n    --- end ---\n", tok.decode(out, false).c_str());
      steps_all += steps;
      toks_all += gen;
      plain_ms_all += plain_ms;
      spec_ms_all += spec_ms;
      verify_ms_all += vms;
      draft_ms_all += dms;
    }
    std::printf("spec: %zu prompts, %lld tokens in %lld verify steps = %.3f tokens per step (accepted rows per step:", prompts.size(),
                static_cast<long long>(toks_all), static_cast<long long>(steps_all), static_cast<double>(toks_all) / std::max<int64_t>(steps_all, 1));
    for (size_t n = 1; n < hist.size(); ++n) std::printf(" %zu:%d", n, hist[n]);
    std::printf("); plain %.2f ms per token (%.1f tok/s), spec %.2f (%.1f tok/s, verify %.2f + drafts %.2f ms per step) = %.2fx\n", plain_ms_all / toks_all,
                1e3 * toks_all / plain_ms_all, spec_ms_all / toks_all, 1e3 * toks_all / spec_ms_all, verify_ms_all / std::max<int64_t>(steps_all, 1),
                draft_ms_all / std::max<int64_t>(steps_all, 1), plain_ms_all / spec_ms_all);
    if (wide_steps_all > 0)
      std::printf("spec: %lld 16-row verifies (%.1f %% of steps), %.2f ms each with the accept (--dflash-wide %d)\n", static_cast<long long>(wide_steps_all),
                  100.0 * static_cast<double>(wide_steps_all) / std::max<int64_t>(steps_all, 1), wide_ms_all / static_cast<double>(wide_steps_all), g_dflash_wide);
    check(same_all, "spec: speculative greedy == plain greedy, token for token, every prompt");
    if (g_attn_select > 0)
      check(bound_errors == 0, "spec: key bounds == the min / max of the stored keys on the " + std::to_string(bound_pages) + " pages the spec runs wrote and left (" +
                                   std::to_string(bound_errors) + " page x layer x kv head units differ)");
    if (conf_f) std::fclose(conf_f);
    if (m.dflash()) std::printf("spec: DFlash pass graphs %d recorded in %.1f ms\n", m.dflash()->graphs(), m.dflash()->graphs_ms());
    if (m.dflash() && g_dflash_prof && m.dflash()->stage_passes() > 0) {  // where a drafter pass goes
      const auto *df = m.dflash();
      double all = 0;
      for (int i = 0; i < q35::DFlash::kStages; ++i) all += df->stage_ms()[i];
      std::printf("dflash-prof: %d passes, %.3f ms per pass (every stage waited)\n", df->stage_passes(), all / df->stage_passes());
      for (int i = 0; i < q35::DFlash::kStages; ++i)
        std::printf("  %-28s %8.3f ms per pass (%4.1f %%)\n", q35::DFlash::stage_names()[i], df->stage_ms()[i] / df->stage_passes(), 100 * df->stage_ms()[i] / std::max(all, 1e-9));
    }
    check(m.wire_errors() == 0, "spec: no mailbox timeout");
  }
  // ---- several sequences in one step: --seqs N prompts (chunk i of the first token file, its last 3 i
  // tokens dropped so the positions differ) on slots 0..N-1, T tokens each.
  // alone: each prompt on slot 0 by itself (prefill, then 1-row greedy steps): its tokens and every row's logits bits;
  // batch: the N slots prefilled one after another, then steps of every live sequence (1 row each); sequence i sits
  // out the steps where (t + i) % 3 == 2 (a slot left out keeps its state): tokens and logits bits == alone;
  // spec (--seq-depths, with the drafter): verify segments of 1 + d rows per sequence, accept, draft_batch; the
  // emitted tokens == alone, token for token.
  // Reports ms per step and tokens per second for each. ----
  if (seqs > 0) {
    if (sets.empty() || sets[0].empty()) throw std::runtime_error("model35 --seqs: needs a token file");
    const int N = seqs, T = seq_tokens;
    if (N > m.slots() || (N > 1 && m.slots() != N)) throw std::runtime_error("model35 --seqs: slots");
    std::vector<std::vector<int32_t>> pr;
    for (int i = 0; i < N; ++i) {
      auto p = sets[0][static_cast<size_t>(i) % sets[0].size()];
      p.resize(p.size() - std::min(p.size() - 4, static_cast<size_t>(3 * i)));
      pr.push_back(p);
    }
    auto bits = [](const std::vector<float> &v, size_t r, size_t V) {
      uint64_t h = 1469598103934665603ull;
      for (size_t i = r * V; i < (r + 1) * V; ++i) {
        uint32_t b;
        std::memcpy(&b, &v[i], 4);
        h = (h ^ b) * 1099511628211ull;
      }
      return h;
    };
    const size_t V = static_cast<size_t>(m.config().vocab);
    std::vector<float> lg;
    std::vector<std::vector<int32_t>> ref(static_cast<size_t>(N));
    std::vector<std::vector<uint64_t>> refh(static_cast<size_t>(N));
    double alone_ms = 0;
    int64_t alone_steps = 0;
    for (int i = 0; i < N; ++i) {
      m.use_slot(0);
      m.reset();
      const auto &p = pr[static_cast<size_t>(i)];
      m.prefill(p.data(), static_cast<int>(p.size()));
      const int lr = static_cast<int>((p.size() - 1) % 4);
      ref[i].push_back(m.picks(0, lr + 1)[static_cast<size_t>(lr)]);
      const auto t0 = std::chrono::steady_clock::now();
      while (static_cast<int>(ref[i].size()) < T) {
        m.step(&ref[i].back(), 1);
        m.accept(1);
        m.logits(1, lg);
        refh[i].push_back(bits(lg, 0, V));
        ref[i].push_back(m.picks(0, 1)[0]);
        ++alone_steps;
      }
      alone_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    std::printf("\nseqs: %d sequences x %d tokens; alone: %.2f ms per 1-row step (%.1f tok/s)\n", N, T, alone_ms / alone_steps, 1e3 * alone_steps / alone_ms);
    auto prefill_all = [&](int depth, std::vector<std::vector<int32_t>> &out, std::vector<std::vector<int32_t>> *drafts) {
      for (int i = 0; i < N; ++i) {
        m.use_slot(i);
        m.reset();
        const auto &p = pr[static_cast<size_t>(i)];
        m.prefill(p.data(), static_cast<int>(p.size()));
        const int lr = static_cast<int>((p.size() - 1) % 4);
        out[i] = {m.picks(0, lr + 1)[static_cast<size_t>(lr)]};
        if (m.has_drafter()) {  // the prompt's last drafter row (its KV) even when nothing drafts
          auto d = m.draft_after_prefill(out[i][0], std::max(depth, 1), N);  // the drafter of an N-sequence step
          d.resize(static_cast<size_t>(std::max(depth, 0)));
          if (drafts) (*drafts)[i] = d;
        }
      }
    };
    // plain batch, with sit-outs
    {
      std::vector<std::vector<int32_t>> out(static_cast<size_t>(N));
      std::vector<std::vector<uint64_t>> oh(static_cast<size_t>(N));
      prefill_all(0, out, nullptr);
      double ms = 0;
      int64_t steps = 0, rows_all = 0;
      bool hsame = true;
      for (int t = 0;; ++t) {
        std::vector<q35::Model::SeqRows> segs;
        std::vector<int> who;
        for (int i = 0; i < N; ++i)
          if (static_cast<int>(out[i].size()) < T && !(N > 1 && (t + i) % 3 == 2)) segs.push_back({i, &out[i].back(), 1}), who.push_back(i);
        bool left = false;
        for (int i = 0; i < N; ++i) left |= static_cast<int>(out[i].size()) < T;
        if (!left) break;
        if (segs.empty()) continue;
        const auto t0 = std::chrono::steady_clock::now();
        m.verify_batch(segs);
        const auto pk = m.picks(0, static_cast<int>(segs.size()));
        m.accept_batch(std::vector<int>(segs.size(), 1));
        if (m.has_drafter()) m.draft_batch(pk.data(), 0);  // the drafter KV of every row, as the server's steps
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        m.logits(static_cast<int>(segs.size()), lg);
        for (size_t s = 0; s < segs.size(); ++s) {
          const int i = who[s];
          oh[i].push_back(bits(lg, s, V));
          out[i].push_back(pk[s]);
        }
        ++steps;
        rows_all += static_cast<int64_t>(segs.size());
      }
      bool same = true;
      for (int i = 0; i < N; ++i) same &= out[i] == ref[i], hsame &= oh[i] == refh[i];
      std::printf("seqs: batch (sit-outs): %lld steps of %.2f rows, %.2f ms per step, %.1f tok/s; tokens %s, logits bits %s\n", static_cast<long long>(steps),
                  static_cast<double>(rows_all) / steps, ms / steps, 1e3 * rows_all / ms, same ? "== alone" : "DIFFER", hsame ? "== alone" : "DIFFER");
      check(same && hsame, "seqs: every sequence in batch steps (with sit-outs) == alone, tokens and logits bits");
    }
    // a prefill in parts: slot 1's prompt in parts of 7 layers with a step of slot 0 between the parts, then
    // both decode together: tokens and logits bits == alone
    {
      std::vector<std::vector<int32_t>> out(2);
      std::vector<std::vector<uint64_t>> oh(2);
      m.use_slot(0);
      m.reset();
      m.prefill(pr[0].data(), static_cast<int>(pr[0].size()));
      int lr = static_cast<int>((pr[0].size() - 1) % 4);
      out[0] = {m.picks(0, lr + 1)[static_cast<size_t>(lr)]};
      if (m.has_drafter()) m.draft_after_prefill(out[0][0], 1);
      auto step0 = [&] {
        m.verify_batch({{0, &out[0].back(), 1}});
        const auto pk = m.picks(0, 1);
        m.accept_batch({1});
        if (m.has_drafter()) m.draft_batch(pk.data(), 0);
        m.logits(1, lg);
        oh[0].push_back(bits(lg, 0, V));
        out[0].push_back(pk[0]);
      };
      m.use_slot(1);
      m.reset();
      m.prefill_begin(pr[1].data(), static_cast<int>(pr[1].size()));
      int parts = 1;
      while (!m.prefill_advance(7)) {
        m.use_slot(0);  // as the engine's begin / first_drafts of other slots between the parts (advance switches back)
        step0();
        ++parts;
      }
      lr = static_cast<int>((pr[1].size() - 1) % 4);
      out[1] = {m.picks(0, lr + 1)[static_cast<size_t>(lr)]};
      if (m.has_drafter()) m.draft_after_prefill(out[1][0], 1);
      while (static_cast<int>(out[0].size()) < T || static_cast<int>(out[1].size()) < T) {
        std::vector<q35::Model::SeqRows> segs;
        std::vector<int> who;
        for (int i = 0; i < 2; ++i)
          if (static_cast<int>(out[i].size()) < T) segs.push_back({i, &out[i].back(), 1}), who.push_back(i);
        m.verify_batch(segs);
        const auto pk = m.picks(0, static_cast<int>(segs.size()));
        m.accept_batch(std::vector<int>(segs.size(), 1));
        if (m.has_drafter()) m.draft_batch(pk.data(), 0);
        m.logits(static_cast<int>(segs.size()), lg);
        for (size_t s = 0; s < segs.size(); ++s) oh[who[s]].push_back(bits(lg, s, V)), out[who[s]].push_back(pk[s]);
      }
      const bool same = out[0] == ref[0] && out[1] == ref[1] && oh[0] == refh[0] && oh[1] == refh[1];
      std::printf("seqs: slot 1 prefilled in %d parts of 7 layers with slot 0 decoding between them: tokens and logits bits %s\n", parts,
                  same ? "== alone" : "DIFFER");
      check(same, "seqs: a prefill in parts with another slot's steps between == alone, bit for bit");
    }
    // full plain batches (no sit-outs): the decode rate at N sequences
    {
      std::vector<std::vector<int32_t>> out(static_cast<size_t>(N));
      prefill_all(0, out, nullptr);
      double ms = 0;
      int64_t steps = 0;
      while (static_cast<int>(out[0].size()) < T) {
        std::vector<q35::Model::SeqRows> segs;
        for (int i = 0; i < N; ++i) segs.push_back({i, &out[i].back(), 1});
        const auto t0 = std::chrono::steady_clock::now();
        m.verify_batch(segs);
        const auto pk = m.picks(0, N);
        m.accept_batch(std::vector<int>(static_cast<size_t>(N), 1));
        if (m.has_drafter()) m.draft_batch(pk.data(), 0);
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        for (int i = 0; i < N; ++i) out[i].push_back(pk[static_cast<size_t>(i)]);
        ++steps;
      }
      bool same = true;
      for (int i = 0; i < N; ++i) same &= out[i] == ref[i];
      std::printf("seqs: plain batch of %d: %.2f ms per step (%s), %.1f tok/s in all; tokens %s\n", N, ms / steps, m.has_drafter() ? "+ drafter KV pass" : "no drafter",
                  1e3 * N * steps / ms, same ? "== alone" : "DIFFER");
      check(same, "seqs: plain batches == alone, token for token");
    }
    // speculative batches
    std::vector<int> depths;
    {
      std::string t;
      for (char ch : seq_depths + ",")
        if (ch == ',') {
          // cD = cycle: 8 steps without drafts (depth 0), 8 at depth D... (the drafter's depth changes)
          if (!t.empty()) depths.push_back(t[0] == 'c' ? -std::stoi(t.substr(1)) : std::stoi(t));
          t.clear();
        } else {
          t += ch;
        }
    }
    for (int d : depths) {
      if (!m.has_drafter()) throw std::runtime_error("model35 --seq-depths: needs the drafter (--spec)");
      const bool cyc = d < 0;
      if (cyc) d = -d;
      const int sr = 1 + d;
      if (sr > m.seg_rows_max() || sr * N > m.step_rows()) {  // 16 rows (qwen35)
        std::printf("seqs: depth %d skipped (%d x %d rows > %d)\n", d, N, sr, m.step_rows());
        continue;
      }
      std::vector<std::vector<int32_t>> out(static_cast<size_t>(N)), dr(static_cast<size_t>(N));
      prefill_all(cyc ? 0 : d, out, &dr);
      if (m.dflash()) m.dflash()->clear_stages();  // --dflash-prof: the stepped passes only
      double ms = 0, vms = 0;
      int dprev = cyc ? 0 : d;  // the depth the drafts in dr were made at
      int64_t steps = 0, emitted = 0;
      std::vector<int> hist(static_cast<size_t>(sr) + 1, 0);
      const auto t0 = std::chrono::steady_clock::now();
      for (;;) {
        std::vector<int> who;
        for (int i = 0; i < N; ++i)
          if (static_cast<int>(out[i].size()) < T) who.push_back(i);
        if (who.empty()) break;
        const int ns = static_cast<int>(who.size());
        const int ssr = 1 + dprev;  // this step's rows per segment
        std::vector<std::vector<int32_t>> rt(static_cast<size_t>(ns));
        std::vector<q35::Model::SeqRows> segs;
        for (int s = 0; s < ns; ++s) {
          const int i = who[s];
          rt[s] = {out[i].back()};
          rt[s].insert(rt[s].end(), dr[i].begin(), dr[i].end());
          rt[s].resize(static_cast<size_t>(ssr), 0);
          segs.push_back({i, rt[s].data(), ssr});
        }
        const auto tv = std::chrono::steady_clock::now();
        m.verify_batch(segs);
        const auto pk = m.picks(0, ns * ssr);
        vms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv).count();
        std::vector<int> keep;
        std::vector<int32_t> nx(static_cast<size_t>(ns * ssr));
        for (int s = 0; s < ns; ++s) {
          int n = 1;
          while (n < ssr && static_cast<int>(dr[who[s]].size()) >= n && rt[s][static_cast<size_t>(n)] == pk[static_cast<size_t>(s * ssr + n - 1)]) ++n;
          keep.push_back(n);
          if (ssr > 1) ++hist[static_cast<size_t>(n)];  // cycles: the drafted steps only
          for (int r = 0; r < n; ++r) out[who[s]].push_back(pk[static_cast<size_t>(s * ssr + r)]);
          emitted += n;
          for (int r = 0; r < ssr; ++r) nx[static_cast<size_t>(s * ssr + r)] = pk[static_cast<size_t>(s * ssr + std::min(r, n - 1))];
        }
        m.accept_batch(keep);
        const int dd = cyc ? (((steps / 8) & 1) ? d : 0) : d;
        auto d2 = m.draft_batch(nx.data(), dd);
        for (int s = 0; s < ns; ++s) dr[who[s]] = dd > 0 ? d2[static_cast<size_t>(s)] : std::vector<int32_t>{};
        dprev = dd;
        ++steps;
      }
      ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      bool same = true;
      for (int i = 0; i < N; ++i) {
        out[i].resize(static_cast<size_t>(T));
        same &= out[i] == ref[i];
      }
      std::printf("seqs: spec depth %s%d, %d x %d rows: %lld steps, %.3f tokens per segment (accepted rows:", cyc ? "cycle 0/" : "", d, N, sr, static_cast<long long>(steps),
                  [&] {
                    if (!cyc) return static_cast<double>(emitted) / std::max(1, std::accumulate(hist.begin(), hist.end(), 0));
                    double acc = 0;  // cycles: tokens per drafted step
                    for (int n = 1; n <= sr; ++n) acc += static_cast<double>(n) * hist[static_cast<size_t>(n)];
                    return acc / std::max(1, std::accumulate(hist.begin(), hist.end(), 0));
                  }());
      if (m.dflash()) std::printf(" [deferred-context passes %d]", m.dflash()->flushes());
      if (m.drafter() && m.drafter()->defers()) std::printf(" [deferred MTP KV passes %d]", m.drafter()->flushes());
      for (int n = 1; n <= sr; ++n) std::printf(" %d:%d", n, hist[static_cast<size_t>(n)]);
      std::printf("); verify %.2f ms + drafts %.2f ms per step; %.1f tok/s in all; tokens %s\n", vms / steps, (ms - vms) / steps, 1e3 * emitted / ms,
                  same ? "== alone" : "DIFFER");
      check(same, "seqs: speculative batches (depth " + std::to_string(d) + ") == plain greedy alone, token for token");
      if (m.dflash() && g_dflash_prof && m.dflash()->stage_passes() > 0) {  // --dflash-prof 1: the pass's stages at this shape
        const auto *df = m.dflash();
        double all = 0;
        for (int i = 0; i < q35::DFlash::kStages; ++i) all += df->stage_ms()[i];
        std::printf("dflash-prof: %d x %d rows, %d passes, %.3f ms per pass (every stage waited)\n", N, sr, df->stage_passes(), all / df->stage_passes());
        for (int i = 0; i < q35::DFlash::kStages; ++i)
          std::printf("  %-28s %8.3f ms per pass (%4.1f %%)\n", q35::DFlash::stage_names()[i], df->stage_ms()[i] / df->stage_passes(), 100 * df->stage_ms()[i] / std::max(all, 1e-9));
      }
    }
    // Unequal segments. Each step the sequences holding drafts run 1 + d rows
    // (first), the rest one row (kernels:seg_het); after the accept a rotating set drafts d (draft_batch with a depth per
    // segment), as many as fit the step's rows. Tokens == alone, and every accepted row's logits bits == alone's.
    // Two classes (kernels:seg_het2): the sequences holding dhi drafts run 1 + dhi rows (first), the
    // rest 1 + dlo rows with their dlo drafts; (0, d) is 's phase, (d, d + 1) follows it per depth where the
    // drafter drafts d + 1 at this count and the rows fit.
    auto het_phase = [&](int dlo, int dhi) {
      const int B = 1 + dhi, B2 = 1 + dlo, fit = std::min(N - 1, (m.step_rows() - N * B2) / (dhi - dlo));
      if (fit < 1) return;
      std::vector<std::vector<int32_t>> out(static_cast<size_t>(N)), dr(static_cast<size_t>(N));
      std::vector<std::vector<uint64_t>> oh(static_cast<size_t>(N));
      prefill_all(dhi, out, &dr);
      for (int i = fit; i < N; ++i) dr[i].resize(std::min(dr[i].size(), static_cast<size_t>(dlo)));  // the first step: the first `fit` draft deep
      m.set_draft_noise(0, 0x5eedULL, 0, 1.0f);  // slot 0 samples its drafts (coupled noise: draft_gumbel on these layouts)
      int64_t steps = 0, emitted = 0, het = 0, wide = 0;
      double ms = 0;
      for (;;) {
        std::vector<int> who, ord;
        for (int i = 0; i < N; ++i)
          if (static_cast<int>(out[i].size()) < T) who.push_back(i);
        if (who.empty()) break;
        auto deep = [&](int i) { return static_cast<int>(dr[i].size()) >= dhi; };
        for (int i : who)
          if (deep(i)) ord.push_back(i);
        const int kw = static_cast<int>(ord.size());
        for (int i : who)
          if (!deep(i)) ord.push_back(i);
        const int ns = static_cast<int>(ord.size());
        std::vector<std::vector<int32_t>> rt(static_cast<size_t>(ns));
        std::vector<q35::Model::SeqRows> segs;
        std::vector<int> off;
        int rows = 0;
        for (int s = 0; s < ns; ++s) {
          const int i = ord[s], rs = s < kw ? B : B2;
          rt[s] = {out[i].back()};
          if (rs > 1) rt[s].insert(rt[s].end(), dr[i].begin(), dr[i].end());
          rt[s].resize(static_cast<size_t>(rs), 0);  // (a sequence holding fewer drafts than its class: padded rows, never kept)
          segs.push_back({i, rt[s].data(), rs});
          off.push_back(rows);
          rows += rs;
        }
        het += kw > 0 && kw < ns;
        wide += kw;
        const auto t0 = std::chrono::steady_clock::now();
        m.verify_batch(segs);
        const auto pk = m.picks(0, rows);
        std::vector<int> keep;
        std::vector<int32_t> nx(static_cast<size_t>(rows));
        for (int s = 0; s < ns; ++s) {
          const int rs = segs[static_cast<size_t>(s)].rows, o = off[static_cast<size_t>(s)], held = static_cast<int>(dr[ord[s]].size());
          int n = 1;
          while (n < rs && n <= held && rt[s][static_cast<size_t>(n)] == pk[static_cast<size_t>(o + n - 1)]) ++n;
          keep.push_back(n);
          for (int r = 0; r < rs; ++r) nx[static_cast<size_t>(o + r)] = pk[static_cast<size_t>(o + std::min(r, n - 1))];
        }
        m.logits(rows, lg);
        for (int s = 0; s < ns; ++s) {
          const int i = ord[s], o = off[static_cast<size_t>(s)];
          for (int r = 0; r < keep[static_cast<size_t>(s)]; ++r) oh[i].push_back(bits(lg, static_cast<size_t>(o + r), V)), out[i].push_back(pk[static_cast<size_t>(o + r)]);
          emitted += keep[static_cast<size_t>(s)];
        }
        m.accept_batch(keep);
        // the next deep set: `fit` of the going sequences, rotating with the step
        std::vector<int> dep(static_cast<size_t>(ns), 0);
        const bool all = steps % 3 == 0 && N * B <= m.step_rows();  // every third step all draft deep where they fit
        for (int s = 0; s < ns; ++s) dep[static_cast<size_t>(s)] = all || ((ord[s] + steps) % N) < fit ? dhi : dlo;
        auto d2 = m.draft_batch(nx.data(), dep);
        for (int s = 0; s < ns; ++s) dr[ord[s]] = dep[static_cast<size_t>(s)] > 0 ? d2[static_cast<size_t>(s)] : std::vector<int32_t>{};
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++steps;
      }
      bool same = true, hsame = true;
      for (int i = 0; i < N; ++i) {
        out[i].resize(static_cast<size_t>(T));
        same &= out[i] == ref[i];
        const size_t nh = std::min(oh[i].size(), refh[i].size());
        hsame &= nh > 0 && std::equal(oh[i].begin(), oh[i].begin() + static_cast<std::ptrdiff_t>(nh), refh[i].begin());
      }
      std::printf("seqs: unequal segments, depth %d / %d (%d of %d draft %d, %d x %d + %d x %d rows): %lld steps (%lld unequal, %.2f deep per step), %.2f ms "
                  "per step, %.1f tok/s in all; tokens %s, logits bits %s\n",
                  dhi, dlo, fit, N, dhi, fit, B, N - fit, B2, static_cast<long long>(steps), static_cast<long long>(het), static_cast<double>(wide) / std::max<int64_t>(1, steps),
                  ms / std::max<int64_t>(1, steps), 1e3 * emitted / ms, same ? "== alone" : "DIFFER", hsame ? "== alone" : "DIFFER");
      m.set_draft_noise(0, 0, 0, 0.0f);
      check(same && hsame, "seqs: unequal segments (depth " + std::to_string(dhi) + (dlo > 0 ? " / " + std::to_string(dlo) : std::string()) + ") == alone, tokens and accepted rows' logits bits");
    };
    for (int d : depths) {
      if (d <= 0 || 1 + d > m.seg_rows_max() || N + d > m.step_rows() || N < 2) continue;
      het_phase(0, d);
      if (2 + d <= m.seg_rows_max() && d + 1 <= m.draft_depth_max(N) && N * (1 + d) + 1 <= m.step_rows()) het_phase(d, d + 1);
    }
    check(m.wire_errors() == 0, "seqs: no mailbox timeout");
  }
  std::printf("model35: %s\n", rc ? "FAIL" : "PASS");
  return rc;
}
