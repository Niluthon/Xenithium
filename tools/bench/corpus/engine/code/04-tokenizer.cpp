// Xenithium · BPE tokenizer over tokenizer.json: byte-level (the Qwen family) or
// SentencePiece-style (Metaspace + byte fallback: Llama-family checkpoints such as Nanbeige4.3).
// Golden ids: tests/data/tokenizer_golden.json / tokenizer_golden_sp.json.
#include "xenithium/text/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "xenithium/text/unicode.hpp"
#include "xenithium/util/json.hpp"

namespace xenithium::text {
namespace uni = unicode;

// ---- UTF-8 ------------------------------------------------------------------------------------

Utf8 utf8_at(std::string_view s, size_t i, size_t &len) {
  uint8_t b = static_cast<uint8_t>(s[i]);
  if (b < 0x80) {
    len = 1;
    return Utf8::complete;
  }
  // Well-formed sequences only (Unicode table 3-7): no overlongs, surrogates or > U+10FFFF.
  size_t need;
  uint8_t lo = 0x80, hi = 0xBF;
  if (b >= 0xC2 && b <= 0xDF) {
    need = 1;
  } else if (b >= 0xE0 && b <= 0xEF) {
    need = 2;
    if (b == 0xE0) lo = 0xA0;
    if (b == 0xED) hi = 0x9F;
  } else if (b >= 0xF0 && b <= 0xF4) {
    need = 3;
    if (b == 0xF0) lo = 0x90;
    if (b == 0xF4) hi = 0x8F;
  } else {
    len = 1;
    return Utf8::invalid;
  }
  for (size_t k = 1; k <= need; ++k) {
    if (i + k >= s.size()) {
      len = k;
      return Utf8::incomplete;
    }
    uint8_t t = static_cast<uint8_t>(s[i + k]);
    if (t < (k == 1 ? lo : 0x80) || t > (k == 1 ? hi : 0xBF)) {
      len = 1;
      return Utf8::invalid;
    }
  }
  len = need + 1;
  return Utf8::complete;
}

namespace {

// Bytes of an invalid sequence at s[i] that become one U+FFFD: its maximal well-formed prefix
// (at least 1), as Python's errors="replace" and Rust's from_utf8_lossy both do.
size_t invalid_run(std::string_view s, size_t i) {
  size_t best = 1, len;
  for (size_t j = 2; j <= 3 && i + j <= s.size(); ++j)
    if (utf8_at(s.substr(i, j), 0, len) == Utf8::incomplete) best = j;
  return best;
}

constexpr std::string_view kReplacement = "\xEF\xBF\xBD";

// Appends s[0..n) with invalid UTF-8 replaced; returns the bytes consumed (an incomplete tail
// is left when keep_tail, replaced otherwise).
size_t sanitize(std::string_view s, std::string &out, bool keep_tail) {
  size_t i = 0, len;
  while (i < s.size()) {
    Utf8 u = utf8_at(s, i, len);
    if (u == Utf8::complete) {
      out.append(s.substr(i, len));
      i += len;
    } else if (u == Utf8::incomplete) {
      if (keep_tail) break;
      out.append(kReplacement);
      i = s.size();
    } else {
      out.append(kReplacement);
      i += invalid_run(s, i);
    }
  }
  return i;
}

// ---- GPT-2 byte-level alphabet ------------------------------------------------------------------
// Printable Latin-1 bytes stand for themselves; the other 68 bytes map, in byte order, to
// U+0100 upwards so every byte is a visible character in vocab strings.
struct ByteLevel {
  std::array<char32_t, 256> to_char{};
  std::array<int16_t, 324> to_byte{};  // code point -> byte, -1 when not in the alphabet
  ByteLevel() {
    to_byte.fill(-1);
    char32_t next = 256;
    for (int b = 0; b < 256; ++b) {
      bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
      to_char[b] = printable ? static_cast<char32_t>(b) : next++;
      to_byte[to_char[b]] = static_cast<int16_t>(b);
    }
  }
  // A vocab string -> raw bytes; false when a character is outside the alphabet.
  bool unmap(std::string_view s, std::string &out) const {
    std::u32string cps;
    uni::decode_utf8(s, cps);
    for (char32_t c : cps) {
      if (c >= to_byte.size() || to_byte[c] < 0) return false;
      out.push_back(static_cast<char>(to_byte[c]));
    }
    return true;
  }
};
const ByteLevel &byte_level() {
  static const ByteLevel b;
  return b;
}

// The split patterns implemented (tokenizer.json of the Qwen 3.x family). Which one applies: see
// load (a Qwen2Tokenizer always splits by kSplitPattern; e.g. "नमस्ते" as "नमस" "्त"
// "े", golden case "devanagari-namaste").
// kSplitPattern: letters \p{L}.
// kSplitPatternLM (newer Qwen checkpoints): letter runs [\p{L}\p{M}]+ and \p{M} out of the punctuation class; its
// optional lead still excludes only \p{L} \p{N} (a mark there starts the run either way: same end).
constexpr std::string_view kSplitPattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
constexpr std::string_view kSplitPatternLM =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

bool is_nl(char32_t c) { return c == U'\r' || c == U'\n'; }

// End of the pattern's match at c[i] (alternatives in order, each with the backtracking its
// quantifiers imply). Every code point starts some alternative, so pieces tile the text.
// marks: the kSplitPatternLM variant.
size_t match_at(const char32_t *c, size_t n, size_t i, bool marks) {
  auto word = [&](char32_t ch) { return uni::is_letter(ch) || (marks && uni::is_mark(ch)); };
  // [^\s\p{L}\p{N}] / [^\s\p{L}\p{M}\p{N}]
  auto other = [&](char32_t ch) { return !uni::is_space(ch) && !word(ch) && !uni::is_number(ch); };
  char32_t x = c[i];
  // (?i:'s|'t|'re|'ve|'m|'ll|'d)
  if (x == U'\'' && i + 1 < n) {
    char32_t y = c[i + 1];
    if (uni::folds_to(y, 's') || uni::folds_to(y, 't') || uni::folds_to(y, 'm') || uni::folds_to(y, 'd')) return i + 2;
    if (i + 2 < n) {
      char32_t z = c[i + 2];
      if ((uni::folds_to(y, 'r') || uni::folds_to(y, 'v')) && uni::folds_to(z, 'e')) return i + 3;
      if (uni::folds_to(y, 'l') && uni::folds_to(z, 'l')) return i + 3;
    }
  }
  // [^\r\n\p{L}\p{N}]?\p{L}+ (or [\p{L}\p{M}]+)
  size_t j = i;
  if (word(x)) {
    j = i;
  } else if (!is_nl(x) && !uni::is_number(x) && i + 1 < n && word(c[i + 1])) {
    j = i + 1;
  } else {
    j = n + 1;  // no match
  }
  if (j <= n) {
    while (j < n && word(c[j])) ++j;
    return j;
  }
  // \p{N}
  if (uni::is_number(x)) return i + 1;
  // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
  j = n + 1;
  if (x == U' ' && i + 1 < n && other(c[i + 1]))
    j = i + 1;
  else if (other(x))
    j = i;
  if (j <= n) {
    while (j < n && other(c[j])) ++j;
    while (j < n && is_nl(c[j])) ++j;
    return j;
  }
  // \s*[\r\n]+ | \s+(?!\S) | \s+ (x is whitespace here)
  size_t k = i;
  while (k < n && uni::is_space(c[k])) ++k;
  for (size_t q = k; q > i; --q)
    if (is_nl(c[q - 1])) return q;  // ends after the run's last newline
  if (k == n || k - 1 == i) return k;
  return k - 1;  // leaves one space to lead the next word
}

// ---- the Flint pre-tokenizer: Sequence[Split(kFlintDigits), Split(kFlintCjk), Split(kFlintWords), ByteLevel] --------
// Each Split is Isolated (matches and the gaps between them both become pieces) and runs on every piece of the one before.
constexpr std::string_view kFlintDigits = R"(\p{N}{1,3})";
constexpr std::string_view kFlintCjk = "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+";
constexpr std::string_view kFlintWords =
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

bool is_cjk(char32_t c) { return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF); }

// End of kFlintWords' match at c[i] inside the piece [i, hi) (the regex sees only the piece), or 0 when nothing matches
// there (every alternative consumes at least one code point, so 0 is never a real end). Alternatives in order, each with
// the backtracking its quantifiers imply (leftmost-first, as a backtracking regex engine matches).
size_t match_flint(const char32_t *c, size_t hi, size_t i) {
  auto A = [&](size_t k) { return k < hi && uni::is_case_lead(c[k]); };
  auto B = [&](size_t k) { return k < hi && uni::is_case_tail(c[k]); };
  // (?i:'s|'t|'re|'ve|'m|'ll|'d)? after a word ending at e
  auto contraction = [&](size_t e) -> size_t {
    if (e + 1 < hi && c[e] == U'\'') {
      const char32_t y = c[e + 1];
      if (uni::folds_to(y, 's') || uni::folds_to(y, 't')) return e + 2;
      if (e + 2 < hi) {
        const char32_t z = c[e + 2];
        if ((uni::folds_to(y, 'r') || uni::folds_to(y, 'v')) && uni::folds_to(z, 'e')) return e + 3;
      }
      if (uni::folds_to(y, 'm')) return e + 2;
      if (e + 2 < hi && uni::folds_to(y, 'l') && uni::folds_to(c[e + 2], 'l')) return e + 3;
      if (uni::folds_to(y, 'd')) return e + 2;
    }
    return e;
  };
  // A* B+ from p: A* greedy, then given back one code point at a time until B+ can start
  auto word1 = [&](size_t p) -> size_t {
    size_t a = p;
    while (A(a)) ++a;
    for (size_t t = a + 1; t-- > p;)
      if (B(t)) {
        size_t e = t;
        while (B(e)) ++e;
        return contraction(e);
      }
    return 0;
  };
  // A+ B* from p
  auto word2 = [&](size_t p) -> size_t {
    if (!A(p)) return 0;
    size_t e = p;
    while (A(e)) ++e;
    while (B(e)) ++e;
    return contraction(e);
  };
  const char32_t x = c[i];
  const bool lead = !is_nl(x) && !uni::is_letter(x) && !uni::is_number(x);  // [^\r\n\p{L}\p{N}]
  if (lead && i + 1 < hi)
    if (size_t e = word1(i + 1)) return e;
  if (size_t e = word1(i)) return e;
  if (lead && i + 1 < hi)
    if (size_t e = word2(i + 1)) return e;
  if (size_t e = word2(i)) return e;
  // ' ?[\p{P}\p{S}]+[\r\n]*'
  {
    size_t j = hi + 1;
    if (x == U' ' && i + 1 < hi && uni::is_punct_symbol(c[i + 1])) j = i + 1;
    else if (uni::is_punct_symbol(x)) j = i;
    if (j <= hi) {
      while (j < hi && uni::is_punct_symbol(c[j])) ++j;
      while (j < hi && is_nl(c[j])) ++j;
      return j;
    }
  }
  if (!uni::is_space(x)) return 0;
  // \s*[\r\n]+ | \s+(?!\S) | \s+
  size_t k = i;
  while (k < hi && uni::is_space(c[k])) ++k;
  for (size_t q = k; q > i; --q)
    if (is_nl(c[q - 1])) return q;
  if (k == hi || k - 1 == i) return k;
  return k - 1;
}

// Piece ends of the Flint pre-tokenizer over c[0, n), in order.
void flint_pieces(const char32_t *c, size_t n, std::vector<size_t> &ends) {
  // Split 3 over one piece [lo, hi): matches, and the gaps between them as pieces of their own.
  auto words = [&](size_t lo, size_t hi) {
    size_t gap = lo;
    for (size_t i = lo; i < hi;) {
      const size_t e = match_flint(c, hi, i);
      if (!e) {
        ++i;
        continue;
      }
      if (gap < i) ends.push_back(i);
      ends.push_back(e);
      i = gap = e;
    }
    if (gap < hi) ends.push_back(hi);
  };
  // Split 2 over [lo, hi): CJK / kana runs isolated.
  auto cjk = [&](size_t lo, size_t hi) {
    size_t i = lo;
    while (i < hi) {
      size_t j = i;
      const bool in = is_cjk(c[i]);
      while (j < hi && is_cjk(c[j]) == in) ++j;
      words(i, j);  // a CJK run is a piece of its own; split 3 still runs over it
      i = j;
    }
  };
  // Split 1: runs of \p{N} in groups of up to 3.
  size_t i = 0;
  while (i < n) {
    if (uni::is_number(c[i])) {
      size_t j = i;
      while (j < n && j - i < 3 && uni::is_number(c[j])) ++j;
      cjk(i, j);  // splits 2 and 3 run over every piece of split 1
      i = j;
    } else {
      size_t j = i;
      while (j < n && !uni::is_number(c[j])) ++j;
      cjk(i, j);
      i = j;
    }
  }
}

// ---- merge table: (left id, right id) -> (rank, merged id), open addressing --------------------
struct MergeTable {
  std::vector<uint64_t> keys;
  std::vector<std::pair<uint32_t, int32_t>> vals;
  uint64_t mask = 0;
  static constexpr uint64_t kEmpty = ~0ull;
  static uint64_t key(int32_t a, int32_t b) { return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b); }
  static uint64_t hash(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    return k;
  }
  void init(size_t n) {
    size_t cap = 16;
    while (cap < n * 2) cap <<= 1;
    keys.assign(cap, kEmpty);
    vals.assign(cap, {0, 0});
    mask = cap - 1;
  }
  bool insert(int32_t a, int32_t b, uint32_t rank, int32_t merged) {  // false when present
    uint64_t k = key(a, b);
    for (uint64_t h = hash(k) & mask;; h = (h + 1) & mask) {
      if (keys[h] == k) return false;
      if (keys[h] == kEmpty) {
        keys[h] = k;
        vals[h] = {rank, merged};
        return true;
      }
    }
  }
  const std::pair<uint32_t, int32_t> *find(int32_t a, int32_t b) const {
    uint64_t k = key(a, b);
    for (uint64_t h = hash(k) & mask;; h = (h + 1) & mask) {
      if (keys[h] == k) return &vals[h];
      if (keys[h] == kEmpty) return nullptr;
    }
  }
};

bool read_file(const std::string &path, std::string &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  out = std::move(ss).str();
  return true;
}

bool is_null_or(const json::Value *v, std::string_view s) { return !v || v->is_null() || (v->is_string() && v->str == s); }
bool is_false(const json::Value *v) { return !v || v->is_null() || (v->is_bool() && !v->b); }
bool is_type(const json::Value *v, std::string_view t) {
  const std::string *s = v && v->is_object() ? v->get_string("type") : nullptr;
  return s && *s == t;
}

}  // namespace

// ---- Tokenizer ------------------------------------------------------------------------------------

struct Tokenizer::Impl {
  struct Added {
    std::string text;
    int32_t id;
  };
  bool nfc = true;
  bool split_marks = false;  // the pre-tokenizer is kSplitPatternLM
  bool ignore_merges = false;  // BPE ignore_merges: a piece that is itself a vocabulary entry is that one token
  bool flint = false;  // the pre-tokenizer is the Flint sequence (flint_pieces)
  // SentencePiece-style (Metaspace pre-tokenizer, BPE over characters with byte fallback): every ' ' becomes U+2581,
  // the text's first piece gets a leading U+2581 (prepend scheme first / always), a character missing from the vocab
  // becomes its UTF-8 bytes' <0xXX> tokens; decoding maps U+2581 back to ' ', joins each run of byte tokens (a run that is
  // not valid UTF-8 as a whole decodes to one U+FFFD per byte) and strips one leading space of the whole text.
  bool sp = false;
  bool sp_prepend_always = false;  // false: prepend scheme "first" (only a piece at offset 0 of the text)
  bool sp_strip = true;  // the decoder's Strip(' ', 1, 0)
  std::vector<int16_t> byte_of;  // sp: id -> the byte of a <0xXX> token, -1 otherwise
  std::array<int32_t, 256> byte_id{};
  std::unordered_map<std::string, int32_t> by_bytes;  // vocab entry's raw bytes (sp: its text) -> id
  MergeTable merges;
  std::string piece_bytes;  // decode: every id's raw bytes, concatenated
  std::vector<uint32_t> piece_off;
  std::vector<uint8_t> special;
  std::vector<Added> added;  // longest first
  std::array<bool, 256> added_first{};
  int32_t n_ids = 0;

  // Pre-token bytes -> ids. Bounded: cleared when full (texts repeat their words heavily, and
  // a cold cache only costs the BPE again).
  static constexpr size_t kCacheMax = 1 << 16;
  mutable std::mutex cache_mu;
  mutable std::unordered_map<std::string, std::vector<int32_t>> cache;

  bool load(const std::string &dir, std::string &err);
  bool load_sp(const json::Value &root, const std::string &path, std::string &err);
  bool load_added(const json::Value &root, const std::string &path, std::string &err, std::vector<std::pair<int32_t, std::string>> &entries);
  void bpe(std::string_view bytes, std::vector<int32_t> &out) const;
  void merge(std::vector<int32_t> &ids) const;  // rank-ordered merges over the initial symbols
  void bpe_sp(std::string_view word, std::vector<int32_t> &out) const;
  // first: the piece starts at offset 0 of the encoded text (the Metaspace prepend scheme "first")
  void segment(std::string_view text, std::vector<int32_t> &out, bool first) const;
};

bool Tokenizer::Impl::load(const std::string &dir, std::string &err) {
  std::string text;
  std::string path = dir + "/tokenizer.json";
  if (!read_file(path, text)) {
    err = "cannot read " + path;
    return false;
  }
  json::Value root;
  std::string perr;
  if (!json::parse(text, root, &perr)) {
    err = path + ": " + perr;
    return false;
  }
  auto fail = [&](const std::string &m) {
    err = path + ": " + m;
    return false;
  };
  const json::Value *trunc = root.get("truncation");
  if (trunc && !trunc->is_null()) return fail("truncation is not supported");
  // normalizer: NFC or none
  const json::Value *norm = root.get("normalizer");
  if (!norm || norm->is_null()) nfc = false;
  else if (!is_type(norm, "NFC")) return fail("normalizer: only NFC is supported");
  // pre_tokenizer: Metaspace (SentencePiece-style) or Sequence[Split(<the pattern>, Isolated), ByteLevel(no prefix space,
  // no regex)]
  const json::Value *pre = root.get("pre_tokenizer");
  if (is_type(pre, "Metaspace")) return load_sp(root, path, err);
  const json::Value *seq = is_type(pre, "Sequence") ? pre->get_array("pretokenizers") : nullptr;
  if (!seq || (seq->arr.size() != 2 && seq->arr.size() != 4)) return fail("pre_tokenizer: expected Sequence[Split, ByteLevel] or the Flint Sequence[Split x3, ByteLevel]");
  auto split_rx = [&](const json::Value &sv) -> const std::string * {
    const json::Value *pat = is_type(&sv, "Split") ? sv.get_object("pattern") : nullptr;
    const std::string *behavior = sv.get_string("behavior");
    if (!behavior || *behavior != "Isolated" || !is_false(sv.get("invert"))) return nullptr;
    return pat ? pat->get_string("Regex") : nullptr;
  };
  if (seq->arr.size() == 4) {  // Flint: digits, CJK runs, cased words, then ByteLevel
    const std::string *r0 = split_rx(seq->arr[0]), *r1 = split_rx(seq->arr[1]), *r2 = split_rx(seq->arr[2]);
    if (!r0 || !r1 || !r2 || *r0 != kFlintDigits || *r1 != kFlintCjk || *r2 != kFlintWords) return fail("pre_tokenizer: split pattern sequence not implemented");
    flint = true;
  }
  const json::Value &split = seq->arr[0], &bl = seq->arr.back();
  const json::Value *pat = is_type(&split, "Split") ? split.get_object("pattern") : nullptr;
  const std::string *rx = pat ? pat->get_string("Regex") : nullptr;
  if (!flint && (!rx || (*rx != kSplitPattern && *rx != kSplitPatternLM))) return fail("pre_tokenizer: split pattern not implemented");
  // A Qwen2Tokenizer's pre-tokenizer always splits by kSplitPattern, whatever tokenizer.json says (golden
  // tests/data/tokenizer_golden_lm.json). So the mark-aware split applies only to a kSplitPatternLM
  // tokenizer.json whose tokenizer_config.json names another class.
  split_marks = false;
  if (!flint && *rx == kSplitPatternLM) {
    std::string cfg_text;
    json::Value cfg;
    const std::string *cls = nullptr;
    if (read_file(dir + "/tokenizer_config.json", cfg_text) && json::parse(cfg_text, cfg, nullptr)) cls = cfg.get_string("tokenizer_class");
    split_marks = !cls || *cls != "Qwen2Tokenizer";
  }
  const std::string *behavior = split.get_string("behavior");
  if (!behavior || *behavior != "Isolated" || !is_false(split.get("invert")))
    return fail("pre_tokenizer: only an Isolated, non-inverted split is supported");
  if (!is_type(&bl, "ByteLevel") || !is_false(bl.get("add_prefix_space")) || !is_false(bl.get("use_regex")))
    return fail("pre_tokenizer: only ByteLevel(add_prefix_space=false, use_regex=false) is supported");
  if (!is_type(root.get("decoder"), "ByteLevel")) return fail("decoder: only ByteLevel is supported");
  // model: plain byte-level BPE
  const json::Value *model = root.get_object("model");
  if (!is_type(model, "BPE")) return fail("model: only BPE is supported");
  const json::Value *dropout = model->get("dropout");
  if ((dropout && !dropout->is_null()) || !is_null_or(model->get("unk_token"), "") ||
      !is_null_or(model->get("continuing_subword_prefix"), "") || !is_null_or(model->get("end_of_word_suffix"), "") ||
      !is_false(model->get("byte_fallback")))
    return fail("model: dropout, unk, subword affixes and byte_fallback are not supported");
  {
    const json::Value *im = model->get("ignore_merges");
    if (im && !im->is_null() && !im->is_bool()) return fail("model: ignore_merges is not a bool");
    ignore_merges = im && im->is_bool() && im->b;
  }
  const json::Value *vocab = model->get_object("vocab");
  const json::Value *mlist = model->get_array("merges");
  if (!vocab || !mlist) return fail("model: vocab or merges missing");

  const ByteLevel &blv = byte_level();
  std::vector<std::string> bytes_of;
  bytes_of.reserve(vocab->obj.size());
  by_bytes.reserve(vocab->obj.size() * 2);
  std::vector<std::pair<int32_t, std::string>> entries;
  entries.reserve(vocab->obj.size());
  for (const json::Member &m : vocab->obj) {
    if (!m.second.is_int()) return fail("vocab: non-integer id");
    long long id = std::stoll(m.second.str);
    if (id < 0 || id > (1 << 30)) return fail("vocab: id out of range");
    std::string b;
    if (!blv.unmap(m.first, b)) return fail("vocab: entry outside the byte-level alphabet: " + m.first);
    if (!by_bytes.emplace(b, static_cast<int32_t>(id)).second) return fail("vocab: duplicate entry");
    n_ids = std::max(n_ids, static_cast<int32_t>(id) + 1);
    entries.emplace_back(static_cast<int32_t>(id), std::move(b));
  }
  for (int b = 0; b < 256; ++b) {
    auto it = by_bytes.find(std::string(1, static_cast<char>(b)));
    if (it == by_bytes.end()) return fail("vocab: a byte has no token");
    byte_id[b] = it->second;
  }
  merges.init(mlist->arr.size());
  std::string a, b;
  for (size_t r = 0; r < mlist->arr.size(); ++r) {
    const json::Value &m = mlist->arr[r];
    std::string_view sa, sb;
    if (m.is_array() && m.arr.size() == 2 && m.arr[0].is_string() && m.arr[1].is_string()) {
      sa = m.arr[0].str, sb = m.arr[1].str;
    } else if (m.is_string() && m.str.find(' ') != std::string::npos) {  // the older "a b" form
      size_t sp = m.str.find(' ');
      sa = std::string_view(m.str).substr(0, sp), sb = std::string_view(m.str).substr(sp + 1);
    } else {
      return fail("merges: malformed entry");
    }
    a.clear(), b.clear();
    if (!blv.unmap(sa, a) || !blv.unmap(sb, b)) return fail("merges: entry outside the byte-level alphabet");
    auto ia = by_bytes.find(a), ib = by_bytes.find(b), ic = by_bytes.find(a + b);
    if (ia == by_bytes.end() || ib == by_bytes.end() || ic == by_bytes.end())
      return fail("merges: entry " + std::to_string(r) + " refers to a token not in the vocab");
    merges.insert(ia->second, ib->second, static_cast<uint32_t>(r), ic->second);  // first rank wins
  }
  if (!load_added(root, path, err, entries)) return false;
  return true;
}

// Added tokens: matched on the raw text first (a normalised one only when there is no normaliser: then the two match
// the same text); the decode table (an added token's text replaces a vocab entry of the same id) and the special flags.
// entries: the vocab's (id, decoded piece).
bool Tokenizer::Impl::load_added(const json::Value &root, const std::string &path, std::string &err, std::vector<std::pair<int32_t, std::string>> &entries) {
  auto fail = [&](const std::string &m) {
    err = path + ": " + m;
    return false;
  };
  const json::Value *at = root.get_array("added_tokens");
  std::vector<std::pair<int32_t, std::string>> added_entries;
  std::vector<std::pair<int32_t, bool>> added_special;
  if (at) {
    for (const json::Value &t : at->arr) {
      const std::string *content = t.get_string("content");
      const json::Value *id = t.get("id");
      if (!content || content->empty() || !id || !id->is_int()) return fail("added_tokens: malformed entry");
      if (!is_false(t.get("single_word")) || !is_false(t.get("lstrip")) || !is_false(t.get("rstrip")) || (nfc && !is_false(t.get("normalized"))))
        return fail("added_tokens: single_word, lstrip, rstrip and normalized (with a normalizer) are not supported (" + *content + ")");
      int32_t i = static_cast<int32_t>(std::stoll(id->str));
      const json::Value *sp = t.get("special");
      added.push_back({*content, i});
      added_special.emplace_back(i, sp && sp->is_bool() && sp->b);
      added_entries.emplace_back(i, *content);
      n_ids = std::max(n_ids, i + 1);
    }
  }
  std::stable_sort(added.begin(), added.end(), [](const Added &x, const Added &y) { return x.text.size() > y.text.size(); });
  for (const Added &x : added) added_first[static_cast<uint8_t>(x.text[0])] = true;
  // decode table: added tokens' text replaces a vocab entry with the same id
  std::vector<std::string_view> by_id(n_ids);
  for (const auto &e : entries) by_id[e.first] = e.second;
  for (const auto &e : added_entries) by_id[e.first] = e.second;
  piece_off.assign(n_ids + 1, 0);
  for (int32_t i = 0; i < n_ids; ++i) {
    piece_off[i] = static_cast<uint32_t>(piece_bytes.size());
    piece_bytes.append(by_id[i]);
  }
  piece_off[n_ids] = static_cast<uint32_t>(piece_bytes.size());
  special.assign(n_ids, 0);
  for (const auto &s : added_special) special[s.first] = s.second;
  if (sp) {
    byte_of.resize(n_ids, -1);
    for (const auto &e : added_entries) byte_of[e.first] = -1;  // an added token is never a byte
  }
  return true;
}

// tokenizer.json of a SentencePiece-style BPE (e.g. Nanbeige4.3, LlamaTokenizer): no normaliser, Metaspace(U+2581,
// prepend first / always, split false), BPE with byte_fallback over the pieces' characters, decoder Sequence[Replace(U+2581
// -> ' '), ByteFallback, Fuse, Strip(' ', 1, 0)].
bool Tokenizer::Impl::load_sp(const json::Value &root, const std::string &path, std::string &err) {
  auto fail = [&](const std::string &m) {
    err = path + ": " + m;
    return false;
  };
  static const std::string kMeta = "\xE2\x96\x81";  // U+2581
  sp = true;
  if (nfc) return fail("Metaspace pre_tokenizer with a normalizer is not supported");
  const json::Value *pre = root.get("pre_tokenizer");
  const std::string *rep = pre->get_string("replacement"), *scheme = pre->get_string("prepend_scheme");
  if (!rep || *rep != kMeta) return fail("Metaspace: only the replacement U+2581 is supported");
  if (!scheme || (*scheme != "first" && *scheme != "always")) return fail("Metaspace: only prepend_scheme first / always is supported");
  sp_prepend_always = *scheme == "always";
  if (!is_false(pre->get("split"))) return fail("Metaspace: split true is not supported");
  const json::Value *dec = root.get("decoder");
  const json::Value *ds = is_type(dec, "Sequence") ? dec->get_array("decoders") : nullptr;
  if (!ds || ds->arr.size() < 3 || ds->arr.size() > 4 || !is_type(&ds->arr[0], "Replace") || !is_type(&ds->arr[1], "ByteFallback") ||
      !is_type(&ds->arr[2], "Fuse"))
    return fail("decoder: expected Sequence[Replace, ByteFallback, Fuse(, Strip)]");
  {
    const json::Value *pat = ds->arr[0].get_object("pattern");
    const std::string *from = pat ? pat->get_string("String") : nullptr, *to = ds->arr[0].get_string("content");
    if (!from || *from != kMeta || !to || *to != " ") return fail("decoder: Replace must map U+2581 to ' '");
  }
  sp_strip = false;
  if (ds->arr.size() == 4) {
    const json::Value &st = ds->arr[3];
    const std::string *c = st.get_string("content");
    const json::Value *a = st.get("start"), *b = st.get("stop");
    if (!is_type(&st, "Strip") || !c || *c != " " || !a || !a->is_int() || a->str != "1" || !b || !b->is_int() || b->str != "0")
      return fail("decoder: only Strip(' ', 1, 0) is supported");
    sp_strip = true;
  }
  const json::Value *model = root.get_object("model");
  if (!is_type(model, "BPE")) return fail("model: only BPE is supported");
  const json::Value *dropout = model->get("dropout");
  if ((dropout && !dropout->is_null()) || !is_null_or(model->get("continuing_subword_prefix"), "") ||
      !is_null_or(model->get("end_of_word_suffix"), "") || !is_false(model->get("ignore_merges")))
    return fail("model: dropout, subword affixes and ignore_merges are not supported");
  const json::Value *bf = model->get("byte_fallback");
  if (!bf || !bf->is_bool() || !bf->b) return fail("model: a Metaspace tokenizer needs byte_fallback");
  const json::Value *vocab = model->get_object("vocab");
  const json::Value *mlist = model->get_array("merges");
  if (!vocab || !mlist) return fail("model: vocab or merges missing");
  std::vector<std::pair<int32_t, std::string>> entries;
  entries.reserve(vocab->obj.size());
  by_bytes.reserve(vocab->obj.size() * 2);
  for (const json::Member &m : vocab->obj) {
    if (!m.second.is_int()) return fail("vocab: non-integer id");
    long long id = std::stoll(m.second.str);
    if (id < 0 || id > (1 << 30)) return fail("vocab: id out of range");
    if (!by_bytes.emplace(m.first, static_cast<int32_t>(id)).second) return fail("vocab: duplicate entry");
    n_ids = std::max(n_ids, static_cast<int32_t>(id) + 1);
    std::string piece;  // the decoded text: U+2581 -> ' '
    for (size_t i = 0; i < m.first.size();) {
      if (m.first.compare(i, kMeta.size(), kMeta) == 0) {
        piece.push_back(' ');
        i += kMeta.size();
      } else {
        piece.push_back(m.first[i++]);
      }
    }
    entries.emplace_back(static_cast<int32_t>(id), std::move(piece));
  }
  byte_of.assign(n_ids, -1);
  for (int b = 0; b < 256; ++b) {
    char name[8];
    std::snprintf(name, sizeof name, "<0x%02X>", b);
    auto it = by_bytes.find(name);
    if (it == by_bytes.end()) return fail(std::string("vocab: no byte-fallback token ") + name);
    byte_id[b] = it->second;
    byte_of[it->second] = static_cast<int16_t>(b);
  }
  for (auto &e : entries)
    if (byte_of[e.first] >= 0) e.second.assign(1, static_cast<char>(byte_of[e.first]));
  merges.init(mlist->arr.size());
  for (size_t r = 0; r < mlist->arr.size(); ++r) {
    const json::Value &m = mlist->arr[r];
    std::string a, b;
    if (m.is_array() && m.arr.size() == 2 && m.arr[0].is_string() && m.arr[1].is_string()) {
      a = m.arr[0].str, b = m.arr[1].str;
    } else if (m.is_string() && m.str.find(' ') != std::string::npos) {
      size_t sp_at = m.str.find(' ');
      a = m.str.substr(0, sp_at), b = m.str.substr(sp_at + 1);
    } else {
      return fail("merges: malformed entry");
    }
    auto ia = by_bytes.find(a), ib = by_bytes.find(b), ic = by_bytes.find(a + b);
    if (ia == by_bytes.end() || ib == by_bytes.end() || ic == by_bytes.end())
      return fail("merges: entry " + std::to_string(r) + " refers to a token not in the vocab");
    merges.insert(ia->second, ib->second, static_cast<uint32_t>(r), ic->second);  // first rank wins
  }
  return load_added(root, path, err, entries);
}

void Tokenizer::Impl::merge(std::vector<int32_t> &ids) const {
  // Symbols in a linked list; candidate pairs in a min-heap ordered by (rank, position): the
  // lowest-ranked merge applies first, leftmost first among equal ranks. A popped candidate is
  // stale when either side has changed since it was pushed.
  struct Sym {
    int32_t id, prev, next;
  };
  struct Cand {
    uint32_t rank;
    int32_t pos, left, right, merged;
    bool operator>(const Cand &o) const { return rank != o.rank ? rank > o.rank : pos > o.pos; }
  };
  const int32_t n = static_cast<int32_t>(ids.size());
  if (n < 2) return;
  std::vector<Sym> sym(n);
  for (int32_t i = 0; i < n; ++i) sym[i] = {ids[i], i - 1, i + 1 < n ? i + 1 : -1};
  std::vector<Cand> heap;
  heap.reserve(n);
  auto push = [&](int32_t p) {
    int32_t q = sym[p].next;
    if (q < 0) return;
    if (const auto *m = merges.find(sym[p].id, sym[q].id)) {
      heap.push_back({m->first, p, sym[p].id, sym[q].id, m->second});
      std::push_heap(heap.begin(), heap.end(), std::greater<>());
    }
  };
  for (int32_t i = 0; i + 1 < n; ++i) push(i);
  while (!heap.empty()) {
    std::pop_heap(heap.begin(), heap.end(), std::greater<>());
    Cand c = heap.back();
    heap.pop_back();
    Sym &l = sym[c.pos];
    if (l.id != c.left || l.next < 0 || sym[l.next].id != c.right) continue;
    int32_t r = l.next;
    l.id = c.merged;
    l.next = sym[r].next;
    if (l.next >= 0) sym[l.next].prev = c.pos;
    sym[r].id = -1;  // dead
    if (l.prev >= 0) push(l.prev);
    push(c.pos);
  }
  ids.clear();
  for (int32_t i = 0; i >= 0; i = sym[i].next) ids.push_back(sym[i].id);
}

void Tokenizer::Impl::bpe(std::string_view bytes, std::vector<int32_t> &out) const {
  if (bytes.size() == 1) {
    out.push_back(byte_id[static_cast<uint8_t>(bytes[0])]);
    return;
  }
  if (ignore_merges) {
    auto it = by_bytes.find(std::string(bytes));
    if (it != by_bytes.end()) {
      out.push_back(it->second);
      return;
    }
  }
  {
    std::lock_guard<std::mutex> lock(cache_mu);
    auto it = cache.find(std::string(bytes));
    if (it != cache.end()) {
      out.insert(out.end(), it->second.begin(), it->second.end());
      return;
    }
  }
  std::vector<int32_t> ids(bytes.size());
  for (size_t i = 0; i < bytes.size(); ++i) ids[i] = byte_id[static_cast<uint8_t>(bytes[i])];
  merge(ids);
  out.insert(out.end(), ids.begin(), ids.end());
  std::lock_guard<std::mutex> lock(cache_mu);
  if (cache.size() >= kCacheMax) cache.clear();
  cache.emplace(std::string(bytes), std::move(ids));
}

// One Metaspace piece (already U+2581-mapped): the initial symbols are its characters (a character missing from the
// vocab: its UTF-8 bytes' <0xXX> tokens), then the merges. Pieces are whole texts between added tokens (split false), so
// only short ones are cached.
void Tokenizer::Impl::bpe_sp(std::string_view word, std::vector<int32_t> &out) const {
  constexpr size_t kCacheWord = 256;
  if (word.size() <= kCacheWord) {
    std::lock_guard<std::mutex> lock(cache_mu);
    auto it = cache.find(std::string(word));
    if (it != cache.end()) {
      out.insert(out.end(), it->second.begin(), it->second.end());
      return;
    }
  }
  std::vector<int32_t> ids;
  ids.reserve(word.size());
  std::u32string cps;
  uni::decode_utf8(word, cps);
  std::string ch;
  for (char32_t c : cps) {
    ch.clear();
    uni::append_utf8(c, ch);
    auto it = by_bytes.find(ch);
    if (it != by_bytes.end()) {
      ids.push_back(it->second);
    } else {
      for (char b : ch) ids.push_back(byte_id[static_cast<uint8_t>(b)]);
    }
  }
  merge(ids);
  out.insert(out.end(), ids.begin(), ids.end());
  if (word.size() <= kCacheWord) {
    std::lock_guard<std::mutex> lock(cache_mu);
    if (cache.size() >= kCacheMax) cache.clear();
    cache.emplace(std::string(word), std::move(ids));
  }
}

void Tokenizer::Impl::segment(std::string_view text, std::vector<int32_t> &out, bool first) const {
  if (text.empty()) return;
  if (sp) {
    static const std::string_view kMeta = "\xE2\x96\x81";
    std::string w;
    w.reserve(text.size() + 3 * 8);
    for (char ch : text) {
      if (ch == ' ') w.append(kMeta);
      else w.push_back(ch);
    }
    if ((first || sp_prepend_always) && w.compare(0, kMeta.size(), kMeta) != 0) w.insert(0, kMeta);
    bpe_sp(w, out);
    return;
  }
  bool ascii = std::all_of(text.begin(), text.end(), [](char ch) { return static_cast<uint8_t>(ch) < 0x80; });
  std::u32string cps;
  uni::decode_utf8(text, cps);
  if (!ascii && nfc) {  // ASCII is its own NFC
    std::u32string n;
    uni::nfc(cps, n);
    cps.swap(n);
  }
  std::string piece;
  if (flint) {
    std::vector<size_t> ends;
    flint_pieces(cps.data(), cps.size(), ends);
    size_t i = 0;
    for (size_t j : ends) {
      if (ascii) {
        bpe(text.substr(i, j - i), out);
      } else {
        piece.clear();
        for (size_t k = i; k < j; ++k) uni::append_utf8(cps[k], piece);
        bpe(piece, out);
      }
      i = j;
    }
    return;
  }
  for (size_t i = 0; i < cps.size();) {
    size_t j = match_at(cps.data(), cps.size(), i, split_marks);
    if (ascii) {
      bpe(text.substr(i, j - i), out);  // code points are bytes
    } else {
      piece.clear();
      for (size_t k = i; k < j; ++k) uni::append_utf8(cps[k], piece);
      bpe(piece, out);
    }
    i = j;
  }
}

Tokenizer::Tokenizer() : impl_(std::make_unique<Impl>()) {}
Tokenizer::~Tokenizer() = default;

bool Tokenizer::load(const std::string &model_dir, std::string &err) {
  auto fresh = std::make_unique<Impl>();
  if (!fresh->load(model_dir, err)) return false;
  impl_ = std::move(fresh);
  return true;
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
  std::vector<int32_t> out;
  const Impl &m = *impl_;
  // Added tokens split the text first: at each position the longest one starting there wins.
  size_t seg = 0;
  for (size_t i = 0; i < text.size();) {
    if (m.added_first[static_cast<uint8_t>(text[i])]) {
      const Impl::Added *hit = nullptr;
      for (const Impl::Added &a : m.added)
        if (text.compare(i, a.text.size(), a.text) == 0) {
          hit = &a;  // sorted longest first
          break;
        }
      if (hit) {
        m.segment(text.substr(seg, i - seg), out, seg == 0);
        out.push_back(hit->id);
        i += hit->text.size();
        seg = i;
        continue;
      }
    }
    ++i;
  }
  m.segment(text.substr(seg), out, seg == 0);
  return out;
}

int32_t Tokenizer::vocab_size() const { return impl_->n_ids; }

void Tokenizer::decode_piece(int32_t id, std::string &out, bool skip_special) const {
  const Impl &m = *impl_;
  if (id < 0 || id >= m.n_ids || (skip_special && m.special[id])) return;
  out.append(m.piece_bytes, m.piece_off[id], m.piece_off[id + 1] - m.piece_off[id]);
}

std::string Tokenizer::decode(const std::vector<int32_t> &ids, bool skip_special) const {
  if (impl_->sp) {  // the streaming rules are the decoder's (byte runs, the leading-space strip): one implementation
    Detokenizer d(*this, skip_special);
    std::string out;
    for (int32_t id : ids) d.push(id);
    d.flush(out);
    return out;
  }
  std::string raw, out;
  for (int32_t id : ids) decode_piece(id, raw, skip_special);
  sanitize(raw, out, false);
  return out;
}

int32_t Tokenizer::token_id(std::string_view token) const {
  const Impl &m = *impl_;
  for (const Impl::Added &a : m.added)
    if (a.text == token) return a.id;
  if (m.sp) {
    auto it = m.by_bytes.find(std::string(token));
    return it == m.by_bytes.end() ? -1 : it->second;
  }
  std::string b;
  if (!byte_level().unmap(token, b)) return -1;
  auto it = m.by_bytes.find(b);
  return it == m.by_bytes.end() ? -1 : it->second;
}

bool Tokenizer::is_special(int32_t id) const { return id >= 0 && id < impl_->n_ids && impl_->special[id]; }

bool Tokenizer::byte_fallback() const { return impl_->sp; }

int Tokenizer::byte_value(int32_t id) const {
  const Impl &m = *impl_;
  return m.sp && id >= 0 && id < m.n_ids ? m.byte_of[id] : -1;
}

bool Tokenizer::strips_leading_space() const { return impl_->sp && impl_->sp_strip; }

// ---- Detokenizer ----------------------------------------------------------------------------------

// SentencePiece-style decoding (Tokenizer:byte_fallback): byte tokens collect in run_ until a text token (or the
// flush) ends the run: valid UTF-8 as a whole -> its text, else one U+FFFD per byte (the decoder's ByteFallback); a
// skipped special token is dropped before the decoder, so a run continues across it. The first non-empty text loses
// one leading ' ' (Strip). pending_ holds only resolved text there.
void Detokenizer::push(int32_t id) {
  if (!tok_->byte_fallback()) {
    tok_->decode_piece(id, pending_, skip_special_);
    return;
  }
  if (id < 0 || id >= tok_->vocab_size() || (skip_special_ && tok_->is_special(id))) return;
  const int b = tok_->byte_value(id);
  if (b >= 0) {
    run_.push_back(static_cast<char>(b));
    return;
  }
  end_run();
  tok_->decode_piece(id, pending_, skip_special_);
}

void Detokenizer::end_run() {
  if (run_.empty()) return;
  size_t i = 0, len;
  while (i < run_.size() && utf8_at(run_, i, len) == Utf8::complete) i += len;
  if (i == run_.size()) {
    pending_.append(run_);
  } else {
    for (size_t k = 0; k < run_.size(); ++k) pending_.append(kReplacement);
  }
  run_.clear();
}

void Detokenizer::emit(std::string &out) {
  if (pending_.empty()) return;
  size_t from = 0;
  if (!started_) {
    started_ = true;
    if (tok_->strips_leading_space() && pending_[0] == ' ') from = 1;
  }
  out.append(pending_, from, std::string::npos);
  pending_.clear();
}

void Detokenizer::take(std::string &out) {
  if (tok_->byte_fallback()) {
    emit(out);
    return;
  }
  size_t used = sanitize(pending_, out, true);
  pending_.erase(0, used);
}

void Detokenizer::flush(std::string &out) {
  if (tok_->byte_fallback()) {
    end_run();
    emit(out);
    return;
  }
  sanitize(pending_, out, false);
  pending_.clear();
}

// ---- StopMatcher ----------------------------------------------------------------------------------

StopMatcher::StopMatcher(std::vector<std::string> stops) {
  for (std::string &s : stops)
    if (!s.empty()) {
      max_len_ = std::max(max_len_, s.size());
      stops_.push_back(std::move(s));
    }
}

bool StopMatcher::feed(std::string_view piece, std::string &out) {
  if (stopped_) return true;
  held_.append(piece);
  if (stops_.empty()) {
    out.append(held_);
    held_.clear();
    return false;
  }
  // The earliest match wins (every stop string is searched; with streaming that is the one that
  // completes first, except when two complete within one piece).
  size_t best = std::string::npos;
  for (const std::string &s : stops_) best = std::min(best, held_.find(s));
  if (best != std::string::npos) {
    out.append(held_, 0, best);
    held_.clear();
    stopped_ = true;
    return true;
  }
  // Hold back the longest suffix that could still begin a stop string. Stop strings are valid
  // UTF-8, so such a suffix starts on a character boundary.
  size_t keep = 0;
  for (size_t k = std::min(held_.size(), max_len_ - 1); k > 0 && !keep; --k) {
    std::string_view tail = std::string_view(held_).substr(held_.size() - k);
    for (const std::string &s : stops_)
      if (s.size() > k && s.compare(0, k, tail) == 0) {
        keep = k;
        break;
      }
  }
  out.append(held_, 0, held_.size() - keep);
  held_.erase(0, held_.size() - keep);
  return false;
}

void StopMatcher::flush(std::string &out) {
  if (!stopped_) out.append(held_);
  held_.clear();
}

}  // namespace xenithium:text
