// The chat-template Jinja interpreter (jinja.hpp). One pass splits the source into text and tags (with the
// environment's whitespace rules), a recursive-descent parser builds statement and expression trees, and a tree-walking
// evaluator renders them over Python-like values.
#include "xenithium/text/jinja.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xenithium::text::jinja {
namespace {

struct Error {
  std::string msg;
};
[[noreturn]] void fail(std::string m) { throw Error{std::move(m)}; }

// ---- UTF-8 -----------------------------------------------------------------------------------------------------

size_t cp_size(unsigned char c) { return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1; }

std::vector<std::string_view> code_points(std::string_view s) {
  std::vector<std::string_view> v;
  for (size_t i = 0; i < s.size();) {
    const size_t n = std::min(cp_size(static_cast<unsigned char>(s[i])), s.size() - i);
    v.push_back(s.substr(i, n));
    i += n;
  }
  return v;
}

uint32_t decode_cp(std::string_view c) {
  const auto u = [&](size_t i) { return static_cast<uint32_t>(static_cast<unsigned char>(c[i])); };
  switch (c.size()) {
    case 1: return u(0);
    case 2: return ((u(0) & 0x1f) << 6) | (u(1) & 0x3f);
    case 3: return ((u(0) & 0x0f) << 12) | ((u(1) & 0x3f) << 6) | (u(2) & 0x3f);
    default: return ((u(0) & 0x07) << 18) | ((u(1) & 0x3f) << 12) | ((u(2) & 0x3f) << 6) | (u(3) & 0x3f);
  }
}

void encode_cp(uint32_t c, std::string &o) {
  if (c < 0x80) {
    o += static_cast<char>(c);
  } else if (c < 0x800) {
    o += static_cast<char>(0xc0 | (c >> 6)), o += static_cast<char>(0x80 | (c & 0x3f));
  } else if (c < 0x10000) {
    o += static_cast<char>(0xe0 | (c >> 12)), o += static_cast<char>(0x80 | ((c >> 6) & 0x3f)), o += static_cast<char>(0x80 | (c & 0x3f));
  } else {
    o += static_cast<char>(0xf0 | (c >> 18)), o += static_cast<char>(0x80 | ((c >> 12) & 0x3f));
    o += static_cast<char>(0x80 | ((c >> 6) & 0x3f)), o += static_cast<char>(0x80 | (c & 0x3f));
  }
}

// Python's str.isspace code points
bool py_space(uint32_t c) {
  return c == ' ' || (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x1f) || c == 0x85 || c == 0xa0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

// ---- values ----------------------------------------------------------------------------------------------------

struct Val;
struct Frame;
using Args = std::vector<Val>;
using Kw = std::vector<std::pair<std::string, Val>>;
using Fn = std::function<Val(Args &, Kw &)>;
using Items = std::vector<std::pair<std::string, Val>>;

struct Val {
  enum K : uint8_t { undef, none, boolean, integer, real, str, list, dict, func };
  K k = undef;
  bool b = false;
  bool tuple = false;  // list: a tuple
  bool ns = false;  // dict: a namespace (attributes assignable)
  int64_t i = 0;
  double d = 0;
  std::string s;  // str payload; integer: the lexeme when it does not fit int64; undef: what was looked up
  std::shared_ptr<std::vector<Val>> l;
  std::shared_ptr<Items> m;
  std::shared_ptr<Fn> f;

  static Val Undef(std::string what = {}) {
    Val v;
    v.s = std::move(what);
    return v;
  }
  static Val None() {
    Val v;
    v.k = none;
    return v;
  }
  static Val Bool(bool x) {
    Val v;
    v.k = boolean, v.b = x;
    return v;
  }
  static Val Int(int64_t x) {
    Val v;
    v.k = integer, v.i = x;
    return v;
  }
  static Val Real(double x) {
    Val v;
    v.k = real, v.d = x;
    return v;
  }
  static Val Str(std::string x) {
    Val v;
    v.k = str, v.s = std::move(x);
    return v;
  }
  static Val List(std::vector<Val> x = {}, bool tup = false) {
    Val v;
    v.k = list, v.l = std::make_shared<std::vector<Val>>(std::move(x)), v.tuple = tup;
    return v;
  }
  static Val Dict() {
    Val v;
    v.k = dict, v.m = std::make_shared<Items>();
    return v;
  }
  static Val Func(Fn fn) {
    Val v;
    v.k = func, v.f = std::make_shared<Fn>(std::move(fn));
    return v;
  }
  const Val *find(std::string_view key) const {
    for (const auto &kv : *m)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  void put(const std::string &key, Val v) {
    for (auto &kv : *m)
      if (kv.first == key) {
        kv.second = std::move(v);
        return;
      }
    m->emplace_back(key, std::move(v));
  }
  bool numeric() const { return k == boolean || k == integer || k == real; }
  double as_double() const { return k == real ? d : k == boolean ? (b ? 1.0 : 0.0) : static_cast<double>(i); }
  int64_t as_int() const { return k == boolean ? (b ? 1 : 0) : i; }
};

Val from_json(const json::Value &j) {
  switch (j.type) {
    case json::Type::null: return Val::None();
    case json::Type::boolean: return Val::Bool(j.b);
    case json::Type::number: {
      if (j.is_int() && std::isfinite(j.num) && !j.str.empty() && (std::isdigit(static_cast<unsigned char>(j.str[0])) || j.str[0] == '-')) {
        Val v = Val::Int(0);
        errno = 0;
        char *end = nullptr;
        const long long x = std::strtoll(j.str.c_str(), &end, 10);
        if (errno == ERANGE) v.s = j.str;  // past int64: kept as Python prints it
        else v.i = x;
        return v;
      }
      return Val::Real(j.num);
    }
    case json::Type::string: return Val::Str(j.str);
    case json::Type::array: {
      std::vector<Val> v;
      v.reserve(j.arr.size());
      for (const auto &e : j.arr) v.push_back(from_json(e));
      return Val::List(std::move(v));
    }
    case json::Type::object: {
      Val d = Val::Dict();
      for (const auto &kv : j.obj) d.put(kv.first, from_json(kv.second));
      return d;
    }
  }
  return Val::None();
}

bool truthy(const Val &v) {
  switch (v.k) {
    case Val::undef:
    case Val::none: return false;
    case Val::boolean: return v.b;
    case Val::integer: return !v.s.empty() || v.i != 0;
    case Val::real: return v.d != 0;
    case Val::str: return !v.s.empty();
    case Val::list: return !v.l->empty();
    case Val::dict: return !v.m->empty();
    case Val::func: return true;
  }
  return false;
}

const char *type_name(const Val &v) {
  switch (v.k) {
    case Val::undef: return "Undefined";
    case Val::none: return "NoneType";
    case Val::boolean: return "bool";
    case Val::integer: return "int";
    case Val::real: return "float";
    case Val::str: return "str";
    case Val::list: return v.tuple ? "tuple" : "list";
    case Val::dict: return v.ns ? "Namespace" : "dict";
    case Val::func: return "function";
  }
  return "object";
}

void py_repr(const Val &v, std::string &o);

std::string int_str(const Val &v) { return v.s.empty() ? std::to_string(v.i) : v.s; }

// Python's str
void py_str(const Val &v, std::string &o) {
  switch (v.k) {
    case Val::undef: return;
    case Val::none: o += "None"; return;
    case Val::boolean: o += v.b ? "True" : "False"; return;
    case Val::integer: o += int_str(v); return;
    case Val::real: o += json::python_float_repr(v.d); return;
    case Val::str: o += v.s; return;
    default: py_repr(v, o);
  }
}
std::string to_str(const Val &v) {
  std::string o;
  py_str(v, o);
  return o;
}

void py_repr_string(std::string_view s, std::string &o) {
  const bool dq = s.find('\'') != std::string_view::npos && s.find('"') == std::string_view::npos;
  const char q = dq ? '"' : '\'';
  o += q;
  for (const auto cp : code_points(s)) {
    const uint32_t c = decode_cp(cp);
    if (c == static_cast<uint32_t>(q) || c == '\\') o += '\\', o += static_cast<char>(c);
    else if (c == '\n') o += "\\n";
    else if (c == '\r') o += "\\r";
    else if (c == '\t') o += "\\t";
    else if (c < 0x20 || c == 0x7f) {
      char b[8];
      std::snprintf(b, sizeof b, "\\x%02x", c);
      o += b;
    } else o += cp;
  }
  o += q;
}

void py_repr(const Val &v, std::string &o) {
  switch (v.k) {
    case Val::str: py_repr_string(v.s, o); return;
    case Val::list: {
      o += v.tuple ? '(' : '[';
      for (size_t i = 0; i < v.l->size(); ++i) {
        if (i) o += ", ";
        py_repr((*v.l)[i], o);
      }
      if (v.tuple && v.l->size() == 1) o += ',';
      o += v.tuple ? ')' : ']';
      return;
    }
    case Val::dict: {
      if (v.ns) o += "<Namespace ";
      o += '{';
      bool first = true;
      for (const auto &kv : *v.m) {
        if (!first) o += ", ";
        first = false;
        py_repr_string(kv.first, o);
        o += ": ";
        py_repr(kv.second, o);
      }
      o += '}';
      if (v.ns) o += '>';
      return;
    }
    case Val::func: o += "<function>"; return;
    case Val::undef: return;
    default: py_str(v, o);
  }
}

// json.dumps as the chat templates' tojson filter calls it
struct DumpOpt {
  bool ensure_ascii = false, sort_keys = false;
  bool indent = false;
  std::string indent_str, item_sep = ", ", key_sep = ": ";
};

void dump_string(std::string_view s, bool ascii, std::string &o) {
  o += '"';
  for (const auto cp : code_points(s)) {
    const uint32_t c = decode_cp(cp);
    switch (c) {
      case '"': o += "\\\""; continue;
      case '\\': o += "\\\\"; continue;
      case '\n': o += "\\n"; continue;
      case '\r': o += "\\r"; continue;
      case '\t': o += "\\t"; continue;
      case '\b': o += "\\b"; continue;
      case '\f': o += "\\f"; continue;
      default: break;
    }
    char b[16];
    if (c < 0x20) {
      std::snprintf(b, sizeof b, "\\u%04x", c);
      o += b;
    } else if (ascii && c >= 0x80) {
      if (c >= 0x10000) {
        const uint32_t x = c - 0x10000;
        std::snprintf(b, sizeof b, "\\u%04x\\u%04x", 0xd800 + (x >> 10), 0xdc00 + (x & 0x3ff));
      } else {
        std::snprintf(b, sizeof b, "\\u%04x", c);
      }
      o += b;
    } else {
      o += cp;
    }
  }
  o += '"';
}

void dump(const Val &v, const DumpOpt &op, int level, std::string &o) {
  auto nl = [&](int lv) {
    o += '\n';
    for (int k = 0; k < lv; ++k) o += op.indent_str;
  };
  switch (v.k) {
    case Val::none: o += "null"; return;
    case Val::boolean: o += v.b ? "true" : "false"; return;
    case Val::integer: o += int_str(v); return;
    case Val::real:
      if (std::isnan(v.d)) o += "NaN";
      else if (std::isinf(v.d)) o += v.d > 0 ? "Infinity" : "-Infinity";
      else o += json::python_float_repr(v.d);
      return;
    case Val::str: dump_string(v.s, op.ensure_ascii, o); return;
    case Val::list: {
      if (v.l->empty()) {
        o += "[]";
        return;
      }
      o += '[';
      for (size_t i = 0; i < v.l->size(); ++i) {
        if (i) o += op.item_sep;
        if (op.indent) nl(level + 1);
        dump((*v.l)[i], op, level + 1, o);
      }
      if (op.indent) nl(level);
      o += ']';
      return;
    }
    case Val::dict: {
      if (v.ns) fail("Object of type Namespace is not JSON serializable");
      if (v.m->empty()) {
        o += "{}";
        return;
      }
      std::vector<const std::pair<std::string, Val> *> items;
      for (const auto &kv : *v.m) items.push_back(&kv);
      if (op.sort_keys) std::stable_sort(items.begin(), items.end(), [](auto *a, auto *b) { return a->first < b->first; });
      o += '{';
      for (size_t i = 0; i < items.size(); ++i) {
        if (i) o += op.item_sep;
        if (op.indent) nl(level + 1);
        dump_string(items[i]->first, op.ensure_ascii, o);
        o += op.key_sep;
        dump(items[i]->second, op, level + 1, o);
      }
      if (op.indent) nl(level);
      o += '}';
      return;
    }
    default: fail(std::string("Object of type ") + type_name(v) + " is not JSON serializable");
  }
}

bool py_eq(const Val &a, const Val &b) {
  if (a.numeric() && b.numeric()) {
    if (a.k == Val::real || b.k == Val::real) return a.as_double() == b.as_double();
    if (!a.s.empty() || !b.s.empty()) return a.k == b.k && int_str(a) == int_str(b);
    return a.as_int() == b.as_int();
  }
  if (a.k != b.k) return false;
  switch (a.k) {
    case Val::undef:
    case Val::none: return true;
    case Val::str: return a.s == b.s;
    case Val::list:
      if (a.tuple != b.tuple || a.l->size() != b.l->size()) return false;
      for (size_t i = 0; i < a.l->size(); ++i)
        if (!py_eq((*a.l)[i], (*b.l)[i])) return false;
      return true;
    case Val::dict:
      if (a.m->size() != b.m->size()) return false;
      for (const auto &kv : *a.m) {
        const Val *o = b.find(kv.first);
        if (!o || !py_eq(kv.second, *o)) return false;
      }
      return true;
    case Val::func: return a.f == b.f;
    default: return false;
  }
}

int py_cmp(const Val &a, const Val &b) {  // <0, 0, >0; errors when unordered
  if (a.numeric() && b.numeric()) {
    const double x = a.as_double(), y = b.as_double();
    if (a.k != Val::real && b.k != Val::real) return a.as_int() < b.as_int() ? -1 : a.as_int() > b.as_int();
    return x < y ? -1 : x > y;
  }
  if (a.k == Val::str && b.k == Val::str) return a.s < b.s ? -1 : a.s > b.s;
  if (a.k == Val::list && b.k == Val::list) {
    for (size_t i = 0; i < std::min(a.l->size(), b.l->size()); ++i) {
      if (py_eq((*a.l)[i], (*b.l)[i])) continue;
      return py_cmp((*a.l)[i], (*b.l)[i]);
    }
    return a.l->size() < b.l->size() ? -1 : a.l->size() > b.l->size();
  }
  fail(std::string("'<' not supported between instances of '") + type_name(a) + "' and '" + type_name(b) + "'");
}

[[noreturn]] void undefined_error(const Val &u) {
  if (u.s.find("' has no ") != std::string::npos) fail("'" + u.s + "'");
  fail(u.s.empty() ? "undefined value" : "'" + u.s + "' is undefined");
}

std::vector<Val> iterate(const Val &v) {
  switch (v.k) {
    case Val::undef: return {};
    case Val::list: return *v.l;
    case Val::dict: {
      std::vector<Val> r;
      for (const auto &kv : *v.m) r.push_back(Val::Str(kv.first));
      return r;
    }
    case Val::str: {
      std::vector<Val> r;
      for (const auto cp : code_points(v.s)) r.push_back(Val::Str(std::string(cp)));
      return r;
    }
    default: fail(std::string("'") + type_name(v) + "' object is not iterable");
  }
}

int64_t py_len(const Val &v) {
  switch (v.k) {
    case Val::undef: return 0;
    case Val::str: return static_cast<int64_t>(code_points(v.s).size());
    case Val::list: return static_cast<int64_t>(v.l->size());
    case Val::dict: return static_cast<int64_t>(v.m->size());
    default: fail(std::string("object of type '") + type_name(v) + "' has no len()");
  }
}

// str.strip family: chars = null -> whitespace
std::string py_strip(std::string_view s, const Val *chars, bool left, bool right) {
  auto cps = code_points(s);
  std::vector<uint32_t> set;
  const bool ws = !chars || chars->k == Val::none || chars->k == Val::undef;
  if (!ws) {
    if (chars->k != Val::str) fail("strip arg must be None or str");
    for (const auto c : code_points(chars->s)) set.push_back(decode_cp(c));
  }
  auto hit = [&](std::string_view cp) {
    const uint32_t c = decode_cp(cp);
    return ws ? py_space(c) : std::find(set.begin(), set.end(), c) != set.end();
  };
  size_t a = 0, b = cps.size();
  if (left)
    while (a < b && hit(cps[a])) ++a;
  if (right)
    while (b > a && hit(cps[b - 1])) --b;
  std::string o;
  for (size_t i = a; i < b; ++i) o += cps[i];
  return o;
}

std::string py_replace(const std::string &s, const std::string &from, const std::string &to, int64_t count) {
  std::string o;
  if (from.empty()) {
    const auto cps = code_points(s);
    int64_t n = 0;
    for (size_t i = 0; i <= cps.size(); ++i) {
      if (count < 0 || n < count) o += to, ++n;
      if (i < cps.size()) o += cps[i];
    }
    return o;
  }
  size_t pos = 0;
  int64_t n = 0;
  while (count < 0 || n < count) {
    const size_t at = s.find(from, pos);
    if (at == std::string::npos) break;
    o.append(s, pos, at - pos);
    o += to;
    pos = at + from.size();
    ++n;
  }
  o.append(s, pos, std::string::npos);
  return o;
}

std::vector<Val> py_split(const std::string &s, const Val *sep, int64_t maxsplit) {
  std::vector<Val> r;
  if (!sep || sep->k == Val::none || sep->k == Val::undef) {
    const auto cps = code_points(s);
    size_t i = 0;
    while (true) {
      while (i < cps.size() && py_space(decode_cp(cps[i]))) ++i;
      if (i >= cps.size()) break;
      if (maxsplit >= 0 && static_cast<int64_t>(r.size()) >= maxsplit) {
        std::string rest;
        for (size_t j = i; j < cps.size(); ++j) rest += cps[j];
        // Python strips the trailing whitespace of the remainder only when it splits again; the remainder is kept as is
        r.push_back(Val::Str(rest));
        return r;
      }
      std::string w;
      while (i < cps.size() && !py_space(decode_cp(cps[i]))) w += cps[i++];
      r.push_back(Val::Str(w));
    }
    return r;
  }
  if (sep->k != Val::str) fail("must be str or None");
  if (sep->s.empty()) fail("empty separator");
  size_t pos = 0;
  while (true) {
    const size_t at = s.find(sep->s, pos);
    if (at == std::string::npos || (maxsplit >= 0 && static_cast<int64_t>(r.size()) >= maxsplit)) break;
    r.push_back(Val::Str(s.substr(pos, at - pos)));
    pos = at + sep->s.size();
  }
  r.push_back(Val::Str(s.substr(pos)));
  return r;
}

std::string ascii_case(std::string s, bool upper) {
  for (auto &c : s)
    if (upper ? (c >= 'a' && c <= 'z') : (c >= 'A' && c <= 'Z')) c = static_cast<char>(c ^ 0x20);
  return s;
}

// ---- template source -> segments --------------------------------------------------------------------------------

struct Seg {
  enum K : uint8_t { text, expr, stmt } k = text;
  std::string body;
  size_t line = 1;
};

std::vector<Seg> segment(std::string_view src) {
  // Jinja drops one trailing newline of the source (keep_trailing_newline = False)
  if (!src.empty() && src.back() == '\n') {
    src.remove_suffix(1);
    if (!src.empty() && src.back() == '\r') src.remove_suffix(1);
  }
  struct Raw {
    Seg::K k;
    std::string body;
    bool lminus = false, rminus = false, lplus = false, rplus = false, comment = false;
    bool ate_newline = false;  // the tag's right side consumed a newline (the next text starts a line)
    size_t line = 1;
  };
  std::vector<Raw> raw;
  size_t i = 0, line = 1;
  auto count_lines = [&](size_t a, size_t b) {
    for (size_t x = a; x < b && x < src.size(); ++x) line += src[x] == '\n';
  };
  auto find_end = [&](size_t from, std::string_view close, bool strings) -> size_t {
    char q = 0;
    for (size_t x = from; x + close.size() <= src.size(); ++x) {
      const char c = src[x];
      if (q) {
        if (c == '\\') ++x;
        else if (c == q) q = 0;
        continue;
      }
      if (strings && (c == '\'' || c == '"')) {
        q = c;
        continue;
      }
      if (src.compare(x, close.size(), close) == 0) return x;
    }
    return std::string_view::npos;
  };
  std::string text;
  while (i < src.size()) {
    const size_t open = src.find('{', i);
    size_t at = open;
    while (at != std::string_view::npos && at + 1 < src.size() && src[at + 1] != '{' && src[at + 1] != '%' && src[at + 1] != '#') at = src.find('{', at + 1);
    if (at == std::string_view::npos || at + 1 >= src.size()) {
      text.append(src.substr(i));
      count_lines(i, src.size());
      break;
    }
    text.append(src.substr(i, at - i));
    count_lines(i, at);
    const char kind = src[at + 1];
    const std::string_view close = kind == '{' ? "}}" : kind == '%' ? "%}" : "#}";
    size_t b = at + 2;
    Raw r;
    r.line = line;
    if (b < src.size() && src[b] == '-') r.lminus = true, ++b;
    else if (b < src.size() && src[b] == '+') r.lplus = true, ++b;
    const size_t e = find_end(b, close, kind != '#');
    if (e == std::string_view::npos) fail("line " + std::to_string(line) + ": unclosed tag");
    size_t be = e;
    if (be > b && src[be - 1] == '-') r.rminus = true, --be;
    else if (be > b && src[be - 1] == '+' && kind != '{') r.rplus = true, --be;
    r.body = std::string(src.substr(b, be - b));
    r.k = kind == '{' ? Seg::expr : Seg::stmt;
    r.comment = kind == '#';
    count_lines(at, e + 2);
    i = e + 2;
    // raw blocks: the text up to {% endraw %} verbatim
    if (kind == '%') {
      std::string t = r.body;
      t.erase(0, t.find_first_not_of(" \t\r\n"));
      t.erase(t.find_last_not_of(" \t\r\n") + 1);
      if (t == "raw") {
        size_t x = i;
        while (true) {
          const size_t o = src.find("{%", x);
          if (o == std::string_view::npos) fail("line " + std::to_string(line) + ": raw block without endraw");
          size_t y = o + 2;
          if (y < src.size() && (src[y] == '-' || src[y] == '+')) ++y;
          while (y < src.size() && (src[y] == ' ' || src[y] == '\t')) ++y;
          if (src.compare(y, 6, "endraw") == 0) {
            const size_t ce = src.find("%}", y);
            raw.push_back({Seg::text, std::move(text)});
            text.clear();
            raw.push_back(r);  // the raw tag itself: whitespace rules, no statement
            raw.back().body.clear();
            raw.back().comment = true;
            Raw body{Seg::text, std::string(src.substr(i, o - i))};
            raw.push_back(body);
            Raw end{Seg::stmt, ""};
            end.comment = true;
            end.lminus = src[o + 2] == '-';
            end.rminus = ce > 0 && src[ce - 1] == '-';
            raw.push_back(end);
            count_lines(i, ce + 2);
            i = ce + 2;
            break;
          }
          x = o + 2;
        }
        continue;
      }
    }
    raw.push_back({Seg::text, std::move(text)});
    text.clear();
    raw.push_back(r);
  }
  raw.push_back({Seg::text, std::move(text)});
  // whitespace: '-' strips all whitespace on that side; lstrip_blocks strips a block / comment tag's line indentation;
  // trim_blocks removes the first newline after a block / comment tag
  for (size_t t = 0; t < raw.size(); ++t) {
    Raw &r = raw[t];
    if (r.k == Seg::text) continue;
    const bool block = r.k == Seg::stmt;
    if (t > 0 && raw[t - 1].k == Seg::text) {
      std::string &p = raw[t - 1].body;
      if (r.lminus) {
        p.erase(p.find_last_not_of(" \t\r\n\v\f") + 1);
      } else if (block && !r.lplus) {
        const size_t nl = p.rfind('\n');
        const size_t from = nl == std::string::npos ? 0 : nl + 1;
        // at a line start: the text holds a newline, or it is the template's first, or the tag before it ate one
        const bool line_start = nl != std::string::npos || t == 1 || (t >= 2 && raw[t - 2].k != Seg::text && raw[t - 2].ate_newline);
        if (line_start && p.find_first_not_of(" \t", from) == std::string::npos) p.erase(from);
      }
    }
    if (t + 1 < raw.size() && raw[t + 1].k == Seg::text) {
      std::string &n = raw[t + 1].body;
      if (r.rminus) {
        const size_t k = std::min(n.find_first_not_of(" \t\r\n\v\f"), n.size());
        r.ate_newline = k > 0 && n[k - 1] == '\n';
        n.erase(0, k);
      } else if (block && !r.rplus) {
        if (!n.empty() && n[0] == '\n') n.erase(0, 1), r.ate_newline = true;
        else if (n.size() >= 2 && n[0] == '\r' && n[1] == '\n') n.erase(0, 2), r.ate_newline = true;
      }
    }
  }
  std::vector<Seg> out;
  for (auto &r : raw) {
    if (r.comment) continue;
    if (r.k == Seg::text) {
      if (r.body.empty()) continue;
      if (!out.empty() && out.back().k == Seg::text) {
        out.back().body += r.body;
        continue;
      }
    }
    out.push_back({r.k, std::move(r.body), r.line});
  }
  return out;
}

// ---- expressions ------------------------------------------------------------------------------------------------

struct Tok {
  enum K : uint8_t { name, num, str, op, end } k = end;
  std::string s;
  Val v;  // num / str literal
  Tok() = default;
  Tok(K kind, std::string text) : k(kind), s(std::move(text)) {}
};

std::vector<Tok> tokenize(std::string_view s, size_t line) {
  std::vector<Tok> t;
  auto where = [&] { return "line " + std::to_string(line) + ": "; };
  size_t i = 0;
  while (i < s.size()) {
    const char c = s[i];
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      ++i;
      continue;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      size_t j = i;
      while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_')) ++j;
      t.push_back({Tok::name, std::string(s.substr(i, j - i))});
      i = j;
      continue;
    }
    if (std::isdigit(static_cast<unsigned char>(c))) {
      size_t j = i;
      bool fl = false;
      while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '_')) ++j;
      if (j + 1 < s.size() && s[j] == '.' && std::isdigit(static_cast<unsigned char>(s[j + 1]))) {
        fl = true, ++j;
        while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
      }
      if (j < s.size() && (s[j] == 'e' || s[j] == 'E')) {
        size_t k = j + 1;
        if (k < s.size() && (s[k] == '+' || s[k] == '-')) ++k;
        if (k < s.size() && std::isdigit(static_cast<unsigned char>(s[k]))) {
          fl = true, j = k;
          while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
        }
      }
      std::string lex(s.substr(i, j - i));
      lex.erase(std::remove(lex.begin(), lex.end(), '_'), lex.end());
      Tok k{Tok::num, lex};
      k.v = fl ? Val::Real(std::strtod(lex.c_str(), nullptr)) : Val::Int(std::strtoll(lex.c_str(), nullptr, 10));
      t.push_back(std::move(k));
      i = j;
      continue;
    }
    if (c == '\'' || c == '"') {
      std::string v;
      size_t j = i + 1;
      for (; j < s.size() && s[j] != c; ++j) {
        if (s[j] != '\\' || j + 1 >= s.size()) {
          v += s[j];
          continue;
        }
        const char e = s[++j];
        auto hex = [&](int n) {
          if (j + n >= s.size()) fail(where() + "truncated escape");
          const uint32_t x = static_cast<uint32_t>(std::stoul(std::string(s.substr(j + 1, n)), nullptr, 16));
          j += n;
          return x;
        };
        switch (e) {
          case 'n': v += '\n'; break;
          case 't': v += '\t'; break;
          case 'r': v += '\r'; break;
          case '0': v += '\0'; break;
          case 'a': v += '\a'; break;
          case 'b': v += '\b'; break;
          case 'f': v += '\f'; break;
          case 'v': v += '\v'; break;
          case '\\': v += '\\'; break;
          case '\'': v += '\''; break;
          case '"': v += '"'; break;
          case '\n': break;
          case 'x': encode_cp(hex(2), v); break;
          case 'u': encode_cp(hex(4), v); break;
          case 'U': encode_cp(hex(8), v); break;
          default: v += '\\', v += e;  // Python keeps an unknown escape as written
        }
      }
      if (j >= s.size()) fail(where() + "unterminated string");
      Tok k{Tok::str, ""};
      k.v = Val::Str(std::move(v));
      t.push_back(std::move(k));
      i = j + 1;
      continue;
    }
    static const char *ops2[] = {"**", "//", "==", "!=", "<=", ">="};
    bool two = false;
    for (const char *o : ops2)
      if (s.compare(i, 2, o) == 0) {
        t.push_back({Tok::op, o});
        i += 2;
        two = true;
        break;
      }
    if (two) continue;
    if (std::strchr("+-*/%~|.,:()[]{}<>=", c)) {
      t.push_back({Tok::op, std::string(1, c)});
      ++i;
      continue;
    }
    fail(where() + "unexpected character '" + std::string(1, c) + "'");
  }
  t.push_back({Tok::end, ""});
  return t;
}

struct Expr;
using Ex = std::unique_ptr<Expr>;
struct Expr {
  enum K : uint8_t { lit, name, attr, item, slice, call, filter, test, bin, neg, pos, not_, cond, list, tuple, dict, and_, or_ } k = lit;
  Val v;
  std::string s;
  std::vector<Ex> a;  // operands / target + positional arguments; slice: obj, start, stop, step (null = absent)
  std::vector<std::pair<std::string, Ex>> kw;
  bool negate = false;  // test: `is not`
};

Ex mk(Expr::K k, std::string s = {}) {
  auto e = std::make_unique<Expr>();
  e->k = k, e->s = std::move(s);
  return e;
}

class ExprParser {
 public:
  ExprParser(std::vector<Tok> t, size_t line) : t_(std::move(t)), line_(line) {}
  const Tok &cur() const { return t_[p_]; }
  bool at_end() const { return cur().k == Tok::end; }
  bool is_op(std::string_view o) const { return cur().k == Tok::op && cur().s == o; }
  bool is_name(std::string_view n) const { return cur().k == Tok::name && cur().s == n; }
  bool is_name_at(size_t off, std::string_view n) const { return p_ + off < t_.size() && t_[p_ + off].k == Tok::name && t_[p_ + off].s == n; }
  void next() {
    if (!at_end()) ++p_;
  }
  [[noreturn]] void error(const std::string &m) const { fail("line " + std::to_string(line_) + ": " + m + (at_end() ? " (at the end)" : " near '" + cur().s + "'")); }
  void expect_op(std::string_view o) {
    if (!is_op(o)) error("expected '" + std::string(o) + "'");
    next();
  }
  std::string expect_name() {
    if (cur().k != Tok::name) error("expected a name");
    std::string n = cur().s;
    next();
    return n;
  }
  void expect_end() {
    if (!at_end()) error("unexpected token");
  }

  Ex expression(bool with_cond = true) {
    Ex e = or_();
    if (with_cond && is_name("if")) {
      next();
      Ex c = or_();
      Ex el;
      if (is_name("else")) {
        next();
        el = expression();
      }
      Ex r = mk(Expr::cond);
      r->a.push_back(std::move(e));
      r->a.push_back(std::move(c));
      r->a.push_back(std::move(el));
      return r;
    }
    return e;
  }
  // a, b, c (no parentheses) -> tuple; one expression -> itself
  Ex tuple_expr(bool with_cond = true) {
    Ex first = expression(with_cond);
    if (!is_op(",")) return first;
    Ex t = mk(Expr::tuple);
    t->a.push_back(std::move(first));
    while (is_op(",")) {
      next();
      if (at_end() || is_op(")") || is_name("if")) break;
      t->a.push_back(expression(with_cond));
    }
    return t;
  }

 private:
  Ex bin(Expr::K k, std::string op, Ex a, Ex b) {
    Ex r = mk(k, std::move(op));
    r->a.push_back(std::move(a));
    r->a.push_back(std::move(b));
    return r;
  }
  Ex or_() {
    Ex e = and_();
    while (is_name("or")) {
      next();
      e = bin(Expr::or_, "or", std::move(e), and_());
    }
    return e;
  }
  Ex and_() {
    Ex e = not_();
    while (is_name("and")) {
      next();
      e = bin(Expr::and_, "and", std::move(e), not_());
    }
    return e;
  }
  Ex not_() {
    if (is_name("not")) {
      next();
      Ex r = mk(Expr::not_);
      r->a.push_back(not_());
      return r;
    }
    return compare();
  }
  Ex compare() {
    Ex e = math1();
    while (true) {
      std::string op;
      if (cur().k == Tok::op && (cur().s == "==" || cur().s == "!=" || cur().s == "<" || cur().s == ">" || cur().s == "<=" || cur().s == ">=")) {
        op = cur().s;
        next();
      } else if (is_name("in")) {
        op = "in";
        next();
      } else if (is_name("not") && is_name_at(1, "in")) {
        op = "notin";
        next(), next();
      } else {
        break;
      }
      e = bin(Expr::bin, op, std::move(e), math1());
    }
    return e;
  }
  Ex math1() {
    Ex e = concat();
    while (is_op("+") || is_op("-")) {
      const std::string op = cur().s;
      next();
      e = bin(Expr::bin, op, std::move(e), concat());
    }
    return e;
  }
  Ex concat() {
    Ex e = math2();
    while (is_op("~")) {
      next();
      e = bin(Expr::bin, "~", std::move(e), math2());
    }
    return e;
  }
  Ex math2() {
    Ex e = pow_();
    while (is_op("*") || is_op("/") || is_op("//") || is_op("%")) {
      const std::string op = cur().s;
      next();
      e = bin(Expr::bin, op, std::move(e), pow_());
    }
    return e;
  }
  Ex pow_() {
    Ex e = unary(true);
    while (is_op("**")) {
      next();
      e = bin(Expr::bin, "**", std::move(e), unary(true));
    }
    return e;
  }
  Ex unary(bool with_filter) {
    Ex e;
    if (is_op("-") || is_op("+")) {
      const bool minus = is_op("-");
      next();
      e = mk(minus ? Expr::neg : Expr::pos);
      e->a.push_back(unary(false));
    } else {
      e = postfix(primary());
    }
    if (with_filter) return filters(std::move(e));
    return e;
  }
  void call_args(Expr &e) {
    expect_op("(");
    while (!is_op(")")) {
      if (cur().k == Tok::name && p_ + 1 < t_.size() && t_[p_ + 1].k == Tok::op && t_[p_ + 1].s == "=") {
        std::string n = cur().s;
        next(), next();
        e.kw.emplace_back(std::move(n), expression());
      } else {
        if (!e.kw.empty()) error("positional argument after a keyword argument");
        e.a.push_back(expression());
      }
      if (!is_op(",")) break;
      next();
    }
    expect_op(")");
  }
  Ex primary() {
    const Tok &t = cur();
    if (t.k == Tok::name) {
      std::string n = t.s;
      next();
      Ex e = mk(Expr::lit);
      if (n == "true" || n == "True") e->v = Val::Bool(true);
      else if (n == "false" || n == "False") e->v = Val::Bool(false);
      else if (n == "none" || n == "None") e->v = Val::None();
      else e = mk(Expr::name, n);
      return e;
    }
    if (t.k == Tok::str) {
      std::string v = t.v.s;
      next();
      while (cur().k == Tok::str) v += cur().v.s, next();  // adjacent literals concatenate
      Ex e = mk(Expr::lit);
      e->v = Val::Str(std::move(v));
      return e;
    }
    if (t.k == Tok::num) {
      Ex e = mk(Expr::lit);
      e->v = t.v;
      next();
      return e;
    }
    if (is_op("(")) {
      next();
      if (is_op(")")) {
        next();
        return mk(Expr::tuple);
      }
      Ex e = expression();
      if (is_op(",")) {
        Ex tt = mk(Expr::tuple);
        tt->a.push_back(std::move(e));
        while (is_op(",")) {
          next();
          if (is_op(")")) break;
          tt->a.push_back(expression());
        }
        e = std::move(tt);
      }
      expect_op(")");
      return e;
    }
    if (is_op("[")) {
      next();
      Ex e = mk(Expr::list);
      while (!is_op("]")) {
        e->a.push_back(expression());
        if (!is_op(",")) break;
        next();
      }
      expect_op("]");
      return e;
    }
    if (is_op("{")) {
      next();
      Ex e = mk(Expr::dict);
      while (!is_op("}")) {
        e->a.push_back(expression());
        expect_op(":");
        e->a.push_back(expression());
        if (!is_op(",")) break;
        next();
      }
      expect_op("}");
      return e;
    }
    error("expected an expression");
  }
  Ex postfix(Ex e) {
    while (true) {
      if (is_op(".")) {
        next();
        if (cur().k == Tok::num && cur().v.k == Val::integer) {
          Ex it = mk(Expr::item);
          it->a.push_back(std::move(e));
          Ex idx = mk(Expr::lit);
          idx->v = cur().v;
          next();
          it->a.push_back(std::move(idx));
          e = std::move(it);
          continue;
        }
        Ex at = mk(Expr::attr, expect_name());
        at->a.push_back(std::move(e));
        e = std::move(at);
      } else if (is_op("[")) {
        next();
        Ex parts[3];
        bool slice = false;
        for (int k = 0; k < 3; ++k) {
          if (!is_op(":") && !is_op("]")) parts[k] = expression();
          if (k < 2 && is_op(":")) {
            slice = true;
            next();
          } else {
            break;
          }
        }
        expect_op("]");
        if (slice) {
          Ex sl = mk(Expr::slice);
          sl->a.push_back(std::move(e));
          for (auto &p : parts) sl->a.push_back(std::move(p));
          e = std::move(sl);
        } else {
          if (!parts[0]) error("empty subscript");
          Ex it = mk(Expr::item);
          it->a.push_back(std::move(e));
          it->a.push_back(std::move(parts[0]));
          e = std::move(it);
        }
      } else if (is_op("(")) {
        Ex c = mk(Expr::call);
        c->a.push_back(std::move(e));
        call_args(*c);
        e = std::move(c);
      } else {
        return e;
      }
    }
  }
  Ex filters(Ex e) {
    while (true) {
      if (is_op("|")) {
        next();
        std::string n = expect_name();
        while (is_op(".")) next(), n += "." + expect_name();
        Ex f = mk(Expr::filter, n);
        f->a.push_back(std::move(e));
        if (is_op("(")) call_args(*f);
        e = std::move(f);
      } else if (is_name("is")) {
        next();
        bool neg = false;
        if (is_name("not")) neg = true, next();
        std::string n = expect_name();
        while (is_op(".")) next(), n += "." + expect_name();
        Ex t = mk(Expr::test, n);
        t->negate = neg;
        t->a.push_back(std::move(e));
        if (is_op("(")) {
          call_args(*t);
        } else if ((cur().k == Tok::name || cur().k == Tok::str || cur().k == Tok::num || is_op("[") || is_op("{")) &&
                   !(is_name("else") || is_name("or") || is_name("and") || is_name("if") || is_name("not") || is_name("in") || is_name("is"))) {
          t->a.push_back(postfix(primary()));
        }
        e = std::move(t);
      } else if (is_op("(")) {
        Ex c = mk(Expr::call);
        c->a.push_back(std::move(e));
        call_args(*c);
        e = std::move(c);
      } else {
        return e;
      }
    }
  }

  std::vector<Tok> t_;
  size_t p_ = 0;
  size_t line_ = 1;
};

// ---- statements -------------------------------------------------------------------------------------------------

struct Node;
using NodeP = std::unique_ptr<Node>;
using Body = std::vector<NodeP>;
struct Node {
  enum K : uint8_t { text, out, if_, for_, set, set_attr, set_block, macro, brk, cont, filter_block, scope } k = text;
  std::string s;  // text; set_attr: the attribute; macro: its name
  Ex e;  // out / set value / for iterable / filter block's filter (target null)
  Ex filt;  // for... if <filt>
  std::vector<std::string> names;  // targets / macro parameters
  std::vector<Ex> defaults;  // macro parameter defaults (null = none)
  std::vector<Ex> conds;  // if / elif conditions
  std::vector<Body> bodies;  // if: one per condition (+ else); for / set_block / macro / scope: [0] body, for [1] else
};

class Parser {
 public:
  explicit Parser(std::vector<Seg> segs) : s_(std::move(segs)) {}
  Body parse_all() {
    std::string end;
    Body b = body({}, end);
    if (!end.empty()) fail("unexpected '" + end + "'");
    return b;
  }
  bool content_loop = false;  // a for over `content` / `x.content` / x['content']

 private:
  // parse until one of `ends` (a statement keyword); `end` gets it ("" at the end of the source)
  Body body(const std::vector<std::string> &ends, std::string &end) {
    Body b;
    while (i_ < s_.size()) {
      const Seg &g = s_[i_];
      if (g.k == Seg::text) {
        auto n = std::make_unique<Node>();
        n->k = Node::text, n->s = g.body;
        b.push_back(std::move(n));
        ++i_;
        continue;
      }
      ExprParser p(tokenize(g.body, g.line), g.line);
      if (g.k == Seg::expr) {
        auto n = std::make_unique<Node>();
        n->k = Node::out;
        n->e = p.tuple_expr();
        p.expect_end();
        b.push_back(std::move(n));
        ++i_;
        continue;
      }
      const std::string kw = p.expect_name();
      if (std::find(ends.begin(), ends.end(), kw) != ends.end()) {
        end = kw;
        cur_ = std::make_unique<ExprParser>(std::move(p));
        ++i_;
        return b;
      }
      ++i_;
      b.push_back(statement(kw, p, g.line));
    }
    if (!ends.empty()) fail("missing '" + ends.back() + "'");
    end.clear();
    return b;
  }
  static void mark_content(const Expr *it, bool &hit) {
    while (it && it->k == Expr::filter) it = it->a[0].get();
    if (!it) return;
    if ((it->k == Expr::name && it->s == "content") || (it->k == Expr::attr && it->s == "content") ||
        (it->k == Expr::item && it->a[1]->k == Expr::lit && it->a[1]->v.k == Val::str && it->a[1]->v.s == "content"))
      hit = true;
  }
  NodeP statement(const std::string &kw, ExprParser &p, size_t line) {
    auto n = std::make_unique<Node>();
    std::string end;
    auto where = [&] { return "line " + std::to_string(line) + ": "; };
    if (kw == "if") {
      n->k = Node::if_;
      n->conds.push_back(p.expression());
      p.expect_end();
      while (true) {
        n->bodies.push_back(body({"elif", "else", "endif"}, end));
        if (end == "elif") {
          n->conds.push_back(cur_->expression());
          cur_->expect_end();
          continue;
        }
        cur_->expect_end();
        if (end == "else") {
          n->bodies.push_back(body({"endif"}, end));
          cur_->expect_end();
        }
        break;
      }
      return n;
    }
    if (kw == "for") {
      n->k = Node::for_;
      n->names.push_back(p.expect_name());
      while (p.is_op(",")) p.next(), n->names.push_back(p.expect_name());
      if (!p.is_name("in")) p.error("expected 'in'");
      p.next();
      n->e = p.tuple_expr(false);
      mark_content(n->e.get(), content_loop);
      if (p.is_name("if")) {
        p.next();
        n->filt = p.expression();
      }
      if (p.is_name("recursive")) p.error("recursive loops are not supported");
      p.expect_end();
      n->bodies.push_back(body({"else", "endfor"}, end));
      cur_->expect_end();
      if (end == "else") {
        n->bodies.push_back(body({"endfor"}, end));
        cur_->expect_end();
      }
      return n;
    }
    if (kw == "set") {
      std::string first = p.expect_name();
      if (p.is_op(".")) {
        p.next();
        n->k = Node::set_attr;
        n->names.push_back(first);
        n->s = p.expect_name();
        p.expect_op("=");
        n->e = p.tuple_expr();
        p.expect_end();
        return n;
      }
      n->names.push_back(first);
      while (p.is_op(",")) p.next(), n->names.push_back(p.expect_name());
      if (p.is_op("=")) {
        p.next();
        n->k = Node::set;
        n->e = p.tuple_expr();
        p.expect_end();
        return n;
      }
      if (n->names.size() != 1) p.error("a block set takes one name");
      n->k = Node::set_block;
      if (p.is_op("|")) {  // {% set x | filter %}: the filter over the block's output
        Ex f = mk(Expr::lit);
        f->v = Val::Str("");
        n->filt = std::move(f);
        ExprParser &q = p;
        (void)q;
        p.error("filtered block set is not supported");
      }
      p.expect_end();
      n->bodies.push_back(body({"endset"}, end));
      cur_->expect_end();
      return n;
    }
    if (kw == "macro") {
      n->k = Node::macro;
      n->s = p.expect_name();
      p.expect_op("(");
      while (!p.is_op(")")) {
        n->names.push_back(p.expect_name());
        if (p.is_op("=")) {
          p.next();
          n->defaults.push_back(p.expression());
        } else {
          n->defaults.push_back(nullptr);
        }
        if (!p.is_op(",")) break;
        p.next();
      }
      p.expect_op(")");
      p.expect_end();
      n->bodies.push_back(body({"endmacro"}, end));
      return n;  // `endmacro name` allowed: the rest is ignored
    }
    if (kw == "break" || kw == "continue") {
      p.expect_end();
      n->k = kw == "break" ? Node::brk : Node::cont;
      return n;
    }
    if (kw == "filter") {
      n->k = Node::filter_block;
      std::string fname = p.expect_name();
      Ex f = mk(Expr::filter, fname);
      f->a.push_back(nullptr);
      if (p.is_op("(")) {
        // reuse the expression parser's argument syntax: parse "x(args)" as a call
        Ex c = p.expression();
        (void)c;
        p.error("filter block arguments are not supported");
      }
      p.expect_end();
      n->e = std::move(f);
      n->bodies.push_back(body({"endfilter"}, end));
      cur_->expect_end();
      return n;
    }
    if (kw == "generation") {  // the assistant-mask extension: transparent here
      p.expect_end();
      n->k = Node::scope;
      n->bodies.push_back(body({"endgeneration"}, end));
      cur_->expect_end();
      return n;
    }
    fail(where() + "unsupported statement '" + kw + "'");
  }

  std::vector<Seg> s_;
  size_t i_ = 0;
  std::unique_ptr<ExprParser> cur_;  // the closing tag's tokens (elif condition, endfor...)
};

// ---- evaluation -------------------------------------------------------------------------------------------------

struct Frame {
  std::unordered_map<std::string, Val> vars;
  Frame *parent = nullptr;
  const Val *lookup(const std::string &n) const {
    for (const Frame *f = this; f; f = f->parent) {
      auto it = f->vars.find(n);
      if (it != f->vars.end()) return &it->second;
    }
    return nullptr;
  }
};

enum class Flow : uint8_t { normal, brk, cont };

struct Eval {
  Frame *root = nullptr;
  int depth = 0;

  Val eval(const Expr &e, Frame &f);
  Flow exec(const Body &b, Frame &f, std::string &o);

  Val call(const Val &fn, Args &args, Kw &kw) {
    if (fn.k == Val::undef) undefined_error(fn);
    if (fn.k != Val::func) fail(std::string("'") + type_name(fn) + "' object is not callable");
    if (++depth > 200) fail("recursion too deep");
    Val r = (*fn.f)(args, kw);
    --depth;
    return r;
  }

  Val method(const Val &obj, const std::string &name);
  Val get_attr(const Val &obj, const std::string &name) {
    if (obj.k == Val::undef) undefined_error(obj);
    if (obj.k == Val::dict) {
      if (!obj.ns) {
        Val m = method(obj, name);
        if (m.k == Val::func) return m;
      }
      if (const Val *v = obj.find(name)) return *v;
      return Val::Undef(std::string(obj.ns ? "namespace" : "dict object") + "' has no attribute '" + name);
    }
    Val m = method(obj, name);
    if (m.k == Val::func) return m;
    return Val::Undef(std::string(type_name(obj)) + " object' has no attribute '" + name);
  }
  Val get_item(const Val &obj, const Val &key) {
    if (obj.k == Val::undef) undefined_error(obj);
    if (obj.k == Val::dict) {
      if (key.k == Val::str)
        if (const Val *v = obj.find(key.s)) return *v;
      if (key.k == Val::str) {  // jinja's getitem falls back to the attribute
        Val m = method(obj, key.s);
        if (m.k == Val::func) return m;
      }
      return Val::Undef("dict object' has no attribute '" + to_str(key));
    }
    if (obj.k == Val::list || obj.k == Val::str) {
      if (key.k == Val::str) return get_attr(obj, key.s);
      if (key.k != Val::integer && key.k != Val::boolean) return Val::Undef("invalid index");
      int64_t i = key.as_int();
      if (obj.k == Val::list) {
        const int64_t n = static_cast<int64_t>(obj.l->size());
        if (i < 0) i += n;
        if (i < 0 || i >= n) return Val::Undef("index out of range");
        return (*obj.l)[static_cast<size_t>(i)];
      }
      const auto cps = code_points(obj.s);
      const int64_t n = static_cast<int64_t>(cps.size());
      if (i < 0) i += n;
      if (i < 0 || i >= n) return Val::Undef("index out of range");
      return Val::Str(std::string(cps[static_cast<size_t>(i)]));
    }
    return Val::Undef(std::string(type_name(obj)) + " object' has no item");
  }
  Val slice(const Val &obj, const Val *a, const Val *b, const Val *c) {
    if (obj.k == Val::undef) undefined_error(obj);
    std::vector<Val> items;
    std::vector<std::string_view> cps;
    int64_t n;
    if (obj.k == Val::list) n = static_cast<int64_t>(obj.l->size());
    else if (obj.k == Val::str) cps = code_points(obj.s), n = static_cast<int64_t>(cps.size());
    else fail(std::string("'") + type_name(obj) + "' object is not subscriptable");
    auto num = [](const Val *v, bool &set) -> int64_t {
      set = v && v->k != Val::none && v->k != Val::undef;
      if (set && v->k != Val::integer && v->k != Val::boolean) fail("slice indices must be integers or None");
      return set ? v->as_int() : 0;
    };
    bool sa, sb, sc;
    int64_t st = num(c, sc);
    if (!sc) st = 1;
    if (st == 0) fail("slice step cannot be zero");
    int64_t lo = num(a, sa), hi = num(b, sb);
    auto clamp = [&](int64_t v, bool set, int64_t dflt) {
      if (!set) return dflt;
      if (v < 0) v += n;
      if (st > 0) return std::clamp<int64_t>(v, 0, n);
      return std::clamp<int64_t>(v, -1, n - 1);
    };
    lo = clamp(lo, sa, st > 0 ? 0 : n - 1);
    hi = clamp(hi, sb, st > 0 ? n : -1);
    std::string so;
    for (int64_t i = lo; st > 0 ? i < hi : i > hi; i += st) {
      if (obj.k == Val::list) items.push_back((*obj.l)[static_cast<size_t>(i)]);
      else so += cps[static_cast<size_t>(i)];
    }
    if (obj.k == Val::str) return Val::Str(so);
    return Val::List(std::move(items), obj.tuple);
  }

  Val filter(const std::string &name, Val v, Args &args, Kw &kw);
  bool test(const std::string &name, const Val &v, Args &args);

  Val binop(const std::string &op, const Val &a, const Val &b) {
    if (op == "~") return Val::Str(to_str(a) + to_str(b));
    if (op == "==") return Val::Bool(py_eq(a, b));
    if (op == "!=") return Val::Bool(!py_eq(a, b));
    if (op == "in" || op == "notin") {
      bool r = false;
      if (b.k == Val::str) {
        if (a.k != Val::str) fail("'in <string>' requires string as left operand");
        r = b.s.find(a.s) != std::string::npos;
      } else if (b.k == Val::dict) {
        r = a.k == Val::str && b.find(a.s);
      } else if (b.k == Val::list) {
        for (const auto &x : *b.l) r = r || py_eq(a, x);
      } else if (b.k != Val::undef) {
        fail(std::string("argument of type '") + type_name(b) + "' is not iterable");
      }
      return Val::Bool(op == "in" ? r : !r);
    }
    if (op == "<" || op == ">" || op == "<=" || op == ">=") {
      if (a.k == Val::undef) undefined_error(a);
      if (b.k == Val::undef) undefined_error(b);
      const int c = py_cmp(a, b);
      return Val::Bool(op == "<" ? c < 0 : op == ">" ? c > 0 : op == "<=" ? c <= 0 : c >= 0);
    }
    if (a.k == Val::undef) undefined_error(a);
    if (b.k == Val::undef) undefined_error(b);
    if (op == "+") {
      if (a.k == Val::str && b.k == Val::str) return Val::Str(a.s + b.s);
      if (a.k == Val::list && b.k == Val::list) {
        std::vector<Val> r = *a.l;
        r.insert(r.end(), b.l->begin(), b.l->end());
        return Val::List(std::move(r), a.tuple);
      }
    }
    if (op == "*") {
      const Val *seq = a.k == Val::str || a.k == Val::list ? &a : b.k == Val::str || b.k == Val::list ? &b : nullptr;
      const Val *cnt = seq == &a ? &b : &a;
      if (seq && (cnt->k == Val::integer || cnt->k == Val::boolean)) {
        const int64_t c = cnt->as_int();
        if (seq->k == Val::str) {
          std::string r;
          for (int64_t i = 0; i < c; ++i) r += seq->s;
          return Val::Str(r);
        }
        std::vector<Val> r;
        for (int64_t i = 0; i < c; ++i) r.insert(r.end(), seq->l->begin(), seq->l->end());
        return Val::List(std::move(r), seq->tuple);
      }
    }
    if (!a.numeric() || !b.numeric())
      fail(std::string("unsupported operand type(s) for ") + op + ": '" + type_name(a) + "' and '" + type_name(b) + "'");
    const bool ints = a.k != Val::real && b.k != Val::real;
    if (op == "/") {
      if (b.as_double() == 0) fail("division by zero");
      return Val::Real(a.as_double() / b.as_double());
    }
    if (op == "**") {
      if (ints && b.as_int() >= 0) {
        int64_t r = 1;
        for (int64_t i = 0; i < b.as_int(); ++i) r *= a.as_int();
        return Val::Int(r);
      }
      return Val::Real(std::pow(a.as_double(), b.as_double()));
    }
    if (ints) {
      const int64_t x = a.as_int(), y = b.as_int();
      if (op == "+") return Val::Int(x + y);
      if (op == "-") return Val::Int(x - y);
      if (op == "*") return Val::Int(x * y);
      if (y == 0) fail("integer division or modulo by zero");
      int64_t q = x / y, r = x % y;
      if (r != 0 && ((r < 0) != (y < 0))) --q, r += y;
      if (op == "//") return Val::Int(q);
      if (op == "%") return Val::Int(r);
    } else {
      const double x = a.as_double(), y = b.as_double();
      if (op == "+") return Val::Real(x + y);
      if (op == "-") return Val::Real(x - y);
      if (op == "*") return Val::Real(x * y);
      if (y == 0) fail("float division by zero");
      if (op == "//") return Val::Real(std::floor(x / y));
      if (op == "%") {
        double r = std::fmod(x, y);
        if (r != 0 && ((r < 0) != (y < 0))) r += y;
        return Val::Real(r);
      }
    }
    fail("unsupported operator " + op);
  }

  Val call_macro(const Node &m, Args &args, Kw &kw) {
    Frame fr;
    fr.parent = root;
    if (args.size() > m.names.size()) fail("macro '" + m.s + "' takes not more than " + std::to_string(m.names.size()) + " argument(s)");
    for (size_t i = 0; i < m.names.size(); ++i) {
      if (i < args.size()) {
        fr.vars[m.names[i]] = args[i];
        continue;
      }
      bool found = false;
      for (auto &kv : kw)
        if (kv.first == m.names[i]) fr.vars[m.names[i]] = kv.second, found = true;
      if (!found) fr.vars[m.names[i]] = m.defaults[i] ? eval(*m.defaults[i], fr) : Val::Undef(m.names[i]);
    }
    for (auto &kv : kw)
      if (std::find(m.names.begin(), m.names.end(), kv.first) == m.names.end()) fail("macro '" + m.s + "' takes no keyword argument '" + kv.first + "'");
    std::string o;
    exec(m.bodies[0], fr, o);
    return Val::Str(o);
  }

  void assign(Frame &f, const std::vector<std::string> &names, const Val &v) {
    if (names.size() == 1) {
      f.vars[names[0]] = v;
      return;
    }
    const auto items = iterate(v);
    if (items.size() != names.size()) fail("cannot unpack " + std::to_string(items.size()) + " values into " + std::to_string(names.size()));
    for (size_t i = 0; i < names.size(); ++i) f.vars[names[i]] = items[i];
  }
};

Val Eval::method(const Val &obj, const std::string &name) {
  auto arg = [](Args &a, Kw &kw, size_t i, const char *n) -> const Val * {
    if (i < a.size()) return &a[i];
    for (auto &kv : kw)
      if (kv.first == n) return &kv.second;
    return nullptr;
  };
  if (obj.k == Val::str) {
    const std::string s = obj.s;
    if (name == "startswith" || name == "endswith") {
      const bool sw = name == "startswith";
      return Val::Func([s, sw](Args &a, Kw &) {
        if (a.empty()) fail("startswith / endswith take a prefix");
        auto one = [&](const Val &p) {
          if (p.k != Val::str) fail("startswith first arg must be str or a tuple of str");
          return sw ? s.compare(0, p.s.size(), p.s) == 0 && s.size() >= p.s.size()
                    : s.size() >= p.s.size() && s.compare(s.size() - p.s.size(), p.s.size(), p.s) == 0;
        };
        if (a[0].k == Val::list) {
          for (const auto &p : *a[0].l)
            if (one(p)) return Val::Bool(true);
          return Val::Bool(false);
        }
        return Val::Bool(one(a[0]));
      });
    }
    if (name == "strip" || name == "lstrip" || name == "rstrip") {
      const bool l = name != "rstrip", r = name != "lstrip";
      return Val::Func([s, l, r, arg](Args &a, Kw &kw) { return Val::Str(py_strip(s, arg(a, kw, 0, "chars"), l, r)); });
    }
    if (name == "split" || name == "rsplit") {
      const bool rs = name == "rsplit";
      return Val::Func([s, arg, rs](Args &a, Kw &kw) {
        const Val *ms = arg(a, kw, 1, "maxsplit");
        const int64_t max = ms ? ms->as_int() : -1;
        if (rs && max >= 0) {  // from the right: split all, re-join the head
          auto all = py_split(s, arg(a, kw, 0, "sep"), -1);
          const Val *sep = arg(a, kw, 0, "sep");
          if (!sep || sep->k != Val::str || static_cast<int64_t>(all.size()) <= max + 1) return Val::List(std::move(all));
          std::string head;
          const size_t cut = all.size() - static_cast<size_t>(max);
          for (size_t i = 0; i < cut; ++i) head += (i ? sep->s : "") + all[i].s;
          std::vector<Val> r{Val::Str(head)};
          r.insert(r.end(), all.begin() + static_cast<long>(cut), all.end());
          return Val::List(std::move(r));
        }
        return Val::List(py_split(s, arg(a, kw, 0, "sep"), max));
      });
    }
    if (name == "replace")
      return Val::Func([s, arg](Args &a, Kw &kw) {
        const Val *o = arg(a, kw, 0, "old"), *n = arg(a, kw, 1, "new"), *c = arg(a, kw, 2, "count");
        if (!o || !n || o->k != Val::str || n->k != Val::str) fail("replace() arguments must be str");
        return Val::Str(py_replace(s, o->s, n->s, c ? c->as_int() : -1));
      });
    if (name == "upper" || name == "lower") {
      const bool up = name == "upper";
      return Val::Func([s, up](Args &, Kw &) { return Val::Str(ascii_case(s, up)); });
    }
    if (name == "title" || name == "capitalize") {
      const bool title = name == "title";
      return Val::Func([s, title](Args &, Kw &) {
        std::string r = ascii_case(s, false);
        bool start = true;
        for (size_t i = 0; i < r.size(); ++i) {
          const bool alpha = std::isalpha(static_cast<unsigned char>(r[i]));
          if (alpha && start && (title || i == 0)) r[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(r[i])));
          start = title ? !alpha : false;
        }
        return Val::Str(r);
      });
    }
    if (name == "find" || name == "rfind" || name == "count" || name == "index") {
      const std::string nm = name;
      return Val::Func([s, nm](Args &a, Kw &) {
        if (a.empty() || a[0].k != Val::str) fail(nm + "() takes a str");
        const std::string &x = a[0].s;
        auto cp_index = [&](size_t byte) { return static_cast<int64_t>(code_points(std::string_view(s).substr(0, byte)).size()); };
        if (nm == "count") {
          if (x.empty()) return Val::Int(static_cast<int64_t>(code_points(s).size()) + 1);
          int64_t c = 0;
          for (size_t p = s.find(x); p != std::string::npos; p = s.find(x, p + x.size())) ++c;
          return Val::Int(c);
        }
        const size_t p = nm == "rfind" ? s.rfind(x) : s.find(x);
        if (p == std::string::npos) {
          if (nm == "index") fail("substring not found");
          return Val::Int(-1);
        }
        return Val::Int(cp_index(p));
      });
    }
    if (name == "join")
      return Val::Func([s](Args &a, Kw &) {
        if (a.empty()) fail("join() takes an iterable");
        std::string r;
        bool first = true;
        for (const auto &x : iterate(a[0])) {
          if (x.k != Val::str) fail(std::string("sequence item: expected str instance, ") + type_name(x) + " found");
          if (!first) r += s;
          first = false;
          r += x.s;
        }
        return Val::Str(r);
      });
    if (name == "splitlines")
      return Val::Func([s](Args &, Kw &) {
        std::vector<Val> r;
        size_t p = 0;
        while (p < s.size()) {
          size_t e = s.find_first_of("\r\n", p);
          if (e == std::string::npos) {
            r.push_back(Val::Str(s.substr(p)));
            break;
          }
          r.push_back(Val::Str(s.substr(p, e - p)));
          p = e + (s[e] == '\r' && e + 1 < s.size() && s[e + 1] == '\n' ? 2 : 1);
        }
        return Val::List(std::move(r));
      });
    if (name == "isdigit" || name == "isspace" || name == "isalpha" || name == "isalnum") {
      const std::string nm = name;
      return Val::Func([s, nm](Args &, Kw &) {
        if (s.empty()) return Val::Bool(false);
        for (const auto cp : code_points(s)) {
          const uint32_t c = decode_cp(cp);
          const bool ok = nm == "isspace" ? py_space(c)
                          : nm == "isdigit" ? (c >= '0' && c <= '9')
                          : nm == "isalpha" ? (c < 0x80 ? std::isalpha(static_cast<int>(c)) != 0 : true)
                                            : (c < 0x80 ? std::isalnum(static_cast<int>(c)) != 0 : true);
          if (!ok) return Val::Bool(false);
        }
        return Val::Bool(true);
      });
    }
  }
  if (obj.k == Val::dict && !obj.ns) {
    const Val d = obj;
    if (name == "get")
      return Val::Func([d](Args &a, Kw &) {
        if (a.empty()) fail("get expected at least 1 argument");
        if (a[0].k == Val::str)
          if (const Val *v = d.find(a[0].s)) return *v;
        return a.size() > 1 ? a[1] : Val::None();
      });
    if (name == "items" || name == "keys" || name == "values") {
      const int which = name == "items" ? 0 : name == "keys" ? 1 : 2;
      return Val::Func([d, which](Args &, Kw &) {
        std::vector<Val> r;
        for (const auto &kv : *d.m) {
          if (which == 0) r.push_back(Val::List({Val::Str(kv.first), kv.second}, true));
          else r.push_back(which == 1 ? Val::Str(kv.first) : kv.second);
        }
        return Val::List(std::move(r));
      });
    }
  }
  return Val::Undef();
}

bool Eval::test(const std::string &name, const Val &v, Args &a) {
  auto need = [&](size_t n) {
    if (a.size() < n) fail("test '" + name + "' needs an argument");
  };
  if (name == "defined") return v.k != Val::undef;
  if (name == "undefined") return v.k == Val::undef;
  if (name == "none") return v.k == Val::none;
  if (name == "boolean") return v.k == Val::boolean;
  if (name == "true") return v.k == Val::boolean && v.b;
  if (name == "false") return v.k == Val::boolean && !v.b;
  if (name == "string") return v.k == Val::str;
  if (name == "number") return v.k == Val::integer || v.k == Val::real || v.k == Val::boolean;
  if (name == "integer") return v.k == Val::integer;
  if (name == "float") return v.k == Val::real;
  if (name == "mapping") return v.k == Val::dict && !v.ns;
  if (name == "iterable") return v.k == Val::str || v.k == Val::list || (v.k == Val::dict && !v.ns) || v.k == Val::undef;
  if (name == "sequence") return v.k == Val::str || v.k == Val::list || (v.k == Val::dict && !v.ns);
  if (name == "callable") return v.k == Val::func || v.k == Val::undef;  // Undefined is callable (calling it raises)
  if (name == "odd" || name == "even") {
    if (v.k != Val::integer && v.k != Val::boolean) fail("odd / even test on a non-integer");
    return (v.as_int() % 2 != 0) == (name == "odd");
  }
  if (name == "divisibleby") {
    need(1);
    return a[0].as_int() != 0 && v.as_int() % a[0].as_int() == 0;
  }
  if (name == "lower" || name == "upper") return v.k == Val::str && ascii_case(v.s, name == "upper") == v.s;
  if (name == "equalto" || name == "eq" || name == "==") return need(1), py_eq(v, a[0]);
  if (name == "ne" || name == "!=") return need(1), !py_eq(v, a[0]);
  if (name == "sameas") {
    need(1);
    if (v.k != a[0].k) return false;
    if (v.k == Val::none || v.k == Val::undef) return true;
    if (v.k == Val::boolean) return v.b == a[0].b;
    if (v.k == Val::list) return v.l == a[0].l;
    if (v.k == Val::dict) return v.m == a[0].m;
    return py_eq(v, a[0]);
  }
  if (name == "lt" || name == "<" || name == "lessthan") return need(1), py_cmp(v, a[0]) < 0;
  if (name == "le" || name == "<=") return need(1), py_cmp(v, a[0]) <= 0;
  if (name == "gt" || name == ">" || name == "greaterthan") return need(1), py_cmp(v, a[0]) > 0;
  if (name == "ge" || name == ">=") return need(1), py_cmp(v, a[0]) >= 0;
  if (name == "in") return need(1), truthy(binop("in", v, a[0]));
  if (name == "escaped") return false;
  fail("no test named '" + name + "'");
}

Val Eval::filter(const std::string &name, Val v, Args &a, Kw &kw) {
  auto arg = [&](size_t i, const char *n) -> const Val * {
    if (i < a.size()) return &a[i];
    for (auto &x : kw)
      if (x.first == n) return &x.second;
    return nullptr;
  };
  auto flag = [&](size_t i, const char *n, bool d) {
    const Val *x = arg(i, n);
    return x ? truthy(*x) : d;
  };
  if (name == "safe") return v;
  if (name == "string") return v.k == Val::str ? v : Val::Str(to_str(v));
  if (name == "trim") return Val::Str(py_strip(to_str(v), arg(0, "chars"), true, true));
  if (name == "default" || name == "d") {
    const bool boolean = flag(1, "boolean", false);
    if (v.k == Val::undef || (boolean && !truthy(v))) {
      const Val *d = arg(0, "default_value");
      return d ? *d : Val::Str("");
    }
    return v;
  }
  if (name == "length" || name == "count") return Val::Int(py_len(v));
  if (name == "items") {
    if (v.k == Val::undef) return Val::List();
    if (v.k != Val::dict) fail("can only get items from a mapping");
    std::vector<Val> r;
    for (const auto &x : *v.m) r.push_back(Val::List({Val::Str(x.first), x.second}, true));
    return Val::List(std::move(r));
  }
  if (name == "dictsort") {
    if (v.k != Val::dict) fail("dictsort on a non-mapping");
    std::vector<Val> r;
    for (const auto &x : *v.m) r.push_back(Val::List({Val::Str(x.first), x.second}, true));
    const bool by_value = arg(1, "by") && arg(1, "by")->k == Val::str && arg(1, "by")->s == "value";
    std::stable_sort(r.begin(), r.end(), [&](const Val &x, const Val &y) { return py_cmp((*x.l)[by_value], (*y.l)[by_value]) < 0; });
    if (flag(2, "reverse", false)) std::reverse(r.begin(), r.end());
    return Val::List(std::move(r));
  }
  if (name == "replace") {
    const Val *o = arg(0, "old"), *n = arg(1, "new"), *c = arg(2, "count");
    if (!o || !n) fail("replace needs old and new");
    return Val::Str(py_replace(to_str(v), to_str(*o), to_str(*n), c && c->k != Val::none ? c->as_int() : -1));
  }
  if (name == "tojson") {
    DumpOpt op;
    op.ensure_ascii = flag(0, "ensure_ascii", false);
    if (const Val *ind = arg(1, "indent"); ind && ind->k != Val::none && ind->k != Val::undef) {
      op.indent = true, op.item_sep = ",";
      if (ind->k == Val::str) op.indent_str = ind->s;
      else op.indent_str.assign(static_cast<size_t>(std::max<int64_t>(0, ind->as_int())), ' ');
    }
    if (const Val *sep = arg(2, "separators"); sep && sep->k == Val::list && sep->l->size() == 2)
      op.item_sep = to_str((*sep->l)[0]), op.key_sep = to_str((*sep->l)[1]);
    op.sort_keys = flag(3, "sort_keys", false);
    std::string o;
    dump(v, op, 0, o);
    return Val::Str(o);
  }
  if (name == "upper" || name == "lower") return Val::Str(ascii_case(to_str(v), name == "upper"));
  if (name == "capitalize" || name == "title") {
    Args none;
    Kw nk;
    return (*method(Val::Str(to_str(v)), name).f)(none, nk);
  }
  if (name == "first" || name == "last") {
    const auto it = iterate(v);
    if (it.empty()) return Val::Undef("No first item, sequence was empty");
    return name == "first" ? it.front() : it.back();
  }
  if (name == "list") return Val::List(iterate(v));
  if (name == "reverse") {
    if (v.k == Val::str) {
      auto cps = code_points(v.s);
      std::string r;
      for (size_t i = cps.size(); i-- > 0;) r += cps[i];
      return Val::Str(r);
    }
    auto it = iterate(v);
    std::reverse(it.begin(), it.end());
    return Val::List(std::move(it));
  }
  if (name == "join") {
    const Val *sep = arg(0, "d");
    const Val *attr = arg(1, "attribute");
    std::string r;
    bool first = true;
    for (const auto &x : iterate(v)) {
      if (!first && sep) r += to_str(*sep);
      first = false;
      r += to_str(attr ? get_attr(x, to_str(*attr)) : x);
    }
    return Val::Str(r);
  }
  if (name == "int") {
    if (v.k == Val::integer) return v;
    if (v.k == Val::real) return Val::Int(static_cast<int64_t>(v.d));
    if (v.k == Val::boolean) return Val::Int(v.b);
    if (v.k == Val::str) {
      char *e = nullptr;
      const std::string t = py_strip(v.s, nullptr, true, true);
      const long long x = std::strtoll(t.c_str(), &e, 10);
      if (!t.empty() && *e == 0) return Val::Int(x);
      const double d = std::strtod(t.c_str(), &e);
      if (!t.empty() && *e == 0) return Val::Int(static_cast<int64_t>(d));
    }
    const Val *d = arg(0, "default");
    return d ? *d : Val::Int(0);
  }
  if (name == "float") {
    if (v.numeric()) return Val::Real(v.as_double());
    if (v.k == Val::str) {
      char *e = nullptr;
      const std::string t = py_strip(v.s, nullptr, true, true);
      const double d = std::strtod(t.c_str(), &e);
      if (!t.empty() && *e == 0) return Val::Real(d);
    }
    const Val *d = arg(0, "default");
    return d ? *d : Val::Real(0);
  }
  if (name == "abs") {
    if (v.k == Val::real) return Val::Real(std::fabs(v.d));
    if (v.numeric()) return Val::Int(std::llabs(v.as_int()));
    fail("bad operand type for abs()");
  }
  if (name == "round") {
    const Val *p = arg(0, "precision");
    const int prec = p ? static_cast<int>(p->as_int()) : 0;
    const std::string meth = arg(1, "method") ? to_str(*arg(1, "method")) : "common";
    const double m = std::pow(10.0, prec), x = v.as_double() * m;
    const double r = meth == "floor" ? std::floor(x) : meth == "ceil" ? std::ceil(x) : std::nearbyint(x);  // Python's round: half to even
    return Val::Real(r / m);
  }
  if (name == "sum") {
    const Val *attr = arg(0, "attribute");
    Val acc = arg(1, "start") ? *arg(1, "start") : Val::Int(0);
    for (const auto &x : iterate(v)) acc = binop("+", acc, attr ? get_attr(x, to_str(*attr)) : x);
    return acc;
  }
  if (name == "min" || name == "max") {
    const auto it = iterate(v);
    if (it.empty()) return Val::Undef("empty sequence");
    Val best = it[0];
    for (const auto &x : it)
      if ((name == "min" ? py_cmp(x, best) < 0 : py_cmp(x, best) > 0)) best = x;
    return best;
  }
  if (name == "unique") {
    std::vector<Val> r;
    for (const auto &x : iterate(v)) {
      bool seen = false;
      for (const auto &y : r) seen = seen || py_eq(x, y);
      if (!seen) r.push_back(x);
    }
    return Val::List(std::move(r));
  }
  if (name == "sort") {
    auto it = iterate(v);
    const Val *attr = arg(2, "attribute");
    auto key = [&](const Val &x) { return attr ? get_attr(x, to_str(*attr)) : x; };
    std::stable_sort(it.begin(), it.end(), [&](const Val &x, const Val &y) { return py_cmp(key(x), key(y)) < 0; });
    if (flag(0, "reverse", false)) std::reverse(it.begin(), it.end());
    return Val::List(std::move(it));
  }
  if (name == "map") {
    std::vector<Val> r;
    const Val *attr = nullptr;
    for (auto &x : kw)
      if (x.first == "attribute") attr = &x.second;
    if (attr) {
      const Val *dflt = nullptr;
      for (auto &x : kw)
        if (x.first == "default") dflt = &x.second;
      for (const auto &x : iterate(v)) {
        Val y = get_attr(x, to_str(*attr));
        r.push_back(y.k == Val::undef && dflt ? *dflt : y);
      }
      return Val::List(std::move(r));
    }
    if (a.empty()) fail("map needs a filter name or attribute=");
    const std::string fn = to_str(a[0]);
    Args rest(a.begin() + 1, a.end());
    Kw nk;
    for (const auto &x : iterate(v)) r.push_back(filter(fn, x, rest, nk));
    return Val::List(std::move(r));
  }
  if (name == "select" || name == "reject" || name == "selectattr" || name == "rejectattr") {
    const bool attr = name.size() > 6, keep = name[0] == 's';
    size_t ti = attr ? 1 : 0;
    std::vector<Val> r;
    for (const auto &x : iterate(v)) {
      const Val y = attr ? get_attr(x, to_str(a.at(0))) : x;
      bool ok;
      if (a.size() > ti) {
        Args targs(a.begin() + static_cast<long>(ti) + 1, a.end());
        ok = test(to_str(a[ti]), y, targs);
      } else {
        ok = truthy(y);
      }
      if (ok == keep) r.push_back(x);
    }
    return Val::List(std::move(r));
  }
  if (name == "attr") {  // Python's getattr only: a dict's keys are not attributes
    if (v.k == Val::undef) undefined_error(v);
    Val m = method(v, to_str(a.at(0)));
    if (m.k == Val::func) return m;
    if (v.k == Val::dict && v.ns)
      if (const Val *x = v.find(to_str(a.at(0)))) return *x;
    return Val::Undef(std::string(type_name(v)) + " object' has no attribute '" + to_str(a.at(0)));
  }
  if (name == "indent") {
    const Val *w = arg(0, "width");
    std::string pad = w && w->k == Val::str ? w->s : std::string(static_cast<size_t>(w ? w->as_int() : 4), ' ');
    const bool first = flag(1, "first", false), blank = flag(2, "blank", false);
    const std::string s = to_str(v);
    std::string o;
    size_t p = 0;
    bool line0 = true;
    while (true) {
      const size_t e = s.find('\n', p);
      const std::string ln = s.substr(p, e == std::string::npos ? std::string::npos : e - p);
      if ((!line0 || first) && (blank || !ln.empty())) o += pad;
      o += ln;
      if (e == std::string::npos) break;
      o += '\n';
      p = e + 1;
      line0 = false;
    }
    return Val::Str(o);
  }
  if (name == "escape" || name == "e" || name == "forceescape") {
    std::string o;
    for (const char c : to_str(v)) {
      switch (c) {
        case '&': o += "&amp;"; break;
        case '<': o += "&lt;"; break;
        case '>': o += "&gt;"; break;
        case '"': o += "&#34;"; break;
        case '\'': o += "&#39;"; break;
        default: o += c;
      }
    }
    return Val::Str(o);
  }
  if (name == "wordcount") return Val::Int(static_cast<int64_t>(py_split(to_str(v), nullptr, -1).size()));
  fail("no filter named '" + name + "'");
}

Val Eval::eval(const Expr &e, Frame &f) {
  switch (e.k) {
    case Expr::lit: return e.v;
    case Expr::name: {
      if (const Val *v = f.lookup(e.s)) return *v;
      return Val::Undef(e.s);
    }
    case Expr::attr: return get_attr(eval(*e.a[0], f), e.s);
    case Expr::item: return get_item(eval(*e.a[0], f), eval(*e.a[1], f));
    case Expr::slice: {
      Val o = eval(*e.a[0], f);
      Val p[3];
      for (int k = 0; k < 3; ++k)
        if (e.a[k + 1]) p[k] = eval(*e.a[k + 1], f);
      return slice(o, e.a[1] ? &p[0] : nullptr, e.a[2] ? &p[1] : nullptr, e.a[3] ? &p[2] : nullptr);
    }
    case Expr::call: {
      Val fn = eval(*e.a[0], f);
      Args args;
      for (size_t k = 1; k < e.a.size(); ++k) args.push_back(eval(*e.a[k], f));
      Kw kw;
      for (const auto &x : e.kw) kw.emplace_back(x.first, eval(*x.second, f));
      return call(fn, args, kw);
    }
    case Expr::filter: {
      Val v = eval(*e.a[0], f);
      Args args;
      for (size_t k = 1; k < e.a.size(); ++k) args.push_back(eval(*e.a[k], f));
      Kw kw;
      for (const auto &x : e.kw) kw.emplace_back(x.first, eval(*x.second, f));
      return filter(e.s, std::move(v), args, kw);
    }
    case Expr::test: {
      Val v = eval(*e.a[0], f);
      Args args;
      for (size_t k = 1; k < e.a.size(); ++k) args.push_back(eval(*e.a[k], f));
      return Val::Bool(test(e.s, v, args) != e.negate);
    }
    case Expr::bin: return binop(e.s, eval(*e.a[0], f), eval(*e.a[1], f));
    case Expr::and_: {
      Val a = eval(*e.a[0], f);
      return truthy(a) ? eval(*e.a[1], f) : a;
    }
    case Expr::or_: {
      Val a = eval(*e.a[0], f);
      return truthy(a) ? a : eval(*e.a[1], f);
    }
    case Expr::not_: return Val::Bool(!truthy(eval(*e.a[0], f)));
    case Expr::neg:
    case Expr::pos: {
      Val v = eval(*e.a[0], f);
      if (v.k == Val::undef) undefined_error(v);
      if (!v.numeric()) fail(std::string("bad operand type for unary ") + (e.k == Expr::neg ? "-" : "+") + ": '" + type_name(v) + "'");
      if (e.k == Expr::pos) return v.k == Val::real ? v : Val::Int(v.as_int());
      return v.k == Val::real ? Val::Real(-v.d) : Val::Int(-v.as_int());
    }
    case Expr::cond: {
      if (truthy(eval(*e.a[1], f))) return eval(*e.a[0], f);
      return e.a[2] ? eval(*e.a[2], f) : Val::Undef("the conditional expression's else");
    }
    case Expr::list:
    case Expr::tuple: {
      std::vector<Val> r;
      for (const auto &x : e.a) r.push_back(eval(*x, f));
      return Val::List(std::move(r), e.k == Expr::tuple);
    }
    case Expr::dict: {
      Val d = Val::Dict();
      for (size_t k = 0; k + 1 < e.a.size(); k += 2) d.put(to_str(eval(*e.a[k], f)), eval(*e.a[k + 1], f));
      return d;
    }
  }
  return Val::Undef();
}

Flow Eval::exec(const Body &b, Frame &f, std::string &o) {
  for (const auto &np : b) {
    const Node &n = *np;
    switch (n.k) {
      case Node::text: o += n.s; break;
      case Node::out: py_str(eval(*n.e, f), o); break;
      case Node::if_: {
        size_t k = 0;
        for (; k < n.conds.size(); ++k)
          if (truthy(eval(*n.conds[k], f))) break;
        if (k < n.bodies.size()) {
          const Flow fl = exec(n.bodies[k], f, o);
          if (fl != Flow::normal) return fl;
        }
        break;
      }
      case Node::for_: {
        std::vector<Val> items = iterate(eval(*n.e, f));
        Frame lf;
        lf.parent = &f;
        if (n.filt) {
          std::vector<Val> kept;
          for (const auto &x : items) {
            assign(lf, n.names, x);
            if (truthy(eval(*n.filt, lf))) kept.push_back(x);
          }
          items = std::move(kept);
        }
        const int64_t len = static_cast<int64_t>(items.size());
        for (int64_t i = 0; i < len; ++i) {
          lf.vars.clear();
          assign(lf, n.names, items[static_cast<size_t>(i)]);
          Val loop = Val::Dict();
          loop.put("index0", Val::Int(i));
          loop.put("index", Val::Int(i + 1));
          loop.put("revindex0", Val::Int(len - i - 1));
          loop.put("revindex", Val::Int(len - i));
          loop.put("first", Val::Bool(i == 0));
          loop.put("last", Val::Bool(i == len - 1));
          loop.put("length", Val::Int(len));
          loop.put("depth", Val::Int(1));
          loop.put("depth0", Val::Int(0));
          loop.put("previtem", i > 0 ? items[static_cast<size_t>(i - 1)] : Val::Undef("there is no previous item"));
          loop.put("nextitem", i + 1 < len ? items[static_cast<size_t>(i + 1)] : Val::Undef("there is no next item"));
          loop.put("cycle", Val::Func([i](Args &a, Kw &) {
            if (a.empty()) fail("no items for cycling given");
            return a[static_cast<size_t>(i) % a.size()];
          }));
          lf.vars["loop"] = loop;
          const Flow fl = exec(n.bodies[0], lf, o);
          if (fl == Flow::brk) break;
        }
        if (len == 0 && n.bodies.size() > 1) {
          const Flow fl = exec(n.bodies[1], f, o);
          if (fl != Flow::normal) return fl;
        }
        break;
      }
      case Node::set: assign(f, n.names, eval(*n.e, f)); break;
      case Node::set_attr: {
        const Val *ns = f.lookup(n.names[0]);
        if (!ns || ns->k != Val::dict || !ns->ns) fail("cannot assign attribute on non-namespace object");
        Val v = eval(*n.e, f);
        const_cast<Val *>(ns)->put(n.s, std::move(v));
        break;
      }
      case Node::set_block: {
        std::string cap;
        exec(n.bodies[0], f, cap);
        f.vars[n.names[0]] = Val::Str(cap);
        break;
      }
      case Node::macro: {
        const Node *m = &n;
        Eval *self = this;
        f.vars[n.s] = Val::Func([self, m](Args &a, Kw &kw) { return self->call_macro(*m, a, kw); });
        break;
      }
      case Node::brk: return Flow::brk;
      case Node::cont: return Flow::cont;
      case Node::filter_block: {
        std::string cap;
        exec(n.bodies[0], f, cap);
        Args a;
        Kw kw;
        py_str(filter(n.e->s, Val::Str(cap), a, kw), o);
        break;
      }
      case Node::scope: {
        const Flow fl = exec(n.bodies[0], f, o);
        if (fl != Flow::normal) return fl;
        break;
      }
    }
  }
  return Flow::normal;
}

}  // namespace

struct Template::Impl {
  Body body;
  bool content_loop = false;
};

Template::Template(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Template::~Template() = default;

std::shared_ptr<const Template> Template::parse(std::string_view source, std::string *err) {
  try {
    Parser p(segment(source));
    auto impl = std::make_unique<Impl>();
    impl->body = p.parse_all();
    impl->content_loop = p.content_loop;
    return std::shared_ptr<const Template>(new Template(std::move(impl)));
  } catch (const Error &e) {
    if (err) *err = e.msg;
  } catch (const std::exception &e) {
    if (err) *err = e.what();
  }
  return nullptr;
}

bool Template::iterates_content() const { return impl_->content_loop; }

bool Template::render(const json::Value &ctx, std::string &out, std::string &err) const {
  out.clear();
  try {
    Frame globals;
    globals.vars["namespace"] = Val::Func([](Args &a, Kw &kw) {
      Val d = Val::Dict();
      d.ns = true;
      for (const auto &x : a)
        if (x.k == Val::dict)
          for (const auto &kv : *x.m) d.put(kv.first, kv.second);
      for (auto &kv : kw) d.put(kv.first, kv.second);
      return d;
    });
    globals.vars["dict"] = Val::Func([](Args &a, Kw &kw) {
      Val d = Val::Dict();
      for (const auto &x : a)
        if (x.k == Val::dict)
          for (const auto &kv : *x.m) d.put(kv.first, kv.second);
      for (auto &kv : kw) d.put(kv.first, kv.second);
      return d;
    });
    globals.vars["range"] = Val::Func([](Args &a, Kw &) {
      int64_t lo = 0, hi = 0, st = 1;
      if (a.size() == 1) hi = a[0].as_int();
      else if (a.size() >= 2) lo = a[0].as_int(), hi = a[1].as_int();
      if (a.size() >= 3) st = a[2].as_int();
      if (st == 0) fail("range() arg 3 must not be zero");
      std::vector<Val> r;
      for (int64_t i = lo; st > 0 ? i < hi : i > hi; i += st) {
        r.push_back(Val::Int(i));
        if (r.size() > 1000000) fail("range too large");
      }
      return Val::List(std::move(r));
    });
    globals.vars["raise_exception"] = Val::Func([](Args &a, Kw &) -> Val { fail(a.empty() ? std::string() : to_str(a[0])); });
    globals.vars["strftime_now"] = Val::Func([](Args &a, Kw &) {
      const std::time_t t = std::time(nullptr);
      std::tm tm{};
      localtime_r(&t, &tm);
      char b[256];
      const size_t n = std::strftime(b, sizeof b, a.empty() ? "%Y-%m-%d" : to_str(a[0]).c_str(), &tm);
      return Val::Str(std::string(b, n));
    });
    if (ctx.is_object())
      for (const auto &kv : ctx.obj) globals.vars[kv.first] = from_json(kv.second);
    Eval ev;
    ev.root = &globals;
    ev.exec(impl_->body, globals, out);
    return true;
  } catch (const Error &e) {
    err = e.msg;
  } catch (const std::exception &e) {
    err = e.what();
  }
  return false;
}

std::string load_chat_template(const std::string &model_dir, json::Value *globals) {
  auto read = [](const std::string &p, std::string &s) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    s = ss.str();
    return true;
  };
  std::string tpl, cfg;
  json::Value tc;
  const bool have_cfg = read(model_dir + "/tokenizer_config.json", cfg) && json::parse(cfg, tc) && tc.is_object();
  if (globals) {
    *globals = json::Value::object();
    if (have_cfg)
      for (const auto &kv : tc.obj) {
        const std::string &k = kv.first;
        if (k.size() < 6 || k.compare(k.size() - 6, 6, "_token") != 0) continue;
        const json::Value &v = kv.second;
        if (v.is_string()) globals->set(k, v);
        else if (v.is_object() && v.get_string("content")) globals->set(k, json::Value::string(*v.get_string("content")));
      }
  }
  if (read(model_dir + "/chat_template.jinja", tpl)) return tpl;
  std::string js;
  json::Value j;
  if (read(model_dir + "/chat_template.json", js) && json::parse(js, j) && j.get_string("chat_template")) return *j.get_string("chat_template");
  if (!have_cfg) return {};
  const json::Value *ct = tc.get("chat_template");
  if (!ct) return {};
  if (ct->is_string()) return ct->str;
  if (ct->is_array())
    for (const auto &e : ct->arr)
      if (const std::string *n = e.get_string("name"); n && *n == "default" && e.get_string("template")) return *e.get_string("template");
  return {};
}

}  // namespace xenithium:text:jinja
