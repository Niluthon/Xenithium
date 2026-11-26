#include "corpus.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace xenithium::bench {
namespace {

namespace fs = std::filesystem;

constexpr uint64_t kFnvOffset = 14695981039346656037ull, kFnvPrime = 1099511628211ull;

void fnv(uint64_t &h, std::string_view bytes) {
  for (const unsigned char c : bytes) h = (h ^ c) * kFnvPrime;
  h = (h ^ 0xff) * kFnvPrime;  // a separator, so ("ab", "c") and ("a", "bc") differ
}

std::string file_header(const CorpusFile &f) { return "### " + f.name + "\n```\n"; }
constexpr std::string_view kFileFooter = "\n```\n\n";

}  // namespace

namespace {
constexpr std::array kSets = {
    CorpusSet{"engine", kCorpusVersion, "this engine's own sources and documentation: real code and text"},
    CorpusSet{"generic", kGenericCorpusVersion, "conventional template-made code and prose (CRUD layers, tests, guides, manuals)"},
};
}  // namespace

std::span<const CorpusSet> corpus_sets() { return kSets; }

const CorpusSet *find_corpus_set(std::string_view name) {
  for (const CorpusSet &s : kSets)
    if (s.name == name) return &s;
  return nullptr;
}

Corpus::Corpus(std::vector<CorpusFile> files) : files_(std::move(files)) {}

Corpus Corpus::from_embedded(std::string_view set) {
  std::vector<CorpusFile> files;
  for (const embedded::File &f : std::span(embedded::kFiles, embedded::kFileCount))
    if (f.set == set) files.push_back({f.kind, f.name, std::string(reinterpret_cast<const char *>(f.data), f.size)});
  return Corpus(std::move(files));
}

std::optional<Corpus> Corpus::from_dir(const std::string &dir, std::string &err) {
  std::vector<CorpusFile> files;
  std::error_code ec;
  for (const auto &kind : fs::directory_iterator(dir, ec)) {
    if (!kind.is_directory()) continue;
    for (const auto &file : fs::directory_iterator(kind.path(), ec)) {
      if (!file.is_regular_file()) continue;
      std::ifstream in(file.path(), std::ios::binary);
      files.push_back({kind.path().filename().string(), file.path().filename().string(), std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>())});
    }
  }
  if (files.empty()) return err = "no <kind>/<file> text under " + dir, std::nullopt;
  std::ranges::sort(files, {}, [](const CorpusFile &f) { return f.kind + "/" + f.name; });
  return Corpus(std::move(files));
}

std::string Corpus::hash() const {
  uint64_t h = kFnvOffset;
  for (const CorpusFile &f : files_) fnv(h, f.kind), fnv(h, f.name), fnv(h, f.text);
  char hex[17];
  std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(h));
  return hex;
}

std::vector<std::string> Corpus::kinds() const {
  std::vector<std::string> kinds;
  for (const CorpusFile &f : files_)
    if (std::ranges::find(kinds, f.kind) == kinds.end()) kinds.push_back(f.kind);
  return kinds;
}

std::size_t Corpus::chars(std::string_view kind) const {
  std::size_t n = 0;
  for (const CorpusFile &f : files_)
    if (f.kind == kind) n += f.text.size();
  return n;
}

Corpus::Document Corpus::document(std::string_view kind, std::size_t chars, std::size_t first_file) const {
  std::vector<const CorpusFile *> of_kind;
  for (const CorpusFile &f : files_)
    if (f.kind == kind && !f.text.empty()) of_kind.push_back(&f);
  Document doc;
  if (of_kind.empty()) return doc;
  doc.text.reserve(chars + 64);
  for (std::size_t i = 0; doc.text.size() < chars; ++i) {
    doc.wrapped |= i >= of_kind.size();
    const CorpusFile &f = *of_kind[(first_file + i) % of_kind.size()];
    doc.text += file_header(f);
    const std::size_t room = chars > doc.text.size() ? chars - doc.text.size() : 0;
    std::string_view part(f.text);
    if (part.size() > room) {  // the last file: up to a line end inside the room (or the bare room in one long line)
      const std::size_t line_end = part.rfind('\n', room);
      std::size_t cut = line_end == std::string_view::npos ? room : line_end;
      // engine round 277 (backlog 306): never inside a UTF-8 character (the server refuses invalid UTF-8)
      while (cut > 0 && (static_cast<unsigned char>(part[cut]) & 0xC0) == 0x80) --cut;
      part = part.substr(0, cut);
    }
    doc.text.append(part);
    doc.text.append(kFileFooter);
  }
  return doc;
}

}  // namespace xenithium::bench
