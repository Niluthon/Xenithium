// xenithium-bench: the text the prompts are made of.
//
// Long prompts need real text: random tokens give untypical attention, routing and draft acceptance. A corpus is a
// frozen set of files by kind ("code", "prose"). Two sets are embedded in the binary (tools/bench/corpus/<set> through
// xenithium-bench-embed): "engine", this engine's own sources and documentation (the default: real code and text), and
// "generic", conventional template-made code and prose. --corpus DIR reads another directory of the <kind>/<file> layout. The set,
// its version and its hash go into every run file: two runs compare only on the same corpus.
#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace xenithium::bench {

// Bumped by hand whenever a file of the set under tools/bench/corpus/<set> changes (tests/test_bench.cpp pins each
// set's hash beside it).
inline constexpr int kCorpusVersion = 4;         // "engine" (named "hard" before the modes were merged)
inline constexpr int kGenericCorpusVersion = 1;  // "generic" (named "medium", and "easy" before that)

// The measuring method: the short prompts (prompts.cpp), how replies end and the decode headline's mix. Bumped by hand
// when any of them changes; two runs compare only at one version (runs from before it recorded a mode instead).
inline constexpr int kMethodVersion = 1;

// The embedded sets, the default first.
struct CorpusSet {
  std::string_view name;
  int version;
  std::string_view about;
};
std::span<const CorpusSet> corpus_sets();
const CorpusSet *find_corpus_set(std::string_view name);  // nullptr when no set has that name

namespace embedded {
struct File {
  const char *set;
  const char *kind;
  const char *name;
  const unsigned char *data;
  std::size_t size;
};
extern const File kFiles[];
extern const std::size_t kFileCount;
}  // namespace embedded

struct CorpusFile {
  std::string kind;
  std::string name;
  std::string text;
};

class Corpus {
 public:
  explicit Corpus(std::vector<CorpusFile> files);
  // One embedded set ("engine", "generic"); an unknown name gives an empty corpus.
  static Corpus from_embedded(std::string_view set = "engine");
  // DIR/<kind>/<file>, files in name order. nullopt + err when the directory holds none.
  static std::optional<Corpus> from_dir(const std::string &dir, std::string &err);

  // FNV-1a 64 over every file's kind, name and text, as 16 hex digits.
  std::string hash() const;
  std::vector<std::string> kinds() const;
  std::size_t chars(std::string_view kind) const;

  struct Document {
    std::string text;
    bool wrapped = false;  // the kind holds less text than asked: files repeat
  };
  // About `chars` characters of `kind`: whole files from the kind's file number `first_file` on, each under a line
  // naming it and inside a code fence, the last one cut at a line end. An unknown kind gives an empty document.
  Document document(std::string_view kind, std::size_t chars, std::size_t first_file = 0) const;

 private:
  std::vector<CorpusFile> files_;
};

// A turn said before the prompt's own: the prompt continues a conversation.
struct EarlierTurn {
  std::string_view role;  // "user" or "assistant"
  std::string_view text;
};

// A short prompt of the decode / concurrency scenarios: one user turn, after `earlier` turns when it continues a chat.
struct ShortPrompt {
  std::string_view id;        // "code-merge-lists"
  std::string_view category;  // "code"
  std::string_view text;
  int max_tokens;             // the reply length the category is measured at
  std::span<const EarlierTurn> earlier = {};
};

// The two kinds of short prompts. `open`: tasks the model writes from scratch, three per category. `given`: tasks whose
// reply mostly repeats or reshapes what the prompt gives - a diff or a whole file back, JSON from given data, a fix of
// given code, a chat that goes on - four per category, with the shorter reply lengths such tasks end at. Real work
// holds both, so the decode, concurrency and arrivals scenarios measure both (measured_prompts); the other scenarios
// pick open ones by index.
enum class PromptSet { open, given };
std::span<const ShortPrompt> short_prompts(PromptSet set = PromptSet::open);
// Every short prompt, per category in the usual order: its open tasks, then its tasks on given input.
std::span<const ShortPrompt> measured_prompts();

// The decode headline's weight per category: the share of a deployment's generated tokens that kind of output is.
struct CategoryWeight {
  std::string_view category;
  double weight;
};
// How a mix turns the categories' rates into one figure. `harmonic`: the rate at which that mix of tokens comes out
// (each category's time per token, weighted). `arithmetic`: the weighted mean of the categories' rates, the figure most
// public benchmarks quote; it reads higher, since a fast category lifts it more than a slow one pulls it down.
enum class MixAverage { harmonic, arithmetic };
// A named set of weights over every category (they sum to 1; a category may weigh 0).
struct WeightMix {
  std::string_view id;     // "benchmark": the decode scenario's `weights` parameter
  std::string_view about;  // where the shares come from (`list`, the docs)
  std::span<const CategoryWeight> weights;
  MixAverage average = MixAverage::harmonic;
};
std::span<const WeightMix> weight_mixes();          // benchmark (the default), agentic, even
// The two mixes every decode run reports, whichever heads it: benchmark and agentic (the report's switch between them).
std::span<const WeightMix> headline_mixes();
std::span<const CategoryWeight> category_weights();  // every category in the usual order (the agentic mix's weights)

}  // namespace xenithium::bench
