// Xenithium · gate cards: the Battlemage card table, identification and the XEN_EMU_CARD emulation (util/cards.hpp).
// Written for this engine. Host only: the PCI ids of the three Arc Pro cards, the name and cores + VRAM fallbacks (a B580
// stays unknown), the emulation's value set, its refusals (a bigger card, an unknown key), the VRAM it hides (B60 on a
// B70: 8 GiB; B65: none), the cores and bandwidth it reports, the log line's wording, and the plan card it budgets for.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "xenithium/util/cards.hpp"

namespace {

int g_checks = 0, g_failures = 0;
#define CHECK(cond, ...)                                \
  do {                                                  \
    ++g_checks;                                         \
    if (!(cond)) {                                      \
      ++g_failures;                                     \
      std::printf("[FAIL] %s:%d: ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                         \
      std::printf("\n");                                \
    }                                                   \
  } while (0)

bool has(const std::string &s, const char *part) { return s.find(part) != std::string::npos; }

// the module sizes zeDeviceGetMemoryProperties reports (rt:card_view's input): a B70's (30.30 GiB; sysman says
// 31.89), the others by the same ratio
constexpr uint64_t kB70Vram = 31027ull << 20;  // 30.30 GiB
constexpr uint64_t kB60Vram = 22835ull << 20;  // 22.30 GiB
constexpr uint64_t kB580Vram = 11635ull << 20;

}  // namespace

int main() {
  namespace u = xenithium::util;
  const auto *b70 = u::card_by_key("b70"), *b65 = u::card_by_key("b65"), *b60 = u::card_by_key("b60");
  CHECK(b70 && b65 && b60 && !u::card_by_key("b580") && !u::card_by_key("off"), "the three keys, nothing else");
  CHECK(b70->xe_cores == 32 && b65->xe_cores == 20 && b60->xe_cores == 20, "Xe cores 32 / 20 / 20");
  CHECK(b70->vram_gb == 32 && b65->vram_gb == 32 && b60->vram_gb == 24, "VRAM 32 / 32 / 24 GB");
  CHECK(b70->bw_gbs == 608 && b65->bw_gbs == 608 && b60->bw_gbs == 456, "bandwidth 608 / 608 / 456 GB/s");
  CHECK(b70->floor_gbs == 604 && b65->floor_gbs == 604 && b60->floor_gbs == 453, "read floors 604 / 604 / 453 (= 456 x 604 / 608)");
  CHECK(b70->free_gib == 31.85 && std::abs(b60->free_gib - (31.85 - 8)) < 1e-9, "free memory: the B70's measured 31.85, the B60 8 GiB less");

  // identification: PCI id first
  CHECK(u::card_by_pci(0xe223) == b70 && u::card_by_pci(0xe222) == b65 && u::card_by_pci(0xe211) == b60, "pci ids e223 / e222 / e211");
  CHECK(u::identify_card(0xe223, "Intel(R) Graphics [0xe223]", 32, kB70Vram).spec == b70, "a B70 whose driver reports no marketing name");
  CHECK(std::string(u::identify_card(0xe211, "x", 20, kB60Vram).how) == "pci id", "how = pci id");
  // then the driver's name (an id the table does not list)
  const auto byname = u::identify_card(0x1234, "Intel(R) Arc(TM) Pro B65 Graphics", 20, kB70Vram);
  CHECK(byname.spec == b65 && std::string(byname.how) == "name", "by name: Pro B65");
  CHECK(u::nominal_vram_gb(b65, 32530182144ull) == 32 && u::nominal_vram_gb(b60, 1) == 24 && u::nominal_vram_gb(nullptr, 32530182144ull) == 32 &&
            u::nominal_vram_gb(nullptr, 0) == 0,
        "nominal GB: the table's for a known card, else scaled from the driver's figure (a B65's 30.3 GiB = 32 GB)");
  CHECK(u::nominal_vram_gb(b70, 32530182144ull) == 32 && u::nominal_vram_gb(b70, kB70Vram) == 32 && u::nominal_vram_gb(nullptr, kB70Vram) == 32,
        "nominal GB: a B70 (the driver's 32530182144 bytes from a 2x B70 field run, and the module size) is 32 GB, known or scaled");
  CHECK(!u::identify_card(0x1234, "Intel(R) Arc(TM) Pro B605 Graphics", 20, kB70Vram).spec, "a name that only starts with a key is no match");
  // then cores + VRAM for an unlisted Battlemage id
  CHECK(u::identify_card(0xe220, "Intel(R) Graphics", 32, kB70Vram).spec == b70, "G31 id, 32 cores, 32 GB = B70");
  CHECK(u::identify_card(0xe221, "Intel(R) Graphics", 20, kB70Vram).spec == b65, "G31 id, 20 cores, 32 GB = B65");
  CHECK(u::identify_card(0xe210, "Intel(R) Graphics", 20, kB60Vram).spec == b60, "G21 id, 20 cores, 24 GB = B60");
  CHECK(!u::identify_card(0xe20b, "Intel(R) Arc(TM) B580 Graphics", 20, kB580Vram).spec, "a B580 (20 cores, 12 GB) stays unknown");
  CHECK(!u::identify_card(0x56a0, "Intel(R) Arc(TM) A770 Graphics", 32, 16ull << 30).spec, "not Battlemage: unknown");
  CHECK(std::string(u::battlemage_die(0xe223)) == "BMG-G31" && std::string(u::battlemage_die(0xe211)) == "BMG-G21" && !*u::battlemage_die(0x56a0),
        "die by id");

  // no emulation
  unsetenv("XEN_EMU_CARD");
  {
    const auto v = u::card_view(0xe223, "Intel(R) Arc(TM) Pro B70 Graphics", 32, kB70Vram);
    CHECK(v.phys == b70 && !v.emu && v.cores == 32 && v.vram == kB70Vram && v.vram_cut == 0 && v.bw_gbs == 608, "B70 as itself");
    const std::string d = v.describe();
    CHECK(has(d, "Arc Pro B70 (BMG-G31, by pci id): 32 Xe cores, 30.3 GiB, 608 GB/s, 2,800 MHz"), "the log line: %s", d.c_str());
    CHECK(has(u::card_view(0xe20b, "B580", 20, kB580Vram).describe(), "unknown card"), "an unknown card says so");
    CHECK(&u::plan_card() == b70 || &u::plan_card() == b65 || &u::plan_card() == b60, "plan card: a card of the table");
  }
  for (const char *off : {"", "off", "0", "none"}) {
    setenv("XEN_EMU_CARD", off, 1);
    CHECK(!u::emulated_card() && u::emu_card_valid(off) && !u::card_view(0xe223, "B70", 32, kB70Vram).emu, "XEN_EMU_CARD='%s' = off", off);
  }

  // B60 on a B70: 20 cores, 8 GiB hidden, 456 GB/s
  setenv("XEN_EMU_CARD", "b60", 1);
  {
    CHECK(u::emulated_card() == b60 && &u::plan_card() == b60, "b60: the emulated card, also for --plan");
    const auto v = u::card_view(0xe223, "Intel(R) Arc(TM) Pro B70 Graphics", 32, kB70Vram);
    CHECK(v.phys == b70 && v.emu == b60 && v.cores == 20 && v.phys_cores == 32, "B60 on B70: 20 of 32 cores");
    CHECK(v.vram_cut == 8ull << 30 && v.vram == kB70Vram - (8ull << 30), "B60 on B70: 8 GiB hidden (%llu)", static_cast<unsigned long long>(v.vram_cut));
    CHECK(v.bw_gbs == 456, "B60 on B70: 456 GB/s");
    const std::string d = v.describe();
    CHECK(has(d, "emulating Arc Pro B60") && has(d, "launch shapes for 20 cores, 8.0 GiB of VRAM hidden") && has(d, "the other 12 cores still run") &&
              has(d, "bandwidth (456 GB/s)") && has(d, "clock (2,400 vs 2,800 MHz)"),
          "the emulation line names what is not emulated: %s", d.c_str());
    // a B60 on a B65: memory and bandwidth only
    const auto w = u::card_view(0xe222, "Intel(R) Arc(TM) Pro B65 Graphics", 20, kB70Vram);
    CHECK(w.emu == b60 && w.cores == 20 && w.vram_cut == 8ull << 30, "B60 on B65");
    // an unknown 32 GB card (by its reported 30.30 GiB): the same 8 GiB
    const auto z = u::card_view(0x1234, "Some Xe2 card", 32, kB70Vram);
    CHECK(!z.phys && z.emu == b60 && z.vram_cut == 8ull << 30, "B60 on an unknown 32 GB card: 8 GiB hidden");
    // already a B60: nothing to take away
    const auto x = u::card_view(0xe211, "Intel(R) Arc(TM) Pro B60 Graphics", 20, kB60Vram);
    CHECK(!x.emu && x.vram_cut == 0 && x.cores == 20, "B60 on a B60: no emulation");
  }

  // B65 on a B70: cores only
  setenv("XEN_EMU_CARD", "b65", 1);
  {
    const auto v = u::card_view(0xe223, "Intel(R) Arc(TM) Pro B70 Graphics", 32, kB70Vram);
    CHECK(v.emu == b65 && v.cores == 20 && v.vram_cut == 0 && v.vram == kB70Vram && v.bw_gbs == 608, "B65 on B70: 20 cores, all memory");
    CHECK(!has(v.describe(), "bandwidth ("), "B65: no bandwidth to emulate");
    bool threw = false;
    try {
      (void)u::card_view(0xe211, "Intel(R) Arc(TM) Pro B60 Graphics", 20, kB60Vram);  // 32 GB on a 24 GB card
    } catch (const std::runtime_error &e) {
      threw = has(e.what(), "larger than this Arc Pro B60");
    }
    CHECK(threw, "a B65 on a B60 is refused (more memory and bandwidth)");
  }
  setenv("XEN_EMU_CARD", "b70", 1);
  {
    bool threw = false;
    try {
      (void)u::card_view(0xe222, "B65", 20, kB70Vram);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    CHECK(threw, "a B70 on a B65 is refused (more cores)");
  }
  setenv("XEN_EMU_CARD", "b580", 1);
  {
    CHECK(!u::emulated_card() && !u::emu_card_valid("b580"), "an unknown key: no card, not valid");
    bool threw = false;
    try {
      (void)u::card_view(0xe223, "B70", 32, kB70Vram);
    } catch (const std::runtime_error &e) {
      threw = has(e.what(), "off|b70|b65|b60");
    }
    CHECK(threw, "an unknown key is refused, naming the choices");
  }
  unsetenv("XEN_EMU_CARD");

  std::printf("%s: %d checks, %d failures\n", g_failures ? "FAIL" : "PASS", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
