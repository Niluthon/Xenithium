// Xenithium · the Battlemage cards the engine knows: their specs, how a device is identified as one, and the card an
// emulation run pretends to be (XEN_EMU_CARD).
// Header-only and host-only (no SYCL / Level Zero): the launcher, the memory plan, the rt layer and the 256-GRF kernel
// library (a separate shared object) all read the same table, so it must not live in one library's static state.
// Sources: PCI ids from Intel's Xe KMD GPU list (dgpu-docs.intel.com, "Xe KMD Driver GPUs": E223 Arc Pro B70, E222 Arc
// Pro B65, E211 Arc Pro B60) and the xe module's aliases; the B70 row is read from the card (xpu-smi discovery: 256 XVE
// = 32 Xe cores, 2,800 MHz max; sysman 31.85 GiB free of 31.89 on an idle card, zeDeviceGetMemoryProperties 30.30); the
// B65 row is its spec sheet (20 Xe cores, 32 GB GDDR6 256-bit, 608 GB/s, 2,400 MHz, Gen5 x16); the
// B60 row is Intel's public spec (20 Xe cores, 24 GB GDDR6 192-bit, 456 GB/s, 2,400 MHz). The B65 / B60 memory figures
// (free_gib, l0_gib) and read floors (floor_gbs: spec x 604 / 608) are arithmetic from the B70's (same overhead; the B60
// has 8 GiB less).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <stdexcept>

namespace xenithium::util {

struct CardSpec {
  const char *key;  // "b70" | "b65" | "b60": the XEN_EMU_CARD value and the log key
  const char *name;  // "Arc Pro B70"
  const char *die;  // "BMG-G31"
  uint32_t pci_id;  // PCI device id (vendor 0x8086)
  int xe_cores;  // Xe2 cores (8 XVE each, 8 hardware threads per XVE)
  int vram_gb;  // GDDR6 (GiB-sized parts)
  int bus_bits;  // memory bus width
  double bw_gbs;  // peak memory bandwidth, GB/s (1e9)
  double floor_gbs;  // the read floor the budgets use: the B70's 604 of 608; the others scaled by it
  int max_mhz;  // graphics clock
  double free_gib;  // sysman free device memory on an idle card (what the loads budget from)
  double l0_gib;  // zeDeviceGetMemoryProperties' figure (the launcher's rough fit)
  bool measured;  // the row was read from a card (else spec sheet / arithmetic)
};

inline constexpr CardSpec kCards[] = {
    {"b70", "Arc Pro B70", "BMG-G31", 0xe223, 32, 32, 256, 608, 604, 2800, 31.85, 30.30, true},
    {"b65", "Arc Pro B65", "BMG-G31", 0xe222, 20, 32, 256, 608, 604, 2400, 31.85, 30.30, false},
    {"b60", "Arc Pro B60", "BMG-G21", 0xe211, 20, 24, 192, 456, 453, 2400, 23.85, 22.30, false},
};

inline const CardSpec *card_by_key(std::string_view key) {
  for (const auto &c : kCards)
    if (key == c.key) return &c;
  return nullptr;
}

inline const CardSpec *card_by_pci(uint32_t id) {
  for (const auto &c : kCards)
    if (c.pci_id == id) return &c;
  return nullptr;
}

// Battlemage = PCI ids 0xE200..0xE23F (the xe driver's BMG list: E202..E216 are G21 parts, E220..E223 G31).
inline bool is_battlemage(uint32_t id) { return (id & 0xffc0u) == 0xe200u; }
inline const char *battlemage_die(uint32_t id) { return !is_battlemage(id) ? "" : (id & 0xfff0u) == 0xe220u ? "BMG-G31" : "BMG-G21"; }

struct CardMatch {
  const CardSpec *spec = nullptr;  // nullptr = not a card of the table
  const char *how = "unknown";  // "pci id" | "name" | "cores + VRAM" | "unknown"
};

// What card a device is: its PCI id first (exact), then the driver's marketing name ("Intel(R) Arc(TM) Pro B60
// Graphics"; some drivers report only "Intel(R) Graphics [0xe223]"), then, for an unlisted Battlemage id, its Xe cores and
// memory (32 cores and more than 28 GiB = B70; 20 cores with more than 28 GiB = B65, with 20-28 GiB = B60).
inline CardMatch identify_card(uint32_t pci_id, std::string_view name, int xe_cores, uint64_t vram_bytes) {
  if (const CardSpec *c = card_by_pci(pci_id)) return {c, "pci id"};
  for (const auto &c : kCards) {
    const std::string tag = std::string("Pro ") + static_cast<char>(c.key[0] - 'a' + 'A') + (c.key + 1);  // "Pro B60"
    const size_t at = name.find(tag);
    if (at != std::string_view::npos && (at + tag.size() == name.size() || name[at + tag.size()] == ' ')) return {&c, "name"};
  }
  if (is_battlemage(pci_id)) {
    const double gib = static_cast<double>(vram_bytes) / 1073741824.0;
    if (xe_cores == 32 && gib > 28) return {card_by_key("b70"), "cores + VRAM"};
    if (xe_cores == 20 && gib > 28) return {card_by_key("b65"), "cores + VRAM"};
    if (xe_cores == 20 && gib > 20) return {card_by_key("b60"), "cores + VRAM"};  // a B580 (20 cores, 12 GB) stays unknown
  }
  return {};
}

// A device's nominal memory in GB (the size on the box): the table's for a known card, else from the reported module
// size (zeDeviceGetMemoryProperties: a 32 GB B70 reports 30.30 GiB, sysman 31.89; the B70's ratio scales an unknown card).
inline int nominal_vram_gb(const CardSpec *spec, uint64_t vram_bytes) {
  if (spec) return spec->vram_gb;
  return static_cast<int>(std::lround(static_cast<double>(vram_bytes) / 1073741824.0 * 32 / 30.30));
}

// XEN_EMU_CARD: the card this process pretends to run on (launch shapes for its cores, its VRAM; docs/server-env.md).
// Unset, empty or "off" = none. An unknown value returns nullptr here (kernels must not throw); the server's option check
// and rt's Fabric refuse it (emu_card_valid).
inline std::string_view emu_card_value() {
  const char *v = std::getenv("XEN_EMU_CARD");
  return v ? std::string_view(v) : std::string_view();
}
inline bool emu_card_off(std::string_view v) { return v.empty() || v == "off" || v == "0" || v == "none"; }
inline bool emu_card_valid(std::string_view v) { return emu_card_off(v) || card_by_key(v) != nullptr; }
inline const CardSpec *emulated_card() {
  const std::string_view v = emu_card_value();
  return emu_card_off(v) ? nullptr : card_by_key(v);
}

// What a device is and what it runs as. card_view throws std:runtime_error for an unknown XEN_EMU_CARD value or an
// emulated card larger than the device in cores, memory or bandwidth (an emulation can only take away).
struct CardView {
  const CardSpec *phys = nullptr;  // the identified card (nullptr = not in kCards)
  const char *how = "unknown";  // how it was identified (identify_card)
  const CardSpec *emu = nullptr;  // the emulated card (XEN_EMU_CARD), nullptr = none (also when it is the card itself)
  int phys_cores = 0, cores = 0;  // Xe cores: the device's, the effective
  uint64_t phys_vram = 0, vram = 0;  // memory module bytes: the device's, the effective
  uint64_t vram_cut = 0;  // bytes the emulation hides from every budget: the nominal GB difference (B60 on a B70: 8 GiB)
  double bw_gbs = 0;  // effective peak memory bandwidth (spec), 0 = unknown card
  std::string describe() const;  // the "card N..." log line's text
};
CardView card_view(uint32_t pci_id, std::string_view name, int xe_cores, uint64_t vram_bytes);

// The card a host-only plan should budget for without opening a device: the emulated card, else the first Battlemage
// card in sysfs (/sys/class/drm/card*/device/device), else the B70 (the default).
const CardSpec &plan_card();

}  // namespace xenithium:util
