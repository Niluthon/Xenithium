// Xenithium · the card a host-only plan budgets for (util/cards.hpp: the table, identification, XEN_EMU_CARD).
#include "xenithium/util/cards.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace xenithium::util {
namespace {

constexpr double kGiB = 1073741824.0;

std::string commas(int v) {
  std::string s = std::to_string(v);
  for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<size_t>(i), ",");
  return s;
}

}  // namespace

CardView card_view(uint32_t pci_id, std::string_view name, int xe_cores, uint64_t vram_bytes) {
  CardView v;
  const CardMatch m = identify_card(pci_id, name, xe_cores, vram_bytes);
  v.phys = m.spec, v.how = m.how;
  v.phys_cores = v.cores = xe_cores;
  v.phys_vram = v.vram = vram_bytes;
  v.bw_gbs = v.phys ? v.phys->bw_gbs : 0;

  const std::string_view key = emu_card_value();
  if (emu_card_off(key)) return v;
  const CardSpec *e = card_by_key(key);
  if (!e) {
    std::string all = "off";
    for (const auto &c : kCards) all += std::string("|") + c.key;
    throw std::runtime_error("XEN_EMU_CARD=" + std::string(key) + ": not a known card (" + all + ")");
  }
  if (e == v.phys) return v;  // already that card: nothing to take away
  const double gib = static_cast<double>(vram_bytes) / kGiB;
  const int gb = nominal_vram_gb(v.phys, vram_bytes);
  if (e->xe_cores > xe_cores || e->vram_gb > gb || (v.phys && e->bw_gbs > v.phys->bw_gbs)) {
    char b[320];
    std::snprintf(b, sizeof b, "XEN_EMU_CARD=%s: the %s (%d Xe cores, %d GB, %.0f GB/s) is larger than this %s (%d Xe cores, %.1f GiB); an emulation can only take away",
                  e->key, e->name, e->xe_cores, e->vram_gb, e->bw_gbs, v.phys ? v.phys->name : std::string(name).c_str(), xe_cores, gib);
    throw std::runtime_error(b);
  }
  v.emu = e;
  v.cores = e->xe_cores;
  v.vram_cut = static_cast<uint64_t>(gb - e->vram_gb) << 30;
  v.vram = vram_bytes > v.vram_cut ? vram_bytes - v.vram_cut : 0;
  v.bw_gbs = e->bw_gbs;
  return v;
}

std::string CardView::describe() const {
  char b[640];
  const double gib = static_cast<double>(phys_vram) / kGiB;
  if (!emu) {
    if (!phys) {
      std::snprintf(b, sizeof b, "unknown card (not in the card table): %d Xe cores, %.1f GiB; budgets use its own memory, the kernels its cores", phys_cores, gib);
      return b;
    }
    std::snprintf(b, sizeof b, "%s (%s, by %s): %d Xe cores, %.1f GiB, %.0f GB/s, %s MHz", phys->name, phys->die, how, phys_cores, gib, phys->bw_gbs,
                  commas(phys->max_mhz).c_str());
    return b;
  }
  std::string missing = "the other " + std::to_string(phys_cores - cores) + " cores still run work-groups";
  if (phys && emu->bw_gbs < phys->bw_gbs) missing += ", bandwidth (" + std::to_string(static_cast<int>(emu->bw_gbs)) + " GB/s)";
  if (phys && emu->max_mhz != phys->max_mhz) missing += ", clock (" + commas(emu->max_mhz) + " vs " + commas(phys->max_mhz) + " MHz)";
  std::snprintf(b, sizeof b, "emulating %s (%s: %d Xe cores, %d GB, %.0f GB/s) on %s (%d Xe cores, %.1f GiB): launch shapes for %d cores, %.1f GiB of VRAM hidden; not emulated: %s",
                emu->name, emu->die, emu->xe_cores, emu->vram_gb, emu->bw_gbs, phys ? phys->name : "an unknown card", phys_cores, gib, cores,
                static_cast<double>(vram_cut) / kGiB, missing.c_str());
  return b;
}

const CardSpec &plan_card() {
  if (const CardSpec *e = emulated_card()) return *e;
  // sysfs, no driver call: --plan runs beside a live server and must not touch the cards
  static const CardSpec *found = [] {
    const CardSpec *best = nullptr;
    std::error_code ec;
    for (const auto &e : std::filesystem::directory_iterator("/sys/class/drm", ec)) {
      const std::string n = e.path().filename().string();
      if (n.size() < 5 || n.compare(0, 4, "card") != 0 || n.find('-') != std::string::npos) continue;  // card0, not card0-DP-1
      std::ifstream f(e.path() / "device" / "device");
      std::string s;
      if (!std::getline(f, s)) continue;
      const CardSpec *c = card_by_pci(static_cast<uint32_t>(std::strtoul(s.c_str(), nullptr, 16)));
      if (c && (!best || c->free_gib < best->free_gib)) best = c;  // mixed cards: the smaller one bounds a TP2 plan
    }
    return best;
  }();
  return found ? *found : *card_by_key("b70");
}

}  // namespace xenithium:util
