// xenithium-probe peers [--mib M] [--iters I]: the TP wire's peer path on this host, measured.
//
// The wire (kernels/comm/wire.cpp) writes every partial it owes a peer straight into that peer's mailbox, an allocation
// made resident on every card (Fabric::alloc_mailbox). Whether those stores land in the peer's VRAM over the PCIe switch
// or in a host mirror of it is the driver's decision, and it is the difference between a TP4 step that scales and one
// that does not: this command times the store the wire issues, per ordered pair, against a host-staged copy of the same
// bytes. About twice the staged rate is the peer path; at the staged rate the wire is paying for the mirror.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <level_zero/ze_api.h>
#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/rt/fabric.hpp"
#include "xenithium/rt/l0.hpp"

#include "probe.hpp"

namespace {

namespace esimd = sycl::ext::intel::esimd;

// The store mail_send issues (kernels/comm/mailbox_esimd.hpp): one 128-word block per item (the LSC block limit), L1 and
// L2 uncached, 128 B aligned. That header is private to the kernel library, so the store is repeated here; a change
// there has to be mirrored here for this number to stay the wire's number.
inline constexpr auto kUc = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>, esimd::cache_hint_L2<esimd::cache_hint::uncached>,
                                             esimd::alignment<128>};
constexpr int kWords = 128;

class PeerStoreK;

sycl::event peer_store(sycl::queue &q, uint32_t *peer, size_t words, uint32_t word) {
  return q.parallel_for<PeerStoreK>(sycl::range<1>(words / static_cast<size_t>(kWords)), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
    esimd::block_store<uint32_t, kWords>(peer + static_cast<size_t>(id[0]) * kWords, esimd::simd<uint32_t, kWords>(word), kUc);
  });
}

class PeerLoadK;

// The load the wire's mail_take issues, run on the owning card: what a kernel on the peer sees of the store above. A peer
// store that lands in a host mirror of the buffer instead of the card's VRAM is invisible here, even when a copy-engine
// read of the same pointer returns the data.
sycl::event peer_load(sycl::queue &q, const uint32_t *src, uint32_t *dst, size_t words) {
  return q.parallel_for<PeerLoadK>(sycl::range<1>(words / static_cast<size_t>(kWords)), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
    const size_t at = static_cast<size_t>(id[0]) * kWords;
    esimd::block_store<uint32_t, kWords>(dst + at, esimd::block_load<uint32_t, kWords>(src + at, kUc), kUc);
  });
}

double seconds_since(const std::chrono::steady_clock::time_point &t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// The PCIe link a card reports (sysfs, the endpoint's own view). On this host the switch chain reports 2.5 GT/s x1 for
// every card while the data path is far wider than that (the 121 GB checkpoint loads in seconds), so the platform's
// number is printed as it stands and the ceiling the store is judged against is the one the staged copy measures.
std::string link_state(const std::string &bdf) {
  const std::string base = "/sys/bus/pci/devices/" + bdf + "/";
  std::string speed, width;
  if (FILE *f = std::fopen((base + "current_link_speed").c_str(), "r")) {
    char buf[64] = {0};
    if (std::fgets(buf, sizeof buf, f)) {
      speed = buf;
      while (!speed.empty() && (speed.back() == '\n' || speed.back() == ' ')) speed.pop_back();
      const size_t sp = speed.find(' ');
      if (sp != std::string::npos) speed = speed.substr(0, sp) + " GT/s";
    }
    std::fclose(f);
  }
  if (FILE *f = std::fopen((base + "current_link_width").c_str(), "r")) {
    char buf[32] = {0};
    if (std::fgets(buf, sizeof buf, f)) width = std::string(" x") + buf;
    std::fclose(f);
    while (!width.empty() && (width.back() == '\n' || width.back() == ' ')) width.pop_back();
  }
  return speed.empty() ? std::string("?") : speed + width;
}

std::string p2p_flags(uint32_t flags) {
  std::string s;
  if (flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS) s += "access";
  if (flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS) s += s.empty() ? "atomics" : "+atomics";
  return s.empty() ? std::string("none") : s;
}

}  // namespace

int cmd_peers(int mib, int iters) {
  const auto loader = xenithium::l0::Loader::open();
  const size_t cards = loader.size();
  if (cards < 2) {
    std::printf("peers: %zu device(s) visible; a peer link needs two\n", cards);
    return 0;
  }
  std::vector<size_t> ranks(cards);
  for (size_t r = 0; r < cards; ++r) ranks[r] = r;
  xenithium::Fabric fab(loader, ranks, true);

  const size_t bytes = static_cast<size_t>(mib) << 20, words = bytes / 4;
  const uint32_t word = 0x3f800000u;  // 1.0f: never the wire's empty-slot word
  const std::vector<uint32_t> want(words, word);
  // The card an allocation belongs to, from Level Zero: the mailbox must be the peer's.
  auto owner = [&](void *p) {
    ze_memory_allocation_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    ze_device_handle_t dev = nullptr;
    if (zeMemGetAllocProperties(fab.ze_context(), p, &props, &dev) != ZE_RESULT_SUCCESS) return std::string("?");
    if (props.type != ZE_MEMORY_TYPE_DEVICE || dev == nullptr) return std::string(props.type == ZE_MEMORY_TYPE_HOST ? "host" : "shared");
    for (size_t r = 0; r < cards; ++r)
      if (loader.device(r) == dev) return std::to_string(r);
    return std::string("other");
  };

  std::printf("peers: %zu cards, %d MiB per direction, %d timed runs each, one store per 512 B (the wire's mail_send block)\n", cards, mib, iters);
  std::printf("links, as the platform reports them (sysfs, the endpoint's own view):");
  for (size_t r = 0; r < cards; ++r) std::printf(" card %zu %s", r, link_state(loader.info(r).pci_bdf()).c_str());
  std::printf("\n");
  std::printf("  judged against a measured ceiling, not that figure: a staged copy is two hops (B/2), so one hop is 2 x staged\n");
  std::printf("  and a peer store crosses one hop.\n\n");
  // Each card's one-way host<->device rate: that is the link ceiling, measured, not taken from sysfs (which reports
  // 2.5 GT/s x1 for every card here while the 121 GB checkpoint loads in seconds). A peer store leaves the sender's link
  // and enters the receiver's, so the ceiling for a pair is the slower of the two.
  std::vector<double> link_gbs(cards, 0.0);
  {
    const size_t probe_bytes = std::min<size_t>(bytes, size_t{32} << 20);
    void *stage = fab.alloc_host(probe_bytes);
    for (size_t c = 0; c < cards; ++c) {
      void *buf = fab.alloc(c, probe_bytes);
      auto &q = fab.copy(c);
      const int reps = 8;
      q.copy(stage, buf, probe_bytes, true);  // warm-up
      auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < reps; ++r) q.copy(stage, buf, probe_bytes, true);  // card -> host
      const double out_gbs = static_cast<double>(probe_bytes) * reps / seconds_since(t0) / 1e9;
      q.copy(buf, stage, probe_bytes, true);
      t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < reps; ++r) q.copy(buf, stage, probe_bytes, true);  // host -> card
      const double in_gbs = static_cast<double>(probe_bytes) * reps / seconds_since(t0) / 1e9;
      link_gbs[c] = std::min(out_gbs, in_gbs);
      fab.release(buf);
      std::printf("link card %zu (%s): one way %.1f GB/s out / %.1f GB/s in -> ceiling %.1f GB/s\n", c, loader.info(c).pci_bdf().c_str(), out_gbs, in_gbs,
                  link_gbs[c]);
    }
    fab.release(stage);
    std::printf("\n");
  }
  std::printf("%-6s %-14s %-14s %6s %10s %11s %12s %10s %9s\n", "pair", "peer bdf", "driver p2p", "owner", "peer GB/s", "staged GB/s", "one hop GB/s",
              "peer/onehop", "copy/dev/fresh");
  int peer_ok = 0, pairs = 0;
  double peer_gbs_sum = 0, one_hop_sum = 0, link_sum = 0;
  for (size_t i = 0; i < cards; ++i) {
    auto &q = fab.compute(i);
    auto &ci = fab.copy(i);
    for (size_t j = 0; j < cards; ++j) {
      if (i == j) continue;
      ++pairs;
      // an allocation on card j that card i stores into: exactly what the wire allocates for its inbox
      void *mbx = fab.alloc_mailbox(j, bytes);
      void *src = fab.alloc(i, bytes);
      void *dst = fab.alloc(j, bytes);
      void *stage = fab.alloc_host(bytes);

      (void)peer_store(q, static_cast<uint32_t *>(mbx), words, word).wait();  // warm-up
      const auto t_peer = std::chrono::steady_clock::now();
      for (int it = 0; it < iters; ++it) (void)peer_store(q, static_cast<uint32_t *>(mbx), words, word).wait();
      const double peer_s = seconds_since(t_peer);

      std::vector<uint32_t> back(words), seen(words);
      fab.copy(j).copy(back.data(), mbx, bytes, true);
      const bool copy_ok = back == want;
      void *own = fab.alloc(j, bytes);
      peer_load(fab.compute(j), static_cast<const uint32_t *>(mbx), static_cast<uint32_t *>(own), words).wait();
      fab.copy(j).copy(seen.data(), own, bytes, true);
      fab.release(own);
      const bool dev_ok = seen == want;
      // A second store into the same mailbox: the peer's kernel read the lines once already, so a read that still sees
      // the first value is a stale L2 line the uncached hint did not bypass — the wire's spin would take it as the
      // message and the sum would be wrong with no error and no timeout.
      const uint32_t word2 = 0x40000000u;  // 2.0f
      (void)peer_store(q, static_cast<uint32_t *>(mbx), words, word2).wait();
      const std::vector<uint32_t> want2(words, word2);
      void *own2 = fab.alloc(j, bytes);
      std::vector<uint32_t> seen2(words);
      peer_load(fab.compute(j), static_cast<const uint32_t *>(mbx), static_cast<uint32_t *>(own2), words).wait();
      fab.copy(j).copy(seen2.data(), own2, bytes, true);
      fab.release(own2);
      const bool fresh_ok = seen2 == want2;
      const bool ok = copy_ok && dev_ok && fresh_ok;

      ci.copy(stage, src, bytes, true);  // warm-up through the two copy queues
      fab.copy(j).copy(dst, stage, bytes, true);
      const auto t_staged = std::chrono::steady_clock::now();
      for (int it = 0; it < iters; ++it) {
        ci.copy(stage, src, bytes, true);
        fab.copy(j).copy(dst, stage, bytes, true);
      }
      const double staged_s = seconds_since(t_staged);

      const double peer_gbs = static_cast<double>(bytes) * iters / peer_s / 1e9;
      const double staged_gbs = static_cast<double>(bytes) * iters / staged_s / 1e9;
      const double ratio = staged_gbs > 0 ? peer_gbs / staged_gbs : 0;
      const bool peer_path = ok && ratio >= 1.5;
      peer_ok += peer_path;
      const double one_hop_gbs = 2.0 * staged_gbs;
      const double link = std::min(link_gbs[i], link_gbs[j]);
      const double share = one_hop_gbs > 0 ? 100.0 * peer_gbs / one_hop_gbs : 0.0;
      const double link_share = link > 0 ? 100.0 * peer_gbs / link : 0.0;
      link_sum += link;
      peer_gbs_sum += peer_gbs;
      one_hop_sum += one_hop_gbs;
      std::printf("%-6s %-14s %-14s %6s %10.1f %11.1f %12.1f %9.0f%% %9s\n", (std::to_string(i) + "->" + std::to_string(j)).c_str(),
                  loader.info(j).pci_bdf().c_str(), p2p_flags(fab.p2p_flags(i, j)).c_str(), owner(mbx).c_str(), peer_gbs, staged_gbs, one_hop_gbs,
                  share, (std::string(copy_ok ? "ok/" : "BAD/") + (dev_ok ? "ok/" : "BAD/") + (fresh_ok ? "ok" : "STALE")).c_str());
      std::printf("       %-6s link ceiling %.1f GB/s (card %zu %.1f, card %zu %.1f): the peer store is %.0f%% of it\n",
                  "", link, i, link_gbs[i], j, link_gbs[j], link_share);
      if (!peer_path)
        std::printf("       ^ card %zu -> %zu: the peer store did not land as a peer store (%.2fx the staged copy; the copy engine %s, the peer's kernel %s, a rewrite %s)\n", i, j,
                    ratio, copy_ok ? "sees it" : "does not see it", dev_ok ? "sees it" : "does not see it", fresh_ok ? "lands" : "STAYS STALE");
      fab.release(mbx);
      fab.release(src);
      fab.release(dst);
      fab.release(stage);
    }
  }
  std::printf("\n%d of %d ordered pairs store into the peer's VRAM at >= 1.25x the host-staged rate (the P2P path)\n", peer_ok, pairs);
  if (pairs) {
    const double mean = peer_gbs_sum / pairs, hop = one_hop_sum / pairs;
    std::printf("mean peer store %.1f GB/s = %.0f%% of the %.1f GB/s one-hop ceiling the staged copy measures, and %.0f%% of the %.1f GB/s\n",
                mean, hop > 0 ? 100.0 * mean / hop : 0.0, hop, link_sum > 0 ? 100.0 * mean / (link_sum / pairs) : 0.0, link_sum / pairs);
    std::printf("measured one-way link ceiling (mean over the pairs)\n");
  }
  return peer_ok == pairs ? 0 : 1;
}
