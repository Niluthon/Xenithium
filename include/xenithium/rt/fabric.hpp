// Xenithium · one Level Zero context over several cards.
// The all-reduce needs this and nothing else does: Level Zero only permits a
// peer copy between devices that share a context, and the design's wire is a peer
// write into the other card's mailbox plus a reduce fused into the local kernel.
#pragma once

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <cstddef>
#include <memory>
#include <vector>

#include "xenithium/rt/l0.hpp"

namespace xenithium {

namespace kernels { struct StepCopy; }

class Fabric {
 public:
  // profiling: the compute queues record event timestamps (probes and tests read them). Servers pass false unless a
  // profile is requested: every submit to a profiling queue costs extra host time.
  Fabric(const l0::Loader &loader, std::vector<size_t> ranks, bool profiling = true);
  bool profiling() const { return profiling_; }
  // the property list of a compute queue (in order, plus profiling when on); side queues of a model take the same
  static sycl::property_list queue_props(bool profiling);

  size_t size() const { return ranks_.size(); }
  size_t rank_index(size_t r) const { return ranks_.at(r); }
  ze_device_handle_t ze_device(size_t r) const { return devs_.at(r); }
  ze_context_handle_t ze_context() const { return ctx_.handle(); }
  // The SYCL context over every card: Level Zero borrows its native handle (allocations, peer residency). Queues do not
  // use it: they take card_context(r).
  sycl::context &sycl_context() { return *sctx_; }
  // The SYCL context of card r's queues: made from the same native Level Zero context (so USM and peer-resident mailboxes
  // are shared), holding only card r. With one SYCL context over both cards the oneAPI 2026.1 / 2026.2 runtime's fast
  // kernel cache misses on every launch of a kernel that runs on both cards, and every miss appends an entry it never
  // frees (KernelProgramCache:saveKernel), so the host RAM grows without bound. Every queue of card r (compute, side,
  // snapshot, graphs) must use this context: a queue on sycl_context brings the leak back and its events would cross
  // contexts.
  sycl::context &card_context(size_t r) { return cctx_.at(r); }
  sycl::device &device(size_t r) { return sdevs_.at(r); }
  sycl::queue &compute(size_t r) { return compute_.at(r); }
  l0::Queue &copy(size_t r) { return copy_.at(r); }

  void *alloc(size_t r, size_t bytes) { return ctx_.alloc(bytes, l0::MemKind::Device, devs_.at(r)); }
  void *alloc_host(size_t bytes, l0::MemKind kind = l0::MemKind::Host) { return ctx_.alloc(bytes, kind, nullptr); }
  // Device memory on rank r that kernels on the other ranks may also store to / load
  // from: made resident on every rank. Without it a peer kernel access faults and the
  // device is lost. Mailboxes only: peer residency may mirror the buffer
  // into host RAM (compute-runtime #968), so keep these small.
  void *alloc_mailbox(size_t r, size_t bytes);
  // Fused kernel mailboxes use pinned host USM. Concurrent kernel reads and
  // peer stores into device-backed inboxes can return stale or torn sectors
  // on Battlemage. DMA buffers and separately sequenced flags still use
  // alloc_mailbox() so their device-resident transport is unchanged.
  void *alloc_kernel_mailbox(size_t bytes) { return alloc_host(bytes); }
  void alloc_step_copy(kernels::StepCopy &buffers, int rows, int n, bool needs_partial, std::vector<void *> &owned);
  void release(void *ptr) { ctx_.free(ptr); }

  // Driver's answer, not a promise: a peer copy across VMM blocks can still hang
  // (compute-runtime #995), which is exactly what the probe is for.
  bool can_access(size_t from, size_t to) const;
  uint32_t p2p_flags(size_t from, size_t to) const;

 private:
  std::vector<size_t> ranks_;
  bool profiling_ = true;
  std::vector<ze_device_handle_t> devs_;
  std::vector<sycl::device> sdevs_;
  std::unique_ptr<sycl::context> sctx_;
  std::vector<sycl::context> cctx_;  // [rank]: card_context
  l0::Context ctx_;
  std::vector<sycl::queue> compute_;
  std::vector<l0::Queue> copy_;
};

}  // namespace xenithium
