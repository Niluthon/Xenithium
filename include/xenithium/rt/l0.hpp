// Xenithium · Level Zero layer (the engine talks to L0 directly).
// Everything the engine needs from Level Zero and nothing from oneCCL: init,
// device topology, context + USM allocation, and command queues we own.
#pragma once

#include <level_zero/ze_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace xenithium::l0 {

const char *result_name(ze_result_t rc);
[[noreturn]] void throw_error(ze_result_t rc, const char *what);

inline void check(ze_result_t rc, const char *what) {
  if (rc != ZE_RESULT_SUCCESS) throw_error(rc, what);
}

struct DeviceInfo {
  std::string name;
  uint32_t vendor_id = 0;
  uint32_t device_id = 0;
  uint16_t pci_domain = 0, pci_bus = 0, pci_device = 0, pci_function = 0;
  bool has_pci = false;
  uint64_t vram_bytes = 0;
  uint32_t slices = 0;
  uint32_t subslices_per_slice = 0;
  uint32_t eu_per_subslice = 0;
  uint32_t threads_per_eu = 0;
  uint32_t clock_mhz = 0;
  bool integrated = false;

  uint32_t eu_total() const { return slices * subslices_per_slice * eu_per_subslice; }
  std::string pci_bdf() const;
};

// Owns zeInit and the (affinity-filtered) device list.
class Loader {
 public:
  static Loader open();

  const std::vector<ze_driver_handle_t> &drivers() const { return drivers_; }
  const std::vector<ze_device_handle_t> &devices() const { return devices_; }
  size_t size() const { return devices_.size(); }
  ze_device_handle_t device(size_t i) const { return devices_.at(i); }
  ze_driver_handle_t driver_of(size_t i) const { return device_driver_.at(i); }
  DeviceInfo info(size_t i) const;
  bool can_access_peer(size_t i, size_t j) const;
  std::string api_version() const;
  uint32_t driver_version(size_t i) const;
  // The GPU driver's own version ("1.17.39395+16" for compute runtime 26.31.39395.16: Intel's version-string extension, as sycl-ls shows it), else "build 39395"
  // from driver_version; empty = not known.
  std::string driver_version_text(size_t i) const;

 private:
  std::vector<ze_driver_handle_t> drivers_;
  std::vector<ze_device_handle_t> devices_;
  std::vector<ze_driver_handle_t> device_driver_;
};

// HostUncached / HostWC: host USM with ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED /
// _WRITE_COMBINED (the CPU <-> card request wire of the cold-expert pool).
enum class MemKind { Device, Shared, Host, HostUncached, HostWC };

// Owns one ze_context. Allocations come from Level Zero, not from SYCL, so the
// engine controls residency (the tiering design depends on it).
class Context {
 public:
  Context() = default;
  explicit Context(ze_driver_handle_t driver, const std::vector<ze_device_handle_t> &devices);
  // Borrow a context someone else owns (the SYCL context): not destroyed here.
  static Context borrow(ze_context_handle_t ctx);
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&other) noexcept;
  Context &operator=(Context &&other) noexcept;
  ~Context();

  ze_context_handle_t handle() const { return ctx_; }
  // `device` is required for Device/Shared USM and ignored for Host.
  void *alloc(size_t bytes, MemKind kind, ze_device_handle_t device = nullptr) const;
  void free(void *ptr) const;
  // Physical allocation size of a pointer (proves which memory it landed in).
  uint64_t alloc_size(void *ptr, ze_memory_type_t *type = nullptr) const;

 private:
  ze_context_handle_t ctx_ = nullptr;
  bool owns_ = true;
};

// A command queue plus its immediate command list. Two of these per card
// (compute, copy) is the design's execution model; they carry copies only,
// kernels go through SYCL.
class Queue {
 public:
  Queue() = default;
  Queue(ze_context_handle_t ctx, ze_device_handle_t dev, uint32_t ordinal, const char *label);
  Queue(const Queue &) = delete;
  Queue &operator=(const Queue &) = delete;
  Queue(Queue &&other) noexcept;
  Queue &operator=(Queue &&other) noexcept;
  ~Queue();

  ze_command_queue_handle_t handle() const { return queue_; }
  uint32_t ordinal() const { return ordinal_; }
  const char *label() const { return label_; }

  // A non-blocking submit may still run when the next call arrives: every call first waits
  // for it (the queue has one command list), so non-blocking means "returns before done",
  // not "overlaps with the next call". Several transfers that should run back to back go
  // into one copy_batch.
  void copy(void *dst, const void *src, size_t bytes, bool blocking);
  struct CopyOp {
    void *dst;
    const void *src;
    size_t bytes;
  };
  void copy_batch(const std::vector<CopyOp> &ops, bool blocking = true);
  // `count` copies of the same transfer in ONE submit: separates wire cost from
  // host submit cost when timing small messages.
  void copy_many(void *dst, const void *src, size_t bytes, int count, bool blocking = true);
  void fill(void *dst, uint8_t value, size_t bytes, bool blocking);
  void wait();
  bool pending() const { return pending_; }
  // Non-blocking: true when nothing submitted is still running (then the queue is settled).
  bool done();

  // Timed copies, in microseconds, best of `iters` runs (event-to-event).
  double time_copy(void *dst, const void *src, size_t bytes, int iters = 20);
  double time_fill(void *dst, uint8_t value, size_t bytes, int iters = 20);

 private:
  ze_command_queue_handle_t queue_ = nullptr;
  void settle();
  void submit(bool blocking);

  ze_command_list_handle_t list_ = nullptr;
  uint32_t ordinal_ = 0;
  const char *label_ = "";
  bool pending_ = false;
};

// Queue-group ordinals by capability.
struct QueueFamily {
  int compute = -1;
  int copy = -1;
  uint32_t groups = 0;
};

QueueFamily query_queue_families(ze_device_handle_t dev);

const char *mem_kind_name(ze_memory_type_t type);

// Device-local memory as the driver's sysman reports it (zesMemoryGetState), matched to a
// card by PCI address. Allocation APIs never report free space, and an allocation beyond the
// physical memory is not guaranteed to fail, so placement must read this before trusting a
// budget. ok = false when sysman is unavailable. Under XEN_EMU_CARD both figures are the emulated card's (rt/card.hpp).
struct VramState {
  bool ok = false;
  uint64_t size = 0, free = 0;
};
VramState vram_state(const DeviceInfo &card);

}  // namespace xenithium:l0