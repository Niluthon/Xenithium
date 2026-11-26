#include "xenithium/rt/l0.hpp"

#include <level_zero/zes_api.h>

#include "xenithium/rt/card.hpp"

#include <mutex>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace xenithium::l0 {
namespace {

// Every USM allocation is 4 KiB aligned: ESIMD block loads read up to 512 B at
// a time and a 128 B-aligned buffer can fault the card.
constexpr size_t kAllocAlign = 4096;

std::string trim(const char *s) {
  std::string str(s ? s : "");
  while (!str.empty() && str.back() == '\0') str.pop_back();
  return str;
}

}  // namespace

const char *result_name(ze_result_t rc) {
  switch (rc) {
    case ZE_RESULT_SUCCESS: return "ZE_RESULT_SUCCESS";
    case ZE_RESULT_NOT_READY: return "ZE_RESULT_NOT_READY";
    case ZE_RESULT_ERROR_DEVICE_LOST: return "ZE_RESULT_ERROR_DEVICE_LOST";
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
    case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
    case ZE_RESULT_ERROR_UNSUPPORTED_VERSION: return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
    case ZE_RESULT_ERROR_UNINITIALIZED: return "ZE_RESULT_ERROR_UNINITIALIZED";
    case ZE_RESULT_ERROR_INVALID_NULL_HANDLE: return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
    case ZE_RESULT_ERROR_INVALID_ENUMERATION: return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
    case ZE_RESULT_ERROR_NOT_AVAILABLE: return "ZE_RESULT_ERROR_NOT_AVAILABLE";
    case ZE_RESULT_ERROR_UNSUPPORTED_SIZE: return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
    default: return "ZE_RESULT_ERROR_<other>";
  }
}

void throw_error(ze_result_t rc, const char *what) {
  throw std::runtime_error(std::string("level zero: ") + what + ": " + result_name(rc));
}

const char *mem_kind_name(ze_memory_type_t type) {
  switch (type) {
    case ZE_MEMORY_TYPE_HOST: return "host";
    case ZE_MEMORY_TYPE_DEVICE: return "device";
    case ZE_MEMORY_TYPE_SHARED: return "shared";
    case ZE_MEMORY_TYPE_UNKNOWN: return "unknown";
    default: return "?";
  }
}

std::string DeviceInfo::pci_bdf() const {
  if (!has_pci) return "-";
  char buf[32];
  std::snprintf(buf, sizeof buf, "%04x:%02x:%02x.%x", pci_domain, pci_bus, pci_device, pci_function);
  return buf;
}

Loader Loader::open() {
  Loader self;
  const ze_result_t init = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  if (init != ZE_RESULT_SUCCESS) throw_error(init, "zeInit");

  uint32_t driver_count = 0;
  check(zeDriverGet(&driver_count, nullptr), "zeDriverGet(count)");
  self.drivers_.resize(driver_count);
  if (driver_count) check(zeDriverGet(&driver_count, self.drivers_.data()), "zeDriverGet");

  for (auto driver : self.drivers_) {
    uint32_t device_count = 0;
    check(zeDeviceGet(driver, &device_count, nullptr), "zeDeviceGet(count)");
    std::vector<ze_device_handle_t> devices(device_count);
    if (device_count) check(zeDeviceGet(driver, &device_count, devices.data()), "zeDeviceGet");
    for (auto dev : devices) {
      self.devices_.push_back(dev);
      self.device_driver_.push_back(driver);
    }
  }
  return self;
}

DeviceInfo Loader::info(size_t i) const {
  ze_device_properties_t props{};
  props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  check(zeDeviceGetProperties(device(i), &props), "zeDeviceGetProperties");

  DeviceInfo out;
  out.name = trim(props.name);
  out.vendor_id = props.vendorId;
  out.device_id = props.deviceId;
  out.vram_bytes = 0;  // filled from the memory modules below
  out.slices = props.numSlices;
  out.subslices_per_slice = props.numSubslicesPerSlice;
  out.eu_per_subslice = props.numEUsPerSubslice;
  out.threads_per_eu = props.numThreadsPerEU;
  out.clock_mhz = props.coreClockRate;
  out.integrated = (props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0;

  uint32_t mem_count = 0;
  check(zeDeviceGetMemoryProperties(device(i), &mem_count, nullptr), "zeDeviceGetMemoryProperties(count)");
  if (mem_count) {
    std::vector<ze_device_memory_properties_t> mems(mem_count);
    for (auto &m : mems) {
      m.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
      m.pNext = nullptr;
    }
    check(zeDeviceGetMemoryProperties(device(i), &mem_count, mems.data()), "zeDeviceGetMemoryProperties");
    // Sum the local memory modules (the B70 exposes one 32 GiB module per card).
    uint64_t total = 0;
    for (uint32_t m = 0; m < mem_count; ++m) total += mems[m].totalSize;
    if (total) out.vram_bytes = total;
  }

  ze_pci_ext_properties_t pci{};
  pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
  if (zeDevicePciGetPropertiesExt(device(i), &pci) == ZE_RESULT_SUCCESS) {
    out.pci_domain = pci.address.domain;
    out.pci_bus = pci.address.bus;
    out.pci_device = pci.address.device;
    out.pci_function = pci.address.function;
    out.has_pci = true;
  }
  return out;
}

bool Loader::can_access_peer(size_t i, size_t j) const {
  ze_bool_t can = 0;
  if (zeDeviceCanAccessPeer(device(i), device(j), &can) != ZE_RESULT_SUCCESS) return false;
  return can != 0;
}

std::string Loader::api_version() const {
  ze_api_version_t version{};
  if (zeDriverGetApiVersion(drivers_.empty() ? nullptr : drivers_.front(), &version) != ZE_RESULT_SUCCESS)
    return "?";
  char buf[32];
  std::snprintf(buf, sizeof buf, "%u.%u", ZE_MAJOR_VERSION(version), ZE_MINOR_VERSION(version));
  return buf;
}

uint32_t Loader::driver_version(size_t i) const {
  ze_driver_properties_t props{};
  props.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
  if (zeDriverGetProperties(driver_of(i), &props) != ZE_RESULT_SUCCESS) return 0;
  return props.driverVersion;
}

std::string Loader::driver_version_text(size_t i) const {
  // Intel's driver extension ZE_intel_get_driver_version_string: the Level Zero driver's version ("1.17.39395+16"); not in
  // the loader's headers, so looked up by name. Its signature: (driver, char *version, size_t *size); a null version
  // asks for the size.
  using GetVersionString = ze_result_t (*)(ze_driver_handle_t, char *, size_t *);
  void *fn = nullptr;
  if (zeDriverGetExtensionFunctionAddress(driver_of(i), "zeIntelGetDriverVersionString", &fn) == ZE_RESULT_SUCCESS && fn) {
    const auto get = reinterpret_cast<GetVersionString>(fn);
    size_t size = 0;
    if (get(driver_of(i), nullptr, &size) == ZE_RESULT_SUCCESS && size > 0 && size < 256) {
      std::string text(size, '\0');
      if (get(driver_of(i), text.data(), &size) == ZE_RESULT_SUCCESS) {
        text.resize(std::strlen(text.c_str()));
        if (!text.empty()) return text;
      }
    }
  }
  // else the build number, the one part of driverVersion every Intel driver fills the same way (low 16 bits)
  const uint32_t v = driver_version(i);
  return v ? "build " + std::to_string(v & 0xffffu) : std::string();
}

Context::Context(ze_driver_handle_t driver, const std::vector<ze_device_handle_t> &devices) {
  ze_context_desc_t desc{};
  desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  check(zeContextCreate(driver, &desc, &ctx_), "zeContextCreate");
}

Context Context::borrow(ze_context_handle_t ctx) {
  Context self;
  self.ctx_ = ctx;
  self.owns_ = false;
  return self;
}

Context::Context(Context &&other) noexcept : ctx_(other.ctx_), owns_(other.owns_) { other.ctx_ = nullptr; }

Context &Context::operator=(Context &&other) noexcept {
  if (this != &other) {
    if (ctx_ && owns_) zeContextDestroy(ctx_);
    ctx_ = other.ctx_;
    owns_ = other.owns_;
    other.ctx_ = nullptr;
  }
  return *this;
}

Context::~Context() {
  if (ctx_ && owns_) zeContextDestroy(ctx_);
}

void *Context::alloc(size_t bytes, MemKind kind, ze_device_handle_t device) const {
  ze_device_mem_alloc_desc_t device_desc{};
  device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  ze_host_mem_alloc_desc_t host_desc{};
  host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;

  void *ptr = nullptr;
  ze_result_t rc = ZE_RESULT_ERROR_UNINITIALIZED;
  switch (kind) {
    case MemKind::Device:
      rc = zeMemAllocDevice(ctx_, &device_desc, bytes, kAllocAlign, device, &ptr);
      break;
    case MemKind::Shared:
      rc = zeMemAllocShared(ctx_, &device_desc, &host_desc, bytes, kAllocAlign, device, &ptr);
      break;
    case MemKind::Host:
      rc = zeMemAllocHost(ctx_, &host_desc, bytes, kAllocAlign, &ptr);
      break;
    case MemKind::HostUncached:
      host_desc.flags = ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED;
      rc = zeMemAllocHost(ctx_, &host_desc, bytes, kAllocAlign, &ptr);
      break;
    case MemKind::HostWC:
      host_desc.flags = ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
      rc = zeMemAllocHost(ctx_, &host_desc, bytes, kAllocAlign, &ptr);
      break;
  }
  check(rc, "zeMemAlloc");
  if (!ptr) throw std::runtime_error("level zero: zeMemAlloc returned null");
  return ptr;
}

void Context::free(void *ptr) const {
  if (ptr) check(zeMemFree(ctx_, ptr), "zeMemFree");
}

uint64_t Context::alloc_size(void *ptr, ze_memory_type_t *type) const {
  ze_memory_allocation_properties_t info{};
  info.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
  check(zeMemGetAllocProperties(ctx_, ptr, &info, nullptr), "zeMemGetAllocProperties");
  if (type) *type = info.type;
  void *base = nullptr;
  size_t size = 0;
  check(zeMemGetAddressRange(ctx_, ptr, &base, &size), "zeMemGetAddressRange");
  return size;
}

Queue::Queue(ze_context_handle_t ctx, ze_device_handle_t dev, uint32_t ordinal, const char *label)
    : ordinal_(ordinal), label_(label) {
  ze_command_queue_desc_t qdesc{};
  qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  qdesc.ordinal = ordinal;
  qdesc.index = 0;
  qdesc.flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY;  // we append + submit ourselves
  qdesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  qdesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  check(zeCommandQueueCreate(ctx, dev, &qdesc, &queue_), "zeCommandQueueCreate");

  ze_command_list_desc_t ldesc{};
  ldesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  ldesc.commandQueueGroupOrdinal = ordinal;
  check(zeCommandListCreate(ctx, dev, &ldesc, &list_), "zeCommandListCreate");
}

Queue::Queue(Queue &&other) noexcept
    : queue_(other.queue_), list_(other.list_), ordinal_(other.ordinal_), label_(other.label_), pending_(other.pending_) {
  other.pending_ = false;
  other.queue_ = nullptr;
  other.list_ = nullptr;
}

Queue &Queue::operator=(Queue &&other) noexcept {
  if (this != &other) {
    // Same order as the destructor: the list before the queue that executes it.
    if (pending_ && queue_) zeCommandQueueSynchronize(queue_, UINT64_MAX);
    if (list_) zeCommandListDestroy(list_);
    if (queue_) zeCommandQueueDestroy(queue_);
    queue_ = other.queue_;
    list_ = other.list_;
    ordinal_ = other.ordinal_;
    label_ = other.label_;
    pending_ = other.pending_;
    other.pending_ = false;
    other.queue_ = nullptr;
    other.list_ = nullptr;
  }
  return *this;
}

Queue::~Queue() {
  if (pending_ && queue_) zeCommandQueueSynchronize(queue_, UINT64_MAX);
  if (list_) zeCommandListDestroy(list_);
  if (queue_) zeCommandQueueDestroy(queue_);
}

// The queue owns ONE command list: resetting it while a previous non-blocking submit still
// runs drops that work. Every reset therefore first waits for the last submit; batches go
// through copy_batch.
void Queue::settle() {
  if (pending_) {
    check(zeCommandQueueSynchronize(queue_, UINT64_MAX), "zeCommandQueueSynchronize");
    pending_ = false;
  }
}

bool Queue::done() {
  if (!pending_) return true;
  const ze_result_t r = zeCommandQueueSynchronize(queue_, 0);
  if (r == ZE_RESULT_NOT_READY) return false;
  check(r, "zeCommandQueueSynchronize");
  pending_ = false;
  return true;
}

void Queue::submit(bool blocking) {
  check(zeCommandListClose(list_), "zeCommandListClose");
  check(zeCommandQueueExecuteCommandLists(queue_, 1, &list_, nullptr), "zeCommandQueueExecuteCommandLists");
  pending_ = true;
  if (blocking) settle();
}

void Queue::copy_batch(const std::vector<CopyOp> &ops, bool blocking) {
  settle();
  check(zeCommandListReset(list_), "zeCommandListReset");
  for (const auto &op : ops)
    check(zeCommandListAppendMemoryCopy(list_, op.dst, op.src, op.bytes, nullptr, 0, nullptr), "zeCommandListAppendMemoryCopy");
  submit(blocking);
}

void Queue::copy(void *dst, const void *src, size_t bytes, bool blocking) {
  settle();
  check(zeCommandListReset(list_), "zeCommandListReset");
  check(zeCommandListAppendMemoryCopy(list_, dst, src, bytes, nullptr, 0, nullptr),
        "zeCommandListAppendMemoryCopy");
  submit(blocking);
}

void Queue::copy_many(void *dst, const void *src, size_t bytes, int count, bool blocking) {
  settle();
  check(zeCommandListReset(list_), "zeCommandListReset");
  for (int i = 0; i < count; ++i) {
    check(zeCommandListAppendMemoryCopy(list_, dst, src, bytes, nullptr, 0, nullptr),
          "zeCommandListAppendMemoryCopy");
  }
  submit(blocking);
}

void Queue::fill(void *dst, uint8_t value, size_t bytes, bool blocking) {
  settle();
  check(zeCommandListReset(list_), "zeCommandListReset");
  check(zeCommandListAppendMemoryFill(list_, dst, &value, 1, bytes, nullptr, 0, nullptr),
        "zeCommandListAppendMemoryFill");
  submit(blocking);
}

void Queue::wait() {
  check(zeCommandQueueSynchronize(queue_, UINT64_MAX), "zeCommandQueueSynchronize");
  pending_ = false;
}

double Queue::time_copy(void *dst, const void *src, size_t bytes, int iters) {
  // Level Zero reports kernel timestamps on events, not copy timestamps, so a
  // copy is timed host-side around submit+sync. At the sizes that matter here
  // (>= 64 MiB) the submit overhead is negligible beside the transfer.
  auto once = [&]() -> double {
    const auto t0 = std::chrono::steady_clock::now();
    copy(dst, src, bytes, /*blocking=*/true);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
  };
  (void)once();  // warm-up: the first submit pays driver/list setup
  double best = 1e30;
  for (int i = 0; i < iters; ++i) best = std::min(best, once());
  return best;
}

double Queue::time_fill(void *dst, uint8_t value, size_t bytes, int iters) {
  auto once = [&]() -> double {
    const auto t0 = std::chrono::steady_clock::now();
    fill(dst, value, bytes, /*blocking=*/true);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
  };
  (void)once();
  double best = 1e30;
  for (int i = 0; i < iters; ++i) best = std::min(best, once());
  return best;
}

QueueFamily query_queue_families(ze_device_handle_t dev) {
  QueueFamily fam;
  uint32_t count = 0;
  if (zeDeviceGetCommandQueueGroupProperties(dev, &count, nullptr) != ZE_RESULT_SUCCESS) return fam;
  std::vector<ze_command_queue_group_properties_t> props(count);
  for (auto &p : props) p.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  if (zeDeviceGetCommandQueueGroupProperties(dev, &count, props.data()) != ZE_RESULT_SUCCESS) return fam;
  fam.groups = count;
  for (uint32_t i = 0; i < count; ++i) {
    const bool compute = (props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0;
    const bool copy = (props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) != 0;
    if (compute && fam.compute < 0) fam.compute = static_cast<int>(i);
    if (copy && !compute && fam.copy < 0) fam.copy = static_cast<int>(i);
  }
  if (fam.copy < 0) fam.copy = fam.compute;
  if (fam.compute < 0) fam.compute = 0;
  return fam;
}

VramState vram_state(const DeviceInfo &card) {
  VramState out;
  static std::once_flag once;
  static bool sysman = false;
  std::call_once(once, [] { sysman = zesInit(0) == ZE_RESULT_SUCCESS; });
  if (!sysman || !card.has_pci) return out;
  uint32_t nd = 0;
  if (zesDriverGet(&nd, nullptr) != ZE_RESULT_SUCCESS) return out;
  std::vector<zes_driver_handle_t> drivers(nd);
  zesDriverGet(&nd, drivers.data());
  for (auto drv : drivers) {
    uint32_t n = 0;
    zesDeviceGet(drv, &n, nullptr);
    std::vector<zes_device_handle_t> devs(n);
    zesDeviceGet(drv, &n, devs.data());
    for (auto d : devs) {
      zes_pci_properties_t pci{};
      pci.stype = ZES_STRUCTURE_TYPE_PCI_PROPERTIES;
      if (zesDevicePciGetProperties(d, &pci) != ZE_RESULT_SUCCESS) continue;
      if (pci.address.domain != card.pci_domain || pci.address.bus != card.pci_bus || pci.address.device != card.pci_device ||
          pci.address.function != card.pci_function)
        continue;
      uint32_t nm = 0;
      zesDeviceEnumMemoryModules(d, &nm, nullptr);
      std::vector<zes_mem_handle_t> mems(nm);
      zesDeviceEnumMemoryModules(d, &nm, mems.data());
      for (auto m : mems) {
        zes_mem_properties_t props{};
        props.stype = ZES_STRUCTURE_TYPE_MEM_PROPERTIES;
        zes_mem_state_t st{};
        st.stype = ZES_STRUCTURE_TYPE_MEM_STATE;
        if (zesMemoryGetProperties(m, &props) != ZE_RESULT_SUCCESS || props.location != ZES_MEM_LOC_DEVICE) continue;
        if (zesMemoryGetState(m, &st) != ZE_RESULT_SUCCESS) continue;
        out.size += st.size;
        out.free += st.free;
        out.ok = true;
      }
      // XEN_EMU_CARD (rt/card.hpp): an emulated card with less memory hides the difference from every budget
      if (const uint64_t cut = rt::emulated_vram_cut(card); out.ok && cut) {
        out.size = out.size > cut ? out.size - cut : 0;
        out.free = out.free > cut ? out.free - cut : 0;
      }
      return out;
    }
  }
  return out;
}

}  // namespace xenithium:l0
