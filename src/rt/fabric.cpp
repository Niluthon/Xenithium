#include "xenithium/rt/fabric.hpp"

#include <charconv>
#include <cstdlib>
#include <cstring>

#include <stdexcept>

#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include "xenithium/kernels/comm/wire.hpp"
#include "xenithium/rt/card.hpp"
#include "xenithium/rt/device.hpp"

namespace xenithium {

sycl::property_list Fabric::queue_props(bool profiling) {
  if (profiling) return sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
  return sycl::property_list{sycl::property::queue::in_order()};
}

Fabric::Fabric(const l0::Loader &loader, std::vector<size_t> ranks, bool profiling) : ranks_(std::move(ranks)), profiling_(profiling) {
  // one device = a single-card deployment (no peer link: alloc_mailbox makes nothing resident elsewhere)
  if (ranks_.empty()) throw std::runtime_error("fabric: no device");
  for (size_t r : ranks_)  // e.g. TP2 on a one-card host: name it, not an out_of_range from the device table
    if (r >= loader.size())
      throw std::runtime_error("fabric: device " + std::to_string(r) + " requested, " + std::to_string(loader.size()) + " visible (" +
                               std::to_string(ranks_.size()) + " card(s) needed)");

  // name every card (and refuse a bad XEN_EMU_CARD) before anything is allocated on it
  for (size_t r : ranks_) rt::log_card(r, loader.info(r));

  for (size_t r : ranks_) {
    devs_.push_back(loader.device(r));
    sdevs_.push_back(find_sycl_device(loader.device(r)));
  }

  // Report the driver's peer-access capability for the device-backed copy
  // buffers. The fused kernel mailboxes are separately allocated in host USM.
  if (devs_.size() > 1) {
    static bool said = false;
    std::string missing;
    size_t no_link = 0, pairs = 0;
    for (size_t i = 0; i < devs_.size(); ++i)
      for (size_t j = 0; j < devs_.size(); ++j) {
        if (i == j) continue;
        ++pairs;
        if (can_access(i, j)) continue;
        ++no_link;
        missing += " " + std::to_string(i) + "->" + std::to_string(j);
      }
    if (!said) {
      said = true;
      if (no_link == 0)
        std::fprintf(stderr, "fabric: %zu cards, peer access on all %zu ordered pairs\n", devs_.size(), pairs);
      else
        std::fprintf(stderr,
                     "fabric: %zu cards, %zu of %zu ordered pairs have no peer access (%s)\n",
                     devs_.size(), no_link, pairs, missing.c_str());
    }
  }

  // One SYCL context over all ranks (the only way a kernel may later touch a
  // peer-written buffer); Level Zero borrows its native handle.
  sctx_ = std::make_unique<sycl::context>(sdevs_);
  ctx_ = l0::Context::borrow(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(*sctx_));

  // One SYCL context per card over that same native context (see card_context in the header: the runtime's fast kernel
  // cache leaks host RAM on every launch when one SYCL context holds both cards). A one-card fabric keeps sctx_.
  const auto native = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(*sctx_);
  cctx_.reserve(sdevs_.size());
  for (size_t i = 0; i < sdevs_.size(); ++i) {
    if (sdevs_.size() == 1) {
      cctx_.push_back(*sctx_);
    } else {
      cctx_.push_back(sycl::make_context<sycl::backend::ext_oneapi_level_zero>(
          {native, {sdevs_[i]}, sycl::ext::oneapi::level_zero::ownership::keep}));
    }
  }

  const sycl::property_list props = queue_props(profiling_);
  compute_.reserve(sdevs_.size());
  copy_.reserve(sdevs_.size());
  for (size_t i = 0; i < sdevs_.size(); ++i) {
    compute_.emplace_back(cctx_[i], sdevs_[i], props);
    const l0::QueueFamily fam = l0::query_queue_families(devs_[i]);
    copy_.emplace_back(ctx_.handle(), devs_[i], static_cast<uint32_t>(fam.copy), "copy");
  }
}

void *Fabric::alloc_mailbox(size_t r, size_t bytes) {
  void *p = alloc(r, bytes);
  for (size_t o = 0; o < devs_.size(); ++o) {
    if (o == r) continue;
    if (zeContextMakeMemoryResident(ctx_.handle(), devs_[o], p, bytes) != ZE_RESULT_SUCCESS) {
      release(p);
      throw std::runtime_error("fabric: zeContextMakeMemoryResident failed for a mailbox");
    }
  }
  return p;
}

void Fabric::alloc_step_copy(kernels::StepCopy &buffers, int rows, int n, bool needs_partial, std::vector<void *> &owned) {
  constexpr size_t C = kernels::kWireCardsMax;
  if (size() != C || rows < 1 || n < 1) throw std::invalid_argument("fabric: step copy requires four cards and a positive shape");
  buffers.stride = static_cast<size_t>(rows) * n;
  constexpr size_t ready_bytes = 2 * C * 64;
  buffers.ready = static_cast<uint32_t *>(alloc_host(ready_bytes));
  owned.push_back(buffers.ready);
  std::memset(buffers.ready, 0, ready_bytes);
  if (const char *setting = std::getenv("XEN_Q38_COPY_SPLIT_ROWS")) {
    const char *end = setting + std::strlen(setting);
    int threshold = 0;
    const auto parsed = std::from_chars(setting, end, threshold);
    if (parsed.ec == std::errc{} && parsed.ptr == end) buffers.split_min_rows = threshold;
  }
  if (buffers.split_min_rows >= static_cast<int>(C) && buffers.split_min_rows <= rows) {
    buffers.shared_ready = static_cast<uint32_t *>(alloc_host(ready_bytes));
    owned.push_back(buffers.shared_ready);
    std::memset(buffers.shared_ready, 0, ready_bytes);
  }

  for (size_t c = 0; c < C; ++c) {
    buffers.recv[c] = static_cast<float *>(alloc_mailbox(c, 2 * C * buffers.stride * sizeof(float)));
    owned.push_back(buffers.recv[c]);
    if (buffers.shared_ready) {
      buffers.shared[c] = static_cast<sycl::ext::oneapi::bfloat16 *>(alloc_mailbox(c, 2 * buffers.stride * sizeof(sycl::ext::oneapi::bfloat16)));
      owned.push_back(buffers.shared[c]);
    }
    buffers.state[c] = static_cast<uint32_t *>(alloc(c, 64));
    owned.push_back(buffers.state[c]);
    compute(c).memset(buffers.state[c], 0, 64).wait();
    if (needs_partial) {
      buffers.partial[c] = static_cast<float *>(alloc(c, buffers.stride * sizeof(float)));
      owned.push_back(buffers.partial[c]);
    }
  }
}

bool Fabric::can_access(size_t from, size_t to) const {
  ze_bool_t can = 0;
  if (zeDeviceCanAccessPeer(ze_device(from), ze_device(to), &can) != ZE_RESULT_SUCCESS) return false;
  return can != 0;
}

uint32_t Fabric::p2p_flags(size_t from, size_t to) const {
  ze_device_p2p_properties_t props{};
  props.stype = ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES;
  if (zeDeviceGetP2PProperties(ze_device(from), ze_device(to), &props) != ZE_RESULT_SUCCESS) return 0;
  return props.flags;
}

}  // namespace xenithium
