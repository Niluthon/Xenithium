#include "xenithium/rt/fabric.hpp"

#include <stdexcept>

#include <sycl/ext/oneapi/backend/level_zero.hpp>

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
  // [hostmail patch] Pinned host USM instead of peer-resident device VRAM. On xe 7.3.x
  // a peer store into another card's VRAM can be partially visible to the owner's
  // spin-read (some lane runs land, others read back zero), so mail_take accepted a
  // half-written message: wrong sums, non-finite drafter activations, CCS page faults.
  // Host USM is coherent behind the system-scope fence in mail_take (cf. PR #2).
  (void)r;
  return alloc_host(bytes);
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
