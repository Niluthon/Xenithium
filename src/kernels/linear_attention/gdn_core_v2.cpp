// Xenithium · GDN decode core, 8-row shapes at 4 value rows per item in the 128-GRF library (two work-groups per Xe core, the default; XEN_GDN_CORE128=0 off).

#include "linear_attention/gdn_core.hpp"

namespace xenithium::kernels::gdn_detail {

template <bool SWISH>
sycl::event core_launch8_v2(sycl::queue &q, const GdnShape &s, int rows, const bf16 *x, const GdnWeights &w, const GdnState &st, int parity,
                            const float *ystep, bf16 *core_out) {
  return core_launch_km<SWISH, kGdnMaxRows, kVR, 1>(q, s, rows, x, w, st, parity, ystep, core_out);
}
template <bool SWISH>
sycl::event core_launch8_tree_v2(sycl::queue &q, const GdnShape &s, int rows, const bf16 *x, const GdnWeights &w, const GdnState &st, int parity,
                                 const float *ystep, bf16 *core_out) {
  return core_abi<SWISH, kGdnMaxRows, kGdnMaxRows, kGdnMaxRows, true, kVR, 1>(q, s, rows, x, w, st, parity, ystep, core_out);
}
#define XEN_GDN_V2_INST(name, sw) \
  template sycl::event name<sw>(sycl::queue &, const GdnShape &, int, const bf16 *, const GdnWeights &, const GdnState &, int, const float *, bf16 *);
XEN_GDN_V2_INST(core_launch8_v2, true)
XEN_GDN_V2_INST(core_launch8_v2, false)
XEN_GDN_V2_INST(core_launch8_tree_v2, true)
XEN_GDN_V2_INST(core_launch8_tree_v2, false)

}  // namespace xenithium::kernels::gdn_detail
