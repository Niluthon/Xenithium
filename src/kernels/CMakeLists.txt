# Kernels, one block per family (the directory under src/kernels/ and include/xenithium/kernels/). A new kernel = its
# .cpp in the family's list: SOURCES (128 GRF), XL (only kernels that spill at 128 GRF) or REF (exact host references).
#
# Why two device libraries: the grf_size<256> kernel property does not reach ESIMD kernels under AOT (every kernel
# compiled at 128 GRF), so the 256-GRF kernels live in a shared library whose device image is compiled with
# -doubleGRF; the static kernels keep 128 GRF. Only kernels that spill at 128 belong in XL: at 256 a kernel gets half
# the threads per EU, which slows a kernel that does not spill.

xenithium_kernel_family(gemm
  SOURCES int8_gemm.cpp w4a16_gemm.cpp dense.cpp pf_dense.cpp mlp.cpp w4kn_rows.cpp
  XL      w4a16_rows.cpp onednn_pf.cpp)
xenithium_kernel_family(attention
  SOURCES gqa.cpp gqa_sel.cpp qsa.cpp kv_copy.cpp
  XL      gqa_rows.cpp gqa_staged.cpp gqa_fa.cpp gqa_seg.cpp gqa_sdpa.cpp
  REF     qsa_ref.cpp)
xenithium_kernel_family(linear_attention
  SOURCES gdn.cpp gdn_rows.cpp gdn_core_v2.cpp
  XL      gdn_scan.cpp gdn_core8.cpp gdn_core16.cpp)
xenithium_kernel_family(moe
  SOURCES moe.cpp moe_rows.cpp moe_aff.cpp
  XL      moe_rows_gu.cpp moe_aff_rows.cpp
  REF     moe_cpu.cpp)
xenithium_kernel_family(residual
  SOURCES rmsnorm.cpp hc.cpp hc_fused.cpp)
xenithium_kernel_family(embed
  SOURCES embed_aff.cpp ple.cpp
  REF     ple_ref.cpp)
xenithium_kernel_family(head
  SOURCES head.cpp logprob.cpp)
xenithium_kernel_family(drafter
  SOURCES mtp.cpp dflash.cpp dspine.cpp)
xenithium_kernel_family(comm
  SOURCES wire.cpp)
xenithium_kernel_family(encoder
  SOURCES vit.cpp
  XL      vit_onednn.cpp
  REF     vit_ref.cpp)
# the vision encoder's own kernels keep the header's rounding points on the card (no contraction / reassociation)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/encoder/vit.cpp PROPERTIES COMPILE_OPTIONS "-fp-model=precise")

# Weight packers and the exact CPU references: host code shared by kernels, loaders and tests.
add_library(xenithium_pack STATIC pack.cpp ${XENITHIUM_KERNELS_REF})
xenithium_host(xenithium_pack)
target_link_libraries(xenithium_pack PUBLIC Threads::Threads)  # the MoE CPU path's pool (moe/moe_cpu.cpp)

add_library(xenithium_kernels_xl SHARED ${XENITHIUM_KERNELS_XL})
target_compile_options(xenithium_kernels_xl PRIVATE ${XENITHIUM_SYCL_FLAGS})
target_link_options(xenithium_kernels_xl PRIVATE -fsycl -fsycl-max-parallel-link-jobs=${XENITHIUM_LINK_JOBS} "-fsycl-targets=${XENITHIUM_SYCL_TARGET}"
                    "-Xsycl-target-backend=${XENITHIUM_SYCL_TARGET}" "-device ${XENITHIUM_SYCL_DEVICE} -options -doubleGRF")
target_include_directories(xenithium_kernels_xl PUBLIC ${PROJECT_SOURCE_DIR}/include PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(xenithium_kernels STATIC ${XENITHIUM_KERNELS_SOURCES})
xenithium_sycl(xenithium_kernels)
# the families' private device headers (e.g. "gemm/w4a16_c32.hpp") are included relative to src/kernels
target_include_directories(xenithium_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(xenithium_kernels PUBLIC xenithium_rt xenithium_pack xenithium_kernels_xl)
# onednn-refactor: the W4A8 prefill GEMMs on oneDNN (gemm/onednn_pf.cpp in the XL library beside w4a16_rows.cpp, its
# caller; dnnl from $DNNLROOT when scripts/oneapi-env.sh got XEN_DNNL_ROOT, else oneAPI's)
find_library(XENITHIUM_DNNL dnnl HINTS $ENV{DNNLROOT}/lib NO_DEFAULT_PATH)  # $DNNLROOT first: oneapi-vars' CMAKE_PREFIX_PATH beats HINTS
find_library(XENITHIUM_DNNL dnnl HINTS /opt/intel/oneapi/dnnl/latest/lib REQUIRED)
get_filename_component(XENITHIUM_DNNL_DIR ${XENITHIUM_DNNL} DIRECTORY)
find_path(XENITHIUM_DNNL_INCLUDE oneapi/dnnl/dnnl.hpp HINTS ${XENITHIUM_DNNL_DIR}/../include NO_DEFAULT_PATH REQUIRED)
target_link_libraries(xenithium_kernels_xl PUBLIC ${XENITHIUM_DNNL})
# BEFORE: the headers must match the linked lib, not the oneAPI copy on CPATH
target_include_directories(xenithium_kernels_xl BEFORE PRIVATE ${XENITHIUM_DNNL_INCLUDE} ${XENITHIUM_DNNL_INCLUDE}/oneapi/dnnl)
