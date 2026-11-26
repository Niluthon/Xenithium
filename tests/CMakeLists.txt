# The gates: xenithium_test(NAME [GPU] [TIMEOUT s] LIBS ...) builds build/xenithium-test-NAME from test_NAME.cpp and
# registers it with ctest under the label host or gpu. scripts/test.sh runs `ctest -L host`, then `-L gpu` when no
# other process holds the cards. A new gate = one line here.

# ---- host --------------------------------------------------------------------------------------------------------------
xenithium_test(model LIBS xenithium_model)  # checkpoint parsing (synthetic shards + the real model when present)
xenithium_test(qsa-host LIBS xenithium_pack)
xenithium_test(ple-host LIBS xenithium_model)
xenithium_test(eval LIBS xenithium_eval Threads::Threads)
xenithium_test(tokenizer LIBS xenithium_text)
xenithium_test(chat LIBS xenithium_text)
xenithium_test(jinja LIBS xenithium_text)
target_compile_definitions(xenithium-test-jinja PRIVATE XENITHIUM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
xenithium_test(server LIBS xenithium_serve)
target_compile_definitions(xenithium-test-server PRIVATE XENITHIUM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")  # tests/data/vision
xenithium_test(launch TIMEOUT 120 LIBS xenithium_launch)
target_compile_definitions(xenithium-test-launch PRIVATE XEN_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
xenithium_test(bench LIBS xenithium_bench)  # the benchmark tool: decoders, corpus pin, suites, run file, a local endpoint end to end
xenithium_test(quant-host LIBS xenithium_quant)
xenithium_test(route-record LIBS xenithium_model)  # the hot-expert ranking recorded while serving
xenithium_test(placement LIBS xenithium_model)  # the live expert exchange's policy
xenithium_test(moe-cpu-host LIBS xenithium_pack)  # the decode MoE's CPU path (ISAs bit-equal, reference, pool)
xenithium_test(cards LIBS xenithium_util)  # B60 / B65 support: the card table, identification, XEN_EMU_CARD
xenithium_test(image LIBS xenithium_util)  # data URLs, the PNG / JPEG decoders vs the reference's pixels
target_compile_definitions(xenithium-test-image PRIVATE XENITHIUM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
xenithium_test(fetch LIBS xenithium_util Threads::Threads)  # media URLs (http, https via dlopen'd OpenSSL)
target_compile_definitions(xenithium-test-fetch PRIVATE XENITHIUM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
xenithium_test(video-host LIBS xenithium_model xenithium_text)  # video sampling / sizing / pixels / tokens vs the reference, the ffmpeg frame source
target_compile_definitions(xenithium-test-video-host PRIVATE XENITHIUM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
xenithium_test(vision-host LIBS xenithium_model)  # the image processor (resize, patches, tables) vs the reference
target_compile_definitions(xenithium-test-vision-host PRIVATE XENITHIUM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# ---- gpu ---------------------------------------------------------------------------------------------------------------
xenithium_test(l0 GPU TIMEOUT 120 LIBS xenithium_rt)
xenithium_test(kernels GPU LIBS xenithium_kernels)
xenithium_test(w4kn GPU LIBS xenithium_kernels)  # onednn-refactor: the kn-layout decode GEMM vs w4a16_gemm_rows
xenithium_test(head GPU LIBS xenithium_kernels)
xenithium_test(moe GPU LIBS xenithium_kernels)
xenithium_test(gdn GPU LIBS xenithium_kernels)
xenithium_test(qsa GPU LIBS xenithium_kernels)
# Real trained weights and both reserved B60s are required; not a default ctest.
add_executable(xenithium-check-trained-qsa-storage check_trained_qsa_storage.cpp)
xenithium_sycl(xenithium-check-trained-qsa-storage)
target_link_libraries(xenithium-check-trained-qsa-storage PRIVATE xenithium_arch)
# XRV replay of the trained QSA decode score stage: the FMA reference vs the XMX scorer (XEN_QSA_XMX_SCORE) on fixed inputs,
# timing + top-512 identity against an fp64 CPU reference (exit 1 on a selection mismatch beyond near ties). Not a default ctest.
add_executable(xenithium-xrv-qsa-trained-score xrv_qsa_trained_score.cpp)
xenithium_sycl(xenithium-xrv-qsa-trained-score)
target_link_libraries(xenithium-xrv-qsa-trained-score PRIVATE xenithium_kernels)
xenithium_test(gqa GPU LIBS xenithium_kernels)
xenithium_test(gqa128 GPU TIMEOUT 600 LIBS xenithium_kernels)
xenithium_test(gqa-flint GPU TIMEOUT 600 LIBS xenithium_kernels)
xenithium_test(moe-aff GPU TIMEOUT 600 LIBS xenithium_kernels)
xenithium_test(dflash GPU LIBS xenithium_kernels)
xenithium_test(ple GPU LIBS xenithium_kernels)
xenithium_test(lora GPU LIBS xenithium_arch)  # the PEFT adapter: config / name / shape rejects, load + lora_apply vs the host reference, zero-adapter invariance
xenithium_test(wire GPU LIBS xenithium_kernels)  # the two-card sums' bits for every row count and slot width
xenithium_test(rows16 GPU LIBS xenithium_kernels)  # the 16-row kernel shape keeps every row's bits
xenithium_test(vision GPU TIMEOUT 900 LIBS xenithium_arch)  # the encoder kernels + the vision tower vs the reference
target_compile_definitions(xenithium-test-vision PRIVATE XENITHIUM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
xenithium_test(vision-model GPU TIMEOUT 900 LIBS xenithium_arch xenithium_text)  # the 27B's M-RoPE (decode == prefill after images, snapshots)
target_compile_definitions(xenithium-test-vision-model PRIVATE XENITHIUM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
xenithium_test(autoround GPU LIBS xenithium_quant_gpu xenithium_quant)
