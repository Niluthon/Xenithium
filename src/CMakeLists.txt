# The engine's libraries, in layer order (a layer links only the ones above it):
#
#   xenithium_util        JSON, HTTP client helpers, image input (PNG / JPEG / data URLs), card specs host
#   xenithium_text        tokenizer, Unicode, Jinja, chat formats + parsers            host
#   xenithium_serve       core (engine interface, batch engine, sampler) + HTTP server host
#   xenithium_eval        log-prob records, the KL gate                                 host
#   xenithium_pack        weight packers + exact host references (src/kernels)          host
#   xenithium_model       checkpoint, config, placement, the arch registry + HOST arch code   host
#   xenithium_rt          Level Zero devices, queues, the two-card fabric               SYCL
#   xenithium_kernels(_xl) the kernels (src/kernels/CMakeLists.txt)                     SYCL
#   xenithium_arch        DEVICE arch code: loaders, each arch's Model                  SYCL
#   xenithium_gpu_engine  SERVE arch code: StepModel adapters + serving backends        SYCL
#   xenithium_quant(_gpu) the offline bake's calibration and AutoRound                  host / SYCL
#
# Archs contribute their sources through src/arch/<id>/sources.cmake (cmake/Xenithium.cmake: xenithium_arch_sources).

set(XENITHIUM_ARCHS common qwen38 qwen35 llama flint CACHE INTERNAL "arch modules under src/arch/")
foreach(arch IN LISTS XENITHIUM_ARCHS)
  include(arch/${arch}/sources.cmake)
endforeach()

# ---- host libraries --------------------------------------------------------------------------------------------------
add_library(xenithium_util STATIC util/json.cpp util/http.cpp util/image.cpp util/image_png.cpp util/image_jpeg.cpp
            util/fetch.cpp util/tls.cpp util/video.cpp util/cards.cpp)  # media URLs (https by the system's OpenSSL, loaded with dlopen)
xenithium_host(xenithium_util)
target_link_libraries(xenithium_util PUBLIC ${CMAKE_DL_LIBS})

add_library(xenithium_text STATIC
  text/tokenizer.cpp text/unicode.cpp text/json_constraint.cpp text/jinja.cpp text/chat_registry.cpp
  text/chat/request.cpp text/chat/template_qwen38.cpp text/chat/template_nanbeige.cpp text/chat/template_flint.cpp
  text/chat/parse_args.cpp text/chat/parser_qwen3.cpp text/chat/parser_glm47.cpp text/chat/formats.cpp)
xenithium_host(xenithium_text)
target_link_libraries(xenithium_text PUBLIC xenithium_util)

add_library(xenithium_serve STATIC core/engine.cpp core/sampler.cpp core/stub_engine.cpp core/batch_engine.cpp server/http.cpp server/server.cpp
            server/responses.cpp)  # /v1/responses
xenithium_host(xenithium_serve)
target_link_libraries(xenithium_serve PUBLIC xenithium_text Threads::Threads)

add_library(xenithium_eval STATIC eval/logprobs.cpp eval/reference.cpp)
xenithium_host(xenithium_eval)
target_link_libraries(xenithium_eval PUBLIC xenithium_util)

add_library(xenithium_model STATIC
  model/checkpoint.cpp model/bulk_read.cpp model/row_reader.cpp model/route_counts.cpp model/route_record.cpp model/placement.cpp model/config.cpp model/registry.cpp model/vision.cpp
  ${XENITHIUM_ARCH_HOST})
xenithium_host(xenithium_model)
target_link_libraries(xenithium_model PUBLIC xenithium_util xenithium_pack Threads::Threads)

# ---- device libraries ------------------------------------------------------------------------------------------------
add_library(xenithium_rt STATIC rt/l0.cpp rt/device.cpp rt/fabric.cpp rt/card.cpp)
xenithium_sycl(xenithium_rt)
target_link_libraries(xenithium_rt PUBLIC xenithium_util Threads::Threads)

add_subdirectory(kernels)

add_library(xenithium_arch STATIC ${XENITHIUM_ARCH_DEVICE} load/dense_store.cpp load/lora.cpp)
xenithium_sycl(xenithium_arch)
target_link_libraries(xenithium_arch PUBLIC xenithium_rt xenithium_model xenithium_kernels)

add_library(xenithium_gpu_engine STATIC ${XENITHIUM_ARCH_SERVE} serve/backend.cpp serve/registry.cpp)
xenithium_sycl(xenithium_gpu_engine)
target_link_libraries(xenithium_gpu_engine PUBLIC xenithium_arch xenithium_serve)

# ---- offline bake ----------------------------------------------------------------------------------------------------
# Calibration mix (native parquet / snappy / JSON readers), AutoRound tuning of the routed experts on the cards and / or
# the CPU, baked engine-native weights + an AutoRound export.
add_library(xenithium_quant STATIC quant/parquet.cpp quant/calib.cpp quant/autoround.cpp quant/export.cpp)
xenithium_host(xenithium_quant)
target_compile_options(xenithium_quant PRIVATE -O3)
target_link_libraries(xenithium_quant PUBLIC xenithium_text Threads::Threads)

# The AutoRound GPU worker (SYCL kernels). Precise fp and IEEE fp32 division on the device (by default device division
# is approximate even under -fp-model=precise): the quantiser's w / s must round
# like the host's, or codes flip at the boundaries.
add_library(xenithium_quant_gpu STATIC quant/autoround_gpu.cpp)
xenithium_sycl(xenithium_quant_gpu)
target_compile_options(xenithium_quant_gpu PRIVATE -O3 -fp-model=precise -foffload-fp32-prec-div)
# AOT compiles the device code when the executable links, and the division precision is chosen there: without this on
# the link line the compile flag above has no effect.
target_link_options(xenithium_quant_gpu INTERFACE -foffload-fp32-prec-div)
target_link_libraries(xenithium_quant_gpu PUBLIC xenithium_quant)
