# syntax=docker/dockerfile:1.7
#
# Xenithium image: builds the engine with the toolchain it is tested with (oneAPI 2026.2 DPC++, oneAPI's oneDNN, Level
# Zero + the Intel GPU user-mode driver with its OpenCL device) and ships a runtime image without the compiler.
#
#   XEN_BUILD_REV=$(git describe --always --dirty) docker compose build    (compose.yaml at the repo root; the usual way)
#   docker build -f docker/Dockerfile --build-arg XEN_BUILD_REV=$(git describe --always --dirty) -t xenithium:beta .
#
# XEN_BUILD_REV: the revision GET /v1/models reports as "build" (the context has no .git: .dockerignore); without it,
# "unknown", and a benchmark page cannot say which revision to check out.
#
# Stages: deps (compiler, oneDNN, build tools: scripts/install-deps.sh) -> build (the engine, then the host test gates)
# -> runtime (SYCL runtime, oneDNN, Level Zero, the GPU driver, ffmpeg, the binaries, scripts/, models/presets/, docs/).
# The host supplies only the kernel driver (xe) and /dev/dri. Layout inside: /opt/xenithium is the project root
# (build/, scripts/, models/, docs/), as on the host, so scripts/start-server.sh and the launcher run unchanged.

ARG BASE=ubuntu:26.04

# ---------------------------------------------------------------- deps
FROM ${BASE} AS deps
ARG ONEAPI_VERSION=2026.2.0-863
WORKDIR /opt/xenithium
COPY docker/keys/ docker/keys/
COPY scripts/install-deps.sh scripts/
RUN ONEAPI_VERSION=${ONEAPI_VERSION} scripts/install-deps.sh && rm -rf /var/lib/apt/lists/*
# No oneCCL, by design (the two-card exchange is the engine's own): fail if anything pulls it in.
RUN if dpkg -l | grep -qi 'intel-oneapi-ccl'; then echo "oneCCL must not be in this image" >&2; exit 1; fi

# ---------------------------------------------------------------- build
FROM deps AS build
ARG XEN_BUILD_REV=
COPY . /opt/xenithium
ENV XEN_ONEAPI=/opt/intel/oneapi/2026.2
# Same configure as scripts/build-host.sh (Release, icpx, AOT for bmg), then the host gates (ctest label host; the gpu
# gates need the cards: run scripts/test.sh on the host or in the container).
RUN bash -c 'set -euo pipefail; source scripts/oneapi-env.sh; \
      cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DXENITHIUM_BUILD_REV="${XEN_BUILD_REV}"; \
      cmake --build build -j "$(nproc)"; \
      ctest --test-dir build -L host --output-on-failure'
# What the runtime image takes: the tools (not the test binaries) and the kernel library they load from build/.
RUN mkdir -p /stage/build && \
    find build -maxdepth 1 -type f \( -name 'xenithium-*' ! -name 'xenithium-test-*' -o -name '*.so' \) \
      -exec cp -a {} /stage/build/ \; && \
    ls /stage/build

# ---------------------------------------------------------------- runtime
FROM ${BASE} AS runtime
# marks this project's containers: the "containers hold the cards" checks skip them (their servers are seen as processes)
LABEL org.xenithium=1
ARG ONEAPI_VERSION=2026.2.0-863
WORKDIR /opt/xenithium
COPY docker/keys/ docker/keys/
COPY scripts/install-deps.sh scripts/
RUN ONEAPI_VERSION=${ONEAPI_VERSION} scripts/install-deps.sh --runtime && rm -rf /var/lib/apt/lists/*
COPY --from=build /stage/build/ build/
COPY scripts/ scripts/
COPY models/presets/ models/presets/
COPY docs/server-env.md docs/launcher.md docs/bench.md docs/
COPY docker/entrypoint.sh docker/
# The docker client (compose.yaml mounts the host's socket): the launcher and start-server.sh check which containers hold
# the cards, as on the host.
COPY --from=docker:29-cli /usr/local/bin/docker /usr/local/bin/docker
ENV XEN_ONEAPI=/opt/intel/oneapi/2026.2 ZE_ENABLE_VALIDATION_LAYER=0
# Every library the binaries and the SYCL Level Zero adapters load must resolve (a missing libumf / libhwloc makes SYCL
# report "no device" while Level Zero works), and the OpenCL ICD oneDNN builds its GPU kernels through must be there.
RUN bash -c 'set -euo pipefail; source scripts/oneapi-env.sh; \
      missing=$(ldd build/xenithium-* build/*.so "$CMPLR_ROOT"/lib/libur_adapter_level_zero*.so.0 | grep "not found" || true); \
      [[ -z "$missing" ]] || { echo "unresolved libraries:"; echo "$missing"; exit 1; }; \
      ls /etc/OpenCL/vendors/intel*.icd; \
      ldd build/xenithium-server | grep -E "dnnl|sycl"'
ENTRYPOINT ["/opt/xenithium/docker/entrypoint.sh"]
CMD ["scripts/start-server.sh"]
