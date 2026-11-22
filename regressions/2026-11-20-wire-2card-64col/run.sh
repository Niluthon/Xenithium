#!/usr/bin/env bash
# Regression: the two-card mailbox lost its peer payload in 64-byte lane runs, and the sum came out as the owner's
# partial alone with no error and no timeout (2026-11-20).
#
# A peer store could become partly visible: some 64-byte runs read as zero while the
# rest of the message was present. Zero is valid payload, so checking the empty
# pattern alone accepted a partial message. mail_take now confirms the payload
# with a second read after a system-scope fence and reports persistent disagreement.
#
# This is a fixed-shape test. The wire still chooses its slot width by shape.
#
# The case below is the one that failed first in `xenithium-test-wire`: 2 cards, fp32 wire, 1 row, 64 columns per item,
# 6 calls alternating the parity with new data each call. Before the fix it reported 208..528 wrong elements per run
# (the card that starts its exchange first loses the peer's payload); after it, 0.
#
# Needs two idle cards. Run from anywhere: scripts/oneapi-env.sh, then the probe's wiremin mode.
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source scripts/oneapi-env.sh
scripts/build-host.sh xenithium-probe >/dev/null
exec ./build/xenithium-probe wiremin --rows 1 --n 2560 --calls 6
