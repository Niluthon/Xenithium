#!/usr/bin/env bash
# Regression: the two-card mailbox lost its peer payload in 64-byte lane runs, and the sum came out as the owner's
# partial alone with no error and no timeout (2026-11-20).
#
# What was wrong (two defects, both in the wire):
#   1. `mail_take` accepted a message as soon as every word was not the empty pattern. On this host a peer store can be
#      taken as *partly* visible: some 64-byte lane runs read back as zeroes while the rest of the message is there, so
#      the first all-non-empty read handed the caller a partial message. Zero is a legal data word, so nothing noticed:
#      the fp32 two-card sum at 1 row / 64 columns per item (the smallest wire shape) got its peer's contribution as 0
#      and the number came out as the owner's partial, silently.
#   2. The slot an item used depended on the call's shape (`pair_cols` returned 64, 128 or 256 columns per item by the
#      row count). A call of another shape then read a slot a previous call never emptied: those words are already
#      non-empty, so the receiver took that stale message instead of waiting for this call's.
#
# The fix: `mail_take` acknowledges every word (a second read behind a system-scope fence must carry every word and
# agree with the first, re-read up to kTakeConfirmTries times, then error[0] = 2 — loud, never a partial sum), and
# `pair_cols` / `pair_cols_i8` return one 64-column slot for every shape, so a wider exchange is more messages, not
# bigger ones.
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
