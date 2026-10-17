// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
constexpr unsigned AUGUR_PREFILL_COMBINE_SHARED_BYTES = 34816U;
constexpr unsigned AUGUR_PREFILL_NINFER_SHARED_BYTES = 69632U;
// Whole output tiles only: do not alter stream-K seam storage or small queries.
inline bool augur_prefill_compact_combine(uint32_t rows, uint32_t history, uint32_t fixup, bool split2 = false) {
    return rows >= 256U && rows <= 512U && history >= rows && history <= 32768U && (fixup == 0U || (split2 && fixup == 1U && rows == 512U && history >= 1024U));
}

// Bounded 8-warp/64-key reader; history uses the real frontier, not padding.
// Full 262144 capacity uses bounded split3 metadata, including small suffixes and the 128K 480+256 versus 512+224 seam.
inline bool augur_prefill_ninfer_tile(uint32_t rows, uint32_t history) {
    return history <= 262144U && ((rows == 512U && history >= 1024U)
        || ((rows == 128U || rows == 224U || rows == 256U || rows == 480U) && history >= 32768U));
}

// SM70 component sweep: short history avoids a fixup; long history uses
// three uniform partitions. Keep the measured 1024..2047 boundary on split2.
inline unsigned augur_prefill_ninfer_partitions(uint32_t rows, uint32_t history) {
    if (!augur_prefill_ninfer_tile(rows, history)) return 0U;
    // Three uniform partitions fill 2/4 waves on the 72-SM owner for small rows; 224/480 avoid the long serial half accumulator.
    if (rows != 512U) return 3U;
    if (history < 2048U) return 2U;
    return history < 8192U ? 1U : 3U;
}

inline bool augur_prefill_split2(uint32_t rows, uint32_t history) {
    return augur_prefill_ninfer_partitions(rows, history) == 2U;
}
