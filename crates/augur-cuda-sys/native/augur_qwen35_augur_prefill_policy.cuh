// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
// Large Augur input-side contractions only. Do not fuse small gates, K/V,
// down/out projections, B1/B8 verification, or short-prefill epilogues: cuBLAS
// may scale their split partials, changing rounding. No algorithm override.
inline bool augur_prefill_fuse_scale(uint32_t rows, uint32_t cols, uint32_t tokens) {
    return cols == 5120U && tokens >= 2048U && tokens <= 6400U
        && (rows == 17408U || rows == 10240U || rows == 12288U);
}
