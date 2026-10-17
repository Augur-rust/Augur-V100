// Exact output-row reuse. Production admits only RowTile=2 on a frozen full8
// input/gate-up shape set; RowTile=4 remains a component-only rejected variant.
// Each real (row, token) retains the canonical 128-thread K assignment,
// integer dot, scale/FMA order and four-warp reduction. Only activation loads
// are shared across two/four independent output rows in one CTA.
// Native row-reuse admission binds cols5120/tokens8 before reaching this
// specialization. Q8 keeps its existing full K unrolling and reconstructs
// its unaligned code words from aligned loads with a safe halfword tail.
// NVFP4 predecodes
// both 16-value subblocks, then shares each token's Q8_1 scale between them.
// The 8-block launch bound controls register pressure from those live weights.
// K/pair accumulation order is unchanged. Unknown shapes/tails use other kernels.
#pragma once

// Exact ue4m3_scale_to_f32(value)*0.5f, including its zero/0x7f policy.
// Fold only the representable power-of-two scale; no projection reassociation.
__device__ __forceinline__ float exact_half_ue4m3_scale(uint8_t value) {
    if (value == 0 || value == 0x7f) return 0.0f;
    const uint32_t exponent = (static_cast<uint32_t>(value) >> 3) & 0x0fU;
    const uint32_t mantissa = static_cast<uint32_t>(value) & 0x07U;
    if (exponent == 0) return static_cast<float>(mantissa) * 0x1p-10f;
    return __uint_as_float(((exponent + 119U) << 23U) | (mantissa << 20U));
}

// Exact B8 gate/up pair, two output rows: four independent projection
// accumulators share each activation read. The five-CTA register bound is
// measured on SM70; larger row tiles and six-CTA forcing were rejected.
// Both original FP32 output boundaries remain materialized.
// No weight relayout, activation fusion or arithmetic reassociation.
__global__ __launch_bounds__(128, 5)
void quantized_ffn_nvfp4_pair_b8_kernel(
    const uint8_t* gate, const uint8_t* up, const uint8_t* activation,
    float* gate_output, float* up_output, float gate_scale, float up_scale) {
    constexpr uint32_t RowPairs = 2, Outputs = 2 * RowPairs;
    constexpr uint32_t rows = 17408, blocks = 5120 / 64;
    constexpr uint64_t activation_stride = 5120 / 32 * 36;
    const uint32_t row = blockIdx.x * RowPairs;
    const uint32_t thread = threadIdx.x, lane = thread & 31U, warp = thread / 32U;
    float partial[Outputs][8] = {};
    for (uint32_t block = thread / 2U; block < blocks; block += 64U) {
        int2 w0[Outputs][2] = {}, w1[Outputs][2] = {};
        float scale[Outputs][2] = {};
        #pragma unroll
        for (uint32_t pair = 0; pair < 2; ++pair) {
            const uint32_t word = (thread & 1U) * 4U + pair * 2U;
            #pragma unroll
            for (uint32_t r = 0; r < Outputs; ++r) {
                const uint8_t* w = ((r & 1U) == 0 ? gate : up)
                    + (uint64_t(row + r / 2) * blocks + block) * 36U;
                const auto* packed = reinterpret_cast<const uint32_t*>(w + 4U);
                w0[r][pair] = small_batch_e2m1_words(packed[word]);
                w1[r][pair] = small_batch_e2m1_words(packed[word + 1U]);
                scale[r][pair] = exact_half_ue4m3_scale(w[word / 2U]);
            }
        }
        #pragma unroll
        for (uint32_t t = 0; t < 8; ++t) {
            const uint8_t* a = activation + t * activation_stride
                + uint64_t(block * 2U + (thread & 1U)) * 36U;
            const auto* codes = reinterpret_cast<const int32_t*>(a + 4U);
            const float a_scale = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(a)));
            #pragma unroll
            for (uint32_t pair = 0; pair < 2; ++pair) {
                const uint32_t offset = pair * 4U;
                const int32_t a0 = codes[offset], a1 = codes[offset + 2U];
                const int32_t a2 = codes[offset + 1U], a3 = codes[offset + 3U];
                #pragma unroll
                for (uint32_t r = 0; r < Outputs; ++r) {
                    int32_t dot = __dp4a(w0[r][pair].x, a0, 0);
                    dot = __dp4a(w0[r][pair].y, a1, dot);
                    dot = __dp4a(w1[r][pair].x, a2, dot);
                    dot = __dp4a(w1[r][pair].y, a3, dot);
                    const float combined = scale[r][pair] * a_scale;
                    partial[r][t] = fmaf(combined, static_cast<float>(dot), partial[r][t]);
                }
            }
        }
    }
    __shared__ float other_warps[Outputs][8][3][32];
    if (warp > 0U) {
        #pragma unroll
        for (uint32_t r = 0; r < Outputs; ++r) {
            #pragma unroll
            for (uint32_t t = 0; t < 8; ++t) other_warps[r][t][warp - 1U][lane] = partial[r][t];
        }
    }
    __syncthreads();
    if (warp > 0U) return;
    #pragma unroll
    for (uint32_t t = 0; t < 8; ++t) {
        #pragma unroll
        for (uint32_t r = 0; r < Outputs; ++r) {
            #pragma unroll
            for (uint32_t other = 0; other < 3; ++other) partial[r][t] += other_warps[r][t][other][lane];
            #pragma unroll
            for (uint32_t offset = 16; offset != 0; offset >>= 1U)
                partial[r][t] += __shfl_down_sync(0xffffffffU, partial[r][t], offset);
        }
        if (lane == 0U) {
            #pragma unroll
            for (uint32_t rp = 0; rp < RowPairs; ++rp) {
                gate_output[uint64_t(t) * rows + row + rp] = partial[2 * rp][t] * gate_scale;
                up_output[uint64_t(t) * rows + row + rp] = partial[2 * rp + 1][t] * up_scale;
            }
        }
    }
}

// FullRows is set only by admitted full8 shapes with an exact row grid.
// Generic/tail callers retain the original row checks by default.
template<uint32_t Format, uint32_t RowTile, bool FullTokens, bool RowPlanes = false, bool FullRows = false>
__global__ __launch_bounds__(128, Format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1 ? 8 : 1)
void quantized_small_batch_row_reuse_cols5120_kernel(
    const uint8_t* weight, const uint8_t* activation, float* output,
    uint32_t rows, uint32_t cols_arg, uint32_t tokens, float tensor_scale) {
    constexpr uint32_t cols = 5120;
    const uint32_t first_row = blockIdx.x * RowTile;
    const uint32_t thread = threadIdx.x, lane = thread & 31U, warp = thread / 32U;
    const uint64_t activation_stride = static_cast<uint64_t>(cols / 32U) * 36U;
    float partial[RowTile][8] = {};
    if constexpr (Format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1) {
        const uint32_t blocks = cols / 32U;
        #pragma unroll
        for (uint32_t block = thread / 4U; block < blocks; block += 32U) {
            const uint32_t offset = (thread & 3U) * 8U;
            int32_t w0[RowTile] = {}, w1[RowTile] = {};
            float scale[RowTile] = {};
            #pragma unroll
            for (uint32_t r = 0; r < RowTile; ++r) {
                if (!FullRows && first_row + r >= rows) continue;
                if constexpr (RowPlanes) {
                    const auto* base = weight + uint64_t(first_row + r) * blocks * 34U;
                    const auto* code = reinterpret_cast<const int32_t*>(base + block * 32U + offset);
                    w0[r] = code[0]; w1[r] = code[1];
                    scale[r] = __half2float(__ushort_as_half(
                        reinterpret_cast<const uint16_t*>(base + cols)[block]));
                } else {
                const uint8_t* w = weight + (static_cast<uint64_t>(first_row + r) * blocks + block) * 34U;
                const auto* aligned = reinterpret_cast<const uint32_t*>(w + offset + (block & 1U) * 2U);
                const uint32_t lo = aligned[0], hi = aligned[1];
                const uint32_t tail = *reinterpret_cast<const uint16_t*>(w + offset + 8U);
                const uint32_t shift = ((block & 1U) ^ 1U) * 16U;
                w0[r] = static_cast<int32_t>(__funnelshift_r(lo, hi, shift));
                w1[r] = static_cast<int32_t>(__funnelshift_r(hi, tail, shift));
                scale[r] = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(w)));
                }
            }
            #pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                if (!FullTokens && t >= tokens) continue;
                const uint8_t* a = activation + t * activation_stride + static_cast<uint64_t>(block) * 36U;
                const auto* codes = reinterpret_cast<const int32_t*>(a + 4U + offset);
                const int32_t a0 = codes[0], a1 = codes[1];
                const float a_scale = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(a)));
                #pragma unroll
                for (uint32_t r = 0; r < RowTile; ++r) {
                    int32_t dot = __dp4a(w0[r], a0, 0);
                    dot = __dp4a(w1[r], a1, dot);
                    const float combined = scale[r] * a_scale;
                    partial[r][t] = fmaf(combined, static_cast<float>(dot), partial[r][t]);
                }
            }
        }
    } else {
        const uint32_t blocks = cols / 64U;
        for (uint32_t block = thread / 2U; block < blocks; block += 64U) {
            int2 w0[RowTile][2] = {}, w1[RowTile][2] = {};
            float scale[RowTile][2] = {};
            #pragma unroll
            for (uint32_t pair = 0; pair < 2; ++pair) {
                const uint32_t word = (thread & 1U) * 4U + pair * 2U;
                const uint32_t sub = word / 2U;
                #pragma unroll
                for (uint32_t r = 0; r < RowTile; ++r) {
                    if (!FullRows && first_row + r >= rows) continue;
                    const uint8_t* w = weight + (static_cast<uint64_t>(first_row + r) * blocks + block) * 36U;
                    const auto* packed = reinterpret_cast<const uint32_t*>(w + 4U);
                    w0[r][pair] = small_batch_e2m1_words(packed[word]);
                    w1[r][pair] = small_batch_e2m1_words(packed[word + 1U]);
                    scale[r][pair] = ue4m3_scale_to_f32(w[sub]) * 0.5f;
                }
            }
            #pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                if (!FullTokens && t >= tokens) continue;
                const uint8_t* a = activation + t * activation_stride
                    + static_cast<uint64_t>(block * 2U + (thread & 1U)) * 36U;
                const auto* codes = reinterpret_cast<const int32_t*>(a + 4U);
                const float a_scale = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(a)));
                #pragma unroll
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    const uint32_t offset = pair * 4U;
                    const int32_t a0 = codes[offset], a1 = codes[offset + 2U];
                    const int32_t a2 = codes[offset + 1U], a3 = codes[offset + 3U];
                    #pragma unroll
                    for (uint32_t r = 0; r < RowTile; ++r) {
                        int32_t dot = __dp4a(w0[r][pair].x, a0, 0);
                        dot = __dp4a(w0[r][pair].y, a1, dot);
                        dot = __dp4a(w1[r][pair].x, a2, dot);
                        dot = __dp4a(w1[r][pair].y, a3, dot);
                        const float combined = scale[r][pair] * a_scale;
                        partial[r][t] = fmaf(combined, static_cast<float>(dot), partial[r][t]);
                    }
                }
            }
        }
    }
    __shared__ float other_warps[RowTile][8][3][32];
    if (warp > 0U) {
        #pragma unroll
        for (uint32_t r = 0; r < RowTile; ++r) {
            #pragma unroll
            for (uint32_t t = 0; t < 8; ++t) other_warps[r][t][warp - 1U][lane] = partial[r][t];
        }
    }
    __syncthreads();
    if (warp > 0U) return;
    #pragma unroll
    for (uint32_t r = 0; r < RowTile; ++r) {
        if (!FullRows && first_row + r >= rows) continue;
        #pragma unroll
        for (uint32_t t = 0; t < 8; ++t) {
            if (!FullTokens && t >= tokens) continue;
            #pragma unroll
            for (uint32_t other = 0; other < 3; ++other) partial[r][t] += other_warps[r][t][other][lane];
            #pragma unroll
            for (uint32_t offset = 16; offset != 0; offset >>= 1U)
                partial[r][t] += __shfl_down_sync(0xffffffffU, partial[r][t], offset);
            if (lane == 0U) output[static_cast<uint64_t>(t) * rows + first_row + r] =
                Format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1
                    ? partial[r][t] * tensor_scale : partial[r][t];
        }
    }
}
