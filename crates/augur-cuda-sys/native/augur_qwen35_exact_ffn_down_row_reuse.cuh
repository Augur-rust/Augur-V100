// Exact B8 FFN-down: reuse activation reads across four Q8 output rows or
// two NVFP4 rows. Wider NVFP4 reuse regressed and is deliberately not selected.
// Preserve original 128-thread K assignment, explicit FP32 FMA order and reduction.
// No new workspace, activation quantization or precision policy.
// The native admission requires rows5120/cols17408/tokens8 and the exact grid.
// No partial row CTA is reachable. NVFP4 scale bytes are read as one aligned
// word. Four warps finalize disjoint tokens; each output keeps the original
// warp0+warp1+warp2+warp3 additions and lane16/8/4/2/1 reduction tree.
#pragma once

template<uint32_t Format>
__global__ __launch_bounds__(128, Format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1 ? 1 : 2)
void quantized_ffn_down_row2_b8_kernel(
    const uint8_t* weight, const uint8_t* activation, float* output,
    uint32_t rows, uint32_t cols_arg, uint32_t tokens, float tensor_scale) {
    constexpr uint32_t RowTile = Format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1 ? 4 : 2;
    constexpr uint32_t TokenTile = 8, cols = 17408;
    constexpr bool FullTokens = true, RowPlanes = false;
    const uint32_t first_row = blockIdx.x * RowTile; const uint32_t first_token = blockIdx.y * TokenTile;
    const uint32_t thread = threadIdx.x, lane = thread & 31U, warp = thread / 32U;
    const uint64_t activation_stride = static_cast<uint64_t>(cols / 32U) * 36U;
    float partial[RowTile][TokenTile] = {};
    if constexpr (Format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1) {
        const uint32_t blocks = cols / 32U;
        #pragma unroll
        for (uint32_t block = thread / 4U; block < blocks; block += 32U) {
            const uint32_t offset = (thread & 3U) * 8U;
            int32_t w0[RowTile] = {}, w1[RowTile] = {};
            float scale[RowTile] = {};
            #pragma unroll
            for (uint32_t r = 0; r < RowTile; ++r) {
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
            for (uint32_t t = 0; t < TokenTile; ++t) {
                if (!FullTokens && t >= tokens) continue;
                const uint8_t* a = activation + (first_token + t) * activation_stride + static_cast<uint64_t>(block) * 36U;
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
            uint32_t scale_words[RowTile] = {};
            #pragma unroll
            for (uint32_t r = 0; r < RowTile; ++r) {
                const uint8_t* w = weight + (static_cast<uint64_t>(first_row + r) * blocks + block) * 36U;
                scale_words[r] = *reinterpret_cast<const uint32_t*>(w);
            }
            #pragma unroll
            for (uint32_t pair = 0; pair < 2; ++pair) {
                const uint32_t word = (thread & 1U) * 4U + pair * 2U;
                const uint32_t sub = word / 2U;
                #pragma unroll
                for (uint32_t r = 0; r < RowTile; ++r) {
                    const uint8_t* w = weight + (static_cast<uint64_t>(first_row + r) * blocks + block) * 36U;
                    const auto* packed = reinterpret_cast<const uint32_t*>(w + 4U);
                    w0[r][pair] = small_batch_e2m1_words(packed[word]);
                    w1[r][pair] = small_batch_e2m1_words(packed[word + 1U]);
                    scale[r][pair] = ue4m3_scale_to_f32(static_cast<uint8_t>(scale_words[r] >> (sub * 8U))) * 0.5f;
                }
            }
            #pragma unroll
            for (uint32_t t = 0; t < TokenTile; ++t) {
                if (!FullTokens && t >= tokens) continue;
                const uint8_t* a = activation + (first_token + t) * activation_stride
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
    __shared__ float all_warps[RowTile][TokenTile][4][32];
    #pragma unroll
    for (uint32_t r = 0; r < RowTile; ++r) {
        #pragma unroll
        for (uint32_t t = 0; t < TokenTile; ++t)
            all_warps[r][t][warp][lane] = partial[r][t];
    }
    __syncthreads();
    #pragma unroll
    for (uint32_t local = 0; local < 2; ++local) {
        const uint32_t t = warp + local * 4U;
        #pragma unroll
        for (uint32_t r = 0; r < RowTile; ++r) {
            float sum = all_warps[r][t][0][lane];
            #pragma unroll
            for (uint32_t other = 1; other < 4; ++other)
                sum += all_warps[r][t][other][lane];
            #pragma unroll
            for (uint32_t offset = 16; offset != 0; offset >>= 1U)
                sum += __shfl_down_sync(0xffffffffU, sum, offset);
            if (lane == 0U)
                output[static_cast<uint64_t>(first_token + t) * rows + first_row + r] =
                    Format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1
                        ? sum * tensor_scale : sum;
        }
    }
}
