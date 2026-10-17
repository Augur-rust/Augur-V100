// Exact experimental tile scheduling; arithmetic copied from the component-qualified kernel.
// Included after the canonical packed-load helpers inside the production TU namespace.
#pragma once

template<uint32_t Format, uint32_t Tile, uint32_t MinBlocks, bool FullTile>
__global__ __launch_bounds__(128, MinBlocks) void quantized_small_batch_tiled_kernel(
    const uint8_t* weight, const uint8_t* activation, float* output,
    uint32_t rows, uint32_t cols, uint32_t tokens, float tensor_scale) {
    const uint32_t row = blockIdx.x, first_token = blockIdx.y * Tile;
    const uint32_t thread = threadIdx.x, lane = thread & 31U, warp = thread / 32U;
    const uint64_t activation_stride = static_cast<uint64_t>(cols / 32U) * 36U;
    float partial[Tile] = {};
    if constexpr (Format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1) {
        const uint32_t blocks = cols / 32U;
        const uint8_t* source = weight + static_cast<uint64_t>(row) * blocks * 34U;
        for (uint32_t block = thread / 4U; block < blocks; block += 32U) {
            const uint8_t* w = source + static_cast<uint64_t>(block) * 34U;
            const uint32_t offset = (thread & 3U) * 8U;
            const int32_t w0 = static_cast<int32_t>(small_batch_q8_word(w + 2U + offset));
            const int32_t w1 = static_cast<int32_t>(small_batch_q8_word(w + 6U + offset));
            const float scale = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(w)));
            #pragma unroll
            for (uint32_t t = 0; t < Tile; ++t) {
                if (!FullTile && first_token + t >= tokens) continue;
                const uint8_t* a = activation + (first_token + t) * activation_stride
                    + static_cast<uint64_t>(block) * 36U;
                const auto* codes = reinterpret_cast<const int32_t*>(a + 4U + offset);
                int32_t dot = __dp4a(w0, codes[0], 0);
                dot = __dp4a(w1, codes[1], dot);
                const float combined = scale * __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(a)));
                partial[t] = fmaf(combined, static_cast<float>(dot), partial[t]);
            }
        }
    } else {
        const uint32_t blocks = cols / 64U;
        const uint8_t* source = weight + static_cast<uint64_t>(row) * blocks * 36U;
        for (uint32_t block = thread / 2U; block < blocks; block += 64U) {
            const uint8_t* w = source + static_cast<uint64_t>(block) * 36U;
            const auto* packed = reinterpret_cast<const uint32_t*>(w + 4U);
            #pragma unroll
            for (uint32_t pair = 0; pair < 2; ++pair) {
                const uint32_t word = (thread & 1U) * 4U + pair * 2U;
                const uint32_t sub = word / 2U;
                const int2 w0 = small_batch_e2m1_words(packed[word]);
                const int2 w1 = small_batch_e2m1_words(packed[word + 1U]);
                const float scale = ue4m3_scale_to_f32(w[sub]) * 0.5f;
                #pragma unroll
                for (uint32_t t = 0; t < Tile; ++t) {
                    if (!FullTile && first_token + t >= tokens) continue;
                    const uint8_t* a = activation + (first_token + t) * activation_stride
                        + static_cast<uint64_t>(block * 2U + sub / 2U) * 36U;
                    const auto* codes = reinterpret_cast<const int32_t*>(a + 4U);
                    const uint32_t offset = (sub & 1U) * 4U;
                    int32_t dot = __dp4a(w0.x, codes[offset], 0);
                    dot = __dp4a(w0.y, codes[offset + 2U], dot);
                    dot = __dp4a(w1.x, codes[offset + 1U], dot);
                    dot = __dp4a(w1.y, codes[offset + 3U], dot);
                    const float combined = scale * __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(a)));
                    partial[t] = fmaf(combined, static_cast<float>(dot), partial[t]);
                }
            }
        }
    }
    __shared__ float other_warps[Tile][3][32];
    if (warp > 0U) {
        #pragma unroll
        for (uint32_t t = 0; t < Tile; ++t) other_warps[t][warp - 1U][lane] = partial[t];
    }
    __syncthreads();
    if (warp > 0U) return;
    #pragma unroll
    for (uint32_t t = 0; t < Tile; ++t) {
        if (!FullTile && first_token + t >= tokens) continue;
        #pragma unroll
        for (uint32_t other = 0; other < 3; ++other) partial[t] += other_warps[t][other][lane];
        #pragma unroll
        for (uint32_t offset = 16; offset != 0; offset >>= 1U)
            partial[t] += __shfl_down_sync(0xffffffffU, partial[t], offset);
        if (lane == 0U) output[static_cast<uint64_t>(first_token + t) * rows + row] =
            Format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1 ? partial[t] * tensor_scale : partial[t];
    }
}
// NVFP4 FFN-down only: frozen rows5120/cols17408/full8 selector.
// Keep the original DP4A/FMA and reduction body; make only K scheduling static.
// Q8, tails and all other shapes remain on the original tiled kernel.
template<uint32_t Format, uint32_t Tile, uint32_t MinBlocks, bool FullTile>
__global__ __launch_bounds__(128, MinBlocks) void quantized_small_batch_down_cols17408_kernel(
    const uint8_t* weight, const uint8_t* activation, float* output,
    uint32_t rows, uint32_t cols_arg, uint32_t tokens, float tensor_scale) {
    constexpr uint32_t cols = 17408;
    const uint32_t row = blockIdx.x, first_token = blockIdx.y * Tile;
    const uint32_t thread = threadIdx.x, lane = thread & 31U, warp = thread / 32U;
    const uint64_t activation_stride = static_cast<uint64_t>(cols / 32U) * 36U;
    float partial[Tile] = {};
    if constexpr (Format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1) {
        const uint32_t blocks = cols / 32U;
        const uint8_t* source = weight + static_cast<uint64_t>(row) * blocks * 34U;
        #pragma unroll
        for (uint32_t block = thread / 4U; block < blocks; block += 32U) {
            const uint8_t* w = source + static_cast<uint64_t>(block) * 34U;
            const uint32_t offset = (thread & 3U) * 8U;
            const int32_t w0 = static_cast<int32_t>(small_batch_q8_word(w + 2U + offset));
            const int32_t w1 = static_cast<int32_t>(small_batch_q8_word(w + 6U + offset));
            const float scale = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(w)));
            #pragma unroll
            for (uint32_t t = 0; t < Tile; ++t) {
                if (!FullTile && first_token + t >= tokens) continue;
                const uint8_t* a = activation + (first_token + t) * activation_stride
                    + static_cast<uint64_t>(block) * 36U;
                const auto* codes = reinterpret_cast<const int32_t*>(a + 4U + offset);
                int32_t dot = __dp4a(w0, codes[0], 0);
                dot = __dp4a(w1, codes[1], dot);
                const float combined = scale * __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(a)));
                partial[t] = fmaf(combined, static_cast<float>(dot), partial[t]);
            }
        }
    } else {
        const uint32_t blocks = cols / 64U;
        const uint8_t* source = weight + static_cast<uint64_t>(row) * blocks * 36U;
        #pragma unroll
        for (uint32_t block = thread / 2U; block < blocks; block += 64U) {
            const uint8_t* w = source + static_cast<uint64_t>(block) * 36U;
            const auto* packed = reinterpret_cast<const uint32_t*>(w + 4U);
            #pragma unroll
            for (uint32_t pair = 0; pair < 2; ++pair) {
                const uint32_t word = (thread & 1U) * 4U + pair * 2U;
                const uint32_t sub = word / 2U;
                const int2 w0 = small_batch_e2m1_words(packed[word]);
                const int2 w1 = small_batch_e2m1_words(packed[word + 1U]);
                const float scale = ue4m3_scale_to_f32(w[sub]) * 0.5f;
                #pragma unroll
                for (uint32_t t = 0; t < Tile; ++t) {
                    if (!FullTile && first_token + t >= tokens) continue;
                    const uint8_t* a = activation + (first_token + t) * activation_stride
                        + static_cast<uint64_t>(block * 2U + sub / 2U) * 36U;
                    const auto* codes = reinterpret_cast<const int32_t*>(a + 4U);
                    const uint32_t offset = (sub & 1U) * 4U;
                    int32_t dot = __dp4a(w0.x, codes[offset], 0);
                    dot = __dp4a(w0.y, codes[offset + 2U], dot);
                    dot = __dp4a(w1.x, codes[offset + 1U], dot);
                    dot = __dp4a(w1.y, codes[offset + 3U], dot);
                    const float combined = scale * __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(a)));
                    partial[t] = fmaf(combined, static_cast<float>(dot), partial[t]);
                }
            }
        }
    }
    __shared__ float other_warps[Tile][3][32];
    if (warp > 0U) {
        #pragma unroll
        for (uint32_t t = 0; t < Tile; ++t) other_warps[t][warp - 1U][lane] = partial[t];
    }
    __syncthreads();
    if (warp > 0U) return;
    #pragma unroll
    for (uint32_t t = 0; t < Tile; ++t) {
        if (!FullTile && first_token + t >= tokens) continue;
        #pragma unroll
        for (uint32_t other = 0; other < 3; ++other) partial[t] += other_warps[t][other][lane];
        #pragma unroll
        for (uint32_t offset = 16; offset != 0; offset >>= 1U)
            partial[t] += __shfl_down_sync(0xffffffffU, partial[t], offset);
        if (lane == 0U) output[static_cast<uint64_t>(first_token + t) * rows + row] =
            Format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1 ? partial[t] * tensor_scale : partial[t];
    }
}
