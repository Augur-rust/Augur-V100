#pragma once

// Same NVFP4 element mapping, FP32 multiplication and FP16 rounding as the
// one-warp/64-value llama decoder. Grouping changes only memory/CTA scheduling.
// Like our llama-derived Q8 converter, stage contiguous packed words before
// emitting adjacent half2 values. No tensor scale is folded into this decode.
// The caller admits complete2048-value groups; other extents stay scalar.
template<unsigned Threads>
__global__ __launch_bounds__(Threads) void decode_nvfp4_llama_grouped_kernel(
    const uint8_t* source, __half* dest) {
    static_assert(Threads == 32 || Threads == 128 || Threads == 256, "grouped NVFP4 schedule");
    constexpr unsigned elements = 2048, words = 288;
    __shared__ uint32_t packed[words];
    const auto* input = reinterpret_cast<const uint32_t*>(source) + uint64_t(blockIdx.x) * words;
    for (unsigned i = threadIdx.x; i < words; i += Threads) packed[i] = input[i];
    __syncthreads();
    auto* output = reinterpret_cast<__half2*>(dest + uint64_t(blockIdx.x) * elements);
    #pragma unroll
    for (unsigned i = 0; i < elements / 2; i += Threads) {
        const unsigned element = (i + threadIdx.x) * 2U;
        const auto* p = reinterpret_cast<const uint8_t*>(packed) + (element / 64U) * 36U;
        const unsigned sub = (element % 64U) / 16U, j = element % 16U;
        const unsigned shift = j >= 8U ? 4U : 0U;
        const uint16_t codes = *reinterpret_cast<const uint16_t*>(p + 4U + sub * 8U + (j & 7U));
        const float scale = ue4m3_scale_to_f32(p[sub]);
        const float a = scale * e2m1_value((codes >> shift) & 15U);
        const float b = scale * e2m1_value((codes >> (shift + 8U)) & 15U);
        output[i + threadIdx.x] = __floats2half2_rn(a, b);
    }
}
