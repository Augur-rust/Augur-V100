// Independent small-gate scheduling contract. Include after the exact NVFP4
// decode helpers. Packed weights and the F32 pair expression are unchanged.
// SplitK partitions consecutive 64-element blocks per lane, folds partials in
// increasing split order in F64, then uses the original warp shuffle tree.
// No atomics, padding, activation quantization, hot allocation or HBM scratch.
// Different F64 addition order is NOT a bit-exact contract to the scalar path.
template<int SplitK>
__global__ __launch_bounds__(SplitK * 32, 1)
void nvfp4_sm70_small_gate_split_f64_kernel(
    const uint8_t* __restrict__ weight, const float* __restrict__ input,
    float* __restrict__ output, uint32_t rows, uint32_t cols, float tensor_scale
) {
    __shared__ double partials[SplitK][32];
    const uint32_t row = blockIdx.x, token = blockIdx.y;
    const uint32_t split = threadIdx.x / 32U, lane = threadIdx.x & 31U;
    if (row >= rows) return;
    input += static_cast<uint64_t>(token) * cols;
    output += static_cast<uint64_t>(token) * rows;
    const uint32_t blocks_per_row = cols / 64U;
    const uint32_t quotient = blocks_per_row / SplitK;
    const uint32_t begin = split * quotient;
    const uint32_t end = split + 1U == SplitK ? blocks_per_row : begin + quotient;
    const uint8_t* row_source = weight + static_cast<uint64_t>(row) * blocks_per_row * 36U;
    double partial = 0.0;
    for (uint32_t block = begin; block < end; ++block) {
        const uint8_t* block_source = row_source + static_cast<uint64_t>(block) * 36U;
        const uint32_t sub_block = lane / 8U, sub_lane = lane & 7U;
        const uint32_t input_base = block * 64U + sub_block * 16U;
        const uint8_t packed = block_source[4U + lane];
        float scale = lane < 4U ? ue4m3_scale_to_f32(block_source[lane]) : 0.0f;
        scale = __shfl_sync(0xffffffffU, scale, sub_block);
        const float low = e2m1_value(packed & 0x0fU);
        const float high = e2m1_value(packed >> 4U);
        const float pair = low * input[input_base + sub_lane]
            + high * input[input_base + 8U + sub_lane];
        partial += static_cast<double>(pair) * static_cast<double>(scale);
    }
    partials[split][lane] = partial;
    __syncthreads();
    if (split != 0) return;
    partial = partials[0][lane];
    #pragma unroll
    for (int s = 1; s < SplitK; ++s) partial += partials[s][lane];
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U)
        partial += __shfl_down_sync(0xffffffffU, partial, offset);
    if (lane == 0) output[row] = static_cast<float>(partial * static_cast<double>(tensor_scale));
}
