// Expression/launch geometry adapted from ggml CUDA unary_gated_op_kernel
// and norm.cu/common.cuh. Only explicitly selected input primitives use it.
// Copyright (c) 2023-2026 The ggml authors. SPDX-License-Identifier: MIT.
// This translation unit alone uses --use_fast_math; callers remain default-off.
#include <cuda_runtime.h>
#include <cstdint>

namespace {
__global__ void llama_prefill_swiglu_kernel(const float* gate, const float* up,
    float* output, uint32_t elements) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elements) {
        const float value = gate[i];
        output[i] = (value / (1.0f + expf(-value))) * up[i];
    }
}

__device__ __forceinline__ float llama_norm_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, offset, 32);
    }
    return value;
}

// Exact-input Qwen35 hidden5120 only. The two XOR reductions, 1024-thread
// partition and fast-math TU match actual-DLL same-operand replay. No new
// scratch, allocation, host wait or change to canonical/decode norm arithmetic.
__global__ void llama_input_rmsnorm5120_kernel(const float* input,
    const float* weight, float* output) {
    constexpr uint32_t cols = 5120;
    input += static_cast<uint64_t>(blockIdx.x) * cols;
    output += static_cast<uint64_t>(blockIdx.x) * cols;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid % 32;
    const uint32_t warp = tid / 32;
    float sum = 0.0f;
    for (uint32_t col = tid; col < cols; col += 1024) {
        const float value = input[col];
        sum += value * value;
    }
    __shared__ float partial[32];
    sum = llama_norm_warp_sum(sum);
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    sum = llama_norm_warp_sum(partial[lane]);
    const float mean = sum / cols;
    const float scale = rsqrtf(mean + 1e-6f);
    for (uint32_t col = tid; col < cols; col += 1024) {
        output[col] = scale * input[col] * weight[col];
    }
}
}

extern "C" cudaError_t augur_qwen35_launch_llama_swiglu_sm70(
    const float* gate, const float* up, float* output, uint32_t elements,
    cudaStream_t stream) {
    llama_prefill_swiglu_kernel<<<(elements + 255U) / 256U, 256, 0, stream>>>(
        gate, up, output, elements);
    return cudaGetLastError();
}

extern "C" cudaError_t augur_qwen35_launch_llama_input_rmsnorm5120_sm70(
    const float* input, const float* weight, float* output, uint32_t rows,
    cudaStream_t stream) {
    llama_input_rmsnorm5120_kernel<<<rows, 1024, 0, stream>>>(input, weight, output);
    return cudaGetLastError();
}
