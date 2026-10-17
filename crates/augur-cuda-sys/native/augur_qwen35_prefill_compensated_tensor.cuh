// Default-off experimental arithmetic. Include after the native decode helpers.
// Split quantized operands into FP16 high and a scaled FP16 residual. NVFP4
// E2M1*UE4M3 is exactly representable in FP16 when its global scale is factored
// out. Q8_0 additionally needs both weight cross terms and the low*low term.
#pragma once

namespace compensated_tensor {
constexpr float residual_scale = 4096.0f;

__device__ __forceinline__ void split4(float4 values, __half* high, __half* low) {
    const __half2 h0 = __floats2half2_rn(values.x, values.y);
    const __half2 h1 = __floats2half2_rn(values.z, values.w);
    const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
    reinterpret_cast<__half2*>(high)[0] = h0;
    reinterpret_cast<__half2*>(high)[1] = h1;
    reinterpret_cast<__half2*>(low)[0] = __floats2half2_rn(
        (values.x - f0.x) * residual_scale, (values.y - f0.y) * residual_scale);
    reinterpret_cast<__half2*>(low)[1] = __floats2half2_rn(
        (values.z - f1.x) * residual_scale, (values.w - f1.y) * residual_scale);
}

template<bool Activation>
__global__ void split_q8_quad_kernel(const uint8_t* source, __half* high,
                                    __half* low, uint64_t elements) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t quad = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         quad < elements / 4U; quad += stride) {
        const uint64_t index = quad * 4U;
        const uint8_t* block = source + (index / 32U) * (Activation ? 36U : 34U);
        const float scale = __half2float(__ushort_as_half(
            *reinterpret_cast<const uint16_t*>(block)));
        const int8_t* codes = reinterpret_cast<const int8_t*>(block + (Activation ? 4U : 2U))
            + index % 32U;
        split4(make_float4(scale * static_cast<float>(codes[0]),
                          scale * static_cast<float>(codes[1]),
                          scale * static_cast<float>(codes[2]),
                          scale * static_cast<float>(codes[3])), high + index, low + index);
    }
}

__global__ void apply_global_scale_kernel(float* output, uint64_t elements, float scale) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += stride) output[index] *= scale;
}

inline cublasStatus_t gemm(cublasHandle_t handle, const __half* weight, const __half* input,
                          float* output, unsigned rows, unsigned cols, unsigned tokens,
                          unsigned output_stride, float alpha, float beta) {
    return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, tokens, cols,
        &alpha, weight, CUDA_R_16F, cols, input, CUDA_R_16F, cols,
        &beta, output, CUDA_R_32F, output_stride, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

inline cublasStatus_t submit(cublasHandle_t handle, cudaStream_t stream,
    const uint8_t* weight, const uint8_t* activation, __half* weight_high,
    __half* weight_low, __half* input_high, __half* input_low, float* output,
    unsigned tokens, unsigned rows, unsigned cols, unsigned format, float tensor_scale,
    unsigned weight_rows) {
    split_q8_quad_kernel<true><<<launch_blocks(static_cast<uint64_t>(tokens) * cols / 4U),
        256, 0, stream>>>(activation, input_high, input_low, static_cast<uint64_t>(tokens) * cols);
    for (unsigned first = 0; first < rows; first += weight_rows) {
        const unsigned count = std::min(weight_rows, rows - first);
        const uint8_t* tile = weight + static_cast<uint64_t>(first) * cols
            / (format == 1U ? 32U : 64U) * (format == 1U ? 34U : 36U);
        const uint64_t elements = static_cast<uint64_t>(count) * cols;
        if (format == 1U) {
            split_q8_quad_kernel<false><<<launch_blocks(elements / 4U), 256, 0, stream>>>(
                tile, weight_high, weight_low, elements);
        } else {
            decode_weight_quad_kernel<2><<<launch_blocks(elements / 4U), 256, 0, stream>>>(
                tile, weight_high, elements, 1.0f);
        }
        cublasStatus_t status = gemm(handle, weight_high, input_high, output + first,
            count, cols, tokens, rows, 1.0f, 0.0f);
        if (status != CUBLAS_STATUS_SUCCESS) return status;
        status = gemm(handle, weight_high, input_low, output + first,
            count, cols, tokens, rows, 1.0f / residual_scale, 1.0f);
        if (status != CUBLAS_STATUS_SUCCESS) return status;
        if (format == 1U) {
            status = gemm(handle, weight_low, input_high, output + first,
                count, cols, tokens, rows, 1.0f / residual_scale, 1.0f);
            if (status != CUBLAS_STATUS_SUCCESS) return status;
            status = gemm(handle, weight_low, input_low, output + first,
                count, cols, tokens, rows, 1.0f / (residual_scale * residual_scale), 1.0f);
            if (status != CUBLAS_STATUS_SUCCESS) return status;
        }
    }
    if (format == 2U) apply_global_scale_kernel<<<
        launch_blocks(static_cast<uint64_t>(tokens) * rows), 256, 0, stream>>>(
            output, static_cast<uint64_t>(tokens) * rows, tensor_scale);
    return CUBLAS_STATUS_SUCCESS;
}
}
