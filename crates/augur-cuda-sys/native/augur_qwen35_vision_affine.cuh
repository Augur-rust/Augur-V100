// Included once by the stream-owned cuBLAS translation unit, opt-in only.
// W and X are already FP16. This operation performs NO input conversion.
static_assert(sizeof(augur_cuda_qwen35_vision_f16_affine_args_v1) == 88,
    "Vision FP16 affine ABI drift");
static_assert(offsetof(augur_cuda_qwen35_vision_f16_affine_args_v1, weight_f16) == 24,
    "Vision FP16 weight offset drift");
static_assert(offsetof(augur_cuda_qwen35_vision_f16_affine_args_v1, output_f32) == 72,
    "Vision FP32 output offset drift");

namespace {
bool vision_affine_metadata_valid(int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_vision_f16_affine_args_v1* args, bool inspect_device) {
    if (ordinal < 0 || stream == nullptr || stream->stream == nullptr
        || stream->ordinal != ordinal || plan == nullptr || plan->handle == nullptr
        || plan->ordinal != ordinal || plan->owner != stream || args == nullptr
        || args->abi_version != 1 || args->struct_size != sizeof(*args)
        || args->reserved != 0 || args->tokens == 0 || args->rows == 0 || args->cols == 0
        || args->tokens > plan->snapshot.max_tokens
        || args->tokens > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || args->rows > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || args->cols > static_cast<uint32_t>(std::numeric_limits<int>::max())) return false;
    uint64_t weight_bytes = 0, input_bytes = 0, bias_bytes = 0, output_bytes = 0;
    if (!checked_mul(args->rows, args->cols, &weight_bytes)
        || !checked_mul(weight_bytes, sizeof(__half), &weight_bytes)
        || !checked_mul(args->tokens, args->cols, &input_bytes)
        || !checked_mul(input_bytes, sizeof(__half), &input_bytes)
        || !checked_mul(args->rows, sizeof(float), &bias_bytes)
        || !checked_mul(args->tokens, bias_bytes, &output_bytes)
        || args->weight_len_bytes != weight_bytes || args->input_len_bytes != input_bytes
        || args->bias_len_bytes != bias_bytes || args->output_len_bytes != output_bytes) return false;
    const Range ranges[] = {
        {args->weight_f16, weight_bytes, alignof(__half)},
        {args->input_f16, input_bytes, alignof(__half)},
        {args->bias_f32, bias_bytes, alignof(float)},
        {args->output_f32, output_bytes, alignof(float)},
    };
    if (!all_disjoint(ranges, 4)) return false;
    for (const Range range : ranges) {
        if (range.bytes > SIZE_MAX || (inspect_device && !device_range(range, ordinal))) return false;
    }
    return true;
}

__global__ void vision_affine_bias_kernel(float* output, const float* bias,
    uint64_t elements, uint32_t rows) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements; i += stride) output[i] += bias[i % rows];
}
}  // namespace

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_vision_f16_affine_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_vision_f16_affine_args_v1* args) {
    return vision_affine_metadata_valid(ordinal, stream, plan, args, true)
        ? result(AUGUR_CUDA_STATUS_OK, 0) : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_vision_f16_affine_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_vision_f16_affine_args_v1* args) {
    if (!vision_affine_metadata_valid(ordinal, stream, plan, args, false))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    const float alpha = 1.0f, beta = 0.0f;
    // W is row-major [rows,cols], X is row-major [tokens,cols]. Their column-
    // major views compute W^T * X into row-major [tokens,rows], with FP32 C.
    const cublasStatus_t blas = cublasGemmEx(plan->handle, CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(args->rows), static_cast<int>(args->tokens), static_cast<int>(args->cols),
        &alpha, reinterpret_cast<const void*>(args->weight_f16), CUDA_R_16F,
        static_cast<int>(args->cols), reinterpret_cast<const void*>(args->input_f16), CUDA_R_16F,
        static_cast<int>(args->cols), &beta, reinterpret_cast<void*>(args->output_f32), CUDA_R_32F,
        static_cast<int>(args->rows), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (blas != CUBLAS_STATUS_SUCCESS) return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, blas);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    const uint64_t elements = args->output_len_bytes / sizeof(float);
    vision_affine_bias_kernel<<<launch_blocks(elements), kThreads, 0, stream->stream>>>(
        reinterpret_cast<float*>(args->output_f32), reinterpret_cast<const float*>(args->bias_f32),
        elements, args->rows);
    error = cudaGetLastError();
    return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}
