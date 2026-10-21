// Research-only B1 pair: original BF16 weights, F32 activations, explicit
// FMA lane partials and two-stage F32 tree. No casts to half or FP64 in kernel.
// The original a/b buffers are retained; no concatenated weight copy is needed.
namespace {
__global__ void nerkyor_bf16_gate_pair_f32_cta_kernel_v1(
    const uint16_t* a, const uint16_t* b, const float* x, float* ya, float* yb) {
    const int row = blockIdx.x % 48;
    const bool right = blockIdx.x >= 48;
    const uint16_t* w = (right ? b : a) + size_t(row) * 5120;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    __shared__ float partials[8];
    float sum = 0.0f;
    for (int k = threadIdx.x; k < 5120; k += 256)
        sum = fmaf(__uint_as_float(uint32_t(w[k]) << 16), x[k], sum);
    for (int d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
    if (lane == 0) partials[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? partials[lane] : 0.0f;
        for (int d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
        if (lane == 0) (right ? yb : ya)[row] = sum;
    }
}
bool nerkyor_gate_args_valid(const augur_cuda_bf16_f32_matvec_args_v1* a) {
    return a != nullptr && a->abi_version == AUGUR_CUDA_BF16_F32_MATVEC_ARGS_VERSION_V1
        && a->struct_size == sizeof(*a) && a->rows == 48 && a->cols == 5120
        && a->weight != 0 && a->input != 0 && a->output != 0
        && a->weight % 2 == 0 && a->input % 4 == 0 && a->output % 4 == 0
        && a->weight_len_bytes == uint64_t(48) * 5120 * 2
        && a->input_len_bytes == 5120 * 4 && a->output_len_bytes == 48 * 4
        && a->weight <= UINT64_MAX - a->weight_len_bytes
        && a->input <= UINT64_MAX - a->input_len_bytes
        && a->output <= UINT64_MAX - a->output_len_bytes;
}
bool nerkyor_gate_overlap(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) {
    return a < b + bn && b < a + an;
}
} // namespace
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nerkyor_bf16_gate_pair_f32_cta_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_bf16_f32_matvec_args_v1* a,
    const augur_cuda_bf16_f32_matvec_args_v1* b) {
    if (stream == nullptr || stream->stream == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || !nerkyor_gate_args_valid(a) || !nerkyor_gate_args_valid(b)
        || a->input != b->input || a->input_len_bytes != b->input_len_bytes)
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    const uint64_t starts[] = {a->weight, b->weight, a->input, a->output, b->output};
    const uint64_t sizes[] = {a->weight_len_bytes, b->weight_len_bytes,
        a->input_len_bytes, a->output_len_bytes, b->output_len_bytes};
    for (int i = 0; i < 5; ++i) for (int j = i + 1; j < 5; ++j)
        if (nerkyor_gate_overlap(starts[i], sizes[i], starts[j], sizes[j]))
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    auto device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    cudaError_t error;
    nerkyor_bf16_gate_pair_f32_cta_kernel_v1<<<96, 256, 0, stream->stream>>>(
        reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(a->weight)),
        reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(b->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(b->output)));
    error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}
