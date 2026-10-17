#include "augur_cuda_internal.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>

static_assert(sizeof(augur_cuda_qwen35_token_handoff_receipt_v1) == 72,
    "Qwen35 token-handoff receipt ABI drift");
static_assert(alignof(augur_cuda_qwen35_token_handoff_receipt_v1) == 8,
    "Qwen35 token-handoff receipt alignment drift");
static_assert(offsetof(augur_cuda_qwen35_token_handoff_receipt_v1, token_id) == 0,
    "Qwen35 token-handoff token offset drift");
static_assert(offsetof(augur_cuda_qwen35_token_handoff_receipt_v1, finite_count) == 16,
    "Qwen35 token-handoff finite-count offset drift");
static_assert(offsetof(augur_cuda_qwen35_token_handoff_receipt_v1, status) == 24,
    "Qwen35 token-handoff status offset drift");
static_assert(offsetof(augur_cuda_qwen35_token_handoff_receipt_v1, step_generation) == 32,
    "Qwen35 token-handoff generation offset drift");
static_assert(offsetof(augur_cuda_qwen35_token_handoff_receipt_v1, plan_digest) == 40,
    "Qwen35 token-handoff digest offset drift");

static_assert(sizeof(augur_cuda_qwen35_token_handoff_args_v1) == 104,
    "Qwen35 token-handoff args ABI drift");
static_assert(alignof(augur_cuda_qwen35_token_handoff_args_v1) == 8,
    "Qwen35 token-handoff args alignment drift");
static_assert(offsetof(augur_cuda_qwen35_token_handoff_args_v1, logits) == 8,
    "Qwen35 token-handoff logits offset drift");
static_assert(offsetof(augur_cuda_qwen35_token_handoff_args_v1, partials) == 24,
    "Qwen35 token-handoff partials offset drift");
static_assert(offsetof(augur_cuda_qwen35_token_handoff_args_v1, receipt) == 40,
    "Qwen35 token-handoff receipt pointer offset drift");
static_assert(offsetof(augur_cuda_qwen35_token_handoff_args_v1, step_generation) == 56,
    "Qwen35 token-handoff generation arg offset drift");
static_assert(offsetof(augur_cuda_qwen35_token_handoff_args_v1, vocab) == 64,
    "Qwen35 token-handoff vocab offset drift");
static_assert(offsetof(augur_cuda_qwen35_token_handoff_args_v1, plan_digest) == 72,
    "Qwen35 token-handoff digest arg offset drift");

struct alignas(8) augur_qwen35_token_handoff_partial_v1 {
    uint32_t top1_id;
    uint32_t top1_value_bits;
    uint32_t top2_id;
    uint32_t top2_value_bits;
    uint32_t first_non_finite_id;
    uint32_t finite_count;
    uint64_t reserved;
};

struct alignas(8) augur_qwen35_token_handoff_digest_v1 {
    uint8_t bytes[32];
};

static_assert(sizeof(augur_qwen35_token_handoff_partial_v1) == 32,
    "Qwen35 token-handoff partial layout drift");
static_assert(sizeof(augur_qwen35_token_handoff_digest_v1) == 32,
    "Qwen35 token-handoff digest layout drift");

static __device__ __forceinline__ augur_qwen35_token_handoff_partial_v1
augur_qwen35_token_handoff_empty_partial() {
    augur_qwen35_token_handoff_partial_v1 value{};
    value.top1_id = UINT32_MAX;
    value.top2_id = UINT32_MAX;
    value.first_non_finite_id = UINT32_MAX;
    return value;
}

static __device__ __forceinline__ bool augur_qwen35_token_handoff_is_finite_bits(
    uint32_t bits
) {
    return ((bits >> 23U) & 0xffU) != 0xffU;
}

static __device__ __forceinline__ bool augur_qwen35_token_handoff_better(
    uint32_t left_id,
    uint32_t left_bits,
    uint32_t right_id,
    uint32_t right_bits
) {
    if (left_id == UINT32_MAX) return false;
    if (right_id == UINT32_MAX) return true;
    const float left = __uint_as_float(left_bits);
    const float right = __uint_as_float(right_bits);
    return left > right || (left == right && left_id < right_id);
}

static __device__ __forceinline__ void augur_qwen35_token_handoff_insert(
    augur_qwen35_token_handoff_partial_v1& output,
    uint32_t id,
    uint32_t bits
) {
    if (id == UINT32_MAX || id == output.top1_id || id == output.top2_id) return;
    if (augur_qwen35_token_handoff_better(
            id, bits, output.top1_id, output.top1_value_bits)) {
        output.top2_id = output.top1_id;
        output.top2_value_bits = output.top1_value_bits;
        output.top1_id = id;
        output.top1_value_bits = bits;
    } else if (augur_qwen35_token_handoff_better(
                   id, bits, output.top2_id, output.top2_value_bits)) {
        output.top2_id = id;
        output.top2_value_bits = bits;
    }
}

static __device__ __forceinline__ augur_qwen35_token_handoff_partial_v1
augur_qwen35_token_handoff_merge(
    const augur_qwen35_token_handoff_partial_v1& left,
    const augur_qwen35_token_handoff_partial_v1& right
) {
    augur_qwen35_token_handoff_partial_v1 output =
        augur_qwen35_token_handoff_empty_partial();
    augur_qwen35_token_handoff_insert(output, left.top1_id, left.top1_value_bits);
    augur_qwen35_token_handoff_insert(output, left.top2_id, left.top2_value_bits);
    augur_qwen35_token_handoff_insert(output, right.top1_id, right.top1_value_bits);
    augur_qwen35_token_handoff_insert(output, right.top2_id, right.top2_value_bits);
    output.first_non_finite_id = min(
        left.first_non_finite_id, right.first_non_finite_id);
    output.finite_count = left.finite_count + right.finite_count;
    return output;
}

__global__ static void augur_qwen35_token_handoff_scan_kernel(
    const float* logits,
    augur_qwen35_token_handoff_partial_v1* partials
) {
    __shared__ augur_qwen35_token_handoff_partial_v1
        shared[AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_THREADS_V1];
    const uint32_t lane = threadIdx.x;
    const uint32_t first = blockIdx.x * blockDim.x + lane;
    const uint32_t stride = gridDim.x * blockDim.x;
    augur_qwen35_token_handoff_partial_v1 local =
        augur_qwen35_token_handoff_empty_partial();
    for (uint32_t index = first;
         index < AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_VOCAB_V1;
         index += stride) {
        const uint32_t bits = __float_as_uint(logits[index]);
        if (!augur_qwen35_token_handoff_is_finite_bits(bits)) {
            local.first_non_finite_id = min(local.first_non_finite_id, index);
        } else {
            augur_qwen35_token_handoff_insert(local, index, bits);
            ++local.finite_count;
        }
    }
    shared[lane] = local;
    __syncthreads();
    for (uint32_t reduction_stride = blockDim.x / 2U;
         reduction_stride > 0;
         reduction_stride >>= 1U) {
        if (lane < reduction_stride) {
            shared[lane] = augur_qwen35_token_handoff_merge(
                shared[lane], shared[lane + reduction_stride]);
        }
        __syncthreads();
    }
    if (lane == 0) partials[blockIdx.x] = shared[0];
}

__global__ static void augur_qwen35_token_handoff_finalize_kernel(
    const augur_qwen35_token_handoff_partial_v1* partials,
    augur_cuda_qwen35_token_handoff_receipt_v1* receipt,
    uint64_t step_generation,
    augur_qwen35_token_handoff_digest_v1 plan_digest
) {
    __shared__ augur_qwen35_token_handoff_partial_v1
        shared[AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_THREADS_V1];
    const uint32_t lane = threadIdx.x;
    shared[lane] = lane < AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_PARTIAL_COUNT_V1
        ? partials[lane]
        : augur_qwen35_token_handoff_empty_partial();
    __syncthreads();
    for (uint32_t reduction_stride = blockDim.x / 2U;
         reduction_stride > 0;
         reduction_stride >>= 1U) {
        if (lane < reduction_stride) {
            shared[lane] = augur_qwen35_token_handoff_merge(
                shared[lane], shared[lane + reduction_stride]);
        }
        __syncthreads();
    }
    if (lane != 0) return;

    const augur_qwen35_token_handoff_partial_v1 result = shared[0];
    const bool non_finite = result.first_non_finite_id != UINT32_MAX;
    const bool incomplete = result.finite_count != AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_VOCAB_V1
        || result.top1_id == UINT32_MAX || result.top2_id == UINT32_MAX;
    const bool failed = non_finite || incomplete;
    receipt->token_id = failed ? UINT32_MAX : result.top1_id;
    receipt->top1_value_bits = failed ? 0U : result.top1_value_bits;
    receipt->runner_up_id = failed ? UINT32_MAX : result.top2_id;
    receipt->runner_up_value_bits = failed ? 0U : result.top2_value_bits;
    receipt->finite_count = result.finite_count;
    receipt->first_non_finite_id = result.first_non_finite_id;
    receipt->status = failed
        ? AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_STATUS_ERROR_V1
        : AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_STATUS_OK_V1;
    receipt->error_code = non_finite
        ? AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ERROR_NON_FINITE_LOGIT_V1
        : (incomplete
            ? AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ERROR_INCOMPLETE_REDUCTION_V1
            : AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ERROR_NONE_V1);
    receipt->step_generation = step_generation;
    for (uint32_t index = 0; index < 32; ++index) {
        receipt->plan_digest[index] = plan_digest.bytes[index];
    }
}

static augur_cuda_result_v1 augur_qwen35_token_handoff_result(
    augur_cuda_status_t status,
    cudaError_t error
) {
    return augur_cuda_result_v1{status, static_cast<int32_t>(error)};
}

static bool augur_qwen35_token_handoff_nonzero_digest(const uint8_t digest[32]) {
    uint8_t any = 0;
    for (uint32_t index = 0; index < 32; ++index) any |= digest[index];
    return any != 0;
}

static bool augur_qwen35_token_handoff_range(
    uint64_t address,
    uint64_t extent,
    uint64_t alignment,
    uint64_t* end
) {
    if (address == 0 || extent == 0 || address % alignment != 0
        || address > std::numeric_limits<uint64_t>::max() - extent) {
        return false;
    }
    *end = address + extent;
    return true;
}

static bool augur_qwen35_token_handoff_overlap(
    uint64_t left_start,
    uint64_t left_end,
    uint64_t right_start,
    uint64_t right_end
) {
    return left_start < right_end && right_start < left_end;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_token_handoff_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_token_handoff_args_v1* args
) {
    if (args == nullptr || stream == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_qwen35_token_handoff_result(
            AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args)) {
        return augur_qwen35_token_handoff_result(
            AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (args->vocab != AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_VOCAB_V1
        || args->partial_count != AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_PARTIAL_COUNT_V1
        || args->logits_len_bytes != AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_LOGITS_BYTES_V1
        || args->partials_len_bytes != AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_PARTIAL_BYTES_V1
        || args->receipt_len_bytes != AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES_V1
        || args->step_generation == 0
        || !augur_qwen35_token_handoff_nonzero_digest(args->plan_digest)) {
        return augur_qwen35_token_handoff_result(
            AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    uint64_t logits_end = 0;
    uint64_t partials_end = 0;
    uint64_t receipt_end = 0;
    if (!augur_qwen35_token_handoff_range(
            args->logits, args->logits_len_bytes, 4, &logits_end)
        || !augur_qwen35_token_handoff_range(
            args->partials, args->partials_len_bytes, 16, &partials_end)
        || !augur_qwen35_token_handoff_range(
            args->receipt, args->receipt_len_bytes, 8, &receipt_end)
        || augur_qwen35_token_handoff_overlap(
            args->logits, logits_end, args->partials, partials_end)
        || augur_qwen35_token_handoff_overlap(
            args->logits, logits_end, args->receipt, receipt_end)
        || augur_qwen35_token_handoff_overlap(
            args->partials, partials_end, args->receipt, receipt_end)) {
        return augur_qwen35_token_handoff_result(
            AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) {
        return augur_qwen35_token_handoff_result(
            AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    }
    const auto* logits = reinterpret_cast<const float*>(
        static_cast<uintptr_t>(args->logits));
    auto* partials = reinterpret_cast<augur_qwen35_token_handoff_partial_v1*>(
        static_cast<uintptr_t>(args->partials));
    auto* receipt = reinterpret_cast<augur_cuda_qwen35_token_handoff_receipt_v1*>(
        static_cast<uintptr_t>(args->receipt));
    augur_qwen35_token_handoff_digest_v1 digest{};
    for (uint32_t index = 0; index < 32; ++index) {
        digest.bytes[index] = args->plan_digest[index];
    }

    augur_qwen35_token_handoff_scan_kernel<<<
        AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_PARTIAL_COUNT_V1,
        AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_THREADS_V1,
        0,
        stream->stream>>>(logits, partials);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return augur_qwen35_token_handoff_result(
            AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    }
    augur_qwen35_token_handoff_finalize_kernel<<<
        1,
        AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_THREADS_V1,
        0,
        stream->stream>>>(partials, receipt, args->step_generation, digest);
    error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_qwen35_token_handoff_result(AUGUR_CUDA_STATUS_OK, cudaSuccess)
        : augur_qwen35_token_handoff_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}
