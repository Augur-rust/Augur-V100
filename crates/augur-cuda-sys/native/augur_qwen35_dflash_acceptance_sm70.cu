#include "augur_cuda_internal.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <limits>

static_assert(sizeof(augur_cuda_qwen35_vocab_top1_args_v1) == 48, "vocab top1 ABI size");
static_assert(alignof(augur_cuda_qwen35_vocab_top1_args_v1) == 8, "vocab top1 ABI alignment");
static_assert(offsetof(augur_cuda_qwen35_vocab_top1_args_v1, logits) == 16, "vocab top1 logits offset");
static_assert(offsetof(augur_cuda_qwen35_vocab_top1_args_v1, ids) == 32, "vocab top1 ids offset");

static_assert(sizeof(augur_cuda_qwen35_dflash_acceptance_receipt_v1) == 64,
    "Qwen35 DFlash acceptance receipt ABI drift");
static_assert(alignof(augur_cuda_qwen35_dflash_acceptance_receipt_v1) == 8,
    "Qwen35 DFlash acceptance receipt alignment drift");
static_assert(offsetof(augur_cuda_qwen35_dflash_acceptance_receipt_v1, draft_tokens) == 0,
    "Qwen35 DFlash acceptance draft count offset drift");
static_assert(offsetof(augur_cuda_qwen35_dflash_acceptance_receipt_v1, step_generation) == 24,
    "Qwen35 DFlash acceptance generation offset drift");
static_assert(offsetof(augur_cuda_qwen35_dflash_acceptance_receipt_v1, plan_digest) == 32,
    "Qwen35 DFlash acceptance digest offset drift");
static_assert(sizeof(augur_cuda_qwen35_dflash_acceptance_args_v1) == 104,
    "Qwen35 DFlash acceptance args ABI drift");
static_assert(alignof(augur_cuda_qwen35_dflash_acceptance_args_v1) == 8,
    "Qwen35 DFlash acceptance args alignment drift");
static_assert(offsetof(augur_cuda_qwen35_dflash_acceptance_args_v1, draft_ids) == 16,
    "Qwen35 DFlash acceptance draft ids offset drift");
static_assert(offsetof(augur_cuda_qwen35_dflash_acceptance_args_v1, target_ids) == 32,
    "Qwen35 DFlash acceptance target ids offset drift");
static_assert(offsetof(augur_cuda_qwen35_dflash_acceptance_args_v1, receipt) == 48,
    "Qwen35 DFlash acceptance receipt offset drift");
static_assert(offsetof(augur_cuda_qwen35_dflash_acceptance_args_v1, plan_digest) == 72,
    "Qwen35 DFlash acceptance digest arg offset drift");
static_assert(sizeof(augur_cuda_qwen35_dflash_logits_acceptance_args_v1) == 128,
    "Qwen35 DFlash logits acceptance args ABI drift");
static_assert(alignof(augur_cuda_qwen35_dflash_logits_acceptance_args_v1) == 8,
    "Qwen35 DFlash logits acceptance args alignment drift");
static_assert(offsetof(augur_cuda_qwen35_dflash_logits_acceptance_args_v1, draft_ids) == 16,
    "Qwen35 DFlash logits acceptance draft ids offset drift");
static_assert(offsetof(augur_cuda_qwen35_dflash_logits_acceptance_args_v1, target_logits) == 32,
    "Qwen35 DFlash logits acceptance target logits offset drift");
static_assert(offsetof(augur_cuda_qwen35_dflash_logits_acceptance_args_v1, target_ids) == 48,
    "Qwen35 DFlash logits acceptance target ids offset drift");
static_assert(offsetof(augur_cuda_qwen35_dflash_logits_acceptance_args_v1, receipt) == 64,
    "Qwen35 DFlash logits acceptance receipt offset drift");
static_assert(offsetof(augur_cuda_qwen35_dflash_logits_acceptance_args_v1, step_generation) == 80,
    "Qwen35 DFlash logits acceptance generation offset drift");
static_assert(offsetof(augur_cuda_qwen35_dflash_logits_acceptance_args_v1, plan_digest) == 96,
    "Qwen35 DFlash logits acceptance digest offset drift");

struct alignas(8) augur_qwen35_dflash_acceptance_digest_v1 {
    uint8_t bytes[32];
};

__host__ __device__ static bool augur_qwen35_dflash_acceptance_nonzero_digest(
    const augur_qwen35_dflash_acceptance_digest_v1& digest
) {
    uint8_t any = 0;
    for (uint32_t index = 0; index < 32; ++index) any |= digest.bytes[index];
    return any != 0;
}

__global__ static void augur_qwen35_dflash_acceptance_kernel(
    const uint32_t* draft_ids,
    const uint32_t* target_ids,
    uint32_t draft_tokens,
    augur_cuda_qwen35_dflash_acceptance_receipt_v1* receipt,
    uint64_t step_generation,
    augur_qwen35_dflash_acceptance_digest_v1 plan_digest
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint32_t accepted = 0;
    uint32_t rejected = draft_tokens;
    uint32_t replacement = UINT32_MAX;
    uint32_t status = AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_STATUS_OK_V1;
    uint32_t error = AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ERROR_NONE_V1;

    for (uint32_t index = 0; index < draft_tokens; ++index) {
        const uint32_t draft = draft_ids[index];
        const uint32_t target = target_ids[index];
        if (draft >= AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_VOCAB_V1
            || target >= AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_VOCAB_V1) {
            status = AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_STATUS_ERROR_V1;
            error = AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ERROR_INVALID_TOKEN_V1;
            rejected = index;
            accepted = 0;
            replacement = UINT32_MAX;
            break;
        }
        if (draft != target) {
            rejected = index;
            replacement = target;
            break;
        }
        accepted = index + 1U;
    }

    receipt->draft_tokens = draft_tokens;
    receipt->accepted_tokens = accepted;
    receipt->rejected_position = rejected;
    receipt->replacement_token = replacement;
    receipt->status = status;
    receipt->error_code = error;
    receipt->step_generation = step_generation;
    for (uint32_t index = 0; index < 32; ++index) {
        receipt->plan_digest[index] = plan_digest.bytes[index];
    }
}

__global__ static void augur_qwen35_dflash_logits_acceptance_kernel(
    const uint32_t* draft_ids,
    const uint32_t* target_ids,
    uint32_t draft_tokens,
    augur_cuda_qwen35_dflash_acceptance_receipt_v1* receipt,
    uint64_t step_generation,
    augur_qwen35_dflash_acceptance_digest_v1 plan_digest
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint32_t accepted = 0;
    uint32_t rejected = draft_tokens;
    uint32_t replacement = UINT32_MAX;
    uint32_t status = AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_STATUS_OK_V1;
    uint32_t error = AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ERROR_NONE_V1;

    for (uint32_t index = 0; index < draft_tokens; ++index) {
        const uint32_t draft = draft_ids[index];
        const uint32_t target = target_ids[index];
        if (draft >= AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_VOCAB_V1) {
            status = AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_STATUS_ERROR_V1;
            error = AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ERROR_INVALID_TOKEN_V1;
            rejected = index;
            accepted = 0;
            replacement = UINT32_MAX;
            break;
        }
        if (target == UINT32_MAX) {
            status = AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_STATUS_ERROR_V1;
            error = AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ERROR_NON_FINITE_LOGIT_V1;
            rejected = index;
            accepted = 0;
            replacement = UINT32_MAX;
            break;
        }
        if (target >= AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_VOCAB_V1) {
            status = AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_STATUS_ERROR_V1;
            error = AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ERROR_INVALID_TOKEN_V1;
            rejected = index;
            accepted = 0;
            replacement = UINT32_MAX;
            break;
        }
        if (draft != target) {
            rejected = index;
            replacement = target;
            break;
        }
        accepted = index + 1U;
    }

    receipt->draft_tokens = draft_tokens;
    receipt->accepted_tokens = accepted;
    receipt->rejected_position = rejected;
    receipt->replacement_token = replacement;
    receipt->status = status;
    receipt->error_code = error;
    receipt->step_generation = step_generation;
    for (uint32_t index = 0; index < 32; ++index) {
        receipt->plan_digest[index] = plan_digest.bytes[index];
    }
}

static __device__ __forceinline__ bool augur_qwen35_dflash_logits_finite(uint32_t bits) {
    return ((bits >> 23U) & 0xffU) != 0xffU;
}

static __device__ __forceinline__ bool augur_qwen35_dflash_logits_better(
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

// One CTA per row, with all 32 warps scanning the full vocabulary.
// Two shuffle reductions replace the old single-lane 256-item final loop.
// Finite logits retain lowest-id ties (including +/-0); any NaN/Inf still
// invalidates the entire row. No extra allocation, scratch ABI or launch.
template<int Threads>
__global__ static void augur_qwen35_dflash_target_argmax_kernel(
    const float* target_logits, uint32_t rows, uint32_t vocab, uint32_t* target_ids
) {
    static_assert(Threads>=32 && Threads<=1024 && Threads%32==0,"whole warps");
    const uint32_t row=blockIdx.x, lane=threadIdx.x&31U, warp=threadIdx.x>>5U;
    if(row>=rows)return;
    __shared__ uint32_t warp_ids[Threads/32],warp_bits[Threads/32],warp_bad[Threads/32];
    uint32_t best_id=UINT32_MAX,best_bits=0,bad=0;
    const float* logits=target_logits+static_cast<size_t>(row)*vocab;
    for(uint32_t index=threadIdx.x;index<vocab;index+=Threads){
        const uint32_t bits=__float_as_uint(logits[index]);
        bad|=!augur_qwen35_dflash_logits_finite(bits);
        if(augur_qwen35_dflash_logits_better(index,bits,best_id,best_bits)) {best_id=index;best_bits=bits;}
    }
    bad=__any_sync(0xffffffffU,bad);
    #pragma unroll
    for(int delta=16;delta>0;delta>>=1){
        const uint32_t id=__shfl_down_sync(0xffffffffU,best_id,delta);
        const uint32_t bits=__shfl_down_sync(0xffffffffU,best_bits,delta);
        if(augur_qwen35_dflash_logits_better(id,bits,best_id,best_bits)){best_id=id;best_bits=bits;}
    }
    if(lane==0){warp_ids[warp]=best_id;warp_bits[warp]=best_bits;warp_bad[warp]=bad;}
    __syncthreads();
    if(warp==0){
        best_id=lane<Threads/32?warp_ids[lane]:UINT32_MAX;
        best_bits=lane<Threads/32?warp_bits[lane]:0;
        bad=lane<Threads/32?warp_bad[lane]:0;
        bad=__any_sync(0xffffffffU,bad);
        #pragma unroll
        for(int delta=16;delta>0;delta>>=1){
            const uint32_t id=__shfl_down_sync(0xffffffffU,best_id,delta);
            const uint32_t bits=__shfl_down_sync(0xffffffffU,best_bits,delta);
            if(augur_qwen35_dflash_logits_better(id,bits,best_id,best_bits)){best_id=id;best_bits=bits;}
        }
        if(lane==0)target_ids[row]=bad?UINT32_MAX:best_id;
    }
}

static augur_cuda_result_v1 augur_qwen35_dflash_acceptance_result(
    augur_cuda_status_t status,
    cudaError_t error
) {
    return augur_cuda_result_v1{status, static_cast<int32_t>(error)};
}

static_assert(sizeof(augur_cuda_qwen35_vocab_topk_args_v1) == 72, "vocab topk ABI size");
static_assert(alignof(augur_cuda_qwen35_vocab_topk_args_v1) == 8, "vocab topk ABI alignment");
static_assert(offsetof(augur_cuda_qwen35_vocab_topk_args_v1, logits) == 24, "vocab topk logits offset");
static_assert(offsetof(augur_cuda_qwen35_vocab_topk_args_v1, scores) == 56, "vocab topk scores offset");

// Integer ordering avoids -use_fast_math/FTZ changing subnormal comparisons.
// Signed zeros are numerically equal; the token-id rule breaks their tie.
static __device__ __forceinline__ uint32_t augur_qwen35_topk_order(uint32_t bits) {
    if ((bits & 0x7fffffffU) == 0) return 0x80000000U;
    return (bits & 0x80000000U) != 0 ? ~bits : (bits ^ 0x80000000U);
}

static __device__ __forceinline__ bool augur_qwen35_topk_better(
    uint32_t a, uint32_t a_bits, uint32_t b, uint32_t b_bits
) {
    if (a == UINT32_MAX) return false;
    if (b == UINT32_MAX) return true;
    const uint32_t ka = augur_qwen35_topk_order(a_bits);
    const uint32_t kb = augur_qwen35_topk_order(b_bits);
    return ka > kb || (ka == kb && a < b);
}

// Bounded reference kernel: repeat a full-row reduction below the previous
// winner's strict ordering key. No input mutation, heap, host sort or scratch.
__global__ static void augur_qwen35_vocab_topk_kernel(
    const float* logits, uint32_t vocab, uint32_t k, uint32_t* out_ids, float* out_scores
) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    __shared__ uint32_t ids[256], values[256], bad[256];
    __shared__ uint32_t previous_id, previous_bits, invalid;
    const float* input = logits + static_cast<size_t>(row) * vocab;
    const size_t output = static_cast<size_t>(row) * k;
    for (uint32_t rank = 0; rank < k; ++rank) {
        uint32_t best_id = UINT32_MAX, best_bits = 0, nonfinite = 0;
        for (uint32_t id = lane; id < vocab; id += blockDim.x) {
            const uint32_t bits = __float_as_uint(input[id]);
            if (!augur_qwen35_dflash_logits_finite(bits)) { nonfinite = 1; continue; }
            if (rank != 0 && !augur_qwen35_topk_better(previous_id, previous_bits, id, bits)) continue;
            if (augur_qwen35_topk_better(id, bits, best_id, best_bits)) {
                best_id = id; best_bits = bits;
            }
        }
        ids[lane] = best_id; values[lane] = best_bits; bad[lane] = nonfinite;
        __syncthreads();
        if (lane == 0) {
            previous_id = UINT32_MAX; previous_bits = 0; invalid = 0;
            for (uint32_t i = 0; i < blockDim.x; ++i) {
                invalid |= bad[i];
                if (augur_qwen35_topk_better(ids[i], values[i], previous_id, previous_bits)) {
                    previous_id = ids[i]; previous_bits = values[i];
                }
            }
            invalid |= previous_id == UINT32_MAX;
            if (invalid != 0) {
                for (uint32_t i = 0; i < k; ++i) {
                    out_ids[output + i] = UINT32_MAX;
                    out_scores[output + i] = __uint_as_float(0x7fc00000U);
                }
            } else {
                out_ids[output + rank] = previous_id;
                out_scores[output + rank] = __uint_as_float(previous_bits);
            }
        }
        __syncthreads();
        if (invalid != 0) return;
    }
}

// K16 draft candidates / K20 target sampling. Each lane retains its best K
// from one vocabulary scan; the global top K is in their union. Merge heads
// with the same integer total order as the reference, preserving score bits.
// One CTA per row, fixed256 threads; no global workspace or input mutation.
template<uint32_t K>
__global__ static void augur_qwen35_vocab_topk_local_kernel(
    const float* logits, uint32_t vocab, uint32_t k, uint32_t* out_ids, float* out_scores
) {
    const uint32_t row = blockIdx.x, lane = threadIdx.x & 31U, warp = threadIdx.x / 32U;
    uint32_t local_ids[K], local_bits[K];
    #pragma unroll
    for (uint32_t p = 0; p < K; ++p) { local_ids[p] = UINT32_MAX; local_bits[p] = 0; }
    const float* input = logits + static_cast<size_t>(row) * vocab;
    uint32_t nonfinite = 0;
    for (uint32_t id = threadIdx.x; id < vocab; id += blockDim.x) {
        const uint32_t bits = __float_as_uint(input[id]);
        if (!augur_qwen35_dflash_logits_finite(bits)) { nonfinite = 1; continue; }
        if (!augur_qwen35_topk_better(id, bits, local_ids[K - 1], local_bits[K - 1])) continue;
        uint32_t carry_id = id, carry_bits = bits;
        #pragma unroll
        for (uint32_t p = 0; p < K; ++p) {
            if (augur_qwen35_topk_better(carry_id, carry_bits, local_ids[p], local_bits[p])) {
                const uint32_t old_id = local_ids[p], old_bits = local_bits[p];
                local_ids[p] = carry_id; local_bits[p] = carry_bits;
                carry_id = old_id; carry_bits = old_bits;
            }
        }
    }
    const size_t output = static_cast<size_t>(row) * k;
    if (__syncthreads_or(nonfinite != 0)) {
        if (threadIdx.x < k) {
            out_ids[output + threadIdx.x] = UINT32_MAX;
            out_scores[output + threadIdx.x] = __uint_as_float(0x7fc00000U);
        }
        return;
    }
    __shared__ uint32_t warp_ids[8], warp_bits[8], winner;
    for (uint32_t rank = 0; rank < k; ++rank) {
        uint32_t id = local_ids[0], bits = local_bits[0];
        #pragma unroll
        for (uint32_t offset = 16; offset != 0; offset >>= 1) {
            const uint32_t other = __shfl_down_sync(0xffffffffU, id, offset);
            const uint32_t value = __shfl_down_sync(0xffffffffU, bits, offset);
            if (augur_qwen35_topk_better(other, value, id, bits)) { id = other; bits = value; }
        }
        if (lane == 0) { warp_ids[warp] = id; warp_bits[warp] = bits; }
        __syncthreads();
        if (warp == 0) {
            id = lane < 8 ? warp_ids[lane] : UINT32_MAX;
            bits = lane < 8 ? warp_bits[lane] : 0;
            #pragma unroll
            for (uint32_t offset = 16; offset != 0; offset >>= 1) {
                const uint32_t other = __shfl_down_sync(0xffffffffU, id, offset);
                const uint32_t value = __shfl_down_sync(0xffffffffU, bits, offset);
                if (augur_qwen35_topk_better(other, value, id, bits)) { id = other; bits = value; }
            }
            if (lane == 0) {
                winner = id;
                out_ids[output + rank] = id;
                out_scores[output + rank] = __uint_as_float(bits);
            }
        }
        __syncthreads();
        if (local_ids[0] == winner) {
            #pragma unroll
            for (uint32_t p = 0; p + 1 < K; ++p) { local_ids[p] = local_ids[p + 1]; local_bits[p] = local_bits[p + 1]; }
            local_ids[K - 1] = UINT32_MAX; local_bits[K - 1] = 0;
        }
    }
}

// Each tile preserves the exact global token-id tie rule and original score
// bits. The global top K is contained in the union of the per-tile top K.
// Scratch belongs to the caller and remains alive through stream completion.
template<uint32_t K>
__global__ static void augur_qwen35_vocab_topk_tiled_kernel(
    const float* logits, const uint32_t* source_ids, uint32_t vocab, uint32_t parts, uint32_t k, uint32_t* out_ids, float* out_scores
) {
    const uint32_t row = blockIdx.x / parts, part = blockIdx.x % parts, lane = threadIdx.x & 31U, warp = threadIdx.x / 32U;
    uint32_t local_ids[K], local_bits[K];
    #pragma unroll
    for (uint32_t p = 0; p < K; ++p) { local_ids[p] = UINT32_MAX; local_bits[p] = 0; }
    const uint32_t span=(vocab+parts-1)/parts, begin=part*span, count=min(span,vocab-begin);
    const size_t base=static_cast<size_t>(row)*vocab+begin;
    const float* input=logits+base;
    uint32_t nonfinite = 0;
    for (uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
        const uint32_t id=source_ids?source_ids[base+index]:begin+index;
        const uint32_t bits = __float_as_uint(input[index]);
        if(id==UINT32_MAX){nonfinite=1;continue;}
        if (!augur_qwen35_dflash_logits_finite(bits)) { nonfinite = 1; continue; }
        if (!augur_qwen35_topk_better(id, bits, local_ids[K - 1], local_bits[K - 1])) continue;
        uint32_t carry_id = id, carry_bits = bits;
        #pragma unroll
        for (uint32_t p = 0; p < K; ++p) {
            if (augur_qwen35_topk_better(carry_id, carry_bits, local_ids[p], local_bits[p])) {
                const uint32_t old_id = local_ids[p], old_bits = local_bits[p];
                local_ids[p] = carry_id; local_bits[p] = carry_bits;
                carry_id = old_id; carry_bits = old_bits;
            }
        }
    }
    const size_t output = (static_cast<size_t>(row)*parts+part)*k;
    if (__syncthreads_or(nonfinite != 0)) {
        if (threadIdx.x < k) {
            out_ids[output + threadIdx.x] = UINT32_MAX;
            out_scores[output + threadIdx.x] = __uint_as_float(0x7fc00000U);
        }
        return;
    }
    __shared__ uint32_t warp_ids[8], warp_bits[8], winner;
    for (uint32_t rank = 0; rank < k; ++rank) {
        uint32_t id = local_ids[0], bits = local_bits[0];
        #pragma unroll
        for (uint32_t offset = 16; offset != 0; offset >>= 1) {
            const uint32_t other = __shfl_down_sync(0xffffffffU, id, offset);
            const uint32_t value = __shfl_down_sync(0xffffffffU, bits, offset);
            if (augur_qwen35_topk_better(other, value, id, bits)) { id = other; bits = value; }
        }
        if (lane == 0) { warp_ids[warp] = id; warp_bits[warp] = bits; }
        __syncthreads();
        if (warp == 0) {
            id = lane < 8 ? warp_ids[lane] : UINT32_MAX;
            bits = lane < 8 ? warp_bits[lane] : 0;
            #pragma unroll
            for (uint32_t offset = 16; offset != 0; offset >>= 1) {
                const uint32_t other = __shfl_down_sync(0xffffffffU, id, offset);
                const uint32_t value = __shfl_down_sync(0xffffffffU, bits, offset);
                if (augur_qwen35_topk_better(other, value, id, bits)) { id = other; bits = value; }
            }
            if (lane == 0) {
                winner = id;
                out_ids[output + rank] = id;
                out_scores[output + rank] = __uint_as_float(bits);
            }
        }
        __syncthreads();
        if (local_ids[0] == winner) {
            #pragma unroll
            for (uint32_t p = 0; p + 1 < K; ++p) { local_ids[p] = local_ids[p + 1]; local_bits[p] = local_bits[p + 1]; }
            local_ids[K - 1] = UINT32_MAX; local_bits[K - 1] = 0;
        }
    }
}


static bool augur_qwen35_dflash_acceptance_range(
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

static bool augur_qwen35_dflash_acceptance_overlap(
    uint64_t left_start,
    uint64_t left_end,
    uint64_t right_start,
    uint64_t right_end
) {
    return left_start < right_end && right_start < left_end;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_acceptance_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_acceptance_args_v1* args
) {
    if (args == nullptr || stream == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_qwen35_dflash_acceptance_result(
            AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args)
        || args->draft_tokens == 0
        || args->draft_tokens > AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS_V1
        || args->draft_ids_len_bytes != static_cast<uint64_t>(args->draft_tokens) * sizeof(uint32_t)
        || args->target_ids_len_bytes != static_cast<uint64_t>(args->draft_tokens) * sizeof(uint32_t)
        || args->receipt_len_bytes != AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES_V1
        || args->step_generation == 0) {
        return augur_qwen35_dflash_acceptance_result(
            AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    augur_qwen35_dflash_acceptance_digest_v1 digest{};
    for (uint32_t index = 0; index < 32; ++index) digest.bytes[index] = args->plan_digest[index];
    if (!augur_qwen35_dflash_acceptance_nonzero_digest(digest)) {
        return augur_qwen35_dflash_acceptance_result(
            AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    uint64_t draft_end = 0;
    uint64_t target_end = 0;
    uint64_t receipt_end = 0;
    if (!augur_qwen35_dflash_acceptance_range(
            args->draft_ids, args->draft_ids_len_bytes, 4, &draft_end)
        || !augur_qwen35_dflash_acceptance_range(
            args->target_ids, args->target_ids_len_bytes, 4, &target_end)
        || !augur_qwen35_dflash_acceptance_range(
            args->receipt, args->receipt_len_bytes, 8, &receipt_end)
        || augur_qwen35_dflash_acceptance_overlap(args->draft_ids, draft_end, args->target_ids, target_end)
        || augur_qwen35_dflash_acceptance_overlap(args->draft_ids, draft_end, args->receipt, receipt_end)
        || augur_qwen35_dflash_acceptance_overlap(args->target_ids, target_end, args->receipt, receipt_end)) {
        return augur_qwen35_dflash_acceptance_result(
            AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) {
        return augur_qwen35_dflash_acceptance_result(
            AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    }
    const auto* draft_ids = reinterpret_cast<const uint32_t*>(
        static_cast<uintptr_t>(args->draft_ids));
    const auto* target_ids = reinterpret_cast<const uint32_t*>(
        static_cast<uintptr_t>(args->target_ids));
    auto* receipt = reinterpret_cast<augur_cuda_qwen35_dflash_acceptance_receipt_v1*>(
        static_cast<uintptr_t>(args->receipt));
    augur_qwen35_dflash_acceptance_kernel<<<1, 1, 0, stream->stream>>>(
        draft_ids, target_ids, args->draft_tokens, receipt, args->step_generation, digest);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_OK, cudaSuccess)
        : augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_vocab_top1_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_vocab_top1_args_v1* args
) {
    if (!args || !stream || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != 1 || args->struct_size != sizeof(*args)
        || args->rows == 0 || args->rows > AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS_V1
        || args->vocab != AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_VOCAB_V1
        || args->logits_len_bytes != static_cast<uint64_t>(args->rows) * args->vocab * 4
        || args->ids_len_bytes != static_cast<uint64_t>(args->rows) * 4) {
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    uint64_t logits_end = 0, ids_end = 0;
    if (!augur_qwen35_dflash_acceptance_range(args->logits, args->logits_len_bytes, 4, &logits_end)
        || !augur_qwen35_dflash_acceptance_range(args->ids, args->ids_len_bytes, 4, &ids_end)
        || augur_qwen35_dflash_acceptance_overlap(args->logits, logits_end, args->ids, ids_end)) {
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) {
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    }
    augur_qwen35_dflash_target_argmax_kernel<1024><<<args->rows, 1024, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->logits)),
        args->rows, args->vocab,
        reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(args->ids)));
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_OK, cudaSuccess)
        : augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_vocab_topk_tiled_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_vocab_topk_args_v1* args,
    uint64_t scratch, uint64_t scratch_len_bytes, uint32_t parts
) {
    if (args == nullptr || stream == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != 1 || args->struct_size != sizeof(*args) || args->reserved != 0
        || args->rows == 0 || args->rows > AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS_V1
        || args->vocab != AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_VOCAB_V1
        || args->top_k == 0 || args->top_k > 20
        || args->logits_len_bytes != static_cast<uint64_t>(args->rows) * args->vocab * 4
        || args->ids_len_bytes != static_cast<uint64_t>(args->rows) * args->top_k * 4
        || args->scores_len_bytes != args->ids_len_bytes) {
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    uint64_t logits_end = 0, ids_end = 0, scores_end = 0;
    if (!augur_qwen35_dflash_acceptance_range(args->logits, args->logits_len_bytes, 4, &logits_end)
        || !augur_qwen35_dflash_acceptance_range(args->ids, args->ids_len_bytes, 4, &ids_end)
        || !augur_qwen35_dflash_acceptance_range(args->scores, args->scores_len_bytes, 4, &scores_end)
        || augur_qwen35_dflash_acceptance_overlap(args->logits, logits_end, args->ids, ids_end)
        || augur_qwen35_dflash_acceptance_overlap(args->logits, logits_end, args->scores, scores_end)
        || augur_qwen35_dflash_acceptance_overlap(args->ids, ids_end, args->scores, scores_end)) {
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if ((args->top_k != 16 && args->top_k != 20)
        || (parts != 8 && parts != 16 && parts != 32 && parts != 64)) {
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t half = static_cast<uint64_t>(args->rows) * parts * args->top_k * 4;
    uint64_t scratch_end = 0;
    if (scratch_len_bytes != half * 2
        || !augur_qwen35_dflash_acceptance_range(scratch, scratch_len_bytes, 4, &scratch_end)
        || augur_qwen35_dflash_acceptance_overlap(scratch, scratch_end, args->logits, logits_end)
        || augur_qwen35_dflash_acceptance_overlap(scratch, scratch_end, args->ids, ids_end)
        || augur_qwen35_dflash_acceptance_overlap(scratch, scratch_end, args->scores, scores_end)) {
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    const float* input = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->logits));
    uint32_t* tile_ids = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(scratch));
    float* tile_scores = reinterpret_cast<float*>(static_cast<uintptr_t>(scratch + half));
    uint32_t* ids = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(args->ids));
    float* scores = reinterpret_cast<float*>(static_cast<uintptr_t>(args->scores));
    if (args->top_k == 20) {
        augur_qwen35_vocab_topk_tiled_kernel<20><<<args->rows * parts,256,0,stream->stream>>>(input,nullptr,args->vocab,parts,20,tile_ids,tile_scores);
    } else {
        augur_qwen35_vocab_topk_tiled_kernel<16><<<args->rows * parts,256,0,stream->stream>>>(input,nullptr,args->vocab,parts,16,tile_ids,tile_scores);
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
    if (args->top_k == 20) {
        augur_qwen35_vocab_topk_tiled_kernel<20><<<args->rows,256,0,stream->stream>>>(tile_scores,tile_ids,parts*20,1,20,ids,scores);
    } else {
        augur_qwen35_vocab_topk_tiled_kernel<16><<<args->rows,256,0,stream->stream>>>(tile_scores,tile_ids,parts*16,1,16,ids,scores);
    }
    error = cudaGetLastError();
    return error == cudaSuccess ? augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_OK,cudaSuccess)
        : augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_vocab_topk_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_vocab_topk_args_v1* args
) {
    if (args == nullptr || stream == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != 1 || args->struct_size != sizeof(*args) || args->reserved != 0
        || args->rows == 0 || args->rows > AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS_V1
        || args->vocab != AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_VOCAB_V1
        || args->top_k == 0 || args->top_k > 20
        || args->logits_len_bytes != static_cast<uint64_t>(args->rows) * args->vocab * 4
        || args->ids_len_bytes != static_cast<uint64_t>(args->rows) * args->top_k * 4
        || args->scores_len_bytes != args->ids_len_bytes) {
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    uint64_t logits_end = 0, ids_end = 0, scores_end = 0;
    if (!augur_qwen35_dflash_acceptance_range(args->logits, args->logits_len_bytes, 4, &logits_end)
        || !augur_qwen35_dflash_acceptance_range(args->ids, args->ids_len_bytes, 4, &ids_end)
        || !augur_qwen35_dflash_acceptance_range(args->scores, args->scores_len_bytes, 4, &scores_end)
        || augur_qwen35_dflash_acceptance_overlap(args->logits, logits_end, args->ids, ids_end)
        || augur_qwen35_dflash_acceptance_overlap(args->logits, logits_end, args->scores, scores_end)
        || augur_qwen35_dflash_acceptance_overlap(args->ids, ids_end, args->scores, scores_end)) {
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) {
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    }
    if (args->rows == 8 && args->top_k == 16) {
        augur_qwen35_vocab_topk_local_kernel<16><<<args->rows, 256, 0, stream->stream>>>(
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->logits)), args->vocab, args->top_k,
            reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(args->ids)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->scores)));
    } else if (args->top_k == 20) {
        augur_qwen35_vocab_topk_local_kernel<20><<<args->rows, 256, 0, stream->stream>>>(
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->logits)), args->vocab, args->top_k,
            reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(args->ids)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->scores)));
    } else {
        augur_qwen35_vocab_topk_kernel<<<args->rows, 256, 0, stream->stream>>>(
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->logits)), args->vocab, args->top_k,
            reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(args->ids)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->scores)));
    }
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_OK, cudaSuccess)
        : augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

static_assert(sizeof(augur_cuda_qwen35_dflash_selector_args_v1) == 176, "selector ABI size");
static_assert(alignof(augur_cuda_qwen35_dflash_selector_args_v1) == 8, "selector ABI alignment");
static_assert(offsetof(augur_cuda_qwen35_dflash_selector_args_v1, hidden_rows) == 32, "selector hidden offset");
static_assert(offsetof(augur_cuda_qwen35_dflash_selector_args_v1, path_ids) == 160, "selector path offset");

__global__ static void augur_qwen35_dflash_selector_gate_kernel(
    const float* hidden, const __half* weight, float* gates
) {
    const uint32_t lane = threadIdx.x;
    const uint32_t feature = blockIdx.x;
    const uint32_t row = blockIdx.y;
    __shared__ float partial[256];
    float sum = 0.0f;
    for (uint32_t col = lane; col < 5120; col += 256) {
        sum = __fadd_rn(sum, __fmul_rn(hidden[row * 5120 + col], __half2float(weight[feature * 5120 + col])));
    }
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t step = 128; step != 0; step >>= 1) {
        if (lane < step) partial[lane] = __fadd_rn(partial[lane], partial[lane + step]);
        __syncthreads();
    }
    if (lane == 0) gates[row * 256 + feature] = partial[0];
}

template<bool Q8> __device__ static float augur_qwen35_selector_codebook(const void* base, size_t index);
template<> __device__ float augur_qwen35_selector_codebook<false>(const void* base, size_t index) {
    return __half2float(static_cast<const __half*>(base)[index]);
}
template<> __device__ float augur_qwen35_selector_codebook<true>(const void* base, size_t index) {
    const auto* block = static_cast<const uint8_t*>(base) + (index / 32) * 34;
    const float scale = __half2float(*reinterpret_cast<const __half*>(block));
    const float code = static_cast<float>(reinterpret_cast<const int8_t*>(block + 2)[index % 32]);
    return __fmul_rn(scale, code);
}

// Counter domains match augur_speculative_sampling.rs. Proposal never shares
// its random draw with acceptance, correction or the target bonus token.
__device__ static double augur_qwen35_proposal_uniform(uint64_t position) {
    uint64_t x = UINT64_C(0x445241465450524f) + (position + 1) * UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return double(x >> 11) * (1.0 / 9007199254740992.0);
}

// Per-row proposal calibration passed by value (no device memory). Row 0 is the anchor.
struct augur_qwen35_proposal_calibration { float temperature[8]; float pair_scale[8]; };

template<bool Q8, bool SCALED = false, bool TAIL_SCALED = false, bool RANK_COMPENSATED = false, bool EARLY_SCALED = false, bool POSITION_SCALED = false, bool BF16_CODEBOOK = false, bool PHASE_GATES = false, bool PROBABILITY = false, bool CALIBRATED = false> __global__ static void augur_qwen35_dflash_selector_walk_kernel(
    const uint32_t* candidates, const float* unary, const float* gates,
    const void* predecessor, const void* successor, uint32_t anchor,
    float* transitions, uint32_t* path, float transition_scale = 1.0f, float rank_bonus = 0.0f,
    float second_scale = 1.0f, float third_scale = 1.0f,
    const float* after_gates = nullptr, uint32_t close_token = UINT32_MAX,
    double* proposal_q = nullptr, uint64_t absolute_start = 0, uint32_t probability_positions = 0,
    augur_qwen35_proposal_calibration calibration = {}
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t candidate = tid / 16, lane = tid % 16;
    __shared__ float partial[256], scores[16];
    __shared__ uint32_t bad[256], invalid, chosen[8], previous, after_close;
    if (tid < 8) { path[tid] = UINT32_MAX; chosen[tid] = UINT32_MAX; }
    if (tid < 128) transitions[tid] = __uint_as_float(0x7fc00000U);
    if constexpr (PROBABILITY) { if (tid < 128) proposal_q[tid] = 0.0; }
    uint32_t local_bad = 0;
    for (uint32_t i = tid; i < 2048; i += 256) {
        local_bad |= !augur_qwen35_dflash_logits_finite(__float_as_uint(gates[i]));
        if constexpr (PHASE_GATES) local_bad |= !augur_qwen35_dflash_logits_finite(__float_as_uint(after_gates[i]));
    }
    if (tid < 128) {
        local_bad |= candidates[tid] >= 248320;
        local_bad |= !augur_qwen35_dflash_logits_finite(__float_as_uint(unary[tid]));
        const uint32_t base = (tid / 16) * 16;
        for (uint32_t i = base; i < tid; ++i) local_bad |= candidates[i] == candidates[tid];
        if (tid != base) {
            local_bad |= !augur_qwen35_topk_better(candidates[tid-1], __float_as_uint(unary[tid-1]),
                                               candidates[tid], __float_as_uint(unary[tid]));
        }
    }
    bad[tid] = local_bad;
    __syncthreads();
    if (tid == 0) {
        invalid = 0;
        for (uint32_t i = 0; i < 256; ++i) invalid |= bad[i];
        previous = anchor; chosen[0] = anchor; after_close = PHASE_GATES && anchor == close_token;
    }
    __syncthreads();
    if (invalid != 0) return;
    if (tid < 16) transitions[tid] = 0.0f; // anchor row is not a draft position
    for (uint32_t row = 1; row < 8; ++row) {
        float dot = 0.0f;
        local_bad = 0;
        const uint32_t next = candidates[row * 16 + candidate];
        for (uint32_t r = lane; r < 256; r += 16) {
            const float p = BF16_CODEBOOK
                ? __uint_as_float(uint32_t(static_cast<const uint16_t*>(predecessor)[static_cast<size_t>(previous)*256+r])<<16)
                : augur_qwen35_selector_codebook<Q8>(predecessor, static_cast<size_t>(previous) * 256 + r);
            const float s = BF16_CODEBOOK
                ? __uint_as_float(uint32_t(static_cast<const uint16_t*>(successor)[static_cast<size_t>(next)*256+r])<<16)
                : augur_qwen35_selector_codebook<Q8>(successor, static_cast<size_t>(next) * 256 + r);
            local_bad |= !augur_qwen35_dflash_logits_finite(__float_as_uint(p));
            local_bad |= !augur_qwen35_dflash_logits_finite(__float_as_uint(s));
            const float g = PHASE_GATES && after_close ? after_gates[row * 256 + r] : gates[row * 256 + r];
            const float conditioned = __fmul_rn(p, g);
            dot = __fadd_rn(dot, __fmul_rn(conditioned, s));
        }
        partial[tid] = dot; bad[tid] = local_bad;
        __syncthreads();
        for (uint32_t step = 8; step != 0; step >>= 1) {
            if (lane < step) partial[tid] = __fadd_rn(partial[tid], partial[tid + step]);
            __syncthreads();
        }
        if (lane == 0) {
            // The default instantiation retains the original arithmetic exactly.
            float pair_score = partial[tid];
            if constexpr (CALIBRATED) {
                // Proposal-only conditioning; FP32 gate weights are unchanged.
                // Scale 1 is an exact identity, so v2 rows outside 2-4 stay bit-identical.
                pair_score = __fmul_rn(pair_score, calibration.pair_scale[row]);
            }
            if constexpr (SCALED) {
                const float row_scale = POSITION_SCALED
                    ? (row == 1 ? transition_scale : row == 2 ? second_scale : row == 3 ? third_scale : 1.0f)
                    : (EARLY_SCALED
                        ? ((row <= 3) ? transition_scale : 1.0f)
                        : ((TAIL_SCALED && row <= 2) ? 1.0f : transition_scale));
                pair_score = __fmul_rn(pair_score, row_scale);
            }
            float score = __fadd_rn(pair_score, unary[row * 16 + candidate]);
            if constexpr (RANK_COMPENSATED) {
                // Candidate rows are sorted by unary score. A positive bonus
                // deliberately softens the top-1 bias by rewarding later
                // ranked candidates. Tail calibration leaves the first two
                // positions untouched so early path stability is preserved.
                const float active_bonus = EARLY_SCALED
                    ? ((row <= 3) ? rank_bonus : 0.0f)
                    : ((TAIL_SCALED && row <= 2) ? 0.0f : rank_bonus);
                score = __fadd_rn(score, __fmul_rn(active_bonus, static_cast<float>(candidate)));
            }
            scores[candidate] = score;
            transitions[row * 16 + candidate] = scores[candidate];
        }
        __syncthreads();
        if (tid == 0) {
            for (uint32_t i = 0; i < 256; ++i) invalid |= bad[i];
            uint32_t best = 0;
            for (uint32_t i = 0; i < 16; ++i) {
                invalid |= !augur_qwen35_dflash_logits_finite(__float_as_uint(scores[i]));
                // std::max_element semantics: first candidate wins finite ties.
                if (scores[i] > scores[best]) best = i;
            }
            if constexpr (PROBABILITY) {
                if (invalid == 0 && row <= probability_positions) {
                    double weights[16], total = 0.0;
                    const double maximum = double(scores[best]);
                    const double temperature = CALIBRATED ? double(calibration.temperature[row]) : 1.0;
                    for (uint32_t i=0; i<16; ++i) { weights[i]=exp((double(scores[i])-maximum)/temperature); total+=weights[i]; }
                    const double u = augur_qwen35_proposal_uniform(absolute_start + row);
                    double cumulative=0.0; uint32_t pick=best; bool found=false;
                    for (uint32_t i=0; i<16; ++i) {
                        const double q=weights[i]/total; proposal_q[row*16+i]=q;
                        cumulative+=q;
                        if (!found && u<cumulative) { pick=i; found=true; }
                    }
                    if (!found) { for(uint32_t i=0;i<16;++i) if(weights[i]>0.0) pick=i; }
                    best=pick;
                } else if (invalid == 0) { proposal_q[row*16+best]=1.0; }
            }
            previous = candidates[row * 16 + best]; chosen[row] = previous;
            if constexpr (PHASE_GATES) after_close |= previous == close_token;
        }
        __syncthreads();
        if (invalid != 0) return; // no partially selected path is published
    }
    if (tid < 8) path[tid] = chosen[tid];
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_selector_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_dflash_selector_args_v1* a
) {
    if (a == nullptr || stream == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || a->abi_version != 1 || a->struct_size != sizeof(*a)
        || a->rows != 8 || a->top_k != 16 || a->rank != 256 || a->hidden != 5120
        || a->vocab != 248320 || a->anchor_token >= a->vocab) {
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t ptrs[9] = { a->hidden_rows, a->gate_weight, a->predecessor, a->successor,
        a->candidate_ids, a->unary_scores, a->gates, a->transition_scores, a->path_ids };
    const uint64_t lens[9] = { a->hidden_rows_len_bytes, a->gate_weight_len_bytes,
        a->predecessor_len_bytes, a->successor_len_bytes, a->candidate_ids_len_bytes,
        a->unary_scores_len_bytes, a->gates_len_bytes, a->transition_scores_len_bytes, a->path_ids_len_bytes };
    const uint64_t expected[9] = { 8*5120*4, 256*5120*2, 248320*256*2, 248320*256*2,
        8*16*4, 8*16*4, 8*256*4, 8*16*4, 8*4 };
    uint64_t ends[9] = {};
    for (uint32_t i = 0; i < 9; ++i) {
        const uint64_t alignment = i >= 1 && i <= 3 ? 2 : 4;
        if (lens[i] != expected[i] || !augur_qwen35_dflash_acceptance_range(ptrs[i], lens[i], alignment, &ends[i])) {
            return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (augur_qwen35_dflash_acceptance_overlap(ptrs[i], ends[i], ptrs[j], ends[j])) {
                return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    augur_qwen35_dflash_selector_gate_kernel<<<dim3(256,8),256,0,stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->hidden_rows)),
        reinterpret_cast<const __half*>(static_cast<uintptr_t>(a->gate_weight)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(a->gates)));
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    augur_qwen35_dflash_selector_walk_kernel<false><<<1,256,0,stream->stream>>>(
        reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a->candidate_ids)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->unary_scores)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->gates)),
        reinterpret_cast<const __half*>(static_cast<uintptr_t>(a->predecessor)),
        reinterpret_cast<const __half*>(static_cast<uintptr_t>(a->successor)), a->anchor_token,
        reinterpret_cast<float*>(static_cast<uintptr_t>(a->transition_scores)),
        reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a->path_ids)));
    error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_OK, cudaSuccess)
        : augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

static_assert(sizeof(augur_cuda_qwen35_dflash_q8_selector_args_v1) == 144, "Q8 selector ABI size");
static_assert(offsetof(augur_cuda_qwen35_dflash_q8_selector_args_v1, predecessor) == 32, "Q8 selector input offset");
static_assert(offsetof(augur_cuda_qwen35_dflash_q8_selector_args_v1, path_ids) == 128, "Q8 selector output offset");

// Opt-in causal dynamic-programming selector. Unlike the historical greedy
// walk, each row scores all 16 predecessor candidates and retains the best
// cumulative path. The descriptor, candidate tensors and output path remain
// unchanged; this is deliberately a separate entry point so the default
// greedy kernel is bit-for-bit untouched.
__global__ static void augur_qwen35_dflash_selector_dp_kernel(
    const uint32_t* candidates, const float* unary, const float* gates,
    const void* predecessor, const void* successor, uint32_t anchor,
    float* transitions, uint32_t* path, float transition_scale
) {
    const uint32_t tid = threadIdx.x;
    __shared__ float pair_scores[256], dp_prev[16], dp_next[16];
    __shared__ uint32_t bad[256], invalid, chosen[8];
    __shared__ uint8_t back[6 * 16]; // rows 2..7, predecessor index
    if (tid < 8) { path[tid] = UINT32_MAX; chosen[tid] = UINT32_MAX; }
    uint32_t local_bad = 0;
    for (uint32_t i = tid; i < 8 * 256; i += 256) {
        local_bad |= !augur_qwen35_dflash_logits_finite(__float_as_uint(gates[i]));
    }
    if (tid < 128) {
        local_bad |= candidates[tid] >= 248320;
        local_bad |= !augur_qwen35_dflash_logits_finite(__float_as_uint(unary[tid]));
        const uint32_t base = (tid / 16) * 16;
        for (uint32_t i = base; i < tid; ++i) local_bad |= candidates[i] == candidates[tid];
        if (tid != base) {
            local_bad |= !augur_qwen35_topk_better(candidates[tid - 1],
                __float_as_uint(unary[tid - 1]), candidates[tid], __float_as_uint(unary[tid]));
        }
    }
    bad[tid] = local_bad;
    __syncthreads();
    if (tid == 0) {
        invalid = 0;
        for (uint32_t i = 0; i < 256; ++i) invalid |= bad[i];
        chosen[0] = anchor;
    }
    __syncthreads();
    if (invalid != 0) return;
    if (tid < 16) transitions[tid] = 0.0f;

    // Row 1 is conditioned on the authenticated target anchor.
    local_bad = 0;
    if (tid < 16) {
        float dot = 0.0f;
        const uint32_t next = candidates[16 + tid];
        for (uint32_t r = 0; r < 256; ++r) {
            const float p = augur_qwen35_selector_codebook<true>(predecessor,
                static_cast<size_t>(anchor) * 256 + r);
            const float g = gates[256 + r];
            const float s = augur_qwen35_selector_codebook<true>(successor,
                static_cast<size_t>(next) * 256 + r);
            local_bad |= !augur_qwen35_dflash_logits_finite(__float_as_uint(p));
            local_bad |= !augur_qwen35_dflash_logits_finite(__float_as_uint(g));
            local_bad |= !augur_qwen35_dflash_logits_finite(__float_as_uint(s));
            dot = __fadd_rn(dot, __fmul_rn(__fmul_rn(p, g), s));
        }
        const float local = __fadd_rn(unary[16 + tid], __fmul_rn(dot, transition_scale));
        local_bad |= !augur_qwen35_dflash_logits_finite(__float_as_uint(local));
        dp_prev[tid] = local;
        transitions[16 + tid] = local;
    }
    bad[tid] = (tid < 16) ? local_bad : 0;
    __syncthreads();
    if (tid == 0) { for (uint32_t i = 0; i < 256; ++i) invalid |= bad[i]; }
    __syncthreads();
    if (invalid != 0) return;

    for (uint32_t row = 2; row < 8; ++row) {
        const uint32_t previous_index = tid / 16;
        const uint32_t next_index = tid % 16;
        const uint32_t previous = candidates[(row - 1) * 16 + previous_index];
        const uint32_t next = candidates[row * 16 + next_index];
        float dot = 0.0f;
        local_bad = 0;
        for (uint32_t r = 0; r < 256; ++r) {
            const float p = augur_qwen35_selector_codebook<true>(predecessor,
                static_cast<size_t>(previous) * 256 + r);
            const float g = gates[row * 256 + r];
            const float s = augur_qwen35_selector_codebook<true>(successor,
                static_cast<size_t>(next) * 256 + r);
            local_bad |= !augur_qwen35_dflash_logits_finite(__float_as_uint(p));
            local_bad |= !augur_qwen35_dflash_logits_finite(__float_as_uint(g));
            local_bad |= !augur_qwen35_dflash_logits_finite(__float_as_uint(s));
            dot = __fadd_rn(dot, __fmul_rn(__fmul_rn(p, g), s));
        }
        const float pair = __fadd_rn(dp_prev[previous_index],
            __fadd_rn(unary[row * 16 + next_index], __fmul_rn(dot, transition_scale)));
        local_bad |= !augur_qwen35_dflash_logits_finite(__float_as_uint(pair));
        pair_scores[tid] = pair;
        bad[tid] = local_bad;
        __syncthreads();
        if (tid == 0) { for (uint32_t i = 0; i < 256; ++i) invalid |= bad[i]; }
        __syncthreads();
        if (invalid != 0) return;
        if (tid < 16) {
            uint32_t best = 0;
            for (uint32_t previous_candidate = 1; previous_candidate < 16; ++previous_candidate) {
                if (pair_scores[previous_candidate * 16 + tid] > pair_scores[best * 16 + tid]) {
                    best = previous_candidate;
                }
            }
            dp_next[tid] = pair_scores[best * 16 + tid];
            transitions[row * 16 + tid] = __fsub_rn(dp_next[tid], dp_prev[best]);
            back[(row - 2) * 16 + tid] = static_cast<uint8_t>(best);
        }
        __syncthreads();
        if (tid < 16) dp_prev[tid] = dp_next[tid];
        __syncthreads();
    }
    if (tid == 0) {
        uint32_t best = 0;
        for (uint32_t candidate = 1; candidate < 16; ++candidate) {
            if (dp_prev[candidate] > dp_prev[best]) best = candidate;
        }
        chosen[7] = best;
        for (int row = 7; row >= 2; --row) {
            chosen[row - 1] = back[(row - 2) * 16 + chosen[row]];
        }
        path[0] = anchor;
        for (uint32_t row = 1; row < 8; ++row) {
            path[row] = candidates[row * 16 + chosen[row]];
        }
    }
}

static augur_cuda_result_v1 augur_qwen35_dflash_q8_selector_dp_submit(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_q8_selector_args_v1* a, float transition_scale
) {
    if (!(transition_scale >= 0.0f && transition_scale <= 2.0f)
        || !a || !stream || ordinal < 0 || stream->ordinal != ordinal
        || a->abi_version != 1 || a->struct_size != sizeof(*a) || a->reserved != 0
        || a->rows != 8 || a->top_k != 16 || a->rank != 256 || a->vocab != 248320
        || a->anchor_token >= a->vocab) {
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t ptrs[7] = { a->predecessor, a->successor, a->candidate_ids,
        a->unary_scores, a->gates, a->transition_scores, a->path_ids };
    const uint64_t lens[7] = { a->predecessor_len_bytes, a->successor_len_bytes,
        a->candidate_ids_len_bytes, a->unary_scores_len_bytes, a->gates_len_bytes,
        a->transition_scores_len_bytes, a->path_ids_len_bytes };
    const uint64_t expected[7] = { 248320*8*34, 248320*8*34, 8*16*4, 8*16*4,
        8*256*4, 8*16*4, 8*4 };
    uint64_t ends[7] = {};
    for (uint32_t i = 0; i < 7; ++i) {
        if (lens[i] != expected[i] || !augur_qwen35_dflash_acceptance_range(ptrs[i], lens[i], i < 2 ? 2 : 4, &ends[i])) {
            return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (augur_qwen35_dflash_acceptance_overlap(ptrs[i], ends[i], ptrs[j], ends[j])) {
                return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    augur_qwen35_dflash_selector_dp_kernel<<<1,256,0,stream->stream>>>(
        reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a->candidate_ids)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->unary_scores)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->gates)),
        reinterpret_cast<const void*>(static_cast<uintptr_t>(a->predecessor)),
        reinterpret_cast<const void*>(static_cast<uintptr_t>(a->successor)), a->anchor_token,
        reinterpret_cast<float*>(static_cast<uintptr_t>(a->transition_scores)),
        reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a->path_ids)), transition_scale);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_OK, cudaSuccess)
        : augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

static augur_cuda_result_v1 augur_qwen35_dflash_q8_selector_submit_scaled(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_dflash_q8_selector_args_v1* a,
    float transition_scale, bool rank_compensated = false, float rank_bonus = 0.0f
) {
    if (!(transition_scale >= 0.0f && transition_scale <= 2.0f)
        || (rank_compensated && !(rank_bonus >= 0.0f && rank_bonus <= 2.0f))
        || !a || !stream || ordinal < 0 || stream->ordinal != ordinal
        || a->abi_version != 1 || a->struct_size != sizeof(*a) || a->reserved != 0
        || a->rows != 8 || a->top_k != 16 || a->rank != 256 || a->vocab != 248320
        || a->anchor_token >= a->vocab) {
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t ptrs[7] = { a->predecessor, a->successor, a->candidate_ids,
        a->unary_scores, a->gates, a->transition_scores, a->path_ids };
    const uint64_t lens[7] = { a->predecessor_len_bytes, a->successor_len_bytes,
        a->candidate_ids_len_bytes, a->unary_scores_len_bytes, a->gates_len_bytes,
        a->transition_scores_len_bytes, a->path_ids_len_bytes };
    const uint64_t expected[7] = { 248320*8*34, 248320*8*34, 8*16*4, 8*16*4, 8*256*4, 8*16*4, 8*4 };
    uint64_t ends[7] = {};
    for (uint32_t i = 0; i < 7; ++i) {
        if (lens[i] != expected[i] || !augur_qwen35_dflash_acceptance_range(ptrs[i], lens[i], i < 2 ? 2 : 4, &ends[i])) {
            return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (augur_qwen35_dflash_acceptance_overlap(ptrs[i], ends[i], ptrs[j], ends[j])) {
                return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    if (rank_compensated) {
        augur_qwen35_dflash_selector_walk_kernel<true, false, false, true><<<1,256,0,stream->stream>>>(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a->candidate_ids)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->unary_scores)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->gates)),
            reinterpret_cast<const void*>(static_cast<uintptr_t>(a->predecessor)),
            reinterpret_cast<const void*>(static_cast<uintptr_t>(a->successor)), a->anchor_token,
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->transition_scores)),
            reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a->path_ids)), 1.0f, rank_bonus);
    } else if (transition_scale == 1.0f) {
        augur_qwen35_dflash_selector_walk_kernel<true><<<1,256,0,stream->stream>>>(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a->candidate_ids)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->unary_scores)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->gates)),
            reinterpret_cast<const void*>(static_cast<uintptr_t>(a->predecessor)),
            reinterpret_cast<const void*>(static_cast<uintptr_t>(a->successor)), a->anchor_token,
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->transition_scores)),
            reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a->path_ids)));
    } else {
        augur_qwen35_dflash_selector_walk_kernel<true, true><<<1,256,0,stream->stream>>>(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a->candidate_ids)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->unary_scores)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->gates)),
            reinterpret_cast<const void*>(static_cast<uintptr_t>(a->predecessor)),
            reinterpret_cast<const void*>(static_cast<uintptr_t>(a->successor)), a->anchor_token,
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->transition_scores)),
            reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a->path_ids)), transition_scale);
    }
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_OK, cudaSuccess)
        : augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}



extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_phase_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_dflash_q8_selector_args_v1* a,
    uint64_t after_gates, uint64_t after_bytes, uint32_t close_token
) {
    if (close_token >= 248320 || after_bytes != 8*256*4
        || !a || !stream || ordinal < 0 || stream->ordinal != ordinal
        || a->abi_version != 1 || a->struct_size != sizeof(*a) || a->reserved != 0
        || a->rows != 8 || a->top_k != 16 || a->rank != 256 || a->vocab != 248320
        || a->anchor_token >= a->vocab) {
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t ptrs[7] = { a->predecessor, a->successor, a->candidate_ids,
        a->unary_scores, a->gates, a->transition_scores, a->path_ids };
    const uint64_t lens[7] = { a->predecessor_len_bytes, a->successor_len_bytes,
        a->candidate_ids_len_bytes, a->unary_scores_len_bytes, a->gates_len_bytes,
        a->transition_scores_len_bytes, a->path_ids_len_bytes };
    const uint64_t expected[7] = { 248320*8*34, 248320*8*34, 8*16*4, 8*16*4, 8*256*4, 8*16*4, 8*4 };
    uint64_t ends[7] = {};
    for (uint32_t i = 0; i < 7; ++i) {
        if (lens[i] != expected[i] || !augur_qwen35_dflash_acceptance_range(ptrs[i], lens[i], i < 2 ? 2 : 4, &ends[i])) {
            return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (augur_qwen35_dflash_acceptance_overlap(ptrs[i], ends[i], ptrs[j], ends[j])) {
                return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);

    uint64_t after_end = 0;
    if (!augur_qwen35_dflash_acceptance_range(after_gates,after_bytes,4,&after_end))
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    for (uint32_t i=0;i<7;++i) if (augur_qwen35_dflash_acceptance_overlap(after_gates,after_end,ptrs[i],ends[i]))
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    augur_qwen35_dflash_selector_walk_kernel<true,false,false,false,false,false,false,true><<<1,256,0,stream->stream>>>(
        reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a->candidate_ids)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->unary_scores)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->gates)),
        reinterpret_cast<const void*>(static_cast<uintptr_t>(a->predecessor)),
        reinterpret_cast<const void*>(static_cast<uintptr_t>(a->successor)),a->anchor_token,
        reinterpret_cast<float*>(static_cast<uintptr_t>(a->transition_scores)),
        reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a->path_ids)),1.0f,0.0f,1.0f,1.0f,
        reinterpret_cast<const float*>(static_cast<uintptr_t>(after_gates)),close_token);
    const cudaError_t error=cudaGetLastError();
    return augur_qwen35_dflash_acceptance_result(error==cudaSuccess?AUGUR_CUDA_STATUS_OK:AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
}

static augur_cuda_result_v1 augur_qwen35_dflash_q8_selector_tail_scaled_submit(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_q8_selector_args_v1* a, float transition_scale,
    bool rank_compensated = false, float rank_bonus = 0.0f, bool early_scaled = false,
    bool position_scaled = false, float second_scale = 1.0f, float third_scale = 1.0f
) {
    if ((position_scaled && (!(second_scale >= 0.0f && second_scale <= 2.0f)
            || !(third_scale >= 0.0f && third_scale <= 2.0f)))
        || !(transition_scale >= 0.0f && transition_scale <= 2.0f)
        || (rank_compensated && !(rank_bonus >= 0.0f && rank_bonus <= 2.0f))
        || !a || !stream || ordinal < 0 || stream->ordinal != ordinal
        || a->abi_version != 1 || a->struct_size != sizeof(*a) || a->reserved != 0
        || a->rows != 8 || a->top_k != 16 || a->rank != 256 || a->vocab != 248320
        || a->anchor_token >= a->vocab) {
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t ptrs[7] = { a->predecessor, a->successor, a->candidate_ids,
        a->unary_scores, a->gates, a->transition_scores, a->path_ids };
    const uint64_t lens[7] = { a->predecessor_len_bytes, a->successor_len_bytes,
        a->candidate_ids_len_bytes, a->unary_scores_len_bytes, a->gates_len_bytes,
        a->transition_scores_len_bytes, a->path_ids_len_bytes };
    const uint64_t expected[7] = { 248320*8*34, 248320*8*34, 8*16*4, 8*16*4,
        8*256*4, 8*16*4, 8*4 };
    uint64_t ends[7] = {};
    for (uint32_t i = 0; i < 7; ++i) {
        if (lens[i] != expected[i] || !augur_qwen35_dflash_acceptance_range(ptrs[i], lens[i], i < 2 ? 2 : 4, &ends[i])) {
            return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (augur_qwen35_dflash_acceptance_overlap(ptrs[i], ends[i], ptrs[j], ends[j])) {
                return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    if (position_scaled) {
        // One fused walk; three launch scalars, no new buffer, copy, or kernel.
        augur_qwen35_dflash_selector_walk_kernel<true, true, false, false, false, true><<<1,256,0,stream->stream>>>(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a->candidate_ids)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->unary_scores)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->gates)),
            reinterpret_cast<const void*>(static_cast<uintptr_t>(a->predecessor)),
            reinterpret_cast<const void*>(static_cast<uintptr_t>(a->successor)), a->anchor_token,
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->transition_scores)),
            reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a->path_ids)),
            transition_scale, 0.0f, second_scale, third_scale);
    } else if (early_scaled) {
        if (rank_compensated) {
            augur_qwen35_dflash_selector_walk_kernel<true, true, false, true, true><<<1,256,0,stream->stream>>>(
                reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a->candidate_ids)),
                reinterpret_cast<const float*>(static_cast<uintptr_t>(a->unary_scores)),
                reinterpret_cast<const float*>(static_cast<uintptr_t>(a->gates)),
                reinterpret_cast<const void*>(static_cast<uintptr_t>(a->predecessor)),
                reinterpret_cast<const void*>(static_cast<uintptr_t>(a->successor)), a->anchor_token,
                reinterpret_cast<float*>(static_cast<uintptr_t>(a->transition_scores)),
                reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a->path_ids)), transition_scale, rank_bonus);
        } else {
            augur_qwen35_dflash_selector_walk_kernel<true, true, false, false, true><<<1,256,0,stream->stream>>>(
                reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a->candidate_ids)),
                reinterpret_cast<const float*>(static_cast<uintptr_t>(a->unary_scores)),
                reinterpret_cast<const float*>(static_cast<uintptr_t>(a->gates)),
                reinterpret_cast<const void*>(static_cast<uintptr_t>(a->predecessor)),
                reinterpret_cast<const void*>(static_cast<uintptr_t>(a->successor)), a->anchor_token,
                reinterpret_cast<float*>(static_cast<uintptr_t>(a->transition_scores)),
                reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a->path_ids)), transition_scale);
        }
    } else if (rank_compensated) {
        augur_qwen35_dflash_selector_walk_kernel<true, true, true, true><<<1,256,0,stream->stream>>>(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a->candidate_ids)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->unary_scores)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->gates)),
            reinterpret_cast<const void*>(static_cast<uintptr_t>(a->predecessor)),
            reinterpret_cast<const void*>(static_cast<uintptr_t>(a->successor)), a->anchor_token,
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->transition_scores)),
            reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a->path_ids)), transition_scale, rank_bonus);
    } else {
        augur_qwen35_dflash_selector_walk_kernel<true, true, true><<<1,256,0,stream->stream>>>(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a->candidate_ids)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->unary_scores)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->gates)),
            reinterpret_cast<const void*>(static_cast<uintptr_t>(a->predecessor)),
            reinterpret_cast<const void*>(static_cast<uintptr_t>(a->successor)), a->anchor_token,
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->transition_scores)),
            reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a->path_ids)), transition_scale);
    }
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_OK, cudaSuccess)
        : augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_position_scaled_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_dflash_q8_selector_args_v1* a,
    float first_scale, float second_scale, float third_scale
) { return augur_qwen35_dflash_q8_selector_tail_scaled_submit(ordinal, stream, a, first_scale,
        false, 0.0f, false, true, second_scale, third_scale); }

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_early_scaled_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_dflash_q8_selector_args_v1* a, float transition_scale
) { return augur_qwen35_dflash_q8_selector_tail_scaled_submit(ordinal, stream, a, transition_scale, false, 0.0f, true); }

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_dflash_q8_selector_args_v1* a
) { return augur_qwen35_dflash_q8_selector_submit_scaled(ordinal, stream, a, 1.0f); }

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_scaled_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_dflash_q8_selector_args_v1* a, float transition_scale
) { return augur_qwen35_dflash_q8_selector_submit_scaled(ordinal, stream, a, transition_scale); }

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_rank_comp_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_dflash_q8_selector_args_v1* a, float rank_bonus
) { return augur_qwen35_dflash_q8_selector_submit_scaled(ordinal, stream, a, 1.0f, true, rank_bonus); }

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_dp_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_dflash_q8_selector_args_v1* a, float transition_scale
) { return augur_qwen35_dflash_q8_selector_dp_submit(ordinal, stream, a, transition_scale); }

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_tail_scaled_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_dflash_q8_selector_args_v1* a, float transition_scale
) { return augur_qwen35_dflash_q8_selector_tail_scaled_submit(ordinal, stream, a, transition_scale); }

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_tail_rank_comp_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_dflash_q8_selector_args_v1* a,
    float transition_scale, float rank_bonus
) { return augur_qwen35_dflash_q8_selector_tail_scaled_submit(ordinal, stream, a, transition_scale, true, rank_bonus); }

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_logits_acceptance_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_logits_acceptance_args_v1* args
) {
    if (args == nullptr || stream == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_qwen35_dflash_acceptance_result(
            AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != AUGUR_CUDA_QWEN35_DFLASH_LOGITS_ACCEPTANCE_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args)
        || args->draft_tokens == 0
        || args->draft_tokens > AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS_V1
        || args->vocab != AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_VOCAB_V1
        || args->draft_ids_len_bytes != static_cast<uint64_t>(args->draft_tokens) * sizeof(uint32_t)
        || args->target_logits_len_bytes
            != static_cast<uint64_t>(args->draft_tokens) * args->vocab * sizeof(float)
        || args->target_ids_len_bytes != static_cast<uint64_t>(args->draft_tokens) * sizeof(uint32_t)
        || args->receipt_len_bytes != AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES_V1
        || args->step_generation == 0) {
        return augur_qwen35_dflash_acceptance_result(
            AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    uint8_t any = 0;
    for (uint32_t index = 0; index < 32; ++index) any |= args->plan_digest[index];
    if (any == 0) {
        return augur_qwen35_dflash_acceptance_result(
            AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    uint64_t draft_end = 0;
    uint64_t logits_end = 0;
    uint64_t target_end = 0;
    uint64_t receipt_end = 0;
    if (!augur_qwen35_dflash_acceptance_range(
            args->draft_ids, args->draft_ids_len_bytes, 4, &draft_end)
        || !augur_qwen35_dflash_acceptance_range(
            args->target_logits, args->target_logits_len_bytes, 4, &logits_end)
        || !augur_qwen35_dflash_acceptance_range(
            args->target_ids, args->target_ids_len_bytes, 4, &target_end)
        || !augur_qwen35_dflash_acceptance_range(
            args->receipt, args->receipt_len_bytes, 8, &receipt_end)
        || augur_qwen35_dflash_acceptance_overlap(args->draft_ids, draft_end, args->target_logits, logits_end)
        || augur_qwen35_dflash_acceptance_overlap(args->draft_ids, draft_end, args->target_ids, target_end)
        || augur_qwen35_dflash_acceptance_overlap(args->draft_ids, draft_end, args->receipt, receipt_end)
        || augur_qwen35_dflash_acceptance_overlap(args->target_logits, logits_end, args->target_ids, target_end)
        || augur_qwen35_dflash_acceptance_overlap(args->target_logits, logits_end, args->receipt, receipt_end)
        || augur_qwen35_dflash_acceptance_overlap(args->target_ids, target_end, args->receipt, receipt_end)) {
        return augur_qwen35_dflash_acceptance_result(
            AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) {
        return augur_qwen35_dflash_acceptance_result(
            AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    }
    const auto* draft_ids = reinterpret_cast<const uint32_t*>(
        static_cast<uintptr_t>(args->draft_ids));
    const auto* target_logits = reinterpret_cast<const float*>(
        static_cast<uintptr_t>(args->target_logits));
    auto* target_ids = reinterpret_cast<uint32_t*>(
        static_cast<uintptr_t>(args->target_ids));
    auto* receipt = reinterpret_cast<augur_cuda_qwen35_dflash_acceptance_receipt_v1*>(
        static_cast<uintptr_t>(args->receipt));
    augur_qwen35_dflash_acceptance_digest_v1 digest{};
    for (uint32_t index = 0; index < 32; ++index) digest.bytes[index] = args->plan_digest[index];

    augur_qwen35_dflash_target_argmax_kernel<1024><<<args->draft_tokens, 1024, 0, stream->stream>>>(
        target_logits, args->draft_tokens, args->vocab, target_ids);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return augur_qwen35_dflash_acceptance_result(
            AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    }
    augur_qwen35_dflash_logits_acceptance_kernel<<<1, 1, 0, stream->stream>>>(
        draft_ids, target_ids, args->draft_tokens, receipt, args->step_generation, digest);
    error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_OK, cudaSuccess)
        : augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

// Original BF16 codebooks, already projected F32 gates. Same FP32 scoring order.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_bf16_codebooks_submit_v1(
    int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_qwen35_dflash_q8_selector_args_v1* a) {
    if(!a || !stream || ordinal<0 || stream->ordinal!=ordinal || a->abi_version!=1
        || a->struct_size!=sizeof(*a) || a->reserved!=0 || a->rows!=8 || a->top_k!=16
        || a->rank!=256 || a->vocab!=248320 || a->anchor_token>=a->vocab)
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    const uint64_t ptrs[7] = { a->predecessor, a->successor, a->candidate_ids,
        a->unary_scores, a->gates, a->transition_scores, a->path_ids };
    const uint64_t lens[7] = { a->predecessor_len_bytes, a->successor_len_bytes,
        a->candidate_ids_len_bytes, a->unary_scores_len_bytes, a->gates_len_bytes,
        a->transition_scores_len_bytes, a->path_ids_len_bytes };
    const uint64_t expected[7] = { 248320*256*2, 248320*256*2, 8*16*4, 8*16*4, 8*256*4, 8*16*4, 8*4 };
    uint64_t ends[7] = {};
    for (uint32_t i = 0; i < 7; ++i) {
        if (lens[i] != expected[i] || !augur_qwen35_dflash_acceptance_range(ptrs[i], lens[i], i < 2 ? 2 : 4, &ends[i])) {
            return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (augur_qwen35_dflash_acceptance_overlap(ptrs[i], ends[i], ptrs[j], ends[j])) {
                return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    augur_qwen35_dflash_selector_walk_kernel<false,false,false,false,false,false,true><<<1,256,0,stream->stream>>>(
        reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a->candidate_ids)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->unary_scores)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->gates)),
        reinterpret_cast<const void*>(static_cast<uintptr_t>(a->predecessor)),
        reinterpret_cast<const void*>(static_cast<uintptr_t>(a->successor)),a->anchor_token,
        reinterpret_cast<float*>(static_cast<uintptr_t>(a->transition_scores)),
        reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a->path_ids)));
    const cudaError_t error=cudaGetLastError();
    return augur_qwen35_dflash_acceptance_result(error==cudaSuccess?AUGUR_CUDA_STATUS_OK:AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
}

template<bool CALIBRATED> static augur_cuda_result_v1 augur_qwen35_selector_probability_submit(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_dflash_q8_selector_args_v1* a,
    uint64_t q, uint64_t q_bytes, uint64_t absolute_start, uint32_t positions,
    const augur_qwen35_proposal_calibration& calibration
) {
    bool calibration_valid = true;
    for (uint32_t row = 1; row < 8; ++row) {
        calibration_valid &= calibration.temperature[row] >= 0.5f && calibration.temperature[row] <= 1.5f;
        calibration_valid &= calibration.pair_scale[row] >= 0.0f && calibration.pair_scale[row] <= 2.0f;
    }
    if (!a || !stream || ordinal<0 || stream->ordinal!=ordinal || a->abi_version!=1
        || a->struct_size!=sizeof(*a) || a->reserved!=0 || a->rows!=8 || a->top_k!=16
        || a->rank!=256 || a->vocab!=248320 || a->anchor_token>=248320
        || positions<1 || positions>7 || absolute_start>UINT64_MAX-8
        || !calibration_valid
        || (CALIBRATED && positions != 7)) {
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    }
    const uint64_t ptrs[8]={a->predecessor,a->successor,a->candidate_ids,a->unary_scores,a->gates,a->transition_scores,a->path_ids,q};
    const uint64_t lens[8]={a->predecessor_len_bytes,a->successor_len_bytes,a->candidate_ids_len_bytes,a->unary_scores_len_bytes,a->gates_len_bytes,a->transition_scores_len_bytes,a->path_ids_len_bytes,q_bytes};
    const uint64_t sizes[8]={248320*8*34,248320*8*34,8*16*4,8*16*4,8*256*4,8*16*4,8*4,8*16*8};
    uint64_t ends[8]={};
    for(uint32_t i=0;i<8;++i) {
        if(lens[i]!=sizes[i] || !augur_qwen35_dflash_acceptance_range(ptrs[i],lens[i],i==7?8:(i<2?2:4),&ends[i]))
            return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
        for(uint32_t j=0;j<i;++j) if(augur_qwen35_dflash_acceptance_overlap(ptrs[i],ends[i],ptrs[j],ends[j]))
            return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    }
    cudaError_t e=cudaSetDevice(ordinal);
    if(e!=cudaSuccess) return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,e);
    augur_qwen35_dflash_selector_walk_kernel<true,false,false,false,false,false,false,false,true,CALIBRATED><<<1,256,0,stream->stream>>>(
        reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a->candidate_ids)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->unary_scores)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->gates)),
        reinterpret_cast<const void*>(static_cast<uintptr_t>(a->predecessor)),
        reinterpret_cast<const void*>(static_cast<uintptr_t>(a->successor)),a->anchor_token,
        reinterpret_cast<float*>(static_cast<uintptr_t>(a->transition_scores)),
        reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a->path_ids)),1.0f,0.0f,1.0f,1.0f,nullptr,UINT32_MAX,
        reinterpret_cast<double*>(static_cast<uintptr_t>(q)),absolute_start,positions,calibration);
    e=cudaGetLastError();return augur_qwen35_dflash_acceptance_result(e==cudaSuccess?AUGUR_CUDA_STATUS_OK:AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,e);
}

static augur_qwen35_proposal_calibration augur_qwen35_uniform_calibration(float temperature, float pair_scale, uint32_t first, uint32_t last) {
    augur_qwen35_proposal_calibration c;
    for (uint32_t row = 0; row < 8; ++row) {
        const bool active = row >= first && row <= last;
        c.temperature[row] = active ? temperature : 1.0f;
        c.pair_scale[row] = active ? pair_scale : 1.0f;
    }
    return c;
}

// v1 ABI and arithmetic remain unchanged. v2 calibrates proposal rows2-4 only.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_probability_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_dflash_q8_selector_args_v1* a,
    uint64_t q, uint64_t q_bytes, uint64_t absolute_start, uint32_t positions
) { return augur_qwen35_selector_probability_submit<false>(ordinal,stream,a,q,q_bytes,absolute_start,positions,augur_qwen35_uniform_calibration(1.0f,1.0f,1,7)); }
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_probability_submit_v2(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_dflash_q8_selector_args_v1* a,
    uint64_t q, uint64_t q_bytes, uint64_t absolute_start, uint32_t positions,
    float temperature234, float pair_scale234
) {
    // NaN must fail closed before it reaches the per-row table.
    if (!(temperature234 >= 0.5f && temperature234 <= 1.5f) || !(pair_scale234 >= 0.0f && pair_scale234 <= 2.0f))
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    return augur_qwen35_selector_probability_submit<true>(ordinal,stream,a,q,q_bytes,absolute_start,positions,augur_qwen35_uniform_calibration(temperature234,pair_scale234,2,4));
}
// v3: per-position proposal temperature/pair scale for rows1-7 (host arrays of seven floats,
// copied by value into kernel parameters). Row1 has no draft predecessor; its pair scale must be 1.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_probability_submit_v3(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_dflash_q8_selector_args_v1* a,
    uint64_t q, uint64_t q_bytes, uint64_t absolute_start, uint32_t positions,
    const float* temperatures7, const float* pair_scales7
) {
    if (!temperatures7 || !pair_scales7 || !(pair_scales7[0] == 1.0f))
        return augur_qwen35_dflash_acceptance_result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    augur_qwen35_proposal_calibration c = augur_qwen35_uniform_calibration(1.0f,1.0f,1,7);
    for (uint32_t row = 1; row < 8; ++row) { c.temperature[row] = temperatures7[row-1]; c.pair_scale[row] = pair_scales7[row-1]; }
    return augur_qwen35_selector_probability_submit<true>(ordinal,stream,a,q,q_bytes,absolute_start,positions,c);
}
