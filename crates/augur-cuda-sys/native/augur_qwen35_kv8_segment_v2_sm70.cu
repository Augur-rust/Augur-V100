#include "augur_cuda_internal.h"
#include "augur_qwen35_kv8_segment_v2_abi.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>

namespace {

constexpr uint32_t kQueryHeads = 24;
constexpr uint32_t kKvHeads = 4;
constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kBlockElements = 32;
constexpr uint32_t kBlockBytes = 34;
constexpr uint32_t kBlocksPerHead = kHeadDim / kBlockElements;
constexpr uint32_t kHeadRowBytes = kBlocksPerHead * kBlockBytes;
constexpr uint32_t kCacheRowBytes = kKvHeads * kHeadRowBytes;
constexpr float kAttentionScale = 0.0625f; // 1/sqrt(256)
constexpr uint32_t kPageDigestBytes = 32;

static augur_cuda_result_v1 result(int32_t status, cudaError_t raw) {
    augur_cuda_result_v1 out{};
    out.status = status;
    out.raw_cuda_status = static_cast<int32_t>(raw);
    return out;
}

static bool range(uint64_t address, uint64_t extent, uint64_t alignment, uint64_t* end) {
    if (address == 0 || extent == 0 || address % alignment != 0
        || address > std::numeric_limits<uint64_t>::max() - extent) {
        return false;
    }
    *end = address + extent;
    return true;
}

static bool overlap(uint64_t a, uint64_t a_end, uint64_t b, uint64_t b_end) {
    return a < b_end && b < a_end;
}

static bool valid_digest(const augur_cuda_qwen35_kv8_page_digest_args_v1* a) {
    uint64_t input_end = 0;
    uint64_t output_end = 0;
    if (a == nullptr
        || a->abi_version != AUGUR_CUDA_QWEN35_KV8_PAGE_DIGEST_ARGS_VERSION_V1
        || a->struct_size != sizeof(*a)
        || a->reserved0 != 0 || a->reserved1 != 0
        || a->input_len_bytes == 0
        || a->output_len_bytes != kPageDigestBytes
        || !range(a->input, a->input_len_bytes, 1, &input_end)
        || !range(a->output, a->output_len_bytes, 8, &output_end)
        || overlap(a->input, input_end, a->output, output_end)) {
        return false;
    }
    return true;
}

__device__ __forceinline__ uint64_t digest_mix(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

__global__ void page_digest_kernel(augur_cuda_qwen35_kv8_page_digest_args_v1 a) {
    __shared__ uint64_t partial[256][4];
    const uint32_t tid = threadIdx.x;
    uint64_t local[4] = {0, 0, 0, 0};
    const auto* input = reinterpret_cast<const uint8_t*>(a.input);
    for (uint64_t index = tid; index < a.input_len_bytes; index += blockDim.x) {
        const uint32_t lane = static_cast<uint32_t>(index & 3U);
        const uint64_t tagged = static_cast<uint64_t>(input[index])
            + UINT64_C(0x9e3779b97f4a7c15)
            + index * UINT64_C(0x100000001b3);
        local[lane] ^= digest_mix(tagged ^ (static_cast<uint64_t>(lane) << 61));
    }
    for (uint32_t lane = 0; lane < 4; ++lane) partial[tid][lane] = local[lane];
    __syncthreads();
    if (tid == 0) {
        auto* output = reinterpret_cast<uint64_t*>(a.output);
        const uint64_t seeds[4] = {
            UINT64_C(0xcbf29ce484222325), UINT64_C(0x84222325cbf29ce4),
            UINT64_C(0x9e3779b185ebca87), UINT64_C(0xd6e8feb86659fd93)};
        for (uint32_t lane = 0; lane < 4; ++lane) {
            uint64_t digest = seeds[lane];
            for (uint32_t row = 0; row < 256; ++row) digest ^= partial[row][lane];
            output[lane] = digest;
        }
    }
}

// Exact same four lanes as V1. A separate ordered initialization launch avoids
// cross-CTA initialization races. XOR is associative on integer bit patterns;
// the byte's absolute index (not its CTA-local offset) remains part of the tag.
__global__ void page_digest_seed_kernel(uint64_t output_address) {
    auto* output = reinterpret_cast<uint64_t*>(output_address);
    const uint64_t seeds[4] = {
        UINT64_C(0xcbf29ce484222325), UINT64_C(0x84222325cbf29ce4),
        UINT64_C(0x9e3779b185ebca87), UINT64_C(0xd6e8feb86659fd93)};
    if (threadIdx.x < 4) output[threadIdx.x] = seeds[threadIdx.x];
}

__global__ void page_digest_parallel_kernel(augur_cuda_qwen35_kv8_page_digest_args_v1 a) {
    __shared__ uint64_t partial[256];
    const uint32_t tid = threadIdx.x;
    // Both the CTA width and grid stride are multiples of four, so each
    // thread owns one of the four V1 lanes for all of its bytes.
    const uint32_t lane = tid & 3U;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    uint64_t local = 0;
    const auto* input = reinterpret_cast<const uint8_t*>(a.input);
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + tid;
         index < a.input_len_bytes;) {
        const uint64_t tagged = static_cast<uint64_t>(input[index])
            + UINT64_C(0x9e3779b97f4a7c15)
            + index * UINT64_C(0x100000001b3);
        local ^= digest_mix(tagged ^ (static_cast<uint64_t>(lane) << 61));
        if (a.input_len_bytes - index <= stride) break;
        index += stride;
    }
    partial[tid] = local;
    __syncthreads();
    if (tid < 4) {
        uint64_t value = 0;
        for (uint32_t row = tid; row < 256; row += 4) value ^= partial[row];
        atomicXor(reinterpret_cast<unsigned long long*>(a.output) + tid,
                  static_cast<unsigned long long>(value));
    }
}

static bool valid(const augur_cuda_qwen35_kv8_segment_partial_args_v2* a) {
    const uint64_t slot_end = a == nullptr
        ? 0
        : static_cast<uint64_t>(a->cache_position_base)
            + static_cast<uint64_t>(a->slot_capacity_positions);
    if (a == nullptr
        || a->abi_version != AUGUR_CUDA_QWEN35_KV8_SEGMENT_PARTIAL_ARGS_VERSION_V2
        || a->struct_size != sizeof(*a)
        || a->tokens == 0
        || a->query_heads != kQueryHeads
        || a->kv_heads != kKvHeads
        || a->head_dim != kHeadDim
        || a->mode > 3
        || (a->mode == 3 && a->tokens != 1)
        || a->reserved0 != 0
        || a->reserved1 != 0
        || a->segment_begin >= a->segment_end
        || a->segment_begin < a->cache_position_base
        || static_cast<uint64_t>(a->segment_end) > slot_end
        || a->segment_end > a->visible_position_end) {
        return false;
    }
    const uint64_t query_bytes = static_cast<uint64_t>(a->tokens) * kQueryHeads * kHeadDim * 4;
    const uint64_t cache_bytes = static_cast<uint64_t>(a->slot_capacity_positions) * kCacheRowBytes;
    const uint64_t partial_bytes = static_cast<uint64_t>(a->tokens) * kQueryHeads * 4;
    const uint64_t weighted_bytes = static_cast<uint64_t>(a->tokens) * kQueryHeads * kHeadDim * 4;
    if (a->query_len_bytes != query_bytes
        || a->key_cache_len_bytes != cache_bytes
        || a->value_cache_len_bytes != cache_bytes
        || a->partial_max_len_bytes != partial_bytes
        || a->partial_denominator_len_bytes != partial_bytes
        || a->partial_weighted_value_len_bytes != weighted_bytes) {
        return false;
    }
    uint64_t q_end = 0, k_end = 0, v_end = 0, m_end = 0, d_end = 0, n_end = 0;
    if (!range(a->query_f32, query_bytes, 4, &q_end)
        || !range(a->key_cache_q8_0, cache_bytes, 2, &k_end)
        || !range(a->value_cache_q8_0, cache_bytes, 2, &v_end)
        || !range(a->partial_max_f32, partial_bytes, 4, &m_end)
        || !range(a->partial_denominator_f32, partial_bytes, 4, &d_end)
        || !range(a->partial_weighted_value_f32, weighted_bytes, 4, &n_end)) {
        return false;
    }
    const uint64_t starts[] = {a->query_f32, a->key_cache_q8_0, a->value_cache_q8_0,
        a->partial_max_f32, a->partial_denominator_f32, a->partial_weighted_value_f32};
    const uint64_t ends[] = {q_end, k_end, v_end, m_end, d_end, n_end};
    for (uint32_t i = 0; i < 6; ++i) {
        for (uint32_t j = i + 1; j < 6; ++j) {
            if (overlap(starts[i], ends[i], starts[j], ends[j])) return false;
        }
    }
    return true;
}

__device__ float decode(const uint8_t* row, uint32_t element) {
    const uint8_t* block = row + (element / kBlockElements) * kBlockBytes;
    const __half scale = *reinterpret_cast<const __half*>(block);
    const int8_t code = static_cast<int8_t>(block[2 + (element % kBlockElements)]);
    return __half2float(scale) * static_cast<float>(code);
}

__global__ void segment_partial_kernel(augur_cuda_qwen35_kv8_segment_partial_args_v2 a) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t items = a.tokens * a.query_heads;
    if (item >= items) return;
    const uint32_t qh = item % a.query_heads;
    const uint32_t token = item / a.query_heads;
    const uint32_t kvh = qh / (a.query_heads / a.kv_heads);
    const float* query = reinterpret_cast<const float*>(static_cast<uintptr_t>(a.query_f32))
        + (static_cast<uint64_t>(token) * a.query_heads + qh) * a.head_dim;
    const uint8_t* keys = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a.key_cache_q8_0));
    const uint8_t* values = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a.value_cache_q8_0));
    float maximum = -INFINITY;
    for (uint32_t position = a.segment_begin; position < a.segment_end; ++position) {
        const uint32_t row_index = position - a.cache_position_base;
        const uint8_t* key_row = keys + static_cast<uint64_t>(row_index) * kCacheRowBytes
            + kvh * kHeadRowBytes;
        float score = 0.0f;
        for (uint32_t d = 0; d < a.head_dim; ++d) score += query[d] * decode(key_row, d);
        maximum = fmaxf(maximum, score * kAttentionScale);
    }
    float denominator = 0.0f;
    float* numerator = reinterpret_cast<float*>(static_cast<uintptr_t>(a.partial_weighted_value_f32))
        + (static_cast<uint64_t>(token) * a.query_heads + qh) * a.head_dim;
    for (uint32_t d = 0; d < a.head_dim; ++d) numerator[d] = 0.0f;
    for (uint32_t position = a.segment_begin; position < a.segment_end; ++position) {
        const uint32_t row_index = position - a.cache_position_base;
        const uint8_t* key_row = keys + static_cast<uint64_t>(row_index) * kCacheRowBytes
            + kvh * kHeadRowBytes;
        const uint8_t* value_row = values + static_cast<uint64_t>(row_index) * kCacheRowBytes
            + kvh * kHeadRowBytes;
        float score = 0.0f;
        for (uint32_t d = 0; d < a.head_dim; ++d) score += query[d] * decode(key_row, d);
        const float weight = expf(score * kAttentionScale - maximum);
        denominator += weight;
        for (uint32_t d = 0; d < a.head_dim; ++d) numerator[d] += weight * decode(value_row, d);
    }
    const uint32_t out = token * a.query_heads + qh;
    reinterpret_cast<float*>(static_cast<uintptr_t>(a.partial_max_f32))[out] = maximum;
    reinterpret_cast<float*>(static_cast<uintptr_t>(a.partial_denominator_f32))[out] = denominator;
}

// Mode 1 is a separate precision candidate. One CTA owns one query head;
// dot products and online numerator/denominator accumulation use F64. The
// emitted partial ABI remains F32, so mode 0 evidence remains unchanged.
__global__ void segment_partial_precise_kernel(augur_cuda_qwen35_kv8_segment_partial_args_v2 a) {
    const uint32_t item = blockIdx.x, component = threadIdx.x;
    const uint32_t qh = item % kQueryHeads;
    const uint32_t kvh = qh / (kQueryHeads / kKvHeads);
    const float* query = reinterpret_cast<const float*>(a.query_f32) + static_cast<uint64_t>(item) * kHeadDim;
    const uint8_t* keys = reinterpret_cast<const uint8_t*>(a.key_cache_q8_0);
    const uint8_t* values = reinterpret_cast<const uint8_t*>(a.value_cache_q8_0);
    __shared__ double reduction[kHeadDim];
    double maximum = -INFINITY, denominator = 0.0, numerator = 0.0;
    for (uint32_t position = a.segment_begin; position < a.segment_end; ++position) {
        const uint64_t offset = static_cast<uint64_t>(position - a.cache_position_base)
            * kCacheRowBytes + kvh * kHeadRowBytes;
        reduction[component] = static_cast<double>(query[component]) * decode(keys + offset, component);
        __syncthreads();
        for (uint32_t stride = kHeadDim / 2; stride != 0; stride >>= 1) {
            if (component < stride) reduction[component] += reduction[component + stride];
            __syncthreads();
        }
        const double score = static_cast<float>(reduction[0]) * kAttentionScale;
        __syncthreads(); // every warp captures the dot before the next row
        const double next_maximum = fmax(maximum, score);
        const double old_scale = denominator == 0.0 ? 0.0 : exp(maximum - next_maximum);
        const double current_scale = exp(score - next_maximum);
        denominator = denominator * old_scale + current_scale;
        numerator = numerator * old_scale + current_scale * decode(values + offset, component);
        maximum = next_maximum;
    }
    reinterpret_cast<float*>(a.partial_weighted_value_f32)[static_cast<uint64_t>(item) * kHeadDim + component]
        = static_cast<float>(numerator);
    if (component == 0) {
        reinterpret_cast<float*>(a.partial_max_f32)[item] = static_cast<float>(maximum);
        reinterpret_cast<float*>(a.partial_denominator_f32)[item] = static_cast<float>(denominator);
    }
}

// Mode 2 mirrors the already audited V1 SIMT reference ordering.  A CTA owns
// one token/KV-head pair; each warp owns one GQA query head, computes its
// eight-lane-fragment dot product in double, and then all 256 threads apply
// the same FP32 online-softmax recurrence in position order.  The output is
// still the V2 partial ABI, so LSE merge and ownership rules remain unchanged.
template<bool ContinueAcrossPages = false>
__global__ void segment_partial_v1_match_kernel(
    augur_cuda_qwen35_kv8_segment_partial_args_v2 a) {
    constexpr uint32_t queries_per_kv = kQueryHeads / kKvHeads;
    const uint32_t item = blockIdx.x;
    const uint32_t component = threadIdx.x;
    const uint32_t token = item / kKvHeads;
    const uint32_t kvh = item % kKvHeads;
    if (token >= a.tokens || component >= kHeadDim) return;

    const uint32_t warp = component / 32U;
    const uint32_t lane = component & 31U;
    const bool active_query = warp < queries_per_kv;
    const uint32_t warp_qh = kvh * queries_per_kv + warp;
    const float* query = reinterpret_cast<const float*>(a.query_f32);
    const uint8_t* keys = reinterpret_cast<const uint8_t*>(a.key_cache_q8_0);
    const uint8_t* values = reinterpret_cast<const uint8_t*>(a.value_cache_q8_0);
    __shared__ float scores[queries_per_kv];

    float query_fragment[8] = {};
    if (active_query) {
        const uint64_t query_base =
            (static_cast<uint64_t>(token) * kQueryHeads + warp_qh) * kHeadDim;
        #pragma unroll
        for (uint32_t i = 0; i < 8U; ++i) {
            query_fragment[i] = query[query_base + lane + i * 32U];
        }
    }

    float running_maximum[queries_per_kv];
    float running_sum[queries_per_kv] = {};
    float running_output[queries_per_kv] = {};
    #pragma unroll
    for (uint32_t q = 0; q < queries_per_kv; ++q) {
        running_maximum[q] = -INFINITY;
        if (ContinueAcrossPages && a.segment_begin != 0) {
            const uint32_t row = token * kQueryHeads + kvh * queries_per_kv + q;
            running_maximum[q] = reinterpret_cast<const float*>(a.partial_max_f32)[row];
            running_sum[q] = reinterpret_cast<const float*>(a.partial_denominator_f32)[row];
            running_output[q] = reinterpret_cast<const float*>(a.partial_weighted_value_f32)[
                static_cast<uint64_t>(row) * kHeadDim + component];
        }
    }
    // Mode 3 reads the preceding ordered prefix before any in-place stores.
    __syncthreads();

    for (uint32_t position = a.segment_begin; position < a.segment_end; ++position) {
        const uint32_t row_index = position - a.cache_position_base;
        const uint8_t* key_row = keys
            + static_cast<uint64_t>(row_index) * kCacheRowBytes
            + kvh * kHeadRowBytes;
        double score = 0.0;
        if (active_query) {
            #pragma unroll
            for (uint32_t i = 0; i < 8U; ++i) {
                score += static_cast<double>(query_fragment[i])
                    * static_cast<double>(decode(key_row, lane + i * 32U));
            }
            #pragma unroll
            for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
                score += __shfl_down_sync(0xffffffffU, score, offset);
            }
            if (lane == 0U) scores[warp] = static_cast<float>(score) * kAttentionScale;
        }
        __syncthreads();

        const uint8_t* value_row = values
            + static_cast<uint64_t>(row_index) * kCacheRowBytes
            + kvh * kHeadRowBytes;
        const float value = decode(value_row, component);
        #pragma unroll
        for (uint32_t q = 0; q < queries_per_kv; ++q) {
            const float next_maximum = fmaxf(running_maximum[q], scores[q]);
            const float previous_scale = running_sum[q] == 0.0f
                ? 0.0f : expf(running_maximum[q] - next_maximum);
            const float current_scale = expf(scores[q] - next_maximum);
            running_sum[q] = running_sum[q] * previous_scale + current_scale;
            running_output[q] = running_output[q] * previous_scale
                + current_scale * value;
            running_maximum[q] = next_maximum;
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t q = 0; q < queries_per_kv; ++q) {
        const uint32_t qh = kvh * queries_per_kv + q;
        const uint64_t out =
            (static_cast<uint64_t>(token) * kQueryHeads + qh) * kHeadDim;
        reinterpret_cast<float*>(a.partial_weighted_value_f32)[out + component] =
            running_output[q];
        if (component == 0U) {
            const uint32_t scalar = token * kQueryHeads + qh;
            reinterpret_cast<float*>(a.partial_max_f32)[scalar] = running_maximum[q];
            reinterpret_cast<float*>(a.partial_denominator_f32)[scalar] = running_sum[q];
        }
    }
}

static bool valid_merge(const augur_cuda_qwen35_kv8_lse_merge_args_v2* a) {
    if (a == nullptr || a->abi_version != 2 || a->struct_size != sizeof(*a)
        || a->query_rows == 0 || a->query_rows > 24U * 64U
        || a->query_rows % kQueryHeads != 0 || a->head_dim != kHeadDim
        || a->operation > 3 || a->reserved != 0) return false;
    const uint64_t rows = a->query_rows;
    const uint64_t elements = rows * kHeadDim;
    const uint64_t starts[] = {a->partial_max_f32, a->partial_denominator_f32,
        a->partial_weighted_f32, a->accumulated_max_f64,
        a->accumulated_denominator_f64, a->accumulated_weighted_f64,
        a->output_f32, a->status_u32};
    const uint64_t sizes[] = {a->partial_max_len_bytes, a->partial_denominator_len_bytes,
        a->partial_weighted_len_bytes, a->accumulated_max_len_bytes,
        a->accumulated_denominator_len_bytes, a->accumulated_weighted_len_bytes,
        a->output_len_bytes, a->status_len_bytes};
    const uint64_t expected[] = {rows * 4, rows * 4, elements * 4,
        rows * 8, rows * 8, elements * 8, elements * 4, rows * 4};
    uint64_t ends[8];
    for (uint32_t i = 0; i < 8; ++i) {
        const uint64_t alignment = (i >= 3 && i <= 5) ? 8 : 4;
        if (sizes[i] != expected[i]
            || !range(starts[i], sizes[i], alignment, &ends[i])) return false;
        for (uint32_t j = 0; j < i; ++j) {
            if (overlap(starts[i], ends[i], starts[j], ends[j])) return false;
        }
    }
    return true;
}

__global__ void lse_merge_kernel(augur_cuda_qwen35_kv8_lse_merge_args_v2 a) {
    const uint32_t row = blockIdx.x;
    const uint32_t component = threadIdx.x;
    const uint64_t index = static_cast<uint64_t>(row) * kHeadDim + component;
    auto* maximum = reinterpret_cast<double*>(a.accumulated_max_f64);
    auto* denominator = reinterpret_cast<double*>(a.accumulated_denominator_f64);
    auto* numerator = reinterpret_cast<double*>(a.accumulated_weighted_f64);
    auto* output = reinterpret_cast<float*>(a.output_f32);
    auto* status = reinterpret_cast<uint32_t*>(a.status_u32);
    if (a.operation == 0) {
        numerator[index] = 0.0;
        output[index] = NAN;
        if (component == 0) {
            maximum[row] = -INFINITY;
            denominator[row] = 0.0;
            status[row] = 0;
        }
        return;
    }
    if (a.operation == 2) {
        const double d = denominator[row];
        const double value = numerator[index] / d;
        const bool bad = status[row] != 0 || !isfinite(maximum[row])
            || !isfinite(d) || d <= 0.0 || !isfinite(value)
            || fabs(value) > static_cast<double>(FLT_MAX);
        const int row_bad = __syncthreads_or(bad);
        output[index] = row_bad ? NAN : static_cast<float>(value);
        if (component == 0 && row_bad) status[row] = 1;
        return;
    }
    const auto* part_m = reinterpret_cast<const float*>(a.partial_max_f32);
    const auto* part_d = reinterpret_cast<const float*>(a.partial_denominator_f32);
    const auto* part_n = reinterpret_cast<const float*>(a.partial_weighted_f32);
    const double pm = part_m[row], pd = part_d[row], pn = part_n[index];
    const double am = maximum[row], ad = denominator[row], an = numerator[index];
    const bool part_empty = pd == 0.0 && pm == -INFINITY;
    const bool acc_empty = ad == 0.0 && am == -INFINITY;
    const bool bad = status[row] != 0 || !isfinite(pn) || !isfinite(an)
        || !(part_empty ? pn == 0.0 : (isfinite(pm) && isfinite(pd) && pd > 0.0))
        || !(acc_empty ? an == 0.0 : (isfinite(am) && isfinite(ad) && ad > 0.0));
    if (__syncthreads_or(bad)) {
        if (component == 0) status[row] = 1;
        return;
    }
    // Ordered-prefix import is NOT an independent-page accumulate. Its
    // numerator/denominator already include preceding pages; adding them a
    // second time would double-count history. Poison remains sticky.
    if (a.operation == 3) {
        if (__syncthreads_or(part_empty)) {
            if (component == 0) status[row] = 1;
            return;
        }
        numerator[index] = pn;
        if (component == 0) {
            maximum[row] = pm;
            denominator[row] = pd;
        }
        return;
    }
    if (part_empty) return;
    const double m = acc_empty ? pm : fmax(am, pm);
    const double old_scale = acc_empty ? 0.0 : exp(am - m);
    const double part_scale = exp(pm - m);
    const double d = ad * old_scale + pd * part_scale;
    const double n = an * old_scale + pn * part_scale;
    const int merged_bad = __syncthreads_or(!isfinite(d) || d <= 0.0 || !isfinite(n));
    if (merged_bad) {
        if (component == 0) status[row] = 1;
        return;
    }
    numerator[index] = n;
    if (component == 0) {
        maximum[row] = m;
        denominator[row] = d;
    }
}

} // namespace

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_segment_partial_preflight_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_segment_partial_args_v2* args) {
    if (stream == nullptr || ordinal < 0 || stream->ordinal != ordinal || !valid(args)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return result(AUGUR_CUDA_STATUS_OK, cudaSuccess);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_segment_partial_submit_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_segment_partial_args_v2* args) {
    if (stream == nullptr || ordinal < 0 || stream->ordinal != ordinal || !valid(args)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    const uint32_t items = args->tokens * args->query_heads;
    if (args->mode == 3) {
        segment_partial_v1_match_kernel<true><<<args->tokens * kKvHeads, kHeadDim, 0, stream->stream>>>(*args);
    } else if (args->mode == 2) {
        segment_partial_v1_match_kernel<false><<<args->tokens * kKvHeads, kHeadDim, 0, stream->stream>>>(*args);
    } else if (args->mode == 1) {
        segment_partial_precise_kernel<<<items, kHeadDim, 0, stream->stream>>>(*args);
    } else {
        segment_partial_kernel<<<(items + 127) / 128, 128, 0, stream->stream>>>(*args);
    }
    const cudaError_t launched = cudaGetLastError();
    return launched == cudaSuccess
        ? result(AUGUR_CUDA_STATUS_OK, launched)
        : result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launched);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_lse_merge_submit_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_lse_merge_args_v2* args) {
    if (stream == nullptr || ordinal < 0 || stream->ordinal != ordinal || !valid_merge(args))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    lse_merge_kernel<<<args->query_rows, kHeadDim, 0, stream->stream>>>(*args);
    const cudaError_t launched = cudaGetLastError();
    return launched == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, launched)
        : result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launched);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_page_digest_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_page_digest_args_v1* args) {
    if (stream == nullptr || ordinal < 0 || stream->ordinal != ordinal || !valid_digest(args))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    return result(AUGUR_CUDA_STATUS_OK, cudaSuccess);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_page_digest_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_page_digest_args_v1* args) {
    if (stream == nullptr || ordinal < 0 || stream->ordinal != ordinal || !valid_digest(args))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    page_digest_kernel<<<1, 256, 0, stream->stream>>>(*args);
    const cudaError_t launched = cudaGetLastError();
    return launched == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, launched)
        : result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launched);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_page_digest_parallel_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_page_digest_args_v1* args) {
    if (stream == nullptr || ordinal < 0 || stream->ordinal != ordinal || !valid_digest(args))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    if (args->input_len_bytes <= 4096) {
        page_digest_kernel<<<1, 256, 0, stream->stream>>>(*args);
    } else {
        const uint64_t wanted = 1 + (args->input_len_bytes - 1) / 4096;
        const uint32_t blocks = static_cast<uint32_t>(wanted < 256 ? wanted : 256);
        page_digest_seed_kernel<<<1, 32, 0, stream->stream>>>(args->output);
        const cudaError_t initialized = cudaGetLastError();
        if (initialized != cudaSuccess)
            return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, initialized);
        page_digest_parallel_kernel<<<blocks, 256, 0, stream->stream>>>(*args);
    }
    const cudaError_t launched = cudaGetLastError();
    return launched == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, launched)
        : result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launched);
}
