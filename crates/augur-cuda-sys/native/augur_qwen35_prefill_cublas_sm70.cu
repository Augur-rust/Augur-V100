#include "augur_cuda_internal.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

#include "augur_qwen35_prefill_llama_small_sm70.cuh"

extern "C" cudaError_t augur_qwen35_launch_llama_swiglu_sm70(
    const float*, const float*, float*, uint32_t, cudaStream_t);
extern "C" cudaError_t augur_qwen35_launch_llama_input_rmsnorm5120_sm70(
    const float*, const float*, float*, uint32_t, cudaStream_t);
extern "C" cudaError_t augur_qwen35_launch_llama_mmq_sm70(
    const void*, const float*, void*, float*, float*, unsigned, unsigned,
    unsigned, unsigned, float, unsigned, cudaStream_t);

static_assert(sizeof(augur_cuda_qwen35_prefill_cublas_plan_args_v1) == 64,
    "Qwen prefill cuBLAS plan ABI drift");
static_assert(sizeof(augur_cuda_qwen35_prefill_cublas_projection_args_v1) == 80,
    "Qwen prefill cuBLAS projection ABI drift");
static_assert(sizeof(augur_cuda_qwen35_prefill_f32_projection_args_v1) == 80,
    "Qwen prefill F32 activation projection ABI drift");
static_assert(sizeof(augur_cuda_qwen35_prefill_rmsnorm_args_v1) == 72,
    "Qwen prefill RMSNorm ABI drift");
static_assert(sizeof(augur_cuda_qwen35_prefill_linear_chunk_args_v1) == 232,
    "Qwen prefill linear chunk ABI drift");
static_assert(sizeof(augur_cuda_qwen35_prefill_full_attention_args_v1) == 256,
    "Qwen prefill full-attention ABI drift");
#if defined(AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION)
static_assert(
    sizeof(augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1) == 288,
    "Qwen prefill KV8 chunk-attention ABI drift");
static_assert(
    sizeof(augur_cuda_qwen35_prefill_kv8_cache_quantize_args_v1) == 96,
    "Qwen prefill KV8 cache handoff ABI drift");
#endif
#if defined(AUGUR_CUDA_QWEN35_TC_PREFIX_ATTENTION)
static_assert(
    sizeof(augur_cuda_qwen35_tc_prefix_attention_args_v1) == 256,
    "Qwen Tensor-Core prefix-attention ABI drift");
#endif

struct augur_cuda_qwen35_prefill_cublas_plan_v1 {
    int32_t ordinal;
    uint32_t multiprocessors;
    augur_cuda_stream_t owner;
    cublasHandle_t handle;
    augur_cuda_qwen35_prefill_cublas_plan_args_v1 snapshot;
};

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxBlocks = 65535;
constexpr uint32_t kQ8BlockElements = 32;
constexpr uint32_t kQ8WeightBlockBytes = 34;
constexpr uint32_t kQ8ActivationBlockBytes = 36;
constexpr uint32_t kNvfp4BlockElements = 64;
constexpr uint32_t kNvfp4BlockBytes = 36;
constexpr uint32_t kLinearKeyHeads = AUGUR_CUDA_QWEN35_LINEAR_STEP_KEY_HEADS_V1;
constexpr uint32_t kLinearValueHeads = AUGUR_CUDA_QWEN35_LINEAR_STEP_VALUE_HEADS_V1;
constexpr uint32_t kLinearHeadDim = AUGUR_CUDA_QWEN35_LINEAR_STEP_HEAD_DIM_V1;
constexpr uint32_t kLinearKernelSize = AUGUR_CUDA_QWEN35_LINEAR_STEP_KERNEL_SIZE_V1;
constexpr uint32_t kLinearQkvElements = AUGUR_CUDA_QWEN35_LINEAR_STEP_QKV_ELEMENTS_V1;
constexpr uint32_t kLinearOutputElements = AUGUR_CUDA_QWEN35_LINEAR_STEP_OUTPUT_ELEMENTS_V1;
constexpr uint32_t kLinearStateElements = AUGUR_CUDA_QWEN35_LINEAR_STEP_STATE_ELEMENTS_V1;
constexpr uint32_t kLinearHistoryElements =
    AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_HISTORY_ELEMENTS_V1;
constexpr uint32_t kLinearConvWeightElements =
    AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_WEIGHT_ELEMENTS_V1;
constexpr uint32_t kFullQueryHeads =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1;
constexpr uint32_t kFullKvHeads =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1;
constexpr uint32_t kFullHeadDim =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1;
constexpr uint32_t kFullRotaryDim =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1;
constexpr uint32_t kFullQueryElements =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1;
constexpr uint32_t kFullKvElements =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1;
constexpr uint32_t kFullMaxCapacity =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_MAX_CAPACITY_V1;
#if defined(AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION)
constexpr uint32_t kKv8MaxCapacity =
    AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY_V3;
constexpr uint32_t kFullTilePositions = 8U;
constexpr uint32_t kKv8BlockElements = AUGUR_CUDA_QWEN35_KV8_BLOCK_ELEMENTS_V3;
constexpr uint32_t kKv8BlockBytes = AUGUR_CUDA_QWEN35_KV8_BLOCK_BYTES_V3;
constexpr uint32_t kKv8BlocksPerHead = AUGUR_CUDA_QWEN35_KV8_BLOCKS_PER_HEAD_V3;
constexpr uint32_t kKv8HeadRowBytes = AUGUR_CUDA_QWEN35_KV8_HEAD_ROW_BYTES_V3;
constexpr uint32_t kKv8CacheRowBytes = AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3;
static_assert(kKv8BlockElements == 32 && kKv8BlockBytes == 34,
    "Qwen prefill KV8 block geometry drift");
static_assert(kKv8BlocksPerHead * kKv8BlockElements == kFullHeadDim,
    "Qwen prefill KV8 head geometry drift");
static_assert(kFullKvHeads * kKv8HeadRowBytes == kKv8CacheRowBytes,
    "Qwen prefill KV8 row geometry drift");
#endif
static_assert(kLinearKeyHeads == 16 && kLinearValueHeads == 48
        && kLinearHeadDim == 128 && kLinearKernelSize == 4,
    "Qwen prefill linear geometry drift");
static_assert(kFullQueryHeads == 24 && kFullKvHeads == 4
        && kFullHeadDim == 256 && kFullRotaryDim == 64,
    "Qwen prefill full-attention geometry drift");

struct Range {
    uint64_t address;
    uint64_t bytes;
    uint64_t alignment;
};

augur_cuda_result_v1 result(augur_cuda_status_t status, int32_t raw_status) {
    return augur_cuda_result_v1{status, raw_status};
}

augur_cuda_result_v1 cuda_result(augur_cuda_status_t status, cudaError_t error) {
    return result(status, static_cast<int32_t>(error));
}

augur_cuda_result_v1 blas_result(augur_cuda_status_t status, cublasStatus_t error) {
    return result(status, static_cast<int32_t>(error));
}

bool checked_add(uint64_t left, uint64_t right, uint64_t* sum) {
    if (sum == nullptr || right > std::numeric_limits<uint64_t>::max() - left) {
        return false;
    }
    *sum = left + right;
    return true;
}

bool checked_mul(uint64_t left, uint64_t right, uint64_t* product) {
    if (product == nullptr
        || (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)) {
        return false;
    }
    *product = left * right;
    return true;
}

bool range_valid(Range range) {
    uint64_t end = 0;
    return range.address != 0 && range.bytes != 0 && range.alignment != 0
        && range.address % range.alignment == 0
        && checked_add(range.address, range.bytes, &end);
}

bool ranges_overlap(Range left, Range right) {
    uint64_t left_end = 0;
    uint64_t right_end = 0;
    if (!checked_add(left.address, left.bytes, &left_end)
        || !checked_add(right.address, right.bytes, &right_end)) {
        return true;
    }
    return left.address < right_end && right.address < left_end;
}

bool device_range(Range range, int32_t ordinal) {
    if (!range_valid(range) || range.bytes > SIZE_MAX) return false;
    const auto* pointer = reinterpret_cast<const void*>(
        static_cast<uintptr_t>(range.address));
    cudaPointerAttributes attributes{};
    cudaError_t error = cudaPointerGetAttributes(&attributes, pointer);
    if (error != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != ordinal) {
        return false;
    }
    CUdeviceptr allocation_base = 0;
    size_t allocation_bytes = 0;
    const CUresult driver_error = cuMemGetAddressRange(
        &allocation_base, &allocation_bytes,
        static_cast<CUdeviceptr>(range.address));
    if (driver_error != CUDA_SUCCESS) return false;
    const uint64_t base = static_cast<uint64_t>(allocation_base);
    const uint64_t offset = range.address >= base ? range.address - base : UINT64_MAX;
    return offset <= allocation_bytes && range.bytes <= allocation_bytes - offset;
}

bool all_disjoint(const Range* ranges, size_t count) {
    for (size_t left = 0; left < count; ++left) {
        if (!range_valid(ranges[left])) return false;
        for (size_t right = left + 1; right < count; ++right) {
            if (ranges_overlap(ranges[left], ranges[right])) return false;
        }
    }
    return true;
}

bool sm70_device(int32_t ordinal) {
    if (ordinal < 0 || cudaSetDevice(ordinal) != cudaSuccess) return false;
    cudaDeviceProp properties{};
    return cudaGetDeviceProperties(&properties, ordinal) == cudaSuccess
        && properties.major == 7 && properties.minor == 0;
}

bool plan_metadata_valid(
    const augur_cuda_qwen35_prefill_cublas_plan_args_v1* args) {
    if (args == nullptr
        || args->abi_version
            != AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_PLAN_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args)
        || args->max_tokens == 0
        || args->max_tokens > AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1
        || args->reserved0 != 0 || args->reserved[0] != 0 || args->reserved[1] != 0
        || args->decoded_weight_len_bytes % sizeof(__half) != 0
        || args->decoded_input_len_bytes % sizeof(__half) != 0) {
        return false;
    }
    const Range ranges[] = {
        {args->decoded_weight_f16, args->decoded_weight_len_bytes, 16},
        {args->decoded_input_f16, args->decoded_input_len_bytes, 16},
    };
    return all_disjoint(ranges, 2);
}

bool plan_preflight_valid(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_cublas_plan_args_v1* args) {
    if (stream == nullptr || stream->stream == nullptr || stream->ordinal != ordinal
        || !plan_metadata_valid(args) || !sm70_device(ordinal)) {
        return false;
    }
    const Range ranges[] = {
        {args->decoded_weight_f16, args->decoded_weight_len_bytes, 16},
        {args->decoded_input_f16, args->decoded_input_len_bytes, 16},
    };
    return device_range(ranges[0], ordinal) && device_range(ranges[1], ordinal);
}

struct Footprint {
    uint64_t weight_bytes;
    uint64_t input_bytes;
    uint64_t output_bytes;
    uint64_t decoded_weight_bytes;
    uint64_t decoded_input_bytes;
    uint64_t weight_elements;
    uint64_t input_elements;
};

bool projection_footprint(
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args,
    Footprint* footprint, bool input_f32 = false) {
    if (args == nullptr || footprint == nullptr || args->tokens == 0
        || args->rows == 0 || args->cols == 0
        || args->rows > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || args->cols > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || args->tokens > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    const uint32_t block_elements = args->weight_format
            == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1
        ? kQ8BlockElements
        : args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1
            ? kNvfp4BlockElements
            : args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1
                ? 1U
            : 0;
    const uint32_t block_bytes = args->weight_format
            == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1
        ? kQ8WeightBlockBytes
        : args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1
            ? kNvfp4BlockBytes
            : args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1
                ? sizeof(uint16_t)
            : 0;
    if (block_elements == 0 || args->cols % block_elements != 0
        || args->cols % kQ8BlockElements != 0) return false;

    uint64_t weight_blocks = 0;
    if (!checked_mul(args->rows, args->cols / block_elements, &weight_blocks)
        || !checked_mul(weight_blocks, block_bytes, &footprint->weight_bytes)
        || !checked_mul(args->rows, args->cols, &footprint->weight_elements)
        || !checked_mul(footprint->weight_elements, sizeof(__half),
            &footprint->decoded_weight_bytes)
        || !checked_mul(args->tokens, args->cols, &footprint->input_elements)
        || !checked_mul(footprint->input_elements, sizeof(__half),
            &footprint->decoded_input_bytes)) {
        return false;
    }
    if (input_f32) {
        if (!checked_mul(footprint->input_elements, sizeof(float),
                &footprint->input_bytes)) return false;
    } else if (!checked_mul(footprint->input_elements / kQ8BlockElements,
            kQ8ActivationBlockBytes, &footprint->input_bytes)) {
        return false;
    }
    uint64_t output_elements = 0;
    return checked_mul(args->tokens, args->rows, &output_elements)
        && checked_mul(output_elements, sizeof(float), &footprint->output_bytes)
        && footprint->weight_bytes <= SIZE_MAX && footprint->input_bytes <= SIZE_MAX
        && footprint->output_bytes <= SIZE_MAX
        && footprint->decoded_weight_bytes <= SIZE_MAX
        && footprint->decoded_input_bytes <= SIZE_MAX;
}

bool projection_metadata_valid(int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args,
    bool inspect_device, Footprint* footprint, bool input_f32 = false) {
    if (plan == nullptr || plan->handle == nullptr || stream == nullptr
        || stream->stream == nullptr || plan->ordinal != ordinal
        || plan->owner != stream || stream->ordinal != ordinal || args == nullptr
        || args->abi_version
            != AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_PROJECTION_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args) || args->reserved != 0
        || args->tokens > plan->snapshot.max_tokens
        || !projection_footprint(args, footprint, input_f32)
        || args->weight_len_bytes != footprint->weight_bytes
        || args->input_len_bytes != footprint->input_bytes
        || args->output_len_bytes != footprint->output_bytes
        || plan->snapshot.decoded_weight_len_bytes < footprint->decoded_weight_bytes
        || plan->snapshot.decoded_input_len_bytes < footprint->decoded_input_bytes) {
        return false;
    }
    if (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1) {
        if (args->tensor_scale != 1.0f) return false;
    } else if (args->weight_format
            == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1) {
        if (!std::isfinite(args->tensor_scale) || args->tensor_scale <= 0.0f) {
            return false;
        }
    } else if (args->weight_format
            == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1) {
        if (args->tensor_scale != 1.0f) return false;
    } else {
        return false;
    }

    const uint64_t weight_alignment = args->weight_format
            == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1
        ? alignof(uint32_t)
        : alignof(uint16_t);
    const Range ranges[] = {
        {args->weight, args->weight_len_bytes, weight_alignment},
        {args->input_q8_1, args->input_len_bytes, alignof(uint32_t)},
        {args->output_f32, args->output_len_bytes, alignof(float)},
        {plan->snapshot.decoded_weight_f16,
            plan->snapshot.decoded_weight_len_bytes, 16},
        {plan->snapshot.decoded_input_f16,
            plan->snapshot.decoded_input_len_bytes, 16},
    };
    if (!all_disjoint(ranges, 5)) return false;
    if (!inspect_device) return true;
    for (const Range range : ranges) {
        if (!device_range(range, ordinal)) return false;
    }
    return true;
}

// Packed-only small batches have no cuBLAS handle or decoded FP16 workspace.
// Keep their admission separate so a large output head cannot silently grow
// the ordinary prefill plan's scratch or bypass that plan's existing checks.
bool packed_projection_metadata_valid(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args,
    uint32_t exact_tiles, bool inspect_device) {
    Footprint footprint{};
    if (ordinal < 0 || stream == nullptr || stream->stream == nullptr
        || stream->ordinal != ordinal || args == nullptr || exact_tiles > 1U
        || args->abi_version != AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_PROJECTION_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args) || args->reserved != 0
        || args->tokens > 8U
        || (args->weight_format != AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1
            && args->weight_format != AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1)
        || !projection_footprint(args, &footprint)
        || args->weight_len_bytes != footprint.weight_bytes
        || args->input_len_bytes != footprint.input_bytes
        || args->output_len_bytes != footprint.output_bytes
        || !std::isfinite(args->tensor_scale) || args->tensor_scale <= 0.0f
        || (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1
            && args->tensor_scale != 1.0f)) return false;
    const Range ranges[] = {
        {args->weight, args->weight_len_bytes,
            args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1 ? 4U : 2U},
        {args->input_q8_1, args->input_len_bytes, 4U},
        {args->output_f32, args->output_len_bytes, 4U},
    };
    if (!all_disjoint(ranges, 3)) return false;
    if (!inspect_device) return true;
    if (!sm70_device(ordinal)) return false;
    for (const Range range : ranges) if (!device_range(range, ordinal)) return false;
    return true;
}

__device__ __forceinline__ float e2m1_value(uint8_t nibble) {
    const uint8_t magnitude = nibble & 0x07U;
    const uint32_t doubled = magnitude < 4U
        ? magnitude
        : (2U + (magnitude & 1U)) << ((magnitude >> 1U) - 1U);
    const float value = static_cast<float>(doubled) * 0.5f;
    return (nibble & 0x08U) != 0 ? -value : value;
}

__device__ __forceinline__ float ue4m3_scale_to_f32(uint8_t value) {
    if (value == 0 || value == 0x7f) return 0.0f;
    const uint32_t exponent = (static_cast<uint32_t>(value) >> 3) & 0x0fU;
    const uint32_t mantissa = static_cast<uint32_t>(value) & 0x07U;
    if (exponent == 0) return static_cast<float>(mantissa) * 0x1p-9f;
    return __uint_as_float(((exponent + 120U) << 23U) | (mantissa << 20U));
}

// Experimental small-N path. Keep the canonical 128-thread matvec block
// assignment, FMA order and cross-warp reduction independently for every token,
// but load/decode each packed weight segment once for up to eight token rows.
__device__ __forceinline__ uint32_t small_batch_q8_word(const uint8_t* p) {
    const auto* halves = reinterpret_cast<const uint16_t*>(p);
    return static_cast<uint32_t>(halves[0]) | (static_cast<uint32_t>(halves[1]) << 16U);
}

__device__ __forceinline__ int2 small_batch_e2m1_words(uint32_t packed) {
    const uint32_t table[4] = {0x03020100U, 0x0c080604U, 0xfdfeff00U, 0xf4f8fafcU};
    const uint32_t selection = 0x32103210U | ((packed & 0x88888888U) >> 1U);
    uint32_t temporary[2];
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t low = __byte_perm(table[0], table[1], packed >> (16U * i));
        const uint32_t high = __byte_perm(table[2], table[3], packed >> (16U * i));
        temporary[i] = __byte_perm(low, high, selection >> (16U * i));
    }
    return make_int2(static_cast<int32_t>(__byte_perm(temporary[0], temporary[1], 0x6420U)),
                    static_cast<int32_t>(__byte_perm(temporary[0], temporary[1], 0x7531U)));
}

template<uint32_t Format>
__global__ __launch_bounds__(128, 1) void quantized_small_batch_kernel(
    const uint8_t* weight, const uint8_t* activation, float* output,
    uint32_t rows, uint32_t cols, uint32_t tokens, float tensor_scale) {
    const uint32_t row = blockIdx.x;
    const uint32_t thread = threadIdx.x, lane = thread & 31U, warp = thread / 32U;
    const uint64_t activation_stride = static_cast<uint64_t>(cols / 32U) * 36U;
    float partial[8] = {};
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
            for (uint32_t t = 0; t < 8; ++t) {
                if (t >= tokens) continue;
                const uint8_t* a = activation + t * activation_stride + static_cast<uint64_t>(block) * 36U;
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
                for (uint32_t t = 0; t < 8; ++t) {
                    if (t >= tokens) continue;
                    const uint8_t* a = activation + t * activation_stride
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
    __shared__ float other_warps[8][3][32];
    if (warp > 0U) {
        #pragma unroll
        for (uint32_t t = 0; t < 8; ++t) other_warps[t][warp - 1U][lane] = partial[t];
    }
    __syncthreads();
    if (warp > 0U) return;
    #pragma unroll
    for (uint32_t t = 0; t < 8; ++t) {
        if (t >= tokens) continue;
        #pragma unroll
        for (uint32_t other = 0; other < 3; ++other) partial[t] += other_warps[t][other][lane];
        #pragma unroll
        for (uint32_t offset = 16; offset != 0; offset >>= 1U)
            partial[t] += __shfl_down_sync(0xffffffffU, partial[t], offset);
        if (lane == 0U) output[static_cast<uint64_t>(t) * rows + row] =
            Format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1 ? partial[t] * tensor_scale : partial[t];
    }
}

#include "augur_qwen35_exact_small_batch_tiles.cuh"
#include "augur_qwen35_exact_projection_row_reuse.cuh"
#include "augur_qwen35_exact_ffn_down_row_reuse.cuh"

// Byte-exact resident layout; original canonical K/FMA/four-warp reduction.
// Conversion is load-time only. This kernel owns no scratch or allocation.
template<uint32_t Tile, bool FullTile>
__global__ __launch_bounds__(128, 1) void planar_q8_head_kernel(
    const uint8_t* weight, const uint8_t* activation, float* output, uint32_t tokens) {
    constexpr uint32_t rows = 248320, cols = 5120, blocks = cols / 32;
    const uint32_t row = blockIdx.x, thread = threadIdx.x;
    const uint32_t lane = thread & 31U, warp = thread / 32U;
    const auto* scales = reinterpret_cast<const uint16_t*>(weight + uint64_t(rows) * cols);
    float partial[Tile] = {};
    for (uint32_t block = thread / 4U; block < blocks; block += 32U) {
        const uint32_t offset = (thread & 3U) * 8U;
        const auto* code = reinterpret_cast<const int32_t*>(weight + uint64_t(row)*cols + block*32U + offset);
        const int32_t w0 = code[0], w1 = code[1];
        const float scale = __half2float(__ushort_as_half(scales[uint64_t(row)*blocks + block]));
        #pragma unroll
        for (uint32_t t = 0; t < Tile; ++t) {
            if (!FullTile && t >= tokens) continue;
            const auto* a = activation + (uint64_t(t)*blocks + block)*36U;
            const auto* x = reinterpret_cast<const int32_t*>(a + 4U + offset);
            int32_t dot = __dp4a(w0, x[0], 0);
            dot = __dp4a(w1, x[1], dot);
            const float combined = scale * __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(a)));
            partial[t] = fmaf(combined, static_cast<float>(dot), partial[t]);
        }
    }
    __shared__ float other_warps[Tile][3][32];
    if (warp > 0U) {
        #pragma unroll
        for (uint32_t t = 0; t < Tile; ++t) other_warps[t][warp-1U][lane] = partial[t];
    }
    __syncthreads();
    if (warp > 0U) return;
    #pragma unroll
    for (uint32_t t = 0; t < Tile; ++t) {
        if (!FullTile && t >= tokens) continue;
        #pragma unroll
        for (uint32_t other = 0; other < 3; ++other) partial[t] += other_warps[t][other][lane];
        #pragma unroll
        for (uint32_t offset = 16; offset; offset >>= 1U)
            partial[t] += __shfl_down_sync(0xffffffffU, partial[t], offset);
        if (lane == 0U) output[uint64_t(t)*rows + row] = partial[t];
    }
}

bool planar_q8_head_metadata_valid(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args, bool inspect_device) {
    if (args == nullptr || args->weight_format != AUGUR_CUDA_QWEN35_PLANAR_Q8_HEAD_WEIGHT_V1
        || args->rows != 248320U || args->cols != 5120U || args->weight % 4U != 0U)
        return false;
    // Extents are identical, not the layout. Only this dedicated entry maps
    // metadata to the shared allocation/range checker; old dispatch rejects4.
    auto logical = *args;
    logical.weight_format = AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1;
    return packed_projection_metadata_valid(ordinal, stream, &logical, 1U, inspect_device);
}

// Only the measured full8 input/gate-up family is promoted to the default-off
// row2 candidate. Heads, FFN-down, unsupported formats/shapes and tails retain
// the existing exact selector and canonical fallback.
uint32_t projection_row_reuse_width(uint32_t format, uint32_t rows,
    uint32_t cols, uint32_t tokens) {
    if (tokens != 8U || cols != 5120U) return 0U;
    if (rows == 17408U && (format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1
        || format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1)) return 2U;
    if (format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1
        && (rows == 10240U || rows == 6144U || rows == 12288U)) return 2U;
    return 0U;
}

// Frozen component-tested shape set. Unknown shapes and tails stay canonical.
uint32_t exact_projection_tile_width(uint32_t format, uint32_t rows,
    uint32_t cols, uint32_t tokens) {
    if (tokens != 8U) return 0U;
    if (format != AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1
        && format != AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1) return 0U;
    if (rows == 5120U && cols == 17408U) return 4U;
    if (rows == 17408U && cols == 5120U) return 8U;
    if (format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1
        && ((cols == 5120U && (rows == 10240U || rows == 6144U
            || rows == 12288U || rows == 1024U || rows == 248320U))
            || (rows == 5120U && cols == 6144U))) return 8U;
    return 0U;
}

__device__ __forceinline__ float compare_summary_value(const uint8_t* p, uint64_t index, uint32_t format) {
    if (format == 0U) return reinterpret_cast<const float*>(p)[index];
    const uint8_t* block = p + (index / 32U) * 34U;
    return __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(block)))
        * static_cast<float>(static_cast<int8_t>(block[2U + index % 32U]));
}

// One CTA per tensor, diagnostic only. FP64 reductions keep the summary stable
// without copying the tensors to CPU. No atomic accumulation or publication.
__global__ void compare_summary_kernel(const uint8_t* candidate, const uint8_t* reference,
    uint64_t elements, uint32_t format, uint64_t* output) {
    const uint32_t t = threadIdx.x;
    uint64_t mismatches = 0, nonfinite = 0;
    double maximum = 0, delta = 0, ref = 0, cand = 0, dot = 0;
    for (uint64_t i = t; i < elements; i += 256U) {
        const float a = compare_summary_value(candidate, i, format);
        const float b = compare_summary_value(reference, i, format);
        mismatches += __float_as_uint(a) != __float_as_uint(b);
        if (!isfinite(a) || !isfinite(b)) { ++nonfinite; continue; }
        const double d = static_cast<double>(a) - static_cast<double>(b);
        maximum = fmax(maximum, fabs(d));
        delta += d * d;
        ref += static_cast<double>(b) * b;
        cand += static_cast<double>(a) * a;
        dot += static_cast<double>(a) * b;
    }
    __shared__ uint64_t counts[2][256];
    __shared__ double sums[5][256];
    counts[0][t] = mismatches; counts[1][t] = nonfinite;
    sums[0][t] = maximum; sums[1][t] = delta; sums[2][t] = ref; sums[3][t] = cand; sums[4][t] = dot;
    __syncthreads();
    for (uint32_t offset = 128; offset != 0; offset >>= 1U) {
        if (t < offset) {
            counts[0][t] += counts[0][t + offset]; counts[1][t] += counts[1][t + offset];
            sums[0][t] = fmax(sums[0][t], sums[0][t + offset]);
            #pragma unroll
            for (uint32_t field = 1; field < 5; ++field) sums[field][t] += sums[field][t + offset];
        }
        __syncthreads();
    }
    if (t == 0) {
        output[0] = elements; output[1] = counts[0][0]; output[2] = counts[1][0];
        #pragma unroll
        for (uint32_t field = 0; field < 5; ++field) reinterpret_cast<double*>(output)[3U + field] = sums[field][0];
    }
}

__global__ void decode_q8_0_weight_kernel(
    const uint8_t* source, __half* destination, uint64_t elements) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x
            + threadIdx.x;
         index < elements; index += stride) {
        const uint64_t block = index / kQ8BlockElements;
        const uint32_t lane = static_cast<uint32_t>(index % kQ8BlockElements);
        const uint8_t* bytes = source + block * kQ8WeightBlockBytes;
        const uint16_t scale_bits = static_cast<uint16_t>(bytes[0])
            | static_cast<uint16_t>(bytes[1]) << 8U;
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const int8_t code = static_cast<int8_t>(bytes[2U + lane]);
        destination[index] = __float2half_rn(scale * static_cast<float>(code));
    }
}

// Adapted from ggml convert.cu dequantize_block_q8_0_f16.
// Copyright (c) 2023-2026 The ggml authors. MIT; full license in NOTICE.
// Admission below requires complete 2048-element groups (no tail reads).
template<unsigned Threads>
__global__ __launch_bounds__(Threads) void decode_q8_0_llama_coalesced_kernel(
    const uint8_t* source, __half* destination) {
    constexpr unsigned values = 2048, packed_words = 544;
    __shared__ uint32_t packed[packed_words];
    const auto* words = reinterpret_cast<const uint32_t*>(source)
        + uint64_t(blockIdx.x) * packed_words;
    for (unsigned i = threadIdx.x; i < packed_words; i += Threads) packed[i] = words[i];
    __syncthreads();
    auto* output = reinterpret_cast<__half2*>(destination + uint64_t(blockIdx.x) * values);
    #pragma unroll
    for (unsigned i = 0; i < values / 2; i += Threads) {
        const unsigned element = (i + threadIdx.x) * 2;
        const auto* block = reinterpret_cast<const uint8_t*>(packed) + (element / 32) * 34;
        const __half scale = __ushort_as_half(*reinterpret_cast<const uint16_t*>(block));
        const auto code = *reinterpret_cast<const char2*>(block + 2 + (element % 32));
        output[i + threadIdx.x] = __hmul2(
            __floats2half2_rn(float(code.x), float(code.y)), __half2half2(scale));
    }
}

__global__ void decode_q8_1_input_kernel(
    const uint8_t* source, __half* destination, uint64_t elements) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x
            + threadIdx.x;
         index < elements; index += stride) {
        const uint64_t block = index / kQ8BlockElements;
        const uint32_t lane = static_cast<uint32_t>(index % kQ8BlockElements);
        const uint8_t* bytes = source + block * kQ8ActivationBlockBytes;
        const uint16_t scale_bits = static_cast<uint16_t>(bytes[0])
            | static_cast<uint16_t>(bytes[1]) << 8U;
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const int8_t code = static_cast<int8_t>(bytes[4U + lane]);
        destination[index] = __float2half_rn(scale * static_cast<float>(code));
    }
}

__global__ void decode_nvfp4_weight_kernel(const uint8_t* source,
    __half* destination, uint64_t elements, float tensor_scale) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x
            + threadIdx.x;
         index < elements; index += stride) {
        const uint64_t block = index / kNvfp4BlockElements;
        const uint32_t lane = static_cast<uint32_t>(index % kNvfp4BlockElements);
        const uint32_t sub_block = lane / 16U;
        const uint32_t sub_lane = lane % 16U;
        const uint8_t* bytes = source + block * kNvfp4BlockBytes;
        const uint8_t packed = bytes[4U + sub_block * 8U + (sub_lane & 7U)];
        const uint8_t nibble = sub_lane < 8U ? packed & 0x0fU : packed >> 4U;
        const float value = e2m1_value(nibble)
            * ue4m3_scale_to_f32(bytes[sub_block]) * tensor_scale;
        destination[index] = __float2half_rn(value);
    }
}

__global__ void decode_bf16_weight_kernel(
    const uint16_t* source, __half* destination, uint64_t elements) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x
            + threadIdx.x;
         index < elements; index += stride) {
        destination[index] = __float2half_rn(
            __uint_as_float(static_cast<uint32_t>(source[index]) << 16U));
    }
}

// Two adjacent values per thread. Arithmetic stays scalar FP32 with the same
// RN conversion; half2 is a store format, not a lower-precision computation.
template<uint32_t Format>
__global__ void decode_weight_pair_kernel(const uint8_t* source,
    __half* destination, uint64_t elements, float tensor_scale) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t pair = static_cast<uint64_t>(blockIdx.x) * blockDim.x
            + threadIdx.x; pair < elements / 2U; pair += stride) {
        const uint64_t index = pair * 2U;
        float first;
        float second;
        if constexpr (Format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1) {
            const uint64_t block = index / kQ8BlockElements;
            const uint32_t lane = static_cast<uint32_t>(index % kQ8BlockElements);
            const uint8_t* bytes = source + block * kQ8WeightBlockBytes;
            const uint16_t bits = static_cast<uint16_t>(bytes[0])
                | static_cast<uint16_t>(bytes[1]) << 8U;
            const float scale = __half2float(__ushort_as_half(bits));
            first = scale * static_cast<float>(static_cast<int8_t>(bytes[2U + lane]));
            second = scale * static_cast<float>(static_cast<int8_t>(bytes[3U + lane]));
        } else if constexpr (Format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1) {
            const uint64_t block = index / kNvfp4BlockElements;
            const uint32_t lane = static_cast<uint32_t>(index % kNvfp4BlockElements);
            const uint32_t sub_block = lane / 16U;
            const uint32_t sub_lane = lane % 16U;
            const uint8_t* bytes = source + block * kNvfp4BlockBytes;
            const uint32_t offset = 4U + sub_block * 8U + (sub_lane & 7U);
            const uint32_t shift = sub_lane < 8U ? 0U : 4U;
            const float scale = ue4m3_scale_to_f32(bytes[sub_block]);
            first = e2m1_value((bytes[offset] >> shift) & 0x0fU) * scale * tensor_scale;
            second = e2m1_value((bytes[offset + 1U] >> shift) & 0x0fU) * scale * tensor_scale;
        } else {
            const uint16_t* words = reinterpret_cast<const uint16_t*>(source);
            first = __uint_as_float(static_cast<uint32_t>(words[index]) << 16U);
            second = __uint_as_float(static_cast<uint32_t>(words[index + 1U]) << 16U);
        }
        reinterpret_cast<__half2*>(destination)[pair] = __halves2half2(
            __float2half_rn(first), __float2half_rn(second));
    }
}

template<uint32_t Format>
__global__ void decode_weight_quad_kernel(const uint8_t* source,
    __half* destination, uint64_t elements, float tensor_scale) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t quad = static_cast<uint64_t>(blockIdx.x) * blockDim.x
            + threadIdx.x; quad < elements / 4U; quad += stride) {
        const uint64_t index = quad * 4U;
        float values[4];
        if constexpr (Format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1) {
            const uint8_t* bytes = source + index / kQ8BlockElements * kQ8WeightBlockBytes;
            const uint32_t lane = static_cast<uint32_t>(index % kQ8BlockElements);
            const uint16_t bits = static_cast<uint16_t>(bytes[0])
                | static_cast<uint16_t>(bytes[1]) << 8U;
            const float scale = __half2float(__ushort_as_half(bits));
            #pragma unroll
            for (uint32_t item = 0; item < 4U; ++item) {
                values[item] = scale * static_cast<float>(static_cast<int8_t>(bytes[2U + lane + item]));
            }
        } else if constexpr (Format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1) {
            const uint8_t* bytes = source + index / kNvfp4BlockElements * kNvfp4BlockBytes;
            const uint32_t lane = static_cast<uint32_t>(index % kNvfp4BlockElements);
            const uint32_t sub_block = lane / 16U;
            const uint32_t sub_lane = lane % 16U;
            const uint32_t offset = 4U + sub_block * 8U + (sub_lane & 7U);
            const uint32_t shift = sub_lane < 8U ? 0U : 4U;
            const float scale = ue4m3_scale_to_f32(bytes[sub_block]);
            // Native NVFP4 input alignment is4, block stride36 and offset is4-aligned.
            const uint32_t packed = *reinterpret_cast<const uint32_t*>(bytes + offset);
            #pragma unroll
            for (uint32_t item = 0; item < 4U; ++item) {
                values[item] = e2m1_value((packed >> (8U * item + shift)) & 0x0fU)
                    * scale * tensor_scale;
            }
        } else {
            const uint16_t* words = reinterpret_cast<const uint16_t*>(source);
            #pragma unroll
            for (uint32_t item = 0; item < 4U; ++item) {
                values[item] = __uint_as_float(static_cast<uint32_t>(words[index + item]) << 16U);
            }
        }
        const uint32_t lo = static_cast<uint32_t>(__half_as_ushort(__float2half_rn(values[0])))
            | static_cast<uint32_t>(__half_as_ushort(__float2half_rn(values[1]))) << 16U;
        const uint32_t hi = static_cast<uint32_t>(__half_as_ushort(__float2half_rn(values[2])))
            | static_cast<uint32_t>(__half_as_ushort(__float2half_rn(values[3]))) << 16U;
        reinterpret_cast<uint2*>(destination)[quad] = make_uint2(lo, hi);
    }
}

uint32_t launch_blocks(uint64_t elements) {
    const uint64_t needed = (elements + kThreads - 1U) / kThreads;
    return static_cast<uint32_t>(needed < kMaxBlocks ? needed : kMaxBlocks);
}

#include "augur_qwen35_prefill_compensated_tensor.cuh"
#include "augur_qwen35_prefill_integer_mma.cuh"
#include "augur_qwen35_prefill_integer_q8.cuh"


__global__ void pack_f32_activation_kernel(
    const float* source, __half* destination, uint64_t elements) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x
            + threadIdx.x; index < elements; index += stride) {
        destination[index] = __float2half_rn(source[index]);
    }
}

augur_cuda_result_v1 launch(int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args,
    const Footprint& footprint, bool input_f32 = false, uint32_t stage = 3U,
    bool paired_weight_decode = false, bool quad_weight_decode = false,
    bool llama_q8_decode = false) {
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) {
        return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    }
    const auto* weight = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(args->weight));
    const auto* input = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(args->input_q8_1));
    auto* decoded_weight = reinterpret_cast<__half*>(
        static_cast<uintptr_t>(plan->snapshot.decoded_weight_f16));
    auto* decoded_input = reinterpret_cast<__half*>(
        static_cast<uintptr_t>(plan->snapshot.decoded_input_f16));
    auto* output = reinterpret_cast<float*>(
        static_cast<uintptr_t>(args->output_f32));

    if (stage == 0U || stage == 3U) {
    if (llama_q8_decode) {
        decode_q8_0_llama_coalesced_kernel<32><<<
            static_cast<uint32_t>(footprint.weight_elements / 2048U),
            32, 0, stream->stream>>>(weight, decoded_weight);
    } else if (quad_weight_decode) {
        const uint32_t blocks = launch_blocks(footprint.weight_elements / 4U);
        if (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1) {
            decode_weight_quad_kernel<AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1>
                <<<blocks, kThreads, 0, stream->stream>>>(weight, decoded_weight,
                    footprint.weight_elements, args->tensor_scale);
        } else if (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1) {
            decode_weight_quad_kernel<AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1>
                <<<blocks, kThreads, 0, stream->stream>>>(weight, decoded_weight,
                    footprint.weight_elements, args->tensor_scale);
        } else {
            decode_weight_quad_kernel<AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1>
                <<<blocks, kThreads, 0, stream->stream>>>(weight, decoded_weight,
                    footprint.weight_elements, args->tensor_scale);
        }
    } else if (paired_weight_decode) {
        const uint32_t blocks = launch_blocks(footprint.weight_elements / 2U);
        if (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1) {
            decode_weight_pair_kernel<AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1>
                <<<blocks, kThreads, 0, stream->stream>>>(weight, decoded_weight,
                    footprint.weight_elements, args->tensor_scale);
        } else if (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1) {
            decode_weight_pair_kernel<AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1>
                <<<blocks, kThreads, 0, stream->stream>>>(weight, decoded_weight,
                    footprint.weight_elements, args->tensor_scale);
        } else {
            decode_weight_pair_kernel<AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1>
                <<<blocks, kThreads, 0, stream->stream>>>(weight, decoded_weight,
                    footprint.weight_elements, args->tensor_scale);
        }
    } else {
    if (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1) {
        decode_q8_0_weight_kernel<<<launch_blocks(footprint.weight_elements),
            kThreads, 0, stream->stream>>>(weight, decoded_weight,
                footprint.weight_elements);
    } else if (args->weight_format
            == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1) {
        decode_nvfp4_weight_kernel<<<launch_blocks(footprint.weight_elements),
            kThreads, 0, stream->stream>>>(weight, decoded_weight,
                footprint.weight_elements, args->tensor_scale);
    } else {
        decode_bf16_weight_kernel<<<launch_blocks(footprint.weight_elements),
            kThreads, 0, stream->stream>>>(
                reinterpret_cast<const uint16_t*>(weight), decoded_weight,
                footprint.weight_elements);
    }
    }
    }
    if (stage == 1U || stage == 3U) {
    if (input_f32) {
        pack_f32_activation_kernel<<<launch_blocks(footprint.input_elements),
            kThreads, 0, stream->stream>>>(reinterpret_cast<const float*>(input),
            decoded_input, footprint.input_elements);
    } else
    {
        decode_q8_1_input_kernel<<<launch_blocks(footprint.input_elements),
            kThreads, 0, stream->stream>>>(input, decoded_input,
                footprint.input_elements);
    }
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    }

    if (stage < 2U) return result(AUGUR_CUDA_STATUS_OK, 0);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t blas = cublasGemmEx(
        plan->handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(args->rows),
        static_cast<int>(args->tokens),
        static_cast<int>(args->cols),
        &alpha,
        decoded_weight,
        CUDA_R_16F,
        static_cast<int>(args->cols),
        decoded_input,
        CUDA_R_16F,
        static_cast<int>(args->cols),
        &beta,
        output,
        CUDA_R_32F,
        static_cast<int>(args->rows),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (blas != CUBLAS_STATUS_SUCCESS) {
        return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, blas);
    }
    error = cudaGetLastError();
    return error == cudaSuccess
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

bool rmsnorm_metadata_valid(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_rmsnorm_args_v1* args,
    bool inspect_device) {
    if (stream == nullptr || stream->stream == nullptr || args == nullptr
        || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_QWEN35_PREFILL_RMSNORM_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args) || args->reserved != 0
        || args->tokens == 0
        || args->tokens > AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1
        || args->cols == 0) {
        return false;
    }
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    uint64_t elements = 0;
    uint64_t batch_bytes = 0;
    uint64_t weight_bytes = 0;
    if (!std::isfinite(epsilon) || epsilon <= 0.0f
        || !checked_mul(args->tokens, args->cols, &elements)
        || !checked_mul(elements, sizeof(float), &batch_bytes)
        || !checked_mul(args->cols, sizeof(float), &weight_bytes)
        || args->input_len_bytes != batch_bytes
        || args->weight_len_bytes != weight_bytes
        || args->output_len_bytes != batch_bytes) {
        return false;
    }
    const Range ranges[] = {
        {args->input_f32, args->input_len_bytes, alignof(float)},
        {args->weight_f32, args->weight_len_bytes, alignof(float)},
        {args->output_f32, args->output_len_bytes, alignof(float)},
    };
    if (!all_disjoint(ranges, 3)) return false;
    if (!inspect_device) return true;
    return sm70_device(ordinal) && device_range(ranges[0], ordinal)
        && device_range(ranges[1], ordinal) && device_range(ranges[2], ordinal);
}

__global__ void prefill_rmsnorm_kernel(const float* input,
    const float* weight, float* output, uint32_t cols, float epsilon) {
    const uint32_t token = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint64_t base = static_cast<uint64_t>(token) * cols;
    __shared__ float partial[kThreads];
    float sum = 0.0f;
    for (uint32_t col = lane; col < cols; col += blockDim.x) {
        const float value = input[base + col];
        sum += value * value;
    }
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / static_cast<float>(cols) + epsilon);
    for (uint32_t col = lane; col < cols; col += blockDim.x) {
        output[base + col] = input[base + col] * scale * weight[col];
    }
}

augur_cuda_result_v1 submit_rmsnorm(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_rmsnorm_args_v1* args) {
    if (!rmsnorm_metadata_valid(ordinal, stream, args, false)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    prefill_rmsnorm_kernel<<<args->tokens, kThreads, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->weight_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        args->cols, epsilon);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

bool linear_chunk_metadata_valid(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args,
    bool inspect_device) {
    if (stream == nullptr || stream->stream == nullptr || args == nullptr
        || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version
            != AUGUR_CUDA_QWEN35_PREFILL_LINEAR_CHUNK_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args) || args->reserved0 != 0
        || args->reserved1 != 0 || args->tokens == 0
        || args->tokens > AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1) {
        return false;
    }
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) return false;
    const uint64_t tokens = args->tokens;
    const uint64_t f32 = sizeof(float);
    const uint64_t expected[] = {
        tokens * kLinearQkvElements * f32,
        tokens * kLinearOutputElements * f32,
        static_cast<uint64_t>(kLinearConvWeightElements) * f32,
        static_cast<uint64_t>(kLinearHistoryElements) * f32,
        tokens * kLinearValueHeads * f32,
        tokens * kLinearValueHeads * f32,
        static_cast<uint64_t>(kLinearValueHeads) * f32,
        static_cast<uint64_t>(kLinearValueHeads) * f32,
        static_cast<uint64_t>(kLinearStateElements) * f32,
        static_cast<uint64_t>(kLinearHeadDim) * f32,
        tokens * kLinearQkvElements * f32,
        tokens * kLinearOutputElements * f32,
        tokens * kLinearOutputElements * f32,
    };
    const Range ranges[] = {
        {args->qkv_f32, args->qkv_len_bytes, alignof(float)},
        {args->z_f32, args->z_len_bytes, alignof(float)},
        {args->conv_weights_f32, args->conv_weights_len_bytes, alignof(float)},
        {args->conv_history_f32, args->conv_history_len_bytes, alignof(float)},
        {args->alpha_raw_f32, args->alpha_raw_len_bytes, alignof(float)},
        {args->beta_raw_f32, args->beta_raw_len_bytes, alignof(float)},
        {args->dt_bias_f32, args->dt_bias_len_bytes, alignof(float)},
        {args->ssm_a_f32, args->ssm_a_len_bytes, alignof(float)},
        {args->state_f32, args->state_len_bytes, alignof(float)},
        {args->norm_weight_f32, args->norm_weight_len_bytes, alignof(float)},
        {args->activated_f32, args->activated_len_bytes, alignof(float)},
        {args->recurrent_f32, args->recurrent_len_bytes, alignof(float)},
        {args->output_f32, args->output_len_bytes, alignof(float)},
    };
    for (size_t index = 0; index < 13; ++index) {
        if (ranges[index].bytes != expected[index]) return false;
    }
    if (!all_disjoint(ranges, 13)) return false;
    if (!inspect_device) return true;
    if (!sm70_device(ordinal)) return false;
    for (const Range range : ranges) {
        if (!device_range(range, ordinal)) return false;
    }
    return true;
}

// Read-only checkpoint inputs must not alias any bound operand or each other.
// Device ownership/extents are inspected once by preflight, never at submit.
bool linear_checkpoint_metadata_valid(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args,
    uint64_t history, uint64_t history_bytes, uint64_t state, uint64_t state_bytes,
    bool inspect_device) {
    if (!linear_chunk_metadata_valid(ordinal, stream, args, inspect_device)
        || args->tokens > 8U || history_bytes != args->conv_history_len_bytes
        || state_bytes != args->state_len_bytes) return false;
    const Range sources[] = {{history, history_bytes, 4}, {state, state_bytes, 4}};
    if (!all_disjoint(sources, 2)) return false;
    const Range operands[] = {
        {args->qkv_f32, args->qkv_len_bytes, 4},
        {args->z_f32, args->z_len_bytes, 4},
        {args->conv_weights_f32, args->conv_weights_len_bytes, 4},
        {args->conv_history_f32, args->conv_history_len_bytes, 4},
        {args->alpha_raw_f32, args->alpha_raw_len_bytes, 4},
        {args->beta_raw_f32, args->beta_raw_len_bytes, 4},
        {args->dt_bias_f32, args->dt_bias_len_bytes, 4},
        {args->ssm_a_f32, args->ssm_a_len_bytes, 4},
        {args->state_f32, args->state_len_bytes, 4},
        {args->norm_weight_f32, args->norm_weight_len_bytes, 4},
        {args->activated_f32, args->activated_len_bytes, 4},
        {args->recurrent_f32, args->recurrent_len_bytes, 4},
        {args->output_f32, args->output_len_bytes, 4},
    };
    for (const Range source : sources) {
        for (const Range operand : operands) if (ranges_overlap(source, operand)) return false;
        if (inspect_device && !device_range(source, ordinal)) return false;
    }
    return true;
}

__device__ __forceinline__ float prefill_silu(float value) {
    if (value >= 0.0f) return value / (1.0f + expf(-value));
    const float exponent = expf(value);
    return value * exponent / (1.0f + exponent);
}

__device__ __forceinline__ float prefill_sigmoid(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
    const float exponent = expf(value);
    return exponent / (1.0f + exponent);
}

__device__ __forceinline__ float prefill_softplus(float value) {
    if (value > 20.0f) return value;
    if (value < -20.0f) return expf(value);
    return logf(1.0f + expf(value));
}

__device__ __forceinline__ float prefill_warp_sum(float value) {
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ __forceinline__ double prefill_warp_sum_f64(double value) {
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__global__ void prefill_linear_conv_chunk_kernel(const float* qkv,
    const float* weights, float* history, float* activated, uint32_t tokens) {
    const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= kLinearQkvElements) return;
    const uint32_t history_base = channel * (kLinearKernelSize - 1U);
    const uint32_t weight_base = channel * kLinearKernelSize;
    float h0 = history[history_base];
    float h1 = history[history_base + 1U];
    float h2 = history[history_base + 2U];
    const float w0 = weights[weight_base];
    const float w1 = weights[weight_base + 1U];
    const float w2 = weights[weight_base + 2U];
    const float w3 = weights[weight_base + 3U];
    for (uint32_t token = 0; token < tokens; ++token) {
        const uint64_t index = static_cast<uint64_t>(token)
            * kLinearQkvElements + channel;
        const float current = qkv[index];
        activated[index] = prefill_silu(
            h0 * w0 + h1 * w1 + h2 * w2 + current * w3);
        h0 = h1;
        h1 = h2;
        h2 = current;
    }
    history[history_base] = h0;
    history[history_base + 1U] = h1;
    history[history_base + 2U] = h2;
}

#include "augur_qwen35_augur_prefill_conv.cuh"

__global__ __launch_bounds__(128, 2) void prefill_linear_delta_chunk_kernel(
    const float* activated, const float* alpha_raw, const float* beta_raw,
    const float* dt_bias, const float* ssm_a, float* state,
    float* recurrent, uint32_t tokens, float epsilon) {
    constexpr uint32_t warps_per_block = 4U;
    constexpr uint32_t values_per_lane = kLinearHeadDim / 32U;
    const uint32_t head = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t output_row = blockIdx.y * warps_per_block + threadIdx.y;
    if (head >= kLinearValueHeads || output_row >= kLinearHeadDim) return;

    __shared__ float q_scale;
    __shared__ float k_scale;
    __shared__ float decay;
    __shared__ float beta;
    const uint32_t q_head = head % kLinearKeyHeads;
    float* matrix_row = state
        + (static_cast<uint64_t>(head) * kLinearHeadDim + output_row)
            * kLinearHeadDim;
    float state_values[values_per_lane];
    #pragma unroll
    for (uint32_t item = 0; item < values_per_lane; ++item) {
        state_values[item] = matrix_row[item * 32U + lane];
    }

    for (uint32_t token = 0; token < tokens; ++token) {
        const float* token_activated = activated
            + static_cast<uint64_t>(token) * kLinearQkvElements;
        const float* q = token_activated + q_head * kLinearHeadDim;
        const float* k = token_activated
            + kLinearKeyHeads * kLinearHeadDim + q_head * kLinearHeadDim;
        const float* v = token_activated
            + 2U * kLinearKeyHeads * kLinearHeadDim + head * kLinearHeadDim;
        if (threadIdx.x == 0U && threadIdx.y == 0U) {
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            for (uint32_t col = 0; col < kLinearHeadDim; ++col) {
                q_sum += q[col] * q[col];
                k_sum += k[col] * k[col];
            }
            const float floor = epsilon * epsilon;
            q_scale = rsqrtf(fmaxf(q_sum, floor));
            k_scale = rsqrtf(fmaxf(k_sum, floor));
            const uint64_t gate_index = static_cast<uint64_t>(token)
                * kLinearValueHeads + head;
            decay = expf(prefill_softplus(alpha_raw[gate_index]
                + dt_bias[head]) * ssm_a[head]);
            beta = prefill_sigmoid(beta_raw[gate_index]);
        }
        __syncthreads();

        float projection = 0.0f;
        float q_values[values_per_lane];
        float k_values[values_per_lane];
        #pragma unroll
        for (uint32_t item = 0; item < values_per_lane; ++item) {
            const uint32_t col = item * 32U + lane;
            q_values[item] = q[col] * q_scale;
            k_values[item] = k[col] * k_scale;
            state_values[item] *= decay;
            projection += state_values[item] * k_values[item];
        }
        projection = __shfl_sync(
            0xffffffffU, prefill_warp_sum(projection), 0);
        const float delta = (v[output_row] - projection) * beta;
        float query_projection = 0.0f;
        #pragma unroll
        for (uint32_t item = 0; item < values_per_lane; ++item) {
            state_values[item] += k_values[item] * delta;
            query_projection += state_values[item] * q_values[item];
        }
        query_projection = prefill_warp_sum(query_projection);
        if (lane == 0U) {
            recurrent[static_cast<uint64_t>(token) * kLinearOutputElements
                + head * kLinearHeadDim + output_row] = query_projection
                    * (1.0f / sqrtf(static_cast<float>(kLinearHeadDim)));
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t item = 0; item < values_per_lane; ++item) {
        matrix_row[item * 32U + lane] = state_values[item];
    }
}

// Opt-in numerical-contract candidate.  The state ABI remains FP32, but all
// reductions and recurrent updates are accumulated in device double.  This
// is deliberately separate from the established kernel so the oracle and
// candidate can be compared without changing the default execution path.
__global__ __launch_bounds__(128, 2) void prefill_linear_precise_delta_chunk_kernel(
    const float* activated, const float* alpha_raw, const float* beta_raw,
    const float* dt_bias, const float* ssm_a, float* state,
    float* recurrent, uint32_t tokens, float epsilon) {
    constexpr uint32_t warps_per_block = 4U;
    constexpr uint32_t values_per_lane = kLinearHeadDim / 32U;
    const uint32_t head = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t output_row = blockIdx.y * warps_per_block + threadIdx.y;
    if (head >= kLinearValueHeads || output_row >= kLinearHeadDim) return;

    const uint32_t q_head = head % kLinearKeyHeads;
    float* matrix_row = state
        + (static_cast<uint64_t>(head) * kLinearHeadDim + output_row)
            * kLinearHeadDim;
    double state_values[values_per_lane];
    #pragma unroll
    for (uint32_t item = 0; item < values_per_lane; ++item) {
        state_values[item] = static_cast<double>(matrix_row[item * 32U + lane]);
    }

    for (uint32_t token = 0; token < tokens; ++token) {
        const float* token_activated = activated
            + static_cast<uint64_t>(token) * kLinearQkvElements;
        const float* q = token_activated + q_head * kLinearHeadDim;
        const float* k = token_activated
            + kLinearKeyHeads * kLinearHeadDim + q_head * kLinearHeadDim;
        const float* v = token_activated
            + 2U * kLinearKeyHeads * kLinearHeadDim + head * kLinearHeadDim;
        const uint64_t gate_index = static_cast<uint64_t>(token)
            * kLinearValueHeads + head;

        double q_sum = 0.0;
        double k_sum = 0.0;
        for (uint32_t col = 0; col < kLinearHeadDim; ++col) {
            const double q_value = static_cast<double>(q[col]);
            const double k_value = static_cast<double>(k[col]);
            q_sum = q_value * q_value + q_sum;
            k_sum = k_value * k_value + k_sum;
        }
        const double floor = static_cast<double>(epsilon) * static_cast<double>(epsilon);
        const double q_scale = 1.0 / sqrt(q_sum > floor ? q_sum : floor);
        const double k_scale = 1.0 / sqrt(k_sum > floor ? k_sum : floor);

        const double alpha = static_cast<double>(alpha_raw[gate_index])
            + static_cast<double>(dt_bias[head]);
        const double softplus = alpha > 20.0
            ? alpha
            : (alpha < -20.0 ? exp(alpha) : log1p(exp(alpha)));
        const double decay = exp(softplus * static_cast<double>(ssm_a[head]));
        const double beta_value = static_cast<double>(beta_raw[gate_index]) >= 0.0
            ? 1.0 / (1.0 + exp(-static_cast<double>(beta_raw[gate_index])))
            : exp(static_cast<double>(beta_raw[gate_index]))
                / (1.0 + exp(static_cast<double>(beta_raw[gate_index])));

        double projection = 0.0;
        double q_values[values_per_lane];
        double k_values[values_per_lane];
        #pragma unroll
        for (uint32_t item = 0; item < values_per_lane; ++item) {
            const uint32_t col = item * 32U + lane;
            q_values[item] = static_cast<double>(q[col]) * q_scale;
            k_values[item] = static_cast<double>(k[col]) * k_scale;
            state_values[item] *= decay;
            projection = state_values[item] * k_values[item] + projection;
        }
        projection = __shfl_sync(
            0xffffffffU, prefill_warp_sum_f64(projection), 0);
        const double delta = (static_cast<double>(v[output_row])
            - static_cast<double>(projection)) * beta_value;

        double query_projection = 0.0;
        #pragma unroll
        for (uint32_t item = 0; item < values_per_lane; ++item) {
            const uint32_t col = item * 32U + lane;
            state_values[item] += k_values[item] * delta;
            matrix_row[col] = static_cast<float>(state_values[item]);
            query_projection = state_values[item] * q_values[item] + query_projection;
        }
        query_projection = __shfl_sync(
            0xffffffffU, prefill_warp_sum_f64(query_projection), 0);
        if (lane == 0U) {
            recurrent[static_cast<uint64_t>(token) * kLinearOutputElements
                + head * kLinearHeadDim + output_row] = static_cast<float>(
                    query_projection / sqrt(static_cast<double>(kLinearHeadDim)));
        }
    }
}

// Candidate scalar preparation preserves the reference's serial FP32 sums.
// output is scratch here: the entire delta kernel finishes on the same stream
// before gated_norm overwrites it with the actual output. No state/cache alias.
template<bool CheckpointSource>
__global__ void prefill_linear_shared_scalars_kernel(
    const float* activated, const float* alpha_raw, const float* beta_raw,
    const float* dt_bias, const float* ssm_a, float* scalars,
    uint32_t tokens, float epsilon) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tokens * kLinearValueHeads) return;
    const uint32_t token = index / kLinearValueHeads;
    const uint32_t head = index % kLinearValueHeads;
    const uint32_t q_head = CheckpointSource
        ? head / (kLinearValueHeads / kLinearKeyHeads) : head % kLinearKeyHeads;
    const float* q = activated + static_cast<uint64_t>(token)
        * kLinearQkvElements + q_head * kLinearHeadDim;
    const float* k = q + kLinearKeyHeads * kLinearHeadDim;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint32_t col = 0; col < kLinearHeadDim; ++col) {
        q_sum += q[col] * q[col];
        k_sum += k[col] * k[col];
    }
    const float floor = epsilon * epsilon;
    scalars[4U * index] = rsqrtf(CheckpointSource ? q_sum + 1.0e-6f : fmaxf(q_sum, floor));
    scalars[4U * index + 1U] = rsqrtf(CheckpointSource ? k_sum + 1.0e-6f : fmaxf(k_sum, floor));
    scalars[4U * index + 2U] = expf(prefill_softplus(alpha_raw[index]
        + dt_bias[head]) * ssm_a[head]);
    scalars[4U * index + 3U] = prefill_sigmoid(beta_raw[index]);
}

template<bool CheckpointSource>
__global__ __launch_bounds__(128, 2) void prefill_linear_shared_delta_chunk_kernel(
    const float* activated, const float* scalars, float* state,
    float* recurrent, uint32_t tokens) {
    constexpr uint32_t warps_per_block = 4U;
    constexpr uint32_t values_per_lane = kLinearHeadDim / 32U;
    const uint32_t head = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t output_row = blockIdx.y * warps_per_block + threadIdx.y;
    if (head >= kLinearValueHeads || output_row >= kLinearHeadDim) return;
    const uint32_t q_head = CheckpointSource
        ? head / (kLinearValueHeads / kLinearKeyHeads) : head % kLinearKeyHeads;
    float* matrix_row = state
        + (static_cast<uint64_t>(head) * kLinearHeadDim + output_row)
            * kLinearHeadDim;
    float state_values[values_per_lane];
    #pragma unroll
    for (uint32_t item = 0; item < values_per_lane; ++item) {
        state_values[item] = matrix_row[item * 32U + lane];
    }
    for (uint32_t token = 0; token < tokens; ++token) {
        const float* token_activated = activated
            + static_cast<uint64_t>(token) * kLinearQkvElements;
        const float* q = token_activated + q_head * kLinearHeadDim;
        const float* k = token_activated
            + kLinearKeyHeads * kLinearHeadDim + q_head * kLinearHeadDim;
        const float* v = token_activated
            + 2U * kLinearKeyHeads * kLinearHeadDim + head * kLinearHeadDim;
        const float* coefficient = scalars
            + 4U * (static_cast<uint64_t>(token) * kLinearValueHeads + head);
        const float q_scale = coefficient[0];
        const float k_scale = coefficient[1];
        const float decay = coefficient[2];
        const float beta = coefficient[3];
        float projection = 0.0f;
        float q_values[values_per_lane];
        float k_values[values_per_lane];
        #pragma unroll
        for (uint32_t item = 0; item < values_per_lane; ++item) {
            const uint32_t col = item * 32U + lane;
            q_values[item] = q[col] * q_scale;
            k_values[item] = k[col] * k_scale;
            state_values[item] *= decay;
            projection += state_values[item] * k_values[item];
        }
        projection = __shfl_sync(
            0xffffffffU, prefill_warp_sum(projection), 0);
        const float delta = (v[output_row] - projection) * beta;
        float query_projection = 0.0f;
        #pragma unroll
        for (uint32_t item = 0; item < values_per_lane; ++item) {
            state_values[item] += k_values[item] * delta;
            query_projection += state_values[item] * q_values[item];
        }
        query_projection = prefill_warp_sum(query_projection);
        if (lane == 0U) {
            recurrent[static_cast<uint64_t>(token) * kLinearOutputElements
                + head * kLinearHeadDim + output_row] = query_projection
                    * (1.0f / sqrtf(static_cast<float>(kLinearHeadDim)));
        }
    }
    #pragma unroll
    for (uint32_t item = 0; item < values_per_lane; ++item) {
        matrix_row[item * 32U + lane] = state_values[item];
    }
}

static_assert(kLinearHeadDim == 128 && kLinearValueHeads == 48 && kLinearKeyHeads == 16 && kLinearQkvElements == 10240 && kLinearOutputElements == 6144, "Augur row-reuse GDN shape");
#include "augur_qwen35_augur_prefill_delta.cuh"

#include "augur_qwen35_augur_prefill_norm.cuh"

__global__ void prefill_linear_gated_norm_chunk_kernel(
    const float* recurrent, const float* norm_weight, const float* z,
    float* output, uint32_t tokens, float epsilon) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t component = threadIdx.x;
    if (token >= tokens || head >= kLinearValueHeads
        || component >= kLinearHeadDim) return;
    const uint64_t base = static_cast<uint64_t>(token)
        * kLinearOutputElements + head * kLinearHeadDim;
    __shared__ float scale;
    if (component == 0U) {
        float sum = 0.0f;
        for (uint32_t index = 0; index < kLinearHeadDim; ++index) {
            const float value = recurrent[base + index];
            sum += value * value;
        }
        scale = rsqrtf(sum / static_cast<float>(kLinearHeadDim) + epsilon);
    }
    __syncthreads();
    output[base + component] = recurrent[base + component] * scale
        * norm_weight[component] * prefill_silu(z[base + component]);
}

// State-only replay of FP32 projected records. This translation unit keeps
// --fmad=false, matching the canonical recurrent update rounding order.
__global__ __launch_bounds__(128,2) void prefill_linear_fold_state_kernel(const float* a,const float* ar,const float* br,
    const float* bias,const float* sa,float* s,unsigned rows,float eps){
    unsigned head=blockIdx.x,lane=threadIdx.x,row=blockIdx.y*4+threadIdx.y;
    if(head>=kLinearValueHeads||row>=kLinearHeadDim)return;
    __shared__ float scale,decay,beta;
    float* dst=s+(head*kLinearHeadDim+row)*kLinearHeadDim;
    float values[4];
    #pragma unroll
    for(unsigned i=0;i<4;++i)values[i]=dst[i*32+lane];
    for(unsigned t=0;t<rows;++t){
        const float* k=a+t*kLinearQkvElements+kLinearKeyHeads*kLinearHeadDim+(head%kLinearKeyHeads)*kLinearHeadDim;
        const float* v=a+t*kLinearQkvElements+2*kLinearKeyHeads*kLinearHeadDim+head*kLinearHeadDim;
        if(lane==0&&threadIdx.y==0){
            float sum=0;for(unsigned c=0;c<kLinearHeadDim;++c)sum+=k[c]*k[c];
            scale=rsqrtf(fmaxf(sum,eps*eps));
            decay=expf(prefill_softplus(ar[t*kLinearValueHeads+head]+bias[head])*sa[head]);
            beta=prefill_sigmoid(br[t*kLinearValueHeads+head]);
        }
        __syncthreads();
        float proj=0,keys[4];
        #pragma unroll
        for(unsigned i=0;i<4;++i){keys[i]=k[i*32+lane]*scale;values[i]*=decay;proj+=values[i]*keys[i];}
        proj=__shfl_sync(0xffffffff,prefill_warp_sum(proj),0);
        float delta=(v[row]-proj)*beta;
        #pragma unroll
        for(unsigned i=0;i<4;++i)values[i]+=keys[i]*delta;
        __syncthreads();
    }
    #pragma unroll
    for(unsigned i=0;i<4;++i)dst[i*32+lane]=values[i];
}

#include "augur_qwen35_replay_fold_shared.cuh"

// Only the 1..8-row recorded ReplaySSM prefix is admitted. Output is dead
// scratch during a state-only fold; the next layer may reuse it on this stream.
augur_cuda_result_v1 submit_linear_chunk_fold_shared(int32_t ordinal,
    augur_cuda_stream_t stream, const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args) {
    if (!linear_chunk_metadata_valid(ordinal, stream, args, false) || args->tokens > 8U) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    const float* qkv = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->qkv_f32));
    const float* weights = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->conv_weights_f32));
    float* history = reinterpret_cast<float*>(static_cast<uintptr_t>(args->conv_history_f32));
    float* activated = reinterpret_cast<float*>(static_cast<uintptr_t>(args->activated_f32));
    const float* alpha = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->alpha_raw_f32));
    const float* beta = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->beta_raw_f32));
    const float* bias = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->dt_bias_f32));
    const float* ssm_a = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->ssm_a_f32));
    float* scalars = reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32));
    float* state = reinterpret_cast<float*>(static_cast<uintptr_t>(args->state_f32));
    static_assert(kLinearHeadDim >= 3U, "fold scalar scratch fits bound output");
    if (args->tokens <= 3U) {
        prefill_linear_conv_fold_scalars_kernel<<<kLinearQkvElements / kLinearHeadDim,
            kLinearHeadDim, 0, stream->stream>>>(qkv, weights, history, activated,
                alpha, beta, bias, ssm_a, scalars, args->tokens, epsilon);
    } else {
        prefill_linear_conv_chunk_kernel<<<(kLinearQkvElements + kThreads - 1U) / kThreads,
            kThreads, 0, stream->stream>>>(qkv, weights, history, activated, args->tokens);
        prefill_linear_fold_scalars_kernel<<<(args->tokens * kLinearValueHeads + 127U) / 128U,
            128, 0, stream->stream>>>(activated, alpha, beta, bias, ssm_a, scalars,
                args->tokens, epsilon);
    }
    prefill_linear_shared_fold_state_kernel<<<dim3(kLinearValueHeads, kLinearHeadDim / 4U),
        dim3(32U, 4U), 0, stream->stream>>>(activated, scalars, state, args->tokens);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

augur_cuda_result_v1 submit_linear_checkpoint_fold(int32_t ordinal,
    augur_cuda_stream_t stream, const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args, uint64_t source_history, uint64_t history_bytes, uint64_t source_state, uint64_t state_bytes) {
    if (!linear_checkpoint_metadata_valid(ordinal, stream, args, source_history, history_bytes, source_state, state_bytes, false)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    const float* qkv = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->qkv_f32));
    const float* weights = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->conv_weights_f32));
    float* history = reinterpret_cast<float*>(static_cast<uintptr_t>(args->conv_history_f32));
    float* activated = reinterpret_cast<float*>(static_cast<uintptr_t>(args->activated_f32));
    const float* alpha = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->alpha_raw_f32));
    const float* beta = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->beta_raw_f32));
    const float* bias = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->dt_bias_f32));
    const float* ssm_a = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->ssm_a_f32));
    float* scalars = reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32));
    float* state = reinterpret_cast<float*>(static_cast<uintptr_t>(args->state_f32));
    static_assert(kLinearHeadDim >= 3U, "fold scalar scratch fits bound output");
    if (args->tokens <= 3U) {
        prefill_checkpoint_conv_scalars<<<kLinearQkvElements / kLinearHeadDim,
            kLinearHeadDim, 0, stream->stream>>>(qkv, weights, history, reinterpret_cast<const float*>(static_cast<uintptr_t>(source_history)), activated,
                alpha, beta, bias, ssm_a, scalars, args->tokens, epsilon);
    } else {
        prefill_checkpoint_conv<<<(kLinearQkvElements + kThreads - 1U) / kThreads,
            kThreads, 0, stream->stream>>>(qkv, weights, history, reinterpret_cast<const float*>(static_cast<uintptr_t>(source_history)), activated, args->tokens);
        prefill_linear_fold_scalars_kernel<<<(args->tokens * kLinearValueHeads + 127U) / 128U,
            128, 0, stream->stream>>>(activated, alpha, beta, bias, ssm_a, scalars,
                args->tokens, epsilon);
    }
    prefill_checkpoint_state<<<dim3(kLinearValueHeads, kLinearHeadDim / 4U),
        dim3(32U, 4U), 0, stream->stream>>>(activated, scalars, state, reinterpret_cast<const float*>(static_cast<uintptr_t>(source_state)), args->tokens);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

template<bool SharedScalars, bool StateOnly = false, bool CheckpointSource = false>
augur_cuda_result_v1 submit_linear_chunk(int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args) {
    if (!linear_chunk_metadata_valid(ordinal, stream, args, false)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    if (CheckpointSource && args->tokens >= 512U) {
        augur_prefill_conv_time_tiles<16><<<
            dim3((kLinearQkvElements + kThreads - 1U) / kThreads,
                 (args->tokens + 15U) / 16U), kThreads, 0, stream->stream>>>(
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->qkv_f32)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->conv_weights_f32)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->conv_history_f32)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->activated_f32)),
            args->tokens);
    } else {
    prefill_linear_conv_chunk_kernel<<<
        (kLinearQkvElements + kThreads - 1U) / kThreads,
        kThreads, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->qkv_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->conv_weights_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->conv_history_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->activated_f32)),
        args->tokens);
    }
    const dim3 delta_grid(kLinearValueHeads,
        (kLinearHeadDim + 3U) / 4U, 1U);
    const dim3 delta_block(32U, 4U, 1U);
    if constexpr (StateOnly) {
        prefill_linear_fold_state_kernel<<<delta_grid, delta_block, 0, stream->stream>>>(
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->activated_f32)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->alpha_raw_f32)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->beta_raw_f32)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->dt_bias_f32)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->ssm_a_f32)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->state_f32)),
            args->tokens, epsilon);
        const cudaError_t error = cudaGetLastError();
        return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, 0)
            : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    }
    if constexpr (SharedScalars) {
        static_assert(kLinearHeadDim >= 4U, "scalar scratch fits output extent");
        prefill_linear_shared_scalars_kernel<CheckpointSource><<<
            (args->tokens * kLinearValueHeads + kThreads - 1U) / kThreads,
            kThreads, 0, stream->stream>>>(
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->activated_f32)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->alpha_raw_f32)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->beta_raw_f32)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->dt_bias_f32)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->ssm_a_f32)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
            args->tokens, epsilon);
        if (CheckpointSource && args->tokens >= 512U) {
            augur_prefill_delta_rows<4,4><<<dim3(kLinearValueHeads,kLinearHeadDim/16U),dim3(32,4),0,stream->stream>>>(
                reinterpret_cast<const float*>(static_cast<uintptr_t>(args->activated_f32)),
                reinterpret_cast<const float*>(static_cast<uintptr_t>(args->output_f32)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(args->state_f32)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(args->recurrent_f32)),args->tokens);
        } else {
            prefill_linear_shared_delta_chunk_kernel<CheckpointSource><<<delta_grid, delta_block, 0,
                stream->stream>>>(
                reinterpret_cast<const float*>(static_cast<uintptr_t>(args->activated_f32)),
                reinterpret_cast<const float*>(static_cast<uintptr_t>(args->output_f32)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(args->state_f32)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(args->recurrent_f32)),
                args->tokens);
        }
    } else {
    prefill_linear_delta_chunk_kernel<<<delta_grid, delta_block, 0,
        stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->activated_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->alpha_raw_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->beta_raw_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->dt_bias_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->ssm_a_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->state_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->recurrent_f32)),
        args->tokens, epsilon);
    }
    // Only the screened checkpoint-source long prefill schedule changes.
    // Preserve the original sequential 128-element FP32 sum and scalar SiLU.
    if (CheckpointSource && args->tokens >= 512U && args->tokens <= 6400U) {
        augur_prefill_gated_norm_warps<4,true><<<
            args->tokens*kLinearValueHeads/4U,dim3(32,4),0,stream->stream>>>(
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->recurrent_f32)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->norm_weight_f32)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->z_f32)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
            args->tokens,epsilon);
    } else {
    const dim3 norm_grid(args->tokens, kLinearValueHeads, 1U);
    prefill_linear_gated_norm_chunk_kernel<<<norm_grid, kLinearHeadDim, 0,
        stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->recurrent_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->norm_weight_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->z_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        args->tokens, epsilon);
    }
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

augur_cuda_result_v1 submit_linear_chunk_precise(int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args) {
    if (!linear_chunk_metadata_valid(ordinal, stream, args, false)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    prefill_linear_conv_chunk_kernel<<<
        (kLinearQkvElements + kThreads - 1U) / kThreads,
        kThreads, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->qkv_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->conv_weights_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->conv_history_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->activated_f32)),
        args->tokens);
    const dim3 delta_grid(kLinearValueHeads,
        (kLinearHeadDim + 3U) / 4U, 1U);
    const dim3 delta_block(32U, 4U, 1U);
    prefill_linear_precise_delta_chunk_kernel<<<delta_grid, delta_block, 0,
        stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->activated_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->alpha_raw_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->beta_raw_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->dt_bias_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->ssm_a_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->state_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->recurrent_f32)),
        args->tokens, epsilon);
    const dim3 norm_grid(args->tokens, kLinearValueHeads, 1U);
    prefill_linear_gated_norm_chunk_kernel<<<norm_grid, kLinearHeadDim, 0,
        stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->recurrent_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->norm_weight_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->z_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        args->tokens, epsilon);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

bool full_attention_metadata_valid(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_full_attention_args_v1* args,
    bool inspect_device) {
    if (stream == nullptr || stream->stream == nullptr || args == nullptr
        || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version
            != AUGUR_CUDA_QWEN35_PREFILL_FULL_ATTENTION_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args) || args->reserved0 != 0
        || args->tokens == 0
        || args->tokens > AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1
        || args->capacity == 0 || args->capacity > kFullMaxCapacity
        || args->position_base > args->capacity
        || args->tokens > args->capacity - args->position_base
        || args->query_heads != kFullQueryHeads
        || args->kv_heads != kFullKvHeads
        || args->head_dim != kFullHeadDim
        || args->rotary_dim != kFullRotaryDim) {
        return false;
    }
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) return false;

    const uint64_t tokens = args->tokens;
    const uint64_t f32 = sizeof(float);
    const uint64_t expected[] = {
        tokens * kFullQueryElements * 2U * f32,
        tokens * kFullKvElements * f32,
        tokens * kFullKvElements * f32,
        static_cast<uint64_t>(kFullHeadDim) * f32,
        static_cast<uint64_t>(kFullHeadDim) * f32,
        tokens * (kFullRotaryDim / 2U) * f32,
        tokens * (kFullRotaryDim / 2U) * f32,
        static_cast<uint64_t>(args->capacity) * kFullKvElements * f32,
        static_cast<uint64_t>(args->capacity) * kFullKvElements * f32,
        tokens * kFullQueryElements * f32,
        tokens * kFullQueryElements * f32,
        tokens * kFullQueryElements * f32,
        tokens * kFullQueryElements * f32,
    };
    const Range ranges[] = {
        {args->q_full_f32, args->q_full_len_bytes, alignof(float)},
        {args->k_raw_f32, args->k_raw_len_bytes, alignof(float)},
        {args->value_f32, args->value_len_bytes, alignof(float)},
        {args->q_norm_weight_f32, args->q_norm_weight_len_bytes, alignof(float)},
        {args->k_norm_weight_f32, args->k_norm_weight_len_bytes, alignof(float)},
        {args->rope_cos_f32, args->rope_cos_len_bytes, alignof(float)},
        {args->rope_sin_f32, args->rope_sin_len_bytes, alignof(float)},
        {args->key_cache_f32, args->key_cache_len_bytes, alignof(float)},
        {args->value_cache_f32, args->value_cache_len_bytes, alignof(float)},
        {args->query_f32, args->query_len_bytes, alignof(float)},
        {args->gate_f32, args->gate_len_bytes, alignof(float)},
        {args->attention_f32, args->attention_len_bytes, alignof(float)},
        {args->gated_f32, args->gated_len_bytes, alignof(float)},
    };
    for (size_t index = 0; index < 13; ++index) {
        if (ranges[index].bytes != expected[index]) return false;
    }
    if (!all_disjoint(ranges, 13)) return false;
    if (!inspect_device) return true;
    if (!sm70_device(ordinal)) return false;
    for (const Range range : ranges) {
        if (!device_range(range, ordinal)) return false;
    }
    return true;
}

__global__ void prefill_full_prepare_qkv_kernel(
    const float* q_full, const float* k_raw, const float* value,
    const float* q_norm_weight, const float* k_norm_weight,
    const float* rope_cos, const float* rope_sin,
    float* key_cache, float* value_cache, float* query, float* gate,
    uint32_t tokens, uint32_t position_base, float epsilon) {
    const uint32_t combined_heads = kFullQueryHeads + kFullKvHeads;
    const uint32_t token = blockIdx.x / combined_heads;
    const uint32_t combined_head = blockIdx.x % combined_heads;
    const uint32_t component = threadIdx.x;
    if (token >= tokens || component >= kFullHeadDim) return;

    const bool is_query = combined_head < kFullQueryHeads;
    const uint32_t head = is_query ? combined_head
        : combined_head - kFullQueryHeads;
    const uint64_t q_stride = static_cast<uint64_t>(kFullQueryElements) * 2U;
    const uint64_t source_base = is_query
        ? static_cast<uint64_t>(token) * q_stride
            + static_cast<uint64_t>(head) * kFullHeadDim
        : static_cast<uint64_t>(token) * kFullKvElements
            + static_cast<uint64_t>(head) * kFullHeadDim;
    const float* source = is_query ? q_full : k_raw;
    const float* norm_weight = is_query ? q_norm_weight : k_norm_weight;

    __shared__ float partial[kFullHeadDim];
    __shared__ float normed[kFullHeadDim];
    const float input = source[source_base + component];
    partial[component] = input * input;
    __syncthreads();
    for (uint32_t stride = kFullHeadDim / 2U; stride != 0U; stride >>= 1U) {
        if (component < stride) partial[component] += partial[component + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(
        partial[0] / static_cast<float>(kFullHeadDim) + epsilon);
    normed[component] = input * scale * norm_weight[component];
    __syncthreads();

    float rotated = normed[component];
    if (component < kFullRotaryDim) {
#if defined(AUGUR_CUDA_QWEN35_DIAGNOSTIC_SPLIT_ROPE)
        const uint32_t half = kFullRotaryDim / 2U;
        const uint32_t pair = component % half;
        const bool upper = component >= half;
        const uint64_t rope_index = static_cast<uint64_t>(token) * half + pair;
        const float cosine = rope_cos[rope_index];
        const float sine = rope_sin[rope_index];
        const float mate = normed[upper ? component - half : component + half];
        rotated = upper
            ? mate * sine + normed[component] * cosine
            : normed[component] * cosine - mate * sine;
#else
        const uint32_t pair = component / 2U;
        const uint64_t rope_index = static_cast<uint64_t>(token)
            * (kFullRotaryDim / 2U) + pair;
        const float cosine = rope_cos[rope_index];
        const float sine = rope_sin[rope_index];
        const float mate = normed[component ^ 1U];
        rotated = (component & 1U)
            ? mate * sine + normed[component] * cosine
            : normed[component] * cosine - mate * sine;
#endif
    }

    if (is_query) {
        const uint64_t output_index = static_cast<uint64_t>(token)
            * kFullQueryElements + static_cast<uint64_t>(head) * kFullHeadDim
            + component;
        query[output_index] = rotated;
        gate[output_index] = q_full[static_cast<uint64_t>(token) * q_stride
            + kFullQueryElements + static_cast<uint64_t>(head) * kFullHeadDim
            + component];
    } else {
        const uint64_t cache_index =
            (static_cast<uint64_t>(position_base) + token) * kFullKvElements
            + static_cast<uint64_t>(head) * kFullHeadDim + component;
        key_cache[cache_index] = rotated;
        value_cache[cache_index] = value[
            static_cast<uint64_t>(token) * kFullKvElements
            + static_cast<uint64_t>(head) * kFullHeadDim + component];
    }
}

__global__ void prefill_full_causal_attention_kernel(
    const float* query, const float* gate, const float* key_cache,
    const float* value_cache, float* attention, float* gated,
    uint32_t tokens, uint32_t position_base) {
    const uint32_t token = blockIdx.x / kFullQueryHeads;
    const uint32_t query_head = blockIdx.x % kFullQueryHeads;
    const uint32_t component = threadIdx.x;
    if (token >= tokens || component >= kFullHeadDim) return;

    const uint32_t visible_positions = position_base + token + 1U;
    const uint32_t kv_head = query_head
        / (kFullQueryHeads / kFullKvHeads);
    const uint64_t query_base = static_cast<uint64_t>(token)
        * kFullQueryElements + static_cast<uint64_t>(query_head) * kFullHeadDim;
    __shared__ float weights[kFullMaxCapacity];
    __shared__ float reduction[kFullHeadDim];

    const uint32_t lane = component & 31U;
    const uint32_t warp = component >> 5U;
    for (uint32_t position = warp; position < visible_positions; position += 8U) {
        const uint64_t key_base = static_cast<uint64_t>(position)
            * kFullKvElements + static_cast<uint64_t>(kv_head) * kFullHeadDim;
        float score = 0.0f;
        for (uint32_t dimension = lane; dimension < kFullHeadDim; dimension += 32U) {
            score += query[query_base + dimension] * key_cache[key_base + dimension];
        }
        score = prefill_warp_sum(score);
        if (lane == 0U) {
            weights[position] = score * rsqrtf(static_cast<float>(kFullHeadDim));
        }
    }
    __syncthreads();

    float local_maximum = -INFINITY;
    for (uint32_t position = component; position < visible_positions;
         position += kFullHeadDim) {
        local_maximum = fmaxf(local_maximum, weights[position]);
    }
    reduction[component] = local_maximum;
    __syncthreads();
    for (uint32_t stride = kFullHeadDim / 2U; stride != 0U; stride >>= 1U) {
        if (component < stride) {
            reduction[component] = fmaxf(
                reduction[component], reduction[component + stride]);
        }
        __syncthreads();
    }
    const float maximum = reduction[0];
    float local_sum = 0.0f;
    for (uint32_t position = component; position < visible_positions;
         position += kFullHeadDim) {
        const float weight = expf(weights[position] - maximum);
        weights[position] = weight;
        local_sum += weight;
    }
    reduction[component] = local_sum;
    __syncthreads();
    for (uint32_t stride = kFullHeadDim / 2U; stride != 0U; stride >>= 1U) {
        if (component < stride) reduction[component] += reduction[component + stride];
        __syncthreads();
    }
    const float inverse = 1.0f / reduction[0];
    float output = 0.0f;
    for (uint32_t position = 0; position < visible_positions; ++position) {
        const uint64_t value_index = static_cast<uint64_t>(position)
            * kFullKvElements + static_cast<uint64_t>(kv_head) * kFullHeadDim
            + component;
        output += weights[position] * inverse * value_cache[value_index];
    }
    attention[query_base + component] = output;
    gated[query_base + component] = output
        * prefill_sigmoid(gate[query_base + component]);
}

augur_cuda_result_v1 submit_full_attention(int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_full_attention_args_v1* args) {
    if (!full_attention_metadata_valid(ordinal, stream, args, false)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    const uint32_t prepare_blocks = args->tokens
        * (kFullQueryHeads + kFullKvHeads);
    prefill_full_prepare_qkv_kernel<<<prepare_blocks, kFullHeadDim, 0,
        stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->q_full_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->k_raw_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->value_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->q_norm_weight_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->k_norm_weight_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->rope_cos_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->rope_sin_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->key_cache_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->value_cache_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->query_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->gate_f32)),
        args->tokens, args->position_base, epsilon);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    }
    const uint32_t attention_blocks = args->tokens * kFullQueryHeads;
    prefill_full_causal_attention_kernel<<<attention_blocks, kFullHeadDim, 0,
        stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->key_cache_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->value_cache_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->attention_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->gated_f32)),
        args->tokens, args->position_base);
    error = cudaGetLastError();
    return error == cudaSuccess
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

// Independent Nerkyor research profile: canonical B1 score/softmax arithmetic,
// with eight causal query rows scheduled together. Legacy prefill is unchanged.
__global__ void nerkyor_canonical_f32_kv_attention_kernel(
    const float* query, const float* gate, const float* key_cache,
    const float* value_cache, float* attention, float* gated,
    uint32_t tokens, uint32_t position_base) {
    const uint32_t token = blockIdx.x / kFullQueryHeads;
    const uint32_t head = blockIdx.x % kFullQueryHeads;
    const uint32_t component = threadIdx.x;
    if (token >= tokens || component >= kFullHeadDim) return;
    const uint32_t visible = position_base + token + 1U;
    const uint32_t kv_head = head / (kFullQueryHeads / kFullKvHeads);
    const uint64_t base = static_cast<uint64_t>(token) * kFullQueryElements
        + static_cast<uint64_t>(head) * kFullHeadDim;
    __shared__ float weights[kFullMaxCapacity];
    const uint32_t lane = component & 31U, warp = component >> 5U;
    for (uint32_t pos = warp; pos < visible; pos += 8U) {
        const uint64_t key = static_cast<uint64_t>(pos) * kFullKvElements
            + static_cast<uint64_t>(kv_head) * kFullHeadDim;
        double score = 0.0;
        for (uint32_t col = lane; col < kFullHeadDim; col += 32U)
            score += static_cast<double>(query[base + col]) * static_cast<double>(key_cache[key + col]);
        for (uint32_t offset = 16U; offset != 0U; offset >>= 1U)
            score += __shfl_down_sync(0xffffffffU, score, offset);
        if (lane == 0U) weights[pos] = static_cast<float>(score) * rsqrtf(static_cast<float>(kFullHeadDim));
    }
    __syncthreads();
    if (component == 0U) {
        float maximum = -INFINITY;
        for (uint32_t pos = 0; pos < visible; ++pos) maximum = fmaxf(maximum, weights[pos]);
        float denominator = 0.0f;
        for (uint32_t pos = 0; pos < visible; ++pos) {
            const float weight = expf(weights[pos] - maximum);
            weights[pos] = weight; denominator += weight;
        }
        const float inverse = 1.0f / denominator;
        for (uint32_t pos = 0; pos < visible; ++pos) weights[pos] *= inverse;
    }
    __syncthreads();
    float result = 0.0f;
    for (uint32_t pos = 0; pos < visible; ++pos) {
        const uint64_t value = static_cast<uint64_t>(pos) * kFullKvElements
            + static_cast<uint64_t>(kv_head) * kFullHeadDim;
        result += weights[pos] * value_cache[value + component];
    }
    attention[base + component] = result;
    gated[base + component] = result * prefill_sigmoid(gate[base + component]);
}

augur_cuda_result_v1 submit_nerkyor_canonical_full_attention(int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_full_attention_args_v1* args) {
    if (!full_attention_metadata_valid(ordinal, stream, args, false) || args->tokens > 8U) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    const uint32_t prepare_blocks = args->tokens
        * (kFullQueryHeads + kFullKvHeads);
    prefill_full_prepare_qkv_kernel<<<prepare_blocks, kFullHeadDim, 0,
        stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->q_full_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->k_raw_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->value_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->q_norm_weight_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->k_norm_weight_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->rope_cos_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->rope_sin_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->key_cache_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->value_cache_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->query_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->gate_f32)),
        args->tokens, args->position_base, epsilon);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    }
    const uint32_t attention_blocks = args->tokens * kFullQueryHeads;
    nerkyor_canonical_f32_kv_attention_kernel<<<attention_blocks, kFullHeadDim, 0,
        stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->key_cache_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->value_cache_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->attention_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->gated_f32)),
        args->tokens, args->position_base);
    error = cudaGetLastError();
    return error == cudaSuccess
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

#if defined(AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION)
__device__ __forceinline__ float prefill_kv8_warp_max(float value) {
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    return __shfl_sync(0xffffffffU, value, 0);
}

__device__ __forceinline__ float prefill_kv8_chunk_decode(
    const uint8_t* cache,
    uint32_t position,
    uint32_t kv_head,
    uint32_t component,
    uint32_t capacity) {
    const uint32_t block = component / kKv8BlockElements;
    const uint32_t lane = component % kKv8BlockElements;
    (void)capacity;
    const uint64_t block_base =
        static_cast<uint64_t>(position) * kKv8CacheRowBytes
        + static_cast<uint64_t>(kv_head) * kKv8HeadRowBytes
        + static_cast<uint64_t>(block) * kKv8BlockBytes;
    const uint16_t scale_bits =
        *reinterpret_cast<const uint16_t*>(cache + block_base);
    const int8_t code = static_cast<int8_t>(cache[block_base + 2U + lane]);
    return __half2float(__ushort_as_half(scale_bits)) * static_cast<float>(code);
}

/* Normalize/RoPE the projected chunk into query and temporary K/V rows. */
template<bool QueryOnly = false>
__global__ void prefill_kv8_chunk_prepare_qkv_kernel(
    const float* q_full,
    const float* k_raw,
    const float* value,
    const float* q_norm_weight,
    const float* k_norm_weight,
    const float* rope_cos,
    const float* rope_sin,
    float* key_work,
    float* value_work,
    float* query,
    float* gate,
    uint32_t tokens,
    float epsilon) {
    const uint32_t combined_heads = QueryOnly ? kFullQueryHeads : kFullQueryHeads + kFullKvHeads;
    const uint32_t token = blockIdx.x / combined_heads;
    const uint32_t combined_head = blockIdx.x % combined_heads;
    const uint32_t component = threadIdx.x;
    if (token >= tokens || component >= kFullHeadDim) return;

    const bool is_query = combined_head < kFullQueryHeads;
    const uint32_t head = is_query
        ? combined_head
        : combined_head - kFullQueryHeads;
    const uint64_t q_stride = static_cast<uint64_t>(kFullQueryElements) * 2U;
    const uint64_t source_base = is_query
        ? static_cast<uint64_t>(token) * q_stride
            + static_cast<uint64_t>(head) * kFullHeadDim
        : static_cast<uint64_t>(token) * kFullKvElements
            + static_cast<uint64_t>(head) * kFullHeadDim;
    const float* source = is_query ? q_full : k_raw;
    const float* norm_weight = is_query ? q_norm_weight : k_norm_weight;

    __shared__ float partial[kFullHeadDim];
    __shared__ float normed[kFullHeadDim];
    const float input = source[source_base + component];
    partial[component] = input * input;
    __syncthreads();
    for (uint32_t stride = kFullHeadDim / 2U; stride != 0U; stride >>= 1U) {
        if (component < stride) partial[component] += partial[component + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(
        partial[0] / static_cast<float>(kFullHeadDim) + epsilon);
    normed[component] = input * scale * norm_weight[component];
    __syncthreads();

    float rotated = normed[component];
    if (component < kFullRotaryDim) {
#if defined(AUGUR_CUDA_QWEN35_DIAGNOSTIC_SPLIT_ROPE)
        const uint32_t half = kFullRotaryDim / 2U;
        const uint32_t pair = component % half;
        const bool upper = component >= half;
        const uint64_t rope_index = static_cast<uint64_t>(token) * half + pair;
        const float cosine = rope_cos[rope_index];
        const float sine = rope_sin[rope_index];
        const float mate = normed[upper ? component - half : component + half];
        rotated = upper
            ? mate * sine + normed[component] * cosine
            : normed[component] * cosine - mate * sine;
#else
        const uint32_t pair = component / 2U;
        const uint64_t rope_index = static_cast<uint64_t>(token)
            * (kFullRotaryDim / 2U) + pair;
        const float cosine = rope_cos[rope_index];
        const float sine = rope_sin[rope_index];
        const float mate = normed[component ^ 1U];
        rotated = (component & 1U)
            ? mate * sine + normed[component] * cosine
            : normed[component] * cosine - mate * sine;
#endif
    }

    if (is_query) {
        const uint64_t output_index = static_cast<uint64_t>(token)
            * kFullQueryElements + static_cast<uint64_t>(head) * kFullHeadDim
            + component;
        query[output_index] = rotated;
        gate[output_index] = q_full[static_cast<uint64_t>(token) * q_stride
            + kFullQueryElements + static_cast<uint64_t>(head) * kFullHeadDim
            + component];
    } else {
        const uint64_t output_index = static_cast<uint64_t>(token)
            * kFullKvElements + static_cast<uint64_t>(head) * kFullHeadDim
            + component;
        key_work[output_index] = rotated;
        value_work[output_index] = value[output_index];
    }
}

/* Quantize only the newly produced chunk rows; historical rows are untouched. */
__global__ void prefill_kv8_chunk_store_kernel(
    const float* key_work,
    const float* value_work,
    uint8_t* key_cache_q8,
    uint8_t* value_cache_q8,
    uint32_t position_base,
    uint32_t tokens,
    uint32_t capacity) {
    const uint32_t row_head = blockIdx.x;
    const uint32_t token = row_head / kFullKvHeads;
    const uint32_t kv_head = row_head % kFullKvHeads;
    const uint32_t component = threadIdx.x;
    if (token >= tokens || component >= kFullHeadDim) return;

    const uint32_t position = position_base + token;
    const uint32_t block = component / kKv8BlockElements;
    const uint32_t lane = component % kKv8BlockElements;
    const uint64_t source = static_cast<uint64_t>(token) * kFullKvElements
        + static_cast<uint64_t>(kv_head) * kFullHeadDim + component;
    const float key = key_work[source];
    const float value = value_work[source];
    const float key_abs_max = prefill_kv8_warp_max(fabsf(key));
    const float value_abs_max = prefill_kv8_warp_max(fabsf(value));
    const float key_inverse = key_abs_max != 0.0f ? 127.0f / key_abs_max : 0.0f;
    const float value_inverse = value_abs_max != 0.0f ? 127.0f / value_abs_max : 0.0f;
    const uint16_t key_scale =
        __half_as_ushort(__float2half_rn(key_abs_max / 127.0f));
    const uint16_t value_scale =
        __half_as_ushort(__float2half_rn(value_abs_max / 127.0f));
    const int8_t key_code = static_cast<int8_t>(max(
        -128, min(127, __float2int_rn(key * key_inverse))));
    const int8_t value_code = static_cast<int8_t>(max(
        -128, min(127, __float2int_rn(value * value_inverse))));
    (void)capacity;
    const uint64_t destination = static_cast<uint64_t>(position) * kKv8CacheRowBytes
        + static_cast<uint64_t>(kv_head) * kKv8HeadRowBytes
        + static_cast<uint64_t>(block) * kKv8BlockBytes;
    if (lane == 0U) {
        *reinterpret_cast<uint16_t*>(key_cache_q8 + destination) = key_scale;
        *reinterpret_cast<uint16_t*>(value_cache_q8 + destination) = value_scale;
    }
    key_cache_q8[destination + 2U + lane] = static_cast<uint8_t>(key_code);
    value_cache_q8[destination + 2U + lane] = static_cast<uint8_t>(value_code);
}

__global__ void prefill_kv8_chunk_gate_kernel(
    const float* attention,
    const float* gate,
    float* gated,
    uint64_t elements) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index < elements) {
        gated[index] = attention[index] * prefill_sigmoid(gate[index]);
    }
}

/*
 * Fixed-shared-memory causal attention over the packed KV8 cache. One CTA
 * owns a token/KV-head pair, so all six GQA query heads reuse each decoded
 * K/V tile. The per-head online-softmax update order is unchanged.
 */
template<bool SharedSoftmax, uint32_t TilePositions = kFullTilePositions, bool UnitPreviousScale = false, bool RegisterSoftmaxState = false, bool ParallelScales = false>
__global__ void prefill_kv8_chunk_attention_kernel(
    const float* query,
    const float* gate,
    const uint8_t* key_cache_q8,
    const uint8_t* value_cache_q8,
    float* attention,
    float* gated,
    uint32_t tokens,
    uint32_t position_base,
    uint32_t capacity) {
    constexpr uint32_t kFullTilePositions = TilePositions;
    static_assert(TilePositions == 8U || (SharedSoftmax && TilePositions == 16U),
        "bounded SIMT tile specializations only");
    static_assert(!UnitPreviousScale || (SharedSoftmax && TilePositions == 8U),
        "unit-scale candidate requires shared tile8");
    static_assert(!RegisterSoftmaxState || (SharedSoftmax && TilePositions == 8U && !UnitPreviousScale),
        "register-state candidate changes storage only on shared tile8");
    static_assert(!ParallelScales || (SharedSoftmax && TilePositions == 8U && !UnitPreviousScale),
        "parallel-scales candidate uses shared tile8 and exact expf");
    constexpr uint32_t queries_per_kv = kFullQueryHeads / kFullKvHeads;
    const uint32_t token = blockIdx.x / kFullKvHeads;
    const uint32_t kv_head = blockIdx.x % kFullKvHeads;
    const uint32_t component = threadIdx.x;
    if (token >= tokens || component >= kFullHeadDim) return;

    const uint32_t position = position_base + token;
    const uint32_t visible_positions = position + 1U;
    __shared__ float key_tile[kFullTilePositions * kFullHeadDim];
    __shared__ float value_tile[kFullTilePositions * kFullHeadDim];
    __shared__ float scores[queries_per_kv * kFullTilePositions];
    // The candidate computes each head's scalar recurrence once, preserving
    // position order. Every output component consumes the same FP32 scales.
    // The false specialization is the unchanged independent control.
    __shared__ float previous_scales[SharedSoftmax ? queries_per_kv * kFullTilePositions : 1];
    __shared__ float current_scales[SharedSoftmax ? queries_per_kv * kFullTilePositions : 1];
    __shared__ float denominators[SharedSoftmax ? queries_per_kv : 1];
    __shared__ float tile_seed_maxima[ParallelScales ? queries_per_kv : 1];

    const uint32_t warp = component >> 5U;
    const uint32_t lane = component & 31U;
    const bool active_query = warp < queries_per_kv;
    const uint32_t warp_query_head = kv_head * queries_per_kv + warp;
    float query_fragment[8] = {};
    if (active_query) {
        const uint64_t query_base = static_cast<uint64_t>(token) * kFullQueryElements
            + static_cast<uint64_t>(warp_query_head) * kFullHeadDim;
        #pragma unroll
        for (uint32_t item = 0; item < 8U; ++item) {
            query_fragment[item] = query[query_base + lane + item * 32U];
        }
    }

    float outputs[queries_per_kv] = {};
    // In the shared specialization each of the first six threads owns exactly
    // one head's scalar recurrence. The opt-in candidate indexes one scalar,
    // avoiding the dynamically indexed six-element thread-local stack arrays.
    // QK reduction, expf/FMA expressions, key order and output updates are unchanged.
    constexpr uint32_t softmax_states = (RegisterSoftmaxState || ParallelScales) ? 1U : queries_per_kv;
    float running_maxima[softmax_states];
    float running_sums[softmax_states] = {};
    #pragma unroll
    for (uint32_t query_index = 0; query_index < softmax_states; ++query_index) {
        running_maxima[query_index] = -INFINITY;
    }
    if constexpr (ParallelScales) {
        if (component < queries_per_kv) tile_seed_maxima[component] = -INFINITY;
    }

    for (uint32_t tile_begin = 0; tile_begin < visible_positions;
         tile_begin += kFullTilePositions) {
        const uint32_t remaining = visible_positions - tile_begin;
        const uint32_t tile_positions = remaining < kFullTilePositions
            ? remaining
            : kFullTilePositions;
        for (uint32_t tile_position = 0;
             tile_position < kFullTilePositions; ++tile_position) {
            const uint32_t shared_index = tile_position * kFullHeadDim + component;
            if (tile_position < tile_positions) {
                const uint32_t absolute_position = tile_begin + tile_position;
                key_tile[shared_index] = prefill_kv8_chunk_decode(
                    key_cache_q8, absolute_position, kv_head, component, capacity);
                value_tile[shared_index] = prefill_kv8_chunk_decode(
                    value_cache_q8, absolute_position, kv_head, component, capacity);
            }
        }
        __syncthreads();

        if (active_query) {
            for (uint32_t tile_position = 0;
                 tile_position < tile_positions; ++tile_position) {
                const uint32_t shared_base = tile_position * kFullHeadDim;
                double score = 0.0;
                #pragma unroll
                for (uint32_t item = 0; item < 8U; ++item) {
                    score += static_cast<double>(query_fragment[item])
                        * static_cast<double>(
                            key_tile[shared_base + lane + item * 32U]);
                }
                #pragma unroll
                for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
                    score += __shfl_down_sync(0xffffffffU, score, offset);
                }
                if (lane == 0U) {
                    scores[warp * kFullTilePositions + tile_position] =
                        static_cast<float>(score)
                        * rsqrtf(static_cast<float>(kFullHeadDim));
                }
            }
        }
        __syncthreads();
        if constexpr (ParallelScales) {
            // Two complete warps, eight adjacent lanes per head. Prefix max
            // is exact for finite scores; unlike an approximate block-softmax,
            // the denominator and each output still fold in increasing key order.
            if (component < 64U) {
                const uint32_t head = component / 8U;
                const uint32_t pos = component % 8U;
                const bool valid = head < queries_per_kv && pos < tile_positions;
                const float seed = head < queries_per_kv ? tile_seed_maxima[head] : -INFINITY;
                const float scaled = valid ? scores[head * 8U + pos] : -INFINITY;
                float maximum = fmaxf(seed, scaled);
                #pragma unroll
                for (uint32_t offset = 1U; offset < 8U; offset <<= 1U) {
                    const float preceding = __shfl_up_sync(0xffffffffU, maximum, offset, 8);
                    if (pos >= offset) maximum = fmaxf(preceding, maximum);
                }
                const float preceding = __shfl_up_sync(0xffffffffU, maximum, 1, 8);
                const float old_maximum = pos == 0U ? seed : preceding;
                if (valid) {
                    previous_scales[head * 8U + pos] = expf(old_maximum - maximum);
                    current_scales[head * 8U + pos] = expf(scaled - maximum);
                    if (pos + 1U == tile_positions) tile_seed_maxima[head] = maximum;
                }
            }
            __syncthreads();
            if (component < queries_per_kv) {
                for (uint32_t pos = 0; pos < tile_positions; ++pos) {
                    const uint32_t index = component * 8U + pos;
                    running_sums[0] = running_sums[0] * previous_scales[index] + current_scales[index];
                }
                denominators[component] = running_sums[0];
            }
            #pragma unroll
            for (uint32_t head = 0; head < queries_per_kv; ++head) {
                for (uint32_t pos = 0; pos < tile_positions; ++pos) {
                    const uint32_t index = head * 8U + pos;
                    outputs[head] = outputs[head] * previous_scales[index]
                        + current_scales[index] * value_tile[pos * kFullHeadDim + component];
                }
            }
        } else if constexpr (SharedSoftmax) {
            if (component < queries_per_kv) {
                const uint32_t query_index = component;
                const uint32_t state_index = RegisterSoftmaxState ? 0U : query_index;
                for (uint32_t tile_position = 0;
                     tile_position < tile_positions; ++tile_position) {
                    const uint32_t index = query_index * kFullTilePositions + tile_position;
                    const float next_maximum = fmaxf(running_maxima[state_index], scores[index]);
                    const float previous_scale = running_sums[state_index] == 0.0f
                        ? 0.0f : (UnitPreviousScale
                            && isfinite(running_maxima[state_index])
                            && running_maxima[state_index] == next_maximum)
                            ? 1.0f : expf(running_maxima[state_index] - next_maximum);
                    const float current_scale = expf(scores[index] - next_maximum);
                    previous_scales[index] = previous_scale;
                    current_scales[index] = current_scale;
                    running_sums[state_index] = running_sums[state_index] * previous_scale
                        + current_scale;
                    running_maxima[state_index] = next_maximum;
                }
                denominators[query_index] = running_sums[state_index];
            }
            __syncthreads();
            #pragma unroll
            for (uint32_t query_index = 0; query_index < queries_per_kv; ++query_index) {
                for (uint32_t tile_position = 0;
                     tile_position < tile_positions; ++tile_position) {
                    const uint32_t index = query_index * kFullTilePositions + tile_position;
                    outputs[query_index] = outputs[query_index] * previous_scales[index]
                        + current_scales[index]
                            * value_tile[tile_position * kFullHeadDim + component];
                }
            }
        } else {
          #pragma unroll
          for (uint32_t query_index = 0; query_index < queries_per_kv; ++query_index) {
            for (uint32_t tile_position = 0;
                 tile_position < tile_positions; ++tile_position) {
                const float scaled_score =
                    scores[query_index * kFullTilePositions + tile_position];
                const float next_maximum =
                    fmaxf(running_maxima[query_index], scaled_score);
                const float previous_scale = running_sums[query_index] == 0.0f
                    ? 0.0f
                    : expf(running_maxima[query_index] - next_maximum);
                const float current_scale = expf(scaled_score - next_maximum);
                running_sums[query_index] =
                    running_sums[query_index] * previous_scale + current_scale;
                outputs[query_index] = outputs[query_index] * previous_scale
                    + current_scale
                        * value_tile[tile_position * kFullHeadDim + component];
                running_maxima[query_index] = next_maximum;
            }
          }
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t query_index = 0; query_index < queries_per_kv; ++query_index) {
        const uint32_t query_head = kv_head * queries_per_kv + query_index;
        const uint64_t output_index = static_cast<uint64_t>(token) * kFullQueryElements
            + static_cast<uint64_t>(query_head) * kFullHeadDim + component;
        const float denominator = SharedSoftmax ? denominators[query_index] : running_sums[query_index];
        const float result = denominator > 0.0f
            ? outputs[query_index] / denominator
            : 0.0f;
        attention[output_index] = result;
        gated[output_index] = result * prefill_sigmoid(gate[output_index]);
    }
}

// Explicit component candidate: each warp owns one query head and all eight
// components per lane. Preserve the control's exact FP64 dot/reduction tree
// and increasing-position FP32 softmax/output recurrence. The direct variant
// repeats KV decode across heads without block communication; SharedKv retains
// tile8 cooperative decode and two barriers, but no cross-warp scalar exchange.
// Both are explicitly selected experiments, not default request paths.
template<bool SharedKv>
__global__ void prefill_kv8_chunk_warp_ordered_kernel(
    const float* query, const float* gate,
    const uint8_t* key_cache_q8, const uint8_t* value_cache_q8,
    float* attention, float* gated,
    uint32_t tokens, uint32_t position_base, uint32_t capacity) {
    const uint32_t head_index = blockIdx.x * (blockDim.x / 32U) + threadIdx.x / 32U;
    const uint32_t token = SharedKv ? blockIdx.x / kFullKvHeads : head_index / kFullQueryHeads;
    if (token >= tokens) return; // whole warp exits together
    const uint32_t warp = threadIdx.x / 32U;
    const bool active = !SharedKv || warp < kFullQueryHeads / kFullKvHeads;
    const uint32_t head = SharedKv
        ? (blockIdx.x % kFullKvHeads) * (kFullQueryHeads / kFullKvHeads) + warp
        : head_index % kFullQueryHeads;
    const uint32_t kv_head = head / (kFullQueryHeads / kFullKvHeads);
    const uint32_t lane = threadIdx.x & 31U;
    const uint64_t base = static_cast<uint64_t>(token) * kFullQueryElements
        + static_cast<uint64_t>(head) * kFullHeadDim;
    float q[8];
    float outputs[8] = {};
    #pragma unroll
    for (uint32_t item = 0; item < 8U; ++item) {
        q[item] = active ? query[base + lane + item * 32U] : 0.0f;
    }
    float running_maximum = -INFINITY;
    float running_sum = 0.0f;
    const uint32_t visible_positions = position_base + token + 1U;
    constexpr uint32_t tile_width = SharedKv ? 8U : 1U;
    __shared__ float key_tile[SharedKv ? 8U * kFullHeadDim : 1U];
    __shared__ float value_tile[SharedKv ? 8U * kFullHeadDim : 1U];
    for (uint32_t tile = 0; tile < visible_positions; tile += tile_width) {
      const uint32_t count = min(tile_width, visible_positions - tile);
      if constexpr (SharedKv) {
        const uint32_t shared_kv_head = blockIdx.x % kFullKvHeads;
        for (uint32_t pos = 0; pos < count; ++pos) {
            key_tile[pos * kFullHeadDim + threadIdx.x] = prefill_kv8_chunk_decode(
                key_cache_q8, tile + pos, shared_kv_head, threadIdx.x, capacity);
            value_tile[pos * kFullHeadDim + threadIdx.x] = prefill_kv8_chunk_decode(
                value_cache_q8, tile + pos, shared_kv_head, threadIdx.x, capacity);
        }
        __syncthreads();
      }
      if (active) {
       for (uint32_t tile_pos = 0; tile_pos < count; ++tile_pos) {
        const uint32_t position = tile + tile_pos;
        double score = 0.0;
        #pragma unroll
        for (uint32_t item = 0; item < 8U; ++item) {
            const float key = SharedKv ? key_tile[tile_pos * kFullHeadDim + lane + item * 32U]
                : prefill_kv8_chunk_decode(
                    key_cache_q8, position, kv_head, lane + item * 32U, capacity);
            score += static_cast<double>(q[item]) * static_cast<double>(key);
        }
        #pragma unroll
        for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
            score += __shfl_down_sync(0xffffffffU, score, offset);
        }
        float previous_scale = 0.0f;
        float current_scale = 0.0f;
        if (lane == 0U) {
            const float scaled_score = static_cast<float>(score)
                * rsqrtf(static_cast<float>(kFullHeadDim));
            const float next_maximum = fmaxf(running_maximum, scaled_score);
            previous_scale = running_sum == 0.0f
                ? 0.0f : expf(running_maximum - next_maximum);
            current_scale = expf(scaled_score - next_maximum);
            running_sum = running_sum * previous_scale + current_scale;
            running_maximum = next_maximum;
        }
        previous_scale = __shfl_sync(0xffffffffU, previous_scale, 0);
        current_scale = __shfl_sync(0xffffffffU, current_scale, 0);
        #pragma unroll
        for (uint32_t item = 0; item < 8U; ++item) {
            const float value = SharedKv ? value_tile[tile_pos * kFullHeadDim + lane + item * 32U]
                : prefill_kv8_chunk_decode(
                    value_cache_q8, position, kv_head, lane + item * 32U, capacity);
            outputs[item] = outputs[item] * previous_scale + current_scale * value;
        }
       }
      }
      if constexpr (SharedKv) __syncthreads();
    }
    if (!active) return; // all block barriers have completed
    const float denominator = __shfl_sync(0xffffffffU, running_sum, 0);
    #pragma unroll
    for (uint32_t item = 0; item < 8U; ++item) {
        const uint64_t index = base + lane + item * 32U;
        const float result = denominator > 0.0f ? outputs[item] / denominator : 0.0f;
        attention[index] = result;
        gated[index] = result * prefill_sigmoid(gate[index]);
    }
}

#if defined(AUGUR_CUDA_QWEN35_KV8_TOKEN_PAIR)
/*
 * Research-only adjacent-token candidate. A 512-thread CTA owns two
 * consecutive tokens and one KV head. The key/value tile is decoded once and
 * consumed by both token halves; each half keeps an independent ordered
 * online-softmax recurrence, so causal visibility and output layout remain
 * identical to the one-token control.
 */
template<bool UnitPreviousScale = false>
__global__ void prefill_kv8_chunk_token_pair_kernel(
    const float* query,
    const float* gate,
    const uint8_t* key_cache_q8,
    const uint8_t* value_cache_q8,
    float* attention,
    float* gated,
    uint32_t tokens,
    uint32_t position_base,
    uint32_t capacity) {
    constexpr uint32_t kPairTokens = 2U;
    constexpr uint32_t kTilePositions = 8U;
    constexpr uint32_t queries_per_kv = kFullQueryHeads / kFullKvHeads;
    const uint32_t pair = blockIdx.x / kFullKvHeads;
    const uint32_t kv_head = blockIdx.x % kFullKvHeads;
    const uint32_t slot = threadIdx.x / kFullHeadDim;
    const uint32_t component = threadIdx.x % kFullHeadDim;
    if (slot >= kPairTokens || component >= kFullHeadDim) return;
    const uint32_t token = pair * kPairTokens + slot;
    const bool token_valid = token < tokens;
    const uint32_t position = position_base + token;
    const uint32_t visible_positions = position + 1U;
    const uint32_t pair_tokens = tokens - pair * kPairTokens < kPairTokens
        ? tokens - pair * kPairTokens
        : kPairTokens;
    const uint32_t pair_visible_positions =
        position_base + pair * kPairTokens + pair_tokens;

    __shared__ float key_tile[kTilePositions * kFullHeadDim];
    __shared__ float value_tile[kTilePositions * kFullHeadDim];
    __shared__ float scores[kPairTokens * queries_per_kv * kTilePositions];
    __shared__ float previous_scales[
        kPairTokens * queries_per_kv * kTilePositions];
    __shared__ float current_scales[
        kPairTokens * queries_per_kv * kTilePositions];
    __shared__ float denominators[kPairTokens * queries_per_kv];

    const uint32_t warp = component >> 5U;
    const uint32_t lane = component & 31U;
    const bool active_query = token_valid && warp < queries_per_kv;
    const uint32_t warp_query_head = kv_head * queries_per_kv + warp;
    float query_fragment[8] = {};
    if (active_query) {
        const uint64_t query_base = static_cast<uint64_t>(token) * kFullQueryElements
            + static_cast<uint64_t>(warp_query_head) * kFullHeadDim;
        #pragma unroll
        for (uint32_t item = 0; item < 8U; ++item) {
            query_fragment[item] = query[query_base + lane + item * 32U];
        }
    }

    float outputs[queries_per_kv] = {};
    float running_maxima[queries_per_kv];
    float running_sums[queries_per_kv] = {};
    #pragma unroll
    for (uint32_t query_index = 0; query_index < queries_per_kv; ++query_index) {
        running_maxima[query_index] = -INFINITY;
    }

    for (uint32_t tile_begin = 0; tile_begin < pair_visible_positions;
         tile_begin += kTilePositions) {
        const uint32_t remaining = pair_visible_positions - tile_begin;
        const uint32_t tile_positions = remaining < kTilePositions
            ? remaining
            : kTilePositions;
        if (slot == 0U) {
            for (uint32_t tile_position = 0; tile_position < kTilePositions;
                 ++tile_position) {
                const uint32_t shared_index = tile_position * kFullHeadDim + component;
                if (tile_position < tile_positions) {
                    const uint32_t absolute_position = tile_begin + tile_position;
                    key_tile[shared_index] = prefill_kv8_chunk_decode(
                        key_cache_q8, absolute_position, kv_head, component, capacity);
                    value_tile[shared_index] = prefill_kv8_chunk_decode(
                        value_cache_q8, absolute_position, kv_head, component, capacity);
                }
            }
        }
        __syncthreads();

        if (active_query) {
            for (uint32_t tile_position = 0; tile_position < tile_positions;
                 ++tile_position) {
                const uint32_t absolute_position = tile_begin + tile_position;
                double score = 0.0;
                if (absolute_position < visible_positions) {
                    const uint32_t shared_base = tile_position * kFullHeadDim;
                    #pragma unroll
                    for (uint32_t item = 0; item < 8U; ++item) {
                        score += static_cast<double>(query_fragment[item])
                            * static_cast<double>(key_tile[shared_base + lane + item * 32U]);
                    }
                    #pragma unroll
                    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
                        score += __shfl_down_sync(0xffffffffU, score, offset);
                    }
                } else {
                    score = -INFINITY;
                }
                if (lane == 0U) {
                    const uint32_t index =
                        (slot * queries_per_kv + warp) * kTilePositions + tile_position;
                    scores[index] = static_cast<float>(score)
                        * rsqrtf(static_cast<float>(kFullHeadDim));
                }
            }
        }
        __syncthreads();

        if (component < queries_per_kv && token_valid) {
            const uint32_t query_index = component;
            for (uint32_t tile_position = 0; tile_position < tile_positions;
                 ++tile_position) {
                const uint32_t absolute_position = tile_begin + tile_position;
                if (absolute_position >= visible_positions) continue;
                const uint32_t index =
                    (slot * queries_per_kv + query_index) * kTilePositions + tile_position;
                const float next_maximum = fmaxf(running_maxima[query_index], scores[index]);
                const float previous_scale = running_sums[query_index] == 0.0f
                    ? 0.0f
                    : (UnitPreviousScale
                        && isfinite(running_maxima[query_index])
                        && running_maxima[query_index] == next_maximum)
                        ? 1.0f
                        : expf(running_maxima[query_index] - next_maximum);
                const float current_scale = expf(scores[index] - next_maximum);
                previous_scales[index] = previous_scale;
                current_scales[index] = current_scale;
                running_sums[query_index] = running_sums[query_index] * previous_scale
                    + current_scale;
                running_maxima[query_index] = next_maximum;
            }
            denominators[slot * queries_per_kv + query_index] = running_sums[query_index];
        }
        __syncthreads();

        if (token_valid) {
            #pragma unroll
            for (uint32_t query_index = 0; query_index < queries_per_kv; ++query_index) {
                for (uint32_t tile_position = 0; tile_position < tile_positions;
                     ++tile_position) {
                    const uint32_t absolute_position = tile_begin + tile_position;
                    if (absolute_position >= visible_positions) continue;
                    const uint32_t index =
                        (slot * queries_per_kv + query_index) * kTilePositions + tile_position;
                    outputs[query_index] = outputs[query_index] * previous_scales[index]
                        + current_scales[index]
                            * value_tile[tile_position * kFullHeadDim + component];
                }
            }
        }
        __syncthreads();
    }

    if (!token_valid) return;
    #pragma unroll
    for (uint32_t query_index = 0; query_index < queries_per_kv; ++query_index) {
        const uint32_t query_head = kv_head * queries_per_kv + query_index;
        const uint64_t output_index = static_cast<uint64_t>(token) * kFullQueryElements
            + static_cast<uint64_t>(query_head) * kFullHeadDim + component;
        const float denominator = denominators[slot * queries_per_kv + query_index];
        const float result = denominator > 0.0f ? outputs[query_index] / denominator : 0.0f;
        attention[output_index] = result;
        gated[output_index] = result * prefill_sigmoid(gate[output_index]);
    }
}

augur_cuda_result_v1 submit_prefill_kv8_chunk_prepare(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args);

augur_cuda_result_v1 submit_prefill_kv8_chunk_token_pair(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args) {
    const augur_cuda_result_v1 prepared = submit_prefill_kv8_chunk_prepare(
        ordinal, stream, args);
    if (prepared.status != AUGUR_CUDA_STATUS_OK) return prepared;
    const uint32_t pairs = (args->tokens + 1U) / 2U;
    prefill_kv8_chunk_token_pair_kernel<true><<<
        pairs * kFullKvHeads, 2U * kFullHeadDim, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate_f32)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->key_cache_q8_0)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->value_cache_q8_0)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->attention_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->gated_f32)),
        args->tokens, args->position_base, args->capacity);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}
#endif

bool prefill_kv8_chunk_attention_metadata_valid(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    bool inspect_device) {
    if (stream == nullptr || stream->stream == nullptr || args == nullptr
        || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version
            != AUGUR_CUDA_QWEN35_PREFILL_KV8_CHUNK_ATTENTION_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args) || args->reserved0 != 0
        || args->tokens == 0
        || args->tokens > AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1
        || args->capacity == 0 || args->capacity > kKv8MaxCapacity
        || args->position_base > args->capacity
        || args->tokens > args->capacity - args->position_base
        || args->query_heads != kFullQueryHeads
        || args->kv_heads != kFullKvHeads
        || args->head_dim != kFullHeadDim
        || args->rotary_dim != kFullRotaryDim) {
        return false;
    }
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) return false;

    const uint64_t tokens = args->tokens;
    const uint64_t f32 = sizeof(float);
    const uint64_t expected[] = {
        tokens * kFullQueryElements * 2U * f32,
        tokens * kFullKvElements * f32,
        tokens * kFullKvElements * f32,
        static_cast<uint64_t>(kFullHeadDim) * f32,
        static_cast<uint64_t>(kFullHeadDim) * f32,
        tokens * (kFullRotaryDim / 2U) * f32,
        tokens * (kFullRotaryDim / 2U) * f32,
        tokens * kFullKvElements * f32,
        tokens * kFullKvElements * f32,
        static_cast<uint64_t>(args->capacity) * kKv8CacheRowBytes,
        static_cast<uint64_t>(args->capacity) * kKv8CacheRowBytes,
        tokens * kFullQueryElements * f32,
        tokens * kFullQueryElements * f32,
        tokens * kFullQueryElements * f32,
        tokens * kFullQueryElements * f32,
    };
    const uint64_t addresses[] = {
        args->q_full_f32, args->k_raw_f32, args->value_f32,
        args->q_norm_weight_f32, args->k_norm_weight_f32,
        args->rope_cos_f32, args->rope_sin_f32,
        args->key_work_f32, args->value_work_f32,
        args->key_cache_q8_0, args->value_cache_q8_0,
        args->query_f32, args->gate_f32, args->attention_f32, args->gated_f32,
    };
    const uint64_t lengths[] = {
        args->q_full_len_bytes, args->k_raw_len_bytes, args->value_len_bytes,
        args->q_norm_weight_len_bytes, args->k_norm_weight_len_bytes,
        args->rope_cos_len_bytes, args->rope_sin_len_bytes,
        args->key_work_len_bytes, args->value_work_len_bytes,
        args->key_cache_len_bytes, args->value_cache_len_bytes,
        args->query_len_bytes, args->gate_len_bytes,
        args->attention_len_bytes, args->gated_len_bytes,
    };
    constexpr size_t count = sizeof(expected) / sizeof(expected[0]);
    Range ranges[15];
    for (size_t index = 0; index < count; ++index) {
        ranges[index] = {
            addresses[index], lengths[index],
            index == 9U || index == 10U ? alignof(uint16_t) : alignof(float)};
        if (lengths[index] != expected[index]) return false;
    }
    if (!all_disjoint(ranges, count)) return false;
    if (!inspect_device) return true;
    if (!sm70_device(ordinal)) return false;
    for (const Range range : ranges) {
        if (!device_range(range, ordinal)) return false;
    }
    return true;
}

augur_cuda_result_v1 submit_prefill_kv8_chunk_prepare(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args) {
    if (!prefill_kv8_chunk_attention_metadata_valid(
            ordinal, stream, args, false)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    const uint32_t prepare_blocks = args->tokens
        * (kFullQueryHeads + kFullKvHeads);
    prefill_kv8_chunk_prepare_qkv_kernel<false><<<
        prepare_blocks, kFullHeadDim, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->q_full_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->k_raw_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->value_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->q_norm_weight_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->k_norm_weight_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->rope_cos_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->rope_sin_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->key_work_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->value_work_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->query_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->gate_f32)),
        args->tokens, epsilon);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    }
    prefill_kv8_chunk_store_kernel<<<
        args->tokens * kFullKvHeads, kFullHeadDim, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->key_work_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->value_work_f32)),
        reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(args->key_cache_q8_0)),
        reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(args->value_cache_q8_0)),
        args->position_base, args->tokens, args->capacity);
    error = cudaGetLastError();
    return error == cudaSuccess
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

augur_cuda_result_v1 submit_prefill_kv8_chunk_gate(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args) {
    if (!prefill_kv8_chunk_attention_metadata_valid(
            ordinal, stream, args, false)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    const uint64_t elements = static_cast<uint64_t>(args->tokens)
        * kFullQueryElements;
    prefill_kv8_chunk_gate_kernel<<<
        launch_blocks(elements), kThreads, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->attention_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->gated_f32)),
        elements);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

template<bool SharedSoftmax = false, uint32_t TilePositions = kFullTilePositions, bool UnitPreviousScale = false, bool RegisterSoftmaxState = false, bool WarpOrdered = false, bool WarpSharedKv = false, bool ParallelScales = false>
augur_cuda_result_v1 submit_prefill_kv8_chunk_attention(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args) {
    const augur_cuda_result_v1 prepared = submit_prefill_kv8_chunk_prepare(
        ordinal, stream, args);
    if (prepared.status != AUGUR_CUDA_STATUS_OK) return prepared;
    if constexpr (WarpOrdered || WarpSharedKv) {
        const uint32_t blocks = WarpSharedKv ? args->tokens * kFullKvHeads
            : (args->tokens * kFullQueryHeads + 3U) / 4U;
        prefill_kv8_chunk_warp_ordered_kernel<WarpSharedKv><<<
            blocks, WarpSharedKv ? 256U : 128U, 0, stream->stream>>>(
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query_f32)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate_f32)),
            reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->key_cache_q8_0)),
            reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->value_cache_q8_0)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->attention_f32)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->gated_f32)),
            args->tokens, args->position_base, args->capacity);
    } else {
      prefill_kv8_chunk_attention_kernel<SharedSoftmax, TilePositions, UnitPreviousScale, RegisterSoftmaxState, ParallelScales><<<
        args->tokens * kFullKvHeads, kFullHeadDim, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate_f32)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->key_cache_q8_0)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->value_cache_q8_0)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->attention_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->gated_f32)),
        args->tokens, args->position_base, args->capacity);
    }
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

// Bounded two-stage candidate. At most16 query rows have materialized scores;
// every row/head keeps the retained32-lane FP64 dot reduction. No TC/FP16 dot,
// sparse attention, precision change or extra KV representation is introduced.
__global__ void prefill_kv8_staged_scores_kernel(
    const float* query, const uint8_t* key_cache, float* scores,
    uint32_t token_begin, uint32_t position_base, uint32_t capacity) {
    const uint32_t head = blockIdx.y;
    const uint32_t row = blockIdx.z;
    const uint32_t token = token_begin + row;
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5U;
    const uint32_t kv_head = head / (kFullQueryHeads / kFullKvHeads);
    const uint32_t visible = position_base + token + 1U;
    const uint64_t query_base = static_cast<uint64_t>(token) * kFullQueryElements
        + static_cast<uint64_t>(head) * kFullHeadDim;
    const uint64_t score_base = (static_cast<uint64_t>(row) * kFullQueryHeads + head) * capacity;
    float q[8];
    #pragma unroll
    for (uint32_t item = 0; item < 8U; ++item) q[item] = query[query_base + lane + item * 32U];
    for (uint32_t local = warp; local < 32U; local += 8U) {
        const uint32_t position = blockIdx.x * 32U + local;
        if (position >= visible) continue; // whole warp
        double score = 0.0;
        #pragma unroll
        for (uint32_t item = 0; item < 8U; ++item) {
            const float key = prefill_kv8_chunk_decode(
                key_cache, position, kv_head, lane + item * 32U, capacity);
            score += static_cast<double>(q[item]) * static_cast<double>(key);
        }
        #pragma unroll
        for (uint32_t offset = 16U; offset != 0U; offset >>= 1U)
            score += __shfl_down_sync(0xffffffffU, score, offset);
        if (lane == 0U) scores[score_base + position] = static_cast<float>(score)
            * rsqrtf(static_cast<float>(kFullHeadDim));
    }
}

__global__ void prefill_kv8_staged_consume_kernel(
    const float* scores, const uint8_t* value_cache, const float* gate,
    float* attention, float* gated,
    uint32_t token_begin, uint32_t position_base, uint32_t capacity) {
    constexpr uint32_t tile_width = 32U;
    const uint32_t head = blockIdx.x % kFullQueryHeads;
    const uint32_t row = blockIdx.x / kFullQueryHeads;
    const uint32_t token = token_begin + row;
    const uint32_t component = threadIdx.x;
    const uint32_t kv_head = head / (kFullQueryHeads / kFullKvHeads);
    const uint32_t visible = position_base + token + 1U;
    const uint64_t score_base = (static_cast<uint64_t>(row) * kFullQueryHeads + head) * capacity;
    __shared__ float values[tile_width * kFullHeadDim];
    __shared__ float previous[tile_width];
    __shared__ float current[tile_width];
    __shared__ float seed_maximum;
    __shared__ float denominator;
    float output = 0.0f;
    float running_sum = 0.0f;
    if (component == 0U) seed_maximum = -INFINITY;
    __syncthreads();
    for (uint32_t begin = 0; begin < visible; begin += tile_width) {
        const uint32_t count = min(tile_width, visible - begin);
        for (uint32_t pos = 0; pos < count; ++pos)
            values[pos * kFullHeadDim + component] = prefill_kv8_chunk_decode(
                value_cache, begin + pos, kv_head, component, capacity);
        if (component < 32U) {
            const float seed = seed_maximum;
            const float score = component < count ? scores[score_base + begin + component] : -INFINITY;
            float maximum = fmaxf(seed, score);
            #pragma unroll
            for (uint32_t offset = 1U; offset < 32U; offset <<= 1U) {
                const float preceding = __shfl_up_sync(0xffffffffU, maximum, offset);
                if (component >= offset) maximum = fmaxf(preceding, maximum);
            }
            const float preceding = __shfl_up_sync(0xffffffffU, maximum, 1);
            const float old_maximum = component == 0U ? seed : preceding;
            if (component < count) {
                previous[component] = expf(old_maximum - maximum);
                current[component] = expf(score - maximum);
                if (component + 1U == count) seed_maximum = maximum;
            }
        }
        __syncthreads();
        if (component == 0U) {
            for (uint32_t pos = 0; pos < count; ++pos)
                running_sum = running_sum * previous[pos] + current[pos];
            denominator = running_sum;
        }
        for (uint32_t pos = 0; pos < count; ++pos)
            output = output * previous[pos] + current[pos] * values[pos * kFullHeadDim + component];
        __syncthreads();
    }
    const uint64_t index = static_cast<uint64_t>(token) * kFullQueryElements
        + static_cast<uint64_t>(head) * kFullHeadDim + component;
    const float result = denominator > 0.0f ? output / denominator : 0.0f;
    attention[index] = result;
    gated[index] = result * prefill_sigmoid(gate[index]);
}

// Two-dimensional staging keeps enough query rows in flight without a full
// context score matrix. One KV decode feeds all six GQA query heads.
template<bool CacheQuery>
__global__ void prefill_kv8_tiled_scores_kernel(
    const float* query, const uint8_t* key_cache, float* scores,
    uint32_t token_begin, uint32_t position_base, uint32_t capacity,
    uint32_t key_begin, uint32_t key_count, uint32_t score_stride) {
    constexpr uint32_t queries_per_kv = kFullQueryHeads / kFullKvHeads;
    const uint32_t kv_head = blockIdx.y;
    const uint32_t row = blockIdx.z;
    const uint32_t token = token_begin + row;
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5U;
    const uint32_t visible = position_base + token + 1U;
    __shared__ double query_tile[CacheQuery ? queries_per_kv * kFullHeadDim : 1U];
    if constexpr (CacheQuery) {
        #pragma unroll
        for (uint32_t head = 0; head < queries_per_kv; ++head) {
            const uint64_t index = static_cast<uint64_t>(token) * kFullQueryElements
                + static_cast<uint64_t>(kv_head * queries_per_kv + head) * kFullHeadDim + threadIdx.x;
            query_tile[head * kFullHeadDim + threadIdx.x] = static_cast<double>(query[index]);
        }
        __syncthreads();
    }
    for (uint32_t local = warp; local < 32U; local += 8U) {
        const uint32_t key_offset = blockIdx.x * 32U + local;
        const uint32_t position = key_begin + key_offset;
        if (key_offset >= key_count || position >= visible) continue;
        double key[8];
        #pragma unroll
        for (uint32_t item = 0; item < 8U; ++item)
            key[item] = static_cast<double>(prefill_kv8_chunk_decode(
                key_cache, position, kv_head, lane + item * 32U, capacity));
        for (uint32_t head_local = 0; head_local < queries_per_kv; ++head_local) {
            const uint32_t head = kv_head * queries_per_kv + head_local;
            const uint64_t query_base = static_cast<uint64_t>(token) * kFullQueryElements
                + static_cast<uint64_t>(head) * kFullHeadDim;
            double score = 0.0;
            #pragma unroll
            for (uint32_t item = 0; item < 8U; ++item)
                score += (CacheQuery ? query_tile[head_local * kFullHeadDim + lane + item * 32U]
                    : static_cast<double>(query[query_base + lane + item * 32U])) * key[item];
            #pragma unroll
            for (uint32_t offset = 16U; offset != 0U; offset >>= 1U)
                score += __shfl_down_sync(0xffffffffU, score, offset);
            if (lane == 0U) {
                const uint64_t index = (static_cast<uint64_t>(row) * kFullQueryHeads + head)
                    * score_stride + key_offset;
                scores[index] = static_cast<float>(score) * rsqrtf(static_cast<float>(kFullHeadDim));
            }
        }
    }
}

__global__ void prefill_kv8_tiled_consume_kernel(
    const float* scores, float* state, const uint8_t* value_cache, const float* gate,
    float* attention, float* gated,
    uint32_t token_begin, uint32_t position_base, uint32_t capacity,
    uint32_t key_begin, uint32_t key_count, uint32_t score_stride) {
    constexpr uint32_t heads = kFullQueryHeads / kFullKvHeads;
    constexpr uint32_t tile_width = 16U;
    const uint32_t row = blockIdx.x / kFullKvHeads;
    const uint32_t kv_head = blockIdx.x % kFullKvHeads;
    const uint32_t token = token_begin + row;
    const uint32_t component = threadIdx.x;
    const uint32_t visible = position_base + token + 1U;
    if (key_begin >= visible) return; // entire CTA, before any barrier
    const uint32_t end = min(visible, key_begin + key_count);
    const uint64_t state_base = (static_cast<uint64_t>(row) * kFullQueryHeads
        + kv_head * heads) * 2U;
    __shared__ float values[tile_width * kFullHeadDim];
    __shared__ float previous[heads * tile_width];
    __shared__ float current[heads * tile_width];
    __shared__ float maxima[heads];
    __shared__ float denominators[heads];
    float running_sum = 0.0f;
    if (component < heads) {
        maxima[component] = key_begin == 0U ? -INFINITY : state[state_base + component * 2U];
        running_sum = key_begin == 0U ? 0.0f : state[state_base + component * 2U + 1U];
        denominators[component] = running_sum;
    }
    float outputs[heads];
    #pragma unroll
    for (uint32_t head = 0; head < heads; ++head) {
        const uint64_t index = static_cast<uint64_t>(token) * kFullQueryElements
            + static_cast<uint64_t>(kv_head * heads + head) * kFullHeadDim + component;
        outputs[head] = key_begin == 0U ? 0.0f : attention[index];
    }
    __syncthreads();
    for (uint32_t begin = key_begin; begin < end; begin += tile_width) {
        const uint32_t count = min(tile_width, end - begin);
        for (uint32_t pos = 0; pos < count; ++pos)
            values[pos * kFullHeadDim + component] = prefill_kv8_chunk_decode(
                value_cache, begin + pos, kv_head, component, capacity);
        if (component < heads * tile_width) {
            const uint32_t head = component / tile_width;
            const uint32_t pos = component % tile_width;
            const uint64_t score_base = (static_cast<uint64_t>(row) * kFullQueryHeads
                + kv_head * heads + head) * score_stride;
            const float seed = maxima[head];
            const float score = pos < count ? scores[score_base + begin - key_begin + pos] : -INFINITY;
            float maximum = fmaxf(seed, score);
            #pragma unroll
            for (uint32_t offset = 1U; offset < tile_width; offset <<= 1U) {
                const float preceding = __shfl_up_sync(0xffffffffU, maximum, offset, 16);
                if (pos >= offset) maximum = fmaxf(preceding, maximum);
            }
            const float preceding = __shfl_up_sync(0xffffffffU, maximum, 1, 16);
            const float old_maximum = pos == 0U ? seed : preceding;
            if (pos < count) {
                previous[head * tile_width + pos] = expf(old_maximum - maximum);
                current[head * tile_width + pos] = expf(score - maximum);
                if (pos + 1U == count) maxima[head] = maximum;
            }
        }
        __syncthreads();
        if (component < heads) {
            for (uint32_t pos = 0; pos < count; ++pos) {
                const uint32_t index = component * tile_width + pos;
                running_sum = running_sum * previous[index] + current[index];
            }
            denominators[component] = running_sum;
        }
        #pragma unroll
        for (uint32_t head = 0; head < heads; ++head) {
            for (uint32_t pos = 0; pos < count; ++pos) {
                const uint32_t index = head * tile_width + pos;
                outputs[head] = outputs[head] * previous[index]
                    + current[index] * values[pos * kFullHeadDim + component];
            }
        }
        __syncthreads();
    }
    if (component < heads) {
        state[state_base + component * 2U] = maxima[component];
        state[state_base + component * 2U + 1U] = running_sum;
    }
    const bool final_tile = end == visible;
    #pragma unroll
    for (uint32_t head = 0; head < heads; ++head) {
        const uint64_t index = static_cast<uint64_t>(token) * kFullQueryElements
            + static_cast<uint64_t>(kv_head * heads + head) * kFullHeadDim + component;
        const float result = final_tile
            ? (denominators[head] > 0.0f ? outputs[head] / denominators[head] : 0.0f)
            : outputs[head];
        attention[index] = result;
        if (final_tile) gated[index] = result * prefill_sigmoid(gate[index]);
    }
}

uint64_t dgemm_score_workspace_bytes(uint32_t tokens, uint32_t capacity) {
    const uint64_t rows = std::min(tokens, 128U);
    const uint64_t stride = std::min(capacity, 8192U);
    return rows * kFullQueryHeads * (stride + 2U) * sizeof(float)
        + (rows * 6U * kFullHeadDim + stride * kFullHeadDim + rows * 6U * stride) * sizeof(double);
}

bool staged_scores_metadata_valid(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes, bool inspect_device, bool tiled = false, bool dgemm = false) {
    if (!prefill_kv8_chunk_attention_metadata_valid(ordinal, stream, args, inspect_device)) return false;
    const uint64_t rows = std::min(args->tokens, tiled ? 128U : 16U);
    const uint64_t stride = tiled ? std::min(args->capacity, 8192U) : args->capacity;
    const Range score_range{score_address, score_bytes, dgemm ? alignof(double) : alignof(float)};
    const uint64_t required = dgemm ? dgemm_score_workspace_bytes(args->tokens, args->capacity)
        : rows * kFullQueryHeads * (stride + (tiled ? 2U : 0U)) * sizeof(float);
    if (score_bytes != required
        || !range_valid(score_range)) return false;
    const Range inputs[] = {
        {args->q_full_f32, args->q_full_len_bytes, 4},
        {args->k_raw_f32, args->k_raw_len_bytes, 4},
        {args->value_f32, args->value_len_bytes, 4},
        {args->q_norm_weight_f32, args->q_norm_weight_len_bytes, 4},
        {args->k_norm_weight_f32, args->k_norm_weight_len_bytes, 4},
        {args->rope_cos_f32, args->rope_cos_len_bytes, 4},
        {args->rope_sin_f32, args->rope_sin_len_bytes, 4},
        {args->key_work_f32, args->key_work_len_bytes, 4},
        {args->value_work_f32, args->value_work_len_bytes, 4},
        {args->key_cache_q8_0, args->key_cache_len_bytes, 2},
        {args->value_cache_q8_0, args->value_cache_len_bytes, 2},
        {args->query_f32, args->query_len_bytes, 4},
        {args->gate_f32, args->gate_len_bytes, 4},
        {args->attention_f32, args->attention_len_bytes, 4},
        {args->gated_f32, args->gated_len_bytes, 4},
    };
    for (const Range input : inputs) if (ranges_overlap(input, score_range)) return false;
    return !inspect_device || device_range(score_range, ordinal);
}

augur_cuda_result_v1 submit_staged_scores(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes) {
    if (!staged_scores_metadata_valid(ordinal, stream, args, score_address, score_bytes, false))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const auto prepared = submit_prefill_kv8_chunk_prepare(ordinal, stream, args);
    if (prepared.status != AUGUR_CUDA_STATUS_OK) return prepared;
    auto* scores = reinterpret_cast<float*>(static_cast<uintptr_t>(score_address));
    for (uint32_t begin = 0; begin < args->tokens; begin += 16U) {
        const uint32_t rows = std::min(16U, args->tokens - begin);
        const uint32_t positions = args->position_base + begin + rows;
        const dim3 grid((positions + 31U) / 32U, kFullQueryHeads, rows);
        prefill_kv8_staged_scores_kernel<<<grid, 256U, 0, stream->stream>>>(
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query_f32)),
            reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->key_cache_q8_0)),
            scores, begin, args->position_base, args->capacity);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
        prefill_kv8_staged_consume_kernel<<<rows * kFullQueryHeads, 256U, 0, stream->stream>>>(
            scores,
            reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->value_cache_q8_0)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate_f32)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->attention_f32)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->gated_f32)),
            begin, args->position_base, args->capacity);
        error = cudaGetLastError();
        if (error != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    }
    return result(AUGUR_CUDA_STATUS_OK, 0);
}

template<bool CacheQuery = false, bool Prepared = false>
augur_cuda_result_v1 submit_tiled_scores(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes) {
    if (!staged_scores_metadata_valid(ordinal, stream, args, score_address, score_bytes, false, true))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    if (!Prepared) {
        const auto prepared = submit_prefill_kv8_chunk_prepare(ordinal, stream, args);
        if (prepared.status != AUGUR_CUDA_STATUS_OK) return prepared;
    }
    auto* scores = reinterpret_cast<float*>(static_cast<uintptr_t>(score_address));
    const uint32_t row_capacity = std::min(128U, args->tokens);
    const uint32_t stride = std::min(8192U, args->capacity);
    float* state = scores + static_cast<uint64_t>(row_capacity) * kFullQueryHeads * stride;
    for (uint32_t begin = 0; begin < args->tokens; begin += row_capacity) {
        const uint32_t rows = std::min(row_capacity, args->tokens - begin);
        const uint32_t positions = args->position_base + begin + rows;
        for (uint32_t key_begin = 0; key_begin < positions; key_begin += stride) {
            const uint32_t key_count = std::min(stride, positions - key_begin);
            const dim3 grid((key_count + 31U) / 32U, kFullKvHeads, rows);
            prefill_kv8_tiled_scores_kernel<CacheQuery><<<grid, 256U, 0, stream->stream>>>(
                reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query_f32)),
                reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->key_cache_q8_0)),
                scores, begin, args->position_base, args->capacity, key_begin, key_count, stride);
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
            prefill_kv8_tiled_consume_kernel<<<rows * kFullKvHeads, 256U, 0, stream->stream>>>(
                scores, state,
                reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->value_cache_q8_0)),
                reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate_f32)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(args->attention_f32)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(args->gated_f32)),
                begin, args->position_base, args->capacity, key_begin, key_count, stride);
            error = cudaGetLastError();
            if (error != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
        }
    }
    return result(AUGUR_CUDA_STATUS_OK, 0);
}

__global__ void prefill_kv8_dgemm_pack_kernel(
    const float* query, const uint8_t* key_cache, double* q, double* k,
    uint32_t token_begin, uint32_t rows, uint32_t kv_head,
    uint32_t key_begin, uint32_t key_count, uint32_t capacity) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * 6U * kFullHeadDim) {
        const uint32_t row = index / (6U * kFullHeadDim);
        const uint32_t within = index % (6U * kFullHeadDim);
        q[index] = static_cast<double>(query[static_cast<uint64_t>(token_begin + row) * kFullQueryElements
            + kv_head * 6U * kFullHeadDim + within]);
    }
    if (index < key_count * kFullHeadDim) {
        k[index] = static_cast<double>(prefill_kv8_chunk_decode(
            key_cache, key_begin + index / kFullHeadDim, kv_head, index % kFullHeadDim, capacity));
    }
}

__global__ void prefill_kv8_dgemm_store_kernel(
    const double* products, float* scores, uint32_t rows, uint32_t kv_head,
    uint32_t key_count, uint32_t stride) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<uint64_t>(rows) * 6U * key_count) return;
    const uint32_t column = index / key_count;
    const uint32_t pos = index % key_count;
    const uint32_t row = column / 6U;
    const uint32_t head = kv_head * 6U + column % 6U;
    scores[(static_cast<uint64_t>(row) * kFullQueryHeads + head) * stride + pos] =
        static_cast<float>(products[static_cast<uint64_t>(column) * stride + pos])
        * rsqrtf(static_cast<float>(kFullHeadDim));
}

bool dgemm_scores_metadata_valid(int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes, bool inspect_device) {
    if (plan == nullptr || plan->handle == nullptr || plan->owner != stream || plan->ordinal != ordinal
        || !staged_scores_metadata_valid(ordinal, stream, args, score_address, score_bytes, inspect_device, true, true)) return false;
    const Range scratch{score_address, score_bytes, alignof(double)};
    const Range weight{plan->snapshot.decoded_weight_f16, plan->snapshot.decoded_weight_len_bytes, 16};
    const Range input{plan->snapshot.decoded_input_f16, plan->snapshot.decoded_input_len_bytes, 16};
    return !ranges_overlap(scratch, weight) && !ranges_overlap(scratch, input);
}

augur_cuda_result_v1 submit_tiled_dgemm(int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes) {
    if (!dgemm_scores_metadata_valid(ordinal, stream, plan, args, score_address, score_bytes, false))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const auto prepared = submit_prefill_kv8_chunk_prepare(ordinal, stream, args);
    if (prepared.status != AUGUR_CUDA_STATUS_OK) return prepared;
    auto* scores = reinterpret_cast<float*>(static_cast<uintptr_t>(score_address));
    const uint32_t row_capacity = std::min(128U, args->tokens);
    const uint32_t stride = std::min(8192U, args->capacity);
    float* state = scores + static_cast<uint64_t>(row_capacity) * kFullQueryHeads * stride;
    auto* q = reinterpret_cast<double*>(state + static_cast<uint64_t>(row_capacity) * kFullQueryHeads * 2U);
    double* k = q + static_cast<uint64_t>(row_capacity) * 6U * kFullHeadDim;
    double* products = k + static_cast<uint64_t>(stride) * kFullHeadDim;
    const double alpha = 1.0, beta = 0.0;
    for (uint32_t begin = 0; begin < args->tokens; begin += row_capacity) {
        const uint32_t rows = std::min(row_capacity, args->tokens - begin);
        const uint32_t positions = args->position_base + begin + rows;
        for (uint32_t key_begin = 0; key_begin < positions; key_begin += stride) {
            const uint32_t key_count = std::min(stride, positions - key_begin);
            for (uint32_t kv_head = 0; kv_head < kFullKvHeads; ++kv_head) {
                prefill_kv8_dgemm_pack_kernel<<<(std::max(rows * 6U, key_count) * kFullHeadDim + 255U) / 256U, 256U, 0, stream->stream>>>(
                    reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query_f32)),
                    reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->key_cache_q8_0)),
                    q, k, begin, rows, kv_head, key_begin, key_count, args->capacity);
                cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
                // Full FP64 GEMM. This changes the dot summation tree and thus
                // requires the numerical oracle; no FP16/TF32/TC approximation.
                const cublasStatus_t status = cublasDgemm(plan->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    key_count, rows * 6U, kFullHeadDim, &alpha, k, kFullHeadDim,
                    q, kFullHeadDim, &beta, products, stride);
                if (status != CUBLAS_STATUS_SUCCESS) return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, status);
                const uint64_t elements = static_cast<uint64_t>(rows) * 6U * key_count;
                prefill_kv8_dgemm_store_kernel<<<(elements + 255U) / 256U, 256U, 0, stream->stream>>>(
                    products, scores, rows, kv_head, key_count, stride);
                error = cudaGetLastError();
                if (error != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
            }
            prefill_kv8_tiled_consume_kernel<<<rows * kFullKvHeads, 256U, 0, stream->stream>>>(
                scores, state,
                reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->value_cache_q8_0)),
                reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate_f32)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(args->attention_f32)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(args->gated_f32)),
                begin, args->position_base, args->capacity, key_begin, key_count, stride);
            const cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
        }
    }
    return result(AUGUR_CUDA_STATUS_OK, 0);
}

#include "augur_qwen35_canonical_prefill_scores.cuh"

__global__ void prefill_kv8_cache_quantize_kernel(
    const float* key_cache_f32,
    const float* value_cache_f32,
    uint8_t* key_cache_q8,
    uint8_t* value_cache_q8,
    uint32_t position_base,
    uint32_t tokens,
    uint32_t capacity) {
    const uint32_t row_head = blockIdx.x;
    const uint32_t token = row_head / kFullKvHeads;
    const uint32_t kv_head = row_head % kFullKvHeads;
    const uint32_t component = threadIdx.x;
    if (token >= tokens || component >= kFullHeadDim) return;

    const uint32_t position = position_base + token;
    const uint32_t block = component / kKv8BlockElements;
    const uint32_t lane = component % kKv8BlockElements;
    const uint64_t source = static_cast<uint64_t>(position) * kFullKvElements
        + static_cast<uint64_t>(kv_head) * kFullHeadDim + component;
    const float key = key_cache_f32[source];
    const float value = value_cache_f32[source];
    const float key_abs_max = prefill_kv8_warp_max(fabsf(key));
    const float value_abs_max = prefill_kv8_warp_max(fabsf(value));
    const float key_inverse = key_abs_max != 0.0f ? 127.0f / key_abs_max : 0.0f;
    const float value_inverse = value_abs_max != 0.0f ? 127.0f / value_abs_max : 0.0f;
    const uint16_t key_scale = __half_as_ushort(__float2half_rn(key_abs_max / 127.0f));
    const uint16_t value_scale = __half_as_ushort(__float2half_rn(value_abs_max / 127.0f));
    const int8_t key_code = static_cast<int8_t>(max(
        -128, min(127, __float2int_rn(key * key_inverse))));
    const int8_t value_code = static_cast<int8_t>(max(
        -128, min(127, __float2int_rn(value * value_inverse))));
    (void)capacity;
    const uint64_t destination = static_cast<uint64_t>(position) * kKv8CacheRowBytes
        + static_cast<uint64_t>(kv_head) * kKv8HeadRowBytes
        + static_cast<uint64_t>(block) * kKv8BlockBytes;
    if (lane == 0U) {
        *reinterpret_cast<uint16_t*>(key_cache_q8 + destination) = key_scale;
        *reinterpret_cast<uint16_t*>(value_cache_q8 + destination) = value_scale;
    }
    key_cache_q8[destination + 2U + lane] = static_cast<uint8_t>(key_code);
    value_cache_q8[destination + 2U + lane] = static_cast<uint8_t>(value_code);
}

bool prefill_kv8_cache_quantize_metadata_valid(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_cache_quantize_args_v1* args,
    bool inspect_device) {
    if (stream == nullptr || stream->stream == nullptr || args == nullptr
        || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version
            != AUGUR_CUDA_QWEN35_PREFILL_KV8_CACHE_QUANTIZE_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args) || args->reserved != 0
        || args->capacity == 0 || args->capacity > kFullMaxCapacity
        || args->tokens == 0
        || args->position_base > args->capacity
        || args->tokens > args->capacity - args->position_base
        || args->kv_heads != kFullKvHeads || args->head_dim != kFullHeadDim) {
        return false;
    }
    const uint64_t f32_bytes = static_cast<uint64_t>(args->capacity)
        * kFullKvElements * sizeof(float);
    const uint64_t q8_bytes = static_cast<uint64_t>(args->capacity)
        * kKv8CacheRowBytes;
    const Range ranges[] = {
        {args->key_cache_f32, args->key_cache_f32_len_bytes, alignof(float)},
        {args->value_cache_f32, args->value_cache_f32_len_bytes, alignof(float)},
        {args->key_cache_q8_0, args->key_cache_q8_0_len_bytes, alignof(uint16_t)},
        {args->value_cache_q8_0, args->value_cache_q8_0_len_bytes, alignof(uint16_t)},
    };
    const uint64_t expected[] = {f32_bytes, f32_bytes, q8_bytes, q8_bytes};
    for (size_t index = 0; index < 4; ++index) {
        if (ranges[index].bytes != expected[index]) return false;
    }
    if (!all_disjoint(ranges, 4)) return false;
    if (!inspect_device) return true;
    if (!sm70_device(ordinal)) return false;
    for (const Range range : ranges) {
        if (!device_range(range, ordinal)) return false;
    }
    return true;
}

augur_cuda_result_v1 submit_prefill_kv8_cache_quantize(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_cache_quantize_args_v1* args) {
    if (!prefill_kv8_cache_quantize_metadata_valid(
            ordinal, stream, args, false)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    prefill_kv8_cache_quantize_kernel<<<
        args->tokens * kFullKvHeads, kFullHeadDim, 0, stream->stream>>>(
        reinterpret_cast<const float*>(
            static_cast<uintptr_t>(args->key_cache_f32)),
        reinterpret_cast<const float*>(
            static_cast<uintptr_t>(args->value_cache_f32)),
        reinterpret_cast<uint8_t*>(
            static_cast<uintptr_t>(args->key_cache_q8_0)),
        reinterpret_cast<uint8_t*>(
            static_cast<uintptr_t>(args->value_cache_q8_0)),
        args->position_base, args->tokens, args->capacity);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

#if defined(AUGUR_CUDA_QWEN35_TC_PREFIX_ATTENTION)
constexpr uint32_t kTcQueriesPerKv = kFullQueryHeads / kFullKvHeads;
constexpr uint64_t kTcWorkspaceLimitBytes = UINT64_C(256) * 1024U * 1024U;
constexpr float kTcResidualScale = 4096.0f;

struct TcPrefixFootprint {
    uint64_t expected[12];
    uint64_t workspace_bytes;
    uint32_t query_rows;
    uint32_t total_positions;
};

bool tc_prefix_footprint(
    const augur_cuda_qwen35_tc_prefix_attention_args_v1* args,
    TcPrefixFootprint* footprint) {
    if (args == nullptr || footprint == nullptr || args->tokens == 0
        || args->tokens > AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_MAX_TOKENS_V1
        || args->capacity == 0 || args->capacity > kKv8MaxCapacity
        || (!args->page_slot_mode
            && (args->prefix_positions > args->capacity
                || args->tokens > args->capacity - args->prefix_positions))
        || (args->tile_positions
                != AUGUR_CUDA_QWEN35_TC_PREFIX_ATTENTION_TILE_2K_V1
            && args->tile_positions
                != AUGUR_CUDA_QWEN35_TC_PREFIX_ATTENTION_TILE_4K_V1)) {
        return false;
    }

    uint64_t query_elements = 0;
    uint64_t lse_elements = 0;
    uint64_t query_rows = 0;
    uint64_t group_elements = 0;
    uint64_t tile_elements = 0;
    uint64_t score_elements = 0;
    uint64_t cache_bytes = 0;
    if (!checked_mul(args->tokens, kFullQueryElements, &query_elements)
        || !checked_mul(args->tokens, kFullQueryHeads, &lse_elements)
        || !checked_mul(args->tokens, kTcQueriesPerKv, &query_rows)
        || !checked_mul(query_rows, kFullHeadDim, &group_elements)
        || !checked_mul(args->tile_positions, kFullHeadDim, &tile_elements)
        || !checked_mul(query_rows, args->tile_positions, &score_elements)
        || !checked_mul(args->capacity, kKv8CacheRowBytes, &cache_bytes)) {
        return false;
    }

    const uint64_t f32 = sizeof(float);
    const uint64_t f16 = sizeof(__half);
    uint64_t query_bytes = 0;
    uint64_t lse_bytes = 0;
    uint64_t group_f32_bytes = 0;
    uint64_t group_f16_bytes = 0;
    uint64_t query_workspace_f16_bytes = 0;
    uint64_t tile_f16_bytes = 0;
    uint64_t score_f32_bytes = 0;
    uint64_t probability_pair_bytes = 0;
    uint64_t tile_lse_bytes = 0;
    if (!checked_mul(query_elements, f32, &query_bytes)
        || !checked_mul(lse_elements, f32, &lse_bytes)
        || !checked_mul(group_elements, f32, &group_f32_bytes)
        || !checked_mul(group_elements, f16, &group_f16_bytes)
        || !checked_mul(group_f16_bytes, 2U, &query_workspace_f16_bytes)
        || !checked_mul(tile_elements, f16, &tile_f16_bytes)
        || !checked_mul(score_elements, f32, &score_f32_bytes)
        || !checked_mul(score_elements, 2U * f16, &probability_pair_bytes)
        || !checked_mul(query_rows, f32, &tile_lse_bytes)) {
        return false;
    }

    const uint64_t expected[] = {
        query_bytes,
        cache_bytes,
        cache_bytes,
        query_bytes,
        lse_bytes,
        query_workspace_f16_bytes,
        tile_f16_bytes,
        tile_f16_bytes,
        score_f32_bytes,
        probability_pair_bytes,
        group_f32_bytes,
        tile_lse_bytes,
    };
    uint64_t workspace_bytes = 0;
    for (size_t index = 0; index < 12; ++index) {
        footprint->expected[index] = expected[index];
        if (index >= 5
            && !checked_add(workspace_bytes, expected[index], &workspace_bytes)) {
            return false;
        }
    }
    if (workspace_bytes > kTcWorkspaceLimitBytes
        || query_rows > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    footprint->workspace_bytes = workspace_bytes;
    footprint->query_rows = static_cast<uint32_t>(query_rows);
    footprint->total_positions = args->prefix_positions + args->tokens;
    return true;
}

bool tc_prefix_metadata_valid(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_tc_prefix_attention_args_v1* args,
    bool inspect_device,
    TcPrefixFootprint* footprint) {
    if (plan == nullptr || plan->handle == nullptr || stream == nullptr
        || stream->stream == nullptr || plan->ordinal != ordinal
        || plan->owner != stream || stream->ordinal != ordinal || args == nullptr
        || args->abi_version
            != AUGUR_CUDA_QWEN35_TC_PREFIX_ATTENTION_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args)
        || args->query_heads != kFullQueryHeads
        || args->kv_heads != kFullKvHeads
        || args->head_dim != kFullHeadDim
        || args->tokens > plan->snapshot.max_tokens
        || !tc_prefix_footprint(args, footprint)
        || args->key_begin > args->key_end
        || args->key_end > footprint->total_positions
        || args->continuation_mode > 1U
        || args->page_slot_mode > 1U
        || (!args->page_slot_mode
            && static_cast<uint64_t>(args->key_end)
                > static_cast<uint64_t>(args->capacity))
        || args->key_begin < args->cache_position_base
        || args->key_end < args->cache_position_base
        || static_cast<uint64_t>(args->key_end)
            > static_cast<uint64_t>(args->cache_position_base)
                + static_cast<uint64_t>(args->capacity)) {
        return false;
    }
    float scale = 0.0f;
    std::memcpy(&scale, &args->scale_bits, sizeof(scale));
    if (scale != 0.0625f) return false;

    const uint64_t addresses[] = {
        args->query_f32,
        args->key_cache_q8_0,
        args->value_cache_q8_0,
        args->output_f32,
        args->lse_f32,
        args->query_workspace_f16,
        args->key_workspace_f16,
        args->value_workspace_f16,
        args->score_workspace_f32,
        args->probability_workspace_f16,
        args->tile_output_workspace_f32,
        args->tile_lse_workspace_f32,
        plan->snapshot.decoded_weight_f16,
        plan->snapshot.decoded_input_f16,
    };
    const uint64_t lengths[] = {
        args->query_len_bytes,
        args->key_cache_len_bytes,
        args->value_cache_len_bytes,
        args->output_len_bytes,
        args->lse_len_bytes,
        args->query_workspace_len_bytes,
        args->key_workspace_len_bytes,
        args->value_workspace_len_bytes,
        args->score_workspace_len_bytes,
        args->probability_workspace_len_bytes,
        args->tile_output_workspace_len_bytes,
        args->tile_lse_workspace_len_bytes,
        plan->snapshot.decoded_weight_len_bytes,
        plan->snapshot.decoded_input_len_bytes,
    };
    Range ranges[14];
    for (size_t index = 0; index < 14; ++index) {
        const uint64_t alignment = index == 1 || index == 2
            ? alignof(uint16_t)
            : index == 5 || index == 6 || index == 7 || index == 9
                || index == 12 || index == 13
                ? 16U
                : alignof(float);
        ranges[index] = {addresses[index], lengths[index], alignment};
        if (index < 12 && lengths[index] != footprint->expected[index]) {
            return false;
        }
    }
    if (!all_disjoint(ranges, 14)) return false;
    if (!inspect_device) return true;
    if (!sm70_device(ordinal)) return false;
    for (const Range range : ranges) {
        if (!device_range(range, ordinal)) return false;
    }
    return true;
}

__device__ __forceinline__ float tc_decode_q8(
    const uint8_t* cache,
    uint32_t capacity,
    uint32_t cache_position_base,
    uint32_t position,
    uint32_t kv_head,
    uint32_t component) {
    const uint32_t block = component / kKv8BlockElements;
    const uint32_t lane = component % kKv8BlockElements;
    const uint64_t slot_position = static_cast<uint64_t>(position)
        - static_cast<uint64_t>(cache_position_base);
    const uint64_t offset = slot_position * kKv8CacheRowBytes
        + static_cast<uint64_t>(kv_head) * kKv8HeadRowBytes
        + static_cast<uint64_t>(block) * kKv8BlockBytes;
    const uint8_t* bytes = cache + offset;
    const uint16_t scale_bits = static_cast<uint16_t>(bytes[0])
        | static_cast<uint16_t>(bytes[1]) << 8U;
    const float scale = __half2float(__ushort_as_half(scale_bits));
    return scale * static_cast<float>(static_cast<int8_t>(bytes[2U + lane]));
}

__global__ void tc_prefix_initialize_kernel(
    float* output,
    float* lse,
    uint32_t tokens,
    uint32_t kv_head) {
    const uint32_t row = blockIdx.x;
    const uint32_t component = threadIdx.x;
    if (row >= tokens * kTcQueriesPerKv || component >= kFullHeadDim) return;
    const uint32_t query_local = row / tokens;
    const uint32_t token = row % tokens;
    const uint32_t query_head = kv_head * kTcQueriesPerKv + query_local;
    const uint64_t output_index =
        (static_cast<uint64_t>(token) * kFullQueryHeads + query_head)
            * kFullHeadDim + component;
    output[output_index] = 0.0f;
    if (component == 0) {
        lse[static_cast<uint64_t>(token) * kFullQueryHeads + query_head]
            = -INFINITY;
    }
}

__global__ void tc_prefix_pack_query_kernel(
    const float* query,
    __half* packed,
    uint32_t tokens,
    uint32_t kv_head,
    bool residual) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const uint64_t elements = static_cast<uint64_t>(tokens)
        * kTcQueriesPerKv * kFullHeadDim;
    if (index >= elements) return;
    const uint32_t component = static_cast<uint32_t>(index % kFullHeadDim);
    const uint32_t row = static_cast<uint32_t>(index / kFullHeadDim);
    const uint32_t query_local = row / tokens;
    const uint32_t token = row % tokens;
    const uint32_t query_head = kv_head * kTcQueriesPerKv + query_local;
    const uint64_t query_index =
        (static_cast<uint64_t>(token) * kFullQueryHeads + query_head)
            * kFullHeadDim + component;
    const float value = query[query_index];
    const __half high = __float2half_rn(value);
    // A power-of-two scale keeps the low term out of FP16 subnormal range.
    // The second QK submission undoes this scale in FP32 accumulation.
    packed[index] = residual
        ? __float2half_rn((value - __half2float(high)) * 4096.0f)
        : high;
}

__global__ void tc_prefix_dequant_tile_kernel(
    const uint8_t* key_cache,
    const uint8_t* value_cache,
    __half* key,
    __half* value,
    uint32_t capacity,
    uint32_t cache_position_base,
    uint32_t kv_head,
    uint32_t tile_begin,
    uint32_t tile_count,
    bool key_residual,
    bool value_residual) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const uint64_t elements = static_cast<uint64_t>(tile_count) * kFullHeadDim;
    if (index >= elements) return;
    const uint32_t component = static_cast<uint32_t>(index % kFullHeadDim);
    const uint32_t position = tile_begin
        + static_cast<uint32_t>(index / kFullHeadDim);
    const float key_value = tc_decode_q8(
        key_cache, capacity, cache_position_base, position, kv_head, component);
    const __half key_high = __float2half_rn(key_value);
    // Keep the dequantized K correction in the same reusable FP16 tile.  A
    // power-of-two scale preserves the low bits through the tensor-core
    // input conversion; the caller applies the inverse scale in FP32.
    key[index] = key_residual
        ? __float2half_rn((key_value - __half2float(key_high)) * 4096.0f)
        : key_high;
    const float value_value = tc_decode_q8(
        value_cache, capacity, cache_position_base, position, kv_head, component);
    const __half value_high = __float2half_rn(value_value);
    value[index] = value_residual
        ? __float2half_rn((value_value - __half2float(value_high)) * 4096.0f)
        : value_high;
}

__global__ void tc_prefix_dequant_value_residual_kernel(
    const uint8_t* value_cache,
    __half* value,
    uint32_t capacity,
    uint32_t cache_position_base,
    uint32_t kv_head,
    uint32_t tile_begin,
    uint32_t tile_count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * kThreads
        + threadIdx.x;
    const uint64_t elements = static_cast<uint64_t>(tile_count) * kFullHeadDim;
    if (index >= elements) return;
    const uint32_t component = static_cast<uint32_t>(index % kFullHeadDim);
    const uint32_t position = tile_begin
        + static_cast<uint32_t>(index / kFullHeadDim);
    const float value_value = tc_decode_q8(
        value_cache, capacity, cache_position_base, position, kv_head, component);
    const __half value_high = __float2half_rn(value_value);
    value[index] = __float2half_rn(
        (value_value - __half2float(value_high)) * 4096.0f);
}

__global__ void tc_prefix_dequant_key_residual_kernel(
    const uint8_t* key_cache,
    __half* key,
    uint32_t capacity,
    uint32_t cache_position_base,
    uint32_t kv_head,
    uint32_t tile_begin,
    uint32_t tile_count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * kThreads
        + threadIdx.x;
    const uint64_t elements = static_cast<uint64_t>(tile_count) * kFullHeadDim;
    if (index >= elements) return;
    const uint32_t component = static_cast<uint32_t>(index % kFullHeadDim);
    const uint32_t position = tile_begin
        + static_cast<uint32_t>(index / kFullHeadDim);
    const float key_value = tc_decode_q8(
        key_cache, capacity, cache_position_base, position, kv_head, component);
    const __half key_high = __float2half_rn(key_value);
    key[index] = __float2half_rn(
        (key_value - __half2float(key_high)) * 4096.0f);
}

__global__ void tc_prefix_softmax_tile_kernel(
    const float* scores,
    __half* probabilities,
    float* tile_lse,
    uint32_t tokens,
    uint32_t prefix_positions,
    uint32_t tile_begin,
    uint32_t tile_count) {
    __shared__ float reduction[256];
    const uint32_t row = blockIdx.x;
    const uint32_t thread = threadIdx.x;
    const uint32_t token = row % tokens;
    const uint32_t visible = prefix_positions + token + 1U;
    const uint64_t plane_elements = static_cast<uint64_t>(tokens)
        * kTcQueriesPerKv * tile_count;
    float local_maximum = -INFINITY;
    for (uint32_t position = thread; position < tile_count;
         position += blockDim.x) {
        if (tile_begin + position < visible) {
            local_maximum = fmaxf(
                local_maximum,
                scores[static_cast<uint64_t>(row) * tile_count + position]);
        }
    }
    reduction[thread] = local_maximum;
    __syncthreads();
    for (uint32_t stride = 128U; stride != 0U; stride >>= 1U) {
        if (thread < stride) {
            reduction[thread] = fmaxf(
                reduction[thread], reduction[thread + stride]);
        }
        __syncthreads();
    }
    const float maximum = reduction[0];
    // Each warp must capture maximum before any warp reuses reduction for
    // the denominator. A reduction-end barrier alone does not order these
    // later reads against the next phase's writes.
    __syncthreads();
    if (!isfinite(maximum)) {
        for (uint32_t position = thread; position < tile_count;
             position += blockDim.x) {
            const uint64_t index = static_cast<uint64_t>(row) * tile_count
                + position;
            probabilities[index] = __float2half_rn(0.0f);
            probabilities[plane_elements + index] = __float2half_rn(0.0f);
        }
        if (thread == 0) tile_lse[row] = -INFINITY;
        return;
    }

    float local_sum = 0.0f;
    for (uint32_t position = thread; position < tile_count;
         position += blockDim.x) {
        if (tile_begin + position < visible) {
            local_sum += expf(
                scores[static_cast<uint64_t>(row) * tile_count + position]
                - maximum);
        }
    }
    reduction[thread] = local_sum;
    __syncthreads();
    for (uint32_t stride = 128U; stride != 0U; stride >>= 1U) {
        if (thread < stride) reduction[thread] += reduction[thread + stride];
        __syncthreads();
    }
    const float sum = reduction[0];
    for (uint32_t position = thread; position < tile_count;
         position += blockDim.x) {
        const bool valid = tile_begin + position < visible;
        const float probability = valid
            ? expf(scores[static_cast<uint64_t>(row) * tile_count + position]
                    - maximum) / sum
            : 0.0f;
        const uint64_t index = static_cast<uint64_t>(row) * tile_count + position;
        const __half high = __float2half_rn(probability);
        probabilities[index] = high;
        probabilities[plane_elements + index] = __float2half_rn(
            (probability - __half2float(high)) * kTcResidualScale);
    }
    if (thread == 0) tile_lse[row] = maximum + logf(sum);
}

__global__ void tc_prefix_merge_tile_kernel(
    float* output,
    float* lse,
    const float* tile_output,
    const float* tile_lse,
    uint32_t tokens,
    uint32_t kv_head) {
    __shared__ float old_weight;
    __shared__ float tile_weight;
    __shared__ float merged_lse;
    const uint32_t row = blockIdx.x;
    const uint32_t component = threadIdx.x;
    const uint32_t query_local = row / tokens;
    const uint32_t token = row % tokens;
    const uint32_t query_head = kv_head * kTcQueriesPerKv + query_local;
    const uint64_t lse_index =
        static_cast<uint64_t>(token) * kFullQueryHeads + query_head;
    if (component == 0) {
        const float old_lse = lse[lse_index];
        const float next_lse = tile_lse[row];
        if (!isfinite(next_lse)) {
            merged_lse = old_lse;
            old_weight = 1.0f;
            tile_weight = 0.0f;
        } else if (!isfinite(old_lse)) {
            merged_lse = next_lse;
            old_weight = 0.0f;
            tile_weight = 1.0f;
        } else {
            const float maximum = fmaxf(old_lse, next_lse);
            const float old_scale = expf(old_lse - maximum);
            const float next_scale = expf(next_lse - maximum);
            const float denominator = old_scale + next_scale;
            merged_lse = maximum + logf(denominator);
            old_weight = old_scale / denominator;
            tile_weight = next_scale / denominator;
        }
    }
    __syncthreads();
    const uint64_t output_index =
        (static_cast<uint64_t>(token) * kFullQueryHeads + query_head)
            * kFullHeadDim + component;
    output[output_index] = output[output_index] * old_weight
        + tile_output[static_cast<uint64_t>(row) * kFullHeadDim + component]
            * tile_weight;
    if (component == 0) lse[lse_index] = merged_lse;
}

__global__ void tc_prefix_simt_reference_kernel(
    const float* query,
    const uint8_t* key_cache,
    const uint8_t* value_cache,
    float* output,
    float* lse,
    uint32_t tokens,
    uint32_t prefix_positions,
    uint32_t capacity,
    uint32_t cache_position_base,
    uint32_t key_begin,
    uint32_t key_end,
    float scale) {
    __shared__ float scores[kTcQueriesPerKv];
    __shared__ float previous_lse[kTcQueriesPerKv];
    const uint32_t token = blockIdx.x / kFullKvHeads;
    const uint32_t kv_head = blockIdx.x % kFullKvHeads;
    const uint32_t component = threadIdx.x;
    const uint32_t warp = component / 32U;
    const uint32_t lane = component & 31U;
    if (component < kTcQueriesPerKv) {
        const uint32_t query_head = kv_head * kTcQueriesPerKv + component;
        previous_lse[component] =
            lse[static_cast<uint64_t>(token) * kFullQueryHeads + query_head];
    }
    __syncthreads();
    float query_fragment[8];
    #pragma unroll
    for (uint32_t item = 0; item < 8U; ++item) {
        if (warp < kTcQueriesPerKv) {
            const uint32_t query_head = kv_head * kTcQueriesPerKv + warp;
            const uint32_t query_component = lane + item * 32U;
            const uint64_t query_index =
                (static_cast<uint64_t>(token) * kFullQueryHeads + query_head)
                    * kFullHeadDim + query_component;
            query_fragment[item] = query[query_index];
        } else {
            query_fragment[item] = 0.0f;
        }
    }

    float running_maximum[kTcQueriesPerKv];
    float running_sum[kTcQueriesPerKv];
    float running_output[kTcQueriesPerKv];
    #pragma unroll
    for (uint32_t query_local = 0; query_local < kTcQueriesPerKv;
         ++query_local) {
        running_maximum[query_local] = -INFINITY;
        running_sum[query_local] = 0.0f;
        running_output[query_local] = 0.0f;
    }

    const uint32_t visible = prefix_positions + token + 1U;
    const uint32_t segment_end = key_end < visible ? key_end : visible;
    for (uint32_t position = key_begin; position < segment_end; ++position) {
        double score = 0.0;
        if (warp < kTcQueriesPerKv) {
            #pragma unroll
            for (uint32_t item = 0; item < 8U; ++item) {
                const uint32_t query_component = lane + item * 32U;
                score += static_cast<double>(query_fragment[item])
                    * static_cast<double>(tc_decode_q8(
                        key_cache, capacity, cache_position_base, position, kv_head,
                        query_component));
            }
            #pragma unroll
            for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
                score += __shfl_down_sync(0xffffffffU, score, offset);
            }
            if (lane == 0) scores[warp] = static_cast<float>(score) * scale;
        }
        __syncthreads();
        const float value = tc_decode_q8(
            value_cache, capacity, cache_position_base, position, kv_head, component);
        #pragma unroll
        for (uint32_t query_local = 0; query_local < kTcQueriesPerKv;
             ++query_local) {
            const float next_maximum = fmaxf(
                running_maximum[query_local], scores[query_local]);
            const float previous_scale = running_sum[query_local] == 0.0f
                ? 0.0f
                : expf(running_maximum[query_local] - next_maximum);
            const float current_scale = expf(scores[query_local] - next_maximum);
            running_sum[query_local] = running_sum[query_local] * previous_scale
                + current_scale;
            running_output[query_local] =
                running_output[query_local] * previous_scale
                + current_scale * value;
            running_maximum[query_local] = next_maximum;
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t query_local = 0; query_local < kTcQueriesPerKv;
         ++query_local) {
        const uint32_t query_head = kv_head * kTcQueriesPerKv + query_local;
        const uint64_t output_index =
            (static_cast<uint64_t>(token) * kFullQueryHeads + query_head)
                * kFullHeadDim + component;
        if (running_sum[query_local] == 0.0f) continue;
        const float segment_lse = running_maximum[query_local]
            + logf(running_sum[query_local]);
        const float old_lse = previous_lse[query_local];
        float old_weight = 0.0f;
        float segment_weight = 1.0f;
        float merged_lse = segment_lse;
        if (isfinite(old_lse)) {
            const float maximum = fmaxf(old_lse, segment_lse);
            const float old_scale = expf(old_lse - maximum);
            const float segment_scale = expf(segment_lse - maximum);
            const float denominator = old_scale + segment_scale;
            old_weight = old_scale / denominator;
            segment_weight = segment_scale / denominator;
            merged_lse = maximum + logf(denominator);
        }
        output[output_index] = output[output_index] * old_weight
            + (running_output[query_local] / running_sum[query_local])
                * segment_weight;
        if (component == 0) {
            lse[static_cast<uint64_t>(token) * kFullQueryHeads + query_head]
                = merged_lse;
        }
    }
}

augur_cuda_result_v1 submit_tc_prefix_attention(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_tc_prefix_attention_args_v1* args) {
    TcPrefixFootprint footprint{};
    if (!tc_prefix_metadata_valid(
            ordinal, stream, plan, args, false, &footprint)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) {
        return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    }
    const auto* query = reinterpret_cast<const float*>(
        static_cast<uintptr_t>(args->query_f32));
    const auto* key_cache = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(args->key_cache_q8_0));
    const auto* value_cache = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(args->value_cache_q8_0));
    auto* output = reinterpret_cast<float*>(
        static_cast<uintptr_t>(args->output_f32));
    auto* lse = reinterpret_cast<float*>(
        static_cast<uintptr_t>(args->lse_f32));
    auto* packed_query = reinterpret_cast<__half*>(
        static_cast<uintptr_t>(args->query_workspace_f16));
    auto* decoded_key = reinterpret_cast<__half*>(
        static_cast<uintptr_t>(args->key_workspace_f16));
    auto* decoded_value = reinterpret_cast<__half*>(
        static_cast<uintptr_t>(args->value_workspace_f16));
    auto* scores = reinterpret_cast<float*>(
        static_cast<uintptr_t>(args->score_workspace_f32));
    auto* probabilities = reinterpret_cast<__half*>(
        static_cast<uintptr_t>(args->probability_workspace_f16));
    auto* tile_output = reinterpret_cast<float*>(
        static_cast<uintptr_t>(args->tile_output_workspace_f32));
    auto* tile_lse = reinterpret_cast<float*>(
        static_cast<uintptr_t>(args->tile_lse_workspace_f32));
    float attention_scale = 0.0f;
    std::memcpy(&attention_scale, &args->scale_bits, sizeof(attention_scale));
    const float residual_attention_scale = attention_scale / kTcResidualScale;
    const float second_order_attention_scale =
        residual_attention_scale / kTcResidualScale;
    // PV acts on normalized probabilities: unlike QK it has no attention
    // scale. Reusing residual_attention_scale here under-corrects by 16x.
    const float residual_value_scale = 1.0f / kTcResidualScale;
    const float one = 1.0f;
    const float zero = 0.0f;
    const uint64_t packed_elements = static_cast<uint64_t>(footprint.query_rows)
        * kFullHeadDim;
    auto* packed_query_residual = packed_query + packed_elements;

    for (uint32_t kv_head = 0; kv_head < kFullKvHeads; ++kv_head) {
        if (args->key_begin == 0 && args->continuation_mode == 0) {
            const uint32_t rows = args->tokens * kTcQueriesPerKv;
            tc_prefix_initialize_kernel<<<
                rows, kFullHeadDim, 0, stream->stream>>>(
                output, lse, args->tokens, kv_head);
        }
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
        }

        // The query tile is invariant across the KV tiles for this KV head.
        // Pack it once per head instead of repeating the same launch for every
        // causal key tile; this is a large launch/memory-write reduction at
        // long context and does not change the numerical wire.
        tc_prefix_pack_query_kernel<<<
            launch_blocks(packed_elements), kThreads, 0, stream->stream>>>(
            query, packed_query, args->tokens, kv_head, false);
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
        }
        tc_prefix_pack_query_kernel<<<
            launch_blocks(packed_elements), kThreads, 0, stream->stream>>>(
            query, packed_query_residual, args->tokens, kv_head, true);
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
        }

        for (uint32_t tile_begin = args->key_begin;
             tile_begin < args->key_end;
             tile_begin += args->tile_positions) {
            const uint32_t remaining = args->key_end - tile_begin;
            const uint32_t tile_count = remaining < args->tile_positions
                ? remaining
                : args->tile_positions;
            const uint64_t tile_elements =
                static_cast<uint64_t>(tile_count) * kFullHeadDim;
            tc_prefix_dequant_tile_kernel<<<
                launch_blocks(tile_elements), kThreads, 0, stream->stream>>>(
                key_cache, value_cache, decoded_key, decoded_value,
                args->capacity, args->cache_position_base, kv_head,
                tile_begin, tile_count, false, false);
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
            }

            const cublasStatus_t qk = cublasGemmEx(
                plan->handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                static_cast<int>(tile_count),
                static_cast<int>(footprint.query_rows),
                static_cast<int>(kFullHeadDim),
                &attention_scale,
                decoded_key,
                CUDA_R_16F,
                static_cast<int>(kFullHeadDim),
                packed_query,
                CUDA_R_16F,
                static_cast<int>(kFullHeadDim),
                &zero,
                scores,
                CUDA_R_32F,
                static_cast<int>(tile_count),
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (qk != CUBLAS_STATUS_SUCCESS) {
                return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, qk);
            }

            const cublasStatus_t qk_residual = cublasGemmEx(
                plan->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(tile_count),
                static_cast<int>(footprint.query_rows),
                static_cast<int>(kFullHeadDim),
                &residual_attention_scale,
                decoded_key, CUDA_R_16F, static_cast<int>(kFullHeadDim),
                packed_query_residual, CUDA_R_16F, static_cast<int>(kFullHeadDim),
                &one, scores, CUDA_R_32F, static_cast<int>(tile_count),
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (qk_residual != CUBLAS_STATUS_SUCCESS) {
                return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, qk_residual);
            }

            // Recover the low bits lost by the FP16 K conversion.  The Q_hi
            // term is repacked after Q_residual has consumed the query
            // workspace; decoded_key is likewise reused for K_residual.  This
            // adds Q_hi*K_residual/4096 without allocating a second tile.
            tc_prefix_dequant_key_residual_kernel<<<
                launch_blocks(tile_elements), kThreads, 0, stream->stream>>>(
                key_cache, decoded_key, args->capacity,
                args->cache_position_base, kv_head,
                tile_begin, tile_count);
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
            }
            // Q_low remains live in the second query plane. Include its
            // product with K_low without repacking per tile.
            const cublasStatus_t qk_second_order = cublasGemmEx(
                plan->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(tile_count),
                static_cast<int>(footprint.query_rows),
                static_cast<int>(kFullHeadDim),
                &second_order_attention_scale,
                decoded_key, CUDA_R_16F, static_cast<int>(kFullHeadDim),
                packed_query_residual, CUDA_R_16F, static_cast<int>(kFullHeadDim),
                &one, scores, CUDA_R_32F, static_cast<int>(tile_count),
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (qk_second_order != CUBLAS_STATUS_SUCCESS) {
                return blas_result(
                    AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, qk_second_order);
            }
            const cublasStatus_t qk_key_residual = cublasGemmEx(
                plan->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(tile_count),
                static_cast<int>(footprint.query_rows),
                static_cast<int>(kFullHeadDim),
                &residual_attention_scale,
                decoded_key, CUDA_R_16F, static_cast<int>(kFullHeadDim),
                packed_query, CUDA_R_16F, static_cast<int>(kFullHeadDim),
                &one, scores, CUDA_R_32F, static_cast<int>(tile_count),
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (qk_key_residual != CUBLAS_STATUS_SUCCESS) {
                return blas_result(
                    AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, qk_key_residual);
            }

            tc_prefix_softmax_tile_kernel<<<
                footprint.query_rows, kThreads, 0, stream->stream>>>(
                scores, probabilities, tile_lse, args->tokens,
                args->prefix_positions, tile_begin, tile_count);
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
            }

            // Two FP16 probability planes encode P_hi and 4096*P_low.
            // Accumulate all four products of the P/V expansions in FP32.
            // Both GEMM inputs have the same supported type; no mixed-type
            // fallback, host wait, or runtime allocation is required.
            const uint64_t probability_plane_elements =
                static_cast<uint64_t>(footprint.query_rows) * tile_count;
            for (uint32_t value_part = 0; value_part < 2U; ++value_part) {
                if (value_part != 0U) {
                    tc_prefix_dequant_value_residual_kernel<<<
                        launch_blocks(tile_elements), kThreads, 0, stream->stream>>>(
                        value_cache, decoded_value, args->capacity,
                        args->cache_position_base, kv_head,
                        tile_begin, tile_count);
                    error = cudaGetLastError();
                    if (error != cudaSuccess) {
                        return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
                    }
                }
                for (uint32_t probability_part = 0; probability_part < 2U;
                     ++probability_part) {
                    const float alpha =
                        (value_part == 0U ? one : residual_value_scale)
                        * (probability_part == 0U ? one : residual_value_scale);
                    const float beta = value_part == 0U && probability_part == 0U
                        ? zero : one;
                    const cublasStatus_t pv = cublasGemmEx(
                        plan->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        static_cast<int>(kFullHeadDim),
                        static_cast<int>(footprint.query_rows),
                        static_cast<int>(tile_count), &alpha,
                        decoded_value, CUDA_R_16F, static_cast<int>(kFullHeadDim),
                        probabilities + probability_part * probability_plane_elements,
                        CUDA_R_16F, static_cast<int>(tile_count),
                        &beta, tile_output, CUDA_R_32F, static_cast<int>(kFullHeadDim),
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    if (pv != CUBLAS_STATUS_SUCCESS) {
                        return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, pv);
                    }
                }
            }

            tc_prefix_merge_tile_kernel<<<
                footprint.query_rows, kFullHeadDim, 0, stream->stream>>>(
                output, lse, tile_output, tile_lse, args->tokens, kv_head);
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
            }
        }
    }
    return result(AUGUR_CUDA_STATUS_OK, 0);
}

augur_cuda_result_v1 submit_simt_prefix_attention(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_tc_prefix_attention_args_v1* args) {
    TcPrefixFootprint footprint{};
    if (!tc_prefix_metadata_valid(
            ordinal, stream, plan, args, false, &footprint)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) {
        return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    }
    float scale = 0.0f;
    std::memcpy(&scale, &args->scale_bits, sizeof(scale));
    if (args->key_begin == 0 && args->continuation_mode == 0) {
        const uint32_t rows = args->tokens * kTcQueriesPerKv;
        for (uint32_t kv_head = 0; kv_head < kFullKvHeads; ++kv_head) {
            tc_prefix_initialize_kernel<<<
                rows, kFullHeadDim, 0, stream->stream>>>(
                reinterpret_cast<float*>(
                    static_cast<uintptr_t>(args->output_f32)),
                reinterpret_cast<float*>(
                    static_cast<uintptr_t>(args->lse_f32)),
                args->tokens, kv_head);
        }
    }
    if (args->key_begin == args->key_end) {
        const cudaError_t error = cudaGetLastError();
        return error == cudaSuccess
            ? result(AUGUR_CUDA_STATUS_OK, 0)
            : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    }
    tc_prefix_simt_reference_kernel<<<
        args->tokens * kFullKvHeads, kFullHeadDim, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query_f32)),
        reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(args->key_cache_q8_0)),
        reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(args->value_cache_q8_0)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->lse_f32)),
        args->tokens, args->prefix_positions, args->capacity,
        args->cache_position_base, args->key_begin, args->key_end, scale);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}
#endif
#endif

}  // namespace

extern "C" uint32_t AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_cublas_compiled_max_tokens_v1(void) {
    return AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_cublas_plan_create_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_cublas_plan_args_v1* args,
    augur_cuda_qwen35_prefill_cublas_plan_v1** out_plan) {
    if (out_plan == nullptr) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    *out_plan = nullptr;
    if (!plan_preflight_valid(ordinal, stream, args)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    auto* plan = new (std::nothrow) augur_cuda_qwen35_prefill_cublas_plan_v1{};
    if (plan == nullptr) return result(AUGUR_CUDA_STATUS_OUT_OF_MEMORY, 0);
    plan->ordinal = ordinal;
    plan->owner = stream;
    plan->handle = nullptr;
    plan->snapshot = *args;

    int multiprocessors = 0;
    const cudaError_t device_status = cudaDeviceGetAttribute(
        &multiprocessors, cudaDevAttrMultiProcessorCount, ordinal);
    if (device_status != cudaSuccess || multiprocessors <= 0) {
        delete plan;
        return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,
            device_status != cudaSuccess ? device_status : cudaErrorInvalidDevice);
    }
    plan->multiprocessors = static_cast<uint32_t>(multiprocessors);

    cublasStatus_t status = cublasCreate(&plan->handle);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasSetStream(plan->handle, stream->stream);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasSetPointerMode(plan->handle, CUBLAS_POINTER_MODE_HOST);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasSetMathMode(plan->handle, CUBLAS_TENSOR_OP_MATH);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasSetAtomicsMode(plan->handle, CUBLAS_ATOMICS_NOT_ALLOWED);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        if (plan->handle != nullptr) cublasDestroy(plan->handle);
        delete plan;
        return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, status);
    }
    *out_plan = plan;
    return result(AUGUR_CUDA_STATUS_OK, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_cublas_projection_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    Footprint footprint{};
    return projection_metadata_valid(
        ordinal, stream, plan, args, true, &footprint)
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_cublas_projection_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    Footprint footprint{};
    if (!projection_metadata_valid(
            ordinal, stream, plan, args, false, &footprint)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    return launch(ordinal, stream, plan, args, footprint);
}

extern "C" uint64_t AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_input_integer_scratch_bytes_v1(
    uint32_t format, uint32_t rows, uint32_t cols, uint32_t tokens) {
    if (format != AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1
        || tokens != 512U || cols != 5120U
        || !(rows == 10240U || rows == 6144U || rows == 12288U || rows == 1024U))
        return 0U;
    return uint64_t(augur_prefill_integer_q8::tile_rows + tokens)
        * (cols / augur_prefill_integer_q8::dot_codes)
        * (augur_prefill_integer_q8::dot_codes * 2U + 4U);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_input_integer_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    Footprint footprint{};
    if (!projection_metadata_valid(ordinal, stream, plan, args, false, &footprint))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const uint64_t scratch = augur_cuda_qwen35_prefill_input_integer_scratch_bytes_v1(
        args->weight_format, args->rows, args->cols, args->tokens);
    if (scratch == 0U || scratch > plan->snapshot.decoded_weight_len_bytes
        || args->tensor_scale != 1.0f)
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    const cudaError_t error = augur_prefill_integer_q8::launch_prevalidated(stream->stream,
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->input_q8_1)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(plan->snapshot.decoded_weight_f16)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        args->tokens, args->rows, args->cols);
    return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_input_llama_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    Footprint footprint{};
    if (!projection_metadata_valid(ordinal, stream, plan, args, false, &footprint))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    // Same bounded input shapes, but NOT the exact integer arithmetic policy.
    // Full decoded extents, disjoint views and owner/stream were checked above.
    if (augur_cuda_qwen35_prefill_input_integer_scratch_bytes_v1(
            args->weight_format, args->rows, args->cols, args->tokens) == 0U
        || args->tensor_scale != 1.0f || args->weight % 4U != 0U
        || footprint.weight_elements % 2048U != 0U)
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    return launch(ordinal, stream, plan, args, footprint, false, 3U, false, false, true);
}

extern "C" uint64_t AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_integer_mma_scratch_bytes_v1(
    uint32_t format, uint32_t rows, uint32_t cols, uint32_t tokens) {
    const bool nv = format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1;
    const bool q8 = format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1;
    const bool admitted = nv ? (tokens == 128U || tokens == 256U || tokens == 512U)
                             : (q8 && (tokens == 256U || tokens == 512U));
    if (!admitted || !((rows == 17408U && cols == 5120U) || (rows == 5120U && cols == 17408U)))
        return 0U;
    // Shared upper bound, borrowed from the preflighted plan's weight scratch.
    return nv ? augur_prefill_integer_mma::scratch_bytes(tokens, format)
              : augur_prefill_integer_q8::scratch_bytes(tokens, format);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_integer_mma_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    Footprint footprint{};
    if (!projection_metadata_valid(ordinal, stream, plan, args, false, &footprint))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const uint64_t scratch = augur_cuda_qwen35_prefill_integer_mma_scratch_bytes_v1(
        args->weight_format, args->rows, args->cols, args->tokens);
    if (scratch == 0U || scratch > plan->snapshot.decoded_weight_len_bytes)
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const bool q8 = args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1;
    if (q8 && args->tensor_scale != 1.0f) return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    const cudaError_t error = q8 ? augur_prefill_integer_q8::launch_prevalidated(stream->stream,
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->input_q8_1)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(plan->snapshot.decoded_weight_f16)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        args->tokens, args->rows, args->cols)
        : augur_prefill_integer_mma::launch_prevalidated(stream->stream,
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->input_q8_1)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(plan->snapshot.decoded_weight_f16)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        args->tokens, args->rows, args->cols, args->tensor_scale);
    return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" uint32_t AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_down_tensor_tile_rows_v1(
    uint32_t format, uint32_t rows, uint32_t cols, uint32_t tokens) {
    return (format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1
            || format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1)
        && rows == 5120U && cols == 17408U
        && (tokens == 64U || tokens == 128U) ? 896U : 0U;
}

extern "C" uint32_t AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_down_compensated_tile_rows_v1(
    uint32_t format, uint32_t rows, uint32_t cols, uint32_t tokens) {
    return format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1
        && rows == 5120U && cols == 17408U && tokens == 128U ? 896U : 0U;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_down_compensated_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    Footprint footprint{};
    if (!projection_metadata_valid(ordinal, stream, plan, args, false, &footprint))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const uint32_t tile_rows = augur_cuda_qwen35_prefill_down_compensated_tile_rows_v1(
        args->weight_format, args->rows, args->cols, args->tokens);
    if (tile_rows == 0U) return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const uint64_t weight_tile_bytes = uint64_t(tile_rows) * args->cols * sizeof(__half);
    // The low activation plane borrows unused space in the existing full
    // weight scratch, beyond the largest working tile; no new allocation.
    uint64_t borrowed_bytes = 0;
    if (!checked_add(weight_tile_bytes, footprint.decoded_input_bytes, &borrowed_bytes)
        || borrowed_bytes > plan->snapshot.decoded_weight_len_bytes)
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    cublasMath_t previous_math;
    cublasStatus_t status = cublasGetMathMode(plan->handle, &previous_math);
    if (status != CUBLAS_STATUS_SUCCESS) return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, status);
    status = cublasSetMathMode(plan->handle, static_cast<cublasMath_t>(
        CUBLAS_TENSOR_OP_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    if (status != CUBLAS_STATUS_SUCCESS) return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, status);
    auto* high_weight = reinterpret_cast<__half*>(static_cast<uintptr_t>(plan->snapshot.decoded_weight_f16));
    auto* low_input = reinterpret_cast<__half*>(
        static_cast<uintptr_t>(plan->snapshot.decoded_weight_f16 + weight_tile_bytes));
    status = compensated_tensor::submit(plan->handle, stream->stream,
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->input_q8_1)),
        high_weight, nullptr,
        reinterpret_cast<__half*>(static_cast<uintptr_t>(plan->snapshot.decoded_input_f16)),
        low_input, reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        args->tokens, args->rows, args->cols, args->weight_format, args->tensor_scale, tile_rows);
    // Restore the shared handle's policy, including on a rejected submission.
    const cublasStatus_t restored = cublasSetMathMode(plan->handle, previous_math);
    if (status != CUBLAS_STATUS_SUCCESS) return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, status);
    if (restored != CUBLAS_STATUS_SUCCESS) return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, restored);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_down_tensor_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    Footprint footprint{};
    // Keep the ordinary plan's owner/range checks. This first integration
    // borrows its existing scratch; it does not allocate a second workspace.
    if (!projection_metadata_valid(ordinal, stream, plan, args, false, &footprint)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    const uint32_t tile_rows = augur_cuda_qwen35_prefill_down_tensor_tile_rows_v1(
        args->weight_format, args->rows, args->cols, args->tokens);
    if (tile_rows == 0U) return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) {
        return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    }
    const auto* weight = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight));
    const auto* input = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->input_q8_1));
    auto* half_weight = reinterpret_cast<__half*>(static_cast<uintptr_t>(plan->snapshot.decoded_weight_f16));
    auto* half_input = reinterpret_cast<__half*>(static_cast<uintptr_t>(plan->snapshot.decoded_input_f16));
    auto* output = reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32));
    decode_q8_1_input_kernel<<<launch_blocks(footprint.input_elements), kThreads, 0, stream->stream>>>(
        input, half_input, footprint.input_elements);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    const uint64_t row_bytes = footprint.weight_bytes / args->rows;
    const float alpha = 1.0f, beta = 0.0f;
    for (uint32_t first = 0U; first < args->rows; first += tile_rows) {
        const uint32_t count = args->rows - first < tile_rows ? args->rows - first : tile_rows;
        const uint64_t elements = uint64_t(count) * args->cols;
        const auto* tile = weight + uint64_t(first) * row_bytes;
        if (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1) {
            decode_weight_quad_kernel<AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1>
                <<<launch_blocks(elements / 4U), kThreads, 0, stream->stream>>>(
                    tile, half_weight, elements, args->tensor_scale);
        } else {
            decode_weight_quad_kernel<AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1>
                <<<launch_blocks(elements / 4U), kThreads, 0, stream->stream>>>(
                    tile, half_weight, elements, args->tensor_scale);
        }
        error = cudaGetLastError();
        if (error != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
        const cublasStatus_t status = cublasGemmEx(plan->handle, CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(count), static_cast<int>(args->tokens), static_cast<int>(args->cols),
            &alpha, half_weight, CUDA_R_16F, static_cast<int>(args->cols),
            half_input, CUDA_R_16F, static_cast<int>(args->cols), &beta,
            output + first, CUDA_R_32F, static_cast<int>(args->rows),
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (status != CUBLAS_STATUS_SUCCESS) {
            return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, status);
        }
    }
    error = cudaGetLastError();
    return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_compare_summary_v1(int32_t ordinal, augur_cuda_stream_t stream,
    uint64_t candidate, uint64_t reference, uint64_t input_bytes,
    uint64_t output, uint64_t output_bytes, uint32_t format) {
    if (stream == nullptr || stream->stream == nullptr || stream->ordinal != ordinal
        || ordinal < 0 || input_bytes == 0 || format > 1U || output_bytes != 64U
        || input_bytes % (format == 0U ? 4U : 34U) != 0) return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const Range ranges[] = {{candidate, input_bytes, format == 0U ? 4U : 2U},
        {reference, input_bytes, format == 0U ? 4U : 2U}, {output, output_bytes, 8U}};
    if (!range_valid(ranges[0]) || !range_valid(ranges[1]) || !range_valid(ranges[2])
        || ranges_overlap(ranges[0], ranges[2]) || ranges_overlap(ranges[1], ranges[2]))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    for (const Range range : ranges) if (!device_range(range, ordinal)) return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    const uint64_t elements = format == 0U ? input_bytes / 4U : (input_bytes / 34U) * 32U;
    compare_summary_kernel<<<1, 256, 0, stream->stream>>>(
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(candidate)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(reference)), elements, format,
        reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(output)));
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, 0) : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

static augur_cuda_result_v1 launch_quantized_small_batch(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args, bool exact_tiles,
    bool row_reuse = false, bool down_row_reuse = false) {
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    const auto* w = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight));
    const auto* a = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->input_q8_1));
    auto* y = reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32));
    const uint32_t tile = exact_tiles ? exact_projection_tile_width(
        args->weight_format, args->rows, args->cols, args->tokens) : 0U;
    const uint32_t row_tile = row_reuse ? projection_row_reuse_width(
        args->weight_format, args->rows, args->cols, args->tokens) : 0U;
    if (down_row_reuse) {
        if (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1)
            quantized_ffn_down_row2_b8_kernel<1>
                <<<args->rows / 4U,128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
        else quantized_ffn_down_row2_b8_kernel<2>
            <<<args->rows / 2U,128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
    } else if (row_tile == 2U) {
        if (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1)
            quantized_small_batch_row_reuse_cols5120_kernel<1,2,true,false,true>
                <<<(args->rows + 1U) / 2U,128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
        else quantized_small_batch_row_reuse_cols5120_kernel<2,2,true,false,true>
            <<<(args->rows + 1U) / 2U,128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
    } else if (tile != 0U) {
        if (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1) {
            if (tile == 4U) quantized_small_batch_tiled_kernel<1,4,2,true>
                <<<dim3(args->rows,2),128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
            else quantized_small_batch_tiled_kernel<1,8,1,true>
                <<<args->rows,128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
        } else {
            // Component-selected schedule only for full8 NVFP4 FFN-down.
            // Q8 was slower with static K/unroll and intentionally stays above.
            if (tile == 4U && args->rows == 5120U && args->cols == 17408U && args->tokens == 8U)
                quantized_small_batch_down_cols17408_kernel<2,4,2,true>
                <<<dim3(args->rows,2),128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
            else if (tile == 4U) quantized_small_batch_tiled_kernel<2,4,2,true>
                <<<dim3(args->rows,2),128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
            else quantized_small_batch_tiled_kernel<2,8,1,true>
                <<<args->rows,128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
        }
    } else if (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1) {
        quantized_small_batch_kernel<AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1>
            <<<args->rows, 128, 0, stream->stream>>>(w, a, y, args->rows, args->cols, args->tokens, args->tensor_scale);
    } else {
        quantized_small_batch_kernel<AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1>
            <<<args->rows, 128, 0, stream->stream>>>(w, a, y, args->rows, args->cols, args->tokens, args->tensor_scale);
    }
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, 0) : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

static augur_cuda_result_v1 quantized_small_batch_submit(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args, bool exact_tiles,
    bool row_reuse = false, bool down_row_reuse = false) {
    Footprint footprint{};
    if (!projection_metadata_valid(ordinal, stream, plan, args, false, &footprint)
        || args->tokens > 8U
        || (down_row_reuse && (args->rows != 5120U || args->cols != 17408U || args->tokens != 8U))
        || (args->weight_format != AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1
            && args->weight_format != AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1)
        || (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1 && args->tensor_scale != 1.0f)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    return launch_quantized_small_batch(ordinal, stream, args, exact_tiles, row_reuse, down_row_reuse);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_packed_small_batch_projection_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args, uint32_t exact_tiles) {
    return packed_projection_metadata_valid(ordinal, stream, args, exact_tiles, true)
        ? result(AUGUR_CUDA_STATUS_OK, 0) : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_packed_small_batch_projection_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args, uint32_t exact_tiles) {
    if (!packed_projection_metadata_valid(ordinal, stream, args, exact_tiles, false))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    return launch_quantized_small_batch(ordinal, stream, args, exact_tiles != 0U);
}

extern "C" uint32_t AUGUR_CUDA_CALL augur_cuda_qwen35_exact_projection_tile_width_v1(
    uint32_t format, uint32_t rows, uint32_t cols, uint32_t tokens) {
    return exact_projection_tile_width(format, rows, cols, tokens);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_planar_q8_head_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    return result(planar_q8_head_metadata_valid(ordinal, stream, args, true)
        ? AUGUR_CUDA_STATUS_OK : AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_planar_q8_head_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    if (!planar_q8_head_metadata_valid(ordinal, stream, args, false))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    const auto* w = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight));
    const auto* a = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->input_q8_1));
    auto* y = reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32));
    if (args->tokens == 1U) planar_q8_head_kernel<1,true><<<args->rows,128,0,stream->stream>>>(w,a,y,1U);
    else if (args->tokens == 4U) planar_q8_head_kernel<4,true><<<args->rows,128,0,stream->stream>>>(w,a,y,4U);
    else if (args->tokens == 8U) planar_q8_head_kernel<8,true><<<args->rows,128,0,stream->stream>>>(w,a,y,8U);
    else planar_q8_head_kernel<8,false><<<args->rows,128,0,stream->stream>>>(w,a,y,args->tokens);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK,0) : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,e);
}

extern "C" uint32_t AUGUR_CUDA_CALL augur_cuda_qwen35_projection_row_reuse_width_v1(
    uint32_t format, uint32_t rows, uint32_t cols, uint32_t tokens) {
    return projection_row_reuse_width(format, rows, cols, tokens);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_projection_row_reuse_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    return quantized_small_batch_submit(ordinal, stream, plan, args, true, true);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_ffn_down_row_reuse_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    return quantized_small_batch_submit(ordinal, stream, plan, args, true, false, true);
}

namespace {
bool ffn_nvfp4_pair_metadata_valid(int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* gate,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* up, bool inspect) {
    Footprint g{}, u{};
    if (!projection_metadata_valid(ordinal, stream, plan, gate, inspect, &g)
        || !projection_metadata_valid(ordinal, stream, plan, up, inspect, &u)
        || gate->weight_format != AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1
        || up->weight_format != gate->weight_format
        || gate->rows != 17408U || up->rows != gate->rows
        || gate->cols != 5120U || up->cols != gate->cols
        || gate->tokens != 8U || up->tokens != gate->tokens
        || gate->input_q8_1 != up->input_q8_1
        || gate->input_len_bytes != up->input_len_bytes) return false;
    // Individual bindings cover scratch; also reject every cross-pair alias.
    const Range ranges[] = {{gate->weight,g.weight_bytes,4}, {up->weight,u.weight_bytes,4},
        {gate->input_q8_1,g.input_bytes,4}, {gate->output_f32,g.output_bytes,4},
        {up->output_f32,u.output_bytes,4}};
    return all_disjoint(ranges, 5);
}
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_ffn_nvfp4_pair_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* gate,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* up) {
    return result(ffn_nvfp4_pair_metadata_valid(ordinal,stream,plan,gate,up,true)
        ? AUGUR_CUDA_STATUS_OK : AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_ffn_nvfp4_pair_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* gate,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* up) {
    if (!ffn_nvfp4_pair_metadata_valid(ordinal,stream,plan,gate,up,false))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,selected);
    quantized_ffn_nvfp4_pair_b8_kernel<<<17408/2,128,0,stream->stream>>>(
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(gate->weight)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(up->weight)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(gate->input_q8_1)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(gate->output_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(up->output_f32)),
        gate->tensor_scale,up->tensor_scale);
    const cudaError_t error=cudaGetLastError();
    return error==cudaSuccess ? result(AUGUR_CUDA_STATUS_OK,0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_quantized_small_batch_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    return quantized_small_batch_submit(ordinal, stream, plan, args, false);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_exact_projection_tiles_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    return quantized_small_batch_submit(ordinal, stream, plan, args, true);
}

// Explicit diagnostic grid schedule only. Same per-token DP4A/FMA/reduction
// kernel as the exact B8 path, extended in grid.y without changing B8 admission.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_canonical_prefill_grid_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    Footprint footprint{};
    if (!projection_metadata_valid(ordinal, stream, plan, args, false, &footprint)
        || args->tokens > 512U
        || (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1
            && args->tensor_scale != 1.0f)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    const uint32_t tile = exact_projection_tile_width(
        args->weight_format, args->rows, args->cols, 8U);
    if (tile == 0U) return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const auto* w = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight));
    const auto* a = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->input_q8_1));
    auto* y = reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32));
    const dim3 grid(args->rows, (args->tokens + tile - 1U) / tile);
    if (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1) {
        if (tile == 4U) {
            if (args->tokens % 4U == 0U) quantized_small_batch_tiled_kernel<1,4,2,true>
                <<<grid,128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
            else quantized_small_batch_tiled_kernel<1,4,2,false>
                <<<grid,128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
        } else {
            if (args->tokens % 8U == 0U) quantized_small_batch_tiled_kernel<1,8,1,true>
                <<<grid,128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
            else quantized_small_batch_tiled_kernel<1,8,1,false>
                <<<grid,128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
        }
    } else {
        if (tile == 4U) {
            if (args->tokens % 4U == 0U) quantized_small_batch_tiled_kernel<2,4,2,true>
                <<<grid,128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
            else quantized_small_batch_tiled_kernel<2,4,2,false>
                <<<grid,128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
        } else {
            if (args->tokens % 8U == 0U) quantized_small_batch_tiled_kernel<2,8,1,true>
                <<<grid,128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
            else quantized_small_batch_tiled_kernel<2,8,1,false>
                <<<grid,128,0,stream->stream>>>(w,a,y,args->rows,args->cols,args->tokens,args->tensor_scale);
        }
    }
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK,0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_cublas_projection_profile_stage_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args,
    uint32_t stage) {
    Footprint footprint{};
    if (stage > 2U || !projection_metadata_valid(
            ordinal, stream, plan, args, false, &footprint)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    // Diagnostic caller must submit 0 -> 1 -> 2 without intervening scratch use.
    return launch(ordinal, stream, plan, args, footprint, false, stage);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_cublas_projection_pair_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    Footprint footprint{};
    if (!projection_metadata_valid(ordinal, stream, plan, args, false, &footprint)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    return launch(ordinal, stream, plan, args, footprint, false, 3U, true);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_cublas_projection_pair_profile_stage_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args, uint32_t stage) {
    Footprint footprint{};
    if (stage > 2U || !projection_metadata_valid(ordinal, stream, plan, args, false, &footprint)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    return launch(ordinal, stream, plan, args, footprint, false, stage, true);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_cublas_projection_quad_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    Footprint footprint{};
    if (!projection_metadata_valid(ordinal, stream, plan, args, false, &footprint)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    return launch(ordinal, stream, plan, args, footprint, false, 3U, false, true);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_cublas_projection_quad_profile_stage_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args, uint32_t stage) {
    Footprint footprint{};
    if (stage > 2U || !projection_metadata_valid(ordinal, stream, plan, args, false, &footprint)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    return launch(ordinal, stream, plan, args, footprint, false, stage, false, true);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_cublas_projection_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args) {
    Footprint footprint{};
    if (!projection_metadata_valid(
            ordinal, stream, plan, args, true, &footprint)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    const augur_cuda_result_v1 submitted = launch(
        ordinal, stream, plan, args, footprint);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    const cudaError_t error = cudaStreamSynchronize(stream->stream);
    return error == cudaSuccess
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, error);
}

namespace {
bool f32_projection_metadata_valid(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args,
    bool inspect_device, augur_cuda_qwen35_prefill_cublas_projection_args_v1* common,
    Footprint* footprint) {
    if (args == nullptr || args->abi_version
            != AUGUR_CUDA_QWEN35_PREFILL_F32_PROJECTION_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args) || args->reserved != 0) return false;
    // Translate fields explicitly; do not alias distinct C ABI structs.
    *common = {AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_PROJECTION_ARGS_VERSION_V1,
        sizeof(*common), args->weight_format, args->tokens, args->rows, args->cols,
        args->weight, args->weight_len_bytes, args->input_f32, args->input_len_bytes,
        args->output_f32, args->output_len_bytes, args->tensor_scale, 0};
    return projection_metadata_valid(
        ordinal, stream, plan, common, inspect_device, footprint, true);
}
}  // namespace

namespace {
bool original_input_metadata_valid(int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args,
    bool inspect_device, augur_cuda_qwen35_prefill_cublas_projection_args_v1* common,
    Footprint* footprint) {
    return f32_projection_metadata_valid(ordinal, stream, plan, args,
            inspect_device, common, footprint)
        && augur_cuda_qwen35_prefill_input_integer_scratch_bytes_v1(
            args->weight_format, args->rows, args->cols, args->tokens) != 0U
        && args->tensor_scale == 1.0f && args->weight % 4U == 0U
        && footprint->weight_elements % 2048U == 0U;
}
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_input_original_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args) {
    Footprint footprint{};
    augur_cuda_qwen35_prefill_cublas_projection_args_v1 common{};
    return original_input_metadata_valid(ordinal, stream, plan, args, true,
        &common, &footprint) ? result(AUGUR_CUDA_STATUS_OK, 0)
        : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_input_original_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args) {
    Footprint footprint{};
    augur_cuda_qwen35_prefill_cublas_projection_args_v1 common{};
    if (!original_input_metadata_valid(ordinal, stream, plan, args, false,
            &common, &footprint)) return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    return launch(ordinal, stream, plan, &common, footprint, true, 3U, false, false, true);
}

namespace {
bool llama_projection_metadata_valid(int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args,
    bool inspect_device, Footprint* footprint, bool exact_input_rows = false,
    bool dflash_projection = false) {
    augur_cuda_qwen35_prefill_cublas_projection_args_v1 common{};
    if (!f32_projection_metadata_valid(ordinal, stream, plan, args,
            inspect_device, &common, footprint)
        || (exact_input_rows ? (args->tokens < 1U || args->tokens > 512U)
                             : args->tokens != 512U)
        || args->weight % 4U != 0U
        || footprint->weight_elements % 2048U != 0U) return false;
    const bool q8 = args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1;
    const bool nv4 = args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1;
    const bool input = q8 && args->cols == 5120U && (args->rows == 10240U
        || args->rows == 6144U || args->rows == 12288U || args->rows == 1024U);
    const bool output = q8 && args->cols == 6144U && args->rows == 5120U;
    const bool ffn = (q8 || nv4) && ((args->cols == 5120U && args->rows == 17408U)
        || (args->cols == 17408U && args->rows == 5120U));
    const bool gdn_bf16 = args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1
        && args->rows == 48U && args->cols == 5120U && args->tensor_scale == 1.0f;
    if (exact_input_rows && args->tokens >= 9U && args->tokens <= 63U && (q8 || nv4)) {
        // MMQ reuses dead F16 decode buffers. Its activation scale is F32,
        // unlike MMVQ's half scale. Keep upstream trailing quantizer padding.
        const uint64_t packed = footprint->input_elements / 32U * 36U + 64U * 144U;
        const uint64_t fixup = uint64_t(plan->multiprocessors) * 128U * 64U * sizeof(float);
        if (plan->multiprocessors == 0U || args->rows % 128U || args->cols % 512U
            || plan->snapshot.decoded_input_len_bytes < packed
            || plan->snapshot.decoded_weight_len_bytes < fixup) return false;
    }
    // DFlash2 uses the SAME llama kernels, but a separate shape admission.
    // Do not broaden the target projection contract to admit draft-only matrices.
    if (dflash_projection) {
        const bool draft_shape = (args->cols == 5120U && (args->rows == 1024U
            || args->rows == 1280U || args->rows == 4096U || args->rows == 17408U
            || args->rows == 256U))
            || (args->rows == 5120U && (args->cols == 25600U
                || args->cols == 4096U || args->cols == 17408U));
        return exact_input_rows && q8 && args->tensor_scale == 1.0f && draft_shape;
    }
    return input || output || ffn || gdn_bf16;
}

// Actual llama SM70 alpha/beta path: round original F32 activations to BF16,
// then BF16/BF16 -> F32 GEMM. This is deliberately NOT the more accurate
// native BF16-weight/F32-input FP64 dot product. Original-input opt-in only.
__global__ void llama_gdn_bf16_input_kernel(const float* src, __nv_bfloat16* dst, uint64_t n) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += uint64_t(blockDim.x) * gridDim.x) dst[i] = __float2bfloat16_rn(src[i]);
}

// One fresh conversion and one pointer table per invocation. The table reuses
// dead decoded-weight scratch: resident BF16 weights are consumed directly.
// Same-stream completion makes every table/operand write visible to cuBLAS.
__global__ void llama_gdn_bf16_pair_input_kernel(const float* src,
    __nv_bfloat16* dst, uint64_t n, uint64_t* table,
    uint64_t wa, uint64_t wb, uint64_t ya, uint64_t yb) {
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
        table[0] = wa; table[1] = wb;
        table[2] = reinterpret_cast<uint64_t>(dst); table[3] = table[2];
        table[4] = ya; table[5] = yb;
    }
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += uint64_t(blockDim.x) * gridDim.x) dst[i] = __float2bfloat16_rn(src[i]);
}

augur_cuda_result_v1 llama_gdn_bf16_pair_submit(int32_t ordinal,
    augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args) {
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    auto* x = reinterpret_cast<const float*>(static_cast<uintptr_t>(args[0].input_f32));
    auto* xb = reinterpret_cast<__nv_bfloat16*>(static_cast<uintptr_t>(plan->snapshot.decoded_input_f16));
    auto* table = reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(plan->snapshot.decoded_weight_f16));
    const uint64_t count = uint64_t(args[0].tokens) * args[0].cols;
    llama_gdn_bf16_pair_input_kernel<<<launch_blocks(count), kThreads, 0, stream->stream>>>(
        x, xb, count, table, args[0].weight, args[1].weight, args[0].output_f32, args[1].output_f32);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    cublasMath_t previous;
    cublasStatus_t status = cublasGetMathMode(plan->handle, &previous);
    if (status != CUBLAS_STATUS_SUCCESS) return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, status);
    status = cublasSetMathMode(plan->handle, CUBLAS_TF32_TENSOR_OP_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, status);
    const float alpha = 1.0f, beta = 0.0f;
    status = cublasGemmBatchedEx(plan->handle, CUBLAS_OP_T, CUBLAS_OP_N,
        48, args[0].tokens, 5120, &alpha,
        reinterpret_cast<const void* const*>(table), CUDA_R_16BF, 5120,
        reinterpret_cast<const void* const*>(table + 2), CUDA_R_16BF, 5120,
        &beta, reinterpret_cast<void* const*>(table + 4), CUDA_R_32F, 48, 2,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    const cublasStatus_t restored = cublasSetMathMode(plan->handle, previous);
    if (status != CUBLAS_STATUS_SUCCESS) return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, status);
    return restored == CUBLAS_STATUS_SUCCESS ? result(AUGUR_CUDA_STATUS_OK, 0)
        : blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, restored);
}

augur_cuda_result_v1 llama_gdn_bf16_submit(augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args, const Footprint& footprint) {
    auto* x = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input_f32));
    auto* xb = reinterpret_cast<__nv_bfloat16*>(static_cast<uintptr_t>(plan->snapshot.decoded_input_f16));
    llama_gdn_bf16_input_kernel<<<launch_blocks(footprint.input_elements), kThreads, 0, stream->stream>>>(
        x, xb, footprint.input_elements);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    cublasMath_t previous_math;
    cublasStatus_t status = cublasGetMathMode(plan->handle, &previous_math);
    if (status != CUBLAS_STATUS_SUCCESS) return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, status);
    status = cublasSetMathMode(plan->handle, CUBLAS_TF32_TENSOR_OP_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, status);
    const float alpha = 1.0f, beta = 0.0f;
    status = cublasGemmEx(plan->handle, CUBLAS_OP_T, CUBLAS_OP_N,
        args->rows, args->tokens, args->cols, &alpha,
        reinterpret_cast<const void*>(static_cast<uintptr_t>(args->weight)), CUDA_R_16BF, args->cols,
        xb, CUDA_R_16BF, args->cols, &beta,
        reinterpret_cast<void*>(static_cast<uintptr_t>(args->output_f32)), CUDA_R_32F, args->rows,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    // Shared handle: restore on success AND rejected submission. No allocation,
    // upload or stream fence is introduced in this submit path.
    const cublasStatus_t restored = cublasSetMathMode(plan->handle, previous_math);
    if (status != CUBLAS_STATUS_SUCCESS) return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, status);
    return restored == CUBLAS_STATUS_SUCCESS ? result(AUGUR_CUDA_STATUS_OK, 0)
        : blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, restored);
}

// One lane pair per NVFP4 subblock. Decode only the block scale; tensor scale
// stays after GEMM in FP32. E2M1 here is the true half-step representation.
__global__ void decode_nvfp4_llama_unscaled_kernel(const uint8_t* source, __half* dest) {
    const uint64_t block = blockIdx.x;
    const uint32_t sub = threadIdx.x / 8U, j = threadIdx.x % 8U;
    const uint8_t* p = source + block * 36U;
    const uint8_t code = p[4U + sub * 8U + j];
    const float scale = ue4m3_scale_to_f32(p[sub]);
    dest[block * 64U + sub * 16U + j] = __float2half_rn(scale * e2m1_value(code & 15U));
    dest[block * 64U + sub * 16U + j + 8U] = __float2half_rn(scale * e2m1_value(code >> 4U));
}

#include "augur_qwen35_nvfp4_grouped_decode.cuh"

__global__ void llama_projection_postscale_kernel(float* output, uint64_t count, float scale) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += uint64_t(blockDim.x) * gridDim.x) output[i] *= scale;
}

bool llama_swiglu_metadata_valid(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_f32_args_v1* args, bool inspect_device,
    bool exact_input_rows = false) {
    if (args == nullptr) return false;
    const uint32_t count = args->elements;
    if (exact_input_rows ? (count % 17408U != 0U || count / 17408U < 1U
                           || count / 17408U > 512U)
                         : count != 512U * 17408U) return false;
    const uint64_t bytes = uint64_t(count) * sizeof(float);
    if (ordinal < 0 || stream == nullptr || stream->stream == nullptr
        || stream->ordinal != ordinal || args == nullptr
        || args->abi_version != AUGUR_CUDA_SILU_MUL_F32_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args) || args->reserved != 0U
        || args->elements != count || args->gate_len_bytes != bytes
        || args->up_len_bytes != bytes || args->output_len_bytes != bytes) return false;
    const Range ranges[] = {{args->gate, bytes, 4}, {args->up, bytes, 4}, {args->output, bytes, 4}};
    if (!all_disjoint(ranges, 3)) return false;
    if (!inspect_device) return true;
    if (!sm70_device(ordinal)) return false;
    for (auto range : ranges) if (!device_range(range, ordinal)) return false;
    return true;
}
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_llama_projection_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args) {
    Footprint footprint{};
    return llama_projection_metadata_valid(ordinal, stream, plan, args, true, &footprint)
        ? result(AUGUR_CUDA_STATUS_OK, 0) : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

namespace {
// Shared by the common projection lane and the QUASAR QPN adapter.
// Identical FP16 operands, FP32 Tensor Core accumulation and F32 postscale.
augur_cuda_result_v1 llama_f16_gemm_postscale(augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan, uint32_t rows,uint32_t cols,uint32_t tokens,
    const __half* weight,const __half* input,float* output,float scale,bool postscale,
    bool fused_postscale = false) {
    const float alpha=fused_postscale ? scale : 1.0f,beta=0.0f;
    const auto status=cublasGemmEx(plan->handle,CUBLAS_OP_T,CUBLAS_OP_N,
        rows,tokens,cols,&alpha,weight,CUDA_R_16F,cols,input,CUDA_R_16F,cols,
        &beta,output,CUDA_R_32F,rows,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if(status!=CUBLAS_STATUS_SUCCESS)return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,status);
    const uint64_t count=uint64_t(rows)*tokens;
    if(postscale && !fused_postscale)llama_projection_postscale_kernel<<<launch_blocks(count),kThreads,0,stream->stream>>>(output,count,scale);
    const auto error=cudaGetLastError();
    return error==cudaSuccess?result(AUGUR_CUDA_STATUS_OK,0):cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
}

augur_cuda_result_v1 llama_projection_submit_impl(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args, bool exact_input_rows,
    bool input_already_packed = false, bool dflash_projection = false) {
    Footprint footprint{};
    if (!llama_projection_metadata_valid(ordinal, stream, plan, args, false,
            &footprint, exact_input_rows, dflash_projection))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    if (exact_input_rows && args->tokens < 64U) {
        const void* w = reinterpret_cast<const void*>(static_cast<uintptr_t>(args->weight));
        const float* x = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input_f32));
        float* y = reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32));
        void* packed = reinterpret_cast<void*>(static_cast<uintptr_t>(plan->snapshot.decoded_input_f16));
        cudaError_t error;
        if (args->tokens <= 8U) {
            error = dflash_projection
                ? llama_small::launch_dflash_q8(w, x, packed, y, args->tokens, args->rows,args->cols,stream->stream)
                : llama_small::launch(w, x, packed, y, args->tokens, args->rows,
                    args->cols, args->weight_format, args->tensor_scale, stream->stream);
        } else if (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1) {
            return llama_gdn_bf16_submit(stream, plan, args, footprint);
        } else {
            float* fixup = reinterpret_cast<float*>(static_cast<uintptr_t>(plan->snapshot.decoded_weight_f16));
            error = augur_qwen35_launch_llama_mmq_sm70(w, x, packed, y, fixup,
                args->tokens, args->rows, args->cols, args->weight_format,
                args->tensor_scale, plan->multiprocessors, stream->stream);
        }
        return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, 0)
            : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    }
    if (args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1)
        return llama_gdn_bf16_submit(stream, plan, args, footprint);
    auto* w = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight));
    auto* x = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input_f32));
    auto* y = reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32));
    auto* wh = reinterpret_cast<__half*>(static_cast<uintptr_t>(plan->snapshot.decoded_weight_f16));
    auto* xh = reinterpret_cast<__half*>(static_cast<uintptr_t>(plan->snapshot.decoded_input_f16));
    const bool nv4 = args->weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1;
    if (nv4 && footprint.weight_elements % 2048U == 0U)
        decode_nvfp4_llama_grouped_kernel<128><<<
            static_cast<uint32_t>(footprint.weight_elements / 2048U), 128, 0, stream->stream>>>(w, wh);
    else if (nv4) decode_nvfp4_llama_unscaled_kernel<<<
        static_cast<uint32_t>(footprint.weight_elements / 64U), 32, 0, stream->stream>>>(w, wh);
    // Same half2 multiply/rounding as the scalar CTA schedule. The small-row
    // projection branches return above and retain their established kernels.
    else decode_q8_0_llama_coalesced_kernel<256><<<
        static_cast<uint32_t>(footprint.weight_elements / 2048U), 256, 0, stream->stream>>>(w, wh);
    if (!input_already_packed)
        pack_f32_activation_kernel<<<launch_blocks(footprint.input_elements), kThreads, 0, stream->stream>>>(
            x, xh, footprint.input_elements);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return llama_f16_gemm_postscale(stream,plan,args->rows,args->cols,args->tokens,
        wh,xh,y,args->tensor_scale,nv4);
}

} // namespace

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_llama_projection_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args) {
    return llama_projection_submit_impl(ordinal, stream, plan, args, false);
}

// Input-only GEMM family: the hash-bound SM70 DLL changes quantized dispatch
// below 64 rows. Small tails must not silently inherit this arithmetic.
// Draft-only entry points: exact same upstream-derived MMVQ/MMQ/FP16 kernels.
// Serial stream and shared plan scratch follow the existing owner contract.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_dflash_llama_projection_preflight_v1(int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args) {
    Footprint footprint{};
    return llama_projection_metadata_valid(ordinal, stream, plan, args, true,
        &footprint, true, true) ? result(AUGUR_CUDA_STATUS_OK, 0)
        : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_dflash_llama_projection_submit_v1(int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args) {
    return llama_projection_submit_impl(ordinal, stream, plan, args, true, false, true);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_llama_projection_rows_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args) {
    Footprint footprint{};
    return llama_projection_metadata_valid(ordinal, stream, plan, args, true, &footprint, true)
        ? result(AUGUR_CUDA_STATUS_OK, 0) : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_llama_projection_rows_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args) {
    return llama_projection_submit_impl(ordinal, stream, plan, args, true);
}

// One bounded call is the complete reuse lifetime. No pointer-keyed cache or
// reuse across calls: the first projection always prepares current input bytes.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_llama_projection_group_rows_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args, uint32_t count) {
    if (args == nullptr || count < 2U || count > 3U)
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const bool bf16_pair = args[0].weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1;
    // Validate the entire group before any submission, including cross-output
    // and cross-weight aliasing not covered by individual projection admission.
    for (uint32_t i = 0; i < count; ++i) {
        Footprint footprint{};
        if (!llama_projection_metadata_valid(ordinal, stream, plan, args + i,
                false, &footprint, true)
            || args[i].tokens < 64U
            || (args[i].weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1) != bf16_pair
            || args[i].tokens != args[0].tokens || args[i].cols != args[0].cols
            || args[i].input_f32 != args[0].input_f32
            || args[i].input_len_bytes != args[0].input_len_bytes)
            return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    if (bf16_pair && (count != 2U || plan->snapshot.decoded_weight_len_bytes < 6U * sizeof(uint64_t)
            || plan->snapshot.decoded_weight_f16 % alignof(uint64_t) != 0U))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    for (uint32_t i = 0; i < count; ++i) {
        const Range output{args[i].output_f32, args[i].output_len_bytes, 4};
        for (uint32_t j = 0; j < count; ++j) {
            if (i != j && (ranges_overlap(output,
                    {args[j].output_f32, args[j].output_len_bytes, 4})
                || ranges_overlap(output, {args[j].weight, args[j].weight_len_bytes, 2})))
                return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
        }
    }
    if (bf16_pair) return llama_gdn_bf16_pair_submit(ordinal, stream, plan, args);
    for (uint32_t i = 0; i < count; ++i) {
        const auto status = llama_projection_submit_impl(
            ordinal, stream, plan, args + i, true, i != 0U);
        if (status.status != AUGUR_CUDA_STATUS_OK) return status;
    }
    return result(AUGUR_CUDA_STATUS_OK, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_llama_swiglu_preflight_v1(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_f32_args_v1* args) {
    return llama_swiglu_metadata_valid(ordinal, stream, args, true)
        ? result(AUGUR_CUDA_STATUS_OK, 0) : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

namespace {
augur_cuda_result_v1 llama_swiglu_submit_impl(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_f32_args_v1* args, bool exact_input_rows) {
    if (!llama_swiglu_metadata_valid(ordinal, stream, args, false, exact_input_rows))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    const cudaError_t error = augur_qwen35_launch_llama_swiglu_sm70(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->up)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)), args->elements, stream->stream);
    return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

} // namespace

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_llama_swiglu_prevalidated_submit_v1(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_f32_args_v1* args) {
    return llama_swiglu_submit_impl(ordinal, stream, args, false);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_llama_swiglu_rows_preflight_v1(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_f32_args_v1* args) {
    return llama_swiglu_metadata_valid(ordinal, stream, args, true, true)
        ? result(AUGUR_CUDA_STATUS_OK, 0) : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_llama_swiglu_rows_prevalidated_submit_v1(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_f32_args_v1* args) {
    return llama_swiglu_submit_impl(ordinal, stream, args, true);
}

#if defined(AUGUR_CUDA_QWEN35_TC_PREFIX_ATTENTION)
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_f32_projection_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args) {
    Footprint footprint{};
    augur_cuda_qwen35_prefill_cublas_projection_args_v1 common{};
    return f32_projection_metadata_valid(
        ordinal, stream, plan, args, true, &common, &footprint)
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_f32_projection_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args) {
    Footprint footprint{};
    augur_cuda_qwen35_prefill_cublas_projection_args_v1 common{};
    if (!f32_projection_metadata_valid(
            ordinal, stream, plan, args, false, &common, &footprint)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    return launch(ordinal, stream, plan, &common, footprint, true);
}
#endif

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_rmsnorm_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_rmsnorm_args_v1* args) {
    return rmsnorm_metadata_valid(ordinal, stream, args, true)
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_rmsnorm_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_rmsnorm_args_v1* args) {
    return submit_rmsnorm(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_llama_input_rmsnorm_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_rmsnorm_args_v1* args) {
    const float epsilon = 1e-6f;
    uint32_t epsilon_bits = 0;
    std::memcpy(&epsilon_bits, &epsilon, sizeof(epsilon));
    if (!rmsnorm_metadata_valid(ordinal, stream, args, false)
        || args->tokens > 512 || args->cols != 5120
        || args->epsilon_bits != epsilon_bits) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    }
    const cudaError_t error = augur_qwen35_launch_llama_input_rmsnorm5120_sm70(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->weight_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        args->tokens, stream->stream);
    return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, 0)
        : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_linear_chunk_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args) {
    return linear_chunk_metadata_valid(ordinal, stream, args, true)
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_linear_chunk_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args) {
    return submit_linear_chunk<false>(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_linear_chunk_fold_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args) {
    return submit_linear_chunk<false, true>(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_linear_chunk_fold_shared_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args) {
    return submit_linear_chunk_fold_shared(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_linear_checkpoint_fold_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args,
    uint64_t history, uint64_t history_bytes, uint64_t state, uint64_t state_bytes) {
    return linear_checkpoint_metadata_valid(ordinal, stream, args,
        history, history_bytes, state, state_bytes, true)
        ? result(AUGUR_CUDA_STATUS_OK, 0) : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_linear_checkpoint_fold_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args,
    uint64_t history, uint64_t history_bytes, uint64_t state, uint64_t state_bytes) {
    return submit_linear_checkpoint_fold(ordinal, stream, args,
        history, history_bytes, state, state_bytes);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_linear_chunk_shared_scalars_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args) {
    return submit_linear_chunk<true>(ordinal, stream, args);
}

// Authenticated safetensors order is contiguous groups of three value heads,
// with additive 1e-6 Q/K normalization. Never alias this to legacy interleaving.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_checkpoint_prefill_linear_chunk_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args) {
    return submit_linear_chunk<true, false, true>(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_linear_chunk_precise_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args) {
    return submit_linear_chunk_precise(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_full_attention_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_full_attention_args_v1* args) {
    return full_attention_metadata_valid(ordinal, stream, args, true)
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_full_attention_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_full_attention_args_v1* args) {
    return submit_full_attention(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nerkyor_prefill_full_attention_canonical_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_full_attention_args_v1* args) {
#if defined(AUGUR_CUDA_QWEN35_WARP_FULL_ATTENTION) && defined(AUGUR_CUDA_QWEN35_DIAGNOSTIC_SPLIT_ROPE)
    return submit_nerkyor_canonical_full_attention(ordinal, stream, args);
#else
    return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
#endif
}

#if defined(AUGUR_CUDA_QWEN35_TC_PREFIX_ATTENTION)
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_prepare_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args) {
    return submit_prefill_kv8_chunk_prepare(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_gate_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args) {
    return submit_prefill_kv8_chunk_gate(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_tc_prefix_attention_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_tc_prefix_attention_args_v1* args) {
    TcPrefixFootprint footprint{};
    return tc_prefix_metadata_valid(
        ordinal, stream, plan, args, true, &footprint)
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_tc_prefix_attention_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_tc_prefix_attention_args_v1* args) {
    return submit_tc_prefix_attention(ordinal, stream, plan, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_simt_prefix_attention_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_tc_prefix_attention_args_v1* args) {
    return submit_simt_prefix_attention(ordinal, stream, plan, args);
}
#endif

#if defined(AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION)
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_attention_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args) {
    return prefill_kv8_chunk_attention_metadata_valid(
        ordinal, stream, args, true)
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_attention_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args) {
    return submit_prefill_kv8_chunk_attention(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_shared_softmax_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args) {
    return submit_prefill_kv8_chunk_attention<true>(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_register_softmax_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args) {
    return submit_prefill_kv8_chunk_attention<true, 8U, false, true>(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_warp_ordered_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args) {
    return submit_prefill_kv8_chunk_attention<true, 8U, false, false, true>(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_warp_shared_kv_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args) {
    return submit_prefill_kv8_chunk_attention<true, 8U, false, false, false, true>(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_parallel_scales_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args) {
    return submit_prefill_kv8_chunk_attention<true, 8U, false, false, false, false, true>(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_tiled_dgemm_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes) {
    return dgemm_scores_metadata_valid(ordinal, stream, plan, args, score_address, score_bytes, true)
        ? result(AUGUR_CUDA_STATUS_OK, 0) : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_tiled_dgemm_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes) {
    return submit_tiled_dgemm(ordinal, stream, plan, args, score_address, score_bytes);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_canonical_prefill_scores_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_canonical_prefill_args_v1* args,
    uint64_t address, uint64_t bytes, uint32_t policy) {
    return canonical_scores_metadata_valid(ordinal, stream, plan, args, address, bytes, policy, true)
        ? result(AUGUR_CUDA_STATUS_OK, 0) : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_canonical_prefill_scores_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_canonical_prefill_args_v1* args,
    uint64_t address, uint64_t bytes, uint32_t policy) {
    return submit_canonical_scores(ordinal, stream, plan, args, address, bytes, policy);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_tiled_scores_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes) {
    return staged_scores_metadata_valid(ordinal, stream, args, score_address, score_bytes, true, true)
        ? result(AUGUR_CUDA_STATUS_OK, 0) : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_tiled_scores_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes) {
    return submit_tiled_scores(ordinal, stream, args, score_address, score_bytes);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_tiled_cached_query_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes) {
    return submit_tiled_scores<true>(ordinal, stream, args, score_address, score_bytes);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_staged_scores_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes) {
    return staged_scores_metadata_valid(ordinal, stream, args, score_address, score_bytes, true)
        ? result(AUGUR_CUDA_STATUS_OK, 0) : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_staged_scores_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes) {
    return submit_staged_scores(ordinal, stream, args, score_address, score_bytes);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_shared_softmax_unit_scale_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args) {
    return submit_prefill_kv8_chunk_attention<true, 8U, true>(ordinal, stream, args);
}

#if defined(AUGUR_CUDA_QWEN35_KV8_TOKEN_PAIR)
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_token_pair_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args) {
    return submit_prefill_kv8_chunk_token_pair(ordinal, stream, args);
}
#endif

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_shared_softmax_tile16_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args) {
    return submit_prefill_kv8_chunk_attention<true, 16U>(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_cache_quantize_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_cache_quantize_args_v1* args) {
    return prefill_kv8_cache_quantize_metadata_valid(
        ordinal, stream, args, true)
        ? result(AUGUR_CUDA_STATUS_OK, 0)
        : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_cache_quantize_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_cache_quantize_args_v1* args) {
    return submit_prefill_kv8_cache_quantize(ordinal, stream, args);
}
#endif

extern "C" void AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_cublas_plan_destroy_v1(
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan) {
    if (plan == nullptr) return;
    cudaSetDevice(plan->ordinal);
    if (plan->handle != nullptr) cublasDestroy(plan->handle);
    delete plan;
}

#if defined(AUGUR_CUDA_QWEN35_VISION_AFFINE)
#include "augur_qwen35_vision_affine.cuh"
#include "augur_qwen35_vision_control.cuh"
#endif

#include "augur_qwen35_row_q8_input.cuh"
#include "augur_qwen35_qpn_prefill.cuh"
#include "augur_qwen35_gdn_input_b8.cuh"




#include "augur_qwen35_augur_prefill.cuh"

// File-scope inclusion avoids nested anonymous namespaces in NVCC/MSVC stubs.
#include "augur_dflash_llama_bf16_head.cuh"
#include "augur_dflash_q8_b8_direct.cuh"
#include "augur_dflash_selector_gate_f32.cuh"
