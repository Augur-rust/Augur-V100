#include "augur_cuda_internal.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr uint32_t kBlockElements = 64;
constexpr uint32_t kBlockBytes = 36;
constexpr uint32_t kPackedBytes = 32;
constexpr uint32_t kWarpsPerBlock = 4;
constexpr uint32_t kQ8ActivationBlockElements = 32;
constexpr uint32_t kQ8ActivationBlockBytes = 36;

static_assert(sizeof(augur_cuda_nvfp4_sm70_q8_1_matvec_args_v1) == 72,
    "augur_cuda_nvfp4_sm70_q8_1_matvec_args_v1 ABI size changed");
static_assert(alignof(augur_cuda_nvfp4_sm70_q8_1_matvec_args_v1) == 8,
    "augur_cuda_nvfp4_sm70_q8_1_matvec_args_v1 ABI alignment changed");
#if defined(AUGUR_CUDA_QWEN35_FUSED_FFN_SWIGLU)
static_assert(sizeof(augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_args_v1) == 88,
    "augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_args_v1 ABI size changed");
static_assert(alignof(augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_args_v1) == 8,
    "augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_args_v1 ABI alignment changed");
#endif

__device__ __constant__ __align__(16) int8_t kE2m1DoubledValues[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

inline augur_cuda_result_v1 nvfp4_ok() {
    return {AUGUR_CUDA_STATUS_OK, static_cast<int32_t>(cudaSuccess)};
}

inline augur_cuda_result_v1 nvfp4_error(augur_cuda_status_t status, cudaError_t error) {
    return {status, static_cast<int32_t>(error)};
}

inline augur_cuda_result_v1 nvfp4_set_device(int32_t ordinal) {
    const cudaError_t error = cudaSetDevice(ordinal);
    return error == cudaSuccess
        ? nvfp4_ok()
        : nvfp4_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, error);
}

bool checked_footprint(
    uint32_t rows,
    uint32_t cols,
    uint64_t* weight_bytes,
    uint64_t* input_bytes,
    uint64_t* output_bytes
) {
    if (weight_bytes == nullptr || input_bytes == nullptr || output_bytes == nullptr
        || rows == 0 || cols == 0 || (cols % kBlockElements) != 0) {
        return false;
    }
    const uint64_t blocks_per_row = static_cast<uint64_t>(cols) / kBlockElements;
    if (blocks_per_row > std::numeric_limits<uint64_t>::max() / kBlockBytes) return false;
    const uint64_t row_bytes = blocks_per_row * kBlockBytes;
    if (static_cast<uint64_t>(rows) > std::numeric_limits<uint64_t>::max() / row_bytes) {
        return false;
    }
    *weight_bytes = static_cast<uint64_t>(rows) * row_bytes;
    if (static_cast<uint64_t>(cols) > std::numeric_limits<uint64_t>::max() / sizeof(float)
        || static_cast<uint64_t>(rows) > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        return false;
    }
    *input_bytes = static_cast<uint64_t>(cols) * sizeof(float);
    *output_bytes = static_cast<uint64_t>(rows) * sizeof(float);
    return *weight_bytes <= SIZE_MAX && *input_bytes <= SIZE_MAX && *output_bytes <= SIZE_MAX;
}

bool checked_range(uint64_t address, uint64_t extent) {
    return address != 0 && extent != 0 && address <= UINT64_MAX - extent;
}

bool ranges_disjoint(uint64_t left, uint64_t left_len, uint64_t right, uint64_t right_len) {
    return left + left_len <= right || right + right_len <= left;
}

__device__ __forceinline__ float e2m1_value(uint8_t nibble) {
    const uint8_t magnitude = nibble & 0x07U;
    // Twice the ordinary E2M1 value is an exact small integer. Constructing
    // it arithmetically avoids a local/constant-memory lookup in the hot loop.
    const uint32_t doubled = magnitude < 4U
        ? magnitude
        : (2U + (magnitude & 1U)) << ((magnitude >> 1U) - 1U);
    const float value = static_cast<float>(doubled) * 0.5f;
    return (nibble & 0x08U) != 0 ? -value : value;
}

__device__ __forceinline__ float ue4m3_scale_to_f32(uint8_t value) {
    // This is ggml_ue4m3_to_fp32 before llama.cpp's final * 0.5.
    if (value == 0 || value == 0x7f) return 0.0f;
    const uint32_t exponent = (static_cast<uint32_t>(value) >> 3) & 0x0fU;
    const uint32_t mantissa = static_cast<uint32_t>(value) & 0x07U;
    if (exponent == 0) {
        return static_cast<float>(mantissa) * 0x1p-9f;
    }
    // Every normal UE4M3 value is exactly representable as f32. Build those
    // bits directly instead of issuing an exp2f for every participating lane.
    return __uint_as_float(((exponent + 120U) << 23U) | (mantissa << 20U));
}

__device__ __forceinline__ float q8_1_f16_to_f32(uint16_t bits) {
#if defined(AUGUR_CUDA_QWEN35_NATIVE_F16_SCALE)
    return __half2float(__ushort_as_half(bits));
#else
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
    const uint32_t exponent = (bits >> 10U) & 0x1fU;
    const uint32_t mantissa = bits & 0x03ffU;
    if (exponent == 0U) {
        if (mantissa == 0U) return __uint_as_float(sign);
        uint32_t normalized = mantissa;
        int32_t shift = 0;
        while ((normalized & 0x0400U) == 0U) {
            normalized <<= 1U;
            --shift;
        }
        normalized &= 0x03ffU;
        const uint32_t f32_exponent = static_cast<uint32_t>(127 - 15 + 1 + shift);
        return __uint_as_float(sign | (f32_exponent << 23U) | (normalized << 13U));
    }
    if (exponent == 0x1fU) {
        return __uint_as_float(sign | 0x7f800000U | (mantissa << 13U));
    }
    return __uint_as_float(sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
#endif
}

// Convert eight packed E2M1 nibbles to two signed-int8 words. This is the
// four-byte-permute sequence used by llama.cpp's CUDA NVFP4 dot product.
__device__ __forceinline__ int2 e2m1_doubled_words(uint32_t packed) {
    const auto* table = reinterpret_cast<const uint32_t*>(kE2m1DoubledValues);
    const uint32_t selection = 0x32103210U | ((packed & 0x88888888U) >> 1U);
    uint32_t temporary[2];
    #pragma unroll
    for (uint32_t index = 0; index < 2U; ++index) {
        const uint32_t shift = 16U * index;
        const uint32_t low = __byte_perm(table[0], table[1], packed >> shift);
        const uint32_t high = __byte_perm(table[2], table[3], packed >> shift);
        temporary[index] = __byte_perm(low, high, selection >> shift);
    }
    return make_int2(
        static_cast<int32_t>(__byte_perm(temporary[0], temporary[1], 0x6420U)),
        static_cast<int32_t>(__byte_perm(temporary[0], temporary[1], 0x7531U)));
}

#include "augur_nvfp4_small_gate.cuh"
#include "augur_nvfp4_small_gate_pair.cuh"
#include "augur_nvfp4_qpn_source_pair.cuh"

__global__ void nvfp4_sm70_f32_matvec_kernel(
    const uint8_t* weight,
    const float* input,
    float* output,
    uint32_t rows,
    uint32_t cols,
    float tensor_scale
) {
    const uint32_t warp = threadIdx.x / 32U;
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= rows) return;

    const uint32_t blocks_per_row = cols / kBlockElements;
    const uint64_t row_bytes = static_cast<uint64_t>(blocks_per_row) * kBlockBytes;
    const uint8_t* row_source = weight + static_cast<uint64_t>(row) * row_bytes;
    double partial = 0.0;

    for (uint32_t block = 0; block < blocks_per_row; ++block) {
        const uint8_t* block_source = row_source + static_cast<uint64_t>(block) * kBlockBytes;
        const uint32_t sub_block = lane / 8U;
        const uint32_t sub_lane = lane & 7U;
        const uint32_t input_base = block * kBlockElements + sub_block * 16U;
        const uint8_t packed = block_source[4U + lane];
        float scale = lane < 4U ? ue4m3_scale_to_f32(block_source[lane]) : 0.0f;
        scale = __shfl_sync(0xffffffffU, scale, sub_block);
        const float low = e2m1_value(packed & 0x0fU);
        const float high = e2m1_value(packed >> 4U);
        const float pair = low * input[input_base + sub_lane]
            + high * input[input_base + 8U + sub_lane];
        partial += static_cast<double>(pair) * static_cast<double>(scale);
    }

    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U) {
        partial += __shfl_down_sync(0xffffffffU, partial, offset);
    }
    if (lane == 0) {
        output[row] = static_cast<float>(partial * static_cast<double>(tensor_scale));
    }
}

// Same per-token pair, double accumulation and warp-reduction order as the
// scalar F32 activation reference. Only weight decode is shared across B rows.
// This is NOT Tensor Core, activation quantization or compensated FP32 math.
template<int B>
__global__ __launch_bounds__(128, 1) void nvfp4_sm70_f32_batch_kernel(
    const uint8_t* weight, const float* input, float* output,
    uint32_t rows, uint32_t cols, float tensor_scale
) {
    const uint32_t warp = threadIdx.x / 32U;
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= rows) return;
    const uint32_t blocks_per_row = cols / kBlockElements;
    const uint64_t row_bytes = static_cast<uint64_t>(blocks_per_row) * kBlockBytes;
    const uint8_t* row_source = weight + static_cast<uint64_t>(row) * row_bytes;
    double partial[B] = {};
    for (uint32_t block = 0; block < blocks_per_row; ++block) {
        const uint8_t* block_source = row_source + static_cast<uint64_t>(block) * kBlockBytes;
        const uint32_t sub_block = lane / 8U;
        const uint32_t sub_lane = lane & 7U;
        const uint32_t input_base = block * kBlockElements + sub_block * 16U;
        const uint8_t packed = block_source[4U + lane];
        float scale = lane < 4U ? ue4m3_scale_to_f32(block_source[lane]) : 0.0f;
        scale = __shfl_sync(0xffffffffU, scale, sub_block);
        const float low = e2m1_value(packed & 0x0fU);
        const float high = e2m1_value(packed >> 4U);
        #pragma unroll
        for (int t = 0; t < B; ++t) {
            const float* x = input + static_cast<uint64_t>(t) * cols;
            const float pair = low * x[input_base + sub_lane]
                + high * x[input_base + 8U + sub_lane];
            partial[t] += static_cast<double>(pair) * static_cast<double>(scale);
        }
    }
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U) {
        #pragma unroll
        for (int t = 0; t < B; ++t)
            partial[t] += __shfl_down_sync(0xffffffffU, partial[t], offset);
    }
    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < B; ++t)
            output[static_cast<uint64_t>(t) * rows + row] =
                static_cast<float>(partial[t] * static_cast<double>(tensor_scale));
    }
}

__global__ __launch_bounds__(128, 1) void nvfp4_sm70_q8_1_matvec_kernel(
    const uint8_t* weight,
    const uint8_t* activation,
    float* output,
    uint32_t rows,
    uint32_t cols,
    float tensor_scale
) {
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x / 32U;
    const uint32_t thread = threadIdx.x;
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;

    const uint32_t blocks_per_row = cols / kBlockElements;
    const uint64_t row_bytes = static_cast<uint64_t>(blocks_per_row) * kBlockBytes;
    const uint8_t* row_source = weight + static_cast<uint64_t>(row) * row_bytes;
    float partial = 0.0f;

    // Two threads split one 64-value NVFP4 block. Four warps therefore cover
    // 64 blocks per iteration, matching llama.cpp's Volta MMVQ decomposition.
    const uint32_t half = thread & 1U;
    for (uint32_t block = thread / 2U; block < blocks_per_row; block += 64U) {
        const uint8_t* weight_block = row_source
            + static_cast<uint64_t>(block) * kBlockBytes;
        const auto* packed_words = reinterpret_cast<const uint32_t*>(weight_block + 4U);
        const uint32_t word_base = half * 4U;
        #pragma unroll
        for (uint32_t pair = 0; pair < 2U; ++pair) {
            const uint32_t first_word = word_base + pair * 2U;
            const uint32_t second_word = first_word + 1U;
            const uint32_t sub_block = first_word / 2U;
            const int2 values0 = e2m1_doubled_words(packed_words[first_word]);
            const int2 values1 = e2m1_doubled_words(packed_words[second_word]);
            const uint32_t activation_block_index = block * 2U + sub_block / 2U;
            const uint8_t* activation_block = activation
                + static_cast<uint64_t>(activation_block_index) * kQ8ActivationBlockBytes;
            const auto* activation_words =
                reinterpret_cast<const int32_t*>(activation_block + 4U);
            const uint32_t input_word = (sub_block & 1U) * 4U;
            int32_t dot = __dp4a(values0.x, activation_words[input_word], 0);
            dot = __dp4a(values0.y, activation_words[input_word + 2U], dot);
            dot = __dp4a(values1.x, activation_words[input_word + 1U], dot);
            dot = __dp4a(values1.y, activation_words[input_word + 3U], dot);
            const uint16_t activation_scale_bits =
                static_cast<uint16_t>(activation_block[0])
                | static_cast<uint16_t>(activation_block[1]) << 8U;
            const float combined_scale = ue4m3_scale_to_f32(weight_block[sub_block])
                * 0.5f * q8_1_f16_to_f32(activation_scale_bits);
            partial = fmaf(combined_scale, static_cast<float>(dot), partial);
        }
    }

    __shared__ float other_warps[kWarpsPerBlock - 1U][32];
    if (warp > 0U) other_warps[warp - 1U][lane] = partial;
    __syncthreads();
    if (warp > 0U) return;
    #pragma unroll
    for (uint32_t other = 0; other < kWarpsPerBlock - 1U; ++other) {
        partial += other_warps[other][lane];
    }
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
        partial += __shfl_down_sync(0xffffffffU, partial, offset);
    }
    if (lane == 0U) output[row] = partial * tensor_scale;
}

#if defined(AUGUR_CUDA_QWEN35_FUSED_FFN_SWIGLU)
#if defined(AUGUR_CUDA_QWEN35_NVFP4_FUSED_LB10)
__global__ __launch_bounds__(128, 10) void nvfp4_sm70_q8_1_fused_swiglu_kernel(
#else
__global__ __launch_bounds__(128, 1) void nvfp4_sm70_q8_1_fused_swiglu_kernel(
#endif
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    const uint8_t* activation,
    float* output,
    uint32_t rows,
    uint32_t cols,
    float gate_tensor_scale,
    float up_tensor_scale
) {
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x / 32U;
    const uint32_t thread = threadIdx.x;
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;

    const uint32_t blocks_per_row = cols / kBlockElements;
    const uint64_t row_bytes = static_cast<uint64_t>(blocks_per_row) * kBlockBytes;
    const uint8_t* gate_row = gate_weight + static_cast<uint64_t>(row) * row_bytes;
    const uint8_t* up_row = up_weight + static_cast<uint64_t>(row) * row_bytes;
    float gate_partial = 0.0f;
    float up_partial = 0.0f;

    const uint32_t half = thread & 1U;
    for (uint32_t block = thread / 2U; block < blocks_per_row; block += 64U) {
        const uint8_t* gate_block = gate_row + static_cast<uint64_t>(block) * kBlockBytes;
        const uint8_t* up_block = up_row + static_cast<uint64_t>(block) * kBlockBytes;
        const auto* gate_words = reinterpret_cast<const uint32_t*>(gate_block + 4U);
        const auto* up_words = reinterpret_cast<const uint32_t*>(up_block + 4U);
        const uint32_t word_base = half * 4U;
        #pragma unroll
        for (uint32_t pair = 0; pair < 2U; ++pair) {
            const uint32_t first_word = word_base + pair * 2U;
            const uint32_t second_word = first_word + 1U;
            const uint32_t sub_block = first_word / 2U;
            const int2 gate0 = e2m1_doubled_words(gate_words[first_word]);
            const int2 gate1 = e2m1_doubled_words(gate_words[second_word]);
            const int2 up0 = e2m1_doubled_words(up_words[first_word]);
            const int2 up1 = e2m1_doubled_words(up_words[second_word]);
            const uint32_t activation_block_index = block * 2U + sub_block / 2U;
            const uint8_t* activation_block = activation
                + static_cast<uint64_t>(activation_block_index) * kQ8ActivationBlockBytes;
            const auto* activation_words =
                reinterpret_cast<const int32_t*>(activation_block + 4U);
            const uint32_t input_word = (sub_block & 1U) * 4U;
            int32_t gate_dot = __dp4a(gate0.x, activation_words[input_word], 0);
            gate_dot = __dp4a(gate0.y, activation_words[input_word + 2U], gate_dot);
            gate_dot = __dp4a(gate1.x, activation_words[input_word + 1U], gate_dot);
            gate_dot = __dp4a(gate1.y, activation_words[input_word + 3U], gate_dot);
            int32_t up_dot = __dp4a(up0.x, activation_words[input_word], 0);
            up_dot = __dp4a(up0.y, activation_words[input_word + 2U], up_dot);
            up_dot = __dp4a(up1.x, activation_words[input_word + 1U], up_dot);
            up_dot = __dp4a(up1.y, activation_words[input_word + 3U], up_dot);
            const uint16_t activation_scale_bits =
                static_cast<uint16_t>(activation_block[0])
                | static_cast<uint16_t>(activation_block[1]) << 8U;
            const float activation_scale = q8_1_f16_to_f32(activation_scale_bits) * 0.5f;
            gate_partial = fmaf(
                ue4m3_scale_to_f32(gate_block[sub_block]) * activation_scale,
                static_cast<float>(gate_dot),
                gate_partial);
            up_partial = fmaf(
                ue4m3_scale_to_f32(up_block[sub_block]) * activation_scale,
                static_cast<float>(up_dot),
                up_partial);
        }
    }

    __shared__ float gate_other_warps[kWarpsPerBlock - 1U][32];
    __shared__ float up_other_warps[kWarpsPerBlock - 1U][32];
    if (warp > 0U) {
        gate_other_warps[warp - 1U][lane] = gate_partial;
        up_other_warps[warp - 1U][lane] = up_partial;
    }
    __syncthreads();
    if (warp > 0U) return;
    #pragma unroll
    for (uint32_t other = 0; other < kWarpsPerBlock - 1U; ++other) {
        gate_partial += gate_other_warps[other][lane];
        up_partial += up_other_warps[other][lane];
    }
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
        gate_partial += __shfl_down_sync(0xffffffffU, gate_partial, offset);
        up_partial += __shfl_down_sync(0xffffffffU, up_partial, offset);
    }
    if (lane == 0U) {
        const float gate = gate_partial * gate_tensor_scale;
        const float up = up_partial * up_tensor_scale;
        float sigmoid;
        if (gate >= 0.0f) {
            sigmoid = 1.0f / (1.0f + expf(-gate));
        } else {
            const float exponent = expf(gate);
            sigmoid = exponent / (1.0f + exponent);
        }
        output[row] = (gate * sigmoid) * up;
    }
}
#endif

bool checked_q8_1_footprint(
    uint32_t rows,
    uint32_t cols,
    uint64_t* weight_bytes,
    uint64_t* input_bytes,
    uint64_t* output_bytes
) {
    if (!checked_footprint(rows, cols, weight_bytes, input_bytes, output_bytes)) return false;
    const uint64_t activation_blocks = static_cast<uint64_t>(cols)
        / kQ8ActivationBlockElements;
    if (activation_blocks > std::numeric_limits<uint64_t>::max()
        / kQ8ActivationBlockBytes) {
        return false;
    }
    *input_bytes = activation_blocks * kQ8ActivationBlockBytes;
    return *input_bytes <= SIZE_MAX;
}

bool valid_args(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* args,
    uint64_t* weight_bytes,
    uint64_t* input_bytes,
    uint64_t* output_bytes
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_NVFP4_SM70_F32_MATVEC_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_nvfp4_sm70_f32_matvec_args_v1)
        || args->reserved != 0
        || !checked_footprint(args->rows, args->cols, weight_bytes, input_bytes, output_bytes)
        || args->weight_len_bytes != *weight_bytes
        || args->input_len_bytes < *input_bytes
        || args->output_len_bytes < *output_bytes
        || !checked_range(args->weight, *weight_bytes)
        || !checked_range(args->input, *input_bytes)
        || !checked_range(args->output, *output_bytes)
        || (args->input % alignof(float)) != 0
        || (args->output % alignof(float)) != 0
        || !std::isfinite(args->tensor_scale)
        || args->tensor_scale <= 0.0f) {
        return false;
    }
    return true;
}

bool valid_q8_1_args(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_q8_1_matvec_args_v1* args,
    uint64_t* weight_bytes,
    uint64_t* input_bytes,
    uint64_t* output_bytes
) {
    return stream != nullptr && stream->stream != nullptr && args != nullptr && ordinal >= 0
        && stream->ordinal == ordinal
        && args->abi_version == AUGUR_CUDA_NVFP4_SM70_Q8_1_MATVEC_ARGS_VERSION_V1
        && args->struct_size == sizeof(augur_cuda_nvfp4_sm70_q8_1_matvec_args_v1)
        && args->reserved == 0
        && checked_q8_1_footprint(
            args->rows, args->cols, weight_bytes, input_bytes, output_bytes)
        && args->weight_len_bytes == *weight_bytes
        && args->input_len_bytes == *input_bytes
        && args->output_len_bytes >= *output_bytes
        && checked_range(args->weight, *weight_bytes)
        && checked_range(args->input_q8_1, *input_bytes)
        && checked_range(args->output, *output_bytes)
        && (args->weight % alignof(uint32_t)) == 0
        && (args->input_q8_1 % alignof(uint32_t)) == 0
        && (args->output % alignof(float)) == 0
        && std::isfinite(args->tensor_scale)
        && args->tensor_scale > 0.0f;
}

#if defined(AUGUR_CUDA_QWEN35_FUSED_FFN_SWIGLU)
bool valid_q8_1_fused_swiglu_args(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_args_v1* args,
    uint64_t* weight_bytes,
    uint64_t* input_bytes,
    uint64_t* output_bytes
) {
    return stream != nullptr && stream->stream != nullptr && args != nullptr && ordinal >= 0
        && stream->ordinal == ordinal
        && args->abi_version == AUGUR_CUDA_NVFP4_SM70_Q8_1_FUSED_SWIGLU_ARGS_VERSION_V1
        && args->struct_size == sizeof(augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_args_v1)
        && checked_q8_1_footprint(
            args->rows, args->cols, weight_bytes, input_bytes, output_bytes)
        && args->gate_weight_len_bytes == *weight_bytes
        && args->up_weight_len_bytes == *weight_bytes
        && args->input_len_bytes == *input_bytes
        && args->output_len_bytes >= *output_bytes
        && checked_range(args->gate_weight, *weight_bytes)
        && checked_range(args->up_weight, *weight_bytes)
        && checked_range(args->input_q8_1, *input_bytes)
        && checked_range(args->output, *output_bytes)
        && ranges_disjoint(
            args->gate_weight, *weight_bytes, args->up_weight, *weight_bytes)
        && ranges_disjoint(
            args->gate_weight, *weight_bytes, args->input_q8_1, *input_bytes)
        && ranges_disjoint(
            args->gate_weight, *weight_bytes, args->output, *output_bytes)
        && ranges_disjoint(
            args->up_weight, *weight_bytes, args->input_q8_1, *input_bytes)
        && ranges_disjoint(
            args->up_weight, *weight_bytes, args->output, *output_bytes)
        && ranges_disjoint(
            args->input_q8_1, *input_bytes, args->output, *output_bytes)
        && (args->gate_weight % alignof(uint32_t)) == 0
        && (args->up_weight % alignof(uint32_t)) == 0
        && (args->input_q8_1 % alignof(uint32_t)) == 0
        && (args->output % alignof(float)) == 0
        && std::isfinite(args->gate_tensor_scale)
        && args->gate_tensor_scale > 0.0f
        && std::isfinite(args->up_tensor_scale)
        && args->up_tensor_scale > 0.0f;
}
#endif

// A pair has one canonical F32 input and two real, disjoint output vectors.
// Read-only weight/weight overlap is allowed; all writable cross-aliases reject.
bool small_gate_pair_valid(int32_t d,augur_cuda_stream_t s,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* a,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* b,uint32_t schedule) {
    uint64_t wa=0,xa=0,ya=0,wb=0,xb=0,yb=0;
    if((schedule!=1&&schedule!=2)||!valid_args(d,s,a,&wa,&xa,&ya)
       ||!valid_args(d,s,b,&wb,&xb,&yb)||s->stream==nullptr
       ||a->rows!=48||b->rows!=48||a->cols!=5120||b->cols!=5120
       ||a->input!=b->input||a->input_len_bytes!=xa||b->input_len_bytes!=xb
       ||a->output_len_bytes!=ya||b->output_len_bytes!=yb
       ||(a->weight%4)!=0||(b->weight%4)!=0) return false;
    return ranges_disjoint(a->weight,wa,a->input,xa)
        &&ranges_disjoint(b->weight,wb,a->input,xa)
        &&ranges_disjoint(a->weight,wa,a->output,ya)
        &&ranges_disjoint(b->weight,wb,b->output,yb)
        &&ranges_disjoint(a->input,xa,a->output,ya)
        &&ranges_disjoint(a->input,xa,b->output,yb)
        &&ranges_disjoint(a->weight,wa,b->output,yb)
        &&ranges_disjoint(b->weight,wb,a->output,ya)
        &&ranges_disjoint(a->output,ya,b->output,yb);
}
} // namespace

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_f32_matvec_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* args
) {
    uint64_t weight_bytes = 0;
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    if (!valid_args(ordinal, stream, args, &weight_bytes, &input_bytes, &output_bytes)) {
        return nvfp4_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const augur_cuda_result_v1 device = nvfp4_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;

    const uint32_t blocks = (args->rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
    nvfp4_sm70_f32_matvec_kernel<<<blocks, kWarpsPerBlock * 32U, 0, stream->stream>>>(
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->rows,
        args->cols,
        args->tensor_scale);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? nvfp4_ok()
        : nvfp4_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

// Distinct physical-layout entry: the weight MUST be QPN planes, not GGML36.
// The Rust owner is typed to QPN and retains every operand/workspace allocation.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_qpn_source_pair_submit_v1(
    int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* a,
    uint32_t mode,uint32_t splits,uint64_t partial,uint64_t partial_bytes) {
    uint64_t w=0,x=0,y=0;
    if(!valid_args(ordinal,stream,a,&w,&x,&y)||stream->stream==nullptr
       ||a->rows%32U||a->cols%64U||a->weight%16U||a->input_len_bytes!=x||a->output_len_bytes!=y
       ||!ranges_disjoint(a->weight,w,a->input,x)||!ranges_disjoint(a->weight,w,a->output,y)
       ||!ranges_disjoint(a->input,x,a->output,y)||mode>4U
       ||(mode==0U&&splits!=1U&&splits!=4U&&splits!=8U&&splits!=16U)
       ||(mode==1U&&splits!=4U&&splits!=8U&&splits!=16U&&splits!=32U)
       ||(mode>=2U&&splits!=4U)
       ||(mode>=3U&&(a->input%16U)))
        return nvfp4_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    if(mode!=1U) {
        if(partial||partial_bytes)return nvfp4_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    } else if(partial_bytes!=uint64_t(a->rows)*splits*8U||partial%8U||!checked_range(partial,partial_bytes)
        ||!ranges_disjoint(partial,partial_bytes,a->weight,w)||!ranges_disjoint(partial,partial_bytes,a->input,x)
        ||!ranges_disjoint(partial,partial_bytes,a->output,y))
        return nvfp4_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    const auto device=nvfp4_set_device(ordinal);if(device.status!=AUGUR_CUDA_STATUS_OK)return device;
    const auto* weight=reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a->weight));
    const auto* input=reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input));
    auto* output=reinterpret_cast<float*>(static_cast<uintptr_t>(a->output));
    if(mode==4U) {
        // Bench-selected: 16 K-streams, 32 for the narrow 1024-row K/V.
        if(a->rows<=2048U)
            qpn_source_pair::source_reassoc_kernel<32><<<a->rows/32U,1024,0,stream->stream>>>(weight,input,output,a->rows,a->cols,a->tensor_scale);
        else
            qpn_source_pair::source_reassoc_kernel<16><<<a->rows/32U,512,0,stream->stream>>>(weight,input,output,a->rows,a->cols,a->tensor_scale);
    } else if(mode==3U) {
        // Order-identical prefetch issue of mode 2. The bench-selected split:
        // wide-output shapes keep all eight chains per thread, narrow ones
        // split them over two warps per stream for more loads in flight.
        if(a->rows>=10240U&&a->rows!=17408U)
            qpn_source_pair::source_prefetch_kernel<8,4><<<a->rows/32U,128,0,stream->stream>>>(weight,input,output,a->rows,a->cols,a->tensor_scale);
        else
            qpn_source_pair::source_prefetch_kernel<4,4><<<a->rows/32U,256,0,stream->stream>>>(weight,input,output,a->rows,a->cols,a->tensor_scale);
    } else if(mode==2U) {
        qpn_source_pair::source_exact_kernel<<<a->rows/32U,128,0,stream->stream>>>(weight,input,output,a->rows,a->cols,a->tensor_scale);
    } else if(mode==0U) {
        #define QPN_SOURCE_CTA(S) case S:qpn_source_pair::cta_kernel<S><<<a->rows/32U,S*32U,0,stream->stream>>>(weight,input,output,a->rows,a->cols,a->tensor_scale);break
        switch(splits){QPN_SOURCE_CTA(1);QPN_SOURCE_CTA(4);QPN_SOURCE_CTA(8);QPN_SOURCE_CTA(16);}
        #undef QPN_SOURCE_CTA
    } else {
        auto* scratch=reinterpret_cast<double*>(static_cast<uintptr_t>(partial));
        qpn_source_pair::parts_kernel<<<dim3((a->rows+127U)/128U,splits),128,0,stream->stream>>>(weight,input,scratch,a->rows,a->cols,splits);
        const auto e=cudaGetLastError();if(e!=cudaSuccess)return nvfp4_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,e);
        qpn_source_pair::merge_kernel<<<(a->rows+255U)/256U,256,0,stream->stream>>>(scratch,output,a->rows,splits,a->tensor_scale);
    }
    const auto e=cudaGetLastError();return e==cudaSuccess?nvfp4_ok():nvfp4_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,e);
}

// Separate large-projection research entry. Reuse the established source-F32
// pair / ordered F64 SplitK fold, but never widen the small-gate ABI itself.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_large_split8_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* args
) {
    uint64_t w=0,x=0,y=0;
    if (!valid_args(ordinal,stream,args,&w,&x,&y) || stream->stream==nullptr
        || args->rows<1024 || args->rows%32 || args->cols<5120 || args->cols%512
        || args->input_len_bytes!=x || args->output_len_bytes!=y
        || args->weight%alignof(uint32_t)
        || !ranges_disjoint(args->weight,w,args->input,x)
        || !ranges_disjoint(args->weight,w,args->output,y)
        || !ranges_disjoint(args->input,x,args->output,y))
        return nvfp4_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    const auto device=nvfp4_set_device(ordinal);
    if(device.status!=AUGUR_CUDA_STATUS_OK)return device;
    nvfp4_sm70_small_gate_split_f64_kernel<8><<<args->rows,256,0,stream->stream>>>(
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->rows,args->cols,args->tensor_scale);
    const auto error=cudaGetLastError();
    return error==cudaSuccess?nvfp4_ok():nvfp4_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
}

// Separately selected small-gate numerical ABI. Never used by the original
// scalar submit entry point; reject all shapes except the real 48x5120 gates.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_small_gate_split16_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* args
) {
    uint64_t w = 0, x = 0, y = 0;
    if (!valid_args(ordinal, stream, args, &w, &x, &y)
        || stream->stream == nullptr || args->rows != 48 || args->cols != 5120
        || args->input_len_bytes != x || args->output_len_bytes != y
        || (args->weight % alignof(uint32_t)) != 0
        || !ranges_disjoint(args->weight,w,args->input,x)
        || !ranges_disjoint(args->weight,w,args->output,y)
        || !ranges_disjoint(args->input,x,args->output,y)) {
        return nvfp4_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const augur_cuda_result_v1 device = nvfp4_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    nvfp4_sm70_small_gate_split_f64_kernel<16><<<48,512,0,stream->stream>>>(
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->rows,args->cols,args->tensor_scale);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? nvfp4_ok() : nvfp4_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_small_gate_pair_prevalidate_v1(
    int32_t d,augur_cuda_stream_t s,const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* a,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* b,uint32_t schedule
) {
    return small_gate_pair_valid(d,s,a,b,schedule)?nvfp4_ok():nvfp4_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_small_gate_pair_submit_v1(
    int32_t d,augur_cuda_stream_t s,const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* a,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* b,uint32_t schedule
) {
    if(!small_gate_pair_valid(d,s,a,b,schedule))return nvfp4_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    const augur_cuda_result_v1 device=nvfp4_set_device(d);if(device.status!=AUGUR_CUDA_STATUS_OK)return device;
    const auto* wa=reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a->weight));
    const auto* wb=reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(b->weight));
    const auto* x=reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input));
    auto* ya=reinterpret_cast<float*>(static_cast<uintptr_t>(a->output));
    auto* yb=reinterpret_cast<float*>(static_cast<uintptr_t>(b->output));
    if(schedule==1)nvfp4_sm70_small_gate_pair_concat_kernel<true><<<96,512,0,s->stream>>>(wa,wb,x,ya,yb,5120,a->tensor_scale,b->tensor_scale);
    else nvfp4_sm70_small_gate_pair_shared_input_kernel<true,true><<<48,512,0,s->stream>>>(wa,wb,x,ya,yb,5120,a->tensor_scale,b->tensor_scale);
    const cudaError_t error=cudaGetLastError();
    return error==cudaSuccess?nvfp4_ok():nvfp4_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_f32_matvec_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* args
) {
    const augur_cuda_result_v1 submitted =
        augur_cuda_nvfp4_sm70_f32_matvec_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    const cudaError_t error = cudaStreamSynchronize(stream->stream);
    return error == cudaSuccess
        ? nvfp4_ok()
        : nvfp4_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_f32_batch_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* args, uint32_t batch_rows
) {
    uint64_t w = 0, x = 0, y = 0;
    if (batch_rows == 0 || batch_rows > 8
        || !valid_args(ordinal, stream, args, &w, &x, &y)
        || stream->stream == nullptr
        || x > UINT64_MAX / batch_rows || y > UINT64_MAX / batch_rows) {
        return nvfp4_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    x *= batch_rows; y *= batch_rows;
    if (args->input_len_bytes != x || args->output_len_bytes != y
        || !checked_range(args->input,x) || !checked_range(args->output,y)
        || (args->weight % alignof(uint32_t)) != 0
        || !ranges_disjoint(args->weight,w,args->input,x)
        || !ranges_disjoint(args->weight,w,args->output,y)
        || !ranges_disjoint(args->input,x,args->output,y)) {
        return nvfp4_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const augur_cuda_result_v1 device = nvfp4_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    const uint32_t blocks = (args->rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
    const uint8_t* weight = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight));
    const float* input = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input));
    float* output = reinterpret_cast<float*>(static_cast<uintptr_t>(args->output));
    // B1 keeps the existing launch and arithmetic exactly.
    if (batch_rows == 1) {
        nvfp4_sm70_f32_matvec_kernel<<<blocks,128,0,stream->stream>>>(weight,input,output,args->rows,args->cols,args->tensor_scale);
    } else {
        #define AUGUR_BATCH_CASE(B) case B: nvfp4_sm70_f32_batch_kernel<B><<<blocks,128,0,stream->stream>>>(weight,input,output,args->rows,args->cols,args->tensor_scale); break
        switch (batch_rows) { AUGUR_BATCH_CASE(2); AUGUR_BATCH_CASE(3); AUGUR_BATCH_CASE(4); AUGUR_BATCH_CASE(5); AUGUR_BATCH_CASE(6); AUGUR_BATCH_CASE(7); AUGUR_BATCH_CASE(8); }
        #undef AUGUR_BATCH_CASE
    }
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? nvfp4_ok() : nvfp4_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_q8_1_matvec_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_q8_1_matvec_args_v1* args
) {
    uint64_t weight_bytes = 0;
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    if (!valid_q8_1_args(
            ordinal, stream, args, &weight_bytes, &input_bytes, &output_bytes)) {
        return nvfp4_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const augur_cuda_result_v1 device = nvfp4_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;

    nvfp4_sm70_q8_1_matvec_kernel<<<args->rows, 128U, 0, stream->stream>>>(
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->input_q8_1)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->rows,
        args->cols,
        args->tensor_scale);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? nvfp4_ok()
        : nvfp4_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_q8_1_matvec_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_q8_1_matvec_args_v1* args
) {
    const augur_cuda_result_v1 submitted =
        augur_cuda_nvfp4_sm70_q8_1_matvec_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    const cudaError_t error = cudaStreamSynchronize(stream->stream);
    return error == cudaSuccess
        ? nvfp4_ok()
        : nvfp4_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, error);
}

#if defined(AUGUR_CUDA_QWEN35_FUSED_FFN_SWIGLU)
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_args_v1* args
) {
    uint64_t weight_bytes = 0;
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    if (!valid_q8_1_fused_swiglu_args(
            ordinal, stream, args, &weight_bytes, &input_bytes, &output_bytes)) {
        return nvfp4_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const augur_cuda_result_v1 device = nvfp4_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;

    nvfp4_sm70_q8_1_fused_swiglu_kernel<<<args->rows, 128U, 0, stream->stream>>>(
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->gate_weight)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->up_weight)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->input_q8_1)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->rows,
        args->cols,
        args->gate_tensor_scale,
        args->up_tensor_scale);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? nvfp4_ok()
        : nvfp4_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_args_v1* args
) {
    const augur_cuda_result_v1 submitted =
        augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    const cudaError_t error = cudaStreamSynchronize(stream->stream);
    return error == cudaSuccess
        ? nvfp4_ok()
        : nvfp4_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, error);
}
#endif
