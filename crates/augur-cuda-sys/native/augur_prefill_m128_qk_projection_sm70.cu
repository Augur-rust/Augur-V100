#include "augur_cuda_internal.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr uint32_t kDecodedBlockElements = 256;
constexpr uint32_t kQ4BlockBytes = 160;
constexpr uint32_t kQ5BlockBytes = 192;
constexpr uint32_t kQ6BlockBytes = 224;
constexpr uint32_t kTile = 16;
constexpr uint32_t kRowGroup = 32;
constexpr uint32_t kWarpsPerCta = 4;
constexpr uint32_t kColumnsPerCta = kTile * kWarpsPerCta;

static_assert(sizeof(augur_cuda_prefill_m128_qk_projection_args_v1) == 160,
    "M128 QK projection args ABI drift");
static_assert(offsetof(augur_cuda_prefill_m128_qk_projection_args_v1, weight) == 40,
    "M128 QK projection weight offset drift");
static_assert(offsetof(augur_cuda_prefill_m128_qk_projection_args_v1, output_f32) == 120,
    "M128 QK projection output offset drift");
static_assert(offsetof(augur_cuda_prefill_m128_qk_projection_args_v1, reserved1) == 136,
    "M128 QK projection reserved offset drift");

augur_cuda_result_v1 result(augur_cuda_status_t status, cudaError_t error) {
    return augur_cuda_result_v1{status, static_cast<int32_t>(error)};
}

bool checked_mul(uint64_t left, uint64_t right, uint64_t* product) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        return false;
    }
    *product = left * right;
    return true;
}

bool finite_bucket(uint32_t rows) {
    return rows != 0 && rows <= AUGUR_CUDA_PREFILL_M128_QK_MAX_BUCKET_ROWS_V1
        && (rows & (rows - 1)) == 0;
}

uint32_t matrix_rows(uint32_t bucket_rows) {
    return bucket_rows < kTile ? kTile : bucket_rows;
}

struct Range {
    uint64_t address;
    uint64_t bytes;
    uint64_t alignment;
};

bool range_metadata_valid(const Range range) {
    return range.address != 0 && range.bytes != 0
        && range.address % range.alignment == 0
        && range.address <= std::numeric_limits<uint64_t>::max() - range.bytes;
}

bool device_range(const Range range, int32_t ordinal) {
    if (!range_metadata_valid(range)) return false;
    cudaPointerAttributes attributes{};
    const cudaError_t error = cudaPointerGetAttributes(
        &attributes,
        reinterpret_cast<const void*>(static_cast<uintptr_t>(range.address)));
    if (error != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return attributes.type == cudaMemoryTypeDevice && attributes.device == ordinal;
}

bool overlap(const Range left, const Range right) {
    return left.address < right.address + right.bytes
        && right.address < left.address + left.bytes;
}

bool metadata_valid(const augur_cuda_prefill_m128_qk_projection_args_v1* args) {
    if (args == nullptr
        || args->abi_version != AUGUR_CUDA_PREFILL_M128_QK_PROJECTION_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args)
        || (args->format != AUGUR_CUDA_PREFILL_M128_QK_FORMAT_Q4K_V1
            && args->format != AUGUR_CUDA_PREFILL_M128_QK_FORMAT_Q5K_V1
            && args->format != AUGUR_CUDA_PREFILL_M128_QK_FORMAT_Q6K_V1)
        || !finite_bucket(args->bucket_rows)
        || args->valid_rows == 0 || args->valid_rows > args->bucket_rows
        || args->output_rows == 0 || args->output_rows % kColumnsPerCta != 0
        || args->input_cols == 0 || args->input_cols % kDecodedBlockElements != 0
        || args->blocks_per_output_row != args->input_cols / kDecodedBlockElements
        || args->reserved0 != 0) {
        return false;
    }
    for (uint32_t index = 0; index < 6; ++index) {
        if (args->reserved1[index] != 0) return false;
    }

    const uint32_t block_bytes = args->format == AUGUR_CUDA_PREFILL_M128_QK_FORMAT_Q4K_V1
        ? kQ4BlockBytes
        : args->format == AUGUR_CUDA_PREFILL_M128_QK_FORMAT_Q5K_V1
            ? kQ5BlockBytes : kQ6BlockBytes;
    if (args->weight_block_stride_bytes != block_bytes) return false;

    uint64_t weight_blocks = 0;
    uint64_t weight_bytes = 0;
    uint64_t weight_elements = 0;
    uint64_t decoded_weight_bytes = 0;
    uint64_t input_elements = 0;
    uint64_t input_bytes = 0;
    uint64_t converted_elements = 0;
    uint64_t converted_bytes = 0;
    uint64_t output_elements = 0;
    uint64_t output_bytes = 0;
    uint64_t wmma_elements = 0;
    uint64_t wmma_bytes = 0;
    const uint64_t padded_rows = matrix_rows(args->bucket_rows);
    if (!checked_mul(args->output_rows, args->blocks_per_output_row, &weight_blocks)
        || !checked_mul(weight_blocks, block_bytes, &weight_bytes)
        || !checked_mul(args->output_rows, args->input_cols, &weight_elements)
        || !checked_mul(weight_elements, sizeof(__half), &decoded_weight_bytes)
        || !checked_mul(args->bucket_rows, args->input_cols, &input_elements)
        || !checked_mul(input_elements, sizeof(float), &input_bytes)
        || !checked_mul(padded_rows, args->input_cols, &converted_elements)
        || !checked_mul(converted_elements, sizeof(__half), &converted_bytes)
        || !checked_mul(args->bucket_rows, args->output_rows, &output_elements)
        || !checked_mul(output_elements, sizeof(float), &output_bytes)
        || !checked_mul(padded_rows, args->output_rows, &wmma_elements)
        || !checked_mul(wmma_elements, sizeof(float), &wmma_bytes)) {
        return false;
    }
    return args->weight_len_bytes == weight_bytes
        && args->input_len_bytes == input_bytes
        && args->decoded_weight_len_bytes == decoded_weight_bytes
        && args->converted_input_len_bytes == converted_bytes
        && args->wmma_output_len_bytes == wmma_bytes
        && args->output_len_bytes == output_bytes;
}

bool full_preflight(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_qk_projection_args_v1* args) {
    if (ordinal < 0 || stream == nullptr || stream->ordinal != ordinal
        || !metadata_valid(args) || cudaSetDevice(ordinal) != cudaSuccess) {
        return false;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, ordinal) != cudaSuccess
        || properties.major != 7 || properties.minor != 0) {
        return false;
    }
    const Range ranges[] = {
        {args->weight, args->weight_len_bytes, 32},
        {args->input_f32, args->input_len_bytes, alignof(float)},
        {args->decoded_weight_f16, args->decoded_weight_len_bytes, 32},
        {args->converted_input_f16, args->converted_input_len_bytes, 32},
        {args->wmma_output_f32, args->wmma_output_len_bytes, 32},
        {args->output_f32, args->output_len_bytes, alignof(float)},
    };
    for (const Range range : ranges) {
        if (!device_range(range, ordinal)) return false;
    }
    constexpr size_t count = sizeof(ranges) / sizeof(ranges[0]);
    for (size_t left = 0; left < count; ++left) {
        for (size_t right = left + 1; right < count; ++right) {
            if (overlap(ranges[left], ranges[right])) return false;
        }
    }
    return true;
}

bool prevalidated_metadata(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_qk_projection_args_v1* args) {
    return ordinal >= 0 && stream != nullptr && stream->ordinal == ordinal
        && metadata_valid(args);
}

__device__ __forceinline__ uint16_t load_u16_le(const unsigned char* pointer) {
    return static_cast<uint16_t>(pointer[0])
        | (static_cast<uint16_t>(pointer[1]) << 8);
}

__device__ __forceinline__ float half_bits_to_float(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ float q4k_weight(
    const unsigned char* block, uint32_t lane) {
    const float d = half_bits_to_float(load_u16_le(block));
    const float dmin = half_bits_to_float(load_u16_le(block + 2));
    const uint32_t group = lane / 32;
    const uint32_t within = lane % 64;
    const uint32_t q_index = (lane / 64) * 32 + (within % 32);
    const unsigned char packed = block[32 + q_index];
    const unsigned char q = within < 32 ? packed & 0x0f : packed >> 4;
    return d * static_cast<float>(block[4 + group]) * static_cast<float>(q)
        - dmin * static_cast<float>(block[12 + group]);
}

__device__ __forceinline__ float q5k_weight(
    const unsigned char* block, uint32_t lane) {
    const float d = half_bits_to_float(load_u16_le(block));
    const float dmin = half_bits_to_float(load_u16_le(block + 2));
    const uint32_t group = lane / 32;
    const uint32_t chunk = lane / 64;
    const uint32_t within = lane % 64;
    const uint32_t half = within / 32;
    const uint32_t q_lane = within % 32;
    const unsigned char packed = block[64 + chunk * 32 + q_lane];
    const unsigned char low = half == 0 ? packed & 0x0f : packed >> 4;
    const unsigned char high = (block[32 + q_lane] >> (2 * chunk + half)) & 1;
    const unsigned char q = low | (high << 4);
    return d * static_cast<float>(block[4 + group]) * static_cast<float>(q)
        - dmin * static_cast<float>(block[12 + group]);
}

__device__ __forceinline__ float q6k_weight(
    const unsigned char* block, uint32_t lane) {
    const uint32_t half = lane / 128;
    const uint32_t local = lane % 128;
    const uint32_t quarter = local / 32;
    const uint32_t q_lane = local % 32;
    const uint32_t low_base = half * 64;
    const unsigned char high = block[128 + half * 32 + q_lane];
    unsigned char q = 0;
    if (quarter == 0) {
        q = (block[low_base + q_lane] & 0x0f) | ((high & 3) << 4);
    } else if (quarter == 1) {
        q = (block[low_base + 32 + q_lane] & 0x0f)
            | (((high >> 2) & 3) << 4);
    } else if (quarter == 2) {
        q = (block[low_base + q_lane] >> 4) | (((high >> 4) & 3) << 4);
    } else {
        q = (block[low_base + 32 + q_lane] >> 4)
            | (((high >> 6) & 3) << 4);
    }
    const int8_t scale = static_cast<int8_t>(
        block[192 + half * 8 + quarter * 2 + q_lane / 16]);
    const float d = half_bits_to_float(load_u16_le(block + 208));
    return d * static_cast<float>(scale) * (static_cast<int>(q) - 32);
}

template <uint32_t BlockBytes, float (*Decode)(const unsigned char*, uint32_t)>
__global__ void decode_weight_f16_kernel(
    const unsigned char* weight,
    __half* decoded,
    uint64_t elements,
    uint32_t input_cols,
    uint32_t blocks_per_row) {
    const uint64_t linear = static_cast<uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (linear >= elements) return;
    const uint32_t output_row = static_cast<uint32_t>(linear / input_cols);
    const uint32_t input_col = static_cast<uint32_t>(linear % input_cols);
    const uint32_t block_index = input_col / kDecodedBlockElements;
    const uint32_t lane = input_col % kDecodedBlockElements;
    const unsigned char* block = weight
        + (static_cast<uint64_t>(output_row) * blocks_per_row + block_index)
            * BlockBytes;
    decoded[linear] = __float2half_rn(Decode(block, lane));
}

__global__ void convert_and_pad_input_kernel(
    const float* input,
    __half* converted,
    uint64_t elements,
    uint32_t valid_rows,
    uint32_t input_cols) {
    const uint64_t linear = static_cast<uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (linear >= elements) return;
    const uint32_t row = static_cast<uint32_t>(linear / input_cols);
    converted[linear] = row < valid_rows
        ? __float2half_rn(input[linear]) : __float2half_rn(0.0f);
}

__global__ __launch_bounds__(128) void wmma_rowgroup1_kernel(
    const __half* weight,
    const __half* input,
    float* output,
    uint32_t rows,
    uint32_t output_rows,
    uint32_t input_cols) {
    using namespace nvcuda;
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t row_base = blockIdx.y * kTile;
    const uint32_t column_base = blockIdx.x * kColumnsPerCta + warp * kTile;
    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (uint32_t inner = 0; inner < input_cols; inner += kTile) {
        wmma::load_matrix_sync(a, input + static_cast<uint64_t>(row_base) * input_cols + inner,
            input_cols);
        wmma::load_matrix_sync(b,
            weight + static_cast<uint64_t>(column_base) * input_cols + inner, input_cols);
        wmma::mma_sync(accumulator, a, b, accumulator);
    }
    wmma::store_matrix_sync(
        output + static_cast<uint64_t>(row_base) * output_rows + column_base,
        accumulator, output_rows, wmma::mem_row_major);
}

__global__ __launch_bounds__(128) void wmma_rowgroup2_kernel(
    const __half* weight,
    const __half* input,
    float* output,
    uint32_t rows,
    uint32_t output_rows,
    uint32_t input_cols) {
    using namespace nvcuda;
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t row_base = blockIdx.y * kRowGroup;
    const uint32_t column_base = blockIdx.x * kColumnsPerCta + warp * kTile;
    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __half, wmma::row_major> a0;
    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __half, wmma::row_major> a1;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> accumulator0;
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> accumulator1;
    wmma::fill_fragment(accumulator0, 0.0f);
    wmma::fill_fragment(accumulator1, 0.0f);
    for (uint32_t inner = 0; inner < input_cols; inner += kTile) {
        wmma::load_matrix_sync(b,
            weight + static_cast<uint64_t>(column_base) * input_cols + inner, input_cols);
        wmma::load_matrix_sync(a0,
            input + static_cast<uint64_t>(row_base) * input_cols + inner, input_cols);
        wmma::mma_sync(accumulator0, a0, b, accumulator0);
        wmma::load_matrix_sync(a1,
            input + static_cast<uint64_t>(row_base + kTile) * input_cols + inner,
            input_cols);
        wmma::mma_sync(accumulator1, a1, b, accumulator1);
    }
    wmma::store_matrix_sync(
        output + static_cast<uint64_t>(row_base) * output_rows + column_base,
        accumulator0, output_rows, wmma::mem_row_major);
    wmma::store_matrix_sync(
        output + static_cast<uint64_t>(row_base + kTile) * output_rows + column_base,
        accumulator1, output_rows, wmma::mem_row_major);
}

__global__ void copy_valid_rows_kernel(
    const float* staged, float* output, uint64_t valid_elements) {
    const uint64_t linear = static_cast<uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (linear < valid_elements) output[linear] = staged[linear];
}

augur_cuda_result_v1 launch(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_qk_projection_args_v1* args) {
    const cudaError_t set_device = cudaSetDevice(ordinal);
    if (set_device != cudaSuccess) {
        return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, set_device);
    }
    const uint64_t decoded_elements = static_cast<uint64_t>(args->output_rows)
        * args->input_cols;
    const uint32_t padded_rows = matrix_rows(args->bucket_rows);
    const uint64_t converted_elements = static_cast<uint64_t>(padded_rows)
        * args->input_cols;
    const uint64_t valid_output_elements = static_cast<uint64_t>(args->valid_rows)
        * args->output_rows;
    constexpr uint32_t threads = 256;
    const auto decode_blocks = static_cast<uint32_t>((decoded_elements + threads - 1) / threads);
    const auto convert_blocks = static_cast<uint32_t>((converted_elements + threads - 1) / threads);
    const auto copy_blocks = static_cast<uint32_t>((valid_output_elements + threads - 1) / threads);
    const auto* weight = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(args->weight));
    auto* decoded = reinterpret_cast<__half*>(
        static_cast<uintptr_t>(args->decoded_weight_f16));
    const auto* input = reinterpret_cast<const float*>(
        static_cast<uintptr_t>(args->input_f32));
    auto* converted = reinterpret_cast<__half*>(
        static_cast<uintptr_t>(args->converted_input_f16));
    auto* staged_output = reinterpret_cast<float*>(
        static_cast<uintptr_t>(args->wmma_output_f32));
    auto* output = reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32));

    if (args->format == AUGUR_CUDA_PREFILL_M128_QK_FORMAT_Q4K_V1) {
        decode_weight_f16_kernel<kQ4BlockBytes, q4k_weight>
            <<<decode_blocks, threads, 0, stream->stream>>>(weight, decoded,
                decoded_elements, args->input_cols, args->blocks_per_output_row);
    } else if (args->format == AUGUR_CUDA_PREFILL_M128_QK_FORMAT_Q5K_V1) {
        decode_weight_f16_kernel<kQ5BlockBytes, q5k_weight>
            <<<decode_blocks, threads, 0, stream->stream>>>(weight, decoded,
                decoded_elements, args->input_cols, args->blocks_per_output_row);
    } else {
        decode_weight_f16_kernel<kQ6BlockBytes, q6k_weight>
            <<<decode_blocks, threads, 0, stream->stream>>>(weight, decoded,
                decoded_elements, args->input_cols, args->blocks_per_output_row);
    }
    convert_and_pad_input_kernel<<<convert_blocks, threads, 0, stream->stream>>>(
        input, converted, converted_elements, args->valid_rows, args->input_cols);
    const dim3 grid(args->output_rows / kColumnsPerCta,
        padded_rows < kRowGroup ? padded_rows / kTile : padded_rows / kRowGroup);
    if (padded_rows < kRowGroup) {
        wmma_rowgroup1_kernel<<<grid, 128, 0, stream->stream>>>(
            decoded, converted, staged_output, padded_rows, args->output_rows,
            args->input_cols);
    } else {
        wmma_rowgroup2_kernel<<<grid, 128, 0, stream->stream>>>(
            decoded, converted, staged_output, padded_rows, args->output_rows,
            args->input_cols);
    }
    copy_valid_rows_kernel<<<copy_blocks, threads, 0, stream->stream>>>(
        staged_output, output, valid_output_elements);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, error)
        : result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

}  // namespace

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_qk_projection_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_qk_projection_args_v1* args) {
    return full_preflight(ordinal, stream, args)
        ? result(AUGUR_CUDA_STATUS_OK, cudaSuccess)
        : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_qk_projection_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_qk_projection_args_v1* args) {
    if (!full_preflight(ordinal, stream, args)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return launch(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_qk_projection_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_qk_projection_args_v1* args) {
    if (!prevalidated_metadata(ordinal, stream, args)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return launch(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_qk_projection_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_qk_projection_args_v1* args) {
    if (!full_preflight(ordinal, stream, args)) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const augur_cuda_result_v1 submitted = launch(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    const cudaError_t error = cudaStreamSynchronize(stream->stream);
    return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, error)
        : result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, error);
}
