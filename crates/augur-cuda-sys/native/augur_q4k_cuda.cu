#include "augur_cuda_runtime.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

static_assert(
    sizeof(augur_cuda_q4k_f32_decode_args_v1) == 72,
    "Q4K decode args ABI size changed"
);
static_assert(
    sizeof(augur_cuda_q4k_f32_prefill_args_v1) == 80,
    "Q4K prefill args ABI size changed"
);
static_assert(sizeof(augur_cuda_q5k_f32_decode_args_v1) == 72, "Q5K decode args ABI size changed");
static_assert(sizeof(augur_cuda_q6k_f32_decode_args_v1) == 72, "Q6K decode args ABI size changed");
#if defined(AUGUR_CUDA_P4_EXPERT_MICROSLICE_GPU_REPACK)
static_assert(sizeof(augur_cuda_q4k_compact_to_hot_args_v1) == 64, "P4 Q4K compact-to-hot args ABI size changed");
static_assert(sizeof(augur_cuda_q5k_compact_to_hot_args_v1) == 64, "P4 Q5K compact-to-hot args ABI size changed");
#endif
#if defined(AUGUR_CUDA_M306_K15_Q5_PAIRED_PROJECTION)
static_assert(
    sizeof(augur_cuda_batch01_b4_q5k_f32_paired_projection_args_v1) == 176,
    "BATCH-01 B4 Q5 paired args ABI size changed"
);
#if defined(AUGUR_CUDA_BATCH01_B8_Q5_ROW_TILE)
static_assert(
    sizeof(augur_cuda_batch01_b8_q5k_row_tile_args_v1) == 272,
    "BATCH-01 B8 Q5 row-tile args ABI size changed"
);
#endif
#endif

static constexpr uint32_t AUGUR_Q4K_DECODED_LEN = 256;
static constexpr uint32_t AUGUR_Q4K_GROUPS = 8;
static constexpr uint32_t AUGUR_Q4K_GROUP_SIZE = 32;
static constexpr uint32_t AUGUR_Q4K_QS_BYTES = 128;
static constexpr uint32_t AUGUR_Q4K_BLOCK_BYTES = 160;
static constexpr uint32_t AUGUR_Q4K_BLOCK_ALIGNMENT = 32;

#if defined(AUGUR_CUDA_P4_EXPERT_MICROSLICE_GPU_REPACK)
static constexpr uint32_t AUGUR_Q4K_COMPACT_BLOCK_BYTES = 144;
static constexpr uint32_t AUGUR_Q4K_HOT_BLOCK_BYTES = 160;
static constexpr uint32_t AUGUR_Q5K_COMPACT_BLOCK_BYTES = 176;
static constexpr uint32_t AUGUR_Q5K_HOT_BLOCK_BYTES = 192;

__device__ __forceinline__ void augur_p4_expand_k_scales(
    const unsigned char* compact,
    unsigned char* hot,
    uint32_t lane
) {
    if (lane < 4) {
        const unsigned char low_scale = compact[4 + lane];
        const unsigned char low_min = compact[8 + lane];
        const unsigned char high = compact[12 + lane];
        hot[4 + lane] = low_scale & 0x3f;
        hot[12 + lane] = low_min & 0x3f;
        hot[8 + lane] = (high & 0x0f) | ((low_scale >> 6) << 4);
        hot[16 + lane] = (high >> 4) | ((low_min >> 6) << 4);
    }
}

template <uint32_t CompactBytes, uint32_t HotBytes>
__global__ void augur_p4_compact_to_hot_kernel(
    const unsigned char* compact,
    unsigned char* hot,
    uint32_t rows,
    uint32_t blocks_per_row
) {
    const uint32_t block_index = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint64_t block_count = static_cast<uint64_t>(rows) * blocks_per_row;
    if (block_index >= block_count || lane >= 32) return;
    const unsigned char* source = compact + static_cast<size_t>(block_index) * CompactBytes;
    unsigned char* destination = hot + static_cast<size_t>(block_index) * HotBytes;
    if (lane < 4) destination[lane] = source[lane];
    if (lane < 12) destination[20 + lane] = 0;
    for (uint32_t offset = lane; offset < CompactBytes - 16; offset += 32) {
        destination[32 + offset] = source[16 + offset];
    }
    augur_p4_expand_k_scales(source, destination, lane);
}

cudaError_t augur_p4_q4k_compact_to_hot_launch(
    const unsigned char* compact, unsigned char* hot,
    uint32_t rows, uint32_t blocks_per_row, cudaStream_t stream
) {
    const uint64_t blocks = static_cast<uint64_t>(rows) * blocks_per_row;
    augur_p4_compact_to_hot_kernel<AUGUR_Q4K_COMPACT_BLOCK_BYTES, AUGUR_Q4K_HOT_BLOCK_BYTES>
        <<<static_cast<unsigned int>(blocks), 32, 0, stream>>>(
            compact, hot, rows, blocks_per_row);
    return cudaGetLastError();
}

cudaError_t augur_p4_q5k_compact_to_hot_launch(
    const unsigned char* compact, unsigned char* hot,
    uint32_t rows, uint32_t blocks_per_row, cudaStream_t stream
) {
    const uint64_t blocks = static_cast<uint64_t>(rows) * blocks_per_row;
    augur_p4_compact_to_hot_kernel<AUGUR_Q5K_COMPACT_BLOCK_BYTES, AUGUR_Q5K_HOT_BLOCK_BYTES>
        <<<static_cast<unsigned int>(blocks), 32, 0, stream>>>(
            compact, hot, rows, blocks_per_row);
    return cudaGetLastError();
}
#endif

static __device__ __forceinline__ uint16_t load_u16_le(
    const unsigned char* ptr
) {
    return static_cast<uint16_t>(ptr[0])
        | (static_cast<uint16_t>(ptr[1]) << 8);
}

static __device__ __forceinline__ float half_bits_to_float(
    uint16_t bits
) {
    return __half2float(__ushort_as_half(bits));
}

static __device__ __forceinline__ float q4k_weight(
    const unsigned char* block,
    uint32_t lane
) {
    const float d = half_bits_to_float(
        load_u16_le(block + 0)
    );
    const float dmin = half_bits_to_float(
        load_u16_le(block + 2)
    );

    const uint32_t group = lane / AUGUR_Q4K_GROUP_SIZE;
    const uint32_t within_chunk = lane % 64;
    const uint32_t q_byte_index = (lane / 64) * 32 + (within_chunk % 32);
    const unsigned char q_byte = block[32 + q_byte_index];

    const unsigned char q = within_chunk < 32
        ? static_cast<unsigned char>(q_byte & 0x0f)
        : static_cast<unsigned char>(q_byte >> 4);

    const float scale = static_cast<float>(block[4 + group]);
    const float min_value = static_cast<float>(block[12 + group]);

    return d * scale * static_cast<float>(q)
        - dmin * min_value;
}

__global__ void augur_q4k_f32_decode_kernel_v1(
    const unsigned char* weight,
    const float* input,
    float* output,
    uint32_t rows,
    uint32_t cols,
    uint32_t blocks_per_row
) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows) {
        return;
    }

    float acc = 0.0f;

    for (uint32_t block_index = 0;
         block_index < blocks_per_row;
         ++block_index) {
        const unsigned char* block =
            weight
            + (static_cast<size_t>(row) * blocks_per_row
               + block_index) * AUGUR_Q4K_BLOCK_BYTES;

        const uint32_t input_offset =
            block_index * AUGUR_Q4K_DECODED_LEN;

        for (uint32_t lane = 0;
             lane < AUGUR_Q4K_DECODED_LEN;
             ++lane) {
            const float w = q4k_weight(block, lane);
            acc += w * input[input_offset + lane];
        }
    }

    output[row] = acc;
}

__global__ void augur_q4k_f32_prefill_kernel_v1(
    const unsigned char* weight,
    const float* input,
    float* output,
    uint32_t m,
    uint32_t rows,
    uint32_t cols,
    uint32_t blocks_per_row
) {
    const uint64_t linear =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t elements = static_cast<uint64_t>(m) * rows;
    if (linear >= elements) {
        return;
    }
    const uint32_t sample = static_cast<uint32_t>(linear / rows);
    const uint32_t row = static_cast<uint32_t>(linear % rows);
    float acc = 0.0f;
    for (uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const unsigned char* block =
            weight
            + (static_cast<size_t>(row) * blocks_per_row + block_index)
                * AUGUR_Q4K_BLOCK_BYTES;
        const uint64_t input_offset =
            static_cast<uint64_t>(sample) * cols
            + static_cast<uint64_t>(block_index) * AUGUR_Q4K_DECODED_LEN;
        for (uint32_t lane = 0; lane < AUGUR_Q4K_DECODED_LEN; ++lane) {
            acc += q4k_weight(block, lane) * input[input_offset + lane];
        }
    }
    output[linear] = acc;
}

static __device__ __forceinline__ float q5k_weight(
    const unsigned char* block, uint32_t lane
) {
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

static __device__ __forceinline__ float q6k_weight(
    const unsigned char* block, uint32_t lane
) {
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
        q = (block[low_base + 32 + q_lane] & 0x0f) | (((high >> 2) & 3) << 4);
    } else if (quarter == 2) {
        q = (block[low_base + q_lane] >> 4) | (((high >> 4) & 3) << 4);
    } else {
        q = (block[low_base + 32 + q_lane] >> 4) | (((high >> 6) & 3) << 4);
    }
    const int8_t scale = static_cast<int8_t>(
        block[192 + half * 8 + quarter * 2 + q_lane / 16]);
    const float d = half_bits_to_float(load_u16_le(block + 208));
    return d * static_cast<float>(scale) * (static_cast<int>(q) - 32);
}

template <uint32_t BlockBytes, float (*Weight)(const unsigned char*, uint32_t)>
__global__ void augur_qk_f32_decode_kernel_v1(
    const unsigned char* weight,
    const float* input,
    float* output,
    uint32_t rows,
    uint32_t blocks_per_row
) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float acc = 0.0f;
    for (uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const unsigned char* block = weight
            + (static_cast<size_t>(row) * blocks_per_row + block_index) * BlockBytes;
        const uint32_t input_offset = block_index * 256;
        for (uint32_t lane = 0; lane < 256; ++lane) {
            acc += Weight(block, lane) * input[input_offset + lane];
        }
    }
    output[row] = acc;
}

constexpr uint32_t AUGUR_Q5K_STAGED_BLOCK_BYTES = 192;
constexpr uint32_t AUGUR_Q5K_STAGED_BLOCK_VECTORS = AUGUR_Q5K_STAGED_BLOCK_BYTES / sizeof(uint4);
constexpr uint32_t AUGUR_Q5K_STAGED_THREADS = 128;
constexpr uint32_t AUGUR_Q5K_STAGED_WARPS = AUGUR_Q5K_STAGED_THREADS / 32;
constexpr uint32_t AUGUR_Q5K_STAGED_WORDS = 49;

static __device__ __forceinline__ float augur_q5k_dot_staged(
    const unsigned char* weight_row,
    const float* input,
    uint32_t* row_words,
    uint32_t blocks_per_row
) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks_per_row; ++b) {
        const uint4* source = reinterpret_cast<const uint4*>(
            weight_row + static_cast<size_t>(b) * AUGUR_Q5K_STAGED_BLOCK_BYTES);
        #pragma unroll
        for (uint32_t vector = 0; vector < AUGUR_Q5K_STAGED_BLOCK_VECTORS; ++vector) {
            const uint4 value = source[vector];
            row_words[vector * 4 + 0] = value.x;
            row_words[vector * 4 + 1] = value.y;
            row_words[vector * 4 + 2] = value.z;
            row_words[vector * 4 + 3] = value.w;
        }
        const unsigned char* block = reinterpret_cast<const unsigned char*>(row_words);
        for (uint32_t lane = 0; lane < 256; ++lane) {
            acc += q5k_weight(block, lane) * input[b * 256 + lane];
        }
    }
    return acc;
}

__global__ void augur_q5k_f32_decode_kernel_v1(
    const unsigned char* weight, const float* input, float* output,
    uint32_t rows, uint32_t blocks_per_row
) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t row = blockIdx.x * AUGUR_Q5K_STAGED_THREADS + threadIdx.x;
    __shared__ uint32_t staged[AUGUR_Q5K_STAGED_WARPS][32][AUGUR_Q5K_STAGED_WORDS];
    if (row < rows) {
        output[row] = augur_q5k_dot_staged(
            weight + static_cast<size_t>(row) * blocks_per_row * AUGUR_Q5K_STAGED_BLOCK_BYTES,
            input,
            staged[warp][lane],
            blocks_per_row);
    }
}

constexpr uint32_t AUGUR_Q5K_GRID32_THREADS = 32;
constexpr uint32_t AUGUR_Q5K_GRID32_WARPS = 1;

__global__ void augur_q5k_f32_decode_grid32_kernel_v2(
    const unsigned char* weight, const float* input, float* output,
    uint32_t rows, uint32_t blocks_per_row
) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x * AUGUR_Q5K_GRID32_THREADS + lane;
    __shared__ uint32_t staged[AUGUR_Q5K_GRID32_WARPS][32][AUGUR_Q5K_STAGED_WORDS];
    if (row < rows) {
        output[row] = augur_q5k_dot_staged(
            weight + static_cast<size_t>(row) * blocks_per_row * AUGUR_Q5K_STAGED_BLOCK_BYTES,
            input,
            staged[0][lane],
            blocks_per_row);
    }
}

#if defined(AUGUR_CUDA_M306_K22_Q5_FP32_KAHAN)
constexpr uint32_t AUGUR_M306_K22_Q5_THREADS = 128;
constexpr uint32_t AUGUR_M306_K22_Q5_WARPS = AUGUR_M306_K22_Q5_THREADS / 32;

static __device__ __forceinline__ void augur_m306_k22_two_sum(
    float left,
    float right,
    float* sum,
    float* residual
) {
    const float merged = left + right;
    const float right_virtual = merged - left;
    const float left_virtual = merged - right_virtual;
    *sum = merged;
    *residual = (left - left_virtual) + (right - right_virtual);
}

static __device__ __forceinline__ float augur_m306_k22_q5k_dot_warp_fp32_kahan(
    const unsigned char* weight_row,
    const float* input,
    uint32_t* block_words,
    uint32_t blocks_per_row,
    uint32_t lane
) {
    float sum = 0.0f;
    float correction = 0.0f;
    for (uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        if (lane < AUGUR_Q5K_STAGED_BLOCK_VECTORS) {
            const uint4 value = reinterpret_cast<const uint4*>(
                weight_row + static_cast<size_t>(block_index) * AUGUR_Q5K_STAGED_BLOCK_BYTES)[lane];
            block_words[lane * 4 + 0] = value.x;
            block_words[lane * 4 + 1] = value.y;
            block_words[lane * 4 + 2] = value.z;
            block_words[lane * 4 + 3] = value.w;
        }
        __syncwarp();
        const unsigned char* block = reinterpret_cast<const unsigned char*>(block_words);
        #pragma unroll
        for (uint32_t element = lane; element < 256; element += 32) {
            const float product = q5k_weight(block, element)
                * input[block_index * 256 + element];
            const float adjusted = product - correction;
            const float next = sum + adjusted;
            correction = (next - sum) - adjusted;
            sum = next;
        }
        __syncwarp();
    }
    float value = sum + correction;
    #pragma unroll
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(0xffffffffU, value, offset);
        if (lane < offset) {
            float merged;
            float residual;
            augur_m306_k22_two_sum(value, other, &merged, &residual);
            value = merged + residual;
        }
    }
    return value;
}

__global__ void augur_m306_k22_q5k_f32_decode_bounded_fp32_kahan_kernel_v1(
    const unsigned char* weight,
    const float* input,
    float* output,
    uint32_t rows,
    uint32_t blocks_per_row
) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t row = blockIdx.x * AUGUR_M306_K22_Q5_WARPS + warp;
    __shared__ uint32_t staged[AUGUR_M306_K22_Q5_WARPS][AUGUR_Q5K_STAGED_WORDS];
    if (row >= rows) return;
    const float value = augur_m306_k22_q5k_dot_warp_fp32_kahan(
        weight + static_cast<size_t>(row) * blocks_per_row * AUGUR_Q5K_STAGED_BLOCK_BYTES,
        input,
        staged[warp],
        blocks_per_row,
        lane);
    if (lane == 0) output[row] = value;
}
#endif

#if defined(AUGUR_CUDA_M306_K15_Q5_PAIRED_PROJECTION)
struct augur_q5k_f32x4 {
    float row0;
    float row1;
    float row2;
    float row3;
};

static __device__ __forceinline__ augur_q5k_f32x4 augur_q5k_dot_staged_batch4(
    const unsigned char* weight_row,
    const float* input0,
    const float* input1,
    const float* input2,
    const float* input3,
    uint32_t* row_words,
    uint32_t blocks_per_row
) {
    augur_q5k_f32x4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = 0; b < blocks_per_row; ++b) {
        const uint4* source = reinterpret_cast<const uint4*>(
            weight_row + static_cast<size_t>(b) * AUGUR_Q5K_STAGED_BLOCK_BYTES);
        #pragma unroll
        for (uint32_t vector = 0; vector < AUGUR_Q5K_STAGED_BLOCK_VECTORS; ++vector) {
            const uint4 value = source[vector];
            row_words[vector * 4 + 0] = value.x;
            row_words[vector * 4 + 1] = value.y;
            row_words[vector * 4 + 2] = value.z;
            row_words[vector * 4 + 3] = value.w;
        }
        const unsigned char* block = reinterpret_cast<const unsigned char*>(row_words);
        const uint32_t input_offset = b * 256;
        for (uint32_t lane = 0; lane < 256; ++lane) {
            const float weight = q5k_weight(block, lane);
            const uint32_t input_index = input_offset + lane;
            acc.row0 += weight * input0[input_index];
            acc.row1 += weight * input1[input_index];
            acc.row2 += weight * input2[input_index];
            acc.row3 += weight * input3[input_index];
        }
    }
    return acc;
}

__global__ void augur_q5k_f32_paired_projection_kernel_v1(
    const unsigned char* weight_w1,
    const unsigned char* weight_w3,
    const float* input,
    float* output_w1,
    float* output_w3,
    uint32_t rows,
    uint32_t blocks_per_row
) {
    const uint32_t projection = blockIdx.x / 16;
    const uint32_t projection_block = blockIdx.x % 16;
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t row = projection_block * AUGUR_Q5K_STAGED_THREADS + threadIdx.x;
    __shared__ uint32_t staged[AUGUR_Q5K_STAGED_WARPS][32][AUGUR_Q5K_STAGED_WORDS];
    if (row < rows) {
        const unsigned char* weight = projection == 0 ? weight_w1 : weight_w3;
        float* output = projection == 0 ? output_w1 : output_w3;
        output[row] = augur_q5k_dot_staged(
            weight + static_cast<size_t>(row) * blocks_per_row * AUGUR_Q5K_STAGED_BLOCK_BYTES,
            input,
            staged[warp][lane],
            blocks_per_row);
    }
}

__global__ void augur_batch01_b4_q5k_f32_paired_projection_kernel_v1(
    const unsigned char* weight_w1,
    const unsigned char* weight_w3,
    const float* input0,
    const float* input1,
    const float* input2,
    const float* input3,
    float* output_w1_0,
    float* output_w1_1,
    float* output_w1_2,
    float* output_w1_3,
    float* output_w3_0,
    float* output_w3_1,
    float* output_w3_2,
    float* output_w3_3,
    uint32_t rows,
    uint32_t blocks_per_row
) {
    const uint32_t projection = blockIdx.x / 16;
    const uint32_t projection_block = blockIdx.x % 16;
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t row = projection_block * AUGUR_Q5K_STAGED_THREADS + threadIdx.x;
    __shared__ uint32_t staged[AUGUR_Q5K_STAGED_WARPS][32][AUGUR_Q5K_STAGED_WORDS];
    if (row >= rows) return;

    const unsigned char* weight = projection == 0 ? weight_w1 : weight_w3;
    const augur_q5k_f32x4 value = augur_q5k_dot_staged_batch4(
        weight + static_cast<size_t>(row) * blocks_per_row * AUGUR_Q5K_STAGED_BLOCK_BYTES,
        input0,
        input1,
        input2,
        input3,
        staged[warp][lane],
        blocks_per_row);
    float* output0 = projection == 0 ? output_w1_0 : output_w3_0;
    float* output1 = projection == 0 ? output_w1_1 : output_w3_1;
    float* output2 = projection == 0 ? output_w1_2 : output_w3_2;
    float* output3 = projection == 0 ? output_w1_3 : output_w3_3;
    output0[row] = value.row0;
    output1[row] = value.row1;
    output2[row] = value.row2;
    output3[row] = value.row3;
}

#if defined(AUGUR_CUDA_BATCH01_B8_Q5_ROW_TILE)
constexpr uint32_t AUGUR_BATCH01_B8_Q5_SUBGROUP = 8;
constexpr uint32_t AUGUR_BATCH01_B8_Q5_ROWS_PER_WARP = 4;

__global__ void augur_batch01_b8_q5k_f32_paired_projection_row_tile_kernel_v1(
    augur_cuda_batch01_b8_q5k_row_tile_args_v1 args
) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t warp_lane = threadIdx.x % 32;
    const uint32_t row_in_warp = warp_lane / AUGUR_BATCH01_B8_Q5_SUBGROUP;
    const uint32_t batch_lane = warp_lane % AUGUR_BATCH01_B8_Q5_SUBGROUP;
    const uint32_t rows_per_block =
        AUGUR_Q5K_STAGED_WARPS * AUGUR_BATCH01_B8_Q5_ROWS_PER_WARP;
    const uint32_t blocks_per_projection =
        (args.rows + rows_per_block - 1) / rows_per_block;
    const uint32_t projection = blockIdx.x / blocks_per_projection;
    const uint32_t projection_block = blockIdx.x % blocks_per_projection;
    const uint32_t row = projection_block * rows_per_block
        + warp * AUGUR_BATCH01_B8_Q5_ROWS_PER_WARP
        + row_in_warp;

    __shared__ uint32_t
        staged[AUGUR_Q5K_STAGED_WARPS][AUGUR_BATCH01_B8_Q5_ROWS_PER_WARP]
              [AUGUR_Q5K_STAGED_WORDS];
    if (row >= args.rows) return;

    const uint64_t weight_address = projection == 0
        ? args.weight_w1
        : args.weight_w3;
    const auto* weight = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(weight_address));
    const auto* input = reinterpret_cast<const float*>(
        static_cast<uintptr_t>(args.input[batch_lane]));
    float acc = 0.0f;

    for (uint32_t block_index = 0;
         block_index < args.blocks_per_row;
         ++block_index) {
        const auto* source = reinterpret_cast<const uint4*>(
            weight
            + (static_cast<size_t>(row) * args.blocks_per_row + block_index)
                * AUGUR_Q5K_STAGED_BLOCK_BYTES);
        uint32_t* row_words = staged[warp][row_in_warp];
        for (uint32_t vector = batch_lane;
             vector < AUGUR_Q5K_STAGED_BLOCK_VECTORS;
             vector += AUGUR_BATCH01_B8_Q5_SUBGROUP) {
            const uint4 value = source[vector];
            row_words[vector * 4 + 0] = value.x;
            row_words[vector * 4 + 1] = value.y;
            row_words[vector * 4 + 2] = value.z;
            row_words[vector * 4 + 3] = value.w;
        }
        __syncwarp();

        const auto* block =
            reinterpret_cast<const unsigned char*>(row_words);
        const uint32_t input_offset = block_index * 256;
        for (uint32_t element = 0; element < 256; ++element) {
            float input_value = row_in_warp == 0
                ? input[input_offset + element]
                : 0.0f;
            input_value = __shfl_sync(
                0xffffffffU, input_value, batch_lane);
            float decoded = batch_lane == 0
                ? q5k_weight(block, element)
                : 0.0f;
            decoded = __shfl_sync(0xffffffffU, decoded, 0, 8);
            acc += decoded * input_value;
        }
        __syncwarp();
    }

    const uint64_t output_address = projection == 0
        ? args.output_w1[batch_lane]
        : args.output_w3[batch_lane];
    auto* output = reinterpret_cast<float*>(
        static_cast<uintptr_t>(output_address));
    output[row] = acc;
}

__global__ void augur_batch01_b8_q5k_f32_paired_projection_row_tile_kernel_v2(
    augur_cuda_batch01_b8_q5k_row_tile_args_v1 args
) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t warp_lane = threadIdx.x % 32;
    const uint32_t row_in_warp = warp_lane / AUGUR_BATCH01_B8_Q5_SUBGROUP;
    const uint32_t batch_lane = warp_lane % AUGUR_BATCH01_B8_Q5_SUBGROUP;
    const uint32_t rows_per_block =
        AUGUR_Q5K_STAGED_WARPS * AUGUR_BATCH01_B8_Q5_ROWS_PER_WARP;
    const uint32_t blocks_per_projection = args.rows / rows_per_block;
    const uint32_t projection = blockIdx.x / blocks_per_projection;
    const uint32_t projection_block = blockIdx.x % blocks_per_projection;
    const uint32_t row = projection_block * rows_per_block
        + warp * AUGUR_BATCH01_B8_Q5_ROWS_PER_WARP
        + row_in_warp;

    __shared__ uint32_t
        staged[AUGUR_Q5K_STAGED_WARPS][AUGUR_BATCH01_B8_Q5_ROWS_PER_WARP]
              [AUGUR_Q5K_STAGED_WORDS];

    const uint64_t weight_address = projection == 0
        ? args.weight_w1
        : args.weight_w3;
    const auto* weight = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(weight_address));
    const auto* input = reinterpret_cast<const float*>(
        static_cast<uintptr_t>(args.input[batch_lane]));
    float acc = 0.0f;

    for (uint32_t block_index = 0;
         block_index < args.blocks_per_row;
         ++block_index) {
        const auto* source = reinterpret_cast<const uint4*>(
            weight
            + (static_cast<size_t>(row) * args.blocks_per_row + block_index)
                * AUGUR_Q5K_STAGED_BLOCK_BYTES);
        uint32_t* row_words = staged[warp][row_in_warp];
        for (uint32_t vector = batch_lane;
             vector < AUGUR_Q5K_STAGED_BLOCK_VECTORS;
             vector += AUGUR_BATCH01_B8_Q5_SUBGROUP) {
            const uint4 value = source[vector];
            row_words[vector * 4 + 0] = value.x;
            row_words[vector * 4 + 1] = value.y;
            row_words[vector * 4 + 2] = value.z;
            row_words[vector * 4 + 3] = value.w;
        }
        __syncwarp();

        const auto* block =
            reinterpret_cast<const unsigned char*>(row_words);
        const uint32_t input_offset = block_index * 256;
        for (uint32_t element_base = 0; element_base < 256; element_base += 8) {
            const float decoded = q5k_weight(block, element_base + batch_lane);
            #pragma unroll
            for (uint32_t source_lane = 0; source_lane < 8; ++source_lane) {
                const float weight_value = __shfl_sync(
                    0xffffffffU, decoded, source_lane, 8);
                const float input_value =
                    input[input_offset + element_base + source_lane];
                acc += weight_value * input_value;
            }
        }
        __syncwarp();
    }

    const uint64_t output_address = projection == 0
        ? args.output_w1[batch_lane]
        : args.output_w3[batch_lane];
    auto* output = reinterpret_cast<float*>(
        static_cast<uintptr_t>(output_address));
    output[row] = acc;
}
#endif
#endif

__global__ void augur_m277_q5k_six_expert_down_bf16_kernel_v1(
    const unsigned char* weight_0,
    const unsigned char* weight_1,
    const unsigned char* weight_2,
    const unsigned char* weight_3,
    const unsigned char* weight_4,
    const unsigned char* weight_5,
    const float* input,
    float* output
) {
    const uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= 6 * 4096) return;
    const uint32_t expert = linear / 4096;
    const uint32_t row = linear % 4096;
    const unsigned char* weights[6] = {
        weight_0, weight_1, weight_2, weight_3, weight_4, weight_5};
    const unsigned char* weight = weights[expert];
    const float* expert_input = input + static_cast<size_t>(expert) * 2 * 2048;
    float acc = 0.0f;
    for (uint32_t b = 0; b < 8; ++b) {
        const unsigned char* block = weight + (static_cast<size_t>(row) * 8 + b) * 192;
        for (uint32_t lane = 0; lane < 256; ++lane) {
            acc += q5k_weight(block, lane) * expert_input[b * 256 + lane];
        }
    }
    uint32_t bits = __float_as_uint(acc);
    bits += UINT32_C(0x7fff) + ((bits >> 16) & UINT32_C(1));
    output[linear] = __uint_as_float(bits & UINT32_C(0xffff0000));
}

constexpr uint32_t AUGUR_Q6K_STAGED_BLOCK_BYTES = 224;
constexpr uint32_t AUGUR_Q6K_STAGED_BLOCK_VECTORS = AUGUR_Q6K_STAGED_BLOCK_BYTES / sizeof(uint4);
constexpr uint32_t AUGUR_Q6K_STAGED_THREADS = 128;
constexpr uint32_t AUGUR_Q6K_STAGED_WARPS = AUGUR_Q6K_STAGED_THREADS / 32;
constexpr uint32_t AUGUR_Q6K_STAGED_WORDS = 57;

static __device__ __forceinline__ float augur_q6k_dot_staged(
    const unsigned char* weight_row,
    const float* input,
    uint32_t* row_words,
    uint32_t blocks_per_row
) {
    float acc = 0.0f;
    for (uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto* source = reinterpret_cast<const uint4*>(
            weight_row + static_cast<size_t>(block_index) * AUGUR_Q6K_STAGED_BLOCK_BYTES);
        #pragma unroll
        for (uint32_t vector = 0; vector < AUGUR_Q6K_STAGED_BLOCK_VECTORS; ++vector) {
            const uint4 value = source[vector];
            row_words[vector * 4 + 0] = value.x;
            row_words[vector * 4 + 1] = value.y;
            row_words[vector * 4 + 2] = value.z;
            row_words[vector * 4 + 3] = value.w;
        }
        const auto* block = reinterpret_cast<const unsigned char*>(row_words);
        for (uint32_t lane = 0; lane < 256; ++lane) {
            acc += q6k_weight(block, lane) * input[block_index * 256 + lane];
        }
    }
    return acc;
}

__global__ void augur_q6k_f32_decode_kernel_v1(
    const unsigned char* weight, const float* input, float* output,
    uint32_t rows, uint32_t blocks_per_row
) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t row = blockIdx.x * AUGUR_Q6K_STAGED_THREADS + threadIdx.x;
    __shared__ uint32_t staged[AUGUR_Q6K_STAGED_WARPS][32][AUGUR_Q6K_STAGED_WORDS];
    if (row < rows) {
        output[row] = augur_q6k_dot_staged(
            weight + static_cast<size_t>(row) * blocks_per_row * AUGUR_Q6K_STAGED_BLOCK_BYTES,
            input,
            staged[warp][lane],
            blocks_per_row);
    }
}
