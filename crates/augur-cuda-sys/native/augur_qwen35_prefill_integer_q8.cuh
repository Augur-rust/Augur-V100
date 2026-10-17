// Default-off, measured B256/B512 Q8 FFN; canonical eight-code MMA only.
// Q8_0 x Q8_1: exact eight-code integer dots; original128-thread FMA tree.
// Independent component candidate. NVFP4 delegates to the measured vector MMA.
#pragma once
#include "augur_qwen35_prefill_integer_mma.cuh"
namespace augur_prefill_integer_q8 {
using augur_prefill_integer_mma::mma_row;
using augur_prefill_integer_mma::mma_col;
constexpr unsigned tile_rows = 384U;
constexpr unsigned dot_codes = 8U;
constexpr unsigned partitions = 4U;
inline size_t scratch_bytes(unsigned tokens, unsigned format) {
    return format == 1U ? size_t(tile_rows + tokens) * (17408U / dot_codes) * (dot_codes * 2U + 4U)
                        : augur_prefill_integer_mma::scratch_bytes(tokens, format);
}

template<bool Weight>
__global__ void pack(const uint8_t* source, __half* codes, float* scales,
    unsigned rows, unsigned cols, unsigned first, unsigned work_rows) {
    const unsigned quanta = cols / dot_codes;
    const size_t total = size_t((work_rows + 15U) / 16U) * quanta * 16U;
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += size_t(blockDim.x) * gridDim.x) {
        const unsigned m = index % 16U, q = (index / 16U) % quanta;
        const unsigned row = first + unsigned(index / (16U * quanta)) * 16U + m;
        uint32_t h[dot_codes / 2U] = {}; float scale = 0;
        if (row < rows && row - first < work_rows) {
            const uint8_t* block = source + (size_t(row) * (cols / 32U) + q / (32U / dot_codes)) * (Weight ? 34U : 36U);
            // Q8_0 has only two-byte alignment. Never cast its code start to uint32_t.
            const auto* words = reinterpret_cast<const uint16_t*>(block + (Weight ? 2U : 4U) + (q % (32U / dot_codes)) * dot_codes);
            #pragma unroll
            for (unsigned pair = 0; pair < dot_codes / 2U; ++pair) {
                const uint16_t v = words[pair];
                const __half lo = __float2half_rn(float(static_cast<int8_t>(v)));
                const __half hi = __float2half_rn(float(static_cast<int8_t>(v >> 8U)));
                h[pair] = unsigned(__half_as_ushort(lo)) | (unsigned(__half_as_ushort(hi)) << 16U);
            }
            scale = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(block)));
        }
        auto* dest = reinterpret_cast<uint4*>(codes + (index / 16U) * (16U * dot_codes) + m * dot_codes);
        #pragma unroll
        for (unsigned v = 0; v < dot_codes / 8U; ++v)
            dest[v] = make_uint4(h[v*4U],h[v*4U+1U],h[v*4U+2U],h[v*4U+3U]);
        scales[index] = scale;
    }
}

template<unsigned Warps,unsigned MinBlocks>
__global__ __launch_bounds__(Warps*32U,MinBlocks) void project(
    const __half* weight, const __half* input, const float* ws, const float* as,
    float* output, unsigned rows, unsigned cols, unsigned tokens, unsigned first, unsigned work_rows) {
    __shared__ __align__(32) float shared[32U * 257U];
    const unsigned physical_warp = threadIdx.x / 32U, lane = threadIdx.x & 31U;
    const unsigned quanta = cols / dot_codes, row0 = blockIdx.x * 16U, token0 = blockIdx.y * 16U;
    #pragma unroll
    for (unsigned warp=physical_warp; warp<32U; warp+=Warps) {
        float combined[8] = {};
        #pragma unroll
        for (unsigned partition = 0; partition < partitions; ++partition) {
            const unsigned reference_thread = warp + partition * 32U;
            float partial[8] = {};
            for (unsigned block = reference_thread / (32U / dot_codes); block < cols / 32U; block += 32U) {
                const unsigned q = block * (32U / dot_codes) + reference_thread % (32U / dot_codes);
                const size_t wt = size_t(blockIdx.x) * quanta + q;
                const size_t at = size_t(blockIdx.y) * quanta + q;
                float result[8] = {};
                const unsigned arow = ((lane >> 2U) & 1U) * 8U + lane % 4U + (lane >= 16U ? 4U : 0U);
                const unsigned bcol = ((lane >> 3U) & 1U) * 8U + lane % 4U + (lane >= 16U ? 4U : 0U);
                #pragma unroll
                for (unsigned k = 0; k < dot_codes; k += 4U) {
                    const uint2 a = *reinterpret_cast<const uint2*>(weight + wt * (16U * dot_codes) + arow * dot_codes + k);
                    const uint2 b = *reinterpret_cast<const uint2*>(input + at * (16U * dot_codes) + bcol * dot_codes + k);
                    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
                        : "+f"(result[0]),"+f"(result[1]),"+f"(result[2]),"+f"(result[3]),
                          "+f"(result[4]),"+f"(result[5]),"+f"(result[6]),"+f"(result[7])
                        : "r"(a.x),"r"(a.y),"r"(b.x),"r"(b.y));
                }
                #pragma unroll
                for (unsigned i = 0; i < 8U; ++i) {
                    const float scale = ws[wt * 16U + mma_row(lane,i)] * as[at * 16U + mma_col(lane,i)];
                    partial[i] = fmaf(scale, result[i], partial[i]);
                }
            }
            #pragma unroll
            for (unsigned i = 0; i < 8U; ++i) {
                if (partition == 0U) combined[i] = partial[i];
                else combined[i] += partial[i];
            }
        }

        #pragma unroll
        for (unsigned i = 0; i < 8U; ++i) shared[warp * 257U + lane + i * 32U] = combined[i];

    }
    __syncthreads();
    #pragma unroll
    for (unsigned warp=physical_warp; warp<32U; warp+=Warps) {

        #pragma unroll
        for (unsigned i = 0; i < 8U; ++i) {
            float value = shared[lane * 257U + warp + i * 32U];
            #pragma unroll
            for (unsigned offset = 16U; offset; offset >>= 1U)
                value += __shfl_down_sync(0xffffffffU, value, offset);
            const unsigned row = row0 + mma_row(warp,i), token = token0 + mma_col(warp,i);
            if (lane == 0U && row < work_rows && token < tokens)
                output[size_t(token) * rows + first + row] = value;
        }

    }
}

inline cudaError_t launch_prevalidated(cudaStream_t stream, const uint8_t* weight,
    const uint8_t* activation, float* scratch, float* output,
    unsigned tokens, unsigned rows, unsigned cols) {
    // Native ABI admission has already validated Q8 scale1, shape and views.
    const unsigned quanta = cols / dot_codes;
    auto* packed_weight = reinterpret_cast<__half*>(scratch);
    float* ws = reinterpret_cast<float*>(packed_weight + size_t(tile_rows) * quanta * dot_codes);
    auto* packed_input = reinterpret_cast<__half*>(ws + size_t(tile_rows) * quanta);
    float* as = reinterpret_cast<float*>(packed_input + size_t(tokens) * quanta * dot_codes);
    pack<false><<<launch_blocks(size_t(tokens)*quanta),256,0,stream>>>(activation,packed_input,as,tokens,cols,0,tokens);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    for (unsigned first = 0; first < rows; first += tile_rows) {
        const unsigned count = std::min(tile_rows, rows - first);
        pack<true><<<launch_blocks(size_t(count)*quanta),256,0,stream>>>(weight,packed_weight,ws,rows,cols,first,count);
        error = cudaGetLastError();
        if (error != cudaSuccess) return error;
        project<16U,1U><<<dim3((count+15U)/16U,tokens/16U),512U,0,stream>>>(packed_weight,packed_input,ws,as,output,rows,cols,tokens,first,count);
        error = cudaGetLastError();
        if (error != cudaSuccess) return error;
    }
    return cudaSuccess;
}
}
