// Default-off B128 NVFP4 FFN candidate; exact integer-MMA/scale/reduction.
// Kernel bodies lifted from the source-bound vector-all component. Ownership
// and bounded admission belong to the native preflight, never to this helper.
// One CTA owns a16x16 output tile and all32 canonical reduction lanes.
// Four native K partitions are summed in their original order; no global
// partial planes or separately launched reduction are needed.
#pragma once
namespace augur_prefill_integer_mma {
// PTX ISA8.7 section9.7.14.5.1 defines m8n8k4 f16/f32 lane fragments.
// Four independent8x8 products are assigned to one16x16 output tile.
__device__ __forceinline__ unsigned mma_row(unsigned lane,unsigned i) {
    return ((lane >> 2U) & 1U) * 8U + (lane & 1U) + (i & 2U) + (lane >= 16U ? 4U : 0U);
}
__device__ __forceinline__ unsigned mma_col(unsigned lane,unsigned i) {
    return ((lane >> 3U) & 1U) * 8U + (i & 4U) + (lane & 2U) + (i & 1U);
}
constexpr unsigned tile_rows = 384;
inline size_t scratch_bytes(unsigned tokens, unsigned format) {
    return format == 2U ? size_t(tile_rows + tokens) * (17408U / 16U) * 36U : 0U;
}

__device__ __forceinline__ uint2 nvfp4_half_registers(uint32_t packed,unsigned shift) {
    const unsigned n = (packed >> shift) & 0x0f0f0f0fU;
    const unsigned pairs = (n & 0x000f000fU) | ((n >> 4U) & 0x00f000f0U);
    const unsigned selector = (pairs | (pairs >> 8U)) & 0xffffU;
    const unsigned high_bytes = __byte_perm(0x42403c00U,0x4a484644U,selector) | ((n & 0x08080808U) << 4U);
    return make_uint2(__byte_perm(0U,high_bytes,0x5140U),__byte_perm(0U,high_bytes,0x7160U));
}

__global__ void pack_weight_vector(const uint8_t* source,__half* packed,float* scales,
    unsigned rows,unsigned cols,unsigned first,unsigned work_rows) {
    const unsigned quanta=cols/16U;
    const size_t total=size_t((work_rows+15U)/16U)*quanta*16U;
    for(size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;index<total;index+=size_t(blockDim.x)*gridDim.x) {
        const unsigned m=index%16U,q=(index/16U)%quanta;
        const unsigned row=first+unsigned(index/(16U*quanta))*16U+m;
        uint4 low=make_uint4(0,0,0,0),high=low;float scale=0;
        if(row<rows && row-first<work_rows) {
            const uint8_t* block=source+(size_t(row)*(cols/64U)+q/4U)*36U;
            const auto* words=reinterpret_cast<const uint32_t*>(block+4U+(q%4U)*8U);
            const uint32_t w0=words[0],w1=words[1];
            const uint2 l0=nvfp4_half_registers(w0,0),l1=nvfp4_half_registers(w1,0);
            const uint2 h0=nvfp4_half_registers(w0,4),h1=nvfp4_half_registers(w1,4);
            low=make_uint4(l0.x,l0.y,l1.x,l1.y);high=make_uint4(h0.x,h0.y,h1.x,h1.y);
            scale=ue4m3_scale_to_f32(block[q%4U])*0.5f;
        }
        auto* dest=reinterpret_cast<uint4*>(packed+(index/16U)*256U+m*16U);
        dest[0]=low;dest[1]=high;scales[index]=scale;
    }
}

__global__ void pack_input_vector(const uint8_t* source,__half* packed,float* scales,unsigned tokens,unsigned cols) {
    const unsigned quanta=cols/16U;
    const size_t total=size_t((tokens+15U)/16U)*quanta*16U;
    for(size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;index<total;index+=size_t(blockDim.x)*gridDim.x) {
        const unsigned n=index%16U,q=(index/16U)%quanta;
        const unsigned token=unsigned(index/(16U*quanta))*16U+n;
        uint32_t h[8]={};float scale=0;
        if(token<tokens) {
            const uint8_t* block=source+(size_t(token)*(cols/32U)+q/2U)*36U;
            const auto* words=reinterpret_cast<const uint32_t*>(block+4U+(q%2U)*16U);
            #pragma unroll
            for(unsigned w=0;w<4U;++w) {
                const uint32_t codes=words[w];
                #pragma unroll
                for(unsigned pair=0;pair<2U;++pair) {
                    const float lo=float(static_cast<int8_t>(codes>>(pair*16U)));
                    const float hi=float(static_cast<int8_t>(codes>>(pair*16U+8U)));
                    h[w*2U+pair]=unsigned(__half_as_ushort(__float2half_rn(lo)))
                        | (unsigned(__half_as_ushort(__float2half_rn(hi)))<<16U);
                }
            }
            scale=__half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(block)));
        }
        auto* dest=reinterpret_cast<uint4*>(packed+(index/16U)*256U+n*16U);
        dest[0]=make_uint4(h[0],h[1],h[2],h[3]);dest[1]=make_uint4(h[4],h[5],h[6],h[7]);
        scales[index]=scale;
    }
}

// Keep the independently measured B128 vector-loading kernel unchanged.
__global__ __launch_bounds__(1024,1) void project_small(const __half* weight, const __half* input,
    const float* weight_scales, const float* input_scales, const uint8_t* raw_weight, float* output,
    unsigned rows, unsigned cols, unsigned tokens, unsigned first_row, unsigned work_rows, float tensor_scale) {
    __shared__ __align__(32) float shared[32U * 257U];
    const unsigned warp = threadIdx.x / 32U, lane = threadIdx.x & 31U, quanta = cols / 16U;
    const unsigned row0 = blockIdx.x * 16U, token0 = blockIdx.y * 16U;
    float combined[8] = {};

    #pragma unroll
    for (unsigned partition = 0; partition < 4U; ++partition) {
        const unsigned reference_thread = warp + partition * 32U;
        float partial[8] = {};
        for (unsigned block = reference_thread / 2U; block < cols / 64U; block += 64U) {
            #pragma unroll
            for (unsigned pair = 0; pair < 2U; ++pair) {
                const unsigned q = block * 4U + (reference_thread & 1U) * 2U + pair;
                const size_t wt = size_t(blockIdx.x) * quanta + q;
                const size_t at = size_t(blockIdx.y) * quanta + q;
                float result[8] = {};
                const unsigned arow = ((lane >> 2U) & 1U) * 8U + lane % 4U + (lane >= 16U ? 4U : 0U);
                const unsigned bcol = ((lane >> 3U) & 1U) * 8U + lane % 4U + (lane >= 16U ? 4U : 0U);
                // Load the current quantum in aligned vectors. MMA, scale/FMA,
                // partition and reduction order remain canonical; no lookahead
                // quantum or extra scratch is introduced.
                uint2 frag_a[4], frag_b[4];
                #pragma unroll
                for (unsigned f = 0; f < 4U; f += 2U) {
                    const uint4 va = *reinterpret_cast<const uint4*>(weight + wt * 256U + arow * 16U + f * 4U);
                    const uint4 vb = *reinterpret_cast<const uint4*>(input + at * 256U + bcol * 16U + f * 4U);
                    frag_a[f] = make_uint2(va.x, va.y);
                    frag_a[f + 1U] = make_uint2(va.z, va.w);
                    frag_b[f] = make_uint2(vb.x, vb.y);
                    frag_b[f + 1U] = make_uint2(vb.z, vb.w);
                }
                #pragma unroll
                for (unsigned k = 0; k < 16U; k += 4U) {
                    const uint2 a = frag_a[k / 4U];
                    const uint2 b = frag_b[k / 4U];
                    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
                        : "+f"(result[0]),"+f"(result[1]),"+f"(result[2]),"+f"(result[3]),
                          "+f"(result[4]),"+f"(result[5]),"+f"(result[6]),"+f"(result[7])
                        : "r"(a.x),"r"(a.y),"r"(b.x),"r"(b.y));
                }
                #pragma unroll
                for (unsigned i = 0; i < 8U; ++i) {
                    const float scale = weight_scales[wt * 16U + mma_row(lane,i)]
                        * input_scales[at * 16U + mma_col(lane,i)];
                    partial[i] = fmaf(scale,result[i],partial[i]);
                }
            }
        }
        #pragma unroll
        for (unsigned i = 0; i < 8U; ++i) {
            if (partition == 0U) combined[i] = partial[i];
            else combined[i] += partial[i];
        }
    }
    // Cross-warp partial read uses an odd shared stride.
    __syncthreads();
    #pragma unroll
    for (unsigned i = 0; i < 8U; ++i) shared[warp * 257U + lane + i * 32U] = combined[i];
    __syncthreads();
    #pragma unroll
    for (unsigned i = 0; i < 8U; ++i) {
        const unsigned index = warp + i * 32U;
        float value = shared[lane * 257U + index];
        #pragma unroll
        for (unsigned offset = 16U; offset; offset >>= 1U)
            value += __shfl_down_sync(0xffffffffU,value,offset);
        const unsigned row = row0 + mma_row(warp,i), token = token0 + mma_col(warp,i);
        if (lane == 0U && row < work_rows && token < tokens)
            output[size_t(token) * rows + first_row + row] = value * tensor_scale;
    }
}

template<unsigned Warps,unsigned MinBlocks>
__global__ __launch_bounds__(Warps*32U,MinBlocks) void project(const __half* weight, const __half* input,
    const float* weight_scales, const float* input_scales, const uint8_t* raw_weight, float* output,
    unsigned rows, unsigned cols, unsigned tokens, unsigned first_row, unsigned work_rows, float tensor_scale) {
    __shared__ __align__(32) float shared[32U * 257U];
    const unsigned physical_warp = threadIdx.x / 32U, lane = threadIdx.x & 31U, quanta = cols / 16U;
    const unsigned row0 = blockIdx.x * 16U, token0 = blockIdx.y * 16U;
    #pragma unroll
    for (unsigned warp=physical_warp; warp<32U; warp+=Warps) {
        float combined[8] = {};

        #pragma unroll
        for (unsigned partition = 0; partition < 4U; ++partition) {
            const unsigned reference_thread = warp + partition * 32U;
            float partial[8] = {};
            for (unsigned block = reference_thread / 2U; block < cols / 64U; block += 64U) {
                #pragma unroll
                for (unsigned pair = 0; pair < 2U; ++pair) {
                    const unsigned q = block * 4U + (reference_thread & 1U) * 2U + pair;
                    const size_t wt = size_t(blockIdx.x) * quanta + q;
                    const size_t at = size_t(blockIdx.y) * quanta + q;
                    float result[8] = {};
                    const unsigned arow = ((lane >> 2U) & 1U) * 8U + lane % 4U + (lane >= 16U ? 4U : 0U);
                    const unsigned bcol = ((lane >> 3U) & 1U) * 8U + lane % 4U + (lane >= 16U ? 4U : 0U);
                    // Load the current quantum in aligned vectors. MMA, scale/FMA,
                    // partition and reduction order remain canonical; no lookahead
                    // quantum or extra scratch is introduced.
                    uint2 frag_a[4], frag_b[4];
                    #pragma unroll
                    for (unsigned f = 0; f < 4U; f += 2U) {
                        const uint4 va = *reinterpret_cast<const uint4*>(weight + wt * 256U + arow * 16U + f * 4U);
                        const uint4 vb = *reinterpret_cast<const uint4*>(input + at * 256U + bcol * 16U + f * 4U);
                        frag_a[f] = make_uint2(va.x, va.y);
                        frag_a[f + 1U] = make_uint2(va.z, va.w);
                        frag_b[f] = make_uint2(vb.x, vb.y);
                        frag_b[f + 1U] = make_uint2(vb.z, vb.w);
                    }
                    #pragma unroll
                    for (unsigned k = 0; k < 16U; k += 4U) {
                        const uint2 a = frag_a[k / 4U];
                        const uint2 b = frag_b[k / 4U];
                        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
                            : "+f"(result[0]),"+f"(result[1]),"+f"(result[2]),"+f"(result[3]),
                              "+f"(result[4]),"+f"(result[5]),"+f"(result[6]),"+f"(result[7])
                            : "r"(a.x),"r"(a.y),"r"(b.x),"r"(b.y));
                    }
                    #pragma unroll
                    for (unsigned i = 0; i < 8U; ++i) {
                        const float scale = weight_scales[wt * 16U + mma_row(lane,i)]
                            * input_scales[at * 16U + mma_col(lane,i)];
                        partial[i] = fmaf(scale,result[i],partial[i]);
                    }
                }
            }
            #pragma unroll
            for (unsigned i = 0; i < 8U; ++i) {
                if (partition == 0U) combined[i] = partial[i];
                else combined[i] += partial[i];
            }
        }
        // Cross-warp partial read uses an odd shared stride.

        #pragma unroll
        for (unsigned i = 0; i < 8U; ++i) shared[warp * 257U + lane + i * 32U] = combined[i];

    }
    __syncthreads();
    #pragma unroll
    for (unsigned warp=physical_warp; warp<32U; warp+=Warps) {

        #pragma unroll
        for (unsigned i = 0; i < 8U; ++i) {
            const unsigned index = warp + i * 32U;
            float value = shared[lane * 257U + index];
            #pragma unroll
            for (unsigned offset = 16U; offset; offset >>= 1U)
                value += __shfl_down_sync(0xffffffffU,value,offset);
            const unsigned row = row0 + mma_row(warp,i), token = token0 + mma_col(warp,i);
            if (lane == 0U && row < work_rows && token < tokens)
                output[size_t(token) * rows + first_row + row] = value * tensor_scale;
        }

    }
}

inline cudaError_t launch_prevalidated(cudaStream_t stream,const uint8_t* weight,const uint8_t* activation,
    float* scratch,float* output,unsigned tokens,unsigned rows,unsigned cols,float scale) {
    // The native admission below this helper validates shape, views and owner.
    const unsigned quanta = cols / 16U;
    auto* packed_weight = reinterpret_cast<__half*>(scratch);
    float* weight_scales = reinterpret_cast<float*>(packed_weight + size_t(tile_rows) * quanta * 16U);
    auto* packed_input = reinterpret_cast<__half*>(weight_scales + size_t(tile_rows) * quanta);
    float* input_scales = reinterpret_cast<float*>(packed_input + size_t(tokens) * quanta * 16U);
    pack_input_vector<<<launch_blocks(size_t(tokens)*quanta),256,0,stream>>>(activation,packed_input,input_scales,tokens,cols);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    for (unsigned first = 0; first < rows; first += tile_rows) {
        const unsigned count = std::min(tile_rows,rows-first);
        pack_weight_vector<<<launch_blocks(size_t(count)*quanta),256,0,stream>>>(weight,packed_weight,weight_scales,rows,cols,first,count);
        error = cudaGetLastError();
        if (error != cudaSuccess) return error;
        if (tokens == 128U) {
            project_small<<<dim3((count+15U)/16U,tokens/16U),1024,0,stream>>>(packed_weight,packed_input,
                weight_scales,input_scales,weight,output,rows,cols,tokens,first,count,scale);
        } else {
            project<8U,2U><<<dim3((count+15U)/16U,tokens/16U),256U,0,stream>>>(packed_weight,packed_input,
                weight_scales,input_scales,weight,output,rows,cols,tokens,first,count,scale);
        }
        error = cudaGetLastError();
        if (error != cudaSuccess) return error;
    }
    return cudaSuccess;
}
}
