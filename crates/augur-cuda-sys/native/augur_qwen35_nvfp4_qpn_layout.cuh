// SPDX-License-Identifier: Apache-2.0
#pragma once

// Exact GGML36 <-> QPN geometry, following NInfer's Apache-2.0
// nvfp4_prepack_sm70.cu. This adapter is independent of its GEMM arithmetic.
// One contiguous allocation: N*K/2 code bytes, then N*K/16 scale bytes.
// Callers admit N%32=0, K%64=0, disjoint source/destination and finite scales.
// New typed dispatch is required: a legacy GGML kernel must never see QPN.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace augur_qwen35_qpn {

// Pack once per activation producer, then share between gate/up consumers.
// Converting redundantly in each QPN CTA is numerically valid but much slower.
// The checked caller must provide source alignment8 and destination alignment4;
// QPN consumers additionally require the retained FP16 base to be aligned16.
__global__ void stage_activation_f16(const float* source,half* destination,uint64_t elements) {
    const uint64_t first=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)*2U;
    if(first>=elements) return;
    if(first+1U<elements) {
        const float2 value=reinterpret_cast<const float2*>(source)[first/2U];
        reinterpret_cast<half2*>(destination)[first/2U]=__floats2half2_rn(value.x,value.y);
    } else destination[first]=__float2half_rn(source[first]);
}

// Explicit compensated profile: x = hi + 2^-12 * lo + residual.
// Scaling the low term avoids FP16 underflow for normal model activations.
// Finite residual encoding saturates at65504: at x=32784 the otherwise valid
// hi=32768 leaves16*4096=65536, which would round lo to infinity. Saturating
// ONLY this finite low component retains the stated decomposition bound.
// Original activation/hi overflow and nonfinite inputs still propagate.
__global__ void stage_activation_residual_f16(const float* source, half* destination, uint64_t elements) {
    const uint64_t first=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)*2U;
    if(first>=elements) return;
    #pragma unroll
    for(int j=0;j<2;++j) if(first+j<elements) {
        const uint64_t i=first+j;
        const float x=source[i];
        const half hi=__float2half_rn(x);
        destination[i]=hi;
        float residual=__fmul_rn(__fsub_rn(x,__half2float(hi)),4096.0f);
        if(isfinite(residual)) residual=fminf(65504.0f,fmaxf(-65504.0f,residual));
        destination[elements+i]=__float2half_rn(residual);
    }
}

__device__ __forceinline__ uint64_t group_index(uint32_t row, uint32_t group, uint32_t cols) {
    const uint32_t r=row&31U;
    const uint32_t lane=(r/8U)*4U+(r&3U)+(r&4U)*4U;
    return (uint64_t(row/32U)*(cols/16U)+group)*32U+lane;
}

__global__ void prepack_ggml(const uint8_t* source, uint8_t* destination, uint32_t rows, uint32_t cols) {
    const uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const uint32_t groups=cols/16U;
    if (index>=uint64_t(rows)*groups) return;
    const uint32_t lane=index%32U, group=(index/32U)%groups, tile=index/(uint64_t(groups)*32U);
    const uint32_t row=tile*32U+((lane>>2U)&3U)*8U+(lane&3U)+((lane&16U)?4U:0U);
    const uint8_t* block=source+(uint64_t(row)*(cols/64U)+group/4U)*36U;
    const uint8_t* codes=block+4U+(group%4U)*8U;
    constexpr uint32_t order[16]={0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15};
    #pragma unroll
    for (uint32_t byte=0;byte<8;++byte) {
        const uint32_t a=order[byte*2U],b=order[byte*2U+1U];
        destination[index*8U+byte]=((codes[a%8U]>>((a/8U)*4U))&15U)
            | (((codes[b%8U]>>((b/8U)*4U))&15U)<<4U);
    }
    destination[uint64_t(rows)*cols/2U+index]=block[group%4U];
}

// Original checkpoint codes followed by their untouched scale plane. This
// removes only the CPU GGML intermediate; the resulting QPN bytes are identical.
__global__ void prepack_checkpoint(const uint8_t* source, uint8_t* destination,
                                  uint32_t rows, uint32_t cols) {
    const uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const uint32_t groups=cols/16U;
    if(index>=uint64_t(rows)*groups) return;
    const uint32_t lane=index%32U,group=(index/32U)%groups,tile=index/(uint64_t(groups)*32U);
    const uint32_t row=tile*32U+((lane>>2U)&3U)*8U+(lane&3U)+((lane&16U)?4U:0U);
    const uint64_t logical=uint64_t(row)*groups+group;
    const uint8_t* codes=source+logical*8U;
    constexpr uint32_t order[16]={0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15};
    #pragma unroll
    for(uint32_t byte=0;byte<8;++byte) {
        const uint32_t a=order[byte*2U],b=order[byte*2U+1U];
        destination[index*8U+byte]=((codes[a/2U]>>((a&1U)*4U))&15U)
            | (((codes[b/2U]>>((b&1U)*4U))&15U)<<4U);
    }
    destination[uint64_t(rows)*cols/2U+index]=source[uint64_t(rows)*cols/2U+logical];
}

__device__ __forceinline__ uint8_t natural_code(const uint8_t* source, uint64_t group, uint32_t element) {
    const uint32_t j=element&15U;
    const uint32_t ordered=(j/8U)*8U+(j%8U)/2U+(j&1U)*4U;
    return (source[group*8U+ordered/2U]>>((ordered&1U)*4U))&15U;
}

__device__ __forceinline__ float e2m1(uint8_t code) {
    const uint32_t m=code&7U;
    const uint32_t twice=m<4U?m:((2U+(m&1U))<<((m>>1U)-1U));
    const float v=float(twice)*0.5f;
    return (code&8U)?-v:v;
}

__device__ __forceinline__ float e4m3(uint8_t scale) {
    const uint32_t exponent=scale>>3U, mantissa=scale&7U;
    if (exponent==0U) return float(mantissa)*0x1p-9f;
    return __uint_as_float(((exponent+120U)<<23U)|(mantissa<<20U));
}

// Unscaled reproduces the legacy GGML->FP16 prefill materialization exactly.
// Folded reproduces QPN's FP16 scale multiplication boundaries, so wide GEMM
// and B1/B8 consume the same weight values in the *new* numerical profile.
// Merely changing physical layout must not implicitly select Folded=true.
template<bool Folded>
__global__ void decode_f16(const uint8_t* source, half* destination,
                          uint32_t rows, uint32_t cols, float tensor_scale) {
    const uint64_t pair=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (pair>=uint64_t(rows)*cols/2U) return;
    const uint64_t element=pair*2U;
    const uint32_t row=element/cols, col=element%cols;
    const uint64_t group=group_index(row,col/16U,cols);
    const uint8_t scale=source[uint64_t(rows)*cols/2U+group];
    const float a=e2m1(natural_code(source,group,col));
    const float b=e2m1(natural_code(source,group,col+1U));
    half2 result;
    if constexpr (Folded) {
        const half s=__hmul(__ushort_as_half(uint16_t(scale)<<7U),__float2half_rn(tensor_scale*256.0f));
        result=__halves2half2(__hmul(__float2half_rn(a),s),__hmul(__float2half_rn(b),s));
    } else {
        const float s=e4m3(scale);
        result=__floats2half2_rn(a*s,b*s);
    }
    reinterpret_cast<half2*>(destination)[pair]=result;
}

// One thread expands a complete 16-value group. The physical addresses and
// FP16 rounding boundaries are identical to decode_f16; scale/address work
// is shared by eight output pairs. Keep the scalar-pair kernel as an oracle.
template<bool Folded>
__global__ void decode_f16_grouped(const uint8_t* source, half* destination,
                                  uint32_t rows, uint32_t cols, float tensor_scale) {
    // The checked ABI bounds rows*cols to UINT32_MAX, hence group addressing
    // also fits32. Avoid a 64-bit divide for every group on SM70.
    const uint32_t logical=blockIdx.x*blockDim.x+threadIdx.x;
    const uint32_t groups=cols/16U;
    if(logical>=rows*groups) return;
    const uint32_t row=logical/groups, column_group=logical%groups;
    const uint64_t physical=group_index(row,column_group,cols);
    const uint8_t scale=source[uint64_t(rows)*cols/2U+physical];
    const uint2 codes=reinterpret_cast<const uint2*>(source)[physical];
    const uint64_t packed=uint64_t(codes.x)|(uint64_t(codes.y)<<32U);
    [[maybe_unused]] half folded{};
    [[maybe_unused]] float unscaled=0.0f;
    if constexpr(Folded) folded=__hmul(__ushort_as_half(uint16_t(scale)<<7U),__float2half_rn(tensor_scale*256.0f));
    else unscaled=e4m3(scale);
    uint32_t words[8];
    #pragma unroll
    for(uint32_t pair=0;pair<8;++pair) {
        const uint32_t j=pair*2U;
        const uint32_t lo=(j/8U)*8U+(j%8U)/2U;
        const float a=e2m1(uint8_t((packed>>(lo*4U))&15U));
        const float b=e2m1(uint8_t((packed>>((lo+4U)*4U))&15U));
        half2 value;
        if constexpr(Folded) value=__halves2half2(__hmul(__float2half_rn(a),folded),__hmul(__float2half_rn(b),folded));
        else value=__floats2half2_rn(a*unscaled,b*unscaled);
        const __half2_raw bits=value;
        words[pair]=uint32_t(bits.x)|(uint32_t(bits.y)<<16U);
    }
    // The grouped entry point requires an aligned16 output; callers with only
    // the legacy aligned4 contract must use decode_f16 instead.
    reinterpret_cast<uint4*>(destination)[logical*2U]=make_uint4(words[0],words[1],words[2],words[3]);
    reinterpret_cast<uint4*>(destination)[logical*2U+1U]=make_uint4(words[4],words[5],words[6],words[7]);
}

} // namespace augur_qwen35_qpn
