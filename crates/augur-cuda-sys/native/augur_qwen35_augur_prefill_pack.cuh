// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
// Pure layout/value helpers mirror the legacy decoder; no duplicate global
// kernels from its translation unit are linked into the cuBLAS implementation.
__device__ __forceinline__ uint64_t augur_expand_group_index(uint32_t row,uint32_t group,uint32_t cols) {
    const uint32_t r=row&31U,lane=(r/8U)*4U+(r&3U)+(r&4U)*4U;
    return (uint64_t(row/32U)*(cols/16U)+group)*32U+lane;
}
__device__ __forceinline__ float augur_expand_e2m1(uint8_t code) {
    const uint32_t m=code&7U,twice=m<4U?m:((2U+(m&1U))<<((m>>1U)-1U));
    const float v=float(twice)*0.5f;return (code&8U)?-v:v;
}
__device__ __forceinline__ float augur_expand_e4m3(uint8_t scale) {
    const uint32_t exponent=scale>>3U,mantissa=scale&7U;
    if(exponent==0U)return float(mantissa)*0x1p-9f;
    return __uint_as_float(((exponent+120U)<<23U)|(mantissa<<20U));
}
// Pair decoding uses the same NInfer half2 bit layout as the retained QPN
// projection. The >=184 scale branch preserves the full legacy byte domain.
__device__ __forceinline__ void augur_unpack_four(uint32_t q,half2 (&out)[4]) {
    constexpr uint32_t sign=0x80008000U,em=0x0e000e00U;
    uint32_t bits[4]={((q<<12)&sign)|((q<<9)&em),((q<<8)&sign)|((q<<5)&em),
        ((q<<4)&sign)|((q<<1)&em),(q&sign)|((q>>3)&em)};
    const half2 rebias=__float2half2_rn(16384.0f);
    #pragma unroll
    for(int j=0;j<4;++j)out[j]=__hmul2(*reinterpret_cast<half2*>(&bits[j]),rebias);
}
__global__ void augur_prefill_expand_half2(const uint8_t* source,half* destination,uint32_t rows,uint32_t cols) {
    const uint32_t logical=blockIdx.x*blockDim.x+threadIdx.x,groups=cols/16;
    if(logical>=rows*groups)return;
    const uint64_t physical=augur_expand_group_index(logical/groups,logical%groups,cols);
    const uint8_t scale=source[uint64_t(rows)*cols/2+physical];
    const uint2 codes=reinterpret_cast<const uint2*>(source)[physical];
    half2 values[8];
    if(scale<184U) {
        augur_unpack_four(codes.x,*reinterpret_cast<half2(*)[4]>(values));
        augur_unpack_four(codes.y,*reinterpret_cast<half2(*)[4]>(values+4));
        const half2 s=__float2half2_rn(augur_expand_e4m3(scale));
        #pragma unroll
        for(int j=0;j<8;++j)values[j]=__hmul2(values[j],s);
    } else {
        const uint64_t packed=uint64_t(codes.x)|(uint64_t(codes.y)<<32);
        const float s=augur_expand_e4m3(scale);
        #pragma unroll
        for(unsigned pair=0;pair<8;++pair) {
            const unsigned j=pair*2,lo=(j/8)*8+(j%8)/2;
            values[pair]=__floats2half2_rn(augur_expand_e2m1((packed>>(lo*4))&15)*s,
                augur_expand_e2m1((packed>>((lo+4)*4))&15)*s);
        }
    }
    const uint4* words=reinterpret_cast<const uint4*>(values);
    reinterpret_cast<uint4*>(destination)[logical*2]=words[0];
    reinterpret_cast<uint4*>(destination)[logical*2+1]=words[1];
}
template<unsigned Width> __global__ void augur_prefill_pack_vector(const float* source,half* destination,uint64_t elements) {
    static_assert(Width==2||Width==4,"admitted vector widths");
    const uint64_t stride=uint64_t(blockDim.x)*gridDim.x*Width;
    for(uint64_t i=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)*Width;i<elements;i+=stride) {
        if(i+Width<=elements) {
            if constexpr(Width==4) {
                const float4 v=*reinterpret_cast<const float4*>(source+i);
                half2 a=__floats2half2_rn(v.x,v.y),b=__floats2half2_rn(v.z,v.w);
                *reinterpret_cast<uint2*>(destination+i)=make_uint2(*reinterpret_cast<unsigned*>(&a),*reinterpret_cast<unsigned*>(&b));
            } else {
                const float2 v=*reinterpret_cast<const float2*>(source+i);
                *reinterpret_cast<half2*>(destination+i)=__floats2half2_rn(v.x,v.y);
            }
        } else for(uint64_t j=i;j<elements;++j)destination[j]=__float2half_rn(source[j]);
    }
}
