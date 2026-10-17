// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
// Compile in the canonical runtime TU: preserve its FMA/FTZ contract.
namespace augur_rmsnorm_warp_tree5120 {
constexpr int K=5120,T=256,PER=K/T;

// Same 256-lane tree: reduce across8 warps FIRST, then down the32 lanes.
// Do not replace this by independently summing each warp before an8-way fold.
template<bool CacheInput,bool CacheWeight,bool Fused>
__global__ void warp_tree_norm(const float* input,const float* weight,float* output,
                              half* staged,float epsilon,int batch) {
    const int lane=threadIdx.x;
    const int offset=blockIdx.x*K;
    __shared__ float partial[T];
    __shared__ float scale;
    float values[CacheInput?PER:1],weights[CacheWeight?PER:1];
    float sum=0.0f;
    if constexpr(CacheInput) {
        #pragma unroll
        for(int j=0;j<PER;++j) {
            values[j]=input[offset+lane+j*T];
            if constexpr(CacheWeight) weights[j]=weight[lane+j*T];
            sum+=values[j]*values[j];
        }
    } else {
        for(int index=lane;index<K;index+=T) sum+=input[offset+index]*input[offset+index];
    }
    partial[lane]=sum;
    __syncthreads();
    if(lane<32) {
        const float a=__fadd_rn(partial[lane],partial[lane+128]);
        const float b=__fadd_rn(partial[lane+64],partial[lane+192]);
        const float c=__fadd_rn(partial[lane+32],partial[lane+160]);
        const float d=__fadd_rn(partial[lane+96],partial[lane+224]);
        float root=__fadd_rn(__fadd_rn(a,b),__fadd_rn(c,d));
        #pragma unroll
        for(int stride=16;stride;stride>>=1) root=__fadd_rn(root,__shfl_down_sync(0xffffffffu,root,stride));
        if(lane==0) scale=rsqrtf(root/float(K)+epsilon);
    }
    __syncthreads();
    #pragma unroll
    for(int j=0;j<PER;++j) {
        const int i=lane+j*T;
        const float x=CacheInput?values[j]:input[offset+i];
        const float w=CacheWeight?weights[j]:weight[i];
        const float y=x*scale*w;
        output[offset+i]=y; // preserve canonical F32 producer for all consumers
        if constexpr(Fused) {
            const half hi=__float2half_rn(y);
            float low=__fmul_rn(__fsub_rn(y,__half2float(hi)),4096.0f);
            if(isfinite(low)) low=fminf(65504.0f,fmaxf(-65504.0f,low));
            staged[offset+i]=hi;
            staged[batch*K+offset+i]=__float2half_rn(low);
        }
    }
}

}
