// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
__device__ __forceinline__ float augur_silu_pack_value(float g,float u) {
    float sigmoid;
    if(g>=0.0f) sigmoid=1.0f/(1.0f+expf(-g));
    else {const float exponent=expf(g);sigmoid=exponent/(1.0f+exponent);}
    return (g*sigmoid)*u;
}
// Preserve the observable F32 plane, including exact gate/output aliasing.
// Do not mark gate/output restrict: the production arena aliases them.
template<int V> __global__ void augur_silu_pack_f32_f16(
 const float* gate,const float* up,float* output,half* packed,uint32_t elements) {
 const uint64_t stride=uint64_t(blockDim.x)*gridDim.x*V;
 for(uint64_t i=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)*V;i<elements;i+=stride) {
  if constexpr(V==4) {
   if(i+3<elements) {
    const float4 g=reinterpret_cast<const float4*>(gate)[i/4],u=reinterpret_cast<const float4*>(up)[i/4];
    const float4 y=make_float4(augur_silu_pack_value(g.x,u.x),augur_silu_pack_value(g.y,u.y),augur_silu_pack_value(g.z,u.z),augur_silu_pack_value(g.w,u.w));
    reinterpret_cast<float4*>(output)[i/4]=y;
    const half2 a=__floats2half2_rn(y.x,y.y),b=__floats2half2_rn(y.z,y.w);
    reinterpret_cast<uint2*>(packed)[i/4]=make_uint2(*reinterpret_cast<const uint32_t*>(&a),*reinterpret_cast<const uint32_t*>(&b));continue;
   }
  } else if constexpr(V==2) {
   if(i+1<elements) {
    const float2 g=reinterpret_cast<const float2*>(gate)[i/2],u=reinterpret_cast<const float2*>(up)[i/2];
    const float2 y=make_float2(augur_silu_pack_value(g.x,u.x),augur_silu_pack_value(g.y,u.y));
    reinterpret_cast<float2*>(output)[i/2]=y;reinterpret_cast<half2*>(packed)[i/2]=__floats2half2_rn(y.x,y.y);continue;
   }
  }
  for(int j=0;j<V&&i+j<elements;++j) {const float y=augur_silu_pack_value(gate[i+j],up[i+j]);output[i+j]=y;packed[i+j]=__float2half_rn(y);}
 }
}
