// SPDX-License-Identifier: Apache-2.0
#pragma once
// One warp/head, multiple independent heads/CTA. The sum remains the exact
// sequential FP32 loop; this is scheduling/load reuse, not a tree reduction.
template<unsigned Heads,bool Stage> __global__ void augur_prefill_gated_norm_warps(
 const float* recurrent,const float* norm_weight,const float* z,float* output,
 uint32_t tokens,float epsilon) {
 const unsigned lane=threadIdx.x,warp=threadIdx.y;
 const uint64_t head=uint64_t(blockIdx.x)*Heads+warp;
 if(head>=uint64_t(tokens)*48)return;
 const uint64_t base=head*128;
 __shared__ float staged[Heads][128];
 float values[4];
 #pragma unroll
 for(unsigned j=0;j<4;++j){values[j]=recurrent[base+j*32+lane];if constexpr(Stage)staged[warp][j*32+lane]=values[j];}
 if constexpr(Stage)__syncwarp();
 float scale=0.0f;
 if(lane==0){float sum=0.0f;for(unsigned i=0;i<128;++i){const float v=Stage?staged[warp][i]:recurrent[base+i];sum+=v*v;}scale=rsqrtf(sum/128.0f+epsilon);}
 scale=__shfl_sync(0xffffffffU,scale,0);
 #pragma unroll
 for(unsigned j=0;j<4;++j){const unsigned c=j*32+lane;output[base+c]=values[j]*scale*norm_weight[c]*prefill_silu(z[base+c]);}
}
