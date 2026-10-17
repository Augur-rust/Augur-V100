// SPDX-License-Identifier: Apache-2.0
// Same per-row reduction tree, with several independent output rows per warp.
// Four rows share Q/K and decay/beta loads; no cross-row reductions or new
// workspace. Compiled with the parent TU's fmad=false and ftz=false contract.
// Launch is restricted to the authenticated CheckpointSource shape, B>=512.
template<unsigned Warps,unsigned Rows> __global__ __launch_bounds__(32*Warps,2)
void augur_prefill_delta_rows(const float* activated,const float* scalars,float* state,float* recurrent,unsigned tokens){
 constexpr unsigned N=4; const unsigned head=blockIdx.x,lane=threadIdx.x;
 const unsigned first=(blockIdx.y*Warps+threadIdx.y)*Rows;
 const unsigned qhead=head/3;
 float st[Rows][N];
 #pragma unroll
 for(unsigned r=0;r<Rows;++r){
  #pragma unroll
  for(unsigned j=0;j<N;++j)st[r][j]=state[(head*128+first+r)*128+j*32+lane];
 }
 for(unsigned t=0;t<tokens;++t){
  const float* a=activated+uint64_t(t)*10240;const float* q=a+qhead*128;const float* k=q+2048;const float* v=a+4096+head*128;
  const float* c=scalars+4*(uint64_t(t)*48+head);float qs=c[0],ks=c[1],decay=c[2],beta=c[3];
  float qv[N],kv[N],p[Rows]{};
  #pragma unroll
  for(unsigned j=0;j<N;++j){qv[j]=q[j*32+lane]*qs;kv[j]=k[j*32+lane]*ks;
   #pragma unroll
   for(unsigned r=0;r<Rows;++r){st[r][j]*=decay;p[r]+=st[r][j]*kv[j];}
  }
  float d[Rows];
  #pragma unroll
  for(unsigned r=0;r<Rows;++r){p[r]=__shfl_sync(0xffffffff,prefill_warp_sum(p[r]),0);d[r]=(v[first+r]-p[r])*beta;}
  float qp[Rows]{};
  #pragma unroll
  for(unsigned j=0;j<N;++j){
   #pragma unroll
   for(unsigned r=0;r<Rows;++r){st[r][j]+=kv[j]*d[r];qp[r]+=st[r][j]*qv[j];}
  }
  #pragma unroll
  for(unsigned r=0;r<Rows;++r){qp[r]=prefill_warp_sum(qp[r]);if(lane==0)recurrent[uint64_t(t)*6144+head*128+first+r]=qp[r]*(1.0f/sqrtf(128.0f));}
 }
 #pragma unroll
 for(unsigned r=0;r<Rows;++r){
  #pragma unroll
  for(unsigned j=0;j<N;++j)state[(head*128+first+r)*128+j*32+lane]=st[r][j];
 }
}
