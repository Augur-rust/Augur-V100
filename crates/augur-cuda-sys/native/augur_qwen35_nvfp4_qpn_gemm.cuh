// SPDX-License-Identifier: Apache-2.0
// Adapted from NInfer V100 Duo nvfp4_volta_qpn_gemm.cuh, Apache-2.0.
// See licenses/NINFER-NOTICE.txt and licenses/NINFER-APACHE-2.0.txt.
// Volta MMA primitive is llama/ggml-derived (MIT; root NOTICE).
// Changes: standalone namespace, one 8-row tile, contiguous F32 output;
// accept original F32 or FP16 activations with explicit FP16 RNE boundary.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include "augur_qwen35_nvfp4_qpn_layout.cuh"
#include "augur_qwen35_qpn_group.h"

namespace augur_qwen35_qpn {

__device__ __forceinline__ void mma(float (&d)[8],unsigned a0,unsigned a1,unsigned b0,unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7])
                 : "r"(a0),"r"(a1),"r"(b0),"r"(b1));
}

__device__ __forceinline__ void decode_quad(uint32_t q,half2 rebias,half2 (&out)[4]) {
    constexpr uint32_t sign=0x80008000U,em=0x0e000e00U;
    uint32_t bits[4]={((q<<12)&sign)|((q<<9)&em),((q<<8)&sign)|((q<<5)&em),
                      ((q<<4)&sign)|((q<<1)&em),(q&sign)|((q>>3)&em)};
    #pragma unroll
    for(int j=0;j<4;++j) out[j]=__hmul2(*reinterpret_cast<half2*>(&bits[j]),rebias);
}

// The same K partition/reduction is used for every admitted width 1..8.
// Batch-size changes must not change the represented per-token contraction.
// One 32-row tile. `tile` replaces blockIdx.x so a grouped launch can run several
// projections' tiles in one grid with the identical per-tile instruction stream.
// Output row r of the tile lands at output[token*out_stride+col] for col<out_cols.
template<int SplitK,class Activation,bool PostScale=false,bool Refine=false,bool Chunked=false,int ChunkGroups=4,bool ChunkLow=true>
__device__ __forceinline__
void project_tile(int tile,const uint8_t* __restrict__ physical,const Activation* __restrict__ x,
             float* __restrict__ output,int rows,int cols,int tokens,float tensor_scale,int out_stride,int out_cols,
             float* __restrict__ mirror=nullptr) {
    static_assert(SplitK==8||SplitK==16,"qualified QPN schedules");
    static_assert(ChunkGroups==4||ChunkGroups==16,"explicit accumulation schedules");
    static_assert(std::is_same_v<Activation,half>||std::is_same_v<Activation,float>,"explicit activation dtype");
    // Static shared arrays of a device function are per-CTA, exactly as before.
    __shared__ float partial[Chunked?1:SplitK][256];
    __shared__ float partial_low[Refine&&!Chunked?SplitK:1][256];
    __shared__ double partial_precise[Chunked?SplitK:1][256];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int qp=(lane>>2)&3,r=(lane&3)+((lane&16)?4:0);
    const int groups=cols/16,quotient=groups/SplitK;
    const int begin=warp*quotient,end=warp==SplitK-1?groups:begin+quotient;
    const uint64_t base=uint64_t(tile)*groups*32U+lane;
    const auto* scales=physical+uint64_t(rows)*cols/2U;
    const half2 rebias=__float2half2_rn(16384.0f);
    const half2 divisor=__float2half2_rn(PostScale?256.0f:tensor_scale*256.0f);
    float accum[2][8]={};
    float accum_low[2][8]={};
    double precise[8]={};
    for(int group=begin;group<end;++group) {
        const uint64_t index=base+uint64_t(group)*32U;
        const uint2 q=__ldg(reinterpret_cast<const uint2*>(physical+index*8U));
        const half2 s=__hmul2(__half2half2(__ushort_as_half(uint16_t(scales[index])<<7U)),divisor);
        half2 weight[8];
        decode_quad(q.x,rebias,*reinterpret_cast<half2(*)[4]>(&weight[0]));
        decode_quad(q.y,rebias,*reinterpret_cast<half2(*)[4]>(&weight[4]));
        #pragma unroll
        for(int j=0;j<8;++j) weight[j]=__hmul2(weight[j],s);
        half values[16];
        if(r<tokens) {
            const Activation* source=x+uint64_t(r)*cols+group*16;
            if constexpr(std::is_same_v<Activation,half>) {
                const uint4 lo=*reinterpret_cast<const uint4*>(source),hi=*reinterpret_cast<const uint4*>(source+8);
                const auto* a=reinterpret_cast<const half*>(&lo);
                const auto* b=reinterpret_cast<const half*>(&hi);
                #pragma unroll
                for(int j=0;j<8;++j) { values[j]=a[j];values[j+8]=b[j]; }
            } else {
                #pragma unroll
                for(int j=0;j<16;++j) values[j]=__float2half_rn(source[j]);
            }
        } else {
            #pragma unroll
            for(int j=0;j<16;++j) values[j]=__ushort_as_half(0);
        }
        const auto* a=reinterpret_cast<const unsigned*>(values);
        const auto* b=reinterpret_cast<const unsigned*>(weight);
        mma(accum[0],a[0],a[1],b[0],b[1]);
        mma(accum[1],a[2],a[3],b[2],b[3]);
        mma(accum[0],a[4],a[5],b[4],b[5]);
        mma(accum[1],a[6],a[7],b[6],b[7]);
        if constexpr(Refine) {
            // Same resident weights, second activation component; no second
            // decoded weight representation or a scalar full-matvec repair.
            if(r<tokens) {
                const half* low=x+uint64_t(tokens)*cols+uint64_t(r)*cols+group*16;
                // Both planes are aligned16 and K is a multiple of64. Keep
                // residual loads vectorized like hi: scalar half loads compile
                // to16 separate LDG.U16 plus repacking instructions on SM70.
                const uint4 lo=*reinterpret_cast<const uint4*>(low),hi=*reinterpret_cast<const uint4*>(low+8);
                const auto* l=reinterpret_cast<const half*>(&lo);
                const auto* h=reinterpret_cast<const half*>(&hi);
                #pragma unroll
                for(int j=0;j<8;++j) { values[j]=l[j]; values[j+8]=h[j]; }
            }
            mma(accum_low[0],a[0],a[1],b[0],b[1]);
            mma(accum_low[1],a[2],a[3],b[2],b[3]);
            mma(accum_low[0],a[4],a[5],b[4],b[5]);
            mma(accum_low[1],a[6],a[7],b[6],b[7]);
        }
        if constexpr(Chunked) {
            // Bound the dominant hi MMA chain. The fast profile keeps the
            // much smaller residual chain over the whole K slice; its error
            // is explicitly scaled by2^-12, not silently declared exact.
            if((group-begin+1)%ChunkGroups==0 || group+1==end) {
                #pragma unroll
                for(int i=0;i<8;++i) {
                    if constexpr(ChunkLow) {
                        precise[i]+=double(accum[0][i])+double(accum[1][i])
                            +(double(accum_low[0][i])+double(accum_low[1][i]))*0x1p-12;
                        accum_low[0][i]=accum_low[1][i]=0.0f;
                    } else precise[i]+=double(accum[0][i])+double(accum[1][i]);
                    accum[0][i]=accum[1][i]=0.0f;
                }
            }
        }
    }
    #pragma unroll
    for(int i=0;i<8;++i) {
        accum[0][i]+=accum[1][i];
        const int row=(i&2)|((lane&16)?4:0)|(lane&1);
        const int col=(i&1)|(((lane>>1)&1)<<1)|((i>>2)<<2);
        if constexpr(Chunked) {
            if constexpr(!ChunkLow) precise[i]+=(double(accum_low[0][i])+double(accum_low[1][i]))*0x1p-12;
            partial_precise[warp][row*32+qp*8+col]=precise[i];
        }
        else {
            partial[warp][row*32+qp*8+col]=accum[0][i];
            if constexpr(Refine) partial_low[warp][row*32+qp*8+col]=accum_low[0][i]+accum_low[1][i];
        }
    }
    __syncthreads();
    for(int e=threadIdx.x;e<256;e+=SplitK*32) {
        const int row=e/32,col=tile*32+e%32;
        if(row<tokens&&col<rows&&col<out_cols) {
            if constexpr(Chunked) {
                double value=0;
                #pragma unroll
                for(int w=0;w<SplitK;++w) value+=partial_precise[w][e];
                output[uint64_t(row)*out_stride+col]=float(value*double(tensor_scale));
                continue;
            } else if constexpr(Refine) {
                // Explicit new reduction contract: deterministic double merge
                // of F32 MMA partials. MMA accumulation is still F32, not F64.
                double value=0;
                #pragma unroll
                for(int w=0;w<SplitK;++w) value+=double(partial[w][e])+double(partial_low[w][e])*0x1p-12;
                output[uint64_t(row)*out_stride+col]=float(value*double(tensor_scale));
                continue;
            }
            float value=0;
            #pragma unroll
            for(int w=0;w<SplitK;++w) value+=partial[w][e];
            if constexpr(PostScale) value=__fmul_rn(value,tensor_scale);
            output[uint64_t(row)*out_stride+col]=value;
            if(mirror) mirror[uint64_t(row)*out_stride+col]=value;
        }
    }
}
template<int SplitK,class Activation,bool PostScale=false,bool Refine=false,bool Chunked=false,int ChunkGroups=4,bool ChunkLow=true>
__global__ __launch_bounds__(SplitK*32,Refine?1:32/SplitK)
void project(const uint8_t* __restrict__ physical,const Activation* __restrict__ x,
             float* __restrict__ output,int rows,int cols,int tokens,float tensor_scale) {
    project_tile<SplitK,Activation,PostScale,Refine,Chunked,ChunkGroups,ChunkLow>(
        blockIdx.x,physical,x,output,rows,cols,tokens,tensor_scale,rows,rows);
}

// GDN input group (profile2, SplitK8, same staged FP16 input): one grid walks the
// tiles of up to four projections in order. Each CTA runs exactly project<8,half,true>'s
// tile code for its own projection; only CTA scheduling, the output row stride and an
// optional second store differ, so every output word is bit-identical to separate launches.
__global__ __launch_bounds__(256,4)
void project_group(augur_qpn_group g,const half* __restrict__ x) {
    int i=0;
    #pragma unroll
    for(int j=1;j<4;++j) if(j<g.count&&int(blockIdx.x)>=g.m[j].first_tile) i=j;
    const augur_qpn_group_member& m=g.m[i];
    const auto* w=reinterpret_cast<const uint8_t*>(m.weight);
    auto* y=reinterpret_cast<float*>(m.output);
    const int tile=int(blockIdx.x)-m.first_tile;
    // One instantiation (one set of shared arrays); the mirror store is uniform per CTA.
    project_tile<8,half,true>(tile,w,x,y,m.rows,g.cols,g.tokens,m.scale,m.out_cols,m.out_cols,
        reinterpret_cast<float*>(m.mirror));
}

// Workspace profile6: independent hi/lo CTAs with four K partitions per CTA.
// Adjacent x blocks consume the same packed tile from different input planes.
// All outputs are deterministic disjoint F64 partials; no atomics/host waits.
template<int SplitK,bool Packed>
__global__ __launch_bounds__(128,4)
void project_workspace_parts(const uint8_t* __restrict__ physical,
                             const half* __restrict__ x,double* __restrict__ partials,
                             int rows,int cols,int tokens) {
    const int lane=threadIdx.x&31,part=blockIdx.y*4+(threadIdx.x>>5);
    const int tile=Packed?blockIdx.x:blockIdx.x>>1,plane=Packed?0:blockIdx.x&1;
    // Small widths use otherwise idle MMA rows for the second component.
    // No padded weights or tokens: exactly2*tokens components are staged.
    const int active_rows=Packed?tokens*2:tokens;
    const int qp=(lane>>2)&3,r=(lane&3)+((lane&16)?4:0);
    const int groups=cols/16,quotient=groups/SplitK;
    const int begin=part*quotient,end=part==SplitK-1?groups:begin+quotient;
    const uint64_t base=uint64_t(tile)*groups*32U+lane;
    const auto* scales=physical+uint64_t(rows)*cols/2U;
    const half2 rebias=__float2half2_rn(16384.0f),divisor=__float2half2_rn(256.0f);
    float accum[2][8]={};
    double precise[8]={};
    for(int group=begin;group<end;++group) {
        const uint64_t index=base+uint64_t(group)*32U;
        const uint2 q=__ldg(reinterpret_cast<const uint2*>(physical+index*8U));
        const half2 scale=__hmul2(__half2half2(__ushort_as_half(uint16_t(scales[index])<<7U)),divisor);
        half2 weight[8];
        decode_quad(q.x,rebias,*reinterpret_cast<half2(*)[4]>(&weight[0]));
        decode_quad(q.y,rebias,*reinterpret_cast<half2(*)[4]>(&weight[4]));
        #pragma unroll
        for(int j=0;j<8;++j) weight[j]=__hmul2(weight[j],scale);
        half values[16];
        if(r<active_rows) {
            const half* source=x+(uint64_t(plane)*tokens+r)*cols+group*16;
            const uint4 lo=*reinterpret_cast<const uint4*>(source),hi=*reinterpret_cast<const uint4*>(source+8);
            const auto* l=reinterpret_cast<const half*>(&lo);
            const auto* h=reinterpret_cast<const half*>(&hi);
            #pragma unroll
            for(int j=0;j<8;++j) { values[j]=l[j]; values[j+8]=h[j]; }
        } else {
            #pragma unroll
            for(int j=0;j<16;++j) values[j]=__ushort_as_half(0);
        }
        const auto* a=reinterpret_cast<const unsigned*>(values);
        const auto* b=reinterpret_cast<const unsigned*>(weight);
        mma(accum[0],a[0],a[1],b[0],b[1]);
        mma(accum[1],a[2],a[3],b[2],b[3]);
        mma(accum[0],a[4],a[5],b[4],b[5]);
        mma(accum[1],a[6],a[7],b[6],b[7]);
        if((group-begin+1)%4==0 || group+1==end) {
            #pragma unroll
            for(int i=0;i<8;++i) {
                precise[i]+=double(accum[0][i])+double(accum[1][i]);
                accum[0][i]=accum[1][i]=0.0f;
            }
        }
    }
    #pragma unroll
    for(int i=0;i<8;++i) {
        const int row=(i&2)|((lane&16)?4:0)|(lane&1);
        const int column=tile*32+qp*8+(i&1)+(((lane>>1)&1)<<1)+((i>>2)<<2);
        if(row<active_rows) {
            const int component=Packed?int(row>=tokens):plane;
            const int logical_row=Packed?row-component*tokens:row;
            partials[((uint64_t(component)*SplitK+part)*tokens+logical_row)*rows+column]=precise[i];
        }
    }
}
// Profile7: statically unrolled four-group MMA chunks, then an ordered
// four-warp F64 sum per component in8KiB shared memory. Writes one quarter
// of profile6 partials. This is a new reduction tree, not a bit-identity
// promise against profile6. Every admitted batch width uses the same tree.
template<int SplitK,bool Packed,bool ScalarLiveRows=false>
__global__ __launch_bounds__(128,2)
void project_workspace_cta_fold_parts(const uint8_t* __restrict__ physical,const half* __restrict__ x,
           double* __restrict__ partials,int rows,int cols,int tokens) {
    constexpr bool Fold=true,Unroll=true;
    constexpr int ChunkGroups=4;
    static_assert(!ScalarLiveRows || Packed,"B1 hi/lo rows share one MMA tile");
    // B1 has only two live MMA rows (hi and residual). Keep all warp MMA
    // participants, but do not accumulate/store six discarded output rows.
    // The active-row F32 chunk and ordered four-warp F64 sum are unchanged.
    constexpr int SharedElements=ScalarLiveRows?64:256;
    __shared__ double local[4][SharedElements];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,part=blockIdx.y*4+warp;
    const int tile=Packed?blockIdx.x:blockIdx.x>>1,plane=Packed?0:blockIdx.x&1;
    const int active_rows=ScalarLiveRows?2:(Packed?tokens*2:tokens);
    const int qp=(lane>>2)&3,r=(lane&3)+((lane&16)?4:0);
    const int groups=cols/16,quotient=groups/SplitK;
    const int begin=part*quotient,end=part==SplitK-1?groups:begin+quotient;
    const uint64_t base=uint64_t(tile)*groups*32U+lane;
    const auto* scales=physical+uint64_t(rows)*cols/2U;
    const half2 rebias=__float2half2_rn(16384.0f),divisor=__float2half2_rn(256.0f);
    float accum[2][8]={};
    double precise[8]={};
    for(int chunk=begin;chunk<end;chunk+=ChunkGroups) {
        #pragma unroll (Unroll?ChunkGroups:1)
        for(int g=0;g<ChunkGroups;++g) {
            const int group=chunk+g;
            if(group<end) {
                const uint64_t index=base+uint64_t(group)*32U;
                const uint2 q=__ldg(reinterpret_cast<const uint2*>(physical+index*8U));
                const half2 scale=__hmul2(__half2half2(__ushort_as_half(uint16_t(scales[index])<<7U)),divisor);
                half2 weight[8];
                decode_quad(q.x,rebias,*reinterpret_cast<half2(*)[4]>(&weight[0]));
                decode_quad(q.y,rebias,*reinterpret_cast<half2(*)[4]>(&weight[4]));
                #pragma unroll
                for(int j=0;j<8;++j) weight[j]=__hmul2(weight[j],scale);
                half values[16];
                if(r<active_rows) {
                    const half* source=x+(uint64_t(plane)*tokens+r)*cols+group*16;
                    const uint4 lo=*reinterpret_cast<const uint4*>(source),hi=*reinterpret_cast<const uint4*>(source+8);
                    const auto* l=reinterpret_cast<const half*>(&lo);
                    const auto* h=reinterpret_cast<const half*>(&hi);
                    #pragma unroll
                    for(int j=0;j<8;++j) { values[j]=l[j];values[j+8]=h[j]; }
                } else {
                    #pragma unroll
                    for(int j=0;j<16;++j) values[j]=__ushort_as_half(0);
                }
                const auto* a=reinterpret_cast<const unsigned*>(values);
                const auto* b=reinterpret_cast<const unsigned*>(weight);
                mma(accum[0],a[0],a[1],b[0],b[1]);
                mma(accum[1],a[2],a[3],b[2],b[3]);
                mma(accum[0],a[4],a[5],b[4],b[5]);
                mma(accum[1],a[6],a[7],b[6],b[7]);
            }
        }
        #pragma unroll
        for(int i=0;i<8;++i) {
            if constexpr(ScalarLiveRows) {
                // i bit1 addresses rows2/3; lane bit4 addresses rows4..7.
                if((i&2)==0 && (lane&16)==0)
                    precise[i]+=double(accum[0][i])+double(accum[1][i]);
            } else precise[i]+=double(accum[0][i])+double(accum[1][i]);
            accum[0][i]=accum[1][i]=0.0f;
        }
    }
    #pragma unroll
    for(int i=0;i<8;++i) {
        const int row=(i&2)|((lane&16)?4:0)|(lane&1);
        const int column=qp*8+(i&1)+(((lane>>1)&1)<<1)+((i>>2)<<2);
        if constexpr(Fold) {
            if(!ScalarLiveRows || row<2) local[warp][row*32+column]=precise[i];
        }
        else if(row<active_rows) {
            const int component=Packed?int(row>=tokens):plane;
            const int logical_row=Packed?row-component*tokens:row;
            partials[((uint64_t(component)*SplitK+part)*tokens+logical_row)*rows+tile*32+column]=precise[i];
        }
    }
    if constexpr(Fold) {
        __syncthreads();
        for(int e=threadIdx.x;e<SharedElements;e+=128) {
            const int row=e/32,column=e%32;
            if(row<active_rows) {
                double value=0;
                #pragma unroll
                for(int w=0;w<4;++w) value+=local[w][e];
                const int component=Packed?int(row>=tokens):plane;
                const int logical_row=Packed?row-component*tokens:row;
                partials[((uint64_t(component)*(SplitK/4)+blockIdx.y)*tokens+logical_row)*rows+tile*32+column]=value;
            }
        }
    }
}

// Profile8 B1 only, shape-whitelisted by the host dispatch. Fuse parts+merge
// inside one CTA without changing K slices, four-group F32 MMA chunks,
// ordered four-warp F64 folds, hi/lo merge or final scale/cast. The shared
// array replaces the old intermediate global partials; no persistent storage.
// Gate/up and Q use16 warps; Z uses8 warps with2 original slices per warp.
// Do not generalize this schedule: other measured shapes were slower.
template<int PartsPerWarp>
__global__ __launch_bounds__((16/PartsPerWarp)*32,1)
void project_b1_single_cta(const uint8_t* __restrict__ physical,const half* __restrict__ x,
                         float* __restrict__ output,int rows,int cols,float tensor_scale) {
    static_assert(PartsPerWarp==1 || PartsPerWarp==2,"screened B1 schedules only");
    constexpr int SplitK=16,Warps=SplitK/PartsPerWarp,ChunkGroups=4;
    constexpr bool ScalarLiveRows=true,Unroll=true;
    constexpr int active_rows=2,plane=0,tokens=1;
    __shared__ double local[SplitK][64];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,tile=blockIdx.x;
    const int qp=(lane>>2)&3,r=(lane&3)+((lane&16)?4:0);
    const int groups=cols/16,quotient=groups/SplitK;
    const uint64_t base=uint64_t(tile)*groups*32U+lane;
    const auto* scales=physical+uint64_t(rows)*cols/2U;
    const half2 rebias=__float2half2_rn(16384.0f),divisor=__float2half2_rn(256.0f);
    #pragma unroll 1
    for(int iteration=0;iteration<PartsPerWarp;++iteration) {
        const int part=warp+iteration*Warps;
        const int begin=part*quotient,end=part==SplitK-1?groups:begin+quotient;
        float accum[2][8]={};
        double precise[8]={};
        for(int chunk=begin;chunk<end;chunk+=ChunkGroups) {
            #pragma unroll (Unroll?ChunkGroups:1)
            for(int g=0;g<ChunkGroups;++g) {
                const int group=chunk+g;
                if(group<end) {
                    const uint64_t index=base+uint64_t(group)*32U;
                    const uint2 q=__ldg(reinterpret_cast<const uint2*>(physical+index*8U));
                    const half2 scale=__hmul2(__half2half2(__ushort_as_half(uint16_t(scales[index])<<7U)),divisor);
                    half2 weight[8];
                    decode_quad(q.x,rebias,*reinterpret_cast<half2(*)[4]>(&weight[0]));
                    decode_quad(q.y,rebias,*reinterpret_cast<half2(*)[4]>(&weight[4]));
                    #pragma unroll
                    for(int j=0;j<8;++j) weight[j]=__hmul2(weight[j],scale);
                    half values[16];
                    // Gate/up and Q: unused MMA rows may duplicate the two
                    // live rows. Avoid a divergent input load; only the original
                    // live rows enter the unchanged precise fold/output.
                    if(PartsPerWarp==1 || r<active_rows) {
                        const half* source=x+(uint64_t(plane)*tokens+(PartsPerWarp==1?(r&1):r))*cols+group*16;
                        const uint4 lo=*reinterpret_cast<const uint4*>(source),hi=*reinterpret_cast<const uint4*>(source+8);
                        const auto* l=reinterpret_cast<const half*>(&lo);
                        const auto* h=reinterpret_cast<const half*>(&hi);
                        #pragma unroll
                        for(int j=0;j<8;++j) { values[j]=l[j];values[j+8]=h[j]; }
                    } else {
                        #pragma unroll
                        for(int j=0;j<16;++j) values[j]=__ushort_as_half(0);
                    }
                    const auto* a=reinterpret_cast<const unsigned*>(values);
                    const auto* b=reinterpret_cast<const unsigned*>(weight);
                    mma(accum[0],a[0],a[1],b[0],b[1]);
                    mma(accum[1],a[2],a[3],b[2],b[3]);
                    mma(accum[0],a[4],a[5],b[4],b[5]);
                    mma(accum[1],a[6],a[7],b[6],b[7]);
                }
            }
        #pragma unroll
        for(int i=0;i<8;++i) {
            if constexpr(ScalarLiveRows) {
                // i bit1 addresses rows2/3; lane bit4 addresses rows4..7.
                if((i&2)==0 && (lane&16)==0)
                    precise[i]+=double(accum[0][i])+double(accum[1][i]);
            } else precise[i]+=double(accum[0][i])+double(accum[1][i]);
            accum[0][i]=accum[1][i]=0.0f;
        }
    }

        #pragma unroll
        for(int i=0;i<8;++i) {
            const int row=(i&2)|((lane&16)?4:0)|(lane&1);
            const int column=qp*8+(i&1)+(((lane>>1)&1)<<1)+((i>>2)<<2);
            if(row<2)local[part][row*32+column]=precise[i];
        }
    }
    __syncthreads();
    if(threadIdx.x<32) {
        const int e=threadIdx.x;
        double value=0;
        #pragma unroll
        for(int part=0;part<SplitK/4;++part) {
            double high=0,low=0;
            #pragma unroll
            for(int w=0;w<4;++w){high+=local[part*4+w][e];low+=local[part*4+w][32+e];}
            // Same ordered four-warp folds, then same hi+lo*2^-12 merge.
            value+=high+low*0x1p-12;
        }
        output[tile*32+e]=float(value*double(tensor_scale));
    }
}

// Packed hi/lo rows with a depth-D code/scale prefetch, one CTA per 32-row
// tile. Same K partition, same four-group F32 MMA chunks, same ordered
// four-warp F64 fold and same hi+lo*2^-12 merge as project_b1_single_cta; only
// the issue order of the weight loads and the cache policy change. The codes
// are read once per step and never reused, so __ldcs keeps them from evicting
// the activation planes that ARE reused.
//
// Residency is the point: 256 threads at Blk=4 is the largest block count that
// still compiles at 64 registers per thread. Every higher value spills and
// collapses the kernel (measured: Blk=6 47.8 ms, Blk=8 223 ms), so Blk is not
// a tuning knob past 4.
template<int SplitK,int Blk,int Depth,bool ChunkFold=true,bool StreamLoad=true>
__global__ __launch_bounds__(SplitK*32,Blk)
void project_packed_prefetch(const uint8_t* __restrict__ physical,const half* __restrict__ x,
                             float* __restrict__ output,int rows,int cols,float tensor_scale) {
    static_assert(SplitK==8||SplitK==16,"qualified QPN schedules");
    static_assert(Depth>=1&&Depth<=8,"measured prefetch depths");
    constexpr int ChunkGroups=4;
    __shared__ float partial[SplitK][256];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int qp=(lane>>2)&3,r=(lane&3)+((lane&16)?4:0);
    const int groups=cols/16,quotient=groups/SplitK;
    const int begin=warp*quotient,end=warp==SplitK-1?groups:begin+quotient;
    const uint64_t base=uint64_t(blockIdx.x)*groups*32U+lane;
    const auto* scales=physical+uint64_t(rows)*cols/2U;
    const half2 rebias=__float2half2_rn(16384.0f),divisor=__float2half2_rn(256.0f);
    float accum[2][8]={};
    double precise[8]={};
    for(int chunk=begin;chunk<end;chunk+=Depth) {
        const int span=min(Depth,end-chunk);
        uint2 qq[Depth]; uint8_t ss[Depth];
        #pragma unroll
        for(int k=0;k<Depth;++k) if(k<span) {
            const uint64_t index=base+uint64_t(chunk+k)*32U;
            const uint2* codes=reinterpret_cast<const uint2*>(physical+index*8U);
            if constexpr(StreamLoad) { qq[k]=__ldcs(codes); ss[k]=__ldcs(scales+index); }
            else { qq[k]=__ldg(codes); ss[k]=__ldg(scales+index); }
        }
        #pragma unroll
        for(int k=0;k<Depth;++k) {
            if(k>=span) break;
            const int group=chunk+k;
            const half2 scale=__hmul2(__half2half2(__ushort_as_half(uint16_t(ss[k])<<7U)),divisor);
            half2 weight[8];
            decode_quad(qq[k].x,rebias,*reinterpret_cast<half2(*)[4]>(&weight[0]));
            decode_quad(qq[k].y,rebias,*reinterpret_cast<half2(*)[4]>(&weight[4]));
            #pragma unroll
            for(int j=0;j<8;++j) weight[j]=__hmul2(weight[j],scale);
            half values[16];
            // Row0 carries the FP16 activation, row1 its residual plane. Rows
            // 2..7 are idle at B1 and stay zero so they fold to nothing.
            if(r<2) {
                const half* source=x+uint64_t(r)*cols+group*16;
                const uint4 lo=*reinterpret_cast<const uint4*>(source),hi=*reinterpret_cast<const uint4*>(source+8);
                const auto* l=reinterpret_cast<const half*>(&lo);
                const auto* h=reinterpret_cast<const half*>(&hi);
                #pragma unroll
                for(int j=0;j<8;++j) { values[j]=l[j];values[j+8]=h[j]; }
            } else {
                #pragma unroll
                for(int j=0;j<16;++j) values[j]=__ushort_as_half(0);
            }
            const auto* a=reinterpret_cast<const unsigned*>(values);
            const auto* b=reinterpret_cast<const unsigned*>(weight);
            mma(accum[0],a[0],a[1],b[0],b[1]);
            mma(accum[1],a[2],a[3],b[2],b[3]);
            mma(accum[0],a[4],a[5],b[4],b[5]);
            mma(accum[1],a[6],a[7],b[6],b[7]);
            if constexpr(ChunkFold) {
                if((group-begin+1)%ChunkGroups==0 || group+1==end) {
                    #pragma unroll
                    for(int i=0;i<8;++i) {
                        // i bit1 addresses rows2/3; lane bit4 addresses rows4..7.
                        if((i&2)==0 && (lane&16)==0)
                            precise[i]+=double(accum[0][i])+double(accum[1][i]);
                        accum[0][i]=accum[1][i]=0.0f;
                    }
                }
            }
        }
    }
    #pragma unroll
    for(int i=0;i<8;++i) {
        if constexpr(!ChunkFold) {
            // Whole-slice F32 accumulation, widened once here. The F64 chunk
            // fold above bounds the F32 chain instead; this variant keeps the
            // MMA chain intact and pays for it with one conversion per element.
            if((i&2)==0 && (lane&16)==0) precise[i]=double(accum[0][i])+double(accum[1][i]);
        }
        const int row=(i&2)|((lane&16)?4:0)|(lane&1);
        const int column=qp*8+(i&1)+(((lane>>1)&1)<<1)+((i>>2)<<2);
        if(row<2) partial[warp][row*32+column]=precise[i];
    }
    __syncthreads();
    for(int e=threadIdx.x;e<256;e+=SplitK*32) {
        const int row=e/32,column=blockIdx.x*32+e%32;
        if(row<1 && column<rows) {
            double value=0;
            #pragma unroll
            for(int w=0;w<SplitK;++w)
                value+=double(partial[w][e])+double(partial[w][32+e])*0x1p-12;
            output[column]=float(value*double(tensor_scale));
        }
    }
}

template<int SplitK>
__global__ void project_workspace_merge(const double* __restrict__ partials,
                                        float* __restrict__ output,
                                        uint64_t elements,float tensor_scale) {
    const uint64_t e=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(e>=elements) return;
    double value=0;
    #pragma unroll
    for(int part=0;part<SplitK;++part)
        value+=partials[uint64_t(part)*elements+e]
              +partials[uint64_t(SplitK+part)*elements+e]*0x1p-12;
    output[e]=float(value*double(tensor_scale));
}

} // namespace augur_qwen35_qpn
