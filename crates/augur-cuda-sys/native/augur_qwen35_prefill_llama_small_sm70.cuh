// Narrow SM70/no-ID/no-bias adaptation of ggml CUDA MMVQ/MMVF/quantize order.
// Derived algorithm: llama.cpp local b10759 checkout, actual DLL revision unknown.
// MIT, Copyright (c) 2023-2026 The ggml authors; see llama_attention_vendor/LICENSE.
#include <cuda_bf16.h>
namespace llama_small {
struct Q8 { __half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q8)==36,"q8_1 layout");
__device__ __forceinline__ float reduce_sum(float v){
    for(int d=16;d;d>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,d));return v;}
__device__ __forceinline__ float reduce_max(float v){
    for(int d=16;d;d>>=1)v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,d));return v;}
__global__ void quantize(const float* x,Q8* y,unsigned k){
    const unsigned col=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y;
    if(col>=k)return;const float v=x[size_t(row)*k+col];
    const float a=reduce_max(fabsf(v)),s=reduce_sum(v),d=a*(1.0f/127.0f);
    const int q=a==0?0:int(roundf(__fdividef(v,d)));
    Q8& b=y[size_t(row)*(k/32)+col/32];b.qs[col%32]=int8_t(q);
    if(col%32==0)b.ds=__floats2half2_rn(d,s);
}
__device__ __forceinline__ float half_scale(const uint8_t* p){return __half2float(__ushort_as_half(uint16_t(p[0])|uint16_t(p[1])<<8));}
__device__ __forceinline__ float subscale(uint8_t code){
    if(code==0x7f||code==0xff)return 0;
    const unsigned e=(code>>3)&15,m=code&7;
    const float raw=e?ldexpf(1.0f+float(m)*0.125f,int(e)-7):ldexpf(float(m),-9);
    return (code&128?-raw:raw)*0.5f;
}
__device__ __forceinline__ int value(unsigned c){
    const int tab[8]={0,1,2,3,4,6,8,12};return c&8?-tab[c&7]:tab[c&7];}
template<unsigned M,unsigned F,bool RowPlanes = false,bool DraftDp4a = false>
__global__ void mmvq(const uint8_t* w,const Q8* x,float* y,unsigned n,unsigned k,float tensor_scale){
    constexpr unsigned W=M<=4?4:2,R=M==1?1:2,QK=F==1?32:64,VDR=F==1?2:4;
    constexpr unsigned TPB=8/VDR,STEP=VDR*W*32/8;
    const unsigned lane=threadIdx.x,warp=threadIdx.y,tid=warp*32+lane,row0=blockIdx.x*R,blocks=k/QK;
    float sums[M][R]={};
    for(unsigned b=tid/TPB;b<blocks;b+=STEP){const unsigned part=VDR*(tid%TPB);
        for(unsigned j=0;j<M;++j)for(unsigned i=0;i<R;++i){if(row0+i>=n)continue;
            const uint8_t* wb = RowPlanes ? w+size_t(row0+i)*blocks*34+b*32
                : w+(size_t(row0+i)*blocks+b)*(F==1?34:36);
            const Q8* xb=x+size_t(j)*(k/32)+b*(QK/32);
            if constexpr(F==1){int dot=0;
                if constexpr(DraftDp4a) {
                    // llama.cpp b10759 vecdotq.cuh: get_int_b2 + Q8 dot DP4A.
                    // Q8_0 payload has only2-byte alignment, Q8_1 has4.
                    const uint16_t* qw=reinterpret_cast<const uint16_t*>(wb+(RowPlanes?0:2)+part*4);
                    const int* qx=reinterpret_cast<const int*>(xb->qs+part*4);
                    #pragma unroll
                    for(unsigned t=0;t<2;++t) {
                        const int packed=int(uint32_t(qw[t*2]) | (uint32_t(qw[t*2+1])<<16));
                        dot=__dp4a(packed,qx[t],dot);
                    }
                } else {for(unsigned t=0;t<8;++t)dot+=int(int8_t(wb[(RowPlanes?0:2)+part*4+t]))*int(xb->qs[part*4+t]);}
                const float scale=__fmul_rn(half_scale(RowPlanes ? w+size_t(row0+i)*blocks*34+k+b*2 : wb),__low2float(xb->ds));
                sums[j][i]=__fmaf_rn(scale,float(dot),sums[j][i]);
            }else{float part_sum=0;
                for(unsigned sub=part/2;sub<part/2+2;++sub){int dot=0;const Q8& a=xb[sub/2];
                    for(unsigned t=0;t<8;++t){unsigned code=wb[4+sub*8+t];dot+=value(code&15)*int(a.qs[(sub%2)*16+t]);dot+=value(code>>4)*int(a.qs[(sub%2)*16+t+8]);}
                    const float scale=__fmul_rn(subscale(wb[sub]),__low2float(a.ds));
                    part_sum=__fmaf_rn(scale,float(dot),part_sum);
                }
                sums[j][i]=__fadd_rn(sums[j][i],part_sum);
            }
        }
    }
    __shared__ float partial[W-1][M][R][32];
    if(warp)for(unsigned j=0;j<M;++j)for(unsigned i=0;i<R;++i)partial[warp-1][j][i][lane]=sums[j][i];
    __syncthreads();if(warp)return;
    for(unsigned j=0;j<M;++j)for(unsigned i=0;i<R;++i){for(unsigned z=0;z<W-1;++z)sums[j][i]=__fadd_rn(sums[j][i],partial[z][j][i][lane]);
        float v=reduce_sum(sums[j][i]);if(lane==i&&row0+i<n)y[size_t(j)*n+row0+i]=F==2?__fmul_rn(v,tensor_scale):v;}
}
template<unsigned M>
__global__ void mmvf(const __nv_bfloat16* w,const float* x,float* y,unsigned n,unsigned k){
    const unsigned row=blockIdx.x,tid=threadIdx.x;float sums[M]={};
    // The production shared BF16 head is MMVF<8>. Reduce all eight output
    // columns from the eight warp partials behind one shared-memory barrier,
    // while retaining the original XOR reduction order (the remaining lanes
    // are zero just as in the scalar loop). Other M values keep the reference
    // schedule unchanged.
    if constexpr (M==8) {
        __shared__ float scratch8[64];
        if(tid<64)scratch8[tid]=0;__syncthreads();
        for(unsigned pair=tid;pair<k/2;pair+=256){
            // k is fixed at 5120 for the shared BF16 head, so every pair is
            // naturally four-byte aligned. Load the two BF16 weights as one
            // vector while preserving the original scalar conversion and FMA
            // order below. This is a load-only candidate: no reduction or
            // rounding order changes are allowed here.
            const __nv_bfloat162 packed =
                *reinterpret_cast<const __nv_bfloat162*>(w + size_t(row)*k + pair*2);
            const float2 weights = __bfloat1622float2(packed);
            const float w0=weights.x;
            const float w1=weights.y;
            #pragma unroll
            for(unsigned j=0;j<8;++j){
                sums[j]=__fmaf_rn(w0,x[size_t(j)*k+pair*2],sums[j]);
                sums[j]=__fmaf_rn(w1,x[size_t(j)*k+pair*2+1],sums[j]);
            }
        }
        #pragma unroll
        for(unsigned j=0;j<8;++j){
            sums[j]=reduce_sum(sums[j]);
            if(tid%32==0)scratch8[(tid/32)*8+j]=sums[j];
        }
        __syncthreads();
        if(tid<32){
            #pragma unroll
            for(unsigned j=0;j<8;++j){
                float v=tid<8?scratch8[tid*8+j]:0.0f;
                v=reduce_sum(v);
                if(tid==j)y[size_t(j)*n+row]=v;
            }
        }
    } else {
        __shared__ float scratch[32];if(tid<32)scratch[tid]=0;__syncthreads();
        for(unsigned pair=tid;pair<k/2;pair+=256){const float w0=__bfloat162float(w[size_t(row)*k+pair*2]),w1=__bfloat162float(w[size_t(row)*k+pair*2+1]);
            for(unsigned j=0;j<M;++j){sums[j]=__fmaf_rn(w0,x[size_t(j)*k+pair*2],sums[j]);sums[j]=__fmaf_rn(w1,x[size_t(j)*k+pair*2+1],sums[j]);}}
        for(unsigned j=0;j<M;++j){sums[j]=reduce_sum(sums[j]);
            if(tid%32==0)scratch[tid/32]=sums[j];__syncthreads();
            if(tid<32)sums[j]=reduce_sum(scratch[tid]);__syncthreads();}
        if(tid<M)y[size_t(tid)*n+row]=sums[tid];
    }
}
inline cudaError_t launch_row_q8_input(const void* w,const float* x,void* packed,float* y,unsigned m,unsigned n,cudaStream_t stream){
    if(m<1||m>8||(n!=1024&&n!=6144&&n!=10240&&n!=12288))return cudaErrorInvalidValue;
    quantize<<<dim3(20,m),256,0,stream>>>(x,(Q8*)packed,5120);
    const cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return e;
    #define ROW_SMALL_CASE(M) case M: mmvq<M,1,true><<<(n+(M==1?0:1))/(M==1?1:2),dim3(32,M<=4?4:2),0,stream>>>((const uint8_t*)w,(Q8*)packed,y,n,5120,1.0f);break
    switch(m){ROW_SMALL_CASE(1);ROW_SMALL_CASE(2);ROW_SMALL_CASE(3);ROW_SMALL_CASE(4);ROW_SMALL_CASE(5);ROW_SMALL_CASE(6);ROW_SMALL_CASE(7);ROW_SMALL_CASE(8);}
    #undef ROW_SMALL_CASE
    return cudaGetLastError();
}
// Draft-only instantiations keep existing target dispatch unchanged. Integer
// dot totals and subsequent FP32 reduction order are identical to scalar MMVQ.
inline cudaError_t launch_dflash_q8(const void* w,const float* x,void* packed,float* y,unsigned m,unsigned n,unsigned k,cudaStream_t stream){
    if(m<1||m>8||k%256||n%2)return cudaErrorInvalidValue;
    quantize<<<dim3((k+255)/256,m),256,0,stream>>>(x,(Q8*)packed,k);
    const cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return e;
    #define DRAFT_CASE(M) case M: mmvq<M,1,false,true><<<(n+(M==1?0:1))/(M==1?1:2),dim3(32,M<=4?4:2),0,stream>>>((const uint8_t*)w,(Q8*)packed,y,n,k,1.0f);break
    switch(m){DRAFT_CASE(1);DRAFT_CASE(2);DRAFT_CASE(3);DRAFT_CASE(4);DRAFT_CASE(5);DRAFT_CASE(6);DRAFT_CASE(7);DRAFT_CASE(8);}
    #undef DRAFT_CASE
    return cudaGetLastError();
}
inline cudaError_t launch(const void* w,const float* x,void* packed,float* y,unsigned m,unsigned n,unsigned k,unsigned f,float scale,cudaStream_t stream){
    if(m<1||m>8||k%256||n%2||(f!=1&&f!=2&&f!=3)||(f==3&&(n!=48||k!=5120)))return cudaErrorInvalidValue;
    if(f!=3){quantize<<<dim3((k+255)/256,m),256,0,stream>>>(x,(Q8*)packed,k);
        const cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return e;}
    #define SMALL_CASE(M) case M: if(f==3)mmvf<M><<<n,256,0,stream>>>((const __nv_bfloat16*)w,x,y,n,k); else if(f==1)mmvq<M,1><<<(n+(M==1?0:1))/(M==1?1:2),dim3(32,M<=4?4:2),0,stream>>>((const uint8_t*)w,(Q8*)packed,y,n,k,scale); else mmvq<M,2><<<(n+(M==1?0:1))/(M==1?1:2),dim3(32,M<=4?4:2),0,stream>>>((const uint8_t*)w,(Q8*)packed,y,n,k,scale); break
    switch(m){SMALL_CASE(1);SMALL_CASE(2);SMALL_CASE(3);SMALL_CASE(4);SMALL_CASE(5);SMALL_CASE(6);SMALL_CASE(7);SMALL_CASE(8);}
    #undef SMALL_CASE
    return cudaGetLastError();
}
} // namespace llama_small
