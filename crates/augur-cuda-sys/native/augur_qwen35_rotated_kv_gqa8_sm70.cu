// Default-off native adapter. Not product-qualified.
// FWHT/quantization order adapted from ggml CUDA fwht.cu/cpy-utils.cuh.
// Direct large-query template is compiled from pinned local ggml headers.
// Copyright (c) 2023-2026 The ggml authors. MIT; see
// llama_attention_vendor/LICENSE alongside the adapted kernel.
#include "llama_attention_vendor/augur_gqa8_cohort_sm70.cuh"
#include "augur_cuda_internal.h"
#include "augur_qwen35_rotated_kv_abi.h"
#include <cuda_fp16.h>
#include <limits>
#include <atomic>
extern "C" int augur_qwen35_rotated_input_calibrate_v1(int,unsigned*,unsigned*,unsigned*);

namespace {
static_assert(sizeof(augur_cuda_qwen35_rotated_kv_args_v1)==168,"rotated KV ABI drift");
uint3 bounded_divisor(unsigned rows){
    unsigned shift=0;while((1u<<shift)<rows)++shift;
    const unsigned multiplier=unsigned((uint64_t(1)<<32)*((uint64_t(1)<<shift)-rows)/rows+1);
    return make_uint3(multiplier,shift,rows);
}

template<int N> __global__ void transform_rows(const float* src,float* dst,size_t rows){
    const size_t row=size_t(blockIdx.x)*4+threadIdx.y;
    if(row>=rows)return;
    const unsigned lane=threadIdx.x;float v[N/32];
    #pragma unroll
    for(unsigned i=0;i<N/32;++i)v[i]=src[row*N+i*32+lane]*(N==64?0.125f:0.0625f);
    #pragma unroll
    for(unsigned h=1;h<32;h*=2){
        #pragma unroll
        for(unsigned i=0;i<N/32;++i){float x=v[i],y=__shfl_xor_sync(0xffffffff,x,h);v[i]=(lane&h)?y-x:x+y;}
    }
    #pragma unroll
    for(unsigned h=32;h<N;h*=2){
        #pragma unroll
        for(unsigned i=0;i<N/32;i+=2*(h/32)){
            #pragma unroll
            for(unsigned j=0;j<h/32;++j){float x=v[i+j],y=v[i+j+h/32];v[i+j]=x+y;v[i+j+h/32]=x-y;}
        }
    }
    #pragma unroll
    for(unsigned i=0;i<N/32;++i)dst[row*N+i*32+lane]=v[i];
}
int attention_transform(cudaStream_t s,const float* x,float* y,unsigned n,size_t elements){
    if(!x||!y||(n!=64&&n!=256)||elements==0||elements%n||elements>512ULL*6144)return 1;
    const size_t rows=elements/n;const dim3 grid(unsigned((rows+3)/4)),block(32,4);
    if(n==64)transform_rows<64><<<grid,block,0,s>>>(x,y,rows);else transform_rows<256><<<grid,block,0,s>>>(x,y,rows);
    return int(cudaGetLastError());
}
template<unsigned ALIGN> __global__ void pack_q8_cache(const float* x,unsigned char* cache,unsigned base,unsigned rows){
    const unsigned span=((base+rows+ALIGN-1)&~(ALIGN-1))-base;
    const unsigned b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=span*32)return;
    const float* input=x+size_t(b)*32;
#ifdef AUGUR_QWEN35_PLANAR_KV
    const size_t block=size_t(base)*32+b;
    unsigned char* output=cache+(block/8)*272+(block%8)*32;
    auto* scale_output=reinterpret_cast<__half*>(cache+(block/8)*272+256+2*(block%8));
#else
    unsigned char* output=cache+(size_t(base)*32+b)*34;
#endif
    if(b>=rows*32){
#ifdef AUGUR_QWEN35_PLANAR_KV
        for(unsigned j=0;j<16;++j)reinterpret_cast<unsigned short*>(output)[j]=0;
        *scale_output=__ushort_as_half(0);
#else
        #pragma unroll
        for(unsigned j=0;j<17;++j)reinterpret_cast<unsigned short*>(output)[j]=0;
#endif
        return;
    }
    float amax=0;for(unsigned j=0;j<32;++j)amax=fmaxf(amax,fabsf(input[j]));
    const float scale=amax/127.f,inverse=scale?1.f/scale:0;
#ifdef AUGUR_QWEN35_PLANAR_KV
    *scale_output=__float2half_rn(scale);
    for(unsigned j=0;j<32;++j)output[j]=static_cast<unsigned char>(static_cast<int8_t>(roundf(input[j]*inverse)));
#else
    *reinterpret_cast<__half*>(output)=__float2half_rn(scale);
    for(unsigned j=0;j<32;++j)output[2+j]=static_cast<unsigned char>(static_cast<int8_t>(roundf(input[j]*inverse)));
#endif
}
int attention_pack(cudaStream_t s,const float* x,unsigned char* cache,unsigned base,unsigned rows,unsigned capacity){
    if(!x||!cache||!rows||rows>512||base>capacity||rows>capacity-base||capacity>196608||capacity%32)return 1;
    const unsigned span=((base+rows+31u)&~31u)-base;
    pack_q8_cache<32><<<(span*32+127)/128,128,0,s>>>(x,cache,base,rows);return int(cudaGetLastError());
}

__global__ void gate_rows(const float* x,const float* gate,float* y,size_t n){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)y[i]=x[i]*(1.f/(1.f+expf(-gate[i])));
}
int attention_gate(cudaStream_t s,const float* x,const float* gate,float* y,size_t n){
    if(!x||!gate||!y||!n||n>512ULL*6144)return 1;
    gate_rows<<<unsigned((n+255)/256),256,0,s>>>(x,gate,y,n);return int(cudaGetLastError());
}
int attention_rows(cudaStream_t s,const float* q,const void* k,const void* v,const void* mask,float* y,unsigned history,unsigned rows){
    // B1..512 contiguous single-sequence suffix. Complete output tiles,
    // no stream-K seam fixup and no global mask/F16 KV scratch.
    if(!q||!k||!v||mask||!y||!rows||rows>512||history<rows||history>196608)return 1;
    constexpr size_t shared=67584;
    const unsigned physical_history=(history+31u)&~31u;
    auto kernel=small_mma_gqa8_ext_f16<256,256,32,2,false,false>;
    kernel<<<((rows+31)/32)*12,dim3(32,4),shared,s>>>(
        reinterpret_cast<const char*>(q),reinterpret_cast<const char*>(k),reinterpret_cast<const char*>(v),
        reinterpret_cast<const char*>(mask),nullptr,nullptr,y,nullptr,
        0.0625f,0.f,1.f,1.f,16,0.f,
        256,bounded_divisor(rows),24,1,24*256*4,256*4,rows*24*256*4,
        256,physical_history,4,1,1088,272,int64_t(physical_history)*1088,
        1088,272,int64_t(physical_history)*1088,
        rows,1,1,history*2,history*rows*2,int64_t(history)*rows*2);
    return int(cudaGetLastError());
}

augur_cuda_result_v1 result(int status,int raw=0){return {status,raw};}
constexpr uint64_t INPUT_META_BYTES=uint64_t(96)*64*130*sizeof(float2);
bool validate(const augur_cuda_qwen35_rotated_kv_args_v1* a,bool exact_input=false){
#ifdef AUGUR_QWEN35_PLANAR_KV
    constexpr unsigned format=2;
#else
    constexpr unsigned format=1;
#endif
    if(!a||a->abi_version!=1||a->struct_size!=sizeof(*a)||a->cache_format!=format||
       !a->capacity||a->capacity>(format==2?262144u:196608u)||a->capacity%32||!a->rows||a->rows>8||
       a->position>=a->capacity||a->rows>a->capacity-a->position)return false;
    if(exact_input&&(a->rows<1||a->capacity%256||a->workspace%16))return false;
    const uint64_t q=uint64_t(a->rows)*6144*4,k=uint64_t(a->rows)*1024*4,c=uint64_t(a->capacity)*1088;
    struct Range{uint64_t p,n,expected;};
    const Range ranges[]={{a->query,a->query_bytes,q},{a->gate,a->gate_bytes,q},
        {a->key,a->key_bytes,k},{a->value,a->value_bytes,k},{a->key_cache,a->key_cache_bytes,c},
        {a->value_cache,a->value_cache_bytes,c},{a->attention,a->attention_bytes,q},{a->gated,a->gated_bytes,q},
        {a->workspace,a->workspace_bytes,q+2*k+(exact_input?INPUT_META_BYTES:0)}};
    for(unsigned i=0;i<9;++i){
        const auto& r=ranges[i];
        if(!r.p||r.p%4||r.n!=r.expected||r.p>std::numeric_limits<uint64_t>::max()-r.n)return false;
        for(unsigned j=0;j<i;++j)if(r.p<ranges[j].p+ranges[j].n&&ranges[j].p<r.p+r.n)return false;
    }
    return true;
}
} // namespace


#include "augur_qwen35_rotated_kv_gqa8_sm70.cuh"
