// Default-off B1..B8 consumer of the same rotated Q8 cache representation.
// MIT, Copyright (c) 2023-2026 The ggml authors; see llama_attention_vendor/LICENSE.
#include "llama_attention_vendor/augur_vec_sync_sm70.cuh"
#include "llama_attention_vendor/augur_packed_tile_sm70.cuh"
#include "augur_cuda_internal.h"
#include "augur_qwen35_rotated_kv_abi.h"
#include <atomic>
#include <limits>
extern "C" int augur_qwen35_rotated_small_calibrate_v1(int,unsigned*,unsigned*);

namespace {
constexpr unsigned MAX_ORDINAL=128,MAX_PARTS=32;
// Keep the original tile8 partition count and live-row arithmetic. Qualify the
// physical history, not the initial prompt length: a 32768-token input grows
// into physical33024 on its first decode and must not lose the compact kernel.
constexpr bool compact_b1_admitted(unsigned policy,unsigned count,unsigned physical) {
    return policy==2 && count==1 && physical>=8192 && physical<=65536;
}
static_assert(!compact_b1_admitted(2,1,7936) && compact_b1_admitted(2,1,8192));
static_assert(compact_b1_admitted(2,1,32768) && compact_b1_admitted(2,1,33024));
static_assert(compact_b1_admitted(2,1,65536) && !compact_b1_admitted(2,1,65792));
static_assert(!compact_b1_admitted(1,1,33024) && !compact_b1_admitted(0,1,33024));
static_assert(!compact_b1_admitted(2,2,33024) && !compact_b1_admitted(2,8,33024));

struct Policy {std::atomic<unsigned> sm_count{0};std::atomic<unsigned> occupancy[4]{};};
Policy policies[MAX_ORDINAL];
uint3 divisor(unsigned rows){
    // Validated rows1..8. Same integer construction without ggml's host abort
    // dependency; this native archive must link into Rust without a ggml DLL.
    unsigned shift=0;while((1u<<shift)<rows)++shift;
    return make_uint3(unsigned((uint64_t(1)<<32)*((uint64_t(1)<<shift)-rows)/rows+1),shift,rows);
}
uint64_t workspace_size(unsigned capacity,unsigned rows){
    return uint64_t(rows)*(32768+uint64_t(capacity)*2+6144*MAX_PARTS*4+24*MAX_PARTS*8);
}
bool validate(const augur_cuda_qwen35_rotated_kv_args_v1* a){
    if(!a||a->abi_version!=1||a->struct_size!=sizeof(*a)||a->cache_format!=1||!a->capacity||(a->capacity>196608&&!(a->capacity==262144&&a->rows==1))||
       a->capacity%256||!a->rows||a->rows>8||a->workspace%16||a->position>=a->capacity||a->rows>a->capacity-a->position)return false;
    const uint64_t q=uint64_t(a->rows)*6144*4,k=uint64_t(a->rows)*1024*4,c=uint64_t(a->capacity)*1088;
    struct Range{uint64_t p,n,expected;};
    const Range ranges[]={{a->query,a->query_bytes,q},{a->gate,a->gate_bytes,q},{a->key,a->key_bytes,k},{a->value,a->value_bytes,k},
        {a->key_cache,a->key_cache_bytes,c},{a->value_cache,a->value_cache_bytes,c},{a->attention,a->attention_bytes,q},
        {a->gated,a->gated_bytes,q},{a->workspace,a->workspace_bytes,workspace_size(a->capacity,a->rows)}};
    for(unsigned i=0;i<9;++i){const auto& r=ranges[i];
        if(!r.p||r.p%4||r.n!=r.expected||r.p>std::numeric_limits<uint64_t>::max()-r.n)return false;
        for(unsigned j=0;j<i;++j)if(r.p<ranges[j].p+ranges[j].n&&ranges[j].p<r.p+r.n)return false;
    }
    return true;
}
augur_cuda_result_v1 result(int status,int raw=0){return {status,raw};}
template<int N> __global__ void rotate(const float* src,float* dst,size_t rows){
    const size_t row=size_t(blockIdx.x)*4+threadIdx.y;if(row>=rows)return;
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
__global__ void pack(const float* src,unsigned char* dst,unsigned position,unsigned rows,unsigned span){
    const unsigned b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=span*32)return;
    auto* out=dst+(size_t(position)*32+b)*34;
    if(b>=rows*32){for(unsigned j=0;j<17;++j)reinterpret_cast<unsigned short*>(out)[j]=0;return;}
    const float* input=src+size_t(b)*32;float amax=0;
    for(unsigned j=0;j<32;++j)amax=fmaxf(amax,fabsf(input[j]));
    const float scale=amax/127.f,inverse=scale?1.f/scale:0;
    *reinterpret_cast<half*>(out)=__float2half_rn(scale);
    for(unsigned j=0;j<32;++j)out[2+j]=static_cast<unsigned char>(static_cast<int8_t>(roundf(input[j]*inverse)));
}
__global__ void causal_mask(half* mask,unsigned physical,unsigned position,unsigned rows){
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=physical*rows)return;
    mask[i]=__float2half_rn(i%physical<=position+i/physical?0.f:-INFINITY);
}
__global__ void gate_output(const float* x,const float* gate,float* y,unsigned count){
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)y[i]=x[i]*(1.f/(1.f+expf(-gate[i])));
}
unsigned partitions(unsigned sm_count,unsigned max_blocks,unsigned rows,unsigned physical){
    const unsigned kv_tiles=physical/(rows==1?256:64),dst_tiles=rows==1?24:12,wave=sm_count*max_blocks;
    unsigned selected=std::min(max_blocks,kv_tiles),best_eff=0,best_waves=0;
    for(unsigned split=selected;split<=kv_tiles;++split){
        const unsigned blocks=dst_tiles*split,waves=(blocks+wave-1)/wave,eff=100*blocks/(waves*wave);
        if(best_eff>=95&&waves>best_waves)break;
        if(eff>best_eff){best_eff=eff;best_waves=waves;selected=split;}
    }
    return selected;
}
} // namespace

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_small_validate_v1(const augur_cuda_qwen35_rotated_kv_args_v1* a){
    return result(validate(a)?AUGUR_CUDA_STATUS_OK:AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_small_configure_v1(int32_t ordinal){
    if(ordinal<0||unsigned(ordinal)>=MAX_ORDINAL)return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
    if(policies[ordinal].sm_count.load(std::memory_order_acquire))return result(AUGUR_CUDA_STATUS_OK);
    unsigned sm_count=0,occupancy[4]{};const int e=augur_qwen35_rotated_small_calibrate_v1(ordinal,&sm_count,occupancy);
    if(e)return result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,e);
    for(unsigned i=0;i<4;++i)policies[ordinal].occupancy[i].store(occupancy[i],std::memory_order_relaxed);
    policies[ordinal].sm_count.store(sm_count,std::memory_order_release);return result(AUGUR_CUDA_STATUS_OK);
}
static augur_cuda_result_v1 small_submit(int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_qwen35_rotated_kv_args_v1* a,unsigned row_policy){
    if(row_policy>2||!validate(a)||ordinal<0||unsigned(ordinal)>=MAX_ORDINAL||!stream||!stream->stream||stream->ordinal!=ordinal)return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
    const bool vector_rows=row_policy==1;
    const unsigned sm_count=policies[ordinal].sm_count.load(std::memory_order_acquire);
    if(!sm_count)return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
    const unsigned rows=a->rows,physical=(a->position+rows+255)&~255u,span=physical-a->position;
    const unsigned dispatch_rows=vector_rows?1:row_policy==2?8:rows;
    const unsigned slot=dispatch_rows==1?0:dispatch_rows==2?1:dispatch_rows<=4?2:3;
    const unsigned parts=partitions(sm_count,policies[ordinal].occupancy[slot].load(std::memory_order_relaxed),dispatch_rows,physical);
    if(!parts||parts>MAX_PARTS)return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
    auto e=cudaSetDevice(ordinal);if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int(e));
    auto* q=reinterpret_cast<float*>(a->workspace);auto* k=q+rows*6144;auto* v=k+rows*1024;
    auto* mask=reinterpret_cast<half*>(v+rows*1024);
    auto* partial=reinterpret_cast<float*>(mask+size_t(a->capacity)*rows);
    auto* meta=reinterpret_cast<float2*>(partial+size_t(rows)*6144*MAX_PARTS);
    auto* y=reinterpret_cast<float*>(a->attention);const auto s=stream->stream;
    rotate<256><<<((rows*24)+3)/4,dim3(32,4),0,s>>>(reinterpret_cast<const float*>(a->query),q,rows*24);
    e=cudaGetLastError();if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,int(e));
    rotate<256><<<rows,dim3(32,4),0,s>>>(reinterpret_cast<const float*>(a->key),k,rows*4);
    e=cudaGetLastError();if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,int(e));
    rotate<64><<<rows*4,dim3(32,4),0,s>>>(reinterpret_cast<const float*>(a->value),v,rows*16);
    e=cudaGetLastError();if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,int(e));
    pack<<<(span*32+127)/128,128,0,s>>>(k,reinterpret_cast<unsigned char*>(a->key_cache),a->position,rows,span);
    e=cudaGetLastError();if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,int(e));
    pack<<<(span*32+127)/128,128,0,s>>>(v,reinterpret_cast<unsigned char*>(a->value_cache),a->position,rows,span);
    e=cudaGetLastError();if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,int(e));
    causal_mask<<<(physical*rows+255)/256,256,0,s>>>(mask,physical,a->position,rows);
    e=cudaGetLastError();if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,int(e));
    if(row_policy){
        // Preserve each B1 query's arithmetic and partition count, but execute
        // independent queries as grid.x blocks. A B8 suffix can cross at most
        // one physical256 boundary. Split that boundary into two cohorts so a
        // query never inherits a later row's padded extent/reduction schedule.
        // Policy2 instead fixes tile8 arithmetic for ALL B1..B8 invocations.
        // Its inactive query lanes repeat via the kernel's modulo loader and
        // are not written; real queries retain the tile's KV sharing.
        for(unsigned first=0;first<rows;){
            const unsigned row_physical=(a->position+first+256)&~255u;
            const unsigned count=std::min(rows-first,row_physical-a->position-first);
            const unsigned row_parts=partitions(sm_count,policies[ordinal].occupancy[slot].load(std::memory_order_relaxed),dispatch_rows,row_physical);
            if(!row_parts||row_parts>MAX_PARTS)return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
            auto* row_y=y+size_t(first)*6144;
            // Do not recalibrate splits from the smaller kernel: tile8 occupancy
            // above fixes the live-row reduction grouping. Only a single-query
            // cohort in the qualified 8192..65536 extent uses the compact schedule.
            // A B8 boundary cohort of one is the same B1 contract; other rows and
            // short contexts retain the existing kernel unchanged.
            const bool compact_b1=compact_b1_admitted(row_policy,count,row_physical);
            auto row_kernel=vector_rows?augur_qwen35_small_vec_sync_v1<256,1,GGML_TYPE_Q8_0,GGML_TYPE_Q8_0,false>:
                compact_b1?flash_attn_tile<256,256,4,2,false,true>:flash_attn_tile<256,256,8,2,false>;
            row_kernel<<<dim3(vector_rows?count:1,row_parts,vector_rows?24:12),dim3(32,vector_rows||compact_b1?4:8),0,s>>>(
                reinterpret_cast<const char*>(q+size_t(first)*6144),reinterpret_cast<const char*>(a->key_cache),reinterpret_cast<const char*>(a->value_cache),
                reinterpret_cast<const char*>(mask+size_t(first)*physical),nullptr,nullptr,row_parts>1?partial:row_y,row_parts>1?meta:nullptr,
                0.0625f,0.f,1.f,1.f,16,0.f,256,divisor(count),24,1,6144*4,256*4,count*6144*4,
                256,row_physical,4,1,1088,272,int64_t(row_physical)*1088,1088,272,int64_t(row_physical)*1088,
                count,1,1,physical*2,physical*count*2,int64_t(physical)*count*2);
            e=cudaGetLastError();if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,int(e));
            if(row_parts>1){flash_attn_combine_results<256><<<dim3(count,24),256,row_parts*sizeof(float2),s>>>(partial,meta,row_y,row_parts);
                e=cudaGetLastError();if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,int(e));}
            first+=count;
        }
    }else{
    auto kernel=rows==1?augur_qwen35_small_vec_sync_v1<256,1,GGML_TYPE_Q8_0,GGML_TYPE_Q8_0,false>:
        rows==2?flash_attn_tile<256,256,2,2,false>:rows<=4?flash_attn_tile<256,256,4,2,false>:flash_attn_tile<256,256,8,2,false>;
    kernel<<<dim3(1,parts,rows==1?24:12),dim3(32,rows<=2?4:8),0,s>>>(
        reinterpret_cast<const char*>(q),reinterpret_cast<const char*>(a->key_cache),reinterpret_cast<const char*>(a->value_cache),
        reinterpret_cast<const char*>(mask),nullptr,nullptr,parts>1?partial:y,parts>1?meta:nullptr,
        0.0625f,0.f,1.f,1.f,16,0.f,256,divisor(rows),24,1,6144*4,256*4,rows*6144*4,
        256,physical,4,1,1088,272,int64_t(physical)*1088,1088,272,int64_t(physical)*1088,
        rows,1,1,physical*2,physical*rows*2,int64_t(physical)*rows*2);
    e=cudaGetLastError();if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,int(e));
    if(parts>1){flash_attn_combine_results<256><<<dim3(rows,24),256,parts*sizeof(float2),s>>>(partial,meta,y,parts);
        e=cudaGetLastError();if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,int(e));}
    }
    rotate<64><<<rows*24,dim3(32,4),0,s>>>(y,y,rows*96);
    e=cudaGetLastError();if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,int(e));
    gate_output<<<rows*24,256,0,s>>>(y,reinterpret_cast<const float*>(a->gate),reinterpret_cast<float*>(a->gated),rows*6144);
    e=cudaGetLastError();return result(e==cudaSuccess?AUGUR_CUDA_STATUS_OK:AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,int(e));
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_small_submit_v1(int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_qwen35_rotated_kv_args_v1* a){
    return small_submit(ordinal,stream,a,0);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_small_vector_submit_v1(int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_qwen35_rotated_kv_args_v1* a){
    return small_submit(ordinal,stream,a,1);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_small_tile8_submit_v1(int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_qwen35_rotated_kv_args_v1* a){
    return small_submit(ordinal,stream,a,2);
}
