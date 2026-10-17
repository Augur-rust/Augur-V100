// Explicit real-input MMA consumer, rows9..512. Legacy prefix ABI is unchanged.
// Stream-K partition and fixup order adapted from ggml, MIT (vendor/LICENSE).
// Included after the shared transform/pack/validate helpers in the SM70 TU.
#include "augur_qwen35_prefill_attention_policy.cuh"
namespace {
constexpr unsigned INPUT_MAX_ORDINAL=128;
struct InputPolicy {std::atomic<unsigned> sm_count{0};};
InputPolicy input_policies[INPUT_MAX_ORDINAL];
struct InputDispatch {unsigned cols,tiles,blocks,fixup;};
InputDispatch input_dispatch(unsigned sm_count,unsigned history,unsigned rows){
    const unsigned cols=rows<=16?16:32,tiles=((rows+cols-1)/cols)*12;
    const unsigned nkv=((history+255)&~255u)/32;
    unsigned blocks=tiles;
    if(100*tiles/(sm_count*((tiles+sm_count-1)/sm_count))<75){
        const unsigned raw=std::min(sm_count,nkv*tiles),rounded=(raw/tiles)*tiles;
        blocks=rounded && 100*(raw-rounded)/raw<=5?rounded:raw;
    }
#ifdef AUGUR_QWEN35_PLANAR_KV
    const unsigned partitions = augur_prefill_ninfer_partitions(rows, history);
    if(partitions)blocks=tiles*partitions;
#endif
    return {cols,tiles,blocks,blocks>tiles&&blocks%tiles==0?1u:tiles%blocks?2u:0u};
}
template<unsigned C, bool Compact = false, bool NinferTile = false> int input_attention_rows(cudaStream_t s,const float* q,const void* k,const void* v,
        float* y,float2* meta,unsigned history,unsigned rows,InputDispatch d){
    const unsigned physical=(history+255)&~255u,x=(rows+C-1)/C,nkv=physical/(NinferTile?64:32);
    static_assert(!Compact || C == 32, "compact combine is prefill C32 only");
    static_assert(!NinferTile || (C == 32 && Compact), "bounded Ninfer prefill");
    constexpr unsigned shared = NinferTile ? AUGUR_PREFILL_NINFER_SHARED_BYTES :
        Compact ? AUGUR_PREFILL_COMBINE_SHARED_BYTES : 67584U;
    flash_attn_ext_f16<256,256,C,2,false,false,Compact,NinferTile><<<d.blocks,dim3(32,NinferTile?8:4),shared,s>>>(
        reinterpret_cast<const char*>(q),reinterpret_cast<const char*>(k),reinterpret_cast<const char*>(v),
        nullptr,nullptr,nullptr,y,d.fixup?meta:nullptr,
        0.0625f,0.f,1.f,1.f,16,0.f,
        256,bounded_divisor(rows),24,1,24*256*4,256*4,rows*24*256*4,
        256,physical,4,1,1088,272,int64_t(physical)*1088,
        1088,272,int64_t(physical)*1088,
        rows,1,1,history*2,history*rows*2,int64_t(history)*rows*2);
    auto e=cudaGetLastError();if(e!=cudaSuccess)return int(e);
    if(d.fixup==1){
        flash_attn_stream_k_fixup_uniform<256,C,2><<<dim3(d.tiles,C,2),256,0,s>>>(
            y,meta,rows,24,4,d.blocks,6,d.blocks/d.tiles,
            bounded_divisor(x*12),bounded_divisor(x*3),bounded_divisor(x));
    }else if(d.fixup==2){
        flash_attn_stream_k_fixup_general<256,C,2><<<dim3(d.blocks,C,2),256,0,s>>>(
            y,meta,rows,24,6,nkv*d.tiles,bounded_divisor(nkv*x*12),bounded_divisor(nkv*x*3),
            bounded_divisor(nkv*x),bounded_divisor(nkv));
    }
    return int(cudaGetLastError());
}
} // namespace

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_input_validate_v1(
        const augur_cuda_qwen35_rotated_kv_args_v1* a){
    return result(validate(a,true)?AUGUR_CUDA_STATUS_OK:AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_input_configure_v1(int32_t ordinal){
    if(ordinal<0||unsigned(ordinal)>=INPUT_MAX_ORDINAL)return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
    if(input_policies[ordinal].sm_count.load(std::memory_order_acquire))return result(AUGUR_CUDA_STATUS_OK);
    unsigned sm_count=0,occupancy[2]{},shared[2]{};
    int status=augur_qwen35_rotated_input_calibrate_v1(ordinal,&sm_count,occupancy,shared);
    if(status)return result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,status);
    // SM70 shared memory67584 limits each reference template to one CTA/SM.
    // Fail closed if the calibrated contract changes; do not silently regroup.
    if(sm_count<1||sm_count>96||occupancy[0]!=1||occupancy[1]!=1||shared[0]!=67584||shared[1]!=67584)
        return result(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE);
    const fattn_kernel_t kernels[]={flash_attn_ext_f16<256,256,16,2,false,false>,
        flash_attn_ext_f16<256,256,32,2,false,false>};
    for(auto kernel:kernels){
        const auto e=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,67584);
        if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int(e));
    }
    // Configure only the new staging variant. Keep the reference calibration
    // above and its stream-K block/partition decisions unchanged.
    const auto compact = flash_attn_ext_f16<256,256,32,2,false,false,true>;
    auto e = cudaFuncSetAttribute(compact, cudaFuncAttributeMaxDynamicSharedMemorySize,
        AUGUR_PREFILL_COMBINE_SHARED_BYTES);
    if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int(e));
    int active = 0;
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, compact, 128,
        AUGUR_PREFILL_COMBINE_SHARED_BYTES);
    if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int(e));
    if(active!=2)return result(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE);
#ifdef AUGUR_QWEN35_PLANAR_KV
    const auto ninfer = flash_attn_ext_f16<256,256,32,2,false,false,true,true>;
    e = cudaFuncSetAttribute(ninfer, cudaFuncAttributeMaxDynamicSharedMemorySize,
        AUGUR_PREFILL_NINFER_SHARED_BYTES);
    if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int(e));
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, ninfer, 256,
        AUGUR_PREFILL_NINFER_SHARED_BYTES);
    if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int(e));
    if(active!=1)return result(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE);
#endif
    input_policies[ordinal].sm_count.store(sm_count,std::memory_order_release);
    return result(AUGUR_CUDA_STATUS_OK);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_input_submit_v1(
        int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_qwen35_rotated_kv_args_v1* a){
    if(!validate(a,true)||ordinal<0||unsigned(ordinal)>=INPUT_MAX_ORDINAL||!stream||!stream->stream||stream->ordinal!=ordinal)
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
    const unsigned sm_count=input_policies[ordinal].sm_count.load(std::memory_order_acquire);
    if(!sm_count)return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
    const auto e=cudaSetDevice(ordinal);if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int(e));
    const size_t nq=size_t(a->rows)*6144,nkv=size_t(a->rows)*1024;
    auto* q=reinterpret_cast<float*>(a->workspace);auto* k=q+nq;auto* v=k+nkv;
    auto* meta=reinterpret_cast<float2*>(v+nkv);
    auto* y=reinterpret_cast<float*>(a->attention);auto* gated=reinterpret_cast<float*>(a->gated);
    const auto s=stream->stream;
    int status=attention_transform(s,reinterpret_cast<const float*>(a->query),q,256,nq);
#ifdef AUGUR_QWEN35_PLANAR_KV
    if(!status)status=attention_transform_pack_q8_planar<256>(s,reinterpret_cast<const float*>(a->key),
        reinterpret_cast<unsigned char*>(a->key_cache),a->position,a->rows,a->capacity);
    if(!status)status=attention_transform_pack_q8_planar<64>(s,reinterpret_cast<const float*>(a->value),
        reinterpret_cast<unsigned char*>(a->value_cache),a->position,a->rows,a->capacity);
#else
    if(!status)status=attention_transform(s,reinterpret_cast<const float*>(a->key),k,256,nkv);
    if(!status)status=attention_transform(s,reinterpret_cast<const float*>(a->value),v,64,nkv);
    const unsigned span=((a->position+a->rows+255)&~255u)-a->position;
    if(!status){
        pack_q8_cache<256><<<(span*32+127)/128,128,0,s>>>(k,reinterpret_cast<unsigned char*>(a->key_cache),a->position,a->rows);
        status=int(cudaGetLastError());
    }
    if(!status){
        pack_q8_cache<256><<<(span*32+127)/128,128,0,s>>>(v,reinterpret_cast<unsigned char*>(a->value_cache),a->position,a->rows);
        status=int(cudaGetLastError());
    }
#endif
    if(!status){
        const auto d=input_dispatch(sm_count,a->position+a->rows,a->rows);
        const auto* kc=reinterpret_cast<const void*>(a->key_cache);const auto* vc=reinterpret_cast<const void*>(a->value_cache);
        const unsigned history=a->position+a->rows;
#ifdef AUGUR_QWEN35_PLANAR_KV
        const unsigned partitions = augur_prefill_ninfer_partitions(a->rows,history);
        if(partitions && d.blocks==d.tiles*partitions &&
                ((partitions==1 && d.fixup==0) || (partitions>1 && d.fixup==1))) {
            status=input_attention_rows<32,true,true>(s,q,kc,vc,y,meta,history,a->rows,d);
        } else
#endif
        if(augur_prefill_compact_combine(a->rows,history,d.fixup,
                d.blocks == d.tiles*2 && augur_prefill_split2(a->rows,history))) {
            status=input_attention_rows<32,true>(s,q,kc,vc,y,meta,history,a->rows,d);
        } else {
            status=d.cols==16?input_attention_rows<16>(s,q,kc,vc,y,meta,history,a->rows,d):
                input_attention_rows<32>(s,q,kc,vc,y,meta,history,a->rows,d);
        }
    }
    if(!status)status=attention_transform(s,y,y,64,nq);
    if(!status)status=attention_gate(s,y,reinterpret_cast<const float*>(a->gate),gated,nq);
    return result(status?AUGUR_CUDA_STATUS_SUBMISSION_FAILURE:AUGUR_CUDA_STATUS_OK,status);
}
