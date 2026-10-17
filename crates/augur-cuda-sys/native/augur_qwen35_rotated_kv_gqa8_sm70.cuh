// Research GQA8 B1..8 consumer. Legacy prefix and default small ABI unchanged.
// Stream-K partition and fixup order adapted from ggml, MIT (vendor/LICENSE).
// Included after the shared transform/pack/validate helpers in the SM70 TU.
namespace {
constexpr unsigned INPUT_MAX_ORDINAL=128;
#ifdef AUGUR_QWEN35_GQA8_C4
// C4 shared-Q(16896)+K/V(33792)+mask(320); combine requires34816.
constexpr unsigned INPUT_SHARED_BYTES=51008;
constexpr unsigned INPUT_QUERY_COLS=4;
#elif defined(AUGUR_QWEN35_PLANAR_KV)
constexpr unsigned INPUT_QUERY_COLS=8;
// Q(33792)+K(16896)+V(16896)+mask(640); old combine scratch was67584.
constexpr unsigned INPUT_SHARED_BYTES=68224;
#else
constexpr unsigned INPUT_QUERY_COLS=8;
constexpr unsigned INPUT_SHARED_BYTES=67584;
#endif
struct InputPolicy {std::atomic<unsigned> sm_count{0};};
InputPolicy input_policies[INPUT_MAX_ORDINAL];
struct InputDispatch {unsigned cols,tiles,blocks,fixup;};
InputDispatch input_dispatch(unsigned sm_count,unsigned history,unsigned rows){
    const unsigned cols=INPUT_QUERY_COLS,tiles=((rows+cols-1)/cols)*4;
#ifdef AUGUR_QWEN35_GQA8_C4
    // Keep the B1 partition count per output tile, including a second C4 tile.
    // max192 CTA *32 columns fits the unchanged96 CTA *64-column metadata.
    const unsigned nkv=((history+255)&~255u)/32;
    const unsigned blocks=std::min(sm_count,nkv*4)*(tiles/4);
    return {cols,tiles,blocks,blocks>tiles?1u:0u};
#else
    const unsigned nkv=((history+255)&~255u)/32;
    unsigned blocks=tiles;
    if(100*tiles/(sm_count*((tiles+sm_count-1)/sm_count))<75){
        const unsigned raw=std::min(sm_count,nkv*tiles),rounded=(raw/tiles)*tiles;
        blocks=rounded && 100*(raw-rounded)/raw<=5?rounded:raw;
    }
    return {cols,tiles,blocks,blocks>tiles&&blocks%tiles==0?1u:tiles%blocks?2u:0u};
#endif
}
template<unsigned C> int input_attention_rows(cudaStream_t s,const float* q,const void* k,const void* v,
        float* y,float2* meta,unsigned history,unsigned rows,InputDispatch d){
    const unsigned physical=(history+255)&~255u,x=(rows+C-1)/C,nkv=physical/32;
    small_mma_gqa8_ext_f16<256,256,C,8,false,false><<<d.blocks,dim3(32,4),INPUT_SHARED_BYTES,s>>>(
        reinterpret_cast<const char*>(q),reinterpret_cast<const char*>(k),reinterpret_cast<const char*>(v),
        nullptr,nullptr,nullptr,y,d.fixup?meta:nullptr,
        0.0625f,0.f,1.f,1.f,16,0.f,
        256,bounded_divisor(rows),24,1,24*256*4,256*4,rows*24*256*4,
        256,physical,4,1,1088,272,int64_t(physical)*1088,
        1088,272,int64_t(physical)*1088,
        rows,1,1,history*2,history*rows*2,int64_t(history)*rows*2);
    auto e=cudaGetLastError();if(e!=cudaSuccess)return int(e);
    if(d.fixup==1){
        flash_attn_stream_k_fixup_uniform<256,C,8><<<dim3(d.tiles,C,8),256,0,s>>>(
            y,meta,rows,24,4,d.blocks,6,d.blocks/d.tiles,
            bounded_divisor(x*4),bounded_divisor(x*1),bounded_divisor(x));
    }else if(d.fixup==2){
        flash_attn_stream_k_fixup_general<256,C,8><<<dim3(d.blocks,C,8),256,0,s>>>(
            y,meta,rows,24,6,nkv*d.tiles,bounded_divisor(nkv*x*4),bounded_divisor(nkv*x*1),
            bounded_divisor(nkv*x),bounded_divisor(nkv));
    }
    return int(cudaGetLastError());
}
} // namespace

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_gqa8_validate_v1(
        const augur_cuda_qwen35_rotated_kv_args_v1* a){
    return result(validate(a,true)?AUGUR_CUDA_STATUS_OK:AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_gqa8_configure_v1(int32_t ordinal){
    if(ordinal<0||unsigned(ordinal)>=INPUT_MAX_ORDINAL)return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
    if(input_policies[ordinal].sm_count.load(std::memory_order_acquire))return result(AUGUR_CUDA_STATUS_OK);
    unsigned sm_count=0,occupancy[2]{},shared[2]{};
    int status=augur_qwen35_rotated_input_calibrate_v1(ordinal,&sm_count,occupancy,shared);
    if(status)return result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,status);
    // SM70 shared memory67584 limits each reference template to one CTA/SM.
    // Fail closed if the calibrated contract changes; do not silently regroup.
    if(sm_count<1||sm_count>96||occupancy[0]!=1||occupancy[1]!=1||shared[0]!=67584||shared[1]!=67584)
        return result(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE);
#ifdef AUGUR_QWEN35_GQA8_C4
    // The C4 component/row-equivalence evidence currently covers this GPU.
    if(sm_count!=72)return result(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE);
#endif
    const fattn_kernel_t kernels[]={small_mma_gqa8_ext_f16<256,256,INPUT_QUERY_COLS,8,false,false>};
    for(auto kernel:kernels){
        const auto e=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,INPUT_SHARED_BYTES);
        if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int(e));
    }
    input_policies[ordinal].sm_count.store(sm_count,std::memory_order_release);
    return result(AUGUR_CUDA_STATUS_OK);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_gqa8_submit_v1(
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
    if(!status){
        const auto* kc=reinterpret_cast<const void*>(a->key_cache);
        const auto* vc=reinterpret_cast<const void*>(a->value_cache);
        // A row must use the same physical extent/Stream-K partition as B1.
        // At most two cohorts exist for a suffix of at most eight real rows.
        for(unsigned first=0;first<a->rows&&!status;){
            const unsigned physical=(a->position+first+256)&~255u;
            const unsigned count=std::min(a->rows-first,physical-a->position-first);
            const unsigned history=a->position+first+count;
            const auto d=input_dispatch(sm_count,history,count);
            status=input_attention_rows<INPUT_QUERY_COLS>(s,q+size_t(first)*6144,kc,vc,
                y+size_t(first)*6144,meta,history,count,d);
            first+=count;
        }
    }
    if(!status)status=attention_transform(s,y,y,64,nq);
    if(!status)status=attention_gate(s,y,reinterpret_cast<const float*>(a->gate),gated,nq);
    return result(status?AUGUR_CUDA_STATUS_SUBMISSION_FAILURE:AUGUR_CUDA_STATUS_OK,status);
}
