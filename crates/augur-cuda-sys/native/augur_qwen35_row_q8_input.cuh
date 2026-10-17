// Closed Qwen35 input-only row planes. Wire Q8 and whole-plane head APIs do
// not accept format5. Metadata copies below reuse range/extent validation;
// transformed weight pointers are NEVER submitted to a legacy wire reader.
extern "C" cudaError_t augur_qwen35_launch_row_q8_input_mmq_sm70(
    const void*,const float*,void*,float*,float*,unsigned,unsigned,unsigned,
    unsigned,float,unsigned,cudaStream_t);
namespace {
// Scalar replay must not pay for eight masked accumulators/shared planes.
// Preserve the wire matvec's K assignment, scale/FMA and four-warp sum order.
__global__ __launch_bounds__(128,1) void row_q8_input_scalar_kernel(
    const uint8_t* weight,const uint8_t* activation,float* output,uint32_t cols){
    const uint32_t blocks=cols/32U,thread=threadIdx.x;
    const uint32_t lane=thread&31U,warp=thread/32U;
    const auto* row=weight+uint64_t(blockIdx.x)*blocks*34U;
    float partial=0;
    for(uint32_t block=thread/4U;block<blocks;block+=32U){
        const uint32_t offset=(thread&3U)*8U;
        const auto* w=reinterpret_cast<const int32_t*>(row+block*32U+offset);
        const auto* a=activation+uint64_t(block)*36U;
        const auto* x=reinterpret_cast<const int32_t*>(a+4U+offset);
        int32_t dot=__dp4a(w[0],x[0],0);dot=__dp4a(w[1],x[1],dot);
        const float scale=__half2float(__ushort_as_half(
            reinterpret_cast<const uint16_t*>(row+blocks*32U)[block]));
        const float combined=scale*__half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(a)));
        partial=fmaf(combined,static_cast<float>(dot),partial);
    }
    __shared__ float others[3][32];
    if(warp)others[warp-1U][lane]=partial;
    __syncthreads();if(warp)return;
    #pragma unroll
    for(uint32_t i=0;i<3;++i)partial+=others[i][lane];
    #pragma unroll
    for(uint32_t i=16;i;i>>=1)partial+=__shfl_down_sync(0xffffffffU,partial,i);
    if(!lane)output[blockIdx.x]=partial;
}

bool row_q8_input_shape(uint32_t format,uint32_t rows,uint32_t cols){
    return format==AUGUR_CUDA_QWEN35_ROW_Q8_INPUT_WEIGHT_V1 && cols==5120U
        && (rows==1024U||rows==6144U||rows==10240U||rows==12288U);
}
bool row_q8_input_packed_valid(int32_t ordinal,augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* a,bool inspect){
    if(!a||!row_q8_input_shape(a->weight_format,a->rows,a->cols)||a->weight%4U)return false;
    auto logical=*a;logical.weight_format=AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1;
    return packed_projection_metadata_valid(ordinal,stream,&logical,1U,inspect);
}
bool row_q8_input_original_valid(int32_t ordinal,augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* a,bool inspect,Footprint* f){
    if(!a||!row_q8_input_shape(a->weight_format,a->rows,a->cols)||a->weight%4U)return false;
    auto logical=*a;logical.weight_format=AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1;
    return llama_projection_metadata_valid(ordinal,stream,plan,&logical,inspect,f,true);
}

__global__ void row_q8_input_decode_shared128(const uint8_t* w,__half* out){
    __shared__ uint32_t packed[544];
    const uint64_t base=uint64_t(blockIdx.x)*2048;
    for(unsigned i=threadIdx.x;i<512;i+=128){
        const uint64_t at=base+uint64_t(i)*4;
        packed[i]=*reinterpret_cast<const uint32_t*>(w+at+(at/5120)*320);
    }
    if(threadIdx.x<32){
        const uint64_t at=base+uint64_t(threadIdx.x)*64;
        packed[512+threadIdx.x]=*reinterpret_cast<const uint32_t*>(w+5120+(at/5120)*5120+(at/32)*2);
    }
    __syncthreads();
    const auto* codes=reinterpret_cast<const char2*>(packed);
    const auto* scales=reinterpret_cast<const __half*>(packed+512);
    auto* dst=reinterpret_cast<__half2*>(out+base);
    #pragma unroll
    for(unsigned i=threadIdx.x;i<1024;i+=128){const char2 q=codes[i];
        dst[i]=__hmul2(__floats2half2_rn(float(q.x),float(q.y)),__half2half2(scales[i/16]));}
}

augur_cuda_result_v1 row_q8_input_original_submit(int32_t ordinal,augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* a,bool already_packed=false){
    Footprint f{};
    if(!row_q8_input_original_valid(ordinal,stream,plan,a,false,&f))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    const auto selected=cudaSetDevice(ordinal);
    if(selected!=cudaSuccess)return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,selected);
    const auto* w=reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a->weight));
    const auto* x=reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input_f32));
    auto* y=reinterpret_cast<float*>(static_cast<uintptr_t>(a->output_f32));
    auto* wh=reinterpret_cast<__half*>(static_cast<uintptr_t>(plan->snapshot.decoded_weight_f16));
    auto* xh=reinterpret_cast<__half*>(static_cast<uintptr_t>(plan->snapshot.decoded_input_f16));
    cudaError_t e;
    if(a->tokens<=8U)e=llama_small::launch_row_q8_input(w,x,xh,y,a->tokens,a->rows,stream->stream);
    else if(a->tokens<64U)e=augur_qwen35_launch_row_q8_input_mmq_sm70(w,x,xh,y,
        reinterpret_cast<float*>(wh),a->tokens,a->rows,a->cols,a->weight_format,1.0f,
        plan->multiprocessors,stream->stream);
    else {
        row_q8_input_decode_shared128<<<uint32_t(f.weight_elements/2048),128,0,stream->stream>>>(w,wh);
        if(!already_packed)pack_f32_activation_kernel<<<launch_blocks(f.input_elements),kThreads,0,stream->stream>>>(x,xh,f.input_elements);
        e=cudaGetLastError();if(e!=cudaSuccess)return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,e);
        const float alpha=1.0f,beta=0.0f;
        const auto b=cublasGemmEx(plan->handle,CUBLAS_OP_T,CUBLAS_OP_N,a->rows,a->tokens,a->cols,&alpha,
            wh,CUDA_R_16F,a->cols,xh,CUDA_R_16F,a->cols,&beta,y,CUDA_R_32F,a->rows,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if(b!=CUBLAS_STATUS_SUCCESS)return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,b);
        e=cudaGetLastError();
    }
    return e==cudaSuccess?result(AUGUR_CUDA_STATUS_OK,0):cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,e);
}
} // namespace

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_row_q8_input_binding_preflight_v1(
    int32_t ordinal,augur_cuda_stream_t stream,augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* a){
    // Bind placeholder Q8 input before selecting the final original-F32 policy.
    // A wide Q8 placeholder is not executable; packed submit below is <=8 only.
    if(!a||!row_q8_input_shape(a->weight_format,a->rows,a->cols)||a->weight%4U)
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    auto logical=*a;logical.weight_format=AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1;
    Footprint f{};
    return result(projection_metadata_valid(ordinal,stream,plan,&logical,true,&f)
        ?AUGUR_CUDA_STATUS_OK:AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_row_q8_input_packed_preflight_v1(
    int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_qwen35_prefill_cublas_projection_args_v1* a){
    return result(row_q8_input_packed_valid(ordinal,stream,a,true)?AUGUR_CUDA_STATUS_OK:AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_row_q8_input_packed_submit_v1(
    int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_qwen35_prefill_cublas_projection_args_v1* a){
    if(!row_q8_input_packed_valid(ordinal,stream,a,false))return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    const auto selected=cudaSetDevice(ordinal);
    if(selected!=cudaSuccess)return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,selected);
    const auto* w=reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a->weight));
    const auto* x=reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a->input_q8_1));
    auto* y=reinterpret_cast<float*>(static_cast<uintptr_t>(a->output_f32));
    if(a->tokens==8U)quantized_small_batch_row_reuse_cols5120_kernel<1,2,true,true,true>
        <<<(a->rows+1U)/2U,128,0,stream->stream>>>(w,x,y,a->rows,a->cols,a->tokens,1.0f);
    else if(a->tokens==1U)row_q8_input_scalar_kernel
        <<<a->rows,128,0,stream->stream>>>(w,x,y,a->cols);
    else quantized_small_batch_row_reuse_cols5120_kernel<1,2,false,true>
        <<<(a->rows+1U)/2U,128,0,stream->stream>>>(w,x,y,a->rows,a->cols,a->tokens,1.0f);
    const auto e=cudaGetLastError();
    return e==cudaSuccess?result(AUGUR_CUDA_STATUS_OK,0):cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,e);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_row_q8_input_original_preflight_v1(
    int32_t ordinal,augur_cuda_stream_t stream,augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* a){
    Footprint f{};return result(row_q8_input_original_valid(ordinal,stream,plan,a,true,&f)
        ?AUGUR_CUDA_STATUS_OK:AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_row_q8_input_original_submit_v1(
    int32_t ordinal,augur_cuda_stream_t stream,augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* a){
    return row_q8_input_original_submit(ordinal,stream,plan,a);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_row_q8_input_group_submit_v1(
    int32_t ordinal,augur_cuda_stream_t stream,augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* a,uint32_t count){
    if(!a||count<2U||count>3U)return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    for(uint32_t i=0;i<count;++i){Footprint f{};
        if(!row_q8_input_original_valid(ordinal,stream,plan,a+i,false,&f)||a[i].tokens<64U
            ||a[i].tokens!=a[0].tokens||a[i].input_f32!=a[0].input_f32||a[i].input_len_bytes!=a[0].input_len_bytes)
            return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    }
    for(uint32_t i=0;i<count;++i){const Range output{a[i].output_f32,a[i].output_len_bytes,4};
        for(uint32_t j=0;j<count;++j)if(i!=j&&(ranges_overlap(output,{a[j].output_f32,a[j].output_len_bytes,4})
            ||ranges_overlap(output,{a[j].weight,a[j].weight_len_bytes,4})))return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    }
    for(uint32_t i=0;i<count;++i){const auto r=row_q8_input_original_submit(ordinal,stream,plan,a+i,i!=0U);
        if(r.status!=AUGUR_CUDA_STATUS_OK)return r;}
    return result(AUGUR_CUDA_STATUS_OK,0);
}
