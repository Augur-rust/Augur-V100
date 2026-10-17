// Experimental B8 selector gate: original weights expanded to F32 once.
// This exact shape does not broaden target/Q8 projection admission.
namespace {
// Diagnostic exact B8 shape: original BF16-expanded F32 weights and F32
// hidden inputs, FP64 products/reduction, a single final F32 rounding.
// Each warp owns one output. No scratch, atomics or cross-block reductions.
__global__ void dflash_gate_f64_accum(const float* w,const float* x,float* y) {
    const int lane=threadIdx.x&31;
    const int row=blockIdx.x*8+threadIdx.x/32;
    const int token=blockIdx.y;
    double sum=0.0;
    for(int col=lane;col<5120;col+=32)
        sum+=static_cast<double>(w[row*5120+col])*static_cast<double>(x[token*5120+col]);
    for(int offset=16;offset>0;offset/=2) sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if(lane==0)y[token*256+row]=static_cast<float>(sum);
}
bool dflash_gate_f32_valid(int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* p,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* a, bool inspect) {
    if (!p || !p->handle || !stream || !stream->stream || p->ordinal!=ordinal
        || p->owner!=stream || stream->ordinal!=ordinal || !a
        || a->abi_version!=1 || a->struct_size!=sizeof(*a) || a->reserved!=0
        || (a->weight_format!=AUGUR_CUDA_DFLASH_SELECTOR_GATE_WEIGHT_F32_V1
            && a->weight_format!=AUGUR_CUDA_DFLASH_SELECTOR_GATE_F64_ACCUM_V1)
        || a->tensor_scale!=1.0f || a->tokens!=8 || a->rows!=256 || a->cols!=5120
        || p->snapshot.max_tokens<8 || a->weight_len_bytes!=uint64_t(256)*5120*4
        || a->input_len_bytes!=uint64_t(8)*5120*4
        || a->output_len_bytes!=uint64_t(8)*256*4) return false;
    const Range ranges[]={{a->weight,a->weight_len_bytes,4},
        {a->input_f32,a->input_len_bytes,4},{a->output_f32,a->output_len_bytes,4},
        {p->snapshot.decoded_weight_f16,p->snapshot.decoded_weight_len_bytes,16},
        {p->snapshot.decoded_input_f16,p->snapshot.decoded_input_len_bytes,16}};
    if (!all_disjoint(ranges,5)) return false;
    if (inspect) {for (const auto r:ranges) if (!device_range(r,ordinal)) return false;}
    return true;
}
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_selector_gate_f32_preflight_v1(
    int32_t ordinal,augur_cuda_stream_t stream,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* a) {
    return dflash_gate_f32_valid(ordinal,stream,p,a,true)?result(AUGUR_CUDA_STATUS_OK,0):result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_selector_gate_f32_submit_v1(
    int32_t ordinal,augur_cuda_stream_t stream,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* a) {
    if(!dflash_gate_f32_valid(ordinal,stream,p,a,false)) return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    const auto e=cudaSetDevice(ordinal);if(e!=cudaSuccess)return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,e);
    if(a->weight_format==AUGUR_CUDA_DFLASH_SELECTOR_GATE_F64_ACCUM_V1) {
        dflash_gate_f64_accum<<<dim3(32,8),256,0,stream->stream>>>(
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->weight)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input_f32)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->output_f32)));
        const auto launch=cudaGetLastError();
        return launch==cudaSuccess?result(AUGUR_CUDA_STATUS_OK,0):cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,launch);
    }
    const float alpha=1.0f,beta=0.0f;
    // Row-major W[256,5120], X[8,5120] => column-major W^T X, Y[256,8].
    // PEDANTIC keeps every phase F32, even if the shared plan uses tensor math.
    const auto b=cublasGemmEx(p->handle,CUBLAS_OP_T,CUBLAS_OP_N,256,8,5120,&alpha,
        reinterpret_cast<const void*>(static_cast<uintptr_t>(a->weight)),CUDA_R_32F,5120,
        reinterpret_cast<const void*>(static_cast<uintptr_t>(a->input_f32)),CUDA_R_32F,5120,&beta,
        reinterpret_cast<void*>(static_cast<uintptr_t>(a->output_f32)),CUDA_R_32F,256,
        CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT);
    return b==CUBLAS_STATUS_SUCCESS?result(AUGUR_CUDA_STATUS_OK,0):blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,b);
}
