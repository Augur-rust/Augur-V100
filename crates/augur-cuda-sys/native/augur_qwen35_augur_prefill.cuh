// SPDX-License-Identifier: Apache-2.0
// Speed-first Augur adapter. This is not source-F64 equivalence.
// QPN is a physical weight-layout adapter only; the contraction is shared
// with the shared FP16/FP32 implementation rather than a new compensated GEMM.
#include "augur_qwen35_augur_prefill_policy.cuh"
#include "augur_qwen35_augur_prefill_pack.cuh"
namespace {
bool augur_valid(int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
 const augur_cuda_qwen35_qpn_prefill_args_v1* a,bool inspect){
 if(!a||a->numerical_profile!=2||a->split_k!=8||!a->tokens||a->tokens>AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1)return false;
 if(!qpn_prefill_valid(d,s,p,a,inspect))return false;
 // Unlike legacy B1..8, tails also use this full-K Tensor Core path.
 return p->snapshot.decoded_weight_len_bytes>=uint64_t(a->rows)*a->cols*2;
}
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_augur_prefill_preflight_v1(
 int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,const augur_cuda_qwen35_qpn_prefill_args_v1* a){
 return result(augur_valid(d,s,p,a,true)?AUGUR_CUDA_STATUS_OK:AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
}
namespace {
augur_cuda_result_v1 augur_submit(
 int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
 const augur_cuda_qwen35_qpn_prefill_args_v1* a,bool pack_input){
 if(!augur_valid(d,s,p,a,false))return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
 auto error=cudaSetDevice(d);if(error!=cudaSuccess)return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,error);
 const uint64_t e=uint64_t(a->rows)*a->cols,n=uint64_t(a->tokens)*a->cols;
 const bool fast_expand=a->tokens>=512U &&
     ((a->cols==5120U && (a->rows==64U||a->rows==1024U||a->rows==5120U||
       a->rows==6144U||a->rows==10240U||a->rows==12288U||a->rows==17408U)) ||
      (a->rows==5120U && (a->cols==6144U||a->cols==17408U)));
 if(fast_expand) {
  const unsigned threads=(a->cols==5120U&&(a->rows==1024U||a->rows==5120U))?512U:128U;
  augur_prefill_expand_half2<<<unsigned((e/16U+threads-1U)/threads),threads,0,s->stream>>>(
      (const uint8_t*)a->weight,(half*)p->snapshot.decoded_weight_f16,a->rows,a->cols);
  error=cudaGetLastError();if(error!=cudaSuccess)return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
 } else {
  const augur_cuda_qwen35_qpn_expand_args_v1 expand={1,sizeof(expand),a->weight,a->weight_bytes,
      p->snapshot.decoded_weight_f16,e*2,a->rows,a->cols,1.0f,0};
  auto r=augur_cuda_qwen35_qpn_expand_prevalidated_submit_v1(d,s,&expand);if(r.status!=AUGUR_CUDA_STATUS_OK)return r;
 }
 auto* xh=(__half*)p->snapshot.decoded_input_f16;
 if(pack_input) {
  if(a->tokens>=512U) {
   if((a->input_f32&15U)==0U)
    augur_prefill_pack_vector<4><<<launch_blocks((n+3U)/4U),kThreads,0,s->stream>>>((const float*)a->input_f32,xh,n);
   else
    augur_prefill_pack_vector<2><<<launch_blocks((n+1U)/2U),kThreads,0,s->stream>>>((const float*)a->input_f32,xh,n);
  } else {
  pack_f32_activation_kernel<<<launch_blocks(n),kThreads,0,s->stream>>>((const float*)a->input_f32,xh,n);
  }
  error=cudaGetLastError();if(error!=cudaSuccess)return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
 }
 return llama_f16_gemm_postscale(s,p,a->rows,a->cols,a->tokens,
     (const __half*)p->snapshot.decoded_weight_f16,xh,(float*)a->output_f32,a->tensor_scale,true,
     augur_prefill_fuse_scale(a->rows,a->cols,a->tokens));
}
} // namespace
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_augur_prefill_submit_v1(
 int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
 const augur_cuda_qwen35_qpn_prefill_args_v1* a){
 return augur_submit(d,s,p,a,true);
}
// A closed same-input group, never a pointer-keyed cache. Every invocation
// repacks its first projection; no packed-input validity survives this call.
// The existing disjointness checks exclude every output/weight from the common
// input and both scratch planes, so no intervening contraction changes input.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_augur_prefill_group_submit_v1(
 int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
 const augur_cuda_qwen35_qpn_prefill_args_v1* args,uint32_t count){
 if(!args||count<2||count>4)return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
 for(uint32_t i=0;i<count;++i) {
  const auto& a=args[i];
  if(!augur_valid(d,s,p,&a,false)||a.tokens<64U||
      a.input_f32!=args[0].input_f32||a.input_bytes!=args[0].input_bytes||
      a.cols!=args[0].cols||a.tokens!=args[0].tokens)
   return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
 }
 for(uint32_t i=0;i<count;++i) {
  auto r=augur_submit(d,s,p,args+i,i==0);
  if(r.status!=AUGUR_CUDA_STATUS_OK)return r;
 }
 return result(AUGUR_CUDA_STATUS_OK,0);
}

// The producer is compiled beside the original SiLU kernel, not in this TU's
// --fmad=false profile. Preserve its exact expf/arithmetic identity.
extern "C" cudaError_t augur_silu_pack_enqueue(
    cudaStream_t stream,const augur_cuda_silu_mul_f32_args_v1* a,uint64_t packed);
// Closed producer+consumer; never publish a reusable already-packed bit.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_augur_silu_down_submit_v1(
    int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
    const augur_cuda_silu_mul_f32_args_v1* a,const augur_cuda_qwen35_qpn_prefill_args_v1* down) {
    if(!a||!augur_valid(d,s,p,down,false)||a->abi_version!=1||a->struct_size!=sizeof(*a)
       ||a->reserved!=0||down->tokens<512U||down->tokens>6400U||down->rows!=5120U||down->cols!=17408U
       ||a->elements!=uint64_t(down->tokens)*down->cols||a->output!=down->input_f32
       ||a->output_len_bytes!=down->input_bytes||a->gate_len_bytes!=down->input_bytes
       ||a->up_len_bytes!=down->input_bytes) return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    const Range ranges[]={{a->gate,a->gate_len_bytes,4},{a->up,a->up_len_bytes,4},
        {a->output,a->output_len_bytes,4},{down->weight,down->weight_bytes,16},
        {down->output_f32,down->output_bytes,4},
        {p->snapshot.decoded_input_f16,p->snapshot.decoded_input_len_bytes,16},
        {p->snapshot.decoded_weight_f16,p->snapshot.decoded_weight_len_bytes,16}};
    for(unsigned i=0;i<7;++i) {
        if(!range_valid(ranges[i]))return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
        for(unsigned j=i+1;j<7;++j) {
            // Exact gate overwrite only; partial aliases, up overwrites and
            // scratch/weight/output overlaps are never admitted.
            if(i==0&&j==2&&a->gate==a->output)continue;
            if(ranges_overlap(ranges[i],ranges[j]))return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
        }
    }
    auto error=cudaSetDevice(d);if(error!=cudaSuccess)return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,error);
    error=augur_silu_pack_enqueue(s->stream,a,p->snapshot.decoded_input_f16);
    if(error!=cudaSuccess)return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
    // Expansion cannot invalidate the freshly packed input: ranges disjoint.
    return augur_submit(d,s,p,down,false);
}
