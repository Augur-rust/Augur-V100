// SPDX-License-Identifier: Apache-2.0
// Included once in the cuBLAS plan implementation; no new persistent scratch.
#include "augur_qwen35_nvfp4_qpn_abi.h"
// Shared legacy-arithmetic enqueue used by the wide QUASAR SiLU/down path.
// This declaration lets direct-B8 fuse F32 SiLU with the existing FP16 staging
// buffer without changing the QPN projection mathematics.
extern "C" cudaError_t augur_silu_pack_enqueue(
    cudaStream_t stream,const augur_cuda_silu_mul_f32_args_v1* a,uint64_t packed);
static_assert(sizeof(augur_cuda_qwen35_qpn_prefill_args_v1)==80,"QPN prefill ABI");
namespace {
bool qpn_prefill_valid(int32_t ordinal,augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_qpn_prefill_args_v1* a,bool inspect_device) {
    if(!plan||!plan->handle||!stream||!stream->stream||plan->owner!=stream
       ||plan->ordinal!=ordinal||stream->ordinal!=ordinal||!a
       ||a->abi_version!=1||a->struct_size!=sizeof(*a)||(a->numerical_profile<1||a->numerical_profile>9)
       ||!a->rows||a->rows%32||!a->cols||a->cols%64
       ||a->rows>INT32_MAX||a->cols>INT32_MAX||!a->tokens||a->tokens>plan->snapshot.max_tokens
       ||(a->numerical_profile==8?a->split_k!=augur_qwen35_qpn_shape_split_k_v1(a->rows,a->cols)
          :a->numerical_profile==9?(a->split_k!=8):(a->split_k!=8&&a->split_k!=16))||!std::isfinite(a->tensor_scale)
       ||a->tensor_scale<0x1p-32f||a->tensor_scale>24.0f) return false;
    const uint64_t e=uint64_t(a->rows)*a->cols,x=uint64_t(a->tokens)*a->cols;
    if(e>UINT32_MAX||x>UINT32_MAX||a->weight_bytes!=e/64*36||a->input_bytes!=x*4
       ||a->output_bytes!=uint64_t(a->rows)*a->tokens*4
       ||(a->numerical_profile>=3&&a->tokens>8)
       ||plan->snapshot.decoded_input_len_bytes<x*(a->numerical_profile>=3?4:2)
       ||(a->tokens>8&&plan->snapshot.decoded_weight_len_bytes<e*2)
       ||(a->numerical_profile>=6&&plan->snapshot.decoded_weight_len_bytes<uint64_t(a->rows)*a->tokens*a->split_k*(a->numerical_profile>=7?4U:16U))) return false;
    const Range ranges[]={{a->weight,a->weight_bytes,16},{a->input_f32,a->input_bytes,8},
        {a->output_f32,a->output_bytes,4},
        {plan->snapshot.decoded_input_f16,plan->snapshot.decoded_input_len_bytes,16},
        {plan->snapshot.decoded_weight_f16,plan->snapshot.decoded_weight_len_bytes,16}};
    if(!all_disjoint(ranges,5)) return false;
    if(inspect_device) {
        // The opaque plan can only be created by plan_preflight_valid, which
        // admits SM70 once. Its ordinal and exact stream owner were checked
        // above and are immutable for its lifetime. Do not re-query device
        // properties for every layer/projection binding. Allocation ownership,
        // extents and aliases are still checked here for every new binding.
        // Preserve device selection before driver range queries; another live
        // owner may have changed the caller thread's current CUDA device.
        if(cudaSetDevice(ordinal)!=cudaSuccess) return false;
        for(const auto& r:ranges) if(!device_range(r,ordinal)) return false;
    }
    return true;
}
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_prefill_preflight_v1(
    int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
    const augur_cuda_qwen35_qpn_prefill_args_v1* a) {
    return result(qpn_prefill_valid(d,s,p,a,true)?AUGUR_CUDA_STATUS_OK:AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
}
static augur_cuda_result_v1 qpn_prefill_submit(
    int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
    const augur_cuda_qwen35_qpn_prefill_args_v1* a,bool stage_input) {
    if(!qpn_prefill_valid(d,s,p,a,false)) return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    const uint64_t x=uint64_t(a->tokens)*a->cols;
    const uint64_t staged_bytes=x*(a->numerical_profile>=3?4:2);
    const augur_cuda_qwen35_qpn_stage_args_v1 stage={1,sizeof(stage),a->input_f32,a->input_bytes,
        p->snapshot.decoded_input_f16,staged_bytes,x};
    auto r=result(AUGUR_CUDA_STATUS_OK,0);
    if(stage_input) r=a->numerical_profile>=3?
        augur_cuda_qwen35_qpn_stage_residual_prevalidated_submit_v1(d,s,&stage):
        augur_cuda_qwen35_qpn_stage_prevalidated_submit_v1(d,s,&stage);
    if(r.status!=AUGUR_CUDA_STATUS_OK) return r;
    if(a->tokens<=8) {
        const augur_cuda_qwen35_qpn_project_args_v1 projection={1,sizeof(projection),a->weight,a->weight_bytes,
            p->snapshot.decoded_input_f16,staged_bytes,a->output_f32,a->output_bytes,
            a->rows,a->cols,a->tokens,a->split_k,a->tensor_scale,a->numerical_profile};
        if(a->numerical_profile>=6)
            return augur_cuda_qwen35_qpn_workspace_prevalidated_submit_v1(d,s,&projection,
                p->snapshot.decoded_weight_f16,p->snapshot.decoded_weight_len_bytes);
        return augur_cuda_qwen35_qpn_project_prevalidated_submit_v1(d,s,&projection);
    }
    const augur_cuda_qwen35_qpn_expand_args_v1 expand={1,sizeof(expand),a->weight,a->weight_bytes,
        p->snapshot.decoded_weight_f16,uint64_t(a->rows)*a->cols*2,a->rows,a->cols,a->tensor_scale,a->numerical_profile==2?0U:1U};
    r=augur_cuda_qwen35_qpn_expand_prevalidated_submit_v1(d,s,&expand);
    if(r.status!=AUGUR_CUDA_STATUS_OK) return r;
    const float alpha=a->numerical_profile==2?a->tensor_scale:1.0f,beta=0.0f;
    const auto status=cublasGemmEx(p->handle,CUBLAS_OP_T,CUBLAS_OP_N,a->rows,a->tokens,a->cols,
        &alpha,reinterpret_cast<void*>(p->snapshot.decoded_weight_f16),CUDA_R_16F,a->cols,
        reinterpret_cast<void*>(p->snapshot.decoded_input_f16),CUDA_R_16F,a->cols,
        &beta,reinterpret_cast<void*>(a->output_f32),CUDA_R_32F,a->rows,
        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if(status!=CUBLAS_STATUS_SUCCESS) return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,status);
    const auto e=cudaGetLastError();
    return e==cudaSuccess?result(AUGUR_CUDA_STATUS_OK,0):cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,e);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_prefill_prevalidated_submit_v1(
    int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
    const augur_cuda_qwen35_qpn_prefill_args_v1* a) {
    return qpn_prefill_submit(d,s,p,a,true);
}
// Pair-local reuse only: never cache readiness by address across submissions.
// Both kernels/GEMMs consume the same freshly staged input. The second weight
// may overwrite weight scratch, never input scratch, on this exact stream.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_prefill_pair_prevalidated_submit_v1(
    int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
    const augur_cuda_qwen35_qpn_prefill_args_v1* a,
    const augur_cuda_qwen35_qpn_prefill_args_v1* b) {
    if(!qpn_prefill_valid(d,s,p,a,false)||!qpn_prefill_valid(d,s,p,b,false)
       ||a->input_f32!=b->input_f32||a->input_bytes!=b->input_bytes
       ||a->cols!=b->cols||a->tokens!=b->tokens
       ||((a->numerical_profile>=3)!=(b->numerical_profile>=3)))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    // Include cross-projection aliases, not only each projection in isolation.
    const Range ranges[]={{a->weight,a->weight_bytes,16},{b->weight,b->weight_bytes,16},
        {a->input_f32,a->input_bytes,8},{a->output_f32,a->output_bytes,4},
        {b->output_f32,b->output_bytes,4},
        {p->snapshot.decoded_input_f16,p->snapshot.decoded_input_len_bytes,16},
        {p->snapshot.decoded_weight_f16,p->snapshot.decoded_weight_len_bytes,16}};
    if(!all_disjoint(ranges,7)) return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    auto r=qpn_prefill_submit(d,s,p,a,true);
    return r.status==AUGUR_CUDA_STATUS_OK?qpn_prefill_submit(d,s,p,b,false):r;
}
// Direct-B8 closed producer/consumer. The existing direct QPN profile stages
// an F32 activation to the plan's FP16 scratch in a separate kernel. Reuse the
// established Augur packer so the observable F32 activation is preserved while
// the same pass also fills that scratch; no weight or projection math changes.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_silu_down_direct_b8_prevalidated_submit_v1(
    int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
    const augur_cuda_silu_mul_f32_args_v1* a,const augur_cuda_qwen35_qpn_prefill_args_v1* down) {
    if(!a||!qpn_prefill_valid(d,s,p,down,false)||a->abi_version!=1||a->struct_size!=sizeof(*a)
       ||a->reserved!=0||down->numerical_profile!=2||down->tokens!=8
       ||down->rows!=5120||down->cols!=17408
       ||a->elements!=uint32_t(down->tokens)*down->cols
       ||a->output!=down->input_f32||a->output_len_bytes!=down->input_bytes
       ||a->gate_len_bytes<down->input_bytes||a->up_len_bytes<down->input_bytes)
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    const Range ranges[]={{a->gate,a->gate_len_bytes,4},{a->up,a->up_len_bytes,4},
        {a->output,a->output_len_bytes,4},{down->weight,down->weight_bytes,16},
        {down->output_f32,down->output_bytes,4},
        {p->snapshot.decoded_input_f16,p->snapshot.decoded_input_len_bytes,16},
        {p->snapshot.decoded_weight_f16,p->snapshot.decoded_weight_len_bytes,16}};
    for(unsigned i=0;i<7;++i) {
        if(!range_valid(ranges[i]))return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
        for(unsigned j=i+1;j<7;++j) {
            // Production arena permits the exact gate/output alias used by the
            // in-place F32 SiLU plane; every other overlap remains forbidden.
            if(i==0&&j==2&&a->gate==a->output&&a->gate_len_bytes==a->output_len_bytes)continue;
            if(ranges_overlap(ranges[i],ranges[j]))return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
        }
    }
    const auto e=augur_silu_pack_enqueue(s->stream,a,p->snapshot.decoded_input_f16);
    if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,int(e));
    return qpn_prefill_submit(d,s,p,down,false);
}

namespace {
bool qpn_norm_group_valid(int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
    const augur_cuda_rmsnorm_f32_args_v1* n,const augur_cuda_qwen35_qpn_prefill_args_v1* a,
    uint32_t count,bool inspect_device) {
    if (!n || !a || count<2 || count>3 || n->abi_version!=1 || n->struct_size!=sizeof(*n)
        || n->elements!=5120 || n->input_len_bytes!=20480 || n->weight_len_bytes!=20480
        || n->output_len_bytes!=20480) return false;
    float epsilon=0.0f; std::memcpy(&epsilon,&n->epsilon_bits,sizeof(epsilon));
    if (!std::isfinite(epsilon) || epsilon<=0.0f) return false;
    for (uint32_t i=0;i<count;++i)
        if (!qpn_prefill_valid(d,s,p,a+i,inspect_device) || a[i].tokens!=1 || a[i].cols!=5120
            || a[i].numerical_profile!=8 || a[i].input_f32!=n->output) return false;
    Range ranges[11]={{n->input,n->input_len_bytes,4},{n->weight,n->weight_len_bytes,4},
        {n->output,n->output_len_bytes,4},
        {p->snapshot.decoded_input_f16,p->snapshot.decoded_input_len_bytes,16},
        {p->snapshot.decoded_weight_f16,p->snapshot.decoded_weight_len_bytes,16}};
    for (uint32_t i=0;i<count;++i) {
        ranges[5+2*i]={a[i].weight,a[i].weight_bytes,16};
        ranges[6+2*i]={a[i].output_f32,a[i].output_bytes,4};
    }
    if (!all_disjoint(ranges,5+2*count)) return false;
    if (inspect_device)
        for (uint32_t i=0;i<5+2*count;++i) if (!device_range(ranges[i],d)) return false;
    return true;
}
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_norm_group_preflight_v1(
    int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
    const augur_cuda_rmsnorm_f32_args_v1* n,const augur_cuda_qwen35_qpn_prefill_args_v1* a,uint32_t count) {
    return result(qpn_norm_group_valid(d,s,p,n,a,count,true)?AUGUR_CUDA_STATUS_OK:AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_norm_group_prevalidated_submit_v1(
    int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
    const augur_cuda_rmsnorm_f32_args_v1* n,const augur_cuda_qwen35_qpn_prefill_args_v1* a,uint32_t count) {
    if (!qpn_norm_group_valid(d,s,p,n,a,count,false)) return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    auto r=augur_cuda_rmsnorm_f32_warp_tree5120_qpn_submit_v1(d,s,n,p->snapshot.decoded_input_f16,20480);
    for (uint32_t i=0;i<count && r.status==AUGUR_CUDA_STATUS_OK;++i)
        r=qpn_prefill_submit(d,s,p,a+i,false);
    return r;
}

// B1/profile8 FFN only. Materialize the same F32 activation for diagnostics,
// and its identical hi/residual F16 planes in the existing plan scratch.
namespace {
__global__ void qpn_silu_residual_stage(const float* gate,const float* up,
                                      float* activation,half* staged,unsigned elements) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=elements)return;
    const float g=gate[i],u=up[i];float sigmoid;
    if(g>=0.f) sigmoid=1.f/(1.f+expf(-g));
    else {const float exponent=expf(g);sigmoid=exponent/(1.f+exponent);}
    const float x=(g*sigmoid)*u;activation[i]=x;
    const half hi=__float2half_rn(x);staged[i]=hi;
    float residual=__fmul_rn(__fsub_rn(x,__half2float(hi)),4096.f);
    if(isfinite(residual))residual=fminf(65504.f,fmaxf(-65504.f,residual));
    staged[elements+i]=__float2half_rn(residual);
}
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_silu_down_prevalidated_submit_v1(
    int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
    const augur_cuda_silu_mul_f32_args_v1* a,const augur_cuda_qwen35_qpn_prefill_args_v1* down) {
    if(!a||!qpn_prefill_valid(d,s,p,down,false)||a->abi_version!=1||a->struct_size!=sizeof(*a)
       ||down->numerical_profile!=8||down->tokens!=1||down->rows!=5120||down->cols!=17408
       ||a->elements!=17408||a->output!=down->input_f32||a->output_len_bytes!=down->input_bytes
       ||a->gate_len_bytes!=17408*4||a->up_len_bytes!=17408*4)
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    const Range ranges[]={{a->gate,a->gate_len_bytes,4},{a->up,a->up_len_bytes,4},
        {a->output,a->output_len_bytes,4},{down->weight,down->weight_bytes,16},
        {down->output_f32,down->output_bytes,4},
        {p->snapshot.decoded_input_f16,p->snapshot.decoded_input_len_bytes,16},
        {p->snapshot.decoded_weight_f16,p->snapshot.decoded_weight_len_bytes,16}};
    if(!all_disjoint(ranges,7))return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    qpn_silu_residual_stage<<<(a->elements+255)/256,256,0,s->stream>>>(
        reinterpret_cast<const float*>(a->gate),reinterpret_cast<const float*>(a->up),
        reinterpret_cast<float*>(a->output),reinterpret_cast<half*>(p->snapshot.decoded_input_f16),a->elements);
    const auto e=cudaGetLastError();if(e!=cudaSuccess)return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,int(e));
    return qpn_prefill_submit(d,s,p,down,false);
}
