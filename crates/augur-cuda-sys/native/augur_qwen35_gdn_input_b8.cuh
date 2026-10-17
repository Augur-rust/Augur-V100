// SPDX-License-Identifier: Apache-2.0
// Fused norm -> FP16 stage -> grouped QPN projections for direct-B8 verification
// (profile2, SplitK8). Included once in the cuBLAS plan TU (--fmad=false/--ftz=false).
//
// GDN input:      rmsnorm, 2 stagings, 4 projection launches, 2 strided 64->48 copies
//                 and 4 replay-record copies become 2 launches.
// Post-attention: residual add, rmsnorm, staging and the gate/up pair become 2 launches.
// Arithmetic is unchanged:
//  * norm_stage keeps prefill_rmsnorm_kernel's per-thread serial sum and shared tree,
//    stores the same F32 output and its FP16 RNE (= stage_activation_f16 of that output);
//    the optional residual is one FP32 add per element, as augur_add_f32_vec4_kernel_v1;
//  * the grouped projection runs project<8,half,true>'s tile code in the QPN TU and may
//    trim a 64-row tile to its first 48 columns, plus an identical second store.
#include "augur_qwen35_qpn_group.h"
static_assert(sizeof(augur_cuda_qwen35_qpn_norm_group_b8_args_v1)==512,"norm group B8 ABI");
namespace {
constexpr uint32_t kNormGroupCols=5120,kNormGroupPer=kNormGroupCols/kThreads;
template<bool Residual>
__global__ __launch_bounds__(256) void qpn_norm_stage_b8(const float* add_a,const float* add_b,
    float* input,const float* weight,float* output,half* staged,float epsilon) {
    const uint32_t token=blockIdx.x,lane=threadIdx.x;
    const uint64_t base=uint64_t(token)*kNormGroupCols;
    __shared__ float partial[kThreads];
    float values[kNormGroupPer];
    float sum=0.0f;
    #pragma unroll
    for(uint32_t j=0;j<kNormGroupPer;++j) {
        const uint32_t col=lane+j*kThreads;
        float value;
        if constexpr(Residual) {value=add_a[base+col]+add_b[base+col];input[base+col]=value;}
        else value=input[base+col];
        values[j]=value;
        sum+=value*value;
    }
    partial[lane]=sum;
    __syncthreads();
    for(uint32_t stride=kThreads/2U;stride!=0;stride>>=1U) {
        if(lane<stride) partial[lane]+=partial[lane+stride];
        __syncthreads();
    }
    const float scale=rsqrtf(partial[0]/static_cast<float>(kNormGroupCols)+epsilon);
    #pragma unroll
    for(uint32_t j=0;j<kNormGroupPer;++j) {
        const uint32_t col=lane+j*kThreads;
        const float y=values[j]*scale*weight[col];
        output[base+col]=y;
        staged[base+col]=__float2half_rn(y);
    }
}
bool norm_group_member_shape(uint32_t rows,uint32_t out) {
    return (out==rows&&(rows==10240||rows==6144||rows==17408))||(rows==64&&out==48);
}
bool qpn_norm_group_b8_valid(int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
    const augur_cuda_qwen35_qpn_norm_group_b8_args_v1* a,bool inspect) {
    if(!a||a->abi_version!=1||a->struct_size!=sizeof(*a)||a->residual>1||a->count<2||a->count>4) return false;
    if(!rmsnorm_metadata_valid(d,s,&a->norm,inspect)||a->norm.tokens!=8||a->norm.cols!=kNormGroupCols) return false;
    if(p==nullptr||p->owner!=s||p->ordinal!=d
       ||p->snapshot.decoded_input_len_bytes<uint64_t(8)*kNormGroupCols*2) return false;
    Range r[20]={{a->norm.input_f32,a->norm.input_len_bytes,16},{a->norm.weight_f32,a->norm.weight_len_bytes,4},
        {a->norm.output_f32,a->norm.output_len_bytes,16},
        {p->snapshot.decoded_input_f16,p->snapshot.decoded_input_len_bytes,16}};
    size_t n=4;
    for(uint32_t i=0;i<4;++i) {
        const auto& m=a->projections[i];const uint32_t out=a->out_cols[i];
        if(i>=a->count) {
            if(m.weight||a->records[i]||out) return false;
            continue;
        }
        // qpn_prefill_valid checks the untrimmed producer extent; the true (trimmed)
        // output and record ranges are checked below together with every other range.
        auto v=m;v.output_bytes=uint64_t(m.rows)*m.tokens*4;
        if(!qpn_prefill_valid(d,s,p,&v,false)||m.tokens!=8||m.cols!=kNormGroupCols
           ||m.numerical_profile!=2||m.split_k!=8||m.input_f32!=a->norm.output_f32
           ||!norm_group_member_shape(m.rows,out)||m.output_bytes!=uint64_t(out)*8*4) return false;
        if((a->records[i]!=0)!=(a->record_bytes[i]!=0)) return false;
        r[n++]={m.weight,m.weight_bytes,16};
        r[n++]={m.output_f32,m.output_bytes,4};
        if(a->records[i]) {
            if(a->record_bytes[i]!=m.output_bytes) return false;
            r[n++]={a->records[i],a->record_bytes[i],4};
        }
    }
    if(a->residual) {
        if(a->add_len_bytes!=a->norm.input_len_bytes) return false;
        r[n++]={a->add_a,a->add_len_bytes,16};r[n++]={a->add_b,a->add_len_bytes,16};
    } else if(a->add_a||a->add_b||a->add_len_bytes) return false;
    if(!all_disjoint(r,n)) return false;
    if(inspect) for(size_t i=0;i<n;++i) if(!device_range(r[i],d)) return false;
    return true;
}
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_norm_group_b8_preflight_v1(
    int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
    const augur_cuda_qwen35_qpn_norm_group_b8_args_v1* a) {
    return result(qpn_norm_group_b8_valid(d,s,p,a,true)?AUGUR_CUDA_STATUS_OK:AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_norm_group_b8_prevalidated_submit_v1(
    int32_t d,augur_cuda_stream_t s,augur_cuda_qwen35_prefill_cublas_plan_v1* p,
    const augur_cuda_qwen35_qpn_norm_group_b8_args_v1* a) {
    if(!qpn_norm_group_b8_valid(d,s,p,a,false)) return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    const cudaError_t selected=cudaSetDevice(d);
    if(selected!=cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,selected);
    float epsilon=0.0f;std::memcpy(&epsilon,&a->norm.epsilon_bits,sizeof(epsilon));
    const auto f=[](uint64_t v){return reinterpret_cast<float*>(static_cast<uintptr_t>(v));};
    auto* staged=reinterpret_cast<half*>(static_cast<uintptr_t>(p->snapshot.decoded_input_f16));
    if(a->residual) qpn_norm_stage_b8<true><<<8,kThreads,0,s->stream>>>(f(a->add_a),f(a->add_b),
        f(a->norm.input_f32),f(a->norm.weight_f32),f(a->norm.output_f32),staged,epsilon);
    else qpn_norm_stage_b8<false><<<8,kThreads,0,s->stream>>>(nullptr,nullptr,
        f(a->norm.input_f32),f(a->norm.weight_f32),f(a->norm.output_f32),staged,epsilon);
    cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,e);
    augur_qpn_group g{};
    int32_t tile=0;
    for(uint32_t i=0;i<a->count;++i) {
        const auto& m=a->projections[i];
        g.m[i]={m.weight,m.output_f32,a->records[i],int32_t(m.rows),int32_t(a->out_cols[i]),tile,m.tensor_scale};
        tile+=int32_t(m.rows/32U);
    }
    g.count=int32_t(a->count);g.cols=int32_t(kNormGroupCols);g.tokens=8;g.tiles=tile;
    e=augur_qpn_project_group_b8_enqueue(s->stream,&g,p->snapshot.decoded_input_f16);
    return e==cudaSuccess?result(AUGUR_CUDA_STATUS_OK,0):cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,e);
}
