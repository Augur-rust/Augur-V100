// SPDX-License-Identifier: Apache-2.0
#include "augur_cuda_internal.h"
#include "augur_qwen35_nvfp4_qpn_abi.h"
#include "augur_qwen35_nvfp4_qpn_gemm.cuh"
#include <cuda.h>
#include <cmath>
#include <climits>

static_assert(sizeof(augur_cuda_qwen35_qpn_pack_args_v1)==48, "pack ABI");
static_assert(sizeof(augur_cuda_qwen35_qpn_stage_args_v1)==48, "stage ABI");
static_assert(sizeof(augur_cuda_qwen35_qpn_expand_args_v1)==56, "expand ABI");
static_assert(sizeof(augur_cuda_qwen35_qpn_project_args_v1)==80, "project ABI");
namespace {
struct Range { uint64_t address, bytes, alignment; };
bool valid(Range r) {
    return r.address && r.bytes && r.address%r.alignment==0
        && r.bytes<=UINT64_MAX-r.address;
}
bool disjoint(Range a,Range b) {
    return valid(a)&&valid(b)&&(a.address+a.bytes<=b.address||b.address+b.bytes<=a.address);
}
bool shape(uint32_t n,uint32_t k,uint64_t& elements) {
    elements=uint64_t(n)*k;
    return n && k && n%32==0 && k%64==0 && n<=INT_MAX && k<=INT_MAX
        && elements<=UINT32_MAX;
}
// Conservative FP16-safe scalar range for every finite UE4M3/E2M1 word.
bool scale(float s) { return std::isfinite(s)&&s>=0x1p-32f&&s<=24.0f; }
template<class T> bool header(int32_t d,augur_cuda_stream_t s,const T* a) {
    return d>=0&&s&&s->ordinal==d&&s->stream&&a&&a->abi_version==1&&a->struct_size==sizeof(T);
}
bool device_range(Range r,int32_t d) {
    if(!valid(r)||r.bytes>SIZE_MAX) return false;
    cudaPointerAttributes attr{};
    if(cudaPointerGetAttributes(&attr,reinterpret_cast<void*>(r.address))!=cudaSuccess) {
        cudaGetLastError(); return false;
    }
    if(attr.type!=cudaMemoryTypeDevice||attr.device!=d) return false;
    CUdeviceptr base=0; size_t bytes=0;
    if(cuMemGetAddressRange(&base,&bytes,r.address)!=CUDA_SUCCESS||r.address<base) return false;
    const uint64_t offset=r.address-base;
    return offset<=bytes&&r.bytes<=bytes-offset;
}
bool inspect(int32_t d,const Range* r,size_t count) {
    if(cudaSetDevice(d)!=cudaSuccess) return false;
    cudaDeviceProp p{};
    if(cudaGetDeviceProperties(&p,d)!=cudaSuccess||p.major!=7||p.minor!=0) return false;
    for(size_t i=0;i<count;++i) if(!device_range(r[i],d)) return false;
    return true;
}
augur_cuda_result_v1 result(bool ok) { return {ok?AUGUR_CUDA_STATUS_OK:AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0}; }
augur_cuda_result_v1 submitted() {
    const auto e=cudaGetLastError();
    return {e==cudaSuccess?AUGUR_CUDA_STATUS_OK:AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,int32_t(e)};
}
bool check(int32_t d,augur_cuda_stream_t s,const augur_cuda_qwen35_qpn_pack_args_v1* a,bool full) {
    uint64_t e=0;
    if(!header(d,s,a)||!shape(a->rows,a->cols,e)||a->source_bytes!=e/64*36||a->destination_bytes!=a->source_bytes) return false;
    const Range r[]={{a->source,a->source_bytes,4},{a->destination,a->destination_bytes,16}};
    return disjoint(r[0],r[1])&&(!full||inspect(d,r,2));
}
bool check(int32_t d,augur_cuda_stream_t s,const augur_cuda_qwen35_qpn_stage_args_v1* a,bool full) {
    if(!header(d,s,a)||!a->elements||a->elements>UINT32_MAX||a->source_bytes!=a->elements*4||a->destination_bytes!=a->elements*2) return false;
    const Range r[]={{a->source,a->source_bytes,8},{a->destination,a->destination_bytes,16}};
    return disjoint(r[0],r[1])&&(!full||inspect(d,r,2));
}
bool check(int32_t d,augur_cuda_stream_t s,const augur_cuda_qwen35_qpn_expand_args_v1* a,bool full) {
    uint64_t e=0;
    if(!header(d,s,a)||!shape(a->rows,a->cols,e)||a->source_bytes!=e/64*36||a->destination_bytes!=e*2
       ||a->numerical_profile>1||!scale(a->tensor_scale)) return false;
    const Range r[]={{a->source,a->source_bytes,16},{a->destination,a->destination_bytes,4}};
    return disjoint(r[0],r[1])&&(!full||inspect(d,r,2));
}
bool check(int32_t d,augur_cuda_stream_t s,const augur_cuda_qwen35_qpn_project_args_v1* a,bool full) {
    uint64_t e=0;
    if(!header(d,s,a)||!shape(a->rows,a->cols,e)||a->weight_bytes!=e/64*36||a->tokens<1||a->tokens>8
       ||(a->split_k!=8&&a->split_k!=16)||(a->numerical_profile<1||a->numerical_profile>5)||!scale(a->tensor_scale)
       ||a->input_bytes!=uint64_t(a->tokens)*a->cols*(a->numerical_profile>=3?4:2)||a->output_bytes!=uint64_t(a->tokens)*a->rows*4) return false;
    const Range r[]={{a->weight,a->weight_bytes,16},{a->input,a->input_bytes,16},{a->output,a->output_bytes,4}};
    return disjoint(r[0],r[1])&&disjoint(r[0],r[2])&&disjoint(r[1],r[2])&&(!full||inspect(d,r,3));
}
void launch(augur_cuda_stream_t s,const augur_cuda_qwen35_qpn_pack_args_v1* a) {
    const uint64_t groups=uint64_t(a->rows)*a->cols/16;
    augur_qwen35_qpn::prepack_ggml<<<uint32_t((groups+255)/256),256,0,s->stream>>>(
        reinterpret_cast<const uint8_t*>(a->source),reinterpret_cast<uint8_t*>(a->destination),a->rows,a->cols);
}
void launch(augur_cuda_stream_t s,const augur_cuda_qwen35_qpn_stage_args_v1* a) {
    augur_qwen35_qpn::stage_activation_f16<<<uint32_t((a->elements+511)/512),256,0,s->stream>>>(
        reinterpret_cast<const float*>(a->source),reinterpret_cast<half*>(a->destination),a->elements);
}
void launch(augur_cuda_stream_t s,const augur_cuda_qwen35_qpn_expand_args_v1* a) {
    const auto* w=reinterpret_cast<const uint8_t*>(a->source); auto* y=reinterpret_cast<half*>(a->destination);
    if((a->destination&15U)==0) {
        const uint32_t blocks=uint32_t((uint64_t(a->rows)*a->cols/16+255)/256);
        if(a->numerical_profile==1) augur_qwen35_qpn::decode_f16_grouped<true><<<blocks,256,0,s->stream>>>(w,y,a->rows,a->cols,a->tensor_scale);
        else augur_qwen35_qpn::decode_f16_grouped<false><<<blocks,256,0,s->stream>>>(w,y,a->rows,a->cols,a->tensor_scale);
    } else {
        // Preserve the original ABI's aligned4 destination contract.
        const uint32_t blocks=uint32_t((uint64_t(a->rows)*a->cols+511)/512);
        if(a->numerical_profile==1) augur_qwen35_qpn::decode_f16<true><<<blocks,256,0,s->stream>>>(w,y,a->rows,a->cols,a->tensor_scale);
        else augur_qwen35_qpn::decode_f16<false><<<blocks,256,0,s->stream>>>(w,y,a->rows,a->cols,a->tensor_scale);
    }
}
void launch(augur_cuda_stream_t s,const augur_cuda_qwen35_qpn_project_args_v1* a) {
    const auto* w=reinterpret_cast<const uint8_t*>(a->weight); const auto* x=reinterpret_cast<const half*>(a->input);
    auto* y=reinterpret_cast<float*>(a->output);
    if(a->numerical_profile==5) {
        if(a->split_k==16) augur_qwen35_qpn::project<16,half,true,true,true,16,false><<<a->rows/32,512,0,s->stream>>>(w,x,y,a->rows,a->cols,a->tokens,a->tensor_scale);
        else augur_qwen35_qpn::project<8,half,true,true,true,16,false><<<a->rows/32,256,0,s->stream>>>(w,x,y,a->rows,a->cols,a->tokens,a->tensor_scale);
    } else if(a->numerical_profile==4) {
        if(a->split_k==16) augur_qwen35_qpn::project<16,half,true,true,true><<<a->rows/32,512,0,s->stream>>>(w,x,y,a->rows,a->cols,a->tokens,a->tensor_scale);
        else augur_qwen35_qpn::project<8,half,true,true,true><<<a->rows/32,256,0,s->stream>>>(w,x,y,a->rows,a->cols,a->tokens,a->tensor_scale);
    } else if(a->numerical_profile==3) {
        if(a->split_k==16) augur_qwen35_qpn::project<16,half,true,true><<<a->rows/32,512,0,s->stream>>>(w,x,y,a->rows,a->cols,a->tokens,a->tensor_scale);
        else augur_qwen35_qpn::project<8,half,true,true><<<a->rows/32,256,0,s->stream>>>(w,x,y,a->rows,a->cols,a->tokens,a->tensor_scale);
    } else if(a->numerical_profile==2) {
        if(a->split_k==16) augur_qwen35_qpn::project<16,half,true><<<a->rows/32,512,0,s->stream>>>(w,x,y,a->rows,a->cols,a->tokens,a->tensor_scale);
        else augur_qwen35_qpn::project<8,half,true><<<a->rows/32,256,0,s->stream>>>(w,x,y,a->rows,a->cols,a->tokens,a->tensor_scale);
    } else {
        if(a->split_k==16) augur_qwen35_qpn::project<16,half><<<a->rows/32,512,0,s->stream>>>(w,x,y,a->rows,a->cols,a->tokens,a->tensor_scale);
        else augur_qwen35_qpn::project<8,half><<<a->rows/32,256,0,s->stream>>>(w,x,y,a->rows,a->cols,a->tokens,a->tensor_scale);
    }
}
} // namespace
#define AUGUR_QPN_ENTRY(NAME, TYPE) \
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_##NAME##_preflight_v1(int32_t d,augur_cuda_stream_t s,const TYPE* a) { \
    return result(check(d,s,a,true)); } \
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_##NAME##_prevalidated_submit_v1(int32_t d,augur_cuda_stream_t s,const TYPE* a) { \
    if(!check(d,s,a,false)) return result(false); \
    const auto e=cudaSetDevice(d); if(e!=cudaSuccess) return {AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int32_t(e)}; \
    launch(s,a); return submitted(); }
AUGUR_QPN_ENTRY(pack,augur_cuda_qwen35_qpn_pack_args_v1)
AUGUR_QPN_ENTRY(stage,augur_cuda_qwen35_qpn_stage_args_v1)
AUGUR_QPN_ENTRY(expand,augur_cuda_qwen35_qpn_expand_args_v1)
AUGUR_QPN_ENTRY(project,augur_cuda_qwen35_qpn_project_args_v1)
#undef AUGUR_QPN_ENTRY

// Private grouped enqueue (see augur_qwen35_qpn_group.h). The caller in the plan TU
// validated every member, range and alias before reaching here; this TU only launches,
// so the projection tiles compile with exactly the flags of project<8,half,true>.
extern "C" cudaError_t augur_qpn_project_group_b8_enqueue(
    cudaStream_t stream,const augur_qpn_group* group,uint64_t staged_f16) {
    augur_qwen35_qpn::project_group<<<uint32_t(group->tiles),256,0,stream>>>(
        *group,reinterpret_cast<const half*>(staged_f16));
    return cudaGetLastError();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_pack_checkpoint_preflight_v1(
    int32_t d,augur_cuda_stream_t s,const augur_cuda_qwen35_qpn_pack_args_v1* a) {
    return result(check(d,s,a,true));
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_pack_checkpoint_prevalidated_submit_v1(
    int32_t d,augur_cuda_stream_t s,const augur_cuda_qwen35_qpn_pack_args_v1* a) {
    if(!check(d,s,a,false)) return result(false);
    const auto e=cudaSetDevice(d);
    if(e!=cudaSuccess) return {AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int32_t(e)};
    const uint64_t groups=uint64_t(a->rows)*a->cols/16;
    augur_qwen35_qpn::prepack_checkpoint<<<uint32_t((groups+255)/256),256,0,s->stream>>>(
        reinterpret_cast<const uint8_t*>(a->source),reinterpret_cast<uint8_t*>(a->destination),a->rows,a->cols);
    return submitted();
}

namespace {
bool check_residual_stage(int32_t d,augur_cuda_stream_t s,const augur_cuda_qwen35_qpn_stage_args_v1* a,bool full) {
    if(!header(d,s,a)||!a->elements||a->elements>UINT32_MAX
        ||a->source_bytes!=a->elements*4||a->destination_bytes!=a->elements*4) return false;
    const Range r[]={{a->source,a->source_bytes,8},{a->destination,a->destination_bytes,16}};
    return disjoint(r[0],r[1])&&(!full||inspect(d,r,2));
}
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_stage_residual_preflight_v1(
    int32_t d,augur_cuda_stream_t s,const augur_cuda_qwen35_qpn_stage_args_v1* a) {
    return result(check_residual_stage(d,s,a,true));
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_stage_residual_prevalidated_submit_v1(
    int32_t d,augur_cuda_stream_t s,const augur_cuda_qwen35_qpn_stage_args_v1* a) {
    if(!check_residual_stage(d,s,a,false)) return result(false);
    const auto e=cudaSetDevice(d); if(e!=cudaSuccess) return {AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int32_t(e)};
    augur_qwen35_qpn::stage_activation_residual_f16<<<uint32_t((a->elements+511)/512),256,0,s->stream>>>(
        reinterpret_cast<const float*>(a->source),reinterpret_cast<half*>(a->destination),a->elements);
    return submitted();
}

namespace {
bool check_workspace(int32_t d,augur_cuda_stream_t s,const augur_cuda_qwen35_qpn_project_args_v1* a,
                     uint64_t workspace,uint64_t workspace_bytes,bool full) {
    if(!a||(a->numerical_profile!=6&&a->numerical_profile!=7&&a->numerical_profile!=8)) return false;
    if(a->numerical_profile==8&&a->split_k!=augur_qwen35_qpn_shape_split_k_v1(a->rows,a->cols)) return false;
    auto base=*a; base.numerical_profile=4;
    // Reuse the common operand-range validation without inheriting its old
    // split admission; the new shape policy above independently admits32.
    if(a->numerical_profile==8) base.split_k=8;
    if(!check(d,s,&base,full)||workspace_bytes<uint64_t(a->rows)*a->tokens*a->split_k*(a->numerical_profile>=7?4U:16U)) return false;
    const Range r[]={{a->weight,a->weight_bytes,16},{a->input,a->input_bytes,16},
        {a->output,a->output_bytes,4},{workspace,workspace_bytes,16}};
    for(int i=0;i<3;++i) if(!disjoint(r[i],r[3])) return false;
    return !full||inspect(d,r,4);
}
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_workspace_preflight_v1(
    int32_t d,augur_cuda_stream_t s,const augur_cuda_qwen35_qpn_project_args_v1* a,uint64_t workspace,uint64_t bytes) {
    return result(check_workspace(d,s,a,workspace,bytes,true));
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_workspace_prevalidated_submit_v1(
    int32_t d,augur_cuda_stream_t s,const augur_cuda_qwen35_qpn_project_args_v1* a,uint64_t workspace,uint64_t bytes) {
    if(!check_workspace(d,s,a,workspace,bytes,false)) return result(false);
    auto error=cudaSetDevice(d); if(error!=cudaSuccess) return {AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int32_t(error)};
    const auto* weight=reinterpret_cast<const uint8_t*>(a->weight);
    const auto* input=reinterpret_cast<const half*>(a->input);
    auto* partials=reinterpret_cast<double*>(workspace); auto* output=reinterpret_cast<float*>(a->output);
    const dim3 grid(a->rows/32*(a->tokens<=4?1:2),a->split_k/4);
    const uint64_t elements=uint64_t(a->rows)*a->tokens;
    // Keep every non-winning shape/profile/width on the established path.
    // Workspace admission remains unchanged: the owner is also used by B2..B8.
    if(a->numerical_profile==8 && a->tokens==1 && a->split_k==16 && a->cols==5120) {
        if(a->rows==17408 || a->rows==12288) {
            augur_qwen35_qpn::project_b1_single_cta<1><<<a->rows/32,512,0,s->stream>>>(weight,input,output,a->rows,a->cols,a->tensor_scale);
            return submitted();
        }
        if(a->rows==6144) {
            augur_qwen35_qpn::project_b1_single_cta<2><<<a->rows/32,256,0,s->stream>>>(weight,input,output,a->rows,a->cols,a->tensor_scale);
            return submitted();
        }
    }
    if(a->numerical_profile>=7) {
        if(a->split_k==32) {
            if(a->tokens==1) augur_qwen35_qpn::project_workspace_cta_fold_parts<32,true,true><<<grid,128,0,s->stream>>>(weight,input,partials,a->rows,a->cols,a->tokens);
            else if(a->tokens<=4) augur_qwen35_qpn::project_workspace_cta_fold_parts<32,true><<<grid,128,0,s->stream>>>(weight,input,partials,a->rows,a->cols,a->tokens);
            else augur_qwen35_qpn::project_workspace_cta_fold_parts<32,false><<<grid,128,0,s->stream>>>(weight,input,partials,a->rows,a->cols,a->tokens);
            error=cudaGetLastError(); if(error!=cudaSuccess) return {AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int32_t(error)};
            augur_qwen35_qpn::project_workspace_merge<8><<<uint32_t((elements+255)/256),256,0,s->stream>>>(partials,output,elements,a->tensor_scale);
        } else if(a->split_k==16) {
            if(a->tokens==1) augur_qwen35_qpn::project_workspace_cta_fold_parts<16,true,true><<<grid,128,0,s->stream>>>(weight,input,partials,a->rows,a->cols,a->tokens);
            else if(a->tokens<=4) augur_qwen35_qpn::project_workspace_cta_fold_parts<16,true><<<grid,128,0,s->stream>>>(weight,input,partials,a->rows,a->cols,a->tokens);
            else augur_qwen35_qpn::project_workspace_cta_fold_parts<16,false><<<grid,128,0,s->stream>>>(weight,input,partials,a->rows,a->cols,a->tokens);
            error=cudaGetLastError(); if(error!=cudaSuccess) return {AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int32_t(error)};
            augur_qwen35_qpn::project_workspace_merge<4><<<uint32_t((elements+255)/256),256,0,s->stream>>>(partials,output,elements,a->tensor_scale);
        } else {
            if(a->tokens==1) augur_qwen35_qpn::project_workspace_cta_fold_parts<8,true,true><<<grid,128,0,s->stream>>>(weight,input,partials,a->rows,a->cols,a->tokens);
            else if(a->tokens<=4) augur_qwen35_qpn::project_workspace_cta_fold_parts<8,true><<<grid,128,0,s->stream>>>(weight,input,partials,a->rows,a->cols,a->tokens);
            else augur_qwen35_qpn::project_workspace_cta_fold_parts<8,false><<<grid,128,0,s->stream>>>(weight,input,partials,a->rows,a->cols,a->tokens);
            error=cudaGetLastError(); if(error!=cudaSuccess) return {AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int32_t(error)};
            augur_qwen35_qpn::project_workspace_merge<2><<<uint32_t((elements+255)/256),256,0,s->stream>>>(partials,output,elements,a->tensor_scale);
        }
        return submitted();
    }
    if(a->split_k==16) {
        if(a->tokens<=4) augur_qwen35_qpn::project_workspace_parts<16,true><<<grid,128,0,s->stream>>>(weight,input,partials,a->rows,a->cols,a->tokens);
        else augur_qwen35_qpn::project_workspace_parts<16,false><<<grid,128,0,s->stream>>>(weight,input,partials,a->rows,a->cols,a->tokens);
        error=cudaGetLastError(); if(error!=cudaSuccess) return {AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int32_t(error)};
        augur_qwen35_qpn::project_workspace_merge<16><<<uint32_t((elements+255)/256),256,0,s->stream>>>(partials,output,elements,a->tensor_scale);
    } else {
        if(a->tokens<=4) augur_qwen35_qpn::project_workspace_parts<8,true><<<grid,128,0,s->stream>>>(weight,input,partials,a->rows,a->cols,a->tokens);
        else augur_qwen35_qpn::project_workspace_parts<8,false><<<grid,128,0,s->stream>>>(weight,input,partials,a->rows,a->cols,a->tokens);
        error=cudaGetLastError(); if(error!=cudaSuccess) return {AUGUR_CUDA_STATUS_RUNTIME_FAILURE,int32_t(error)};
        augur_qwen35_qpn::project_workspace_merge<8><<<uint32_t((elements+255)/256),256,0,s->stream>>>(partials,output,elements,a->tensor_scale);
    }
    return submitted();
}
