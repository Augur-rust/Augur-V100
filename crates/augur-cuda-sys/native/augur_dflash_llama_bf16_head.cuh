// Shared BF16 vocabulary head for the DFlash draft and B8 verifier: the
// llama-derived MMVF<8> kernel and the Augur register-blocked FP32 kernel
// below. Both read the original weights; no duplicate vocabulary allocation.
namespace {
bool dflash_head_valid(int32_t ordinal,augur_cuda_stream_t stream,
    const augur_cuda_bf16_f32_matvec_args_v1* a,bool inspect) {
    if(!stream || !stream->stream || stream->ordinal!=ordinal || !a
        || a->abi_version!=1 || a->struct_size!=sizeof(*a)
        || a->rows!=248320U || a->cols!=5120U
        || a->weight_len_bytes!=uint64_t(248320)*5120*2
        || a->input_len_bytes!=uint64_t(8)*5120*4
        || a->output_len_bytes!=uint64_t(8)*248320*4) return false;
    const Range r[]={{a->weight,a->weight_len_bytes,2},{a->input,a->input_len_bytes,4},{a->output,a->output_len_bytes,4}};
    for(unsigned i=0;i<3;++i) {
        if(!r[i].address || r[i].address%r[i].alignment) return false;
        for(unsigned j=0;j<i;++j) if(ranges_overlap(r[i],r[j])) return false;
        if(inspect && !device_range(r[i],ordinal)) return false;
    }
    if(inspect) {cudaDeviceProp p{};if(cudaGetDeviceProperties(&p,ordinal)!=cudaSuccess || p.major!=7 || p.minor!=0)return false;}
    return true;
}
}
// Augur B8 shared head (original kernel, not llama-derived): register-blocked
// BF16 x FP32 on CUDA cores. Each warp owns 4 vocabulary rows; each lane loads
// 8 BF16 weights per row as one 16-byte vector and reuses every weight for all
// 8 activation rows. BF16->FP32 is exact, products are exact FP32 FMAs, and
// accumulation is FP32; only the summation order differs from MMVF<8>.
namespace augur_head {
constexpr unsigned K=5120,RW=4,TPB=256,ROWS_PER_BLOCK=(TPB/32)*RW;
__global__ void __launch_bounds__(TPB) f32_rw4(const uint16_t* __restrict__ w,
    const float* __restrict__ x,float* __restrict__ y,unsigned n) {
    const unsigned warp=threadIdx.x>>5,lane=threadIdx.x&31,row0=(blockIdx.x*(TPB/32)+warp)*RW;
    float acc[RW][8]={};
    #pragma unroll 2
    for(unsigned k=lane*8;k<K;k+=256) {
        uint4 p[RW];
        #pragma unroll
        for(unsigned r=0;r<RW;++r)p[r]=__ldg(reinterpret_cast<const uint4*>(w+size_t(row0+r)*K+k));
        #pragma unroll
        for(unsigned half=0;half<2;++half) {
            float4 xv[8];
            #pragma unroll
            for(unsigned j=0;j<8;++j)xv[j]=__ldg(reinterpret_cast<const float4*>(x+size_t(j)*K+k+half*4));
            #pragma unroll
            for(unsigned r=0;r<RW;++r) {
                const uint32_t a=half?p[r].z:p[r].x,b=half?p[r].w:p[r].y;
                const float w0=__uint_as_float(a<<16),w1=__uint_as_float(a&0xffff0000u);
                const float w2=__uint_as_float(b<<16),w3=__uint_as_float(b&0xffff0000u);
                #pragma unroll
                for(unsigned j=0;j<8;++j) {
                    acc[r][j]=__fmaf_rn(w0,xv[j].x,acc[r][j]);acc[r][j]=__fmaf_rn(w1,xv[j].y,acc[r][j]);
                    acc[r][j]=__fmaf_rn(w2,xv[j].z,acc[r][j]);acc[r][j]=__fmaf_rn(w3,xv[j].w,acc[r][j]);
                }
            }
        }
    }
    #pragma unroll
    for(unsigned r=0;r<RW;++r) {
        #pragma unroll
        for(unsigned j=0;j<8;++j) {
            float v=acc[r][j];
            for(int d=16;d;d>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,d));
            acc[r][j]=v;
        }
    }
    if(lane<8) {
        #pragma unroll
        for(unsigned r=0;r<RW;++r)y[size_t(lane)*n+row0+r]=acc[r][lane];
    }
}
static_assert(248320%ROWS_PER_BLOCK==0,"vocabulary rows must tile exactly");
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_bf16_head_f32_rw4_preflight_v1(
    int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_bf16_f32_matvec_args_v1* a) {
    // The 16-byte vector loads additionally require 16-byte weight/input bases.
    return dflash_head_valid(ordinal,stream,a,true) && a->weight%16==0 && a->input%16==0
        ?result(AUGUR_CUDA_STATUS_OK,0):result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_bf16_head_f32_rw4_submit_v1(
    int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_bf16_f32_matvec_args_v1* a) {
    if(!dflash_head_valid(ordinal,stream,a,false) || a->weight%16 || a->input%16)
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    const cudaError_t selected=cudaSetDevice(ordinal);
    if(selected!=cudaSuccess)return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,selected);
    augur_head::f32_rw4<<<248320/augur_head::ROWS_PER_BLOCK,augur_head::TPB,0,stream->stream>>>(
        reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(a->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),248320);
    const cudaError_t e=cudaGetLastError();
    return e==cudaSuccess?result(AUGUR_CUDA_STATUS_OK,0):cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,e);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_llama_bf16_head_preflight_v1(
    int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_bf16_f32_matvec_args_v1* a) {
    return dflash_head_valid(ordinal,stream,a,true)?result(AUGUR_CUDA_STATUS_OK,0):result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_llama_bf16_head_submit_v1(
    int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_bf16_f32_matvec_args_v1* a) {
    if(!dflash_head_valid(ordinal,stream,a,false))return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    const cudaError_t selected=cudaSetDevice(ordinal);
    if(selected!=cudaSuccess)return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,selected);
    llama_small::mmvf<8><<<248320,256,0,stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(a->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),248320,5120);
    const cudaError_t e=cudaGetLastError();
    return e==cudaSuccess?result(AUGUR_CUDA_STATUS_OK,0):cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,e);
}
