// Scoped no-ID SM70 wrapper around pinned ggml device templates only.
// MIT; Copyright (c) 2023-2026 The ggml authors. See llama_attention_vendor/LICENSE.
#include "llama_attention_vendor/mmq.cuh"
#include "llama_attention_vendor/augur_quantize_mmq_device.cuh"
namespace {
uint3 mmq_divisor(unsigned n){unsigned shift=0;while((1u<<shift)<n)++shift;
    unsigned multiplier=unsigned((uint64_t(1)<<32)*((uint64_t(1)<<shift)-n)/n+1);return make_uint3(multiplier,shift,n);}
template<ggml_type F,int J>
cudaError_t submit_mmq(const void* w,const void* packed,float* out,float* fixup,unsigned m,unsigned n,unsigned k,unsigned sm,cudaStream_t stream){
    const auto c=ggml_cuda_mmq_get_config_ampere(F,J,false);
    const unsigned shared=J*sizeof(int)+ggml_cuda_mmq_get_nbytes_shared_x(c,700)
        +((J*sizeof(block_q8_1_mmq)+c.nthreads*sizeof(int)-1)/(c.nthreads*sizeof(int)))*(c.nthreads*sizeof(int));
    cudaError_t e=cudaFuncSetAttribute(mul_mat_q<F,J,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);if(e!=cudaSuccess)return e;
    const unsigned ntiles=(n+c.I-1)/c.I,eff=100*ntiles/(sm*((ntiles+sm-1)/sm));
    const unsigned grid=eff>=90?ntiles:sm;const bool needs_fixup=ntiles%grid!=0;
    const auto one=mmq_divisor(1),blocks=mmq_divisor(k/ggml_cuda_type_traits<F>::qk);
    const unsigned stride=k/ggml_cuda_type_traits<F>::qk,ys=m*k/32*36/4;
    mul_mat_q<F,J,false><<<grid,dim3(32,c.nthreads/32),shared,stream>>>(
        (const char*)w,(const int*)packed,nullptr,nullptr,out,needs_fixup?fixup:nullptr,nullptr,
        blocks,n,m,stride,m,n,one,one,n*stride,ys,m*n,one,one,n*stride,ys,m*n,one);
    e=cudaGetLastError();if(e!=cudaSuccess)return e;
    if(needs_fixup)mul_mat_q_stream_k_fixup<F,J,false><<<dim3(grid,c.I/32),dim3(32,c.nthreads/64),0,stream>>>(
        nullptr,nullptr,out,fixup,blocks,n,m,n,one,m*n,one,m*n,one);
    return cudaGetLastError();
}
__global__ void mmq_scale(float* output,size_t count,float scale){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x)output[i]*=scale;}
}
extern "C" cudaError_t augur_qwen35_launch_llama_mmq_sm70(const void* w,const float* x,void* packed,float* output,float* fixup,
    unsigned m,unsigned n,unsigned k,unsigned format,float scale,unsigned sm,cudaStream_t stream){
#if defined(AUGUR_QWEN35_ROW_Q8_INPUT)
    if(m<9||m>63||k!=5120||!sm||format!=5||(n!=1024&&n!=6144&&n!=10240&&n!=12288))return cudaErrorInvalidValue;
#else
    if(m<9||m>63||n%128||k%512||!sm||(format!=1&&format!=2))return cudaErrorInvalidValue;
#endif
    quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4,false><<<dim3(m,(k+511)/512),128,0,stream>>>(x,nullptr,packed,k,k,m*k,m*k,k,m,1,0);
    cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return e;
    unsigned j=m<=16?16:m<=24?24:m<=32?32:m<=40?40:m<=48?48:64;
    #if defined(AUGUR_QWEN35_ROW_Q8_INPUT)
    #define MMQ_CASE(J) case J: e=submit_mmq<GGML_TYPE_Q8_0,J>(w,packed,output,fixup,m,n,k,sm,stream);break
    #else
    #define MMQ_CASE(J) case J: e=format==1?submit_mmq<GGML_TYPE_Q8_0,J>(w,packed,output,fixup,m,n,k,sm,stream):submit_mmq<GGML_TYPE_NVFP4,J>(w,packed,output,fixup,m,n,k,sm,stream);break
    #endif
    switch(j){MMQ_CASE(16);MMQ_CASE(24);MMQ_CASE(32);MMQ_CASE(40);MMQ_CASE(48);MMQ_CASE(64);}
    #undef MMQ_CASE
    if(e!=cudaSuccess)return e;
    if(format==2)mmq_scale<<<(size_t(m)*n+255)/256,256,0,stream>>>(output,size_t(m)*n,scale);
    return cudaGetLastError();
}
