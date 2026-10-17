// MIT, Copyright (c) 2023-2026 The ggml authors; see llama_attention_vendor/LICENSE.
// Calibration only. Candidate register pressure must not change reduction grouping.
#include "llama_attention_vendor/fattn-vec-reference.cuh"
#include "llama_attention_vendor/fattn-tile-reference.cuh"
#include "llama_attention_vendor/fattn-mma-reference.cuh"

extern "C" int augur_qwen35_rotated_small_calibrate_v1(int ordinal,unsigned* sm_count,unsigned* occupancy){
    if(ordinal<0||!sm_count||!occupancy)return int(cudaErrorInvalidValue);
    auto e=cudaSetDevice(ordinal);if(e!=cudaSuccess)return int(e);
    cudaDeviceProp device{};e=cudaGetDeviceProperties(&device,ordinal);if(e!=cudaSuccess)return int(e);
    if(device.major!=7||device.minor!=0||device.multiProcessorCount<1||device.multiProcessorCount>96)return int(cudaErrorInvalidDevice);
    const fattn_kernel_t kernels[]={flash_attn_ext_vec<256,1,GGML_TYPE_Q8_0,GGML_TYPE_Q8_0,false>,
        flash_attn_tile<256,256,2,2,false>,flash_attn_tile<256,256,4,2,false>,flash_attn_tile<256,256,8,2,false>};
    for(unsigned i=0;i<4;++i){
        int active=0;e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernels[i],i<2?128:256,0);
        if(e!=cudaSuccess)return int(e);
        if(active<1||active>4)return int(cudaErrorInvalidConfiguration);
        occupancy[i]=unsigned(active);
    }
    *sm_count=unsigned(device.multiProcessorCount);return 0;
}

extern "C" int augur_qwen35_rotated_input_calibrate_v1(int ordinal, unsigned* sm, unsigned* occupancy, unsigned* shared) {
    if (ordinal<0 || !sm || !occupancy || !shared) return int(cudaErrorInvalidValue);
    cudaDeviceProp p{};
    auto e=cudaSetDevice(ordinal);
    if(e!=cudaSuccess)return int(e);
    e=cudaGetDeviceProperties(&p,ordinal);
    if(e!=cudaSuccess)return int(e);
    if(p.major!=7||p.minor!=0||p.multiProcessorCount<1||p.multiProcessorCount>96)
        return int(cudaErrorInvalidDevice);
    *sm=unsigned(p.multiProcessorCount);
    const fattn_kernel_t kernels[]={flash_attn_ext_f16<256,256,16,2,false,false>,
        flash_attn_ext_f16<256,256,32,2,false,false>};
    for(unsigned i=0;i<2;++i){
        const int cols1=i?32:16,cols=cols1*2,cc=700;
        // Device family was checked above. Avoid the generic host dispatcher's
        // ggml_abort dependency; Rust intentionally links no ggml runtime.
        const auto c=ggml_cuda_fattn_mma_get_config_volta(256,256,cols);
        const int swk=ggml_cuda_fattn_smem_swizzle::tile_stride(c.nbatch_K2,cc);
        const int swv=ggml_cuda_fattn_smem_swizzle::tile_stride(c.nbatch_V2,cc);
        const size_t kv=c.nbatch_fa*std::max(swk,swv)*sizeof(half2);
        const size_t q=cols*(128+4)*sizeof(half2);
        const size_t mask=cols1*(c.nbatch_fa/2+4)*sizeof(half2);
        const size_t combine=(c.nthreads/32)*std::min(cols,get_cols_per_warp(cc))*(c.nbatch_combine+4)*sizeof(half2);
        shared[i]=unsigned(std::max(combine,c.Q_in_reg?std::max(q,kv+mask):q+kv+mask));
        if(c.nthreads!=128||c.nbatch_fa!=32)return int(cudaErrorInvalidConfiguration);
        e=cudaFuncSetAttribute(kernels[i],cudaFuncAttributeMaxDynamicSharedMemorySize,shared[i]);
        if(e!=cudaSuccess)return int(e);
        int active=0;
        e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernels[i],c.nthreads,shared[i]);
        if(e!=cudaSuccess)return int(e);
        if(active<1||active>4)return int(cudaErrorInvalidConfiguration);
        occupancy[i]=unsigned(active);
    }
    return 0;
}
