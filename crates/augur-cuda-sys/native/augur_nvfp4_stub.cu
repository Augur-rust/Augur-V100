#include "augur_cuda_internal.h"

static augur_cuda_result_v1 unsupported(void) {
    return {
        AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE,
        static_cast<int32_t>(cudaSuccess)
    };
}

extern "C" uint32_t AUGUR_CUDA_CALL
augur_cuda_nvfp4_sm120_build_capability_v1(void) {
    return 0;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_pack_scales_host_v1(
    uint32_t,
    uint32_t,
    const uint8_t*,
    uint64_t,
    uint8_t*,
    uint64_t
) {
    return unsupported();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_w2_m128_a_scale_offsets_host_v1(uint32_t*, uint64_t) {
    return unsupported();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_w13_m128_workspace_size_v1(uint64_t* out_size_bytes) {
    if (out_size_bytes != nullptr) {
        *out_size_bytes = 0;
    }
    return unsupported();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_w2_m128_workspace_size_v1(uint64_t* out_size_bytes) {
    if (out_size_bytes != nullptr) {
        *out_size_bytes = 0;
    }
    return unsupported();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_scale_extent_host_v1(uint32_t, uint32_t, uint64_t* out_extent_bytes) {
    if (out_extent_bytes != nullptr) *out_extent_bytes = 0;
    return unsupported();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_shape_workspace_size_v1(uint32_t, uint64_t* out_size_bytes) {
    if (out_size_bytes != nullptr) *out_size_bytes = 0;
    return unsupported();
}

augur_cuda_result_v1 augur_nvfp4_cutlass_w13_m128_launch_v1(
    const augur_cuda_nvfp4_aot_args_v1*,
    cudaStream_t
) {
    return unsupported();
}

augur_cuda_result_v1 augur_nvfp4_cutlass_w2_m128_launch_v1(
    const augur_cuda_nvfp4_aot_args_v1*,
    cudaStream_t
) {
    return unsupported();
}

augur_cuda_result_v1 augur_nvfp4_cutlass_shape_launch_v1(
    uint32_t, const augur_cuda_nvfp4_aot_args_v1*, cudaStream_t
) {
    return unsupported();
}

augur_cuda_result_v1 augur_nvfp4_w4a16_shape_launch_v1(
    uint32_t,
    const augur_cuda_nvfp4_w4a16_args_v1*,
    cudaStream_t
) {
    return unsupported();
}

augur_cuda_result_v1 augur_m68_bf16_interstage_launch_v1(
    const augur_cuda_m68_bf16_interstage_args_v1*,
    cudaStream_t
) {
    return unsupported();
}

augur_cuda_result_v1 augur_nvfp4_interstage_m128_launch_v1(
    const augur_cuda_nvfp4_interstage_args_v1*,
    cudaStream_t
) {
    return unsupported();
}

augur_cuda_result_v1 augur_nvfp4_shape_interstage_launch_v1(
    const augur_cuda_nvfp4_shape_interstage_args_v1*, cudaStream_t
) {
    return unsupported();
}

augur_cuda_result_v1 augur_bf16_w13_m128_launch_v1(const augur_cuda_bf16_gemm_args_v1*, cudaStream_t) { return unsupported(); }
augur_cuda_result_v1 augur_bf16_w2_m128_launch_v1(const augur_cuda_bf16_gemm_args_v1*, cudaStream_t) { return unsupported(); }
augur_cuda_result_v1 augur_m54_shared_interstage_m128_launch_v1(const augur_cuda_m54_shared_interstage_args_v1*, cudaStream_t) { return unsupported(); }
augur_cuda_result_v1 augur_m54_sum_m128_launch_v1(const augur_cuda_m54_sum_args_v1*, cudaStream_t) { return unsupported(); }
augur_cuda_result_v1 augur_m54_router_top6_launch_v1(const augur_cuda_m54_router_args_v1*, cudaStream_t) { return unsupported(); }
augur_cuda_result_v1 augur_m58_router_group_launch_v1(const augur_cuda_m58_router_group_args_v1*, cudaStream_t) { return unsupported(); }
augur_cuda_result_v1 augur_m58_scatter_launch_v1(const augur_cuda_m58_scatter_args_v1*, cudaStream_t) { return unsupported(); }
augur_cuda_result_v1 augur_m58_finalize_launch_v1(const augur_cuda_m58_finalize_args_v1*, cudaStream_t) { return unsupported(); }
