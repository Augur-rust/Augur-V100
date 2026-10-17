#ifndef CRYSTAL_QUANT_CUDA_INTERNAL_H
#define CRYSTAL_QUANT_CUDA_INTERNAL_H

#include "augur_cuda_runtime.h"

#include <cuda_runtime_api.h>

// Some CUDA toolkits omit the __dp4a declaration for SM70 translation units.
// Bind the signed four-lane dot product directly to the SM61+ PTX instruction
// so V100 Q8/KV8 kernels cannot silently fall back to scalar multiplies.
#if defined(__CUDACC__)
__device__ __forceinline__ int32_t augur_cuda_dp4a_s8(
    int32_t left,
    int32_t right,
    int32_t accumulator
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
    int32_t result;
    asm volatile(
        "dp4a.s32.s32 %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(left), "r"(right), "r"(accumulator));
    return result;
#else
    const int8_t* left_bytes = reinterpret_cast<const int8_t*>(&left);
    const int8_t* right_bytes = reinterpret_cast<const int8_t*>(&right);
    return accumulator
        + static_cast<int32_t>(left_bytes[0]) * static_cast<int32_t>(right_bytes[0])
        + static_cast<int32_t>(left_bytes[1]) * static_cast<int32_t>(right_bytes[1])
        + static_cast<int32_t>(left_bytes[2]) * static_cast<int32_t>(right_bytes[2])
        + static_cast<int32_t>(left_bytes[3]) * static_cast<int32_t>(right_bytes[3]);
#endif
}
#ifndef __dp4a
#define __dp4a(left, right, accumulator) augur_cuda_dp4a_s8((left), (right), (accumulator))
#endif
#endif

struct augur_cuda_stream_handle {
    int32_t ordinal;
    cudaStream_t stream;
};

struct augur_cuda_event_handle {
    int32_t ordinal;
    cudaEvent_t event;
};

augur_cuda_result_v1 augur_nvfp4_cutlass_w13_m128_launch_v1(
    const augur_cuda_nvfp4_aot_args_v1* args,
    cudaStream_t stream
);

augur_cuda_result_v1 augur_nvfp4_cutlass_shape_launch_v1(
    uint32_t shape,
    const augur_cuda_nvfp4_aot_args_v1* args,
    cudaStream_t stream
);

augur_cuda_result_v1 augur_nvfp4_w4a16_shape_launch_v1(
    uint32_t shape,
    const augur_cuda_nvfp4_w4a16_args_v1* args,
    cudaStream_t stream
);

augur_cuda_result_v1 augur_m68_bf16_interstage_launch_v1(
    const augur_cuda_m68_bf16_interstage_args_v1* args,
    cudaStream_t stream
);

augur_cuda_result_v1 augur_nvfp4_shape_interstage_launch_v1(
    const augur_cuda_nvfp4_shape_interstage_args_v1* args,
    cudaStream_t stream
);

augur_cuda_result_v1 augur_nvfp4_cutlass_w2_m128_launch_v1(
    const augur_cuda_nvfp4_aot_args_v1* args,
    cudaStream_t stream
);

augur_cuda_result_v1 augur_nvfp4_interstage_m128_launch_v1(
    const augur_cuda_nvfp4_interstage_args_v1* args,
    cudaStream_t stream
);

augur_cuda_result_v1 augur_bf16_w13_m128_launch_v1(
    const augur_cuda_bf16_gemm_args_v1* args,
    cudaStream_t stream
);
augur_cuda_result_v1 augur_bf16_w2_m128_launch_v1(
    const augur_cuda_bf16_gemm_args_v1* args,
    cudaStream_t stream
);
augur_cuda_result_v1 augur_m54_shared_interstage_m128_launch_v1(
    const augur_cuda_m54_shared_interstage_args_v1* args,
    cudaStream_t stream
);
augur_cuda_result_v1 augur_m54_sum_m128_launch_v1(
    const augur_cuda_m54_sum_args_v1* args,
    cudaStream_t stream
);
augur_cuda_result_v1 augur_m54_router_top6_launch_v1(
    const augur_cuda_m54_router_args_v1* args,
    cudaStream_t stream
);
augur_cuda_result_v1 augur_m58_router_group_launch_v1(
    const augur_cuda_m58_router_group_args_v1* args,
    cudaStream_t stream
);
augur_cuda_result_v1 augur_m58_scatter_launch_v1(
    const augur_cuda_m58_scatter_args_v1* args,
    cudaStream_t stream
);
augur_cuda_result_v1 augur_m58_finalize_launch_v1(
    const augur_cuda_m58_finalize_args_v1* args,
    cudaStream_t stream
);

cudaError_t augur_m120_q4k_2048x4096_launch(
    const unsigned char* weight, const float* input, float* output, cudaStream_t stream);
cudaError_t augur_m120_q5k_4096x2048_launch(
    const unsigned char* weight, const float* input, float* output, cudaStream_t stream);
cudaError_t augur_m120_q6k_4096x4096_launch(
    const unsigned char* weight, const float* input, float* output, cudaStream_t stream);
cudaError_t augur_m121_q4k_dp4a_2048x4096_launch(
    const unsigned char* weight, const int8_t* q8_input, const float* q8_scales,
    float* output, cudaStream_t stream);

#if defined(AUGUR_CUDA_P4_EXPERT_MICROSLICE_GPU_REPACK)
cudaError_t augur_p4_q4k_compact_to_hot_launch(
    const unsigned char* compact, unsigned char* hot,
    uint32_t rows, uint32_t blocks_per_row, cudaStream_t stream);
cudaError_t augur_p4_q5k_compact_to_hot_launch(
    const unsigned char* compact, unsigned char* hot,
    uint32_t rows, uint32_t blocks_per_row, cudaStream_t stream);
#endif

#endif
