#include "augur_cuda_runtime.h"

#include <cuda_runtime.h>

static_assert(
    sizeof(augur_cuda_add_f32_args_v1) == 64,
    "augur_cuda_add_f32_args_v1 ABI size changed"
);

__global__ void augur_add_f32_kernel_v1(
    const float* a,
    const float* b,
    float* output,
    uint32_t elements
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = a[index] + b[index];
    }
}

// The GDN residual and final-add planes are allocated as 16-byte aligned F32
// arena slices. Keep the scalar kernel as the ABI-safe fallback for arbitrary
// callers, but use one vector transaction per thread when all three views have
// the stronger alignment and a whole number of float4 values. Arithmetic stays
// componentwise F32 addition; this changes only the memory transaction shape.
__global__ void augur_add_f32_vec4_kernel_v1(
    const float4* a,
    const float4* b,
    float4* output,
    uint32_t vectors
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < vectors) {
        const float4 av = a[index];
        const float4 bv = b[index];
        output[index] = make_float4(
            av.x + bv.x,
            av.y + bv.y,
            av.z + bv.z,
            av.w + bv.w);
    }
}
