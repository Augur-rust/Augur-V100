#include "augur_cuda_runtime.h"

#include <cuda_runtime.h>

static_assert(
    sizeof(augur_cuda_gqa_repeat_f32_args_v1) == 56,
    "augur_cuda_gqa_repeat_f32_args_v1 ABI size changed"
);

__global__ void augur_gqa_repeat_f32_kernel_v1(
    const float* input,
    float* output,
    uint32_t source_elements,
    uint32_t head_dim,
    uint32_t repeats
) {
    const uint64_t output_elements =
        static_cast<uint64_t>(source_elements) * repeats;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index =
             static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < output_elements;
         index += stride) {
        const uint64_t grouped_head_width =
            static_cast<uint64_t>(head_dim) * repeats;
        const uint64_t source_head = index / grouped_head_width;
        const uint64_t within_head = index % head_dim;
        const uint64_t source_index = source_head * head_dim + within_head;
        output[index] = input[source_index];
    }
}
