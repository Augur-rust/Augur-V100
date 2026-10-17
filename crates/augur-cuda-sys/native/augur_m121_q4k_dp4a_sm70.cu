#include "augur_cuda_internal.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <sm_61_intrinsics.h>
#include <cstdint>

__global__ void augur_m121_q4k_dp4a_kernel(
    const unsigned char* weight, const int8_t* q8_input, const float* q8_scales, float* output
) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= 2048) return;
    float acc = 0.0f;
    for (uint32_t block = 0; block < 16; ++block) {
        const unsigned char* packed = weight + (static_cast<size_t>(row) * 16 + block) * 160;
        const float d = __half2float(__ushort_as_half(static_cast<uint16_t>(packed[0]) | (static_cast<uint16_t>(packed[1]) << 8)));
        const float dmin = __half2float(__ushort_as_half(static_cast<uint16_t>(packed[2]) | (static_cast<uint16_t>(packed[3]) << 8)));
        for (uint32_t group = 0; group < 8; ++group) {
            int dot = 0;
            int sum = 0;
            const uint32_t lane_base = block * 256 + group * 32;
            for (uint32_t quad = 0; quad < 8; ++quad) {
                int qpack = 0;
                const uint32_t lane = group * 32 + quad * 4;
                for (uint32_t i = 0; i < 4; ++i) {
                    const uint32_t absolute = lane + i;
                    const uint32_t within = absolute % 64;
                    const uint32_t qindex = (absolute / 64) * 32 + (within % 32);
                    const unsigned char byte = packed[32 + qindex];
                    const int q = within < 32 ? byte & 0x0f : byte >> 4;
                    qpack |= q << (8 * i);
                }
                const int xpack = *reinterpret_cast<const int*>(q8_input + lane_base + quad * 4);
                dot = __dp4a(qpack, xpack, dot);
                sum = __dp4a(0x01010101, xpack, sum);
            }
            const float dx = q8_scales[lane_base / 32];
            acc += dx * (d * static_cast<float>(packed[4 + group]) * static_cast<float>(dot)
                - dmin * static_cast<float>(packed[12 + group]) * static_cast<float>(sum));
        }
    }
    output[row] = acc;
}

cudaError_t augur_m121_q4k_dp4a_2048x4096_launch(
    const unsigned char* weight, const int8_t* q8_input, const float* q8_scales,
    float* output, cudaStream_t stream
) {
    augur_m121_q4k_dp4a_kernel<<<16, 128, 0, stream>>>(weight, q8_input, q8_scales, output);
    return cudaGetLastError();
}
