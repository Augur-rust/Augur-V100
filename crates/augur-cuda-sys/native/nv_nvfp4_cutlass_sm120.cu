// SPDX-License-Identifier: BSD-3-Clause
// CUTLASS SM120 construction follows NVIDIA CUTLASS example 79a.

#include "augur_cuda_internal.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>
#include <mma.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#if __CUDACC_VER_MAJOR__ < 12 || \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)
#error "M52 SM120 NVFP4 requires CUDA Toolkit 12.8 or newer"
#endif

#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
#error "M52 SM120 NVFP4 must be compiled for the compute_120a accelerated target"
#endif

using namespace cute;

namespace augur_nvfp4_detail {

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ThreadBlockShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementC,
    LayoutCTag,
    AlignmentC,
    ElementD,
    LayoutDTag,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    LayoutATag,
    AlignmentA,
    ElementB,
    LayoutBTag,
    AlignmentB,
    ElementAccumulator,
    ThreadBlockShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void
>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;
using ScaleConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

augur_cuda_result_v1 ok() {
    return {AUGUR_CUDA_STATUS_OK, static_cast<int32_t>(cudaSuccess)};
}

augur_cuda_result_v1 invalid() {
    return {AUGUR_CUDA_STATUS_INVALID_ARGUMENT, static_cast<int32_t>(cudaSuccess)};
}

augur_cuda_result_v1 cutlass_error(cutlass::Status status) {
    return {
        AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,
        status == cutlass::Status::kSuccess
            ? static_cast<int32_t>(cudaSuccess)
            : static_cast<int32_t>(cudaErrorInvalidValue)
    };
}

template <int M, int N, int K>
typename Gemm::Arguments make_arguments(
    const augur_cuda_nvfp4_aot_args_v1* args
) {
    StrideA stride_a = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(
        StrideD{}, cute::make_shape(M, N, 1));
    auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    auto* a_values = reinterpret_cast<ElementA::DataType*>(
        static_cast<uintptr_t>(args == nullptr ? 0 : args->a_values));
    auto* a_scales = reinterpret_cast<ElementA::ScaleFactorType*>(
        static_cast<uintptr_t>(args == nullptr ? 0 : args->a_scales));
    auto* b_values = reinterpret_cast<ElementB::DataType*>(
        static_cast<uintptr_t>(args == nullptr ? 0 : args->b_values));
    auto* b_scales = reinterpret_cast<ElementB::ScaleFactorType*>(
        static_cast<uintptr_t>(args == nullptr ? 0 : args->b_scales));
    auto* output = reinterpret_cast<ElementD*>(
        static_cast<uintptr_t>(args == nullptr ? 0 : args->output));
    const float alpha = args == nullptr ? 1.0f : args->alpha;

    return typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            a_values, stride_a,
            b_values, stride_b,
            a_scales, layout_sfa,
            b_scales, layout_sfb
        },
        {
            {alpha, 0.0f},
            output, stride_c,
            output, stride_d
        }
    };
}

template <int M, int N, int K>
augur_cuda_result_v1 workspace_size(uint64_t* out_size_bytes) {
    if (out_size_bytes == nullptr) {
        return invalid();
    }
    auto arguments = make_arguments<M, N, K>(nullptr);
    *out_size_bytes = static_cast<uint64_t>(Gemm::get_workspace_size(arguments));
    return ok();
}

template <int M, int N, int K>
augur_cuda_result_v1 launch(
    const augur_cuda_nvfp4_aot_args_v1* args,
    cudaStream_t stream
) {
    if (args == nullptr || stream == nullptr) {
        return invalid();
    }
    auto arguments = make_arguments<M, N, K>(args);
    Gemm gemm;
    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cutlass_error(status);
    }
    void* workspace = reinterpret_cast<void*>(static_cast<uintptr_t>(args->workspace));
    status = gemm.initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        return cutlass_error(status);
    }
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        return cutlass_error(status);
    }
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? ok()
        : augur_cuda_result_v1{
            AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,
            static_cast<int32_t>(error)
        };
}

template <int M, int N, int K>
augur_cuda_result_v1 pack_scales(
    uint32_t operand,
    const uint8_t* source,
    uint64_t source_len_bytes,
    uint8_t* destination,
    uint64_t destination_len_bytes
) {
    if (source == nullptr || destination == nullptr) {
        return invalid();
    }
    auto pack = [&](auto layout, int rows) -> augur_cuda_result_v1 {
        const uint64_t logical_size = static_cast<uint64_t>(rows) * (K / 16);
        const uint64_t tiled_size = static_cast<uint64_t>(
            size(filter_zeros(layout)));
        if (source_len_bytes != logical_size
            || destination_len_bytes != tiled_size) {
            return invalid();
        }
        std::memset(destination, 0, static_cast<size_t>(destination_len_bytes));
        for (int row = 0; row < rows; ++row) {
            for (int group = 0; group < K / 16; ++group) {
                const auto index = layout(make_coord(row, group * 16, 0));
                destination[static_cast<size_t>(index)] =
                    source[static_cast<size_t>(row) * (K / 16) + group];
            }
        }
        return ok();
    };
    const auto problem_shape = make_shape(M, N, K, 1);
    if (operand == AUGUR_CUDA_NVFP4_OPERAND_A_V1) {
        return pack(ScaleConfig::tile_atom_to_shape_SFA(problem_shape), M);
    }
    if (operand == AUGUR_CUDA_NVFP4_OPERAND_B_V1) {
        return pack(ScaleConfig::tile_atom_to_shape_SFB(problem_shape), N);
    }
    return invalid();
}

template <int M, int N, int K>
augur_cuda_result_v1 scale_extent(uint32_t operand, uint64_t* out_size_bytes) {
    if (out_size_bytes == nullptr) {
        return invalid();
    }
    const auto problem_shape = make_shape(M, N, K, 1);
    if (operand == AUGUR_CUDA_NVFP4_OPERAND_A_V1) {
        *out_size_bytes = static_cast<uint64_t>(size(filter_zeros(
            ScaleConfig::tile_atom_to_shape_SFA(problem_shape))));
        return ok();
    }
    if (operand == AUGUR_CUDA_NVFP4_OPERAND_B_V1) {
        *out_size_bytes = static_cast<uint64_t>(size(filter_zeros(
            ScaleConfig::tile_atom_to_shape_SFB(problem_shape))));
        return ok();
    }
    return invalid();
}

template <int M, int N, int K>
augur_cuda_result_v1 scale_offsets_a(
    uint32_t* destination,
    uint64_t destination_count
) {
    if (destination == nullptr) {
        return invalid();
    }
    auto layout = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    const uint64_t logical_size = static_cast<uint64_t>(M) * (K / 16);
    if (destination_count != logical_size) {
        return invalid();
    }
    for (int row = 0; row < M; ++row) {
        for (int group = 0; group < K / 16; ++group) {
            const auto index = layout(make_coord(row, group * 16, 0));
            destination[static_cast<size_t>(row) * (K / 16) + group] =
                static_cast<uint32_t>(index);
        }
    }
    return ok();
}

__device__ float positive_e4m3_value(uint8_t code) {
    const int exponent = (code >> 3) & 0x0f;
    const int mantissa = code & 0x07;
    if (exponent == 0) {
        return ldexpf(static_cast<float>(mantissa) / 8.0f, -6);
    }
    return ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

__device__ uint8_t e2m1_nearest_code(float value) {
    constexpr float levels[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    uint8_t best = 0;
    float best_distance = fabsf(value - levels[0]);
    for (uint8_t code = 1; code < 16; ++code) {
        const float distance = fabsf(value - levels[code]);
        if (distance < best_distance) {
            best = code;
            best_distance = distance;
        }
    }
    return best;
}

__device__ float e2m1_value(uint8_t code) {
    constexpr float levels[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return levels[code & 0x0f];
}

template <int M, int N, int K>
__global__ void nvfp4_w4a16_kernel(
    const __nv_bfloat16* activation,
    const uint8_t* weight_values,
    const uint8_t* weight_scales,
    __nv_bfloat16* output,
    float weight_global_scale
) {
    using namespace nvcuda;
    const int tile_n = static_cast<int>(blockIdx.x) * 16;
    const int tile_m = static_cast<int>(blockIdx.y) * 16;
    const int lane = static_cast<int>(threadIdx.x);
    __shared__ __align__(16) __nv_bfloat16 activation_tile[16 * 16];
    __shared__ __align__(16) __nv_bfloat16 weight_tile[16 * 16];
    __shared__ __align__(16) float output_tile[16 * 16];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    for (int inner = 0; inner < K; inner += 16) {
        for (int index = lane; index < 256; index += 32) {
            const int row = index / 16;
            const int column = index % 16;
            const int source_row = tile_m + row;
            activation_tile[index] = source_row < M
                ? activation[source_row * K + inner + column]
                : __float2bfloat16_rn(0.0f);

            const int weight_row = tile_n + row;
            const int weight_column = inner + column;
            const uint8_t packed = weight_values[
                weight_row * (K / 2) + weight_column / 2];
            const uint8_t code = (weight_column & 1) == 0
                ? packed & 0x0f
                : packed >> 4;
            const uint8_t scale_code = weight_scales[
                weight_row * (K / 16) + weight_column / 16];
            weight_tile[index] = __float2bfloat16_rn(
                e2m1_value(code) * positive_e4m3_value(scale_code)
                    * weight_global_scale);
        }
        __syncwarp();
        wmma::load_matrix_sync(a, activation_tile, 16);
        wmma::load_matrix_sync(b, weight_tile, 16);
        wmma::mma_sync(accumulator, a, b, accumulator);
        __syncwarp();
    }
    wmma::store_matrix_sync(output_tile, accumulator, 16, wmma::mem_row_major);
    __syncwarp();
    for (int index = lane; index < 256; index += 32) {
        const int row = index / 16;
        const int column = index % 16;
        if (tile_m + row < M) {
            output[(tile_m + row) * N + tile_n + column] =
                __float2bfloat16_rn(output_tile[index]);
        }
    }
}

__global__ void nvfp4_interstage_m128_kernel(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    uint8_t* output_values,
    uint8_t* output_scales,
    const uint32_t* scale_offsets,
    float route_weight,
    float activation_global_scale,
    float swiglu_limit
) {
    constexpr int kColumns = 2048;
    constexpr int kGroupsPerRow = kColumns / 16;
    const int logical_group = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (logical_group >= 128 * kGroupsPerRow || lane >= 16) {
        return;
    }
    const int row = logical_group / kGroupsPerRow;
    const int group = logical_group % kGroupsPerRow;
    const int element = row * kColumns + group * 16 + lane;
    float gate_value = __bfloat162float(gate[element]);
    float up_value = __bfloat162float(up[element]);
    gate_value = fminf(gate_value, swiglu_limit);
    up_value = fminf(fmaxf(up_value, -swiglu_limit), swiglu_limit);
    const float silu = gate_value / (1.0f + expf(-gate_value));

    __shared__ float values[16];
    __shared__ uint8_t codes[16];
    __shared__ float divisor;
    values[lane] = silu * up_value * route_weight;
    __syncthreads();

    if (lane == 0) {
        float maximum = 0.0f;
        for (int index = 0; index < 16; ++index) {
            maximum = fmaxf(maximum, fabsf(values[index]));
        }
        const float desired = maximum / (6.0f * activation_global_scale);
        uint8_t scale_code = 0;
        while (scale_code < 126 && positive_e4m3_value(scale_code) < desired) {
            ++scale_code;
        }
        divisor = positive_e4m3_value(scale_code) * activation_global_scale;
        output_scales[scale_offsets[logical_group]] = scale_code;
    }
    __syncthreads();
    const float normalized = divisor > 0.0f ? values[lane] / divisor : 0.0f;
    codes[lane] = e2m1_nearest_code(normalized);
    __syncthreads();
    if (lane < 8) {
        output_values[row * (kColumns / 2) + group * 8 + lane] =
            static_cast<uint8_t>(codes[lane * 2] | (codes[lane * 2 + 1] << 4));
    }
}

__global__ void nvfp4_shape_interstage_kernel(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    const float* route_weights,
    uint8_t* output_values,
    uint8_t* output_scales,
    const uint32_t* scale_offsets,
    uint32_t capacity,
    uint32_t valid_rows,
    float activation_global_scale,
    float swiglu_limit
) {
    constexpr int kColumns = 2048;
    constexpr int kGroupsPerRow = kColumns / 16;
    const int logical_group = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (logical_group >= static_cast<int>(capacity) * kGroupsPerRow || lane >= 16) {
        return;
    }
    const int row = logical_group / kGroupsPerRow;
    const int group = logical_group % kGroupsPerRow;
    const int element = row * kColumns + group * 16 + lane;
    const bool valid = row < static_cast<int>(valid_rows);
    float gate_value = valid ? __bfloat162float(gate[element]) : 0.0f;
    float up_value = valid ? __bfloat162float(up[element]) : 0.0f;
    gate_value = fminf(gate_value, swiglu_limit);
    up_value = fminf(fmaxf(up_value, -swiglu_limit), swiglu_limit);
    const float silu = gate_value / (1.0f + expf(-gate_value));

    __shared__ float values[16];
    __shared__ uint8_t codes[16];
    __shared__ float divisor;
    values[lane] = valid ? silu * up_value * route_weights[row] : 0.0f;
    __syncthreads();

    if (lane == 0) {
        float maximum = 0.0f;
        for (int index = 0; index < 16; ++index) {
            maximum = fmaxf(maximum, fabsf(values[index]));
        }
        const float desired = maximum / (6.0f * activation_global_scale);
        uint8_t scale_code = 0;
        float scale = 0.0f;
        if (desired > 0.0f) {
            for (uint16_t candidate = 1; candidate <= 126; ++candidate) {
                const float decoded = positive_e4m3_value(static_cast<uint8_t>(candidate));
                if (decoded >= desired) {
                    scale_code = static_cast<uint8_t>(candidate);
                    scale = decoded;
                    break;
                }
            }
            if (scale_code == 0) {
                scale_code = 126;
                scale = positive_e4m3_value(scale_code);
            }
        }
        output_scales[scale_offsets[logical_group]] = scale_code;
        divisor = scale * activation_global_scale;
    }
    __syncthreads();
    const float normalized = divisor == 0.0f ? 0.0f : values[lane] / divisor;
    codes[lane] = e2m1_nearest_code(fminf(fmaxf(normalized, -6.0f), 6.0f));
    __syncthreads();
    if (lane < 8) {
        output_values[row * (kColumns / 2) + group * 8 + lane] =
            static_cast<uint8_t>(codes[2 * lane] | (codes[2 * lane + 1] << 4));
    }
}

__global__ void m68_bf16_interstage_kernel(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    const float* route_weights,
    __nv_bfloat16* output,
    uint32_t capacity,
    uint32_t valid_rows,
    float swiglu_limit
) {
    constexpr uint32_t kColumns = 2048;
    const uint32_t index = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= capacity * kColumns) {
        return;
    }
    const uint32_t row = index / kColumns;
    if (row >= valid_rows) {
        output[index] = __float2bfloat16_rn(0.0f);
        return;
    }
    float gate_value = fminf(__bfloat162float(gate[index]), swiglu_limit);
    const float up_value = fminf(
        fmaxf(__bfloat162float(up[index]), -swiglu_limit), swiglu_limit);
    const float silu = gate_value / (1.0f + expf(-gate_value));
    output[index] = __float2bfloat16_rn(
        silu * up_value * route_weights[row]);
}

template <int N, int K>
__global__ void bf16_gemm_m128_kernel(
    const __nv_bfloat16* activation,
    const __nv_bfloat16* weight,
    float* output,
    float alpha
) {
    using namespace nvcuda;
    const int tile_n = static_cast<int>(blockIdx.x) * 16;
    const int tile_m = static_cast<int>(blockIdx.y) * 16;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int inner = 0; inner < K; inner += 16) {
        wmma::load_matrix_sync(a, activation + tile_m * K + inner, K);
        wmma::load_matrix_sync(b, weight + tile_n * K + inner, K);
        wmma::mma_sync(accumulator, a, b, accumulator);
    }
    for (int index = 0; index < accumulator.num_elements; ++index) {
        accumulator.x[index] *= alpha;
    }
    __shared__ float tile[16 * 16];
    wmma::store_matrix_sync(tile, accumulator, 16, wmma::mem_row_major);
    __syncthreads();
    for (int index = static_cast<int>(threadIdx.x); index < 256; index += 32) {
        const int row = index / 16;
        const int col = index % 16;
        output[(tile_m + row) * N + tile_n + col] = tile[index];
    }
}

__global__ void m54_shared_interstage_m128_kernel(
    const float* gate,
    const float* up,
    __nv_bfloat16* output,
    float limit
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= 128 * 2048) {
        return;
    }
    const float gate_value = fminf(gate[index], limit);
    const float up_value = fminf(fmaxf(up[index], -limit), limit);
    output[index] = __float2bfloat16_rn(
        gate_value / (1.0f + expf(-gate_value)) * up_value);
}

__global__ void m54_sum_m128_kernel(
    const __nv_bfloat16* routed0,
    const __nv_bfloat16* routed1,
    const __nv_bfloat16* routed2,
    const __nv_bfloat16* routed3,
    const __nv_bfloat16* routed4,
    const __nv_bfloat16* routed5,
    const float* shared,
    const int32_t* router_status,
    __nv_bfloat16* output
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= 128 * 4096) {
        return;
    }
    if (router_status[0] != 1) {
        output[index] = __float2bfloat16_rn(NAN);
        return;
    }
    const float sum = shared[index]
        + __bfloat162float(routed0[index])
        + __bfloat162float(routed1[index])
        + __bfloat162float(routed2[index])
        + __bfloat162float(routed3[index])
        + __bfloat162float(routed4[index])
        + __bfloat162float(routed5[index]);
    output[index] = __float2bfloat16_rn(sum);
}

__device__ float m54_softplus(float value) {
    if (value > 20.0f) {
        return value;
    }
    if (value < -20.0f) {
        return expf(value);
    }
    return log1pf(expf(value));
}

__global__ void m54_router_top6_kernel(
    const float* logits,
    const float* bias,
    int32_t* output_ids,
    float* output_weights,
    int32_t* status,
    float routed_scale,
    float tolerance
) {
    __shared__ float scores[256];
    __shared__ float choices[256];
    const int lane = static_cast<int>(threadIdx.x);
    scores[lane] = sqrtf(m54_softplus(logits[lane]));
    choices[lane] = scores[lane] + bias[lane];
    __syncthreads();
    if (lane != 0) {
        return;
    }
    bool selected[256] = {};
    for (int rank = 0; rank < 6; ++rank) {
        int best = -1;
        for (int expert = 0; expert < 256; ++expert) {
            if (!selected[expert]
                && (best < 0 || choices[expert] > choices[best]
                    || (choices[expert] == choices[best] && expert < best))) {
                best = expert;
            }
        }
        selected[best] = true;
        output_ids[rank] = best;
    }
    float denominator = 0.0f;
    for (int rank = 0; rank < 6; ++rank) {
        denominator += scores[output_ids[rank]];
    }
    constexpr int32_t expected_ids[6] = {139, 64, 143, 118, 79, 175};
    constexpr float expected_weights[6] = {
        0.5385668873786926f, 0.2694627344608307f, 0.18221795558929443f,
        0.17385046184062958f, 0.1706775575876236f, 0.16522446274757385f
    };
    int32_t valid = isfinite(denominator) && denominator > 0.0f ? 1 : 0;
    for (int rank = 0; rank < 6; ++rank) {
        output_weights[rank] = scores[output_ids[rank]] / denominator * routed_scale;
        if (output_ids[rank] != expected_ids[rank]
            || fabsf(output_weights[rank] - expected_weights[rank]) > tolerance) {
            valid = 0;
        }
    }
    status[0] = valid;
}

__global__ void m58_router_group_kernel(
    const float* scores,
    const float* bias,
    const uint16_t* expected_ids,
    const float* expected_weights,
    uint16_t* output_ids,
    float* output_weights,
    uint32_t* counts,
    uint32_t* offsets,
    uint16_t* gathered_tokens,
    float* gathered_weights,
    int32_t* status,
    float* routed_accum,
    uint32_t tokens,
    float routed_scale,
    float tolerance
) {
    const uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    if (lane == 0) {
        status[0] = 1;
        for (uint32_t expert = 0; expert < 256; ++expert) {
            counts[expert] = 0;
        }
    }
    __syncthreads();
    if (lane < tokens) {
        bool selected[256] = {};
        float denominator = 0.0f;
        for (uint32_t rank = 0; rank < 6; ++rank) {
            int best = -1;
            for (int expert = 0; expert < 256; ++expert) {
                const float candidate = scores[lane * 256 + expert] + bias[expert];
                const float incumbent = best < 0
                    ? -INFINITY : scores[lane * 256 + best] + bias[best];
                if (!selected[expert]
                    && (best < 0 || candidate > incumbent
                        || (candidate == incumbent && expert < best))) {
                    best = expert;
                }
            }
            if (best < 0) {
                atomicExch(status, 0);
                best = 0;
            }
            selected[best] = true;
            output_ids[lane * 6 + rank] = static_cast<uint16_t>(best);
            denominator += scores[lane * 256 + best];
        }
        if (!isfinite(denominator) || denominator <= 0.0f) {
            atomicExch(status, 0);
        } else {
            for (uint32_t rank = 0; rank < 6; ++rank) {
                const uint32_t index = lane * 6 + rank;
                const uint16_t expert = output_ids[index];
                const float weight = scores[lane * 256 + expert] / denominator * routed_scale;
                output_weights[index] = weight;
                if (expert != expected_ids[index]
                    || !isfinite(weight)
                    || fabsf(weight - expected_weights[index]) > tolerance) {
                    atomicExch(status, 0);
                }
            }
        }
    }
    __syncthreads();
    if (lane == 0) {
        for (uint32_t token = 0; token < tokens; ++token) {
            for (uint32_t rank = 0; rank < 6; ++rank) {
                ++counts[output_ids[token * 6 + rank]];
            }
        }
        offsets[0] = 0;
        for (uint32_t expert = 0; expert < 256; ++expert) {
            offsets[expert + 1] = offsets[expert] + counts[expert];
        }
        uint32_t cursor[256];
        for (uint32_t expert = 0; expert < 256; ++expert) {
            cursor[expert] = offsets[expert];
        }
        for (uint32_t token = 0; token < tokens; ++token) {
            for (uint32_t rank = 0; rank < 6; ++rank) {
                const uint32_t source = token * 6 + rank;
                const uint16_t expert = output_ids[source];
                const uint32_t destination = cursor[expert]++;
                gathered_tokens[destination] = static_cast<uint16_t>(token);
                gathered_weights[destination] = output_weights[source];
            }
        }
        if (offsets[256] != tokens * 6) {
            status[0] = 0;
        }
    }
    __syncthreads();
    for (uint32_t index = lane; index < tokens * 4096; index += blockDim.x) {
        routed_accum[index] = 0.0f;
    }
}

__global__ void m58_scatter_kernel(
    const __nv_bfloat16* expert_output,
    const uint16_t* gathered_tokens,
    float* routed_accum,
    const int32_t* status,
    uint32_t group_offset,
    uint32_t valid_rows,
    uint32_t tokens,
    uint32_t hidden
) {
    const uint32_t index = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= valid_rows * hidden || status[0] != 1) {
        return;
    }
    const uint32_t row = index / hidden;
    const uint32_t column = index % hidden;
    const uint32_t token = gathered_tokens[group_offset + row];
    if (token < tokens) {
        routed_accum[token * hidden + column] += __bfloat162float(expert_output[index]);
    }
}

__global__ void m58_finalize_kernel(
    const float* routed,
    const float* shared,
    __nv_bfloat16* output,
    const int32_t* status,
    uint32_t tokens,
    uint32_t hidden
) {
    const uint32_t index = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= 128 * hidden) {
        return;
    }
    if (index >= tokens * hidden || status[0] != 1) {
        output[index] = __float2bfloat16_rn(0.0f);
        return;
    }
    output[index] = __float2bfloat16_rn(routed[index] + shared[index]);
}

}  // namespace augur_nvfp4_detail

using namespace augur_nvfp4_detail;

extern "C" uint32_t AUGUR_CUDA_CALL
augur_cuda_nvfp4_sm120_build_capability_v1(void) {
    return 1;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_pack_scales_host_v1(
    uint32_t shape,
    uint32_t operand,
    const uint8_t* source,
    uint64_t source_len_bytes,
    uint8_t* destination,
    uint64_t destination_len_bytes
) {
#define AUGUR_PACK_CASE(ID, M, N, K) case ID: return pack_scales<M, N, K>(operand, source, source_len_bytes, destination, destination_len_bytes)
    switch (shape) {
        AUGUR_PACK_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M1_V1, 1, 2048, 4096);
        AUGUR_PACK_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M2_V1, 2, 2048, 4096);
        AUGUR_PACK_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M4_V1, 4, 2048, 4096);
        AUGUR_PACK_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M8_V1, 8, 2048, 4096);
        AUGUR_PACK_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M16_V1, 16, 2048, 4096);
        AUGUR_PACK_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M32_V1, 32, 2048, 4096);
        AUGUR_PACK_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M64_V1, 64, 2048, 4096);
        AUGUR_PACK_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M128_V1, 128, 2048, 4096);
        AUGUR_PACK_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M1_V1, 1, 4096, 2048);
        AUGUR_PACK_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M2_V1, 2, 4096, 2048);
        AUGUR_PACK_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M4_V1, 4, 4096, 2048);
        AUGUR_PACK_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M8_V1, 8, 4096, 2048);
        AUGUR_PACK_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M16_V1, 16, 4096, 2048);
        AUGUR_PACK_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M32_V1, 32, 4096, 2048);
        AUGUR_PACK_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M64_V1, 64, 4096, 2048);
        AUGUR_PACK_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M128_V1, 128, 4096, 2048);
        default: return invalid();
    }
#undef AUGUR_PACK_CASE
}

#define AUGUR_SHAPE_CASES(FN) \
    case AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M1_V1: return FN(1, 2048, 4096); \
    case AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M2_V1: return FN(2, 2048, 4096); \
    case AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M4_V1: return FN(4, 2048, 4096); \
    case AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M8_V1: return FN(8, 2048, 4096); \
    case AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M16_V1: return FN(16, 2048, 4096); \
    case AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M32_V1: return FN(32, 2048, 4096); \
    case AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M64_V1: return FN(64, 2048, 4096); \
    case AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M128_V1: return FN(128, 2048, 4096); \
    case AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M1_V1: return FN(1, 4096, 2048); \
    case AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M2_V1: return FN(2, 4096, 2048); \
    case AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M4_V1: return FN(4, 4096, 2048); \
    case AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M8_V1: return FN(8, 4096, 2048); \
    case AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M16_V1: return FN(16, 4096, 2048); \
    case AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M32_V1: return FN(32, 4096, 2048); \
    case AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M64_V1: return FN(64, 4096, 2048); \
    case AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M128_V1: return FN(128, 4096, 2048)

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_scale_extent_host_v1(
    uint32_t shape, uint32_t operand, uint64_t* out_size_bytes) {
#define AUGUR_EXTENT(M, N, K) scale_extent<M, N, K>(operand, out_size_bytes)
    switch (shape) { AUGUR_SHAPE_CASES(AUGUR_EXTENT); default: return invalid(); }
#undef AUGUR_EXTENT
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_shape_a_scale_offsets_host_v1(
    uint32_t shape, uint32_t* destination, uint64_t destination_count) {
#define AUGUR_OFFSETS(M, N, K) scale_offsets_a<M, N, K>(destination, destination_count)
    switch (shape) { AUGUR_SHAPE_CASES(AUGUR_OFFSETS); default: return invalid(); }
#undef AUGUR_OFFSETS
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_shape_workspace_size_v1(
    uint32_t shape, uint64_t* out_size_bytes) {
#define AUGUR_WORKSPACE(M, N, K) workspace_size<M, N, K>(out_size_bytes)
    switch (shape) { AUGUR_SHAPE_CASES(AUGUR_WORKSPACE); default: return invalid(); }
#undef AUGUR_WORKSPACE
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_w2_m128_a_scale_offsets_host_v1(
    uint32_t* destination,
    uint64_t destination_count
) {
    return scale_offsets_a<128, 4096, 2048>(destination, destination_count);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_w13_m128_workspace_size_v1(uint64_t* out_size_bytes) {
    return workspace_size<128, 2048, 4096>(out_size_bytes);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_w2_m128_workspace_size_v1(uint64_t* out_size_bytes) {
    return workspace_size<128, 4096, 2048>(out_size_bytes);
}

augur_cuda_result_v1 augur_nvfp4_cutlass_w13_m128_launch_v1(
    const augur_cuda_nvfp4_aot_args_v1* args,
    cudaStream_t stream
) {
    return launch<128, 2048, 4096>(args, stream);
}

augur_cuda_result_v1 augur_nvfp4_cutlass_w2_m128_launch_v1(
    const augur_cuda_nvfp4_aot_args_v1* args,
    cudaStream_t stream
) {
    return launch<128, 4096, 2048>(args, stream);
}

augur_cuda_result_v1 augur_nvfp4_cutlass_shape_launch_v1(
    uint32_t shape, const augur_cuda_nvfp4_aot_args_v1* args, cudaStream_t stream) {
#define AUGUR_LAUNCH(M, N, K) launch<M, N, K>(args, stream)
    switch (shape) { AUGUR_SHAPE_CASES(AUGUR_LAUNCH); default: return invalid(); }
#undef AUGUR_LAUNCH
}

template <int M, int N, int K>
augur_cuda_result_v1 launch_w4a16(
    const augur_cuda_nvfp4_w4a16_args_v1* args,
    cudaStream_t stream
) {
    if (args == nullptr || stream == nullptr) {
        return invalid();
    }
    const dim3 grid(N / 16, (M + 15) / 16, 1);
    nvfp4_w4a16_kernel<M, N, K><<<grid, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            static_cast<uintptr_t>(args->activation_bf16)),
        reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(args->weight_values)),
        reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(args->weight_scales)),
        reinterpret_cast<__nv_bfloat16*>(
            static_cast<uintptr_t>(args->output_bf16)),
        args->weight_global_scale);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? ok() : augur_cuda_result_v1{
        AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, static_cast<int32_t>(error)};
}

augur_cuda_result_v1 augur_nvfp4_w4a16_shape_launch_v1(
    uint32_t shape,
    const augur_cuda_nvfp4_w4a16_args_v1* args,
    cudaStream_t stream
) {
#define AUGUR_W4A16_LAUNCH(M, N, K) launch_w4a16<M, N, K>(args, stream)
    switch (shape) {
        AUGUR_SHAPE_CASES(AUGUR_W4A16_LAUNCH);
        default: return invalid();
    }
#undef AUGUR_W4A16_LAUNCH
}

augur_cuda_result_v1 augur_nvfp4_shape_interstage_launch_v1(
    const augur_cuda_nvfp4_shape_interstage_args_v1* args, cudaStream_t stream) {
    if (args == nullptr || stream == nullptr) {
        return invalid();
    }
    nvfp4_shape_interstage_kernel<<<args->capacity * (2048 / 16), 16, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(args->gate_bf16)),
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(args->up_bf16)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->route_weights_f32)),
        reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(args->output_values)),
        reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(args->output_scales)),
        reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(args->scale_offsets)),
        args->capacity, args->valid_rows, args->activation_global_scale, args->swiglu_limit);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? ok() : augur_cuda_result_v1{
        AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, static_cast<int32_t>(error)};
}

augur_cuda_result_v1 augur_m68_bf16_interstage_launch_v1(
    const augur_cuda_m68_bf16_interstage_args_v1* args,
    cudaStream_t stream
) {
    if (args == nullptr || stream == nullptr) {
        return invalid();
    }
    const uint32_t elements = args->capacity * 2048;
    m68_bf16_interstage_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(args->gate_bf16)),
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(args->up_bf16)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->route_weights_f32)),
        reinterpret_cast<__nv_bfloat16*>(static_cast<uintptr_t>(args->output_bf16)),
        args->capacity, args->valid_rows, args->swiglu_limit);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? ok() : augur_cuda_result_v1{
        AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, static_cast<int32_t>(error)};
}

#undef AUGUR_SHAPE_CASES

augur_cuda_result_v1 augur_nvfp4_interstage_m128_launch_v1(
    const augur_cuda_nvfp4_interstage_args_v1* args,
    cudaStream_t stream
) {
    if (args == nullptr || stream == nullptr) {
        return invalid();
    }
    nvfp4_interstage_m128_kernel<<<128 * (2048 / 16), 16, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(args->gate_bf16)),
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(args->up_bf16)),
        reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(args->output_values)),
        reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(args->output_scales)),
        reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(args->scale_offsets)),
        args->route_weight,
        args->activation_global_scale,
        args->swiglu_limit
    );
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? ok()
        : augur_cuda_result_v1{
            AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,
            static_cast<int32_t>(error)
        };
}

template <int N, int K>
augur_cuda_result_v1 launch_bf16_gemm_m128(
    const augur_cuda_bf16_gemm_args_v1* args,
    cudaStream_t stream
) {
    if (args == nullptr || stream == nullptr) {
        return invalid();
    }
    const dim3 grid(N / 16, 128 / 16, 1);
    bf16_gemm_m128_kernel<N, K><<<grid, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(args->activation_bf16)),
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(args->weight_bf16)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        args->alpha
    );
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? ok() : augur_cuda_result_v1{
        AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, static_cast<int32_t>(error)};
}

augur_cuda_result_v1 augur_bf16_w13_m128_launch_v1(
    const augur_cuda_bf16_gemm_args_v1* args,
    cudaStream_t stream
) {
    return launch_bf16_gemm_m128<2048, 4096>(args, stream);
}

augur_cuda_result_v1 augur_bf16_w2_m128_launch_v1(
    const augur_cuda_bf16_gemm_args_v1* args,
    cudaStream_t stream
) {
    return launch_bf16_gemm_m128<4096, 2048>(args, stream);
}

augur_cuda_result_v1 augur_m54_shared_interstage_m128_launch_v1(
    const augur_cuda_m54_shared_interstage_args_v1* args,
    cudaStream_t stream
) {
    if (args == nullptr || stream == nullptr) {
        return invalid();
    }
    m54_shared_interstage_m128_kernel<<<(128 * 2048 + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->up_f32)),
        reinterpret_cast<__nv_bfloat16*>(static_cast<uintptr_t>(args->output_bf16)),
        args->swiglu_limit
    );
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? ok() : augur_cuda_result_v1{
        AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, static_cast<int32_t>(error)};
}

augur_cuda_result_v1 augur_m54_sum_m128_launch_v1(
    const augur_cuda_m54_sum_args_v1* args,
    cudaStream_t stream
) {
    if (args == nullptr || stream == nullptr) {
        return invalid();
    }
    m54_sum_m128_kernel<<<(128 * 4096 + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(args->routed_bf16[0])),
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(args->routed_bf16[1])),
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(args->routed_bf16[2])),
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(args->routed_bf16[3])),
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(args->routed_bf16[4])),
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(args->routed_bf16[5])),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->shared_f32)),
        reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(args->router_status)),
        reinterpret_cast<__nv_bfloat16*>(static_cast<uintptr_t>(args->output_bf16))
    );
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? ok() : augur_cuda_result_v1{
        AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, static_cast<int32_t>(error)};
}

augur_cuda_result_v1 augur_m54_router_top6_launch_v1(
    const augur_cuda_m54_router_args_v1* args,
    cudaStream_t stream
) {
    if (args == nullptr || stream == nullptr) {
        return invalid();
    }
    m54_router_top6_kernel<<<1, 256, 0, stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->logits_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->bias_f32)),
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(args->output_ids_i32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_weights_f32)),
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(args->status_i32)),
        args->routed_scale,
        args->weight_tolerance
    );
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? ok() : augur_cuda_result_v1{
        AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, static_cast<int32_t>(error)};
}

augur_cuda_result_v1 augur_m58_router_group_launch_v1(
    const augur_cuda_m58_router_group_args_v1* args,
    cudaStream_t stream
) {
    if (args == nullptr || stream == nullptr) {
        return invalid();
    }
    m58_router_group_kernel<<<1, 128, 0, stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->scores_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->bias_f32)),
        reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(args->expected_ids_u16)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->expected_weights_f32)),
        reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(args->output_ids_u16)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_weights_f32)),
        reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(args->counts_u32)),
        reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(args->offsets_u32)),
        reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(args->gathered_tokens_u16)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->gathered_weights_f32)),
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(args->status_i32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->routed_accum_f32)),
        args->tokens,
        args->routed_scale,
        args->weight_tolerance
    );
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? ok() : augur_cuda_result_v1{
        AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, static_cast<int32_t>(error)};
}

augur_cuda_result_v1 augur_m58_scatter_launch_v1(
    const augur_cuda_m58_scatter_args_v1* args,
    cudaStream_t stream
) {
    if (args == nullptr || stream == nullptr) {
        return invalid();
    }
    const uint32_t elements = args->valid_rows * args->hidden;
    m58_scatter_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(args->expert_output_bf16)),
        reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(args->gathered_tokens_u16)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->routed_accum_f32)),
        reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(args->status_i32)),
        args->group_offset,
        args->valid_rows,
        args->tokens,
        args->hidden
    );
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? ok() : augur_cuda_result_v1{
        AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, static_cast<int32_t>(error)};
}

augur_cuda_result_v1 augur_m58_finalize_launch_v1(
    const augur_cuda_m58_finalize_args_v1* args,
    cudaStream_t stream
) {
    if (args == nullptr || stream == nullptr) {
        return invalid();
    }
    m58_finalize_kernel<<<(128 * 4096 + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->routed_accum_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->shared_f32)),
        reinterpret_cast<__nv_bfloat16*>(static_cast<uintptr_t>(args->output_bf16)),
        reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(args->status_i32)),
        args->tokens,
        args->hidden
    );
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? ok() : augur_cuda_result_v1{
        AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, static_cast<int32_t>(error)};
}
