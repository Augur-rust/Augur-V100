#include "augur_cuda_internal.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

constexpr uint32_t KEY_HEADS = AUGUR_CUDA_QWEN35_LINEAR_STEP_KEY_HEADS_V1;
constexpr uint32_t VALUE_HEADS = AUGUR_CUDA_QWEN35_LINEAR_STEP_VALUE_HEADS_V1;
constexpr uint32_t HEAD_DIM = AUGUR_CUDA_QWEN35_LINEAR_STEP_HEAD_DIM_V1;
constexpr uint32_t KERNEL_SIZE = AUGUR_CUDA_QWEN35_LINEAR_STEP_KERNEL_SIZE_V1;
constexpr uint32_t QKV_ELEMENTS = AUGUR_CUDA_QWEN35_LINEAR_STEP_QKV_ELEMENTS_V1;
constexpr uint32_t OUTPUT_ELEMENTS = AUGUR_CUDA_QWEN35_LINEAR_STEP_OUTPUT_ELEMENTS_V1;
constexpr uint32_t STATE_ELEMENTS = AUGUR_CUDA_QWEN35_LINEAR_STEP_STATE_ELEMENTS_V1;
// The depthwise convolution runs over the complete [Q, K, V] projection,
// not just the value heads.  Keep these derived from the ABI extent so the
// native validation and kernels cannot silently disagree with the Rust views.
constexpr uint32_t CONV_CHANNELS = QKV_ELEMENTS;
constexpr uint32_t CONV_HISTORY_ELEMENTS = CONV_CHANNELS * (KERNEL_SIZE - 1);
constexpr uint32_t CONV_WEIGHT_ELEMENTS = CONV_CHANNELS * KERNEL_SIZE;
constexpr uint32_t PROJECTION_ELEMENTS =
    AUGUR_CUDA_QWEN35_LINEAR_PROJECTION_POSTPROCESS_ELEMENTS_V1;
static_assert(
    CONV_HISTORY_ELEMENTS == AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_HISTORY_ELEMENTS_V1,
    "Qwen linear-step history extent drift"
);
static_assert(
    CONV_WEIGHT_ELEMENTS == AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_WEIGHT_ELEMENTS_V1,
    "Qwen linear-step weight extent drift"
);

static augur_cuda_result_v1 ok() {
    return {AUGUR_CUDA_STATUS_OK, static_cast<int32_t>(cudaSuccess)};
}

static augur_cuda_result_v1 error(augur_cuda_status_t status, cudaError_t raw) {
    return {status, static_cast<int32_t>(raw)};
}

static augur_cuda_result_v1 select_device(int32_t ordinal) {
    if (ordinal < 0) return error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    const cudaError_t result = cudaSetDevice(ordinal);
    return result == cudaSuccess ? ok() : error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, result);
}

static bool range_ok(uint64_t address, uint64_t length) {
    return address != 0 && length != 0 && address <= UINT64_MAX - length;
}

static bool overlap(uint64_t left, uint64_t left_len, uint64_t right, uint64_t right_len) {
    return left < right + right_len && right < left + left_len;
}

static bool exact(uint64_t length, uint64_t elements) {
    return length == elements * sizeof(float);
}

static augur_cuda_result_v1 validate(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_linear_step_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != AUGUR_CUDA_QWEN35_LINEAR_STEP_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_qwen35_linear_step_args_v1)
        || args->reserved != 0) {
        return error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (!exact(args->qkv_len_bytes, QKV_ELEMENTS)
        || !exact(args->z_len_bytes, OUTPUT_ELEMENTS)
        || !exact(args->conv_weights_len_bytes, CONV_WEIGHT_ELEMENTS)
        || !exact(args->conv_history_len_bytes, CONV_HISTORY_ELEMENTS)
        || !exact(args->decay_gate_len_bytes, VALUE_HEADS)
        || !exact(args->beta_len_bytes, VALUE_HEADS)
        || !exact(args->state_len_bytes, STATE_ELEMENTS)
        || !exact(args->norm_weight_len_bytes, HEAD_DIM)
        || !exact(args->activated_len_bytes, QKV_ELEMENTS)
        || !exact(args->recurrent_len_bytes, OUTPUT_ELEMENTS)
        || !exact(args->output_len_bytes, OUTPUT_ELEMENTS)) {
        return error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t addresses[] = {
        args->qkv_f32, args->z_f32, args->conv_weights_f32, args->conv_history_f32,
        args->decay_gate_f32, args->beta_f32, args->state_f32, args->norm_weight_f32,
        args->activated_f32, args->recurrent_f32, args->output_f32,
    };
    const uint64_t lengths[] = {
        args->qkv_len_bytes, args->z_len_bytes, args->conv_weights_len_bytes,
        args->conv_history_len_bytes, args->decay_gate_len_bytes, args->beta_len_bytes,
        args->state_len_bytes, args->norm_weight_len_bytes, args->activated_len_bytes,
        args->recurrent_len_bytes, args->output_len_bytes,
    };
    for (uint32_t i = 0; i < 11; ++i) {
        if (!range_ok(addresses[i], lengths[i]) || (addresses[i] % alignof(float)) != 0) {
            return error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t j = i + 1; j < 11; ++j) {
            if (overlap(addresses[i], lengths[i], addresses[j], lengths[j])) {
                return error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        return error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return select_device(ordinal);
}

__device__ inline float silu(float value) {
    if (value >= 0.0f) {
        return value / (1.0f + expf(-value));
    }
    const float exponent = expf(value);
    return value * exponent / (1.0f + exponent);
}

__global__ void conv_kernel(
    const float* qkv,
    const float* weights,
    float* history,
    float* activated
) {
    const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= CONV_CHANNELS) return;
    const uint32_t history_base = channel * (KERNEL_SIZE - 1);
    const uint32_t weight_base = channel * KERNEL_SIZE;
    float sum = 0.0f;
    sum += history[history_base] * weights[weight_base];
    sum += history[history_base + 1] * weights[weight_base + 1];
    sum += history[history_base + 2] * weights[weight_base + 2];
    sum += qkv[channel] * weights[weight_base + 3];
    activated[channel] = silu(sum);
    history[history_base] = history[history_base + 1];
    history[history_base + 1] = history[history_base + 2];
    history[history_base + 2] = qkv[channel];
}

template<bool CHECKPOINT_SOURCE>
__global__ void delta_kernel(
    const float* activated,
    const float* decay_gate,
    const float* beta,
    float* state,
    float* recurrent,
    float epsilon
) {
    const uint32_t head = blockIdx.x;
    const uint32_t row = threadIdx.x;
    if (head >= VALUE_HEADS || row >= HEAD_DIM) return;
    const uint32_t q_head = CHECKPOINT_SOURCE
        ? head / (VALUE_HEADS / KEY_HEADS) : head % KEY_HEADS;
    __shared__ float q_scale;
    __shared__ float k_scale;
    if (row == 0) {
        const uint32_t q_base = 0;
        const uint32_t k_base = KEY_HEADS * HEAD_DIM;
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (uint32_t col = 0; col < HEAD_DIM; ++col) {
            const float q = activated[q_base + q_head * HEAD_DIM + col];
            const float k = activated[k_base + q_head * HEAD_DIM + col];
            q_sum += q * q;
            k_sum += k * k;
        }
        const float floor = epsilon * epsilon;
        q_scale = rsqrtf(CHECKPOINT_SOURCE ? q_sum + 1.0e-6f : fmaxf(q_sum, floor));
        k_scale = rsqrtf(CHECKPOINT_SOURCE ? k_sum + 1.0e-6f : fmaxf(k_sum, floor));
    }
    __syncthreads();

    const float* q = activated + q_head * HEAD_DIM;
    const float* k = activated + KEY_HEADS * HEAD_DIM + q_head * HEAD_DIM;
    const float* v = activated + 2 * KEY_HEADS * HEAD_DIM + head * HEAD_DIM;
    float* matrix = state + head * HEAD_DIM * HEAD_DIM;
    const float decay = expf(decay_gate[head]);
    for (uint32_t col = 0; col < HEAD_DIM; ++col) {
        matrix[row * HEAD_DIM + col] *= decay;
    }
    __syncthreads();

    float projection = 0.0f;
    for (uint32_t col = 0; col < HEAD_DIM; ++col) {
        projection += matrix[row * HEAD_DIM + col] * (k[col] * k_scale);
    }
    const float delta = (v[row] - projection) * beta[head];
    for (uint32_t col = 0; col < HEAD_DIM; ++col) {
        matrix[row * HEAD_DIM + col] += (k[col] * k_scale) * delta;
    }
    float query_projection = 0.0f;
    for (uint32_t col = 0; col < HEAD_DIM; ++col) {
        query_projection += matrix[row * HEAD_DIM + col] * (q[col] * q_scale);
    }
    recurrent[head * HEAD_DIM + row] = query_projection * (1.0f / sqrtf(static_cast<float>(HEAD_DIM)));
}

#if defined(AUGUR_CUDA_QWEN35_GDN_WARP_ROW)
__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

template<bool CHECKPOINT_SOURCE,int Warps=4,bool WarpNorm=false>
__global__ __launch_bounds__(Warps*32, 2) void delta_warp_row_kernel(
    const float* activated,
    const float* decay_gate,
    const float* beta,
    float* state,
    float* recurrent,
    float epsilon
) {
    constexpr uint32_t warps_per_block = Warps;
    constexpr uint32_t values_per_lane = HEAD_DIM / 32U;
    static_assert((HEAD_DIM % 32U) == 0U, "Qwen GDN head dimension must tile one warp");

    const uint32_t head = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t output_row = blockIdx.y * warps_per_block + threadIdx.y;
    if (head >= VALUE_HEADS || output_row >= HEAD_DIM) return;

    __shared__ float q_scale;
    __shared__ float k_scale;
    __shared__ float decay;
    __shared__ float beta_value;
    const uint32_t q_head = CHECKPOINT_SOURCE
        ? head / (VALUE_HEADS / KEY_HEADS) : head % KEY_HEADS;
    const float* q = activated + q_head * HEAD_DIM;
    const float* k = activated + KEY_HEADS * HEAD_DIM + q_head * HEAD_DIM;
    const float* v = activated + 2U * KEY_HEADS * HEAD_DIM + head * HEAD_DIM;
    // Same sequential Q/K sum-of-squares and normalization as the scalar
    // loader, but coalesced loads + ordered broadcasts avoid 256 serialized
    // single-lane loads per CTA. All active lanes use the identical recurrence.
    // Checkpoint B1 only; generic/GGUF and batched GDN paths remain unchanged.
    if constexpr(WarpNorm) {
        if(threadIdx.y==0) {
            float qsum=0.f,ksum=0.f;
            #pragma unroll
            for(unsigned block=0;block<4;++block){
                const float qv=q[block*32+lane],kv=k[block*32+lane];
                #pragma unroll
                for(unsigned from=0;from<32;++from){
                    const float a=__shfl_sync(0xffffffffu,qv,from);
                    const float b=__shfl_sync(0xffffffffu,kv,from);
                    qsum+=a*a;ksum+=b*b;
                }
            }
            if(lane==0){
                const float floor=epsilon*epsilon;
                q_scale=rsqrtf(CHECKPOINT_SOURCE?qsum+1e-6f:fmaxf(qsum,floor));
                k_scale=rsqrtf(CHECKPOINT_SOURCE?ksum+1e-6f:fmaxf(ksum,floor));
                decay=expf(decay_gate[head]);beta_value=beta[head];
            }
        }
    } else {
    if (threadIdx.x == 0U && threadIdx.y == 0U) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (uint32_t col = 0; col < HEAD_DIM; ++col) {
            q_sum += q[col] * q[col];
            k_sum += k[col] * k[col];
        }
        const float floor = epsilon * epsilon;
        q_scale = rsqrtf(CHECKPOINT_SOURCE ? q_sum + 1.0e-6f : fmaxf(q_sum, floor));
        k_scale = rsqrtf(CHECKPOINT_SOURCE ? k_sum + 1.0e-6f : fmaxf(k_sum, floor));
        decay = expf(decay_gate[head]);
        beta_value = beta[head];
    }

    }
    __syncthreads();

    float* matrix_row = state + (head * HEAD_DIM + output_row) * HEAD_DIM;
    float state_values[values_per_lane];
    float q_values[values_per_lane];
    float k_values[values_per_lane];
    float projection = 0.0f;
    #pragma unroll
    for (uint32_t item = 0; item < values_per_lane; ++item) {
        const uint32_t col = item * 32U + lane;
        q_values[item] = q[col] * q_scale;
        k_values[item] = k[col] * k_scale;
        state_values[item] = matrix_row[col] * decay;
        projection += state_values[item] * k_values[item];
    }
    projection = __shfl_sync(0xffffffffU, warp_sum(projection), 0);
    const float delta = (v[output_row] - projection) * beta_value;

    float query_projection = 0.0f;
    #pragma unroll
    for (uint32_t item = 0; item < values_per_lane; ++item) {
        const uint32_t col = item * 32U + lane;
        state_values[item] += k_values[item] * delta;
        matrix_row[col] = state_values[item];
        query_projection += state_values[item] * q_values[item];
    }
    query_projection = warp_sum(query_projection);
    if (lane == 0U) {
        recurrent[head * HEAD_DIM + output_row] =
            query_projection * (1.0f / sqrtf(static_cast<float>(HEAD_DIM)));
    }
}
#endif

__global__ void gated_norm_kernel(
    const float* recurrent,
    const float* norm_weight,
    const float* z,
    float* output,
    float epsilon
) {
    const uint32_t head = blockIdx.x;
    const uint32_t component = threadIdx.x;
    if (head >= VALUE_HEADS || component >= HEAD_DIM) return;
    __shared__ float scale;
    if (component == 0) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < HEAD_DIM; ++i) {
            const float value = recurrent[head * HEAD_DIM + i];
            sum += value * value;
        }
        scale = rsqrtf(sum / static_cast<float>(HEAD_DIM) + epsilon);
    }
    __syncthreads();
    const uint32_t index = head * HEAD_DIM + component;
    output[index] = recurrent[index] * scale * norm_weight[component] * silu(z[index]);
}

static augur_cuda_result_v1 validate_projection(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_linear_projection_postprocess_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != AUGUR_CUDA_QWEN35_LINEAR_PROJECTION_POSTPROCESS_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_qwen35_linear_projection_postprocess_args_v1)
        || args->elements != PROJECTION_ELEMENTS
        || args->reserved != 0) {
        return error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    const uint64_t addresses[] = {
        args->alpha_raw_f32, args->beta_raw_f32, args->dt_bias_f32,
        args->ssm_a_f32, args->decay_gate_f32, args->beta_out_f32,
    };
    const uint64_t lengths[] = {
        args->alpha_raw_len_bytes, args->beta_raw_len_bytes, args->dt_bias_len_bytes,
        args->ssm_a_len_bytes, args->decay_gate_len_bytes, args->beta_out_len_bytes,
    };
    for (uint32_t i = 0; i < 6; ++i) {
        if (!exact(lengths[i], PROJECTION_ELEMENTS)
            || !range_ok(addresses[i], lengths[i])
            || (addresses[i] % alignof(float)) != 0) {
            return error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t j = i + 1; j < 6; ++j) {
            if (overlap(addresses[i], lengths[i], addresses[j], lengths[j])) {
                return error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    return select_device(ordinal);
}

__device__ inline float projection_softplus(float value) {
    if (value > 20.0f) return value;
    if (value < -20.0f) return expf(value);
    return logf(1.0f + expf(value));
}

__device__ inline float projection_sigmoid(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
    const float exponent = expf(value);
    return exponent / (1.0f + exponent);
}

__global__ void projection_postprocess_kernel(
    const float* alpha_raw,
    const float* beta_raw,
    const float* dt_bias,
    const float* ssm_a,
    float* decay_gate,
    float* beta_out
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= PROJECTION_ELEMENTS) return;
    const float alpha = alpha_raw[index] + dt_bias[index];
    decay_gate[index] = projection_softplus(alpha) * ssm_a[index];
    beta_out[index] = projection_sigmoid(beta_raw[index]);
}

static augur_cuda_result_v1 submit_projection_impl(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_linear_projection_postprocess_args_v1* args
) {
    augur_cuda_result_v1 valid = validate_projection(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    projection_postprocess_kernel<<<1, 64, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->alpha_raw_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->beta_raw_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->dt_bias_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->ssm_a_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->decay_gate_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->beta_out_f32)));
    const cudaError_t launch_error = cudaGetLastError();
    return launch_error == cudaSuccess
        ? ok()
        : error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
}

template<bool CHECKPOINT_SOURCE = false>
static augur_cuda_result_v1 submit_impl(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_linear_step_args_v1* args
) {
    augur_cuda_result_v1 valid = validate(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    conv_kernel<<<(CONV_CHANNELS + 255) / 256, 256, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->qkv_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->conv_weights_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->conv_history_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->activated_f32)));
#if defined(AUGUR_CUDA_QWEN35_GDN_WARP_ROW)
    constexpr uint32_t gdn_warps_per_block = CHECKPOINT_SOURCE ? 8U : 4U;
    const dim3 gdn_grid(VALUE_HEADS, (HEAD_DIM + gdn_warps_per_block - 1U)
        / gdn_warps_per_block, 1U);
    const dim3 gdn_block(32U, gdn_warps_per_block, 1U);
    delta_warp_row_kernel<CHECKPOINT_SOURCE, gdn_warps_per_block, CHECKPOINT_SOURCE><<<gdn_grid, gdn_block, 0, stream->stream>>>(
#else
    delta_kernel<CHECKPOINT_SOURCE><<<VALUE_HEADS, HEAD_DIM, 0, stream->stream>>>(
#endif
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->activated_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->decay_gate_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->beta_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->state_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->recurrent_f32)), epsilon);
    gated_norm_kernel<<<VALUE_HEADS, HEAD_DIM, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->recurrent_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->norm_weight_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->z_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)), epsilon);
    const cudaError_t launch_error = cudaGetLastError();
    return launch_error == cudaSuccess
        ? ok()
        : error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
}

} // namespace

// Safetensors source: grouped value heads and additive Q/K L2 epsilon.
// Separate entrypoint preserves the established GGUF V1 interpretation.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_checkpoint_linear_step_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_linear_step_args_v1* args
) {
    return submit_impl<true>(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_linear_step_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_linear_step_args_v1* args
) {
    return submit_impl(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_linear_step_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_linear_step_args_v1* args
) {
    augur_cuda_result_v1 submitted = submit_impl(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    const cudaError_t sync_error = cudaStreamSynchronize(stream->stream);
    return sync_error == cudaSuccess
        ? ok()
        : error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, sync_error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_linear_projection_postprocess_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_linear_projection_postprocess_args_v1* args
) {
    return submit_projection_impl(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_linear_projection_postprocess_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_linear_projection_postprocess_args_v1* args
) {
    augur_cuda_result_v1 submitted = submit_projection_impl(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    const cudaError_t sync_error = cudaStreamSynchronize(stream->stream);
    return sync_error == cudaSuccess
        ? ok()
        : error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, sync_error);
}
