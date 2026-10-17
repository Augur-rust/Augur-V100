// Included by the stream-owned cuBLAS unit, under AUGUR_CUDA_QWEN35_VISION_AFFINE.
// No S*S buffer, host callback, hidden synchronization or BF16 hardware assumption.
#include <math_constants.h>
static_assert(sizeof(augur_cuda_qwen35_vision_control_args_v1) == 136, "Vision control ABI drift");
static_assert(offsetof(augur_cuda_qwen35_vision_control_args_v1, input_a) == 24, "Vision control input offset");
static_assert(offsetof(augur_cuda_qwen35_vision_control_args_v1, status) == 120, "Vision control status offset");

namespace {
bool vision_control_valid(int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_vision_control_args_v1* a, bool inspect_device) {
    if (ordinal < 0 || stream == nullptr || stream->stream == nullptr || stream->ordinal != ordinal
        || a == nullptr || a->abi_version != 1 || a->struct_size != sizeof(*a) || a->reserved != 0
        || a->tokens == 0 || a->tokens > 4096 || a->status_bytes != 4) return false;
    const uint64_t n = a->tokens, cols = a->cols;
    uint64_t expected[6] = {};
    bool half_output = false, inplace = false;
    switch (a->operation) {
    case 1:  // Learned position interpolation + patch affine output.
        if (cols != 1152) return false;
        expected[0] = n * cols * 4; expected[1] = 2304 * cols * 4;
        expected[2] = n * 4 * 4; expected[3] = n * 4 * 4;
        expected[5] = expected[0]; inplace = true; break;
    case 2:  // Stable two-pass LN, FP32 vectors, RNE F16 output.
        if (cols != 1152) return false;
        expected[0] = n * cols * 4; expected[1] = cols * 4; expected[2] = cols * 4;
        expected[5] = n * cols * 2; half_output = true; break;
    case 3:
        if (cols != 1152 && cols != 4608) return false;
        expected[0] = n * cols * 4; expected[5] = n * cols * 2; half_output = true; break;
    case 4:
        if (cols != 4304) return false;
        expected[0] = n * cols * 4; expected[5] = n * cols * 2; half_output = true; break;
    case 5:
        if (cols != 4608) return false;
        expected[0] = n * cols * 4; expected[5] = n * cols * 2; half_output = true; break;
    case 6:
        if (cols != 1152) return false;
        expected[0] = expected[1] = expected[5] = n * cols * 4; inplace = true; break;
    case 7:
        if (cols != 72) return false;
        expected[0] = expected[5] = n * 3456 * 4;
        expected[1] = expected[2] = n * 72 * 4; inplace = true; break;
    case 8:
        if (cols != 72) return false;
        expected[0] = n * 3456 * 4; expected[5] = n * 1152 * 4; break;
    default: return false;
    }
    const uint64_t addresses[7] = {a->input_a,a->input_b,a->input_c,a->input_d,a->input_e,a->output,a->status};
    const uint64_t sizes[7] = {a->input_a_bytes,a->input_b_bytes,a->input_c_bytes,a->input_d_bytes,a->input_e_bytes,a->output_bytes,a->status_bytes};
    if (inplace && a->output != a->input_a) return false;
    Range ranges[7];
    for (size_t i = 0; i < 7; ++i) {
        const uint64_t required = i < 6 ? expected[i] : 4;
        if (sizes[i] != required || (required == 0 && addresses[i] != 0)) return false;
        ranges[i] = {addresses[i],sizes[i],i == 5 && half_output ? 2U : 4U};
        if (required == 0) continue;
        if (!range_valid(ranges[i]) || (inspect_device && !device_range(ranges[i], ordinal))) return false;
        for (size_t j = 0; j < i; ++j) {
            if (sizes[j] == 0 || (inplace && i == 5 && j == 0)) continue;
            if (ranges_overlap(ranges[i],ranges[j])) return false;
        }
    }
    return true;
}

__device__ float vision_finite(float x, uint32_t* status) {
    if (isfinite(x)) return x;
    atomicOr(status, 1U); return 0.0f;
}
__device__ __half vision_half(float x, uint32_t* status) {
    if (!isfinite(x) || fabsf(x) > 65504.0f) { atomicOr(status, 2U); return __float2half_rn(0.0f); }
    return __float2half_rn(x);
}
__device__ float vision_warp_sum(float x) {
    for (int delta = 16; delta > 0; delta /= 2) x += __shfl_down_sync(0xffffffffU, x, delta);
    return x;
}
__device__ float vision_block_sum_128(float x, float* scratch) {
    x = vision_warp_sum(x);
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x / 32] = x;
    __syncthreads();
    if (threadIdx.x < 32) {
        x = threadIdx.x < 4 ? scratch[threadIdx.x] : 0.0f;
        x = vision_warp_sum(x);
        if (threadIdx.x == 0) scratch[0] = x;
    }
    __syncthreads();
    return scratch[0];
}

__global__ void vision_control_vector_kernel(augur_cuda_qwen35_vision_control_args_v1 a) {
    const float* x = reinterpret_cast<const float*>(a.input_a);
    const float* b = reinterpret_cast<const float*>(a.input_b);
    float* output = reinterpret_cast<float*>(a.output);
    __half* half_output = reinterpret_cast<__half*>(a.output);
    uint32_t* status = reinterpret_cast<uint32_t*>(a.status);
    const uint64_t count = static_cast<uint64_t>(a.tokens) * a.cols;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        float value = vision_finite(x[i], status);
        if (a.operation == 1) {
            const uint64_t row = i / 1152, col = i % 1152;
            const uint32_t* indices = reinterpret_cast<const uint32_t*>(a.input_c) + row * 4;
            const float* weights = reinterpret_cast<const float*>(a.input_d) + row * 4;
            float position = 0.0f;
            for (int j = 0; j < 4; ++j) {
                if (indices[j] >= 2304) { atomicOr(status, 4U); continue; }
                position += b[static_cast<uint64_t>(indices[j]) * 1152 + col] * weights[j];
            }
            output[i] = vision_finite(value + position, status);
        } else if (a.operation == 6) {
            output[i] = vision_finite(value + b[i], status);
        } else {
            if (a.operation == 4) {
                value = 0.5f * value * (1.0f + tanhf(0.7978845608028654f * (value + 0.044715f * value * value * value)));
            } else if (a.operation == 5) {
                value = 0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
            }
            half_output[i] = vision_half(value, status);
        }
    }
}

__global__ void vision_layernorm_half_kernel(augur_cuda_qwen35_vision_control_args_v1 a) {
    __shared__ float scratch[4];
    const float* input = reinterpret_cast<const float*>(a.input_a) + static_cast<uint64_t>(blockIdx.x) * a.cols;
    const float* weight = reinterpret_cast<const float*>(a.input_b);
    const float* bias = reinterpret_cast<const float*>(a.input_c);
    __half* output = reinterpret_cast<__half*>(a.output) + static_cast<uint64_t>(blockIdx.x) * a.cols;
    uint32_t* status = reinterpret_cast<uint32_t*>(a.status);
    float sum = 0.0f;
    for (uint32_t col = threadIdx.x; col < a.cols; col += 128) sum += vision_finite(input[col], status);
    const float mean = vision_block_sum_128(sum, scratch) / a.cols;
    __syncthreads(); // all warps retain mean before scratch is reused for variance
    sum = 0.0f;
    for (uint32_t col = threadIdx.x; col < a.cols; col += 128) {
        const float delta = vision_finite(input[col], status) - mean;
        sum += delta * delta;
    }
    const float inverse = rsqrtf(vision_block_sum_128(sum, scratch) / a.cols + 1e-6f);
    for (uint32_t col = threadIdx.x; col < a.cols; col += 128) {
        const float value = (input[col] - mean) * inverse * weight[col] + bias[col];
        output[col] = vision_half(value, status);
    }
}

__global__ void vision_qk_rotary_kernel(augur_cuda_qwen35_vision_control_args_v1 a) {
    float* qkv = reinterpret_cast<float*>(a.output);
    const float* cosine = reinterpret_cast<const float*>(a.input_b);
    const float* sine = reinterpret_cast<const float*>(a.input_c);
    uint32_t* status = reinterpret_cast<uint32_t*>(a.status);
    const uint64_t count = static_cast<uint64_t>(a.tokens) * 2 * 16 * 36;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        const uint64_t d = i % 36, head = (i / 36) % 16, role = (i / (36 * 16)) % 2;
        const uint64_t token = i / (36 * 16 * 2), at = token * 3456 + role * 1152 + head * 72 + d;
        const float x = qkv[at], y = qkv[at + 36];
        const uint64_t phase = token * 72 + d;
        // Each thread owns both halves. In-place rotation cannot read a neighbor's newly-written value.
        qkv[at] = vision_finite(x * cosine[phase] - y * sine[phase], status);
        qkv[at + 36] = vision_finite(y * cosine[phase + 36] + x * sine[phase + 36], status);
    }
}

__global__ void vision_online_attention_kernel(augur_cuda_qwen35_vision_control_args_v1 a) {
    const uint32_t lane = threadIdx.x & 31;
    const uint64_t query = static_cast<uint64_t>(blockIdx.x) * 4 + threadIdx.x / 32;
    if (query >= static_cast<uint64_t>(a.tokens) * 16) return; // whole warp exits
    const uint64_t token = query / 16, head = query % 16;
    const float* __restrict__ qkv = reinterpret_cast<const float*>(a.input_a);
    float* output = reinterpret_cast<float*>(a.output);
    uint32_t* status = reinterpret_cast<uint32_t*>(a.status);
    float q[3], acc[3] = {0.0f,0.0f,0.0f};
    for (int i = 0; i < 3; ++i) {
        const uint32_t d = lane + i * 32;
        q[i] = d < 72 ? qkv[token * 3456 + head * 72 + d] : 0.0f;
    }
    float maximum = -CUDART_INF_F, denominator = 0.0f;
    for (uint32_t key = 0; key < a.tokens; ++key) {
        const uint64_t base = static_cast<uint64_t>(key) * 3456 + head * 72;
        float dot = 0.0f;
        for (int i = 0; i < 3; ++i) {
            const uint32_t d = lane + i * 32;
            if (d < 72) dot += q[i] * qkv[base + 1152 + d];
        }
        dot = vision_warp_sum(dot);
        dot = vision_finite(__shfl_sync(0xffffffffU, dot, 0) * 0.1178511301977579f, status);
        const float next = fmaxf(maximum, dot), alpha = expf(maximum - next), beta = expf(dot - next);
        denominator = denominator * alpha + beta;
        for (int i = 0; i < 3; ++i) {
            const uint32_t d = lane + i * 32;
            if (d < 72) acc[i] = acc[i] * alpha + beta * vision_finite(qkv[base + 2304 + d], status);
        }
        maximum = next;
    }
    for (int i = 0; i < 3; ++i) {
        const uint32_t d = lane + i * 32;
        if (d < 72) output[token * 1152 + head * 72 + d] = vision_finite(acc[i] / denominator, status);
    }
}
} // namespace

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_vision_control_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_vision_control_args_v1* args) {
    return vision_control_valid(ordinal, stream, args, true)
        ? result(AUGUR_CUDA_STATUS_OK, 0) : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_vision_control_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_vision_control_args_v1* args) {
    if (!vision_control_valid(ordinal, stream, args, false)) return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const cudaError_t selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, selected);
    if (args->operation == 2) vision_layernorm_half_kernel<<<args->tokens,128,0,stream->stream>>>(*args);
    else if (args->operation == 7) {
        const uint64_t count = static_cast<uint64_t>(args->tokens) * 2 * 16 * 36;
        vision_qk_rotary_kernel<<<launch_blocks(count),kThreads,0,stream->stream>>>(*args);
    } else if (args->operation == 8) {
        vision_online_attention_kernel<<<(args->tokens * 16 + 3) / 4,128,0,stream->stream>>>(*args);
    } else {
        vision_control_vector_kernel<<<launch_blocks(static_cast<uint64_t>(args->tokens) * args->cols),kThreads,0,stream->stream>>>(*args);
    }
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? result(AUGUR_CUDA_STATUS_OK, 0) : cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}
