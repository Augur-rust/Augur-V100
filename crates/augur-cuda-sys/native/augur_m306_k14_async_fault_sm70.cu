#include "augur_cuda_internal.h"

#include <cuda_runtime_api.h>

namespace {

__global__ void augur_m306_k14_device_trap_kernel(void) {
    asm volatile("trap;");
}

augur_cuda_result_v1 result(augur_cuda_status_t status, cudaError_t raw_status) {
    return {status, static_cast<int32_t>(raw_status)};
}

}  // namespace

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_m306_k14_faulted_memcpy_d2h_sync_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    void* dst,
    size_t dst_len_bytes,
    augur_cuda_device_ptr_t src,
    size_t src_offset_bytes,
    size_t copy_len_bytes,
    uint32_t* out_progress_mask,
    int32_t* out_launch_raw_cuda_status,
    int32_t* out_peek_raw_cuda_status
) {
    if (stream == nullptr || stream->stream == nullptr || stream->ordinal != ordinal
        || ordinal < 0 || dst == nullptr || src == 0 || copy_len_bytes == 0
        || copy_len_bytes > dst_len_bytes || out_progress_mask == nullptr
        || out_launch_raw_cuda_status == nullptr || out_peek_raw_cuda_status == nullptr) {
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    *out_progress_mask = 0;
    *out_launch_raw_cuda_status = static_cast<int32_t>(cudaSuccess);
    *out_peek_raw_cuda_status = static_cast<int32_t>(cudaSuccess);

    cudaError_t error = cudaSetDevice(ordinal);
    if (error != cudaSuccess) {
        return result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, error);
    }

    const void* src_offset = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(src)
    ) + src_offset_bytes;
    error = cudaMemcpyAsync(
        dst,
        src_offset,
        copy_len_bytes,
        cudaMemcpyDeviceToHost,
        stream->stream
    );
    if (error != cudaSuccess) {
        return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    }
    *out_progress_mask |= AUGUR_CUDA_M306_K14_PROGRESS_RECEIPT_D2H_SUBMITTED_V1;

    augur_m306_k14_device_trap_kernel<<<1, 1, 0, stream->stream>>>();
    const cudaError_t launch_error = cudaGetLastError();
    *out_launch_raw_cuda_status = static_cast<int32_t>(launch_error);
    if (launch_error == cudaSuccess) {
        *out_progress_mask |= AUGUR_CUDA_M306_K14_PROGRESS_TRAP_SUBMITTED_V1;
    }

    cudaError_t peek_error = cudaSuccess;
    if (launch_error == cudaSuccess) {
        peek_error = cudaPeekAtLastError();
        *out_peek_raw_cuda_status = static_cast<int32_t>(peek_error);
        if (peek_error == cudaSuccess) {
            *out_progress_mask |= AUGUR_CUDA_M306_K14_PROGRESS_IMMEDIATE_PEEK_CLEAR_V1;
        }
    }

    // Once the D2H is submitted, every branch performs exactly one terminal
    // wait before returning so the pinned destination never outlives in-flight
    // DMA. No CUDA API may follow this call when the trap taints the context.
    const cudaError_t wait_error = cudaStreamSynchronize(stream->stream);
    if (wait_error != cudaSuccess) {
        *out_progress_mask |= AUGUR_CUDA_M306_K14_PROGRESS_COMPLETION_FAILURE_OBSERVED_V1;
    }
    if (wait_error != cudaSuccess) {
        return result(AUGUR_CUDA_STATUS_COMPLETION_FAILURE, wait_error);
    }
    if (launch_error != cudaSuccess) {
        return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
    }
    if (peek_error != cudaSuccess) {
        return result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, peek_error);
    }
    return result(AUGUR_CUDA_STATUS_OK, cudaSuccess);
}
