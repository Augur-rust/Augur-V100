#ifndef AUGUR_CUDA_RUNTIME_H
#define AUGUR_CUDA_RUNTIME_H

/* TRUST BOUNDARY (docs/UNSAFE_TRUST_BOUNDARIES.md, TB-ABI / TB-ASYNC):
 * This C ABI does not retain the caller's allocations or validate host pointer
 * provenance. Callers provide live handles, correctly sized/aligned argument
 * storage, and device buffers belonging to the stated device and allocation.
 * For asynchronous work, submission success is not completion: retain resources
 * and exclude conflicting access until the relevant stream/event has completed.
 * Null/shape/range error checks are not a sandbox for arbitrary pointer values.
 * Error-path completion and destruction obligations still apply to partial work.
 */

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define AUGUR_CUDA_CALL __cdecl
#else
#define AUGUR_CUDA_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define AUGUR_CUDA_ABI_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_DEVICE_NAME_CAPACITY UINT32_C(256)

typedef int32_t augur_cuda_status_t;

enum {
    AUGUR_CUDA_STATUS_OK = 0,
    AUGUR_CUDA_STATUS_INVALID_ARGUMENT = -1,
    AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE = -2,
    AUGUR_CUDA_STATUS_OUT_OF_MEMORY = -3,
    AUGUR_CUDA_STATUS_RUNTIME_FAILURE = -4,
    AUGUR_CUDA_STATUS_SUBMISSION_FAILURE = -5,
    AUGUR_CUDA_STATUS_COMPLETION_FAILURE = -6,
    AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH = -7,
    AUGUR_CUDA_STATUS_DIAGNOSTIC_FAULT_INJECTED = -8
};

enum {
    AUGUR_CUDA_M306_K14_PROGRESS_RECEIPT_D2H_SUBMITTED_V1 = 1,
    AUGUR_CUDA_M306_K14_PROGRESS_TRAP_SUBMITTED_V1 = 2,
    AUGUR_CUDA_M306_K14_PROGRESS_IMMEDIATE_PEEK_CLEAR_V1 = 4,
    AUGUR_CUDA_M306_K14_PROGRESS_COMPLETION_FAILURE_OBSERVED_V1 = 8
};

typedef struct augur_cuda_result_v1 {
    augur_cuda_status_t status;
    int32_t raw_cuda_status;
} augur_cuda_result_v1;

typedef struct augur_cuda_device_properties_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t ordinal;
    uint16_t compute_major;
    uint16_t compute_minor;
    uint64_t total_global_memory_bytes;
    char name[AUGUR_CUDA_DEVICE_NAME_CAPACITY];
} augur_cuda_device_properties_v1;

typedef struct augur_cuda_stream_handle augur_cuda_stream_handle;
typedef augur_cuda_stream_handle* augur_cuda_stream_t;
typedef struct augur_cuda_event_handle augur_cuda_event_handle;
typedef augur_cuda_event_handle* augur_cuda_event_t;
typedef struct augur_cuda_graph_exec_handle augur_cuda_graph_exec_handle;
typedef augur_cuda_graph_exec_handle* augur_cuda_graph_exec_t;

/*
 * CUDA device address.
 *
 * This value is not a pointer to a host-side object. It is an opaque
 * device-address token and must never be dereferenced by host code.
 */
typedef uint64_t augur_cuda_device_ptr_t;

uint32_t AUGUR_CUDA_CALL augur_cuda_abi_version(void);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_device_count(
    int32_t* out_count
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_get_device_properties(
    int32_t ordinal,
    augur_cuda_device_properties_v1* out_properties
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_get_device_uuid_v1(
    int32_t ordinal,
    uint8_t* out_uuid,
    uint32_t out_uuid_bytes
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_mem_get_info_v1(
    int32_t ordinal,
    uint64_t* out_free_bytes,
    uint64_t* out_total_bytes
);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvtx_m306_steady_token_begin_v1(void);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvtx_m306_steady_token_end_v1(void);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvtx_batch01_b4_row_main_begin_v1(uint32_t row_index);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvtx_batch01_b4_row_main_end_v1(uint32_t row_index);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_stream_create(
    int32_t ordinal,
    augur_cuda_stream_t* out_stream
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_stream_synchronize(
    augur_cuda_stream_t stream
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_stream_wait_event_v1(
    augur_cuda_stream_t stream,
    augur_cuda_event_t event
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_stream_destroy(
    augur_cuda_stream_t stream
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_graph_capture_begin_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_graph_capture_end_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_graph_exec_t* out_graph
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_graph_capture_abort_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_graph_launch_v1(
    int32_t ordinal,
    augur_cuda_graph_exec_t graph,
    augur_cuda_stream_t stream
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_graph_node_count_v1(
    augur_cuda_graph_exec_t graph,
    uint64_t* out_node_count
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_graph_destroy_v1(
    augur_cuda_graph_exec_t graph
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_event_create(
    int32_t ordinal,
    augur_cuda_event_t* out_event
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_event_record(
    augur_cuda_event_t event,
    augur_cuda_stream_t stream
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_event_query_complete_v1(
    augur_cuda_event_t event,
    uint8_t* out_complete
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_event_synchronize(
    augur_cuda_event_t event
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_event_elapsed_ms(
    augur_cuda_event_t started,
    augur_cuda_event_t stopped,
    float* out_milliseconds
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_event_destroy(
    augur_cuda_event_t event
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_device_alloc(
    int32_t ordinal,
    size_t len_bytes,
    augur_cuda_device_ptr_t* out_ptr
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_device_free(
    int32_t ordinal,
    augur_cuda_device_ptr_t ptr
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memset_zero_sync_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_device_ptr_t dst,
    size_t dst_len_bytes
);


// On success, all len_bytes are initialized to zero before out_ptr is
// published to the caller.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_host_alloc_pinned_v1(
    int32_t ordinal,
    size_t len_bytes,
    void** out_ptr
);

// Allocates page-locked host memory that is also addressable by kernels on
// `ordinal`. `out_device_ptr` is an opaque CUDA device-address token for the
// same bytes; it must not be passed to augur_cuda_device_free.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_host_alloc_mapped_v1(
    int32_t ordinal,
    size_t len_bytes,
    void** out_host_ptr,
    augur_cuda_device_ptr_t* out_device_ptr
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_host_free_pinned_v1(
    int32_t ordinal,
    void* ptr
);

// Fences every device consumer on `ordinal` before releasing a mapped host
// allocation. If the completion fence fails, the allocation remains owned by
// the caller and must not be reused or freed through another API.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_host_free_mapped_sync_v1(
    int32_t ordinal,
    void* ptr
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_h2d_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_device_ptr_t dst,
    size_t dst_offset_bytes,
    size_t dst_len_bytes,
    const void* src,
    size_t src_len_bytes,
    size_t copy_len_bytes
);

/*
 * Synchronous host/device copies. The implementation submits the copy to the
 * supplied stream and synchronizes that stream before returning, so host
 * staging memory may be released immediately after a successful call.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_h2d_sync(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_device_ptr_t dst,
    size_t dst_offset_bytes,
    const void* src,
    size_t src_len_bytes,
    size_t copy_len_bytes
);

/*
 * Extent-complete successors for safe callers. The historical unversioned
 * entrypoints above are retained for ABI compatibility only.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_h2d_sync_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_device_ptr_t dst,
    size_t dst_offset_bytes,
    size_t dst_len_bytes,
    const void* src,
    size_t src_len_bytes,
    size_t copy_len_bytes
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_d2h_sync(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    void* dst,
    size_t dst_len_bytes,
    augur_cuda_device_ptr_t src,
    size_t src_offset_bytes,
    size_t copy_len_bytes
);

/*
 * Asynchronous extent-complete D2H submit. The destination must be
 * page-locked host memory and remain live and unmodified until a completion
 * event recorded after this submission has been proven. This function never
 * synchronizes the stream.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_d2h_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    void* dst,
    size_t dst_len_bytes,
    augur_cuda_device_ptr_t src,
    size_t src_offset_bytes,
    size_t src_len_bytes,
    size_t copy_len_bytes
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_d2h_sync_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    void* dst,
    size_t dst_len_bytes,
    augur_cuda_device_ptr_t src,
    size_t src_offset_bytes,
    size_t src_len_bytes,
    size_t copy_len_bytes
);

augur_cuda_result_v1 AUGUR_CUDA_CALL
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
);

typedef struct augur_cuda_d2d_copy_entry_v1 {
    uint64_t source;
    uint64_t destination;
    uint64_t bytes;
} augur_cuda_d2d_copy_entry_v1;

// 1..128 independent, 16-byte-aligned ranges. No overlapping destinations or
// source/destination dependencies anywhere in the batch. Caller retains all
// allocations on this stream through completion, including submission errors.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_d2d_independent_batch_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_d2d_copy_entry_v1* entries, uint32_t count);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_d2d_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_device_ptr_t dst,
    size_t dst_offset_bytes,
    size_t dst_len_bytes,
    augur_cuda_device_ptr_t src,
    size_t src_offset_bytes,
    size_t src_len_bytes,
    size_t copy_len_bytes
);

// In-place bitwise row-left-shift. Writes only retained_rows, leaves the tail
// untouched. Zero shift/retention is a validated no-op. Allocation/stream must
// remain exclusively owned and live until stream completion.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_shift_rows_u32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_device_ptr_t data,
    size_t len_bytes, size_t row_bytes, size_t retained_rows, size_t shift_rows
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_d2d_2d_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_device_ptr_t dst,
    size_t dst_offset_bytes,
    size_t dst_len_bytes,
    size_t dst_pitch_bytes,
    augur_cuda_device_ptr_t src,
    size_t src_offset_bytes,
    size_t src_len_bytes,
    size_t src_pitch_bytes,
    size_t width_bytes,
    size_t height
);

/*
 * Fixed balanced-width6 full-slot assembly.
 *
 * This is a prevalidated asynchronous submit ABI. The six destination slots
 * and all 48 W1/W2/W3 source group views must be live, same-device, immutable
 * (sources), and exclusively owned (destinations) through stream completion.
 * The two kernels preserve the ordinary full-slot byte layout while reducing
 * 144 runtime memcpy submissions to two kernel submissions.
 */
#define AUGUR_CUDA_P4_WIDTH6_FULL_SLOT_ASSEMBLY_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_P4_WIDTH6_FULL_SLOT_RANKS_V1 UINT32_C(6)
#define AUGUR_CUDA_P4_WIDTH6_FULL_SLOT_GROUPS_V1 UINT32_C(8)
#define AUGUR_CUDA_P4_WIDTH6_FULL_SLOT_SOURCE_COUNT_V1 UINT32_C(48)
#define AUGUR_CUDA_P4_WIDTH6_FULL_SLOT_BYTES_V1 UINT64_C(16777216)
#define AUGUR_CUDA_P4_WIDTH6_GROUP_W1_BYTES_V1 UINT64_C(655360)
#define AUGUR_CUDA_P4_WIDTH6_GROUP_W2_BYTES_V1 UINT64_C(786432)
#define AUGUR_CUDA_P4_WIDTH6_GROUP_W3_BYTES_V1 UINT64_C(655360)

typedef struct augur_cuda_p4_width6_full_slot_assembly_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t ranks;
    uint32_t groups;
    augur_cuda_device_ptr_t destinations[AUGUR_CUDA_P4_WIDTH6_FULL_SLOT_RANKS_V1];
    augur_cuda_device_ptr_t w1_sources[AUGUR_CUDA_P4_WIDTH6_FULL_SLOT_SOURCE_COUNT_V1];
    augur_cuda_device_ptr_t w2_sources[AUGUR_CUDA_P4_WIDTH6_FULL_SLOT_SOURCE_COUNT_V1];
    augur_cuda_device_ptr_t w3_sources[AUGUR_CUDA_P4_WIDTH6_FULL_SLOT_SOURCE_COUNT_V1];
    uint64_t reserved[4];
} augur_cuda_p4_width6_full_slot_assembly_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_p4_width6_full_slot_assembly_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_p4_width6_full_slot_assembly_args_v1* args
);

/*
 * Exact width6 assembly with placement-resident ranks omitted.  Bits 0..5
 * select the destination/source rows that are present in `args`; all inactive
 * rows must remain zero.  The two launches retain the V1 destination layout.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_p4_width6_full_slot_assembly_masked_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_p4_width6_full_slot_assembly_args_v1* args,
    uint32_t active_rank_mask
);

/*
 * Mixed packed-group / full-expert assembly. `reserved[0]` is a 48-bit mask:
 * a set bit selects the full-expert Q5 row pitch (1536 bytes) for that source;
 * a clear bit retains the packed-group pitch (192 bytes). `reserved[1..3]`
 * remain zero. W1/W3 source addresses already point at the selected group.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_p4_width6_full_slot_assembly_mixed_masked_prevalidated_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_p4_width6_full_slot_assembly_args_v1* args,
    uint32_t active_rank_mask
);

/*
 * Q4K 脳 f32 decode launch arguments (versioned POD).
 *
 * stream context (ordinal + stream handle) is passed separately
 * to the launch function and does NOT appear in this struct.
 */
#define AUGUR_CUDA_Q4K_F32_DECODE_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_Q4K_BLOCK_DECODED_LEN_V1   UINT32_C(256)
#define AUGUR_CUDA_Q4K_BLOCK_BYTES_V1         UINT32_C(160)
#define AUGUR_CUDA_Q4K_BLOCK_ALIGNMENT_V1     UINT32_C(32)

typedef struct augur_cuda_q4k_f32_decode_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;

    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;

    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;

    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;

    uint32_t rows;
    uint32_t cols;
    uint32_t blocks_per_row;
    uint32_t weight_block_stride_bytes;
} augur_cuda_q4k_f32_decode_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q4k_f32_decode_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q4k_f32_decode_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q4k_f32_decode_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q4k_f32_decode_args_v1* args
);

#define AUGUR_CUDA_Q4K_F32_PREFILL_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_q4k_f32_prefill_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;

    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;

    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;

    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;

    uint32_t m;
    uint32_t rows;
    uint32_t cols;
    uint32_t blocks_per_row;
    uint32_t weight_block_stride_bytes;
} augur_cuda_q4k_f32_prefill_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q4k_f32_prefill_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q4k_f32_prefill_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q4k_f32_prefill_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q4k_f32_prefill_args_v1* args
);

#define AUGUR_CUDA_Q5K_F32_DECODE_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_Q5K_BLOCK_DECODED_LEN_V1       UINT32_C(256)
#define AUGUR_CUDA_Q5K_BLOCK_BYTES_V1             UINT32_C(192)
#define AUGUR_CUDA_Q5K_BLOCK_ALIGNMENT_V1         UINT32_C(32)

typedef struct augur_cuda_q5k_f32_decode_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t rows;
    uint32_t cols;
    uint32_t blocks_per_row;
    uint32_t weight_block_stride_bytes;
} augur_cuda_q5k_f32_decode_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q5k_f32_decode_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q5k_f32_decode_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q5k_f32_decode_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q5k_f32_decode_args_v1* args
);

#if defined(AUGUR_CUDA_P4_EXPERT_MICROSLICE_GPU_REPACK)
/* Isolated P4 compact-source -> existing CUDA-hot expert-slot repack.
 * Source and destination are distinct, exact owner-backed device views. */
#define AUGUR_CUDA_Q4K_COMPACT_TO_HOT_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_Q4K_COMPACT_BLOCK_BYTES_V1 UINT32_C(144)
#define AUGUR_CUDA_Q4K_HOT_BLOCK_BYTES_V1 UINT32_C(160)
typedef struct augur_cuda_q4k_compact_to_hot_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t compact;
    uint64_t compact_len_bytes;
    augur_cuda_device_ptr_t hot;
    uint64_t hot_len_bytes;
    uint32_t rows;
    uint32_t cols;
    uint32_t compact_block_stride_bytes;
    uint32_t hot_block_stride_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
} augur_cuda_q4k_compact_to_hot_args_v1;

#define AUGUR_CUDA_Q5K_COMPACT_TO_HOT_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_Q5K_COMPACT_BLOCK_BYTES_V1 UINT32_C(176)
#define AUGUR_CUDA_Q5K_HOT_BLOCK_BYTES_V1 UINT32_C(192)
typedef struct augur_cuda_q5k_compact_to_hot_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t compact;
    uint64_t compact_len_bytes;
    augur_cuda_device_ptr_t hot;
    uint64_t hot_len_bytes;
    uint32_t rows;
    uint32_t cols;
    uint32_t compact_block_stride_bytes;
    uint32_t hot_block_stride_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
} augur_cuda_q5k_compact_to_hot_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q4k_compact_to_hot_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q4k_compact_to_hot_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q5k_compact_to_hot_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q5k_compact_to_hot_args_v1* args
);
#endif

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q5k_f32_decode_grid32_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q5k_f32_decode_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q5k_f32_decode_grid32_submit_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q5k_f32_decode_args_v1* args
);

#if defined(AUGUR_CUDA_M306_K22_Q5_FP32_KAHAN)
#define AUGUR_CUDA_M306_K22_Q5_ROWS_V1 UINT32_C(2048)
#define AUGUR_CUDA_M306_K22_Q5_COLS_V1 UINT32_C(4096)

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k22_q5k_f32_decode_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q5k_f32_decode_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k22_q5k_f32_decode_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q5k_f32_decode_args_v1* args
);
#endif

#if defined(AUGUR_CUDA_M306_K15_Q5_PAIRED_PROJECTION)
#define AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_ROWS_V1        UINT32_C(2048)
#define AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_COLS_V1        UINT32_C(4096)
#define AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_BLOCKS_PER_ROW_V1 UINT32_C(16)
#define AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_GRID_BLOCKS_V1 UINT32_C(32)

typedef struct augur_cuda_q5k_f32_paired_projection_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight_w1;
    uint64_t weight_w1_len_bytes;
    augur_cuda_device_ptr_t weight_w3;
    uint64_t weight_w3_len_bytes;
    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output_w1;
    uint64_t output_w1_len_bytes;
    augur_cuda_device_ptr_t output_w3;
    uint64_t output_w3_len_bytes;
    uint32_t rows;
    uint32_t cols;
    uint32_t blocks_per_row;
    uint32_t weight_block_stride_bytes;
} augur_cuda_q5k_f32_paired_projection_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q5k_f32_paired_projection_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q5k_f32_paired_projection_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q5k_f32_paired_projection_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q5k_f32_paired_projection_args_v1* args
);

#define AUGUR_CUDA_BATCH01_B4_Q5K_F32_PAIRED_PROJECTION_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_BATCH01_B4_Q5K_F32_PAIRED_PROJECTION_BATCH_V1 UINT32_C(4)

typedef struct augur_cuda_batch01_b4_q5k_f32_paired_projection_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight_w1;
    uint64_t weight_w1_len_bytes;
    augur_cuda_device_ptr_t weight_w3;
    uint64_t weight_w3_len_bytes;
    augur_cuda_device_ptr_t input[4];
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output_w1[4];
    augur_cuda_device_ptr_t output_w3[4];
    uint64_t output_len_bytes;
    uint32_t batch;
    uint32_t rows;
    uint32_t cols;
    uint32_t blocks_per_row;
    uint32_t weight_block_stride_bytes;
} augur_cuda_batch01_b4_q5k_f32_paired_projection_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b4_q5k_f32_paired_projection_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q5k_f32_paired_projection_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b4_q5k_f32_paired_projection_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q5k_f32_paired_projection_args_v1* args
);

#if defined(AUGUR_CUDA_BATCH01_B8_Q5_ROW_TILE)
#define AUGUR_CUDA_BATCH01_B8_Q5K_ROW_TILE_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_BATCH01_B8_Q5K_ROW_TILE_BATCH_V1 UINT32_C(8)
#define AUGUR_CUDA_BATCH01_B8_Q5K_ROW_TILE_THREADS_V1 UINT32_C(128)
#define AUGUR_CUDA_BATCH01_B8_Q5K_ROW_TILE_GRID_BLOCKS_V1 UINT32_C(256)

typedef struct augur_cuda_batch01_b8_q5k_row_tile_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight_w1;
    uint64_t weight_w1_len_bytes;
    augur_cuda_device_ptr_t weight_w3;
    uint64_t weight_w3_len_bytes;
    augur_cuda_device_ptr_t input[8];
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output_w1[8];
    augur_cuda_device_ptr_t output_w3[8];
    uint64_t output_len_bytes;
    uint32_t batch;
    uint32_t rows;
    uint32_t cols;
    uint32_t blocks_per_row;
    uint32_t weight_block_stride_bytes;
} augur_cuda_batch01_b8_q5k_row_tile_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b8_q5k_row_tile_validate_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b8_q5k_row_tile_submit_prevalidated_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b8_q5k_row_tile_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b8_q5k_row_tile_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b8_q5k_row_tile_submit_prevalidated_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b8_q5k_row_tile_submit_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b8_q5k_row_tile_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args
);
#endif
#endif

#define AUGUR_CUDA_Q6K_F32_DECODE_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_Q6K_BLOCK_DECODED_LEN_V1       UINT32_C(256)
#define AUGUR_CUDA_Q6K_BLOCK_BYTES_V1             UINT32_C(224)
#define AUGUR_CUDA_Q6K_BLOCK_ALIGNMENT_V1         UINT32_C(32)

typedef struct augur_cuda_q6k_f32_decode_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t rows;
    uint32_t cols;
    uint32_t blocks_per_row;
    uint32_t weight_block_stride_bytes;
} augur_cuda_q6k_f32_decode_args_v1;

#if defined(AUGUR_CUDA_DSPARK_FAST_Q6_ROW_REUSE)
#define AUGUR_CUDA_DSPARK_FAST_Q6K_ROW_REUSE_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_DSPARK_FAST_Q6K_ROW_REUSE_MAX_ACTIVE_ROWS_V1 UINT32_C(4)

typedef struct augur_cuda_dspark_fast_q6k_row_reuse_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t active_rows;
    uint32_t rows;
    uint32_t cols;
    uint32_t input_row_stride_elements;
    uint32_t output_row_stride_elements;
    uint32_t blocks_per_row;
    uint32_t weight_block_stride_bytes;
    uint32_t reserved[11];
} augur_cuda_dspark_fast_q6k_row_reuse_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dspark_fast_q6k_row_reuse_validate_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dspark_fast_q6k_row_reuse_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dspark_fast_q6k_row_reuse_submit_prevalidated_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dspark_fast_q6k_row_reuse_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dspark_fast_q6k_row_reuse_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dspark_fast_q6k_row_reuse_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dspark_fast_q6k_row_reuse_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dspark_fast_q6k_row_reuse_args_v1* args
);
#endif

#if defined(AUGUR_CUDA_BATCH01_B4_Q6_ROW_REUSE)
#define AUGUR_CUDA_BATCH01_B4_Q6K_BOUNDED_FP64_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_BATCH01_B4_Q6K_BOUNDED_FP64_BATCH_V1 UINT32_C(4)
#define AUGUR_CUDA_BATCH01_B4_Q6K_BOUNDED_FP64_ROWS_V1 UINT32_C(4096)
#define AUGUR_CUDA_BATCH01_B4_Q6K_BOUNDED_FP64_COLS_V1 UINT32_C(2048)
#define AUGUR_CUDA_BATCH01_B4_Q6K_BOUNDED_FP64_BLOCKS_PER_ROW_V1 UINT32_C(8)

typedef struct augur_cuda_batch01_b4_q6k_bounded_fp64_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input[4];
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output[4];
    uint64_t output_len_bytes;
    uint32_t batch;
    uint32_t rows;
    uint32_t cols;
    uint32_t blocks_per_row;
    uint32_t weight_block_stride_bytes;
} augur_cuda_batch01_b4_q6k_bounded_fp64_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b4_q6k_bounded_fp64_validate_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6k_bounded_fp64_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b4_q6k_bounded_fp64_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6k_bounded_fp64_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b4_q6k_bounded_fp64_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6k_bounded_fp64_args_v1* args
);

#if defined(AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_ROW_REUSE)
#define AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_BOUNDED_FP64_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_BOUNDED_FP64_BATCH_V1 UINT32_C(4)
#define AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_BOUNDED_FP64_GROUPS_V1 UINT32_C(8)
#define AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_BOUNDED_FP64_ROWS_PER_GROUP_V1 UINT32_C(1024)
#define AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_BOUNDED_FP64_COLS_PER_GROUP_V1 UINT32_C(4096)
#define AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_BOUNDED_FP64_BLOCKS_PER_OUTPUT_ROW_V1 UINT32_C(16)

#define AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_CONTROL_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t groups;
    uint32_t rows_per_group;
    uint32_t cols_per_group;
    uint32_t blocks_per_output_row;
    uint32_t weight_block_stride_bytes;
} augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q6_block_diagonal_control_validate_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q6_block_diagonal_control_submit_prevalidated_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1* args
);

typedef struct augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input[4];
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output[4];
    uint64_t output_len_bytes;
    uint32_t batch;
    uint32_t groups;
    uint32_t rows_per_group;
    uint32_t cols_per_group;
    uint32_t blocks_per_output_row;
    uint32_t weight_block_stride_bytes;
} augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_validate_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_submit_prevalidated_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1* args
);
#endif
#endif

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q6k_f32_decode_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q6k_f32_decode_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q6k_f32_decode_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q6k_f32_decode_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m120_q4k_exact_f32_decode_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q4k_f32_decode_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m120_q5k_exact_f32_decode_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q5k_f32_decode_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m120_q6k_exact_f32_decode_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q6k_f32_decode_args_v1* args
);

#define AUGUR_CUDA_M121_Q4K_DP4A_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m121_q4k_dp4a_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t q8_input;
    uint64_t q8_input_len_bytes;
    augur_cuda_device_ptr_t q8_scales;
    uint64_t q8_scales_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t rows;
    uint32_t cols;
    uint32_t blocks_per_row;
    uint32_t group_size;
} augur_cuda_m121_q4k_dp4a_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m121_q4k_dp4a_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m121_q4k_dp4a_args_v1* args
);

#define AUGUR_CUDA_ADD_F32_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_add_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t a;
    uint64_t a_len_bytes;
    augur_cuda_device_ptr_t b;
    uint64_t b_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t elements;
    uint32_t reserved;
} augur_cuda_add_f32_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_add_f32_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_add_f32_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_add_f32_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_add_f32_args_v1* args
);

#define AUGUR_CUDA_GQA_REPEAT_F32_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_gqa_repeat_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t source_elements;
    uint32_t head_dim;
    uint32_t repeats;
    uint32_t reserved;
} augur_cuda_gqa_repeat_f32_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_gqa_repeat_f32_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_gqa_repeat_f32_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_gqa_repeat_f32_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_gqa_repeat_f32_args_v1* args
);

#define AUGUR_CUDA_SWIGLU_F32_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_swiglu_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t gate;
    uint64_t gate_len_bytes;
    augur_cuda_device_ptr_t up;
    uint64_t up_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t elements;
    uint32_t limit_bits;
} augur_cuda_swiglu_f32_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_swiglu_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_swiglu_f32_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_swiglu_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_swiglu_f32_args_v1* args
);

/*
 * Unclamped FP32 SiLU(gate) * up.  This is deliberately a separate ABI from
 * augur_cuda_swiglu_f32_args_v1: the older entry point applies an explicit
 * one-sided/symmetric clamp and is not a valid Qwen activation contract.
 */
#define AUGUR_CUDA_SILU_MUL_F32_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_silu_mul_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t gate;
    uint64_t gate_len_bytes;
    augur_cuda_device_ptr_t up;
    uint64_t up_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t elements;
    uint32_t reserved;
} augur_cuda_silu_mul_f32_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_silu_mul_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_f32_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_silu_mul_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_f32_args_v1* args
);

/* One-token Qwen35 DFlash grouped convolution boundary.  The previous
 * convolution tap is intentionally zero in this ABI; the caller supplies
 * both [side, tap, group] dynamic coefficients and [side, tap, hidden] base
 * coefficients so the geometry remains explicit and auditable. */
#define AUGUR_CUDA_DFLASH_GROUPED_CONV_F32_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_dflash_grouped_conv_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t hidden_f32;
    uint64_t hidden_len_bytes;
    augur_cuda_device_ptr_t dynamic_f32;
    uint64_t dynamic_len_bytes;
    augur_cuda_device_ptr_t base_f32;
    uint64_t base_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    uint32_t hidden_elements;
    uint32_t groups;
    uint32_t group_size;
    uint32_t kernel_size;
    uint32_t side;
    uint32_t reserved;
} augur_cuda_dflash_grouped_conv_f32_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_grouped_conv_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_grouped_conv_f32_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_grouped_conv_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_grouped_conv_f32_args_v1* args
);

/* Rolling one-token variant with an explicit previous hidden row. */
#define AUGUR_CUDA_DFLASH_GROUPED_CONV_HISTORY_F32_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_dflash_grouped_conv_history_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t hidden_f32;
    uint64_t hidden_len_bytes;
    augur_cuda_device_ptr_t history_f32;
    uint64_t history_len_bytes;
    augur_cuda_device_ptr_t dynamic_f32;
    uint64_t dynamic_len_bytes;
    augur_cuda_device_ptr_t base_f32;
    uint64_t base_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    uint32_t hidden_elements;
    uint32_t groups;
    uint32_t group_size;
    uint32_t kernel_size;
    uint32_t side;
    uint32_t reserved;
} augur_cuda_dflash_grouped_conv_history_f32_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_grouped_conv_history_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_grouped_conv_history_f32_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_grouped_conv_history_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_grouped_conv_history_f32_args_v1* args
);

/* Contiguous token-major rolling variant.  Row zero reads the explicit zero
 * history; each later row reads the preceding row from hidden_f32.  This is
 * the exact 8-row DFlash block boundary, with no arithmetic or precision
 * change relative to eight history_f32 launches. */
#define AUGUR_CUDA_DFLASH_GROUPED_CONV_HISTORY_ROWS_F32_ARGS_VERSION_V2 UINT32_C(2)

typedef struct augur_cuda_dflash_grouped_conv_history_rows_f32_args_v2 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t hidden_f32;
    uint64_t hidden_len_bytes;
    augur_cuda_device_ptr_t zero_f32;
    uint64_t zero_len_bytes;
    augur_cuda_device_ptr_t dynamic_f32;
    uint64_t dynamic_len_bytes;
    augur_cuda_device_ptr_t base_f32;
    uint64_t base_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    uint32_t rows;
    uint32_t hidden_elements;
    uint32_t groups;
    uint32_t group_size;
    uint32_t kernel_size;
    uint32_t side;
    uint32_t reserved;
} augur_cuda_dflash_grouped_conv_history_rows_f32_args_v2;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_grouped_conv_history_rows_f32_submit_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_grouped_conv_history_rows_f32_args_v2* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_grouped_conv_history_rows_f32_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_grouped_conv_history_rows_f32_args_v2* args
);

/* Fused unclamped FP32 SiLU(gate) * up followed by Q8_1 activation output.
 * The arithmetic and quantization contract matches the two separate entry
 * points, but the intermediate FP32 activation is not materialized. */
#define AUGUR_CUDA_SILU_MUL_Q8_1_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_silu_mul_q8_1_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t gate;
    uint64_t gate_len_bytes;
    augur_cuda_device_ptr_t up;
    uint64_t up_len_bytes;
    augur_cuda_device_ptr_t output_q8_1;
    uint64_t output_len_bytes;
    uint32_t elements;
    uint32_t reserved;
} augur_cuda_silu_mul_q8_1_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_silu_mul_q8_1_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_q8_1_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_silu_mul_q8_1_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_q8_1_args_v1* args
);

/*
 * Qwen3.8 layer-0 Gated DeltaNet recurrent step.
 *
 * This ABI is opt-in and intentionally fixed to the audited Qwen35 geometry:
 * 16 key heads, 48 value heads, head dimension 128, and a four-tap causal
 * convolution.  All buffers are device-resident FP32.  The submit entrypoint
 * enqueues causal convolution/SiLU, the in-place DeltaNet state update, and
 * the gated RMSNorm output without a host synchronization.
 */
#define AUGUR_CUDA_QWEN35_LINEAR_STEP_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_QWEN35_LINEAR_STEP_KEY_HEADS_V1 UINT32_C(16)
#define AUGUR_CUDA_QWEN35_LINEAR_STEP_VALUE_HEADS_V1 UINT32_C(48)
#define AUGUR_CUDA_QWEN35_LINEAR_STEP_HEAD_DIM_V1 UINT32_C(128)
#define AUGUR_CUDA_QWEN35_LINEAR_STEP_KERNEL_SIZE_V1 UINT32_C(4)
#define AUGUR_CUDA_QWEN35_LINEAR_STEP_QKV_ELEMENTS_V1 UINT32_C(10240)
#define AUGUR_CUDA_QWEN35_LINEAR_STEP_OUTPUT_ELEMENTS_V1 UINT32_C(6144)
#define AUGUR_CUDA_QWEN35_LINEAR_STEP_STATE_ELEMENTS_V1 UINT32_C(786432)
#define AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_HISTORY_ELEMENTS_V1 UINT32_C(30720)
#define AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_WEIGHT_ELEMENTS_V1 UINT32_C(40960)

typedef struct augur_cuda_qwen35_linear_step_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t qkv_f32;
    uint64_t qkv_len_bytes;
    augur_cuda_device_ptr_t z_f32;
    uint64_t z_len_bytes;
    augur_cuda_device_ptr_t conv_weights_f32;
    uint64_t conv_weights_len_bytes;
    augur_cuda_device_ptr_t conv_history_f32;
    uint64_t conv_history_len_bytes;
    augur_cuda_device_ptr_t decay_gate_f32;
    uint64_t decay_gate_len_bytes;
    augur_cuda_device_ptr_t beta_f32;
    uint64_t beta_len_bytes;
    augur_cuda_device_ptr_t state_f32;
    uint64_t state_len_bytes;
    augur_cuda_device_ptr_t norm_weight_f32;
    uint64_t norm_weight_len_bytes;
    augur_cuda_device_ptr_t activated_f32;
    uint64_t activated_len_bytes;
    augur_cuda_device_ptr_t recurrent_f32;
    uint64_t recurrent_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    uint32_t epsilon_bits;
    uint32_t reserved;
} augur_cuda_qwen35_linear_step_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_linear_step_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_linear_step_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_linear_step_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_linear_step_args_v1* args
);

/* Explicit safetensors source head order and additive Q/K L2 epsilon.
 * Uses the same checked extents, not the GGUF V1 numerical interpretation. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_checkpoint_linear_step_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_linear_step_args_v1* args
);

/*
 * Qwen3.8 linear-attention projection postprocess.
 *
 * Converts the 48 raw alpha/beta projection values into the exact recurrent
 * decay and beta gates consumed by the linear-step ABI.  All views are fixed
 * width FP32 device buffers; the submit entrypoint is asynchronous and is
 * intended to be ordered before augur_cuda_qwen35_linear_step_submit_v1 on the
 * same stream.
 */
#define AUGUR_CUDA_QWEN35_LINEAR_PROJECTION_POSTPROCESS_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_QWEN35_LINEAR_PROJECTION_POSTPROCESS_ELEMENTS_V1 UINT32_C(48)

typedef struct augur_cuda_qwen35_linear_projection_postprocess_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t alpha_raw_f32;
    uint64_t alpha_raw_len_bytes;
    augur_cuda_device_ptr_t beta_raw_f32;
    uint64_t beta_raw_len_bytes;
    augur_cuda_device_ptr_t dt_bias_f32;
    uint64_t dt_bias_len_bytes;
    augur_cuda_device_ptr_t ssm_a_f32;
    uint64_t ssm_a_len_bytes;
    augur_cuda_device_ptr_t decay_gate_f32;
    uint64_t decay_gate_len_bytes;
    augur_cuda_device_ptr_t beta_out_f32;
    uint64_t beta_out_len_bytes;
    uint32_t elements;
    uint32_t reserved;
} augur_cuda_qwen35_linear_projection_postprocess_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_linear_projection_postprocess_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_linear_projection_postprocess_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_linear_projection_postprocess_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_linear_projection_postprocess_args_v1* args
);

/*
 * Qwen3.8 full-attention Q/K per-head RMSNorm plus RoPE boundary.
 *
 * One 256-thread CTA owns one Q or K head. The reduction tree and arithmetic
 * order match the scalar-head RMSNorm/RoPE composition while all 28 heads are
 * submitted as one kernel.
 */
#define AUGUR_CUDA_QWEN35_FUSED_HEAD_NORM_ROPE_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_qwen35_fused_head_norm_rope_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t q_input_f32;
    uint64_t q_input_len_bytes;
    augur_cuda_device_ptr_t q_weight_f32;
    uint64_t q_weight_len_bytes;
    augur_cuda_device_ptr_t q_normed_f32;
    uint64_t q_normed_len_bytes;
    augur_cuda_device_ptr_t q_output_f32;
    uint64_t q_output_len_bytes;
    augur_cuda_device_ptr_t k_input_f32;
    uint64_t k_input_len_bytes;
    augur_cuda_device_ptr_t k_weight_f32;
    uint64_t k_weight_len_bytes;
    augur_cuda_device_ptr_t k_normed_f32;
    uint64_t k_normed_len_bytes;
    augur_cuda_device_ptr_t k_output_f32;
    uint64_t k_output_len_bytes;
    augur_cuda_device_ptr_t rope_cos_f32;
    uint64_t rope_cos_len_bytes;
    augur_cuda_device_ptr_t rope_sin_f32;
    uint64_t rope_sin_len_bytes;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t epsilon_bits;
    uint32_t reserved;
} augur_cuda_qwen35_fused_head_norm_rope_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_fused_head_norm_rope_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_fused_head_norm_rope_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_fused_head_norm_rope_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_fused_head_norm_rope_args_v1* args
);

/* Bounded input-only batch. Slots: Q+gate, Q weight, Q output, K input,
 * K weight, K output, cos, sin, gate output, V input, V output. All F32. */
typedef struct augur_cuda_qwen35_batched_head_norm_rope_args_v1 {
    uint32_t abi_version, struct_size, rows, epsilon_bits, reserved0, reserved1;
    uint64_t addresses[11];
    uint64_t lengths[11];
} augur_cuda_qwen35_batched_head_norm_rope_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_batched_head_norm_rope_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_batched_head_norm_rope_args_v1* args
);

/*
 * Qwen3.8 layer-3 grouped-query full-attention boundary.
 *
 * The caller supplies already-normalized/rotated Q and K plus the raw V and
 * output gate vectors.  The asynchronous entrypoint appends one KV row,
 * performs causal GQA softmax, and writes both the attention and gated
 * attention vectors.  This is intentionally a component ABI: projections
 * and RoPE preparation may be performed by a separate, audited boundary.
 */
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1 UINT32_C(24)
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1 UINT32_C(4)
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1 UINT32_C(256)
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 UINT32_C(64)
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1 UINT32_C(6144)
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1 UINT32_C(1024)
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_MAX_CAPACITY_V1 UINT32_C(4096)

typedef struct augur_cuda_qwen35_full_attention_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t query_f32;
    uint64_t query_len_bytes;
    augur_cuda_device_ptr_t gate_f32;
    uint64_t gate_len_bytes;
    augur_cuda_device_ptr_t key_f32;
    uint64_t key_len_bytes;
    augur_cuda_device_ptr_t value_f32;
    uint64_t value_len_bytes;
    augur_cuda_device_ptr_t key_cache_f32;
    uint64_t key_cache_len_bytes;
    augur_cuda_device_ptr_t value_cache_f32;
    uint64_t value_cache_len_bytes;
    augur_cuda_device_ptr_t attention_f32;
    uint64_t attention_len_bytes;
    augur_cuda_device_ptr_t gated_f32;
    uint64_t gated_len_bytes;
    uint32_t capacity;
    uint32_t position;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t reserved;
} augur_cuda_qwen35_full_attention_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_full_attention_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_full_attention_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_full_attention_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_full_attention_args_v1* args
);

/*
 * Opt-in grouped-GQA decode boundary.  V2 preserves the V1 cache and output
 * layouts while replacing the per-query-head scan with 64 KV-head splits
 * and a deterministic online-softmax merge.  The workspace contains the
 * split partials followed by a Q8_1 encoding of the gated output, allowing
 * the next Q8_0 x Q8_1 projection to consume it without another launch.  The
 * caller owns the exact-size workspace for the entire submission lifetime;
 * the operation allocates no memory and performs no host synchronization.
 */
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_ARGS_VERSION_V2 UINT32_C(2)
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_SPLITS_V2 UINT32_C(72)
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_STAGE1_THREADS_V2 UINT32_C(256)
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_TILE_POSITIONS_V2 UINT32_C(8)
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_MERGE_SEGMENTS_V2 UINT32_C(2)
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_MERGE_THREADS_V2 UINT32_C(128)
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_PARTIAL_STRIDE_V2 UINT32_C(258)
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2 UINT32_C(445824)
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2 UINT32_C(6912)
#define AUGUR_CUDA_QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2 UINT32_C(447552)

typedef struct augur_cuda_qwen35_full_attention_args_v2 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t query_f32;
    uint64_t query_len_bytes;
    augur_cuda_device_ptr_t gate_f32;
    uint64_t gate_len_bytes;
    augur_cuda_device_ptr_t key_f32;
    uint64_t key_len_bytes;
    augur_cuda_device_ptr_t value_f32;
    uint64_t value_len_bytes;
    augur_cuda_device_ptr_t key_cache_f32;
    uint64_t key_cache_len_bytes;
    augur_cuda_device_ptr_t value_cache_f32;
    uint64_t value_cache_len_bytes;
    augur_cuda_device_ptr_t attention_f32;
    uint64_t attention_len_bytes;
    augur_cuda_device_ptr_t gated_f32;
    uint64_t gated_len_bytes;
    augur_cuda_device_ptr_t workspace_f32;
    uint64_t workspace_len_bytes;
    uint32_t capacity;
    uint32_t position;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t split_count;
    uint32_t reserved;
} augur_cuda_qwen35_full_attention_args_v2;

#if defined(AUGUR_CUDA_QWEN35_GROUPED_GQA_ONLINE_SOFTMAX)
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_full_attention_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_full_attention_args_v2* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_full_attention_submit_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_full_attention_args_v2* args
);
#endif

/*
 * Opt-in Q8_0 KV-cache boundary for the exact same grouped-GQA computation.
 * The current K/V row is quantized in-kernel with llama.cpp-compatible
 * 32-value Q8_0 blocks. Historical rows stay packed in device memory and are
 * dequantized only into the request-private shared-memory tile. V3 does not
 * grant host-tier, session, decode, CLI, semantic, or throughput authority.
 */
#define AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION_ARGS_VERSION_V3 UINT32_C(3)
#define AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY_V3 UINT32_C(262144)
#define AUGUR_CUDA_QWEN35_KV8_BLOCK_ELEMENTS_V3 UINT32_C(32)
#define AUGUR_CUDA_QWEN35_KV8_BLOCK_BYTES_V3 UINT32_C(34)
#define AUGUR_CUDA_QWEN35_KV8_BLOCKS_PER_HEAD_V3 UINT32_C(8)
#define AUGUR_CUDA_QWEN35_KV8_HEAD_ROW_BYTES_V3 UINT32_C(272)
#define AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3 UINT32_C(1088)

typedef struct augur_cuda_qwen35_kv8_full_attention_args_v3 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t query_f32;
    uint64_t query_len_bytes;
    augur_cuda_device_ptr_t gate_f32;
    uint64_t gate_len_bytes;
    augur_cuda_device_ptr_t key_f32;
    uint64_t key_len_bytes;
    augur_cuda_device_ptr_t value_f32;
    uint64_t value_len_bytes;
    augur_cuda_device_ptr_t key_cache_q8_0;
    uint64_t key_cache_len_bytes;
    augur_cuda_device_ptr_t value_cache_q8_0;
    uint64_t value_cache_len_bytes;
    augur_cuda_device_ptr_t attention_f32;
    uint64_t attention_len_bytes;
    augur_cuda_device_ptr_t gated_f32;
    uint64_t gated_len_bytes;
    augur_cuda_device_ptr_t workspace_f32;
    uint64_t workspace_len_bytes;
    uint32_t capacity;
    uint32_t position;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t split_count;
    uint32_t reserved;
} augur_cuda_qwen35_kv8_full_attention_args_v3;

#if defined(AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION)
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_full_attention_v3(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_full_attention_args_v3* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_full_attention_submit_v3(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_full_attention_args_v3* args
);
#endif

/*
 * Opt-in contiguous multi-token Q8 KV attention boundary.  V4 batches up to
 * eight causal rows sharing historical KV tiles while preserving V3 per-row
 * split boundaries, causal masks, reduction order and arithmetic.
 * All non-cache tensors are contiguous row-major arrays; each row has the
 * exact V3 extent.  This ABI is candidate-only and never changes the V3 path.
 */
#define AUGUR_CUDA_QWEN35_KV8_BATCHED_ATTENTION_ARGS_VERSION_V4 UINT32_C(4)
#define AUGUR_CUDA_QWEN35_KV8_BATCHED_ATTENTION_MAX_TOKENS_V4 UINT32_C(8)

typedef struct augur_cuda_qwen35_kv8_batched_attention_args_v4 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t query_f32;
    uint64_t query_len_bytes;
    augur_cuda_device_ptr_t gate_f32;
    uint64_t gate_len_bytes;
    augur_cuda_device_ptr_t key_f32;
    uint64_t key_len_bytes;
    augur_cuda_device_ptr_t value_f32;
    uint64_t value_len_bytes;
    augur_cuda_device_ptr_t key_cache_q8_0;
    uint64_t key_cache_len_bytes;
    augur_cuda_device_ptr_t value_cache_q8_0;
    uint64_t value_cache_len_bytes;
    augur_cuda_device_ptr_t attention_f32;
    uint64_t attention_len_bytes;
    augur_cuda_device_ptr_t gated_f32;
    uint64_t gated_len_bytes;
    augur_cuda_device_ptr_t workspace_f32;
    uint64_t workspace_len_bytes;
    uint32_t capacity;
    uint32_t position_base;
    uint32_t tokens;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t split_count;
    uint32_t reserved;
} augur_cuda_qwen35_kv8_batched_attention_args_v4;

/* Input-only canonical prefill. The address layout is shared with V4, but
 * this operation has its own version and permits at most 128 causal rows.
 * Quantization, 72 split intervals and the fused gated-Q8 merge remain V4;
 * only score production is replaceable. It is never a decode/verify selector.
 */
#define AUGUR_CUDA_QWEN35_CANONICAL_PREFILL_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_QWEN35_CANONICAL_PREFILL_MAX_TOKENS_V1 UINT32_C(128)
#define AUGUR_CUDA_QWEN35_CANONICAL_PREFILL_SCORE_WARP_V1 UINT32_C(0)
#define AUGUR_CUDA_QWEN35_CANONICAL_PREFILL_SCORE_DGEMM_V1 UINT32_C(1)
typedef augur_cuda_qwen35_kv8_batched_attention_args_v4
    augur_cuda_qwen35_canonical_prefill_args_v1;

#if defined(AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION)
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_batched_attention_v4(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_batched_attention_args_v4* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_batched_attention_submit_v4(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_batched_attention_args_v4* args
);
#endif

/*
 * Explicit batched-prefill handoff from the F32 causal workspace into the
 * resident Q8_0 decode cache. The source and destination capacities are
 * identical; only [position_base, position_base + tokens) is converted.
 */
#define AUGUR_CUDA_QWEN35_PREFILL_KV8_CACHE_QUANTIZE_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_qwen35_prefill_kv8_cache_quantize_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t key_cache_f32;
    uint64_t key_cache_f32_len_bytes;
    augur_cuda_device_ptr_t value_cache_f32;
    uint64_t value_cache_f32_len_bytes;
    augur_cuda_device_ptr_t key_cache_q8_0;
    uint64_t key_cache_q8_0_len_bytes;
    augur_cuda_device_ptr_t value_cache_q8_0;
    uint64_t value_cache_q8_0_len_bytes;
    uint32_t capacity;
    uint32_t position_base;
    uint32_t tokens;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t reserved;
} augur_cuda_qwen35_prefill_kv8_cache_quantize_args_v1;

#if defined(AUGUR_CUDA_QWEN35_PREFILL_CUBLAS) \
    && defined(AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION)
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_cache_quantize_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_cache_quantize_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_cache_quantize_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_cache_quantize_args_v1* args
);
#endif

#define AUGUR_CUDA_BF16_F32_MATVEC_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_bf16_f32_matvec_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t rows;
    uint32_t cols;
} augur_cuda_bf16_f32_matvec_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_f32_matvec_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_bf16_f32_matvec_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_f32_matvec_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_bf16_f32_matvec_args_v1* args
);
/* Explicit SM70 research B1 pair, original BF16 [48,5120] each; exact
   extents, shared input, disjoint retained operands, no allocation or drain. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nerkyor_bf16_gate_pair_f32_cta_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_bf16_f32_matvec_args_v1* a,
    const augur_cuda_bf16_f32_matvec_args_v1* b
);
/* Canonical warp-row arithmetic, independent token-major rows, tokens1..512.
   All extents are exact and disjoint; no allocation or synchronization. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_f32_matvec_rows_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_bf16_f32_matvec_args_v1* args, uint32_t tokens
);

#define AUGUR_CUDA_Q8_0_F32_MATVEC_ARGS_VERSION_V1 UINT32_C(1)

/* Row-major GGML Q8_0 blocks: each row has cols / 32 blocks of 34 bytes. */
typedef struct augur_cuda_q8_0_f32_matvec_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t rows;
    uint32_t cols;
} augur_cuda_q8_0_f32_matvec_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q8_0_f32_matvec_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q8_0_f32_matvec_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q8_0_f32_matvec_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q8_0_f32_matvec_args_v1* args
);

#define AUGUR_CUDA_Q8_1_ACTIVATION_QUANTIZE_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_Q8_1_ACTIVATION_BLOCK_ELEMENTS_V1 UINT32_C(32)
#define AUGUR_CUDA_Q8_1_ACTIVATION_BLOCK_BYTES_V1 UINT32_C(36)

/*
 * Converts one contiguous f32 vector to llama.cpp-compatible Q8_1 blocks:
 * FP16 scale, FP16 sum, then 32 signed int8 codes. The quantized codes are
 * derived from the unrounded f32 scale; the stored scale/sum use FP16 RNE.
 */
typedef struct augur_cuda_q8_1_activation_quantize_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output_q8_1;
    uint64_t output_len_bytes;
    uint32_t elements;
    uint32_t reserved;
} augur_cuda_q8_1_activation_quantize_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q8_1_activation_quantize_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q8_1_activation_quantize_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q8_1_activation_quantize_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q8_1_activation_quantize_args_v1* args
);

/* Exact B8x17408 boundary; preserves both the F32 activation and Q8 bytes.
 * Preflight checks live disjoint allocations. Submit reuses that binding;
 * all backing owners must remain alive through stream completion. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_silu_q8_preserved_b8_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_f32_args_v1* silu,
    const augur_cuda_q8_1_activation_quantize_args_v1* quantize);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_silu_q8_preserved_b8_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_f32_args_v1* silu,
    const augur_cuda_q8_1_activation_quantize_args_v1* quantize);

/* Fixed B8x5120 output-head preparation. mode1: target norm+Q8 and three
 * last-row outputs. mode2: already-normalized draft Q8 and two last-row outputs;
 * weight/normalized/last_hidden and epsilon must be zero in mode2. */
typedef struct augur_cuda_qwen35_head_prepare_b8_args_v1 {
    uint32_t abi_version, struct_size, mode, epsilon_bits;
    augur_cuda_device_ptr_t input; uint64_t input_len_bytes;
    augur_cuda_device_ptr_t weight; uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t normalized; uint64_t normalized_len_bytes;
    augur_cuda_device_ptr_t quantized; uint64_t quantized_len_bytes;
    augur_cuda_device_ptr_t last_hidden; uint64_t last_hidden_len_bytes;
    augur_cuda_device_ptr_t last_normalized; uint64_t last_normalized_len_bytes;
    augur_cuda_device_ptr_t last_quantized; uint64_t last_quantized_len_bytes;
} augur_cuda_qwen35_head_prepare_b8_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_head_prepare_b8_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_head_prepare_b8_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_head_prepare_b8_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_head_prepare_b8_args_v1* args);

#define AUGUR_CUDA_Q8_0_Q8_1_MATVEC_ARGS_VERSION_V1 UINT32_C(1)

/* Q8_0 row-major weights multiplied by one Q8_1 activation vector. */
typedef struct augur_cuda_q8_0_q8_1_matvec_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input_q8_1;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t rows;
    uint32_t cols;
} augur_cuda_q8_0_q8_1_matvec_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q8_0_q8_1_matvec_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q8_0_q8_1_matvec_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q8_0_q8_1_matvec_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q8_0_q8_1_matvec_args_v1* args
);

#if defined(AUGUR_CUDA_QWEN35_FUSED_FFN_SWIGLU)
#define AUGUR_CUDA_Q8_0_Q8_1_FUSED_SWIGLU_ARGS_VERSION_V1 UINT32_C(1)

/* Two Q8_0 projections sharing one Q8_1 input, fused with SiLU(gate) * up. */
typedef struct augur_cuda_q8_0_q8_1_fused_swiglu_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t gate_weight;
    uint64_t gate_weight_len_bytes;
    augur_cuda_device_ptr_t up_weight;
    uint64_t up_weight_len_bytes;
    augur_cuda_device_ptr_t input_q8_1;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t rows;
    uint32_t cols;
} augur_cuda_q8_0_q8_1_fused_swiglu_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q8_0_q8_1_fused_swiglu_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q8_0_q8_1_fused_swiglu_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q8_0_q8_1_fused_swiglu_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q8_0_q8_1_fused_swiglu_args_v1* args
);
#endif

#define AUGUR_CUDA_NVFP4_SM70_F32_MATVEC_ARGS_VERSION_V1 UINT32_C(1)

/*
 * Row-major GGML NVFP4 blocks for the SM70 component path.  Each row has
 * cols / 64 blocks of 36 bytes: four UE4M3 scale bytes followed by 32 bytes
 * containing two E2M1 nibbles per byte.  The tensor-level f32 scale is
 * applied after the dot product and is carried inline in this ABI.
 */
typedef struct augur_cuda_nvfp4_sm70_f32_matvec_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t rows;
    uint32_t cols;
    float tensor_scale;
    uint32_t reserved;
} augur_cuda_nvfp4_sm70_f32_matvec_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_f32_matvec_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_f32_matvec_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* args
);

/* QPN physical planes only, source F32 pairs with F64 fold. mode=0: CTA,
 * splits=1/4/8/16 with null workspace; mode=1: parts/merge, splits=4/8/16/32,
 * workspace exactly N*splits*sizeof(double). mode=2: source-exact lane/tree
 * order, splits=4, null/zero workspace. Separate from all GGML entries. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_qpn_source_pair_submit_v1(
    int32_t, augur_cuda_stream_t, const augur_cuda_nvfp4_sm70_f32_matvec_args_v1*,
    uint32_t mode, uint32_t splits, uint64_t partial, uint64_t partial_bytes
);

/* Independent large source-F32 pair / SplitK8 F64 research schedule.
 * N>=1024 and N%32==0; K>=5120 and K%512==0. No scalar-bit-exact promise. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_large_split8_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* args
);
/* Independent 48x5120-only F32-pair, SplitK16 F64 contract; not scalar-bit-exact. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_small_gate_split16_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* args
);

/* Two real 48x5120 SplitK16 gates, identical input range, disjoint outputs.
 * B1 only. Schedule1=fixed-K row-concat; schedule2=shared-input prefetch5.
 * Per-row numerical contract is unchanged from small_gate_split16_v1. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_small_gate_pair_prevalidate_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* a,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* b, uint32_t schedule
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_small_gate_pair_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* a,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* b, uint32_t schedule
);

/* Explicit row-major batch1..8; original descriptor/version/reserved semantics
 * stay unchanged. Input/output extents cover the entire real batch, no padding. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_f32_batch_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_f32_matvec_args_v1* args, uint32_t batch_rows
);

#define AUGUR_CUDA_NVFP4_SM70_Q8_1_MATVEC_ARGS_VERSION_V1 UINT32_C(1)

/* NVFP4 row-major weights multiplied by one Q8_1 activation vector. */
typedef struct augur_cuda_nvfp4_sm70_q8_1_matvec_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input_q8_1;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t rows;
    uint32_t cols;
    float tensor_scale;
    uint32_t reserved;
} augur_cuda_nvfp4_sm70_q8_1_matvec_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_q8_1_matvec_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_q8_1_matvec_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_q8_1_matvec_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_q8_1_matvec_args_v1* args
);

#if defined(AUGUR_CUDA_QWEN35_FUSED_FFN_SWIGLU)
#define AUGUR_CUDA_NVFP4_SM70_Q8_1_FUSED_SWIGLU_ARGS_VERSION_V1 UINT32_C(1)

/* Two NVFP4 projections sharing one Q8_1 input, fused with SiLU(gate) * up. */
typedef struct augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t gate_weight;
    uint64_t gate_weight_len_bytes;
    augur_cuda_device_ptr_t up_weight;
    uint64_t up_weight_len_bytes;
    augur_cuda_device_ptr_t input_q8_1;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t rows;
    uint32_t cols;
    float gate_tensor_scale;
    float up_tensor_scale;
} augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_args_v1* args
);
#endif

#if defined(AUGUR_CUDA_QWEN35_PREFILL_CUBLAS)
#define AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_PLAN_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_PROJECTION_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1 UINT32_C(1)
#define AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1 UINT32_C(2)
#define AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1 UINT32_C(3)
/* Accepted only by the exact B8 DFlash selector-gate entry point. */
#define AUGUR_CUDA_DFLASH_SELECTOR_GATE_WEIGHT_F32_V1 UINT32_C(4)
#define AUGUR_CUDA_DFLASH_SELECTOR_GATE_F64_ACCUM_V1 UINT32_C(5)
// Dedicated planar-head entry points only; old projection entries reject it.
#define AUGUR_CUDA_QWEN35_PLANAR_Q8_HEAD_WEIGHT_V1 UINT32_C(4)
#define AUGUR_CUDA_QWEN35_ROW_Q8_INPUT_WEIGHT_V1 UINT32_C(5)
#define AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_MAX_TOKENS_V1 UINT32_C(1024)
/* Research-build capability only; the ordinary runtime and TC admission
 * remain at MAX_TOKENS_V1. No argument layout or state ABI changes. */
#if defined(AUGUR_CUDA_QWEN35_AUGUR_LARGE_PREFILL)
// Additional headroom avoids a separate44-row 8K weight-decode round.
#define AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1 UINT32_C(6400)
#else
#define AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1 AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_MAX_TOKENS_V1
#endif
#define AUGUR_CUDA_QWEN35_PREFILL_RMSNORM_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_QWEN35_PREFILL_LINEAR_CHUNK_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_QWEN35_PREFILL_FULL_ATTENTION_ARGS_VERSION_V1 UINT32_C(1)

/*
 * One stream-owned cuBLAS handle and two reusable FP16 decode workspaces.
 * The weight workspace is sized for the largest resident projection; the
 * activation workspace is sized for max_tokens * largest input width.
 */
typedef struct augur_cuda_qwen35_prefill_cublas_plan_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t max_tokens;
    uint32_t reserved0;
    augur_cuda_device_ptr_t decoded_weight_f16;
    uint64_t decoded_weight_len_bytes;
    augur_cuda_device_ptr_t decoded_input_f16;
    uint64_t decoded_input_len_bytes;
    uint64_t reserved[2];
} augur_cuda_qwen35_prefill_cublas_plan_args_v1;

/*
 * Row-major quantized W[rows, cols] times Q8_1 X[tokens, cols]. The output
 * is row-major F32 Y[tokens, rows]. NVFP4 applies tensor_scale while decoding;
 * Q8_0 requires tensor_scale == 1.0f.
 */
typedef struct augur_cuda_qwen35_prefill_cublas_projection_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t weight_format;
    uint32_t tokens;
    uint32_t rows;
    uint32_t cols;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input_q8_1;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    float tensor_scale;
    uint32_t reserved;
} augur_cuda_qwen35_prefill_cublas_projection_args_v1;

/* Opt-in F32 activation input, converted directly to the plan's FP16 scratch.
 * Separate entrypoints/type: the existing Q8_1 projection ABI is unchanged. */
#define AUGUR_CUDA_QWEN35_PREFILL_F32_PROJECTION_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_qwen35_prefill_f32_projection_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t weight_format;
    uint32_t tokens;
    uint32_t rows;
    uint32_t cols;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    float tensor_scale;
    uint32_t reserved;
} augur_cuda_qwen35_prefill_f32_projection_args_v1;

#if defined(AUGUR_CUDA_QWEN35_VISION_AFFINE)
/* Direct resident FP16 W[rows,cols] and X[tokens,cols], FP32 bias[rows]
 * and Y[tokens,rows]. No weight decoding, Q8 recoding or input casting.
 * Consumers separately own finite-value and FP16-packing admission. */
typedef struct augur_cuda_qwen35_vision_f16_affine_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t tokens;
    uint32_t rows;
    uint32_t cols;
    uint32_t reserved;
    augur_cuda_device_ptr_t weight_f16;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input_f16;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t bias_f32;
    uint64_t bias_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
} augur_cuda_qwen35_vision_f16_affine_args_v1;
/* Control stages for the fixed Qwen3.5 Vision encoder, FP32 control/FP16 affine input.
 * op: 1 position-add, 2 LN->F16, 3 cast->F16, 4 tanh-GELU->F16,
 * 5 erf-GELU->F16, 6 residual-add, 7 in-place QK rotary, 8 online attention.
 * status is a sticky uint32 error bit, initialized to zero by the owner. */
typedef struct augur_cuda_qwen35_vision_control_args_v1 {
    uint32_t abi_version, struct_size, operation, tokens, cols, reserved;
    augur_cuda_device_ptr_t input_a; uint64_t input_a_bytes;
    augur_cuda_device_ptr_t input_b; uint64_t input_b_bytes;
    augur_cuda_device_ptr_t input_c; uint64_t input_c_bytes;
    augur_cuda_device_ptr_t input_d; uint64_t input_d_bytes;
    augur_cuda_device_ptr_t input_e; uint64_t input_e_bytes;
    augur_cuda_device_ptr_t output; uint64_t output_bytes;
    augur_cuda_device_ptr_t status; uint64_t status_bytes;
} augur_cuda_qwen35_vision_control_args_v1;
#endif

/* Per-token RMSNorm over a row-major F32 activation batch. */
typedef struct augur_cuda_qwen35_prefill_rmsnorm_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t tokens;
    uint32_t cols;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t weight_f32;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    uint32_t epsilon_bits;
    uint32_t reserved;
} augur_cuda_qwen35_prefill_rmsnorm_args_v1;

/*
 * Causal DeltaNet chunk over row-major projected activations. Conv history
 * and recurrent state are updated in place and therefore carry continuity
 * between adjacent chunks. Alpha/beta are unprocessed projection outputs;
 * dt_bias and ssm_a are broadcast over tokens inside the kernel.
 */
typedef struct augur_cuda_qwen35_prefill_linear_chunk_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t tokens;
    uint32_t reserved0;
    augur_cuda_device_ptr_t qkv_f32;
    uint64_t qkv_len_bytes;
    augur_cuda_device_ptr_t z_f32;
    uint64_t z_len_bytes;
    augur_cuda_device_ptr_t conv_weights_f32;
    uint64_t conv_weights_len_bytes;
    augur_cuda_device_ptr_t conv_history_f32;
    uint64_t conv_history_len_bytes;
    augur_cuda_device_ptr_t alpha_raw_f32;
    uint64_t alpha_raw_len_bytes;
    augur_cuda_device_ptr_t beta_raw_f32;
    uint64_t beta_raw_len_bytes;
    augur_cuda_device_ptr_t dt_bias_f32;
    uint64_t dt_bias_len_bytes;
    augur_cuda_device_ptr_t ssm_a_f32;
    uint64_t ssm_a_len_bytes;
    augur_cuda_device_ptr_t state_f32;
    uint64_t state_len_bytes;
    augur_cuda_device_ptr_t norm_weight_f32;
    uint64_t norm_weight_len_bytes;
    augur_cuda_device_ptr_t activated_f32;
    uint64_t activated_len_bytes;
    augur_cuda_device_ptr_t recurrent_f32;
    uint64_t recurrent_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    uint32_t epsilon_bits;
    uint32_t reserved1;
} augur_cuda_qwen35_prefill_linear_chunk_args_v1;

/*
 * One causal full-attention chunk. Q rows are already reordered as contiguous
 * query then gate halves. The operation performs per-head RMSNorm/RoPE,
 * appends K/V at position_base, and evaluates every query against the exact
 * visible cache prefix. Cache ownership remains with the layer plan.
 */
typedef struct augur_cuda_qwen35_prefill_full_attention_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t q_full_f32;
    uint64_t q_full_len_bytes;
    augur_cuda_device_ptr_t k_raw_f32;
    uint64_t k_raw_len_bytes;
    augur_cuda_device_ptr_t value_f32;
    uint64_t value_len_bytes;
    augur_cuda_device_ptr_t q_norm_weight_f32;
    uint64_t q_norm_weight_len_bytes;
    augur_cuda_device_ptr_t k_norm_weight_f32;
    uint64_t k_norm_weight_len_bytes;
    augur_cuda_device_ptr_t rope_cos_f32;
    uint64_t rope_cos_len_bytes;
    augur_cuda_device_ptr_t rope_sin_f32;
    uint64_t rope_sin_len_bytes;
    augur_cuda_device_ptr_t key_cache_f32;
    uint64_t key_cache_len_bytes;
    augur_cuda_device_ptr_t value_cache_f32;
    uint64_t value_cache_len_bytes;
    augur_cuda_device_ptr_t query_f32;
    uint64_t query_len_bytes;
    augur_cuda_device_ptr_t gate_f32;
    uint64_t gate_len_bytes;
    augur_cuda_device_ptr_t attention_f32;
    uint64_t attention_len_bytes;
    augur_cuda_device_ptr_t gated_f32;
    uint64_t gated_len_bytes;
    uint32_t tokens;
    uint32_t position_base;
    uint32_t capacity;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t epsilon_bits;
    uint32_t reserved0;
} augur_cuda_qwen35_prefill_full_attention_args_v1;

/*
 * Opt-in chunk-owned KV8 prefill attention.  The projected Q/K/V rows are
 * chunk-sized; only the packed KV8 cache is capacity-sized.  Submission
 * preserves causal order on the owner stream and uses fixed per-request
 * scratch, so a 256K capacity never requires a capacity-sized F32 cache.
 */
#define AUGUR_CUDA_QWEN35_PREFILL_KV8_CHUNK_ATTENTION_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t q_full_f32;
    uint64_t q_full_len_bytes;
    augur_cuda_device_ptr_t k_raw_f32;
    uint64_t k_raw_len_bytes;
    augur_cuda_device_ptr_t value_f32;
    uint64_t value_len_bytes;
    augur_cuda_device_ptr_t q_norm_weight_f32;
    uint64_t q_norm_weight_len_bytes;
    augur_cuda_device_ptr_t k_norm_weight_f32;
    uint64_t k_norm_weight_len_bytes;
    augur_cuda_device_ptr_t rope_cos_f32;
    uint64_t rope_cos_len_bytes;
    augur_cuda_device_ptr_t rope_sin_f32;
    uint64_t rope_sin_len_bytes;
    augur_cuda_device_ptr_t key_work_f32;
    uint64_t key_work_len_bytes;
    augur_cuda_device_ptr_t value_work_f32;
    uint64_t value_work_len_bytes;
    augur_cuda_device_ptr_t key_cache_q8_0;
    uint64_t key_cache_len_bytes;
    augur_cuda_device_ptr_t value_cache_q8_0;
    uint64_t value_cache_len_bytes;
    augur_cuda_device_ptr_t query_f32;
    uint64_t query_len_bytes;
    augur_cuda_device_ptr_t gate_f32;
    uint64_t gate_len_bytes;
    augur_cuda_device_ptr_t attention_f32;
    uint64_t attention_len_bytes;
    augur_cuda_device_ptr_t gated_f32;
    uint64_t gated_len_bytes;
    uint32_t tokens;
    uint32_t position_base;
    uint32_t capacity;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t epsilon_bits;
    uint32_t reserved0;
} augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1;

#if defined(AUGUR_CUDA_QWEN35_PREFILL_CUBLAS) \
    && defined(AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION)
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_attention_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_attention_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_warp_ordered_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_warp_shared_kv_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_parallel_scales_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_tiled_dgemm_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, struct augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes);

/* Preflight is non-submitting. The submit call binds the same stream/plan and
 * retains every input, canonical partial state and score workspace to drain.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_canonical_prefill_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_canonical_prefill_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_canonical_prefill_prepare_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_canonical_prefill_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_canonical_prefill_consume_scores_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_canonical_prefill_args_v1* args,
    uint64_t scores, uint32_t key_begin, uint32_t key_count, uint32_t score_stride);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_canonical_prefill_finish_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_canonical_prefill_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_canonical_prefill_scores_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_canonical_prefill_args_v1* args,
    uint64_t score_address, uint64_t score_bytes, uint32_t score_policy);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_canonical_prefill_scores_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_canonical_prefill_args_v1* args,
    uint64_t score_address, uint64_t score_bytes, uint32_t score_policy);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_tiled_dgemm_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, struct augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_tiled_scores_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_tiled_scores_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_tiled_cached_query_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_staged_scores_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_staged_scores_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args,
    uint64_t score_address, uint64_t score_bytes);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_register_softmax_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_shared_softmax_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_shared_softmax_tile16_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_shared_softmax_unit_scale_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args
);
#if defined(AUGUR_CUDA_QWEN35_KV8_TOKEN_PAIR)
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_token_pair_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args
);
#endif
#endif

#if defined(AUGUR_CUDA_QWEN35_TC_PREFIX_ATTENTION)
#define AUGUR_CUDA_QWEN35_TC_PREFIX_ATTENTION_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_QWEN35_TC_PREFIX_ATTENTION_TILE_2K_V1 UINT32_C(2048)
#define AUGUR_CUDA_QWEN35_TC_PREFIX_ATTENTION_TILE_4K_V1 UINT32_C(4096)

/*
 * Split submissions for the integrated layer path. Prepare normalizes and
 * rotates Q/K, publishes the current K/V rows to the canonical KV8 cache,
 * and leaves query/gate rows resident. Gate consumes an externally computed
 * attention result. Both calls reuse the preflighted chunk binding.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_prepare_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_kv8_chunk_gate_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_kv8_chunk_attention_args_v1* args
);

/*
 * One complete four-KV-group causal attention component. Query/output are
 * token-major F32 [tokens, 24, 256]. K/V use the canonical capacity-major
 * Q8_0 cache [capacity, 4, 256]. The remaining ranges are fixed reusable
 * workspaces for one six-query-head KV group and one prefix tile.
 * probability_workspace_f16 contains two contiguous planes (FP16 high and
 * 4096-scaled residual), each [6*tokens, tile_positions]. Submission uses
 * tightly packed planes for the current tile_count, within the admitted
 * maximum extent. Old single-plane extents are rejected by preflight.
 */
typedef struct augur_cuda_qwen35_tc_prefix_attention_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t query_f32;
    uint64_t query_len_bytes;
    augur_cuda_device_ptr_t key_cache_q8_0;
    uint64_t key_cache_len_bytes;
    augur_cuda_device_ptr_t value_cache_q8_0;
    uint64_t value_cache_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    augur_cuda_device_ptr_t lse_f32;
    uint64_t lse_len_bytes;
    augur_cuda_device_ptr_t query_workspace_f16;
    uint64_t query_workspace_len_bytes;
    augur_cuda_device_ptr_t key_workspace_f16;
    uint64_t key_workspace_len_bytes;
    augur_cuda_device_ptr_t value_workspace_f16;
    uint64_t value_workspace_len_bytes;
    augur_cuda_device_ptr_t score_workspace_f32;
    uint64_t score_workspace_len_bytes;
    augur_cuda_device_ptr_t probability_workspace_f16;
    uint64_t probability_workspace_len_bytes;
    augur_cuda_device_ptr_t tile_output_workspace_f32;
    uint64_t tile_output_workspace_len_bytes;
    augur_cuda_device_ptr_t tile_lse_workspace_f32;
    uint64_t tile_lse_workspace_len_bytes;
    uint32_t tokens;
    uint32_t prefix_positions;
    uint32_t capacity;
    uint32_t tile_positions;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t scale_bits;
    uint32_t key_begin;
    uint32_t key_end;
    /* Optional bounded-cache slot metadata.  A zero base preserves the
     * capacity-major ABI; non-zero base maps logical_position to
     * slot_position = logical_position - cache_position_base. */
    uint32_t cache_position_base;
    uint32_t continuation_mode;
    uint32_t page_slot_mode;
} augur_cuda_qwen35_tc_prefix_attention_args_v1;
#endif

typedef struct augur_cuda_qwen35_prefill_cublas_plan_v1
    augur_cuda_qwen35_prefill_cublas_plan_v1;

/* B512 Q8_0 input projections only; direct original-F32 activation input.
 * Independent of the TC-prefix-attention experiment; reuses plan scratch. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_input_original_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_input_original_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args);

/* Separate opt-in B512 original-F32 projection contract. Includes bounded
 * input/output/FFN Q8_0 and FFN NVFP4 shapes. NVFP4 tensor_scale is applied
 * after the F32-output GEMM, never folded into F16 weights. Reuses plan scratch. */
/* Separate exact-input dispatch, 1..512 real rows; not B8 verify/replay. */
/* DFlash2 Q8_0 shape admission; llama small/MMQ/FP16 dispatch, shared scratch. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_llama_bf16_head_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_bf16_f32_matvec_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_llama_bf16_head_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_bf16_f32_matvec_args_v1* args);
/* Same B8 shared-head geometry; register-blocked FP32 kernel, 16-byte weight/input bases. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_bf16_head_f32_rw4_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_bf16_f32_matvec_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_bf16_head_f32_rw4_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_bf16_f32_matvec_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_selector_gate_f32_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_selector_gate_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_llama_projection_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_llama_projection_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args);
/* Same draft Q8_0 shape/range admission at exactly 8 tokens; original blocks read in
   place with FP32 activations (no Q8_1 quantization, no plan scratch); input aligned16. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_b8_direct_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_b8_direct_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_llama_projection_rows_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_llama_projection_rows_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args);
/* Same immutable F32 input, 2..3 projections, 64..512 rows. Reuse is confined
 * to this call; the first projection always packs anew. No extra workspace.
 * Two BF16 48x5120 GDN projections use one conversion and batched GEMM.
 * Mixed BF16/quantized groups are rejected; small-row dispatch is unchanged. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_llama_projection_group_rows_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args, uint32_t count);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_llama_swiglu_rows_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_silu_mul_f32_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_llama_swiglu_rows_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_silu_mul_f32_args_v1* args);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_llama_projection_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_llama_projection_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args);
/* Fixed B512 x17408 separate fast-math SwiGLU; no default primitive changes. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_llama_swiglu_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_f32_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_llama_swiglu_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_f32_args_v1* args);

#if defined(AUGUR_CUDA_QWEN35_TC_PREFIX_ATTENTION)
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_f32_projection_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_f32_projection_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args);

#endif

/* Pure capability query: no CUDA API, allocation or device selection. */
uint32_t AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_cublas_compiled_max_tokens_v1(void);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_cublas_plan_create_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_cublas_plan_args_v1* args,
    augur_cuda_qwen35_prefill_cublas_plan_v1** out_plan
);
#if defined(AUGUR_CUDA_QWEN35_VISION_AFFINE)
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_vision_f16_affine_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_vision_f16_affine_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_vision_f16_affine_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_vision_f16_affine_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_vision_control_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_vision_control_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_vision_control_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_vision_control_args_v1* args);
#endif
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_cublas_projection_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_cublas_projection_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args
);
// Default-off measured FFN: NVFP4 B128/256/512, Q8 B256/512 only.
// Exact integer MMA/canonical scale order; no reassociated group32 path.
// Borrows existing plan scratch; no allocation, synchronization or new owner.
uint64_t AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_integer_mma_scratch_bytes_v1(
    uint32_t format, uint32_t rows, uint32_t cols, uint32_t tokens);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_integer_mma_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args);

// Default-off B512 Q8 input projections only, cols5120 and rows1024/6144/10240/12288.
// Independent of FFN admission; reuses the same plan scratch, no new allocation.
uint64_t AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_input_integer_scratch_bytes_v1(
    uint32_t format, uint32_t rows, uint32_t cols, uint32_t tokens);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_input_integer_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args);

// Opt-in llama-style B512 input only: full Q8_0/Q8_1 -> F16, FP32 GEMM.
// Uses the same plan/stream and full preflighted scratch; no new allocation.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_input_llama_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args);

// Default-off FFN-down only: packed Q8_1 input, bounded decoded weight tiles,
// FP16 operands and FP32 Tensor Core accumulation. No allocation or sync.
uint32_t AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_down_tensor_tile_rows_v1(
    uint32_t format, uint32_t rows, uint32_t cols, uint32_t tokens);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_down_tensor_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args);
// Default-off compensated NVFP4 B128 down only, same owned plan/stream.
uint32_t AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_down_compensated_tile_rows_v1(
    uint32_t format, uint32_t rows, uint32_t cols, uint32_t tokens);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_down_compensated_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args);
// Diagnostic 64-byte summary: elements, decoded-bit mismatches, nonfinite pairs,
// then five FP64 fields: max absolute, squared error/ref/candidate, dot product.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_compare_summary_v1(
    int32_t ordinal, augur_cuda_stream_t stream, uint64_t candidate, uint64_t reference,
    uint64_t input_bytes, uint64_t output, uint64_t output_bytes, uint32_t format
);
// Experimental quantized 1..8-row projection; preserves canonical matvec order.
// Packed-only binding: no cuBLAS plan, allocation or decoded-weight workspace.
// exact_tiles is strictly 0 or 1. Retain all three owners through completion;
// preflight must precede prevalidated submission on the same stream/ordinal.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_packed_small_batch_projection_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args, uint32_t exact_tiles);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_packed_small_batch_projection_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args, uint32_t exact_tiles);
uint32_t AUGUR_CUDA_CALL augur_cuda_qwen35_exact_projection_tile_width_v1(
    uint32_t format, uint32_t rows, uint32_t cols, uint32_t tokens);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_planar_q8_head_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args);
// Row-local code/scale planes: K5120 and input rows1024/6144/10240/12288 only.
// Binding preflight checks the initial Q8 placeholder up to512 rows; only
// packed<=8 or explicitly bound original-F321..512 is executable.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_row_q8_input_binding_preflight_v1(
    int32_t, augur_cuda_stream_t, augur_cuda_qwen35_prefill_cublas_plan_v1*,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1*);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_row_q8_input_packed_preflight_v1(
    int32_t, augur_cuda_stream_t, const augur_cuda_qwen35_prefill_cublas_projection_args_v1*);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_row_q8_input_packed_submit_v1(
    int32_t, augur_cuda_stream_t, const augur_cuda_qwen35_prefill_cublas_projection_args_v1*);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_row_q8_input_original_preflight_v1(
    int32_t, augur_cuda_stream_t, augur_cuda_qwen35_prefill_cublas_plan_v1*,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1*);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_row_q8_input_original_submit_v1(
    int32_t, augur_cuda_stream_t, augur_cuda_qwen35_prefill_cublas_plan_v1*,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1*);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_row_q8_input_group_submit_v1(
    int32_t, augur_cuda_stream_t, augur_cuda_qwen35_prefill_cublas_plan_v1*,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1*, uint32_t);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_planar_q8_head_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_exact_projection_tiles_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args);
// Default-off row2 reuse on full8 input/gate-up shapes. Zero selector result
// means exact-tiles/canonical fallback, not rejection or a padded computation.
uint32_t AUGUR_CUDA_CALL augur_cuda_qwen35_projection_row_reuse_width_v1(
    uint32_t format, uint32_t rows, uint32_t cols, uint32_t tokens);
// Strict B8 5120x17408 Q8_0/NVFP4 FFN-down. Same preflight/plan/owner contract.
// No fallback: other dimensions/formats are rejected before launch.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_ffn_down_row_reuse_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args);
// Two independent NVFP4 B8 17408x5120 projections, same input/plan/stream.
// Preserves both FP32 outputs; no allocation, relayout or activation fusion.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_ffn_nvfp4_pair_preflight_v1(
    int32_t, augur_cuda_stream_t, augur_cuda_qwen35_prefill_cublas_plan_v1*,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1*,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1*);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_ffn_nvfp4_pair_prevalidated_submit_v1(
    int32_t, augur_cuda_stream_t, augur_cuda_qwen35_prefill_cublas_plan_v1*,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1*,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1*);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_projection_row_reuse_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args);
// Explicit canonical <=512-row diagnostic grid; does not alter small-batch ABI.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_canonical_prefill_grid_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_quantized_small_batch_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args
);
// Diagnostic split only. Caller must run stages 0(weight), 1(input), 2(GEMM)
// consecutively on the exact same plan/stream/binding, no scratch interference.
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_prefill_cublas_projection_profile_stage_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args,
    uint32_t stage
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_cublas_projection_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args
);
#if defined(AUGUR_CUDA_QWEN35_TC_PREFIX_ATTENTION)
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_tc_prefix_attention_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_tc_prefix_attention_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_tc_prefix_attention_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_tc_prefix_attention_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_qwen35_simt_prefix_attention_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_tc_prefix_attention_args_v1* args
);
#endif
/* Bind-time device-range validation followed by allocation-free submission. */
// Explicit paired weight-decode candidate, otherwise identical full projection.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_cublas_projection_pair_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args
);
// Same ordered profiling-only 0 -> 1 -> 2 scratch lease as the scalar variant.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_cublas_projection_pair_profile_stage_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args, uint32_t stage
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_cublas_projection_quad_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_cublas_projection_quad_profile_stage_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_cublas_projection_args_v1* args, uint32_t stage
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_rmsnorm_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_rmsnorm_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_rmsnorm_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_rmsnorm_args_v1* args
);
// Explicit input-only rows1..512, hidden5120, epsilon1e-6. Uses the ordinary
// live-range preflight; never substitutes for the canonical/B8/decode API.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_llama_input_rmsnorm_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_rmsnorm_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_linear_chunk_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_linear_chunk_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args
);
// State-only fold shares the validated chunk ABI, but writes only conv_history,
// activated and state. It never produces recurrent/query or gated output.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_linear_chunk_fold_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args
);
// Replay-only 1..8-row candidate. Also writes the first tokens*48*3 F32 words
// of output as dead scalar scratch; no recurrent/query/gated output is produced.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_linear_chunk_fold_shared_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args
);
// Fold from immutable checkpoints into the bound working history/state. All
// checkpoint allocations must remain live through same-stream completion.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_linear_checkpoint_fold_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args,
    uint64_t history, uint64_t history_bytes, uint64_t state, uint64_t state_bytes
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_linear_checkpoint_fold_prevalidated_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args,
    uint64_t history, uint64_t history_bytes, uint64_t state, uint64_t state_bytes
);
// Explicit candidate: same validated ranges, output reused as scalar scratch
// only before the same-stream gated output kernel. Default submission unchanged.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_linear_chunk_shared_scalars_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_checkpoint_prefill_linear_chunk_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args
);
// Explicit numerical-contract candidate: FP64 device accumulation with the
// existing FP32 state/output ABI. Default submission remains unchanged.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_linear_chunk_precise_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_linear_chunk_args_v1* args
);
/* Separate research ABI; never changes legacy QUASAR prefill math. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nerkyor_prefill_full_attention_canonical_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_full_attention_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_full_attention_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_full_attention_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_full_attention_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_prefill_full_attention_args_v1* args
);
void AUGUR_CUDA_CALL augur_cuda_qwen35_prefill_cublas_plan_destroy_v1(
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan
);
#endif

#if defined(AUGUR_CUDA_P0_PREFILL_V22_PROJECTION_RESUME_BATCH_EXACT)
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_p0_prefill_v22_bf16_batch4_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_bf16_f32_matvec_args_v1* args,
    uint32_t tokens
);
#endif

#define AUGUR_CUDA_BF16_TOKEN_ROW_GATHER_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_bf16_token_row_gather_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t token_source_u32;
    uint64_t token_source_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    uint32_t rows;
    uint32_t cols;
} augur_cuda_bf16_token_row_gather_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_token_row_gather_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_bf16_token_row_gather_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_token_row_gather_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_bf16_token_row_gather_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_token_rows_gather_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_bf16_token_row_gather_args_v1* args, uint32_t tokens
);

#define AUGUR_CUDA_RMSNORM_F32_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_rmsnorm_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t elements;
    uint32_t epsilon_bits;
} augur_cuda_rmsnorm_f32_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_rmsnorm_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_rmsnorm_f32_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_rmsnorm_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_rmsnorm_f32_args_v1* args
);
// Canonical independent rows, 1..512. args.elements is the row width;
// input/output lengths are exact full-batch extents; weight is one shared row.
// Caller retains validated live disjoint buffers until stream completion.
// Explicit5120-only schedule; same original256-lane arithmetic tree.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_rmsnorm_f32_warp_tree5120_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_rmsnorm_f32_args_v1* args);

// B1 only. Preserve F32 output while producing both residual-v2 QPN planes.
// The caller retains all operands; no readiness survives a submission.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_rmsnorm_f32_warp_tree5120_qpn_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_rmsnorm_f32_args_v1* args,
    uint64_t staged, uint64_t staged_bytes);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_rmsnorm_f32_rows_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_rmsnorm_f32_args_v1* args, uint32_t rows
);

#define AUGUR_CUDA_SPLIT_RMSNORM_F32_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_split_rmsnorm_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    augur_cuda_device_ptr_t scale_workspace;
    uint64_t scale_workspace_len_bytes;
    uint32_t elements;
    uint32_t epsilon_bits;
} augur_cuda_split_rmsnorm_f32_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_split_rmsnorm_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_split_rmsnorm_f32_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_split_rmsnorm_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_split_rmsnorm_f32_args_v1* args
);

#define AUGUR_CUDA_ROPE_F32_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_rope_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    augur_cuda_device_ptr_t cos;
    uint64_t cos_len_bytes;
    augur_cuda_device_ptr_t sin;
    uint64_t sin_len_bytes;
    uint32_t elements;
    uint32_t dimensions;
} augur_cuda_rope_f32_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_rope_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_rope_f32_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_rope_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_rope_f32_args_v1* args
);

// DFlash K/V uses NeoX half-split pairing rather than adjacent-dimension pairing.
#define AUGUR_CUDA_DFLASH_ROPE_F32_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_dflash_rope_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t elements;
    uint32_t dimensions;
    uint32_t rotary_dimensions;
    uint32_t reserved;
    uint64_t position;
    float theta;
} augur_cuda_dflash_rope_f32_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_rope_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_rope_f32_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_rope_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_rope_f32_args_v1* args
);

// Bounded DFlash cache consumer. The query is 32 heads x 128 values, while
// the cache is 8 KV heads x 128 values in position-major order. This ABI is a
// component boundary only; it does not encode draft-state or acceptance.
// capacity is 1..2048. Absolute [start,end) is nonempty, no longer than
// capacity, and maps to physical position % capacity. The owner guarantees
// every selected slot is initialized, current, and ordered on this stream.
#define AUGUR_CUDA_DFLASH_CACHE_ATTENTION_F32_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_dflash_cache_attention_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t query;
    uint64_t query_len_bytes;
    augur_cuda_device_ptr_t key_cache;
    uint64_t key_cache_len_bytes;
    augur_cuda_device_ptr_t value_cache;
    uint64_t value_cache_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t capacity_positions;
    uint32_t visible_position_start;
    uint32_t visible_position_end;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t reserved;
    float scale;
} augur_cuda_dflash_cache_attention_f32_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_cache_attention_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_cache_attention_f32_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_cache_attention_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_cache_attention_f32_args_v1* args
);

// Dense [context rows || eight draft rows], not the single-query circular cache.
// All eight queries see future draft rows. SWA imposes only a left boundary.
#define AUGUR_CUDA_DFLASH_BLOCK_ATTENTION_F32_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_dflash_block_attention_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t query;
    uint64_t query_len_bytes;
    augur_cuda_device_ptr_t key;
    uint64_t key_len_bytes;
    augur_cuda_device_ptr_t value;
    uint64_t value_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t context_len;
    uint32_t context_start;
    uint32_t block_start;
    uint32_t sliding_window;
    uint32_t block_size;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    float scale;
    uint32_t reserved;
} augur_cuda_dflash_block_attention_f32_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_block_attention_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_block_attention_f32_args_v1* args
);

#define AUGUR_CUDA_F8E4M3_F32_MATVEC_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_f8e4m3_f32_matvec_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t scales;
    uint64_t scales_len_bytes;
    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    uint32_t rows;
    uint32_t cols;
} augur_cuda_f8e4m3_f32_matvec_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_f8e4m3_f32_matvec_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_f8e4m3_f32_matvec_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_bitdecode_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_f8e4m3_f32_matvec_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_bitdecode_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_f8e4m3_f32_matvec_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_warp_fp32_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_f8e4m3_f32_matvec_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_warp_fp32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_f8e4m3_f32_matvec_args_v1* args
);
#if defined(AUGUR_CUDA_P0_PREFILL_V22_PROJECTION_RESUME_BATCH_EXACT)
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_p0_prefill_v22_f8e4m3_batch4_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_f8e4m3_f32_matvec_args_v1* args,
    uint32_t tokens
);
#endif
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_row32_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_f8e4m3_f32_matvec_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_row32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_f8e4m3_f32_matvec_args_v1* args
);

// Row-scale variant for safetensors F8_E4M3 matrices (one FP32 scale per output row).
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_row_scale_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_f8e4m3_f32_matvec_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_row_scale_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_f8e4m3_f32_matvec_args_v1* args
);

#define AUGUR_CUDA_M139_RATIO4_STAGED_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_M139_KIND_MAIN_V1 UINT32_C(0)
#define AUGUR_CUDA_M139_KIND_INDEX_V1 UINT32_C(1)
typedef struct augur_cuda_m139_ratio4_staged_args_v1 {
    uint32_t abi_version; uint32_t struct_size;
    augur_cuda_device_ptr_t kv_projection; uint64_t kv_projection_len_bytes;
    augur_cuda_device_ptr_t gate_projection; uint64_t gate_projection_len_bytes;
    augur_cuda_device_ptr_t ape_f32; uint64_t ape_len_bytes;
    augur_cuda_device_ptr_t norm; uint64_t norm_len_bytes;
    augur_cuda_device_ptr_t gate_plus_ape; uint64_t gate_plus_ape_len_bytes;
    augur_cuda_device_ptr_t pooled_values; uint64_t pooled_values_len_bytes;
    augur_cuda_device_ptr_t pooled_scores; uint64_t pooled_scores_len_bytes;
    augur_cuda_device_ptr_t softmax; uint64_t softmax_len_bytes;
    augur_cuda_device_ptr_t weighted_pool; uint64_t weighted_pool_len_bytes;
    augur_cuda_device_ptr_t bf16_boundary; uint64_t bf16_boundary_len_bytes;
    augur_cuda_device_ptr_t rms_norm; uint64_t rms_norm_len_bytes;
    augur_cuda_device_ptr_t rope; uint64_t rope_len_bytes;
    augur_cuda_device_ptr_t hadamard; uint64_t hadamard_len_bytes;
    augur_cuda_device_ptr_t qdq; uint64_t qdq_len_bytes;
    augur_cuda_device_ptr_t output; uint64_t output_len_bytes;
    uint32_t tokens; uint32_t head_dim; uint32_t kind; uint32_t norm_is_bf16;
} augur_cuda_m139_ratio4_staged_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m139_ratio4_staged_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m139_ratio4_staged_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m287_ratio4_staged_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m139_ratio4_staged_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m287_ratio4_staged_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m139_ratio4_staged_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_toc_fast_ratio4_unquantized_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m139_ratio4_staged_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_toc_fast_ratio4_fused_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m139_ratio4_staged_args_v1* args
);

#define AUGUR_CUDA_M142_ORDERED_TOPK_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m142_ordered_topk_args_v1 {
    uint32_t abi_version; uint32_t struct_size;
    augur_cuda_device_ptr_t scores; uint64_t scores_len_bytes;
    augur_cuda_device_ptr_t indices; uint64_t indices_len_bytes;
    augur_cuda_device_ptr_t selected_scores; uint64_t selected_scores_len_bytes;
    uint32_t tokens; uint32_t candidates; uint32_t top_k; uint32_t causal_ratio;
    int32_t index_base; int32_t padding;
} augur_cuda_m142_ordered_topk_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m142_ordered_topk_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m142_ordered_topk_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m142_ordered_topk_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m142_ordered_topk_args_v1* args
);

#define AUGUR_CUDA_M144_RATIO128_STAGED_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m144_ratio128_staged_args_v1 {
    uint32_t abi_version; uint32_t struct_size;
    augur_cuda_device_ptr_t kv_projection; uint64_t kv_projection_len_bytes;
    augur_cuda_device_ptr_t gate_projection; uint64_t gate_projection_len_bytes;
    augur_cuda_device_ptr_t ape_f32; uint64_t ape_len_bytes;
    augur_cuda_device_ptr_t norm_f32; uint64_t norm_len_bytes;
    augur_cuda_device_ptr_t gate_plus_ape; uint64_t gate_plus_ape_len_bytes;
    augur_cuda_device_ptr_t softmax; uint64_t softmax_len_bytes;
    augur_cuda_device_ptr_t weighted; uint64_t weighted_len_bytes;
    augur_cuda_device_ptr_t reduced; uint64_t reduced_len_bytes;
    augur_cuda_device_ptr_t bf16_boundary; uint64_t bf16_boundary_len_bytes;
    augur_cuda_device_ptr_t rms_norm; uint64_t rms_norm_len_bytes;
    augur_cuda_device_ptr_t rope; uint64_t rope_len_bytes;
    augur_cuda_device_ptr_t qdq; uint64_t qdq_len_bytes;
    augur_cuda_device_ptr_t output; uint64_t output_len_bytes;
} augur_cuda_m144_ratio128_staged_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m144_ratio128_staged_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m144_ratio128_staged_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m288_ratio128_staged_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m144_ratio128_staged_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m288_ratio128_staged_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m144_ratio128_staged_args_v1* args
);

/*
 * M289 owns the first executable 128-position attention ABI.  These layouts
 * are intentionally separate from the historical M146 160-byte dispatcher:
 * the old fixed-eight symbols and operation numbers remain replayable.
 */
#define AUGUR_CUDA_M289_ROPE_POSITION_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_M289_INDEX_SCORE_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_M289_ATTENTION_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_M289_CAPACITY_V1 UINT32_C(128)
#define AUGUR_CUDA_M289_HEADS_V1 UINT32_C(64)
#define AUGUR_CUDA_M289_HEAD_DIM_V1 UINT32_C(512)
#define AUGUR_CUDA_M289_INDEX_HEAD_DIM_V1 UINT32_C(128)
#define AUGUR_CUDA_M289_VARIANT_RAW_V1 UINT32_C(0)
#define AUGUR_CUDA_M289_VARIANT_RATIO4_V1 UINT32_C(1)
#define AUGUR_CUDA_M289_VARIANT_RATIO128_V1 UINT32_C(2)
#define AUGUR_CUDA_M289_RATIO_RAW_V1 UINT32_C(0)
#define AUGUR_CUDA_M289_RATIO4_V1 UINT32_C(4)
#define AUGUR_CUDA_M289_RATIO128_V1 UINT32_C(128)
#define AUGUR_CUDA_M289_INDEX_BASE_V1 UINT32_C(128)
#define AUGUR_CUDA_M289_FLAG_RING_CURSOR_V1 UINT32_C(1)

typedef struct augur_cuda_m289_device_view_v1 {
    augur_cuda_device_ptr_t base;
    uint64_t allocation_extent_bytes;
    uint64_t view_offset_bytes;
    uint64_t view_extent_bytes;
} augur_cuda_m289_device_view_v1;

typedef struct augur_cuda_m289_rope_position_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t capacity;
    uint32_t position;
    uint32_t rows;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t repeat;
    uint32_t flags;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t reserved2[2];
    augur_cuda_m289_device_view_v1 input;
    augur_cuda_m289_device_view_v1 output;
} augur_cuda_m289_rope_position_args_v1;

typedef struct augur_cuda_m289_index_score_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t capacity;
    uint32_t position;
    uint32_t candidate_rows;
    uint32_t top_k;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t index_base;
    uint32_t variant;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t reserved1;
    augur_cuda_m289_device_view_v1 query;
    augur_cuda_m289_device_view_v1 compressed;
    augur_cuda_m289_device_view_v1 head_weight;
    augur_cuda_m289_device_view_v1 scores;
} augur_cuda_m289_index_score_args_v1;

typedef struct augur_cuda_m289_attention_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t capacity;
    uint32_t position;
    uint32_t query_position;
    uint32_t variant;
    uint32_t compression_ratio;
    uint32_t raw_rows_visible;
    uint32_t compressed_rows_visible;
    uint32_t ids_row_index;
    uint32_t top_k;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t flags;
    uint32_t reserved0;
    augur_cuda_m289_device_view_v1 query;
    augur_cuda_m289_device_view_v1 raw;
    augur_cuda_m289_device_view_v1 compressed;
    augur_cuda_m289_device_view_v1 ids;
    augur_cuda_m289_device_view_v1 sink;
    augur_cuda_m289_device_view_v1 weighted;
    augur_cuda_m289_device_view_v1 inverse;
} augur_cuda_m289_attention_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m289_rope_position_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m289_rope_position_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m289_rope_position_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m289_rope_position_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m289_index_score_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m289_index_score_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m289_index_score_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m289_index_score_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_toc_fast_m289_index_score_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m289_index_score_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m289_attention_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m289_attention_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m289_attention_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m289_attention_args_v1* args
);
/*
 * Product Request MLA successor.  It retains the frozen M289 argument layout
 * but applies ids_row_index before consuming Ratio4 ordered IDs.  The M289
 * symbols above intentionally preserve their historical behavior.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_request_mla_attention_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m289_attention_args_v1* args
);
/*
 * Isolated 1K Request MLA successor. These symbols reuse the frozen argument
 * layouts while binding logical 1024, Raw 128, Ratio4 256 and Ratio128 8.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_request_mla_index_score_1k_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m289_index_score_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_request_mla_ordered_topk_1k_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m142_ordered_topk_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_request_mla_attention_1k_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m289_attention_args_v1* args
);

/*
 * CTX-2 device-resident Group-64 KV8 append ABI.
 *
 * Each block quantizes one prevalidated F32 source row into the frozen
 * values-first signed INT8 + little-endian FP16 scale-tail wire layout. The
 * sticky status is reset only by the layer-zero single-row launch and copied
 * to pinned host memory immediately before the owner's terminal boundary.
 */
#define AUGUR_CUDA_KV8_DSA_QUANTIZE_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_QUANTIZE_M128_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_QUANTIZE_M128_ROWS_V1 UINT32_C(128)
#define AUGUR_CUDA_KV8_DSA_STATUS_COPY_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_MAX_ROWS_V1 UINT32_C(3)
#define AUGUR_CUDA_KV8_DSA_GROUP_VALUES_V1 UINT32_C(64)
#define AUGUR_CUDA_KV8_DSA_MAIN_VALUES_V1 UINT32_C(512)
#define AUGUR_CUDA_KV8_DSA_INDEX_VALUES_V1 UINT32_C(128)
#define AUGUR_CUDA_KV8_DSA_MAIN_WIRE_BYTES_V1 UINT32_C(528)
#define AUGUR_CUDA_KV8_DSA_INDEX_WIRE_BYTES_V1 UINT32_C(132)
#define AUGUR_CUDA_KV8_DSA_ROLE_RAW_MAIN_V1 UINT32_C(0)
#define AUGUR_CUDA_KV8_DSA_ROLE_RATIO4_MAIN_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_ROLE_RATIO4_INDEX_V1 UINT32_C(2)
#define AUGUR_CUDA_KV8_DSA_ROLE_RATIO128_MAIN_V1 UINT32_C(3)
#define AUGUR_CUDA_KV8_DSA_STATUS_OK_V1 UINT64_MAX
#define AUGUR_CUDA_KV8_DSA_STATUS_ERROR_BITS_V1 UINT32_C(2)
#define AUGUR_CUDA_KV8_DSA_STATUS_ELEMENT_BITS_V1 UINT32_C(9)
#define AUGUR_CUDA_KV8_DSA_STATUS_ROW_BITS_V1 UINT32_C(2)
#define AUGUR_CUDA_KV8_DSA_STATUS_LAYER_BITS_V1 UINT32_C(6)
#define AUGUR_CUDA_KV8_DSA_STATUS_POSITION_BITS_V1 UINT32_C(18)
#define AUGUR_CUDA_KV8_DSA_STATUS_ERROR_SHIFT_V1 UINT32_C(0)
#define AUGUR_CUDA_KV8_DSA_STATUS_ELEMENT_SHIFT_V1 UINT32_C(2)
#define AUGUR_CUDA_KV8_DSA_STATUS_ROW_SHIFT_V1 UINT32_C(11)
#define AUGUR_CUDA_KV8_DSA_STATUS_LAYER_SHIFT_V1 UINT32_C(13)
#define AUGUR_CUDA_KV8_DSA_STATUS_POSITION_SHIFT_V1 UINT32_C(19)
#define AUGUR_CUDA_KV8_DSA_STATUS_ERROR_MASK_V1 UINT64_C(0x3)
#define AUGUR_CUDA_KV8_DSA_STATUS_ELEMENT_MASK_V1 UINT64_C(0x1ff)
#define AUGUR_CUDA_KV8_DSA_STATUS_ROW_MASK_V1 UINT64_C(0x3)
#define AUGUR_CUDA_KV8_DSA_STATUS_LAYER_MASK_V1 UINT64_C(0x3f)
#define AUGUR_CUDA_KV8_DSA_STATUS_POSITION_MASK_V1 UINT64_C(0x3ffff)
#define AUGUR_CUDA_KV8_DSA_STATUS_USED_BITS_V1 UINT32_C(37)
#define AUGUR_CUDA_KV8_DSA_ERROR_NON_FINITE_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_ERROR_OUT_OF_RANGE_V1 UINT32_C(2)

/* CTX-3 device-resident Ratio4 score and stable Top-10 selection ABI. */
#define AUGUR_CUDA_KV8_DSA_SELECT_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_SELECT_RECEIPT_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_SELECT_HEADS_V1 UINT32_C(64)
#define AUGUR_CUDA_KV8_DSA_SELECT_HEAD_DIM_V1 UINT32_C(128)
#define AUGUR_CUDA_KV8_DSA_SELECT_TOP_K_V1 UINT32_C(10)
#define AUGUR_CUDA_KV8_DSA_SELECT_INDEX_BASE_V1 UINT32_C(128)
#define AUGUR_CUDA_KV8_DSA_SELECT_PADDING_V1 INT32_C(-1)
#define AUGUR_CUDA_KV8_DSA_SELECT_MAX_CANDIDATES_V1 UINT32_C(65536)
#define AUGUR_CUDA_KV8_DSA_SELECT_PARTITION_CANDIDATES_V1 UINT32_C(256)
#define AUGUR_CUDA_KV8_DSA_SELECT_MAX_PARTITIONS_V1 UINT32_C(256)
#define AUGUR_CUDA_KV8_DSA_SELECT_QUERY_BYTES_V1 UINT64_C(32768)
#define AUGUR_CUDA_KV8_DSA_SELECT_HEAD_WEIGHT_BYTES_V1 UINT64_C(256)
#define AUGUR_CUDA_KV8_DSA_SELECT_SCORES_BYTES_V1 UINT64_C(262144)
#define AUGUR_CUDA_KV8_DSA_SELECT_PARTIAL_IDS_BYTES_V1 UINT64_C(10240)
#define AUGUR_CUDA_KV8_DSA_SELECT_PARTIAL_SCORES_BYTES_V1 UINT64_C(10240)
#define AUGUR_CUDA_KV8_DSA_SELECT_IDS_BYTES_V1 UINT64_C(40)
#define AUGUR_CUDA_KV8_DSA_SELECT_SELECTED_SCORES_BYTES_V1 UINT64_C(40)
#define AUGUR_CUDA_KV8_DSA_SELECT_RECEIPT_BYTES_V1 UINT64_C(48)

#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_MARKER_V1 (UINT64_C(1) << 63)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_RESERVED_BITS_V1 UINT32_C(6)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_ERROR_BITS_V1 UINT32_C(3)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_ELEMENT_BITS_V1 UINT32_C(14)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_CANDIDATE_BITS_V1 UINT32_C(16)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_LAYER_BITS_V1 UINT32_C(6)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_POSITION_BITS_V1 UINT32_C(18)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_ERROR_SHIFT_V1 UINT32_C(6)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_ELEMENT_SHIFT_V1 UINT32_C(9)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_CANDIDATE_SHIFT_V1 UINT32_C(23)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_LAYER_SHIFT_V1 UINT32_C(39)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_POSITION_SHIFT_V1 UINT32_C(45)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_RESERVED_MASK_V1 UINT64_C(0x3f)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_ERROR_MASK_V1 UINT64_C(0x7)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_ELEMENT_MASK_V1 UINT64_C(0x3fff)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_CANDIDATE_MASK_V1 UINT64_C(0xffff)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_LAYER_MASK_V1 UINT64_C(0x3f)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_POSITION_MASK_V1 UINT64_C(0x3ffff)
#define AUGUR_CUDA_KV8_DSA_SELECT_ERROR_NON_FINITE_QUERY_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_SELECT_ERROR_NON_FINITE_HEAD_WEIGHT_V1 UINT32_C(2)
#define AUGUR_CUDA_KV8_DSA_SELECT_ERROR_INVALID_CODE_V1 UINT32_C(3)
#define AUGUR_CUDA_KV8_DSA_SELECT_ERROR_INVALID_SCALE_V1 UINT32_C(4)
#define AUGUR_CUDA_KV8_DSA_SELECT_ERROR_NON_FINITE_SCORE_V1 UINT32_C(5)

/* CTX-4 device-resident KV8 fused sparse-attention ABI. */
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_HEADS_V1 UINT32_C(64)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_HEAD_DIM_V1 UINT32_C(512)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_RAW_WINDOW_V1 UINT32_C(128)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_MAX_RATIO128_ROWS_V1 UINT32_C(2048)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_FAMILY_RAW_V1 UINT32_C(0)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_FAMILY_RATIO4_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_FAMILY_RATIO128_V1 UINT32_C(2)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_QUERY_BYTES_V1 UINT64_C(131072)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_RAW_BYTES_V1 UINT64_C(67584)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_SINK_BYTES_V1 UINT64_C(256)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_OUTPUT_BYTES_V1 UINT64_C(131072)

#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_MARKER_V1 (UINT64_C(1) << 62)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_RESERVED_BITS_V1 UINT32_C(12)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ERROR_BITS_V1 UINT32_C(4)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ELEMENT_BITS_V1 UINT32_C(10)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ROW_BITS_V1 UINT32_C(12)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_LAYER_BITS_V1 UINT32_C(6)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_POSITION_BITS_V1 UINT32_C(18)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ERROR_SHIFT_V1 UINT32_C(12)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ELEMENT_SHIFT_V1 UINT32_C(16)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ROW_SHIFT_V1 UINT32_C(26)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_LAYER_SHIFT_V1 UINT32_C(38)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_POSITION_SHIFT_V1 UINT32_C(44)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_RESERVED_MASK_V1 UINT64_C(0xfff)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ERROR_MASK_V1 UINT64_C(0xf)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ELEMENT_MASK_V1 UINT64_C(0x3ff)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ROW_MASK_V1 UINT64_C(0xfff)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_LAYER_MASK_V1 UINT64_C(0x3f)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_POSITION_MASK_V1 UINT64_C(0x3ffff)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ERROR_RECEIPT_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ERROR_SELECTED_ID_V1 UINT32_C(2)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ERROR_INVALID_CODE_V1 UINT32_C(3)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ERROR_INVALID_SCALE_V1 UINT32_C(4)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ERROR_NON_FINITE_QUERY_V1 UINT32_C(5)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ERROR_NON_FINITE_SINK_V1 UINT32_C(6)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ERROR_NON_FINITE_SCORE_V1 UINT32_C(7)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ERROR_NON_FINITE_OUTPUT_V1 UINT32_C(8)

#define AUGUR_CUDA_KV8_DSA_BASE_STRUCTS_V1_DEFINED 1

typedef struct augur_cuda_kv8_dsa_device_view_v1 {
    augur_cuda_device_ptr_t base;
    uint64_t allocation_extent_bytes;
    uint64_t view_offset_bytes;
    uint64_t view_extent_bytes;
} augur_cuda_kv8_dsa_device_view_v1;

typedef struct augur_cuda_kv8_dsa_row_v1 {
    uint32_t role;
    uint32_t values;
    augur_cuda_kv8_dsa_device_view_v1 source;
    augur_cuda_kv8_dsa_device_view_v1 destination;
} augur_cuda_kv8_dsa_row_v1;

typedef struct augur_cuda_kv8_dsa_quantize_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t row_count;
    uint32_t reset_status;
    uint32_t layer;
    uint32_t position;
    uint32_t reserved0;
    uint32_t reserved1;
    augur_cuda_kv8_dsa_device_view_v1 status;
    augur_cuda_kv8_dsa_row_v1 rows[AUGUR_CUDA_KV8_DSA_MAX_ROWS_V1];
} augur_cuda_kv8_dsa_quantize_args_v1;

/* Bounded prefill append ABI.  The logical owner may be 256K, but this
 * operation only covers one reusable 128-row physical work block. */
typedef struct augur_cuda_kv8_dsa_quantize_m128_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t family;
    uint32_t absolute_start;
    uint32_t rows;
    uint32_t reset_status;
    uint32_t reserved0;
    augur_cuda_kv8_dsa_device_view_v1 raw_source;
    augur_cuda_kv8_dsa_device_view_v1 raw_destination;
    augur_cuda_kv8_dsa_device_view_v1 compressed_main_source;
    augur_cuda_kv8_dsa_device_view_v1 compressed_main_destination;
    augur_cuda_kv8_dsa_device_view_v1 compressed_index_source;
    augur_cuda_kv8_dsa_device_view_v1 compressed_index_destination;
    augur_cuda_kv8_dsa_device_view_v1 status;
} augur_cuda_kv8_dsa_quantize_m128_args_v1;

typedef struct augur_cuda_kv8_dsa_status_copy_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_kv8_dsa_device_view_v1 status;
    uint64_t host_ptr;
    uint64_t host_len_bytes;
} augur_cuda_kv8_dsa_status_copy_args_v1;

typedef struct augur_cuda_kv8_dsa_select_receipt_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t position;
    uint32_t candidate_count;
    uint32_t top_k;
    uint32_t selected_count;
    uint32_t index_base;
    uint64_t reservation_epoch;
    int32_t padding;
    uint32_t reserved;
} augur_cuda_kv8_dsa_select_receipt_v1;

typedef struct augur_cuda_kv8_dsa_select_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t position;
    uint32_t candidate_count;
    uint32_t top_k;
    uint32_t index_base;
    int32_t padding;
    uint64_t reservation_epoch;
    uint32_t reserved0;
    uint32_t reserved1;
    augur_cuda_kv8_dsa_device_view_v1 query;
    augur_cuda_kv8_dsa_device_view_v1 index_rows;
    augur_cuda_kv8_dsa_device_view_v1 head_weight;
    augur_cuda_kv8_dsa_device_view_v1 scores;
    augur_cuda_kv8_dsa_device_view_v1 partial_ids;
    augur_cuda_kv8_dsa_device_view_v1 partial_scores;
    augur_cuda_kv8_dsa_device_view_v1 selected_ids;
    augur_cuda_kv8_dsa_device_view_v1 selected_scores;
    augur_cuda_kv8_dsa_device_view_v1 receipt;
    augur_cuda_kv8_dsa_device_view_v1 status;
} augur_cuda_kv8_dsa_select_args_v1;

typedef struct augur_cuda_kv8_dsa_attention_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t position;
    uint32_t family;
    uint32_t raw_rows_visible;
    uint32_t raw_start;
    uint32_t compressed_rows_visible;
    uint32_t selected_count;
    uint32_t top_k;
    int32_t index_base;
    int32_t padding;
    uint64_t reservation_epoch;
    uint32_t reserved0;
    uint32_t reserved1;
    augur_cuda_kv8_dsa_device_view_v1 query;
    augur_cuda_kv8_dsa_device_view_v1 raw_main;
    augur_cuda_kv8_dsa_device_view_v1 compressed_main;
    augur_cuda_kv8_dsa_device_view_v1 selected_ids;
    augur_cuda_kv8_dsa_device_view_v1 receipt;
    augur_cuda_kv8_dsa_device_view_v1 sink;
    augur_cuda_kv8_dsa_device_view_v1 weighted;
    augur_cuda_kv8_dsa_device_view_v1 inverse;
    augur_cuda_kv8_dsa_device_view_v1 status;
} augur_cuda_kv8_dsa_attention_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_quantize_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_quantize_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_quantize_m128_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_quantize_m128_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_status_copy_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_status_copy_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_select_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_select_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_attention_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_attention_args_v1* args
);

/*
 * CTX-2 device-resident Group-64 KV8 append ABI.
 *
 * Each block quantizes one prevalidated F32 source row into the frozen
 * values-first signed INT8 + little-endian FP16 scale-tail wire layout. The
 * sticky status is reset only by the layer-zero single-row launch and copied
 * to pinned host memory immediately before the owner's terminal boundary.
 */
#define AUGUR_CUDA_KV8_DSA_QUANTIZE_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_STATUS_COPY_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_MAX_ROWS_V1 UINT32_C(3)
#define AUGUR_CUDA_KV8_DSA_GROUP_VALUES_V1 UINT32_C(64)
#define AUGUR_CUDA_KV8_DSA_MAIN_VALUES_V1 UINT32_C(512)
#define AUGUR_CUDA_KV8_DSA_INDEX_VALUES_V1 UINT32_C(128)
#define AUGUR_CUDA_KV8_DSA_MAIN_WIRE_BYTES_V1 UINT32_C(528)
#define AUGUR_CUDA_KV8_DSA_INDEX_WIRE_BYTES_V1 UINT32_C(132)
#define AUGUR_CUDA_KV8_DSA_ROLE_RAW_MAIN_V1 UINT32_C(0)
#define AUGUR_CUDA_KV8_DSA_ROLE_RATIO4_MAIN_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_ROLE_RATIO4_INDEX_V1 UINT32_C(2)
#define AUGUR_CUDA_KV8_DSA_ROLE_RATIO128_MAIN_V1 UINT32_C(3)
#define AUGUR_CUDA_KV8_DSA_STATUS_OK_V1 UINT64_MAX
#define AUGUR_CUDA_KV8_DSA_STATUS_ERROR_BITS_V1 UINT32_C(2)
#define AUGUR_CUDA_KV8_DSA_STATUS_ELEMENT_BITS_V1 UINT32_C(9)
#define AUGUR_CUDA_KV8_DSA_STATUS_ROW_BITS_V1 UINT32_C(2)
#define AUGUR_CUDA_KV8_DSA_STATUS_LAYER_BITS_V1 UINT32_C(6)
#define AUGUR_CUDA_KV8_DSA_STATUS_POSITION_BITS_V1 UINT32_C(18)
#define AUGUR_CUDA_KV8_DSA_STATUS_ERROR_SHIFT_V1 UINT32_C(0)
#define AUGUR_CUDA_KV8_DSA_STATUS_ELEMENT_SHIFT_V1 UINT32_C(2)
#define AUGUR_CUDA_KV8_DSA_STATUS_ROW_SHIFT_V1 UINT32_C(11)
#define AUGUR_CUDA_KV8_DSA_STATUS_LAYER_SHIFT_V1 UINT32_C(13)
#define AUGUR_CUDA_KV8_DSA_STATUS_POSITION_SHIFT_V1 UINT32_C(19)
#define AUGUR_CUDA_KV8_DSA_STATUS_ERROR_MASK_V1 UINT64_C(0x3)
#define AUGUR_CUDA_KV8_DSA_STATUS_ELEMENT_MASK_V1 UINT64_C(0x1ff)
#define AUGUR_CUDA_KV8_DSA_STATUS_ROW_MASK_V1 UINT64_C(0x3)
#define AUGUR_CUDA_KV8_DSA_STATUS_LAYER_MASK_V1 UINT64_C(0x3f)
#define AUGUR_CUDA_KV8_DSA_STATUS_POSITION_MASK_V1 UINT64_C(0x3ffff)
#define AUGUR_CUDA_KV8_DSA_STATUS_USED_BITS_V1 UINT32_C(37)
#define AUGUR_CUDA_KV8_DSA_ERROR_NON_FINITE_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_ERROR_OUT_OF_RANGE_V1 UINT32_C(2)

/* CTX-3 device-resident Ratio4 score and stable Top-10 selection ABI. */
#define AUGUR_CUDA_KV8_DSA_SELECT_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_SELECT_RECEIPT_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_SELECT_HEADS_V1 UINT32_C(64)
#define AUGUR_CUDA_KV8_DSA_SELECT_HEAD_DIM_V1 UINT32_C(128)
#define AUGUR_CUDA_KV8_DSA_SELECT_TOP_K_V1 UINT32_C(10)
#define AUGUR_CUDA_KV8_DSA_SELECT_INDEX_BASE_V1 UINT32_C(128)
#define AUGUR_CUDA_KV8_DSA_SELECT_PADDING_V1 INT32_C(-1)
#define AUGUR_CUDA_KV8_DSA_SELECT_MAX_CANDIDATES_V1 UINT32_C(65536)
#define AUGUR_CUDA_KV8_DSA_SELECT_PARTITION_CANDIDATES_V1 UINT32_C(256)
#define AUGUR_CUDA_KV8_DSA_SELECT_MAX_PARTITIONS_V1 UINT32_C(256)
#define AUGUR_CUDA_KV8_DSA_SELECT_QUERY_BYTES_V1 UINT64_C(32768)
#define AUGUR_CUDA_KV8_DSA_SELECT_HEAD_WEIGHT_BYTES_V1 UINT64_C(256)
#define AUGUR_CUDA_KV8_DSA_SELECT_SCORES_BYTES_V1 UINT64_C(262144)
#define AUGUR_CUDA_KV8_DSA_SELECT_PARTIAL_IDS_BYTES_V1 UINT64_C(10240)
#define AUGUR_CUDA_KV8_DSA_SELECT_PARTIAL_SCORES_BYTES_V1 UINT64_C(10240)
#define AUGUR_CUDA_KV8_DSA_SELECT_IDS_BYTES_V1 UINT64_C(40)
#define AUGUR_CUDA_KV8_DSA_SELECT_SELECTED_SCORES_BYTES_V1 UINT64_C(40)
#define AUGUR_CUDA_KV8_DSA_SELECT_RECEIPT_BYTES_V1 UINT64_C(48)

#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_MARKER_V1 (UINT64_C(1) << 63)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_RESERVED_BITS_V1 UINT32_C(6)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_ERROR_BITS_V1 UINT32_C(3)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_ELEMENT_BITS_V1 UINT32_C(14)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_CANDIDATE_BITS_V1 UINT32_C(16)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_LAYER_BITS_V1 UINT32_C(6)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_POSITION_BITS_V1 UINT32_C(18)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_ERROR_SHIFT_V1 UINT32_C(6)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_ELEMENT_SHIFT_V1 UINT32_C(9)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_CANDIDATE_SHIFT_V1 UINT32_C(23)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_LAYER_SHIFT_V1 UINT32_C(39)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_POSITION_SHIFT_V1 UINT32_C(45)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_RESERVED_MASK_V1 UINT64_C(0x3f)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_ERROR_MASK_V1 UINT64_C(0x7)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_ELEMENT_MASK_V1 UINT64_C(0x3fff)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_CANDIDATE_MASK_V1 UINT64_C(0xffff)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_LAYER_MASK_V1 UINT64_C(0x3f)
#define AUGUR_CUDA_KV8_DSA_SELECT_STATUS_POSITION_MASK_V1 UINT64_C(0x3ffff)
#define AUGUR_CUDA_KV8_DSA_SELECT_ERROR_NON_FINITE_QUERY_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_SELECT_ERROR_NON_FINITE_HEAD_WEIGHT_V1 UINT32_C(2)
#define AUGUR_CUDA_KV8_DSA_SELECT_ERROR_INVALID_CODE_V1 UINT32_C(3)
#define AUGUR_CUDA_KV8_DSA_SELECT_ERROR_INVALID_SCALE_V1 UINT32_C(4)
#define AUGUR_CUDA_KV8_DSA_SELECT_ERROR_NON_FINITE_SCORE_V1 UINT32_C(5)

/* CTX-4 device-resident KV8 fused sparse-attention ABI. */
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_HEADS_V1 UINT32_C(64)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_HEAD_DIM_V1 UINT32_C(512)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_RAW_WINDOW_V1 UINT32_C(128)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_MAX_RATIO128_ROWS_V1 UINT32_C(2048)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_FAMILY_RAW_V1 UINT32_C(0)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_FAMILY_RATIO4_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_FAMILY_RATIO128_V1 UINT32_C(2)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_QUERY_BYTES_V1 UINT64_C(131072)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_RAW_BYTES_V1 UINT64_C(67584)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_SINK_BYTES_V1 UINT64_C(256)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_OUTPUT_BYTES_V1 UINT64_C(131072)

#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_MARKER_V1 (UINT64_C(1) << 62)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_RESERVED_BITS_V1 UINT32_C(12)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ERROR_BITS_V1 UINT32_C(4)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ELEMENT_BITS_V1 UINT32_C(10)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ROW_BITS_V1 UINT32_C(12)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_LAYER_BITS_V1 UINT32_C(6)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_POSITION_BITS_V1 UINT32_C(18)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ERROR_SHIFT_V1 UINT32_C(12)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ELEMENT_SHIFT_V1 UINT32_C(16)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ROW_SHIFT_V1 UINT32_C(26)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_LAYER_SHIFT_V1 UINT32_C(38)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_POSITION_SHIFT_V1 UINT32_C(44)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_RESERVED_MASK_V1 UINT64_C(0xfff)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ERROR_MASK_V1 UINT64_C(0xf)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ELEMENT_MASK_V1 UINT64_C(0x3ff)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_ROW_MASK_V1 UINT64_C(0xfff)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_LAYER_MASK_V1 UINT64_C(0x3f)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_STATUS_POSITION_MASK_V1 UINT64_C(0x3ffff)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ERROR_RECEIPT_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ERROR_SELECTED_ID_V1 UINT32_C(2)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ERROR_INVALID_CODE_V1 UINT32_C(3)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ERROR_INVALID_SCALE_V1 UINT32_C(4)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ERROR_NON_FINITE_QUERY_V1 UINT32_C(5)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ERROR_NON_FINITE_SINK_V1 UINT32_C(6)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ERROR_NON_FINITE_SCORE_V1 UINT32_C(7)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_ERROR_NON_FINITE_OUTPUT_V1 UINT32_C(8)

/* CTX-6R bounded Ratio128 split-row/split-K successor.  The v1 argument
 * remains the single-pass oracle; this ABI is deliberately additive. */
#define AUGUR_CUDA_KV8_DSA_ATTENTION_SPLIT_ARGS_VERSION_V2 UINT32_C(2)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_SPLIT_MIN_SPLITS_V2 UINT32_C(2)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_SPLIT_MAX_SPLITS_V2 UINT32_C(8)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_SPLIT_PARTIAL_FLOATS_V2 UINT32_C(514)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_SPLIT_PARTIAL_STRIDE_BYTES_V2 UINT64_C(2056)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_SPLIT_WORKSPACE_BYTES_V2 UINT64_C(1052672)
/* The v3 warp-tile entry points retain the v2 argument layout but admit a
 * wider component-only split topology. Their exact workspace extent is
 * heads * split_count * PARTIAL_STRIDE; this constant is only the admitted
 * maximum. The v2 control ABI and product allocation stay fixed. */
#define AUGUR_CUDA_KV8_DSA_ATTENTION_WARP_TILE_MAX_SPLITS_V3 UINT32_C(64)
#define AUGUR_CUDA_KV8_DSA_ATTENTION_WARP_TILE_WORKSPACE_BYTES_V3 UINT64_C(8421376)

#if !defined(AUGUR_CUDA_KV8_DSA_BASE_STRUCTS_V1_DEFINED)
typedef struct augur_cuda_kv8_dsa_device_view_v1 {
    augur_cuda_device_ptr_t base;
    uint64_t allocation_extent_bytes;
    uint64_t view_offset_bytes;
    uint64_t view_extent_bytes;
} augur_cuda_kv8_dsa_device_view_v1;

typedef struct augur_cuda_kv8_dsa_row_v1 {
    uint32_t role;
    uint32_t values;
    augur_cuda_kv8_dsa_device_view_v1 source;
    augur_cuda_kv8_dsa_device_view_v1 destination;
} augur_cuda_kv8_dsa_row_v1;

typedef struct augur_cuda_kv8_dsa_quantize_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t row_count;
    uint32_t reset_status;
    uint32_t layer;
    uint32_t position;
    uint32_t reserved0;
    uint32_t reserved1;
    augur_cuda_kv8_dsa_device_view_v1 status;
    augur_cuda_kv8_dsa_row_v1 rows[AUGUR_CUDA_KV8_DSA_MAX_ROWS_V1];
} augur_cuda_kv8_dsa_quantize_args_v1;

typedef struct augur_cuda_kv8_dsa_status_copy_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_kv8_dsa_device_view_v1 status;
    uint64_t host_ptr;
    uint64_t host_len_bytes;
} augur_cuda_kv8_dsa_status_copy_args_v1;

typedef struct augur_cuda_kv8_dsa_select_receipt_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t position;
    uint32_t candidate_count;
    uint32_t top_k;
    uint32_t selected_count;
    uint32_t index_base;
    uint64_t reservation_epoch;
    int32_t padding;
    uint32_t reserved;
} augur_cuda_kv8_dsa_select_receipt_v1;

typedef struct augur_cuda_kv8_dsa_select_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t position;
    uint32_t candidate_count;
    uint32_t top_k;
    uint32_t index_base;
    int32_t padding;
    uint64_t reservation_epoch;
    uint32_t reserved0;
    uint32_t reserved1;
    augur_cuda_kv8_dsa_device_view_v1 query;
    augur_cuda_kv8_dsa_device_view_v1 index_rows;
    augur_cuda_kv8_dsa_device_view_v1 head_weight;
    augur_cuda_kv8_dsa_device_view_v1 scores;
    augur_cuda_kv8_dsa_device_view_v1 partial_ids;
    augur_cuda_kv8_dsa_device_view_v1 partial_scores;
    augur_cuda_kv8_dsa_device_view_v1 selected_ids;
    augur_cuda_kv8_dsa_device_view_v1 selected_scores;
    augur_cuda_kv8_dsa_device_view_v1 receipt;
    augur_cuda_kv8_dsa_device_view_v1 status;
} augur_cuda_kv8_dsa_select_args_v1;
#endif

#define AUGUR_CUDA_KV8_DSA_SELECT_DP4A_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_SELECT_DP4A_QUERY_BYTES_V1 UINT64_C(8192)
#define AUGUR_CUDA_KV8_DSA_SELECT_DP4A_SCALE_BYTES_V1 UINT64_C(1024)

typedef struct augur_cuda_kv8_dsa_select_dp4a_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_kv8_dsa_select_args_v1 selection;
    augur_cuda_kv8_dsa_device_view_v1 q8_query;
    augur_cuda_kv8_dsa_device_view_v1 q8_scales;
} augur_cuda_kv8_dsa_select_dp4a_args_v1;

/* CTX-6R Ratio4 two-level resident selector.  These entry points are
 * additive component candidates; the v1 full-scan selector remains the
 * product control and fallback. */
#define AUGUR_CUDA_KV8_DSA_RATIO4_COARSE_BUILD_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_KV8_DSA_RATIO4_FINE_PER_COARSE_V1 UINT32_C(32)
#define AUGUR_CUDA_KV8_DSA_RATIO4_SELECTED_COARSE_V1 UINT32_C(16)
#define AUGUR_CUDA_KV8_DSA_RATIO4_MAX_COARSE_CANDIDATES_V1 UINT32_C(2048)
#define AUGUR_CUDA_KV8_DSA_RATIO4_MAX_FINE_RESCORE_V1 UINT32_C(512)
#define AUGUR_CUDA_KV8_DSA_RATIO4_FULL_SCAN_THRESHOLD_V1 UINT32_C(2048)
#define AUGUR_CUDA_KV8_DSA_RATIO4_COARSE_SCORES_BYTES_V1 UINT64_C(8192)
#define AUGUR_CUDA_KV8_DSA_RATIO4_SELECTED_COARSE_IDS_BYTES_V1 UINT64_C(64)
#define AUGUR_CUDA_KV8_DSA_RATIO4_FINE_SCORES_BYTES_V1 UINT64_C(2048)
#define AUGUR_CUDA_KV8_DSA_RATIO4_FINE_IDS_BYTES_V1 UINT64_C(2048)

typedef struct augur_cuda_kv8_dsa_ratio4_coarse_build_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t position;
    uint32_t coarse_group;
    uint32_t fine_count;
    uint32_t reserved0;
    uint32_t reserved1;
    augur_cuda_kv8_dsa_device_view_v1 fine_index_rows;
    augur_cuda_kv8_dsa_device_view_v1 coarse_index_row;
    augur_cuda_kv8_dsa_device_view_v1 status;
} augur_cuda_kv8_dsa_ratio4_coarse_build_args_v1;

#define AUGUR_CUDA_KV8_DSA_RATIO4_TWO_LEVEL_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_kv8_dsa_ratio4_two_level_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t position;
    uint32_t candidate_count;
    uint32_t top_k;
    uint32_t index_base;
    int32_t padding;
    uint64_t reservation_epoch;
    uint32_t coarse_count;
    uint32_t fine_per_coarse;
    uint32_t selected_coarse;
    uint32_t full_scan_threshold;
    uint32_t reserved0;
    uint32_t reserved1;
    augur_cuda_kv8_dsa_device_view_v1 query;
    augur_cuda_kv8_dsa_device_view_v1 fine_index_rows;
    augur_cuda_kv8_dsa_device_view_v1 coarse_index_rows;
    augur_cuda_kv8_dsa_device_view_v1 head_weight;
    augur_cuda_kv8_dsa_device_view_v1 coarse_scores;
    augur_cuda_kv8_dsa_device_view_v1 selected_coarse_ids;
    augur_cuda_kv8_dsa_device_view_v1 fine_scores;
    augur_cuda_kv8_dsa_device_view_v1 fine_ids;
    augur_cuda_kv8_dsa_device_view_v1 selected_ids;
    augur_cuda_kv8_dsa_device_view_v1 selected_scores;
    augur_cuda_kv8_dsa_device_view_v1 receipt;
    augur_cuda_kv8_dsa_device_view_v1 status;
    augur_cuda_kv8_dsa_device_view_v1 q8_query;
    augur_cuda_kv8_dsa_device_view_v1 q8_scales;
} augur_cuda_kv8_dsa_ratio4_two_level_args_v1;

#if !defined(AUGUR_CUDA_KV8_DSA_BASE_STRUCTS_V1_DEFINED)
typedef struct augur_cuda_kv8_dsa_attention_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t position;
    uint32_t family;
    uint32_t raw_rows_visible;
    uint32_t raw_start;
    uint32_t compressed_rows_visible;
    uint32_t selected_count;
    uint32_t top_k;
    int32_t index_base;
    int32_t padding;
    uint64_t reservation_epoch;
    uint32_t reserved0;
    uint32_t reserved1;
    augur_cuda_kv8_dsa_device_view_v1 query;
    augur_cuda_kv8_dsa_device_view_v1 raw_main;
    augur_cuda_kv8_dsa_device_view_v1 compressed_main;
    augur_cuda_kv8_dsa_device_view_v1 selected_ids;
    augur_cuda_kv8_dsa_device_view_v1 receipt;
    augur_cuda_kv8_dsa_device_view_v1 sink;
    augur_cuda_kv8_dsa_device_view_v1 weighted;
    augur_cuda_kv8_dsa_device_view_v1 inverse;
    augur_cuda_kv8_dsa_device_view_v1 status;
} augur_cuda_kv8_dsa_attention_args_v1;
#endif

typedef struct augur_cuda_kv8_dsa_attention_split_args_v2 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_kv8_dsa_attention_args_v1 base;
    augur_cuda_kv8_dsa_device_view_v1 workspace;
    uint32_t split_count;
    uint32_t reserved;
} augur_cuda_kv8_dsa_attention_split_args_v2;

#define AUGUR_CUDA_KV8_DSA_RATIO4_HOST_GATHER_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_kv8_dsa_ratio4_host_gather_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t position;
    uint32_t compressed_rows_visible;
    uint32_t selected_count;
    uint32_t top_k;
    int32_t index_base;
    int32_t padding;
    uint32_t reserved0;
    uint64_t reservation_epoch;
    augur_cuda_kv8_dsa_device_view_v1 host_main;
    augur_cuda_kv8_dsa_device_view_v1 selected_ids;
    augur_cuda_kv8_dsa_device_view_v1 receipt;
    augur_cuda_kv8_dsa_device_view_v1 packed_main;
    augur_cuda_kv8_dsa_device_view_v1 status;
} augur_cuda_kv8_dsa_ratio4_host_gather_args_v1;

#if !defined(AUGUR_CUDA_KV8_DSA_BASE_STRUCTS_V1_DEFINED)
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_quantize_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_quantize_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_status_copy_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_status_copy_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_select_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_select_args_v1* args
);
#endif
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_select_dp4a_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_select_dp4a_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_ratio4_coarse_build_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_ratio4_coarse_build_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_ratio4_two_level_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_ratio4_two_level_args_v1* args
);
#if defined(AUGUR_CUDA_CTX31_RATIO4_COOPERATIVE_SELECTOR_CANDIDATE)
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_kv8_dsa_ratio4_two_level_cooperative_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_ratio4_two_level_args_v1* args
);
#endif
#if !defined(AUGUR_CUDA_KV8_DSA_BASE_STRUCTS_V1_DEFINED)
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_attention_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_attention_args_v1* args
);
#endif
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_attention_split_submit_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_attention_split_args_v2* args
);
#if defined(AUGUR_CUDA_KV8_DSA_RATIO4_HOST_TIER_CANDIDATE)
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_ratio4_host_gather_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_ratio4_host_gather_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_kv8_dsa_attention_ratio4_host_packed_warp2_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_attention_split_args_v2* args
);
#endif
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_attention_warp2_split_submit_v3(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_attention_split_args_v2* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_attention_warp4_split_submit_v3(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_attention_split_args_v2* args
);
#if defined(AUGUR_CUDA_CTX13_RATIO128_HEADS8_CANDIDATE)
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_attention_warp8_split_submit_v3(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_attention_split_args_v2* args
);
#endif
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_attention_warp2_family_split_submit_v4(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_attention_split_args_v2* args
);
#if defined(AUGUR_CUDA_CTX32_KV8_DSA_EXACT_H2_CANDIDATE)
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_attention_exact_h2_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_attention_args_v1* args
);
#endif
#if defined(AUGUR_CUDA_CTX33_KV8_DSA_EXACT_ROW_STAGE_CANDIDATE)
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_attention_exact_row_stage_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_attention_args_v1* args
);
#endif
/*
 * Candidate-only submit entry points. The Rust binder performs the complete
 * device ownership, exact extent, range, alignment and pairwise non-alias
 * audit before calling these functions. Native code retains cheap ABI and
 * metadata checks, but intentionally does not call cudaPointerGetAttributes
 * or cudaSetDevice on the hot submit path. Every referenced allocation and
 * the stream must remain valid until the enclosing asynchronous completion.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_select_submit_prevalidated_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_select_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_select_dp4a_submit_prevalidated_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_select_dp4a_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_ratio4_coarse_build_submit_prevalidated_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_ratio4_coarse_build_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_ratio4_two_level_submit_prevalidated_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_ratio4_two_level_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_attention_submit_prevalidated_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_attention_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_attention_split_submit_prevalidated_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_attention_split_args_v2* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_attention_warp2_split_submit_prevalidated_v3(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_attention_split_args_v2* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_attention_warp4_split_submit_prevalidated_v3(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_attention_split_args_v2* args
);
#if defined(AUGUR_CUDA_CTX13_RATIO128_HEADS8_CANDIDATE)
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_attention_warp8_split_submit_prevalidated_v3(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_attention_split_args_v2* args
);
#endif
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_kv8_dsa_attention_warp2_family_split_submit_prevalidated_v4(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_kv8_dsa_attention_split_args_v2* args
);

#define AUGUR_CUDA_M283_KV_TRANSITION_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_M283_KV_CAPACITY_V1 UINT32_C(128)
#define AUGUR_CUDA_M283_KV_VARIANT_RAW_V1 UINT32_C(0)
#define AUGUR_CUDA_M283_KV_VARIANT_RATIO4_V1 UINT32_C(1)
#define AUGUR_CUDA_M283_KV_VARIANT_RATIO128_V1 UINT32_C(2)
typedef struct augur_cuda_m283_device_view_v1 {
    augur_cuda_device_ptr_t base;
    uint64_t allocation_extent_bytes;
    uint64_t view_offset_bytes;
    uint64_t view_extent_bytes;
} augur_cuda_m283_device_view_v1;
typedef struct augur_cuda_m283_kv_transition_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t variant;
    uint32_t reserved0;
    uint32_t layer;
    uint32_t capacity;
    uint32_t position;
    uint32_t next_position;
    uint32_t compression_ratio;
    uint32_t source_start_position;
    uint32_t source_rows;
    uint32_t compressed_row_index;
    uint32_t compressed_position;
    uint32_t compressed_main_rows_before;
    uint32_t compressed_index_rows_before;
    uint32_t reserved1;
    augur_cuda_m283_device_view_v1 raw_current;
    augur_cuda_m283_device_view_v1 raw_cache;
    augur_cuda_m283_device_view_v1 compressed_main_current;
    augur_cuda_m283_device_view_v1 compressed_main_cache;
    augur_cuda_m283_device_view_v1 compressed_index_current;
    augur_cuda_m283_device_view_v1 compressed_index_cache;
} augur_cuda_m283_kv_transition_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m283_validate_kv_transition_v1(
    const augur_cuda_m283_kv_transition_args_v1* args
);

#define AUGUR_CUDA_M285_FILL_SPAN_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m285_fill_span_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t allocation_base;
    uint64_t allocation_extent_bytes;
    uint64_t view_offset_bytes;
    uint64_t view_extent_bytes;
    uint64_t pattern_seed;
    uint64_t reserved;
} augur_cuda_m285_fill_span_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m285_fill_span_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m285_fill_span_args_v1* args
);

#define AUGUR_CUDA_M292_HBM_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_M292_HBM_READ_ONLY UINT32_C(1)
#define AUGUR_CUDA_M292_HBM_READ_WRITE UINT32_C(2)
typedef struct augur_cuda_m292_hbm_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t operation;
    uint32_t blocks;
    augur_cuda_device_ptr_t source_base;
    uint64_t source_allocation_extent_bytes;
    uint64_t source_view_offset_bytes;
    uint64_t source_view_extent_bytes;
    augur_cuda_device_ptr_t destination_base;
    uint64_t destination_allocation_extent_bytes;
    uint64_t destination_view_offset_bytes;
    uint64_t destination_view_extent_bytes;
    uint32_t repeats;
    uint32_t reserved;
} augur_cuda_m292_hbm_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m292_hbm_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m292_hbm_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m292_l2_cache_bytes_v1(
    int32_t ordinal,
    uint64_t* out_l2_cache_bytes
);

#define AUGUR_CUDA_M306_K7_HC_FP64_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_M306_K7_HC_HIDDEN_V1 UINT32_C(16384)
#define AUGUR_CUDA_M306_K7_HC_OUTPUTS_V1 UINT32_C(24)
#define AUGUR_CUDA_M306_K7_HC_CHUNKS_V1 UINT32_C(64)
#define AUGUR_CUDA_M306_K7_HC_THREADS_V1 UINT32_C(256)
#define AUGUR_CUDA_M306_K7_HC_MAX_ROWS_V1 UINT32_C(65535)
#define AUGUR_CUDA_M306_K7_HC_SCRATCH_ALIGNMENT_V1 UINT64_C(8)
#define AUGUR_CUDA_M306_K7_HC_HIDDEN_ROW_BYTES_V1 UINT64_C(65536)
#define AUGUR_CUDA_M306_K7_HC_FUNCTION_BYTES_V1 UINT64_C(1572864)
#define AUGUR_CUDA_M306_K7_HC_MIX_ROW_BYTES_V1 UINT64_C(96)
#define AUGUR_CUDA_M306_K7_HC_SCRATCH_ROW_BYTES_V1 UINT64_C(12800)

typedef struct augur_cuda_m306_k7_hc_fp64_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t rows;
    uint32_t hidden_elements;
    uint32_t outputs;
    uint32_t chunks;
    uint32_t threads;
    uint32_t reserved;
    augur_cuda_device_ptr_t hidden;
    uint64_t hidden_len_bytes;
    augur_cuda_device_ptr_t function;
    uint64_t function_len_bytes;
    augur_cuda_device_ptr_t mix;
    uint64_t mix_len_bytes;
    augur_cuda_device_ptr_t scratch;
    uint64_t scratch_len_bytes;
} augur_cuda_m306_k7_hc_fp64_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k7_hc_fp64_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k7_hc_fp64_args_v1* args
);

#define AUGUR_CUDA_M306_K11_HC_EXACT_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_M306_K11_HC_EXACT_HIDDEN_V1 UINT32_C(16384)
#define AUGUR_CUDA_M306_K11_HC_EXACT_OUTPUTS_V1 UINT32_C(24)
#define AUGUR_CUDA_M306_K11_HC_EXACT_ACCUMULATORS_V1 UINT32_C(25)
#define AUGUR_CUDA_M306_K11_HC_EXACT_THREADS_V1 UINT32_C(32)
#define AUGUR_CUDA_M306_K11_HC_EXACT_MAX_ROWS_V1 UINT32_C(65535)
#define AUGUR_CUDA_M306_K11_HC_EXACT_SCRATCH_ALIGNMENT_V1 UINT64_C(4)
#define AUGUR_CUDA_M306_K11_HC_EXACT_HIDDEN_ROW_BYTES_V1 UINT64_C(65536)
#define AUGUR_CUDA_M306_K11_HC_EXACT_FUNCTION_BYTES_V1 UINT64_C(1572864)
#define AUGUR_CUDA_M306_K11_HC_EXACT_MIX_ROW_BYTES_V1 UINT64_C(96)
#define AUGUR_CUDA_M306_K11_HC_EXACT_SCRATCH_ROW_BYTES_V1 UINT64_C(100)
#define AUGUR_CUDA_M306_K11_HC_EXACT_FAULT_ACCUMULATOR_SUBMIT_V1 UINT32_C(0)
#define AUGUR_CUDA_M306_K11_HC_EXACT_FAULT_AFTER_ACCUMULATOR_COMPLETE_V1 UINT32_C(1)
#define AUGUR_CUDA_M306_K11_HC_EXACT_FAULT_FINALIZER_SUBMIT_V1 UINT32_C(2)
#define AUGUR_CUDA_M306_K11_HC_EXACT_FAULT_AFTER_FINALIZER_COMPLETE_V1 UINT32_C(3)
#define AUGUR_CUDA_M306_K11_HC_EXACT_PROGRESS_ACCUMULATOR_SUBMITTED_V1 UINT32_C(1)
#define AUGUR_CUDA_M306_K11_HC_EXACT_PROGRESS_ACCUMULATOR_COMPLETED_V1 UINT32_C(2)
#define AUGUR_CUDA_M306_K11_HC_EXACT_PROGRESS_FINALIZER_SUBMITTED_V1 UINT32_C(4)
#define AUGUR_CUDA_M306_K11_HC_EXACT_PROGRESS_FINALIZER_COMPLETED_V1 UINT32_C(8)

typedef struct augur_cuda_m306_k11_hc_exact_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t rows;
    uint32_t hidden_elements;
    uint32_t outputs;
    uint32_t accumulators;
    uint32_t threads;
    uint32_t reserved;
    augur_cuda_device_ptr_t hidden;
    uint64_t hidden_len_bytes;
    augur_cuda_device_ptr_t function;
    uint64_t function_len_bytes;
    augur_cuda_device_ptr_t mix;
    uint64_t mix_len_bytes;
    augur_cuda_device_ptr_t scratch;
    uint64_t scratch_len_bytes;
} augur_cuda_m306_k11_hc_exact_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k11_hc_exact_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k11_hc_exact_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k25_hc_exact_vec8_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k11_hc_exact_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k11_hc_exact_fault_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k11_hc_exact_args_v1* args,
    uint32_t fault_phase,
    uint32_t* native_progress
);

#define AUGUR_CUDA_M306_K23_HC_LANE_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_M306_K23_HC_LANE_HIDDEN_V1 UINT32_C(16384)
#define AUGUR_CUDA_M306_K23_HC_LANE_OUTPUTS_V1 UINT32_C(24)
#define AUGUR_CUDA_M306_K23_HC_LANE_ACCUMULATORS_V1 UINT32_C(25)
#define AUGUR_CUDA_M306_K23_HC_LANE_THREADS_V1 UINT32_C(32)
#define AUGUR_CUDA_M306_K23_HC_LANE_FUNCTION_LAYOUT_SOURCE_MAJOR_V1 UINT32_C(1)
#define AUGUR_CUDA_M306_K23_HC_LANE_MAX_ROWS_V1 UINT32_C(65535)
#define AUGUR_CUDA_M306_K23_HC_LANE_HIDDEN_ROW_BYTES_V1 UINT64_C(65536)
#define AUGUR_CUDA_M306_K23_HC_LANE_FUNCTION_BYTES_V1 UINT64_C(1572864)
#define AUGUR_CUDA_M306_K23_HC_LANE_MIX_ROW_BYTES_V1 UINT64_C(96)
#define AUGUR_CUDA_M306_K23_HC_LANE_SCRATCH_ROW_BYTES_V1 UINT64_C(100)
#define AUGUR_CUDA_M306_K23_HC_LANE_SCRATCH_ALIGNMENT_V1 UINT64_C(4)

typedef struct augur_cuda_m306_k23_hc_lane_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t rows;
    uint32_t hidden_elements;
    uint32_t outputs;
    uint32_t accumulators;
    uint32_t threads;
    uint32_t function_layout;
    augur_cuda_device_ptr_t hidden;
    uint64_t hidden_len_bytes;
    augur_cuda_device_ptr_t function;
    uint64_t function_len_bytes;
    augur_cuda_device_ptr_t mix;
    uint64_t mix_len_bytes;
    augur_cuda_device_ptr_t scratch;
    uint64_t scratch_len_bytes;
} augur_cuda_m306_k23_hc_lane_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k23_hc_lane_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k23_hc_lane_args_v1* args
);

#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_MAX_TOKENS_V1 UINT32_C(8)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_HIDDEN_V1 UINT32_C(4096)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_EXPERTS_V1 UINT32_C(256)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_TOP_K_V1 UINT32_C(6)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_TOPOLOGY_VECTORIZED_1X256_V1 UINT32_C(0)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_TOPOLOGY_VECTORIZED_2X128_V1 UINT32_C(1)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_TOPOLOGY_VECTORIZED_4X64_V1 UINT32_C(2)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_TOPOLOGY_VECTORIZED_8X32_V1 UINT32_C(3)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_TOPOLOGY_ROW_BLOCK_256X32_LANE0_V1 UINT32_C(4)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_DENOMINATOR_ROUTE_RANK_V1 UINT32_C(0)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_COMPENSATED_DENOMINATOR_V1 UINT32_C(0)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_VECTOR_ALIGNMENT_V1 UINT64_C(16)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_WEIGHT_BYTES_V1 UINT64_C(2097152)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_INPUT_ROW_BYTES_V1 UINT64_C(16384)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_BIAS_BYTES_V1 UINT64_C(1024)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_SCRATCH_ROW_BYTES_V1 UINT64_C(1024)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_LOGITS_ROW_BYTES_V1 UINT64_C(1024)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_IDS_ROW_BYTES_V1 UINT64_C(12)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_WEIGHTS_ROW_BYTES_V1 UINT64_C(24)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_STATUS_ROW_BYTES_V1 UINT64_C(4)

typedef struct augur_cuda_m306_k12_learned_router_exact_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t tokens;
    uint32_t hidden;
    uint32_t experts;
    uint32_t top_k;
    uint32_t topology;
    uint32_t denominator_order;
    uint32_t compensated_denominator;
    uint32_t reserved0;
    float routed_scale;
    uint32_t reserved1;
    augur_cuda_device_ptr_t weight_bf16;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t bias_f32;
    uint64_t bias_len_bytes;
    augur_cuda_device_ptr_t scratch_f32;
    uint64_t scratch_len_bytes;
    augur_cuda_device_ptr_t logits_f32;
    uint64_t logits_len_bytes;
    augur_cuda_device_ptr_t output_ids_u16;
    uint64_t output_ids_len_bytes;
    augur_cuda_device_ptr_t output_weights_f32;
    uint64_t output_weights_len_bytes;
    augur_cuda_device_ptr_t status_i32;
    uint64_t status_len_bytes;
} augur_cuda_m306_k12_learned_router_exact_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k12_learned_router_exact_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k12_learned_router_exact_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v1* args
);

/*
 * The safe Rust binding preflights once at bind time and repeats the audit in
 * its safe submit/v1 methods. Only callers that separately prove all bound
 * allocations and the stream remain live may use the prevalidated entry
 * points. Direct C callers should use the fully audited entry points above.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k12_learned_router_exact_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k12_learned_router_exact_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k12_learned_router_exact_prevalidated_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v1* args
);

#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_ARGS_VERSION_V2 UINT32_C(2)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_MAX_TOKENS_V2 UINT32_C(8)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_HIDDEN_V2 UINT32_C(4096)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_EXPERTS_V2 UINT32_C(256)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_TOP_K_V2 UINT32_C(6)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_TOPOLOGY_ROW_BLOCK_256X32_LANE0_V2 UINT32_C(4)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_DENOMINATOR_ROUTE_RANK_V2 UINT32_C(0)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_UNCOMPENSATED_DENOMINATOR_V2 UINT32_C(0)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_COMPENSATED_DENOMINATOR_V2 UINT32_C(1)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_VECTOR_ALIGNMENT_V2 UINT64_C(16)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_WEIGHT_BYTES_V2 UINT64_C(2097152)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_INPUT_ROW_BYTES_V2 UINT64_C(16384)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_BIAS_BYTES_V2 UINT64_C(1024)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_SCRATCH_ROW_BYTES_V2 UINT64_C(1024)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_LOGITS_ROW_BYTES_V2 UINT64_C(1024)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_IDS_ROW_BYTES_V2 UINT64_C(12)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_WEIGHTS_ROW_BYTES_V2 UINT64_C(24)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_STATUS_ROW_BYTES_V2 UINT64_C(4)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_FAULT_ACCUMULATOR_SUBMIT_V2 UINT32_C(0)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_FAULT_AFTER_ACCUMULATOR_COMPLETE_V2 UINT32_C(1)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_FAULT_FINALIZER_SUBMIT_V2 UINT32_C(2)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_FAULT_AFTER_FINALIZER_COMPLETE_V2 UINT32_C(3)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_PROGRESS_ACCUMULATOR_SUBMITTED_V2 UINT32_C(1)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_PROGRESS_ACCUMULATOR_COMPLETED_V2 UINT32_C(2)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_PROGRESS_FINALIZER_SUBMITTED_V2 UINT32_C(4)
#define AUGUR_CUDA_M306_K12_LEARNED_ROUTER_EXACT_PROGRESS_FINALIZER_COMPLETED_V2 UINT32_C(8)

typedef struct augur_cuda_m306_k12_learned_router_exact_args_v2 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t tokens;
    uint32_t hidden;
    uint32_t experts;
    uint32_t top_k;
    uint32_t topology;
    uint32_t denominator_order;
    uint32_t compensated_denominator;
    uint32_t reserved0;
    float routed_scale;
    uint32_t reserved1;
    augur_cuda_device_ptr_t weight_bf16;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t bias_f32;
    uint64_t bias_len_bytes;
    augur_cuda_device_ptr_t scratch_f32;
    uint64_t scratch_len_bytes;
    augur_cuda_device_ptr_t logits_f32;
    uint64_t logits_len_bytes;
    augur_cuda_device_ptr_t output_ids_u16;
    uint64_t output_ids_len_bytes;
    augur_cuda_device_ptr_t output_weights_f32;
    uint64_t output_weights_len_bytes;
    augur_cuda_device_ptr_t status_i32;
    uint64_t status_len_bytes;
} augur_cuda_m306_k12_learned_router_exact_args_v2;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k12_learned_router_exact_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v2* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k12_learned_router_exact_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v2* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k12_learned_router_exact_preflight_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v2* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k12_learned_router_exact_prevalidated_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v2* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k12_learned_router_exact_prevalidated_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v2* args
);

/*
 * M306-K27 reuses the frozen K12 V2 argument layout and accumulator, but
 * replaces only the final Top-K selection kernel. The historical K12 symbols
 * remain unchanged.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k27_learned_router_exact_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v2* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_m306_k27_learned_router_exact_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v2* args
);

/*
 * ToC B=1 fast successor. The K12 V2 argument/ownership contract is reused,
 * while one warp cooperates on each expert dot product with fixed FP32
 * accumulation and reduction. Historical K12/K27 symbols remain unchanged.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_toc_fast_learned_router_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v2* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_toc_fast_learned_router_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v2* args
);

/*
 * BATCH-01 qualification-only exact successor. Exactly four token rows are
 * mapped to lanes 0..3 of one warp per expert so the lanes issue the same
 * router-weight loads while preserving each row's selected K12 accumulation
 * order. The selected K27 finalizer and the K12 V2 argument layout are reused.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_learned_router_shared_weight_lane4_exact_k27_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v2* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_learned_router_shared_weight_lane4_exact_k27_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v2* args
);

/* Diagnostic-only. Production submit ABIs remain asynchronous and unchanged. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k12_learned_router_exact_fault_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k12_learned_router_exact_args_v2* args,
    uint32_t fault_phase,
    uint32_t* native_progress
);

#define AUGUR_CUDA_M304_R3_GREEDY_RECEIPT_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_M304_R3_VOCAB_V1 UINT32_C(129280)
#define AUGUR_CUDA_M304_R3_PARTIAL_COUNT_V1 UINT32_C(128)
#define AUGUR_CUDA_M304_R3_THREADS_V1 UINT32_C(256)
#define AUGUR_CUDA_M304_R3_LOGITS_BYTES_V1 UINT64_C(517120)
#define AUGUR_CUDA_M304_R3_PARTIAL_BYTES_V1 UINT64_C(4096)
#define AUGUR_CUDA_M304_R3_RECEIPT_BYTES_V1 UINT64_C(64)
#define AUGUR_CUDA_M304_R3_RECEIPT_STATUS_UNWRITTEN_V1 UINT32_C(0)
#define AUGUR_CUDA_M304_R3_RECEIPT_STATUS_OK_V1 UINT32_C(1)
#define AUGUR_CUDA_M304_R3_RECEIPT_STATUS_ERROR_V1 UINT32_C(2)
#define AUGUR_CUDA_M304_R3_RECEIPT_ERROR_NONE_V1 UINT32_C(0)
#define AUGUR_CUDA_M304_R3_RECEIPT_ERROR_NON_FINITE_LOGIT_V1 UINT32_C(1)

typedef struct augur_cuda_m304_r3_greedy_receipt_v1 {
    uint32_t token_id;
    uint32_t top1_value_bits;
    uint32_t runner_up_id;
    uint32_t runner_up_value_bits;
    uint32_t status;
    uint32_t error_code;
    uint64_t step_generation;
    uint8_t plan_digest[32];
} augur_cuda_m304_r3_greedy_receipt_v1;

typedef struct augur_cuda_m304_r3_greedy_receipt_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t logits;
    uint64_t logits_len_bytes;
    augur_cuda_device_ptr_t partials;
    uint64_t partials_len_bytes;
    augur_cuda_device_ptr_t receipt;
    uint64_t receipt_len_bytes;
    uint64_t step_generation;
    uint32_t vocab;
    uint32_t partial_count;
    uint8_t plan_digest[32];
} augur_cuda_m304_r3_greedy_receipt_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m304_r3_greedy_receipt_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_greedy_receipt_args_v1* args
);

#define AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_VOCAB_V1 UINT32_C(248320)
#define AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_PARTIAL_COUNT_V1 UINT32_C(256)
#define AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_THREADS_V1 UINT32_C(256)
#define AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_LOGITS_BYTES_V1 UINT64_C(993280)
#define AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_PARTIAL_BYTES_V1 UINT64_C(8192)
#define AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES_V1 UINT64_C(72)
#define AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_STATUS_UNWRITTEN_V1 UINT32_C(0)
#define AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_STATUS_OK_V1 UINT32_C(1)
#define AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_STATUS_ERROR_V1 UINT32_C(2)
#define AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ERROR_NONE_V1 UINT32_C(0)
#define AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ERROR_NON_FINITE_LOGIT_V1 UINT32_C(1)
#define AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ERROR_INCOMPLETE_REDUCTION_V1 UINT32_C(2)

typedef struct augur_cuda_qwen35_token_handoff_receipt_v1 {
    uint32_t token_id;
    uint32_t top1_value_bits;
    uint32_t runner_up_id;
    uint32_t runner_up_value_bits;
    uint32_t finite_count;
    uint32_t first_non_finite_id;
    uint32_t status;
    uint32_t error_code;
    uint64_t step_generation;
    uint8_t plan_digest[32];
} augur_cuda_qwen35_token_handoff_receipt_v1;

typedef struct augur_cuda_qwen35_token_handoff_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t logits;
    uint64_t logits_len_bytes;
    augur_cuda_device_ptr_t partials;
    uint64_t partials_len_bytes;
    augur_cuda_device_ptr_t receipt;
    uint64_t receipt_len_bytes;
    uint64_t step_generation;
    uint32_t vocab;
    uint32_t partial_count;
    uint8_t plan_digest[32];
} augur_cuda_qwen35_token_handoff_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_token_handoff_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_token_handoff_args_v1* args
);

#define AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS_V1 UINT32_C(8)
#define AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_VOCAB_V1 UINT32_C(248320)
#define AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES_V1 UINT64_C(64)
#define AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_STATUS_UNWRITTEN_V1 UINT32_C(0)
#define AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_STATUS_OK_V1 UINT32_C(1)
#define AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_STATUS_ERROR_V1 UINT32_C(2)
#define AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ERROR_NONE_V1 UINT32_C(0)
#define AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ERROR_INVALID_TOKEN_V1 UINT32_C(1)
#define AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ERROR_NON_FINITE_LOGIT_V1 UINT32_C(2)

typedef struct augur_cuda_qwen35_dflash_acceptance_receipt_v1 {
    uint32_t draft_tokens;
    uint32_t accepted_tokens;
    uint32_t rejected_position;
    uint32_t replacement_token;
    uint32_t status;
    uint32_t error_code;
    uint64_t step_generation;
    uint8_t plan_digest[32];
} augur_cuda_qwen35_dflash_acceptance_receipt_v1;

typedef struct augur_cuda_qwen35_dflash_acceptance_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t draft_tokens;
    uint32_t reserved0;
    augur_cuda_device_ptr_t draft_ids;
    uint64_t draft_ids_len_bytes;
    augur_cuda_device_ptr_t target_ids;
    uint64_t target_ids_len_bytes;
    augur_cuda_device_ptr_t receipt;
    uint64_t receipt_len_bytes;
    uint64_t step_generation;
    uint8_t plan_digest[32];
} augur_cuda_qwen35_dflash_acceptance_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_acceptance_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_acceptance_args_v1* args
);

/*
 * Device-side target verification for a DFlash draft.  `target_logits` is a
 * row-major [draft_tokens, vocab] buffer and `target_ids` is a device-only
 * scratch/output buffer.  The operation computes deterministic greedy target
 * ids (lowest token id wins ties), then applies the same prefix acceptance
 * rule as augur_cuda_qwen35_dflash_acceptance_submit_v1.  Only the compact
 * acceptance receipt is intended to cross the host boundary.
 */
#define AUGUR_CUDA_QWEN35_DFLASH_LOGITS_ACCEPTANCE_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_qwen35_dflash_logits_acceptance_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t draft_tokens;
    uint32_t reserved0;
    augur_cuda_device_ptr_t draft_ids;
    uint64_t draft_ids_len_bytes;
    augur_cuda_device_ptr_t target_logits;
    uint64_t target_logits_len_bytes;
    augur_cuda_device_ptr_t target_ids;
    uint64_t target_ids_len_bytes;
    augur_cuda_device_ptr_t receipt;
    uint64_t receipt_len_bytes;
    uint64_t step_generation;
    uint32_t vocab;
    uint32_t reserved1;
    uint8_t plan_digest[32];
} augur_cuda_qwen35_dflash_logits_acceptance_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_logits_acceptance_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_logits_acceptance_args_v1* args
);

/* Full-vocabulary greedy selection (no causal-prefix masking). A nonfinite
 * logit anywhere in a row produces UINT32_MAX; ties select the lowest id. */
typedef struct augur_cuda_qwen35_vocab_top1_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t rows;
    uint32_t vocab;
    augur_cuda_device_ptr_t logits;
    uint64_t logits_len_bytes;
    augur_cuda_device_ptr_t ids;
    uint64_t ids_len_bytes;
} augur_cuda_qwen35_vocab_top1_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_vocab_top1_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_vocab_top1_args_v1* args
);

/* Exact full-vocabulary top-k, k=1..16. Row-major ids and raw unary scores.
 * Finite ties (including signed zero) use lowest token id. Any NaN/Inf
 * invalidates every entry of that row: UINT32_MAX ids and NaN scores. */
typedef struct augur_cuda_qwen35_vocab_topk_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t rows;
    uint32_t vocab;
    uint32_t top_k;
    uint32_t reserved;
    augur_cuda_device_ptr_t logits;
    uint64_t logits_len_bytes;
    augur_cuda_device_ptr_t ids;
    uint64_t ids_len_bytes;
    augur_cuda_device_ptr_t scores;
    uint64_t scores_len_bytes;
} augur_cuda_qwen35_vocab_topk_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_vocab_topk_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_vocab_topk_args_v1* args
);

/* Optional disjoint caller-retained tile scratch: rows * parts * top_k * 8.
 * top_k is 16 or 20; parts is 8, 16, 32 or 64. Original v1 stays unchanged. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_vocab_topk_tiled_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_vocab_topk_args_v1* args,
    uint64_t scratch, uint64_t scratch_len_bytes, uint32_t parts
);

/* DFlash2 F16-weight/FP32-accumulation selector. Exact profile: 8 rows,
 * K16, rank256, hidden5120, vocab248320. Path[0] is the supplied anchor;
 * path[1..8] is a greedy codebook-conditioned chain (p_min=0).
 * Invalid data publishes UINT32_MAX in every path slot, never a prefix. */
typedef struct augur_cuda_qwen35_dflash_selector_args_v1 {
    uint32_t abi_version, struct_size, rows, top_k, rank, hidden, vocab, anchor_token;
    augur_cuda_device_ptr_t hidden_rows; uint64_t hidden_rows_len_bytes;
    augur_cuda_device_ptr_t gate_weight; uint64_t gate_weight_len_bytes;
    augur_cuda_device_ptr_t predecessor; uint64_t predecessor_len_bytes;
    augur_cuda_device_ptr_t successor; uint64_t successor_len_bytes;
    augur_cuda_device_ptr_t candidate_ids; uint64_t candidate_ids_len_bytes;
    augur_cuda_device_ptr_t unary_scores; uint64_t unary_scores_len_bytes;
    augur_cuda_device_ptr_t gates; uint64_t gates_len_bytes;
    augur_cuda_device_ptr_t transition_scores; uint64_t transition_scores_len_bytes;
    augur_cuda_device_ptr_t path_ids; uint64_t path_ids_len_bytes;
} augur_cuda_qwen35_dflash_selector_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_selector_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_selector_args_v1* args
);

/* Q8_0 codebooks + already projected F32 gates. No hidden projection or
 * dequantized full-codebook copy. Same ordered greedy path/sentinel contract. */
typedef struct augur_cuda_qwen35_dflash_q8_selector_args_v1 {
    uint32_t abi_version, struct_size, rows, top_k, rank, vocab, anchor_token, reserved;
    augur_cuda_device_ptr_t predecessor; uint64_t predecessor_len_bytes;
    augur_cuda_device_ptr_t successor; uint64_t successor_len_bytes;
    augur_cuda_device_ptr_t candidate_ids; uint64_t candidate_ids_len_bytes;
    augur_cuda_device_ptr_t unary_scores; uint64_t unary_scores_len_bytes;
    augur_cuda_device_ptr_t gates; uint64_t gates_len_bytes;
    augur_cuda_device_ptr_t transition_scores; uint64_t transition_scores_len_bytes;
    augur_cuda_device_ptr_t path_ids; uint64_t path_ids_len_bytes;
} augur_cuda_qwen35_dflash_q8_selector_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_bf16_codebooks_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_q8_selector_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_phase_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_dflash_q8_selector_args_v1* args,
    uint64_t after_gates, uint64_t after_bytes, uint32_t close_token);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_q8_selector_args_v1* args
);
// Explicit experimental transition scale; original v1 descriptor and default remain unchanged.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_scaled_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_q8_selector_args_v1* args, float transition_scale
);
// Experimental dynamic-programming selector path. The descriptor and Q8
// candidate geometry remain identical to the greedy selector; the path is
// selected by a separate symbol so the default kernel/ABI is unchanged.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_dp_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_q8_selector_args_v1* args, float transition_scale
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_tail_scaled_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_q8_selector_args_v1* args, float transition_scale
);
// Opt-in independent scales for rows 1/2/3; rows 4-7 retain scale 1.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_position_scaled_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_q8_selector_args_v1* args,
    float first_scale, float second_scale, float third_scale
);
// Opt-in early-position calibration: apply transition scale only to rows 1-3.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_early_scaled_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_q8_selector_args_v1* args, float transition_scale
);
// Experimental sparse proposal q, eight rows of sixteen FP64 probabilities.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_probability_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_q8_selector_args_v1* args,
    uint64_t q, uint64_t q_bytes, uint64_t absolute_start, uint32_t positions
);
// v2: bounded proposal-only temperature/conditional scale, positions2-4.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_probability_submit_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_q8_selector_args_v1* args,
    uint64_t q, uint64_t q_bytes, uint64_t absolute_start, uint32_t positions,
    float temperature234, float pair_scale234
);
// v3: per-position proposal temperature [0.5,1.5] and pair scale [0,2] for rows1-7.
// Host arrays of exactly seven floats; pair_scales7[0] must be 1. Requires positions==7.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_probability_submit_v3(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_q8_selector_args_v1* args,
    uint64_t q, uint64_t q_bytes, uint64_t absolute_start, uint32_t positions,
    const float* temperatures7, const float* pair_scales7
);
// v4: explicit process seed, XORed into the proposal purpose domain. Seed 0 replays v1/v3.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_probability_submit_v4(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_q8_selector_args_v1* args,
    uint64_t q, uint64_t q_bytes, uint64_t absolute_start, uint32_t positions,
    const float* temperatures7, const float* pair_scales7, uint64_t seed
);
// Opt-in rank compensation: keep transition scale at 1 and soften top-1 unary bias.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_rank_comp_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_q8_selector_args_v1* args, float rank_bonus
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_dflash_q8_selector_tail_rank_comp_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_dflash_q8_selector_args_v1* args,
    float transition_scale, float rank_bonus
);

#define AUGUR_CUDA_M304_R3_EXPERT_ROUTE_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_M304_R3_EXPERTS_V1 UINT32_C(256)
#define AUGUR_CUDA_M304_R3_ROUTE_K_V1 UINT32_C(6)
#define AUGUR_CUDA_M304_R3_ROUTE_LAYERS_V1 UINT32_C(43)
#define AUGUR_CUDA_M304_R3_EXPERT_SLOT_TABLE_BYTES_V1 UINT64_C(6144)
#define AUGUR_CUDA_M304_R3_ROUTE_STEP_RECEIPT_BYTES_V1 UINT64_C(1440)
#define AUGUR_CUDA_M304_R3_AGGREGATION_ROUTE_RANK_V1 UINT32_C(0)
#define AUGUR_CUDA_M304_R3_ROUTE_RECEIPT_MAGIC_V1 UINT32_C(0x33525443)
#define AUGUR_CUDA_M304_R3_ROUTE_STATUS_UNWRITTEN_V1 UINT32_C(0)
#define AUGUR_CUDA_M304_R3_ROUTE_STATUS_OK_V1 UINT32_C(1)
#define AUGUR_CUDA_M304_R3_ROUTE_STATUS_ERROR_V1 UINT32_C(2)
#define AUGUR_CUDA_M304_R3_ROUTE_ERROR_NONE_V1 UINT32_C(0)
#define AUGUR_CUDA_M304_R3_ROUTE_ERROR_ROUTER_STATUS_V1 UINT32_C(1)
#define AUGUR_CUDA_M304_R3_ROUTE_ERROR_EXPERT_RANGE_V1 UINT32_C(2)
#define AUGUR_CUDA_M304_R3_ROUTE_ERROR_DUPLICATE_EXPERT_V1 UINT32_C(3)
#define AUGUR_CUDA_M304_R3_ROUTE_ERROR_MISSING_SLOT_V1 UINT32_C(4)

typedef struct augur_cuda_m304_r3_expert_slot_v1 {
    augur_cuda_device_ptr_t w1;
    augur_cuda_device_ptr_t w3;
    augur_cuda_device_ptr_t w2;
} augur_cuda_m304_r3_expert_slot_v1;

/*
 * Request-MLA M128 route grouping. The grouping kernel consumes all 128 x 6
 * device router outputs in row/rank order and publishes one bounded plan. The
 * compact assignment table is ordered by expert ID, then source row/rank.
 * Padded entries use the explicit invalid sentinel and are never model rows.
 */
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ROWS_V1 UINT32_C(128)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ROUTE_K_V1 UINT32_C(6)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_EXPERTS_V1 UINT32_C(256)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_CAPACITY_V1 UINT32_C(768)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_PADDED_CAPACITY_V1 UINT32_C(1536)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_PREFIX_ENTRIES_V1 UINT32_C(257)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_HEADER_BYTES_V1 UINT64_C(64)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_PREFIX_BYTES_V1 UINT64_C(1028)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ACTIVE_EXPERT_BYTES_V1 UINT64_C(512)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_BUCKET_BYTES_V1 UINT64_C(512)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ENTRY_BYTES_V1 UINT64_C(8)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ENTRIES_BYTES_V1 UINT64_C(6144)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_PADDED_ENTRIES_BYTES_V1 UINT64_C(12288)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_RESTORE_BYTES_V1 UINT64_C(1536)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_STATUS_UNWRITTEN_V1 INT32_C(0)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_STATUS_OK_V1 INT32_C(1)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ERROR_ROUTER_STATUS_V1 (-INT32_C(1))
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ERROR_EXPERT_RANGE_V1 (-INT32_C(2))
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ERROR_DUPLICATE_EXPERT_V1 (-INT32_C(3))
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ERROR_MISSING_SLOT_V1 (-INT32_C(4))
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ERROR_NONFINITE_WEIGHT_V1 (-INT32_C(5))
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ERROR_ASSIGNMENT_OVERFLOW_V1 (-INT32_C(6))
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ERROR_PADDED_OVERFLOW_V1 (-INT32_C(7))
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ERROR_METADATA_V1 (-INT32_C(8))
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ERROR_INVALID_SENTINEL_V1 (-INT32_C(9))
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ERROR_PLAN_DRIFT_V1 (-INT32_C(10))
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_ERROR_RESTORE_V1 (-INT32_C(11))
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_INVALID_EXPERT_V1 UINT16_C(0xffff)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_INVALID_ROW_V1 UINT16_C(0xffff)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_INVALID_RANK_V1 UINT8_C(0xff)
#define AUGUR_CUDA_PREFILL_M128_EXPERT_ASSIGNMENT_FLAG_VALID_V1 UINT8_C(1)

typedef struct augur_cuda_prefill_m128_expert_assignment_entry_v1 {
    uint16_t row;
    uint8_t route_rank;
    uint8_t flags;
    uint32_t route_weight_bits;
} augur_cuda_prefill_m128_expert_assignment_entry_v1;

typedef struct augur_cuda_prefill_m128_expert_assignment_header_v1 {
    int32_t status;
    uint32_t valid_assignment_count;
    uint32_t padded_assignment_count;
    uint32_t active_expert_count;
    uint32_t rows;
    uint32_t route_k;
    uint32_t experts;
    uint32_t invalid_assignment;
    uint8_t plan_digest[32];
} augur_cuda_prefill_m128_expert_assignment_header_v1;

typedef struct augur_cuda_prefill_m128_expert_assignment_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t rows;
    uint32_t route_k;
    uint32_t experts;
    uint32_t assignment_capacity;
    uint32_t padded_assignment_capacity;
    uint32_t reserved0;
    augur_cuda_device_ptr_t slot_table;
    uint64_t slot_table_len_bytes;
    augur_cuda_device_ptr_t ids_u16;
    uint64_t ids_len_bytes;
    augur_cuda_device_ptr_t weights_f32;
    uint64_t weights_len_bytes;
    augur_cuda_device_ptr_t router_status_i32;
    uint64_t router_status_len_bytes;
    augur_cuda_device_ptr_t header;
    uint64_t header_len_bytes;
    augur_cuda_device_ptr_t prefix_u32;
    uint64_t prefix_len_bytes;
    augur_cuda_device_ptr_t padded_prefix_u32;
    uint64_t padded_prefix_len_bytes;
    augur_cuda_device_ptr_t active_experts_u16;
    uint64_t active_experts_len_bytes;
    augur_cuda_device_ptr_t bucket_capacity_u16;
    uint64_t bucket_capacity_len_bytes;
    augur_cuda_device_ptr_t assignments;
    uint64_t assignments_len_bytes;
    augur_cuda_device_ptr_t padded_assignments;
    uint64_t padded_assignments_len_bytes;
    augur_cuda_device_ptr_t restore_u16;
    uint64_t restore_len_bytes;
    uint8_t plan_digest[32];
    uint32_t reserved1[8];
} augur_cuda_prefill_m128_expert_assignment_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_expert_assignment_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_expert_assignment_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_expert_assignment_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_expert_assignment_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_expert_assignment_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_expert_assignment_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_expert_assignment_validate_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_expert_assignment_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_expert_assignment_validate_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_expert_assignment_args_v1* args
);

/*
 * Finite-bucket SM70 Q4_K/Q5_K/Q6_K projection used by grouped M128 prefill.
 * Compressed weights are decoded once into FP16 scratch, activations are cast
 * once, and WMMA accumulates into an isolated padded output. Only valid rows
 * are copied to the caller output, so invalid assignment padding stays poison.
 */
#define AUGUR_CUDA_PREFILL_M128_QK_PROJECTION_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_PREFILL_M128_QK_FORMAT_Q4K_V1 UINT32_C(4)
#define AUGUR_CUDA_PREFILL_M128_QK_FORMAT_Q5K_V1 UINT32_C(5)
#define AUGUR_CUDA_PREFILL_M128_QK_FORMAT_Q6K_V1 UINT32_C(6)
#define AUGUR_CUDA_PREFILL_M128_QK_MAX_BUCKET_ROWS_V1 UINT32_C(128)

typedef struct augur_cuda_prefill_m128_qk_projection_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t format;
    uint32_t valid_rows;
    uint32_t bucket_rows;
    uint32_t output_rows;
    uint32_t input_cols;
    uint32_t blocks_per_output_row;
    uint32_t weight_block_stride_bytes;
    uint32_t reserved0;
    augur_cuda_device_ptr_t weight;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t decoded_weight_f16;
    uint64_t decoded_weight_len_bytes;
    augur_cuda_device_ptr_t converted_input_f16;
    uint64_t converted_input_len_bytes;
    augur_cuda_device_ptr_t wmma_output_f32;
    uint64_t wmma_output_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    uint32_t reserved1[6];
} augur_cuda_prefill_m128_qk_projection_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_qk_projection_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_qk_projection_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_qk_projection_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_qk_projection_args_v1* args
);
/*
 * Requires a successful full preflight over identical immutable arguments.
 * Every referenced allocation and the stream owner must remain live through
 * terminal completion.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_qk_projection_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_qk_projection_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_qk_projection_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_qk_projection_args_v1* args
);

/*
 * Isolated SM70 full-bucket Q4_K W1 component.  Preflight creates the
 * cuBLAS handle and fills the device pointer arrays; the timed submit only
 * decodes, converts and runs one grouped Tensor-Core GEMM.  This ABI is
 * deliberately independent from the WMMA projection and Whole-FFN args.
 */
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_BATCH_V1 UINT32_C(6)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_ROWS_V1 UINT32_C(128)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_OUTPUT_ROWS_V1 UINT32_C(2048)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_INPUT_COLS_V1 UINT32_C(4096)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_BLOCKS_PER_ROW_V1 UINT32_C(16)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_WEIGHT_BLOCK_BYTES_V1 UINT32_C(160)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_POINTER_COUNT_V1 UINT32_C(30)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_WEIGHT_BYTES_V1 UINT64_C(5242880)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_INPUT_BYTES_V1 UINT64_C(2097152)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_OUTPUT_BYTES_V1 UINT64_C(1048576)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_DECODED_WEIGHT_BYTES_V1 UINT64_C(16777216)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_CONVERTED_INPUT_BYTES_V1 UINT64_C(1048576)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_POINTER_BYTES_V1 UINT64_C(240)

typedef struct augur_cuda_prefill_m128_q4_cublas_grouped_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t batch;
    uint32_t rows;
    uint32_t output_rows;
    uint32_t input_cols;
    uint32_t blocks_per_output_row;
    uint32_t weight_block_stride_bytes;
    augur_cuda_device_ptr_t weights_q4k[6];
    uint64_t weight_len_bytes[6];
    augur_cuda_device_ptr_t inputs_f32[6];
    uint64_t input_len_bytes[6];
    augur_cuda_device_ptr_t outputs_f32[6];
    uint64_t output_len_bytes[6];
    augur_cuda_device_ptr_t decoded_weights_f16;
    uint64_t decoded_weights_len_bytes;
    augur_cuda_device_ptr_t converted_inputs_f16;
    uint64_t converted_inputs_len_bytes;
    augur_cuda_device_ptr_t pointer_arrays;
    uint64_t pointer_arrays_len_bytes;
    uint64_t reserved[4];
} augur_cuda_prefill_m128_q4_cublas_grouped_args_v1;

typedef struct augur_cuda_prefill_m128_q4_cublas_grouped_plan_v1
    augur_cuda_prefill_m128_q4_cublas_grouped_plan_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_q4_cublas_grouped_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_q4_cublas_grouped_args_v1* args,
    augur_cuda_prefill_m128_q4_cublas_grouped_plan_v1** out_plan
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_q4_cublas_grouped_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_q4_cublas_grouped_args_v1* args,
    augur_cuda_prefill_m128_q4_cublas_grouped_plan_v1* plan
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_q4_cublas_grouped_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_q4_cublas_grouped_args_v1* args,
    augur_cuda_prefill_m128_q4_cublas_grouped_plan_v1* plan
);
void AUGUR_CUDA_CALL augur_cuda_prefill_m128_q4_cublas_grouped_plan_destroy_v1(
    augur_cuda_prefill_m128_q4_cublas_grouped_plan_v1* plan
);

/*
 * Isolated SM70 full-bucket Q4_K W13 component.  This is a distinct ABI from
 * the six-job W1 component above: one preflighted submit owns twelve Q4
 * matrices (W1 and W3), one decode/conversion phase and one batch-12 GEMM.
 * It is an experiment-only execution family and has no request-owner wiring.
 */
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_W13_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_W13_BATCH_V1 UINT32_C(12)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_W13_ROWS_V1 UINT32_C(128)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_W13_OUTPUT_ROWS_V1 UINT32_C(2048)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_W13_INPUT_COLS_V1 UINT32_C(4096)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_W13_BLOCKS_PER_ROW_V1 UINT32_C(16)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_W13_WEIGHT_BLOCK_BYTES_V1 UINT32_C(160)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_W13_POINTER_COUNT_V1 UINT32_C(60)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_W13_WEIGHT_BYTES_V1 UINT64_C(5242880)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_W13_INPUT_BYTES_V1 UINT64_C(2097152)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_W13_OUTPUT_BYTES_V1 UINT64_C(1048576)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_W13_DECODED_WEIGHT_BYTES_V1 UINT64_C(16777216)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_W13_CONVERTED_INPUT_BYTES_V1 UINT64_C(1048576)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_W13_POINTER_BYTES_V1 UINT64_C(480)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_W13_PHASE_REUSE_TAG_V2 \
    UINT64_C(0x4351573133505232)
#define AUGUR_CUDA_PREFILL_M128_Q4_CUBLAS_GROUPED_W13_DEVICE_POINTER_TAG_V11 \
    UINT64_C(0x4351573131503131)

typedef struct augur_cuda_prefill_m128_q4_cublas_grouped_w13_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t batch;
    uint32_t rows;
    uint32_t output_rows;
    uint32_t input_cols;
    uint32_t blocks_per_output_row;
    uint32_t weight_block_stride_bytes;
    augur_cuda_device_ptr_t weights_q4k[12];
    uint64_t weight_len_bytes[12];
    augur_cuda_device_ptr_t inputs_f32[12];
    uint64_t input_len_bytes[12];
    augur_cuda_device_ptr_t outputs_f32[12];
    uint64_t output_len_bytes[12];
    augur_cuda_device_ptr_t decoded_weights_f16;
    uint64_t decoded_weights_len_bytes;
    augur_cuda_device_ptr_t converted_inputs_f16;
    uint64_t converted_inputs_len_bytes;
    augur_cuda_device_ptr_t pointer_arrays;
    uint64_t pointer_arrays_len_bytes;
    uint64_t reserved[4];
} augur_cuda_prefill_m128_q4_cublas_grouped_w13_args_v1;

typedef struct augur_cuda_prefill_m128_q4_cublas_grouped_w13_plan_v1
    augur_cuda_prefill_m128_q4_cublas_grouped_w13_plan_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_q4_cublas_grouped_w13_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_q4_cublas_grouped_w13_args_v1* args,
    augur_cuda_prefill_m128_q4_cublas_grouped_w13_plan_v1** out_plan
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_q4_cublas_grouped_w13_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_q4_cublas_grouped_w13_args_v1* args,
    augur_cuda_prefill_m128_q4_cublas_grouped_w13_plan_v1* plan
);
/* V2 phase-reuse admission permits input/output overlap only. The same-stream
 * conversion of every F32 input precedes all cuBLAS output writes. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_q4_cublas_grouped_w13_phase_reuse_preflight_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_q4_cublas_grouped_w13_args_v1* args,
    augur_cuda_prefill_m128_q4_cublas_grouped_w13_plan_v1** out_plan
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_q4_cublas_grouped_w13_phase_reuse_prevalidated_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_q4_cublas_grouped_w13_args_v1* args,
    augur_cuda_prefill_m128_q4_cublas_grouped_w13_plan_v1* plan
);
/* V11 consumes a pointer table populated by the Whole-FFN device prepare.
 * W1/W3 may share identical read-only input ranges; all output ranges remain
 * disjoint and request-owned. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_q4_cublas_grouped_w13_device_pointer_preflight_v11(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_q4_cublas_grouped_w13_args_v1* args,
    augur_cuda_prefill_m128_q4_cublas_grouped_w13_plan_v1** out_plan
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_q4_cublas_grouped_w13_device_pointer_prevalidated_submit_v11(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_q4_cublas_grouped_w13_args_v1* args,
    augur_cuda_prefill_m128_q4_cublas_grouped_w13_plan_v1* plan
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_q4_cublas_grouped_w13_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_q4_cublas_grouped_w13_args_v1* args,
    augur_cuda_prefill_m128_q4_cublas_grouped_w13_plan_v1* plan
);
void AUGUR_CUDA_CALL augur_cuda_prefill_m128_q4_cublas_grouped_w13_plan_destroy_v1(
    augur_cuda_prefill_m128_q4_cublas_grouped_w13_plan_v1* plan
);

/*
 * Four real one-row expert buckets paired across W1/W3. Each expert input is
 * quantized once to group-32 Q8 and consumed by one paired direct-Q4_K DP4A
 * projection. This experiment-only ABI has no Request-owner wiring.
 */
#define AUGUR_CUDA_PREFILL_M128_Q4_DP4A_TAIL_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_PREFILL_M128_Q4_DP4A_TAIL_EXPERTS_V1 UINT32_C(4)
#define AUGUR_CUDA_PREFILL_M128_Q4_DP4A_TAIL_OUTPUT_ROWS_V1 UINT32_C(2048)
#define AUGUR_CUDA_PREFILL_M128_Q4_DP4A_TAIL_INPUT_COLS_V1 UINT32_C(4096)
#define AUGUR_CUDA_PREFILL_M128_Q4_DP4A_TAIL_BLOCKS_PER_ROW_V1 UINT32_C(16)
#define AUGUR_CUDA_PREFILL_M128_Q4_DP4A_TAIL_WEIGHT_BLOCK_BYTES_V1 UINT32_C(160)
#define AUGUR_CUDA_PREFILL_M128_Q4_DP4A_TAIL_WEIGHT_BYTES_V1 UINT64_C(5242880)
#define AUGUR_CUDA_PREFILL_M128_Q4_DP4A_TAIL_INPUT_BYTES_V1 UINT64_C(16384)
#define AUGUR_CUDA_PREFILL_M128_Q4_DP4A_TAIL_OUTPUT_BYTES_V1 UINT64_C(8192)
#define AUGUR_CUDA_PREFILL_M128_Q4_DP4A_TAIL_Q8_BYTES_V1 UINT64_C(16384)
#define AUGUR_CUDA_PREFILL_M128_Q4_DP4A_TAIL_SCALE_BYTES_V1 UINT64_C(2048)

typedef struct augur_cuda_prefill_m128_q4_dp4a_tail_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t experts;
    uint32_t output_rows;
    uint32_t input_cols;
    uint32_t blocks_per_output_row;
    uint32_t weight_block_stride_bytes;
    uint32_t reserved0;
    augur_cuda_device_ptr_t weights_w1_q4k[4];
    augur_cuda_device_ptr_t weights_w3_q4k[4];
    uint64_t weight_len_bytes[4];
    augur_cuda_device_ptr_t inputs_f32[4];
    uint64_t input_len_bytes[4];
    augur_cuda_device_ptr_t outputs_w1_f32[4];
    augur_cuda_device_ptr_t outputs_w3_f32[4];
    uint64_t output_len_bytes[4];
    augur_cuda_device_ptr_t q8_inputs_i8;
    uint64_t q8_inputs_len_bytes;
    augur_cuda_device_ptr_t q8_scales_f32;
    uint64_t q8_scales_len_bytes;
    uint64_t reserved[4];
} augur_cuda_prefill_m128_q4_dp4a_tail_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_q4_dp4a_tail_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_q4_dp4a_tail_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_q4_dp4a_tail_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_q4_dp4a_tail_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_q4_dp4a_tail_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_q4_dp4a_tail_args_v1* args
);

/*
 * Isolated M128 flash-style causal MLA attention core. The component consumes
 * immutable query/cache snapshots and publishes weighted plus inverse-RoPE
 * rows. KV append, index score, Top-K, WOA/WOB and request commit remain
 * outside this ABI.
 */
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_ROWS_V1 UINT32_C(128)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_CAPACITY_V1 UINT32_C(1024)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_HEADS_V1 UINT32_C(64)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_HEAD_DIM_V1 UINT32_C(512)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_TOP_K_V1 UINT32_C(10)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_BASE_V1 UINT32_C(128)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_FLAG_CAUSAL_V1 UINT32_C(1)
/* Opt-in extension: chunk_base_position stays a physical 1K workspace
 * position; reserved[0] carries the absolute cursor. */
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_FLAG_ABSOLUTE_CURSOR_V1 UINT32_C(2)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_QUERY_BYTES_V1 UINT64_C(16777216)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_RAW_BYTES_V1 UINT64_C(262144)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_RATIO4_BYTES_V1 UINT64_C(524288)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_RATIO128_BYTES_V1 UINT64_C(16384)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_IDS_BYTES_V1 UINT64_C(5120)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_SINK_BYTES_V1 UINT64_C(256)

typedef struct augur_cuda_prefill_m128_causal_mla_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t capacity;
    uint32_t chunk_base_position;
    uint32_t rows;
    uint32_t variant;
    uint32_t compression_ratio;
    uint32_t pre_raw_rows;
    uint32_t top_k;
    uint32_t index_base;
    uint32_t flags;
    augur_cuda_m289_device_view_v1 query;
    augur_cuda_m289_device_view_v1 pre_raw;
    augur_cuda_m289_device_view_v1 chunk_raw;
    augur_cuda_m289_device_view_v1 compressed;
    augur_cuda_m289_device_view_v1 ids;
    augur_cuda_m289_device_view_v1 sink;
    augur_cuda_m289_device_view_v1 weighted;
    augur_cuda_m289_device_view_v1 inverse;
    uint32_t reserved[4];
} augur_cuda_prefill_m128_causal_mla_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_causal_mla_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_causal_mla_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_args_v1* args
);
/* Requires one successful full preflight over identical immutable arguments. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_causal_mla_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_causal_mla_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_args_v1* args
);

/*
 * Isolated Stage-B Ratio4 index-score and ordered Top-K over one M128 chunk.
 * The two kernels share score/ID views but remain separate same-stream
 * submissions; Request owner wiring is intentionally outside this ABI.
 */
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_TOPK_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_TOPK_ROWS_V1 UINT32_C(128)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_TOPK_CAPACITY_V1 UINT32_C(1024)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_TOPK_CANDIDATES_V1 UINT32_C(256)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_TOPK_TOP_K_V1 UINT32_C(10)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_TOPK_HEADS_V1 UINT32_C(64)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_TOPK_HEAD_DIM_V1 UINT32_C(128)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_TOPK_INDEX_BASE_V1 UINT32_C(128)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_TOPK_RATIO_V1 UINT32_C(4)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_TOPK_FLAG_CAUSAL_V1 UINT32_C(1)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_TOPK_FLAG_ABSOLUTE_CURSOR_V1 UINT32_C(2)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_TOPK_QUERY_BYTES_V1 UINT64_C(4194304)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_TOPK_COMPRESSED_BYTES_V1 UINT64_C(131072)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_TOPK_HEAD_WEIGHT_BYTES_V1 UINT64_C(256)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_TOPK_SCORE_BYTES_V1 UINT64_C(131072)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_INDEX_TOPK_IDS_BYTES_V1 UINT64_C(5120)

typedef struct augur_cuda_prefill_m128_causal_mla_index_topk_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t capacity;
    uint32_t chunk_base_position;
    uint32_t rows;
    uint32_t candidates;
    uint32_t top_k;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t index_base;
    uint32_t causal_ratio;
    uint32_t flags;
    uint32_t reserved0;
    augur_cuda_m289_device_view_v1 index_query;
    augur_cuda_m289_device_view_v1 compressed;
    augur_cuda_m289_device_view_v1 head_weight;
    augur_cuda_m289_device_view_v1 scores;
    augur_cuda_m289_device_view_v1 ids;
    augur_cuda_m289_device_view_v1 selected_scores;
    uint64_t reserved[2];
} augur_cuda_prefill_m128_causal_mla_index_topk_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_causal_mla_index_topk_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_index_topk_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_causal_mla_index_topk_index_score_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_index_topk_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_causal_mla_index_topk_ordered_topk_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_index_topk_args_v1* args
);
/* Requires one successful full preflight over identical immutable arguments. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_causal_mla_index_topk_prevalidated_index_score_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_index_topk_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_causal_mla_index_topk_prevalidated_ordered_topk_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_index_topk_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_causal_mla_index_topk_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_index_topk_args_v1* args
);

/*
 * Isolated composition of chunk KV D2D append, optional Ratio4 index/Top-K,
 * and M128 causal attention. Cache bytes remain pending; this ABI publishes no
 * Request owner cursor, receipt, token, or visible output.
 */
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_ROWS_V1 UINT32_C(128)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_CAPACITY_V1 UINT32_C(1024)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_TOP_K_V1 UINT32_C(10)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_INDEX_BASE_V1 UINT32_C(128)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_FLAG_CAUSAL_V1 UINT32_C(1)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_FLAG_ABSOLUTE_CURSOR_V1 UINT32_C(2)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_ATTENTION_QUERY_BYTES_V1 UINT64_C(16777216)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_INDEX_QUERY_BYTES_V1 UINT64_C(4194304)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RAW_CHUNK_BYTES_V1 UINT64_C(262144)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO4_MAIN_CHUNK_BYTES_V1 UINT64_C(65536)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO4_INDEX_CHUNK_BYTES_V1 UINT64_C(16384)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO4_MAIN_CACHE_BYTES_V1 UINT64_C(524288)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO4_INDEX_CACHE_BYTES_V1 UINT64_C(131072)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO128_MAIN_CHUNK_BYTES_V1 UINT64_C(2048)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO128_MAIN_CACHE_BYTES_V1 UINT64_C(16384)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_HEAD_WEIGHT_BYTES_V1 UINT64_C(256)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_SCORE_BYTES_V1 UINT64_C(131072)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_IDS_BYTES_V1 UINT64_C(5120)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_SINK_BYTES_V1 UINT64_C(256)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_OUTPUT_BYTES_V1 UINT64_C(16777216)

typedef struct augur_cuda_prefill_m128_causal_mla_composed_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t variant;
    uint32_t compression_ratio;
    uint32_t capacity;
    uint32_t chunk_base_position;
    uint32_t rows;
    uint32_t pre_raw_rows;
    uint32_t top_k;
    uint32_t index_base;
    uint32_t flags;
    uint32_t reserved0;
    uint32_t reserved1;
    augur_cuda_m289_device_view_v1 attention_query;
    augur_cuda_m289_device_view_v1 index_query;
    augur_cuda_m289_device_view_v1 pre_raw;
    augur_cuda_m289_device_view_v1 chunk_raw;
    augur_cuda_m289_device_view_v1 compressed_main_source;
    augur_cuda_m289_device_view_v1 compressed_index_source;
    augur_cuda_m289_device_view_v1 raw_cache;
    augur_cuda_m289_device_view_v1 compressed_main_cache;
    augur_cuda_m289_device_view_v1 compressed_index_cache;
    augur_cuda_m289_device_view_v1 head_weight;
    augur_cuda_m289_device_view_v1 scores;
    augur_cuda_m289_device_view_v1 ids;
    augur_cuda_m289_device_view_v1 selected_scores;
    augur_cuda_m289_device_view_v1 sink;
    augur_cuda_m289_device_view_v1 weighted;
    augur_cuda_m289_device_view_v1 inverse;
    uint64_t reserved[4];
} augur_cuda_prefill_m128_causal_mla_composed_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_causal_mla_composed_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_composed_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_causal_mla_composed_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_composed_args_v1* args
);
/* Requires one successful full preflight over identical immutable arguments. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_causal_mla_composed_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_composed_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_causal_mla_composed_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_composed_args_v1* args
);

/*
 * V2 of the isolated composed boundary keeps the V1 layout and execution
 * order, but makes Ratio4 head weights row-owned.  V1 remains frozen for the
 * synthetic qualification evidence; product owner wiring must use V2.
 */
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_ARGS_VERSION_V2 UINT32_C(2)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_ROWS_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_ROWS_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_CAPACITY_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_CAPACITY_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_TOP_K_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_TOP_K_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_INDEX_BASE_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_INDEX_BASE_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_FLAG_CAUSAL_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_FLAG_CAUSAL_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_ATTENTION_QUERY_BYTES_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_ATTENTION_QUERY_BYTES_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_INDEX_QUERY_BYTES_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_INDEX_QUERY_BYTES_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RAW_CHUNK_BYTES_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RAW_CHUNK_BYTES_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO4_MAIN_CHUNK_BYTES_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO4_MAIN_CHUNK_BYTES_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO4_INDEX_CHUNK_BYTES_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO4_INDEX_CHUNK_BYTES_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO4_MAIN_CACHE_BYTES_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO4_MAIN_CACHE_BYTES_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO4_INDEX_CACHE_BYTES_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO4_INDEX_CACHE_BYTES_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO128_MAIN_CHUNK_BYTES_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO128_MAIN_CHUNK_BYTES_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO128_MAIN_CACHE_BYTES_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_RATIO128_MAIN_CACHE_BYTES_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_HEAD_WEIGHT_BYTES_V2 \
    UINT64_C(32768)
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_SCORE_BYTES_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_SCORE_BYTES_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_IDS_BYTES_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_IDS_BYTES_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_SINK_BYTES_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_SINK_BYTES_V1
#define AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_OUTPUT_BYTES_V2 \
    AUGUR_CUDA_PREFILL_M128_CAUSAL_MLA_COMPOSED_OUTPUT_BYTES_V1

typedef struct augur_cuda_prefill_m128_causal_mla_composed_args_v2 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t variant;
    uint32_t compression_ratio;
    uint32_t capacity;
    uint32_t chunk_base_position;
    uint32_t rows;
    uint32_t pre_raw_rows;
    uint32_t top_k;
    uint32_t index_base;
    uint32_t flags;
    uint32_t reserved0;
    uint32_t reserved1;
    augur_cuda_m289_device_view_v1 attention_query;
    augur_cuda_m289_device_view_v1 index_query;
    augur_cuda_m289_device_view_v1 pre_raw;
    augur_cuda_m289_device_view_v1 chunk_raw;
    augur_cuda_m289_device_view_v1 compressed_main_source;
    augur_cuda_m289_device_view_v1 compressed_index_source;
    augur_cuda_m289_device_view_v1 raw_cache;
    augur_cuda_m289_device_view_v1 compressed_main_cache;
    augur_cuda_m289_device_view_v1 compressed_index_cache;
    augur_cuda_m289_device_view_v1 head_weight;
    augur_cuda_m289_device_view_v1 scores;
    augur_cuda_m289_device_view_v1 ids;
    augur_cuda_m289_device_view_v1 selected_scores;
    augur_cuda_m289_device_view_v1 sink;
    augur_cuda_m289_device_view_v1 weighted;
    augur_cuda_m289_device_view_v1 inverse;
    uint64_t reserved[4];
} augur_cuda_prefill_m128_causal_mla_composed_args_v2;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_causal_mla_composed_preflight_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_composed_args_v2* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_causal_mla_composed_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_composed_args_v2* args
);
/* Requires one successful full preflight over identical immutable arguments. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_causal_mla_composed_prevalidated_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_composed_args_v2* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_causal_mla_composed_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_causal_mla_composed_args_v2* args
);

/*
 * Feature-isolated Stage-C routed half of the M128 grouped whole FFN. The
 * assignment plan stays device-owned: one preparation pass expands finite
 * expert buckets into bounded Q4/Q5 descriptors, and fixed persistent CTAs
 * consume only the published work count. The final routed output is copied
 * from private staging only after assignment, descriptor and completion
 * validation all succeed.
 */
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_HIDDEN_V1 UINT32_C(4096)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_INTERMEDIATE_V1 UINT32_C(2048)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_Q4_WORK_CAPACITY_V1 UINT32_C(16384)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_Q5_WORK_CAPACITY_V1 UINT32_C(32768)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_WORK_HEADER_BYTES_V1 UINT64_C(128)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_WORK_DESCRIPTOR_BYTES_V1 UINT64_C(16)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_Q4_WORK_BYTES_V1 UINT64_C(262144)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_Q5_WORK_BYTES_V1 UINT64_C(524288)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_NORMALIZED_BYTES_V1 UINT64_C(2097152)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_GATHERED_F16_BYTES_V1 UINT64_C(12582912)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_GATE_BYTES_V1 UINT64_C(12582912)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_UP_BYTES_V1 UINT64_C(12582912)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_RANK_DOWN_BYTES_V1 UINT64_C(12582912)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_ROUTED_BYTES_V1 UINT64_C(2097152)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_STATUS_UNWRITTEN_V1 INT32_C(0)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_STATUS_OK_V1 INT32_C(1)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_ERROR_ASSIGNMENT_V1 (-INT32_C(1))
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_ERROR_PLAN_DRIFT_V1 (-INT32_C(2))
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_ERROR_DESCRIPTOR_OVERFLOW_V1 (-INT32_C(3))
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_ERROR_METADATA_V1 (-INT32_C(4))
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_ERROR_SENTINEL_V1 (-INT32_C(5))
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_ERROR_SLOT_V1 (-INT32_C(6))
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_ERROR_COMPLETION_V1 (-INT32_C(7))
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_V2_STAGE_EVENT_COUNT_V1 UINT32_C(10)
#define AUGUR_CUDA_PREFILL_M128_GROUPED_WHOLE_FFN_V12_EPILOGUE_STAGE_EVENT_COUNT_V1 UINT32_C(9)
#define AUGUR_CUDA_PREFILL_M128_LAYER_EXPERT_F16_CACHE_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_PREFILL_M128_LAYER_EXPERT_F16_CACHE_SLOTS_V1 UINT32_C(32)
#define AUGUR_CUDA_PREFILL_M128_LAYER_EXPERT_F16_CACHE_METADATA_BYTES_V1 UINT64_C(1024)
#define AUGUR_CUDA_PREFILL_M128_LAYER_EXPERT_F16_CACHE_MATRIX_BYTES_V1 UINT64_C(16777216)
#define AUGUR_CUDA_PREFILL_M128_LAYER_EXPERT_F16_CACHE_PROJECTION_BYTES_V1 UINT64_C(536870912)

typedef struct augur_cuda_prefill_m128_grouped_whole_ffn_work_descriptor_v1 {
    uint16_t expert;
    uint16_t padded_base;
    uint16_t valid_rows;
    uint16_t bucket_rows;
    uint16_t row_base;
    uint16_t output_base;
    uint16_t flags;
    uint16_t reserved;
} augur_cuda_prefill_m128_grouped_whole_ffn_work_descriptor_v1;

typedef struct augur_cuda_prefill_m128_grouped_whole_ffn_work_header_v1 {
    int32_t status;
    uint32_t q4_work_count;
    uint32_t q5_work_count;
    uint32_t q4_cursor;
    uint32_t q5_cursor;
    uint32_t q4_completed;
    uint32_t q5_completed;
    uint32_t valid_assignment_count;
    uint32_t padded_assignment_count;
    uint32_t active_expert_count;
    uint32_t rows;
    uint32_t route_k;
    uint32_t experts;
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t invalid_work;
    uint8_t plan_digest[32];
    uint32_t reserved[8];
} augur_cuda_prefill_m128_grouped_whole_ffn_work_header_v1;

typedef struct augur_cuda_prefill_m128_grouped_whole_ffn_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t rows;
    uint32_t route_k;
    uint32_t experts;
    uint32_t assignment_capacity;
    uint32_t padded_assignment_capacity;
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t q4_work_capacity;
    uint32_t q5_work_capacity;
    uint32_t reserved0;
    augur_cuda_device_ptr_t slot_table;
    uint64_t slot_table_len_bytes;
    augur_cuda_device_ptr_t assignment_header;
    uint64_t assignment_header_len_bytes;
    augur_cuda_device_ptr_t prefix_u32;
    uint64_t prefix_len_bytes;
    augur_cuda_device_ptr_t padded_prefix_u32;
    uint64_t padded_prefix_len_bytes;
    augur_cuda_device_ptr_t active_experts_u16;
    uint64_t active_experts_len_bytes;
    augur_cuda_device_ptr_t bucket_capacity_u16;
    uint64_t bucket_capacity_len_bytes;
    augur_cuda_device_ptr_t padded_assignments;
    uint64_t padded_assignments_len_bytes;
    augur_cuda_device_ptr_t normalized_f32;
    uint64_t normalized_len_bytes;
    augur_cuda_device_ptr_t work_header;
    uint64_t work_header_len_bytes;
    augur_cuda_device_ptr_t q4_work;
    uint64_t q4_work_len_bytes;
    augur_cuda_device_ptr_t q5_work;
    uint64_t q5_work_len_bytes;
    augur_cuda_device_ptr_t gathered_f16;
    uint64_t gathered_len_bytes;
    augur_cuda_device_ptr_t gate_f32;
    uint64_t gate_len_bytes;
    augur_cuda_device_ptr_t up_f32;
    uint64_t up_len_bytes;
    augur_cuda_device_ptr_t rank_down_f32;
    uint64_t rank_down_len_bytes;
    augur_cuda_device_ptr_t routed_staging_f32;
    uint64_t routed_staging_len_bytes;
    augur_cuda_device_ptr_t routed_output_f32;
    uint64_t routed_output_len_bytes;
    uint8_t plan_digest[32];
    uint32_t reserved1[8];
} augur_cuda_prefill_m128_grouped_whole_ffn_args_v1;

typedef struct augur_cuda_prefill_m128_layer_expert_f16_cache_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layer;
    uint32_t reset;
    uint32_t slots;
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t reserved0;
    augur_cuda_device_ptr_t metadata;
    uint64_t metadata_len_bytes;
    augur_cuda_device_ptr_t w1_f16;
    uint64_t w1_len_bytes;
    augur_cuda_device_ptr_t w3_f16;
    uint64_t w3_len_bytes;
    augur_cuda_device_ptr_t w2_f16;
    uint64_t w2_len_bytes;
    uint32_t reserved1[8];
} augur_cuda_prefill_m128_layer_expert_f16_cache_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_grouped_whole_ffn_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_grouped_whole_ffn_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
/* Requires one successful full preflight over identical immutable views. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_grouped_whole_ffn_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
/* V2 keeps the V1 assignment/output ABI but uses an expert-major, weight-tile
 * reuse kernel family.  It is an isolated qualification surface. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_v2_prevalidated_submit(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
/* Isolated prefill-native W1/W3 replacement. It retains the V1 args/output
 * ABI and the V2 W2/reduce tail, but fuses Q4 W1/W3 in one row-tile CTA
 * family. Requires one successful full preflight over identical views. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_w13_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
/* Isolated quant-block-reuse successor to the W13 row-tile family. Full
 * 128-row buckets decode each 16x256 Q4 block once per CTA; finite smaller
 * buckets use the audited row-tile tail. Requires one successful full
 * preflight over identical views. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_w13_quant_block_v2_prevalidated_submit(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
/* Isolated warp-broadcast Q4_K decoder successor. Full and finite tail
 * buckets cooperatively decode packed nibble pairs without global scratch. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_w13_warp_broadcast_v3_prevalidated_submit(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
/* Isolated approximate Q8A/direct-Q4_K DP4A successor. It reuses the
 * rank-down allocation as phase-local Q8/scales scratch before the unchanged
 * Q5 stage overwrites that allocation. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_w13_q8a_dp4a_v4_prevalidated_submit(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
/* V10 splits the routed boundary around the independently preflighted V7/V9
 * cuBLAS-plus-tail submissions. Prepare validates the frozen route census and
 * gathers F32 inputs; epilogue resumes SwiGLU, W2, reduce and publication. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_cublas_w13_prepare_v10_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_cublas_w13_epilogue_v10_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
/* V11 prepares a device-selected dense mask and cuBLAS pointer table, then
 * resumes the native fallback/W2/reduce tail after the same-stream cuBLAS
 * submit. Both calls require the same preflighted immutable Whole-FFN args. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_dynamic_w13_prepare_v11_prevalidated_submit(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_dynamic_w13_epilogue_v11_prevalidated_submit(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
/* V12 keeps V11's dynamic dense selection and device pointer table, but
 * phase-reuses rank-down as Q8 scratch for a selected-mask-aware DP4A
 * fallback before the qualified V2 W2/reduce tail. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_dynamic_w13_prepare_v12_prevalidated_submit(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_dynamic_w13_epilogue_v12_prevalidated_submit(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
/* V26 replaces the dynamic W13/cuBLAS/W2 sequence for a complete M128 chunk.
 * The request-owned cache is reset on the first chunk of each layer and reused
 * by the remaining same-layer chunks. All views stay live through terminal
 * completion and the operation has no packed-weight fallback. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_layer_expert_f16_cache_v26_prevalidated_submit(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args,
    const augur_cuda_prefill_m128_layer_expert_f16_cache_args_v1* cache
);
/* Diagnostic-only V12 routed-epilogue submit. The nine events delimit its
 * eight native stages and must remain live through terminal completion. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_dynamic_w13_epilogue_v12_prevalidated_submit_with_stage_events_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args,
    const augur_cuda_event_t* stage_events,
    uint32_t stage_event_count
);
/* Diagnostic-only V2 submit. The ten events delimit the nine native stages
 * and must remain live through terminal completion. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_v2_prevalidated_submit_with_stage_events_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args,
    const augur_cuda_event_t* stage_events,
    uint32_t stage_event_count
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_grouped_whole_ffn_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_prefill_m128_grouped_whole_ffn_w13_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_w13_quant_block_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_w13_warp_broadcast_v3(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_prefill_m128_grouped_whole_ffn_w13_q8a_dp4a_v4(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_prefill_m128_grouped_whole_ffn_args_v1* args
);

#define AUGUR_CUDA_TOC_FAST_RESIDENT_ROUTE_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_TOC_FAST_RESIDENT_ROUTE_EXPERTS_V1 UINT32_C(256)
#define AUGUR_CUDA_TOC_FAST_RESIDENT_ROUTE_K_V1 UINT32_C(6)
#define AUGUR_CUDA_TOC_FAST_RESIDENT_ROUTE_STATUS_INSUFFICIENT_SLOTS_V1 (-INT32_C(31))

typedef struct augur_cuda_toc_fast_resident_route_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t slot_table;
    uint64_t slot_table_len_bytes;
    augur_cuda_device_ptr_t ids_u16;
    uint64_t ids_len_bytes;
    augur_cuda_device_ptr_t router_status_i32;
    uint64_t router_status_len_bytes;
    uint32_t experts;
    uint32_t route_k;
    uint32_t reserved[2];
} augur_cuda_toc_fast_resident_route_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_toc_fast_resident_route_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_toc_fast_resident_route_args_v1* args
);

/* CTX-6D candidate: parallel resident-slot probe with deterministic rewrite. */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_toc_fast_resident_route_parallel_candidate_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_toc_fast_resident_route_args_v1* args
);

typedef struct augur_cuda_m304_r3_route_layer_receipt_v1 {
    uint16_t ids[6];
    uint16_t layer;
    uint16_t position;
    uint32_t status;
    uint32_t error_code;
    uint64_t selected_rank_mask;
} augur_cuda_m304_r3_route_layer_receipt_v1;

typedef struct augur_cuda_m304_r3_route_step_receipt_v1 {
    uint32_t magic;
    uint32_t status;
    uint32_t first_error_layer;
    uint32_t error_code;
    uint64_t step_generation;
    uint64_t completed_layer_mask;
    uint8_t plan_digest[32];
    augur_cuda_m304_r3_route_layer_receipt_v1 layers[43];
} augur_cuda_m304_r3_route_step_receipt_v1;

typedef struct augur_cuda_m304_r3_expert_route_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t slot_table;
    uint64_t slot_table_len_bytes;
    augur_cuda_device_ptr_t ids_u16;
    uint64_t ids_len_bytes;
    augur_cuda_device_ptr_t router_status_i32;
    uint64_t router_status_len_bytes;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    augur_cuda_device_ptr_t step_receipt;
    uint64_t step_receipt_len_bytes;
    uint64_t step_generation;
    uint32_t layer;
    uint32_t position;
    uint32_t aggregation_order;
    uint32_t reserved;
    uint8_t plan_digest[32];
} augur_cuda_m304_r3_expert_route_args_v1;

#if defined(AUGUR_CUDA_SINGLE_SEQUENCE_FAST_PREFILL_ROUTE_ROW8_V14)
#define AUGUR_CUDA_M304_R3_ROW8_ROUTE_FINALIZE_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_M304_R3_ROW8_ROUTE_FINALIZE_ROWS_V1 UINT32_C(8)
#define AUGUR_CUDA_M304_R3_ROW8_ROUTE_FINALIZE_IDS_BYTES_V1 UINT64_C(96)
#define AUGUR_CUDA_M304_R3_ROW8_ROUTE_FINALIZE_STATUS_BYTES_V1 UINT64_C(32)
#define AUGUR_CUDA_M304_R3_ROW8_ROUTE_FINALIZE_RECEIPT_STRIDE_BYTES_V1 UINT64_C(1440)
#define AUGUR_CUDA_M304_R3_ROW8_ROUTE_FINALIZE_RECEIPTS_BYTES_V1 UINT64_C(11520)

typedef struct augur_cuda_m304_r3_row8_route_finalize_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t slot_table;
    uint64_t slot_table_len_bytes;
    augur_cuda_device_ptr_t ids_u16;
    uint64_t ids_len_bytes;
    augur_cuda_device_ptr_t router_status_i32;
    uint64_t router_status_len_bytes;
    augur_cuda_device_ptr_t step_receipts;
    uint64_t step_receipts_len_bytes;
    uint64_t receipt_stride_bytes;
    uint64_t first_step_generation;
    uint32_t layer;
    uint32_t first_position;
    uint32_t rows;
    uint32_t route_k;
    uint8_t plan_digest[32];
    uint32_t reserved[2];
} augur_cuda_m304_r3_row8_route_finalize_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m304_r3_row8_route_finalize_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_row8_route_finalize_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m304_r3_row8_route_finalize_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_row8_route_finalize_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_m304_r3_row8_route_finalize_prevalidated_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_row8_route_finalize_args_v1* args
);
#endif

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m304_r3_expert_input_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_expert_route_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m304_r3_expert_down_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_expert_route_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_toc_fast_expert_input_warp_fp32_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_expert_route_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_toc_fast_expert_down_warp_fp32_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_expert_route_args_v1* args
);
#if defined(AUGUR_CUDA_SINGLE_SEQUENCE_FAST_PREFILL_ROUTE_RECEIPT_ONLY)
/*
 * Publishes only the validated per-layer route receipt. Input/output views
 * remain part of the frozen route ABI but no expert projection is launched.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_toc_fast_expert_receipt_only_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_expert_route_args_v1* args
);
#endif
/*
 * Fast successor for a down projection whose matching input projection has
 * already published the route receipt on the same stream. This entry point
 * still validates the complete host ABI and the down kernel still checks the
 * device receipt status; only the duplicate device validator launch is
 * omitted. Safe callers must enforce the input-to-down validation ticket.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_toc_fast_expert_down_warp_fp32_novalidate_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_expert_route_args_v1* args
);

/*
 * Experimental B=1 successor. It preserves the v1 route/owner ABI and dot
 * order while staging one quantization block per warp in a four-warp CTA.
 * The down entry requires the same validated-input ticket as the v1
 * no-validate path.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_toc_fast_expert_input_warp_fp32_shared_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_expert_route_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_toc_fast_expert_down_warp_fp32_shared_novalidate_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_expert_route_args_v1* args
);

/*
 * Experimental B=1 Q4 input successor. The caller supplies reusable device
 * scratch for one group-32 INT8 activation and its FP32 scales. Submission
 * includes route validation, activation quantization and all twelve routed Q4
 * projections so component timing cannot omit the quantization cost.
 */
#define AUGUR_CUDA_TOC_FAST_EXPERT_INPUT_DP4A_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_TOC_FAST_EXPERT_INPUT_DP4A_Q8_BYTES_V1 UINT64_C(4096)
#define AUGUR_CUDA_TOC_FAST_EXPERT_INPUT_DP4A_SCALE_BYTES_V1 UINT64_C(512)

typedef struct augur_cuda_toc_fast_expert_input_dp4a_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_m304_r3_expert_route_args_v1 route;
    augur_cuda_device_ptr_t q8_input_i8;
    uint64_t q8_input_len_bytes;
    augur_cuda_device_ptr_t q8_scales_f32;
    uint64_t q8_scales_len_bytes;
} augur_cuda_toc_fast_expert_input_dp4a_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_toc_fast_expert_input_dp4a_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_toc_fast_expert_input_dp4a_args_v1* args
);

/*
 * CTX-6D successor for the static resident route. The first Group-32
 * activation-quantize CTA also canonicalizes the six route IDs and publishes
 * the authoritative route receipt before the ordered Q4 projection launch.
 * It uses the v1 argument record and does not relax route or receipt checks.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_ctx6d_route_quantize_expert_input_dp4a_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_toc_fast_expert_input_dp4a_args_v1* args
);
/*
 * B=1 ToC fast epilogue fusion. The input projection keeps the existing
 * validator and warp-row Q4 kernel, then applies all six SwiGLU and route
 * weights in one kernel. The down successor launches the existing warp-row
 * Q5 kernel and one fixed rank-order accumulator. Both retain the v1 route
 * receipt and allocation contract.
 */
#define AUGUR_CUDA_TOC_FAST_EXPERT_INPUT_FUSION_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_TOC_FAST_EXPERT_DOWN_ACCUM_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_toc_fast_expert_input_fusion_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_m304_r3_expert_route_args_v1 route;
    augur_cuda_device_ptr_t route_weights_f32;
    uint64_t route_weights_len_bytes;
    uint32_t swiglu_limit_bits;
    uint32_t reserved;
} augur_cuda_toc_fast_expert_input_fusion_args_v1;

typedef struct augur_cuda_toc_fast_expert_down_accum_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_m304_r3_expert_route_args_v1 route;
    augur_cuda_device_ptr_t routed_output_f32;
    uint64_t routed_output_len_bytes;
} augur_cuda_toc_fast_expert_down_accum_args_v1;

/*
 * B=1 Q5 down successor. SwiGLU leaves six rank-local 2048-element
 * activations in the input projection buffer. The caller supplies compact
 * rank-partitioned group-32 INT8 scratch and scales. Quantization, all six
 * Q5 down projections and fixed-rank accumulation are included in one
 * same-stream submission.
 */
#define AUGUR_CUDA_TOC_FAST_EXPERT_DOWN_DP4A_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_TOC_FAST_EXPERT_DOWN_DP4A_Q8_BYTES_V1 UINT64_C(12288)
#define AUGUR_CUDA_TOC_FAST_EXPERT_DOWN_DP4A_SCALE_BYTES_V1 UINT64_C(1536)

typedef struct augur_cuda_toc_fast_expert_down_dp4a_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_toc_fast_expert_down_accum_args_v1 down;
    augur_cuda_device_ptr_t q8_input_i8;
    uint64_t q8_input_len_bytes;
    augur_cuda_device_ptr_t q8_scales_f32;
    uint64_t q8_scales_len_bytes;
} augur_cuda_toc_fast_expert_down_dp4a_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_toc_fast_expert_input_swiglu_route_warp_fp32_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_toc_fast_expert_input_fusion_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_toc_fast_expert_swiglu_route_epilogue_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_toc_fast_expert_input_fusion_args_v1* args
);

/*
 * Enqueue the DP4A successor's complete routed-expert consumer chain from one
 * host call after the input projection has been submitted. The input kernel
 * can hide host-side down binding while epilogue, down and accumulation remain
 * contiguous on the stream.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_toc_fast_expert_dp4a_epilogue_down_fused_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_toc_fast_expert_input_fusion_args_v1* epilogue,
    const augur_cuda_toc_fast_expert_down_accum_args_v1* down
);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_toc_fast_expert_q5_dp4a_epilogue_down_fused_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_toc_fast_expert_input_fusion_args_v1* epilogue,
    const augur_cuda_toc_fast_expert_down_dp4a_args_v1* down
);

/*
 * CTX-6D stable-address expert Graph candidate. Dynamic position and step
 * generation remain in the validate-only v1 route record. The body record
 * contains only layer-static metadata and stable device addresses, so a
 * captured replay cannot freeze a prior step's dynamic identity.
 */
#define AUGUR_CUDA_CTX6D_EXPERT_GRAPH_BODY_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_ctx6d_expert_graph_body_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t slot_table;
    uint64_t slot_table_len_bytes;
    augur_cuda_device_ptr_t ids_u16;
    uint64_t ids_len_bytes;
    augur_cuda_device_ptr_t router_status_i32;
    uint64_t router_status_len_bytes;
    augur_cuda_device_ptr_t route_weights_f32;
    uint64_t route_weights_len_bytes;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t input_projection_f32;
    uint64_t input_projection_len_bytes;
    augur_cuda_device_ptr_t q4_input_i8;
    uint64_t q4_input_len_bytes;
    augur_cuda_device_ptr_t q4_scales_f32;
    uint64_t q4_scales_len_bytes;
    augur_cuda_device_ptr_t down_projection_f32;
    uint64_t down_projection_len_bytes;
    augur_cuda_device_ptr_t q5_input_i8;
    uint64_t q5_input_len_bytes;
    augur_cuda_device_ptr_t q5_scales_f32;
    uint64_t q5_scales_len_bytes;
    augur_cuda_device_ptr_t routed_output_f32;
    uint64_t routed_output_len_bytes;
    augur_cuda_device_ptr_t step_receipt;
    uint64_t step_receipt_len_bytes;
    uint32_t layer;
    uint32_t swiglu_limit_bits;
    uint8_t plan_digest[32];
} augur_cuda_ctx6d_expert_graph_body_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_ctx6d_expert_route_validate_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_expert_route_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_ctx6d_expert_graph_body_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_ctx6d_expert_graph_body_args_v1* args
);

/*
 * The matching fused input submit must have published the route receipt on
 * this stream. Safe callers enforce that relationship with a linear ticket.
 */
augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_toc_fast_expert_down_rank6_accum_warp_fp32_novalidate_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_toc_fast_expert_down_accum_args_v1* args
);

/*
 * Exact two-row successor for the resident expert path. Each nested v1 row
 * keeps its own position, generation, plan digest and route receipt. The
 * outer identity binds the fixed row order and allows one weight decode to
 * feed both rows when their route sets overlap.
 */
#define AUGUR_CUDA_M304_R3_EXPERT_GROUP2_ARGS_VERSION_V2 UINT32_C(2)
#define AUGUR_CUDA_M304_R3_EXPERT_GROUP2_ROWS_V2 UINT32_C(2)
#define AUGUR_CUDA_M304_R3_EXPERT_GROUP2_PROJECTION_INPUT_V2 UINT32_C(1)
#define AUGUR_CUDA_M304_R3_EXPERT_GROUP2_PROJECTION_DOWN_V2 UINT32_C(2)

typedef struct augur_cuda_m304_r3_expert_group2_args_v2 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t row_count;
    uint32_t projection;
    uint8_t group_plan_digest[32];
    augur_cuda_m304_r3_expert_route_args_v1 rows[2];
} augur_cuda_m304_r3_expert_group2_args_v2;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m304_r3_expert_input_group2_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_expert_group2_args_v2* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m304_r3_expert_down_group2_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_expert_group2_args_v2* args
);

/*
 * Four-row compact-union successor. A small device kernel builds the ordered
 * expert -> row-mask/rank table once per layer. Routed projection kernels then
 * consume that table directly instead of rescanning all 24 route IDs in every
 * output block.
 */
#define AUGUR_CUDA_M304_R3_EXPERT_GROUP4_ARGS_VERSION_V3 UINT32_C(3)
#define AUGUR_CUDA_M304_R3_EXPERT_GROUP4_ROWS_V3 UINT32_C(4)
#define AUGUR_CUDA_M304_R3_EXPERT_GROUP4_MAX_UNION_V3 UINT32_C(24)
#define AUGUR_CUDA_M304_R3_EXPERT_GROUP4_PROJECTION_INPUT_V3 UINT32_C(1)
#define AUGUR_CUDA_M304_R3_EXPERT_GROUP4_PROJECTION_DOWN_V3 UINT32_C(2)
#define AUGUR_CUDA_M304_R3_EXPERT_GROUP4_TABLE_STATUS_UNWRITTEN_V3 UINT32_C(0)
#define AUGUR_CUDA_M304_R3_EXPERT_GROUP4_TABLE_STATUS_OK_V3 UINT32_C(1)
#define AUGUR_CUDA_M304_R3_EXPERT_GROUP4_TABLE_STATUS_ERROR_V3 UINT32_C(2)

typedef struct augur_cuda_m304_r3_expert_group4_union_entry_v3 {
    uint16_t expert;
    uint8_t row_mask;
    uint8_t reserved;
    int8_t route_rank[4];
} augur_cuda_m304_r3_expert_group4_union_entry_v3;

typedef struct augur_cuda_m304_r3_expert_group4_union_table_v3 {
    uint32_t status;
    uint32_t count;
    uint32_t route_error[4];
    augur_cuda_m304_r3_expert_group4_union_entry_v3 entries[24];
} augur_cuda_m304_r3_expert_group4_union_table_v3;

#define AUGUR_CUDA_M304_R3_EXPERT_GROUP4_UNION_TABLE_BYTES_V3 UINT64_C(216)

typedef struct augur_cuda_m304_r3_expert_group4_args_v3 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t row_count;
    uint32_t projection;
    uint8_t group_plan_digest[32];
    augur_cuda_device_ptr_t union_table;
    uint64_t union_table_len_bytes;
    augur_cuda_m304_r3_expert_route_args_v1 rows[4];
} augur_cuda_m304_r3_expert_group4_args_v3;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m304_r3_expert_input_group4_submit_v3(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_expert_group4_args_v3* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m304_r3_expert_down_group4_submit_v3(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_expert_group4_args_v3* args
);

#define AUGUR_CUDA_M306_K4_R2_EXPERT_ROUTE_ARGS_VERSION_V2 UINT32_C(2)
#define AUGUR_CUDA_M306_K4_R2_EXPERT_SLOT_VERSION_V2 UINT32_C(2)
#define AUGUR_CUDA_M306_K4_R2_Q4K_EXPERT_ROW32_CUDA_V2 UINT32_C(110)
#define AUGUR_CUDA_M306_K4_R2_Q5K_EXPERT_ROW32_CUDA_V2 UINT32_C(111)
#define AUGUR_CUDA_M306_K4_R2_PROJECTION_INPUT_V2 UINT32_C(1)
#define AUGUR_CUDA_M306_K4_R2_PROJECTION_DOWN_V2 UINT32_C(2)
#define AUGUR_CUDA_M306_K4_R2_SLOT_PROJECTION_W1_V2 UINT32_C(1)
#define AUGUR_CUDA_M306_K4_R2_SLOT_PROJECTION_W2_V2 UINT32_C(2)
#define AUGUR_CUDA_M306_K4_R2_SLOT_PROJECTION_W3_V2 UINT32_C(3)
#define AUGUR_CUDA_M306_K4_R2_EXPERTS_V2 UINT32_C(256)
#define AUGUR_CUDA_M306_K4_R2_ROUTE_K_V2 UINT32_C(6)
#define AUGUR_CUDA_M306_K4_R2_INPUT_ROWS_V2 UINT32_C(2048)
#define AUGUR_CUDA_M306_K4_R2_INPUT_COLS_V2 UINT32_C(4096)
#define AUGUR_CUDA_M306_K4_R2_INPUT_MATRICES_PER_EXPERT_V2 UINT32_C(2)
#define AUGUR_CUDA_M306_K4_R2_DOWN_ROWS_V2 UINT32_C(4096)
#define AUGUR_CUDA_M306_K4_R2_DOWN_COLS_V2 UINT32_C(2048)
#define AUGUR_CUDA_M306_K4_R2_DOWN_MATRICES_PER_EXPERT_V2 UINT32_C(1)
#define AUGUR_CUDA_M306_K4_R2_ROW_GROUP_V2 UINT32_C(32)
#define AUGUR_CUDA_M306_K4_R2_WEIGHT_ALIGNMENT_V2 UINT64_C(32)
#define AUGUR_CUDA_M306_K4_R2_Q4_WEIGHT_BYTES_V2 UINT64_C(5242880)
#define AUGUR_CUDA_M306_K4_R2_Q5_WEIGHT_BYTES_V2 UINT64_C(6291456)
#define AUGUR_CUDA_M306_K4_R2_EXPERT_SLOT_TABLE_BYTES_V2 UINT64_C(22528)
#define AUGUR_CUDA_M306_K4_R2_ROUTE_ERROR_SLOT_ABI_V2 UINT32_C(5)
#define AUGUR_CUDA_M306_K4_R2_ROUTE_ERROR_LAYOUT_V2 UINT32_C(6)
#define AUGUR_CUDA_M306_K4_R2_ROUTE_ERROR_EXTENT_V2 UINT32_C(7)
#define AUGUR_CUDA_M306_K4_R2_ROUTE_ERROR_PROJECTION_V2 UINT32_C(8)
#define AUGUR_CUDA_M306_K4_R2_ROUTE_ERROR_ROW_GROUP_V2 UINT32_C(9)
#define AUGUR_CUDA_M306_K4_R2_ROUTE_ERROR_SLOT_RESERVED_V2 UINT32_C(10)

typedef struct augur_cuda_m306_k4_r2_expert_slot_v2 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t w1;
    uint64_t w1_len_bytes;
    augur_cuda_device_ptr_t w3;
    uint64_t w3_len_bytes;
    augur_cuda_device_ptr_t w2;
    uint64_t w2_len_bytes;
    uint32_t w1_layout;
    uint32_t w3_layout;
    uint32_t w2_layout;
    uint32_t w1_projection;
    uint32_t w3_projection;
    uint32_t w2_projection;
    uint32_t row_group;
    uint32_t reserved;
} augur_cuda_m306_k4_r2_expert_slot_v2;

typedef struct augur_cuda_m306_k4_r2_expert_route_args_v2 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t slot_table;
    uint64_t slot_table_len_bytes;
    augur_cuda_device_ptr_t ids_u16;
    uint64_t ids_len_bytes;
    augur_cuda_device_ptr_t router_status_i32;
    uint64_t router_status_len_bytes;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    augur_cuda_device_ptr_t step_receipt;
    uint64_t step_receipt_len_bytes;
    uint64_t step_generation;
    uint32_t layer;
    uint32_t position;
    uint32_t aggregation_order;
    uint32_t projection;
    uint32_t slot_count;
    uint32_t route_k;
    uint32_t rows;
    uint32_t cols;
    uint32_t matrices_per_expert;
    uint32_t row_group;
    uint32_t input_layout;
    uint32_t down_layout;
    uint32_t slot_version;
    uint32_t slot_struct_size;
    uint32_t reserved;
    uint8_t plan_digest[32];
} augur_cuda_m306_k4_r2_expert_route_args_v2;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k4_r2_expert_input_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k4_r2_expert_route_args_v2* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k4_r2_expert_down_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k4_r2_expert_route_args_v2* args
);

/*
 * BATCH-01 B=8 row32 shared-weight successor. The input submission builds one
 * ordered expert union and publishes all eight private route receipts. Input
 * and down kernels decode each selected row32 weight once, then update only
 * the request-private accumulator chains named by the union entry.
 */
#define AUGUR_CUDA_BATCH01_B8_ROW32_EXPERT_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_BATCH01_B8_ROW32_EXPERT_ROWS_V1 UINT32_C(8)
#define AUGUR_CUDA_BATCH01_B8_ROW32_EXPERT_MAX_UNION_V1 UINT32_C(48)
#define AUGUR_CUDA_BATCH01_B8_ROW32_EXPERT_TABLE_STATUS_UNWRITTEN_V1 UINT32_C(0)
#define AUGUR_CUDA_BATCH01_B8_ROW32_EXPERT_TABLE_STATUS_OK_V1 UINT32_C(1)
#define AUGUR_CUDA_BATCH01_B8_ROW32_EXPERT_TABLE_STATUS_ERROR_V1 UINT32_C(2)

typedef struct augur_cuda_batch01_b8_row32_expert_union_entry_v1 {
    uint16_t expert;
    uint8_t row_mask;
    uint8_t reserved;
    int8_t route_rank[8];
} augur_cuda_batch01_b8_row32_expert_union_entry_v1;

typedef struct augur_cuda_batch01_b8_row32_expert_union_table_v1 {
    uint32_t status;
    uint32_t count;
    uint32_t route_error[8];
    augur_cuda_batch01_b8_row32_expert_union_entry_v1 entries[48];
} augur_cuda_batch01_b8_row32_expert_union_table_v1;

#define AUGUR_CUDA_BATCH01_B8_ROW32_EXPERT_UNION_TABLE_BYTES_V1 UINT64_C(616)

typedef struct augur_cuda_batch01_b8_row32_expert_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t row_count;
    uint32_t projection;
    uint8_t group_plan_digest[32];
    augur_cuda_device_ptr_t union_table;
    uint64_t union_table_len_bytes;
    augur_cuda_m306_k4_r2_expert_route_args_v2 rows[8];
} augur_cuda_batch01_b8_row32_expert_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b8_row32_expert_input_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_row32_expert_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b8_row32_expert_down_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_row32_expert_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b8_row32_expert_input_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_row32_expert_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b8_row32_expert_down_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_row32_expert_args_v1* args
);

#define AUGUR_CUDA_M306_K4_R3_EXPERT_ROUTE_ARGS_VERSION_V3 UINT32_C(3)
#define AUGUR_CUDA_M306_K4_R3_EXPERT_SLOT_VERSION_V3 UINT32_C(3)
#define AUGUR_CUDA_M306_K4_R3_Q4K_HOT_CUDA_V1 UINT32_C(103)
#define AUGUR_CUDA_M306_K4_R3_Q5K_HOT_CUDA_V1 UINT32_C(105)
#define AUGUR_CUDA_M306_K4_R3_PROJECTION_INPUT_V3 UINT32_C(1)
#define AUGUR_CUDA_M306_K4_R3_PROJECTION_DOWN_V3 UINT32_C(2)
#define AUGUR_CUDA_M306_K4_R3_SLOT_PROJECTION_W1_V3 UINT32_C(1)
#define AUGUR_CUDA_M306_K4_R3_SLOT_PROJECTION_W2_V3 UINT32_C(2)
#define AUGUR_CUDA_M306_K4_R3_SLOT_PROJECTION_W3_V3 UINT32_C(3)
#define AUGUR_CUDA_M306_K4_R3_EXPERTS_V3 UINT32_C(256)
#define AUGUR_CUDA_M306_K4_R3_ROUTE_K_V3 UINT32_C(6)
#define AUGUR_CUDA_M306_K4_R3_INPUT_ROWS_V3 UINT32_C(2048)
#define AUGUR_CUDA_M306_K4_R3_INPUT_COLS_V3 UINT32_C(4096)
#define AUGUR_CUDA_M306_K4_R3_INPUT_MATRICES_PER_EXPERT_V3 UINT32_C(2)
#define AUGUR_CUDA_M306_K4_R3_DOWN_ROWS_V3 UINT32_C(4096)
#define AUGUR_CUDA_M306_K4_R3_DOWN_COLS_V3 UINT32_C(2048)
#define AUGUR_CUDA_M306_K4_R3_DOWN_MATRICES_PER_EXPERT_V3 UINT32_C(1)
#define AUGUR_CUDA_M306_K4_R3_BLOCK_THREADS_V3 UINT32_C(128)
#define AUGUR_CUDA_M306_K4_R3_ROWS_PER_BLOCK_V3 UINT32_C(4)
#define AUGUR_CUDA_M306_K4_R3_PARTIAL_PRECISION_BITS_V3 UINT32_C(64)
#define AUGUR_CUDA_M306_K4_R3_PRODUCT_PRECISION_BITS_V3 UINT32_C(32)
#define AUGUR_CUDA_M306_K4_R3_PRODUCT_FMA_POLICY_SEPARATE_MUL_RN_V3 UINT32_C(0)
#define AUGUR_CUDA_M306_K4_R3_AFFINE_FMA_POLICY_FMA_RN_V3 UINT32_C(1)
#define AUGUR_CUDA_M306_K4_R3_FTZ_POLICY_IEEE_V3 UINT32_C(0)
#define AUGUR_CUDA_M306_K4_R3_FINAL_CAST_F64_TO_F32_RN_BF16_RNE_V3 UINT32_C(1)
#define AUGUR_CUDA_M306_K4_R3_WEIGHT_ALIGNMENT_V3 UINT64_C(32)
#define AUGUR_CUDA_M306_K4_R3_Q4_ROW_BYTES_V3 UINT32_C(2560)
#define AUGUR_CUDA_M306_K4_R3_Q5_ROW_BYTES_V3 UINT32_C(1536)
#define AUGUR_CUDA_M306_K4_R3_Q4_WEIGHT_BYTES_V3 UINT64_C(5242880)
#define AUGUR_CUDA_M306_K4_R3_Q5_WEIGHT_BYTES_V3 UINT64_C(6291456)
#define AUGUR_CUDA_M306_K4_R3_EXPERT_SLOT_TABLE_BYTES_V3 UINT64_C(22528)
#define AUGUR_CUDA_M306_K4_R3_VALIDATION_RECORD_BYTES_V3 UINT64_C(288)
#define AUGUR_CUDA_M306_K4_R3_VALIDATION_RECORD_MAGIC_V3 UINT32_C(0x33524b34)
#define AUGUR_CUDA_M306_K4_R3_VALIDATION_STATUS_UNWRITTEN_V3 UINT32_C(0)
#define AUGUR_CUDA_M306_K4_R3_VALIDATION_STATUS_OK_V3 UINT32_C(1)
#define AUGUR_CUDA_M306_K4_R3_VALIDATION_STATUS_ERROR_V3 UINT32_C(2)
#define AUGUR_CUDA_M306_K4_R3_ROUTE_ERROR_SLOT_ABI_V3 UINT32_C(11)
#define AUGUR_CUDA_M306_K4_R3_ROUTE_ERROR_LAYOUT_V3 UINT32_C(12)
#define AUGUR_CUDA_M306_K4_R3_ROUTE_ERROR_EXTENT_V3 UINT32_C(13)
#define AUGUR_CUDA_M306_K4_R3_ROUTE_ERROR_PROJECTION_V3 UINT32_C(14)
#define AUGUR_CUDA_M306_K4_R3_ROUTE_ERROR_ROW_STRIDE_V3 UINT32_C(15)
#define AUGUR_CUDA_M306_K4_R3_ROUTE_ERROR_SLOT_RESERVED_V3 UINT32_C(16)

typedef struct augur_cuda_m306_k4_r3_expert_slot_v3 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t w1;
    uint64_t w1_len_bytes;
    augur_cuda_device_ptr_t w3;
    uint64_t w3_len_bytes;
    augur_cuda_device_ptr_t w2;
    uint64_t w2_len_bytes;
    uint32_t w1_layout;
    uint32_t w3_layout;
    uint32_t w2_layout;
    uint32_t w1_projection;
    uint32_t w3_projection;
    uint32_t w2_projection;
    uint32_t q4_row_bytes;
    uint32_t q5_row_bytes;
} augur_cuda_m306_k4_r3_expert_slot_v3;

typedef struct augur_cuda_m306_k4_r3_validation_record_v3 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t magic;
    uint32_t status;
    uint32_t error_code;
    uint32_t projection;
    uint32_t route_k;
    uint32_t reserved0;
    uint64_t step_generation;
    uint8_t plan_digest[32];
    uint16_t ids[6];
    uint32_t reserved1;
    augur_cuda_device_ptr_t w1[6];
    augur_cuda_device_ptr_t w3[6];
    augur_cuda_device_ptr_t w2[6];
    uint32_t input_layout;
    uint32_t down_layout;
    uint32_t slot_version;
    uint32_t partial_precision_bits;
    uint32_t block_threads;
    uint32_t rows_per_block;
    uint8_t reduction_tree_offsets[8];
    uint32_t product_precision_bits;
    uint32_t product_fma_policy;
    uint32_t affine_fma_policy;
    uint32_t ftz_policy;
    uint32_t final_cast_policy;
    uint32_t reserved2;
} augur_cuda_m306_k4_r3_validation_record_v3;

typedef struct augur_cuda_m306_k4_r3_expert_route_args_v3 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t slot_table;
    uint64_t slot_table_len_bytes;
    augur_cuda_device_ptr_t ids_u16;
    uint64_t ids_len_bytes;
    augur_cuda_device_ptr_t router_status_i32;
    uint64_t router_status_len_bytes;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    augur_cuda_device_ptr_t step_receipt;
    uint64_t step_receipt_len_bytes;
    augur_cuda_device_ptr_t validation_record;
    uint64_t validation_record_len_bytes;
    uint64_t step_generation;
    uint32_t layer;
    uint32_t position;
    uint32_t aggregation_order;
    uint32_t projection;
    uint32_t slot_count;
    uint32_t route_k;
    uint32_t rows;
    uint32_t cols;
    uint32_t matrices_per_expert;
    uint32_t block_threads;
    uint32_t rows_per_block;
    uint32_t input_layout;
    uint32_t down_layout;
    uint32_t slot_version;
    uint32_t slot_struct_size;
    uint32_t partial_precision_bits;
    uint32_t product_precision_bits;
    uint32_t product_fma_policy;
    uint32_t affine_fma_policy;
    uint32_t ftz_policy;
    uint32_t final_cast_policy;
    uint8_t reduction_tree_offsets[8];
    uint32_t reserved;
    uint8_t plan_digest[32];
} augur_cuda_m306_k4_r3_expert_route_args_v3;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k4_r3_expert_input_submit_v3(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k4_r3_expert_route_args_v3* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k4_r3_expert_down_submit_v3(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k4_r3_expert_route_args_v3* args
);

#define AUGUR_CUDA_M304_R3_COMBINED_COMMIT_ARGS_VERSION_V2 UINT32_C(2)
#define AUGUR_CUDA_M304_R3_COMBINED_COMMIT_BYTES_V2 UINT64_C(64)
#define AUGUR_CUDA_M304_R3_COMBINED_STATUS_UNWRITTEN_V2 UINT32_C(0)
#define AUGUR_CUDA_M304_R3_COMBINED_STATUS_OK_V2 UINT32_C(1)
#define AUGUR_CUDA_M304_R3_COMBINED_STATUS_ERROR_V2 UINT32_C(2)
#define AUGUR_CUDA_M304_R3_COMBINED_ERROR_NONE_V2 UINT32_C(0)
#define AUGUR_CUDA_M304_R3_COMBINED_ERROR_ROUTE_MAGIC_V2 UINT32_C(1)
#define AUGUR_CUDA_M304_R3_COMBINED_ERROR_ROUTE_STATUS_V2 UINT32_C(2)
#define AUGUR_CUDA_M304_R3_COMBINED_ERROR_ROUTE_COMPLETION_V2 UINT32_C(3)
#define AUGUR_CUDA_M304_R3_COMBINED_ERROR_ROUTE_GENERATION_V2 UINT32_C(4)
#define AUGUR_CUDA_M304_R3_COMBINED_ERROR_ROUTE_PLAN_DIGEST_V2 UINT32_C(5)
#define AUGUR_CUDA_M304_R3_COMBINED_ERROR_ROUTER_STATUS_V2 UINT32_C(6)
#define AUGUR_CUDA_M304_R3_COMBINED_ERROR_EXPERT_RANGE_V2 UINT32_C(7)
#define AUGUR_CUDA_M304_R3_COMBINED_ERROR_DUPLICATE_EXPERT_V2 UINT32_C(8)
#define AUGUR_CUDA_M304_R3_COMBINED_ERROR_MISSING_SLOT_V2 UINT32_C(9)
#define AUGUR_CUDA_M304_R3_COMBINED_ERROR_NON_FINITE_LOGIT_V2 UINT32_C(10)
#define AUGUR_CUDA_M304_R3_COMBINED_ERROR_REDUCTION_INCOMPLETE_V2 UINT32_C(11)

typedef struct augur_cuda_m304_r3_combined_commit_v2 {
    uint32_t token_id;
    uint32_t top1_value_bits;
    uint32_t runner_up_id;
    uint32_t runner_up_value_bits;
    uint32_t status;
    uint32_t error_code;
    uint64_t step_generation;
    uint8_t plan_digest[32];
} augur_cuda_m304_r3_combined_commit_v2;

typedef struct augur_cuda_m304_r3_combined_commit_args_v2 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t logits;
    uint64_t logits_len_bytes;
    augur_cuda_device_ptr_t route_step_receipt;
    uint64_t route_step_receipt_len_bytes;
    augur_cuda_device_ptr_t partials;
    uint64_t partials_len_bytes;
    augur_cuda_device_ptr_t commit;
    uint64_t commit_len_bytes;
    uint64_t step_generation;
    uint32_t vocab;
    uint32_t partial_count;
    uint32_t expected_position;
    uint32_t reserved;
    uint8_t plan_digest[32];
} augur_cuda_m304_r3_combined_commit_args_v2;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m304_r3_combined_commit_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m304_r3_combined_commit_args_v2* args
);

#define AUGUR_CUDA_RESIDENT_MOE_F32_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_resident_moe_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weights;
    uint64_t weights_len_bytes;
    augur_cuda_device_ptr_t input;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    augur_cuda_device_ptr_t router_logits;
    uint64_t router_logits_len_bytes;
    augur_cuda_device_ptr_t selected_ids;
    uint64_t selected_ids_len_bytes;
    augur_cuda_device_ptr_t selected_weights;
    uint64_t selected_weights_len_bytes;
    augur_cuda_device_ptr_t scratch;
    uint64_t scratch_len_bytes;
    augur_cuda_device_ptr_t access_counts;
    uint64_t access_counts_len_bytes;
    uint32_t tokens;
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t expert_count;
    uint32_t top_k;
    float routed_scale;
} augur_cuda_resident_moe_f32_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_resident_moe_f32_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_resident_moe_f32_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_resident_moe_f32_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_resident_moe_f32_args_v1* args
);

/*
 * M52 Blackwell NVFP4 W4A4 AOT GEMM.
 *
 * Values contain packed E2M1 nibbles. Scale factors use the CUTLASS SM120
 * tiled E4M3 layout produced by augur_cuda_nvfp4_pack_scales_host_v1. The two
 * launch entry points are deliberately shape-specific; there is no runtime
 * kernel chooser.
 */
#define AUGUR_CUDA_NVFP4_AOT_ARGS_VERSION_V1 UINT32_C(1)
#define AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M128_V1 UINT32_C(1)
#define AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M128_V1 UINT32_C(2)
#define AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M1_V1 UINT32_C(101)
#define AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M2_V1 UINT32_C(102)
#define AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M4_V1 UINT32_C(103)
#define AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M8_V1 UINT32_C(104)
#define AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M16_V1 UINT32_C(105)
#define AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M32_V1 UINT32_C(106)
#define AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M64_V1 UINT32_C(107)
#define AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M1_V1 UINT32_C(201)
#define AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M2_V1 UINT32_C(202)
#define AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M4_V1 UINT32_C(203)
#define AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M8_V1 UINT32_C(204)
#define AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M16_V1 UINT32_C(205)
#define AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M32_V1 UINT32_C(206)
#define AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M64_V1 UINT32_C(207)
#define AUGUR_CUDA_NVFP4_OPERAND_A_V1 UINT32_C(1)
#define AUGUR_CUDA_NVFP4_OPERAND_B_V1 UINT32_C(2)

typedef struct augur_cuda_nvfp4_aot_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t a_values;
    uint64_t a_values_len_bytes;
    augur_cuda_device_ptr_t a_scales;
    uint64_t a_scales_len_bytes;
    augur_cuda_device_ptr_t b_values;
    uint64_t b_values_len_bytes;
    augur_cuda_device_ptr_t b_scales;
    uint64_t b_scales_len_bytes;
    augur_cuda_device_ptr_t output;
    uint64_t output_len_bytes;
    augur_cuda_device_ptr_t workspace;
    uint64_t workspace_len_bytes;
    float alpha;
    uint32_t reserved;
} augur_cuda_nvfp4_aot_args_v1;

/* M68 direct packed NVFP4 weight x BF16 activation. Weight scales remain in
 * logical row-major group-16 order; no BF16 weight matrix is materialized. */
#define AUGUR_CUDA_NVFP4_W4A16_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_nvfp4_w4a16_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t activation_bf16;
    uint64_t activation_len_bytes;
    augur_cuda_device_ptr_t weight_values;
    uint64_t weight_values_len_bytes;
    augur_cuda_device_ptr_t weight_scales;
    uint64_t weight_scales_len_bytes;
    augur_cuda_device_ptr_t output_bf16;
    uint64_t output_len_bytes;
    float weight_global_scale;
    uint32_t reserved;
} augur_cuda_nvfp4_w4a16_args_v1;

#define AUGUR_CUDA_M68_BF16_INTERSTAGE_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m68_bf16_interstage_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t gate_bf16;
    uint64_t gate_len_bytes;
    augur_cuda_device_ptr_t up_bf16;
    uint64_t up_len_bytes;
    augur_cuda_device_ptr_t route_weights_f32;
    uint64_t route_weights_len_bytes;
    augur_cuda_device_ptr_t output_bf16;
    uint64_t output_len_bytes;
    uint32_t capacity;
    uint32_t valid_rows;
    float swiglu_limit;
    uint32_t reserved;
} augur_cuda_m68_bf16_interstage_args_v1;

#define AUGUR_CUDA_NVFP4_INTERSTAGE_ARGS_VERSION_V1 UINT32_C(1)

typedef struct augur_cuda_nvfp4_interstage_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t gate_bf16;
    uint64_t gate_len_bytes;
    augur_cuda_device_ptr_t up_bf16;
    uint64_t up_len_bytes;
    augur_cuda_device_ptr_t output_values;
    uint64_t output_values_len_bytes;
    augur_cuda_device_ptr_t output_scales;
    uint64_t output_scales_len_bytes;
    augur_cuda_device_ptr_t scale_offsets;
    uint64_t scale_offsets_len_bytes;
    float route_weight;
    float activation_global_scale;
    float swiglu_limit;
    uint32_t reserved;
} augur_cuda_nvfp4_interstage_args_v1;

#define AUGUR_CUDA_NVFP4_SHAPE_INTERSTAGE_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_nvfp4_shape_interstage_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t gate_bf16;
    uint64_t gate_len_bytes;
    augur_cuda_device_ptr_t up_bf16;
    uint64_t up_len_bytes;
    augur_cuda_device_ptr_t route_weights_f32;
    uint64_t route_weights_len_bytes;
    augur_cuda_device_ptr_t output_values;
    uint64_t output_values_len_bytes;
    augur_cuda_device_ptr_t output_scales;
    uint64_t output_scales_len_bytes;
    augur_cuda_device_ptr_t scale_offsets;
    uint64_t scale_offsets_len_bytes;
    uint32_t capacity;
    uint32_t valid_rows;
    float activation_global_scale;
    float swiglu_limit;
} augur_cuda_nvfp4_shape_interstage_args_v1;

#define AUGUR_CUDA_BF16_GEMM_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_bf16_gemm_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t activation_bf16;
    uint64_t activation_len_bytes;
    augur_cuda_device_ptr_t weight_bf16;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    float alpha;
    uint32_t reserved;
} augur_cuda_bf16_gemm_args_v1;

#define AUGUR_CUDA_M54_SHARED_INTERSTAGE_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m54_shared_interstage_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t gate_f32;
    uint64_t gate_len_bytes;
    augur_cuda_device_ptr_t up_f32;
    uint64_t up_len_bytes;
    augur_cuda_device_ptr_t output_bf16;
    uint64_t output_len_bytes;
    float swiglu_limit;
    uint32_t reserved;
} augur_cuda_m54_shared_interstage_args_v1;

#define AUGUR_CUDA_M54_SUM_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m54_sum_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t routed_bf16[6];
    uint64_t routed_len_bytes;
    augur_cuda_device_ptr_t shared_f32;
    uint64_t shared_len_bytes;
    augur_cuda_device_ptr_t output_bf16;
    uint64_t output_len_bytes;
    augur_cuda_device_ptr_t router_status;
    uint64_t router_status_len_bytes;
    uint64_t reserved;
} augur_cuda_m54_sum_args_v1;

#define AUGUR_CUDA_M54_ROUTER_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m54_router_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t logits_f32;
    uint64_t logits_len_bytes;
    augur_cuda_device_ptr_t bias_f32;
    uint64_t bias_len_bytes;
    augur_cuda_device_ptr_t output_ids_i32;
    uint64_t output_ids_len_bytes;
    augur_cuda_device_ptr_t output_weights_f32;
    uint64_t output_weights_len_bytes;
    augur_cuda_device_ptr_t status_i32;
    uint64_t status_len_bytes;
    float routed_scale;
    float weight_tolerance;
    uint64_t reserved;
} augur_cuda_m54_router_args_v1;

#define AUGUR_CUDA_M58_ROUTER_GROUP_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m58_router_group_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t scores_f32;
    uint64_t scores_len_bytes;
    augur_cuda_device_ptr_t bias_f32;
    uint64_t bias_len_bytes;
    augur_cuda_device_ptr_t expected_ids_u16;
    uint64_t expected_ids_len_bytes;
    augur_cuda_device_ptr_t expected_weights_f32;
    uint64_t expected_weights_len_bytes;
    augur_cuda_device_ptr_t output_ids_u16;
    uint64_t output_ids_len_bytes;
    augur_cuda_device_ptr_t output_weights_f32;
    uint64_t output_weights_len_bytes;
    augur_cuda_device_ptr_t counts_u32;
    uint64_t counts_len_bytes;
    augur_cuda_device_ptr_t offsets_u32;
    uint64_t offsets_len_bytes;
    augur_cuda_device_ptr_t gathered_tokens_u16;
    uint64_t gathered_tokens_len_bytes;
    augur_cuda_device_ptr_t gathered_weights_f32;
    uint64_t gathered_weights_len_bytes;
    augur_cuda_device_ptr_t status_i32;
    uint64_t status_len_bytes;
    augur_cuda_device_ptr_t routed_accum_f32;
    uint64_t routed_accum_len_bytes;
    uint32_t tokens;
    uint32_t experts;
    uint32_t top_k;
    float routed_scale;
    float weight_tolerance;
    uint32_t reserved;
} augur_cuda_m58_router_group_args_v1;

#define AUGUR_CUDA_M95_ROUTER_GATHER_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m95_router_gather_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t logits_f32;
    uint64_t logits_len_bytes;
    augur_cuda_device_ptr_t bias_f32;
    uint64_t bias_len_bytes;
    augur_cuda_device_ptr_t output_ids_u16;
    uint64_t output_ids_len_bytes;
    augur_cuda_device_ptr_t output_weights_f32;
    uint64_t output_weights_len_bytes;
    augur_cuda_device_ptr_t counts_u32;
    uint64_t counts_len_bytes;
    augur_cuda_device_ptr_t offsets_u32;
    uint64_t offsets_len_bytes;
    augur_cuda_device_ptr_t gathered_tokens_u16;
    uint64_t gathered_tokens_len_bytes;
    augur_cuda_device_ptr_t gathered_weights_f32;
    uint64_t gathered_weights_len_bytes;
    augur_cuda_device_ptr_t status_i32;
    uint64_t status_len_bytes;
    uint32_t tokens;
    uint32_t experts;
    uint32_t top_k;
    float routed_scale;
    uint32_t reserved;
} augur_cuda_m95_router_gather_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m95_router_gather_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m95_router_gather_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m95_router_gather_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m95_router_gather_args_v1* args
);

#define AUGUR_CUDA_M102_DECODE_ROUTER_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m102_decode_router_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight_bf16;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t bias_f32;
    uint64_t bias_len_bytes;
    augur_cuda_device_ptr_t logits_f32;
    uint64_t logits_len_bytes;
    augur_cuda_device_ptr_t output_ids_u16;
    uint64_t output_ids_len_bytes;
    augur_cuda_device_ptr_t output_weights_f32;
    uint64_t output_weights_len_bytes;
    augur_cuda_device_ptr_t counts_u32;
    uint64_t counts_len_bytes;
    augur_cuda_device_ptr_t offsets_u32;
    uint64_t offsets_len_bytes;
    augur_cuda_device_ptr_t gathered_tokens_u16;
    uint64_t gathered_tokens_len_bytes;
    augur_cuda_device_ptr_t gathered_weights_f32;
    uint64_t gathered_weights_len_bytes;
    augur_cuda_device_ptr_t status_i32;
    uint64_t status_len_bytes;
    uint32_t rows;
    uint32_t cols;
    uint32_t experts;
    uint32_t top_k;
    float routed_scale;
    uint32_t reserved;
} augur_cuda_m102_decode_router_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m102_decode_router_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m102_decode_router_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m102_decode_router_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m102_decode_router_args_v1* args
);

#define AUGUR_CUDA_M204_HASH_ROUTER_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m204_hash_router_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight_bf16;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t tid2eid_i64;
    uint64_t tid2eid_len_bytes;
    augur_cuda_device_ptr_t input_ids_i64;
    uint64_t input_ids_len_bytes;
    augur_cuda_device_ptr_t logits_f32;
    uint64_t logits_len_bytes;
    augur_cuda_device_ptr_t output_ids_u16;
    uint64_t output_ids_len_bytes;
    augur_cuda_device_ptr_t output_weights_f32;
    uint64_t output_weights_len_bytes;
    augur_cuda_device_ptr_t status_i32;
    uint64_t status_len_bytes;
    uint32_t tokens;
    uint32_t hidden;
    uint32_t experts;
    uint32_t top_k;
    uint32_t table_rows;
    float routed_scale;
    uint32_t reserved;
} augur_cuda_m204_hash_router_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m204_hash_router_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m204_hash_router_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m204_hash_router_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m204_hash_router_args_v1* args
);

#define AUGUR_CUDA_M306_K38_HASH_ROUTER_SELECTED6_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m306_k38_hash_router_selected6_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight_bf16;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t tid2eid_i64;
    uint64_t tid2eid_len_bytes;
    augur_cuda_device_ptr_t input_ids_i64;
    uint64_t input_ids_len_bytes;
    augur_cuda_device_ptr_t scratch;
    uint64_t scratch_len_bytes;
    augur_cuda_device_ptr_t output_ids_u16;
    uint64_t output_ids_len_bytes;
    augur_cuda_device_ptr_t output_weights_f32;
    uint64_t output_weights_len_bytes;
    augur_cuda_device_ptr_t status_i32;
    uint64_t status_len_bytes;
    uint32_t tokens;
    uint32_t hidden;
    uint32_t experts;
    uint32_t top_k;
    uint32_t table_rows;
    float routed_scale;
    uint32_t reserved;
} augur_cuda_m306_k38_hash_router_selected6_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k38_hash_router_selected6_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k38_hash_router_selected6_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k38_hash_router_selected6_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k38_hash_router_selected6_args_v1* args
);

#define AUGUR_CUDA_M211_LEARNED_ROUTER_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m211_learned_router_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight_bf16;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t bias_f32;
    uint64_t bias_len_bytes;
    augur_cuda_device_ptr_t logits_f32;
    uint64_t logits_len_bytes;
    augur_cuda_device_ptr_t output_ids_u16;
    uint64_t output_ids_len_bytes;
    augur_cuda_device_ptr_t output_weights_f32;
    uint64_t output_weights_len_bytes;
    augur_cuda_device_ptr_t status_i32;
    uint64_t status_len_bytes;
    uint32_t tokens;
    uint32_t hidden;
    uint32_t experts;
    uint32_t top_k;
    float routed_scale;
    uint32_t reserved;
} augur_cuda_m211_learned_router_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m211_learned_router_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m211_learned_router_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m211_learned_router_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m211_learned_router_args_v1* args
);

#define AUGUR_CUDA_M211_LEARNED_ROUTER_ARGS_VERSION_V2 UINT32_C(2)
typedef struct augur_cuda_m211_learned_router_args_v2 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t weight_bf16;
    uint64_t weight_len_bytes;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t bias_f32;
    uint64_t bias_len_bytes;
    augur_cuda_device_ptr_t logits_f32;
    uint64_t logits_len_bytes;
    augur_cuda_device_ptr_t output_ids_u16;
    uint64_t output_ids_len_bytes;
    augur_cuda_device_ptr_t output_weights_f32;
    uint64_t output_weights_len_bytes;
    augur_cuda_device_ptr_t status_i32;
    uint64_t status_len_bytes;
    uint32_t tokens;
    uint32_t hidden;
    uint32_t experts;
    uint32_t top_k;
    float routed_scale;
    uint32_t denominator_order;
    uint32_t compensated_denominator;
} augur_cuda_m211_learned_router_args_v2;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m211_learned_router_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m211_learned_router_args_v2* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m211_learned_router_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m211_learned_router_args_v2* args
);

#define AUGUR_CUDA_M214_LEARNED_ROUTE_TELEMETRY_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m214_learned_route_telemetry_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t logits_f32;
    uint64_t logits_len_bytes;
    augur_cuda_device_ptr_t bias_f32;
    uint64_t bias_len_bytes;
    augur_cuda_device_ptr_t choices_f32;
    uint64_t choices_len_bytes;
    augur_cuda_device_ptr_t top7_ids_u16;
    uint64_t top7_ids_len_bytes;
    augur_cuda_device_ptr_t top7_choices_f32;
    uint64_t top7_choices_len_bytes;
    augur_cuda_device_ptr_t status_i32;
    uint64_t status_len_bytes;
    uint32_t tokens;
    uint32_t experts;
    uint32_t top_n;
    uint32_t reserved;
} augur_cuda_m214_learned_route_telemetry_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m214_learned_route_telemetry_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m214_learned_route_telemetry_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m214_learned_route_telemetry_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m214_learned_route_telemetry_args_v1* args
);

#define AUGUR_CUDA_M205_ROUTE_BF16_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m205_route_bf16_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t route_weights_f32;
    uint64_t route_weights_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    uint32_t token;
    uint32_t rank;
    uint32_t elements;
    uint32_t reserved;
} augur_cuda_m205_route_bf16_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m205_route_bf16_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m205_route_bf16_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m205_route_bf16_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m205_route_bf16_args_v1* args
);

#define AUGUR_CUDA_WEIGHTED_ACCUM_F32_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_weighted_accum_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    uint32_t elements;
    uint32_t weight_bits;
    uint32_t replace;
    uint32_t reserved;
} augur_cuda_weighted_accum_f32_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_weighted_accum_f32_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_weighted_accum_f32_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_weighted_accum_f32_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_weighted_accum_f32_args_v1* args
);

#define AUGUR_CUDA_COMPENSATED_ACCUM_F32_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_compensated_accum_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t input_f32;
    uint64_t input_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    augur_cuda_device_ptr_t compensation_f32;
    uint64_t compensation_len_bytes;
    uint32_t elements;
    uint32_t weight_bits;
    uint32_t replace;
    uint32_t reserved;
} augur_cuda_compensated_accum_f32_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_compensated_accum_f32_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_compensated_accum_f32_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_compensated_accum_f32_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_compensated_accum_f32_args_v1* args
);

#define AUGUR_CUDA_M58_SCATTER_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m58_scatter_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t expert_output_bf16;
    uint64_t expert_output_len_bytes;
    augur_cuda_device_ptr_t gathered_tokens_u16;
    uint64_t gathered_tokens_len_bytes;
    augur_cuda_device_ptr_t routed_accum_f32;
    uint64_t routed_accum_len_bytes;
    augur_cuda_device_ptr_t status_i32;
    uint64_t status_len_bytes;
    uint32_t group_offset;
    uint32_t valid_rows;
    uint32_t tokens;
    uint32_t hidden;
} augur_cuda_m58_scatter_args_v1;

#define AUGUR_CUDA_M58_FINALIZE_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m58_finalize_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t routed_accum_f32;
    uint64_t routed_accum_len_bytes;
    augur_cuda_device_ptr_t shared_f32;
    uint64_t shared_len_bytes;
    augur_cuda_device_ptr_t output_bf16;
    uint64_t output_len_bytes;
    augur_cuda_device_ptr_t status_i32;
    uint64_t status_len_bytes;
    uint32_t tokens;
    uint32_t hidden;
    uint64_t reserved;
} augur_cuda_m58_finalize_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m58_router_group_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m58_router_group_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m58_scatter_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m58_scatter_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m58_finalize_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m58_finalize_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m58_finalize_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m58_finalize_args_v1* args
);

#define AUGUR_CUDA_M111_SPARSE_ATTENTION_F32_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m111_sparse_attention_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t query_f32;
    uint64_t query_len_bytes;
    augur_cuda_device_ptr_t kv_f32;
    uint64_t kv_len_bytes;
    augur_cuda_device_ptr_t sink_f32;
    uint64_t sink_len_bytes;
    augur_cuda_device_ptr_t indices_i32;
    uint64_t indices_len_bytes;
    augur_cuda_device_ptr_t rope_cos_f32;
    uint64_t rope_cos_len_bytes;
    augur_cuda_device_ptr_t rope_sin_f32;
    uint64_t rope_sin_len_bytes;
    augur_cuda_device_ptr_t output_f32;
    uint64_t output_len_bytes;
    uint32_t tokens;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t selected;
} augur_cuda_m111_sparse_attention_f32_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m111_sparse_attention_f32_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m111_sparse_attention_f32_args_v1* args
);

#define AUGUR_CUDA_M131_STATEFUL_ATTENTION_F32_ARGS_VERSION_V1 UINT32_C(1)
typedef struct augur_cuda_m131_stateful_attention_f32_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_cuda_device_ptr_t query_f32; uint64_t query_len_bytes;
    augur_cuda_device_ptr_t current_kv_f32; uint64_t current_kv_len_bytes;
    augur_cuda_device_ptr_t kv_window_f32; uint64_t kv_window_len_bytes;
    augur_cuda_device_ptr_t sink_f32; uint64_t sink_len_bytes;
    augur_cuda_device_ptr_t indices_i32; uint64_t indices_len_bytes;
    augur_cuda_device_ptr_t rope_cos_f32; uint64_t rope_cos_len_bytes;
    augur_cuda_device_ptr_t rope_sin_f32; uint64_t rope_sin_len_bytes;
    augur_cuda_device_ptr_t output_f32; uint64_t output_len_bytes;
    uint32_t position;
    uint32_t visible_kv;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t selected;
    uint32_t reserved;
} augur_cuda_m131_stateful_attention_f32_args_v1;

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m131_stateful_attention_f32_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m131_stateful_attention_f32_args_v1* args
);

uint32_t AUGUR_CUDA_CALL augur_cuda_nvfp4_sm120_build_capability_v1(void);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_pack_scales_host_v1(
    uint32_t shape,
    uint32_t operand,
    const uint8_t* source,
    uint64_t source_len_bytes,
    uint8_t* destination,
    uint64_t destination_len_bytes
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_scale_extent_host_v1(
    uint32_t shape,
    uint32_t operand,
    uint64_t* out_size_bytes
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_shape_a_scale_offsets_host_v1(
    uint32_t shape,
    uint32_t* destination,
    uint64_t destination_count
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_shape_workspace_size_v1(
    uint32_t shape,
    uint64_t* out_size_bytes
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_shape_bf16_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    uint32_t shape,
    const augur_cuda_nvfp4_aot_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_shape_bf16_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    uint32_t shape,
    const augur_cuda_nvfp4_aot_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_w4a16_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    uint32_t shape,
    const augur_cuda_nvfp4_w4a16_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_w4a16_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    uint32_t shape,
    const augur_cuda_nvfp4_w4a16_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m68_bf16_interstage_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m68_bf16_interstage_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_shape_interstage_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_shape_interstage_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_shape_interstage_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_shape_interstage_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_w2_m128_a_scale_offsets_host_v1(
    uint32_t* destination,
    uint64_t destination_count
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_w13_m128_workspace_size_v1(
    uint64_t* out_size_bytes
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_w2_m128_workspace_size_v1(
    uint64_t* out_size_bytes
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_w13_m128_bf16_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_aot_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_w13_m128_bf16_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_aot_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_w2_m128_bf16_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_aot_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_w2_m128_bf16_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_aot_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_interstage_m128_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_interstage_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_interstage_m128_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_interstage_args_v1* args
);

augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_w13_m128_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_bf16_gemm_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_w13_m128_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_bf16_gemm_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_w2_m128_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_bf16_gemm_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_w2_m128_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_bf16_gemm_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m54_shared_interstage_m128_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m54_shared_interstage_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m54_shared_interstage_m128_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m54_shared_interstage_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m54_sum_m128_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m54_sum_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m54_sum_m128_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m54_sum_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m54_router_top6_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m54_router_args_v1* args
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m54_router_top6_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m54_router_args_v1* args
);

/* Independent-row DFlash context submits. Existing scalar ABIs/math unchanged. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_context_rows_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q8_0_f32_matvec_args_v1* args, uint32_t tokens);
/* Separate opt-in four-token weight reuse; same per-token FP64 arithmetic. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_rows_b8_specialized_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q8_0_f32_matvec_args_v1* args, uint32_t tokens
);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_rows_tile4_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q8_0_f32_matvec_args_v1* args, uint32_t tokens);
/* Separate bulk-history four-output-row input reuse; exact 512/2048 FC/KV shapes. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_history_reuse_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q8_0_f32_matvec_args_v1* args, uint32_t tokens);
/* Bulk-history decode panels; caller-owned disjoint scratch on the same stream.
 * Exact scratch bytes: 128*25600*4 for FC, 256*5120*4 for KV; tokens 512/2048. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_history_panel_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q8_0_f32_matvec_args_v1* args, uint32_t tokens,
    uint64_t scratch, uint64_t scratch_bytes);
/* Independent opt-in token8 retile, identical history-panel geometry/scratch. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_history_panel_token8_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q8_0_f32_matvec_args_v1* args, uint32_t tokens,
    uint64_t scratch, uint64_t scratch_bytes);
/* Independent real 1..512-row FC capture slice, no input/output padding.
 * Exact FC 5120x25600; same private 128*25600*4-byte scratch. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_history_fc_slice_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q8_0_f32_matvec_args_v1* args, uint32_t tokens,
    uint64_t scratch, uint64_t scratch_bytes);
/* Eight-head key norm+RoPE; preserve normalized scratch and rotated output. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_key_norm_rope_context_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_rmsnorm_f32_args_v1* norm,
    const augur_cuda_dflash_rope_f32_args_v1* rope, uint32_t tokens, uint32_t heads);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_norm_context_rows_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_rmsnorm_f32_args_v1* args, uint32_t rows);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_rope_context_rows_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_rope_f32_args_v1* args, uint32_t tokens, uint32_t heads);

#ifdef __cplusplus
}
#endif

#endif
