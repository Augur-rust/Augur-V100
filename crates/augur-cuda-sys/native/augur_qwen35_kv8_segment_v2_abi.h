#ifndef AUGUR_QWEN35_KV8_SEGMENT_V2_ABI_H
#define AUGUR_QWEN35_KV8_SEGMENT_V2_ABI_H

/*
 * Opt-in bounded KV8 segment contract. Partial mode 0 retains the serial
 * F32 baseline; mode 1 selects a CTA-per-head F64 accumulation candidate.
 * Mode 2 uses V1-shaped SIMT page-local recurrence. Diagnostic mode 3 is
 * single-query-token ORDERED PREFIX recurrence: begin==0 initializes the
 * partial buffers, later contiguous segments read/write those same buffers
 * on one stream with the same query. They hold cumulative state, NOT an
 * independent page contribution. Use merge operation 3, never operation 1
 * against a nonempty accumulator. Page ordering, complete coverage, ownership,
 * generation and terminal receipts remain caller obligations.
 */

#include <stdint.h>
#include "augur_cuda_runtime.h"

typedef uint64_t augur_qwen35_kv8_segment_device_ptr_t;

enum { AUGUR_CUDA_QWEN35_KV8_SEGMENT_PARTIAL_ARGS_VERSION_V2 = 2 };

typedef struct augur_cuda_qwen35_kv8_segment_partial_args_v2 {
    uint32_t abi_version;
    uint32_t struct_size;
    augur_qwen35_kv8_segment_device_ptr_t query_f32;
    uint64_t query_len_bytes;
    augur_qwen35_kv8_segment_device_ptr_t key_cache_q8_0;
    uint64_t key_cache_len_bytes;
    augur_qwen35_kv8_segment_device_ptr_t value_cache_q8_0;
    uint64_t value_cache_len_bytes;
    augur_qwen35_kv8_segment_device_ptr_t partial_max_f32;
    uint64_t partial_max_len_bytes;
    augur_qwen35_kv8_segment_device_ptr_t partial_denominator_f32;
    uint64_t partial_denominator_len_bytes;
    augur_qwen35_kv8_segment_device_ptr_t partial_weighted_value_f32;
    uint64_t partial_weighted_value_len_bytes;
    uint32_t tokens;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t cache_position_base;
    uint32_t segment_begin;
    uint32_t segment_end;
    uint32_t slot_capacity_positions;
    uint32_t visible_position_end;
    uint32_t mode;
    uint32_t reserved0;
    uint32_t reserved1;
} augur_cuda_qwen35_kv8_segment_partial_args_v2;

/* Merge ABI. operation: 0=reset, 1=accumulate, 2=finalize,
 * 3=import cumulative ordered prefix (mode 3; not disjoint-page addition).
 * F64 state is private to one stream/request. All eight ranges are disjoint.
 * A nonzero per-row status poisons final output; only operation 0 clears it. */
typedef struct augur_cuda_qwen35_kv8_lse_merge_args_v2 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t partial_max_f32;
    uint64_t partial_max_len_bytes;
    uint64_t partial_denominator_f32;
    uint64_t partial_denominator_len_bytes;
    uint64_t partial_weighted_f32;
    uint64_t partial_weighted_len_bytes;
    uint64_t accumulated_max_f64;
    uint64_t accumulated_max_len_bytes;
    uint64_t accumulated_denominator_f64;
    uint64_t accumulated_denominator_len_bytes;
    uint64_t accumulated_weighted_f64;
    uint64_t accumulated_weighted_len_bytes;
    uint64_t output_f32;
    uint64_t output_len_bytes;
    uint64_t status_u32;
    uint64_t status_len_bytes;
    uint32_t query_rows;
    uint32_t head_dim;
    uint32_t operation;
    uint32_t reserved;
} augur_cuda_qwen35_kv8_lse_merge_args_v2;

/* Compact page receipt. The output is four independent u64 lanes (32 bytes)
 * computed over the exact submitted byte extent. It is an audit receipt, not
 * a model/artifact cryptographic identity. */
enum { AUGUR_CUDA_QWEN35_KV8_PAGE_DIGEST_ARGS_VERSION_V1 = 1 };
typedef struct augur_cuda_qwen35_kv8_page_digest_args_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t input;
    uint64_t input_len_bytes;
    uint64_t output;
    uint64_t output_len_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
} augur_cuda_qwen35_kv8_page_digest_args_v1;

/* The native entry points are opt-in and component-scoped.  They produce
 * per-segment stable-softmax partials; ownership/epoch receipts remain a
 * higher-level requirement of the eventual qualification runner. */
#if defined(__cplusplus)
extern "C" {
#endif
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_segment_partial_preflight_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_segment_partial_args_v2* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_segment_partial_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_segment_partial_args_v2* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_lse_merge_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_lse_merge_args_v2* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_page_digest_preflight_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_page_digest_args_v1* args);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_page_digest_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_page_digest_args_v1* args);
/* Opt-in exact V1 receipt, bounded multi-CTA XOR. No extra allocation/copy or
 * synchronization. A failure after initialization requires owner drain/poison. */
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_page_digest_parallel_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_page_digest_args_v1* args);

#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
static_assert(sizeof(augur_cuda_qwen35_kv8_segment_partial_args_v2) == 152,
              "Qwen35 KV8 segment V2 ABI size changed");
static_assert(alignof(augur_cuda_qwen35_kv8_segment_partial_args_v2) == 8,
              "Qwen35 KV8 segment V2 ABI alignment changed");
static_assert(sizeof(augur_cuda_qwen35_kv8_lse_merge_args_v2) == 152,
              "Qwen35 KV8 LSE merge V2 ABI size changed");
static_assert(sizeof(augur_cuda_qwen35_kv8_page_digest_args_v1) == 48,
              "Qwen35 KV8 page digest ABI size changed");
#endif

#endif
