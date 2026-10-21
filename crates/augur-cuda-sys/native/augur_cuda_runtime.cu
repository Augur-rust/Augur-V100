#include "augur_cuda_internal.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <sm_61_intrinsics.h>
#include <nvtx3/nvToolsExt.h>

#include <cstring>
#include <cmath>
#include <mutex>
#include <new>

static_assert(
    sizeof(augur_cuda_q4k_f32_decode_args_v1) == 72,
    "augur_cuda_q4k_f32_decode_args_v1 ABI size changed"
);
static_assert(
    sizeof(augur_cuda_q4k_f32_prefill_args_v1) == 80,
    "augur_cuda_q4k_f32_prefill_args_v1 ABI size changed"
);
static_assert(sizeof(augur_cuda_q5k_f32_decode_args_v1) == 72, "augur_cuda_q5k_f32_decode_args_v1 ABI size changed");
#if defined(AUGUR_CUDA_M306_K15_Q5_PAIRED_PROJECTION)
static_assert(sizeof(augur_cuda_q5k_f32_paired_projection_args_v1) == 104, "augur_cuda_q5k_f32_paired_projection_args_v1 ABI size changed");
static_assert(
    sizeof(augur_cuda_batch01_b4_q5k_f32_paired_projection_args_v1) == 176,
    "augur_cuda_batch01_b4_q5k_f32_paired_projection_args_v1 ABI size changed"
);
#if defined(AUGUR_CUDA_BATCH01_B8_Q5_ROW_TILE)
static_assert(
    sizeof(augur_cuda_batch01_b8_q5k_row_tile_args_v1) == 272,
    "augur_cuda_batch01_b8_q5k_row_tile_args_v1 ABI size changed"
);
#endif
#endif
static_assert(sizeof(augur_cuda_q6k_f32_decode_args_v1) == 72, "augur_cuda_q6k_f32_decode_args_v1 ABI size changed");
#if defined(AUGUR_CUDA_DSPARK_FAST_Q6_ROW_REUSE)
static_assert(
    sizeof(augur_cuda_dspark_fast_q6k_row_reuse_args_v1) == 128,
    "augur_cuda_dspark_fast_q6k_row_reuse_args_v1 ABI size changed"
);
#define AUGUR_DSPARK_FAST_Q6_OFFSET(field, expected) \
    static_assert(offsetof(augur_cuda_dspark_fast_q6k_row_reuse_args_v1, field) == expected, \
                  "augur_cuda_dspark_fast_q6k_row_reuse_args_v1 " #field " offset changed")
AUGUR_DSPARK_FAST_Q6_OFFSET(abi_version, 0);
AUGUR_DSPARK_FAST_Q6_OFFSET(struct_size, 4);
AUGUR_DSPARK_FAST_Q6_OFFSET(weight, 8);
AUGUR_DSPARK_FAST_Q6_OFFSET(weight_len_bytes, 16);
AUGUR_DSPARK_FAST_Q6_OFFSET(input, 24);
AUGUR_DSPARK_FAST_Q6_OFFSET(input_len_bytes, 32);
AUGUR_DSPARK_FAST_Q6_OFFSET(output, 40);
AUGUR_DSPARK_FAST_Q6_OFFSET(output_len_bytes, 48);
AUGUR_DSPARK_FAST_Q6_OFFSET(active_rows, 56);
AUGUR_DSPARK_FAST_Q6_OFFSET(rows, 60);
AUGUR_DSPARK_FAST_Q6_OFFSET(cols, 64);
AUGUR_DSPARK_FAST_Q6_OFFSET(input_row_stride_elements, 68);
AUGUR_DSPARK_FAST_Q6_OFFSET(output_row_stride_elements, 72);
AUGUR_DSPARK_FAST_Q6_OFFSET(blocks_per_row, 76);
AUGUR_DSPARK_FAST_Q6_OFFSET(weight_block_stride_bytes, 80);
AUGUR_DSPARK_FAST_Q6_OFFSET(reserved, 84);
#undef AUGUR_DSPARK_FAST_Q6_OFFSET
#endif
#if defined(AUGUR_CUDA_BATCH01_B4_Q6_ROW_REUSE)
static_assert(
    sizeof(augur_cuda_batch01_b4_q6k_bounded_fp64_args_v1) == 128,
    "augur_cuda_batch01_b4_q6k_bounded_fp64_args_v1 ABI size changed"
);
#if defined(AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_ROW_REUSE)
static_assert(
    sizeof(augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1) == 128,
    "augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1 ABI size changed"
);
static_assert(
    sizeof(augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1) == 80,
    "augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1 ABI size changed"
);
#define AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(type, field, expected) \
    static_assert(offsetof(type, field) == expected, #type " " #field " offset changed")
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1, abi_version, 0);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1, struct_size, 4);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1, weight, 8);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1, weight_len_bytes, 16);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1, input, 24);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1, input_len_bytes, 32);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1, output, 40);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1, output_len_bytes, 48);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1, groups, 56);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1, rows_per_group, 60);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1, cols_per_group, 64);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1, blocks_per_output_row, 68);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1,
    weight_block_stride_bytes, 72);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1, abi_version, 0);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1, struct_size, 4);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1, weight, 8);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1, weight_len_bytes, 16);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1, input, 24);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1, input_len_bytes, 56);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1, output, 64);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1, output_len_bytes, 96);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1, batch, 104);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1, groups, 108);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1, rows_per_group, 112);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1, cols_per_group, 116);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1,
    blocks_per_output_row, 120);
AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET(
    augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1,
    weight_block_stride_bytes, 124);
#undef AUGUR_BATCH01_B4_Q6_BLOCK_DIAGONAL_OFFSET
#endif
#endif
static_assert(sizeof(augur_cuda_m121_q4k_dp4a_args_v1) == 88, "augur_cuda_m121_q4k_dp4a_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_swiglu_f32_args_v1) == 64, "augur_cuda_swiglu_f32_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_silu_mul_f32_args_v1) == 64,
    "augur_cuda_silu_mul_f32_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_bf16_f32_matvec_args_v1) == 64, "augur_cuda_bf16_f32_matvec_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_q8_0_f32_matvec_args_v1) == 64, "augur_cuda_q8_0_f32_matvec_args_v1 ABI size changed");
static_assert(alignof(augur_cuda_q8_0_f32_matvec_args_v1) == 8, "augur_cuda_q8_0_f32_matvec_args_v1 ABI alignment changed");
static_assert(offsetof(augur_cuda_q8_0_f32_matvec_args_v1, weight) == 8, "augur_cuda_q8_0_f32_matvec_args_v1 weight offset changed");
static_assert(offsetof(augur_cuda_q8_0_f32_matvec_args_v1, input) == 24, "augur_cuda_q8_0_f32_matvec_args_v1 input offset changed");
static_assert(offsetof(augur_cuda_q8_0_f32_matvec_args_v1, output) == 40, "augur_cuda_q8_0_f32_matvec_args_v1 output offset changed");
static_assert(sizeof(augur_cuda_q8_1_activation_quantize_args_v1) == 48,
    "augur_cuda_q8_1_activation_quantize_args_v1 ABI size changed");
static_assert(alignof(augur_cuda_q8_1_activation_quantize_args_v1) == 8,
    "augur_cuda_q8_1_activation_quantize_args_v1 ABI alignment changed");
static_assert(sizeof(augur_cuda_q8_0_q8_1_matvec_args_v1) == 64,
    "augur_cuda_q8_0_q8_1_matvec_args_v1 ABI size changed");
static_assert(alignof(augur_cuda_q8_0_q8_1_matvec_args_v1) == 8,
    "augur_cuda_q8_0_q8_1_matvec_args_v1 ABI alignment changed");
static_assert(sizeof(augur_cuda_nvfp4_sm70_f32_matvec_args_v1) == 72,
    "augur_cuda_nvfp4_sm70_f32_matvec_args_v1 ABI size changed");
static_assert(alignof(augur_cuda_nvfp4_sm70_f32_matvec_args_v1) == 8,
    "augur_cuda_nvfp4_sm70_f32_matvec_args_v1 ABI alignment changed");
static_assert(sizeof(augur_cuda_bf16_token_row_gather_args_v1) == 64,
    "augur_cuda_bf16_token_row_gather_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_rmsnorm_f32_args_v1) == 64, "augur_cuda_rmsnorm_f32_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_rope_f32_args_v1) == 80, "augur_cuda_rope_f32_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_dflash_rope_f32_args_v1) == 72, "augur_cuda_dflash_rope_f32_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_dflash_cache_attention_f32_args_v1) == 104, "augur_cuda_dflash_cache_attention_f32_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_f8e4m3_f32_matvec_args_v1) == 80, "augur_cuda_f8e4m3_f32_matvec_args_v1 ABI size changed");
static_assert(
    sizeof(augur_cuda_add_f32_args_v1) == 64,
    "augur_cuda_add_f32_args_v1 ABI size changed"
);
static_assert(
    sizeof(augur_cuda_resident_moe_f32_args_v1) == 160,
    "augur_cuda_resident_moe_f32_args_v1 ABI size changed"
);
static_assert(
    sizeof(augur_cuda_nvfp4_aot_args_v1) == 112,
    "augur_cuda_nvfp4_aot_args_v1 ABI size changed"
);
static_assert(
    sizeof(augur_cuda_nvfp4_interstage_args_v1) == 104,
    "augur_cuda_nvfp4_interstage_args_v1 ABI size changed"
);
static_assert(sizeof(augur_cuda_bf16_gemm_args_v1) == 64, "augur_cuda_bf16_gemm_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m54_shared_interstage_args_v1) == 64, "augur_cuda_m54_shared_interstage_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m54_sum_args_v1) == 120, "augur_cuda_m54_sum_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m54_router_args_v1) == 104, "augur_cuda_m54_router_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m58_router_group_args_v1) == 224, "augur_cuda_m58_router_group_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m95_router_gather_args_v1) == 176, "augur_cuda_m95_router_gather_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m102_decode_router_args_v1) == 208, "augur_cuda_m102_decode_router_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m204_hash_router_args_v1) == 168, "augur_cuda_m204_hash_router_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m306_k38_hash_router_selected6_args_v1) == 168, "augur_cuda_m306_k38_hash_router_selected6_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m211_learned_router_args_v1) == 144, "augur_cuda_m211_learned_router_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m211_learned_router_args_v2) == 152, "augur_cuda_m211_learned_router_args_v2 ABI size changed");
static_assert(sizeof(augur_cuda_m214_learned_route_telemetry_args_v1) == 120, "augur_cuda_m214_learned_route_telemetry_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m205_route_bf16_args_v1) == 72, "augur_cuda_m205_route_bf16_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m111_sparse_attention_f32_args_v1) == 136, "augur_cuda_m111_sparse_attention_f32_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_weighted_accum_f32_args_v1) == 56, "augur_cuda_weighted_accum_f32_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_compensated_accum_f32_args_v1) == 72, "augur_cuda_compensated_accum_f32_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m58_scatter_args_v1) == 88, "augur_cuda_m58_scatter_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m58_finalize_args_v1) == 88, "augur_cuda_m58_finalize_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m139_ratio4_staged_args_v1) == 264, "augur_cuda_m139_ratio4_staged_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m142_ordered_topk_args_v1) == 80, "augur_cuda_m142_ordered_topk_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m144_ratio128_staged_args_v1) == 216, "augur_cuda_m144_ratio128_staged_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m283_device_view_v1) == 32, "augur_cuda_m283_device_view_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m283_kv_transition_args_v1) == 256, "augur_cuda_m283_kv_transition_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m285_fill_span_args_v1) == 56, "augur_cuda_m285_fill_span_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m289_device_view_v1) == 32, "augur_cuda_m289_device_view_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m289_rope_position_args_v1) == 128, "augur_cuda_m289_rope_position_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m289_index_score_args_v1) == 192, "augur_cuda_m289_index_score_args_v1 ABI size changed");
static_assert(sizeof(augur_cuda_m289_attention_args_v1) == 288, "augur_cuda_m289_attention_args_v1 ABI size changed");
static_assert(offsetof(augur_cuda_m289_rope_position_args_v1, input) == 64, "augur_cuda_m289 rope input offset changed");
static_assert(offsetof(augur_cuda_m289_rope_position_args_v1, output) == 96, "augur_cuda_m289 rope output offset changed");
static_assert(offsetof(augur_cuda_m289_index_score_args_v1, query) == 64, "augur_cuda_m289 index query offset changed");
static_assert(offsetof(augur_cuda_m289_index_score_args_v1, compressed) == 96, "augur_cuda_m289 index compressed offset changed");
static_assert(offsetof(augur_cuda_m289_index_score_args_v1, head_weight) == 128, "augur_cuda_m289 index weight offset changed");
static_assert(offsetof(augur_cuda_m289_index_score_args_v1, scores) == 160, "augur_cuda_m289 index scores offset changed");
static_assert(offsetof(augur_cuda_m289_attention_args_v1, query) == 64, "augur_cuda_m289 attention query offset changed");
static_assert(offsetof(augur_cuda_m289_attention_args_v1, raw) == 96, "augur_cuda_m289 attention raw offset changed");
static_assert(offsetof(augur_cuda_m289_attention_args_v1, compressed) == 128, "augur_cuda_m289 attention compressed offset changed");
static_assert(offsetof(augur_cuda_m289_attention_args_v1, ids) == 160, "augur_cuda_m289 attention ids offset changed");
static_assert(offsetof(augur_cuda_m289_attention_args_v1, sink) == 192, "augur_cuda_m289 attention sink offset changed");
static_assert(offsetof(augur_cuda_m289_attention_args_v1, weighted) == 224, "augur_cuda_m289 attention weighted offset changed");
static_assert(offsetof(augur_cuda_m289_attention_args_v1, inverse) == 256, "augur_cuda_m289 attention inverse offset changed");
#define AUGUR_M283_OFFSET(type, field, expected) static_assert(offsetof(type, field) == expected, #type " " #field " offset changed")
AUGUR_M283_OFFSET(augur_cuda_m283_device_view_v1, base, 0);
AUGUR_M283_OFFSET(augur_cuda_m283_device_view_v1, allocation_extent_bytes, 8);
AUGUR_M283_OFFSET(augur_cuda_m283_device_view_v1, view_offset_bytes, 16);
AUGUR_M283_OFFSET(augur_cuda_m283_device_view_v1, view_extent_bytes, 24);
AUGUR_M283_OFFSET(augur_cuda_m283_kv_transition_args_v1, abi_version, 0);
AUGUR_M283_OFFSET(augur_cuda_m283_kv_transition_args_v1, struct_size, 4);
AUGUR_M283_OFFSET(augur_cuda_m283_kv_transition_args_v1, variant, 8);
AUGUR_M283_OFFSET(augur_cuda_m283_kv_transition_args_v1, reserved0, 12);
AUGUR_M283_OFFSET(augur_cuda_m283_kv_transition_args_v1, layer, 16);
AUGUR_M283_OFFSET(augur_cuda_m283_kv_transition_args_v1, capacity, 20);
AUGUR_M283_OFFSET(augur_cuda_m283_kv_transition_args_v1, position, 24);
AUGUR_M283_OFFSET(augur_cuda_m283_kv_transition_args_v1, next_position, 28);
AUGUR_M283_OFFSET(augur_cuda_m283_kv_transition_args_v1, compression_ratio, 32);
AUGUR_M283_OFFSET(augur_cuda_m283_kv_transition_args_v1, source_start_position, 36);
AUGUR_M283_OFFSET(augur_cuda_m283_kv_transition_args_v1, source_rows, 40);
AUGUR_M283_OFFSET(augur_cuda_m283_kv_transition_args_v1, compressed_row_index, 44);
AUGUR_M283_OFFSET(augur_cuda_m283_kv_transition_args_v1, compressed_position, 48);
AUGUR_M283_OFFSET(augur_cuda_m283_kv_transition_args_v1, compressed_main_rows_before, 52);
AUGUR_M283_OFFSET(augur_cuda_m283_kv_transition_args_v1, compressed_index_rows_before, 56);
AUGUR_M283_OFFSET(augur_cuda_m283_kv_transition_args_v1, reserved1, 60);
static_assert(offsetof(augur_cuda_m283_kv_transition_args_v1, raw_current) == 64, "augur_cuda_m283 raw_current offset changed");
static_assert(offsetof(augur_cuda_m283_kv_transition_args_v1, raw_cache) == 96, "augur_cuda_m283 raw_cache offset changed");
static_assert(offsetof(augur_cuda_m283_kv_transition_args_v1, compressed_main_current) == 128, "augur_cuda_m283 main current offset changed");
static_assert(offsetof(augur_cuda_m283_kv_transition_args_v1, compressed_main_cache) == 160, "augur_cuda_m283 main cache offset changed");
static_assert(offsetof(augur_cuda_m283_kv_transition_args_v1, compressed_index_current) == 192, "augur_cuda_m283 index current offset changed");
static_assert(offsetof(augur_cuda_m283_kv_transition_args_v1, compressed_index_cache) == 224, "augur_cuda_m283 index cache offset changed");
#undef AUGUR_M283_OFFSET
#define AUGUR_M285_OFFSET(field, expected) static_assert(offsetof(augur_cuda_m285_fill_span_args_v1, field) == expected, "augur_cuda_m285 " #field " offset changed")
AUGUR_M285_OFFSET(abi_version, 0);
AUGUR_M285_OFFSET(struct_size, 4);
AUGUR_M285_OFFSET(allocation_base, 8);
AUGUR_M285_OFFSET(allocation_extent_bytes, 16);
AUGUR_M285_OFFSET(view_offset_bytes, 24);
AUGUR_M285_OFFSET(view_extent_bytes, 32);
AUGUR_M285_OFFSET(pattern_seed, 40);
AUGUR_M285_OFFSET(reserved, 48);
#undef AUGUR_M285_OFFSET

struct augur_cuda_graph_exec_handle {
    int32_t ordinal;
    cudaGraphExec_t graph;
    uint64_t node_count;
};

static augur_cuda_result_v1 augur_ok(void) {
    return {AUGUR_CUDA_STATUS_OK, static_cast<int32_t>(cudaSuccess)};
}

static augur_cuda_result_v1 augur_error(
    augur_cuda_status_t status,
    cudaError_t raw_status
) {
    return {status, static_cast<int32_t>(raw_status)};
}

static augur_cuda_result_v1 augur_runtime_error(cudaError_t error) {
    if (error == cudaErrorMemoryAllocation) {
        return augur_error(AUGUR_CUDA_STATUS_OUT_OF_MEMORY, error);
    }

    if (error == cudaErrorInvalidDevice || error == cudaErrorNoDevice) {
        return augur_error(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE, error);
    }

    return augur_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, error);
}

static augur_cuda_result_v1 augur_set_device(int32_t ordinal) {
    if (ordinal < 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    const cudaError_t error = cudaSetDevice(ordinal);

    if (error != cudaSuccess) {
        return augur_runtime_error(error);
    }

    return augur_ok();
}

static augur_cuda_result_v1 augur_validate_stream_device(
    int32_t ordinal,
    augur_cuda_stream_t stream
) {
    if (stream == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    return augur_set_device(ordinal);
}

static thread_local bool augur_m306_steady_token_nvtx_active = false;
static thread_local int32_t augur_batch01_b4_row_main_nvtx_active = -1;

static const char* const augur_batch01_b4_row_main_nvtx_names_v1[4] = {
    "CQ_BATCH01_B4_ROW0_MAIN_STREAM_START_V1",
    "CQ_BATCH01_B4_ROW1_MAIN_STREAM_START_V1",
    "CQ_BATCH01_B4_ROW2_MAIN_STREAM_START_V1",
    "CQ_BATCH01_B4_ROW3_MAIN_STREAM_START_V1",
};

static augur_cuda_result_v1 augur_nvtx_result(int result) {
    // Nsight Systems 2024.6.2 on Windows records the event but returns -1
    // when it does not expose a nesting level to the target. This wrapper
    // enforces nesting itself, so both opaque-level sentinels are successful
    // submissions; values below the NVTX sentinel remain failures.
    if (result >= NVTX_NO_PUSH_POP_TRACKING) {
        return augur_ok();
    }

    return {AUGUR_CUDA_STATUS_RUNTIME_FAILURE, static_cast<int32_t>(result)};
}

extern "C" uint32_t AUGUR_CUDA_CALL augur_cuda_abi_version(void) {
    return AUGUR_CUDA_ABI_VERSION_V1;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_device_count(
    int32_t* out_count
) {
    if (out_count == nullptr) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    *out_count = 0;

    int count = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);

    if (error != cudaSuccess) {
        return augur_runtime_error(error);
    }

    *out_count = static_cast<int32_t>(count);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvtx_m306_steady_token_begin_v1(void) {
    if (augur_m306_steady_token_nvtx_active
        || augur_batch01_b4_row_main_nvtx_active != -1) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    const int result = nvtxRangePushA("AUGUR_M306_STEADY_TOKEN_V1");
    const augur_cuda_result_v1 checked = augur_nvtx_result(result);
    if (checked.status != AUGUR_CUDA_STATUS_OK) {
        return checked;
    }

    augur_m306_steady_token_nvtx_active = true;
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvtx_m306_steady_token_end_v1(void) {
    if (!augur_m306_steady_token_nvtx_active
        || augur_batch01_b4_row_main_nvtx_active != -1) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    const int result = nvtxRangePop();
    const augur_cuda_result_v1 checked = augur_nvtx_result(result);
    if (checked.status != AUGUR_CUDA_STATUS_OK) {
        return checked;
    }

    augur_m306_steady_token_nvtx_active = false;
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvtx_batch01_b4_row_main_begin_v1(uint32_t row_index) {
    if (!augur_m306_steady_token_nvtx_active
        || row_index >= 4
        || augur_batch01_b4_row_main_nvtx_active != -1) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    const int result = nvtxRangePushA(
        augur_batch01_b4_row_main_nvtx_names_v1[row_index]);
    const augur_cuda_result_v1 checked = augur_nvtx_result(result);
    if (checked.status != AUGUR_CUDA_STATUS_OK) {
        return checked;
    }

    augur_batch01_b4_row_main_nvtx_active = static_cast<int32_t>(row_index);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvtx_batch01_b4_row_main_end_v1(uint32_t row_index) {
    if (!augur_m306_steady_token_nvtx_active
        || row_index >= 4
        || augur_batch01_b4_row_main_nvtx_active
            != static_cast<int32_t>(row_index)) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    const int result = nvtxRangePop();
    const augur_cuda_result_v1 checked = augur_nvtx_result(result);
    if (checked.status != AUGUR_CUDA_STATUS_OK) {
        return checked;
    }

    augur_batch01_b4_row_main_nvtx_active = -1;
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_get_device_properties(
    int32_t ordinal,
    augur_cuda_device_properties_v1* out_properties
) {
    if (out_properties == nullptr || ordinal < 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    if (out_properties->abi_version != AUGUR_CUDA_ABI_VERSION_V1 ||
        out_properties->struct_size != sizeof(augur_cuda_device_properties_v1)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }

    cudaDeviceProp properties{};
    const cudaError_t error = cudaGetDeviceProperties(&properties, ordinal);

    if (error != cudaSuccess) {
        return augur_runtime_error(error);
    }

    out_properties->ordinal = ordinal;
    out_properties->compute_major =
        static_cast<uint16_t>(properties.major);
    out_properties->compute_minor =
        static_cast<uint16_t>(properties.minor);
    out_properties->total_global_memory_bytes =
        static_cast<uint64_t>(properties.totalGlobalMem);

    std::memset(
        out_properties->name,
        0,
        sizeof(out_properties->name)
    );
    std::strncpy(
        out_properties->name,
        properties.name,
        sizeof(out_properties->name) - 1
    );

    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_get_device_uuid_v1(
    int32_t ordinal,
    uint8_t* out_uuid,
    uint32_t out_uuid_bytes
) {
    constexpr uint32_t uuid_bytes = 16;
    if (ordinal < 0 || out_uuid == nullptr || out_uuid_bytes != uuid_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    std::memset(out_uuid, 0, uuid_bytes);
    cudaDeviceProp properties{};
    const cudaError_t error = cudaGetDeviceProperties(&properties, ordinal);
    if (error != cudaSuccess) {
        return augur_runtime_error(error);
    }
    std::memcpy(out_uuid, properties.uuid.bytes, uuid_bytes);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_mem_get_info_v1(
    int32_t ordinal,
    uint64_t* out_free_bytes,
    uint64_t* out_total_bytes
) {
    if (ordinal < 0 || out_free_bytes == nullptr || out_total_bytes == nullptr) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    *out_free_bytes = 0;
    *out_total_bytes = 0;
    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) {
        return augur_runtime_error(error);
    }
    if (total_bytes == 0 || free_bytes > total_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, cudaSuccess);
    }
    *out_free_bytes = static_cast<uint64_t>(free_bytes);
    *out_total_bytes = static_cast<uint64_t>(total_bytes);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_stream_create(
    int32_t ordinal,
    augur_cuda_stream_t* out_stream
) {
    if (out_stream == nullptr || ordinal < 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    *out_stream = nullptr;

#if defined(AUGUR_CUDA_FAST_SCHEDULE_SPIN_CANDIDATE)
    // This process-local candidate is intentionally scoped to the qualified
    // single-GPU ordinal.  The first CUDA runtime call fixes the primary
    // context wait policy; later stream owners reuse that decision.
    if (ordinal != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    static std::once_flag schedule_once;
    static cudaError_t schedule_error = cudaSuccess;
    std::call_once(schedule_once, [] {
        schedule_error = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    });
    if (schedule_error != cudaSuccess) {
        return augur_runtime_error(schedule_error);
    }
#endif

    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }

    augur_cuda_stream_handle* handle =
        new (std::nothrow) augur_cuda_stream_handle{};

    if (handle == nullptr) {
        return augur_error(AUGUR_CUDA_STATUS_OUT_OF_MEMORY, cudaSuccess);
    }

    handle->ordinal = ordinal;
    handle->stream = nullptr;

    const cudaError_t error = cudaStreamCreateWithFlags(
        &handle->stream,
        cudaStreamNonBlocking
    );

    if (error != cudaSuccess) {
        delete handle;
        return augur_runtime_error(error);
    }

    *out_stream = handle;
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_stream_synchronize(
    augur_cuda_stream_t stream
) {
    if (stream == nullptr) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    augur_cuda_result_v1 selected = augur_set_device(stream->ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }

    const cudaError_t error = cudaStreamSynchronize(stream->stream);

    if (error != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_COMPLETION_FAILURE, error);
    }

    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_stream_wait_event_v1(
    augur_cuda_stream_t stream,
    augur_cuda_event_t event
) {
    if (stream == nullptr || event == nullptr
        || stream->ordinal != event->ordinal
        || stream->stream == nullptr || event->event == nullptr) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    augur_cuda_result_v1 selected = augur_set_device(stream->ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }

    const cudaError_t error = cudaStreamWaitEvent(stream->stream, event->event, 0);
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_stream_destroy(
    augur_cuda_stream_t stream
) {
    if (stream == nullptr) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    augur_cuda_result_v1 selected = augur_set_device(stream->ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }

    const cudaError_t error = cudaStreamDestroy(stream->stream);

    if (error != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, error);
    }

    delete stream;
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_graph_capture_begin_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream
) {
    augur_cuda_result_v1 selected = augur_validate_stream_device(ordinal, stream);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    const cudaError_t error = cudaStreamBeginCapture(
        stream->stream,
        cudaStreamCaptureModeThreadLocal
    );
    return error == cudaSuccess ? augur_ok() : augur_runtime_error(error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_graph_capture_end_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_graph_exec_t* out_graph
) {
    if (out_graph == nullptr) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    *out_graph = nullptr;
    augur_cuda_result_v1 selected = augur_validate_stream_device(ordinal, stream);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    cudaGraph_t captured = nullptr;
    cudaError_t error = cudaStreamEndCapture(stream->stream, &captured);
    if (error != cudaSuccess || captured == nullptr) {
        if (captured != nullptr) {
            cudaGraphDestroy(captured);
        }
        return error == cudaSuccess
            ? augur_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, cudaErrorInvalidValue)
            : augur_runtime_error(error);
    }
    size_t node_count = 0;
    error = cudaGraphGetNodes(captured, nullptr, &node_count);
    if (error != cudaSuccess || node_count == 0) {
        cudaGraphDestroy(captured);
        return error == cudaSuccess
            ? augur_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, cudaErrorInvalidValue)
            : augur_runtime_error(error);
    }
    cudaGraphExec_t executable = nullptr;
    error = cudaGraphInstantiateWithFlags(&executable, captured, 0);
    const cudaError_t destroy_error = cudaGraphDestroy(captured);
    if (error != cudaSuccess) {
        return augur_runtime_error(error);
    }
    if (destroy_error != cudaSuccess) {
        cudaGraphExecDestroy(executable);
        return augur_runtime_error(destroy_error);
    }
    augur_cuda_graph_exec_handle* handle =
        new (std::nothrow) augur_cuda_graph_exec_handle{};
    if (handle == nullptr) {
        cudaGraphExecDestroy(executable);
        return augur_error(AUGUR_CUDA_STATUS_OUT_OF_MEMORY, cudaSuccess);
    }
    handle->ordinal = ordinal;
    handle->graph = executable;
    handle->node_count = static_cast<uint64_t>(node_count);
    *out_graph = handle;
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_graph_capture_abort_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream
) {
    augur_cuda_result_v1 selected = augur_validate_stream_device(ordinal, stream);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    cudaGraph_t captured = nullptr;
    const cudaError_t error = cudaStreamEndCapture(stream->stream, &captured);
    if (captured != nullptr) {
        const cudaError_t destroy_error = cudaGraphDestroy(captured);
        if (error == cudaSuccess && destroy_error != cudaSuccess) {
            return augur_runtime_error(destroy_error);
        }
    }
    if (error == cudaSuccess || error == cudaErrorStreamCaptureInvalidated) {
        cudaGetLastError();
        return augur_ok();
    }
    return augur_runtime_error(error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_graph_launch_v1(
    int32_t ordinal,
    augur_cuda_graph_exec_t graph,
    augur_cuda_stream_t stream
) {
    augur_cuda_result_v1 selected = augur_validate_stream_device(ordinal, stream);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    if (graph == nullptr || graph->ordinal != ordinal || graph->graph == nullptr) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const cudaError_t error = cudaGraphLaunch(graph->graph, stream->stream);
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_graph_node_count_v1(
    augur_cuda_graph_exec_t graph,
    uint64_t* out_node_count
) {
    if (graph == nullptr || out_node_count == nullptr || graph->graph == nullptr) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    *out_node_count = graph->node_count;
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_graph_destroy_v1(
    augur_cuda_graph_exec_t graph
) {
    if (graph == nullptr || graph->graph == nullptr) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_set_device(graph->ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    const cudaError_t error = cudaGraphExecDestroy(graph->graph);
    if (error != cudaSuccess) {
        return augur_runtime_error(error);
    }
    graph->graph = nullptr;
    delete graph;
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_event_create(
    int32_t ordinal,
    augur_cuda_event_t* out_event
) {
    if (out_event == nullptr || ordinal < 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    *out_event = nullptr;
    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    augur_cuda_event_handle* handle = new (std::nothrow) augur_cuda_event_handle{};
    if (handle == nullptr) {
        return augur_error(AUGUR_CUDA_STATUS_OUT_OF_MEMORY, cudaSuccess);
    }
    handle->ordinal = ordinal;
    handle->event = nullptr;
    const cudaError_t error = cudaEventCreate(&handle->event);
    if (error != cudaSuccess) {
        delete handle;
        return augur_runtime_error(error);
    }
    *out_event = handle;
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_event_record(
    augur_cuda_event_t event,
    augur_cuda_stream_t stream
) {
    if (event == nullptr || stream == nullptr || event->ordinal != stream->ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_set_device(event->ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    const cudaError_t error = cudaEventRecord(event->event, stream->stream);
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_event_query_complete_v1(
    augur_cuda_event_t event,
    uint8_t* out_complete
) {
    if (event == nullptr || out_complete == nullptr) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    *out_complete = 0;
    augur_cuda_result_v1 selected = augur_set_device(event->ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    const cudaError_t error = cudaEventQuery(event->event);
    if (error == cudaSuccess) {
        *out_complete = 1;
        return augur_ok();
    }
    if (error == cudaErrorNotReady) {
        return augur_ok();
    }
    return augur_error(AUGUR_CUDA_STATUS_COMPLETION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_event_synchronize(
    augur_cuda_event_t event
) {
    if (event == nullptr) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_set_device(event->ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    const cudaError_t error = cudaEventSynchronize(event->event);
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_COMPLETION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_event_elapsed_ms(
    augur_cuda_event_t started,
    augur_cuda_event_t stopped,
    float* out_milliseconds
) {
    if (started == nullptr || stopped == nullptr || out_milliseconds == nullptr
        || started->ordinal != stopped->ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    *out_milliseconds = 0.0f;
    augur_cuda_result_v1 selected = augur_set_device(started->ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    const cudaError_t error = cudaEventElapsedTime(
        out_milliseconds, started->event, stopped->event);
    return error == cudaSuccess ? augur_ok() : augur_runtime_error(error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_event_destroy(
    augur_cuda_event_t event
) {
    if (event == nullptr) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_set_device(event->ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    const cudaError_t error = cudaEventDestroy(event->event);
    if (error != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, error);
    }
    delete event;
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_device_alloc(
    int32_t ordinal,
    size_t len_bytes,
    augur_cuda_device_ptr_t* out_ptr
) {
    if (out_ptr == nullptr || ordinal < 0 || len_bytes == 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    *out_ptr = 0;

    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }

    void* device_ptr = nullptr;
    const cudaError_t error = cudaMalloc(&device_ptr, len_bytes);

    if (error != cudaSuccess) {
        return augur_runtime_error(error);
    }

    *out_ptr = static_cast<augur_cuda_device_ptr_t>(
        reinterpret_cast<uintptr_t>(device_ptr)
    );

    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_device_free(
    int32_t ordinal,
    augur_cuda_device_ptr_t ptr
) {
    if (ordinal < 0 || ptr == 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }

    void* device_ptr = reinterpret_cast<void*>(
        static_cast<uintptr_t>(ptr)
    );

    const cudaError_t error = cudaFree(device_ptr);

    if (error != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, error);
    }

    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memset_zero_sync_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_device_ptr_t dst,
    size_t dst_len_bytes
) {
    if (dst == 0 || dst_len_bytes == 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_validate_stream_device(ordinal, stream);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    void* destination = reinterpret_cast<void*>(static_cast<uintptr_t>(dst));
    const cudaError_t submitted = cudaMemsetAsync(
        destination,
        0,
        dst_len_bytes,
        stream->stream
    );
    if (submitted != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, submitted);
    }
    const cudaError_t completion = cudaStreamSynchronize(stream->stream);
    return completion == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_COMPLETION_FAILURE, completion);
}


extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_host_alloc_pinned_v1(
    int32_t ordinal,
    size_t len_bytes,
    void** out_ptr
) {
    if (out_ptr == nullptr || ordinal < 0 || len_bytes == 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    *out_ptr = nullptr;
    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    void* host_ptr = nullptr;
    const cudaError_t error = cudaHostAlloc(&host_ptr, len_bytes, cudaHostAllocDefault);
    if (error != cudaSuccess) {
        return augur_runtime_error(error);
    }
    std::memset(host_ptr, 0, len_bytes);
    *out_ptr = host_ptr;
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_host_alloc_mapped_v1(
    int32_t ordinal,
    size_t len_bytes,
    void** out_host_ptr,
    augur_cuda_device_ptr_t* out_device_ptr
) {
    if (out_host_ptr == nullptr || out_device_ptr == nullptr
        || ordinal < 0 || len_bytes == 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    *out_host_ptr = nullptr;
    *out_device_ptr = 0;
    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    void* host_ptr = nullptr;
    cudaError_t error = cudaHostAlloc(&host_ptr, len_bytes, cudaHostAllocMapped);
    if (error != cudaSuccess) {
        return augur_runtime_error(error);
    }
    void* device_ptr = nullptr;
    error = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    if (error != cudaSuccess || device_ptr == nullptr) {
        const cudaError_t free_error = cudaFreeHost(host_ptr);
        return augur_runtime_error(error != cudaSuccess ? error : free_error);
    }
    std::memset(host_ptr, 0, len_bytes);
    *out_host_ptr = host_ptr;
    *out_device_ptr = static_cast<augur_cuda_device_ptr_t>(
        reinterpret_cast<uintptr_t>(device_ptr));
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_host_free_pinned_v1(
    int32_t ordinal,
    void* ptr
) {
    if (ptr == nullptr || ordinal < 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    const cudaError_t error = cudaFreeHost(ptr);
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_host_free_mapped_sync_v1(
    int32_t ordinal,
    void* ptr
) {
    if (ptr == nullptr || ordinal < 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    const cudaError_t completion = cudaDeviceSynchronize();
    if (completion != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_COMPLETION_FAILURE, completion);
    }
    const cudaError_t released = cudaFreeHost(ptr);
    return released == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, released);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_h2d_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_device_ptr_t dst,
    size_t dst_offset_bytes,
    size_t dst_len_bytes,
    const void* src,
    size_t src_len_bytes,
    size_t copy_len_bytes
) {
    if (dst == 0 || src == nullptr || copy_len_bytes == 0
        || dst_offset_bytes > dst_len_bytes
        || copy_len_bytes > dst_len_bytes - dst_offset_bytes
        || copy_len_bytes > src_len_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_validate_stream_device(ordinal, stream);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    unsigned char* destination = reinterpret_cast<unsigned char*>(
        static_cast<uintptr_t>(dst)
    );
    const cudaError_t error = cudaMemcpyAsync(
        destination + dst_offset_bytes,
        src,
        copy_len_bytes,
        cudaMemcpyHostToDevice,
        stream->stream
    );
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_h2d_sync(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_device_ptr_t dst,
    size_t dst_offset_bytes,
    const void* src,
    size_t src_len_bytes,
    size_t copy_len_bytes
) {
    if (dst == 0 || src == nullptr || copy_len_bytes > src_len_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    augur_cuda_result_v1 selected =
        augur_validate_stream_device(ordinal, stream);

    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }

    void* dst_base = reinterpret_cast<void*>(
        static_cast<uintptr_t>(dst)
    );

    void* dst_offset = static_cast<void*>(
        static_cast<unsigned char*>(dst_base) + dst_offset_bytes
    );

    const cudaError_t error = cudaMemcpyAsync(
        dst_offset,
        src,
        copy_len_bytes,
        cudaMemcpyHostToDevice,
        stream->stream
    );

    if (error != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    }

    const cudaError_t completion = cudaStreamSynchronize(stream->stream);

    if (completion != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_COMPLETION_FAILURE, completion);
    }

    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_h2d_sync_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_device_ptr_t dst,
    size_t dst_offset_bytes,
    size_t dst_len_bytes,
    const void* src,
    size_t src_len_bytes,
    size_t copy_len_bytes
) {
    if (dst == 0 || src == nullptr || copy_len_bytes == 0
        || dst_offset_bytes > dst_len_bytes
        || copy_len_bytes > dst_len_bytes - dst_offset_bytes
        || copy_len_bytes > src_len_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_validate_stream_device(ordinal, stream);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    unsigned char* destination = reinterpret_cast<unsigned char*>(
        static_cast<uintptr_t>(dst)
    );
    const cudaError_t error = cudaMemcpyAsync(
        destination + dst_offset_bytes,
        src,
        copy_len_bytes,
        cudaMemcpyHostToDevice,
        stream->stream
    );
    if (error != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    }
    const cudaError_t completion = cudaStreamSynchronize(stream->stream);
    return completion == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_COMPLETION_FAILURE, completion);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_d2h_sync(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    void* dst,
    size_t dst_len_bytes,
    augur_cuda_device_ptr_t src,
    size_t src_offset_bytes,
    size_t copy_len_bytes
) {
    if (dst == nullptr || src == 0 || copy_len_bytes > dst_len_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    augur_cuda_result_v1 selected =
        augur_validate_stream_device(ordinal, stream);

    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }

    void* src_base = reinterpret_cast<void*>(
        static_cast<uintptr_t>(src)
    );

    const void* src_offset = static_cast<const void*>(
        static_cast<const unsigned char*>(src_base) + src_offset_bytes
    );

    const cudaError_t error = cudaMemcpyAsync(
        dst,
        src_offset,
        copy_len_bytes,
        cudaMemcpyDeviceToHost,
        stream->stream
    );

    if (error != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    }

    const cudaError_t completion = cudaStreamSynchronize(stream->stream);

    if (completion != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_COMPLETION_FAILURE, completion);
    }

    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_d2h_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    void* dst,
    size_t dst_len_bytes,
    augur_cuda_device_ptr_t src,
    size_t src_offset_bytes,
    size_t src_len_bytes,
    size_t copy_len_bytes
) {
    if (dst == nullptr || src == 0 || copy_len_bytes == 0
        || copy_len_bytes > dst_len_bytes
        || src_offset_bytes > src_len_bytes
        || copy_len_bytes > src_len_bytes - src_offset_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_validate_stream_device(ordinal, stream);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    const unsigned char* source = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(src)
    );
    // The destination is required to be page-locked host memory. The Rust
    // owner allocates it through cudaHostAlloc and keeps it live until the
    // completion event is proven; this entry point intentionally does not
    // synchronize the stream.
    const cudaError_t error = cudaMemcpyAsync(
        dst,
        source + src_offset_bytes,
        copy_len_bytes,
        cudaMemcpyDeviceToHost,
        stream->stream
    );
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_d2h_sync_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    void* dst,
    size_t dst_len_bytes,
    augur_cuda_device_ptr_t src,
    size_t src_offset_bytes,
    size_t src_len_bytes,
    size_t copy_len_bytes
) {
    if (dst == nullptr || src == 0 || copy_len_bytes == 0
        || copy_len_bytes > dst_len_bytes
        || src_offset_bytes > src_len_bytes
        || copy_len_bytes > src_len_bytes - src_offset_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_validate_stream_device(ordinal, stream);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    const unsigned char* source = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(src)
    );
    const cudaError_t error = cudaMemcpyAsync(
        dst,
        source + src_offset_bytes,
        copy_len_bytes,
        cudaMemcpyDeviceToHost,
        stream->stream
    );
    if (error != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    }
    const cudaError_t completion = cudaStreamSynchronize(stream->stream);
    return completion == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_COMPLETION_FAILURE, completion);
}

namespace {
struct IndependentCopyBatch {
    augur_cuda_d2d_copy_entry_v1 entries[128];
    uint32_t tile_ends[128];
    uint32_t count;
    uint32_t tiles;
};
static_assert(sizeof(IndependentCopyBatch) <= 4096, "bounded copy launch parameters");
static_assert(sizeof(augur_cuda_d2d_copy_entry_v1) == 24, "copy entry ABI");

__global__ void augur_independent_copy_4k_kernel_v1(IndependentCopyBatch batch) {
    uint32_t lo = 0, hi = batch.count;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (blockIdx.x < batch.tile_ends[mid]) hi = mid;
        else lo = mid + 1;
    }
    const auto entry = batch.entries[lo];
    const uint32_t previous = lo ? batch.tile_ends[lo - 1] : 0;
    const uint64_t offset = uint64_t(blockIdx.x - previous) * 4096U;
    const uint64_t remainder = entry.bytes - offset;
    const uint32_t words = static_cast<uint32_t>((remainder < 4096U ? remainder : 4096U) / sizeof(uint4));
    const auto* source = reinterpret_cast<const uint4*>(static_cast<uintptr_t>(entry.source + offset));
    auto* destination = reinterpret_cast<uint4*>(static_cast<uintptr_t>(entry.destination + offset));
    for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) destination[i] = source[i];
}
bool copy_ranges_overlap(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) {
    return a < b + bn && b < a + an;
}
} // namespace

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_d2d_independent_batch_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_d2d_copy_entry_v1* entries, uint32_t count) {
    if (entries == nullptr || count == 0U || count > 128U)
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    IndependentCopyBatch batch{};
    batch.count = count;
    uint64_t tiles = 0;
    // Complete all range/dependency checks before submitting any work.
    for (uint32_t i = 0; i < count; ++i) {
        const auto e = entries[i];
        if (e.source == 0U || e.destination == 0U || e.bytes == 0U
            || (e.source | e.destination | e.bytes) % 16U != 0U
            || e.source > UINT64_MAX - e.bytes || e.destination > UINT64_MAX - e.bytes)
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        tiles += e.bytes / 4096U + (e.bytes % 4096U != 0U);
        if (tiles > INT32_MAX) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        batch.entries[i] = e;
        batch.tile_ends[i] = static_cast<uint32_t>(tiles);
    }
    for (uint32_t i = 0; i < count; ++i) {
        const auto& a = batch.entries[i];
        for (uint32_t j = 0; j < count; ++j) {
            const auto& b = batch.entries[j];
            if (copy_ranges_overlap(a.destination, a.bytes, b.source, b.bytes)
                || (j < i && copy_ranges_overlap(a.destination, a.bytes, b.destination, b.bytes)))
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
    }
    auto selected = augur_validate_stream_device(ordinal, stream);
    if (selected.status != AUGUR_CUDA_STATUS_OK) return selected;
    batch.tiles = static_cast<uint32_t>(tiles);
    augur_independent_copy_4k_kernel_v1<<<batch.tiles, 256, 0, stream->stream>>>(batch);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_d2d_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    augur_cuda_device_ptr_t dst,
    size_t dst_offset_bytes,
    size_t dst_len_bytes,
    augur_cuda_device_ptr_t src,
    size_t src_offset_bytes,
    size_t src_len_bytes,
    size_t copy_len_bytes
) {
    if (dst == 0 || src == 0 || copy_len_bytes == 0
        || dst_offset_bytes > dst_len_bytes || src_offset_bytes > src_len_bytes
        || copy_len_bytes > dst_len_bytes - dst_offset_bytes
        || copy_len_bytes > src_len_bytes - src_offset_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_validate_stream_device(ordinal, stream);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    unsigned char* destination = reinterpret_cast<unsigned char*>(static_cast<uintptr_t>(dst));
    const unsigned char* source = reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(src));
    const cudaError_t error = cudaMemcpyAsync(
        destination + dst_offset_bytes,
        source + src_offset_bytes,
        copy_len_bytes,
        cudaMemcpyDeviceToDevice,
        stream->stream
    );
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

// Every column has a single thread owner. Reading a whole register tile before
// writing it, then walking rows forward, preserves memmove dependencies even
// for a one-row shift. uint32_t preserves all bits (including NaN payloads).
template<unsigned TILE>
__global__ void augur_shift_rows_bits_kernel_v1(uint32_t* data, size_t words, size_t retained, size_t shift) {
    size_t col=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(col>=words)return;
    for(size_t row=0;row<retained;row+=TILE){
        uint32_t values[TILE];
        #pragma unroll
        for(unsigned j=0;j<TILE;++j) if(row+j<retained) values[j]=data[(row+j+shift)*words+col];
        #pragma unroll
        for(unsigned j=0;j<TILE;++j) if(row+j<retained) data[(row+j)*words+col]=values[j];
    }
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_shift_rows_u32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_device_ptr_t data,
    size_t len_bytes, size_t row_bytes, size_t retained_rows, size_t shift_rows
) {
    if (data == 0 || data % 4 != 0 || row_bytes == 0 || row_bytes % 4 != 0
        || row_bytes / 4 > UINT32_MAX || len_bytes == 0 || len_bytes % row_bytes != 0
        || data > UINT64_MAX - len_bytes || shift_rows > len_bytes / row_bytes
        || retained_rows > len_bytes / row_bytes - shift_rows) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_validate_stream_device(ordinal, stream);
    if (selected.status != AUGUR_CUDA_STATUS_OK) return selected;
    if (shift_rows == 0 || retained_rows == 0) return augur_ok();
    const size_t words = row_bytes / 4;
    augur_shift_rows_bits_kernel_v1<16><<<static_cast<unsigned>((words + 255) / 256), 256, 0, stream->stream>>>(
        reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(data)), words, retained_rows, shift_rows);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_memcpy_d2d_2d_submit_v1(
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
) {
    if (dst == 0 || src == 0 || width_bytes == 0 || height == 0
        || dst_pitch_bytes < width_bytes || src_pitch_bytes < width_bytes
        || dst_offset_bytes > dst_len_bytes || src_offset_bytes > src_len_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const size_t row_count_minus_one = height - 1;
    if ((row_count_minus_one != 0
            && (dst_pitch_bytes > (SIZE_MAX - width_bytes) / row_count_minus_one
                || src_pitch_bytes > (SIZE_MAX - width_bytes) / row_count_minus_one))) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const size_t dst_footprint = row_count_minus_one * dst_pitch_bytes + width_bytes;
    const size_t src_footprint = row_count_minus_one * src_pitch_bytes + width_bytes;
    if (dst_footprint > dst_len_bytes - dst_offset_bytes
        || src_footprint > src_len_bytes - src_offset_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_validate_stream_device(ordinal, stream);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    unsigned char* destination = reinterpret_cast<unsigned char*>(static_cast<uintptr_t>(dst));
    const unsigned char* source = reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(src));
    const cudaError_t error = cudaMemcpy2DAsync(
        destination + dst_offset_bytes,
        dst_pitch_bytes,
        source + src_offset_bytes,
        src_pitch_bytes,
        width_bytes,
        height,
        cudaMemcpyDeviceToDevice,
        stream->stream
    );
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

// ── Q4K decode validation and launch ──────────────────────────────────────

static constexpr uint32_t AUGUR_Q4K_ARGS_VERSION = 1;
static constexpr uint32_t AUGUR_Q4K_BLOCK_LEN = 256;
static constexpr uint32_t AUGUR_Q4K_BLOCK_BYTES = 160;
static constexpr uint32_t AUGUR_Q4K_BLOCK_ALIGNMENT = 32;

static augur_cuda_result_v1 augur_validate_q4k_args(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q4k_f32_decode_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != AUGUR_Q4K_ARGS_VERSION
        || args->struct_size != sizeof(augur_cuda_q4k_f32_decode_args_v1)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (args->weight == 0 || args->input == 0 || args->output == 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->rows == 0 || args->cols == 0 || args->blocks_per_row == 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t expected_cols =
        static_cast<uint64_t>(args->blocks_per_row)
        * static_cast<uint64_t>(AUGUR_Q4K_BLOCK_LEN);
    if (expected_cols != args->cols) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->weight_block_stride_bytes != AUGUR_Q4K_BLOCK_BYTES) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t block_count =
        static_cast<uint64_t>(args->rows)
        * static_cast<uint64_t>(args->blocks_per_row);
    const uint64_t expected_weight_bytes =
        block_count * static_cast<uint64_t>(AUGUR_Q4K_BLOCK_BYTES);
    const uint64_t expected_input_bytes =
        static_cast<uint64_t>(args->cols) * sizeof(float);
    const uint64_t expected_output_bytes =
        static_cast<uint64_t>(args->rows) * sizeof(float);
    if (args->weight_len_bytes != expected_weight_bytes
        || args->input_len_bytes < expected_input_bytes
        || args->output_len_bytes < expected_output_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if ((args->weight % AUGUR_Q4K_BLOCK_ALIGNMENT) != 0
        || (args->input % alignof(float)) != 0
        || (args->output % alignof(float)) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_set_device(ordinal);
}

__global__ void augur_q4k_f32_decode_kernel_v1(
    const unsigned char* weight,
    const float* input,
    float* output,
    uint32_t rows,
    uint32_t cols,
    uint32_t blocks_per_row
);

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q4k_f32_decode_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q4k_f32_decode_args_v1* args
) {
    augur_cuda_result_v1 validation =
        augur_validate_q4k_args(ordinal, stream, args);
    if (validation.status != AUGUR_CUDA_STATUS_OK) {
        return validation;
    }
    const unsigned char* weight =
        reinterpret_cast<const unsigned char*>(
            static_cast<uintptr_t>(args->weight));
    const float* input =
        reinterpret_cast<const float*>(
            static_cast<uintptr_t>(args->input));
    float* output =
        reinterpret_cast<float*>(
            static_cast<uintptr_t>(args->output));
    constexpr uint32_t threads = 128;
    const uint32_t blocks =
        (args->rows + threads - 1) / threads;
    augur_q4k_f32_decode_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        weight, input, output,
        args->rows, args->cols, args->blocks_per_row);
    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
    }
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q4k_f32_decode_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q4k_f32_decode_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_q4k_f32_decode_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) {
        return submitted;
    }
    return augur_cuda_stream_synchronize(stream);
}

template <typename Args>
static augur_cuda_result_v1 augur_validate_qk_decode_args(
    int32_t ordinal, augur_cuda_stream_t stream, const Args* args,
    uint32_t version, uint32_t block_bytes
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != version || args->struct_size != sizeof(Args)
        || args->weight == 0 || args->input == 0 || args->output == 0
        || args->rows == 0 || args->cols == 0 || args->blocks_per_row == 0
        || args->weight_block_stride_bytes != block_bytes
        || static_cast<uint64_t>(args->blocks_per_row) * 256 != args->cols) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t weight_bytes = static_cast<uint64_t>(args->rows)
        * args->blocks_per_row * block_bytes;
    if (args->weight_len_bytes != weight_bytes
        || args->input_len_bytes < static_cast<uint64_t>(args->cols) * sizeof(float)
        || args->output_len_bytes < static_cast<uint64_t>(args->rows) * sizeof(float)
        || args->weight % 32 != 0 || args->input % alignof(float) != 0
        || args->output % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_set_device(ordinal);
}

#if defined(AUGUR_CUDA_P4_EXPERT_MICROSLICE_GPU_REPACK)
template <typename Args>
static augur_cuda_result_v1 augur_validate_p4_compact_to_hot_args(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const Args* args,
    uint32_t version,
    uint32_t compact_block_bytes,
    uint32_t hot_block_bytes
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != version || args->struct_size != sizeof(Args)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (args->compact == 0 || args->hot == 0
        || args->rows == 0 || args->cols == 0 || args->cols % 256 != 0
        || args->compact_block_stride_bytes != compact_block_bytes
        || args->hot_block_stride_bytes != hot_block_bytes
        || args->reserved0 != 0 || args->reserved1 != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t blocks_per_row = args->cols / 256;
    const uint64_t block_count = static_cast<uint64_t>(args->rows) * blocks_per_row;
    if (block_count > UINT32_C(2147483647)
        || block_count > UINT64_MAX / compact_block_bytes
        || block_count > UINT64_MAX / hot_block_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t compact_bytes = block_count * compact_block_bytes;
    const uint64_t hot_bytes = block_count * hot_block_bytes;
    if (args->compact_len_bytes != compact_bytes || args->hot_len_bytes != hot_bytes
        || args->compact % 32 != 0 || args->hot % 32 != 0
        || args->compact > UINT64_MAX - compact_bytes
        || args->hot > UINT64_MAX - hot_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t compact_end = args->compact + compact_bytes;
    const uint64_t hot_end = args->hot + hot_bytes;
    if (args->compact < hot_end && args->hot < compact_end) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_set_device(ordinal);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q4k_compact_to_hot_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q4k_compact_to_hot_args_v1* args
) {
    const augur_cuda_result_v1 valid = augur_validate_p4_compact_to_hot_args(
        ordinal, stream, args,
        AUGUR_CUDA_Q4K_COMPACT_TO_HOT_ARGS_VERSION_V1,
        AUGUR_CUDA_Q4K_COMPACT_BLOCK_BYTES_V1,
        AUGUR_CUDA_Q4K_HOT_BLOCK_BYTES_V1);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    const cudaError_t error = augur_p4_q4k_compact_to_hot_launch(
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->compact)),
        reinterpret_cast<unsigned char*>(static_cast<uintptr_t>(args->hot)),
        args->rows, args->cols / 256, stream->stream);
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q5k_compact_to_hot_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q5k_compact_to_hot_args_v1* args
) {
    const augur_cuda_result_v1 valid = augur_validate_p4_compact_to_hot_args(
        ordinal, stream, args,
        AUGUR_CUDA_Q5K_COMPACT_TO_HOT_ARGS_VERSION_V1,
        AUGUR_CUDA_Q5K_COMPACT_BLOCK_BYTES_V1,
        AUGUR_CUDA_Q5K_HOT_BLOCK_BYTES_V1);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    const cudaError_t error = augur_p4_q5k_compact_to_hot_launch(
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->compact)),
        reinterpret_cast<unsigned char*>(static_cast<uintptr_t>(args->hot)),
        args->rows, args->cols / 256, stream->stream);
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}
#endif

__global__ void augur_q5k_f32_decode_kernel_v1(
    const unsigned char*, const float*, float*, uint32_t, uint32_t);
__global__ void augur_q5k_f32_decode_grid32_kernel_v2(
    const unsigned char*, const float*, float*, uint32_t, uint32_t);
#if defined(AUGUR_CUDA_M306_K22_Q5_FP32_KAHAN)
__global__ void augur_m306_k22_q5k_f32_decode_bounded_fp32_kahan_kernel_v1(
    const unsigned char*, const float*, float*, uint32_t, uint32_t);
#endif
#if defined(AUGUR_CUDA_M306_K15_Q5_PAIRED_PROJECTION)
__global__ void augur_q5k_f32_paired_projection_kernel_v1(
    const unsigned char*, const unsigned char*, const float*, float*, float*, uint32_t, uint32_t);
__global__ void augur_batch01_b4_q5k_f32_paired_projection_kernel_v1(
    const unsigned char*, const unsigned char*,
    const float*, const float*, const float*, const float*,
    float*, float*, float*, float*, float*, float*, float*, float*,
    uint32_t, uint32_t);
#if defined(AUGUR_CUDA_BATCH01_B8_Q5_ROW_TILE)
__global__ void augur_batch01_b8_q5k_f32_paired_projection_row_tile_kernel_v1(
    augur_cuda_batch01_b8_q5k_row_tile_args_v1);
__global__ void augur_batch01_b8_q5k_f32_paired_projection_row_tile_kernel_v2(
    augur_cuda_batch01_b8_q5k_row_tile_args_v1);
#endif
#endif
__global__ void augur_q6k_f32_decode_kernel_v1(
    const unsigned char*, const float*, float*, uint32_t, uint32_t);
#if defined(AUGUR_CUDA_DSPARK_FAST_Q6_ROW_REUSE)
__global__ void augur_dspark_fast_q6k_row_reuse_kernel_v1(
    const unsigned char*, const float*, float*, int, int, int,
    uint32_t, uint32_t, uint32_t);
#endif
#if defined(AUGUR_CUDA_BATCH01_B4_Q6_ROW_REUSE)
__global__ void augur_batch01_b4_q6k_bounded_fp64_kernel_v1(
    const unsigned char*,
    const float*, const float*, const float*, const float*,
    float*, float*, float*, float*,
    int, int);
#if defined(AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_ROW_REUSE)
__global__ void augur_batch01_b4_q6_block_diagonal_bounded_fp64_kernel_v1(
    const unsigned char*,
    const float*, const float*, const float*, const float*,
    float*, float*, float*, float*,
    int, int, int);
__global__ void q6_block_diagonal_bounded_fp64(
    const unsigned char*, const float*, float*, int, int, int);
#endif
#endif

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q5k_f32_decode_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q5k_f32_decode_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_qk_decode_args(
        ordinal, stream, args, AUGUR_CUDA_Q5K_F32_DECODE_ARGS_VERSION_V1, AUGUR_CUDA_Q5K_BLOCK_BYTES_V1);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    const uint32_t blocks = (args->rows + 127) / 128;
    augur_q5k_f32_decode_kernel_v1<<<blocks, 128, 0, stream->stream>>>(
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->rows, args->blocks_per_row);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q5k_f32_decode_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q5k_f32_decode_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_q5k_f32_decode_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q5k_f32_decode_grid32_submit_v2(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q5k_f32_decode_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_qk_decode_args(
        ordinal, stream, args, AUGUR_CUDA_Q5K_F32_DECODE_ARGS_VERSION_V1, AUGUR_CUDA_Q5K_BLOCK_BYTES_V1);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    const uint32_t blocks = (args->rows + 31) / 32;
    augur_q5k_f32_decode_grid32_kernel_v2<<<blocks, 32, 0, stream->stream>>>(
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->rows, args->blocks_per_row);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q5k_f32_decode_grid32_v2(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q5k_f32_decode_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_q5k_f32_decode_grid32_submit_v2(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

#if defined(AUGUR_CUDA_M306_K22_Q5_FP32_KAHAN)
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k22_q5k_f32_decode_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q5k_f32_decode_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_qk_decode_args(
        ordinal, stream, args, AUGUR_CUDA_Q5K_F32_DECODE_ARGS_VERSION_V1, AUGUR_CUDA_Q5K_BLOCK_BYTES_V1);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    if (args->rows != AUGUR_CUDA_M306_K22_Q5_ROWS_V1
        || args->cols != AUGUR_CUDA_M306_K22_Q5_COLS_V1
        || args->blocks_per_row != AUGUR_CUDA_M306_K22_Q5_COLS_V1 / AUGUR_CUDA_Q5K_BLOCK_DECODED_LEN_V1) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    constexpr uint32_t threads = 128;
    constexpr uint32_t warps = threads / 32;
    const uint32_t blocks = (args->rows + warps - 1) / warps;
    augur_m306_k22_q5k_f32_decode_bounded_fp32_kahan_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->rows, args->blocks_per_row);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k22_q5k_f32_decode_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q5k_f32_decode_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_m306_k22_q5k_f32_decode_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}
#endif

#if defined(AUGUR_CUDA_M306_K15_Q5_PAIRED_PROJECTION)
static bool augur_q5k_paired_ranges_overlap(
    uint64_t left,
    uint64_t left_len,
    uint64_t right,
    uint64_t right_len,
    bool* overlap
) {
    if (left > UINT64_MAX - left_len || right > UINT64_MAX - right_len) return false;
    *overlap = left < right + right_len && right < left + left_len;
    return true;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q5k_f32_paired_projection_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q5k_f32_paired_projection_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (args->weight_w1 == 0 || args->weight_w3 == 0 || args->input == 0
        || args->output_w1 == 0 || args->output_w3 == 0
        || args->output_w1 == args->output_w3
        || args->rows != AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_ROWS_V1
        || args->cols != AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_COLS_V1
        || args->blocks_per_row != AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_BLOCKS_PER_ROW_V1
        || args->weight_block_stride_bytes != AUGUR_CUDA_Q5K_BLOCK_BYTES_V1) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t weight_bytes = static_cast<uint64_t>(args->rows)
        * args->blocks_per_row * AUGUR_CUDA_Q5K_BLOCK_BYTES_V1;
    const uint64_t input_bytes = static_cast<uint64_t>(args->cols) * sizeof(float);
    const uint64_t output_bytes = static_cast<uint64_t>(args->rows) * sizeof(float);
    if (args->weight_w1_len_bytes != weight_bytes || args->weight_w3_len_bytes != weight_bytes
        || args->input_len_bytes != input_bytes
        || args->output_w1_len_bytes != output_bytes || args->output_w3_len_bytes != output_bytes
        || args->weight_w1 % AUGUR_CUDA_Q5K_BLOCK_ALIGNMENT_V1 != 0
        || args->weight_w3 % AUGUR_CUDA_Q5K_BLOCK_ALIGNMENT_V1 != 0
        || args->input % alignof(float) != 0
        || args->output_w1 % alignof(float) != 0 || args->output_w3 % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t ranges[][2] = {
        {args->weight_w1, args->weight_w1_len_bytes},
        {args->weight_w3, args->weight_w3_len_bytes},
        {args->input, args->input_len_bytes},
        {args->output_w1, args->output_w1_len_bytes},
        {args->output_w3, args->output_w3_len_bytes},
    };
    const uint32_t forbidden_pairs[][2] = {
        {0, 1},
        {0, 2},
        {0, 3}, {0, 4},
        {1, 2},
        {1, 3}, {1, 4},
        {2, 3}, {2, 4},
        {3, 4},
    };
    for (const auto& pair : forbidden_pairs) {
        bool overlap = false;
        if (!augur_q5k_paired_ranges_overlap(
                ranges[pair[0]][0], ranges[pair[0]][1],
                ranges[pair[1]][0], ranges[pair[1]][1], &overlap)
            || overlap) {
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
    }
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    augur_q5k_f32_paired_projection_kernel_v1<<<
        AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_GRID_BLOCKS_V1, 128, 0, stream->stream>>>(
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->weight_w1)),
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->weight_w3)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_w1)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_w3)),
        args->rows,
        args->blocks_per_row);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q5k_f32_paired_projection_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q5k_f32_paired_projection_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_q5k_f32_paired_projection_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q5k_f32_paired_projection_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q5k_f32_paired_projection_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version
            != AUGUR_CUDA_BATCH01_B4_Q5K_F32_PAIRED_PROJECTION_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (args->batch != AUGUR_CUDA_BATCH01_B4_Q5K_F32_PAIRED_PROJECTION_BATCH_V1
        || args->rows != AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_ROWS_V1
        || args->cols != AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_COLS_V1
        || args->blocks_per_row != AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_BLOCKS_PER_ROW_V1
        || args->weight_block_stride_bytes != AUGUR_CUDA_Q5K_BLOCK_BYTES_V1) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t weight_bytes = static_cast<uint64_t>(args->rows)
        * args->blocks_per_row * AUGUR_CUDA_Q5K_BLOCK_BYTES_V1;
    const uint64_t input_bytes = static_cast<uint64_t>(args->cols) * sizeof(float);
    const uint64_t output_bytes = static_cast<uint64_t>(args->rows) * sizeof(float);
    if (args->weight_w1 == 0 || args->weight_w3 == 0
        || args->weight_w1_len_bytes != weight_bytes
        || args->weight_w3_len_bytes != weight_bytes
        || args->input_len_bytes != input_bytes
        || args->output_len_bytes != output_bytes
        || args->weight_w1 % AUGUR_CUDA_Q5K_BLOCK_ALIGNMENT_V1 != 0
        || args->weight_w3 % AUGUR_CUDA_Q5K_BLOCK_ALIGNMENT_V1 != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    uint64_t ranges[14][2] = {
        {args->weight_w1, args->weight_w1_len_bytes},
        {args->weight_w3, args->weight_w3_len_bytes},
        {args->input[0], args->input_len_bytes},
        {args->input[1], args->input_len_bytes},
        {args->input[2], args->input_len_bytes},
        {args->input[3], args->input_len_bytes},
        {args->output_w1[0], args->output_len_bytes},
        {args->output_w1[1], args->output_len_bytes},
        {args->output_w1[2], args->output_len_bytes},
        {args->output_w1[3], args->output_len_bytes},
        {args->output_w3[0], args->output_len_bytes},
        {args->output_w3[1], args->output_len_bytes},
        {args->output_w3[2], args->output_len_bytes},
        {args->output_w3[3], args->output_len_bytes},
    };
    for (uint32_t index = 0; index < 14; ++index) {
        if (ranges[index][0] == 0 || ranges[index][0] % alignof(float) != 0) {
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t other = index + 1; other < 14; ++other) {
            bool overlap = false;
            if (!augur_q5k_paired_ranges_overlap(
                    ranges[index][0], ranges[index][1],
                    ranges[other][0], ranges[other][1], &overlap)
                || overlap) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }

    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    augur_batch01_b4_q5k_f32_paired_projection_kernel_v1<<<
        AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_GRID_BLOCKS_V1,
        128,
        0,
        stream->stream>>>(
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->weight_w1)),
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->weight_w3)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input[0])),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input[1])),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input[2])),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input[3])),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_w1[0])),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_w1[1])),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_w1[2])),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_w1[3])),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_w3[0])),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_w3[1])),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_w3[2])),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_w3[3])),
        args->rows,
        args->blocks_per_row);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q5k_f32_paired_projection_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q5k_f32_paired_projection_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_batch01_b4_q5k_f32_paired_projection_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

#if defined(AUGUR_CUDA_BATCH01_B8_Q5_ROW_TILE)
static bool augur_batch01_b8_q5_device_range_owned(
    uint64_t address,
    uint64_t extent,
    int32_t ordinal
) {
    if (address == 0 || extent == 0 || address > UINT64_MAX - extent) return false;

    const void* pointer =
        reinterpret_cast<const void*>(static_cast<uintptr_t>(address));
    cudaPointerAttributes attributes{};
    const cudaError_t attribute_error = cudaPointerGetAttributes(&attributes, pointer);
    if (attribute_error != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != ordinal) {
        return false;
    }

    CUdeviceptr allocation_base_pointer = 0;
    size_t allocation_extent = 0;
    const CUresult range_error = cuMemGetAddressRange(
        &allocation_base_pointer,
        &allocation_extent,
        static_cast<CUdeviceptr>(address));
    if (range_error != CUDA_SUCCESS) return false;
    const uint64_t allocation_base = static_cast<uint64_t>(allocation_base_pointer);
    if (allocation_base > address
        || allocation_extent > UINT64_MAX - allocation_base) {
        return false;
    }
    return address + extent <= allocation_base + allocation_extent;
}

static augur_cuda_result_v1 augur_validate_batch01_b8_q5k_row_tile(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args,
    bool validate_device_ranges
) {
    if (stream == nullptr || args == nullptr || ordinal < 0
        || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != AUGUR_CUDA_BATCH01_B8_Q5K_ROW_TILE_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (args->batch != AUGUR_CUDA_BATCH01_B8_Q5K_ROW_TILE_BATCH_V1
        || args->rows != AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_ROWS_V1
        || args->cols != AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_COLS_V1
        || args->blocks_per_row
            != AUGUR_CUDA_Q5K_F32_PAIRED_PROJECTION_BLOCKS_PER_ROW_V1
        || args->weight_block_stride_bytes != AUGUR_CUDA_Q5K_BLOCK_BYTES_V1) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    const uint64_t weight_bytes = static_cast<uint64_t>(args->rows)
        * args->blocks_per_row * AUGUR_CUDA_Q5K_BLOCK_BYTES_V1;
    const uint64_t input_bytes = static_cast<uint64_t>(args->cols) * sizeof(float);
    const uint64_t output_bytes = static_cast<uint64_t>(args->rows) * sizeof(float);
    if (args->weight_w1_len_bytes != weight_bytes
        || args->weight_w3_len_bytes != weight_bytes
        || args->input_len_bytes != input_bytes
        || args->output_len_bytes != output_bytes
        || args->weight_w1 % AUGUR_CUDA_Q5K_BLOCK_ALIGNMENT_V1 != 0
        || args->weight_w3 % AUGUR_CUDA_Q5K_BLOCK_ALIGNMENT_V1 != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    uint64_t ranges[26][2] = {
        {args->weight_w1, args->weight_w1_len_bytes},
        {args->weight_w3, args->weight_w3_len_bytes},
    };
    for (uint32_t row = 0; row < 8; ++row) {
        ranges[2 + row][0] = args->input[row];
        ranges[2 + row][1] = args->input_len_bytes;
        ranges[10 + row][0] = args->output_w1[row];
        ranges[10 + row][1] = args->output_len_bytes;
        ranges[18 + row][0] = args->output_w3[row];
        ranges[18 + row][1] = args->output_len_bytes;
    }
    for (uint32_t index = 0; index < 26; ++index) {
        if (ranges[index][0] == 0
            || (index > 1 && ranges[index][0] % alignof(float) != 0)) {
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t other = index + 1; other < 26; ++other) {
            bool overlap = false;
            if (!augur_q5k_paired_ranges_overlap(
                    ranges[index][0], ranges[index][1],
                    ranges[other][0], ranges[other][1], &overlap)
                || overlap) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }

    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    if (validate_device_ranges) {
        cudaDeviceProp properties{};
        const cudaError_t property_error = cudaGetDeviceProperties(&properties, ordinal);
        if (property_error != cudaSuccess) return augur_runtime_error(property_error);
        if (properties.major != 7 || properties.minor != 0) {
            return augur_error(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE, cudaSuccess);
        }
        for (uint32_t index = 0; index < 26; ++index) {
            if (!augur_batch01_b8_q5_device_range_owned(
                    ranges[index][0], ranges[index][1], ordinal)) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    return augur_ok();
}

static augur_cuda_result_v1 augur_launch_batch01_b8_q5k_row_tile(
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args
) {
    augur_batch01_b8_q5k_f32_paired_projection_row_tile_kernel_v1<<<
        AUGUR_CUDA_BATCH01_B8_Q5K_ROW_TILE_GRID_BLOCKS_V1,
        AUGUR_CUDA_BATCH01_B8_Q5K_ROW_TILE_THREADS_V1,
        0,
        stream->stream>>>(*args);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

static augur_cuda_result_v1 augur_launch_batch01_b8_q5k_row_tile_v2(
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args
) {
    augur_batch01_b8_q5k_f32_paired_projection_row_tile_kernel_v2<<<
        AUGUR_CUDA_BATCH01_B8_Q5K_ROW_TILE_GRID_BLOCKS_V1,
        AUGUR_CUDA_BATCH01_B8_Q5K_ROW_TILE_THREADS_V1,
        0,
        stream->stream>>>(*args);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b8_q5k_row_tile_validate_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args
) {
    return augur_validate_batch01_b8_q5k_row_tile(ordinal, stream, args, true);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b8_q5k_row_tile_submit_prevalidated_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args
) {
    augur_cuda_result_v1 valid =
        augur_validate_batch01_b8_q5k_row_tile(ordinal, stream, args, false);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    return augur_launch_batch01_b8_q5k_row_tile(stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b8_q5k_row_tile_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args
) {
    augur_cuda_result_v1 valid =
        augur_validate_batch01_b8_q5k_row_tile(ordinal, stream, args, true);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    return augur_launch_batch01_b8_q5k_row_tile(stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b8_q5k_row_tile_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_batch01_b8_q5k_row_tile_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b8_q5k_row_tile_submit_prevalidated_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args
) {
    augur_cuda_result_v1 valid =
        augur_validate_batch01_b8_q5k_row_tile(ordinal, stream, args, false);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    return augur_launch_batch01_b8_q5k_row_tile_v2(stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b8_q5k_row_tile_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args
) {
    augur_cuda_result_v1 valid =
        augur_validate_batch01_b8_q5k_row_tile(ordinal, stream, args, true);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    return augur_launch_batch01_b8_q5k_row_tile_v2(stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b8_q5k_row_tile_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b8_q5k_row_tile_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_batch01_b8_q5k_row_tile_submit_v2(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}
#endif
#endif

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q6k_f32_decode_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q6k_f32_decode_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_qk_decode_args(
        ordinal, stream, args, AUGUR_CUDA_Q6K_F32_DECODE_ARGS_VERSION_V1, AUGUR_CUDA_Q6K_BLOCK_BYTES_V1);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    const uint32_t blocks = (args->rows + 127) / 128;
    augur_q6k_f32_decode_kernel_v1<<<blocks, 128, 0, stream->stream>>>(
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->rows, args->blocks_per_row);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q6k_f32_decode_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q6k_f32_decode_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_q6k_f32_decode_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

#if defined(AUGUR_CUDA_DSPARK_FAST_Q6_ROW_REUSE)
static bool augur_dspark_fast_q6_ranges_overlap(
    uint64_t left,
    uint64_t left_len,
    uint64_t right,
    uint64_t right_len,
    bool* overlap
) {
    if (left > UINT64_MAX - left_len || right > UINT64_MAX - right_len) return false;
    *overlap = left < right + right_len && right < left + left_len;
    return true;
}

static bool augur_dspark_fast_q6_device_range_owned(
    uint64_t address,
    uint64_t extent,
    int32_t ordinal
) {
    if (address == 0 || extent == 0 || address > UINT64_MAX - extent) return false;
    const void* pointer =
        reinterpret_cast<const void*>(static_cast<uintptr_t>(address));
    cudaPointerAttributes attributes{};
    const cudaError_t attribute_error = cudaPointerGetAttributes(&attributes, pointer);
    if (attribute_error != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != ordinal) {
        return false;
    }
    CUdeviceptr allocation_base_pointer = 0;
    size_t allocation_extent = 0;
    const CUresult range_error = cuMemGetAddressRange(
        &allocation_base_pointer, &allocation_extent,
        static_cast<CUdeviceptr>(address));
    if (range_error != CUDA_SUCCESS) return false;
    const uint64_t allocation_base = static_cast<uint64_t>(allocation_base_pointer);
    if (allocation_base > address
        || allocation_extent > UINT64_MAX - allocation_base) {
        return false;
    }
    return address + extent <= allocation_base + allocation_extent;
}

static augur_cuda_result_v1 augur_validate_dspark_fast_q6_row_reuse(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_dspark_fast_q6k_row_reuse_args_v1* args,
    bool validate_device_ranges
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != AUGUR_CUDA_DSPARK_FAST_Q6K_ROW_REUSE_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    for (uint32_t value : args->reserved) {
        if (value != 0) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->active_rows == 0
        || args->active_rows > AUGUR_CUDA_DSPARK_FAST_Q6K_ROW_REUSE_MAX_ACTIVE_ROWS_V1
        || args->rows == 0 || args->rows > UINT32_C(2147483647)
        || args->cols == 0 || args->cols > UINT32_C(2147483647)
        || args->cols % AUGUR_CUDA_Q6K_BLOCK_DECODED_LEN_V1 != 0
        || args->blocks_per_row
            != args->cols / AUGUR_CUDA_Q6K_BLOCK_DECODED_LEN_V1
        || args->weight_block_stride_bytes != AUGUR_CUDA_Q6K_BLOCK_BYTES_V1
        || args->input_row_stride_elements < args->cols
        || args->output_row_stride_elements < args->rows) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    const uint64_t weight_rows = static_cast<uint64_t>(args->rows);
    const uint64_t blocks = static_cast<uint64_t>(args->blocks_per_row);
    const uint64_t weight_bytes = weight_rows * blocks
        * static_cast<uint64_t>(AUGUR_CUDA_Q6K_BLOCK_BYTES_V1);
    const uint64_t input_last_row = static_cast<uint64_t>(args->active_rows - 1)
        * static_cast<uint64_t>(args->input_row_stride_elements);
    const uint64_t output_last_row = static_cast<uint64_t>(args->active_rows - 1)
        * static_cast<uint64_t>(args->output_row_stride_elements);
    if (input_last_row > UINT64_MAX - args->cols
        || output_last_row > UINT64_MAX - args->rows) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t input_elements = input_last_row + args->cols;
    const uint64_t output_elements = output_last_row + args->rows;
    if (input_elements > UINT64_MAX / sizeof(float)
        || output_elements > UINT64_MAX / sizeof(float)) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t input_bytes = input_elements * sizeof(float);
    const uint64_t output_bytes = output_elements * sizeof(float);
    if (args->weight_len_bytes != weight_bytes
        || args->input_len_bytes < input_bytes
        || args->output_len_bytes < output_bytes
        || args->weight % AUGUR_CUDA_Q6K_BLOCK_ALIGNMENT_V1 != 0
        || args->input % alignof(float) != 0
        || args->output % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    const uint64_t ranges[3][2] = {
        {args->weight, args->weight_len_bytes},
        {args->input, args->input_len_bytes},
        {args->output, args->output_len_bytes},
    };
    for (uint32_t index = 0; index < 3; ++index) {
        if (ranges[index][0] == 0) {
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t other = index + 1; other < 3; ++other) {
            bool overlap = false;
            if (!augur_dspark_fast_q6_ranges_overlap(
                    ranges[index][0], ranges[index][1],
                    ranges[other][0], ranges[other][1], &overlap)
                || overlap) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }

    if (validate_device_ranges) {
        augur_cuda_result_v1 device = augur_set_device(ordinal);
        if (device.status != AUGUR_CUDA_STATUS_OK) return device;
        cudaDeviceProp properties{};
        const cudaError_t property_error = cudaGetDeviceProperties(&properties, ordinal);
        if (property_error != cudaSuccess) return augur_runtime_error(property_error);
        if (properties.major != 7 || properties.minor != 0) {
            return augur_error(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE, cudaSuccess);
        }
        for (const auto& range : ranges) {
            if (!augur_dspark_fast_q6_device_range_owned(range[0], range[1], ordinal)) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    return augur_ok();
}

static augur_cuda_result_v1 augur_launch_dspark_fast_q6_row_reuse(
    augur_cuda_stream_t stream,
    const augur_cuda_dspark_fast_q6k_row_reuse_args_v1* args
) {
    constexpr uint32_t threads = 128;
    constexpr uint32_t warps = threads / 32;
    const uint32_t blocks = (args->rows + warps - 1) / warps;
    augur_dspark_fast_q6k_row_reuse_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        static_cast<int>(args->active_rows),
        static_cast<int>(args->rows),
        static_cast<int>(args->blocks_per_row),
        args->input_row_stride_elements,
        args->output_row_stride_elements,
        args->weight_block_stride_bytes);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_dspark_fast_q6k_row_reuse_validate_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_dspark_fast_q6k_row_reuse_args_v1* args
) {
    return augur_validate_dspark_fast_q6_row_reuse(ordinal, stream, args, true);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_dspark_fast_q6k_row_reuse_submit_prevalidated_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_dspark_fast_q6k_row_reuse_args_v1* args
) {
    augur_cuda_result_v1 valid =
        augur_validate_dspark_fast_q6_row_reuse(ordinal, stream, args, false);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    return augur_launch_dspark_fast_q6_row_reuse(stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_dspark_fast_q6k_row_reuse_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_dspark_fast_q6k_row_reuse_args_v1* args
) {
    augur_cuda_result_v1 valid =
        augur_validate_dspark_fast_q6_row_reuse(ordinal, stream, args, true);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    return augur_launch_dspark_fast_q6_row_reuse(stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_dspark_fast_q6k_row_reuse_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_dspark_fast_q6k_row_reuse_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_dspark_fast_q6k_row_reuse_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}
#endif

#if defined(AUGUR_CUDA_BATCH01_B4_Q6_ROW_REUSE)
static bool augur_batch01_b4_q6_ranges_overlap(
    uint64_t left,
    uint64_t left_len,
    uint64_t right,
    uint64_t right_len,
    bool* overlap
) {
    if (left > UINT64_MAX - left_len || right > UINT64_MAX - right_len) return false;
    *overlap = left < right + right_len && right < left + left_len;
    return true;
}

static bool augur_batch01_b4_q6_device_range_owned(
    uint64_t address,
    uint64_t extent,
    int32_t ordinal
) {
    if (address == 0 || extent == 0 || address > UINT64_MAX - extent) return false;

    const void* pointer =
        reinterpret_cast<const void*>(static_cast<uintptr_t>(address));
    cudaPointerAttributes attributes{};
    const cudaError_t attribute_error = cudaPointerGetAttributes(&attributes, pointer);
    if (attribute_error != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != ordinal) {
        return false;
    }

    CUdeviceptr allocation_base_pointer = 0;
    size_t allocation_extent = 0;
    const CUresult range_error = cuMemGetAddressRange(
        &allocation_base_pointer, &allocation_extent,
        static_cast<CUdeviceptr>(address));
    if (range_error != CUDA_SUCCESS) return false;
    const uint64_t allocation_base = static_cast<uint64_t>(allocation_base_pointer);
    if (allocation_base > address
        || allocation_extent > UINT64_MAX - allocation_base) {
        return false;
    }
    return address + extent <= allocation_base + allocation_extent;
}

static augur_cuda_result_v1 augur_validate_batch01_b4_q6k_bounded_fp64(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6k_bounded_fp64_args_v1* args,
    bool validate_device_ranges
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != AUGUR_CUDA_BATCH01_B4_Q6K_BOUNDED_FP64_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (args->batch != AUGUR_CUDA_BATCH01_B4_Q6K_BOUNDED_FP64_BATCH_V1
        || args->rows != AUGUR_CUDA_BATCH01_B4_Q6K_BOUNDED_FP64_ROWS_V1
        || args->cols != AUGUR_CUDA_BATCH01_B4_Q6K_BOUNDED_FP64_COLS_V1
        || args->blocks_per_row
            != AUGUR_CUDA_BATCH01_B4_Q6K_BOUNDED_FP64_BLOCKS_PER_ROW_V1
        || args->weight_block_stride_bytes != AUGUR_CUDA_Q6K_BLOCK_BYTES_V1) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    const uint64_t weight_bytes = static_cast<uint64_t>(args->rows)
        * args->blocks_per_row * AUGUR_CUDA_Q6K_BLOCK_BYTES_V1;
    const uint64_t input_bytes = static_cast<uint64_t>(args->cols) * sizeof(float);
    const uint64_t output_bytes = static_cast<uint64_t>(args->rows) * sizeof(float);
    if (args->weight_len_bytes != weight_bytes
        || args->input_len_bytes != input_bytes
        || args->output_len_bytes != output_bytes
        || args->weight % AUGUR_CUDA_Q6K_BLOCK_ALIGNMENT_V1 != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    uint64_t ranges[9][2] = {
        {args->weight, args->weight_len_bytes},
        {args->input[0], args->input_len_bytes},
        {args->input[1], args->input_len_bytes},
        {args->input[2], args->input_len_bytes},
        {args->input[3], args->input_len_bytes},
        {args->output[0], args->output_len_bytes},
        {args->output[1], args->output_len_bytes},
        {args->output[2], args->output_len_bytes},
        {args->output[3], args->output_len_bytes},
    };
    for (uint32_t index = 0; index < 9; ++index) {
        if (ranges[index][0] == 0
            || (index > 0 && ranges[index][0] % alignof(float) != 0)) {
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t other = index + 1; other < 9; ++other) {
            bool overlap = false;
            if (!augur_batch01_b4_q6_ranges_overlap(
                    ranges[index][0], ranges[index][1],
                    ranges[other][0], ranges[other][1], &overlap)
                || overlap) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }

    if (validate_device_ranges) {
        augur_cuda_result_v1 device = augur_set_device(ordinal);
        if (device.status != AUGUR_CUDA_STATUS_OK) return device;
        cudaDeviceProp properties{};
        const cudaError_t property_error = cudaGetDeviceProperties(&properties, ordinal);
        if (property_error != cudaSuccess) return augur_runtime_error(property_error);
        if (properties.major != 7 || properties.minor != 0) {
            return augur_error(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE, cudaSuccess);
        }
        for (uint32_t index = 0; index < 9; ++index) {
            if (!augur_batch01_b4_q6_device_range_owned(
                    ranges[index][0], ranges[index][1], ordinal)) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    return augur_ok();
}

static augur_cuda_result_v1 augur_launch_batch01_b4_q6k_bounded_fp64(
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6k_bounded_fp64_args_v1* args
) {
    constexpr uint32_t threads = 128;
    constexpr uint32_t warps = threads / 32;
    const uint32_t blocks = (args->rows + warps - 1) / warps;
    augur_batch01_b4_q6k_bounded_fp64_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input[0])),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input[1])),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input[2])),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input[3])),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output[0])),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output[1])),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output[2])),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output[3])),
        static_cast<int>(args->rows),
        static_cast<int>(args->blocks_per_row));
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q6k_bounded_fp64_validate_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6k_bounded_fp64_args_v1* args
) {
    return augur_validate_batch01_b4_q6k_bounded_fp64(ordinal, stream, args, true);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q6k_bounded_fp64_submit_prevalidated_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6k_bounded_fp64_args_v1* args
) {
    augur_cuda_result_v1 valid =
        augur_validate_batch01_b4_q6k_bounded_fp64(ordinal, stream, args, false);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;

    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    return augur_launch_batch01_b4_q6k_bounded_fp64(stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q6k_bounded_fp64_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6k_bounded_fp64_args_v1* args
) {
    augur_cuda_result_v1 valid =
        augur_validate_batch01_b4_q6k_bounded_fp64(ordinal, stream, args, true);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    return augur_launch_batch01_b4_q6k_bounded_fp64(stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_batch01_b4_q6k_bounded_fp64_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6k_bounded_fp64_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_batch01_b4_q6k_bounded_fp64_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

#if defined(AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_ROW_REUSE)
static augur_cuda_result_v1 augur_validate_batch01_b4_q6_block_diagonal_control(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1* args,
    bool validate_device_ranges
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version
            != AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_CONTROL_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (args->groups != AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_BOUNDED_FP64_GROUPS_V1
        || args->rows_per_group
            != AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_BOUNDED_FP64_ROWS_PER_GROUP_V1
        || args->cols_per_group
            != AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_BOUNDED_FP64_COLS_PER_GROUP_V1
        || args->blocks_per_output_row
            != AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_BOUNDED_FP64_BLOCKS_PER_OUTPUT_ROW_V1
        || args->weight_block_stride_bytes != AUGUR_CUDA_Q6K_BLOCK_BYTES_V1) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    const uint64_t rows = static_cast<uint64_t>(args->groups) * args->rows_per_group;
    const uint64_t weight_bytes = rows
        * args->blocks_per_output_row * AUGUR_CUDA_Q6K_BLOCK_BYTES_V1;
    const uint64_t input_bytes = static_cast<uint64_t>(args->groups)
        * args->cols_per_group * sizeof(float);
    const uint64_t output_bytes = rows * sizeof(float);
    if (args->weight_len_bytes != weight_bytes
        || args->input_len_bytes != input_bytes
        || args->output_len_bytes != output_bytes
        || args->weight % AUGUR_CUDA_Q6K_BLOCK_ALIGNMENT_V1 != 0
        || args->input % alignof(float) != 0
        || args->output % alignof(float) != 0
        || static_cast<uint64_t>(args->blocks_per_output_row) * 256
            != args->cols_per_group) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    uint64_t ranges[3][2] = {
        {args->weight, args->weight_len_bytes},
        {args->input, args->input_len_bytes},
        {args->output, args->output_len_bytes},
    };
    for (uint32_t index = 0; index < 3; ++index) {
        if (ranges[index][0] == 0) {
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t other = index + 1; other < 3; ++other) {
            bool overlap = false;
            if (!augur_batch01_b4_q6_ranges_overlap(
                    ranges[index][0], ranges[index][1],
                    ranges[other][0], ranges[other][1], &overlap)
                || overlap) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }

    if (validate_device_ranges) {
        augur_cuda_result_v1 device = augur_set_device(ordinal);
        if (device.status != AUGUR_CUDA_STATUS_OK) return device;
        cudaDeviceProp properties{};
        const cudaError_t property_error = cudaGetDeviceProperties(&properties, ordinal);
        if (property_error != cudaSuccess) return augur_runtime_error(property_error);
        if (properties.major != 7 || properties.minor != 0) {
            return augur_error(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE, cudaSuccess);
        }
        for (uint32_t index = 0; index < 3; ++index) {
            if (!augur_batch01_b4_q6_device_range_owned(
                    ranges[index][0], ranges[index][1], ordinal)) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    return augur_ok();
}

static augur_cuda_result_v1 augur_launch_batch01_b4_q6_block_diagonal_control(
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1* args
) {
    constexpr uint32_t threads = 128;
    constexpr uint32_t warps = threads / 32;
    const uint32_t rows = args->groups * args->rows_per_group;
    const uint32_t blocks = (rows + warps - 1) / warps;
    q6_block_diagonal_bounded_fp64<<<blocks, threads, 0, stream->stream>>>(
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        static_cast<int>(args->groups),
        static_cast<int>(args->rows_per_group),
        static_cast<int>(args->blocks_per_output_row));
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q6_block_diagonal_control_validate_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1* args
) {
    return augur_validate_batch01_b4_q6_block_diagonal_control(
        ordinal, stream, args, true);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q6_block_diagonal_control_submit_prevalidated_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6_block_diagonal_control_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_batch01_b4_q6_block_diagonal_control(
        ordinal, stream, args, false);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;

    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    return augur_launch_batch01_b4_q6_block_diagonal_control(stream, args);
}

static augur_cuda_result_v1 augur_validate_batch01_b4_q6_block_diagonal_bounded_fp64(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1* args,
    bool validate_device_ranges
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version
            != AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_BOUNDED_FP64_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (args->batch != AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_BOUNDED_FP64_BATCH_V1
        || args->groups != AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_BOUNDED_FP64_GROUPS_V1
        || args->rows_per_group
            != AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_BOUNDED_FP64_ROWS_PER_GROUP_V1
        || args->cols_per_group
            != AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_BOUNDED_FP64_COLS_PER_GROUP_V1
        || args->blocks_per_output_row
            != AUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_BOUNDED_FP64_BLOCKS_PER_OUTPUT_ROW_V1
        || args->weight_block_stride_bytes != AUGUR_CUDA_Q6K_BLOCK_BYTES_V1) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    const uint64_t rows = static_cast<uint64_t>(args->groups) * args->rows_per_group;
    const uint64_t weight_bytes = rows
        * args->blocks_per_output_row * AUGUR_CUDA_Q6K_BLOCK_BYTES_V1;
    const uint64_t input_bytes = static_cast<uint64_t>(args->groups)
        * args->cols_per_group * sizeof(float);
    const uint64_t output_bytes = rows * sizeof(float);
    if (args->weight_len_bytes != weight_bytes
        || args->input_len_bytes != input_bytes
        || args->output_len_bytes != output_bytes
        || args->weight % AUGUR_CUDA_Q6K_BLOCK_ALIGNMENT_V1 != 0
        || static_cast<uint64_t>(args->blocks_per_output_row) * 256
            != args->cols_per_group) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    uint64_t ranges[9][2] = {
        {args->weight, args->weight_len_bytes},
        {args->input[0], args->input_len_bytes},
        {args->input[1], args->input_len_bytes},
        {args->input[2], args->input_len_bytes},
        {args->input[3], args->input_len_bytes},
        {args->output[0], args->output_len_bytes},
        {args->output[1], args->output_len_bytes},
        {args->output[2], args->output_len_bytes},
        {args->output[3], args->output_len_bytes},
    };
    for (uint32_t index = 0; index < 9; ++index) {
        if (ranges[index][0] == 0
            || (index > 0 && ranges[index][0] % alignof(float) != 0)) {
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t other = index + 1; other < 9; ++other) {
            bool overlap = false;
            if (!augur_batch01_b4_q6_ranges_overlap(
                    ranges[index][0], ranges[index][1],
                    ranges[other][0], ranges[other][1], &overlap)
                || overlap) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }

    if (validate_device_ranges) {
        augur_cuda_result_v1 device = augur_set_device(ordinal);
        if (device.status != AUGUR_CUDA_STATUS_OK) return device;
        cudaDeviceProp properties{};
        const cudaError_t property_error = cudaGetDeviceProperties(&properties, ordinal);
        if (property_error != cudaSuccess) return augur_runtime_error(property_error);
        if (properties.major != 7 || properties.minor != 0) {
            return augur_error(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE, cudaSuccess);
        }
        for (uint32_t index = 0; index < 9; ++index) {
            if (!augur_batch01_b4_q6_device_range_owned(
                    ranges[index][0], ranges[index][1], ordinal)) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    return augur_ok();
}

static augur_cuda_result_v1 augur_launch_batch01_b4_q6_block_diagonal_bounded_fp64(
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1* args
) {
    constexpr uint32_t threads = 128;
    constexpr uint32_t warps = threads / 32;
    const uint32_t rows = args->groups * args->rows_per_group;
    const uint32_t blocks = (rows + warps - 1) / warps;
    augur_batch01_b4_q6_block_diagonal_bounded_fp64_kernel_v1
        <<<blocks, threads, 0, stream->stream>>>(
            reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->weight)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input[0])),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input[1])),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input[2])),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input[3])),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->output[0])),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->output[1])),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->output[2])),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->output[3])),
            static_cast<int>(args->groups),
            static_cast<int>(args->rows_per_group),
            static_cast<int>(args->blocks_per_output_row));
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_validate_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1* args
) {
    return augur_validate_batch01_b4_q6_block_diagonal_bounded_fp64(
        ordinal, stream, args, true);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_submit_prevalidated_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1* args
) {
    augur_cuda_result_v1 valid =
        augur_validate_batch01_b4_q6_block_diagonal_bounded_fp64(
            ordinal, stream, args, false);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;

    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    return augur_launch_batch01_b4_q6_block_diagonal_bounded_fp64(stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1* args
) {
    augur_cuda_result_v1 valid =
        augur_validate_batch01_b4_q6_block_diagonal_bounded_fp64(
            ordinal, stream, args, true);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    return augur_launch_batch01_b4_q6_block_diagonal_bounded_fp64(stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_batch01_b4_q6_block_diagonal_bounded_fp64_submit_v1(
            ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}
#endif
#endif

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m120_q4k_exact_f32_decode_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q4k_f32_decode_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_q4k_args(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    if (args->rows != 2048 || args->cols != 4096 || args->blocks_per_row != 16) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const cudaError_t error = augur_m120_q4k_2048x4096_launch(
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)), stream->stream);
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m120_q5k_exact_f32_decode_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q5k_f32_decode_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_qk_decode_args(
        ordinal, stream, args, AUGUR_CUDA_Q5K_F32_DECODE_ARGS_VERSION_V1, AUGUR_CUDA_Q5K_BLOCK_BYTES_V1);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    if (args->rows != 4096 || args->cols != 2048 || args->blocks_per_row != 8) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const cudaError_t error = augur_m120_q5k_4096x2048_launch(
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)), stream->stream);
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m120_q6k_exact_f32_decode_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q6k_f32_decode_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_qk_decode_args(
        ordinal, stream, args, AUGUR_CUDA_Q6K_F32_DECODE_ARGS_VERSION_V1, AUGUR_CUDA_Q6K_BLOCK_BYTES_V1);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    if (args->rows != 4096 || args->cols != 4096 || args->blocks_per_row != 16) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const cudaError_t error = augur_m120_q6k_4096x4096_launch(
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)), stream->stream);
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m121_q4k_dp4a_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_m121_q4k_dp4a_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_M121_Q4K_DP4A_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_m121_q4k_dp4a_args_v1)
        || args->weight == 0 || args->q8_input == 0 || args->q8_scales == 0 || args->output == 0
        || args->rows != 2048 || args->cols != 4096 || args->blocks_per_row != 16
        || args->group_size != 32 || args->weight_len_bytes != UINT64_C(5242880)
        || args->q8_input_len_bytes != 4096 || args->q8_scales_len_bytes != 512
        || args->output_len_bytes < 8192 || args->weight % 32 != 0
        || args->q8_input % 4 != 0 || args->q8_scales % 4 != 0 || args->output % 4 != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) return selected;
    const cudaError_t error = augur_m121_q4k_dp4a_2048x4096_launch(
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const int8_t*>(static_cast<uintptr_t>(args->q8_input)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->q8_scales)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)), stream->stream);
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

static augur_cuda_result_v1 augur_validate_q4k_prefill_args(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q4k_f32_prefill_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != AUGUR_CUDA_Q4K_F32_PREFILL_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_q4k_f32_prefill_args_v1)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (args->weight == 0 || args->input == 0 || args->output == 0
        || args->m == 0 || args->rows == 0 || args->cols == 0
        || args->blocks_per_row == 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t expected_cols =
        static_cast<uint64_t>(args->blocks_per_row) * AUGUR_Q4K_BLOCK_LEN;
    if (expected_cols != args->cols
        || args->weight_block_stride_bytes != AUGUR_Q4K_BLOCK_BYTES) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t expected_weight_bytes =
        static_cast<uint64_t>(args->rows) * args->blocks_per_row * AUGUR_Q4K_BLOCK_BYTES;
    const uint64_t expected_input_bytes =
        static_cast<uint64_t>(args->m) * args->cols * sizeof(float);
    const uint64_t expected_output_bytes =
        static_cast<uint64_t>(args->m) * args->rows * sizeof(float);
    if (args->weight_len_bytes != expected_weight_bytes
        || args->input_len_bytes < expected_input_bytes
        || args->output_len_bytes < expected_output_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if ((args->weight % AUGUR_Q4K_BLOCK_ALIGNMENT) != 0
        || (args->input % alignof(float)) != 0
        || (args->output % alignof(float)) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_set_device(ordinal);
}

__global__ void augur_q4k_f32_prefill_kernel_v1(
    const unsigned char* weight,
    const float* input,
    float* output,
    uint32_t m,
    uint32_t rows,
    uint32_t cols,
    uint32_t blocks_per_row
);

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q4k_f32_prefill_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q4k_f32_prefill_args_v1* args
) {
    augur_cuda_result_v1 validation =
        augur_validate_q4k_prefill_args(ordinal, stream, args);
    if (validation.status != AUGUR_CUDA_STATUS_OK) {
        return validation;
    }
    const unsigned char* weight = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(args->weight));
    const float* input = reinterpret_cast<const float*>(
        static_cast<uintptr_t>(args->input));
    float* output = reinterpret_cast<float*>(
        static_cast<uintptr_t>(args->output));
    constexpr uint32_t threads = 128;
    const uint64_t elements = static_cast<uint64_t>(args->m) * args->rows;
    const uint32_t blocks = static_cast<uint32_t>((elements + threads - 1) / threads);
    augur_q4k_f32_prefill_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        weight, input, output, args->m, args->rows, args->cols, args->blocks_per_row);
    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
    }
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q4k_f32_prefill_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q4k_f32_prefill_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_q4k_f32_prefill_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) {
        return submitted;
    }
    return augur_cuda_stream_synchronize(stream);
}

static augur_cuda_result_v1 augur_validate_add_args(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_add_f32_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0
        || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != AUGUR_CUDA_ADD_F32_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_add_f32_args_v1)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (args->a == 0 || args->b == 0 || args->output == 0
        || args->elements == 0 || args->reserved != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t expected =
        static_cast<uint64_t>(args->elements) * sizeof(float);
    if (args->a_len_bytes < expected || args->b_len_bytes < expected
        || args->output_len_bytes < expected
        || (args->a % alignof(float)) != 0
        || (args->b % alignof(float)) != 0
        || (args->output % alignof(float)) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_set_device(ordinal);
}

__global__ void augur_add_f32_kernel_v1(
    const float* a,
    const float* b,
    float* output,
    uint32_t elements
);

__global__ void augur_add_f32_vec4_kernel_v1(
    const float4* a,
    const float4* b,
    float4* output,
    uint32_t vectors
);

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_add_f32_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_add_f32_args_v1* args
) {
    augur_cuda_result_v1 validation = augur_validate_add_args(ordinal, stream, args);
    if (validation.status != AUGUR_CUDA_STATUS_OK) {
        return validation;
    }
    const float* a = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->a));
    const float* b = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->b));
    float* output = reinterpret_cast<float*>(static_cast<uintptr_t>(args->output));
    constexpr uint32_t threads = 256;
    // Arena F32 views are normally 16-byte aligned. Use the vectorized path
    // only when the bound ABI pointers prove that contract; arbitrary aligned
    // callers retain the original scalar kernel.
    if ((args->elements % 4U) == 0U
        && (args->a % alignof(float4)) == 0U
        && (args->b % alignof(float4)) == 0U
        && (args->output % alignof(float4)) == 0U) {
        const uint32_t vectors = args->elements / 4U;
        const uint32_t blocks = static_cast<uint32_t>(
            (static_cast<uint64_t>(vectors) + threads - 1) / threads);
        augur_add_f32_vec4_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
            reinterpret_cast<const float4*>(a), reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(output), vectors);
    } else {
        const uint32_t blocks = static_cast<uint32_t>(
            (static_cast<uint64_t>(args->elements) + threads - 1) / threads);
        augur_add_f32_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
            a, b, output, args->elements);
    }
    const cudaError_t launch_error = cudaGetLastError();
    return launch_error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_add_f32_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_add_f32_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_add_f32_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) {
        return submitted;
    }
    return augur_cuda_stream_synchronize(stream);
}

static augur_cuda_result_v1 augur_validate_gqa_repeat_args(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_gqa_repeat_f32_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t source_bytes = static_cast<uint64_t>(args->source_elements) * sizeof(float);
    const uint64_t output_elements = static_cast<uint64_t>(args->source_elements) * args->repeats;
    const bool output_bytes_overflow = output_elements > UINT64_MAX / sizeof(float);
    const uint64_t output_bytes = output_bytes_overflow
        ? 0
        : output_elements * sizeof(float);
    if (args->abi_version != AUGUR_CUDA_GQA_REPEAT_F32_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_gqa_repeat_f32_args_v1)
        || args->input == 0 || args->output == 0
        || args->source_elements == 0 || args->head_dim == 0 || args->repeats == 0
        || args->reserved != 0 || args->source_elements % args->head_dim != 0
        || output_bytes_overflow
        || args->input_len_bytes < source_bytes || args->output_len_bytes < output_bytes
        || args->input % alignof(float) != 0 || args->output % alignof(float) != 0
        || args->input == args->output || output_elements == 0
        || output_elements > (static_cast<uint64_t>(0x7fffffffU) * 256U)) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_set_device(ordinal);
}

__global__ void augur_gqa_repeat_f32_kernel_v1(
    const float* input,
    float* output,
    uint32_t source_elements,
    uint32_t head_dim,
    uint32_t repeats
);

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_gqa_repeat_f32_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_gqa_repeat_f32_args_v1* args
) {
    augur_cuda_result_v1 validation = augur_validate_gqa_repeat_args(ordinal, stream, args);
    if (validation.status != AUGUR_CUDA_STATUS_OK) {
        return validation;
    }
    const float* input = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input));
    float* output = reinterpret_cast<float*>(static_cast<uintptr_t>(args->output));
    constexpr uint32_t threads = 256;
    const uint64_t output_elements =
        static_cast<uint64_t>(args->source_elements) * args->repeats;
    const uint32_t blocks = static_cast<uint32_t>((output_elements + threads - 1) / threads);
    augur_gqa_repeat_f32_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        input, output, args->source_elements, args->head_dim, args->repeats);
    const cudaError_t launch_error = cudaGetLastError();
    return launch_error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_gqa_repeat_f32_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_gqa_repeat_f32_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_gqa_repeat_f32_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) {
        return submitted;
    }
    return augur_cuda_stream_synchronize(stream);
}

static augur_cuda_result_v1 augur_validate_swiglu_args(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_swiglu_f32_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    float limit = 0.0f;
    std::memcpy(&limit, &args->limit_bits, sizeof(limit));
    const uint64_t expected = static_cast<uint64_t>(args->elements) * sizeof(float);
    if (args->abi_version != AUGUR_CUDA_SWIGLU_F32_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_swiglu_f32_args_v1)
        || args->gate == 0 || args->up == 0 || args->output == 0 || args->elements == 0
        || !isfinite(limit) || limit <= 0.0f
        || args->gate_len_bytes < expected || args->up_len_bytes < expected
        || args->output_len_bytes < expected || args->gate % alignof(float) != 0
        || args->up % alignof(float) != 0 || args->output % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_set_device(ordinal);
}

__global__ void augur_swiglu_f32_kernel_v1(
    const float* gate, const float* up, float* output, uint32_t elements, float limit
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float g = fminf(gate[index], limit);
    const float u = fminf(fmaxf(up[index], -limit), limit);
    output[index] = (g / (1.0f + expf(-g))) * u;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_swiglu_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_swiglu_f32_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_swiglu_args(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    constexpr uint32_t threads = 256;
    const uint32_t blocks = (args->elements + threads - 1) / threads;
    float limit = 0.0f;
    std::memcpy(&limit, &args->limit_bits, sizeof(limit));
    augur_swiglu_f32_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->up)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->elements, limit);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_swiglu_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_swiglu_f32_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_swiglu_f32_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

static augur_cuda_result_v1 augur_validate_silu_mul_args(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_f32_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0
        || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != AUGUR_CUDA_SILU_MUL_F32_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_silu_mul_f32_args_v1)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (args->gate == 0 || args->up == 0 || args->output == 0
        || args->elements == 0 || args->reserved != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t expected =
        static_cast<uint64_t>(args->elements) * sizeof(float);
    if (args->gate_len_bytes < expected || args->up_len_bytes < expected
        || args->output_len_bytes < expected
        || (args->gate % alignof(float)) != 0
        || (args->up % alignof(float)) != 0
        || (args->output % alignof(float)) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_set_device(ordinal);
}

#include "augur_qwen35_augur_silu_pack.cuh"

__global__ void augur_silu_mul_f32_kernel_v1(
    const float* gate, const float* up, float* output, uint32_t elements
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float g = gate[index];
    const float u = up[index];
    // No clamp is intentional.  Qwen's activation boundary is the raw
    // finite FP32 SiLU(gate) multiplied by the up projection.
    float sigmoid;
    if (g >= 0.0f) {
        sigmoid = 1.0f / (1.0f + expf(-g));
    } else {
        const float exponent = expf(g);
        sigmoid = exponent / (1.0f + exponent);
    }
    output[index] = (g * sigmoid) * u;
}

// Private cross-TU enqueue, called only after the closed down-projection entry
// validates every range and the stream. Same TU/arithmetic flags as legacy SiLU.
extern "C" cudaError_t augur_silu_pack_enqueue(
    cudaStream_t stream, const augur_cuda_silu_mul_f32_args_v1* a, uint64_t packed) {
    constexpr unsigned threads=256;
    const unsigned width=((a->gate|a->up|a->output)&15U)==0U?4U:
        (((a->gate|a->up|a->output)&7U)==0U?2U:1U);
    const uint64_t vectors=(uint64_t(a->elements)+width-1U)/width;
    const unsigned blocks=unsigned(vectors/threads+(vectors%threads!=0));
    const unsigned grid=blocks>65535U?65535U:blocks;
    if(width==4U) augur_silu_pack_f32_f16<4><<<grid,threads,0,stream>>>(
        (const float*)a->gate,(const float*)a->up,(float*)a->output,(half*)packed,a->elements);
    else if(width==2U) augur_silu_pack_f32_f16<2><<<grid,threads,0,stream>>>(
        (const float*)a->gate,(const float*)a->up,(float*)a->output,(half*)packed,a->elements);
    else augur_silu_pack_f32_f16<1><<<grid,threads,0,stream>>>(
        (const float*)a->gate,(const float*)a->up,(float*)a->output,(half*)packed,a->elements);
    return cudaGetLastError();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_silu_mul_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_f32_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_silu_mul_args(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    constexpr uint32_t threads = 256;
    const uint32_t blocks = (args->elements + threads - 1) / threads;
    augur_silu_mul_f32_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->up)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->elements);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_silu_mul_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_f32_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_silu_mul_f32_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

static augur_cuda_result_v1 augur_validate_dflash_grouped_conv_f32(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_grouped_conv_f32_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0
        || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_DFLASH_GROUPED_CONV_F32_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_dflash_grouped_conv_f32_args_v1)
        || args->hidden_f32 == 0 || args->dynamic_f32 == 0 || args->base_f32 == 0
        || args->output_f32 == 0 || args->hidden_elements == 0 || args->groups == 0
        || args->group_size == 0 || args->kernel_size == 0 || args->side >= 2
        || args->reserved != 0 || args->groups * args->group_size != args->hidden_elements) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t hidden_bytes = static_cast<uint64_t>(args->hidden_elements) * sizeof(float);
    const uint64_t dynamic_elements = static_cast<uint64_t>(2) * args->kernel_size * args->groups;
    const uint64_t base_elements = static_cast<uint64_t>(2) * args->kernel_size * args->hidden_elements;
    const uint64_t dynamic_bytes = dynamic_elements * sizeof(float);
    const uint64_t base_bytes = base_elements * sizeof(float);
    if (args->hidden_len_bytes != hidden_bytes || args->dynamic_len_bytes != dynamic_bytes
        || args->base_len_bytes != base_bytes || args->output_len_bytes != hidden_bytes
        || args->hidden_f32 % alignof(float) != 0 || args->dynamic_f32 % alignof(float) != 0
        || args->base_f32 % alignof(float) != 0 || args->output_f32 % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_set_device(ordinal);
}

__global__ void augur_dflash_grouped_conv_f32_kernel_v1(
    const float* hidden, const float* dynamic, const float* base, float* output,
    uint32_t hidden_elements, uint32_t groups, uint32_t group_size,
    uint32_t kernel_size, uint32_t side
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden_elements) return;
    const uint32_t group = index / group_size;
    float value = 0.0f;
    for (uint32_t tap = 0; tap < kernel_size; ++tap) {
        // The one-token boundary has no previous history, so only tap zero
        // contributes. Keeping the loop makes the ABI geometry explicit.
        if (tap != 0) continue;
        const uint32_t dynamic_index = (side * kernel_size + tap) * groups + group;
        const uint32_t base_index = (side * kernel_size + tap) * hidden_elements + index;
        value += (base[base_index] + dynamic[dynamic_index]) * hidden[index];
    }
    output[index] = value;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_grouped_conv_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_grouped_conv_f32_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_dflash_grouped_conv_f32(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    constexpr uint32_t threads = 256;
    const uint32_t blocks = (args->hidden_elements + threads - 1) / threads;
    augur_dflash_grouped_conv_f32_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->hidden_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->dynamic_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->base_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        args->hidden_elements, args->groups, args->group_size, args->kernel_size, args->side);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_grouped_conv_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_grouped_conv_f32_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_dflash_grouped_conv_f32_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

static augur_cuda_result_v1 augur_validate_dflash_grouped_conv_history_f32(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_grouped_conv_history_f32_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0
        || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_DFLASH_GROUPED_CONV_HISTORY_F32_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_dflash_grouped_conv_history_f32_args_v1)
        || args->hidden_f32 == 0 || args->history_f32 == 0 || args->dynamic_f32 == 0
        || args->base_f32 == 0 || args->output_f32 == 0 || args->hidden_elements == 0
        || args->groups == 0 || args->group_size == 0 || args->kernel_size != 2
        || args->side >= 2 || args->reserved != 0
        || args->groups * args->group_size != args->hidden_elements) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t hidden_bytes = static_cast<uint64_t>(args->hidden_elements) * sizeof(float);
    const uint64_t dynamic_bytes = static_cast<uint64_t>(2) * args->kernel_size
        * args->groups * sizeof(float);
    const uint64_t base_bytes = static_cast<uint64_t>(2) * args->kernel_size
        * args->hidden_elements * sizeof(float);
    if (args->hidden_len_bytes != hidden_bytes || args->history_len_bytes != hidden_bytes
        || args->dynamic_len_bytes != dynamic_bytes || args->base_len_bytes != base_bytes
        || args->output_len_bytes != hidden_bytes
        || args->hidden_f32 % alignof(float) != 0 || args->history_f32 % alignof(float) != 0
        || args->dynamic_f32 % alignof(float) != 0 || args->base_f32 % alignof(float) != 0
        || args->output_f32 % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_set_device(ordinal);
}

__global__ void augur_dflash_grouped_conv_history_f32_kernel_v1(
    const float* hidden, const float* history, const float* dynamic, const float* base,
    float* output, uint32_t hidden_elements, uint32_t groups, uint32_t group_size,
    uint32_t kernel_size, uint32_t side
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden_elements) return;
    const uint32_t group = index / group_size;
    float value = 0.0f;
    for (uint32_t tap = 0; tap < kernel_size; ++tap) {
        const float value_at_tap = tap == 0 ? hidden[index] : history[index];
        const uint32_t dynamic_index = (side * kernel_size + tap) * groups + group;
        const uint32_t base_index = (side * kernel_size + tap) * hidden_elements + index;
        value += (base[base_index] + dynamic[dynamic_index]) * value_at_tap;
    }
    output[index] = value;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_grouped_conv_history_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_grouped_conv_history_f32_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_dflash_grouped_conv_history_f32(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    constexpr uint32_t threads = 256;
    const uint32_t blocks = (args->hidden_elements + threads - 1) / threads;
    augur_dflash_grouped_conv_history_f32_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->hidden_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->history_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->dynamic_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->base_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        args->hidden_elements, args->groups, args->group_size, args->kernel_size, args->side);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_grouped_conv_history_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_grouped_conv_history_f32_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_dflash_grouped_conv_history_f32_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

static bool augur_q8_0_checked_range(uint64_t address, uint64_t extent);
static bool augur_q8_0_ranges_disjoint(
    uint64_t left, uint64_t left_len, uint64_t right, uint64_t right_len);

static augur_cuda_result_v1 augur_validate_dflash_grouped_conv_history_rows_f32(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_grouped_conv_history_rows_f32_args_v2* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0
        || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_DFLASH_GROUPED_CONV_HISTORY_ROWS_F32_ARGS_VERSION_V2
        || args->struct_size != sizeof(augur_cuda_dflash_grouped_conv_history_rows_f32_args_v2)
        || args->hidden_f32 == 0 || args->zero_f32 == 0 || args->dynamic_f32 == 0
        || args->base_f32 == 0 || args->output_f32 == 0 || args->rows == 0
        || args->rows > 8 || args->hidden_elements == 0 || args->groups == 0
        || args->group_size == 0 || args->kernel_size != 2 || args->side >= 2
        || args->reserved != 0 || args->groups * args->group_size != args->hidden_elements) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t row_bytes = static_cast<uint64_t>(args->hidden_elements) * sizeof(float);
    const uint64_t hidden_bytes = static_cast<uint64_t>(args->rows) * row_bytes;
    const uint64_t dynamic_bytes = static_cast<uint64_t>(args->rows) * 2U
        * args->kernel_size * args->groups * sizeof(float);
    const uint64_t base_bytes = static_cast<uint64_t>(2) * args->kernel_size
        * args->hidden_elements * sizeof(float);
    if (args->hidden_len_bytes != hidden_bytes || args->zero_len_bytes != row_bytes
        || args->dynamic_len_bytes != dynamic_bytes || args->base_len_bytes != base_bytes
        || args->output_len_bytes != hidden_bytes
        || args->hidden_f32 % alignof(float) != 0 || args->zero_f32 % alignof(float) != 0
        || args->dynamic_f32 % alignof(float) != 0 || args->base_f32 % alignof(float) != 0
        || args->output_f32 % alignof(float) != 0
        || !augur_q8_0_checked_range(args->hidden_f32, hidden_bytes)
        || !augur_q8_0_checked_range(args->zero_f32, row_bytes)
        || !augur_q8_0_checked_range(args->dynamic_f32, dynamic_bytes)
        || !augur_q8_0_checked_range(args->base_f32, base_bytes)
        || !augur_q8_0_checked_range(args->output_f32, hidden_bytes)
        || !augur_q8_0_ranges_disjoint(args->hidden_f32, hidden_bytes, args->zero_f32, row_bytes)
        || !augur_q8_0_ranges_disjoint(args->hidden_f32, hidden_bytes, args->dynamic_f32, dynamic_bytes)
        || !augur_q8_0_ranges_disjoint(args->hidden_f32, hidden_bytes, args->base_f32, base_bytes)
        || !augur_q8_0_ranges_disjoint(args->hidden_f32, hidden_bytes, args->output_f32, hidden_bytes)
        || !augur_q8_0_ranges_disjoint(args->zero_f32, row_bytes, args->dynamic_f32, dynamic_bytes)
        || !augur_q8_0_ranges_disjoint(args->zero_f32, row_bytes, args->base_f32, base_bytes)
        || !augur_q8_0_ranges_disjoint(args->zero_f32, row_bytes, args->output_f32, hidden_bytes)
        || !augur_q8_0_ranges_disjoint(args->dynamic_f32, dynamic_bytes, args->base_f32, base_bytes)
        || !augur_q8_0_ranges_disjoint(args->dynamic_f32, dynamic_bytes, args->output_f32, hidden_bytes)
        || !augur_q8_0_ranges_disjoint(args->base_f32, base_bytes, args->output_f32, hidden_bytes)) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_set_device(ordinal);
}

__global__ void augur_dflash_grouped_conv_history_rows_f32_kernel_v2(
    const float* hidden, const float* zero, const float* dynamic, const float* base,
    float* output, uint32_t rows, uint32_t hidden_elements, uint32_t groups,
    uint32_t group_size, uint32_t kernel_size, uint32_t side
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = rows * hidden_elements;
    if (index >= total) return;
    const uint32_t row = index / hidden_elements;
    const uint32_t element = index - row * hidden_elements;
    const uint32_t group = element / group_size;
    float value = 0.0f;
    for (uint32_t tap = 0; tap < kernel_size; ++tap) {
        const float* previous = row == 0 ? zero : hidden + static_cast<uint64_t>(row - 1) * hidden_elements;
        const float value_at_tap = tap == 0 ? hidden[index] : previous[element];
        const uint32_t dynamic_index = (static_cast<uint64_t>(row) * 2U * kernel_size
            + side * kernel_size + tap) * groups + group;
        const uint32_t base_index = (side * kernel_size + tap) * hidden_elements + element;
        value += (base[base_index] + dynamic[dynamic_index]) * value_at_tap;
    }
    output[index] = value;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_grouped_conv_history_rows_f32_submit_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_grouped_conv_history_rows_f32_args_v2* args
) {
    augur_cuda_result_v1 valid = augur_validate_dflash_grouped_conv_history_rows_f32(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    constexpr uint32_t threads = 256;
    const uint32_t total = args->rows * args->hidden_elements;
    const uint32_t blocks = (total + threads - 1) / threads;
    augur_dflash_grouped_conv_history_rows_f32_kernel_v2<<<blocks, threads, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->hidden_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->zero_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->dynamic_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->base_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)), args->rows,
        args->hidden_elements, args->groups, args->group_size, args->kernel_size, args->side);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_grouped_conv_history_rows_f32_v2(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_grouped_conv_history_rows_f32_args_v2* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_dflash_grouped_conv_history_rows_f32_submit_v2(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

static bool augur_q8_0_checked_range(uint64_t address, uint64_t extent);
static bool augur_q8_0_ranges_disjoint(
    uint64_t left, uint64_t left_len, uint64_t right, uint64_t right_len);

__global__ void augur_silu_mul_q8_1_kernel_v1(
    const float* gate, const float* up, uint8_t* output, uint32_t elements
) {
    const uint32_t warp = threadIdx.x / 32U;
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t activation_block = blockIdx.x * 4U + warp;
    const uint32_t block_count = elements / 32U;
    if (activation_block >= block_count) return;
    const uint64_t base = static_cast<uint64_t>(activation_block)
        * 32U;
    const float g = gate[base + lane];
    const float u = up[base + lane];
    float sigmoid;
    if (g >= 0.0f) {
        sigmoid = 1.0f / (1.0f + expf(-g));
    } else {
        const float exponent = expf(g);
        sigmoid = exponent / (1.0f + exponent);
    }
    const float value = (g * sigmoid) * u;
    float maximum = fabsf(value);
    float sum = value;
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    maximum = __shfl_sync(0xffffffffU, maximum, 0);
    const float scale = maximum / 127.0f;
    const float inverse = maximum != 0.0f ? 127.0f / maximum : 0.0f;
    int32_t code = __float2int_rn(value * inverse);
    code = max(-128, min(127, code));
    uint8_t* destination = output
        + static_cast<uint64_t>(activation_block) * 36U;
    destination[4U + lane] = static_cast<uint8_t>(static_cast<int8_t>(code));
    if (lane == 0) {
        auto* metadata = reinterpret_cast<uint16_t*>(destination);
        metadata[0] = __half_as_ushort(__float2half_rn(scale));
        metadata[1] = __half_as_ushort(__float2half_rn(sum));
    }
}

static augur_cuda_result_v1 augur_validate_silu_mul_q8_1_args(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_q8_1_args_v1* args
) {
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    bool footprint_ok = false;
    if (args != nullptr && args->elements != 0U
        && (args->elements % 32U) == 0U) {
        const uint64_t block_count = static_cast<uint64_t>(args->elements) / 32U;
        if (static_cast<uint64_t>(args->elements) <= UINT64_MAX / sizeof(float)
            && block_count <= UINT64_MAX / 36U) {
            input_bytes = static_cast<uint64_t>(args->elements) * sizeof(float);
            output_bytes = block_count * 36U;
            footprint_ok = input_bytes <= SIZE_MAX && output_bytes <= SIZE_MAX;
        }
    }
    if (stream == nullptr || stream->stream == nullptr || args == nullptr
        || ordinal < 0 || stream->ordinal != ordinal || !footprint_ok
        || args->abi_version != AUGUR_CUDA_SILU_MUL_Q8_1_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_silu_mul_q8_1_args_v1)
        || args->reserved != 0 || args->gate_len_bytes < input_bytes
        || args->up_len_bytes < input_bytes
        || args->output_len_bytes != output_bytes
        || !augur_q8_0_checked_range(args->gate, input_bytes)
        || !augur_q8_0_checked_range(args->up, input_bytes)
        || !augur_q8_0_checked_range(args->output_q8_1, output_bytes)
        || (args->gate % alignof(float)) != 0
        || (args->up % alignof(float)) != 0
        || (args->output_q8_1 % alignof(uint32_t)) != 0
        || !augur_q8_0_ranges_disjoint(args->gate, input_bytes, args->up, input_bytes)
        || !augur_q8_0_ranges_disjoint(args->gate, input_bytes, args->output_q8_1, output_bytes)
        || !augur_q8_0_ranges_disjoint(args->up, input_bytes, args->output_q8_1, output_bytes)) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_set_device(ordinal);
}

// Keep this in the runtime TU with the separate SiLU/quantizer arithmetic.
// Unlike the legacy fused primitive, the F32 activation remains observable.
__global__ void augur_silu_q8_preserved_b8_kernel_v1(
    const float* gate, const float* up, float* preserved, uint8_t* output) {
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t block = blockIdx.x * 4U + threadIdx.x / 32U;
    const uint32_t index = block * 32U + lane;
    const float g = gate[index], u = up[index];
    float sigmoid;
    if (g >= 0.0f) {
        sigmoid = 1.0f / (1.0f + expf(-g));
    } else {
        const float exponent = expf(g);
        sigmoid = exponent / (1.0f + exponent);
    }
    const float value = (g * sigmoid) * u;
    preserved[index] = value;
    float maximum = fabsf(value), sum = value;
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    maximum = __shfl_sync(0xffffffffU, maximum, 0);
    const float scale = maximum / 127.0f;
    const float inverse = maximum != 0.0f ? 127.0f / maximum : 0.0f;
    int32_t code = __float2int_rn(value * inverse);
    code = max(-128, min(127, code));
    uint8_t* destination = output + static_cast<uint64_t>(block) * 36U;
    destination[4U + lane] = static_cast<uint8_t>(static_cast<int8_t>(code));
    if (lane == 0U) {
        auto* metadata = reinterpret_cast<uint16_t*>(destination);
        metadata[0] = __half_as_ushort(__float2half_rn(scale));
        metadata[1] = __half_as_ushort(__float2half_rn(sum));
    }
}

static bool augur_qwen35_boundary_device_range(uint64_t address, uint64_t bytes, int32_t ordinal) {
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes,
            reinterpret_cast<const void*>(static_cast<uintptr_t>(address))) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != ordinal) return false;
    CUdeviceptr base = 0;
    size_t extent = 0;
    if (cuMemGetAddressRange(&base, &extent, static_cast<CUdeviceptr>(address)) != CUDA_SUCCESS)
        return false;
    return base <= address && extent <= UINT64_MAX - base && address + bytes <= base + extent;
}

static augur_cuda_result_v1 augur_validate_silu_q8_preserved_b8(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_silu_mul_f32_args_v1* s,
    const augur_cuda_q8_1_activation_quantize_args_v1* q, bool inspect_device) {
    constexpr uint32_t elements = 8U * 17408U;
    constexpr uint64_t f32_bytes = elements * 4U, q8_bytes = elements / 32U * 36U;
    if (ordinal < 0 || stream == nullptr || stream->stream == nullptr || stream->ordinal != ordinal
        || s == nullptr || q == nullptr || s->abi_version != 1U || s->struct_size != sizeof(*s)
        || q->abi_version != 1U || q->struct_size != sizeof(*q)
        || s->reserved != 0U || q->reserved != 0U || s->elements != elements || q->elements != elements
        || s->gate_len_bytes != f32_bytes || s->up_len_bytes != f32_bytes || s->output_len_bytes != f32_bytes
        || q->input_f32 != s->output || q->input_len_bytes != f32_bytes || q->output_len_bytes != q8_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t address[] = {s->gate, s->up, s->output, q->output_q8_1};
    const uint64_t bytes[] = {f32_bytes, f32_bytes, f32_bytes, q8_bytes};
    for (uint32_t i = 0; i < 4U; ++i) {
        if (!augur_q8_0_checked_range(address[i], bytes[i]) || (address[i] % 4U) != 0U)
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        for (uint32_t j = 0; j < i; ++j) {
            if (!augur_q8_0_ranges_disjoint(address[i], bytes[i], address[j], bytes[j]))
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
    }
#if defined(AUGUR_CUDA_QWEN35_DIAGNOSTIC_Q8_AWAY)
    // This ties-to-even B8 candidate never silently overrides a diagnostic build.
    return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
#else
    const augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    if (inspect_device) {
        for (uint32_t i = 0; i < 4U; ++i) {
            if (!augur_qwen35_boundary_device_range(address[i], bytes[i], ordinal))
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
    }
    return augur_ok();
#endif
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_silu_q8_preserved_b8_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_silu_mul_f32_args_v1* s,
    const augur_cuda_q8_1_activation_quantize_args_v1* q) {
    return augur_validate_silu_q8_preserved_b8(ordinal, stream, s, q, true);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_silu_q8_preserved_b8_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_silu_mul_f32_args_v1* s,
    const augur_cuda_q8_1_activation_quantize_args_v1* q) {
    const augur_cuda_result_v1 valid = augur_validate_silu_q8_preserved_b8(ordinal, stream, s, q, false);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    augur_silu_q8_preserved_b8_kernel_v1<<<(8U * 17408U) / 128U, 128U, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(s->gate)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(s->up)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(s->output)),
        reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(q->output_q8_1)));
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

// Same runtime-TU norm leaf assignment, FMA policy and reduction tree. Preserve
// last-row scratches in their original buffers, not aliases into batch storage.
__global__ void augur_qwen35_head_norm_preserved_b8_kernel_v1(
    const float* input, const float* weight, float* output, float epsilon,
    float* last_hidden, float* last_normalized) {
    constexpr uint32_t elements = 5120U;
    input += static_cast<uint64_t>(blockIdx.x) * elements;
    output += static_cast<uint64_t>(blockIdx.x) * elements;
    __shared__ float partial[256];
    const uint32_t lane = threadIdx.x;
    float sum = 0.0f;
    for (uint32_t index = lane; index < elements; index += blockDim.x) sum += input[index] * input[index];
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / static_cast<float>(elements) + epsilon);
    for (uint32_t index = lane; index < elements; index += blockDim.x) {
        const float value = input[index] * scale * weight[index];
        output[index] = value;
        if (blockIdx.x == 7U) { last_hidden[index] = input[index]; last_normalized[index] = value; }
    }
}

template<bool Draft>
__global__ void augur_qwen35_head_quant_preserved_b8_kernel_v1(
    const float* input, uint8_t* output, uint8_t* last_q, float* last_normalized) {
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t block = blockIdx.x * 4U + threadIdx.x / 32U;
    const float value = input[block * 32U + lane];
    const bool last_row = block >= 7U * 160U;
    const uint32_t local_block = block - 7U * 160U;
    if constexpr (Draft) {
        if (last_row) last_normalized[local_block * 32U + lane] = value;
    }
    float maximum = fabsf(value), sum = value;
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    maximum = __shfl_sync(0xffffffffU, maximum, 0);
    const float scale = maximum / 127.0f;
    const float inverse = maximum != 0.0f ? 127.0f / maximum : 0.0f;
    int32_t code = max(-128, min(127, __float2int_rn(value * inverse)));
    uint8_t* destination = output + static_cast<uint64_t>(block) * 36U;
    destination[4U + lane] = static_cast<uint8_t>(static_cast<int8_t>(code));
    if (last_row) last_q[local_block * 36U + 4U + lane] = static_cast<uint8_t>(static_cast<int8_t>(code));
    if (lane == 0U) {
        auto* metadata = reinterpret_cast<uint16_t*>(destination);
        metadata[0] = __half_as_ushort(__float2half_rn(scale));
        metadata[1] = __half_as_ushort(__float2half_rn(sum));
        if (last_row) {
            auto* last = reinterpret_cast<uint16_t*>(last_q + local_block * 36U);
            last[0] = __half_as_ushort(__float2half_rn(scale));
            last[1] = __half_as_ushort(__float2half_rn(sum));
        }
    }
}

static_assert(sizeof(augur_cuda_qwen35_head_prepare_b8_args_v1) == 128, "head prepare ABI");
static augur_cuda_result_v1 augur_validate_qwen35_head_prepare_b8(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_head_prepare_b8_args_v1* a,
    bool inspect_device) {
    if (ordinal < 0 || stream == nullptr || stream->stream == nullptr || stream->ordinal != ordinal
        || a == nullptr || a->abi_version != 1U || a->struct_size != sizeof(*a)
        || (a->mode != 1U && a->mode != 2U)) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    constexpr uint64_t row = 5120U * 4U, qrow = 5120U / 32U * 36U;
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &a->epsilon_bits, sizeof(epsilon));
    if ((a->mode == 1U && (!isfinite(epsilon) || epsilon <= 0.0f))
        || (a->mode == 2U && a->epsilon_bits != 0U)) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    const uint64_t ptr[] = {a->input, a->quantized, a->last_normalized, a->last_quantized,
        a->weight, a->normalized, a->last_hidden};
    const uint64_t len[] = {a->input_len_bytes, a->quantized_len_bytes, a->last_normalized_len_bytes,
        a->last_quantized_len_bytes, a->weight_len_bytes, a->normalized_len_bytes, a->last_hidden_len_bytes};
    const uint64_t want[] = {row * 8U, qrow * 8U, row, qrow, row, row * 8U, row};
    const uint32_t active = a->mode == 1U ? 7U : 4U;
    for (uint32_t i = 0; i < 7U; ++i) {
        if (i >= active) { if (ptr[i] != 0U || len[i] != 0U) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess); continue; }
        if (len[i] != want[i] || !augur_q8_0_checked_range(ptr[i], len[i]) || ptr[i] % 4U != 0U)
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        for (uint32_t j = 0; j < i; ++j) if (!augur_q8_0_ranges_disjoint(ptr[i], len[i], ptr[j], len[j]))
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
#if defined(AUGUR_CUDA_QWEN35_DIAGNOSTIC_Q8_AWAY)
    return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
#else
    const augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    if (inspect_device) for (uint32_t i = 0; i < active; ++i)
        if (!augur_qwen35_boundary_device_range(ptr[i], len[i], ordinal)) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    return augur_ok();
#endif
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_head_prepare_b8_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_head_prepare_b8_args_v1* a) {
    return augur_validate_qwen35_head_prepare_b8(ordinal, stream, a, true);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_head_prepare_b8_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_qwen35_head_prepare_b8_args_v1* a) {
    const augur_cuda_result_v1 valid = augur_validate_qwen35_head_prepare_b8(ordinal, stream, a, false);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    const auto* input = reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input));
    auto* q = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(a->quantized));
    auto* last_q = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(a->last_quantized));
    auto* last_n = reinterpret_cast<float*>(static_cast<uintptr_t>(a->last_normalized));
    if (a->mode == 1U) {
        float epsilon = 0.0f; std::memcpy(&epsilon, &a->epsilon_bits, sizeof(epsilon));
        auto* normalized = reinterpret_cast<float*>(static_cast<uintptr_t>(a->normalized));
        augur_qwen35_head_norm_preserved_b8_kernel_v1<<<8U, 256U, 0, stream->stream>>>(input,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->weight)), normalized, epsilon,
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->last_hidden)), last_n);
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
        augur_qwen35_head_quant_preserved_b8_kernel_v1<false><<<320U, 128U, 0, stream->stream>>>(normalized, q, last_q, last_n);
    } else {
        augur_qwen35_head_quant_preserved_b8_kernel_v1<true><<<320U, 128U, 0, stream->stream>>>(input, q, last_q, last_n);
    }
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_silu_mul_q8_1_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_q8_1_args_v1* args
) {
    const augur_cuda_result_v1 valid = augur_validate_silu_mul_q8_1_args(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    const uint32_t blocks = (args->elements / 32U + 4U - 1U) / 4U;
    augur_silu_mul_q8_1_kernel_v1<<<blocks, 128U, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->up)),
        reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(args->output_q8_1)),
        args->elements);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_silu_mul_q8_1_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_silu_mul_q8_1_args_v1* args
) {
    const augur_cuda_result_v1 submitted = augur_cuda_silu_mul_q8_1_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

__global__ void augur_bf16_f32_matvec_kernel_v1(
    const uint16_t* weight, const float* input, float* output,
    uint32_t rows, uint32_t cols
) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    double sum = 0.0;
    const uint16_t* source = weight + static_cast<size_t>(row) * cols;
    for (uint32_t col = 0; col < cols; ++col) {
        sum += __uint_as_float(static_cast<uint32_t>(source[col]) << 16) * input[col];
    }
    output[row] = sum;
}

constexpr uint32_t AUGUR_M307_BF16_STAGED_THREADS = 128;
constexpr uint32_t AUGUR_M307_BF16_STAGED_WARPS = AUGUR_M307_BF16_STAGED_THREADS / 32;
constexpr uint32_t AUGUR_M307_BF16_STAGED_ELEMENTS = 64;
constexpr uint32_t AUGUR_M307_BF16_STAGED_VECTORS =
    AUGUR_M307_BF16_STAGED_ELEMENTS * sizeof(uint16_t) / sizeof(uint4);
constexpr uint32_t AUGUR_M307_BF16_STAGED_WORDS = 33;

__global__ void augur_m307_bf16_f32_matvec_staged_kernel_v1(
    const uint16_t* weight, const float* input, float* output,
    uint32_t rows, uint32_t cols
) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t row = blockIdx.x * AUGUR_M307_BF16_STAGED_THREADS + threadIdx.x;
    __shared__ uint32_t staged[AUGUR_M307_BF16_STAGED_WARPS][32][AUGUR_M307_BF16_STAGED_WORDS];
    if (row >= rows) return;
    const uint16_t* source = weight + static_cast<size_t>(row) * cols;
    double sum = 0.0;
    for (uint32_t tile = 0; tile < cols; tile += AUGUR_M307_BF16_STAGED_ELEMENTS) {
        const auto* vectors = reinterpret_cast<const uint4*>(source + tile);
        #pragma unroll
        for (uint32_t vector = 0; vector < AUGUR_M307_BF16_STAGED_VECTORS; ++vector) {
            const uint4 value = vectors[vector];
            staged[warp][lane][vector * 4 + 0] = value.x;
            staged[warp][lane][vector * 4 + 1] = value.y;
            staged[warp][lane][vector * 4 + 2] = value.z;
            staged[warp][lane][vector * 4 + 3] = value.w;
        }
        const auto* tile_values = reinterpret_cast<const uint16_t*>(staged[warp][lane]);
        for (uint32_t local = 0; local < AUGUR_M307_BF16_STAGED_ELEMENTS; ++local) {
            const float value = __uint_as_float(static_cast<uint32_t>(tile_values[local]) << 16);
            sum += value * input[tile + local];
        }
    }
    output[row] = sum;
}

#if defined(AUGUR_CUDA_M306_K17_BF16_WARP_ROW)
__global__ void augur_m306_k17_bf16_f32_matvec_warp_row_fp64_v1(
    const uint16_t* weight, const float* input, float* output,
    uint32_t rows, uint32_t cols
) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t row = blockIdx.x * AUGUR_M307_BF16_STAGED_WARPS + warp;
    if (row >= rows) return;
    const uint16_t* source = weight + static_cast<size_t>(row) * cols;
    double partial = 0.0;
    for (uint32_t col = lane; col < cols; col += 32) {
        const float value = __uint_as_float(static_cast<uint32_t>(source[col]) << 16);
        partial += value * input[col];
    }
    #pragma unroll
    for (uint32_t offset = 16; offset != 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    }
    if (lane == 0) output[row] = static_cast<float>(partial);
}
#endif

#if defined(AUGUR_CUDA_M306_K17_BF16_WARP_ROW)
// The token grid changes only independent row scheduling. Preserve the scalar
// entry above, including F32 products, FP64 lane partials and its warp tree.
__global__ void augur_bf16_f32_matvec_rows_warp_fp64_kernel_v1(
    const uint16_t* weight, const float* input, float* output,
    uint32_t rows, uint32_t cols
) {
    input += static_cast<size_t>(blockIdx.y) * cols;
    output += static_cast<size_t>(blockIdx.y) * rows;
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t row = blockIdx.x * AUGUR_M307_BF16_STAGED_WARPS + warp;
    if (row >= rows) return;
    const uint16_t* source = weight + static_cast<size_t>(row) * cols;
    double partial = 0.0;
    for (uint32_t col = lane; col < cols; col += 32) {
        const float value = __uint_as_float(static_cast<uint32_t>(source[col]) << 16);
        partial += value * input[col];
    }
    #pragma unroll
    for (uint32_t offset = 16; offset != 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    }
    if (lane == 0) output[row] = static_cast<float>(partial);
}
#endif

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_f32_matvec_rows_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_bf16_f32_matvec_args_v1* args, uint32_t tokens
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_BF16_F32_MATVEC_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_bf16_f32_matvec_args_v1)
        || tokens == 0 || tokens > 512 || args->rows == 0 || args->rows > 65535
        || args->cols == 0 || args->cols > 65535 || args->weight == 0 || args->input == 0 || args->output == 0
        || args->weight % alignof(uint16_t) != 0 || args->input % alignof(float) != 0
        || args->output % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    // Bounded rows/tokens make all products non-overflowing, including u32 cols.
    const uint64_t w = static_cast<uint64_t>(args->rows) * args->cols * 2;
    const uint64_t x = static_cast<uint64_t>(tokens) * args->cols * 4;
    const uint64_t y = static_cast<uint64_t>(tokens) * args->rows * 4;
    if (args->weight_len_bytes != w || args->input_len_bytes != x || args->output_len_bytes != y) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t ranges[3][2] = {{args->weight, w}, {args->input, x}, {args->output, y}};
    for (uint32_t i = 0; i < 3; ++i) {
        if (ranges[i][0] > UINT64_MAX - ranges[i][1]) {
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (ranges[i][0] < ranges[j][0] + ranges[j][1]
                && ranges[j][0] < ranges[i][0] + ranges[i][1]) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
#if defined(AUGUR_CUDA_M306_K17_BF16_WARP_ROW)
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    const uint32_t blocks = (args->rows + AUGUR_M307_BF16_STAGED_WARPS - 1) / AUGUR_M307_BF16_STAGED_WARPS;
    augur_bf16_f32_matvec_rows_warp_fp64_kernel_v1
        <<<dim3(blocks, tokens), AUGUR_M307_BF16_STAGED_THREADS, 0, stream->stream>>>(
            reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(args->weight)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)), args->rows, args->cols);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
#else
    // No silent arithmetic fallback for a canonical-row contract.
    return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
#endif
}

#if defined(AUGUR_CUDA_P0_PREFILL_V22_PROJECTION_RESUME_BATCH_EXACT)
constexpr uint32_t AUGUR_P0_PREFILL_V22_TOKENS_PER_WARP = 4;

__global__ void augur_p0_prefill_v22_bf16_batch4_kernel_v1(
    const uint16_t* weight,
    const float* input,
    float* output,
    uint32_t rows,
    uint32_t cols
) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t row = blockIdx.x * AUGUR_M307_BF16_STAGED_WARPS + warp;
    const uint32_t token_base = blockIdx.y * AUGUR_P0_PREFILL_V22_TOKENS_PER_WARP;
    if (row >= rows) return;

    const uint16_t* source = weight + static_cast<size_t>(row) * cols;
    double partial[AUGUR_P0_PREFILL_V22_TOKENS_PER_WARP] = {};
    for (uint32_t col = lane; col < cols; col += 32) {
        const float value = __uint_as_float(static_cast<uint32_t>(source[col]) << 16);
        #pragma unroll
        for (uint32_t token = 0; token < AUGUR_P0_PREFILL_V22_TOKENS_PER_WARP; ++token) {
            partial[token] += value * input[static_cast<size_t>(token_base + token) * cols + col];
        }
    }
    #pragma unroll
    for (uint32_t offset = 16; offset != 0; offset >>= 1) {
        #pragma unroll
        for (uint32_t token = 0; token < AUGUR_P0_PREFILL_V22_TOKENS_PER_WARP; ++token) {
            partial[token] += __shfl_down_sync(0xffffffffu, partial[token], offset);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (uint32_t token = 0; token < AUGUR_P0_PREFILL_V22_TOKENS_PER_WARP; ++token) {
            output[static_cast<size_t>(token_base + token) * rows + row] =
                static_cast<float>(partial[token]);
        }
    }
}
#endif

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_f32_matvec_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_bf16_f32_matvec_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_BF16_F32_MATVEC_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_bf16_f32_matvec_args_v1)
        || args->weight == 0 || args->input == 0 || args->output == 0
        || args->rows == 0 || args->cols == 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t weight_bytes = static_cast<uint64_t>(args->rows) * args->cols * 2;
    if (args->weight_len_bytes != weight_bytes
        || args->input_len_bytes < static_cast<uint64_t>(args->cols) * 4
        || args->output_len_bytes < static_cast<uint64_t>(args->rows) * 4
        || args->weight % alignof(uint16_t) != 0 || args->input % alignof(float) != 0
        || args->output % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    constexpr uint32_t threads = AUGUR_M307_BF16_STAGED_THREADS;
    const uint32_t blocks = (args->rows + threads - 1) / threads;
    const auto* weight = reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(args->weight));
    const auto* input = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input));
    auto* output = reinterpret_cast<float*>(static_cast<uintptr_t>(args->output));
#if defined(AUGUR_CUDA_M306_K17_BF16_WARP_ROW)
    const uint32_t warp_row_blocks =
        (args->rows + AUGUR_M307_BF16_STAGED_WARPS - 1) / AUGUR_M307_BF16_STAGED_WARPS;
    augur_m306_k17_bf16_f32_matvec_warp_row_fp64_v1
        <<<warp_row_blocks, threads, 0, stream->stream>>>(
            weight, input, output, args->rows, args->cols);
#else
    if (args->weight % alignof(uint4) == 0 && args->cols % AUGUR_M307_BF16_STAGED_ELEMENTS == 0) {
        augur_m307_bf16_f32_matvec_staged_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
            weight, input, output, args->rows, args->cols);
    } else {
        augur_bf16_f32_matvec_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
            weight, input, output, args->rows, args->cols);
    }
#endif
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_ok();
}

#if defined(AUGUR_CUDA_P0_PREFILL_V22_PROJECTION_RESUME_BATCH_EXACT)
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_p0_prefill_v22_bf16_batch4_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_bf16_f32_matvec_args_v1* args,
    uint32_t tokens
) {
    const bool shape_ok = args != nullptr
        && ((args->rows == 256 && args->cols == 4096)
            || (args->rows == 64 && args->cols == 4096));
    const uint64_t weight_bytes = shape_ok
        ? static_cast<uint64_t>(args->rows) * args->cols * 2
        : 0;
    const uint64_t input_bytes = shape_ok
        ? static_cast<uint64_t>(tokens) * args->cols * 4
        : 0;
    const uint64_t output_bytes = shape_ok
        ? static_cast<uint64_t>(tokens) * args->rows * 4
        : 0;
    if (stream == nullptr || stream->stream == nullptr || args == nullptr || ordinal < 0
        || stream->ordinal != ordinal || tokens != 128 || !shape_ok
        || args->abi_version != AUGUR_CUDA_BF16_F32_MATVEC_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_bf16_f32_matvec_args_v1)
        || args->weight == 0 || args->input == 0 || args->output == 0
        || args->weight_len_bytes != weight_bytes
        || args->input_len_bytes != input_bytes
        || args->output_len_bytes != output_bytes
        || args->weight % alignof(uint16_t) != 0
        || args->input % alignof(float) != 0
        || args->output % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    constexpr uint32_t threads = AUGUR_M307_BF16_STAGED_THREADS;
    const dim3 grid(
        (args->rows + AUGUR_M307_BF16_STAGED_WARPS - 1) / AUGUR_M307_BF16_STAGED_WARPS,
        tokens / AUGUR_P0_PREFILL_V22_TOKENS_PER_WARP);
    augur_p0_prefill_v22_bf16_batch4_kernel_v1<<<grid, threads, 0, stream->stream>>>(
        reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->rows,
        args->cols);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}
#endif

#include "nerkyor_bf16_gate_pair_f32_cta.cuh"

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_f32_matvec_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_bf16_f32_matvec_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_bf16_f32_matvec_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

__device__ __forceinline__ float augur_q8_0_f16_to_f32(uint16_t bits) {
#if defined(AUGUR_CUDA_QWEN35_NATIVE_F16_SCALE)
    return __half2float(__ushort_as_half(bits));
#else
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fU;
    const uint32_t mantissa = bits & 0x03ffU;
    if (exponent == 0) {
        if (mantissa == 0) return __int_as_float(sign);
        uint32_t normalized = mantissa;
        int32_t shift = 0;
        while ((normalized & 0x0400U) == 0) {
            normalized <<= 1;
            --shift;
        }
        normalized &= 0x03ffU;
        const uint32_t f32_exponent = static_cast<uint32_t>(127 - 15 + 1 + shift);
        return __int_as_float(sign | (f32_exponent << 23) | (normalized << 13));
    }
    if (exponent == 0x1fU) {
        return __int_as_float(sign | 0x7f800000U | (mantissa << 13));
    }
    return __int_as_float(sign | ((exponent + 112U) << 23) | (mantissa << 13));
#endif
}

__global__ void augur_q8_0_f32_matvec_kernel_v1(
    const uint8_t* weight, const float* input, float* output,
    uint32_t rows, uint32_t cols
) {
    constexpr uint32_t warp_size = 32;
    constexpr uint32_t warps_per_block = 4;
    const uint32_t warp = threadIdx.x / warp_size;
    const uint32_t lane = threadIdx.x & (warp_size - 1U);
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * warps_per_block + warp;
    if (row >= rows) return;
    constexpr uint32_t block_elements = 32;
    constexpr uint32_t block_bytes = 34;
    const uint32_t blocks_per_row = cols / block_elements;
    const uint64_t row_bytes = static_cast<uint64_t>(blocks_per_row) * block_bytes;
    const uint8_t* source = weight + static_cast<size_t>(row * row_bytes);
    double partial = 0.0;
    for (uint32_t block = 0; block < blocks_per_row; ++block) {
        uint32_t scale_bits = lane == 0
            ? static_cast<uint32_t>(source[0]) | static_cast<uint32_t>(source[1]) << 8
            : 0U;
        scale_bits = __shfl_sync(0xffffffffU, scale_bits, 0);
        const double scale = static_cast<double>(augur_q8_0_f16_to_f32(scale_bits));
        const uint32_t input_base = block * block_elements;
        const uint8_t encoded = source[2 + lane];
        const int32_t code = encoded < 128U
            ? static_cast<int32_t>(encoded)
            : static_cast<int32_t>(encoded) - 256;
        partial += static_cast<double>(code) * scale
            * static_cast<double>(input[input_base + lane]);
        source += block_bytes;
    }
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U) {
        partial += __shfl_down_sync(0xffffffffU, partial, offset);
    }
    if (lane == 0) output[row] = static_cast<float>(partial);
}

static bool augur_q8_0_checked_footprint(
    uint32_t rows, uint32_t cols,
    uint64_t* weight_bytes, uint64_t* input_bytes, uint64_t* output_bytes
) {
    if (weight_bytes == nullptr || input_bytes == nullptr || output_bytes == nullptr
        || rows == 0 || cols == 0 || (cols % 32U) != 0) {
        return false;
    }
    const uint64_t blocks_per_row = static_cast<uint64_t>(cols) / 32U;
    if (blocks_per_row > UINT64_MAX / 34U) return false;
    const uint64_t row_bytes = blocks_per_row * 34U;
    if (static_cast<uint64_t>(rows) > UINT64_MAX / row_bytes) return false;
    *weight_bytes = static_cast<uint64_t>(rows) * row_bytes;
    if (static_cast<uint64_t>(cols) > UINT64_MAX / sizeof(float)
        || static_cast<uint64_t>(rows) > UINT64_MAX / sizeof(float)) {
        return false;
    }
    *input_bytes = static_cast<uint64_t>(cols) * sizeof(float);
    *output_bytes = static_cast<uint64_t>(rows) * sizeof(float);
    return *weight_bytes <= SIZE_MAX && *input_bytes <= SIZE_MAX && *output_bytes <= SIZE_MAX;
}

static bool augur_q8_0_checked_range(uint64_t address, uint64_t extent) {
    return address != 0 && extent != 0 && address <= UINT64_MAX - extent;
}

static bool augur_q8_0_ranges_disjoint(
    uint64_t left,
    uint64_t left_len,
    uint64_t right,
    uint64_t right_len
) {
    return left + left_len <= right || right + right_len <= left;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q8_0_f32_matvec_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q8_0_f32_matvec_args_v1* args
) {
    uint64_t weight_bytes = 0;
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    const bool footprint_ok = args != nullptr && augur_q8_0_checked_footprint(
        args->rows, args->cols, &weight_bytes, &input_bytes, &output_bytes);
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || !footprint_ok
        || args->abi_version != AUGUR_CUDA_Q8_0_F32_MATVEC_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_q8_0_f32_matvec_args_v1)
        || args->weight_len_bytes != weight_bytes
        || args->input_len_bytes < input_bytes
        || args->output_len_bytes < output_bytes
        || !augur_q8_0_checked_range(args->weight, weight_bytes)
        || !augur_q8_0_checked_range(args->input, input_bytes)
        || !augur_q8_0_checked_range(args->output, output_bytes)
        || (args->weight % alignof(uint16_t)) != 0
        || (args->input % alignof(float)) != 0
        || (args->output % alignof(float)) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    constexpr uint32_t warps_per_block = 4;
    constexpr uint32_t threads = warps_per_block * 32;
    const uint32_t blocks = (args->rows + warps_per_block - 1) / warps_per_block;
    augur_q8_0_f32_matvec_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->rows, args->cols);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q8_0_f32_matvec_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_q8_0_f32_matvec_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_q8_0_f32_matvec_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

namespace {

constexpr uint32_t augur_q8_1_block_elements_v1 = 32U;
constexpr uint32_t augur_q8_1_block_bytes_v1 = 36U;
constexpr uint32_t augur_q8_1_matvec_warps_v1 = 4U;

__device__ __forceinline__ uint32_t augur_q8_load_u32_le_v1(const uint8_t* source) {
    return static_cast<uint32_t>(source[0])
        | (static_cast<uint32_t>(source[1]) << 8U)
        | (static_cast<uint32_t>(source[2]) << 16U)
        | (static_cast<uint32_t>(source[3]) << 24U);
}

#if defined(AUGUR_CUDA_QWEN35_Q8_PACKED_LOAD)
__device__ __forceinline__ uint32_t augur_q8_load_u32_b2_v1(const uint8_t* source) {
    const auto* source_u16 = reinterpret_cast<const uint16_t*>(source);
    return static_cast<uint32_t>(source_u16[0])
        | (static_cast<uint32_t>(source_u16[1]) << 16U);
}

__device__ __forceinline__ uint32_t augur_q8_load_u32_b4_v1(const uint8_t* source) {
    return *reinterpret_cast<const uint32_t*>(source);
}

__device__ __forceinline__ uint16_t augur_q8_load_u16_b2_v1(const uint8_t* source) {
    return *reinterpret_cast<const uint16_t*>(source);
}
#endif

__global__ void augur_q8_1_activation_quantize_kernel_v1(
    const float* input,
    uint8_t* output,
    uint32_t elements
) {
    const uint32_t warp = threadIdx.x / 32U;
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t activation_block = blockIdx.x * augur_q8_1_matvec_warps_v1 + warp;
    const uint32_t block_count = elements / augur_q8_1_block_elements_v1;
    if (activation_block >= block_count) return;

    const float value = input[activation_block * augur_q8_1_block_elements_v1 + lane];
    float maximum = fabsf(value);
    float sum = value;
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    maximum = __shfl_sync(0xffffffffU, maximum, 0);
    const float scale = maximum / 127.0f;
#if defined(AUGUR_CUDA_QWEN35_DIAGNOSTIC_Q8_AWAY)
    // Explicit research-build identity. Match the local llama Q8_1 division
    // and ties-away rule; no change to KV quantization or default builds.
    int32_t code = maximum != 0.0f ? static_cast<int32_t>(roundf(value / scale)) : 0;
#else
    const float inverse = maximum != 0.0f ? 127.0f / maximum : 0.0f;
    int32_t code = __float2int_rn(value * inverse);
#endif
    code = max(-128, min(127, code));

    uint8_t* destination = output
        + static_cast<uint64_t>(activation_block) * augur_q8_1_block_bytes_v1;
    destination[4U + lane] = static_cast<uint8_t>(static_cast<int8_t>(code));
    if (lane == 0) {
        auto* metadata = reinterpret_cast<uint16_t*>(destination);
        metadata[0] = __half_as_ushort(__float2half_rn(scale));
        metadata[1] = __half_as_ushort(__float2half_rn(sum));
    }
}


__global__ __launch_bounds__(128, 1) void augur_q8_0_q8_1_matvec_kernel_v1(
    const uint8_t* weight,
    const uint8_t* activation,
    float* output,
    uint32_t rows,
    uint32_t cols
) {
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x / 32U;
    const uint32_t thread = threadIdx.x;
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;

    constexpr uint32_t weight_block_bytes = 34U;
    const uint32_t blocks_per_row = cols / augur_q8_1_block_elements_v1;
    const uint64_t row_bytes = static_cast<uint64_t>(blocks_per_row) * weight_block_bytes;
    const uint8_t* row_source = weight + static_cast<uint64_t>(row) * row_bytes;
    float partial = 0.0f;

    const uint32_t segment = thread & 3U;
    for (uint32_t block = thread / 4U; block < blocks_per_row; block += 32U) {
        const uint8_t* weight_block = row_source
            + static_cast<uint64_t>(block) * weight_block_bytes;
        const uint8_t* activation_block = activation
            + static_cast<uint64_t>(block) * augur_q8_1_block_bytes_v1;
        const uint32_t code_offset = segment * 8U;
#if defined(AUGUR_CUDA_QWEN35_Q8_PACKED_LOAD)
        const int32_t weight0 = static_cast<int32_t>(
            augur_q8_load_u32_b2_v1(weight_block + 2U + code_offset));
        const int32_t weight1 = static_cast<int32_t>(
            augur_q8_load_u32_b2_v1(weight_block + 6U + code_offset));
        const int32_t input0 = static_cast<int32_t>(
            augur_q8_load_u32_b4_v1(activation_block + 4U + code_offset));
        const int32_t input1 = static_cast<int32_t>(
            augur_q8_load_u32_b4_v1(activation_block + 8U + code_offset));
#else
        const int32_t weight0 = static_cast<int32_t>(
            augur_q8_load_u32_le_v1(weight_block + 2U + code_offset));
        const int32_t weight1 = static_cast<int32_t>(
            augur_q8_load_u32_le_v1(weight_block + 6U + code_offset));
        const int32_t input0 = static_cast<int32_t>(
            augur_q8_load_u32_le_v1(activation_block + 4U + code_offset));
        const int32_t input1 = static_cast<int32_t>(
            augur_q8_load_u32_le_v1(activation_block + 8U + code_offset));
#endif
        int32_t dot = augur_cuda_dp4a_s8(weight0, input0, 0);
        dot = augur_cuda_dp4a_s8(weight1, input1, dot);
#if defined(AUGUR_CUDA_QWEN35_Q8_PACKED_LOAD)
        const uint16_t weight_scale_bits = augur_q8_load_u16_b2_v1(weight_block);
        const uint16_t activation_scale_bits = augur_q8_load_u16_b2_v1(activation_block);
#else
        const uint16_t weight_scale_bits = static_cast<uint16_t>(weight_block[0])
            | static_cast<uint16_t>(weight_block[1]) << 8U;
        const uint16_t activation_scale_bits = static_cast<uint16_t>(activation_block[0])
            | static_cast<uint16_t>(activation_block[1]) << 8U;
#endif
        const float combined_scale = augur_q8_0_f16_to_f32(weight_scale_bits)
            * augur_q8_0_f16_to_f32(activation_scale_bits);
        partial = fmaf(combined_scale, static_cast<float>(dot), partial);
    }

    __shared__ float other_warps[augur_q8_1_matvec_warps_v1 - 1U][32];
    if (warp > 0U) other_warps[warp - 1U][lane] = partial;
    __syncthreads();
    if (warp > 0U) return;
    #pragma unroll
    for (uint32_t other = 0; other < augur_q8_1_matvec_warps_v1 - 1U; ++other) {
        partial += other_warps[other][lane];
    }
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U) {
        partial += __shfl_down_sync(0xffffffffU, partial, offset);
    }
    if (lane == 0U) output[row] = partial;
}

#if defined(AUGUR_CUDA_QWEN35_FUSED_FFN_SWIGLU)
__global__ __launch_bounds__(128, 1) void augur_q8_0_q8_1_fused_swiglu_kernel_v1(
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    const uint8_t* activation,
    float* output,
    uint32_t rows,
    uint32_t cols
) {
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x / 32U;
    const uint32_t thread = threadIdx.x;
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;

    constexpr uint32_t weight_block_bytes = 34U;
    const uint32_t blocks_per_row = cols / augur_q8_1_block_elements_v1;
    const uint64_t row_bytes = static_cast<uint64_t>(blocks_per_row) * weight_block_bytes;
    const uint8_t* gate_row = gate_weight + static_cast<uint64_t>(row) * row_bytes;
    const uint8_t* up_row = up_weight + static_cast<uint64_t>(row) * row_bytes;
    float gate_partial = 0.0f;
    float up_partial = 0.0f;

    const uint32_t segment = thread & 3U;
    for (uint32_t block = thread / 4U; block < blocks_per_row; block += 32U) {
        const uint8_t* gate_block = gate_row
            + static_cast<uint64_t>(block) * weight_block_bytes;
        const uint8_t* up_block = up_row
            + static_cast<uint64_t>(block) * weight_block_bytes;
        const uint8_t* activation_block = activation
            + static_cast<uint64_t>(block) * augur_q8_1_block_bytes_v1;
        const uint32_t code_offset = segment * 8U;
#if defined(AUGUR_CUDA_QWEN35_Q8_PACKED_LOAD)
        const int32_t input0 = static_cast<int32_t>(
            augur_q8_load_u32_b4_v1(activation_block + 4U + code_offset));
        const int32_t input1 = static_cast<int32_t>(
            augur_q8_load_u32_b4_v1(activation_block + 8U + code_offset));
        const uint16_t activation_scale_bits = augur_q8_load_u16_b2_v1(activation_block);
        const uint16_t gate_scale_bits = augur_q8_load_u16_b2_v1(gate_block);
        const uint16_t up_scale_bits = augur_q8_load_u16_b2_v1(up_block);
        const int32_t gate0 = static_cast<int32_t>(
            augur_q8_load_u32_b2_v1(gate_block + 2U + code_offset));
        const int32_t gate1 = static_cast<int32_t>(
            augur_q8_load_u32_b2_v1(gate_block + 6U + code_offset));
        const int32_t up0 = static_cast<int32_t>(
            augur_q8_load_u32_b2_v1(up_block + 2U + code_offset));
        const int32_t up1 = static_cast<int32_t>(
            augur_q8_load_u32_b2_v1(up_block + 6U + code_offset));
#else
        const int32_t input0 = static_cast<int32_t>(
            augur_q8_load_u32_le_v1(activation_block + 4U + code_offset));
        const int32_t input1 = static_cast<int32_t>(
            augur_q8_load_u32_le_v1(activation_block + 8U + code_offset));
        const uint16_t activation_scale_bits = static_cast<uint16_t>(activation_block[0])
            | static_cast<uint16_t>(activation_block[1]) << 8U;
        const uint16_t gate_scale_bits = static_cast<uint16_t>(gate_block[0])
            | static_cast<uint16_t>(gate_block[1]) << 8U;
        const uint16_t up_scale_bits = static_cast<uint16_t>(up_block[0])
            | static_cast<uint16_t>(up_block[1]) << 8U;
        const int32_t gate0 = static_cast<int32_t>(
            augur_q8_load_u32_le_v1(gate_block + 2U + code_offset));
        const int32_t gate1 = static_cast<int32_t>(
            augur_q8_load_u32_le_v1(gate_block + 6U + code_offset));
        const int32_t up0 = static_cast<int32_t>(
            augur_q8_load_u32_le_v1(up_block + 2U + code_offset));
        const int32_t up1 = static_cast<int32_t>(
            augur_q8_load_u32_le_v1(up_block + 6U + code_offset));
#endif
        int32_t gate_dot = augur_cuda_dp4a_s8(gate0, input0, 0);
        gate_dot = augur_cuda_dp4a_s8(gate1, input1, gate_dot);
        int32_t up_dot = augur_cuda_dp4a_s8(up0, input0, 0);
        up_dot = augur_cuda_dp4a_s8(up1, input1, up_dot);
        const float activation_scale = augur_q8_0_f16_to_f32(activation_scale_bits);
        gate_partial = fmaf(
            augur_q8_0_f16_to_f32(gate_scale_bits) * activation_scale,
            static_cast<float>(gate_dot),
            gate_partial);
        up_partial = fmaf(
            augur_q8_0_f16_to_f32(up_scale_bits) * activation_scale,
            static_cast<float>(up_dot),
            up_partial);
    }

    __shared__ float gate_other_warps[augur_q8_1_matvec_warps_v1 - 1U][32];
    __shared__ float up_other_warps[augur_q8_1_matvec_warps_v1 - 1U][32];
    if (warp > 0U) {
        gate_other_warps[warp - 1U][lane] = gate_partial;
        up_other_warps[warp - 1U][lane] = up_partial;
    }
    __syncthreads();
    if (warp > 0U) return;
    #pragma unroll
    for (uint32_t other = 0; other < augur_q8_1_matvec_warps_v1 - 1U; ++other) {
        gate_partial += gate_other_warps[other][lane];
        up_partial += up_other_warps[other][lane];
    }
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
        gate_partial += __shfl_down_sync(0xffffffffU, gate_partial, offset);
        up_partial += __shfl_down_sync(0xffffffffU, up_partial, offset);
    }
    if (lane == 0U) {
        float sigmoid;
        if (gate_partial >= 0.0f) {
            sigmoid = 1.0f / (1.0f + expf(-gate_partial));
        } else {
            const float exponent = expf(gate_partial);
            sigmoid = exponent / (1.0f + exponent);
        }
        output[row] = (gate_partial * sigmoid) * up_partial;
    }
}
#endif

bool augur_q8_1_activation_footprint_v1(
    uint32_t elements,
    uint64_t* input_bytes,
    uint64_t* output_bytes
) {
    if (input_bytes == nullptr || output_bytes == nullptr || elements == 0
        || (elements % augur_q8_1_block_elements_v1) != 0) {
        return false;
    }
    const uint64_t block_count = static_cast<uint64_t>(elements)
        / augur_q8_1_block_elements_v1;
    if (static_cast<uint64_t>(elements) > UINT64_MAX / sizeof(float)
        || block_count > UINT64_MAX / augur_q8_1_block_bytes_v1) {
        return false;
    }
    *input_bytes = static_cast<uint64_t>(elements) * sizeof(float);
    *output_bytes = block_count * augur_q8_1_block_bytes_v1;
    return *input_bytes <= SIZE_MAX && *output_bytes <= SIZE_MAX;
}

bool augur_q8_0_q8_1_matvec_footprint_v1(
    uint32_t rows,
    uint32_t cols,
    uint64_t* weight_bytes,
    uint64_t* input_bytes,
    uint64_t* output_bytes
) {
    if (!augur_q8_0_checked_footprint(rows, cols, weight_bytes, input_bytes, output_bytes)) {
        return false;
    }
    const uint64_t activation_blocks = static_cast<uint64_t>(cols)
        / augur_q8_1_block_elements_v1;
    if (activation_blocks > UINT64_MAX / augur_q8_1_block_bytes_v1) return false;
    *input_bytes = activation_blocks * augur_q8_1_block_bytes_v1;
    return *input_bytes <= SIZE_MAX;
}

} // namespace

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q8_1_activation_quantize_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q8_1_activation_quantize_args_v1* args
) {
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    const bool footprint_ok = args != nullptr && augur_q8_1_activation_footprint_v1(
        args->elements, &input_bytes, &output_bytes);
    if (stream == nullptr || stream->stream == nullptr || args == nullptr || ordinal < 0
        || stream->ordinal != ordinal || !footprint_ok
        || args->abi_version != AUGUR_CUDA_Q8_1_ACTIVATION_QUANTIZE_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_q8_1_activation_quantize_args_v1)
        || args->reserved != 0 || args->input_len_bytes < input_bytes
        || args->output_len_bytes != output_bytes
        || !augur_q8_0_checked_range(args->input_f32, input_bytes)
        || !augur_q8_0_checked_range(args->output_q8_1, output_bytes)
        || (args->input_f32 % alignof(float)) != 0
        || (args->output_q8_1 % alignof(uint32_t)) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    const uint32_t activation_blocks = args->elements / augur_q8_1_block_elements_v1;
    const uint32_t grid = (activation_blocks + augur_q8_1_matvec_warps_v1 - 1U)
        / augur_q8_1_matvec_warps_v1;
    augur_q8_1_activation_quantize_kernel_v1<<<
        grid, augur_q8_1_matvec_warps_v1 * 32U, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input_f32)),
        reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(args->output_q8_1)),
        args->elements);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q8_1_activation_quantize_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q8_1_activation_quantize_args_v1* args
) {
    const augur_cuda_result_v1 submitted =
        augur_cuda_q8_1_activation_quantize_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q8_0_q8_1_matvec_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q8_0_q8_1_matvec_args_v1* args
) {
    uint64_t weight_bytes = 0;
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    const bool footprint_ok = args != nullptr && augur_q8_0_q8_1_matvec_footprint_v1(
        args->rows, args->cols, &weight_bytes, &input_bytes, &output_bytes);
    if (stream == nullptr || stream->stream == nullptr || args == nullptr || ordinal < 0
        || stream->ordinal != ordinal || !footprint_ok
        || args->abi_version != AUGUR_CUDA_Q8_0_Q8_1_MATVEC_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_q8_0_q8_1_matvec_args_v1)
        || args->weight_len_bytes != weight_bytes || args->input_len_bytes != input_bytes
        || args->output_len_bytes < output_bytes
        || !augur_q8_0_checked_range(args->weight, weight_bytes)
        || !augur_q8_0_checked_range(args->input_q8_1, input_bytes)
        || !augur_q8_0_checked_range(args->output, output_bytes)
        || (args->weight % alignof(uint16_t)) != 0
        || (args->input_q8_1 % alignof(uint32_t)) != 0
        || (args->output % alignof(float)) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    augur_q8_0_q8_1_matvec_kernel_v1<<<args->rows, 128U, 0, stream->stream>>>(
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->input_q8_1)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->rows,
        args->cols);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q8_0_q8_1_matvec_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q8_0_q8_1_matvec_args_v1* args
) {
    const augur_cuda_result_v1 submitted =
        augur_cuda_q8_0_q8_1_matvec_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

#if defined(AUGUR_CUDA_QWEN35_FUSED_FFN_SWIGLU)
static augur_cuda_result_v1 augur_validate_q8_0_q8_1_fused_swiglu_args_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q8_0_q8_1_fused_swiglu_args_v1* args
) {
    uint64_t weight_bytes = 0;
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    const bool footprint_ok = args != nullptr && augur_q8_0_q8_1_matvec_footprint_v1(
        args->rows, args->cols, &weight_bytes, &input_bytes, &output_bytes);
    if (stream == nullptr || stream->stream == nullptr || args == nullptr || ordinal < 0
        || stream->ordinal != ordinal || !footprint_ok
        || args->abi_version != AUGUR_CUDA_Q8_0_Q8_1_FUSED_SWIGLU_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_q8_0_q8_1_fused_swiglu_args_v1)
        || args->gate_weight_len_bytes != weight_bytes
        || args->up_weight_len_bytes != weight_bytes
        || args->input_len_bytes != input_bytes
        || args->output_len_bytes < output_bytes
        || !augur_q8_0_checked_range(args->gate_weight, weight_bytes)
        || !augur_q8_0_checked_range(args->up_weight, weight_bytes)
        || !augur_q8_0_checked_range(args->input_q8_1, input_bytes)
        || !augur_q8_0_checked_range(args->output, output_bytes)
        || (args->gate_weight % alignof(uint16_t)) != 0
        || (args->up_weight % alignof(uint16_t)) != 0
        || (args->input_q8_1 % alignof(uint32_t)) != 0
        || (args->output % alignof(float)) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t addresses[] = {
        args->gate_weight, args->up_weight, args->input_q8_1, args->output};
    const uint64_t lengths[] = {
        weight_bytes, weight_bytes, input_bytes, output_bytes};
    for (uint32_t left = 0; left < 4U; ++left) {
        for (uint32_t right = left + 1U; right < 4U; ++right) {
            if (!augur_q8_0_ranges_disjoint(
                    addresses[left], lengths[left], addresses[right], lengths[right])) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    return augur_set_device(ordinal);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q8_0_q8_1_fused_swiglu_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q8_0_q8_1_fused_swiglu_args_v1* args
) {
    const augur_cuda_result_v1 valid =
        augur_validate_q8_0_q8_1_fused_swiglu_args_v1(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    augur_q8_0_q8_1_fused_swiglu_kernel_v1<<<args->rows, 128U, 0, stream->stream>>>(
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->gate_weight)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->up_weight)),
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->input_q8_1)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->rows,
        args->cols);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_q8_0_q8_1_fused_swiglu_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_q8_0_q8_1_fused_swiglu_args_v1* args
) {
    const augur_cuda_result_v1 submitted =
        augur_cuda_q8_0_q8_1_fused_swiglu_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}
#endif

__global__ void augur_bf16_token_row_gather_kernel_v1(
    const uint16_t* weight,
    const uint32_t* token_source,
    float* output,
    uint32_t rows,
    uint32_t cols
) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    const uint32_t token = token_source[0];
    if (token >= rows) {
        output[col] = __int_as_float(0x7fc00000);
        return;
    }
    const uint16_t value = weight[static_cast<size_t>(token) * cols + col];
    output[col] = __uint_as_float(static_cast<uint32_t>(value) << 16);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_token_row_gather_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_bf16_token_row_gather_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_BF16_TOKEN_ROW_GATHER_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_bf16_token_row_gather_args_v1)
        || args->weight == 0 || args->token_source_u32 == 0 || args->output_f32 == 0
        || args->rows == 0 || args->cols == 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t weight_bytes = static_cast<uint64_t>(args->rows) * args->cols * 2;
    const uint64_t output_bytes = static_cast<uint64_t>(args->cols) * 4;
    if (args->weight_len_bytes != weight_bytes
        || args->token_source_len_bytes < sizeof(uint32_t)
        || args->output_len_bytes != output_bytes
        || args->weight % alignof(uint16_t) != 0
        || args->token_source_u32 % alignof(uint32_t) != 0
        || args->output_f32 % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    constexpr uint32_t threads = 256;
    const uint32_t blocks = (args->cols + threads - 1) / threads;
    augur_bf16_token_row_gather_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(args->token_source_u32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        args->rows,
        args->cols);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

// Same bit conversion and invalid-ID poison as the scalar operation. Rows
// are independent; grid.y changes scheduling only, not floating-point math.
__global__ void augur_bf16_token_rows_gather_kernel_v1(
    const uint16_t* weight, const uint32_t* ids, float* output,
    uint32_t vocabulary, uint32_t cols
) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    const uint32_t token = ids[blockIdx.y];
    const size_t at = static_cast<size_t>(blockIdx.y) * cols + col;
    output[at] = token >= vocabulary ? __int_as_float(0x7fc00000)
        : __uint_as_float(static_cast<uint32_t>(weight[static_cast<size_t>(token) * cols + col]) << 16);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_token_rows_gather_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_bf16_token_row_gather_args_v1* args, uint32_t tokens
) {
    if (!stream || !args || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_BF16_TOKEN_ROW_GATHER_ARGS_VERSION_V1
        || args->struct_size != sizeof(*args) || !args->rows || !args->cols
        || tokens == 0 || tokens > 512
        || static_cast<uint64_t>(args->rows) * args->cols > UINT64_MAX / 2) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t sizes[] = {static_cast<uint64_t>(args->rows) * args->cols * 2,
        static_cast<uint64_t>(tokens) * 4, static_cast<uint64_t>(tokens) * args->cols * 4};
    const uint64_t addresses[] = {args->weight, args->token_source_u32, args->output_f32};
    const uint64_t lengths[] = {args->weight_len_bytes, args->token_source_len_bytes, args->output_len_bytes};
    for (int i = 0; i < 3; ++i) {
        if (!addresses[i] || addresses[i] > UINT64_MAX - sizes[i]
            || lengths[i] != sizes[i] || addresses[i] % (i == 0 ? 2 : 4) != 0) {
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (addresses[2] < addresses[i] + sizes[i] && addresses[i] < addresses[2] + sizes[2]) {
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
    }
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    const dim3 grid((args->cols - 1) / 256 + 1, tokens);
    augur_bf16_token_rows_gather_kernel_v1<<<grid, 256, 0, stream->stream>>>(
        reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(args->token_source_u32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)), args->rows, args->cols);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_token_row_gather_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_bf16_token_row_gather_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_bf16_token_row_gather_submit_v1(
        ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

static __device__ __forceinline__ float augur_m95_softplus(float value) {
    if (value > 20.0f) return value;
    if (value < -20.0f) return expf(value);
    return log1pf(expf(value));
}

__global__ void augur_m95_router_gather_kernel_v1(
    const float* logits,
    const float* bias,
    uint16_t* output_ids,
    float* output_weights,
    uint32_t* counts,
    uint32_t* offsets,
    uint16_t* gathered_tokens,
    float* gathered_weights,
    int32_t* status,
    uint32_t tokens,
    float routed_scale
) {
    const uint32_t token = static_cast<uint32_t>(threadIdx.x);
    if (token == 0) {
        status[0] = 1;
        for (uint32_t expert = 0; expert < 256; ++expert) counts[expert] = 0;
    }
    __syncthreads();

    if (token < tokens) {
        uint16_t selected_ids[6] = {};
        float denominator = 0.0f;
        for (uint32_t rank = 0; rank < 6; ++rank) {
            int best = -1;
            float best_choice = -INFINITY;
            for (int expert = 0; expert < 256; ++expert) {
                bool selected = false;
                for (uint32_t prior = 0; prior < rank; ++prior) {
                    selected = selected || selected_ids[prior] == expert;
                }
                if (selected) continue;
                const float logit = logits[token * 256 + expert];
                const float correction = bias[expert];
                if (!isfinite(logit) || !isfinite(correction)) {
                    atomicExch(status, 0);
                    continue;
                }
                const float score = sqrtf(augur_m95_softplus(logit));
                const float choice = score + correction;
                if (!isfinite(score) || !isfinite(choice)) {
                    atomicExch(status, 0);
                    continue;
                }
                if (best < 0 || choice > best_choice
                    || (choice == best_choice && expert < best)) {
                    best = expert;
                    best_choice = choice;
                }
            }
            if (best < 0) {
                atomicExch(status, 0);
                best = 0;
            }
            selected_ids[rank] = static_cast<uint16_t>(best);
            output_ids[token * 6 + rank] = static_cast<uint16_t>(best);
            denominator += sqrtf(augur_m95_softplus(logits[token * 256 + best]));
        }
        if (!isfinite(denominator) || denominator <= 0.0f) {
            atomicExch(status, 0);
            denominator = 1.0f;
        }
        for (uint32_t rank = 0; rank < 6; ++rank) {
            const uint16_t expert = selected_ids[rank];
            const float score = sqrtf(augur_m95_softplus(logits[token * 256 + expert]));
            const float weight = score / denominator * routed_scale;
            output_weights[token * 6 + rank] = weight;
            if (!isfinite(weight)) atomicExch(status, 0);
        }
    }
    __syncthreads();

    if (token == 0) {
        for (uint32_t row = 0; row < tokens; ++row) {
            for (uint32_t rank = 0; rank < 6; ++rank) {
                ++counts[output_ids[row * 6 + rank]];
            }
        }
        offsets[0] = 0;
        for (uint32_t expert = 0; expert < 256; ++expert) {
            offsets[expert + 1] = offsets[expert] + counts[expert];
        }
        uint32_t cursor[256];
        for (uint32_t expert = 0; expert < 256; ++expert) cursor[expert] = offsets[expert];
        for (uint32_t row = 0; row < tokens; ++row) {
            for (uint32_t rank = 0; rank < 6; ++rank) {
                const uint32_t source = row * 6 + rank;
                const uint16_t expert = output_ids[source];
                const uint32_t destination = cursor[expert]++;
                gathered_tokens[destination] = static_cast<uint16_t>(row);
                gathered_weights[destination] = output_weights[source];
            }
        }
        if (offsets[256] != tokens * 6) status[0] = 0;
    }
}

static augur_cuda_result_v1 augur_validate_m95_router_gather(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m95_router_gather_args_v1* args
) {
    augur_cuda_result_v1 device = augur_validate_stream_device(ordinal, stream);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    if (args == nullptr
        || args->abi_version != AUGUR_CUDA_M95_ROUTER_GATHER_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_m95_router_gather_args_v1)
        || args->logits_f32 == 0 || args->bias_f32 == 0
        || args->output_ids_u16 == 0 || args->output_weights_f32 == 0
        || args->counts_u32 == 0 || args->offsets_u32 == 0
        || args->gathered_tokens_u16 == 0 || args->gathered_weights_f32 == 0
        || args->status_i32 == 0 || args->tokens == 0 || args->tokens > 128
        || args->experts != 256 || args->top_k != 6
        || !isfinite(args->routed_scale) || args->routed_scale <= 0.0f) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t routes = static_cast<uint64_t>(args->tokens) * 6;
    if (args->logits_len_bytes != static_cast<uint64_t>(args->tokens) * 256 * 4
        || args->bias_len_bytes != 256 * 4
        || args->output_ids_len_bytes != routes * 2
        || args->output_weights_len_bytes != routes * 4
        || args->counts_len_bytes != 256 * 4
        || args->offsets_len_bytes != 257 * 4
        || args->gathered_tokens_len_bytes != routes * 2
        || args->gathered_weights_len_bytes != routes * 4
        || args->status_len_bytes != 4
        || args->logits_f32 % alignof(float) != 0
        || args->bias_f32 % alignof(float) != 0
        || args->output_ids_u16 % alignof(uint16_t) != 0
        || args->output_weights_f32 % alignof(float) != 0
        || args->counts_u32 % alignof(uint32_t) != 0
        || args->offsets_u32 % alignof(uint32_t) != 0
        || args->gathered_tokens_u16 % alignof(uint16_t) != 0
        || args->gathered_weights_f32 % alignof(float) != 0
        || args->status_i32 % alignof(int32_t) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m95_router_gather_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m95_router_gather_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_m95_router_gather(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    augur_m95_router_gather_kernel_v1<<<1, 128, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->logits_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->bias_f32)),
        reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(args->output_ids_u16)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_weights_f32)),
        reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(args->counts_u32)),
        reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(args->offsets_u32)),
        reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(args->gathered_tokens_u16)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->gathered_weights_f32)),
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(args->status_i32)),
        args->tokens,
        args->routed_scale);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m95_router_gather_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m95_router_gather_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_m95_router_gather_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

__global__ void augur_m102_decode_router_kernel_v1(
    const uint16_t* weight,
    const float* input,
    const float* bias,
    float* logits,
    uint16_t* output_ids,
    float* output_weights,
    uint32_t* counts,
    uint32_t* offsets,
    uint16_t* gathered_tokens,
    float* gathered_weights,
    int32_t* status,
    float routed_scale
) {
    __shared__ float scores[256];
    __shared__ float choices[256];
    const uint32_t expert = static_cast<uint32_t>(threadIdx.x);
    if (expert == 0) status[0] = 1;
    __syncthreads();
    float sum = 0.0f;
    const uint16_t* row = weight + static_cast<size_t>(expert) * 4096;
    for (uint32_t column = 0; column < 4096; ++column) {
        sum += __uint_as_float(static_cast<uint32_t>(row[column]) << 16) * input[column];
    }
    logits[expert] = sum;
    const float score = sqrtf(augur_m95_softplus(sum));
    scores[expert] = score;
    choices[expert] = score + bias[expert];
    if (!isfinite(sum) || !isfinite(score) || !isfinite(bias[expert])
        || !isfinite(choices[expert])) {
        atomicExch(status, 0);
    }
    __syncthreads();
    if (expert != 0) return;

    bool selected[256] = {};
    float denominator = 0.0f;
    for (uint32_t rank = 0; rank < 6; ++rank) {
        int best = -1;
        for (int candidate = 0; candidate < 256; ++candidate) {
            if (!selected[candidate]
                && (best < 0 || choices[candidate] > choices[best]
                    || (choices[candidate] == choices[best] && candidate < best))) {
                best = candidate;
            }
        }
        if (best < 0) {
            status[0] = 0;
            best = 0;
        }
        selected[best] = true;
        output_ids[rank] = static_cast<uint16_t>(best);
        denominator += scores[best];
    }
    if (!isfinite(denominator) || denominator <= 0.0f) {
        status[0] = 0;
        denominator = 1.0f;
    }
    for (uint32_t rank = 0; rank < 6; ++rank) {
        const float value = scores[output_ids[rank]] / denominator * routed_scale;
        output_weights[rank] = value;
        if (!isfinite(value)) status[0] = 0;
    }
    for (uint32_t index = 0; index < 256; ++index) counts[index] = 0;
    for (uint32_t rank = 0; rank < 6; ++rank) ++counts[output_ids[rank]];
    offsets[0] = 0;
    for (uint32_t index = 0; index < 256; ++index) offsets[index + 1] = offsets[index] + counts[index];
    uint32_t cursor[256];
    for (uint32_t index = 0; index < 256; ++index) cursor[index] = offsets[index];
    for (uint32_t rank = 0; rank < 6; ++rank) {
        const uint16_t selected_expert = output_ids[rank];
        const uint32_t destination = cursor[selected_expert]++;
        gathered_tokens[destination] = 0;
        gathered_weights[destination] = output_weights[rank];
    }
    if (offsets[256] != 6) status[0] = 0;
}

static augur_cuda_result_v1 augur_validate_m102_decode_router(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m102_decode_router_args_v1* args
) {
    augur_cuda_result_v1 device = augur_validate_stream_device(ordinal, stream);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    if (args == nullptr
        || args->abi_version != AUGUR_CUDA_M102_DECODE_ROUTER_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_m102_decode_router_args_v1)
        || args->weight_bf16 == 0 || args->input_f32 == 0 || args->bias_f32 == 0
        || args->logits_f32 == 0 || args->output_ids_u16 == 0
        || args->output_weights_f32 == 0 || args->counts_u32 == 0
        || args->offsets_u32 == 0 || args->gathered_tokens_u16 == 0
        || args->gathered_weights_f32 == 0 || args->status_i32 == 0
        || args->rows != 256 || args->cols != 4096
        || args->experts != 256 || args->top_k != 6
        || !isfinite(args->routed_scale) || args->routed_scale <= 0.0f) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->weight_len_bytes != UINT64_C(256) * 4096 * 2
        || args->input_len_bytes != 4096 * 4
        || args->bias_len_bytes != 256 * 4 || args->logits_len_bytes != 256 * 4
        || args->output_ids_len_bytes != 6 * 2 || args->output_weights_len_bytes != 6 * 4
        || args->counts_len_bytes != 256 * 4 || args->offsets_len_bytes != 257 * 4
        || args->gathered_tokens_len_bytes != 6 * 2 || args->gathered_weights_len_bytes != 6 * 4
        || args->status_len_bytes != 4 || args->weight_bf16 % alignof(uint16_t) != 0
        || args->input_f32 % alignof(float) != 0 || args->bias_f32 % alignof(float) != 0
        || args->logits_f32 % alignof(float) != 0 || args->output_ids_u16 % alignof(uint16_t) != 0
        || args->output_weights_f32 % alignof(float) != 0 || args->counts_u32 % alignof(uint32_t) != 0
        || args->offsets_u32 % alignof(uint32_t) != 0 || args->gathered_tokens_u16 % alignof(uint16_t) != 0
        || args->gathered_weights_f32 % alignof(float) != 0 || args->status_i32 % alignof(int32_t) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m102_decode_router_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m102_decode_router_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_m102_decode_router(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    augur_m102_decode_router_kernel_v1<<<1, 256, 0, stream->stream>>>(
        reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(args->weight_bf16)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->bias_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->logits_f32)),
        reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(args->output_ids_u16)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_weights_f32)),
        reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(args->counts_u32)),
        reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(args->offsets_u32)),
        reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(args->gathered_tokens_u16)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->gathered_weights_f32)),
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(args->status_i32)),
        args->routed_scale);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m102_decode_router_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m102_decode_router_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_m102_decode_router_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

__global__ void augur_m204_hash_router_kernel_v1(
    const uint16_t* weight,
    const float* input,
    const int64_t* tid2eid,
    const int64_t* input_ids,
    float* logits,
    uint16_t* output_ids,
    float* output_weights,
    int32_t* status,
    float routed_scale
) {
    __shared__ float scores[256];
    const uint32_t token = static_cast<uint32_t>(blockIdx.x);
    const uint32_t expert = static_cast<uint32_t>(threadIdx.x);
    if (expert == 0) status[token] = 1;
    __syncthreads();

    float sum = 0.0f;
    const uint16_t* row = weight + static_cast<size_t>(expert) * 4096;
    const float* token_input = input + static_cast<size_t>(token) * 4096;
    for (uint32_t column = 0; column < 4096; ++column) {
        sum += __uint_as_float(static_cast<uint32_t>(row[column]) << 16) * token_input[column];
    }
    const float score = sqrtf(augur_m95_softplus(sum));
    logits[static_cast<size_t>(token) * 256 + expert] = sum;
    scores[expert] = score;
    if (!isfinite(sum) || !isfinite(score)) atomicExch(status + token, 0);
    __syncthreads();
    if (expert != 0) return;

    const int64_t token_id = input_ids[token];
    if (token_id < 0 || token_id >= 129280) {
        status[token] = 0;
        return;
    }
    const int64_t* ids = tid2eid + static_cast<size_t>(token_id) * 6;
    float denominator = 0.0f;
    for (uint32_t rank = 0; rank < 6; ++rank) {
        const int64_t id = ids[rank];
        if (id < 0 || id >= 256) {
            status[token] = 0;
            return;
        }
        for (uint32_t previous = 0; previous < rank; ++previous) {
            if (ids[previous] == id) {
                status[token] = 0;
                return;
            }
        }
        output_ids[static_cast<size_t>(token) * 6 + rank] = static_cast<uint16_t>(id);
        denominator += scores[id];
    }
    if (!isfinite(denominator) || denominator <= 0.0f) {
        status[token] = 0;
        return;
    }
    for (uint32_t rank = 0; rank < 6; ++rank) {
        const uint16_t id = output_ids[static_cast<size_t>(token) * 6 + rank];
        const float value = scores[id] / denominator * routed_scale;
        output_weights[static_cast<size_t>(token) * 6 + rank] = value;
        if (!isfinite(value)) status[token] = 0;
    }
}

static augur_cuda_result_v1 augur_validate_m204_hash_router(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m204_hash_router_args_v1* args
) {
    augur_cuda_result_v1 device = augur_validate_stream_device(ordinal, stream);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    if (args == nullptr
        || args->abi_version != AUGUR_CUDA_M204_HASH_ROUTER_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_m204_hash_router_args_v1)
        || args->weight_bf16 == 0 || args->input_f32 == 0 || args->tid2eid_i64 == 0
        || args->input_ids_i64 == 0 || args->logits_f32 == 0
        || args->output_ids_u16 == 0 || args->output_weights_f32 == 0
        || args->status_i32 == 0 || args->tokens == 0 || args->tokens > 8 || args->hidden != 4096
        || args->experts != 256 || args->top_k != 6 || args->table_rows != 129280
        || !isfinite(args->routed_scale) || args->routed_scale <= 0.0f) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->weight_len_bytes != UINT64_C(256) * 4096 * 2
        || args->input_len_bytes != static_cast<uint64_t>(args->tokens) * 4096 * 4
        || args->tid2eid_len_bytes != UINT64_C(129280) * 6 * 8
        || args->input_ids_len_bytes != static_cast<uint64_t>(args->tokens) * 8
        || args->logits_len_bytes != static_cast<uint64_t>(args->tokens) * 256 * 4
        || args->output_ids_len_bytes != static_cast<uint64_t>(args->tokens) * 6 * 2
        || args->output_weights_len_bytes != static_cast<uint64_t>(args->tokens) * 6 * 4
        || args->status_len_bytes != static_cast<uint64_t>(args->tokens) * 4
        || args->weight_bf16 % alignof(uint16_t) != 0
        || args->input_f32 % alignof(float) != 0
        || args->tid2eid_i64 % alignof(int64_t) != 0
        || args->input_ids_i64 % alignof(int64_t) != 0
        || args->logits_f32 % alignof(float) != 0
        || args->output_ids_u16 % alignof(uint16_t) != 0
        || args->output_weights_f32 % alignof(float) != 0
        || args->status_i32 % alignof(int32_t) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m204_hash_router_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m204_hash_router_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_m204_hash_router(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    augur_m204_hash_router_kernel_v1<<<args->tokens, 256, 0, stream->stream>>>(
        reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(args->weight_bf16)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input_f32)),
        reinterpret_cast<const int64_t*>(static_cast<uintptr_t>(args->tid2eid_i64)),
        reinterpret_cast<const int64_t*>(static_cast<uintptr_t>(args->input_ids_i64)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->logits_f32)),
        reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(args->output_ids_u16)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_weights_f32)),
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(args->status_i32)),
        args->routed_scale);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m204_hash_router_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m204_hash_router_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_m204_hash_router_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

extern "C" __global__ void augur_m306_k38_hash_router_selected6_compute_v1(
    const uint16_t* weight,
    const float* input,
    const int64_t* tid2eid,
    const int64_t* input_ids,
    float* selected_logits,
    float* selected_scores,
    uint16_t* selected_ids,
    int32_t* rank_status
) {
    if (threadIdx.x != 0) return;
    const uint32_t token = static_cast<uint32_t>(blockIdx.x) / 6;
    const uint32_t rank = static_cast<uint32_t>(blockIdx.x) % 6;
    const size_t slot = static_cast<size_t>(token) * 6 + rank;
    const int64_t token_id = input_ids[token];
    if (token_id < 0 || token_id >= 129280) {
        rank_status[slot] = 0;
        return;
    }
    const int64_t expert = tid2eid[static_cast<size_t>(token_id) * 6 + rank];
    if (expert < 0 || expert >= 256) {
        rank_status[slot] = 0;
        return;
    }
    float sum = 0.0f;
    const uint16_t* row = weight + static_cast<size_t>(expert) * 4096;
    const float* token_input = input + static_cast<size_t>(token) * 4096;
    for (uint32_t column = 0; column < 4096; ++column) {
        sum += __uint_as_float(static_cast<uint32_t>(row[column]) << 16) * token_input[column];
    }
    const float score = sqrtf(augur_m95_softplus(sum));
    selected_ids[slot] = static_cast<uint16_t>(expert);
    selected_logits[slot] = sum;
    selected_scores[slot] = score;
    rank_status[slot] = isfinite(sum) && isfinite(score) ? 1 : 0;
}

extern "C" __global__ void augur_m306_k38_hash_router_selected6_finalize_v1(
    const float* selected_scores,
    const uint16_t* selected_ids,
    const int32_t* rank_status,
    uint16_t* output_ids,
    float* output_weights,
    int32_t* status,
    float routed_scale
) {
    if (threadIdx.x != 0) return;
    const uint32_t token = static_cast<uint32_t>(blockIdx.x);
    const size_t base = static_cast<size_t>(token) * 6;
    status[token] = 1;
    float denominator = 0.0f;
    for (uint32_t rank = 0; rank < 6; ++rank) {
        const uint16_t id = selected_ids[base + rank];
        if (rank_status[base + rank] != 1 || id >= 256) {
            status[token] = 0;
            return;
        }
        for (uint32_t previous = 0; previous < rank; ++previous) {
            if (selected_ids[base + previous] == id) {
                status[token] = 0;
                return;
            }
        }
        output_ids[base + rank] = id;
        denominator += selected_scores[base + rank];
    }
    if (!isfinite(denominator) || denominator <= 0.0f) {
        status[token] = 0;
        return;
    }
    for (uint32_t rank = 0; rank < 6; ++rank) {
        const float value = selected_scores[base + rank] / denominator * routed_scale;
        output_weights[base + rank] = value;
        if (!isfinite(value)) status[token] = 0;
    }
}

static bool augur_m306_k38_ranges_overlap(
    augur_cuda_device_ptr_t left,
    uint64_t left_len,
    augur_cuda_device_ptr_t right,
    uint64_t right_len
) {
    if (left > UINT64_MAX - left_len || right > UINT64_MAX - right_len) return true;
    return left < right + right_len && right < left + left_len;
}

static augur_cuda_result_v1 augur_validate_m306_k38_hash_router_selected6(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k38_hash_router_selected6_args_v1* args
) {
    augur_cuda_result_v1 device = augur_validate_stream_device(ordinal, stream);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    if (args == nullptr
        || args->abi_version != AUGUR_CUDA_M306_K38_HASH_ROUTER_SELECTED6_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_m306_k38_hash_router_selected6_args_v1)
        || args->weight_bf16 == 0 || args->input_f32 == 0 || args->tid2eid_i64 == 0
        || args->input_ids_i64 == 0 || args->scratch == 0 || args->output_ids_u16 == 0
        || args->output_weights_f32 == 0 || args->status_i32 == 0
        || args->tokens == 0 || args->tokens > 8 || args->hidden != 4096
        || args->experts != 256 || args->top_k != 6 || args->table_rows != 129280
        || !isfinite(args->routed_scale) || args->routed_scale <= 0.0f
        || args->reserved != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->weight_len_bytes != UINT64_C(256) * 4096 * 2
        || args->input_len_bytes != static_cast<uint64_t>(args->tokens) * 4096 * 4
        || args->tid2eid_len_bytes != UINT64_C(129280) * 6 * 8
        || args->input_ids_len_bytes != static_cast<uint64_t>(args->tokens) * 8
        || args->scratch_len_bytes != static_cast<uint64_t>(args->tokens) * 6 * 14
        || args->output_ids_len_bytes != static_cast<uint64_t>(args->tokens) * 6 * 2
        || args->output_weights_len_bytes != static_cast<uint64_t>(args->tokens) * 6 * 4
        || args->status_len_bytes != static_cast<uint64_t>(args->tokens) * 4
        || args->weight_bf16 % alignof(uint16_t) != 0
        || args->input_f32 % alignof(float) != 0
        || args->tid2eid_i64 % alignof(int64_t) != 0
        || args->input_ids_i64 % alignof(int64_t) != 0
        || args->scratch % alignof(float) != 0
        || args->output_ids_u16 % alignof(uint16_t) != 0
        || args->output_weights_f32 % alignof(float) != 0
        || args->status_i32 % alignof(int32_t) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const augur_cuda_device_ptr_t pointers[] = {
        args->weight_bf16, args->input_f32, args->tid2eid_i64, args->input_ids_i64,
        args->scratch, args->output_ids_u16, args->output_weights_f32, args->status_i32,
    };
    const uint64_t lengths[] = {
        args->weight_len_bytes, args->input_len_bytes, args->tid2eid_len_bytes,
        args->input_ids_len_bytes, args->scratch_len_bytes, args->output_ids_len_bytes,
        args->output_weights_len_bytes, args->status_len_bytes,
    };
    for (uint32_t left = 0; left < 8; ++left) {
        for (uint32_t right = left + 1; right < 8; ++right) {
            if (augur_m306_k38_ranges_overlap(
                    pointers[left], lengths[left], pointers[right], lengths[right])) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k38_hash_router_selected6_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k38_hash_router_selected6_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_m306_k38_hash_router_selected6(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    uint8_t* scratch = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(args->scratch));
    const size_t slots = static_cast<size_t>(args->tokens) * 6;
    float* selected_logits = reinterpret_cast<float*>(scratch);
    float* selected_scores = reinterpret_cast<float*>(scratch + slots * sizeof(float));
    uint16_t* selected_ids = reinterpret_cast<uint16_t*>(scratch + slots * sizeof(float) * 2);
    int32_t* rank_status = reinterpret_cast<int32_t*>(scratch + slots * (sizeof(float) * 2 + sizeof(uint16_t)));
    augur_m306_k38_hash_router_selected6_compute_v1<<<args->tokens * 6, 32, 0, stream->stream>>>(
        reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(args->weight_bf16)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input_f32)),
        reinterpret_cast<const int64_t*>(static_cast<uintptr_t>(args->tid2eid_i64)),
        reinterpret_cast<const int64_t*>(static_cast<uintptr_t>(args->input_ids_i64)),
        selected_logits, selected_scores, selected_ids, rank_status);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    augur_m306_k38_hash_router_selected6_finalize_v1<<<args->tokens, 32, 0, stream->stream>>>(
        selected_scores, selected_ids, rank_status,
        reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(args->output_ids_u16)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_weights_f32)),
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(args->status_i32)),
        args->routed_scale);
    error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m306_k38_hash_router_selected6_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m306_k38_hash_router_selected6_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_m306_k38_hash_router_selected6_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

__global__ void augur_m211_learned_router_kernel_v1(
    const uint16_t* weight,
    const float* input,
    const float* bias,
    float* logits,
    uint16_t* output_ids,
    float* output_weights,
    int32_t* status,
    float routed_scale
) {
    __shared__ float scores[256];
    __shared__ float choices[256];
    const uint32_t token = static_cast<uint32_t>(blockIdx.x);
    const uint32_t expert = static_cast<uint32_t>(threadIdx.x);
    if (expert == 0) status[token] = 1;
    __syncthreads();

    double sum = 0.0;
    const uint16_t* row = weight + static_cast<size_t>(expert) * 4096;
    const float* token_input = input + static_cast<size_t>(token) * 4096;
    for (uint32_t column = 0; column < 4096; ++column) {
        sum += static_cast<double>(__uint_as_float(static_cast<uint32_t>(row[column]) << 16))
            * static_cast<double>(token_input[column]);
    }
    const float logit = static_cast<float>(sum);
    const float score = sqrtf(augur_m95_softplus(logit));
    const float choice = score + bias[expert];
    logits[static_cast<size_t>(token) * 256 + expert] = logit;
    scores[expert] = score;
    choices[expert] = choice;
    if (!isfinite(logit) || !isfinite(score) || !isfinite(bias[expert]) || !isfinite(choice)) {
        atomicExch(status + token, 0);
    }
    __syncthreads();
    if (expert != 0) return;

    bool selected[256] = {};
    float denominator = 0.0f;
    for (uint32_t rank = 0; rank < 6; ++rank) {
        int best = -1;
        for (int candidate = 0; candidate < 256; ++candidate) {
            if (!selected[candidate]
                && (best < 0 || choices[candidate] > choices[best]
                    || (choices[candidate] == choices[best] && candidate < best))) {
                best = candidate;
            }
        }
        if (best < 0) {
            status[token] = 0;
            best = 0;
        }
        selected[best] = true;
        output_ids[static_cast<size_t>(token) * 6 + rank] = static_cast<uint16_t>(best);
        denominator += scores[best];
    }
    if (!isfinite(denominator) || denominator <= 0.0f) {
        status[token] = 0;
        denominator = 1.0f;
    }
    for (uint32_t rank = 0; rank < 6; ++rank) {
        const uint16_t id = output_ids[static_cast<size_t>(token) * 6 + rank];
        const float value = scores[id] / denominator * routed_scale;
        output_weights[static_cast<size_t>(token) * 6 + rank] = value;
        if (!isfinite(value)) status[token] = 0;
    }
}

__global__ void augur_m211_learned_router_kernel_v2(
    const uint16_t* weight,
    const float* input,
    const float* bias,
    float* logits,
    uint16_t* output_ids,
    float* output_weights,
    int32_t* status,
    float routed_scale,
    uint32_t denominator_order,
    uint32_t compensated_denominator
) {
    __shared__ float scores[256];
    __shared__ float choices[256];
    const uint32_t token = static_cast<uint32_t>(blockIdx.x);
    const uint32_t expert = static_cast<uint32_t>(threadIdx.x);
    if (expert == 0) status[token] = 1;
    __syncthreads();

    double sum = 0.0;
    const uint16_t* row = weight + static_cast<size_t>(expert) * 4096;
    const float* token_input = input + static_cast<size_t>(token) * 4096;
    for (uint32_t column = 0; column < 4096; ++column) {
        sum += static_cast<double>(__uint_as_float(static_cast<uint32_t>(row[column]) << 16))
            * static_cast<double>(token_input[column]);
    }
    const float logit = static_cast<float>(sum);
    const float score = sqrtf(augur_m95_softplus(logit));
    const float choice = score + bias[expert];
    logits[static_cast<size_t>(token) * 256 + expert] = logit;
    scores[expert] = score;
    choices[expert] = choice;
    if (!isfinite(logit) || !isfinite(score) || !isfinite(bias[expert]) || !isfinite(choice)) {
        atomicExch(status + token, 0);
    }
    __syncthreads();
    if (expert != 0) return;

    bool selected[256] = {};
    uint16_t route_ids[6] = {};
    for (uint32_t rank = 0; rank < 6; ++rank) {
        int best = -1;
        for (int candidate = 0; candidate < 256; ++candidate) {
            if (!selected[candidate]
                && (best < 0 || choices[candidate] > choices[best]
                    || (choices[candidate] == choices[best] && candidate < best))) {
                best = candidate;
            }
        }
        if (best < 0) {
            status[token] = 0;
            best = 0;
        }
        selected[best] = true;
        route_ids[rank] = static_cast<uint16_t>(best);
        output_ids[static_cast<size_t>(token) * 6 + rank] = static_cast<uint16_t>(best);
    }

    float denominator = 0.0f;
    float correction = 0.0f;
    bool consumed[6] = {};
    for (uint32_t order = 0; order < 6; ++order) {
        int rank = -1;
        for (int candidate = 0; candidate < 6; ++candidate) {
            if (!consumed[candidate]
                && (rank < 0 || (denominator_order == 1
                    ? route_ids[candidate] < route_ids[rank]
                    : candidate < rank))) {
                rank = candidate;
            }
        }
        if (rank < 0) {
            status[token] = 0;
            rank = 0;
        }
        consumed[rank] = true;
        const float value = scores[route_ids[rank]];
        if (compensated_denominator != 0) {
            const float adjusted = value - correction;
            const float next = denominator + adjusted;
            correction = (next - denominator) - adjusted;
            denominator = next;
        } else {
            denominator += value;
        }
    }
    if (!isfinite(denominator) || denominator <= 0.0f) {
        status[token] = 0;
        denominator = 1.0f;
    }
    for (uint32_t rank = 0; rank < 6; ++rank) {
        const uint16_t id = output_ids[static_cast<size_t>(token) * 6 + rank];
        const float value = scores[id] / denominator * routed_scale;
        output_weights[static_cast<size_t>(token) * 6 + rank] = value;
        if (!isfinite(value)) status[token] = 0;
    }
}

static augur_cuda_result_v1 augur_validate_m211_learned_router(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m211_learned_router_args_v1* args
) {
    augur_cuda_result_v1 device = augur_validate_stream_device(ordinal, stream);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    if (args == nullptr
        || args->abi_version != AUGUR_CUDA_M211_LEARNED_ROUTER_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_m211_learned_router_args_v1)
        || args->weight_bf16 == 0 || args->input_f32 == 0 || args->bias_f32 == 0
        || args->logits_f32 == 0 || args->output_ids_u16 == 0
        || args->output_weights_f32 == 0 || args->status_i32 == 0
        || args->tokens != 8 || args->hidden != 4096 || args->experts != 256
        || args->top_k != 6 || !isfinite(args->routed_scale) || args->routed_scale <= 0.0f) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->weight_len_bytes != UINT64_C(256) * 4096 * 2
        || args->input_len_bytes != UINT64_C(8) * 4096 * 4
        || args->bias_len_bytes != 256 * 4
        || args->logits_len_bytes != UINT64_C(8) * 256 * 4
        || args->output_ids_len_bytes != 8 * 6 * 2
        || args->output_weights_len_bytes != 8 * 6 * 4
        || args->status_len_bytes != 8 * 4
        || args->weight_bf16 % alignof(uint16_t) != 0
        || args->input_f32 % alignof(float) != 0 || args->bias_f32 % alignof(float) != 0
        || args->logits_f32 % alignof(float) != 0
        || args->output_ids_u16 % alignof(uint16_t) != 0
        || args->output_weights_f32 % alignof(float) != 0
        || args->status_i32 % alignof(int32_t) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_ok();
}

static augur_cuda_result_v1 augur_validate_m211_learned_router_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m211_learned_router_args_v2* args
) {
    augur_cuda_result_v1 device = augur_validate_stream_device(ordinal, stream);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    if (args == nullptr
        || args->abi_version != AUGUR_CUDA_M211_LEARNED_ROUTER_ARGS_VERSION_V2
        || args->struct_size != sizeof(augur_cuda_m211_learned_router_args_v2)
        || args->weight_bf16 == 0 || args->input_f32 == 0 || args->bias_f32 == 0
        || args->logits_f32 == 0 || args->output_ids_u16 == 0
        || args->output_weights_f32 == 0 || args->status_i32 == 0
        || args->tokens == 0 || args->tokens > 8 || args->hidden != 4096 || args->experts != 256
        || args->top_k != 6 || !isfinite(args->routed_scale) || args->routed_scale <= 0.0f
        || args->denominator_order > 1 || args->compensated_denominator > 1
        || args->weight_len_bytes != static_cast<uint64_t>(args->experts) * args->hidden * 2
        || args->input_len_bytes != static_cast<uint64_t>(args->tokens) * args->hidden * 4
        || args->bias_len_bytes != static_cast<uint64_t>(args->experts) * 4
        || args->logits_len_bytes != static_cast<uint64_t>(args->tokens) * args->experts * 4
        || args->output_ids_len_bytes != static_cast<uint64_t>(args->tokens) * args->top_k * 2
        || args->output_weights_len_bytes != static_cast<uint64_t>(args->tokens) * args->top_k * 4
        || args->status_len_bytes != static_cast<uint64_t>(args->tokens) * 4
        || args->weight_bf16 % alignof(uint16_t) != 0
        || args->input_f32 % alignof(float) != 0 || args->bias_f32 % alignof(float) != 0
        || args->logits_f32 % alignof(float) != 0
        || args->output_ids_u16 % alignof(uint16_t) != 0
        || args->output_weights_f32 % alignof(float) != 0
        || args->status_i32 % alignof(int32_t) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m211_learned_router_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m211_learned_router_args_v2* args
) {
    augur_cuda_result_v1 valid = augur_validate_m211_learned_router_v2(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    augur_m211_learned_router_kernel_v2<<<args->tokens, 256, 0, stream->stream>>>(
        reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(args->weight_bf16)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->bias_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->logits_f32)),
        reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(args->output_ids_u16)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_weights_f32)),
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(args->status_i32)),
        args->routed_scale,
        args->denominator_order,
        args->compensated_denominator);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m211_learned_router_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m211_learned_router_args_v2* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_m211_learned_router_submit_v2(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m211_learned_router_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m211_learned_router_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_m211_learned_router(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    augur_m211_learned_router_kernel_v1<<<8, 256, 0, stream->stream>>>(
        reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(args->weight_bf16)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->bias_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->logits_f32)),
        reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(args->output_ids_u16)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_weights_f32)),
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(args->status_i32)),
        args->routed_scale);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m211_learned_router_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m211_learned_router_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_m211_learned_router_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

__global__ void augur_m214_learned_route_telemetry_kernel_v1(
    const float* logits,
    const float* bias,
    float* choices,
    uint16_t* top7_ids,
    float* top7_choices,
    int32_t* status
) {
    __shared__ float row_choices[256];
    const uint32_t token = static_cast<uint32_t>(blockIdx.x);
    const uint32_t expert = static_cast<uint32_t>(threadIdx.x);
    if (expert == 0) status[token] = 1;
    const float logit = logits[static_cast<size_t>(token) * 256 + expert];
    const float correction = bias[expert];
    const float choice = sqrtf(augur_m95_softplus(logit)) + correction;
    choices[static_cast<size_t>(token) * 256 + expert] = choice;
    row_choices[expert] = choice;
    if (!isfinite(logit) || !isfinite(correction) || !isfinite(choice)) {
        atomicExch(status + token, 0);
    }
    __syncthreads();
    if (expert != 0) return;

    bool selected[256] = {};
    for (uint32_t rank = 0; rank < 7; ++rank) {
        int best = -1;
        for (int candidate = 0; candidate < 256; ++candidate) {
            if (!selected[candidate]
                && (best < 0 || row_choices[candidate] > row_choices[best]
                    || (row_choices[candidate] == row_choices[best] && candidate < best))) {
                best = candidate;
            }
        }
        if (best < 0) {
            status[token] = 0;
            best = 0;
        }
        selected[best] = true;
        top7_ids[static_cast<size_t>(token) * 7 + rank] = static_cast<uint16_t>(best);
        top7_choices[static_cast<size_t>(token) * 7 + rank] = row_choices[best];
    }
}

static augur_cuda_result_v1 augur_validate_m214_learned_route_telemetry(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m214_learned_route_telemetry_args_v1* args
) {
    augur_cuda_result_v1 device = augur_validate_stream_device(ordinal, stream);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    if (args == nullptr
        || args->abi_version != AUGUR_CUDA_M214_LEARNED_ROUTE_TELEMETRY_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_m214_learned_route_telemetry_args_v1)
        || args->logits_f32 == 0 || args->bias_f32 == 0 || args->choices_f32 == 0
        || args->top7_ids_u16 == 0 || args->top7_choices_f32 == 0 || args->status_i32 == 0
        || args->tokens == 0 || args->tokens > 8 || args->experts != 256 || args->top_n != 7) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->logits_len_bytes != static_cast<uint64_t>(args->tokens) * 256 * 4
        || args->bias_len_bytes != 256 * 4
        || args->choices_len_bytes != static_cast<uint64_t>(args->tokens) * 256 * 4
        || args->top7_ids_len_bytes != static_cast<uint64_t>(args->tokens) * 7 * 2
        || args->top7_choices_len_bytes != static_cast<uint64_t>(args->tokens) * 7 * 4
        || args->status_len_bytes != static_cast<uint64_t>(args->tokens) * 4
        || args->logits_f32 % alignof(float) != 0 || args->bias_f32 % alignof(float) != 0
        || args->choices_f32 % alignof(float) != 0
        || args->top7_ids_u16 % alignof(uint16_t) != 0
        || args->top7_choices_f32 % alignof(float) != 0
        || args->status_i32 % alignof(int32_t) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m214_learned_route_telemetry_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m214_learned_route_telemetry_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_m214_learned_route_telemetry(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    augur_m214_learned_route_telemetry_kernel_v1<<<args->tokens, 256, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->logits_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->bias_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->choices_f32)),
        reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(args->top7_ids_u16)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->top7_choices_f32)),
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(args->status_i32)));
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m214_learned_route_telemetry_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m214_learned_route_telemetry_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_m214_learned_route_telemetry_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

__device__ __forceinline__ float augur_m205_bf16(float value) {
    uint32_t bits = __float_as_uint(value);
    const uint32_t lsb = (bits >> 16) & 1U;
    bits += UINT32_C(0x7fff) + lsb;
    return __uint_as_float(bits & UINT32_C(0xffff0000));
}

__global__ void augur_m205_route_bf16_kernel_v1(
    const float* input,
    const float* route_weights,
    float* output,
    uint32_t token,
    uint32_t rank,
    uint32_t elements
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    output[index] = augur_m205_bf16(input[index] * route_weights[token * 6 + rank]);
}

static augur_cuda_result_v1 augur_validate_m205_route_bf16(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m205_route_bf16_args_v1* args
) {
    augur_cuda_result_v1 device = augur_validate_stream_device(ordinal, stream);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    if (args == nullptr
        || args->abi_version != AUGUR_CUDA_M205_ROUTE_BF16_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_m205_route_bf16_args_v1)
        || args->input_f32 == 0 || args->route_weights_f32 == 0 || args->output_f32 == 0
        || args->elements == 0 || args->elements > 2048
        || args->input_len_bytes != static_cast<uint64_t>(args->elements) * 4
        || args->route_weights_len_bytes == 0
        || args->route_weights_len_bytes > 8 * 6 * 4 || args->route_weights_len_bytes % (6 * 4)
        || args->output_len_bytes != static_cast<uint64_t>(args->elements) * 4
        || args->token >= args->route_weights_len_bytes / (6 * 4) || args->rank >= 6
        || args->input_f32 % alignof(float) != 0
        || args->route_weights_f32 % alignof(float) != 0
        || args->output_f32 % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m205_route_bf16_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m205_route_bf16_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_m205_route_bf16(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    const uint32_t blocks = (args->elements + 255) / 256;
    augur_m205_route_bf16_kernel_v1<<<blocks, 256, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->route_weights_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        args->token,
        args->rank,
        args->elements);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m205_route_bf16_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m205_route_bf16_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_m205_route_bf16_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

__global__ void augur_weighted_accum_f32_kernel_v1(
    const float* input,
    float* output,
    uint32_t elements,
    float weight,
    uint32_t replace
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float value = input[index] * weight;
    output[index] = replace != 0 ? value : output[index] + value;
}

static augur_cuda_result_v1 augur_validate_weighted_accum_f32(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_weighted_accum_f32_args_v1* args
) {
    augur_cuda_result_v1 device = augur_validate_stream_device(ordinal, stream);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    float weight = 0.0f;
    if (args != nullptr) std::memcpy(&weight, &args->weight_bits, sizeof(weight));
    const uint64_t bytes = args == nullptr ? 0 : static_cast<uint64_t>(args->elements) * 4;
    if (args == nullptr || args->abi_version != AUGUR_CUDA_WEIGHTED_ACCUM_F32_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_weighted_accum_f32_args_v1)
        || args->input_f32 == 0 || args->output_f32 == 0 || args->elements == 0
        || args->input_f32 == args->output_f32 || !isfinite(weight)
        || (args->replace != 0 && args->replace != 1)
        || args->input_len_bytes != bytes || args->output_len_bytes != bytes
        || args->input_f32 % alignof(float) != 0 || args->output_f32 % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_weighted_accum_f32_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_weighted_accum_f32_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_weighted_accum_f32(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    float weight = 0.0f;
    std::memcpy(&weight, &args->weight_bits, sizeof(weight));
    constexpr uint32_t threads = 256;
    const uint32_t blocks = (args->elements + threads - 1) / threads;
    augur_weighted_accum_f32_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        args->elements,
        weight,
        args->replace);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_weighted_accum_f32_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_weighted_accum_f32_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_weighted_accum_f32_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

__device__ __forceinline__ void augur_two_sum_f32(float a, float b, float* sum, float* error) {
    const float value = a + b;
    const float restored = value - a;
    *sum = value;
    *error = (a - (value - restored)) + (b - restored);
}

__global__ void augur_compensated_accum_f32_kernel_v1(
    const float* input,
    float* output,
    float* compensation,
    uint32_t elements,
    float weight,
    uint32_t replace
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float product = input[index] * weight;
    if (replace != 0) {
        output[index] = product;
        compensation[index] = fmaf(input[index], weight, -product);
        return;
    }
    float sum = 0.0f;
    float add_error = 0.0f;
    augur_two_sum_f32(output[index], product, &sum, &add_error);
    const float product_error = fmaf(input[index], weight, -product);
    float correction = 0.0f;
    float correction_error = 0.0f;
    augur_two_sum_f32(
        compensation[index],
        add_error + product_error,
        &correction,
        &correction_error);
    float normalized = 0.0f;
    float normalized_error = 0.0f;
    augur_two_sum_f32(sum, correction + correction_error, &normalized, &normalized_error);
    output[index] = normalized;
    compensation[index] = normalized_error;
}

static augur_cuda_result_v1 augur_validate_compensated_accum_f32(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_compensated_accum_f32_args_v1* args
) {
    augur_cuda_result_v1 device = augur_validate_stream_device(ordinal, stream);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    float weight = 0.0f;
    if (args != nullptr) std::memcpy(&weight, &args->weight_bits, sizeof(weight));
    const uint64_t bytes = args == nullptr ? 0 : static_cast<uint64_t>(args->elements) * 4;
    if (args == nullptr
        || args->abi_version != AUGUR_CUDA_COMPENSATED_ACCUM_F32_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_compensated_accum_f32_args_v1)
        || args->input_f32 == 0 || args->output_f32 == 0 || args->compensation_f32 == 0
        || args->elements == 0 || args->input_f32 == args->output_f32
        || args->input_f32 == args->compensation_f32 || args->output_f32 == args->compensation_f32
        || !isfinite(weight) || (args->replace != 0 && args->replace != 1)
        || args->input_len_bytes != bytes || args->output_len_bytes != bytes
        || args->compensation_len_bytes != bytes
        || args->input_f32 % alignof(float) != 0 || args->output_f32 % alignof(float) != 0
        || args->compensation_f32 % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_compensated_accum_f32_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_compensated_accum_f32_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_compensated_accum_f32(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    float weight = 0.0f;
    std::memcpy(&weight, &args->weight_bits, sizeof(weight));
    constexpr uint32_t threads = 256;
    const uint32_t blocks = (args->elements + threads - 1) / threads;
    augur_compensated_accum_f32_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->compensation_f32)),
        args->elements,
        weight,
        args->replace);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_compensated_accum_f32_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_compensated_accum_f32_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_compensated_accum_f32_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

#include "augur_rmsnorm_warp_tree5120.cuh"

__global__ void augur_rmsnorm_f32_kernel_v1(
    const float* input, const float* weight, float* output, uint32_t elements, float epsilon
) {
    __shared__ float partial[256];
    const uint32_t lane = threadIdx.x;
    float sum = 0.0f;
    for (uint32_t index = lane; index < elements; index += blockDim.x) sum += input[index] * input[index];
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / static_cast<float>(elements) + epsilon);
    for (uint32_t index = lane; index < elements; index += blockDim.x) output[index] = input[index] * scale * weight[index];
}

// Keep this in the canonical runtime translation unit: the prefill unit is
// compiled with different FMA flags. One CTA owns one independent token row;
// the scalar kernel above remains the unchanged arithmetic/control reference.
__global__ void augur_rmsnorm_f32_rows_kernel_v1(
    const float* input, const float* weight, float* output, uint32_t elements, float epsilon
) {
    input += static_cast<uint64_t>(blockIdx.x) * elements;
    output += static_cast<uint64_t>(blockIdx.x) * elements;
    __shared__ float partial[256];
    const uint32_t lane = threadIdx.x;
    float sum = 0.0f;
    for (uint32_t index = lane; index < elements; index += blockDim.x) sum += input[index] * input[index];
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / static_cast<float>(elements) + epsilon);
    for (uint32_t index = lane; index < elements; index += blockDim.x) output[index] = input[index] * scale * weight[index];
}

static augur_cuda_result_v1 augur_validate_rmsnorm_args(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_rmsnorm_f32_args_v1* args
) {
    float epsilon = 0.0f; if (args != nullptr) std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    const uint64_t expected = args == nullptr ? 0 : static_cast<uint64_t>(args->elements) * 4;
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_RMSNORM_F32_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_rmsnorm_f32_args_v1) || args->input == 0
        || args->weight == 0 || args->output == 0 || args->elements == 0
        || !isfinite(epsilon) || epsilon <= 0.0f || args->input_len_bytes < expected
        || args->weight_len_bytes < expected || args->output_len_bytes < expected
        || args->input % alignof(float) != 0 || args->weight % alignof(float) != 0
        || args->output % alignof(float) != 0) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    return augur_set_device(ordinal);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_rmsnorm_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_rmsnorm_f32_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_rmsnorm_args(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    augur_rmsnorm_f32_kernel_v1<<<1, 256, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)), args->elements, epsilon);
    const cudaError_t error = cudaGetLastError(); if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_rmsnorm_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_rmsnorm_f32_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_rmsnorm_f32_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_rmsnorm_f32_warp_tree5120_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_rmsnorm_f32_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_rmsnorm_args(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    if (args->elements != 5120) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    augur_rmsnorm_warp_tree5120::warp_tree_norm<true,true,false><<<1,256,0,stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)), nullptr, epsilon, 1);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_rmsnorm_f32_warp_tree5120_qpn_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_rmsnorm_f32_args_v1* args,
    uint64_t staged, uint64_t staged_bytes
) {
    augur_cuda_result_v1 valid = augur_validate_rmsnorm_args(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    if (args->elements != 5120 || staged == 0 || staged % 16 != 0 || staged_bytes != 20480)
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    const uint64_t ranges[4][2] = {{args->input,args->input_len_bytes},
        {args->weight,args->weight_len_bytes},{args->output,args->output_len_bytes},{staged,staged_bytes}};
    for (int i=0;i<4;++i) {
        if (ranges[i][0] > UINT64_MAX-ranges[i][1])
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        for (int j=0;j<i;++j)
            if (ranges[i][0]<ranges[j][0]+ranges[j][1] && ranges[j][0]<ranges[i][0]+ranges[i][1])
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    augur_rmsnorm_warp_tree5120::warp_tree_norm<true,true,true><<<1,256,0,stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        reinterpret_cast<half*>(static_cast<uintptr_t>(staged)), epsilon, 1);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_rmsnorm_f32_rows_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_rmsnorm_f32_args_v1* args,
    uint32_t rows
) {
    if (args == nullptr || rows == 0 || rows > 512) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    // The bounded u32 dimensions make these u64 products non-overflowing.
    const uint64_t row_bytes = static_cast<uint64_t>(args->elements) * sizeof(float);
    const uint64_t batch_bytes = row_bytes * rows;
    if (args->input_len_bytes != batch_bytes || args->output_len_bytes != batch_bytes
        || args->weight_len_bytes != row_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t ranges[3][2] = {
        {args->input, batch_bytes}, {args->weight, row_bytes}, {args->output, batch_bytes}
    };
    for (uint32_t i = 0; i < 3; ++i) {
        if (ranges[i][0] > UINT64_MAX - ranges[i][1]) {
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (ranges[i][0] < ranges[j][0] + ranges[j][1]
                && ranges[j][0] < ranges[i][0] + ranges[i][1]) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    augur_cuda_result_v1 valid = augur_validate_rmsnorm_args(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    augur_rmsnorm_f32_rows_kernel_v1<<<rows, 256, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)), args->elements, epsilon);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

static_assert(sizeof(augur_cuda_split_rmsnorm_f32_args_v1) == 80,
    "augur_cuda_split_rmsnorm_f32_args_v1 ABI size changed");
static_assert(alignof(augur_cuda_split_rmsnorm_f32_args_v1) == 8,
    "augur_cuda_split_rmsnorm_f32_args_v1 ABI alignment changed");

__global__ void augur_split_rmsnorm_reduce_f32_kernel_v1(
    const float* input,
    float* scale_workspace,
    uint32_t elements,
    float epsilon
) {
    __shared__ float partial[256];
    const uint32_t lane = threadIdx.x;
    float sum = 0.0f;
    for (uint32_t index = lane; index < elements; index += blockDim.x) {
        sum += input[index] * input[index];
    }
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) {
        scale_workspace[0] = rsqrtf(
            partial[0] / static_cast<float>(elements) + epsilon
        );
    }
}

__global__ void augur_split_rmsnorm_apply_f32_kernel_v1(
    const float* input,
    const float* weight,
    const float* scale_workspace,
    float* output,
    uint32_t elements
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = input[index] * scale_workspace[0] * weight[index];
    }
}

static augur_cuda_result_v1 augur_validate_split_rmsnorm_args(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_split_rmsnorm_f32_args_v1* args
) {
    float epsilon = 0.0f;
    if (args != nullptr) std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    const uint64_t expected = args == nullptr
        ? 0
        : static_cast<uint64_t>(args->elements) * sizeof(float);
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_SPLIT_RMSNORM_F32_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_split_rmsnorm_f32_args_v1)
        || args->input == 0 || args->weight == 0 || args->output == 0
        || args->scale_workspace == 0 || args->elements == 0
        || !isfinite(epsilon) || epsilon <= 0.0f
        || args->input_len_bytes < expected || args->weight_len_bytes < expected
        || args->output_len_bytes < expected || args->scale_workspace_len_bytes < sizeof(float)
        || args->input % alignof(float) != 0 || args->weight % alignof(float) != 0
        || args->output % alignof(float) != 0
        || args->scale_workspace % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_set_device(ordinal);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_split_rmsnorm_f32_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_split_rmsnorm_f32_args_v1* args
) {
    const augur_cuda_result_v1 valid = augur_validate_split_rmsnorm_args(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    const auto* input = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input));
    const auto* weight = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->weight));
    auto* output = reinterpret_cast<float*>(static_cast<uintptr_t>(args->output));
    auto* scale_workspace =
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->scale_workspace));
    augur_split_rmsnorm_reduce_f32_kernel_v1<<<1, 256, 0, stream->stream>>>(
        input, scale_workspace, args->elements, epsilon
    );
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    }
    constexpr uint32_t threads = 256;
    const uint32_t blocks = (args->elements + threads - 1) / threads;
    augur_split_rmsnorm_apply_f32_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        input, weight, scale_workspace, output, args->elements
    );
    error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_split_rmsnorm_f32_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_split_rmsnorm_f32_args_v1* args
) {
    const augur_cuda_result_v1 submitted =
        augur_cuda_split_rmsnorm_f32_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

__global__ void augur_rope_f32_kernel_v1(
    const float* input, float* output, const float* cos, const float* sin, uint32_t elements, uint32_t dimensions
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    if (index < dimensions) {
        const uint32_t pair = index / 2; const float c = cos[pair]; const float s = sin[pair];
        const float x = input[index]; const float mate = input[index ^ 1];
        output[index] = (index & 1) ? mate * s + x * c : x * c - mate * s;
    } else output[index] = input[index];
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_rope_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_rope_f32_args_v1* args
) {
    const uint64_t expected = args == nullptr ? 0 : static_cast<uint64_t>(args->elements) * 4;
    const uint32_t pairs = args == nullptr ? 0 : args->dimensions / 2;
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_ROPE_F32_ARGS_VERSION_V1 || args->struct_size != sizeof(augur_cuda_rope_f32_args_v1)
        || args->input == 0 || args->output == 0 || args->cos == 0 || args->sin == 0 || args->elements == 0
        || args->dimensions == 0 || args->dimensions > args->elements || (args->dimensions & 1) != 0
        || args->input_len_bytes < expected || args->output_len_bytes < expected
        || args->cos_len_bytes < static_cast<uint64_t>(pairs) * 4 || args->sin_len_bytes < static_cast<uint64_t>(pairs) * 4
        || args->input % alignof(float) != 0 || args->output % alignof(float) != 0
        || args->cos % alignof(float) != 0 || args->sin % alignof(float) != 0) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    augur_cuda_result_v1 device = augur_set_device(ordinal); if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    constexpr uint32_t threads = 256; const uint32_t blocks = (args->elements + threads - 1) / threads;
    augur_rope_f32_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)), reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->cos)), reinterpret_cast<const float*>(static_cast<uintptr_t>(args->sin)), args->elements, args->dimensions);
    const cudaError_t error = cudaGetLastError(); if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_rope_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_rope_f32_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_rope_f32_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

__global__ void augur_dflash_rope_f32_kernel_v1(
    const float* input,
    float* output,
    uint32_t elements,
    uint32_t dimensions,
    uint32_t rotary_dimensions,
    uint64_t position,
    float theta
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    if (index >= dimensions || index >= rotary_dimensions) {
        output[index] = input[index];
        return;
    }
    const uint32_t half = rotary_dimensions / 2;
    const uint32_t pair = index < half ? index : index - half;
    // Same phase contract as the CPU oracle: high-precision frequency rounded
    // once to F32, followed by F32 absolute-position multiplication. A 1-ulp
    // platform powf discrepancy otherwise becomes a visible long-window error.
    const double exponent = -2.0 * static_cast<double>(pair) / static_cast<double>(rotary_dimensions);
    const float inverse_frequency = static_cast<float>(pow(static_cast<double>(theta), exponent));
    const float angle = static_cast<float>(position) * inverse_frequency;
    float sine;
    float cosine;
    sincosf(angle, &sine, &cosine);
    const float mate = input[index < half ? index + half : index - half];
    const float value = input[index];
    output[index] = index < half ? value * cosine - mate * sine : mate * sine + value * cosine;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_rope_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_dflash_rope_f32_args_v1* args
) {
    const uint64_t expected = args == nullptr ? 0 : static_cast<uint64_t>(args->elements) * 4;
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_DFLASH_ROPE_F32_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_dflash_rope_f32_args_v1)
        || args->input == 0 || args->output == 0 || args->elements == 0
        || args->dimensions == 0 || args->dimensions > args->elements
        || args->rotary_dimensions == 0 || args->rotary_dimensions > args->dimensions
        || (args->rotary_dimensions & 1) != 0 || args->theta <= 1.0f || !isfinite(args->theta)
        || args->input_len_bytes < expected || args->output_len_bytes < expected
        || args->input % alignof(float) != 0 || args->output % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    constexpr uint32_t threads = 128;
    const uint32_t blocks = (args->elements + threads - 1) / threads;
    augur_dflash_rope_f32_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->elements,
        args->dimensions,
        args->rotary_dimensions,
        args->position,
        args->theta
    );
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_rope_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_dflash_rope_f32_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_dflash_rope_f32_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

// These independent-row kernels deliberately retain the scalar entry points'
// FP64 matvec accumulation, 256-thread norm reduction and split-half RoPE.
// Only the row grid changes; no GEMM/dequant precision substitution is made.
__global__ void augur_dflash_q8_context_rows_kernel_v1(
    const uint8_t* weight, const float* inputs, float* outputs,
    uint32_t rows, uint32_t cols
) {
    const float* input = inputs + static_cast<size_t>(blockIdx.y) * cols;
    float* output = outputs + static_cast<size_t>(blockIdx.y) * rows;
    constexpr uint32_t warp_size = 32;
    constexpr uint32_t warps_per_block = 4;
    const uint32_t warp = threadIdx.x / warp_size;
    const uint32_t lane = threadIdx.x & (warp_size - 1U);
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks_per_row = cols / 32;
    const uint64_t row_bytes = static_cast<uint64_t>(blocks_per_row) * 34;
    const uint8_t* source = weight + static_cast<size_t>(row * row_bytes);
    double partial = 0.0;
    for (uint32_t block = 0; block < blocks_per_row; ++block) {
        uint32_t scale_bits = lane == 0
            ? static_cast<uint32_t>(source[0]) | static_cast<uint32_t>(source[1]) << 8 : 0U;
        scale_bits = __shfl_sync(0xffffffffU, scale_bits, 0);
        const double scale = static_cast<double>(augur_q8_0_f16_to_f32(scale_bits));
        const uint32_t input_base = block * 32;
        const uint8_t encoded = source[2 + lane];
        const int32_t code = encoded < 128U ? static_cast<int32_t>(encoded) : static_cast<int32_t>(encoded) - 256;
        partial += static_cast<double>(code) * scale * static_cast<double>(input[input_base + lane]);
        source += 34;
    }
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U) {
        partial += __shfl_down_sync(0xffffffffU, partial, offset);
    }
    if (lane == 0) output[row] = static_cast<float>(partial);
}

__global__ void augur_dflash_norm_context_rows_kernel_v1(
    const float* inputs, const float* weight, float* outputs, uint32_t elements, float epsilon
) {
    const float* input = inputs + static_cast<size_t>(blockIdx.x) * elements;
    float* output = outputs + static_cast<size_t>(blockIdx.x) * elements;
    __shared__ float partial[256];
    const uint32_t lane = threadIdx.x;
    float sum = 0.0f;
    for (uint32_t index = lane; index < elements; index += blockDim.x) sum += input[index] * input[index];
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / static_cast<float>(elements) + epsilon);
    for (uint32_t index = lane; index < elements; index += blockDim.x) output[index] = input[index] * scale * weight[index];
}

// Four independent token accumulators reuse each Q8 weight code/scale load.
// No horizontal token reduction, activation quantization, or precision change.
__global__ void augur_dflash_q8_rows_tile4_kernel_v1(
    const uint8_t* weight, const float* inputs, float* outputs,
    uint32_t rows, uint32_t cols, uint32_t tokens
) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x & 31U;
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * 4 + warp;
    if (row >= rows) return;
    const uint32_t first = blockIdx.y * 4;
    const uint32_t blocks_per_row = cols / 32;
    const uint8_t* source = weight + static_cast<size_t>(row) * blocks_per_row * 34;
    double partial[4] = {};
    for (uint32_t block = 0; block < blocks_per_row; ++block) {
        uint32_t scale_bits = lane == 0
            ? static_cast<uint32_t>(source[0]) | static_cast<uint32_t>(source[1]) << 8 : 0U;
        scale_bits = __shfl_sync(0xffffffffU, scale_bits, 0);
        const double scale = static_cast<double>(augur_q8_0_f16_to_f32(scale_bits));
        const uint8_t encoded = source[2 + lane];
        const int32_t code = encoded < 128U ? static_cast<int32_t>(encoded) : static_cast<int32_t>(encoded) - 256;
        #pragma unroll
        for (uint32_t t = 0; t < 4; ++t) {
            if (first + t < tokens) {
                const float x = inputs[static_cast<size_t>(first + t) * cols + block * 32 + lane];
                partial[t] += static_cast<double>(code) * scale * static_cast<double>(x);
            }
        }
        source += 34;
    }
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U) {
        #pragma unroll
        for (uint32_t t = 0; t < 4; ++t) {
            partial[t] += __shfl_down_sync(0xffffffffU, partial[t], offset);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (uint32_t t = 0; t < 4; ++t) {
            if (first + t < tokens) outputs[static_cast<size_t>(first + t) * rows + row] = static_cast<float>(partial[t]);
        }
    }
}

// Short draft scheduling only. Keep the original context/large-row kernel
// unchanged. Four consecutive K blocks retain each lane's FP64 add order;
// explicit unrolling exposes independent weight/input loads to the compiler.
// Same four-token reuse, scale conversion, reduction and FP32 output as above.
__global__ void augur_dflash_q8_rows_tile4_short_unroll4_kernel_v1(
    const uint8_t* weight, const float* inputs, float* outputs,
    uint32_t rows, uint32_t cols, uint32_t tokens
) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x & 31U;
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * 4 + warp;
    if (row >= rows) return;
    const uint32_t first = blockIdx.y * 4;
    const uint32_t blocks_per_row = cols / 32;
    const uint8_t* source = weight + static_cast<size_t>(row) * blocks_per_row * 34;
    double partial[4] = {};
    #pragma unroll 4
    for (uint32_t block = 0; block < blocks_per_row; ++block) {
        uint32_t scale_bits = lane == 0
            ? static_cast<uint32_t>(source[0]) | static_cast<uint32_t>(source[1]) << 8 : 0U;
        scale_bits = __shfl_sync(0xffffffffU, scale_bits, 0);
        const double scale = static_cast<double>(augur_q8_0_f16_to_f32(scale_bits));
        const uint8_t encoded = source[2 + lane];
        const int32_t code = encoded < 128U ? static_cast<int32_t>(encoded) : static_cast<int32_t>(encoded) - 256;
        #pragma unroll
        for (uint32_t t = 0; t < 4; ++t) {
            if (first + t < tokens) {
                const float x = inputs[static_cast<size_t>(first + t) * cols + block * 32 + lane];
                partial[t] += static_cast<double>(code) * scale * static_cast<double>(x);
            }
        }
        source += 34;
    }
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U) {
        #pragma unroll
        for (uint32_t t = 0; t < 4; ++t) {
            partial[t] += __shfl_down_sync(0xffffffffU, partial[t], offset);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (uint32_t t = 0; t < 4; ++t) {
            if (first + t < tokens) outputs[static_cast<size_t>(first + t) * rows + row] = static_cast<float>(partial[t]);
        }
    }
}

// Four-token exact-weight tile: a finite FP16 scale (<=11 significand bits) times int8
// fits FP32 exactly. Materialize that weight in FP32, then preserve the
// original FP64 product/accumulation order. Direct aligned halfword loads
// avoid the lane0 byte assembly and scale shuffle. Admit only the measured
// 512/2048-row history FC/KV shapes below; short B8 uses its separate tiles.
__global__ void augur_dflash_q8_rows_tile4_exact_dequant8_kernel_v1(
    const uint8_t* weight, const float* inputs, float* outputs,
    uint32_t rows, uint32_t cols, uint32_t tokens
) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x & 31U;
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * 4 + warp;
    if (row >= rows) return;
    const uint32_t first = blockIdx.y * 4;
    const uint32_t blocks_per_row = cols / 32;
    const uint8_t* source = weight + static_cast<size_t>(row) * blocks_per_row * 34;
    double partial[4] = {};
    #pragma unroll 4
    for (uint32_t block = 0; block < blocks_per_row; ++block) {
        const uint32_t scale_bits = *reinterpret_cast<const uint16_t*>(source);
        const float scale = augur_q8_0_f16_to_f32(scale_bits);
        const uint8_t encoded = source[2 + lane];
        const int32_t code = encoded < 128U ? static_cast<int32_t>(encoded) : static_cast<int32_t>(encoded) - 256;
        const double decoded = static_cast<double>(__fmul_rn(static_cast<float>(code), scale));
        #pragma unroll
        for (uint32_t t = 0; t < 4; ++t) {
            if (first + t < tokens) {
                const float x = inputs[static_cast<size_t>(first + t) * cols + block * 32 + lane];
                partial[t] += decoded * static_cast<double>(x);
            }
        }
        source += 34;
    }
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U) {
        #pragma unroll
        for (uint32_t t = 0; t < 4; ++t) {
            partial[t] += __shfl_down_sync(0xffffffffU, partial[t], offset);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (uint32_t t = 0; t < 4; ++t) {
            if (first + t < tokens) outputs[static_cast<size_t>(first + t) * rows + row] = static_cast<float>(partial[t]);
        }
    }
}

// Bulk-history alternative: keep the four-token tile and each output's FP64
// K/reduction order, but reuse input loads/conversions across four output rows.
// Only the separate bounded history ABI below selects this component.
__global__ void augur_dflash_q8_history_row4_kernel_v1(
    const uint8_t* weight, const float* inputs, float* outputs,
    uint32_t rows, uint32_t cols, uint32_t tokens) {
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t first_row = blockIdx.x * 16 + (threadIdx.x / 32) * 4;
    const uint32_t first_token = blockIdx.y * 4;
    if (first_row + 4 > rows) return;
    const uint32_t blocks = cols / 32;
    const uint8_t* source[4];
    double partial[4][4] = {};
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        source[r] = weight + static_cast<size_t>(first_row + r) * blocks * 34;
    #pragma unroll 4
    for (uint32_t block = 0; block < blocks; ++block) {
        double decoded[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float scale = augur_q8_0_f16_to_f32(*reinterpret_cast<const uint16_t*>(source[r]));
            const uint8_t encoded = source[r][2 + lane];
            const int32_t code = encoded < 128U ? static_cast<int32_t>(encoded) : static_cast<int32_t>(encoded) - 256;
            decoded[r] = static_cast<double>(__fmul_rn(static_cast<float>(code), scale));
            source[r] += 34;
        }
        #pragma unroll
        for (uint32_t t = 0; t < 4; ++t) {
            if (first_token + t < tokens) {
                const double x = static_cast<double>(inputs[static_cast<size_t>(first_token + t) * cols + block * 32 + lane]);
                #pragma unroll
                for (int r = 0; r < 4; ++r) partial[r][t] += decoded[r] * x;
            }
        }
    }
    #pragma unroll
    for (uint32_t offset = 16; offset; offset >>= 1) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            #pragma unroll
            for (int t = 0; t < 4; ++t)
                partial[r][t] += __shfl_down_sync(0xffffffffU, partial[r][t], offset);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            #pragma unroll
            for (uint32_t t = 0; t < 4; ++t)
                if (first_token + t < tokens)
                    outputs[static_cast<size_t>(first_token + t) * rows + first_row + r] = static_cast<float>(partial[r][t]);
        }
    }
}

// Bulk history only: bounded FP32 weight-decode scratch, identical FP64 dots.
// The caller owns scratch; one stream serializes panels and their consumers.
__global__ void augur_dflash_history_decode_panel_kernel_v1(
    const uint8_t* weight, float* panel, uint32_t first_row,
    uint32_t panel_rows, uint32_t cols) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(panel_rows) * cols) return;
    const uint32_t row = first_row + i / cols;
    const uint32_t col = i % cols;
    const uint8_t* source = weight + (size_t(row) * (cols / 32) + col / 32) * 34;
    const float scale = augur_q8_0_f16_to_f32(*reinterpret_cast<const uint16_t*>(source));
    const uint8_t encoded = source[2 + col % 32];
    const int32_t code = encoded < 128U ? int32_t(encoded) : int32_t(encoded) - 256;
    panel[i] = __fmul_rn(float(code), scale);
}

__global__ void augur_dflash_history_panel_dot_kernel_v1(
    const float* weight, const float* inputs, float* outputs,
    uint32_t rows, uint32_t cols, uint32_t tokens,
    uint32_t panel_first, uint32_t panel_rows) {
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t first_row = blockIdx.x * 16 + (threadIdx.x / 32) * 4;
    const uint32_t first_token = blockIdx.y * 4;
    if (first_row + 4 > panel_rows) return;
    const uint32_t blocks = cols / 32;
    double partial[4][4] = {};
    #pragma unroll 4
    for (uint32_t block = 0; block < blocks; ++block) {
        double decoded[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r)
            decoded[r] = double(weight[size_t(first_row + r) * cols + block * 32 + lane]);
        #pragma unroll
        for (uint32_t t = 0; t < 4; ++t) {
            if (first_token + t < tokens) {
                const double x = double(inputs[size_t(first_token + t) * cols + block * 32 + lane]);
                #pragma unroll
                for (int r = 0; r < 4; ++r) partial[r][t] += decoded[r] * x;
            }
        }
    }
    #pragma unroll
    for (uint32_t offset = 16; offset; offset >>= 1) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            #pragma unroll
            for (int t = 0; t < 4; ++t)
                partial[r][t] += __shfl_down_sync(0xffffffffU, partial[r][t], offset);
        }
    }
    if (!lane) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            #pragma unroll
            for (uint32_t t = 0; t < 4; ++t)
                if (first_token + t < tokens)
                    outputs[size_t(first_token + t) * rows + panel_first + first_row + r] = float(partial[r][t]);
        }
    }
}

// Bulk decoded panels only; each warp reuses four rows over eight tokens.
// The checked caller admits only complete 512/2048 histories. Keep FP64
// per-lane K sequence, contraction policy and shuffle reduction unchanged.
__global__ void augur_dflash_history_panel_token8_kernel_v1(
    const float* weight,const float* input,float* output,
    uint32_t rows,uint32_t cols,uint32_t tokens,uint32_t panel_first,uint32_t panel_rows) {
    const uint32_t lane=threadIdx.x&31U;
    const uint32_t row=blockIdx.x*16+(threadIdx.x/32U)*4,token=blockIdx.y*8;
    double partial[4][8]={};
    #pragma unroll 4
    for(uint32_t b=0;b<cols/32;++b) {
        double decoded[4];
        #pragma unroll
        for(int r=0;r<4;++r)decoded[r]=double(weight[size_t(row+r)*cols+b*32+lane]);
        #pragma unroll
        for(int t=0;t<8;++t) {
            const double x=double(input[size_t(token+t)*cols+b*32+lane]);
            #pragma unroll
            for(int r=0;r<4;++r)partial[r][t]+=decoded[r]*x;
        }
    }
    #pragma unroll
    for(uint32_t offset=16;offset;offset>>=1) {
        #pragma unroll
        for(int r=0;r<4;++r) {
            #pragma unroll
            for(int t=0;t<8;++t)partial[r][t]+=__shfl_down_sync(0xffffffffU,partial[r][t],offset);
        }
    }
    if(!lane) {
        #pragma unroll
        for(int r=0;r<4;++r) {
            #pragma unroll
            for(int t=0;t<8;++t)output[size_t(token+t)*rows+panel_first+row+r]=float(partial[r][t]);
        }
    }
}

// Capture FC slice only: guard incomplete token tiles without changing any
// real row's per-lane K accumulation, contraction or shuffle order.
__global__ void augur_dflash_history_fc_slice_kernel_v1(
    const float* weight,const float* input,float* output,
    uint32_t rows,uint32_t cols,uint32_t tokens,uint32_t panel_first,uint32_t panel_rows) {
    const uint32_t lane=threadIdx.x&31U;
    const uint32_t row=blockIdx.x*16+(threadIdx.x/32U)*4,token=blockIdx.y*8;
    double partial[4][8]={};
    #pragma unroll 4
    for(uint32_t b=0;b<cols/32;++b) {
        double decoded[4];
        #pragma unroll
        for(int r=0;r<4;++r)decoded[r]=double(weight[size_t(row+r)*cols+b*32+lane]);
        #pragma unroll
        for(int t=0;t<8;++t) {
            const double x=token+t<tokens ? double(input[size_t(token+t)*cols+b*32+lane]) : 0.0;
            #pragma unroll
            for(int r=0;r<4;++r)partial[r][t]+=decoded[r]*x;
        }
    }
    #pragma unroll
    for(uint32_t offset=16;offset;offset>>=1) {
        #pragma unroll
        for(int r=0;r<4;++r) {
            #pragma unroll
            for(int t=0;t<8;++t)partial[r][t]+=__shfl_down_sync(0xffffffffU,partial[r][t],offset);
        }
    }
    if(!lane) {
        #pragma unroll
        for(int r=0;r<4;++r) {
            #pragma unroll
            for(int t=0;t<8;++t)if(token+t<tokens)
                output[size_t(token+t)*rows+panel_first+row+r]=float(partial[r][t]);
        }
    }
}

// Only the existing full8 short-draft selector admits this tile. Reuse one
// loaded Q8 weight/scale over all8 token rows. Each output keeps identical
// FP64 K accumulation and warp reduction order; tails/history stay tile4.
__global__ void augur_dflash_q8_rows_tile8_exact_dequant_kernel_v1(
    const uint8_t* weight, const float* inputs, float* outputs,
    uint32_t rows, uint32_t cols, uint32_t tokens
) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x & 31U;
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * 4 + warp;
    if (row >= rows) return;
    const uint32_t first = blockIdx.y * 8;
    const uint32_t blocks_per_row = cols / 32;
    const uint8_t* source = weight + static_cast<size_t>(row) * blocks_per_row * 34;
    double partial[8] = {};
    #pragma unroll 4
    for (uint32_t block = 0; block < blocks_per_row; ++block) {
        const uint32_t scale_bits = *reinterpret_cast<const uint16_t*>(source);
        const float scale = augur_q8_0_f16_to_f32(scale_bits);
        const uint8_t encoded = source[2 + lane];
        const int32_t code = encoded < 128U ? static_cast<int32_t>(encoded) : static_cast<int32_t>(encoded) - 256;
        const double decoded = static_cast<double>(__fmul_rn(static_cast<float>(code), scale));
        #pragma unroll
        for (uint32_t t = 0; t < 8; ++t) {
            if (first + t < tokens) {
                const float x = inputs[static_cast<size_t>(first + t) * cols + block * 32 + lane];
                partial[t] += decoded * static_cast<double>(x);
            }
        }
        source += 34;
    }
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U) {
        #pragma unroll
        for (uint32_t t = 0; t < 8; ++t) {
            partial[t] += __shfl_down_sync(0xffffffffU, partial[t], offset);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (uint32_t t = 0; t < 8; ++t) {
            if (first + t < tokens) outputs[static_cast<size_t>(first + t) * rows + row] = static_cast<float>(partial[t]);
        }
    }
}

// Experimental output-row reuse. Each result keeps the frozen FP64 K order
// and warp reduction tree; an input conversion serves multiple output rows.
template<uint32_t RowReuse>
__global__ void augur_dflash_q8_rows_tile8_row_reuse_kernel_v1(
    const uint8_t* weight, const float* inputs, float* outputs,
    uint32_t rows, uint32_t cols, uint32_t tokens
) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x & 31U;
    const uint64_t first_row = (static_cast<uint64_t>(blockIdx.x) * 4 + warp) * RowReuse;
    if (first_row >= rows) return;
    const uint32_t first_token = blockIdx.y * 8;
    const uint32_t blocks_per_row = cols / 32;
    // This component admits complete output-row groups only.
    if (rows % (4 * RowReuse) != 0) return;
    const uint8_t* source[RowReuse];
    #pragma unroll
    for (uint32_t r = 0; r < RowReuse; ++r)
        source[r] = weight + (first_row + r) * blocks_per_row * 34;
    double partial[RowReuse][8] = {};
    #pragma unroll 4
    for (uint32_t block = 0; block < blocks_per_row; ++block) {
        double decoded[RowReuse];
        #pragma unroll
        for (uint32_t r = 0; r < RowReuse; ++r) {
                const float scale = augur_q8_0_f16_to_f32(*reinterpret_cast<const uint16_t*>(source[r]));
                const uint8_t encoded = source[r][2 + lane];
                const int32_t code = encoded < 128U ? static_cast<int32_t>(encoded) : static_cast<int32_t>(encoded) - 256;
                decoded[r] = static_cast<double>(__fmul_rn(static_cast<float>(code), scale));
                source[r] += 34;
        }
        #pragma unroll
        for (uint32_t t = 0; t < 8; ++t) {
            if (first_token + t < tokens) {
                const double x = static_cast<double>(inputs[static_cast<size_t>(first_token + t) * cols + block * 32 + lane]);
                #pragma unroll
                for (uint32_t r = 0; r < RowReuse; ++r) {
                    partial[r][t] += decoded[r] * x;
                }
            }
        }
    }
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U) {
        #pragma unroll
        for (uint32_t r = 0; r < RowReuse; ++r) {
            #pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                partial[r][t] += __shfl_down_sync(0xffffffffU, partial[r][t], offset);
            }
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (uint32_t r = 0; r < RowReuse; ++r) {
            #pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                if (first_row + r < rows && first_token + t < tokens)
                    outputs[static_cast<size_t>(first_token + t) * rows + first_row + r] = static_cast<float>(partial[r][t]);
            }
        }
    }
}

// Exact full-B8 specialization: no tail rows, unchanged FP64 accumulation.
template<uint32_t RowReuse>
__global__ void augur_dflash_q8_full8_exact_kernel_v1(
    const uint8_t* weight, const float* inputs, float* outputs,
    uint32_t rows, uint32_t cols, uint32_t tokens
) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x & 31U;
    const uint64_t first_row = (static_cast<uint64_t>(blockIdx.x) * 4 + warp) * RowReuse;
    if (first_row >= rows) return;
    constexpr uint32_t first_token = 0;
    const uint32_t blocks_per_row = cols / 32;
    // This component admits complete output-row groups only.
    if (rows % (4 * RowReuse) != 0) return;
    const uint8_t* source[RowReuse];
    #pragma unroll
    for (uint32_t r = 0; r < RowReuse; ++r)
        source[r] = weight + (first_row + r) * blocks_per_row * 34;
    double partial[RowReuse][8] = {};
    #pragma unroll 4
    for (uint32_t block = 0; block < blocks_per_row; ++block) {
        double decoded[RowReuse];
        #pragma unroll
        for (uint32_t r = 0; r < RowReuse; ++r) {
                const float scale = augur_q8_0_f16_to_f32(*reinterpret_cast<const uint16_t*>(source[r]));
                const uint8_t encoded = source[r][2 + lane];
                const int32_t code = encoded < 128U ? static_cast<int32_t>(encoded) : static_cast<int32_t>(encoded) - 256;
                decoded[r] = static_cast<double>(__fmul_rn(static_cast<float>(code), scale));
                source[r] += 34;
        }
        #pragma unroll
        for (uint32_t t = 0; t < 8; ++t) {
            if (true) {
                const double x = static_cast<double>(inputs[static_cast<size_t>(first_token + t) * cols + block * 32 + lane]);
                #pragma unroll
                for (uint32_t r = 0; r < RowReuse; ++r) {
                    partial[r][t] += decoded[r] * x;
                }
            }
        }
    }
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U) {
        #pragma unroll
        for (uint32_t r = 0; r < RowReuse; ++r) {
            #pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                partial[r][t] += __shfl_down_sync(0xffffffffU, partial[r][t], offset);
            }
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (uint32_t r = 0; r < RowReuse; ++r) {
            #pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                if (first_row + r < rows && true)
                    outputs[static_cast<size_t>(first_token + t) * rows + first_row + r] = static_cast<float>(partial[r][t]);
            }
        }
    }
}
template<uint32_t RowReuse,uint32_t Cols,uint32_t MinBlocks>
__global__ __launch_bounds__(128,MinBlocks) void augur_dflash_q8_static_b8_exact_kernel_v1(
    const uint8_t* weight, const float* inputs, float* outputs,
    uint32_t rows, uint32_t ignored_cols, uint32_t tokens
) {
    constexpr uint32_t cols = Cols;
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x & 31U;
    const uint64_t first_row = (static_cast<uint64_t>(blockIdx.x) * 4 + warp) * RowReuse;
    if (first_row >= rows) return;
    constexpr uint32_t first_token = 0;
    const uint32_t blocks_per_row = cols / 32;
    // This component admits complete output-row groups only.
    if (rows % (4 * RowReuse) != 0) return;
    const uint8_t* source[RowReuse];
    #pragma unroll
    for (uint32_t r = 0; r < RowReuse; ++r)
        source[r] = weight + (first_row + r) * blocks_per_row * 34;
    double partial[RowReuse][8] = {};
    #pragma unroll 4
    for (uint32_t block = 0; block < blocks_per_row; ++block) {
        double decoded[RowReuse];
        #pragma unroll
        for (uint32_t r = 0; r < RowReuse; ++r) {
                const float scale = augur_q8_0_f16_to_f32(*reinterpret_cast<const uint16_t*>(source[r]));
                const uint8_t encoded = source[r][2 + lane];
                const int32_t code = encoded < 128U ? static_cast<int32_t>(encoded) : static_cast<int32_t>(encoded) - 256;
                decoded[r] = static_cast<double>(__fmul_rn(static_cast<float>(code), scale));
                source[r] += 34;
        }
        #pragma unroll
        for (uint32_t t = 0; t < 8; ++t) {
            if (true) {
                const double x = static_cast<double>(inputs[static_cast<size_t>(first_token + t) * cols + block * 32 + lane]);
                #pragma unroll
                for (uint32_t r = 0; r < RowReuse; ++r) {
                    partial[r][t] += decoded[r] * x;
                }
            }
        }
    }
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0; offset >>= 1U) {
        #pragma unroll
        for (uint32_t r = 0; r < RowReuse; ++r) {
            #pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                partial[r][t] += __shfl_down_sync(0xffffffffU, partial[r][t], offset);
            }
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (uint32_t r = 0; r < RowReuse; ++r) {
            #pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                if (first_row + r < rows && true)
                    outputs[static_cast<size_t>(first_token + t) * rows + first_row + r] = static_cast<float>(partial[r][t]);
            }
        }
    }
}

__global__ void augur_dflash_rope_context_rows_kernel_v1(
    const float* inputs, float* outputs, uint32_t heads, uint64_t position, float theta
) {
    const uint32_t index = threadIdx.x;
    const float* input = inputs + static_cast<size_t>(blockIdx.x) * 128;
    float* output = outputs + static_cast<size_t>(blockIdx.x) * 128;
    const uint32_t half = 64;
    const uint32_t pair = index < half ? index : index - half;
    const double exponent = -2.0 * static_cast<double>(pair) / static_cast<double>(128);
    const float inverse_frequency = static_cast<float>(pow(static_cast<double>(theta), exponent));
    const float angle = static_cast<float>(position + blockIdx.x / heads) * inverse_frequency;
    float sine;
    float cosine;
    sincosf(angle, &sine, &cosine);
    const float mate = input[index < half ? index + half : index - half];
    const float value = input[index];
    output[index] = index < half ? value * cosine - mate * sine : mate * sine + value * cosine;
}

static bool augur_dflash_rows_range(uint64_t address, uint64_t bytes, uint32_t alignment) {
    return bytes <= SIZE_MAX && augur_q8_0_checked_range(address, bytes) && address % alignment == 0;
}

// Preserve the separately observable normalized scratch, not only the rotated
// output. Same 256-leaf norm tree and rounded FP32 boundary as the two kernels.
__global__ void augur_dflash_key_norm_rope_context_kernel_v1(
    const float* inputs, const float* weight, float* normalized, float* outputs,
    uint32_t heads, uint64_t position, float theta, float epsilon) {
    const uint32_t lane = threadIdx.x;
    const size_t first = static_cast<size_t>(blockIdx.x) * 128;
    const float* input = inputs + first;
    __shared__ float partial[256];
    __shared__ float normed[128];
    float sum = 0.0f;
    for (uint32_t index = lane; index < 128; index += blockDim.x) sum += input[index] * input[index];
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / 128.0f + epsilon);
    if (lane < 128) {
        const float value = input[lane] * scale * weight[lane];
        normed[lane] = value;
        normalized[first + lane] = value;
    }
    __syncthreads();
    if (lane < 128) {
        const uint32_t pair = lane < 64 ? lane : lane - 64;
        const double exponent = -2.0 * static_cast<double>(pair) / 128.0;
        const float inverse = static_cast<float>(pow(static_cast<double>(theta), exponent));
        const float angle = static_cast<float>(position + blockIdx.x / heads) * inverse;
        float sine, cosine;
        sincosf(angle, &sine, &cosine);
        const float mate = normed[lane < 64 ? lane + 64 : lane - 64];
        const float value = normed[lane];
        outputs[first + lane] = lane < 64 ? value * cosine - mate * sine : mate * sine + value * cosine;
    }
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_history_reuse_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q8_0_f32_matvec_args_v1* a, uint32_t tokens
) {
    uint64_t w = 0, i = 0, o = 0;
    if (stream == nullptr || a == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || (tokens != 512 && tokens != 2048)
        || !((a->rows == 5120 && a->cols == 25600) || (a->rows == 1024 && a->cols == 5120))
        || !augur_q8_0_checked_footprint(a->rows, a->cols, &w, &i, &o))
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    i *= tokens; o *= tokens;
    if (a->abi_version != AUGUR_CUDA_Q8_0_F32_MATVEC_ARGS_VERSION_V1 || a->struct_size != sizeof(*a)
        || a->weight_len_bytes != w || a->input_len_bytes != i || a->output_len_bytes != o
        || !augur_dflash_rows_range(a->weight,w,2) || !augur_dflash_rows_range(a->input,i,4) || !augur_dflash_rows_range(a->output,o,4)
        || !augur_q8_0_ranges_disjoint(a->weight,w,a->input,i) || !augur_q8_0_ranges_disjoint(a->weight,w,a->output,o)
        || !augur_q8_0_ranges_disjoint(a->input,i,a->output,o))
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    augur_dflash_q8_history_row4_kernel_v1<<<dim3(a->rows / 16,tokens / 4),128,0,stream->stream>>>(
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),a->rows,a->cols,tokens);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
}

static augur_cuda_result_v1 augur_dflash_q8_history_panel_impl(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q8_0_f32_matvec_args_v1* a,
    uint32_t tokens, uint64_t scratch, uint64_t scratch_bytes, bool token8, bool fc_slice = false
) {
    uint64_t w = 0, i = 0, o = 0;
    if (stream == nullptr || a == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || (fc_slice ? (tokens == 0 || tokens > 512 || a->rows != 5120 || a->cols != 25600)
                     : (tokens != 512 && tokens != 2048))
        || !((a->rows == 5120 && a->cols == 25600) || (a->rows == 1024 && a->cols == 5120))
        || !augur_q8_0_checked_footprint(a->rows, a->cols, &w, &i, &o))
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    i *= tokens; o *= tokens;
    const uint32_t panel_rows = a->cols == 25600 ? 128 : 256;
    const uint64_t panel_bytes = uint64_t(panel_rows) * a->cols * sizeof(float);
    if (a->abi_version != AUGUR_CUDA_Q8_0_F32_MATVEC_ARGS_VERSION_V1 || a->struct_size != sizeof(*a)
        || a->weight_len_bytes != w || a->input_len_bytes != i || a->output_len_bytes != o
        || scratch_bytes != panel_bytes || !augur_dflash_rows_range(scratch, panel_bytes, 4)
        || !augur_dflash_rows_range(a->weight,w,2) || !augur_dflash_rows_range(a->input,i,4) || !augur_dflash_rows_range(a->output,o,4)
        || !augur_q8_0_ranges_disjoint(a->weight,w,a->input,i) || !augur_q8_0_ranges_disjoint(a->weight,w,a->output,o)
        || !augur_q8_0_ranges_disjoint(a->input,i,a->output,o)
        || !augur_q8_0_ranges_disjoint(scratch,panel_bytes,a->weight,w)
        || !augur_q8_0_ranges_disjoint(scratch,panel_bytes,a->input,i)
        || !augur_q8_0_ranges_disjoint(scratch,panel_bytes,a->output,o))
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    float* panel = reinterpret_cast<float*>(static_cast<uintptr_t>(scratch));
    for (uint32_t first = 0; first < a->rows; first += panel_rows) {
        augur_dflash_history_decode_panel_kernel_v1<<<(size_t(panel_rows)*a->cols+255)/256,256,0,stream->stream>>>(
            reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a->weight)),panel,first,panel_rows,a->cols);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
        if (fc_slice && tokens%8) augur_dflash_history_fc_slice_kernel_v1<<<dim3(panel_rows/16,(tokens+7)/8),128,0,stream->stream>>>(
            panel,reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),a->rows,a->cols,tokens,first,panel_rows);
        else if (token8) augur_dflash_history_panel_token8_kernel_v1<<<dim3(panel_rows/16,tokens/8),128,0,stream->stream>>>(
            panel,reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),a->rows,a->cols,tokens,first,panel_rows);
        else augur_dflash_history_panel_dot_kernel_v1<<<dim3(panel_rows/16,tokens/4),128,0,stream->stream>>>(
            panel,reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),a->rows,a->cols,tokens,first,panel_rows);
        error = cudaGetLastError();
        if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
    }
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_history_panel_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q8_0_f32_matvec_args_v1* a,
    uint32_t tokens, uint64_t scratch, uint64_t scratch_bytes) {
    return augur_dflash_q8_history_panel_impl(ordinal,stream,a,tokens,scratch,scratch_bytes,false);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_history_panel_token8_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q8_0_f32_matvec_args_v1* a,
    uint32_t tokens, uint64_t scratch, uint64_t scratch_bytes) {
    return augur_dflash_q8_history_panel_impl(ordinal,stream,a,tokens,scratch,scratch_bytes,true);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_history_fc_slice_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q8_0_f32_matvec_args_v1* a,
    uint32_t tokens, uint64_t scratch, uint64_t scratch_bytes) {
    return augur_dflash_q8_history_panel_impl(ordinal,stream,a,tokens,scratch,scratch_bytes,true,true);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_key_norm_rope_context_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_rmsnorm_f32_args_v1* n,
    const augur_cuda_dflash_rope_f32_args_v1* r, uint32_t tokens, uint32_t heads
) {
    if (stream == nullptr || n == nullptr || r == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || tokens == 0 || tokens > 2048 || heads != 8)
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    const uint64_t bytes = static_cast<uint64_t>(tokens) * heads * 128 * 4;
    float eps; memcpy(&eps,&n->epsilon_bits,4);
    if (n->abi_version != AUGUR_CUDA_RMSNORM_F32_ARGS_VERSION_V1 || n->struct_size != sizeof(*n)
        || r->abi_version != AUGUR_CUDA_DFLASH_ROPE_F32_ARGS_VERSION_V1 || r->struct_size != sizeof(*r)
        || n->elements != 128 || r->elements != 128 || r->dimensions != 128 || r->rotary_dimensions != 128
        || !isfinite(eps) || eps <= 0 || !isfinite(r->theta) || r->theta <= 1
        || r->position > 262144U - tokens || n->input_len_bytes != bytes || n->output_len_bytes != bytes
        || n->weight_len_bytes != 512 || r->input_len_bytes != bytes || r->output_len_bytes != bytes
        || r->input != n->output
        || !augur_dflash_rows_range(n->input,bytes,4) || !augur_dflash_rows_range(n->weight,512,4)
        || !augur_dflash_rows_range(n->output,bytes,4) || !augur_dflash_rows_range(r->output,bytes,4)
        || !augur_q8_0_ranges_disjoint(n->input,bytes,n->output,bytes)
        || !augur_q8_0_ranges_disjoint(n->input,bytes,r->output,bytes)
        || !augur_q8_0_ranges_disjoint(n->weight,512,n->output,bytes)
        || !augur_q8_0_ranges_disjoint(n->weight,512,r->output,bytes)
        || !augur_q8_0_ranges_disjoint(n->output,bytes,r->output,bytes))
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    augur_dflash_key_norm_rope_context_kernel_v1<<<tokens*heads,256,0,stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(n->input)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(n->weight)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(n->output)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(r->output)),heads,r->position,r->theta,eps);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_context_rows_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q8_0_f32_matvec_args_v1* a, uint32_t tokens
) {
    uint64_t w = 0, i = 0, o = 0;
    if (stream == nullptr || a == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || tokens == 0 || tokens > 2048 || a->rows > 65536 || a->cols > 65536
        || !augur_q8_0_checked_footprint(a->rows, a->cols, &w, &i, &o))
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    i *= tokens; o *= tokens;
    if (a->abi_version != AUGUR_CUDA_Q8_0_F32_MATVEC_ARGS_VERSION_V1 || a->struct_size != sizeof(*a)
        || a->weight_len_bytes != w || a->input_len_bytes != i || a->output_len_bytes != o
        || !augur_dflash_rows_range(a->weight, w, 2) || !augur_dflash_rows_range(a->input, i, 4) || !augur_dflash_rows_range(a->output, o, 4)
        || !augur_q8_0_ranges_disjoint(a->weight,w,a->input,i) || !augur_q8_0_ranges_disjoint(a->weight,w,a->output,o)
        || !augur_q8_0_ranges_disjoint(a->input,i,a->output,o))
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    augur_dflash_q8_context_rows_kernel_v1<<<dim3((a->rows + 3)/4,tokens),128,0,stream->stream>>>(
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),a->rows,a->cols);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_norm_context_rows_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_rmsnorm_f32_args_v1* a, uint32_t rows
) {
    if (stream == nullptr || a == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || rows == 0 || rows > 16384 || a->elements == 0 || a->elements > 65536)
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    float eps; std::memcpy(&eps,&a->epsilon_bits,sizeof(eps));
    const uint64_t w = static_cast<uint64_t>(a->elements)*4, bytes = w*rows;
    if (a->abi_version != AUGUR_CUDA_RMSNORM_F32_ARGS_VERSION_V1 || a->struct_size != sizeof(*a)
        || !isfinite(eps) || eps <= 0 || a->input_len_bytes != bytes || a->output_len_bytes != bytes || a->weight_len_bytes != w
        || !augur_dflash_rows_range(a->input,bytes,4) || !augur_dflash_rows_range(a->output,bytes,4) || !augur_dflash_rows_range(a->weight,w,4)
        || !augur_q8_0_ranges_disjoint(a->input,bytes,a->output,bytes) || !augur_q8_0_ranges_disjoint(a->weight,w,a->output,bytes))
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    augur_dflash_norm_context_rows_kernel_v1<<<rows,256,0,stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->weight)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),a->elements,eps);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
}

static augur_cuda_result_v1 augur_dflash_q8_rows_tile4_submit_impl(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q8_0_f32_matvec_args_v1* a, uint32_t tokens,
    bool specialize_b8
) {
    uint64_t w = 0, i = 0, o = 0;
    if (stream == nullptr || a == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || tokens == 0 || tokens > 2048 || a->rows > 65536 || a->cols > 65536
        || !augur_q8_0_checked_footprint(a->rows, a->cols, &w, &i, &o))
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    i *= tokens; o *= tokens;
    if (a->abi_version != AUGUR_CUDA_Q8_0_F32_MATVEC_ARGS_VERSION_V1 || a->struct_size != sizeof(*a)
        || a->weight_len_bytes != w || a->input_len_bytes != i || a->output_len_bytes != o
        || !augur_dflash_rows_range(a->weight,w,2) || !augur_dflash_rows_range(a->input,i,4) || !augur_dflash_rows_range(a->output,o,4)
        || !augur_q8_0_ranges_disjoint(a->weight,w,a->input,i) || !augur_q8_0_ranges_disjoint(a->weight,w,a->output,o)
        || !augur_q8_0_ranges_disjoint(a->input,i,a->output,o))
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    const bool specialized_shape = tokens == 8 &&
        ((a->cols == 5120 && (a->rows == 4096 || a->rows == 1024 || a->rows == 17408))
         || (a->cols == 17408 && a->rows == 5120));
    if (specialize_b8 && !specialized_shape)
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    // Keep the short-draft admission. Independently measured full history
    // FC/KV512/2048 uses exact weight materialization and direct scale loads,
    // retaining four-token reuse and FP64 order. Other sizes/shapes retain
    // tile4. Only full8 short shapes reuse weights over all8 tokens;
    // the previously rejected history8/16-row schedule is not selected.
    const bool short_unroll = tokens <= 8 &&
        ((a->cols == 5120 && (a->rows == 4096 || a->rows == 1024
            || a->rows == 1280 || a->rows == 17408))
         || (a->cols == 17408 && a->rows == 5120));
    const bool history_unroll = (tokens == 512 || tokens == 2048) &&
        ((a->cols == 25600 && a->rows == 5120)
         || (a->cols == 5120 && a->rows == 1024));
    // The measured two-output-row tile reuses input loads/conversions.
    // Conv1280 regressed. Exactly8 newly appended history KV rows can also
    // select row2; bulk history512/2048 and token tails use separate paths.
    const bool row2_full8 = tokens == 8 &&
        ((a->cols == 5120 && (a->rows == 4096 || a->rows == 1024 || a->rows == 17408))
         || (a->cols == 17408 && a->rows == 5120));
    // Wider row reuse helps full Q and FFN, but harms the 1024-row K/V
    // projection. Keep that shape on row2 and retain every FP64 K chain.
    if (specialize_b8 && a->cols == 5120 && a->rows != 1024) {
        augur_dflash_q8_static_b8_exact_kernel_v1<4,5120,4><<<dim3(a->rows/16,1),128,0,stream->stream>>>(
            reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a->weight)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),a->rows,a->cols,tokens);
    } else if (specialize_b8 && a->cols == 5120) {
        augur_dflash_q8_static_b8_exact_kernel_v1<2,5120,4><<<dim3(a->rows/8,1),128,0,stream->stream>>>(
            reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a->weight)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),a->rows,a->cols,tokens);
    } else if (specialize_b8) {
        augur_dflash_q8_static_b8_exact_kernel_v1<4,17408,2><<<dim3(a->rows/16,1),128,0,stream->stream>>>(
            reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a->weight)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),a->rows,a->cols,tokens);
    } else if (row2_full8) {
        augur_dflash_q8_rows_tile8_row_reuse_kernel_v1<2><<<dim3((a->rows + 7)/8,1),128,0,stream->stream>>>(
            reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a->weight)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),a->rows,a->cols,tokens);
    } else if (short_unroll && tokens == 8) {
        augur_dflash_q8_rows_tile8_exact_dequant_kernel_v1<<<dim3((a->rows + 3)/4,1),128,0,stream->stream>>>(
            reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a->weight)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),a->rows,a->cols,tokens);
    } else if (history_unroll) {
        augur_dflash_q8_rows_tile4_exact_dequant8_kernel_v1<<<dim3((a->rows + 3)/4,(tokens + 3)/4),128,0,stream->stream>>>(
            reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a->weight)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),a->rows,a->cols,tokens);
    } else if (short_unroll) {
        augur_dflash_q8_rows_tile4_short_unroll4_kernel_v1<<<dim3((a->rows + 3)/4,(tokens + 3)/4),128,0,stream->stream>>>(
            reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a->weight)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),a->rows,a->cols,tokens);
    } else {
        augur_dflash_q8_rows_tile4_kernel_v1<<<dim3((a->rows + 3)/4,(tokens + 3)/4),128,0,stream->stream>>>(
            reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a->weight)),
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),a->rows,a->cols,tokens);
    }
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_rows_tile4_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q8_0_f32_matvec_args_v1* a, uint32_t tokens
) { return augur_dflash_q8_rows_tile4_submit_impl(ordinal,stream,a,tokens,false); }

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_rows_b8_specialized_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_q8_0_f32_matvec_args_v1* a, uint32_t tokens
) { return augur_dflash_q8_rows_tile4_submit_impl(ordinal,stream,a,tokens,true); }

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_rope_context_rows_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_dflash_rope_f32_args_v1* a, uint32_t tokens, uint32_t heads
) {
    if (stream == nullptr || a == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || tokens == 0 || tokens > 2048 || heads == 0 || heads > 32)
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    const uint64_t bytes = static_cast<uint64_t>(tokens)*heads*128*4;
    if (a->abi_version != AUGUR_CUDA_DFLASH_ROPE_F32_ARGS_VERSION_V1 || a->struct_size != sizeof(*a)
        || a->elements != 128 || a->dimensions != 128 || a->rotary_dimensions != 128
        || a->position > 262144-tokens || !isfinite(a->theta) || a->theta <= 1
        || a->input_len_bytes != bytes || a->output_len_bytes != bytes
        || !augur_dflash_rows_range(a->input,bytes,4) || !augur_dflash_rows_range(a->output,bytes,4)
        || !augur_q8_0_ranges_disjoint(a->input,bytes,a->output,bytes))
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    augur_dflash_rope_context_rows_kernel_v1<<<tokens*heads,128,0,stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),heads,a->position,a->theta);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);
}

__global__ void augur_dflash_cache_attention_f32_kernel_v1(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    float* output,
    uint32_t capacity_positions,
    uint32_t visible_position_start,
    uint32_t visible_position_end,
    uint32_t query_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    float scale
) {
    const uint32_t query_head = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (query_head >= query_heads || lane >= head_dim) return;
    const uint32_t query_per_kv = query_heads / kv_heads;
    const uint32_t kv_head = query_head / query_per_kv;
    extern __shared__ float shared[];
    float* dot = shared;
    float* scores = shared + head_dim;
    const uint64_t query_base = static_cast<uint64_t>(query_head) * head_dim;
    for (uint32_t position = visible_position_start;
         position < visible_position_end;
         ++position) {
        const uint64_t cache_base =
            (static_cast<uint64_t>(position % capacity_positions) * kv_heads + kv_head) * head_dim;
        dot[lane] = query[query_base + lane] * key_cache[cache_base + lane];
        __syncthreads();
        for (uint32_t stride = head_dim / 2; stride > 0; stride >>= 1) {
            if (lane < stride) dot[lane] += dot[lane + stride];
            __syncthreads();
        }
        if (lane == 0) scores[position - visible_position_start] = dot[0] * scale;
        __syncthreads();
    }
    if (lane == 0) {
        float maximum = scores[0];
        for (uint32_t index = 0; index < visible_position_end - visible_position_start; ++index) {
            maximum = fmaxf(maximum, scores[index]);
        }
        float denominator = 0.0f;
        for (uint32_t index = 0; index < visible_position_end - visible_position_start; ++index) {
            const float weight = expf(scores[index] - maximum);
            scores[index] = weight;
            denominator += weight;
        }
        const float inverse = 1.0f / denominator;
        for (uint32_t index = 0; index < visible_position_end - visible_position_start; ++index) {
            scores[index] *= inverse;
        }
    }
    __syncthreads();
    float result = 0.0f;
    for (uint32_t index = 0; index < visible_position_end - visible_position_start; ++index) {
        const uint32_t position = visible_position_start + index;
        const uint64_t cache_base =
            (static_cast<uint64_t>(position % capacity_positions) * kv_heads + kv_head) * head_dim;
        result += scores[index] * value_cache[cache_base + lane];
    }
    output[query_base + lane] = result;
}

static augur_cuda_result_v1 augur_validate_dflash_cache_attention_f32(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_dflash_cache_attention_f32_args_v1* args
) {
    const uint64_t query_bytes = 32ULL * 128ULL * sizeof(float);
    const uint64_t cache_row_bytes = 8ULL * 128ULL * sizeof(float);
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_DFLASH_CACHE_ATTENTION_F32_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_dflash_cache_attention_f32_args_v1)
        || args->query == 0 || args->key_cache == 0 || args->value_cache == 0 || args->output == 0
        || args->reserved != 0 || args->capacity_positions == 0 || args->capacity_positions > 2048
        || args->visible_position_start >= args->visible_position_end
        || args->visible_position_end - args->visible_position_start > args->capacity_positions
        || args->query_heads != 32 || args->kv_heads != 8 || args->head_dim != 128
        || !isfinite(args->scale) || args->scale <= 0.0f
        || args->query_len_bytes != query_bytes || args->output_len_bytes != query_bytes
        || args->key_cache_len_bytes != static_cast<uint64_t>(args->capacity_positions) * cache_row_bytes
        || args->value_cache_len_bytes != static_cast<uint64_t>(args->capacity_positions) * cache_row_bytes
        || args->query % alignof(float) != 0 || args->key_cache % alignof(float) != 0
        || args->value_cache % alignof(float) != 0 || args->output % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (!augur_q8_0_checked_range(args->query, args->query_len_bytes)
        || !augur_q8_0_checked_range(args->key_cache, args->key_cache_len_bytes)
        || !augur_q8_0_checked_range(args->value_cache, args->value_cache_len_bytes)
        || !augur_q8_0_checked_range(args->output, args->output_len_bytes)
        || !augur_q8_0_ranges_disjoint(args->output, args->output_len_bytes, args->query, args->query_len_bytes)
        || !augur_q8_0_ranges_disjoint(args->output, args->output_len_bytes, args->key_cache, args->key_cache_len_bytes)
        || !augur_q8_0_ranges_disjoint(args->output, args->output_len_bytes, args->value_cache, args->value_cache_len_bytes)) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_set_device(ordinal);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_cache_attention_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_cache_attention_f32_args_v1* args
) {
    augur_cuda_result_v1 valid = augur_validate_dflash_cache_attention_f32(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    constexpr uint32_t head_dim = 128;
    const size_t shared_bytes = (head_dim + 2048) * sizeof(float);
    augur_dflash_cache_attention_f32_kernel_v1<<<32, head_dim, shared_bytes, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->key_cache)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->value_cache)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->capacity_positions,
        args->visible_position_start,
        args->visible_position_end,
        args->query_heads,
        args->kv_heads,
        args->head_dim,
        args->scale
    );
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_cache_attention_f32_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_cache_attention_f32_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_dflash_cache_attention_f32_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

static_assert(sizeof(augur_cuda_dflash_block_attention_f32_args_v1) == 112,
    "DFlash block-attention ABI size changed");

// Four warps compute four keys concurrently. The complete draft block is
// present in dense K/V; do not apply a causal right mask or ring modulo here.
__global__ void augur_dflash_block_attention_f32_kernel_v1(
    const float* query, const float* key, const float* value, float* output,
    uint32_t context_len, uint32_t sliding_window, float scale
) {
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t query_token = blockIdx.x / 32u, head = blockIdx.x % 32u;
    const uint32_t kv_head = head / 4u, total = context_len + 8u;
    const uint32_t query_row = context_len + query_token;
    const uint32_t lower = query_row + 1u > sliding_window ? query_row + 1u - sliding_window : 0u;
    const uint32_t q_base = (query_token * 32u + head) * 128u;
    __shared__ float scores[2056];
    __shared__ float reduce[128];
    __shared__ int invalid;
    if (tid == 0) invalid = 0;
    __syncthreads();
    float q[4];
    #pragma unroll
    for (uint32_t j = 0; j < 4; ++j) q[j] = query[q_base + lane + 32u * j];
    for (uint32_t row = warp; row < total; row += 4u) {
        float dot = 0.0f;
        bool bad = false;
        #pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            const float k = key[(row * 8u + kv_head) * 128u + lane + 32u * j];
            bad |= !isfinite(q[j]) || !isfinite(k);
            dot = __fadd_rn(dot, __fmul_rn(q[j], k));
        }
        if (__any_sync(0xffffffffu, bad)) atomicExch(&invalid, 1);
        #pragma unroll
        for (uint32_t stride = 16; stride > 0; stride >>= 1)
            dot = __fadd_rn(dot, __shfl_down_sync(0xffffffffu, dot, stride));
        if (lane == 0) {
            const float s = __fmul_rn(dot, scale);
            if (!isfinite(s)) atomicExch(&invalid, 1);
            scores[row] = row >= lower ? s : -__int_as_float(0x7f800000);
        }
    }
    __syncthreads();
    float m = -__int_as_float(0x7f800000);
    for (uint32_t row = tid; row < total; row += 128u) m = fmaxf(m, scores[row]);
    reduce[tid] = m;
    __syncthreads();
    for (uint32_t stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
        __syncthreads();
    }
    const float maximum = reduce[0];
    float sum = 0.0f;
    for (uint32_t row = tid; row < total; row += 128u) {
        const float p = row >= lower ? expf(scores[row] - maximum) : 0.0f;
        scores[row] = p;
        sum = __fadd_rn(sum, p);
    }
    // Every thread consumed the previous reduction before reusing it.
    __syncthreads();
    reduce[tid] = sum;
    __syncthreads();
    for (uint32_t stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] = __fadd_rn(reduce[tid], reduce[tid + stride]);
        __syncthreads();
    }
    const float denominator = reduce[0];
    if (!isfinite(denominator) || denominator <= 0.0f) atomicExch(&invalid, 1);
    // Reuse each rounded probability across all 128 value dimensions.
    // Preserve the division and the original serial PV accumulation order.
    for (uint32_t row = tid; row < total; row += 128u) {
        scores[row] = scores[row] / denominator;
    }
    __syncthreads();
    float acc = 0.0f;
    for (uint32_t row = 0; row < total; ++row) {
        const float v = value[(row * 8u + kv_head) * 128u + tid];
        if (!isfinite(v)) atomicExch(&invalid, 1);
        acc = __fadd_rn(acc, __fmul_rn(scores[row], v));
    }
    if (!isfinite(acc)) atomicExch(&invalid, 1);
    __syncthreads();
    // Poison the whole query/head, never silently turn invalid math into zero.
    output[q_base + tid] = invalid ? __int_as_float(0x7fc00000) : acc;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_block_attention_f32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_dflash_block_attention_f32_args_v1* a
) {
    if (stream == nullptr || a == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || a->abi_version != AUGUR_CUDA_DFLASH_BLOCK_ATTENTION_F32_ARGS_VERSION_V1
        || a->struct_size != sizeof(augur_cuda_dflash_block_attention_f32_args_v1)
        || a->reserved != 0 || a->context_len > 2048 || a->sliding_window == 0 || a->sliding_window > 2048
        || static_cast<uint64_t>(a->context_start) + a->context_len != a->block_start
        || static_cast<uint64_t>(a->block_start) + 8u > 262144u
        || a->block_size != 8 || a->query_heads != 32 || a->kv_heads != 8 || a->head_dim != 128
        || !isfinite(a->scale) || a->scale <= 0.0f
        || a->query_len_bytes != 8ULL * 32 * 128 * sizeof(float)
        || a->output_len_bytes != a->query_len_bytes
        || a->key_len_bytes != (static_cast<uint64_t>(a->context_len) + 8) * 8 * 128 * sizeof(float)
        || a->value_len_bytes != a->key_len_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t ptrs[4] = {a->query, a->key, a->value, a->output};
    const uint64_t lens[4] = {a->query_len_bytes, a->key_len_bytes, a->value_len_bytes, a->output_len_bytes};
    for (uint32_t i = 0; i < 4; ++i) {
        if (ptrs[i] == 0 || ptrs[i] % alignof(float) != 0 || !augur_q8_0_checked_range(ptrs[i], lens[i]))
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        for (uint32_t j = 0; j < i; ++j)
            if (!augur_q8_0_ranges_disjoint(ptrs[i], lens[i], ptrs[j], lens[j]))
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 valid = augur_set_device(ordinal);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    augur_dflash_block_attention_f32_kernel_v1<<<256, 128, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->query)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->key)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(a->value)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(a->output)),
        a->context_len, a->sliding_window, a->scale);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

__device__ float augur_f8e4m3_to_f32(uint8_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x80) << 24;
    const uint32_t exponent = (value >> 3) & 0x0f;
    const uint32_t mantissa = value & 0x07;
    if (exponent == 0) {
        const float magnitude = static_cast<float>(mantissa) / 8.0f * exp2f(-6.0f);
        return (value & 0x80) ? -magnitude : magnitude;
    }
    if (exponent == 0x0f && mantissa == 0x07) return __int_as_float(0x7fc00000);
    const int32_t unbiased = static_cast<int32_t>(exponent) - 7;
    const float magnitude = (1.0f + static_cast<float>(mantissa) / 8.0f) * exp2f(static_cast<float>(unbiased));
    return (value & 0x80) ? -magnitude : magnitude;
}

__global__ void augur_f8e4m3_f32_matvec_kernel_v1(
    const uint8_t* weight, const float* scales, const float* input, float* output,
    uint32_t rows, uint32_t cols
) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (uint32_t col = 0; col < cols; ++col) {
        const float scale = scales[(row / 128) * (cols / 128) + (col / 128)];
        sum += augur_f8e4m3_to_f32(weight[static_cast<size_t>(row) * cols + col]) * scale * input[col];
    }
    output[row] = sum;
}

__device__ __forceinline__ float augur_f8e4m3_to_f32_bits_exact(uint8_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x80) << 24;
    const uint32_t exponent = (value >> 3) & 0x0f;
    const uint32_t mantissa = value & 0x07;
    if (exponent == 0x0f && mantissa == 0x07) {
        return __int_as_float(0x7fc00000);
    }
    if (exponent == 0) {
        if (mantissa == 0) {
            return __int_as_float(sign);
        }
        const float magnitude = static_cast<float>(mantissa) * 0x1p-9f;
        return __int_as_float(__float_as_uint(magnitude) | sign);
    }
    const uint32_t fp32_exponent = exponent + 120;
    return __int_as_float(sign | (fp32_exponent << 23) | (mantissa << 20));
}

__global__ void augur_f8e4m3_f32_matvec_bitdecode_kernel_v1(
    const uint8_t* weight, const float* scales, const float* input, float* output,
    uint32_t rows, uint32_t cols
) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (uint32_t col = 0; col < cols; ++col) {
        const float scale = scales[(row / 128) * (cols / 128) + (col / 128)];
        sum += augur_f8e4m3_to_f32_bits_exact(weight[static_cast<size_t>(row) * cols + col]) * scale * input[col];
    }
    output[row] = sum;
}

__global__ void augur_f8e4m3_f32_matvec_warp_fp32_kernel_v1(
    const uint8_t* weight, const float* scales, const float* input, float* output,
    uint32_t rows, uint32_t cols
) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t row = blockIdx.x * 4 + warp;
    if (row >= rows) return;

    const uint8_t* weight_row = weight + static_cast<size_t>(row) * cols;
    const float* scale_row = scales + static_cast<size_t>(row / 128) * (cols / 128);
    float partial = 0.0f;
    for (uint32_t scale_col = 0; scale_col < cols / 128; ++scale_col) {
        const uint32_t col_base = scale_col * 128;
        const uint32_t packed = reinterpret_cast<const uint32_t*>(weight_row + col_base)[lane];
        const float scale = scale_row[scale_col];
        #pragma unroll
        for (uint32_t item = 0; item < 4; ++item) {
            const uint8_t encoded = static_cast<uint8_t>(packed >> (item * 8));
            const float decoded = __fmul_rn(augur_f8e4m3_to_f32_bits_exact(encoded), scale);
            partial = __fmaf_rn(decoded, input[col_base + lane * 4 + item], partial);
        }
    }
    partial = __fadd_rn(partial, __shfl_down_sync(0xffffffffU, partial, 16));
    partial = __fadd_rn(partial, __shfl_down_sync(0xffffffffU, partial, 8));
    partial = __fadd_rn(partial, __shfl_down_sync(0xffffffffU, partial, 4));
    partial = __fadd_rn(partial, __shfl_down_sync(0xffffffffU, partial, 2));
    partial = __fadd_rn(partial, __shfl_down_sync(0xffffffffU, partial, 1));
    if (lane == 0) output[row] = partial;
}

#if defined(AUGUR_CUDA_P0_PREFILL_V22_PROJECTION_RESUME_BATCH_EXACT)
__global__ void augur_p0_prefill_v22_f8e4m3_batch4_kernel_v1(
    const uint8_t* weight,
    const float* scales,
    const float* input,
    float* output,
    uint32_t rows,
    uint32_t cols
) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t row = blockIdx.x * 4 + warp;
    const uint32_t token_base = blockIdx.y * AUGUR_P0_PREFILL_V22_TOKENS_PER_WARP;
    if (row >= rows) return;

    const uint8_t* weight_row = weight + static_cast<size_t>(row) * cols;
    const float* scale_row = scales + static_cast<size_t>(row / 128) * (cols / 128);
    float partial[AUGUR_P0_PREFILL_V22_TOKENS_PER_WARP] = {};
    for (uint32_t scale_col = 0; scale_col < cols / 128; ++scale_col) {
        const uint32_t col_base = scale_col * 128;
        const uint32_t packed = reinterpret_cast<const uint32_t*>(weight_row + col_base)[lane];
        const float scale = scale_row[scale_col];
        #pragma unroll
        for (uint32_t item = 0; item < 4; ++item) {
            const uint8_t encoded = static_cast<uint8_t>(packed >> (item * 8));
            const float decoded = __fmul_rn(augur_f8e4m3_to_f32_bits_exact(encoded), scale);
            const uint32_t col = col_base + lane * 4 + item;
            #pragma unroll
            for (uint32_t token = 0; token < AUGUR_P0_PREFILL_V22_TOKENS_PER_WARP; ++token) {
                partial[token] = __fmaf_rn(
                    decoded,
                    input[static_cast<size_t>(token_base + token) * cols + col],
                    partial[token]);
            }
        }
    }
    #pragma unroll
    for (uint32_t offset = 16; offset != 0; offset >>= 1) {
        #pragma unroll
        for (uint32_t token = 0; token < AUGUR_P0_PREFILL_V22_TOKENS_PER_WARP; ++token) {
            partial[token] = __fadd_rn(
                partial[token],
                __shfl_down_sync(0xffffffffU, partial[token], offset));
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (uint32_t token = 0; token < AUGUR_P0_PREFILL_V22_TOKENS_PER_WARP; ++token) {
            output[static_cast<size_t>(token_base + token) * rows + row] = partial[token];
        }
    }
}
#endif

__global__ void augur_f8e4m3_f32_matvec_row32_kernel_v1(
    const uint8_t* weight, const float* scales, const float* input, float* output,
    uint32_t rows, uint32_t cols
) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint32_t row_tile = row / 32;
    const uint32_t row_lane = row % 32;
    const size_t tile_base = static_cast<size_t>(row_tile) * cols * 32;
    float sum = 0.0f;
    for (uint32_t col = 0; col < cols; ++col) {
        const float scale = scales[(row / 128) * (cols / 128) + (col / 128)];
        const uint8_t encoded = weight[tile_base + static_cast<size_t>(col) * 32 + row_lane];
        sum += augur_f8e4m3_to_f32_bits_exact(encoded) * scale * input[col];
    }
    output[row] = sum;
}


__global__ void augur_f8e4m3_f32_matvec_row_scale_kernel_v1(
    const uint8_t* weight, const float* row_scales, const float* input, float* output,
    uint32_t rows, uint32_t cols
);
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_row_scale_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_f8e4m3_f32_matvec_args_v1* args
) {
    const uint64_t weight_bytes = args == nullptr ? 0 : static_cast<uint64_t>(args->rows) * args->cols;
    const uint64_t scale_bytes = args == nullptr ? 0 : static_cast<uint64_t>(args->rows) * sizeof(float);
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_F8E4M3_F32_MATVEC_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_f8e4m3_f32_matvec_args_v1)
        || args->weight == 0 || args->scales == 0 || args->input == 0 || args->output == 0
        || args->rows == 0 || args->cols == 0 || args->rows % 128 != 0 || args->cols % 128 != 0
        || args->weight_len_bytes != weight_bytes || args->scales_len_bytes < scale_bytes
        || args->input_len_bytes < static_cast<uint64_t>(args->cols) * sizeof(float)
        || args->output_len_bytes < static_cast<uint64_t>(args->rows) * sizeof(float)
        || args->scales % alignof(float) != 0 || args->input % alignof(float) != 0
        || args->output % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    constexpr uint32_t threads = 128;
    const uint32_t blocks = (args->rows + threads - 1) / threads;
    augur_f8e4m3_f32_matvec_row_scale_kernel_v1<<<blocks, threads, 0, stream->stream>>>(
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->scales)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->rows, args->cols);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_row_scale_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_f8e4m3_f32_matvec_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_f8e4m3_f32_matvec_row_scale_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}
__global__ void augur_f8e4m3_f32_matvec_row_scale_kernel_v1(
    const uint8_t* weight, const float* row_scales, const float* input, float* output,
    uint32_t rows, uint32_t cols
) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float scale = row_scales[row];
    float sum = 0.0f;
    const uint8_t* weight_row = weight + static_cast<size_t>(row) * cols;
    for (uint32_t col = 0; col < cols; ++col) {
        sum += augur_f8e4m3_to_f32_bits_exact(weight_row[col]) * scale * input[col];
    }
    output[row] = sum;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_f8e4m3_f32_matvec_args_v1* args
) {
    const uint64_t weight_bytes = args == nullptr ? 0 : static_cast<uint64_t>(args->rows) * args->cols;
    const uint64_t scale_count = args == nullptr ? 0 : static_cast<uint64_t>(args->rows / 128) * (args->cols / 128);
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_F8E4M3_F32_MATVEC_ARGS_VERSION_V1 || args->struct_size != sizeof(augur_cuda_f8e4m3_f32_matvec_args_v1)
        || args->weight == 0 || args->scales == 0 || args->input == 0 || args->output == 0 || args->rows == 0 || args->cols == 0
        || args->rows % 128 != 0 || args->cols % 128 != 0 || args->weight_len_bytes != weight_bytes
        || args->scales_len_bytes < scale_count * 4 || args->input_len_bytes < static_cast<uint64_t>(args->cols) * 4
        || args->output_len_bytes < static_cast<uint64_t>(args->rows) * 4) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    augur_cuda_result_v1 device = augur_set_device(ordinal); if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    constexpr uint32_t threads = 128; const uint32_t blocks = (args->rows + threads - 1) / threads;
    augur_f8e4m3_f32_matvec_kernel_v1<<<blocks, threads, 0, stream->stream>>>(reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)), reinterpret_cast<const float*>(static_cast<uintptr_t>(args->scales)), reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)), reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)), args->rows, args->cols);
    const cudaError_t error = cudaGetLastError(); if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_f8e4m3_f32_matvec_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_f8e4m3_f32_matvec_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_row32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_f8e4m3_f32_matvec_args_v1* args
) {
    const uint64_t weight_bytes = args == nullptr ? 0 : static_cast<uint64_t>(args->rows) * args->cols;
    const uint64_t scale_count = args == nullptr ? 0 : static_cast<uint64_t>(args->rows / 128) * (args->cols / 128);
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_F8E4M3_F32_MATVEC_ARGS_VERSION_V1 || args->struct_size != sizeof(augur_cuda_f8e4m3_f32_matvec_args_v1)
        || args->weight == 0 || args->scales == 0 || args->input == 0 || args->output == 0 || args->rows == 0 || args->cols == 0
        || args->rows % 128 != 0 || args->cols % 128 != 0 || args->weight_len_bytes != weight_bytes
        || args->scales_len_bytes < scale_count * 4 || args->input_len_bytes < static_cast<uint64_t>(args->cols) * 4
        || args->output_len_bytes < static_cast<uint64_t>(args->rows) * 4
        || args->scales % alignof(float) != 0 || args->input % alignof(float) != 0
        || args->output % alignof(float) != 0) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    augur_cuda_result_v1 device = augur_set_device(ordinal); if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    constexpr uint32_t threads = 128; const uint32_t blocks = (args->rows + threads - 1) / threads;
    augur_f8e4m3_f32_matvec_row32_kernel_v1<<<blocks, threads, 0, stream->stream>>>(reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)), reinterpret_cast<const float*>(static_cast<uintptr_t>(args->scales)), reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)), reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)), args->rows, args->cols);
    const cudaError_t error = cudaGetLastError(); if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_bitdecode_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_f8e4m3_f32_matvec_args_v1* args
) {
    const uint64_t weight_bytes = args == nullptr ? 0 : static_cast<uint64_t>(args->rows) * args->cols;
    const uint64_t scale_count = args == nullptr ? 0 : static_cast<uint64_t>(args->rows / 128) * (args->cols / 128);
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_F8E4M3_F32_MATVEC_ARGS_VERSION_V1 || args->struct_size != sizeof(augur_cuda_f8e4m3_f32_matvec_args_v1)
        || args->weight == 0 || args->scales == 0 || args->input == 0 || args->output == 0 || args->rows == 0 || args->cols == 0
        || args->rows % 128 != 0 || args->cols % 128 != 0 || args->weight_len_bytes != weight_bytes
        || args->scales_len_bytes < scale_count * 4 || args->input_len_bytes < static_cast<uint64_t>(args->cols) * 4
        || args->output_len_bytes < static_cast<uint64_t>(args->rows) * 4) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    augur_cuda_result_v1 device = augur_set_device(ordinal); if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    constexpr uint32_t threads = 128; const uint32_t blocks = (args->rows + threads - 1) / threads;
    augur_f8e4m3_f32_matvec_bitdecode_kernel_v1<<<blocks, threads, 0, stream->stream>>>(reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)), reinterpret_cast<const float*>(static_cast<uintptr_t>(args->scales)), reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)), reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)), args->rows, args->cols);
    const cudaError_t error = cudaGetLastError(); if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_bitdecode_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_f8e4m3_f32_matvec_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_f8e4m3_f32_matvec_bitdecode_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_warp_fp32_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_f8e4m3_f32_matvec_args_v1* args
) {
    const uint64_t weight_bytes = args == nullptr ? 0 : static_cast<uint64_t>(args->rows) * args->cols;
    const uint64_t scale_count = args == nullptr ? 0 : static_cast<uint64_t>(args->rows / 128) * (args->cols / 128);
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || args->abi_version != AUGUR_CUDA_F8E4M3_F32_MATVEC_ARGS_VERSION_V1 || args->struct_size != sizeof(augur_cuda_f8e4m3_f32_matvec_args_v1)
        || args->weight == 0 || args->scales == 0 || args->input == 0 || args->output == 0 || args->rows == 0 || args->cols == 0
        || args->rows % 128 != 0 || args->cols % 128 != 0 || args->weight_len_bytes != weight_bytes
        || args->scales_len_bytes < scale_count * 4 || args->input_len_bytes < static_cast<uint64_t>(args->cols) * 4
        || args->output_len_bytes < static_cast<uint64_t>(args->rows) * 4
        || args->weight % alignof(uint32_t) != 0 || args->scales % alignof(float) != 0
        || args->input % alignof(float) != 0 || args->output % alignof(float) != 0) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    augur_cuda_result_v1 device = augur_set_device(ordinal); if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    constexpr uint32_t threads = 128; const uint32_t blocks = (args->rows + 3) / 4;
    augur_f8e4m3_f32_matvec_warp_fp32_kernel_v1<<<blocks, threads, 0, stream->stream>>>(reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)), reinterpret_cast<const float*>(static_cast<uintptr_t>(args->scales)), reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)), reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)), args->rows, args->cols);
    const cudaError_t error = cudaGetLastError(); if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_ok();
}

#if defined(AUGUR_CUDA_P0_PREFILL_V22_PROJECTION_RESUME_BATCH_EXACT)
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_p0_prefill_v22_f8e4m3_batch4_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_f8e4m3_f32_matvec_args_v1* args,
    uint32_t tokens
) {
    const bool shape_ok = args != nullptr && args->rows == 8192 && args->cols == 1024;
    const uint64_t weight_bytes = shape_ok
        ? static_cast<uint64_t>(args->rows) * args->cols
        : 0;
    const uint64_t scale_bytes = shape_ok
        ? static_cast<uint64_t>(args->rows / 128) * (args->cols / 128) * 4
        : 0;
    const uint64_t input_bytes = shape_ok
        ? static_cast<uint64_t>(tokens) * args->cols * 4
        : 0;
    const uint64_t output_bytes = shape_ok
        ? static_cast<uint64_t>(tokens) * args->rows * 4
        : 0;
    if (stream == nullptr || stream->stream == nullptr || args == nullptr || ordinal < 0
        || stream->ordinal != ordinal || tokens != 128 || !shape_ok
        || args->abi_version != AUGUR_CUDA_F8E4M3_F32_MATVEC_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_f8e4m3_f32_matvec_args_v1)
        || args->weight == 0 || args->scales == 0 || args->input == 0 || args->output == 0
        || args->weight_len_bytes != weight_bytes
        || args->scales_len_bytes != scale_bytes
        || args->input_len_bytes != input_bytes
        || args->output_len_bytes != output_bytes
        || args->weight % alignof(uint32_t) != 0
        || args->scales % alignof(float) != 0
        || args->input % alignof(float) != 0
        || args->output % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 device = augur_set_device(ordinal);
    if (device.status != AUGUR_CUDA_STATUS_OK) return device;
    constexpr uint32_t threads = 128;
    const dim3 grid(
        (args->rows + 3) / 4,
        tokens / AUGUR_P0_PREFILL_V22_TOKENS_PER_WARP);
    augur_p0_prefill_v22_f8e4m3_batch4_kernel_v1<<<grid, threads, 0, stream->stream>>>(
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->scales)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output)),
        args->rows,
        args->cols);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}
#endif

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_warp_fp32_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_f8e4m3_f32_matvec_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_f8e4m3_f32_matvec_warp_fp32_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_f8e4m3_f32_matvec_row32_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_f8e4m3_f32_matvec_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_f8e4m3_f32_matvec_row32_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    return augur_cuda_stream_synchronize(stream);
}

static bool augur_checked_mul_u64(uint64_t left, uint64_t right, uint64_t* output) {
    if (output == nullptr || (right != 0 && left > UINT64_MAX / right)) {
        return false;
    }
    *output = left * right;
    return true;
}

static augur_cuda_result_v1 augur_validate_resident_moe_args(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_resident_moe_f32_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != AUGUR_CUDA_RESIDENT_MOE_F32_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_resident_moe_f32_args_v1)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (args->weights == 0 || args->input == 0 || args->output == 0
        || args->router_logits == 0 || args->selected_ids == 0
        || args->selected_weights == 0 || args->scratch == 0
        || args->access_counts == 0 || args->tokens == 0 || args->hidden == 0
        || args->intermediate == 0 || (args->expert_count != 4 && args->expert_count != 8)
        || (args->top_k != 2 && args->top_k != 4) || args->top_k > args->expert_count
        || !std::isfinite(args->routed_scale) || args->routed_scale <= 0.0f) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    uint64_t matrix = 0;
    uint64_t activation = 0;
    uint64_t router = 0;
    uint64_t routes = 0;
    uint64_t scratch = 0;
    uint64_t expert_payload = 0;
    if (!augur_checked_mul_u64(args->hidden, args->intermediate, &matrix)
        || !augur_checked_mul_u64(args->tokens, args->hidden, &activation)
        || !augur_checked_mul_u64(args->tokens, args->expert_count, &router)
        || !augur_checked_mul_u64(args->tokens, args->top_k, &routes)
        || !augur_checked_mul_u64(args->tokens, 3ULL * args->intermediate, &scratch)
        || !augur_checked_mul_u64(args->expert_count + 1ULL, 3ULL * matrix, &expert_payload)) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t weight_floats = static_cast<uint64_t>(args->expert_count) * args->hidden
        + expert_payload;
    const bool lengths = args->weights_len_bytes == weight_floats * sizeof(float)
        && args->input_len_bytes == activation * sizeof(float)
        && args->output_len_bytes == activation * sizeof(float)
        && args->router_logits_len_bytes == router * sizeof(float)
        && args->selected_ids_len_bytes == routes * sizeof(uint32_t)
        && args->selected_weights_len_bytes == routes * sizeof(float)
        && args->scratch_len_bytes == scratch * sizeof(float)
        && args->access_counts_len_bytes == (args->expert_count + 1ULL) * sizeof(uint32_t);
    if (!lengths) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_set_device(ordinal);
}

__device__ static float augur_resident_dot(const float* left, const float* right, uint32_t length) {
    float sum = 0.0f;
    for (uint32_t index = 0; index < length; ++index) {
        sum += left[index] * right[index];
    }
    return sum;
}

__device__ static void augur_resident_expert(
    const float* expert,
    const float* activation,
    float factor,
    uint32_t hidden,
    uint32_t intermediate,
    float* scratch,
    float* output
) {
    const uint64_t matrix = static_cast<uint64_t>(hidden) * intermediate;
    const float* gate_weight = expert;
    const float* up_weight = expert + matrix;
    const float* down_weight = expert + 2 * matrix;
    float* gate = scratch;
    float* up = scratch + intermediate;
    float* values = scratch + 2 * intermediate;
    for (uint32_t row = 0; row < intermediate; ++row) {
        gate[row] = augur_resident_dot(gate_weight + static_cast<uint64_t>(row) * hidden, activation, hidden);
        up[row] = augur_resident_dot(up_weight + static_cast<uint64_t>(row) * hidden, activation, hidden);
        values[row] = (gate[row] / (1.0f + expf(-gate[row]))) * up[row];
    }
    for (uint32_t row = 0; row < hidden; ++row) {
        output[row] += factor * augur_resident_dot(
            down_weight + static_cast<uint64_t>(row) * intermediate, values, intermediate);
    }
}

__global__ void augur_resident_moe_f32_kernel_v1(augur_cuda_resident_moe_f32_args_v1 args) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const float* weights = reinterpret_cast<const float*>(static_cast<uintptr_t>(args.weights));
    const float* input = reinterpret_cast<const float*>(static_cast<uintptr_t>(args.input));
    float* output = reinterpret_cast<float*>(static_cast<uintptr_t>(args.output));
    float* router_logits = reinterpret_cast<float*>(static_cast<uintptr_t>(args.router_logits));
    uint32_t* selected_ids = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(args.selected_ids));
    float* selected_weights = reinterpret_cast<float*>(static_cast<uintptr_t>(args.selected_weights));
    float* scratch = reinterpret_cast<float*>(static_cast<uintptr_t>(args.scratch));
    uint32_t* access = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(args.access_counts));
    for (uint32_t expert = 0; expert <= args.expert_count; ++expert) {
        access[expert] = 0;
    }
    const uint64_t router_count = static_cast<uint64_t>(args.expert_count) * args.hidden;
    const uint64_t matrix = static_cast<uint64_t>(args.hidden) * args.intermediate;
    const uint64_t expert_stride = 3 * matrix;
    const float* expert_base = weights + router_count;
    const float* shared = expert_base + static_cast<uint64_t>(args.expert_count) * expert_stride;
    for (uint32_t token = 0; token < args.tokens; ++token) {
        const float* activation = input + static_cast<uint64_t>(token) * args.hidden;
        float* destination = output + static_cast<uint64_t>(token) * args.hidden;
        float* logits = router_logits + static_cast<uint64_t>(token) * args.expert_count;
        uint32_t* ids = selected_ids + static_cast<uint64_t>(token) * args.top_k;
        float* route_weights = selected_weights + static_cast<uint64_t>(token) * args.top_k;
        float* token_scratch = scratch + static_cast<uint64_t>(token) * 3 * args.intermediate;
        for (uint32_t row = 0; row < args.hidden; ++row) {
            destination[row] = activation[row];
        }
        for (uint32_t expert = 0; expert < args.expert_count; ++expert) {
            logits[expert] = augur_resident_dot(weights + static_cast<uint64_t>(expert) * args.hidden,
                                             activation, args.hidden);
        }
        for (uint32_t route = 0; route < args.top_k; ++route) {
            ids[route] = UINT32_MAX;
            route_weights[route] = -INFINITY;
        }
        for (uint32_t expert = 0; expert < args.expert_count; ++expert) {
            for (uint32_t slot = 0; slot < args.top_k; ++slot) {
                if (logits[expert] > route_weights[slot]
                    || (logits[expert] == route_weights[slot] && expert < ids[slot])) {
                    for (uint32_t shift = args.top_k - 1; shift > slot; --shift) {
                        route_weights[shift] = route_weights[shift - 1];
                        ids[shift] = ids[shift - 1];
                    }
                    route_weights[slot] = logits[expert];
                    ids[slot] = expert;
                    break;
                }
            }
        }
        const float max_logit = route_weights[0];
        float denominator = 0.0f;
        for (uint32_t route = 0; route < args.top_k; ++route) {
            route_weights[route] = expf(route_weights[route] - max_logit);
            denominator += route_weights[route];
        }
        for (uint32_t route = 0; route < args.top_k; ++route) {
            route_weights[route] /= denominator;
            ++access[ids[route]];
            augur_resident_expert(expert_base + static_cast<uint64_t>(ids[route]) * expert_stride,
                               activation, route_weights[route] * args.routed_scale,
                               args.hidden, args.intermediate, token_scratch, destination);
        }
        ++access[args.expert_count];
        augur_resident_expert(shared, activation, 1.0f, args.hidden, args.intermediate,
                           token_scratch, destination);
    }
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_resident_moe_f32_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_resident_moe_f32_args_v1* args
) {
    augur_cuda_result_v1 validation = augur_validate_resident_moe_args(ordinal, stream, args);
    if (validation.status != AUGUR_CUDA_STATUS_OK) {
        return validation;
    }
    augur_resident_moe_f32_kernel_v1<<<1, 1, 0, stream->stream>>>(*args);
    const cudaError_t launch_error = cudaGetLastError();
    return launch_error == cudaSuccess ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_resident_moe_f32_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_resident_moe_f32_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_resident_moe_f32_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) {
        return submitted;
    }
    return augur_cuda_stream_synchronize(stream);
}

#if defined(AUGUR_CUDA_BLACKWELL_NVFP4)

struct augur_nvfp4_shape_bytes_v1 {
    uint64_t a_values;
    uint64_t a_scales;
    uint64_t b_values;
    uint64_t b_scales;
    uint64_t output;
};

static constexpr augur_nvfp4_shape_bytes_v1 AUGUR_NVFP4_W13_M128_BYTES = {
    UINT64_C(262144), UINT64_C(32768), UINT64_C(4194304),
    UINT64_C(524288), UINT64_C(524288)
};

static constexpr augur_nvfp4_shape_bytes_v1 AUGUR_NVFP4_W2_M128_BYTES = {
    UINT64_C(131072), UINT64_C(16384), UINT64_C(4194304),
    UINT64_C(524288), UINT64_C(1048576)
};

typedef augur_cuda_result_v1 (*augur_nvfp4_workspace_query_v1)(uint64_t*);

static augur_cuda_result_v1 augur_validate_nvfp4_aot_args(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_aot_args_v1* args,
    augur_nvfp4_shape_bytes_v1 expected,
    augur_nvfp4_workspace_query_v1 workspace_query
) {
    if (stream == nullptr || args == nullptr || ordinal < 0
        || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != AUGUR_CUDA_NVFP4_AOT_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_nvfp4_aot_args_v1)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (augur_cuda_nvfp4_sm120_build_capability_v1() != 1) {
        return augur_error(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE, cudaSuccess);
    }
    if (args->a_values == 0 || args->a_scales == 0 || args->b_values == 0
        || args->b_scales == 0 || args->output == 0 || args->reserved != 0
        || !std::isfinite(args->alpha) || args->alpha <= 0.0f
        || args->a_values_len_bytes != expected.a_values
        || args->a_scales_len_bytes != expected.a_scales
        || args->b_values_len_bytes != expected.b_values
        || args->b_scales_len_bytes != expected.b_scales
        || args->output_len_bytes != expected.output
        || (args->a_values % 16) != 0 || (args->a_scales % 16) != 0
        || (args->b_values % 16) != 0 || (args->b_scales % 16) != 0
        || (args->output % 2) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    uint64_t required_workspace = 0;
    augur_cuda_result_v1 queried = workspace_query(&required_workspace);
    if (queried.status != AUGUR_CUDA_STATUS_OK) {
        return queried;
    }
    if (args->workspace_len_bytes < required_workspace
        || (required_workspace != 0 && args->workspace == 0)) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    cudaDeviceProp properties{};
    const cudaError_t property_error = cudaGetDeviceProperties(&properties, ordinal);
    if (property_error != cudaSuccess) {
        return augur_runtime_error(property_error);
    }
    if (properties.major < 12) {
        return augur_error(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE, cudaSuccess);
    }
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_w13_m128_bf16_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_aot_args_v1* args
) {
    augur_cuda_result_v1 validation = augur_validate_nvfp4_aot_args(
        ordinal, stream, args, AUGUR_NVFP4_W13_M128_BYTES,
        augur_cuda_nvfp4_w13_m128_workspace_size_v1
    );
    if (validation.status != AUGUR_CUDA_STATUS_OK) {
        return validation;
    }
    return augur_nvfp4_cutlass_w13_m128_launch_v1(args, stream->stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_w13_m128_bf16_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_aot_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_nvfp4_w13_m128_bf16_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) {
        return submitted;
    }
    return augur_cuda_stream_synchronize(stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_w2_m128_bf16_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_aot_args_v1* args
) {
    augur_cuda_result_v1 validation = augur_validate_nvfp4_aot_args(
        ordinal, stream, args, AUGUR_NVFP4_W2_M128_BYTES,
        augur_cuda_nvfp4_w2_m128_workspace_size_v1
    );
    if (validation.status != AUGUR_CUDA_STATUS_OK) {
        return validation;
    }
    return augur_nvfp4_cutlass_w2_m128_launch_v1(args, stream->stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_w2_m128_bf16_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_aot_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_nvfp4_w2_m128_bf16_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) {
        return submitted;
    }
    return augur_cuda_stream_synchronize(stream);
}

struct augur_nvfp4_shape_descriptor_v1 {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

static bool augur_nvfp4_describe_shape(
    uint32_t shape,
    augur_nvfp4_shape_descriptor_v1* out
) {
    if (out == nullptr) {
        return false;
    }
#define AUGUR_DESC_CASE(ID, M, N, K) case ID: *out = {M, N, K}; return true
    switch (shape) {
        AUGUR_DESC_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M1_V1, 1, 2048, 4096);
        AUGUR_DESC_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M2_V1, 2, 2048, 4096);
        AUGUR_DESC_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M4_V1, 4, 2048, 4096);
        AUGUR_DESC_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M8_V1, 8, 2048, 4096);
        AUGUR_DESC_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M16_V1, 16, 2048, 4096);
        AUGUR_DESC_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M32_V1, 32, 2048, 4096);
        AUGUR_DESC_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M64_V1, 64, 2048, 4096);
        AUGUR_DESC_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M128_V1, 128, 2048, 4096);
        AUGUR_DESC_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M1_V1, 1, 4096, 2048);
        AUGUR_DESC_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M2_V1, 2, 4096, 2048);
        AUGUR_DESC_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M4_V1, 4, 4096, 2048);
        AUGUR_DESC_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M8_V1, 8, 4096, 2048);
        AUGUR_DESC_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M16_V1, 16, 4096, 2048);
        AUGUR_DESC_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M32_V1, 32, 4096, 2048);
        AUGUR_DESC_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M64_V1, 64, 4096, 2048);
        AUGUR_DESC_CASE(AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M128_V1, 128, 4096, 2048);
        default: return false;
    }
#undef AUGUR_DESC_CASE
}

static uint32_t augur_nvfp4_w2_shape_for_capacity(uint32_t capacity) {
    switch (capacity) {
        case 1: return AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M1_V1;
        case 2: return AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M2_V1;
        case 4: return AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M4_V1;
        case 8: return AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M8_V1;
        case 16: return AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M16_V1;
        case 32: return AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M32_V1;
        case 64: return AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M64_V1;
        case 128: return AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M128_V1;
        default: return 0;
    }
}

static augur_cuda_result_v1 augur_validate_nvfp4_shape_args(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    uint32_t shape,
    const augur_cuda_nvfp4_aot_args_v1* args
) {
    augur_nvfp4_shape_descriptor_v1 descriptor{};
    if (!augur_nvfp4_describe_shape(shape, &descriptor) || stream == nullptr
        || args == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    uint64_t a_scale_bytes = 0;
    uint64_t b_scale_bytes = 0;
    uint64_t workspace_bytes = 0;
    augur_cuda_result_v1 a_query = augur_cuda_nvfp4_scale_extent_host_v1(
        shape, AUGUR_CUDA_NVFP4_OPERAND_A_V1, &a_scale_bytes);
    augur_cuda_result_v1 b_query = augur_cuda_nvfp4_scale_extent_host_v1(
        shape, AUGUR_CUDA_NVFP4_OPERAND_B_V1, &b_scale_bytes);
    augur_cuda_result_v1 workspace_query = augur_cuda_nvfp4_shape_workspace_size_v1(
        shape, &workspace_bytes);
    if (a_query.status != AUGUR_CUDA_STATUS_OK || b_query.status != AUGUR_CUDA_STATUS_OK
        || workspace_query.status != AUGUR_CUDA_STATUS_OK) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t a_values = static_cast<uint64_t>(descriptor.m) * descriptor.k / 2;
    const uint64_t b_values = static_cast<uint64_t>(descriptor.n) * descriptor.k / 2;
    const uint64_t output = static_cast<uint64_t>(descriptor.m) * descriptor.n * 2;
    if (args->abi_version != AUGUR_CUDA_NVFP4_AOT_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_nvfp4_aot_args_v1)
        || args->a_values == 0 || args->a_scales == 0 || args->b_values == 0
        || args->b_scales == 0 || args->output == 0 || args->reserved != 0
        || !std::isfinite(args->alpha) || args->alpha <= 0.0f
        || args->a_values_len_bytes != a_values
        || args->a_scales_len_bytes != a_scale_bytes
        || args->b_values_len_bytes != b_values
        || args->b_scales_len_bytes != b_scale_bytes
        || args->output_len_bytes != output
        || args->workspace_len_bytes < workspace_bytes
        || (workspace_bytes != 0 && args->workspace == 0)
        || (args->a_values % 16) != 0 || (args->a_scales % 16) != 0
        || (args->b_values % 16) != 0 || (args->b_scales % 16) != 0
        || (args->output % 2) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    cudaDeviceProp properties{};
    const cudaError_t property_error = cudaGetDeviceProperties(&properties, ordinal);
    if (property_error != cudaSuccess) {
        return augur_runtime_error(property_error);
    }
    if (properties.major != 12 || properties.minor != 0) {
        return augur_error(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE, cudaSuccess);
    }
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_shape_bf16_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, uint32_t shape,
    const augur_cuda_nvfp4_aot_args_v1* args) {
    augur_cuda_result_v1 validation = augur_validate_nvfp4_shape_args(
        ordinal, stream, shape, args);
    if (validation.status != AUGUR_CUDA_STATUS_OK) {
        return validation;
    }
    return augur_nvfp4_cutlass_shape_launch_v1(shape, args, stream->stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_shape_bf16_v1(
    int32_t ordinal, augur_cuda_stream_t stream, uint32_t shape,
    const augur_cuda_nvfp4_aot_args_v1* args) {
    augur_cuda_result_v1 result = augur_cuda_nvfp4_shape_bf16_submit_v1(
        ordinal, stream, shape, args);
    return result.status == AUGUR_CUDA_STATUS_OK ? augur_cuda_stream_synchronize(stream) : result;
}

static augur_cuda_result_v1 augur_validate_nvfp4_w4a16_args(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    uint32_t shape,
    const augur_cuda_nvfp4_w4a16_args_v1* args
) {
    augur_nvfp4_shape_descriptor_v1 descriptor{};
    if (!augur_nvfp4_describe_shape(shape, &descriptor) || stream == nullptr
        || args == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t activation = static_cast<uint64_t>(descriptor.m) * descriptor.k * 2;
    const uint64_t weight_values = static_cast<uint64_t>(descriptor.n) * descriptor.k / 2;
    const uint64_t weight_scales = static_cast<uint64_t>(descriptor.n) * descriptor.k / 16;
    const uint64_t output = static_cast<uint64_t>(descriptor.m) * descriptor.n * 2;
    if (args->abi_version != AUGUR_CUDA_NVFP4_W4A16_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_nvfp4_w4a16_args_v1)
        || args->activation_bf16 == 0 || args->weight_values == 0
        || args->weight_scales == 0 || args->output_bf16 == 0
        || args->activation_len_bytes != activation
        || args->weight_values_len_bytes != weight_values
        || args->weight_scales_len_bytes != weight_scales
        || args->output_len_bytes != output
        || !std::isfinite(args->weight_global_scale)
        || args->weight_global_scale <= 0.0f || args->reserved != 0
        || (args->activation_bf16 % 2) != 0 || (args->output_bf16 % 2) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    cudaDeviceProp properties{};
    const cudaError_t property_error = cudaGetDeviceProperties(&properties, ordinal);
    if (property_error != cudaSuccess) {
        return augur_runtime_error(property_error);
    }
    return properties.major == 12 && properties.minor == 0
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE, cudaSuccess);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_w4a16_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, uint32_t shape,
    const augur_cuda_nvfp4_w4a16_args_v1* args
) {
    augur_cuda_result_v1 validation = augur_validate_nvfp4_w4a16_args(
        ordinal, stream, shape, args);
    if (validation.status != AUGUR_CUDA_STATUS_OK) {
        return validation;
    }
    return augur_nvfp4_w4a16_shape_launch_v1(shape, args, stream->stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_w4a16_v1(
    int32_t ordinal, augur_cuda_stream_t stream, uint32_t shape,
    const augur_cuda_nvfp4_w4a16_args_v1* args
) {
    augur_cuda_result_v1 result = augur_cuda_nvfp4_w4a16_submit_v1(
        ordinal, stream, shape, args);
    return result.status == AUGUR_CUDA_STATUS_OK
        ? augur_cuda_stream_synchronize(stream)
        : result;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m68_bf16_interstage_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_m68_bf16_interstage_args_v1* args
) {
    const bool capacity_valid = args != nullptr && (
        args->capacity == 1 || args->capacity == 2 || args->capacity == 4
        || args->capacity == 8 || args->capacity == 16 || args->capacity == 32
        || args->capacity == 64 || args->capacity == 128);
    const uint64_t tensor_bytes = capacity_valid
        ? static_cast<uint64_t>(args->capacity) * 2048 * 2
        : 0;
    if (stream == nullptr || ordinal < 0 || stream->ordinal != ordinal
        || !capacity_valid || args->valid_rows == 0
        || args->valid_rows > args->capacity
        || args->abi_version != AUGUR_CUDA_M68_BF16_INTERSTAGE_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_m68_bf16_interstage_args_v1)
        || args->gate_bf16 == 0 || args->up_bf16 == 0
        || args->route_weights_f32 == 0 || args->output_bf16 == 0
        || args->gate_len_bytes != tensor_bytes || args->up_len_bytes != tensor_bytes
        || args->output_len_bytes != tensor_bytes
        || args->route_weights_len_bytes != static_cast<uint64_t>(args->capacity) * 4
        || !std::isfinite(args->swiglu_limit) || args->swiglu_limit <= 0.0f
        || args->reserved != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    cudaDeviceProp properties{};
    const cudaError_t property_error = cudaGetDeviceProperties(&properties, ordinal);
    if (property_error != cudaSuccess) {
        return augur_runtime_error(property_error);
    }
    if (properties.major != 12 || properties.minor != 0) {
        return augur_error(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE, cudaSuccess);
    }
    return augur_m68_bf16_interstage_launch_v1(args, stream->stream);
}

static augur_cuda_result_v1 augur_validate_nvfp4_shape_interstage(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_shape_interstage_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0
        || stream->ordinal != ordinal || args->abi_version != AUGUR_CUDA_NVFP4_SHAPE_INTERSTAGE_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_nvfp4_shape_interstage_args_v1)
        || augur_nvfp4_w2_shape_for_capacity(args->capacity) == 0
        || args->valid_rows == 0 || args->valid_rows > args->capacity
        || !std::isfinite(args->activation_global_scale) || args->activation_global_scale <= 0.0f
        || !std::isfinite(args->swiglu_limit) || args->swiglu_limit <= 0.0f) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    uint64_t scale_bytes = 0;
    augur_cuda_result_v1 queried = augur_cuda_nvfp4_scale_extent_host_v1(
        augur_nvfp4_w2_shape_for_capacity(args->capacity),
        AUGUR_CUDA_NVFP4_OPERAND_A_V1, &scale_bytes);
    const uint64_t elements = static_cast<uint64_t>(args->capacity) * 2048;
    if (queried.status != AUGUR_CUDA_STATUS_OK || args->gate_bf16 == 0 || args->up_bf16 == 0
        || args->route_weights_f32 == 0 || args->output_values == 0
        || args->output_scales == 0 || args->scale_offsets == 0
        || args->gate_len_bytes != elements * 2 || args->up_len_bytes != elements * 2
        || args->route_weights_len_bytes != static_cast<uint64_t>(args->capacity) * 4
        || args->output_values_len_bytes != elements / 2
        || args->output_scales_len_bytes != scale_bytes
        || args->scale_offsets_len_bytes != static_cast<uint64_t>(args->capacity) * (2048 / 16) * 4) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_set_device(ordinal);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_shape_interstage_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_shape_interstage_args_v1* args) {
    augur_cuda_result_v1 validation = augur_validate_nvfp4_shape_interstage(ordinal, stream, args);
    if (validation.status != AUGUR_CUDA_STATUS_OK) {
        return validation;
    }
    return augur_nvfp4_shape_interstage_launch_v1(args, stream->stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_nvfp4_shape_interstage_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_shape_interstage_args_v1* args) {
    augur_cuda_result_v1 result = augur_cuda_nvfp4_shape_interstage_submit_v1(ordinal, stream, args);
    return result.status == AUGUR_CUDA_STATUS_OK ? augur_cuda_stream_synchronize(stream) : result;
}

static augur_cuda_result_v1 augur_validate_nvfp4_interstage_args(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_interstage_args_v1* args
) {
    if (stream == nullptr || args == nullptr || ordinal < 0
        || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (args->abi_version != AUGUR_CUDA_NVFP4_INTERSTAGE_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_nvfp4_interstage_args_v1)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (augur_cuda_nvfp4_sm120_build_capability_v1() != 1) {
        return augur_error(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE, cudaSuccess);
    }
    if (args->gate_bf16 == 0 || args->up_bf16 == 0
        || args->output_values == 0 || args->output_scales == 0
        || args->scale_offsets == 0 || args->reserved != 0
        || args->gate_len_bytes != UINT64_C(524288)
        || args->up_len_bytes != UINT64_C(524288)
        || args->output_values_len_bytes != UINT64_C(131072)
        || args->output_scales_len_bytes != UINT64_C(16384)
        || args->scale_offsets_len_bytes != UINT64_C(65536)
        || (args->gate_bf16 % 2) != 0 || (args->up_bf16 % 2) != 0
        || (args->output_values % 16) != 0 || (args->output_scales % 16) != 0
        || (args->scale_offsets % 4) != 0
        || !std::isfinite(args->route_weight) || args->route_weight <= 0.0f
        || !std::isfinite(args->activation_global_scale)
        || args->activation_global_scale <= 0.0f
        || !std::isfinite(args->swiglu_limit) || args->swiglu_limit <= 0.0f) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    cudaDeviceProp properties{};
    const cudaError_t property_error = cudaGetDeviceProperties(&properties, ordinal);
    if (property_error != cudaSuccess) {
        return augur_runtime_error(property_error);
    }
    if (properties.major != 12 || properties.minor != 0) {
        return augur_error(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE, cudaSuccess);
    }
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_interstage_m128_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_interstage_args_v1* args
) {
    augur_cuda_result_v1 validation =
        augur_validate_nvfp4_interstage_args(ordinal, stream, args);
    if (validation.status != AUGUR_CUDA_STATUS_OK) {
        return validation;
    }
    return augur_nvfp4_interstage_m128_launch_v1(args, stream->stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_nvfp4_interstage_m128_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_nvfp4_interstage_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_nvfp4_interstage_m128_submit_v1(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) {
        return submitted;
    }
    return augur_cuda_stream_synchronize(stream);
}

static augur_cuda_result_v1 augur_validate_m54_device(
    int32_t ordinal,
    augur_cuda_stream_t stream
) {
    if (stream == nullptr || ordinal < 0 || stream->ordinal != ordinal) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (augur_cuda_nvfp4_sm120_build_capability_v1() != 1) {
        return augur_error(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    cudaDeviceProp properties{};
    const cudaError_t error = cudaGetDeviceProperties(&properties, ordinal);
    if (error != cudaSuccess) {
        return augur_runtime_error(error);
    }
    return properties.major == 12 && properties.minor == 0
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE, cudaSuccess);
}

static augur_cuda_result_v1 augur_validate_bf16_gemm(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_bf16_gemm_args_v1* args,
    uint64_t activation_bytes,
    uint64_t weight_bytes,
    uint64_t output_bytes
) {
    if (args == nullptr
        || args->abi_version != AUGUR_CUDA_BF16_GEMM_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_bf16_gemm_args_v1)
        || args->activation_bf16 == 0 || args->weight_bf16 == 0 || args->output_f32 == 0
        || args->activation_len_bytes != activation_bytes
        || args->weight_len_bytes != weight_bytes
        || args->output_len_bytes != output_bytes
        || (args->activation_bf16 % 16) != 0 || (args->weight_bf16 % 16) != 0
        || (args->output_f32 % 4) != 0 || !std::isfinite(args->alpha)
        || args->alpha <= 0.0f || args->reserved != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    return augur_validate_m54_device(ordinal, stream);
}

#define AUGUR_M54_BF16_WRAPPERS(NAME, A_BYTES, B_BYTES, O_BYTES, LAUNCH) \
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_##NAME##_submit_v1( \
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_bf16_gemm_args_v1* args) { \
    augur_cuda_result_v1 validation = augur_validate_bf16_gemm(ordinal, stream, args, A_BYTES, B_BYTES, O_BYTES); \
    return validation.status == AUGUR_CUDA_STATUS_OK ? LAUNCH(args, stream->stream) : validation; \
} \
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_bf16_##NAME##_v1( \
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_bf16_gemm_args_v1* args) { \
    augur_cuda_result_v1 submitted = augur_cuda_bf16_##NAME##_submit_v1(ordinal, stream, args); \
    return submitted.status == AUGUR_CUDA_STATUS_OK ? augur_cuda_stream_synchronize(stream) : submitted; \
}

AUGUR_M54_BF16_WRAPPERS(w13_m128, UINT64_C(1048576), UINT64_C(16777216), UINT64_C(1048576), augur_bf16_w13_m128_launch_v1)
AUGUR_M54_BF16_WRAPPERS(w2_m128, UINT64_C(524288), UINT64_C(16777216), UINT64_C(2097152), augur_bf16_w2_m128_launch_v1)

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_m54_shared_interstage_m128_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m54_shared_interstage_args_v1* args
) {
    if (args == nullptr
        || args->abi_version != AUGUR_CUDA_M54_SHARED_INTERSTAGE_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_m54_shared_interstage_args_v1)
        || args->gate_f32 == 0 || args->up_f32 == 0 || args->output_bf16 == 0
        || args->gate_len_bytes != UINT64_C(1048576)
        || args->up_len_bytes != UINT64_C(1048576)
        || args->output_len_bytes != UINT64_C(524288)
        || !std::isfinite(args->swiglu_limit) || args->swiglu_limit != 10.0f
        || args->reserved != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 validation = augur_validate_m54_device(ordinal, stream);
    return validation.status == AUGUR_CUDA_STATUS_OK
        ? augur_m54_shared_interstage_m128_launch_v1(args, stream->stream)
        : validation;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_m54_shared_interstage_m128_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m54_shared_interstage_args_v1* args
) {
    augur_cuda_result_v1 submitted =
        augur_cuda_m54_shared_interstage_m128_submit_v1(ordinal, stream, args);
    return submitted.status == AUGUR_CUDA_STATUS_OK
        ? augur_cuda_stream_synchronize(stream) : submitted;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_m54_sum_m128_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m54_sum_args_v1* args
) {
    bool bad_routed = args == nullptr;
    if (args != nullptr) {
        for (int index = 0; index < 6; ++index) {
            bad_routed = bad_routed || args->routed_bf16[index] == 0
                || (args->routed_bf16[index] % 2) != 0;
        }
    }
    if (args == nullptr || bad_routed
        || args->abi_version != AUGUR_CUDA_M54_SUM_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_m54_sum_args_v1)
        || args->routed_len_bytes != UINT64_C(1048576)
        || args->shared_f32 == 0 || args->shared_len_bytes != UINT64_C(2097152)
        || args->output_bf16 == 0 || args->output_len_bytes != UINT64_C(1048576)
        || args->router_status == 0 || args->router_status_len_bytes != UINT64_C(4)
        || args->reserved != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 validation = augur_validate_m54_device(ordinal, stream);
    return validation.status == AUGUR_CUDA_STATUS_OK
        ? augur_m54_sum_m128_launch_v1(args, stream->stream) : validation;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_m54_sum_m128_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m54_sum_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_m54_sum_m128_submit_v1(ordinal, stream, args);
    return submitted.status == AUGUR_CUDA_STATUS_OK
        ? augur_cuda_stream_synchronize(stream) : submitted;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_m54_router_top6_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m54_router_args_v1* args
) {
    if (args == nullptr
        || args->abi_version != AUGUR_CUDA_M54_ROUTER_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_m54_router_args_v1)
        || args->logits_f32 == 0 || args->logits_len_bytes != UINT64_C(1024)
        || args->bias_f32 == 0 || args->bias_len_bytes != UINT64_C(1024)
        || args->output_ids_i32 == 0 || args->output_ids_len_bytes != UINT64_C(24)
        || args->output_weights_f32 == 0 || args->output_weights_len_bytes != UINT64_C(24)
        || args->status_i32 == 0 || args->status_len_bytes != UINT64_C(4)
        || args->routed_scale != 1.5f || args->weight_tolerance != 1.0e-5f
        || args->reserved != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 validation = augur_validate_m54_device(ordinal, stream);
    return validation.status == AUGUR_CUDA_STATUS_OK
        ? augur_m54_router_top6_launch_v1(args, stream->stream) : validation;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_m54_router_top6_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m54_router_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_m54_router_top6_submit_v1(ordinal, stream, args);
    return submitted.status == AUGUR_CUDA_STATUS_OK
        ? augur_cuda_stream_synchronize(stream) : submitted;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_m58_router_group_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m58_router_group_args_v1* args
) {
    if (args == nullptr
        || args->abi_version != AUGUR_CUDA_M58_ROUTER_GROUP_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_m58_router_group_args_v1)
        || args->tokens == 0 || args->tokens > 128
        || args->experts != 256 || args->top_k != 6
        || args->routed_scale != 1.5f || args->weight_tolerance != 2.0e-6f
        || args->scores_f32 == 0 || args->scores_len_bytes != static_cast<uint64_t>(args->tokens) * 1024
        || args->bias_f32 == 0 || args->bias_len_bytes != 1024
        || args->expected_ids_u16 == 0 || args->expected_ids_len_bytes != static_cast<uint64_t>(args->tokens) * 12
        || args->expected_weights_f32 == 0 || args->expected_weights_len_bytes != static_cast<uint64_t>(args->tokens) * 24
        || args->output_ids_u16 == 0 || args->output_ids_len_bytes != static_cast<uint64_t>(args->tokens) * 12
        || args->output_weights_f32 == 0 || args->output_weights_len_bytes != static_cast<uint64_t>(args->tokens) * 24
        || args->counts_u32 == 0 || args->counts_len_bytes != 1024
        || args->offsets_u32 == 0 || args->offsets_len_bytes != 1028
        || args->gathered_tokens_u16 == 0 || args->gathered_tokens_len_bytes != 1536
        || args->gathered_weights_f32 == 0 || args->gathered_weights_len_bytes != 3072
        || args->status_i32 == 0 || args->status_len_bytes != 4
        || args->routed_accum_f32 == 0
        || args->routed_accum_len_bytes != static_cast<uint64_t>(args->tokens) * 16384
        || args->reserved != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 validation = augur_validate_m54_device(ordinal, stream);
    return validation.status == AUGUR_CUDA_STATUS_OK
        ? augur_m58_router_group_launch_v1(args, stream->stream) : validation;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_m58_scatter_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m58_scatter_args_v1* args
) {
    const uint64_t minimum_output = args == nullptr
        ? 0 : static_cast<uint64_t>(args->valid_rows) * args->hidden * 2;
    if (args == nullptr
        || args->abi_version != AUGUR_CUDA_M58_SCATTER_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_m58_scatter_args_v1)
        || args->tokens == 0 || args->tokens > 128 || args->hidden != 4096
        || args->valid_rows == 0 || args->valid_rows > 128
        || args->group_offset > 768 || args->valid_rows > 768 - args->group_offset
        || args->expert_output_bf16 == 0 || args->expert_output_len_bytes < minimum_output
        || args->gathered_tokens_u16 == 0 || args->gathered_tokens_len_bytes != 1536
        || args->routed_accum_f32 == 0
        || args->routed_accum_len_bytes != static_cast<uint64_t>(args->tokens) * 16384
        || args->status_i32 == 0 || args->status_len_bytes != 4) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 validation = augur_validate_m54_device(ordinal, stream);
    return validation.status == AUGUR_CUDA_STATUS_OK
        ? augur_m58_scatter_launch_v1(args, stream->stream) : validation;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_m58_finalize_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m58_finalize_args_v1* args
) {
    if (args == nullptr
        || args->abi_version != AUGUR_CUDA_M58_FINALIZE_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_m58_finalize_args_v1)
        || args->tokens == 0 || args->tokens > 128 || args->hidden != 4096
        || args->routed_accum_f32 == 0
        || args->routed_accum_len_bytes != static_cast<uint64_t>(args->tokens) * 16384
        || args->shared_f32 == 0 || args->shared_len_bytes != 2097152
        || args->output_bf16 == 0 || args->output_len_bytes != 1048576
        || args->status_i32 == 0 || args->status_len_bytes != 4
        || args->reserved != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 validation = augur_validate_m54_device(ordinal, stream);
    if (validation.status != AUGUR_CUDA_STATUS_OK) {
        return validation;
    }
    return augur_m58_finalize_launch_v1(args, stream->stream);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_m58_finalize_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m58_finalize_args_v1* args
) {
    augur_cuda_result_v1 submitted = augur_cuda_m58_finalize_submit_v1(ordinal, stream, args);
    return submitted.status == AUGUR_CUDA_STATUS_OK
        ? augur_cuda_stream_synchronize(stream) : submitted;
}

#endif  // defined(AUGUR_CUDA_BLACKWELL_NVFP4)

__device__ __forceinline__ float augur_m111_bf16_round(float value) {
    uint32_t bits = __float_as_uint(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return __uint_as_float(bits & 0xffff0000U);
}

__global__ void augur_m111_sparse_attention_kernel_v1(
    const float * query,
    const float * kv,
    const float * sink,
    const int32_t * indices,
    const float * rope_cos,
    const float * rope_sin,
    float * output
) {
    constexpr int TOKENS = 8;
    constexpr int HEADS = 64;
    constexpr int HEAD_DIM = 512;
    constexpr int SELECTED = 10;
    const int token = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int pair = static_cast<int>(threadIdx.x);
    if (token >= TOKENS || head >= HEADS || pair >= HEAD_DIM / 2) return;

    __shared__ float reduction[256];
    __shared__ float scores[SELECTED];
    __shared__ float weights[SELECTED];
    const int q_base = (token * HEADS + head) * HEAD_DIM;
    const int kv_base = token * SELECTED;
    const int d0 = pair * 2;
    const int d1 = d0 + 1;

    for (int selected = 0; selected < SELECTED; ++selected) {
        const int32_t index = indices[kv_base + selected];
        reduction[pair] = index < 0
            ? 0.0f
            : query[q_base + d0] * kv[index * HEAD_DIM + d0]
                + query[q_base + d1] * kv[index * HEAD_DIM + d1];
        __syncthreads();
        for (int stride = 128; stride != 0; stride >>= 1) {
            if (pair < stride) reduction[pair] += reduction[pair + stride];
            __syncthreads();
        }
        if (pair == 0) {
            scores[selected] = index < 0
                ? -INFINITY
                : reduction[0] * rsqrtf(static_cast<float>(HEAD_DIM));
        }
        __syncthreads();
    }

    if (pair == 0) {
        float maximum = sink[head];
        for (int selected = 0; selected < SELECTED; ++selected) maximum = fmaxf(maximum, scores[selected]);
        float denominator = expf(sink[head] - maximum);
        for (int selected = 0; selected < SELECTED; ++selected) {
            weights[selected] = isfinite(scores[selected]) ? expf(scores[selected] - maximum) : 0.0f;
            denominator += weights[selected];
        }
        const float inverse = 1.0f / denominator;
        for (int selected = 0; selected < SELECTED; ++selected) weights[selected] *= inverse;
    }
    __syncthreads();

    float out0 = 0.0f;
    float out1 = 0.0f;
    for (int selected = 0; selected < SELECTED; ++selected) {
        const int32_t index = indices[kv_base + selected];
        if (index >= 0) {
            out0 += weights[selected] * kv[index * HEAD_DIM + d0];
            out1 += weights[selected] * kv[index * HEAD_DIM + d1];
        }
    }
    out0 = augur_m111_bf16_round(out0);
    out1 = augur_m111_bf16_round(out1);
    if (d0 >= HEAD_DIM - 64) {
        const int rope_pair = pair - (HEAD_DIM - 64) / 2;
        const float cosine = rope_cos[token * 32 + rope_pair];
        const float sine = rope_sin[token * 32 + rope_pair];
        const float rotated0 = out0 * cosine + out1 * sine;
        const float rotated1 = -out0 * sine + out1 * cosine;
        out0 = augur_m111_bf16_round(rotated0);
        out1 = augur_m111_bf16_round(rotated1);
    }
    const int output_base = (token * HEADS + head) * HEAD_DIM;
    output[output_base + d0] = out0;
    output[output_base + d1] = out1;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL
augur_cuda_m111_sparse_attention_f32_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m111_sparse_attention_f32_args_v1* args
) {
    constexpr uint64_t QUERY_BYTES = 8ULL * 64ULL * 512ULL * 4ULL;
    constexpr uint64_t KV_BYTES = 10ULL * 512ULL * 4ULL;
    constexpr uint64_t SINK_BYTES = 64ULL * 4ULL;
    constexpr uint64_t INDICES_BYTES = 8ULL * 10ULL * 4ULL;
    constexpr uint64_t ROPE_BYTES = 8ULL * 32ULL * 4ULL;
    if (stream == nullptr || args == nullptr
        || args->abi_version != AUGUR_CUDA_M111_SPARSE_ATTENTION_F32_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_m111_sparse_attention_f32_args_v1)
        || args->query_f32 == 0 || args->query_len_bytes != QUERY_BYTES
        || args->kv_f32 == 0 || args->kv_len_bytes != KV_BYTES
        || args->sink_f32 == 0 || args->sink_len_bytes != SINK_BYTES
        || args->indices_i32 == 0 || args->indices_len_bytes != INDICES_BYTES
        || args->rope_cos_f32 == 0 || args->rope_cos_len_bytes != ROPE_BYTES
        || args->rope_sin_f32 == 0 || args->rope_sin_len_bytes != ROPE_BYTES
        || args->output_f32 == 0 || args->output_len_bytes != QUERY_BYTES
        || args->tokens != 8 || args->heads != 64 || args->head_dim != 512 || args->selected != 10
        || args->query_f32 % alignof(float) != 0 || args->kv_f32 % alignof(float) != 0
        || args->sink_f32 % alignof(float) != 0 || args->indices_i32 % alignof(int32_t) != 0
        || args->rope_cos_f32 % alignof(float) != 0 || args->rope_sin_f32 % alignof(float) != 0
        || args->output_f32 % alignof(float) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_set_device(ordinal);
    if (selected.status != AUGUR_CUDA_STATUS_OK || stream->ordinal != ordinal) return selected;
    dim3 grid(8, 64, 1);
    augur_m111_sparse_attention_kernel_v1<<<grid, 256, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->kv_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->sink_f32)),
        reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(args->indices_i32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->rope_cos_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->rope_sin_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_cuda_stream_synchronize(stream);
}

__global__ void augur_m131_store_kv_kernel_v1(const float* current, float* window, uint32_t position) {
    const int pair = static_cast<int>(threadIdx.x);
    if (pair < 256) {
        window[position * 512 + pair * 2] = current[pair * 2];
        window[position * 512 + pair * 2 + 1] = current[pair * 2 + 1];
    }
}

__global__ void augur_m131_stateful_attention_kernel_v1(
    const float* query, const float* kv, const float* sink, const int32_t* indices,
    const float* rope_cos, const float* rope_sin, float* output, uint32_t visible_kv) {
    constexpr int HEAD_DIM=512, SELECTED=10;
    const int head=static_cast<int>(blockIdx.x), pair=static_cast<int>(threadIdx.x);
    if(head>=64 || pair>=256) return;
    __shared__ float reduction[256]; __shared__ float scores[SELECTED]; __shared__ float weights[SELECTED];
    const int d0=pair*2,d1=d0+1,q=head*HEAD_DIM;
    for(int s=0;s<SELECTED;++s){
        const int32_t index=indices[s];
        reduction[pair]=(index<0 || static_cast<uint32_t>(index)>=visible_kv)?0.0f:query[q+d0]*kv[index*HEAD_DIM+d0]+query[q+d1]*kv[index*HEAD_DIM+d1];
        __syncthreads(); for(int stride=128;stride;stride>>=1){if(pair<stride) reduction[pair]+=reduction[pair+stride]; __syncthreads();}
        if(pair==0) scores[s]=(index<0 || static_cast<uint32_t>(index)>=visible_kv)?-INFINITY:reduction[0]*rsqrtf(512.0f); __syncthreads();
    }
    if(pair==0){float maximum=sink[head]; for(int s=0;s<SELECTED;++s) maximum=fmaxf(maximum,scores[s]); float denominator=expf(sink[head]-maximum); for(int s=0;s<SELECTED;++s){weights[s]=isfinite(scores[s])?expf(scores[s]-maximum):0.0f; denominator+=weights[s];} const float inverse=1.0f/denominator; for(int s=0;s<SELECTED;++s) weights[s]*=inverse;} __syncthreads();
    float o0=0.0f,o1=0.0f; for(int s=0;s<SELECTED;++s){const int32_t index=indices[s]; if(index>=0 && static_cast<uint32_t>(index)<visible_kv){o0+=weights[s]*kv[index*HEAD_DIM+d0]; o1+=weights[s]*kv[index*HEAD_DIM+d1];}}
    o0=augur_m111_bf16_round(o0); o1=augur_m111_bf16_round(o1);
    if(d0>=448){const int rp=pair-224; const float c=rope_cos[rp],si=rope_sin[rp]; const float r0=o0*c+o1*si,r1=-o0*si+o1*c; o0=augur_m111_bf16_round(r0); o1=augur_m111_bf16_round(r1);}
    output[q+d0]=o0; output[q+d1]=o1;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m131_stateful_attention_f32_v1(int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_m131_stateful_attention_f32_args_v1* a){
    constexpr uint64_t Q=64ULL*512ULL*4ULL, KV=512ULL*4ULL, WINDOW=128ULL*512ULL*4ULL, SINK=64ULL*4ULL, IDX=10ULL*4ULL, ROPE=32ULL*4ULL;
    if(stream==nullptr||a==nullptr||a->abi_version!=AUGUR_CUDA_M131_STATEFUL_ATTENTION_F32_ARGS_VERSION_V1||a->struct_size!=sizeof(*a)||a->query_f32==0||a->query_len_bytes!=Q||a->current_kv_f32==0||a->current_kv_len_bytes!=KV||a->kv_window_f32==0||a->kv_window_len_bytes!=WINDOW||a->sink_f32==0||a->sink_len_bytes!=SINK||a->indices_i32==0||a->indices_len_bytes!=IDX||a->rope_cos_f32==0||a->rope_cos_len_bytes!=ROPE||a->rope_sin_f32==0||a->rope_sin_len_bytes!=ROPE||a->output_f32==0||a->output_len_bytes!=Q||a->position>=128||a->visible_kv==0||a->visible_kv>128||a->position>=a->visible_kv||a->heads!=64||a->head_dim!=512||a->selected!=10||a->reserved!=0) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    augur_cuda_result_v1 selected=augur_set_device(ordinal); if(selected.status!=AUGUR_CUDA_STATUS_OK||stream->ordinal!=ordinal) return selected;
    augur_m131_store_kv_kernel_v1<<<1,256,0,stream->stream>>>(reinterpret_cast<const float*>(static_cast<uintptr_t>(a->current_kv_f32)),reinterpret_cast<float*>(static_cast<uintptr_t>(a->kv_window_f32)),a->position);
    augur_m131_stateful_attention_kernel_v1<<<64,256,0,stream->stream>>>(reinterpret_cast<const float*>(static_cast<uintptr_t>(a->query_f32)),reinterpret_cast<const float*>(static_cast<uintptr_t>(a->kv_window_f32)),reinterpret_cast<const float*>(static_cast<uintptr_t>(a->sink_f32)),reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(a->indices_i32)),reinterpret_cast<const float*>(static_cast<uintptr_t>(a->rope_cos_f32)),reinterpret_cast<const float*>(static_cast<uintptr_t>(a->rope_sin_f32)),reinterpret_cast<float*>(static_cast<uintptr_t>(a->output_f32)),a->visible_kv);
    const cudaError_t error=cudaGetLastError(); if(error!=cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error); return augur_cuda_stream_synchronize(stream);
}

__device__ __forceinline__ float augur_m139_bf16(float value) {
    uint32_t bits = __float_as_uint(value);
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1U);
    return __uint_as_float(bits & UINT32_C(0xffff0000));
}

__device__ __forceinline__ float augur_m139_norm_value(const void* norm, bool bf16, int index) {
    if (!bf16) return reinterpret_cast<const float*>(norm)[index];
    return __uint_as_float(static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(norm)[index]) << 16);
}

__global__ void augur_m139_gate_ape(const float* gate, const float* ape, float* out, int tokens, int head) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * 2 * head;
    if (index < total) out[index] = gate[index] + ape[index % (8 * head)];
}

__global__ void augur_m139_pool_build(const float* kv, const float* score, float* values, float* scores, int groups, int head) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = groups * 8 * head;
    if (index >= total) return;
    int d = index % head, slot = (index / head) % 8, group = index / (8 * head);
    if (slot < 4) {
        if (group == 0) { values[index] = 0.0f; scores[index] = -INFINITY; }
        else { int token = (group - 1) * 4 + slot; values[index] = kv[(token * 2) * head + d]; scores[index] = score[(token * 2) * head + d]; }
    } else {
        int token = group * 4 + slot - 4;
        values[index] = kv[(token * 2 + 1) * head + d]; scores[index] = score[(token * 2 + 1) * head + d];
    }
}

__global__ void augur_m139_softmax(const float* scores, float* output, int groups, int head) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = groups * head;
    if (index >= total) return;
    int group = index / head, d = index % head, base = group * 8 * head + d;
    float maximum = -INFINITY;
    for (int slot=0; slot<8; ++slot) maximum = fmaxf(maximum, scores[base + slot * head]);
    float denominator = 0.0f;
    for (int slot=0; slot<8; ++slot) denominator += expf(scores[base + slot * head] - maximum);
    for (int slot=0; slot<8; ++slot) output[base + slot * head] = expf(scores[base + slot * head] - maximum) / denominator;
}

__global__ void augur_m139_reduce(const float* values, const float* weights, float* output, int groups, int head) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = groups * head;
    if (index >= total) return;
    int group = index / head, d = index % head, base = group * 8 * head + d;
    float sum = 0.0f;
    for (int slot=0; slot<8; ++slot) sum += values[base + slot * head] * weights[base + slot * head];
    output[index] = sum;
}

__global__ void augur_m139_bf16_cast(const float* input, float* output, int elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = augur_m139_bf16(input[index]);
}

__global__ void augur_m139_rms(const float* input, const void* norm, float* output, int groups, int head, bool norm_bf16) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= groups) return;
    float sum = 0.0f;
    for (int d=0; d<head; ++d) { float v=input[row*head+d]; sum += v*v; }
    float scale = rsqrtf(sum / static_cast<float>(head) + 1.0e-6f);
    for (int d=0; d<head; ++d) output[row*head+d] = augur_m139_bf16(input[row*head+d] * scale * augur_m139_norm_value(norm,norm_bf16,d));
}

__global__ void augur_m139_rope(const float* input, float* output, int groups, int head) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = groups * head;
    if (index >= total) return;
    int row=index/head, dimension=index%head, tail_dimension=dimension-(head-64);
    if (tail_dimension < 0) { output[index] = input[index]; return; }
    if (tail_dimension & 1) return;
    int pair=tail_dimension/2, tail=row*head+head-64;
    constexpr float pi=3.14159265358979323846f;
    float lowf=64.0f*logf(65536.0f/(32.0f*2.0f*pi))/(2.0f*logf(160000.0f));
    float highf=64.0f*logf(65536.0f/(1.0f*2.0f*pi))/(2.0f*logf(160000.0f));
    int low=max(0,static_cast<int>(floorf(lowf))), high=min(63,static_cast<int>(ceilf(highf)));
    float ramp=fminf(1.0f,fmaxf(0.0f,(pair-static_cast<float>(low))/static_cast<float>(max(1,high-low))));
    float smooth=1.0f-ramp;
    float base_frequency=1.0f/powf(160000.0f,(pair*2.0f)/64.0f);
    float frequency=base_frequency/16.0f*(1.0f-smooth)+base_frequency*smooth;
    float angle=static_cast<float>(row*4)*frequency, c=cosf(angle), s=sinf(angle);
    float left=input[tail+pair*2], right=input[tail+pair*2+1];
    output[tail+pair*2]=augur_m139_bf16(left*c-right*s);
    output[tail+pair*2+1]=augur_m139_bf16(left*s+right*c);
}

__global__ void augur_m139_hadamard(const float* input, float* output, int groups) {
    int row=blockIdx.x, lane=threadIdx.x; __shared__ float values[128];
    if(row>=groups || lane>=128) return; values[lane]=input[row*128+lane]; __syncthreads();
    for(int step=1;step<128;step*=2){int group=(lane/(step*2))*(step*2), offset=lane%(step*2); float value;
        if(offset<step) value=values[group+offset]+values[group+offset+step]; else value=values[group+offset-step]-values[group+offset];
        __syncthreads(); values[lane]=value; __syncthreads(); }
    output[row*128+lane]=augur_m139_bf16(values[lane]*0.08838834764831845f);
}

__device__ float augur_m139_nearest_fp4(float x){const float c[15]={-6,-4,-3,-2,-1.5f,-1,-.5f,0,.5f,1,1.5f,2,3,4,6};float best=c[0],dist=fabsf(x-best);for(int i=1;i<15;++i){float d=fabsf(x-c[i]);if(d<dist){dist=d;best=c[i];}}return best;}
__device__ float augur_m139_nearest_fp8(float x){float best=0.0f,dist=fabsf(x);for(int code=0;code<256;++code){int sign=(code&128)?-1:1,exp=(code>>3)&15,mant=code&7;if(exp==15&&mant==7)continue;float v=exp==0?sign*(mant/8.0f)*ldexpf(1.0f,-6):sign*(1.0f+mant/8.0f)*ldexpf(1.0f,exp-7);float d=fabsf(x-v);if(d<dist||(d==dist&&v<best)){dist=d;best=v;}}return best;}
#if defined(AUGUR_CUDA_P0_PREFILL_V21_RATIO4_FAST_EXACT_QDQ)
__device__ __forceinline__ float augur_m139_fp8_finite_value(int code){int exponent=(code>>3)&15,mantissa=code&7;return exponent==0?(mantissa/8.0f)*ldexpf(1.0f,-6):(1.0f+mantissa/8.0f)*ldexpf(1.0f,exponent-7);}
__device__ __forceinline__ float augur_m139_nearest_fp8_fast_exact(float x){
    if(isnan(x))return 0.0f;
    const float magnitude=fabsf(x);
    int lower_code=0,upper_code=126;
    while(lower_code<upper_code){
        const int middle=(lower_code+upper_code+1)>>1;
        if(augur_m139_fp8_finite_value(middle)<=magnitude)lower_code=middle;
        else upper_code=middle-1;
    }
    const int higher_code=min(lower_code+1,126);
    const float lower=augur_m139_fp8_finite_value(lower_code),higher=augur_m139_fp8_finite_value(higher_code);
    const float lower_distance=magnitude-lower,higher_distance=higher-magnitude;
    const float value=(lower_distance<higher_distance||(lower_distance==higher_distance&&x>=0.0f))?lower:higher;
    if(value==0.0f)return 0.0f;
    return x<0.0f?-value:value;
}
__global__ void augur_m139_qdq_fast_exact(const float* input,float* qdq,float* output,int groups,int head,int kind){
    int group=blockIdx.x*blockDim.x+threadIdx.x;int width=kind==0?head-64:head,gs=kind==0?64:32,total_groups=groups*(width/gs);if(group>=total_groups)return;int row=group/(width/gs),local=group%(width/gs),base=row*width+local*gs;float amax=0;for(int i=0;i<gs;++i)amax=fmaxf(amax,fabsf(input[row*head+local*gs+i]));float maximum=kind==0?448.0f:6.0f,minimum=kind==0?1.0e-4f:ldexpf(6.0f,-126);amax=fmaxf(amax,minimum);float scale=exp2f(ceilf(log2f(amax/maximum)));for(int i=0;i<gs;++i){float x=fminf(maximum,fmaxf(-maximum,input[row*head+local*gs+i]/scale));float v=(kind==0?augur_m139_nearest_fp8_fast_exact(x):augur_m139_nearest_fp4(x))*scale;qdq[base+i]=v;output[row*head+local*gs+i]=v;}if(kind==0&&local==0){for(int i=head-64;i<head;++i)output[row*head+i]=input[row*head+i];}}
#endif
__global__ void augur_m139_qdq(const float* input,float* qdq,float* output,int groups,int head,int kind){int group=blockIdx.x*blockDim.x+threadIdx.x;int width=kind==0?head-64:head,gs=kind==0?64:32,total_groups=groups*(width/gs);if(group>=total_groups)return;int row=group/(width/gs),local=group%(width/gs),base=row*width+local*gs;float amax=0;for(int i=0;i<gs;++i)amax=fmaxf(amax,fabsf(input[row*head+local*gs+i]));float maximum=kind==0?448.0f:6.0f,minimum=kind==0?1.0e-4f:ldexpf(6.0f,-126);amax=fmaxf(amax,minimum);float scale=exp2f(ceilf(log2f(amax/maximum)));for(int i=0;i<gs;++i){float x=fminf(maximum,fmaxf(-maximum,input[row*head+local*gs+i]/scale));float v=(kind==0?augur_m139_nearest_fp8(x):augur_m139_nearest_fp4(x))*scale;qdq[base+i]=v;output[row*head+local*gs+i]=v;}if(kind==0&&local==0){for(int i=head-64;i<head;++i)output[row*head+i]=input[row*head+i];}}

// Historical M139 evidence binds this symbol and its reduction order. Keep its
// implementation frozen; later milestones must use a new versioned entrypoint.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m139_ratio4_staged_v1(int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_m139_ratio4_staged_args_v1* a){
    if(!stream||!a||a->abi_version!=AUGUR_CUDA_M139_RATIO4_STAGED_ARGS_VERSION_V1||a->struct_size!=sizeof(*a)||a->tokens==0||a->tokens%4||!((a->kind==0&&a->head_dim==512&&!a->norm_is_bf16)||(a->kind==1&&a->head_dim==128&&a->norm_is_bf16))) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    int tokens=a->tokens,head=a->head_dim,groups=tokens/4,projection=tokens*2*head,pool=groups*8*head,rows=groups*head,qdq=groups*(a->kind==0?head-64:head);
    auto exact=[](uint64_t p,uint64_t n,uint64_t e){return p!=0&&n==e;};
    if(!exact(a->kv_projection,a->kv_projection_len_bytes,projection*4ULL)||!exact(a->gate_projection,a->gate_projection_len_bytes,projection*4ULL)||!exact(a->ape_f32,a->ape_len_bytes,8ULL*head*4ULL)||!exact(a->norm,a->norm_len_bytes,head*(a->norm_is_bf16?2ULL:4ULL))||!exact(a->gate_plus_ape,a->gate_plus_ape_len_bytes,projection*4ULL)||!exact(a->pooled_values,a->pooled_values_len_bytes,pool*4ULL)||!exact(a->pooled_scores,a->pooled_scores_len_bytes,pool*4ULL)||!exact(a->softmax,a->softmax_len_bytes,pool*4ULL)||!exact(a->weighted_pool,a->weighted_pool_len_bytes,rows*4ULL)||!exact(a->bf16_boundary,a->bf16_boundary_len_bytes,rows*4ULL)||!exact(a->rms_norm,a->rms_norm_len_bytes,rows*4ULL)||!exact(a->rope,a->rope_len_bytes,rows*4ULL)||!exact(a->qdq,a->qdq_len_bytes,qdq*4ULL)||!exact(a->output,a->output_len_bytes,rows*4ULL)||(a->kind==1&&!exact(a->hadamard,a->hadamard_len_bytes,rows*4ULL))||(a->kind==0&&(a->hadamard||a->hadamard_len_bytes))) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    augur_cuda_result_v1 selected=augur_set_device(ordinal);if(selected.status!=0||stream->ordinal!=ordinal)return selected;int threads=256;
    const float*kv=(const float*)(uintptr_t)a->kv_projection;const float*gate=(const float*)(uintptr_t)a->gate_projection;const float*ape=(const float*)(uintptr_t)a->ape_f32;float*g=(float*)(uintptr_t)a->gate_plus_ape;float*pv=(float*)(uintptr_t)a->pooled_values;float*ps=(float*)(uintptr_t)a->pooled_scores;float*sm=(float*)(uintptr_t)a->softmax;float*wp=(float*)(uintptr_t)a->weighted_pool;float*bf=(float*)(uintptr_t)a->bf16_boundary;float*rn=(float*)(uintptr_t)a->rms_norm;float*ro=(float*)(uintptr_t)a->rope;float*ha=(float*)(uintptr_t)a->hadamard;float*q=(float*)(uintptr_t)a->qdq;float*out=(float*)(uintptr_t)a->output;
    augur_m139_gate_ape<<<(projection+threads-1)/threads,threads,0,stream->stream>>>(gate,ape,g,tokens,head);augur_m139_pool_build<<<(pool+threads-1)/threads,threads,0,stream->stream>>>(kv,g,pv,ps,groups,head);augur_m139_softmax<<<(rows+threads-1)/threads,threads,0,stream->stream>>>(ps,sm,groups,head);augur_m139_reduce<<<(rows+threads-1)/threads,threads,0,stream->stream>>>(pv,sm,wp,groups,head);augur_m139_bf16_cast<<<(rows+threads-1)/threads,threads,0,stream->stream>>>(wp,bf,rows);augur_m139_rms<<<1,32,0,stream->stream>>>(bf,(const void*)(uintptr_t)a->norm,rn,groups,head,a->norm_is_bf16!=0);augur_m139_rope<<<(max(rows,groups*32)+threads-1)/threads,threads,0,stream->stream>>>(rn,ro,groups,head);const float*source=ro;if(a->kind==1){augur_m139_hadamard<<<groups,128,0,stream->stream>>>(ro,ha,groups);source=ha;}augur_m139_qdq<<<(groups*((a->kind==0?head-64:head)/(a->kind==0?64:32))+threads-1)/threads,threads,0,stream->stream>>>(source,q,out,groups,head,a->kind);
    cudaError_t error=cudaGetLastError();if(error!=cudaSuccess)return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);return augur_cuda_stream_synchronize(stream);
}

// M287 owns the Graph-exact RMS path. This separate symbol avoids changing the
// M139 numerical identity while retaining the already frozen argument layout.
__global__ void augur_m287_rms_graph_exact(const float* input,const void* norm,float* output,int groups,int head,bool norm_bf16){int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=groups)return;float sum=0.0f;for(int d=0;d<head;++d){float v=input[row*head+d];sum=__fadd_rn(sum,__fmul_rn(v,v));}float mean=__fdiv_rn(sum,(float)head);float scale=__fdiv_rn(1.0f,sqrtf(__fadd_rn(mean,1.0e-6f)));for(int d=0;d<head;++d){float scaled=__fmul_rn(input[row*head+d],scale);output[row*head+d]=augur_m139_bf16(__fmul_rn(scaled,augur_m139_norm_value(norm,norm_bf16,d)));}}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m287_ratio4_staged_submit_v1(int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_m139_ratio4_staged_args_v1* a){
    if(!stream||!a||a->abi_version!=AUGUR_CUDA_M139_RATIO4_STAGED_ARGS_VERSION_V1||a->struct_size!=sizeof(*a)||a->tokens==0||a->tokens%4||!((a->kind==0&&a->head_dim==512&&!a->norm_is_bf16)||(a->kind==1&&a->head_dim==128&&a->norm_is_bf16))) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    int tokens=a->tokens,head=a->head_dim,groups=tokens/4,projection=tokens*2*head,pool=groups*8*head,rows=groups*head,qdq=groups*(a->kind==0?head-64:head);
    auto exact=[](uint64_t p,uint64_t n,uint64_t e){return p!=0&&n==e;};
    if(!exact(a->kv_projection,a->kv_projection_len_bytes,projection*4ULL)||!exact(a->gate_projection,a->gate_projection_len_bytes,projection*4ULL)||!exact(a->ape_f32,a->ape_len_bytes,8ULL*head*4ULL)||!exact(a->norm,a->norm_len_bytes,head*(a->norm_is_bf16?2ULL:4ULL))||!exact(a->gate_plus_ape,a->gate_plus_ape_len_bytes,projection*4ULL)||!exact(a->pooled_values,a->pooled_values_len_bytes,pool*4ULL)||!exact(a->pooled_scores,a->pooled_scores_len_bytes,pool*4ULL)||!exact(a->softmax,a->softmax_len_bytes,pool*4ULL)||!exact(a->weighted_pool,a->weighted_pool_len_bytes,rows*4ULL)||!exact(a->bf16_boundary,a->bf16_boundary_len_bytes,rows*4ULL)||!exact(a->rms_norm,a->rms_norm_len_bytes,rows*4ULL)||!exact(a->rope,a->rope_len_bytes,rows*4ULL)||!exact(a->qdq,a->qdq_len_bytes,qdq*4ULL)||!exact(a->output,a->output_len_bytes,rows*4ULL)||(a->kind==1&&!exact(a->hadamard,a->hadamard_len_bytes,rows*4ULL))||(a->kind==0&&(a->hadamard||a->hadamard_len_bytes))) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
    augur_cuda_result_v1 selected=augur_set_device(ordinal);if(selected.status!=0||stream->ordinal!=ordinal)return selected;int threads=256;
    const float*kv=(const float*)(uintptr_t)a->kv_projection;const float*gate=(const float*)(uintptr_t)a->gate_projection;const float*ape=(const float*)(uintptr_t)a->ape_f32;float*g=(float*)(uintptr_t)a->gate_plus_ape;float*pv=(float*)(uintptr_t)a->pooled_values;float*ps=(float*)(uintptr_t)a->pooled_scores;float*sm=(float*)(uintptr_t)a->softmax;float*wp=(float*)(uintptr_t)a->weighted_pool;float*bf=(float*)(uintptr_t)a->bf16_boundary;float*rn=(float*)(uintptr_t)a->rms_norm;float*ro=(float*)(uintptr_t)a->rope;float*ha=(float*)(uintptr_t)a->hadamard;float*q=(float*)(uintptr_t)a->qdq;float*out=(float*)(uintptr_t)a->output;
    augur_m139_gate_ape<<<(projection+threads-1)/threads,threads,0,stream->stream>>>(gate,ape,g,tokens,head);augur_m139_pool_build<<<(pool+threads-1)/threads,threads,0,stream->stream>>>(kv,g,pv,ps,groups,head);augur_m139_softmax<<<(rows+threads-1)/threads,threads,0,stream->stream>>>(ps,sm,groups,head);augur_m139_reduce<<<(rows+threads-1)/threads,threads,0,stream->stream>>>(pv,sm,wp,groups,head);augur_m139_bf16_cast<<<(rows+threads-1)/threads,threads,0,stream->stream>>>(wp,bf,rows);augur_m287_rms_graph_exact<<<1,32,0,stream->stream>>>(bf,(const void*)(uintptr_t)a->norm,rn,groups,head,a->norm_is_bf16!=0);augur_m139_rope<<<(max(rows,groups*32)+threads-1)/threads,threads,0,stream->stream>>>(rn,ro,groups,head);const float*source=ro;if(a->kind==1){augur_m139_hadamard<<<groups,128,0,stream->stream>>>(ro,ha,groups);source=ha;}
#if defined(AUGUR_CUDA_P0_PREFILL_V21_RATIO4_FAST_EXACT_QDQ)
    augur_m139_qdq_fast_exact<<<(groups*((a->kind==0?head-64:head)/(a->kind==0?64:32))+threads-1)/threads,threads,0,stream->stream>>>(source,q,out,groups,head,a->kind);
#else
    augur_m139_qdq<<<(groups*((a->kind==0?head-64:head)/(a->kind==0?64:32))+threads-1)/threads,threads,0,stream->stream>>>(source,q,out,groups,head,a->kind);
#endif
    cudaError_t error=cudaGetLastError();if(error!=cudaSuccess)return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,error);return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m287_ratio4_staged_v1(int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_m139_ratio4_staged_args_v1* a){
    augur_cuda_result_v1 submitted=augur_cuda_m287_ratio4_staged_submit_v1(ordinal,stream,a);
    return submitted.status==AUGUR_CUDA_STATUS_OK?augur_cuda_stream_synchronize(stream):submitted;
}

__global__ void augur_m142_ordered_topk_kernel(const float* scores, int32_t* indices,
    float* selected, int tokens, int candidates, int top_k, int ratio, int base, int padding) {
    int row = blockIdx.x;
    if (row >= tokens || threadIdx.x != 0) return;
    int valid = (row + 1) / ratio;
    for (int out = 0; out < top_k; ++out) { indices[row * top_k + out] = padding; selected[row * top_k + out] = -1.0e30f; }
    for (int candidate = 0; candidate < candidates && candidate < valid; ++candidate) {
        float value = scores[row * candidates + candidate];
        int slot = -1;
        for (int out = 0; out < top_k; ++out) {
            float current = selected[row * top_k + out];
            int current_id = indices[row * top_k + out] - base;
            if (slot < 0 && (value > current || (value == current && (current_id < 0 || candidate < current_id)))) slot = out;
        }
        if (slot >= 0) {
            for (int out = top_k - 1; out > slot; --out) { selected[row * top_k + out] = selected[row * top_k + out - 1]; indices[row * top_k + out] = indices[row * top_k + out - 1]; }
            selected[row * top_k + slot] = value; indices[row * top_k + slot] = candidate + base;
        }
    }
}

static bool augur_m142_range_valid(uint64_t pointer, uint64_t length) {
    return pointer != 0 && length != 0 && (pointer & UINT64_C(3)) == 0 &&
        pointer <= UINT64_MAX - length;
}

static bool augur_m142_ranges_overlap(
    uint64_t left_pointer,
    uint64_t left_length,
    uint64_t right_pointer,
    uint64_t right_length
) {
    return left_pointer < right_pointer + right_length &&
        right_pointer < left_pointer + left_length;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m142_ordered_topk_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_m142_ordered_topk_args_v1* a) {
    if (!stream || !a || a->abi_version != AUGUR_CUDA_M142_ORDERED_TOPK_ARGS_VERSION_V1 ||
        a->struct_size != sizeof(*a) || a->tokens == 0 || a->candidates == 0 ||
        a->top_k == 0 || a->top_k > a->candidates || a->causal_ratio == 0 || a->index_base < 0 ||
        a->padding >= a->index_base) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    const uint64_t token_count = a->tokens;
    const uint64_t candidate_count = a->candidates;
    const uint64_t top_k = a->top_k;
    if (candidate_count != 0 && token_count > UINT64_MAX / candidate_count) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t score_elements = token_count * candidate_count;
    if (top_k != 0 && token_count > UINT64_MAX / top_k) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t output_elements = token_count * top_k;
    if (score_elements > UINT64_MAX / 4 || output_elements > UINT64_MAX / 4 ||
        a->scores_len_bytes != score_elements * 4 ||
        a->indices_len_bytes != output_elements * 4 ||
        a->selected_scores_len_bytes != output_elements * 4 ||
        !augur_m142_range_valid(a->scores, a->scores_len_bytes) ||
        !augur_m142_range_valid(a->indices, a->indices_len_bytes) ||
        !augur_m142_range_valid(a->selected_scores, a->selected_scores_len_bytes) ||
        augur_m142_ranges_overlap(a->scores, a->scores_len_bytes, a->indices, a->indices_len_bytes) ||
        augur_m142_ranges_overlap(a->scores, a->scores_len_bytes, a->selected_scores, a->selected_scores_len_bytes) ||
        augur_m142_ranges_overlap(a->indices, a->indices_len_bytes, a->selected_scores, a->selected_scores_len_bytes)) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected_device = augur_set_device(ordinal); if (selected_device.status != 0 || stream->ordinal != ordinal) return selected_device;
    augur_m142_ordered_topk_kernel<<<a->tokens, 32, 0, stream->stream>>>(
        (const float*)(uintptr_t)a->scores, (int32_t*)(uintptr_t)a->indices,
        (float*)(uintptr_t)a->selected_scores, a->tokens, a->candidates, a->top_k,
        a->causal_ratio, a->index_base, a->padding);
    cudaError_t error = cudaGetLastError(); if (error != cudaSuccess) return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
    return augur_ok();
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m142_ordered_topk_v1(
    int32_t ordinal, augur_cuda_stream_t stream, const augur_cuda_m142_ordered_topk_args_v1* a) {
    augur_cuda_result_v1 submitted = augur_cuda_m142_ordered_topk_submit_v1(ordinal, stream, a);
    return submitted.status == AUGUR_CUDA_STATUS_OK ? augur_cuda_stream_synchronize(stream) : submitted;
}

__global__ void augur_m144_gate(const float* gate,const float* ape,float* out){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<65536)out[i]=gate[i]+ape[i];}
__global__ void augur_m144_softmax(const float* score,float* out){int d=blockIdx.x*blockDim.x+threadIdx.x;if(d>=512)return;float m=-1.0e30f;for(int t=0;t<128;++t)m=fmaxf(m,score[t*512+d]);float s=0;for(int t=0;t<128;++t)s+=expf(score[t*512+d]-m);for(int t=0;t<128;++t)out[t*512+d]=expf(score[t*512+d]-m)/s;}
__global__ void augur_m144_weight(const float* kv,const float* sm,float* out){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<65536)out[i]=kv[i]*sm[i];}
__global__ void augur_m144_reduce(const float* in,float* out){int d=blockIdx.x*blockDim.x+threadIdx.x;if(d>=512)return;float s=0;for(int t=0;t<128;++t)s+=in[t*512+d];out[d]=s;}
__global__ void augur_m144_cast(const float* in,float* out){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<512)out[i]=augur_m139_bf16(in[i]);}
__global__ void augur_m144_rms(const float* in,const float* norm,float* out){if(threadIdx.x)return;float s=0;for(int d=0;d<512;++d)s+=in[d]*in[d];float scale=rsqrtf(s/512.0f+1.0e-6f);for(int d=0;d<512;++d)out[d]=augur_m139_bf16(in[d]*scale*norm[d]);}
#if defined(AUGUR_CUDA_CTX12_RATIO128_SKIP_ROW_VALIDATION_CANDIDATE)
__device__ float augur_ctx12_nearest_fp8_unrolled(float x) {
    float best = 0.0f;
    float dist = fabsf(x);
#pragma unroll 256
    for (int code = 0; code < 256; ++code) {
        const int sign = (code & 128) ? -1 : 1;
        const int exp = (code >> 3) & 15;
        const int mant = code & 7;
        if (exp == 15 && mant == 7) continue;
        const float v = exp == 0
            ? sign * (mant / 8.0f) * ldexpf(1.0f, -6)
            : sign * (1.0f + mant / 8.0f) * ldexpf(1.0f, exp - 7);
        const float d = fabsf(x - v);
        if (d < dist || (d == dist && v < best)) {
            dist = d;
            best = v;
        }
    }
    return best;
}
__global__ void augur_m144_qdq(const float* in, float* qdq, float* out) {
    const int g = threadIdx.x;
    if (g >= 7) return;
    float a = 0.0f;
    for (int i = 0; i < 64; ++i) a = fmaxf(a, fabsf(in[g * 64 + i]));
    a = fmaxf(a, 1.0e-4f);
    const float scale = exp2f(ceilf(log2f(a / 448.0f)));
    for (int i = 0; i < 64; ++i) {
        const float x = fminf(448.0f, fmaxf(-448.0f, in[g * 64 + i] / scale));
        const float v = augur_ctx12_nearest_fp8_unrolled(x) * scale;
        qdq[g * 64 + i] = v;
        out[g * 64 + i] = v;
    }
    if (g == 0) {
        for (int i = 448; i < 512; ++i) out[i] = in[i];
    }
}
#else
__global__ void augur_m144_qdq(const float* in,float* qdq,float* out){int g=threadIdx.x;if(g>=7)return;float a=0;for(int i=0;i<64;++i)a=fmaxf(a,fabsf(in[g*64+i]));a=fmaxf(a,1.0e-4f);float scale=exp2f(ceilf(log2f(a/448.0f)));for(int i=0;i<64;++i){float x=fminf(448.0f,fmaxf(-448.0f,in[g*64+i]/scale));float v=augur_m139_nearest_fp8(x)*scale;qdq[g*64+i]=v;out[g*64+i]=v;}if(g==0)for(int i=448;i<512;++i)out[i]=in[i];}
#endif
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m144_ratio128_staged_v1(int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_m144_ratio128_staged_args_v1* a){
 if(!stream||!a||a->abi_version!=AUGUR_CUDA_M144_RATIO128_STAGED_ARGS_VERSION_V1||a->struct_size!=sizeof(*a))return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);auto ex=[](uint64_t p,uint64_t n,uint64_t e){return p&&n==e;};
 if(!ex(a->kv_projection,a->kv_projection_len_bytes,262144)||!ex(a->gate_projection,a->gate_projection_len_bytes,262144)||!ex(a->ape_f32,a->ape_len_bytes,262144)||!ex(a->norm_f32,a->norm_len_bytes,2048)||!ex(a->gate_plus_ape,a->gate_plus_ape_len_bytes,262144)||!ex(a->softmax,a->softmax_len_bytes,262144)||!ex(a->weighted,a->weighted_len_bytes,262144)||!ex(a->reduced,a->reduced_len_bytes,2048)||!ex(a->bf16_boundary,a->bf16_boundary_len_bytes,2048)||!ex(a->rms_norm,a->rms_norm_len_bytes,2048)||!ex(a->rope,a->rope_len_bytes,2048)||!ex(a->qdq,a->qdq_len_bytes,1792)||!ex(a->output,a->output_len_bytes,2048))return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
 augur_cuda_result_v1 s=augur_set_device(ordinal);if(s.status||stream->ordinal!=ordinal)return s;int th=256;augur_m144_gate<<<256,th,0,stream->stream>>>((const float*)(uintptr_t)a->gate_projection,(const float*)(uintptr_t)a->ape_f32,(float*)(uintptr_t)a->gate_plus_ape);augur_m144_softmax<<<2,256,0,stream->stream>>>((const float*)(uintptr_t)a->gate_plus_ape,(float*)(uintptr_t)a->softmax);augur_m144_weight<<<256,th,0,stream->stream>>>((const float*)(uintptr_t)a->kv_projection,(const float*)(uintptr_t)a->softmax,(float*)(uintptr_t)a->weighted);augur_m144_reduce<<<2,256,0,stream->stream>>>((const float*)(uintptr_t)a->weighted,(float*)(uintptr_t)a->reduced);augur_m144_cast<<<2,256,0,stream->stream>>>((const float*)(uintptr_t)a->reduced,(float*)(uintptr_t)a->bf16_boundary);augur_m144_rms<<<1,32,0,stream->stream>>>((const float*)(uintptr_t)a->bf16_boundary,(const float*)(uintptr_t)a->norm_f32,(float*)(uintptr_t)a->rms_norm);augur_m144_cast<<<2,256,0,stream->stream>>>((const float*)(uintptr_t)a->rms_norm,(float*)(uintptr_t)a->rope);augur_m144_qdq<<<1,32,0,stream->stream>>>((const float*)(uintptr_t)a->rope,(float*)(uintptr_t)a->qdq,(float*)(uintptr_t)a->output);cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,e);return augur_cuda_stream_synchronize(stream);}

// M288 reuses the frozen 216-byte layout through a separate identity. Keep the
// M144 entrypoint unchanged even when a later milestone needs new semantics.
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m288_ratio128_staged_submit_v1(int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_m144_ratio128_staged_args_v1* a){
 if(!stream||!a)return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);if(a->abi_version!=AUGUR_CUDA_M144_RATIO128_STAGED_ARGS_VERSION_V1||a->struct_size!=sizeof(*a))return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH,cudaSuccess);auto ex=[](uint64_t p,uint64_t n,uint64_t e){return p&&n==e;};
 if(!ex(a->kv_projection,a->kv_projection_len_bytes,262144)||!ex(a->gate_projection,a->gate_projection_len_bytes,262144)||!ex(a->ape_f32,a->ape_len_bytes,262144)||!ex(a->norm_f32,a->norm_len_bytes,2048)||!ex(a->gate_plus_ape,a->gate_plus_ape_len_bytes,262144)||!ex(a->softmax,a->softmax_len_bytes,262144)||!ex(a->weighted,a->weighted_len_bytes,262144)||!ex(a->reduced,a->reduced_len_bytes,2048)||!ex(a->bf16_boundary,a->bf16_boundary_len_bytes,2048)||!ex(a->rms_norm,a->rms_norm_len_bytes,2048)||!ex(a->rope,a->rope_len_bytes,2048)||!ex(a->qdq,a->qdq_len_bytes,1792)||!ex(a->output,a->output_len_bytes,2048))return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);
 augur_cuda_result_v1 s=augur_set_device(ordinal);if(s.status)return s;if(stream->ordinal!=ordinal)return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,cudaSuccess);int th=256;augur_m144_gate<<<256,th,0,stream->stream>>>((const float*)(uintptr_t)a->gate_projection,(const float*)(uintptr_t)a->ape_f32,(float*)(uintptr_t)a->gate_plus_ape);augur_m144_softmax<<<2,256,0,stream->stream>>>((const float*)(uintptr_t)a->gate_plus_ape,(float*)(uintptr_t)a->softmax);augur_m144_weight<<<256,th,0,stream->stream>>>((const float*)(uintptr_t)a->kv_projection,(const float*)(uintptr_t)a->softmax,(float*)(uintptr_t)a->weighted);augur_m144_reduce<<<2,256,0,stream->stream>>>((const float*)(uintptr_t)a->weighted,(float*)(uintptr_t)a->reduced);augur_m144_cast<<<2,256,0,stream->stream>>>((const float*)(uintptr_t)a->reduced,(float*)(uintptr_t)a->bf16_boundary);augur_m287_rms_graph_exact<<<1,32,0,stream->stream>>>((const float*)(uintptr_t)a->bf16_boundary,(const void*)(uintptr_t)a->norm_f32,(float*)(uintptr_t)a->rms_norm,1,512,false);augur_m144_cast<<<2,256,0,stream->stream>>>((const float*)(uintptr_t)a->rms_norm,(float*)(uintptr_t)a->rope);augur_m144_qdq<<<1,32,0,stream->stream>>>((const float*)(uintptr_t)a->rope,(float*)(uintptr_t)a->qdq,(float*)(uintptr_t)a->output);cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,e);return augur_ok();}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m288_ratio128_staged_v1(int32_t ordinal,augur_cuda_stream_t stream,const augur_cuda_m144_ratio128_staged_args_v1* a){
 augur_cuda_result_v1 submitted=augur_cuda_m288_ratio128_staged_submit_v1(ordinal,stream,a);return submitted.status==AUGUR_CUDA_STATUS_OK?augur_cuda_stream_synchronize(stream):submitted;}

static bool augur_m283_view_is_zero(const augur_cuda_m283_device_view_v1& view) {
    return view.base == 0 && view.allocation_extent_bytes == 0 &&
        view.view_offset_bytes == 0 && view.view_extent_bytes == 0;
}

static bool augur_m283_active_view_valid(
    const augur_cuda_m283_device_view_v1& view,
    uint64_t required_extent
) {
    const uint64_t maximum = ~UINT64_C(0);
    if (view.base == 0 || view.allocation_extent_bytes == 0 ||
        view.view_extent_bytes != required_extent ||
        (view.base & UINT64_C(3)) != 0 ||
        (view.allocation_extent_bytes & UINT64_C(3)) != 0 ||
        (view.view_offset_bytes & UINT64_C(3)) != 0 ||
        (view.view_extent_bytes & UINT64_C(3)) != 0) {
        return false;
    }
    if (view.base > maximum - view.allocation_extent_bytes ||
        view.view_offset_bytes > maximum - view.view_extent_bytes) {
        return false;
    }
    const uint64_t relative_end = view.view_offset_bytes + view.view_extent_bytes;
    if (relative_end > view.allocation_extent_bytes ||
        view.base > maximum - view.view_offset_bytes) {
        return false;
    }
    const uint64_t resolved = view.base + view.view_offset_bytes;
    return resolved <= maximum - view.view_extent_bytes;
}

static bool augur_m283_views_overlap(
    const augur_cuda_m283_device_view_v1& left,
    const augur_cuda_m283_device_view_v1& right
) {
    const uint64_t left_start = left.base + left.view_offset_bytes;
    const uint64_t right_start = right.base + right.view_offset_bytes;
    const uint64_t left_end = left_start + left.view_extent_bytes;
    const uint64_t right_end = right_start + right.view_extent_bytes;
    return left_start < right_end && right_start < left_end;
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m283_validate_kv_transition_v1(
    const augur_cuda_m283_kv_transition_args_v1* a
) {
    if (a == nullptr) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    if (a->abi_version != AUGUR_CUDA_M283_KV_TRANSITION_ARGS_VERSION_V1 ||
        a->struct_size != sizeof(*a)) {
        return augur_error(AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH, cudaSuccess);
    }
    if (a->reserved0 != 0 || a->reserved1 != 0 || a->layer >= 43 ||
        a->capacity != AUGUR_CUDA_M283_KV_CAPACITY_V1 ||
        a->position >= a->capacity || a->next_position != a->position) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint32_t expected_variant = a->layer < 2
        ? AUGUR_CUDA_M283_KV_VARIANT_RAW_V1
        : ((a->layer & 1U) == 0U
            ? AUGUR_CUDA_M283_KV_VARIANT_RATIO4_V1
            : AUGUR_CUDA_M283_KV_VARIANT_RATIO128_V1);
    if (a->variant != expected_variant) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    constexpr uint64_t raw_row_bytes = UINT64_C(512) * UINT64_C(4);
    constexpr uint64_t raw_cache_bytes = UINT64_C(128) * raw_row_bytes;
    constexpr uint64_t ratio4_main_cache_bytes = UINT64_C(32) * raw_row_bytes;
    constexpr uint64_t ratio4_index_row_bytes = UINT64_C(128) * UINT64_C(4);
    constexpr uint64_t ratio4_index_cache_bytes = UINT64_C(32) * ratio4_index_row_bytes;
    if (!augur_m283_active_view_valid(a->raw_current, raw_row_bytes) ||
        !augur_m283_active_view_valid(a->raw_cache, raw_cache_bytes)) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    const augur_cuda_m283_device_view_v1* active[6] = {&a->raw_current, &a->raw_cache};
    int active_count = 2;
    bool metadata_valid = false;
    if (a->variant == AUGUR_CUDA_M283_KV_VARIANT_RAW_V1) {
        metadata_valid = a->compression_ratio == 0 &&
            a->source_start_position == a->position && a->source_rows == 1 &&
            a->compressed_row_index == 0 && a->compressed_position == 0 &&
            a->compressed_main_rows_before == 0 &&
            a->compressed_index_rows_before == 0 &&
            augur_m283_view_is_zero(a->compressed_main_current) &&
            augur_m283_view_is_zero(a->compressed_main_cache) &&
            augur_m283_view_is_zero(a->compressed_index_current) &&
            augur_m283_view_is_zero(a->compressed_index_cache);
    } else if (a->variant == AUGUR_CUDA_M283_KV_VARIANT_RATIO4_V1) {
        const uint32_t before = a->position / 4;
        const bool due = (a->position & 3U) == 3U;
        metadata_valid = a->compression_ratio == 4 &&
            a->compressed_row_index == before &&
            a->compressed_main_rows_before == before &&
            a->compressed_index_rows_before == before &&
            augur_m283_active_view_valid(a->compressed_main_cache, ratio4_main_cache_bytes) &&
            augur_m283_active_view_valid(a->compressed_index_cache, ratio4_index_cache_bytes);
        if (due) {
            const uint32_t expected_start = a->position == 3 ? 0 : a->position - 7;
            const uint32_t expected_rows = a->position == 3 ? 4 : 8;
            metadata_valid = metadata_valid &&
                a->source_start_position == expected_start &&
                a->source_rows == expected_rows &&
                a->compressed_position == a->position - 3 &&
                augur_m283_active_view_valid(a->compressed_main_current, raw_row_bytes) &&
                augur_m283_active_view_valid(a->compressed_index_current, ratio4_index_row_bytes);
            active[active_count++] = &a->compressed_main_current;
            active[active_count++] = &a->compressed_index_current;
        } else {
            metadata_valid = metadata_valid && a->source_start_position == 0 &&
                a->source_rows == 0 && a->compressed_position == 0 &&
                augur_m283_view_is_zero(a->compressed_main_current) &&
                augur_m283_view_is_zero(a->compressed_index_current);
        }
        active[active_count++] = &a->compressed_main_cache;
        active[active_count++] = &a->compressed_index_cache;
    } else if (a->variant == AUGUR_CUDA_M283_KV_VARIANT_RATIO128_V1) {
        const bool due = a->position == 127;
        metadata_valid = a->compression_ratio == 128 &&
            a->compressed_row_index == 0 && a->compressed_position == 0 &&
            a->compressed_main_rows_before == 0 &&
            a->compressed_index_rows_before == 0 &&
            augur_m283_active_view_valid(a->compressed_main_cache, raw_row_bytes) &&
            augur_m283_view_is_zero(a->compressed_index_current) &&
            augur_m283_view_is_zero(a->compressed_index_cache);
        if (due) {
            metadata_valid = metadata_valid && a->source_start_position == 0 &&
                a->source_rows == 128 &&
                augur_m283_active_view_valid(a->compressed_main_current, raw_row_bytes);
            active[active_count++] = &a->compressed_main_current;
        } else {
            metadata_valid = metadata_valid && a->source_start_position == 0 &&
                a->source_rows == 0 && augur_m283_view_is_zero(a->compressed_main_current);
        }
        active[active_count++] = &a->compressed_main_cache;
    }
    if (!metadata_valid) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    for (int left = 0; left < active_count; ++left) {
        for (int right = left + 1; right < active_count; ++right) {
            if (active[left]->base == active[right]->base &&
                active[left]->allocation_extent_bytes != active[right]->allocation_extent_bytes) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
            if (augur_m283_views_overlap(*active[left], *active[right])) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    return augur_ok();
}

__device__ __forceinline__ unsigned char augur_m285_pattern_byte(
    uint64_t seed,
    uint64_t byte_index
) {
    uint64_t value = seed + (byte_index + UINT64_C(1)) * UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return static_cast<unsigned char>(value);
}

__global__ void augur_m285_fill_span_kernel(
    unsigned char* destination,
    uint64_t extent_bytes,
    uint64_t pattern_seed
) {
    const uint64_t byte_index =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (byte_index < extent_bytes) {
        destination[byte_index] = augur_m285_pattern_byte(pattern_seed, byte_index);
    }
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m285_fill_span_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m285_fill_span_args_v1* args
) {
    if (args == nullptr ||
        args->abi_version != AUGUR_CUDA_M285_FILL_SPAN_ARGS_VERSION_V1 ||
        args->struct_size != sizeof(augur_cuda_m285_fill_span_args_v1) ||
        args->allocation_base == 0 ||
        args->allocation_extent_bytes == 0 ||
        args->view_extent_bytes == 0 ||
        args->reserved != 0 ||
        args->view_offset_bytes > args->allocation_extent_bytes ||
        args->view_extent_bytes > args->allocation_extent_bytes - args->view_offset_bytes ||
        args->allocation_base > UINT64_MAX - args->allocation_extent_bytes ||
        args->view_extent_bytes > UINT64_MAX - UINT64_C(255)) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_validate_stream_device(ordinal, stream);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    unsigned char* allocation = reinterpret_cast<unsigned char*>(
        static_cast<uintptr_t>(args->allocation_base)
    );
    const uint64_t blocks = (args->view_extent_bytes + UINT64_C(255)) / UINT64_C(256);
    if (blocks == 0 || blocks > UINT32_MAX) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_m285_fill_span_kernel<<<static_cast<uint32_t>(blocks), 256, 0, stream->stream>>>(
        allocation + args->view_offset_bytes,
        args->view_extent_bytes,
        args->pattern_seed
    );
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

static_assert(
    sizeof(augur_cuda_m292_hbm_args_v1) == 88,
    "augur_cuda_m292_hbm_args_v1 ABI size changed"
);

__global__ void augur_m292_hbm_read_only_kernel_v1(
    const uint4* source,
    uint4* block_checksums,
    uint64_t elements,
    uint32_t repeats
) {
    uint4 checksum = make_uint4(0, 0, 0, 0);
    const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        for (uint64_t index = start; index < elements; index += stride) {
            const uint4 value = source[index];
            checksum.x ^= value.x + repeat;
            checksum.y ^= value.y;
            checksum.z ^= value.z;
            checksum.w ^= value.w;
        }
    }
    for (uint32_t delta = 16; delta != 0; delta >>= 1) {
        checksum.x ^= __shfl_down_sync(0xffffffffu, checksum.x, delta);
        checksum.y ^= __shfl_down_sync(0xffffffffu, checksum.y, delta);
        checksum.z ^= __shfl_down_sync(0xffffffffu, checksum.z, delta);
        checksum.w ^= __shfl_down_sync(0xffffffffu, checksum.w, delta);
    }
    __shared__ uint4 warp_checksums[8];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_checksums[warp] = checksum;
    }
    __syncthreads();
    if (warp == 0) {
        checksum = lane < (blockDim.x >> 5) ? warp_checksums[lane] : make_uint4(0, 0, 0, 0);
        for (uint32_t delta = 16; delta != 0; delta >>= 1) {
            checksum.x ^= __shfl_down_sync(0xffffffffu, checksum.x, delta);
            checksum.y ^= __shfl_down_sync(0xffffffffu, checksum.y, delta);
            checksum.z ^= __shfl_down_sync(0xffffffffu, checksum.z, delta);
            checksum.w ^= __shfl_down_sync(0xffffffffu, checksum.w, delta);
        }
        if (lane == 0) {
            block_checksums[blockIdx.x] = checksum;
        }
    }
}

__global__ void augur_m292_hbm_read_write_kernel_v1(
    const uint4* source,
    uint4* destination,
    uint64_t elements,
    uint32_t repeats
) {
    const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        for (uint64_t index = start; index < elements; index += stride) {
            destination[index] = source[index];
        }
    }
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m292_hbm_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_m292_hbm_args_v1* args
) {
    if (args == nullptr ||
        args->abi_version != AUGUR_CUDA_M292_HBM_ARGS_VERSION_V1 ||
        args->struct_size != sizeof(augur_cuda_m292_hbm_args_v1) ||
        (args->operation != AUGUR_CUDA_M292_HBM_READ_ONLY &&
         args->operation != AUGUR_CUDA_M292_HBM_READ_WRITE) ||
        args->blocks == 0 || args->repeats == 0 || args->reserved != 0 ||
        args->source_base == 0 || args->destination_base == 0 ||
        args->source_view_extent_bytes == 0 ||
        (args->source_view_extent_bytes & UINT64_C(15)) != 0 ||
        (args->source_view_offset_bytes & UINT64_C(15)) != 0 ||
        (args->destination_view_offset_bytes & UINT64_C(15)) != 0 ||
        args->source_view_offset_bytes > args->source_allocation_extent_bytes ||
        args->source_view_extent_bytes >
            args->source_allocation_extent_bytes - args->source_view_offset_bytes ||
        args->destination_view_offset_bytes > args->destination_allocation_extent_bytes ||
        args->source_base > UINT64_MAX - args->source_allocation_extent_bytes ||
        args->destination_base > UINT64_MAX - args->destination_allocation_extent_bytes ||
        (args->source_base & UINT64_C(15)) != 0 ||
        (args->destination_base & UINT64_C(15)) != 0) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t required_destination = args->operation == AUGUR_CUDA_M292_HBM_READ_ONLY
        ? static_cast<uint64_t>(args->blocks) * sizeof(uint4)
        : args->source_view_extent_bytes;
    if (args->destination_view_extent_bytes != required_destination ||
        required_destination >
            args->destination_allocation_extent_bytes - args->destination_view_offset_bytes) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    augur_cuda_result_v1 selected = augur_validate_stream_device(ordinal, stream);
    if (selected.status != AUGUR_CUDA_STATUS_OK) {
        return selected;
    }
    const uint4* source = reinterpret_cast<const uint4*>(
        static_cast<uintptr_t>(args->source_base + args->source_view_offset_bytes));
    uint4* destination = reinterpret_cast<uint4*>(
        static_cast<uintptr_t>(args->destination_base + args->destination_view_offset_bytes));
    const uint64_t elements = args->source_view_extent_bytes / sizeof(uint4);
    constexpr uint32_t threads = 256;
    if (args->operation == AUGUR_CUDA_M292_HBM_READ_ONLY) {
        augur_m292_hbm_read_only_kernel_v1<<<args->blocks, threads, 0, stream->stream>>>(
            source, destination, elements, args->repeats);
    } else {
        augur_m292_hbm_read_write_kernel_v1<<<args->blocks, threads, 0, stream->stream>>>(
            source, destination, elements, args->repeats);
    }
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_m292_l2_cache_bytes_v1(
    int32_t ordinal,
    uint64_t* out_l2_cache_bytes
) {
    if (ordinal < 0 || out_l2_cache_bytes == nullptr) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    *out_l2_cache_bytes = 0;
    int l2_cache_bytes = 0;
    const cudaError_t error = cudaDeviceGetAttribute(
        &l2_cache_bytes,
        cudaDevAttrL2CacheSize,
        ordinal
    );
    if (error != cudaSuccess) {
        return augur_runtime_error(error);
    }
    if (l2_cache_bytes <= 0) {
        return augur_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, cudaSuccess);
    }
    *out_l2_cache_bytes = static_cast<uint64_t>(l2_cache_bytes);
    return augur_ok();
}

/* ------------------------------------------------------------------------- */
/* Qwen35 layer-3 grouped-query full-attention component.                  */

static_assert(
    sizeof(augur_cuda_qwen35_full_attention_args_v1) == 168,
    "augur_cuda_qwen35_full_attention_args_v1 ABI size changed"
);
#if defined(AUGUR_CUDA_QWEN35_GROUPED_GQA_ONLINE_SOFTMAX)
static_assert(
    sizeof(augur_cuda_qwen35_full_attention_args_v2) == 184,
    "augur_cuda_qwen35_full_attention_args_v2 ABI size changed"
);
#endif
#if defined(AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION)
static_assert(
    sizeof(augur_cuda_qwen35_kv8_full_attention_args_v3) == 184,
    "augur_cuda_qwen35_kv8_full_attention_args_v3 ABI size changed"
);
#endif

namespace {

constexpr uint32_t AUGUR_QWEN35_FULL_QUERY_HEADS =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1;
constexpr uint32_t AUGUR_QWEN35_FULL_KV_HEADS =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1;
constexpr uint32_t AUGUR_QWEN35_FULL_HEAD_DIM =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1;
constexpr uint32_t AUGUR_QWEN35_FULL_QUERY_ELEMENTS =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1;
constexpr uint32_t AUGUR_QWEN35_FULL_KV_ELEMENTS =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1;
constexpr uint32_t AUGUR_QWEN35_FULL_MAX_CAPACITY =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_MAX_CAPACITY_V1;

static bool augur_qwen35_full_range_ok(uint64_t address, uint64_t length) {
    return address != 0 && length != 0 && address <= UINT64_MAX - length;
}

static bool augur_qwen35_full_overlap(
    uint64_t left,
    uint64_t left_len,
    uint64_t right,
    uint64_t right_len
) {
    return left < right + right_len && right < left + left_len;
}

#if defined(AUGUR_CUDA_QWEN35_FUSED_HEAD_NORM_ROPE)
static_assert(
    sizeof(augur_cuda_qwen35_fused_head_norm_rope_args_v1) == 192,
    "augur_cuda_qwen35_fused_head_norm_rope_args_v1 ABI size changed"
);

__global__ void augur_qwen35_fused_head_norm_rope_kernel_v1(
    const float* q_input,
    const float* q_weight,
    float* q_normed,
    float* q_output,
    const float* k_input,
    const float* k_weight,
    float* k_normed,
    float* k_output,
    const float* rope_cos,
    const float* rope_sin,
    uint32_t query_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t rotary_dim,
    float epsilon
) {
    const uint32_t block_head = blockIdx.x;
    const bool is_query = block_head < query_heads;
    const uint32_t head = is_query ? block_head : block_head - query_heads;
    if ((!is_query && head >= kv_heads) || threadIdx.x >= head_dim) return;

    const float* input = is_query ? q_input : k_input;
    const float* weight = is_query ? q_weight : k_weight;
    float* normed = is_query ? q_normed : k_normed;
    float* output = is_query ? q_output : k_output;
    const uint32_t index = head * head_dim + threadIdx.x;

    __shared__ float partial[AUGUR_QWEN35_FULL_HEAD_DIM];
    const float input_value = input[index];
    partial[threadIdx.x] = input_value * input_value;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / static_cast<float>(head_dim) + epsilon);
    normed[index] = input_value * scale * weight[index];
    __syncthreads();

    if (threadIdx.x < rotary_dim) {
#if defined(AUGUR_CUDA_QWEN35_DIAGNOSTIC_SPLIT_ROPE)
        const uint32_t half = rotary_dim / 2;
        const uint32_t pair = threadIdx.x % half;
        const bool upper = threadIdx.x >= half;
        const float cosine = rope_cos[pair];
        const float sine = rope_sin[pair];
        const float value = normed[index];
        const float mate = normed[upper ? index - half : index + half];
        output[index] = upper
            ? mate * sine + value * cosine
            : value * cosine - mate * sine;
#else
        const uint32_t pair = threadIdx.x / 2;
        const float cosine = rope_cos[pair];
        const float sine = rope_sin[pair];
        const float value = normed[index];
        const float mate = normed[index ^ 1];
        output[index] = (threadIdx.x & 1)
            ? mate * sine + value * cosine
            : value * cosine - mate * sine;
#endif
    } else {
        output[index] = normed[index];
    }
}

static_assert(sizeof(augur_cuda_qwen35_batched_head_norm_rope_args_v1) == 200, "batch head RoPE ABI");

// Keep each head's 256-lane reduction and scalar RoPE expression identical to
// the reference above. Rows add only independent CTAs; no row reduction occurs.
__global__ void augur_qwen35_batched_head_norm_rope_kernel_v1(
    augur_cuda_qwen35_batched_head_norm_rope_args_v1 a, float epsilon
) {
    const uint32_t row = blockIdx.y, lane = threadIdx.x;
    const bool query = blockIdx.x < AUGUR_QWEN35_FULL_QUERY_HEADS;
    const uint32_t head = query ? blockIdx.x : blockIdx.x - AUGUR_QWEN35_FULL_QUERY_HEADS;
    const uint32_t index = head * AUGUR_QWEN35_FULL_HEAD_DIM + lane;
    const uint32_t width = query ? AUGUR_QWEN35_FULL_QUERY_ELEMENTS : AUGUR_QWEN35_FULL_KV_ELEMENTS;
    const uint32_t stride = query ? 2U * AUGUR_QWEN35_FULL_QUERY_ELEMENTS : AUGUR_QWEN35_FULL_KV_ELEMENTS;
    const float* input = reinterpret_cast<const float*>(static_cast<uintptr_t>(a.addresses[query ? 0 : 3]));
    const float* weight = reinterpret_cast<const float*>(static_cast<uintptr_t>(a.addresses[query ? 1 : 4]));
    float* output = reinterpret_cast<float*>(static_cast<uintptr_t>(a.addresses[query ? 2 : 5]));
    const float* cosine = reinterpret_cast<const float*>(static_cast<uintptr_t>(a.addresses[6]));
    const float* sine = reinterpret_cast<const float*>(static_cast<uintptr_t>(a.addresses[7]));
    __shared__ float partial[AUGUR_QWEN35_FULL_HEAD_DIM];
    __shared__ float normalized[AUGUR_QWEN35_FULL_HEAD_DIM];
    const float value = input[row * stride + index];
    partial[lane] = value * value;
    __syncthreads();
    for (uint32_t step = blockDim.x / 2; step != 0; step /= 2) {
        if (lane < step) partial[lane] += partial[lane + step];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / static_cast<float>(AUGUR_QWEN35_FULL_HEAD_DIM) + epsilon);
    normalized[lane] = value * scale * weight[index];
    __syncthreads();
    float rotated = normalized[lane];
    if (lane < AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1) {
        const uint32_t half = AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 / 2;
#if defined(AUGUR_CUDA_QWEN35_DIAGNOSTIC_SPLIT_ROPE)
        const uint32_t pair = lane % half;
        const bool upper = lane >= half;
        const float mate = normalized[upper ? lane - half : lane + half];
        const float c = cosine[row * half + pair], s = sine[row * half + pair];
        rotated = upper ? mate * s + rotated * c : rotated * c - mate * s;
#else
        const uint32_t pair = lane / 2;
        const float mate = normalized[lane ^ 1];
        const float c = cosine[row * half + pair], s = sine[row * half + pair];
        rotated = (lane & 1) ? mate * s + rotated * c : rotated * c - mate * s;
#endif
    }
    output[row * width + index] = rotated;
    if (query) {
        float* gate = reinterpret_cast<float*>(static_cast<uintptr_t>(a.addresses[8]));
        gate[row * width + index] = input[row * stride + width + index];
    } else {
        const float* source = reinterpret_cast<const float*>(static_cast<uintptr_t>(a.addresses[9]));
        float* dest = reinterpret_cast<float*>(static_cast<uintptr_t>(a.addresses[10]));
        dest[row * width + index] = source[row * width + index];
    }
}

static augur_cuda_result_v1 augur_qwen35_batched_head_norm_rope_submit_impl(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_batched_head_norm_rope_args_v1* a
) {
    float epsilon = 0.0f;
    if (a != nullptr) std::memcpy(&epsilon, &a->epsilon_bits, sizeof(epsilon));
    if (a == nullptr || a->abi_version != 1 || a->struct_size != sizeof(*a)
        || a->rows == 0 || a->rows > 512 || a->reserved0 || a->reserved1
        || !isfinite(epsilon) || epsilon <= 0.0f) return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    const uint64_t q = uint64_t(AUGUR_QWEN35_FULL_QUERY_ELEMENTS) * sizeof(float);
    const uint64_t k = uint64_t(AUGUR_QWEN35_FULL_KV_ELEMENTS) * sizeof(float);
    const uint64_t rope = uint64_t(a->rows) * (AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 / 2) * sizeof(float);
    const uint64_t expected[11] = {2*a->rows*q, q, a->rows*q, a->rows*k, k, a->rows*k,
        rope, rope, a->rows*q, a->rows*k, a->rows*k};
    for (uint32_t i = 0; i < 11; ++i) {
        if (a->lengths[i] != expected[i] || a->addresses[i] % alignof(float)
            || !augur_qwen35_full_range_ok(a->addresses[i], a->lengths[i]))
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        for (uint32_t j = i + 1; j < 11; ++j) {
            if (augur_qwen35_full_overlap(a->addresses[i], a->lengths[i], a->addresses[j], a->lengths[j]))
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
    }
    const augur_cuda_result_v1 valid = augur_validate_stream_device(ordinal, stream);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    augur_qwen35_batched_head_norm_rope_kernel_v1<<<
        dim3(AUGUR_QWEN35_FULL_QUERY_HEADS + AUGUR_QWEN35_FULL_KV_HEADS, a->rows),
        AUGUR_QWEN35_FULL_HEAD_DIM, 0, stream->stream>>>(*a, epsilon);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

static augur_cuda_result_v1 augur_qwen35_fused_head_norm_rope_validate(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_fused_head_norm_rope_args_v1* args
) {
    float epsilon = 0.0f;
    if (args != nullptr) std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    if (stream == nullptr || args == nullptr
        || args->abi_version != AUGUR_CUDA_QWEN35_FUSED_HEAD_NORM_ROPE_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_qwen35_fused_head_norm_rope_args_v1)
        || args->reserved != 0
        || args->query_heads != AUGUR_QWEN35_FULL_QUERY_HEADS
        || args->kv_heads != AUGUR_QWEN35_FULL_KV_HEADS
        || args->head_dim != AUGUR_QWEN35_FULL_HEAD_DIM
        || args->rotary_dim != AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1
        || !isfinite(epsilon) || epsilon <= 0.0f) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }

    const uint64_t query_bytes =
        static_cast<uint64_t>(AUGUR_QWEN35_FULL_QUERY_ELEMENTS) * sizeof(float);
    const uint64_t kv_bytes =
        static_cast<uint64_t>(AUGUR_QWEN35_FULL_KV_ELEMENTS) * sizeof(float);
    const uint64_t rope_bytes =
        static_cast<uint64_t>(args->rotary_dim / 2) * sizeof(float);
    const uint64_t addresses[] = {
        args->q_input_f32, args->q_weight_f32, args->q_normed_f32, args->q_output_f32,
        args->k_input_f32, args->k_weight_f32, args->k_normed_f32, args->k_output_f32,
        args->rope_cos_f32, args->rope_sin_f32,
    };
    const uint64_t lengths[] = {
        args->q_input_len_bytes, args->q_weight_len_bytes, args->q_normed_len_bytes,
        args->q_output_len_bytes, args->k_input_len_bytes, args->k_weight_len_bytes,
        args->k_normed_len_bytes, args->k_output_len_bytes, args->rope_cos_len_bytes,
        args->rope_sin_len_bytes,
    };
    const uint64_t expected[] = {
        query_bytes, query_bytes, query_bytes, query_bytes,
        kv_bytes, kv_bytes, kv_bytes, kv_bytes, rope_bytes, rope_bytes,
    };
    for (uint32_t index = 0; index < 10; ++index) {
        if (lengths[index] != expected[index]
            || !augur_qwen35_full_range_ok(addresses[index], lengths[index])
            || (addresses[index] % alignof(float)) != 0) {
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t other = index + 1; other < 10; ++other) {
            if (augur_qwen35_full_overlap(
                    addresses[index], lengths[index], addresses[other], lengths[other])) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    return augur_validate_stream_device(ordinal, stream);
}

static augur_cuda_result_v1 augur_qwen35_fused_head_norm_rope_submit_impl(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_fused_head_norm_rope_args_v1* args
) {
    const augur_cuda_result_v1 valid =
        augur_qwen35_fused_head_norm_rope_validate(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    float epsilon = 0.0f;
    std::memcpy(&epsilon, &args->epsilon_bits, sizeof(epsilon));
    augur_qwen35_fused_head_norm_rope_kernel_v1<<<
        AUGUR_QWEN35_FULL_QUERY_HEADS + AUGUR_QWEN35_FULL_KV_HEADS,
        AUGUR_QWEN35_FULL_HEAD_DIM,
        0,
        stream->stream
    >>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->q_input_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->q_weight_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->q_normed_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->q_output_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->k_input_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->k_weight_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->k_normed_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->k_output_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->rope_cos_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->rope_sin_f32)),
        args->query_heads,
        args->kv_heads,
        args->head_dim,
        args->rotary_dim,
        epsilon
    );
    const cudaError_t launch_error = cudaGetLastError();
    return launch_error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
}
#endif

__device__ __forceinline__ float augur_qwen35_full_sigmoid(float value) {
    if (value >= 0.0f) {
        return 1.0f / (1.0f + expf(-value));
    }
    const float exponent = expf(value);
    return exponent / (1.0f + exponent);
}

__global__ void augur_qwen35_full_store_kv_kernel(
    const float* key,
    const float* value,
    float* key_cache,
    float* value_cache,
    uint32_t position
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= AUGUR_QWEN35_FULL_KV_ELEMENTS) return;
    const uint64_t destination = static_cast<uint64_t>(position) * AUGUR_QWEN35_FULL_KV_ELEMENTS + index;
    key_cache[destination] = key[index];
    value_cache[destination] = value[index];
}

__global__ void augur_qwen35_full_attention_kernel(
    const float* query,
    const float* gate,
    const float* key_cache,
    const float* value_cache,
    float* attention,
    float* gated,
    uint32_t visible_positions
) {
    const uint32_t query_head = blockIdx.x;
    const uint32_t component = threadIdx.x;
    if (query_head >= AUGUR_QWEN35_FULL_QUERY_HEADS || component >= AUGUR_QWEN35_FULL_HEAD_DIM) return;

    /* One score/weight row per query head.  The fixed 4096 bound is part of
     * the ABI and keeps the component free of runtime allocation. */
    __shared__ float weights[AUGUR_QWEN35_FULL_MAX_CAPACITY];
#if defined(AUGUR_CUDA_QWEN35_WARP_FULL_ATTENTION)
    const uint32_t lane = component & 31u;
    const uint32_t warp = component >> 5u;
    const uint32_t kv_head = query_head / (AUGUR_QWEN35_FULL_QUERY_HEADS / AUGUR_QWEN35_FULL_KV_HEADS);
    const uint32_t query_base = query_head * AUGUR_QWEN35_FULL_HEAD_DIM;
    for (uint32_t position = warp; position < visible_positions; position += 8u) {
        const uint64_t key_base = static_cast<uint64_t>(position) * AUGUR_QWEN35_FULL_KV_ELEMENTS
            + static_cast<uint64_t>(kv_head) * AUGUR_QWEN35_FULL_HEAD_DIM;
        double score = 0.0;
        for (uint32_t dimension = lane; dimension < AUGUR_QWEN35_FULL_HEAD_DIM; dimension += 32u) {
            score += static_cast<double>(query[query_base + dimension])
                * static_cast<double>(key_cache[key_base + dimension]);
        }
        for (uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
            score += __shfl_down_sync(0xffffffffu, score, offset);
        }
        if (lane == 0u) {
            const float scaled_score = static_cast<float>(score)
                * rsqrtf(static_cast<float>(AUGUR_QWEN35_FULL_HEAD_DIM));
            weights[position] = scaled_score;
        }
    }
    __syncthreads();
#if defined(AUGUR_CUDA_QWEN35_FULL_ATTENTION_PARALLEL_EXP)
    __shared__ float softmax_maximum;
    __shared__ float softmax_inverse;
    if (component == 0) {
        float maximum = -INFINITY;
        for (uint32_t position = 0; position < visible_positions; ++position) {
            maximum = fmaxf(maximum, weights[position]);
        }
        softmax_maximum = maximum;
    }
    __syncthreads();
    for (uint32_t position = component; position < visible_positions;
         position += AUGUR_QWEN35_FULL_HEAD_DIM) {
        weights[position] = expf(weights[position] - softmax_maximum);
    }
    __syncthreads();
    if (component == 0) {
        float denominator = 0.0f;
        for (uint32_t position = 0; position < visible_positions; ++position) {
            denominator += weights[position];
        }
        softmax_inverse = 1.0f / denominator;
    }
    __syncthreads();
    for (uint32_t position = component; position < visible_positions;
         position += AUGUR_QWEN35_FULL_HEAD_DIM) {
        weights[position] *= softmax_inverse;
    }
#else
    if (component == 0) {
        float maximum = -INFINITY;
        for (uint32_t position = 0; position < visible_positions; ++position) {
            maximum = fmaxf(maximum, weights[position]);
        }
        float denominator = 0.0f;
        for (uint32_t position = 0; position < visible_positions; ++position) {
            const float value = expf(weights[position] - maximum);
            weights[position] = value;
            denominator += value;
        }
        const float inverse = 1.0f / denominator;
        for (uint32_t position = 0; position < visible_positions; ++position) {
            weights[position] *= inverse;
        }
    }
#endif
#else
    if (component == 0) {
        const uint32_t kv_head = query_head / (AUGUR_QWEN35_FULL_QUERY_HEADS / AUGUR_QWEN35_FULL_KV_HEADS);
        const uint32_t query_base = query_head * AUGUR_QWEN35_FULL_HEAD_DIM;
        float maximum = -INFINITY;
        for (uint32_t position = 0; position < visible_positions; ++position) {
            const uint64_t key_base = static_cast<uint64_t>(position) * AUGUR_QWEN35_FULL_KV_ELEMENTS
                + static_cast<uint64_t>(kv_head) * AUGUR_QWEN35_FULL_HEAD_DIM;
            float score = 0.0f;
            for (uint32_t dimension = 0; dimension < AUGUR_QWEN35_FULL_HEAD_DIM; ++dimension) {
                score += query[query_base + dimension] * key_cache[key_base + dimension];
            }
            score *= rsqrtf(static_cast<float>(AUGUR_QWEN35_FULL_HEAD_DIM));
            weights[position] = score;
            maximum = fmaxf(maximum, score);
        }
        float denominator = 0.0f;
        for (uint32_t position = 0; position < visible_positions; ++position) {
            const float value = expf(weights[position] - maximum);
            weights[position] = value;
            denominator += value;
        }
        const float inverse = 1.0f / denominator;
        for (uint32_t position = 0; position < visible_positions; ++position) {
            weights[position] *= inverse;
        }
    }
#endif
    __syncthreads();

#if !defined(AUGUR_CUDA_QWEN35_WARP_FULL_ATTENTION)
    const uint32_t kv_head = query_head / (AUGUR_QWEN35_FULL_QUERY_HEADS / AUGUR_QWEN35_FULL_KV_HEADS);
    const uint32_t query_base = query_head * AUGUR_QWEN35_FULL_HEAD_DIM;
#endif
    float result = 0.0f;
    for (uint32_t position = 0; position < visible_positions; ++position) {
        const uint64_t value_base = static_cast<uint64_t>(position) * AUGUR_QWEN35_FULL_KV_ELEMENTS
            + static_cast<uint64_t>(kv_head) * AUGUR_QWEN35_FULL_HEAD_DIM;
        result += weights[position] * value_cache[value_base + component];
    }
    attention[query_base + component] = result;
    gated[query_base + component] = result * augur_qwen35_full_sigmoid(gate[query_base + component]);
}

static augur_cuda_result_v1 augur_qwen35_full_validate(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_full_attention_args_v1* args
) {
    if (stream == nullptr || args == nullptr || args->abi_version != AUGUR_CUDA_QWEN35_FULL_ATTENTION_ARGS_VERSION_V1
        || args->struct_size != sizeof(augur_cuda_qwen35_full_attention_args_v1)
        || args->reserved != 0
        || args->capacity == 0 || args->capacity > AUGUR_QWEN35_FULL_MAX_CAPACITY
        || args->position >= args->capacity
        || args->query_heads != AUGUR_QWEN35_FULL_QUERY_HEADS
        || args->kv_heads != AUGUR_QWEN35_FULL_KV_HEADS
        || args->head_dim != AUGUR_QWEN35_FULL_HEAD_DIM
        || args->rotary_dim != AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t query_bytes = static_cast<uint64_t>(AUGUR_QWEN35_FULL_QUERY_ELEMENTS) * sizeof(float);
    const uint64_t kv_bytes = static_cast<uint64_t>(AUGUR_QWEN35_FULL_KV_ELEMENTS) * sizeof(float);
    const uint64_t cache_bytes = static_cast<uint64_t>(args->capacity) * kv_bytes;
    const uint64_t addresses[] = {
        args->query_f32, args->gate_f32, args->key_f32, args->value_f32,
        args->key_cache_f32, args->value_cache_f32, args->attention_f32, args->gated_f32,
    };
    const uint64_t lengths[] = {
        args->query_len_bytes, args->gate_len_bytes, args->key_len_bytes, args->value_len_bytes,
        args->key_cache_len_bytes, args->value_cache_len_bytes, args->attention_len_bytes,
        args->gated_len_bytes,
    };
    const uint64_t expected[] = {
        query_bytes, query_bytes, kv_bytes, kv_bytes, cache_bytes, cache_bytes, query_bytes, query_bytes,
    };
    for (uint32_t index = 0; index < 8; ++index) {
        if (lengths[index] != expected[index]
            || !augur_qwen35_full_range_ok(addresses[index], lengths[index])
            || (addresses[index] % alignof(float)) != 0) {
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t other = index + 1; other < 8; ++other) {
            if (augur_qwen35_full_overlap(addresses[index], lengths[index], addresses[other], lengths[other])) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    return augur_validate_stream_device(ordinal, stream);
}

static augur_cuda_result_v1 augur_qwen35_full_submit_impl(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_full_attention_args_v1* args
) {
    const augur_cuda_result_v1 valid = augur_qwen35_full_validate(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    float* key_cache = reinterpret_cast<float*>(static_cast<uintptr_t>(args->key_cache_f32));
    float* value_cache = reinterpret_cast<float*>(static_cast<uintptr_t>(args->value_cache_f32));
    augur_qwen35_full_store_kv_kernel<<<(AUGUR_QWEN35_FULL_KV_ELEMENTS + 255) / 256, 256, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->key_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->value_f32)),
        key_cache,
        value_cache,
        args->position
    );
    augur_qwen35_full_attention_kernel<<<AUGUR_QWEN35_FULL_QUERY_HEADS, AUGUR_QWEN35_FULL_HEAD_DIM, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate_f32)),
        key_cache,
        value_cache,
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->attention_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->gated_f32)),
        args->position + 1
    );
    const cudaError_t launch_error = cudaGetLastError();
    return launch_error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
}

#if defined(AUGUR_CUDA_QWEN35_GROUPED_GQA_ONLINE_SOFTMAX)
constexpr uint32_t AUGUR_QWEN35_FULL_SPLITS_V2 =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_SPLITS_V2;
constexpr uint32_t AUGUR_QWEN35_FULL_STAGE1_THREADS_V2 =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_STAGE1_THREADS_V2;
constexpr uint32_t AUGUR_QWEN35_FULL_TILE_POSITIONS_V2 =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_TILE_POSITIONS_V2;
constexpr uint32_t AUGUR_QWEN35_FULL_MERGE_SEGMENTS_V2 =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_MERGE_SEGMENTS_V2;
constexpr uint32_t AUGUR_QWEN35_FULL_MERGE_THREADS_V2 =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_MERGE_THREADS_V2;
constexpr uint32_t AUGUR_QWEN35_FULL_PARTIAL_STRIDE_V2 =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_PARTIAL_STRIDE_V2;
constexpr uint32_t AUGUR_QWEN35_FULL_PARTIAL_ELEMENTS_V2 =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2;
constexpr uint32_t AUGUR_QWEN35_FULL_FUSED_Q8_BYTES_V2 =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2;
constexpr uint32_t AUGUR_QWEN35_FULL_WORKSPACE_ELEMENTS_V2 =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2;
constexpr uint32_t AUGUR_QWEN35_FULL_QUERIES_PER_KV =
    AUGUR_QWEN35_FULL_QUERY_HEADS / AUGUR_QWEN35_FULL_KV_HEADS;

static_assert(AUGUR_QWEN35_FULL_QUERIES_PER_KV == 6,
    "Qwen35 grouped-GQA ratio changed");
static_assert(AUGUR_QWEN35_FULL_STAGE1_THREADS_V2 == AUGUR_QWEN35_FULL_HEAD_DIM,
    "Qwen35 grouped-GQA stage-1 thread geometry changed");
static_assert(
    AUGUR_QWEN35_FULL_MERGE_THREADS_V2
        == AUGUR_QWEN35_FULL_HEAD_DIM / AUGUR_QWEN35_FULL_MERGE_SEGMENTS_V2,
    "Qwen35 grouped-GQA merge thread/segment geometry changed");
static_assert(AUGUR_QWEN35_FULL_PARTIAL_STRIDE_V2 == AUGUR_QWEN35_FULL_HEAD_DIM + 2,
    "Qwen35 grouped-GQA partial stride changed");
static_assert(AUGUR_QWEN35_FULL_MERGE_THREADS_V2 == 128,
    "Qwen35 grouped-GQA merge segment geometry changed");
static_assert(
    AUGUR_QWEN35_FULL_PARTIAL_ELEMENTS_V2
        == AUGUR_QWEN35_FULL_QUERY_HEADS * AUGUR_QWEN35_FULL_SPLITS_V2
            * AUGUR_QWEN35_FULL_PARTIAL_STRIDE_V2,
    "Qwen35 grouped-GQA partial geometry changed"
);
static_assert(
    AUGUR_QWEN35_FULL_FUSED_Q8_BYTES_V2
        == (AUGUR_QWEN35_FULL_QUERY_ELEMENTS / augur_q8_1_block_elements_v1)
            * augur_q8_1_block_bytes_v1,
    "Qwen35 grouped-GQA fused Q8_1 geometry changed"
);
static_assert(
    AUGUR_QWEN35_FULL_WORKSPACE_ELEMENTS_V2 * sizeof(float)
        == AUGUR_QWEN35_FULL_PARTIAL_ELEMENTS_V2 * sizeof(float)
            + AUGUR_QWEN35_FULL_FUSED_Q8_BYTES_V2,
    "Qwen35 grouped-GQA workspace geometry changed"
);

__global__ void augur_qwen35_grouped_gqa_stage1_v2(
    const float* query,
    const float* key,
    const float* value,
    float* key_cache,
    float* value_cache,
    float* workspace,
    uint32_t position
) {
    const uint32_t visible_positions = position + 1U;
    const uint32_t split = blockIdx.x % AUGUR_QWEN35_FULL_SPLITS_V2;
    const uint32_t kv_head = blockIdx.x / AUGUR_QWEN35_FULL_SPLITS_V2;
    const uint32_t thread = threadIdx.x;
    const uint32_t warp = thread >> 5U;
    const uint32_t lane = thread & 31U;
    if (kv_head >= AUGUR_QWEN35_FULL_KV_HEADS) return;

    const uint32_t split_begin = static_cast<uint32_t>(
        (static_cast<uint64_t>(visible_positions) * split) / AUGUR_QWEN35_FULL_SPLITS_V2);
    const uint32_t split_end = static_cast<uint32_t>(
        (static_cast<uint64_t>(visible_positions) * (split + 1U))
            / AUGUR_QWEN35_FULL_SPLITS_V2);
    const bool active_query = warp < AUGUR_QWEN35_FULL_QUERIES_PER_KV;
    const uint32_t query_head = kv_head * AUGUR_QWEN35_FULL_QUERIES_PER_KV + warp;

    __shared__ float key_tile[
        AUGUR_QWEN35_FULL_TILE_POSITIONS_V2 * AUGUR_QWEN35_FULL_HEAD_DIM];
    __shared__ float value_tile[
        AUGUR_QWEN35_FULL_TILE_POSITIONS_V2 * AUGUR_QWEN35_FULL_HEAD_DIM];

    float query_fragment[8] = {};
    float output_fragment[8] = {};
    if (active_query) {
        const uint32_t query_base = query_head * AUGUR_QWEN35_FULL_HEAD_DIM;
        #pragma unroll
        for (uint32_t item = 0; item < 8; ++item) {
            query_fragment[item] = query[query_base + lane + item * 32U];
        }
    }
    float running_maximum = -INFINITY;
    float running_sum = 0.0f;

    for (uint32_t tile_begin = split_begin; tile_begin < split_end;
         tile_begin += AUGUR_QWEN35_FULL_TILE_POSITIONS_V2) {
        const uint32_t remaining = split_end - tile_begin;
        const uint32_t tile_positions = remaining < AUGUR_QWEN35_FULL_TILE_POSITIONS_V2
            ? remaining
            : AUGUR_QWEN35_FULL_TILE_POSITIONS_V2;
        #pragma unroll
        for (uint32_t tile_position = 0;
             tile_position < AUGUR_QWEN35_FULL_TILE_POSITIONS_V2;
             ++tile_position) {
            if (tile_position < tile_positions) {
                const uint64_t cache_base =
                    static_cast<uint64_t>(tile_begin + tile_position)
                        * AUGUR_QWEN35_FULL_KV_ELEMENTS
                    + static_cast<uint64_t>(kv_head) * AUGUR_QWEN35_FULL_HEAD_DIM;
                const uint32_t shared_index =
                    tile_position * AUGUR_QWEN35_FULL_HEAD_DIM + thread;
                const uint32_t absolute_position = tile_begin + tile_position;
                if (absolute_position == position) {
                    const uint32_t source_index =
                        kv_head * AUGUR_QWEN35_FULL_HEAD_DIM + thread;
                    const float key_value = key[source_index];
                    const float value_value = value[source_index];
                    key_cache[cache_base + thread] = key_value;
                    value_cache[cache_base + thread] = value_value;
                    key_tile[shared_index] = key_value;
                    value_tile[shared_index] = value_value;
                } else {
                    key_tile[shared_index] = key_cache[cache_base + thread];
                    value_tile[shared_index] = value_cache[cache_base + thread];
                }
            }
        }
        __syncthreads();

        if (active_query) {
            for (uint32_t tile_position = 0; tile_position < tile_positions; ++tile_position) {
                const uint32_t shared_base = tile_position * AUGUR_QWEN35_FULL_HEAD_DIM;
                double score = 0.0;
                #pragma unroll
                for (uint32_t item = 0; item < 8; ++item) {
                    score += static_cast<double>(query_fragment[item])
                        * static_cast<double>(key_tile[shared_base + lane + item * 32U]);
                }
                #pragma unroll
                for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
                    score += __shfl_down_sync(0xffffffffU, score, offset);
                }
                score = __shfl_sync(0xffffffffU, score, 0);
                const float scaled_score = static_cast<float>(score)
                    * rsqrtf(static_cast<float>(AUGUR_QWEN35_FULL_HEAD_DIM));
                const float next_maximum = fmaxf(running_maximum, scaled_score);
                const float previous_scale = running_sum == 0.0f
                    ? 0.0f
                    : expf(running_maximum - next_maximum);
                const float current_scale = expf(scaled_score - next_maximum);
                running_sum = running_sum * previous_scale + current_scale;
                #pragma unroll
                for (uint32_t item = 0; item < 8; ++item) {
                    const uint32_t component = lane + item * 32U;
                    output_fragment[item] = output_fragment[item] * previous_scale
                        + current_scale * value_tile[shared_base + component];
                }
                running_maximum = next_maximum;
            }
        }
        __syncthreads();
    }

    if (active_query) {
        const uint64_t partial_base =
            (static_cast<uint64_t>(query_head) * AUGUR_QWEN35_FULL_SPLITS_V2 + split)
                * AUGUR_QWEN35_FULL_PARTIAL_STRIDE_V2;
        if (lane == 0U) {
            workspace[partial_base] = running_maximum;
            workspace[partial_base + 1U] = running_sum;
        }
        #pragma unroll
        for (uint32_t item = 0; item < 8; ++item) {
            workspace[partial_base + 2U + lane + item * 32U] = output_fragment[item];
        }
    }
}

__global__ void augur_qwen35_grouped_gqa_merge_v2(
    const float* gate,
    float* workspace,
    float* attention,
    float* gated
) {
    const uint32_t query_head = blockIdx.x / AUGUR_QWEN35_FULL_MERGE_SEGMENTS_V2;
    const uint32_t segment = blockIdx.x % AUGUR_QWEN35_FULL_MERGE_SEGMENTS_V2;
    const uint32_t component =
        segment * AUGUR_QWEN35_FULL_MERGE_THREADS_V2 + threadIdx.x;
    if (query_head >= AUGUR_QWEN35_FULL_QUERY_HEADS
        || component >= AUGUR_QWEN35_FULL_HEAD_DIM) return;

    __shared__ float split_scales[AUGUR_QWEN35_FULL_SPLITS_V2];
    __shared__ float denominator_inverse;
    if (threadIdx.x == 0U) {
        float maximum = -INFINITY;
        #pragma unroll
        for (uint32_t split = 0; split < AUGUR_QWEN35_FULL_SPLITS_V2; ++split) {
            const uint64_t partial_base =
                (static_cast<uint64_t>(query_head) * AUGUR_QWEN35_FULL_SPLITS_V2 + split)
                    * AUGUR_QWEN35_FULL_PARTIAL_STRIDE_V2;
            if (workspace[partial_base + 1U] > 0.0f) {
                maximum = fmaxf(maximum, workspace[partial_base]);
            }
        }
        float denominator = 0.0f;
        #pragma unroll
        for (uint32_t split = 0; split < AUGUR_QWEN35_FULL_SPLITS_V2; ++split) {
            const uint64_t partial_base =
                (static_cast<uint64_t>(query_head) * AUGUR_QWEN35_FULL_SPLITS_V2 + split)
                    * AUGUR_QWEN35_FULL_PARTIAL_STRIDE_V2;
            const float partial_sum = workspace[partial_base + 1U];
            const float scale = partial_sum > 0.0f
                ? expf(workspace[partial_base] - maximum)
                : 0.0f;
            split_scales[split] = scale;
            denominator += partial_sum * scale;
        }
        denominator_inverse = 1.0f / denominator;
    }
    __syncthreads();

    float result = 0.0f;
    #pragma unroll
    for (uint32_t split = 0; split < AUGUR_QWEN35_FULL_SPLITS_V2; ++split) {
        const uint64_t partial_base =
            (static_cast<uint64_t>(query_head) * AUGUR_QWEN35_FULL_SPLITS_V2 + split)
                * AUGUR_QWEN35_FULL_PARTIAL_STRIDE_V2;
        result += workspace[partial_base + 2U + component] * split_scales[split];
    }
    result *= denominator_inverse;
    const uint32_t output_index = query_head * AUGUR_QWEN35_FULL_HEAD_DIM + component;
    const float gated_result = result * augur_qwen35_full_sigmoid(gate[output_index]);
    attention[output_index] = result;
    gated[output_index] = gated_result;

    const uint32_t lane = component & 31U;
    const uint32_t warp = component >> 5U;
    float maximum = fabsf(gated_result);
    float sum = gated_result;
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    maximum = __shfl_sync(0xffffffffU, maximum, 0);
    const float scale = maximum / 127.0f;
    const float inverse = maximum != 0.0f ? 127.0f / maximum : 0.0f;
    int32_t code = __float2int_rn(gated_result * inverse);
    code = max(-128, min(127, code));
    const uint32_t activation_block = query_head * 8U + warp;
    uint8_t* destination = reinterpret_cast<uint8_t*>(workspace)
        + static_cast<uint64_t>(AUGUR_QWEN35_FULL_PARTIAL_ELEMENTS_V2) * sizeof(float)
        + static_cast<uint64_t>(activation_block) * augur_q8_1_block_bytes_v1;
    destination[4U + lane] = static_cast<uint8_t>(static_cast<int8_t>(code));
    if (lane == 0U) {
        auto* metadata = reinterpret_cast<uint16_t*>(destination);
        metadata[0] = __half_as_ushort(__float2half_rn(scale));
        metadata[1] = __half_as_ushort(__float2half_rn(sum));
    }
}

static augur_cuda_result_v1 augur_qwen35_full_validate_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_full_attention_args_v2* args
) {
    if (stream == nullptr || args == nullptr
        || args->abi_version != AUGUR_CUDA_QWEN35_FULL_ATTENTION_ARGS_VERSION_V2
        || args->struct_size != sizeof(augur_cuda_qwen35_full_attention_args_v2)
        || args->reserved != 0
        || args->capacity == 0 || args->capacity > AUGUR_QWEN35_FULL_MAX_CAPACITY
        || args->position >= args->capacity
        || args->query_heads != AUGUR_QWEN35_FULL_QUERY_HEADS
        || args->kv_heads != AUGUR_QWEN35_FULL_KV_HEADS
        || args->head_dim != AUGUR_QWEN35_FULL_HEAD_DIM
        || args->rotary_dim != AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1
        || args->split_count != AUGUR_QWEN35_FULL_SPLITS_V2) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t query_bytes =
        static_cast<uint64_t>(AUGUR_QWEN35_FULL_QUERY_ELEMENTS) * sizeof(float);
    const uint64_t kv_bytes =
        static_cast<uint64_t>(AUGUR_QWEN35_FULL_KV_ELEMENTS) * sizeof(float);
    const uint64_t cache_bytes = static_cast<uint64_t>(args->capacity) * kv_bytes;
    const uint64_t workspace_bytes =
        static_cast<uint64_t>(AUGUR_QWEN35_FULL_WORKSPACE_ELEMENTS_V2) * sizeof(float);
    const uint64_t addresses[] = {
        args->query_f32, args->gate_f32, args->key_f32, args->value_f32,
        args->key_cache_f32, args->value_cache_f32, args->attention_f32,
        args->gated_f32, args->workspace_f32,
    };
    const uint64_t lengths[] = {
        args->query_len_bytes, args->gate_len_bytes, args->key_len_bytes,
        args->value_len_bytes, args->key_cache_len_bytes, args->value_cache_len_bytes,
        args->attention_len_bytes, args->gated_len_bytes, args->workspace_len_bytes,
    };
    const uint64_t expected[] = {
        query_bytes, query_bytes, kv_bytes, kv_bytes, cache_bytes, cache_bytes,
        query_bytes, query_bytes, workspace_bytes,
    };
    for (uint32_t index = 0; index < 9; ++index) {
        if (lengths[index] != expected[index]
            || !augur_qwen35_full_range_ok(addresses[index], lengths[index])
            || (addresses[index] % alignof(float)) != 0) {
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t other = index + 1; other < 9; ++other) {
            if (augur_qwen35_full_overlap(
                    addresses[index], lengths[index], addresses[other], lengths[other])) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    return augur_validate_stream_device(ordinal, stream);
}

static augur_cuda_result_v1 augur_qwen35_full_submit_impl_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_full_attention_args_v2* args
) {
    const augur_cuda_result_v1 valid = augur_qwen35_full_validate_v2(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    float* key_cache = reinterpret_cast<float*>(static_cast<uintptr_t>(args->key_cache_f32));
    float* value_cache = reinterpret_cast<float*>(static_cast<uintptr_t>(args->value_cache_f32));
    augur_qwen35_grouped_gqa_stage1_v2<<<
        AUGUR_QWEN35_FULL_KV_HEADS * AUGUR_QWEN35_FULL_SPLITS_V2,
        AUGUR_QWEN35_FULL_STAGE1_THREADS_V2,
        0,
        stream->stream
    >>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->key_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->value_f32)),
        key_cache,
        value_cache,
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->workspace_f32)),
        args->position
    );
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
    }
    augur_qwen35_grouped_gqa_merge_v2<<<
        AUGUR_QWEN35_FULL_QUERY_HEADS * AUGUR_QWEN35_FULL_MERGE_SEGMENTS_V2,
        AUGUR_QWEN35_FULL_MERGE_THREADS_V2,
        0,
        stream->stream
    >>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->workspace_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->attention_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->gated_f32))
    );
    launch_error = cudaGetLastError();
    return launch_error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
}

#if defined(AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION)
constexpr uint32_t AUGUR_QWEN35_KV8_MAX_CAPACITY_V3 =
    AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY_V3;
constexpr uint32_t AUGUR_QWEN35_KV8_BLOCK_ELEMENTS_V3 =
    AUGUR_CUDA_QWEN35_KV8_BLOCK_ELEMENTS_V3;
constexpr uint32_t AUGUR_QWEN35_KV8_BLOCK_BYTES_V3 =
    AUGUR_CUDA_QWEN35_KV8_BLOCK_BYTES_V3;
constexpr uint32_t AUGUR_QWEN35_KV8_BLOCKS_PER_HEAD_V3 =
    AUGUR_CUDA_QWEN35_KV8_BLOCKS_PER_HEAD_V3;
constexpr uint32_t AUGUR_QWEN35_KV8_HEAD_ROW_BYTES_V3 =
    AUGUR_CUDA_QWEN35_KV8_HEAD_ROW_BYTES_V3;
constexpr uint32_t AUGUR_QWEN35_KV8_CACHE_ROW_BYTES_V3 =
    AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3;

static_assert(AUGUR_QWEN35_KV8_BLOCK_ELEMENTS_V3 == 32,
    "Qwen35 KV8 warp block geometry changed");
static_assert(
    AUGUR_QWEN35_KV8_BLOCKS_PER_HEAD_V3 * AUGUR_QWEN35_KV8_BLOCK_ELEMENTS_V3
        == AUGUR_QWEN35_FULL_HEAD_DIM,
    "Qwen35 KV8 head geometry changed");
static_assert(
    AUGUR_QWEN35_KV8_BLOCKS_PER_HEAD_V3 * AUGUR_QWEN35_KV8_BLOCK_BYTES_V3
        == AUGUR_QWEN35_KV8_HEAD_ROW_BYTES_V3,
    "Qwen35 KV8 head row bytes changed");
static_assert(
    AUGUR_QWEN35_FULL_KV_HEADS * AUGUR_QWEN35_KV8_HEAD_ROW_BYTES_V3
        == AUGUR_QWEN35_KV8_CACHE_ROW_BYTES_V3,
    "Qwen35 KV8 cache row bytes changed");

__device__ __forceinline__ float augur_qwen35_kv8_warp_max(float value) {
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    return __shfl_sync(0xffffffffU, value, 0);
}

__device__ __forceinline__ float augur_qwen35_kv8_decode(
    const uint8_t* cache,
    uint32_t position,
    uint32_t kv_head,
    uint32_t component,
    uint32_t capacity
) {
    const uint32_t block = component / AUGUR_QWEN35_KV8_BLOCK_ELEMENTS_V3;
    const uint32_t lane = component % AUGUR_QWEN35_KV8_BLOCK_ELEMENTS_V3;
    (void)capacity;
    const uint64_t block_base =
        static_cast<uint64_t>(position) * AUGUR_QWEN35_KV8_CACHE_ROW_BYTES_V3
        + static_cast<uint64_t>(kv_head) * AUGUR_QWEN35_KV8_HEAD_ROW_BYTES_V3
        + static_cast<uint64_t>(block) * AUGUR_QWEN35_KV8_BLOCK_BYTES_V3;
    const uint16_t scale_bits =
        *reinterpret_cast<const uint16_t*>(cache + block_base);
    const int8_t code = static_cast<int8_t>(cache[block_base + 2U + lane]);
    return __half2float(__ushort_as_half(scale_bits)) * static_cast<float>(code);
}

__global__ void augur_qwen35_grouped_gqa_kv8_stage1_v3(
    const float* query,
    const float* key,
    const float* value,
    uint8_t* key_cache,
    uint8_t* value_cache,
    float* workspace,
    uint32_t position,
    uint32_t capacity
) {
    const uint32_t visible_positions = position + 1U;
    const uint32_t split = blockIdx.x % AUGUR_QWEN35_FULL_SPLITS_V2;
    const uint32_t kv_head = blockIdx.x / AUGUR_QWEN35_FULL_SPLITS_V2;
    const uint32_t thread = threadIdx.x;
    const uint32_t warp = thread >> 5U;
    const uint32_t lane = thread & 31U;
    if (kv_head >= AUGUR_QWEN35_FULL_KV_HEADS) return;

    const uint32_t source_index = kv_head * AUGUR_QWEN35_FULL_HEAD_DIM + thread;
    const float current_key = key[source_index];
    const float current_value = value[source_index];
    const float key_abs_max = augur_qwen35_kv8_warp_max(fabsf(current_key));
    const float value_abs_max = augur_qwen35_kv8_warp_max(fabsf(current_value));
    const float key_inverse = key_abs_max != 0.0f ? 127.0f / key_abs_max : 0.0f;
    const float value_inverse = value_abs_max != 0.0f ? 127.0f / value_abs_max : 0.0f;
    const uint16_t key_scale_bits =
        __half_as_ushort(__float2half_rn(key_abs_max / 127.0f));
    const uint16_t value_scale_bits =
        __half_as_ushort(__float2half_rn(value_abs_max / 127.0f));
    const int8_t key_code = static_cast<int8_t>(max(
        -128, min(127, __float2int_rn(current_key * key_inverse))));
    const int8_t value_code = static_cast<int8_t>(max(
        -128, min(127, __float2int_rn(current_value * value_inverse))));
    const float current_key_q8 =
        __half2float(__ushort_as_half(key_scale_bits)) * static_cast<float>(key_code);
    const float current_value_q8 =
        __half2float(__ushort_as_half(value_scale_bits)) * static_cast<float>(value_code);

    if (split == 0U) {
        const uint64_t block_base =
            static_cast<uint64_t>(position) * AUGUR_QWEN35_KV8_CACHE_ROW_BYTES_V3
            + static_cast<uint64_t>(kv_head) * AUGUR_QWEN35_KV8_HEAD_ROW_BYTES_V3
            + static_cast<uint64_t>(warp) * AUGUR_QWEN35_KV8_BLOCK_BYTES_V3;
        if (lane == 0U) {
            *reinterpret_cast<uint16_t*>(key_cache + block_base) = key_scale_bits;
            *reinterpret_cast<uint16_t*>(value_cache + block_base) = value_scale_bits;
        }
        key_cache[block_base + 2U + lane] = static_cast<uint8_t>(key_code);
        value_cache[block_base + 2U + lane] = static_cast<uint8_t>(value_code);
    }

    const uint32_t split_begin = static_cast<uint32_t>(
        (static_cast<uint64_t>(visible_positions) * split) / AUGUR_QWEN35_FULL_SPLITS_V2);
    const uint32_t split_end = static_cast<uint32_t>(
        (static_cast<uint64_t>(visible_positions) * (split + 1U))
            / AUGUR_QWEN35_FULL_SPLITS_V2);
    const bool active_query = warp < AUGUR_QWEN35_FULL_QUERIES_PER_KV;
    const uint32_t query_head = kv_head * AUGUR_QWEN35_FULL_QUERIES_PER_KV + warp;

    __shared__ float key_tile[
        AUGUR_QWEN35_FULL_TILE_POSITIONS_V2 * AUGUR_QWEN35_FULL_HEAD_DIM];
    __shared__ float value_tile[
        AUGUR_QWEN35_FULL_TILE_POSITIONS_V2 * AUGUR_QWEN35_FULL_HEAD_DIM];

    float query_fragment[8] = {};
    float output_fragment[8] = {};
    if (active_query) {
        const uint32_t query_base = query_head * AUGUR_QWEN35_FULL_HEAD_DIM;
        #pragma unroll
        for (uint32_t item = 0; item < 8; ++item) {
            query_fragment[item] = query[query_base + lane + item * 32U];
        }
    }
    float running_maximum = -INFINITY;
    float running_sum = 0.0f;

    for (uint32_t tile_begin = split_begin; tile_begin < split_end;
         tile_begin += AUGUR_QWEN35_FULL_TILE_POSITIONS_V2) {
        const uint32_t remaining = split_end - tile_begin;
        const uint32_t tile_positions = remaining < AUGUR_QWEN35_FULL_TILE_POSITIONS_V2
            ? remaining
            : AUGUR_QWEN35_FULL_TILE_POSITIONS_V2;
        #pragma unroll
        for (uint32_t tile_position = 0;
             tile_position < AUGUR_QWEN35_FULL_TILE_POSITIONS_V2;
             ++tile_position) {
            if (tile_position < tile_positions) {
                const uint32_t absolute_position = tile_begin + tile_position;
                const uint32_t shared_index =
                    tile_position * AUGUR_QWEN35_FULL_HEAD_DIM + thread;
                if (absolute_position == position) {
                    key_tile[shared_index] = current_key_q8;
                    value_tile[shared_index] = current_value_q8;
                } else {
                    key_tile[shared_index] = augur_qwen35_kv8_decode(
                        key_cache, absolute_position, kv_head, thread, capacity);
                    value_tile[shared_index] = augur_qwen35_kv8_decode(
                        value_cache, absolute_position, kv_head, thread, capacity);
                }
            }
        }
        __syncthreads();

        if (active_query) {
            for (uint32_t tile_position = 0; tile_position < tile_positions; ++tile_position) {
                const uint32_t shared_base = tile_position * AUGUR_QWEN35_FULL_HEAD_DIM;
                double score = 0.0;
                #pragma unroll
                for (uint32_t item = 0; item < 8; ++item) {
                    score += static_cast<double>(query_fragment[item])
                        * static_cast<double>(key_tile[shared_base + lane + item * 32U]);
                }
                #pragma unroll
                for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
                    score += __shfl_down_sync(0xffffffffU, score, offset);
                }
                score = __shfl_sync(0xffffffffU, score, 0);
                const float scaled_score = static_cast<float>(score)
                    * rsqrtf(static_cast<float>(AUGUR_QWEN35_FULL_HEAD_DIM));
                const float next_maximum = fmaxf(running_maximum, scaled_score);
                const float previous_scale = running_sum == 0.0f
                    ? 0.0f
                    : expf(running_maximum - next_maximum);
                const float current_scale = expf(scaled_score - next_maximum);
                running_sum = running_sum * previous_scale + current_scale;
                #pragma unroll
                for (uint32_t item = 0; item < 8; ++item) {
                    const uint32_t component = lane + item * 32U;
                    output_fragment[item] = output_fragment[item] * previous_scale
                        + current_scale * value_tile[shared_base + component];
                }
                running_maximum = next_maximum;
            }
        }
        __syncthreads();
    }

    if (active_query) {
        const uint64_t partial_base =
            (static_cast<uint64_t>(query_head) * AUGUR_QWEN35_FULL_SPLITS_V2 + split)
                * AUGUR_QWEN35_FULL_PARTIAL_STRIDE_V2;
        if (lane == 0U) {
            workspace[partial_base] = running_maximum;
            workspace[partial_base + 1U] = running_sum;
        }
        #pragma unroll
        for (uint32_t item = 0; item < 8; ++item) {
            workspace[partial_base + 2U + lane + item * 32U] = output_fragment[item];
        }
    }
}

/* V4 quantizes all new rows in a separate launch.  This is required because
 * later rows may attend to earlier rows in the same batch; a single kernel
 * cannot provide a grid-wide publication barrier for those cache writes. */
__global__ void augur_qwen35_grouped_gqa_kv8_quantize_v4(
    const float* key,
    const float* value,
    uint8_t* key_cache,
    uint8_t* value_cache,
    uint32_t position_base,
    uint32_t tokens,
    uint32_t capacity
) {
    const uint32_t blocks_per_row = AUGUR_QWEN35_FULL_KV_HEADS * 8U;
    const uint32_t row = blockIdx.x / blocks_per_row;
    const uint32_t block = blockIdx.x % blocks_per_row;
    const uint32_t kv_head = block / 8U;
    const uint32_t warp = block % 8U;
    const uint32_t thread = threadIdx.x;
    const uint32_t lane = thread & 31U;
    if (row >= tokens || kv_head >= AUGUR_QWEN35_FULL_KV_HEADS || (thread >> 5U) != warp) return;
    const uint32_t position = position_base + row;
    const uint64_t row_base = static_cast<uint64_t>(row) * AUGUR_QWEN35_FULL_KV_ELEMENTS;
    const uint32_t source_index = kv_head * AUGUR_QWEN35_FULL_HEAD_DIM + thread;
    const float current_key = key[row_base + source_index];
    const float current_value = value[row_base + source_index];
    const float key_abs_max = augur_qwen35_kv8_warp_max(fabsf(current_key));
    const float value_abs_max = augur_qwen35_kv8_warp_max(fabsf(current_value));
    const float key_inverse = key_abs_max != 0.0f ? 127.0f / key_abs_max : 0.0f;
    const float value_inverse = value_abs_max != 0.0f ? 127.0f / value_abs_max : 0.0f;
    const uint16_t key_scale_bits =
        __half_as_ushort(__float2half_rn(key_abs_max / 127.0f));
    const uint16_t value_scale_bits =
        __half_as_ushort(__float2half_rn(value_abs_max / 127.0f));
    const int8_t key_code = static_cast<int8_t>(max(
        -128, min(127, __float2int_rn(current_key * key_inverse))));
    const int8_t value_code = static_cast<int8_t>(max(
        -128, min(127, __float2int_rn(current_value * value_inverse))));
    const uint64_t block_base = static_cast<uint64_t>(position)
        * AUGUR_QWEN35_KV8_CACHE_ROW_BYTES_V3
        + static_cast<uint64_t>(kv_head) * AUGUR_QWEN35_KV8_HEAD_ROW_BYTES_V3
        + static_cast<uint64_t>(warp) * AUGUR_QWEN35_KV8_BLOCK_BYTES_V3;
    (void)capacity;
    if (lane == 0U) {
        *reinterpret_cast<uint16_t*>(key_cache + block_base) = key_scale_bits;
        *reinterpret_cast<uint16_t*>(value_cache + block_base) = value_scale_bits;
    }
    key_cache[block_base + 2U + lane] = static_cast<uint8_t>(key_code);
    value_cache[block_base + 2U + lane] = static_cast<uint8_t>(value_code);
}

/* V4 keeps the V3 arithmetic byte-for-byte at the row level while adding a
 * launch dimension for a small contiguous causal batch.  The preceding
 * quantize launch is a grid-wide publication barrier for newly written KV. */
// Preserve each row's original split boundaries and reduction order. The union
// tile only shares decode/load; each row has independent softmax and causal mask.
template<unsigned Q> __global__ void augur_qwen35_grouped_gqa_kv8_shared_rows_v4(const float* query, const unsigned char* key,
    const unsigned char* value, float* workspace, unsigned base, unsigned rows) {
    constexpr unsigned PAR=Q<4?Q:4, STATES=Q/PAR;
    const unsigned split=blockIdx.x%72, head=blockIdx.x/72;
    const unsigned row0=blockIdx.y*Q;
    const unsigned group=threadIdx.x/32, local_row=group/6, qhead=head*6+group%6;
    const unsigned lane=threadIdx.x%32;
    unsigned first[STATES],last[STATES],row[STATES];
    float q[STATES][8], out[STATES][8]={};
    float maximum[STATES],sum[STATES];
    #pragma unroll
    for(unsigned s=0;s<STATES;++s){
        row[s]=row0+local_row+s*PAR;
        first[s]=((uint64_t(base)+row[s]+1)*split)/72;
        last[s]=((uint64_t(base)+row[s]+1)*(split+1))/72;
        maximum[s]=-INFINITY;sum[s]=0;
        #pragma unroll
        for(unsigned i=0;i<8;++i)q[s][i]=row[s]<rows?query[size_t(row[s])*6144+qhead*256+lane+i*32]:0;
    }
    __shared__ float keys[8*256],values[8*256];
    const unsigned tile_first=((uint64_t(base)+row0+1)*split)/72;
    const unsigned tile_last=((uint64_t(base)+min(row0+Q,rows))*(split+1))/72;
    for(unsigned begin=tile_first;begin<tile_last;begin+=8){
        const unsigned count=min(8U,tile_last-begin);
        for(unsigned index=threadIdx.x;index<count*256;index+=blockDim.x){
            unsigned position=begin+index/256,component=index%256;
            keys[index]=augur_qwen35_kv8_decode(key,position,head,component,0);
            values[index]=augur_qwen35_kv8_decode(value,position,head,component,0);
        }
        __syncthreads();
        #pragma unroll
        for(unsigned s=0;s<STATES;++s){
            if(row[s]<rows){
                for(unsigned t=0;t<count;++t){
                    unsigned position=begin+t;
                    if(position<first[s]||position>=last[s])continue;
                    double score=0;
                    #pragma unroll
                    for(unsigned i=0;i<8;++i)score+=double(q[s][i])*double(keys[t*256+lane+i*32]);
                    #pragma unroll
                    for(unsigned offset=16;offset;offset>>=1)score+=__shfl_down_sync(0xffffffff,score,offset);
                    score=__shfl_sync(0xffffffff,score,0);
                    float scaled=float(score)*rsqrtf(256.f), next=fmaxf(maximum[s],scaled);
                    float previous=sum[s]==0?0:expf(maximum[s]-next),current=expf(scaled-next);
                    sum[s]=sum[s]*previous+current;
                    #pragma unroll
                    for(unsigned i=0;i<8;++i)out[s][i]=out[s][i]*previous+current*values[t*256+lane+i*32];
                    maximum[s]=next;
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(unsigned s=0;s<STATES;++s){if(row[s]<rows){
        size_t dest=size_t(row[s])*447552+(qhead*72+split)*258;
        if(lane==0){workspace[dest]=maximum[s];workspace[dest+1]=sum[s];}
        #pragma unroll
        for(unsigned i=0;i<8;++i)workspace[dest+2+lane+i*32]=out[s][i];
    }}
}

__global__ void augur_qwen35_grouped_gqa_merge_v4(
    const float* gate,
    float* workspace,
    float* attention,
    float* gated,
    uint32_t tokens
) {
    const uint32_t blocks_per_row =
        AUGUR_QWEN35_FULL_QUERY_HEADS * AUGUR_QWEN35_FULL_MERGE_SEGMENTS_V2;
    const uint32_t row = blockIdx.x / blocks_per_row;
    const uint32_t block = blockIdx.x % blocks_per_row;
    if (row >= tokens) return;
    const uint32_t query_head = block / AUGUR_QWEN35_FULL_MERGE_SEGMENTS_V2;
    const uint32_t segment = block % AUGUR_QWEN35_FULL_MERGE_SEGMENTS_V2;
    const uint32_t component = segment * AUGUR_QWEN35_FULL_MERGE_THREADS_V2 + threadIdx.x;
    if (query_head >= AUGUR_QWEN35_FULL_QUERY_HEADS
        || component >= AUGUR_QWEN35_FULL_HEAD_DIM) return;
    const uint64_t query_elements = AUGUR_QWEN35_FULL_QUERY_ELEMENTS;
    const uint64_t workspace_elements = AUGUR_QWEN35_FULL_WORKSPACE_ELEMENTS_V2;
    const float* gate_ptr = gate + static_cast<uint64_t>(row) * query_elements;
    float* workspace_ptr = workspace + static_cast<uint64_t>(row) * workspace_elements;
    float* attention_ptr = attention + static_cast<uint64_t>(row) * query_elements;
    float* gated_ptr = gated + static_cast<uint64_t>(row) * query_elements;
    __shared__ float split_scales[AUGUR_QWEN35_FULL_SPLITS_V2];
    __shared__ float denominator_inverse;
    if (threadIdx.x == 0U) {
        float maximum = -INFINITY;
        #pragma unroll
        for (uint32_t split = 0; split < AUGUR_QWEN35_FULL_SPLITS_V2; ++split) {
            const uint64_t partial_base =
                (static_cast<uint64_t>(query_head) * AUGUR_QWEN35_FULL_SPLITS_V2 + split)
                    * AUGUR_QWEN35_FULL_PARTIAL_STRIDE_V2;
            if (workspace_ptr[partial_base + 1U] > 0.0f) {
                maximum = fmaxf(maximum, workspace_ptr[partial_base]);
            }
        }
        float denominator = 0.0f;
        #pragma unroll
        for (uint32_t split = 0; split < AUGUR_QWEN35_FULL_SPLITS_V2; ++split) {
            const uint64_t partial_base =
                (static_cast<uint64_t>(query_head) * AUGUR_QWEN35_FULL_SPLITS_V2 + split)
                    * AUGUR_QWEN35_FULL_PARTIAL_STRIDE_V2;
            const float partial_sum = workspace_ptr[partial_base + 1U];
            const float scale = partial_sum > 0.0f
                ? expf(workspace_ptr[partial_base] - maximum) : 0.0f;
            split_scales[split] = scale;
            denominator += partial_sum * scale;
        }
        denominator_inverse = 1.0f / denominator;
    }
    __syncthreads();
    float result = 0.0f;
    #pragma unroll
    for (uint32_t split = 0; split < AUGUR_QWEN35_FULL_SPLITS_V2; ++split) {
        const uint64_t partial_base =
            (static_cast<uint64_t>(query_head) * AUGUR_QWEN35_FULL_SPLITS_V2 + split)
                * AUGUR_QWEN35_FULL_PARTIAL_STRIDE_V2;
        result += workspace_ptr[partial_base + 2U + component] * split_scales[split];
    }
    result *= denominator_inverse;
    const uint32_t output_index = query_head * AUGUR_QWEN35_FULL_HEAD_DIM + component;
    const float gated_result = result * augur_qwen35_full_sigmoid(gate_ptr[output_index]);
    attention_ptr[output_index] = result;
    gated_ptr[output_index] = gated_result;
    const uint32_t lane = component & 31U;
    const uint32_t warp = component >> 5U;
    float maximum = fabsf(gated_result);
    float sum = gated_result;
    #pragma unroll
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    maximum = __shfl_sync(0xffffffffU, maximum, 0);
    const float scale = maximum / 127.0f;
    const float inverse = maximum != 0.0f ? 127.0f / maximum : 0.0f;
    int32_t code = __float2int_rn(gated_result * inverse);
    code = max(-128, min(127, code));
    const uint32_t activation_block = query_head * 8U + warp;
    uint8_t* destination = reinterpret_cast<uint8_t*>(workspace_ptr)
        + static_cast<uint64_t>(AUGUR_QWEN35_FULL_PARTIAL_ELEMENTS_V2) * sizeof(float)
        + static_cast<uint64_t>(activation_block) * augur_q8_1_block_bytes_v1;
    destination[4U + lane] = static_cast<uint8_t>(static_cast<int8_t>(code));
    if (lane == 0U) {
        auto* metadata = reinterpret_cast<uint16_t*>(destination);
        metadata[0] = __half_as_ushort(__float2half_rn(scale));
        metadata[1] = __half_as_ushort(__float2half_rn(sum));
    }
}

static augur_cuda_result_v1 augur_qwen35_kv8_full_validate_v3(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_full_attention_args_v3* args
) {
    if (stream == nullptr || args == nullptr
        || args->abi_version != AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION_ARGS_VERSION_V3
        || args->struct_size != sizeof(augur_cuda_qwen35_kv8_full_attention_args_v3)
        || args->reserved != 0
        || args->capacity == 0 || args->capacity > AUGUR_QWEN35_KV8_MAX_CAPACITY_V3
        || args->position >= args->capacity
        || args->query_heads != AUGUR_QWEN35_FULL_QUERY_HEADS
        || args->kv_heads != AUGUR_QWEN35_FULL_KV_HEADS
        || args->head_dim != AUGUR_QWEN35_FULL_HEAD_DIM
        || args->rotary_dim != AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1
        || args->split_count != AUGUR_QWEN35_FULL_SPLITS_V2) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t query_bytes =
        static_cast<uint64_t>(AUGUR_QWEN35_FULL_QUERY_ELEMENTS) * sizeof(float);
    const uint64_t kv_bytes =
        static_cast<uint64_t>(AUGUR_QWEN35_FULL_KV_ELEMENTS) * sizeof(float);
    const uint64_t cache_bytes =
        static_cast<uint64_t>(args->capacity) * AUGUR_QWEN35_KV8_CACHE_ROW_BYTES_V3;
    const uint64_t workspace_bytes =
        static_cast<uint64_t>(AUGUR_QWEN35_FULL_WORKSPACE_ELEMENTS_V2) * sizeof(float);
    const uint64_t addresses[] = {
        args->query_f32, args->gate_f32, args->key_f32, args->value_f32,
        args->key_cache_q8_0, args->value_cache_q8_0, args->attention_f32,
        args->gated_f32, args->workspace_f32,
    };
    const uint64_t lengths[] = {
        args->query_len_bytes, args->gate_len_bytes, args->key_len_bytes,
        args->value_len_bytes, args->key_cache_len_bytes, args->value_cache_len_bytes,
        args->attention_len_bytes, args->gated_len_bytes, args->workspace_len_bytes,
    };
    const uint64_t expected[] = {
        query_bytes, query_bytes, kv_bytes, kv_bytes, cache_bytes, cache_bytes,
        query_bytes, query_bytes, workspace_bytes,
    };
    for (uint32_t index = 0; index < 9; ++index) {
        const uint64_t alignment = index == 4U || index == 5U
            ? alignof(uint16_t)
            : alignof(float);
        if (lengths[index] != expected[index]
            || !augur_qwen35_full_range_ok(addresses[index], lengths[index])
            || (addresses[index] % alignment) != 0) {
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t other = index + 1; other < 9; ++other) {
            if (augur_qwen35_full_overlap(
                    addresses[index], lengths[index], addresses[other], lengths[other])) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    return augur_validate_stream_device(ordinal, stream);
}

static augur_cuda_result_v1 augur_qwen35_kv8_full_submit_impl_v3(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_full_attention_args_v3* args
) {
    const augur_cuda_result_v1 valid = augur_qwen35_kv8_full_validate_v3(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    augur_qwen35_grouped_gqa_kv8_stage1_v3<<<
        AUGUR_QWEN35_FULL_KV_HEADS * AUGUR_QWEN35_FULL_SPLITS_V2,
        AUGUR_QWEN35_FULL_STAGE1_THREADS_V2,
        0,
        stream->stream
    >>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->key_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->value_f32)),
        reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(args->key_cache_q8_0)),
        reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(args->value_cache_q8_0)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->workspace_f32)),
        args->position,
        args->capacity
    );
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
    }
    augur_qwen35_grouped_gqa_merge_v2<<<
        AUGUR_QWEN35_FULL_QUERY_HEADS * AUGUR_QWEN35_FULL_MERGE_SEGMENTS_V2,
        AUGUR_QWEN35_FULL_MERGE_THREADS_V2,
        0,
        stream->stream
    >>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->workspace_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->attention_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->gated_f32))
    );
    launch_error = cudaGetLastError();
    return launch_error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
}

#if defined(AUGUR_CUDA_QWEN35_PREFILL_CUBLAS)
#include "augur_qwen35_canonical_prefill.cuh"
#endif

static augur_cuda_result_v1 augur_qwen35_kv8_batched_validate_v4(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_batched_attention_args_v4* args,
    uint32_t version = AUGUR_CUDA_QWEN35_KV8_BATCHED_ATTENTION_ARGS_VERSION_V4,
    uint32_t max_tokens = AUGUR_CUDA_QWEN35_KV8_BATCHED_ATTENTION_MAX_TOKENS_V4
) {
    if (stream == nullptr || args == nullptr
        || args->abi_version != version
        || args->struct_size != sizeof(augur_cuda_qwen35_kv8_batched_attention_args_v4)
        || args->reserved != 0
        || args->capacity == 0 || args->capacity > AUGUR_QWEN35_KV8_MAX_CAPACITY_V3
        || args->tokens == 0 || args->tokens > max_tokens
        || args->position_base > args->capacity
        || args->tokens > args->capacity - args->position_base
        || args->query_heads != AUGUR_QWEN35_FULL_QUERY_HEADS
        || args->kv_heads != AUGUR_QWEN35_FULL_KV_HEADS
        || args->head_dim != AUGUR_QWEN35_FULL_HEAD_DIM
        || args->rotary_dim != AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1
        || args->split_count != AUGUR_QWEN35_FULL_SPLITS_V2) {
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    }
    const uint64_t row_query_bytes =
        static_cast<uint64_t>(AUGUR_QWEN35_FULL_QUERY_ELEMENTS) * sizeof(float);
    const uint64_t row_kv_bytes =
        static_cast<uint64_t>(AUGUR_QWEN35_FULL_KV_ELEMENTS) * sizeof(float);
    const uint64_t row_workspace_bytes =
        static_cast<uint64_t>(AUGUR_QWEN35_FULL_WORKSPACE_ELEMENTS_V2) * sizeof(float);
    const uint64_t cache_bytes =
        static_cast<uint64_t>(args->capacity) * AUGUR_QWEN35_KV8_CACHE_ROW_BYTES_V3;
    const uint64_t addresses[] = {
        args->query_f32, args->gate_f32, args->key_f32, args->value_f32,
        args->key_cache_q8_0, args->value_cache_q8_0, args->attention_f32,
        args->gated_f32, args->workspace_f32,
    };
    const uint64_t lengths[] = {
        static_cast<uint64_t>(args->tokens) * row_query_bytes,
        static_cast<uint64_t>(args->tokens) * row_query_bytes,
        static_cast<uint64_t>(args->tokens) * row_kv_bytes,
        static_cast<uint64_t>(args->tokens) * row_kv_bytes,
        cache_bytes, cache_bytes,
        static_cast<uint64_t>(args->tokens) * row_query_bytes,
        static_cast<uint64_t>(args->tokens) * row_query_bytes,
        static_cast<uint64_t>(args->tokens) * row_workspace_bytes,
    };
    const uint64_t declared[] = {
        args->query_len_bytes, args->gate_len_bytes, args->key_len_bytes,
        args->value_len_bytes, args->key_cache_len_bytes, args->value_cache_len_bytes,
        args->attention_len_bytes, args->gated_len_bytes, args->workspace_len_bytes,
    };
    for (uint32_t index = 0; index < 9; ++index) {
        const uint64_t alignment = index == 4U || index == 5U
            ? alignof(uint16_t) : alignof(float);
        if (lengths[index] == 0 || declared[index] != lengths[index]
            || !augur_qwen35_full_range_ok(addresses[index], lengths[index])
            || (addresses[index] % alignment) != 0) {
            return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
        }
        for (uint32_t other = index + 1; other < 9; ++other) {
            if (augur_qwen35_full_overlap(
                    addresses[index], lengths[index], addresses[other], lengths[other])) {
                return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
            }
        }
    }
    return augur_validate_stream_device(ordinal, stream);
}

static augur_cuda_result_v1 augur_qwen35_kv8_batched_submit_impl_v4(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_batched_attention_args_v4* args
) {
    const augur_cuda_result_v1 valid = augur_qwen35_kv8_batched_validate_v4(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    augur_qwen35_grouped_gqa_kv8_quantize_v4<<<
        args->tokens * AUGUR_QWEN35_FULL_KV_HEADS * 8U,
        AUGUR_QWEN35_FULL_STAGE1_THREADS_V2, 0, stream->stream
    >>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->key_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->value_f32)),
        reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(args->key_cache_q8_0)),
        reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(args->value_cache_q8_0)),
        args->position_base, args->tokens, args->capacity
    );
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
    }
    const dim3 grid(AUGUR_QWEN35_FULL_KV_HEADS * AUGUR_QWEN35_FULL_SPLITS_V2, 1);
    const float* query = reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query_f32));
    const uint8_t* keys = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->key_cache_q8_0));
    const uint8_t* values = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->value_cache_q8_0));
    float* workspace = reinterpret_cast<float*>(static_cast<uintptr_t>(args->workspace_f32));
    if (args->tokens > 4U) {
        augur_qwen35_grouped_gqa_kv8_shared_rows_v4<8><<<grid, 768, 0, stream->stream>>>(
            query, keys, values, workspace, args->position_base, args->tokens);
    } else if (args->tokens > 2U) {
        augur_qwen35_grouped_gqa_kv8_shared_rows_v4<4><<<grid, 768, 0, stream->stream>>>(
            query, keys, values, workspace, args->position_base, args->tokens);
    } else if (args->tokens == 2U) {
        augur_qwen35_grouped_gqa_kv8_shared_rows_v4<2><<<grid, 384, 0, stream->stream>>>(
            query, keys, values, workspace, args->position_base, args->tokens);
    } else {
        augur_qwen35_grouped_gqa_kv8_shared_rows_v4<1><<<grid, 192, 0, stream->stream>>>(
            query, keys, values, workspace, args->position_base, args->tokens);
    }
    launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
    }
    const uint32_t merge_blocks = args->tokens
        * AUGUR_QWEN35_FULL_QUERY_HEADS * AUGUR_QWEN35_FULL_MERGE_SEGMENTS_V2;
    augur_qwen35_grouped_gqa_merge_v4<<<
        merge_blocks, AUGUR_QWEN35_FULL_MERGE_THREADS_V2, 0, stream->stream
    >>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->workspace_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->attention_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->gated_f32)),
        args->tokens
    );
    launch_error = cudaGetLastError();
    return launch_error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, launch_error);
}
#endif
#endif

} // namespace

#if defined(AUGUR_CUDA_QWEN35_FUSED_HEAD_NORM_ROPE)
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_batched_head_norm_rope_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_batched_head_norm_rope_args_v1* args
) {
    return augur_qwen35_batched_head_norm_rope_submit_impl(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_fused_head_norm_rope_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_fused_head_norm_rope_args_v1* args
) {
    return augur_qwen35_fused_head_norm_rope_submit_impl(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_fused_head_norm_rope_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_fused_head_norm_rope_args_v1* args
) {
    const augur_cuda_result_v1 submitted =
        augur_qwen35_fused_head_norm_rope_submit_impl(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    const cudaError_t sync_error = cudaStreamSynchronize(stream->stream);
    return sync_error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, sync_error);
}
#endif

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_full_attention_submit_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_full_attention_args_v1* args
) {
    return augur_qwen35_full_submit_impl(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_full_attention_v1(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_full_attention_args_v1* args
) {
    const augur_cuda_result_v1 submitted = augur_qwen35_full_submit_impl(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    const cudaError_t sync_error = cudaStreamSynchronize(stream->stream);
    return sync_error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, sync_error);
}

#if defined(AUGUR_CUDA_QWEN35_GROUPED_GQA_ONLINE_SOFTMAX)
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_full_attention_submit_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_full_attention_args_v2* args
) {
    return augur_qwen35_full_submit_impl_v2(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_full_attention_v2(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_full_attention_args_v2* args
) {
    const augur_cuda_result_v1 submitted = augur_qwen35_full_submit_impl_v2(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    const cudaError_t sync_error = cudaStreamSynchronize(stream->stream);
    return sync_error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, sync_error);
}
#endif

#if defined(AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION)
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_full_attention_submit_v3(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_full_attention_args_v3* args
) {
    return augur_qwen35_kv8_full_submit_impl_v3(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_full_attention_v3(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_full_attention_args_v3* args
) {
    const augur_cuda_result_v1 submitted =
        augur_qwen35_kv8_full_submit_impl_v3(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    const cudaError_t sync_error = cudaStreamSynchronize(stream->stream);
    return sync_error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, sync_error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_batched_attention_submit_v4(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_batched_attention_args_v4* args
) {
    return augur_qwen35_kv8_batched_submit_impl_v4(ordinal, stream, args);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_kv8_batched_attention_v4(
    int32_t ordinal,
    augur_cuda_stream_t stream,
    const augur_cuda_qwen35_kv8_batched_attention_args_v4* args
) {
    const augur_cuda_result_v1 submitted =
        augur_qwen35_kv8_batched_submit_impl_v4(ordinal, stream, args);
    if (submitted.status != AUGUR_CUDA_STATUS_OK) return submitted;
    const cudaError_t sync_error = cudaStreamSynchronize(stream->stream);
    return sync_error == cudaSuccess
        ? augur_ok()
        : augur_error(AUGUR_CUDA_STATUS_RUNTIME_FAILURE, sync_error);
}
#if defined(AUGUR_CUDA_QWEN35_PREFILL_CUBLAS)
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_canonical_prefill_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_canonical_prefill_args_v1* args) {
    return augur_qwen35_kv8_batched_validate_v4(ordinal, stream, args,
        AUGUR_CUDA_QWEN35_CANONICAL_PREFILL_ARGS_VERSION_V1,
        AUGUR_CUDA_QWEN35_CANONICAL_PREFILL_MAX_TOKENS_V1);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_canonical_prefill_prepare_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_canonical_prefill_args_v1* args) {
    const auto valid = augur_cuda_qwen35_canonical_prefill_preflight_v1(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    augur_qwen35_grouped_gqa_kv8_quantize_v4<<<
        args->tokens * AUGUR_QWEN35_FULL_KV_HEADS * 8U,
        AUGUR_QWEN35_FULL_STAGE1_THREADS_V2, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->key_f32)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->value_f32)),
        reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(args->key_cache_q8_0)),
        reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(args->value_cache_q8_0)),
        args->position_base, args->tokens, args->capacity);
    const auto error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_canonical_prefill_consume_scores_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_canonical_prefill_args_v1* args,
    uint64_t scores, uint32_t key_begin, uint32_t key_count, uint32_t stride) {
    const auto valid = augur_cuda_qwen35_canonical_prefill_preflight_v1(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    if (stride == 0 || stride > 8192 || key_count == 0 || key_count > stride
        || key_begin > args->position_base + args->tokens
        || key_count > args->position_base + args->tokens - key_begin
        || scores % alignof(float) != 0
        || !augur_qwen35_full_range_ok(scores, uint64_t(args->tokens) * 24 * stride * sizeof(float)))
        return augur_error(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, cudaSuccess);
    const auto* s = reinterpret_cast<const float*>(static_cast<uintptr_t>(scores));
    const auto* v = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->value_cache_q8_0));
    auto* w = reinterpret_cast<float*>(static_cast<uintptr_t>(args->workspace_f32));
    const uint32_t groups = (args->tokens + 7U) / 8U;
    const dim3 grid(AUGUR_QWEN35_FULL_KV_HEADS * AUGUR_QWEN35_FULL_SPLITS_V2, groups);
    if (args->tokens > 4U)
        augur_qwen35_canonical_consume_scores_v1<8><<<grid, 768, 0, stream->stream>>>(s, v, w,
            args->position_base, args->tokens, key_begin, key_count, stride);
    else if (args->tokens > 2U)
        augur_qwen35_canonical_consume_scores_v1<4><<<grid, 768, 0, stream->stream>>>(s, v, w,
            args->position_base, args->tokens, key_begin, key_count, stride);
    else if (args->tokens == 2U)
        augur_qwen35_canonical_consume_scores_v1<2><<<grid, 384, 0, stream->stream>>>(s, v, w,
            args->position_base, args->tokens, key_begin, key_count, stride);
    else
        augur_qwen35_canonical_consume_scores_v1<1><<<grid, 192, 0, stream->stream>>>(s, v, w,
            args->position_base, args->tokens, key_begin, key_count, stride);
    const auto error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}

extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_canonical_prefill_finish_v1(
    int32_t ordinal, augur_cuda_stream_t stream,
    const augur_cuda_qwen35_canonical_prefill_args_v1* args) {
    const auto valid = augur_cuda_qwen35_canonical_prefill_preflight_v1(ordinal, stream, args);
    if (valid.status != AUGUR_CUDA_STATUS_OK) return valid;
    augur_qwen35_grouped_gqa_merge_v4<<<
        args->tokens * AUGUR_QWEN35_FULL_QUERY_HEADS * AUGUR_QWEN35_FULL_MERGE_SEGMENTS_V2,
        AUGUR_QWEN35_FULL_MERGE_THREADS_V2, 0, stream->stream>>>(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->gate_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->workspace_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->attention_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->gated_f32)), args->tokens);
    const auto error = cudaGetLastError();
    return error == cudaSuccess ? augur_ok() : augur_error(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
}
#endif
#endif
