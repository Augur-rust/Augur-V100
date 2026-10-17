// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "augur_cuda_runtime.h"

// Physical QPN planes are NOT the GGML36 layout. Profile1 folds the tensor
// scalar at FP16 scale/weight boundaries and uses FP16 inputs, FP32 accumulation.
// Profile0 expansion is legacy-unscaled and is never admitted by projection.
// Profile2 projection/prefill keeps unscaled E4M3*E2M1 weights in FP16 and
// applies tensor_scale once to each FP32 contraction output. FP16 activation
// staging remains explicit. Profile1 and its default bindings are unchanged.
#ifdef __cplusplus
extern "C" {
#endif
typedef struct augur_cuda_qwen35_qpn_pack_args_v1 {
    uint32_t abi_version, struct_size;
    uint64_t source, source_bytes, destination, destination_bytes;
    uint32_t rows, cols;
} augur_cuda_qwen35_qpn_pack_args_v1;
typedef struct augur_cuda_qwen35_qpn_stage_args_v1 {
    uint32_t abi_version, struct_size;
    uint64_t source, source_bytes, destination, destination_bytes, elements;
} augur_cuda_qwen35_qpn_stage_args_v1;
typedef struct augur_cuda_qwen35_qpn_expand_args_v1 {
    uint32_t abi_version, struct_size;
    uint64_t source, source_bytes, destination, destination_bytes;
    uint32_t rows, cols;
    float tensor_scale;
    uint32_t numerical_profile;
} augur_cuda_qwen35_qpn_expand_args_v1;
typedef struct augur_cuda_qwen35_qpn_project_args_v1 {
    uint32_t abi_version, struct_size;
    uint64_t weight, weight_bytes, input, input_bytes, output, output_bytes;
    uint32_t rows, cols, tokens, split_k;
    float tensor_scale;
    uint32_t numerical_profile;
} augur_cuda_qwen35_qpn_project_args_v1;

// Original F32 activation consumer using an existing stream-owned cuBLAS
// workspace. Width1..8 uses QPN directly; wider batches expand FP16 according
// to the selected scalar-placement profile, with FP32 GEMM alpha for profile2.
// Profiles3/4/5 are B1..B8 ONLY and require two FP16 activation planes. Profile3
// adds a 2^-12 scaled residual contraction and F64 final merge. Finite lo
// encodes RNE_F16(clamp((x-hi)*4096,-65504,65504)); activation/hi nonfinite
// values propagate. Residual policies carry runtime contract revision2.
// Profile4 also
// limits each F32 MMA accumulation chain to four16-column groups and merges
// chunks in F64. Neither promises original-F32 bit identity or native W4A4.
// Profile5 flushes only hi every16 groups; lo uses the whole K slice before
// its2^-12 scaled F64 merge. Its separate contract needs separate admission.
// Profile6 packs hi/lo into unused MMA rows for B1..B4, uses separate hi/lo
// CTAs for B5..B8, four-group F32 chunks and plan-owned F64 VRAM partials.
// Per-component K order and the final F64 merge are unchanged (no atomics).
// Profile7 unrolls four groups and folds four K-warps per component in F64
// before the global merge. Same operands/chunks, distinct reduction tree.
// Profile8 uses profile7 kernels with a shape-fixed SplitK8/16/32 map.
// K1024x5120 SplitK32 has ten groups per slice (4+4+2 chunks).
// Wide tokens under profiles3..8 reject rather than changing precision.
typedef struct augur_cuda_qwen35_qpn_prefill_args_v1 {
    uint32_t abi_version, struct_size;
    uint64_t weight, weight_bytes, input_f32, input_bytes, output_f32, output_bytes;
    uint32_t rows, cols, tokens, split_k;
    float tensor_scale;
    uint32_t numerical_profile;
} augur_cuda_qwen35_qpn_prefill_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_prefill_preflight_v1(
    int32_t, augur_cuda_stream_t, augur_cuda_qwen35_prefill_cublas_plan_v1*, const augur_cuda_qwen35_qpn_prefill_args_v1*);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_prefill_prevalidated_submit_v1(
    int32_t, augur_cuda_stream_t, augur_cuda_qwen35_prefill_cublas_plan_v1*, const augur_cuda_qwen35_qpn_prefill_args_v1*);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_prefill_pair_prevalidated_submit_v1(
    int32_t, augur_cuda_stream_t, augur_cuda_qwen35_prefill_cublas_plan_v1*,
    const augur_cuda_qwen35_qpn_prefill_args_v1*, const augur_cuda_qwen35_qpn_prefill_args_v1*);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_silu_down_direct_b8_prevalidated_submit_v1(
    int32_t, augur_cuda_stream_t, augur_cuda_qwen35_prefill_cublas_plan_v1*,
    const augur_cuda_silu_mul_f32_args_v1*, const augur_cuda_qwen35_qpn_prefill_args_v1*);

// Separate wide numerical ABI. This is NOT profile8 with its width guard
// removed: unscaled FP16 weight, residual-v2 input, two full-K F32 GEMMs,
// F64 hi/lo/global-scale merge and one F32 output cast. No width invariance
// is implied; exact replay must bind the width and library identity.
// Explicit wide profile2 admits ONLY down5120x17408 and B1..128: splitK8,
// hi/lo F32 GEMMs per2176 columns, ordered F64 merge and one final F32 cast.
// Needs3*output_bytes partial space; profile1 and legacy profiles are unchanged.
// One atomic host-side producer/consumer program: fixed5120 cached norm,
// F32 output + residual-v2 planes, followed immediately by2..3 profile8 B1
// projections. Full cross-operation validation occurs BEFORE any enqueue.
// Scratch is freshly produced each time; no public stage-ready handle.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_norm_group_preflight_v1(
    int32_t, augur_cuda_stream_t, augur_cuda_qwen35_prefill_cublas_plan_v1*,
    const augur_cuda_rmsnorm_f32_args_v1*, const augur_cuda_qwen35_qpn_prefill_args_v1*, uint32_t);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_norm_group_prevalidated_submit_v1(
    int32_t, augur_cuda_stream_t, augur_cuda_qwen35_prefill_cublas_plan_v1*,
    const augur_cuda_rmsnorm_f32_args_v1*, const augur_cuda_qwen35_qpn_prefill_args_v1*, uint32_t);

// Direct-B8 producer/consumer program: [optional residual add] -> fixed5120 rmsnorm
// (prefill_rmsnorm_kernel arithmetic) -> FP16 stage -> 2..4 profile2/SplitK8 projections
// of that normalized input in ONE grouped launch. Member i writes out_cols[i] columns
// per token (= rows, or 48 of a 64-row gate tile) at stride out_cols[i], and optionally
// the identical words to records[i]. Every range and alias is validated before enqueue.
// Bit-identical to rmsnorm + stage + separate project<8,half,true> launches + copies.
typedef struct augur_cuda_qwen35_qpn_norm_group_b8_args_v1 {
    uint32_t abi_version, struct_size;
    uint32_t count, residual;
    augur_cuda_qwen35_prefill_rmsnorm_args_v1 norm;
    augur_cuda_qwen35_qpn_prefill_args_v1 projections[4];
    uint32_t out_cols[4];
    uint64_t records[4], record_bytes[4];
    uint64_t add_a, add_b, add_len_bytes;
} augur_cuda_qwen35_qpn_norm_group_b8_args_v1;
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_norm_group_b8_preflight_v1(
    int32_t, augur_cuda_stream_t, augur_cuda_qwen35_prefill_cublas_plan_v1*,
    const augur_cuda_qwen35_qpn_norm_group_b8_args_v1*);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_norm_group_b8_prevalidated_submit_v1(
    int32_t, augur_cuda_stream_t, augur_cuda_qwen35_prefill_cublas_plan_v1*,
    const augur_cuda_qwen35_qpn_norm_group_b8_args_v1*);

// Profiles6/7/8. Explicit retained scratch, at least rows*tokens*split_k*16
// bytes for6, or rows*tokens*split_k*4 bytes for7/8, disjoint from every operand/output. Original project ABI unchanged.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_workspace_preflight_v1(
    int32_t, augur_cuda_stream_t, const augur_cuda_qwen35_qpn_project_args_v1*, uint64_t, uint64_t);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_workspace_prevalidated_submit_v1(
    int32_t, augur_cuda_stream_t, const augur_cuda_qwen35_qpn_project_args_v1*, uint64_t, uint64_t);

// Existing descriptors; no allocation or change to either descriptor layout.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_silu_down_prevalidated_submit_v1(
    int32_t, augur_cuda_stream_t, augur_cuda_qwen35_prefill_cublas_plan_v1*,
    const augur_cuda_silu_mul_f32_args_v1*, const augur_cuda_qwen35_qpn_prefill_args_v1*);

// Profile8 shape policy; both the owner and native ABI enforce this map.
// Unknown admitted component-test geometries retain the eight-way schedule.
static inline uint32_t augur_qwen35_qpn_shape_split_k_v1(uint32_t rows,uint32_t cols) {
    if((rows==1024&&cols==5120)||(rows==5120&&cols==6144)) return 32;
    if((rows==17408&&cols==5120)||(rows==5120&&cols==17408)
       ||(cols==5120&&(rows==10240||rows==6144||rows==12288))) return 16;
    return 8;
}

// Preflight checks actual allocation extents/device ownership/SM70. Submission
// repeats scalar/extent/overlap checks but not driver queries; live owners and
// the same stream must remain retained. None of these calls synchronizes.
#define AUGUR_QPN_ENTRY(NAME, TYPE) \
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_##NAME##_preflight_v1( \
    int32_t ordinal, augur_cuda_stream_t stream, const TYPE* args); \
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_qpn_##NAME##_prevalidated_submit_v1( \
    int32_t ordinal, augur_cuda_stream_t stream, const TYPE* args);
AUGUR_QPN_ENTRY(pack, augur_cuda_qwen35_qpn_pack_args_v1)
// Same extent/ownership checks, but source is natural checkpoint codes followed
// by scales, NOT GGML36. The separate entry prevents an implicit layout change.
AUGUR_QPN_ENTRY(pack_checkpoint, augur_cuda_qwen35_qpn_pack_args_v1)
AUGUR_QPN_ENTRY(stage, augur_cuda_qwen35_qpn_stage_args_v1)
// Separate ABI entry, two FP16 planes, no change to original stage extent.
AUGUR_QPN_ENTRY(stage_residual, augur_cuda_qwen35_qpn_stage_args_v1)
AUGUR_QPN_ENTRY(expand, augur_cuda_qwen35_qpn_expand_args_v1)
AUGUR_QPN_ENTRY(project, augur_cuda_qwen35_qpn_project_args_v1)
#undef AUGUR_QPN_ENTRY
#ifdef __cplusplus
}
#endif
