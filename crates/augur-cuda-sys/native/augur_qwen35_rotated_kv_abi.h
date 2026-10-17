#ifndef AUGUR_QWEN35_ROTATED_KV_ABI_H
#define AUGUR_QWEN35_ROTATED_KV_ABI_H
#include "augur_cuda_runtime.h"

// Independent cache ABI. Never reinterpret unrotated V3/V4 KV with this API.
// Format1: Q/K Hadamard256, V/output Hadamard64, Q8_0 row-major1088B.
// Single contiguous suffix; capacity is physically32-padded. Packing zeros
// [position+rows, round_up(position+rows,32)), never previous committed rows.
#define AUGUR_QWEN35_ROTATED_KV_VERSION_V1 1u
#define AUGUR_QWEN35_ROTATED_KV_FORMAT_V1 1u
// Explicit format2 entrypoints only: per-head256 codes +16 FP16-scale bytes.
// Same1088-byte position extent; never pass these bytes to format1 consumers.
#define AUGUR_QWEN35_ROTATED_KV_PLANAR_FORMAT_V2 2u
typedef struct augur_cuda_qwen35_rotated_kv_args_v1 {
    uint32_t abi_version,struct_size;
    uint64_t query,query_bytes;
    uint64_t gate,gate_bytes;
    uint64_t key,key_bytes;
    uint64_t value,value_bytes;
    uint64_t key_cache,key_cache_bytes;
    uint64_t value_cache,value_cache_bytes;
    uint64_t attention,attention_bytes;
    uint64_t gated,gated_bytes;
    uint64_t workspace,workspace_bytes;
    uint32_t capacity,position,rows,cache_format;
} augur_cuda_qwen35_rotated_kv_args_v1;

// Validate every extent/range/overlap; no allocation, launch or synchronization.
// This is structural validation, not CUDA pointer allocation ownership proof.
#ifdef __cplusplus
extern "C" {
#endif
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_validate_v1(
    const augur_cuda_qwen35_rotated_kv_args_v1*);
// Configure this exact SM70 kernel before graph capture/timing.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_configure_v1(int32_t ordinal);
// All buffers must belong to stream/ordinal and remain alive until stream drain.
// On partial submission failure the caller still owns/drains all buffers.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_submit_v1(
    int32_t ordinal,augur_cuda_stream_t,const augur_cuda_qwen35_rotated_kv_args_v1*);

// Scalar/verify consumer: identical format1 Q8 representation and argument
// layout, but rows1..8 and physical capacity256-aligned. Writes/zeros only
// [position, round_up(position+rows,256)); committed history remains unchanged.
// Workspace bytes = rows*(32768 + 2*capacity + 6144*32*4 + 24*32*8).
// Split arithmetic is calibrated once from the original SM70 reference kernels.
// Submission allocates no memory and does not query occupancy/read back/sync.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_small_validate_v1(
    const augur_cuda_qwen35_rotated_kv_args_v1*);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_small_configure_v1(int32_t ordinal);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_small_submit_v1(
    int32_t ordinal,augur_cuda_stream_t,const augur_cuda_qwen35_rotated_kv_args_v1*);
// Experimental batched execution of unchanged B1 vec arithmetic. Same ABI,
// workspace/cache format and validation; preserves per-query256 extent/splits.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_small_vector_submit_v1(
    int32_t ordinal,augur_cuda_stream_t,const augur_cuda_qwen35_rotated_kv_args_v1*);
// Separate experimental numerical contract: tile8 for ALL rows1..8, including
// scalar decode/replay. Not numerically interchangeable with B1 Q8-query vec.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_small_tile8_submit_v1(
    int32_t ordinal,augur_cuda_stream_t,const augur_cuda_qwen35_rotated_kv_args_v1*);

// Research-only GQA8 MMA B1..8. Same Q8 cache codec, different arithmetic from
// vec/tile8; select once for all scalar/verify/replay paths of one owner.
// Workspace = rows*32768 + 96*64*130*8; capacity256/workspace16 alignment.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_gqa8_validate_v1(
    const augur_cuda_qwen35_rotated_kv_args_v1*);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_gqa8_configure_v1(int32_t ordinal);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_gqa8_submit_v1(
    int32_t ordinal,augur_cuda_stream_t,const augur_cuda_qwen35_rotated_kv_args_v1*);

// Opt-in planar C4 query tile, immutable for scalar/verify/replay of its owner.
// B1..8 uses the same workspace extent as planar GQA8; current admission SM70/72SM.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_planar_gqa8_c4_validate_v1(
    const augur_cuda_qwen35_rotated_kv_args_v1*);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_planar_gqa8_c4_configure_v1(int32_t ordinal);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_planar_gqa8_c4_submit_v1(
    int32_t ordinal,augur_cuda_stream_t,const augur_cuda_qwen35_rotated_kv_args_v1*);

// Exact input-only consumer: rows9..512, capacity256-aligned, workspace16-aligned.
// Workspace = rows*32768 + metadata_blocks*64*130*8 bytes.
// metadata_blocks: rows128=144, rows224=252, rows256=288, rows480=540, rows512=576; otherwise96.
// Physical KV work extent is256-padded; logical causal history stays position+rows.
// Packing only modifies [position, round_up(position+rows,256)). Calibrates
// original16x2/32x2 occupancy once; submit has no allocation/readback/wait.
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_input_validate_v1(
    const augur_cuda_qwen35_rotated_kv_args_v1*);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_input_configure_v1(int32_t ordinal);
augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_qwen35_rotated_kv_input_submit_v1(
    int32_t ordinal,augur_cuda_stream_t,const augur_cuda_qwen35_rotated_kv_args_v1*);
#ifdef __cplusplus
}
#endif
#endif
