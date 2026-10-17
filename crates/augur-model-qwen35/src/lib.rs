//! CPU-only identity and manifest checks for the Qwen3.8/27B lane.
//!
//! This crate deliberately has no dependency on `augur-infer`, CUDA, or the
//! DeepSeek product runtime.  It validates model artifacts and produces
//! identity-bound receipts; it does not execute a model or grant a throughput
//! or publication claim.

mod checkpoint_quant;
mod checkpoint_profile;
mod checkpoint_artifact;
mod checkpoint_dense;
mod checkpoint_layout;
mod config;
mod error;
mod gguf;
mod identity;
mod layer;
mod memory;
mod nvfp4_qpn;
mod quant;
mod reference;
mod safetensors;
mod schema;
mod strict_json;
mod tokenizer;
mod vision_geometry;
mod vision_input;
mod vision_mrope;
mod vision_precision;
mod vision_product;
mod vision_weights;
mod weights;

/// Validate UTF-8 JSON with duplicate-key and non-finite-number rejection.
pub fn validate_strict_json(bytes: &[u8]) -> Result<(), Qwen35Error> {
    strict_json::StrictValue::parse(bytes).map(|_| ())
}

pub use checkpoint_quant::{CheckpointFp8Matrix, CheckpointNvfp4Matrix, CheckpointNvfp4Scale};
pub use checkpoint_profile::{checkpoint_linear_specs, CheckpointLinearSpec, CheckpointLinearStorage, Qwen35CheckpointProfile};
pub use checkpoint_artifact::{CheckpointArtifact, CheckpointExpectedIdentity, LoadedCheckpointLinear};
pub use checkpoint_dense::{checkpoint_dense_specs, CheckpointDenseSpec, CheckpointDenseTransform};
pub use config::{DFlashConfig, QuantizationProfile, Qwen35Config};
pub use error::Qwen35Error;
pub use gguf::{GgufAuditOptions, GgufLoadOverlapTiming, GgufManifest, GgufPinnedArtifact, GgufTensor, GgufTensorReader, GgufValue};
pub use identity::{ArtifactPaths, ModelIdentity, ModelIdentityReceipt};
pub use layer::{
    qwen35_layer_forward, split_interleaved_query_gate, GgufProjection, Qwen35FfnWeights,
    Qwen35FullAttentionLayerWeights, Qwen35LayerCheckpoint, Qwen35LayerKind, Qwen35LayerOutput,
    Qwen35LayerSpec, Qwen35LayerState, Qwen35LayerWeights, Qwen35LinearLayerWeights,
    Qwen35Projection, ReferenceProjection,
};
pub use memory::{KvDType, MemoryAdmission, Qwen35StateLayout};
pub use nvfp4_qpn::Qwen35Nvfp4QpnGeometry;
pub use quant::{
    dequantize_bf16, dequantize_f32, dequantize_nvfp4_block, dequantize_q8_0_block,
    dequantize_tensor, quantize_q8_0_activation, quantized_matvec,
    quantized_matvec_activation_bf16, quantized_matvec_activation_q8_0, quantized_matvec_fused,
    quantized_matvec_fused_simd, read_dequantized_tensor, Qwen35QuantType, BF16_ELEMENT_BYTES,
    F32_ELEMENT_BYTES, GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_NVFP4, GGML_TYPE_Q8_0,
    NVFP4_BLOCK_BYTES, NVFP4_BLOCK_ELEMENTS, NVFP4_SUB_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMENTS,
};
pub use reference::{
    add_residual, bf16_round, embedding_lookup, l2_norm_last_dim, matmul_row_major,
    rms_norm_gated_last_dim, rms_norm_last_dim, rope_interleaved_last_dim,
    rope_split_half_last_dim, silu, softmax_last_dim, target_rope_last_dim, CausalConv1dState,
    GatedDeltaNetState, Qwen35FullAttentionCheckpoint, Qwen35FullAttentionState,
    Qwen35LinearAttentionBlock, Qwen35LinearAttentionCheckpoint, Qwen35LinearAttentionInputs,
    ReferenceTensor, TARGET_ROPE_SPLIT_HALF,
};
pub use safetensors::{SafetensorsManifest, SafetensorsPinnedShard, SafetensorsRecord};
pub use schema::{Qwen35TensorSchema, TensorSchemaSummary};
pub use tokenizer::TokenizerIdentity;
pub use vision_geometry::{
    Qwen35VisionGeometry, Qwen35VisionWorkspacePlan, QWEN35_VISION_HEADS, QWEN35_VISION_HEAD_DIM,
    QWEN35_VISION_HIDDEN, QWEN35_VISION_MAX_PATCHES,
};
pub use vision_input::{
    prepare_qwen35_aligned_rgb8, Qwen35ImageGrid, Qwen35ImagePromptLayout, Qwen35PreparedImage,
    QWEN35_IMAGE_PATCH_ELEMENTS,
};
pub use vision_mrope::{Qwen35ImageMropePlan, Qwen35ImageMropeReceipt, QWEN35_IMAGE_MROPE_ABI};
pub use vision_precision::{
    convert_qwen35_vision_bf16, qwen35_bf16_to_f16_bits, Qwen35VisionConversion,
    Qwen35VisionConversionStats, Qwen35VisionStorage,
};
pub use vision_product::{
    prepare_qwen35_aligned_rgb8_f16, Qwen35PreparedF16Image, Qwen35VisionInputReceipt,
    QWEN35_VISION_INPUT_ABI,
};
pub use vision_weights::{Qwen35VisionProfile, Qwen35VisionTensorBinding, Qwen35VisionWeightMap};
pub use weights::{
    read_qwen35_embedding_row, GgufEmbeddingRow, GgufEmbeddingRowProvider, GgufProjectionNumerics,
    GgufStreamedOutputHead, OwnedGgufProjection, DEFAULT_OUTPUT_HEAD_CHUNK_ROWS,
    MAX_OUTPUT_HEAD_CHUNK_ROWS,
};
