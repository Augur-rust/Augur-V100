//! Optional resident owners for the Qwen35 autoregressive lane.
//!
//! This module owns the physical placement boundary for the auxiliary
//! DFlash/Vision payloads and the bounded Q8 selector consumer. It deliberately
//! does not execute the complete DFlash draft or a vision forward pass; those
//! claims remain closed until their end-to-end consumers are audited.

use crate::dflash_q8_features::encode_target_features_kv_q8;
use augur_backend::{
    CudaDflashCacheAttentionF32ArgsV1, CudaDflashRopeF32ArgsV1, CudaF32View,
    CudaGqaRepeatF32ArgsV1, CudaQ8_0F32MatvecArgsV1, CudaQ8_0WeightViewV1, CudaRmsnormF32ArgsV1,
    CudaStreamSubmissionReceipt, OwnedCudaBuffer, OwnedCudaEvent, OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;
use augur_model_qwen35::{
    quantized_matvec, GgufAuditOptions, GgufManifest, GgufTensor, GGML_TYPE_F32, GGML_TYPE_Q8_0,
};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::Instant;
use std::{
    fs,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

const MAX_SAFETENSORS_HEADER_BYTES: usize = 16 * 1024 * 1024;
const VISUAL_PREFIX: &str = "model.visual.";

#[path = "dflash_cache_contract.rs"]
mod cache_contract;

#[derive(Clone, Debug, Serialize)]
pub struct ResidentAuxiliaryArtifactReport {
    pub path: PathBuf,
    pub file_bytes: usize,
    pub resident_device_bytes: usize,
    pub payload_sha256: String,
    pub device_resident: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResidentAuxiliaryReport {
    pub q8_dflash: Option<ResidentAuxiliaryArtifactReport>,
    pub vision: Option<ResidentAuxiliaryArtifactReport>,
    pub owner_bytes: usize,
    pub h2d_bytes: usize,
    pub owner_count: usize,
    pub upload_wall_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q8_selector_consumer: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q8_feature_kv_consumer: Option<Value>,
    pub cleanup_zero: bool,
    pub decode_consumer_executed: bool,
    pub vision_forward_executed: bool,
    pub semantic_parity: bool,
    pub product_ready: bool,
}

pub struct ResidentAuxiliaryOwners {
    _q8_dflash: Option<OwnedCudaBuffer>,
    _vision: Option<OwnedCudaBuffer>,
    q8_selector_consumer: Option<Q8SelectorConsumer>,
    q8_feature_kv_consumer: Option<Q8FeatureKvConsumer>,
    report: ResidentAuxiliaryReport,
}

/// Same-owner Q8 DFlash feature encoder and five-layer K/V projector.
///
/// This is deliberately a component consumer: it captures real target-layer
/// hidden rows via same-stream D2D, then writes K/V workspaces in place. The
/// attention uses repeated rotated K as a diagnostic query, not a learned
/// draft query. The anchor/mask decoder and acceptance remain separate gates.
struct Q8FeatureKvConsumer {
    fc_weight: CudaQ8_0WeightViewV1,
    encoder_norm: CudaF32View,
    k_weights: [CudaQ8_0WeightViewV1; 5],
    v_weights: [CudaQ8_0WeightViewV1; 5],
    k_norm_weights: [CudaF32View; 5],
    feature_input: OwnedCudaBuffer,
    projected: OwnedCudaBuffer,
    normalized: OwnedCudaBuffer,
    k_outputs: [OwnedCudaBuffer; 5],
    k_rope_outputs: [OwnedCudaBuffer; 5],
    v_outputs: [OwnedCudaBuffer; 5],
    k_normed_outputs: [OwnedCudaBuffer; 5],
    query_repeat_outputs: [OwnedCudaBuffer; 5],
    cache_attention_outputs: [OwnedCudaBuffer; 5],
    k_cache: OwnedCudaBuffer,
    v_cache: OwnedCudaBuffer,
    artifact_path: PathBuf,
    artifact_sha256: String,
    submissions: usize,
    last_step: Option<usize>,
    last_position: Option<usize>,
    capture_mask: u8,
    cache_attention_submissions: usize,
    cache_valid_positions: usize,
    poisoned: bool,
}

/// A target-bound Q8 selector consumer backed by the same full-file HBM owner
/// used by the resident auxiliary lifecycle.  This deliberately consumes
/// only `selector_hidden.weight`; it is not the complete DFlash draft.
struct Q8SelectorConsumer {
    weight: CudaQ8_0WeightViewV1,
    output: OwnedCudaBuffer,
    tensor: GgufTensor,
    payload: Vec<u8>,
    file_bytes: usize,
    file_sha256: String,
    tensor_offset: usize,
    tensor_bytes: usize,
    rows: usize,
    cols: usize,
    submissions: usize,
    last_step: Option<usize>,
    terminal_output: Option<Vec<f32>>,
    terminal: Option<(OwnedCudaEvent, CudaStreamSubmissionReceipt)>,
}

const Q8_SELECTOR_TENSOR: &str = "selector_hidden.weight";
const Q8_SELECTOR_ROWS: usize = 256;
const Q8_SELECTOR_COLS: usize = 5120;
const Q8_FEATURE_HIDDEN: usize = 5120;
const Q8_FEATURE_INPUT: usize = 25600;
const Q8_FEATURE_KV: usize = 1024;
const Q8_FEATURE_HEAD_DIM: usize = 128;
const Q8_FEATURE_ROPE_THETA: f32 = 10_000_000.0;
const Q8_FEATURE_ROPE_DIM: usize = 128;
const Q8_FEATURE_CACHE_POSITIONS: usize = 2048;
const Q8_FEATURE_TARGET_LAYERS: [usize; 5] = [5, 19, 33, 47, 61];

fn numerical_ab(actual: &[f32], expected: &[f32]) -> Value {
    let mut max_abs = 0.0_f32;
    let mut max_relative = 0.0_f32;
    let mut finite = true;
    for (&a, &e) in actual.iter().zip(expected) {
        finite &= a.is_finite() && e.is_finite();
        max_abs = max_abs.max((a - e).abs());
        max_relative = max_relative.max((a - e).abs() / e.abs().max(1.0e-6));
    }
    json!({
        "pass": finite && actual.len() == expected.len() && max_abs <= 0.25 && max_relative <= 0.05,
        "finite": finite,
        "count": actual.len(),
        "expected_count": expected.len(),
        "max_abs": max_abs,
        "max_relative": max_relative,
        "threshold_max_abs": 0.25,
        "threshold_max_relative": 0.05,
    })
}

fn rotate_neox_f32(values: &[f32], position: usize, theta: f32) -> Vec<f32> {
    let mut output = values.to_vec();
    for head in output.chunks_exact_mut(Q8_FEATURE_HEAD_DIM) {
        let source = head.to_vec();
        let half = Q8_FEATURE_HEAD_DIM / 2;
        for index in 0..half {
            let exponent = -2.0_f64 * index as f64 / Q8_FEATURE_ROPE_DIM as f64;
            let inverse_frequency = (theta as f64).powf(exponent) as f32;
            let (sine, cosine) = (position as f32 * inverse_frequency).sin_cos();
            let first = source[index];
            let second = source[index + half];
            head[index] = first * cosine - second * sine;
            head[index + half] = second * cosine + first * sine;
        }
    }
    output
}

fn attention_ab(actual: &[f32], expected: &[f32]) -> Value {
    let atol = 2.0e-4_f32;
    let rtol = 1.0e-4_f32;
    let mut max_abs = 0.0_f32;
    let mut pass = actual.len() == expected.len() && !actual.is_empty();
    for (&a, &e) in actual.iter().zip(expected) {
        max_abs = max_abs.max((a - e).abs());
        pass &= a.is_finite() && e.is_finite() && (a - e).abs() <= atol + rtol * e.abs();
    }
    json!({"pass": pass, "count": actual.len(), "max_abs": max_abs,
        "atol": atol, "rtol": rtol, "rule": "abs(a-e)<=atol+rtol*abs(e)"})
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(core::mem::size_of::<f32>()) {
        return Err("F32 readback extent is not aligned".to_owned());
    }
    Ok(bytes
        .chunks_exact(core::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("aligned F32 chunk")))
        .collect())
}

fn prefix(values: &[f32]) -> Vec<f32> {
    values.iter().copied().take(8).collect()
}

fn q8_weight_view(
    owner: &OwnedCudaBuffer,
    tensor: &GgufTensor,
    rows: usize,
    cols: usize,
) -> Result<CudaQ8_0WeightViewV1, String> {
    let offset = usize::try_from(tensor.absolute_offset)
        .map_err(|_| format!("{} offset overflow", tensor.name))?;
    let bytes = usize::try_from(tensor.byte_len)
        .map_err(|_| format!("{} byte length overflow", tensor.name))?;
    CudaQ8_0WeightViewV1::new(
        owner
            .view(offset, bytes)
            .map_err(|error| error.to_string())?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
        rows,
        cols,
    )
    .map_err(|error| format!("{} Q8 view: {error}", tensor.name))
}

fn f32_weight_view(
    owner: &OwnedCudaBuffer,
    tensor: &GgufTensor,
    elements: usize,
) -> Result<CudaF32View, String> {
    let offset = usize::try_from(tensor.absolute_offset)
        .map_err(|_| format!("{} offset overflow", tensor.name))?;
    let bytes = elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| format!("{} byte length overflow", tensor.name))?;
    CudaF32View::new(
        owner
            .view(offset, bytes)
            .map_err(|error| error.to_string())?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        elements,
    )
    .map_err(|error| format!("{} F32 view: {error}", tensor.name))
}

impl Q8FeatureKvConsumer {
    fn bind(
        path: &Path,
        file: &[u8],
        owner: &OwnedCudaBuffer,
        device: i32,
    ) -> Result<Self, String> {
        let manifest = GgufManifest::inspect(
            path,
            GgufAuditOptions {
                hash_file: false,
                ..Default::default()
            },
        )
        .map_err(|error| format!("Q8 feature-KV manifest: {error}"))?;
        if manifest.file_size != file.len() as u64 {
            return Err("Q8 feature-KV artifact extent changed after inspection".into());
        }
        let fc = manifest
            .tensor("fc.weight")
            .ok_or_else(|| "fc.weight missing".to_owned())?;
        let encoder_norm = manifest
            .tensor("enc.output_norm.weight")
            .ok_or_else(|| "enc.output_norm.weight missing".to_owned())?;
        if fc.ggml_type != GGML_TYPE_Q8_0
            || fc.dimensions != vec![Q8_FEATURE_INPUT as u64, Q8_FEATURE_HIDDEN as u64]
            || encoder_norm.ggml_type != GGML_TYPE_F32
            || encoder_norm.dimensions != vec![Q8_FEATURE_HIDDEN as u64]
        {
            return Err("Q8 feature-KV encoder tensor geometry".into());
        }
        let fc_weight = q8_weight_view(owner, fc, Q8_FEATURE_HIDDEN, Q8_FEATURE_INPUT)?;
        let encoder_norm = f32_weight_view(owner, encoder_norm, Q8_FEATURE_HIDDEN)?;
        let mut k_weights = Vec::with_capacity(5);
        let mut v_weights = Vec::with_capacity(5);
        let mut k_norm_weights = Vec::with_capacity(5);
        for layer in 0..5 {
            let k_name = format!("blk.{layer}.attn_k.weight");
            let v_name = format!("blk.{layer}.attn_v.weight");
            let n_name = format!("blk.{layer}.attn_k_norm.weight");
            let k = manifest
                .tensor(&k_name)
                .ok_or_else(|| format!("{k_name} missing"))?;
            let v = manifest
                .tensor(&v_name)
                .ok_or_else(|| format!("{v_name} missing"))?;
            let n = manifest
                .tensor(&n_name)
                .ok_or_else(|| format!("{n_name} missing"))?;
            if k.ggml_type != GGML_TYPE_Q8_0
                || v.ggml_type != GGML_TYPE_Q8_0
                || k.dimensions != vec![Q8_FEATURE_HIDDEN as u64, Q8_FEATURE_KV as u64]
                || v.dimensions != vec![Q8_FEATURE_HIDDEN as u64, Q8_FEATURE_KV as u64]
                || n.ggml_type != GGML_TYPE_F32
                || n.dimensions != vec![128]
            {
                return Err(format!("{k_name}/{v_name}/{n_name} geometry"));
            }
            k_weights.push(q8_weight_view(owner, k, Q8_FEATURE_KV, Q8_FEATURE_HIDDEN)?);
            v_weights.push(q8_weight_view(owner, v, Q8_FEATURE_KV, Q8_FEATURE_HIDDEN)?);
            k_norm_weights.push(f32_weight_view(owner, n, 128)?);
        }
        let k_weights: [CudaQ8_0WeightViewV1; 5] = k_weights
            .try_into()
            .map_err(|_| "Q8 feature-KV K weight array".to_owned())?;
        let v_weights: [CudaQ8_0WeightViewV1; 5] = v_weights
            .try_into()
            .map_err(|_| "Q8 feature-KV V weight array".to_owned())?;
        let k_norm_weights: [CudaF32View; 5] = k_norm_weights
            .try_into()
            .map_err(|_| "Q8 feature-KV norm weight array".to_owned())?;
        let make_buffers = |count: usize, elements: usize| {
            (0..count)
                .map(|_| {
                    OwnedCudaBuffer::allocate(device, elements * 4)
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let k_outputs: [OwnedCudaBuffer; 5] = make_buffers(5, Q8_FEATURE_KV)?
            .try_into()
            .map_err(|_| "Q8 feature-KV K output array".to_owned())?;
        let k_rope_outputs: [OwnedCudaBuffer; 5] = make_buffers(5, Q8_FEATURE_KV)?
            .try_into()
            .map_err(|_| "Q8 feature-KV rotated K output array".to_owned())?;
        let v_outputs: [OwnedCudaBuffer; 5] = make_buffers(5, Q8_FEATURE_KV)?
            .try_into()
            .map_err(|_| "Q8 feature-KV V output array".to_owned())?;
        let k_normed_outputs: [OwnedCudaBuffer; 5] = make_buffers(5, Q8_FEATURE_KV)?
            .try_into()
            .map_err(|_| "Q8 feature-KV K norm output array".to_owned())?;
        let cache_bytes = 5usize
            .checked_mul(Q8_FEATURE_CACHE_POSITIONS)
            .and_then(|value| value.checked_mul(Q8_FEATURE_KV))
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| "Q8 feature-KV cache extent overflow".to_owned())?;
        Ok(Self {
            fc_weight,
            encoder_norm,
            k_weights,
            v_weights,
            k_norm_weights,
            feature_input: OwnedCudaBuffer::allocate(device, Q8_FEATURE_INPUT * 4)
                .map_err(|error| error.to_string())?,
            projected: OwnedCudaBuffer::allocate(device, Q8_FEATURE_HIDDEN * 4)
                .map_err(|error| error.to_string())?,
            normalized: OwnedCudaBuffer::allocate(device, Q8_FEATURE_HIDDEN * 4)
                .map_err(|error| error.to_string())?,
            k_outputs,
            k_rope_outputs,
            v_outputs,
            k_normed_outputs,
            query_repeat_outputs: make_buffers(5, 4 * Q8_FEATURE_KV)?
                .try_into()
                .map_err(|_| "Q8 feature-KV query repeat output array".to_owned())?,
            cache_attention_outputs: make_buffers(5, 4 * Q8_FEATURE_KV)?
                .try_into()
                .map_err(|_| "Q8 feature-KV cache attention output array".to_owned())?,
            k_cache: OwnedCudaBuffer::allocate(device, cache_bytes)
                .map_err(|error| format!("Q8 feature-KV K cache: {error}"))?,
            v_cache: OwnedCudaBuffer::allocate(device, cache_bytes)
                .map_err(|error| format!("Q8 feature-KV V cache: {error}"))?,
            artifact_path: path.to_owned(),
            artifact_sha256: sha256_hex(file),
            submissions: 0,
            last_step: None,
            last_position: None,
            capture_mask: 0,
            cache_attention_submissions: 0,
            cache_valid_positions: 0,
            poisoned: false,
        })
    }

    fn capture_layer(
        &mut self,
        layer: usize,
        hidden: CudaF32View,
        stream: &OwnedCudaStream,
    ) -> Result<bool, String> {
        let Some(index) = Q8_FEATURE_TARGET_LAYERS
            .iter()
            .position(|&value| value == layer)
        else {
            return Ok(false);
        };
        if hidden.element_count() != Q8_FEATURE_HIDDEN {
            return Err("Q8 feature-KV target hidden geometry".into());
        }
        if self.poisoned || self.capture_mask & (1 << index) != 0 {
            return Err("Q8 feature-KV poisoned owner or duplicate layer capture".into());
        }
        let offset = index * Q8_FEATURE_HIDDEN * 4;
        self.poisoned = true;
        unsafe {
            self.feature_input.copy_from_device_async_unchecked(
                stream,
                offset,
                hidden.buffer_view(),
            )
        }
        .map_err(|error| format!("Q8 feature-KV target capture: {error}"))?;
        self.poisoned = false;
        self.capture_mask |= 1_u8 << index;
        Ok(true)
    }

    fn submit(
        &mut self,
        step: usize,
        position: usize,
        stream: &OwnedCudaStream,
    ) -> Result<(), String> {
        if self.poisoned || self.capture_mask != 0b1_1111 {
            return Err("Q8 feature-KV submit requires all five target layers".into());
        }
        let cache_window = cache_contract::next_window(
            self.last_step,
            self.last_position,
            self.cache_valid_positions,
            Q8_FEATURE_CACHE_POSITIONS,
            step,
            position,
        )?;
        // Any subsequent error may follow a CUDA write. No reuse until drain;
        // commit all five sibling windows only after successful enqueue.
        self.poisoned = true;
        let input = CudaF32View::new(
            self.feature_input
                .view(0, Q8_FEATURE_INPUT * 4)
                .map_err(|error| error.to_string())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            Q8_FEATURE_INPUT,
        )
        .map_err(|error| error.to_string())?;
        let projected = CudaF32View::new(
            self.projected
                .view(0, Q8_FEATURE_HIDDEN * 4)
                .map_err(|error| error.to_string())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            Q8_FEATURE_HIDDEN,
        )
        .map_err(|error| error.to_string())?;
        let normalized = CudaF32View::new(
            self.normalized
                .view(0, Q8_FEATURE_HIDDEN * 4)
                .map_err(|error| error.to_string())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            Q8_FEATURE_HIDDEN,
        )
        .map_err(|error| error.to_string())?;
        CudaQ8_0F32MatvecArgsV1::bind(stream, self.fc_weight, input, projected)
            .map_err(|error| error.to_string())?
            .submit_for_plan(stream)
            .map_err(|error| error.to_string())?;
        CudaRmsnormF32ArgsV1::bind(stream, projected, self.encoder_norm, normalized, 1.0e-6)
            .map_err(|error| error.to_string())?
            .submit_for_plan(stream)
            .map_err(|error| error.to_string())?;
        for layer in 0..5 {
            let k_view = CudaF32View::new(
                self.k_outputs[layer]
                    .view(0, Q8_FEATURE_KV * 4)
                    .map_err(|error| error.to_string())?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                Q8_FEATURE_KV,
            )
            .map_err(|error| error.to_string())?;
            let v_view = CudaF32View::new(
                self.v_outputs[layer]
                    .view(0, Q8_FEATURE_KV * 4)
                    .map_err(|error| error.to_string())?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                Q8_FEATURE_KV,
            )
            .map_err(|error| error.to_string())?;
            CudaQ8_0F32MatvecArgsV1::bind(stream, self.k_weights[layer], normalized, k_view)
                .map_err(|error| error.to_string())?
                .submit_for_plan(stream)
                .map_err(|error| error.to_string())?;
            CudaQ8_0F32MatvecArgsV1::bind(stream, self.v_weights[layer], normalized, v_view)
                .map_err(|error| error.to_string())?
                .submit_for_plan(stream)
                .map_err(|error| error.to_string())?;
            for head in 0..8 {
                let offset = head * Q8_FEATURE_HEAD_DIM * 4;
                let raw = CudaF32View::new(
                    self.k_outputs[layer]
                        .view(offset, Q8_FEATURE_HEAD_DIM * 4)
                        .map_err(|error| error.to_string())?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    Q8_FEATURE_HEAD_DIM,
                )
                .map_err(|error| error.to_string())?;
                let output = CudaF32View::new(
                    self.k_normed_outputs[layer]
                        .view(offset, Q8_FEATURE_HEAD_DIM * 4)
                        .map_err(|error| error.to_string())?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    Q8_FEATURE_HEAD_DIM,
                )
                .map_err(|error| error.to_string())?;
                CudaRmsnormF32ArgsV1::bind(stream, raw, self.k_norm_weights[layer], output, 1.0e-6)
                    .map_err(|error| error.to_string())?
                    .submit_for_plan(stream)
                    .map_err(|error| error.to_string())?;
            }
            for head in 0..8 {
                let offset = head * Q8_FEATURE_HEAD_DIM * 4;
                let input = CudaF32View::new(
                    self.k_normed_outputs[layer]
                        .view(offset, Q8_FEATURE_HEAD_DIM * 4)
                        .map_err(|error| error.to_string())?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    Q8_FEATURE_HEAD_DIM,
                )
                .map_err(|error| error.to_string())?;
                let output = CudaF32View::new(
                    self.k_rope_outputs[layer]
                        .view(offset, Q8_FEATURE_HEAD_DIM * 4)
                        .map_err(|error| error.to_string())?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    Q8_FEATURE_HEAD_DIM,
                )
                .map_err(|error| error.to_string())?;
                CudaDflashRopeF32ArgsV1::bind(
                    stream,
                    input,
                    output,
                    Q8_FEATURE_HEAD_DIM,
                    Q8_FEATURE_ROPE_DIM,
                    position,
                    Q8_FEATURE_ROPE_THETA,
                )
                .map_err(|error| error.to_string())?
                .submit_for_plan(stream)
                .map_err(|error| error.to_string())?;
            }
            let cache_slot = position % Q8_FEATURE_CACHE_POSITIONS;
            let cache_row_bytes = Q8_FEATURE_KV * 4;
            let cache_offset = (layer * Q8_FEATURE_CACHE_POSITIONS + cache_slot)
                .checked_mul(cache_row_bytes)
                .ok_or_else(|| "Q8 feature-KV cache offset overflow".to_owned())?;
            let k_source = self.k_rope_outputs[layer]
                .view(0, cache_row_bytes)
                .map_err(|error| error.to_string())?;
            let v_source = self.v_outputs[layer]
                .view(0, cache_row_bytes)
                .map_err(|error| error.to_string())?;
            // SAFETY: cache and source owners live for the consumer/session;
            // all copies are ordered on the same stream.
            unsafe {
                self.k_cache
                    .copy_from_device_async_unchecked(stream, cache_offset, k_source)
                    .map_err(|error| error.to_string())?;
                self.v_cache
                    .copy_from_device_async_unchecked(stream, cache_offset, v_source)
                    .map_err(|error| error.to_string())?;
            }
            let query = CudaF32View::new(
                self.k_rope_outputs[layer]
                    .view(0, cache_row_bytes)
                    .map_err(|error| error.to_string())?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                Q8_FEATURE_KV,
            )
            .map_err(|error| error.to_string())?;
            let repeated_query = CudaF32View::new(
                self.query_repeat_outputs[layer]
                    .view(0, 4 * cache_row_bytes)
                    .map_err(|error| error.to_string())?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                4 * Q8_FEATURE_KV,
            )
            .map_err(|error| error.to_string())?;
            CudaGqaRepeatF32ArgsV1::bind(stream, query, repeated_query, Q8_FEATURE_HEAD_DIM, 4)
                .map_err(|error| error.to_string())?
                .submit_for_plan(stream)
                .map_err(|error| error.to_string())?;
            let key_cache = CudaF32View::new(
                self.k_cache
                    .view(
                        layer
                            .checked_mul(Q8_FEATURE_CACHE_POSITIONS)
                            .and_then(|value| value.checked_mul(cache_row_bytes))
                            .ok_or_else(|| {
                                "Q8 cache attention K layer offset overflow".to_owned()
                            })?,
                        Q8_FEATURE_CACHE_POSITIONS * cache_row_bytes,
                    )
                    .map_err(|error| error.to_string())?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                Q8_FEATURE_CACHE_POSITIONS * Q8_FEATURE_KV,
            )
            .map_err(|error| error.to_string())?;
            let value_cache = CudaF32View::new(
                self.v_cache
                    .view(
                        layer
                            .checked_mul(Q8_FEATURE_CACHE_POSITIONS)
                            .and_then(|value| value.checked_mul(cache_row_bytes))
                            .ok_or_else(|| {
                                "Q8 cache attention V layer offset overflow".to_owned()
                            })?,
                        Q8_FEATURE_CACHE_POSITIONS * cache_row_bytes,
                    )
                    .map_err(|error| error.to_string())?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                Q8_FEATURE_CACHE_POSITIONS * Q8_FEATURE_KV,
            )
            .map_err(|error| error.to_string())?;
            let attention_output = CudaF32View::new(
                self.cache_attention_outputs[layer]
                    .view(0, 4 * cache_row_bytes)
                    .map_err(|error| error.to_string())?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                4 * Q8_FEATURE_KV,
            )
            .map_err(|error| error.to_string())?;
            CudaDflashCacheAttentionF32ArgsV1::bind(
                stream,
                repeated_query,
                key_cache,
                value_cache,
                attention_output,
                Q8_FEATURE_CACHE_POSITIONS,
                cache_window.start,
                cache_window.end,
                1.0 / (Q8_FEATURE_HEAD_DIM as f32).sqrt(),
            )
            .map_err(|error| error.to_string())?
            .submit_for_plan(stream)
            .map_err(|error| error.to_string())?;
            self.cache_attention_submissions = self.cache_attention_submissions.saturating_add(1);
        }
        self.submissions = self.submissions.saturating_add(1);
        self.last_step = Some(step);
        self.last_position = Some(position);
        self.cache_valid_positions = cache_window.len();
        self.capture_mask = 0;
        self.poisoned = false;
        Ok(())
    }

    fn finish(&mut self, stream: &OwnedCudaStream) -> Result<Value, String> {
        if self.poisoned || self.capture_mask != 0 || self.cache_valid_positions == 0 {
            return Err("Q8 feature-KV no complete healthy cache window".into());
        }
        // D2H below is terminal diagnostic work, never part of per-token decode.
        let mut feature_bytes = vec![0_u8; Q8_FEATURE_INPUT * 4];
        self.feature_input
            .copy_to_host_sync(stream, &mut feature_bytes)
            .map_err(|error| error.to_string())?;
        let mut k_hashes = Vec::with_capacity(5);
        let mut v_hashes = Vec::with_capacity(5);
        let mut k_normed_hashes = Vec::with_capacity(5);
        let mut k_rope_hashes = Vec::with_capacity(5);
        let mut cache_k_hashes = Vec::with_capacity(5);
        let mut cache_v_hashes = Vec::with_capacity(5);
        let mut cache_attention_hashes = Vec::with_capacity(5);
        let mut cache_attention_ab = Vec::with_capacity(5);
        let mut cache_attention_finite = true;
        let mut k_ab = Vec::with_capacity(5);
        let mut v_ab = Vec::with_capacity(5);
        let mut k_normed_ab = Vec::with_capacity(5);
        let mut k_rope_ab = Vec::with_capacity(5);
        let mut cache_k_ab = Vec::with_capacity(5);
        let mut cache_v_ab = Vec::with_capacity(5);
        let mut rope_debug = Vec::with_capacity(1);
        let cache_slot = self
            .last_position
            .map(|position| position % Q8_FEATURE_CACHE_POSITIONS);
        let cache_row_bytes = Q8_FEATURE_KV * 4;
        let feature_values = bytes_to_f32(&feature_bytes)?;
        let oracle = encode_target_features_kv_q8(&self.artifact_path, &feature_values, 1.0e-6)
            .map_err(|error| format!("Q8 feature-KV CPU oracle: {error}"))?;
        let position = self
            .last_position
            .ok_or_else(|| "Q8 feature-KV missing last position".to_owned())?;
        for layer in 0..5 {
            let mut kb = vec![0_u8; Q8_FEATURE_KV * 4];
            let mut vb = vec![0_u8; Q8_FEATURE_KV * 4];
            let mut nb = vec![0_u8; Q8_FEATURE_KV * 4];
            let mut rb = vec![0_u8; Q8_FEATURE_KV * 4];
            self.k_outputs[layer]
                .copy_to_host_sync(stream, &mut kb)
                .map_err(|error| error.to_string())?;
            self.v_outputs[layer]
                .copy_to_host_sync(stream, &mut vb)
                .map_err(|error| error.to_string())?;
            self.k_normed_outputs[layer]
                .copy_to_host_sync(stream, &mut nb)
                .map_err(|error| error.to_string())?;
            self.k_rope_outputs[layer]
                .copy_to_host_sync(stream, &mut rb)
                .map_err(|error| error.to_string())?;
            let actual_k = bytes_to_f32(&kb)?;
            let actual_v = bytes_to_f32(&vb)?;
            let actual_k_normed = bytes_to_f32(&nb)?;
            let actual_k_rope = bytes_to_f32(&rb)?;
            let expected_k_rope =
                rotate_neox_f32(&oracle.k_normed[layer], position, Q8_FEATURE_ROPE_THETA);
            k_ab.push(numerical_ab(&actual_k, &oracle.k[layer]));
            v_ab.push(numerical_ab(&actual_v, &oracle.v[layer]));
            k_normed_ab.push(numerical_ab(&actual_k_normed, &oracle.k_normed[layer]));
            k_rope_ab.push(numerical_ab(&actual_k_rope, &expected_k_rope));
            if layer == 0 {
                rope_debug.push(json!({
                    "position": position,
                    "gpu_k_normed_prefix": prefix(&actual_k_normed),
                    "gpu_k_rope_prefix": prefix(&actual_k_rope),
                    "cpu_k_normed_prefix": prefix(&oracle.k_normed[layer]),
                    "cpu_k_rope_prefix": prefix(&expected_k_rope),
                }));
            }
            k_hashes.push(sha256_hex(&kb));
            v_hashes.push(sha256_hex(&vb));
            k_normed_hashes.push(sha256_hex(&nb));
            k_rope_hashes.push(sha256_hex(&rb));
            if let Some(slot) = cache_slot {
                let offset = (layer * Q8_FEATURE_CACHE_POSITIONS + slot)
                    .checked_mul(cache_row_bytes)
                    .ok_or_else(|| "Q8 feature-KV cache readback offset overflow".to_owned())?;
                let mut cache_k = vec![0_u8; cache_row_bytes];
                let mut cache_v = vec![0_u8; cache_row_bytes];
                self.k_cache
                    .copy_range_to_host_sync(stream, offset, &mut cache_k)
                    .map_err(|error| error.to_string())?;
                self.v_cache
                    .copy_range_to_host_sync(stream, offset, &mut cache_v)
                    .map_err(|error| error.to_string())?;
                cache_k_ab.push(numerical_ab(&bytes_to_f32(&cache_k)?, &expected_k_rope));
                cache_v_ab.push(numerical_ab(&bytes_to_f32(&cache_v)?, &oracle.v[layer]));
                cache_k_hashes.push(sha256_hex(&cache_k));
                cache_v_hashes.push(sha256_hex(&cache_v));
            }
            let mut attention_bytes = vec![0_u8; 4 * Q8_FEATURE_KV * 4];
            self.cache_attention_outputs[layer]
                .copy_to_host_sync(stream, &mut attention_bytes)
                .map_err(|error| error.to_string())?;
            let attention_values = bytes_to_f32(&attention_bytes)?;
            cache_attention_finite &= attention_values.iter().all(|value| value.is_finite());
            cache_attention_hashes.push(sha256_hex(&attention_bytes));
            let start = position + 1 - self.cache_valid_positions;
            let mut keys = Vec::with_capacity(self.cache_valid_positions * Q8_FEATURE_KV);
            let mut values = Vec::with_capacity(keys.capacity());
            // At most two contiguous copies per layer, only initialized slots.
            let mut p = start;
            while p <= position {
                let slot = p % Q8_FEATURE_CACHE_POSITIONS;
                let count = (position + 1 - p).min(Q8_FEATURE_CACHE_POSITIONS - slot);
                let offset = (layer * Q8_FEATURE_CACHE_POSITIONS + slot) * cache_row_bytes;
                let mut k = vec![0; count * cache_row_bytes];
                let mut v = vec![0; count * cache_row_bytes];
                self.k_cache
                    .copy_range_to_host_sync(stream, offset, &mut k)
                    .map_err(|e| e.to_string())?;
                self.v_cache
                    .copy_range_to_host_sync(stream, offset, &mut v)
                    .map_err(|e| e.to_string())?;
                keys.extend(bytes_to_f32(&k)?);
                values.extend(bytes_to_f32(&v)?);
                p += count;
            }
            let mut query_bytes = vec![0; 4 * cache_row_bytes];
            self.query_repeat_outputs[layer]
                .copy_to_host_sync(stream, &mut query_bytes)
                .map_err(|e| e.to_string())?;
            let expected_attention =
                cache_contract::attention_f64(&bytes_to_f32(&query_bytes)?, &keys, &values)?;
            cache_attention_ab.push(attention_ab(&attention_values, &expected_attention));
        }
        Ok(json!({
            "enabled": true,
            "component_consumed": self.submissions != 0,
            "submissions": self.submissions,
            "last_step": self.last_step,
            "last_position": self.last_position,
            "target_layer_ids": Q8_FEATURE_TARGET_LAYERS,
            "transport_path": "same-stream-target-hidden-D2D->resident-Q8-fc-encoder->five-layer-KV",
            "q8_full_file_device_resident": true,
            "q8_file_bytes": 2056415552_u64,
            "q8_file_sha256": self.artifact_sha256,
            "artifact_path": self.artifact_path,
            "target_feature_sha256": sha256_hex(&feature_bytes),
            "per_token_h2d_bytes": 0,
            "k_sha256": k_hashes,
            "v_sha256": v_hashes,
            "k_normed_sha256": k_normed_hashes,
            "k_rope_sha256": k_rope_hashes,
            "cache_k_sha256": cache_k_hashes,
            "cache_v_sha256": cache_v_hashes,
            "cache_attention_sha256": cache_attention_hashes,
            "cache_attention_numerical_ab": cache_attention_ab,
            "cache_attention_consumer": self.cache_attention_submissions != 0,
            "cache_attention_submissions": self.cache_attention_submissions,
            "cache_attention_output_finite": cache_attention_finite,
            "cache_attention_contract_pass": cache_attention_finite && cache_attention_ab.iter().all(|ab| ab["pass"] == true),
            "cache_attention_history": "only_contiguous_same_owner_submitted_rows; absolute_position_mod_capacity",
            "cache_prefill_history_populated": false,
            "cache_valid_positions": self.cache_valid_positions,
            "cache_visible_start": position + 1 - self.cache_valid_positions,
            "cache_visible_end": position + 1,
            "cache_attention_query_kind": "diagnostic_repeated_rotated_key_not_learned_draft_query",
            "cache_attention_oracle_kind": "f64_attention_over_actual_valid_cache_and_query_readback",
            "cache_attention_per_token_d2h_bytes": 0,
            "cache_attention_terminal_d2h_bytes": 5 * (2 * self.cache_valid_positions * Q8_FEATURE_KV + 8 * Q8_FEATURE_KV) * 4,
            "dflash_decoder_executed": false,
            "draft_semantic_parity": false,
            "cpu_oracle_executed": true,
            "cpu_oracle_source_sha256": oracle.source_sha256,
            "cpu_oracle_input_sha256": oracle.input_sha256,
            "k_numerical_ab": k_ab,
            "v_numerical_ab": v_ab,
            "k_normed_numerical_ab": k_normed_ab,
            "k_rope_numerical_ab": k_rope_ab,
            "cache_k_numerical_ab": cache_k_ab,
            "cache_v_numerical_ab": cache_v_ab,
            "rope_debug": rope_debug,
            "cache_layout": "layer_major_position_major_head_major_f32",
            "cache_capacity_positions": Q8_FEATURE_CACHE_POSITIONS,
            "cache_last_slot": cache_slot,
            "cache_device_bytes": 2 * 5 * Q8_FEATURE_CACHE_POSITIONS * Q8_FEATURE_KV * 4,
            "cache_write_component": true,
            "rope_applied": true,
            "rope_layout": "neox_half_split",
            "rope_theta": Q8_FEATURE_ROPE_THETA,
            "rope_rotary_dimensions": Q8_FEATURE_ROPE_DIM,
            "rope_required_before_cache": true,
            "claim_ceiling": "q8_feature_kv_ring_and_diagnostic_cache_attention_component_only; learned_draft_decoder_acceptance_semantics_product_tps_not_qualified"
        }))
    }

    /// Close the component owner without terminal CPU-oracle readback.
    ///
    /// The receipt-only decode path must be able to prove that the Q8 feature
    /// consumer was submitted and remained healthy while keeping its K/V
    /// workspaces device-resident.  The full `finish` method is intentionally
    /// diagnostic and copies every terminal workspace back to the host; that
    /// copy is not part of the product transport contract, so receipt-only
    /// runs use this metadata-only close instead.
    fn finish_transport_only(&self) -> Result<Value, String> {
        if self.poisoned || self.capture_mask != 0 || self.cache_valid_positions == 0 {
            return Err("Q8 feature-KV no complete healthy cache window".into());
        }
        let position = self
            .last_position
            .ok_or_else(|| "Q8 feature-KV terminal position missing".to_owned())?;
        Ok(json!({
            "enabled": true,
            "component_consumed": self.submissions != 0,
            "submissions": self.submissions,
            "last_step": self.last_step,
            "last_position": self.last_position,
            "target_layer_ids": Q8_FEATURE_TARGET_LAYERS,
            "transport_path": "same-stream-target-hidden-D2D->resident-Q8-fc-encoder->five-layer-KV",
            "q8_full_file_device_resident": true,
            "q8_file_bytes": 2056415552_u64,
            "q8_file_sha256": self.artifact_sha256,
            "artifact_path": self.artifact_path,
            "per_token_h2d_bytes": 0,
            "cache_attention_consumer": self.cache_attention_submissions != 0,
            "cache_attention_submissions": self.cache_attention_submissions,
            "cache_attention_contract_pass": Value::Null,
            "cache_attention_transport_contract": true,
            "cache_attention_history": "only_contiguous_same_owner_submitted_rows; absolute_position_mod_capacity",
            "cache_prefill_history_populated": false,
            "cache_valid_positions": self.cache_valid_positions,
            "cache_visible_start": position + 1 - self.cache_valid_positions,
            "cache_visible_end": position + 1,
            "cache_attention_per_token_d2h_bytes": 0,
            "cache_attention_terminal_d2h_bytes": 0,
            "dflash_decoder_executed": false,
            "draft_semantic_parity": false,
            "cpu_oracle_executed": false,
            "terminal_readback_skipped": true,
            "claim_ceiling": "q8_feature_kv_ring transport-only component; terminal CPU oracle omitted; learned draft decoder acceptance semantics product tps not qualified",
            "cache_layout": "layer_major_position_major_head_major_f32",
            "cache_capacity_positions": Q8_FEATURE_CACHE_POSITIONS,
            "cache_last_slot": position % Q8_FEATURE_CACHE_POSITIONS,
            "cache_device_bytes": 2 * 5 * Q8_FEATURE_CACHE_POSITIONS * Q8_FEATURE_KV * 4,
            "cache_write_component": true,
            "rope_applied": true,
            "rope_layout": "neox_half_split",
            "rope_theta": Q8_FEATURE_ROPE_THETA,
            "rope_rotary_dimensions": Q8_FEATURE_ROPE_DIM,
            "rope_required_before_cache": true
        }))
    }

    fn abort(&mut self, stream: &OwnedCudaStream) -> Result<(), String> {
        self.poisoned = true;
        stream.synchronize().map_err(|error| error.to_string())?;
        Ok(())
    }
}

impl Q8SelectorConsumer {
    fn bind(
        path: &Path,
        file: &[u8],
        owner: &OwnedCudaBuffer,
        device: i32,
    ) -> Result<Self, String> {
        let manifest = GgufManifest::inspect(
            path,
            GgufAuditOptions {
                hash_file: false,
                ..Default::default()
            },
        )
        .map_err(|error| format!("Q8 selector manifest: {error}"))?;
        let tensor = manifest
            .tensor(Q8_SELECTOR_TENSOR)
            .ok_or_else(|| "Q8 selector_hidden.weight missing".to_owned())?
            .clone();
        if tensor.ggml_type != GGML_TYPE_Q8_0
            || tensor.dimensions != vec![Q8_SELECTOR_COLS as u64, Q8_SELECTOR_ROWS as u64]
        {
            return Err("Q8 selector_hidden.weight must be Q8_0 [5120,256]".into());
        }
        if manifest.file_size != file.len() as u64 {
            return Err("Q8 GGUF extent changed after inspection".into());
        }
        let mut reader = manifest
            .open_tensor_reader(path)
            .map_err(|error| format!("Q8 selector tensor reader: {error}"))?;
        let payload = reader
            .read_tensor_bytes(Q8_SELECTOR_TENSOR)
            .map_err(|error| format!("Q8 selector tensor payload: {error}"))?;
        let tensor_offset = usize::try_from(tensor.absolute_offset)
            .map_err(|_| "Q8 selector tensor offset overflow".to_owned())?;
        let tensor_bytes = usize::try_from(tensor.byte_len)
            .map_err(|_| "Q8 selector tensor extent overflow".to_owned())?;
        if tensor_offset
            .checked_add(tensor_bytes)
            .is_none_or(|end| end > file.len())
            || tensor_bytes == 0
            || payload.len() != tensor_bytes
        {
            return Err("Q8 selector tensor payload extent".into());
        }
        let weight = CudaQ8_0WeightViewV1::new(
            owner
                .view(tensor_offset, tensor_bytes)
                .map_err(|error| format!("Q8 selector device view: {error}"))?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            Q8_SELECTOR_ROWS,
            Q8_SELECTOR_COLS,
        )
        .map_err(|error| format!("Q8 selector weight view: {error}"))?;
        let output = OwnedCudaBuffer::allocate(device, Q8_SELECTOR_ROWS * 4)
            .map_err(|error| format!("Q8 selector output allocation: {error}"))?;
        Ok(Self {
            weight,
            output,
            tensor,
            payload,
            file_bytes: file.len(),
            file_sha256: sha256_hex(file),
            tensor_offset,
            tensor_bytes,
            rows: Q8_SELECTOR_ROWS,
            cols: Q8_SELECTOR_COLS,
            submissions: 0,
            last_step: None,
            terminal_output: None,
            terminal: None,
        })
    }

    fn retire(&mut self, stream: &OwnedCudaStream) -> Result<(), String> {
        let Some((event, receipt)) = self.terminal.take() else {
            return Ok(());
        };
        loop {
            if stream
                .query_event_completion(&event, receipt)
                .map_err(|error| format!("Q8 selector completion: {error}"))?
                .is_some()
            {
                break;
            }
            std::hint::spin_loop();
        }
        Ok(())
    }

    fn consume_device(
        &mut self,
        step: usize,
        hidden_view: CudaF32View,
        stream: &OwnedCudaStream,
    ) -> Result<(), String> {
        if hidden_view.element_count() != self.cols {
            return Err("Q8 selector device hidden input geometry".into());
        }
        self.submit(step, hidden_view, stream)
    }

    fn submit(
        &mut self,
        step: usize,
        hidden_view: CudaF32View,
        stream: &OwnedCudaStream,
    ) -> Result<(), String> {
        self.retire(stream)?;
        let output = CudaF32View::new(
            self.output
                .view(0, self.rows * 4)
                .map_err(|error| format!("Q8 selector output view: {error}"))?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            self.rows,
        )
        .map_err(|error| format!("Q8 selector output geometry: {error}"))?;
        CudaQ8_0F32MatvecArgsV1::bind(stream, self.weight, hidden_view, output)
            .map_err(|error| format!("Q8 selector matvec bind: {error}"))?
            .submit_for_plan(stream)
            .map_err(|error| format!("Q8 selector matvec submit: {error}"))?;
        let terminal = OwnedCudaEvent::create(stream.device_ordinal())
            .map_err(|error| format!("Q8 selector terminal event: {error}"))?;
        let receipt = stream
            .submission_receipt()
            .map_err(|error| format!("Q8 selector submission receipt: {error}"))?;
        terminal
            .record_for_submission(stream, receipt)
            .map_err(|error| format!("Q8 selector terminal record: {error}"))?;
        self.terminal = Some((terminal, receipt));
        self.last_step = Some(step);
        self.submissions = self.submissions.saturating_add(1);
        Ok(())
    }

    fn finish(&mut self, stream: &OwnedCudaStream) -> Result<Value, String> {
        self.retire(stream)?;
        let mut bytes = vec![0_u8; self.rows * 4];
        self.output
            .copy_to_host_sync(stream, &mut bytes)
            .map_err(|error| format!("Q8 selector output readback: {error}"))?;
        let actual = bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().expect("Q8 output word")))
            .collect::<Vec<_>>();
        self.terminal_output = Some(actual);
        Ok(json!({
            "enabled": true,
            "component_consumed": self.submissions != 0,
            "submissions": self.submissions,
            "last_step": self.last_step,
            "transport_path": "one-time-GGUF-H2D->resident-HBM->Q8_0-f32-matvec",
            "q8_full_file_device_resident": true,
            "q8_file_bytes": self.file_bytes,
            "q8_file_sha256": self.file_sha256,
            "selector_tensor": Q8_SELECTOR_TENSOR,
            "selector_tensor_offset": self.tensor_offset,
            "selector_tensor_bytes": self.tensor_bytes,
            "selector_rows": self.rows,
            "selector_cols": self.cols,
            "initial_h2d_bytes": self.file_bytes,
            "per_token_h2d_bytes": 0,
            "output_sha256": sha256_hex(&bytes),
            "numerical_ab": Value::Null,
            "cpu_oracle_executed": false,
            "claim_ceiling": "component_only_target_hidden_projection_consumer; full_dflash_backbone_not_connected"
        }))
    }

    fn finish_transport_only(&self) -> Value {
        json!({
            "enabled": true,
            "component_consumed": self.submissions != 0,
            "submissions": self.submissions,
            "last_step": self.last_step,
            "transport_path": "one-time-GGUF-H2D->resident-HBM->Q8_0-f32-matvec",
            "q8_full_file_device_resident": true,
            "q8_file_bytes": self.file_bytes,
            "q8_file_sha256": self.file_sha256,
            "selector_tensor": Q8_SELECTOR_TENSOR,
            "selector_tensor_offset": self.tensor_offset,
            "selector_tensor_bytes": self.tensor_bytes,
            "selector_rows": self.rows,
            "selector_cols": self.cols,
            "initial_h2d_bytes": self.file_bytes,
            "per_token_h2d_bytes": 0,
            "output_sha256": Value::Null,
            "numerical_ab": Value::Null,
            "cpu_oracle_executed": false,
            "terminal_readback_skipped": true,
            "claim_ceiling": "component_only_target_hidden_projection transport-only consumer; terminal CPU oracle omitted; full dflash backbone not connected"
        })
    }

    fn validate_terminal(&self, hidden: &[f32]) -> Result<Value, String> {
        if hidden.len() != self.cols {
            return Err("Q8 selector terminal hidden geometry".into());
        }
        let actual = self
            .terminal_output
            .as_ref()
            .ok_or_else(|| "Q8 selector terminal output missing".to_owned())?;
        let expected = quantized_matvec(&self.tensor, &self.payload, hidden)
            .map_err(|error| format!("Q8 selector CPU oracle: {error}"))?;
        Ok(numerical_ab(actual, &expected))
    }

    fn abort(&mut self, stream: &OwnedCudaStream) -> Result<(), String> {
        stream
            .synchronize()
            .map_err(|error| format!("Q8 selector abort synchronize: {error}"))?;
        self.terminal = None;
        Ok(())
    }
}

struct VisualPayload {
    bytes: Vec<u8>,
    file_bytes: usize,
}

impl ResidentAuxiliaryOwners {
    pub fn load(
        q8_dflash_path: Option<&Path>,
        vision_path: Option<&Path>,
        device: i32,
        stream: &OwnedCudaStream,
    ) -> Result<Option<Self>, String> {
        if q8_dflash_path.is_none() && vision_path.is_none() {
            return Ok(None);
        }

        let mut q8_owner = None;
        let mut q8_selector_consumer = None;
        let mut q8_feature_kv_consumer = None;
        let mut vision_owner = None;
        let mut q8_report = None;
        let mut vision_report = None;
        let mut h2d_bytes = 0usize;
        let started = Instant::now();

        if let Some(path) = q8_dflash_path {
            let bytes = read_stable_regular_file(path, "Q8 DFlash artifact")?;
            if bytes.is_empty() {
                return Err("Q8 DFlash artifact is empty".into());
            }
            let digest = sha256_hex(&bytes);
            let owner = OwnedCudaBuffer::allocate(device, bytes.len())
                .map_err(|error| format!("Q8 DFlash owner allocation: {error}"))?;
            owner
                .copy_from_host_sync(stream, &bytes)
                .map_err(|error| format!("Q8 DFlash H2D: {error}"))?;
            q8_selector_consumer = Some(Q8SelectorConsumer::bind(path, &bytes, &owner, device)?);
            q8_feature_kv_consumer = Some(Q8FeatureKvConsumer::bind(path, &bytes, &owner, device)?);
            h2d_bytes = h2d_bytes.saturating_add(bytes.len());
            q8_report = Some(ResidentAuxiliaryArtifactReport {
                path: path.to_owned(),
                file_bytes: bytes.len(),
                resident_device_bytes: owner.len_bytes(),
                payload_sha256: digest,
                device_resident: owner.len_bytes() == bytes.len(),
            });
            q8_owner = Some(owner);
        }

        if let Some(path) = vision_path {
            let visual = read_visual_payload(path)?;
            if visual.bytes.is_empty() {
                return Err("Vision payload is empty".into());
            }
            let digest = sha256_hex(&visual.bytes);
            let owner = OwnedCudaBuffer::allocate(device, visual.bytes.len())
                .map_err(|error| format!("Vision owner allocation: {error}"))?;
            owner
                .copy_from_host_sync(stream, &visual.bytes)
                .map_err(|error| format!("Vision H2D: {error}"))?;
            h2d_bytes = h2d_bytes.saturating_add(visual.bytes.len());
            vision_report = Some(ResidentAuxiliaryArtifactReport {
                path: path.to_owned(),
                file_bytes: visual.file_bytes,
                resident_device_bytes: owner.len_bytes(),
                payload_sha256: digest,
                device_resident: owner.len_bytes() == visual.bytes.len(),
            });
            vision_owner = Some(owner);
        }

        stream
            .synchronize()
            .map_err(|error| format!("resident auxiliary stream synchronize: {error}"))?;
        let owner_bytes = q8_owner
            .as_ref()
            .map_or(0, OwnedCudaBuffer::len_bytes)
            .saturating_add(vision_owner.as_ref().map_or(0, OwnedCudaBuffer::len_bytes));
        let owner_count =
            if q8_owner.is_some() { 1 } else { 0 } + if vision_owner.is_some() { 1 } else { 0 };
        Ok(Some(Self {
            _q8_dflash: q8_owner,
            _vision: vision_owner,
            q8_selector_consumer,
            q8_feature_kv_consumer,
            report: ResidentAuxiliaryReport {
                q8_dflash: q8_report,
                vision: vision_report,
                owner_bytes,
                h2d_bytes,
                owner_count,
                upload_wall_ms: started.elapsed().as_secs_f64() * 1_000.0,
                q8_selector_consumer: None,
                q8_feature_kv_consumer: None,
                cleanup_zero: false,
                decode_consumer_executed: false,
                vision_forward_executed: false,
                semantic_parity: false,
                product_ready: false,
            },
        }))
    }

    pub fn report(&self) -> ResidentAuxiliaryReport {
        self.report.clone()
    }

    pub fn owner_bytes(&self) -> usize {
        self.report.owner_bytes
    }

    pub fn owner_count(&self) -> usize {
        self.report.owner_count
    }

    pub(crate) fn consume_q8_device(
        &mut self,
        step: usize,
        hidden_view: CudaF32View,
        stream: &OwnedCudaStream,
    ) -> Result<(), String> {
        if let Some(consumer) = self.q8_selector_consumer.as_mut() {
            if let Err(error) = consumer.consume_device(step, hidden_view, stream) {
                return self.drain_consumer_error(stream, error);
            }
            self.report.decode_consumer_executed = true;
        }
        Ok(())
    }

    pub(crate) fn capture_q8_feature_layer(
        &mut self,
        layer: usize,
        hidden_view: CudaF32View,
        stream: &OwnedCudaStream,
    ) -> Result<(), String> {
        if let Some(consumer) = self.q8_feature_kv_consumer.as_mut() {
            if let Err(error) = consumer.capture_layer(layer, hidden_view, stream) {
                return self.drain_consumer_error(stream, error);
            }
        }
        Ok(())
    }

    pub(crate) fn consume_q8_feature_device(
        &mut self,
        step: usize,
        position: usize,
        stream: &OwnedCudaStream,
    ) -> Result<(), String> {
        if let Some(consumer) = self.q8_feature_kv_consumer.as_mut() {
            if let Err(error) = consumer.submit(step, position, stream) {
                return self.drain_consumer_error(stream, error);
            }
            self.report.decode_consumer_executed = true;
        }
        Ok(())
    }

    pub(crate) fn finish_consumers(
        &mut self,
        stream: &OwnedCudaStream,
        terminal_diagnostics: bool,
    ) -> Result<(), String> {
        self.finish_consumers_with_policy(stream, terminal_diagnostics, terminal_diagnostics)
    }

    /// Finish selector and feature-KV owners with independent terminal policies.
    pub(crate) fn finish_consumers_with_policy(
        &mut self,
        stream: &OwnedCudaStream,
        selector_terminal_diagnostics: bool,
        feature_terminal_diagnostics: bool,
    ) -> Result<(), String> {
        if let Some(consumer) = self.q8_selector_consumer.as_mut() {
            let report = if selector_terminal_diagnostics {
                consumer.finish(stream)?
            } else {
                consumer.finish_transport_only()
            };
            self.report.q8_selector_consumer = Some(report);
        }
        if let Some(consumer) = self.q8_feature_kv_consumer.as_mut() {
            if consumer.submissions != 0 {
                let report = if feature_terminal_diagnostics {
                    consumer.finish(stream)?
                } else {
                    consumer.finish_transport_only()?
                };
                self.report.q8_feature_kv_consumer = Some(report);
            }
        }
        Ok(())
    }

    pub(crate) fn validate_q8_terminal(&mut self, hidden: &[f32]) -> Result<(), String> {
        if let Some(consumer) = self.q8_selector_consumer.as_ref() {
            let numerical = consumer.validate_terminal(hidden)?;
            if let Some(report) = self.report.q8_selector_consumer.as_mut() {
                report["numerical_ab"] = numerical;
                report["cpu_oracle_executed"] = Value::Bool(true);
            }
        }
        Ok(())
    }

    pub(crate) fn abort_consumers(&mut self, stream: &OwnedCudaStream) -> Result<(), String> {
        // Attempt both drains even if one fails. All owners still live here.
        let feature = self
            .q8_feature_kv_consumer
            .as_mut()
            .map_or(Ok(()), |c| c.abort(stream));
        let selector = self
            .q8_selector_consumer
            .as_mut()
            .map_or(Ok(()), |c| c.abort(stream));
        feature.and(selector)
    }

    fn drain_consumer_error<T>(
        &mut self,
        stream: &OwnedCudaStream,
        error: String,
    ) -> Result<T, String> {
        match self.abort_consumers(stream) {
            Ok(()) => Err(format!(
                "{error}; auxiliary owner poisoned and stream drained"
            )),
            Err(drain) => Err(format!("{error}; auxiliary drain failed: {drain}")),
        }
    }
}

fn read_stable_regular_file(path: &Path, label: &str) -> Result<Vec<u8>, String> {
    let before =
        fs::symlink_metadata(path).map_err(|error| format!("{label} metadata: {error}"))?;
    if !before.is_file() || before.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    let bytes = fs::read(path).map_err(|error| format!("{label} read: {error}"))?;
    let after = fs::symlink_metadata(path).map_err(|error| format!("{label} recheck: {error}"))?;
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        return Err(format!("{label} changed during read"));
    }
    Ok(bytes)
}

/// CPU-only identity of the exact packed visual bytes used by the resident loader.
/// Drops the temporary host payload before returning; does not allocate on CUDA.
/// This identifies weight residency only, not a Vision forward implementation.
pub fn inspect_qwen35_vision_payload(path: &Path) -> Result<(usize, String), String> {
    let visual = read_visual_payload(path)?;
    if visual.bytes.is_empty() {
        return Err("Vision payload is empty".into());
    }
    Ok((visual.bytes.len(), sha256_hex(&visual.bytes)))
}

fn read_visual_payload(path: &Path) -> Result<VisualPayload, String> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("Vision metadata: {error}"))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err("Vision safetensors must be a regular non-symlink file".into());
    }
    let file_bytes =
        usize::try_from(metadata.len()).map_err(|_| "Vision file too large".to_owned())?;
    let mut file = fs::File::open(path).map_err(|error| format!("Vision open: {error}"))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|error| format!("Vision header length: {error}"))?;
    let header_bytes = usize::try_from(u64::from_le_bytes(len_bytes))
        .map_err(|_| "Vision header length overflow".to_owned())?;
    if header_bytes == 0 || header_bytes > MAX_SAFETENSORS_HEADER_BYTES {
        return Err("Vision safetensors header exceeds bound".into());
    }
    let mut header = vec![0u8; header_bytes];
    file.read_exact(&mut header)
        .map_err(|error| format!("Vision header: {error}"))?;
    let root: serde_json::Value =
        serde_json::from_slice(&header).map_err(|error| format!("Vision header JSON: {error}"))?;
    let root = root
        .as_object()
        .ok_or_else(|| "Vision safetensors header must be an object".to_owned())?;
    let payload_offset = 8u64
        .checked_add(u64::try_from(header_bytes).map_err(|_| "Vision offset overflow".to_owned())?)
        .ok_or_else(|| "Vision payload offset overflow".to_owned())?;
    let mut descriptors = Vec::<(u64, u64)>::new();
    for (name, descriptor) in root {
        if !name.starts_with(VISUAL_PREFIX) {
            continue;
        }
        let descriptor = descriptor
            .as_object()
            .ok_or_else(|| "Vision tensor descriptor must be an object".to_owned())?;
        let offsets = descriptor
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "Vision tensor data_offsets missing".to_owned())?;
        if offsets.len() != 2 {
            return Err("Vision tensor data_offsets must have two entries".into());
        }
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| "Vision tensor start offset invalid".to_owned())?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| "Vision tensor end offset invalid".to_owned())?;
        if end <= start {
            return Err("Vision tensor extent invalid".into());
        }
        let absolute_end = payload_offset
            .checked_add(end)
            .ok_or_else(|| "Vision tensor extent overflow".to_owned())?;
        if absolute_end > metadata.len() {
            return Err("Vision tensor exceeds file extent".into());
        }
        descriptors.push((start, end));
    }
    descriptors.sort_unstable_by_key(|&(start, _)| start);
    let mut payload = Vec::new();
    for (start, end) in descriptors {
        let length =
            usize::try_from(end - start).map_err(|_| "Vision tensor length overflow".to_owned())?;
        file.seek(SeekFrom::Start(payload_offset + start))
            .map_err(|error| format!("Vision tensor seek: {error}"))?;
        let old_len = payload.len();
        payload.resize(old_len.saturating_add(length), 0);
        file.read_exact(&mut payload[old_len..])
            .map_err(|error| format!("Vision tensor read: {error}"))?;
    }
    let after = fs::symlink_metadata(path).map_err(|error| format!("Vision recheck: {error}"))?;
    if after.len() != metadata.len() || after.modified().ok() != metadata.modified().ok() {
        return Err("Vision safetensors changed during read".into());
    }
    Ok(VisualPayload {
        bytes: payload,
        file_bytes,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod vision_identity_tests {
    use super::*;

    #[test]
    fn cpu_vision_identity_matches_offset_ordered_loader_payload() {
        let path = std::env::temp_dir().join(format!(
            "augur-vision-identity-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let header = br#"{"model.visual.z":{"data_offsets":[0,2]},"language":{"data_offsets":[2,6]},"model.visual.a":{"data_offsets":[6,9]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(b"abSKIPcde");
        fs::write(&path, bytes).unwrap();
        let result = inspect_qwen35_vision_payload(&path);
        fs::remove_file(&path).unwrap();
        assert_eq!(result.unwrap(), (5, sha256_hex(b"abcde")));
    }

    #[test]
    fn cpu_vision_identity_rejects_missing_file() {
        let path = std::env::temp_dir().join(format!(
            "augur-vision-missing-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert!(inspect_qwen35_vision_payload(&path).is_err());
    }
}
