//! CPU-only real-weight DFlash2 block producer.
//!
//! This is the first reusable producer boundary beyond the one-token example.
//! It consumes an identity-bound F16 DFlash backbone, same-request target
//! feature rows, and an explicit `[anchor, mask x 7]` noise embedding block.
//! It executes all five draft layers with FP64 accumulation for dense matvecs,
//! the block-local convolution oracle, and non-causal block GQA attention.
//!
//! It deliberately stops before candidate logits, selector path publication,
//! target verification, acceptance, or rollback.  The output is therefore a
//! numerical component receipt, not a speculative-decoding result.

use crate::{
    apply_block_local_grouped_conv, apply_block_noncausal_gqa_attention, DFlashBackboneError,
    DFlashBackboneManifest, DFlashBlockWindow, DFlashDraftBlockInput, DFlashDraftContractError,
    DFlashTargetFeatureCaptureError, DFlashTargetFeatureCaptureReceipt,
};
use augur_model_qwen35::DFlashConfig;
use half::f16;
use sha2::{Digest, Sha256};
use std::fmt;

pub const DFLASH_BLOCK_PRODUCER_SCHEMA: &str = "qwen35-dflash-five-layer-block-producer-cpu-v4";
pub const DFLASH_BLOCK_INJECTED_ATTENTION_SCHEMA: &str =
    "qwen35-dflash-five-layer-block-injected-attention-v2";
const HIDDEN: usize = 5_120;
const INTERMEDIATE: usize = 17_408;
const Q_HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const GROUP_SIZE: usize = 16;
const KERNEL_SIZE: usize = 2;
const RMS_EPS: f32 = 1.0e-6;
const TARGET_FEATURE_ROW: usize = 5 * HIDDEN;

#[derive(Clone, Debug)]
pub struct DFlashBlockProducerInput<'a> {
    pub seed: &'a DFlashDraftBlockInput,
    pub target_features: &'a [f32],
    /// Optional identity receipt for the target feature rows. `None` is kept
    /// only for legacy/synthetic component fixtures; a real target handoff
    /// should always provide the same-owner receipt.
    pub target_capture: Option<&'a DFlashTargetFeatureCaptureReceipt>,
    pub context_start_position: usize,
    pub noise_embedding: &'a [f32],
    pub sliding_window: usize,
}

/// Borrowed, real-weight intermediate boundary for component attention A/B.
/// Q/K already include head normalization and absolute-position RoPE; K/V
/// are dense `[target context || eight draft rows]`, not a circular KV cache.
/// An executor must complete before returning its owned F32 output. This is
/// a diagnostic numerical seam, not a token/acceptance publication capability.
#[derive(Clone, Copy, Debug)]
pub struct DFlashBlockAttentionInput<'a> {
    pub config: &'a DFlashConfig,
    pub layer: usize,
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub context_len: usize,
    pub context_start_position: usize,
    pub block_start_position: usize,
    pub sliding_window: usize,
}

impl DFlashBlockAttentionInput<'_> {
    pub fn cpu_reference(self) -> Result<Vec<f32>, DFlashBlockProducerError> {
        apply_block_noncausal_gqa_attention(
            self.config,
            self.query,
            self.key,
            self.value,
            self.context_len,
            DFlashBlockWindow {
                context_start_position: self.context_start_position,
                block_start_position: self.block_start_position,
                sliding_window: self.sliding_window,
            },
        )
        .map_err(|error| invalid(error.to_string()))
    }
}

#[derive(Clone, Debug)]
pub struct DFlashBlockProducerOutput {
    pub schema: String,
    pub request_generation: u64,
    pub absolute_position_start: usize,
    pub context_len: usize,
    pub hidden: Vec<f32>,
    pub selector_hidden: Vec<f32>,
    pub target_features_sha256: [u8; 32],
    pub noise_embedding_sha256: [u8; 32],
    pub hidden_sha256: [u8; 32],
    pub selector_hidden_sha256: [u8; 32],
    pub layer_stage_sha256: Vec<(String, [u8; 32])>,
    /// Diagnostic CPU oracle checkpoints; never token-publication authority.
    pub layer_hidden: Vec<Vec<f32>>,
    pub block_local_zero_pad: bool,
    pub candidate_logits_produced: bool,
    pub acceptance_executed: bool,
}

#[derive(Debug)]
pub enum DFlashBlockProducerError {
    Invalid(String),
    Backbone(DFlashBackboneError),
    Contract(DFlashDraftContractError),
    TargetCapture(DFlashTargetFeatureCaptureError),
}

impl fmt::Display for DFlashBlockProducerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid DFlash block producer: {message}"),
            Self::Backbone(error) => error.fmt(f),
            Self::Contract(error) => error.fmt(f),
            Self::TargetCapture(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for DFlashBlockProducerError {}

impl From<DFlashBackboneError> for DFlashBlockProducerError {
    fn from(error: DFlashBackboneError) -> Self {
        Self::Backbone(error)
    }
}

impl From<DFlashDraftContractError> for DFlashBlockProducerError {
    fn from(error: DFlashDraftContractError) -> Self {
        Self::Contract(error)
    }
}

impl From<DFlashTargetFeatureCaptureError> for DFlashBlockProducerError {
    fn from(error: DFlashTargetFeatureCaptureError) -> Self {
        Self::TargetCapture(error)
    }
}

/// Execute the complete five-layer CPU block producer against the real F16
/// DFlash artifact.  `target_features` is `[context_len, 25600]` and
/// `noise_embedding` is `[8, 5120]` for the exact checked profile.
pub fn produce_dflash_block_cpu(
    manifest: &DFlashBackboneManifest,
    input: DFlashBlockProducerInput<'_>,
) -> Result<DFlashBlockProducerOutput, DFlashBlockProducerError> {
    let mut output =
        produce_dflash_block_with_attention(manifest, input, |input| input.cpu_reference())?;
    output.schema = DFLASH_BLOCK_PRODUCER_SCHEMA.into();
    Ok(output)
}

/// Execute the F16/F64 CPU producer with a replaceable attention component.
/// Each returned attention row is consumed by the real output projection,
/// convolution, residual, FFN and subsequent layers. All other arithmetic is
/// still CPU. The distinct schema prevents labeling this a pure CPU oracle.
pub fn produce_dflash_block_with_attention(
    manifest: &DFlashBackboneManifest,
    input: DFlashBlockProducerInput<'_>,
    attention_executor: impl FnMut(
        DFlashBlockAttentionInput<'_>,
    ) -> Result<Vec<f32>, DFlashBlockProducerError>,
) -> Result<DFlashBlockProducerOutput, DFlashBlockProducerError> {
    if manifest.config().dtype != "float16" {
        return Err(invalid(
            "F16 block producer cannot reinterpret BF16 weights",
        ));
    }
    produce_dflash_block_from_weights(DFlashBlockWeights::F16(manifest), input, attention_executor)
}

#[derive(Clone, Copy)]
pub(crate) enum DFlashBlockWeights<'a> {
    F16(&'a DFlashBackboneManifest),
    Q8(&'a crate::DFlashQ8Backbone),
}
impl DFlashBlockWeights<'_> {
    fn config(&self) -> &DFlashConfig {
        match self {
            Self::F16(m) => m.config(),
            Self::Q8(m) => m.config(),
        }
    }
}

pub(crate) fn produce_dflash_block_from_weights(
    manifest: DFlashBlockWeights<'_>,
    input: DFlashBlockProducerInput<'_>,
    mut attention_executor: impl FnMut(
        DFlashBlockAttentionInput<'_>,
    ) -> Result<Vec<f32>, DFlashBlockProducerError>,
) -> Result<DFlashBlockProducerOutput, DFlashBlockProducerError> {
    let config = manifest.config();
    let target_digest = validate_dflash_block_input(config, &input)?;

    let mut stages = Vec::new();
    let target_hidden = encode_target_context(manifest, input.target_features)?;
    stages.push((
        "target_context_fc_hidden_norm".into(),
        sha256_f32(&target_hidden),
    ));
    let mut layer_hidden = Vec::new();
    let mut hidden = input.noise_embedding.to_vec();
    for layer in 0..config.num_hidden_layers {
        hidden = execute_layer(
            manifest,
            &target_hidden,
            &hidden,
            DFlashBlockWindow {
                context_start_position: input.context_start_position,
                block_start_position: input.seed.absolute_position_start,
                sliding_window: input.sliding_window,
            },
            layer,
            &mut stages,
            &mut attention_executor,
        )?;
        layer_hidden.push(hidden.clone());
    }
    hidden = rms_norm_rows(
        &hidden,
        config.block_size,
        HIDDEN,
        &read_vector(manifest, "norm.weight", HIDDEN)?,
    )?;
    stages.push(("backbone_final_norm".into(), sha256_f32(&hidden)));
    let selector_hidden = matvec_batch(
        manifest,
        "candidate_selector.hidden_projection.weight",
        config.selector_rank,
        HIDDEN,
        &hidden,
        config.block_size,
    )?;
    stages.push((
        "candidate_selector_hidden_projection".into(),
        sha256_f32(&selector_hidden),
    ));
    Ok(DFlashBlockProducerOutput {
        schema: DFLASH_BLOCK_INJECTED_ATTENTION_SCHEMA.into(),
        request_generation: input.seed.request_generation,
        absolute_position_start: input.seed.absolute_position_start,
        context_len: input.context_len(),
        hidden_sha256: sha256_f32(&hidden),
        selector_hidden_sha256: sha256_f32(&selector_hidden),
        target_features_sha256: target_digest,
        noise_embedding_sha256: sha256_f32(input.noise_embedding),
        hidden,
        selector_hidden,
        layer_stage_sha256: stages,
        layer_hidden,
        block_local_zero_pad: true,
        candidate_logits_produced: false,
        acceptance_executed: false,
    })
}

pub(crate) fn validate_dflash_block_input(
    config: &DFlashConfig,
    input: &DFlashBlockProducerInput<'_>,
) -> Result<[u8; 32], DFlashBlockProducerError> {
    config
        .validate_exact_profile()
        .map_err(|error| invalid(error.to_string()))?;
    input.seed.validate_initial_seed(config)?;
    if input.sliding_window == 0 {
        return Err(invalid("sliding window must be nonzero"));
    }
    if input
        .context_start_position
        .checked_add(input.context_len())
        .ok_or_else(|| invalid("context position overflow"))?
        != input.seed.absolute_position_start
    {
        return Err(invalid("context range does not end at block start"));
    }
    if input.target_features.len() != input.context_len() * TARGET_FEATURE_ROW {
        return Err(invalid("target feature row geometry"));
    }
    if input.noise_embedding.len() != config.block_size * HIDDEN {
        return Err(invalid("noise embedding block geometry"));
    }
    if input
        .target_features
        .iter()
        .chain(input.noise_embedding)
        .any(|value| !value.is_finite())
    {
        return Err(invalid("producer inputs must be finite"));
    }
    let target_digest = sha256_f32(input.target_features);
    if target_digest != input.seed.target_feature_sha256 {
        return Err(invalid("target feature digest differs from seed"));
    }
    if let Some(capture) = input.target_capture {
        capture.validate_for_block(
            config,
            input.seed,
            input.target_features.len(),
            target_digest,
        )?;
    }

    Ok(target_digest)
}

impl<'a> DFlashBlockProducerInput<'a> {
    fn context_len(&self) -> usize {
        self.target_features.len() / TARGET_FEATURE_ROW
    }
}

fn execute_layer(
    manifest: DFlashBlockWeights<'_>,
    target_hidden: &[f32],
    hidden: &[f32],
    window: DFlashBlockWindow,
    layer: usize,
    stages: &mut Vec<(String, [u8; 32])>,
    attention_executor: &mut impl FnMut(
        DFlashBlockAttentionInput<'_>,
    ) -> Result<Vec<f32>, DFlashBlockProducerError>,
) -> Result<Vec<f32>, DFlashBlockProducerError> {
    let config = manifest.config();
    let DFlashBlockWindow {
        context_start_position,
        block_start_position,
        sliding_window,
    } = window;
    let prefix = format!("layers.{layer}");
    let input_norm = rms_norm_rows(
        hidden,
        config.block_size,
        HIDDEN,
        &read_vector(
            manifest,
            &format!("{prefix}.input_layernorm.weight"),
            HIDDEN,
        )?,
    )?;
    let dynamic_rows = 2 * KERNEL_SIZE * (HIDDEN / GROUP_SIZE);
    let attention_dynamic = matvec_batch(
        manifest,
        &format!("{prefix}.attention_conv.kernel_projection.weight"),
        dynamic_rows,
        HIDDEN,
        &input_norm,
        config.block_size,
    )?;
    let attention_conv_in = apply_block_local_grouped_conv(
        config,
        &input_norm,
        &attention_dynamic,
        &read_tensor_f16(
            manifest,
            &format!("{prefix}.attention_conv.base_kernel"),
            &[2, KERNEL_SIZE, HIDDEN],
        )?,
        0,
    )
    .map_err(|error| invalid(error.to_string()))?;
    let mut q = matvec_batch(
        manifest,
        &format!("{prefix}.self_attn.q_proj.weight"),
        Q_HEADS * HEAD_DIM,
        HIDDEN,
        &attention_conv_in,
        config.block_size,
    )?;
    let mut k_context = matvec_batch(
        manifest,
        &format!("{prefix}.self_attn.k_proj.weight"),
        KV_HEADS * HEAD_DIM,
        HIDDEN,
        target_hidden,
        target_hidden.len() / HIDDEN,
    )?;
    let mut k_noise = matvec_batch(
        manifest,
        &format!("{prefix}.self_attn.k_proj.weight"),
        KV_HEADS * HEAD_DIM,
        HIDDEN,
        &attention_conv_in,
        config.block_size,
    )?;
    let v_context = matvec_batch(
        manifest,
        &format!("{prefix}.self_attn.v_proj.weight"),
        KV_HEADS * HEAD_DIM,
        HIDDEN,
        target_hidden,
        target_hidden.len() / HIDDEN,
    )?;
    let v_noise = matvec_batch(
        manifest,
        &format!("{prefix}.self_attn.v_proj.weight"),
        KV_HEADS * HEAD_DIM,
        HIDDEN,
        &attention_conv_in,
        config.block_size,
    )?;
    let q_norm = read_vector(
        manifest,
        &format!("{prefix}.self_attn.q_norm.weight"),
        HEAD_DIM,
    )?;
    let k_norm = read_vector(
        manifest,
        &format!("{prefix}.self_attn.k_norm.weight"),
        HEAD_DIM,
    )?;
    normalize_heads_rope(
        &mut q,
        config.block_size,
        Q_HEADS,
        &q_norm,
        block_start_position,
    )?;
    normalize_heads_rope(
        &mut k_context,
        target_hidden.len() / HIDDEN,
        KV_HEADS,
        &k_norm,
        context_start_position,
    )?;
    normalize_heads_rope(
        &mut k_noise,
        config.block_size,
        KV_HEADS,
        &k_norm,
        block_start_position,
    )?;
    let context_len = target_hidden.len() / HIDDEN;
    let mut k = Vec::with_capacity((context_len + config.block_size) * KV_HEADS * HEAD_DIM);
    let mut v = Vec::with_capacity(k.capacity());
    k.extend_from_slice(&k_context);
    k.extend_from_slice(&k_noise);
    v.extend_from_slice(&v_context);
    v.extend_from_slice(&v_noise);
    let attention = attention_executor(DFlashBlockAttentionInput {
        config,
        layer,
        query: &q,
        key: &k,
        value: &v,
        context_len,
        context_start_position,
        block_start_position,
        sliding_window,
    })?;
    validate_attention_output(&attention, config.block_size * Q_HEADS * HEAD_DIM)?;
    let attention_projected = matvec_batch(
        manifest,
        &format!("{prefix}.self_attn.o_proj.weight"),
        HIDDEN,
        Q_HEADS * HEAD_DIM,
        &attention,
        config.block_size,
    )?;
    let attention_conv_out = apply_block_local_grouped_conv(
        config,
        &attention_projected,
        &attention_dynamic,
        &read_tensor_f16(
            manifest,
            &format!("{prefix}.attention_conv.base_kernel"),
            &[2, KERNEL_SIZE, HIDDEN],
        )?,
        1,
    )
    .map_err(|error| invalid(error.to_string()))?;
    let ffn_input = add_rows(hidden, &attention_conv_out, config.block_size, HIDDEN)?;
    let ffn_norm = rms_norm_rows(
        &ffn_input,
        config.block_size,
        HIDDEN,
        &read_vector(
            manifest,
            &format!("{prefix}.post_attention_layernorm.weight"),
            HIDDEN,
        )?,
    )?;
    let mlp_dynamic = matvec_batch(
        manifest,
        &format!("{prefix}.mlp_conv.kernel_projection.weight"),
        dynamic_rows,
        HIDDEN,
        &ffn_norm,
        config.block_size,
    )?;
    let mlp_conv_in = apply_block_local_grouped_conv(
        config,
        &ffn_norm,
        &mlp_dynamic,
        &read_tensor_f16(
            manifest,
            &format!("{prefix}.mlp_conv.base_kernel"),
            &[2, KERNEL_SIZE, HIDDEN],
        )?,
        0,
    )
    .map_err(|error| invalid(error.to_string()))?;
    let gate = matvec_batch(
        manifest,
        &format!("{prefix}.mlp.gate_proj.weight"),
        INTERMEDIATE,
        HIDDEN,
        &mlp_conv_in,
        config.block_size,
    )?;
    let up = matvec_batch(
        manifest,
        &format!("{prefix}.mlp.up_proj.weight"),
        INTERMEDIATE,
        HIDDEN,
        &mlp_conv_in,
        config.block_size,
    )?;
    let activation = gate
        .iter()
        .zip(&up)
        .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
        .collect::<Vec<_>>();
    let down = matvec_batch(
        manifest,
        &format!("{prefix}.mlp.down_proj.weight"),
        HIDDEN,
        INTERMEDIATE,
        &activation,
        config.block_size,
    )?;
    let mlp_conv_out = apply_block_local_grouped_conv(
        config,
        &down,
        &mlp_dynamic,
        &read_tensor_f16(
            manifest,
            &format!("{prefix}.mlp_conv.base_kernel"),
            &[2, KERNEL_SIZE, HIDDEN],
        )?,
        1,
    )
    .map_err(|error| invalid(error.to_string()))?;
    let output = add_rows(&ffn_input, &mlp_conv_out, config.block_size, HIDDEN)?;
    stages.push((format!("layer{layer}_input_norm"), sha256_f32(&input_norm)));
    stages.push((format!("layer{layer}_attention"), sha256_f32(&attention)));
    stages.push((format!("layer{layer}_ffn"), sha256_f32(&mlp_conv_out)));
    stages.push((format!("layer{layer}_output"), sha256_f32(&output)));
    Ok(output)
}

fn read_vector(
    manifest: DFlashBlockWeights<'_>,
    name: &str,
    elements: usize,
) -> Result<Vec<f32>, DFlashBlockProducerError> {
    read_tensor_f16(manifest, name, &[elements])
}

fn read_tensor_f16(
    manifest: DFlashBlockWeights<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>, DFlashBlockProducerError> {
    let manifest = match manifest {
        DFlashBlockWeights::Q8(weights) => return weights.read_tensor(name, shape),
        DFlashBlockWeights::F16(weights) => weights,
    };
    let elements = shape
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
        .ok_or_else(|| invalid(format!("tensor {name} element overflow")))?;
    let bytes = manifest.tensor_bytes(name)?;
    if bytes.len() != elements * 2 {
        return Err(invalid(format!("tensor {name} extent")));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|word| f16::from_bits(u16::from_le_bytes([word[0], word[1]])).to_f32())
        .collect())
}

fn encode_target_context(
    manifest: DFlashBlockWeights<'_>,
    target_features: &[f32],
) -> Result<Vec<f32>, DFlashBlockProducerError> {
    let context_len = target_features.len() / TARGET_FEATURE_ROW;
    let projected = matvec_batch(
        manifest,
        "fc.weight",
        HIDDEN,
        TARGET_FEATURE_ROW,
        target_features,
        context_len,
    )?;
    rms_norm_rows(
        &projected,
        context_len,
        HIDDEN,
        &read_vector(manifest, "hidden_norm.weight", HIDDEN)?,
    )
}

fn matvec_batch(
    manifest: DFlashBlockWeights<'_>,
    name: &str,
    rows: usize,
    cols: usize,
    input: &[f32],
    batch: usize,
) -> Result<Vec<f32>, DFlashBlockProducerError> {
    let manifest = match manifest {
        DFlashBlockWeights::Q8(weights) => {
            return weights.matvec_batch(name, rows, cols, input, batch)
        }
        DFlashBlockWeights::F16(weights) => weights,
    };
    if input.len() != batch * cols {
        return Err(invalid(format!("tensor {name} input extent")));
    }
    let bytes = manifest.tensor_bytes(name)?;
    if bytes.len() != rows * cols * 2 {
        return Err(invalid(format!("tensor {name} byte extent")));
    }
    let mut output = vec![0.0_f32; batch * rows];
    for item in 0..batch {
        for row in 0..rows {
            let weight_start = row * cols * 2;
            let input_start = item * cols;
            let mut sum = 0.0_f64;
            for col in 0..cols {
                let index = weight_start + col * 2;
                let weight =
                    f16::from_bits(u16::from_le_bytes([bytes[index], bytes[index + 1]])).to_f32();
                sum += f64::from(weight) * f64::from(input[input_start + col]);
            }
            output[item * rows + row] = sum as f32;
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!("tensor {name} output is non-finite")));
    }
    Ok(output)
}

fn rms_norm_rows(
    input: &[f32],
    batch: usize,
    width: usize,
    weight: &[f32],
) -> Result<Vec<f32>, DFlashBlockProducerError> {
    if input.len() != batch * width || weight.len() != width {
        return Err(invalid("RMSNorm geometry"));
    }
    let mut output = vec![0.0_f32; input.len()];
    for row in 0..batch {
        let source = &input[row * width..(row + 1) * width];
        let mean = source
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            / width as f64;
        let scale = (mean as f32 + RMS_EPS).sqrt().recip();
        for col in 0..width {
            output[row * width + col] = source[col] * scale * weight[col];
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(invalid("RMSNorm output is non-finite"));
    }
    Ok(output)
}

fn add_rows(
    left: &[f32],
    right: &[f32],
    batch: usize,
    width: usize,
) -> Result<Vec<f32>, DFlashBlockProducerError> {
    if left.len() != batch * width || right.len() != left.len() {
        return Err(invalid("residual geometry"));
    }
    let output = left
        .iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect::<Vec<_>>();
    if output.iter().any(|value| !value.is_finite()) {
        return Err(invalid("residual output is non-finite"));
    }
    Ok(output)
}

fn normalize_heads_rope(
    values: &mut [f32],
    rows: usize,
    heads: usize,
    weight: &[f32],
    position_start: usize,
) -> Result<(), DFlashBlockProducerError> {
    if values.len() != rows * heads * HEAD_DIM || weight.len() != HEAD_DIM {
        return Err(invalid("head norm/RoPE geometry"));
    }
    for row in 0..rows {
        let position = position_start
            .checked_add(row)
            .ok_or_else(|| invalid("RoPE position overflow"))?;
        for head in 0..heads {
            let begin = (row * heads + head) * HEAD_DIM;
            let slice = &mut values[begin..begin + HEAD_DIM];
            let mean = slice
                .iter()
                .map(|value| f64::from(*value) * f64::from(*value))
                .sum::<f64>()
                / HEAD_DIM as f64;
            let scale = (mean as f32 + RMS_EPS).sqrt().recip();
            for value in slice.iter_mut() {
                *value *= scale;
            }
            for dim in 0..HEAD_DIM {
                slice[dim] *= weight[dim];
            }
            crate::apply_dflash_neox_rope(slice, position).map_err(invalid)?;
        }
    }
    Ok(())
}

fn sha256_f32(values: &[f32]) -> [u8; 32] {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
    digest.finalize().into()
}

fn invalid(message: impl Into<String>) -> DFlashBlockProducerError {
    DFlashBlockProducerError::Invalid(message.into())
}

fn validate_attention_output(
    output: &[f32],
    expected_elements: usize,
) -> Result<(), DFlashBlockProducerError> {
    if output.len() != expected_elements || output.iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            "injected attention output extent or non-finite arithmetic",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_executor_output_fails_closed_before_projection() {
        assert!(validate_attention_output(&[1.0, -2.0], 2).is_ok());
        for invalid in [
            vec![],
            vec![1.0],
            vec![1.0, 2.0, 3.0],
            vec![f32::NAN, 0.0],
            vec![0.0, f32::INFINITY],
        ] {
            assert!(validate_attention_output(&invalid, 2).is_err());
        }
        assert_ne!(
            DFLASH_BLOCK_INJECTED_ATTENTION_SCHEMA,
            DFLASH_BLOCK_PRODUCER_SCHEMA
        );
    }

    #[test]
    fn digest_is_stable_for_f32_bits() {
        assert_eq!(sha256_f32(&[0.0, 1.0, -2.0]), sha256_f32(&[0.0, 1.0, -2.0]));
        assert_ne!(sha256_f32(&[0.0]), sha256_f32(&[-0.0]));
    }
}
