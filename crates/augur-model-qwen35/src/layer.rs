//! Checked CPU execution boundary for one Qwen3.8 hybrid decoder layer.
//!
//! The layer owns no model files, CUDA resources, tokenizer state, or sampler.
//! A projection is supplied through [`Qwen35Projection`], which keeps the
//! operation usable with either a materialized CPU reference matrix or a
//! descriptor-bound quantized matvec.  The state checkpoint is deliberately
//! part of this boundary: attention/recurrent state is restored when any
//! later operation (including the FFN) fails.

use crate::config::Qwen35Config;
use crate::error::Qwen35Error;
use crate::gguf::GgufTensor;
use crate::quant::{quantized_matvec_fused_simd, Qwen35QuantType};
use crate::reference::{
    add_residual, rms_norm_last_dim, silu, target_rope_last_dim, Qwen35FullAttentionCheckpoint,
    Qwen35FullAttentionState, Qwen35LinearAttentionBlock, Qwen35LinearAttentionCheckpoint,
    Qwen35LinearAttentionInputs, ReferenceTensor,
};

/// A checked one-token projection operation.
///
/// Implementations must return `output_width()` finite values for a finite
/// input of `input_width()` values.  The trait intentionally exposes no raw
/// device pointer or allocator handle, so a future CUDA implementation can
/// preserve the same layer contract.
pub trait Qwen35Projection {
    fn input_width(&self) -> usize;
    fn output_width(&self) -> usize;
    fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, Qwen35Error>;
}

/// Materialized row-major `[output, input]` CPU projection.
#[derive(Clone, Debug)]
pub struct ReferenceProjection {
    weight: ReferenceTensor,
}

impl ReferenceProjection {
    pub fn new(weight: ReferenceTensor) -> Result<Self, Qwen35Error> {
        if weight.shape().len() != 2 {
            return Err(Qwen35Error::invalid(
                "layer.reference_projection.shape",
                format!("expected rank-2 [output, input], got {:?}", weight.shape()),
            ));
        }
        if weight.values().iter().any(|value| !value.is_finite()) {
            return Err(Qwen35Error::invalid(
                "layer.reference_projection.weight",
                "weights must be finite",
            ));
        }
        Ok(Self { weight })
    }

    pub fn from_values(
        output_width: usize,
        input_width: usize,
        values: impl Into<Vec<f32>>,
    ) -> Result<Self, Qwen35Error> {
        Self::new(ReferenceTensor::new(
            vec![output_width, input_width],
            values,
        )?)
    }

    pub fn weight(&self) -> &ReferenceTensor {
        &self.weight
    }
}

impl Qwen35Projection for ReferenceProjection {
    fn input_width(&self) -> usize {
        self.weight.shape()[1]
    }

    fn output_width(&self) -> usize {
        self.weight.shape()[0]
    }

    fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, Qwen35Error> {
        if input.len() != self.input_width() {
            return Err(Qwen35Error::invalid(
                "layer.reference_projection.input",
                format!(
                    "expected {} values, got {}",
                    self.input_width(),
                    input.len()
                ),
            ));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(Qwen35Error::invalid(
                "layer.reference_projection.input",
                "input values must be finite",
            ));
        }
        let mut output = vec![0.0f32; self.output_width()];
        for (row, destination) in output.iter_mut().enumerate() {
            let weights =
                &self.weight.values()[row * self.input_width()..(row + 1) * self.input_width()];
            let mut sum = 0.0f32;
            for (&weight, &value) in weights.iter().zip(input) {
                sum += weight * value;
                if !sum.is_finite() {
                    return Err(Qwen35Error::invalid(
                        "layer.reference_projection.accumulator",
                        format!("row {row} accumulated a non-finite value"),
                    ));
                }
            }
            *destination = sum;
        }
        Ok(output)
    }
}

/// Descriptor-bound quantized projection used by the real GGUF CPU path.
///
/// GGUF dimensions are `[input, output]`, unlike [`ReferenceProjection`]'s
/// conventional row-major `[output, input]`.  Multipliers are explicit so a
/// caller can bind the converter's activation/weight scale convention without
/// hiding it in the layer implementation.
pub struct GgufProjection<'a> {
    descriptor: &'a GgufTensor,
    payload: &'a [u8],
    input_width: usize,
    output_width: usize,
    input_multiplier: f32,
    output_multiplier: f32,
}

impl<'a> GgufProjection<'a> {
    pub fn new(descriptor: &'a GgufTensor, payload: &'a [u8]) -> Result<Self, Qwen35Error> {
        Self::with_multipliers(descriptor, payload, 1.0, 1.0)
    }

    pub fn with_multipliers(
        descriptor: &'a GgufTensor,
        payload: &'a [u8],
        input_multiplier: f32,
        output_multiplier: f32,
    ) -> Result<Self, Qwen35Error> {
        if descriptor.dimensions.len() != 2 {
            return Err(Qwen35Error::invalid(
                "layer.gguf_projection.dimensions",
                format!(
                    "expected rank-2 [input, output], got {:?}",
                    descriptor.dimensions
                ),
            ));
        }
        let input_width =
            usize::try_from(descriptor.dimensions[0]).map_err(|_| Qwen35Error::SizeOverflow)?;
        let output_width =
            usize::try_from(descriptor.dimensions[1]).map_err(|_| Qwen35Error::SizeOverflow)?;
        Qwen35QuantType::from_ggml_type(descriptor.ggml_type)?;
        if !input_multiplier.is_finite()
            || input_multiplier <= 0.0
            || !output_multiplier.is_finite()
            || output_multiplier <= 0.0
        {
            return Err(Qwen35Error::invalid(
                "layer.gguf_projection.multiplier",
                "multipliers must be finite and positive",
            ));
        }
        Ok(Self {
            descriptor,
            payload,
            input_width,
            output_width,
            input_multiplier,
            output_multiplier,
        })
    }

    pub fn descriptor(&self) -> &GgufTensor {
        self.descriptor
    }
}

impl Qwen35Projection for GgufProjection<'_> {
    fn input_width(&self) -> usize {
        self.input_width
    }

    fn output_width(&self) -> usize {
        self.output_width
    }

    fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, Qwen35Error> {
        if input.len() != self.input_width() {
            return Err(Qwen35Error::invalid(
                "layer.gguf_projection.input",
                format!(
                    "expected {} values, got {}",
                    self.input_width(),
                    input.len()
                ),
            ));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(Qwen35Error::invalid(
                "layer.gguf_projection.input",
                "input values must be finite",
            ));
        }
        let scaled_input;
        let effective_input = if self.input_multiplier == 1.0 {
            input
        } else {
            scaled_input = scale_values(
                input,
                self.input_multiplier,
                "layer.gguf_projection.input_multiplier",
            )?;
            &scaled_input
        };
        let mut output =
            quantized_matvec_fused_simd(self.descriptor, self.payload, effective_input)?;
        for value in &mut output {
            *value *= self.output_multiplier;
            if !value.is_finite() {
                return Err(Qwen35Error::invalid(
                    "layer.gguf_projection.output",
                    "scaled output is not finite",
                ));
            }
        }
        Ok(output)
    }
}

/// Whether a decoder layer uses Gated DeltaNet or full attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen35LayerKind {
    LinearAttention,
    FullAttention,
}

/// Geometry and numerical policy for one layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35LayerSpec {
    pub kind: Qwen35LayerKind,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub linear_key_heads: usize,
    pub linear_value_heads: usize,
    pub linear_head_dim: usize,
    pub linear_conv_kernel: usize,
}

impl Qwen35LayerSpec {
    /// Derive a layer spec from the authenticated model configuration.
    pub fn from_config(config: &Qwen35Config, layer: usize) -> Result<Self, Qwen35Error> {
        config.validate_exact_profile()?;
        if layer >= config.num_hidden_layers {
            return Err(Qwen35Error::invalid(
                "layer.index",
                format!(
                    "layer {layer} is outside {} layers",
                    config.num_hidden_layers
                ),
            ));
        }
        let kind = if config.full_attention_layers.contains(&layer) {
            Qwen35LayerKind::FullAttention
        } else if config.linear_attention_layers.contains(&layer) {
            Qwen35LayerKind::LinearAttention
        } else {
            return Err(Qwen35Error::invalid(
                "layer.kind",
                format!("layer {layer} is not present in either attention partition"),
            ));
        };
        let spec = Self {
            kind,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            query_heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rotary_dim: config.rope_dimension,
            rope_theta: config.rope_theta as f32,
            rms_norm_eps: config.rms_norm_eps as f32,
            linear_key_heads: config.linear_num_key_heads,
            linear_value_heads: config.linear_num_value_heads,
            linear_head_dim: config.linear_value_head_dim,
            linear_conv_kernel: config.linear_conv_kernel_dim,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<(), Qwen35Error> {
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("query_heads", self.query_heads),
            ("kv_heads", self.kv_heads),
            ("head_dim", self.head_dim),
            ("rotary_dim", self.rotary_dim),
            ("linear_key_heads", self.linear_key_heads),
            ("linear_value_heads", self.linear_value_heads),
            ("linear_head_dim", self.linear_head_dim),
            ("linear_conv_kernel", self.linear_conv_kernel),
        ] {
            if value == 0 {
                return Err(Qwen35Error::invalid(
                    format!("layer.spec.{name}"),
                    "must be nonzero",
                ));
            }
        }
        if self.rotary_dim > self.head_dim || !self.rotary_dim.is_multiple_of(2) {
            return Err(Qwen35Error::invalid(
                "layer.spec.rotary_dim",
                "must be even and no wider than head_dim",
            ));
        }
        if !self.query_heads.is_multiple_of(self.kv_heads) {
            return Err(Qwen35Error::invalid(
                "layer.spec.heads",
                "query heads must be divisible by KV heads",
            ));
        }
        self.linear_key_heads
            .checked_mul(self.linear_head_dim)
            .ok_or(Qwen35Error::SizeOverflow)?;
        self.linear_value_heads
            .checked_mul(self.linear_head_dim)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if !self
            .linear_value_heads
            .is_multiple_of(self.linear_key_heads)
        {
            return Err(Qwen35Error::invalid(
                "layer.spec.linear_heads",
                "value heads must be divisible by key heads",
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 1.0 {
            return Err(Qwen35Error::invalid(
                "layer.spec.rope_theta",
                "must be finite and greater than one",
            ));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(Qwen35Error::invalid(
                "layer.spec.rms_norm_eps",
                "must be finite and positive",
            ));
        }
        Ok(())
    }
}

/// Dense SwiGLU FFN weights for one layer.
pub struct Qwen35FfnWeights<'a> {
    pub gate: &'a dyn Qwen35Projection,
    pub up: &'a dyn Qwen35Projection,
    pub down: &'a dyn Qwen35Projection,
    pub gate_multiplier: f32,
    pub up_multiplier: f32,
    pub down_multiplier: f32,
}

/// Weights needed by one Gated DeltaNet layer.
pub struct Qwen35LinearLayerWeights<'a> {
    pub attn_norm: &'a [f32],
    pub qkv: &'a dyn Qwen35Projection,
    pub z: &'a dyn Qwen35Projection,
    pub alpha: &'a dyn Qwen35Projection,
    pub beta: &'a dyn Qwen35Projection,
    pub conv_weights: &'a [f32],
    pub dt_bias: &'a [f32],
    pub ssm_a: &'a [f32],
    pub ssm_norm: &'a [f32],
    pub ssm_out: &'a dyn Qwen35Projection,
    pub post_attention_norm: &'a [f32],
    pub ffn: Qwen35FfnWeights<'a>,
}

/// Weights needed by one full-attention layer.
pub struct Qwen35FullAttentionLayerWeights<'a> {
    pub attn_norm: &'a [f32],
    /// Joint query + output gate projection, ordered `[Q, gate]`.
    pub q: &'a dyn Qwen35Projection,
    pub k: &'a dyn Qwen35Projection,
    pub v: &'a dyn Qwen35Projection,
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub attn_output: &'a dyn Qwen35Projection,
    pub post_attention_norm: &'a [f32],
    pub ffn: Qwen35FfnWeights<'a>,
}

/// Borrowed layer weights, tagged with the corresponding attention kind.
pub enum Qwen35LayerWeights<'a> {
    Linear(Qwen35LinearLayerWeights<'a>),
    Full(Qwen35FullAttentionLayerWeights<'a>),
}

/// Request-local mutable state for one layer.
pub enum Qwen35LayerState {
    Linear {
        block: Qwen35LinearAttentionBlock,
        position: usize,
    },
    Full(Qwen35FullAttentionState),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Qwen35LayerCheckpoint {
    Linear {
        checkpoint: Qwen35LinearAttentionCheckpoint,
        position: usize,
    },
    Full(Qwen35FullAttentionCheckpoint),
}

impl Qwen35LayerCheckpoint {
    /// Return the request-state payload in a stable, layer-kind-independent
    /// order for audit and replay receipts.  Linear checkpoints are encoded as
    /// convolution history followed by the DeltaNet matrix; full-attention
    /// checkpoints are encoded as keys followed by values.  The position
    /// counter is intentionally kept outside this byte payload and remains
    /// available through the owning layer state.
    pub fn state_values(&self) -> Vec<f32> {
        match self {
            Self::Linear { checkpoint, .. } => {
                let mut values = Vec::with_capacity(
                    checkpoint.conv_history().len() + checkpoint.gated_delta_values().len(),
                );
                values.extend_from_slice(checkpoint.conv_history());
                values.extend_from_slice(checkpoint.gated_delta_values());
                values
            }
            Self::Full(checkpoint) => {
                let mut values =
                    Vec::with_capacity(checkpoint.keys().len() + checkpoint.values().len());
                values.extend_from_slice(checkpoint.keys());
                values.extend_from_slice(checkpoint.values());
                values
            }
        }
    }
}

impl Qwen35LayerState {
    pub fn for_spec(spec: &Qwen35LayerSpec, max_positions: usize) -> Result<Self, Qwen35Error> {
        spec.validate()?;
        if max_positions == 0 {
            return Err(Qwen35Error::invalid(
                "layer.state.max_positions",
                "must be nonzero",
            ));
        }
        match spec.kind {
            Qwen35LayerKind::LinearAttention => Ok(Self::Linear {
                block: Qwen35LinearAttentionBlock::zeros(
                    spec.linear_conv_kernel,
                    spec.linear_key_heads,
                    spec.linear_value_heads,
                    spec.linear_head_dim,
                )?,
                position: 0,
            }),
            Qwen35LayerKind::FullAttention => Ok(Self::Full(Qwen35FullAttentionState::zeros(
                spec.kv_heads,
                spec.head_dim,
                max_positions,
            )?)),
        }
    }

    pub fn position_count(&self) -> usize {
        match self {
            Self::Linear { position, .. } => *position,
            Self::Full(state) => state.position_count(),
        }
    }

    pub fn checkpoint(&self) -> Qwen35LayerCheckpoint {
        match self {
            Self::Linear { block, position } => Qwen35LayerCheckpoint::Linear {
                checkpoint: block.checkpoint(),
                position: *position,
            },
            Self::Full(state) => Qwen35LayerCheckpoint::Full(state.checkpoint()),
        }
    }

    pub fn restore(&mut self, checkpoint: &Qwen35LayerCheckpoint) -> Result<(), Qwen35Error> {
        match (self, checkpoint) {
            (
                Self::Linear { block, position },
                Qwen35LayerCheckpoint::Linear {
                    checkpoint,
                    position: checkpoint_position,
                },
            ) => {
                block.restore(checkpoint)?;
                *position = *checkpoint_position;
                Ok(())
            }
            (Self::Full(state), Qwen35LayerCheckpoint::Full(checkpoint)) => {
                state.restore(checkpoint)
            }
            _ => Err(Qwen35Error::invalid(
                "layer.state.checkpoint",
                "checkpoint kind does not match layer state",
            )),
        }
    }
}

/// Intermediate and final values emitted by one checked layer step.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35LayerOutput {
    pub attention_output: Vec<f32>,
    pub attention_residual: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub ffn_output: Vec<f32>,
    pub hidden: Vec<f32>,
}

/// Execute one layer step atomically.
///
/// `position` is zero-based and must equal the state's current position.  No
/// state is advanced when validation, projection, attention, or FFN execution
/// fails.  This is a CPU/reference contract; it intentionally makes no claim
/// about a complete model, token generation, semantic parity, or throughput.
pub fn qwen35_layer_forward(
    spec: &Qwen35LayerSpec,
    weights: &Qwen35LayerWeights<'_>,
    state: &mut Qwen35LayerState,
    hidden: &[f32],
    position: usize,
) -> Result<Qwen35LayerOutput, Qwen35Error> {
    spec.validate()?;
    if hidden.len() != spec.hidden_size {
        return Err(Qwen35Error::invalid(
            "layer.hidden",
            format!("expected {} values, got {}", spec.hidden_size, hidden.len()),
        ));
    }
    if hidden.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35Error::invalid(
            "layer.hidden",
            "hidden values must be finite",
        ));
    }
    if state.position_count() != position {
        return Err(Qwen35Error::invalid(
            "layer.position",
            format!(
                "position {position} does not match state position {}",
                state.position_count()
            ),
        ));
    }
    let checkpoint = state.checkpoint();
    let result = qwen35_layer_forward_inner(spec, weights, state, hidden);
    if result.is_err() {
        let restored = state.restore(&checkpoint);
        debug_assert!(restored.is_ok());
    }
    result
}

fn qwen35_layer_forward_inner(
    spec: &Qwen35LayerSpec,
    weights: &Qwen35LayerWeights<'_>,
    state: &mut Qwen35LayerState,
    hidden: &[f32],
) -> Result<Qwen35LayerOutput, Qwen35Error> {
    let attention_output = match (spec.kind, weights, state) {
        (
            Qwen35LayerKind::LinearAttention,
            Qwen35LayerWeights::Linear(weights),
            Qwen35LayerState::Linear { block, position },
        ) => {
            let output = run_linear_attention(spec, weights, block, hidden)?;
            *position = position.checked_add(1).ok_or(Qwen35Error::SizeOverflow)?;
            output
        }
        (
            Qwen35LayerKind::FullAttention,
            Qwen35LayerWeights::Full(weights),
            Qwen35LayerState::Full(state),
        ) => run_full_attention(spec, weights, state, hidden)?,
        _ => {
            return Err(Qwen35Error::invalid(
                "layer.kind",
                "weights and state do not match the layer specification",
            ));
        }
    };

    let hidden_tensor = ReferenceTensor::new(vec![1, spec.hidden_size], hidden.to_vec())?;
    let attention_tensor =
        ReferenceTensor::new(vec![1, spec.hidden_size], attention_output.clone())?;
    let attention_residual = add_residual(&hidden_tensor, &attention_tensor)?.into_values();

    let post_norm_weights = match weights {
        Qwen35LayerWeights::Linear(weights) => weights.post_attention_norm,
        Qwen35LayerWeights::Full(weights) => weights.post_attention_norm,
    };
    let post_norm = rms_norm_last_dim(
        &ReferenceTensor::new(vec![1, spec.hidden_size], attention_residual.clone())?,
        &ReferenceTensor::new(vec![spec.hidden_size], post_norm_weights.to_vec())?,
        spec.rms_norm_eps,
    )?
    .into_values();

    let ffn = match weights {
        Qwen35LayerWeights::Linear(weights) => &weights.ffn,
        Qwen35LayerWeights::Full(weights) => &weights.ffn,
    };
    let ffn_output = run_ffn(spec, ffn, &post_norm)?;
    let hidden_output = add_residual(
        &ReferenceTensor::new(vec![1, spec.hidden_size], attention_residual.clone())?,
        &ReferenceTensor::new(vec![1, spec.hidden_size], ffn_output.clone())?,
    )?
    .into_values();

    Ok(Qwen35LayerOutput {
        attention_output,
        attention_residual,
        post_attention_norm: post_norm,
        ffn_output,
        hidden: hidden_output,
    })
}

fn run_linear_attention(
    spec: &Qwen35LayerSpec,
    weights: &Qwen35LinearLayerWeights<'_>,
    block: &mut Qwen35LinearAttentionBlock,
    hidden: &[f32],
) -> Result<Vec<f32>, Qwen35Error> {
    let normalized = rms_norm_last_dim(
        &ReferenceTensor::new(vec![1, spec.hidden_size], hidden.to_vec())?,
        &ReferenceTensor::new(vec![spec.hidden_size], weights.attn_norm.to_vec())?,
        spec.rms_norm_eps,
    )?;
    let normalized = normalized.values();

    let key_values = spec
        .linear_key_heads
        .checked_mul(spec.linear_head_dim)
        .ok_or(Qwen35Error::SizeOverflow)?;
    let value_values = spec
        .linear_value_heads
        .checked_mul(spec.linear_head_dim)
        .ok_or(Qwen35Error::SizeOverflow)?;
    let qkv_width = key_values
        .checked_mul(2)
        .and_then(|value| value.checked_add(value_values))
        .ok_or(Qwen35Error::SizeOverflow)?;
    let qkv = checked_projection(
        weights.qkv,
        normalized,
        spec.hidden_size,
        qkv_width,
        "linear.qkv",
    )?;
    let z = checked_projection(
        weights.z,
        normalized,
        spec.hidden_size,
        value_values,
        "linear.z",
    )?;
    let alpha = checked_projection(
        weights.alpha,
        normalized,
        spec.hidden_size,
        spec.linear_value_heads,
        "linear.alpha",
    )?;
    let beta = checked_projection(
        weights.beta,
        normalized,
        spec.hidden_size,
        spec.linear_value_heads,
        "linear.beta",
    )?;
    if weights.attn_norm.len() != spec.hidden_size
        || weights.dt_bias.len() != spec.linear_value_heads
        || weights.ssm_a.len() != spec.linear_value_heads
        || weights.ssm_norm.len() != spec.linear_head_dim
    {
        return Err(Qwen35Error::invalid(
            "linear.parameters",
            "normalization and decay parameter widths do not match the spec",
        ));
    }
    let mut decay_gate = Vec::with_capacity(spec.linear_value_heads);
    let mut beta_gate = Vec::with_capacity(spec.linear_value_heads);
    for index in 0..spec.linear_value_heads {
        let alpha_value = stable_softplus(alpha[index] + weights.dt_bias[index])?;
        let decay = alpha_value * weights.ssm_a[index];
        if !decay.is_finite() {
            return Err(Qwen35Error::invalid(
                "linear.decay_gate",
                format!("value {index} is not finite"),
            ));
        }
        decay_gate.push(decay);
        beta_gate.push(stable_sigmoid(beta[index])?);
    }

    let conv_channels = key_values
        .checked_mul(2)
        .and_then(|value| value.checked_add(value_values))
        .ok_or(Qwen35Error::SizeOverflow)?;
    let expected_conv_weights = spec
        .linear_conv_kernel
        .checked_mul(conv_channels)
        .ok_or(Qwen35Error::SizeOverflow)?;
    if weights.conv_weights.len() != expected_conv_weights {
        return Err(Qwen35Error::invalid(
            "linear.conv_weights",
            format!(
                "expected {expected_conv_weights} values, got {}",
                weights.conv_weights.len()
            ),
        ));
    }
    let recurrent = block.step(Qwen35LinearAttentionInputs {
        qkv_mixed: &qkv,
        conv_weights: weights.conv_weights,
        z: &z,
        decay_gate: &decay_gate,
        beta: &beta_gate,
        norm_weight: weights.ssm_norm,
        epsilon: spec.rms_norm_eps,
    })?;
    checked_projection(
        weights.ssm_out,
        &recurrent,
        value_values,
        spec.hidden_size,
        "linear.ssm_out",
    )
}

fn run_full_attention(
    spec: &Qwen35LayerSpec,
    weights: &Qwen35FullAttentionLayerWeights<'_>,
    state: &mut Qwen35FullAttentionState,
    hidden: &[f32],
) -> Result<Vec<f32>, Qwen35Error> {
    let normalized = rms_norm_last_dim(
        &ReferenceTensor::new(vec![1, spec.hidden_size], hidden.to_vec())?,
        &ReferenceTensor::new(vec![spec.hidden_size], weights.attn_norm.to_vec())?,
        spec.rms_norm_eps,
    )?;
    let normalized = normalized.values();
    let query_values = spec
        .query_heads
        .checked_mul(spec.head_dim)
        .ok_or(Qwen35Error::SizeOverflow)?;
    let kv_values = spec
        .kv_heads
        .checked_mul(spec.head_dim)
        .ok_or(Qwen35Error::SizeOverflow)?;
    let joint_q_width = query_values
        .checked_mul(2)
        .ok_or(Qwen35Error::SizeOverflow)?;
    let q_full = checked_projection(
        weights.q,
        normalized,
        spec.hidden_size,
        joint_q_width,
        "full.q_gate",
    )?;
    let k_raw = checked_projection(weights.k, normalized, spec.hidden_size, kv_values, "full.k")?;
    let v_raw = checked_projection(weights.v, normalized, spec.hidden_size, kv_values, "full.v")?;
    if weights.q_norm.len() != spec.head_dim || weights.k_norm.len() != spec.head_dim {
        return Err(Qwen35Error::invalid(
            "full.norm",
            "query/key norm widths do not match head_dim",
        ));
    }
    let (q_raw, gate) = split_interleaved_query_gate(&q_full, spec.query_heads, spec.head_dim)?;
    let q = rms_norm_last_dim(
        &ReferenceTensor::new(vec![spec.query_heads, spec.head_dim], q_raw)?,
        &ReferenceTensor::new(vec![spec.head_dim], weights.q_norm.to_vec())?,
        spec.rms_norm_eps,
    )?;
    let k = rms_norm_last_dim(
        &ReferenceTensor::new(vec![spec.kv_heads, spec.head_dim], k_raw)?,
        &ReferenceTensor::new(vec![spec.head_dim], weights.k_norm.to_vec())?,
        spec.rms_norm_eps,
    )?;
    let q = target_rope_last_dim(&q, state.position_count(), spec.rotary_dim, spec.rope_theta)?;
    let k = target_rope_last_dim(&k, state.position_count(), spec.rotary_dim, spec.rope_theta)?;
    let attention = state.step(q.values(), k.values(), &v_raw, spec.query_heads)?;
    if attention.len() != query_values {
        return Err(Qwen35Error::invalid(
            "full.attention",
            format!("expected {query_values} values, got {}", attention.len()),
        ));
    }
    let mut gated = Vec::with_capacity(query_values);
    for (&value, &gate_value) in attention.iter().zip(&gate) {
        let gated_value = value * stable_sigmoid(gate_value)?;
        if !gated_value.is_finite() {
            return Err(Qwen35Error::invalid(
                "full.gated_attention",
                "gated attention value is not finite",
            ));
        }
        gated.push(gated_value);
    }
    checked_projection(
        weights.attn_output,
        &gated,
        query_values,
        spec.hidden_size,
        "full.output",
    )
}

pub fn split_interleaved_query_gate(
    values: &[f32],
    query_heads: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>), Qwen35Error> {
    let head_stride = head_dim.checked_mul(2).ok_or(Qwen35Error::SizeOverflow)?;
    let expected = query_heads
        .checked_mul(head_stride)
        .ok_or(Qwen35Error::SizeOverflow)?;
    if values.len() != expected {
        return Err(Qwen35Error::invalid(
            "full.q_gate_layout",
            format!(
                "expected {expected} interleaved query/gate values, got {}",
                values.len()
            ),
        ));
    }

    let query_values = query_heads
        .checked_mul(head_dim)
        .ok_or(Qwen35Error::SizeOverflow)?;
    let mut query = Vec::with_capacity(query_values);
    let mut gate = Vec::with_capacity(query_values);
    for head in values.chunks_exact(head_stride) {
        query.extend_from_slice(&head[..head_dim]);
        gate.extend_from_slice(&head[head_dim..]);
    }
    Ok((query, gate))
}

fn run_ffn(
    spec: &Qwen35LayerSpec,
    weights: &Qwen35FfnWeights<'_>,
    input: &[f32],
) -> Result<Vec<f32>, Qwen35Error> {
    let gate = checked_projection(
        weights.gate,
        input,
        spec.hidden_size,
        spec.intermediate_size,
        "ffn.gate",
    )?;
    let up = checked_projection(
        weights.up,
        input,
        spec.hidden_size,
        spec.intermediate_size,
        "ffn.up",
    )?;
    let gate = scale_values(&gate, weights.gate_multiplier, "ffn.gate_multiplier")?;
    let up = scale_values(&up, weights.up_multiplier, "ffn.up_multiplier")?;
    let gate = silu(&ReferenceTensor::new(vec![spec.intermediate_size], gate)?)?;
    let mut activated = Vec::with_capacity(spec.intermediate_size);
    for (&gate_value, &up_value) in gate.values().iter().zip(&up) {
        let value = gate_value * up_value;
        if !value.is_finite() {
            return Err(Qwen35Error::invalid(
                "ffn.activation",
                "SwiGLU activation is not finite",
            ));
        }
        activated.push(value);
    }
    let down = checked_projection(
        weights.down,
        &activated,
        spec.intermediate_size,
        spec.hidden_size,
        "ffn.down",
    )?;
    scale_values(&down, weights.down_multiplier, "ffn.down_multiplier")
}

fn checked_projection(
    projection: &dyn Qwen35Projection,
    input: &[f32],
    expected_input: usize,
    expected_output: usize,
    field: &str,
) -> Result<Vec<f32>, Qwen35Error> {
    if projection.input_width() != expected_input || projection.output_width() != expected_output {
        return Err(Qwen35Error::invalid(
            field,
            format!(
                "projection geometry is {}x{}, expected {}x{}",
                projection.output_width(),
                projection.input_width(),
                expected_output,
                expected_input
            ),
        ));
    }
    let output = projection.matvec(input)?;
    if output.len() != expected_output || output.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35Error::invalid(
            field,
            "projection returned an invalid output",
        ));
    }
    Ok(output)
}

fn scale_values(values: &[f32], multiplier: f32, field: &str) -> Result<Vec<f32>, Qwen35Error> {
    if !multiplier.is_finite() || multiplier <= 0.0 {
        return Err(Qwen35Error::invalid(
            field,
            "multiplier must be finite and positive",
        ));
    }
    let mut output = Vec::with_capacity(values.len());
    for (index, value) in values.iter().copied().enumerate() {
        let scaled = value * multiplier;
        if !scaled.is_finite() {
            return Err(Qwen35Error::invalid(
                field,
                format!("scaled value {index} is not finite"),
            ));
        }
        output.push(scaled);
    }
    Ok(output)
}

fn stable_sigmoid(value: f32) -> Result<f32, Qwen35Error> {
    let result = if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponent = value.exp();
        exponent / (1.0 + exponent)
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(Qwen35Error::invalid(
            "layer.sigmoid",
            "sigmoid result is not finite",
        ))
    }
}

fn stable_softplus(value: f32) -> Result<f32, Qwen35Error> {
    let result = if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        (1.0 + value.exp()).ln()
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(Qwen35Error::invalid(
            "layer.softplus",
            "softplus result is not finite",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_spec(kind: Qwen35LayerKind) -> Qwen35LayerSpec {
        let spec = Qwen35LayerSpec {
            kind,
            hidden_size: 4,
            intermediate_size: 4,
            query_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            rotary_dim: 2,
            rope_theta: 10_000.0,
            rms_norm_eps: 1.0e-5,
            linear_key_heads: 1,
            linear_value_heads: 1,
            linear_head_dim: 2,
            linear_conv_kernel: 2,
        };
        spec.validate().unwrap();
        spec
    }

    fn identity(width: usize) -> ReferenceProjection {
        let mut values = vec![0.0; width * width];
        for index in 0..width {
            values[index * width + index] = 1.0;
        }
        ReferenceProjection::from_values(width, width, values).unwrap()
    }

    fn q8_descriptor(input_width: usize, output_width: usize) -> GgufTensor {
        assert!(input_width.is_multiple_of(crate::quant::Q8_0_BLOCK_ELEMENTS));
        GgufTensor {
            name: "test.q8".into(),
            dimensions: vec![input_width as u64, output_width as u64],
            ggml_type: crate::quant::GGML_TYPE_Q8_0,
            relative_offset: 0,
            absolute_offset: 0,
            byte_len: (input_width / crate::quant::Q8_0_BLOCK_ELEMENTS)
                .checked_mul(output_width)
                .and_then(|blocks| blocks.checked_mul(crate::quant::Q8_0_BLOCK_BYTES))
                .unwrap() as u64,
        }
    }

    fn q8_payload(input_width: usize, output_width: usize) -> Vec<u8> {
        let blocks_per_row = input_width / crate::quant::Q8_0_BLOCK_ELEMENTS;
        let mut payload =
            vec![0_u8; output_width * blocks_per_row * crate::quant::Q8_0_BLOCK_BYTES];
        for row in 0..output_width {
            for block in 0..blocks_per_row {
                let start = (row * blocks_per_row + block) * crate::quant::Q8_0_BLOCK_BYTES;
                payload[start..start + 2].copy_from_slice(&0x3c00_u16.to_le_bytes());
                for lane in 0..crate::quant::Q8_0_BLOCK_ELEMENTS {
                    payload[start + 2 + lane] = (lane as u8).wrapping_add(row as u8 + 1);
                }
            }
        }
        payload
    }

    #[test]
    fn gguf_projection_matches_q8_reference_and_applies_multipliers() {
        let descriptor = q8_descriptor(32, 2);
        let payload = q8_payload(32, 2);
        let input: Vec<f32> = (0..32).map(|value| value as f32 / 8.0).collect();
        let scaled_input: Vec<f32> = input.iter().map(|value| value * 2.0).collect();
        let reference =
            crate::quant::quantized_matvec_fused_simd(&descriptor, &payload, &scaled_input)
                .unwrap();
        let projection = GgufProjection::with_multipliers(&descriptor, &payload, 2.0, 0.5).unwrap();
        assert_eq!(projection.input_width(), 32);
        assert_eq!(projection.output_width(), 2);
        let actual = projection.matvec(&input).unwrap();
        assert_eq!(actual.len(), reference.len());
        for (actual, expected) in actual.iter().zip(reference) {
            assert!((actual - expected * 0.5).abs() <= 1.0e-5);
        }
    }

    #[test]
    fn gguf_projection_rejects_invalid_descriptor_payload_and_multipliers() {
        let descriptor = q8_descriptor(32, 1);
        let payload = q8_payload(32, 1);
        assert!(GgufProjection::with_multipliers(&descriptor, &payload, 0.0, 1.0).is_err());
        assert!(GgufProjection::with_multipliers(&descriptor, &payload, 1.0, f32::NAN).is_err());

        let mut rank_one = descriptor.clone();
        rank_one.dimensions.pop();
        assert!(GgufProjection::new(&rank_one, &payload).is_err());

        let mut unsupported = descriptor.clone();
        unsupported.ggml_type = 999;
        assert!(GgufProjection::new(&unsupported, &payload).is_err());

        let truncated = &payload[..payload.len() - 1];
        let projection = GgufProjection::new(&descriptor, truncated).unwrap();
        assert!(projection.matvec(&[1.0; 32]).is_err());
        assert!(projection.matvec(&[f32::INFINITY; 32]).is_err());
    }

    fn ffn<'a>(
        gate: &'a ReferenceProjection,
        up: &'a ReferenceProjection,
        down: &'a ReferenceProjection,
    ) -> Qwen35FfnWeights<'a> {
        Qwen35FfnWeights {
            gate,
            up,
            down,
            gate_multiplier: 1.0,
            up_multiplier: 1.0,
            down_multiplier: 1.0,
        }
    }

    #[test]
    fn linear_layer_advances_and_replays_atomically() {
        let spec = tiny_spec(Qwen35LayerKind::LinearAttention);
        let identity4 = identity(4);
        let qkv = ReferenceProjection::from_values(
            6,
            4,
            vec![
                1.0, 0.0, 0.0, 0.0, // q0
                0.0, 1.0, 0.0, 0.0, // q1
                0.0, 0.0, 1.0, 0.0, // k0
                0.0, 0.0, 0.0, 1.0, // k1
                1.0, 0.0, 0.0, 0.0, // v0
                0.0, 1.0, 0.0, 0.0, // v1
            ],
        )
        .unwrap();
        let z =
            ReferenceProjection::from_values(2, 4, vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
                .unwrap();
        let scalar = ReferenceProjection::from_values(1, 4, vec![0.0; 4]).unwrap();
        let ssm_out =
            ReferenceProjection::from_values(4, 2, vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5])
                .unwrap();
        let conv_weights = vec![0.0; 2 * 6];
        let linear = Qwen35LinearLayerWeights {
            attn_norm: &[1.0; 4],
            qkv: &qkv,
            z: &z,
            alpha: &scalar,
            beta: &scalar,
            conv_weights: &conv_weights,
            dt_bias: &[0.0],
            ssm_a: &[-1.0],
            ssm_norm: &[1.0, 1.0],
            ssm_out: &ssm_out,
            post_attention_norm: &[1.0; 4],
            ffn: ffn(&identity4, &identity4, &identity4),
        };
        let weights = Qwen35LayerWeights::Linear(linear);
        let mut state = Qwen35LayerState::for_spec(&spec, 4).unwrap();
        let hidden = vec![1.0, 2.0, 3.0, 4.0];
        let first = qwen35_layer_forward(&spec, &weights, &mut state, &hidden, 0).unwrap();
        assert_eq!(state.position_count(), 1);
        let checkpoint = state.checkpoint();
        let second = qwen35_layer_forward(&spec, &weights, &mut state, &hidden, 1).unwrap();
        assert_eq!(state.position_count(), 2);
        state.restore(&checkpoint).unwrap();
        let replay = qwen35_layer_forward(&spec, &weights, &mut state, &hidden, 1).unwrap();
        assert_eq!(second, replay);
        assert!(first.hidden.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn full_layer_advances_cache_and_replays_exactly() {
        let spec = tiny_spec(Qwen35LayerKind::FullAttention);
        let identity4 = identity(4);
        let q = ReferenceProjection::from_values(
            8,
            4,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap();
        let k =
            ReferenceProjection::from_values(2, 4, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                .unwrap();
        let v =
            ReferenceProjection::from_values(2, 4, vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                .unwrap();
        let output = ReferenceProjection::from_values(
            4,
            4,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap();
        let full = Qwen35FullAttentionLayerWeights {
            attn_norm: &[1.0; 4],
            q: &q,
            k: &k,
            v: &v,
            q_norm: &[1.0, 1.0],
            k_norm: &[1.0, 1.0],
            attn_output: &output,
            post_attention_norm: &[1.0; 4],
            ffn: ffn(&identity4, &identity4, &identity4),
        };
        let weights = Qwen35LayerWeights::Full(full);
        let mut state = Qwen35LayerState::for_spec(&spec, 4).unwrap();
        let hidden = vec![1.0, 2.0, 3.0, 4.0];
        let _ = qwen35_layer_forward(&spec, &weights, &mut state, &hidden, 0).unwrap();
        let checkpoint = state.checkpoint();
        let second = qwen35_layer_forward(&spec, &weights, &mut state, &hidden, 1).unwrap();
        state.restore(&checkpoint).unwrap();
        let replay = qwen35_layer_forward(&spec, &weights, &mut state, &hidden, 1).unwrap();
        assert_eq!(second, replay);
        assert_eq!(state.position_count(), 2);
    }

    #[test]
    fn full_attention_splits_query_and_gate_within_each_head() {
        let values = [0.0, 1.0, 10.0, 11.0, 2.0, 3.0, 12.0, 13.0];
        let (query, gate) = split_interleaved_query_gate(&values, 2, 2).unwrap();
        assert_eq!(query, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(gate, vec![10.0, 11.0, 12.0, 13.0]);
        assert!(split_interleaved_query_gate(&values[..7], 2, 2).is_err());
    }

    #[test]
    fn ffn_geometry_failure_restores_recurrent_state() {
        let spec = tiny_spec(Qwen35LayerKind::LinearAttention);
        let identity4 = identity(4);
        let qkv = ReferenceProjection::from_values(6, 4, vec![0.0; 24]).unwrap();
        let z = ReferenceProjection::from_values(2, 4, vec![0.0; 8]).unwrap();
        let scalar = ReferenceProjection::from_values(1, 4, vec![0.0; 4]).unwrap();
        let ssm_out = ReferenceProjection::from_values(4, 2, vec![0.0; 8]).unwrap();
        let wrong_down = ReferenceProjection::from_values(3, 4, vec![0.0; 12]).unwrap();
        let conv_weights = vec![0.0; 12];
        let linear = Qwen35LinearLayerWeights {
            attn_norm: &[1.0; 4],
            qkv: &qkv,
            z: &z,
            alpha: &scalar,
            beta: &scalar,
            conv_weights: &conv_weights,
            dt_bias: &[0.0],
            ssm_a: &[-1.0],
            ssm_norm: &[1.0, 1.0],
            ssm_out: &ssm_out,
            post_attention_norm: &[1.0; 4],
            ffn: Qwen35FfnWeights {
                gate: &identity4,
                up: &identity4,
                down: &wrong_down,
                gate_multiplier: 1.0,
                up_multiplier: 1.0,
                down_multiplier: 1.0,
            },
        };
        let weights = Qwen35LayerWeights::Linear(linear);
        let mut state = Qwen35LayerState::for_spec(&spec, 2).unwrap();
        let before = state.checkpoint();
        assert!(qwen35_layer_forward(&spec, &weights, &mut state, &[1.0; 4], 0).is_err());
        assert_eq!(state.checkpoint(), before);
    }
}
