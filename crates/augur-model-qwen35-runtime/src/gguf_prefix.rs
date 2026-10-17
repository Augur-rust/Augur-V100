//! CPU-only real-weight prefix execution for the isolated Qwen35 lane.
//!
//! This module owns the packed layer-zero payloads after they have been read
//! through one identity-bound GGUF reader.  It implements the
//! TargetOnlyLayerExecutor contract for bounded linear-attention layers; it is
//! intentionally not a complete transformer executor and does not publish
//! tokens or throughput claims.

use crate::{TargetOnlyError, TargetOnlyLayerExecutor};
use augur_model_qwen35::{
    dequantize_tensor, qwen35_layer_forward, GgufProjectionNumerics, GgufTensor, GgufTensorReader,
    OwnedGgufProjection, Qwen35Config, Qwen35FfnWeights, Qwen35FullAttentionLayerWeights,
    Qwen35LayerKind, Qwen35LayerOutput, Qwen35LayerSpec, Qwen35LayerState, Qwen35LayerWeights,
    Qwen35LinearLayerWeights, Qwen35TensorSchema, GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_NVFP4,
    GGML_TYPE_Q8_0,
};

const FIRST_LAYER: usize = 0;

/// A fully owned, real-weight linear-attention layer executor.
///
/// The projections retain their GGML payloads in their source quantization;
/// only the small vector parameters are decoded to f32.  The executor owns
/// no file handle, tokenizer, sampler, or CUDA resource, so it can be used by
/// a request-local TargetOnlySession without a self-referential lifetime.
pub struct GgufLinearLayer0Executor {
    layer: usize,
    spec: Qwen35LayerSpec,
    attn_norm: Vec<f32>,
    qkv: OwnedGgufProjection,
    z: OwnedGgufProjection,
    alpha: OwnedGgufProjection,
    beta: OwnedGgufProjection,
    conv_weights: Vec<f32>,
    dt_bias: Vec<f32>,
    ssm_a: Vec<f32>,
    ssm_norm: Vec<f32>,
    ssm_out: OwnedGgufProjection,
    post_attention_norm: Vec<f32>,
    ffn_gate: OwnedGgufProjection,
    ffn_up: OwnedGgufProjection,
    ffn_down: OwnedGgufProjection,
    ffn_gate_scale: f32,
    ffn_up_scale: f32,
    ffn_down_scale: f32,
    ffn_input_scales: [f32; 3],
}

impl GgufLinearLayer0Executor {
    /// Load the complete real layer-zero linear-attention block from a bound
    /// Qwen35 GGUF reader.
    pub fn for_qwen35(
        reader: &mut GgufTensorReader<'_>,
        config: &Qwen35Config,
    ) -> Result<Self, TargetOnlyError> {
        Self::for_layer(reader, config, FIRST_LAYER)
    }

    /// Load one linear-attention layer from a bound Qwen35 GGUF reader.
    ///
    /// Full-attention layers are rejected here; they require a separate
    /// grouped-query attention executor and state contract.
    pub fn for_layer(
        reader: &mut GgufTensorReader<'_>,
        config: &Qwen35Config,
        layer: usize,
    ) -> Result<Self, TargetOnlyError> {
        Self::for_layer_with_numerics(
            reader,
            config,
            layer,
            GgufProjectionNumerics::NativeF32Activation,
        )
    }

    /// Load one linear-attention layer with an explicit projection ABI.
    pub fn for_layer_with_numerics(
        reader: &mut GgufTensorReader<'_>,
        config: &Qwen35Config,
        layer: usize,
        numerics: GgufProjectionNumerics,
    ) -> Result<Self, TargetOnlyError> {
        config.validate_exact_profile()?;
        if layer >= config.num_hidden_layers {
            return Err(invalid(
                "layer.index",
                format!(
                    "layer {layer} is outside {} layers",
                    config.num_hidden_layers
                ),
            ));
        }
        let spec = Qwen35LayerSpec::from_config(config, layer)?;
        if spec.kind != Qwen35LayerKind::LinearAttention {
            return Err(invalid(
                "layer.kind",
                format!("layer {layer} is not linear attention"),
            ));
        }
        let hidden = spec.hidden_size;
        let key_values = spec
            .linear_key_heads
            .checked_mul(spec.linear_head_dim)
            .ok_or(augur_model_qwen35::Qwen35Error::SizeOverflow)?;
        let value_values = spec
            .linear_value_heads
            .checked_mul(spec.linear_head_dim)
            .ok_or(augur_model_qwen35::Qwen35Error::SizeOverflow)?;
        let qkv_width = key_values
            .checked_mul(2)
            .and_then(|value| value.checked_add(value_values))
            .ok_or(augur_model_qwen35::Qwen35Error::SizeOverflow)?;
        let conv_channels = qkv_width;
        let prefix = format!("blk.{layer}");
        let (projection_numerics, output_numerics, ffn_numerics) =
            split_layer_numerics(numerics, false);

        let attn_norm = load_dense(
            reader,
            &format!("{prefix}.attn_norm.weight"),
            &[hidden as u64],
            GGML_TYPE_F32,
        )?;
        let qkv = load_projection(
            reader,
            &format!("{prefix}.attn_qkv.weight"),
            &[hidden as u64, qkv_width as u64],
            GGML_TYPE_Q8_0,
            projection_numerics,
        )?;
        let z = load_projection(
            reader,
            &format!("{prefix}.attn_gate.weight"),
            &[hidden as u64, value_values as u64],
            GGML_TYPE_Q8_0,
            projection_numerics,
        )?;
        let alpha = load_projection(
            reader,
            &format!("{prefix}.ssm_alpha.weight"),
            &[hidden as u64, spec.linear_value_heads as u64],
            GGML_TYPE_BF16,
            projection_numerics,
        )?;
        let beta = load_projection(
            reader,
            &format!("{prefix}.ssm_beta.weight"),
            &[hidden as u64, spec.linear_value_heads as u64],
            GGML_TYPE_BF16,
            projection_numerics,
        )?;
        let conv_weights = load_dense(
            reader,
            &format!("{prefix}.ssm_conv1d.weight"),
            &[spec.linear_conv_kernel as u64, conv_channels as u64],
            GGML_TYPE_F32,
        )?;
        let ssm_a = load_dense(
            reader,
            &format!("{prefix}.ssm_a"),
            &[spec.linear_value_heads as u64],
            GGML_TYPE_F32,
        )?;
        let dt_bias = load_dense(
            reader,
            &format!("{prefix}.ssm_dt.bias"),
            &[spec.linear_value_heads as u64],
            GGML_TYPE_F32,
        )?;
        let ssm_norm = load_dense(
            reader,
            &format!("{prefix}.ssm_norm.weight"),
            &[spec.linear_head_dim as u64],
            GGML_TYPE_F32,
        )?;
        let ssm_out = load_projection(
            reader,
            &format!("{prefix}.ssm_out.weight"),
            &[value_values as u64, hidden as u64],
            GGML_TYPE_Q8_0,
            output_numerics,
        )?;
        let post_attention_norm = load_dense(
            reader,
            &format!("{prefix}.post_attention_norm.weight"),
            &[hidden as u64],
            GGML_TYPE_F32,
        )?;

        let (ffn_type, ffn_gate_scale, ffn_up_scale, ffn_down_scale, ffn_input_scales) =
            load_ffn_quantization(reader, config, layer, &prefix)?;
        let ffn_gate = load_projection(
            reader,
            &format!("{prefix}.ffn_gate.weight"),
            &[hidden as u64, spec.intermediate_size as u64],
            ffn_type,
            ffn_numerics,
        )?;
        let ffn_up = load_projection(
            reader,
            &format!("{prefix}.ffn_up.weight"),
            &[hidden as u64, spec.intermediate_size as u64],
            ffn_type,
            ffn_numerics,
        )?;
        let ffn_down = load_projection(
            reader,
            &format!("{prefix}.ffn_down.weight"),
            &[spec.intermediate_size as u64, hidden as u64],
            ffn_type,
            ffn_numerics,
        )?;

        Ok(Self {
            layer,
            spec,
            attn_norm,
            qkv,
            z,
            alpha,
            beta,
            conv_weights,
            dt_bias,
            ssm_a,
            ssm_norm,
            ssm_out,
            post_attention_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            ffn_gate_scale,
            ffn_up_scale,
            ffn_down_scale,
            ffn_input_scales,
        })
    }

    pub fn spec(&self) -> &Qwen35LayerSpec {
        &self.spec
    }

    pub fn layer_index(&self) -> usize {
        self.layer
    }

    pub fn ffn_input_scales(&self) -> [f32; 3] {
        self.ffn_input_scales
    }

    /// Number of bytes retained by the owned packed projection payloads.
    pub fn projection_payload_bytes(&self) -> usize {
        self.qkv
            .payload()
            .len()
            .saturating_add(self.z.payload().len())
            .saturating_add(self.alpha.payload().len())
            .saturating_add(self.beta.payload().len())
            .saturating_add(self.ssm_out.payload().len())
            .saturating_add(self.ffn_gate.payload().len())
            .saturating_add(self.ffn_up.payload().len())
            .saturating_add(self.ffn_down.payload().len())
    }

    /// Execute this linear-attention layer against a request-local state.
    pub fn forward_layer(
        &self,
        state: &mut Qwen35LayerState,
        hidden: &[f32],
        position: usize,
    ) -> Result<Qwen35LayerOutput, TargetOnlyError> {
        if !matches!(state, Qwen35LayerState::Linear { .. }) {
            return Err(invalid(
                "layer.state",
                "linear-layer executor requires a linear-attention state",
            ));
        }
        let weights = Qwen35LayerWeights::Linear(Qwen35LinearLayerWeights {
            attn_norm: &self.attn_norm,
            qkv: &self.qkv,
            z: &self.z,
            alpha: &self.alpha,
            beta: &self.beta,
            conv_weights: &self.conv_weights,
            dt_bias: &self.dt_bias,
            ssm_a: &self.ssm_a,
            ssm_norm: &self.ssm_norm,
            ssm_out: &self.ssm_out,
            post_attention_norm: &self.post_attention_norm,
            ffn: Qwen35FfnWeights {
                gate: &self.ffn_gate,
                up: &self.ffn_up,
                down: &self.ffn_down,
                gate_multiplier: self.ffn_gate_scale,
                up_multiplier: self.ffn_up_scale,
                down_multiplier: self.ffn_down_scale,
            },
        });
        qwen35_layer_forward(&self.spec, &weights, state, hidden, position).map_err(Into::into)
    }
}

impl TargetOnlyLayerExecutor for GgufLinearLayer0Executor {
    fn forward(
        &mut self,
        layer: usize,
        spec: &Qwen35LayerSpec,
        state: &mut Qwen35LayerState,
        hidden: &[f32],
        position: usize,
    ) -> Result<Vec<f32>, TargetOnlyError> {
        if layer != self.layer {
            return Err(TargetOnlyError::Executor {
                layer,
                detail: format!(
                    "executor is bound to linear layer {} and cannot service another layer",
                    self.layer
                ),
            });
        }
        if spec != &self.spec {
            return Err(TargetOnlyError::Executor {
                layer,
                detail: format!(
                    "caller layer specification differs from bound linear layer {}",
                    self.layer
                ),
            });
        }
        Ok(self.forward_layer(state, hidden, position)?.hidden)
    }
}

/// A fully owned, real-weight full-attention layer executor.
///
/// The Qwen35 full-attention layer uses grouped-query attention (24 query
/// heads and 4 KV heads in the authenticated profile).  Its recurrent KV
/// state remains request-local in [`Qwen35LayerState`]; this object owns only
/// immutable model payloads and therefore remains safe to reuse across
/// sessions.
pub struct GgufFullAttentionExecutor {
    layer: usize,
    spec: Qwen35LayerSpec,
    attn_norm: Vec<f32>,
    q: OwnedGgufProjection,
    k: OwnedGgufProjection,
    v: OwnedGgufProjection,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    attn_output: OwnedGgufProjection,
    post_attention_norm: Vec<f32>,
    ffn_gate: OwnedGgufProjection,
    ffn_up: OwnedGgufProjection,
    ffn_down: OwnedGgufProjection,
    ffn_gate_scale: f32,
    ffn_up_scale: f32,
    ffn_down_scale: f32,
    ffn_input_scales: [f32; 3],
}

impl GgufFullAttentionExecutor {
    /// Load one full-attention layer from a bound Qwen35 GGUF reader.
    pub fn for_layer(
        reader: &mut GgufTensorReader<'_>,
        config: &Qwen35Config,
        layer: usize,
    ) -> Result<Self, TargetOnlyError> {
        Self::for_layer_with_numerics(
            reader,
            config,
            layer,
            GgufProjectionNumerics::NativeF32Activation,
        )
    }

    /// Load one full-attention layer with an explicit projection ABI.
    pub fn for_layer_with_numerics(
        reader: &mut GgufTensorReader<'_>,
        config: &Qwen35Config,
        layer: usize,
        numerics: GgufProjectionNumerics,
    ) -> Result<Self, TargetOnlyError> {
        config.validate_exact_profile()?;
        if layer >= config.num_hidden_layers {
            return Err(invalid(
                "layer.index",
                format!(
                    "layer {layer} is outside {} layers",
                    config.num_hidden_layers
                ),
            ));
        }
        let spec = Qwen35LayerSpec::from_config(config, layer)?;
        if spec.kind != Qwen35LayerKind::FullAttention {
            return Err(invalid(
                "layer.kind",
                format!("layer {layer} is not full attention"),
            ));
        }
        let hidden = spec.hidden_size;
        let query_values = spec
            .query_heads
            .checked_mul(spec.head_dim)
            .ok_or(augur_model_qwen35::Qwen35Error::SizeOverflow)?;
        let kv_values = spec
            .kv_heads
            .checked_mul(spec.head_dim)
            .ok_or(augur_model_qwen35::Qwen35Error::SizeOverflow)?;
        let prefix = format!("blk.{layer}");
        let (projection_numerics, output_numerics, ffn_numerics) =
            split_layer_numerics(numerics, true);

        let attn_norm = load_dense(
            reader,
            &format!("{prefix}.attn_norm.weight"),
            &[hidden as u64],
            GGML_TYPE_F32,
        )?;
        let q = load_projection(
            reader,
            &format!("{prefix}.attn_q.weight"),
            &[hidden as u64, (query_values * 2) as u64],
            GGML_TYPE_Q8_0,
            projection_numerics,
        )?;
        let k = load_projection(
            reader,
            &format!("{prefix}.attn_k.weight"),
            &[hidden as u64, kv_values as u64],
            GGML_TYPE_Q8_0,
            projection_numerics,
        )?;
        let v = load_projection(
            reader,
            &format!("{prefix}.attn_v.weight"),
            &[hidden as u64, kv_values as u64],
            GGML_TYPE_Q8_0,
            projection_numerics,
        )?;
        let q_norm = load_dense(
            reader,
            &format!("{prefix}.attn_q_norm.weight"),
            &[spec.head_dim as u64],
            GGML_TYPE_F32,
        )?;
        let k_norm = load_dense(
            reader,
            &format!("{prefix}.attn_k_norm.weight"),
            &[spec.head_dim as u64],
            GGML_TYPE_F32,
        )?;
        let attn_output = load_projection(
            reader,
            &format!("{prefix}.attn_output.weight"),
            &[query_values as u64, hidden as u64],
            GGML_TYPE_Q8_0,
            output_numerics,
        )?;
        let post_attention_norm = load_dense(
            reader,
            &format!("{prefix}.post_attention_norm.weight"),
            &[hidden as u64],
            GGML_TYPE_F32,
        )?;
        let (ffn_type, ffn_gate_scale, ffn_up_scale, ffn_down_scale, ffn_input_scales) =
            load_ffn_quantization(reader, config, layer, &prefix)?;
        let ffn_gate = load_projection(
            reader,
            &format!("{prefix}.ffn_gate.weight"),
            &[hidden as u64, spec.intermediate_size as u64],
            ffn_type,
            ffn_numerics,
        )?;
        let ffn_up = load_projection(
            reader,
            &format!("{prefix}.ffn_up.weight"),
            &[hidden as u64, spec.intermediate_size as u64],
            ffn_type,
            ffn_numerics,
        )?;
        let ffn_down = load_projection(
            reader,
            &format!("{prefix}.ffn_down.weight"),
            &[spec.intermediate_size as u64, hidden as u64],
            ffn_type,
            ffn_numerics,
        )?;

        Ok(Self {
            layer,
            spec,
            attn_norm,
            q,
            k,
            v,
            q_norm,
            k_norm,
            attn_output,
            post_attention_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            ffn_gate_scale,
            ffn_up_scale,
            ffn_down_scale,
            ffn_input_scales,
        })
    }

    pub fn spec(&self) -> &Qwen35LayerSpec {
        &self.spec
    }

    pub fn layer_index(&self) -> usize {
        self.layer
    }

    pub fn ffn_input_scales(&self) -> [f32; 3] {
        self.ffn_input_scales
    }

    /// Number of bytes retained by the owned packed projection payloads.
    pub fn projection_payload_bytes(&self) -> usize {
        self.q
            .payload()
            .len()
            .saturating_add(self.k.payload().len())
            .saturating_add(self.v.payload().len())
            .saturating_add(self.attn_output.payload().len())
            .saturating_add(self.ffn_gate.payload().len())
            .saturating_add(self.ffn_up.payload().len())
            .saturating_add(self.ffn_down.payload().len())
    }

    pub fn forward_layer(
        &self,
        state: &mut Qwen35LayerState,
        hidden: &[f32],
        position: usize,
    ) -> Result<Qwen35LayerOutput, TargetOnlyError> {
        if !matches!(state, Qwen35LayerState::Full(_)) {
            return Err(invalid(
                "layer.state",
                "full-attention executor requires a full-attention state",
            ));
        }
        let weights = Qwen35LayerWeights::Full(Qwen35FullAttentionLayerWeights {
            attn_norm: &self.attn_norm,
            q: &self.q,
            k: &self.k,
            v: &self.v,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            attn_output: &self.attn_output,
            post_attention_norm: &self.post_attention_norm,
            ffn: Qwen35FfnWeights {
                gate: &self.ffn_gate,
                up: &self.ffn_up,
                down: &self.ffn_down,
                gate_multiplier: self.ffn_gate_scale,
                up_multiplier: self.ffn_up_scale,
                down_multiplier: self.ffn_down_scale,
            },
        });
        qwen35_layer_forward(&self.spec, &weights, state, hidden, position).map_err(Into::into)
    }
}

impl TargetOnlyLayerExecutor for GgufFullAttentionExecutor {
    fn forward(
        &mut self,
        layer: usize,
        spec: &Qwen35LayerSpec,
        state: &mut Qwen35LayerState,
        hidden: &[f32],
        position: usize,
    ) -> Result<Vec<f32>, TargetOnlyError> {
        if layer != self.layer {
            return Err(TargetOnlyError::Executor {
                layer,
                detail: format!(
                    "executor is bound to full-attention layer {} and cannot service another layer",
                    self.layer
                ),
            });
        }
        if spec != &self.spec {
            return Err(TargetOnlyError::Executor {
                layer,
                detail: format!(
                    "caller layer specification differs from bound full-attention layer {}",
                    self.layer
                ),
            });
        }
        Ok(self.forward_layer(state, hidden, position)?.hidden)
    }
}

/// A contiguous prefix of real-weight linear-attention layers.
///
/// Qwen35 places a full-attention layer at every fourth position (3, 7, ...).
/// This executor therefore accepts only prefixes that end before the first
/// unsupported full-attention layer.  It owns all loaded layers and dispatches
/// by the authenticated layer index supplied by [`TargetOnlySession`].
pub struct GgufLinearPrefixExecutor {
    layers: Vec<GgufLinearLayer0Executor>,
}

impl GgufLinearPrefixExecutor {
    /// Load a contiguous layer-0..`layer_count` prefix from one bound reader.
    ///
    /// A zero-length prefix is not a useful execution boundary.  Any prefix
    /// containing a full-attention layer is rejected before that layer's
    /// tensors are read; a dedicated grouped-query attention executor will be
    /// required before such a prefix can be extended.
    pub fn for_prefix(
        reader: &mut GgufTensorReader<'_>,
        config: &Qwen35Config,
        layer_count: usize,
    ) -> Result<Self, TargetOnlyError> {
        Self::for_prefix_with_numerics(
            reader,
            config,
            layer_count,
            GgufProjectionNumerics::NativeF32Activation,
        )
    }

    /// Load a linear prefix with an explicit projection ABI.
    pub fn for_prefix_with_numerics(
        reader: &mut GgufTensorReader<'_>,
        config: &Qwen35Config,
        layer_count: usize,
        numerics: GgufProjectionNumerics,
    ) -> Result<Self, TargetOnlyError> {
        config.validate_exact_profile()?;
        if layer_count == 0 {
            return Err(invalid(
                "prefix.layer_count",
                "must contain at least one layer",
            ));
        }
        if layer_count > config.num_hidden_layers {
            return Err(invalid(
                "prefix.layer_count",
                format!(
                    "requested {layer_count} layers but profile has {}",
                    config.num_hidden_layers
                ),
            ));
        }
        let mut layers = Vec::with_capacity(layer_count);
        for layer in 0..layer_count {
            layers.push(GgufLinearLayer0Executor::for_layer_with_numerics(
                reader, config, layer, numerics,
            )?);
        }
        Ok(Self { layers })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layers(&self) -> &[GgufLinearLayer0Executor] {
        &self.layers
    }

    /// Return cloned specs suitable for constructing a request-local session.
    ///
    /// Specs are small authenticated geometry records; returning owned values
    /// prevents a session from borrowing the executor and keeps the existing
    /// session ownership contract unchanged.
    pub fn specs(&self) -> Vec<Qwen35LayerSpec> {
        self.layers
            .iter()
            .map(|layer| layer.spec().clone())
            .collect()
    }

    pub fn projection_payload_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(GgufLinearLayer0Executor::projection_payload_bytes)
            .sum()
    }

    pub fn projection_payload_bytes_by_layer(&self) -> Vec<usize> {
        self.layers
            .iter()
            .map(GgufLinearLayer0Executor::projection_payload_bytes)
            .collect()
    }

    pub fn ffn_input_scales_by_layer(&self) -> Vec<[f32; 3]> {
        self.layers
            .iter()
            .map(GgufLinearLayer0Executor::ffn_input_scales)
            .collect()
    }
}

impl TargetOnlyLayerExecutor for GgufLinearPrefixExecutor {
    fn forward(
        &mut self,
        layer: usize,
        spec: &Qwen35LayerSpec,
        state: &mut Qwen35LayerState,
        hidden: &[f32],
        position: usize,
    ) -> Result<Vec<f32>, TargetOnlyError> {
        let layer_count = self.layers.len();
        let executor = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| TargetOnlyError::Executor {
                layer,
                detail: format!(
                    "prefix contains {} layers and cannot service layer {layer}",
                    layer_count
                ),
            })?;
        executor.forward(layer, spec, state, hidden, position)
    }
}

enum GgufHybridLayerExecutor {
    Linear(GgufLinearLayer0Executor),
    Full(GgufFullAttentionExecutor),
}

/// A contiguous real-weight prefix that can cross the Qwen35 attention
/// partition boundary.
///
/// Linear layers use [`GgufLinearLayer0Executor`] and full-attention layers
/// use [`GgufFullAttentionExecutor`].  The prefix still starts at layer zero;
/// arbitrary sparse layer sets are rejected by construction so the
/// `TargetOnlySession` layer/state indices remain authoritative.
pub struct GgufHybridPrefixExecutor {
    layers: Vec<GgufHybridLayerExecutor>,
}

impl GgufHybridPrefixExecutor {
    /// Load a contiguous layer-0..`layer_count` prefix, selecting the checked
    /// executor that matches each authenticated layer kind.
    pub fn for_prefix(
        reader: &mut GgufTensorReader<'_>,
        config: &Qwen35Config,
        layer_count: usize,
    ) -> Result<Self, TargetOnlyError> {
        Self::for_prefix_with_numerics(
            reader,
            config,
            layer_count,
            GgufProjectionNumerics::NativeF32Activation,
        )
    }

    /// Load a hybrid prefix with an explicit projection ABI.
    pub fn for_prefix_with_numerics(
        reader: &mut GgufTensorReader<'_>,
        config: &Qwen35Config,
        layer_count: usize,
        numerics: GgufProjectionNumerics,
    ) -> Result<Self, TargetOnlyError> {
        config.validate_exact_profile()?;
        if layer_count == 0 {
            return Err(invalid(
                "prefix.layer_count",
                "must contain at least one layer",
            ));
        }
        if layer_count > config.num_hidden_layers {
            return Err(invalid(
                "prefix.layer_count",
                format!(
                    "requested {layer_count} layers but profile has {}",
                    config.num_hidden_layers
                ),
            ));
        }
        let mut layers = Vec::with_capacity(layer_count);
        for layer in 0..layer_count {
            let spec = Qwen35LayerSpec::from_config(config, layer)?;
            let executor = match spec.kind {
                Qwen35LayerKind::LinearAttention => GgufHybridLayerExecutor::Linear(
                    GgufLinearLayer0Executor::for_layer_with_numerics(
                        reader, config, layer, numerics,
                    )?,
                ),
                Qwen35LayerKind::FullAttention => GgufHybridLayerExecutor::Full(
                    GgufFullAttentionExecutor::for_layer_with_numerics(
                        reader, config, layer, numerics,
                    )?,
                ),
            };
            layers.push(executor);
        }
        Ok(Self { layers })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn specs(&self) -> Vec<Qwen35LayerSpec> {
        self.layers
            .iter()
            .map(|layer| match layer {
                GgufHybridLayerExecutor::Linear(executor) => executor.spec().clone(),
                GgufHybridLayerExecutor::Full(executor) => executor.spec().clone(),
            })
            .collect()
    }

    pub fn layer_kinds(&self) -> Vec<Qwen35LayerKind> {
        self.layers
            .iter()
            .map(|layer| match layer {
                GgufHybridLayerExecutor::Linear(_) => Qwen35LayerKind::LinearAttention,
                GgufHybridLayerExecutor::Full(_) => Qwen35LayerKind::FullAttention,
            })
            .collect()
    }

    pub fn projection_payload_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| match layer {
                GgufHybridLayerExecutor::Linear(executor) => executor.projection_payload_bytes(),
                GgufHybridLayerExecutor::Full(executor) => executor.projection_payload_bytes(),
            })
            .sum()
    }

    pub fn projection_payload_bytes_by_layer(&self) -> Vec<usize> {
        self.layers
            .iter()
            .map(|layer| match layer {
                GgufHybridLayerExecutor::Linear(executor) => executor.projection_payload_bytes(),
                GgufHybridLayerExecutor::Full(executor) => executor.projection_payload_bytes(),
            })
            .collect()
    }

    pub fn ffn_input_scales_by_layer(&self) -> Vec<[f32; 3]> {
        self.layers
            .iter()
            .map(|layer| match layer {
                GgufHybridLayerExecutor::Linear(executor) => executor.ffn_input_scales(),
                GgufHybridLayerExecutor::Full(executor) => executor.ffn_input_scales(),
            })
            .collect()
    }

    /// Execute one layer while retaining every checked intermediate value.
    ///
    /// The target-only session normally needs only the final hidden, but a
    /// mixed CPU/CUDA qualification must compare the attention residual, norm,
    /// FFN output, and final hidden at each handoff.  This method keeps that
    /// instrumentation on the same authenticated executor and does not alter
    /// the ordinary session contract.
    pub fn forward_layer_output(
        &mut self,
        layer: usize,
        state: &mut Qwen35LayerState,
        hidden: &[f32],
        position: usize,
    ) -> Result<Qwen35LayerOutput, TargetOnlyError> {
        let layer_count = self.layers.len();
        let executor = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| TargetOnlyError::Executor {
                layer,
                detail: format!(
                    "hybrid prefix contains {} layers and cannot service layer {layer}",
                    layer_count
                ),
            })?;
        let spec = match executor {
            GgufHybridLayerExecutor::Linear(executor) => executor.spec(),
            GgufHybridLayerExecutor::Full(executor) => executor.spec(),
        };
        if spec.kind == Qwen35LayerKind::LinearAttention
            && !matches!(state, Qwen35LayerState::Linear { .. })
        {
            return Err(TargetOnlyError::Executor {
                layer,
                detail: "linear layer received a non-linear state".into(),
            });
        }
        if spec.kind == Qwen35LayerKind::FullAttention
            && !matches!(state, Qwen35LayerState::Full(_))
        {
            return Err(TargetOnlyError::Executor {
                layer,
                detail: "full-attention layer received a non-full state".into(),
            });
        }
        match executor {
            GgufHybridLayerExecutor::Linear(executor) => {
                executor.forward_layer(state, hidden, position)
            }
            GgufHybridLayerExecutor::Full(executor) => {
                executor.forward_layer(state, hidden, position)
            }
        }
    }
}

impl TargetOnlyLayerExecutor for GgufHybridPrefixExecutor {
    fn forward(
        &mut self,
        layer: usize,
        spec: &Qwen35LayerSpec,
        state: &mut Qwen35LayerState,
        hidden: &[f32],
        position: usize,
    ) -> Result<Vec<f32>, TargetOnlyError> {
        let layer_count = self.layers.len();
        let executor = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| TargetOnlyError::Executor {
                layer,
                detail: format!(
                    "hybrid prefix contains {} layers and cannot service layer {layer}",
                    layer_count
                ),
            })?;
        match executor {
            GgufHybridLayerExecutor::Linear(executor) => {
                executor.forward(layer, spec, state, hidden, position)
            }
            GgufHybridLayerExecutor::Full(executor) => {
                executor.forward(layer, spec, state, hidden, position)
            }
        }
    }
}

fn invalid(field: impl Into<String>, detail: impl Into<String>) -> TargetOnlyError {
    TargetOnlyError::Invalid {
        field: field.into(),
        detail: detail.into(),
    }
}

fn require_tensor(
    reader: &GgufTensorReader<'_>,
    name: &str,
    dimensions: &[u64],
    ggml_type: u32,
) -> Result<GgufTensor, TargetOnlyError> {
    let descriptor = reader
        .tensor(name)
        .ok_or_else(|| TargetOnlyError::Model(format!("missing tensor {name:?}")))?;
    if descriptor.dimensions != dimensions {
        return Err(invalid(
            format!("{name}.dimensions"),
            format!("expected {dimensions:?}, got {:?}", descriptor.dimensions),
        ));
    }
    if descriptor.ggml_type != ggml_type {
        return Err(invalid(
            format!("{name}.ggml_type"),
            format!("expected {ggml_type}, got {}", descriptor.ggml_type),
        ));
    }
    Ok(descriptor.clone())
}

fn split_layer_numerics(
    numerics: GgufProjectionNumerics,
    full_attention: bool,
) -> (
    GgufProjectionNumerics,
    GgufProjectionNumerics,
    GgufProjectionNumerics,
) {
    if numerics == GgufProjectionNumerics::GgmlFfnQ8_0Activation {
        (
            GgufProjectionNumerics::NativeF32Activation,
            GgufProjectionNumerics::NativeF32Activation,
            GgufProjectionNumerics::GgmlQ8_0Activation,
        )
    } else if numerics == GgufProjectionNumerics::GgmlOutputFfnQ8_0Activation {
        (
            if full_attention {
                GgufProjectionNumerics::GgmlQ8_0Activation
            } else {
                GgufProjectionNumerics::NativeF32Activation
            },
            GgufProjectionNumerics::GgmlQ8_0Activation,
            GgufProjectionNumerics::GgmlQ8_0Activation,
        )
    } else {
        (numerics, numerics, numerics)
    }
}

fn load_projection(
    reader: &mut GgufTensorReader<'_>,
    name: &str,
    dimensions: &[u64],
    ggml_type: u32,
    numerics: GgufProjectionNumerics,
) -> Result<OwnedGgufProjection, TargetOnlyError> {
    let descriptor = require_tensor(reader, name, dimensions, ggml_type)?;
    OwnedGgufProjection::from_reader_with_numerics(reader, &descriptor.name, 1.0, 1.0, numerics)
        .map_err(Into::into)
}

/// Load the FFN quantization metadata for one authenticated layer.
///
/// The audited artifact is deliberately mixed: layers 0..55 use NVFP4 and
/// carry tensor/input scales, while layers 56..63 use plain Q8_0 and carry no
/// companion scale tensors.  Keeping this decision in one helper prevents the
/// linear and full-attention executors from drifting apart at the boundary.
fn load_ffn_quantization(
    reader: &mut GgufTensorReader<'_>,
    config: &Qwen35Config,
    layer: usize,
    prefix: &str,
) -> Result<(u32, f32, f32, f32, [f32; 3]), TargetOnlyError> {
    let ffn_type = Qwen35TensorSchema::expected_ffn_weight_ggml_type(config, layer)?;
    match ffn_type {
        GGML_TYPE_NVFP4 => {
            let gate_scale = load_positive_scalar(reader, &format!("{prefix}.ffn_gate.scale"))?;
            let up_scale = load_positive_scalar(reader, &format!("{prefix}.ffn_up.scale"))?;
            let down_scale = load_positive_scalar(reader, &format!("{prefix}.ffn_down.scale"))?;
            // Input scales are audited and retained for the receipt, but the
            // existing fused-NVFP4 reference policy intentionally does not
            // apply them. Applying them would change the numerical ABI.
            let input_scales = [
                load_positive_scalar(reader, &format!("{prefix}.ffn_gate.input_scale"))?,
                load_positive_scalar(reader, &format!("{prefix}.ffn_up.input_scale"))?,
                load_positive_scalar(reader, &format!("{prefix}.ffn_down.input_scale"))?,
            ];
            Ok((ffn_type, gate_scale, up_scale, down_scale, input_scales))
        }
        GGML_TYPE_Q8_0 => Ok((ffn_type, 1.0, 1.0, 1.0, [1.0; 3])),
        other => Err(invalid(
            format!("layer.{layer}.ffn.ggml_type"),
            format!("unsupported authenticated FFN type {other}"),
        )),
    }
}

fn load_dense(
    reader: &mut GgufTensorReader<'_>,
    name: &str,
    dimensions: &[u64],
    ggml_type: u32,
) -> Result<Vec<f32>, TargetOnlyError> {
    let descriptor = require_tensor(reader, name, dimensions, ggml_type)?;
    let payload = reader
        .read_tensor_bytes(&descriptor.name)
        .map_err(TargetOnlyError::from)?;
    let values = dequantize_tensor(&descriptor, &payload).map_err(TargetOnlyError::from)?;
    let expected = dimensions.iter().try_fold(1usize, |count, dimension| {
        count.checked_mul(usize::try_from(*dimension).ok()?)
    });
    if expected != Some(values.len()) || values.iter().any(|value| !value.is_finite()) {
        return Err(TargetOnlyError::Model(format!(
            "tensor {name:?} decoded values do not match finite geometry"
        )));
    }
    Ok(values)
}

fn load_positive_scalar(
    reader: &mut GgufTensorReader<'_>,
    name: &str,
) -> Result<f32, TargetOnlyError> {
    let values = load_dense(reader, name, &[1], GGML_TYPE_F32)?;
    let value = values[0];
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(name, "scale must be finite and positive"));
    }
    Ok(value)
}
