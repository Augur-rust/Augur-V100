//! Same-input diagnostic for the full-attention output projection boundary.
//! This intentionally does not establish trajectory qualification.

use super::*;
use augur_model_qwen35::quantized_matvec_activation_q8_0;

pub(super) struct FullAttentionIntermediateReadback {
    pub(super) position: usize,
    pub(super) attention: Vec<f32>,
    pub(super) gated: Vec<f32>,
    pub(super) query: Vec<f32>,
    pub(super) gate: Option<Vec<f32>>,
    pub(super) key_cache: Vec<f32>,
    pub(super) value_cache: Vec<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FullAttentionOperatorAuditReport {
    pub layer: usize,
    pub token: usize,
    pub reference: &'static str,
    pub qualification_permitted: bool,
    pub gated_input_sha256: String,
    pub attention_from_same_input: Option<NumericalReport>,
    pub gated_from_same_input: Option<NumericalReport>,
    pub output_projection_from_same_input: NumericalReport,
}

pub(super) fn readback(
    plan: &DevicePlan,
    stream: &OwnedCudaStream,
    position: usize,
) -> Result<Option<FullAttentionIntermediateReadback>, Qwen35MixedPrefixCudaError> {
    let DevicePlan::Full(plan) = plan else {
        return Ok(None);
    };
    let receipt = stream.submission_receipt()?;
    Ok(Some(FullAttentionIntermediateReadback {
        position,
        attention: read_f32_buffer(stream, &plan.attention, receipt)?,
        gated: read_f32_buffer(stream, &plan.gated, receipt)?,
        query: read_f32_buffer(stream, &plan.query, receipt)?,
        #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
        gate: Some(read_f32_buffer(stream, &plan.gate, receipt)?),
        #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
        gate: None,
        key_cache: read_full_cache_buffer(stream, &plan.key_cache, receipt)?,
        value_cache: read_full_cache_buffer(stream, &plan.value_cache, receipt)?,
    }))
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let e = value.exp();
        e / (1.0 + e)
    }
}

fn reference_gated(
    query: &[f32],
    gate: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    positions: usize,
) -> Result<(Vec<f32>, Vec<f32>), Qwen35MixedPrefixCudaError> {
    if query_heads == 0
        || kv_heads == 0
        || !query_heads.is_multiple_of(kv_heads)
        || head_dim == 0
        || positions == 0
        || query.len() != query_heads * head_dim
        || gate.len() != query.len()
        || key_cache.len() < positions * kv_heads * head_dim
        || value_cache.len() < positions * kv_heads * head_dim
    {
        return Err(invalid("full-attention same-input audit geometry"));
    }
    let group = query_heads / kv_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut attention = vec![0.0; query.len()];
    for head in 0..query_heads {
        let kv_head = head / group;
        let q0 = head * head_dim;
        let mut scores = vec![0.0; positions];
        for (position, score) in scores.iter_mut().enumerate() {
            let k0 = (position * kv_heads + kv_head) * head_dim;
            let mut dot = 0.0;
            for dim in 0..head_dim {
                dot += query[q0 + dim] * key_cache[k0 + dim];
            }
            *score = dot * scale;
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0.0;
        for score in &mut scores {
            *score = (*score - maximum).exp();
            denominator += *score;
        }
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(invalid("full-attention same-input audit softmax"));
        }
        for score in &mut scores {
            *score /= denominator;
        }
        for (position, probability) in scores.into_iter().enumerate() {
            let v0 = (position * kv_heads + kv_head) * head_dim;
            for dim in 0..head_dim {
                attention[q0 + dim] += probability * value_cache[v0 + dim];
            }
        }
    }
    let gated = attention
        .iter()
        .zip(gate)
        .map(|(&value, &control)| value * sigmoid(control))
        .collect::<Vec<_>>();
    if attention.iter().any(|v| !v.is_finite()) || gated.iter().any(|v| !v.is_finite()) {
        return Err(invalid("full-attention same-input audit nonfinite"));
    }
    Ok((attention, gated))
}

pub(super) fn audit(
    material: &LayerMaterial,
    actual: &DeviceLayerReadback,
    layer: usize,
    token: usize,
) -> Result<FullAttentionOperatorAuditReport, Qwen35MixedPrefixCudaError> {
    let material = match material {
        LayerMaterial::Full(material) => material,
        LayerMaterial::Linear(_) => {
            return Err(invalid("full-attention audit received linear material"));
        }
    };
    let readback = actual
        .full_attention_intermediates
        .as_ref()
        .ok_or_else(|| invalid("full-attention audit readback missing"))?;
    let local = readback.gate.as_ref().map(|gate| {
        reference_gated(
            &readback.query,
            gate,
            &readback.key_cache,
            &readback.value_cache,
            material.spec.query_heads,
            material.spec.kv_heads,
            material.spec.head_dim,
            readback.position + 1,
        )
    });
    let (attention_expected, gated_expected) = match local {
        Some(result) => {
            let (attention, gated) = result?;
            (Some(attention), Some(gated))
        }
        None => (None, None),
    };
    let expected = quantized_matvec_activation_q8_0(
        &material.attn_output.descriptor,
        &material.attn_output.payload,
        &readback.gated,
    )?;
    if expected.iter().any(|value| !value.is_finite()) {
        return Err(invalid("full-attention audit projection is nonfinite"));
    }
    Ok(FullAttentionOperatorAuditReport {
        layer,
        token,
        reference: "cpu_ggml_q8_0_same_gpu_gated_input_output_projection_only",
        qualification_permitted: false,
        gated_input_sha256: sha256_f32_bits(&readback.gated),
        attention_from_same_input: attention_expected.as_ref().map(|expected| {
            compare(
                &format!("layer{layer}.token{token}.full_attention.attention"),
                &readback.attention,
                expected,
                1.0e-2,
                5.0e-4,
            )
        }),
        gated_from_same_input: gated_expected.as_ref().map(|expected| {
            compare(
                &format!("layer{layer}.token{token}.full_attention.gated"),
                &readback.gated,
                expected,
                1.0e-2,
                5.0e-4,
            )
        }),
        output_projection_from_same_input: compare(
            &format!("layer{layer}.token{token}.full_attention.output_projection"),
            &actual.attention_output,
            &expected,
            1.0e-2,
            5.0e-4,
        ),
    })
}
