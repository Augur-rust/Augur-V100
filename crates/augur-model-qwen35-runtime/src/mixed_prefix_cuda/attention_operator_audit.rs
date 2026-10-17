//! Same-input diagnostics, never an alternative trajectory qualification.
//! Every recurrence is restored from the preceding GPU readback, not the
//! free-running CPU oracle. This isolates local error from accumulated drift.

use super::*;
use augur_model_qwen35::{
    l2_norm_last_dim, quantize_q8_0_activation, quantized_matvec_activation_q8_0,
    quantized_matvec_fused_simd, rms_norm_gated_last_dim, rms_norm_last_dim, silu,
    CausalConv1dState, GatedDeltaNetState, ReferenceTensor,
};

pub(super) struct AttentionIntermediateReadback {
    normalized: Option<Vec<f32>>,
    qkv: Vec<f32>,
    z: Vec<f32>,
    decay: Vec<f32>,
    beta: Vec<f32>,
    activated: Vec<f32>,
    recurrent: Vec<f32>,
    gated: Vec<f32>,
    output_q8_1: Vec<u8>,
}

#[derive(Clone, Debug, Serialize)]
pub struct QuantizedTrajectoryDrift {
    pub compared_blocks: usize,
    pub different_scale_blocks: usize,
    pub different_codes: usize,
    pub first_different_code: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AttentionOperatorAuditReport {
    pub layer: usize,
    pub token: usize,
    pub reference: &'static str,
    pub qualification_permitted: bool,
    pub previous_state_source: &'static str,
    pub input_sha256: String,
    pub local_operators: Vec<NumericalReport>,
    pub chain_same_projection_inputs: Vec<NumericalReport>,
    pub output_quantizer: ffn_operator_audit::QuantizerAuditReport,
    pub ssm_input_chain_quantization: QuantizedTrajectoryDrift,
    pub ffn_input_trajectory_quantization: QuantizedTrajectoryDrift,
}

pub(super) fn readback(
    plan: &DevicePlan,
    stream: &OwnedCudaStream,
) -> Result<Option<AttentionIntermediateReadback>, Qwen35MixedPrefixCudaError> {
    let DevicePlan::Linear(p) = plan else {
        return Ok(None);
    };
    // execute() has already sealed completion for this submission. Audit reads
    // add no kernels and are included in the parent's readback boundary ledger.
    let receipt = stream.submission_receipt()?;
    let read = |b| read_f32_buffer(stream, b, receipt);
    let mut output_q8_1 = vec![0; p.output_q8_1.len_bytes()];
    p.output_q8_1
        .copy_to_host_sync_with_submission_proof(stream, &mut output_q8_1, receipt)?;
    Ok(Some(AttentionIntermediateReadback {
        normalized: p
            .projection
            .as_ref()
            .map(|v| read(&v.normalized))
            .transpose()?,
        qkv: read(&p.qkv)?,
        z: read(&p.z)?,
        decay: read(&p.decay)?,
        beta: read(&p.beta)?,
        activated: read(&p.activated)?,
        recurrent: read(&p.recurrent)?,
        gated: read(&p.linear_output)?,
        output_q8_1,
    }))
}

fn tensor(values: &[f32], width: usize) -> Result<ReferenceTensor, Qwen35MixedPrefixCudaError> {
    if width == 0 || values.is_empty() || !values.len().is_multiple_of(width) {
        return Err(invalid("attention audit tensor shape mismatch"));
    }
    Ok(ReferenceTensor::new(
        vec![values.len() / width, width],
        values.to_vec(),
    )?)
}

fn norm(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, Qwen35MixedPrefixCudaError> {
    Ok(rms_norm_last_dim(
        &tensor(input, weight.len())?,
        &ReferenceTensor::new(vec![weight.len()], weight.to_vec())?,
        epsilon,
    )?
    .values()
    .to_vec())
}

fn projection(
    weight: &LoadedTensor,
    input: &[f32],
    q8: bool,
) -> Result<Vec<f32>, Qwen35MixedPrefixCudaError> {
    Ok(if q8 {
        quantized_matvec_activation_q8_0(&weight.descriptor, &weight.payload, input)?
    } else {
        quantized_matvec_fused_simd(&weight.descriptor, &weight.payload, input)?
    })
}

fn quantized_drift(
    actual: &[f32],
    expected: &[f32],
) -> Result<QuantizedTrajectoryDrift, Qwen35MixedPrefixCudaError> {
    if actual.len() != expected.len() || actual.is_empty() || !actual.len().is_multiple_of(32) {
        return Err(invalid(
            "attention audit trajectory quantizer shape mismatch",
        ));
    }
    let actual = quantize_q8_0_activation(actual)?;
    let expected = quantize_q8_0_activation(expected)?;
    let mut report = QuantizedTrajectoryDrift {
        compared_blocks: actual.len() / 34,
        different_scale_blocks: 0,
        different_codes: 0,
        first_different_code: None,
    };
    for (block, (a, e)) in actual
        .chunks_exact(34)
        .zip(expected.chunks_exact(34))
        .enumerate()
    {
        report.different_scale_blocks += usize::from(a[..2] != e[..2]);
        for index in 0..32 {
            if a[index + 2] != e[index + 2] {
                report.different_codes += 1;
                report
                    .first_different_code
                    .get_or_insert(block * 32 + index);
            }
        }
    }
    Ok(report)
}

pub(super) fn audit(
    material: &LayerMaterial,
    actual: &DeviceLayerReadback,
    previous: Option<&DeviceLayerReadback>,
    expected: &ExpectedLayerResult,
    layer: usize,
    token: usize,
    epsilon: f32,
) -> Result<AttentionOperatorAuditReport, Qwen35MixedPrefixCudaError> {
    if (token == 0) != previous.is_none() {
        return Err(invalid(
            "attention audit missing or unexpected preceding GPU state",
        ));
    }
    let LayerMaterial::Linear(m) = material else {
        return Err(invalid("attention audit requires a linear layer"));
    };
    let gpu = actual
        .attention_intermediates
        .as_ref()
        .ok_or_else(|| invalid("attention audit readback missing"))?;
    let mut local_operators = Vec::new();
    let mut record = |name: &str, a: &[f32], e: &[f32]| {
        local_operators.push(compare(
            &format!("layer{layer}.token{token}.attention_operator.{name}"),
            a,
            e,
            1.0e-2,
            5.0e-4,
        ));
    };
    let normalized_cpu = norm(&actual.input_hidden, &m.attn_norm, epsilon)?;
    let normalized = gpu.normalized.as_deref().unwrap_or(&normalized_cpu);
    if let Some(value) = &gpu.normalized {
        record("input_norm_local", value, &normalized_cpu);
    }
    let q8 = gpu.normalized.is_some();
    let qkv = projection(&m.qkv, normalized, q8)?;
    let z = projection(&m.z, normalized, q8)?;
    let alpha = projection(&m.alpha, normalized, false)?;
    let beta_raw = projection(&m.beta, normalized, false)?;
    let mut decay = Vec::with_capacity(linear::VALUE_HEADS);
    let mut beta = Vec::with_capacity(linear::VALUE_HEADS);
    for index in 0..linear::VALUE_HEADS {
        let x = alpha[index] + m.dt_bias[index];
        let softplus = if x > 20.0 {
            x
        } else if x < -20.0 {
            x.exp()
        } else {
            (1.0 + x.exp()).ln()
        };
        decay.push(softplus * m.ssm_a[index]);
        let b = beta_raw[index];
        beta.push(if b >= 0.0 {
            1.0 / (1.0 + (-b).exp())
        } else {
            let e = b.exp();
            e / (1.0 + e)
        });
    }
    record("qkv_same_normalized", &gpu.qkv, &qkv);
    record("z_same_normalized", &gpu.z, &z);
    record("decay_same_normalized", &gpu.decay, &decay);
    record("beta_same_normalized", &gpu.beta, &beta);

    let mut conv = CausalConv1dState::zeros(linear::KERNEL_SIZE, linear::QKV_ELEMENTS)?;
    let mut recurrent_state = GatedDeltaNetState::zeros(linear::VALUE_HEADS, linear::HEAD_DIM)?;
    if let Some(p) = previous {
        conv.restore(&p.history)?;
        recurrent_state.restore(&p.state)?;
    }
    let conv_output = conv.step(&gpu.qkv, &m.conv_weights)?;
    let activated = silu(&tensor(&conv_output, linear::QKV_ELEMENTS)?)?;
    record(
        "conv_silu_same_qkv_history",
        &gpu.activated,
        activated.values(),
    );
    record("conv_history", &actual.history, conv.history());
    let key_values = linear::KEY_HEADS * linear::HEAD_DIM;
    let q = l2_norm_last_dim(
        &tensor(&gpu.activated[..key_values], linear::HEAD_DIM)?,
        epsilon,
    )?;
    let k = l2_norm_last_dim(
        &tensor(&gpu.activated[key_values..2 * key_values], linear::HEAD_DIM)?,
        epsilon,
    )?;
    let recurrent = recurrent_state.step(
        q.values(),
        k.values(),
        &gpu.activated[2 * key_values..],
        &gpu.decay,
        &gpu.beta,
    )?;
    record("recurrent_same_activated_state", &gpu.recurrent, &recurrent);
    record(
        "state_same_activated_state",
        &actual.state,
        recurrent_state.values(),
    );
    let gated = rms_norm_gated_last_dim(
        &tensor(&gpu.recurrent, linear::HEAD_DIM)?,
        &ReferenceTensor::new(vec![linear::HEAD_DIM], m.ssm_norm.clone())?,
        &tensor(&gpu.z, linear::HEAD_DIM)?,
        epsilon,
    )?;
    record("ssm_norm_gate_same_recurrent_z", &gpu.gated, gated.values());
    let attention = projection(&m.ssm_out, &gpu.gated, true)?;
    record(
        "ssm_output_same_gated",
        &actual.attention_output,
        &attention,
    );
    if actual.input_hidden.len() != actual.attention_output.len() {
        return Err(invalid("attention audit residual shape mismatch"));
    }
    let residual = actual
        .input_hidden
        .iter()
        .zip(&actual.attention_output)
        .map(|(h, a)| h + a)
        .collect::<Vec<_>>();
    record(
        "residual_same_hidden_attention",
        &actual.attention_residual,
        &residual,
    );
    let post_norm = norm(&actual.attention_residual, &m.post_norm, epsilon)?;
    record("post_norm_same_residual", &actual.post_norm, &post_norm);

    // Second axis: a complete attention subchain with the SAME projected
    // inputs and previous GPU state. Unlike the local checks above, it does
    // not reset to GPU intermediates at every operator. Any additional jump
    // across Q8 here is therefore a measured composition effect.
    let mut chain_state = GatedDeltaNetState::zeros(linear::VALUE_HEADS, linear::HEAD_DIM)?;
    if let Some(p) = previous {
        chain_state.restore(&p.state)?;
    }
    let chain_q = l2_norm_last_dim(
        &tensor(&activated.values()[..key_values], linear::HEAD_DIM)?,
        epsilon,
    )?;
    let chain_k = l2_norm_last_dim(
        &tensor(
            &activated.values()[key_values..2 * key_values],
            linear::HEAD_DIM,
        )?,
        epsilon,
    )?;
    let chain_recurrent = chain_state.step(
        chain_q.values(),
        chain_k.values(),
        &activated.values()[2 * key_values..],
        &gpu.decay,
        &gpu.beta,
    )?;
    let chain_gated = rms_norm_gated_last_dim(
        &tensor(&chain_recurrent, linear::HEAD_DIM)?,
        &ReferenceTensor::new(vec![linear::HEAD_DIM], m.ssm_norm.clone())?,
        &tensor(&gpu.z, linear::HEAD_DIM)?,
        epsilon,
    )?;
    let chain_attention = projection(&m.ssm_out, chain_gated.values(), true)?;
    let chain_cmp = |name: &str, a: &[f32], e: &[f32]| {
        compare(
            &format!("layer{layer}.token{token}.attention_chain.{name}"),
            a,
            e,
            1.0e-2,
            5.0e-4,
        )
    };
    Ok(AttentionOperatorAuditReport {
        layer,
        token,
        reference:
            "cpu_same_gpu_operator_inputs_and_preceding_gpu_state_not_trajectory_qualification",
        qualification_permitted: false,
        previous_state_source: if token == 0 {
            "zero_initial_state"
        } else {
            "previous_gpu_readback"
        },
        input_sha256: sha256_f32_bits(&actual.input_hidden),
        local_operators,
        chain_same_projection_inputs: vec![
            chain_cmp("recurrent", &gpu.recurrent, &chain_recurrent),
            chain_cmp("gated", &gpu.gated, chain_gated.values()),
            chain_cmp("ssm_output_q8", &actual.attention_output, &chain_attention),
        ],
        output_quantizer: ffn_operator_audit::quantizer_audit(&gpu.gated, &gpu.output_q8_1)?,
        ssm_input_chain_quantization: quantized_drift(&gpu.gated, chain_gated.values())?,
        ffn_input_trajectory_quantization: quantized_drift(
            &actual.post_norm,
            &expected.output.post_attention_norm,
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantization_drift_separates_input_roundoff_from_code_changes() {
        let mut x = vec![0.0; 32];
        x[0] = 1.0;
        let mut y = x.clone();
        y[1] = 1.0e-8;
        let near = quantized_drift(&x, &y).unwrap();
        assert_eq!(near.different_codes, 0);
        assert_eq!(near.different_scale_blocks, 0);
        y[1] = 0.25;
        let drift = quantized_drift(&x, &y).unwrap();
        assert_eq!(drift.different_codes, 1);
        assert_eq!(drift.first_different_code, Some(1));
        assert_eq!(drift.different_scale_blocks, 0);
        y[0] = 2.0;
        assert_eq!(quantized_drift(&x, &y).unwrap().different_scale_blocks, 1);
    }

    #[test]
    fn audit_shape_and_nonfinite_fail_closed() {
        assert!(quantized_drift(&[], &[]).is_err());
        assert!(quantized_drift(&[0.0; 31], &[0.0; 31]).is_err());
        assert!(quantized_drift(&[0.0; 32], &[0.0; 64]).is_err());
        assert!(quantized_drift(&[f32::NAN; 32], &[0.0; 32]).is_err());
        assert!(tensor(&[0.0; 5], 2).is_err());
        assert!(tensor(&[], 0).is_err());
    }
}
