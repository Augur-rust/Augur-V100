//! Opt-in diagnostic: separate operator error from incoming trajectory drift.
//! No tolerances or qualification bits in the parent runner are changed here.

use super::*;
use augur_model_qwen35::{quantize_q8_0_activation, quantized_matvec_activation_q8_0};

pub(super) struct FfnIntermediateReadback {
    gate: Vec<f32>,
    up: Vec<f32>,
    activation: Vec<f32>,
    input_q8_1: Vec<u8>,
    down_q8_1: Vec<u8>,
}

#[derive(Clone, Debug, Serialize)]
pub struct QuantizerAuditReport {
    pub blocks: usize,
    pub mismatched_scale_blocks: usize,
    pub mismatched_codes: usize,
    pub first_mismatched_code: Option<usize>,
    pub gpu_q8_1_sha256: String,
    pub cpu_q8_0_sha256: String,
    pub sum_metadata_compared: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct FfnOperatorAuditReport {
    pub layer: usize,
    pub token: usize,
    pub reference: &'static str,
    pub qualification_permitted: bool,
    pub input_sha256: String,
    pub gate_from_same_input: NumericalReport,
    pub up_from_same_input: NumericalReport,
    pub swiglu_from_gpu_gate_up: NumericalReport,
    pub activation_from_same_input: NumericalReport,
    pub down_from_gpu_activation: NumericalReport,
    pub ffn_from_same_input: NumericalReport,
    pub input_quantizer: QuantizerAuditReport,
    pub down_quantizer: QuantizerAuditReport,
    /// Quantizer comparison after propagating the CPU gate/up/SwiGLU values,
    /// not after resetting the CPU quantizer to the GPU activation. A mismatch
    /// here diagnoses composition drift; it is not a quantizer implementation failure.
    pub down_quantizer_from_same_input_chain: QuantizerAuditReport,
}

pub(super) fn readback(
    plan: &DevicePlan,
    stream: &OwnedCudaStream,
) -> Result<FfnIntermediateReadback, Qwen35MixedPrefixCudaError> {
    if cfg!(feature = "qwen35-fused-ffn-swiglu") {
        return Err(invalid(
            "fused SwiGLU has no materialized gate/up audit buffers",
        ));
    }
    let (gate, up, activation, input_q8_1, down_q8_1) = match plan {
        DevicePlan::Linear(p) => (
            &p.ffn_gate_output,
            &p.ffn_up_output,
            &p.ffn_activation,
            &p.ffn_input_q8_1,
            &p.ffn_down_q8_1,
        ),
        DevicePlan::Full(p) => (
            &p.ffn_gate_output,
            &p.ffn_up_output,
            &p.ffn_activation,
            &p.ffn_input_q8_1,
            &p.ffn_down_q8_1,
        ),
    };
    // execute() has already obtained an owner/epoch completion proof. These
    // readbacks bind the unchanged current receipt and are charged separately.
    let receipt = stream.submission_receipt()?;
    let read_bytes = |buffer: &OwnedCudaBuffer| -> Result<Vec<u8>, Qwen35MixedPrefixCudaError> {
        let mut bytes = vec![0; buffer.len_bytes()];
        buffer.copy_to_host_sync_with_submission_proof(stream, &mut bytes, receipt)?;
        Ok(bytes)
    };
    Ok(FfnIntermediateReadback {
        gate: read_f32_buffer(stream, gate, receipt)?,
        up: read_f32_buffer(stream, up, receipt)?,
        activation: read_f32_buffer(stream, activation, receipt)?,
        input_q8_1: read_bytes(input_q8_1)?,
        down_q8_1: read_bytes(down_q8_1)?,
    })
}

pub(super) fn quantizer_audit(
    input: &[f32],
    gpu: &[u8],
) -> Result<QuantizerAuditReport, Qwen35MixedPrefixCudaError> {
    let cpu = quantize_q8_0_activation(input)?;
    let blocks = input.len() / 32;
    if gpu.len() != blocks * 36 || cpu.len() != blocks * 34 {
        return Err(invalid("FFN audit Q8_0/Q8_1 extent mismatch"));
    }
    let mut mismatched_scale_blocks = 0;
    let mut mismatched_codes = 0;
    let mut first_mismatched_code = None;
    for (index, (gpu, cpu)) in gpu.chunks_exact(36).zip(cpu.chunks_exact(34)).enumerate() {
        mismatched_scale_blocks += usize::from(gpu[..2] != cpu[..2]);
        for code in 0..32 {
            if gpu[4 + code] != cpu[2 + code] {
                mismatched_codes += 1;
                first_mismatched_code.get_or_insert(index * 32 + code);
            }
        }
    }
    Ok(QuantizerAuditReport {
        blocks,
        mismatched_scale_blocks,
        mismatched_codes,
        first_mismatched_code,
        gpu_q8_1_sha256: sha256_hex(gpu),
        cpu_q8_0_sha256: sha256_hex(&cpu),
        // Q8_0 has no sum metadata; Q8_0/NVFP4 matvec does not consume Q8_1 sum.
        sum_metadata_compared: false,
    })
}

fn swiglu(gate: &[f32], up: &[f32]) -> Result<Vec<f32>, Qwen35MixedPrefixCudaError> {
    if gate.len() != up.len() || gate.is_empty() {
        return Err(invalid("FFN audit gate/up shape mismatch"));
    }
    let values = gate
        .iter()
        .zip(up)
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect::<Vec<_>>();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("FFN audit activation is nonfinite"));
    }
    Ok(values)
}

fn projection(
    weight: &LoadedTensor,
    input: &[f32],
    scale: f32,
) -> Result<Vec<f32>, Qwen35MixedPrefixCudaError> {
    let mut result = quantized_matvec_activation_q8_0(&weight.descriptor, &weight.payload, input)?;
    for value in &mut result {
        *value *= scale;
    }
    if result.iter().any(|v| !v.is_finite()) {
        return Err(invalid("FFN audit projection is nonfinite"));
    }
    Ok(result)
}

pub(super) fn audit(
    material: &LayerMaterial,
    actual: &DeviceLayerReadback,
    layer: usize,
    token: usize,
) -> Result<FfnOperatorAuditReport, Qwen35MixedPrefixCudaError> {
    let ffn = match material {
        LayerMaterial::Linear(m) => &m.ffn,
        LayerMaterial::Full(m) => &m.ffn,
    };
    let gpu = actual
        .ffn_intermediates
        .as_ref()
        .ok_or_else(|| invalid("FFN audit readback missing"))?;
    let gate = projection(&ffn.gate, &actual.post_norm, ffn.gate_scale)?;
    let up = projection(&ffn.up, &actual.post_norm, ffn.up_scale)?;
    let activation = swiglu(&gate, &up)?;
    let activation_local = swiglu(&gpu.gate, &gpu.up)?;
    let down_local = projection(&ffn.down, &gpu.activation, ffn.down_scale)?;
    let output = projection(&ffn.down, &activation, ffn.down_scale)?;
    let cmp = |name: &str, a: &[f32], e: &[f32]| {
        compare(
            &format!("layer{layer}.token{token}.ffn_operator.{name}"),
            a,
            e,
            1.0e-2,
            5.0e-4,
        )
    };
    Ok(FfnOperatorAuditReport {
        layer,
        token,
        reference: "cpu_ggml_q8_0_same_gpu_input_operator_audit_not_trajectory_qualification",
        qualification_permitted: false,
        input_sha256: sha256_f32_bits(&actual.post_norm),
        gate_from_same_input: cmp("gate", &gpu.gate, &gate),
        up_from_same_input: cmp("up", &gpu.up, &up),
        swiglu_from_gpu_gate_up: cmp("swiglu_local", &gpu.activation, &activation_local),
        activation_from_same_input: cmp("activation", &gpu.activation, &activation),
        down_from_gpu_activation: cmp("down_local", &actual.ffn_output, &down_local),
        ffn_from_same_input: cmp("ffn_same_input", &actual.ffn_output, &output),
        input_quantizer: quantizer_audit(&actual.post_norm, &gpu.input_q8_1)?,
        down_quantizer: quantizer_audit(&gpu.activation, &gpu.down_q8_1)?,
        down_quantizer_from_same_input_chain: quantizer_audit(&activation, &gpu.down_q8_1)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantizer_audit_ignores_sum_but_detects_scale_code_and_extent_drift() {
        let input = (0..32)
            .map(|x| (x as f32 - 16.0) / 17.0)
            .collect::<Vec<_>>();
        let cpu = quantize_q8_0_activation(&input).unwrap();
        let mut gpu = vec![0; 36];
        gpu[..2].copy_from_slice(&cpu[..2]);
        gpu[2..4].copy_from_slice(&[7, 19]);
        gpu[4..].copy_from_slice(&cpu[2..]);
        let audit = quantizer_audit(&input, &gpu).unwrap();
        assert_eq!(audit.mismatched_scale_blocks, 0);
        assert_eq!(audit.mismatched_codes, 0);
        gpu[0] ^= 1;
        gpu[11] ^= 1;
        let audit = quantizer_audit(&input, &gpu).unwrap();
        assert_eq!(audit.mismatched_scale_blocks, 1);
        assert_eq!(audit.mismatched_codes, 1);
        assert_eq!(audit.first_mismatched_code, Some(7));
        assert!(quantizer_audit(&input, &gpu[..35]).is_err());
    }

    #[test]
    fn swiglu_audit_rejects_shape_and_nonfinite_inputs() {
        assert!(swiglu(&[1.0], &[]).is_err());
        assert!(swiglu(&[f32::NAN], &[1.0]).is_err());
        assert_eq!(swiglu(&[0.0], &[2.0]).unwrap(), vec![0.0]);
    }
}
