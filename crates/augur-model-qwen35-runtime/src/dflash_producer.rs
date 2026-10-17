//! Real-weight DFlash producer prefix.
//!
//! This module is deliberately one step short of a decoder.  It binds the
//! identity-validated FP16 DFlash artifact and executes the first real graph
//! boundary used by llama.cpp's DFlash2 implementation:
//!
//! ```text
//! target-layer feature concat (25,600) -> fc.weight (5,120) -> hidden_norm
//! ```
//!
//! The implementation is a checked CPU oracle.  It consumes the real
//! safetensors bytes, accumulates in FP32, and returns a digest-bearing output
//! that a future CUDA producer can compare against.  It never fabricates
//! candidate tokens and it does not authorize DFlash decode or product claims.

use crate::dflash_artifact::DFlashRamResident;
use crate::dflash_lifecycle::DFlashDType;
use half::f16;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fmt;

pub const DFLASH_PRODUCER_PREFIX_SCHEMA: &str = "qwen35-dflash-real-fc-hidden-norm-prefix-cpu-v1";
pub const DFLASH_TARGET_FEATURE_ELEMENTS: usize = 5 * 5_120;
pub const DFLASH_HIDDEN_ELEMENTS: usize = 5_120;
pub const DFLASH_RMS_NORM_EPS: f32 = 1.0e-6;

const FC_WEIGHT: &str = "fc.weight";
const HIDDEN_NORM: &str = "hidden_norm.weight";

#[derive(Clone, Debug)]
pub struct DFlashRealProducerPrefix {
    resident: DFlashRamResident,
    fc_range: (usize, usize),
    hidden_norm_range: (usize, usize),
    input_elements: usize,
    output_elements: usize,
    rms_norm_eps: f32,
}

#[derive(Clone, Debug)]
pub struct DFlashProducerPrefixOutput {
    /// Output before the final RMS normalization.  This is useful for a
    /// CUDA/CPU A/B and is not a target-model logit tensor.
    pub projected: Vec<f32>,
    /// Common encoder features consumed by all draft layers' K/V projections.
    /// The decoder hidden instead starts from target anchor/mask embeddings.
    pub normalized: Vec<f32>,
    pub input_sha256: [u8; 32],
    pub projected_sha256: [u8; 32],
    pub normalized_sha256: [u8; 32],
}

#[derive(Debug)]
pub enum DFlashProducerError {
    Invalid(String),
}

impl fmt::Display for DFlashProducerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(error) => write!(f, "invalid DFlash producer prefix: {error}"),
        }
    }
}

impl std::error::Error for DFlashProducerError {}

impl DFlashRealProducerPrefix {
    /// Bind the real FP16 `fc.weight` and `hidden_norm.weight` tensors.
    pub fn bind(resident: DFlashRamResident) -> Result<Self, DFlashProducerError> {
        if resident.inspection().artifact.dtype != DFlashDType::F16 {
            return Err(DFlashProducerError::Invalid(
                "real producer prefix requires the explicitly opted-in F16 artifact".into(),
            ));
        }
        let header_len = usize::try_from(resident.inspection().header_bytes)
            .map_err(|_| DFlashProducerError::Invalid("header extent".into()))?;
        let header_end = 8usize
            .checked_add(header_len)
            .ok_or_else(|| DFlashProducerError::Invalid("header extent overflow".into()))?;
        let header = resident
            .file_bytes()
            .get(8..header_end)
            .ok_or_else(|| DFlashProducerError::Invalid("header extent".into()))?;
        let root: Value = serde_json::from_slice(header)
            .map_err(|error| DFlashProducerError::Invalid(format!("header JSON: {error}")))?;
        let payload_len = resident.payload().len();
        let fc_range = tensor_range(
            &root,
            FC_WEIGHT,
            &[DFLASH_HIDDEN_ELEMENTS, DFLASH_TARGET_FEATURE_ELEMENTS],
            payload_len,
        )?;
        let hidden_norm_range =
            tensor_range(&root, HIDDEN_NORM, &[DFLASH_HIDDEN_ELEMENTS], payload_len)?;
        Ok(Self {
            resident,
            fc_range,
            hidden_norm_range,
            input_elements: DFLASH_TARGET_FEATURE_ELEMENTS,
            output_elements: DFLASH_HIDDEN_ELEMENTS,
            rms_norm_eps: DFLASH_RMS_NORM_EPS,
        })
    }

    pub fn resident(&self) -> &DFlashRamResident {
        &self.resident
    }

    pub const fn input_elements(&self) -> usize {
        self.input_elements
    }

    pub const fn output_elements(&self) -> usize {
        self.output_elements
    }

    pub const fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }

    /// Return the identity-bound F16 `fc.weight` payload slice.
    ///
    /// The returned bytes borrow the validated resident artifact; callers may
    /// widen them for a CUDA component probe, but must not reinterpret them as
    /// a candidate or product weight owner without preserving the artifact
    /// identity receipt.
    pub fn fc_weight_f16_bytes(&self) -> &[u8] {
        &self.resident.payload()[self.fc_range.0..self.fc_range.1]
    }

    /// Return the identity-bound F16 `hidden_norm.weight` payload slice.
    pub fn hidden_norm_f16_bytes(&self) -> &[u8] {
        &self.resident.payload()[self.hidden_norm_range.0..self.hidden_norm_range.1]
    }

    /// Execute the real `fc -> RMSNorm` prefix with FP32 accumulation.
    pub fn execute(
        &self,
        target_features: &[f32],
    ) -> Result<DFlashProducerPrefixOutput, DFlashProducerError> {
        if target_features.len() != self.input_elements {
            return Err(DFlashProducerError::Invalid(format!(
                "target feature extent: expected {}, got {}",
                self.input_elements,
                target_features.len()
            )));
        }
        if target_features.iter().any(|value| !value.is_finite()) {
            return Err(DFlashProducerError::Invalid(
                "target features contain non-finite values".into(),
            ));
        }
        let payload = self.resident.payload();
        let fc_bytes = payload
            .get(self.fc_range.0..self.fc_range.1)
            .ok_or_else(|| DFlashProducerError::Invalid("fc payload extent".into()))?;
        let norm_bytes = payload
            .get(self.hidden_norm_range.0..self.hidden_norm_range.1)
            .ok_or_else(|| DFlashProducerError::Invalid("hidden norm payload extent".into()))?;
        let projected = fc_hidden_norm_matmul(
            fc_bytes,
            norm_bytes,
            self.output_elements,
            self.input_elements,
            target_features,
            self.rms_norm_eps,
        )?;
        let (projected, normalized) = projected;
        if projected
            .iter()
            .chain(&normalized)
            .any(|value| !value.is_finite())
        {
            return Err(DFlashProducerError::Invalid(
                "producer prefix generated non-finite values".into(),
            ));
        }
        Ok(DFlashProducerPrefixOutput {
            projected_sha256: sha256_f32(&projected),
            normalized_sha256: sha256_f32(&normalized),
            input_sha256: sha256_f32(target_features),
            projected,
            normalized,
        })
    }
}

fn tensor_range(
    root: &Value,
    name: &str,
    expected_shape: &[usize],
    payload_len: usize,
) -> Result<(usize, usize), DFlashProducerError> {
    let tensor = root
        .get(name)
        .ok_or_else(|| DFlashProducerError::Invalid(format!("missing tensor {name}")))?;
    if tensor.get("dtype").and_then(Value::as_str) != Some("F16") {
        return Err(DFlashProducerError::Invalid(format!(
            "tensor {name} is not F16"
        )));
    }
    let shape = tensor
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| DFlashProducerError::Invalid(format!("tensor {name} shape")))?;
    let actual_shape: Vec<usize> = shape
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| DFlashProducerError::Invalid(format!("tensor {name} shape")))
        })
        .collect::<Result<_, _>>()?;
    if actual_shape != expected_shape {
        return Err(DFlashProducerError::Invalid(format!(
            "tensor {name} shape mismatch: expected {expected_shape:?}, got {actual_shape:?}"
        )));
    }
    let offsets = tensor
        .get("data_offsets")
        .and_then(Value::as_array)
        .filter(|offsets| offsets.len() == 2)
        .ok_or_else(|| DFlashProducerError::Invalid(format!("tensor {name} offsets")))?;
    let start = offsets[0]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| DFlashProducerError::Invalid(format!("tensor {name} start")))?;
    let end = offsets[1]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| DFlashProducerError::Invalid(format!("tensor {name} end")))?;
    let elements = expected_shape.iter().try_fold(1usize, |acc, value| {
        acc.checked_mul(*value)
            .ok_or_else(|| DFlashProducerError::Invalid(format!("tensor {name} size overflow")))
    })?;
    let expected_bytes = elements
        .checked_mul(2)
        .ok_or_else(|| DFlashProducerError::Invalid(format!("tensor {name} byte overflow")))?;
    if start >= end || end - start != expected_bytes || end > payload_len {
        return Err(DFlashProducerError::Invalid(format!(
            "tensor {name} extent mismatch"
        )));
    }
    Ok((start, end))
}

fn fc_hidden_norm_matmul(
    fc_bytes: &[u8],
    norm_bytes: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>), DFlashProducerError> {
    let expected_fc_bytes = rows
        .checked_mul(columns)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| DFlashProducerError::Invalid("fc size overflow".into()))?;
    if fc_bytes.len() != expected_fc_bytes || norm_bytes.len() != rows * 2 {
        return Err(DFlashProducerError::Invalid(
            "producer tensor byte extent".into(),
        ));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(DFlashProducerError::Invalid("RMS norm epsilon".into()));
    }
    let mut projected = vec![0.0_f32; rows];
    for (row, output) in projected.iter_mut().enumerate() {
        let row_bytes = &fc_bytes[row * columns * 2..(row + 1) * columns * 2];
        let mut sum = 0.0_f32;
        for (index, word) in row_bytes.chunks_exact(2).enumerate() {
            let weight = f16::from_bits(u16::from_le_bytes([word[0], word[1]])).to_f32();
            sum += weight * input[index];
        }
        *output = sum;
    }
    let mean_square = projected.iter().map(|value| value * value).sum::<f32>() / rows as f32;
    let inv_rms = (mean_square + eps).sqrt().recip();
    let normalized = projected
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let scale = f16::from_bits(u16::from_le_bytes([
                norm_bytes[index * 2],
                norm_bytes[index * 2 + 1],
            ]))
            .to_f32();
            value * inv_rms * scale
        })
        .collect();
    Ok((projected, normalized))
}

fn sha256_f32(values: &[f32]) -> [u8; 32] {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
    digest.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::fc_hidden_norm_matmul;
    use half::f16;

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    #[test]
    fn fc_hidden_norm_uses_real_f16_and_fp32_accumulation() {
        let weights = f16_bytes(&[1.0, 2.0, -1.0, 0.5]);
        let norm = f16_bytes(&[1.0, 2.0]);
        let (projected, normalized) =
            fc_hidden_norm_matmul(&weights, &norm, 2, 2, &[3.0, 4.0], 1.0e-6).unwrap();
        assert_eq!(projected, vec![11.0, -1.0]);
        let rms = ((121.0_f32 + 1.0) / 2.0 + 1.0e-6_f32).sqrt();
        assert!((normalized[0] - 11.0 / rms).abs() < 1.0e-5);
        assert!((normalized[1] + 2.0 / rms).abs() < 1.0e-5);
    }

    #[test]
    fn fc_hidden_norm_rejects_bad_extent() {
        let error = fc_hidden_norm_matmul(&[0; 2], &[0; 2], 2, 2, &[1.0, 2.0], 1.0e-6)
            .expect_err("bad extents must fail closed");
        assert!(error.to_string().contains("extent"));
    }
}
