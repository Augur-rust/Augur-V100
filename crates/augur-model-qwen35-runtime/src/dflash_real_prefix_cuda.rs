#![cfg(feature = "cuda")]

//! CUDA A/B for the real DFlash producer prefix.
//!
//! This is intentionally a component boundary: it consumes the target
//! oracle's real five-layer feature concat and compares CUDA BF16/F32 output
//! with the checked CPU F16 oracle.  It does not produce candidate logits or
//! authorize acceptance.

use crate::{DFlashRealProducerPrefix, DFLASH_RMS_NORM_EPS};
use augur_backend::{
    CudaBf16F32MatvecArgsV1, CudaBf16WeightViewV1, CudaF32View, CudaRmsnormF32ArgsV1,
    OwnedCudaBuffer, OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;
use half::f16;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug)]
pub struct DFlashRealPrefixCudaAb {
    pub projected_max_abs: f32,
    pub projected_max_relative: f32,
    pub projected_relative_l2: f32,
    pub normalized_max_abs: f32,
    pub normalized_max_relative: f32,
    pub normalized_relative_l2: f32,
    pub projected_sha256: [u8; 32],
    pub normalized_sha256: [u8; 32],
    pub cpu_projected_sha256: [u8; 32],
    pub cpu_normalized_sha256: [u8; 32],
    pub host_wall_ms: f64,
}

fn f16_to_bf16_bytes(bytes: &[u8]) -> Result<Vec<u8>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err("F16 payload is not two-byte aligned".into());
    }
    let mut out = Vec::with_capacity(bytes.len());
    for word in bytes.chunks_exact(2) {
        let value = f16::from_bits(u16::from_le_bytes([word[0], word[1]])).to_f32();
        if !value.is_finite() {
            return Err("F16 payload contains non-finite value".into());
        }
        let bits = value.to_bits();
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
        out.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
    }
    Ok(out)
}

fn f16_to_f32_bytes(bytes: &[u8]) -> Result<Vec<u8>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err("F16 payload is not two-byte aligned".into());
    }
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for word in bytes.chunks_exact(2) {
        let value = f16::from_bits(u16::from_le_bytes([word[0], word[1]])).to_f32();
        if !value.is_finite() {
            return Err("F16 payload contains non-finite value".into());
        }
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    Ok(out)
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

fn digest(values: &[f32]) -> [u8; 32] {
    let mut sha = Sha256::new();
    for value in values {
        sha.update(value.to_bits().to_le_bytes());
    }
    sha.finalize().into()
}

fn metrics(actual: &[f32], expected: &[f32]) -> (f32, f32, f32) {
    let mut max_abs = 0.0_f32;
    let mut max_relative = 0.0_f32;
    let mut squared = 0.0_f64;
    let mut expected_squared = 0.0_f64;
    for (&a, &e) in actual.iter().zip(expected) {
        let abs = (a - e).abs();
        max_abs = max_abs.max(abs);
        max_relative = max_relative.max(abs / e.abs().max(1.0e-6));
        squared += f64::from(abs) * f64::from(abs);
        expected_squared += f64::from(e) * f64::from(e);
    }
    (
        max_abs,
        max_relative,
        (squared / expected_squared.max(1.0e-30)).sqrt() as f32,
    )
}

/// Execute the real producer prefix on CUDA and compare it to the CPU oracle.
pub fn run_dflash_real_prefix_cuda_ab(
    stream: &OwnedCudaStream,
    producer: &DFlashRealProducerPrefix,
    input: &[f32],
) -> Result<DFlashRealPrefixCudaAb, String> {
    if input.len() != producer.input_elements() || input.iter().any(|v| !v.is_finite()) {
        return Err("real target feature input geometry/non-finite value".into());
    }
    let cpu = producer.execute(input).map_err(|e| e.to_string())?;
    let fc_bf16 = f16_to_bf16_bytes(producer.fc_weight_f16_bytes())?;
    let norm_f32 = f16_to_f32_bytes(producer.hidden_norm_f16_bytes())?;
    let mut input_bytes = Vec::with_capacity(input.len() * 4);
    for value in input {
        input_bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    let device = stream.device_ordinal();
    let start = std::time::Instant::now();
    let fc = OwnedCudaBuffer::allocate(device, fc_bf16.len()).map_err(|e| e.to_string())?;
    let norm = OwnedCudaBuffer::allocate(device, norm_f32.len()).map_err(|e| e.to_string())?;
    let input_buf =
        OwnedCudaBuffer::allocate(device, input_bytes.len()).map_err(|e| e.to_string())?;
    let projected_buf = OwnedCudaBuffer::allocate(device, producer.output_elements() * 4)
        .map_err(|e| e.to_string())?;
    let normalized_buf = OwnedCudaBuffer::allocate(device, producer.output_elements() * 4)
        .map_err(|e| e.to_string())?;
    fc.copy_from_host_sync(stream, &fc_bf16)
        .map_err(|e| e.to_string())?;
    norm.copy_from_host_sync(stream, &norm_f32)
        .map_err(|e| e.to_string())?;
    input_buf
        .copy_from_host_sync(stream, &input_bytes)
        .map_err(|e| e.to_string())?;
    let fc_view = CudaBf16WeightViewV1::new(
        fc.view(0, fc_bf16.len()).map_err(|e| e.to_string())?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
        producer.output_elements(),
        producer.input_elements(),
    )
    .map_err(|e| e.to_string())?;
    let input_view = CudaF32View::new(
        input_buf
            .view(0, input_bytes.len())
            .map_err(|e| e.to_string())?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        producer.input_elements(),
    )
    .map_err(|e| e.to_string())?;
    let projected_view = CudaF32View::new(
        projected_buf
            .view(0, producer.output_elements() * 4)
            .map_err(|e| e.to_string())?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        producer.output_elements(),
    )
    .map_err(|e| e.to_string())?;
    CudaBf16F32MatvecArgsV1::bind(stream, fc_view, input_view, projected_view)
        .map_err(|e| e.to_string())?
        .launch(stream)
        .map_err(|e| e.to_string())?;
    let norm_view = CudaF32View::new(
        norm.view(0, norm_f32.len()).map_err(|e| e.to_string())?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        producer.output_elements(),
    )
    .map_err(|e| e.to_string())?;
    let normalized_view = CudaF32View::new(
        normalized_buf
            .view(0, producer.output_elements() * 4)
            .map_err(|e| e.to_string())?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        producer.output_elements(),
    )
    .map_err(|e| e.to_string())?;
    CudaRmsnormF32ArgsV1::bind(
        stream,
        projected_view,
        norm_view,
        normalized_view,
        DFLASH_RMS_NORM_EPS,
    )
    .map_err(|e| e.to_string())?
    .launch(stream)
    .map_err(|e| e.to_string())?;
    let mut projected_bytes = vec![0_u8; producer.output_elements() * 4];
    let mut normalized_bytes = vec![0_u8; producer.output_elements() * 4];
    projected_buf
        .copy_to_host_sync(stream, &mut projected_bytes)
        .map_err(|e| e.to_string())?;
    normalized_buf
        .copy_to_host_sync(stream, &mut normalized_bytes)
        .map_err(|e| e.to_string())?;
    let projected = decode_f32(&projected_bytes);
    let normalized = decode_f32(&normalized_bytes);
    let projected_metrics = metrics(&projected, &cpu.projected);
    let normalized_metrics = metrics(&normalized, &cpu.normalized);
    Ok(DFlashRealPrefixCudaAb {
        projected_max_abs: projected_metrics.0,
        projected_max_relative: projected_metrics.1,
        projected_relative_l2: projected_metrics.2,
        normalized_max_abs: normalized_metrics.0,
        normalized_max_relative: normalized_metrics.1,
        normalized_relative_l2: normalized_metrics.2,
        projected_sha256: digest(&projected),
        normalized_sha256: digest(&normalized),
        cpu_projected_sha256: cpu.projected_sha256,
        cpu_normalized_sha256: cpu.normalized_sha256,
        host_wall_ms: start.elapsed().as_secs_f64() * 1_000.0,
    })
}
