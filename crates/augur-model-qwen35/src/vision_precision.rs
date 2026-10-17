//! CPU conversion only. Stored FP16 does not prove FP32 accumulation or Vision
//! semantics. Overflow/non-finite values reject; subnormal/zero rounding is
//! counted explicitly. Tensor consumers must separately validate their layout.

use crate::{
    quant::{f16_to_f32, f32_to_f16},
    Qwen35Error, Qwen35VisionTensorBinding,
};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum Qwen35VisionStorage {
    F16,
    F32,
}

impl Qwen35VisionStorage {
    /// Affine GEMM/patch-convolution weights use FP16; vectors and interpolated
    /// position embeddings remain FP32. This is a future-forward candidate.
    pub fn for_binding(binding: &Qwen35VisionTensorBinding) -> Self {
        if binding.shape.len() == 1 || binding.stage == "pos_embed" {
            Self::F32
        } else {
            Self::F16
        }
    }
    pub fn element_bytes(self) -> usize {
        if self == Self::F32 {
            4
        } else {
            2
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35VisionConversionStats {
    pub elements: usize,
    pub exact_elements: usize,
    pub rounded_elements: usize,
    pub nonzero_to_zero: usize,
    pub output_subnormals: usize,
    pub source_max_abs: f32,
    pub max_abs_error: f32,
    pub max_relative_error_nonzero: f32,
    pub output_bytes: usize,
}

pub struct Qwen35VisionConversion {
    pub bytes: Vec<u8>,
    pub stats: Qwen35VisionConversionStats,
}

/// Checked round-to-nearest-even, preserving signed zero. The finite BF16
/// inputs representable in the F16 normal range are exact; only smaller values
/// can round, with absolute error at most half the minimum F16 subnormal.
pub fn qwen35_bf16_to_f16_bits(bits: u16) -> Result<u16, Qwen35Error> {
    let value = f32::from_bits(u32::from(bits) << 16);
    if !value.is_finite() || value.abs() > 65_504.0 {
        return Err(Qwen35Error::invalid(
            "vision_precision",
            "non-finite or out-of-FP16-range BF16 input",
        ));
    }
    let output = f32_to_f16(value);
    let decoded = f16_to_f32(output);
    if !decoded.is_finite() || (decoded - value).abs() > 2.980_232_2e-8 {
        return Err(Qwen35Error::invalid(
            "vision_precision",
            "BF16-to-FP16 rounding contract violated",
        ));
    }
    Ok(output)
}

/// Produces a private byte vector; errors cannot expose partial converted data.
/// The caller must bind tensor names/shapes and source bytes independently.
pub fn convert_qwen35_vision_bf16(
    input: &[u8],
    storage: Qwen35VisionStorage,
    max_output_bytes: usize,
) -> Result<Qwen35VisionConversion, Qwen35Error> {
    if input.is_empty() || !input.len().is_multiple_of(2) {
        return Err(Qwen35Error::invalid(
            "vision_precision",
            "nonempty even-sized BF16 payload required",
        ));
    }
    let elements = input.len() / 2;
    let output_bytes = elements
        .checked_mul(storage.element_bytes())
        .ok_or(Qwen35Error::SizeOverflow)?;
    if output_bytes > max_output_bytes {
        return Err(Qwen35Error::invalid(
            "vision_precision",
            "CPU output byte budget exceeded",
        ));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(output_bytes)
        .map_err(|_| Qwen35Error::invalid("vision_precision", "CPU allocation failed"))?;
    let mut stats = Qwen35VisionConversionStats {
        elements,
        exact_elements: 0,
        rounded_elements: 0,
        nonzero_to_zero: 0,
        output_subnormals: 0,
        source_max_abs: 0.0,
        max_abs_error: 0.0,
        max_relative_error_nonzero: 0.0,
        output_bytes,
    };
    for pair in input.chunks_exact(2) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        let value = f32::from_bits(u32::from(bits) << 16);
        if !value.is_finite() {
            return Err(Qwen35Error::invalid(
                "vision_precision",
                "non-finite BF16 input",
            ));
        }
        let decoded = match storage {
            Qwen35VisionStorage::F16 => {
                let half = qwen35_bf16_to_f16_bits(bits)?;
                bytes.extend_from_slice(&half.to_le_bytes());
                stats.output_subnormals += usize::from(half & 0x7c00 == 0 && half & 0x03ff != 0);
                f16_to_f32(half)
            }
            Qwen35VisionStorage::F32 => {
                bytes.extend_from_slice(&value.to_le_bytes());
                stats.output_subnormals += usize::from(value.is_subnormal());
                value
            }
        };
        if value.to_bits() == decoded.to_bits() {
            stats.exact_elements += 1;
        } else {
            stats.rounded_elements += 1;
        }
        stats.nonzero_to_zero += usize::from(value != 0.0 && decoded == 0.0);
        stats.source_max_abs = stats.source_max_abs.max(value.abs());
        let error = (value - decoded).abs();
        stats.max_abs_error = stats.max_abs_error.max(error);
        if value != 0.0 {
            stats.max_relative_error_nonzero =
                stats.max_relative_error_nonzero.max(error / value.abs());
        }
    }
    Ok(Qwen35VisionConversion { bytes, stats })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn storage_policy_keeps_vectors_and_positions_fp32() {
        let mut binding = Qwen35VisionTensorBinding {
            name: "model.visual.blocks.0.attn.qkv.weight".into(),
            stage: "blocks".into(),
            block: Some(0),
            dtype: "BF16".into(),
            shape: vec![3456, 1152],
            file_data_offsets: [0, 3456 * 1152 * 2],
            resident_offset: 0,
            byte_len: 3456 * 1152 * 2,
        };
        assert_eq!(
            Qwen35VisionStorage::for_binding(&binding),
            Qwen35VisionStorage::F16
        );
        binding.shape = vec![1152];
        assert_eq!(
            Qwen35VisionStorage::for_binding(&binding),
            Qwen35VisionStorage::F32
        );
        binding.shape = vec![2304, 1152];
        binding.stage = "pos_embed".into();
        assert_eq!(
            Qwen35VisionStorage::for_binding(&binding),
            Qwen35VisionStorage::F32
        );
        binding.shape = vec![1152, 3, 2, 16, 16];
        binding.stage = "patch_embed".into();
        assert_eq!(
            Qwen35VisionStorage::for_binding(&binding),
            Qwen35VisionStorage::F16
        );
    }
    #[test]
    fn signed_zero_normal_and_boundaries() {
        for (bf, half) in [
            (0, 0),
            (0x8000, 0x8000),
            (0x3f80, 0x3c00),
            (0xc000, 0xc000),
            (0x477f, 0x7bf8),
            (0x3380, 1),
        ] {
            assert_eq!(qwen35_bf16_to_f16_bits(bf).unwrap(), half);
        }
        for bf in [0x4780, 0xc780, 0x7f80, 0xff80, 0x7fc0, 0x7f81] {
            assert!(qwen35_bf16_to_f16_bits(bf).is_err());
        }
    }
    #[test]
    fn ties_use_even_subnormal_and_zero() {
        assert_eq!(qwen35_bf16_to_f16_bits(0x3300).unwrap(), 0); // 2^-25
        assert_eq!(qwen35_bf16_to_f16_bits(0x33c0).unwrap(), 2); // 1.5 * 2^-24
        assert_eq!(qwen35_bf16_to_f16_bits(0x3420).unwrap(), 2); // 2.5 * 2^-24
        assert_eq!(qwen35_bf16_to_f16_bits(0xb300).unwrap(), 0x8000);
    }
    #[test]
    fn all_bf16_codes_obey_finite_range_and_error_contract() {
        for code in 0..=u16::MAX {
            let f = f32::from_bits(u32::from(code) << 16);
            let actual = qwen35_bf16_to_f16_bits(code);
            assert_eq!(actual.is_ok(), f.is_finite() && f.abs() <= 65504.0);
            if let Ok(half) = actual {
                assert_eq!(half & 0x8000, code & 0x8000);
                assert!((f16_to_f32(half) - f).abs() <= 2.980_232_2e-8);
            }
        }
    }
    #[test]
    fn payload_budget_size_and_late_invalid_reject() {
        assert!(convert_qwen35_vision_bf16(&[], Qwen35VisionStorage::F16, 100).is_err());
        assert!(convert_qwen35_vision_bf16(&[0], Qwen35VisionStorage::F16, 100).is_err());
        assert!(convert_qwen35_vision_bf16(&[0, 0], Qwen35VisionStorage::F32, 3).is_err());
        assert!(
            convert_qwen35_vision_bf16(&[0, 0, 0x80, 0x7f], Qwen35VisionStorage::F16, 100).is_err()
        );
    }
    #[test]
    fn exact_rounded_and_underflow_counts_conserve() {
        let values = [0u16, 0x8000, 0x3f80, 0x3380, 0x3300, 0x33c0, 1];
        let input: Vec<_> = values.into_iter().flat_map(u16::to_le_bytes).collect();
        let f16 = convert_qwen35_vision_bf16(&input, Qwen35VisionStorage::F16, 14).unwrap();
        assert_eq!(f16.stats.exact_elements, 4);
        assert_eq!(f16.stats.rounded_elements, 3);
        assert_eq!(f16.stats.nonzero_to_zero, 2);
        assert_eq!(f16.stats.output_subnormals, 2);
        assert_eq!(
            f16.stats.elements,
            f16.stats.exact_elements + f16.stats.rounded_elements
        );
        let f32 = convert_qwen35_vision_bf16(&input, Qwen35VisionStorage::F32, 28).unwrap();
        assert_eq!(f32.stats.exact_elements, 7);
        assert_eq!(f32.stats.max_abs_error, 0.0);
    }
}
