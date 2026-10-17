//! Bounded host image preparation for the V100 Vision product boundary.
//! This is not a Vision encoder or permission to publish language tokens.

use crate::quant::f32_to_f16;
use crate::{Qwen35Error, Qwen35ImageGrid, QWEN35_IMAGE_PATCH_ELEMENTS};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const QWEN35_VISION_INPUT_ABI: &str = "qwen35-rgb8-f16-patch-v1";

#[derive(Debug)]
pub struct Qwen35PreparedF16Image {
    pub grid: Qwen35ImageGrid,
    /// Little-endian [group_h,group_w,local_h,local_w,c,t,y,x] FP16.
    pub pixel_values_f16_le: Vec<u8>,
    pub patch_positions_hw: Vec<[usize; 2]>,
    pub rgb8_sha256: String,
    pub payload_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35VisionInputReceipt {
    pub schema: &'static str,
    pub input_abi: &'static str,
    pub rgb8_sha256: String,
    pub prepared_sha256: String,
    pub input_width: usize,
    pub input_height: usize,
    pub grid_thw: [usize; 3],
    pub patch_count: usize,
    pub merged_tokens: usize,
    pub patch_elements: usize,
    pub prepared_bytes: usize,
    pub normalization: &'static str,
    pub patch_order: &'static str,
    pub accumulation_policy: &'static str,
    pub fp16_max_abs_rounding_error_bound: f32,
    pub vision_forward_executed: bool,
    pub target_image_consumed: bool,
    pub product_ready: bool,
}

impl Qwen35PreparedF16Image {
    /// Revalidate before a content-addressed embedding-cache lookup, not only
    /// on cache misses. Public host buffers cannot spoof an existing cache key.
    pub fn validate_payload_identity(&self) -> Result<(), Qwen35Error> {
        let [_, h, w] = self.grid.grid_thw();
        if self.pixel_values_f16_le.len() != h * w * QWEN35_IMAGE_PATCH_ELEMENTS * 2
            || self.patch_positions_hw.len() != h * w
        {
            return Err(Qwen35Error::invalid(
                "vision_payload",
                "prepared extent drift",
            ));
        }
        for (i, &actual) in self.patch_positions_hw.iter().enumerate() {
            let group = i / 4;
            let local = i % 4;
            let expected = [
                (group / (w / 2)) * 2 + local / 2,
                (group % (w / 2)) * 2 + local % 2,
            ];
            if actual != expected {
                return Err(Qwen35Error::invalid(
                    "vision_payload",
                    "merge-group position drift",
                ));
            }
        }
        let mut digest = Sha256::new();
        digest.update(QWEN35_VISION_INPUT_ABI.as_bytes());
        for n in self.grid.grid_thw() {
            digest.update((n as u64).to_le_bytes());
        }
        digest.update(&self.pixel_values_f16_le);
        if format!("{:x}", digest.finalize()) != self.payload_sha256 {
            return Err(Qwen35Error::invalid(
                "vision_payload",
                "prepared payload identity drift",
            ));
        }
        Ok(())
    }
    pub fn receipt(&self) -> Qwen35VisionInputReceipt {
        let [_, h, w] = self.grid.grid_thw();
        Qwen35VisionInputReceipt {
            schema: "qwen35-vision-input-receipt-v1",
            input_abi: QWEN35_VISION_INPUT_ABI,
            rgb8_sha256: self.rgb8_sha256.clone(),
            prepared_sha256: self.payload_sha256.clone(),
            input_width: w * 16,
            input_height: h * 16,
            grid_thw: self.grid.grid_thw(),
            patch_count: h * w,
            merged_tokens: self.grid.merged_tokens(),
            patch_elements: QWEN35_IMAGE_PATCH_ELEMENTS,
            prepared_bytes: self.pixel_values_f16_le.len(),
            normalization: "rgb8_div255_minus_half_div_half_f32_then_rne_f16",
            patch_order: "merge_group_major_c_t_y_x",
            accumulation_policy: "encoder_requires_fp32_control_not_executed_here",
            fp16_max_abs_rounding_error_bound: 0.000_244_141,
            vision_forward_executed: false,
            target_image_consumed: false,
            product_ready: false,
        }
    }
}

/// Fuse normalization, merge-group packing and RNE FP16 storage. Avoids a
/// complete FP32 intermediate and computes each still-image value once; the
/// second temporal plane reuses the exact same 16x16 channel bytes. Decoding
/// and resize are separate, explicitly selected image-processor policies.
pub fn prepare_qwen35_aligned_rgb8_f16(
    rgb: &[u8],
    grid: Qwen35ImageGrid,
    max_prepared_bytes: usize,
) -> Result<Qwen35PreparedF16Image, Qwen35Error> {
    if rgb.len() != grid.rgb8_bytes() {
        return Err(Qwen35Error::invalid("rgb8", "input byte extent mismatch"));
    }
    let [_, gh, gw] = grid.grid_thw();
    let bytes = gh
        .checked_mul(gw)
        .and_then(|n| n.checked_mul(QWEN35_IMAGE_PATCH_ELEMENTS))
        .and_then(|n| n.checked_mul(2))
        .ok_or(Qwen35Error::SizeOverflow)?;
    if bytes > max_prepared_bytes {
        return Err(Qwen35Error::invalid(
            "vision_input_budget",
            "FP16 patch payload exceeds budget",
        ));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes)
        .map_err(|_| Qwen35Error::invalid("vision_input", "CPU payload allocation failed"))?;
    let mut positions = Vec::new();
    positions
        .try_reserve_exact(gh * gw)
        .map_err(|_| Qwen35Error::invalid("vision_input", "CPU positions allocation failed"))?;
    let width = gw * 16;
    let mut channel = [0u8; 16 * 16 * 2];
    for group_h in 0..gh / 2 {
        for group_w in 0..gw / 2 {
            for local_h in 0..2 {
                for local_w in 0..2 {
                    let ph = group_h * 2 + local_h;
                    let pw = group_w * 2 + local_w;
                    positions.push([ph, pw]);
                    for c in 0..3 {
                        for y in 0..16 {
                            for x in 0..16 {
                                let offset = ((ph * 16 + y) * width + pw * 16 + x) * 3 + c;
                                let value = (f32::from(rgb[offset]) * (1.0f32 / 255.0) - 0.5) / 0.5;
                                let target = (y * 16 + x) * 2;
                                channel[target..target + 2]
                                    .copy_from_slice(&f32_to_f16(value).to_le_bytes());
                            }
                        }
                        values.extend_from_slice(&channel);
                        values.extend_from_slice(&channel);
                    }
                }
            }
        }
    }
    let mut hash = Sha256::new();
    hash.update(QWEN35_VISION_INPUT_ABI.as_bytes());
    for n in grid.grid_thw() {
        hash.update((n as u64).to_le_bytes());
    }
    hash.update(&values);
    Ok(Qwen35PreparedF16Image {
        grid,
        pixel_values_f16_le: values,
        patch_positions_hw: positions,
        rgb8_sha256: format!("{:x}", Sha256::digest(rgb)),
        payload_sha256: format!("{:x}", hash.finalize()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_lookup_cannot_trust_mutable_payload_or_patch_metadata() {
        let fresh = || {
            prepare_qwen35_aligned_rgb8_f16(
                &vec![17; 32 * 32 * 3],
                Qwen35ImageGrid::from_aligned_pixels(32, 32).unwrap(),
                16384,
            )
            .unwrap()
        };
        assert!(fresh().validate_payload_identity().is_ok());
        let mut image = fresh();
        image.pixel_values_f16_le[0] ^= 1;
        assert!(image.validate_payload_identity().is_err());
        let mut image = fresh();
        image.patch_positions_hw[0] = [1, 1];
        assert!(image.validate_payload_identity().is_err());
        let mut image = fresh();
        image.pixel_values_f16_le.pop();
        assert!(image.validate_payload_identity().is_err());
        let mut image = fresh();
        image.payload_sha256 = "0".repeat(64);
        assert!(image.validate_payload_identity().is_err());
    }
    use crate::prepare_qwen35_aligned_rgb8;
    use crate::quant::f16_to_f32;

    #[test]
    fn fused_patches_match_the_existing_fp32_contract_and_rne() {
        for (h, w) in [(32, 32), (32, 64), (64, 96), (256, 256)] {
            let grid = Qwen35ImageGrid::from_aligned_pixels(h, w).unwrap();
            let rgb: Vec<_> = (0..grid.rgb8_bytes())
                .map(|i| ((i * 37 + i / 19) % 256) as u8)
                .collect();
            let original = prepare_qwen35_aligned_rgb8(&rgb, grid, usize::MAX).unwrap();
            let fused = prepare_qwen35_aligned_rgb8_f16(&rgb, grid, usize::MAX).unwrap();
            assert_eq!(fused.patch_positions_hw, original.patch_positions_hw);
            assert_eq!(
                fused.pixel_values_f16_le.len(),
                original.pixel_values.len() * 2
            );
            for (raw, expected) in fused
                .pixel_values_f16_le
                .chunks_exact(2)
                .zip(original.pixel_values)
            {
                let bits = u16::from_le_bytes([raw[0], raw[1]]);
                assert_eq!(bits, f32_to_f16(expected));
                assert!((f16_to_f32(bits) - expected).abs() <= 0.000_244_141);
            }
        }
    }

    #[test]
    fn all_rgb8_values_are_finite_and_inside_the_declared_rounding_bound() {
        for byte in 0u16..=255 {
            let value = (f32::from(byte) * (1.0f32 / 255.0) - 0.5) / 0.5;
            let actual = f16_to_f32(f32_to_f16(value));
            assert!(actual.is_finite());
            assert!((actual - value).abs() <= 0.000_244_141);
        }
    }

    #[test]
    fn invalid_extent_and_one_byte_short_budget_reject() {
        let grid = Qwen35ImageGrid::from_aligned_pixels(32, 32).unwrap();
        let rgb = vec![0; grid.rgb8_bytes()];
        let expected = 4 * QWEN35_IMAGE_PATCH_ELEMENTS * 2;
        assert!(prepare_qwen35_aligned_rgb8_f16(&rgb[..rgb.len() - 1], grid, expected).is_err());
        assert!(prepare_qwen35_aligned_rgb8_f16(&rgb, grid, expected - 1).is_err());
        assert_eq!(
            prepare_qwen35_aligned_rgb8_f16(&rgb, grid, expected)
                .unwrap()
                .pixel_values_f16_le
                .len(),
            expected
        );
    }

    #[test]
    fn receipt_does_not_promote_input_preparation_to_inference() {
        let grid = Qwen35ImageGrid::from_aligned_pixels(32, 32).unwrap();
        let prepared =
            prepare_qwen35_aligned_rgb8_f16(&vec![128; grid.rgb8_bytes()], grid, usize::MAX)
                .unwrap();
        let receipt = prepared.receipt();
        assert!(
            !receipt.vision_forward_executed
                && !receipt.target_image_consumed
                && !receipt.product_ready
        );
        assert_eq!(receipt.grid_thw, [1, 2, 2]);
        assert_eq!(receipt.merged_tokens, 1);
        assert_eq!(receipt.prepared_bytes, 12288);
    }

    #[test]
    fn payload_identity_binds_geometry_and_rgb_mutations() {
        let a = Qwen35ImageGrid::from_aligned_pixels(32, 64).unwrap();
        let b = Qwen35ImageGrid::from_aligned_pixels(64, 32).unwrap();
        let mut rgb = vec![128; a.rgb8_bytes()];
        let first = prepare_qwen35_aligned_rgb8_f16(&rgb, a, usize::MAX).unwrap();
        let rotated = prepare_qwen35_aligned_rgb8_f16(&rgb, b, usize::MAX).unwrap();
        assert_ne!(first.payload_sha256, rotated.payload_sha256);
        rgb[0] = 0;
        let changed = prepare_qwen35_aligned_rgb8_f16(&rgb, a, usize::MAX).unwrap();
        assert_ne!(first.rgb8_sha256, changed.rgb8_sha256);
        assert_ne!(first.payload_sha256, changed.payload_sha256);
    }
}
