//! Language-model MRoPE schedule for image-only prompts. Cache slots remain
//! physical sequence indices; THW coordinates/rope_delta are not KV cursors.
use crate::{Qwen35Error, Qwen35ImagePromptLayout};
use serde::Serialize;
use sha2::{Digest, Sha256};
pub const QWEN35_IMAGE_MROPE_ABI: &str = "qwen35-image-interleaved-mrope-64-v1";
const PAIRS: usize = 32;
const THETA: f32 = 10_000_000.0;
const MAX_POSITIONS: usize = 262_144;

#[derive(Debug)]
pub struct Qwen35ImageMropePlan {
    prompt_positions: Vec<[i64; 3]>,
    rope_delta: i64,
    capacity: usize,
    identity_sha256: String,
    inverse_frequencies: [f32; PAIRS],
}
#[derive(Debug, Serialize)]
pub struct Qwen35ImageMropeReceipt {
    pub abi: &'static str,
    pub prompt_physical_tokens: usize,
    pub capacity_physical_tokens: usize,
    pub rotary_dimension: usize,
    pub mrope_section: [usize; 3],
    pub rope_delta: i64,
    pub position_identity_sha256: String,
    pub packed_cos_sin_bytes: usize,
    pub cache_cursor_is_physical: bool,
    pub inverse_frequency_policy: &'static str,
    pub target_image_consumed: bool,
    pub product_ready: bool,
}
impl Qwen35ImageMropePlan {
    pub fn new(
        layout: &Qwen35ImagePromptLayout,
        capacity: usize,
        table_budget: usize,
    ) -> Result<Self, Qwen35Error> {
        let bytes = capacity
            .checked_mul(PAIRS * 8)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if capacity < layout.positions_thw().len()
            || capacity > MAX_POSITIONS
            || bytes > table_budget
        {
            return Err(Qwen35Error::invalid(
                "image_mrope_capacity",
                "capacity/packed table budget mismatch",
            ));
        }
        let mut digest = Sha256::new();
        let inverse_frequencies = std::array::from_fn(|pair| THETA.powf(-2.0 * pair as f32 / 64.0));
        digest.update(QWEN35_IMAGE_MROPE_ABI.as_bytes());
        for coefficient in inverse_frequencies {
            digest.update(coefficient.to_bits().to_le_bytes());
        }
        digest.update((capacity as u64).to_le_bytes());
        digest.update(layout.rope_delta().to_le_bytes());
        for row in layout.positions_thw() {
            if row.iter().any(|&p| !(0..MAX_POSITIONS as i64).contains(&p)) {
                return Err(Qwen35Error::invalid(
                    "image_mrope_position",
                    "THW coordinate outside model range",
                ));
            }
            for p in row {
                digest.update(p.to_le_bytes());
            }
        }
        Ok(Self {
            prompt_positions: layout.positions_thw().to_vec(),
            rope_delta: layout.rope_delta(),
            capacity,
            identity_sha256: format!("{:x}", digest.finalize()),
            inverse_frequencies,
        })
    }
    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn inverse_frequencies(&self) -> &[f32; PAIRS] {
        &self.inverse_frequencies
    }
    pub fn position_thw(&self, cache_slot: usize) -> Result<[i64; 3], Qwen35Error> {
        if cache_slot >= self.capacity {
            return Err(Qwen35Error::invalid(
                "image_mrope_slot",
                "cache slot outside reserved extent",
            ));
        }
        if let Some(&row) = self.prompt_positions.get(cache_slot) {
            return Ok(row);
        }
        let position = cache_slot as i64 + self.rope_delta;
        if !(0..MAX_POSITIONS as i64).contains(&position) {
            return Err(Qwen35Error::invalid(
                "image_mrope_continuation",
                "continuation coordinate outside model range",
            ));
        }
        Ok([position; 3])
    }
    /// Half-packed rows match the existing split-half CUDA ABI. Frequencies
    /// retain their original index; only the corresponding THW axis changes.
    pub fn row(&self, cache_slot: usize) -> Result<([f32; PAIRS], [f32; PAIRS]), Qwen35Error> {
        let position = self.position_thw(cache_slot)?;
        let mut cosine = [0.0; PAIRS];
        let mut sine = [0.0; PAIRS];
        for pair in 0..PAIRS {
            let axis = match pair % 3 {
                1 if pair < 33 => 1,
                2 if pair < 30 => 2,
                _ => 0,
            };
            let angle = position[axis] as f32 * self.inverse_frequencies[pair];
            let (s, c) = angle.sin_cos();
            if !s.is_finite() || !c.is_finite() {
                return Err(Qwen35Error::invalid(
                    "image_mrope_value",
                    "non-finite RoPE value",
                ));
            }
            cosine[pair] = c;
            sine[pair] = s;
        }
        Ok((cosine, sine))
    }
    pub fn tables(&self) -> Result<(Vec<f32>, Vec<f32>), Qwen35Error> {
        let elements = self.capacity * PAIRS;
        let mut cosine = Vec::new();
        let mut sine = Vec::new();
        cosine
            .try_reserve_exact(elements)
            .map_err(|_| Qwen35Error::invalid("image_mrope_tables", "CPU allocation failed"))?;
        sine.try_reserve_exact(elements)
            .map_err(|_| Qwen35Error::invalid("image_mrope_tables", "CPU allocation failed"))?;
        for slot in 0..self.capacity {
            let (c, s) = self.row(slot)?;
            cosine.extend(c);
            sine.extend(s);
        }
        Ok((cosine, sine))
    }
    pub fn receipt(&self) -> Qwen35ImageMropeReceipt {
        Qwen35ImageMropeReceipt {
            abi: QWEN35_IMAGE_MROPE_ABI,
            prompt_physical_tokens: self.prompt_positions.len(),
            capacity_physical_tokens: self.capacity,
            rotary_dimension: 64,
            mrope_section: [11, 11, 10],
            rope_delta: self.rope_delta,
            position_identity_sha256: self.identity_sha256.clone(),
            packed_cos_sin_bytes: self.capacity * PAIRS * 8,
            cache_cursor_is_physical: true,
            inverse_frequency_policy: "existing_native_fp32_negative_powf_v1",
            target_image_consumed: false,
            product_ready: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Qwen35ImageGrid;
    fn image_layout() -> Qwen35ImagePromptLayout {
        let mut ids = vec![1, 248053];
        ids.extend(vec![248056; 6]);
        ids.extend([248054, 7]);
        let kinds = ids
            .iter()
            .map(|&id| u8::from(id == 248056))
            .collect::<Vec<_>>();
        Qwen35ImagePromptLayout::new(
            &ids,
            &kinds,
            &[Qwen35ImageGrid::from_aligned_pixels(64, 96).unwrap()],
        )
        .unwrap()
    }
    #[test]
    fn image_axes_and_continuation_are_not_the_physical_cache_cursor() {
        let layout = image_layout();
        let plan = Qwen35ImageMropePlan::new(&layout, 16, 4096).unwrap();
        assert_eq!(plan.position_thw(7).unwrap(), [2, 3, 4]);
        assert_eq!(layout.rope_delta(), -3);
        assert_eq!(plan.position_thw(10).unwrap(), [7; 3]);
        let (c, s) = plan.row(7).unwrap();
        for (pair, position) in [(0, 2.0), (1, 3.0), (2, 4.0), (31, 3.0)] {
            let angle = position * THETA.powf(-2.0 * pair as f32 / 64.0);
            let (sn, cs) = angle.sin_cos();
            assert_eq!(c[pair].to_bits(), cs.to_bits());
            assert_eq!(s[pair].to_bits(), sn.to_bits());
        }
    }
    #[test]
    fn text_tables_preserve_the_existing_fp32_math_bit_for_bit() {
        let layout = Qwen35ImagePromptLayout::new(&[1, 2, 3], &[0, 0, 0], &[]).unwrap();
        let plan = Qwen35ImageMropePlan::new(&layout, 8, 2048).unwrap();
        for p in 0..8 {
            let (c, s) = plan.row(p).unwrap();
            for pair in 0..32 {
                let (sn, cs) = (p as f32 * THETA.powf(-2.0 * pair as f32 / 64.0)).sin_cos();
                assert_eq!(c[pair].to_bits(), cs.to_bits());
                assert_eq!(s[pair].to_bits(), sn.to_bits());
            }
        }
    }
    #[test]
    fn schedule_identity_binds_capacity_delta_and_coordinates() {
        let layout = image_layout();
        let a = Qwen35ImageMropePlan::new(&layout, 16, 4096).unwrap();
        let b = Qwen35ImageMropePlan::new(&layout, 17, 8192).unwrap();
        assert_ne!(a.identity_sha256(), b.identity_sha256());
        assert!(Qwen35ImageMropePlan::new(&layout, 9, 4096).is_err());
        assert!(Qwen35ImageMropePlan::new(&layout, 16, 4095).is_err());
        assert!(a.row(16).is_err());
        let (c, s) = a.tables().unwrap();
        assert_eq!(c.len(), 512);
        assert_eq!(s.len(), 512);
        assert!(!a.receipt().target_image_consumed);
        assert!(!a.receipt().product_ready);
    }
}
