//! Geometry-only encoder inputs. A position plan does not execute Vision.
use crate::{Qwen35Error, Qwen35ImageGrid};
use serde::Serialize;

pub const QWEN35_VISION_HIDDEN: usize = 1152;
pub const QWEN35_VISION_HEADS: usize = 16;
pub const QWEN35_VISION_HEAD_DIM: usize = 72;
pub const QWEN35_VISION_MAX_PATCHES: usize = 4096;

#[derive(Debug, Serialize)]
pub struct Qwen35VisionGeometry {
    grid_thw: [usize; 3],
    positions_hw: Vec<[u32; 2]>,
    bilinear_indices: Vec<[u32; 4]>,
    bilinear_weights: Vec<[f32; 4]>,
    rotary_cos: Vec<f32>,
    rotary_sin: Vec<f32>,
}

impl Qwen35VisionGeometry {
    /// Fixed 48x48 learned-position grid and 2x2 merge-group patch order.
    /// Interpolation and rotary phases stay FP32; no learned weight is approximated here.
    pub fn new(grid: Qwen35ImageGrid, max_plan_bytes: usize) -> Result<Self, Qwen35Error> {
        let [t, h, w] = grid.grid_thw();
        let patches = h * w;
        let bytes = patches
            .checked_mul(40 + QWEN35_VISION_HEAD_DIM * 8)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if t != 1 || patches > QWEN35_VISION_MAX_PATCHES || bytes > max_plan_bytes {
            return Err(Qwen35Error::invalid(
                "vision_geometry",
                "still-image geometry exceeds bounded encoder profile",
            ));
        }
        let mut result = Self {
            grid_thw: [t, h, w],
            positions_hw: Vec::with_capacity(patches),
            bilinear_indices: Vec::with_capacity(patches),
            bilinear_weights: Vec::with_capacity(patches),
            rotary_cos: Vec::with_capacity(patches * QWEN35_VISION_HEAD_DIM),
            rotary_sin: Vec::with_capacity(patches * QWEN35_VISION_HEAD_DIM),
        };
        let inv: Vec<f32> = (0..18)
            .map(|i| 1.0 / 10_000.0f32.powf((2 * i) as f32 / 36.0))
            .collect();
        for group_h in 0..h / 2 {
            for group_w in 0..w / 2 {
                for local_h in 0..2 {
                    for local_w in 0..2 {
                        let ph = group_h * 2 + local_h;
                        let pw = group_w * 2 + local_w;
                        result.positions_hw.push([ph as u32, pw as u32]);
                        let y = linspace(ph, h);
                        let x = linspace(pw, w);
                        let y0 = y.floor() as u32;
                        let x0 = x.floor() as u32;
                        let y1 = (y0 + 1).min(47);
                        let x1 = (x0 + 1).min(47);
                        let fy = y - y0 as f32;
                        let fx = x - x0 as f32;
                        result.bilinear_indices.push([
                            y0 * 48 + x0,
                            y0 * 48 + x1,
                            y1 * 48 + x0,
                            y1 * 48 + x1,
                        ]);
                        result.bilinear_weights.push([
                            (1.0 - fy) * (1.0 - fx),
                            (1.0 - fy) * fx,
                            fy * (1.0 - fx),
                            fy * fx,
                        ]);
                        for d in 0..QWEN35_VISION_HEAD_DIM {
                            let half = d % 36;
                            let position = if half < 18 { ph } else { pw };
                            let angle = position as f32 * inv[half % 18];
                            result.rotary_cos.push(angle.cos());
                            result.rotary_sin.push(angle.sin());
                        }
                    }
                }
            }
        }
        Ok(result)
    }
    pub fn patch_count(&self) -> usize {
        self.positions_hw.len()
    }
    pub fn grid_thw(&self) -> [usize; 3] {
        self.grid_thw
    }
    pub fn positions_hw(&self) -> &[[u32; 2]] {
        &self.positions_hw
    }
    pub fn bilinear_indices(&self) -> &[[u32; 4]] {
        &self.bilinear_indices
    }
    pub fn bilinear_weights(&self) -> &[[f32; 4]] {
        &self.bilinear_weights
    }
    pub fn rotary_cos(&self) -> &[f32] {
        &self.rotary_cos
    }
    pub fn rotary_sin(&self) -> &[f32] {
        &self.rotary_sin
    }
}

fn linspace(index: usize, count: usize) -> f32 {
    let step = 47.0f64 / (count - 1) as f64;
    // Symmetric endpoint construction avoids accumulating incremental FP32 drift.
    if index < count / 2 {
        (step * index as f64) as f32
    } else {
        (47.0 - step * (count - 1 - index) as f64) as f32
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Qwen35VisionWorkspacePlan {
    pub patches: usize,
    pub merged_tokens: usize,
    pub hidden_pair_bytes: usize,
    pub affine_scratch_bytes: usize,
    pub f16_scratch_bytes: usize,
    pub image_input_bytes: usize,
    pub position_input_bytes: usize,
    pub output_embedding_bytes: usize,
    pub explicit_workspace_bytes: usize,
    pub quadratic_attention_buffer_bytes: usize,
}

impl Qwen35VisionWorkspacePlan {
    /// Hidden ping-pong + phase-reused affine/FP16 scratch; no S*S score matrix.
    /// Driver/cuBLAS private workspace and resident weights are not included.
    pub fn new(patches: usize, max_bytes: usize) -> Result<Self, Qwen35Error> {
        if patches == 0 || patches > QWEN35_VISION_MAX_PATCHES || !patches.is_multiple_of(4) {
            return Err(Qwen35Error::invalid(
                "vision_workspace",
                "invalid patch count",
            ));
        }
        let hidden_pair_bytes = patches * 1152 * 4 * 2;
        let affine_scratch_bytes = patches * 4304 * 4;
        let f16_scratch_bytes = patches * 4304 * 2;
        let image_input_bytes = patches * 1536 * 2;
        let position_input_bytes = patches * (4 * 4 * 2 + 72 * 4 * 2);
        let output_embedding_bytes = patches / 4 * 5120 * 4;
        let explicit_workspace_bytes = hidden_pair_bytes
            + affine_scratch_bytes
            + f16_scratch_bytes
            + image_input_bytes
            + position_input_bytes
            + output_embedding_bytes
            + 4;
        if explicit_workspace_bytes > max_bytes {
            return Err(Qwen35Error::invalid(
                "vision_workspace",
                "workspace budget exceeded",
            ));
        }
        Ok(Self {
            patches,
            merged_tokens: patches / 4,
            hidden_pair_bytes,
            affine_scratch_bytes,
            f16_scratch_bytes,
            image_input_bytes,
            position_input_bytes,
            output_embedding_bytes,
            explicit_workspace_bytes,
            quadratic_attention_buffer_bytes: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn positions_follow_merge_groups_not_raster_order() {
        let plan =
            Qwen35VisionGeometry::new(Qwen35ImageGrid::from_aligned_pixels(32, 64).unwrap(), 8192)
                .unwrap();
        assert_eq!(
            plan.positions_hw(),
            &[
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3]
            ]
        );
        assert_eq!(plan.bilinear_indices()[0][0], 0);
        assert_eq!(plan.bilinear_indices()[7][0], 2303);
    }
    #[test]
    fn position_weights_and_rotary_are_finite() {
        for (h, w) in [(32, 32), (64, 96), (416, 640), (1024, 1024)] {
            let plan = Qwen35VisionGeometry::new(
                Qwen35ImageGrid::from_aligned_pixels(h, w).unwrap(),
                4 * 1024 * 1024,
            )
            .unwrap();
            for (ids, weights) in plan.bilinear_indices.iter().zip(&plan.bilinear_weights) {
                assert!(ids.iter().all(|&i| i < 2304));
                assert!(weights
                    .iter()
                    .all(|&w| w.is_finite() && (0.0..=1.0).contains(&w)));
                assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            }
            assert!(plan
                .rotary_cos
                .iter()
                .chain(&plan.rotary_sin)
                .all(|v| v.is_finite()));
            assert!(plan.rotary_cos[..72].iter().all(|&v| v == 1.0));
            assert!(plan.rotary_sin[..72].iter().all(|&v| v == 0.0));
        }
    }
    #[test]
    fn geometry_and_workspace_budget_reject_before_allocation() {
        let grid = Qwen35ImageGrid::from_aligned_pixels(32, 32).unwrap();
        assert!(Qwen35VisionGeometry::new(grid, 2463).is_err());
        assert!(Qwen35VisionGeometry::new(
            Qwen35ImageGrid::from_aligned_pixels(2048, 2048).unwrap(),
            usize::MAX
        )
        .is_err());
        assert!(Qwen35VisionWorkspacePlan::new(0, usize::MAX).is_err());
        assert!(Qwen35VisionWorkspacePlan::new(5, usize::MAX).is_err());
        let plan = Qwen35VisionWorkspacePlan::new(4096, usize::MAX).unwrap();
        assert!(plan.explicit_workspace_bytes < 192 * 1024 * 1024);
        assert_eq!(plan.quadratic_attention_buffer_bytes, 0);
        assert!(Qwen35VisionWorkspacePlan::new(4096, plan.explicit_workspace_bytes - 1).is_err());
    }
}
