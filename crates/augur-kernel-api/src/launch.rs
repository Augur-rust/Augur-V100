use std::sync::Arc;

use crate::{
    DualScaleHotCpuTensor, Fp8E4M3ScaleGridHotCpuTensor, Iq2XxsHotCpuTensor, KernelLaunchError,
    Mxfp4HotCpuTensor, Q2KHotCpuTensor, Q4KHotCpuTensor, Q8KActivationHotCpuVector,
    Q8_0HotCpuTensor,
};

pub type Fp8ScaleGridF32CpuLaunch = fn(
    weight: &Fp8E4M3ScaleGridHotCpuTensor,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), KernelLaunchError>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FusedMlpF32Config {
    pub gate_min: Option<f32>,
    pub gate_max: Option<f32>,
    pub up_min: Option<f32>,
    pub up_max: Option<f32>,
}

impl FusedMlpF32Config {
    pub fn validate(self) -> Result<(), KernelLaunchError> {
        for value in [self.gate_min, self.gate_max, self.up_min, self.up_max]
            .into_iter()
            .flatten()
        {
            if !value.is_finite() {
                return Err(KernelLaunchError::BadRuntimeConfig);
            }
        }
        if (self.gate_min.is_none() && self.gate_max.is_none())
            || (self.up_min.is_none() && self.up_max.is_none())
            || matches!((self.gate_min, self.gate_max), (Some(min), Some(max)) if min > max)
            || matches!((self.up_min, self.up_max), (Some(min), Some(max)) if min > max)
        {
            return Err(KernelLaunchError::BadRuntimeConfig);
        }
        Ok(())
    }
}

pub type Fp8FusedMlpF32CpuLaunch = fn(
    gate_weight: &Fp8E4M3ScaleGridHotCpuTensor,
    up_weight: &Fp8E4M3ScaleGridHotCpuTensor,
    down_weight: &Fp8E4M3ScaleGridHotCpuTensor,
    input: &[f32],
    config: FusedMlpF32Config,
    workspace: &mut Fp8FusedMlpF32Workspace,
    output: &mut [f32],
) -> Result<(), KernelLaunchError>;

#[derive(Clone, Debug)]
pub struct Fp8FusedMlpF32Workspace {
    gate: Vec<f32>,
    up: Vec<f32>,
}

impl Fp8FusedMlpF32Workspace {
    pub fn new(intermediate: usize) -> Result<Self, KernelLaunchError> {
        if intermediate == 0 {
            return Err(KernelLaunchError::BadWorkspace);
        }
        Ok(Self {
            gate: vec![0.0; intermediate],
            up: vec![0.0; intermediate],
        })
    }

    pub fn intermediate(&self) -> usize {
        self.gate.len()
    }

    pub fn payload_bytes(&self) -> usize {
        (self.gate.len() + self.up.len()) * core::mem::size_of::<f32>()
    }

    pub fn parts_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        (&mut self.gate, &mut self.up)
    }
}

pub type Mxfp4F32CpuLaunch = fn(
    weight: &Mxfp4HotCpuTensor,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), KernelLaunchError>;
use crate::{ResidentMoeF32HotCpu, ResidentMoeF32Workspace};

pub type QMatmulF32Launch = fn(
    weight: &Q4KHotCpuTensor,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), KernelLaunchError>;

/// Rank-2 row-major `[m, k] x [n, k]^T -> [m, n]` Q4K prefill launch.
pub type QMatmulF32PrefillLaunch = fn(
    weight: &Q4KHotCpuTensor,
    input: &[f32],
    m: usize,
    output: &mut [f32],
) -> Result<(), KernelLaunchError>;

pub type QMatmulBiasF32Launch = fn(
    weight: &Q4KHotCpuTensor,
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
) -> Result<(), KernelLaunchError>;

pub type QuantizeQ8HotCpuLaunch =
    fn(input: &[f32], output: &mut Q8KActivationHotCpuVector) -> Result<(), KernelLaunchError>;

pub type DualScaleQ8HotCpuLaunch = fn(
    weight: &DualScaleHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
) -> Result<(), KernelLaunchError>;

pub type Q8_0Q8HotCpuLaunch = fn(
    weight: &Q8_0HotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
) -> Result<(), KernelLaunchError>;

pub type Q2KQ8HotCpuLaunch = fn(
    weight: &Q2KHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
) -> Result<(), KernelLaunchError>;

pub type Iq2XxsQ8HotCpuLaunch = fn(
    weight: &Iq2XxsHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
) -> Result<(), KernelLaunchError>;

/// Binder-produced proof that a DualScale HotWeight is structurally complete.
///
/// Construction performs the O(rows) reachability preflight once. The owner is
/// immutable and all fields are private, so execution can use an infallible
/// launch without repeating that work for every token.
#[derive(Clone, Debug)]
pub struct BoundDualScaleQ8Hot {
    weight: Arc<DualScaleHotCpuTensor>,
}

impl BoundDualScaleQ8Hot {
    pub fn try_new(weight: Arc<DualScaleHotCpuTensor>) -> Result<Self, KernelLaunchError> {
        let expected_cols = weight
            .blocks_per_row()
            .checked_mul(crate::DualScaleHotCpuBlock::DECODED_LEN)
            .ok_or(KernelLaunchError::BadWeightShape)?;
        if weight.rows() == 0 || weight.cols() != expected_cols {
            return Err(KernelLaunchError::BadWeightShape);
        }
        let expected_blocks = weight
            .rows()
            .checked_mul(weight.blocks_per_row())
            .ok_or(KernelLaunchError::BadWeightShape)?;
        if weight.blocks().len() != expected_blocks {
            return Err(KernelLaunchError::BadWeightBlockCount {
                expected: expected_blocks,
                actual: weight.blocks().len(),
            });
        }
        if weight.outliers().rows() != weight.rows() || weight.outliers().cols() != weight.cols() {
            return Err(KernelLaunchError::BadWeightShape);
        }
        weight
            .outliers()
            .validate_rows()
            .ok_or(KernelLaunchError::BadWeight)?;
        for row in 0..weight.rows() {
            weight.row_blocks(row).ok_or(KernelLaunchError::BadWeight)?;
        }
        Ok(Self { weight })
    }

    pub fn weight(&self) -> &DualScaleHotCpuTensor {
        &self.weight
    }

    pub fn weight_owner(&self) -> &Arc<DualScaleHotCpuTensor> {
        &self.weight
    }
}

pub type DualScaleQ8HotCpuPrevalidatedLaunch =
    fn(bound: &BoundDualScaleQ8Hot, activation: &Q8KActivationHotCpuVector, output: &mut [f32]);

pub type ResidentMoeF32CpuLaunch = fn(
    weights: &ResidentMoeF32HotCpu,
    input: &[f32],
    top_k: usize,
    routed_scale: f32,
    workspace: &mut ResidentMoeF32Workspace,
    output: &mut [f32],
) -> Result<(), KernelLaunchError>;
