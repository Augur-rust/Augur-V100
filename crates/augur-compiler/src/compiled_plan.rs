use augur_backend::{CpuBackend, CpuBuffer};
use augur_core::{FormatVersion, PhysicalLayoutId, QuantFormatId, StorageId};
use augur_exec::{BufferView, MemoryPlan};
use augur_kernel_api::{
    DualScaleQ8HotCpuLaunch, Fp8FusedMlpF32CpuLaunch, Fp8ScaleGridF32CpuLaunch, FusedMlpF32Config,
    Iq2XxsQ8HotCpuLaunch, Mxfp4F32CpuLaunch, Q2KQ8HotCpuLaunch, Q8_0Q8HotCpuLaunch,
    QMatmulBiasF32Launch, QMatmulF32Launch, QMatmulF32PrefillLaunch, QuantizeQ8HotCpuLaunch,
};

use crate::LoweringError;

/// Compiled-plan-local binding identity.
///
/// This identity exists only between lowering and resource binding. It must
/// not enter Graph IR or the final executable step.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct WeightBindingId(u32);

impl WeightBindingId {
    pub fn index(self) -> usize {
        self.0 as usize
    }

    #[cfg(test)]
    pub fn test(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuWeightResourceKind {
    Q4KHotCpuPortable,
    DualScaleHotCpuPortable,
    Q8_0HotCpuPortable,
    Q2KHotCpuPortable,
    Iq2XxsHotCpuPortable,
    Mxfp4HotCpuPortable,
    Fp8ScaleGridHotCpuPortable,
}

/// Exact resource requirement established by lowering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuWeightRequirement {
    pub binding: WeightBindingId,
    pub storage: StorageId,
    pub kind: CpuWeightResourceKind,
    pub quant_format: QuantFormatId,
    pub format_version: FormatVersion,
    pub block_size: usize,
    pub rows: usize,
    pub cols: usize,
    pub physical_layout: PhysicalLayoutId,
}

/// Requirement before a compiled-plan-local binding ID is assigned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuWeightRequirementSpec {
    pub storage: StorageId,
    pub kind: CpuWeightResourceKind,
    pub quant_format: QuantFormatId,
    pub format_version: FormatVersion,
    pub block_size: usize,
    pub rows: usize,
    pub cols: usize,
    pub physical_layout: PhysicalLayoutId,
}

impl CpuWeightRequirement {
    fn matches_spec(&self, spec: CpuWeightRequirementSpec) -> bool {
        self.storage == spec.storage
            && self.kind == spec.kind
            && self.quant_format == spec.quant_format
            && self.format_version == spec.format_version
            && self.block_size == spec.block_size
            && self.rows == spec.rows
            && self.cols == spec.cols
            && self.physical_layout == spec.physical_layout
    }
}

/// Compiler-side interning table for immutable weight requirements.
///
/// The expected number of model weights in one graph is small enough that a
/// linear scan is preferable to adding a dynamic lookup structure to the
/// initial protocol.
#[derive(Clone, Debug, Default)]
pub struct WeightRequirementBuilder {
    requirements: Vec<CpuWeightRequirement>,
}

impl WeightRequirementBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(
        &mut self,
        spec: CpuWeightRequirementSpec,
    ) -> Result<WeightBindingId, LoweringError> {
        if let Some(existing) = self.requirements.iter().find(|r| r.storage == spec.storage) {
            if existing.matches_spec(spec) {
                return Ok(existing.binding);
            }
            return Err(LoweringError::ConflictingWeightRequirement {
                storage: spec.storage,
            });
        }

        let index = u32::try_from(self.requirements.len())
            .map_err(|_| LoweringError::TooManyWeightRequirements)?;

        let binding = WeightBindingId(index);

        self.requirements.push(CpuWeightRequirement {
            binding,
            storage: spec.storage,
            kind: spec.kind,
            quant_format: spec.quant_format,
            format_version: spec.format_version,
            block_size: spec.block_size,
            rows: spec.rows,
            cols: spec.cols,
            physical_layout: spec.physical_layout,
        });

        Ok(binding)
    }

    pub fn requirements(&self) -> &[CpuWeightRequirement] {
        &self.requirements
    }

    pub fn finish(self) -> Box<[CpuWeightRequirement]> {
        self.requirements.into_boxed_slice()
    }
}

#[derive(Clone, Debug)]
pub struct CompiledAdd {
    pub dst: BufferView<CpuBackend>,
    pub a: BufferView<CpuBackend>,
    pub b: BufferView<CpuBackend>,
    pub n: usize,
}

#[derive(Clone, Debug)]
pub struct CompiledMul {
    pub dst: BufferView<CpuBackend>,
    pub a: BufferView<CpuBackend>,
    pub b: BufferView<CpuBackend>,
    pub n: usize,
}

#[derive(Clone, Debug)]
pub struct CompiledClamp {
    pub dst: BufferView<CpuBackend>,
    pub input: BufferView<CpuBackend>,
    pub n: usize,
    pub min: Option<f32>,
    pub max: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct CompiledSilu {
    pub dst: BufferView<CpuBackend>,
    pub input: BufferView<CpuBackend>,
    pub n: usize,
}

/// Compiled allocation-free f32 -> CPU portable Q8Hot preparation.
///
/// The launch target is already fixed. This step does not imply registry
/// lookup or automatic Graph insertion.
#[derive(Clone, Debug)]
pub struct CompiledQuantizeQ8Hot {
    pub input: BufferView<CpuBackend>,
    pub output: BufferView<CpuBackend>,
    pub launch: QuantizeQ8HotCpuLaunch,
}

#[derive(Clone, Debug)]
pub struct CompiledQMatmulF32Base {
    pub weight: WeightBindingId,
    pub input: BufferView<CpuBackend>,
    pub output: BufferView<CpuBackend>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct CompiledQMatmulF32 {
    pub base: CompiledQMatmulF32Base,
    pub launch: QMatmulF32Launch,
}

#[derive(Clone, Debug)]
pub struct CompiledQMatmulF32Prefill {
    pub weight: WeightBindingId,
    pub input: BufferView<CpuBackend>,
    pub output: BufferView<CpuBackend>,
    pub m: usize,
    pub rows: usize,
    pub cols: usize,
    pub launch: QMatmulF32PrefillLaunch,
}

#[derive(Clone, Debug)]
pub struct CompiledQMatmulBiasF32 {
    pub base: CompiledQMatmulF32Base,
    pub bias: BufferView<CpuBackend>,
    pub launch: QMatmulBiasF32Launch,
}

/// Compiled CPU portable DualScale × Q8Hot Identity qmatvec.
///
/// Registry selection is already complete. The external immutable HotWeight
/// owner remains represented by `WeightBindingId` until binding.
#[derive(Clone, Debug)]
pub struct CompiledDualScaleQ8Hot {
    pub weight: WeightBindingId,

    pub activation: BufferView<CpuBackend>,
    pub output: BufferView<CpuBackend>,

    pub rows: usize,
    pub cols: usize,
    pub blocks_per_row: usize,

    pub launch: DualScaleQ8HotCpuLaunch,
}

#[derive(Clone, Debug)]
pub struct CompiledQ8_0Q8Hot {
    pub weight: WeightBindingId,
    pub activation: BufferView<CpuBackend>,
    pub output: BufferView<CpuBackend>,
    pub rows: usize,
    pub cols: usize,
    pub launch: Q8_0Q8HotCpuLaunch,
}

#[derive(Clone, Debug)]
pub struct CompiledQ2KQ8Hot {
    pub weight: WeightBindingId,
    pub activation: BufferView<CpuBackend>,
    pub output: BufferView<CpuBackend>,
    pub rows: usize,
    pub cols: usize,
    pub launch: Q2KQ8HotCpuLaunch,
}

#[derive(Clone, Debug)]
pub struct CompiledIq2XxsQ8Hot {
    pub weight: WeightBindingId,
    pub activation: BufferView<CpuBackend>,
    pub output: BufferView<CpuBackend>,
    pub rows: usize,
    pub cols: usize,
    pub launch: Iq2XxsQ8HotCpuLaunch,
}

#[derive(Clone, Debug)]
pub struct CompiledMxfp4F32 {
    pub weight: WeightBindingId,
    pub input: BufferView<CpuBackend>,
    pub output: BufferView<CpuBackend>,
    pub rows: usize,
    pub cols: usize,
    pub launch: Mxfp4F32CpuLaunch,
}

#[derive(Clone, Debug)]
pub struct CompiledFp8ScaleGridF32 {
    pub weight: WeightBindingId,
    pub input: BufferView<CpuBackend>,
    pub output: BufferView<CpuBackend>,
    pub rows: usize,
    pub cols: usize,
    pub launch: Fp8ScaleGridF32CpuLaunch,
}

#[derive(Clone, Debug)]
pub struct CompiledFp8FusedMlpF32 {
    pub gate_weight: WeightBindingId,
    pub up_weight: WeightBindingId,
    pub down_weight: WeightBindingId,
    pub input: BufferView<CpuBackend>,
    pub output: BufferView<CpuBackend>,
    pub hidden: usize,
    pub intermediate: usize,
    pub config: FusedMlpF32Config,
    pub launch: Fp8FusedMlpF32CpuLaunch,
}

#[derive(Clone, Debug)]
pub enum CpuCompiledStep {
    Add(CompiledAdd),
    Mul(CompiledMul),
    Clamp(CompiledClamp),
    Silu(CompiledSilu),
    QuantizeQ8Hot(CompiledQuantizeQ8Hot),

    DualScaleQ8Hot(CompiledDualScaleQ8Hot),
    Q8_0Q8Hot(CompiledQ8_0Q8Hot),
    Q2KQ8Hot(CompiledQ2KQ8Hot),
    Iq2XxsQ8Hot(CompiledIq2XxsQ8Hot),
    Mxfp4F32(CompiledMxfp4F32),
    Fp8ScaleGridF32(CompiledFp8ScaleGridF32),
    Fp8FusedMlpF32(CompiledFp8FusedMlpF32),

    QMatmulF32(CompiledQMatmulF32),
    QMatmulF32Prefill(CompiledQMatmulF32Prefill),
    QMatmulBiasF32(CompiledQMatmulBiasF32),
    QMatmulBiasSiluF32(CompiledQMatmulBiasF32),
}

/// A CPU plan with all kernel decisions fixed but external weight owners not
/// yet bound.
///
/// This type must not contain `CpuWeightStore`, concrete HotWeight owners,
/// Compact owners or Graph references.
#[derive(Clone, Debug)]
pub struct CpuCompiledPlan {
    pub(crate) steps: Box<[CpuCompiledStep]>,
    pub(crate) buffers: Vec<CpuBuffer>,
    pub(crate) memory: MemoryPlan,
    pub(crate) weight_requirements: Box<[CpuWeightRequirement]>,
}

impl CpuCompiledPlan {
    pub fn steps(&self) -> &[CpuCompiledStep] {
        &self.steps
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    pub fn weight_requirements(&self) -> &[CpuWeightRequirement] {
        &self.weight_requirements
    }

    pub fn memory(&self) -> &MemoryPlan {
        &self.memory
    }
}

pub fn q4k_requirement(
    storage: StorageId,
    format_version: FormatVersion,
    block_size: usize,
    rows: usize,
    cols: usize,
) -> CpuWeightRequirementSpec {
    CpuWeightRequirementSpec {
        storage,
        kind: CpuWeightResourceKind::Q4KHotCpuPortable,
        quant_format: QuantFormatId::Q4K,
        format_version,
        block_size,
        rows,
        cols,
        physical_layout: PhysicalLayoutId::Q4K_HOT_CPU_PORTABLE_V1,
    }
}

/// Exact immutable HotWeight requirement for the CPU portable DualScale
/// layout.
///
/// This function creates binding metadata only. It does not perform repack,
/// registry lookup, executable-step construction or kernel execution.
pub fn dual_scale_requirement(
    storage: StorageId,
    format_version: FormatVersion,
    block_size: usize,
    rows: usize,
    cols: usize,
) -> CpuWeightRequirementSpec {
    CpuWeightRequirementSpec {
        storage,
        kind: CpuWeightResourceKind::DualScaleHotCpuPortable,
        quant_format: QuantFormatId::DualScale,
        format_version,
        block_size,
        rows,
        cols,
        physical_layout: PhysicalLayoutId::DUAL_SCALE_HOT_CPU_PORTABLE_V1,
    }
}

pub fn q8_0_requirement(
    storage: StorageId,
    format_version: FormatVersion,
    block_size: usize,
    rows: usize,
    cols: usize,
) -> CpuWeightRequirementSpec {
    CpuWeightRequirementSpec {
        storage,
        kind: CpuWeightResourceKind::Q8_0HotCpuPortable,
        quant_format: QuantFormatId::Q8_0,
        format_version,
        block_size,
        rows,
        cols,
        physical_layout: PhysicalLayoutId::Q8_0_HOT_CPU_PORTABLE_V1,
    }
}

pub fn q2k_requirement(
    storage: StorageId,
    format_version: FormatVersion,
    block_size: usize,
    rows: usize,
    cols: usize,
) -> CpuWeightRequirementSpec {
    CpuWeightRequirementSpec {
        storage,
        kind: CpuWeightResourceKind::Q2KHotCpuPortable,
        quant_format: QuantFormatId::Q2K,
        format_version,
        block_size,
        rows,
        cols,
        physical_layout: PhysicalLayoutId::Q2K_HOT_CPU_PORTABLE_V1,
    }
}

pub fn iq2_xxs_requirement(
    storage: StorageId,
    format_version: FormatVersion,
    block_size: usize,
    rows: usize,
    cols: usize,
) -> CpuWeightRequirementSpec {
    CpuWeightRequirementSpec {
        storage,
        kind: CpuWeightResourceKind::Iq2XxsHotCpuPortable,
        quant_format: QuantFormatId::IQ2_XXS,
        format_version,
        block_size,
        rows,
        cols,
        physical_layout: PhysicalLayoutId::IQ2_XXS_HOT_CPU_PORTABLE_V1,
    }
}

pub fn mxfp4_requirement(
    storage: StorageId,
    format_version: FormatVersion,
    block_size: usize,
    rows: usize,
    cols: usize,
) -> CpuWeightRequirementSpec {
    CpuWeightRequirementSpec {
        storage,
        kind: CpuWeightResourceKind::Mxfp4HotCpuPortable,
        quant_format: QuantFormatId::MXFP4,
        format_version,
        block_size,
        rows,
        cols,
        physical_layout: PhysicalLayoutId::MXFP4_HOT_CPU_PORTABLE_V1,
    }
}

pub fn fp8_scale_grid_requirement(
    storage: StorageId,
    format_version: FormatVersion,
    block_size: usize,
    rows: usize,
    cols: usize,
) -> CpuWeightRequirementSpec {
    CpuWeightRequirementSpec {
        storage,
        kind: CpuWeightResourceKind::Fp8ScaleGridHotCpuPortable,
        quant_format: QuantFormatId::FP8E4M3ScaleGrid,
        format_version,
        block_size,
        rows,
        cols,
        physical_layout: PhysicalLayoutId::FP8_E4M3_SCALE_GRID_HOT_CPU_PORTABLE_V1,
    }
}
