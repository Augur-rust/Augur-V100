#![cfg(feature = "cuda")]

use augur_core::{FormatVersion, PhysicalLayoutId, QuantFormatId, StorageId};
use augur_exec::MemoryPlan;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CudaWeightBindingId(u32);

impl CudaWeightBindingId {
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaQ4kWeightRequirement {
    pub binding: CudaWeightBindingId,
    pub storage: StorageId,
    pub quant_format: QuantFormatId,
    pub format_version: FormatVersion,
    pub physical_layout: PhysicalLayoutId,
    pub rows: usize,
    pub cols: usize,
    pub block_size: usize,
    pub payload_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaCompiledQ4kF32Decode {
    pub weight: CudaWeightBindingId,
    pub input: CudaCompiledF32View,
    pub output: CudaCompiledF32View,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaCompiledQ4kF32Prefill {
    pub weight: CudaWeightBindingId,
    pub input: CudaCompiledF32View,
    pub output: CudaCompiledF32View,
    pub m: usize,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaCompiledAddF32 {
    pub a: CudaCompiledF32View,
    pub b: CudaCompiledF32View,
    pub output: CudaCompiledF32View,
    pub elements: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaCompiledF32View {
    pub buffer_index: usize,
    pub len_elements: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaCompiledStep {
    Q4kF32Decode(CudaCompiledQ4kF32Decode),
    Q4kF32Prefill(CudaCompiledQ4kF32Prefill),
    AddF32(CudaCompiledAddF32),
}

#[derive(Clone, Debug)]
pub struct CudaCompiledPlan {
    pub(crate) device_ordinal: i32,
    pub(crate) input_f32: Box<[f32]>,
    pub(crate) steps: Box<[CudaCompiledStep]>,
    pub(crate) weight_requirements: Box<[CudaQ4kWeightRequirement]>,
    pub(crate) memory: MemoryPlan,
    pub(crate) input: CudaCompiledF32View,
    pub(crate) output: CudaCompiledF32View,
}

impl CudaCompiledPlan {
    pub fn steps(&self) -> &[CudaCompiledStep] {
        &self.steps
    }

    pub fn weight_requirements(&self) -> &[CudaQ4kWeightRequirement] {
        &self.weight_requirements
    }

    pub const fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    pub fn memory(&self) -> &MemoryPlan {
        &self.memory
    }

    pub const fn input(&self) -> CudaCompiledF32View {
        self.input
    }

    pub const fn output(&self) -> CudaCompiledF32View {
        self.output
    }
}
