use augur_core::{DeviceKind, RuntimeError};
use augur_kernel_api::{HotLayoutError, Q8KActivationHotCpuVector};

use crate::{Backend, BackendCapabilities};

#[derive(Clone, Debug, Default)]
pub struct CpuBackend;

#[derive(Clone, Debug, Default)]
pub struct CpuStream;

#[derive(Clone, Debug)]
pub enum CpuBufferData {
    F32(Vec<f32>),
    PackedU8(Vec<u8>),
    I8(Vec<i8>),
    I32(Vec<i32>),
    /// Fixed-shape CPU portable Q8Hot activation owner.
    Q8Hot(Q8KActivationHotCpuVector),
}

#[derive(Clone, Debug)]
pub struct CpuBuffer {
    pub data: CpuBufferData,
}

impl CpuBuffer {
    pub fn from_f32(data: Vec<f32>) -> Self {
        Self {
            data: CpuBufferData::F32(data),
        }
    }

    pub fn zeros_f32(n: usize) -> Self {
        Self::from_f32(vec![0.0; n])
    }

    pub fn from_q8hot(value: Q8KActivationHotCpuVector) -> Self {
        Self {
            data: CpuBufferData::Q8Hot(value),
        }
    }

    /// Binding/initialization-time Q8Hot allocation.
    ///
    /// This function must not be called from an ExecutionPlan execute loop.
    pub fn new_q8hot_zeroed(len_elements: usize) -> Result<Self, HotLayoutError> {
        let value = Q8KActivationHotCpuVector::new_zeroed(len_elements)?;
        Ok(Self::from_q8hot(value))
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.data {
            CpuBufferData::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        match &mut self.data {
            CpuBufferData::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_q8hot(&self) -> Option<&Q8KActivationHotCpuVector> {
        match &self.data {
            CpuBufferData::Q8Hot(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_q8hot_mut(&mut self) -> Option<&mut Q8KActivationHotCpuVector> {
        match &mut self.data {
            CpuBufferData::Q8Hot(v) => Some(v),
            _ => None,
        }
    }
}

impl Backend for CpuBackend {
    type Device = ();
    type Stream = CpuStream;
    type Buffer = CpuBuffer;
    type Event = ();

    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::cpu_baseline()
    }

    fn synchronize(&self, _stream: &mut Self::Stream) -> Result<(), RuntimeError> {
        Ok(())
    }
}
