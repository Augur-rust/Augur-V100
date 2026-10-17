use augur_core::{DeviceKind, RuntimeError};

use crate::{Backend, BackendCapabilities};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CudaBackend;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaDevice {
    pub ordinal: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaStream {
    pub handle: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEvent {
    pub handle: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaBuffer {
    /// Opaque device pointer value.
    ///
    /// M12 skeleton is non-owning: this handle never allocates and never
    /// frees device memory. Ownership and Drop belong to the future Raw
    /// FFI / runtime layer.
    pub ptr: u64,

    pub len_bytes: usize,
    pub alignment: usize,
}

impl Backend for CudaBackend {
    type Device = CudaDevice;
    type Stream = CudaStream;
    type Buffer = CudaBuffer;
    type Event = CudaEvent;

    fn kind(&self) -> DeviceKind {
        DeviceKind::Cuda
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::cuda_skeleton()
    }

    fn synchronize(&self, _stream: &mut Self::Stream) -> Result<(), RuntimeError> {
        Err(RuntimeError::BackendUnavailable {
            backend: DeviceKind::Cuda,
        })
    }
}
