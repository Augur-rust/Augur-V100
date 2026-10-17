use augur_core::{DeviceKind, RuntimeError};

use crate::BackendCapabilities;

pub trait Backend: 'static {
    type Device;
    type Stream;
    type Buffer;
    type Event;

    fn kind(&self) -> DeviceKind;
    fn capabilities(&self) -> BackendCapabilities;
    fn synchronize(&self, stream: &mut Self::Stream) -> Result<(), RuntimeError>;
}
