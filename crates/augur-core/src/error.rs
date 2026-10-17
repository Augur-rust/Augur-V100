use crate::{DType, DeviceKind, QuantFormatId};

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GgmlError {
    Ok = 0,
    Null = -1,
    Oom = -2,
    Cycle = -3,
    BadData = -4,
    UnsupportedOp = -5,
    MissingFreeCallback = -6,
    ShapeMismatch = -7,
    DTypeMismatch = -8,
    InvalidTensor = -9,
    InvalidNode = -10,
    BackendUnsupported = -11,
    DuplicateProducer = -12,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompileError {
    Graph(GgmlError),
    Cycle,
    InvalidInput,
    InvalidOutput,
    InvalidNode,
    UnsupportedOp,
    BackendUnsupported,
    DuplicateProducer,

    /// Lowering expected a plain f32 QMatmul input but received a quantized
    /// or non-f32 activation.
    ///
    /// This protects the existing f32 QMatmul lowering path. It does not
    /// mean quantized activations are globally unsupported. A future
    /// quantized-activation lowering path must add its own PlannedOp / input
    /// abstraction.
    ExpectedPlainF32QMatmulInput {
        actual_dtype: DType,
        actual_quant: Option<QuantFormatId>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeError {
    BadPlan,
    BadBuffer,
    KernelFailed,
    BackendUnsupported,

    /// M12: the backend is a compile-time skeleton and does not support
    /// runtime operations (synchronize, etc.).
    BackendUnavailable {
        backend: DeviceKind,
    },
}
