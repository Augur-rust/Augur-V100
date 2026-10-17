use augur_core::{
    CompileError, DeviceKind, FormatVersion, PhysicalLayoutId, QuantFormatId, StorageId,
};
use augur_kernel_api::{BackendKind, CpuFeatureSet, CpuKernelAbi, KernelRegistrationId};

use crate::{
    CpuKernelRequest, CpuTuningShapeError, CpuWeightResourceKind, KernelSelectionError,
    WeightBindingId,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompilerError {
    Graph(CompileError),
    Lowering(LoweringError),
    Binding(BindError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LoweringError {
    Core(CompileError),
    KernelSelection(KernelSelectionError),
    RegistryInvariant(RegistryInvariantError),

    ConflictingWeightRequirement {
        storage: StorageId,
    },

    TooManyWeightRequirements,

    InvalidWeightBinding {
        binding: WeightBindingId,
    },

    UnsupportedDualScaleEpilogue {
        epilogue: augur_kernel_api::EpilogueKind,
    },

    CpuTuningShape(CpuTuningShapeError),

    CompiledPlanInvariant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceInsertError {
    TooManyResources,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceLookupError {
    IndexOutOfRange {
        requested: StorageId,
    },
    GenerationMismatch {
        requested: StorageId,
        actual: StorageId,
    },
    AlreadyTaken {
        requested: StorageId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuBindStepKind {
    Add,
    Mul,
    Clamp,
    Silu,
    QuantizeQ8Hot,
    DualScaleQ8Hot,
    Q8_0Q8Hot,
    Q2KQ8Hot,
    Iq2XxsQ8Hot,
    Mxfp4F32,
    Fp8ScaleGridF32,
    Fp8FusedMlpF32,
    QMatmulF32,
    QMatmulF32Prefill,
    QMatmulBiasF32,
    QMatmulBiasSiluF32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuBindBufferRole {
    Input,
    Left,
    Right,
    Activation,
    Bias,
    Output,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompiledStepBindFailure {
    BufferIndexOutOfRange {
        role: CpuBindBufferRole,
        index: usize,
        buffer_count: usize,
    },

    AliasedBuffers {
        read: usize,
        write: usize,
    },

    MetadataMismatch {
        role: CpuBindBufferRole,
        buffer_index: usize,
    },

    UnsupportedOffset {
        role: CpuBindBufferRole,
        actual: usize,
    },

    OwnerKindMismatch {
        role: CpuBindBufferRole,
        buffer_index: usize,
    },

    BufferBounds {
        role: CpuBindBufferRole,
        buffer_index: usize,
        offset: usize,
        len: usize,
        owner_len: usize,
    },

    OwnerShapeMismatch {
        role: CpuBindBufferRole,
        buffer_index: usize,
        expected_len: usize,
        actual_len: usize,
        expected_blocks: Option<usize>,
        actual_blocks: Option<usize>,
    },

    InvalidDimensions {
        rows: usize,
        cols: usize,
        blocks_per_row: usize,
    },

    WeightRequirementMismatch {
        binding: WeightBindingId,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BindError {
    ResourceLookup(ResourceLookupError),

    InvalidBindingId {
        binding: WeightBindingId,
    },

    DuplicateStorageRequirement {
        storage: StorageId,
    },

    WrongResourceKind {
        storage: StorageId,
        expected: CpuWeightResourceKind,
        actual: CpuWeightResourceKind,
    },

    FormatMismatch {
        storage: StorageId,
        expected: QuantFormatId,
        actual: QuantFormatId,
    },

    VersionMismatch {
        storage: StorageId,
        expected: FormatVersion,
        actual: FormatVersion,
    },

    LayoutMismatch {
        storage: StorageId,
        expected: PhysicalLayoutId,
        actual: PhysicalLayoutId,
    },

    ShapeMismatch {
        storage: StorageId,
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },

    BlockSizeMismatch {
        storage: StorageId,
        expected: usize,
        actual: usize,
    },

    InvalidWeightShape {
        storage: StorageId,
    },

    CompiledStepShapeMismatch {
        binding: WeightBindingId,
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },

    QuantizeBufferIndexOutOfRange {
        index: usize,
        buffer_count: usize,
    },

    QuantizeAliasedBuffers {
        buffer_index: usize,
    },

    QuantizeInputMetadataMismatch {
        buffer_index: usize,
    },

    QuantizeOutputMetadataMismatch {
        buffer_index: usize,
    },

    /// M26-C3 supports only a complete mutable Q8Hot owner view.
    QuantizeOutputOffsetUnsupported {
        actual: usize,
    },

    QuantizeLengthMismatch {
        input: usize,
        output: usize,
    },

    QuantizeOutputShapeInvalid {
        len_elements: usize,
    },

    QuantizeInputOwnerMismatch {
        buffer_index: usize,
    },

    QuantizeOutputOwnerMismatch {
        buffer_index: usize,
    },

    QuantizeInputBounds {
        buffer_index: usize,
        offset: usize,
        len: usize,
        owner_len: usize,
    },

    QuantizeOutputOwnerShapeMismatch {
        buffer_index: usize,
        view_len: usize,
        owner_len: usize,
        owner_blocks: usize,
    },

    CompiledStep {
        step: CpuBindStepKind,
        failure: CompiledStepBindFailure,
    },

    AccountingOverflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelSelectionFailure {
    BackendMismatch {
        actual: DeviceKind,
    },

    /// A CPU selector received a kernel identity for another backend.
    RequestBackendMismatch {
        actual: BackendKind,
    },

    NoExactKey,
    MissingCpuFeatures {
        available: CpuFeatureSet,
        required: CpuFeatureSet,
    },
    UnsupportedShape,
    LaunchAbiMismatch {
        expected: CpuKernelAbi,
        actual: CpuKernelAbi,
    },
    AmbiguousHighestPriority {
        priority: u16,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistryInvariantError {
    UnexpectedLaunchAbi {
        registration: KernelRegistrationId,
        expected: CpuKernelAbi,
        actual: CpuKernelAbi,
    },
}

impl From<CompileError> for LoweringError {
    fn from(value: CompileError) -> Self {
        Self::Core(value)
    }
}

impl From<CpuTuningShapeError> for LoweringError {
    fn from(value: CpuTuningShapeError) -> Self {
        Self::CpuTuningShape(value)
    }
}

impl From<KernelSelectionError> for LoweringError {
    fn from(value: KernelSelectionError) -> Self {
        Self::KernelSelection(value)
    }
}

impl From<ResourceLookupError> for BindError {
    fn from(value: ResourceLookupError) -> Self {
        Self::ResourceLookup(value)
    }
}

impl CompilerError {
    pub fn lowering_core(error: CompileError) -> Self {
        Self::Lowering(LoweringError::Core(error))
    }
}

impl KernelSelectionFailure {
    pub fn request(self, request: CpuKernelRequest) -> KernelSelectionError {
        KernelSelectionError {
            request,
            failure: self,
        }
    }
}
