use crate::{
    CpuFeatureSet, DualScaleQ8HotCpuLaunch, Fp8FusedMlpF32CpuLaunch, Fp8ScaleGridF32CpuLaunch,
    Iq2XxsQ8HotCpuLaunch, KernelKey, Mxfp4F32CpuLaunch, Q2KQ8HotCpuLaunch, Q8_0Q8HotCpuLaunch,
    QMatmulBiasF32Launch, QMatmulF32Launch, QMatmulF32PrefillLaunch, ShapeClass,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct KernelRegistrationId(pub &'static str);

/// Stable semantic identity of a registration shape predicate.
///
/// This ID, rather than the function-pointer address, participates in the
/// persistent registry fingerprint.
///
/// Predicate semantics must bump the versioned ID whenever they change:
///
/// ```text
/// cpu.shape.decode_qmatmul.v1
/// → cpu.shape.decode_qmatmul.v2
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CpuShapeSupportId(pub &'static str);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CpuKernelAbi {
    QMatmulF32,
    QMatmulF32Prefill,
    QMatmulBiasF32,
    Mxfp4F32,
    Fp8ScaleGridF32,
    Fp8FusedMlpF32,

    /// Concrete Rust ABI for:
    ///
    /// ```text
    /// DualScaleHotCpuTensor
    /// × Q8KActivationHotCpuVector
    /// → f32 output
    /// ```
    DualScaleQ8Hot,
    Q8_0Q8Hot,
    Q2KQ8Hot,
    Iq2XxsQ8Hot,
}

/// Concrete Rust-to-Rust launch target stored in a static registration.
///
/// `Copy` copies only the function pointer value. It does not clone kernel
/// resources, buffers, HotWeight owners, plans, closures or captured state.
///
/// This enum is used only at registry/lowering boundaries. ExecutionPlan
/// extracts and stores the already-typed function pointer.
#[derive(Clone, Copy, Debug)]
pub enum CpuKernelLaunch {
    QMatmulF32(QMatmulF32Launch),

    QMatmulF32Prefill(QMatmulF32PrefillLaunch),

    QMatmulBiasF32(QMatmulBiasF32Launch),
    Mxfp4F32(Mxfp4F32CpuLaunch),
    Fp8ScaleGridF32(Fp8ScaleGridF32CpuLaunch),
    Fp8FusedMlpF32(Fp8FusedMlpF32CpuLaunch),

    DualScaleQ8Hot(DualScaleQ8HotCpuLaunch),
    Q8_0Q8Hot(Q8_0Q8HotCpuLaunch),
    Q2KQ8Hot(Q2KQ8HotCpuLaunch),
    Iq2XxsQ8Hot(Iq2XxsQ8HotCpuLaunch),
}

impl CpuKernelLaunch {
    pub const fn abi(self) -> CpuKernelAbi {
        match self {
            Self::QMatmulF32(_) => CpuKernelAbi::QMatmulF32,

            Self::QMatmulF32Prefill(_) => CpuKernelAbi::QMatmulF32Prefill,

            Self::QMatmulBiasF32(_) => CpuKernelAbi::QMatmulBiasF32,
            Self::Mxfp4F32(_) => CpuKernelAbi::Mxfp4F32,
            Self::Fp8ScaleGridF32(_) => CpuKernelAbi::Fp8ScaleGridF32,
            Self::Fp8FusedMlpF32(_) => CpuKernelAbi::Fp8FusedMlpF32,

            Self::DualScaleQ8Hot(_) => CpuKernelAbi::DualScaleQ8Hot,
            Self::Q8_0Q8Hot(_) => CpuKernelAbi::Q8_0Q8Hot,
            Self::Q2KQ8Hot(_) => CpuKernelAbi::Q2KQ8Hot,
            Self::Iq2XxsQ8Hot(_) => CpuKernelAbi::Iq2XxsQ8Hot,
        }
    }

    pub const fn qmatmul_f32(self) -> Option<QMatmulF32Launch> {
        match self {
            Self::QMatmulF32(launch) => Some(launch),

            Self::QMatmulF32Prefill(_)
            | Self::QMatmulBiasF32(_)
            | Self::Mxfp4F32(_)
            | Self::Fp8ScaleGridF32(_)
            | Self::Fp8FusedMlpF32(_)
            | Self::DualScaleQ8Hot(_)
            | Self::Q8_0Q8Hot(_)
            | Self::Q2KQ8Hot(_)
            | Self::Iq2XxsQ8Hot(_) => None,
        }
    }

    pub const fn qmatmul_f32_prefill(self) -> Option<QMatmulF32PrefillLaunch> {
        match self {
            Self::QMatmulF32Prefill(launch) => Some(launch),
            Self::QMatmulF32(_)
            | Self::QMatmulBiasF32(_)
            | Self::Mxfp4F32(_)
            | Self::Fp8ScaleGridF32(_)
            | Self::Fp8FusedMlpF32(_)
            | Self::DualScaleQ8Hot(_)
            | Self::Q8_0Q8Hot(_)
            | Self::Q2KQ8Hot(_)
            | Self::Iq2XxsQ8Hot(_) => None,
        }
    }

    pub const fn qmatmul_bias_f32(self) -> Option<QMatmulBiasF32Launch> {
        match self {
            Self::QMatmulBiasF32(launch) => Some(launch),

            Self::QMatmulF32(_)
            | Self::QMatmulF32Prefill(_)
            | Self::Mxfp4F32(_)
            | Self::Fp8ScaleGridF32(_)
            | Self::Fp8FusedMlpF32(_)
            | Self::DualScaleQ8Hot(_)
            | Self::Q8_0Q8Hot(_)
            | Self::Q2KQ8Hot(_)
            | Self::Iq2XxsQ8Hot(_) => None,
        }
    }

    pub const fn dual_scale_q8hot(self) -> Option<DualScaleQ8HotCpuLaunch> {
        match self {
            Self::DualScaleQ8Hot(launch) => Some(launch),

            Self::QMatmulF32(_)
            | Self::QMatmulF32Prefill(_)
            | Self::QMatmulBiasF32(_)
            | Self::Mxfp4F32(_)
            | Self::Fp8ScaleGridF32(_)
            | Self::Fp8FusedMlpF32(_)
            | Self::Q8_0Q8Hot(_)
            | Self::Q2KQ8Hot(_)
            | Self::Iq2XxsQ8Hot(_) => None,
        }
    }

    pub const fn q8_0_q8hot(self) -> Option<Q8_0Q8HotCpuLaunch> {
        match self {
            Self::Q8_0Q8Hot(launch) => Some(launch),
            _ => None,
        }
    }

    pub const fn mxfp4_f32(self) -> Option<Mxfp4F32CpuLaunch> {
        match self {
            Self::Mxfp4F32(launch) => Some(launch),
            _ => None,
        }
    }

    pub const fn fp8_scale_grid_f32(self) -> Option<Fp8ScaleGridF32CpuLaunch> {
        match self {
            Self::Fp8ScaleGridF32(launch) => Some(launch),
            _ => None,
        }
    }

    pub const fn fp8_fused_mlp_f32(self) -> Option<Fp8FusedMlpF32CpuLaunch> {
        match self {
            Self::Fp8FusedMlpF32(launch) => Some(launch),
            _ => None,
        }
    }

    pub const fn q2k_q8hot(self) -> Option<Q2KQ8HotCpuLaunch> {
        match self {
            Self::Q2KQ8Hot(launch) => Some(launch),
            _ => None,
        }
    }

    pub const fn iq2_xxs_q8hot(self) -> Option<Iq2XxsQ8HotCpuLaunch> {
        match self {
            Self::Iq2XxsQ8Hot(launch) => Some(launch),
            _ => None,
        }
    }
}

pub type CpuShapePredicate = fn(&ShapeClass) -> bool;

/// Static CPU kernel registration contract.
///
/// This is protocol metadata, not a dynamic kernel object. The registration
/// owns no resources and performs no execution-time selection.
#[derive(Clone, Copy, Debug)]
pub struct CpuKernelRegistration {
    pub id: KernelRegistrationId,
    pub key: KernelKey,

    /// Stable versioned semantic identity of `shape_support`.
    pub shape_support_id: CpuShapeSupportId,

    pub shape_support: CpuShapePredicate,

    /// Higher values have higher selection priority.
    pub priority: u16,

    pub required_features: CpuFeatureSet,
    pub launch: CpuKernelLaunch,
}
