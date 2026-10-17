use augur_core::{DType, DeviceKind, PhysicalLayoutId, QuantFormatId};
use augur_kernel_api::{CpuFeatureSet, EpilogueKind, OpKind};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CapabilityQuery {
    pub op: OpKind,
    pub dtype: DType,
    pub quant: Option<QuantFormatId>,
    pub layout: PhysicalLayoutId,
    pub epilogue: EpilogueKind,
}

#[derive(Clone, Debug)]
pub struct BackendCapabilities {
    pub backend: DeviceKind,
    pub cpu_features: CpuFeatureSet,
}

impl BackendCapabilities {
    pub fn cpu_baseline() -> Self {
        Self {
            backend: DeviceKind::Cpu,
            cpu_features: CpuFeatureSet::NONE,
        }
    }

    pub fn cuda_skeleton() -> Self {
        Self {
            backend: DeviceKind::Cuda,
            cpu_features: CpuFeatureSet::NONE,
        }
    }

    /// Detects available CPU features on the current host.
    ///
    /// Feature detection happens at construction time (compile/lowering entry).
    /// It must not enter execution or kernel inner loops.
    ///
    /// On non-x86_64 platforms this returns `CpuFeatureSet::NONE`.
    pub fn cpu_detected() -> Self {
        let mut features = CpuFeatureSet::NONE;

        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                features = features.union(CpuFeatureSet::AVX2);
            }
            if std::is_x86_feature_detected!("fma") {
                features = features.union(CpuFeatureSet::FMA);
            }
        }

        Self {
            backend: DeviceKind::Cpu,
            cpu_features: features,
        }
    }

    pub fn supports(&self, query: CapabilityQuery) -> bool {
        match self.backend {
            DeviceKind::Cpu => supports_cpu_baseline(query),
            DeviceKind::Cuda | DeviceKind::Rocm => false,
        }
    }

    pub fn supports_cpu_features(&self, required: CpuFeatureSet) -> bool {
        self.backend == DeviceKind::Cpu && self.cpu_features.contains_all(required)
    }
}

fn supports_cpu_baseline(query: CapabilityQuery) -> bool {
    matches!(
        query.op,
        OpKind::Add | OpKind::Mul | OpKind::Clamp | OpKind::Silu
    ) && query.dtype == DType::F32
        && query.quant.is_none()
        && query.layout == PhysicalLayoutId::HOST_CONTIGUOUS_F32
        && query.epilogue == EpilogueKind::Identity
}
