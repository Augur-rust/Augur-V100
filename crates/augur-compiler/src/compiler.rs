use std::sync::Arc;

use augur_backend::{BackendCapabilities, CpuBackend};
use augur_core::DeviceKind;
use augur_exec::ExecutionPlan;
use augur_graph::{analyze_graph, Graph};
use augur_kernels::cpu_kernel_registrations;

use crate::{
    bind_cpu_plan, validate_cpu_tuning_attachment, CompilerError, CpuCompiledPlan, CpuLowerer,
    CpuTuningAttachError, CpuWeightResources, LoweringContext, LoweringError,
    ValidatedCpuTuningProfile,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompileTarget {
    Cpu,
    Cuda,
    Rocm,
}

impl CompileTarget {
    pub fn backend_kind(self) -> DeviceKind {
        match self {
            Self::Cpu => DeviceKind::Cpu,
            Self::Cuda => DeviceKind::Cuda,
            Self::Rocm => DeviceKind::Rocm,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptLevel {
    Debug,
    Baseline,
    Aggressive,
}

#[derive(Clone, Debug)]
pub struct GraphCompiler {
    pub target: CompileTarget,
    pub opt_level: OptLevel,
    pub(crate) capabilities: BackendCapabilities,

    /// Optional validated compiler-only CPU tuning profile.
    ///
    /// This owner never enters CpuCompiledPlan or ExecutionPlan.
    pub(crate) cpu_tuning: Option<Arc<ValidatedCpuTuningProfile>>,
}

impl GraphCompiler {
    pub fn cpu_baseline() -> Self {
        Self {
            target: CompileTarget::Cpu,
            opt_level: OptLevel::Baseline,
            capabilities: BackendCapabilities::cpu_baseline(),
            cpu_tuning: None,
        }
    }

    /// Creates a compiler that uses the detected host CPU features.
    ///
    /// Feature detection happens once at construction time. The returned
    /// compiler selects AVX2/FMA kernels when available and falls back to
    /// portable when those features are absent.
    ///
    /// On non-x86_64 platforms or when AVX2 is unavailable, this behaves
    /// identically to `cpu_baseline`.
    pub fn cpu_native() -> Self {
        Self {
            target: CompileTarget::Cpu,
            opt_level: OptLevel::Aggressive,
            capabilities: BackendCapabilities::cpu_detected(),
            cpu_tuning: None,
        }
    }

    /// Attaches an already-validated single-threaded CPU tuning profile.
    ///
    /// This method performs no file I/O. The caller owns profile parsing and
    /// initial validation.
    ///
    /// Attachment revalidates the current CPU environment and production
    /// registry fingerprint.
    pub fn with_cpu_tuning_profile(
        mut self,
        profile: Arc<ValidatedCpuTuningProfile>,
    ) -> Result<Self, CpuTuningAttachError> {
        validate_cpu_tuning_attachment(
            profile.as_ref(),
            cpu_kernel_registrations(),
            &self.capabilities,
            1,
        )?;

        self.cpu_tuning = Some(profile);

        Ok(self)
    }

    pub fn without_cpu_tuning_profile(mut self) -> Self {
        self.cpu_tuning = None;
        self
    }

    /// Compiles all logical, physical and kernel-selection decisions without
    /// requiring concrete external weight owners.
    pub fn compile_cpu_plan(
        &self,
        graph: &Graph,
        input_buffers: Vec<Vec<f32>>,
    ) -> Result<CpuCompiledPlan, CompilerError> {
        if self.target != CompileTarget::Cpu {
            return Err(CompilerError::Lowering(LoweringError::Core(
                augur_core::CompileError::BackendUnsupported,
            )));
        }

        let analyzed = analyze_graph(graph).map_err(CompilerError::Graph)?;

        let context = LoweringContext::new(
            analyzed,
            self.capabilities.clone(),
            self.cpu_tuning.clone(),
            input_buffers,
        )
        .map_err(CompilerError::lowering_core)?;

        CpuLowerer::new(context)
            .lower()
            .map_err(CompilerError::Lowering)
    }

    /// Convenience compile-and-bind entry for plans without external weight
    /// requirements, such as Add-only graphs.
    pub fn compile_cpu(
        &self,
        graph: &Graph,
        input_buffers: Vec<Vec<f32>>,
    ) -> Result<ExecutionPlan<CpuBackend>, CompilerError> {
        self.compile_cpu_with_weights(graph, input_buffers, CpuWeightResources::new())
    }

    /// Compiles and binds an independently executable CPU plan.
    ///
    /// The resource store is consumed by value. The returned ExecutionPlan
    /// owns all bound HotWeight resources through concrete Arc owners.
    pub fn compile_cpu_with_weights(
        &self,
        graph: &Graph,
        input_buffers: Vec<Vec<f32>>,
        resources: CpuWeightResources,
    ) -> Result<ExecutionPlan<CpuBackend>, CompilerError> {
        let compiled = self.compile_cpu_plan(graph, input_buffers)?;
        bind_cpu_plan(compiled, resources).map_err(CompilerError::Binding)
    }
}
