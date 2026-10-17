//! Typed, shared FFN execution for authenticated raw checkpoints. QPN keeps
//! original NVFP4 codes/scales, but stages FP16 operands, so it is explicitly a
//! different numerical profile from the reference's original-F32 matvecs.
use crate::checkpoint_projection_cuda::CheckpointProjectionPolicy;
use augur_backend::{OwnedCudaBuffer, OwnedCudaStream};
use augur_model_qwen35::CheckpointArtifact;
use std::{error::Error, rc::Rc};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[cfg(feature = "qwen35-prefill-cublas")]
pub(crate) use batch::{
    CheckpointFfnBatchProgram, CheckpointFfnBatchSource, CheckpointFfnBatchWorkspace,
};

#[cfg(feature = "qwen35-prefill-cublas")]
mod batch {
    use super::*;
    use crate::checkpoint_linear_cuda::empty;
    use crate::checkpoint_projection_cuda::{
        CheckpointExactBatchProjection, CheckpointScalarCudaProjection,
    };
    use augur_backend::cuda_qwen35_prefill_cublas::{
        CudaQwen35PrefillCublasPlanV1, CudaQwen35QpnFfnProgramV1, CudaQwen35QpnFfnWorkspaceV1,
    };
    use augur_backend::{CudaF32View, CudaSiluMulF32ArgsV1};
    use augur_core::PhysicalLayoutId;
    type Buffer = Rc<OwnedCudaBuffer>;
    #[derive(Clone, Copy)]
    pub(crate) enum CheckpointFfnBatchSource<'a> {
        Qpn(&'a CheckpointFfnProgram),
        Exact([&'a CheckpointScalarCudaProjection; 3]),
    }
    impl CheckpointFfnBatchSource<'_> {
        pub(crate) fn tail_operations(&self) -> usize {
            match self {
                Self::Qpn(_) => 2,
                Self::Exact(_) => 5,
            }
        }
        pub(crate) fn uses_partial_workspace(&self) -> bool {
            matches!(self, Self::Qpn(p) if p.retained().uses_partial_workspace())
        }
        pub(crate) fn is_residual(&self) -> bool {
            matches!(self, Self::Qpn(p) if p.retained().uses_residual_f32())
        }
        pub(crate) fn is_qpn(&self) -> bool {
            matches!(self, Self::Qpn(_))
        }
    }
    pub(crate) struct CheckpointFfnBatchWorkspace {
        stream: Rc<OwnedCudaStream>,
        qpn: Option<(
            Rc<CudaQwen35PrefillCublasPlanV1>,
            CudaQwen35QpnFfnWorkspaceV1,
        )>,
        exact: Option<[Buffer; 3]>,
    }
    pub(crate) enum CheckpointFfnBatchProgram {
        Qpn(CudaQwen35QpnFfnProgramV1),
        Exact {
            stream: Rc<OwnedCudaStream>,
            projections: [CheckpointExactBatchProjection; 3],
            silu: CudaSiluMulF32ArgsV1,
        },
    }
    fn prefix(buffer: &OwnedCudaBuffer, n: usize) -> Result<CudaF32View> {
        Ok(CudaF32View::new(
            buffer.view(0, n * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            n,
        )?)
    }
    impl CheckpointFfnBatchWorkspace {
        pub(crate) fn new(
            stream: Rc<OwnedCudaStream>,
            qpn: bool,
            exact: bool,
            residual: bool,
            partial_workspace: bool,
        ) -> Result<Self> {
            let qpn = if qpn {
                Some((
                    Rc::new(CudaQwen35PrefillCublasPlanV1::new(
                        &stream,
                        if partial_workspace {
                            17408 * 8 * 16 * 8
                        } else {
                            1
                        },
                        if residual { 34816 } else { 17408 },
                        8,
                    )?),
                    CudaQwen35QpnFfnWorkspaceV1::new(Rc::clone(&stream), 17408, 8)?,
                ))
            } else {
                None
            };
            let exact = if exact {
                Some([
                    empty(&stream, 8 * 17408)?,
                    empty(&stream, 8 * 17408)?,
                    empty(&stream, 8 * 17408)?,
                ])
            } else {
                None
            };
            Ok(Self { stream, qpn, exact })
        }
        pub(crate) fn bind(
            &self,
            source: CheckpointFfnBatchSource<'_>,
            input: Buffer,
            output: Buffer,
            rows: usize,
        ) -> Result<CheckpointFfnBatchProgram> {
            if !(1..=8).contains(&rows) {
                return Err("FFN batch rows out of bounds".into());
            }
            match source {
                CheckpointFfnBatchSource::Qpn(p) => {
                    let (plan, workspace) = self.qpn.as_ref().ok_or("missing QPN arena")?;
                    Ok(CheckpointFfnBatchProgram::Qpn(
                        p.retained().rebind_shared_weights(
                            Rc::clone(plan),
                            Rc::clone(&self.stream),
                            workspace,
                            input,
                            0,
                            output,
                            0,
                            rows,
                        )?,
                    ))
                }
                CheckpointFfnBatchSource::Exact([gate, up, down]) => {
                    let [g, u, s] = self.exact.as_ref().ok_or("missing exact FFN arena")?;
                    let projections = [
                        gate.rebind_exact_batch(
                            &self.stream,
                            Rc::clone(&input),
                            Rc::clone(g),
                            rows,
                        )?,
                        up.rebind_exact_batch(&self.stream, input, Rc::clone(u), rows)?,
                        down.rebind_exact_batch(&self.stream, Rc::clone(s), output, rows)?,
                    ];
                    let silu = CudaSiluMulF32ArgsV1::bind(
                        &self.stream,
                        prefix(g, rows * 17408)?,
                        prefix(u, rows * 17408)?,
                        prefix(s, rows * 17408)?,
                    )?;
                    Ok(CheckpointFfnBatchProgram::Exact {
                        stream: Rc::clone(&self.stream),
                        projections,
                        silu,
                    })
                }
            }
        }
    }
    impl CheckpointFfnBatchProgram {
        pub(crate) fn submit_for_plan_on(&self, stream: &OwnedCudaStream) -> Result<()> {
            match self {
                Self::Qpn(p) => p.submit_for_plan_on(stream)?,
                Self::Exact {
                    stream: owner,
                    projections,
                    silu,
                } => {
                    if !std::ptr::eq(owner.as_ref(), stream) {
                        return Err("exact FFN batch stream owner drift".into());
                    }
                    projections[0].submit()?;
                    projections[1].submit()?;
                    silu.submit_for_plan(stream)?;
                    projections[2].submit()?;
                }
            }
            Ok(())
        }
    }
}

pub(crate) struct CheckpointFfnContext {
    policy: CheckpointProjectionPolicy,
    #[cfg(feature = "qwen35-prefill-cublas")]
    inner: Option<enabled::Context>,
}
impl CheckpointFfnContext {
    pub fn new(stream: Rc<OwnedCudaStream>, policy: CheckpointProjectionPolicy) -> Result<Self> {
        #[cfg(not(feature = "qwen35-prefill-cublas"))]
        {
            let _ = stream;
            if policy.uses_qpn_ffn() {
                return Err("QPN FFN requires prefill-cublas feature".into());
            }
            Ok(Self { policy })
        }
        #[cfg(feature = "qwen35-prefill-cublas")]
        Ok(Self {
            policy,
            inner: if policy.uses_qpn_ffn() {
                Some(enabled::Context::new(
                    stream,
                    policy.qpn_residual(),
                    policy.qpn_workspace(),
                )?)
            } else {
                None
            },
        })
    }
    pub fn upload_projection(
        &self,
        stream: Rc<OwnedCudaStream>,
        source: augur_model_qwen35::LoadedCheckpointLinear,
    ) -> Result<Rc<crate::checkpoint_projection_cuda::CheckpointScalarCudaWeight>> {
        #[cfg(feature = "qwen35-prefill-cublas")]
        if self.policy.uses_qpn_projections() {
            let owner = self
                .inner
                .as_ref()
                .ok_or("missing QPN projection context")?;
            if !Rc::ptr_eq(&stream, &owner.stream) {
                return Err("QPN projection stream mismatch".into());
            }
            return crate::checkpoint_projection_cuda::CheckpointScalarCudaWeight::upload_with_shared_qpn(
                stream, source, self.policy, Rc::clone(&owner.plan));
        }
        crate::checkpoint_projection_cuda::CheckpointScalarCudaWeight::upload_with_policy(
            stream,
            source,
            self.policy,
        )
    }
    pub fn policy(&self) -> CheckpointProjectionPolicy {
        self.policy
    }
    pub fn bind(
        &self,
        reader: &mut CheckpointArtifact,
        layer: usize,
        stream: &Rc<OwnedCudaStream>,
        input: Rc<OwnedCudaBuffer>,
        output: Rc<OwnedCudaBuffer>,
    ) -> Result<CheckpointFfnProgram> {
        if !self.policy.qpn_ffn_layer(layer) {
            return Err("QPN FFN layer is outside explicit precision profile".into());
        }
        #[cfg(feature = "qwen35-prefill-cublas")]
        {
            let owner = self.inner.as_ref().ok_or("QPN FFN not selected")?;
            if !Rc::ptr_eq(stream, &owner.stream) {
                return Err("QPN FFN shared stream drift".into());
            }
            let (program, source_scales) = owner.bind(reader, layer, input, output, self.policy)?;
            Ok(CheckpointFfnProgram {
                program,
                _source_scales: source_scales,
            })
        }
        #[cfg(not(feature = "qwen35-prefill-cublas"))]
        {
            let _ = (reader, layer, stream, input, output);
            Err("QPN FFN feature absent".into())
        }
    }
}

pub(crate) struct CheckpointFfnProgram {
    #[cfg(feature = "qwen35-prefill-cublas")]
    program: augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnFfnProgramV1,
    #[cfg(feature = "qwen35-prefill-cublas")]
    _source_scales: [augur_model_qwen35::CheckpointNvfp4Scale; 3],
}
impl CheckpointFfnProgram {


    #[cfg(feature = "qwen35-prefill-cublas")]
    pub(crate) fn retained(
        &self,
    ) -> &augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnFfnProgramV1 {
        &self.program
    }
    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<()> {
        #[cfg(feature = "qwen35-prefill-cublas")]
        {
            self.program.submit_for_plan_on(stream)?;
            Ok(())
        }
        #[cfg(not(feature = "qwen35-prefill-cublas"))]
        {
            let _ = stream;
            Err("QPN FFN feature absent".into())
        }
    }
}

#[cfg(feature = "qwen35-prefill-cublas")]
mod enabled {
    use super::*;
    use augur_backend::cuda_qwen35_prefill_cublas::{
        CudaQwen35PrefillCublasPlanV1, CudaQwen35QpnFfnProgramV1, CudaQwen35QpnFfnWorkspaceV1,
    };
    use augur_backend::cuda_qwen35_qpn::OwnedCudaQwen35QpnWeightV1;
    use augur_model_qwen35::{CheckpointNvfp4Scale, LoadedCheckpointLinear};
    pub(super) struct Context {
        pub stream: Rc<OwnedCudaStream>,
        pub(super) plan: Rc<CudaQwen35PrefillCublasPlanV1>,
        workspace: CudaQwen35QpnFfnWorkspaceV1,
    }
    impl Context {
        pub fn new(
            stream: Rc<OwnedCudaStream>,
            residual: bool,
            partial_workspace: bool,
        ) -> Result<Self> {
            // One scalar plan for the entire target, not64 arenas. Profile6
            // reserves4.25MiB for reusable F64 partials; other profiles keep
            // the tiny weight scratch. No small-batch weight expansion.
            let plan = Rc::new(CudaQwen35PrefillCublasPlanV1::new(
                &stream,
                if partial_workspace { 17408 * 16 * 8 } else { 1 },
                if residual { 34816 } else { 17408 },
                1,
            )?);
            let workspace = CudaQwen35QpnFfnWorkspaceV1::new(Rc::clone(&stream), 17408, 1)?;
            Ok(Self {
                stream,
                plan,
                workspace,
            })
        }
        pub fn bind(
            &self,
            reader: &mut CheckpointArtifact,
            layer: usize,
            input: Rc<OwnedCudaBuffer>,
            output: Rc<OwnedCudaBuffer>,
            policy: CheckpointProjectionPolicy,
        ) -> Result<(CudaQwen35QpnFfnProgramV1, [CheckpointNvfp4Scale; 3])> {
            if layer >= 64 {
                return Err("invalid checkpoint FFN layer".into());
            }
            let mut weights = Vec::with_capacity(3);
            let mut source_scales = Vec::with_capacity(3);
            for (suffix, rows, cols) in [
                ("gate_proj", 17408, 5120),
                ("up_proj", 17408, 5120),
                ("down_proj", 5120, 17408),
            ] {
                let source = reader
                    .load_linear(&format!("model.language_model.layers.{layer}.mlp.{suffix}"))?;
                let LoadedCheckpointLinear::Nvfp4 {
                    spec,
                    packed,
                    scales,
                    global,
                } = source
                else {
                    return Err(
                        "QPN checkpoint FFN requires original NVFP4, not a requantized fallback"
                            .into(),
                    );
                };
                if spec.rows != rows || spec.cols != cols {
                    return Err("QPN checkpoint FFN geometry drift".into());
                }
                weights.push(Rc::new(crate::checkpoint_projection_cuda::startup_qpn::upload(
                    &self.stream, &packed, &scales, rows, cols, global,
                )?));
                source_scales.push(global);
            }
            let weights: [_; 3] = weights.try_into().map_err(|_| "FFN weight count")?;
            let scales = source_scales.try_into().map_err(|_| "FFN scale count")?;
            // The program retains plan, weights, stream, input, output and
            // workspace buffers. The context can be dropped after all binds.
            let mut program = CudaQwen35QpnFfnProgramV1::bind(
                Rc::clone(&self.plan),
                Rc::clone(&self.stream),
                weights,
                &self.workspace,
                input,
                0,
                output,
                0,
                1,
            )?;
            if policy.qpn_shape_split() {
                program = program.with_scale_placement(
                    augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnScalePlacement::ResidualWorkspaceShapeF64)?;
            } else if policy == CheckpointProjectionPolicy::AugurQpnAllWorkspaceCtaFoldF64 {
                program = program.with_scale_placement(
                    augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnScalePlacement::ResidualWorkspaceCtaFoldF64)?;
            } else if policy == CheckpointProjectionPolicy::AugurQpnAllWorkspaceF64 {
                program = program.with_scale_placement(
                    augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnScalePlacement::ResidualWorkspaceF64)?;
            } else if policy == CheckpointProjectionPolicy::AugurQpnAllRefinedFastF64 {
                program = program.with_scale_placement(
                    augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnScalePlacement::ResidualHiChunk16F64)?;
            } else if policy == CheckpointProjectionPolicy::AugurQpnAllRefinedF64 {
                program = program.with_scale_placement(
                    augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnScalePlacement::ResidualChunkedF64)?;
            } else if policy.qpn_residual() {
                program = program.with_scale_placement(
                    augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnScalePlacement::ResidualF32AfterAccumulation)?;
            } else if policy.qpn_postscale() {
                program = program.with_scale_placement(
                    augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnScalePlacement::AfterF32Accumulation)?;
            }
            Ok((program, scales))
        }
    }
}
