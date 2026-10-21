//! QPN original-input profile using the existing cuBLAS plan's scratch.
use super::*;
use crate::cuda_qwen35_qpn::OwnedCudaQwen35QpnWeightV1;
use augur_cuda_sys::AugurCudaQwen35QpnPrefillArgsV1;
use std::rc::Rc;
mod norm;
pub use norm::CudaQwen35NormQpnGroupV1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum CudaQwen35QpnScalePlacement {
    FoldedIntoF16Weight = 1,
    AfterF32Accumulation = 2,
    /// B1..B8 only: hi+scaled residual FP16 activations, two F32 MMA
    /// contractions, F64 partial merge and global scale. Not bit-exact F32.
    ResidualF32AfterAccumulation = 3,
    /// Same hi/lo input, four-group F32 MMA chunks merged in F64.
    ResidualChunkedF64 = 4,
    /// Hi MMA chains of16 groups, residual over the whole slice; F64 merge.
    ResidualHiChunk16F64 = 5,
    /// Parallel hi/lo CTAs, four-group F32 chains, plan-owned F64 VRAM partials.
    ResidualWorkspaceF64 = 6,
    /// Four-group unroll and four-warp per-component F64 CTA reduction.
    ResidualWorkspaceCtaFoldF64 = 7,
    /// Same CTA fold, but immutable shape-fixed SplitK8/16/32.
    ResidualWorkspaceShapeF64 = 8,
    /// B1 only: packed hi/lo MMA rows, depth-4 code prefetch, whole-slice F32
    /// accumulation widened once. No global partials. Changes the reduction
    /// tree, so it is a distinct profile rather than a retune of 7/8.
    ResidualPackedF32Prefetch = 9,
}

fn qpn_shape_split_k(rows: u32, cols: u32) -> u32 {
    match (rows, cols) {
        (1024, 5120) | (5120, 6144) => 32,
        (17408, 5120) | (5120, 17408) | (10240, 5120) | (6144, 5120) | (12288, 5120) => 16,
        _ => 8,
    }
}
fn set_qpn_placement(raw: &mut AugurCudaQwen35QpnPrefillArgsV1, placement: CudaQwen35QpnScalePlacement) {
    if placement == CudaQwen35QpnScalePlacement::ResidualWorkspaceShapeF64 {
        raw.split_k = qpn_shape_split_k(raw.rows, raw.cols);
    } else if raw.numerical_profile == 8 {
        raw.split_k = if raw.rows == 17408 && raw.cols == 5120 {16} else {8};
    } else if placement == CudaQwen35QpnScalePlacement::ResidualPackedF32Prefetch {
        raw.split_k = 8;
    }
    raw.numerical_profile = placement as u32;
}

/// A persistent execution binding without a self-referential Rust lifetime.
/// Keeps the exact plan, stream and every allocation alive. Rc is intentional:
/// this is the single-owner serial lane, not a concurrent scratch scheduler.
/// Weights are shared, never copied or kept in two physical representations.
#[derive(Clone)]
pub struct CudaQwen35QpnRetainedProjectionV1 {
    owners: Rc<QpnProjectionOwners>,
    raw: AugurCudaQwen35QpnPrefillArgsV1,
}
struct QpnProjectionOwners {
    // Drop the native plan before releasing the last stream reference.
    plan: Rc<CudaQwen35PrefillCublasPlanV1>,
    _weight: Rc<OwnedCudaQwen35QpnWeightV1>,
    _input: Rc<OwnedCudaBuffer>,
    _output: Rc<OwnedCudaBuffer>,
    stream: Rc<OwnedCudaStream>,
}
impl Drop for QpnProjectionOwners {
    fn drop(&mut self) {
        // Owner teardown only, not a submit-time synchronization. Explicit
        // synchronize remains the fallible completion API for result handling.
        let _ = self.stream.synchronize();
    }
}
impl CudaQwen35QpnRetainedProjectionV1 {
    /// Share only the original compressed weight, not the scalar arithmetic.
    pub fn original_weight(&self) -> Rc<OwnedCudaQwen35QpnWeightV1> {
        Rc::clone(&self.owners._weight)
    }

    /// Create an independently validated descriptor for a different numerical
    /// profile. Existing clones/graphs retain their old descriptor unchanged.
    /// Target owners select this only at construction, never mid-session.
    pub fn with_scale_placement(
        mut self,
        placement: CudaQwen35QpnScalePlacement,
    ) -> Result<Self, CudaRuntimeError> {
        set_qpn_placement(&mut self.raw, placement);
        let o = &self.owners;
        o.plan.require_stream(&o.stream)?;
        // SAFETY: the immutable owners retain every original descriptor range;
        // native preflight independently validates the new numerical profile.
        unsafe {
            CudaRuntimeError::from_result(
                OPERATION,
                augur_cuda_sys::augur_cuda_qwen35_qpn_prefill_preflight_v1(
                    o.stream.device_ordinal(),
                    o.stream.raw()?,
                    o.plan.raw,
                    &self.raw,
                ),
            )?;
        }
        Ok(self)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        plan: Rc<CudaQwen35PrefillCublasPlanV1>,
        stream: Rc<OwnedCudaStream>,
        weight: Rc<OwnedCudaQwen35QpnWeightV1>,
        input: Rc<OwnedCudaBuffer>,
        input_offset: usize,
        output: Rc<OwnedCudaBuffer>,
        output_offset: usize,
        tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let raw = plan
            .bind_qpn_projection(
                &stream,
                &weight,
                &input,
                input_offset,
                &output,
                output_offset,
                tokens,
            )?
            .raw;
        Ok(Self {
            owners: Rc::new(QpnProjectionOwners {
                plan,
                stream,
                _weight: weight,
                _input: input,
                _output: output,
            }),
            raw,
        })
    }
    /// Construction-time rebinding for a bounded prefill arena. Retains the
    /// same physical weight, numerical profile and Split-K; never uploads or
    /// expands weights and never silently falls back to the wide cuBLAS path.
    /// The new plan may own separate scratch, but both bindings must stay on
    /// the identical serial stream. This is not a concurrent scheduler.
    #[allow(clippy::too_many_arguments)]
    pub fn rebind_shared_weight(
        &self,
        plan: Rc<CudaQwen35PrefillCublasPlanV1>,
        stream: Rc<OwnedCudaStream>,
        input: Rc<OwnedCudaBuffer>,
        input_offset: usize,
        output: Rc<OwnedCudaBuffer>,
        output_offset: usize,
        tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if !Rc::ptr_eq(&stream, &self.owners.stream) {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        if self.raw.numerical_profile >= 3 && tokens > 8 {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let mut rebound = Self::bind(
            plan, stream, Rc::clone(&self.owners._weight), input, input_offset,
            output, output_offset, tokens,
        )?;
        // Preserve the descriptor rather than infer a policy from dimensions.
        // Bind checked all new ranges; repeat native preflight with the exact
        // original mathematics and the new plan's scratch before publication.
        rebound.raw.numerical_profile = self.raw.numerical_profile;
        rebound.raw.split_k = self.raw.split_k;
        let o = &rebound.owners;
        unsafe {
            CudaRuntimeError::from_result(
                OPERATION,
                augur_cuda_sys::augur_cuda_qwen35_qpn_prefill_preflight_v1(
                    o.stream.device_ordinal(), o.stream.raw()?, o.plan.raw,
                    &rebound.raw,
                ),
            )?;
        }
        Ok(rebound)
    }
    pub fn submit_for_plan(&self) -> Result<(), CudaRuntimeError> {
        let o = &self.owners;
        o.plan.require_stream(&o.stream)?;
        // SAFETY: immutable preflight descriptor; owners retain exact plan,
        // stream, weight and input/output through asynchronous completion.
        unsafe {
            CudaRuntimeError::from_result(
                OPERATION,
                augur_cuda_sys::augur_cuda_qwen35_qpn_prefill_prevalidated_submit_v1(
                    o.stream.device_ordinal(),
                    o.stream.raw()?,
                    o.plan.raw,
                    &self.raw,
                ),
            )
        }
    }
    /// One fresh activation stage per pair, for scalar, suffix and wide prefill.
    /// Both outputs must be disjoint from every pair input/weight/output. No
    /// cached-input state survives this call, even when addresses are unchanged.
    pub fn submit_shared_input_pair(&self, other: &Self) -> Result<(), CudaRuntimeError> {
        let o = &self.owners;
        if !Rc::ptr_eq(&o.plan, &other.owners.plan) || !Rc::ptr_eq(&o.stream, &other.owners.stream)
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        o.plan.require_stream(&o.stream)?;
        // SAFETY: individual descriptors retain all owners; native checks the
        // shared input and cross-projection aliases before any enqueue.
        unsafe {
            CudaRuntimeError::from_result(
                OPERATION,
                augur_cuda_sys::augur_cuda_qwen35_qpn_prefill_pair_prevalidated_submit_v1(
                    o.stream.device_ordinal(),
                    o.stream.raw()?,
                    o.plan.raw,
                    &self.raw,
                    &other.raw,
                ),
            )
        }
    }
}

/// Reusable full FFN body. The same profile covers B1..B8 decode/verify/replay
/// and wide prefill; only reduction scheduling changes at the cuBLAS boundary.
/// These three intermediates are shared across layers and widths by the runtime.
pub struct CudaQwen35QpnFfnWorkspaceV1 {
    gate: Rc<OwnedCudaBuffer>,
    up: Rc<OwnedCudaBuffer>,
    activation: Rc<OwnedCudaBuffer>,
    intermediate: usize,
    max_tokens: usize,
    stream: Rc<OwnedCudaStream>,
}
impl CudaQwen35QpnFfnWorkspaceV1 {
    pub fn new(
        stream: Rc<OwnedCudaStream>,
        intermediate: usize,
        max_tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if intermediate == 0
            || intermediate % 64 != 0
            || max_tokens == 0
            || max_tokens > CudaQwen35PrefillCublasPlanV1::compiled_max_tokens()
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let bytes = intermediate
            .checked_mul(max_tokens)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let ordinal = stream.device_ordinal();
        Ok(Self {
            gate: Rc::new(OwnedCudaBuffer::allocate(ordinal, bytes)?),
            up: Rc::new(OwnedCudaBuffer::allocate(ordinal, bytes)?),
            activation: Rc::new(OwnedCudaBuffer::allocate(ordinal, bytes)?),
            intermediate,
            max_tokens,
            stream,
        })
    }
    pub fn len_bytes(&self) -> usize {
        self.gate.len_bytes() + self.up.len_bytes() + self.activation.len_bytes()
    }
    /// Lease the caller's preallocated FFN intermediates; no duplicate arena.
    pub fn from_shared_buffers(
        stream: Rc<OwnedCudaStream>,
        intermediate: usize,
        max_tokens: usize,
        gate: Rc<OwnedCudaBuffer>,
        up: Rc<OwnedCudaBuffer>,
        activation: Rc<OwnedCudaBuffer>,
    ) -> Result<Self, CudaRuntimeError> {
        let bytes = intermediate
            .checked_mul(max_tokens)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        if intermediate == 0
            || max_tokens == 0
            || max_tokens > CudaQwen35PrefillCublasPlanV1::compiled_max_tokens()
            || [&gate, &up, &activation]
                .iter()
                .any(|b| b.len_bytes() != bytes)
            || Rc::ptr_eq(&gate, &up)
            || Rc::ptr_eq(&gate, &activation)
            || Rc::ptr_eq(&up, &activation)
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        Ok(Self {
            gate,
            up,
            activation,
            intermediate,
            max_tokens,
            stream,
        })
    }
}
#[derive(Clone)]
pub struct CudaQwen35QpnFfnProgramV1 {
    gate: CudaQwen35QpnRetainedProjectionV1,
    up: CudaQwen35QpnRetainedProjectionV1,
    down: CudaQwen35QpnRetainedProjectionV1,
    silu: crate::CudaSiluMulF32ArgsV1,
}
impl CudaQwen35QpnFfnProgramV1 {
    /// Share original QPN allocations without changing any execution binding.
    pub fn original_weights(&self)->[Rc<OwnedCudaQwen35QpnWeightV1>;3]{
        [Rc::clone(&self.gate.owners._weight),Rc::clone(&self.up.owners._weight),Rc::clone(&self.down.owners._weight)]
    }

    pub fn with_scale_placement(
        mut self,
        placement: CudaQwen35QpnScalePlacement,
    ) -> Result<Self, CudaRuntimeError> {
        self.gate = self.gate.with_scale_placement(placement)?;
        self.up = self.up.with_scale_placement(placement)?;
        self.down = self.down.with_scale_placement(placement)?;
        Ok(self)
    }
    /// Immutable original-F32-input FFN width, validated across all three
    /// projections at bind time. This is QPN math, not llama's quantized path.
    pub fn uses_partial_workspace(&self) -> bool {
        matches!(self.gate.raw.numerical_profile, 6 | 7 | 8 | 9)
    }
    pub fn uses_residual_f32(&self) -> bool {
        self.gate.raw.numerical_profile >= 3
    }
    pub fn original_input_rows(&self) -> usize {
        self.gate.raw.tokens as usize
    }


    /// Bind a different scalar/suffix/prefill arena to the SAME physical
    /// weights and numerical profile. This is construction-time work, not a
    /// hot-path rebind; the old program/graphs remain immutable and usable.
    /// Restrict to the same serial stream because shared plan scratch is not
    /// a concurrent scheduler. No weight upload or CUDA allocation occurs.
    #[allow(clippy::too_many_arguments)]
    pub fn rebind_shared_weights(
        &self,
        plan: Rc<CudaQwen35PrefillCublasPlanV1>,
        stream: Rc<OwnedCudaStream>,
        workspace: &CudaQwen35QpnFfnWorkspaceV1,
        input: Rc<OwnedCudaBuffer>,
        input_offset: usize,
        output: Rc<OwnedCudaBuffer>,
        output_offset: usize,
        tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if !Rc::ptr_eq(&stream, &self.gate.owners.stream)
            || self.up.raw.numerical_profile != self.gate.raw.numerical_profile
            || self.down.raw.numerical_profile != self.gate.raw.numerical_profile
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        let placement = match self.gate.raw.numerical_profile {
            1 => CudaQwen35QpnScalePlacement::FoldedIntoF16Weight,
            2 => CudaQwen35QpnScalePlacement::AfterF32Accumulation,
            3 => CudaQwen35QpnScalePlacement::ResidualF32AfterAccumulation,
            4 => CudaQwen35QpnScalePlacement::ResidualChunkedF64,
            5 => CudaQwen35QpnScalePlacement::ResidualHiChunk16F64,
            6 => CudaQwen35QpnScalePlacement::ResidualWorkspaceF64,
            7 => CudaQwen35QpnScalePlacement::ResidualWorkspaceCtaFoldF64,
            8 => CudaQwen35QpnScalePlacement::ResidualWorkspaceShapeF64,
            _ => return Err(validation(CudaRuntimeErrorKind::InvalidMetadata)),
        };
        Self::bind(
            plan,
            stream,
            [
                Rc::clone(&self.gate.owners._weight),
                Rc::clone(&self.up.owners._weight),
                Rc::clone(&self.down.owners._weight),
            ],
            workspace,
            input,
            input_offset,
            output,
            output_offset,
            tokens,
        )?
        .with_scale_placement(placement)
    }

    /// Verify the caller's sequencing stream before dispatching the retained
    /// program. This does not retarget the binding to a different stream.
    pub fn submit_for_plan_on(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.gate.owners.plan.require_stream(stream)?;
        self.submit_for_plan()
    }
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        plan: Rc<CudaQwen35PrefillCublasPlanV1>,
        stream: Rc<OwnedCudaStream>,
        weights: [Rc<OwnedCudaQwen35QpnWeightV1>; 3],
        workspace: &CudaQwen35QpnFfnWorkspaceV1,
        input: Rc<OwnedCudaBuffer>,
        input_offset: usize,
        output: Rc<OwnedCudaBuffer>,
        output_offset: usize,
        tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let [gate, up, down] = weights;
        if !Rc::ptr_eq(&stream, &workspace.stream) {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        if tokens == 0
            || tokens > workspace.max_tokens
            || gate.rows() != workspace.intermediate
            || gate.rows() != up.rows()
            || gate.cols() != up.cols()
            || down.cols() != gate.rows()
            || down.rows() != gate.cols()
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let f32_view = |b: &OwnedCudaBuffer| {
            CudaF32View::new(
                b.view(0, tokens * workspace.intermediate * 4)?,
                augur_core::PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                tokens * workspace.intermediate,
            )
        };
        let silu = crate::CudaSiluMulF32ArgsV1::bind(
            &stream,
            f32_view(&workspace.gate)?,
            f32_view(&workspace.up)?,
            f32_view(&workspace.activation)?,
        )?;
        let bind = |weight, source, source_offset, destination, destination_offset| {
            CudaQwen35QpnRetainedProjectionV1::bind(
                Rc::clone(&plan),
                Rc::clone(&stream),
                weight,
                source,
                source_offset,
                destination,
                destination_offset,
                tokens,
            )
        };
        Ok(Self {
            gate: bind(
                gate,
                Rc::clone(&input),
                input_offset,
                Rc::clone(&workspace.gate),
                0,
            )?,
            up: bind(up, input, input_offset, Rc::clone(&workspace.up), 0)?,
            down: bind(
                down,
                Rc::clone(&workspace.activation),
                0,
                output,
                output_offset,
            )?,
            silu,
        })
    }
    pub fn submit_for_plan(&self) -> Result<(), CudaRuntimeError> {
        self.gate.submit_shared_input_pair(&self.up)?;
        self.submit_activation_and_down_on(&self.gate.owners.stream)
    }
    /// Fresh B1 norm+gate/up binding; the group retains all input owners.
    pub fn bind_post_norm_group(
        &self, norm: crate::CudaRmsnormF32ArgsV1,
        input: Rc<OwnedCudaBuffer>, weight: Rc<OwnedCudaBuffer>,
    ) -> Result<CudaQwen35NormQpnGroupV1, CudaRuntimeError> {
        CudaQwen35NormQpnGroupV1::bind(norm, input, weight,
            vec![self.gate.clone(), self.up.clone()])
    }
    /// Whole B1 FFN tail only; split activation/down APIs remain unchanged.
    pub fn submit_activation_and_down_on(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        let o=&self.down.owners;o.plan.require_stream(stream)?;
        if self.down.raw.numerical_profile==8 && self.down.raw.tokens==1
            && self.down.raw.rows==5120 && self.down.raw.cols==17408 {
            // SAFETY: immutable checked descriptors and all operand/plan owners
            // retained by this program; native checks complete cross-op aliases.
            // Same stream consumes the fresh stage before any other scratch user.
            unsafe { CudaRuntimeError::from_result(OPERATION,
                augur_cuda_sys::augur_cuda_qwen35_qpn_silu_down_prevalidated_submit_v1(
                    stream.device_ordinal(),stream.raw()?,o.plan.raw,self.silu.as_raw(),&self.down.raw)) }
        } else {
            self.silu.submit_for_plan(stream)?;
            self.down.submit_for_plan()
        }
    }
    pub fn submit_gate_up_on(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.gate.owners.plan.require_stream(stream)?;
        self.gate.submit_shared_input_pair(&self.up)
    }
    pub fn submit_activation_on(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.gate.owners.plan.require_stream(stream)?;
        self.silu.submit_for_plan(stream)
    }
    pub fn submit_down_on(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.gate.owners.plan.require_stream(stream)?;
        self.down.submit_for_plan()
    }
    /// A captured graph must retain the program's owners through its last
    /// replay, not merely through capture. This wrapper makes that explicit.
    pub fn capture(&self) -> Result<CudaQwen35QpnFfnGraphV1, CudaRuntimeError> {
        let graph = crate::OwnedCudaGraphExec::capture(&self.gate.owners.stream, || {
            self.submit_for_plan()
        })?;
        Ok(CudaQwen35QpnFfnGraphV1 {
            graph,
            program: self.clone(),
        })
    }
}
pub struct CudaQwen35QpnFfnGraphV1 {
    // Destruction order: executable first, then retained buffers and stream.
    graph: crate::OwnedCudaGraphExec,
    program: CudaQwen35QpnFfnProgramV1,
}
impl CudaQwen35QpnFfnGraphV1 {
    pub fn node_count(&self) -> u64 {
        self.graph.node_count()
    }
    pub fn launch_for_plan(&self) -> Result<(), CudaRuntimeError> {
        self.graph.launch(&self.program.gate.owners.stream)
    }
}

pub struct CudaQwen35QpnPrefillProjectionV1<'a> {
    plan: &'a CudaQwen35PrefillCublasPlanV1,
    stream: &'a OwnedCudaStream,
    _weight: &'a OwnedCudaQwen35QpnWeightV1,
    _input: &'a OwnedCudaBuffer,
    _output: &'a OwnedCudaBuffer,
    raw: AugurCudaQwen35QpnPrefillArgsV1,
}
impl CudaQwen35PrefillCublasPlanV1 {
    /// Explicit profile: original F32 -> FP16, folded-FP16 weights and FP32
    /// accumulation. Width1..8 uses QPN; wider batches use the same represented
    /// weights in cuBLAS. No Q8 activation quantizer and no new allocation.
    pub fn bind_qpn_projection<'a>(
        &'a self,
        stream: &'a OwnedCudaStream,
        weight: &'a OwnedCudaQwen35QpnWeightV1,
        input: &'a OwnedCudaBuffer,
        input_offset: usize,
        output: &'a OwnedCudaBuffer,
        output_offset: usize,
        tokens: usize,
    ) -> Result<CudaQwen35QpnPrefillProjectionV1<'a>, CudaRuntimeError> {
        self.require_stream(stream)?;
        if tokens == 0 || tokens > self.max_tokens {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let extent = |cols: usize| {
            tokens
                .checked_mul(cols)
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))
        };
        let w = weight.physical_view()?;
        let x = input.view(input_offset, extent(weight.cols())?)?;
        let y = output.view(output_offset, extent(weight.rows())?)?;
        for v in [w, x, y] {
            if v.device_ordinal() != stream.device_ordinal() {
                return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
            }
            v.checked_range(OPERATION)?;
        }
        let mut raw = AugurCudaQwen35QpnPrefillArgsV1::new();
        raw.weight = w.checked_address(OPERATION)?;
        raw.weight_bytes = w.len_bytes() as u64;
        raw.input_f32 = x.checked_address(OPERATION)?;
        raw.input_bytes = x.len_bytes() as u64;
        raw.output_f32 = y.checked_address(OPERATION)?;
        raw.output_bytes = y.len_bytes() as u64;
        raw.rows = weight.rows() as u32;
        raw.cols = weight.cols() as u32;
        raw.tokens = tokens as u32;
        raw.split_k = if weight.rows() == 17408 && weight.cols() == 5120 {
            16
        } else {
            8
        };
        raw.tensor_scale = weight.tensor_scale();
        raw.numerical_profile = 1;
        // SAFETY: the binding borrows this plan, exact stream and every buffer.
        // Native preflight also checks its owned scratch, alignment and aliases.
        unsafe {
            CudaRuntimeError::from_result(
                OPERATION,
                augur_cuda_sys::augur_cuda_qwen35_qpn_prefill_preflight_v1(
                    stream.device_ordinal(),
                    stream.raw()?,
                    self.raw,
                    &raw,
                ),
            )?;
        }
        Ok(CudaQwen35QpnPrefillProjectionV1 {
            plan: self,
            stream,
            _weight: weight,
            _input: input,
            _output: output,
            raw,
        })
    }
}
impl CudaQwen35QpnPrefillProjectionV1<'_> {
    /// Select an explicit numerical profile without changing the borrowed
    /// owners or the representation. Validate before any submission.
    pub fn with_scale_placement(
        mut self,
        placement: CudaQwen35QpnScalePlacement,
    ) -> Result<Self, CudaRuntimeError> {
        set_qpn_placement(&mut self.raw, placement);
        self.plan.require_stream(self.stream)?;
        // SAFETY: this binding still borrows the exact preflighted ranges.
        unsafe {
            CudaRuntimeError::from_result(
                OPERATION,
                augur_cuda_sys::augur_cuda_qwen35_qpn_prefill_preflight_v1(
                    self.stream.device_ordinal(),
                    self.stream.raw()?,
                    self.plan.raw,
                    &self.raw,
                ),
            )?;
        }
        Ok(self)
    }
    pub fn submit_for_plan(&self) -> Result<(), CudaRuntimeError> {
        self.plan.require_stream(self.stream)?;
        // SAFETY: immutable preflight descriptor and borrowed owners; no stream
        // substitution is permitted. The owning execution plan drains the stream.
        unsafe {
            CudaRuntimeError::from_result(
                OPERATION,
                augur_cuda_sys::augur_cuda_qwen35_qpn_prefill_prevalidated_submit_v1(
                    self.stream.device_ordinal(),
                    self.stream.raw()?,
                    self.plan.raw,
                    &self.raw,
                ),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn floats(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
    fn difference(actual: &[u8], reference: &[u8]) -> serde_json::Value {
        assert_eq!(actual.len(), reference.len());
        let (mut d2, mut r2, mut max_abs) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (a, r) in floats(actual).into_iter().zip(floats(reference)) {
            assert!(a.is_finite() && r.is_finite());
            let d = f64::from(a) - f64::from(r);
            d2 += d * d;
            r2 += f64::from(r) * f64::from(r);
            max_abs = max_abs.max(d.abs());
        }
        serde_json::json!({"max_abs":max_abs,"relative_l2":(d2 / r2.max(1e-300)).sqrt()})
    }

    #[test]
    #[ignore = "exclusive SM70; AUGUR_QPN_FFN_FIXTURE binds actual layer weights and captured post-norm inputs"]
    fn retained_ffn_real_capture() {
        let root = std::path::PathBuf::from(std::env::var_os("AUGUR_QPN_FFN_FIXTURE").unwrap());
        let m: serde_json::Value = serde_json::from_slice(&std::fs::read(&root).unwrap()).unwrap();
        let placement = match m["scale_placement"].as_str() {
            None | Some("folded") => CudaQwen35QpnScalePlacement::FoldedIntoF16Weight,
            Some("postscale") => CudaQwen35QpnScalePlacement::AfterF32Accumulation,
            _ => panic!("invalid fixture scale placement"),
        };
        let read = |key: &str| {
            let v = &m[key];
            let bytes = std::fs::read(v["path"].as_str().unwrap()).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                v["sha256"].as_str().unwrap(),
                "{key}"
            );
            bytes
        };
        const H: usize = 5120;
        const I: usize = 17408;
        const B: usize = 128;
        let input_bytes = read("input");
        assert!(input_bytes.len() >= H * B * 4);
        let input_bytes = &input_bytes[..H * B * 4];
        assert!(floats(input_bytes).iter().all(|v| v.is_finite()));
        let before = crate::cuda_device_allocation_snapshot();
        let metrics;
        {
            let stream = Rc::new(OwnedCudaStream::create(0).unwrap());
            let plan = Rc::new(CudaQwen35PrefillCublasPlanV1::new(&stream, H * I, I, B).unwrap());
            let upload = |key: &str, rows, cols| {
                Rc::new(
                    OwnedCudaQwen35QpnWeightV1::upload_ggml(
                        &stream,
                        &read(key),
                        rows,
                        cols,
                        m[key]["scale"].as_f64().unwrap() as f32,
                    )
                    .unwrap(),
                )
            };
            let weights = [
                upload("gate", I, H),
                upload("up", I, H),
                upload("down", H, I),
            ];
            let weak_weights = weights.each_ref().map(Rc::downgrade);
            let workspace = CudaQwen35QpnFfnWorkspaceV1::new(Rc::clone(&stream), I, B).unwrap();
            let input = Rc::new(OwnedCudaBuffer::allocate(0, input_bytes.len()).unwrap());
            let output = Rc::new(OwnedCudaBuffer::allocate(0, H * B * 4).unwrap());
            input.copy_from_host_sync(&stream, input_bytes).unwrap();
            let bind = |tokens, offset| {
                CudaQwen35QpnFfnProgramV1::bind(
                    Rc::clone(&plan),
                    Rc::clone(&stream),
                    weights.clone(),
                    &workspace,
                    Rc::clone(&input),
                    offset * H * 4,
                    Rc::clone(&output),
                    offset * H * 4,
                    tokens,
                )
                .unwrap()
                .with_scale_placement(placement)
                .unwrap()
            };
            let first = bind(8, 0);
            assert!(
                first.gate.submit_shared_input_pair(&first.gate).is_err(),
                "cross-output alias"
            );
            let other_input = bind(8, 8);
            assert!(
                first
                    .gate
                    .submit_shared_input_pair(&other_input.up)
                    .is_err(),
                "different source range"
            );
            let foreign = Rc::new(OwnedCudaStream::create(0).unwrap());
            let foreign_plan =
                Rc::new(CudaQwen35PrefillCublasPlanV1::new(&foreign, H * I, I, B).unwrap());
            assert!(
                CudaQwen35QpnFfnProgramV1::bind(
                    foreign_plan,
                    foreign,
                    weights.clone(),
                    &workspace,
                    Rc::clone(&input),
                    0,
                    Rc::clone(&output),
                    0,
                    8
                )
                .is_err(),
                "foreign workspace stream"
            );

            // Independent sequential assembly uses the previously checked
            // borrowed API (including three separate input stages).
            let reference = |tokens, offset| {
                let xoff = offset * H * 4;
                plan.bind_qpn_projection(
                    &stream,
                    &weights[0],
                    &input,
                    xoff,
                    &workspace.gate,
                    0,
                    tokens,
                )
                .unwrap()
                .with_scale_placement(placement)
                .unwrap()
                .submit_for_plan()
                .unwrap();
                plan.bind_qpn_projection(
                    &stream,
                    &weights[1],
                    &input,
                    xoff,
                    &workspace.up,
                    0,
                    tokens,
                )
                .unwrap()
                .with_scale_placement(placement)
                .unwrap()
                .submit_for_plan()
                .unwrap();
                let f32v = |v: &OwnedCudaBuffer| {
                    CudaF32View::new(
                        v.view(0, tokens * I * 4).unwrap(),
                        augur_core::PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                        tokens * I,
                    )
                    .unwrap()
                };
                crate::CudaSiluMulF32ArgsV1::bind(
                    &stream,
                    f32v(&workspace.gate),
                    f32v(&workspace.up),
                    f32v(&workspace.activation),
                )
                .unwrap()
                .submit_for_plan(&stream)
                .unwrap();
                plan.bind_qpn_projection(
                    &stream,
                    &weights[2],
                    &workspace.activation,
                    0,
                    &output,
                    xoff,
                    tokens,
                )
                .unwrap()
                .with_scale_placement(placement)
                .unwrap()
                .submit_for_plan()
                .unwrap();
            };
            let read_output = || {
                let mut bytes = vec![0; H * B * 4];
                output.copy_to_host_sync(&stream, &mut bytes).unwrap();
                assert!(floats(&bytes).iter().all(|v| v.is_finite()));
                bytes
            };
            for offset in (0..B).step_by(8) {
                reference(8, offset);
            }
            let canonical = read_output();
            // Start from a genuinely scalar arena, as the checkpoint target
            // does. Rebind its retained weights into a separate wide plan;
            // this must neither upload nor duplicate the weight allocation.
            let scalar_plan =
                Rc::new(CudaQwen35PrefillCublasPlanV1::new(&stream, 1, I, 1).unwrap());
            let scalar_workspace =
                CudaQwen35QpnFfnWorkspaceV1::new(Rc::clone(&stream), I, 1).unwrap();
            let scalar_input = Rc::new(OwnedCudaBuffer::allocate(0, H * 4).unwrap());
            let scalar_output = Rc::new(OwnedCudaBuffer::allocate(0, H * 4).unwrap());
            let scalar = CudaQwen35QpnFfnProgramV1::bind(
                Rc::clone(&scalar_plan),
                Rc::clone(&stream),
                weights.clone(),
                &scalar_workspace,
                Rc::clone(&scalar_input),
                0,
                Rc::clone(&scalar_output),
                0,
                1,
            )
            .unwrap()
            .with_scale_placement(placement)
            .unwrap();
            let rebind = |tokens, offset| {
                let allocations = crate::cuda_device_allocation_snapshot().successful_allocations;
                let transfers = crate::cuda_host_boundary_snapshot();
                let program = scalar
                    .rebind_shared_weights(
                        Rc::clone(&plan),
                        Rc::clone(&stream),
                        &workspace,
                        Rc::clone(&input),
                        offset * H * 4,
                        Rc::clone(&output),
                        offset * H * 4,
                        tokens,
                    )
                    .unwrap();
                assert_eq!(program.gate.raw.numerical_profile, placement as u32);
                assert_eq!(program.original_input_rows(), tokens);
                for (projection, weight) in [&program.gate, &program.up, &program.down]
                    .iter()
                    .zip(&weights)
                {
                    assert!(
                        Rc::ptr_eq(&projection.owners._weight, weight),
                        "exact weight owner"
                    );
                }
                assert_eq!(
                    allocations,
                    crate::cuda_device_allocation_snapshot().successful_allocations
                );
                assert_eq!(
                    transfers,
                    crate::cuda_host_boundary_snapshot(),
                    "rebind is transfer-free"
                );
                program
            };
            assert!(
                scalar
                    .rebind_shared_weights(
                        Rc::clone(&plan),
                        Rc::new(OwnedCudaStream::create(0).unwrap()),
                        &workspace,
                        Rc::clone(&input),
                        0,
                        Rc::clone(&output),
                        0,
                        8,
                    )
                    .is_err(),
                "foreign serial owner"
            );
            for width in 1..=8 {
                // Bind/preflight outside execution, then exercise all tails.
                let programs: Vec<_> = (0..B)
                    .step_by(width)
                    .map(|row| rebind(width.min(B - row), row))
                    .collect();
                let allocations = crate::cuda_device_allocation_snapshot().successful_allocations;
                for program in &programs {
                    program.submit_for_plan().unwrap();
                }
                assert_eq!(
                    read_output(),
                    canonical,
                    "same-profile FFN B{width} versus B8"
                );
                assert_eq!(
                    crate::cuda_device_allocation_snapshot().successful_allocations,
                    allocations
                );
            }
            reference(B, 0);
            let wide_reference = read_output();
            let wide = rebind(B, 0);
            // The rebound plan must survive destruction of all original
            // scalar execution owners, not just a borrowed weight handle.
            drop(scalar);
            drop(scalar_plan);
            drop(scalar_workspace);
            drop(scalar_input);
            drop(scalar_output);
            let allocations = crate::cuda_device_allocation_snapshot().successful_allocations;
            wide.submit_for_plan().unwrap();
            let wide_actual = read_output();
            assert_eq!(
                wide_actual, wide_reference,
                "whole wide FFN pair reuse versus sequential assembly"
            );
            assert_eq!(
                crate::cuda_device_allocation_snapshot().successful_allocations,
                allocations
            );
            let llama = read("llama_output");
            metrics = serde_json::json!({"layer":m["layer"],"input_kind":"captured_llama_post_norm",
                "scale_placement":match placement { CudaQwen35QpnScalePlacement::FoldedIntoF16Weight=>"folded", CudaQwen35QpnScalePlacement::AfterF32Accumulation=>"postscale", CudaQwen35QpnScalePlacement::ResidualF32AfterAccumulation=>"residual-f32", CudaQwen35QpnScalePlacement::ResidualChunkedF64=>"residual-chunked-f64", CudaQwen35QpnScalePlacement::ResidualHiChunk16F64=>"residual-hi16-f64", CudaQwen35QpnScalePlacement::ResidualWorkspaceF64=>"residual-workspace-f64", CudaQwen35QpnScalePlacement::ResidualWorkspaceCtaFoldF64=>"residual-workspace-cta-fold-f64", CudaQwen35QpnScalePlacement::ResidualWorkspaceShapeF64=>"residual-workspace-shape-f64" },
                "sample_rows":B,"ffn_scratch_bytes":workspace.len_bytes(),
                "scalar_to_batched_weight_owner_shared":true,"rebind_cuda_allocations":0,
                "rebind_host_transfers":0,"original_scalar_arena_dropped":true,
                "b8_vs_llama":difference(&canonical,&llama[..H*B*4]),
                "b128_vs_llama":difference(&wide_actual,&llama[..H*B*4]),
                "b128_vs_b8":difference(&wide_actual,&canonical),
                "math_qualified":false,"product_ready":false,"visible_tps":false});

            let graph = first.capture().unwrap();
            assert_eq!(graph.node_count(), 6, "stage,gate,up,SiLU,stage,down");
            let mut reversed = input_bytes.to_vec();
            for row in 0..8 {
                reversed[row * H * 4..(row + 1) * H * 4]
                    .copy_from_slice(&input_bytes[(7 - row) * H * 4..(8 - row) * H * 4]);
            }
            input.copy_from_host_sync(&stream, &reversed).unwrap();
            for _ in 0..3 {
                graph.launch_for_plan().unwrap();
            }
            let refreshed = read_output();
            for row in 0..8 {
                assert_eq!(
                    &refreshed[row * H * 4..(row + 1) * H * 4],
                    &canonical[(7 - row) * H * 4..(8 - row) * H * 4]
                );
            }
            // These locals can leave scope while execution bindings survive.
            // No borrowed/self-reference trick, second weight copy or raw
            // owner-free descriptor is needed in a future target layer plan.
            drop(weights);
            drop(workspace);
            drop(plan);
            assert!(weak_weights.iter().all(|w| w.upgrade().is_some()));
            first.submit_for_plan().unwrap();
            assert_eq!(read_output()[..H * 8 * 4], refreshed[..H * 8 * 4]);
            drop(first);
            drop(other_input);
            drop(wide);
            assert!(
                weak_weights.iter().all(|w| w.upgrade().is_some()),
                "graph retains owners"
            );
            graph.launch_for_plan().unwrap();
            assert_eq!(read_output()[..H * 8 * 4], refreshed[..H * 8 * 4]);
            drop(graph);
            assert!(weak_weights.iter().all(|w| w.upgrade().is_none()));
            stream.synchronize().unwrap();
        }
        let after = crate::cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
        assert_eq!(
            after.successful_allocations - before.successful_allocations,
            after.successful_frees - before.successful_frees
        );
        eprintln!("QPN_FFN_METRICS={metrics}");
    }
}

#[cfg(test)]
mod refinement_tests {
    use super::*;

    #[test]
    #[ignore = "exclusive SM70; selected fused shapes vs unchanged profile7, widths1..8"]
    fn fused_b1_shapes_match_unfused_widths_and_guards() {
        let baseline=crate::cuda_device_allocation_snapshot();
        // Profile7 can retain the same explicit SplitK16 while dispatching the
        // unfused oracle. The production fusion is restricted to profile8/B1.
        for n in [17408_usize,12288,6144] {
            const K: usize=5120;
            const B: usize=8;
            let stream=Rc::new(OwnedCudaStream::create(0).unwrap());
            let plan=Rc::new(CudaQwen35PrefillCublasPlanV1::new(&stream,n*B*16*8,K*2,B).unwrap());
            let packed=(0..n*K/2).map(|i| ((i*13+i/19)&255) as u8).collect::<Vec<_>>();
            let scales=(0..n*K/16).map(|i| ((i*7+i/37)%127) as u8).collect::<Vec<_>>();
            let weight=Rc::new(OwnedCudaQwen35QpnWeightV1::upload_checkpoint_nvfp4(
                &stream,&packed,&scales,n,K,0.00015694754).unwrap());
            let input=Rc::new(OwnedCudaBuffer::allocate(0,K*B*4+32).unwrap());
            let output=Rc::new(OwnedCudaBuffer::allocate(0,n*B*4+32).unwrap());
            let mut values=(0..K*B).map(|i| ((i*29%503) as f32-251.)/73.0031).collect::<Vec<_>>();
            values[..10].copy_from_slice(&[32784.,-32784.,32816.,-32816.,65504.,-65504.,3e-8,-3e-8,0.,-0.]);
            let mut bytes=vec![0xa5;16];bytes.extend(values.iter().flat_map(|v|v.to_le_bytes()));bytes.extend([0xa5;16]);
            input.copy_from_host_sync(&stream,&bytes).unwrap();
            let guard=vec![0xa5;n*B*4+32];
            for width in 1..=B {
                let mut expected=None;
                for placement in [CudaQwen35QpnScalePlacement::ResidualWorkspaceCtaFoldF64,
                                  CudaQwen35QpnScalePlacement::ResidualWorkspaceShapeF64] {
                    let mut op=CudaQwen35QpnRetainedProjectionV1::bind(Rc::clone(&plan),Rc::clone(&stream),
                        Rc::clone(&weight),Rc::clone(&input),16,Rc::clone(&output),16,width).unwrap();
                    op.raw.split_k=16;
                    let op=op.with_scale_placement(placement).unwrap();
                    output.copy_from_host_sync(&stream,&guard).unwrap();
                    let before=crate::cuda_device_allocation_snapshot();
                    op.submit_for_plan().unwrap();stream.synchronize().unwrap();
                    let after=crate::cuda_device_allocation_snapshot();
                    assert_eq!(after.successful_allocations,before.successful_allocations);
                    assert_eq!(after.successful_frees,before.successful_frees);
                    let mut actual=vec![0;guard.len()];output.copy_to_host_sync(&stream,&mut actual).unwrap();
                    assert!(actual[..16].iter().chain(actual[16+n*width*4..].iter()).all(|&v|v==0xa5));
                    let values=&actual[16..16+n*width*4];
                    assert!(values.chunks_exact(4).all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()));
                    if let Some(expected)=&expected {assert_eq!(values,expected,"n={n} width={width}");}
                    else {expected=Some(values.to_vec());}
                }
            }
        }
        let after=crate::cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes,baseline.live_bytes);
        assert_eq!(after.live_allocations,baseline.live_allocations);
    }
    #[test]
    #[ignore = "exclusive SM70 device; compensated QPN native safety and numerical contract"]
    fn residual_profiles_widths_reference_rejection_and_cleanup() {
        let before = crate::cuda_device_allocation_snapshot();
        for (n, k) in [(32,5120), (32,576), (32,64), (1024,5120), (5120,6144)] {
            const B: usize = 8;
            let stream = Rc::new(OwnedCudaStream::create(0).unwrap());
            let plan = Rc::new(
                CudaQwen35PrefillCublasPlanV1::new(&stream, n * B * 16 * 8, k * 2, B).unwrap(),
            );
            let mut wire = vec![0u8; n * (k / 64) * 36];
            let mut dense = vec![0f64; n * k];
            let table = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];
            for row in 0..n {
                for col in 0..k {
                    let code = (row * 7 + col * 3 + col / 16) % 16;
                    let block = (row * (k / 64) + col / 64) * 36;
                    wire[block + col % 64 / 16] = 56; // E4M3 scale1, exact in half.
                    let byte = block + 4 + (col % 64 / 16) * 8 + col % 8;
                    wire[byte] |= (code as u8) << ((col % 16 / 8) * 4);
                    dense[row * k + col] = table[code & 7] * if code & 8 != 0 { -1. } else { 1. };
                }
            }
            let weight = Rc::new(
                OwnedCudaQwen35QpnWeightV1::upload_ggml(&stream, &wire, n, k, 0.75).unwrap(),
            );
            let input = Rc::new(OwnedCudaBuffer::allocate(0, k * B * 4).unwrap());
            let output = Rc::new(OwnedCudaBuffer::allocate(0, n * B * 4).unwrap());
            let mut x: Vec<f32> = (0..k * B)
                .map(|i| ((i * 29 % 503) as f32 - 251.) / 73.0031)
                .collect();
            x[..8].copy_from_slice(&[
                32784.,
                -32784.,
                32816.,
                -32816.,
                65504.,
                -65504.,
                0.00000003,
                -0.00000003,
            ]);
            let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            input.copy_from_host_sync(&stream, &bytes).unwrap();
            let read = |count| {
                let mut bytes = vec![0; count * 4];
                output.copy_to_host_sync(&stream, &mut bytes).unwrap();
                bytes
                    .chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>()
            };
            for split_k in [8, 16] {
                for placement in [
                    CudaQwen35QpnScalePlacement::ResidualF32AfterAccumulation,
                    CudaQwen35QpnScalePlacement::ResidualChunkedF64,
                    CudaQwen35QpnScalePlacement::ResidualHiChunk16F64,
                    CudaQwen35QpnScalePlacement::ResidualWorkspaceF64,
                    CudaQwen35QpnScalePlacement::ResidualWorkspaceCtaFoldF64,
                    CudaQwen35QpnScalePlacement::ResidualWorkspaceShapeF64,
                ] {
                    if n > 32 && (placement != CudaQwen35QpnScalePlacement::ResidualWorkspaceShapeF64 || split_k != 8) { continue; }
                    let programs: Vec<_> = (1..=B)
                        .map(|width| {
                            let mut program = CudaQwen35QpnRetainedProjectionV1::bind(
                                Rc::clone(&plan),
                                Rc::clone(&stream),
                                Rc::clone(&weight),
                                Rc::clone(&input),
                                0,
                                Rc::clone(&output),
                                0,
                                width,
                            )
                            .unwrap();
                            program.raw.split_k = split_k;
                            program.with_scale_placement(placement).unwrap()
                        })
                        .collect();
                    let alloc = crate::cuda_device_allocation_snapshot().successful_allocations;
                    programs[B - 1].submit_for_plan().unwrap();
                    stream.synchronize().unwrap();
                    let canonical = read(n * B);
                    let mut staged = vec![0u8; plan.decoded_input_f16.len_bytes()];
                    plan.decoded_input_f16
                        .copy_to_host_sync(&stream, &mut staged)
                        .unwrap();
                    let decode = |i: usize| {
                        let bits = u16::from_le_bytes(staged[i * 2..i * 2 + 2].try_into().unwrap());
                        let exponent = (bits >> 10) & 31;
                        let mantissa = bits & 1023;
                        assert_ne!(exponent, 31, "finite original input must have finite hi/lo");
                        let v = if exponent == 0 {
                            f64::from(mantissa) * 2f64.powi(-24)
                        } else {
                            (1. + f64::from(mantissa) / 1024.) * 2f64.powi(i32::from(exponent) - 15)
                        };
                        if bits & 0x8000 != 0 {
                            -v
                        } else {
                            v
                        }
                    };
                    for (i, &original) in x.iter().enumerate() {
                        let error =
                            (decode(i) + decode(k * B + i) / 4096. - f64::from(original)).abs();
                        assert!(
                            error
                                <= (f64::from(original).abs() * 2f64.powi(-22)).max(2f64.powi(-37))
                        );
                    }

                    let mut e2 = 0.;
                    let mut r2 = 0.;
                    let mut maxerr = 0f64;
                    let mut maxref = 0f64;
                    for token in 0..B {
                        for row in 0..n {
                            let expected = (0..k)
                                .map(|col| f64::from(x[token * k + col]) * dense[row * k + col])
                                .sum::<f64>()
                                * 0.75;
                            let error = f64::from(canonical[token * n + row]) - expected;
                            e2 += error * error;
                            r2 += expected * expected;
                            maxerr = maxerr.max(error.abs());
                            maxref = maxref.max(expected.abs());
                        }
                    }
                    eprintln!(
                        "QPN_REFINEMENT cols={} profile={} split_k={} relative_l2={} scaled_max={}",
                        k, placement as u32,
                        programs[0].raw.split_k,
                        (e2 / r2).sqrt(),
                        maxerr / (1. + maxref)
                    );
                    assert!((e2 / r2).sqrt() < 2e-6 && maxerr / (1. + maxref) < 2e-6);
                    for (i, program) in programs.iter().enumerate() {
                        program.submit_for_plan().unwrap();
                        stream.synchronize().unwrap();
                        // Buffer read uses the whole owner; compare only the actual width.
                        let actual = read(n * B);
                        assert_eq!(actual[..n * (i + 1)], canonical[..n * (i + 1)]);
                    }
                    assert_eq!(
                        crate::cuda_device_allocation_snapshot().successful_allocations,
                        alloc
                    );
                    let raw = programs[0].raw;
                    let reject = |raw: &AugurCudaQwen35QpnPrefillArgsV1| {
                        // SAFETY: retained owners keep all otherwise valid ranges alive;
                        // each deliberately malformed descriptor must reject before launch.
                        let status = unsafe {
                            augur_cuda_sys::augur_cuda_qwen35_qpn_prefill_preflight_v1(
                                0,
                                stream.raw().unwrap(),
                                plan.raw,
                                raw,
                            )
                        };
                        assert!(CudaRuntimeError::from_result(OPERATION, status).is_err());
                    };
                    let mut bad = raw;
                    bad.numerical_profile = 9;
                    reject(&bad);
                    if placement == CudaQwen35QpnScalePlacement::ResidualWorkspaceShapeF64 {
                        let mut bad = raw;
                        bad.split_k = if raw.split_k == 8 {16} else {8};
                        reject(&bad);
                    }
                    let mut bad = raw;
                    bad.input_bytes -= 4;
                    reject(&bad);
                    let mut bad = raw;
                    bad.output_f32 = bad.input_f32;
                    reject(&bad);
                    let mut bad = raw;
                    bad.tensor_scale = f32::NAN;
                    reject(&bad);
                    if matches!(placement, CudaQwen35QpnScalePlacement::ResidualWorkspaceF64 | CudaQwen35QpnScalePlacement::ResidualWorkspaceCtaFoldF64 | CudaQwen35QpnScalePlacement::ResidualWorkspaceShapeF64) {
                        let missing_partials = Rc::new(
                            CudaQwen35PrefillCublasPlanV1::new(&stream, 1, k * 2, 1).unwrap(),
                        );
                        assert!(CudaQwen35QpnRetainedProjectionV1::bind(
                            missing_partials,
                            Rc::clone(&stream),
                            Rc::clone(&weight),
                            Rc::clone(&input),
                            0,
                            Rc::clone(&output),
                            0,
                            1
                        )
                        .unwrap()
                        .with_scale_placement(placement)
                        .is_err());
                    }
                    let small =
                        Rc::new(CudaQwen35PrefillCublasPlanV1::new(&stream, 1, k, 1).unwrap());
                    assert!(CudaQwen35QpnRetainedProjectionV1::bind(
                        small,
                        Rc::clone(&stream),
                        Rc::clone(&weight),
                        Rc::clone(&input),
                        0,
                        Rc::clone(&output),
                        0,
                        1
                    )
                    .unwrap()
                    .with_scale_placement(placement)
                    .is_err());
                }
            }
            stream.synchronize().unwrap();
        }
        let after = crate::cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
        assert_eq!(
            after.successful_allocations - before.successful_allocations,
            after.successful_frees - before.successful_frees
        );
    }
}

#[cfg(test)]
mod scalar_tail_tests;
