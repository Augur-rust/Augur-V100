#![allow(clippy::too_many_arguments, clippy::needless_lifetimes)]

use augur_backend::{Backend, CpuBackend, CpuBuffer, CpuStream};
use augur_core::RuntimeError;
use augur_kernel_api::{
    Q4KHotCpuTensor, QMatmulBiasF32Launch, QMatmulF32Launch, QMatmulF32PrefillLaunch,
};
use std::time::Instant;

use crate::{
    BufferIdentity, BufferPurpose, BufferView, MemoryPlan, PlannedDualScaleQ8HotArgs,
    PlannedFp8FusedMlpF32Args, PlannedFp8ScaleGridF32Args, PlannedIq2XxsQ8HotArgs,
    PlannedMxfp4F32Args, PlannedOp, PlannedQ2KQ8HotArgs, PlannedQ8_0Q8HotArgs,
    PlannedQuantizeQ8HotArgs,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PlanResourceAccounting {
    pub unique_weight_count: usize,
    pub hot_weight_payload_bytes: usize,
    pub executable_weight_refs: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CpuExecutionTelemetry {
    pub step_count: usize,
    pub activation_quantize_ns: u128,
    pub q8_0_q8hot_ns: u128,
    pub q4k_f32_ns: u128,
    pub q2k_q8hot_ns: u128,
    pub iq2_xxs_q8hot_ns: u128,
    pub mxfp4_f32_ns: u128,
    pub fp8_scale_grid_f32_ns: u128,
    pub fp8_fused_mlp_f32_ns: u128,
    pub other_ns: u128,
}

impl CpuExecutionTelemetry {
    pub fn total_ns(self) -> u128 {
        self.activation_quantize_ns
            .saturating_add(self.q8_0_q8hot_ns)
            .saturating_add(self.q4k_f32_ns)
            .saturating_add(self.q2k_q8hot_ns)
            .saturating_add(self.iq2_xxs_q8hot_ns)
            .saturating_add(self.mxfp4_f32_ns)
            .saturating_add(self.fp8_scale_grid_f32_ns)
            .saturating_add(self.fp8_fused_mlp_f32_ns)
            .saturating_add(self.other_ns)
    }

    pub fn merge(&mut self, other: Self) {
        self.step_count = self.step_count.saturating_add(other.step_count);
        self.activation_quantize_ns = self
            .activation_quantize_ns
            .saturating_add(other.activation_quantize_ns);
        self.q8_0_q8hot_ns = self.q8_0_q8hot_ns.saturating_add(other.q8_0_q8hot_ns);
        self.q4k_f32_ns = self.q4k_f32_ns.saturating_add(other.q4k_f32_ns);
        self.q2k_q8hot_ns = self.q2k_q8hot_ns.saturating_add(other.q2k_q8hot_ns);
        self.iq2_xxs_q8hot_ns = self.iq2_xxs_q8hot_ns.saturating_add(other.iq2_xxs_q8hot_ns);
        self.mxfp4_f32_ns = self.mxfp4_f32_ns.saturating_add(other.mxfp4_f32_ns);
        self.fp8_scale_grid_f32_ns = self
            .fp8_scale_grid_f32_ns
            .saturating_add(other.fp8_scale_grid_f32_ns);
        self.fp8_fused_mlp_f32_ns = self
            .fp8_fused_mlp_f32_ns
            .saturating_add(other.fp8_fused_mlp_f32_ns);
        self.other_ns = self.other_ns.saturating_add(other.other_ns);
    }
}

#[derive(Clone, Debug)]
pub struct ExecutionPlan<B: Backend> {
    backend: B,
    steps: Vec<PlannedOp<B>>,
    buffers: Vec<B::Buffer>,
    memory: MemoryPlan,
    resource_accounting: PlanResourceAccounting,
}

impl<B: Backend> ExecutionPlan<B> {
    pub fn new(
        backend: B,
        steps: Vec<PlannedOp<B>>,
        buffers: Vec<B::Buffer>,
        memory: MemoryPlan,
        resource_accounting: PlanResourceAccounting,
    ) -> Self {
        Self {
            backend,
            steps,
            buffers,
            memory,
            resource_accounting,
        }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn steps(&self) -> &[PlannedOp<B>] {
        &self.steps
    }

    pub fn memory(&self) -> &MemoryPlan {
        &self.memory
    }

    pub const fn resource_accounting(&self) -> PlanResourceAccounting {
        self.resource_accounting
    }

    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    pub fn buffers(&self) -> &[B::Buffer] {
        &self.buffers
    }

    pub fn buffer(&self, buffer_index: usize) -> Option<&B::Buffer> {
        self.buffers.get(buffer_index)
    }
}

impl ExecutionPlan<CpuBackend> {
    /// Replaces a graph input owner while preserving the memory-plan contract.
    pub fn replace_input(
        &mut self,
        buffer_index: usize,
        replacement: CpuBuffer,
    ) -> Result<CpuBuffer, RuntimeError> {
        let slot = self
            .memory
            .buffers()
            .get(buffer_index)
            .ok_or(RuntimeError::BadBuffer)?;
        if slot.buffer_index != buffer_index
            || slot.purpose != BufferPurpose::Input
            || !matches!(slot.identity, BufferIdentity::Tensor(_))
        {
            return Err(RuntimeError::BadPlan);
        }

        let current = self
            .buffers
            .get(buffer_index)
            .ok_or(RuntimeError::BadBuffer)?;
        let valid_replacement = match (current.as_f32(), replacement.as_f32()) {
            (Some(current), Some(replacement)) => {
                current.len() == slot.len_elements && replacement.len() == slot.len_elements
            }
            _ => false,
        };
        if !valid_replacement {
            return Err(RuntimeError::BadBuffer);
        }

        Ok(std::mem::replace(
            self.buffers
                .get_mut(buffer_index)
                .ok_or(RuntimeError::BadBuffer)?,
            replacement,
        ))
    }

    pub fn set_input_f32(
        &mut self,
        buffer_index: usize,
        input: &[f32],
    ) -> Result<(), RuntimeError> {
        let slot = self
            .memory
            .buffers()
            .get(buffer_index)
            .ok_or(RuntimeError::BadBuffer)?;
        if slot.buffer_index != buffer_index
            || slot.purpose != BufferPurpose::Input
            || !matches!(slot.identity, BufferIdentity::Tensor(_))
        {
            return Err(RuntimeError::BadPlan);
        }

        let dst = self
            .buffers
            .get_mut(buffer_index)
            .and_then(CpuBuffer::as_f32_mut)
            .ok_or(RuntimeError::BadBuffer)?;
        if dst.len() != input.len() {
            return Err(RuntimeError::BadBuffer);
        }
        dst.copy_from_slice(input);
        Ok(())
    }

    pub fn output_f32(&self, buffer_index: usize) -> Result<&[f32], RuntimeError> {
        self.buffers
            .get(buffer_index)
            .and_then(CpuBuffer::as_f32)
            .ok_or(RuntimeError::BadBuffer)
    }

    pub fn execute_cpu(&mut self, _stream: &mut CpuStream) -> Result<(), RuntimeError> {
        for step in &mut self.steps {
            execute_cpu_step(&mut self.buffers, step)?;
        }
        Ok(())
    }

    pub fn execute_cpu_profiled(
        &mut self,
        _stream: &mut CpuStream,
    ) -> Result<CpuExecutionTelemetry, RuntimeError> {
        let mut telemetry = CpuExecutionTelemetry::default();
        for step in &mut self.steps {
            let started = Instant::now();
            execute_cpu_step(&mut self.buffers, step)?;
            let elapsed = started.elapsed().as_nanos();
            telemetry.step_count = telemetry.step_count.saturating_add(1);
            match &*step {
                PlannedOp::QuantizeQ8Hot(_) => {
                    telemetry.activation_quantize_ns =
                        telemetry.activation_quantize_ns.saturating_add(elapsed);
                }
                PlannedOp::Q8_0Q8Hot(_) => {
                    telemetry.q8_0_q8hot_ns = telemetry.q8_0_q8hot_ns.saturating_add(elapsed);
                }
                PlannedOp::QMatmulF32(_) => {
                    telemetry.q4k_f32_ns = telemetry.q4k_f32_ns.saturating_add(elapsed);
                }
                PlannedOp::Q2KQ8Hot(_) => {
                    telemetry.q2k_q8hot_ns = telemetry.q2k_q8hot_ns.saturating_add(elapsed);
                }
                PlannedOp::Iq2XxsQ8Hot(_) => {
                    telemetry.iq2_xxs_q8hot_ns = telemetry.iq2_xxs_q8hot_ns.saturating_add(elapsed);
                }
                PlannedOp::Mxfp4F32(_) => {
                    telemetry.mxfp4_f32_ns = telemetry.mxfp4_f32_ns.saturating_add(elapsed);
                }
                PlannedOp::Fp8ScaleGridF32(_) => {
                    telemetry.fp8_scale_grid_f32_ns =
                        telemetry.fp8_scale_grid_f32_ns.saturating_add(elapsed);
                }
                PlannedOp::Fp8FusedMlpF32(_) => {
                    telemetry.fp8_fused_mlp_f32_ns =
                        telemetry.fp8_fused_mlp_f32_ns.saturating_add(elapsed);
                }
                _ => telemetry.other_ns = telemetry.other_ns.saturating_add(elapsed),
            }
        }
        Ok(telemetry)
    }
}

fn execute_cpu_step(
    buffers: &mut [CpuBuffer],
    step: &mut PlannedOp<CpuBackend>,
) -> Result<(), RuntimeError> {
    match step {
        PlannedOp::AddContiguous { dst, a, b, n } => add_contiguous_f32_buffers(
            buffers,
            dst.buffer_index(),
            dst.offset(),
            a.buffer_index(),
            a.offset(),
            b.buffer_index(),
            b.offset(),
            *n,
        ),
        PlannedOp::MulContiguous { dst, a, b, n } => {
            binary_contiguous_f32_buffers(buffers, dst, a, b, *n, |left, right| left * right)
        }
        PlannedOp::ClampContiguous {
            dst,
            input,
            n,
            min,
            max,
        } => unary_contiguous_f32_buffers(buffers, dst, input, *n, |value| {
            let value = min.map_or(value, |bound| if value < bound { bound } else { value });
            max.map_or(value, |bound| if value > bound { bound } else { value })
        }),
        PlannedOp::SiluContiguous { dst, input, n } => {
            unary_contiguous_f32_buffers(buffers, dst, input, *n, |value| {
                value / (1.0 + (-value).exp())
            })
        }
        PlannedOp::QuantizeQ8Hot(args) => execute_quantize_q8hot(buffers, args),
        PlannedOp::DualScaleQ8Hot(args) => execute_dual_scale_q8hot(buffers, args),
        PlannedOp::Q8_0Q8Hot(args) => execute_q8_0_q8hot(buffers, args),
        PlannedOp::Q2KQ8Hot(args) => execute_q2k_q8hot(buffers, args),
        PlannedOp::Iq2XxsQ8Hot(args) => execute_iq2_xxs_q8hot(buffers, args),
        PlannedOp::Mxfp4F32(args) => execute_mxfp4_f32(buffers, args),
        PlannedOp::Fp8ScaleGridF32(args) => execute_fp8_scale_grid_f32(buffers, args),
        PlannedOp::Fp8FusedMlpF32(args) => execute_fp8_fused_mlp_f32(buffers, args),
        PlannedOp::QMatmulF32(args) => execute_qmatmul_f32(
            buffers,
            args.base().weight().as_ref(),
            args.launch(),
            args.base().input(),
            args.base().output(),
            args.base().rows(),
            args.base().cols(),
        ),
        PlannedOp::QMatmulF32Prefill(args) => execute_qmatmul_f32_prefill(
            buffers,
            args.weight().as_ref(),
            args.launch(),
            args.input(),
            args.output(),
            args.m(),
            args.rows(),
            args.cols(),
        ),
        PlannedOp::QMatmulBiasF32(args) | PlannedOp::QMatmulBiasSiluF32(args) => {
            execute_qmatmul_bias_f32(
                buffers,
                args.base().weight().as_ref(),
                args.launch(),
                args.base().input(),
                args.bias(),
                args.base().output(),
                args.base().rows(),
                args.base().cols(),
            )
        }
    }
}

fn execute_fp8_scale_grid_f32(
    buffers: &mut [CpuBuffer],
    args: &PlannedFp8ScaleGridF32Args<CpuBackend>,
) -> Result<(), RuntimeError> {
    if args.input().len() != args.cols() || args.output().len() != args.rows() {
        return Err(RuntimeError::BadPlan);
    }
    let (input_buffer, output_buffer) = get_read_write(
        buffers,
        args.input().buffer_index(),
        args.output().buffer_index(),
    )?;
    let input = checked_f32_view(
        input_buffer.as_f32().ok_or(RuntimeError::BadBuffer)?,
        args.input(),
    )?;
    let output = checked_f32_view_mut(
        output_buffer.as_f32_mut().ok_or(RuntimeError::BadBuffer)?,
        args.output(),
    )?;
    (args.launch())(args.weight(), input, output).map_err(|_| RuntimeError::KernelFailed)
}

fn execute_fp8_fused_mlp_f32(
    buffers: &mut [CpuBuffer],
    args: &mut PlannedFp8FusedMlpF32Args<CpuBackend>,
) -> Result<(), RuntimeError> {
    let hidden = args.hidden();
    let (gate, up, down, input_view, output_view, config, workspace, launch) = args.launch_parts();
    if input_view.len() != hidden || output_view.len() != hidden {
        return Err(RuntimeError::BadPlan);
    }
    let (input_buffer, output_buffer) = get_read_write(
        buffers,
        input_view.buffer_index(),
        output_view.buffer_index(),
    )?;
    let input = checked_f32_view(
        input_buffer.as_f32().ok_or(RuntimeError::BadBuffer)?,
        input_view,
    )?;
    let output = checked_f32_view_mut(
        output_buffer.as_f32_mut().ok_or(RuntimeError::BadBuffer)?,
        output_view,
    )?;
    launch(gate, up, down, input, config, workspace, output).map_err(|_| RuntimeError::KernelFailed)
}

fn execute_mxfp4_f32(
    buffers: &mut [CpuBuffer],
    args: &PlannedMxfp4F32Args<CpuBackend>,
) -> Result<(), RuntimeError> {
    if args.input().len() != args.cols() || args.output().len() != args.rows() {
        return Err(RuntimeError::BadPlan);
    }
    let (input_buffer, output_buffer) = get_read_write(
        buffers,
        args.input().buffer_index(),
        args.output().buffer_index(),
    )?;
    let input_owner = input_buffer.as_f32().ok_or(RuntimeError::BadBuffer)?;
    let input = checked_f32_view(input_owner, args.input())?;
    let output_owner = output_buffer.as_f32_mut().ok_or(RuntimeError::BadBuffer)?;
    let output = checked_f32_view_mut(output_owner, args.output())?;
    (args.launch())(args.weight(), input, output).map_err(|_| RuntimeError::KernelFailed)
}

fn get_read_write<'a, T>(
    bufs: &'a mut [T],
    r: usize,
    w: usize,
) -> Result<(&'a T, &'a mut T), RuntimeError> {
    if r >= bufs.len() || w >= bufs.len() {
        return Err(RuntimeError::BadBuffer);
    }
    if r == w {
        return Err(RuntimeError::BadPlan);
    }
    if r < w {
        let (l, ri) = bufs.split_at_mut(w);
        Ok((&l[r], &mut ri[0]))
    } else {
        let (l, ri) = bufs.split_at_mut(r);
        Ok((&ri[0], &mut l[w]))
    }
}

fn get_two_reads_one_write<'a, T>(
    bufs: &'a mut [T],
    r0: usize,
    r1: usize,
    w: usize,
) -> Result<(&'a T, &'a T, &'a mut T), RuntimeError> {
    if r0 >= bufs.len() || r1 >= bufs.len() || w >= bufs.len() {
        return Err(RuntimeError::BadBuffer);
    }
    if w == r0 || w == r1 {
        return Err(RuntimeError::BadPlan);
    }
    let (left, right) = bufs.split_at_mut(w);
    let (wc, rest) = right.split_at_mut(1);
    let resolve = |idx: usize| -> &T {
        if idx < w {
            &left[idx]
        } else {
            &rest[idx - w - 1]
        }
    };
    Ok((resolve(r0), resolve(r1), &mut wc[0]))
}

fn add_contiguous_f32_buffers(
    bufs: &mut [CpuBuffer],
    dst: usize,
    dst_off: usize,
    a: usize,
    a_off: usize,
    b: usize,
    b_off: usize,
    n: usize,
) -> Result<(), RuntimeError> {
    if dst == a || dst == b {
        return Err(RuntimeError::BadPlan);
    }
    if dst_off != 0 || a_off != 0 || b_off != 0 {
        return Err(RuntimeError::BadPlan);
    }
    let (a_ref, b_ref, dst_ref) = get_two_reads_one_write(bufs, a, b, dst)?;
    let av = a_ref.as_f32().ok_or(RuntimeError::BadBuffer)?;
    let bv = b_ref.as_f32().ok_or(RuntimeError::BadBuffer)?;
    let d = dst_ref.as_f32_mut().ok_or(RuntimeError::BadBuffer)?;
    if av.len() < n || bv.len() < n || d.len() < n {
        return Err(RuntimeError::BadBuffer);
    }
    for i in 0..n {
        d[i] = av[i] + bv[i];
    }
    Ok(())
}

fn binary_contiguous_f32_buffers(
    bufs: &mut [CpuBuffer],
    dst: &BufferView<CpuBackend>,
    a: &BufferView<CpuBackend>,
    b: &BufferView<CpuBackend>,
    n: usize,
    op: impl Fn(f32, f32) -> f32,
) -> Result<(), RuntimeError> {
    if dst.offset() != 0 || a.offset() != 0 || b.offset() != 0 {
        return Err(RuntimeError::BadPlan);
    }
    let (a_ref, b_ref, dst_ref) =
        get_two_reads_one_write(bufs, a.buffer_index(), b.buffer_index(), dst.buffer_index())?;
    let av = a_ref.as_f32().ok_or(RuntimeError::BadBuffer)?;
    let bv = b_ref.as_f32().ok_or(RuntimeError::BadBuffer)?;
    let output = dst_ref.as_f32_mut().ok_or(RuntimeError::BadBuffer)?;
    if av.len() < n || bv.len() < n || output.len() < n {
        return Err(RuntimeError::BadBuffer);
    }
    for index in 0..n {
        output[index] = op(av[index], bv[index]);
    }
    Ok(())
}

fn unary_contiguous_f32_buffers(
    bufs: &mut [CpuBuffer],
    dst: &BufferView<CpuBackend>,
    input: &BufferView<CpuBackend>,
    n: usize,
    op: impl Fn(f32) -> f32,
) -> Result<(), RuntimeError> {
    if dst.offset() != 0 || input.offset() != 0 {
        return Err(RuntimeError::BadPlan);
    }
    let (input_buffer, output_buffer) =
        get_read_write(bufs, input.buffer_index(), dst.buffer_index())?;
    let values = input_buffer.as_f32().ok_or(RuntimeError::BadBuffer)?;
    let output = output_buffer.as_f32_mut().ok_or(RuntimeError::BadBuffer)?;
    if values.len() < n || output.len() < n {
        return Err(RuntimeError::BadBuffer);
    }
    for index in 0..n {
        output[index] = op(values[index]);
    }
    Ok(())
}

fn execute_quantize_q8hot(
    bufs: &mut [CpuBuffer],
    args: &PlannedQuantizeQ8HotArgs<CpuBackend>,
) -> Result<(), RuntimeError> {
    let (input_buffer, output_buffer) = get_read_write(
        bufs,
        args.input().buffer_index(),
        args.output().buffer_index(),
    )?;
    let input_owner = input_buffer.as_f32().ok_or(RuntimeError::BadBuffer)?;
    let input = checked_f32_view(input_owner, args.input())?;
    let output = output_buffer
        .as_q8hot_mut()
        .ok_or(RuntimeError::BadBuffer)?;
    if args.output().offset() != 0
        || args.output().len() != input.len()
        || output.len_elements() != args.output().len()
    {
        return Err(RuntimeError::BadPlan);
    }
    (args.launch())(input, output).map_err(|_| RuntimeError::KernelFailed)
}

fn execute_dual_scale_q8hot(
    buffers: &mut [CpuBuffer],
    args: &PlannedDualScaleQ8HotArgs<CpuBackend>,
) -> Result<(), RuntimeError> {
    let (activation_buffer, output_buffer) = get_read_write(
        buffers,
        args.activation().buffer_index(),
        args.output().buffer_index(),
    )?;

    let activation = activation_buffer
        .as_q8hot()
        .ok_or(RuntimeError::BadBuffer)?;
    if args.activation().offset() != 0
        || args.activation().len() != args.cols()
        || activation.len_elements() != args.cols()
        || args.output().len() != args.rows()
    {
        return Err(RuntimeError::BadPlan);
    }

    let output_owner = output_buffer.as_f32_mut().ok_or(RuntimeError::BadBuffer)?;
    let output = checked_f32_view_mut(output_owner, args.output())?;

    (args.launch())(args.bound(), activation, output);
    Ok(())
}

fn execute_q8_0_q8hot(
    buffers: &mut [CpuBuffer],
    args: &PlannedQ8_0Q8HotArgs<CpuBackend>,
) -> Result<(), RuntimeError> {
    execute_mixed_q8hot(
        buffers,
        args.activation(),
        args.output(),
        args.rows(),
        args.cols(),
        |activation, output| (args.launch())(args.weight(), activation, output),
    )
}

fn execute_q2k_q8hot(
    buffers: &mut [CpuBuffer],
    args: &PlannedQ2KQ8HotArgs<CpuBackend>,
) -> Result<(), RuntimeError> {
    execute_mixed_q8hot(
        buffers,
        args.activation(),
        args.output(),
        args.rows(),
        args.cols(),
        |activation, output| (args.launch())(args.weight(), activation, output),
    )
}

fn execute_iq2_xxs_q8hot(
    buffers: &mut [CpuBuffer],
    args: &PlannedIq2XxsQ8HotArgs<CpuBackend>,
) -> Result<(), RuntimeError> {
    execute_mixed_q8hot(
        buffers,
        args.activation(),
        args.output(),
        args.rows(),
        args.cols(),
        |activation, output| (args.launch())(args.weight(), activation, output),
    )
}

fn execute_mixed_q8hot(
    buffers: &mut [CpuBuffer],
    activation_view: &BufferView<CpuBackend>,
    output_view: &BufferView<CpuBackend>,
    rows: usize,
    cols: usize,
    launch: impl FnOnce(
        &augur_kernel_api::Q8KActivationHotCpuVector,
        &mut [f32],
    ) -> Result<(), augur_kernel_api::KernelLaunchError>,
) -> Result<(), RuntimeError> {
    let (activation_buffer, output_buffer) = get_read_write(
        buffers,
        activation_view.buffer_index(),
        output_view.buffer_index(),
    )?;
    let activation = activation_buffer
        .as_q8hot()
        .ok_or(RuntimeError::BadBuffer)?;
    if activation_view.offset() != 0
        || activation_view.len() != cols
        || activation.len_elements() != cols
        || output_view.len() != rows
    {
        return Err(RuntimeError::BadPlan);
    }
    let output_owner = output_buffer.as_f32_mut().ok_or(RuntimeError::BadBuffer)?;
    let output = checked_f32_view_mut(output_owner, output_view)?;
    launch(activation, output).map_err(|_| RuntimeError::KernelFailed)
}

fn checked_f32_view<'a, B: Backend>(
    owner: &'a [f32],
    view: &BufferView<B>,
) -> Result<&'a [f32], RuntimeError> {
    let end = view
        .offset()
        .checked_add(view.len())
        .ok_or(RuntimeError::BadBuffer)?;
    owner.get(view.offset()..end).ok_or(RuntimeError::BadBuffer)
}

fn checked_f32_view_mut<'a, B: Backend>(
    owner: &'a mut [f32],
    view: &BufferView<B>,
) -> Result<&'a mut [f32], RuntimeError> {
    let end = view
        .offset()
        .checked_add(view.len())
        .ok_or(RuntimeError::BadBuffer)?;
    owner
        .get_mut(view.offset()..end)
        .ok_or(RuntimeError::BadBuffer)
}

fn execute_qmatmul_f32(
    bufs: &mut [CpuBuffer],
    weight: &Q4KHotCpuTensor,
    launch: QMatmulF32Launch,
    input: &BufferView<CpuBackend>,
    output: &BufferView<CpuBackend>,
    rows: usize,
    cols: usize,
) -> Result<(), RuntimeError> {
    if input.len() != cols || output.len() != rows {
        return Err(RuntimeError::BadPlan);
    }
    let (ib, ob) = get_read_write(bufs, input.buffer_index(), output.buffer_index())?;
    let act = checked_f32_view(ib.as_f32().ok_or(RuntimeError::BadBuffer)?, input)?;
    let dst = checked_f32_view_mut(ob.as_f32_mut().ok_or(RuntimeError::BadBuffer)?, output)?;
    launch(weight, act, dst).map_err(|_| RuntimeError::KernelFailed)
}

#[allow(clippy::too_many_arguments)]
fn execute_qmatmul_f32_prefill(
    bufs: &mut [CpuBuffer],
    weight: &Q4KHotCpuTensor,
    launch: QMatmulF32PrefillLaunch,
    input: &BufferView<CpuBackend>,
    output: &BufferView<CpuBackend>,
    m: usize,
    rows: usize,
    cols: usize,
) -> Result<(), RuntimeError> {
    let input_len = m.checked_mul(cols).ok_or(RuntimeError::BadPlan)?;
    let output_len = m.checked_mul(rows).ok_or(RuntimeError::BadPlan)?;
    if m == 0 || input.len() != input_len || output.len() != output_len {
        return Err(RuntimeError::BadPlan);
    }
    let (input_buffer, output_buffer) =
        get_read_write(bufs, input.buffer_index(), output.buffer_index())?;
    let input_owner = input_buffer.as_f32().ok_or(RuntimeError::BadBuffer)?;
    let input = checked_f32_view(input_owner, input)?;
    let output_owner = output_buffer.as_f32_mut().ok_or(RuntimeError::BadBuffer)?;
    let output = checked_f32_view_mut(output_owner, output)?;
    launch(weight, input, m, output).map_err(|_| RuntimeError::KernelFailed)
}

fn execute_qmatmul_bias_f32(
    bufs: &mut [CpuBuffer],
    weight: &Q4KHotCpuTensor,
    launch: QMatmulBiasF32Launch,
    input: &BufferView<CpuBackend>,
    bias: &BufferView<CpuBackend>,
    output: &BufferView<CpuBackend>,
    rows: usize,
    cols: usize,
) -> Result<(), RuntimeError> {
    if input.len() != cols || bias.len() != rows || output.len() != rows {
        return Err(RuntimeError::BadPlan);
    }
    let (ib, bb, ob) = get_two_reads_one_write(
        bufs,
        input.buffer_index(),
        bias.buffer_index(),
        output.buffer_index(),
    )?;
    let act = checked_f32_view(ib.as_f32().ok_or(RuntimeError::BadBuffer)?, input)?;
    let bv = checked_f32_view(bb.as_f32().ok_or(RuntimeError::BadBuffer)?, bias)?;
    let dst = checked_f32_view_mut(ob.as_f32_mut().ok_or(RuntimeError::BadBuffer)?, output)?;
    launch(weight, act, bv, dst).map_err(|_| RuntimeError::KernelFailed)
}

#[cfg(test)]
mod tests {
    use super::{checked_f32_view, checked_f32_view_mut};
    use crate::BufferView;
    use augur_backend::CpuBackend;
    use augur_core::RuntimeError;

    #[test]
    fn checked_view_rejects_extreme_offset() {
        let view = BufferView::<CpuBackend>::dense_f32(0, 0)
            .with_range(usize::MAX, 0)
            .expect("non-overflowing range");
        let owner = [0.0_f32; 4];
        assert_eq!(
            checked_f32_view(&owner, &view),
            Err(RuntimeError::BadBuffer)
        );
    }

    #[test]
    fn checked_mut_view_rejects_range_past_owner() {
        let view = BufferView::<CpuBackend>::dense_f32(0, 3)
            .with_range(3, 3)
            .expect("range metadata");
        let mut owner = [0.0_f32; 4];
        assert_eq!(
            checked_f32_view_mut(&mut owner, &view).map(|slice| slice.len()),
            Err(RuntimeError::BadBuffer)
        );
    }
}
