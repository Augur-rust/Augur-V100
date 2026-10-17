use std::sync::Arc;

use augur_backend::{CpuBackend, CpuBackend as BackendType, CpuBuffer};
use augur_core::{PhysicalLayoutId, QuantFormatId};
use augur_exec::{
    ExecutionPlan, PlanResourceAccounting, PlannedDualScaleQ8HotArgs, PlannedFp8FusedMlpF32Args,
    PlannedFp8ScaleGridF32Args, PlannedIq2XxsQ8HotArgs, PlannedMxfp4F32Args, PlannedOp,
    PlannedQ2KQ8HotArgs, PlannedQ8_0Q8HotArgs, PlannedQMatmulBiasF32Args, PlannedQMatmulF32Args,
    PlannedQMatmulF32Base, PlannedQMatmulF32PrefillArgs, PlannedQuantizeQ8HotArgs,
};
use augur_kernel_api::{
    BoundDualScaleQ8Hot, DualScaleHotCpuBlock, DualScaleQ8HotCpuLaunch,
    DualScaleQ8HotCpuPrevalidatedLaunch, Fp8FusedMlpF32Workspace, Q4KHotCpuTensor,
    Q8K_HOT_CPU_BLOCK_LEN,
};
use augur_kernels::portable::{
    launch_qmatvec_dual_scale_q8k_hot_cpu_portable,
    launch_qmatvec_dual_scale_q8k_hot_cpu_portable_prevalidated,
};
#[cfg(target_arch = "x86_64")]
use augur_kernels::x86::avx2::{
    launch_qmatvec_dual_scale_q8k_hot_cpu_avx2,
    launch_qmatvec_dual_scale_q8k_hot_cpu_avx2_prevalidated,
};

use crate::{
    bind_cpu_weight_resources, BindError, CompiledQMatmulF32Base, CompiledQuantizeQ8Hot,
    CompiledStepBindFailure, CpuBindBufferRole, CpuBindStepKind, CpuBoundWeights, CpuCompiledPlan,
    CpuCompiledStep, CpuWeightRequirement, CpuWeightResourceKind, CpuWeightResources,
    WeightBindingId,
};

// ── E3 binding pipeline ────────────────────────────────────────────────────

/// Executable census collected before any resource is taken.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct StepResourceAccounting {
    executable_weight_refs: usize,
}

pub fn bind_cpu_plan(
    compiled: CpuCompiledPlan,
    resources: CpuWeightResources,
) -> Result<ExecutionPlan<CpuBackend>, BindError> {
    let CpuCompiledPlan {
        steps,
        buffers,
        memory,
        weight_requirements,
    } = compiled;

    // Phase 1: validate all steps and buffers, compute accounting.
    let step_accounting = validate_compiled_steps(&steps, &buffers, &weight_requirements)?;

    // Phase 2: validate and take all weight owners.
    let bound_weights = bind_cpu_weight_resources(&weight_requirements, resources)?;

    let binding_accounting = bound_weights.accounting();

    // Phase 3: mechanical conversion of pre-validated steps.
    let mut planned_steps = Vec::with_capacity(steps.len());

    for step in steps {
        let planned = bind_prevalidated_step(step, &bound_weights)?;
        planned_steps.push(planned);
    }

    Ok(ExecutionPlan::new(
        BackendType,
        planned_steps,
        buffers,
        memory,
        PlanResourceAccounting {
            unique_weight_count: binding_accounting.unique_weight_count,
            hot_weight_payload_bytes: binding_accounting.hot_weight_payload_bytes,
            executable_weight_refs: step_accounting.executable_weight_refs,
        },
    ))
}

// ── Phase 1: step preflight ────────────────────────────────────────────────

fn validate_compiled_steps(
    steps: &[CpuCompiledStep],
    buffers: &[CpuBuffer],
    requirements: &[CpuWeightRequirement],
) -> Result<StepResourceAccounting, BindError> {
    let mut accounting = StepResourceAccounting::default();

    for step in steps {
        match step {
            CpuCompiledStep::QuantizeQ8Hot(args) => {
                validate_quantize_q8hot(args, buffers)?;
            }

            CpuCompiledStep::DualScaleQ8Hot(args) => {
                validate_dual_scale_q8hot(args, buffers, requirements)?;

                accounting.executable_weight_refs = accounting
                    .executable_weight_refs
                    .checked_add(1)
                    .ok_or(BindError::AccountingOverflow)?;
            }

            CpuCompiledStep::Q8_0Q8Hot(args) => {
                validate_mixed_q8hot(
                    args.weight,
                    &args.activation,
                    &args.output,
                    args.rows,
                    args.cols,
                    CpuBindStepKind::Q8_0Q8Hot,
                    CpuWeightResourceKind::Q8_0HotCpuPortable,
                    QuantFormatId::Q8_0,
                    PhysicalLayoutId::Q8_0_HOT_CPU_PORTABLE_V1,
                    augur_kernel_api::Q8_0HotCpuBlock::DECODED_LEN,
                    buffers,
                    requirements,
                )?;
                accounting.executable_weight_refs = accounting
                    .executable_weight_refs
                    .checked_add(1)
                    .ok_or(BindError::AccountingOverflow)?;
            }

            CpuCompiledStep::Q2KQ8Hot(args) => {
                validate_mixed_q8hot(
                    args.weight,
                    &args.activation,
                    &args.output,
                    args.rows,
                    args.cols,
                    CpuBindStepKind::Q2KQ8Hot,
                    CpuWeightResourceKind::Q2KHotCpuPortable,
                    QuantFormatId::Q2K,
                    PhysicalLayoutId::Q2K_HOT_CPU_PORTABLE_V1,
                    augur_kernel_api::Q2KHotCpuBlock::DECODED_LEN,
                    buffers,
                    requirements,
                )?;
                accounting.executable_weight_refs = accounting
                    .executable_weight_refs
                    .checked_add(1)
                    .ok_or(BindError::AccountingOverflow)?;
            }

            CpuCompiledStep::Iq2XxsQ8Hot(args) => {
                validate_mixed_q8hot(
                    args.weight,
                    &args.activation,
                    &args.output,
                    args.rows,
                    args.cols,
                    CpuBindStepKind::Iq2XxsQ8Hot,
                    CpuWeightResourceKind::Iq2XxsHotCpuPortable,
                    QuantFormatId::IQ2_XXS,
                    PhysicalLayoutId::IQ2_XXS_HOT_CPU_PORTABLE_V1,
                    augur_kernel_api::Iq2XxsHotCpuBlock::DECODED_LEN,
                    buffers,
                    requirements,
                )?;
                accounting.executable_weight_refs = accounting
                    .executable_weight_refs
                    .checked_add(1)
                    .ok_or(BindError::AccountingOverflow)?;
            }

            CpuCompiledStep::Mxfp4F32(args) => {
                validate_mxfp4_f32(args, buffers, requirements)?;
                accounting.executable_weight_refs = accounting
                    .executable_weight_refs
                    .checked_add(1)
                    .ok_or(BindError::AccountingOverflow)?;
            }

            CpuCompiledStep::Fp8ScaleGridF32(args) => {
                validate_fp8_scale_grid_f32(args, buffers, requirements)?;
                accounting.executable_weight_refs = accounting
                    .executable_weight_refs
                    .checked_add(1)
                    .ok_or(BindError::AccountingOverflow)?;
            }

            CpuCompiledStep::Fp8FusedMlpF32(args) => {
                validate_fp8_fused_mlp_f32(args, buffers, requirements)?;
                accounting.executable_weight_refs = accounting
                    .executable_weight_refs
                    .checked_add(3)
                    .ok_or(BindError::AccountingOverflow)?;
            }

            CpuCompiledStep::QMatmulF32(args) => {
                validate_qmatmul(
                    &args.base,
                    None,
                    CpuBindStepKind::QMatmulF32,
                    buffers,
                    requirements,
                )?;
                accounting.executable_weight_refs = accounting
                    .executable_weight_refs
                    .checked_add(1)
                    .ok_or(BindError::AccountingOverflow)?;
            }
            CpuCompiledStep::QMatmulF32Prefill(args) => {
                validate_qmatmul_prefill(args, buffers, requirements)?;
                accounting.executable_weight_refs = accounting
                    .executable_weight_refs
                    .checked_add(1)
                    .ok_or(BindError::AccountingOverflow)?;
            }
            CpuCompiledStep::QMatmulBiasF32(args) => {
                validate_qmatmul(
                    &args.base,
                    Some(&args.bias),
                    CpuBindStepKind::QMatmulBiasF32,
                    buffers,
                    requirements,
                )?;
                accounting.executable_weight_refs = accounting
                    .executable_weight_refs
                    .checked_add(1)
                    .ok_or(BindError::AccountingOverflow)?;
            }
            CpuCompiledStep::QMatmulBiasSiluF32(args) => {
                validate_qmatmul(
                    &args.base,
                    Some(&args.bias),
                    CpuBindStepKind::QMatmulBiasSiluF32,
                    buffers,
                    requirements,
                )?;
                accounting.executable_weight_refs = accounting
                    .executable_weight_refs
                    .checked_add(1)
                    .ok_or(BindError::AccountingOverflow)?;
            }

            CpuCompiledStep::Add(args) => validate_binary_elementwise(
                CpuBindStepKind::Add,
                &args.a,
                &args.b,
                &args.dst,
                args.n,
                buffers,
            )?,
            CpuCompiledStep::Mul(args) => validate_binary_elementwise(
                CpuBindStepKind::Mul,
                &args.a,
                &args.b,
                &args.dst,
                args.n,
                buffers,
            )?,
            CpuCompiledStep::Clamp(args) => {
                validate_unary_elementwise(
                    CpuBindStepKind::Clamp,
                    &args.input,
                    &args.dst,
                    args.n,
                    buffers,
                )?;
                if (args.min.is_none() && args.max.is_none())
                    || args.min.is_some_and(|value| !value.is_finite())
                    || args.max.is_some_and(|value| !value.is_finite())
                    || matches!((args.min, args.max), (Some(min), Some(max)) if min > max)
                {
                    return Err(qmatmul_error(
                        CpuBindStepKind::Clamp,
                        CompiledStepBindFailure::InvalidDimensions {
                            rows: args.n,
                            cols: 1,
                            blocks_per_row: 0,
                        },
                    ));
                }
            }
            CpuCompiledStep::Silu(args) => validate_unary_elementwise(
                CpuBindStepKind::Silu,
                &args.input,
                &args.dst,
                args.n,
                buffers,
            )?,
        }
    }

    Ok(accounting)
}

fn validate_unary_elementwise(
    step: CpuBindStepKind,
    input: &augur_exec::BufferView<CpuBackend>,
    output: &augur_exec::BufferView<CpuBackend>,
    n: usize,
    buffers: &[CpuBuffer],
) -> Result<(), BindError> {
    if n == 0 {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::InvalidDimensions {
                rows: n,
                cols: 1,
                blocks_per_row: 0,
            },
        ));
    }
    validate_qmatmul_f32_view(step, CpuBindBufferRole::Input, input, n, buffers)?;
    validate_qmatmul_f32_view(step, CpuBindBufferRole::Output, output, n, buffers)?;
    if input.buffer_index() == output.buffer_index() {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::AliasedBuffers {
                read: input.buffer_index(),
                write: output.buffer_index(),
            },
        ));
    }
    Ok(())
}

fn validate_binary_elementwise(
    step: CpuBindStepKind,
    left: &augur_exec::BufferView<CpuBackend>,
    right: &augur_exec::BufferView<CpuBackend>,
    output: &augur_exec::BufferView<CpuBackend>,
    n: usize,
    buffers: &[CpuBuffer],
) -> Result<(), BindError> {
    validate_unary_elementwise(step, left, output, n, buffers)?;
    validate_qmatmul_f32_view(step, CpuBindBufferRole::Right, right, n, buffers)?;
    if right.buffer_index() == output.buffer_index() {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::AliasedBuffers {
                read: right.buffer_index(),
                write: output.buffer_index(),
            },
        ));
    }
    Ok(())
}

fn qmatmul_error(step: CpuBindStepKind, failure: CompiledStepBindFailure) -> BindError {
    BindError::CompiledStep { step, failure }
}

fn validate_qmatmul(
    base: &CompiledQMatmulF32Base,
    bias: Option<&augur_exec::BufferView<CpuBackend>>,
    step: CpuBindStepKind,
    buffers: &[CpuBuffer],
    requirements: &[CpuWeightRequirement],
) -> Result<(), BindError> {
    for (role, view, expected_len) in [
        (CpuBindBufferRole::Input, &base.input, base.cols),
        (CpuBindBufferRole::Output, &base.output, base.rows),
    ] {
        validate_qmatmul_f32_view(step, role, view, expected_len, buffers)?;
    }
    if base.rows == 0 || base.cols == 0 {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::InvalidDimensions {
                rows: base.rows,
                cols: base.cols,
                blocks_per_row: 0,
            },
        ));
    }
    if base.input.buffer_index() == base.output.buffer_index() {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::AliasedBuffers {
                read: base.input.buffer_index(),
                write: base.output.buffer_index(),
            },
        ));
    }
    if let Some(bias) = bias {
        validate_qmatmul_f32_view(step, CpuBindBufferRole::Bias, bias, base.rows, buffers)?;
        if bias.buffer_index() == base.output.buffer_index() {
            return Err(qmatmul_error(
                step,
                CompiledStepBindFailure::AliasedBuffers {
                    read: bias.buffer_index(),
                    write: base.output.buffer_index(),
                },
            ));
        }
    }
    let requirement = requirements
        .get(base.weight.index())
        .ok_or(BindError::InvalidBindingId {
            binding: base.weight,
        })?;
    if requirement.binding != base.weight
        || requirement.kind != CpuWeightResourceKind::Q4KHotCpuPortable
        || requirement.quant_format != QuantFormatId::Q4K
        || requirement.physical_layout != Q4KHotCpuTensor::PHYSICAL_LAYOUT
        || requirement.block_size != augur_kernel_api::Q4KHotCpuBlock::DECODED_LEN
        || requirement.rows != base.rows
        || requirement.cols != base.cols
    {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::WeightRequirementMismatch {
                binding: base.weight,
            },
        ));
    }
    Ok(())
}

fn validate_qmatmul_prefill(
    args: &crate::CompiledQMatmulF32Prefill,
    buffers: &[CpuBuffer],
    requirements: &[CpuWeightRequirement],
) -> Result<(), BindError> {
    let step = CpuBindStepKind::QMatmulF32Prefill;
    let input_len = args.m.checked_mul(args.cols).ok_or_else(|| {
        qmatmul_error(
            step,
            CompiledStepBindFailure::InvalidDimensions {
                rows: args.rows,
                cols: args.cols,
                blocks_per_row: 0,
            },
        )
    })?;
    let output_len = args.m.checked_mul(args.rows).ok_or_else(|| {
        qmatmul_error(
            step,
            CompiledStepBindFailure::InvalidDimensions {
                rows: args.rows,
                cols: args.cols,
                blocks_per_row: 0,
            },
        )
    })?;
    if args.m == 0 || args.rows == 0 || args.cols == 0 {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::InvalidDimensions {
                rows: args.rows,
                cols: args.cols,
                blocks_per_row: 0,
            },
        ));
    }
    validate_qmatmul_f32_view(
        step,
        CpuBindBufferRole::Input,
        &args.input,
        input_len,
        buffers,
    )?;
    validate_qmatmul_f32_view(
        step,
        CpuBindBufferRole::Output,
        &args.output,
        output_len,
        buffers,
    )?;
    if args.input.buffer_index() == args.output.buffer_index() {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::AliasedBuffers {
                read: args.input.buffer_index(),
                write: args.output.buffer_index(),
            },
        ));
    }
    let requirement = requirements
        .get(args.weight.index())
        .ok_or(BindError::InvalidBindingId {
            binding: args.weight,
        })?;
    if requirement.binding != args.weight
        || requirement.kind != CpuWeightResourceKind::Q4KHotCpuPortable
        || requirement.quant_format != QuantFormatId::Q4K
        || requirement.physical_layout != Q4KHotCpuTensor::PHYSICAL_LAYOUT
        || requirement.block_size != augur_kernel_api::Q4KHotCpuBlock::DECODED_LEN
        || requirement.rows != args.rows
        || requirement.cols != args.cols
    {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::WeightRequirementMismatch {
                binding: args.weight,
            },
        ));
    }
    Ok(())
}

fn validate_qmatmul_f32_view(
    step: CpuBindStepKind,
    role: CpuBindBufferRole,
    view: &augur_exec::BufferView<CpuBackend>,
    expected_len: usize,
    buffers: &[CpuBuffer],
) -> Result<(), BindError> {
    let index = view.buffer_index();
    if index >= buffers.len() {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::BufferIndexOutOfRange {
                role,
                index,
                buffer_count: buffers.len(),
            },
        ));
    }
    if !view.is_cpu_contiguous_f32() {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::MetadataMismatch {
                role,
                buffer_index: index,
            },
        ));
    }
    if view.len() != expected_len {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::OwnerShapeMismatch {
                role,
                buffer_index: index,
                expected_len,
                actual_len: view.len(),
                expected_blocks: None,
                actual_blocks: None,
            },
        ));
    }
    let owner = buffers[index].as_f32().ok_or_else(|| {
        qmatmul_error(
            step,
            CompiledStepBindFailure::OwnerKindMismatch {
                role,
                buffer_index: index,
            },
        )
    })?;
    let bounds = || {
        qmatmul_error(
            step,
            CompiledStepBindFailure::BufferBounds {
                role,
                buffer_index: index,
                offset: view.offset(),
                len: view.len(),
                owner_len: owner.len(),
            },
        )
    };
    let end = view.offset().checked_add(view.len()).ok_or_else(bounds)?;
    if end > owner.len() {
        return Err(bounds());
    }
    Ok(())
}

// ── DualScale step preflight ───────────────────────────────────────────────

fn dual_scale_bind_error(failure: CompiledStepBindFailure) -> BindError {
    BindError::CompiledStep {
        step: CpuBindStepKind::DualScaleQ8Hot,
        failure,
    }
}

fn invalid_dual_scale_dimensions(args: &crate::CompiledDualScaleQ8Hot) -> BindError {
    dual_scale_bind_error(CompiledStepBindFailure::InvalidDimensions {
        rows: args.rows,
        cols: args.cols,
        blocks_per_row: args.blocks_per_row,
    })
}

fn output_bounds_error(args: &crate::CompiledDualScaleQ8Hot, owner_len: usize) -> BindError {
    dual_scale_bind_error(CompiledStepBindFailure::BufferBounds {
        role: CpuBindBufferRole::Output,
        buffer_index: args.output.buffer_index(),
        offset: args.output.offset(),
        len: args.output.len(),
        owner_len,
    })
}

fn validate_dual_scale_q8hot(
    args: &crate::CompiledDualScaleQ8Hot,
    buffers: &[CpuBuffer],
    requirements: &[crate::CpuWeightRequirement],
) -> Result<(), BindError> {
    let activation_index = args.activation.buffer_index();
    let output_index = args.output.buffer_index();
    let buffer_count = buffers.len();

    if activation_index >= buffer_count {
        return Err(dual_scale_bind_error(
            CompiledStepBindFailure::BufferIndexOutOfRange {
                role: CpuBindBufferRole::Activation,
                index: activation_index,
                buffer_count,
            },
        ));
    }

    if output_index >= buffer_count {
        return Err(dual_scale_bind_error(
            CompiledStepBindFailure::BufferIndexOutOfRange {
                role: CpuBindBufferRole::Output,
                index: output_index,
                buffer_count,
            },
        ));
    }

    if activation_index == output_index {
        return Err(dual_scale_bind_error(
            CompiledStepBindFailure::AliasedBuffers {
                read: activation_index,
                write: output_index,
            },
        ));
    }

    if !args.activation.is_cpu_q8hot() {
        return Err(dual_scale_bind_error(
            CompiledStepBindFailure::MetadataMismatch {
                role: CpuBindBufferRole::Activation,
                buffer_index: activation_index,
            },
        ));
    }

    if !args.output.is_cpu_contiguous_f32() {
        return Err(dual_scale_bind_error(
            CompiledStepBindFailure::MetadataMismatch {
                role: CpuBindBufferRole::Output,
                buffer_index: output_index,
            },
        ));
    }

    if args.activation.offset() != 0 {
        return Err(dual_scale_bind_error(
            CompiledStepBindFailure::UnsupportedOffset {
                role: CpuBindBufferRole::Activation,
                actual: args.activation.offset(),
            },
        ));
    }

    if args.output.offset() != 0 {
        return Err(dual_scale_bind_error(
            CompiledStepBindFailure::UnsupportedOffset {
                role: CpuBindBufferRole::Output,
                actual: args.output.offset(),
            },
        ));
    }

    if args.rows == 0 || args.cols == 0 || args.blocks_per_row == 0 {
        return Err(invalid_dual_scale_dimensions(args));
    }

    let expected_cols = args
        .blocks_per_row
        .checked_mul(DualScaleHotCpuBlock::DECODED_LEN)
        .ok_or_else(|| invalid_dual_scale_dimensions(args))?;

    if expected_cols != args.cols
        || args.activation.len() != args.cols
        || args.output.len() != args.rows
    {
        return Err(invalid_dual_scale_dimensions(args));
    }

    let requirement = requirements
        .get(args.weight.index())
        .ok_or(BindError::InvalidBindingId {
            binding: args.weight,
        })?;

    if requirement.binding != args.weight
        || requirement.kind != CpuWeightResourceKind::DualScaleHotCpuPortable
        || requirement.quant_format != QuantFormatId::DualScale
        || requirement.physical_layout != PhysicalLayoutId::DUAL_SCALE_HOT_CPU_PORTABLE_V1
        || requirement.block_size != DualScaleHotCpuBlock::DECODED_LEN
        || requirement.rows != args.rows
        || requirement.cols != args.cols
    {
        return Err(dual_scale_bind_error(
            CompiledStepBindFailure::WeightRequirementMismatch {
                binding: args.weight,
            },
        ));
    }

    let activation_owner = buffers
        .get(activation_index)
        .and_then(CpuBuffer::as_q8hot)
        .ok_or_else(|| {
            dual_scale_bind_error(CompiledStepBindFailure::OwnerKindMismatch {
                role: CpuBindBufferRole::Activation,
                buffer_index: activation_index,
            })
        })?;

    if activation_owner.len_elements() != args.cols
        || activation_owner.block_count() != args.blocks_per_row
    {
        return Err(dual_scale_bind_error(
            CompiledStepBindFailure::OwnerShapeMismatch {
                role: CpuBindBufferRole::Activation,
                buffer_index: activation_index,
                expected_len: args.cols,
                actual_len: activation_owner.len_elements(),
                expected_blocks: Some(args.blocks_per_row),
                actual_blocks: Some(activation_owner.block_count()),
            },
        ));
    }

    let output_owner = buffers
        .get(output_index)
        .and_then(CpuBuffer::as_f32)
        .ok_or_else(|| {
            dual_scale_bind_error(CompiledStepBindFailure::OwnerKindMismatch {
                role: CpuBindBufferRole::Output,
                buffer_index: output_index,
            })
        })?;

    let output_end = args
        .output
        .offset()
        .checked_add(args.output.len())
        .ok_or_else(|| output_bounds_error(args, output_owner.len()))?;

    if output_end > output_owner.len() {
        return Err(output_bounds_error(args, output_owner.len()));
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_mixed_q8hot(
    weight: WeightBindingId,
    activation: &augur_exec::BufferView<CpuBackend>,
    output: &augur_exec::BufferView<CpuBackend>,
    rows: usize,
    cols: usize,
    step: CpuBindStepKind,
    kind: CpuWeightResourceKind,
    format: QuantFormatId,
    layout: PhysicalLayoutId,
    block_size: usize,
    buffers: &[CpuBuffer],
    requirements: &[CpuWeightRequirement],
) -> Result<(), BindError> {
    let error = |failure| BindError::CompiledStep { step, failure };
    let activation_index = activation.buffer_index();
    let output_index = output.buffer_index();
    for (role, index) in [
        (CpuBindBufferRole::Activation, activation_index),
        (CpuBindBufferRole::Output, output_index),
    ] {
        if index >= buffers.len() {
            return Err(error(CompiledStepBindFailure::BufferIndexOutOfRange {
                role,
                index,
                buffer_count: buffers.len(),
            }));
        }
    }
    if activation_index == output_index {
        return Err(error(CompiledStepBindFailure::AliasedBuffers {
            read: activation_index,
            write: output_index,
        }));
    }
    if !activation.is_cpu_q8hot() || !output.is_cpu_contiguous_f32() {
        return Err(error(CompiledStepBindFailure::MetadataMismatch {
            role: if !activation.is_cpu_q8hot() {
                CpuBindBufferRole::Activation
            } else {
                CpuBindBufferRole::Output
            },
            buffer_index: if !activation.is_cpu_q8hot() {
                activation_index
            } else {
                output_index
            },
        }));
    }
    if activation.offset() != 0 || output.offset() != 0 {
        return Err(error(CompiledStepBindFailure::UnsupportedOffset {
            role: if activation.offset() != 0 {
                CpuBindBufferRole::Activation
            } else {
                CpuBindBufferRole::Output
            },
            actual: if activation.offset() != 0 {
                activation.offset()
            } else {
                output.offset()
            },
        }));
    }
    if rows == 0
        || cols == 0
        || !cols.is_multiple_of(block_size)
        || activation.len() != cols
        || output.len() != rows
    {
        return Err(error(CompiledStepBindFailure::InvalidDimensions {
            rows,
            cols,
            blocks_per_row: cols.checked_div(block_size).unwrap_or(0),
        }));
    }
    let requirement = requirements
        .get(weight.index())
        .ok_or(BindError::InvalidBindingId { binding: weight })?;
    if requirement.binding != weight
        || requirement.kind != kind
        || requirement.quant_format != format
        || requirement.physical_layout != layout
        || requirement.block_size != block_size
        || requirement.rows != rows
        || requirement.cols != cols
    {
        return Err(error(CompiledStepBindFailure::WeightRequirementMismatch {
            binding: weight,
        }));
    }
    let activation_owner = buffers[activation_index].as_q8hot().ok_or_else(|| {
        error(CompiledStepBindFailure::OwnerKindMismatch {
            role: CpuBindBufferRole::Activation,
            buffer_index: activation_index,
        })
    })?;
    let expected_activation_blocks = cols / Q8K_HOT_CPU_BLOCK_LEN;
    if activation_owner.len_elements() != cols
        || activation_owner.block_count() != expected_activation_blocks
    {
        return Err(error(CompiledStepBindFailure::OwnerShapeMismatch {
            role: CpuBindBufferRole::Activation,
            buffer_index: activation_index,
            expected_len: cols,
            actual_len: activation_owner.len_elements(),
            expected_blocks: Some(expected_activation_blocks),
            actual_blocks: Some(activation_owner.block_count()),
        }));
    }
    let output_owner = buffers[output_index].as_f32().ok_or_else(|| {
        error(CompiledStepBindFailure::OwnerKindMismatch {
            role: CpuBindBufferRole::Output,
            buffer_index: output_index,
        })
    })?;
    if output.len() > output_owner.len() {
        return Err(error(CompiledStepBindFailure::BufferBounds {
            role: CpuBindBufferRole::Output,
            buffer_index: output_index,
            offset: output.offset(),
            len: output.len(),
            owner_len: output_owner.len(),
        }));
    }
    Ok(())
}

fn validate_mxfp4_f32(
    args: &crate::CompiledMxfp4F32,
    buffers: &[CpuBuffer],
    requirements: &[CpuWeightRequirement],
) -> Result<(), BindError> {
    let step = CpuBindStepKind::Mxfp4F32;
    validate_qmatmul_f32_view(
        step,
        CpuBindBufferRole::Input,
        &args.input,
        args.cols,
        buffers,
    )?;
    validate_qmatmul_f32_view(
        step,
        CpuBindBufferRole::Output,
        &args.output,
        args.rows,
        buffers,
    )?;
    if args.rows == 0
        || args.cols == 0
        || !args
            .cols
            .is_multiple_of(augur_kernel_api::MXFP4_HOT_CPU_BLOCK_LEN)
    {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::InvalidDimensions {
                rows: args.rows,
                cols: args.cols,
                blocks_per_row: args.cols / augur_kernel_api::MXFP4_HOT_CPU_BLOCK_LEN,
            },
        ));
    }
    if args.input.buffer_index() == args.output.buffer_index() {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::AliasedBuffers {
                read: args.input.buffer_index(),
                write: args.output.buffer_index(),
            },
        ));
    }
    let requirement = requirements
        .get(args.weight.index())
        .ok_or(BindError::InvalidBindingId {
            binding: args.weight,
        })?;
    if requirement.binding != args.weight
        || requirement.kind != CpuWeightResourceKind::Mxfp4HotCpuPortable
        || requirement.quant_format != QuantFormatId::MXFP4
        || requirement.physical_layout != PhysicalLayoutId::MXFP4_HOT_CPU_PORTABLE_V1
        || requirement.block_size != augur_kernel_api::MXFP4_HOT_CPU_BLOCK_LEN
        || requirement.rows != args.rows
        || requirement.cols != args.cols
    {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::WeightRequirementMismatch {
                binding: args.weight,
            },
        ));
    }
    Ok(())
}

fn validate_fp8_weight_requirement(
    binding: WeightBindingId,
    rows: usize,
    cols: usize,
    step: CpuBindStepKind,
    requirements: &[CpuWeightRequirement],
) -> Result<(), BindError> {
    let requirement = requirements
        .get(binding.index())
        .ok_or(BindError::InvalidBindingId { binding })?;
    if requirement.binding != binding
        || requirement.kind != CpuWeightResourceKind::Fp8ScaleGridHotCpuPortable
        || requirement.quant_format != QuantFormatId::FP8E4M3ScaleGrid
        || requirement.physical_layout != PhysicalLayoutId::FP8_E4M3_SCALE_GRID_HOT_CPU_PORTABLE_V1
        || requirement.block_size != augur_kernel_api::FP8_E4M3_SCALE_TILE
        || requirement.rows != rows
        || requirement.cols != cols
    {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::WeightRequirementMismatch { binding },
        ));
    }
    Ok(())
}

fn validate_fp8_scale_grid_f32(
    args: &crate::CompiledFp8ScaleGridF32,
    buffers: &[CpuBuffer],
    requirements: &[CpuWeightRequirement],
) -> Result<(), BindError> {
    let step = CpuBindStepKind::Fp8ScaleGridF32;
    validate_qmatmul_f32_view(
        step,
        CpuBindBufferRole::Input,
        &args.input,
        args.cols,
        buffers,
    )?;
    validate_qmatmul_f32_view(
        step,
        CpuBindBufferRole::Output,
        &args.output,
        args.rows,
        buffers,
    )?;
    if args.rows == 0
        || args.cols == 0
        || args.output.len() != args.rows
        || !args.cols.is_multiple_of(augur_kernel_api::FP8_E4M3_SCALE_TILE)
    {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::InvalidDimensions {
                rows: args.rows,
                cols: args.cols,
                blocks_per_row: args.cols / augur_kernel_api::FP8_E4M3_SCALE_TILE,
            },
        ));
    }
    if args.input.buffer_index() == args.output.buffer_index() {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::AliasedBuffers {
                read: args.input.buffer_index(),
                write: args.output.buffer_index(),
            },
        ));
    }
    validate_fp8_weight_requirement(args.weight, args.rows, args.cols, step, requirements)
}

fn validate_fp8_fused_mlp_f32(
    args: &crate::CompiledFp8FusedMlpF32,
    buffers: &[CpuBuffer],
    requirements: &[CpuWeightRequirement],
) -> Result<(), BindError> {
    let step = CpuBindStepKind::Fp8FusedMlpF32;
    validate_unary_elementwise(step, &args.input, &args.output, args.hidden, buffers)?;
    if args.hidden == 0
        || args.intermediate == 0
        || !args
            .hidden
            .is_multiple_of(augur_kernel_api::FP8_E4M3_SCALE_TILE)
        || !args
            .intermediate
            .is_multiple_of(augur_kernel_api::FP8_E4M3_SCALE_TILE)
        || args.config.validate().is_err()
    {
        return Err(qmatmul_error(
            step,
            CompiledStepBindFailure::InvalidDimensions {
                rows: args.hidden,
                cols: args.intermediate,
                blocks_per_row: 0,
            },
        ));
    }
    validate_fp8_weight_requirement(
        args.gate_weight,
        args.intermediate,
        args.hidden,
        step,
        requirements,
    )?;
    validate_fp8_weight_requirement(
        args.up_weight,
        args.intermediate,
        args.hidden,
        step,
        requirements,
    )?;
    validate_fp8_weight_requirement(
        args.down_weight,
        args.hidden,
        args.intermediate,
        step,
        requirements,
    )
}

// ── Phase 3: mechanical step conversion ────────────────────────────────────

/// Converts a pre-validated compiled step into an executable planned op.
///
/// All buffer metadata, owner kinds, dimensions, aliasing, and
/// step-to-requirement links were already validated by `validate_compiled_steps`.
/// Requirement-to-owner shape was already validated by `bind_cpu_weight_resources`.
///
/// This function performs only:
///
/// - typed owner lookup (defensive `Option → BindError`);
/// - `Arc::clone`;
/// - struct construction.
///
/// It does not repeat shape, layout, format, block-count, CSR, buffer-metadata
/// or alias validation.
fn bind_prevalidated_step(
    step: CpuCompiledStep,
    bound_weights: &CpuBoundWeights,
) -> Result<PlannedOp<CpuBackend>, BindError> {
    match step {
        CpuCompiledStep::Add(add) => Ok(PlannedOp::AddContiguous {
            dst: add.dst,
            a: add.a,
            b: add.b,
            n: add.n,
        }),
        CpuCompiledStep::Mul(args) => Ok(PlannedOp::MulContiguous {
            dst: args.dst,
            a: args.a,
            b: args.b,
            n: args.n,
        }),
        CpuCompiledStep::Clamp(args) => Ok(PlannedOp::ClampContiguous {
            dst: args.dst,
            input: args.input,
            n: args.n,
            min: args.min,
            max: args.max,
        }),
        CpuCompiledStep::Silu(args) => Ok(PlannedOp::SiluContiguous {
            dst: args.dst,
            input: args.input,
            n: args.n,
        }),

        CpuCompiledStep::QuantizeQ8Hot(args) => Ok(PlannedOp::QuantizeQ8Hot(
            PlannedQuantizeQ8HotArgs::new(args.input, args.output, args.launch),
        )),

        CpuCompiledStep::QMatmulF32(args) => {
            let base = bind_qmatmul_base(args.base, bound_weights)?;
            Ok(PlannedOp::QMatmulF32(PlannedQMatmulF32Args::new(
                base,
                args.launch,
            )))
        }

        CpuCompiledStep::QMatmulF32Prefill(args) => {
            let owner = q4k_owner(bound_weights, args.weight)?;
            Ok(PlannedOp::QMatmulF32Prefill(
                PlannedQMatmulF32PrefillArgs::new(
                    Arc::clone(owner),
                    args.input,
                    args.output,
                    args.m,
                    args.rows,
                    args.cols,
                    args.launch,
                ),
            ))
        }

        CpuCompiledStep::QMatmulBiasF32(args) => {
            let base = bind_qmatmul_base(args.base, bound_weights)?;
            Ok(PlannedOp::QMatmulBiasF32(PlannedQMatmulBiasF32Args::new(
                base,
                args.bias,
                args.launch,
            )))
        }

        CpuCompiledStep::QMatmulBiasSiluF32(args) => {
            let base = bind_qmatmul_base(args.base, bound_weights)?;
            Ok(PlannedOp::QMatmulBiasSiluF32(
                PlannedQMatmulBiasF32Args::new(base, args.bias, args.launch),
            ))
        }

        CpuCompiledStep::DualScaleQ8Hot(args) => {
            let owner = bound_weights
                .dual_scale_hot_cpu_portable(args.weight)
                .ok_or(BindError::InvalidBindingId {
                    binding: args.weight,
                })?;

            let bound = BoundDualScaleQ8Hot::try_new(Arc::clone(owner)).map_err(|_| {
                BindError::InvalidBindingId {
                    binding: args.weight,
                }
            })?;
            let launch =
                prevalidated_dual_scale_launch(args.launch).ok_or(BindError::InvalidBindingId {
                    binding: args.weight,
                })?;

            Ok(PlannedOp::DualScaleQ8Hot(PlannedDualScaleQ8HotArgs::new(
                bound,
                args.activation,
                args.output,
                args.rows,
                args.cols,
                args.blocks_per_row,
                launch,
            )))
        }

        CpuCompiledStep::Q8_0Q8Hot(args) => {
            let owner = bound_weights.q8_0_hot_cpu_portable(args.weight).ok_or(
                BindError::InvalidBindingId {
                    binding: args.weight,
                },
            )?;
            Ok(PlannedOp::Q8_0Q8Hot(PlannedQ8_0Q8HotArgs::new(
                Arc::clone(owner),
                args.activation,
                args.output,
                args.rows,
                args.cols,
                args.launch,
            )))
        }

        CpuCompiledStep::Q2KQ8Hot(args) => {
            let owner = bound_weights.q2k_hot_cpu_portable(args.weight).ok_or(
                BindError::InvalidBindingId {
                    binding: args.weight,
                },
            )?;
            Ok(PlannedOp::Q2KQ8Hot(PlannedQ2KQ8HotArgs::new(
                Arc::clone(owner),
                args.activation,
                args.output,
                args.rows,
                args.cols,
                args.launch,
            )))
        }

        CpuCompiledStep::Iq2XxsQ8Hot(args) => {
            let owner = bound_weights.iq2_xxs_hot_cpu_portable(args.weight).ok_or(
                BindError::InvalidBindingId {
                    binding: args.weight,
                },
            )?;
            Ok(PlannedOp::Iq2XxsQ8Hot(PlannedIq2XxsQ8HotArgs::new(
                Arc::clone(owner),
                args.activation,
                args.output,
                args.rows,
                args.cols,
                args.launch,
            )))
        }

        CpuCompiledStep::Mxfp4F32(args) => {
            let owner = bound_weights.mxfp4_hot_cpu_portable(args.weight).ok_or(
                BindError::InvalidBindingId {
                    binding: args.weight,
                },
            )?;
            Ok(PlannedOp::Mxfp4F32(PlannedMxfp4F32Args::new(
                Arc::clone(owner),
                args.input,
                args.output,
                args.rows,
                args.cols,
                args.launch,
            )))
        }
        CpuCompiledStep::Fp8ScaleGridF32(args) => {
            let owner = bound_weights
                .fp8_scale_grid_hot_cpu_portable(args.weight)
                .ok_or(BindError::InvalidBindingId {
                    binding: args.weight,
                })?;
            Ok(PlannedOp::Fp8ScaleGridF32(PlannedFp8ScaleGridF32Args::new(
                Arc::clone(owner),
                args.input,
                args.output,
                args.rows,
                args.cols,
                args.launch,
            )))
        }
        CpuCompiledStep::Fp8FusedMlpF32(args) => {
            let gate = bound_weights
                .fp8_scale_grid_hot_cpu_portable(args.gate_weight)
                .ok_or(BindError::InvalidBindingId {
                    binding: args.gate_weight,
                })?;
            let up = bound_weights
                .fp8_scale_grid_hot_cpu_portable(args.up_weight)
                .ok_or(BindError::InvalidBindingId {
                    binding: args.up_weight,
                })?;
            let down = bound_weights
                .fp8_scale_grid_hot_cpu_portable(args.down_weight)
                .ok_or(BindError::InvalidBindingId {
                    binding: args.down_weight,
                })?;
            let workspace = Fp8FusedMlpF32Workspace::new(args.intermediate).map_err(|_| {
                BindError::CompiledStep {
                    step: CpuBindStepKind::Fp8FusedMlpF32,
                    failure: CompiledStepBindFailure::InvalidDimensions {
                        rows: args.hidden,
                        cols: args.intermediate,
                        blocks_per_row: 0,
                    },
                }
            })?;
            Ok(PlannedOp::Fp8FusedMlpF32(PlannedFp8FusedMlpF32Args::new(
                Arc::clone(gate),
                Arc::clone(up),
                Arc::clone(down),
                args.input,
                args.output,
                args.hidden,
                args.intermediate,
                args.config,
                workspace,
                args.launch,
            )))
        }
    }
}

fn prevalidated_dual_scale_launch(
    launch: DualScaleQ8HotCpuLaunch,
) -> Option<DualScaleQ8HotCpuPrevalidatedLaunch> {
    if std::ptr::fn_addr_eq(
        launch,
        launch_qmatvec_dual_scale_q8k_hot_cpu_portable as DualScaleQ8HotCpuLaunch,
    ) {
        return Some(launch_qmatvec_dual_scale_q8k_hot_cpu_portable_prevalidated);
    }
    #[cfg(target_arch = "x86_64")]
    if std::ptr::fn_addr_eq(
        launch,
        launch_qmatvec_dual_scale_q8k_hot_cpu_avx2 as DualScaleQ8HotCpuLaunch,
    ) {
        return Some(launch_qmatvec_dual_scale_q8k_hot_cpu_avx2_prevalidated);
    }
    None
}

fn bind_qmatmul_base(
    base: CompiledQMatmulF32Base,
    bound_weights: &CpuBoundWeights,
) -> Result<PlannedQMatmulF32Base<CpuBackend>, BindError> {
    let owner = q4k_owner(bound_weights, base.weight)?;
    if owner.rows() != base.rows || owner.cols() != base.cols {
        return Err(BindError::CompiledStepShapeMismatch {
            binding: base.weight,
            expected_rows: base.rows,
            expected_cols: base.cols,
            actual_rows: owner.rows(),
            actual_cols: owner.cols(),
        });
    }
    Ok(PlannedQMatmulF32Base::new(
        Arc::clone(owner),
        base.input,
        base.output,
        base.rows,
        base.cols,
    ))
}

fn q4k_owner(
    bound_weights: &CpuBoundWeights,
    binding: WeightBindingId,
) -> Result<&Arc<Q4KHotCpuTensor>, BindError> {
    bound_weights
        .q4k_hot_cpu_portable(binding)
        .ok_or(BindError::InvalidBindingId { binding })
}

// ── C3 QuantizeQ8Hot binding validation ────────────────────────────────────

fn validate_quantize_q8hot(
    args: &CompiledQuantizeQ8Hot,
    buffers: &[CpuBuffer],
) -> Result<(), BindError> {
    let input_index = args.input.buffer_index();
    let output_index = args.output.buffer_index();
    let buffer_count = buffers.len();

    if input_index >= buffer_count {
        return Err(BindError::QuantizeBufferIndexOutOfRange {
            index: input_index,
            buffer_count,
        });
    }
    if output_index >= buffer_count {
        return Err(BindError::QuantizeBufferIndexOutOfRange {
            index: output_index,
            buffer_count,
        });
    }
    if input_index == output_index {
        return Err(BindError::QuantizeAliasedBuffers {
            buffer_index: input_index,
        });
    }

    if !args.input.is_cpu_contiguous_f32() {
        return Err(BindError::QuantizeInputMetadataMismatch {
            buffer_index: input_index,
        });
    }
    if !args.output.is_cpu_q8hot() {
        return Err(BindError::QuantizeOutputMetadataMismatch {
            buffer_index: output_index,
        });
    }
    if args.output.offset() != 0 {
        return Err(BindError::QuantizeOutputOffsetUnsupported {
            actual: args.output.offset(),
        });
    }
    if args.input.len() != args.output.len() {
        return Err(BindError::QuantizeLengthMismatch {
            input: args.input.len(),
            output: args.output.len(),
        });
    }
    if args.output.is_empty() || !args.output.len().is_multiple_of(Q8K_HOT_CPU_BLOCK_LEN) {
        return Err(BindError::QuantizeOutputShapeInvalid {
            len_elements: args.output.len(),
        });
    }

    let input_owner =
        buffers[input_index]
            .as_f32()
            .ok_or(BindError::QuantizeInputOwnerMismatch {
                buffer_index: input_index,
            })?;
    let bounds_err = || BindError::QuantizeInputBounds {
        buffer_index: input_index,
        offset: args.input.offset(),
        len: args.input.len(),
        owner_len: input_owner.len(),
    };
    let input_end = args
        .input
        .offset()
        .checked_add(args.input.len())
        .ok_or_else(bounds_err)?;
    if input_end > input_owner.len() {
        return Err(bounds_err());
    }

    let output_owner =
        buffers[output_index]
            .as_q8hot()
            .ok_or(BindError::QuantizeOutputOwnerMismatch {
                buffer_index: output_index,
            })?;
    let expected_blocks = args.output.len() / Q8K_HOT_CPU_BLOCK_LEN;
    if output_owner.len_elements() != args.output.len()
        || output_owner.block_count() != expected_blocks
    {
        return Err(BindError::QuantizeOutputOwnerShapeMismatch {
            buffer_index: output_index,
            view_len: args.output.len(),
            owner_len: output_owner.len_elements(),
            owner_blocks: output_owner.block_count(),
        });
    }

    Ok(())
}

// ── E3 synthetic plan tests ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use augur_backend::CpuStream;
    use augur_core::F16Bits;
    use augur_formats::{quantize_q8k_activation_vector_ref, Q8K_BLOCK_SIZE};
    use augur_kernel_api::{
        DualScaleHotCpuBlock, DualScaleHotCpuTensor, OutlierCsrHotCpu, QuantizeQ8HotCpuLaunch,
        DUAL_SCALE_HOT_CPU_GROUPS, DUAL_SCALE_HOT_CPU_MEMBERSHIP_BYTES,
        DUAL_SCALE_HOT_CPU_QS_BYTES,
    };
    use augur_kernels::portable::{
        launch_qmatvec_dual_scale_q8k_hot_cpu_portable,
        launch_qmatvec_dual_scale_q8k_hot_cpu_portable_prevalidated,
        launch_quantize_q8k_hot_cpu_portable,
    };
    use augur_repack::repack_q8k_activation_vector_cpu_ref;

    use super::*;
    use crate::{CompiledDualScaleQ8Hot, CpuCompiledPlan, CpuCompiledStep};

    fn synthetic_quantize_plan(len: usize, reverse: bool) -> CpuCompiledPlan {
        let input = vec![0.25_f32; len];
        let input_buffer = CpuBuffer::from_f32(input);
        let output_buffer = CpuBuffer::new_q8hot_zeroed(len).unwrap();
        let buffers = if reverse {
            vec![output_buffer, input_buffer]
        } else {
            vec![input_buffer, output_buffer]
        };

        let input_index = if reverse { 1 } else { 0 };
        let output_index = if reverse { 0 } else { 1 };

        let step = CpuCompiledStep::QuantizeQ8Hot(CompiledQuantizeQ8Hot {
            input: augur_exec::BufferView::<CpuBackend>::dense_f32(input_index, len),
            output: augur_exec::BufferView::<CpuBackend>::q8hot_cpu(output_index, len),
            launch: launch_quantize_q8k_hot_cpu_portable as QuantizeQ8HotCpuLaunch,
        });

        let mut graph = augur_graph::Graph::new();
        let desc = augur_core::TensorDesc::f32_contiguous(augur_core::Shape::from_dims(&[4]));
        let a = graph.add_input(desc.clone());
        let b = graph.add_input(desc.clone());
        let output = graph.add_add(a, b, desc);
        graph.try_set_output(output).unwrap();

        let mut compiled = crate::GraphCompiler::cpu_baseline()
            .compile_cpu_plan(&graph, vec![vec![0.0_f32; 4], vec![0.0_f32; 4]])
            .unwrap();

        compiled.steps = vec![step].into_boxed_slice();
        compiled.buffers = buffers;
        compiled
    }

    fn dual_scale_tensor(
        rows: usize,
        cols: usize,
        outlier_col: Option<usize>,
    ) -> DualScaleHotCpuTensor {
        let blocks_per_row = cols / DualScaleHotCpuBlock::DECODED_LEN;
        let block = DualScaleHotCpuBlock {
            scale_lo: [F16Bits::from_f32(1.0); DUAL_SCALE_HOT_CPU_GROUPS],
            scale_hi: [F16Bits::ZERO; DUAL_SCALE_HOT_CPU_GROUPS],
            membership: [0; DUAL_SCALE_HOT_CPU_MEMBERSHIP_BYTES],
            qs: [0; DUAL_SCALE_HOT_CPU_QS_BYTES],
        };

        let (row_ptr, col_idx, delta) = if let Some(col) = outlier_col {
            (
                vec![0, 1],
                vec![u32::try_from(col).unwrap()],
                vec![F16Bits::from_f32(3.0)],
            )
        } else {
            (vec![0, 0], Vec::new(), Vec::new())
        };

        let outliers = OutlierCsrHotCpu::new(rows, cols, row_ptr, col_idx, delta).unwrap();
        let blocks = vec![block; rows * blocks_per_row];
        DualScaleHotCpuTensor::new(rows, cols, blocks, outliers).unwrap()
    }

    fn synthetic_dual_scale_plan(
        rows: usize,
        cols: usize,
        _outlier_col: Option<usize>,
    ) -> (CpuCompiledPlan, usize, usize, usize) {
        let blocks_per_row = cols / DualScaleHotCpuBlock::DECODED_LEN;

        // Buffers: 0=f32 input, 1=Q8Hot, 2=f32 output
        let input = vec![0.25_f32; cols];
        let input_buffer = CpuBuffer::from_f32(input);
        let q8hot_buffer = CpuBuffer::new_q8hot_zeroed(cols).unwrap();
        let output_buffer = CpuBuffer::zeros_f32(rows);

        let q8hot_step = CpuCompiledStep::QuantizeQ8Hot(CompiledQuantizeQ8Hot {
            input: augur_exec::BufferView::<CpuBackend>::dense_f32(0, cols),
            output: augur_exec::BufferView::<CpuBackend>::q8hot_cpu(1, cols),
            launch: launch_quantize_q8k_hot_cpu_portable as QuantizeQ8HotCpuLaunch,
        });

        let dual_scale_step = CpuCompiledStep::DualScaleQ8Hot(CompiledDualScaleQ8Hot {
            weight: WeightBindingId::test(0),
            activation: augur_exec::BufferView::<CpuBackend>::q8hot_cpu(1, cols),
            output: augur_exec::BufferView::<CpuBackend>::dense_f32(2, rows),
            rows,
            cols,
            blocks_per_row,
            launch: launch_qmatvec_dual_scale_q8k_hot_cpu_portable,
        });

        let mut graph = augur_graph::Graph::new();
        let desc = augur_core::TensorDesc::f32_contiguous(augur_core::Shape::from_dims(&[4]));
        let a = graph.add_input(desc.clone());
        let b = graph.add_input(desc.clone());
        let output = graph.add_add(a, b, desc);
        graph.try_set_output(output).unwrap();

        let mut compiled = crate::GraphCompiler::cpu_baseline()
            .compile_cpu_plan(&graph, vec![vec![0.0_f32; 4], vec![0.0_f32; 4]])
            .unwrap();

        let mut builder = crate::WeightRequirementBuilder::new();
        let _binding = builder
            .intern(crate::dual_scale_requirement(
                augur_core::StorageId {
                    index: 0,
                    generation: 0,
                },
                augur_core::FormatVersion(1),
                DualScaleHotCpuBlock::DECODED_LEN,
                rows,
                cols,
            ))
            .unwrap();

        compiled.steps = vec![q8hot_step, dual_scale_step].into_boxed_slice();
        compiled.buffers = vec![input_buffer, q8hot_buffer, output_buffer];
        compiled.weight_requirements = builder.finish();
        (compiled, 0, 1, 2)
    }

    // ── C3 regression ──────────────────────────────────────────────────────

    #[test]
    fn binding_preserves_quantize_launch_identity() {
        let compiled = synthetic_quantize_plan(Q8K_BLOCK_SIZE, false);
        let executable = bind_cpu_plan(compiled, CpuWeightResources::new()).unwrap();

        let actual = match &executable.steps()[0] {
            PlannedOp::QuantizeQ8Hot(args) => args.launch(),
            other => panic!("unexpected step: {other:?}"),
        };
        let expected = launch_quantize_q8k_hot_cpu_portable as QuantizeQ8HotCpuLaunch;
        let actual_addr = actual as *const ();
        let expected_addr = expected as *const ();
        assert!(std::ptr::addr_eq(actual_addr, expected_addr));
    }

    #[test]
    fn quantize_forward_buffer_order_bit_exact() {
        let len = Q8K_BLOCK_SIZE * 2;
        let input = (0..len)
            .map(|i| ((i % 127) as f32 - 63.0) / 64.0)
            .collect::<Vec<_>>();

        let mut compiled = synthetic_quantize_plan(len, false);
        compiled.buffers[0] = CpuBuffer::from_f32(input.clone());

        let mut executable = bind_cpu_plan(compiled, CpuWeightResources::new()).unwrap();
        let mut stream = CpuStream;
        executable.execute_cpu(&mut stream).unwrap();

        let output = executable.buffers()[1].as_q8hot().unwrap();
        let reference = quantize_q8k_activation_vector_ref(&input).unwrap();
        let expected = repack_q8k_activation_vector_cpu_ref(&reference).unwrap();
        assert_eq!(output.blocks(), expected.blocks());
    }

    #[test]
    fn quantize_reverse_buffer_order_bit_exact() {
        let len = Q8K_BLOCK_SIZE;
        let input = (0..len)
            .map(|i| ((i % 127) as f32 - 63.0) / 64.0)
            .collect::<Vec<_>>();

        let mut compiled = synthetic_quantize_plan(len, true);
        compiled.buffers[1] = CpuBuffer::from_f32(input.clone());

        let mut executable = bind_cpu_plan(compiled, CpuWeightResources::new()).unwrap();
        let mut stream = CpuStream;
        executable.execute_cpu(&mut stream).unwrap();

        let output = executable.buffers()[0].as_q8hot().unwrap();
        let reference = quantize_q8k_activation_vector_ref(&input).unwrap();
        let expected = repack_q8k_activation_vector_cpu_ref(&reference).unwrap();
        assert_eq!(output.blocks(), expected.blocks());
    }

    #[test]
    fn repeated_quantize_execution_preserves_owner() {
        let len = Q8K_BLOCK_SIZE;
        let compiled = synthetic_quantize_plan(len, false);
        let mut executable = bind_cpu_plan(compiled, CpuWeightResources::new()).unwrap();
        let mut stream = CpuStream;

        let pointer_before = executable.buffers()[1]
            .as_q8hot()
            .unwrap()
            .blocks()
            .as_ptr();
        let block_count_before = executable.buffers()[1].as_q8hot().unwrap().block_count();
        let len_before = executable.buffers()[1].as_q8hot().unwrap().len_elements();

        for iteration in 0..16 {
            let input = (0..len)
                .map(|i| {
                    let x = i.wrapping_mul(37).wrapping_add(iteration * 17) % 127;
                    (x as f32 - 63.0) / 64.0
                })
                .collect::<Vec<_>>();
            executable.set_input_f32(0, &input).unwrap();
            executable.execute_cpu(&mut stream).unwrap();

            assert_eq!(
                executable.buffers()[1]
                    .as_q8hot()
                    .unwrap()
                    .blocks()
                    .as_ptr(),
                pointer_before
            );
            assert_eq!(
                executable.buffers()[1].as_q8hot().unwrap().block_count(),
                block_count_before
            );
            assert_eq!(
                executable.buffers()[1].as_q8hot().unwrap().len_elements(),
                len_before
            );
        }
    }

    #[test]
    fn alias_is_rejected_at_binding() {
        let mut compiled = synthetic_quantize_plan(Q8K_BLOCK_SIZE, false);
        if let CpuCompiledStep::QuantizeQ8Hot(ref mut args) = compiled.steps[0] {
            args.output = augur_exec::BufferView::<CpuBackend>::q8hot_cpu(
                args.input.buffer_index(),
                args.output.len(),
            );
        }
        let err = bind_cpu_plan(compiled, CpuWeightResources::new()).unwrap_err();
        assert!(matches!(err, BindError::QuantizeAliasedBuffers { .. }));
    }

    // ── E3 DualScale binding identity ──────────────────────────────────────

    #[test]
    fn dual_scale_launch_binding_preserves_address_identity() {
        let rows = 1;
        let cols = DualScaleHotCpuBlock::DECODED_LEN;
        let (compiled, _, _, _) = synthetic_dual_scale_plan(rows, cols, None);

        let mut resources = CpuWeightResources::new();
        resources
            .add_dual_scale_hot_cpu_portable(dual_scale_tensor(rows, cols, None))
            .unwrap();

        let executable = bind_cpu_plan(compiled, resources).unwrap();

        let actual = match &executable.steps()[1] {
            PlannedOp::DualScaleQ8Hot(args) => args.launch(),
            other => panic!("unexpected step: {other:?}"),
        };
        let expected = launch_qmatvec_dual_scale_q8k_hot_cpu_portable_prevalidated;
        let actual_addr = actual as *const ();
        let expected_addr = expected as *const ();
        assert!(std::ptr::addr_eq(actual_addr, expected_addr));
    }

    // ── E3 full synthetic chain ────────────────────────────────────────────

    #[test]
    fn synthetic_quantize_dual_scale_matches_direct_kernel() {
        let rows = 1;
        let cols = DualScaleHotCpuBlock::DECODED_LEN;
        let (compiled, input_idx, _q8hot_idx, output_idx) =
            synthetic_dual_scale_plan(rows, cols, None);

        let mut resources = CpuWeightResources::new();
        resources
            .add_dual_scale_hot_cpu_portable(dual_scale_tensor(rows, cols, None))
            .unwrap();

        let mut executable = bind_cpu_plan(compiled, resources).unwrap();
        let mut stream = CpuStream;
        executable.execute_cpu(&mut stream).unwrap();

        let activation = executable.buffers()[input_idx].as_f32().unwrap();
        let got = executable.buffers()[output_idx].as_f32().unwrap();

        let weight = dual_scale_tensor(rows, cols, None);
        let q8 = quantize_q8k_activation_vector_ref(activation).unwrap();
        let q8_hot = repack_q8k_activation_vector_cpu_ref(&q8).unwrap();
        let mut expected = vec![0.0_f32; rows];
        launch_qmatvec_dual_scale_q8k_hot_cpu_portable(&weight, &q8_hot, &mut expected).unwrap();

        assert_eq!(got, expected.as_slice());
    }

    #[test]
    fn repeated_dual_scale_execution_preserves_owners() {
        let rows = 1;
        let cols = DualScaleHotCpuBlock::DECODED_LEN;
        let (compiled, _input_idx, q8hot_idx, output_idx) =
            synthetic_dual_scale_plan(rows, cols, None);

        let mut resources = CpuWeightResources::new();
        resources
            .add_dual_scale_hot_cpu_portable(dual_scale_tensor(rows, cols, None))
            .unwrap();

        let mut executable = bind_cpu_plan(compiled, resources).unwrap();
        let mut stream = CpuStream;

        let q8hot_ptr_before = executable.buffers()[q8hot_idx]
            .as_q8hot()
            .unwrap()
            .blocks()
            .as_ptr();
        let output_ptr_before = executable.buffers()[output_idx].as_f32().unwrap().as_ptr();
        let step_count = executable.steps().len();

        for _ in 0..8 {
            executable.execute_cpu(&mut stream).unwrap();

            assert_eq!(
                executable.buffers()[q8hot_idx]
                    .as_q8hot()
                    .unwrap()
                    .blocks()
                    .as_ptr(),
                q8hot_ptr_before
            );
            assert_eq!(
                executable.buffers()[output_idx].as_f32().unwrap().as_ptr(),
                output_ptr_before
            );
            assert_eq!(executable.steps().len(), step_count);
        }
    }

    // ── E3 preflight errors ────────────────────────────────────────────────

    #[test]
    fn dual_scale_alias_is_rejected_before_resource_take() {
        let rows = 1;
        let cols = DualScaleHotCpuBlock::DECODED_LEN;
        let (mut compiled, _, _, _) = synthetic_dual_scale_plan(rows, cols, None);

        // Force activation and output to share a buffer index.
        if let CpuCompiledStep::DualScaleQ8Hot(ref mut args) = compiled.steps[1] {
            args.output = augur_exec::BufferView::<CpuBackend>::dense_f32(
                args.activation.buffer_index(),
                args.rows,
            );
        }

        let mut resources = CpuWeightResources::new();
        let _storage = resources
            .add_dual_scale_hot_cpu_portable(dual_scale_tensor(rows, cols, None))
            .unwrap();

        let err = bind_cpu_plan(compiled, resources).unwrap_err();
        assert!(matches!(
            err,
            BindError::CompiledStep {
                step: CpuBindStepKind::DualScaleQ8Hot,
                failure: CompiledStepBindFailure::AliasedBuffers { .. },
            }
        ));

        // Note: bind_cpu_plan consumes resources, so we cannot assert
        // take-availability here. The preflight-then-take contract is
        // tested in E2.
    }

    #[test]
    fn dual_scale_invalid_dimensions_rejected_before_resources() {
        let rows = 1;
        let cols = DualScaleHotCpuBlock::DECODED_LEN;
        let (mut compiled, _, _, _) = synthetic_dual_scale_plan(rows, cols, None);

        // blocks_per_row × 256 != cols
        if let CpuCompiledStep::DualScaleQ8Hot(ref mut args) = compiled.steps[1] {
            args.blocks_per_row = 2;
        }

        let err = bind_cpu_plan(compiled, CpuWeightResources::new()).unwrap_err();
        assert!(matches!(
            err,
            BindError::CompiledStep {
                step: CpuBindStepKind::DualScaleQ8Hot,
                failure: CompiledStepBindFailure::InvalidDimensions { .. },
            }
        ));
    }

    #[test]
    fn dual_scale_activation_not_q8hot_metadata_rejected() {
        let rows = 1;
        let cols = DualScaleHotCpuBlock::DECODED_LEN;
        let (mut compiled, _, _, _) = synthetic_dual_scale_plan(rows, cols, None);

        if let CpuCompiledStep::DualScaleQ8Hot(ref mut args) = compiled.steps[1] {
            args.activation = augur_exec::BufferView::<CpuBackend>::dense_f32(
                args.activation.buffer_index(),
                args.cols,
            );
        }

        let err = bind_cpu_plan(compiled, CpuWeightResources::new()).unwrap_err();
        assert!(matches!(
            err,
            BindError::CompiledStep {
                step: CpuBindStepKind::DualScaleQ8Hot,
                failure: CompiledStepBindFailure::MetadataMismatch { .. },
            }
        ));
    }

    // Post-binding mutation is intentionally unavailable: executable step
    // internals are private, so aliasing cannot be introduced after preflight.
}
