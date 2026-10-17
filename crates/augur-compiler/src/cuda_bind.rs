#![cfg(feature = "cuda")]

use std::collections::HashSet;

use augur_backend::{
    CudaAddF32ArgsV1, CudaF32View, CudaQ4kF32DecodeArgsV1, CudaQ4kF32PrefillArgsV1,
    CudaQ4kWeightViewV1, OwnedCudaBuffer, OwnedCudaEvent, OwnedCudaStream,
};
use augur_core::{FormatVersion, PhysicalLayoutId, QuantFormatId};
use augur_exec::{
    BufferIdentity, BufferLifetimeEnd, BufferPurpose, CudaExecutionPlan,
    CudaPlanResourceAccounting, CudaPlannedAddF32, CudaPlannedOp, CudaPlannedQ4kF32Decode,
    CudaPlannedQ4kF32Prefill, M91ArenaPlan, M91ExecutionArena, MemoryPlan,
};
use augur_kernel_api::{
    Q4KHotCudaStagingTensorV1, Q4K_HOT_CUDA_V1_BLOCK_BYTES, Q4K_HOT_CUDA_V1_DECODED_LEN,
};

use crate::{
    CudaBindError, CudaCompiledF32View, CudaCompiledPlan, CudaCompiledStep,
    CudaQ4kWeightRequirement, CudaWeightBindingId, CudaWeightResources,
};

const EVENT_COUNT: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaPlanPreflight {
    pub accounting: CudaPlanResourceAccounting,
}

#[derive(Clone, Copy, Debug)]
struct ValidatedCudaBinding {
    accounting: CudaPlanResourceAccounting,
}

pub fn preflight_cuda_plan(
    compiled: &CudaCompiledPlan,
    resources: &CudaWeightResources,
) -> Result<CudaPlanPreflight, CudaBindError> {
    let validated = preflight_cuda_binding(compiled, resources)?;
    Ok(CudaPlanPreflight {
        accounting: validated.accounting,
    })
}

pub fn bind_cuda_plan(
    compiled: CudaCompiledPlan,
    mut resources: CudaWeightResources,
) -> Result<CudaExecutionPlan, CudaBindError> {
    // This call is the global mutation boundary. No resource take, CUDA owner
    // creation, device allocation or upload may move above it.
    let validated = preflight_cuda_binding(&compiled, &resources)?;
    let host_arena = build_host_arena(&compiled.memory)?;
    let input_bytes = f32_slice_to_le_bytes(&compiled.input_f32)?;

    let mut staging_resources = Vec::with_capacity(compiled.weight_requirements.len());
    for requirement in &compiled.weight_requirements {
        staging_resources.push(resources.take(requirement.storage)?);
    }

    let stream = OwnedCudaStream::create(compiled.device_ordinal)?;
    let started_event = OwnedCudaEvent::create(compiled.device_ordinal)?;
    let stopped_event = OwnedCudaEvent::create(compiled.device_ordinal)?;
    let weight_count = compiled.weight_requirements.len();
    let mut buffers = Vec::with_capacity(validated.accounting.device_allocation_count);

    for (requirement, resource) in compiled.weight_requirements.iter().zip(&staging_resources) {
        let buffer = OwnedCudaBuffer::allocate(compiled.device_ordinal, requirement.payload_bytes)?;
        buffer.copy_from_host_sync(&stream, resource.staging.payload_bytes())?;
        buffers.push(buffer);
    }
    for slot in compiled.memory.buffers() {
        let bytes = f32_bytes(slot.len_elements)?;
        buffers.push(OwnedCudaBuffer::allocate(compiled.device_ordinal, bytes)?);
    }

    let input_owner = weight_count + compiled.input.buffer_index;
    buffers[input_owner].copy_from_host_sync(&stream, &input_bytes)?;

    let mut planned = Vec::with_capacity(compiled.steps.len());
    for step in &compiled.steps {
        planned.push(bind_step(
            step,
            &compiled.weight_requirements,
            &buffers,
            weight_count,
            &stream,
        )?);
    }

    let input_buffer_index = weight_count + compiled.input.buffer_index;
    let output_buffer_index = weight_count + compiled.output.buffer_index;
    Ok(CudaExecutionPlan::new(
        stream,
        buffers.into_boxed_slice(),
        planned.into_boxed_slice(),
        started_event,
        stopped_event,
        input_buffer_index,
        output_buffer_index,
        compiled.input.len_elements,
        compiled.output.len_elements,
        validated.accounting,
        host_arena,
        compiled.input.buffer_index,
        compiled.output.buffer_index,
    ))
}

fn build_host_arena(memory: &MemoryPlan) -> Result<M91ExecutionArena, CudaBindError> {
    const ALIGNMENT_BYTES: usize = 64;
    let buffer_count = memory.buffers().len();
    let payload_bytes = memory.buffers().iter().try_fold(0usize, |total, slot| {
        total
            .checked_add(f32_bytes(slot.len_elements)?)
            .ok_or(CudaBindError::ArithmeticOverflow)
    })?;
    let maximum_padding = buffer_count
        .checked_mul(ALIGNMENT_BYTES - 1)
        .ok_or(CudaBindError::ArithmeticOverflow)?;
    let arena_limit_bytes = payload_bytes
        .checked_add(maximum_padding)
        .ok_or(CudaBindError::ArithmeticOverflow)?;
    let bytes_per_element = vec![core::mem::size_of::<f32>(); buffer_count];
    let alignments = vec![ALIGNMENT_BYTES; buffer_count];
    let plan = M91ArenaPlan::new(memory, &bytes_per_element, &alignments, arena_limit_bytes)
        .map_err(|_| CudaBindError::InvalidMemoryPlan)?;
    Ok(M91ExecutionArena::new(plan))
}

fn preflight_cuda_binding(
    compiled: &CudaCompiledPlan,
    resources: &CudaWeightResources,
) -> Result<ValidatedCudaBinding, CudaBindError> {
    if compiled.steps.is_empty() {
        return Err(CudaBindError::UnexpectedStepCount {
            expected: 1,
            actual: 0,
        });
    }
    if resources.len() != compiled.weight_requirements.len() {
        return Err(CudaBindError::UnexpectedResourceCount {
            expected: compiled.weight_requirements.len(),
            actual: resources.len(),
        });
    }
    validate_memory_plan(&compiled.memory, compiled.input, compiled.output)?;

    let mut storage_indices = HashSet::with_capacity(compiled.weight_requirements.len());
    for (index, requirement) in compiled.weight_requirements.iter().copied().enumerate() {
        if requirement.binding.index() != index {
            return Err(CudaBindError::InvalidBindingId {
                binding: requirement.binding,
            });
        }
        if !storage_indices.insert(requirement.storage.index) {
            return Err(CudaBindError::DuplicateStorage {
                storage: requirement.storage,
            });
        }
        validate_requirement(requirement)?;
        let resource = resources.get(requirement.storage)?;
        validate_resource(requirement, &resource.staging, resource.format_version)?;
    }

    let mut used_requirements = vec![false; compiled.weight_requirements.len()];
    let mut logical_intermediate_payload_bytes = 0usize;
    for (step_index, step) in compiled.steps.iter().enumerate() {
        validate_step(
            step_index,
            step,
            &compiled.memory,
            &compiled.weight_requirements,
            &mut used_requirements,
        )?;
        let output = step_output(*step);
        if compiled.memory.buffers()[output.buffer_index].purpose == BufferPurpose::Intermediate {
            logical_intermediate_payload_bytes = logical_intermediate_payload_bytes
                .checked_add(f32_bytes(output.len_elements)?)
                .ok_or(CudaBindError::ArithmeticOverflow)?;
        }
    }
    if let Some(index) = used_requirements.iter().position(|used| !used) {
        return Err(CudaBindError::UnusedRequirement {
            binding: CudaWeightBindingId::new(index as u32),
        });
    }

    let weight_payload_bytes =
        compiled
            .weight_requirements
            .iter()
            .try_fold(0usize, |total, requirement| {
                total
                    .checked_add(requirement.payload_bytes)
                    .ok_or(CudaBindError::ArithmeticOverflow)
            })?;
    let activation_payload_bytes =
        compiled
            .memory
            .buffers()
            .iter()
            .try_fold(0usize, |total, slot| {
                total
                    .checked_add(f32_bytes(slot.len_elements)?)
                    .ok_or(CudaBindError::ArithmeticOverflow)
            })?;
    let physical_intermediate_payload_bytes = compiled
        .memory
        .buffers()
        .iter()
        .filter(|slot| slot.purpose == BufferPurpose::Intermediate)
        .try_fold(0usize, |total, slot| {
            total
                .checked_add(f32_bytes(slot.len_elements)?)
                .ok_or(CudaBindError::ArithmeticOverflow)
        })?;
    let device_payload_bytes = weight_payload_bytes
        .checked_add(activation_payload_bytes)
        .ok_or(CudaBindError::ArithmeticOverflow)?;
    let device_allocation_count = compiled
        .weight_requirements
        .len()
        .checked_add(compiled.memory.buffers().len())
        .ok_or(CudaBindError::ArithmeticOverflow)?;

    Ok(ValidatedCudaBinding {
        accounting: CudaPlanResourceAccounting {
            device_allocation_count,
            device_payload_bytes,
            weight_payload_bytes,
            input_payload_bytes: f32_bytes(compiled.input.len_elements)?,
            output_payload_bytes: f32_bytes(compiled.output.len_elements)?,
            executable_step_count: compiled.steps.len(),
            weight_allocation_count: compiled.weight_requirements.len(),
            activation_allocation_count: compiled.memory.buffers().len(),
            event_count: EVENT_COUNT,
            logical_intermediate_payload_bytes,
            physical_intermediate_payload_bytes,
            reuse_transition_count: compiled.memory.reuse_transitions().len(),
        },
    })
}

fn validate_memory_plan(
    memory: &MemoryPlan,
    input: CudaCompiledF32View,
    output: CudaCompiledF32View,
) -> Result<(), CudaBindError> {
    if memory.buffers().is_empty() {
        return Err(CudaBindError::InvalidMemoryPlan);
    }
    for (index, slot) in memory.buffers().iter().enumerate() {
        if slot.buffer_index != index
            || slot.len_elements == 0
            || slot.len_elements > u32::MAX as usize
        {
            return Err(CudaBindError::InvalidMemoryPlan);
        }
        f32_bytes(slot.len_elements)?;
    }
    validate_view(memory, input)?;
    validate_view(memory, output)?;
    if memory.buffers()[input.buffer_index].purpose != BufferPurpose::Input
        || memory.buffers()[output.buffer_index].purpose != BufferPurpose::Output
        || input.buffer_index == output.buffer_index
    {
        return Err(CudaBindError::InvalidMemoryPlan);
    }
    for transition in memory.reuse_transitions() {
        let slot = memory
            .buffers()
            .get(transition.buffer_index)
            .ok_or(CudaBindError::InvalidMemoryPlan)?;
        let (previous_end, next_first) = match (
            transition.previous_reusable_after,
            transition.next_first_use,
        ) {
            (
                BufferLifetimeEnd::GraphTopoStep(previous),
                BufferLifetimeEnd::GraphTopoStep(next),
            ) => (previous, next),
            _ => return Err(CudaBindError::InvalidMemoryPlan),
        };
        let (previous, next) = match (transition.previous, transition.next) {
            (BufferIdentity::Tensor(previous), BufferIdentity::Tensor(next)) => (previous, next),
            _ => return Err(CudaBindError::InvalidMemoryPlan),
        };
        if slot.purpose != BufferPurpose::Intermediate
            || previous_end >= next_first
            || memory.buffer_for(previous) != Some(transition.buffer_index)
            || memory.buffer_for(next) != Some(transition.buffer_index)
        {
            return Err(CudaBindError::InvalidMemoryPlan);
        }
    }
    Ok(())
}

fn validate_step(
    step_index: usize,
    step: &CudaCompiledStep,
    memory: &MemoryPlan,
    requirements: &[CudaQ4kWeightRequirement],
    used_requirements: &mut [bool],
) -> Result<(), CudaBindError> {
    match *step {
        CudaCompiledStep::Q4kF32Decode(step) => {
            validate_view(memory, step.input)?;
            validate_view(memory, step.output)?;
            validate_no_write_alias(step_index, step.input, step.output)?;
            if step.input.len_elements != step.cols || step.output.len_elements != step.rows {
                return Err(CudaBindError::InvalidStep { step: step_index });
            }
            mark_requirement(
                step_index,
                step.weight,
                step.rows,
                step.cols,
                requirements,
                used_requirements,
            )
        }
        CudaCompiledStep::Q4kF32Prefill(step) => {
            validate_view(memory, step.input)?;
            validate_view(memory, step.output)?;
            validate_no_write_alias(step_index, step.input, step.output)?;
            if step.m == 0
                || step.input.len_elements != step.m.saturating_mul(step.cols)
                || step.output.len_elements != step.m.saturating_mul(step.rows)
            {
                return Err(CudaBindError::InvalidStep { step: step_index });
            }
            mark_requirement(
                step_index,
                step.weight,
                step.rows,
                step.cols,
                requirements,
                used_requirements,
            )
        }
        CudaCompiledStep::AddF32(step) => {
            validate_view(memory, step.a)?;
            validate_view(memory, step.b)?;
            validate_view(memory, step.output)?;
            validate_no_write_alias(step_index, step.a, step.output)?;
            validate_no_write_alias(step_index, step.b, step.output)?;
            if step.elements == 0
                || step.a.len_elements != step.elements
                || step.b.len_elements != step.elements
                || step.output.len_elements != step.elements
            {
                return Err(CudaBindError::InvalidStep { step: step_index });
            }
            Ok(())
        }
    }
}

fn validate_view(memory: &MemoryPlan, view: CudaCompiledF32View) -> Result<(), CudaBindError> {
    let slot = memory
        .buffers()
        .get(view.buffer_index)
        .ok_or(CudaBindError::InvalidMemoryPlan)?;
    if view.len_elements == 0
        || view.len_elements > slot.len_elements
        || view.len_elements > u32::MAX as usize
    {
        return Err(CudaBindError::InvalidMemoryPlan);
    }
    Ok(())
}

fn validate_no_write_alias(
    step: usize,
    read: CudaCompiledF32View,
    write: CudaCompiledF32View,
) -> Result<(), CudaBindError> {
    if read.buffer_index == write.buffer_index {
        return Err(CudaBindError::AliasedBuffers {
            step,
            read: read.buffer_index,
            write: write.buffer_index,
        });
    }
    Ok(())
}

fn mark_requirement(
    step_index: usize,
    binding: CudaWeightBindingId,
    rows: usize,
    cols: usize,
    requirements: &[CudaQ4kWeightRequirement],
    used: &mut [bool],
) -> Result<(), CudaBindError> {
    let requirement = requirements
        .get(binding.index())
        .ok_or(CudaBindError::InvalidBindingId { binding })?;
    if requirement.binding != binding || requirement.rows != rows || requirement.cols != cols {
        return Err(CudaBindError::InvalidStep { step: step_index });
    }
    used[binding.index()] = true;
    Ok(())
}

fn validate_requirement(requirement: CudaQ4kWeightRequirement) -> Result<(), CudaBindError> {
    if requirement.rows == 0
        || requirement.cols == 0
        || requirement.rows > u32::MAX as usize
        || requirement.cols > u32::MAX as usize
        || !requirement.cols.is_multiple_of(Q4K_HOT_CUDA_V1_DECODED_LEN)
        || requirement.quant_format != QuantFormatId::Q4K
        || requirement.format_version != FormatVersion(1)
        || requirement.physical_layout != PhysicalLayoutId::Q4K_HOT_CUDA_V1
        || requirement.block_size != Q4K_HOT_CUDA_V1_DECODED_LEN
    {
        return Err(CudaBindError::RequirementMismatch {
            storage: requirement.storage,
        });
    }
    let expected_payload = requirement
        .rows
        .checked_mul(requirement.cols / Q4K_HOT_CUDA_V1_DECODED_LEN)
        .and_then(|count| count.checked_mul(Q4K_HOT_CUDA_V1_BLOCK_BYTES))
        .ok_or(CudaBindError::ArithmeticOverflow)?;
    if requirement.payload_bytes != expected_payload {
        return Err(CudaBindError::RequirementMismatch {
            storage: requirement.storage,
        });
    }
    Ok(())
}

fn validate_resource(
    requirement: CudaQ4kWeightRequirement,
    staging: &Q4KHotCudaStagingTensorV1,
    format_version: FormatVersion,
) -> Result<(), CudaBindError> {
    if format_version != requirement.format_version
        || staging.rows() != requirement.rows
        || staging.cols() != requirement.cols
        || staging.payload_len_bytes() != requirement.payload_bytes
    {
        return Err(CudaBindError::RequirementMismatch {
            storage: requirement.storage,
        });
    }
    Ok(())
}

fn bind_step(
    step: &CudaCompiledStep,
    requirements: &[CudaQ4kWeightRequirement],
    buffers: &[OwnedCudaBuffer],
    weight_count: usize,
    stream: &OwnedCudaStream,
) -> Result<CudaPlannedOp, CudaBindError> {
    Ok(match *step {
        CudaCompiledStep::Q4kF32Decode(step) => {
            let args = CudaQ4kF32DecodeArgsV1::bind(
                stream,
                weight_view(step.weight, requirements, buffers)?,
                f32_view(step.input, buffers, weight_count)?,
                f32_view(step.output, buffers, weight_count)?,
            )?;
            CudaPlannedOp::Q4kF32Decode(CudaPlannedQ4kF32Decode::new(args))
        }
        CudaCompiledStep::Q4kF32Prefill(step) => {
            let args = CudaQ4kF32PrefillArgsV1::bind(
                stream,
                weight_view(step.weight, requirements, buffers)?,
                f32_view(step.input, buffers, weight_count)?,
                f32_view(step.output, buffers, weight_count)?,
                step.m,
            )?;
            CudaPlannedOp::Q4kF32Prefill(CudaPlannedQ4kF32Prefill::new(args))
        }
        CudaCompiledStep::AddF32(step) => {
            let args = CudaAddF32ArgsV1::bind(
                stream,
                f32_view(step.a, buffers, weight_count)?,
                f32_view(step.b, buffers, weight_count)?,
                f32_view(step.output, buffers, weight_count)?,
            )?;
            CudaPlannedOp::AddF32(CudaPlannedAddF32::new(args))
        }
    })
}

fn weight_view(
    binding: CudaWeightBindingId,
    requirements: &[CudaQ4kWeightRequirement],
    buffers: &[OwnedCudaBuffer],
) -> Result<CudaQ4kWeightViewV1, CudaBindError> {
    let requirement = requirements
        .get(binding.index())
        .ok_or(CudaBindError::InvalidBindingId { binding })?;
    let owner = buffers
        .get(binding.index())
        .ok_or(CudaBindError::InvalidBindingId { binding })?;
    Ok(CudaQ4kWeightViewV1::new(
        owner.view(0, requirement.payload_bytes)?,
        PhysicalLayoutId::Q4K_HOT_CUDA_V1,
        requirement.rows,
        requirement.cols,
    )?)
}

fn f32_view(
    view: CudaCompiledF32View,
    buffers: &[OwnedCudaBuffer],
    weight_count: usize,
) -> Result<CudaF32View, CudaBindError> {
    let owner = buffers
        .get(weight_count + view.buffer_index)
        .ok_or(CudaBindError::InvalidMemoryPlan)?;
    let bytes = f32_bytes(view.len_elements)?;
    Ok(CudaF32View::new(
        owner.view(0, bytes)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        view.len_elements,
    )?)
}

fn step_output(step: CudaCompiledStep) -> CudaCompiledF32View {
    match step {
        CudaCompiledStep::Q4kF32Decode(step) => step.output,
        CudaCompiledStep::Q4kF32Prefill(step) => step.output,
        CudaCompiledStep::AddF32(step) => step.output,
    }
}

fn f32_bytes(elements: usize) -> Result<usize, CudaBindError> {
    elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(CudaBindError::ArithmeticOverflow)
}

fn f32_slice_to_le_bytes(values: &[f32]) -> Result<Vec<u8>, CudaBindError> {
    let mut bytes = Vec::with_capacity(f32_bytes(values.len())?);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}
