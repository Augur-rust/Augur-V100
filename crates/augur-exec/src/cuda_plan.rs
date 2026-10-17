#![cfg(feature = "cuda")]

use augur_backend::{
    CudaAddF32ArgsV1, CudaQ4kF32DecodeArgsV1, CudaQ4kF32PrefillArgsV1, CudaRuntimeError,
    CudaRuntimeErrorKind, OwnedCudaBuffer, OwnedCudaEvent, OwnedCudaStream,
};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::M91ExecutionArena;

#[derive(Clone, Copy, Debug)]
pub struct CudaPlannedQ4kF32Decode {
    args: CudaQ4kF32DecodeArgsV1,
}

impl CudaPlannedQ4kF32Decode {
    pub fn new(args: CudaQ4kF32DecodeArgsV1) -> Self {
        Self { args }
    }

    fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.args.submit_for_plan(stream)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaPlannedQ4kF32Prefill {
    args: CudaQ4kF32PrefillArgsV1,
}

impl CudaPlannedQ4kF32Prefill {
    pub fn new(args: CudaQ4kF32PrefillArgsV1) -> Self {
        Self { args }
    }

    fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.args.submit_for_plan(stream)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaPlannedAddF32 {
    args: CudaAddF32ArgsV1,
}

impl CudaPlannedAddF32 {
    pub fn new(args: CudaAddF32ArgsV1) -> Self {
        Self { args }
    }

    fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.args.submit_for_plan(stream)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum CudaPlannedOp {
    Q4kF32Decode(CudaPlannedQ4kF32Decode),
    Q4kF32Prefill(CudaPlannedQ4kF32Prefill),
    AddF32(CudaPlannedAddF32),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaPlanResourceAccounting {
    pub device_allocation_count: usize,
    pub device_payload_bytes: usize,
    pub weight_payload_bytes: usize,
    pub input_payload_bytes: usize,
    pub output_payload_bytes: usize,
    pub executable_step_count: usize,
    pub weight_allocation_count: usize,
    pub activation_allocation_count: usize,
    pub event_count: usize,
    pub logical_intermediate_payload_bytes: usize,
    pub physical_intermediate_payload_bytes: usize,
    pub reuse_transition_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaPlanTimings {
    pub kernel_event_ms: f32,
    pub launch_host: Duration,
    pub complete_host: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaHostArenaAccounting {
    pub allocation_count: usize,
    pub buffer_count: usize,
    pub payload_bytes: usize,
    pub max_alignment_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaHostArenaIdentity {
    pub address: usize,
    pub len_bytes: usize,
}

#[derive(Debug)]
pub struct CudaExecutionPlan {
    // Rust drops fields in declaration order. Keep dependent steps and buffers
    // ahead of the stream so the stream remains valid throughout teardown.
    steps: Box<[CudaPlannedOp]>,
    buffers: Box<[OwnedCudaBuffer]>,
    started_event: OwnedCudaEvent,
    stopped_event: OwnedCudaEvent,
    stream: OwnedCudaStream,
    input_buffer_index: usize,
    output_buffer_index: usize,
    input_elements: usize,
    output_elements: usize,
    accounting: CudaPlanResourceAccounting,
    host_arena: Mutex<M91ExecutionArena>,
    input_staging_index: usize,
    output_staging_index: usize,
    host_arena_accounting: CudaHostArenaAccounting,
}

impl CudaExecutionPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stream: OwnedCudaStream,
        buffers: Box<[OwnedCudaBuffer]>,
        steps: Box<[CudaPlannedOp]>,
        started_event: OwnedCudaEvent,
        stopped_event: OwnedCudaEvent,
        input_buffer_index: usize,
        output_buffer_index: usize,
        input_elements: usize,
        output_elements: usize,
        accounting: CudaPlanResourceAccounting,
        host_arena: M91ExecutionArena,
        input_staging_index: usize,
        output_staging_index: usize,
    ) -> Self {
        let host_arena_accounting = CudaHostArenaAccounting {
            allocation_count: 1,
            buffer_count: host_arena.plan().bindings().len(),
            payload_bytes: host_arena.plan().total_bytes(),
            max_alignment_bytes: host_arena.plan().max_alignment_bytes(),
        };
        Self {
            stream,
            buffers,
            steps,
            started_event,
            stopped_event,
            input_buffer_index,
            output_buffer_index,
            input_elements,
            output_elements,
            accounting,
            host_arena: Mutex::new(host_arena),
            input_staging_index,
            output_staging_index,
            host_arena_accounting,
        }
    }

    pub fn execute_cuda(&self) -> Result<(), CudaRuntimeError> {
        self.submit_steps()?;
        self.stream.synchronize()
    }

    pub fn execute_cuda_profiled(&self) -> Result<CudaPlanTimings, CudaRuntimeError> {
        let complete_started = Instant::now();
        self.started_event.record(&self.stream)?;
        let launch_started = Instant::now();
        self.submit_steps()?;
        let launch_host = launch_started.elapsed();
        self.stopped_event.record(&self.stream)?;
        self.stopped_event.synchronize()?;
        let complete_host = complete_started.elapsed();
        let kernel_event_ms = self.started_event.elapsed_ms(&self.stopped_event)?;
        Ok(CudaPlanTimings {
            kernel_event_ms,
            launch_host,
            complete_host,
        })
    }

    fn submit_steps(&self) -> Result<(), CudaRuntimeError> {
        for step in &self.steps {
            match step {
                CudaPlannedOp::Q4kF32Decode(step) => {
                    step.submit(&self.stream)?;
                }
                CudaPlannedOp::Q4kF32Prefill(step) => {
                    step.submit(&self.stream)?;
                }
                CudaPlannedOp::AddF32(step) => {
                    step.submit(&self.stream)?;
                }
            }
        }
        Ok(())
    }

    pub fn set_input_f32(&self, input: &[f32]) -> Result<(), CudaRuntimeError> {
        if input.len() != self.input_elements {
            return Err(CudaRuntimeError::validation(
                "cuda_plan_set_input",
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }

        let mut arena = self.lock_host_arena("cuda_plan_set_input")?;
        let bytes = arena
            .view_mut(self.input_staging_index)
            .map_err(|_| host_arena_error("cuda_plan_set_input"))?;
        encode_f32_le(input, bytes, "cuda_plan_set_input")?;

        self.buffers[self.input_buffer_index].copy_from_host_sync(&self.stream, bytes)
    }

    pub fn read_output_f32(&self, output: &mut [f32]) -> Result<(), CudaRuntimeError> {
        if output.len() != self.output_elements {
            return Err(CudaRuntimeError::validation(
                "cuda_plan_read_output",
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        let mut arena = self.lock_host_arena("cuda_plan_read_output")?;
        let bytes = arena
            .view_mut(self.output_staging_index)
            .map_err(|_| host_arena_error("cuda_plan_read_output"))?;
        self.buffers[self.output_buffer_index].copy_to_host_sync(&self.stream, bytes)?;
        decode_f32_le(bytes, output, "cuda_plan_read_output")
    }

    pub const fn accounting(&self) -> CudaPlanResourceAccounting {
        self.accounting
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    pub const fn host_arena_accounting(&self) -> CudaHostArenaAccounting {
        self.host_arena_accounting
    }

    pub fn host_arena_identity(&self) -> Result<CudaHostArenaIdentity, CudaRuntimeError> {
        let arena = self.lock_host_arena("cuda_plan_host_arena_identity")?;
        let (address, len_bytes) = arena.allocation_identity();
        Ok(CudaHostArenaIdentity {
            address: address as usize,
            len_bytes,
        })
    }

    pub fn reset_host_arena(&self) -> Result<(), CudaRuntimeError> {
        self.lock_host_arena("cuda_plan_reset_host_arena")?.reset();
        Ok(())
    }

    fn lock_host_arena(
        &self,
        operation: &'static str,
    ) -> Result<std::sync::MutexGuard<'_, M91ExecutionArena>, CudaRuntimeError> {
        self.host_arena
            .lock()
            .map_err(|_| host_arena_error(operation))
    }
}

fn encode_f32_le(
    values: &[f32],
    bytes: &mut [u8],
    operation: &'static str,
) -> Result<(), CudaRuntimeError> {
    let len_bytes = values
        .len()
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| {
            CudaRuntimeError::validation(operation, CudaRuntimeErrorKind::RangeOverflow)
        })?;
    if bytes.len() != len_bytes {
        return Err(CudaRuntimeError::validation(
            operation,
            CudaRuntimeErrorKind::InvalidDimensions,
        ));
    }
    for (value, encoded) in values.iter().zip(bytes.chunks_exact_mut(4)) {
        encoded.copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn decode_f32_le(
    bytes: &[u8],
    output: &mut [f32],
    operation: &'static str,
) -> Result<(), CudaRuntimeError> {
    let expected = output
        .len()
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| {
            CudaRuntimeError::validation(operation, CudaRuntimeErrorKind::RangeOverflow)
        })?;
    if bytes.len() != expected {
        return Err(CudaRuntimeError::validation(
            operation,
            CudaRuntimeErrorKind::InvalidDimensions,
        ));
    }
    for (dst, encoded) in output.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = f32::from_le_bytes(encoded.try_into().expect("exact f32 byte chunk"));
    }
    Ok(())
}

fn host_arena_error(operation: &'static str) -> CudaRuntimeError {
    CudaRuntimeError::validation(operation, CudaRuntimeErrorKind::InvalidMetadata)
}
