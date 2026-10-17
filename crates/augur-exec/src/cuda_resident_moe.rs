#![cfg(feature = "cuda")]

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use augur_backend::{
    CudaResidentMoeF32ArgsV1, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaBuffer,
    OwnedCudaEvent, OwnedCudaStream,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaResidentMoePlanAccounting {
    pub device_allocation_count: usize,
    pub device_payload_bytes: usize,
    pub weight_payload_bytes: usize,
    pub input_payload_bytes: usize,
    pub output_payload_bytes: usize,
    pub workspace_payload_bytes: usize,
    pub logical_weight_count: usize,
    pub expert_count: usize,
    pub top_k: usize,
    pub tokens: usize,
    pub event_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaResidentMoeTimings {
    pub kernel_event_ms: f32,
    pub launch_host: Duration,
    pub complete_host: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaResidentMoeOwnerIdentity {
    pub weights: u64,
    pub input: u64,
    pub output: u64,
    pub router_logits: u64,
    pub selected_ids: u64,
    pub selected_weights: u64,
    pub scratch: u64,
    pub access_counts: u64,
}

#[derive(Debug)]
pub struct CudaResidentMoeExecutionPlan {
    args: CudaResidentMoeF32ArgsV1,
    weights: OwnedCudaBuffer,
    input: OwnedCudaBuffer,
    output: OwnedCudaBuffer,
    router_logits: OwnedCudaBuffer,
    selected_ids: OwnedCudaBuffer,
    selected_weights: OwnedCudaBuffer,
    scratch: OwnedCudaBuffer,
    access_counts: OwnedCudaBuffer,
    started_event: OwnedCudaEvent,
    stopped_event: OwnedCudaEvent,
    stream: OwnedCudaStream,
    input_elements: usize,
    output_elements: usize,
    router_elements: usize,
    route_elements: usize,
    access_elements: usize,
    accounting: CudaResidentMoePlanAccounting,
    host_staging: Mutex<Box<[u8]>>,
}

impl CudaResidentMoeExecutionPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        args: CudaResidentMoeF32ArgsV1,
        weights: OwnedCudaBuffer,
        input: OwnedCudaBuffer,
        output: OwnedCudaBuffer,
        router_logits: OwnedCudaBuffer,
        selected_ids: OwnedCudaBuffer,
        selected_weights: OwnedCudaBuffer,
        scratch: OwnedCudaBuffer,
        access_counts: OwnedCudaBuffer,
        started_event: OwnedCudaEvent,
        stopped_event: OwnedCudaEvent,
        stream: OwnedCudaStream,
        input_elements: usize,
        router_elements: usize,
        route_elements: usize,
        access_elements: usize,
        accounting: CudaResidentMoePlanAccounting,
    ) -> Result<Self, CudaRuntimeError> {
        let staging_elements = input_elements
            .max(router_elements)
            .max(route_elements)
            .max(access_elements);
        let staging_bytes = staging_elements
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaRuntimeError::validation(
                    "cuda_resident_moe_host_staging",
                    CudaRuntimeErrorKind::RangeOverflow,
                )
            })?;
        let mut host_staging = Vec::new();
        host_staging.try_reserve_exact(staging_bytes).map_err(|_| {
            CudaRuntimeError::validation(
                "cuda_resident_moe_host_staging",
                CudaRuntimeErrorKind::InvalidMetadata,
            )
        })?;
        host_staging.resize(staging_bytes, 0);
        Ok(Self {
            args,
            weights,
            input,
            output,
            router_logits,
            selected_ids,
            selected_weights,
            scratch,
            access_counts,
            started_event,
            stopped_event,
            stream,
            input_elements,
            output_elements: input_elements,
            router_elements,
            route_elements,
            access_elements,
            accounting,
            host_staging: Mutex::new(host_staging.into_boxed_slice()),
        })
    }

    pub fn execute_cuda(&self) -> Result<(), CudaRuntimeError> {
        self.args.launch(&self.stream)
    }

    pub fn execute_cuda_profiled(&self) -> Result<CudaResidentMoeTimings, CudaRuntimeError> {
        let complete_started = Instant::now();
        self.started_event.record(&self.stream)?;
        let launch_started = Instant::now();
        self.args.submit_for_plan(&self.stream)?;
        let launch_host = launch_started.elapsed();
        self.stopped_event.record(&self.stream)?;
        self.stopped_event.synchronize()?;
        let complete_host = complete_started.elapsed();
        Ok(CudaResidentMoeTimings {
            kernel_event_ms: self.started_event.elapsed_ms(&self.stopped_event)?,
            launch_host,
            complete_host,
        })
    }

    pub fn set_input_f32(&self, input: &[f32]) -> Result<(), CudaRuntimeError> {
        if input.len() != self.input_elements {
            return Err(validation("cuda_resident_moe_set_input"));
        }
        let mut staging = self.lock_host_staging("cuda_resident_moe_set_input")?;
        let bytes = &mut staging[..core::mem::size_of_val(input)];
        encode_f32(input, bytes);
        self.input.copy_from_host_sync(&self.stream, bytes)
    }

    pub fn read_output_f32(&self, output: &mut [f32]) -> Result<(), CudaRuntimeError> {
        self.read_f32(&self.output, self.output_elements, output)
    }

    pub fn read_router_logits(&self, output: &mut [f32]) -> Result<(), CudaRuntimeError> {
        self.read_f32(&self.router_logits, self.router_elements, output)
    }

    pub fn read_selected_ids(&self, output: &mut [u32]) -> Result<(), CudaRuntimeError> {
        self.read_u32(&self.selected_ids, self.route_elements, output)
    }

    pub fn read_selected_weights(&self, output: &mut [f32]) -> Result<(), CudaRuntimeError> {
        self.read_f32(&self.selected_weights, self.route_elements, output)
    }

    pub fn read_access_counts(&self, output: &mut [u32]) -> Result<(), CudaRuntimeError> {
        self.read_u32(&self.access_counts, self.access_elements, output)
    }

    pub const fn accounting(&self) -> CudaResidentMoePlanAccounting {
        self.accounting
    }

    pub fn owner_identity(&self) -> CudaResidentMoeOwnerIdentity {
        CudaResidentMoeOwnerIdentity {
            weights: address(&self.weights),
            input: address(&self.input),
            output: address(&self.output),
            router_logits: address(&self.router_logits),
            selected_ids: address(&self.selected_ids),
            selected_weights: address(&self.selected_weights),
            scratch: address(&self.scratch),
            access_counts: address(&self.access_counts),
        }
    }

    fn read_f32(
        &self,
        buffer: &OwnedCudaBuffer,
        expected: usize,
        output: &mut [f32],
    ) -> Result<(), CudaRuntimeError> {
        if output.len() != expected {
            return Err(validation("cuda_resident_moe_read_f32"));
        }
        let mut staging = self.lock_host_staging("cuda_resident_moe_read_f32")?;
        let bytes = &mut staging[..expected * core::mem::size_of::<f32>()];
        buffer.copy_to_host_sync(&self.stream, bytes)?;
        decode_f32(bytes, output);
        Ok(())
    }

    fn read_u32(
        &self,
        buffer: &OwnedCudaBuffer,
        expected: usize,
        output: &mut [u32],
    ) -> Result<(), CudaRuntimeError> {
        if output.len() != expected {
            return Err(validation("cuda_resident_moe_read_u32"));
        }
        let mut staging = self.lock_host_staging("cuda_resident_moe_read_u32")?;
        let bytes = &mut staging[..expected * core::mem::size_of::<u32>()];
        buffer.copy_to_host_sync(&self.stream, bytes)?;
        decode_u32(bytes, output);
        Ok(())
    }

    fn lock_host_staging(
        &self,
        operation: &'static str,
    ) -> Result<MutexGuard<'_, Box<[u8]>>, CudaRuntimeError> {
        self.host_staging.lock().map_err(|_| {
            CudaRuntimeError::validation(operation, CudaRuntimeErrorKind::InvalidMetadata)
        })
    }
}

fn address(buffer: &OwnedCudaBuffer) -> u64 {
    buffer.device_ptr().map(|ptr| ptr.raw_value()).unwrap_or(0)
}

fn validation(operation: &'static str) -> CudaRuntimeError {
    CudaRuntimeError::validation(operation, CudaRuntimeErrorKind::InvalidDimensions)
}

fn encode_f32(values: &[f32], bytes: &mut [u8]) {
    debug_assert_eq!(bytes.len(), core::mem::size_of_val(values));
    for (value, encoded) in values.iter().zip(bytes.chunks_exact_mut(4)) {
        encoded.copy_from_slice(&value.to_le_bytes());
    }
}

fn decode_f32(bytes: &[u8], output: &mut [f32]) {
    debug_assert_eq!(bytes.len(), core::mem::size_of_val(output));
    for (dst, src) in output.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = f32::from_le_bytes(src.try_into().expect("f32 bytes"));
    }
}

fn decode_u32(bytes: &[u8], output: &mut [u32]) {
    debug_assert_eq!(bytes.len(), core::mem::size_of_val(output));
    for (dst, src) in output.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = u32::from_le_bytes(src.try_into().expect("u32 bytes"));
    }
}
