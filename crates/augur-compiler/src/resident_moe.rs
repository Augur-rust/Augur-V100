use std::sync::Arc;

use augur_core::{PhysicalLayoutId, RuntimeError, StorageId};
use augur_exec::{ResidentMoeExecutionPlan, ResidentMoePlanAccounting};
use augur_graph::{
    ResidentMoeAnalysis, ResidentMoeExpertStorages, ResidentMoeGraph, ResidentMoeTensorRole,
    ResidentMoeValidationError, ResidentMoeWeightStorages,
};
use augur_kernel_api::{ResidentMoeF32CpuLaunch, ResidentMoeF32HotCpu};

#[cfg(feature = "cuda")]
use augur_backend::{
    cuda_device_properties, CudaComputeCapability, CudaResidentMoeF32ArgsV1, CudaRuntimeError,
    OwnedCudaBuffer, OwnedCudaEvent, OwnedCudaStream,
};
#[cfg(feature = "cuda")]
use augur_exec::{CudaResidentMoeExecutionPlan, CudaResidentMoePlanAccounting};

#[derive(Clone, Debug)]
pub struct CpuResidentMoeCompiledPlan {
    analysis: ResidentMoeAnalysis,
    launch: ResidentMoeF32CpuLaunch,
    physical_layout: PhysicalLayoutId,
}

impl CpuResidentMoeCompiledPlan {
    pub fn analysis(&self) -> &ResidentMoeAnalysis {
        &self.analysis
    }
    pub const fn launch(&self) -> ResidentMoeF32CpuLaunch {
        self.launch
    }
    pub const fn physical_layout(&self) -> PhysicalLayoutId {
        self.physical_layout
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResidentMoeCompileError {
    Validation(ResidentMoeValidationError),
}

impl From<ResidentMoeValidationError> for ResidentMoeCompileError {
    fn from(value: ResidentMoeValidationError) -> Self {
        Self::Validation(value)
    }
}

#[derive(Clone, Debug, Default)]
pub struct CpuResidentMoeCompiler;

impl CpuResidentMoeCompiler {
    pub fn compile(
        &self,
        graph: &ResidentMoeGraph,
    ) -> Result<CpuResidentMoeCompiledPlan, ResidentMoeCompileError> {
        Ok(CpuResidentMoeCompiledPlan {
            analysis: graph.analyze()?,
            launch: augur_kernels::portable::launch_resident_moe_f32_cpu_portable,
            physical_layout: ResidentMoeF32HotCpu::PHYSICAL_LAYOUT,
        })
    }
}

#[derive(Debug)]
pub struct CpuResidentMoeResource {
    pub weight_storages: ResidentMoeWeightStorages,
    pub weights: ResidentMoeF32HotCpu,
}

#[derive(Debug, Default)]
pub struct CpuResidentMoeResources {
    resource: Option<CpuResidentMoeResource>,
}

impl CpuResidentMoeResources {
    pub fn new(resource: CpuResidentMoeResource) -> Self {
        Self {
            resource: Some(resource),
        }
    }

    pub fn is_available(&self) -> bool {
        self.resource.is_some()
    }

    pub fn get(&self) -> Option<&CpuResidentMoeResource> {
        self.resource.as_ref()
    }

    fn take(&mut self) -> Result<CpuResidentMoeResource, ResidentMoeBindError> {
        self.resource
            .take()
            .ok_or(ResidentMoeBindError::ResourceConsumed)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResidentMoeBindError {
    MissingResource,
    ResourceConsumed,
    StorageMismatch {
        role: ResidentMoeTensorRole,
        required: StorageId,
        actual: StorageId,
    },
    GenerationMismatch {
        role: ResidentMoeTensorRole,
        required: StorageId,
        actual: StorageId,
    },
    ExpertCountMismatch {
        expected: usize,
        actual: usize,
    },
    WeightShapeMismatch,
    WeightPayloadMismatch {
        expected: usize,
        actual: usize,
    },
    PhysicalLayoutMismatch,
    ArithmeticOverflow,
    Runtime(RuntimeError),
    #[cfg(feature = "cuda")]
    CudaRuntime(CudaRuntimeError),
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct CudaResidentMoeCompiledPlan {
    analysis: ResidentMoeAnalysis,
    device_ordinal: i32,
}

#[cfg(feature = "cuda")]
impl CudaResidentMoeCompiledPlan {
    pub fn analysis(&self) -> &ResidentMoeAnalysis {
        &self.analysis
    }
    pub const fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaResidentMoeCompileError {
    Validation(ResidentMoeValidationError),
    Runtime(CudaRuntimeError),
    UnsupportedCapability,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct CudaResidentMoeCompiler {
    device_ordinal: i32,
}

#[cfg(feature = "cuda")]
impl CudaResidentMoeCompiler {
    pub fn new(device_ordinal: i32) -> Result<Self, CudaResidentMoeCompileError> {
        let properties =
            cuda_device_properties(device_ordinal).map_err(CudaResidentMoeCompileError::Runtime)?;
        if !properties
            .compute_capability
            .is_at_least(CudaComputeCapability::SM70)
        {
            return Err(CudaResidentMoeCompileError::UnsupportedCapability);
        }
        Ok(Self { device_ordinal })
    }

    pub fn compile(
        &self,
        graph: &ResidentMoeGraph,
    ) -> Result<CudaResidentMoeCompiledPlan, CudaResidentMoeCompileError> {
        Ok(CudaResidentMoeCompiledPlan {
            analysis: graph
                .analyze()
                .map_err(CudaResidentMoeCompileError::Validation)?,
            device_ordinal: self.device_ordinal,
        })
    }
}

impl From<RuntimeError> for ResidentMoeBindError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

pub fn preflight_resident_moe_plan(
    compiled: &CpuResidentMoeCompiledPlan,
    resources: &CpuResidentMoeResources,
) -> Result<ResidentMoePlanAccounting, ResidentMoeBindError> {
    validate_resident_resource(&compiled.analysis, resources)?;
    if compiled.physical_layout != ResidentMoeF32HotCpu::PHYSICAL_LAYOUT {
        return Err(ResidentMoeBindError::PhysicalLayoutMismatch);
    }

    let activation_elements = compiled
        .analysis
        .tokens
        .checked_mul(compiled.analysis.hidden)
        .ok_or(ResidentMoeBindError::ArithmeticOverflow)?;
    let activation_bytes = activation_elements
        .checked_mul(size_of::<f32>())
        .ok_or(ResidentMoeBindError::ArithmeticOverflow)?;
    let float_workspace_elements = compiled
        .analysis
        .tokens
        .checked_mul(compiled.analysis.expert_count)
        .and_then(|router| {
            compiled
                .analysis
                .tokens
                .checked_mul(compiled.analysis.top_k)
                .and_then(|routes| router.checked_add(routes))
        })
        .and_then(|elements| {
            compiled
                .analysis
                .intermediate
                .checked_mul(3)
                .and_then(|scratch| elements.checked_add(scratch))
        })
        .ok_or(ResidentMoeBindError::ArithmeticOverflow)?;
    let u32_workspace_elements = compiled
        .analysis
        .tokens
        .checked_mul(compiled.analysis.top_k)
        .and_then(|routes| routes.checked_add(compiled.analysis.expert_count))
        .and_then(|elements| elements.checked_add(1))
        .ok_or(ResidentMoeBindError::ArithmeticOverflow)?;
    let workspace_payload_bytes = float_workspace_elements
        .checked_mul(size_of::<f32>())
        .and_then(|bytes| {
            u32_workspace_elements
                .checked_mul(size_of::<u32>())
                .and_then(|ids| bytes.checked_add(ids))
        })
        .ok_or(ResidentMoeBindError::ArithmeticOverflow)?;

    Ok(ResidentMoePlanAccounting {
        logical_weight_count: compiled.analysis.logical_weight_count,
        weight_payload_bytes: compiled.analysis.weight_payload_bytes,
        input_payload_bytes: activation_bytes,
        output_payload_bytes: activation_bytes,
        workspace_payload_bytes,
        expert_count: compiled.analysis.expert_count,
        top_k: compiled.analysis.top_k,
        tokens: compiled.analysis.tokens,
    })
}

fn validate_resident_resource<'a>(
    analysis: &ResidentMoeAnalysis,
    resources: &'a CpuResidentMoeResources,
) -> Result<&'a CpuResidentMoeResource, ResidentMoeBindError> {
    let resource = resources
        .get()
        .ok_or(ResidentMoeBindError::MissingResource)?;
    compare_storages(&analysis.weight_storages, &resource.weight_storages)?;
    let weights = &resource.weights;
    if weights.expert_count() != analysis.expert_count {
        return Err(ResidentMoeBindError::ExpertCountMismatch {
            expected: analysis.expert_count,
            actual: weights.expert_count(),
        });
    }
    if weights.hidden() != analysis.hidden || weights.intermediate() != analysis.intermediate {
        return Err(ResidentMoeBindError::WeightShapeMismatch);
    }
    if weights.payload_bytes() != analysis.weight_payload_bytes {
        return Err(ResidentMoeBindError::WeightPayloadMismatch {
            expected: analysis.weight_payload_bytes,
            actual: weights.payload_bytes(),
        });
    }
    Ok(resource)
}

#[cfg(feature = "cuda")]
pub fn preflight_cuda_resident_moe_plan(
    compiled: &CudaResidentMoeCompiledPlan,
    resources: &CpuResidentMoeResources,
) -> Result<CudaResidentMoePlanAccounting, ResidentMoeBindError> {
    validate_resident_resource(&compiled.analysis, resources)?;
    let a = &compiled.analysis;
    let activation = bytes(
        a.tokens
            .checked_mul(a.hidden)
            .ok_or(ResidentMoeBindError::ArithmeticOverflow)?,
    )?;
    let router = bytes(
        a.tokens
            .checked_mul(a.expert_count)
            .ok_or(ResidentMoeBindError::ArithmeticOverflow)?,
    )?;
    let routes = a
        .tokens
        .checked_mul(a.top_k)
        .ok_or(ResidentMoeBindError::ArithmeticOverflow)?;
    let route_bytes = bytes(routes)?;
    let scratch = bytes(
        a.tokens
            .checked_mul(3)
            .and_then(|v| v.checked_mul(a.intermediate))
            .ok_or(ResidentMoeBindError::ArithmeticOverflow)?,
    )?;
    let access = (a.expert_count + 1)
        .checked_mul(4)
        .ok_or(ResidentMoeBindError::ArithmeticOverflow)?;
    let workspace = router
        .checked_add(route_bytes * 2)
        .and_then(|v| v.checked_add(scratch))
        .and_then(|v| v.checked_add(access))
        .ok_or(ResidentMoeBindError::ArithmeticOverflow)?;
    let device_payload = a
        .weight_payload_bytes
        .checked_add(activation * 2)
        .and_then(|v| v.checked_add(workspace))
        .ok_or(ResidentMoeBindError::ArithmeticOverflow)?;
    Ok(CudaResidentMoePlanAccounting {
        device_allocation_count: 8,
        device_payload_bytes: device_payload,
        weight_payload_bytes: a.weight_payload_bytes,
        input_payload_bytes: activation,
        output_payload_bytes: activation,
        workspace_payload_bytes: workspace,
        logical_weight_count: a.logical_weight_count,
        expert_count: a.expert_count,
        top_k: a.top_k,
        tokens: a.tokens,
        event_count: 2,
    })
}

#[cfg(feature = "cuda")]
pub fn bind_cuda_resident_moe_plan(
    compiled: CudaResidentMoeCompiledPlan,
    mut resources: CpuResidentMoeResources,
) -> Result<CudaResidentMoeExecutionPlan, ResidentMoeBindError> {
    let accounting = preflight_cuda_resident_moe_plan(&compiled, &resources)?;
    let resource = resources.take()?;
    let a = &compiled.analysis;
    let stream = OwnedCudaStream::create(compiled.device_ordinal).map_err(runtime_bind)?;
    let started = OwnedCudaEvent::create(compiled.device_ordinal).map_err(runtime_bind)?;
    let stopped = OwnedCudaEvent::create(compiled.device_ordinal).map_err(runtime_bind)?;
    let weights =
        OwnedCudaBuffer::allocate(compiled.device_ordinal, accounting.weight_payload_bytes)
            .map_err(runtime_bind)?;
    let input = OwnedCudaBuffer::allocate(compiled.device_ordinal, accounting.input_payload_bytes)
        .map_err(runtime_bind)?;
    let output =
        OwnedCudaBuffer::allocate(compiled.device_ordinal, accounting.output_payload_bytes)
            .map_err(runtime_bind)?;
    let router_len = bytes(a.tokens * a.expert_count)?;
    let route_len = bytes(a.tokens * a.top_k)?;
    let scratch_len = bytes(a.tokens * 3 * a.intermediate)?;
    let access_len = (a.expert_count + 1) * 4;
    let router_logits =
        OwnedCudaBuffer::allocate(compiled.device_ordinal, router_len).map_err(runtime_bind)?;
    let selected_ids =
        OwnedCudaBuffer::allocate(compiled.device_ordinal, route_len).map_err(runtime_bind)?;
    let selected_weights =
        OwnedCudaBuffer::allocate(compiled.device_ordinal, route_len).map_err(runtime_bind)?;
    let scratch =
        OwnedCudaBuffer::allocate(compiled.device_ordinal, scratch_len).map_err(runtime_bind)?;
    let access_counts =
        OwnedCudaBuffer::allocate(compiled.device_ordinal, access_len).map_err(runtime_bind)?;
    let weight_bytes = flatten_weight_bytes(&resource.weights);
    weights
        .copy_from_host_sync(&stream, &weight_bytes)
        .map_err(runtime_bind)?;
    let args = CudaResidentMoeF32ArgsV1::bind(
        &stream,
        weights
            .view(0, accounting.weight_payload_bytes)
            .map_err(runtime_bind)?,
        input
            .view(0, accounting.input_payload_bytes)
            .map_err(runtime_bind)?,
        output
            .view(0, accounting.output_payload_bytes)
            .map_err(runtime_bind)?,
        router_logits.view(0, router_len).map_err(runtime_bind)?,
        selected_ids.view(0, route_len).map_err(runtime_bind)?,
        selected_weights.view(0, route_len).map_err(runtime_bind)?,
        scratch.view(0, scratch_len).map_err(runtime_bind)?,
        access_counts.view(0, access_len).map_err(runtime_bind)?,
        a.tokens,
        a.hidden,
        a.intermediate,
        a.expert_count,
        a.top_k,
        a.routed_scale,
    )
    .map_err(runtime_bind)?;
    CudaResidentMoeExecutionPlan::new(
        args,
        weights,
        input,
        output,
        router_logits,
        selected_ids,
        selected_weights,
        scratch,
        access_counts,
        started,
        stopped,
        stream,
        a.tokens * a.hidden,
        a.tokens * a.expert_count,
        a.tokens * a.top_k,
        a.expert_count + 1,
        accounting,
    )
    .map_err(runtime_bind)
}

#[cfg(feature = "cuda")]
fn runtime_bind(error: CudaRuntimeError) -> ResidentMoeBindError {
    ResidentMoeBindError::CudaRuntime(error)
}

#[cfg(feature = "cuda")]
fn flatten_weight_bytes(weights: &ResidentMoeF32HotCpu) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(weights.payload_bytes());
    append_f32_le(&mut bytes, weights.router());
    for index in 0..weights.expert_count() {
        let expert = weights.expert(index).expect("validated expert");
        append_f32_le(&mut bytes, expert.gate());
        append_f32_le(&mut bytes, expert.up());
        append_f32_le(&mut bytes, expert.down());
    }
    let shared = weights.shared_expert();
    append_f32_le(&mut bytes, shared.gate());
    append_f32_le(&mut bytes, shared.up());
    append_f32_le(&mut bytes, shared.down());
    debug_assert_eq!(bytes.len(), weights.payload_bytes());
    bytes
}

#[cfg(feature = "cuda")]
fn append_f32_le(bytes: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

#[cfg(feature = "cuda")]
fn bytes(elements: usize) -> Result<usize, ResidentMoeBindError> {
    elements
        .checked_mul(4)
        .ok_or(ResidentMoeBindError::ArithmeticOverflow)
}

pub fn bind_resident_moe_plan(
    compiled: CpuResidentMoeCompiledPlan,
    mut resources: CpuResidentMoeResources,
) -> Result<ResidentMoeExecutionPlan, ResidentMoeBindError> {
    let accounting = preflight_resident_moe_plan(&compiled, &resources)?;
    let resource = resources.take()?;
    ResidentMoeExecutionPlan::new(
        Arc::new(resource.weights),
        compiled.analysis.tokens,
        compiled.analysis.top_k,
        compiled.analysis.routed_scale,
        compiled.launch,
        accounting,
    )
    .map_err(Into::into)
}

fn compare_storages(
    required: &ResidentMoeWeightStorages,
    actual: &ResidentMoeWeightStorages,
) -> Result<(), ResidentMoeBindError> {
    compare_storage(
        ResidentMoeTensorRole::Router,
        required.router,
        actual.router,
    )?;
    if required.experts.len() != actual.experts.len() {
        return Err(ResidentMoeBindError::ExpertCountMismatch {
            expected: required.experts.len(),
            actual: actual.experts.len(),
        });
    }
    for (index, (required, actual)) in required
        .experts
        .iter()
        .zip(actual.experts.iter())
        .enumerate()
    {
        compare_expert_storages(required, actual, index)?;
    }
    compare_storage(
        ResidentMoeTensorRole::SharedGate,
        required.shared_expert.gate,
        actual.shared_expert.gate,
    )?;
    compare_storage(
        ResidentMoeTensorRole::SharedUp,
        required.shared_expert.up,
        actual.shared_expert.up,
    )?;
    compare_storage(
        ResidentMoeTensorRole::SharedDown,
        required.shared_expert.down,
        actual.shared_expert.down,
    )
}

fn compare_expert_storages(
    required: &ResidentMoeExpertStorages,
    actual: &ResidentMoeExpertStorages,
    index: usize,
) -> Result<(), ResidentMoeBindError> {
    compare_storage(
        ResidentMoeTensorRole::ExpertGate(index),
        required.gate,
        actual.gate,
    )?;
    compare_storage(
        ResidentMoeTensorRole::ExpertUp(index),
        required.up,
        actual.up,
    )?;
    compare_storage(
        ResidentMoeTensorRole::ExpertDown(index),
        required.down,
        actual.down,
    )
}

fn compare_storage(
    role: ResidentMoeTensorRole,
    required: StorageId,
    actual: StorageId,
) -> Result<(), ResidentMoeBindError> {
    if required == actual {
        return Ok(());
    }
    if required.index == actual.index {
        Err(ResidentMoeBindError::GenerationMismatch {
            role,
            required,
            actual,
        })
    } else {
        Err(ResidentMoeBindError::StorageMismatch {
            role,
            required,
            actual,
        })
    }
}
