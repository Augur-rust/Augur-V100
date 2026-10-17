use std::sync::Arc;

use augur_backend::{CpuBackend, CpuStream};
use augur_core::RuntimeError;
use augur_kernel_api::{
    ResidentMoeF32CpuLaunch, ResidentMoeF32HotCpu, ResidentMoeF32OwnerIdentity,
    ResidentMoeF32Workspace,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResidentMoePlanAccounting {
    pub logical_weight_count: usize,
    pub weight_payload_bytes: usize,
    pub input_payload_bytes: usize,
    pub output_payload_bytes: usize,
    pub workspace_payload_bytes: usize,
    pub expert_count: usize,
    pub top_k: usize,
    pub tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentMoeBufferIdentity {
    pub input: usize,
    pub output: usize,
    pub router_logits: usize,
    pub selected_ids: usize,
    pub selected_weights: usize,
    pub expert_access_counts: usize,
}

#[derive(Debug)]
pub struct ResidentMoeExecutionPlan {
    backend: CpuBackend,
    weights: Arc<ResidentMoeF32HotCpu>,
    input: Box<[f32]>,
    output: Box<[f32]>,
    workspace: ResidentMoeF32Workspace,
    top_k: usize,
    routed_scale: f32,
    launch: ResidentMoeF32CpuLaunch,
    accounting: ResidentMoePlanAccounting,
}

impl ResidentMoeExecutionPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weights: Arc<ResidentMoeF32HotCpu>,
        tokens: usize,
        top_k: usize,
        routed_scale: f32,
        launch: ResidentMoeF32CpuLaunch,
        accounting: ResidentMoePlanAccounting,
    ) -> Result<Self, RuntimeError> {
        let elements = tokens
            .checked_mul(weights.hidden())
            .ok_or(RuntimeError::BadPlan)?;
        let workspace = ResidentMoeF32Workspace::new(
            tokens,
            weights.hidden(),
            weights.intermediate(),
            weights.expert_count(),
            top_k,
        )
        .map_err(|_| RuntimeError::BadPlan)?;
        if !routed_scale.is_finite()
            || routed_scale <= 0.0
            || accounting.weight_payload_bytes != weights.payload_bytes()
            || accounting.input_payload_bytes != elements * size_of::<f32>()
            || accounting.output_payload_bytes != elements * size_of::<f32>()
            || accounting.workspace_payload_bytes != workspace.payload_bytes()
            || accounting.expert_count != weights.expert_count()
            || accounting.top_k != top_k
            || accounting.tokens != tokens
        {
            return Err(RuntimeError::BadPlan);
        }
        Ok(Self {
            backend: CpuBackend,
            weights,
            input: vec![0.0; elements].into_boxed_slice(),
            output: vec![0.0; elements].into_boxed_slice(),
            workspace,
            top_k,
            routed_scale,
            launch,
            accounting,
        })
    }

    pub fn backend(&self) -> &CpuBackend {
        &self.backend
    }

    pub fn set_input(&mut self, input: &[f32]) -> Result<(), RuntimeError> {
        if input.len() != self.input.len() {
            return Err(RuntimeError::BadBuffer);
        }
        self.input.copy_from_slice(input);
        Ok(())
    }

    pub fn execute_cpu(&mut self, _stream: &mut CpuStream) -> Result<(), RuntimeError> {
        (self.launch)(
            &self.weights,
            &self.input,
            self.top_k,
            self.routed_scale,
            &mut self.workspace,
            &mut self.output,
        )
        .map_err(|_| RuntimeError::KernelFailed)
    }

    pub fn output(&self) -> &[f32] {
        &self.output
    }

    pub fn router_logits(&self) -> &[f32] {
        self.workspace.router_logits()
    }

    pub fn selected_expert_ids(&self) -> &[u32] {
        self.workspace.selected_ids()
    }

    pub fn selected_routing_weights(&self) -> &[f32] {
        self.workspace.selected_weights()
    }

    pub fn expert_access_counts(&self) -> &[u32] {
        self.workspace.expert_access_counts()
    }

    pub const fn shared_access_count(&self) -> u32 {
        self.workspace.shared_access_count()
    }

    pub const fn accounting(&self) -> ResidentMoePlanAccounting {
        self.accounting
    }

    pub fn weight_owner_identity(&self) -> ResidentMoeF32OwnerIdentity {
        self.weights.owner_identity()
    }

    pub fn buffer_identity(&self) -> ResidentMoeBufferIdentity {
        ResidentMoeBufferIdentity {
            input: self.input.as_ptr() as usize,
            output: self.output.as_ptr() as usize,
            router_logits: self.workspace.router_logits().as_ptr() as usize,
            selected_ids: self.workspace.selected_ids().as_ptr() as usize,
            selected_weights: self.workspace.selected_weights().as_ptr() as usize,
            expert_access_counts: self.workspace.expert_access_counts().as_ptr() as usize,
        }
    }
}
