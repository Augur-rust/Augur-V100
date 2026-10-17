use augur_core::PhysicalLayoutId;

use crate::HotLayoutError;

pub const RESIDENT_MOE_MAX_EXPERTS: usize = 8;
pub const RESIDENT_MOE_MAX_TOP_K: usize = 4;

#[derive(Clone, Debug, PartialEq)]
pub struct ResidentMoeF32HotExpert {
    gate: Box<[f32]>,
    up: Box<[f32]>,
    down: Box<[f32]>,
}

impl ResidentMoeF32HotExpert {
    pub fn new(
        hidden: usize,
        intermediate: usize,
        gate: Vec<f32>,
        up: Vec<f32>,
        down: Vec<f32>,
    ) -> Result<Self, HotLayoutError> {
        let up_gate_len = hidden
            .checked_mul(intermediate)
            .ok_or(HotLayoutError::BadShape)?;
        let down_len = intermediate
            .checked_mul(hidden)
            .ok_or(HotLayoutError::BadShape)?;
        if hidden == 0
            || intermediate == 0
            || gate.len() != up_gate_len
            || up.len() != up_gate_len
            || down.len() != down_len
        {
            return Err(HotLayoutError::BadShape);
        }
        for (index, value) in gate.iter().chain(&up).chain(&down).enumerate() {
            if !value.is_finite() {
                return Err(HotLayoutError::NonFinitePayload { index });
            }
        }
        Ok(Self {
            gate: gate.into_boxed_slice(),
            up: up.into_boxed_slice(),
            down: down.into_boxed_slice(),
        })
    }

    pub fn gate(&self) -> &[f32] {
        &self.gate
    }

    pub fn up(&self) -> &[f32] {
        &self.up
    }

    pub fn down(&self) -> &[f32] {
        &self.down
    }

    pub fn payload_bytes(&self) -> usize {
        (self.gate.len() + self.up.len() + self.down.len()) * size_of::<f32>()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResidentMoeF32HotCpu {
    hidden: usize,
    intermediate: usize,
    router: Box<[f32]>,
    experts: Box<[ResidentMoeF32HotExpert]>,
    shared_expert: ResidentMoeF32HotExpert,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentMoeF32OwnerIdentity {
    pub owner: usize,
    pub router: usize,
    pub expert_gate: [usize; RESIDENT_MOE_MAX_EXPERTS],
    pub expert_up: [usize; RESIDENT_MOE_MAX_EXPERTS],
    pub expert_down: [usize; RESIDENT_MOE_MAX_EXPERTS],
    pub shared_gate: usize,
    pub shared_up: usize,
    pub shared_down: usize,
    pub expert_count: usize,
}

impl ResidentMoeF32HotCpu {
    pub const PHYSICAL_LAYOUT: PhysicalLayoutId =
        PhysicalLayoutId::RESIDENT_MOE_F32_HOT_CPU_PORTABLE_V1;

    pub fn new(
        hidden: usize,
        intermediate: usize,
        router: Vec<f32>,
        experts: Vec<ResidentMoeF32HotExpert>,
        shared_expert: ResidentMoeF32HotExpert,
    ) -> Result<Self, HotLayoutError> {
        if hidden == 0
            || intermediate == 0
            || !matches!(experts.len(), 4 | 8)
            || router.len()
                != experts
                    .len()
                    .checked_mul(hidden)
                    .ok_or(HotLayoutError::BadShape)?
        {
            return Err(HotLayoutError::BadShape);
        }
        if let Some(index) = router.iter().position(|value| !value.is_finite()) {
            return Err(HotLayoutError::NonFinitePayload { index });
        }
        let expected_expert_bytes = hidden
            .checked_mul(intermediate)
            .and_then(|elements| elements.checked_mul(3))
            .and_then(|elements| elements.checked_mul(size_of::<f32>()))
            .ok_or(HotLayoutError::BadShape)?;
        if experts
            .iter()
            .chain(std::iter::once(&shared_expert))
            .any(|expert| expert.payload_bytes() != expected_expert_bytes)
        {
            return Err(HotLayoutError::BadShape);
        }
        Ok(Self {
            hidden,
            intermediate,
            router: router.into_boxed_slice(),
            experts: experts.into_boxed_slice(),
            shared_expert,
        })
    }

    pub const fn hidden(&self) -> usize {
        self.hidden
    }

    pub const fn intermediate(&self) -> usize {
        self.intermediate
    }

    pub fn expert_count(&self) -> usize {
        self.experts.len()
    }

    pub fn router(&self) -> &[f32] {
        &self.router
    }

    pub fn expert(&self, index: usize) -> Option<&ResidentMoeF32HotExpert> {
        self.experts.get(index)
    }

    pub fn shared_expert(&self) -> &ResidentMoeF32HotExpert {
        &self.shared_expert
    }

    pub fn payload_bytes(&self) -> usize {
        self.router.len() * size_of::<f32>()
            + self
                .experts
                .iter()
                .chain(std::iter::once(&self.shared_expert))
                .map(ResidentMoeF32HotExpert::payload_bytes)
                .sum::<usize>()
    }

    pub fn owner_identity(&self) -> ResidentMoeF32OwnerIdentity {
        let mut expert_gate = [0; RESIDENT_MOE_MAX_EXPERTS];
        let mut expert_up = [0; RESIDENT_MOE_MAX_EXPERTS];
        let mut expert_down = [0; RESIDENT_MOE_MAX_EXPERTS];
        for (index, expert) in self.experts.iter().enumerate() {
            expert_gate[index] = expert.gate().as_ptr() as usize;
            expert_up[index] = expert.up().as_ptr() as usize;
            expert_down[index] = expert.down().as_ptr() as usize;
        }
        ResidentMoeF32OwnerIdentity {
            owner: self as *const Self as usize,
            router: self.router().as_ptr() as usize,
            expert_gate,
            expert_up,
            expert_down,
            shared_gate: self.shared_expert.gate().as_ptr() as usize,
            shared_up: self.shared_expert.up().as_ptr() as usize,
            shared_down: self.shared_expert.down().as_ptr() as usize,
            expert_count: self.experts.len(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ResidentMoeF32Workspace {
    tokens: usize,
    hidden: usize,
    intermediate: usize,
    expert_count: usize,
    top_k: usize,
    router_logits: Box<[f32]>,
    selected_ids: Box<[u32]>,
    selected_weights: Box<[f32]>,
    gate: Box<[f32]>,
    up: Box<[f32]>,
    hidden_values: Box<[f32]>,
    expert_access_counts: Box<[u32]>,
    shared_access_count: u32,
}

impl ResidentMoeF32Workspace {
    pub fn new(
        tokens: usize,
        hidden: usize,
        intermediate: usize,
        expert_count: usize,
        top_k: usize,
    ) -> Result<Self, HotLayoutError> {
        if tokens == 0
            || hidden == 0
            || intermediate == 0
            || !matches!(expert_count, 4 | 8)
            || !matches!(top_k, 2 | 4)
            || top_k > expert_count
        {
            return Err(HotLayoutError::BadShape);
        }
        let router_len = tokens
            .checked_mul(expert_count)
            .ok_or(HotLayoutError::BadShape)?;
        let route_len = tokens.checked_mul(top_k).ok_or(HotLayoutError::BadShape)?;
        Ok(Self {
            tokens,
            hidden,
            intermediate,
            expert_count,
            top_k,
            router_logits: vec![0.0; router_len].into_boxed_slice(),
            selected_ids: vec![u32::MAX; route_len].into_boxed_slice(),
            selected_weights: vec![0.0; route_len].into_boxed_slice(),
            gate: vec![0.0; intermediate].into_boxed_slice(),
            up: vec![0.0; intermediate].into_boxed_slice(),
            hidden_values: vec![0.0; intermediate].into_boxed_slice(),
            expert_access_counts: vec![0; expert_count].into_boxed_slice(),
            shared_access_count: 0,
        })
    }

    pub const fn tokens(&self) -> usize {
        self.tokens
    }
    pub const fn hidden(&self) -> usize {
        self.hidden
    }
    pub const fn intermediate(&self) -> usize {
        self.intermediate
    }
    pub const fn expert_count(&self) -> usize {
        self.expert_count
    }
    pub const fn top_k(&self) -> usize {
        self.top_k
    }
    pub fn router_logits(&self) -> &[f32] {
        &self.router_logits
    }
    pub fn selected_ids(&self) -> &[u32] {
        &self.selected_ids
    }
    pub fn selected_weights(&self) -> &[f32] {
        &self.selected_weights
    }
    pub fn expert_access_counts(&self) -> &[u32] {
        &self.expert_access_counts
    }
    pub const fn shared_access_count(&self) -> u32 {
        self.shared_access_count
    }
    pub fn router_logits_token_mut(&mut self, token: usize) -> Option<&mut [f32]> {
        token_slice_mut(&mut self.router_logits, token, self.expert_count)
    }
    pub fn selected_ids_token_mut(&mut self, token: usize) -> Option<&mut [u32]> {
        token_slice_mut(&mut self.selected_ids, token, self.top_k)
    }
    pub fn selected_weights_token_mut(&mut self, token: usize) -> Option<&mut [f32]> {
        token_slice_mut(&mut self.selected_weights, token, self.top_k)
    }
    pub fn gate_mut(&mut self) -> &mut [f32] {
        &mut self.gate
    }
    pub fn up_mut(&mut self) -> &mut [f32] {
        &mut self.up
    }
    pub fn hidden_values(&self) -> &[f32] {
        &self.hidden_values
    }
    pub fn hidden_values_mut(&mut self) -> &mut [f32] {
        &mut self.hidden_values
    }
    pub fn scratch_mut(&mut self) -> (&mut [f32], &mut [f32], &mut [f32]) {
        (&mut self.gate, &mut self.up, &mut self.hidden_values)
    }
    pub fn reset_trace(&mut self) {
        self.router_logits.fill(0.0);
        self.selected_ids.fill(u32::MAX);
        self.selected_weights.fill(0.0);
        self.expert_access_counts.fill(0);
        self.shared_access_count = 0;
    }
    pub fn record_expert_access(&mut self, expert: usize) -> bool {
        let Some(count) = self.expert_access_counts.get_mut(expert) else {
            return false;
        };
        *count = count.saturating_add(1);
        true
    }
    pub fn record_shared_access(&mut self) {
        self.shared_access_count = self.shared_access_count.saturating_add(1);
    }
    pub fn payload_bytes(&self) -> usize {
        (self.router_logits.len()
            + self.selected_weights.len()
            + self.gate.len()
            + self.up.len()
            + self.hidden_values.len())
            * size_of::<f32>()
            + self.selected_ids.len() * size_of::<u32>()
            + self.expert_access_counts.len() * size_of::<u32>()
            + size_of::<u32>()
    }
}

fn token_slice_mut<T>(values: &mut [T], token: usize, width: usize) -> Option<&mut [T]> {
    let start = token.checked_mul(width)?;
    let end = start.checked_add(width)?;
    values.get_mut(start..end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_accounting_and_trace_reset_are_exact() {
        let mut workspace = ResidentMoeF32Workspace::new(3, 8, 12, 4, 2).unwrap();
        workspace.record_expert_access(2);
        workspace.record_shared_access();
        workspace.selected_ids_token_mut(1).unwrap()[0] = 2;
        workspace.reset_trace();
        assert_eq!(workspace.expert_access_counts(), &[0; 4]);
        assert_eq!(workspace.shared_access_count(), 0);
        assert!(workspace.selected_ids().iter().all(|id| *id == u32::MAX));
        assert_eq!(workspace.payload_bytes(), 260);
    }
}
