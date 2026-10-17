use augur_core::TensorId;

use crate::{M138CompiledPlan, M138StepKind};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M141FusionCandidate {
    pub first_step: usize,
    pub last_step: usize,
    pub covered_steps: usize,
    pub inputs: Vec<TensorId>,
    pub observable_outputs: Vec<TensorId>,
    pub preserved_boundaries: Vec<u16>,
    pub accumulation: &'static str,
    pub rounding: &'static str,
    pub input_output_alias: bool,
    pub native_kernel_present: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum M141CandidateError {
    RegionMissing,
    RegionSemantics,
    BoundaryCrossed,
}

pub fn discover_m141_pool_candidate(
    plan: &M138CompiledPlan,
) -> Result<M141FusionCandidate, M141CandidateError> {
    let first = plan
        .steps()
        .iter()
        .position(|step| step.kind == M138StepKind::BroadcastAdd)
        .ok_or(M141CandidateError::RegionMissing)?;
    let softmax = plan
        .steps()
        .iter()
        .position(|step| step.kind == M138StepKind::Softmax)
        .ok_or(M141CandidateError::RegionMissing)?;
    if softmax <= first + 2 {
        return Err(M141CandidateError::RegionSemantics);
    }
    let region = &plan.steps()[first..softmax];
    if region.iter().any(|step| {
        !matches!(
            step.kind,
            M138StepKind::BroadcastAdd
                | M138StepKind::Reshape
                | M138StepKind::Slice
                | M138StepKind::Pad
                | M138StepKind::Concat
        )
    }) {
        return Err(M141CandidateError::RegionSemantics);
    }
    if region.iter().any(|step| step.boundary.is_some()) {
        return Err(M141CandidateError::BoundaryCrossed);
    }
    let inputs = plan.steps()[first].inputs.clone();
    let observable_outputs = vec![
        plan.steps()[first].output,
        plan.steps()[softmax - 2].output,
        plan.steps()[softmax - 1].output,
    ];
    Ok(M141FusionCandidate {
        first_step: first,
        last_step: softmax - 1,
        covered_steps: softmax - first,
        inputs,
        observable_outputs,
        preserved_boundaries: Vec::new(),
        accumulation: "F32",
        rounding: "M139_STAGED_EQUIVALENCE_REQUIRED",
        input_output_alias: false,
        native_kernel_present: false,
    })
}
