//! Explicit opt-in, retained BF16 a/b pair for the new-model experiment.
use super::*;
pub(crate) struct NerkyorBf16GatePair {
    op: augur_backend::CudaNerkyorBf16GatePairF32CtaArgsV1,
    left: CheckpointScalarCudaProjection,
    _right: CheckpointScalarCudaProjection,
}
impl NerkyorBf16GatePair {
    pub(crate) fn bind(left: CheckpointScalarCudaProjection, right: CheckpointScalarCudaProjection) -> Result<Self> {
        let prefix = left.weight.spec.name.strip_suffix(".in_proj_a");
        if !left.weight.policy.uses_nerkyor_gate_f32_cta_research()
            || left.weight.policy != right.weight.policy || prefix.is_none()
            || prefix != right.weight.spec.name.strip_suffix(".in_proj_b")
            || !Rc::ptr_eq(&left.weight.stream, &right.weight.stream)
            || !Rc::ptr_eq(&left._input, &right._input) {
            return Err("Nerkyor research gate requires the same layer, policy, stream and input owner".into());
        }
        let (Storage::Bf16(_), Storage::Bf16(_), Operation::Bf16(a), Operation::Bf16(b)) =
            (&left.weight.storage, &right.weight.storage, &left.operation, &right.operation) else {
            return Err("Nerkyor research gate requires retained original BF16 operands".into());
        };
        let op = augur_backend::CudaNerkyorBf16GatePairF32CtaArgsV1::bind(&left.weight.stream, *a, *b)?;
        Ok(Self {op, left, _right:right})
    }
    pub(crate) fn submit_for_plan(&self) -> Result<()> {
        self.op.submit_for_plan(&self.left.weight.stream)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn research_gate_policy_never_selects_qpn_fp8_or_nvfp4_gate_schedules() {
        let policy = CheckpointProjectionPolicy::NerkyorBf16GateF32CtaResearch;
        assert!(policy.uses_nerkyor_gate_f32_cta_research());
        assert!(!CheckpointProjectionPolicy::default().uses_nerkyor_gate_f32_cta_research());
        assert!(!policy.uses_qpn_projections());
        assert!(!policy.uses_qpn_ffn());
        assert!(!policy.uses_fp8_warp());
        assert!(!policy.uses_small_gate_split16());
        assert!(policy.small_gate_pair_schedule().is_none());
        assert_ne!(policy.label(), CheckpointProjectionPolicy::OriginalF32.label());
    }
}

#[cfg(test)]
#[path = "nerkyor_gate_gpu_tests.rs"]
mod gpu_tests;

#[cfg(feature = "qwen35-prefill-cublas")]
impl NerkyorBf16GatePair {
    pub(crate) fn rebind_rows(&self, stream: &Rc<OwnedCudaStream>, input: Rc<OwnedCudaBuffer>,
        alpha: Rc<OwnedCudaBuffer>, beta: Rc<OwnedCudaBuffer>, rows: usize) -> Result<Vec<Self>> {
        if !Rc::ptr_eq(stream, &self.left.weight.stream) || !(1..=8).contains(&rows) {
            return Err("research gate rows require same stream and B1..8".into());
        }
        (0..rows).map(|row| Self::bind(
            self.left.weight.bind(Rc::clone(&input),row*5120*4,Rc::clone(&alpha),row*48*4)?,
            self._right.weight.bind(Rc::clone(&input),row*5120*4,Rc::clone(&beta),row*48*4)?,
        )).collect()
    }
}
