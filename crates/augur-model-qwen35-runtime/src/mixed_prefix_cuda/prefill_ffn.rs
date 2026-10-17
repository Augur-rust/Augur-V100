//! Mutually exclusive physical/numerical FFN profiles. No GGML view may ever
//! be constructed over QPN bytes, including when installing batch policies.
use super::*;

pub(super) struct LegacyPrefillFfn {
    pub ffn_input_quantize: CudaQ8_1ActivationQuantizeArgsV1,
    pub ffn_gate: CudaQwen35PrefillCublasProjectionV1,
    pub ffn_up: CudaQwen35PrefillCublasProjectionV1,
    pub silu_mul: CudaSiluMulF32ArgsV1,
    pub preserved_silu_q8: Option<augur_backend::CudaSiluQ8PreservedB8V1>,
    pub ffn_down_quantize: CudaQ8_1ActivationQuantizeArgsV1,
    pub ffn_down: CudaQwen35PrefillCublasProjectionV1,
}

pub(super) enum PrefillFfn {
    Legacy(LegacyPrefillFfn),
    Qpn(augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnFfnProgramV1),
}

impl PrefillFfn {
    pub fn qpn_rows(&self) -> Option<usize> {
        match self { Self::Qpn(p) => Some(p.original_input_rows()), Self::Legacy(_) => None }
    }
    pub fn legacy_mut_opt(&mut self) -> Option<&mut LegacyPrefillFfn> {
        match self { Self::Legacy(p) => Some(p), Self::Qpn(_) => None }
    }
    pub fn validate_original_input(&self, rows: usize, exact_input: bool) -> Result<(), Qwen35MixedPrefixCudaError> {
        let valid = match self {
            Self::Qpn(p) => p.original_input_rows() == rows,
            Self::Legacy(p) => p.ffn_down.raw().tokens as usize == rows
                && p.ffn_down.llama_full_original_enabled()
                && p.ffn_down.llama_exact_input_rows_enabled() == exact_input,
        };
        if !valid { return Err(invalid("FFN original input profile/extent mismatch")); }
        Ok(())
    }
    pub fn legacy_opt(&self) -> Option<&LegacyPrefillFfn> {
        match self { Self::Legacy(p) => Some(p), Self::Qpn(_) => None }
    }
    pub fn legacy(&self) -> Result<&LegacyPrefillFfn, Qwen35MixedPrefixCudaError> {
        self.legacy_opt().ok_or_else(|| invalid("legacy FFN policy cannot reinterpret QPN weights"))
    }
    pub fn legacy_mut(&mut self) -> Result<&mut LegacyPrefillFfn, Qwen35MixedPrefixCudaError> {
        match self { Self::Legacy(p) => Ok(p), Self::Qpn(_) => Err(invalid("legacy FFN policy cannot reinterpret QPN weights")) }
    }
    pub fn submit_gate_up(&self, stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1, shared: bool) -> Result<(), Qwen35MixedPrefixCudaError> {
        match self {
            Self::Qpn(p) => p.submit_gate_up_on(stream)?,
            Self::Legacy(p) => {
                cublas.submit_required_q8_input(stream, &p.ffn_input_quantize, &[&p.ffn_gate, &p.ffn_up])?;
                if shared { cublas.submit_shared_input_group(stream, &[&p.ffn_gate, &p.ffn_up])?; }
                else { cublas.submit(stream, &p.ffn_gate)?; cublas.submit(stream, &p.ffn_up)?; }
            }
        }
        Ok(())
    }
    pub fn submit_activation(&self, stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1) -> Result<(), Qwen35MixedPrefixCudaError> {
        match self {
            Self::Qpn(p) => p.submit_activation_on(stream)?,
            Self::Legacy(p) => {
                if let Some(fused) = &p.preserved_silu_q8 { fused.submit(stream)?; }
                else {
                    if p.ffn_down.llama_exact_input_rows_enabled() { p.silu_mul.submit_llama_input_rows(stream)?; }
                    else if p.ffn_down.llama_full_original_enabled() { p.silu_mul.submit_llama_prefix(stream)?; }
                    else { p.silu_mul.submit_for_plan(stream)?; }
                    cublas.submit_required_q8_input(stream, &p.ffn_down_quantize, &[&p.ffn_down])?;
                }
            }
        }
        Ok(())
    }
    pub fn submit_down(&self, stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1) -> Result<(), Qwen35MixedPrefixCudaError> {
        match self { Self::Qpn(p) => p.submit_down_on(stream)?, Self::Legacy(p) => cublas.submit(stream, &p.ffn_down)? }
        Ok(())
    }
}

impl PrefillLayerOps {
    pub fn legacy_projections(&self) -> Vec<&CudaQwen35PrefillCublasProjectionV1> {
        let mut projections = match self {
            Self::Linear(p) => vec![&p.qkv, &p.z, &p.output],
            Self::Full(p) => vec![&p.q, &p.k, &p.v, &p.output],
        };
        if let Some(p) = self.ffn().legacy_opt() {
            projections.extend([&p.ffn_gate, &p.ffn_up, &p.ffn_down]);
        }
        projections
    }
    pub fn ffn(&self) -> &PrefillFfn {
        match self { Self::Linear(p) => &p.ffn, Self::Full(p) => &p.ffn }
    }
    pub fn ffn_mut(&mut self) -> &mut PrefillFfn {
        match self { Self::Linear(p) => &mut p.ffn, Self::Full(p) => &mut p.ffn }
    }
}
