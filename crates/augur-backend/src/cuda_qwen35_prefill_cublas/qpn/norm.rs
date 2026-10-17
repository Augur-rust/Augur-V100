//! Fresh producer -> consumers on a single retained serial plan.
//! No pointer-identity readiness cache and no externally callable skip-stage path.
use super::*;
use crate::CudaRmsnormF32ArgsV1;

pub struct CudaQwen35NormQpnGroupV1 {
    projections: Vec<CudaQwen35QpnRetainedProjectionV1>,
    raw: Vec<AugurCudaQwen35QpnPrefillArgsV1>,
    norm: augur_cuda_sys::AugurCudaRmsnormF32ArgsV1,
    _norm_input: Rc<OwnedCudaBuffer>,
    _norm_weight: Rc<OwnedCudaBuffer>,
}
impl CudaQwen35NormQpnGroupV1 {
    pub fn bind(
        norm: CudaRmsnormF32ArgsV1,
        input: Rc<OwnedCudaBuffer>,
        weight: Rc<OwnedCudaBuffer>,
        projections: Vec<CudaQwen35QpnRetainedProjectionV1>,
    ) -> Result<Self, CudaRuntimeError> {
        if !(2..=3).contains(&projections.len()) {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let norm = norm.qpn_fused_descriptor()?;
        let o = &projections[0].owners;
        for p in &projections {
            if !Rc::ptr_eq(&o.plan,&p.owners.plan) || !Rc::ptr_eq(&o.stream,&p.owners.stream) {
                return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
            }
        }
        o.plan.require_stream(&o.stream)?;
        for (buffer,address,bytes) in [(&input,norm.input,norm.input_len_bytes),
                                      (&weight,norm.weight,norm.weight_len_bytes)] {
            let view=buffer.view(0,buffer.len_bytes())?;
            if view.device_ordinal()!=o.stream.device_ordinal() || bytes!=buffer.len_bytes() as u64
                || address!=view.device_ptr().raw_value() {
                return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
            }
        }
        let raw: Vec<_> = projections.iter().map(|p|p.raw).collect();
        // SAFETY: each projection retains the normalized F32 output, its weights,
        // output and exact plan/stream. We additionally retain both norm sources.
        // Native preflight validates the whole cross-operation alias graph.
        unsafe {
            CudaRuntimeError::from_result(OPERATION,
                augur_cuda_sys::augur_cuda_qwen35_qpn_norm_group_preflight_v1(
                    o.stream.device_ordinal(),o.stream.raw()?,o.plan.raw,&norm,raw.as_ptr(),raw.len() as u32))?;
        }
        Ok(Self { projections,raw,norm,_norm_input:input,_norm_weight:weight })
    }
    pub fn submit_for_plan(&self) -> Result<(), CudaRuntimeError> {
        let o=&self.projections[0].owners;
        o.plan.require_stream(&o.stream)?;
        // SAFETY: immutable preflight descriptors and retained owners. Each call
        // re-produces the stage before ANY consumer; no intervening scratch user.
        unsafe {
            CudaRuntimeError::from_result(OPERATION,
                augur_cuda_sys::augur_cuda_qwen35_qpn_norm_group_prevalidated_submit_v1(
                    o.stream.device_ordinal(),o.stream.raw()?,o.plan.raw,&self.norm,
                    self.raw.as_ptr(),self.raw.len() as u32))
        }
    }
}
impl Drop for CudaQwen35NormQpnGroupV1 {
    fn drop(&mut self) {
        // Wait before releasing norm operands, even when other projection clones
        // keep the underlying plan alive (their Drop is not a completion proof).
        let _=self.projections[0].owners.stream.synchronize();
    }
}

#[cfg(test)]
mod tests;
