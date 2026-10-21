//! Retained QPN physical weights with original F32 pair arithmetic.
use super::*;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Schedule { SourceExact, SourcePrefetch, SourceF64Reassoc, Cta1, Cta4, Cta8, Cta16, Workspace4, Workspace8, Workspace16, Workspace32 }
impl Schedule {
    pub const ALL: [Self; 11] = [Self::SourceExact,Self::SourcePrefetch,Self::SourceF64Reassoc,Self::Cta1,Self::Cta4,Self::Cta8,Self::Cta16,Self::Workspace4,Self::Workspace8,Self::Workspace16,Self::Workspace32];
    pub fn geometry(self) -> (u32,u32) { match self {
        Self::SourceExact=>(2,4),
        // Same chains and reduction tree as SourceExact; needs a 16-byte input.
        Self::SourcePrefetch=>(3,4),
        // Same F32 pairs, re-associated F64 sum: a separate numerical identity.
        Self::SourceF64Reassoc=>(4,4),
        Self::Cta1=>(0,1),Self::Cta4=>(0,4),Self::Cta8=>(0,8),Self::Cta16=>(0,16),
        Self::Workspace4=>(1,4),Self::Workspace8=>(1,8),Self::Workspace16=>(1,16),Self::Workspace32=>(1,32),
    }}
}
pub struct CudaQpnSourcePairProjection {
    raw: AugurCudaNvfp4Sm70F32MatvecArgsV1,
    schedule: Schedule,
    stream: Rc<OwnedCudaStream>,
    _weight: Rc<OwnedCudaQwen35QpnWeightV1>,
    _input: Rc<OwnedCudaBuffer>,
    _output: Rc<OwnedCudaBuffer>,
    partial: Option<OwnedCudaBuffer>,
}
impl CudaQpnSourcePairProjection {
    #[allow(clippy::too_many_arguments)]
    pub fn bind(stream: Rc<OwnedCudaStream>, weight: Rc<OwnedCudaQwen35QpnWeightV1>,
        input: Rc<OwnedCudaBuffer>, input_offset: usize, output: Rc<OwnedCudaBuffer>, output_offset: usize,
        schedule: Schedule) -> Result<Self,CudaRuntimeError> {
        if weight.layout()!=PhysicalLayoutId::DEVICE_QWEN35_NVFP4_QPN_PLANES_V1 {
            return Err(error(CudaRuntimeErrorKind::InvalidMetadata));
        }
        let w=weight.physical_view()?;
        let x=view(&input,&stream,input_offset,weight.cols()*4)?;
        if matches!(schedule,Schedule::SourcePrefetch|Schedule::SourceF64Reassoc)&&x.checked_address(OP)?%16!=0 {
            return Err(error(CudaRuntimeErrorKind::InvalidMetadata));
        }
        let y=view(&output,&stream,output_offset,weight.rows()*4)?;
        if w.device_ordinal()!=stream.device_ordinal() {return Err(error(CudaRuntimeErrorKind::DeviceMismatch));}
        if w.overlaps(x,OP)?||w.overlaps(y,OP)?||x.overlaps(y,OP)? {
            return Err(error(CudaRuntimeErrorKind::OverlappingDeviceRanges));
        }
        let mut raw=AugurCudaNvfp4Sm70F32MatvecArgsV1::new();
        raw.weight=w.checked_address(OP)?;raw.weight_len_bytes=w.len_bytes() as u64;
        raw.input=x.checked_address(OP)?;raw.input_len_bytes=x.len_bytes() as u64;
        raw.output=y.checked_address(OP)?;raw.output_len_bytes=y.len_bytes() as u64;
        raw.rows=weight.rows() as u32;raw.cols=weight.cols() as u32;raw.tensor_scale=weight.tensor_scale();
        let (mode,splits)=schedule.geometry();
        let partial=if mode==1 {Some(OwnedCudaBuffer::allocate(stream.device_ordinal(),weight.rows()*splits as usize*8)?)}else{None};
        Ok(Self{raw,schedule,stream,_weight:weight,_input:input,_output:output,partial})
    }
    pub fn workspace_bytes(&self)->usize {self.partial.as_ref().map_or(0,|b|b.len_bytes())}
    pub fn submit(&self)->Result<(),CudaRuntimeError> {
        let (mode,splits)=self.schedule.geometry();
        let partial=match &self.partial {Some(p)=>p.view(0,p.len_bytes())?.checked_address(OP)?,None=>0};
        // SAFETY: immutable typed QPN owner, same retained stream and disjoint
        // input/output/scratch survive completion; native repeats ABI/range checks.
        unsafe {CudaRuntimeError::from_result("cuda_qpn_source_pair_submit_v1",
            augur_cuda_nvfp4_sm70_qpn_source_pair_submit_v1(self.stream.device_ordinal(),self.stream.raw()?,&self.raw,
                mode,splits,partial,self.workspace_bytes() as u64))}
    }
}
impl Drop for CudaQpnSourcePairProjection {fn drop(&mut self){let _=self.stream.synchronize();}}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_pair_schedules_have_distinct_closed_native_geometry() {
        let mut pairs=std::collections::BTreeSet::new();
        for s in Schedule::ALL {let (m,k)=s.geometry();assert!(m<=4);assert!(pairs.insert((m,k)));}
        assert_eq!(pairs.len(),11);
    }
}
