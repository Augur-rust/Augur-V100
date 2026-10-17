//! Shared BF16 vocabulary head rows for the DFlash draft and the B8 verifier:
//! original owner, FP32 accumulation. `AUGUR_SHARED_HEAD_KERNEL` selects the
//! B8 schedule (`llama` = MMVF<8> default, `rw4` = register-blocked FP32).
use super::*;
use augur_backend::cuda_qwen35_prefill_cublas::CudaDflashBf16HeadKernel;
pub(crate) fn shared_head_kernel()->Result<CudaDflashBf16HeadKernel> {
    match std::env::var("AUGUR_SHARED_HEAD_KERNEL").as_deref() {
        Err(std::env::VarError::NotPresent)|Ok("llama")=>Ok(CudaDflashBf16HeadKernel::LlamaMmvf8),
        Ok("rw4")=>Ok(CudaDflashBf16HeadKernel::F32Rw4),
        _=>Err("AUGUR_SHARED_HEAD_KERNEL must be llama or rw4".into()),
    }
}
enum HeadRowsOp {
    B8(augur_backend::cuda_qwen35_prefill_cublas::CudaDflashBf16HeadV1),
    Rows(CudaBf16F32MatvecRowsArgsV1),
}
pub(crate) struct CheckpointSharedBf16HeadRows {
    op: HeadRowsOp,
    stream: Rc<OwnedCudaStream>,
    _weight: Rc<CheckpointScalarCudaWeight>,
    _input: Rc<OwnedCudaBuffer>, _output: Rc<OwnedCudaBuffer>,
}
impl CheckpointSharedBf16HeadRows {
    pub(crate) fn kernel_label(&self)->&'static str {
        match &self.op {HeadRowsOp::B8(op)=>op.kernel().label(),HeadRowsOp::Rows(_)=>"bf16_f32_matvec_rows_v1"}
    }
    pub(crate) fn submit(&self)->Result<()> {
        match &self.op {
            HeadRowsOp::B8(op) => op.submit(&self.stream)?,
            HeadRowsOp::Rows(op) => op.submit_for_plan(&self.stream)?,
        }
        Ok(())
    }
}
impl Drop for CheckpointSharedBf16HeadRows {
    fn drop(&mut self) { let _=self.stream.synchronize(); }
}
impl CheckpointScalarCudaProjection {
    pub(crate) fn rebind_shared_bf16_head_rows(&self,stream:&Rc<OwnedCudaStream>,
        input:Rc<OwnedCudaBuffer>,output:Rc<OwnedCudaBuffer>,rows:usize)->Result<CheckpointSharedBf16HeadRows> {
        if !(1..=8).contains(&rows) {
            return Err("shared BF16 head row width must be in 1..=8".into());
        }
        if !Rc::ptr_eq(stream,&self.weight.stream) || self.weight.spec.rows!=248320 || self.weight.spec.cols!=5120 {
            return Err("shared BF16 head requires the same retained vocabulary owner".into());
        }
        let Storage::Bf16(weight)=&self.weight.storage else {return Err("shared BF16 head requires original BF16".into())};
        let view=CudaBf16WeightViewV1::new(weight.view(0,weight.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,248320,5120)?;
        let op = if rows == 8 {
            HeadRowsOp::B8(augur_backend::cuda_qwen35_prefill_cublas::CudaDflashBf16HeadV1::bind_kernel(
                stream, view, f32_view(&input, 0, rows * 5120)?,
                f32_view(&output, 0, rows * 248320)?, shared_head_kernel()?)?)
        } else {
            HeadRowsOp::Rows(CudaBf16F32MatvecRowsArgsV1::bind(
                stream, view,
                f32_view(&input, 0, rows * 5120)?,
                f32_view(&output, 0, rows * 248320)?, rows)?)
        };
        Ok(CheckpointSharedBf16HeadRows{op,stream:Rc::clone(stream),_weight:Rc::clone(&self.weight),_input:input,_output:output})
    }
    pub(crate) fn rebind_shared_bf16_head_b8(&self,stream:&Rc<OwnedCudaStream>,
        input:Rc<OwnedCudaBuffer>,output:Rc<OwnedCudaBuffer>)->Result<CheckpointSharedBf16HeadRows> {
        self.rebind_shared_bf16_head_rows(stream,input,output,8)
    }
}
