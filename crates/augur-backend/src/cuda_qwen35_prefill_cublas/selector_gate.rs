//! Opt-in exact B8 F32 selector gate. Reuses the existing serial cuBLAS plan.
use super::*;
impl CudaQwen35PrefillCublasPlanV1 {
    pub fn submit_dflash_selector_gate_f32(&self, stream:&OwnedCudaStream,
        weight:CudaF32View,input:CudaF32View,output:CudaF32View)->Result<(),CudaRuntimeError> {
        self.submit_dflash_selector_gate_with_accumulation(stream,weight,input,output,false)
    }
    pub fn submit_dflash_selector_gate_with_accumulation(&self,stream:&OwnedCudaStream,
        weight:CudaF32View,input:CudaF32View,output:CudaF32View,fp64:bool)->Result<(),CudaRuntimeError> {
        self.require_stream(stream)?;
        if self.max_tokens<8 || weight.element_count()!=256*5120
            || input.element_count()!=8*5120 || output.element_count()!=8*256 {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let views=[weight.buffer_view(),input.buffer_view(),output.buffer_view()];
        validate_disjoint_views(self.device_ordinal,&views)?;
        let raw=AugurCudaQwen35PrefillF32ProjectionArgsV1 {
            abi_version:1,struct_size:core::mem::size_of::<AugurCudaQwen35PrefillF32ProjectionArgsV1>() as u32,
            weight_format:if fp64 {augur_cuda_sys::AUGUR_CUDA_DFLASH_SELECTOR_GATE_F64_ACCUM_V1}
                else {augur_cuda_sys::AUGUR_CUDA_DFLASH_SELECTOR_GATE_WEIGHT_F32_V1},
            tokens:8,rows:256,cols:5120,
            weight:views[0].checked_address(OPERATION)?,weight_len_bytes:views[0].len_bytes() as u64,
            input_f32:views[1].checked_address(OPERATION)?,input_len_bytes:views[1].len_bytes() as u64,
            output_f32:views[2].checked_address(OPERATION)?,output_len_bytes:views[2].len_bytes() as u64,
            tensor_scale:1.0,reserved:0,
        };
        // SAFETY: exact disjoint live views, same stream/plan and retained owners.
        unsafe {
            CudaRuntimeError::from_result(OPERATION,augur_cuda_sys::augur_cuda_dflash_selector_gate_f32_preflight_v1(self.device_ordinal,stream.raw()?,self.raw,&raw))?;
            CudaRuntimeError::from_result(OPERATION,augur_cuda_sys::augur_cuda_dflash_selector_gate_f32_submit_v1(self.device_ordinal,stream.raw()?,self.raw,&raw))
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_device_allocation_snapshot;
    use augur_core::PhysicalLayoutId as L;
    fn fv(b:&OwnedCudaBuffer,offset:usize,n:usize)->CudaF32View {
        CudaF32View::new(b.view(offset,n*4).unwrap(),L::DEVICE_CONTIGUOUS_F32_CUDA_V1,n).unwrap()
    }
    #[test]
    #[ignore="exclusive SM70; FP32 gate component and allocation closure"]
    fn f32_gate_all_rows_guards_and_cleanup() {
        let before=cuda_device_allocation_snapshot();let mut max_error=[0f64;2];
        {
            let stream=OwnedCudaStream::create(0).unwrap();let foreign=OwnedCudaStream::create(0).unwrap();
            let plan=CudaQwen35PrefillCublasPlanV1::new(&stream,256*5120,5120,8).unwrap();
            let wv:Vec<f32>=(0..256*5120).map(|i|(((i/5120*19+i%5120*7)%255) as i32-127) as f32/127.0).collect();
            let xv:Vec<f32>=(0..8*5120).map(|i|(((i/5120*31+i%5120*13)%255) as i32-127) as f32/113.0).collect();
            let upload=|values:&[f32]| {
                let mut bytes=vec![0xa5;values.len()*4+512];
                for (v,b) in values.iter().zip(bytes[256..256+values.len()*4].chunks_exact_mut(4)){b.copy_from_slice(&v.to_le_bytes());}
                let owner=OwnedCudaBuffer::allocate(0,bytes.len()).unwrap();owner.copy_from_host_sync(&stream,&bytes).unwrap();owner
            };
            let w=upload(&wv);let x=upload(&xv);let y=upload(&vec![0.;8*256]);
            for (lane,fp64) in [false,true].into_iter().enumerate() {
            let run=||plan.submit_dflash_selector_gate_with_accumulation(&stream,fv(&w,256,256*5120),fv(&x,256,8*5120),fv(&y,256,8*256),fp64);
            assert!(plan.submit_dflash_selector_gate_with_accumulation(&foreign,fv(&w,256,256*5120),fv(&x,256,8*5120),fv(&y,256,8*256),fp64).is_err());
            assert!(plan.submit_dflash_selector_gate_with_accumulation(&stream,fv(&w,256,256*5120),fv(&x,256,7*5120),fv(&y,256,8*256),fp64).is_err());
            assert!(plan.submit_dflash_selector_gate_with_accumulation(&stream,fv(&w,256,256*5120),fv(&x,256,8*5120),fv(&x,256,8*256),fp64).is_err());
            run().unwrap();stream.synchronize().unwrap();let hot=cuda_device_allocation_snapshot();
            for _ in 0..3 {run().unwrap();}stream.synchronize().unwrap();assert_eq!(cuda_device_allocation_snapshot(),hot);
            let mut bytes=vec![0;y.len_bytes()];y.copy_to_host_sync(&stream,&mut bytes).unwrap();
            assert!(bytes[..256].iter().chain(&bytes[256+8*256*4..]).all(|v|*v==0xa5));
            for token in 0..8 {for row in 0..256 {
                let expected:f64=(0..5120).map(|col|wv[row*5120+col] as f64*xv[token*5120+col] as f64).sum();
                let i=256+(token*256+row)*4;let got=f32::from_le_bytes(bytes[i..i+4].try_into().unwrap()) as f64;
                assert!(got.is_finite());max_error[lane]=max_error[lane].max((got-expected).abs());
                let bound=if fp64 {1e-8+expected.abs()*6e-8}else{1e-4+expected.abs()*1e-5};
                assert!((got-expected).abs()<=bound,"lane={lane} row={row} token={token} got={got} expected={expected}");
            }}
            }
        }
        let after=cuda_device_allocation_snapshot();assert_eq!(after.live_bytes,before.live_bytes);assert_eq!(after.live_allocations,before.live_allocations);
        let output=std::env::var_os("AUGUR_DFLASH_GATE_OUTPUT").unwrap();
        std::fs::write(output,serde_json::to_vec_pretty(&serde_json::json!({"passed":true,"all_2048_outputs_checked":true,
            "fp32_max_absolute_error":max_error[0],"fp64_max_absolute_error":max_error[1],"hot_allocations":0,"cleanup_live_bytes_delta":0,"HTTP_qualified":false})).unwrap()).unwrap();
    }
}
