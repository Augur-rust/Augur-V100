//! DFlash draft Q8_0 projections: llama's MMVQ/MMQ/FP16 GEMM implementations
//! with shared plan-owned scratch, plus the Augur B8 direct kernel (no scratch).
//! No submit-time allocations on either path.
use super::*;

fn dflash_projection_shape(rows: usize, cols: usize, tokens: usize) -> bool {
    (1..=512).contains(&tokens) &&
        ((cols == 5120 && matches!(rows, 256 | 1024 | 1280 | 4096 | 17408))
        || (rows == 5120 && matches!(cols, 4096 | 17408 | 25600)))
}
impl CudaQwen35PrefillCublasPlanV1 {
    /// Caller retains buffers through completion and serializes plan users.
    /// Native preflight checks all ranges, including shared plan scratch.
    pub fn submit_dflash_llama_projection(
        &self, stream: &OwnedCudaStream, weight: CudaQ8_0WeightViewV1,
        input: CudaF32View, output: CudaF32View, tokens: usize,
    ) -> Result<(), CudaRuntimeError> {
        self.submit_dflash_projection(stream,weight,input,output,tokens,false)
    }
    /// Augur B8 direct kernel: identical admission and ranges, exactly 8 tokens,
    /// original Q8_0 blocks in place, FP32 activations, no plan scratch use.
    pub fn submit_dflash_q8_b8_direct(
        &self, stream: &OwnedCudaStream, weight: CudaQ8_0WeightViewV1,
        input: CudaF32View, output: CudaF32View,
    ) -> Result<(), CudaRuntimeError> {
        self.submit_dflash_projection(stream,weight,input,output,8,true)
    }
    fn submit_dflash_projection(
        &self, stream: &OwnedCudaStream, weight: CudaQ8_0WeightViewV1,
        input: CudaF32View, output: CudaF32View, tokens: usize, direct: bool,
    ) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        let (rows, cols) = (weight.rows() as usize, weight.cols() as usize);
        if !dflash_projection_shape(rows, cols, tokens) || tokens > self.max_tokens
            || input.element_count() != tokens * cols || output.element_count() != tokens * rows
            || input.buffer_view().len_bytes() != tokens * cols * 4
            || output.buffer_view().len_bytes() != tokens * rows * 4 {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        validate_disjoint_views(self.device_ordinal,
            &[weight.buffer_view(), input.buffer_view(), output.buffer_view()])?;
        let raw = AugurCudaQwen35PrefillF32ProjectionArgsV1 {
            abi_version: 1, struct_size: core::mem::size_of::<AugurCudaQwen35PrefillF32ProjectionArgsV1>() as u32,
            weight_format: AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1,
            tokens: tokens as u32, rows: rows as u32, cols: cols as u32,
            weight: weight.buffer_view().checked_address(OPERATION)?,
            weight_len_bytes: weight.buffer_view().len_bytes() as u64,
            input_f32: input.buffer_view().checked_address(OPERATION)?,
            input_len_bytes: input.buffer_view().len_bytes() as u64,
            output_f32: output.buffer_view().checked_address(OPERATION)?,
            output_len_bytes: output.buffer_view().len_bytes() as u64,
            tensor_scale: 1.0, reserved: 0,
        };
        // SAFETY: exact live views, immutable weights, same-stream scratch.
        unsafe {
            if direct {
                CudaRuntimeError::from_result(OPERATION,
                    augur_cuda_sys::augur_cuda_dflash_q8_b8_direct_preflight_v1(
                        self.device_ordinal, stream.raw()?, self.raw, &raw))?;
                return CudaRuntimeError::from_result(OPERATION,
                    augur_cuda_sys::augur_cuda_dflash_q8_b8_direct_submit_v1(
                        self.device_ordinal, stream.raw()?, self.raw, &raw));
            }
            CudaRuntimeError::from_result(OPERATION,
                augur_cuda_sys::augur_cuda_dflash_llama_projection_preflight_v1(
                    self.device_ordinal, stream.raw()?, self.raw, &raw))?;
            CudaRuntimeError::from_result(OPERATION,
                augur_cuda_sys::augur_cuda_dflash_llama_projection_submit_v1(
                    self.device_ordinal, stream.raw()?, self.raw, &raw))
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn draft_shape_admission_is_separate_from_target_and_bounded() {
        for (rows, cols) in [(5120,25600),(1024,5120),(4096,5120),(5120,4096),
            (1280,5120),(17408,5120),(5120,17408),(256,5120)] {
            for tokens in [1,7,8,9,63,64,511,512] { assert!(dflash_projection_shape(rows,cols,tokens)); }
            for tokens in [0,513,usize::MAX] { assert!(!dflash_projection_shape(rows,cols,tokens)); }
        }
        for (rows,cols) in [(248320,5120),(5120,6144),(48,5120),(0,5120),(5120,0)] {
            assert!(!dflash_projection_shape(rows,cols,8));
        }
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use crate::{cuda_device_allocation_snapshot, cuda_dflash_context_rows::CudaDflashContextRowsV1};
    use augur_core::PhysicalLayoutId as L;
    use serde_json::json;
    struct Guard { owner: OwnedCudaBuffer, bytes: usize }
    impl Guard {
        fn new(stream: &OwnedCudaStream, bytes: &[u8]) -> Self {
            let owner=OwnedCudaBuffer::allocate(0,bytes.len()+512).unwrap();
            let mut all=vec![0xa5;bytes.len()+512];all[256..256+bytes.len()].copy_from_slice(bytes);
            owner.copy_from_host_sync(stream,&all).unwrap();Self{owner,bytes:bytes.len()}
        }
        fn view(&self) -> CudaBufferView { self.owner.view(256,self.bytes).unwrap() }
        fn floats(&self) -> CudaF32View { CudaF32View::new(self.view(),L::DEVICE_CONTIGUOUS_F32_CUDA_V1,self.bytes/4).unwrap() }
        fn read(&self,stream:&OwnedCudaStream)->Vec<u8>{
            let mut all=vec![0;self.bytes+512];self.owner.copy_to_host_sync(stream,&mut all).unwrap();
            assert!(all[..256].iter().chain(&all[256+self.bytes..]).all(|v|*v==0xa5));
            all[256..256+self.bytes].to_vec()
        }
    }
    fn wq(row:usize,col:usize)->i32 { ((row*37+col*17+col/32*11)%255) as i32-127 }
    fn xq(token:usize,col:usize)->i32 { if col%32==0 {127} else {((token*31+col*13)%255) as i32-127} }
    #[test]
    #[ignore = "exclusive SM70; all draft shapes and dispatch edges, synthetic component only"]
    fn llama_draft_projection_shapes_outputs_guards_and_cleanup() {
        let before=cuda_device_allocation_snapshot();let mut cases=Vec::new();
        {
            let stream=OwnedCudaStream::create(0).unwrap();let foreign=OwnedCudaStream::create(0).unwrap();
            let plan=CudaQwen35PrefillCublasPlanV1::new(&stream,5120*25600,25600,512).unwrap();
            for (rows,cols) in [(5120,25600),(1024,5120),(4096,5120),(5120,4096),
                (1280,5120),(17408,5120),(5120,17408),(256,5120)] {
                let mut wb=Vec::with_capacity(rows*cols/32*34);
                for row in 0..rows {for block in 0..cols/32 {
                    wb.extend_from_slice(&0x2000u16.to_le_bytes()); // exact half1/128
                    for j in 0..32 {wb.push(wq(row,block*32+j) as i8 as u8);}
                }}
                let w=Guard::new(&stream,&wb);
                let weight=CudaQ8_0WeightViewV1::new(w.view(),L::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,rows,cols).unwrap();
                for tokens in [1,7,8,9,63,64,511,512] {
                    // Every block has max127/128: activation scale exactly1/128.
                    // Independent integer oracle tests layout/dispatch, not
                    // arbitrary-real rounding or product numerical qualification.
                    let xb:Vec<u8>=(0..tokens*cols).flat_map(|i|(xq(i/cols,i%cols) as f32/128.0).to_le_bytes()).collect();
                    let x=Guard::new(&stream,&xb);let y=Guard::new(&stream,&vec![0xff;tokens*rows*4]);
                    assert!(plan.submit_dflash_llama_projection(&foreign,weight,x.floats(),y.floats(),tokens).is_err());
                    assert!(plan.submit_dflash_llama_projection(&stream,weight,x.floats(),x.floats(),tokens).is_err());
                    let hot=cuda_device_allocation_snapshot();
                    plan.submit_dflash_llama_projection(&stream,weight,x.floats(),y.floats(),tokens).unwrap();stream.synchronize().unwrap();
                    assert_eq!(cuda_device_allocation_snapshot(),hot);
                    let raw=y.read(&stream);let values:Vec<f32>=raw.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
                    assert!(values.iter().all(|v|v.is_finite()));let mut max_error=0.0f64;
                    for token in [0,tokens/2,tokens-1] {for row in [0,1,rows/3,rows/2,rows-2,rows-1] {
                        let sum:i64=(0..cols).map(|col|i64::from(wq(row,col))*i64::from(xq(token,col))).sum();
                        let expected=sum as f64/16384.0;let actual=f64::from(values[token*rows+row]);
                        max_error=max_error.max((actual-expected).abs());
                        assert!((actual-expected).abs() <= 1e-4+expected.abs()*1e-5,
                            "rows={rows} cols={cols} tokens={tokens} at {token},{row}: {actual} != {expected}");
                    }}
                    assert_eq!(x.read(&stream),xb);
                    let mut old_wall=Vec::new();let mut new_wall=Vec::new();
                    if matches!(tokens,8|512) {
                        let old=CudaDflashContextRowsV1::projection_tile4(&stream,weight,x.floats(),y.floats(),tokens).unwrap();
                        old.submit_for_plan(&stream).unwrap();stream.synchronize().unwrap();
                        for repeat in 0..3 {for new in if repeat%2==0 {[false,true]}else{[true,false]} {
                            let start=std::time::Instant::now();
                            if new {plan.submit_dflash_llama_projection(&stream,weight,x.floats(),y.floats(),tokens).unwrap();}
                            else {old.submit_for_plan(&stream).unwrap();}
                            stream.synchronize().unwrap();
                            if new {new_wall.push(start.elapsed().as_secs_f64());}else{old_wall.push(start.elapsed().as_secs_f64());}
                        }}
                        assert_eq!(cuda_device_allocation_snapshot(),hot);
                    }
                    cases.push(json!({"rows":rows,"cols":cols,"tokens":tokens,"sampled_absolute_error":max_error,
                        "all_outputs_finite":true,"sampled_integer_oracle":true,"guarded":true,
                        "hot_allocations":0,"hot_frees":0,"f32_tile4_wall_s":old_wall,"llama_wall_s":new_wall}));
                }
                assert_eq!(w.read(&stream),wb);
            }
        }
        let after=cuda_device_allocation_snapshot();assert_eq!(after.live_bytes,before.live_bytes);
        assert_eq!(after.live_allocations,before.live_allocations);
        let output=std::path::PathBuf::from(std::env::var_os("AUGUR_DFLASH_PROJECTION_OUTPUT").unwrap());
        std::fs::write(output,serde_json::to_vec_pretty(&json!({"schema":"dflash-llama-projection-component-v1",
            "passed":true,"synthetic_inputs":true,"HTTP_qualified":false,"cases":cases,
            "cleanup_live_bytes_delta":0,"cleanup_live_allocations_delta":0})).unwrap()).unwrap();
    }

    /// B8 direct kernel on every draft block shape: real-valued activations with
    /// outlier channels and varied FP16 block scales, FP64 reference on every
    /// output, guard bytes, rejection of non-8 widths and misaligned input.
    #[test]
    #[ignore = "exclusive SM70; all draft block shapes, synthetic values"]
    fn dflash_q8_b8_direct_matches_f64_on_all_block_shapes() {
        let before=cuda_device_allocation_snapshot();
        {
            let stream=OwnedCudaStream::create(0).unwrap();
            let plan=CudaQwen35PrefillCublasPlanV1::new(&stream,5120*25600,25600,512).unwrap();
            let mut state=0x2545f4914f6cdd1du64;
            let mut next=||{state^=state<<13;state^=state>>7;state^=state<<17;(state>>11) as f64/(1u64<<53) as f64};
            for (rows,cols) in [(1280,5120),(4096,5120),(1024,5120),(5120,4096),(17408,5120),(5120,17408)] {
                let mut wb=Vec::with_capacity(rows*cols/32*34);let mut wf=vec![0f64;rows*cols];
                for row in 0..rows {for block in 0..cols/32 {
                    // Normal FP16 scale in [2^-11, 2^-8): exponent field 4..6, random 10-bit mantissa.
                    let (e,m)=(4+(next()*3.0) as u16,(next()*1024.0) as u16);
                    wb.extend_from_slice(&((e<<10)|m).to_le_bytes());
                    let d=(1.0+m as f64/1024.0)*2f64.powi(e as i32-15);
                    for j in 0..32 {let q=(next()*255.0) as i32-127;wb.push(q as i8 as u8);wf[row*cols+block*32+j]=d*q as f64;}
                }}
                let xs:Vec<f32>=(0..8*cols).map(|i|((next()*2.0-1.0)*1.5*if i%cols%389==5 {25.0} else {1.0}) as f32).collect();
                let xb:Vec<u8>=xs.iter().flat_map(|v|v.to_le_bytes()).collect();
                let w=Guard::new(&stream,&wb);let x=Guard::new(&stream,&xb);let y=Guard::new(&stream,&vec![0xff;8*rows*4]);
                let weight=CudaQ8_0WeightViewV1::new(w.view(),L::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,rows,cols).unwrap();
                let hot=cuda_device_allocation_snapshot();
                plan.submit_dflash_q8_b8_direct(&stream,weight,x.floats(),y.floats()).unwrap();stream.synchronize().unwrap();
                assert_eq!(cuda_device_allocation_snapshot(),hot);
                let got:Vec<f64>=y.read(&stream).chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap()) as f64).collect();
                for t in 0..8 {for r in 0..rows {
                    let (mut exact,mut mass)=(0f64,0f64);
                    for c in 0..cols {let p=wf[r*cols+c]*xs[t*cols+c] as f64;exact+=p;mass+=p.abs();}
                    let e=(got[t*rows+r]-exact).abs();
                    assert!(e<=2e-6*mass+1e-6,"rows={rows} cols={cols} token={t} row={r}: {} vs {exact}",got[t*rows+r]);
                }}
                assert_eq!(x.read(&stream),xb);assert_eq!(w.read(&stream),wb);
                // Width and alignment admission fail closed.
                let x7=CudaF32View::new(x.owner.view(256,7*cols*4).unwrap(),L::DEVICE_CONTIGUOUS_F32_CUDA_V1,7*cols).unwrap();
                let y7=CudaF32View::new(y.owner.view(256,7*rows*4).unwrap(),L::DEVICE_CONTIGUOUS_F32_CUDA_V1,7*rows).unwrap();
                assert!(plan.submit_dflash_q8_b8_direct(&stream,weight,x7,y7).is_err());
                let xs4=CudaF32View::new(x.owner.view(260,8*cols*4).unwrap(),L::DEVICE_CONTIGUOUS_F32_CUDA_V1,8*cols).unwrap();
                assert!(plan.submit_dflash_q8_b8_direct(&stream,weight,xs4,y.floats()).is_err());
            }
        }
        let after=cuda_device_allocation_snapshot();assert_eq!(after.live_bytes,before.live_bytes);
        assert_eq!(after.live_allocations,before.live_allocations);
    }
}
