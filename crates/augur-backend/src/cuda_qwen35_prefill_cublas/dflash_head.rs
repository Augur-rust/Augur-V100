//! Draft-only B8 MMVF from the original retained BF16 vocabulary head.
use super::*;
/// Preflighted address/geometry plan, not an owner of head/input/output buffers.
///
/// Current `bind`/`submit` are safe functions but store only raw descriptors.
/// The surrounding target/draft owner must retain the original allocations,
/// prevent overlapping access, and establish completion before mutation/free.
/// Geometry and stream-owner checks do not encode those lifetime obligations;
/// see GAP-VIEW in `docs/UNSAFE_TRUST_BOUNDARIES.md` before reusing this as a
/// general-purpose library API. This annotation does not close the API gap.
pub struct CudaDflashBf16HeadV1 {
    raw: augur_cuda_sys::AugurCudaBf16F32MatvecArgsV1,
    ordinal: i32, stream_owner: usize,
    kernel: CudaDflashBf16HeadKernel,
}
/// Arithmetic schedule for the B8 shared head. Both read the original BF16
/// weights and accumulate exact FP32 products in FP32; they differ only in
/// summation order and memory schedule.
#[derive(Clone,Copy,Debug,Default,Eq,PartialEq)]
pub enum CudaDflashBf16HeadKernel {
    /// llama.cpp-derived MMVF<8>: one CTA per vocabulary row.
    #[default]
    LlamaMmvf8,
    /// Augur register-blocked FP32: 4 rows per warp, 16-byte weight vectors.
    F32Rw4,
}
impl CudaDflashBf16HeadKernel {
    pub fn label(self)->&'static str {match self {Self::LlamaMmvf8=>"llama_mmvf8_bf16_f32_v1",Self::F32Rw4=>"augur_f32_rw4_bf16_f32_v1"}}
}
impl CudaDflashBf16HeadV1 {
    pub fn bind(stream:&OwnedCudaStream,weight:CudaBf16WeightViewV1,input:CudaF32View,output:CudaF32View)->Result<Self,CudaRuntimeError> {
        Self::bind_kernel(stream,weight,input,output,CudaDflashBf16HeadKernel::LlamaMmvf8)
    }
    pub fn kernel(&self)->CudaDflashBf16HeadKernel {self.kernel}
    pub fn bind_kernel(stream:&OwnedCudaStream,weight:CudaBf16WeightViewV1,input:CudaF32View,output:CudaF32View,kernel:CudaDflashBf16HeadKernel)->Result<Self,CudaRuntimeError> {
        if weight.rows()!=248320 || weight.cols()!=5120 || input.element_count()!=8*5120 || output.element_count()!=8*248320 {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let views=[weight.buffer_view(),input.buffer_view(),output.buffer_view()];
        validate_disjoint_views(stream.device_ordinal(),&views)?;
        let mut raw=augur_cuda_sys::AugurCudaBf16F32MatvecArgsV1::new();
        raw.weight=views[0].checked_address(OPERATION)?;raw.weight_len_bytes=views[0].len_bytes() as u64;
        raw.input=views[1].checked_address(OPERATION)?;raw.input_len_bytes=views[1].len_bytes() as u64;
        raw.output=views[2].checked_address(OPERATION)?;raw.output_len_bytes=views[2].len_bytes() as u64;
        raw.rows=248320;raw.cols=5120;
        // SAFETY: exact immutable views; owner must retain buffers through completion.
        unsafe {CudaRuntimeError::from_result(OPERATION,match kernel {
            CudaDflashBf16HeadKernel::LlamaMmvf8=>augur_cuda_sys::augur_cuda_dflash_llama_bf16_head_preflight_v1(stream.device_ordinal(),stream.raw()?,&raw),
            CudaDflashBf16HeadKernel::F32Rw4=>augur_cuda_sys::augur_cuda_dflash_bf16_head_f32_rw4_preflight_v1(stream.device_ordinal(),stream.raw()?,&raw),
        })?;}
        Ok(Self{raw,ordinal:stream.device_ordinal(),stream_owner:stream.owner_instance_key()?,kernel})
    }
    pub fn submit(&self,stream:&OwnedCudaStream)->Result<(),CudaRuntimeError> {
        if stream.device_ordinal()!=self.ordinal || stream.owner_instance_key()?!=self.stream_owner {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        // SAFETY: the argument struct and matched live stream handle are valid
        // for this call. Device allocation lifetime/access through completion
        // is supplied by the enclosing executor, not by this safe signature;
        // output must be exclusively writable (GAP-VIEW).
        unsafe {CudaRuntimeError::from_result(OPERATION,match self.kernel {
            CudaDflashBf16HeadKernel::LlamaMmvf8=>augur_cuda_sys::augur_cuda_dflash_llama_bf16_head_submit_v1(self.ordinal,stream.raw()?,&self.raw),
            CudaDflashBf16HeadKernel::F32Rw4=>augur_cuda_sys::augur_cuda_dflash_bf16_head_f32_rw4_submit_v1(self.ordinal,stream.raw()?,&self.raw),
        })}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cuda_device_allocation_snapshot,CudaBf16F32MatvecArgsV1};
    use augur_core::PhysicalLayoutId as L;
    fn fv(b:&OwnedCudaBuffer,offset:usize,count:usize)->CudaF32View {
        CudaF32View::new(b.view(offset,count*4).unwrap(),L::DEVICE_CONTIGUOUS_F32_CUDA_V1,count).unwrap()
    }
    #[test]
    #[ignore="exclusive SM70; full original-BF16 vocabulary geometry, synthetic values"]
    fn llama_b8_head_full_vocabulary_and_allocation_closure() {
        const H:usize=5120;const V:usize=248320;const B:usize=8;
        let before=cuda_device_allocation_snapshot();let mut timings=Vec::new();let mut max_error=0f64;
        {
            let stream=OwnedCudaStream::create(0).unwrap();let foreign=OwnedCudaStream::create(0).unwrap();
            let mut wb=vec![0xa5;V*H*2+512];
            for row in 0..V {for col in 0..H {
                let v=(((row*19+col*7)%255) as i32-127) as f32/128.0;
                wb[256+(row*H+col)*2..258+(row*H+col)*2].copy_from_slice(&((v.to_bits()>>16) as u16).to_le_bytes());
            }}
            let w=OwnedCudaBuffer::allocate(0,wb.len()).unwrap();w.copy_from_host_sync(&stream,&wb).unwrap();drop(wb);
            let xb:Vec<u8>=(0..B*H).flat_map(|i|((((i/H*31+i%H*13)%255) as i32-127) as f32/128.0).to_le_bytes()).collect();
            let x=OwnedCudaBuffer::allocate(0,xb.len()).unwrap();x.copy_from_host_sync(&stream,&xb).unwrap();
            let y=OwnedCudaBuffer::allocate(0,B*V*4+512).unwrap();y.copy_from_host_sync(&stream,&vec![0xa5;y.len_bytes()]).unwrap();
            let weight=CudaBf16WeightViewV1::new(w.view(256,V*H*2).unwrap(),L::DEVICE_CONTIGUOUS_BF16_CUDA_V1,V,H).unwrap();
            let op=CudaDflashBf16HeadV1::bind(&stream,weight,fv(&x,0,B*H),fv(&y,256,B*V)).unwrap();
            assert!(op.submit(&foreign).is_err());
            assert!(CudaDflashBf16HeadV1::bind(&stream,weight,fv(&y,256,B*H),fv(&y,256,B*V)).is_err());
            let old:Vec<_>=(0..B).map(|r|CudaBf16F32MatvecArgsV1::bind(&stream,weight,fv(&x,r*H*4,H),fv(&y,256+r*V*4,V)).unwrap()).collect();
            let hot=cuda_device_allocation_snapshot();op.submit(&stream).unwrap();stream.synchronize().unwrap();
            assert_eq!(cuda_device_allocation_snapshot(),hot);
            let mut candidate=vec![0;y.len_bytes()];y.copy_to_host_sync(&stream,&mut candidate).unwrap();
            assert!(candidate[..256].iter().chain(&candidate[256+B*V*4..]).all(|v|*v==0xa5));
            for p in &old {p.submit_for_plan(&stream).unwrap();}stream.synchronize().unwrap();
            let mut control=vec![0;y.len_bytes()];y.copy_to_host_sync(&stream,&mut control).unwrap();
            for (a,b) in candidate[256..256+B*V*4].chunks_exact(4).zip(control[256..256+B*V*4].chunks_exact(4)) {
                let a=f32::from_le_bytes(a.try_into().unwrap()) as f64;let b=f32::from_le_bytes(b.try_into().unwrap()) as f64;
                assert!(a.is_finite() && b.is_finite());max_error=max_error.max((a-b).abs());
                assert!((a-b).abs()<=1e-4+1e-5*b.abs());
            }
            for token in [0,3,7] {for row in [0,1,V/2,V-1] {
                let sum:i64=(0..H).map(|col|(((row*19+col*7)%255) as i64-127)*(((token*31+col*13)%255) as i64-127)).sum();
                let start=256+(token*V+row)*4;let got=f32::from_le_bytes(candidate[start..start+4].try_into().unwrap()) as f64;
                assert!((got-sum as f64/16384.0).abs()<=1e-4);
            }}
            for repeat in 0..3 {for new in if repeat%2==0 {[false,true]}else{[true,false]} {
                let start=std::time::Instant::now();
                if new {op.submit(&stream).unwrap();} else {for p in &old {p.submit_for_plan(&stream).unwrap();}}
                stream.synchronize().unwrap();timings.push(serde_json::json!({"llama_b8":new,"wall_s":start.elapsed().as_secs_f64()}));
            }}
            assert_eq!(cuda_device_allocation_snapshot(),hot);
        }
        let after=cuda_device_allocation_snapshot();assert_eq!(after.live_bytes,before.live_bytes);assert_eq!(after.live_allocations,before.live_allocations);
        let Some(path)=std::env::var_os("AUGUR_DFLASH_HEAD_OUTPUT") else {return};let path=std::path::PathBuf::from(path);
        std::fs::write(path,serde_json::to_vec_pretty(&serde_json::json!({"passed":true,"schema":"dflash-llama-b8-head-component-v1",
            "tokens":B,"rows":V,"cols":H,"synthetic_inputs":true,"all_logits_vs_scalar_max_error":max_error,
            "wall_samples":timings,"hot_allocations":0,"cleanup_live_bytes_delta":0,"HTTP_qualified":false})).unwrap()).unwrap();
    }

    /// Rw4 vs MMVF<8> and an FP64 host reference on full vocabulary geometry.
    /// Inputs carry outlier channels and sign/magnitude spread typical of a
    /// normalized residual stream; values are not exactly representable sums.
    #[test]
    #[ignore="exclusive SM70; ~2.6 GB device memory"]
    fn f32_rw4_head_matches_mmvf8_and_f64_reference() {
        const H:usize=5120;const V:usize=248320;const B:usize=8;
        let before=cuda_device_allocation_snapshot();
        {
            let stream=OwnedCudaStream::create(0).unwrap();
            let mut state=0x9e3779b97f4a7c15u64;
            let mut next=||{state^=state<<13;state^=state>>7;state^=state<<17;((state>>40) as f64/16777216.0)*2.0-1.0};
            let mut wb=vec![0u8;V*H*2];
            for v in wb.chunks_exact_mut(2) {v.copy_from_slice(&(((next()*0.05) as f32).to_bits()>>16).to_le_bytes()[..2]);}
            let wf=|row:usize,col:usize,wb:&[u8]|f32::from_bits((u16::from_le_bytes([wb[(row*H+col)*2],wb[(row*H+col)*2+1]]) as u32)<<16) as f64;
            let mut xs=vec![0f32;B*H];
            for (i,v) in xs.iter_mut().enumerate() {*v=(next()*1.7) as f32*if i%H%431==0 {40.0} else {1.0};}
            let w=OwnedCudaBuffer::allocate(0,wb.len()).unwrap();w.copy_from_host_sync(&stream,&wb).unwrap();
            let xb:Vec<u8>=xs.iter().flat_map(|v|v.to_le_bytes()).collect();
            let x=OwnedCudaBuffer::allocate(0,xb.len()).unwrap();x.copy_from_host_sync(&stream,&xb).unwrap();
            let y=OwnedCudaBuffer::allocate(0,B*V*4+512).unwrap();y.copy_from_host_sync(&stream,&vec![0xa5;y.len_bytes()]).unwrap();
            let weight=CudaBf16WeightViewV1::new(w.view(0,V*H*2).unwrap(),L::DEVICE_CONTIGUOUS_BF16_CUDA_V1,V,H).unwrap();
            let read=|y:&OwnedCudaBuffer|{let mut b=vec![0;y.len_bytes()];y.copy_to_host_sync(&stream,&mut b).unwrap();b};
            let rw4=CudaDflashBf16HeadV1::bind_kernel(&stream,weight,fv(&x,0,B*H),fv(&y,256,B*V),CudaDflashBf16HeadKernel::F32Rw4).unwrap();
            assert_eq!(rw4.kernel(),CudaDflashBf16HeadKernel::F32Rw4);
            // 16-byte vector loads: a 4-byte-aligned input base must be rejected at bind.
            let shifted=OwnedCudaBuffer::allocate(0,B*H*4+16).unwrap();
            assert!(CudaDflashBf16HeadV1::bind_kernel(&stream,weight,fv(&shifted,4,B*H),fv(&y,256,B*V),CudaDflashBf16HeadKernel::F32Rw4).is_err());
            let hot=cuda_device_allocation_snapshot();
            rw4.submit(&stream).unwrap();stream.synchronize().unwrap();let a=read(&y);
            assert_eq!(cuda_device_allocation_snapshot(),hot);
            assert!(a[..256].iter().chain(&a[256+B*V*4..]).all(|v|*v==0xa5),"rw4 wrote outside its output view");
            CudaDflashBf16HeadV1::bind(&stream,weight,fv(&x,0,B*H),fv(&y,256,B*V)).unwrap().submit(&stream).unwrap();stream.synchronize().unwrap();
            let b=read(&y);
            let at=|v:&[u8],t:usize,r:usize|{let s=256+(t*V+r)*4;f32::from_le_bytes(v[s..s+4].try_into().unwrap()) as f64};
            let mut max_vs_mmvf=0f64;
            for t in 0..B {for r in 0..V {let (p,q)=(at(&a,t,r),at(&b,t,r));assert!(p.is_finite());max_vs_mmvf=max_vs_mmvf.max((p-q).abs());}}
            assert!(max_vs_mmvf<=2e-4,"rw4 vs mmvf8 {max_vs_mmvf}");
            for t in 0..B {for r in (0..V).step_by(997).chain([V-4,V-3,V-2,V-1]) {
                let exact:f64=(0..H).map(|c|wf(r,c,&wb)*xs[t*H+c] as f64).sum();
                let norm:f64=(0..H).map(|c|(wf(r,c,&wb)*xs[t*H+c] as f64).abs()).sum();
                assert!((at(&a,t,r)-exact).abs()<=1e-6*norm+1e-6,"row {r} token {t}");
            }}
            for t in 0..B {
                let top=|v:&[u8]|{let mut i:Vec<usize>=(0..V).collect();i.select_nth_unstable_by(20,|&p,&q|at(v,t,q).total_cmp(&at(v,t,p)));let mut s=i[..20].to_vec();s.sort();s};
                assert_eq!(top(&a),top(&b),"top20 set token {t}");
            }
        }
        let after=cuda_device_allocation_snapshot();assert_eq!(after.live_bytes,before.live_bytes);assert_eq!(after.live_allocations,before.live_allocations);
    }
}
