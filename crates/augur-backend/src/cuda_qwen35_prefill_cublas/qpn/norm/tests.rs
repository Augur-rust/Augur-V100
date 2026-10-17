use super::*;
use crate::cuda_device_allocation_snapshot;
use augur_core::PhysicalLayoutId;
const K:usize=5120;
fn read(stream:&OwnedCudaStream,b:&OwnedCudaBuffer)->Vec<u8> {
    let mut bytes=vec![0;b.len_bytes()]; b.copy_to_host_sync(stream,&mut bytes).unwrap(); bytes
}
fn upload(stream:&OwnedCudaStream,values:impl Iterator<Item=f32>)->Rc<OwnedCudaBuffer> {
    let bytes:Vec<u8>=values.flat_map(|x|x.to_le_bytes()).collect();
    let b=Rc::new(OwnedCudaBuffer::allocate(0,bytes.len()).unwrap());
    b.copy_from_host_sync(stream,&bytes).unwrap();b
}
fn view(b:&OwnedCudaBuffer)->CudaF32View {
    CudaF32View::new(b.view(0,K*4).unwrap(),PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,K).unwrap()
}
#[test]
#[ignore="exclusive SM70; real shapes, owner/alias/reuse/bit-identity gates"]
fn real_input_norm_qpn_group_identity_cleanup() {
    let baseline=cuda_device_allocation_snapshot();
    for shapes in [vec![10240,6144],vec![12288,1024,1024],vec![17408,17408]] {
        let stream=Rc::new(OwnedCudaStream::create(0).unwrap());
        let plan=Rc::new(CudaQwen35PrefillCublasPlanV1::new(&stream,17408*16*16,K*2,1).unwrap());
        let x=upload(&stream,(0..K).map(|i|i as f32*0.01));
        let w=upload(&stream,(0..K).map(|i|0.3+(i%43) as f32*0.041));
        let y=upload(&stream,(0..K).map(|_|0.));
        let original=CudaRmsnormF32ArgsV1::bind(&stream,view(&x),view(&w),view(&y),1e-6).unwrap();
        let cached=original.with_exact_warp_tree5120().unwrap();
        let mut projections=Vec::new();
        let mut outputs=Vec::new();
        for (index,&rows) in shapes.iter().enumerate() {
            let mut wire=vec![0u8;rows*(K/64)*36];
            for block in 0..rows*(K/64) {
                for group in 0..4 { wire[block*36+group]=48+((block+group+index)%9) as u8; }
                for byte in 4..36 {wire[block*36+byte]=((block*7+byte*19+index*31)%256) as u8;}
            }
            let weight=Rc::new(OwnedCudaQwen35QpnWeightV1::upload_ggml(&stream,&wire,rows,K,0.75).unwrap());
            let output=upload(&stream,(0..rows).map(|_|0.));
            projections.push(CudaQwen35QpnRetainedProjectionV1::bind(Rc::clone(&plan),Rc::clone(&stream),
                weight,Rc::clone(&y),0,Rc::clone(&output),0,1).unwrap()
                .with_scale_placement(CudaQwen35QpnScalePlacement::ResidualWorkspaceShapeF64).unwrap());
            outputs.push(output);
        }
        assert!(CudaQwen35NormQpnGroupV1::bind(original,Rc::clone(&x),Rc::clone(&w),projections.clone()).is_err());
        assert!(CudaQwen35NormQpnGroupV1::bind(cached,Rc::clone(&w),Rc::clone(&x),projections.clone()).is_err());
        assert!(CudaQwen35NormQpnGroupV1::bind(cached,Rc::clone(&x),Rc::clone(&w),vec![]).is_err());
        let mut wrong=projections.clone();wrong[1]=wrong[0].clone();
        assert!(CudaQwen35NormQpnGroupV1::bind(cached,Rc::clone(&x),Rc::clone(&w),wrong).is_err());
        let mut wrong=projections.clone();wrong[1]=wrong[1].clone()
            .with_scale_placement(CudaQwen35QpnScalePlacement::ResidualWorkspaceCtaFoldF64).unwrap();
        assert!(CudaQwen35NormQpnGroupV1::bind(cached,Rc::clone(&x),Rc::clone(&w),wrong).is_err());
        let group=CudaQwen35NormQpnGroupV1::bind(cached,Rc::clone(&x),Rc::clone(&w),projections.clone()).unwrap();
        let mut fingerprints=Vec::new();
        for (iteration,scale) in [0.0,1e-20,0.1,1.0,13.0,1e8].iter().enumerate() {
            let bytes:Vec<u8>=(0..K).flat_map(|i|(((i*29+iteration*17)%503) as f32-251.)
                .mul_add(*scale/73.0031,0.0).to_le_bytes()).collect();
            x.copy_from_host_sync(&stream,&bytes).unwrap();
            original.submit_for_plan(&stream).unwrap();
            for projection in &projections {projection.submit_for_plan().unwrap();}
            let reference=read(&stream,&y);
            let staged=read(&stream,&plan.decoded_input_f16);
            let expected:Vec<_>=outputs.iter().map(|o|read(&stream,o)).collect();
            // Poison BOTH reusable arenas between runs at the same input pointer.
            y.copy_from_host_sync(&stream,&vec![0x55;K*4]).unwrap();
            plan.decoded_input_f16.copy_from_host_sync(&stream,&vec![0x6b;plan.decoded_input_f16.len_bytes()]).unwrap();
            let before=cuda_device_allocation_snapshot();
            group.submit_for_plan().unwrap();stream.synchronize().unwrap();
            let after=cuda_device_allocation_snapshot();
            assert_eq!(before.successful_allocations,after.successful_allocations);
            assert_eq!(reference,read(&stream,&y),"F32 norm changed");
            assert_eq!(staged,read(&stream,&plan.decoded_input_f16),"hi/lo stage changed");
            for (o,e) in outputs.iter().zip(&expected) {assert_eq!(&read(&stream,o),e,"QPN output changed");}
            fingerprints.push(reference);
        }
        assert!(fingerprints.windows(2).all(|p|p[0]!=p[1]),"test did not refresh input");
        // Malformed final consumers must reject BEFORE norm or any earlier output.
        let sentinel=vec![0x5a;K*4];y.copy_from_host_sync(&stream,&sentinel).unwrap();
        let old_outputs:Vec<_>=outputs.iter().map(|o|read(&stream,o)).collect();
        for case in 0..12 {
            let mut norm=group.norm;let mut raw=group.raw.clone();let last=raw.len()-1;let mut count=raw.len() as u32;
            match case {
                0=>raw[last].output_f32=norm.input,
                1=>raw[last].output_f32=raw[0].output_f32,
                2=>raw[last].weight=norm.weight,
                3=>norm.output=norm.input,
                4=>norm.elements=5119,
                5=>norm.epsilon_bits=f32::NAN.to_bits(),
                6=>raw[last].numerical_profile=7,
                7=>raw[last].input_f32=norm.weight,
                8=>raw[last].tokens=2,
                9=>count=1,
                10=>count=4,
                _=>raw[last].output_f32=u64::MAX-15,
            }
            let rejected=unsafe {augur_cuda_sys::augur_cuda_qwen35_qpn_norm_group_prevalidated_submit_v1(
                0,stream.raw().unwrap(),plan.raw,&norm,raw.as_ptr(),count)};
            assert!(CudaRuntimeError::from_result("reject before enqueue",rejected).is_err());
            assert_eq!(read(&stream,&y),sentinel);
            for (o,e) in outputs.iter().zip(&old_outputs) {assert_eq!(&read(&stream,o),e);}
        }
        // Reject a same-device but different native stream in the native ABI.
        let other=OwnedCudaStream::create(0).unwrap();
        let rejected=unsafe {augur_cuda_sys::augur_cuda_qwen35_qpn_norm_group_preflight_v1(
            0,other.raw().unwrap(),plan.raw,&group.norm,group.raw.as_ptr(),group.raw.len() as u32)};
        assert!(CudaRuntimeError::from_result("wrong stream",rejected).is_err());
        let weak=Rc::downgrade(&x);let weak_w=Rc::downgrade(&w);
        drop(x);drop(w);drop(projections);drop(plan);drop(stream);drop(outputs);drop(y);
        assert!(weak.upgrade().is_some() && weak_w.upgrade().is_some());
        group.submit_for_plan().unwrap(); // only program owns ALL required operands
        drop(group); // must drain before releasing the sources
        assert!(weak.upgrade().is_none() && weak_w.upgrade().is_none());
    }
    let after=cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes,baseline.live_bytes);
    assert_eq!(after.live_allocations,baseline.live_allocations);
    assert_eq!(after.successful_allocations-baseline.successful_allocations,
               after.successful_frees-baseline.successful_frees);
}
