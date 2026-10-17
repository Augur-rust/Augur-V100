//! Bit-exact B1 SiLU+stage+down, including native rejection-before-enqueue.
use super::*;
fn read(s:&OwnedCudaStream,b:&OwnedCudaBuffer)->Vec<u8>{
    let mut out=vec![0;b.len_bytes()];b.copy_to_host_sync(s,&mut out).unwrap();out
}
fn view(b:&OwnedCudaBuffer,n:usize)->CudaF32View{
    CudaF32View::new(b.view(0,n*4).unwrap(),augur_core::PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,n).unwrap()
}
#[test]
#[ignore="exclusive SM70; B1/profile8 activation/stage/down identity and alias guards"]
fn scalar_tail_staging_is_exact_and_rejects_bad_metadata(){
    const K:usize=17408;const N:usize=5120;
    let before=crate::cuda_device_allocation_snapshot();
    {
        let stream=Rc::new(OwnedCudaStream::create(0).unwrap());
        let plan=Rc::new(CudaQwen35PrefillCublasPlanV1::new(&stream,N*16*8,K*2,1).unwrap());
        let packed=(0..N*K/2).map(|i|((i*7+i/43)&255)as u8).collect::<Vec<_>>();
        let scales=(0..N*K/16).map(|i|((i*13+i/11)%127)as u8).collect::<Vec<_>>();
        let weight=Rc::new(OwnedCudaQwen35QpnWeightV1::upload_checkpoint_nvfp4(&stream,&packed,&scales,N,K,0.00015694754).unwrap());
        let gate=OwnedCudaBuffer::allocate(0,K*4).unwrap();let up=OwnedCudaBuffer::allocate(0,K*4).unwrap();
        let activation=Rc::new(OwnedCudaBuffer::allocate(0,K*4).unwrap());
        let output=Rc::new(OwnedCudaBuffer::allocate(0,N*4).unwrap());
        let silu=crate::CudaSiluMulF32ArgsV1::bind(&stream,view(&gate,K),view(&up,K),view(&activation,K)).unwrap();
        let down=CudaQwen35QpnRetainedProjectionV1::bind(Rc::clone(&plan),Rc::clone(&stream),weight,
            Rc::clone(&activation),0,Rc::clone(&output),0,1).unwrap()
            .with_scale_placement(CudaQwen35QpnScalePlacement::ResidualWorkspaceShapeF64).unwrap();
        for repeat in 0..4 {
            let mut g=(0..K).map(|i|((i*31+repeat*17)%997)as f32/17.003-29.031).collect::<Vec<_>>();
            let edges=[0.,-0.,3e-8,-3e-8,131136.,-131136.,131264.,65504.,-65504.,-100.,100.];
            g[..edges.len()].copy_from_slice(&edges);
            let u=(0..K).map(|i|if i<edges.len(){0.25}else{((i*19+repeat*3)%251)as f32/503.}).collect::<Vec<_>>();
            gate.copy_from_host_sync(&stream,&g.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
            up.copy_from_host_sync(&stream,&u.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
            silu.submit_for_plan(&stream).unwrap();down.submit_for_plan().unwrap();stream.synchronize().unwrap();
            let a=read(&stream,&activation);let stage=read(&stream,&plan.decoded_input_f16);let y=read(&stream,&output);
            let allocated=crate::cuda_device_allocation_snapshot();
            let result=unsafe{augur_cuda_sys::augur_cuda_qwen35_qpn_silu_down_prevalidated_submit_v1(0,stream.raw().unwrap(),plan.raw,silu.as_raw(),&down.raw)};
            CudaRuntimeError::from_result("fused scalar tail",result).unwrap();stream.synchronize().unwrap();
            let after=crate::cuda_device_allocation_snapshot();
            assert_eq!(after.successful_allocations,allocated.successful_allocations);assert_eq!(after.successful_frees,allocated.successful_frees);
            assert_eq!(read(&stream,&activation),a);assert_eq!(read(&stream,&plan.decoded_input_f16),stage);assert_eq!(read(&stream,&output),y);
            for case in 0..8 {
                let mut a=*silu.as_raw();let mut d=down.raw;
                match case {0=>a.gate=a.output,1=>a.up=d.weight,2=>a.output=d.output_f32,3=>a.elements-=1,
                    4=>d.tokens=2,5=>d.numerical_profile=7,6=>a.gate_len_bytes-=4,_=>a.gate=u64::MAX-15}
                let rejected=unsafe{augur_cuda_sys::augur_cuda_qwen35_qpn_silu_down_prevalidated_submit_v1(0,stream.raw().unwrap(),plan.raw,&a,&d)};
                assert!(CudaRuntimeError::from_result("invalid fused scalar tail",rejected).is_err());
                assert_eq!(read(&stream,&output),y);assert_eq!(read(&stream,&plan.decoded_input_f16),stage);
            }
        }
    }
    let after=crate::cuda_device_allocation_snapshot();assert_eq!(after.live_bytes,before.live_bytes);assert_eq!(after.live_allocations,before.live_allocations);
}
