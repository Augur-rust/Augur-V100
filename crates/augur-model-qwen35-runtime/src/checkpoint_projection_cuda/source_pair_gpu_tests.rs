//! Synthetic independent source-layout oracle and retained-owner checks.
use super::*;
use augur_backend::cuda_device_allocation_snapshot;
use augur_backend::cuda_qwen35_qpn::{OwnedCudaQwen35QpnWeightV1, source_pair::{CudaQpnSourcePairProjection, Schedule}};

#[test]
#[ignore = "exclusive SM70: original GGML source vs QPN source-exact, alias/extent/lifetime checks"]
fn qpn_source_exact_preserves_source_math_and_retains_checked_owners() {
    let before = cuda_device_allocation_snapshot();
    {
        let stream = Rc::new(OwnedCudaStream::create(0).unwrap());
        // 10240 rows takes the eight-chain prefetch kernel, the others the split one.
        for (rows, cols) in [(32usize,64usize),(64,512),(10240,320),(17408,128),(1024,5120)] {
            let mut ggml = vec![0u8; rows*(cols/64)*36];
            for (i,block) in ggml.chunks_exact_mut(36).enumerate() {
                for j in 0..4 {block[j] = [1,57,96,126][(i+j)%4];}
                for j in 4..36 {block[j] = (i*17+j*13+7) as u8;}
            }
            let source = upload(&stream, &ggml).unwrap();
            let qpn = Rc::new(OwnedCudaQwen35QpnWeightV1::upload_ggml(&stream,&ggml,rows,cols,0.125).unwrap());
            // Includes values beyond FP16 range and small values that lose bits in
            // a half decomposition. This policy must never stage either as half.
            let values: Vec<f32> = (0..cols).map(|i| match i%5 {
                0 => 131072.015625, 1 => -65536.0078125,
                2 => f32::from_bits(0x3080_0001 + i as u32),
                _ => f32::from_bits(0x3f00_0001 + (i as u32)*7919),
            }).collect();
            let bytes: Vec<u8> = values.iter().flat_map(|v|v.to_le_bytes()).collect();
            let x = Rc::new(upload(&stream,&bytes).unwrap());
            let y = Rc::new(OwnedCudaBuffer::allocate(0,rows*4).unwrap());
            let reference = OwnedCudaBuffer::allocate(0,rows*4).unwrap();
            let original = CudaNvfp4Sm70F32MatvecArgsV1::bind(&stream,
                CudaNvfp4Sm70WeightViewV1::new(source.view(0,source.len_bytes()).unwrap(),
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,rows,cols).unwrap(),
                f32_view(&x,0,cols).unwrap(),f32_view(&reference,0,rows).unwrap(),0.125).unwrap();
            original.submit_for_plan(&stream).unwrap();
            let bind = |input:Rc<OwnedCudaBuffer>,offset,output:Rc<OwnedCudaBuffer>| {
                CudaQpnSourcePairProjection::bind(Rc::clone(&stream),Rc::clone(&qpn),input,offset,output,0,Schedule::SourceExact)
            };
            assert!(bind(Rc::clone(&x),0,Rc::clone(&x)).is_err());
            assert!(bind(Rc::clone(&x),4,Rc::clone(&y)).is_err());
            let short = Rc::new(OwnedCudaBuffer::allocate(0,rows*4-4).unwrap());
            assert!(bind(Rc::clone(&x),0,short).is_err());
            let mut expected=vec![0;rows*4];
            {
                let fast = CudaQpnSourcePairProjection::bind(Rc::clone(&stream),Rc::clone(&qpn),Rc::clone(&x),0,Rc::clone(&y),0,Schedule::SourcePrefetch).unwrap();
                assert_eq!(fast.workspace_bytes(),0);
                fast.submit().unwrap();stream.synchronize().unwrap();
                let mut actual=vec![0;rows*4];
                y.copy_to_host_sync(&stream,&mut actual).unwrap();
                reference.copy_to_host_sync(&stream,&mut expected).unwrap();
                assert_eq!(actual,expected,"QPN source-prefetch arithmetic changed");
                // The re-associated F64 sum is a distinct identity: not bitwise,
                // but an F64 reordering must stay far below F32 resolution.
                let re = CudaQpnSourcePairProjection::bind(Rc::clone(&stream),Rc::clone(&qpn),Rc::clone(&x),0,Rc::clone(&y),0,Schedule::SourceF64Reassoc).unwrap();
                assert_eq!(re.workspace_bytes(),0);
                re.submit().unwrap();stream.synchronize().unwrap();
                y.copy_to_host_sync(&stream,&mut actual).unwrap();
                for (a,e) in actual.chunks_exact(4).zip(expected.chunks_exact(4)) {
                    let (a,e)=(f32::from_le_bytes(a.try_into().unwrap()),f32::from_le_bytes(e.try_into().unwrap()));
                    assert!(a==e || (a-e).abs() <= e.abs()*f32::EPSILON, "re-associated F64 sum moved past one F32 ulp");
                }
                // An unaligned input row cannot take the float4 path.
                let wide = Rc::new(upload(&stream,&[bytes.as_slice(),&[0u8;4]].concat()).unwrap());
                assert!(CudaQpnSourcePairProjection::bind(Rc::clone(&stream),Rc::clone(&qpn),Rc::clone(&wide),4,Rc::clone(&y),0,Schedule::SourcePrefetch).is_err());
                assert!(CudaQpnSourcePairProjection::bind(Rc::clone(&stream),Rc::clone(&qpn),wide,4,Rc::clone(&y),0,Schedule::SourceF64Reassoc).is_err());
            }
            let op = bind(Rc::clone(&x),0,Rc::clone(&y)).unwrap();
            assert_eq!(op.workspace_bytes(),0);
            let weak_w=Rc::downgrade(&qpn);let weak_x=Rc::downgrade(&x);let weak_y=Rc::downgrade(&y);
            drop(qpn);drop(x);drop(y);
            assert!(weak_w.upgrade().is_some() && weak_x.upgrade().is_some() && weak_y.upgrade().is_some());
            op.submit().unwrap();stream.synchronize().unwrap();
            let mut actual=vec![0;rows*4];
            weak_y.upgrade().unwrap().copy_to_host_sync(&stream,&mut actual).unwrap();
            reference.copy_to_host_sync(&stream,&mut expected).unwrap();
            assert_eq!(actual,expected,"QPN source-exact arithmetic changed");
            drop(op);
            assert!(weak_w.upgrade().is_none() && weak_x.upgrade().is_none() && weak_y.upgrade().is_none());
        }
    }
    let after=cuda_device_allocation_snapshot();
    assert_eq!(before.live_bytes,after.live_bytes);assert_eq!(before.live_allocations,after.live_allocations);
}
