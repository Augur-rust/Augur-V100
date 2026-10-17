use super::*;
use crate::cuda_runtime::test_buffer_view;
fn views(b:[CudaBufferView;7])->CudaQwen35DflashQ8SelectorViewsV1 {
    CudaQwen35DflashQ8SelectorViewsV1{predecessor:b[0],successor:b[1],candidate_ids:b[2],
        unary_scores:b[3],gates:b[4],transition_scores:b[5],path_ids:b[6]}
}
#[test]
fn phase_gate_rejects_wrong_device_extents_alignment_alias_and_overflow() {
    let b=core::array::from_fn(|i|test_buffer_view(0x1000_0000*(i as u64+1),0,0,QWEN35_DFLASH_Q8_SELECTOR_BYTES[i]));
    let good=test_buffer_view(0x8000_0000,0,0,8192);
    assert_eq!(phase_gate_address(0,views(b),good,248069).unwrap(),0x8000_0000);
    assert!(phase_gate_address(0,views(b),good,248320).is_err());
    for x in [test_buffer_view(0x8000_0000,1,0,8192),test_buffer_view(0x8000_0000,0,0,8196),
              test_buffer_view(0x8000_0000,0,0,8188),test_buffer_view(0,0,0,8192),
              test_buffer_view(0x8000_0001,0,0,8192),test_buffer_view(u64::MAX-3,0,8,8192),
              test_buffer_view(u64::MAX-3,0,0,8192)] {
        assert!(phase_gate_address(0,views(b),x,248069).is_err());
    }
    for x in b {assert!(phase_gate_address(0,views(b),test_buffer_view(x.device_ptr().raw_value(),0,0,8192),248069).is_err());}
}
#[test]
#[ignore="requires SM70 CUDA; same-walk phase switching, independent scores, canaries and poison"]
fn phase_selector_gpu_switch_at_every_position() -> Result<(),Box<dyn std::error::Error>> {
    use crate::{OwnedCudaBuffer,cuda_device_allocation_snapshot,cuda_host_boundary_snapshot};
    let before=cuda_device_allocation_snapshot();
    {
        let stream=OwnedCudaStream::create(0)?;let foreign=OwnedCudaStream::create(0)?;
        let sizes=QWEN35_DFLASH_Q8_SELECTOR_BYTES;
        let owners=sizes.iter().map(|n|OwnedCudaBuffer::allocate(0,n+512)).collect::<Result<Vec<_>,_>>()?;
        let after=OwnedCudaBuffer::allocate(0,8192+512)?;
        let b:Vec<_>=owners.iter().zip(sizes).map(|(o,n)|o.view(256,n)).collect::<Result<_,_>>()?;
        let v=views(b.try_into().unwrap());
        let mut prev=vec![0u8;sizes[0]];let mut next=vec![0u8;sizes[1]];
        for id in (0..16).chain([200000,248068,248069]) {
            let at=id*272+4*34;prev[at..at+2].copy_from_slice(&0x3c00u16.to_le_bytes());prev[at+2]=1;
        }
        for (id,code) in [(7,-32i8),(8,32)] {
            let at=id*272+4*34;next[at..at+2].copy_from_slice(&0x3c00u16.to_le_bytes());next[at+2]=code as u8;
        }
        let upload=|o:&OwnedCudaBuffer,payload:&[u8]|->Result<(),Box<dyn std::error::Error>> {
            let mut bytes=vec![0xa5;o.len_bytes()];bytes[256..256+payload.len()].copy_from_slice(payload);
            o.copy_from_host_sync(&stream,&bytes)?;Ok(())
        };
        let fb=|v:&[f32]|v.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>();
        let ib=|v:&[u32]|v.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>();
        upload(&owners[0],&prev)?;upload(&owners[1],&next)?;
        // -1: no close; 0: closed anchor; 1..7: close at each draft position.
        // Follow it with a literal opening token: the Q8 phase must stay sticky.
        for close_pos in -1i32..=7 {
            let anchor=if close_pos==0{248069}else{200000};
            let mut ids=Vec::new();let mut unary=Vec::new();
            for row in 0..8 {
                let forced=if row==close_pos{Some(248069)}else if close_pos>=0 && row==close_pos+1{Some(248068)}else{None};
                if let Some(id)=forced {ids.push(id);unary.push(100.0);}
                for id in 0..(if forced.is_some(){15}else{16}) {ids.push(id);unary.push(16.0-id as f32);}
            }
            let mut gates=vec![0f32;2048];let mut q8=gates.clone();
            for row in 0..8 {gates[row*256+128]=1.;q8[row*256+128]=-1.;}
            for (i,bytes) in [(2,ib(&ids)),(3,fb(&unary)),(4,fb(&gates)),(5,vec![0;512]),(6,vec![0;32])] {upload(&owners[i],&bytes)?;}
            upload(&after,&fb(&q8))?;
            let bound=CudaQwen35DflashQ8SelectorPhaseArgsV1::bind(&stream,v,anchor,after.view(256,8192)?,248069)?;
            assert!(bound.submit(&foreign).is_err());
            let hot=cuda_device_allocation_snapshot();let host=cuda_host_boundary_snapshot();
            for _ in 0..3 {bound.submit(&stream)?;}
            let d=cuda_host_boundary_snapshot().checked_delta_since(host).ok_or("counter regression")?;
            assert_eq!((d.synchronous_h2d_bytes,d.synchronous_d2h_bytes,d.host_waits),(0,0,0));
            stream.synchronize()?;assert_eq!(cuda_device_allocation_snapshot(),hot);
            let mut expected=vec![anchor];let mut expected_scores=vec![0f32;128];let mut closed=anchor==248069;
            for row in 1..8 {
                let mut best=0;
                for k in 0..16 {
                    let value=match ids[row*16+k]{7=>-32.,8=>32.,_=>0.};
                    expected_scores[row*16+k]=value*(if closed{-1.}else{1.})+unary[row*16+k];
                    if expected_scores[row*16+k]>expected_scores[row*16+best]{best=k;}
                }
                let token=ids[row*16+best];expected.push(token);closed|=token==248069;
            }
            for slot in [5,6] {
                let mut bytes=vec![0u8;owners[slot].len_bytes()];owners[slot].copy_to_host_sync(&stream,&mut bytes)?;
                assert!(bytes[..256].iter().chain(bytes[256+sizes[slot]..].iter()).all(|b|*b==0xa5));
                let payload=&bytes[256..256+sizes[slot]];
                if slot==5 {assert_eq!(payload,fb(&expected_scores));} else {assert_eq!(payload,ib(&expected));}
            }
            // Both slabs are required finite even if the trajectory does not use one.
            for which in 0..2 {
                let mut bad=if which==0{gates.clone()}else{q8.clone()};bad[7*256+128]=f32::NAN;
                upload(if which==0{&owners[4]}else{&after},&fb(&bad))?;bound.submit(&stream)?;
                let mut bytes=vec![0u8;owners[6].len_bytes()];owners[6].copy_to_host_sync(&stream,&mut bytes)?;
                assert_eq!(&bytes[256..288],ib(&[u32::MAX;8]));
                upload(&owners[4],&fb(&gates))?;upload(&after,&fb(&q8))?;
            }
            // Finite ties preserve top-k order in either phase.
            let ids:Vec<u32>=(0..8).flat_map(|_|0..16).collect();
            upload(&owners[2],&ib(&ids))?;upload(&owners[3],&fb(&vec![0.;128]))?;
            upload(&owners[4],&fb(&vec![0.;2048]))?;upload(&after,&fb(&vec![0.;2048]))?;
            bound.submit(&stream)?;let mut bytes=vec![0u8;owners[6].len_bytes()];owners[6].copy_to_host_sync(&stream,&mut bytes)?;
            assert_eq!(&bytes[256..288],ib(&[anchor,0,0,0,0,0,0,0]));
        }
    }
    let final_state=cuda_device_allocation_snapshot();
    assert_eq!(final_state.live_bytes,before.live_bytes);assert_eq!(final_state.live_allocations,before.live_allocations);
    Ok(())
}
