use super::*;

#[test]
fn independent_copy_batch_abi_is_bounded() {
    assert_eq!(core::mem::size_of::<augur_cuda_sys::AugurCudaD2dCopyEntryV1>(),24);
    assert_eq!(core::mem::align_of::<augur_cuda_sys::AugurCudaD2dCopyEntryV1>(),8);
}

#[test]
#[ignore = "real CUDA SM70 independent-copy component"]
fn independent_copy_batch_gpu_exact_ranges_guards_and_rejections() {
    let before=cuda_device_allocation_snapshot();
    let mut cases=0;
    let mut rejects=0;
    {
        let stream=OwnedCudaStream::create(0).unwrap();
        for record in [false,true] {
            let mut ranges=Vec::new(); let mut extent=64;
            let mut add=|bytes,full| {ranges.push((extent,bytes,full));extent+=bytes+128;};
            if record {for bytes in [8*10240*4,8*48*4,8*48*4] {add(bytes,false);}}
            else {for layer in 0..64 {if layer%4==3 {add(8*1088,true);add(8*1088,true);}
                else {add(30720*4,false);add(786432*4,false);}}}
            extent+=64;
            let source=OwnedCudaBuffer::allocate(0,extent).unwrap();
            let reference=OwnedCudaBuffer::allocate(0,extent).unwrap();
            let actual=OwnedCudaBuffer::allocate(0,extent).unwrap();
            let poison=vec![0xa5;extent]; let mut expected=vec![0;extent];let mut observed=vec![0;extent];
            for generation in 0..2 {
                let input:Vec<u8>=(0..extent).map(|i|((i.wrapping_mul(131)^((i>>8)*37)) as u8).wrapping_add(generation*71)).collect();
                source.copy_from_host_sync(&stream,&input).unwrap();
                for committed in 0..=if record {0} else {8} {
                    reference.copy_from_host_sync(&stream,&poison).unwrap();actual.copy_from_host_sync(&stream,&poison).unwrap();
                    let mut copies=Vec::new();
                    for &(offset,bytes,full) in &ranges {
                        let skip=if full {committed*1088} else {0};if skip==bytes {continue;}
                        let src=source.view(offset+skip,bytes-skip).unwrap();
                        // SAFETY: all owners retained to synchronous readback on this stream.
                        unsafe {reference.copy_from_device_async_unchecked(&stream,offset+skip,src).unwrap();}
                        copies.push((&actual,offset+skip,src));
                    }
                    let allocated=cuda_device_allocation_snapshot();
                    // SAFETY: ranges are independent; this scope drains before any owner is freed.
                    unsafe {OwnedCudaBuffer::copy_independent_batch_async_unchecked(&stream,&copies).unwrap();}
                    reference.copy_to_host_sync(&stream,&mut expected).unwrap();actual.copy_to_host_sync(&stream,&mut observed).unwrap();
                    assert!(expected==observed,"family record={record} committed={committed} generation={generation}");
                    assert_eq!(cuda_device_allocation_snapshot().successful_allocations,allocated.successful_allocations);
                    assert_eq!(copies.len(),if record {3} else if committed==8 {96} else {128});
                    cases+=1;
                }
                source.copy_to_host_sync(&stream,&mut observed).unwrap();assert!(input==observed,"source changed");
            }
            actual.copy_from_host_sync(&stream,&poison).unwrap();
            let src=source.view(64,64).unwrap();let dst=actual.view(64,64).unwrap();
            let good=augur_cuda_sys::AugurCudaD2dCopyEntryV1 {source:src.checked_address("test").unwrap(),destination:dst.checked_address("test").unwrap(),bytes:64};
            for mutation in 0..14 {
                let mut entries=[good,good];entries[1].source+=128;entries[1].destination+=128;let mut count=2;
                match mutation {
                    0=>count=0,1=>count=129,2=>entries[0].bytes=0,3=>entries[0].source=0,
                    4=>entries[0].destination=0,5=>entries[0].source+=4,6=>entries[0].destination+=4,
                    7=>entries[0].bytes+=4,8=>entries[0].source=u64::MAX-15,
                    9=>entries[0].bytes=u64::MAX-15,10=>entries[0].source=entries[0].destination,
                    11=>entries[1].source=entries[0].destination,
                    12=>entries[1].destination=entries[0].destination+16,
                    _=>entries[1].destination=entries[0].source,
                }
                // SAFETY: every malformed batch must reject before launching. count=129 rejects before entry reads.
                let result=unsafe {augur_cuda_sys::augur_cuda_memcpy_d2d_independent_batch_submit_v1(0,stream.raw().unwrap(),entries.as_ptr(),count)};
                assert_ne!(result.status,0,"mutation={mutation}");rejects+=1;
            }
            // Whole Rust batch validates before first submission, not one entry at a time.
            unsafe {
                assert!(OwnedCudaBuffer::copy_independent_batch_async_unchecked(&stream,&[]).is_err());
                assert!(OwnedCudaBuffer::copy_independent_batch_async_unchecked(&stream,&[(&actual,64,src),(&actual,extent-16,src)]).is_err());
            }
            actual.copy_to_host_sync(&stream,&mut observed).unwrap();assert!(observed==poison,"rejected batch wrote output");
            stream.synchronize().unwrap();
        }
    }
    let after=cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes,before.live_bytes);assert_eq!(after.live_allocations,before.live_allocations);
    assert_eq!(after.successful_allocations-before.successful_allocations,after.successful_frees-before.successful_frees);
    println!("INDEPENDENT_COPY_EXACT cases={cases} native_rejections={rejects} all_bytes_guards_source=true submit_allocations=0 live_bytes_delta=0 live_allocations_delta=0");
}
