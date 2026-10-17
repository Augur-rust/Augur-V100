use super::*;
use augur_core::PhysicalLayoutId as Layout;

#[test]
fn replay_fold_shared_rejects_null_without_cuda() {
    // SAFETY: null metadata/stream reject before any CUDA action.
    let result = unsafe {
        augur_cuda_sys::augur_cuda_qwen35_prefill_linear_chunk_fold_shared_prevalidated_submit_v1(
            0, std::ptr::null_mut(), std::ptr::null(),
        )
    };
    assert_eq!(result.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
    // SAFETY: both checkpoint entry points reject nulls before CUDA access.
    for preflight in [true, false] {
        let result = unsafe {
            let call = if preflight {
                augur_cuda_sys::augur_cuda_qwen35_prefill_linear_checkpoint_fold_preflight_v1
            } else {
                augur_cuda_sys::augur_cuda_qwen35_prefill_linear_checkpoint_fold_prevalidated_submit_v1
            };
            call(0, std::ptr::null_mut(), std::ptr::null(), 0, 0, 0, 0)
        };
        assert_eq!(result.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
    }
}

struct Guard {
    owner: OwnedCudaBuffer,
    bytes: usize,
}
impl Guard {
    fn new(elements: usize) -> Self {
        Self { owner: OwnedCudaBuffer::allocate(0, elements * 4 + 512).unwrap(), bytes: elements * 4 }
    }
    fn view(&self) -> CudaF32View {
        CudaF32View::new(self.owner.view(256, self.bytes).unwrap(),
            Layout::DEVICE_CONTIGUOUS_F32_CUDA_V1, self.bytes / 4).unwrap()
    }
    fn upload(&self, stream: &OwnedCudaStream, value: &[u8]) {
        assert_eq!(value.len(), self.bytes);
        let mut all = vec![0xa5; self.bytes + 512];
        all[256..256+self.bytes].copy_from_slice(value);
        self.owner.copy_from_host_sync(stream, &all).unwrap();
    }
    fn read(&self, stream: &OwnedCudaStream) -> Vec<u8> {
        let mut all = vec![0; self.bytes + 512];
        self.owner.copy_to_host_sync(stream, &mut all).unwrap();
        assert!(all[..256].iter().chain(all[256+self.bytes..].iter()).all(|x| *x == 0xa5));
        all[256..256+self.bytes].to_vec()
    }
}

#[test]
#[ignore = "exclusive SM70 and bound GDN capture/weight fixtures"]
fn replay_prefix_family_exact_subviews_and_host_binding_cost() {
    let capture = std::path::PathBuf::from(std::env::var_os("AUGUR_QWEN35_GDN_CAPTURE_ROOT").expect("capture root"));
    let weights = std::path::PathBuf::from(std::env::var_os("AUGUR_QWEN35_GDN_WEIGHT_ROOT").expect("weights root"));
    let before = crate::cuda_device_allocation_snapshot();
    let mut cases = 0;
    {
        let stream = OwnedCudaStream::create(0).unwrap();
        let foreign = OwnedCudaStream::create(0).unwrap();
        for layer in [0,50] {
            let lengths = [8*10240,8*6144,4*10240,3*10240,8*48,8*48,48,48,48*128*128,128,8*10240,8*6144,8*6144];
            let b: Vec<_> = lengths.into_iter().map(Guard::new).collect();
            let ch = Guard::new(3*10240); let cs = Guard::new(48*128*128);
            let v = |i:usize,n:usize| CudaF32View::new(b[i].owner.view(256,n*4).unwrap(),Layout::DEVICE_CONTIGUOUS_F32_CUDA_V1,n).unwrap();
            let views = |r:usize| CudaQwen35PrefillLinearChunkViewsV1 {
                qkv:v(0,r*10240),z:v(1,r*6144),conv_weights:v(2,4*10240),conv_history:v(3,3*10240),
                alpha_raw:v(4,r*48),beta_raw:v(5,r*48),dt_bias:v(6,48),ssm_a:v(7,48),state:v(8,48*128*128),
                norm_weight:v(9,128),activated:v(10,r*10240),recurrent:v(11,r*6144),output:v(12,r*6144),
            };
            let allocated = crate::cuda_device_allocation_snapshot();
            let bind = |r:usize| CudaQwen35PrefillLinearChunkV1::bind_with_checkpoint_fold(&stream,views(r),r,1e-6,ch.view(),cs.view()).unwrap();
            let full = bind(8);
            assert!(full.checkpoint_prefix_from_b8(&foreign,1).is_err());
            for n in [0,9,usize::MAX] {assert!(full.checkpoint_prefix_from_b8(&stream,n).is_err());}
            let ordinary = CudaQwen35PrefillLinearChunkV1::bind(&stream,views(8),8,1e-6).unwrap();
            assert!(ordinary.checkpoint_prefix_from_b8(&stream,1).is_err());
            assert!(full.checkpoint_prefix_from_b8(&stream,1).unwrap().checkpoint_prefix_from_b8(&stream,2).is_err());
            for mutation in 0..7 {
                let mut bad = full;
                match mutation {0=>bad.raw.qkv_len_bytes-=4,1=>bad.raw.z_len_bytes-=4,
                    2=>bad.raw.alpha_raw_len_bytes-=4,3=>bad.raw.beta_raw_len_bytes-=4,
                    4=>bad.raw.activated_len_bytes-=4,5=>bad.raw.recurrent_len_bytes-=4,_=>bad.raw.output_len_bytes-=4}
                assert!(bad.checkpoint_prefix_from_b8(&stream,1).is_err());
            }
            for batch in [0,5632,0] {
                let read = |name:&str,total:usize,used:usize| {
                    let bytes=std::fs::read(capture.join(format!("batch{batch}-{name}-{layer}.f32le.bin"))).unwrap();
                    assert_eq!(bytes.len(),total*4);bytes[..used*4].to_vec()
                };
                let weight = |name:&str,n:usize| {let x=std::fs::read(weights.join(format!("layer{layer}-{name}.f32le.bin"))).unwrap();assert_eq!(x.len(),n*4);x};
                let input = vec![read("linear_attn_qkv_mixed",512*10240,8*10240),read("z",512*6144,8*6144),
                    weight("conv",4*10240),read("conv_states_reshaped",3*10240,3*10240),read("alpha",512*48,8*48),read("beta",512*48,8*48),
                    weight("bias",48),weight("a",48),read("state_predelta",48*128*128,48*128*128),weight("norm",128),
                    vec![0xa5;8*10240*4],vec![0xa5;8*6144*4],vec![0xa5;8*6144*4]];
                for rows in 1..=8 {
                    let reference = bind(rows); let derived = full.checkpoint_prefix_from_b8(&stream,rows).unwrap();
                    // Debug serializes named fields, never padding/uninitialized struct bytes.
                    assert_eq!(format!("{:?}",reference.raw),format!("{:?}",derived.raw));
                    assert_eq!(reference.checkpoint_sources,derived.checkpoint_sources);
                    let reset = || {
                        for (buffer,data) in b.iter().zip(&input) {buffer.upload(&stream,data);}
                        ch.upload(&stream,&input[3]);cs.upload(&stream,&input[8]);
                        b[3].upload(&stream,&vec![0xff;input[3].len()]);b[8].upload(&stream,&vec![0xff;input[8].len()]);
                    };
                    reset();reference.submit_checkpoint_fold(&stream).unwrap();
                    let expected:Vec<_>=b.iter().map(|buffer|buffer.read(&stream)).collect();
                    reset();derived.submit_checkpoint_fold(&stream).unwrap();
                    for (i,(buffer,data)) in b.iter().zip(&expected).enumerate() {assert_eq!(buffer.read(&stream),*data,"layer={layer} rows={rows} batch={batch} field={i}");}
                    assert_eq!(ch.read(&stream),input[3]);assert_eq!(cs.read(&stream),input[8]);cases+=1;
                }
            }
            stream.synchronize().unwrap();
            let mut old=Vec::new();let mut new=Vec::new();
            for round in 0..12 {
                for slot in 0..2 {
                    let derived=(round+slot)%2==1;let start=std::time::Instant::now();
                    for _ in 0..48 {
                        if derived {let base=bind(8);for r in 1..=8 {std::hint::black_box(base.checkpoint_prefix_from_b8(&stream,r).unwrap());}}
                        else {for r in 1..=8 {std::hint::black_box(bind(r));}}
                    }
                    let ms=start.elapsed().as_secs_f64()*1000.;if derived {new.push(ms);} else {old.push(ms);}
                }
            }
            old.sort_by(f64::total_cmp);new.sort_by(f64::total_cmp);
            println!("REPLAY_PREFIX_FAMILY layer={layer} old_48_layer_median_ms={} derived_48_layer_median_ms={} scope=host_idle_component_not_overlapped_HTTP",old[6],new[6]);
            let after=crate::cuda_device_allocation_snapshot();
            assert_eq!(allocated.successful_allocations,after.successful_allocations);
            assert_eq!(allocated.successful_frees,after.successful_frees);
            assert_eq!(allocated.live_bytes,after.live_bytes);
        }
    }
    let after=crate::cuda_device_allocation_snapshot();
    assert_eq!(after.successful_allocations-before.successful_allocations,30);
    assert_eq!(after.successful_frees-before.successful_frees,30);
    assert_eq!(after.live_bytes,before.live_bytes);assert_eq!(after.live_allocations,before.live_allocations);
    println!("REPLAY_PREFIX_FAMILY_EXACT cases={cases} allocations=30 frees=30 live_delta=0 no_bind_submit_allocations=true");
}

#[test]
#[ignore = "SM70 real GDN inputs; requires AUGUR_QWEN35_GDN_CAPTURE_ROOT and AUGUR_QWEN35_GDN_WEIGHT_ROOT"]
fn replay_fold_shared_real_prefixes_exact_and_no_submit_allocation() {
    let capture = std::path::PathBuf::from(std::env::var_os("AUGUR_QWEN35_GDN_CAPTURE_ROOT").expect("capture root"));
    let weights = std::path::PathBuf::from(std::env::var_os("AUGUR_QWEN35_GDN_WEIGHT_ROOT").expect("weights root"));
    let before = crate::cuda_device_allocation_snapshot();
    let mut cases = 0;
    let mut rejections = 0;
    let mut verify_cases = 0;
    {
        let stream = OwnedCudaStream::create(0).unwrap();
        let foreign = OwnedCudaStream::create(0).unwrap();
        for layer in [0, 50] {
            let weight = |field: &str, elements: usize| {
                let bytes = std::fs::read(weights.join(format!("layer{layer}-{field}.f32le.bin"))).unwrap();
                assert_eq!(bytes.len(), elements * 4);
                bytes
            };
            for rows in 1..=8 {
                let lengths = [rows*10240, rows*6144, 4*10240, 3*10240, rows*48, rows*48,
                    48, 48, 48*128*128, 128, rows*10240, rows*6144, rows*6144];
                let b: Vec<_> = lengths.into_iter().map(Guard::new).collect();
                let views = CudaQwen35PrefillLinearChunkViewsV1 {
                        qkv:b[0].view(), z:b[1].view(), conv_weights:b[2].view(), conv_history:b[3].view(),
                        alpha_raw:b[4].view(), beta_raw:b[5].view(), dt_bias:b[6].view(), ssm_a:b[7].view(),
                        state:b[8].view(), norm_weight:b[9].view(), activated:b[10].view(),
                        recurrent:b[11].view(), output:b[12].view(),
                    };
                let op = CudaQwen35PrefillLinearChunkV1::bind(&stream, views, rows, 1e-6).unwrap();
                assert!(op.submit_state_fold_shared_scalars(&foreign).is_err());
                assert!(op.submit_checkpoint_fold(&stream).is_err());
                let checkpoint_history = Guard::new(3*10240);
                let checkpoint_state = Guard::new(48*128*128);
                assert!(op.bind_checkpoint_fold(&foreign, checkpoint_history.view(), checkpoint_state.view()).is_err());
                assert!(op.bind_checkpoint_fold(&stream, b[3].view(), checkpoint_state.view()).is_err());
                assert!(op.bind_checkpoint_fold(&stream, checkpoint_history.view(), b[8].view()).is_err());
                let direct = CudaQwen35PrefillLinearChunkV1::bind_with_checkpoint_fold(&stream,
                    views, rows, 1e-6, checkpoint_history.view(), checkpoint_state.view()).unwrap();
                assert!(CudaQwen35PrefillLinearChunkV1::bind_with_checkpoint_fold(&stream,
                    views, rows, 1e-6, b[3].view(), checkpoint_state.view()).is_err());
                assert!(direct.submit_checkpoint_fold(&foreign).is_err());
                for batch in [0, 5632, 0] {
                    let captured = |field: &str, total: usize, used: usize| {
                        let bytes = std::fs::read(capture.join(format!("batch{batch}-{field}-{layer}.f32le.bin"))).unwrap();
                        assert_eq!(bytes.len(), total * 4);
                        bytes[..used*4].to_vec()
                    };
                    let input = vec![
                        captured("linear_attn_qkv_mixed",512*10240,rows*10240), captured("z",512*6144,rows*6144),
                        weight("conv",4*10240), captured("conv_states_reshaped",3*10240,3*10240),
                        captured("alpha",512*48,rows*48), captured("beta",512*48,rows*48),
                        weight("bias",48), weight("a",48), captured("state_predelta",48*128*128,48*128*128),
                        weight("norm",128), vec![0xa5;rows*10240*4], vec![0xa5;rows*6144*4], vec![0xa5;rows*6144*4],
                    ];
                    for (buffer,data) in b.iter().zip(&input) { buffer.upload(&stream,data); }
                    op.submit_state_fold(&stream).unwrap();
                    let expected: Vec<_> = [3,8,10].map(|i| b[i].read(&stream)).into();
                    assert!(b[11].read(&stream) == input[11]);
                    assert!(b[12].read(&stream) == input[12]);
                    for (buffer,data) in b.iter().zip(&input) { buffer.upload(&stream,data); }
                    let allocated = crate::cuda_device_allocation_snapshot();
                    op.submit_state_fold_shared_scalars(&stream).unwrap();
                    for (i,expected) in [3,8,10].into_iter().zip(&expected) {
                        assert!(b[i].read(&stream) == *expected,"fold bits layer={layer} rows={rows} batch={batch} field={i}");
                    }
                    for i in [0,1,2,4,5,6,7,9,11] {
                        assert!(b[i].read(&stream) == input[i],"readonly field={i}");
                    }
                    let scratch = b[12].read(&stream);
                    assert!(scratch[rows*48*3*4..].iter().all(|v| *v == 0xa5),"scratch bound");
                    assert_eq!(crate::cuda_device_allocation_snapshot().successful_allocations,allocated.successful_allocations);
                    // Poison working initial values: only the read-only checkpoints
                    // may supply them. Repeat with changed captured generations.
                    checkpoint_history.upload(&stream, &input[3]);
                    checkpoint_state.upload(&stream, &input[8]);
                    for (buffer,data) in b.iter().zip(&input) { buffer.upload(&stream,data); }
                    b[3].upload(&stream, &vec![0xff; input[3].len()]);
                    b[8].upload(&stream, &vec![0xff; input[8].len()]);
                    direct.submit_checkpoint_fold(&stream).unwrap();
                    for (i,expected) in [3,8,10].into_iter().zip(&expected) {
                        assert!(b[i].read(&stream) == *expected,"checkpoint fold bits layer={layer} rows={rows} batch={batch} field={i}");
                    }
                    assert!(b[12].read(&stream) == scratch, "checkpoint scalar scratch exact");
                    for i in [0,1,2,4,5,6,7,9,11] { assert!(b[i].read(&stream) == input[i]); }
                    assert!(checkpoint_history.read(&stream) == input[3]);
                    assert!(checkpoint_state.read(&stream) == input[8]);
                    assert_eq!(crate::cuda_device_allocation_snapshot().successful_allocations,allocated.successful_allocations);
                    cases += 1;
                    if rows == 8 {
                        for (buffer,data) in b.iter().zip(&input) { buffer.upload(&stream,data); }
                        op.submit(&stream).unwrap();
                        let expected: Vec<_> = [3,8,10,11,12].map(|i| b[i].read(&stream)).into();
                        for (buffer,data) in b.iter().zip(&input) { buffer.upload(&stream,data); }
                        let allocated = crate::cuda_device_allocation_snapshot();
                        op.submit_shared_scalars(&stream).unwrap();
                        for (i,expected) in [3,8,10,11,12].into_iter().zip(&expected) {
                            assert!(b[i].read(&stream) == *expected,"verify bits layer={layer} batch={batch} field={i}");
                        }
                        for i in [0,1,2,4,5,6,7,9] { assert!(b[i].read(&stream) == input[i]); }
                        assert_eq!(crate::cuda_device_allocation_snapshot().successful_allocations,allocated.successful_allocations);
                        verify_cases += 1;
                    }
                }
                let stable: Vec<_> = b.iter().map(|buffer| buffer.read(&stream)).collect();
                for mutation in 0..9 {
                    let mut raw = op.raw;
                    match mutation {
                        0 => raw.tokens = 0,
                        1 => raw.tokens = 9,
                        2 => raw.abi_version += 1,
                        3 => raw.struct_size -= 1,
                        4 => raw.reserved0 = 1,
                        5 => raw.output_len_bytes -= 4,
                        6 => raw.output_f32 = 0,
                        7 => raw.epsilon_bits = f32::NAN.to_bits(),
                        _ => raw.epsilon_bits = 0,
                    }
                    // SAFETY: invalid fixed-profile metadata must reject before submission.
                    let result = unsafe {
                        augur_cuda_sys::augur_cuda_qwen35_prefill_linear_chunk_fold_shared_prevalidated_submit_v1(
                            0,stream.raw().unwrap(),&raw,
                        )
                    };
                    assert_ne!(result.status,0,"mutation={mutation}");
                    let [h,hb,s,sb] = direct.checkpoint_sources.unwrap();
                    // SAFETY: these deliberately malformed chunk fields reject.
                    let result = unsafe {
                        augur_cuda_sys::augur_cuda_qwen35_prefill_linear_checkpoint_fold_prevalidated_submit_v1(
                            0, stream.raw().unwrap(), &raw, h,hb,s,sb)
                    };
                    assert_ne!(result.status,0,"checkpoint chunk mutation={mutation}");
                    rejections += 1;
                }
                let sources = direct.checkpoint_sources.unwrap();
                let raw = op.raw;
                let operands = [raw.qkv_f32, raw.z_f32, raw.conv_weights_f32, raw.conv_history_f32,
                    raw.alpha_raw_f32, raw.beta_raw_f32, raw.dt_bias_f32, raw.ssm_a_f32,
                    raw.state_f32, raw.norm_weight_f32, raw.activated_f32, raw.recurrent_f32, raw.output_f32];
                let mut bad_sources = Vec::new();
                for field in [0,2] {
                    for address in operands.into_iter().chain([0, u64::MAX-3, sources[field]+1, sources[2-field]]) {
                        let mut changed=sources; changed[field]=address; bad_sources.push(changed);
                    }
                }
                for field in [1,3] {
                    for length in [0,sources[field]-4,sources[field]+4] {
                        let mut changed=sources; changed[field]=length; bad_sources.push(changed);
                    }
                }
                for [h,hb,s,sb] in bad_sources {
                    for preflight in [true,false] {
                        // SAFETY: malformed or overlapping ranges reject before submission.
                        let result=unsafe {
                            let call=if preflight {
                                augur_cuda_sys::augur_cuda_qwen35_prefill_linear_checkpoint_fold_preflight_v1
                            } else { augur_cuda_sys::augur_cuda_qwen35_prefill_linear_checkpoint_fold_prevalidated_submit_v1 };
                            call(0,stream.raw().unwrap(),&raw,h,hb,s,sb)
                        };
                        assert_ne!(result.status,0,"checkpoint source rejection");
                        rejections+=1;
                    }
                }
                // Host pointers have valid arithmetic but are not device allocations.
                let host=vec![0_f32;3*10240];
                let [_,hb,s,sb]=sources;
                let result=unsafe {
                    augur_cuda_sys::augur_cuda_qwen35_prefill_linear_checkpoint_fold_preflight_v1(
                        0,stream.raw().unwrap(),&raw,host.as_ptr() as u64,hb,s,sb)
                };
                assert_ne!(result.status,0);
                let mut invalid = op;
                invalid.raw.tokens = 9;
                assert!(invalid.submit_state_fold_shared_scalars(&stream).is_err());
                for (buffer,expected) in b.iter().zip(&stable) { assert!(buffer.read(&stream) == *expected); }
                stream.synchronize().unwrap();
            }
        }
    }
    let after = crate::cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes,before.live_bytes);
    assert_eq!(after.live_allocations,before.live_allocations);
    assert_eq!(after.successful_allocations-before.successful_allocations,after.successful_frees-before.successful_frees);
    println!("REPLAY_FOLD_SHARED_EXACT cases={cases} checkpoint_read_exact_cases={cases} poisoned_destinations=true verify_full_chain_cases={verify_cases} native_rejections={rejections} changed_generations=3 guards_and_readonly=true submit_allocations=0 live_bytes_delta=0 live_allocations_delta=0");
}
