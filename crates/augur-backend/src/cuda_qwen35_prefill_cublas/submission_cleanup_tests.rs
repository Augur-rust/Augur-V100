use super::*;
use augur_core::PhysicalLayoutId as Layout;

#[test]
#[ignore = "real SM70 GDN operands; requires AUGUR_QWEN35_GDN_CAPTURE_ROOT and AUGUR_QWEN35_GDN_WEIGHT_ROOT"]
fn submission_cleanup_gdn_pair_real_operands_exact() {
    let capture = std::path::PathBuf::from(std::env::var_os("AUGUR_QWEN35_GDN_CAPTURE_ROOT").expect("capture root"));
    let weights = std::path::PathBuf::from(std::env::var_os("AUGUR_QWEN35_GDN_WEIGHT_ROOT").expect("weights root"));
    let before = crate::cuda_device_allocation_snapshot();
    let mut cases = 0;
    let mut paired = 0;
    let mut rejected = 0;
    {
        let stream = OwnedCudaStream::create(0).unwrap();
        let foreign = OwnedCudaStream::create(0).unwrap();
        for layer in [0, 50] {
            let wa = OwnedCudaBuffer::allocate(0,48*5120*2).unwrap();
            let wb = OwnedCudaBuffer::allocate(0,48*5120*2).unwrap();
            for (w,name) in [(&wa,"alpha"),(&wb,"beta")] {
                let bytes = std::fs::read(weights.join(format!("layer{layer}-{name}.bf16.bin"))).unwrap();
                assert_eq!(bytes.len(),48*5120*2); w.copy_from_host_sync(&stream,&bytes).unwrap();
            }
            let view = |w: &OwnedCudaBuffer| CudaQwen35PrefillWeightV1::Bf16(
                CudaBf16WeightViewV1::new(w.view(0,w.len_bytes()).unwrap(),
                    Layout::DEVICE_CONTIGUOUS_BF16_CUDA_V1,48,5120).unwrap());
            let mut plan = CudaQwen35PrefillCublasPlanV1::new(&stream,48*5120,5120,512).unwrap();
            for tokens in [1,8,31,63,64,65,96,127,128,129,255,256,257,285,511,512] {
                let x = OwnedCudaBuffer::allocate(0,tokens*5120*4).unwrap();
                let q = OwnedCudaBuffer::allocate(0,tokens*5120/32*36).unwrap();
                let ya = OwnedCudaBuffer::allocate(0,tokens*48*4).unwrap();
                let yb = OwnedCudaBuffer::allocate(0,tokens*48*4).unwrap();
                let qv = CudaQ8_1ActivationViewV1::new(q.view(0,q.len_bytes()).unwrap(),
                    Layout::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,tokens*5120).unwrap();
                let mut a = plan.bind_projection(&stream,view(&wa),qv,f32_view(&ya,tokens*48),tokens).unwrap();
                let mut b = plan.bind_projection(&stream,view(&wb),qv,f32_view(&yb,tokens*48),tokens).unwrap();
                plan.bind_llama_exact_input_rows_audit(&stream,&mut a,f32_view(&x,tokens*5120),None).unwrap();
                plan.bind_llama_exact_input_rows_audit(&stream,&mut b,f32_view(&x,tokens*5120),None).unwrap();
                let poison = vec![0xff; q.len_bytes()]; q.copy_from_host_sync(&stream,&poison).unwrap();
                for (generation,batch) in [0,5632,0].into_iter().enumerate() {
                    let input = std::fs::read(capture.join(format!("batch{batch}-attn_norm-{layer}.f32le.bin"))).unwrap();
                    assert_eq!(input.len(),512*5120*4); x.copy_from_host_sync(&stream,&input[..x.len_bytes()]).unwrap();
                    plan.gdn_bf16_pair = false;
                    plan.submit_shared_input_group(&stream,&[&a,&b]).unwrap();
                    let mut ra = vec![0;ya.len_bytes()]; let mut rb = vec![0;yb.len_bytes()];
                    ya.copy_to_host_sync(&stream,&mut ra).unwrap(); yb.copy_to_host_sync(&stream,&mut rb).unwrap();
                    assert!(ra.chunks_exact(4).chain(rb.chunks_exact(4)).all(|v|
                        f32::from_le_bytes(v.try_into().unwrap()).is_finite()));
                    if tokens == 512 {
                        assert!(ra == std::fs::read(capture.join(format!("batch{batch}-alpha-{layer}.f32le.bin"))).unwrap());
                        assert!(rb == std::fs::read(capture.join(format!("batch{batch}-beta-{layer}.f32le.bin"))).unwrap());
                    }
                    plan.gdn_bf16_pair = true;
                    assert!(plan.submit_shared_input_group(&foreign,&[&a,&b]).is_err());
                    let prior = plan.gdn_bf16_pair_submissions();
                    let allocated = crate::cuda_device_allocation_snapshot();
                    // Reversed table order exercises fresh pointer bindings as well as input generations.
                    for group in [[&a,&b],[&b,&a]] {
                        plan.submit_shared_input_group(&stream,&group).unwrap();
                        let mut actual = vec![0;ya.len_bytes()]; ya.copy_to_host_sync(&stream,&mut actual).unwrap();
                        assert!(actual==ra,"alpha layer={layer} tokens={tokens} generation={generation}");
                        yb.copy_to_host_sync(&stream,&mut actual).unwrap();
                        assert!(actual==rb,"beta layer={layer} tokens={tokens} generation={generation}");
                    }
                    assert_eq!(crate::cuda_device_allocation_snapshot().successful_allocations,allocated.successful_allocations);
                    assert_eq!(plan.gdn_bf16_pair_submissions()-prior,if tokens>=64 {2} else {0});
                    paired += if tokens>=64 {2} else {0};
                    let mut untouched = poison.clone(); q.copy_to_host_sync(&stream,&mut untouched).unwrap();
                    assert!(untouched==poison,"BF16 original path must not read/write Q8 scratch");
                    if tokens==64 && generation==0 {
                        for mutation in 0..10 {
                            let mut args=[a.original_input.unwrap(),b.original_input.unwrap(),a.original_input.unwrap()];
                            let mut count=2;
                            match mutation {
                                0=>args[1].input_f32+=4,
                                1=>args[1].output_f32=args[0].output_f32,
                                2=>args[1].output_f32=args[0].weight,
                                3=>args[1].weight_format=AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1,
                                4=>args[0].tokens=63,
                                5=>args[1].rows=47,
                                6=>count=1,
                                7=>count=3,
                                8=>args[0].tensor_scale=2.0,
                                _=>args[1].input_len_bytes-=4,
                            }
                            // SAFETY: live owners retained; deliberately malformed metadata must reject before work.
                            let result=unsafe { augur_cuda_sys::augur_cuda_qwen35_prefill_llama_projection_group_rows_prevalidated_submit_v1(
                                0,stream.raw().unwrap(),plan.raw,args.as_ptr(),count) };
                            assert_ne!(result.status,0,"native mutation={mutation}");
                            rejected+=1;
                        }
                        let mut actual=vec![0;ya.len_bytes()]; ya.copy_to_host_sync(&stream,&mut actual).unwrap(); assert!(actual==ra);
                        yb.copy_to_host_sync(&stream,&mut actual).unwrap(); assert!(actual==rb);
                    }
                    cases+=1;
                }
                stream.synchronize().unwrap();
            }
        }
    }
    let after=crate::cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes,before.live_bytes); assert_eq!(after.live_allocations,before.live_allocations);
    assert_eq!(after.successful_allocations-before.successful_allocations,after.successful_frees-before.successful_frees);
    println!("GDN_PAIR_REAL_EXACT cases={cases} paired_submissions={paired} rejected={rejected} captured_layers=0,50 changed_generations=3 live_bytes_delta=0 live_allocations_delta=0");
}

fn f32_view(buffer: &OwnedCudaBuffer, elements: usize) -> CudaF32View {
    CudaF32View::new(buffer.view(0, elements * 4).unwrap(),
        Layout::DEVICE_CONTIGUOUS_F32_CUDA_V1, elements).unwrap()
}

fn fixture() -> (CudaQwen35PrefillCublasProjectionV1, augur_cuda_sys::AugurCudaQ8_1ActivationQuantizeArgsV1) {
    let mut raw = AugurCudaQwen35PrefillCublasProjectionArgsV1::new();
    raw.weight_format = 1; raw.tokens = 512; raw.rows = 1024; raw.cols = 5120;
    raw.input_q8_1 = 0x20000000; raw.input_len_bytes = 512 * 5120 / 32 * 36;
    let input = AugurCudaQwen35PrefillF32ProjectionArgsV1 {
        abi_version: 1, struct_size: core::mem::size_of::<AugurCudaQwen35PrefillF32ProjectionArgsV1>() as u32,
        weight_format: 1, tokens: raw.tokens, rows: raw.rows, cols: raw.cols,
        weight: 0x10000000, weight_len_bytes: 1024 * 5120 / 32 * 34,
        input_f32: 0x30000000, input_len_bytes: 512 * 5120 * 4,
        output_f32: 0x40000000, output_len_bytes: 512 * 1024 * 4,
        tensor_scale: 1.0, reserved: 0,
    };
    let p = CudaQwen35PrefillCublasProjectionV1 {
        raw, weight_decode_lanes: 1, quantized_small_batch: false,
        exact_projection_tiles: false, projection_row_reuse: false,
        ffn_down_row_reuse: false,
        canonical_tiled_rows: false, canonical_grid_rows: false,
        tensor_ffn_down: false, tensor_ffn_down_compensated: false,
        integer_mma_ffn: false, integer_mma_input: false, llama_f16_input: false,
        original_input: Some(input), llama_full_original: true, llama_exact_input_rows: true,
        device_ordinal: 0, stream_owner_instance_key: 0, plan_instance_key: 0,
    };
    let mut q = augur_cuda_sys::AugurCudaQ8_1ActivationQuantizeArgsV1::new();
    q.input_f32 = input.input_f32; q.input_len_bytes = input.input_len_bytes;
    q.output_q8_1 = raw.input_q8_1; q.output_len_bytes = raw.input_len_bytes;
    q.elements = 512 * 5120;
    (p, q)
}

#[test]
fn submission_cleanup_only_elides_exact_original_consumers() {
    let (p, q) = fixture();
    assert!(original_input_covers_pack(&p, &q));
    for mutation in 0..22 {
        let mut b = p;
        match mutation {
            0 => b.original_input = None, 1 => b.llama_full_original = false,
            2 => b.weight_decode_lanes = 2, 3 => b.quantized_small_batch = true,
            4 => b.exact_projection_tiles = true, 5 => b.projection_row_reuse = true,
            6 => b.canonical_tiled_rows = true, 7 => b.canonical_grid_rows = true,
            8 => b.tensor_ffn_down = true, 9 => b.tensor_ffn_down_compensated = true,
            10 => b.integer_mma_ffn = true, 11 => b.integer_mma_input = true,
            12 => b.llama_f16_input = true, 13 => b.raw.input_q8_1 += 4,
            14 => b.raw.input_len_bytes -= 4, 15 => b.raw.tokens -= 1,
            16 => b.raw.cols -= 1, 17 => b.original_input.as_mut().unwrap().input_f32 += 4,
            18 => b.original_input.as_mut().unwrap().input_len_bytes -= 4,
            19 => b.original_input.as_mut().unwrap().tokens = 0,
            20 => b.original_input.as_mut().unwrap().tokens = 513,
            _ => b.original_input.as_mut().unwrap().cols -= 1,
        }
        assert!(!original_input_covers_pack(&b, &q), "mutation {mutation}");
        assert!(![&p, &b].iter().all(|x| original_input_covers_pack(x, &q)));
    }
    let mut wrong = q; wrong.elements -= 32;
    assert!(!original_input_covers_pack(&p, &wrong));
}

fn weights(format: u32, rows: usize, cols: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; match format { 1 => rows * cols / 32 * 34,
        2 => rows * cols / 64 * 36, _ => rows * cols * 2 }];
    if format == 3 {
        for (i, b) in bytes.chunks_exact_mut(2).enumerate() {
            b.copy_from_slice(&(0x3b80u16 + (i % 127) as u16 | if i % 3 == 0 {0x8000} else {0}).to_le_bytes());
        }
    } else {
        let block = if format == 1 {34} else {36};
        for (i, b) in bytes.chunks_exact_mut(block).enumerate() {
            if format == 1 {
                b[..2].copy_from_slice(&0x2400u16.to_le_bytes());
                for (j, x) in b[2..].iter_mut().enumerate() { *x = (i * 37 + j * 13) as u8; }
            } else {
                b[..4].copy_from_slice(&[0x30, 0x32, 0x34, 0x36]);
                for (j, x) in b[4..].iter_mut().enumerate() { *x = (i * 31 + j * 17) as u8; }
            }
        }
    }
    bytes
}

#[test]
#[ignore = "exclusive SM70: unchanged projection arithmetic, poison unused Q8, real shapes and row tails"]
fn submission_cleanup_original_projection_gpu_exact() {
    let before = crate::cuda_device_allocation_snapshot();
    let mut cases = 0;
    {
        let stream = OwnedCudaStream::create(0).unwrap();
        let foreign = OwnedCudaStream::create(0).unwrap();
        for (format, rows, cols) in [(1,1024,5120), (3,48,5120), (2,17408,5120), (2,5120,17408)] {
            let data = weights(format, rows, cols);
            let w = OwnedCudaBuffer::allocate(0, data.len()).unwrap();
            w.copy_from_host_sync(&stream, &data).unwrap();
            let weight = match format {
                1 => CudaQwen35PrefillWeightV1::Q8_0(CudaQ8_0WeightViewV1::new(
                    w.view(0,data.len()).unwrap(), Layout::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,rows,cols).unwrap()),
                2 => CudaQwen35PrefillWeightV1::Nvfp4 { view: CudaNvfp4Sm70WeightViewV1::new(
                    w.view(0,data.len()).unwrap(), Layout::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,rows,cols).unwrap(), tensor_scale: 0.125 },
                _ => CudaQwen35PrefillWeightV1::Bf16(CudaBf16WeightViewV1::new(
                    w.view(0,data.len()).unwrap(), Layout::DEVICE_CONTIGUOUS_BF16_CUDA_V1,rows,cols).unwrap()),
            };
            let mut plan = CudaQwen35PrefillCublasPlanV1::new(&stream,rows*cols,cols,512).unwrap();
            for tokens in [1,8,31,63,64,285,512] {
                let x = OwnedCudaBuffer::allocate(0,tokens*cols*4).unwrap();
                let q = OwnedCudaBuffer::allocate(0,tokens*cols/32*36).unwrap();
                let y = OwnedCudaBuffer::allocate(0,tokens*rows*4).unwrap();
                let y2 = OwnedCudaBuffer::allocate(0,tokens*rows*4).unwrap();
                let qv = CudaQ8_1ActivationViewV1::new(q.view(0,q.len_bytes()).unwrap(),
                    Layout::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,tokens*cols).unwrap();
                let pack = CudaQ8_1ActivationQuantizeArgsV1::bind(&stream,f32_view(&x,tokens*cols),qv).unwrap();
                let mut p = plan.bind_projection(&stream,weight,qv,f32_view(&y,tokens*rows),tokens).unwrap();
                plan.bind_llama_exact_input_rows_audit(&stream,&mut p,f32_view(&x,tokens*cols),None).unwrap();
                let mut p2 = plan.bind_projection(&stream,weight,qv,f32_view(&y2,tokens*rows),tokens).unwrap();
                plan.bind_llama_exact_input_rows_audit(&stream,&mut p2,f32_view(&x,tokens*cols),None).unwrap();
                for generation in 0..2 {
                    let input: Vec<u8> = (0..tokens*cols).flat_map(|i|
                        (((i*17 + generation*71)%257) as f32 / 128.0 - 1.0).to_le_bytes()).collect();
                    x.copy_from_host_sync(&stream,&input).unwrap();
                    plan.skip_unused_q8 = false;
                    plan.submit_required_q8_input(&stream,&pack,&[&p,&p2]).unwrap();
                    plan.submit(&stream,&p).unwrap(); plan.submit(&stream,&p2).unwrap();
                    let mut reference = vec![0; y.len_bytes()]; y.copy_to_host_sync(&stream,&mut reference).unwrap();
                    assert!(reference.chunks_exact(4).all(|v|f32::from_le_bytes(v.try_into().unwrap()).is_finite()));
                    let poison = vec![0xff; q.len_bytes()]; q.copy_from_host_sync(&stream,&poison).unwrap();
                    plan.skip_unused_q8 = true;
                    assert!(plan.submit_required_q8_input(&foreign,&pack,&[&p]).is_err());
                    assert!(plan.submit_required_q8_input(&stream,&pack,&[]).is_err());
                    let count = plan.q8_input_pack_elisions();
                    plan.submit_required_q8_input(&stream,&pack,&[&p,&p2]).unwrap();
                    assert_eq!(plan.q8_input_pack_elisions(), count+1);
                    plan.submit_shared_input_group(&stream,&[&p,&p2]).unwrap();
                    let mut actual = vec![0; y.len_bytes()]; y.copy_to_host_sync(&stream,&mut actual).unwrap();
                    assert!(actual==reference,"format={format} rows={tokens} generation={generation}");
                    y2.copy_to_host_sync(&stream,&mut actual).unwrap(); assert!(actual==reference);
                    let mut unchanged = poison.clone(); q.copy_to_host_sync(&stream,&mut unchanged).unwrap();
                    assert!(unchanged==poison,"external Q8 must not be read or written");
                    if generation == 1 {
                        let count = plan.q8_input_pack_elisions();
                        // A diagnostic reads this scratch, so it is no longer dead.
                        plan.retain_q8_for_observation();
                        plan.submit_required_q8_input(&stream,&pack,&[&p,&p2]).unwrap();
                        q.copy_to_host_sync(&stream,&mut unchanged).unwrap();
                        assert!(unchanged!=poison);
                        assert_eq!(plan.q8_input_pack_elisions(), count);
                        if tokens==8 {
                            let native=plan.bind_projection(&stream,weight,qv,f32_view(&y,tokens*rows),tokens).unwrap();
                            q.copy_from_host_sync(&stream,&poison).unwrap();
                            plan.skip_unused_q8=true;
                            plan.submit_required_q8_input(&stream,&pack,&[&p,&native]).unwrap();
                            q.copy_to_host_sync(&stream,&mut unchanged).unwrap();
                            assert!(unchanged!=poison,"mixed native B8 consumers must still pack");
                            assert_eq!(plan.q8_input_pack_elisions(),count);
                        }
                    }
                    cases += 1;
                }
                stream.synchronize().unwrap();
            }
            stream.synchronize().unwrap();
        }
    }
    let after=crate::cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes,before.live_bytes); assert_eq!(after.live_allocations,before.live_allocations);
    assert_eq!(after.successful_allocations-before.successful_allocations,after.successful_frees-before.successful_frees);
    println!("SUBMISSION_CLEANUP_EXACT cases={cases} changed_input_generations=2 live_bytes_delta=0 live_allocations_delta=0");
}
