use super::super::tests::bytes;
use super::super::transfer_tests::{ids, open};
use super::*;
use augur_backend::cuda_qwen35_prefill_cublas::{
    augur_prefill_widths, augur_prefill_widths_for_batch,
    AUGUR_DEFAULT_BATCH, AUGUR_MAX_BATCH,
};
use serde_json::{json, Value};
use std::io::Write;
#[test]
fn schedule_is_literal_and_rejects_incomplete_work() {
    let w = augur_prefill_widths(8236).unwrap();
    assert_eq!(
        w,
        augur_prefill_widths_for_batch(8236, AUGUR_DEFAULT_BATCH).unwrap()
    );
    assert_eq!(
        validate_widths(&w, 8236).unwrap(),
        AUGUR_DEFAULT_BATCH
    );
    assert_eq!(Budget::default().safety, 128 * 1024 * 1024);
    for (w, n) in [
        (vec![], 0),
        (vec![0], 0),
        (
            vec![AUGUR_MAX_BATCH + 1],
            AUGUR_MAX_BATCH + 1,
        ),
        (vec![512], 8236),
    ] {
        assert!(validate_widths(&w, n).is_err());
    }
    assert_ne!(ids_hash(&[1, 2, 3]), ids_hash(&[3, 2, 1]));
    if AUGUR_MAX_BATCH >= 4352 {
        assert_eq!(
            augur_prefill_widths_for_batch(8236, 4352).unwrap(),
            vec![4352, 3884]
        );
        assert!(Phase::owned_recipe(4352).unwrap() < 3200 * 1024 * 1024);
    }
}
#[test]
fn resident_source_recipe_moves_only_immutable_gate_weights() {
    let immutable=48*gdn::Source::PADDED_BYTES;
    assert_eq!(immutable,17_694_720);
    for width in [1,8,55,63,64,73,79,511,512,4352,6400]
        .into_iter().filter(|&n|n<=AUGUR_MAX_BATCH) {
        let phase=Phase::owned_recipe(width).unwrap();
        assert_eq!(Phase::owned_recipe_with_sources(width,false).unwrap(),phase);
        assert_eq!(Phase::owned_recipe_with_sources(width,true).unwrap()+immutable,phase);
    }
    assert!(Phase::owned_recipe_with_sources(0,true).is_err());
}
#[test]
fn phase_memory_is_bounded_without_full_history_expansion() {
    for width in [1, 7, 8, 9, 44, 512] {
        let bytes = Phase::owned_recipe(width).unwrap();
        assert!(bytes < 700 * 1024 * 1024);
        assert!(bytes > 17408 * 5120 * 2);
    }
    assert!(Phase::owned_recipe(0).is_err());
    assert!(Phase::owned_recipe(AUGUR_MAX_BATCH + 1).is_err());
    for width in [512, 1024, 2048, 4096]
        .into_iter()
        .filter(|&n| n <= AUGUR_MAX_BATCH)
    {
        assert!(Phase::owned_recipe(width).unwrap() < 3 * 1024 * 1024 * 1024);
        let widths = augur_prefill_widths_for_batch(8236, width).unwrap();
        assert_eq!(widths.iter().sum::<usize>(), 8236);
        assert_eq!(widths.last(), Some(&44));
        assert_eq!(validate_widths(&widths, 8236).unwrap(), width);
    }
}
#[test]
fn arena_recipe_matches_liveness_bill_and_disjoint_reference() {
    for b in [512, 1024, 2048, 4096, 4352, 5120, 6400]
        .into_iter()
        .filter(|&b| b <= AUGUR_MAX_BATCH)
    {
        assert_eq!(Phase::owned_recipe(b).unwrap(), 251068416 + 301056 * b);
        gdn::arena::with_separate(|| {
            assert_eq!(Phase::owned_recipe(b).unwrap(), 251068416 + 707456 * b)
        });
    }
}
/// Output/state bytes only; no allocation addresses, epochs or stale KV capacity.
fn completed_fingerprint(target: &mut CheckpointCudaTarget) -> Value {
    let mut hashes = BTreeMap::new();
    let logits: Vec<u8> = target
        .logits_snapshot()
        .unwrap()
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    hashes.insert(
        "logits".to_owned(),
        format!("{:x}", Sha256::digest(&logits)),
    );
    let features = target.dflash_feature_snapshot().unwrap();
    hashes.insert(
        "features".to_owned(),
        format!("{:x}", Sha256::digest(&features.little_endian_f32)),
    );
    for (i, layer) in target.layers.iter().enumerate() {
        let states = match layer {
            Layer::Linear(l) => gdn::diagnostic_committed_bytes(l).unwrap(),
            Layer::Full(l) => full::diagnostic_committed_bytes(l).unwrap(),
        };
        for (name, bytes) in states {
            hashes.insert(
                format!("layer{i}/{name}"),
                format!("{:x}", Sha256::digest(&bytes)),
            );
        }
    }
    json!(hashes)
}
fn explicit_variants(raw: &str, tokens: usize) -> Result<Vec<Vec<usize>>> {
    let variants: Vec<Vec<usize>> = raw
        .split(';')
        .map(|plan| {
            plan.split('+')
                .map(|v| v.parse::<usize>())
                .collect::<std::result::Result<Vec<_>, _>>()
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if variants.is_empty() || variants.len() > 3 {
        return Err("one to three complete benchmark plans required".into());
    }
    for plan in &variants {
        validate_widths(plan, tokens)?;
    }
    Ok(variants)
}
#[test]
fn explicit_benchmark_plans_never_drop_or_pad_a_tail() {
    assert_eq!(
        explicit_variants("512+256+256;512+512", 1024).unwrap(),
        vec![vec![512, 256, 256], vec![512, 512]]
    );
    for invalid in [
        "",
        "512+256",
        "512+513",
        "0+512+512",
        "512++512",
        "garbage",
        "1024;1024;1024;1024",
    ] {
        assert!(explicit_variants(invalid, 1024).is_err());
    }
    if AUGUR_MAX_BATCH >= 4352 {
        assert_eq!(
            explicit_variants("4352+3884;4352+2048+1836", 8236).unwrap(),
            vec![vec![4352, 3884], vec![4352, 2048, 1836]]
        );
        assert!(explicit_variants("4352+2048", 8236).is_err());
    }
}
fn write_json(path: &Path, v: &Value) {
    std::fs::write(path, serde_json::to_vec_pretty(v).unwrap()).unwrap();
}
fn validate_completed(target: &mut CheckpointCudaTarget, count: usize) -> Value {
    target.validate_cursor().unwrap();
    assert_eq!(target.consumed, count);
    assert!(target.augur_rotated_kv);
    let logits = target.logits_snapshot().unwrap();
    let snapshot = target.dflash_feature_snapshot().unwrap();
    let rows = count.min(2048);
    assert_eq!(snapshot.receipt.context_len, rows);
    assert_eq!(snapshot.receipt.context_start_position, count - rows);
    for layer in &mut target.layers {
        if let Layer::Linear(layer) = layer {
            for name in ["state", "history"] {
                assert!(
                    layer.readback(name).unwrap().iter().all(|v| v.is_finite()),
                    "nonfinite GDN {name}"
                );
            }
        }
    }
    // This is an intentional product-boundary guard, NOT a numerical failure.
    assert!(target
        .consume_sequence(&[1])
        .unwrap_err()
        .to_string()
        .contains("rotated KV"));
    assert!(target
        .consume_selected()
        .unwrap_err()
        .to_string()
        .contains("rotated KV"));
    assert_eq!(target.consumed, count);
    target.validate_cursor().unwrap();
    json!({"finite_logits":logits.len(),"finite_gdn_states":48,"all_layer_frontiers":count,
        "feature_receipt":snapshot.receipt,"ordinary_decode_rejected":true})
}
#[test]
#[ignore = "explicit single-GPU original AUGUR8236-token speed-first benchmark"]
fn real_checkpoint_augur_8k_tps() {
    let fixture = std::env::var_os("AUGUR_CHECKPOINT_TARGET_REFERENCE").expect("fixture");
    let fixture_bytes = std::fs::read(&fixture).unwrap();
    let case: Value = serde_json::from_slice(&fixture_bytes).unwrap();
    let prompt = ids(&case["context_request"]["token_ids"]);
    assert_eq!(
        prompt.len(),
        8236,
        "use real context IDs, never the4-token stub"
    );
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(bytes(&case["metadata"]["tokenizer.json"])).unwrap();
    assert_eq!(
        tokenizer
            .encode(
                case["context_request"]["canonical_prompt"]
                    .as_str()
                    .unwrap(),
                false
            )
            .unwrap()
            .get_ids(),
        prompt
    );
    let output =
        std::path::PathBuf::from(std::env::var_os("AUGUR_CHECKPOINT_TARGET_OUTPUT").expect("output"));
    std::fs::create_dir(&output).unwrap();
    let mut progress = std::fs::File::create(output.join("progress.jsonl")).unwrap();
    let baseline = cuda_device_allocation_snapshot();
    let load = Instant::now();
    let mut target = open(&case);
    target.enable_augur_feature_capture().unwrap();
    target.stream.synchronize().unwrap();
    let load_seconds = load.elapsed().as_secs_f64();
    writeln!(
        progress,
        "{}",
        json!({"stage":"loaded","seconds":load_seconds,"info":target.info()})
    )
    .unwrap();
    progress.flush().unwrap();
    // B7 -> B9 continuation checks both flash adapters and true absolute positions.
    for (start, n) in [(0, 7), (7, 9)] {
        let ids = &prompt[start..start + n];
        let plan = target.plan_augur_sequence(ids, vec![n]).unwrap();
        target
            .consume_augur_impl(
                ids,
                &plan,
                Budget::default(),
                &mut |_| false,
                #[cfg(feature = "qwen35-augur-vision")]
                None,
                Some(&mut |p, s, _, out, _| {
                    let mut b = vec![0; p.rows * H * 4];
                    out.view(0, b.len())?.copy_to_host_sync(s, &mut b)?;
                    if b.chunks_exact(4)
                        .any(|x| !f32::from_le_bytes(x.try_into().unwrap()).is_finite())
                    {
                        return Err("smoke nonfinite hidden".into());
                    }
                    Ok(())
                }),
            )
            .unwrap();
    }
    target.greedy_next_token().unwrap();
    let smoke = validate_completed(&mut target, 16);
    writeln!(
        progress,
        "{}",
        json!({"stage":"smoke_pass","validation":smoke})
    )
    .unwrap();
    progress.flush().unwrap();
    // Resident model/setup remains outside prefill timing, exactly like target load.
    let draft_load = Instant::now();
    let before_draft = cuda_device_allocation_snapshot();
    let draft_bridge = match std::env::var("AUGUR_DFLASH_RESIDENT").as_deref() {
        Err(std::env::VarError::NotPresent) => None,
        Ok("1") => {
            let cfg =
                std::fs::read(std::env::var_os("AUGUR_DRAFT_CONFIG").expect("draft config"))
                    .unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&cfg)),
                std::env::var("AUGUR_DRAFT_CONFIG_SHA256").unwrap()
            );
            let q8 = crate::DFlashQ8Backbone::load_pinned(
                std::env::var_os("AUGUR_DRAFT_PATH").expect("draft path"),
                augur_model_qwen35::DFlashConfig::parse_f16_opt_in(&cfg).unwrap(),
            )
            .unwrap();
            assert_eq!(
                q8.manifest().file_sha256.as_ref().unwrap(),
                &std::env::var("AUGUR_DRAFT_SHA256").unwrap()
            );
            let weights = crate::dflash_q8_backbone_cuda::DFlashQ8CudaWeights::upload(
                &q8,
                target.stream.device_ordinal(),
            )
            .unwrap();
            let mut bridge =
                super::super::dflash_bridge::AugurDflashBridge::new(&target, &weights).unwrap();
            bridge
                .retain_prefill_benchmark_workspace(&mut target)
                .unwrap();
            assert_eq!(
                bridge.resident_info()["q8_draft_resident_bytes"],
                2045465600u64
            );
            assert_eq!(
                bridge.resident_workspace_info()["draft_workspace_bytes"],
                399708736u64
            );
            Some(bridge)
        }
        _ => panic!("invalid DFlash residency selection"),
    };
    let draft_load_seconds = draft_load.elapsed().as_secs_f64();
    let dflash_residency = match &draft_bridge {
        Some(b) => json!({"enabled":true,"weights_and_real_decode_workspaces_retained":true,
            "draft_execution_during_prefill":false,"experimental_feature_decode_qualified":false,
            "bridge":b.resident_info(),"workspace":b.resident_workspace_info(),
            "added_owned_bytes":cuda_device_allocation_snapshot().live_bytes-before_draft.live_bytes,
            "load_seconds_excluded_from_prefill":draft_load_seconds}),
        None => json!({"enabled":false}),
    };
    writeln!(
        progress,
        "{}",
        json!({"stage":"dflash_residency","receipt":dflash_residency})
    )
    .unwrap();
    progress.flush().unwrap();
    // Same F16/F32 math and B512 schedule. This checks storage reuse, not F64 fidelity.
    let arena_check = match std::env::var("AUGUR_ARENA_CHECK").as_deref() {
        Err(std::env::VarError::NotPresent) => json!({"performed":false}),
        Ok("1") => {
            let mut fingerprints = Vec::new();
            let mut receipts = Vec::new();
            for separate in [true, false] {
                target.reset_empty().unwrap();
                let mut execute = || {
                    let plan = target
                        .plan_augur_sequence(
                            &prompt,
                            augur_prefill_widths_for_batch(prompt.len(), 512).unwrap(),
                        )
                        .unwrap();
                    target
                        .consume_sequence_augur(&prompt, &plan, Budget::default(), |_| false)
                        .unwrap()
                };
                let receipt = if separate {
                    gdn::arena::with_separate(execute)
                } else {
                    execute()
                };
                validate_completed(&mut target, prompt.len());
                fingerprints.push(completed_fingerprint(&mut target));
                receipts.push(receipt);
            }
            write_json(
                &output.join("arena-layout-check.json"),
                &json!({"fingerprints":fingerprints,"receipts":receipts}),
            );
            assert_eq!(
                fingerprints[0], fingerprints[1],
                "lifetime arena changed same-schedule output/state bytes"
            );
            json!({"performed":true,"bitwise_equal":true,"tokens":prompt.len(),"batch":512,"math":"same_F16_F32_not_F64"})
        }
        _ => panic!("invalid arena check selection"),
    };
    let diagnostic = match std::env::var("AUGUR_PROFILE").as_deref() {
        Err(std::env::VarError::NotPresent) => false,
        Ok("1") => true,
        _ => panic!("invalid profiler selection"),
    };
    let variants: Vec<Vec<usize>> = match std::env::var("AUGUR_PLANS") {
        Ok(raw) => {
            assert!(
                std::env::var_os("AUGUR_BATCHES").is_none(),
                "select plans OR fixed batches, not both"
            );
            explicit_variants(&raw, prompt.len()).unwrap()
        }
        Err(std::env::VarError::NotPresent) => std::env::var("AUGUR_BATCHES")
            .unwrap_or_else(|_| AUGUR_DEFAULT_BATCH.to_string())
            .split(',')
            .map(|n| {
                augur_prefill_widths_for_batch(
                    prompt.len(),
                    n.parse().expect("integer superbatch"),
                )
                .unwrap()
            })
            .collect(),
        Err(e) => panic!("invalid plan selection: {e}"),
    };
    assert!(!variants.is_empty() && variants.len() <= 3);
    assert!(!diagnostic || variants.len() == 1);
    let mut schedule: Vec<(bool, usize)> = (0..variants.len()).map(|i| (true, i)).collect();
    if diagnostic {
        schedule.push((false, 0));
    } else if variants.len() == 1 {
        schedule.extend((0..3).map(|_| (false, 0)));
    } else {
        let rounds: usize = std::env::var("AUGUR_COMPARE_ROUNDS")
            .unwrap_or_else(|_| "1".into())
            .parse()
            .unwrap();
        assert!((1..=4).contains(&rounds));
        for _ in 0..rounds {
            schedule.extend((0..variants.len()).map(|i| (false, i)));
            schedule.extend((0..variants.len()).rev().map(|i| (false, i)));
        }
    }
    let safety_mib: u64 = std::env::var("AUGUR_SAFETY_MIB")
        .unwrap_or_else(|_| "128".into())
        .parse()
        .unwrap();
    assert!(safety_mib <= 4096);
    let benchmark_budget = Budget {
        safety: safety_mib * 1024 * 1024,
        ..Budget::default()
    };
    let mut runs = Vec::new();
    for (pass, (warmup, variant_index)) in schedule.into_iter().enumerate() {
        let widths = &variants[variant_index];
        let batch = *widths.iter().max().unwrap();
        let variant = widths
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("+");
        target.reset_empty().unwrap();
        target.stream.synchronize().unwrap();
        let before = cuda_device_allocation_snapshot();
        let mut capture = if diagnostic && pass == 1 {
            Some(augur_backend::CudaProfilerCapture::start().unwrap())
        } else {
            None
        };
        let wall = Instant::now();
        let plan = target
            .plan_augur_sequence(&prompt, widths.clone())
            .unwrap();
        let receipt = target
            .consume_sequence_augur(&prompt, &plan, benchmark_budget, |_| false)
            .unwrap();
        let prefill_seconds = wall.elapsed().as_secs_f64();
        let anchor = target.greedy_next_token().unwrap();
        let conditioning_seconds = wall.elapsed().as_secs_f64();
        if let Some(c) = capture.as_mut() {
            c.stop().unwrap();
        }
        // Full diagnostic D2H is strictly OUTSIDE both reported timing intervals.
        assert_eq!(receipt.hot_allocations, 0);
        assert_eq!(receipt.hot_frees, 0);
        assert!(!receipt.diagnostic_io);
        let after = cuda_device_allocation_snapshot();
        assert_eq!(before.live_bytes, after.live_bytes);
        assert_eq!(before.live_allocations, after.live_allocations);
        let validation = validate_completed(&mut target, prompt.len());
        let record = json!({"pass":pass,"warmup":warmup,"batch_rows":batch,"variant":variant,"variant_index":variant_index,"tokens":prompt.len(),
            "prefill_seconds":prefill_seconds,"conditioning_wall_seconds":conditioning_seconds,
            "prefill_tps":if diagnostic {None} else {Some(prompt.len() as f64/prefill_seconds)},
            "conditioning_tps":if diagnostic {None} else {Some(prompt.len() as f64/conditioning_seconds)},
            "diagnostic_profiler":diagnostic,
            "anchor_token":anchor,"receipt":receipt,"validation":validation});
        writeln!(progress, "{record}").unwrap();
        progress.flush().unwrap();
        println!("AUGUR_8K {}", record);
        runs.push(record);
        write_json(&output.join("partial.json"), &json!({"runs":runs}));
    }
    let info = target.info().clone();
    drop(draft_bridge);
    drop(target);
    let after = cuda_device_allocation_snapshot();
    assert_eq!(baseline.live_bytes, after.live_bytes);
    assert_eq!(baseline.live_allocations, after.live_allocations);
    write_json(
        &output.join("result.json"),
        &json!({"schema":"quasar-quasar8k-tps-v1","status":if diagnostic {"PASS_DIAGNOSTIC_ONLY"} else {"PASS_EXECUTION_AND_TIMING"},
        "profile":PROFILE,"workspace_profile":gdn::arena::profile(),"benchmark_safety_bytes":benchmark_budget.safety,"dflash_residency":dflash_residency,"arena_check":arena_check,"fixture_sha256":format!("{:x}",Sha256::digest(&fixture_bytes)),
        "token_ids_sha256":ids_hash(&prompt),"load_seconds":load_seconds,"info":info,"runs":runs,
        "model_cleanup_live_bytes":after.live_bytes,"model_cleanup_live_allocations":after.live_allocations,
        "numerical_qualified":false,"http_qualified":false,"full_capacity":262144,"kv_bytes":16u64*2*262144*1088,
        "timing_includes":"setup,upload,64layers,2048feature-history,last-row-BF16-head,sync,phase-release,4B-anchor-read",
        "timing_excludes":"model-load,reset,full-diagnostic-readback"}),
    );
}

#[test]
#[ignore = "exclusive GPU; exact resident gate source and reset/cancel state comparison"]
fn real_checkpoint_resident_sources_equivalence() {
    let fixture=std::env::var_os("AUGUR_CHECKPOINT_TARGET_REFERENCE").expect("fixture");
    let case:Value=serde_json::from_slice(&std::fs::read(fixture).unwrap()).unwrap();
    let prompt=ids(&case["context_request"]["token_ids"]);
    let output=std::path::PathBuf::from(std::env::var_os("AUGUR_CHECKPOINT_TARGET_OUTPUT").expect("output"));
    std::fs::create_dir(&output).unwrap();
    let before=cuda_device_allocation_snapshot();let mut target=open(&case);
    target.enable_augur_feature_capture().unwrap();
    let widths=[1,8,55,63,64,73,79,327,512];let mut expected=Vec::new();let mut receipts=Vec::new();
    for n in widths {
        target.reset_empty().unwrap();
        let plan=target.plan_augur_sequence(&prompt[..n],vec![n]).unwrap();
        let r=target.consume_sequence_augur(&prompt[..n],&plan,Budget::default(),|_|false).unwrap();
        assert!(!r.padded_gates_reused);assert_eq!(r.resident_padded_gate_bytes,0);
        expected.push(completed_fingerprint(&mut target));receipts.push(serde_json::to_value(r).unwrap());
    }
    target.reset_empty().unwrap();let base=cuda_device_allocation_snapshot();
    let installed=target.retain_augur_source_weights().unwrap();
    let resident=cuda_device_allocation_snapshot();
    assert_eq!(resident.live_bytes-base.live_bytes,17_694_720);
    assert!(target.retain_augur_source_weights().is_err());
    for (index,n) in widths.into_iter().enumerate() {
        target.reset_empty().unwrap();
        let plan=target.plan_augur_sequence(&prompt[..n],vec![n]).unwrap();
        let r=target.consume_sequence_augur(&prompt[..n],&plan,Budget::default(),|_|false).unwrap();
        assert!(r.padded_gates_reused);assert_eq!(r.resident_padded_gate_bytes,17_694_720);
        assert_eq!(completed_fingerprint(&mut target),expected[index],"width {n}");
        assert_eq!(cuda_device_allocation_snapshot().live_bytes,resident.live_bytes);
        receipts.push(serde_json::to_value(r).unwrap());
    }
    target.reset_empty().unwrap();
    let plan=target.plan_augur_sequence(&prompt[..55],vec![55]).unwrap();
    assert!(target.consume_sequence_augur(&prompt[..55],&plan,Budget::default(),|p|p.layer==5).is_err());
    assert!(target.poisoned);target.reset_empty().unwrap();
    let plan=target.plan_augur_sequence(&prompt[..55],vec![55]).unwrap();
    target.consume_sequence_augur(&prompt[..55],&plan,Budget::default(),|_|false).unwrap();
    assert_eq!(completed_fingerprint(&mut target),expected[2]);
    drop(target);let after=cuda_device_allocation_snapshot();
    assert_eq!(before.live_bytes,after.live_bytes);assert_eq!(before.live_allocations,after.live_allocations);
    write_json(&output.join("result.json"),&json!({"status":"PASS","diagnostic_only":true,
        "widths":widths,"full_logits_features_layer_state_history_kv_exact":true,
        "cancel_reset_replay_exact":true,"resident_bytes":17_694_720,"install":installed,
        "receipts":receipts,"fingerprints":expected,"cleanup_live_bytes":after.live_bytes,
        "cleanup_live_allocations":after.live_allocations}));
}
