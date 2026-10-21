//! Explicit ModelOptW4A16 QPN pilot; no public admission or DFlash integration.
use super::*;

fn compare(actual: &[u8], reference: &[u8]) -> Value {
    assert_eq!(actual.len(), reference.len());
    assert_eq!(actual.len() % 4, 0);
    let (mut error2, mut norm2, mut max_abs, mut peak) = (0f64, 0f64, 0f64, 0f64);
    let mut worst_index = 0usize;
    let mut bad_indices = Vec::new();
    for (index, (a, b)) in actual.chunks_exact(4).zip(reference.chunks_exact(4)).enumerate() {
        let a = f64::from(f32::from_le_bytes(a.try_into().unwrap()));
        let b = f64::from(f32::from_le_bytes(b.try_into().unwrap()));
        assert!(a.is_finite() && b.is_finite());
        let d = a - b;
        error2 += d * d;
        norm2 += b * b;
        if d.abs() > max_abs { max_abs = d.abs(); worst_index = index; }
        peak = peak.max(b.abs());
    }
    let relative_l2 = if norm2 == 0.0 {
        if error2 == 0.0 { 0.0 } else { f64::INFINITY }
    } else { (error2 / norm2).sqrt() };
    for (index, (a, b)) in actual.chunks_exact(4).zip(reference.chunks_exact(4)).enumerate() {
        let a = f64::from(f32::from_le_bytes(a.try_into().unwrap()));
        let b = f64::from(f32::from_le_bytes(b.try_into().unwrap()));
        if (a-b).abs() > 1e-4 + 1e-5 * peak { bad_indices.push(index); }
    }
    json!({"max_abs_element_index": worst_index, "max_abs_failure_indices": bad_indices,"elements": actual.len()/4, "bitwise_equal": actual == reference,
        "relative_l2": relative_l2, "max_abs": max_abs, "reference_peak": peak,
        "pass": relative_l2 <= 1e-5 && max_abs <= 1e-4 + 1e-5 * peak})
}
fn snapshot(t: &mut CheckpointCudaTarget) -> Vec<(String, Vec<u8>)> {
    assert!(t.dflash_capture.is_none());
    let mut rows = vec![("logits".into(), t.logits_snapshot().unwrap().iter().flat_map(|v| v.to_le_bytes()).collect())];
    for (i, layer) in t.layers.iter().enumerate() {
        let state = match layer {
            Layer::Linear(l) => crate::checkpoint_linear_cuda::nerkyor_batch::diagnostic_state(l).unwrap(),
            Layer::Full(l) => crate::checkpoint_full_cuda::nerkyor_batch::diagnostic_state(l).unwrap(),
        };
        for (name, raw) in state { rows.push((format!("layer-{i}-{name}"), raw)); }
    }
    rows
}
fn raw_new(path: &Path, bytes: &[u8]) {
    let mut f = fs::OpenOptions::new().create_new(true).write(true).open(path).unwrap();
    f.write_all(bytes).unwrap();
    f.sync_all().unwrap();
}
#[test]
fn nerkyor_qpn_policy_is_distinct_and_guarded() {
    let p = CheckpointProjectionPolicy::NerkyorQpnWorkspaceShapeF64Research;
    assert_ne!(p.label(), CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64.label());
    assert!(p.uses_nerkyor_qpn_research() && p.uses_nerkyor_gate_f32_cta_research());
    assert!(p.uses_qpn_projections() && p.qpn_residual() && p.qpn_workspace() && p.qpn_shape_split() && p.qpn_postscale());
    assert!(p.qpn_ffn_layer(63) && !p.qpn_ffn_layer(64));
    assert!(!p.uses_small_gate_split16() && p.small_gate_pair_schedule().is_none());
    assert!(p.admits_qpn_projection(17408, 5120) && !p.admits_qpn_projection(48, 5120));
    let identity = CheckpointExpectedIdentity {
        config_sha256: "1".repeat(64), index_sha256: "2".repeat(64), shards: Default::default(),
    };
    for (profile, kv) in [(Qwen35CheckpointProfile::QuasarNvfp4, CheckpointKvPolicy::F32),
        (Qwen35CheckpointProfile::NerkyorW4a16, CheckpointKvPolicy::Q8_0)] {
        let err = CheckpointCudaTarget::open_with_execution_policy("not-an-artifact", profile, identity.clone(), 256, p, kv).err().unwrap();
        assert!(err.to_string().contains("explicit new-model profile and F32 KV"));
    }
}
#[test]
fn numerical_monitor_rejects_out_of_contract_and_handles_zero() {
    let raw = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
    assert_eq!(compare(&raw(&[0.,1.]), &raw(&[0.,1.]))["pass"], true);
    assert_eq!(compare(&raw(&[0.,0.]), &raw(&[0.,0.]))["pass"], true);
    assert_eq!(compare(&raw(&[0.,1.01]), &raw(&[0.,1.]))["pass"], false);
    assert_eq!(compare(&raw(&[1.,0.]), &raw(&[0.,0.]))["pass"], false);
}
#[test]
#[ignore = "exclusive SM70; full logits and all committed GDN/KV numerical pilot"]
fn nerkyor_qpn_teacher_forced_full_state_contract() {
    let out = PathBuf::from(std::env::var_os("AUGUR_NODFLASH_OUTPUT").unwrap());
    fs::create_dir(&out).unwrap();
    let guard: Value = serde_json::from_slice(&fs::read(std::env::var_os("AUGUR_NERKYOR_GPU_GUARD").unwrap()).unwrap()).unwrap();
    assert_eq!(guard["gpu_compute_apps_empty"], true);
    assert_eq!(guard["queues_paused"], true);
    assert_eq!(guard["benchmark_auto_resume"], false);
    let raw = fs::read(std::env::var_os("AUGUR_NODFLASH_MANIFEST").unwrap()).unwrap();
    assert_eq!(hash(&raw), std::env::var("AUGUR_NODFLASH_MANIFEST_SHA256").unwrap());
    let manifest: Value = serde_json::from_slice(&raw).unwrap();
    let identity: CheckpointExpectedIdentity = serde_json::from_value(manifest["checkpoint"].clone()).unwrap();
    let raw = fs::read(std::env::var_os("AUGUR_NODFLASH_INPUT").unwrap()).unwrap();
    assert_eq!(hash(&raw), std::env::var("AUGUR_NODFLASH_INPUT_SHA256").unwrap());
    let input: Value = serde_json::from_slice(&raw).unwrap();
    let kind = input["candidate"].as_str().unwrap();
    let candidate_policy = match kind {
        "qpn" => CheckpointProjectionPolicy::NerkyorQpnWorkspaceShapeF64Research,
        "qpn-source-pair" => CheckpointProjectionPolicy::NerkyorQpnSourcePairResearch,
        "qpn-source-reassoc" => CheckpointProjectionPolicy::NerkyorQpnSourceReassocResearch,
        "qpn-ffn" => CheckpointProjectionPolicy::NerkyorQpnFfnShapeF64Research,
        "source-split8" => CheckpointProjectionPolicy::NerkyorSourceSplit8F64Research,
        _ => panic!("unknown numerical candidate"),
    };
    let extended = match input["suite"].as_str() {
        None | Some("short") => false,
        Some("source-pair-extended") => {assert!(matches!(kind, "qpn-source-pair" | "qpn-source-reassoc")); true},
        _ => panic!("unknown numerical suite"),
    };
    let capacity = if extended {4096} else {256};
    let root = PathBuf::from(std::env::var_os("AUGUR_NODFLASH_MODEL_ROOT").unwrap());
    let mut prompt = ids(&input["cases"][0]["token_ids"]);
    // The committed contract is 128 prompt tokens. The override exists only so a
    // diagnostic run can put a specific prefill position last in the sequence:
    // the per-layer snapshots capture the most recent token, so the only way to
    // see what layer 43's projections actually received at position 118 is to
    // make position 118 the last one.
    let truncated = match std::env::var_os("AUGUR_NODFLASH_PROMPT_TOKENS")
        .map(|v| v.to_string_lossy().parse::<usize>().unwrap())
    {
        Some(n) => {
            assert!(n > 32 && n <= prompt.len(), "prompt override out of range");
            prompt.truncate(n);
            true
        }
        None => false,
    };
    if extended {
        assert!(!truncated, "diagnostic override cannot qualify extended context");
        assert!(prompt.len() > 128 && prompt.len() + 32 <= capacity);
    } else if !truncated {
        assert_eq!(prompt.len(), 128);
    }
    let prompt_tokens = prompt.len();
    let checkpoints = [0usize, 1, 8, 32];
    let baseline = cuda_device_allocation_snapshot();
    let mut reference_ids = Vec::new();
    let mut reference_manifests = Vec::new();
    let mut comparisons = Vec::new();
    let mut tokens_equal = true;
    let mut submissions = Vec::new();
    for candidate in [false, true] {
        let p = if candidate { candidate_policy }
            else { CheckpointProjectionPolicy::NerkyorBf16GateF32CtaResearch };
        let mut t = CheckpointCudaTarget::open_with_projection_policy(&root, Qwen35CheckpointProfile::NerkyorW4a16, identity.clone(), capacity, p).unwrap();
        save(&out, if candidate { "candidate-execution.json" } else { "reference-execution.json" }, &t.info);
        let submitted_before = crate::checkpoint_projection_cuda::projection_submission_counts();
        for (i, chunk) in prompt.chunks(8).enumerate() {
            t.consume_sequence(chunk).unwrap();
            if extended && ((i+1)*8).is_multiple_of(512) {
                eprintln!("NUMERICAL_PROGRESS lane={} consumed={}", if candidate {"candidate"} else {"reference"}, t.consumed);
            }
        }
        t.stream.synchronize().unwrap();
        for step in 0..=32 {
            let selected = t.greedy_next_token().unwrap();
            if candidate { tokens_equal &= selected == reference_ids[step]; }
            else { reference_ids.push(selected); }
            if checkpoints.contains(&step) {
                for (name, bytes) in snapshot(&mut t) {
                    let file = format!("reference-s{step}-{name}.bin");
                    if candidate {
                        let raw = fs::read(out.join(&file)).unwrap();
                        let item = reference_manifests.iter().find(|v: &&Value| v["file"] == file).unwrap();
                        assert_eq!(hash(&raw), item["sha256"].as_str().unwrap());
                        let metric = compare(&bytes, &raw);
                        if metric["pass"] != true && step == 0 { raw_new(&out.join(format!("candidate-failed-s{step}-{name}.bin")), &bytes); }
                        comparisons.push(json!({"step": step, "name": name, "metric": metric, "candidate_sha256": hash(&bytes)}));
                    } else {
                        raw_new(&out.join(&file), &bytes);
                        reference_manifests.push(json!({"file": file, "sha256": hash(&bytes), "bytes": bytes.len()}));
                    }
                }
            }
            if step < 32 { t.consume_sequence(&[reference_ids[step]]).unwrap(); }
        }
        let submitted_after = crate::checkpoint_projection_cuda::projection_submission_counts();
        let counts: Vec<u64> = submitted_after.iter().zip(submitted_before).map(|(a,b)| a-b).collect();
        if candidate && kind == "qpn-source-pair" {
            assert_eq!(counts, vec![0, 400 * (prompt_tokens as u64 + 32), 0, 0], "every candidate projection must execute the QPN source-pair path");
        }
        if candidate && kind == "source-split8" {
            assert_eq!(counts[3], 400 * (prompt_tokens as u64 + 32), "SplitK8 must execute rather than only label the owner");
        }
        if !candidate { assert_eq!(counts[2], 400 * (prompt_tokens as u64 + 32)); }
        submissions.push(json!({"policy":p.label(),"qpn_tensor":counts[0],"qpn_source_pair":counts[1],"source_nvfp4":counts[2],"source_split8":counts[3]}));
        assert_eq!(t.consumed, prompt_tokens + 32);
        assert!(t.dflash_capture.is_none());
        drop(t);
        let after = cuda_device_allocation_snapshot();
        assert_eq!(baseline.live_bytes, after.live_bytes);
        assert_eq!(baseline.live_allocations, after.live_allocations);
    }
    let pass = tokens_equal && comparisons.iter().all(|v| v["metric"]["pass"] == true);
    save(&out, "reference-manifest.json", &json!(reference_manifests));
    save(&out, "result.json", &json!({"status": if pass { if extended {"PASS_NERKYOR_TARGET_PROJECTION_EXTENDED_FULL_STATE_CONTRACT"} else {"PASS_NERKYOR_TARGET_PROJECTION_SHORT_FULL_STATE_CONTRACT"} } else { "FAIL_NERKYOR_TARGET_PROJECTION_FULL_STATE_CONTRACT" },
        "candidate":kind,"candidate_policy":candidate_policy.label(),"diagnostic_only":truncated,"formal_short_contract":!truncated && !extended,"extended_state_contract":extended,"fixed_contract": {"relative_l2":1e-5,"max_abs":"1e-4+1e-5*reference_peak"},"prompt_tokens":prompt_tokens,"teacher_forced_steps":32,"capacity":capacity,
        "successful_projection_submissions":submissions,"greedy_tokens_equal":tokens_equal,"reference_ids":reference_ids,"comparisons":comparisons,"checkpoints":checkpoints,
        "F64_metric_accumulation_not_FP64_model_reference":true,"DFlash_loaded":false,"allocation_cleanup":true,
        "long_context_qualified":false,"public_admission_changed":false}));
    assert!(pass, "Nerkyor QPN candidate violates unchanged numerical contract");
}

#[test]
fn source_split8_is_not_qpn_and_keeps_source_activation() {
    let p = CheckpointProjectionPolicy::NerkyorSourceSplit8F64Research;
    assert!(p.uses_nerkyor_source_split8_research() && p.uses_nerkyor_gate_f32_cta_research());
    assert!(!p.uses_qpn_projections() && !p.uses_qpn_ffn() && !p.uses_fp8_warp());
    assert!(!p.uses_small_gate_split16() && !p.uses_nerkyor_qpn_research());
    assert_ne!(p.label(), CheckpointProjectionPolicy::NerkyorQpnWorkspaceShapeF64Research.label());
}

#[test]
fn nerkyor_qpn_ffn_isolation_keeps_attention_and_gates_on_source_path() {
    let p = CheckpointProjectionPolicy::NerkyorQpnFfnShapeF64Research;
    assert!(p.uses_nerkyor_qpn_research() && p.uses_nerkyor_gate_f32_cta_research());
    assert!(p.uses_qpn_ffn() && p.qpn_workspace() && p.qpn_shape_split());
    assert!(!p.uses_qpn_projections() && !p.uses_nerkyor_source_split8_research());
    assert!(!p.uses_small_gate_split16() && !p.uses_fp8_warp());
    assert_ne!(p.label(), CheckpointProjectionPolicy::NerkyorQpnWorkspaceShapeF64Research.label());
    assert_eq!((0..64).filter(|&i| p.qpn_ffn_layer(i)).count(), 64);
}

#[test]
fn nerkyor_qpn_source_pair_does_not_stage_half_or_use_tensor_core_ffn() {
    let p = CheckpointProjectionPolicy::NerkyorQpnSourcePairResearch;
    assert!(p.uses_nerkyor_qpn_source_pair() && p.uses_nerkyor_gate_f32_cta_research());
    assert!(p.uses_qpn_projections());
    assert!(!p.uses_qpn_ffn() && !p.qpn_residual() && !p.qpn_workspace());
    assert!(!p.uses_nerkyor_qpn_research() && !p.uses_nerkyor_source_split8_research());
}
