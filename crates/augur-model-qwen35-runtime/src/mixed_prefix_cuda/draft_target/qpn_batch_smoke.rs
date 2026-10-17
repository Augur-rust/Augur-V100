//! Exclusive-GPU actual DFlash/target A/B, not a component TPS extrapolation.
use super::*;

#[test]
#[ignore = "exclusive SM70 and hash-bound AUGUR_QPN_TARGET_INPUT"]
fn qpn_batched_full_target_smoke() {
    let input = std::fs::read(std::env::var_os("AUGUR_QPN_TARGET_INPUT").unwrap()).unwrap();
    let m: Value = serde_json::from_slice(&input).unwrap();
    for key in ["config", "tokenizer", "generation", "draft_config", "draft"] {
        let bytes = std::fs::read(m[key].as_str().unwrap()).unwrap();
        assert_eq!(sha256_hex(&bytes), m[format!("{key}_sha256")].as_str().unwrap(), "{key}");
    }
    let mut config = Qwen35AutoregressiveCudaConfig {
        gguf_path: m["model"].as_str().unwrap().into(),
        config_path: m["config"].as_str().unwrap().into(),
        tokenizer_path: m["tokenizer"].as_str().unwrap().into(),
        capacity: 512, generation_tokens: 32, warmup: 0,
        source_commit: crate::build_identity::source_commit().into(), ..Default::default()
    };
    let qpn = target_weights::qpn_ffn_switch(std::env::var_os("AUGUR_QWEN35_QPN_FFN").as_deref()).unwrap();
    let before = cuda_device_allocation_snapshot();
    let mut cases = Vec::new();
    let mut shadow_blocks = 0;
    let mut rejection_replays = 0;
    let performance_repeats = m.get("performance_repeats").and_then(Value::as_u64).unwrap_or(0);
    assert!(performance_repeats <= 9);
    {
        let owner = Qwen35ResidentDraftCuda::open_with_target(&config,
            Path::new(m["draft"].as_str().unwrap()), Path::new(m["draft_config"].as_str().unwrap())).unwrap();
        assert_eq!(owner.target.as_ref().unwrap().manifest.file_sha256.as_deref(), m["model_sha256"].as_str());
        assert_eq!(owner.target.as_ref().unwrap().qpn_ffn_enabled(), qpn);
        let resident = cuda_device_allocation_snapshot();
        for case in m["cases"].as_array().unwrap() {
            config.prompt_text = case["rendered_prompt"].as_str().unwrap().into();
            let mut reference = Value::Null;
            let mut runs = Vec::new();
            let mut modes = ["reference", "shadow", "batched", "batched_replay"]
                .map(str::to_owned).to_vec();
            modes.extend((0..performance_repeats).map(|i| format!("batched_performance_{i}")));
            for mode in modes {
                let mode = mode.as_str();
                let state_audit = !mode.starts_with("batched_performance_");
                let mut r = super::tests::batch_request_options();
                r.generation_config_path = m["generation"].as_str().unwrap().into();
                r.max_new_tokens = case["max_new_tokens"].as_u64().unwrap() as usize;
                // Always run audited reference/shadow/replay first. Separate
                // diagnostic timing runs omit only the optional state digest;
                // token equality and allocation/owner closure still apply.
                r.audit_committed_state = state_audit;
                r.target_only_reference = mode == "reference";
                r.batched_target_verify = mode != "shadow";
                r.audit_batched_target_verify = mode == "shadow";
                r.audit_batched_numerics = mode == "shadow";
                r.native_quantized_batch_ffn = mode == "shadow";
                r.canonical_batched_gdn_scalars = mode == "shadow";
                r.native_quantized_batch_attention = mode == "shadow";
                r.canonical_batched_norm = mode == "shadow";
                r.canonical_batched_full_core = mode == "shadow";
                r.reuse_batched_plan = mode.starts_with("batched");
                r.share_target_rope = true;
                r.request_position_budget = true;
                r.target_split_rope = augur_backend::cuda_q8_0::QWEN35_TARGET_SPLIT_ROPE;
                let started = Instant::now();
                let report = run_rolling_with_resident_draft(&config,
                    &owner.q8_path, &owner.draft_config_path, 8, 32, None, true, false, false,
                    Some(&r), None, Some(&owner.weights), owner.target.as_ref(), None,
                    None, None, None, false, false, false, None, None).unwrap();
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(report["cleanup"]["pass"], true);
                if mode == "reference" {
                    assert_eq!(report["schema"], "qwen35-target-only-matched-request-v1");
                    assert_eq!(report["state_audit_scheme"], "128_plane_device_fingerprint_v1_noncryptographic");
                } else {
                    assert_eq!(report["committed_state_audit_enabled"], state_audit);
                }
                let after = cuda_device_allocation_snapshot();
                assert_eq!(after.live_bytes, resident.live_bytes, "request cleanup");
                assert_eq!(after.live_allocations, resident.live_allocations);
                let ids = report["request_output"]["token_ids"].clone();
                assert!(!ids.as_array().unwrap().is_empty());
                if reference.is_null() { reference = ids.clone(); }
                else { assert_eq!(ids, reference, "same target committed sequence: {} {mode}", case["name"]); }
                if let Some(transactions) = report["transactions"].as_array() {
                    for tx in transactions {
                        assert_eq!(tx["pass"], true);
                        if mode == "shadow" {
                            assert_eq!(tx["batched_verification_shadow"]["strict_replacement_gate"], true,
                                "B8 versus B1 exact hidden/state/token gate");
                            shadow_blocks += 1;
                        }
                        if mode.starts_with("batched") && tx["target_forward_counts"]["commit_replay"].as_u64().unwrap() > 0 {
                            rejection_replays += 1;
                        }
                    }
                }
                eprintln!("QPN_BATCH_PROGRESS={} {mode} wall_ms={elapsed_ms:.3}", case["name"]);
                // Preserve the full transaction and timing inventory in evidence.
                runs.push(json!({"mode":mode,"diagnostic_wall_ms":elapsed_ms,
                    "state_digest_audit":state_audit,"report":report}));
            }
            cases.push(json!({"name":case["name"],"runs":runs}));
        }
    }
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes, before.live_bytes);
    assert_eq!(after.live_allocations, before.live_allocations);
    assert_eq!(after.successful_allocations-before.successful_allocations,
        after.successful_frees-before.successful_frees);
    assert!(shadow_blocks > 0 && rejection_replays > 0, "non-vacuous verification and rejection replay");
    eprintln!("QPN_TARGET_RESULT={}", json!({"input_sha256":sha256_hex(&input),"profile_qpn":qpn,
        "cases":cases,"shadow_exact_blocks":shadow_blocks,"rejection_replays":rejection_replays,
        "owner_cleanup_pass":true,"model_load_count":1,"capacity":512,"draft_history":8,
        "dflash":true,"vision_forward":false,"performance_repeats":performance_repeats,
        "peak_engine_owned_cuda_bytes":after.peak_live_bytes,
        "product_ready":false,"semantic_qualified":false,"visible_tps":false}));
}
