//! Fixed B8 after real pure-input checkpoint restoration. No adaptive fallback,
//! product publication or HTTP clocks. Readbacks are outside component timings.
use super::*;
use crate::augur_session::checkpoint::{Bindings, Boundary, Descriptor, SCHEMA};
use crate::session_checkpoint::ParallelRestoreSchedule;
use std::io::Read;
use std::path::{Path, PathBuf};

fn file_sha(path: &Path) -> String {
    let mut file = std::fs::File::open(path).unwrap();
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).unwrap();
        if n == 0 { break; }
        hash.update(&buffer[..n]);
    }
    format!("{:x}", hash.finalize())
}
fn input_end(bindings: &Bindings, tokens: &[u32]) -> Descriptor {
    let d = Descriptor { schema: SCHEMA.into(), bindings: bindings.clone(),
        boundary: Boundary::InputEnd { token_ids: tokens.to_vec() } };
    d.validate(bindings).unwrap();
    d
}
fn reset(t: &mut CheckpointCudaTarget, b: &mut AugurDflashBridge) {
    b.release_batched_verifier().unwrap();
    t.reset_empty().unwrap();
}
fn prefill(t: &mut CheckpointCudaTarget, tokens: &[u32]) -> f64 {
    if tokens.is_empty() { return 0.; }
    let tick = Instant::now();
    consume_prompt(t, tokens);
    t.stream.synchronize().unwrap();
    tick.elapsed().as_secs_f64()
}
fn load(t: &mut CheckpointCudaTarget, d: &Descriptor, path: &Path, sha: &str) -> f64 {
    let tick = Instant::now();
    t.checkpoint_restore_parallel_scheduled(d, path, sha, 8,
        ParallelRestoreSchedule::Pipeline).unwrap();
    t.stream.synchronize().unwrap();
    tick.elapsed().as_secs_f64()
}
fn same_rounds(a: &Run, b: &Run) {
    equal(a, b);
    let ar = a.wire["rounds"].as_array().unwrap();
    let br = b.wire["rounds"].as_array().unwrap();
    assert_eq!(ar.len(), br.len());
    for (a, b) in ar.iter().zip(br) {
        for key in ["canonical_tokens", "proposal_path", "first_rejection", "stop",
                    "start_consumed", "end_consumed"] { assert_eq!(a[key], b[key], "{key}"); }
    }
}

#[test]
#[ignore = "exclusive full256Ki cached English/Chinese fixed-B8 long thinking"]
fn cached_bilingual_fixed_b8() {
    let raw = std::fs::read(std::env::var_os("AUGUR_ACCEPTANCE_INPUT").unwrap()).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&raw)),
        std::env::var("AUGUR_ACCEPTANCE_INPUT_SHA256").unwrap());
    let probes: Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(probes["schema"], "quasar-cached-bilingual-thinking-v1");
    let fixture: Value = serde_json::from_slice(&std::fs::read(
        std::env::var_os("AUGUR_CHECKPOINT_TARGET_REFERENCE").unwrap()).unwrap()).unwrap();
    for (probe, source) in [("tokenizer_sha256", "tokenizer.json"),
                            ("template_sha256", "chat_template.jinja")] {
        assert_eq!(probes[probe], fixture["metadata"][source]["sha256"]);
    }
    let cfg = std::fs::read(std::env::var_os("AUGUR_DRAFT_CONFIG").unwrap()).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&cfg)),
        std::env::var("AUGUR_DRAFT_CONFIG_SHA256").unwrap());
    let q8 = DFlashQ8Backbone::load_pinned(std::env::var_os("AUGUR_DRAFT_PATH").unwrap(),
        DFlashConfig::parse_f16_opt_in(&cfg).unwrap()).unwrap();
    assert_eq!(q8.manifest().file_sha256.as_ref().unwrap(),
        &std::env::var("AUGUR_DRAFT_SHA256").unwrap());
    let bindings = Bindings {
        producer_sha256: file_sha(&std::env::current_exe().unwrap()),
        manifest_sha256: probes["manifest_sha256"].as_str().unwrap().into(),
        profile: crate::augur_session::DFLASH_B8_PROFILE.into(),
        draft_config_sha256: std::env::var("AUGUR_DRAFT_CONFIG_SHA256").unwrap(),
        draft_weights_sha256: std::env::var("AUGUR_DRAFT_SHA256").unwrap(),
        decode_attention: probes["target_attention_mode"].as_str().unwrap().into(),sampling:crate::augur_sampling::GREEDY.into(),
    };
    bindings.validate().unwrap();
    let before = cuda_device_allocation_snapshot();
    let mut target = crate::checkpoint_target_cuda::transfer_tests::open(&fixture);
    assert_eq!(target.capacity, 262144);
    target.enable_augur_decode().unwrap();
    target.enable_augur_feature_capture().unwrap();
    assert_eq!(target.info()["attention_mode"], probes["target_attention_mode"]);
    let weights = DFlashQ8CudaWeights::upload(&q8, 0).unwrap();
    let mut bridge = AugurDflashBridge::new(&target, &weights).unwrap();
    // Allocate reusable prefill and draft scratch before any timed generation.
    prefill(&mut target, &[16; 16]);
    bridge.retain_prefill_benchmark_workspace(&mut target).unwrap();
    bridge.prepare_batched_verifier(&target).unwrap();
    let warm = bridge.propose_resident(&mut target).unwrap();
    bridge.discard_resident(warm).unwrap();
    reset(&mut target, &mut bridge);
    let eos = ids(&fixture["eos_token_ids"]);
    let mut results = Vec::new();
    for group in probes["groups"].as_array().unwrap() {
        let group_id = group["id"].as_str().unwrap();
        assert!(group_id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'));
        let prefix = ids(&group["prefix_token_ids"]);
        let prefix_d = input_end(&bindings, &prefix);
        let prefix_path: PathBuf = root().join(format!("{group_id}-prefix-cache"));
        prefill(&mut target, &prefix);
        let prefix_state = target.checkpoint_test_state_sha256(&prefix_d).unwrap();
        let tick = Instant::now();
        let saved = target.checkpoint_save(&prefix_d, &prefix_path).unwrap();
        let capture_s = tick.elapsed().as_secs_f64();
        save(&format!("{group_id}-prefix.json"), &json!({"descriptor":prefix_d,
            "path":prefix_path,"manifest_sha256":saved.manifest_sha256,
            "state_sha256":prefix_state,"capture_s":capture_s,"payload_bytes":saved.payload_bytes,
            "pure_input":true,"generated_answer_tokens_cached":0}));
        reset(&mut target, &mut bridge);
        let cases = group["cases"].as_array().unwrap();
        assert!(cases.iter().any(|c|c["language"]=="en") && cases.iter().any(|c|c["language"]=="zh"));
        for case in cases {
            let name = case["id"].as_str().unwrap();
            assert!(name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'));
            let tokens = ids(&case["token_ids"]);
            let budget = case["max_new_tokens"].as_u64().unwrap() as usize;
            assert!(budget >= 128 && budget <= 8192 && tokens.len() + budget <= target.capacity);
            assert!(tokens.starts_with(&prefix) && tokens.len() > prefix.len());
            assert_eq!(case["thinking"], true);
            let tail = &tokens[prefix.len()..];
            let end_d = input_end(&bindings, &tokens);
            let end_path = root().join(format!("{name}-exact-cache"));
            let mut end_manifest = String::new();
            let mut end_state = String::new();
            let mut reference: Option<Run> = None;
            let mut rows = Vec::new();
            // Same segmented arithmetic for both cold replays and cached lanes.
            // OS coldness is not asserted. Unsplit product cold is measured separately.
            for (index, lane) in ["cold_segmented", "cached_prefix", "cached_exact", "cold_segmented"].iter().enumerate() {
                reset(&mut target, &mut bridge);
                let (restore_s, prefix_s, suffix_s, matched) = match *lane {
                    "cold_segmented" => (0., prefill(&mut target, &prefix), prefill(&mut target, tail), 0),
                    "cached_prefix" => {
                        let elapsed = load(&mut target, &prefix_d, &prefix_path, &saved.manifest_sha256);
                        assert_eq!(target.checkpoint_test_state_sha256(&prefix_d).unwrap(), prefix_state);
                        (elapsed, 0., prefill(&mut target, tail), prefix.len())
                    },
                    "cached_exact" => (load(&mut target, &end_d, &end_path, &end_manifest), 0., 0., tokens.len()),
                    _ => unreachable!(),
                };
                let state = target.checkpoint_test_state_sha256(&end_d).unwrap();
                if index == 0 {
                    end_state = state;
                    let tick = Instant::now();
                    let full = target.checkpoint_save(&end_d, &end_path).unwrap();
                    end_manifest = full.manifest_sha256;
                    save(&format!("{name}-exact.json"), &json!({"descriptor":end_d,"path":end_path,
                        "manifest_sha256":end_manifest,"state_sha256":end_state,
                        "capture_s":tick.elapsed().as_secs_f64(),"payload_bytes":full.payload_bytes,
                        "pure_input":true,"generated_answer_tokens_cached":0}));
                } else { assert_eq!(state, end_state, "complete pre-generation state, including draft feature ring"); }
                let tick = Instant::now();
                bridge.prepare_batched_verifier(&target).unwrap();
                assert_eq!(bridge.batched.as_ref().unwrap().width, 8);
                let setup_s = tick.elapsed().as_secs_f64();
                let run = resident(&mut target, &mut bridge, budget, &eos, false);
                if let Some(expected) = &reference { same_rounds(&run, expected); }
                let row = json!({"lane":lane,"index":index,"input_tokens":tokens.len(),"matched_tokens":matched,
                    "new_prefill_tokens":tokens.len()-matched,"restore_s":restore_s,"prefix_prefill_s":prefix_s,
                    "suffix_prefill_s":suffix_s,"verifier_setup_s":setup_s,"state_exact":true,
                    "native_components_not_HTTP_TTFT":true,"full_state_readback_outside_timing":true,
                    "output_tokens":run.output,"stop":run.stop,"pending_token":run.pending,
                    "final_cursor":run.cursor,"generation":run.wire,"adaptive_fallback":false});
                let row_name = format!("{name}-{index}.json");
                let digest = save(&row_name, &row);
                rows.push(json!({"file":row_name,"sha256":digest}));
                println!("AUGUR_CACHE_THINKING_ROW={}",json!({"id":name,"lane":lane,"file":row_name,"sha256":digest}));
                std::io::stdout().flush().unwrap();
                if reference.is_none() { reference = Some(run); }
            }
            results.push(json!({"id":name,"language":case["language"],"rows":rows,
                "full_state_exact":true,"output_logits_pending_cursor_rounds_exact":true}));
            reset(&mut target, &mut bridge);
        }
    }
    let workspace = bridge.resident_workspace_info();
    drop(bridge); drop(weights); drop(target);
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes, before.live_bytes);
    assert_eq!(after.live_allocations, before.live_allocations);
    save("native-complete.json", &json!({"status":"PASS_FIXED_B8_CACHE_RESTORATION",
        "results":results,"workspace":workspace,"bindings":bindings,"fixed_B8":true,
        "HTTP":false,"general_quality_qualified":false,"checkpoints_retained":true,
        "timing_readbacks_excluded":true,"allocation_closure":true}));
}
