//! Identical timing boundary for model-local, no-DFlash target decode.
//! Fixed-step throughput intentionally does not apply an EOS stopping policy.
use super::*;
use augur_backend::{cuda_host_boundary_snapshot, CudaProfilerCapture};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::PathBuf};

fn hash(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn save(root: &Path, name: &str, value: &Value) {
    let mut f = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(root.join(name))
        .unwrap();
    f.write_all(&serde_json::to_vec_pretty(value).unwrap())
        .unwrap();
    f.sync_all().unwrap();
}
fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| {
            let n = v.as_u64().unwrap();
            assert!(n < 248320);
            n as u32
        })
        .collect()
}
fn prefill(t: &mut CheckpointCudaTarget, input: &[u32], quasar: bool) -> f64 {
    t.reset_empty().unwrap();
    let tick = Instant::now();
    if quasar {
        #[cfg(feature = "qwen35-augur-http")]
        {
            let plan = t
                .plan_augur_sequence(input, input.chunks(4352).map(|s| s.len()).collect())
                .unwrap();
            t.consume_sequence_augur(input, &plan, augur_prefill::Budget::default(), |_| false)
                .unwrap();
        }
        #[cfg(not(feature = "qwen35-augur-http"))]
        panic!("Quasar benchmark requires its explicit feature build");
    } else {
        for chunk in input.chunks(8) {
            t.consume_sequence(chunk).unwrap();
        }
    }
    t.stream.synchronize().unwrap();
    tick.elapsed().as_secs_f64()
}
fn decode(
    t: &mut CheckpointCudaTarget,
    steps: usize,
    quasar: bool,
    profile: bool,
    eos: &[u32],
) -> Value {
    assert!((1..=256).contains(&steps));
    assert!(t.consumed + steps < t.capacity);
    assert!(t.dflash_capture.is_none());
    let prompt = t.consumed;
    let first = t.greedy_next_token().unwrap();
    t.stream.synchronize().unwrap();
    let before = cuda_device_allocation_snapshot();
    let boundary = cuda_host_boundary_snapshot();
    let mut output = Vec::with_capacity(steps + 1);
    output.push(first);
    let mut step_s = Vec::with_capacity(steps);
    let mut capture = if profile {
        Some(CudaProfilerCapture::start().unwrap())
    } else {
        None
    };
    let tick = Instant::now();
    for _ in 0..steps {
        let one = Instant::now();
        if quasar {
            #[cfg(feature = "qwen35-augur-http")]
            t.consume_selected_augur().unwrap();
            #[cfg(not(feature = "qwen35-augur-http"))]
            panic!("Quasar benchmark requires its explicit feature build");
        } else {
            t.consume_selected().unwrap();
        }
        let next = t.greedy_next_token().unwrap();
        output.push(next);
        step_s.push(one.elapsed().as_secs_f64());
    }
    t.stream.synchronize().unwrap();
    let wall = tick.elapsed().as_secs_f64();
    if let Some(c) = capture.as_mut() {
        c.stop().unwrap();
    }
    let after = cuda_device_allocation_snapshot();
    let transfers = cuda_host_boundary_snapshot()
        .checked_delta_since(boundary)
        .unwrap();
    assert_eq!(after.successful_allocations, before.successful_allocations);
    assert_eq!(after.successful_frees, before.successful_frees);
    assert_eq!(transfers.synchronous_h2d_bytes, 0);
    assert_eq!(transfers.synchronous_d2h_bytes, 4 * steps as u64);
    assert_eq!(t.consumed, prompt + steps);
    assert!(t.dflash_capture.is_none());
    let logits = t.logits_snapshot().unwrap();
    let pending = t.greedy_next_token().unwrap();
    assert_eq!(Some(&pending), output.last());
    json!({"decode_steps":steps,"decode_wall_s":wall,"decode_steps_tps":steps as f64/wall,"step_wall_s":step_s,
        "prompt_tokens":prompt,"consumed":t.consumed,"pending":pending,"output":output,
        "output_sha256":hash(&output.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()),
        "logits_sha256":hash(&logits.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()),
        "first_prefill_token_not_counted":true,"EOS_stop_policy":"ignored_for_fixed_step_throughput_only",
        "observed_EOS_selections":output.iter().filter(|id|eos.contains(id)).count(),
        "gpu_allocations_in_timing":0,"gpu_frees_in_timing":0,"sync_h2d_bytes":0,"sync_d2h_bytes":transfers.synchronous_d2h_bytes,
        "profiler_instrumented":profile,"diagnostic_logit_copy_inside_timing":false,
        "DFlash_loaded":false,"DFlash_capture":false,"MTP_executed":false,"HTTP":false})
}

#[test]
#[ignore = "exclusive SM70; model-local no-DFlash fixed-step decode comparison"]
fn no_dflash_fixed_step_decode_wall_and_profile() {
    let out = PathBuf::from(std::env::var_os("AUGUR_NODFLASH_OUTPUT").unwrap());
    fs::create_dir(&out).unwrap();
    let guard: Value = serde_json::from_slice(
        &fs::read(std::env::var_os("AUGUR_NERKYOR_GPU_GUARD").unwrap()).unwrap(),
    )
    .unwrap();
    assert_eq!(guard["gpu_compute_apps_empty"], true);
    assert_eq!(guard["queues_paused"], true);
    assert_eq!(guard["benchmark_auto_resume"], false);
    assert_eq!(
        cuda_device_properties(0).unwrap().compute_capability,
        CudaComputeCapability::SM70
    );
    let raw = fs::read(std::env::var_os("AUGUR_NODFLASH_INPUT").unwrap()).unwrap();
    assert_eq!(
        hash(&raw),
        std::env::var("AUGUR_NODFLASH_INPUT_SHA256").unwrap()
    );
    let input: Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(input["schema"], "nerkyor-quasar-no-dflash-aligned-input-v1");
    let lane = input["lane"].as_str().unwrap();
    assert!(matches!(lane, "nerkyor" | "quasar"));
    let quasar = lane == "quasar";
    let raw = fs::read(std::env::var_os("AUGUR_NODFLASH_MANIFEST").unwrap()).unwrap();
    assert_eq!(
        hash(&raw),
        std::env::var("AUGUR_NODFLASH_MANIFEST_SHA256").unwrap()
    );
    let manifest: Value = serde_json::from_slice(&raw).unwrap();
    let identity = serde_json::from_value(manifest["checkpoint"].clone()).unwrap();
    let root = PathBuf::from(std::env::var_os("AUGUR_NODFLASH_MODEL_ROOT").unwrap());
    let baseline = cuda_device_allocation_snapshot();
    let start = Instant::now();
    let mut target = if quasar {
        #[cfg(feature = "qwen35-augur-http")]
        {
            let mut t = CheckpointCudaTarget::open_with_execution_policy(
                &root,
                Qwen35CheckpointProfile::QuasarNvfp4,
                identity,
                262144,
                CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64,
                CheckpointKvPolicy::Q8_0,
            )
            .unwrap();
            t.enable_exact_warp_tree_norm().unwrap();
            t.enable_fused_input_qpn_norm().unwrap();
            t.enable_batched_ffn8().unwrap();
            t.enable_augur_decode().unwrap();
            t
        }
        #[cfg(not(feature = "qwen35-augur-http"))]
        panic!("missing Quasar feature");
    } else {
        #[cfg(feature = "qwen35-nerkyor-dflash-research")]
        {
            CheckpointCudaTarget::open_with_projection_policy(
                &root,
                Qwen35CheckpointProfile::NerkyorW4a16,
                identity,
                4096,
                match input["nerkyor_projection_candidate"].as_str() {
                    Some("qpn") => CheckpointProjectionPolicy::NerkyorQpnWorkspaceShapeF64Research,
                    Some("source-split8") => CheckpointProjectionPolicy::NerkyorSourceSplit8F64Research,
                    Some("qpn-source-pair") => CheckpointProjectionPolicy::NerkyorQpnSourcePairResearch,
                    Some("qpn-source-reassoc") => CheckpointProjectionPolicy::NerkyorQpnSourceReassocResearch,
                    None => CheckpointProjectionPolicy::NerkyorBf16GateF32CtaResearch,
                    _ => panic!("unknown no-DFlash candidate"),
                },
            )
            .unwrap()
        }
        #[cfg(not(feature = "qwen35-nerkyor-dflash-research"))]
        panic!("missing Nerkyor gate feature");
    };
    assert!(target.dflash_capture.is_none());
    let load_s = start.elapsed().as_secs_f64();
    let cases = input["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    let steps = input["decode_steps"].as_u64().unwrap() as usize;
    assert!((1..=256).contains(&steps));
    let reps = input["repetitions"].as_u64().unwrap();
    assert!((1..=4).contains(&reps));
    let profile = input["profile"].as_bool().unwrap();
    let eos = ids(&input["eos"]);
    let warm = ids(&cases[0]["token_ids"]);
    prefill(&mut target, &warm[..16.min(warm.len())], quasar);
    let _ = decode(&mut target, 8, quasar, false, &eos);
    let mut results = Vec::new();
    for repeat in 0..reps {
        let mut order = cases.iter().collect::<Vec<_>>();
        if repeat % 2 == 1 {
            order.reverse();
        }
        for case in order {
            let id = case["id"].as_str().unwrap();
            let tokens = ids(&case["token_ids"]);
            assert!(tokens.len() + steps < 4096);
            let prefill_s = prefill(&mut target, &tokens, quasar);
            let mut result = decode(&mut target, steps, quasar, profile, &eos);
            result["prefill_s_excluded"] = prefill_s.into();
            result["id"] = id.into();
            result["repeat"] = repeat.into();
            save(&out, &format!("{id}-r{repeat}.json"), &result);
            println!(
                "NODFLASH_ALIGNED lane={lane} case={id} repeat={repeat} tps={}",
                result["decode_steps_tps"]
            );
            results.push(result);
        }
    }
    let info = target.info().clone();
    let resident = cuda_device_allocation_snapshot().live_bytes - baseline.live_bytes;
    drop(target);
    let freed = cuda_device_allocation_snapshot();
    assert_eq!(baseline.live_bytes, freed.live_bytes);
    assert_eq!(baseline.live_allocations, freed.live_allocations);
    save(
        &out,
        "result.json",
        &json!({"status":"PASS_MODEL_LOCAL_NO_DFLASH_FIXED_STEP_MEASUREMENT","lane":lane,"target":info,"load_s_excluded":load_s,
        "resident_device_bytes":resident,"results":results,"allocation_cleanup":true,"DFlash_loaded":false,"DFlash_capture":false,
        "fixed_decode_steps":steps,"EOS_ignored_for_fixed_step_diagnostic":true,"formal_quality_benchmark":false,"performance_parity_claimed":false}),
    );
}

#[cfg(feature = "qwen35-nerkyor-dflash-research")]
#[path = "nerkyor_qpn_alignment_tests.rs"]
mod nerkyor_qpn;
