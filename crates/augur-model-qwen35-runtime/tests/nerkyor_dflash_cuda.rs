//! Opt-in new-artifact DFlash GPU component probe, not speculative decoding.
//! Synthetic features/noise are NOT a real target capture or target embedding.
#![cfg(feature = "cuda")]
use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, CudaComputeCapability,
};
use augur_model_qwen35::DFlashConfig;
use augur_model_qwen35_runtime::{
    DFlashBlockProducerInput, DFlashDraftBlockInput, DFlashQ8Backbone, DFlashQ8CudaBlock,
    DFlashQ8CudaWeights,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::PathBuf, time::Instant};

fn read_json(path: impl AsRef<std::path::Path>) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn hash(values: &[f32]) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    format!("{:x}", Sha256::digest(bytes))
}
fn synthetic(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed + 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state & 0xffff) as f32 - 32768.0) / (32768.0 * 64.0)
        })
        .collect()
}
fn digest(values: &[f32], len: usize) -> Value {
    assert_eq!(values.len(), len);
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| *v != 0.0));
    json!({"elements":len,"f32_le_sha256":hash(values),
        "max_abs":values.iter().fold(0.0_f32, |a,v| a.max(v.abs()))})
}
fn output(block: &DFlashQ8CudaBlock) -> Value {
    let values = block.readback_diagnostics().unwrap();
    assert_eq!(values.layer_hidden.len(), 5);
    let layers: Vec<Value> = values
        .layer_hidden
        .iter()
        .map(|v| digest(v, 8 * 5120))
        .collect();
    json!({"hidden":digest(&values.hidden,8*5120),
        "selector_hidden":digest(&values.selector_hidden,8*256),"layers":layers})
}

#[test]
#[ignore = "exclusive SM70 + exact new DFlash Q8 identity; explicit fresh GPU guard/output"]
fn real_nerkyor_dflash_q8_synthetic_block_replay_and_cleanup() {
    let guard = read_json(std::env::var_os("AUGUR_NERKYOR_GPU_GUARD").expect("GPU guard"));
    assert_eq!(guard["gpu_compute_apps_empty"], true);
    assert_eq!(guard["queues_paused"], true);
    assert_eq!(guard["benchmark_auto_resume"], false);
    for name in [
        "same-build-three-seeds-20261018",
        "priority-gpqa-seed1-20261018",
    ] {
        let status = read_json(
            PathBuf::from("F:/augur/evidence/benchmarks")
                .join(name)
                .join("status.json"),
        );
        assert_eq!(status["stage"], "paused_by_user");
        assert_eq!(status["automatic_resume"], false);
    }
    assert_eq!(
        cuda_device_properties(0).unwrap().compute_capability,
        CudaComputeCapability::SM70
    );
    let root = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_DFLASH_Q8").expect("new draft"));
    let reference = read_json(
        std::env::var_os("AUGUR_NERKYOR_DFLASH_REFERENCE").expect("independent identity"),
    );
    let destination =
        PathBuf::from(std::env::var_os("AUGUR_NERKYOR_GPU_OUTPUT").expect("fresh output"));
    fs::create_dir(&destination).unwrap();
    let started = Instant::now();
    let baseline = cuda_device_allocation_snapshot();
    let config_bytes = fs::read(root.join("config.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&config_bytes)),
        reference["config_sha256"].as_str().unwrap()
    );
    let config = DFlashConfig::parse(&config_bytes).unwrap();
    let config_json: Value = serde_json::from_slice(&config_bytes).unwrap();
    let mask_token_id = u32::try_from(
        config_json["dflash_config"]["mask_token_id"]
            .as_u64()
            .unwrap(),
    )
    .unwrap();
    let cpu =
        DFlashQ8Backbone::load_pinned(root.join("qwen35-27b-dflash2-q8_0.gguf"), config.clone())
            .unwrap();
    assert_eq!(
        cpu.manifest().file_sha256.as_deref(),
        reference["gguf_sha256"].as_str()
    );
    assert!(cpu.embedded_selector_gate().is_none());
    let resident = DFlashQ8CudaWeights::upload(&cpu, 0).unwrap();
    assert_eq!(
        resident.file_sha256(),
        reference["gguf_sha256"].as_str().unwrap()
    );
    assert!(resident.selector_gate_f32_sha256().is_none());
    let weight_bytes = resident.resident_bytes();
    drop(cpu);
    let load_s = started.elapsed().as_secs_f64();
    let mut results = Vec::new();
    for seed in 0..3 {
        let features = synthetic(3 * 5 * 5120, seed);
        let noise = synthetic(8 * 5120, seed + 31);
        let feature_bytes: Vec<u8> = features.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut tokens = vec![mask_token_id; 8];
        tokens[0] = 42;
        let input_seed = DFlashDraftBlockInput {
            request_generation: 1,
            absolute_position_start: 3,
            anchor_token: 42,
            block_tokens: tokens,
            mask_token_id: mask_token_id,
            target_feature_sha256: Sha256::digest(feature_bytes).into(),
        };
        let input = DFlashBlockProducerInput {
            seed: &input_seed,
            target_features: &features,
            target_capture: None,
            context_start_position: 0,
            noise_embedding: &noise,
            sliding_window: 2048,
        };
        let before = cuda_device_allocation_snapshot();
        let mut block = DFlashQ8CudaBlock::from_resident_weights(&resident, &input).unwrap();
        let workspace = block.workspace_bytes();
        let ready = cuda_device_allocation_snapshot();
        block.prepare_context().unwrap();
        block.submit().unwrap();
        block.synchronize().unwrap();
        let first = output(&block);
        block.rebind(&input).unwrap();
        block.prepare_context().unwrap();
        block.submit().unwrap();
        block.synchronize().unwrap();
        let replay = output(&block);
        assert_eq!(
            first, replay,
            "same synthetic seed must replay bitwise at every layer/final/selector"
        );
        let after = cuda_device_allocation_snapshot();
        assert_eq!(after.successful_allocations, ready.successful_allocations);
        drop(block);
        let freed = cuda_device_allocation_snapshot();
        assert_eq!(freed.live_bytes, before.live_bytes);
        assert_eq!(freed.live_allocations, before.live_allocations);
        results.push(
            json!({"synthetic_activation_seed":seed,"workspace_bytes":workspace,
            "bitwise_rebind_replay":true,"output":first}),
        );
    }
    drop(resident);
    let freed = cuda_device_allocation_snapshot();
    assert_eq!(freed.live_bytes, baseline.live_bytes);
    assert_eq!(freed.live_allocations, baseline.live_allocations);
    assert_eq!(
        freed.successful_allocations - baseline.successful_allocations,
        freed.successful_frees - baseline.successful_frees
    );
    let result = json!({"status":"PASS_NEW_DFLASH_Q8_GPU_SYNTHETIC_BLOCK_REPLAY_AND_CLEANUP",
        "gguf_sha256":reference["gguf_sha256"],"config_sha256":reference["config_sha256"],
        "layers":5,"block_size":8,"context_rows":3,"resident_weight_bytes":weight_bytes,
        "load_s":load_s,"elapsed_s":started.elapsed().as_secs_f64(),"cases":results,
        "completed_blocks":6,"submit_rebind_device_allocations":0,"allocation_cleanup_passed":true,
        "inputs_synthetic":true,"same_owner_target_capture_tested":false,
        "target_embeddings_used":false,"candidates_published":false,"acceptance_measured":false,
        "source_fp8_vs_q8_equivalence_qualified":false,"numerical_reference_equivalence_qualified":false,
        "old_selector_calibration_reused":false,"mtp_uploaded":false,"mtp_executed":false,
        "benchmark_started_or_resumed":false,"speed_qualified":false,"public_admission_changed":false});
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination.join("result.json"))
        .unwrap();
    file.write_all(&serde_json::to_vec_pretty(&result).unwrap())
        .unwrap();
}
