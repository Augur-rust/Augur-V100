//! Same-owner real target -> new DFlash probe. No candidate publication.
#![cfg(feature = "qwen35-nerkyor-dflash-research")]
use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, CudaComputeCapability,
};
use augur_model_qwen35::{CheckpointExpectedIdentity, DFlashConfig, Qwen35CheckpointProfile};
use augur_model_qwen35_runtime::{
    checkpoint_target_cuda::{
        nerkyor_dflash_research::NerkyorDflashResearchInput, CheckpointCudaTarget,
    },
    DFlashBlockProducerInput, DFlashQ8Backbone, DFlashQ8CudaBlock, DFlashQ8CudaDiagnosticOutput,
    DFlashQ8CudaWeights,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
#[path = "common/nerkyor_tokenizer.rs"]
mod tokenizer;
fn read(path: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn write(path: impl AsRef<Path>, bytes: &[u8]) {
    fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
}
fn record(path: impl AsRef<Path>, value: Value) {
    write(path, &serde_json::to_vec_pretty(&value).unwrap());
}
fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn digest(values: &[f32]) -> Value {
    assert!(values.iter().all(|v| v.is_finite()));
    json!({"elements":values.len(),"f32_le_sha256":format!("{:x}",Sha256::digest(bytes(values))),
        "max_abs":values.iter().fold(0.0_f32,|a,v|a.max(v.abs()))})
}
fn summary(output: &DFlashQ8CudaDiagnosticOutput) -> Value {
    assert_eq!(output.layer_hidden.len(), 5);
    assert_eq!(output.hidden.len(), 8 * 5120);
    assert_eq!(output.selector_hidden.len(), 8 * 256);
    json!({"layers":output.layer_hidden.iter().map(|v| {assert_eq!(v.len(),8*5120);digest(v)}).collect::<Vec<_>>(),
        "hidden":digest(&output.hidden),"selector_hidden":digest(&output.selector_hidden)})
}
fn execute(block: &mut DFlashQ8CudaBlock) -> DFlashQ8CudaDiagnosticOutput {
    let before = cuda_device_allocation_snapshot();
    block.prepare_context().unwrap();
    block.submit().unwrap();
    block.synchronize().unwrap();
    let result = block.readback_diagnostics().unwrap();
    assert_eq!(
        cuda_device_allocation_snapshot().successful_allocations,
        before.successful_allocations
    );
    result
}
fn bind_run(
    input: &NerkyorDflashResearchInput,
    target: &mut CheckpointCudaTarget,
    weights: &DFlashQ8CudaWeights,
) -> DFlashQ8CudaDiagnosticOutput {
    let before = cuda_device_allocation_snapshot();
    let mut block = input.bind_block(target, weights).unwrap();
    let first = execute(&mut block);
    let host = DFlashBlockProducerInput {
        seed: input.seed(),
        target_features: input.features(),
        target_capture: Some(input.capture()),
        context_start_position: input.capture().context_start_position,
        noise_embedding: input.noise_embedding(),
        sliding_window: 2048,
    };
    let before_rebind = cuda_device_allocation_snapshot();
    block.rebind(&host).unwrap();
    let replay = execute(&mut block);
    assert_eq!(
        summary(&first),
        summary(&replay),
        "host vs device feature transfer must be bitwise equal"
    );
    assert_eq!(
        cuda_device_allocation_snapshot().successful_allocations,
        before_rebind.successful_allocations
    );
    drop(block);
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes, before.live_bytes);
    assert_eq!(after.live_allocations, before.live_allocations);
    first
}
#[test]
#[ignore = "exclusive SM70 + authenticated target/draft/tokenizer + fresh guard/output; ~12 min target load"]
fn real_nerkyor_target_capture_embedding_to_dflash_replay() {
    let guard = read(std::env::var_os("AUGUR_NERKYOR_GPU_GUARD").unwrap());
    assert_eq!(guard["gpu_compute_apps_empty"], true);
    assert_eq!(guard["queues_paused"], true);
    assert_eq!(guard["benchmark_auto_resume"], false);
    for name in [
        "same-build-three-seeds-20261018",
        "priority-gpqa-seed1-20261018",
    ] {
        let s = read(
            PathBuf::from("F:/augur/evidence/benchmarks")
                .join(name)
                .join("status.json"),
        );
        assert_eq!(s["stage"], "paused_by_user");
        assert_eq!(s["automatic_resume"], false);
    }
    assert_eq!(
        cuda_device_properties(0).unwrap().compute_capability,
        CudaComputeCapability::SM70
    );
    let dest = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_GPU_OUTPUT").unwrap());
    fs::create_dir(&dest).unwrap();
    let model = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_MODEL").unwrap());
    let root = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_DFLASH_Q8").unwrap());
    let reference = read(std::env::var_os("AUGUR_NERKYOR_DFLASH_REFERENCE").unwrap());
    let config_bytes = fs::read(root.join("config.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&config_bytes)),
        reference["config_sha256"].as_str().unwrap()
    );
    assert_eq!(
        Sha256::digest(fs::read(model.join("dflash/config.json")).unwrap()),
        Sha256::digest(&config_bytes)
    );
    let config = DFlashConfig::parse(&config_bytes).unwrap();
    let identity: CheckpointExpectedIdentity = serde_json::from_slice(
        &fs::read(std::env::var_os("AUGUR_NERKYOR_IDENTITY").unwrap()).unwrap(),
    )
    .unwrap();
    let started = Instant::now();
    let baseline = cuda_device_allocation_snapshot();
    let cpu =
        DFlashQ8Backbone::load_pinned(root.join("qwen35-27b-dflash2-q8_0.gguf"), config).unwrap();
    assert_eq!(
        cpu.manifest().file_sha256.as_deref(),
        reference["gguf_sha256"].as_str()
    );
    assert!(cpu.embedded_selector_gate().is_none());
    let weights = DFlashQ8CudaWeights::upload(&cpu, 0).unwrap();
    drop(cpu);
    record(
        dest.join("draft-ready.json"),
        json!({"elapsed_s":started.elapsed().as_secs_f64(),"gguf_sha256":weights.file_sha256()}),
    );
    let mut target =
        CheckpointCudaTarget::open(&model, Qwen35CheckpointProfile::NerkyorW4a16, identity, 64)
            .unwrap();
    let loaded_s = started.elapsed().as_secs_f64();
    assert!(
        target.enable_dflash_feature_capture().is_err(),
        "production guard must remain closed"
    );
    target
        .enable_nerkyor_dflash_research_capture(&weights)
        .unwrap();
    assert!(target
        .enable_nerkyor_dflash_research_capture(&weights)
        .is_err());
    assert!(
        target.capture_nerkyor_dflash_research_input().is_err(),
        "empty features must reject"
    );
    record(
        dest.join("target-ready.json"),
        json!({"elapsed_s":loaded_s,"target_info":target.info()}),
    );
    let tok = tokenizer::load(&model);
    tokenizer::validate_samples(&tok);
    let fixture = tokenizer::reference();
    let prompt = fixture["chat_cases"][0]["rendered"].as_str().unwrap();
    let encoded = tok.encode(prompt, false).unwrap();
    let ids = encoded.get_ids();
    assert_eq!(ids.len(), 22);
    let target_resident = cuda_device_allocation_snapshot();
    target.consume_sequence(ids).unwrap();
    let logits = target.logits_snapshot().unwrap();
    let input = target.capture_nerkyor_dflash_research_input().unwrap();
    assert_eq!(input.capture().target_layer_ids, [5, 19, 33, 47, 61]);
    assert_eq!(input.capture().context_len, 22);
    assert_eq!(input.seed().anchor_token, 17); // Existing own-tokenizer target replay selected "2".
    write(dest.join("target-features.f32le"), &bytes(input.features()));
    write(
        dest.join("target-noise-embedding.f32le"),
        &bytes(input.noise_embedding()),
    );
    record(
        dest.join("bridge-input.json"),
        json!({"capture":input.capture(),"block_tokens":input.seed().block_tokens,
        "absolute_position_start":input.seed().absolute_position_start,"noise_sha256":input.noise_sha256(),
        "features":digest(input.features()),"noise_embedding":digest(input.noise_embedding()),
        "prompt_ids":ids,"prompt_rendered":prompt,"target_logits":digest(&logits),
        "gguf_sha256":weights.file_sha256(),"target_info":target.info()}),
    );
    let mut bad = input.capture().clone();
    bad.request_generation += 1;
    assert!(bad
        .validate_for_block(
            weights.config(),
            input.seed(),
            input.features().len(),
            input.seed().target_feature_sha256
        )
        .is_err());
    bad = input.capture().clone();
    bad.feature_sha256[0] ^= 1;
    assert!(bad
        .validate_for_block(
            weights.config(),
            input.seed(),
            input.features().len(),
            input.seed().target_feature_sha256
        )
        .is_err());
    let output = bind_run(&input, &mut target, &weights);
    for (i, values) in output.layer_hidden.iter().enumerate() {
        write(dest.join(format!("draft-layer-{i}.f32le")), &bytes(values));
    }
    write(dest.join("draft-final.f32le"), &bytes(&output.hidden));
    write(
        dest.join("draft-selector.f32le"),
        &bytes(&output.selector_hidden),
    );
    target.consume_selected().unwrap();
    let pre_reject = cuda_device_allocation_snapshot();
    assert!(
        input.bind_block(&mut target, &weights).is_err(),
        "changed consumed cursor must reject stale input"
    );
    assert_eq!(
        cuda_device_allocation_snapshot().successful_allocations,
        pre_reject.successful_allocations
    );
    target.reset_empty().unwrap();
    assert!(
        input.bind_block(&mut target, &weights).is_err(),
        "reset epoch must reject stale input"
    );
    let mut independently_captured = Vec::new();
    for (i, &token) in ids.iter().enumerate() {
        let captured = target
            .consume(token, i + 1 == ids.len(), &[5, 19, 33, 47, 61])
            .unwrap();
        for (layer, values) in captured {
            assert!([5, 19, 33, 47, 61].contains(&layer));
            independently_captured.extend(values);
        }
    }
    assert_eq!(
        bytes(&independently_captured),
        bytes(input.features()),
        "ring row/layer ordering must equal explicit per-layer readbacks"
    );
    assert_eq!(digest(&target.logits_snapshot().unwrap()), digest(&logits));
    assert!(
        input.bind_block(&mut target, &weights).is_err(),
        "same cursor after reset is still a different epoch"
    );
    let replay_input = target.capture_nerkyor_dflash_research_input().unwrap();
    assert_eq!(bytes(replay_input.features()), bytes(input.features()));
    assert_eq!(replay_input.noise_sha256(), input.noise_sha256());
    let replay = bind_run(&replay_input, &mut target, &weights);
    assert_eq!(summary(&output), summary(&replay));
    drop(replay_input);
    drop(input);
    let pre_drop = cuda_device_allocation_snapshot();
    // Only two explicit block constructors allocate after resident target setup.
    // Decode/reset/rejection themselves were checked around their own calls.
    drop(target);
    drop(weights);
    let freed = cuda_device_allocation_snapshot();
    assert_eq!(freed.live_bytes, baseline.live_bytes);
    assert_eq!(freed.live_allocations, baseline.live_allocations);
    assert_eq!(
        freed.successful_allocations - baseline.successful_allocations,
        freed.successful_frees - baseline.successful_frees
    );
    record(
        dest.join("result.json"),
        json!({"status":"PASS_REAL_NERKYOR_TARGET_CAPTURE_EMBEDDING_DFLASH_BRIDGE_REPLAY",
        "target_features_real":true,"target_embeddings_real":true,"device_feature_handoff":true,"layers":[5,19,33,47,61],
        "context_rows":22,"block_size":8,"completed_draft_blocks":4,"output":summary(&output),
        "capture_vs_explicit_layer_readback_bitwise_equal":true,"host_device_feature_rebind_bitwise_equal":true,
        "reset_replay_bitwise_equal":true,"stale_cursor_and_epoch_rejected":true,"production_capture_guard_closed":true,
        "altered_generation_and_hash_rejected":true,"foreign_owner_predicate_unit_tested_only":true,
        "allocation_cleanup_passed":true,"submit_rebind_device_allocations":0,
        "target_resident_bytes_observed":target_resident.live_bytes-baseline.live_bytes,
        "pre_drop_bytes_observed":pre_drop.live_bytes-baseline.live_bytes,
        "load_s":loaded_s,"elapsed_s":started.elapsed().as_secs_f64(),
        "candidates_published":false,"acceptance_measured":false,"numerical_reference_qualified":false,
        "old_selector_calibration_reused":false,"mtp_uploaded":false,"mtp_executed":false,
        "benchmark_resumed":false,"speed_qualified":false,"public_admission_changed":false}),
    );
}
