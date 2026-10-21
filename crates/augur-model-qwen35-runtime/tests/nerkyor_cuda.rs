//! Explicit local SM70 research probes. Never part of default evaluation queues.
//! Synthetic activations/raw token IDs do NOT qualify tokenizer, quality, or speed.
#![cfg(feature = "cuda")]

use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, CudaComputeCapability,
    OwnedCudaBuffer, OwnedCudaStream,
};
use augur_model_qwen35::{
    CheckpointArtifact, CheckpointExpectedIdentity, CheckpointLinearStorage,
    Qwen35CheckpointProfile,
};
use augur_model_qwen35_runtime::checkpoint_projection_cuda::CheckpointScalarCudaWeight;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::PathBuf, rc::Rc, time::Instant};

#[path = "common/nerkyor_tokenizer.rs"]
mod tokenizer;

const PROFILE: Qwen35CheckpointProfile = Qwen35CheckpointProfile::NerkyorW4a16;
const QUEUES: [&str; 2] = [
    "F:/augur/evidence/benchmarks/same-build-three-seeds-20261018/status.json",
    "F:/augur/evidence/benchmarks/priority-gpqa-seed1-20261018/status.json",
];

fn read_json(path: impl AsRef<std::path::Path>) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn guard_and_output() -> PathBuf {
    let guard = read_json(std::env::var_os("AUGUR_NERKYOR_GPU_GUARD").expect("explicit GPU guard"));
    assert_eq!(guard["gpu_compute_apps_empty"], true);
    assert_eq!(guard["queues_paused"], true);
    assert_eq!(guard["benchmark_auto_resume"], false);
    for path in QUEUES {
        let status = read_json(path);
        assert_eq!(status["stage"], "paused_by_user");
        assert_eq!(status["automatic_resume"], false);
    }
    assert_eq!(
        cuda_device_properties(0).unwrap().compute_capability,
        CudaComputeCapability::SM70
    );
    let output = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_GPU_OUTPUT").expect("fresh output"));
    fs::create_dir(&output).expect("output must not already exist");
    output
}
fn identity() -> CheckpointExpectedIdentity {
    serde_json::from_slice(
        &fs::read(std::env::var_os("AUGUR_NERKYOR_IDENTITY").expect("explicit new-model identity"))
            .unwrap(),
    )
    .unwrap()
}
fn model() -> PathBuf {
    PathBuf::from(std::env::var_os("AUGUR_NERKYOR_MODEL").expect("explicit new model"))
}
fn write_new(path: PathBuf, value: &Value) {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    file.write_all(&serde_json::to_vec_pretty(value).unwrap())
        .unwrap();
    file.write_all(b"\n").unwrap();
}
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn hash_f32(values: &[f32]) -> String {
    format!("{:x}", Sha256::digest(f32_bytes(values)))
}
fn synthetic_input(cols: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_add(1);
    (0..cols)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Exactly representable F32 values in [-1, 1), not benchmark sampling seeds.
            ((state & 0xffff) as f32 - 32768.0) / 32768.0
        })
        .collect()
}

#[test]
#[ignore = "exclusive SM70 + authenticated local new model + fresh output and guard"]
fn real_nerkyor_projection_reference_and_cleanup() {
    let output_root = guard_and_output();
    let started = Instant::now();
    let baseline = cuda_device_allocation_snapshot();
    let expected = identity();
    let references =
        read_json(std::env::var_os("AUGUR_NERKYOR_ROWS").expect("independent Python row hashes"));
    assert_eq!(references["config_sha256"], expected.config_sha256);
    assert_eq!(references["index_sha256"], expected.index_sha256);
    let mut reader =
        CheckpointArtifact::open_with_shard_workers(model(), PROFILE, expected, 2).unwrap();
    let open_s = started.elapsed().as_secs_f64();
    let names = [
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.mlp.up_proj",
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.63.mlp.down_proj",
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.0.linear_attn.in_proj_b",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.62.linear_attn.in_proj_a",
        "model.language_model.layers.62.linear_attn.in_proj_b",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.k_proj",
        "model.language_model.layers.3.self_attn.v_proj",
        "model.language_model.layers.3.self_attn.o_proj",
        "lm_head",
    ];
    let mut results = Vec::new();
    let mut independent_rows = 0;
    let mut finite_outputs = 0;
    for name in names {
        let source = reader.load_linear(name).unwrap();
        let spec = source.spec().clone();
        let reference_rows: Vec<usize> = if spec.rows == 48 {
            (0..48).collect()
        } else {
            vec![0, spec.rows / 2, spec.rows - 1]
        };
        let mut reference_weights = Vec::new();
        let independent = references["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == name)
            .expect("Python matrix oracle");
        for &row in &reference_rows {
            let decoded = reader
                .load_linear_rows(name, row, 1)
                .unwrap()
                .decode_f64()
                .unwrap();
            if let Some(record) = independent["rows"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["row"].as_u64() == Some(row as u64))
            {
                let bytes: Vec<u8> = decoded.iter().flat_map(|v| v.to_le_bytes()).collect();
                assert_eq!(
                    format!("{:x}", Sha256::digest(bytes)),
                    record["f64_le_sha256"].as_str().unwrap()
                );
                independent_rows += 1;
            }
            reference_weights.push(decoded);
        }
        assert!(matches!(
            spec.storage,
            CheckpointLinearStorage::Nvfp4 | CheckpointLinearStorage::Bf16
        ));
        let case_baseline = cuda_device_allocation_snapshot();
        {
            let stream = Rc::new(OwnedCudaStream::create(0).unwrap());
            let weight = CheckpointScalarCudaWeight::upload(Rc::clone(&stream), source).unwrap();
            let policy = weight.policy();
            let input = Rc::new(OwnedCudaBuffer::allocate(0, spec.cols * 4).unwrap());
            let output = Rc::new(OwnedCudaBuffer::allocate(0, spec.rows * 4).unwrap());
            assert!(weight
                .bind(Rc::clone(&input), 4, Rc::clone(&output), 0)
                .is_err());
            let projection = weight
                .bind(Rc::clone(&input), 0, Rc::clone(&output), 0)
                .unwrap();
            let weak_weight = Rc::downgrade(&weight);
            let weak_input = Rc::downgrade(&input);
            let weak_output = Rc::downgrade(&output);
            let weak_stream = Rc::downgrade(&stream);
            drop(weight);
            assert!(weak_weight.upgrade().is_some());
            let resident = cuda_device_allocation_snapshot();
            for phase in 0..4 {
                let x = if phase == 3 {
                    vec![0.0; spec.cols]
                } else {
                    synthetic_input(spec.cols, phase)
                };
                input.copy_from_host_sync(&stream, &f32_bytes(&x)).unwrap();
                projection.submit_for_plan().unwrap();
                let mut bytes = vec![0; spec.rows * 4];
                output.copy_to_host_sync(&stream, &mut bytes).unwrap();
                let values: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                assert!(
                    values.iter().all(|v| v.is_finite()),
                    "{name} nonfinite output"
                );
                finite_outputs += values.len();
                let (mut error2, mut norm2, mut max_abs, mut max_bound_ratio) =
                    (0.0, 0.0, 0.0_f64, 0.0_f64);
                for (&row, weights) in reference_rows.iter().zip(&reference_weights) {
                    let expected: f64 =
                        weights.iter().zip(&x).map(|(w, x)| w * f64::from(*x)).sum();
                    let sum_abs: f64 = weights
                        .iter()
                        .zip(&x)
                        .map(|(w, x)| (w * f64::from(*x)).abs())
                        .sum();
                    let error = (f64::from(values[row]) - expected).abs();
                    // Fixed before execution. Covers product/pair/output/global-scale F32 rounding;
                    // FP64 mathematical dot is the monitor, not a claim of exact F32 execution.
                    let bound = 2e-6 + 16.0 * f64::from(f32::EPSILON) * sum_abs;
                    assert!(
                        error <= bound,
                        "{name} phase {phase} row {row}: {error} > {bound}"
                    );
                    error2 += error * error;
                    norm2 += expected * expected;
                    max_abs = max_abs.max(error);
                    max_bound_ratio = max_bound_ratio.max(error / bound);
                }
                let relative_l2 = (error2 / norm2.max(1e-300)).sqrt();
                assert!(
                    relative_l2 <= 1e-5,
                    "{name} phase {phase}: relative L2 {relative_l2}"
                );
                if phase == 3 {
                    assert!(values.iter().all(|&v| v == 0.0));
                }
                let after = cuda_device_allocation_snapshot();
                assert_eq!(
                    after.successful_allocations, resident.successful_allocations,
                    "submit allocated"
                );
                results.push(json!({"name":name,"rows":spec.rows,"cols":spec.cols,
                    "storage":format!("{:?}",spec.storage),"synthetic_seed":if phase==3 {None} else {Some(phase)},
                    "zero_input":phase==3,"input_sha256":hash_f32(&x),"output_sha256":hash_f32(&values),
                    "checked_rows":reference_rows.len(),"relative_l2":relative_l2,"max_abs":max_abs,
                    "max_roundoff_bound_ratio":max_bound_ratio,"policy":policy}));
            }
            stream.synchronize().unwrap();
            drop(input);
            drop(output);
            drop(stream);
            assert!(
                weak_input.upgrade().is_some()
                    && weak_output.upgrade().is_some()
                    && weak_stream.upgrade().is_some()
            );
            drop(projection);
            assert!(
                weak_weight.upgrade().is_none()
                    && weak_input.upgrade().is_none()
                    && weak_output.upgrade().is_none()
                    && weak_stream.upgrade().is_none()
            );
        }
        let released = cuda_device_allocation_snapshot();
        assert_eq!(released.live_bytes, case_baseline.live_bytes);
        assert_eq!(released.live_allocations, case_baseline.live_allocations);
        eprintln!("NERKYOR_PROJECTION_PASS={name}");
    }
    drop(reader);
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes, baseline.live_bytes);
    assert_eq!(after.live_allocations, baseline.live_allocations);
    assert_eq!(
        after.successful_allocations - baseline.successful_allocations,
        after.successful_frees - baseline.successful_frees
    );
    write_new(
        output_root.join("result.json"),
        &json!({
            "status":"PASS_NERKYOR_REAL_GPU_PROJECTION_REFERENCE_AND_CLEANUP","profile":PROFILE,
            "matrices":names.len(),"cases":results,"independent_python_row_hashes_verified":independent_rows,
            "all_output_values_checked_finite":finite_outputs,"full_shards_authenticated":7,
            "open_hash_s":open_s,"elapsed_s":started.elapsed().as_secs_f64(),
            "relative_l2_limit":1e-5,"per_row_abs_limit":"2e-6 + 16*F32_EPSILON*sum_abs_products",
            "all_weight_rows_compared":false,"synthetic_activations_only":true,
            "allocation_cleanup_passed":true,"mtp_uploaded":false,"mtp_executed":false,
            "dflash_executed":false,"tokenizer_admitted":false,"semantic_qualified":false,
            "speed_qualified":false,"public_admission_changed":false
        }),
    );
}

#[cfg(all(
    feature = "qwen35-fused-head-norm-rope",
    feature = "qwen35-target-split-rope"
))]
mod target_smoke {
    use super::tokenizer;
    use super::*;
    use augur_model_qwen35_runtime::checkpoint_target_cuda::CheckpointCudaTarget;

    fn trajectory(target: &mut CheckpointCudaTarget) -> Value {
        let mut tokens = vec![1_u32, 42, 1024];
        let mut logits_hashes = Vec::new();
        let mut hidden_hashes = Vec::new();
        for &token in &tokens {
            let captured = target
                .consume(token, true, &[0, 3, 15, 31, 47, 63])
                .unwrap();
            let mut step_hidden = Vec::new();
            for (layer, values) in captured {
                assert_eq!(values.len(), 5120);
                assert!(values.iter().all(|v| v.is_finite()));
                step_hidden.push(json!({"layer":layer,"sha256":hash_f32(&values)}));
            }
            hidden_hashes.push(step_hidden);
            let logits = target.logits_snapshot().unwrap();
            assert_eq!(logits.len(), 248320);
            assert!(logits.iter().all(|v| v.is_finite()));
            logits_hashes.push(hash_f32(&logits));
        }
        for _ in 0..2 {
            let next = target.select_next_token().unwrap();
            assert_eq!(target.select_next_token().unwrap(), next);
            assert!(next < 248320);
            tokens.push(next);
            target.consume_selected().unwrap();
            let logits = target.logits_snapshot().unwrap();
            assert_eq!(logits.len(), 248320);
            assert!(logits.iter().all(|v| v.is_finite()));
            logits_hashes.push(hash_f32(&logits));
        }
        assert_eq!(target.consumed_tokens(), 5);
        let final_hash = logits_hashes.last().unwrap().clone();
        assert!(target.consume(248320, true, &[]).is_err());
        assert!(target.consume(1, true, &[3, 0]).is_err());
        assert_eq!(target.consumed_tokens(), 5);
        assert_eq!(hash_f32(&target.logits_snapshot().unwrap()), final_hash);
        json!({"tokens":tokens,"logits_sha256":logits_hashes,"prompt_hidden_sha256":hidden_hashes})
    }

    #[test]
    #[ignore = "exclusive SM70, authenticated complete new model, raw IDs only, fresh output and guard"]
    fn real_nerkyor_target_reset_replay_smoke() {
        let output_root = guard_and_output();
        let started = Instant::now();
        let baseline = cuda_device_allocation_snapshot();
        let mut target = CheckpointCudaTarget::open(model(), PROFILE, identity(), 8).unwrap();
        let loaded_s = started.elapsed().as_secs_f64();
        let info = target.info().clone();
        assert_eq!(info["text_layers"], 64);
        assert_eq!(info["text_linears"], 497);
        assert_eq!(info["kv_dtype"], "F32");
        assert_eq!(info["activation_dtype"], "F32");
        assert_eq!(info["draft_enabled"], false);
        assert_eq!(info["policy"], "checkpoint_original_f32_scalar_v1");
        assert_eq!(target.consumed_tokens(), 0);
        assert!(target.logits_snapshot().is_err());
        assert!(target.select_next_token().is_err());
        let resident = cuda_device_allocation_snapshot();
        let first = trajectory(&mut target);
        target.reset_empty().unwrap();
        assert_eq!(target.consumed_tokens(), 0);
        assert!(target.logits_snapshot().is_err());
        assert!(target.select_next_token().is_err());
        assert!(target.consume_selected().is_err());
        let replay = trajectory(&mut target);
        assert_eq!(
            replay, first,
            "reset replay must be bitwise identical including selected IDs"
        );
        target.reset_empty().unwrap();
        let tokens: Vec<u32> = first["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_u64().unwrap() as u32)
            .collect();
        target.consume_sequence(&tokens).unwrap();
        assert_eq!(target.consumed_tokens(), 5);
        assert_eq!(
            hash_f32(&target.logits_snapshot().unwrap()),
            first["logits_sha256"][4].as_str().unwrap(),
            "sequence path must match scalar + device-selected-token handoff"
        );
        target.consume_sequence(&[1, 2, 3]).unwrap();
        assert_eq!(target.consumed_tokens(), 8);
        let full_hash = hash_f32(&target.logits_snapshot().unwrap());
        assert!(target.consume(1, true, &[]).is_err());
        assert!(target.consume_sequence(&[1]).is_err());
        assert_eq!(target.consumed_tokens(), 8);
        assert_eq!(hash_f32(&target.logits_snapshot().unwrap()), full_hash);
        let after_steps = cuda_device_allocation_snapshot();
        assert_eq!(
            after_steps.successful_allocations, resident.successful_allocations,
            "decode/reset allocated"
        );
        target.reset_empty().unwrap();
        assert_eq!(target.consumed_tokens(), 0);
        drop(target);
        let released = cuda_device_allocation_snapshot();
        assert_eq!(released.live_bytes, baseline.live_bytes);
        assert_eq!(released.live_allocations, baseline.live_allocations);
        assert_eq!(
            released.successful_allocations - baseline.successful_allocations,
            released.successful_frees - baseline.successful_frees
        );
        write_new(
            output_root.join("result.json"),
            &json!({
                "status":"PASS_NERKYOR_TARGET_RAW_IDS_RESET_REPLAY_SMOKE","profile":PROFILE,
                "target_info":info,"trajectory":first,"bitwise_reset_replay":true,
                "sequence_vs_scalar_device_handoff_bitwise_equal":true,
                "invalid_input_and_capacity_atomic_rejection":true,"primary_finite_complete_logits_snapshots":12,
                "capacity":8,"executed_token_steps":18,"load_s":loaded_s,
                "elapsed_s":started.elapsed().as_secs_f64(),"allocation_cleanup_passed":true,
                "decode_reset_allocations":0,"mtp_uploaded":false,"mtp_executed":false,
                "dflash_executed":false,"tokenizer_admitted":false,"raw_token_ids_only":true,
                "numerical_equivalence_to_reference_model_qualified":false,
                "semantic_qualified":false,"speed_qualified":false,"public_admission_changed":false
            }),
        );
    }

    fn text_trajectory(target: &mut CheckpointCudaTarget, eos: u32) -> Value {
        let mut generated = Vec::new();
        let mut logits_hashes = vec![hash_f32(&target.logits_snapshot().unwrap())];
        for _ in 0..16 {
            let token = target.select_next_token().unwrap();
            assert!(token < 248320);
            generated.push(token);
            if token == eos {
                break;
            }
            target.consume_selected().unwrap();
            let logits = target.logits_snapshot().unwrap();
            assert_eq!(logits.len(), 248320);
            assert!(logits.iter().all(|v| v.is_finite()));
            logits_hashes.push(hash_f32(&logits));
        }
        assert!(!generated.is_empty());
        json!({"generated_ids":generated,"logits_sha256":logits_hashes,
            "consumed_tokens":target.consumed_tokens()})
    }

    #[test]
    #[ignore = "exclusive SM70 + exact new tokenizer/HF fixture + authenticated model; fresh output and guard"]
    fn real_nerkyor_text_tokenizer_chunked_replay_smoke() {
        let output_root = guard_and_output();
        let started = Instant::now();
        let baseline = cuda_device_allocation_snapshot();
        let tokenizer = tokenizer::load(model());
        tokenizer::validate_samples(&tokenizer);
        let reference = tokenizer::reference();
        let prompt = &reference["chat_cases"][0];
        assert_eq!(prompt["enable_thinking"], false);
        let prompt_ids = tokenizer
            .encode(prompt["rendered"].as_str().unwrap(), false)
            .unwrap();
        let ids = prompt_ids.get_ids();
        assert_eq!(ids.len(), 22);
        let eos = tokenizer.token_to_id("<|im_end|>").unwrap();
        let mut target = CheckpointCudaTarget::open(model(), PROFILE, identity(), 64).unwrap();
        let loaded_s = started.elapsed().as_secs_f64();
        let info = target.info().clone();
        assert_eq!(info["draft_enabled"], false);
        assert_eq!(info["policy"], "checkpoint_original_f32_scalar_v1");
        let resident = cuda_device_allocation_snapshot();
        target.consume_sequence(ids).unwrap();
        let first = text_trajectory(&mut target, eos);
        let generated: Vec<u32> = first["generated_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let decoded = tokenizer.decode(&generated, false).unwrap();
        assert!(!decoded.is_empty());
        target.reset_empty().unwrap();
        assert_eq!(target.consumed_tokens(), 0);
        target.consume_sequence(&ids[..9]).unwrap();
        target.consume_sequence(&ids[9..]).unwrap();
        let chunked_replay = text_trajectory(&mut target, eos);
        assert_eq!(
            first, chunked_replay,
            "chunk boundaries/reset must not change greedy IDs or logits bits"
        );
        let steps = first["consumed_tokens"].as_u64().unwrap() * 2;
        drop(target);
        let released = cuda_device_allocation_snapshot();
        assert_eq!(released.live_bytes, baseline.live_bytes);
        assert_eq!(released.live_allocations, baseline.live_allocations);
        assert_eq!(
            released.successful_allocations,
            resident.successful_allocations
        );
        assert_eq!(
            released.successful_allocations - baseline.successful_allocations,
            released.successful_frees - baseline.successful_frees
        );
        write_new(
            output_root.join("result.json"),
            &json!({
                "status":"PASS_NERKYOR_OWN_TOKENIZER_TEXT_AND_CHUNKED_REPLAY_SMOKE",
                "profile":PROFILE,"target_info":info,"tokenizer_files":reference["files"],
                "hf_reference":{"class":reference["hf_class"],"transformers":reference["transformers_version"],
                    "tokenizers":reference["tokenizers_version"],"text_samples":20,"rendered_chat_samples":2},
                "prompt_ids":ids,"prompt_rendered":prompt["rendered"],"enable_thinking":false,
                "trajectory":first,"decoded_generated":decoded,"eos_id":eos,
                "max_new_tokens":16,"executed_token_steps":steps,
                "bitwise_chunked_reset_replay":true,"load_s":loaded_s,"elapsed_s":started.elapsed().as_secs_f64(),
                "allocation_cleanup_passed":true,"decode_reset_allocations":0,
                "mtp_uploaded":false,"mtp_executed":false,"dflash_executed":false,
                "own_tokenizer_authenticated":true,"rust_jinja_renderer_tested":false,
                "all_unicode_qualified":false,"numerical_equivalence_to_hf_model_qualified":false,
                "semantic_benchmark_qualified":false,"speed_qualified":false,"public_admission_changed":false
            }),
        );
    }
}
