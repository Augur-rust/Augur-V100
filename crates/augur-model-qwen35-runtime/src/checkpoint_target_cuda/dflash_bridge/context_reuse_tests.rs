//! Diagnostic equality, not timing: same owner, full versus retained context.
use super::*;
use crate::DFlashQ8Backbone;
use augur_model_qwen35::DFlashConfig;
use std::io::Write;

fn prompt(target: &mut CheckpointCudaTarget, ids: &[u32]) {
    let widths = ids.chunks(4352).map(|part| part.len()).collect();
    let plan = target.plan_augur_sequence(ids, widths).unwrap();
    target.consume_sequence_augur(ids, &plan, super::super::augur_prefill::Budget::default(), |_| false).unwrap();
}

fn equal_plane(left: &[f32], right: &[f32], name: &str) -> String {
    assert_eq!(left.len(), right.len(), "{name} extent");
    let mismatch = left.iter().zip(right).position(|(a, b)| a.to_bits() != b.to_bits());
    assert!(mismatch.is_none(), "{name} first bit mismatch at {mismatch:?}");
    let raw: Vec<u8> = left.iter().flat_map(|x| x.to_le_bytes()).collect();
    format!("{:x}", Sha256::digest(raw))
}

fn compare(target: &mut CheckpointCudaTarget, full: &mut AugurDflashBridge,
    reuse: &mut AugurDflashBridge, name: &str, expected_retained: usize) -> Value {
    let before = cuda_device_allocation_snapshot();
    let anchor = target.greedy_next_token().unwrap();
    let full_proposal = full.propose_resident(target).unwrap();
    let full_head = full.head.read_diagnostics(anchor).unwrap();
    let reuse_proposal = reuse.propose_resident(target).unwrap();
    let reuse_head = reuse.head.read_diagnostics(anchor).unwrap();
    let (start, end) = reuse_proposal.initialized_feature_range();
    let full_rows = full.resident.as_ref().unwrap().draft.as_ref().unwrap().context_projection_rows();
    let reuse_rows = reuse.resident.as_ref().unwrap().draft.as_ref().unwrap().context_projection_rows();
    assert_eq!(full_rows, end - start);
    assert_eq!(reuse_rows, end - start - expected_retained);
    assert_eq!(full_head.path, reuse_head.path);
    assert_eq!(full_head.candidate_ids, reuse_head.candidate_ids);
    let hashes = json!({
        "hidden":equal_plane(&full_head.normalized_hidden, &reuse_head.normalized_hidden, "normalized draft hidden"),
        "logits":equal_plane(&full_head.logits, &reuse_head.logits, "all vocabulary logits"),
        "unary":equal_plane(&full_head.unary_scores, &reuse_head.unary_scores, "top16 unary"),
        "transitions":equal_plane(&full_head.transition_scores, &reuse_head.transition_scores, "selector transitions")
    });
    let path = full_head.path.clone();
    full.discard_resident(full_proposal).unwrap();
    reuse.discard_resident(reuse_proposal).unwrap();
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.successful_allocations, before.successful_allocations);
    assert_eq!(after.successful_frees, before.successful_frees);
    json!({"case":name,"context_start":start,"context_rows":end-start,
        "full_projected_rows":full_rows,"retained_projected_rows":reuse_rows,
        "reused_rows":expected_retained,"path":path,"full_plane_sha256":hashes,
        "all_planes_bit_exact":true,"GPU_allocations":0,"GPU_frees":0,"performance_measurement":false})
}

#[test]
#[ignore = "exclusive GPU, full256Ki current target and two draft workspaces; cache equality not timing"]
fn retained_context_matches_full_across_append_roll_reset_and_disjoint_history() {
    use super::super::transfer_tests::{ids, open};
    let fixture: Value = serde_json::from_slice(&std::fs::read(
        std::env::var_os("AUGUR_CHECKPOINT_TARGET_REFERENCE").unwrap()).unwrap()).unwrap();
    let cfg = std::fs::read(std::env::var_os("AUGUR_DRAFT_CONFIG").unwrap()).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&cfg)), std::env::var("AUGUR_DRAFT_CONFIG_SHA256").unwrap());
    let q8 = DFlashQ8Backbone::load_pinned(std::env::var_os("AUGUR_DRAFT_PATH").unwrap(), DFlashConfig::parse_f16_opt_in(&cfg).unwrap()).unwrap();
    assert_eq!(q8.manifest().file_sha256.as_ref().unwrap(), &std::env::var("AUGUR_DRAFT_SHA256").unwrap());
    let before = cuda_device_allocation_snapshot();
    let mut target = open(&fixture);
    assert_eq!(target.capacity, 262144);
    target.enable_augur_decode().unwrap();
    target.enable_augur_feature_capture().unwrap();
    let weights = DFlashQ8CudaWeights::upload(&q8, 0).unwrap();
    let mut full = AugurDflashBridge::new(&target, &weights).unwrap();
    let mut reuse = AugurDflashBridge::new(&target, &weights).unwrap();
    full.reuse_context = false;
    reuse.reuse_context = true;
    let seed = ids(&fixture["token_ids"]);
    assert!(!seed.is_empty());
    // Synthetic valid IDs exercise exact retained-window boundaries; these are
    // not semantic quality prompts or a claim of real256Ki history qualification.
    let prefix: Vec<u32> = seed.iter().copied().cycle().take(2046).collect();
    prompt(&mut target, &prefix);
    full.retain_prefill_benchmark_workspace(&mut target).unwrap();
    reuse.retain_prefill_benchmark_workspace(&mut target).unwrap();
    let mut cases = vec![compare(&mut target, &mut full, &mut reuse, "first_unused_seed", 0)];
    cases.push(compare(&mut target, &mut full, &mut reuse, "unchanged_repeat", 2046));
    for (advance, name, retained) in [(1, "append_without_eviction", 2046), (7, "cross_2048_boundary", 2041), (8, "rolling_window", 2040)] {
        for _ in 0..advance {
            target.greedy_next_token().unwrap();
            target.consume_selected_augur().unwrap();
        }
        cases.push(compare(&mut target, &mut full, &mut reuse, name, retained));
    }
    let suffix: Vec<u32> = seed.iter().copied().cycle().take(2100).collect();
    prompt(&mut target, &suffix);
    cases.push(compare(&mut target, &mut full, &mut reuse, "disjoint_teacher_forced_suffix", 0));
    target.reset_empty().unwrap();
    prompt(&mut target, &seed);
    cases.push(compare(&mut target, &mut full, &mut reuse, "new_epoch_full_refresh", 0));
    let peak = cuda_device_allocation_snapshot();
    drop(reuse); drop(full); drop(weights); drop(target);
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes, before.live_bytes);
    assert_eq!(after.live_allocations, before.live_allocations);
    let value = json!({"schema":"quasar-retained-context-real-gpu-v1","passed":true,"cases":cases,
        "actual_max_history_tokens":2046+1+7+8+2100,"physical_KV_capacity":262144,
        "owned_peak_bytes":peak.peak_live_bytes,"cleanup_live_bytes_delta":after.live_bytes-before.live_bytes,
        "cleanup_live_allocations_delta":after.live_allocations-before.live_allocations,
        "synthetic_boundary_inputs":true,"performance_measurement":false,"HTTP_qualified":false});
    let path = std::path::PathBuf::from(std::env::var_os("AUGUR_BRIDGE_OUTPUT").unwrap()).join("context-reuse-gpu.json");
    let mut file = std::fs::OpenOptions::new().create_new(true).write(true).open(path).unwrap();
    file.write_all(&serde_json::to_vec_pretty(&value).unwrap()).unwrap(); file.sync_all().unwrap();
}
