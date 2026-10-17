//! Resident native request timing, deliberately separate from replay diagnostics.
//! No HTTP/DFlash/product qualification and no change to projection mathematics.
use super::tests::{bytes, hash};
use super::transfer_tests::{boundary, ids, open};
use super::*;
use augur_backend::cuda_host_boundary_snapshot;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[test]
#[ignore = "exclusive CUDA; authenticated actual8K input, full target and resident KV8"]
fn real_checkpoint_context_normal_timing_cleanup() {
    let setup_started = Instant::now();
    let source =
        std::fs::read(std::env::var_os("AUGUR_CHECKPOINT_TARGET_REFERENCE").expect("reference"))
            .unwrap();
    let case: Value = serde_json::from_slice(&source).unwrap();
    let root =
        std::path::PathBuf::from(std::env::var_os("AUGUR_CHECKPOINT_TARGET_OUTPUT").expect("output"));
    std::fs::create_dir(&root).unwrap();
    assert_eq!(case["profile"], "quasar");
    let prompt = ids(&case["context_request"]["token_ids"]);
    assert!((8192..=16384).contains(&prompt.len()));
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
    let eos = ids(&case["eos_token_ids"]);
    let budget = case["context_request"]["max_new_tokens"].as_u64().unwrap() as usize;
    let expected = case["context_request"]["expected_text"].as_str().unwrap();
    assert!((1..=64).contains(&budget));
    assert!(!eos.is_empty() && !expected.is_empty());
    let repetitions: usize = std::env::var("AUGUR_CHECKPOINT_TIMING_REPETITIONS")
        .unwrap_or_else(|_| "3".into())
        .parse()
        .unwrap();
    assert!((1..=5).contains(&repetitions));
    let fixture_setup_s = setup_started.elapsed().as_secs_f64();
    let baseline = cuda_device_allocation_snapshot();
    let load_started = Instant::now();
    let mut target = open(&case);
    assert!(prompt.len() + budget <= target.capacity);
    assert_eq!(target.info()["kv_dtype"], "Q8_0");
    assert_eq!(target.capacity, 196608);
    assert_eq!(target.info()["ffn_batch_max_rows"], 8);
    // The real prompt slot and prebound gather array cover capacity, not 256.
    // The only activation batch remains <=8 inside Batch::submit.
    assert!(target.prompt_tokens.len_bytes() >= prompt.len() * 4);
    assert!(target.prompt_gathers.len() >= prompt.len());
    target.stream.synchronize().unwrap();
    let load_and_setup_s = load_started.elapsed().as_secs_f64();
    let info = target.info().clone();
    let warmup_started = Instant::now();
    target.consume_sequence(&prompt[..8]).unwrap();
    target.greedy_next_token().unwrap();
    let warmup_s = warmup_started.elapsed().as_secs_f64();
    let mut requests = Vec::<Value>::with_capacity(repetitions);
    for repetition in 0..repetitions {
        let reset_started = Instant::now();
        target.reset_empty().unwrap();
        // reset_empty drains prior work before resets; explicitly drain any
        // reset kernels too, keeping them outside the measured request.
        target.stream.synchronize().unwrap();
        let reset_s = reset_started.elapsed().as_secs_f64();
        let owned_before = cuda_device_allocation_snapshot();
        let mut output = Vec::with_capacity(budget);
        let mut pending = 0;
        let mut selected_count = 0u64;
        let mut first_s = 0.0;
        let mut stopped = false;
        let boundary_before = cuda_host_boundary_snapshot();
        let started = Instant::now();
        // Includes one input H2D, the final head and the exact owner's stream
        // barrier: prefill_wall_s measures completed work, not enqueue time.
        target.consume_sequence(&prompt).unwrap();
        let prefill_s = started.elapsed().as_secs_f64();
        for index in 0..budget {
            // Each uncached selection reads4 bytes, completing all preceding
            // work on the same stream; the final EOS read also closes timing.
            let token = target.greedy_next_token().unwrap();
            if index == 0 {
                first_s = started.elapsed().as_secs_f64();
            }
            selected_count += 1;
            pending = token;
            if eos.contains(&token) {
                stopped = true;
                break;
            }
            output.push(token);
            if index + 1 < budget {
                target.consume_selected().unwrap();
            }
        }
        let whole_s = started.elapsed().as_secs_f64();
        let counters = cuda_host_boundary_snapshot()
            .checked_delta_since(boundary_before)
            .unwrap();
        let owned_after = cuda_device_allocation_snapshot();
        let validation_started = Instant::now();
        assert!(stopped, "this canonical-answer fixture must stop with EOS");
        assert!(!output.is_empty());
        assert_eq!(target.consumed_tokens(), prompt.len() + output.len());
        assert_eq!(counters.synchronous_h2d_copies, 1);
        assert_eq!(counters.synchronous_h2d_bytes, (prompt.len() * 4) as u64);
        assert_eq!(counters.synchronous_d2h_copies, selected_count);
        assert_eq!(counters.synchronous_d2h_bytes, selected_count * 4);
        assert_eq!(counters.stream_synchronizations, 1);
        assert_eq!(counters.event_synchronizations, 0);
        assert_eq!(
            owned_before.successful_allocations,
            owned_after.successful_allocations
        );
        assert_eq!(owned_before.live_bytes, owned_after.live_bytes);
        assert_eq!(owned_before.live_allocations, owned_after.live_allocations);
        // No prefill snapshot is taken: pausing the clock for diagnostic work
        // would not be an uninterrupted normal request. Only final logits are
        // read after timing. Same-policy prior replay evidence is bound by the
        // comparison harness; cross-policy hashes need not be equal.
        let final_hash = hash(&target.logits_snapshot().unwrap());
        let text = tokenizer.decode(&output, false).unwrap();
        assert_eq!(text.trim(), expected, "canonical8K task answer");
        if let Some(first) = requests.first() {
            assert_eq!(first["output_token_ids"], json!(output));
            assert_eq!(first["pending_token"], pending);
            assert_eq!(first["final_logits_sha256"], final_hash);
        }
        let validation_s = validation_started.elapsed().as_secs_f64();
        let request = json!({"repetition":repetition,"path":"sequence_device_top1",
            "input_tokens":prompt.len(),"prefill_submissions":1,"output_token_ids":output,
            "pending_token":pending,"selected_tokens_including_eos":selected_count,
            "consumed_tokens":target.consumed_tokens(),"text":text,"expected_text":expected,
            "task_answer_matches":true,"stop_reason":"eos","final_logits_sha256":final_hash,
            "prefill_logits_sha256":null,"prefill_snapshot_omitted_to_keep_timing_uninterrupted":true,
            "prefill_wall_s":prefill_s,"first_token_wall_s":first_s,"whole_wall_s":whole_s,
            "first_token_to_finish_s":whole_s-first_s,"boundary":boundary(counters),
            "reset_and_drain_s":reset_s,"post_timing_validation_s":validation_s,
            "diagnostic_io_in_timed_region":false,"gpu_completion_in_timed_region":true,
            "hot_owned_allocations":0,"owned_live_bytes":owned_after.live_bytes,
            "owned_live_allocations":owned_after.live_allocations,"product_ready":false});
        std::fs::write(
            root.join(format!("request-{repetition}.json")),
            serde_json::to_vec_pretty(&request).unwrap(),
        )
        .unwrap();
        println!("CHECKPOINT_CONTEXT_NORMAL_TIMING repetition={repetition} whole_s={whole_s:.9} answer={text:?}");
        requests.push(request);
    }
    let peak = cuda_device_allocation_snapshot().peak_live_bytes;
    let teardown_started = Instant::now();
    drop(target);
    let teardown_s = teardown_started.elapsed().as_secs_f64();
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes, baseline.live_bytes);
    assert_eq!(after.live_allocations, baseline.live_allocations);
    let allocations = after.successful_allocations - baseline.successful_allocations;
    let frees = after.successful_frees - baseline.successful_frees;
    assert_eq!(allocations, frees);
    std::fs::write(
        root.join("result.json"),
        serde_json::to_vec_pretty(&json!({
        "status":"PASS_CHECKPOINT_NORMAL_CONTEXT_TIMING_CLEANUP",
        "reference_sha256":format!("{:x}",Sha256::digest(&source)),
        "targets":[{"info":info,"requests":requests}],"actual_prompt_tokens":prompt.len(),
        "fixture_setup_s":fixture_setup_s,"load_and_setup_s":load_and_setup_s,
        "warmup_tokens":8,"warmup_s":warmup_s,"teardown_s":teardown_s,
        "peak_owned_live_bytes":peak,"live_bytes_after":after.live_bytes,
        "live_allocations_after":after.live_allocations,"successful_allocations":allocations,
        "successful_frees":frees,"request_timing_scope":"resident_native_not_HTTP",
        "diagnostic_io_in_timed_region":false,"gpu_completion_in_timed_region":true,
        "product_ready":false,"semantic_qualified":false,"http_tested":false,
        "dflash_enabled":false,"vision_enabled":false}))
        .unwrap(),
    )
    .unwrap();
}
