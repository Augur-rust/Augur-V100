//! Experimental whole-prompt B8 teacher forcing. No public/session admission:
//! the existing numerical identity explicitly retains scalar prefill.
use super::*;
use crate::checkpoint_projection_cuda::CheckpointProjectionPolicy;
use crate::checkpoint_target_cuda::nerkyor_dflash_research::batched::Batch;
use super::batched::{compare, snapshot};

// Test-only experiment: one canonical scalar token establishes the committed
// target required by Batch, followed by full B8 commits and a scalar tail.
// Every Batch commit includes GDN, complete canonical KV, capture and logits.
fn candidate(t: &mut CheckpointCudaTarget, ids: &[u32], batch: &mut Option<Batch>) {
    assert!(!ids.is_empty());
    let mut at = 0;
    if t.consumed_tokens() == 0 {
        t.consume_sequence(&ids[..1]).unwrap();
        at = 1;
    }
    if ids.len() - at >= 8 && batch.is_none() {
        let mut prepared=Batch::new(t).unwrap();
        if std::env::var("AUGUR_NERKYOR_PREFILL_ATTENTION").as_deref()==Ok("core") {
            prepared.enable_core_attention_candidate().unwrap();
        }
        *batch = Some(prepared);
    }
    while ids.len() - at >= 8 {
        let b = batch.as_mut().unwrap();
        b.execute(t, &ids[at..at + 8]).unwrap();
        b.commit(t, 8).unwrap();
        at += 8;
    }
    if at < ids.len() {
        t.consume_sequence(&ids[at..]).unwrap();
    }
}
fn scalar(t: &mut CheckpointCudaTarget, ids: &[u32]) {
    for chunk in ids.chunks(8) {
        t.consume_sequence(chunk).unwrap();
    }
}
fn top(raw: &[u8]) -> usize {
    raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .enumerate()
        .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
        .unwrap().0
}
fn metrics(actual: &[(String, Vec<u8>)], reference: &[(String, Vec<u8>)]) -> (Value, bool) {
    assert_eq!(actual.len(), reference.len());
    let mut all = true;
    let rows = actual.iter().zip(reference).map(|((name, a), (other, b))| {
        assert_eq!(name, other);
        let metric = compare(a, b);
        all &= metric["pass"] == true;
        json!({"tensor":name,"metric":metric})
    }).collect::<Vec<_>>();
    assert_eq!(actual[0].0, "logits");
    let top1 = top(&actual[0].1) == top(&reference[0].1);
    all &= top1;
    (json!({"all_tensors_contract_passed":all,"top1_equal":top1,"tensors":rows}), all)
}

#[test]
#[ignore = "exclusive SM70; multi-chunk B8 prefill candidate, full state numerical gate"]
fn real_nerkyor_b8_prefill_candidate_state_and_timing() {
    let arithmetic=std::env::var("AUGUR_NERKYOR_PREFILL_ATTENTION").unwrap_or_else(|_|"original".into());
    assert!(arithmetic=="original" || arithmetic=="core");
    let guard = read(std::env::var_os("AUGUR_NERKYOR_GPU_GUARD").unwrap());
    assert_eq!(guard["gpu_compute_apps_empty"], true);
    assert_eq!(guard["queues_paused"], true);
    assert_eq!(guard["benchmark_auto_resume"], false);
    assert_eq!(cuda_device_properties(0).unwrap().compute_capability, CudaComputeCapability::SM70);
    let out = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_GPU_OUTPUT").unwrap());
    fs::create_dir(&out).unwrap();
    record(out.join("policy.json"), json!({"declared_before_output":true,"attention_arithmetic":arithmetic,"relative_l2_limit":1e-5,
        "absolute_floor":1e-4,"max_abs_over_reference_peak_limit":1e-5,
        "scope":"full logits/capture/all KV/all GDN state across whole teacher-forced prompt",
        "candidate":"test-only B8 full commits; initial scalar token and residual scalar tail",
        "existing_scalar_identity_not_requalified":true,"product_admitted":false}));
    let model = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_MODEL").unwrap());
    let draft = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_DFLASH_Q8").unwrap());
    let reference = read(std::env::var_os("AUGUR_NERKYOR_DFLASH_REFERENCE").unwrap());
    let config = fs::read(draft.join("config.json")).unwrap();
    assert_eq!(format!("{:x}",Sha256::digest(&config)),reference["config_sha256"].as_str().unwrap());
    let identity: CheckpointExpectedIdentity = serde_json::from_slice(
        &fs::read(std::env::var_os("AUGUR_NERKYOR_IDENTITY").unwrap()).unwrap()).unwrap();
    let tokenizer = tokenizers::Tokenizer::from_bytes(&fs::read(model.join("tokenizer.json")).unwrap()).unwrap();
    let baseline = cuda_device_allocation_snapshot();
    let loaded = Instant::now();
    let cpu = DFlashQ8Backbone::load_pinned(draft.join("qwen35-27b-dflash2-q8_0.gguf"),DFlashConfig::parse(&config).unwrap()).unwrap();
    assert_eq!(cpu.manifest().file_sha256.as_deref(),reference["gguf_sha256"].as_str());
    let weights = DFlashQ8CudaWeights::upload(&cpu,0).unwrap(); drop(cpu);
    let mut target = CheckpointCudaTarget::open_with_projection_policy(&model,
        Qwen35CheckpointProfile::NerkyorW4a16,identity,512,
        CheckpointProjectionPolicy::NerkyorBf16GateF32CtaResearch).unwrap();
    target.enable_nerkyor_dflash_research_capture(&weights).unwrap();
    let load_s=loaded.elapsed().as_secs_f64();
    let checkpoints = [1,8,9,16,17,63,64,65,129,257,481];
    let texts = [
        "Reference notes: a stack is last in first out; a queue is first in first out. Python: def add(a, b): return a + b.\n",
        "参考资料：栈遵循后进先出，队列遵循先进先出。这里记录数字、中文与代码，供上下文状态测试使用。\n"
    ];
    let mut all_ok=true;let mut cases=Vec::new();let mut timings=Vec::new();
    for (case,text) in texts.iter().enumerate() {
        let stream=tokenizer.encode(text.repeat(40),false).unwrap().get_ids().to_vec();
        assert!(stream.len()>481);
        let ids=&stream[..481];
        record(out.join(format!("input-{case}.json")),json!({"self_authored":true,"token_ids":ids,"tokens":ids.len(),"checkpoints":checkpoints}));
        target.reset_empty().unwrap();
        let mut previous=0;let mut expected=Vec::new();
        for &end in &checkpoints {
            scalar(&mut target,&ids[previous..end]);
            expected.push(snapshot(&mut target));previous=end;
        }
        target.reset_empty().unwrap();let mut batch=None;previous=0;
        let mut rows=Vec::new();
        for (&end,gold) in checkpoints.iter().zip(&expected) {
            candidate(&mut target,&ids[previous..end],&mut batch);
            assert_eq!(target.consumed_tokens(),end);
            let actual=snapshot(&mut target);
            let (metric,pass)=metrics(&actual,gold);all_ok &= pass;
            rows.push(json!({"tokens":end,"metric":metric}));previous=end;
        }
        drop(batch);drop(expected);
        cases.push(json!({"case":case,"boundaries":rows}));
        // Additional contiguous whole-input runs exercise a different partition
        // and include setup, final head/stream barrier, but exclude model load.
        // Each timing is followed by a full-state contract check, not text-only.
        for (repeat,modes) in [(0,["scalar","candidate"]),(1,["candidate","scalar"])] {
            let mut pair=std::collections::BTreeMap::new();
            for mode in modes {
                target.reset_empty().unwrap();
                let before=cuda_device_allocation_snapshot().successful_allocations;
                let start=Instant::now();
                let mut b=None;
                if mode=="scalar" { scalar(&mut target,&ids[..257]); }
                else { candidate(&mut target,&ids[..257],&mut b); }
                let elapsed=start.elapsed().as_secs_f64();
                let allocations=cuda_device_allocation_snapshot().successful_allocations-before;
                drop(b);
                pair.insert(mode,snapshot(&mut target));
                timings.push(json!({"case":case,"repeat":repeat,"mode":mode,"tokens":257,
                    "prefill_including_plan_setup_s":elapsed,"tokens_per_s":257.0/elapsed,
                    "cuda_allocations_including_plan":allocations}));
            }
            let (metric,pass)=metrics(&pair["candidate"],&pair["scalar"]);all_ok &= pass;
            record(out.join(format!("repeat-{case}-{repeat}.json")),metric);
        }
    }
    target.reset_empty().unwrap();drop(target);drop(weights);
    let freed=cuda_device_allocation_snapshot();
    assert_eq!(freed.live_bytes,baseline.live_bytes);
    assert_eq!(freed.live_allocations,baseline.live_allocations);
    record(out.join("result.json"),json!({"status":if all_ok {"PASS_B8_PREFILL_CANDIDATE_FULL_STATE"} else {"FAIL_B8_PREFILL_CANDIDATE_NUMERICAL_CONTRACT"},
        "attention_arithmetic":arithmetic,"load_s":load_s,"cases":cases,"timings":timings,"full_state_contract_passed":all_ok,
        "allocation_cleanup_passed":true,"existing_execution_identity_not_requalified":true,
        "candidate_in_product":false,"official_BF16_qualification":false,"general_speed_qualification":false,
        "mtp_executed":false,"old_benchmarks_resumed":false}));
    assert!(all_ok,"B8 prefill candidate exceeds fixed numerical contract; do not admit or loosen limits");
}
