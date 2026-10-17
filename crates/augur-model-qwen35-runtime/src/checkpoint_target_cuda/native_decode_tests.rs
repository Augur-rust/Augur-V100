//! Target-only decode wall and separate profiler capture. Never loads DFlash.
use super::*;
use augur_backend::{cuda_host_boundary_snapshot, CudaProfilerCapture};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{io::Write, path::PathBuf};

mod offline_prefix;

fn hash(raw: &[u8]) -> String { format!("{:x}", Sha256::digest(raw)) }
fn save(root: &Path, name: &str, value: &Value) {
    let mut f=std::fs::OpenOptions::new().create_new(true).write(true).open(root.join(name)).unwrap();
    f.write_all(&serde_json::to_vec_pretty(value).unwrap()).unwrap();f.sync_all().unwrap();
}
fn prefill(t: &mut CheckpointCudaTarget, ids: &[u32]) -> f64 {
    t.reset_empty().unwrap();
    let tick=Instant::now();
    let plan=t.plan_augur_sequence(ids,ids.chunks(4352).map(|s|s.len()).collect()).unwrap();
    t.consume_sequence_augur(ids,&plan,augur_prefill::Budget::default(),|_|false).unwrap();
    t.stream.synchronize().unwrap();tick.elapsed().as_secs_f64()
}
fn run(t: &mut CheckpointCudaTarget, ids: &[u32], budget: usize, eos: &[u32], profile: bool) -> Value {
    let prefill_s=prefill(t,ids);
    let mut result=decode_current(t,ids.len(),budget,eos,profile);
    result["prefill_s"]=prefill_s.into();
    result["fresh_prefill_executed"]=true.into();
    result
}
// Shared measurement body: restored logits and fresh prefill use identical
// consume-selected/greedy timing. Restore and the first token remain outside it.
fn decode_current(t: &mut CheckpointCudaTarget, prompt_tokens: usize, budget: usize, eos: &[u32], profile: bool) -> Value {
    assert!(budget>1 && budget<=1024);
    assert_eq!(t.consumed,prompt_tokens);
    assert!(prompt_tokens.checked_add(budget).is_some_and(|n|n<=t.capacity));
    // The prefill-computed first token is NOT charged as a target decode step.
    let first=t.greedy_next_token().unwrap(); assert!(!eos.contains(&first));
    t.stream.synchronize().unwrap();
    let before=cuda_device_allocation_snapshot();let boundary=cuda_host_boundary_snapshot();
    let mut capture=if profile{Some(CudaProfilerCapture::start().unwrap())}else{None};
    let mut output=vec![first];let mut steps=0usize;let mut step_s=Vec::with_capacity(budget);
    let tick=Instant::now();let stop=loop {
        let one=Instant::now();
        t.consume_selected_augur().unwrap();
        let id=t.greedy_next_token().unwrap();
        step_s.push(one.elapsed().as_secs_f64());steps+=1;
        if eos.contains(&id){break "eos";}
        output.push(id);if output.len()==budget{break "length";}
    };
    t.stream.synchronize().unwrap();let wall=tick.elapsed().as_secs_f64();
    if let Some(c)=capture.as_mut(){c.stop().unwrap();}
    let delta=cuda_host_boundary_snapshot().checked_delta_since(boundary).unwrap();
    let after=cuda_device_allocation_snapshot();
    assert_eq!(after.successful_allocations,before.successful_allocations);
    assert_eq!(after.successful_frees,before.successful_frees);
    assert_eq!(delta.synchronous_h2d_bytes,0);assert_eq!(delta.synchronous_d2h_bytes,steps as u64*4);
    assert_eq!(t.consumed,prompt_tokens+steps);
    let logits=t.logits_snapshot().unwrap();let pending=t.greedy_next_token().unwrap();
    json!({"output":output,"output_sha256":hash(&output.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()),
        "output_tokens":output.len(),"decode_steps":steps,"decode_wall_s":wall,"decode_steps_tps":steps as f64/wall,
        "post_prefill_output_tps":(output.len()-1) as f64/wall,"step_wall_s":step_s,"prefill_s":Value::Null,
        "stop":stop,"pending":pending,"consumed":t.consumed,"prompt_tokens":prompt_tokens,
        "logits_sha256":hash(&logits.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()),
        "host_boundary":super::transfer_tests::boundary(delta),"gpu_allocations":0,"gpu_frees":0,
        "profiler_instrumented":profile,"diagnostic_copies_inside_wall":false,"first_token_from_prefill":true})
}
#[test]
#[ignore = "real full262144 native target decode; authenticated external model; exclusive GPU"]
fn native_decode_wall_and_profile() {
    let out=PathBuf::from(std::env::var_os("AUGUR_NATIVE_DECODE_OUT").unwrap());
    let raw=std::fs::read(std::env::var_os("AUGUR_NATIVE_DECODE_INPUT").unwrap()).unwrap();
    assert_eq!(hash(&raw),std::env::var("AUGUR_NATIVE_DECODE_INPUT_SHA256").unwrap());
    let input:Value=serde_json::from_slice(&raw).unwrap();
    let manifest_raw=std::fs::read(std::env::var_os("AUGUR_MANIFEST").unwrap()).unwrap();
    assert_eq!(hash(&manifest_raw),std::env::var("AUGUR_MANIFEST_SHA256").unwrap());
    let manifest:Value=serde_json::from_slice(&manifest_raw).unwrap();
    let identity:CheckpointExpectedIdentity=serde_json::from_value(manifest["checkpoint"].clone()).unwrap();
    let initial=cuda_device_allocation_snapshot();let started=Instant::now();
    let mut t=CheckpointCudaTarget::open_with_execution_policy(std::env::var_os("AUGUR_MODEL_ROOT").unwrap(),
        Qwen35CheckpointProfile::QuasarNvfp4,identity,262144,
        CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64,CheckpointKvPolicy::Q8_0).unwrap();
    t.enable_exact_warp_tree_norm().unwrap();t.enable_fused_input_qpn_norm().unwrap();t.enable_batched_ffn8().unwrap();
    t.enable_augur_decode().unwrap();assert!(t.dflash_capture.is_none());
    let load_s=started.elapsed().as_secs_f64();
    let cases=input["cases"].as_array().unwrap();let eos=super::transfer_tests::ids(&input["eos"]);
    let profile=input["profile"].as_bool().unwrap();
    let warm=super::transfer_tests::ids(&cases[0]["token_ids"]);
    let warmup_tokens=warmup_extent(&input,warm.len()).unwrap();
    let _=prefill(&mut t,&warm[..warmup_tokens]);
    let mut results=Vec::new();
    for repeat in 0..input["repetitions"].as_u64().unwrap(){
        let mut order:Vec<_>=cases.iter().collect();if repeat%2==1{order.reverse();}
        for case in order {
            let id=case["id"].as_str().unwrap();let ids=super::transfer_tests::ids(&case["token_ids"]);
            let mut result=run(&mut t,&ids,case["max_new_tokens"].as_u64().unwrap() as usize,&eos,profile);
            if std::env::var("AUGUR_NATIVE_DECODE_STATE_AUDIT").as_deref() == Ok("1") {
                // Explicit diagnostic after the timed generation and its final drain.
                let mut history = ids.clone();
                let output = super::transfer_tests::ids(&result["output"]);
                let appended = t.consumed.checked_sub(history.len()).unwrap();
                assert!(appended <= output.len());
                history.extend_from_slice(&output[..appended]);
                result["full_state_sha256"] = offline_prefix::diagnostic_digest(&t, &history).into();
                result["full_state_audit_outside_timing"] = true.into();
            }
            result["id"]=id.into();result["repeat"]=repeat.into();result["family"]=case["family"].clone();
            save(&out,&format!("{id}-r{repeat}.json"),&result);
            println!("NATIVE_DECODE {id} r{repeat} steps={} wall={} tps={}",result["decode_steps"],result["decode_wall_s"],result["decode_steps_tps"]);
            results.push(result);
        }
    }
    if let Some(id)=input["preserve_prefix_case"].as_str() {
        let case=cases.iter().find(|case|case["id"].as_str()==Some(id)).expect("offline prefix case");
        let ids=super::transfer_tests::ids(&case["token_ids"]);
        let fresh_s=prefill(&mut t,&ids);
        let receipt=offline_prefix::preserve(&mut t,&ids,&out.join("offline-prefix"));
        save(&out,"offline-prefix-complete.json",&json!({"fresh_prefill_s_excluded_from_score":fresh_s,"cache":receipt}));
    }
    let info=t.info().clone();drop(t);let closed=cuda_device_allocation_snapshot();
    assert_eq!(closed.live_bytes,initial.live_bytes);assert_eq!(closed.live_allocations,initial.live_allocations);
    save(&out,"native-complete.json",&json!({"status":"PASS_NATIVE_DECODE","target":info,"load_s":load_s,
        "dflash_loaded":false,"dflash_feature_capture":false,"http":false,"capacity":262144,
        "profile":profile,"warmup_tokens":warmup_tokens,"results":results,"live_bytes_after":closed.live_bytes,"live_allocations_after":closed.live_allocations}));
}

fn warmup_extent(input:&Value, prompt_tokens:usize)->std::result::Result<usize,&'static str> {
    if prompt_tokens==0 || prompt_tokens>262127 {return Err("warmup prompt extent");}
    match input.get("warmup_prefix_tokens") {
        None=>Ok(prompt_tokens),
        Some(v)=>match v.as_u64() {
            Some(n) if (1..=262127).contains(&n)=>Ok(prompt_tokens.min(n as usize)),
            _=>Err("warmup prefix must be a positive bounded integer"),
        },
    }
}
#[test]
fn native_warmup_extent_is_explicit_and_bounded() {
    assert_eq!(warmup_extent(&json!({}),131040),Ok(131040));
    assert_eq!(warmup_extent(&json!({"warmup_prefix_tokens":8176}),131040),Ok(8176));
    assert_eq!(warmup_extent(&json!({"warmup_prefix_tokens":8176}),26),Ok(26));
    for value in [json!(0),json!(-1),json!(true),json!("8176"),json!(262128),Value::Null] {
        assert!(warmup_extent(&json!({"warmup_prefix_tokens":value}),131040).is_err());
    }
    assert!(warmup_extent(&json!({}),0).is_err());
    assert!(warmup_extent(&json!({}),262128).is_err());
}
