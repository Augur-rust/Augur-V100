//! Real same-owner recovery. Only typed, unpublished cancellation authorizes reset.
use super::*;
use super::gpu_tests::{close, metric, read, record, state, turn};
use augur_backend::{cuda_device_allocation_snapshot, cuda_device_properties, CudaComputeCapability};
use std::{fs, path::PathBuf, thread, time::Duration};

#[test]
#[ignore = "exclusive SM70; cancellation discard and full-state replay without weight reload"]
fn real_request_cancel_discard_prefill_decode_append_replays_same_owner() {
    let guard=read(std::env::var_os("AUGUR_NERKYOR_GPU_GUARD").unwrap());
    assert_eq!(guard["gpu_compute_apps_empty"],true);
    assert_eq!(guard["queues_paused"],true);
    assert_eq!(guard["benchmark_auto_resume"],false);
    assert_eq!(cuda_device_properties(0).unwrap().compute_capability,CudaComputeCapability::SM70);
    let out=PathBuf::from(std::env::var_os("AUGUR_NERKYOR_GPU_OUTPUT").unwrap());
    fs::create_dir(&out).unwrap();
    let root=PathBuf::from(std::env::var_os("AUGUR_NERKYOR_MODEL").unwrap());
    let draft=PathBuf::from(std::env::var_os("AUGUR_NERKYOR_DFLASH_Q8").unwrap());
    let manifest=PathBuf::from(std::env::var_os("AUGUR_NERKYOR_SESSION_MANIFEST").unwrap());
    let pin=std::env::var("AUGUR_NERKYOR_SESSION_MANIFEST_SHA256").unwrap();
    let admitted=crate::nerkyor_session_manifest::load(&root,&draft,&manifest,&pin).unwrap();
    let tokenizer=tokenizers::Tokenizer::from_bytes(admitted.tokenizer_bytes()).unwrap();
    let encode=|s:String|tokenizer.encode(s,false).unwrap().get_ids().to_vec();
    let content=format!("Read these notes: {} Now count from 1 to 20, separated by spaces. Output only the numbers.",
        "A stack is last in first out; a queue is first in first out. ".repeat(16));
    let prompt=encode(format!("<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    assert!(prompt.len()>250 && prompt.len()+32<CAPACITY);
    let short=encode("<|im_start|>user\nCount from 1 to 20, separated by spaces. Output only the numbers.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".into());
    let suffix=encode("<|im_end|>\n<|im_start|>user\nNow count from 1 to 50, separated by spaces. Output only the numbers.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".into());
    let request=|id:&str|TurnRequest { request_id:id.into(),session_id:format!("session-{id}"),
        input:TurnInput::Open{prompt_token_ids:prompt.clone()},max_new_tokens:24,selector_policy:None };
    let baseline=cuda_device_allocation_snapshot();
    let mut s=NerkyorResearchCudaSession::open(admitted).unwrap();
    record(out.join("target-info.json"),s.info());
    // Independent scalar prefill + scalar decode reference, using the same pinned weights.
    s.owner.target_for_test().enabled=false;
    let gold=turn(&mut s,&request("scalar-reference"),&out);
    let reference=state(&mut s);close(&mut s,&gold);
    s.owner.target_for_test().enabled=true;
    // A pre-admission flag is not authority to reset or "recover" another session.
    let early=SessionCancellationToken::new();early.request_cancel();
    assert!(s.turn(&request("early"),&early,|_|panic!("no output"),|_|panic!("no commit")).is_err());
    assert!(!s.poisoned());assert_eq!(s.submitted_tokens(),0);
    assert!(s.discard_cancelled_turn("early-discard","early","session-early",|_|panic!("not authorized")).is_err());
    let mut comparisons=Vec::new();
    for mode in ["prefill","decode","append"] {
        let mut r=request(&format!("cancel-{mode}"));
        if mode=="append" {
            let mut first=request("append-prefix");
            first.input=TurnInput::Open{prompt_token_ids:short.clone()};first.max_new_tokens=12;
            let previous=turn(&mut s,&first,&out);
            r.session_id=previous.session.session_id.clone();
            r.input=TurnInput::Append{revision:previous.session.revision,
                exact_history:previous.session.consumed_token_ids,suffix_token_ids:suffix.clone()};
        }
        let cancel=SessionCancellationToken::new();
        let worker=if mode=="prefill" {let flag=cancel.clone();Some(thread::spawn(move||{
            thread::sleep(Duration::from_millis(250));flag.request_cancel();
        }))}else{None};
        let mut emitted=Vec::new();let mut persisted=false;
        let error=s.turn(&r,&cancel,|event|{
            emitted.push(event.token_id);
            if mode!="prefill" && event.sequence==1 {cancel.request_cancel();}
            Ok(())
        },|_|{persisted=true;Err("unexpected cancellation commit".into())}).unwrap_err();
        if let Some(worker)=worker {worker.join().unwrap();}
        assert!(error.is::<crate::session_cancellation::SessionCancellationRequested>());
        assert!(!persisted);assert!(s.poisoned());assert!(s.snapshot().is_none());
        let consumed=s.submitted_tokens();
        if mode=="prefill" {assert!(emitted.is_empty());assert!(consumed>0&&consumed<prompt.len());assert_eq!((consumed-1)%8,0);}
        else {assert_eq!(emitted.len(),2);}
        for (rid,sid) in [("wrong-request",r.session_id.as_str()),(r.request_id.as_str(),"wrong-session")] {
            assert!(s.discard_cancelled_turn("bad-discard",rid,sid,|_|panic!("foreign identity")).is_err());
            assert!(s.poisoned());assert_eq!(s.submitted_tokens(),consumed);
        }
        let discard_id=format!("discard-{mode}");
        let committed=s.discard_cancelled_turn(&discard_id,&r.request_id,&r.session_id,|receipt|{
            let raw=serde_json::to_vec(receipt)?;record(out.join(format!("{discard_id}.json")),&serde_json::from_slice(&raw)?);
            Ok(session::digest(&raw))
        }).unwrap();
        assert_eq!(committed.receipt.schema,"nerkyor-session-cancellation-discard-v1");
        assert!(committed.receipt.drain_completed&&committed.receipt.reset_completed&&committed.receipt.session_discarded);
        assert!(!committed.receipt.previous_history_reusable);
        assert!(!s.poisoned());assert!(s.snapshot().is_none());assert_eq!(s.submitted_tokens(),0);
        assert!(s.discard_cancelled_turn("again",&r.request_id,&r.session_id,|_|panic!("one-shot")).is_err());
        // Old histories, even with correct IDs, are unusable after cancellation.
        let stale=TurnRequest{request_id:format!("stale-{mode}"),session_id:r.session_id.clone(),
            input:TurnInput::Append{revision:1,exact_history:vec![1],suffix_token_ids:suffix.clone()},max_new_tokens:4,selector_policy:None};
        assert!(s.turn(&stale,&SessionCancellationToken::new(),|_|panic!("no output"),|_|panic!("no commit")).is_err());
        assert!(!s.poisoned());assert_eq!(s.submitted_tokens(),0);
        let replay=turn(&mut s,&request(&format!("replay-{mode}")),&out);
        assert_eq!(replay.output_token_ids,gold.output_token_ids);assert_eq!(replay.stop_reason,gold.stop_reason);
        assert_eq!(replay.session.consumed_token_ids,gold.session.consumed_token_ids);
        assert_eq!(replay.session.pending_token,gold.session.pending_token);
        assert_eq!(replay.session.pending_token_published,gold.session.pending_token_published);
        let actual=state(&mut s);assert_eq!(actual.len(),reference.len());
        let metrics:Vec<_>=actual.iter().zip(&reference).map(|((name,a),(other,b))|{
            assert_eq!(name,other);json!({"tensor":name,"metric":metric(a,b)})
        }).collect();
        assert_eq!(metrics.len(),130);assert!(metrics.iter().all(|m|m["metric"]["pass"]==true));
        comparisons.push(json!({"mode":mode,"submitted_before_discard":consumed,"published_before_cancel":emitted.len(),
            "same_owner_fresh_replay":true,"exact_tokens_history_pending":true,"metrics":metrics}));
        close(&mut s,&replay);
    }
    // Reset success alone is insufficient: failed durable discard publication is owner-fatal.
    let r=request("failed-discard-receipt");let flag=SessionCancellationToken::new();
    assert!(s.turn(&r,&flag,|_|{flag.request_cancel();Ok(())},|_|panic!("no turn commit")).is_err());
    assert!(s.discard_cancelled_turn("failed-discard",&r.request_id,&r.session_id,|_|Err("injected discard persistence failure".into())).is_err());
    assert!(s.poisoned());assert!(s.snapshot().is_none());
    assert!(s.discard_cancelled_turn("retry-discard",&r.request_id,&r.session_id,|_|panic!("no retry")).is_err());
    assert!(s.turn(&request("forbidden-reuse"),&SessionCancellationToken::new(),|_|panic!("no reuse"),|_|panic!("no reuse")).is_err());
    s.retire().unwrap();drop(s);
    let freed=cuda_device_allocation_snapshot();
    assert_eq!(baseline.live_bytes,freed.live_bytes);assert_eq!(baseline.live_allocations,freed.live_allocations);
    record(out.join("result.json"),&json!({"status":"PASS_SAME_OWNER_CANCEL_DISCARD_FULL_STATE_FRESH_REPLAY",
        "comparisons":comparisons,"same_weight_owner_loads":1,"allocation_cleanup":true,"typed_cancellation_only":true,
        "bad_identity_stale_history_and_failed_receipt_fail_closed":true,"product_ready":false,"HTTP_qualified":false}));
}