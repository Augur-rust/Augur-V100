use super::*;
use std::{fs, path::PathBuf};
#[test]
fn adaptive_dflash_gate_waits_for_eight_rounds_and_uses_checked_acceptance() {
    assert!(!adaptive_gate_should_disable(7, 0, 7));
    assert!(!adaptive_gate_should_disable(8, 8, 8));
    assert!(!adaptive_gate_should_disable(8, 4, 5));
    assert!(adaptive_gate_should_disable(8, 7, 10));
    assert!(!adaptive_gate_should_disable(9, 8, 10));
}
#[test]
fn prefill_memory_admission_uses_current_b4352_recipe_and_budget() -> Result<()> {
    let required=crate::checkpoint_target_cuda::augur_prefill::default_required_free_bytes;
    assert!(required(0).is_err());assert!(required(CAPACITY+1).is_err());
    assert!(required(8)?<required(4352)?);assert_eq!(required(4352)?,required(8241)?);
    assert_eq!(required(4352)?,1_561_264_128+134_217_728+368_640);
    use crate::checkpoint_full_cuda::augur_prefill::AttentionMode;
    let planar = |n| crate::checkpoint_target_cuda::augur_prefill::default_required_free_bytes_mode(n, AttentionMode::Gqa8Planar);
    assert!(planar(0).is_err());
    assert_eq!(planar(4352)?, required(4352)?);
    assert_eq!(planar(4359)?, required(4352)?); // wide allocation covers a seven-row tail
    assert_eq!(planar(8)? - required(8)?, (96 * 64 * 130 * 8) as u64);
    Ok(())
}
fn logits(owner:&mut AugurCudaSession)->Result<String> {
    let values=owner.owner.target_for_test().target.logits_snapshot()?;
    let bytes:Vec<_>=values.iter().flat_map(|x|x.to_bits().to_le_bytes()).collect();Ok(digest(&bytes))
}
fn full_state(owner:&mut AugurCudaSession,tokens:&[u32])->Result<Option<String>> {
    if std::env::var("AUGUR_LIFECYCLE_STATE_DIGEST").as_deref()!=Ok("1") {return Ok(None);}
    use crate::augur_session::checkpoint::{Bindings,Boundary,Descriptor,SCHEMA};
    let bindings=Bindings {
        producer_sha256:digest(&fs::read(std::env::current_exe()?)?),
        manifest_sha256:std::env::var("AUGUR_MANIFEST_SHA256")?,
        profile:DFLASH_B8_PROFILE.into(),
        draft_config_sha256:std::env::var("AUGUR_DRAFT_CONFIG_SHA256")?,
        draft_weights_sha256:std::env::var("AUGUR_DRAFT_SHA256")?,
        decode_attention:std::env::var("AUGUR_HTTP_DECODE_ATTENTION")?,sampling:crate::augur_sampling::GREEDY.into(),sampling_seed:0,
    };
    let descriptor=Descriptor {schema:SCHEMA.into(),bindings:bindings.clone(),boundary:Boundary::InputEnd{token_ids:tokens.to_vec()}};
    descriptor.validate(&bindings)?;
    Ok(Some(owner.owner.target_for_test().target.checkpoint_test_state_sha256(&descriptor)?))
}
#[test]
#[ignore="exclusive admitted Augur SM70 GPU; active batched8/park policy and fresh evidence directory"]
fn parked_b8_reset_cancellation_and_exact_logits_replay() -> Result<()> {
    if !requested_dflash_enabled()? || !requested_dflash_plan_cache()? {return Err("requires batched8+park".into());}
    let root=PathBuf::from(std::env::var("AUGUR_MODEL_ROOT")?);
    let manifest=PathBuf::from(std::env::var("AUGUR_MANIFEST")?);
    let hash=std::env::var("AUGUR_MANIFEST_SHA256")?;
    let evidence=PathBuf::from(std::env::var("AUGUR_PLAN_CACHE_EVIDENCE")?);fs::create_dir(&evidence)?;
    let before=augur_backend::cuda_device_allocation_snapshot();
    if before.live_bytes!=0 || before.live_allocations!=0 {return Err("requires empty native allocation scope".into());}
    let mut owner=AugurCudaSession::open(session::manifest::load(&root,&manifest,&hash)?)?;
    persist_receipt(&evidence,"target-info",owner.info())?;
    let ids:Vec<u32>=serde_json::from_str(&std::env::var("AUGUR_CANCEL_DISCARD_PROMPT_IDS")?)?;
    let request=|rid:&str|TurnRequest {selector_policy: None, request_id:rid.into(),session_id:format!("{rid}-session"),max_new_tokens:32,
        input:TurnInput::Open {prompt_token_ids:ids.clone()}};
    let run=|o:&mut AugurCudaSession,r:&TurnRequest|o.turn(r,&SessionCancellationToken::new(),|_|Ok(()),|v|persist_receipt(&evidence,&v.request_id,v));
    let reference=run(&mut owner,&request("reference"))?;let reference_logits=logits(&mut owner)?;
    let reference_state=full_state(&mut owner,&reference.receipt.session.consumed_token_ids)?;
    let detail=reference.receipt.decode_detail.as_ref().ok_or("missing detail")?;
    if detail["dflash"]["plan_builds"]!=1 {return Err("expected first plan build".into());}
    owner.close("close-reference","reference-session",1,|v|persist_receipt(&evidence,&v.request_id,v))?;
    let t=owner.owner.target_for_test();let parked=t.draft.audit_parked_no_cache_lease(&mut t.target)?;
    persist_receipt(&evidence,"parked-invariants",&parked)?;
    let again=run(&mut owner,&request("again"))?;let again_logits=logits(&mut owner)?;
    let again_state=full_state(&mut owner,&again.receipt.session.consumed_token_ids)?;
    if again_state!=reference_state {return Err("parked reset full KV/GDN/capture state changed".into());}
    if again.receipt.output_token_ids!=reference.receipt.output_token_ids || again_logits!=reference_logits
        || again.receipt.session.consumed_token_ids!=reference.receipt.session.consumed_token_ids
        || again.receipt.session.pending_token!=reference.receipt.session.pending_token
        || again.receipt.decode_detail.as_ref().unwrap()["dflash"]["plan_builds"]!=0
        || again.receipt.decode_detail.as_ref().unwrap()["dflash"]["plan_resumes"]!=1 {return Err("parked reset replay changed output/logits or rebuilt".into());}
    owner.close("close-again","again-session",1,|v|persist_receipt(&evidence,&v.request_id,v))?;
    let cancel=SessionCancellationToken::new();let mut emitted=0;
    let error=owner.turn(&request("interrupted"),&cancel,|_|{emitted+=1;if emitted==2{cancel.request_cancel();}Ok(())},
        |_|Err("cancelled turn must never persist success".into())).unwrap_err();
    let ahead=owner.submitted_tokens();
    if !error.is::<crate::session_cancellation::SessionCancellationRequested>() || emitted!=2 || ahead<=ids.len()+emitted {
        return Err("probe did not cancel while publishing an already-committed B8 block".into());
    }
    let discarded=owner.discard_cancelled_turn("discard","interrupted","interrupted-session",|v|persist_receipt(&evidence,&v.request_id,v))?;
    let t=owner.owner.target_for_test();t.draft.audit_parked_no_cache_lease(&mut t.target)?;
    let after=run(&mut owner,&request("after"))?;let after_logits=logits(&mut owner)?;
    let after_state=full_state(&mut owner,&after.receipt.session.consumed_token_ids)?;
    if after_state!=reference_state {return Err("post-cancel full KV/GDN/capture state changed".into());}
    if after.receipt.output_token_ids!=reference.receipt.output_token_ids || after_logits!=reference_logits
        || after.receipt.session.consumed_token_ids!=reference.receipt.session.consumed_token_ids
        || after.receipt.session.pending_token!=reference.receipt.session.pending_token
        || after.receipt.decode_detail.as_ref().unwrap()["dflash"]["plan_resumes"]!=1 {return Err("post-cancel parked replay drift".into());}
    // Negative test only after positive ledger/replay checks. The attempted raw
    // reset may partially clear layers: retire this owner, never reuse its ledger.
    let t=owner.owner.target_for_test();let error=t.target.reset_empty().expect_err("active plan must retain exclusive-reset guard");
    if !error.to_string().contains("cache owner unexpectedly shared") {return Err(error);}
    t.draft.release_batched_verifier()?;t.target.reset_empty()?;owner.retire()?;drop(owner);
    let after_alloc=augur_backend::cuda_device_allocation_snapshot();
    if after_alloc.live_bytes!=0 || after_alloc.live_allocations!=0 {return Err("parked plan cleanup leaked allocations".into());}
    let result=json!({"status":"PASS","dflash_active":true,"plan_cache":"park","reference_logits_sha256":reference_logits,
        "again_logits_sha256":again_logits,"after_cancel_logits_sha256":after_logits,"bitwise_logits_equal":true,
        "reference_full_state_sha256":reference_state,"again_full_state_sha256":again_state,"after_cancel_full_state_sha256":after_state,
        "negative_active_lease_reset_guard":"rejected","parked_invariants":parked,"cancel_emitted":emitted,
        "submitted_at_cancel":ahead,"prompt_tokens":ids.len(),"discard_receipt_sha256":discarded.receipt_sha256,
        "live_bytes_after":0,"live_allocations_after":0,"product_ready":false,"B1_numerical_equivalence_qualified":false});
    persist_receipt(&evidence,"complete",&result)?;println!("{result}");Ok(())
}

#[test]
fn gate_diagnostics_preserve_every_legacy_decision() {
    // Observation must never alter admission, including the equality boundary
    // and a tail with no checked candidates.
    for rounds in 0..=20 {
        for checked in 0..=7*rounds {
            for accepted in 0..=checked {
                let mut stats=Stats {rounds,accepted,checked,..Stats::default()};
                assert_eq!(stats.observe_gate(8192),adaptive_gate_should_disable(rounds,accepted,checked));
                assert_eq!(stats.gate_evaluations,usize::from(rounds>=8 && checked>0));
                assert_eq!(stats.first_gate.is_some(),rounds>=8 && checked>0);
            }
        }
    }
}
#[test]
fn gate_diagnostics_keep_first_sample_and_actual_later_disable_sample() {
    let mut stats=Stats {rounds:7,accepted:28,checked:35,..Stats::default()};
    assert!(!stats.observe_gate(8227));
    assert!(stats.first_gate.is_none());
    stats.rounds=8;stats.accepted=32;stats.checked=40;
    stats.setup_s=0.01;stats.proposal_s=0.2;stats.verify_s=0.4;
    assert!(!stats.observe_gate(8232)); // equality is admitted by the existing gate
    let first=stats.first_gate.unwrap();
    stats.rounds=9;stats.checked=41;stats.proposal_s+=0.03;stats.verify_s+=0.04;
    assert!(stats.observe_gate(8233));
    let disabled=stats.disabled_gate.unwrap();
    assert_eq!(stats.gate_evaluations,2);
    assert_eq!(stats.first_gate.unwrap().rounds,8);
    assert_eq!(first.accepted,32);assert_eq!(disabled.accepted,32);
    assert_eq!(disabled.rounds,9);assert_eq!(disabled.checked,41);
    assert_eq!(disabled.target_consumed,8233);
    assert!((disabled.proposal_s-first.proposal_s-0.03).abs()<1e-12);
    let detail=stats.gate_detail();
    assert_eq!(detail["first_evaluation"]["completed_B8_block_tokens_including_anchors"],40);
    assert_eq!(detail["disable_evaluation"]["completed_B8_block_tokens_including_anchors"],41);
    assert_eq!(detail["disable_evaluation"]["not_whole_decode_or_HTTP_wall"],true);
    assert_eq!(detail["disable_evaluation"]["native_B1_counterfactual_measured"],false);
    assert_eq!(detail["decision_policy_unchanged"],true);
    stats=Stats::default(); // the same reset used for a new request/cache restore
    let reset=stats.gate_detail();
    assert_eq!(reset["eligible_evaluations"],0);
    assert!(reset["first_evaluation"].is_null() && reset["disable_evaluation"].is_null());
}
#[test]
fn gate_diagnostics_can_disable_on_first_eligible_evaluation() {
    let mut stats=Stats {rounds:8,accepted:7,checked:10,..Stats::default()};
    assert!(stats.observe_gate(32768));
    let detail=stats.gate_detail();
    assert_eq!(detail["first_evaluation"],detail["disable_evaluation"]);
    assert_eq!(detail["eligible_evaluations"],1);
    assert_eq!(detail["first_evaluation"]["proposed_draft_tokens"],56);
    assert_eq!(detail["first_evaluation"]["accepted_draft_tokens"],7);
    assert_eq!(detail["first_evaluation"]["checked_draft_tokens"],10);
}
#[test]
fn gate_diagnostics_do_not_invent_uncharged_measurements() {
    let empty=GateSnapshot::from_stats(&Stats::default(),0).detail();
    assert!(empty["accepted_over_checked"].is_null());
    assert!(empty["component_s_per_completed_B8_block_token"].is_null());
    let mut stats=Stats::default();
    stats.speculation_disabled=true;stats.disable_reason=Some("vision");
    let detail=stats.gate_detail();
    assert!(detail["first_evaluation"].is_null() && detail["disable_evaluation"].is_null());
    assert_eq!(detail["eligible_evaluations"],0);
}

#[test]
fn fixed_b8_is_default_and_adaptive_policies_require_opt_in() {
    for active in [false,true] {
        assert_eq!(parse_cost_gate(None,active).unwrap(),if active{CostPolicy::FixedB8}else{CostPolicy::Legacy});
        assert_eq!(parse_cost_gate(Some("legacy"),active).unwrap(),CostPolicy::Legacy);
        for value in ["", "cost", "COST_V1", "cost_v1 ", "0.8"] {
            assert!(parse_cost_gate(Some(value),active).is_err());
        }
    }
    assert_eq!(parse_cost_gate(Some("fixed_b8"),true).unwrap(),CostPolicy::FixedB8);
    assert!(parse_cost_gate(Some("fixed_b8"),false).is_err());
    assert_eq!(parse_cost_gate(Some("cost_v1"),true).unwrap(),CostPolicy::Immediate);
    assert_eq!(parse_cost_gate(Some("cost_deferred_v2"),true).unwrap(),CostPolicy::Deferred);
    assert!(parse_cost_gate(Some("cost_deferred_v2"),false).is_err());
    assert!(parse_cost_gate(Some("cost_v1"),false).is_err());
}
#[test]
fn cost_receipt_distinguishes_enqueued_completed_and_terminal_work() {
    use std::time::Duration;
    let t=Instant::now();let mut runtime=cost_runtime::Runtime::default();
    runtime.selected(8192,t).unwrap();
    runtime.enqueued(cost_gate::Lane::B1,8193).unwrap();
    let open=cost_detail(&runtime);
    assert_eq!(open["all_completed_B1"]["completed_transactions"],0);
    assert_eq!(open["interval_open"],true);
    runtime.selected(8193,t+Duration::from_millis(30)).unwrap();
    runtime.enqueued(cost_gate::Lane::B1,8194).unwrap();
    runtime.drained(8194,t+Duration::from_millis(70)).unwrap();
    let closed=cost_detail(&runtime);
    assert_eq!(closed["policy"],"measured_decode_cost_v1");
    assert_eq!(closed["all_completed_B1"]["wall_ns"],30_000_000);
    assert_eq!(closed["all_completed_B1"]["consumed_tokens"],1);
    assert_eq!(closed["terminal_incomplete_intervals"]["wall_ns"],40_000_000);
    assert_eq!(closed["terminal_incomplete_intervals"]["lane"],"B1");
    assert_eq!(closed["terminal_incomplete_intervals"]["used_for_decisions"],false);
    assert_eq!(closed["start_consumed_tokens"],8192);
    assert_eq!(closed["terminal_consumed_tokens"],8194);
    assert_eq!(closed["request_drained"],true);
    assert_eq!(closed["interval_open"],false);
    assert_eq!(closed["extra_GPU_synchronization"],false);
    assert!(closed["decisions"].as_array().unwrap().is_empty());
}
#[test]
fn cost_receipt_keeps_actual_probe_counters_and_reset_discards_them() {
    use std::time::Duration;
    let t=Instant::now();let mut runtime=cost_runtime::Runtime::default();
    runtime.selected(8192,t).unwrap();
    for n in 0..16 {
        runtime.enqueued(cost_gate::Lane::B1,8193+n).unwrap();
        runtime.selected(8193+n,t+Duration::from_millis(30*(n as u64+1))).unwrap();
    }
    for n in 0..8 {
        runtime.enqueued(cost_gate::Lane::B8,8210+2*n).unwrap();
        runtime.selected(8210+2*n,t+Duration::from_millis(480+40*(n as u64+1))).unwrap();
    }
    runtime.drained(8224,t+Duration::from_millis(802)).unwrap();
    let receipt=cost_detail(&runtime);let decision=&receipt["decisions"][0];
    assert_eq!(decision["selected"],"B8");
    assert_eq!(decision["B1"]["consumed_tokens"],16);
    assert_eq!(decision["B8"]["consumed_tokens"],16);
    assert_eq!(decision["B1"]["wall_ns"],480_000_000);
    assert_eq!(decision["B8"]["wall_ns"],320_000_000);
    assert_eq!(receipt["terminal_incomplete_intervals"]["selection_without_consume_wall_ns"],2_000_000);
    runtime.reset();let empty=cost_detail(&runtime);
    assert_eq!(empty["all_completed_B1"]["consumed_tokens"],0);
    assert!(empty["start_consumed_tokens"].is_null());
    assert!(empty["decisions"].as_array().unwrap().is_empty());
}
