//! Real target-only session lifecycle and execution-path checks.
use super::*;
use super::gpu_tests::{read, record, turn, close};
use crate::checkpoint_projection_cuda::projection_submission_counts;
use std::{fs, path::PathBuf};

#[test]
#[ignore = "exclusive SM70; manifest-pinned QPN reassoc session without draft allocation"]
fn no_dflash_session_uses_reassoc_and_preserves_pending_history() {
    let guard = read(std::env::var_os("AUGUR_NERKYOR_GPU_GUARD").unwrap());
    assert_eq!(guard["gpu_compute_apps_empty"], true);
    assert_eq!(guard["queues_paused"], true);
    let out = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_GPU_OUTPUT").unwrap());
    fs::create_dir(&out).unwrap();
    let root = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_MODEL").unwrap());
    let draft = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_DFLASH_Q8").unwrap());
    let path = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_SESSION_MANIFEST").unwrap());
    let pin = std::env::var("AUGUR_NERKYOR_SESSION_MANIFEST_SHA256").unwrap();
    let admitted = crate::nerkyor_session_manifest::load(&root, &draft, &path, &pin).unwrap();
    assert_eq!(admitted.manifest.execution_profile, crate::nerkyor_session_manifest::NO_DFLASH_EXECUTION);
    let tokenizer = tokenizers::Tokenizer::from_bytes(admitted.tokenizer_bytes()).unwrap();
    let text = "Reference: a stack is last in first out; a queue is first in first out. ".repeat(80);
    let prompt = tokenizer.encode(text, false).unwrap().get_ids()[..128].to_vec();
    let before = augur_backend::cuda_device_allocation_snapshot();
    let mut s = NerkyorResearchCudaSession::open(admitted).unwrap();
    assert_eq!(s.capacity(), 4096);
    assert_eq!(s.profile(), NO_DFLASH_PROFILE);
    assert!(!s.dflash_enabled());
    assert!(s.owner.target_for_test().decoder.is_none());
    assert!(s.owner.target_for_test().weights.is_none());
    assert_eq!(s.info()["policy"], CheckpointProjectionPolicy::NerkyorQpnSourceReassocResearch.label());
    let counts = projection_submission_counts();
    let bad = TurnRequest { selector_policy:None, request_id:"too-long".into(), session_id:"test".into(),
        max_new_tokens:2, input:TurnInput::Open {prompt_token_ids:vec![1;4095]} };
    assert!(s.turn(&bad, &SessionCancellationToken::new(), |_| panic!("unexpected token"), |_| panic!("unexpected receipt")).is_err());
    assert_eq!(projection_submission_counts(), counts);
    assert!(!s.poisoned());
    let request = TurnRequest { selector_policy:None, request_id:"open".into(), session_id:"test".into(),
        max_new_tokens:16, input:TurnInput::Open {prompt_token_ids:prompt.clone()} };
    let first = turn(&mut s, &request, &out);
    let actual = projection_submission_counts();
    let delta: Vec<_> = actual.iter().zip(counts).map(|(a,b)|a-b).collect();
    assert_eq!(delta, vec![0,400*first.session.consumed_token_ids.len() as u64,0,0]);
    assert_eq!(first.decode_detail.as_ref().unwrap()["decode_steps"], first.selected_tokens_including_eos-1);
    assert!(!first.dflash_enabled);
    let append = TurnRequest { selector_policy:None, request_id:"append".into(), session_id:"test".into(),
        max_new_tokens:8, input:TurnInput::Append {revision:first.session.revision,
            exact_history:first.session.consumed_token_ids.clone(), suffix_token_ids:vec![1,2,3]} };
    let second = turn(&mut s, &append, &out);
    assert_eq!(second.session.consumed_token_ids[first.session.consumed_token_ids.len()], first.session.pending_token);
    assert_eq!(second.decode_detail.as_ref().unwrap()["decode_steps"], second.selected_tokens_including_eos-1);
    close(&mut s, &second);
    let one = TurnRequest {selector_policy:None,request_id:"one".into(),session_id:"one".into(),max_new_tokens:1,
        input:TurnInput::Open {prompt_token_ids:prompt[..8].to_vec()}};
    let single = turn(&mut s, &one, &out);
    assert_eq!(single.decode_detail.as_ref().unwrap()["decode_steps"], 0);
    assert!(single.decode_detail.as_ref().unwrap()["decode_tokens_per_second"].is_null());
    close(&mut s, &single);
    s.retire().unwrap(); drop(s);
    let after = augur_backend::cuda_device_allocation_snapshot();
    assert_eq!((before.live_bytes,before.live_allocations),(after.live_bytes,after.live_allocations));
    record(out.join("result.json"), &json!({"status":"PASS_NO_DFLASH_SESSION_LIFECYCLE",
        "draft_loaded":false,"projection_submissions":delta,"pending_append_once":true,
        "over_budget_rejected_before_GPU":true,"first_prefill_token_excluded":true,"allocation_cleanup":true}));
}
