use super::*;
use crate::nerkyor_session_manifest;
use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, CudaComputeCapability,
};
use std::{
    fs,
    path::{Path, PathBuf},
};
pub(super) fn read(p: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&fs::read(p).unwrap()).unwrap()
}
pub(super) fn record(p: impl AsRef<Path>, v: &Value) {
    use std::io::Write;
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(p)
        .unwrap();
    f.write_all(&serde_json::to_vec_pretty(v).unwrap()).unwrap();
}
pub(super) fn state(s: &mut NerkyorResearchCudaSession) -> Vec<(String,Vec<f32>)> {
    s.owner.target_for_test().target.nerkyor_all_state_for_test().unwrap()
}

pub(super) fn metric(a: &[f32], b: &[f32]) -> Value {
    assert_eq!(a.len(), b.len());
    let (mut error, mut norm, mut max, mut peak) = (0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in a.iter().zip(b) {
        assert!(a.is_finite() && b.is_finite());
        let (a, b) = (f64::from(a), f64::from(b));
        error += (a - b).powi(2);
        norm += b * b;
        max = max.max((a - b).abs());
        peak = peak.max(b.abs());
    }
    let rel = if norm == 0.0 {
        if error == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (error / norm).sqrt()
    };
    json!({"elements":a.len(),"relative_l2":rel,"max_abs":max,"reference_peak":peak,"bitwise_equal":a.iter().zip(b).all(|(a,b)|a.to_bits()==b.to_bits()),"pass":rel<=1e-5&&max<=1e-4+1e-5*peak})
}
pub(super) fn turn(s: &mut NerkyorResearchCudaSession, r: &TurnRequest, out: &Path) -> TurnReceipt {
    let mut events = Vec::new();
    let result = s
        .turn(
            r,
            &SessionCancellationToken::new(),
            |e| {
                events.push(e.token_id);
                Ok(())
            },
            |receipt| {
                let raw = serde_json::to_vec(receipt)?;
                record(
                    out.join(format!("{}.json", r.request_id)),
                    &serde_json::from_slice(&raw)?,
                );
                Ok(session::digest(&raw))
            },
        )
        .unwrap();
    assert_eq!(events, result.receipt.output_token_ids);
    assert_eq!(result.receipt.schema, "nerkyor-token-turn-receipt-v1");
    assert_eq!(result.receipt.session.profile, s.owner.target_for_test().context.profile());
    assert!(!result.receipt.product_ready);
    assert_eq!(
        s.submitted_tokens(),
        result.receipt.session.consumed_token_ids.len()
    );
    assert_eq!(
        s.owner
            .target_for_test()
            .target
            .select_next_token()
            .unwrap(),
        result.receipt.session.pending_token
    );
    result.receipt
}
pub(super) fn close(s: &mut NerkyorResearchCudaSession, r: &TurnReceipt) {
    s.close(
        &format!("close-{}", r.request_id),
        &r.session.session_id,
        r.session.revision,
        |c| {
            assert_eq!(c.schema, "nerkyor-session-close-v1");
            Ok(session::digest(&serde_json::to_vec(c)?))
        },
    )
    .unwrap();
    assert_eq!(s.submitted_tokens(), 0);
    assert!(s.snapshot().is_none());
}
#[test]
#[ignore = "exclusive own-manifest Nerkyor GPU token-session lifecycle and scalar replay"]
fn real_own_manifest_sessions_append_eos_budget_reset_cancel_and_scalar_oracle() {
    let guard = read(std::env::var_os("AUGUR_NERKYOR_GPU_GUARD").unwrap());
    assert_eq!(guard["gpu_compute_apps_empty"], true);
    assert_eq!(guard["queues_paused"], true);
    assert_eq!(guard["benchmark_auto_resume"], false);
    assert_eq!(
        cuda_device_properties(0).unwrap().compute_capability,
        CudaComputeCapability::SM70
    );
    let out = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_GPU_OUTPUT").unwrap());
    fs::create_dir(&out).unwrap();
    let root = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_MODEL").unwrap());
    let draft = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_DFLASH_Q8").unwrap());
    let path = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_SESSION_MANIFEST").unwrap());
    let pin = std::env::var("AUGUR_NERKYOR_SESSION_MANIFEST_SHA256").unwrap();
    assert!(nerkyor_session_manifest::load(&root, &draft, &path, &"0".repeat(64)).is_err());
    let admitted = nerkyor_session_manifest::load(&root, &draft, &path, &pin).unwrap();
    let tokenizer = tokenizers::Tokenizer::from_bytes(admitted.tokenizer_bytes()).unwrap();
    let encode = |text: String| tokenizer.encode(text, false).unwrap().get_ids().to_vec();
    let prompts = [
        "Count from 1 to 20, separated by spaces. Output only the numbers.",
        "用中文简短说明：栈和队列的主要区别是什么？",
        "What is 17 * 23? Output only the answer.",
    ];
    let mut prompts: Vec<_> = prompts
        .iter()
        .map(|q| {
            encode(format!(
                "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            ))
        })
        .collect();
    // Longer complete prefill plus actual generation, not only teacher forcing.
    for notes in ["A stack is last in first out; a queue is first in first out. ".repeat(16),
                  "栈是后进先出，队列是先进先出。这里是上下文验证资料。".repeat(12)] {
        let text=format!("Read these notes: {notes} Now output only the number 42.");
        let ids=encode(format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert!(ids.len()>200 && ids.len()+24<CAPACITY, "long prompt has {} tokens", ids.len());
        prompts.push(ids);
    }
    let suffix=encode("<|im_end|>\n<|im_start|>user\nNow output only 42.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".into());
    let baseline = cuda_device_allocation_snapshot();
    let mut s = NerkyorResearchCudaSession::open(admitted).unwrap();
    record(out.join("target-info.json"), s.info());
    let mut reference: Vec<(TurnReceipt, Vec<(String,Vec<f32>)>)> = Vec::new();
    let mut comparisons = Vec::new();
    let mut negative_checked = false;
    for mode in ["dflash", "scalar"] {
        s.owner.target_for_test().enabled = mode == "dflash";
        for (p, prompt) in prompts.iter().enumerate() {
            let request = TurnRequest {
                request_id: format!("{mode}-{p}"),
                session_id: format!("session-{mode}-{p}"),
                input: TurnInput::Open {
                    prompt_token_ids: prompt.clone(),
                },
                max_new_tokens: if p == 2 { 32 } else { 24 },
                selector_policy: None,
            };
            if !negative_checked {
                let alloc = cuda_device_allocation_snapshot().successful_allocations;
                let mut bad = request.clone();
                bad.max_new_tokens = CAPACITY;
                assert!(s
                    .turn(
                        &bad,
                        &SessionCancellationToken::new(),
                        |_| Ok(()),
                        |_| Err("not reached".into())
                    )
                    .is_err());
                bad.max_new_tokens = 24;
                bad.input = TurnInput::Open {
                    prompt_token_ids: vec![1; CAPACITY - 16],
                };
                assert!(s
                    .turn(
                        &bad,
                        &SessionCancellationToken::new(),
                        |_| Ok(()),
                        |_| Err("not reached".into())
                    )
                    .is_err());
                bad = request.clone();
                bad.selector_policy = Some(selector::Policy::GlobalGate045V1);
                assert!(s
                    .turn(
                        &bad,
                        &SessionCancellationToken::new(),
                        |_| Ok(()),
                        |_| Err("not reached".into())
                    )
                    .is_err());
                assert_eq!(
                    alloc,
                    cuda_device_allocation_snapshot().successful_allocations
                );
                assert_eq!(s.submitted_tokens(), 0);
                assert!(!s.poisoned());
                negative_checked = true;
            }
            let mut r = turn(&mut s, &request, &out);
            let mut rows = vec![(r.clone(), state(&mut s))];
            if p == 0 {
                let snap = r.session.clone();
                let mut append = TurnRequest {
                    request_id: format!("{mode}-append"),
                    session_id: snap.session_id.clone(),
                    max_new_tokens: 16,
                    selector_policy: None,
                    input: TurnInput::Append {
                        revision: snap.revision,
                        exact_history: snap.consumed_token_ids.clone(),
                        suffix_token_ids: suffix.clone(),
                    },
                };
                if let TurnInput::Append { exact_history, .. } = &mut append.input {
                    exact_history[0] ^= 1;
                }
                let cursor = s.submitted_tokens();
                assert!(s
                    .turn(
                        &append,
                        &SessionCancellationToken::new(),
                        |_| Ok(()),
                        |_| Err("not reached".into())
                    )
                    .is_err());
                assert_eq!(cursor, s.submitted_tokens());
                assert!(!s.poisoned());
                if let TurnInput::Append { exact_history, .. } = &mut append.input {
                    *exact_history = snap.consumed_token_ids;
                }
                r = turn(&mut s, &append, &out);
                rows.push((r.clone(), state(&mut s)));
            }
            if p == 2 {
                assert_eq!(r.stop_reason, "eos");
                assert!(!r.session.pending_token_published);
            }
            close(&mut s, &r);
            for (row, planes) in rows {
                if mode == "dflash" {
                    reference.push((row, planes));
                    continue;
                }
                let index = comparisons.len();
                let (gold, old) = &reference[index];
                assert_eq!(row.output_token_ids, gold.output_token_ids);
                assert_eq!(row.stop_reason, gold.stop_reason);
                assert_eq!(
                    row.session.consumed_token_ids,
                    gold.session.consumed_token_ids
                );
                assert_eq!(row.session.pending_token, gold.session.pending_token);
                assert_eq!(
                    row.session.pending_token_published,
                    gold.session.pending_token_published
                );
                let m: Vec<_> = old.iter().zip(&planes).map(|((name,a),(other,b))| {
                    assert_eq!(name,other);json!({"tensor":name,"metric":metric(a,b)})
                }).collect();
                assert!(m.iter().all(|x| x["metric"]["pass"] == true));
                assert_eq!(m.len(),130);
                comparisons.push(json!({"case":index,"tokens":row.output_token_ids.len(),"stop":row.stop_reason,
                    "full_logits_capture_KV_GDN_checked":true,"metrics":m,"exact_history_and_pending_match":true}));
            }
        }
    }
    s.owner.target_for_test().enabled = true;
    let cancel = SessionCancellationToken::new();
    let mut emitted = Vec::new();
    let mut persisted = false;
    let request = TurnRequest {
        request_id: "cancel-after-batch".into(),
        session_id: "cancel-session".into(),
        input: TurnInput::Open {
            prompt_token_ids: prompts[0].clone(),
        },
        max_new_tokens: 24,
        selector_policy: None,
    };
    assert!(s
        .turn(
            &request,
            &cancel,
            |event| {
                emitted.push(event.token_id);
                if event.sequence == 1 {
                    cancel.request_cancel();
                }
                Ok(())
            },
            |_| {
                persisted = true;
                Ok("a".repeat(64))
            }
        )
        .is_err());
    assert_eq!(emitted.len(), 2);
    assert!(!persisted);
    assert!(s.poisoned());
    assert!(s.snapshot().is_none());
    assert!(s.owner.target_for_test().rounds > 0);
    let after_cancel = s.submitted_tokens();
    assert!(after_cancel > prompts[0].len());
    assert!(s
        .turn(
            &request,
            &SessionCancellationToken::new(),
            |_| Ok(()),
            |_| Ok("a".repeat(64))
        )
        .is_err());
    s.retire().unwrap();
    drop(s);
    let freed = cuda_device_allocation_snapshot();
    assert_eq!(baseline.live_bytes, freed.live_bytes);
    assert_eq!(baseline.live_allocations, freed.live_allocations);
    record(
        out.join("result.json"),
        &json!({"status":"PASS_OWN_MANIFEST_BOUNDED_SESSION_APPEND_EOS_CANCEL_SCALAR_ORACLE","comparisons":comparisons,"capacity":CAPACITY,"own_schema_and_profile":true,"foreign_selector_and_oversized_request_cpu_rejected":true,"stale_history_rejected":true,"cancel_after_batched_commit_no_receipt":true,"cancel_published_tokens":emitted.len(),"cancel_submitted_tokens":after_cancel,"allocation_cleanup":true,"product_admitted":false,"HTTP_qualified":false,"sampling_qualified":false,"long_context_qualified":false}),
    );
}
