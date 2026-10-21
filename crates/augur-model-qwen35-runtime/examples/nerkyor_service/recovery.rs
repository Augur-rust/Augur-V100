//! Request-bound cancellation on the resident Nerkyor producer.
//! A successful turn receipt is irreversible: a late cancellation closes that
//! committed session. Only the library's typed unpublished cancellation grants
//! discard authority. Unknown errors, EOF and failed receipts remain owner-fatal.
use super::recovery_input::{Cancel, Input};
use super::*;
use session::CommittedTurn;

pub(super) const CONTRACT: &str = "bound_request_unpublished_discard_or_committed_close_v1";
enum Outcome {
    Committed(CommittedTurn),
    Unadmitted,
    Cancelled,
}
struct Last {
    rid: String,
    sid: String,
    outcome: Outcome,
}
fn close_proof(
    owner: &mut NerkyorResearchCudaSession,
    cancel: &Cancel,
    receipts: &Path,
) -> Result<Value> {
    let snap = owner
        .snapshot()
        .ok_or("cancel-close requires an active committed session")?
        .clone();
    if snap.session_id != cancel.session_id {
        return Err("cancel-close session identity mismatch".into());
    }
    let mut proof = None;
    let sha = owner.close(&cancel.request_id, &cancel.session_id, snap.revision, |r| {
        let sha = persist_receipt(receipts, &cancel.request_id, r)?;
        proof = Some(json!({"receipt":r,"receipt_sha256":sha}));
        Ok(sha)
    })?;
    let proof = proof.ok_or("missing persisted cancel-close proof")?;
    if proof["receipt_sha256"] != sha {
        return Err("cancel-close receipt digest mismatch".into());
    }
    Ok(proof)
}
fn resolve(
    owner: &mut NerkyorResearchCudaSession,
    input: &Input,
    cancel: &Cancel,
    outcome: Outcome,
    receipts: &Path,
    recoveries: &Path,
) -> Result<()> {
    let before = owner.submitted_tokens();
    let turn_path = receipts.join(format!("{}.json", cancel.cancelled_request_id));
    let (label, proof, committed_sha) = match outcome {
        Outcome::Cancelled => {
            if turn_path.exists() {
                return Err("cancelled turn has a committed or partial receipt".into());
            }
            let proof = owner.discard_cancelled_turn(
                &cancel.request_id,
                &cancel.cancelled_request_id,
                &cancel.session_id,
                |r| persist_receipt(receipts, &cancel.request_id, r),
            )?;
            ("unpublished_discarded", serde_json::to_value(proof)?, None)
        }
        Outcome::Committed(turn) => {
            if owner.snapshot() != Some(&turn.receipt.session)
                || turn.receipt.request_id != cancel.cancelled_request_id
                || turn.receipt.session.session_id != cancel.session_id
            {
                return Err("late cancellation committed binding mismatch".into());
            }
            let raw = fs::read(&turn_path)?;
            if session::digest(&raw) != turn.receipt_sha256
                || serde_json::from_slice::<Value>(&raw)? != serde_json::to_value(&turn.receipt)?
            {
                return Err("late cancellation requires the exact durable original receipt".into());
            }
            (
                "committed_session_closed",
                close_proof(owner, cancel, receipts)?,
                Some(turn.receipt_sha256),
            )
        }
        Outcome::Unadmitted => {
            if owner.poisoned() || turn_path.exists() {
                return Err(
                    "unadmitted cancellation has unknown or already committed state".into(),
                );
            }
            if owner.snapshot().is_some() {
                (
                    "unadmitted_session_closed",
                    close_proof(owner, cancel, receipts)?,
                    None,
                )
            } else {
                if owner.submitted_tokens() != 0 {
                    return Err("unadmitted cancellation requires an empty target".into());
                }
                ("unadmitted_empty", Value::Null, None)
            }
        }
    };
    if owner.poisoned() || owner.snapshot().is_some() || owner.submitted_tokens() != 0 {
        return Err("cancellation resolution failed empty-owner postcondition".into());
    }
    let receipt = json!({"schema":"nerkyor-request-cancel-resolution-v1","contract":CONTRACT,"profile":PROFILE,
        "request_id":cancel.request_id,"cancelled_request_id":cancel.cancelled_request_id,"session_id":cancel.session_id,
        "manifest_sha256":owner.info()["session_manifest_sha256"],"outcome":label,"cleanup_proof":proof,
        "original_turn_receipt_sha256":committed_sha,"submitted_tokens_before_resolution":before,
        "submitted_tokens_after_resolution":0,"active_session":false,"owner_reusable":true,
        "previous_history_reusable":false,"weights_reloaded":false,"product_ready":false});
    // Reset alone is not publication authority: failed durable acknowledgement
    // propagates to main(), which retires the owner and proves allocation closure.
    let sha = persist_receipt(recoveries, &cancel.request_id, &receipt)?;
    input.resolved(cancel)?;
    output(
        &json!({"event":"cancelled","request_id":cancel.cancelled_request_id,"session_id":cancel.session_id,
        "resolution":{"receipt":receipt,"receipt_sha256":sha},"owner_retired":false,"product_ready":false}),
    )
}
pub(super) fn serve(
    owner: &mut NerkyorResearchCudaSession,
    input: &Input,
    _lifetime: &SessionCancellationToken,
    receipts: &Path,
) -> Result<&'static str> {
    let recoveries = receipts
        .parent()
        .ok_or("missing evidence parent")?
        .join("cancellations");
    fs::create_dir(&recoveries)?;
    let mut last: Option<Last> = None;
    while let Some(work) = input.next()? {
        match parse_command(&work.raw)? {
            Command::Turn { request } => {
                let started = Instant::now();
                let result = owner.turn(
                    &request,
                    &work.token,
                    |t| output(&json!({"event":"token","token":t,"product_ready":false})),
                    |r| persist_receipt(receipts, &r.request_id, r),
                );
                match result {
                    Ok(turn) => {
                        let cancel = input.completed(&request.request_id, &request.session_id)?;
                        output(
                            &json!({"event":"terminal","request_id":request.request_id,"session_id":request.session_id,
                            "committed":turn,"wall_s_through_receipt":started.elapsed().as_secs_f64(),"product_ready":false}),
                        )?;
                        if let Some(cancel) = cancel {
                            resolve(
                                owner,
                                input,
                                &cancel,
                                Outcome::Committed(turn),
                                receipts,
                                &recoveries,
                            )?;
                            last = None;
                        } else {
                            last = Some(Last {
                                rid: request.request_id,
                                sid: request.session_id,
                                outcome: Outcome::Committed(turn),
                            });
                        }
                    }
                    Err(error) => {
                        let typed=error.is::<augur_model_qwen35_runtime::session_cancellation::SessionCancellationRequested>();
                        if (typed && work.token.is_requested()) || !owner.poisoned() {
                            let cancel =
                                input.completed(&request.request_id, &request.session_id)?;
                            if let Some(cancel) = cancel {
                                let outcome = if owner.poisoned() {
                                    Outcome::Cancelled
                                } else {
                                    Outcome::Unadmitted
                                };
                                resolve(owner, input, &cancel, outcome, receipts, &recoveries)?;
                                last = None;
                                continue;
                            }
                        }
                        output(
                            &json!({"event":if owner.poisoned(){"error"}else{"rejected"},"request_id":request.request_id,
                            "session_id":request.session_id,"owner_retired":owner.poisoned(),"state_unchanged":!owner.poisoned(),
                            "reason":error.to_string(),"product_ready":false}),
                        )?;
                        if owner.poisoned() {
                            return Err(error);
                        }
                        last = Some(Last {
                            rid: request.request_id,
                            sid: request.session_id,
                            outcome: Outcome::Unadmitted,
                        });
                    }
                }
            }
            Command::CancelTurn {
                request_id,
                cancelled_request_id,
                session_id,
            } => {
                let cancel = input.cancellation(&cancelled_request_id, &session_id)?;
                if cancel.request_id != request_id {
                    return Err("late cancel control mismatch".into());
                }
                let previous = last
                    .take()
                    .ok_or("late cancellation has no completed turn")?;
                if previous.rid != cancelled_request_id || previous.sid != session_id {
                    return Err("late cancellation belongs to another turn".into());
                }
                resolve(
                    owner,
                    input,
                    &cancel,
                    previous.outcome,
                    receipts,
                    &recoveries,
                )?;
            }
            Command::Close {
                request_id,
                session_id,
                revision,
            } => {
                last = None;
                match owner.close(&request_id, &session_id, revision, |r| {
                    persist_receipt(receipts, &request_id, r)
                }) {
                    Ok(sha) => output(
                        &json!({"event":"closed","request_id":request_id,"session_id":session_id,"revision":revision,
                        "receipt_sha256":sha,"product_ready":false}),
                    )?,
                    Err(error) => {
                        output(
                            &json!({"event":if owner.poisoned(){"error"}else{"rejected"},"request_id":request_id,"session_id":session_id,
                            "owner_retired":owner.poisoned(),"state_unchanged":!owner.poisoned(),"reason":error.to_string(),"product_ready":false}),
                        )?;
                        if owner.poisoned() {
                            return Err(error);
                        }
                    }
                }
            }
            Command::Shutdown {} => return Ok("shutdown"),
        }
    }
    Ok("input_eof")
}
