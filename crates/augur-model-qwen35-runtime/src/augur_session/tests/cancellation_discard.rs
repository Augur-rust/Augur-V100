use super::*;

fn cancel_after_emit(owner: &mut Session<Fake>, request: &TurnRequest) -> SessionCancellationToken {
    let token = SessionCancellationToken::new();
    let error = owner.turn(request, &token, |_| { token.request_cancel(); Ok(()) }, persist).unwrap_err();
    assert!(error.is::<crate::session_cancellation::SessionCancellationRequested>());
    assert!(owner.poisoned());
    assert!(owner.snapshot().is_none());
    token
}
fn discard(owner: &mut Session<Fake>, cancelled: &TurnRequest) -> Result<CommittedCancellationDiscard> {
    owner.discard_cancelled_turn("discard", &cancelled.request_id, &cancelled.session_id,
        |receipt| Ok(digest(&serde_json::to_vec(receipt)?)))
}

#[test]
fn typed_cancel_discard_reopens_only_fresh_session_and_matches_fresh_replay() {
    let (mut o, state) = owner();
    let r = open();
    let old_token = cancel_after_emit(&mut o, &r);
    let reset = discard(&mut o, &r).unwrap();
    assert_eq!(reset.receipt.cancelled_request_id, "r1");
    assert_eq!(reset.receipt.session_id, "s1");
    assert_eq!(reset.receipt.submitted_tokens_before_discard, 2);
    assert_eq!(reset.receipt.submitted_tokens_after_discard, 0);
    assert!(reset.receipt.reset_completed && reset.receipt.drain_completed && reset.receipt.session_discarded);
    assert!(!reset.receipt.previous_history_reusable);
    assert!(!o.poisoned());
    assert!(o.snapshot().is_none());
    assert!(state.borrow().ids.is_empty());
    assert_eq!(state.borrow().selected, None);
    assert!(old_token.is_requested()); // The token itself is never reset/reused.
    assert!(o.turn(&r, &SessionCancellationToken::new(), |_| Ok(()), persist).is_err());
    let mut fresh = open(); fresh.request_id = "fresh-r".into(); fresh.session_id = "fresh-s".into();
    let actual = run(&mut o, &fresh);
    let (mut baseline, _) = owner(); let expected = run(&mut baseline, &fresh);
    assert_eq!(actual.receipt.output_token_ids, expected.receipt.output_token_ids);
    assert_eq!(actual.receipt.session, expected.receipt.session);
}

#[test]
fn append_cancellation_does_not_restore_old_history_or_pending() {
    let (mut o, state) = owner();
    let previous = run(&mut o, &open());
    let r = append(&previous.receipt.session);
    cancel_after_emit(&mut o, &r);
    discard(&mut o, &r).unwrap();
    assert!(state.borrow().ids.is_empty());
    assert!(o.turn(&r, &SessionCancellationToken::new(), |_| Ok(()), persist).is_err());
    assert!(o.snapshot().is_none());
}

#[test]
fn wrong_discard_identity_and_reused_ids_leave_cancelled_state_poisoned() {
    let (mut o, state) = owner(); let r = open(); cancel_after_emit(&mut o, &r);
    let calls = state.borrow().calls;
    for (rid, cancelled, sid) in [("../escape", "r1", "s1"), ("r1", "r1", "s1"),
        ("d1", "foreign-r", "s1"), ("d2", "r1", "foreign-s")] {
        assert!(o.discard_cancelled_turn(rid, cancelled, sid, |_| panic!("no persistence on rejection")).is_err());
    }
    assert_eq!(state.borrow().calls, calls); assert!(o.poisoned());
    discard(&mut o, &r).unwrap();
    assert!(discard(&mut o, &r).is_err());
}

#[test]
fn cancellation_flag_or_matching_error_string_is_not_recovery_authority() {
    let (mut o, _) = owner(); let r = open(); let token = SessionCancellationToken::new();
    o.turn(&r, &token, |_| { token.request_cancel(); Err("session cancellation requested at fake boundary".into()) }, persist).unwrap_err();
    assert!(discard(&mut o, &r).is_err()); assert!(o.poisoned());
}

#[test]
fn compute_and_cursor_errors_never_qualify_for_discard() {
    for fault in ["prefill", "nonfinite", "cursor"] {
        let (mut o, state) = owner(); state.borrow_mut().fail = fault;
        assert!(o.turn(&open(), &SessionCancellationToken::new(), |_| Ok(()), persist).is_err());
        assert!(discard(&mut o, &open()).is_err()); assert!(o.poisoned());
    }
}

#[test]
fn ordinary_error_with_racing_cancel_flag_remains_fatal() {
    let (mut o, _) = owner(); let r = open(); let token = SessionCancellationToken::new();
    o.turn(&r, &token, |_| { token.request_cancel(); Err("output protocol failure".into()) }, persist).unwrap_err();
    assert!(discard(&mut o, &r).is_err());
}

#[test]
fn cancellation_before_admission_cannot_be_used_to_recover_an_owner() {
    let (mut o, _) = owner(); let token = SessionCancellationToken::new(); token.request_cancel();
    assert!(o.turn(&open(), &token, |_| Ok(()), persist).is_err());
    assert!(!o.poisoned()); assert!(discard(&mut o, &open()).is_err());
}

#[test]
fn recovery_drain_reset_and_cursor_failures_are_one_shot_and_never_persisted() {
    for fault in ["drain", "reset", "reset_cursor"] {
        let (mut o, state) = owner(); let r = open(); cancel_after_emit(&mut o, &r);
        state.borrow_mut().fail = fault;
        assert!(o.discard_cancelled_turn("d1", "r1", "s1", |_| panic!("no success receipt")).is_err());
        state.borrow_mut().fail = "";
        assert!(o.discard_cancelled_turn("d2", "r1", "s1", |_| panic!("no retry")).is_err());
        assert!(o.poisoned());
    }
}

#[test]
fn initial_retirement_drain_failure_never_records_recoverable_cancellation() {
    let (mut o, state) = owner(); state.borrow_mut().fail = "drain";
    let token = SessionCancellationToken::new();
    assert!(o.turn(&open(), &token, |_| { token.request_cancel(); Ok(()) }, persist).is_err());
    state.borrow_mut().fail = "";
    assert!(discard(&mut o, &open()).is_err());
}

#[test]
fn discard_receipt_write_failure_or_invalid_digest_keeps_owner_retired() {
    for invalid_digest in [false, true] {
        let (mut o, _) = owner(); let r = open(); cancel_after_emit(&mut o, &r);
        assert!(o.discard_cancelled_turn("d1", "r1", "s1", |_| if invalid_digest {
            Ok("not-a-digest".into())
        } else { Err("injected durable write failure".into()) }).is_err());
        assert!(o.poisoned()); assert!(o.snapshot().is_none());
        assert!(discard(&mut o, &r).is_err());
    }
}

#[test]
fn explicit_retirement_revokes_even_typed_discard_authority() {
    let (mut o, _) = owner(); let r = open(); cancel_after_emit(&mut o, &r);
    o.retire().unwrap(); assert!(discard(&mut o, &r).is_err());
}

#[test]
fn a_typed_error_from_persistence_is_not_an_unpublished_cancellation() {
    let (mut o, _) = owner(); let token = SessionCancellationToken::new();
    let error = o.turn(&open(), &token, |_| Ok(()), |_| {
        token.request_cancel();
        Err(Box::new(crate::session_cancellation::SessionCancellationRequested { boundary: "during persistence" }))
    }).unwrap_err();
    assert!(error.is::<crate::session_cancellation::SessionCancellationRequested>());
    assert!(o.poisoned()); assert!(discard(&mut o, &open()).is_err());
}
