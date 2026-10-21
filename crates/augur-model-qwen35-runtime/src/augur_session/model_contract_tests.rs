use super::*;
#[derive(Default)]
struct Bounded {
    count: usize,
    calls: usize,
}
impl SessionTarget for Bounded {
    fn profile(&self) -> &'static str {
        "other_model_research_v1"
    }
    fn context_capacity(&self) -> Result<usize> {
        Ok(8)
    }
    fn turn_receipt_schema(&self) -> &'static str {
        "other-turn-v1"
    }
    fn close_receipt_schema(&self) -> &'static str {
        "other-close-v1"
    }
    fn cancellation_receipt_schema(&self) -> &'static str {
        "other-cancel-v1"
    }
    fn consumed_tokens(&self) -> usize {
        self.count
    }
    fn consume_sequence(&mut self, t: &[u32]) -> Result<()> {
        self.count += t.len();
        self.calls += 1;
        Ok(())
    }
    fn greedy_next_token(&mut self) -> Result<u32> {
        Ok(7)
    }
    fn consume_selected(&mut self) -> Result<()> {
        self.count += 1;
        self.calls += 1;
        Ok(())
    }
    fn drain(&mut self) -> Result<()> {
        Ok(())
    }
    fn reset_empty(&mut self) -> Result<()> {
        self.count = 0;
        Ok(())
    }
}
fn request() -> TurnRequest {
    TurnRequest {
        request_id: "r".into(),
        session_id: "s".into(),
        max_new_tokens: 3,
        selector_policy: None,
        input: TurnInput::Open {
            prompt_token_ids: vec![1, 2],
        },
    }
}
fn persist(r: &TurnReceipt) -> Result<String> {
    Ok(digest(&serde_json::to_vec(r)?))
}
#[test]
fn per_owner_capacity_and_receipts_do_not_alias_quasar_defaults() {
    let mut owner = Session::new(Bounded::default(), "a".repeat(64), vec![248046]).unwrap();
    let mut r = request();
    r.max_new_tokens = 8;
    assert!(owner
        .turn(&r, &SessionCancellationToken::new(), |_| Ok(()), persist)
        .is_err());
    r.max_new_tokens = 7; // Output bound fits, prompt+output does not.
    assert!(owner
        .turn(&r, &SessionCancellationToken::new(), |_| Ok(()), persist)
        .is_err());
    assert_eq!(owner.target.calls, 0);
    assert!(!owner.poisoned());
    assert!(owner.requests.is_empty());
    let result = owner
        .turn(
            &request(),
            &SessionCancellationToken::new(),
            |_| Ok(()),
            persist,
        )
        .unwrap();
    assert_eq!(result.receipt.schema, "other-turn-v1");
    assert_eq!(result.receipt.session.profile, "other_model_research_v1");
    assert_eq!(result.receipt.session.consumed_token_ids, [1, 2, 7, 7]);
    assert_eq!(result.receipt.session.pending_token, 7);
    owner
        .close("close", "s", 1, |r| {
            assert_eq!(r.schema, "other-close-v1");
            Ok(digest(&serde_json::to_vec(r)?))
        })
        .unwrap();
    assert_eq!(owner.submitted_tokens(), 0);
}
#[cfg(any(feature = "qwen35-augur-http", feature = "qwen35-nerkyor-cancel-recovery"))]
#[test]
fn cancellation_discard_keeps_the_own_model_receipt_namespace() {
    let mut owner = Session::new(Bounded::default(), "a".repeat(64), vec![248046]).unwrap();
    let cancel = SessionCancellationToken::new();
    assert!(owner
        .turn(
            &request(),
            &cancel,
            |_| {
                cancel.request_cancel();
                Ok(())
            },
            persist
        )
        .is_err());
    assert!(owner.poisoned());
    let r = owner
        .discard_cancelled_turn("discard", "r", "s", |r| Ok(digest(&serde_json::to_vec(r)?)))
        .unwrap();
    assert_eq!(r.receipt.schema, "other-cancel-v1");
    assert!(!owner.poisoned());
}
