//! Validate reset authority, not just a still-live child or an empty socket.
use super::*;
pub(super) fn validate_discard(frame: &Value, rid: &str, sid: &str, discard_id: &str, manifest: &str) -> Result<()> {
    let value = &frame["discarded"]; let r = &value["receipt"];
    if frame["event"] != "cancelled" || frame["request_id"] != rid || frame["session_id"] != sid
        || frame["owner_retired"] != false || frame["product_ready"] != false
        || r["schema"] != "quasar-session-cancellation-discard-v1" || r["request_id"] != discard_id
        || r["cancelled_request_id"] != rid || r["session_id"] != sid || r["manifest_sha256"] != manifest
        || r["drain_completed"] != true || r["reset_completed"] != true || r["session_discarded"] != true
        || r["previous_history_reusable"] != false || r["product_ready"] != false || r["dflash_enabled"] != requested_dflash_enabled()?
        || r["submitted_tokens_after_discard"].as_u64() != Some(0)
        || !r["submitted_tokens_before_discard"].as_u64().is_some_and(|v| v <= CAPACITY as u64)
        || !r["cancellation_boundary"].as_str().is_some_and(|v| !v.is_empty() && v.len() < 256)
        || !value["receipt_sha256"].as_str().is_some_and(valid_digest) {
        return Err("cancellation discard identity/drain/reset receipt mismatch".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn valid() -> Value { json!({"event":"cancelled","request_id":"r","session_id":"s",
        "owner_retired":false,"product_ready":false,"discarded":{"receipt_sha256":"a".repeat(64),
        "receipt":{"schema":"quasar-session-cancellation-discard-v1","request_id":"d","cancelled_request_id":"r",
        "session_id":"s","manifest_sha256":"b".repeat(64),"cancellation_boundary":"before decode",
        "submitted_tokens_before_discard":20,"submitted_tokens_after_discard":0,"drain_completed":true,
        "reset_completed":true,"session_discarded":true,"previous_history_reusable":false,
        "product_ready":false,"dflash_enabled":false}}}) }
    #[test]
    fn exact_complete_discard_is_required() {
        let frame = valid(); validate_discard(&frame,"r","s","d",&"b".repeat(64)).unwrap();
        for field in ["drain_completed","reset_completed","session_discarded"] {
            let mut frame = valid(); frame["discarded"]["receipt"][field] = false.into();
            assert!(validate_discard(&frame,"r","s","d",&"b".repeat(64)).is_err());
        }
        for field in ["manifest_sha256","cancelled_request_id","session_id","request_id","schema"] {
            let mut frame = valid(); frame["discarded"]["receipt"][field] = "foreign".into();
            assert!(validate_discard(&frame,"r","s","d",&"b".repeat(64)).is_err());
        }
    }
    #[test]
    fn nonzero_or_boolean_cursor_and_unbound_digest_cannot_clear_fault() {
        for value in [json!(1), json!(false), Value::Null] {
            let mut frame = valid(); frame["discarded"]["receipt"]["submitted_tokens_after_discard"] = value;
            assert!(validate_discard(&frame,"r","s","d",&"b".repeat(64)).is_err());
        }
        let mut frame = valid(); frame["discarded"]["receipt_sha256"] = "wrong".into();
        assert!(validate_discard(&frame,"r","s","d",&"b".repeat(64)).is_err());
    }
}
