//! Augur-specific protocol checks. Never relabel a legacy GGUF producer.
use augur_model_qwen35_runtime::augur_session::*;
use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub(super) struct Active {
    pub snapshot: SessionSnapshot,
    pub receipt_sha256: String,
}
pub(super) struct Prepared {
    pub request: TurnRequest,
    pub prefix: Vec<u32>,
    pub revision: u64,
    pub previous: Option<String>,
    pub image_identity: Option<String>,
}
pub(super) fn prepare(r: TurnRequest, active: Option<&Active>) -> Result<Prepared> {
    if !valid_id(&r.request_id)
        || !valid_id(&r.session_id)
        || !(1..=context_output_limit()?).contains(&r.max_new_tokens)
    {
        return Err("invalid request identity/budget".into());
    }
    let mut image_identity=None;
    let (prefix, revision, previous) = match &r.input {
        TurnInput::Open { prompt_token_ids } | TurnInput::OpenImages { prompt_token_ids, .. } => {
            if active.is_some() || prompt_token_ids.is_empty() {
                return Err("active session or empty prompt".into());
            }
            if let TurnInput::OpenImages {images,..}=&r.input {
                if !cfg!(feature="qwen35-augur-vision") {return Err("Vision unavailable".into());}
                image_identity=Some(images::prepare(prompt_token_ids,images)?.identity_sha256);
            } else if prompt_token_ids.iter().any(|id|matches!(id,248053..=248057)) {
                return Err("Vision markers require images".into());
            }
            (prompt_token_ids.clone(), 1, None)
        }
        TurnInput::Append {
            revision,
            exact_history,
            suffix_token_ids,
        } | TurnInput::AppendImages {revision,exact_history,suffix_token_ids,..} => {
            let a = active.ok_or("no retained Augur session")?;
            if a.snapshot.session_id != r.session_id
                || a.snapshot.revision != *revision
                || a.snapshot.consumed_token_ids != *exact_history
                || suffix_token_ids.is_empty()
            {
                return Err("append exact identity/history mismatch".into());
            }
            image_identity=a.snapshot.image_input_sha256.clone();
            let supplied=match &r.input {TurnInput::AppendImages {image_input_sha256,..}=>Some(image_input_sha256),_=>None};
            if supplied!=image_identity.as_ref() || suffix_token_ids.iter().any(|id|matches!(id,248053..=248057)) {
                return Err("append image identity/new-image mismatch".into());
            }
            let mut prefix = exact_history.clone();
            prefix.push(a.snapshot.pending_token);
            prefix.extend_from_slice(suffix_token_ids);
            (
                prefix,
                revision.checked_add(1).ok_or("revision overflow")?,
                Some(a.receipt_sha256.clone()),
            )
        }
    };
    if prefix.iter().any(|&id| id >= VOCAB)
        || !fits_context(prefix.len().checked_add(r.max_new_tokens))?
    {
        return Err("Augur token/capacity budget; no truncation".into());
    }
    Ok(Prepared {
        request: r,
        prefix,
        revision,
        previous,
        image_identity,
    })
}
pub(super) fn ready(frame: &Value, manifest: &str) -> Result<()> {
    if frame["event"] != "ready"
        || frame["schema"] != "quasar-token-session-jsonl-v1"
        || frame["profile"] != requested_profile()?
        || frame["manifest_sha256"] != manifest
        || frame["capacity"] != context_capacity()?
        || frame["kv"] != "Q8_0"
        || frame["lm_head"] != "original_bf16"
        || frame["maximum_output_tokens"] != context_output_limit()?
        || frame["cooperative_input_cancellation"] != true
        || frame["target"]["policy"] != NUMERICAL_POLICY
        || frame["target"]["kv_capacity"] != context_capacity()?
        || frame["target"]["kv_cache_bytes"] != context_capacity()? as u64 * 34816
        || frame["target"]["qpn_input_staging_schedule"] != "fused_cached_norm_qkv_v1"
        || frame["target"]["rmsnorm_schedule"] != "same_tree_warp_cached5120_v1"
        || frame["target"]["ffn_batch_max_rows"] != 8
        || frame["generation"] != "greedy_lowest_id_tie_nonfinite_fail_closed"
        || frame["pending_contract"]
            != "consumed_history_excludes_pending_append_consumes_pending_once"
    {
        return Err("producer Augur ready identity/numerical/capacity mismatch".into());
    }
    #[cfg(feature = "qwen35-augur-http")]
    if frame["target"]["http_execution"]["profile"] != requested_profile()?
        || frame["target"]["http_execution"]["batch"] != 4352
        || frame["target"]["http_execution"]["numerical_qualified"] != false
        || frame["target"]["http_execution"]["dflash_active"] != requested_dflash_enabled()? {
        return Err("producer speed-first HTTP execution profile mismatch".into());
    }
    for key in [
        "product_ready",
        "actual_long_history_qualified",
        "http_enabled",
    ] {
        if frame[key] != false {
            return Err(format!("producer unsupported ready claim {key}").into());
        }
    }
    if frame["restore_enabled"] != checkpoint::storage_requested() {return Err("producer checkpoint capability mismatch".into());}
    if frame["dflash_enabled"] != requested_dflash_enabled()? {return Err("producer DFlash capability mismatch".into());}
    if frame["vision_enabled"] != cfg!(feature="qwen35-augur-vision") {
        return Err("producer Vision capability mismatch".into());
    }
    Ok(())
}
pub(super) fn token(frame: &Value, p: &Prepared, tokens: &mut Vec<u32>, eos: &[u32]) -> Result<()> {
    let t = &frame["token"];
    let id = t["token_id"]
        .as_u64()
        .filter(|&v| v < VOCAB as u64)
        .ok_or("invalid token ID")? as u32;
    if frame["event"] != "token"
        || frame["product_ready"] != false
        || t["request_id"] != p.request.request_id
        || t["session_id"] != p.request.session_id
        || t["sequence"] != tokens.len()
        || tokens.len() >= p.request.max_new_tokens
        || eos.contains(&id)
    {
        return Err("token frame identity/sequence/EOS mismatch".into());
    }
    tokens.push(id);
    Ok(())
}
pub(super) fn terminal(
    frame: &Value,
    p: &Prepared,
    tokens: &[u32],
    eos: &[u32],
    manifest: &str,
) -> Result<Active> {
    let r = &frame["committed"]["receipt"];
    let hash = frame["committed"]["receipt_sha256"]
        .as_str()
        .filter(|s| valid_digest(s))
        .ok_or("receipt digest")?;
    let snapshot: SessionSnapshot = serde_json::from_value(r["session"].clone())?;
    if frame["event"] != "terminal"
        || frame["product_ready"] != false
        || frame["request_id"] != p.request.request_id
        || frame["session_id"] != p.request.session_id
        || r["schema"] != "quasar-token-turn-receipt-v1"
        || r["request_id"] != p.request.request_id
        || r["previous_receipt_sha256"] != json!(p.previous)
        || r["output_token_ids"] != json!(tokens)
        || snapshot.profile != requested_profile()?
        || snapshot.manifest_sha256 != manifest
        || snapshot.session_id != p.request.session_id
        || snapshot.revision != p.revision
        || snapshot.image_input_sha256 != p.image_identity
    {
        return Err("Augur committed terminal identity mismatch".into());
    }
    for key in ["product_ready"] {
        if r[key] != false {
            return Err("unqualified turn claim".into());
        }
    }
    if r["dflash_enabled"] != requested_dflash_enabled()? {return Err("turn DFlash capability mismatch".into());}
    if requested_dflash_enabled()? && (r["decode_detail"]["dflash"]["profile"] != DFLASH_B8_PROFILE || r["decode_detail"]["dflash"]["active"] != true) {return Err("turn DFlash policy mismatch".into());}
    if r["vision_enabled"] != p.image_identity.is_some() {return Err("turn Vision identity mismatch".into());}
    let selected = tokens.len() + usize::from(!snapshot.pending_token_published);
    if selected == 0
        || selected > p.request.max_new_tokens
        || r["selected_tokens_including_eos"] != selected
    {
        return Err("selected token budget mismatch".into());
    }
    let mut expected = p.prefix.clone();
    if snapshot.pending_token_published {
        if r["stop_reason"] != "length"
            || tokens.len() != p.request.max_new_tokens
            || tokens.last() != Some(&snapshot.pending_token)
        {
            return Err("length pending mismatch".into());
        }
        expected.extend_from_slice(&tokens[..tokens.len() - 1]);
    } else {
        if r["stop_reason"] != "eos" || !eos.contains(&snapshot.pending_token) {
            return Err("EOS pending mismatch".into());
        }
        expected.extend_from_slice(tokens);
    }
    let raw: Vec<u8> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();
    if snapshot.consumed_token_ids != expected || snapshot.consumed_input_sha256 != digest(&raw) {
        return Err("committed exact history conservation mismatch".into());
    }
    Ok(Active {
        snapshot,
        receipt_sha256: hash.into(),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> TurnRequest {
        TurnRequest {
            selector_policy: None, request_id: "r".into(),
            session_id: "s".into(),
            max_new_tokens: 3,
            input: TurnInput::Open {
                prompt_token_ids: vec![1, 2],
            },
        }
    }
    #[test]
    fn capacity_ids_and_exact_history_are_not_weakened_by_http() {
        let p = prepare(request(), None).unwrap();
        assert_eq!(p.prefix, [1, 2]);
        let mut r = request();
        r.max_new_tokens = 0;
        assert!(prepare(r, None).is_err());
        let mut r = request();
        r.input = TurnInput::Open {
            prompt_token_ids: vec![VOCAB],
        };
        assert!(prepare(r, None).is_err());
        let mut r = request();
        r.input = TurnInput::Open {
            prompt_token_ids: vec![0; CAPACITY],
        };
        assert!(prepare(r, None).is_err());
        let mut r = request();
        r.input = TurnInput::Append {
            revision: 1,
            exact_history: vec![1],
            suffix_token_ids: vec![2],
        };
        assert!(prepare(r, None).is_err());
    }
    #[test]
    fn terminal_requires_exact_pending_ledger_and_binding() {
        let p = prepare(request(), None).unwrap();
        let tokens = vec![16, 24];
        let mut ids = p.prefix.clone();
        ids.extend(&tokens);
        let raw: Vec<_> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        let s = SessionSnapshot {
            profile: PROFILE.into(),
            manifest_sha256: "a".repeat(64),
            session_id: "s".into(),
            revision: 1,
            consumed_input_sha256: digest(&raw),
            consumed_token_ids: ids,
            pending_token: 248046,
            pending_token_published: false,
            image_input_sha256: None,
        };
        let f = json!({"event":"terminal","product_ready":false,"request_id":"r","session_id":"s","committed":{
            "receipt_sha256":"b".repeat(64),"receipt":{"schema":"quasar-token-turn-receipt-v1","request_id":"r",
            "previous_receipt_sha256":null,"output_token_ids":tokens,"session":s,"selected_tokens_including_eos":3,
            "stop_reason":"eos","product_ready":false,"dflash_enabled":false,"vision_enabled":false}}});
        assert!(terminal(&f, &p, &tokens, &[248046], &"a".repeat(64)).is_ok());
        for key in [
            "profile",
            "manifest_sha256",
            "pending_token",
            "pending_token_published",
            "consumed_input_sha256",
            "revision",
        ] {
            let mut bad = f.clone();
            bad["committed"]["receipt"]["session"][key] = Value::Null;
            assert!(
                terminal(&bad, &p, &tokens, &[248046], &"a".repeat(64)).is_err(),
                "{key}"
            );
        }
        assert!(terminal(&f, &p, &[16], &[248046], &"a".repeat(64)).is_err());
    }
}
