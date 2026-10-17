use super::*;
use crate::session_checkpoint::metadata::{
    DraftRestoreRecipe, FeatureExtent, HistoryMode, NativeLedger, SamplingPolicy, UserSuffixRange,
    SESSION_SCHEMA,
};

pub(crate) fn native() -> SessionDescriptor {
    SessionDescriptor {
        schema: SESSION_SCHEMA.into(),
        bindings: ModelBindings {
            model_sha256: sha256(b"model"),
            q8_draft_sha256: sha256(b"draft"),
            vision_sha256: sha256(b"vision"),
            vision_payload_sha256: sha256(b"vision-payload"),
            model_config_sha256: sha256(b"config"),
            draft_config_sha256: sha256(b"draft-config"),
            tokenizer_sha256: sha256(b"tokenizer"),
            chat_template_sha256: sha256(b"template"),
            generation_config_sha256: sha256(b"generation"),
            frozen_build_sha256: sha256(b"build"),
            numerical_contract_sha256: sha256(b"math"),
            state_layout_sha256: sha256(b"layout"),
            namespace_sha256: sha256(b"namespace"),
        },
        ledger: NativeLedger {
            origin_session_id: "old-session".into(),
            consumed: vec![1, 2, 3, 10, 248046, 22, 31],
            generated: vec![10, 248046, 31, 248046],
            user_suffix_ranges: vec![UserSuffixRange { start: 5, end: 6 }],
            pending: 248046,
            pending_published: true,
            generation: 4,
            revision: 3,
            capacity: CAPACITY,
        },
        features: FeatureExtent {
            start: 0,
            rows: 7,
            physical_rows: HISTORY,
            mode: HistoryMode::Growing2048 { generation: 2 },
        },
        math: NativeMathPolicy {
            full_attention_kv_basis: Default::default(),
            exact_projection_tiles: true,
            exact_output_head_batch: true,
            draft_top16_batch: true,
            parallel_receipt_digest: true,
            batched_target_verify: true,
            reuse_batched_plan: false,
            batched_suffix: false,
            retain_full_accept_state: true,
            share_target_rope: true,
            batched_draft_context: true,
            batched_draft_block: true,
            draft_q8_tile4: true,
        },
        draft_restore: DraftRestoreRecipe::FreshCanonicalQ8FromTargetFeaturesV1,
        sampling: SamplingPolicy::GreedyArgmaxV1,
        eos_token_ids: vec![248046],
        native_request_identity: sha256(b"native-request"),
    }
}

pub(crate) fn fixture() -> HttpSessionCheckpoint {
    let n = native();
    let turns = [10u32, 31]
        .iter()
        .enumerate()
        .map(|(i, &t)| PublishedTurn {
            owner_id: "old-owner".into(),
            request_id: format!("r{i}"),
            request_sha256: sha256(&[i as u8]),
            native_report_sha256: sha256(b"report"),
            native_revision: 2 * i as u64 + 1,
            terminal_sequence: 2,
            token_ids: vec![t, 248046],
            visible_token_ids: vec![t],
            text: t.to_string(),
            text_sha256: sha256(t.to_string().as_bytes()),
            stop_reason: StopReason::Eos,
        })
        .collect();
    HttpSessionCheckpoint {
        schema: HTTP_SCHEMA.into(),
        checkpoint_id: sha256(b"checkpoint"),
        origin_owner_id: "old-owner".into(),
        origin_owner_epoch: 1,
        native_manifest_sha256: sha256(b"manifest"),
        native_transfer_binding_sha256: n.transfer_plan().unwrap().binding_sha256,
        native_descriptor_sha256: sha256(&n.encode().unwrap()),
        target_seal: seal_fixture(&n),
        bindings: n.bindings.clone(),
        math: n.math.clone(),
        eos_token_ids: n.eos_token_ids.clone(),
        ledger: HttpVisibleLedger {
            origin_session_id: n.ledger.origin_session_id.clone(),
            revision: n.ledger.revision,
            generation: n.ledger.generation,
            consumed_token_count: n.ledger.consumed.len(),
            consumed_input_digest: input_digest(&n.ledger.consumed),
            generated_token_count: n.ledger.generated.len(),
            pending_token: n.ledger.pending,
            pending_token_published: true,
            turns,
        },
        capacity: CAPACITY,
        context_rows: HISTORY,
    }
}

pub(crate) fn seal_fixture(n: &SessionDescriptor) -> TargetStateSeal {
    TargetStateSeal {
        target_committed_state_sha256: sha256(b"target"),
        feature_bytes_sha256: sha256(b"features"),
        state_plane_hashes: n.transfer_plan().unwrap().components[1..129]
            .iter()
            .enumerate()
            .map(|(i, c)| StatePlaneSeal {
                layer: i / 2,
                plane: i % 2,
                bytes: c.bytes,
                file_name: format!("layer-{:02}-plane-{}.bin", i / 2, i % 2),
                dtype: if i / 2 % 4 == 3 {
                    "q8_0_34bytes"
                } else {
                    "f32_le"
                }
                .into(),
                kind: if i / 2 % 4 == 3 {
                    "full_kv8"
                } else {
                    "gdn_conv"
                }
                .into(),
                sha256: sha256(b"synthetic-plane-not-gpu-evidence"),
            })
            .collect(),
    }
}

fn reseal(v: &mut serde_json::Value) {
    v.as_object_mut().unwrap().remove("checkpoint_sha256");
    v["checkpoint_sha256"] = serde_json::json!(sha256(&serde_json::to_vec(v).unwrap()));
}

fn checkpoint(n: &SessionDescriptor, s: &TargetStateSeal) -> serde_json::Value {
    use serde_json::json;
    let l = &n.ledger;
    let ranges: Vec<_> = l
        .user_suffix_ranges
        .iter()
        .map(|r| (r.start, r.end))
        .collect();
    let mut v = json!({"schema":"qwen35-native-growing-session-target-checkpoint-v1",
        "session_id":l.origin_session_id,"revision":l.revision,"generation":l.generation,
        "cursor":l.consumed.len(),"capacity":CAPACITY,"consumed_input_digest":input_digest(&l.consumed),
        "pending_token":l.pending,"pending_token_published":l.pending_published,
        "generated_token_count":l.generated.len(),"user_suffix_ranges":ranges,
        "target_planes":128,"draft_stream_drained":true,"draft_rebind_required":true,"product_ready":false,
        "initialized_history":{"schema":"qwen35-initialized-session-history-v1","start_position":n.features.start,
            "active_rows":n.features.rows,"physical_capacity_rows":HISTORY,"uninitialized_tail_hashed":false},
        "target_committed_state_sha256":s.target_committed_state_sha256,
        "feature_bytes_sha256":s.feature_bytes_sha256,"state_plane_hashes":s.state_plane_hashes,
        "state_audit_d2h_bytes":s.state_plane_hashes.iter().map(|p|p.bytes).sum::<u64>()
            + n.features.rows as u64 * crate::session_checkpoint::metadata::FEATURE_ROW_BYTES});
    reseal(&mut v);
    v
}

#[test]
fn typed_state_seal_checks_new_session_and_every_plane() {
    use serde_json::json;
    let n = native();
    let s = seal_fixture(&n);
    let mut v = checkpoint(&n, &s);
    assert_eq!(TargetStateSeal::from_checkpoint(&v, &n).unwrap(), s);
    assert!(s.validate_restored(&v, &n, "new-session").is_err());
    v["session_id"] = json!("new-session");
    reseal(&mut v);
    s.validate_restored(&v, &n, "new-session").unwrap();
    assert!(s.validate_restored(&v, &n, "old-session").is_err());
    for i in 0..128 {
        let mut changed = v.clone();
        changed["state_plane_hashes"][i]["sha256"] = json!(sha256(b"different-state"));
        reseal(&mut changed);
        assert!(s.validate_restored(&changed, &n, "new-session").is_err());
    }
    for key in ["target_committed_state_sha256", "feature_bytes_sha256"] {
        let mut changed = v.clone();
        changed[key] = json!(sha256(b"different-state"));
        reseal(&mut changed);
        assert!(s.validate_restored(&changed, &n, "new-session").is_err());
    }
}

#[test]
fn seal_rejects_rehashed_geometry_or_ledger_mutation() {
    use serde_json::json;
    let n = native();
    let s = seal_fixture(&n);
    let original = checkpoint(&n, &s);
    for (key, value) in [
        ("cursor", json!(8)),
        ("generation", json!(5)),
        ("revision", json!(4)),
        ("pending_token", json!(10)),
        ("pending_token_published", json!(false)),
        ("target_planes", json!(127)),
        ("state_audit_d2h_bytes", json!(0)),
        ("draft_stream_drained", json!(false)),
        ("product_ready", json!(true)),
        ("consumed_input_digest", json!(sha256(b"wrong"))),
    ] {
        let mut v = original.clone();
        v[key] = value;
        reseal(&mut v);
        assert!(TargetStateSeal::from_checkpoint(&v, &n).is_err(), "{key}");
    }
    for (key, value) in [
        ("layer", json!(1)),
        ("plane", json!(1)),
        ("bytes", json!(0)),
        ("dtype", json!("f16")),
        ("kind", json!("full_kv8")),
        ("file_name", json!("../state")),
        ("unknown_authority", json!(true)),
    ] {
        let mut v = original.clone();
        v["state_plane_hashes"][0][key] = value;
        reseal(&mut v);
        assert!(TargetStateSeal::from_checkpoint(&v, &n).is_err(), "{key}");
    }
    let mut v = original.clone();
    v["initialized_history"]["uninitialized_tail_hashed"] = json!(true);
    reseal(&mut v);
    assert!(TargetStateSeal::from_checkpoint(&v, &n).is_err());
    let mut v = original;
    v["checkpoint_sha256"] = json!(sha256(b"stale"));
    assert!(TargetStateSeal::from_checkpoint(&v, &n).is_err());
}

#[test]
fn live_boundary_cannot_be_used_as_a_persistent_payload_seal() {
    use serde_json::json;
    let descriptor = native();
    let seal = seal_fixture(&descriptor);
    let mut live = checkpoint(&descriptor, &seal);
    live["schema"] = json!("qwen35-native-live-session-boundary-v1");
    // Even retaining old full-hash-looking fields cannot bypass the schema boundary.
    reseal(&mut live);
    assert!(TargetStateSeal::from_checkpoint(&live, &descriptor).is_err());
    for key in [
        "target_committed_state_sha256",
        "feature_bytes_sha256",
        "state_plane_hashes",
        "target_planes",
    ] {
        live.as_object_mut().unwrap().remove(key);
    }
    live["state_audit_d2h_bytes"] = json!(0);
    live["payload_checksum_verified"] = json!(false);
    live["persistable_state_seal"] = json!(false);
    reseal(&mut live);
    assert!(TargetStateSeal::from_checkpoint(&live, &descriptor).is_err());
    // Spoofing the old schema still cannot manufacture missing state bytes/hashes.
    live["schema"] = json!("qwen35-native-growing-session-target-checkpoint-v1");
    reseal(&mut live);
    assert!(TargetStateSeal::from_checkpoint(&live, &descriptor).is_err());
    assert_eq!(
        TargetStateSeal::from_checkpoint(&checkpoint(&descriptor, &seal), &descriptor).unwrap(),
        seal
    );
}

#[test]
fn complete_turns_bind_native_and_roundtrip() {
    let h = fixture();
    h.validate_native(&native(), &sha256(b"manifest")).unwrap();
    let bytes = h.encode().unwrap();
    assert_eq!(
        HttpSessionCheckpoint::decode(&bytes, &sha256(&bytes)).unwrap(),
        h
    );
    h.validate_text(|ids| Ok(ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")))
        .unwrap();
}

#[test]
fn restore_requires_live_bindings_and_distinct_owner_session() {
    let h = fixture();
    let n = native();
    h.validate_restore(
        "new-owner",
        1,
        &NativeRestoreAdmission {
            new_session_id: "new-session".into(),
            bindings: n.bindings.clone(),
            math: n.math.clone(),
        },
        &n,
        &sha256(b"manifest"),
    )
    .unwrap();
    for (owner, epoch, session) in [
        ("old-owner", 2, "new-session"),
        ("new-owner", 1, "old-session"),
        ("new-owner", 0, "new-session"),
        ("new-owner", u64::MAX, "new-session"),
        ("../bad", 1, "new-session"),
    ] {
        assert!(h
            .validate_restore(
                owner,
                epoch,
                &NativeRestoreAdmission {
                    new_session_id: session.into(),
                    bindings: n.bindings.clone(),
                    math: n.math.clone()
                },
                &n,
                &sha256(b"manifest")
            )
            .is_err());
    }
    let mut changed = n.bindings.clone();
    changed.namespace_sha256 = sha256(b"other-user");
    assert!(h
        .validate_restore(
            "new-owner",
            1,
            &NativeRestoreAdmission {
                new_session_id: "new-session".into(),
                bindings: changed,
                math: n.math.clone()
            },
            &n,
            &sha256(b"manifest")
        )
        .is_err());
    let mut changed = n.math.clone();
    changed.draft_top16_batch = false;
    assert!(h
        .validate_restore(
            "new-owner",
            1,
            &NativeRestoreAdmission {
                new_session_id: "new-session".into(),
                bindings: n.bindings.clone(),
                math: changed
            },
            &n,
            &sha256(b"manifest")
        )
        .is_err());
}

#[test]
fn untrusted_outer_hash_and_noncanonical_bytes_reject() {
    let bytes = fixture().encode().unwrap();
    assert!(HttpSessionCheckpoint::decode(&bytes, &sha256(b"wrong")).is_err());
    let mut spaced = bytes.clone();
    spaced.insert(1, b' ');
    assert!(HttpSessionCheckpoint::decode(&spaced, &sha256(&spaced)).is_err());
    let mut duplicate = bytes.clone();
    duplicate.splice(
        1..1,
        br#""schema":"cq-qwen35-http-session-checkpoint-v1","#
            .iter()
            .copied(),
    );
    assert!(HttpSessionCheckpoint::decode(&duplicate, &sha256(&duplicate)).is_err());
    for path in [vec![], vec!["ledger"], vec!["bindings"], vec!["math"]] {
        let mut value = serde_json::to_value(fixture()).unwrap();
        let mut object = &mut value;
        for k in path {
            object = &mut object[k];
        }
        object["authority_override"] = serde_json::Value::Bool(true);
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(HttpSessionCheckpoint::decode(&bytes, &sha256(&bytes)).is_err());
    }
}

#[test]
fn native_binding_hash_model_and_descriptor_drift_reject() {
    let h = fixture();
    let n = native();
    assert!(h.validate_native(&n, &sha256(b"other-manifest")).is_err());
    let mut wrong = h.clone();
    wrong.native_descriptor_sha256 = sha256(b"other");
    assert!(wrong.validate_native(&n, &sha256(b"manifest")).is_err());
    let mut wrong = h.clone();
    wrong.native_transfer_binding_sha256 = sha256(b"other");
    assert!(wrong.validate_native(&n, &sha256(b"manifest")).is_err());
    let mut wrong = h;
    wrong.bindings.model_config_sha256 = sha256(b"other");
    assert!(wrong.validate_native(&n, &sha256(b"manifest")).is_err());
}

#[test]
fn native_and_http_ledger_cannot_disagree() {
    type CheckpointMutation = Box<dyn Fn(&mut HttpSessionCheckpoint)>;
    let mutations: Vec<CheckpointMutation> = vec![
        Box::new(|h| h.ledger.generation += 1),
        Box::new(|h| h.ledger.consumed_token_count += 1),
        Box::new(|h| h.ledger.consumed_input_digest = sha256(b"forged-history")),
        Box::new(|h| h.ledger.origin_session_id = "wrong-session".into()),
        Box::new(|h| h.ledger.turns[0].token_ids[0] = 12),
        Box::new(|h| h.ledger.pending_token = 42),
        Box::new(|h| h.ledger.pending_token_published = false),
    ];
    for mutate in mutations {
        let mut h = fixture();
        mutate(&mut h);
        assert!(h.validate_native(&native(), &sha256(b"manifest")).is_err());
    }
}

#[test]
fn mutually_consistent_forged_http_tokens_still_fail_native_comparison() {
    let mut h = fixture();
    let t = &mut h.ledger.turns[0];
    t.token_ids[0] = 12;
    t.visible_token_ids[0] = 12;
    t.text = "12".into();
    t.text_sha256 = sha256(b"12");
    h.validate().unwrap();
    assert!(h.validate_native(&native(), &sha256(b"manifest")).is_err());
}

#[test]
fn duplicate_request_is_scoped_to_its_provenance_owner() {
    let mut h = fixture();
    h.ledger.turns[1].request_id = "r0".into();
    assert!(h.validate().is_err());
    h.ledger.turns[1].owner_id = "another-owner".into();
    h.validate().unwrap();
    assert!(h
        .validate_restore(
            "another-owner",
            2,
            &NativeRestoreAdmission {
                new_session_id: "new-session".into(),
                bindings: native().bindings,
                math: native().math
            },
            &native(),
            &sha256(b"manifest")
        )
        .is_err());
}

#[test]
fn text_hash_is_not_tokenizer_semantic_equivalence() {
    let mut h = fixture();
    h.ledger.turns[0].text = "forged".into();
    h.ledger.turns[0].text_sha256 = sha256(b"forged");
    h.validate_native(&native(), &sha256(b"manifest")).unwrap();
    assert!(h.validate_text(|ids| Ok(ids[0].to_string())).is_err());
    assert!(fixture()
        .validate_text(|_| Err(invalid("tokenizer failure")))
        .is_err());
}

#[test]
fn eos_must_not_be_visible_or_hidden_mid_turn() {
    let mut h = fixture();
    h.ledger.turns[0].visible_token_ids.push(248046);
    assert!(h.validate().is_err());
    let mut h = fixture();
    h.ledger.turns[0].stop_reason = StopReason::MaxNewTokens;
    assert!(h.validate().is_err());
    let mut h = fixture();
    h.ledger.turns[0].token_ids[0] = 248046;
    h.ledger.turns[0].visible_token_ids[0] = 248046;
    assert!(h.validate().is_err());
}

#[test]
fn terminal_sequence_and_revision_conservation() {
    for sequence in [0, 1, 3, u64::MAX] {
        let mut h = fixture();
        h.ledger.turns[0].terminal_sequence = sequence;
        assert!(h.validate().is_err());
    }
    let mut h = fixture();
    h.ledger.turns[1].native_revision = 2;
    h.ledger.revision = 2;
    h.validate().unwrap();
    assert!(h.validate_native(&native(), &sha256(b"manifest")).is_err());
}

#[test]
fn zero_visible_eos_turn_is_valid_not_fake_text() {
    let t = PublishedTurn {
        owner_id: "o".into(),
        request_id: "r".into(),
        request_sha256: sha256(b"r"),
        native_report_sha256: sha256(b"report"),
        native_revision: 1,
        terminal_sequence: 1,
        token_ids: vec![248046],
        visible_token_ids: vec![],
        text: String::new(),
        text_sha256: sha256(b""),
        stop_reason: StopReason::Eos,
    };
    t.validate(&[248046]).unwrap();
}

#[test]
fn capacity_history_turn_and_metadata_bounds_are_explicit() {
    for (capacity, history) in [
        (0, HISTORY),
        (CAPACITY - 1, HISTORY),
        (CAPACITY, 0),
        (CAPACITY, HISTORY + 1),
    ] {
        let mut h = fixture();
        h.capacity = capacity;
        h.context_rows = history;
        assert!(h.encode().is_err());
    }
    let mut h = fixture();
    h.ledger.turns.clear();
    assert!(h.encode().is_err());
    let mut h = fixture();
    h.ledger.turns = vec![h.ledger.turns[0].clone(); MAX_TURNS + 1];
    assert!(h.encode().is_err());
    let big = vec![b' '; MAX_HTTP_METADATA_BYTES + 1];
    assert!(HttpSessionCheckpoint::decode(&big, &sha256(&big)).is_err());
}
