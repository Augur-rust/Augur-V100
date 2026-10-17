use super::super::metadata::{SessionDescriptor, FEATURE_ROW_BYTES, KV_ROW_BYTES};
use super::*;

#[test]
fn generation_admission_requires_real_label_binding_budget_and_reserve() {
    let d = descriptor(8192);
    let valid = InputPrefixGenerationAdmission {
        bindings: &d.bindings,
        session_id: "new-session",
        max_new_tokens: 64,
    };
    assert!(valid.validate(&d).is_ok());
    for id in ["", "space id", "../session"] {
        assert!(InputPrefixGenerationAdmission {
            session_id: id,
            ..valid
        }
        .validate(&d)
        .is_err());
    }
    for max in [0, 1025, usize::MAX] {
        assert!(InputPrefixGenerationAdmission {
            max_new_tokens: max,
            ..valid
        }
        .validate(&d)
        .is_err());
    }
    let mut other = d.bindings.clone();
    other.namespace_sha256 = sha256(b"other");
    assert!(InputPrefixGenerationAdmission {
        bindings: &other,
        ..valid
    }
    .validate(&d)
    .is_err());
    let full = descriptor(CAPACITY - 69);
    assert!(valid.validate(&full).is_err());
    let fitting = descriptor(CAPACITY - 70);
    assert!(valid.validate(&fitting).is_ok());
}

fn bindings() -> ModelBindings {
    let names = [
        "model_sha256",
        "q8_draft_sha256",
        "vision_sha256",
        "vision_payload_sha256",
        "model_config_sha256",
        "draft_config_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "generation_config_sha256",
        "frozen_build_sha256",
        "numerical_contract_sha256",
        "state_layout_sha256",
        "namespace_sha256",
    ];
    let values: serde_json::Map<String, serde_json::Value> = names
        .into_iter()
        .map(|name| (name.into(), sha256(name.as_bytes()).into()))
        .collect();
    serde_json::from_value(values.into()).unwrap()
}

fn descriptor(tokens: usize) -> InputPrefixDescriptor {
    InputPrefixDescriptor {
        schema: INPUT_PREFIX_SCHEMA.into(),
        bindings: bindings(),
        consumed_input: vec![7; tokens],
        features: FeatureExtent {
            start: tokens - tokens.min(HISTORY),
            rows: tokens.min(HISTORY),
            physical_rows: HISTORY,
            mode: HistoryMode::Growing2048 { generation: 1 },
        },
        math: NativeMathPolicy {
            full_attention_kv_basis: Default::default(),
            exact_projection_tiles: true,
            exact_output_head_batch: true,
            draft_top16_batch: true,
            parallel_receipt_digest: true,
            batched_target_verify: true,
            reuse_batched_plan: true,
            batched_suffix: true,
            retain_full_accept_state: true,
            share_target_rope: true,
            batched_draft_context: true,
            batched_draft_block: true,
            draft_q8_tile4: true,
        },
        draft_restore: DraftRestoreRecipe::FreshCanonicalQ8FromTargetFeaturesV1,
        sampling: SamplingPolicy::GreedyArgmaxV1,
        origin_request_sha256: sha256(b"original-input-request"),
    }
}

#[test]
fn input_prefix_roundtrips_short_long_and_full_capacity_without_fake_output() {
    for count in [1, 7, 2048, 2051, 32768, 65536, 194492, CAPACITY] {
        let value = descriptor(count);
        let bytes = value.encode().unwrap();
        assert_eq!(
            InputPrefixDescriptor::decode(&bytes, &value.bindings, &value.math).unwrap(),
            value
        );
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        for absent in [
            "pending",
            "pending_token",
            "generated",
            "generated_token_ids",
            "terminal",
        ] {
            assert!(json.get(absent).is_none());
        }
    }
}

#[test]
fn input_prefix_inventory_keeps_every_kv_gdn_and_real_feature_plane() {
    let value = descriptor(8193);
    let plan = value.transfer_plan().unwrap();
    assert_eq!(plan.components.len(), 130);
    assert_eq!(plan.components[0].name, "input_prefix.metadata");
    assert_eq!(
        plan.components[0].bytes,
        value.encode().unwrap().len() as u64
    );
    for layer in 0..64 {
        let first = &plan.components[1 + 2 * layer];
        let second = &plan.components[2 + 2 * layer];
        if layer % 4 == 3 {
            assert_eq!(first.name, format!("target.layer{layer:02}.key"));
            assert_eq!(second.name, format!("target.layer{layer:02}.value"));
            assert_eq!(first.bytes, 8193 * KV_ROW_BYTES);
            assert_eq!(second.bytes, first.bytes);
        } else {
            assert_eq!(first.name, format!("target.layer{layer:02}.conv_history"));
            assert_eq!(second.name, format!("target.layer{layer:02}.recurrent"));
            assert_eq!(first.bytes, 30720 * 4);
            assert_eq!(second.bytes, 786432 * 4);
        }
    }
    assert_eq!(plan.components[129].name, "target.features");
    assert_eq!(
        plan.components[129].bytes,
        HISTORY as u64 * FEATURE_ROW_BYTES
    );
    let short = descriptor(7).transfer_plan().unwrap();
    assert_eq!(short.components[129].bytes, 7 * FEATURE_ROW_BYTES);
}

#[test]
fn input_prefix_does_not_alias_legacy_generated_session_schema() {
    let value = descriptor(8);
    assert!(
        SessionDescriptor::decode(&value.encode().unwrap(), &value.bindings, &value.math).is_err()
    );
    let mut forged = serde_json::to_value(&value).unwrap();
    forged["pending_token"] = 0.into();
    assert!(InputPrefixDescriptor::decode(
        &serde_json::to_vec(&forged).unwrap(),
        &value.bindings,
        &value.math
    )
    .is_err());
}

#[test]
fn input_prefix_rejects_wrong_schema_and_noncanonical_bytes() {
    let mut value = descriptor(8);
    let pretty = serde_json::to_vec_pretty(&value).unwrap();
    assert!(InputPrefixDescriptor::decode(&pretty, &value.bindings, &value.math).is_err());
    value.schema = "cq-qwen35-native-session-ssd-descriptor-v1".into();
    assert!(value.validate().is_err());
}

#[test]
fn input_prefix_rejects_fake_history_rows_modes_and_exhausted_counter() {
    for (start, rows, physical) in [(1, 7, HISTORY), (0, 8, HISTORY), (0, 7, 7)] {
        let mut value = descriptor(7);
        value.features.start = start;
        value.features.rows = rows;
        value.features.physical_rows = physical;
        assert!(value.validate().is_err());
    }
    for mode in [
        HistoryMode::Fixed2048,
        HistoryMode::Growing2048 {
            generation: u64::MAX,
        },
    ] {
        let mut value = descriptor(2048);
        value.features.mode = mode;
        assert!(value.validate().is_err());
    }
}

#[test]
fn input_prefix_rejects_empty_oversize_and_out_of_vocabulary() {
    for value in [descriptor(0), descriptor(CAPACITY + 1)] {
        assert!(value.validate().is_err());
    }
    let mut value = descriptor(1);
    value.consumed_input[0] = VOCAB;
    assert!(value.validate().is_err());
    value.consumed_input[0] = 0; // Token zero can be real input; it is never a fake pending output.
    assert!(value.validate().is_ok());
}

#[test]
fn input_prefix_decode_binds_every_artifact_namespace_and_policy_domain() {
    let value = descriptor(8);
    let encoded = value.encode().unwrap();
    let original = serde_json::to_value(&value.bindings).unwrap();
    for key in original.as_object().unwrap().keys() {
        let mut changed = original.clone();
        changed[key] = sha256(b"different-live-binding").into();
        let expected = serde_json::from_value(changed).unwrap();
        assert!(
            InputPrefixDescriptor::decode(&encoded, &expected, &value.math).is_err(),
            "{key}"
        );
    }
    let mut math = value.math.clone();
    math.parallel_receipt_digest = false;
    assert!(InputPrefixDescriptor::decode(&encoded, &value.bindings, &math).is_err());
}

#[test]
fn input_prefix_lookup_accepts_large_real_suffix_without_chat_limit_or_padding() {
    let value = descriptor(8192);
    let mut request = value.consumed_input.clone();
    request.extend(vec![17; 4097]);
    let hit = value
        .lookup_suffix(&value.bindings, &value.math, &request, 128)
        .unwrap()
        .unwrap();
    assert_eq!(hit.restored_tokens(), 8192);
    assert_eq!(hit.suffix(), &request[8192..]);
    assert_eq!(hit.final_input_tokens(), 12289);
    assert_eq!(hit.output_reserve(), 128);
}

#[test]
fn input_prefix_changes_at_any_position_are_misses_not_gdn_truncations() {
    let value = descriptor(8192);
    for position in [0, 4096, 8191] {
        let mut request = value.consumed_input.clone();
        request[position] = 8;
        request.push(9);
        assert!(value
            .lookup_suffix(&value.bindings, &value.math, &request, 128)
            .unwrap()
            .is_none());
    }
    assert!(value
        .lookup_suffix(
            &value.bindings,
            &value.math,
            &value.consumed_input[..4096],
            128
        )
        .unwrap()
        .is_none());
}

#[test]
fn input_prefix_no_suffix_misses_instead_of_replaying_cached_logits() {
    let value = descriptor(8192);
    assert!(value
        .lookup_suffix(&value.bindings, &value.math, &value.consumed_input, 128)
        .unwrap()
        .is_none());
}

#[test]
fn input_prefix_reserve_and_vocabulary_are_checked_before_any_submission() {
    let value = descriptor(CAPACITY - 3);
    let mut request = value.consumed_input.clone();
    request.push(9);
    assert!(value
        .lookup_suffix(&value.bindings, &value.math, &request, 2)
        .unwrap()
        .is_some());
    for reserve in [0, 3, usize::MAX] {
        assert!(value
            .lookup_suffix(&value.bindings, &value.math, &request, reserve)
            .is_err());
    }
    *request.last_mut().unwrap() = VOCAB;
    assert!(value
        .lookup_suffix(&value.bindings, &value.math, &request, 1)
        .is_err());
}

#[test]
fn input_prefix_foreign_identity_or_damaged_metadata_is_not_a_normal_miss() {
    let mut value = descriptor(8);
    let request = vec![7; 16];
    let mut other = value.bindings.clone();
    other.namespace_sha256 = sha256(b"another-user");
    assert!(value
        .lookup_suffix(&other, &value.math, &request, 128)
        .is_err());
    value.features.rows = 9;
    assert!(value
        .lookup_suffix(&value.bindings, &value.math, &request, 128)
        .is_err());
}

#[test]
fn input_prefix_transfer_binding_changes_with_input_not_only_extent() {
    let mut value = descriptor(8);
    let before = value.transfer_plan().unwrap();
    value.consumed_input[3] = 8;
    let after = value.transfer_plan().unwrap();
    assert_eq!(before.components, after.components);
    assert_ne!(before.binding_sha256, after.binding_sha256);
}

#[test]
fn input_prefix_empty_request_is_invalid_not_a_cache_miss() {
    let value = descriptor(8);
    assert!(value
        .lookup_suffix(&value.bindings, &value.math, &[], 128)
        .is_err());
}

#[test]
fn input_prefix_descriptor_bound_precedes_json_decode() {
    let value = descriptor(8);
    let bytes = vec![b' '; MAX_DESCRIPTOR_BYTES + 1];
    assert!(InputPrefixDescriptor::decode(&bytes, &value.bindings, &value.math).is_err());
}
