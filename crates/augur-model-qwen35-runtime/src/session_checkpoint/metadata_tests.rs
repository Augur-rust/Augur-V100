use super::*;

fn bindings() -> ModelBindings {
    ModelBindings {
        model_sha256: sha256(b"model"),
        q8_draft_sha256: sha256(b"draft"),
        vision_sha256: sha256(b"vision"),
        vision_payload_sha256: sha256(b"vision resident payload"),
        model_config_sha256: sha256(b"model config"),
        draft_config_sha256: sha256(b"draft config"),
        tokenizer_sha256: sha256(b"tokenizer"),
        chat_template_sha256: sha256(b"chat template"),
        generation_config_sha256: sha256(b"generation"),
        frozen_build_sha256: sha256(b"build"),
        numerical_contract_sha256: sha256(b"numerics"),
        state_layout_sha256: sha256(b"layout"),
        namespace_sha256: sha256(b"namespace"),
    }
}

fn initial() -> NativeLedger {
    NativeLedger {
        origin_session_id: "old-session".into(),
        consumed: vec![1, 2, 3, 10],
        generated: vec![10, 11],
        user_suffix_ranges: vec![],
        pending: 11,
        pending_published: true,
        generation: 2,
        revision: 1,
        capacity: CAPACITY,
    }
}

fn append(ledger: &mut NativeLedger, suffix: &[u32], prediction: u32) {
    if ledger.pending_published {
        ledger.consumed.push(ledger.pending);
    }
    let start = ledger.consumed.len();
    ledger.consumed.extend_from_slice(suffix);
    ledger.user_suffix_ranges.push(UserSuffixRange {
        start,
        end: ledger.consumed.len(),
    });
    ledger.pending = prediction;
    ledger.pending_published = false;
    ledger.generation += 1;
    ledger.revision += 1;
}

fn decode(ledger: &mut NativeLedger, emitted: &[u32]) {
    assert!(!ledger.pending_published && !emitted.is_empty());
    ledger.generated.push(ledger.pending);
    ledger.consumed.push(ledger.pending);
    ledger
        .consumed
        .extend_from_slice(&emitted[..emitted.len() - 1]);
    ledger.generated.extend_from_slice(emitted);
    ledger.pending = *emitted.last().unwrap();
    ledger.pending_published = true;
    ledger.generation += 1;
    ledger.revision += 1;
}

fn descriptor() -> SessionDescriptor {
    SessionDescriptor {
        schema: SESSION_SCHEMA.into(),
        bindings: bindings(),
        ledger: initial(),
        features: FeatureExtent {
            start: 0,
            rows: 4,
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
        eos_token_ids: vec![248044, 248046],
        native_request_identity: sha256(b"original native identity"),
    }
}

fn at_cursor(cursor: usize) -> SessionDescriptor {
    let mut value = descriptor();
    value.ledger.consumed = vec![1; cursor];
    value.ledger.generated = vec![11];
    value.features.rows = cursor.min(HISTORY);
    value.features.start = cursor - value.features.rows;
    value
}

#[test]
fn rotated_kv_basis_is_explicit_and_never_admitted_as_legacy_payload() {
    let old = descriptor();
    let legacy = serde_json::to_value(&old).unwrap();
    assert!(legacy["math"].get("full_attention_kv_basis").is_none());
    let decoded: SessionDescriptor = serde_json::from_value(legacy.clone()).unwrap();
    assert!(decoded.math.full_attention_kv_basis.is_unrotated());
    let mut rotated = old.clone();
    rotated.math.full_attention_kv_basis = FullAttentionKvBasis::from_rotated(true);
    rotated.validate().unwrap();
    let encoded = serde_json::to_value(&rotated).unwrap();
    assert_eq!(encoded["math"]["full_attention_kv_basis"], "hadamard256_key_hadamard64_value_v1");
    assert_eq!(serde_json::from_value::<SessionDescriptor>(encoded.clone()).unwrap(), rotated);
    let admission = NativeRestoreAdmission {
        new_session_id: "fresh-rotated-binding-test".into(),
        bindings: old.bindings.clone(), math: old.math.clone(),
    };
    assert!(admission.validate_descriptor(&rotated).is_err());
    let mut unknown = encoded;
    unknown["math"]["full_attention_kv_basis"] = serde_json::json!("unknown_same_extent_basis");
    assert!(serde_json::from_value::<SessionDescriptor>(unknown).is_err());
}

#[test]
fn initial_pending_output_is_not_a_consumed_kv_token() {
    let ledger = initial();
    assert_eq!(ledger.validate().unwrap(), 3);
    assert_eq!(ledger.consumed, [1, 2, 3, 10]);
    assert_eq!(ledger.pending, 11);
}

#[test]
fn planar_kv_basis_is_distinct_even_with_identical_position_extents() {
    let mut old = descriptor();
    old.math.full_attention_kv_basis = FullAttentionKvBasis::from_rotated(true);
    let mut planar = old.clone();
    planar.math.full_attention_kv_basis = FullAttentionKvBasis::from_layout(true, true);
    assert!(planar.math.full_attention_kv_basis.is_planar());
    assert!(planar.math.full_attention_kv_basis.is_rotated());
    assert!(!old.math.full_attention_kv_basis.is_planar());
    planar.validate().unwrap();
    let encoded = serde_json::to_value(&planar).unwrap();
    assert_eq!(encoded["math"]["full_attention_kv_basis"],
        "hadamard256_key_hadamard64_value_planar_v2");
    assert_eq!(serde_json::from_value::<SessionDescriptor>(encoded).unwrap(), planar);
    let admission = NativeRestoreAdmission {
        new_session_id: "fresh-planar-binding-test".into(),
        bindings: old.bindings.clone(), math: old.math.clone(),
    };
    assert!(admission.validate_descriptor(&planar).is_err());
}

#[test]
fn suffix_then_decode_keeps_forced_and_generated_ledgers_separate() {
    let mut ledger = initial();
    append(&mut ledger, &[22, 23], 31);
    assert_eq!(ledger.validate().unwrap(), 3);
    assert_eq!(ledger.consumed, [1, 2, 3, 10, 11, 22, 23]);
    assert_eq!(ledger.generated, [10, 11]);
    decode(&mut ledger, &[32, 33]);
    assert_eq!(ledger.validate().unwrap(), 3);
    assert_eq!(ledger.consumed, [1, 2, 3, 10, 11, 22, 23, 31, 32]);
    assert_eq!(ledger.generated, [10, 11, 31, 32, 33]);
    assert_eq!(ledger.pending, 33);
}

#[test]
fn consecutive_suffixes_do_not_consume_an_unpublished_prediction() {
    let mut ledger = initial();
    append(&mut ledger, &[22], 31);
    append(&mut ledger, &[23], 41);
    assert_eq!(ledger.validate().unwrap(), 3);
    assert!(!ledger.consumed.contains(&31));
    assert_eq!(
        ledger.user_suffix_ranges,
        [
            UserSuffixRange { start: 5, end: 6 },
            UserSuffixRange { start: 6, end: 7 }
        ]
    );
}

#[test]
fn multiple_turns_reconstruct_the_same_original_prompt() {
    let mut ledger = initial();
    for i in 0..12 {
        append(&mut ledger, &[100 + i, 200 + i], 300 + i);
        decode(&mut ledger, &[400 + i, 500 + i]);
        assert_eq!(ledger.validate().unwrap(), 3);
    }
}

#[test]
fn mismatched_consumed_generated_and_pending_fail_closed() {
    let mut ledger = initial();
    ledger.consumed[3] = 9;
    assert!(ledger.validate().is_err());
    let mut ledger = initial();
    ledger.pending = 9;
    assert!(ledger.validate().is_err());
    let mut ledger = initial();
    ledger.generated[0] = 9;
    assert!(ledger.validate().is_err());
    let mut ledger = initial();
    ledger.pending_published = false;
    assert!(ledger.validate().is_err());
}

#[test]
fn suffix_overlap_order_original_prompt_and_extent_drift_reject() {
    let mut base = initial();
    append(&mut base, &[22, 23], 31);
    append(&mut base, &[24], 41);
    let mut overlap = base.clone();
    overlap.user_suffix_ranges[1].start -= 1;
    assert!(overlap.validate().is_err());
    let mut order = base.clone();
    order.user_suffix_ranges.swap(0, 1);
    assert!(order.validate().is_err());
    let mut prompt = base.clone();
    prompt.user_suffix_ranges[0].start = 0;
    assert!(prompt.validate().is_err());
    let mut end = base;
    end.user_suffix_ranges[1].end = usize::MAX;
    assert!(end.validate().is_err());
}

#[test]
fn token_vocabulary_counter_and_capacity_bounds_reject() {
    let mut value = initial();
    value.pending = VOCAB;
    assert!(value.validate().is_err());
    let mut value = initial();
    value.consumed[0] = VOCAB;
    assert!(value.validate().is_err());
    let mut value = initial();
    value.generated[0] = VOCAB;
    assert!(value.validate().is_err());
    for count in [0, u64::MAX] {
        let mut value = initial();
        value.generation = count;
        assert!(value.validate().is_err());
        let mut value = initial();
        value.revision = count;
        assert!(value.validate().is_err());
    }
    let mut value = initial();
    value.capacity += 1;
    assert!(value.validate().is_err());
}

#[test]
fn descriptor_roundtrip_binds_actual_expected_policy() {
    let value = descriptor();
    let bytes = value.encode().unwrap();
    assert_eq!(
        SessionDescriptor::decode(&bytes, &value.bindings, &value.math).unwrap(),
        value
    );
    let mut policy = value.math.clone();
    policy.exact_projection_tiles = false;
    assert!(SessionDescriptor::decode(&bytes, &value.bindings, &policy).is_err());
}

#[test]
fn every_external_binding_axis_is_independently_checked() {
    let value = descriptor();
    let bytes = value.encode().unwrap();
    let binding_value = serde_json::to_value(&value.bindings).unwrap();
    for name in binding_value.as_object().unwrap().keys() {
        let mut mutated = binding_value.clone();
        mutated[name] = serde_json::Value::String(sha256(b"other"));
        let expected: ModelBindings = serde_json::from_value(mutated).unwrap();
        assert!(
            SessionDescriptor::decode(&bytes, &expected, &value.math).is_err(),
            "{name}"
        );
    }
}

#[test]
fn no_fake_feature_rows_at_short_input_or_rollover() {
    for cursor in [1, 7, 8, 2047, 2048, 2049, 8192, 194560, CAPACITY] {
        let value = at_cursor(cursor);
        value.validate().unwrap();
        let plan = value.transfer_plan().unwrap();
        assert_eq!(
            plan.components.last().unwrap().bytes,
            cursor.min(HISTORY) as u64 * FEATURE_ROW_BYTES
        );
        let mut bad = value;
        bad.features.rows += 1;
        assert!(bad.validate().is_err());
    }
}

#[test]
fn fixed_history_and_growing_generation_are_explicit() {
    let mut short = descriptor();
    short.features.mode = HistoryMode::Fixed2048;
    assert!(short.validate().is_err());
    let mut full = at_cursor(8192);
    full.features.mode = HistoryMode::Fixed2048;
    full.validate().unwrap();
    full.features.mode = HistoryMode::Growing2048 {
        generation: u64::MAX,
    };
    assert!(full.validate().is_err());
}

#[test]
fn full_target_inventory_has_32_kv_96_linear_and_one_initialized_ring() {
    let value = at_cursor(194560);
    let plan = value.transfer_plan().unwrap();
    assert_eq!(plan.components.len(), 130);
    assert_eq!(plan.components[0].name, "session.metadata");
    assert_eq!(
        plan.components[0].bytes,
        value.encode().unwrap().len() as u64
    );
    let kv: Vec<_> = plan
        .components
        .iter()
        .filter(|p| p.name.ends_with(".key") || p.name.ends_with(".value"))
        .collect();
    let linear: Vec<_> = plan
        .components
        .iter()
        .filter(|p| p.name.ends_with(".conv_history") || p.name.ends_with(".recurrent"))
        .collect();
    assert_eq!(kv.len(), 32);
    assert_eq!(linear.len(), 96);
    assert_eq!(kv.iter().map(|p| p.bytes).sum::<u64>(), 6_773_800_960);
    assert_eq!(linear.iter().map(|p| p.bytes).sum::<u64>(), 156_893_184);
    assert_eq!(plan.components[129].bytes, 209_715_200);
    assert_eq!(
        plan.components.iter().skip(1).map(|p| p.bytes).sum::<u64>(),
        7_140_409_344
    );
    assert!(!plan
        .components
        .iter()
        .any(|p| p.name.contains("scratch") || p.name.contains("cuda")));
}

#[test]
fn full_attention_never_serializes_unwritten_capacity() {
    let value = at_cursor(8192);
    let plan = value.transfer_plan().unwrap();
    let key = plan
        .components
        .iter()
        .find(|p| p.name == "target.layer03.key")
        .unwrap();
    assert_eq!(key.bytes, 8192 * KV_ROW_BYTES);
    assert!(key.bytes < CAPACITY as u64 * KV_ROW_BYTES);
}

#[test]
fn every_metadata_change_rebinds_the_transfer_plan() {
    let value = descriptor();
    let base = value.transfer_plan().unwrap().binding_sha256;
    let mut changed = value.clone();
    changed.ledger.revision += 1;
    assert_ne!(changed.transfer_plan().unwrap().binding_sha256, base);
    let mut changed = value;
    changed.eos_token_ids.reverse();
    assert_ne!(changed.transfer_plan().unwrap().binding_sha256, base);
}

#[test]
fn unknown_claim_keys_recipes_and_sampling_modes_are_rejected() {
    let value = descriptor();
    for (key, data) in [
        ("product_ready", serde_json::Value::Bool(true)),
        ("draft_restore", "reuse_old_cuda_pointers".into()),
        ("sampling", "stochastic_without_rng".into()),
    ] {
        let mut object = serde_json::to_value(&value).unwrap();
        object[key] = data;
        assert!(SessionDescriptor::decode(
            &serde_json::to_vec(&object).unwrap(),
            &value.bindings,
            &value.math
        )
        .is_err());
    }
}

#[test]
fn strict_parse_rejects_duplicate_fields_noncanonical_json_and_size_overflow() {
    let value = descriptor();
    let bytes = value.encode().unwrap();
    let text = String::from_utf8(bytes.clone()).unwrap();
    let duplicate = text.replacen("{", &format!("{{\"schema\":\"{SESSION_SCHEMA}\","), 1);
    assert!(SessionDescriptor::decode(duplicate.as_bytes(), &value.bindings, &value.math).is_err());
    let mut padded = bytes;
    padded.push(b'\n');
    assert!(SessionDescriptor::decode(&padded, &value.bindings, &value.math).is_err());
    assert!(SessionDescriptor::decode(
        &vec![b' '; MAX_DESCRIPTOR_BYTES + 1],
        &value.bindings,
        &value.math
    )
    .is_err());
}

#[test]
fn eos_source_identity_and_required_artifact_digest_are_checked() {
    let mut value = descriptor();
    value.eos_token_ids.push(value.eos_token_ids[0]);
    assert!(value.validate().is_err());
    let mut value = descriptor();
    value.native_request_identity.clear();
    assert!(value.validate().is_err());
    let mut value = descriptor();
    value.bindings.q8_draft_sha256 = "0".repeat(64);
    assert!(value.validate().is_err());
}

#[test]
fn descriptor_only_package_cannot_substitute_for_complete_target_state() {
    let value = descriptor();
    let expected = value.transfer_plan().unwrap();
    let mut partial = expected.clone();
    partial.components.truncate(1);
    let payload = value.encode().unwrap();
    let manifest = crate::session_checkpoint::Manifest {
        schema: crate::session_checkpoint::SCHEMA.into(),
        plan: partial.clone(),
        chunk_bytes: crate::session_checkpoint::MAX_CHUNK_BYTES,
        chunks: vec![crate::session_checkpoint::Chunk {
            component: 0,
            offset: 0,
            bytes: payload.len(),
            sha256: crate::session_checkpoint::chunk_digest(&partial.components[0], 0, &payload),
        }],
    };
    let bytes = manifest.encode().unwrap();
    assert!(
        crate::session_checkpoint::Manifest::decode(&bytes, &sha256(&bytes), &expected).is_err()
    );
}

#[test]
fn native_restore_requires_new_handle_and_independent_admission() {
    let descriptor = descriptor();
    let mut admission = NativeRestoreAdmission {
        new_session_id: "fresh-session".into(),
        bindings: descriptor.bindings.clone(),
        math: descriptor.math.clone(),
    };
    admission.validate_descriptor(&descriptor).unwrap();
    for id in ["", "../other", "old-session", "with space", "会话"] {
        admission.new_session_id = id.into();
        assert!(admission.validate_descriptor(&descriptor).is_err(), "{id}");
    }
    admission.new_session_id = "x".repeat(65);
    assert!(admission.validate_descriptor(&descriptor).is_err());
    admission.new_session_id = "fresh-session".into();
    admission.bindings.namespace_sha256 = sha256(b"another tenant");
    assert!(admission.validate_descriptor(&descriptor).is_err());
    admission.bindings = descriptor.bindings.clone();
    admission.math.parallel_receipt_digest = !admission.math.parallel_receipt_digest;
    assert!(admission.validate_descriptor(&descriptor).is_err());
}

#[test]
fn restore_admission_never_authorizes_an_invalid_saved_ledger() {
    let mut descriptor = descriptor();
    let admission = NativeRestoreAdmission {
        new_session_id: "fresh-session".into(),
        bindings: descriptor.bindings.clone(),
        math: descriptor.math.clone(),
    };
    descriptor.ledger.consumed.push(descriptor.ledger.pending);
    assert!(admission.validate_descriptor(&descriptor).is_err());
}
