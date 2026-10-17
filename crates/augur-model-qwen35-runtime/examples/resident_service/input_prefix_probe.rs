//! Bounded real input-only cold/save/fresh-restore/suffix comparison. This probe
//! can optionally compare real generated turns, but never qualifies HTTP hits.
use super::session_smoke::save;
use super::*;
use augur_model_qwen35_runtime::session_checkpoint::{
    filesystem::http::PinnedHttpCheckpointProfile,
    input_prefix::{InputPrefixGenerationAdmission, InputPrefixRestoreAdmission},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Spec {
    schema: String,
    prefix_tokens: usize,
    cases: Vec<Case>,
    #[serde(default)]
    max_new_tokens: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    prompt: String,
}

fn generate(
    owner: &mut Qwen35ResidentDraftCuda,
    boundary: &Value,
    config: &Qwen35AutoregressiveCudaConfig,
    generation: &std::path::Path,
    admission: &InputPrefixGenerationAdmission<'_>,
    directory: &std::path::Path,
    label: &str,
) -> Result<Value, Box<dyn Error>> {
    let delivered = RefCell::new(Vec::<u32>::new());
    let sink: &Qwen35VerifiedCommitSink<'_> = &|commit| {
        delivered.borrow_mut().extend_from_slice(&commit.token_ids);
        Ok(())
    };
    let started = Instant::now();
    let generated =
        owner.generate_native_input_prefix(boundary, config, generation, admission, sink)?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    save(
        &directory.join(format!("{label}-generation.json")),
        &generated,
    )?;
    if json!(*delivered.borrow()) != generated["request_output"]["token_ids"]
        || generated["transactions"]
            .as_array()
            .is_none_or(|t| t.is_empty())
    {
        return Err(
            "input-prefix generation needs real draft verification and exact delivered ledger"
                .into(),
        );
    }
    let state = owner.audit_native_session_checkpoint(&generated["session_checkpoint"])?;
    save(
        &directory.join(format!("{label}-generated-state.json")),
        &state,
    )?;
    let mut persistence = Value::Null;
    if label == "hit" {
        let descriptor = owner.describe_native_session_checkpoint(
            &generated["session_checkpoint"],
            admission.bindings,
        )?;
        let checkpoint = directory.join("generated-checkpoint");
        let saved = owner.save_native_session_checkpoint(
            &generated["session_checkpoint"],
            admission.bindings.clone(),
            &checkpoint,
        )?;
        save(&directory.join("generated-save.json"), &saved)?;
        owner.close_native_session(&generated["session_checkpoint"])?;
        let restored = owner.restore_native_session_checkpoint(
            config,
            generation,
            &checkpoint,
            saved["manifest_sha256"]
                .as_str()
                .ok_or("generated manifest absent")?,
            &augur_model_qwen35_runtime::session_checkpoint::metadata::NativeRestoreAdmission {
                new_session_id: format!("restored-{}", admission.session_id),
                bindings: admission.bindings.clone(),
                math: descriptor.math,
            },
        )?;
        save(&directory.join("generated-restore.json"), &restored)?;
        for key in [
            "cursor",
            "capacity",
            "pending_token",
            "consumed_input_digest",
            "target_committed_state_sha256",
            "feature_bytes_sha256",
            "state_plane_hashes",
            "initialized_history",
        ] {
            if state[key].is_null() || restored["checkpoint"][key] != state[key] {
                return Err(format!("generated checkpoint restore mismatch: {key}").into());
            }
        }
        owner.close_native_session(&restored["checkpoint"])?;
        persistence = json!({"generated_save_restore_exact":true,"prefix_recompute_tokens":restored["prefix_recompute_tokens"]});
    } else {
        owner.close_native_session(&generated["session_checkpoint"])?;
    }
    Ok(json!({"report":generated,"state":state,"call_wall_ms":wall_ms,"persistence":persistence}))
}

fn compare_generated(a: &Value, b: &Value) -> Result<(), Box<dyn Error>> {
    for key in [
        "token_ids",
        "visible_token_ids",
        "decoded_text",
        "stop_reason",
    ] {
        let left = &a["report"]["request_output"][key];
        let right = &b["report"]["request_output"][key];
        if left.is_null() || right.is_null() || left != right {
            return Err(format!("cold/hit generated output mismatch: {key}").into());
        }
    }
    for key in [
        "cursor",
        "capacity",
        "pending_token",
        "consumed_input_digest",
        "target_committed_state_sha256",
        "feature_bytes_sha256",
        "target_planes",
        "state_plane_hashes",
        "initialized_history",
    ] {
        if a["state"][key].is_null()
            || b["state"][key].is_null()
            || a["state"][key] != b["state"][key]
        {
            return Err(format!("cold/hit generated full state mismatch: {key}").into());
        }
    }
    Ok(())
}

fn compare(a: &Value, b: &Value, anchor: bool) -> Result<(), Box<dyn Error>> {
    for key in [
        "cursor",
        "capacity",
        "consumed_input_digest",
        "target_committed_state_sha256",
        "feature_bytes_sha256",
        "target_planes",
        "feature_bytes",
        "initialized_history",
        "state_plane_hashes",
    ] {
        if a[key].is_null() || b[key].is_null() || a[key] != b[key] {
            return Err(format!("input-prefix exact state mismatch: {key}").into());
        }
    }
    if anchor
        && (a["next_token_computed_for_this_input"] != true
            || b["next_token_computed_for_this_input"] != true
            || a["next_token_diagnostic"].is_null()
            || a["next_token_diagnostic"] != b["next_token_diagnostic"])
    {
        return Err("input-prefix fresh target prediction mismatch".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audit() -> Value {
        json!({"cursor":8193,"capacity":196608,"consumed_input_digest":"inputs",
            "target_committed_state_sha256":"target","feature_bytes_sha256":"features",
            "target_planes":128,"feature_bytes":209715200,
            "initialized_history":{"start_position":6145,"active_rows":2048},
            "state_plane_hashes":[{"sha256":"one-plane"}],
            "next_token_computed_for_this_input":true,"next_token_diagnostic":42})
    }

    #[test]
    fn input_prefix_audit_requires_every_initialized_state_axis() {
        let a = audit();
        assert!(compare(&a, &a, true).is_ok());
        for key in [
            "cursor",
            "capacity",
            "consumed_input_digest",
            "target_committed_state_sha256",
            "feature_bytes_sha256",
            "target_planes",
            "feature_bytes",
            "initialized_history",
            "state_plane_hashes",
        ] {
            let mut changed = a.clone();
            changed[key] = Value::Null;
            assert!(compare(&a, &changed, true).is_err(), "{key}");
        }
    }

    #[test]
    fn input_prefix_audit_does_not_count_an_uncomputed_prediction() {
        let a = audit();
        let mut restored = a.clone();
        restored["next_token_computed_for_this_input"] = json!(false);
        restored["next_token_diagnostic"] = Value::Null;
        assert!(compare(&a, &restored, false).is_ok());
        assert!(compare(&a, &restored, true).is_err());
        restored["next_token_computed_for_this_input"] = json!(true);
        restored["next_token_diagnostic"] = json!(43);
        assert!(compare(&a, &restored, true).is_err());
    }

    #[test]
    fn input_prefix_spec_rejects_unknown_authority() {
        let value = json!({"schema":"qwen35-native-input-prefix-probe-v1","prefix_tokens":8192,
            "cases":[{"id":"cn","prompt":"test"}],"qualified":true});
        assert!(serde_json::from_value::<Spec>(value).is_err());
    }

    #[test]
    fn generated_comparison_rejects_output_stop_and_full_state_drift() {
        let mut state = audit();
        state["pending_token"] = json!(12);
        let a = json!({"report":{"request_output":{"token_ids":[11,12],"visible_token_ids":[11,12],
            "decoded_text":"answer","stop_reason":"max_new_tokens"}},"state":state});
        assert!(compare_generated(&a, &a).is_ok());
        for key in [
            "token_ids",
            "visible_token_ids",
            "decoded_text",
            "stop_reason",
        ] {
            let mut b = a.clone();
            b["report"]["request_output"][key] = Value::Null;
            assert!(compare_generated(&a, &b).is_err());
        }
        for key in [
            "pending_token",
            "consumed_input_digest",
            "target_committed_state_sha256",
            "feature_bytes_sha256",
        ] {
            let mut b = a.clone();
            b["state"][key] = json!("changed");
            assert!(compare_generated(&a, &b).is_err());
        }
    }
}

fn retained_clean(
    expected: augur_backend::CudaDeviceAllocationSnapshot,
) -> Result<(), Box<dyn Error>> {
    let now = augur_backend::cuda_device_allocation_snapshot();
    if now.live_bytes != expected.live_bytes || now.live_allocations != expected.live_allocations {
        return Err("input-prefix mutable owner leak".into());
    }
    Ok(())
}

pub(crate) fn run(
    config: Qwen35AutoregressiveCudaConfig,
    q8: PathBuf,
    draft: PathBuf,
    generation: PathBuf,
    tokenizer: Tokenizer,
    evidence: PathBuf,
    specification: PathBuf,
    profile_input: (PathBuf, String),
) -> Result<(), Box<dyn Error>> {
    if !evidence.is_absolute()
        || evidence.exists()
        || !specification.is_absolute()
        || fs::metadata(&specification)?.len() > 8 * 1024 * 1024
    {
        return Err("input-prefix fresh evidence/bounded specification required".into());
    }
    let bytes = fs::read(&specification)?;
    let spec: Spec = serde_json::from_slice(&bytes)?;
    if spec.schema != "qwen35-native-input-prefix-probe-v1"
        || !(1..=16384).contains(&spec.prefix_tokens)
        || !(2..=4).contains(&spec.cases.len())
        || (spec.max_new_tokens != 0 && !(8..=128).contains(&spec.max_new_tokens))
    {
        return Err("input-prefix first gate needs bounded multi-suffix specification".into());
    }
    let profile = PinnedHttpCheckpointProfile::open(&profile_input.0, &profile_input.1)?;
    let bindings = &profile.profile.bindings;
    let math = &profile.profile.math;
    if bindings.tokenizer_sha256 != digest(&fs::read(&config.tokenizer_path)?)
        || augur_model_qwen35_runtime::build_identity::identity()["source_dirty"] != false
        || bindings.frozen_build_sha256
            != digest(&serde_json::to_vec(
                &augur_model_qwen35_runtime::build_identity::identity(),
            )?)
    {
        return Err("input-prefix profile/live tokenizer or frozen identity mismatch".into());
    }
    let mut ids = BTreeSet::new();
    let mut inputs = Vec::new();
    for case in &spec.cases {
        if case.id.is_empty()
            || case.id.len() > 48
            || !case
                .id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
            || !ids.insert(case.id.clone())
            || case.prompt.is_empty()
            || case.prompt.len() > 2 * 1024 * 1024
        {
            return Err("input-prefix case label/text bound".into());
        }
        let input = tokenizer
            .encode(case.prompt.as_str(), false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec();
        if input.len() <= spec.prefix_tokens || input.len() > 16384 {
            return Err("input-prefix first-gate input extent".into());
        }
        inputs.push(input);
    }
    let prefix = &inputs[0][..spec.prefix_tokens];
    if inputs.iter().any(|input| !input.starts_with(prefix)) {
        return Err("prompts lack the declared exact shared token prefix".into());
    }
    fs::create_dir(&evidence)?;
    save(
        &evidence.join("inputs.json"),
        &json!({"specification_sha256":digest(&bytes),
        "profile_sha256":profile_input.1,"prefix_tokens":prefix.len(),
        "cases":spec.cases.iter().zip(&inputs).map(|(c,t)| json!({"id":c.id,"tokens":t.len(),
            "prompt_sha256":digest(c.prompt.as_bytes()),"token_ids":t})).collect::<Vec<_>>(),
        "product_ready":false}),
    )?;
    let baseline = augur_backend::cuda_device_allocation_snapshot();
    let started = Instant::now();
    let mut owner = Qwen35ResidentDraftCuda::open_with_target_and_vision(&config, &q8, &draft)?;
    let retained = augur_backend::cuda_device_allocation_snapshot();
    save(
        &evidence.join("ready.json"),
        &json!({"owner_open_count":1,
        "owner_open_wall_ms":started.elapsed().as_secs_f64()*1000.0,"resident":allocation_json(retained),
        "vision_forward":false,"product_ready":false}),
    )?;
    let outcome = (|| -> Result<Value, Box<dyn Error>> {
        let prefix_prepare =
            owner.prepare_native_input_prefix(&config, prefix, bindings, options(&generation)?)?;
        save(&evidence.join("prefix-prepare.json"), &prefix_prepare)?;
        let boundary = &prefix_prepare["boundary"];
        let descriptor = owner.describe_native_input_prefix(boundary, bindings)?;
        if descriptor.math != *math {
            return Err("input-prefix live/profile math mismatch".into());
        }
        let prefix_audit = owner.audit_native_input_prefix_state(boundary, bindings)?;
        save(&evidence.join("prefix-audit.json"), &prefix_audit)?;
        let directory = evidence.join("checkpoint");
        let capture = owner.save_native_input_prefix(boundary, bindings.clone(), &directory)?;
        save(&evidence.join("prefix-capture.json"), &capture)?;
        let manifest = capture["manifest_sha256"]
            .as_str()
            .ok_or("input-prefix manifest absent")?;
        owner.close_native_input_prefix(boundary)?;
        retained_clean(retained)?;
        let mut misses = Vec::new();
        for index in [0, prefix.len() / 2, prefix.len() - 1] {
            let mut changed = inputs[0].clone();
            changed[index] = (changed[index] + 1) % 248320;
            misses.push(changed);
        }
        misses.push(prefix.to_vec());
        if prefix.len() > 1 {
            misses.push(prefix[..prefix.len() - 1].to_vec());
        }
        for input in &misses {
            let allocation_before = augur_backend::cuda_device_allocation_snapshot();
            let host_before = augur_backend::cuda_host_boundary_snapshot();
            if owner
                .restore_native_input_prefix(
                    &config,
                    &generation,
                    &directory,
                    manifest,
                    &InputPrefixRestoreAdmission {
                        bindings,
                        math,
                        new_request_id: "ordinary-miss",
                        request_input: input,
                        output_reserve: 134,
                    },
                )?
                .is_some()
                || allocation_before != augur_backend::cuda_device_allocation_snapshot()
                || host_before != augur_backend::cuda_host_boundary_snapshot()
            {
                return Err("input-prefix ordinary miss allocated or submitted CUDA".into());
            }
        }
        save(
            &evidence.join("ordinary-misses.json"),
            &json!({"cases":misses.len(),
            "all_missed_before_cuda":true,"product_ready":false}),
        )?;
        let mut results = Vec::new();
        for (case, tokens) in spec.cases.iter().zip(&inputs) {
            let case_dir = evidence.join(&case.id);
            fs::create_dir(&case_dir)?;
            let cold_start = Instant::now();
            let cold = owner.prepare_native_input_prefix(
                &config,
                tokens,
                bindings,
                options(&generation)?,
            )?;
            let cold_wall_ms = cold_start.elapsed().as_secs_f64() * 1000.0;
            save(&case_dir.join("cold.json"), &cold)?;
            let cold_audit = owner.audit_native_input_prefix_state(&cold["boundary"], bindings)?;
            save(&case_dir.join("cold-audit.json"), &cold_audit)?;
            let cold_generated = if spec.max_new_tokens > 0 {
                Some(generate(
                    &mut owner,
                    &cold["boundary"],
                    &config,
                    &generation,
                    &InputPrefixGenerationAdmission {
                        bindings,
                        session_id: &format!("cold-{}", case.id),
                        max_new_tokens: spec.max_new_tokens,
                    },
                    &case_dir,
                    "cold",
                )?)
            } else {
                owner.close_native_input_prefix(&cold["boundary"])?;
                None
            };
            retained_clean(retained)?;
            let hit_start = Instant::now();
            let restored = owner
                .restore_native_input_prefix(
                    &config,
                    &generation,
                    &directory,
                    manifest,
                    &InputPrefixRestoreAdmission {
                        bindings,
                        math,
                        new_request_id: &case.id,
                        request_input: tokens,
                        output_reserve: 134,
                    },
                )?
                .ok_or("exact input-prefix unexpectedly missed")?;
            let restore_wall_ms = hit_start.elapsed().as_secs_f64() * 1000.0;
            save(&case_dir.join("restore.json"), &restored)?;
            let restored_audit =
                owner.audit_native_input_prefix_state(&restored["boundary"], bindings)?;
            save(&case_dir.join("restore-audit.json"), &restored_audit)?;
            compare(&prefix_audit, &restored_audit, false)?;
            if spec.max_new_tokens > 0 {
                let before = augur_backend::cuda_device_allocation_snapshot();
                let host_before = augur_backend::cuda_host_boundary_snapshot();
                let sink: &Qwen35VerifiedCommitSink<'_> = &|_| {
                    Err(Qwen35MixedPrefixCudaError::InvalidConfiguration(
                        "uncomputed anchor was published".into(),
                    ))
                };
                if owner
                    .generate_native_input_prefix(
                        &restored["boundary"],
                        &config,
                        &generation,
                        &InputPrefixGenerationAdmission {
                            bindings,
                            session_id: "no-old-logits",
                            max_new_tokens: spec.max_new_tokens,
                        },
                        sink,
                    )
                    .is_ok()
                    || before != augur_backend::cuda_device_allocation_snapshot()
                    || host_before != augur_backend::cuda_host_boundary_snapshot()
                {
                    return Err("restored input without suffix admitted generation".into());
                }
            }
            if restored_audit["next_token_computed_for_this_input"] != false
                || !restored_audit["next_token_diagnostic"].is_null()
            {
                return Err("input-prefix restore manufactured a prediction".into());
            }
            let suffix_start = Instant::now();
            let suffix = owner
                .append_native_input_prefix(&restored["boundary"], bindings, tokens, 134)?
                .ok_or("restored exact suffix unexpectedly missed")?;
            let suffix_wall_ms = suffix_start.elapsed().as_secs_f64() * 1000.0;
            save(&case_dir.join("suffix.json"), &suffix)?;
            let suffix_audit =
                owner.audit_native_input_prefix_state(&suffix["boundary"], bindings)?;
            save(&case_dir.join("suffix-audit.json"), &suffix_audit)?;
            compare(&cold_audit, &suffix_audit, true)?;
            let generated = if let Some(cold_generated) = &cold_generated {
                let hit = generate(
                    &mut owner,
                    &suffix["boundary"],
                    &config,
                    &generation,
                    &InputPrefixGenerationAdmission {
                        bindings,
                        session_id: &format!("hit-{}", case.id),
                        max_new_tokens: spec.max_new_tokens,
                    },
                    &case_dir,
                    "hit",
                )?;
                compare_generated(cold_generated, &hit)?;
                Some(
                    json!({"cold_generation_wall_ms":cold_generated["call_wall_ms"],"hit_generation_wall_ms":hit["call_wall_ms"],
                    "exact_output_and_final_state":true,"tokens":hit["report"]["request_output"]["token_ids"].as_array().map(Vec::len),
                    "text":hit["report"]["request_output"]["decoded_text"],"persistence":hit["persistence"]}),
                )
            } else {
                owner.close_native_input_prefix(&suffix["boundary"])?;
                None
            };
            retained_clean(retained)?;
            let result = json!({"id":case.id,"input_tokens":tokens.len(),"suffix_tokens":tokens.len()-prefix.len(),
                "cold_prepare_wall_ms":cold_wall_ms,"restore_wall_ms":restore_wall_ms,"suffix_wall_ms":suffix_wall_ms,
                "restore_plus_suffix_wall_ms_excluding_audit":restore_wall_ms+suffix_wall_ms,
                "complete_target_features_and_next_prediction_exact":true,"prefix_recompute_tokens":0,
                "generation_comparison":generated,"cache_hit_qualified":false,"product_ready":false});
            save(&case_dir.join("comparison.json"), &result)?;
            println!(
                "{}",
                json!({"event":"input_prefix_case_complete","id":case.id,"exact":true})
            );
            results.push(result);
        }
        let mut sink_fault = Value::Null;
        if spec.max_new_tokens > 0 {
            let restored = owner
                .restore_native_input_prefix(
                    &config,
                    &generation,
                    &directory,
                    manifest,
                    &InputPrefixRestoreAdmission {
                        bindings,
                        math,
                        new_request_id: "sink-fault",
                        request_input: &inputs[0],
                        output_reserve: 134,
                    },
                )?
                .ok_or("fault restore missed")?;
            let suffix = owner
                .append_native_input_prefix(&restored["boundary"], bindings, &inputs[0], 134)?
                .ok_or("fault suffix missed")?;
            let commits = Cell::new(0usize);
            let sink: &Qwen35VerifiedCommitSink<'_> = &|commit| {
                commits.set(commits.get() + 1);
                if !commit.initial_anchor {
                    Err(Qwen35MixedPrefixCudaError::InvalidConfiguration(
                        "intentional committed sink failure".into(),
                    ))
                } else {
                    Ok(())
                }
            };
            let failure = owner.generate_native_input_prefix(
                &suffix["boundary"],
                &config,
                &generation,
                &InputPrefixGenerationAdmission {
                    bindings,
                    session_id: "sink-fault",
                    max_new_tokens: spec.max_new_tokens,
                },
                sink,
            );
            if failure.is_ok() || commits.get() < 2 {
                return Err("committed sink failure did not execute".into());
            }
            retained_clean(retained)?;
            let before = augur_backend::cuda_device_allocation_snapshot();
            if owner
                .prepare_native_input_prefix(&config, prefix, bindings, options(&generation)?)
                .is_ok()
                || before != augur_backend::cuda_device_allocation_snapshot()
            {
                return Err("failed sink owner reused".into());
            }
            sink_fault = json!({"commits_observed":commits.get(),"error":failure.err().map(|e|e.to_string()),
                "owner_reuse_rejected_before_cuda":true,"mutable_owner_cleanup":true,"product_ready":false});
            save(&evidence.join("committed-sink-failure.json"), &sink_fault)?;
        }
        Ok(
            json!({"cases":results,"exact_input_prefix_payload_and_suffix":true,
            "generation_tested":spec.max_new_tokens>0,"sink_fault":sink_fault,"http_tested":false,"cache_hit_qualified":false}),
        )
    })();
    drop(owner);
    let after = augur_backend::cuda_device_allocation_snapshot();
    let clean = after.live_bytes == baseline.live_bytes
        && after.live_allocations == baseline.live_allocations
        && after.successful_allocations - baseline.successful_allocations
            == after.successful_frees - baseline.successful_frees;
    save(
        &evidence.join("complete.json"),
        &json!({"schema":"qwen35-native-input-prefix-probe-result-v1",
        "success":outcome.is_ok() && clean,"result":outcome.as_ref().ok(),"error":outcome.as_ref().err().map(|e|e.to_string()),
        "cleanup_ok":clean,"before":allocation_json(baseline),"after":allocation_json(after),
        "product_ready":false,"qualified":false,"semantic_qualification":false,"visible_tps":false}),
    )?;
    outcome?;
    if !clean {
        return Err("input-prefix probe cleanup failed".into());
    }
    Ok(())
}
