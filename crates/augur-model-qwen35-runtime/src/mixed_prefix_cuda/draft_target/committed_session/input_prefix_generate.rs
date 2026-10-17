//! Genuine first-turn DFlash generation from verified input state. The input
//! owner is consumed; a CommittedSession exists only after a real terminal.
use super::super::checkpoint_restore::rebuild_draft;
use super::*;
use crate::session_checkpoint::input_prefix::InputPrefixGenerationAdmission;

fn configure_live_driver(d: &mut Driver) -> Result<LiveContinuationPolicy> {
    let policy = LiveContinuationPolicy::from_switches(
        std::env::var_os("AUGUR_QWEN35_FAST_SESSION_PATH").as_deref(),
        std::env::var_os("AUGUR_QWEN35_FAST_FEATURE_IDENTITY").as_deref(),
        std::env::var_os("AUGUR_QWEN35_REPLAY_HEAD_ELISION").as_deref(),
    )?;
    let switches = batched_verify::prefill_policy::Switches::from_env().resolve(
        batched_verify::prefill_policy::Scope {
            canonical_session: true,
            resident: true,
            exact_batched_target: d.exact_projection_tiles,
            vision: true,
            capacity: d.capacity,
            history_rows: d.context_rows,
        },
    )?;
    let row_reuse = batched_verify::projection_row_reuse_switch(
        switches.row_reuse.as_deref(),
        d.exact_projection_tiles,
        true,
    )?;
    let changed = d.projection_row_reuse != row_reuse;
    let executor = d.batch_verify_execution.take();
    d.projection_row_reuse = row_reuse;
    d.batch_verify_execution = rebind_live_projection_executor(
        executor,
        d.batched_suffix || row_reuse,
        changed,
        |executor| executor.rebind_projection_policy(d),
    )?;
    d.fast_feature_identity = policy.fast_feature_identity;
    d.replay_head_elision = policy.replay_head_elision;
    Ok(policy)
}

impl Qwen35ResidentDraftCuda {
    pub fn generate_native_input_prefix(
        &mut self,
        expected: &Value,
        config: &Qwen35AutoregressiveCudaConfig,
        generation_config: &Path,
        admission: &InputPrefixGenerationAdmission<'_>,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        let descriptor = self.describe_native_input_prefix(expected, admission.bindings)?;
        admission
            .validate(&descriptor)
            .map_err(|e| invalid(e.to_string()))?;
        if self.session.is_some()
            || self.input_prefix.as_ref().is_none_or(|p| !p.anchor_ready)
            || config.capacity != CAPACITY
            || config.device_ordinal != 0
            || config.vision_safetensors_path.as_ref() != self.vision_path.as_ref()
        {
            return Err(invalid(
                "input-prefix generation needs fresh real input prediction and exclusive owner",
            ));
        }
        let target = self
            .target
            .as_ref()
            .ok_or_else(|| invalid("input-prefix generation target weights"))?;
        let model_bytes = read_stable_regular_file(&config.config_path, "input generation config")?;
        let tokenizer_bytes =
            read_stable_regular_file(&config.tokenizer_path, "input generation tokenizer")?;
        let generation_bytes = read_stable_regular_file(generation_config, "input generation EOS")?;
        if target.gguf_path != config.gguf_path
            || sha256_hex(&model_bytes) != descriptor.bindings.model_config_sha256
            || sha256_hex(&tokenizer_bytes) != descriptor.bindings.tokenizer_sha256
            || sha256_hex(&generation_bytes) != descriptor.bindings.generation_config_sha256
        {
            return Err(invalid("input-prefix generation live input identity"));
        }
        let model = Qwen35Config::parse(&model_bytes)?;
        model.validate_exact_profile()?;
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer_bytes).map_err(|e| invalid(e.to_string()))?;
        let generation: Value =
            serde_json::from_slice(&generation_bytes).map_err(|e| invalid(e.to_string()))?;
        let eos: Vec<u32> = match &generation["eos_token_id"] {
            Value::Array(_) => serde_json::from_value(generation["eos_token_id"].clone()),
            Value::Number(_) => {
                serde_json::from_value(generation["eos_token_id"].clone()).map(|id| vec![id])
            }
            _ => return Err(invalid("input-prefix EOS field")),
        }
        .map_err(|e| invalid(e.to_string()))?;
        if eos.is_empty()
            || eos.len() > 8
            || !eos.contains(&model.eos_token_id)
            || eos.iter().any(|&id| id as usize >= V)
            || eos.iter().enumerate().any(|(i, id)| eos[..i].contains(id))
        {
            return Err(invalid("input-prefix generation EOS contract"));
        }
        self.poisoned = true;
        let prefix = self
            .input_prefix
            .take()
            .ok_or_else(|| invalid("input-prefix owner disappeared"))?;
        let mut d = prefix.driver;
        d.check_session_cancellation("input-prefix generation before submit")?;
        let started = Instant::now();
        let policy = configure_live_driver(&mut d)?;
        let mut draft = rebuild_draft(&mut d, &self.weights, target, &descriptor.math, false)?;
        // The fresh draft already contains the complete post-suffix feature
        // window. An input-only append's last plan is therefore not a target
        // commit waiting to be imported by this draft. In an anchor-only turn
        // no verification overwrites it; retaining it would replay the same
        // input transition at the next session append and fail identity checks.
        // Do not submit a private draft or invent a verification transaction:
        // the growing-history refresh accepts this real unprepared owner.
        d.last_growth_plan = None;
        let draft_build_ms = started.elapsed().as_secs_f64() * 1000.0;
        let initial = d.anchor;
        let mut tokens = vec![initial];
        let mut stop = clip_publication(
            &mut tokens,
            Some(PublicationBudget {
                remaining: admission.max_new_tokens,
                eos: &eos,
            }),
        )?;
        d.check_session_cancellation("input-prefix first publication")?;
        dispatch_verified_commit(
            sink,
            Qwen35VerifiedCommit {
                initial_anchor: true,
                block_index: 0,
                generation_before: d.generation as usize,
                generation_after: d.generation as usize,
                cursor_before: d.cursor,
                cursor_after: d.cursor,
                token_ids: vec![initial],
                stop_reason: stop.map(str::to_owned),
            },
        )?;
        let first_token_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut transactions = Vec::new();
        for round in 0..admission.max_new_tokens {
            if stop.is_some() {
                break;
            }
            d.check_session_cancellation("input-prefix draft round")?;
            draft.prepare_context().map_err(invalid)?;
            draft.submit().map_err(invalid)?;
            let previous = d.anchor;
            let mut tx = d.verify_product(
                &draft,
                Some(PublicationBudget {
                    remaining: admission.max_new_tokens - tokens.len(),
                    eos: &eos,
                }),
            )?;
            check_committed_path(&d, &mut tx, previous, policy.fast_session_path)?;
            let emitted: Vec<u32> = serde_json::from_value(tx["emitted_token_ids"].clone())
                .map_err(|e| invalid(e.to_string()))?;
            stop = match tx["publication_stop_reason"].as_str() {
                Some("eos") => Some("eos"),
                Some("max_new_tokens") => Some("max_new_tokens"),
                None => None,
                _ => return Err(invalid("input-prefix generation stop reason")),
            };
            d.check_session_cancellation("input-prefix committed publication")?;
            dispatch_verified_commit(
                sink,
                Qwen35VerifiedCommit {
                    initial_anchor: false,
                    block_index: round,
                    generation_before: tx["generation_before"]
                        .as_u64()
                        .ok_or_else(|| invalid("input-prefix transaction generation"))?
                        as usize,
                    generation_after: d.generation as usize,
                    cursor_before: tx["cursor_before"]
                        .as_u64()
                        .ok_or_else(|| invalid("input-prefix transaction cursor"))?
                        as usize,
                    cursor_after: d.cursor,
                    token_ids: emitted.clone(),
                    stop_reason: stop.map(str::to_owned),
                },
            )?;
            tokens.extend_from_slice(&emitted);
            transactions.push(tx);
            if stop.is_none() {
                rebind_driver_draft(&mut d, &mut draft, Some(emitted.len()), None)?;
            }
        }
        let last_token_ms = started.elapsed().as_secs_f64() * 1000.0;
        let stop = stop.ok_or_else(|| invalid("input-prefix incomplete generation"))?;
        d.stream.synchronize()?;
        draft.synchronize().map_err(invalid)?;
        let visible_end = tokens
            .iter()
            .position(|t| eos.contains(t))
            .unwrap_or(tokens.len());
        let output = json!({"prompt_token_ids":descriptor.consumed_input,"initial_anchor":initial,
            "token_ids":tokens,"visible_token_ids":tokens[..visible_end],
            "decoded_text":tokenizer.decode(&tokens[..visible_end],false).map_err(|e| invalid(e.to_string()))?,
            "eos_token_ids":eos,"stop_reason":stop,"max_new_tokens":admission.max_new_tokens});
        let session = CommittedSession::seal_initial(
            d,
            draft,
            admission.session_id,
            &output,
            &transactions,
            true,
            policy.fast_session_path,
        )?;
        let report = json!({"schema":"qwen35-native-input-prefix-first-generation-v1",
            "dflash_history_prepare":session.draft.history_prepare_report(),
            "output_head_batching":session.driver.output_head_batching_report(),
            "request_output":output,"transactions":transactions,"session_checkpoint":session.checkpoint,
            "draft_build_wall_ms":draft_build_ms,"first_token_wall_ms":first_token_ms,"last_token_wall_ms":last_token_ms,
            "generation_and_seal_wall_ms":started.elapsed().as_secs_f64()*1000.0,
            "real_generated_session_retained":true,"old_generated_ledger_restored":false,
            "product_ready":false,"qualified":false,"semantic_qualification":false,"visible_tps":false});
        self.session = Some(session);
        self.poisoned = false;
        Ok(report)
    }
}
