//! Fresh input-only restore and true teacher-forced suffix. No old CUDA owner,
//! hidden pending-token consumption, generated ledger or publication callback.
use super::super::checkpoint_restore::{fresh_driver, restore_target_payload, FreshTarget};
use super::*;
use crate::session_checkpoint::filesystem::LocalCheckpointReader;
use crate::session_checkpoint::input_prefix::InputPrefixRestoreAdmission;

impl Qwen35ResidentDraftCuda {
    /// None is an ordinary exact-prefix miss before mutable CUDA allocation.
    /// Corrupt/foreign snapshots are errors, never silently promoted to hits.
    pub fn restore_native_input_prefix(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        generation_config: &Path,
        directory: &Path,
        manifest_sha256: &str,
        admission: &InputPrefixRestoreAdmission<'_>,
    ) -> Result<Option<Value>> {
        let InputPrefixRestoreAdmission {
            bindings,
            math,
            new_request_id,
            request_input,
            output_reserve,
        } = *admission;
        if self.poisoned
            || self.session.is_some()
            || self.input_prefix.is_some()
            || !valid_id(new_request_id)
        {
            return Err(invalid(
                "input-prefix restore requires healthy empty owner and fresh request label",
            ));
        }
        self.validate_checkpoint_bindings(bindings)?;
        let target = self
            .target
            .as_ref()
            .ok_or_else(|| invalid("input-prefix target weights missing"))?;
        if config.device_ordinal != 0
            || target.device != 0
            || self.weights.device_ordinal() != 0
            || config.capacity != CAPACITY
            || target.gguf_path != config.gguf_path
            || self.vision_path.as_ref() != config.vision_safetensors_path.as_ref()
            || target.layers.len() != 64
            || output_reserve < B
        {
            return Err(invalid(
                "input-prefix restore residency or verification reserve",
            ));
        }
        let config_bytes =
            read_stable_regular_file(&config.config_path, "input-prefix model config")?;
        if sha256_hex(&config_bytes) != bindings.model_config_sha256
            || sha256_hex(&read_stable_regular_file(
                &config.tokenizer_path,
                "input-prefix tokenizer",
            )?) != bindings.tokenizer_sha256
            || sha256_hex(&read_stable_regular_file(
                generation_config,
                "input-prefix generation config",
            )?) != bindings.generation_config_sha256
        {
            return Err(invalid("input-prefix restore live input identity"));
        }
        let model = Qwen35Config::parse(&config_bytes)?;
        model.validate_exact_profile()?;
        let metadata =
            LocalCheckpointReader::read_input_prefix_descriptor(directory, manifest_sha256)
                .map_err(|e| invalid(e.to_string()))?;
        let descriptor = InputPrefixDescriptor::decode(&metadata, bindings, math)
            .map_err(|e| invalid(e.to_string()))?;
        if descriptor
            .lookup_suffix(bindings, math, request_input, output_reserve)
            .map_err(|e| invalid(e.to_string()))?
            .is_none()
        {
            return Ok(None);
        }
        if math.draft_top16_batch && !cfg!(feature = "qwen35-draft-top16-batch") {
            return Err(invalid("input-prefix restore compiled top16 policy"));
        }
        let plan = descriptor
            .transfer_plan()
            .map_err(|e| invalid(e.to_string()))?;
        let reader = LocalCheckpointReader::open(directory, manifest_sha256, &plan)
            .map_err(|e| invalid(e.to_string()))?;
        self.poisoned = true;
        let started = Instant::now();
        let mut driver = fresh_driver(
            target,
            &model,
            FreshTarget {
                consumed: &descriptor.consumed_input,
                features: &descriptor.features,
                math,
                bindings,
                pending: None,
                generation: 1,
                metadata: &metadata,
            },
            new_request_id,
        )?;
        driver.session_cancellation = self.session_cancellation.clone();
        driver.check_session_cancellation("input-prefix restore before transfer")?;
        let restored =
            restore_target_payload(driver, metadata, plan, reader, manifest_sha256, None)?;
        let mut driver = restored.driver;
        if math.batched_target_verify {
            driver.batch_verify_execution = Some(BatchedVerifyExecution::new(
                &driver,
                CAPACITY,
                model.rms_norm_eps as f32,
                model.rope_theta as f32,
                math.reuse_batched_plan,
                false,
                false,
            )?);
        }
        let owner = InputPrefixOwner::from_driver(driver, descriptor, false)?;
        let report = json!({
            "schema":"qwen35-native-input-prefix-restore-v1",
            "restored_prefix_tokens":owner.descriptor.consumed_input.len(),"prefix_recompute_tokens":0,
            "transfer_wall_ms":restored.wall_ms,"restore_wall_ms":started.elapsed().as_secs_f64()*1000.0,
            "manifest_sha256":restored.receipt.manifest_sha256,"payload_bytes":restored.receipt.payload_bytes,
            "h2d_bytes":restored.h2d_bytes,"staging_bytes":restored.receipt.staging_bytes,
            "fresh_mutable_owner":true,"old_cuda_authority_restored":false,"old_logits_restored":false,
            "draft_executed":false,"generated_tokens":0,"published_tokens":0,"boundary":owner.boundary,
            "suffix_executed":false,"cache_hit_qualified":false,
            "product_ready":false,"qualified":false,"semantic_qualification":false,"visible_tps":false
        });
        self.input_prefix = Some(owner);
        self.poisoned = false;
        Ok(Some(report))
    }

    /// Consume a new exact canonical input's suffix; unlike chat append this
    /// has no 2048-token user-turn limit. Only physical capacity/reserve bound
    /// the total; submissions remain canonical B8 plus a genuine scalar tail.
    pub fn append_native_input_prefix(
        &mut self,
        expected: &Value,
        bindings: &ModelBindings,
        request_input: &[u32],
        output_reserve: usize,
    ) -> Result<Option<Value>> {
        let descriptor = self.describe_native_input_prefix(expected, bindings)?;
        if output_reserve < B {
            return Err(invalid("input-prefix suffix verification reserve"));
        }
        let Some(suffix_plan) = descriptor
            .lookup_suffix(bindings, &descriptor.math, request_input, output_reserve)
            .map_err(|e| invalid(e.to_string()))?
        else {
            return Ok(None);
        };
        let suffix = suffix_plan.suffix();
        let batch_rows = if descriptor.math.batched_suffix {
            suffix.len() / B * B
        } else {
            0
        };
        let current = self
            .input_prefix
            .as_ref()
            .ok_or_else(|| invalid("input-prefix absent"))?;
        if batch_rows > 0 {
            current
                .driver
                .batch_verify_execution
                .as_ref()
                .ok_or_else(|| invalid("input-prefix suffix executor missing"))?
                .validate_suffix(&current.driver, batch_rows)?;
        }
        let mut consumed = Vec::new();
        consumed
            .try_reserve_exact(request_input.len())
            .map_err(|e| invalid(e.to_string()))?;
        consumed.extend_from_slice(request_input);
        self.poisoned = true;
        let mut owner = self
            .input_prefix
            .take()
            .ok_or_else(|| invalid("input-prefix absent"))?;
        let d = &mut owner.driver;
        d.poisoned = true;
        let started = Instant::now();
        for block in suffix[..batch_rows].chunks_exact(B) {
            d.check_session_cancellation("input-prefix suffix B8")?;
            let block: &[u32; B] = block
                .try_into()
                .map_err(|_| invalid("input-prefix suffix B8 extent"))?;
            d.batch_verify_execution
                .as_ref()
                .ok_or_else(|| invalid("input-prefix executor ownership"))?
                .submit_suffix(d, block)?;
            d.promote_verified_features(B)?;
            d.cursor += B;
        }
        for &token in &suffix[batch_rows..] {
            d.check_session_cancellation("input-prefix suffix scalar")?;
            let advance = d.advance_feature_window(1)?;
            d.token
                .copy_from_host_sync(&d.stream, &token.to_le_bytes())?;
            d.bind_gather(view(&d.token)?)?.submit_for_plan(&d.stream)?;
            d.forward(d.cursor, advance.capture_row(0))?;
            d.cursor += 1;
        }
        if d.cursor != request_input.len() {
            return Err(invalid("input-prefix suffix final cursor"));
        }
        if batch_rows == suffix.len() {
            d.head.submit(&d.stream)?;
        }
        d.top1(view(&d.token)?)?;
        d.anchor = words(&d.stream, &d.token, 0, 1)?[0];
        if d.anchor as usize >= V {
            return Err(invalid("input-prefix suffix fresh prediction range"));
        }
        d.next_context_feature_digest()?;
        d.stream.synchronize()?;
        d.poisoned = false;
        let suffix_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        let history = d
            .growing_history
            .ok_or_else(|| invalid("input-prefix suffix initialized history"))?;
        owner.descriptor.consumed_input = consumed;
        owner.descriptor.features = FeatureExtent {
            start: history.start_position(),
            rows: history.active_rows(),
            physical_rows: HISTORY,
            mode: HistoryMode::Growing2048 {
                generation: history.generation(),
            },
        };
        let owner = InputPrefixOwner::from_driver(owner.driver, owner.descriptor, true)?;
        let report = json!({
            "schema":"qwen35-native-input-prefix-suffix-v1",
            "restored_prefix_tokens":descriptor.consumed_input.len(),"input_tokens_executed":suffix.len(),
            "final_input_tokens":request_input.len(),"prefix_recompute_tokens":0,"suffix_wall_ms":suffix_wall_ms,
            "batch_rows":batch_rows,"scalar_tail_rows":suffix.len()-batch_rows,"padding_tokens":0,
            "pending_generated_token_consumed":false,"draft_executed":false,
            "generated_tokens":0,"published_tokens":0,"boundary":owner.boundary,
            "cache_hit_qualified":false,"product_ready":false,"qualified":false,
            "semantic_qualification":false,"visible_tps":false
        });
        self.input_prefix = Some(owner);
        self.poisoned = false;
        Ok(Some(report))
    }
}
