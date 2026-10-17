//! A real consumed-input owner, never a generated CommittedSession. The target
//! driver is the sole mutable KV/GDN/features authority. No old logits, pending
//! output or draft execution is serialized into an input checkpoint.
use super::*;
use crate::session_checkpoint::input_prefix::{InputPrefixDescriptor, INPUT_PREFIX_SCHEMA};
use crate::session_checkpoint::metadata::{
    DraftRestoreRecipe, FeatureExtent, HistoryMode, ModelBindings, NativeMathPolicy,
    SamplingPolicy, CAPACITY, HISTORY, KV_ROW_BYTES,
};
#[path = "input_prefix_restore.rs"]
mod restore;
#[path = "input_prefix_generate.rs"]
mod generate;

pub(crate) struct InputPrefixOwner {
    pub(super) driver: Driver,
    descriptor: InputPrefixDescriptor,
    // An input checkpoint never stores a prediction. Restored owners have no
    // valid anchor until at least one real new input token is consumed.
    anchor_ready: bool,
    pub(in super::super) boundary: Value,
}

impl InputPrefixOwner {
    pub(in super::super) fn seal(
        driver: Driver,
        tokens: &[u32],
        bindings: &ModelBindings,
        request: &Qwen35DraftTargetRequestOptions,
    ) -> Result<Self> {
        let history = driver
            .growing_history
            .ok_or_else(|| invalid("input-prefix history missing"))?;
        let descriptor = InputPrefixDescriptor {
            schema: INPUT_PREFIX_SCHEMA.into(),
            bindings: bindings.clone(),
            consumed_input: tokens.to_vec(),
            features: FeatureExtent {
                start: history.start_position(),
                rows: history.active_rows(),
                physical_rows: HISTORY,
                mode: HistoryMode::Growing2048 {
                    generation: history.generation(),
                },
            },
            math: NativeMathPolicy {
                full_attention_kv_basis: crate::session_checkpoint::metadata::FullAttentionKvBasis::from_layout(driver.llama_rotated_kv, driver.planar_rotated_kv),
                exact_projection_tiles: driver.exact_projection_tiles,
                exact_output_head_batch: driver.exact_output_head_batch,
                draft_top16_batch: driver.draft_top16_batch,
                parallel_receipt_digest: driver.parallel_receipt_digest,
                batched_target_verify: driver.batch_verify_execution.is_some(),
                reuse_batched_plan: driver.checkpoint_reuse_batched_plan,
                batched_suffix: driver.batched_suffix,
                retain_full_accept_state: driver.verified_features.is_some(),
                share_target_rope: driver.checkpoint_share_target_rope,
                batched_draft_context: request.batched_draft_context,
                batched_draft_block: request.batched_draft_block,
                draft_q8_tile4: request.draft_q8_tile4,
            },
            draft_restore: DraftRestoreRecipe::FreshCanonicalQ8FromTargetFeaturesV1,
            sampling: SamplingPolicy::GreedyArgmaxV1,
            origin_request_sha256: driver.identity.iter().map(|b| format!("{b:02x}")).collect(),
        };
        Self::from_driver(driver, descriptor, true)
    }

    fn from_driver(
        driver: Driver,
        descriptor: InputPrefixDescriptor,
        anchor_ready: bool,
    ) -> Result<Self> {
        let mut owner = Self {
            driver,
            descriptor,
            anchor_ready,
            boundary: Value::Null,
        };
        owner
            .driver
            .check_session_cancellation("input-prefix seal")?;
        owner.driver.stream.synchronize()?;
        let descriptor = owner.checkpoint_descriptor(&owner.descriptor.bindings)?;
        let submission = owner.driver.stream.submission_receipt()?;
        owner.boundary = json!({
            "schema":"qwen35-native-input-prefix-boundary-v1",
            "input_tokens":descriptor.consumed_input.len(),"consumed_input_digest":token_hash(&descriptor.consumed_input),
            "descriptor_sha256":crate::session_checkpoint::sha256(&descriptor.encode().map_err(|e| invalid(e.to_string()))?),
            "live_owner_submission":format!("{submission:?}"),
            "target_stream_drained":true,"generated_tokens":0,"published_tokens":0,
            "draft_executed":false,"session_retained":false,
            "product_ready":false,"qualified":false,"visible_tps":false
        });
        Ok(owner)
    }

    pub(super) fn checkpoint_descriptor(
        &self,
        bindings: &ModelBindings,
    ) -> Result<InputPrefixDescriptor> {
        let d = &self.driver;
        let descriptor = &self.descriptor;
        descriptor.validate().map_err(|e| invalid(e.to_string()))?;
        let history = d
            .growing_history
            .ok_or_else(|| invalid("input-prefix history absent"))?;
        if d.poisoned
            || d.llama_rotated_kv != descriptor.math.full_attention_kv_basis.is_rotated()
            || d.planar_rotated_kv != descriptor.math.full_attention_kv_basis.is_planar()
            || d.generation != 1
            || d.capacity != CAPACITY
            || d.plans.len() != 64
            || d.cursor != descriptor.consumed_input.len()
            || &descriptor.bindings != bindings
            || d.checkpoint_model_config_sha256 != bindings.model_config_sha256
            || d.checkpoint_tokenizer_sha256 != bindings.tokenizer_sha256
            || d.checkpoint_generation_config_sha256.as_ref()
                != Some(&bindings.generation_config_sha256)
            || history.cursor() != d.cursor
            || history.capacity_rows() != HISTORY
            || history.start_position() != descriptor.features.start
            || history.active_rows() != descriptor.features.rows
            || descriptor.features.mode
                != (HistoryMode::Growing2048 {
                    generation: history.generation(),
                })
        {
            return Err(invalid("input-prefix live owner/input identity mismatch"));
        }
        let math = &descriptor.math;
        if math.exact_projection_tiles != d.exact_projection_tiles
            || math.exact_output_head_batch != d.exact_output_head_batch
            || math.draft_top16_batch != d.draft_top16_batch
            || math.parallel_receipt_digest != d.parallel_receipt_digest
            || math.batched_target_verify != d.batch_verify_execution.is_some()
            || math.reuse_batched_plan != d.checkpoint_reuse_batched_plan
            || math.batched_suffix != d.batched_suffix
            || math.retain_full_accept_state != d.verified_features.is_some()
            || math.share_target_rope != d.checkpoint_share_target_rope
        {
            return Err(invalid("input-prefix live arithmetic policy drift"));
        }
        let plan = descriptor
            .transfer_plan()
            .map_err(|e| invalid(e.to_string()))?;
        for (layer, device_plan) in d.plans.iter().enumerate() {
            let full = matches!(device_plan, DevicePlan::Full(_));
            let buffers = device_plan.terminal_state_buffers();
            if full != (layer % 4 == 3) || buffers.len() != 2 {
                return Err(invalid("input-prefix target topology drift"));
            }
            for (plane, buffer) in buffers.iter().enumerate() {
                let declared = plan.components[1 + layer * 2 + plane].bytes as usize;
                let physical = if full {
                    CAPACITY * KV_ROW_BYTES as usize
                } else {
                    declared
                };
                if buffer.len_bytes() != physical
                    || declared > physical
                    || buffer.device_ordinal() != d.stream.device_ordinal()
                {
                    return Err(invalid("input-prefix physical plane extent drift"));
                }
            }
        }
        if d.features.len_bytes() != HISTORY * 5 * H * 4
            || d.active_feature_view()?.len_bytes() as u64 != plan.components[129].bytes
        {
            return Err(invalid("input-prefix feature ring extent drift"));
        }
        Ok(descriptor.clone())
    }
}

impl Qwen35ResidentDraftCuda {
    /// Capture exactly these consumed IDs, without generating/publishing a
    /// token. This native experiment is not an HTTP cache-hit qualification.
    pub fn prepare_native_input_prefix(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        tokens: &[u32],
        bindings: &ModelBindings,
        options: Qwen35DraftTargetRollingOptions,
    ) -> Result<Value> {
        self.validate_checkpoint_bindings(bindings)?;
        if self.session.is_some()
            || self.input_prefix.is_some()
            || self.target.is_none()
            || self.vision.is_none()
            || config.capacity != CAPACITY
            || config.device_ordinal != self.weights.device_ordinal()
            || !options.product_mode
            || !options.incremental_draft_context
            || options.validate_draft_context_reuse
            || options
                .request
                .as_ref()
                .is_none_or(|r| r.target_only_reference || r.audit_committed_state)
            || !crate::resident_session_profile::ResidentSessionProfile::Resident
                .admits_initial(tokens.len())
            || tokens.iter().any(|&token| token as usize >= V)
            || std::env::var("AUGUR_QWEN35_TOP16_VALIDATION_MODE").as_deref() != Ok("resident")
        {
            return Err(invalid("input-prefix native admission before submit"));
        }
        validate_resident_vision_selection(
            self.vision_path.as_deref(),
            config.vision_safetensors_path.as_deref(),
            config.vision_safetensors_path.as_deref(),
        )?;
        let request = options
            .request
            .as_ref()
            .ok_or_else(|| invalid("input-prefix request policy"))?;
        for (path, expected) in [
            (&config.config_path, &bindings.model_config_sha256),
            (&config.tokenizer_path, &bindings.tokenizer_sha256),
            (
                &request.generation_config_path,
                &bindings.generation_config_sha256,
            ),
        ] {
            if sha256_hex(&read_stable_regular_file(
                path,
                "input-prefix live input identity",
            )?) != *expected
            {
                return Err(invalid("input-prefix input file identity mismatch"));
            }
        }
        self.poisoned = true;
        let result = run_rolling_with_resident_draft(
            config,
            &self.q8_path,
            &self.draft_config_path,
            HISTORY,
            1024,
            self.vision_path.as_deref(),
            true,
            true,
            false,
            options.request.as_ref(),
            None,
            Some(&self.weights),
            self.target.as_ref(),
            self.vision.as_ref(),
            Some(RetainedOutput::InputPrefix {
                tokens,
                bindings,
                destination: &mut self.input_prefix,
            }),
            None,
            None,
            false,
            true,
            true,
            self.session_cancellation.as_ref(),
            None,
        );
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    pub fn describe_native_input_prefix(
        &self,
        expected: &Value,
        bindings: &ModelBindings,
    ) -> Result<InputPrefixDescriptor> {
        self.validate_checkpoint_bindings(bindings)?;
        let owner = self
            .input_prefix
            .as_ref()
            .ok_or_else(|| invalid("no retained input-prefix"))?;
        if expected != &owner.boundary {
            return Err(invalid("stale input-prefix handle"));
        }
        owner.checkpoint_descriptor(bindings)
    }

    pub fn close_native_input_prefix(&mut self, expected: &Value) -> Result<Value> {
        if self.poisoned {
            return Err(invalid("poisoned prefix owner requires restart"));
        }
        let owner = self
            .input_prefix
            .as_ref()
            .ok_or_else(|| invalid("no retained input-prefix"))?;
        if expected != &owner.boundary {
            return Err(invalid("stale input-prefix close"));
        }
        self.poisoned = true;
        owner.driver.stream.synchronize()?;
        drop(self.input_prefix.take());
        self.poisoned = false;
        Ok(json!({"input_prefix_closed":true,"generated_tokens":0,"product_ready":false}))
    }

    /// Explicit complete initialized-plane audit, not the live handle's hash.
    pub fn audit_native_input_prefix_state(
        &mut self,
        expected: &Value,
        bindings: &ModelBindings,
    ) -> Result<Value> {
        self.describe_native_input_prefix(expected, bindings)?;
        let owner = self
            .input_prefix
            .as_mut()
            .ok_or_else(|| invalid("input-prefix absent"))?;
        self.poisoned = true;
        let mut report = audit_target_state(&mut owner.driver)?;
        report
            .as_object_mut()
            .ok_or_else(|| invalid("input-prefix state audit object"))?
            .remove("pending_token");
        report["input_tokens"] = json!(owner.descriptor.consumed_input.len());
        report["consumed_input_digest"] = json!(token_hash(&owner.descriptor.consumed_input));
        report["next_token_computed_for_this_input"] = json!(owner.anchor_ready);
        report["next_token_diagnostic"] = if owner.anchor_ready {
            json!(owner.driver.anchor)
        } else {
            Value::Null
        };
        report["generated_tokens"] = json!(0);
        report["published_tokens"] = json!(0);
        report["cache_hit_qualified"] = json!(false);
        self.poisoned = false;
        Ok(report)
    }
}
