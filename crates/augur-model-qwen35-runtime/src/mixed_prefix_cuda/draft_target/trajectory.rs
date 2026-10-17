//! Default-off host-serialized observations of the ACTUAL product path.
use super::*;
use crate::dflash_trajectory::{classify, safe_existing_directory, Journal};
use std::sync::atomic::{AtomicU64, Ordering};
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(super) struct Capture {
    journal: Journal,
    consumed_tokens: Vec<u32>,
    capture_wall_ms: f64,
    feature_d2h_bytes: usize,
}

impl Capture {
    pub(super) fn from_env(
        d: &Driver,
        draft: &DFlashQ8CudaBlock,
        prompt: &[u32],
        product: bool,
    ) -> Result<Option<Self>> {
        let Some(root) = std::env::var_os("AUGUR_QWEN35_DFLASH_TRAJECTORY_DIR") else {
            return Ok(None);
        };
        let root = PathBuf::from(root);
        safe_existing_directory(&root).map_err(invalid)?;
        if !product
            || prompt.len() != d.cursor
            || d.batch_verify_execution
                .as_ref()
                .is_none_or(|x| x.output_head_workspace_bytes() == 0)
        {
            return Err(invalid(
                "live trajectory requires packed-head product path and exact consumed prompt",
            ));
        }
        let directory = root.join(format!(
            "request-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let metadata = json!({"build_identity":crate::build_identity::identity(),
            "target_execution_identity":sha256_hex(&d.identity),"draft_artifact_sha256":draft.trajectory_artifact_sha256(),
            "target_layer_ids":DFLASH_TARGET_CAPTURE_LAYER_IDS,"target_hidden_size":H,
            "initial_prompt_token_ids":prompt,"history_capacity":2048,
            "input_contract":"pending_anchor_at_cursor; committed_target_inputs_only; absolute_positions",
            "binary_binding":"external frozen build receipt required", "training_is_offline":true});
        Ok(Some(Self {
            journal: Journal::create(&directory, metadata, 5 * H, 2048, V).map_err(invalid)?,
            consumed_tokens: prompt.into(),
            capture_wall_ms: 0.0,
            feature_d2h_bytes: 0,
        }))
    }

    pub(super) fn begin(&mut self, d: &Driver, draft: &DFlashQ8CudaBlock) -> Result<()> {
        let start = Instant::now();
        if draft.submitted_identity().map_err(invalid)?
            != (d.generation, d.cursor, d.anchor, d.feature_sha)
            || self.consumed_tokens.len() != d.cursor
        {
            return Err(invalid("trajectory live draft/target identity drift"));
        }
        let (context_start, rows, bytes) = draft.trajectory_input_features().map_err(invalid)?;
        if context_start != d.active_context_start() || rows != d.active_context_rows() {
            return Err(invalid(
                "trajectory actual draft context differs from target history",
            ));
        }
        self.feature_d2h_bytes += bytes.len();
        self.journal
            .begin(
                context_start,
                d.generation,
                d.anchor,
                self.consumed_tokens
                    .get(context_start..d.cursor)
                    .ok_or_else(|| invalid("trajectory token history bounds"))?,
                bytes,
            )
            .map_err(invalid)?;
        let (gates, _, _) = draft.selector_inputs_device().map_err(invalid)?;
        for (role, source) in [
            (
                "draft_final_hidden_f32le",
                draft
                    .final_hidden_rows_device()
                    .map_err(invalid)?
                    .buffer_view(),
            ),
            ("selector_gate_f32le", gates),
        ] {
            let mut bytes = vec![0; source.len_bytes()];
            source.copy_to_host_sync_with_submission_proof(
                draft.stream(),
                &mut bytes,
                draft.stream().submission_receipt()?,
            )?;
            self.journal.attach_f32(role, &bytes).map_err(invalid)?;
        }
        self.capture_wall_ms += start.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    pub(super) fn commit(&mut self, d: &Driver, report: &Value) -> Result<()> {
        let start = Instant::now();
        let emitted = report["emitted_token_ids"]
            .as_array()
            .ok_or_else(|| invalid("trajectory emitted tokens"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| invalid("trajectory token range"))
            })
            .collect::<Result<Vec<_>>>()?;
        let consumed = words(&d.stream, &d.path, 0, emitted.len())?;
        let accepted = report["accepted_drafts"]
            .as_u64()
            .ok_or_else(|| invalid("trajectory acceptance absent"))?
            as usize;
        let mut observation = report["trajectory_verification"].clone();
        if !observation.is_object() {
            return Err(invalid("trajectory verification snapshot absent"));
        }
        let ids = |key: &str| -> Result<Vec<u32>> {
            observation[key]
                .as_array()
                .ok_or_else(|| invalid("trajectory observation IDs absent"))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| u32::try_from(n).ok())
                        .ok_or_else(|| invalid("trajectory observation IDs"))
                })
                .collect()
        };
        observation["classification"] = classify(
            accepted,
            &ids("candidate_ids")?,
            &ids("draft_path_ids")?,
            &ids("draft_conditioned_target_ids")?,
        )
        .map_err(invalid)?;
        observation["commit_strategy"] = report["commit_strategy"].clone();
        self.journal
            .commit(
                &emitted,
                &consumed,
                d.cursor,
                d.generation,
                d.anchor,
                observation,
            )
            .map_err(invalid)?;
        self.consumed_tokens.extend_from_slice(&consumed);
        self.capture_wall_ms += start.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    pub(super) fn finish(self) -> Result<Value> {
        let mut receipt = self.journal.finish().map_err(invalid)?;
        receipt["capture_host_wall_ms_excludes_verification_observation"] =
            json!(self.capture_wall_ms);
        receipt["actual_feature_d2h_bytes"] = json!(self.feature_d2h_bytes);
        receipt["performance_evidence"] = json!(false);
        Ok(receipt)
    }
}

/// Called only AFTER the normal acceptance receipt and submit-boundary check,
/// BEFORE rollback/replay can overwrite the verified logits. No audit is ever
/// permitted to weaken the enqueue-only product submission contract.
pub(super) fn observe_verification(d: &Driver) -> Result<Value> {
    let start = Instant::now();
    let target = words(&d.stream, &d.verify_ids, 0, B)?;
    let path = words(&d.stream, &d.path, 0, B)?;
    let candidates = words(&d.stream, &d.candidate_ids, 0, B * 16)?;
    let receipt = d.stream.submission_receipt()?;
    let unary = read_f32_view(&d.stream, view(&d.unary)?, receipt)?;
    let transitions = read_f32_view(&d.stream, view(&d.transitions)?, receipt)?;
    if unary.len() != B * 16
        || transitions.len() != B * 16
        || unary.iter().chain(&transitions).any(|x| !x.is_finite())
    {
        return Err(invalid("trajectory candidate scores invalid"));
    }
    let executor = d
        .batch_verify_execution
        .as_ref()
        .ok_or_else(|| invalid("trajectory packed head absent"))?;
    let summaries = (0..B)
        .map(|row| {
            let logits = executor.trajectory_target_row(d, row)?;
            target_only_reference::summarize_logits(&logits, target[row], row)
        })
        .collect::<Result<Vec<_>>>()?;
    let unary_margins = (0..B)
        .map(|r| unary[r * 16] as f64 - unary[r * 16 + 1] as f64)
        .collect::<Vec<_>>();
    Ok(
        json!({"draft_path_ids":path,"candidate_ids":candidates,"unary_scores":unary,
        "transition_scores":transitions,"unary_top1_top2_margins":unary_margins,
        "draft_conditioned_target_ids":target,"target_logit_summaries":summaries,
        "observation_wall_ms":start.elapsed().as_secs_f64()*1000.0,
        "d2h_bytes":B*V*4+3*B*16*4+2*B*4,"performance_evidence":false,
        "target_logit_summaries_after_first_rejection_are_not_canonical_labels":true}),
    )
}
