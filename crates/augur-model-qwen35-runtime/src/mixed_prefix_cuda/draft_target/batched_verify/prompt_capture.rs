//! Default-off teacher-forced prompt tail, using the canonical batch core.
//! The temporary owner dies before draft/Vision admission; no token publication.
use super::*;
#[cfg(feature = "qwen35-session-continuation")]
mod history_wide;
#[cfg(feature = "qwen35-session-continuation")]
mod exact_input_rows;

#[cfg(test)]
pub(in super::super) fn validate_prompt_capture(
    prompt_tokens: usize,
    rows: usize,
    capacity: usize,
) -> Result<()> {
    validate_prompt_capture_extent(prompt_tokens, rows, capacity, false)
}

pub(in super::super) fn validate_prompt_capture_extent(
    prompt_tokens: usize,
    rows: usize,
    capacity: usize,
    growing: bool,
) -> Result<()> {
    let minimum = if growing { 1 } else { B };
    if !(minimum..=2048).contains(&rows)
        || rows > prompt_tokens
        || prompt_tokens > capacity
        || (growing && rows != prompt_tokens.min(2048))
    {
        return Err(invalid(
            "batch prompt capture requires a real eight-row tail within capacity",
        ));
    }
    Ok(())
}

// Drain on errors AND unwinding, before core fields can release device owners.
// Failed partial conditioning stays poisoned; there is no fallback/replay from
// partially advanced recurrent state. Normal completion also explicitly drains.
struct CaptureOwner<'a> {
    driver: &'a mut Driver,
    core: Option<BatchedVerifyAudit>,
    prefix_profile: Option<CanonicalProfile>,
    drained: bool,
}
impl Drop for CaptureOwner<'_> {
    fn drop(&mut self) {
        if !self.drained && self.driver.stream.synchronize().is_err() {
            self.driver.poisoned = true;
        }
    }
}

pub(in super::super) fn capture_prompt(
    d: &mut Driver,
    prompt: &[usize],
    epsilon: f32,
    theta: f32,
    capture_fc:Option<&mut crate::dflash_q8_backbone_cuda::capture_fc::CaptureFc>,
) -> Result<Value> {
    d.check_session_cancellation("prompt capture admission")?;
    validate_prompt_capture_extent(
        prompt.len(),
        d.context_rows,
        d.capacity,
        d.growing_history.is_some(),
    )?;
    if d.poisoned
        || d.cursor != prompt.len()
        || d.verified_features.is_none()
        || prompt.iter().any(|&id| id >= V)
    {
        return Err(invalid("batch prompt capture owner/cursor/token binding"));
    }
    let rows = d.context_rows;
    #[cfg(feature = "qwen35-session-continuation")]
    if d.llama_exact_input_rows {
        return exact_input_rows::capture(d, prompt, epsilon, theta, capture_fc);
    }
    if capture_fc.is_some() {return Err(invalid("capture FC requires exact global input rows"));}
    #[cfg(feature = "qwen35-session-continuation")]
    if d.wide_history_capture {
        return history_wide::capture(d, prompt, epsilon, theta);
    }
    let start = prompt.len() - rows;
    if rows < B {
        // Explicit real-token bootstrap. Never allocate/execute a padded B8
        // prompt, never read an unwritten feature row, and never generate warmup.
        let timer = Instant::now();
        let before = cuda_device_allocation_snapshot();
        d.poisoned = true;
        let result = (|| {
            for row in 0..rows {
                let position = start + row;
                d.cursor = position;
                d.token
                    .copy_from_host_sync(&d.stream, &(prompt[position] as u32).to_le_bytes())?;
                d.bind_gather(view(&d.token)?)?.submit_for_plan(&d.stream)?;
                d.forward(position, Some(row))?;
                if row + 1 == rows {
                    d.top1(view(&d.token)?)?;
                }
            }
            d.stream.synchronize()?;
            let after = cuda_device_allocation_snapshot();
            if before.live_bytes != after.live_bytes
                || before.live_allocations != after.live_allocations
            {
                return Err(invalid("scalar prompt bootstrap allocation closure"));
            }
            d.cursor = prompt.len();
            Ok(
                json!({"schema":"qwen35-canonical-scalar-prompt-bootstrap-v1",
                "feature_start":start,"feature_end":prompt.len(),"feature_rows":rows,
                "batch_count":0,"serial_tail_rows":rows,"padding_tokens":0,
                "generated_warmup_tokens":0,"publication_enabled":false,
                "workspace_bytes":0,"workspace_released_bytes":0,
                "terminal_proof":"stream_synchronize_returned_success",
                "conditioning_wall_ms":timer.elapsed().as_secs_f64()*1000.0,
                "product_qualification":false}),
            )
        })();
        if result.is_ok() {
            d.poisoned = false;
        } else {
            let _ = d.stream.synchronize();
        }
        return result;
    }
    let batch_rows = rows / B * B;
    let before = cuda_device_allocation_snapshot();
    let timer = Instant::now();
    d.poisoned = true;
    let mut owner = CaptureOwner {
        driver: d,
        core: None,
        prefix_profile: None,
        drained: false,
    };
    owner.driver.cursor = start;
    owner.core = Some(BatchedVerifyAudit::allocate(
        owner.driver,
        prompt.len(),
        epsilon,
        theta,
        false,
        true,
        false,
        true,
        true,
        true,
        true,
    )?);
    {
        let core = owner
            .core
            .as_ref()
            .ok_or_else(|| invalid("prompt capture core"))?;
        let d = &mut *owner.driver;
        // Position-independent bindings; the canonical full core consults the
        // current absolute d.cursor for every submitted row, including RoPE/KV.
        let operations = core.bind_operations(d)?;
        for row in (0..batch_rows).step_by(B) {
            d.cursor = start + row;
            let ids = prompt[d.cursor..d.cursor + B]
                .iter()
                .flat_map(|&id| (id as u32).to_le_bytes())
                .collect::<Vec<_>>();
            d.path.copy_from_host_sync(&d.stream, &ids)?;
            core.submit_backbone(d, &operations, true)?;
            copy(
                &d.stream,
                &d.features,
                row * 5 * H * 4,
                view(
                    d.verified_features
                        .as_ref()
                        .ok_or_else(|| invalid("capture staging"))?,
                )?,
            )?;
        }
        if batch_rows == rows {
            copy(
                &d.stream,
                d.plans[63].final_hidden_buffer(),
                0,
                core.workspace
                    .hidden_output(63, B)?
                    .buffer_view()
                    .subview((B - 1) * H * 4, H * 4)?,
            )?;
            d.head.submit(&d.stream)?;
            d.top1(view(&d.token)?)?;
        } else {
            // No padding, generated warmup or shortened feature window.
            for row in batch_rows..rows {
                let position = start + row;
                d.token
                    .copy_from_host_sync(&d.stream, &(prompt[position] as u32).to_le_bytes())?;
                d.bind_gather(view(&d.token)?)?.submit_for_plan(&d.stream)?;
                d.forward(position, Some(row))?;
                if row + 1 == rows {
                    d.top1(view(&d.token)?)?;
                }
            }
        }
    }
    owner.driver.stream.synchronize()?;
    owner.drained = true;
    let allocated = cuda_device_allocation_snapshot();
    drop(owner.core.take());
    let after = cuda_device_allocation_snapshot();
    if before.live_bytes != after.live_bytes || before.live_allocations != after.live_allocations {
        return Err(invalid(
            "prompt capture temporary workspace cleanup did not close",
        ));
    }
    let workspace_bytes = allocated
        .live_bytes
        .checked_sub(before.live_bytes)
        .ok_or_else(|| invalid("prompt capture allocation accounting"))?;
    let allocations = after
        .successful_allocations
        .checked_sub(before.successful_allocations)
        .ok_or_else(|| invalid("prompt capture allocation counter"))?;
    let frees = after
        .successful_frees
        .checked_sub(before.successful_frees)
        .ok_or_else(|| invalid("prompt capture free counter"))?;
    if allocations != frees || workspace_bytes == 0 {
        return Err(invalid("prompt capture allocation/free closure"));
    }
    owner.driver.cursor = prompt.len();
    owner.driver.poisoned = false;
    Ok(
        json!({"schema":"qwen35-canonical-batched-prompt-capture-v1",
        "arithmetic_policy":"native_quantized_canonical_norm_scalar_full_core_v1",
        "feature_start":start,"feature_end":prompt.len(),"feature_rows":rows,
        "batch_width":B,"batch_count":rows/B,"serial_tail_rows":rows%B,
        "output_head_calls":if rows%B==0 {1}else{rows%B},
        "feature_d2d_bytes":batch_rows*5*H*4,
        "generated_warmup_tokens":0,"padding_tokens":0,"publication_enabled":false,
        "operation_bind_calls":1,"conditioning_wall_ms":timer.elapsed().as_secs_f64()*1000.0,
        "workspace_bytes":workspace_bytes,"workspace_released_bytes":workspace_bytes,
        "live_bytes_before":before.live_bytes,"live_bytes_with_workspace":allocated.live_bytes,
        "live_bytes_after":after.live_bytes,"allocations":allocations,"frees":frees,
        "workspace_released_before_auxiliary_and_decode":true,"product_qualification":false}),
    )
}

/// Isolated arithmetic diagnostic. Consume the real prefix with the same
/// canonical B8 operators as prompt capture; do not publish or capture features.
#[cfg(feature = "qwen35-session-continuation")]
pub(in super::super) fn prefill_full_canonical_audit(
    d: &mut Driver,
    prompt: &[usize],
    epsilon: f32,
    theta: f32,
) -> Result<(AutoregressivePrefillRun, Value)> {
    let requested_rows = d.canonical_prefill_rows;
    #[cfg(feature = "qwen35-kv8-batched-attention")]
    let requested_score_mode = match full_attention::PrefillQueries::request_policy(
        std::env::var_os("AUGUR_QWEN35_CANONICAL_PREFILL_SCORE_MODE").as_deref(),
    )? {
        None => "shared-b8",
        Some(augur_backend::Qwen35CanonicalPrefillScorePolicy::CanonicalWarp) => "warp",
        Some(augur_backend::Qwen35CanonicalPrefillScorePolicy::Dgemm) => "dgemm",
    };
    #[cfg(not(feature = "qwen35-kv8-batched-attention"))]
    let requested_score_mode = {
        if std::env::var_os("AUGUR_QWEN35_CANONICAL_PREFILL_SCORE_MODE").is_some() {
            return Err(invalid("canonical staged-score attention not compiled"));
        }
        "shared-b8"
    };
    d.check_session_cancellation("canonical prefill admission")?;
    validate_prompt_capture_extent(
        prompt.len(),
        d.context_rows,
        d.capacity,
        d.growing_history.is_some(),
    )?;
    let capture_start = prompt.len() - d.context_rows;
    let global_boundary = d.llama_global_chunks && d.wide_history_capture;
    let prefix = if d.llama_exact_input_rows {
        history_capture::exact_input_schedule(prompt.len(), d.context_rows)?.0
    } else if global_boundary {
        history_capture::global_boundary_schedule(prompt.len())?.0
    } else { capture_start };
    let width = prefill_width::select(requested_rows, prefix)?;
    if d.integer_mma_ffn.is_some_and(|rows| rows != requested_rows) {
        return Err(invalid(
            "integer MMA prefix width changed after identity binding",
        ));
    }
    if d.tensor_ffn_down_rows
        .is_some_and(|rows| rows != requested_rows)
    {
        return Err(invalid(
            "FFN-down prefix width changed after request identity binding",
        ));
    }
    let sampled_indices = canonical_profile::prefix_sampling(
        std::env::var_os("AUGUR_QWEN35_CANONICAL_PREFIX_ATTRIBUTION").as_deref(),
        width,
        prefix,
    )?;
    if (d.tensor_ffn_down_rows.is_some() || d.integer_mma_ffn.is_some())
        && (sampled_indices.is_some() || requested_score_mode != "shared-b8")
    {
        return Err(invalid(
            "FFN-down experiment excludes prefix profiling and score ablations",
        ));
    }
    if sampled_indices.is_some() && requested_score_mode != "shared-b8" {
        return Err(invalid(
            "prefix attribution requires the reference shared-B8 score policy",
        ));
    }
    if requested_score_mode != "shared-b8"
        && width >= 128
        && std::env::var("AUGUR_QWEN35_SHARED_KV_QUERIES").as_deref() != Ok("1")
    {
        return Err(invalid(
            "canonical staged scores require explicit shared KV queries",
        ));
    }
    validate_canonical_prefix_with_history(prompt.len(), capture_start, d.growing_history.is_some())?;
    if d.poisoned || d.cursor != prompt.len() || prompt.iter().any(|&id| id >= V) {
        return Err(invalid("canonical prefix audit owner/cursor/tokens"));
    }
    let before = cuda_device_allocation_snapshot();
    let timer = Instant::now();
    if prefix == 0 {
        // No prefix workspace or model token is needed. Drain only the existing
        // plan reset before capture; the following capture owns all real tokens.
        d.stream.synchronize()?;
        let wall_ms = timer.elapsed().as_secs_f64() * 1000.0;
        return Ok((
            AutoregressivePrefillRun {
                host_wall_ms: wall_ms,
                batched_report: None,
                polls: 0,
                proof: true,
                inter_layer_d2d_copy_count: 0,
                inter_layer_d2d_bytes: 0,
            },
            json!({"schema":"qwen35-native-zero-prefix-v1","prefix_tokens":0,
                "batch_count":0,"serial_tail_rows":0,"padding_tokens":0,
                "generated_warmup_tokens":0,"publication_enabled":false,
                "workspace_bytes":0,"workspace_released_bytes":0,"allocations":0,"frees":0,
                "live_bytes_before":before.live_bytes,"live_bytes_after":before.live_bytes,
                "terminal_proof":"existing_plan_reset_stream_synchronized",
                "conditioning_wall_ms":wall_ms,"large_prefill_dgemm_executed":false,
                "requested_prefix_score_mode":requested_score_mode,"prefix_score_mode":"no-prefix",
                "workspace_released_before_auxiliary_and_decode":true,"product_qualification":false}),
        ));
    }
    d.poisoned = true;
    let mut owner = CaptureOwner {
        driver: d,
        core: None,
        prefix_profile: None,
        drained: false,
    };
    owner.driver.cursor = 0;
    owner.core = Some(BatchedVerifyAudit::allocate_prefill_rows(
        owner.driver,
        prompt.len(),
        epsilon,
        theta,
        width,
    )?);
    if sampled_indices.is_some() {
        owner.prefix_profile = Some(CanonicalProfile::new()?);
    }
    let mut prefix_samples = Vec::new();
    let schedule = prefill_width::schedule(width, prefix)?;
    let batch_rows = schedule.wide_rows;
    let tail_end = batch_rows + schedule.b8_tail_rows;
    let mut tail_device_allocations = 0;
    let mut row_reuse_selected_count = 0;
    let tensor_ffn_down_selected_count;
    let integer_mma_selected_count;
    let input_integer_selected_count;
    let llama_input_selected_count;
    let original_input_selected_count;
    let full_original_selected_count;
    let qpn_selected_count;
    let bf16_scalar_selected_count;
    let gdn_token_grid_selected_count;
    let llama_norm_selected_count;
    {
        let core = owner
            .core
            .as_ref()
            .ok_or_else(|| invalid("canonical prefix core"))?;
        let d = &mut *owner.driver;
        let mut operations = core.bind_operations(d)?;
        let directly_bound = prefill_input_integer::original_operand_count(&operations) == 400;
        if directly_bound {
            prefill_input_integer::validate_exact_input_bindings(&operations, width)?;
        }
        if !directly_bound && d.llama_f16_input && width == 512 {
            prefill_input_integer::install_llama(&mut operations, width)?;
        }
        if !directly_bound && d.llama_original_input && width == 512 {
            prefill_input_integer::install_original(&mut operations, width, core, d)?;
        }
        if !directly_bound && d.llama_full_original && width == 512 {
            prefill_input_integer::install_full_original(&mut operations, width, core, d)?;
        }
        if !directly_bound && d.llama_bf16_scalars && width == 512 {
            prefill_input_integer::install_bf16_scalars(&mut operations, width, core, d)?;
        }
        bf16_scalar_selected_count = prefill_input_integer::bf16_scalar_selected_count(&operations);
        llama_norm_selected_count = prefill_input_integer::llama_norm_selected_count(&operations);
        if llama_norm_selected_count != if d.llama_exact_input_rows {128} else {0} {
            return Err(invalid("prefix input norm identity"));
        }
        full_original_selected_count = prefill_input_integer::full_original_selected_count(&operations);
        qpn_selected_count = prefill_input_integer::qpn_projection_count(&operations);
        llama_input_selected_count = prefill_input_integer::llama_selected_count(&operations);
        original_input_selected_count = prefill_input_integer::original_selected_count(&operations);
        gdn_token_grid_selected_count = operations.iter().filter(|op|
            matches!(op, PrefillLayerOps::Linear(p) if p.canonical_gdn_prefill_batch.is_some()
                && !p.alpha.llama_full_original_enabled())).count();
        if gdn_token_grid_selected_count != if width > B && cfg!(feature="qwen35-bf16-warp-row")
            && bf16_scalar_selected_count == 0 {48} else {0} {
            return Err(invalid("canonical prefix GDN token grid layer coverage"));
        }
        tensor_ffn_down_selected_count = prefill_ffn_down::selected_count(&operations);
        integer_mma_selected_count = prefill_integer_mma::selected_count(&operations);
        input_integer_selected_count = prefill_input_integer::selected_count(&operations);
        if width == B {
            row_reuse_selected_count = super::projection_row_reuse_count(&operations);
        }
        for row in (0..batch_rows).step_by(width) {
            d.cursor = row;
            let ids = prompt[row..row + width]
                .iter()
                .flat_map(|&id| (id as u32).to_le_bytes())
                .collect::<Vec<_>>();
            let input = if width == B { &d.path } else { &core.batch_ids };
            input.copy_from_host_sync(&d.stream, &ids)?;
            if row == width {
                // The existing synchronous copy drains this same stream. The
                // first width rows have therefore completed all 64 backbone layers.
                // Observation only: no extra CUDA wait, capture, or publication.
                eprintln!(
                    "QWEN35_CANONICAL_PREFILL_PROGRESS={}",
                    json!({
                        "schema":"qwen35-canonical-prefill-progress-v1",
                        "pid":std::process::id(),"completed_rows":width,
                        "prefix_rows":prefix,"prompt_tokens":prompt.len(),
                        "proof":"same_stream_h2d_sync_after_backbone",
                        "visible_tokens":0
                    })
                );
            }
            if let Some(profile) = owner.prefix_profile.as_ref().filter(|_| {
                sampled_indices
                    .as_ref()
                    .is_some_and(|indices| indices.contains(&(row / width)))
            }) {
                profile.start(d)?;
                core.submit_profiled_prefix_b8(d, &operations, profile)?;
                // Adjacent terminal markers, not an output-head call. Drain
                // just these three enclosing batches; never wait mid-layer.
                profile.head(d)?;
                profile.stop(d)?;
                d.stream.synchronize()?;
                prefix_samples.push(profile.prefix_report(d, row / width)?);
            } else {
                core.submit_backbone(d, &operations, false)?;
            }
        }
        if schedule.b8_tail_rows != 0 {
            let tail_before = cuda_device_allocation_snapshot();
            d.cursor = batch_rows;
            let tail_operations = core.bind_operations_rows(d, B)?;
            row_reuse_selected_count = super::projection_row_reuse_count(&tail_operations);
            for row in (batch_rows..tail_end).step_by(B) {
                d.cursor = row;
                let ids = prompt[row..row + B]
                    .iter()
                    .flat_map(|&id| (id as u32).to_le_bytes())
                    .collect::<Vec<_>>();
                d.path.copy_from_host_sync(&d.stream, &ids)?;
                core.submit_prefix_b8_tail(d, &tail_operations)?;
            }
            let tail_after = cuda_device_allocation_snapshot();
            tail_device_allocations = tail_after
                .successful_allocations
                .checked_sub(tail_before.successful_allocations)
                .ok_or_else(|| invalid("prefix B8 tail allocation counter drift"))?;
            if tail_device_allocations != 0
                || tail_after.successful_frees != tail_before.successful_frees
                || tail_after.live_bytes != tail_before.live_bytes
                || tail_after.live_allocations != tail_before.live_allocations
            {
                return Err(invalid(
                    "prefix B8 tail must reuse the existing device arena",
                ));
            }
        }
        // Only fewer than eight real inputs remain scalar. No padding,
        // speculative feature capture, output head or token publication.
        for position in tail_end..prefix {
            d.cursor = position;
            d.token
                .copy_from_host_sync(&d.stream, &(prompt[position] as u32).to_le_bytes())?;
            d.bind_gather(view(&d.token)?)?.submit_for_plan(&d.stream)?;
            d.forward(position, None)?;
        }
    }
    owner.driver.stream.synchronize()?;
    owner.drained = true;
    let prefix_attribution = if sampled_indices.is_some() {
        Some(canonical_profile::prefix_summary(prefix, prefix_samples)?)
    } else {
        None
    };
    #[cfg(feature = "qwen35-kv8-batched-attention")]
    let score_mode = owner
        .core
        .as_ref()
        .map(|core| {
            core.prefill_queries
                .as_ref()
                .map(|queries| queries.score_mode_name())
                .unwrap_or(if core.shared_queries.is_some() {
                    "shared-b8"
                } else {
                    "serial-canonical"
                })
        })
        .ok_or_else(|| invalid("canonical prefix score-mode owner"))?;
    #[cfg(not(feature = "qwen35-kv8-batched-attention"))]
    let score_mode = "serial-canonical";
    let staged_scores = matches!(
        score_mode,
        "canonical-warp-staged-scores" | "fp64-dgemm-staged-scores"
    ) && batch_rows > 0;
    let dgemm_executed = score_mode == "fp64-dgemm-staged-scores" && batch_rows > 0;
    let allocated = cuda_device_allocation_snapshot();
    let q8_input_pack_elisions = owner.core.as_ref().unwrap().workspace.cublas.q8_input_pack_elisions();
    let gdn_bf16_pair_submissions = owner.core.as_ref().unwrap().workspace.cublas.gdn_bf16_pair_submissions();
    drop(owner.core.take());
    drop(owner.prefix_profile.take());
    let after = cuda_device_allocation_snapshot();
    let workspace_bytes = allocated
        .live_bytes
        .checked_sub(before.live_bytes)
        .ok_or_else(|| invalid("canonical prefix allocation bytes"))?;
    let allocations = after
        .successful_allocations
        .checked_sub(before.successful_allocations)
        .ok_or_else(|| invalid("canonical prefix allocation counter"))?;
    let frees = after
        .successful_frees
        .checked_sub(before.successful_frees)
        .ok_or_else(|| invalid("canonical prefix free counter"))?;
    if after.live_bytes != before.live_bytes
        || after.live_allocations != before.live_allocations
        || allocations != frees
        || workspace_bytes == 0
    {
        return Err(invalid("canonical prefix workspace closure"));
    }
    owner.driver.cursor = prompt.len();
    owner.driver.poisoned = false;
    let wall_ms = timer.elapsed().as_secs_f64() * 1000.0;
    let serial = schedule.serial_rows;
    let mut receipt = json!({"schema":if qpn_selected_count > 0 {"qwen35-native-qpn-ffn-original-input-prefill-v1"}
            else if full_original_selected_count > 0 {"qwen35-native-llama-full-original-prefill-v1"}
            else if original_input_selected_count > 0 {"qwen35-native-llama-original-input-prefill-v1"}
            else if llama_input_selected_count > 0 {"qwen35-native-llama-q8-input-prefill-v1"}
            else if staged_scores {"qwen35-native-staged-score-canonical-prefill-v1"}
            else if width == B {"qwen35-native-full-canonical-prefill-audit-v1"}
            else {"qwen35-native-wide-canonical-prefill-v1"},
        "arithmetic_policy":if qpn_selected_count > 0 {"qpn_original_f16_ffn_llama_other_original_rotated_prefix_v1"}
            else if llama_norm_selected_count > 0 {"llama_original_rotated_prefix512_bf16_rne_gdn_scalars_llama_norm_v1"}
            else if bf16_scalar_selected_count > 0 {"llama_original_rotated_prefix512_bf16_rne_gdn_scalars_v1"}
            else if full_original_selected_count > 0 {prefill_input_integer::FULL_ORIGINAL_ARITHMETIC_POLICY}
            else if original_input_selected_count > 0 {prefill_input_integer::ORIGINAL_ARITHMETIC_POLICY}
            else if llama_input_selected_count > 0 {prefill_input_integer::LLAMA_ARITHMETIC_POLICY}
            else if input_integer_selected_count > 0 {prefill_input_integer::ARITHMETIC_POLICY}
            else if integer_mma_selected_count > 0 {prefill_integer_mma::arithmetic_policy(width)}
            else if owner.driver.tensor_ffn_down_compensated {"canonical_except_nvfp4_down_b128_factored_hilo4096_v1"}
            else if tensor_ffn_down_selected_count > 0 {"canonical_except_packed_q8_f16_f32_tensor_ffn_down_tile896_v1"}
            else if dgemm_executed {"native_quantized_canonical_norm_fp64_dgemm_scores_ordered_full_core_v1"}
            else {"native_quantized_canonical_norm_scalar_full_core_v1"},
        "prefix_tokens":prefix,"batch_width":width,"batch_count":prefix/width,
        "q8_input_pack_elisions":q8_input_pack_elisions,
        "gdn_bf16_pair_submissions":gdn_bf16_pair_submissions,
        "llama_input_norm_selected_count":llama_norm_selected_count,
        "llama_input_norm_executed_count":llama_norm_selected_count*(batch_rows/width),
        "initialized_history_start":capture_start,"global_boundary_audit":global_boundary,
        "global_remainder_dispatch_qualified":false,
        "canonical_gdn_token_grid_layers":gdn_token_grid_selected_count,
        "canonical_gdn_token_grid_launches":2*gdn_token_grid_selected_count*(batch_rows/width),
        "decode_gdn_token_grid":false,
        "requested_prefix_batch_rows":requested_rows,"verify_batch_width":B,
        "prefix_schedule_policy":if width == B {"canonical_b8_reference"}else{"canonical_grid_wide_input_only"},
        "projection_row_reuse_requested":owner.driver.projection_row_reuse,
        "projection_row_reuse_selected_count":row_reuse_selected_count,
        "tensor_ffn_down_requested_rows":owner.driver.tensor_ffn_down_rows,
        "tensor_ffn_down_compensated_nvfp4":owner.driver.tensor_ffn_down_compensated,
        "tensor_ffn_down_selected_count":tensor_ffn_down_selected_count,
        "tensor_ffn_down_executed_projection_count":tensor_ffn_down_selected_count*(batch_rows/width),
        "tensor_ffn_down_working_weight_bytes":if tensor_ffn_down_selected_count>0 {896*17408*2}else{0},
        "tensor_ffn_down_borrowed_input_low_bytes":if owner.driver.tensor_ffn_down_compensated {width*17408*2}else{0},
        "tensor_ffn_down_reuses_existing_plan_scratch":tensor_ffn_down_selected_count>0,
        "tensor_ffn_down_numerics_qualified":false,
        "history_and_decode_tensor_ffn_down":false,
        "integer_mma_ffn_requested":owner.driver.integer_mma_ffn.is_some(),
        "integer_mma_ffn_selected_projection_count":integer_mma_selected_count,
        "integer_mma_ffn_executed_projection_count":integer_mma_selected_count*(batch_rows/width),
        "integer_mma_ffn_scratch_bytes":if integer_mma_selected_count>0 {prefill_integer_mma::scratch_bytes(width)}else{0},
        "integer_mma_ffn_reuses_existing_plan_scratch":integer_mma_selected_count>0,
        "history_and_decode_integer_mma_ffn":if owner.driver.wide_history_capture {Value::Null}else{json!(false)},
        "history_integer_mma_ffn":owner.driver.wide_history_capture && !owner.driver.llama_history_full,"decode_integer_mma_ffn":false,
        "integer_mma_ffn_model_qualified":false,
        "integer_mma_input_requested":owner.driver.integer_mma_input && !owner.driver.llama_f16_input && !owner.driver.llama_original_input && !owner.driver.llama_full_original,
        "llama_full_original_requested":owner.driver.llama_full_original,
        "llama_bf16_scalars_prefix_requested":owner.driver.llama_bf16_scalars,
        "llama_bf16_scalars_selected_projection_count":bf16_scalar_selected_count,
        "llama_bf16_scalars_executed_projection_count":bf16_scalar_selected_count*(batch_rows/width),
        "history_and_decode_llama_bf16_scalars":false,
        "history_llama_bf16_scalars":owner.driver.wide_history_capture && owner.driver.llama_history_full,
        "decode_llama_bf16_scalars":false,
        "llama_full_original_selected_projection_count":full_original_selected_count,
        "qpn_ffn_selected_projection_count":qpn_selected_count,
        "qpn_ffn_executed_projection_count":qpn_selected_count*(batch_rows/width),
        "llama_full_original_executed_projection_count":full_original_selected_count*(batch_rows/width),
        "llama_full_original_swiglu_count":if full_original_selected_count>0 {(64-qpn_selected_count/3)*(batch_rows/width)}else{0},
        "llama_full_original_reuses_existing_plan_scratch":full_original_selected_count>0,
        "llama_full_original_max_decoded_weight_bytes":if full_original_selected_count>0 {17408*5120*2}else{0},
        "llama_full_original_max_decoded_input_bytes":if full_original_selected_count>0 {512*17408*2}else{0},
        "llama_full_original_linear_output_location":if full_original_selected_count>0 {"dead_qkv_scratch_before_next_layer"}else{"unchanged"},
        "history_and_decode_llama_full_original":false,"llama_full_original_model_qualified":false,
        "history_llama_full_original":owner.driver.wide_history_capture && owner.driver.llama_history_full,
        "decode_llama_full_original":false,
        "llama_original_input_requested":owner.driver.llama_original_input,
        "llama_original_input_selected_projection_count":original_input_selected_count,
        "llama_original_input_executed_projection_count":original_input_selected_count*(batch_rows/width),
        "llama_original_input_reuses_existing_plan_scratch":original_input_selected_count>0,
        "llama_original_input_max_decoded_weight_bytes":if original_input_selected_count>0 {12288*5120*2}else{0},
        "llama_original_input_max_decoded_input_bytes":if original_input_selected_count>0 {512*5120*2}else{0},
        "history_and_decode_llama_original_input":false,"llama_original_input_model_qualified":false,
        "llama_f16_input_requested":owner.driver.llama_f16_input,
        "llama_f16_input_selected_projection_count":llama_input_selected_count,
        "llama_f16_input_executed_projection_count":llama_input_selected_count*(batch_rows/width),
        "llama_f16_input_reuses_existing_plan_scratch":llama_input_selected_count>0,
        "llama_f16_input_max_decoded_weight_bytes":if llama_input_selected_count>0 {12288*5120*2}else{0},
        "llama_f16_input_max_decoded_input_bytes":if llama_input_selected_count>0 {512*5120*2}else{0},
        "history_and_decode_llama_f16_input":false,"llama_f16_input_model_qualified":false,
        "integer_mma_input_selected_projection_count":input_integer_selected_count,
        "integer_mma_input_executed_projection_count":input_integer_selected_count*(batch_rows/width),
        "integer_mma_input_scratch_bytes":if input_integer_selected_count>0 {prefill_input_integer::SCRATCH_BYTES}else{0},
        "integer_mma_input_reuses_existing_plan_scratch":input_integer_selected_count>0,
        "history_and_decode_integer_mma_input":if owner.driver.wide_history_capture {Value::Null}else{json!(false)},
        "history_integer_mma_input":owner.driver.wide_history_capture && !owner.driver.llama_history_full,"decode_integer_mma_input":false,
        "wide_history_capture_requested":owner.driver.wide_history_capture,
        "integer_mma_input_model_qualified":false,
        "prefix_tail_policy":"canonical-b8-reuse",
        "b8_tail_batch_count":schedule.b8_tail_rows/B,"b8_tail_rows":schedule.b8_tail_rows,
        "tail_workspace_reused":schedule.b8_tail_rows>0,"tail_device_allocations":tail_device_allocations,
        "serial_tail_rows":serial,"feature_capture":false,"padding_tokens":0,
        "generated_warmup_tokens":0,"publication_enabled":false,
        "operation_bind_calls":1+usize::from(schedule.b8_tail_rows>0),"conditioning_wall_ms":wall_ms,
        "timing_scope":"prefix_workspace_build_submit_drain_release",
        "terminal_proof":"stream_synchronize_returned_success",
        "requested_prefix_score_mode":requested_score_mode,"prefix_score_mode":score_mode,
        "staged_score_query_rows":if staged_scores {128}else{B},
        "canonical_warp_scores_executed":score_mode=="canonical-warp-staged-scores" && batch_rows>0,
        "large_prefill_dgemm_executed":dgemm_executed,
        "staged_score_model_numerics_qualified":false,
        "workspace_bytes":workspace_bytes,"workspace_released_bytes":workspace_bytes,
        "live_bytes_before":before.live_bytes,"live_bytes_after":after.live_bytes,
        "allocations":allocations,"frees":frees,
        "workspace_released_before_auxiliary_and_decode":true,"product_qualification":false});
    if let Some(profile) = prefix_attribution {
        receipt["canonical_prefix_stage_profile"] = profile;
    }
    if let Some(trace) = &owner.driver.prefix_boundary_audit {
        receipt["layer_boundary_witness"] = trace.finish()?;
        receipt["performance_evidence"] = json!(false);
    }
    Ok((
        AutoregressivePrefillRun {
            host_wall_ms: wall_ms,
            batched_report: None,
            polls: 0,
            proof: true, // Real stream synchronization above, never inferred.
            inter_layer_d2d_copy_count: serial * 63,
            inter_layer_d2d_bytes: serial * 63 * H * 4,
        },
        receipt,
    ))
}

#[cfg(feature = "qwen35-session-continuation")]
fn validate_canonical_prefix_with_history(
    prompt: usize,
    prefix: usize,
    growing: bool,
) -> Result<()> {
    if growing {
        // Public profile admission owns the legacy 8K/16K ceilings. This shared
        // initialized-ring partition must also admit the full resident arena;
        // rolling_position_budget has already reserved generation/verify rows.
        if !(1..=crate::resident_session_profile::CAPACITY).contains(&prompt)
            || prefix != prompt.saturating_sub(2048)
        {
            return Err(invalid("growing canonical prefix initialized extent"));
        }
        Ok(())
    } else {
        validate_canonical_prefix_extent(prompt, prefix)
    }
}

#[cfg(feature = "qwen35-session-continuation")]
fn validate_canonical_prefix_extent(prompt: usize, prefix: usize) -> Result<()> {
    let admitted = (B..=16384).contains(&prompt)
        || (cfg!(feature = "qwen35-session-context32-audit") && (32768..=35840).contains(&prompt))
        || (cfg!(feature = "qwen35-session-context64-audit") && (65536..=69632).contains(&prompt))
        || (cfg!(feature = "qwen35-session-near192-audit") && (194560..=196474).contains(&prompt));
    if !admitted || prefix < B || prefix >= prompt {
        return Err(invalid(
            "canonical prefix audit requires bounded real prefix",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "qwen35-session-continuation")]
    #[test]
    fn growing_capture_and_zero_prefix_partition_real_tokens_exactly() {
        for n in [
            1usize, 2, 7, 8, 9, 127, 2047, 2048, 2049, 2055, 2056, 8192, 16384, 16385, 33003,
            47590, 65537, 194560, 196474, 196608,
        ] {
            let rows = n.min(2048);
            let prefix = n - rows;
            validate_prompt_capture_extent(n, rows, 196608, true).unwrap();
            validate_canonical_prefix_with_history(n, prefix, true).unwrap();
            let consumed = (0..prefix).chain(prefix..n).collect::<Vec<_>>();
            assert_eq!(consumed, (0..n).collect::<Vec<_>>());
            if n < B {
                assert!(validate_prompt_capture(n, rows, 196608).is_err());
            }
        }
        for (n, rows) in [(0, 0), (7, 8), (9, 8), (2049, 2047)] {
            assert!(validate_prompt_capture_extent(n, rows, 196608, true).is_err());
        }
        assert!(validate_canonical_prefix_with_history(2049, 0, true).is_err());
        assert!(validate_canonical_prefix_with_history(196609, 194561, true).is_err());
        assert!(validate_canonical_prefix_with_history(47590, 45543, true).is_err());
    }

    #[cfg(feature = "qwen35-session-continuation")]
    #[test]
    fn canonical_prefix_exact_batch_and_serial_remainder_geometry() {
        for prefix in 6144..=7168 {
            validate_canonical_prefix_extent(prefix + 2048, prefix).unwrap();
            let rows = (0..prefix / B * B)
                .step_by(B)
                .flat_map(|s| s..s + B)
                .chain(prefix / B * B..prefix)
                .collect::<Vec<_>>();
            assert_eq!(rows, (0..prefix).collect::<Vec<_>>());
        }
        for (prompt, prefix) in [(0, 0), (8, 8), (8192, 7), (16385, 14337)] {
            assert!(validate_canonical_prefix_extent(prompt, prefix).is_err());
        }
    }

    #[cfg(feature = "qwen35-session-continuation")]
    #[test]
    fn canonical32_and64_prefixes_are_compiled_and_extent_closed() {
        for n in [
            16385, 32767, 32768, 32769, 33472, 35840, 35841, 65535, 65536, 65537, 69632, 69633,
        ] {
            assert_eq!(
                validate_canonical_prefix_extent(n, n - 2048).is_ok(),
                (cfg!(feature = "qwen35-session-context32-audit") && (32768..=35840).contains(&n))
                    || (cfg!(feature = "qwen35-session-context64-audit")
                        && (65536..=69632).contains(&n))
            );
        }
    }

    #[test]
    fn real_tail_geometry_covers_each_token_once_without_padding() {
        for rows in B..=2048 {
            for prompt in [rows, rows + 1, 131072, 196352] {
                validate_prompt_capture(prompt, rows, 196608).unwrap();
                let start = prompt - rows;
                let batches = (0..rows / B * B)
                    .step_by(B)
                    .flat_map(|r| start + r..start + r + B);
                let tail = start + rows / B * B..prompt;
                assert_eq!(
                    batches.chain(tail).collect::<Vec<_>>(),
                    (start..prompt).collect::<Vec<_>>()
                );
                assert!(rows % B < B);
            }
        }
    }

    #[cfg(feature = "qwen35-session-continuation")]
    #[test]
    fn near192_prefix_keeps_full_verify_reserve() {
        for n in [194559, 194560, 195264, 196474, 196475, 196480, 196608] {
            assert_eq!(
                validate_canonical_prefix_extent(n, n - 2048).is_ok(),
                cfg!(feature = "qwen35-session-near192-audit") && (194560..=196474).contains(&n)
            );
        }
    }

    #[test]
    fn short_empty_overflowed_or_unbounded_capture_is_rejected() {
        for (prompt, rows, cap) in [
            (0, 8, 196608),
            (7, 7, 196608),
            (8, 0, 196608),
            (2049, 2049, 196608),
            (8, 9, 196608),
            (196609, 2048, 196608),
        ] {
            assert!(validate_prompt_capture(prompt, rows, cap).is_err());
        }
    }
}
