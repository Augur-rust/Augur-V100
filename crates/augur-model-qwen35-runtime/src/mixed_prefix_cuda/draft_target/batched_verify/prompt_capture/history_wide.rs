//! Isolated initialized2048 history; temporary workspace, same Driver authority.
use super::*;

// Executed paths, not the presence of a bound but superseded scalar kernel.
// Order: integer input, integer FFN, original projections, BF16 scalars, GDN grid.
#[cfg(test)]
fn validate_projection_counts(full: bool, counts: [usize; 5]) -> Result<()> {
    validate_profile_counts(full, counts, 0)
}

fn validate_profile_counts(full: bool, counts: [usize; 5], qpn: usize) -> Result<()> {
    if qpn != 0 {
        if !full || counts[0] != 0 || counts[1] != 0 || counts[4] != 0 {
            return Err(invalid("QPN history requires original input profile"));
        }
        return prefill_input_integer::validate_original_inventory(counts[2], qpn, counts[3]);
    }
    let expected = if full { [0, 0, 400, 96, 0] }
        else { [144, 192, 0, 0, if cfg!(feature="qwen35-bf16-warp-row") {48} else {0}] };
    if counts != expected { return Err(invalid("initialized history projection recipe coverage")); }
    Ok(())
}

pub(super) fn capture(d: &mut Driver, prompt: &[usize], epsilon: f32, theta: f32) -> Result<Value> {
    let width = history_capture::WIDTH;
    let rows = history_capture::HISTORY;
    if !d.wide_history_capture || d.context_rows != rows || prompt.len() < rows {
        return Err(invalid("wide initialized history admission"));
    }
    if d.llama_history_full && !(d.llama_full_original && d.llama_bf16_scalars && d.llama_rotated_kv) {
        return Err(invalid("full initialized history requires its complete llama recipe"));
    }
    let start = prompt.len() - rows;
    let (batch_start, _, scalar_tail) = if d.llama_global_chunks {
        history_capture::global_boundary_schedule(prompt.len())?
    } else { (start, start, 0) };
    let before = cuda_device_allocation_snapshot();
    let timer = Instant::now();
    d.poisoned = true;
    let mut owner = CaptureOwner {
        driver: d,
        core: None,
        prefix_profile: None,
        drained: false,
    };
    owner.driver.cursor = batch_start;
    owner.core = Some(BatchedVerifyAudit::allocate_prefill_rows(
        owner.driver,
        prompt.len(),
        epsilon,
        theta,
        width,
    )?);
    let gdn_token_grid_selected_count;
    let input_selected_count;
    let ffn_selected_count;
    let original_selected_count;
    let qpn_selected_count;
    let bf16_selected_count;
    {
        let core = owner
            .core
            .as_ref()
            .ok_or_else(|| invalid("wide history core"))?;
        let d = &mut *owner.driver;
        let mut operations = core.bind_operations(d)?;
        if prefill_input_integer::original_operand_count(&operations) == 400 {
            prefill_input_integer::validate_exact_input_bindings(&operations, width)?;
        } else if d.llama_history_full {
            // Same admitted B512 implementations as prefix. Binding reuses the
            // existing workspace; capture destination and block schedule stay fixed.
            prefill_input_integer::install_full_original(&mut operations, width, core, d)?;
            prefill_input_integer::install_bf16_scalars(&mut operations, width, core, d)?;
        }
        gdn_token_grid_selected_count = operations.iter().filter(|op|
            matches!(op, PrefillLayerOps::Linear(p) if p.canonical_gdn_prefill_batch.is_some()
                && !p.alpha.llama_full_original_enabled())).count();
        input_selected_count = prefill_input_integer::selected_count(&operations);
        ffn_selected_count = prefill_integer_mma::selected_count(&operations);
        original_selected_count = prefill_input_integer::full_original_selected_count(&operations);
        qpn_selected_count = prefill_input_integer::qpn_projection_count(&operations);
        bf16_selected_count = prefill_input_integer::bf16_scalar_selected_count(&operations);
        validate_profile_counts(d.llama_history_full, [input_selected_count, ffn_selected_count,
            original_selected_count, bf16_selected_count, gdn_token_grid_selected_count], qpn_selected_count)?;
        for row in (0..rows).step_by(width) {
            d.cursor = batch_start + row;
            let ids = prompt[d.cursor..d.cursor + width]
                .iter()
                .flat_map(|&id| (id as u32).to_le_bytes())
                .collect::<Vec<_>>();
            core.batch_ids.copy_from_host_sync(&d.stream, &ids)?;
            if row == width {
                // The existing same-stream H2D synchronization proves the
                // first real block completed. No extra event, wait or sleep;
                // fault tests can now target history instead of guessing time.
                eprintln!(
                    "QWEN35_HISTORY_CAPTURE_PROGRESS={}",
                    json!({
                        "schema":"qwen35-history-capture-progress-v1",
                        "pid":std::process::id(),"completed_rows":row - (start-batch_start),
                        "history_rows":rows,"history_start":start,
                        "prompt_tokens":prompt.len(),"batch_width":width,
                        "proof":"same_stream_h2d_sync_after_history_backbone",
                        "visible_tokens":0
                    })
                );
            }
            core.submit_initialized_history(d, &operations)?;
        }
        if scalar_tail == 0 { copy(
            &d.stream,
            d.plans[63].final_hidden_buffer(),
            0,
            core.workspace
                .hidden_output(63, width)?
                .buffer_view()
                .subview((width - 1) * H * 4, H * 4)?,
        )?; } else {
            // Diagnostic isolation: unchanged native scalar math, one actual
            // input at the global end. No B8/padded tail and no dispatch parity claim.
            let position = prompt.len() - 1;
            d.cursor = position;
            d.token.copy_from_host_sync(&d.stream, &(prompt[position] as u32).to_le_bytes())?;
            d.bind_gather(view(&d.token)?)?.submit_for_plan(&d.stream)?;
            d.check_session_cancellation("global scalar remainder submit")?;
            d.forward_backbone_capturing(position, Some((&d.features, rows - 1)), None)?;
        }
        d.head.submit(&d.stream)?;
        d.top1(view(&d.token)?)?;
    }
    owner.driver.stream.synchronize()?;
    owner.drained = true;
    let allocated = cuda_device_allocation_snapshot();
    drop(owner.core.take());
    let after = cuda_device_allocation_snapshot();
    let workspace = allocated
        .live_bytes
        .checked_sub(before.live_bytes)
        .ok_or_else(|| invalid("wide history workspace counter"))?;
    let allocations = after
        .successful_allocations
        .checked_sub(before.successful_allocations)
        .ok_or_else(|| invalid("wide history allocation counter"))?;
    let frees = after
        .successful_frees
        .checked_sub(before.successful_frees)
        .ok_or_else(|| invalid("wide history free counter"))?;
    if workspace == 0
        || allocations != frees
        || before.live_bytes != after.live_bytes
        || before.live_allocations != after.live_allocations
    {
        return Err(invalid("wide history workspace closure"));
    }
    owner.driver.cursor = prompt.len();
    owner.driver.poisoned = false;
    Ok(
        json!({"schema":"qwen35-canonical-wide-initialized-history-v1",
        "arithmetic_policy":if qpn_selected_count>0 {"qpn_original_f16_ffn_llama_other_initialized_history_v1"}
            else if owner.driver.llama_history_full {"llama_original400_bf16_scalars96_initialized_history_v1"}
            else if owner.driver.llama_rotated_kv {"integer_projection_history_with_llama_rotated_attention_v1"} else {prefill_input_integer::ARITHMETIC_POLICY},
        "rotated_full_attention":owner.driver.llama_rotated_kv,
        "llama_full_history_requested":owner.driver.llama_history_full,
        "llama_full_original_selected_projection_count":original_selected_count,
        "qpn_ffn_selected_projection_count":qpn_selected_count,
        "qpn_ffn_executed_projection_count":qpn_selected_count*(rows/width),
        "llama_full_original_executed_projection_count":original_selected_count*(rows/width),
        "llama_bf16_scalars_selected_projection_count":bf16_selected_count,
        "llama_bf16_scalars_executed_projection_count":bf16_selected_count*(rows/width),
        "llama_full_original_swiglu_count":if original_selected_count>0 {(64-qpn_selected_count/3)*(rows/width)} else {0},
        "verifier_and_decode_llama_history_math":false,
        "feature_start":start,"feature_end":prompt.len(),"feature_rows":rows,
        "batch_width":width,"batch_count":rows/width,"serial_tail_rows":scalar_tail,
        "global_boundary_audit":owner.driver.llama_global_chunks,
        "input_batch_start":batch_start,"input_rows":prompt.len()-batch_start,
        "first_capture_source_row":start-batch_start,
        "batch_feature_rows":rows-scalar_tail,"scalar_feature_rows":scalar_tail,
        "tail_arithmetic":"unchanged_native_scalar_not_llama_dispatch_qualified",
        "global_remainder_dispatch_qualified":false,
        "canonical_gdn_token_grid_layers":gdn_token_grid_selected_count,
        "canonical_gdn_token_grid_launches":2*gdn_token_grid_selected_count*(rows/width),
        "decode_gdn_token_grid":false,
        "output_head_calls":1,"feature_d2d_bytes":rows*5*H*4,"feature_copy_calls":rows*5,
        "capture_destination":"driver_initialized_features_direct","extra_feature_staging_bytes":0,
        "verifier_batch_width":B,"history_integer_ffn_executed_projection_count":ffn_selected_count*(rows/width),
        "history_integer_input_executed_projection_count":input_selected_count*(rows/width),
        "generated_warmup_tokens":0,"padding_tokens":0,"publication_enabled":false,
        "operation_bind_calls":1,"conditioning_wall_ms":timer.elapsed().as_secs_f64()*1000.0,
        "workspace_bytes":workspace,"workspace_released_bytes":workspace,
        "live_bytes_before":before.live_bytes,"live_bytes_with_workspace":allocated.live_bytes,
        "live_bytes_after":after.live_bytes,"allocations":allocations,"frees":frees,
        "workspace_released_before_auxiliary_and_decode":true,"product_qualification":false}),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn qpn_history_counts_remain_explicit_and_closed() {
        assert!(super::validate_profile_counts(true, [0, 0, 232, 96, 0], 168).is_ok());
        for (full, counts, qpn) in [(false, [0,0,232,96,0],168), (true,[0,0,400,96,0],168),
            (true,[0,0,233,96,0],167), (true,[0,0,232,95,0],168), (true,[0,1,232,96,0],168)] {
            assert!(super::validate_profile_counts(full, counts, qpn).is_err());
        }
    }
    use super::*;
    #[test]
    fn history_projection_counts_separate_executed_recipes() {
        let old = [144, 192, 0, 0, if cfg!(feature="qwen35-bf16-warp-row") {48} else {0}];
        let new = [0, 0, 400, 96, 0];
        assert!(validate_projection_counts(false, old).is_ok());
        assert!(validate_projection_counts(true, new).is_ok());
        assert!(validate_projection_counts(true, old).is_err());
        assert!(validate_projection_counts(false, new).is_err());
        for (enabled, expected) in [(false, old), (true, new)] {
            for i in 0..5 {
                let mut bad = expected; bad[i] += 1;
                assert!(validate_projection_counts(enabled, bad).is_err());
                if expected[i] > 0 {
                    let mut bad = expected; bad[i] -= 1;
                    assert!(validate_projection_counts(enabled, bad).is_err());
                }
            }
        }
    }
}
