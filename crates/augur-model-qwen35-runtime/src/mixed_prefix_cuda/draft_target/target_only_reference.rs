//! Diagnostic-only serial greedy reference after the EXACT request conditioning.
//! No draft, speculative verification, rollback, or auxiliary-residency timing.
use super::*;

pub(super) fn summarize_logits(logits: &[f32], selected: u32, output_index: usize) -> Result<Value> {
    if logits.len() < 2
        || selected as usize >= logits.len()
        || logits.iter().any(|x| !x.is_finite())
    {
        return Err(invalid(
            "reference logits invalid extent/selection/nonfinite",
        ));
    }
    let mut ids = (0..logits.len()).collect::<Vec<_>>();
    ids.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b)));
    let maximum = logits[ids[0]] as f64;
    if logits[selected as usize] as f64 != maximum {
        return Err(invalid("reference GPU selection is not the logits maximum"));
    }
    let log_sum = logits
        .iter()
        .map(|&x| (x as f64 - maximum).exp())
        .sum::<f64>()
        .ln();
    let bytes = logits
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();
    let top = ids
        .iter()
        .take(32)
        .map(|&id| {
            json!({"id":id,"logit":logits[id],
        "logprob":logits[id] as f64 - maximum - log_sum})
        })
        .collect::<Vec<_>>();
    Ok(
        json!({"output_index":output_index,"vocab_size":logits.len(),"selected_token":selected,
        "cpu_argmax_token":ids[0],"gpu_selection_is_maximum":true,
        "top1_top2_margin":maximum-logits[ids[1]] as f64,
        "logits_f32_le_sha256":sha256_hex(&bytes),"all_logits_finite":true,
        "full_logits_d2h_bytes":bytes.len(),"normalization":"cpu_f64_logsumexp_inventory_only",
        "top32":top}),
    )
}

fn capture_logits(d: &Driver, output_index: usize) -> Result<Value> {
    let receipt = d.stream.submission_receipt()?;
    let logits = read_f32_buffer(&d.stream, &d.head.logits, receipt)?;
    if logits.len() != V {
        return Err(invalid("reference logits vocabulary extent"));
    }
    summarize_logits(&logits, d.anchor, output_index)
}

pub(super) fn validate_extent(cursor: usize, end: usize, max_tokens: usize) -> Result<()> {
    if !(1..=512).contains(&max_tokens)
        || cursor
            .checked_add(max_tokens - 1)
            .is_none_or(|last| last > end)
    {
        return Err(invalid(
            "target-only reference exceeds the frozen RoPE/position extent",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    mut d: Driver,
    mut report: Value,
    tokenizer: &tokenizers::Tokenizer,
    eos: &[u32],
    max_tokens: usize,
    end: usize,
    before: augur_backend::CudaDeviceAllocationSnapshot,
    audit_logits: bool,
    matched_auxiliary: Option<(DFlashQ8CudaBlock, Option<DFlashQ8CudaWeights>, Option<ResidentAuxiliaryOwners>)>,
) -> Result<Value> {
    validate_extent(d.cursor, end, max_tokens)?;
    let initial_cursor = d.cursor;
    let mut tokens = vec![d.anchor];
    let mut stop = clip_publication(
        &mut tokens,
        Some(PublicationBudget {
            remaining: max_tokens,
            eos,
        }),
    )?;
    let mut steps = Vec::new();
    let mut logit_probes = Vec::new();
    if audit_logits {
        logit_probes.push(capture_logits(&d, 0)?);
    }
    let started = Instant::now();
    while stop.is_none() {
        if d.cursor >= end || d.cursor >= d.capacity || tokens.len() >= max_tokens {
            return Err(invalid("target-only reference cursor/publication bound"));
        }
        let cursor_before = d.cursor;
        let input_token = d.anchor;
        // d.token is the previous GPU argmax. Never feed a candidate's tokens.
        // Driver::drop drains on any partial-submit failure before freeing owners.
        d.poisoned = true;
        let boundary_before = cuda_host_boundary_snapshot();
        d.bind_gather(view(&d.token)?)?.submit_for_plan(&d.stream)?;
        #[cfg(feature = "qwen35-raw-committed-state")]
        let capture_row = if d.raw_state.is_some() {
            d.advance_feature_window(1)?.capture_row(0)
        } else { None };
        #[cfg(not(feature = "qwen35-raw-committed-state"))]
        let capture_row = None;
        d.forward(d.cursor, capture_row)?;
        d.top1(view(&d.token)?)?;
        let next = words(&d.stream, &d.token, 0, 1)?[0];
        if next as usize >= V {
            return Err(invalid("target-only reference invalid argmax"));
        }
        let token_boundary = boundary_delta(boundary_before, cuda_host_boundary_snapshot())?;
        d.cursor += 1;
        d.anchor = next;
        d.generation += 1;
        let mut emitted = vec![next];
        stop = clip_publication(
            &mut emitted,
            Some(PublicationBudget {
                remaining: max_tokens - tokens.len(),
                eos,
            }),
        )?;
        tokens.extend_from_slice(&emitted);
        #[cfg(feature = "qwen35-raw-committed-state")]
        if let Some(capture) = &d.raw_state {
            let draft = &matched_auxiliary.as_ref().ok_or_else(|| invalid("raw reference auxiliary missing"))?.0;
            capture.reference_commit(&d, draft, input_token, stop.is_some())?;
        }
        let fingerprint = d.state_digest()?;
        d.poisoned = false;
        steps.push(json!({
            "cursor_before":cursor_before,"cursor_after":d.cursor,
            "input_token":input_token,"output_token":next,
            "target_forward_count":1,"token_d2h_bytes":4,
            "token_boundary":token_boundary,"committed_state_fingerprint":fingerprint,
            "state_audit_d2h_bytes":32
        }));
        if audit_logits && tokens.len() - 1 == 27 {
            logit_probes.push(capture_logits(&d, 27)?);
        }
    }
    d.stream.synchronize()?;
    let diagnostic_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let visible_end = tokens
        .iter()
        .position(|id| eos.contains(id))
        .unwrap_or(tokens.len());
    let output = &mut report["request_output"];
    output["token_ids"] = json!(tokens);
    output["visible_token_ids"] = json!(&tokens[..visible_end]);
    output["decoded_text"] = json!(tokenizer
        .decode(&tokens[..visible_end], false)
        .map_err(|e| invalid(e.to_string()))?);
    output["stop_reason"] =
        json!(stop.ok_or_else(|| invalid("target-only reference missing stop"))?);
    report["final_cursor"] = json!(d.cursor);
    report["initial_cursor"] = json!(initial_cursor);
    report["steps"] = json!(steps);
    if audit_logits {
        report["reference_logit_audit"] = json!({"schema":"qwen35-reference-logit-audit-v1",
            "requested_output_indices":[0,27],"probes":logit_probes,
            "readback_is_diagnostic_only":true,"sampling_modified":false,
            "performance_comparison_permitted":false,"product_ready":false});
    }
    report["diagnostic_wall_including_every_state_audit_ms"] = json!(diagnostic_wall_ms);
    #[cfg(feature = "qwen35-raw-committed-state")]
    if let Some(capture) = &d.raw_state { report["raw_committed_state"] = capture.finish(&d)?; }
    let auxiliary_present = matched_auxiliary.is_some();
    drop(d);
    drop(matched_auxiliary);
    let after = cuda_device_allocation_snapshot();
    let clean =
        after.live_bytes == before.live_bytes && after.live_allocations == before.live_allocations;
    if !clean {
        return Err(invalid("target-only reference cleanup did not close"));
    }
    report["cleanup"] = json!({"pass":true,"live_bytes":after.live_bytes,
        "allocations":after.successful_allocations-before.successful_allocations,
        "frees":after.successful_frees-before.successful_frees,"peak_live_bytes":after.peak_live_bytes});
    report["schema"] = json!("qwen35-target-only-matched-request-v1");
    report["status"] = json!("PASS_DIAGNOSTIC_TARGET_ONLY_MATCHED_REQUEST");
    report["decode_policy"] = json!("independent_serial_greedy_no_draft_v1");
    report["conditioning_policy"] = json!("shared_draft_target_request_conditioning_v1");
    report["state_audit_scheme"] = json!("128_plane_device_fingerprint_v1_noncryptographic");
    report["claims"] = json!({"draft_executed":false,"speculative_acceptance_executed":false,
        "rollback_executed":false,"auxiliary_residency_matched":auxiliary_present,"semantic_task_qualified":false,
        "product_ready":false,"visible_tps":false,"performance_comparison_permitted":false});
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logit_summary_preserves_argmax_margin_and_stable_hash() {
        let p = summarize_logits(&[1., 3., 2., -4.], 1, 27).unwrap();
        assert_eq!(p["cpu_argmax_token"], 1);
        assert_eq!(p["top1_top2_margin"], 1.);
        assert_eq!(p["full_logits_d2h_bytes"], 16);
        assert_eq!(p["top32"][0]["id"], 1);
        assert_eq!(
            p["logits_f32_le_sha256"],
            summarize_logits(&[1., 3., 2., -4.], 1, 0).unwrap()["logits_f32_le_sha256"]
        );
        assert_ne!(
            p["logits_f32_le_sha256"],
            summarize_logits(&[1., 3., 2., -3.], 1, 0).unwrap()["logits_f32_le_sha256"]
        );
    }

    #[test]
    fn logit_summary_rejects_nonfinite_wrong_argmax_and_bad_extent() {
        for (values, selected) in [
            (vec![], 0),
            (vec![1.], 0),
            (vec![1., 2.], 2),
            (vec![1., 2.], 0),
            (vec![f32::NAN, 2.], 1),
            (vec![1., f32::INFINITY], 1),
        ] {
            assert!(summarize_logits(&values, selected, 0).is_err());
        }
    }

    #[test]
    fn logit_summary_handles_ties_and_signed_zero_without_changing_selection() {
        let p = summarize_logits(&[0., -0., -1.], 1, 0).unwrap();
        assert_eq!(p["selected_token"], 1);
        assert_eq!(p["cpu_argmax_token"], 0);
        assert_eq!(p["top1_top2_margin"], 0.);
        assert_ne!(
            p["logits_f32_le_sha256"],
            summarize_logits(&[0., 0., -1.], 1, 0).unwrap()["logits_f32_le_sha256"]
        );
    }

    #[test]
    fn reference_extent_reserves_actual_future_inputs_and_no_phantom_initial_token() {
        assert!(validate_extent(100, 100, 1).is_ok());
        assert!(validate_extent(100, 107, 8).is_ok());
        assert!(validate_extent(100, 106, 8).is_err());
        assert!(validate_extent(100, 99, 1).is_err());
        assert!(validate_extent(100, 612, 512).is_ok());
        for n in [0, 513, usize::MAX] {
            assert!(validate_extent(100, 1000, n).is_err());
        }
        assert!(validate_extent(usize::MAX, usize::MAX, 2).is_err());
    }
}
