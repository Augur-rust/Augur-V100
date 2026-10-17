//! Event envelopes around actual canonical batch calls, not a shadow ABI.
//! Operators are unchanged. Timers are read only after the existing acceptance
//! receipt completes the same stream. Intervals include submission gaps and
//! instrumentation overhead and never authorize performance publication.
use super::*;

const PHASE_KEYS: [&str; 6] = [
    "input_projection_ms",
    "attention_or_recurrence_ms",
    "output_projection_post_norm_ms",
    "ffn_gate_up_ms",
    "ffn_activation_quantize_ms",
    "ffn_down_residual_ms",
];

pub(super) fn diagnostic_switch(value: Option<&std::ffi::OsStr>) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(value) if value == "1" => Ok(true),
        _ => Err(invalid("verify layer profile requires explicit diagnostic switch exactly1")),
    }
}

pub(super) struct CanonicalProfile {
    start: OwnedCudaEvent,
    layers: Vec<[OwnedCudaEvent; 2]>,
    phases: Vec<[OwnedCudaEvent; 5]>,
    head: OwnedCudaEvent,
    stop: OwnedCudaEvent,
}

fn partition_gap(parts: &[f64], total: f64) -> Result<f64> {
    checked_partition_gap(parts, total, 65)
}

fn checked_partition_gap(parts: &[f64], total: f64, expected: usize) -> Result<f64> {
    if parts.len() != expected
        || !total.is_finite()
        || total <= 0.0
        || parts.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(invalid("canonical profile invalid partition extent/times"));
    }
    let remainder = total - parts.iter().sum::<f64>();
    if !remainder.is_finite() || remainder < -(total * 0.0001).max(0.1) {
        return Err(invalid("canonical profile overlapping event partition"));
    }
    Ok(remainder.max(0.0))
}

fn record(event: &OwnedCudaEvent, stream: &OwnedCudaStream) -> Result<()> {
    event.record_for_submission(stream, stream.submission_receipt()?)?;
    Ok(())
}

#[cfg(feature = "qwen35-session-continuation")]
pub(super) fn prefix_sampling(
    value: Option<&std::ffi::OsStr>,
    width: usize,
    prefix: usize,
) -> Result<Option<Vec<usize>>> {
    match value {
        None => return Ok(None),
        Some(value) if value == "1" => {}
        _ => {
            return Err(invalid(
                "canonical prefix attribution switch must be exactly 1",
            ))
        }
    }
    if width != B || prefix < B {
        return Err(invalid(
            "canonical prefix attribution requires real reference B8 inputs",
        ));
    }
    let count = prefix / B;
    let mut selected = vec![0, count / 2, count - 1];
    selected.sort_unstable();
    selected.dedup();
    Ok(Some(selected))
}

pub(super) fn validate_profiled_prefix_rows(
    rows: usize,
    arena_rows: usize,
    input_only: bool,
    canonical_full_core: bool,
    other_diagnostics: bool,
) -> Result<()> {
    if rows != B || arena_rows != B || !input_only || !canonical_full_core || other_diagnostics {
        return Err(invalid(
            "prefix attribution requires unshadowed input-only canonical B8",
        ));
    }
    Ok(())
}

#[cfg(feature = "qwen35-session-continuation")]
fn prefix_sample(mut report: Value, index: usize) -> Result<Value> {
    if report["schema"] != "qwen35-canonical-target-layer-profile-v2"
        || report["tokens"] != B
        || report["instrumented_only"] != true
        || report["accounting_valid"] != true
    {
        return Err(invalid(
            "canonical prefix sample requires complete canonical phase report",
        ));
    }
    let object = report
        .as_object_mut()
        .ok_or_else(|| invalid("prefix phase object"))?;
    let marker = object
        .remove("target_head_total_ms")
        .ok_or_else(|| invalid("prefix marker interval"))?;
    if !marker
        .as_f64()
        .is_some_and(|value| value.is_finite() && value >= 0.0)
    {
        return Err(invalid("prefix terminal marker interval is not finite"));
    }
    let gap = object
        .remove("gather_features_and_between_stage_ms")
        .ok_or_else(|| invalid("prefix phase gap"))?;
    object.insert(
        "schema".into(),
        json!("qwen35-canonical-prefix-layer-profile-v1"),
    );
    object.insert("batch_index".into(), json!(index));
    object.insert("terminal_marker_interval_ms".into(), marker);
    object.insert("gather_and_between_stage_ms".into(), gap);
    object.insert("output_head_executed".into(), json!(false));
    object.insert("feature_capture".into(), json!(false));
    object.insert("publication_enabled".into(), json!(false));
    object.insert(
        "enclosing_scope".into(),
        json!("real_canonical_prefix_gather_backbone_terminal_markers_only"),
    );
    object.insert(
        "completion".into(),
        json!("explicit_same_stream_enclosing_sample_drain_before_timer_read"),
    );
    object.insert("excluded".into(), json!("other_prefix_batches_scalar_tail_initialized_draft_history_auxiliary_load_output_heads_features_acceptance_commit_checkpoints"));
    Ok(report)
}

#[cfg(feature = "qwen35-session-continuation")]
pub(super) fn prefix_summary(prefix: usize, samples: Vec<Value>) -> Result<Value> {
    let selected = prefix_sampling(Some(std::ffi::OsStr::new("1")), B, prefix)?.unwrap();
    if samples.len() != selected.len()
        || samples.iter().zip(&selected).any(|(sample, &index)| {
            sample["schema"] != "qwen35-canonical-prefix-layer-profile-v1"
                || sample["batch_index"].as_u64() != Some(index as u64)
                || sample["cursor"].as_u64() != Some((index * B) as u64)
                || sample["tokens"] != B
                || sample["output_head_executed"] != false
                || sample["feature_capture"] != false
                || sample["publication_enabled"] != false
                || sample["accounting_valid"] != true
        })
    {
        return Err(invalid(
            "prefix phase samples are missing, unordered or not bound to real input rows",
        ));
    }
    let sampled_rows = selected.len() * B;
    Ok(json!({"schema":"qwen35-canonical-prefix-stage-profile-v1",
        "instrumented_only":true,"performance_claim_permitted":false,
        "prefix_tokens":prefix,"batch_rows":B,"sampled_batch_indices":selected,
        "sampled_real_tokens":sampled_rows,"unprofiled_prefix_tokens":prefix-sampled_rows,
        "sampling":"first_middle_last_complete_real_B8_batches_not_whole_prefill_extrapolation",
        "timing_scope":"cuda_event_intervals_include_submission_gaps_and_instrumentation_not_kernel_only",
        "sample_completion":"explicit_same_stream_enclosing_sample_drain",
        "output_head_executed":false,"feature_capture":false,"publication_enabled":false,
        "samples":samples}))
}

impl CanonicalProfile {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            start: OwnedCudaEvent::create(0)?,
            layers: (0..64)
                .map(|_| Ok([OwnedCudaEvent::create(0)?, OwnedCudaEvent::create(0)?]))
                .collect::<Result<_>>()?,
            phases: (0..64)
                .map(|_| {
                    Ok([
                        OwnedCudaEvent::create(0)?,
                        OwnedCudaEvent::create(0)?,
                        OwnedCudaEvent::create(0)?,
                        OwnedCudaEvent::create(0)?,
                        OwnedCudaEvent::create(0)?,
                    ])
                })
                .collect::<Result<_>>()?,
            head: OwnedCudaEvent::create(0)?,
            stop: OwnedCudaEvent::create(0)?,
        })
    }

    pub(super) fn start(&self, d: &Driver) -> Result<()> {
        record(&self.start, &d.stream)
    }

    pub(super) fn layer(&self, d: &Driver, layer: usize, end: bool) -> Result<()> {
        let events = self
            .layers
            .get(layer)
            .ok_or_else(|| invalid("canonical profile layer extent"))?;
        record(&events[usize::from(end)], &d.stream)
    }

    pub(super) fn head(&self, d: &Driver) -> Result<()> {
        record(&self.head, &d.stream)
    }

    pub(super) fn phase_events(&self, layer: usize) -> Result<&[OwnedCudaEvent; 5]> {
        self.phases
            .get(layer)
            .ok_or_else(|| invalid("canonical phase layer extent"))
    }

    pub(super) fn stop(&self, d: &Driver) -> Result<()> {
        record(&self.stop, &d.stream)
    }

    #[cfg(feature = "qwen35-session-continuation")]
    pub(super) fn prefix_report(&self, d: &Driver, index: usize) -> Result<Value> {
        // The caller has drained this exact enclosing prefix batch. Head/stop
        // are adjacent terminal markers; no output head or feature capture ran.
        prefix_sample(self.report(d)?, index)
    }

    pub(super) fn report(&self, d: &Driver) -> Result<Value> {
        let total = f64::from(self.start.elapsed_ms(&self.stop)?);
        let mut parts = Vec::with_capacity(65);
        let mut layers = Vec::with_capacity(64);
        let mut linear_ms = 0.0;
        let mut full_ms = 0.0;
        let mut phase_totals = [[0.0_f64; 6]; 2];
        for (layer, pair) in self.layers.iter().enumerate() {
            let ms = f64::from(pair[0].elapsed_ms(&pair[1])?);
            let full = layer % 4 == 3;
            if full {
                full_ms += ms;
            } else {
                linear_ms += ms;
            }
            parts.push(ms);
            let events = self.phase_events(layer)?;
            let mut previous = &pair[0];
            let mut phase_parts = [0.0; 6];
            for (index, end) in events.iter().chain(std::iter::once(&pair[1])).enumerate() {
                phase_parts[index] = f64::from(previous.elapsed_ms(end)?);
                phase_totals[usize::from(full)][index] += phase_parts[index];
                previous = end;
            }
            checked_partition_gap(&phase_parts, ms, 6)?;
            // Adjacent event intervals should telescope; tolerate timer resolution only.
            if (phase_parts.iter().sum::<f64>() - ms).abs() > (ms * 0.0001).max(0.002) {
                return Err(invalid("canonical phase partition incomplete"));
            }
            let phase_object = |values: &[f64; 6]| -> Value {
                Value::Object(
                    PHASE_KEYS
                        .iter()
                        .zip(values)
                        .map(|(key, value)| (key.to_string(), json!(value)))
                        .collect(),
                )
            };
            layers.push(
                json!({"layer":layer,"kind":if full{"full_attention"}else{"linear_attention"},
                "canonical_layer_total_ms":ms,"phases":phase_object(&phase_parts)}),
            );
        }
        let head_ms = f64::from(self.head.elapsed_ms(&self.stop)?);
        parts.push(head_ms);
        let gap_ms = partition_gap(&parts, total)?;
        Ok(json!({
            "schema":"qwen35-canonical-target-layer-profile-v2",
            "instrumented_only":true,"performance_claim_permitted":false,
            "cursor":d.cursor,"generation":d.generation,"tokens":B,
            "arithmetic_policy":"native_quantized_canonical_norm_scalar_full_core_v1",
            "timing_scope":"cuda_event_intervals_include_submission_gaps_not_kernel_only",
            "layer_scope":"whole_production_layer_including_projections_attention_ffn",
            "enclosing_scope":"canonical_target_gather_backbone_features_and_head_only",
            "excluded":"draft_forward_draft_head_selector_checkpoint_acceptance_commit_replay",
            "event_records":451,"linear_layer_total_ms":linear_ms,
            "full_layer_total_ms":full_ms,"target_head_total_ms":head_ms,
            "enclosing_event_ms":total,"accounted_stage_ms":parts.iter().sum::<f64>(),
            "gather_features_and_between_stage_ms":gap_ms,"accounting_valid":true,
            "completion":"existing_same_stream_acceptance_receipt_before_timer_read",
            "phase_scope":"canonical_calls_input_norm_and_projections_core_output_residual_postnorm_ffn_gate_up_activation_down_residual",
            "phase_totals": {
                "linear_attention": PHASE_KEYS.iter().zip(phase_totals[0]).map(|(key,v)|
                    (key.to_string(),json!(v))).collect::<serde_json::Map<_,_>>(),
                "full_attention": PHASE_KEYS.iter().zip(phase_totals[1]).map(|(key,v)|
                    (key.to_string(),json!(v))).collect::<serde_json::Map<_,_>>()
            },
            "layers":layers
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verify_layer_profile_is_explicit_diagnostic_only() {
        assert!(!diagnostic_switch(None).unwrap());
        assert!(diagnostic_switch(Some("1".as_ref())).unwrap());
        for value in ["", "0", "true", " 1", "1 "] {
            assert!(diagnostic_switch(Some(value.as_ref())).is_err());
        }
    }

    #[test]
    fn canonical_profile_partition_is_nonoverlapping_and_bounded() {
        assert_eq!(partition_gap(&[1.0; 65], 67.0).unwrap(), 2.0);
        assert_eq!(partition_gap(&[1.0; 65], 65.0).unwrap(), 0.0);
        assert_eq!(partition_gap(&[1.0; 65], 64.99).unwrap(), 0.0);
        assert!(partition_gap(&[1.0; 65], 64.0).is_err());
    }

    #[test]
    fn canonical_profile_rejects_invalid_extents_and_nonfinite_or_negative_times() {
        for n in [0, 1, 64, 66] {
            assert!(partition_gap(&vec![1.0; n], 67.0).is_err());
        }
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let mut parts = [1.0; 65];
            parts[32] = bad;
            assert!(partition_gap(&parts, 67.0).is_err());
            assert!(partition_gap(&[1.0; 65], bad).is_err());
        }
        assert!(partition_gap(&[0.0; 65], 0.0).is_err());
    }

    #[test]
    fn canonical_six_phase_partition_has_exact_extent_and_no_overlap() {
        assert_eq!(checked_partition_gap(&[1.0; 6], 6.0, 6).unwrap(), 0.0);
        assert!(checked_partition_gap(&[1.0; 5], 6.0, 6).is_err());
        assert!(checked_partition_gap(&[1.0; 6], 5.0, 6).is_err());
        assert!(checked_partition_gap(&[f64::NAN; 6], 6.0, 6).is_err());
    }

    #[test]
    fn prefix_profile_cannot_verify_capture_or_borrow_wide_arena() {
        validate_profiled_prefix_rows(B, B, true, true, false).unwrap();
        for (rows, arena, input_only, canonical, diagnostic) in [
            (1, B, true, true, false),
            (16, 16, true, true, false),
            (B, 128, true, true, false),
            (B, B, false, true, false),
            (B, B, true, false, false),
            (B, B, true, true, true),
        ] {
            assert!(
                validate_profiled_prefix_rows(rows, arena, input_only, canonical, diagnostic)
                    .is_err()
            );
        }
    }

    #[cfg(feature = "qwen35-session-continuation")]
    #[test]
    fn prefix_sampling_is_default_off_real_bounded_and_without_padding() {
        use std::ffi::OsStr;
        assert_eq!(prefix_sampling(None, 128, 0).unwrap(), None);
        for value in ["", "0", "true", " 1", "1 "] {
            assert!(prefix_sampling(Some(OsStr::new(value)), B, 7037).is_err());
        }
        for (width, prefix) in [(B, 0), (B, 7), (128, 7037)] {
            assert!(prefix_sampling(Some(OsStr::new("1")), width, prefix).is_err());
        }
        for prefix in [8, 9, 16, 24, 7037, 65765, 193119, usize::MAX] {
            let indices = prefix_sampling(Some(OsStr::new("1")), B, prefix)
                .unwrap()
                .unwrap();
            assert!((1..=3).contains(&indices.len()));
            assert_eq!(indices.first(), Some(&0));
            assert_eq!(indices.last(), Some(&(prefix / B - 1)));
            assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(indices.iter().all(|&index| index * B + B <= prefix));
        }
    }

    #[cfg(feature = "qwen35-session-continuation")]
    fn phase_fixture(index: usize) -> Value {
        json!({"schema":"qwen35-canonical-target-layer-profile-v2","tokens":B,
            "instrumented_only":true,"accounting_valid":true,"cursor":index*B,
            "target_head_total_ms":0.01,"gather_features_and_between_stage_ms":0.2})
    }

    #[cfg(feature = "qwen35-session-continuation")]
    #[test]
    fn prefix_report_never_relabels_terminal_markers_as_an_output_head() {
        let sample = prefix_sample(phase_fixture(0), 0).unwrap();
        assert!(sample.get("target_head_total_ms").is_none());
        assert!(sample.get("gather_features_and_between_stage_ms").is_none());
        assert_eq!(sample["terminal_marker_interval_ms"], 0.01);
        assert_eq!(sample["output_head_executed"], false);
        assert_eq!(sample["feature_capture"], false);
        assert_eq!(sample["publication_enabled"], false);
        for (key, value) in [
            ("schema", json!("legacy")),
            ("tokens", json!(128)),
            ("accounting_valid", json!(false)),
            ("target_head_total_ms", json!(-1.0)),
        ] {
            let mut report = phase_fixture(0);
            report[key] = value;
            assert!(prefix_sample(report, 0).is_err());
        }
    }

    #[cfg(feature = "qwen35-session-continuation")]
    #[test]
    fn prefix_summary_requires_each_expected_completed_batch_once() {
        let indices = prefix_sampling(Some(std::ffi::OsStr::new("1")), B, 7037)
            .unwrap()
            .unwrap();
        let samples = indices
            .iter()
            .map(|&index| prefix_sample(phase_fixture(index), index).unwrap())
            .collect::<Vec<_>>();
        let summary = prefix_summary(7037, samples.clone()).unwrap();
        assert_eq!(summary["sampled_real_tokens"], 24);
        assert_eq!(summary["unprofiled_prefix_tokens"], 7013);
        assert_eq!(summary["performance_claim_permitted"], false);
        assert!(prefix_summary(7037, samples[..2].to_vec()).is_err());
        let mut reversed = samples.clone();
        reversed.reverse();
        assert!(prefix_summary(7037, reversed).is_err());
        for (key, value) in [
            ("cursor", json!(0)),
            ("tokens", json!(16)),
            ("output_head_executed", json!(true)),
            ("publication_enabled", json!(true)),
        ] {
            let mut wrong = samples.clone();
            wrong[1][key] = value;
            assert!(prefix_summary(7037, wrong).is_err());
        }
    }
}
