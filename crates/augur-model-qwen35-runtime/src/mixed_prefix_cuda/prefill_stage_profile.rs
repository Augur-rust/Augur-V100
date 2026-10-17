//! Separate, output-checked SIMT attribution replay; never a throughput sample.
use super::*;

#[derive(Clone, Debug, Default, Serialize)]
pub struct PrefillProjectionStageTiming {
    pub weight_decode_ms: f64,
    pub input_decode_ms: f64,
    pub gemm_ms: f64,
}

impl PrefillProjectionStageTiming {
    fn add(&mut self, other: &Self) {
        self.weight_decode_ms += other.weight_decode_ms;
        self.input_decode_ms += other.input_decode_ms;
        self.gemm_ms += other.gemm_ms;
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PrefillFfnStageTiming {
    pub input_quantize_ms: f64,
    pub gate: PrefillProjectionStageTiming,
    pub up: PrefillProjectionStageTiming,
    pub silu_and_down_quantize_ms: f64,
    pub down: PrefillProjectionStageTiming,
    pub residual_ms: f64,
}

impl PrefillFfnStageTiming {
    fn parts(&self) -> [f64; 12] {
        [
            self.input_quantize_ms,
            self.gate.weight_decode_ms,
            self.gate.input_decode_ms,
            self.gate.gemm_ms,
            self.up.weight_decode_ms,
            self.up.input_decode_ms,
            self.up.gemm_ms,
            self.silu_and_down_quantize_ms,
            self.down.weight_decode_ms,
            self.down.input_decode_ms,
            self.down.gemm_ms,
            self.residual_ms,
        ]
    }

    fn add(&mut self, other: &Self) {
        self.input_quantize_ms += other.input_quantize_ms;
        self.gate.add(&other.gate);
        self.up.add(&other.up);
        self.silu_and_down_quantize_ms += other.silu_and_down_quantize_ms;
        self.down.add(&other.down);
        self.residual_ms += other.residual_ms;
    }
}

pub(super) fn read_ffn_stages(
    start: &OwnedCudaEvent,
    events: &[OwnedCudaEvent],
    stop: &OwnedCudaEvent,
) -> Result<PrefillFfnStageTiming, Qwen35MixedPrefixCudaError> {
    if events.len() != 11 {
        return Err(invalid("FFN profile event extent"));
    }
    let projection = |offset: usize| -> Result<PrefillProjectionStageTiming, CudaRuntimeError> {
        Ok(PrefillProjectionStageTiming {
            weight_decode_ms: f64::from(events[offset].elapsed_ms(&events[offset + 1])?),
            input_decode_ms: f64::from(events[offset + 1].elapsed_ms(&events[offset + 2])?),
            gemm_ms: f64::from(events[offset + 2].elapsed_ms(&events[offset + 3])?),
        })
    };
    Ok(PrefillFfnStageTiming {
        input_quantize_ms: f64::from(start.elapsed_ms(&events[0])?),
        gate: projection(0)?,
        up: projection(3)?,
        silu_and_down_quantize_ms: f64::from(events[6].elapsed_ms(&events[7])?),
        down: projection(7)?,
        residual_ms: f64::from(events[10].elapsed_ms(stop)?),
    })
}

#[derive(Clone, Debug, Serialize)]
pub struct PrefillLayerStageTiming {
    pub chunk_index: usize,
    pub position_base: usize,
    pub tokens: usize,
    pub layer: usize,
    pub kind: &'static str,
    pub input_projection_ms: f64,
    pub attention_or_recurrence_ms: f64,
    pub output_projection_ms: f64,
    pub ffn_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ffn_projection_stages: Option<PrefillFfnStageTiming>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PrefillStageProfileReport {
    pub schema: &'static str,
    pub instrumented_only: bool,
    pub included_in_primary_timed_tps: bool,
    pub event_records: usize,
    pub input_projection_ms: f64,
    pub linear_recurrence_ms: f64,
    pub full_attention_ms: f64,
    pub output_projection_ms: f64,
    pub ffn_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ffn_projection_totals: Option<PrefillFfnStageTiming>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ffn_projection_accounting_valid: Option<bool>,
    pub terminal_head_ms: f64,
    pub instrumented_cuda_ms: f64,
    pub accounted_stage_ms: f64,
    pub between_stage_gap_ms: f64,
    pub diagnostic_host_wall_with_reset_and_readback_ms: f64,
    pub accounting_valid: bool,
    pub replay_hidden_bit_exact: bool,
    pub replay_state_bit_exact: bool,
    pub replay_logits_bit_exact: bool,
    pub owner_epoch_proof: bool,
    pub completion_poll_iterations: u64,
    pub allocation_interval: AllocationIntervalReport,
    pub boundaries_including_reset_and_readback: BoundaryReport,
    pub layers: Vec<PrefillLayerStageTiming>,
    pub pass: bool,
}

fn accounting_valid(parts: &[f64], total: f64) -> bool {
    if !total.is_finite()
        || total <= 0.0
        || parts.is_empty()
        || parts.iter().any(|x| !x.is_finite() || *x < 0.0)
    {
        return false;
    }
    // CUDA event interval rounding can move a telescoping sum slightly.
    let sum: f64 = parts.iter().sum();
    sum.is_finite() && sum <= total + (total * 0.0001).max(0.1)
}

pub(super) fn submit_layer(
    op: &PrefillLayerOps,
    stream: &OwnedCudaStream,
    cublas: &CudaQwen35PrefillCublasPlanV1,
    events: &[OwnedCudaEvent],
    fine: Option<&[OwnedCudaEvent]>,
) -> Result<(), Qwen35MixedPrefixCudaError> {
    if events.len() != 5 || fine.is_some_and(|f| f.len() != 11) {
        return Err(invalid("profile event extent"));
    }
    macro_rules! submit_ffn {
        ($p:ident) => {{
            let legacy_ffn = $p.ffn.legacy()?;
            legacy_ffn.ffn_input_quantize.submit_for_plan(stream)?;
            if let Some(f) = fine {
                f[0].record(stream)?;
                cublas.submit_profiled(stream, &legacy_ffn.ffn_gate, [&f[1], &f[2], &f[3]])?;
                cublas.submit_profiled(stream, &legacy_ffn.ffn_up, [&f[4], &f[5], &f[6]])?;
                legacy_ffn.silu_mul.submit_for_plan(stream)?;
                legacy_ffn.ffn_down_quantize.submit_for_plan(stream)?;
                f[7].record(stream)?;
                cublas.submit_profiled(stream, &legacy_ffn.ffn_down, [&f[8], &f[9], &f[10]])?;
            } else {
                cublas.submit(stream, &legacy_ffn.ffn_gate)?;
                cublas.submit(stream, &legacy_ffn.ffn_up)?;
                legacy_ffn.silu_mul.submit_for_plan(stream)?;
                legacy_ffn.ffn_down_quantize.submit_for_plan(stream)?;
                cublas.submit(stream, &legacy_ffn.ffn_down)?;
            }
            $p.final_residual.submit_for_plan(stream)?;
        }};
    }
    events[0].record(stream)?;
    match op {
        PrefillLayerOps::Linear(p) => {
            p.attn_norm.submit(stream)?;
            p.normalized_quantize.submit_for_plan(stream)?;
            cublas.submit(stream, &p.qkv)?;
            cublas.submit(stream, &p.z)?;
            cublas.submit_shared_input_group(stream, &[&p.alpha, &p.beta])?;
            events[1].record(stream)?;
            p.submit_recurrence(stream)?;
            events[2].record(stream)?;
            p.recurrent_quantize.submit_for_plan(stream)?;
            cublas.submit(stream, &p.output)?;
            p.attention_residual.submit_for_plan(stream)?;
            p.post_norm.submit(stream)?;
            events[3].record(stream)?;
            submit_ffn!(p);
        }
        PrefillLayerOps::Full(p) => {
            p.submit_projection_prefix(stream, cublas)?;
            events[1].record(stream)?;
            if p.simt_shared_softmax {
                p.submit_shared_simt_attention(stream, cublas)?;
            } else {
                p.attention.submit(stream)?;
            }
            events[2].record(stream)?;
            p.gated_quantize.submit_for_plan(stream)?;
            cublas.submit(stream, &p.output)?;
            p.attention_residual.submit_for_plan(stream)?;
            p.post_norm.submit(stream)?;
            events[3].record(stream)?;
            submit_ffn!(p);
        }
        
    }
    events[4].record(stream)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    plans: &mut [DevicePlan],
    workspace: &PrefillWorkspace,
    schedule: &PrefillChunkSchedule,
    stream: &OwnedCudaStream,
    terminal_hidden: &OwnedCudaBuffer,
    output_head: &DeviceOutputHeadPlan,
    input_hidden: &[f32],
    expected: &PrefillChunkRun,
    profile_ffn_projections: bool,
) -> Result<PrefillStageProfileReport, Qwen35MixedPrefixCudaError> {
    let expected_hidden = expected
        .final_hidden
        .as_ref()
        .ok_or_else(|| invalid("profile requires hidden oracle"))?;
    let expected_states = expected
        .states
        .as_ref()
        .ok_or_else(|| invalid("profile requires state oracle"))?;
    let expected_logits = expected
        .logits
        .as_ref()
        .ok_or_else(|| invalid("profile requires logits oracle"))?;
    if plans.len() != FULL_TARGET_LAYER_COUNT
        || schedule.chunks.is_empty()
        || expected_hidden.len() != input_hidden.len()
        || schedule.chunks.iter().any(|c| {
            c.tokens == 0
                || c.tokens > workspace.max_tokens
                || c.operations.len() != FULL_TARGET_LAYER_COUNT
                || c.operations.iter().enumerate().any(|(i, op)| {
                    if i % 4 == 3 {
                        !matches!(op, PrefillLayerOps::Full(_))
                    } else {
                        !matches!(op, PrefillLayerOps::Linear(_))
                    }
                })
        })
    {
        return Err(invalid("profile stage geometry"));
    }
    let total_rows = schedule
        .chunks
        .iter()
        .try_fold(0usize, |sum, c| sum.checked_add(c.tokens))
        .ok_or_else(|| invalid("profile row overflow"))?;
    if total_rows > 4096 || total_rows.checked_mul(linear::HIDDEN) != Some(input_hidden.len()) {
        return Err(invalid("profile input extent"));
    }
    // Allocate once before the pass. Reuse only after the chunk terminal proof
    // and elapsed-time reads have consumed the previous records.
    let events = (0..FULL_TARGET_LAYER_COUNT)
        .map(|_| {
            (0..5)
                .map(|_| OwnedCudaEvent::create(stream.device_ordinal()))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ffn_events = if profile_ffn_projections {
        Some(
            (0..FULL_TARGET_LAYER_COUNT)
                .map(|_| {
                    (0..11)
                        .map(|_| OwnedCudaEvent::create(stream.device_ordinal()))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };
    let head_start = OwnedCudaEvent::create(stream.device_ordinal())?;
    let head_stop = OwnedCudaEvent::create(stream.device_ordinal())?;
    let terminal = OwnedCudaEvent::create(stream.device_ordinal())?;
    let allocation_before = cuda_device_allocation_snapshot();
    let boundary_before = cuda_host_boundary_snapshot();
    let started = Instant::now();
    reset_plans(plans, stream)?;
    let mut layers = Vec::with_capacity(schedule.chunks.len() * FULL_TARGET_LAYER_COUNT);
    let mut position_base = 0;
    let mut instrumented_cuda_ms = 0.0;
    let mut terminal_head_ms = 0.0;
    let mut replay_hidden_bit_exact = true;
    let mut owner_epoch_proof = true;
    let mut completion_poll_iterations = 0u64;
    let mut last_receipt = None;
    let row_bytes = linear::HIDDEN * core::mem::size_of::<f32>();
    for (chunk_index, chunk) in schedule.chunks.iter().enumerate() {
        let begin = position_base * linear::HIDDEN;
        let end = begin + chunk.tokens * linear::HIDDEN;
        workspace
            .hidden_a
            .copy_from_host_sync(stream, &f32_bytes(&input_hidden[begin..end]))?;
        for (layer, (op, e)) in chunk.operations.iter().zip(&events).enumerate() {
            submit_layer(
                op,
                stream,
                &workspace.cublas,
                e,
                ffn_events.as_ref().map(|f| f[layer].as_slice()),
            )?;
        }
        let is_last = chunk_index + 1 == schedule.chunks.len();
        if is_last {
            head_start.record(stream)?;
            let source = workspace
                .hidden_a
                .view((chunk.tokens - 1) * row_bytes, row_bytes)?;
            // SAFETY: identical disjoint live owners and same-stream head copy
            // as the normal runner; the terminal event covers every consumer.
            unsafe {
                terminal_hidden.copy_from_device_async_unchecked(stream, 0, source)?;
            }
            output_head.submit(stream)?;
            head_stop.record(stream)?;
        }
        let receipt = stream.submission_receipt()?;
        terminal.record_for_submission(stream, receipt)?;
        let mut polls = 0u64;
        let completion = loop {
            if let Some(p) = stream.query_event_completion(&terminal, receipt)? {
                break p;
            }
            polls += 1;
            if polls >= MAX_CHUNKED_PREFILL_COMPLETION_POLLS {
                return Err(Qwen35MixedPrefixCudaError::CompletionPollLimit);
            }
            super::bounded_completion_poll_backoff(polls);
        };
        owner_epoch_proof &=
            completion.covers_submission(receipt) && completion.belongs_to_stream(stream)?;
        completion_poll_iterations = completion_poll_iterations.saturating_add(polls);
        let stop = if is_last {
            &head_stop
        } else {
            &events[FULL_TARGET_LAYER_COUNT - 1][4]
        };
        instrumented_cuda_ms += f64::from(events[0][0].elapsed_ms(stop)?);
        if is_last {
            terminal_head_ms = f64::from(head_start.elapsed_ms(&head_stop)?);
        }
        for (layer, e) in events.iter().enumerate() {
            layers.push(PrefillLayerStageTiming {
                chunk_index,
                position_base,
                tokens: chunk.tokens,
                layer,
                kind: if layer % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                },
                input_projection_ms: f64::from(e[0].elapsed_ms(&e[1])?),
                attention_or_recurrence_ms: f64::from(e[1].elapsed_ms(&e[2])?),
                output_projection_ms: f64::from(e[2].elapsed_ms(&e[3])?),
                ffn_ms: f64::from(e[3].elapsed_ms(&e[4])?),
                ffn_projection_stages: ffn_events
                    .as_ref()
                    .map(|f| read_ffn_stages(&e[3], &f[layer], &e[4]))
                    .transpose()?,
            });
        }
        let hidden = read_f32_view(
            stream,
            workspace.hidden_a.view(0, chunk.tokens * row_bytes)?,
            receipt,
        )?;
        replay_hidden_bit_exact &= equal_f32_bits(&hidden, &expected_hidden[begin..end]);
        position_base += chunk.tokens;
        last_receipt = Some(receipt);
    }
    let receipt = last_receipt.ok_or_else(|| invalid("profile missing terminal"))?;
    let states = read_prefill_backbone_states(plans, stream, receipt)?;
    let replay_state_bit_exact = states.len() == expected_states.len()
        && states
            .iter()
            .zip(expected_states)
            .all(|(a, b)| a.0 == b.0 && equal_f32_bits(&a.1, &b.1));
    let logits = read_f32_buffer(stream, &output_head.logits, receipt)?;
    let replay_logits_bit_exact = equal_f32_bits(&logits, expected_logits);
    let diagnostic_host_wall_with_reset_and_readback_ms = started.elapsed().as_secs_f64() * 1000.0;
    let boundaries_including_reset_and_readback =
        boundary_delta(boundary_before, cuda_host_boundary_snapshot())?;
    let allocation_interval =
        allocation_interval_report(allocation_before, cuda_device_allocation_snapshot())?;
    let input_projection_ms = layers.iter().map(|r| r.input_projection_ms).sum::<f64>();
    let linear_recurrence_ms = layers
        .iter()
        .filter(|r| r.layer % 4 != 3)
        .map(|r| r.attention_or_recurrence_ms)
        .sum::<f64>();
    let full_attention_ms = layers
        .iter()
        .filter(|r| r.layer % 4 == 3)
        .map(|r| r.attention_or_recurrence_ms)
        .sum::<f64>();
    let output_projection_ms = layers.iter().map(|r| r.output_projection_ms).sum::<f64>();
    let ffn_ms = layers.iter().map(|r| r.ffn_ms).sum::<f64>();
    let ffn_projection_totals = profile_ffn_projections.then(|| {
        let mut sum = PrefillFfnStageTiming::default();
        for layer in &layers {
            if let Some(value) = &layer.ffn_projection_stages {
                sum.add(value);
            }
        }
        sum
    });
    let ffn_projection_accounting_valid = ffn_projection_totals.as_ref().map(|total| {
        layers.iter().all(|layer| {
            layer
                .ffn_projection_stages
                .as_ref()
                .is_some_and(|s| accounting_valid(&s.parts(), layer.ffn_ms))
        }) && accounting_valid(&total.parts(), ffn_ms)
    });
    let parts = layers
        .iter()
        .flat_map(|r| {
            [
                r.input_projection_ms,
                r.attention_or_recurrence_ms,
                r.output_projection_ms,
                r.ffn_ms,
            ]
        })
        .chain(std::iter::once(terminal_head_ms))
        .collect::<Vec<_>>();
    let accounted_stage_ms = parts.iter().sum::<f64>();
    let accounting_valid = accounting_valid(&parts, instrumented_cuda_ms)
        && layers.len() == FULL_TARGET_LAYER_COUNT * schedule.chunks.len();
    let pass = accounting_valid
        && replay_hidden_bit_exact
        && replay_state_bit_exact
        && replay_logits_bit_exact
        && ffn_projection_accounting_valid != Some(false)
        && owner_epoch_proof
        && allocation_interval.pass;
    Ok(PrefillStageProfileReport {
        schema: if profile_ffn_projections {
            "CQ-QWEN35-PREFILL-SIMT-FFN-STAGE-PROFILE-V2"
        } else {
            "CQ-QWEN35-PREFILL-SIMT-STAGE-PROFILE-V1"
        },
        instrumented_only: true,
        included_in_primary_timed_tps: false,
        event_records: schedule.chunks.len()
            * (FULL_TARGET_LAYER_COUNT * if profile_ffn_projections { 16 } else { 5 } + 1)
            + 2,
        input_projection_ms,
        linear_recurrence_ms,
        full_attention_ms,
        output_projection_ms,
        ffn_ms,
        ffn_projection_totals,
        ffn_projection_accounting_valid,
        terminal_head_ms,
        instrumented_cuda_ms,
        accounted_stage_ms,
        between_stage_gap_ms: (instrumented_cuda_ms - accounted_stage_ms).max(0.0),
        diagnostic_host_wall_with_reset_and_readback_ms,
        accounting_valid,
        replay_hidden_bit_exact,
        replay_state_bit_exact,
        replay_logits_bit_exact,
        owner_epoch_proof,
        completion_poll_iterations,
        allocation_interval,
        boundaries_including_reset_and_readback,
        layers,
        pass,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ffn_substage_accounting_has_twelve_nonoverlapping_intervals() {
        let unit = PrefillFfnStageTiming {
            input_quantize_ms: 1.0,
            gate: PrefillProjectionStageTiming {
                weight_decode_ms: 2.0,
                input_decode_ms: 3.0,
                gemm_ms: 4.0,
            },
            up: PrefillProjectionStageTiming {
                weight_decode_ms: 5.0,
                input_decode_ms: 6.0,
                gemm_ms: 7.0,
            },
            silu_and_down_quantize_ms: 8.0,
            down: PrefillProjectionStageTiming {
                weight_decode_ms: 9.0,
                input_decode_ms: 10.0,
                gemm_ms: 11.0,
            },
            residual_ms: 12.0,
        };
        assert_eq!(
            unit.parts(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
        assert!(accounting_valid(&unit.parts(), 78.0));
        assert!(!accounting_valid(&unit.parts(), 77.0));
        let mut sum = PrefillFfnStageTiming::default();
        sum.add(&unit);
        sum.add(&unit);
        assert_eq!(sum.parts(), unit.parts().map(|x| 2.0 * x));
        sum.down.gemm_ms = f64::NAN;
        assert!(!accounting_valid(&sum.parts(), 156.0));
    }

    #[test]
    fn accounting_accepts_gaps_but_not_nonfinite_negative_or_overcount() {
        assert!(accounting_valid(&[1.0, 2.0, 3.0, 4.0], 10.5));
        assert!(!accounting_valid(&[1.0, 11.0], 10.0));
        assert!(!accounting_valid(&[1.0, -0.1], 10.0));
        assert!(!accounting_valid(&[f64::NAN], 10.0));
        assert!(!accounting_valid(&[f64::MAX, f64::MAX], 10.0));
        assert!(!accounting_valid(&[1.0], f64::INFINITY));
        assert!(!accounting_valid(&[], 10.0));
        assert!(!accounting_valid(&[0.0], 0.0));
    }
}
