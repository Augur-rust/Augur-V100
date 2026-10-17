//! Diagnostic CUDA-event partitions of a real shadow verification. No publication.
use super::*;
use crate::mixed_prefix_cuda::prefill_stage_profile::{
    read_ffn_stages, submit_layer, PrefillLayerStageTiming,
};

pub(super) struct BatchedStageProfile {
    layer_events: Vec<Vec<OwnedCudaEvent>>,
    ffn_events: Vec<Vec<OwnedCudaEvent>>,
    pub(super) head_start: OwnedCudaEvent,
}

fn gap(parts: &[f64], total: f64) -> Result<f64> {
    if !total.is_finite()
        || total <= 0.0
        || parts.is_empty()
        || parts.iter().any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(invalid("invalid batch stage event timing"));
    }
    let remainder = total - parts.iter().sum::<f64>();
    if !remainder.is_finite() || remainder < -(total * 0.0001).max(0.1) {
        return Err(invalid("overlapping batch stage event accounting"));
    }
    Ok(remainder.max(0.0))
}

impl BatchedStageProfile {
    pub(super) fn new() -> Result<Self> {
        // Shared stage helper uses the unfused SiLU + Q8 path. Do not accidentally
        // attribute a different execution path in a fused-feature build.
        if false {
            return Err(invalid(
                "batch stage profile does not support fused SiLU feature",
            ));
        }
        let events = |count| -> Result<Vec<Vec<OwnedCudaEvent>>> {
            (0..64)
                .map(|_| {
                    (0..count)
                        .map(|_| OwnedCudaEvent::create(0).map_err(Into::into))
                        .collect()
                })
                .collect()
        };
        Ok(Self {
            layer_events: events(5)?,
            ffn_events: events(11)?,
            head_start: OwnedCudaEvent::create(0)?,
        })
    }

    pub(super) fn submit(
        &self,
        layer: usize,
        operation: &PrefillLayerOps,
        d: &Driver,
        cublas: &CudaQwen35PrefillCublasPlanV1,
    ) -> Result<()> {
        submit_layer(
            operation,
            &d.stream,
            cublas,
            &self.layer_events[layer],
            Some(&self.ffn_events[layer]),
        )
    }

    // Called only after the enclosing stop event has completed via the existing
    // compact top1 readback. No mid-layer host wait/readback is introduced.
    pub(super) fn report(
        &self,
        cursor: usize,
        total_ms: f32,
        stop: &OwnedCudaEvent,
    ) -> Result<Value> {
        let mut layers = Vec::with_capacity(64);
        let mut ffn_weight_decode_ms = 0.0;
        let mut ffn_input_decode_ms = 0.0;
        let mut ffn_gemm_ms = 0.0;
        let mut ffn_other_ms = 0.0;
        for (layer, e) in self.layer_events.iter().enumerate() {
            let fine = read_ffn_stages(&e[3], &self.ffn_events[layer], &e[4])?;
            let weight =
                fine.gate.weight_decode_ms + fine.up.weight_decode_ms + fine.down.weight_decode_ms;
            let input =
                fine.gate.input_decode_ms + fine.up.input_decode_ms + fine.down.input_decode_ms;
            let gemm = fine.gate.gemm_ms + fine.up.gemm_ms + fine.down.gemm_ms;
            let other = fine.input_quantize_ms + fine.silu_and_down_quantize_ms + fine.residual_ms;
            let ffn_ms = f64::from(e[3].elapsed_ms(&e[4])?);
            gap(&[weight, input, gemm, other], ffn_ms)?;
            ffn_weight_decode_ms += weight;
            ffn_input_decode_ms += input;
            ffn_gemm_ms += gemm;
            ffn_other_ms += other;
            layers.push(PrefillLayerStageTiming {
                chunk_index: 0,
                position_base: cursor,
                tokens: B,
                layer,
                kind: if layer % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                },
                input_projection_ms: f64::from(e[0].elapsed_ms(&e[1])?),
                attention_or_recurrence_ms: f64::from(e[1].elapsed_ms(&e[2])?),
                output_projection_ms: f64::from(e[2].elapsed_ms(&e[3])?),
                ffn_ms,
                ffn_projection_stages: Some(fine),
            });
        }
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
        let heads_ms = f64::from(self.head_start.elapsed_ms(stop)?);
        let parts = [
            input_projection_ms,
            linear_recurrence_ms,
            full_attention_ms,
            output_projection_ms,
            ffn_ms,
            heads_ms,
        ];
        let gather_and_between_stage_ms = gap(&parts, total_ms.into())?;
        gap(
            &[
                ffn_weight_decode_ms,
                ffn_input_decode_ms,
                ffn_gemm_ms,
                ffn_other_ms,
            ],
            ffn_ms,
        )?;
        Ok(json!({
            "schema":"qwen35-shadow-batched-stage-profile-v1", "instrumented_only":true,
            "performance_claim_permitted":false, "timing_scope":"cuda_event_intervals_include_submission_gaps_not_kernel_only",
            "event_records":1025, "cursor":cursor, "tokens":B, "layers":layers,
            "input_projection_ms":input_projection_ms, "linear_recurrence_ms":linear_recurrence_ms,
            "full_attention_ms":full_attention_ms, "output_projection_ms":output_projection_ms, "ffn_ms":ffn_ms,
            "ffn_weight_decode_ms":ffn_weight_decode_ms, "ffn_input_decode_ms":ffn_input_decode_ms,
            "ffn_gemm_ms":ffn_gemm_ms, "ffn_other_ms":ffn_other_ms, "heads_ms":heads_ms,
            "gather_and_between_stage_ms":gather_and_between_stage_ms,
            "accounted_stage_ms":parts.iter().sum::<f64>(), "enclosing_event_ms":total_ms,
            "accounting_valid":true,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn event_accounting_rejects_overlap_nonfinite_and_negative_intervals() {
        assert_eq!(gap(&[1.0, 2.0], 4.0).unwrap(), 1.0);
        assert!(gap(&[3.0, 2.0], 4.0).is_err());
        assert!(gap(&[f64::NAN], 4.0).is_err());
        assert!(gap(&[-1.0], 4.0).is_err());
        assert!(gap(&[1.0], f64::INFINITY).is_err());
        assert!(gap(&[], 4.0).is_err());
    }
}
