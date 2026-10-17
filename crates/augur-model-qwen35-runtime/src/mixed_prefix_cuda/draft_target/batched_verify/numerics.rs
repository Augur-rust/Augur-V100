//! Optional GPU-reduced per-layer error audit. Never influences publication.
use super::*;
use augur_backend::cuda_qwen35_prefill_cublas::submit_qwen35_compare_summary;
mod linear_stages;
use linear_stages::LinearStages;

pub(super) const SUMMARY_BYTES: usize = (64 + 128) * 64;
pub(super) const TOTAL_SUMMARY_BYTES: usize = SUMMARY_BYTES + linear_stages::SUMMARY_BYTES;
const HIDDEN_BYTES: usize = 64 * B * H * 4;

pub(super) struct BatchedNumerics {
    batch_hidden: OwnedCudaBuffer,
    serial_hidden: OwnedCudaBuffer,
    batch_states: Vec<OwnedCudaBuffer>,
    summaries: OwnedCudaBuffer,
    bytes: usize,
    linear_stages: LinearStages,
}

fn decode_summary(bytes: &[u8], expected: usize) -> Result<Value> {
    if bytes.len() != 64 {
        return Err(invalid("numerical summary extent"));
    }
    let u = |i: usize| u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
    let f = |i| f64::from_bits(u(i));
    let elements = u(0);
    let mismatches = u(1);
    let nonfinite = u(2);
    let max_abs = f(3);
    let error = f(4);
    let reference = f(5);
    let candidate = f(6);
    let dot = f(7);
    if elements != expected as u64
        || mismatches > elements
        || nonfinite > elements
        || [max_abs, error, reference, candidate]
            .iter()
            .any(|x| !x.is_finite() || *x < 0.0)
        || !dot.is_finite()
    {
        return Err(invalid("invalid numerical summary"));
    }
    let relative_l2 = (nonfinite == 0 && reference > 0.0).then(|| (error / reference).sqrt());
    let cosine = (nonfinite == 0 && reference > 0.0 && candidate > 0.0)
        .then(|| dot / (reference.sqrt() * candidate.sqrt()));
    Ok(
        json!({"elements":elements,"decoded_bit_mismatches":mismatches,"nonfinite_pairs":nonfinite,
        "max_abs_error":max_abs,"sum_squared_error":error,"sum_squared_reference":reference,
        "sum_squared_candidate":candidate,"dot_product":dot,"relative_l2":relative_l2,"cosine":cosine}),
    )
}

impl BatchedNumerics {
    pub(super) fn new(d: &Driver) -> Result<Self> {
        let batch_states = d
            .backups
            .iter()
            .map(|b| OwnedCudaBuffer::allocate(0, b.len_bytes()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let bytes = 2 * HIDDEN_BYTES
            + SUMMARY_BYTES
            + linear_stages::DEVICE_BYTES
            + batch_states.iter().map(|b| b.len_bytes()).sum::<usize>();
        Ok(Self {
            batch_hidden: OwnedCudaBuffer::allocate(0, HIDDEN_BYTES)?,
            serial_hidden: OwnedCudaBuffer::allocate(0, HIDDEN_BYTES)?,
            summaries: OwnedCudaBuffer::allocate(0, SUMMARY_BYTES)?,
            batch_states,
            bytes,
            linear_stages: LinearStages::new()?,
        })
    }
    pub(super) fn capture_batch_layer(
        &self,
        d: &Driver,
        layer: usize,
        source: CudaBufferView,
        workspace: &PrefillWorkspace,
    ) -> Result<()> {
        copy(&d.stream, &self.batch_hidden, layer * B * H * 4, source)?;
        self.linear_stages.capture_batch(d, layer, workspace)
    }
    pub(super) fn capture_serial_layer(&self, d: &Driver, layer: usize, row: usize) -> Result<()> {
        copy(
            &d.stream,
            &self.serial_hidden,
            (layer * B + row) * H * 4,
            d.plans[layer].final_hidden_view()?,
        )?;
        self.linear_stages.capture_serial(d, layer, row)
    }
    pub(super) fn capture_batch_states(&self, d: &Driver) -> Result<()> {
        for (layer, plan) in d.plans.iter().enumerate() {
            for (plane, state) in plan.terminal_state_buffers().into_iter().enumerate() {
                let (offset, len) = state_range(
                    matches!(plan, DevicePlan::Full(_)),
                    state.len_bytes(),
                    d.cursor,
                    d.capacity,
                )?;
                copy(
                    &d.stream,
                    &self.batch_states[layer * 2 + plane],
                    0,
                    state.view(offset, len)?,
                )?;
            }
        }
        Ok(())
    }
    pub(super) fn finish(&self, d: &Driver) -> Result<Value> {
        let mut state_elements = Vec::with_capacity(128);
        for (layer, plan) in d.plans.iter().enumerate() {
            submit_qwen35_compare_summary(
                &d.stream,
                self.batch_hidden.view(layer * B * H * 4, B * H * 4)?,
                self.serial_hidden.view(layer * B * H * 4, B * H * 4)?,
                self.summaries.view(layer * 64, 64)?,
                0,
            )?;
            for (plane, state) in plan.terminal_state_buffers().into_iter().enumerate() {
                let full = matches!(plan, DevicePlan::Full(_));
                let (offset, len) = state_range(full, state.len_bytes(), d.cursor, d.capacity)?;
                let index = layer * 2 + plane;
                submit_qwen35_compare_summary(
                    &d.stream,
                    view(&self.batch_states[index])?,
                    state.view(offset, len)?,
                    self.summaries.view((64 + index) * 64, 64)?,
                    u32::from(full),
                )?;
                state_elements.push(if full { len / 34 * 32 } else { len / 4 });
            }
        }
        let mut bytes = vec![0u8; SUMMARY_BYTES];
        let receipt = d.stream.submission_receipt()?;
        let proof = self
            .summaries
            .copy_range_to_host_sync_with_submission_proof(&d.stream, 0, &mut bytes, receipt)?;
        if !proof.covers_submission(receipt) || !proof.belongs_to_stream(&d.stream)? {
            return Err(invalid("numeric summary owner/epoch"));
        }
        let mut hidden = Vec::with_capacity(64);
        let mut states = Vec::with_capacity(128);
        for layer in 0..64 {
            let mut h = decode_summary(&bytes[layer * 64..(layer + 1) * 64], B * H)?;
            h["layer"] = json!(layer);
            hidden.push(h);
            for plane in 0..2 {
                let index = layer * 2 + plane;
                let mut state = decode_summary(
                    &bytes[(64 + index) * 64..(65 + index) * 64],
                    state_elements[index],
                )?;
                state["layer"] = json!(layer);
                state["plane"] = json!(plane);
                state["format"] = json!(if layer % 4 == 3 {
                    "decoded_q8_0_kv_tail"
                } else {
                    "f32_recurrent_or_conv"
                });
                states.push(state);
            }
        }
        let first = hidden
            .iter()
            .position(|h| h["decoded_bit_mismatches"].as_u64().unwrap() > 0);
        Ok(
            json!({"schema":"qwen35-shadow-numerical-audit-v3", "cursor":d.cursor,"generation":d.generation,"tokens":B,
            "diagnostic_only":true,"replacement_permitted":false,"performance_claim_permitted":false,
            "state_scope":"full_recurrent_conv_and_only_verified_eight_row_kv_tail", "hidden_scope":"all_64_layer_outputs_all_8_inputs",
            "gpu_reduced":true,"d2h_bytes":TOTAL_SUMMARY_BYTES,"extra_device_bytes":self.bytes,
            "linear_stages":self.linear_stages.finish(d)?,
            "timing_contains_hidden_capture":true,"first_hidden_divergence_layer":first,"hidden":hidden,"states":states}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numerical_summary_checks_extent_counts_and_nonfinite() {
        let values = [
            4u64,
            3,
            1,
            2.0f64.to_bits(),
            5.0f64.to_bits(),
            3.0f64.to_bits(),
            14.0f64.to_bits(),
            6.0f64.to_bits(),
        ];
        let mut bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let s = decode_summary(&bytes, 4).unwrap();
        assert_eq!(s["max_abs_error"], json!(2.0));
        assert!(s["relative_l2"].is_null());
        assert!(decode_summary(&bytes, 5).is_err());
        bytes[8..16].copy_from_slice(&5u64.to_le_bytes());
        assert!(decode_summary(&bytes, 4).is_err());
        assert!(decode_summary(&bytes[..63], 4).is_err());
    }
}
