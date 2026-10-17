//! Fixed early-layer, per-row operator receipts; no publication authority.
use super::*;
const LAYERS: [usize; 2] = [1, 2];
const WIDTHS: [usize; 8] = [5120, 10240, 6144, 48, 48, 10240, 6144, 5120];
const NAMES: [&str; 8] = [
    "normalized",
    "qkv",
    "z",
    "alpha_raw",
    "beta_raw",
    "activated",
    "recurrent",
    "attention_projection",
];
const ROW_ELEMENTS: usize = 43104;
const TAPE_BYTES: usize = 2 * B * ROW_ELEMENTS * 4;
pub(super) const SUMMARY_BYTES: usize = 2 * 8 * B * 64;
pub(super) const DEVICE_BYTES: usize = 2 * TAPE_BYTES + SUMMARY_BYTES;
pub(super) struct LinearStages {
    batch: OwnedCudaBuffer,
    serial: OwnedCudaBuffer,
    summaries: OwnedCudaBuffer,
}
fn offset(slot: usize, stage: usize, row: usize) -> usize {
    (slot * B * ROW_ELEMENTS + B * WIDTHS[..stage].iter().sum::<usize>() + row * WIDTHS[stage]) * 4
}
impl LinearStages {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            batch: OwnedCudaBuffer::allocate(0, TAPE_BYTES)?,
            serial: OwnedCudaBuffer::allocate(0, TAPE_BYTES)?,
            summaries: OwnedCudaBuffer::allocate(0, SUMMARY_BYTES)?,
        })
    }
    pub(super) fn capture_batch(
        &self,
        d: &Driver,
        layer: usize,
        w: &PrefillWorkspace,
    ) -> Result<()> {
        let Some(slot) = LAYERS.iter().position(|x| *x == layer) else {
            return Ok(());
        };
        for (stage, source) in [
            &w.normalized,
            &w.qkv,
            &w.z,
            &w.alpha_raw,
            &w.beta_raw,
            &w.activated,
            &w.recurrent,
            &w.attention_output,
        ]
        .into_iter()
        .enumerate()
        {
            copy(
                &d.stream,
                &self.batch,
                offset(slot, stage, 0),
                source.view(0, B * WIDTHS[stage] * 4)?,
            )?;
        }
        Ok(())
    }
    pub(super) fn capture_serial(&self, d: &Driver, layer: usize, row: usize) -> Result<()> {
        let Some(slot) = LAYERS.iter().position(|x| *x == layer) else {
            return Ok(());
        };
        let DevicePlan::Linear(p) = &d.plans[layer] else {
            return Err(invalid("linear stage geometry"));
        };
        let projection = p
            .projection
            .as_ref()
            .ok_or_else(|| invalid("linear stage projection"))?;
        for (stage, source) in [
            &projection.normalized,
            &p.qkv,
            &p.z,
            &projection.alpha_raw,
            &projection.beta_raw,
            &p.activated,
            &p.recurrent,
            &p.attention,
        ]
        .into_iter()
        .enumerate()
        {
            copy(
                &d.stream,
                &self.serial,
                offset(slot, stage, row),
                source.view(0, WIDTHS[stage] * 4)?,
            )?;
        }
        Ok(())
    }
    pub(super) fn finish(&self, d: &Driver) -> Result<Value> {
        for slot in 0..2 {
            for stage in 0..8 {
                for row in 0..B {
                    let index = (slot * 8 + stage) * B + row;
                    submit_qwen35_compare_summary(
                        &d.stream,
                        self.batch
                            .view(offset(slot, stage, row), WIDTHS[stage] * 4)?,
                        self.serial
                            .view(offset(slot, stage, row), WIDTHS[stage] * 4)?,
                        self.summaries.view(index * 64, 64)?,
                        0,
                    )?;
                }
            }
        }
        let mut bytes = vec![0u8; SUMMARY_BYTES];
        let receipt = d.stream.submission_receipt()?;
        let proof = self
            .summaries
            .copy_range_to_host_sync_with_submission_proof(&d.stream, 0, &mut bytes, receipt)?;
        if !proof.covers_submission(receipt) || !proof.belongs_to_stream(&d.stream)? {
            return Err(invalid("linear stage summary owner/epoch"));
        }
        let mut rows = Vec::with_capacity(128);
        for (slot, layer) in LAYERS.into_iter().enumerate() {
            for stage in 0..8 {
                for row in 0..B {
                    let index = (slot * 8 + stage) * B + row;
                    let mut metric =
                        decode_summary(&bytes[index * 64..(index + 1) * 64], WIDTHS[stage])?;
                    metric["layer"] = json!(layer);
                    metric["stage"] = json!(NAMES[stage]);
                    metric["row"] = json!(row);
                    rows.push(metric);
                }
            }
        }
        Ok(
            json!({"schema":"qwen35-shadow-linear-stage-audit-v2","layers":LAYERS,"rows":rows,
            "d2h_bytes":SUMMARY_BYTES,"device_bytes":DEVICE_BYTES,"diagnostic_only":true,"replacement_permitted":false,
            "scope":"layers_1_2_per_row_operator_outputs_not_causal_attribution"}),
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stage_tape_ranges_partition_without_overlap() {
        assert_eq!(WIDTHS.iter().sum::<usize>(), ROW_ELEMENTS);
        let mut end = 0;
        for slot in 0..2 {
            for stage in 0..8 {
                for row in 0..B {
                    assert_eq!(offset(slot, stage, row), end);
                    end += WIDTHS[stage] * 4;
                }
            }
        }
        assert_eq!(end, TAPE_BYTES);
        assert_eq!(DEVICE_BYTES, 5525504);
        assert_eq!(SUMMARY_BYTES, 8192);
    }
}
