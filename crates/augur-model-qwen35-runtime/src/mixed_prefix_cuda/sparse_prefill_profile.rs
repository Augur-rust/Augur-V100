//! Default-off, first/middle/last-chunk attribution. No operator approximation,
//! extra replay or hidden/state readback. Instrumented time is not throughput.
use super::*;
use serde_json::{json, Value};
use std::cell::{Cell, RefCell};

fn sampled_indices(count: usize) -> Result<Vec<usize>, Qwen35MixedPrefixCudaError> {
    if count == 0 {
        return Err(invalid("sparse profile empty schedule"));
    }
    let mut result = vec![0, count / 2, count - 1];
    result.sort_unstable();
    result.dedup();
    Ok(result)
}

fn partition_valid(parts: &[f64], total: f64) -> bool {
    let sum: f64 = parts.iter().sum();
    total.is_finite()
        && total > 0.0
        && !parts.is_empty()
        && parts.iter().all(|v| v.is_finite() && *v >= 0.0)
        && sum.is_finite()
        && sum <= total + (total * 0.0001).max(0.1)
}

#[allow(clippy::too_many_arguments)]
fn submission_valid(
    owner: usize,
    observed: usize,
    indices: &[usize],
    completed: usize,
    pending: Option<usize>,
    submitted: usize,
    index: usize,
    layer: usize,
) -> bool {
    owner == observed
        && layer < 64
        && indices.get(completed) == Some(&index)
        && if layer == 0 {
            pending.is_none() && submitted == 0
        } else {
            pending == Some(index) && submitted == layer
        }
}

fn completion_valid(
    owner: usize,
    observed: usize,
    pending: Option<usize>,
    submitted: usize,
    index: usize,
    proof: bool,
) -> bool {
    proof && owner == observed && pending == Some(index) && submitted == 64
}

pub(super) struct SparsePrefillProfile {
    owner: usize,
    geometry: Vec<(usize, usize)>,
    chunk_tokens: usize,
    indices: Vec<usize>,
    events: Vec<Vec<OwnedCudaEvent>>,
    pending: Cell<Option<usize>>,
    submitted: Cell<usize>,
    rows: RefCell<Vec<Value>>,
}

impl SparsePrefillProfile {
    pub(super) fn new(
        stream: &OwnedCudaStream,
        schedule: &PrefillChunkSchedule,
    ) -> Result<Self, Qwen35MixedPrefixCudaError> {
        if false {
            return Err(invalid("sparse profile excludes fused SiLU path"));
        }
        let indices = sampled_indices(schedule.chunks.len())?;
        let mut position: usize = 0;
        let mut geometry = Vec::with_capacity(schedule.chunks.len());
        for chunk in &schedule.chunks {
            if chunk.tokens == 0
                || chunk.tokens > schedule.chunk_tokens
                || chunk.operations.len() != 64
            {
                return Err(invalid("sparse profile chunk geometry"));
            }
            for (layer, op) in chunk.operations.iter().enumerate() {
                let valid = match op {
                    PrefillLayerOps::Linear(p) => {
                        layer % 4 != 3
                            && !p.simt_shared_gdn_scalars
                            && p.canonical_gdn_scalars.is_none()
                    }
                    PrefillLayerOps::Full(_) => layer % 4 == 3,
                    
                };
                if !valid {
                    return Err(invalid("sparse profile operator policy mismatch"));
                }
            }
            geometry.push((position, chunk.tokens));
            position = position
                .checked_add(chunk.tokens)
                .ok_or_else(|| invalid("sparse profile position overflow"))?;
        }
        let events = (0..64)
            .map(|_| {
                (0..5)
                    .map(|_| OwnedCudaEvent::create(stream.device_ordinal()))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            owner: stream.owner_instance_key()?,
            geometry,
            chunk_tokens: schedule.chunk_tokens,
            indices,
            events,
            pending: Cell::new(None),
            submitted: Cell::new(0),
            rows: RefCell::new(Vec::new()),
        })
    }

    pub(super) fn selected(&self, index: usize) -> bool {
        self.indices.contains(&index)
    }

    pub(super) fn submit(
        &self,
        index: usize,
        layer: usize,
        op: &PrefillLayerOps,
        stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if !submission_valid(
            self.owner,
            stream.owner_instance_key()?,
            &self.indices,
            self.rows.borrow().len(),
            self.pending.get(),
            self.submitted.get(),
            index,
            layer,
        ) {
            return Err(invalid("sparse profile owner/order/premature event reuse"));
        }
        if layer == 0 {
            self.pending.set(Some(index));
            self.submitted.set(0);
        }
        prefill_stage_profile::submit_layer(op, stream, cublas, &self.events[layer], None)?;
        self.submitted.set(layer + 1);
        Ok(())
    }

    pub(super) fn finish_chunk(
        &self,
        index: usize,
        stream: &OwnedCudaStream,
        enclosing_ms: f64,
        proof: bool,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if !completion_valid(
            self.owner,
            stream.owner_instance_key()?,
            self.pending.get(),
            self.submitted.get(),
            index,
            proof,
        ) {
            return Err(invalid("sparse profile completion/owner"));
        }
        let (position, tokens) = self.geometry[index];
        let mut layers = Vec::with_capacity(64);
        let mut totals = [0.0; 5];
        for (layer, e) in self.events.iter().enumerate() {
            let parts = [
                f64::from(e[0].elapsed_ms(&e[1])?),
                f64::from(e[1].elapsed_ms(&e[2])?),
                f64::from(e[2].elapsed_ms(&e[3])?),
                f64::from(e[3].elapsed_ms(&e[4])?),
            ];
            if parts.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(invalid("sparse profile nonfinite interval"));
            }
            totals[0] += parts[0];
            totals[if layer % 4 == 3 { 2 } else { 1 }] += parts[1];
            totals[3] += parts[2];
            totals[4] += parts[3];
            layers.push(json!({"layer":layer,"kind":if layer%4==3 {"full_attention"}else{"linear_attention"},
                "input_projection_ms":parts[0],"attention_or_recurrence_ms":parts[1],
                "output_projection_ms":parts[2],"ffn_ms":parts[3]}));
        }
        if !partition_valid(&totals, enclosing_ms) {
            return Err(invalid("sparse profile accounting"));
        }
        let accounted: f64 = totals.iter().sum();
        self.rows.borrow_mut().push(json!({"chunk_index":index,"position_base":position,"tokens":tokens,
            "owner_epoch_proof":proof,"enclosing_cuda_ms":enclosing_ms,"input_projection_ms":totals[0],
            "linear_recurrence_ms":totals[1],"full_attention_ms":totals[2],"output_projection_ms":totals[3],
            "ffn_ms":totals[4],"accounted_stage_ms":accounted,"between_stage_gap_ms":enclosing_ms-accounted,"layers":layers}));
        self.pending.set(None);
        self.submitted.set(0);
        Ok(())
    }

    pub(super) fn report(&self) -> Result<Value, Qwen35MixedPrefixCudaError> {
        let rows = self.rows.borrow();
        if self.pending.get().is_some() || rows.len() != self.indices.len() {
            return Err(invalid("sparse profile incomplete"));
        }
        let (position, tokens) = self
            .geometry
            .last()
            .ok_or_else(|| invalid("sparse profile missing geometry"))?;
        Ok(
            json!({"schema":"qwen35-request-sparse-prefill-profile-v1","instrumented_only":true,
            "scope":"first_middle_last_chunks_not_full_prefill_attribution","prefix_tokens":position+tokens,
            "chunk_tokens":self.chunk_tokens,"chunk_count":self.geometry.len(),"sampled_chunk_indices":self.indices,
            "event_owner_count":64*5,"event_records":rows.len()*64*5,"layer_records":rows.len()*64,
            "operator_policy":"retained_simt_original_stage_sequence_v1","adds_tensor_readbacks":false,
            "extrapolated_full_prefill_ms":Value::Null,"numerical_parity_qualified":false,
            "performance_claim_permitted":false,"product_qualification":false,"chunks":*rows}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_event_reuse_requires_exact_owner_order_and_prior_completion() {
        let indices = [0, 126, 251];
        assert!(submission_valid(7, 7, &indices, 0, None, 0, 0, 0));
        assert!(submission_valid(7, 7, &indices, 0, Some(0), 1, 0, 1));
        assert!(submission_valid(7, 7, &indices, 1, None, 0, 126, 0));
        for (observed, completed, pending, submitted, index, layer) in [
            (8, 0, None, 0, 0, 0),
            (7, 0, Some(0), 64, 126, 0),
            (7, 1, None, 0, 0, 0),
            (7, 0, Some(0), 1, 0, 2),
            (7, 0, None, 0, 0, 1),
            (7, 0, Some(0), 1, 0, 0),
            (7, 3, None, 0, 251, 0),
            (7, 1, None, 0, 251, 0),
            (7, 0, Some(0), 64, 0, 64),
        ] {
            assert!(!submission_valid(
                7, observed, &indices, completed, pending, submitted, index, layer
            ));
        }
    }
    #[test]
    fn sparse_elapsed_reads_require_terminal_proof_and_all_layers() {
        assert!(completion_valid(7, 7, Some(126), 64, 126, true));
        assert!(!completion_valid(7, 8, Some(126), 64, 126, true));
        assert!(!completion_valid(7, 7, Some(126), 63, 126, true));
        assert!(!completion_valid(7, 7, Some(0), 64, 126, true));
        assert!(!completion_valid(7, 7, None, 64, 126, true));
        assert!(!completion_valid(7, 7, Some(126), 64, 126, false));
    }
    #[test]
    fn sparse_chunk_set_is_ordered_unique_and_bounded() {
        assert!(sampled_indices(0).is_err());
        for count in 1..=512 {
            let indices = sampled_indices(count).unwrap();
            assert_eq!(indices[0], 0);
            assert_eq!(*indices.last().unwrap(), count - 1);
            assert!(indices.contains(&(count / 2)) && indices.len() <= 3);
            assert!(indices.windows(2).all(|p| p[0] < p[1]));
        }
    }
    #[test]
    fn sparse_accounting_rejects_missing_nonfinite_negative_and_overrun() {
        assert!(partition_valid(&[1., 2., 3., 4., 5.], 15.));
        assert!(partition_valid(&[1., 2., 3., 4., 5.], 16.));
        assert!(partition_valid(&[1., 2., 3., 4., 5.], 14.99999));
        assert!(!partition_valid(&[], 1.));
        assert!(!partition_valid(&[1.], 0.));
        assert!(!partition_valid(&[-1., 1.], 1.));
        assert!(!partition_valid(&[2.], 1.));
        assert!(!partition_valid(&[f64::NAN], 1.));
        assert!(!partition_valid(&[1.], f64::INFINITY));
    }
}
