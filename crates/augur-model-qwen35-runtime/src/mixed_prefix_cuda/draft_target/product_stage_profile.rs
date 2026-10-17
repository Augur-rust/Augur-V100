//! Ten opt-in events on the actual draft and target streams. Read only after
//! the existing acceptance receipt. Stream intervals are never added together.
//! Initial decode setup also has a host-only, non-synchronizing wall partition.
use super::*;
use augur_backend::CudaStreamSubmissionReceipt;
use std::cell::RefCell;

const EXTENTS: [usize; 2] = [3, 7];
const TARGET_KEYS: [&str; 6] = [
    "checkpoint_ms",
    "producer_dependency_wait_ms",
    "draft_head_ms",
    "selector_ms",
    "target_verify_ms",
    "acceptance_ms",
];

// Host wall partition only: never creates CUDA events, copies or synchronizes.
// In particular, a phase containing an async enqueue is NOT device runtime.
const INITIAL_SETUP_KEYS: [&str; 10] = [
    "post_anchor_audit_ms",
    "draft_metadata_ms",
    "noise_embedding_ms",
    "draft_weight_lease_ms",
    "draft_device_owner_ms",
    "draft_policy_ms",
    "auxiliary_audit_ms",
    "target_audit_ms",
    "target_execution_ms",
    "rolling_state_ms",
];

pub(super) struct InitialSetup {
    start: Instant,
    marks: [f64; 10],
    count: usize,
}

impl InitialSetup {
    pub(super) fn new() -> Self {
        Self { start: Instant::now(), marks: [0.0; 10], count: 0 }
    }

    pub(super) fn mark(&mut self, key: &str) -> Result<()> {
        if INITIAL_SETUP_KEYS.get(self.count).copied() != Some(key) {
            return Err(invalid("initial setup host phase order"));
        }
        self.marks[self.count] = self.start.elapsed().as_secs_f64() * 1000.0;
        self.count += 1;
        Ok(())
    }

    pub(super) fn report(&self) -> Result<Value> {
        if self.count != INITIAL_SETUP_KEYS.len() {
            return Err(invalid("initial setup host phase incomplete"));
        }
        let mut previous = 0.0;
        let parts = self.marks.iter().map(|&mark| {
            let delta = mark - previous;
            previous = mark;
            delta
        }).collect::<Vec<_>>();
        valid_times(&parts, previous)?;
        let fields = INITIAL_SETUP_KEYS.iter().zip(parts)
            .map(|(key, value)| ((*key).into(), json!(value)))
            .collect::<serde_json::Map<String, Value>>();
        Ok(json!({
            "schema":"qwen35-initial-decode-host-setup-v1",
            "parts_ms":fields,"enclosing_ms":previous,
            "start_boundary":"initial_anchor_sink_returned",
            "end_boundary":"before_rolling_loop",
            "clock":"host_monotonic_wall","same_clock_disjoint_partition":true,
            "additional_cuda_events":0,"additional_cuda_synchronizations":0,
            "gpu_work_may_overlap_host_phases":true,"device_runtime_measured":false,
            "client_delivery_measured":false,"performance_claim_permitted":false
        }))
    }
}

pub(super) fn requested(
    value: Option<&std::ffi::OsStr>,
    product_batch: bool,
    effective_output_ceiling: usize,
    other_profile_or_graph: bool,
) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(value)
            if value == "1"
                && product_batch
                && (1..=64).contains(&effective_output_ceiling)
                && !other_profile_or_graph => Ok(true),
        _ => Err(invalid(
            "round profile requires exactly1, unprofiled eager product B8 and an effective output ceiling <=64",
        )),
    }
}

#[derive(Clone, Debug)]
struct RoundGuard {
    owners: [usize; 2],
    identity: [u8; 32],
    generation: u64,
    cursor: usize,
    counts: [usize; 2],
}

impl RoundGuard {
    fn check_mark(&self, role: usize, index: usize, owner: usize) -> Result<()> {
        if role >= 2
            || index >= EXTENTS[role]
            || index != self.counts[role]
            || self.owners[role] != owner
        {
            return Err(invalid("round profile event order/owner"));
        }
        Ok(())
    }

    fn check_finish(
        &self,
        owners: [usize; 2],
        identity: [u8; 32],
        generation: u64,
        cursor: usize,
    ) -> Result<()> {
        if self.counts != EXTENTS
            || owners != self.owners
            || identity != self.identity
            || self.generation != generation
            || self.cursor != cursor
        {
            return Err(invalid("round profile incomplete/stale pre-commit binding"));
        }
        Ok(())
    }
}

struct Round {
    guard: RoundGuard,
    receipts: [Option<CudaStreamSubmissionReceipt>; 10],
}

pub(super) struct Profile {
    events: Vec<OwnedCudaEvent>,
    round: RefCell<Option<Round>>,
}

fn valid_times(parts: &[f64], total: f64) -> Result<()> {
    if !total.is_finite()
        || total < 0.0
        || parts.is_empty()
        || parts.iter().any(|x| !x.is_finite() || *x < 0.0)
        || (parts.iter().sum::<f64>() - total).abs() > (total * 0.0001).max(0.1)
    {
        return Err(invalid("round profile event timing partition"));
    }
    Ok(())
}

impl Profile {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            events: (0..10)
                .map(|_| Ok(OwnedCudaEvent::create(0)?))
                .collect::<Result<Vec<_>>>()?,
            round: RefCell::new(None),
        })
    }

    pub(super) fn begin(&self, d: &Driver, draft: &DFlashQ8CudaBlock) -> Result<()> {
        let mut state = self.round.borrow_mut();
        if state.is_some() {
            return Err(invalid("round profile unconsumed previous round"));
        }
        *state = Some(Round {
            guard: RoundGuard {
                owners: [
                    draft.stream().owner_instance_key()?,
                    d.stream.owner_instance_key()?,
                ],
                identity: d.identity,
                generation: d.generation,
                cursor: d.cursor,
                counts: [0; 2],
            },
            receipts: [None; 10],
        });
        Ok(())
    }

    pub(super) fn mark(&self, role: usize, index: usize, stream: &OwnedCudaStream) -> Result<()> {
        let mut state = self.round.borrow_mut();
        let round = state
            .as_mut()
            .ok_or_else(|| invalid("round profile missing round"))?;
        round
            .guard
            .check_mark(role, index, stream.owner_instance_key()?)?;
        let slot = if role == 0 { index } else { 3 + index };
        let receipt = stream.submission_receipt()?;
        self.events[slot].record_for_submission(stream, receipt)?;
        round.receipts[slot] = Some(receipt);
        round.guard.counts[role] += 1;
        Ok(())
    }

    pub(super) fn finish(&self, d: &Driver, draft: &DFlashQ8CudaBlock) -> Result<Value> {
        let mut state = self.round.borrow_mut();
        let round = state
            .as_ref()
            .ok_or_else(|| invalid("round profile missing finished round"))?;
        round.guard.check_finish(
            [
                draft.stream().owner_instance_key()?,
                d.stream.owner_instance_key()?,
            ],
            d.identity,
            d.generation,
            d.cursor,
        )?;
        let mut epochs = Vec::with_capacity(10);
        for (slot, event) in self.events.iter().enumerate() {
            let stream = if slot < 3 { draft.stream() } else { &d.stream };
            let receipt = round.receipts[slot]
                .ok_or_else(|| invalid("round profile missing event receipt"))?;
            // No extra wait: this exact target receipt has already completed.
            // The target dependency event also covers all draft markers.
            if stream.query_event_completion(event, receipt)?.is_none() {
                return Err(invalid(
                    "round profile event pending after acceptance receipt",
                ));
            }
            epochs.push(receipt.submission_epoch());
        }
        let elapsed = |a: usize, b: usize| -> Result<f64> {
            Ok(f64::from(self.events[a].elapsed_ms(&self.events[b])?))
        };
        let draft_parts = [elapsed(0, 1)?, elapsed(1, 2)?];
        let draft_total = elapsed(0, 2)?;
        let target_parts = (3..9)
            .map(|a| elapsed(a, a + 1))
            .collect::<Result<Vec<_>>>()?;
        let target_total = elapsed(3, 9)?;
        valid_times(&draft_parts, draft_total)?;
        valid_times(&target_parts, target_total)?;
        let target = TARGET_KEYS
            .iter()
            .zip(target_parts)
            .map(|(k, v)| ((*k).into(), json!(v)))
            .collect::<serde_json::Map<String, Value>>();
        let result = json!({
            "schema":"qwen35-verify-round-event-profile-v1", "status":"MEASURED_DIAGNOSTIC_ONLY",
            "generation":d.generation, "cursor":d.cursor, "tokens":B,
            "request_identity":sha256_hex(&round.guard.identity), "stream_owner_keys":round.guard.owners,
            "event_submission_epochs":epochs, "event_counts":EXTENTS,
            "draft_stream":{"prepare_context_ms":draft_parts[0],"block_submit_ms":draft_parts[1],"enclosing_ms":draft_total},
            "target_stream":{"parts":target,"enclosing_ms":target_total},
            "completion_boundary":"existing_acceptance_receipt_before_commit", "additional_host_synchronizations":0,
            "event_clock":"cuda_device_elapsed_ms", "same_stream_partitions_only":true,
            "cross_stream_durations_additive":false,
            "event_spans_may_include_idle_waits_and_host_submission_gaps":true,
            "producer_wait_is_exposed_dependency_not_total_draft_work":true,
            "excluded":"rollback_replay_publication_and_remaining_host_wall",
            "kernel_only":false, "instrumented_only":true, "performance_claim_permitted":false,
            "client_delivery_measured":false, "product_tps_permitted":false
        });
        *state = None;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initial_setup_requires_complete_ordered_host_partition() {
        let mut setup = InitialSetup::new();
        assert!(setup.report().is_err());
        assert!(setup.mark("noise_embedding_ms").is_err());
        for key in INITIAL_SETUP_KEYS {
            setup.mark(key).unwrap();
        }
        let report = setup.report().unwrap();
        assert_eq!(report["parts_ms"].as_object().unwrap().len(), 10);
        assert_eq!(report["additional_cuda_synchronizations"], 0);
        assert!(setup.mark("rolling_state_ms").is_err());
        setup.marks[3] = -1.0;
        assert!(setup.report().is_err());
    }
    fn guard() -> RoundGuard {
        RoundGuard {
            owners: [11, 22],
            identity: [7; 32],
            generation: 4,
            cursor: 8192,
            counts: [0, 0],
        }
    }
    #[test]
    fn switch_requires_eager_product_batch_and_no_other_profile() {
        assert!(!requested(None, false, 1024, true).unwrap());
        assert!(requested(Some("1".as_ref()), true, 32, false).unwrap());
        for (product, conflict) in [(false, false), (true, true), (false, true)] {
            assert!(requested(Some("1".as_ref()), product, 32, conflict).is_err());
        }
        for value in ["", "0", "true", " 1", "1 "] {
            assert!(requested(Some(value.as_ref()), true, 32, false).is_err());
        }
    }
    #[test]
    fn resident_budget_uses_effective_ceiling_not_owner_configuration() {
        // The owner template reserves 1024 tokens; HTTP overrides that ceiling.
        // A one-token warmup constructs the driver but runs no draft round.
        for ceiling in [1, 2, 32, 64] {
            assert!(requested(Some("1".as_ref()), true, ceiling, false).unwrap());
        }
        for ceiling in [0, 65, 1024, usize::MAX] {
            assert!(requested(Some("1".as_ref()), true, ceiling, false).is_err());
        }
    }
    #[test]
    fn rejects_marker_owner_order_and_extent() {
        let g = guard();
        assert!(g.check_mark(0, 0, 11).is_ok());
        for (role, index, owner) in [(0, 0, 22), (0, 1, 11), (1, 7, 22), (2, 0, 11)] {
            assert!(g.check_mark(role, index, owner).is_err());
        }
    }
    #[test]
    fn requires_every_draft_and_target_marker() {
        for counts in [[0, 0], [2, 7], [3, 6], [3, 8], [4, 7]] {
            let mut g = guard();
            g.counts = counts;
            assert!(g.check_finish([11, 22], [7; 32], 4, 8192).is_err());
        }
        let mut g = guard();
        g.counts = EXTENTS;
        assert!(g.check_finish([11, 22], [7; 32], 4, 8192).is_ok());
    }
    #[test]
    fn rejects_stale_or_post_commit_identity() {
        let mut g = guard();
        g.counts = EXTENTS;
        for (owners, id, generation, cursor) in [
            ([22, 11], [7; 32], 4, 8192),
            ([11, 22], [8; 32], 4, 8192),
            ([11, 22], [7; 32], 5, 8192),
            ([11, 22], [7; 32], 4, 8193),
        ] {
            assert!(g.check_finish(owners, id, generation, cursor).is_err());
        }
    }
    #[test]
    fn rejects_invalid_timing_partition() {
        assert!(valid_times(&[1.0, 2.0], 3.0).is_ok());
        assert!(valid_times(&[0.0, 0.0], 0.0).is_ok());
        for parts in [
            vec![],
            vec![-1.0],
            vec![f64::NAN],
            vec![f64::INFINITY],
            vec![4.0],
        ] {
            assert!(valid_times(&parts, 3.0).is_err());
        }
        assert!(valid_times(&[1.0], f64::NAN).is_err());
    }
}
