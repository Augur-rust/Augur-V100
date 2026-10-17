//! Same-stream FP32 projection records for partial target commit. No token authority.
use super::*;
use std::cell::Cell;
mod numeric_observer;

const LAYERS: usize = 48;
const QKV: usize = linear::QKV_ELEMENTS;
const HEADS: usize = linear::VALUE_HEADS;
const PER_LAYER: usize = B * (QKV + 2 * HEADS);
pub(super) const RECORD_BYTES: usize = LAYERS * PER_LAYER * 4;

pub(in super::super) fn switch(value: Option<&std::ffi::OsStr>) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" => Ok(true),
        _ => Err(invalid("ReplaySSM requires AUGUR_QWEN35_REPLAY_SSM exactly 1")),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Stamp {
    stream: usize,
    identity: [u8; 32],
    cursor: usize,
    generation: u64,
}
impl Stamp {
    fn of(d: &Driver) -> Result<Self> {
        Ok(Self {
            stream: d.stream.owner_instance_key()?,
            identity: d.identity,
            cursor: d.cursor,
            generation: d.generation,
        })
    }
}
#[derive(Clone, Copy)]
struct Recording {
    stamp: Stamp,
    mask: u64,
    sealed: bool,
}
impl Recording {
    fn with_record(mut self, stamp: Stamp, index: usize) -> Result<Self> {
        if self.stamp != stamp || self.sealed || index >= 64
            || self.mask & (1u64 << index) != 0
        {
            return Err(invalid("ReplaySSM duplicate/stale record"));
        }
        self.mask |= 1u64 << index;
        Ok(self)
    }

    fn require(&self, stamp: Stamp, mask: u64, committed: usize) -> Result<()> {
        if self.stamp != stamp || self.mask != mask || !self.sealed || !(1..=B).contains(&committed)
        {
            return Err(invalid(
                "ReplaySSM stale/incomplete record or committed input extent",
            ));
        }
        Ok(())
    }
}

struct Layer {
    offset: usize,
    prefixes: Vec<CudaQwen35PrefillLinearChunkV1>,
}

pub(super) struct ReplaySsm {
    arena: OwnedCudaBuffer,
    audit_features: Option<OwnedCudaBuffer>,
    numeric_directory: Option<PathBuf>,
    layers: Vec<Option<Layer>>,
    expected_mask: u64,
    recording: Cell<Option<Recording>>,
}
impl ReplaySsm {
    pub(super) fn new(d: &Driver, w: &PrefillWorkspace, epsilon: f32) -> Result<Self> {
        Self::new_before_bind(d, w, epsilon, || Ok(()))
    }

    // All device allocations precede the hook; later work only binds views.
    // The hook can enqueue independent draft history while CPU preflight runs.
    pub(super) fn new_before_bind(
        d: &Driver, w: &PrefillWorkspace, epsilon: f32,
        before_bind: impl FnOnce() -> Result<()>,
    ) -> Result<Self> {
        let audit_enabled = switch(std::env::var_os("AUGUR_QWEN35_REPLAY_SSM_AUDIT").as_deref())?;
        let numeric_directory = numeric_observer::directory(
            std::env::var_os("AUGUR_QWEN35_REPLAY_NUMERIC_DIR").as_deref(), audit_enabled)?;
        if d.replay_prefix_family && !d.replay_checkpoint_read {
            return Err(invalid("ReplaySSM prefix family requires checkpoint-read binding"));
        }
        if d.plans.len() != 64 || d.verified_features.is_none() {
            return Err(invalid(
                "ReplaySSM requires full target and verified features",
            ));
        }
        let arena = OwnedCudaBuffer::allocate(d.stream.device_ordinal(), RECORD_BYTES)?;
        let audit_features = audit_enabled
            .then(|| OwnedCudaBuffer::allocate(d.stream.device_ordinal(), B * 5 * H * 4))
            .transpose()?;
        before_bind()?;
        let mut layers = Vec::with_capacity(64);
        let mut count = 0;
        let mut expected_mask = 0;
        for (index, plan) in d.plans.iter().enumerate() {
            let DevicePlan::Linear(p) = plan else {
                layers.push(None);
                continue;
            };
            if count >= LAYERS {
                return Err(invalid("ReplaySSM linear layer count"));
            }
            let projection = p
                .projection
                .as_ref()
                .ok_or_else(|| invalid("ReplaySSM projection missing"))?;
            let offset = count * PER_LAYER;
            let mut prefixes = Vec::with_capacity(B);
            // A checked full-B8 binding covers the same live backing allocations
            // as every shorter replay. Derivation changes only row extents;
            // submit still validates metadata and the exact stream owner.
            for rows in if d.replay_prefix_family { B..=B } else { 1..=B } {
                let views = CudaQwen35PrefillLinearChunkViewsV1 {
                        qkv: plan_f32_subview(&arena, offset, rows * QKV)?,
                        z: prefill_f32_view(&w.z, rows * linear::RECURRENT_ELEMENTS)?,
                        conv_weights: plan_f32_view(&p.conv, linear::CONV_WEIGHT_ELEMENTS)?,
                        conv_history: plan_f32_view(&p.history, linear::CONV_HISTORY_ELEMENTS)?,
                        alpha_raw: plan_f32_subview(&arena, offset + B * QKV, rows * HEADS)?,
                        beta_raw: plan_f32_subview(
                            &arena,
                            offset + B * (QKV + HEADS),
                            rows * HEADS,
                        )?,
                        dt_bias: plan_f32_view(&projection.dt_bias, HEADS)?,
                        ssm_a: plan_f32_view(&projection.ssm_a, HEADS)?,
                        state: plan_f32_view(&p.state, linear::STATE_ELEMENTS)?,
                        norm_weight: plan_f32_view(&p.ssm_norm, linear::HEAD_DIM)?,
                        activated: prefill_f32_view(&w.activated, rows * QKV)?,
                        recurrent: prefill_f32_view(
                            &w.recurrent,
                            rows * linear::RECURRENT_ELEMENTS,
                        )?,
                        output: prefill_f32_view(
                            &w.attention_output,
                            rows * linear::RECURRENT_ELEMENTS,
                        )?,
                    };
                prefixes.push(if d.replay_checkpoint_read {
                    CudaQwen35PrefillLinearChunkV1::bind_with_checkpoint_fold(&d.stream, views, rows, epsilon,
                        plan_f32_view(&d.backups[index * 2], linear::CONV_HISTORY_ELEMENTS)?,
                        plan_f32_view(&d.backups[index * 2 + 1], linear::STATE_ELEMENTS)?)?
                } else { CudaQwen35PrefillLinearChunkV1::bind(&d.stream, views, rows, epsilon)? });
            }
            if d.replay_prefix_family {
                let full = prefixes.pop().ok_or_else(|| invalid("ReplaySSM full binding missing"))?;
                for rows in 1..=B {
                    prefixes.push(full.checkpoint_prefix_from_b8(&d.stream, rows)?);
                }
            }
            layers.push(Some(Layer { offset, prefixes }));
            expected_mask |= 1u64 << index;
            count += 1;
        }
        if count != LAYERS {
            return Err(invalid("ReplaySSM requires all 48 linear layers"));
        }
        Ok(Self {
            arena,
            audit_features,
            numeric_directory,
            layers,
            expected_mask,
            recording: Cell::new(None),
        })
    }

    pub(super) fn allocated_bytes(&self) -> usize {
        RECORD_BYTES
            + self
                .audit_features
                .as_ref()
                .map_or(0, OwnedCudaBuffer::len_bytes)
    }

    pub(super) fn audit_fold(&self, d: &Driver, committed: usize,
        head: Option<&output_head::PackedOutputHead>, serial_attention_rows: usize, vector_attention_rows: usize) -> Result<Option<Value>> {
        let Some(features) = &self.audit_features else {
            return Ok(None);
        };
        let observation = self.numeric_directory.as_deref().map(|root| {
            numeric_observer::Before::capture(d, committed, root, serial_attention_rows, vector_attention_rows,
                head.ok_or_else(|| invalid("replay numeric observer requires packed target head"))?)
        }).transpose()?;
        let folded_state = d.state_digest()?;
        d.checkpoint(true)?;
        for row in 0..committed {
            d.check_session_cancellation("ReplaySSM reference replay")?;
            d.bind_gather(d.path.view(row * 4, 4)?)?
                .submit_for_plan(&d.stream)?;
            d.forward_capturing(d.cursor + row, Some((features, row)), None)?;
        }
        let reference_state = d.state_digest()?;
        let digest_prefix = |buffer: &OwnedCudaBuffer| -> Result<Vec<u8>> {
            augur_backend::submit_page_digest_parallel_input_view_into_view_v1(
                &d.stream,
                buffer.view(0, committed * 5 * H * 4)?,
                view(&d.feature_digest)?,
            )?;
            let mut bytes = vec![0; QWEN35_KV8_PAGE_DIGEST_BYTES];
            d.feature_digest.copy_to_host_sync(&d.stream, &mut bytes)?;
            Ok(bytes)
        };
        let recorded_features = digest_prefix(
            d.verified_features
                .as_ref()
                .ok_or_else(|| invalid("ReplaySSM audit missing verified features"))?,
        )?;
        let reference_features = digest_prefix(features)?;
        if let Some(observation) = observation {
            observation.finish(d, features, &folded_state, &reference_state)?;
        }
        if folded_state != reference_state || recorded_features != reference_features {
            return Err(invalid(format!(
                "ReplaySSM reference mismatch: state_exact={}, features_exact={}, committed={}",
                folded_state == reference_state,
                recorded_features == reference_features,
                committed
            )));
        }
        Ok(Some(
            json!({"committed_inputs":committed,"state_fingerprint_exact":true,
            "feature_prefix_fingerprint_exact":true,"reference_backbone_rows":committed,
            "all_cache_bytes_including_rejected_tail_checked":true,"performance_eligible":false}),
        ))
    }

    pub(super) fn begin(&self, d: &Driver) -> Result<()> {
        self.recording.set(Some(Recording {
            stamp: Stamp::of(d)?,
            mask: 0,
            sealed: false,
        }));
        Ok(())
    }

    fn next_recording(&self, d: &Driver, index: usize) -> Result<Option<Recording>> {
        let Some(_layer) = self
            .layers
            .get(index)
            .ok_or_else(|| invalid("ReplaySSM layer extent"))?
        else {
            return Ok(None);
        };
        let recording = self
            .recording
            .get()
            .ok_or_else(|| invalid("ReplaySSM record not started"))?;
        Ok(Some(recording.with_record(Stamp::of(d)?, index)?))
    }

    fn copy_record(&self, d: &Driver, w: &PrefillWorkspace, index: usize) -> Result<()> {
        let Some(layer) = self.layers.get(index)
            .ok_or_else(|| invalid("ReplaySSM layer extent"))? else {
            return Ok(());
        };
        let ranges = [
            (&w.qkv, layer.offset, B * QKV),
            (&w.alpha_raw, layer.offset + B * QKV, B * HEADS),
            (&w.beta_raw, layer.offset + B * (QKV + HEADS), B * HEADS),
        ];
        if d.batched_replay_copies {
            let copies = [
                (&self.arena, ranges[0].1 * 4, ranges[0].0.view(0, ranges[0].2 * 4)?),
                (&self.arena, ranges[1].1 * 4, ranges[1].0.view(0, ranges[1].2 * 4)?),
                (&self.arena, ranges[2].1 * 4, ranges[2].0.view(0, ranges[2].2 * 4)?),
            ];
            // SAFETY: this recorder and Driver retain arena/workspace owners
            // through the same stream drain. Sources remain immutable until
            // the three independent copies finish, including graph execution.
            unsafe { OwnedCudaBuffer::copy_independent_batch_async_unchecked(&d.stream, &copies)?; }
            d.replay_copy_batch_enqueues.set(d.replay_copy_batch_enqueues.get() + 1);
            return Ok(());
        }
        for (source, offset, elements) in ranges {
            copy(
                &d.stream,
                &self.arena,
                offset * 4,
                source.view(0, elements * 4)?,
            )?;
        }
        Ok(())
    }

    pub(super) fn capture(&self, d: &Driver, w: &PrefillWorkspace, index: usize) -> Result<()> {
        let Some(recording) = self.next_recording(d, index)? else {
            return Ok(());
        };
        self.copy_record(d, w, index)?;
        self.recording.set(Some(recording));
        Ok(())
    }

    // Only immutable pointer/extent commands enter the CUDA graph. Dynamic
    // cursor/generation bookkeeping stays outside and is checked per launch.
    pub(super) fn capture_graph_copy(&self, d: &Driver, w: &PrefillWorkspace, index: usize) -> Result<()> {
        if self.recording.get().is_some() || self.audit_features.is_some()
            || self.numeric_directory.is_some()
        {
            return Err(invalid("grouped graph requires idle unaudited ReplaySSM records"));
        }
        self.copy_record(d, w, index)
    }

    pub(super) fn mark_graph_capture(&self, d: &Driver, index: usize) -> Result<()> {
        let recording = self.next_recording(d, index)?
            .ok_or_else(|| invalid("grouped graph record must be a linear layer"))?;
        self.recording.set(Some(recording));
        Ok(())
    }

    pub(super) fn seal(&self, d: &Driver) -> Result<()> {
        let mut recording = self
            .recording
            .get()
            .ok_or_else(|| invalid("ReplaySSM missing record"))?;
        if recording.stamp != Stamp::of(d)?
            || recording.mask != self.expected_mask
            || recording.sealed
        {
            return Err(invalid("ReplaySSM incomplete record"));
        }
        recording.sealed = true;
        self.recording.set(Some(recording));
        Ok(())
    }

    pub(super) fn fold(&self, d: &Driver, committed: usize) -> Result<()> {
        // Consume before any state mutation. Failure poisons the enclosing Driver.
        let recording = self
            .recording
            .take()
            .ok_or_else(|| invalid("ReplaySSM record already consumed"))?;
        recording.require(Stamp::of(d)?, self.expected_mask, committed)?;
        d.restore_recorded_prefix_impl(committed, d.replay_checkpoint_read)?;
        for layer in self.layers.iter().flatten() {
            if d.replay_checkpoint_read {
                layer.prefixes[committed - 1].submit_checkpoint_fold(&d.stream)?;
                d.checkpoint_read_fold_layers.set(d.checkpoint_read_fold_layers.get() + 1);
                d.shared_replay_fold_layers.set(d.shared_replay_fold_layers.get() + 1);
            } else if d.shared_replay_fold_scalars {
                layer.prefixes[committed - 1].submit_state_fold_shared_scalars(&d.stream)?;
                d.shared_replay_fold_layers.set(d.shared_replay_fold_layers.get() + 1);
            } else {
                layer.prefixes[committed - 1].submit_state_fold(&d.stream)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grouped_records_rebind_current_stamp_and_reject_duplicates() {
        let stamp = Stamp { stream: 11, identity: [9; 32], cursor: 8191, generation: 7 };
        let mut record = Recording { stamp, mask: 0, sealed: false };
        for first in (0..64).step_by(4) {
            for layer in first..first + 3 {
                record = record.with_record(stamp, layer).unwrap();
                assert!(record.with_record(stamp, layer).is_err());
            }
        }
        assert_eq!(record.mask, 0x7777777777777777);
        assert!(record.with_record(stamp, 64).is_err());
        for wrong in [Stamp { cursor: 8192, ..stamp }, Stamp { generation: 8, ..stamp },
            Stamp { stream: 12, ..stamp }, Stamp { identity: [8; 32], ..stamp }] {
            assert!(record.with_record(wrong, 3).is_err());
        }
        record.sealed = true;
        assert!(record.with_record(stamp, 3).is_err());
        assert!(record.require(stamp, 0x7777777777777777, B).is_ok());
    }

    #[test]
    fn records_bind_complete_same_owner_input_prefix() {
        let stamp = Stamp {
            stream: 11,
            identity: [9; 32],
            cursor: 65440,
            generation: 7,
        };
        let record = Recording {
            stamp,
            mask: 0x7777777777777777,
            sealed: true,
        };
        for committed in 1..=B {
            assert!(record.require(stamp, record.mask, committed).is_ok());
        }
        for committed in [0, B + 1] {
            assert!(record.require(stamp, record.mask, committed).is_err());
        }
        for wrong in [
            Stamp {
                stream: 12,
                ..stamp
            },
            Stamp {
                identity: [0; 32],
                ..stamp
            },
            Stamp {
                cursor: stamp.cursor + 1,
                ..stamp
            },
            Stamp {
                generation: 8,
                ..stamp
            },
        ] {
            assert!(record.require(wrong, record.mask, 4).is_err());
        }
        assert!(record.require(stamp, record.mask ^ 1, 4).is_err());
        assert!(Recording {
            sealed: false,
            ..record
        }
        .require(stamp, record.mask, 4)
        .is_err());
        assert_eq!(RECORD_BYTES, 15_876_096);
    }
    #[test]
    fn switch_requires_explicit_exact_opt_in() {
        use std::ffi::OsStr;
        assert!(!switch(None).unwrap());
        assert!(switch(Some(OsStr::new("1"))).unwrap());
        for value in ["", "true", "0", "yes"] {
            assert!(switch(Some(OsStr::new(value))).is_err());
        }
    }
}
