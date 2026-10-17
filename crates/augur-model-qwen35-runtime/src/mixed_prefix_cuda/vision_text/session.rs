//! One bounded same-session owner: KV, GDN and position tables move together.
//! Metadata never authorizes GPU reuse on a digest match alone. The exact token
//! ledger, image payloads and complete position schedule must all match.
use super::*;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy)]
pub(super) struct Key<'a> {
    pub(super) id: &'a str,
    pub(super) generation: u64,
}
impl Key<'_> {
    pub(super) fn validate(self) -> Result<()> {
        if !(8..=64).contains(&self.id.len())
            || !self
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err(invalid("Vision session identity"));
        }
        Ok(())
    }
}

pub(super) struct Ledger {
    pub(super) id: String,
    pub(super) generation: u64,
    pub(super) tokens: Vec<u32>,
    pub(super) positions: Vec<[i64; 3]>,
    pub(super) images: Vec<String>,
}
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Decision {
    pub(super) reused_tokens: usize,
    pub(super) reason: &'static str,
}
fn decide(
    ledger: &Ledger,
    key: Key<'_>,
    tokens: &[u32],
    positions: &[[i64; 3]],
    images: &[String],
) -> Result<Decision> {
    key.validate()?;
    if ledger.id != key.id {
        if key.generation != 0 {
            return Err(invalid("Vision foreign session generation"));
        }
        return Ok(Decision {
            reused_tokens: 0,
            reason: "new_session_replaces_previous",
        });
    }
    if ledger.generation != key.generation {
        return Err(invalid("Vision stale session generation"));
    }
    let reason = if !tokens.starts_with(&ledger.tokens) {
        "rebuild_edited_or_retokenized_history"
    } else if tokens.len() <= ledger.tokens.len() {
        "rebuild_without_new_input"
    } else if ledger.images != images {
        "rebuild_image_payload_changed"
    } else if ledger.positions != positions {
        // New images can change future THW rows. Do not mutate immutable
        // tables bound into the existing full-attention plans.
        "rebuild_position_schedule_changed"
    } else {
        return Ok(Decision {
            reused_tokens: ledger.tokens.len(),
            reason: "exact_same_session_suffix",
        });
    };
    Ok(Decision {
        reused_tokens: 0,
        reason,
    })
}
fn positions(mrope: &Qwen35ImageMropePlan) -> Result<Vec<[i64; 3]>> {
    (0..mrope.capacity())
        .map(|p| Ok(mrope.position_thw(p)?))
        .collect()
}
fn digest(tokens: &[u32]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"qwen35-vision-committed-input-u32le-v1");
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
impl Ledger {
    pub(super) fn decision(
        &self,
        key: Key<'_>,
        ids: &[u32],
        mrope: &Qwen35ImageMropePlan,
        images: &[String],
    ) -> Result<Decision> {
        decide(self, key, ids, &positions(mrope)?, images)
    }
    pub(super) fn committed(
        key: Key<'_>,
        generation: u64,
        ids: &[u32],
        generated: &[u32],
        consumed: usize,
        mrope: &Qwen35ImageMropePlan,
        images: Vec<String>,
    ) -> Result<Self> {
        key.validate()?;
        let generated_consumed = consumed
            .checked_sub(ids.len())
            .ok_or_else(|| invalid("Vision session input cursor"))?;
        if ids.is_empty()
            || consumed > mrope.capacity()
            || generated_consumed > generated.len()
            || generation == 0
        {
            return Err(invalid("Vision session terminal ledger extent"));
        }
        let mut tokens = Vec::with_capacity(consumed);
        tokens.extend_from_slice(ids);
        // A length-limited last output has not entered KV/GDN yet. It is not
        // committed input: the next turn must consume that real suffix row.
        tokens.extend_from_slice(&generated[..generated_consumed]);
        Ok(Self {
            id: key.id.to_owned(),
            generation,
            tokens,
            positions: positions(mrope)?,
            images,
        })
    }
    pub(super) fn summary(&self) -> Value {
        json!({"session_id":self.id,"generation":self.generation,"committed_input_tokens":self.tokens.len(),
            "committed_input_sha256":digest(&self.tokens),"position_capacity":self.positions.len(),"valid":true})
    }
}

pub(super) struct State {
    // Bound operations retire before their backing owners. Drop drains before
    // any of these fields can be freed, including cancellation/error paths.
    pub(super) selector: CudaQwen35VocabTop1ArgsV1,
    pub(super) token: OwnedCudaBuffer,
    pub(super) head: DeviceOutputHeadPlan,
    pub(super) plans: Vec<DevicePlan>,
    _tables: Arc<FullRopeTables>,
    pub(super) start: OwnedCudaEvent,
    pub(super) stop: OwnedCudaEvent,
    pub(super) terminal: OwnedCudaEvent,
    pub(super) stream: OwnedCudaStream,
    pub(super) ledger: Option<Ledger>,
    pub(super) retained_bytes: u64,
    pub(super) retained_allocations: u64,
}
impl Drop for State {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}
impl State {
    pub(super) fn new(
        target: &target_weights::TargetWeights,
        model: &Qwen35Config,
        capacity: usize,
        mrope: &Qwen35ImageMropePlan,
        first: &[f32],
        cancelled: &impl Fn() -> bool,
    ) -> Result<Self> {
        let before = cuda_device_allocation_snapshot();
        let stream = OwnedCudaStream::create(0)?;
        let tables = Arc::new(FullRopeTables::new_image(&stream, mrope)?);
        let mut plans = Vec::<DevicePlan>::with_capacity(FULL_TARGET_LAYER_COUNT);
        for layer in &target.layers {
            check_cancel(cancelled)?;
            let source = plans
                .last()
                .map(DevicePlan::final_hidden_f32_view)
                .transpose()?;
            plans.push(match layer {
                target_weights::LayerWeights::Linear(w) => {
                    DevicePlan::Linear(bind_linear_plan(&stream, w, first, None, true, source, None)?)
                }
                target_weights::LayerWeights::Full(w) => {
                    DevicePlan::Full(bind_full_plan_with_position_schedule(
                        &stream,
                        w,
                        model,
                        capacity,
                        mrope.capacity(),
                        source,
                        Some(tables.clone()),
                        Some(mrope),
                        None,
                    )?)
                }
            });
        }
        if plans.len() != FULL_TARGET_LAYER_COUNT {
            return Err(invalid("Vision session all-layer coverage"));
        }
        let head = bind_device_output_head_plan(
            &stream,
            &target.head,
            plans
                .last()
                .ok_or_else(|| invalid("Vision session plans missing"))?
                .final_hidden_view()?,
            model.rms_norm_eps as f32,
        )?;
        let token = OwnedCudaBuffer::allocate(0, 4)?;
        let selector = CudaQwen35VocabTop1ArgsV1::bind(
            &stream,
            plan_f32_view(&head.logits, model.vocab_size)?,
            token.view(0, 4)?,
            1,
        )?;
        let start = OwnedCudaEvent::create(0)?;
        let stop = OwnedCudaEvent::create(0)?;
        let terminal = OwnedCudaEvent::create(0)?;
        reset_plans(&mut plans, &stream)?;
        stream.synchronize()?;
        let after = cuda_device_allocation_snapshot();
        let retained_bytes = after
            .live_bytes
            .checked_sub(before.live_bytes)
            .ok_or_else(|| invalid("Vision session byte accounting"))?;
        let retained_allocations = after
            .live_allocations
            .checked_sub(before.live_allocations)
            .ok_or_else(|| invalid("Vision session allocation accounting"))?;
        Ok(Self {
            selector,
            token,
            head,
            plans,
            _tables: tables,
            start,
            stop,
            terminal,
            stream,
            ledger: None,
            retained_bytes,
            retained_allocations,
        })
    }
}

pub(super) fn balanced_with_retention(
    before: CudaDeviceAllocationSnapshot,
    after: CudaDeviceAllocationSnapshot,
    old_bytes: u64,
    old_allocations: u64,
    new_bytes: u64,
    new_allocations: u64,
) -> bool {
    before.live_bytes.checked_sub(old_bytes) == after.live_bytes.checked_sub(new_bytes)
        && before.live_allocations.checked_sub(old_allocations)
            == after.live_allocations.checked_sub(new_allocations)
        && before.live_bytes >= old_bytes
        && after.live_bytes >= new_bytes
        && before.live_allocations >= old_allocations
        && after.live_allocations >= new_allocations
        && after
            .successful_allocations
            .checked_sub(before.successful_allocations)
            .and_then(|v| v.checked_add(old_allocations))
            == after
                .successful_frees
                .checked_sub(before.successful_frees)
                .and_then(|v| v.checked_add(new_allocations))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Ledger {
        Ledger {
            id: "session-good".into(),
            generation: 7,
            tokens: vec![10, 11, 12],
            positions: (0..8).map(|i| [i; 3]).collect(),
            images: vec!["image-hash".into()],
        }
    }
    fn key() -> Key<'static> {
        Key {
            id: "session-good",
            generation: 7,
        }
    }
    #[test]
    fn exact_token_image_and_position_match_reuses_only_committed_prefix() {
        let l = fixture();
        let d = decide(&l, key(), &[10, 11, 12, 13, 14], &l.positions, &l.images).unwrap();
        assert_eq!(d.reused_tokens, 3);
        assert_eq!(d.reason, "exact_same_session_suffix");
    }
    #[test]
    fn edited_or_retokenized_history_is_a_full_rebuild_not_a_hash_hit() {
        let l = fixture();
        let d = decide(&l, key(), &[10, 99, 12, 13], &l.positions, &l.images).unwrap();
        assert_eq!(d.reused_tokens, 0);
        assert_eq!(d.reason, "rebuild_edited_or_retokenized_history");
    }
    #[test]
    fn image_bytes_and_future_mrope_changes_invalidate_the_whole_group() {
        let l = fixture();
        let images = vec!["other-image".into()];
        assert_eq!(
            decide(&l, key(), &[10, 11, 12, 13], &l.positions, &images)
                .unwrap()
                .reused_tokens,
            0
        );
        let mut p = l.positions.clone();
        p[7][1] += 1;
        assert_eq!(
            decide(&l, key(), &[10, 11, 12, 13], &p, &l.images)
                .unwrap()
                .reused_tokens,
            0
        );
    }
    #[test]
    fn stale_and_foreign_generations_never_reuse_or_silently_reset() {
        let l = fixture();
        for k in [
            Key {
                id: "session-good",
                generation: 6,
            },
            Key {
                id: "other-session",
                generation: 7,
            },
        ] {
            assert!(decide(&l, k, &[10, 11, 12, 13], &l.positions, &l.images).is_err());
        }
        assert_eq!(
            decide(
                &l,
                Key {
                    id: "other-session",
                    generation: 0
                },
                &[10, 11, 12, 13],
                &l.positions,
                &l.images
            )
            .unwrap()
            .reused_tokens,
            0
        );
    }
    #[test]
    fn no_new_suffix_cannot_reuse_old_logits() {
        let l = fixture();
        assert_eq!(
            decide(&l, key(), &l.tokens, &l.positions, &l.images)
                .unwrap()
                .reused_tokens,
            0
        );
    }
    #[test]
    fn session_identity_is_bounded_ascii() {
        for id in ["short", "session/foreign", "session-中文"] {
            assert!(Key { id, generation: 0 }.validate().is_err());
        }
        assert!(Key {
            id: "session-good",
            generation: 0
        }
        .validate()
        .is_ok());
    }
}
