//! Sealed Augur owner handoff. No feature bytes or invented payload SHA on CPU.
//! Full refresh is the default. Explicit retained-context reuse is sealed to
//! owner/epoch/initialized extents and still requires real rolling admission.
use super::*;
use crate::checkpoint_target_cuda::dflash_capture::{OwnerFeatureIdentity, OwnerFeatureLease};

fn reusable_phase(prepared: bool, submitted: bool, unused: bool) -> bool {
    if unused { !prepared && !submitted } else { prepared && submitted }
}

fn validate_input(
    weights: &DFlashQ8CudaWeights,
    lease: &OwnerFeatureLease<'_>,
    noise: CudaBufferView,
    anchor: u32,
    generation: u64,
) -> Result<OwnerFeatureIdentity> {
    weights.config().validate_exact_profile().map_err(err)?;
    let identity = lease.identity();
    let features = lease.view().map_err(err)?;
    let device = weights.device_ordinal();
    if generation == 0
        || anchor as usize >= 248320
        || !(1..=2048).contains(&identity.rows())
        || identity.end().checked_add(B).is_none_or(|v| v > 262144)
        || features.len_bytes() != identity.rows() * 5 * H * 4
        || noise.len_bytes() != B * H * 4
        || features.device_ordinal() != device
        || noise.device_ordinal() != device
        || lease.stream().device_ordinal() != device
    {
        return Err("Augur initialized owner/noise handoff mismatch".into());
    }
    Ok(identity)
}
impl DFlashQ8CudaBlock {
    pub(crate) fn from_augur_owner(
        weights: &DFlashQ8CudaWeights,
        lease: &OwnerFeatureLease<'_>,
        noise: CudaBufferView,
        anchor: u32,
        generation: u64,
    ) -> Result<Self> {
        let identity = validate_input(weights, lease, noise, anchor, generation)?;
        // Explicit completion rather than assuming same-device implies ordering.
        lease.stream().synchronize().map_err(err)?;
        Self::allocate_device_features(
            weights,
            lease.view().map_err(err)?,
            InitialNoise::Device(noise),
            identity.rows(),
            identity.start(),
            2048,
            2048,
            None,
            anchor,
            generation,
            DraftFeatureIdentity::OwnerLease(identity),
            None,
        )
    }
    pub(crate) fn refresh_augur_owner(
        &mut self,
        lease: &OwnerFeatureLease<'_>,
        noise: CudaBufferView,
        anchor: u32,
        generation: u64,
    ) -> Result<()> {
        self.rebind_augur_owner(lease, noise, anchor, generation, false, false)
    }

    pub(crate) fn refresh_augur_owner_retaining_context(
        &mut self, lease: &OwnerFeatureLease<'_>, noise: CudaBufferView,
        anchor: u32, generation: u64,
    ) -> Result<()> {
        self.rebind_augur_owner(lease, noise, anchor, generation, false, true)
    }

    /// Bind the first real request into the startup-allocated, never-submitted
    /// block. This does not relax the completed-round refresh state machine.
    pub(crate) fn seed_preallocated_augur_owner(
        &mut self,
        lease: &OwnerFeatureLease<'_>,
        noise: CudaBufferView,
        anchor: u32,
        generation: u64,
    ) -> Result<()> {
        self.rebind_augur_owner(lease, noise, anchor, generation, true, false)
    }

    fn rebind_augur_owner(
        &mut self,
        lease: &OwnerFeatureLease<'_>,
        noise: CudaBufferView,
        anchor: u32,
        generation: u64,
        unused: bool,
        reuse: bool,
    ) -> Result<()> {
        if self.graph.is_some()
            || !reusable_phase(self.prepared, self.submitted, unused)
            || self.poisoned
            || self.physical_capacity != 2048
            || self.window != 2048
            || self.candidates.is_some()
            || generation <= self.request_generation
        {
            return Err("Augur draft workspace is not reusable".into());
        }
        let DraftFeatureIdentity::OwnerLease(prior) = self.feature_identity else {
            return Err("payload-SHA draft may not change provenance to owner lease".into());
        };
        let next = validate_input(&self.weights, lease, noise, anchor, generation)?;
        prior.validate_successor(next).map_err(err)?;
        let (retained, evicted) = if reuse {
            if unused || !self.context_rows_launch || !self.block_rows_launch || !self.q8_tile4 {
                return Err("retained Augur context requires a completed row-projection block".into());
            }
            prior.retained_context_rows(next).map_err(err)?
        } else { (0, 0) };
        if self.start != prior.start() || self.context != prior.rows() {
            return Err("Augur resident extent drift".into());
        }
        // Both streams are completed before reusing target seed/chronological
        // storage; no outstanding reader can race a refresh or reset.
        self.poisoned = true;
        let result = (|| -> Result<()> {
            self.stream.synchronize().map_err(err)?;
            lease.stream().synchronize().map_err(err)?;
            if evicted > 0 && retained > 0 {
                for layer in 0..5 {
                    for kind in ["k", "v"] {
                        // The old block has drained. Only chronological context
                        // rows survive; speculative query rows are never kept.
                        unsafe {
                            self.buffers[&format!("{kind}{layer}")]
                                .shift_rows_u32_async_unchecked(&self.stream, KV * 4, retained, evicted)
                                .map_err(err)?;
                        }
                    }
                }
            }
            unsafe {
                self.buffers["features"]
                    .copy_from_device_async_unchecked(&self.stream, 0, lease.view().map_err(err)?)
                    .map_err(err)?;
                #[cfg(test)]
                if FAIL_AFTER_FEATURE_COPY.with(|flag| flag.replace(false)) {
                    return Err(
                        "injected Augur refresh failure after borrowed feature copy".into(),
                    );
                }
                self.buffers["hidden0"]
                    .copy_from_device_async_unchecked(&self.stream, 0, noise)
                    .map_err(err)?;
            }
            Ok(())
        })();
        // Always drain, even if the second copy/validation fails. The lease is
        // still borrowed here. A failed partial refresh permanently poisons us.
        let drained = self.stream.synchronize().map_err(err);
        result?;
        drained?;
        self.start = next.start();
        self.context = next.rows();
        self.context_first_row = retained;
        self.capture_fc_ready = false;
        self.cached_capture = None;
        self.feature_identity = DraftFeatureIdentity::OwnerLease(next);
        self.anchor_token = anchor;
        self.request_generation = generation;
        self.prepared = false;
        self.submitted = false;
        self.poisoned = false;
        Ok(())
    }
    pub(crate) fn submitted_augur_identity(&self) -> Result<(u64, u32, OwnerFeatureIdentity)> {
        if !self.submitted || self.poisoned {
            return Err("Augur draft not submitted".into());
        }
        let DraftFeatureIdentity::OwnerLease(identity) = self.feature_identity else {
            return Err("SHA-based draft is not an Augur owner lease".into());
        };
        Ok((self.request_generation, self.anchor_token, identity))
    }
}
#[cfg(test)]
thread_local! {
    pub(crate) static FAIL_AFTER_FEATURE_COPY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    #[test]
    fn unused_seed_and_completed_refresh_are_distinct_states() {
        for prepared in [false, true] {
            for submitted in [false, true] {
                assert_eq!(reusable_phase(prepared, submitted, true), !prepared && !submitted);
                assert_eq!(reusable_phase(prepared, submitted, false), prepared && submitted);
                assert!(!(reusable_phase(prepared, submitted, true) && reusable_phase(prepared, submitted, false)));
            }
        }
    }
}
