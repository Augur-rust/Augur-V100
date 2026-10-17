//! Control-plane lifecycle for handing a Qwen prefill over to DFlash.
//!
//! This module intentionally has no CUDA or safetensors dependency.  It is the
//! proof-carrying state machine that a CUDA implementation must satisfy before
//! it may load a DFlash artifact.  The ordering mirrors the DeepSeek DSpark
//! owner: a covering completion proof is required before releasing prefill
//! storage, and DFlash has an independent owner generation and artifact
//! identity.

use core::fmt;

pub const DFLASH_LIFECYCLE_SCHEMA: &str = "qwen35-dflash-prefill-handoff-lifecycle-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DFlashDType {
    Bf16,
    F16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlashArtifactIdentity {
    pub artifact_sha256: [u8; 32],
    pub config_sha256: [u8; 32],
    pub dtype: DFlashDType,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlashLifecycleBudget {
    pub total_hbm_bytes: u64,
    pub safety_reserve_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlashSubmissionReceipt {
    pub owner_generation: u64,
    pub request_sha256: [u8; 32],
    pub request_generation: u64,
    pub submission_epoch: u64,
    pub receipt_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlashCompletionProof {
    pub owner_generation: u64,
    pub request_sha256: [u8; 32],
    pub request_generation: u64,
    pub covering_submission_epoch: u64,
    pub terminal_digest: [u8; 32],
}

/// Physical observations made after the prefill owner has been fenced and
/// released.  This is deliberately separate from [`DFlashCompletionProof`]:
/// a CUDA event proving that work finished does not, by itself, prove that the
/// allocator has actually returned the buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlashPrefillReleaseObservation {
    pub owner_generation: u64,
    pub request_sha256: [u8; 32],
    pub request_generation: u64,
    pub covering_submission_epoch: u64,
    pub observed_full_cache_bytes: u64,
    pub observed_prefill_workspace_bytes: u64,
    pub stream_quiescent: bool,
    pub release_digest: [u8; 32],
}

/// Receipt handed to the DFlash loader after a physical prefill release.
/// Consumers should use [`DFlashLifecycle::admit_dflash_after_handoff`] for
/// the real upload path so a state-only release cannot accidentally authorize
/// an artifact admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlashPrefillHandoffReceipt {
    pub owner_generation: u64,
    pub request_sha256: [u8; 32],
    pub request_generation: u64,
    pub covering_submission_epoch: u64,
    pub full_cache_bytes_released: u64,
    pub prefill_workspace_bytes_released: u64,
    pub release_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlashRequestIdentity {
    pub request_sha256: [u8; 32],
    pub request_generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DFlashLifecycleState {
    Empty,
    PrefillActive,
    PrefillTerminalPending,
    PrefillReleased,
    DFlashLoading,
    DFlashResident,
    DecodeActive,
    DecodeTerminal,
    Releasable,
    Poisoned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DFlashLifecycleError {
    InvalidState,
    InvalidIdentity,
    InvalidReceipt,
    InvalidProof,
    InvalidDigest,
    InvalidGeneration,
    BudgetExceeded,
    ArtifactMismatch,
    DTypeNotPermitted,
    BytesNotReleased,
    QuiescenceNotProven,
    Poisoned,
}

impl fmt::Display for DFlashLifecycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for DFlashLifecycleError {}

#[derive(Clone, Debug)]
pub struct DFlashLifecycle {
    state: DFlashLifecycleState,
    request: Option<DFlashRequestIdentity>,
    prefill_owner_generation: u64,
    dflash_owner_generation: u64,
    next_submission_epoch: u64,
    last_submission: Option<DFlashSubmissionReceipt>,
    prefill_terminal: Option<DFlashCompletionProof>,
    prefill_handoff: Option<DFlashPrefillHandoffReceipt>,
    artifact: Option<DFlashArtifactIdentity>,
    prefill_workspace_bytes: u64,
    full_cache_bytes: u64,
    dflash_bytes: u64,
    budget: DFlashLifecycleBudget,
}

impl DFlashLifecycle {
    pub fn new(budget: DFlashLifecycleBudget) -> Self {
        Self {
            state: DFlashLifecycleState::Empty,
            request: None,
            prefill_owner_generation: 0,
            dflash_owner_generation: 0,
            next_submission_epoch: 0,
            last_submission: None,
            prefill_terminal: None,
            prefill_handoff: None,
            artifact: None,
            prefill_workspace_bytes: 0,
            full_cache_bytes: 0,
            dflash_bytes: 0,
            budget,
        }
    }

    pub const fn state(&self) -> DFlashLifecycleState {
        self.state
    }

    pub const fn prefill_workspace_bytes(&self) -> u64 {
        self.prefill_workspace_bytes
    }

    pub const fn full_cache_bytes(&self) -> u64 {
        self.full_cache_bytes
    }

    pub const fn dflash_bytes(&self) -> u64 {
        self.dflash_bytes
    }

    pub const fn dflash_owner_generation(&self) -> u64 {
        self.dflash_owner_generation
    }

    pub fn begin_prefill(
        &mut self,
        request: DFlashRequestIdentity,
        prefill_owner_generation: u64,
        full_cache_bytes: u64,
        prefill_workspace_bytes: u64,
    ) -> Result<(), DFlashLifecycleError> {
        if self.state != DFlashLifecycleState::Empty
            || request.request_sha256 == [0; 32]
            || request.request_generation == 0
            || prefill_owner_generation == 0
            || full_cache_bytes == 0
        {
            return Err(DFlashLifecycleError::InvalidIdentity);
        }
        self.state = DFlashLifecycleState::PrefillActive;
        self.request = Some(request);
        self.prefill_owner_generation = prefill_owner_generation;
        self.dflash_owner_generation = 0;
        self.next_submission_epoch = 0;
        self.last_submission = None;
        self.prefill_terminal = None;
        self.prefill_handoff = None;
        self.artifact = None;
        self.prefill_workspace_bytes = prefill_workspace_bytes;
        self.full_cache_bytes = full_cache_bytes;
        self.dflash_bytes = 0;
        Ok(())
    }

    pub fn record_prefill_submission(
        &mut self,
        receipt_digest: [u8; 32],
    ) -> Result<DFlashSubmissionReceipt, DFlashLifecycleError> {
        let submission_epoch = self
            .next_submission_epoch
            .checked_add(1)
            .ok_or(DFlashLifecycleError::InvalidGeneration)?;
        self.record_prefill_submission_at_epoch(receipt_digest, submission_epoch)
    }

    /// Binds the lifecycle receipt to the backend's actual stream epoch. The
    /// synthetic method above remains useful for pure control-plane tests;
    /// CUDA producers should use this form for a non-replayable handoff.
    pub fn record_prefill_submission_at_epoch(
        &mut self,
        receipt_digest: [u8; 32],
        submission_epoch: u64,
    ) -> Result<DFlashSubmissionReceipt, DFlashLifecycleError> {
        if self.state != DFlashLifecycleState::PrefillActive
            || receipt_digest == [0; 32]
            || self.prefill_owner_generation == 0
            || submission_epoch == 0
            || submission_epoch <= self.next_submission_epoch
        {
            return Err(DFlashLifecycleError::InvalidReceipt);
        }
        self.next_submission_epoch = submission_epoch;
        let receipt = DFlashSubmissionReceipt {
            owner_generation: self.prefill_owner_generation,
            request_sha256: self
                .request
                .ok_or(DFlashLifecycleError::InvalidIdentity)?
                .request_sha256,
            request_generation: self
                .request
                .ok_or(DFlashLifecycleError::InvalidIdentity)?
                .request_generation,
            submission_epoch: self.next_submission_epoch,
            receipt_digest,
        };
        self.last_submission = Some(receipt);
        Ok(receipt)
    }

    pub fn mark_prefill_terminal(
        &mut self,
        proof: DFlashCompletionProof,
    ) -> Result<(), DFlashLifecycleError> {
        if self.state != DFlashLifecycleState::PrefillActive {
            return Err(DFlashLifecycleError::InvalidState);
        }
        let receipt = self
            .last_submission
            .ok_or(DFlashLifecycleError::InvalidReceipt)?;
        if proof.owner_generation != self.prefill_owner_generation
            || proof.request_sha256 != receipt.request_sha256
            || proof.request_generation != receipt.request_generation
            || proof.covering_submission_epoch != receipt.submission_epoch
            || proof.terminal_digest == [0; 32]
        {
            return Err(DFlashLifecycleError::InvalidProof);
        }
        self.prefill_terminal = Some(proof);
        self.state = DFlashLifecycleState::PrefillTerminalPending;
        Ok(())
    }

    /// Releases all prefill/full-cache allocations after the covering proof.
    /// A failed release poisons the owner in the CUDA implementation; this
    /// pure state machine only accepts the all-or-nothing zero-byte result.
    #[cfg(test)]
    pub(crate) fn release_prefill_after_completion(
        &mut self,
        proof: DFlashCompletionProof,
    ) -> Result<(), DFlashLifecycleError> {
        if self.state != DFlashLifecycleState::PrefillTerminalPending
            || self.prefill_terminal != Some(proof)
        {
            return Err(DFlashLifecycleError::InvalidProof);
        }
        self.prefill_workspace_bytes = 0;
        self.full_cache_bytes = 0;
        self.prefill_handoff = None;
        self.state = DFlashLifecycleState::PrefillReleased;
        Ok(())
    }

    /// Releases prefill storage only after a covering proof *and* a
    /// quiescent, zero-byte allocator observation.  This is the physical
    /// handoff boundary used by the DSpark-derived CUDA adapter.
    pub fn release_prefill_with_observation(
        &mut self,
        proof: DFlashCompletionProof,
        observation: DFlashPrefillReleaseObservation,
    ) -> Result<DFlashPrefillHandoffReceipt, DFlashLifecycleError> {
        if self.state != DFlashLifecycleState::PrefillTerminalPending
            || self.prefill_terminal != Some(proof)
        {
            return Err(DFlashLifecycleError::InvalidProof);
        }
        if observation.owner_generation != proof.owner_generation
            || observation.request_sha256 != proof.request_sha256
            || observation.request_generation != proof.request_generation
            || observation.covering_submission_epoch != proof.covering_submission_epoch
            || observation.release_digest == [0; 32]
        {
            return Err(DFlashLifecycleError::InvalidProof);
        }
        if !observation.stream_quiescent {
            return Err(DFlashLifecycleError::QuiescenceNotProven);
        }
        if observation.observed_full_cache_bytes != 0
            || observation.observed_prefill_workspace_bytes != 0
        {
            return Err(DFlashLifecycleError::BytesNotReleased);
        }
        let receipt = DFlashPrefillHandoffReceipt {
            owner_generation: self.prefill_owner_generation,
            request_sha256: proof.request_sha256,
            request_generation: proof.request_generation,
            covering_submission_epoch: proof.covering_submission_epoch,
            full_cache_bytes_released: self.full_cache_bytes,
            prefill_workspace_bytes_released: self.prefill_workspace_bytes,
            release_digest: observation.release_digest,
        };
        self.prefill_workspace_bytes = 0;
        self.full_cache_bytes = 0;
        self.prefill_handoff = Some(receipt);
        self.state = DFlashLifecycleState::PrefillReleased;
        Ok(receipt)
    }

    pub(crate) fn admit_dflash(
        &mut self,
        artifact: DFlashArtifactIdentity,
        expected_artifact_sha256: [u8; 32],
        expected_config_sha256: [u8; 32],
        allow_f16: bool,
    ) -> Result<(), DFlashLifecycleError> {
        if self.state != DFlashLifecycleState::PrefillReleased {
            return Err(DFlashLifecycleError::InvalidState);
        }
        if self.prefill_workspace_bytes != 0 || self.full_cache_bytes != 0 {
            return Err(DFlashLifecycleError::BytesNotReleased);
        }
        if artifact.artifact_sha256 != expected_artifact_sha256
            || artifact.config_sha256 != expected_config_sha256
            || artifact.artifact_sha256 == [0; 32]
            || artifact.config_sha256 == [0; 32]
            || artifact.bytes == 0
        {
            return Err(DFlashLifecycleError::ArtifactMismatch);
        }
        if artifact.dtype == DFlashDType::F16 && !allow_f16 {
            return Err(DFlashLifecycleError::DTypeNotPermitted);
        }
        let used = artifact
            .bytes
            .checked_add(self.budget.safety_reserve_bytes)
            .ok_or(DFlashLifecycleError::BudgetExceeded)?;
        if used > self.budget.total_hbm_bytes {
            return Err(DFlashLifecycleError::BudgetExceeded);
        }
        self.dflash_owner_generation = self
            .prefill_owner_generation
            .checked_add(1)
            .ok_or(DFlashLifecycleError::InvalidGeneration)?;
        self.artifact = Some(artifact);
        self.dflash_bytes = artifact.bytes;
        self.state = DFlashLifecycleState::DFlashLoading;
        Ok(())
    }

    /// Admission entry point for a real loader.  Unlike the legacy control
    /// method, this requires the physical handoff receipt produced by
    /// [`Self::release_prefill_with_observation`].
    pub fn admit_dflash_after_handoff(
        &mut self,
        handoff: DFlashPrefillHandoffReceipt,
        artifact: DFlashArtifactIdentity,
        expected_artifact_sha256: [u8; 32],
        expected_config_sha256: [u8; 32],
        allow_f16: bool,
    ) -> Result<(), DFlashLifecycleError> {
        if self.state != DFlashLifecycleState::PrefillReleased
            || self.prefill_handoff != Some(handoff)
            || self.prefill_workspace_bytes != 0
            || self.full_cache_bytes != 0
        {
            return Err(DFlashLifecycleError::InvalidReceipt);
        }
        self.admit_dflash(
            artifact,
            expected_artifact_sha256,
            expected_config_sha256,
            allow_f16,
        )
    }

    pub fn mark_dflash_ready(&mut self) -> Result<(), DFlashLifecycleError> {
        if self.state != DFlashLifecycleState::DFlashLoading
            || self.dflash_owner_generation == 0
            || self.artifact.is_none()
        {
            return Err(DFlashLifecycleError::InvalidState);
        }
        self.state = DFlashLifecycleState::DFlashResident;
        Ok(())
    }

    pub fn begin_decode(&mut self) -> Result<(), DFlashLifecycleError> {
        if self.state != DFlashLifecycleState::DFlashResident {
            return Err(DFlashLifecycleError::InvalidState);
        }
        self.state = DFlashLifecycleState::DecodeActive;
        Ok(())
    }

    pub fn mark_decode_terminal(&mut self) -> Result<(), DFlashLifecycleError> {
        if self.state != DFlashLifecycleState::DecodeActive {
            return Err(DFlashLifecycleError::InvalidState);
        }
        self.state = DFlashLifecycleState::DecodeTerminal;
        Ok(())
    }

    pub fn release_dflash(&mut self) -> Result<(), DFlashLifecycleError> {
        if self.state != DFlashLifecycleState::DecodeTerminal {
            return Err(DFlashLifecycleError::InvalidState);
        }
        self.dflash_bytes = 0;
        self.state = DFlashLifecycleState::Releasable;
        Ok(())
    }

    pub fn finish_request(&mut self) -> Result<(), DFlashLifecycleError> {
        if self.state != DFlashLifecycleState::Releasable
            || self.prefill_workspace_bytes != 0
            || self.full_cache_bytes != 0
            || self.dflash_bytes != 0
        {
            return Err(DFlashLifecycleError::BytesNotReleased);
        }
        self.state = DFlashLifecycleState::Empty;
        self.request = None;
        self.prefill_terminal = None;
        self.artifact = None;
        self.prefill_handoff = None;
        Ok(())
    }

    pub fn poison(&mut self) {
        self.state = DFlashLifecycleState::Poisoned;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUEST: DFlashRequestIdentity = DFlashRequestIdentity {
        request_sha256: [1; 32],
        request_generation: 7,
    };
    const ARTIFACT: DFlashArtifactIdentity = DFlashArtifactIdentity {
        artifact_sha256: [2; 32],
        config_sha256: [3; 32],
        dtype: DFlashDType::F16,
        bytes: 3_848_808_960,
    };

    fn proof(owner_generation: u64, epoch: u64) -> DFlashCompletionProof {
        DFlashCompletionProof {
            owner_generation,
            request_sha256: REQUEST.request_sha256,
            request_generation: REQUEST.request_generation,
            covering_submission_epoch: epoch,
            terminal_digest: [9; 32],
        }
    }

    fn ready_to_release() -> DFlashLifecycle {
        let mut lifecycle = DFlashLifecycle::new(DFlashLifecycleBudget {
            total_hbm_bytes: 8_000_000_000,
            safety_reserve_bytes: 256_000_000,
        });
        lifecycle.begin_prefill(REQUEST, 11, 20_000, 4_000).unwrap();
        let receipt = lifecycle.record_prefill_submission([4; 32]).unwrap();
        let terminal = proof(11, receipt.submission_epoch);
        lifecycle.mark_prefill_terminal(terminal).unwrap();
        lifecycle
            .release_prefill_after_completion(terminal)
            .unwrap();
        lifecycle
    }

    #[test]
    fn deepseek_order_is_enforced() {
        let mut lifecycle = DFlashLifecycle::new(DFlashLifecycleBudget {
            total_hbm_bytes: 8_000_000_000,
            safety_reserve_bytes: 256_000_000,
        });
        lifecycle.begin_prefill(REQUEST, 11, 20_000, 4_000).unwrap();
        assert_eq!(
            lifecycle.admit_dflash(ARTIFACT, [2; 32], [3; 32], true),
            Err(DFlashLifecycleError::InvalidState)
        );
        let receipt = lifecycle.record_prefill_submission([4; 32]).unwrap();
        let terminal = proof(11, receipt.submission_epoch);
        lifecycle.mark_prefill_terminal(terminal).unwrap();
        assert_eq!(lifecycle.prefill_workspace_bytes(), 4_000);
        lifecycle
            .release_prefill_after_completion(terminal)
            .unwrap();
        assert_eq!(lifecycle.full_cache_bytes(), 0);
        assert_eq!(lifecycle.prefill_workspace_bytes(), 0);
    }

    #[test]
    fn independent_dflash_generation_and_terminal_release() {
        let mut lifecycle = ready_to_release();
        lifecycle
            .admit_dflash(ARTIFACT, [2; 32], [3; 32], true)
            .unwrap();
        assert_eq!(lifecycle.dflash_owner_generation(), 12);
        lifecycle.mark_dflash_ready().unwrap();
        lifecycle.begin_decode().unwrap();
        lifecycle.mark_decode_terminal().unwrap();
        lifecycle.release_dflash().unwrap();
        lifecycle.finish_request().unwrap();
        assert_eq!(lifecycle.state(), DFlashLifecycleState::Empty);
    }

    #[test]
    fn physical_handoff_requires_quiescent_zero_byte_observation() {
        let mut lifecycle = DFlashLifecycle::new(DFlashLifecycleBudget {
            total_hbm_bytes: 8_000_000_000,
            safety_reserve_bytes: 256_000_000,
        });
        lifecycle.begin_prefill(REQUEST, 11, 20_000, 4_000).unwrap();
        let submission = lifecycle.record_prefill_submission([4; 32]).unwrap();
        let terminal = proof(11, submission.submission_epoch);
        lifecycle.mark_prefill_terminal(terminal).unwrap();
        let base = DFlashPrefillReleaseObservation {
            owner_generation: 11,
            request_sha256: REQUEST.request_sha256,
            request_generation: REQUEST.request_generation,
            covering_submission_epoch: submission.submission_epoch,
            observed_full_cache_bytes: 0,
            observed_prefill_workspace_bytes: 0,
            stream_quiescent: false,
            release_digest: [10; 32],
        };
        assert_eq!(
            lifecycle.release_prefill_with_observation(terminal, base),
            Err(DFlashLifecycleError::QuiescenceNotProven)
        );
        let mut nonzero = base;
        nonzero.stream_quiescent = true;
        nonzero.observed_full_cache_bytes = 1;
        assert_eq!(
            lifecycle.release_prefill_with_observation(terminal, nonzero),
            Err(DFlashLifecycleError::BytesNotReleased)
        );
        let mut good = base;
        good.stream_quiescent = true;
        let handoff = lifecycle
            .release_prefill_with_observation(terminal, good)
            .unwrap();
        assert_eq!(handoff.full_cache_bytes_released, 20_000);
        assert_eq!(handoff.prefill_workspace_bytes_released, 4_000);
        assert_eq!(lifecycle.state(), DFlashLifecycleState::PrefillReleased);
    }

    #[test]
    fn real_admission_requires_the_exact_physical_handoff_receipt() {
        let mut lifecycle = DFlashLifecycle::new(DFlashLifecycleBudget {
            total_hbm_bytes: 8_000_000_000,
            safety_reserve_bytes: 256_000_000,
        });
        lifecycle.begin_prefill(REQUEST, 11, 20_000, 4_000).unwrap();
        let submission = lifecycle.record_prefill_submission([4; 32]).unwrap();
        let terminal = proof(11, submission.submission_epoch);
        lifecycle.mark_prefill_terminal(terminal).unwrap();
        let handoff = lifecycle
            .release_prefill_with_observation(
                terminal,
                DFlashPrefillReleaseObservation {
                    owner_generation: 11,
                    request_sha256: REQUEST.request_sha256,
                    request_generation: REQUEST.request_generation,
                    covering_submission_epoch: submission.submission_epoch,
                    observed_full_cache_bytes: 0,
                    observed_prefill_workspace_bytes: 0,
                    stream_quiescent: true,
                    release_digest: [10; 32],
                },
            )
            .unwrap();
        let mut stale = handoff;
        stale.release_digest = [11; 32];
        assert_eq!(
            lifecycle.admit_dflash_after_handoff(stale, ARTIFACT, [2; 32], [3; 32], true),
            Err(DFlashLifecycleError::InvalidReceipt)
        );
        lifecycle
            .admit_dflash_after_handoff(handoff, ARTIFACT, [2; 32], [3; 32], true)
            .unwrap();
        assert_eq!(lifecycle.state(), DFlashLifecycleState::DFlashLoading);
    }

    #[test]
    fn stale_prefill_proof_cannot_release() {
        let mut lifecycle = DFlashLifecycle::new(DFlashLifecycleBudget {
            total_hbm_bytes: 8_000_000_000,
            safety_reserve_bytes: 256_000_000,
        });
        lifecycle.begin_prefill(REQUEST, 11, 20_000, 4_000).unwrap();
        let receipt = lifecycle.record_prefill_submission([4; 32]).unwrap();
        lifecycle
            .mark_prefill_terminal(proof(11, receipt.submission_epoch + 1))
            .unwrap_err();
        assert_eq!(lifecycle.state(), DFlashLifecycleState::PrefillActive);
    }

    #[test]
    fn artifact_and_dtype_are_bound() {
        let mut lifecycle = ready_to_release();
        assert_eq!(
            lifecycle.admit_dflash(ARTIFACT, [8; 32], [3; 32], true),
            Err(DFlashLifecycleError::ArtifactMismatch)
        );
        assert_eq!(
            lifecycle.admit_dflash(ARTIFACT, [2; 32], [3; 32], false),
            Err(DFlashLifecycleError::DTypeNotPermitted)
        );
    }

    #[test]
    fn budget_is_checked_after_prefill_release() {
        let mut lifecycle = DFlashLifecycle::new(DFlashLifecycleBudget {
            total_hbm_bytes: ARTIFACT.bytes + 1,
            safety_reserve_bytes: 2,
        });
        lifecycle.begin_prefill(REQUEST, 11, 20_000, 4_000).unwrap();
        let receipt = lifecycle.record_prefill_submission([4; 32]).unwrap();
        let terminal = proof(11, receipt.submission_epoch);
        lifecycle.mark_prefill_terminal(terminal).unwrap();
        lifecycle
            .release_prefill_after_completion(terminal)
            .unwrap();
        assert_eq!(
            lifecycle.admit_dflash(ARTIFACT, [2; 32], [3; 32], true),
            Err(DFlashLifecycleError::BudgetExceeded)
        );
    }

    #[test]
    fn poison_does_not_rearm() {
        let mut lifecycle = ready_to_release();
        lifecycle.poison();
        assert_eq!(
            lifecycle.mark_dflash_ready(),
            Err(DFlashLifecycleError::InvalidState)
        );
    }
}
