//! CPU-only identity and lifetime contract for a future DFlash CUDA Graph.
//!
//! This module intentionally does not call CUDA and is not wired into the
//! current one-shot `DFlashQ8CudaBlock`.  It freezes the safety boundary that
//! a later graph owner must satisfy before any capture/replay experiment.

use core::fmt;

pub const DFLASH_GRAPH_BATCH_TOKENS: usize = 8;
pub const DFLASH_GRAPH_MAX_CONTEXT_ROWS: usize = 2048;
pub const DFLASH_GRAPH_MAX_POSITION: usize = 262_144;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DFlashGraphContractError {
    InvalidShape,
    InvalidIdentity,
    AlreadyCaptured,
    NotCaptured,
    Closed,
    Poisoned,
    IdentityMismatch,
    SubmissionEpochRegression,
    TerminalProofTooEarly,
}

impl fmt::Display for DFlashGraphContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::InvalidShape => "invalid graph shape",
            Self::InvalidIdentity => "invalid graph identity",
            Self::AlreadyCaptured => "graph identity already captured",
            Self::NotCaptured => "graph identity not captured",
            Self::Closed => "graph contract is closed",
            Self::Poisoned => "graph contract is poisoned",
            Self::IdentityMismatch => "graph replay identity mismatch",
            Self::SubmissionEpochRegression => "graph submission epoch regressed",
            Self::TerminalProofTooEarly => "terminal proof does not cover graph replay",
        };
        formatter.write_str(text)
    }
}

impl std::error::Error for DFlashGraphContractError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlashGraphShape {
    pub context_rows: usize,
    pub absolute_start_position: usize,
    pub sliding_window: usize,
}

impl DFlashGraphShape {
    pub fn new(
        context_rows: usize,
        absolute_start_position: usize,
        sliding_window: usize,
    ) -> Result<Self, DFlashGraphContractError> {
        if !(1..=DFLASH_GRAPH_MAX_CONTEXT_ROWS).contains(&context_rows)
            || !(1..=DFLASH_GRAPH_MAX_CONTEXT_ROWS).contains(&sliding_window)
            || absolute_start_position
                .checked_add(DFLASH_GRAPH_BATCH_TOKENS)
                .is_none_or(|end| end > DFLASH_GRAPH_MAX_POSITION)
        {
            return Err(DFlashGraphContractError::InvalidShape);
        }
        Ok(Self {
            context_rows,
            absolute_start_position,
            sliding_window,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlashGraphIdentity {
    pub device_ordinal: i32,
    pub stream_owner_instance_id: u64,
    pub generation: u64,
    pub buffer_layout_sha256: [u8; 32],
    pub shape: DFlashGraphShape,
}

impl DFlashGraphIdentity {
    pub fn new(
        device_ordinal: i32,
        stream_owner_instance_id: u64,
        generation: u64,
        buffer_layout_sha256: [u8; 32],
        shape: DFlashGraphShape,
    ) -> Result<Self, DFlashGraphContractError> {
        if device_ordinal < 0 || stream_owner_instance_id == 0 || buffer_layout_sha256 == [0; 32] {
            return Err(DFlashGraphContractError::InvalidIdentity);
        }
        Ok(Self {
            device_ordinal,
            stream_owner_instance_id,
            generation,
            buffer_layout_sha256,
            shape,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlashGraphReplayReceipt {
    pub identity: DFlashGraphIdentity,
    pub submission_epoch: u64,
    pub replay_sequence: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DFlashGraphContract {
    captured: Option<DFlashGraphIdentity>,
    last_submission_epoch: Option<u64>,
    replay_count: u64,
    poisoned: bool,
    closed: bool,
}

impl DFlashGraphContract {
    pub const fn new() -> Self {
        Self {
            captured: None,
            last_submission_epoch: None,
            replay_count: 0,
            poisoned: false,
            closed: false,
        }
    }

    pub fn capture(
        &mut self,
        identity: DFlashGraphIdentity,
    ) -> Result<(), DFlashGraphContractError> {
        self.ensure_open()?;
        if self.captured.is_some() {
            return Err(DFlashGraphContractError::AlreadyCaptured);
        }
        self.captured = Some(identity);
        self.last_submission_epoch = None;
        self.replay_count = 0;
        Ok(())
    }

    pub fn validate_replay(
        &self,
        identity: DFlashGraphIdentity,
    ) -> Result<(), DFlashGraphContractError> {
        self.ensure_open()?;
        if self.captured != Some(identity) {
            return Err(DFlashGraphContractError::IdentityMismatch);
        }
        Ok(())
    }

    pub fn replay(
        &mut self,
        identity: DFlashGraphIdentity,
        submission_epoch: u64,
    ) -> Result<DFlashGraphReplayReceipt, DFlashGraphContractError> {
        if let Err(error) = self.validate_replay(identity) {
            self.poisoned = true;
            return Err(error);
        }
        if self
            .last_submission_epoch
            .is_some_and(|previous| submission_epoch <= previous)
        {
            self.poisoned = true;
            return Err(DFlashGraphContractError::SubmissionEpochRegression);
        }
        self.replay_count = self
            .replay_count
            .checked_add(1)
            .ok_or(DFlashGraphContractError::SubmissionEpochRegression)?;
        self.last_submission_epoch = Some(submission_epoch);
        Ok(DFlashGraphReplayReceipt {
            identity,
            submission_epoch,
            replay_sequence: self.replay_count,
        })
    }

    pub fn complete(
        &self,
        receipt: DFlashGraphReplayReceipt,
        terminal_epoch: u64,
    ) -> Result<(), DFlashGraphContractError> {
        self.ensure_open()?;
        if self.captured != Some(receipt.identity) {
            return Err(DFlashGraphContractError::IdentityMismatch);
        }
        if terminal_epoch < receipt.submission_epoch {
            return Err(DFlashGraphContractError::TerminalProofTooEarly);
        }
        Ok(())
    }

    pub fn fail_closed(&mut self) {
        self.poisoned = true;
    }

    pub fn close(&mut self) -> Result<(), DFlashGraphContractError> {
        self.ensure_open()?;
        self.closed = true;
        self.captured = None;
        Ok(())
    }

    pub const fn captured_identity(&self) -> Option<DFlashGraphIdentity> {
        self.captured
    }

    pub const fn replay_count(&self) -> u64 {
        self.replay_count
    }

    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    fn ensure_open(&self) -> Result<(), DFlashGraphContractError> {
        if self.closed {
            Err(DFlashGraphContractError::Closed)
        } else if self.poisoned {
            Err(DFlashGraphContractError::Poisoned)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(start: usize, generation: u64) -> DFlashGraphIdentity {
        DFlashGraphIdentity::new(
            0,
            77,
            generation,
            [9; 32],
            DFlashGraphShape::new(2048, start, 2048).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn shape_freezes_eight_token_position_and_window() {
        assert!(DFlashGraphShape::new(1, 0, 1).is_ok());
        assert!(DFlashGraphShape::new(2048, DFLASH_GRAPH_MAX_POSITION - 8, 2048).is_ok());
        assert!(DFlashGraphShape::new(0, 0, 1).is_err());
        assert!(DFlashGraphShape::new(1, DFLASH_GRAPH_MAX_POSITION - 7, 1).is_err());
    }

    #[test]
    fn identity_rejects_foreign_device_owner_and_missing_layout_digest() {
        let shape = DFlashGraphShape::new(8, 100, 8).unwrap();
        assert!(DFlashGraphIdentity::new(-1, 1, 0, [1; 32], shape).is_err());
        assert!(DFlashGraphIdentity::new(0, 0, 0, [1; 32], shape).is_err());
        assert!(DFlashGraphIdentity::new(0, 1, 0, [0; 32], shape).is_err());
    }

    #[test]
    fn exact_identity_replays_with_monotonic_epochs_and_terminal_proof() {
        let mut contract = DFlashGraphContract::new();
        let id = identity(196000, 4);
        contract.capture(id).unwrap();
        let receipt = contract.replay(id, 12).unwrap();
        assert_eq!(receipt.replay_sequence, 1);
        assert_eq!(contract.replay_count(), 1);
        assert!(contract.complete(receipt, 12).is_ok());
        assert!(contract.complete(receipt, 11).is_err());
        let second = contract.replay(id, 13).unwrap();
        assert_eq!(second.replay_sequence, 2);
    }

    #[test]
    fn generation_position_owner_and_layout_drift_fail_closed() {
        let mut contract = DFlashGraphContract::new();
        let id = identity(100, 2);
        contract.capture(id).unwrap();
        for drift in [identity(100, 3), identity(108, 2)] {
            assert_eq!(
                contract.replay(drift, 1),
                Err(DFlashGraphContractError::IdentityMismatch)
            );
            assert!(contract.is_poisoned());
            break;
        }
        let mut fresh = DFlashGraphContract::new();
        fresh.capture(id).unwrap();
        let mut changed = id;
        changed.stream_owner_instance_id += 1;
        assert_eq!(
            fresh.validate_replay(changed),
            Err(DFlashGraphContractError::IdentityMismatch)
        );
        let mut changed_layout = id;
        changed_layout.buffer_layout_sha256 = [8; 32];
        assert_eq!(
            fresh.validate_replay(changed_layout),
            Err(DFlashGraphContractError::IdentityMismatch)
        );
    }

    #[test]
    fn epoch_regression_poison_and_close_are_irreversible() {
        let mut contract = DFlashGraphContract::new();
        let id = identity(10, 0);
        contract.capture(id).unwrap();
        contract.replay(id, 9).unwrap();
        assert_eq!(
            contract.replay(id, 9),
            Err(DFlashGraphContractError::SubmissionEpochRegression)
        );
        assert_eq!(
            contract.replay(id, 10),
            Err(DFlashGraphContractError::Poisoned)
        );

        let mut closed = DFlashGraphContract::new();
        closed.capture(id).unwrap();
        closed.close().unwrap();
        assert_eq!(closed.replay(id, 1), Err(DFlashGraphContractError::Closed));
        assert_eq!(closed.close(), Err(DFlashGraphContractError::Closed));
    }

    #[test]
    fn capture_is_singleton_until_close() {
        let mut contract = DFlashGraphContract::new();
        let id = identity(10, 0);
        contract.capture(id).unwrap();
        assert_eq!(
            contract.capture(id),
            Err(DFlashGraphContractError::AlreadyCaptured)
        );
        assert_eq!(contract.captured_identity(), Some(id));
    }

    #[test]
    fn explicit_failure_poison_blocks_all_future_actions() {
        let mut contract = DFlashGraphContract::new();
        let id = identity(10, 0);
        contract.capture(id).unwrap();
        contract.fail_closed();
        assert_eq!(
            contract.validate_replay(id),
            Err(DFlashGraphContractError::Poisoned)
        );
        assert_eq!(contract.close(), Err(DFlashGraphContractError::Poisoned));
    }
}
