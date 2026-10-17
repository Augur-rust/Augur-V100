//! Bounded DFlash hot-window and 256K KV memory contract.
//!
//! This module is deliberately CUDA-free.  It is the control plane shared by
//! the real H2D/compute adapter and the 256K active-HBM/RAM-cold target path.
//! A slot cannot be reused until its dependent consumer has reached a terminal
//! proof, and a request cannot be finished while any slot is still owned.

use crate::DFlashRequestIdentity;
use core::fmt;

pub const DFLASH_HOT_WINDOW_SCHEMA: &str = "qwen35-dflash-hot-window-v1";
pub const QWEN35_KV8_CACHE_ROW_BYTES: usize = 1_088;
pub const QWEN35_KV8_FULL_ATTENTION_PLANES: usize = 32;
pub const QWEN35_KV8_PAGE_POSITIONS: usize = 512;
pub const QWEN35_KV8_ACTIVE_WINDOW_POSITIONS: usize = 2_048;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlashHotWindowMemoryPlan {
    pub total_hbm_bytes: u64,
    pub safety_reserve_bytes: u64,
    pub model_bytes: u64,
    pub dflash_bytes: u64,
    pub kv_capacity_positions: usize,
    pub kv_row_bytes: usize,
    pub kv_planes: usize,
    pub full_kv_bytes: u64,
    pub active_window_positions: usize,
    pub active_kv_bytes: u64,
    pub slot_count: usize,
    pub slot_tile_bytes: u64,
    pub slot_storage_bytes: u64,
    pub full_resident_bytes: u64,
    pub active_resident_bytes: u64,
    pub full_kv_fits: bool,
    pub active_window_fits: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DFlashHotWindowMemoryError {
    InvalidGeometry,
    Overflow,
}

impl fmt::Display for DFlashHotWindowMemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid DFlash hot-window memory plan: {self:?}")
    }
}

impl std::error::Error for DFlashHotWindowMemoryError {}

impl DFlashHotWindowMemoryPlan {
    /// Bind the exact memory accounting used by the target KV8 adapter.
    ///
    /// `model_bytes` and `dflash_bytes` are caller-supplied identity-bound
    /// resident allocations.  The plan reports both an intentionally unsafe
    /// full-KV shape and the active-window shape; it never silently changes a
    /// requested capacity to make the budget pass.
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        total_hbm_bytes: u64,
        safety_reserve_bytes: u64,
        model_bytes: u64,
        dflash_bytes: u64,
        kv_capacity_positions: usize,
        active_window_positions: usize,
        slot_count: usize,
        slot_tile_bytes: usize,
    ) -> Result<Self, DFlashHotWindowMemoryError> {
        if total_hbm_bytes == 0
            || model_bytes == 0
            || dflash_bytes == 0
            || kv_capacity_positions == 0
            || active_window_positions == 0
            || active_window_positions > kv_capacity_positions
            || !kv_capacity_positions.is_multiple_of(QWEN35_KV8_PAGE_POSITIONS)
            || !active_window_positions.is_multiple_of(QWEN35_KV8_PAGE_POSITIONS)
            || slot_count == 0
            || slot_tile_bytes == 0
        {
            return Err(DFlashHotWindowMemoryError::InvalidGeometry);
        }
        let row_bytes = u64::try_from(QWEN35_KV8_CACHE_ROW_BYTES)
            .map_err(|_| DFlashHotWindowMemoryError::Overflow)?;
        let planes = u64::try_from(QWEN35_KV8_FULL_ATTENTION_PLANES)
            .map_err(|_| DFlashHotWindowMemoryError::Overflow)?;
        let capacity = u64::try_from(kv_capacity_positions)
            .map_err(|_| DFlashHotWindowMemoryError::Overflow)?;
        let active = u64::try_from(active_window_positions)
            .map_err(|_| DFlashHotWindowMemoryError::Overflow)?;
        let full_kv_bytes = capacity
            .checked_mul(row_bytes)
            .and_then(|bytes| bytes.checked_mul(planes))
            .ok_or(DFlashHotWindowMemoryError::Overflow)?;
        let active_kv_bytes = active
            .checked_mul(row_bytes)
            .and_then(|bytes| bytes.checked_mul(planes))
            .ok_or(DFlashHotWindowMemoryError::Overflow)?;
        let slot_storage_bytes = u64::try_from(slot_count)
            .ok()
            .and_then(|count| {
                u64::try_from(slot_tile_bytes)
                    .ok()
                    .and_then(|tile| count.checked_mul(tile))
            })
            .ok_or(DFlashHotWindowMemoryError::Overflow)?;
        let base = model_bytes
            .checked_add(dflash_bytes)
            .and_then(|bytes| bytes.checked_add(slot_storage_bytes))
            .and_then(|bytes| bytes.checked_add(safety_reserve_bytes))
            .ok_or(DFlashHotWindowMemoryError::Overflow)?;
        let full_resident_bytes = base
            .checked_add(full_kv_bytes)
            .ok_or(DFlashHotWindowMemoryError::Overflow)?;
        let active_resident_bytes = base
            .checked_add(active_kv_bytes)
            .ok_or(DFlashHotWindowMemoryError::Overflow)?;
        Ok(Self {
            total_hbm_bytes,
            safety_reserve_bytes,
            model_bytes,
            dflash_bytes,
            kv_capacity_positions,
            kv_row_bytes: QWEN35_KV8_CACHE_ROW_BYTES,
            kv_planes: QWEN35_KV8_FULL_ATTENTION_PLANES,
            full_kv_bytes,
            active_window_positions,
            active_kv_bytes,
            slot_count,
            slot_tile_bytes: u64::try_from(slot_tile_bytes)
                .map_err(|_| DFlashHotWindowMemoryError::Overflow)?,
            slot_storage_bytes,
            full_resident_bytes,
            active_resident_bytes,
            full_kv_fits: full_resident_bytes <= total_hbm_bytes,
            active_window_fits: active_resident_bytes <= total_hbm_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SlotState {
    Empty,
    PrefetchSubmitted,
    Ready,
    ComputeActive,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SlotRecord {
    state: SlotState,
    generation: u64,
    lease: Option<DFlashHotWindowLease>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlashHotWindowConfig {
    pub capacity_positions: usize,
    pub active_window_positions: usize,
    pub slot_count: usize,
    pub tile_bytes: usize,
}

impl Default for DFlashHotWindowConfig {
    fn default() -> Self {
        Self {
            capacity_positions: 262_144,
            active_window_positions: QWEN35_KV8_ACTIVE_WINDOW_POSITIONS,
            slot_count: 2,
            tile_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlashHotWindowLease {
    pub owner_generation: u64,
    pub request_sha256: [u8; 32],
    pub request_generation: u64,
    pub slot_index: usize,
    pub slot_generation: u64,
    pub step: usize,
    pub source_offset: usize,
    pub valid_bytes: usize,
    pub submission_epoch: u64,
    pub submission_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlashHotWindowTerminal {
    pub lease: DFlashHotWindowLease,
    pub terminal_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DFlashHotWindowError {
    InvalidConfig,
    InvalidState,
    InvalidRequest,
    InvalidLease,
    SlotBusy,
    StepWindowExceeded,
    OutOfOrderTerminal,
    InvalidExtent,
    InvalidEpoch,
    Poisoned,
}

impl fmt::Display for DFlashHotWindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid DFlash hot-window state: {self:?}")
    }
}

impl std::error::Error for DFlashHotWindowError {}

/// Proof-carrying bounded slot ring for one sequential DFlash request.
#[derive(Clone, Debug)]
pub struct DFlashHotWindow {
    config: DFlashHotWindowConfig,
    request: Option<DFlashRequestIdentity>,
    owner_generation: u64,
    next_step: usize,
    next_terminal_step: usize,
    next_slot_generation: u64,
    slots: Vec<SlotRecord>,
    poisoned: bool,
}

impl DFlashHotWindow {
    pub fn new(config: DFlashHotWindowConfig) -> Result<Self, DFlashHotWindowError> {
        if config.capacity_positions == 0
            || config.active_window_positions == 0
            || config.active_window_positions > config.capacity_positions
            || !config
                .capacity_positions
                .is_multiple_of(QWEN35_KV8_PAGE_POSITIONS)
            || !config
                .active_window_positions
                .is_multiple_of(QWEN35_KV8_PAGE_POSITIONS)
            || config.slot_count == 0
            || config.tile_bytes == 0
        {
            return Err(DFlashHotWindowError::InvalidConfig);
        }
        Ok(Self {
            config,
            request: None,
            owner_generation: 0,
            next_step: 0,
            next_terminal_step: 0,
            next_slot_generation: 0,
            slots: vec![
                SlotRecord {
                    state: SlotState::Empty,
                    generation: 0,
                    lease: None,
                };
                config.slot_count
            ],
            poisoned: false,
        })
    }

    pub const fn config(&self) -> DFlashHotWindowConfig {
        self.config
    }

    pub const fn poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn begin_request(
        &mut self,
        request: DFlashRequestIdentity,
        owner_generation: u64,
    ) -> Result<(), DFlashHotWindowError> {
        if self.poisoned
            || self.request.is_some()
            || request.request_sha256 == [0; 32]
            || request.request_generation == 0
            || owner_generation == 0
        {
            return Err(if self.poisoned {
                DFlashHotWindowError::Poisoned
            } else {
                DFlashHotWindowError::InvalidRequest
            });
        }
        self.request = Some(request);
        self.owner_generation = owner_generation;
        self.next_step = 0;
        self.next_terminal_step = 0;
        self.next_slot_generation = 0;
        for slot in &mut self.slots {
            *slot = SlotRecord {
                state: SlotState::Empty,
                generation: 0,
                lease: None,
            };
        }
        Ok(())
    }

    fn check_lease(&self, lease: DFlashHotWindowLease) -> Result<usize, DFlashHotWindowError> {
        if self.poisoned {
            return Err(DFlashHotWindowError::Poisoned);
        }
        let request = self.request.ok_or(DFlashHotWindowError::InvalidState)?;
        if lease.owner_generation != self.owner_generation
            || lease.request_sha256 != request.request_sha256
            || lease.request_generation != request.request_generation
            || lease.slot_index >= self.slots.len()
        {
            return Err(DFlashHotWindowError::InvalidLease);
        }
        let slot = &self.slots[lease.slot_index];
        if slot.generation != lease.slot_generation || slot.lease != Some(lease) {
            return Err(DFlashHotWindowError::InvalidLease);
        }
        Ok(lease.slot_index)
    }

    pub fn submit_prefetch(
        &mut self,
        slot_index: usize,
        source_offset: usize,
        valid_bytes: usize,
        submission_epoch: u64,
        submission_digest: [u8; 32],
    ) -> Result<DFlashHotWindowLease, DFlashHotWindowError> {
        if self.poisoned {
            return Err(DFlashHotWindowError::Poisoned);
        }
        let request = self.request.ok_or(DFlashHotWindowError::InvalidState)?;
        if slot_index >= self.slots.len() {
            return Err(DFlashHotWindowError::InvalidLease);
        }
        if self.next_step.saturating_sub(self.next_terminal_step) >= self.config.slot_count {
            return Err(DFlashHotWindowError::StepWindowExceeded);
        }
        if valid_bytes == 0 || valid_bytes > self.config.tile_bytes {
            return Err(DFlashHotWindowError::InvalidExtent);
        }
        if submission_epoch == 0 || submission_digest == [0; 32] {
            return Err(DFlashHotWindowError::InvalidEpoch);
        }
        let slot = &mut self.slots[slot_index];
        if slot.state != SlotState::Empty {
            return Err(DFlashHotWindowError::SlotBusy);
        }
        self.next_slot_generation = self
            .next_slot_generation
            .checked_add(1)
            .ok_or(DFlashHotWindowError::InvalidEpoch)?;
        let lease = DFlashHotWindowLease {
            owner_generation: self.owner_generation,
            request_sha256: request.request_sha256,
            request_generation: request.request_generation,
            slot_index,
            slot_generation: self.next_slot_generation,
            step: self.next_step,
            source_offset,
            valid_bytes,
            submission_epoch,
            submission_digest,
        };
        slot.state = SlotState::PrefetchSubmitted;
        slot.generation = lease.slot_generation;
        slot.lease = Some(lease);
        self.next_step = self
            .next_step
            .checked_add(1)
            .ok_or(DFlashHotWindowError::InvalidEpoch)?;
        Ok(lease)
    }

    pub fn mark_ready(&mut self, lease: DFlashHotWindowLease) -> Result<(), DFlashHotWindowError> {
        let slot_index = self.check_lease(lease)?;
        let slot = &mut self.slots[slot_index];
        if slot.state != SlotState::PrefetchSubmitted {
            return Err(DFlashHotWindowError::InvalidState);
        }
        slot.state = SlotState::Ready;
        Ok(())
    }

    pub fn begin_compute(
        &mut self,
        lease: DFlashHotWindowLease,
    ) -> Result<(), DFlashHotWindowError> {
        let slot_index = self.check_lease(lease)?;
        let slot = &mut self.slots[slot_index];
        if slot.state != SlotState::Ready {
            return Err(DFlashHotWindowError::InvalidState);
        }
        slot.state = SlotState::ComputeActive;
        Ok(())
    }

    pub fn mark_terminal(
        &mut self,
        lease: DFlashHotWindowLease,
        terminal_digest: [u8; 32],
    ) -> Result<DFlashHotWindowTerminal, DFlashHotWindowError> {
        let slot_index = self.check_lease(lease)?;
        if lease.step != self.next_terminal_step {
            return Err(DFlashHotWindowError::OutOfOrderTerminal);
        }
        if terminal_digest == [0; 32] {
            return Err(DFlashHotWindowError::InvalidEpoch);
        }
        let slot = &mut self.slots[slot_index];
        if slot.state != SlotState::ComputeActive {
            return Err(DFlashHotWindowError::InvalidState);
        }
        slot.state = SlotState::Empty;
        slot.lease = None;
        self.next_terminal_step = self
            .next_terminal_step
            .checked_add(1)
            .ok_or(DFlashHotWindowError::InvalidEpoch)?;
        Ok(DFlashHotWindowTerminal {
            lease,
            terminal_digest,
        })
    }

    pub fn inflight_slots(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.state != SlotState::Empty)
            .count()
    }

    pub fn finish_request(&mut self) -> Result<(), DFlashHotWindowError> {
        if self.poisoned {
            return Err(DFlashHotWindowError::Poisoned);
        }
        if self.request.is_none() {
            return Err(DFlashHotWindowError::InvalidState);
        }
        if self.inflight_slots() != 0 || self.next_terminal_step != self.next_step {
            return Err(DFlashHotWindowError::InvalidState);
        }
        self.request = None;
        Ok(())
    }

    pub fn poison(&mut self) {
        self.poisoned = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUEST: DFlashRequestIdentity = DFlashRequestIdentity {
        request_sha256: [1; 32],
        request_generation: 3,
    };

    #[test]
    fn memory_plan_rejects_full_256k_but_accepts_active_window() {
        let plan = DFlashHotWindowMemoryPlan::bind(
            32 * 1024 * 1024 * 1024,
            256 * 1024 * 1024,
            23_172_467_776,
            2_056_415_552,
            262_144,
            2_048,
            2,
            64 * 1024 * 1024,
        )
        .unwrap();
        assert_eq!(plan.full_kv_bytes, 9_126_805_504);
        assert!(!plan.full_kv_fits);
        assert!(plan.active_window_fits);
        assert!(plan.active_resident_bytes < plan.total_hbm_bytes);
    }

    #[test]
    fn lookahead_and_terminal_reuse_are_generation_bound() {
        let mut window = DFlashHotWindow::new(DFlashHotWindowConfig {
            capacity_positions: 2_048,
            active_window_positions: 2_048,
            slot_count: 2,
            tile_bytes: 64,
        })
        .unwrap();
        window.begin_request(REQUEST, 11).unwrap();
        let first = window.submit_prefetch(0, 0, 32, 1, [2; 32]).unwrap();
        let second = window.submit_prefetch(1, 16, 32, 2, [3; 32]).unwrap();
        assert_eq!(window.inflight_slots(), 2);
        assert_eq!(
            window.submit_prefetch(0, 32, 32, 3, [4; 32]),
            Err(DFlashHotWindowError::StepWindowExceeded)
        );
        window.mark_ready(first).unwrap();
        window.begin_compute(first).unwrap();
        assert_eq!(
            window.mark_terminal(second, [5; 32]),
            Err(DFlashHotWindowError::OutOfOrderTerminal)
        );
        window.mark_terminal(first, [6; 32]).unwrap();
        let reused = window.submit_prefetch(0, 64, 32, 3, [7; 32]).unwrap();
        assert!(reused.slot_generation > first.slot_generation);
        assert_eq!(
            window.mark_ready(first),
            Err(DFlashHotWindowError::InvalidLease)
        );
        window.mark_ready(second).unwrap();
        window.begin_compute(second).unwrap();
        window.mark_terminal(second, [8; 32]).unwrap();
        window.mark_ready(reused).unwrap();
        window.begin_compute(reused).unwrap();
        window.mark_terminal(reused, [9; 32]).unwrap();
        window.finish_request().unwrap();
    }

    #[test]
    fn bad_extent_epoch_and_poison_fail_closed() {
        let mut window = DFlashHotWindow::new(DFlashHotWindowConfig::default()).unwrap();
        window.begin_request(REQUEST, 11).unwrap();
        assert_eq!(
            window.submit_prefetch(0, 0, 0, 1, [2; 32]),
            Err(DFlashHotWindowError::InvalidExtent)
        );
        assert_eq!(
            window.submit_prefetch(0, 0, 1, 0, [2; 32]),
            Err(DFlashHotWindowError::InvalidEpoch)
        );
        window.poison();
        assert_eq!(
            window.submit_prefetch(0, 0, 1, 1, [2; 32]),
            Err(DFlashHotWindowError::Poisoned)
        );
    }
}
