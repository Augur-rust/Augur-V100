//! Exact logical/physical contract for a growing resident history ring.
//!
//! The CUDA implementation owns the feature/KV payload; this module owns the
//! arithmetic that decides which logical positions are resident in the fixed
//! 2048-row allocation.  Keeping that decision CUDA-free gives the runtime a
//! deterministic admission point for short prompts, non-B8 tails, and the
//! first full-window rollover.  It deliberately never inserts padding or
//! synthetic tokens to make a request fit the draft block.

use core::fmt;

pub const GROWING_HISTORY_SCHEMA: &str = "qwen35-resident-growing-history-v1";
pub const DEFAULT_GROWING_HISTORY_CAPACITY: usize = 2048;
pub const DEFAULT_GROWING_HISTORY_BLOCK: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GrowingHistoryError {
    InvalidGeometry,
    InvalidTransition,
    EmptyPrompt,
    ZeroCommit,
    CommitExceedsBlock,
    PositionOverflow,
}

impl fmt::Display for GrowingHistoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid growing history operation: {self:?}")
    }
}

impl std::error::Error for GrowingHistoryError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GrowingHistoryConfig {
    pub capacity_rows: usize,
    pub block_rows: usize,
}

impl GrowingHistoryConfig {
    pub const fn default_2048() -> Self {
        Self {
            capacity_rows: DEFAULT_GROWING_HISTORY_CAPACITY,
            block_rows: DEFAULT_GROWING_HISTORY_BLOCK,
        }
    }

    pub fn validate(self) -> Result<(), GrowingHistoryError> {
        if self.capacity_rows == 0 || self.block_rows == 0 || self.block_rows > self.capacity_rows {
            return Err(GrowingHistoryError::InvalidGeometry);
        }
        Ok(())
    }
}

/// The physical copy plan for one append transaction.
///
/// `retained_rows` are copied from the old ring after an optional left shift;
/// `append_offset` is the first destination row for the newly committed
/// logical suffix.  The plan is valid for a short, growing, or full ring and
/// does not assume that `committed` is a multiple of the draft block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GrowingHistoryPlan {
    pub old_start_position: usize,
    pub old_active_rows: usize,
    pub new_start_position: usize,
    pub new_active_rows: usize,
    pub retained_rows: usize,
    pub evicted_rows: usize,
    pub shift_rows: usize,
    pub append_offset: usize,
    pub append_rows: usize,
    pub generation_before: u64,
    pub generation_after: u64,
}

impl GrowingHistoryPlan {
    pub fn new_start_row(self) -> usize {
        self.new_start_position
    }

    pub fn logical_position_to_row(self, position: usize) -> Option<usize> {
        let row = position.checked_sub(self.new_start_position)?;
        (row < self.new_active_rows).then_some(row)
    }

    /// Build one physical copy plan spanning several logical appends.
    ///
    /// The CUDA ring is updated one token or block at a time while the draft
    /// owner may be rebound only once after a user suffix.  This aggregate
    /// plan preserves the exact retained prefix and appends only the rows that
    /// are present in the final feature ring; callers reject transitions that
    /// exceed the physical capacity rather than inventing padding.
    pub fn between(
        old: GrowingHistory,
        new: GrowingHistory,
    ) -> Result<Self, GrowingHistoryError> {
        if old.config != new.config
            || old.active_rows == 0
            || new.active_rows == 0
            || old.active_rows > old.config.capacity_rows
            || new.active_rows > new.config.capacity_rows
            || new.cursor <= old.cursor
            || new.start_position < old.start_position
            || new.start_position > old.cursor
        {
            return Err(GrowingHistoryError::InvalidTransition);
        }
        let append_rows = new
            .cursor
            .checked_sub(old.cursor)
            .ok_or(GrowingHistoryError::InvalidTransition)?;
        if append_rows == 0 || append_rows > old.config.capacity_rows {
            return Err(GrowingHistoryError::InvalidTransition);
        }
        let retained_start = old.start_position.max(new.start_position);
        let retained_rows = old.cursor.saturating_sub(retained_start);
        let evicted_rows = new.start_position.saturating_sub(old.start_position);
        if retained_rows.checked_add(append_rows) != Some(new.active_rows)
            || evicted_rows.checked_add(retained_rows) != Some(old.active_rows)
            || new.generation <= old.generation
        {
            return Err(GrowingHistoryError::InvalidTransition);
        }
        Ok(Self {
            old_start_position: old.start_position,
            old_active_rows: old.active_rows,
            new_start_position: new.start_position,
            new_active_rows: new.active_rows,
            retained_rows,
            evicted_rows,
            shift_rows: evicted_rows,
            append_offset: retained_rows,
            append_rows,
            generation_before: old.generation,
            generation_after: new.generation,
        })
    }
}

/// Logical state for one request-local resident feature/KV ring.
///
/// `cursor` is the exclusive logical end.  `start_position..cursor` is the
/// exact active range; there is no implicit padding at either end.  The
/// physical allocation is always `capacity_rows`, but only `active_rows`
/// carry valid state until the ring fills.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GrowingHistory {
    config: GrowingHistoryConfig,
    start_position: usize,
    cursor: usize,
    active_rows: usize,
    generation: u64,
}

impl GrowingHistory {
    /// Restore only logical history after checkpoint admission. This counter is
    /// not a CUDA submission epoch and carries no device/owner authority.
    #[cfg(feature = "qwen35-session-checkpoint")]
    pub(crate) fn from_committed_extent(
        config: GrowingHistoryConfig,
        start_position: usize,
        active_rows: usize,
        generation: u64,
    ) -> Result<Self, GrowingHistoryError> {
        if generation == u64::MAX {
            return Err(GrowingHistoryError::PositionOverflow);
        }
        let mut history = Self::from_extent(config, start_position, active_rows)?;
        history.generation = generation;
        Ok(history)
    }

    pub fn from_extent(
        config: GrowingHistoryConfig,
        start_position: usize,
        active_rows: usize,
    ) -> Result<Self, GrowingHistoryError> {
        config.validate()?;
        if active_rows == 0 || active_rows > config.capacity_rows {
            return Err(GrowingHistoryError::InvalidGeometry);
        }
        let cursor = start_position
            .checked_add(active_rows)
            .ok_or(GrowingHistoryError::PositionOverflow)?;
        Ok(Self {
            config,
            start_position,
            cursor,
            active_rows,
            generation: 0,
        })
    }

    pub fn seed(
        config: GrowingHistoryConfig,
        prompt_tokens: usize,
    ) -> Result<Self, GrowingHistoryError> {
        config.validate()?;
        if prompt_tokens == 0 {
            return Err(GrowingHistoryError::EmptyPrompt);
        }
        let active_rows = prompt_tokens.min(config.capacity_rows);
        Ok(Self {
            config,
            start_position: prompt_tokens - active_rows,
            cursor: prompt_tokens,
            active_rows,
            generation: 0,
        })
    }

    pub const fn config(self) -> GrowingHistoryConfig {
        self.config
    }

    pub const fn start_position(self) -> usize {
        self.start_position
    }

    pub const fn cursor(self) -> usize {
        self.cursor
    }

    pub const fn active_rows(self) -> usize {
        self.active_rows
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }

    pub const fn capacity_rows(self) -> usize {
        self.config.capacity_rows
    }

    /// The draft block can run only when real history contains a full block.
    /// Short prompts remain valid state, but must use the ordinary prefill /
    /// non-draft admission path until enough real tokens exist.
    pub const fn draft_ready(self) -> bool {
        self.active_rows >= self.config.block_rows
    }

    pub fn logical_position_to_row(self, position: usize) -> Option<usize> {
        if position < self.start_position || position >= self.cursor {
            None
        } else {
            Some(position - self.start_position)
        }
    }

    pub fn append(
        &mut self,
        committed_rows: usize,
    ) -> Result<GrowingHistoryPlan, GrowingHistoryError> {
        if committed_rows == 0 {
            return Err(GrowingHistoryError::ZeroCommit);
        }
        if committed_rows > self.config.block_rows {
            return Err(GrowingHistoryError::CommitExceedsBlock);
        }
        let new_cursor = self
            .cursor
            .checked_add(committed_rows)
            .ok_or(GrowingHistoryError::PositionOverflow)?;
        // Absolute cursor is not a row count: an offset extent may still be
        // short. Only previously valid rows plus the committed suffix exist.
        let new_active_rows = self
            .active_rows
            .checked_add(committed_rows)
            .ok_or(GrowingHistoryError::PositionOverflow)?
            .min(self.config.capacity_rows);
        let new_start_position = new_cursor - new_active_rows;
        let old_end = self.cursor;
        let retained_start = self.start_position.max(new_start_position);
        let retained_end = old_end.min(new_cursor);
        let retained_rows = retained_end.saturating_sub(retained_start);
        let evicted_rows = new_start_position.saturating_sub(self.start_position);
        let shift_rows = new_start_position.saturating_sub(self.start_position);
        let append_offset = new_active_rows.saturating_sub(committed_rows);
        let generation_after = self
            .generation
            .checked_add(1)
            .ok_or(GrowingHistoryError::PositionOverflow)?;
        let plan = GrowingHistoryPlan {
            old_start_position: self.start_position,
            old_active_rows: self.active_rows,
            new_start_position,
            new_active_rows,
            retained_rows,
            evicted_rows,
            shift_rows,
            append_offset,
            append_rows: committed_rows,
            generation_before: self.generation,
            generation_after,
        };
        self.start_position = new_start_position;
        self.cursor = new_cursor;
        self.active_rows = new_active_rows;
        self.generation = generation_after;
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history(prompt: usize) -> GrowingHistory {
        GrowingHistory::seed(GrowingHistoryConfig::default_2048(), prompt).unwrap()
    }

    #[test]
    fn seed_never_invents_short_prompt_rows() {
        let h = history(3);
        assert_eq!(h.start_position(), 0);
        assert_eq!(h.cursor(), 3);
        assert_eq!(h.active_rows(), 3);
        assert!(!h.draft_ready());
        assert_eq!(h.logical_position_to_row(2), Some(2));
        assert_eq!(h.logical_position_to_row(3), None);
    }

    #[test]
    fn from_extent_preserves_exact_logical_range() {
        let h =
            GrowingHistory::from_extent(GrowingHistoryConfig::default_2048(), 6144, 2048).unwrap();
        assert_eq!(h.start_position(), 6144);
        assert_eq!(h.cursor(), 8192);
        assert_eq!(h.active_rows(), 2048);
        assert_eq!(h.generation(), 0);
        assert_eq!(h.logical_position_to_row(6144), Some(0));
        assert_eq!(h.logical_position_to_row(8191), Some(2047));
        assert_eq!(h.logical_position_to_row(8192), None);
    }

    #[test]
    fn from_extent_rejects_empty_or_overflowing_ranges() {
        let config = GrowingHistoryConfig::default_2048();
        assert_eq!(
            GrowingHistory::from_extent(config, 0, 0),
            Err(GrowingHistoryError::InvalidGeometry)
        );
        assert_eq!(
            GrowingHistory::from_extent(config, usize::MAX, 2),
            Err(GrowingHistoryError::PositionOverflow)
        );
    }

    #[test]
    fn seed_at_boundary_is_exact() {
        let short = history(2047);
        assert_eq!((short.start_position(), short.active_rows()), (0, 2047));
        let full = history(2048);
        assert_eq!((full.start_position(), full.active_rows()), (0, 2048));
        assert!(full.draft_ready());
        let long = history(2049);
        assert_eq!((long.start_position(), long.active_rows()), (1, 2048));
    }

    #[test]
    fn offset_short_extent_grows_without_inventing_prefix_rows() {
        let mut h =
            GrowingHistory::from_extent(GrowingHistoryConfig::default_2048(), 6144, 3).unwrap();
        let first = h.append(2).unwrap();
        assert_eq!(
            (h.start_position(), h.cursor(), h.active_rows()),
            (6144, 6149, 5)
        );
        assert_eq!(
            (first.retained_rows, first.evicted_rows, first.append_offset),
            (3, 0, 3)
        );
        let second = h.append(7).unwrap();
        assert_eq!(
            (h.start_position(), h.cursor(), h.active_rows()),
            (6144, 6156, 12)
        );
        assert_eq!(
            (
                second.retained_rows,
                second.evicted_rows,
                second.append_offset
            ),
            (5, 0, 5)
        );
        assert_eq!((second.generation_before, second.generation_after), (1, 2));
    }

    #[test]
    fn fill_then_rollover_keeps_exact_2047_2048_2049_positions() {
        let mut h = history(2047);
        let fill = h.append(1).unwrap();
        assert_eq!(
            (h.start_position(), h.cursor(), h.active_rows()),
            (0, 2048, 2048)
        );
        assert_eq!((fill.evicted_rows, fill.append_offset), (0, 2047));
        let roll = h.append(1).unwrap();
        assert_eq!(
            (h.start_position(), h.cursor(), h.active_rows()),
            (1, 2049, 2048)
        );
        assert_eq!((roll.evicted_rows, roll.append_offset), (1, 2047));
    }

    #[test]
    fn bounded_copy_plan_matches_logical_history_without_touching_physical_tail() {
        for start in [0, 17, 196000] {
            for active in [1, 3, 7, 8, 17, 2040, 2047, 2048] {
                for first_commit in 1..=8 {
                    let mut h = GrowingHistory::from_extent(
                        GrowingHistoryConfig::default_2048(),
                        start,
                        active,
                    )
                    .unwrap();
                    let mut physical = vec![usize::MAX; h.capacity_rows()];
                    let mut logical = (start..start + active).collect::<Vec<_>>();
                    physical[..active].copy_from_slice(&logical);
                    for committed in [first_commit, 9 - first_commit, 1, 7] {
                        let old_cursor = h.cursor();
                        let plan = h.append(committed).unwrap();
                        for dst in (0..plan.retained_rows).step_by(8) {
                            let rows = 8.min(plan.retained_rows - dst);
                            let scratch = physical
                                [plan.evicted_rows + dst..plan.evicted_rows + dst + rows]
                                .to_vec();
                            physical[dst..dst + rows].copy_from_slice(&scratch);
                        }
                        for row in 0..plan.append_rows {
                            physical[plan.append_offset + row] = old_cursor + row;
                        }
                        logical.extend(old_cursor..old_cursor + committed);
                        if logical.len() > h.capacity_rows() {
                            logical.drain(..logical.len() - h.capacity_rows());
                        }
                        assert_eq!(&physical[..h.active_rows()], logical.as_slice());
                        assert!(physical[h.active_rows()..].iter().all(|&x| x == usize::MAX));
                        assert_eq!(plan.retained_rows + plan.append_rows, h.active_rows());
                        assert_eq!(h.start_position() + h.active_rows(), h.cursor());
                    }
                }
            }
        }
    }

    #[test]
    fn overflow_append_leaves_the_prior_extent_unchanged() {
        let mut h =
            GrowingHistory::from_extent(GrowingHistoryConfig::default_2048(), usize::MAX - 3, 3)
                .unwrap();
        let prior = h;
        assert_eq!(h.append(1), Err(GrowingHistoryError::PositionOverflow));
        assert_eq!(h, prior);
        let mut h = history(3);
        h.generation = u64::MAX;
        let prior = h;
        assert_eq!(h.append(1), Err(GrowingHistoryError::PositionOverflow));
        assert_eq!(h, prior);
    }

    #[test]
    fn growing_append_keeps_non_block_tail_without_padding() {
        let mut h = history(2047);
        let plan = h.append(3).unwrap();
        assert_eq!(plan.old_active_rows, 2047);
        assert_eq!(plan.new_active_rows, 2048);
        assert_eq!(plan.retained_rows, 2045);
        assert_eq!(plan.evicted_rows, 2);
        assert_eq!(plan.shift_rows, 2);
        assert_eq!(plan.append_offset, 2045);
        assert_eq!(plan.append_rows, 3);
        assert_eq!(h.start_position(), 2);
        assert_eq!(h.cursor(), 2050);
        assert!(h.draft_ready());
    }

    #[test]
    fn full_ring_rolls_by_exact_committed_count() {
        let mut h = history(2048);
        let plan = h.append(1).unwrap();
        assert_eq!(plan.new_start_position, 1);
        assert_eq!(plan.retained_rows, 2047);
        assert_eq!(plan.evicted_rows, 1);
        assert_eq!(plan.append_offset, 2047);
        assert_eq!(plan.logical_position_to_row(1), Some(0));
        assert_eq!(plan.logical_position_to_row(2048), Some(2047));
        assert_eq!(plan.logical_position_to_row(0), None);
    }

    #[test]
    fn multi_row_rollover_and_generation_are_monotonic() {
        let mut h = history(4096);
        let first = h.append(7).unwrap();
        assert_eq!(first.generation_after, 1);
        assert_eq!(first.new_start_position, 2055);
        assert_eq!(first.retained_rows, 2041);
        let second = h.append(8).unwrap();
        assert_eq!(second.generation_before, 1);
        assert_eq!(second.generation_after, 2);
        assert_eq!(second.new_start_position, 2063);
        assert_eq!(second.append_offset, 2040);
    }

    #[test]
    fn aggregate_transition_matches_several_block_appends() {
        for (start, active, commits) in [
            (0, 3, [8usize, 1, 7]),
            (6144, 2047, [8usize, 8, 1]),
            (194560, 2048, [8usize, 8, 8]),
        ] {
            let mut h = GrowingHistory::from_extent(
                GrowingHistoryConfig::default_2048(),
                start,
                active,
            )
            .unwrap();
            let old = h;
            let expected_append = commits.into_iter().sum::<usize>();
            for committed in commits {
                h.append(committed).unwrap();
            }
            let plan = GrowingHistoryPlan::between(old, h).unwrap();
            assert_eq!(plan.append_rows, expected_append);
            assert_eq!(plan.generation_before, old.generation());
            assert_eq!(plan.generation_after, h.generation());
            assert_eq!(plan.new_start_position, h.start_position());
            assert_eq!(plan.new_active_rows, h.active_rows());
            assert_eq!(plan.retained_rows + plan.append_rows, h.active_rows());
            assert_eq!(plan.evicted_rows + plan.retained_rows, old.active_rows());
        }
    }

    #[test]
    fn aggregate_transition_rejects_crossing_more_than_one_physical_ring() {
        let old = GrowingHistory::from_extent(
            GrowingHistoryConfig::default_2048(),
            0,
            1,
        )
        .unwrap();
        let mut new = old;
        for _ in 0..257 {
            new.append(8).unwrap();
        }
        assert_eq!(
            GrowingHistoryPlan::between(old, new),
            Err(GrowingHistoryError::InvalidTransition)
        );
    }

    #[test]
    fn invalid_commits_fail_closed() {
        let mut h = history(8);
        assert_eq!(h.append(0), Err(GrowingHistoryError::ZeroCommit));
        assert_eq!(h.append(9), Err(GrowingHistoryError::CommitExceedsBlock));
        assert_eq!(h.generation(), 0);
    }

    #[cfg(feature = "qwen35-session-checkpoint")]
    #[test]
    fn restored_committed_extent_preserves_next_rollover() {
        for initial in [1, 7, 2047, 2048, 8192, 194560] {
            let mut original = history(initial);
            original.append(3).unwrap();
            let mut restored = GrowingHistory::from_committed_extent(
                original.config(),
                original.start_position(),
                original.active_rows(),
                original.generation(),
            )
            .unwrap();
            assert_eq!(restored, original);
            assert_eq!(restored.append(8).unwrap(), original.append(8).unwrap());
            assert_eq!(restored, original);
        }
    }

    #[cfg(feature = "qwen35-session-checkpoint")]
    #[test]
    fn restored_extent_rejects_exhausted_generation_and_overflow() {
        let config = GrowingHistoryConfig::default_2048();
        assert!(GrowingHistory::from_committed_extent(config, 0, 8, u64::MAX).is_err());
        assert!(GrowingHistory::from_committed_extent(config, usize::MAX, 8, 0).is_err());
        assert!(GrowingHistory::from_committed_extent(config, 0, 0, 0).is_err());
        assert!(GrowingHistory::from_committed_extent(config, 0, 2049, 0).is_err());
    }

    #[test]
    fn invalid_geometry_and_empty_prompt_fail_closed() {
        assert_eq!(
            GrowingHistoryConfig {
                capacity_rows: 0,
                block_rows: 8
            }
            .validate(),
            Err(GrowingHistoryError::InvalidGeometry)
        );
        assert_eq!(
            GrowingHistoryConfig {
                capacity_rows: 4,
                block_rows: 8
            }
            .validate(),
            Err(GrowingHistoryError::InvalidGeometry)
        );
        assert_eq!(
            GrowingHistory::seed(GrowingHistoryConfig::default_2048(), 0),
            Err(GrowingHistoryError::EmptyPrompt)
        );
    }
}
