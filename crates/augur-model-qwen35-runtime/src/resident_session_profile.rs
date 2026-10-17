//! CPU-only admission shared by the native owner and HTTP transport.
//! Capacity is not initialized history and admission is not qualification.

use std::ops::RangeInclusive;

pub const CAPACITY: usize = 196_608;
pub const DRAFT_HISTORY: usize = 2_048;
pub const INITIAL_OUTPUT_LIMIT: usize = 1_024;
pub const CONTINUATION_OUTPUT_LIMIT: usize = 128;
/// Opt-in resident request ceiling. Legacy/default continuations remain 128.
pub const RESIDENT_CONTINUATION_MAX_OUTPUT: usize = 1_024;
pub const VERIFY_WIDTH: usize = 8;
pub const INITIAL_RESERVE: usize = INITIAL_OUTPUT_LIMIT + VERIFY_WIDTH - 2;
pub const CONTINUATION_RESERVE: usize = CONTINUATION_OUTPUT_LIMIT + VERIFY_WIDTH - 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentSessionProfile {
    /// Variable input length, initialized history, full physical KV capacity.
    Resident,
    Growing,
    Canonical8k,
    Context32k,
    Context64k,
    Near192k,
}

impl ResidentSessionProfile {
    pub fn parse(value: &str) -> Result<Self, &'static str> {
        match value {
            "resident" => Ok(Self::Resident),
            "growing" => Ok(Self::Growing),
            "8k" | "canonical8k" => Ok(Self::Canonical8k),
            "32k" | "context32" => Ok(Self::Context32k),
            "64k" | "context64" => Ok(Self::Context64k),
            "near192k" | "near192" => Ok(Self::Near192k),
            _ => Err("unknown resident session profile"),
        }
    }

    pub fn from_legacy(
        context32: bool,
        context64: bool,
        near192: bool,
        growing: bool,
    ) -> Result<Self, &'static str> {
        if [context32, context64, near192, growing]
            .into_iter()
            .filter(|&b| b)
            .count()
            > 1
        {
            return Err("session selectors are mutually exclusive");
        }
        Ok(if context32 {
            Self::Context32k
        } else if context64 {
            Self::Context64k
        } else if near192 {
            Self::Near192k
        } else if growing {
            Self::Growing
        } else {
            Self::Canonical8k
        })
    }

    pub fn validate_compiled(self) -> Result<(), &'static str> {
        let unavailable = (matches!(self, Self::Context32k)
            && !cfg!(feature = "qwen35-session-context32-audit"))
            || (matches!(self, Self::Context64k)
                && !cfg!(feature = "qwen35-session-context64-audit"))
            || (matches!(self, Self::Near192k) && !cfg!(feature = "qwen35-session-near192-audit"));
        if !unavailable {
            Ok(())
        } else {
            Err("selected session profile not compiled")
        }
    }

    pub fn producer_flag(self) -> &'static str {
        match self {
            Self::Resident => "--resident-session-stdin",
            Self::Growing => "--resident-session-growing-stdin",
            Self::Canonical8k => "--resident-session-canonical-stdin",
            Self::Context32k => "--resident-session-context32-stdin",
            Self::Context64k => "--resident-session-context64-stdin",
            Self::Near192k => "--resident-session-near192-stdin",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::Growing => "growing",
            Self::Canonical8k => "canonical8k",
            Self::Context32k => "context32k",
            Self::Context64k => "context64k",
            Self::Near192k => "near192k",
        }
    }

    pub fn schema(self) -> &'static str {
        match self {
            Self::Resident => "qwen35-resident-text-session-service-v1",
            Self::Growing => "qwen35-initialized-growing-session-service-v1",
            Self::Canonical8k => "qwen35-canonical8k-session-service-v1",
            Self::Context32k => "qwen35-canonical32k-session-service-v1",
            Self::Context64k => "qwen35-canonical64k-session-service-v1",
            Self::Near192k => "qwen35-canonicalnear192-session-service-v1",
        }
    }

    pub fn uses_initialized_history(self) -> bool {
        matches!(self, Self::Resident | Self::Growing)
    }

    pub fn initial_prompt_tokens(self) -> Option<usize> {
        match self {
            Self::Resident | Self::Growing => None,
            Self::Canonical8k => Some(8192),
            Self::Context32k => Some(32768),
            Self::Context64k => Some(65536),
            Self::Near192k => Some(194560),
        }
    }

    pub fn maximum_initial_prompt_tokens(self) -> usize {
        match self {
            Self::Resident => CAPACITY - INITIAL_RESERVE,
            Self::Growing => 8192,
            _ => self.initial_prompt_tokens().expect("fixed session profile"),
        }
    }

    pub fn admits_initial(self, tokens: usize) -> bool {
        if let Some(fixed) = self.initial_prompt_tokens() {
            tokens == fixed
        } else {
            (1..=self.maximum_initial_prompt_tokens()).contains(&tokens)
        }
    }

    pub fn maximum_history_tokens(self) -> usize {
        match self {
            Self::Resident | Self::Near192k => CAPACITY,
            Self::Growing | Self::Canonical8k => 16384,
            Self::Context32k => 36864,
            Self::Context64k => 69632,
        }
    }

    pub fn history_bounds(self) -> RangeInclusive<usize> {
        self.initial_prompt_tokens().unwrap_or(1)..=self.maximum_history_tokens()
    }

    /// Reserve the pending token, suffix and the entire verification tail.
    pub fn append_end(self, consumed: usize, suffix: usize) -> Option<usize> {
        self.append_end_with_budget(consumed, suffix, CONTINUATION_OUTPUT_LIMIT)
    }

    pub fn continuation_reserve(self, budget: usize) -> Option<usize> {
        self.output_budget(Some(budget), false)
            .ok()
            .or_else(|| (budget == CONTINUATION_OUTPUT_LIMIT).then_some(budget))?;
        budget
            .max(CONTINUATION_OUTPUT_LIMIT)
            .checked_add(VERIFY_WIDTH - 2)
    }

    /// Check requested output and the complete B8 verification tail before
    /// appending any input or mutating retained CUDA state.
    pub fn append_end_with_budget(
        self,
        consumed: usize,
        suffix: usize,
        budget: usize,
    ) -> Option<usize> {
        if !self.history_bounds().contains(&consumed) || !(1..=DRAFT_HISTORY).contains(&suffix) {
            return None;
        }
        consumed
            .checked_add(1)?
            .checked_add(suffix)?
            .checked_add(self.continuation_reserve(budget)?)
            .filter(|&end| end <= self.maximum_history_tokens())
    }

    /// Explicit resident requests can use a larger publication ceiling without
    /// changing legacy/default budgets or the physical KV allocation.
    /// Legacy diagnostic profiles retain their exact default budgets.
    pub fn output_budget(
        self,
        requested: Option<usize>,
        initial: bool,
    ) -> Result<usize, &'static str> {
        let default = if initial {
            INITIAL_OUTPUT_LIMIT
        } else {
            CONTINUATION_OUTPUT_LIMIT
        };
        let maximum = if !initial && self == Self::Resident {
            RESIDENT_CONTINUATION_MAX_OUTPUT
        } else {
            default
        };
        match requested {
            None => Ok(default),
            Some(value) if self == Self::Resident && (1..=maximum).contains(&value) => Ok(value),
            Some(_) => {
                Err("request output budget requires resident profile and bounded positive tokens")
            }
        }
    }

    pub fn capabilities(self) -> serde_json::Value {
        serde_json::json!({"profile":self.name(), "capacity":CAPACITY,
            "minimum_initial_prompt_tokens":self.initial_prompt_tokens().unwrap_or(1),
            "maximum_initial_prompt_tokens":self.maximum_initial_prompt_tokens(),
            "maximum_history_tokens":self.maximum_history_tokens(),
            "draft_history_capacity":DRAFT_HISTORY, "initialized_history":self.uses_initialized_history(),
            "request_output_budget_supported":self == Self::Resident,
            "initial_output_limit":INITIAL_OUTPUT_LIMIT, "continuation_output_limit":CONTINUATION_OUTPUT_LIMIT,
            "initial_max_output_limit":INITIAL_OUTPUT_LIMIT,
            "continuation_max_output_limit":if self == Self::Resident {RESIDENT_CONTINUATION_MAX_OUTPUT} else {CONTINUATION_OUTPUT_LIMIT},
            "initial_reserved_positions":INITIAL_RESERVE, "continuation_reserved_positions":CONTINUATION_RESERVE})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_output_budget_is_explicit_bounded_and_does_not_shrink_reserve() {
        let p = ResidentSessionProfile::Resident;
        for (initial, maximum) in [
            (true, INITIAL_OUTPUT_LIMIT),
            (false, CONTINUATION_OUTPUT_LIMIT),
        ] {
            assert_eq!(p.output_budget(None, initial), Ok(maximum));
            for n in [1, 2, 7, 8, 9, maximum] {
                assert_eq!(p.output_budget(Some(n), initial), Ok(n));
            }
            let requested_maximum = if initial {
                maximum
            } else {
                RESIDENT_CONTINUATION_MAX_OUTPUT
            };
            for n in [0, requested_maximum + 1, usize::MAX] {
                assert!(p.output_budget(Some(n), initial).is_err());
            }
            for old in [
                ResidentSessionProfile::Growing,
                ResidentSessionProfile::Canonical8k,
                ResidentSessionProfile::Context32k,
                ResidentSessionProfile::Context64k,
                ResidentSessionProfile::Near192k,
            ] {
                assert_eq!(old.output_budget(None, initial), Ok(maximum));
                assert!(old.output_budget(Some(maximum), initial).is_err());
            }
        }
        assert_eq!(
            p.maximum_initial_prompt_tokens(),
            CAPACITY - INITIAL_RESERVE
        );
        assert_eq!(
            p.append_end(4096, 1),
            Some(4096 + 1 + 1 + CONTINUATION_RESERVE)
        );
    }
    #[test]
    fn extended_continuation_binds_requested_budget_and_capacity_before_submit() {
        let p = ResidentSessionProfile::Resident;
        for budget in [129, 256, 512, RESIDENT_CONTINUATION_MAX_OUTPUT] {
            assert_eq!(p.output_budget(Some(budget), false), Ok(budget));
            let reserve = budget + VERIFY_WIDTH - 2;
            assert_eq!(p.continuation_reserve(budget), Some(reserve));
            let consumed = CAPACITY - 1 - 17 - reserve;
            assert_eq!(
                p.append_end_with_budget(consumed, 17, budget),
                Some(CAPACITY)
            );
            assert_eq!(p.append_end_with_budget(consumed + 1, 17, budget), None);
        }
        assert_eq!(p.output_budget(None, false), Ok(128));
        assert_eq!(p.continuation_reserve(1), Some(134));
        for budget in [0, RESIDENT_CONTINUATION_MAX_OUTPUT + 1, usize::MAX] {
            assert_eq!(p.append_end_with_budget(4096, 1, budget), None);
        }
        assert_eq!(
            ResidentSessionProfile::Near192k.append_end_with_budget(194560, 1, 256),
            None
        );
    }
    #[test]
    fn resident_accepts_variable_lengths_without_removing_output_reserve() {
        let p = ResidentSessionProfile::Resident;
        for n in [
            1,
            7,
            2047,
            2048,
            8193,
            17001,
            33003,
            65537,
            194560,
            CAPACITY - INITIAL_RESERVE,
        ] {
            assert!(p.admits_initial(n));
        }
        for n in [0, CAPACITY - INITIAL_RESERVE + 1, CAPACITY, usize::MAX] {
            assert!(!p.admits_initial(n));
        }
    }
    #[test]
    fn append_checks_the_pending_token_and_full_verification_tail() {
        let p = ResidentSessionProfile::Resident;
        let consumed = CAPACITY - 1 - 17 - CONTINUATION_RESERVE;
        assert_eq!(p.append_end(consumed, 17), Some(CAPACITY));
        assert_eq!(p.append_end(consumed + 1, 17), None);
        for (n, suffix) in [(0, 1), (1, 0), (1, 2049), (usize::MAX, 1), (1, usize::MAX)] {
            assert_eq!(p.append_end(n, suffix), None);
        }
    }
    #[test]
    fn historical_input_profiles_do_not_silently_expand() {
        assert!(ResidentSessionProfile::Growing.admits_initial(8192));
        assert!(!ResidentSessionProfile::Growing.admits_initial(8193));
        assert_eq!(
            ResidentSessionProfile::Growing.maximum_history_tokens(),
            16384
        );
        for p in [
            ResidentSessionProfile::Canonical8k,
            ResidentSessionProfile::Context32k,
            ResidentSessionProfile::Context64k,
            ResidentSessionProfile::Near192k,
        ] {
            let n = p.initial_prompt_tokens().unwrap();
            assert!(p.admits_initial(n));
            assert!(!p.admits_initial(n - 1));
            assert!(!p.admits_initial(n + 1));
        }
    }
    #[test]
    fn selectors_and_identity_are_unambiguous() {
        assert!(ResidentSessionProfile::from_legacy(true, false, false, true).is_err());
        assert_ne!(
            ResidentSessionProfile::Resident.schema(),
            ResidentSessionProfile::Growing.schema()
        );
        assert_eq!(
            ResidentSessionProfile::parse("resident"),
            Ok(ResidentSessionProfile::Resident)
        );
        assert!(ResidentSessionProfile::Resident.uses_initialized_history());
    }
}
