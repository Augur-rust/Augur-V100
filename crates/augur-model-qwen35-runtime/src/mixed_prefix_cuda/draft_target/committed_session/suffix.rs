//! Opt-in known-input batching. No padding, acceptance, or publication.
use super::*;

pub(in super::super) fn switch(value: Option<&std::ffi::OsStr>) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" => Ok(true),
        _ => Err(invalid(
            "batched suffix requires AUGUR_QWEN35_BATCHED_SUFFIX=1",
        )),
    }
}

pub(in super::super) fn validate_scope(
    enabled: bool,
    growing: bool,
    canonical: bool,
    reuse: bool,
    profiled: bool,
) -> Result<()> {
    if enabled && (!growing || !canonical || !reuse || profiled) {
        return Err(invalid("batched suffix requires explicit growing session and unprofiled cached canonical executor"));
    }
    Ok(())
}

pub(in super::super) fn retain_plan(suffix_enabled: bool, requested: bool) -> bool {
    // The HTTP canonical profile historically used per-block binding. Suffix
    // opt-in explicitly retains those same operations at owner construction.
    suffix_enabled || requested
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Plan {
    pub batch_rows: usize,
    pub tail_rows: usize,
    pub end: usize,
}

impl Plan {
    pub fn new(enabled: bool, cursor: usize, rows: usize, capacity: usize) -> Result<Self> {
        let end = cursor
            .checked_add(rows)
            .ok_or_else(|| invalid("suffix extent overflow"))?;
        if rows == 0 || rows > 2049 || end.checked_add(B).is_none_or(|v| v > capacity) {
            return Err(invalid("suffix extent/reserved next-block admission"));
        }
        let batch_rows = if enabled { rows / B * B } else { 0 };
        Ok(Self {
            batch_rows,
            tail_rows: rows - batch_rows,
            end,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_switch_is_explicit_and_strict() {
        assert!(!switch(None).unwrap());
        assert!(switch(Some("1".as_ref())).unwrap());
        for value in ["", "0", "true", " 1", "1 "] {
            assert!(switch(Some(value.as_ref())).is_err());
        }
    }

    #[test]
    fn suffix_scope_rejects_fixed_shadow_uncached_and_profiled_execution() {
        for mask in 0..16 {
            let growing = mask & 1 != 0;
            let canonical = mask & 2 != 0;
            let reuse = mask & 4 != 0;
            let profiled = mask & 8 != 0;
            assert!(validate_scope(false, growing, canonical, reuse, profiled).is_ok());
            assert_eq!(
                validate_scope(true, growing, canonical, reuse, profiled).is_ok(),
                growing && canonical && reuse && !profiled
            );
        }
    }

    #[test]
    fn suffix_opt_in_constructs_retained_plan_without_changing_default() {
        assert!(!retain_plan(false, false));
        for flags in [(false, true), (true, false), (true, true)] {
            assert!(retain_plan(flags.0, flags.1));
        }
        assert!(validate_scope(true, true, true, retain_plan(true, false), false).is_ok());
        assert!(validate_scope(true, true, false, retain_plan(true, false), false).is_err());
        assert!(validate_scope(true, true, true, retain_plan(true, false), true).is_err());
    }

    #[test]
    fn every_admitted_extent_has_only_real_full_blocks_and_tail() {
        for rows in 1..=2049 {
            for enabled in [false, true] {
                let p = Plan::new(enabled, 2047, rows, 10000).unwrap();
                assert_eq!(p.end, 2047 + rows);
                assert_eq!(p.batch_rows + p.tail_rows, rows);
                assert_eq!(p.batch_rows % B, 0);
                if enabled {
                    assert!(p.tail_rows < B);
                } else {
                    assert_eq!(p.batch_rows, 0);
                }
            }
        }
    }

    #[test]
    fn rejects_zero_overflow_oversize_and_missing_next_block() {
        for (cursor, rows, capacity) in [
            (0, 0, 4096),
            (0, 2050, 4096),
            (usize::MAX, 1, usize::MAX),
            (usize::MAX - 8, 1, usize::MAX),
            (2048, 8, 2063),
        ] {
            assert!(Plan::new(true, cursor, rows, capacity).is_err());
        }
        assert!(Plan::new(true, 2048, 8, 2064).is_ok());
    }

    #[test]
    fn growing_window_matches_scalar_around_physical_boundary() {
        // Model real feature IDs to expose a lost/duplicated row at rollover.
        for start in [1, 7, 8, 2040, 2047, 2048, 2049, 4095] {
            for rows in [1, 7, 8, 9, 16, 2047, 2048, 2049] {
                let p = Plan::new(true, start, rows, 10000).unwrap();
                let mut history =
                    GrowingHistory::seed(GrowingHistoryConfig::default_2048(), start).unwrap();
                let mut active = (start.saturating_sub(2048)..start).collect::<Vec<_>>();
                let mut cursor = start;
                let groups = std::iter::repeat_n(B, p.batch_rows / B)
                    .chain(std::iter::repeat_n(1, p.tail_rows));
                for count in groups {
                    let advance = history.append(count).unwrap();
                    active.drain(..advance.evicted_rows);
                    active.extend(cursor..cursor + count);
                    cursor += count;
                    assert_eq!(
                        active,
                        (cursor.saturating_sub(2048)..cursor).collect::<Vec<_>>()
                    );
                }
                assert_eq!(cursor, p.end);
                assert_eq!(history.cursor(), p.end);
            }
        }
    }
}
