//! Independent cold-prefix schedule. Draft/verify and initialized feature capture stay B8.
use super::*;

pub(super) const MAX_ROWS: usize = 512;

pub(super) fn validate_exact_input(rows: usize) -> Result<()> {
    if !(1..=MAX_ROWS).contains(&rows) { return Err(invalid("exact input rows must be1..512")); }
    Ok(())
}

pub(super) fn uses_input_ids(exact_input_owner: bool, rows: usize) -> bool {
    exact_input_owner || rows != B
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PrefixSchedule {
    pub wide_rows: usize,
    pub b8_tail_rows: usize,
    pub serial_rows: usize,
}

pub(super) fn schedule(width: usize, prefix: usize) -> Result<PrefixSchedule> {
    validate(width)?;
    let wide_rows = prefix / width * width;
    let remainder = prefix - wide_rows;
    Ok(PrefixSchedule {
        wide_rows,
        b8_tail_rows: remainder / B * B,
        serial_rows: remainder % B,
    })
}

pub(super) fn validate(rows: usize) -> Result<()> {
    if !matches!(rows, 8 | 16 | 32 | 64 | 128 | 256 | MAX_ROWS) {
        return Err(invalid(
            "canonical prefix rows must be 8, 16, 32, 64, 128, 256 or 512",
        ));
    }
    Ok(())
}

pub(in super::super) fn switch(value: Option<&std::ffi::OsStr>) -> Result<usize> {
    let rows = match value {
        None => B,
        Some(value) => match value.to_str() {
            Some("8") => 8,
            Some("16") => 16,
            Some("32") => 32,
            Some("64") => 64,
            Some("128") => 128,
            Some("256") => 256,
            Some("512") => MAX_ROWS,
            _ => return Err(invalid("invalid AUGUR_QWEN35_CANONICAL_PREFILL_ROWS")),
        },
    };
    validate(rows)?;
    Ok(rows)
}

pub(super) fn select(requested: usize, prefix_tokens: usize) -> Result<usize> {
    validate(requested)?;
    Ok(if prefix_tokens < requested {
        B
    } else {
        requested
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn exact_input_does_not_inherit_verifier_ids_or_legacy_admission() {
        for rows in 1..=MAX_ROWS {
            assert!(validate_exact_input(rows).is_ok());
            assert!(uses_input_ids(true, rows));
            assert_eq!(uses_input_ids(false, rows), rows != B);
        }
        for rows in [0, 513, usize::MAX] { assert!(validate_exact_input(rows).is_err()); }
        assert!(validate(9).is_err());
        assert!(!uses_input_ids(false, B));
        assert!(uses_input_ids(true, B));
    }

    #[test]
    fn default_prefix_does_not_expand_draft_or_verify() {
        assert_eq!(B, 8);
        assert_eq!(switch(None).unwrap(), B);
    }

    #[test]
    fn prefix_width_is_explicit_bounded_and_exactly_spelled() {
        for rows in [8, 16, 32, 64, 128, 256, 512] {
            assert_eq!(switch(Some(OsStr::new(&rows.to_string()))).unwrap(), rows);
            assert!(validate(rows).is_ok());
        }
        for value in [
            "", "1", "4", "08", "24", "768", "1024", " 32", "32 ", "true",
        ] {
            assert!(switch(Some(OsStr::new(value))).is_err());
        }
        for rows in [0, 1, 7, 9, 24, 129, 768, usize::MAX] {
            assert!(validate(rows).is_err());
        }
    }

    #[test]
    fn short_prefix_uses_explicit_b8_reference_without_padding() {
        for requested in [8, 16, 32, 64, 128, 256, 512] {
            for prefix in [0, 1, 7, requested - 1] {
                assert_eq!(select(requested, prefix).unwrap(), B);
            }
            assert_eq!(select(requested, requested).unwrap(), requested);
            assert_eq!(select(requested, requested + 1).unwrap(), requested);
        }
        assert!(select(1024, 8192).is_err());
    }

    #[test]
    fn every_width_preserves_real_prefix_with_only_a_sub_b8_scalar_tail() {
        for requested in [8, 16, 32, 64, 128, 256, 512] {
            for prefix in [
                0, 1, 7, 8, 9, 15, 16, 17, 2048, 6145, 8192, 31143, 65765, 193119, 196474,
            ] {
                let width = select(requested, prefix).unwrap();
                let schedule = schedule(width, prefix).unwrap();
                let tail_end = schedule.wide_rows + schedule.b8_tail_rows;
                assert!(schedule.wide_rows <= prefix);
                assert!(schedule.b8_tail_rows < width);
                assert!(schedule.serial_rows < B);
                assert_eq!(tail_end + schedule.serial_rows, prefix);
                let real_rows = (0..schedule.wide_rows)
                    .step_by(width)
                    .flat_map(|start| start..start + width)
                    .chain(
                        (schedule.wide_rows..tail_end)
                            .step_by(B)
                            .flat_map(|start| start..start + B),
                    )
                    .chain(tail_end..prefix)
                    .collect::<Vec<_>>();
                assert_eq!(real_rows, (0..prefix).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn wide_remainder_reuses_b8_and_never_pads_or_overflows() {
        assert_eq!(
            schedule(128, 7037).unwrap(),
            PrefixSchedule {
                wide_rows: 6912,
                b8_tail_rows: 120,
                serial_rows: 5,
            }
        );
        for width in [8, 16, 32, 64, 128, 256, 512] {
            let s = schedule(width, usize::MAX).unwrap();
            assert_eq!(s.wide_rows + s.b8_tail_rows + s.serial_rows, usize::MAX);
            assert!(s.serial_rows < B);
        }
        assert!(schedule(0, 7037).is_err());
        assert!(schedule(24, 7037).is_err());
    }
}
