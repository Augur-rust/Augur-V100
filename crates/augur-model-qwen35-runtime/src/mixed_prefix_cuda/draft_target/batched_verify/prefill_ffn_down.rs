//! Default-off numerical FFN-down experiment. Never selects a decode/history op.
use super::*;

pub(in super::super) fn requested(
    value: Option<&std::ffi::OsStr>,
    prefix_rows: Option<&std::ffi::OsStr>,
) -> Result<Option<usize>> {
    match value {
        None => Ok(None),
        Some(v) if v == "1" => {
            let rows = prefill_width::switch(prefix_rows)?;
            if !matches!(rows, 64 | 128) {
                return Err(invalid(
                    "FFN-down Tensor experiment requires prefix rows 64 or 128",
                ));
            }
            Ok(Some(rows))
        }
        _ => Err(invalid(
            "AUGUR_QWEN35_PREFILL_FFN_DOWN_TENSOR must be exactly 1",
        )),
    }
}

pub(super) fn selected_count(operations: &[PrefillLayerOps]) -> usize {
    operations.iter().filter(|op| op.ffn().legacy_opt().is_some_and(|p| p.ffn_down.tensor_ffn_down_enabled())).count()
}

pub(in super::super) fn requested_compensated(
    value: Option<&std::ffi::OsStr>,
    rows: Option<usize>,
) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" && rows == Some(128) => Ok(true),
        _ => Err(invalid(
            "compensated FFN-down requires explicit Tensor=1 and B128",
        )),
    }
}

pub(super) fn install(
    operations: &mut [PrefillLayerOps],
    rows: usize,
    compensated: bool,
) -> Result<()> {
    if operations.len() != 64 || !matches!(rows, 64 | 128) {
        return Err(invalid(
            "FFN-down Tensor experiment needs all64 real B64/B128 layers",
        ));
    }
    if compensated && rows != 128 {
        return Err(invalid("compensated FFN-down only measured B128"));
    }
    // Validate the complete selection before mutating any local operation.
    for op in operations.iter() {
        let projection = match op {
            PrefillLayerOps::Linear(p) => &p.ffn.legacy()?.ffn_down,
            PrefillLayerOps::Full(p) => &p.ffn.legacy()?.ffn_down,
            
        };
        if projection.raw().tokens as usize != rows || projection.tensor_ffn_down_tile_rows() != 896
        {
            return Err(invalid("FFN-down Tensor shape coverage"));
        }
    }
    for op in operations.iter_mut() {
        let projection = match op {
            PrefillLayerOps::Linear(p) => &mut p.ffn.legacy_mut()?.ffn_down,
            PrefillLayerOps::Full(p) => &mut p.ffn.legacy_mut()?.ffn_down,
            
        };
        if compensated {
            if projection.tensor_ffn_down_compensated_tile_rows() == 896 {
                projection.enable_tensor_ffn_down_compensated_audit()?;
            }
        } else {
            projection.enable_tensor_ffn_down_audit()?;
        }
    }
    let selected = selected_count(operations);
    if (!compensated && selected != 64) || (compensated && !(1..=64).contains(&selected)) {
        return Err(invalid("FFN-down Tensor selection count"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn down_tensor_is_default_off_and_only_accepts_two_measured_widths() {
        assert_eq!(requested(None, None).unwrap(), None);
        for n in ["64", "128"] {
            assert_eq!(
                requested(Some(OsStr::new("1")), Some(OsStr::new(n))).unwrap(),
                Some(n.parse().unwrap())
            );
        }
        for n in ["8", "16", "32", "256", "512", "064", "64 ", "1024"] {
            assert!(requested(Some(OsStr::new("1")), Some(OsStr::new(n))).is_err());
        }
        assert!(requested(Some(OsStr::new("1")), None).is_err());
        for v in ["", "0", "true", "on", " 1", "1 "] {
            assert!(requested(Some(OsStr::new(v)), Some(OsStr::new("128"))).is_err());
        }
    }

    #[test]
    fn down_tensor_does_not_silently_accept_missing_layers() {
        assert_eq!(selected_count(&[]), 0);
        for rows in [8, 64, 128, 512] {
            assert!(install(&mut [], rows, false).is_err());
            assert!(install(&mut [], rows, true).is_err());
        }
    }

    #[test]
    fn compensated_down_requires_an_explicit_b128_parent_policy() {
        assert!(!requested_compensated(None, None).unwrap());
        assert!(requested_compensated(Some(OsStr::new("1")), Some(128)).unwrap());
        for rows in [None, Some(8), Some(64), Some(256)] {
            assert!(requested_compensated(Some(OsStr::new("1")), rows).is_err());
        }
        for value in ["", "0", "true", "on", " 1", "1 "] {
            assert!(requested_compensated(Some(OsStr::new(value)), Some(128)).is_err());
        }
    }
}
