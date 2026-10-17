//! Isolated measured FFN candidate. Integer products, canonical scale/reduce.
//! B8 verifier/history/tails and non-FFN projections stay native.
use super::*;

pub(in super::super) fn requested(
    value: Option<&std::ffi::OsStr>,
    prefix_rows: Option<&std::ffi::OsStr>,
    canonical_prefix: bool,
    other_tensor_policy: bool,
) -> Result<Option<usize>> {
    match value {
        None => Ok(None),
        Some(v) if v == "1" && canonical_prefix && !other_tensor_policy => {
            let rows = prefill_width::switch(prefix_rows)?;
            if ![128, 256, 512].contains(&rows) {
                return Err(invalid("integer MMA FFN requires explicit measured B128/B256/B512 prefix"));
            }
            Ok(Some(rows))
        }
        _ => Err(invalid(
            "AUGUR_QWEN35_PREFILL_INTEGER_MMA requires exactly 1, canonical prefix and no other Tensor policy",
        )),
    }
}

pub(super) fn selected_count(operations: &[PrefillLayerOps]) -> usize {
    operations.iter().filter_map(|op| op.ffn().legacy_opt()).map(|p| [&p.ffn_gate, &p.ffn_up, &p.ffn_down].into_iter().filter(|p| p.integer_mma_ffn_enabled()).count()).sum()
}

pub(super) fn install(operations: &mut [PrefillLayerOps], rows: usize) -> Result<()> {
    if operations.len() != 64 || ![128, 256, 512].contains(&rows) {
        return Err(invalid("integer MMA needs complete measured prefix layers"));
    }
    for op in operations.iter_mut() {
        let projections = match op {
            PrefillLayerOps::Linear(p) => { let ffn = p.ffn.legacy_mut()?; [&mut ffn.ffn_gate, &mut ffn.ffn_up, &mut ffn.ffn_down] },
            PrefillLayerOps::Full(p) => { let ffn = p.ffn.legacy_mut()?; [&mut ffn.ffn_gate, &mut ffn.ffn_up, &mut ffn.ffn_down] },
            
        };
        for projection in projections {
            if projection.raw().tokens as usize != rows {
                return Err(invalid("integer MMA bound token extent"));
            }
            if projection.integer_mma_ffn_scratch_bytes() != 0 {
                projection.enable_integer_mma_ffn_audit()?;
            }
        }
    }
    if selected_count(operations) == 0 {
        return Err(invalid("integer MMA did not select a measured NVFP4 FFN"));
    }
    Ok(())
}

pub(super) fn arithmetic_policy(rows: usize) -> &'static str {
    if rows == 128 {
        "canonical_nvfp4_ffn_b128_integer_mma_tile384_v1"
    } else {
        "canonical_nvfp4_q8_ffn_wide_integer_mma_tile384_v1"
    }
}

pub(super) fn scratch_bytes(rows: usize) -> usize {
    match rows {
        128 => 20_054_016,
        256 => 27_852_800,
        512 => 38_993_920,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn integer_mma_is_default_off_and_requires_exact_independent_b128_policy() {
        assert_eq!(requested(None, None, false, true).unwrap(), None);
        for rows in [128, 256, 512] {
            assert_eq!(
                requested(
                    Some(OsStr::new("1")),
                    Some(OsStr::new(&rows.to_string())),
                    true,
                    false
                )
                .unwrap(),
                Some(rows)
            );
            assert!(scratch_bytes(rows) > 0);
        }
        for rows in [
            None,
            Some("8"),
            Some("64"),
            Some("129"),
            Some("1024"),
            Some("0128"),
            Some("128 "),
        ] {
            assert!(requested(Some(OsStr::new("1")), rows.map(OsStr::new), true, false).is_err());
        }
        for value in ["", "0", "true", "on", " 1", "1 "] {
            assert!(requested(
                Some(OsStr::new(value)),
                Some(OsStr::new("128")),
                true,
                false
            )
            .is_err());
        }
        for (canonical, other) in [(false, false), (true, true), (false, true)] {
            assert!(requested(
                Some(OsStr::new("1")),
                Some(OsStr::new("128")),
                canonical,
                other
            )
            .is_err());
        }
    }

    #[test]
    fn incomplete_layers_never_select_integer_mma() {
        assert_eq!(selected_count(&[]), 0);
        for rows in [8, 64, 128, 256] {
            assert!(install(&mut [], rows).is_err());
        }
    }
}
