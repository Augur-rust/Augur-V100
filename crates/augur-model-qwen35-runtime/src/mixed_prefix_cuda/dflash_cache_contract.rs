//! Component-only cache range planning and independent F64 arithmetic oracle.
use std::ops::Range;

pub(super) fn next_window(
    last_step: Option<usize>,
    last_position: Option<usize>,
    valid: usize,
    capacity: usize,
    step: usize,
    position: usize,
) -> Result<Range<usize>, String> {
    if capacity == 0 || capacity > 2048 || valid > capacity {
        return Err("DFlash cache invalid capacity/valid count".into());
    }
    match (last_step, last_position) {
        (None, None) if valid == 0 && step == 0 => {}
        (Some(s), Some(p))
            if valid > 0
                && s.checked_add(1) == Some(step)
                && p.checked_add(1) == Some(position) => {}
        _ => return Err("DFlash cache stale/missing/noncontiguous step or position".into()),
    }
    let end = position
        .checked_add(1)
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or("DFlash cache absolute position overflow")?;
    let count = (valid + 1).min(capacity);
    let start = end
        .checked_sub(count)
        .ok_or("DFlash cache range underflow")?;
    Ok(start..end)
}

/// Inputs are the actual query and *only* valid K/V rows in logical order.
/// This checks attention arithmetic, not the correctness of the encoder.
pub(super) fn attention_f64(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
) -> Result<Vec<f32>, String> {
    const ROW: usize = 8 * 128;
    if query.len() != 32 * 128
        || keys.is_empty()
        || keys.len() % ROW != 0
        || keys.len() / ROW > 2048
        || keys.len() != values.len()
        || query
            .iter()
            .chain(keys)
            .chain(values)
            .any(|x| !x.is_finite())
    {
        return Err("DFlash attention oracle invalid shape/nonfinite input".into());
    }
    let rows = keys.len() / ROW;
    let scale = (1.0_f32 / 128.0_f32.sqrt()) as f64;
    let mut out = vec![0.0; query.len()];
    for qh in 0..32 {
        let kh = qh / 4;
        let scores: Vec<f64> = (0..rows)
            .map(|r| {
                (0..128)
                    .map(|d| query[qh * 128 + d] as f64 * keys[r * ROW + kh * 128 + d] as f64)
                    .sum::<f64>()
                    * scale
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f64 = weights.iter().sum();
        for d in 0..128 {
            out[qh * 128 + d] = (weights
                .iter()
                .enumerate()
                .map(|(r, w)| w * values[r * ROW + kh * 128 + d] as f64)
                .sum::<f64>()
                / sum) as f32;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn late_start_does_not_imply_prefill_history() {
        assert_eq!(
            next_window(None, None, 0, 2048, 0, 196607).unwrap(),
            196607..196608
        );
    }
    #[test]
    fn window_wrap_evicts_only_oldest_row() {
        let (mut step, mut pos, mut valid) = (None, None, 0);
        for s in 0..12 {
            let range = next_window(step, pos, valid, 4, s, 3 + s).unwrap();
            assert_eq!(range.len(), (s + 1).min(4));
            assert_eq!(range.end, 4 + s);
            let slots: std::collections::BTreeSet<_> = range.clone().map(|p| p % 4).collect();
            assert_eq!(slots.len(), range.len());
            valid = range.len();
            step = Some(s);
            pos = Some(3 + s);
        }
    }
    #[test]
    fn stale_missing_partial_and_overflow_reject() {
        for (s, p, v, c, ns, np) in [
            (Some(0), Some(3), 1, 4, 0, 4),
            (Some(0), Some(3), 1, 4, 1, 3),
            (Some(0), Some(3), 1, 4, 1, 5),
            (None, Some(3), 1, 4, 0, 4),
            (None, None, 1, 4, 0, 4),
            (None, None, 0, 2049, 0, 4),
            (None, None, 0, 4, 0, u32::MAX as usize),
        ] {
            assert!(next_window(s, p, v, c, ns, np).is_err());
        }
    }
    #[test]
    fn oracle_uniform_multirow_and_nonfinite_rejection() {
        let query = vec![0.0; 4096];
        let keys = vec![0.5; 2048];
        let values: Vec<_> = (0..2048)
            .map(|i| if i < 1024 { 2.0 } else { 6.0 })
            .collect();
        assert!(attention_f64(&query, &keys, &values)
            .unwrap()
            .iter()
            .all(|&v| v == 4.0));
        assert!(attention_f64(&query[..4095], &keys, &values).is_err());
        let mut bad = keys;
        bad[0] = f32::NAN;
        assert!(attention_f64(&query, &bad, &values).is_err());
    }
}
