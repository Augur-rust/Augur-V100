//! CPU-only segmented attention oracle for the Qwen35 KV8 audit path.
//!
//! This module intentionally does not own CUDA buffers or model state.  It
//! computes the same partial `(max, denominator, weighted numerator)` values
//! exposed by the V2 segment ABI and merges them with a stable log-sum-exp
//! recurrence.  It is therefore useful for proving page partitioning and
//! absolute-position geometry before wiring the path into a full CUDA layer.
//! Nothing in this module is a product or TPS claim.

use core::fmt;

/// Dense GQA geometry; each oracle entry point still validates its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionGeometry {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SegmentedAttentionOracleError {
    EmptyQuery,
    EmptyCache,
    ShapeMismatch,
    InvalidRange,
    NonFiniteInput,
    InvalidPartial,
}

impl fmt::Display for SegmentedAttentionOracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::EmptyQuery => "empty query",
            Self::EmptyCache => "empty cache",
            Self::ShapeMismatch => "attention shape mismatch",
            Self::InvalidRange => "invalid attention segment range",
            Self::NonFiniteInput => "non-finite attention input",
            Self::InvalidPartial => "invalid attention partial",
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttentionPartial {
    pub maximum: Vec<f32>,
    pub denominator: Vec<f32>,
    pub weighted_numerator: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SegmentedAttentionResult {
    pub output: Vec<f32>,
    pub partial: AttentionPartial,
}

fn validate_shapes(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<usize, SegmentedAttentionOracleError> {
    if query.is_empty() {
        return Err(SegmentedAttentionOracleError::EmptyQuery);
    }
    if keys.is_empty() || values.is_empty() {
        return Err(SegmentedAttentionOracleError::EmptyCache);
    }
    if query_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || query.len() != query_heads * head_dim
        || keys.len() != values.len()
        || !keys.len().is_multiple_of(kv_heads * head_dim)
        || !query_heads.is_multiple_of(kv_heads)
    {
        return Err(SegmentedAttentionOracleError::ShapeMismatch);
    }
    if query
        .iter()
        .chain(keys.iter())
        .chain(values.iter())
        .any(|value| !value.is_finite())
    {
        return Err(SegmentedAttentionOracleError::NonFiniteInput);
    }
    Ok(keys.len() / (kv_heads * head_dim))
}

/// Compute one V2-compatible partial over the absolute range
/// `[segment_begin, segment_end)`, clipped by `visible_position_end`.
pub fn partial_for_segment(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    geometry: AttentionGeometry,
    segment_begin: usize,
    segment_end: usize,
    visible_position_end: usize,
) -> Result<AttentionPartial, SegmentedAttentionOracleError> {
    let AttentionGeometry {
        query_heads,
        kv_heads,
        head_dim,
    } = geometry;
    let positions = validate_shapes(query, keys, values, query_heads, kv_heads, head_dim)?;
    if segment_begin >= segment_end
        || segment_end > positions
        || visible_position_end == 0
        || visible_position_end > positions
    {
        return Err(SegmentedAttentionOracleError::InvalidRange);
    }
    let end = segment_end.min(visible_position_end);
    let group = query_heads / kv_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut maximum = vec![f32::NEG_INFINITY; query_heads];
    let mut denominator = vec![0.0; query_heads];
    let mut weighted_numerator = vec![0.0; query_heads * head_dim];
    if segment_begin >= end {
        return Ok(AttentionPartial {
            maximum,
            denominator,
            weighted_numerator,
        });
    }
    for query_head in 0..query_heads {
        let kv_head = query_head / group;
        let query_row = &query[query_head * head_dim..(query_head + 1) * head_dim];
        let mut max_score = f32::NEG_INFINITY;
        for position in segment_begin..end {
            let base = (position * kv_heads + kv_head) * head_dim;
            let score = query_row
                .iter()
                .zip(&keys[base..base + head_dim])
                .map(|(q, k)| q * k)
                .sum::<f32>()
                * scale;
            max_score = max_score.max(score);
        }
        let mut denom = 0.0;
        let out_base = query_head * head_dim;
        for position in segment_begin..end {
            let base = (position * kv_heads + kv_head) * head_dim;
            let score = query_row
                .iter()
                .zip(&keys[base..base + head_dim])
                .map(|(q, k)| q * k)
                .sum::<f32>()
                * scale;
            let weight = (score - max_score).exp();
            denom += weight;
            for component in 0..head_dim {
                weighted_numerator[out_base + component] += weight * values[base + component];
            }
        }
        if !denom.is_finite() || denom <= 0.0 {
            return Err(SegmentedAttentionOracleError::InvalidPartial);
        }
        maximum[query_head] = max_score;
        denominator[query_head] = denom;
    }
    Ok(AttentionPartial {
        maximum,
        denominator,
        weighted_numerator,
    })
}

/// Stable merge for two partials.  The merge is associative up to floating
/// point rounding and deliberately mirrors the V2 LSE contract.
pub fn merge_partials(
    left: &AttentionPartial,
    right: &AttentionPartial,
    head_dim: usize,
) -> Result<AttentionPartial, SegmentedAttentionOracleError> {
    if head_dim == 0
        || left.maximum.len() != left.denominator.len()
        || right.maximum.len() != right.denominator.len()
        || left.maximum.len() != right.maximum.len()
        || left.weighted_numerator.len() != left.maximum.len() * head_dim
        || right.weighted_numerator.len() != right.maximum.len() * head_dim
    {
        return Err(SegmentedAttentionOracleError::ShapeMismatch);
    }
    let mut maximum = vec![f32::NEG_INFINITY; left.maximum.len()];
    let mut denominator = vec![0.0; left.maximum.len()];
    let mut weighted_numerator = vec![0.0; left.weighted_numerator.len()];
    for row in 0..maximum.len() {
        let lm = left.maximum[row];
        let rm = right.maximum[row];
        if (!lm.is_finite() && lm != f32::NEG_INFINITY)
            || (!rm.is_finite() && rm != f32::NEG_INFINITY)
            || !left.denominator[row].is_finite()
            || !right.denominator[row].is_finite()
            || left.denominator[row] < 0.0
            || right.denominator[row] < 0.0
        {
            return Err(SegmentedAttentionOracleError::InvalidPartial);
        }
        let merged_max = lm.max(rm);
        if merged_max == f32::NEG_INFINITY {
            continue;
        }
        let ls = if lm == f32::NEG_INFINITY {
            0.0
        } else {
            (lm - merged_max).exp()
        };
        let rs = if rm == f32::NEG_INFINITY {
            0.0
        } else {
            (rm - merged_max).exp()
        };
        let denom = left.denominator[row] * ls + right.denominator[row] * rs;
        if !denom.is_finite() || denom <= 0.0 {
            return Err(SegmentedAttentionOracleError::InvalidPartial);
        }
        maximum[row] = merged_max;
        denominator[row] = denom;
        let base = row * head_dim;
        for component in 0..head_dim {
            weighted_numerator[base + component] = left.weighted_numerator[base + component] * ls
                + right.weighted_numerator[base + component] * rs;
        }
    }
    Ok(AttentionPartial {
        maximum,
        denominator,
        weighted_numerator,
    })
}

pub fn normalize_partial(
    partial: &AttentionPartial,
    head_dim: usize,
) -> Result<Vec<f32>, SegmentedAttentionOracleError> {
    if head_dim == 0
        || partial.weighted_numerator.len() != partial.maximum.len() * head_dim
        || partial.maximum.len() != partial.denominator.len()
    {
        return Err(SegmentedAttentionOracleError::ShapeMismatch);
    }
    let mut output = vec![0.0; partial.weighted_numerator.len()];
    for row in 0..partial.maximum.len() {
        let denom = partial.denominator[row];
        if denom == 0.0 {
            if partial.maximum[row] != f32::NEG_INFINITY {
                return Err(SegmentedAttentionOracleError::InvalidPartial);
            }
            continue;
        }
        if !denom.is_finite() || denom < 0.0 {
            return Err(SegmentedAttentionOracleError::InvalidPartial);
        }
        let base = row * head_dim;
        for component in 0..head_dim {
            output[base + component] = partial.weighted_numerator[base + component] / denom;
        }
    }
    Ok(output)
}

/// Compute a contiguous oracle and a page-partitioned result for the same
/// query. `page_positions` is expressed in absolute cache positions.
pub fn compare_page_partition(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    geometry: AttentionGeometry,
    page_positions: usize,
    visible_position_end: usize,
) -> Result<SegmentedAttentionResult, SegmentedAttentionOracleError> {
    let AttentionGeometry {
        query_heads,
        kv_heads,
        head_dim,
    } = geometry;
    let positions = validate_shapes(query, keys, values, query_heads, kv_heads, head_dim)?;
    if page_positions == 0 || visible_position_end == 0 || visible_position_end > positions {
        return Err(SegmentedAttentionOracleError::InvalidRange);
    }
    let mut merged = AttentionPartial {
        maximum: vec![f32::NEG_INFINITY; query_heads],
        denominator: vec![0.0; query_heads],
        weighted_numerator: vec![0.0; query_heads * head_dim],
    };
    let mut page_begin = 0;
    while page_begin < visible_position_end {
        let page_end = (page_begin + page_positions).min(visible_position_end);
        let page = partial_for_segment(
            query,
            keys,
            values,
            geometry,
            page_begin,
            page_end,
            visible_position_end,
        )?;
        merged = merge_partials(&merged, &page, head_dim)?;
        page_begin = page_end;
    }
    let output = normalize_partial(&merged, head_dim)?;
    Ok(SegmentedAttentionResult {
        output,
        partial: merged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_partition_matches_contiguous_reference() {
        const QH: usize = 4;
        const KVH: usize = 2;
        const D: usize = 4;
        const POSITIONS: usize = 11;
        let query: Vec<f32> = (0..QH * D).map(|i| (i as f32 - 5.0) * 0.07).collect();
        let keys: Vec<f32> = (0..POSITIONS * KVH * D)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();
        let values: Vec<f32> = (0..POSITIONS * KVH * D)
            .map(|i| (i as f32 * 0.017).cos())
            .collect();
        let expected = partial_for_segment(
            &query,
            &keys,
            &values,
            AttentionGeometry {
                query_heads: QH,
                kv_heads: KVH,
                head_dim: D,
            },
            0,
            POSITIONS,
            POSITIONS,
        )
        .unwrap();
        let expected_output = normalize_partial(&expected, D).unwrap();
        let actual = compare_page_partition(
            &query,
            &keys,
            &values,
            AttentionGeometry {
                query_heads: QH,
                kv_heads: KVH,
                head_dim: D,
            },
            3,
            POSITIONS,
        )
        .unwrap();
        let max_error = actual
            .output
            .iter()
            .zip(expected_output.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_error < 2e-6, "max_error={max_error}");
    }

    #[test]
    fn absolute_visible_end_excludes_future_pages() {
        let query = vec![1.0, 0.0];
        let keys = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let values = vec![2.0, 3.0, 7.0, 11.0, 99.0, 99.0];
        let result = compare_page_partition(
            &query,
            &keys,
            &values,
            AttentionGeometry {
                query_heads: 1,
                kv_heads: 1,
                head_dim: 2,
            },
            2,
            1,
        )
        .unwrap();
        assert!((result.output[0] - 2.0).abs() < 1e-6);
        assert!((result.output[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn invalid_and_nonfinite_inputs_fail_closed() {
        assert_eq!(
            partial_for_segment(
                &[1.0],
                &[1.0],
                &[1.0, 2.0],
                AttentionGeometry {
                    query_heads: 1,
                    kv_heads: 1,
                    head_dim: 1
                },
                0,
                1,
                1
            ),
            Err(SegmentedAttentionOracleError::ShapeMismatch)
        );
        assert_eq!(
            partial_for_segment(
                &[f32::NAN, 0.0],
                &[1.0, 0.0],
                &[1.0, 0.0],
                AttentionGeometry {
                    query_heads: 1,
                    kv_heads: 1,
                    head_dim: 2
                },
                0,
                1,
                1
            ),
            Err(SegmentedAttentionOracleError::NonFiniteInput)
        );
        assert!(compare_page_partition(
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[1.0, 0.0],
            AttentionGeometry {
                query_heads: 1,
                kv_heads: 1,
                head_dim: 2
            },
            0,
            1
        )
        .is_err());
    }
}
