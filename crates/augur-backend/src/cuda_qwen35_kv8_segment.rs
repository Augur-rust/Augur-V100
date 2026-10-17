#![cfg(feature = "qwen35-kv8-full-attention")]

//! Opt-in contract for bounded Qwen35 KV8 attention segments.
//!
//! The V1 contiguous attention ABI owns the final normalized output.  This
//! contract deliberately exposes the state needed to execute one bounded
//! segment at a time and merge it with another segment without materializing
//! the complete KV cache in HBM.  The native submit entry point is intentionally
//! not part of this first contract commit; callers must not infer CUDA or
//! product qualification from these types or from the CPU merge oracle.

use core::{fmt, mem};

pub mod device_merge;

#[cfg(feature = "cuda")]
use crate::{CudaRuntimeError, OwnedCudaStream};

pub const AUGUR_CUDA_QWEN35_KV8_SEGMENT_PARTIAL_ARGS_VERSION_V2: u32 = 2;
pub const QWEN35_KV8_SEGMENT_PARTIAL_SCHEMA_V2: &str =
    "crystal-quant.qwen35.kv8.segment-partial.v2";
pub const QWEN35_KV8_QUERY_HEADS_V2: usize = 24;
pub const QWEN35_KV8_KV_HEADS_V2: usize = 4;
pub const QWEN35_KV8_HEAD_DIM_V2: usize = 256;
pub const QWEN35_KV8_BLOCK_BYTES_V2: usize = 34;
pub const QWEN35_KV8_BLOCK_ELEMENTS_V2: usize = 32;
pub const QWEN35_KV8_BLOCKS_PER_HEAD_V2: usize =
    QWEN35_KV8_HEAD_DIM_V2 / QWEN35_KV8_BLOCK_ELEMENTS_V2;
pub const QWEN35_KV8_HEAD_ROW_BYTES_V2: usize =
    QWEN35_KV8_BLOCKS_PER_HEAD_V2 * QWEN35_KV8_BLOCK_BYTES_V2;
pub const QWEN35_KV8_CACHE_ROW_BYTES_V2: usize =
    QWEN35_KV8_KV_HEADS_V2 * QWEN35_KV8_HEAD_ROW_BYTES_V2;

/// Device pointer and extent contract for one bounded segment.
///
/// `key_cache_q8_0` and `value_cache_q8_0` point to a slot of exactly
/// `slot_capacity_positions` rows.  A logical position `p` is addressed at
/// slot row `p - cache_position_base`; the requested segment must fit inside
/// that slot.  Partial arrays are indexed `[token][query_head]` and
/// `[token][query_head][head_dim]` respectively.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AugurCudaQwen35Kv8SegmentPartialArgsV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub query_f32: u64,
    pub query_len_bytes: u64,
    pub key_cache_q8_0: u64,
    pub key_cache_len_bytes: u64,
    pub value_cache_q8_0: u64,
    pub value_cache_len_bytes: u64,
    pub partial_max_f32: u64,
    pub partial_max_len_bytes: u64,
    pub partial_denominator_f32: u64,
    pub partial_denominator_len_bytes: u64,
    pub partial_weighted_value_f32: u64,
    pub partial_weighted_value_len_bytes: u64,
    pub tokens: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub cache_position_base: u32,
    pub segment_begin: u32,
    pub segment_end: u32,
    pub slot_capacity_positions: u32,
    pub visible_position_end: u32,
    pub mode: u32,
    pub reserved0: u32,
    pub reserved1: u32,
}

impl AugurCudaQwen35Kv8SegmentPartialArgsV2 {
    pub const fn new() -> Self {
        Self {
            abi_version: AUGUR_CUDA_QWEN35_KV8_SEGMENT_PARTIAL_ARGS_VERSION_V2,
            struct_size: mem::size_of::<Self>() as u32,
            query_f32: 0,
            query_len_bytes: 0,
            key_cache_q8_0: 0,
            key_cache_len_bytes: 0,
            value_cache_q8_0: 0,
            value_cache_len_bytes: 0,
            partial_max_f32: 0,
            partial_max_len_bytes: 0,
            partial_denominator_f32: 0,
            partial_denominator_len_bytes: 0,
            partial_weighted_value_f32: 0,
            partial_weighted_value_len_bytes: 0,
            tokens: 0,
            query_heads: QWEN35_KV8_QUERY_HEADS_V2 as u32,
            kv_heads: QWEN35_KV8_KV_HEADS_V2 as u32,
            head_dim: QWEN35_KV8_HEAD_DIM_V2 as u32,
            cache_position_base: 0,
            segment_begin: 0,
            segment_end: 0,
            slot_capacity_positions: 0,
            visible_position_end: 0,
            mode: 0,
            reserved0: 0,
            reserved1: 0,
        }
    }
}

impl Default for AugurCudaQwen35Kv8SegmentPartialArgsV2 {
    fn default() -> Self {
        Self::new()
    }
}

const _: () = assert!(mem::size_of::<AugurCudaQwen35Kv8SegmentPartialArgsV2>() == 152);
const _: () = assert!(mem::align_of::<AugurCudaQwen35Kv8SegmentPartialArgsV2>() == 8);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35Kv8SegmentGeometryV2 {
    pub tokens: usize,
    pub cache_position_base: usize,
    pub segment_begin: usize,
    pub segment_end: usize,
    pub slot_capacity_positions: usize,
    pub visible_position_end: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SegmentContractError {
    ZeroTokens,
    EmptySegment,
    RangeOverflow,
    SegmentOutsideVisibleRange,
    SegmentOutsideSlot,
    WrongAbi,
    WrongStructSize,
    WrongGeometry,
    NullPointer,
    WrongExtent,
    InvalidMaximum,
    InvalidDenominator,
    InvalidNumerator,
    ShapeMismatch,
}

impl fmt::Display for SegmentContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ZeroTokens => "segment tokens must be non-zero",
            Self::EmptySegment => "segment range must be non-empty",
            Self::RangeOverflow => "segment geometry overflow",
            Self::SegmentOutsideVisibleRange => "segment exceeds visible position end",
            Self::SegmentOutsideSlot => "segment exceeds bounded slot",
            Self::WrongAbi => "segment ABI version mismatch",
            Self::WrongStructSize => "segment ABI struct size mismatch",
            Self::WrongGeometry => "segment ABI geometry mismatch",
            Self::NullPointer => "segment ABI pointer is null for a non-empty extent",
            Self::WrongExtent => "segment ABI byte extent mismatch",
            Self::InvalidMaximum => "partial maximum is not finite or negative infinity",
            Self::InvalidDenominator => "partial denominator is invalid",
            Self::InvalidNumerator => "partial weighted numerator is not finite",
            Self::ShapeMismatch => "partial arrays have incompatible shapes",
        })
    }
}

impl std::error::Error for SegmentContractError {}

impl Qwen35Kv8SegmentGeometryV2 {
    pub fn validate(self) -> Result<SegmentExtentsV2, SegmentContractError> {
        if self.tokens == 0 {
            return Err(SegmentContractError::ZeroTokens);
        }
        if self.segment_begin >= self.segment_end {
            return Err(SegmentContractError::EmptySegment);
        }
        let slot_end = self
            .cache_position_base
            .checked_add(self.slot_capacity_positions)
            .ok_or(SegmentContractError::RangeOverflow)?;
        if self.segment_begin < self.cache_position_base || self.segment_end > slot_end {
            return Err(SegmentContractError::SegmentOutsideSlot);
        }
        if self.segment_end > self.visible_position_end {
            return Err(SegmentContractError::SegmentOutsideVisibleRange);
        }
        let query_elements = self
            .tokens
            .checked_mul(QWEN35_KV8_QUERY_HEADS_V2)
            .and_then(|rows| rows.checked_mul(QWEN35_KV8_HEAD_DIM_V2))
            .ok_or(SegmentContractError::RangeOverflow)?;
        let query_rows = self
            .tokens
            .checked_mul(QWEN35_KV8_QUERY_HEADS_V2)
            .ok_or(SegmentContractError::RangeOverflow)?;
        let cache_bytes = self
            .slot_capacity_positions
            .checked_mul(QWEN35_KV8_CACHE_ROW_BYTES_V2)
            .ok_or(SegmentContractError::RangeOverflow)?;
        let weighted_elements = query_elements;
        Ok(SegmentExtentsV2 {
            query_bytes: query_elements
                .checked_mul(mem::size_of::<f32>())
                .ok_or(SegmentContractError::RangeOverflow)?,
            cache_bytes,
            partial_max_bytes: query_rows
                .checked_mul(mem::size_of::<f32>())
                .ok_or(SegmentContractError::RangeOverflow)?,
            partial_denominator_bytes: query_rows
                .checked_mul(mem::size_of::<f32>())
                .ok_or(SegmentContractError::RangeOverflow)?,
            partial_weighted_value_bytes: weighted_elements
                .checked_mul(mem::size_of::<f32>())
                .ok_or(SegmentContractError::RangeOverflow)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentExtentsV2 {
    pub query_bytes: usize,
    pub cache_bytes: usize,
    pub partial_max_bytes: usize,
    pub partial_denominator_bytes: usize,
    pub partial_weighted_value_bytes: usize,
}

pub fn validate_segment_args_v2(
    args: &AugurCudaQwen35Kv8SegmentPartialArgsV2,
) -> Result<SegmentExtentsV2, SegmentContractError> {
    if args.abi_version != AUGUR_CUDA_QWEN35_KV8_SEGMENT_PARTIAL_ARGS_VERSION_V2 {
        return Err(SegmentContractError::WrongAbi);
    }
    if args.struct_size as usize != mem::size_of::<AugurCudaQwen35Kv8SegmentPartialArgsV2>() {
        return Err(SegmentContractError::WrongStructSize);
    }
    if args.query_heads as usize != QWEN35_KV8_QUERY_HEADS_V2
        || args.kv_heads as usize != QWEN35_KV8_KV_HEADS_V2
        || args.head_dim as usize != QWEN35_KV8_HEAD_DIM_V2
        || args.mode > 3
        || (args.mode == 3 && args.tokens != 1)
        || args.reserved0 != 0
        || args.reserved1 != 0
    {
        return Err(SegmentContractError::WrongGeometry);
    }
    let extents = Qwen35Kv8SegmentGeometryV2 {
        tokens: args.tokens as usize,
        cache_position_base: args.cache_position_base as usize,
        segment_begin: args.segment_begin as usize,
        segment_end: args.segment_end as usize,
        slot_capacity_positions: args.slot_capacity_positions as usize,
        visible_position_end: args.visible_position_end as usize,
    }
    .validate()?;
    let pointers = [
        (args.query_f32, extents.query_bytes),
        (args.key_cache_q8_0, extents.cache_bytes),
        (args.value_cache_q8_0, extents.cache_bytes),
        (args.partial_max_f32, extents.partial_max_bytes),
        (
            args.partial_denominator_f32,
            extents.partial_denominator_bytes,
        ),
        (
            args.partial_weighted_value_f32,
            extents.partial_weighted_value_bytes,
        ),
    ];
    if pointers
        .iter()
        .any(|(pointer, extent)| *pointer == 0 && *extent != 0)
    {
        return Err(SegmentContractError::NullPointer);
    }
    let declared = [
        args.query_len_bytes as usize,
        args.key_cache_len_bytes as usize,
        args.value_cache_len_bytes as usize,
        args.partial_max_len_bytes as usize,
        args.partial_denominator_len_bytes as usize,
        args.partial_weighted_value_len_bytes as usize,
    ];
    let expected = [
        extents.query_bytes,
        extents.cache_bytes,
        extents.cache_bytes,
        extents.partial_max_bytes,
        extents.partial_denominator_bytes,
        extents.partial_weighted_value_bytes,
    ];
    if declared != expected {
        return Err(SegmentContractError::WrongExtent);
    }
    Ok(extents)
}

#[cfg(feature = "cuda")]
fn raw_segment_args_v2(
    args: &AugurCudaQwen35Kv8SegmentPartialArgsV2,
) -> augur_cuda_sys::AugurCudaQwen35Kv8SegmentPartialArgsV2 {
    augur_cuda_sys::AugurCudaQwen35Kv8SegmentPartialArgsV2 {
        abi_version: args.abi_version,
        struct_size: args.struct_size,
        query_f32: args.query_f32,
        query_len_bytes: args.query_len_bytes,
        key_cache_q8_0: args.key_cache_q8_0,
        key_cache_len_bytes: args.key_cache_len_bytes,
        value_cache_q8_0: args.value_cache_q8_0,
        value_cache_len_bytes: args.value_cache_len_bytes,
        partial_max_f32: args.partial_max_f32,
        partial_max_len_bytes: args.partial_max_len_bytes,
        partial_denominator_f32: args.partial_denominator_f32,
        partial_denominator_len_bytes: args.partial_denominator_len_bytes,
        partial_weighted_value_f32: args.partial_weighted_value_f32,
        partial_weighted_value_len_bytes: args.partial_weighted_value_len_bytes,
        tokens: args.tokens,
        query_heads: args.query_heads,
        kv_heads: args.kv_heads,
        head_dim: args.head_dim,
        cache_position_base: args.cache_position_base,
        segment_begin: args.segment_begin,
        segment_end: args.segment_end,
        slot_capacity_positions: args.slot_capacity_positions,
        visible_position_end: args.visible_position_end,
        mode: args.mode,
        reserved0: args.reserved0,
        reserved1: args.reserved1,
    }
}

/// Validate and enqueue one bounded KV8 segment partial on an owned stream.
///
/// This is deliberately component-scoped: page ownership/epoch receipts are
/// not inferred from raw pointers and must be checked by the qualification
/// runner before calling this function.
#[cfg(feature = "cuda")]
pub fn submit_segment_partial_v2(
    stream: &OwnedCudaStream,
    args: &AugurCudaQwen35Kv8SegmentPartialArgsV2,
) -> Result<(), CudaRuntimeError> {
    validate_segment_args_v2(args)
        .map_err(|_| CudaRuntimeError::invalid_argument("qwen35 kv8 segment arguments"))?;
    let raw = raw_segment_args_v2(args);
    let native_stream = stream.raw()?;
    // SAFETY: `raw` is a stack snapshot with validated extents; native code
    // enqueues work on the live stream and does not retain the args pointer.
    let result = unsafe {
        augur_cuda_sys::augur_cuda_qwen35_kv8_segment_partial_submit_v2(
            stream.device_ordinal(),
            native_stream,
            &raw,
        )
    };
    CudaRuntimeError::from_result("cuda_qwen35_kv8_segment_partial_submit_v2", result)
}

#[derive(Clone, Copy, Debug)]
pub struct PartialLseViewV2<'a> {
    pub maximum: &'a [f32],
    pub denominator: &'a [f32],
    pub weighted_numerator: &'a [f32],
    pub head_dim: usize,
}

impl<'a> PartialLseViewV2<'a> {
    pub fn validate(self) -> Result<(), SegmentContractError> {
        if self.head_dim == 0
            || self.maximum.len() != self.denominator.len()
            || self.weighted_numerator.len() != self.maximum.len().saturating_mul(self.head_dim)
        {
            return Err(SegmentContractError::ShapeMismatch);
        }
        for (&maximum, &denominator) in self.maximum.iter().zip(self.denominator) {
            if !maximum.is_finite() && maximum != f32::NEG_INFINITY {
                return Err(SegmentContractError::InvalidMaximum);
            }
            if !denominator.is_finite() || denominator < 0.0 {
                return Err(SegmentContractError::InvalidDenominator);
            }
            if (maximum == f32::NEG_INFINITY) != (denominator == 0.0) {
                return Err(SegmentContractError::InvalidDenominator);
            }
        }
        if self
            .weighted_numerator
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(SegmentContractError::InvalidNumerator);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MergedLseV2 {
    pub maximum: Vec<f32>,
    pub denominator: Vec<f32>,
    pub weighted_numerator: Vec<f32>,
    pub normalized_value: Vec<f32>,
}

pub fn merge_partial_lse_v2(
    left: PartialLseViewV2<'_>,
    right: PartialLseViewV2<'_>,
) -> Result<MergedLseV2, SegmentContractError> {
    left.validate()?;
    right.validate()?;
    if left.head_dim != right.head_dim || left.maximum.len() != right.maximum.len() {
        return Err(SegmentContractError::ShapeMismatch);
    }
    let rows = left.maximum.len();
    let mut maximum = vec![f32::NEG_INFINITY; rows];
    let mut denominator = vec![0.0_f32; rows];
    let mut weighted_numerator = vec![0.0_f32; rows * left.head_dim];
    for row in 0..rows {
        let left_max = left.maximum[row];
        let right_max = right.maximum[row];
        let merged_max = left_max.max(right_max);
        if merged_max == f32::NEG_INFINITY {
            continue;
        }
        let left_scale = if left_max == f32::NEG_INFINITY {
            0.0
        } else {
            (left_max - merged_max).exp()
        };
        let right_scale = if right_max == f32::NEG_INFINITY {
            0.0
        } else {
            (right_max - merged_max).exp()
        };
        let merged_denominator =
            left.denominator[row] * left_scale + right.denominator[row] * right_scale;
        if !merged_denominator.is_finite() || merged_denominator <= 0.0 {
            return Err(SegmentContractError::InvalidDenominator);
        }
        maximum[row] = merged_max;
        denominator[row] = merged_denominator;
        let base = row * left.head_dim;
        for component in 0..left.head_dim {
            weighted_numerator[base + component] = left.weighted_numerator[base + component]
                * left_scale
                + right.weighted_numerator[base + component] * right_scale;
        }
    }
    let mut normalized_value = vec![0.0_f32; weighted_numerator.len()];
    for row in 0..rows {
        let base = row * left.head_dim;
        if denominator[row] > 0.0 {
            for component in 0..left.head_dim {
                normalized_value[base + component] =
                    weighted_numerator[base + component] / denominator[row];
            }
        }
    }
    Ok(MergedLseV2 {
        maximum,
        denominator,
        weighted_numerator,
        normalized_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn partial<'a>(
        maximum: &'a [f32],
        denominator: &'a [f32],
        weighted_numerator: &'a [f32],
    ) -> PartialLseViewV2<'a> {
        PartialLseViewV2 {
            maximum,
            denominator,
            weighted_numerator,
            head_dim: 2,
        }
    }

    #[test]
    fn abi_layout_and_default_are_stable() {
        let args = AugurCudaQwen35Kv8SegmentPartialArgsV2::new();
        assert_eq!(args.abi_version, 2);
        assert_eq!(args.struct_size as usize, mem::size_of_val(&args));
        assert_eq!(mem::size_of_val(&args), 152);
        assert_eq!(mem::align_of_val(&args), 8);
    }

    #[test]
    fn absolute_position_segment_maps_inside_bounded_slot() {
        let extents = Qwen35Kv8SegmentGeometryV2 {
            tokens: 4,
            cache_position_base: 8_192,
            segment_begin: 8_704,
            segment_end: 9_216,
            slot_capacity_positions: 1_024,
            visible_position_end: 16_384,
        }
        .validate()
        .unwrap();
        assert_eq!(extents.cache_bytes, 1_024 * QWEN35_KV8_CACHE_ROW_BYTES_V2);
    }

    #[test]
    fn geometry_rejects_slot_and_visible_range_drift() {
        let base = Qwen35Kv8SegmentGeometryV2 {
            tokens: 1,
            cache_position_base: 100,
            segment_begin: 100,
            segment_end: 132,
            slot_capacity_positions: 32,
            visible_position_end: 132,
        };
        assert_eq!(base.validate().unwrap().query_bytes, 24 * 256 * 4);
        assert_eq!(
            Qwen35Kv8SegmentGeometryV2 {
                segment_begin: 99,
                ..base
            }
            .validate(),
            Err(SegmentContractError::SegmentOutsideSlot)
        );
        assert_eq!(
            Qwen35Kv8SegmentGeometryV2 {
                segment_end: 133,
                ..base
            }
            .validate(),
            Err(SegmentContractError::SegmentOutsideSlot)
        );
        assert_eq!(
            Qwen35Kv8SegmentGeometryV2 {
                visible_position_end: 131,
                ..base
            }
            .validate(),
            Err(SegmentContractError::SegmentOutsideVisibleRange)
        );
    }

    #[test]
    fn args_reject_unknown_mode_reserved_and_extent_drift() {
        let mut args = AugurCudaQwen35Kv8SegmentPartialArgsV2::new();
        args.tokens = 1;
        args.segment_end = 1;
        args.slot_capacity_positions = 1;
        args.visible_position_end = 1;
        let extents = Qwen35Kv8SegmentGeometryV2 {
            tokens: 1,
            cache_position_base: 0,
            segment_begin: 0,
            segment_end: 1,
            slot_capacity_positions: 1,
            visible_position_end: 1,
        }
        .validate()
        .unwrap();
        args.query_len_bytes = extents.query_bytes as u64;
        args.key_cache_len_bytes = extents.cache_bytes as u64;
        args.value_cache_len_bytes = extents.cache_bytes as u64;
        args.partial_max_len_bytes = extents.partial_max_bytes as u64;
        args.partial_denominator_len_bytes = extents.partial_denominator_bytes as u64;
        args.partial_weighted_value_len_bytes = extents.partial_weighted_value_bytes as u64;
        args.query_f32 = 1;
        args.key_cache_q8_0 = 1;
        args.value_cache_q8_0 = 1;
        args.partial_max_f32 = 1;
        args.partial_denominator_f32 = 1;
        args.partial_weighted_value_f32 = 1;
        assert!(validate_segment_args_v2(&args).is_ok());
        for mode in 0..=3 {
            args.mode = mode;
            assert!(validate_segment_args_v2(&args).is_ok());
        }
        args.mode = 3;
        args.tokens = 2;
        assert_eq!(
            validate_segment_args_v2(&args),
            Err(SegmentContractError::WrongGeometry)
        );
        args.tokens = 1;
        args.mode = 4;
        assert_eq!(
            validate_segment_args_v2(&args),
            Err(SegmentContractError::WrongGeometry)
        );
        args.mode = 0;
        args.reserved0 = 1;
        assert_eq!(
            validate_segment_args_v2(&args),
            Err(SegmentContractError::WrongGeometry)
        );
        args.reserved0 = 0;
        args.query_len_bytes -= 4;
        assert_eq!(
            validate_segment_args_v2(&args),
            Err(SegmentContractError::WrongExtent)
        );
    }

    #[test]
    fn merge_matches_two_weighted_segments() {
        let left = partial(&[2.0], &[3.0], &[6.0, 3.0]);
        let right = partial(&[1.0], &[4.0], &[4.0, 8.0]);
        let merged = merge_partial_lse_v2(left, right).unwrap();
        let expected_denom = 3.0 * 1.0 + 4.0 * (-1.0_f32).exp();
        assert!((merged.maximum[0] - 2.0).abs() < 1e-6);
        assert!((merged.denominator[0] - expected_denom).abs() < 1e-6);
        assert!(
            (merged.normalized_value[0] - (6.0 + 4.0 * (-1.0_f32).exp()) / expected_denom).abs()
                < 1e-6
        );
        assert!(
            (merged.normalized_value[1] - (3.0 + 8.0 * (-1.0_f32).exp()) / expected_denom).abs()
                < 1e-6
        );
    }

    #[test]
    fn merge_handles_empty_side_and_both_empty() {
        let empty = partial(&[f32::NEG_INFINITY], &[0.0], &[0.0, 0.0]);
        let live = partial(&[0.5], &[2.0], &[1.0, -2.0]);
        let right_only = merge_partial_lse_v2(empty, live).unwrap();
        assert_eq!(right_only.maximum, vec![0.5]);
        assert_eq!(right_only.denominator, vec![2.0]);
        assert_eq!(right_only.normalized_value, vec![0.5, -1.0]);
        let both_empty = merge_partial_lse_v2(empty, empty).unwrap();
        assert_eq!(both_empty.maximum, vec![f32::NEG_INFINITY]);
        assert_eq!(both_empty.denominator, vec![0.0]);
        assert_eq!(both_empty.normalized_value, vec![0.0, 0.0]);
    }

    #[test]
    fn merge_rejects_nonfinite_and_inconsistent_partials() {
        assert_eq!(
            merge_partial_lse_v2(
                partial(&[f32::NAN], &[1.0], &[1.0, 1.0]),
                partial(&[0.0], &[1.0], &[1.0, 1.0])
            ),
            Err(SegmentContractError::InvalidMaximum)
        );
        assert_eq!(
            merge_partial_lse_v2(
                partial(&[0.0], &[0.0], &[0.0, 0.0]),
                partial(&[0.0], &[1.0], &[1.0, 1.0])
            ),
            Err(SegmentContractError::InvalidDenominator)
        );
        assert_eq!(
            merge_partial_lse_v2(
                partial(&[0.0], &[1.0], &[f32::INFINITY, 0.0]),
                partial(&[0.0], &[1.0], &[1.0, 1.0])
            ),
            Err(SegmentContractError::InvalidNumerator)
        );
        assert_eq!(
            merge_partial_lse_v2(
                PartialLseViewV2 {
                    maximum: &[0.0],
                    denominator: &[1.0],
                    weighted_numerator: &[1.0],
                    head_dim: 1
                },
                partial(&[0.0], &[1.0], &[1.0, 1.0])
            ),
            Err(SegmentContractError::ShapeMismatch)
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an SM70 CUDA device"]
    fn native_segment_partial_sm70_smoke_produces_finite_partial() {
        use crate::{submit_segment_partial_v2, OwnedCudaBuffer, OwnedCudaStream};

        const TOKENS: usize = 1;
        const CAPACITY: usize = 2;
        let stream = OwnedCudaStream::create(0).expect("CUDA stream");
        let query_bytes = TOKENS * QWEN35_KV8_QUERY_HEADS_V2 * QWEN35_KV8_HEAD_DIM_V2 * 4;
        let cache_bytes = CAPACITY * QWEN35_KV8_CACHE_ROW_BYTES_V2;
        let partial_rows = TOKENS * QWEN35_KV8_QUERY_HEADS_V2;
        let weighted_bytes = partial_rows * QWEN35_KV8_HEAD_DIM_V2 * 4;
        let query_host = vec![0.01_f32; query_bytes / 4];
        let query_host_bytes = query_host
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let mut cache_host = vec![0_u8; cache_bytes];
        for block in cache_host.chunks_exact_mut(QWEN35_KV8_BLOCK_BYTES_V2) {
            block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            block[2..].fill(1);
        }
        let query = OwnedCudaBuffer::allocate(0, query_bytes).expect("query allocation");
        query
            .copy_from_host_sync(&stream, &query_host_bytes)
            .expect("query upload");
        let key = OwnedCudaBuffer::allocate(0, cache_bytes).expect("key allocation");
        key.copy_from_host_sync(&stream, &cache_host)
            .expect("key upload");
        let value = OwnedCudaBuffer::allocate(0, cache_bytes).expect("value allocation");
        value
            .copy_from_host_sync(&stream, &cache_host)
            .expect("value upload");
        let maximum = OwnedCudaBuffer::allocate(0, partial_rows * 4).expect("max allocation");
        let denominator =
            OwnedCudaBuffer::allocate(0, partial_rows * 4).expect("denominator allocation");
        let weighted = OwnedCudaBuffer::allocate(0, weighted_bytes).expect("weighted allocation");
        let mut args = AugurCudaQwen35Kv8SegmentPartialArgsV2::new();
        args.query_f32 = query.device_ptr().unwrap().raw_value();
        args.query_len_bytes = query_bytes as u64;
        args.key_cache_q8_0 = key.device_ptr().unwrap().raw_value();
        args.key_cache_len_bytes = cache_bytes as u64;
        args.value_cache_q8_0 = value.device_ptr().unwrap().raw_value();
        args.value_cache_len_bytes = cache_bytes as u64;
        args.partial_max_f32 = maximum.device_ptr().unwrap().raw_value();
        args.partial_max_len_bytes = (partial_rows * 4) as u64;
        args.partial_denominator_f32 = denominator.device_ptr().unwrap().raw_value();
        args.partial_denominator_len_bytes = (partial_rows * 4) as u64;
        args.partial_weighted_value_f32 = weighted.device_ptr().unwrap().raw_value();
        args.partial_weighted_value_len_bytes = weighted_bytes as u64;
        args.tokens = TOKENS as u32;
        args.cache_position_base = 100;
        args.segment_begin = 100;
        args.segment_end = 102;
        args.slot_capacity_positions = CAPACITY as u32;
        args.visible_position_end = 102;
        submit_segment_partial_v2(&stream, &args).expect("native segment submit");
        stream.synchronize().expect("segment completion");
        let mut max_host = vec![0_u8; partial_rows * 4];
        let mut denom_host = vec![0_u8; partial_rows * 4];
        denominator
            .copy_to_host_sync(&stream, &mut denom_host)
            .expect("denominator download");
        maximum
            .copy_to_host_sync(&stream, &mut max_host)
            .expect("maximum download");
        let max_value = f32::from_le_bytes(max_host[..4].try_into().unwrap());
        let denom_value = f32::from_le_bytes(denom_host[..4].try_into().unwrap());
        assert!(max_value.is_finite());
        assert!(denom_value.is_finite() && denom_value > 0.0);
        // The opt-in parallel/F64 producer keeps the same absolute-position
        // and F32 partial wire. Constant K/V gives an analytic two-row oracle.
        args.mode = 1;
        submit_segment_partial_v2(&stream, &args).expect("precise segment submit");
        let mut weighted_host = vec![0_u8; weighted_bytes];
        weighted
            .copy_to_host_sync(&stream, &mut weighted_host)
            .unwrap();
        denominator
            .copy_to_host_sync(&stream, &mut denom_host)
            .unwrap();
        maximum.copy_to_host_sync(&stream, &mut max_host).unwrap();
        for bytes in max_host.chunks_exact(4) {
            assert!((f32::from_le_bytes(bytes.try_into().unwrap()) - 0.16).abs() < 1e-6);
        }
        for bytes in denom_host
            .chunks_exact(4)
            .chain(weighted_host.chunks_exact(4))
        {
            assert_eq!(f32::from_le_bytes(bytes.try_into().unwrap()), 2.0);
        }
        // Keep mode-2 independent output as an arithmetic oracle for the
        // ordered mode. Mode 3 must preserve it across physical page cuts.
        args.mode = 2;
        args.cache_position_base = 0;
        args.segment_begin = 0;
        args.segment_end = 2;
        args.visible_position_end = 2;
        submit_segment_partial_v2(&stream, &args).unwrap();
        maximum.copy_to_host_sync(&stream, &mut max_host).unwrap();
        denominator
            .copy_to_host_sync(&stream, &mut denom_host)
            .unwrap();
        weighted
            .copy_to_host_sync(&stream, &mut weighted_host)
            .unwrap();
        let expected = (max_host.clone(), denom_host.clone(), weighted_host.clone());
        args.mode = 3;
        args.segment_end = 1;
        submit_segment_partial_v2(&stream, &args).unwrap();
        // Use a bounded view of the physical second row with absolute base=1.
        args.key_cache_q8_0 += QWEN35_KV8_CACHE_ROW_BYTES_V2 as u64;
        args.value_cache_q8_0 += QWEN35_KV8_CACHE_ROW_BYTES_V2 as u64;
        args.key_cache_len_bytes /= 2;
        args.value_cache_len_bytes /= 2;
        args.slot_capacity_positions = 1;
        args.cache_position_base = 1;
        args.segment_begin = 1;
        args.segment_end = 2;
        submit_segment_partial_v2(&stream, &args).unwrap();
        maximum.copy_to_host_sync(&stream, &mut max_host).unwrap();
        denominator
            .copy_to_host_sync(&stream, &mut denom_host)
            .unwrap();
        weighted
            .copy_to_host_sync(&stream, &mut weighted_host)
            .unwrap();
        assert_eq!((max_host, denom_host, weighted_host), expected);
    }
}
