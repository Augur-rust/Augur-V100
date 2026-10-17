use augur_core::PhysicalLayoutId;
use augur_formats::{Q6KCompactBlock, Q4K_BLOCK_BYTES, Q5K_BLOCK_BYTES, Q6K_BLOCK_BYTES};
use augur_kernel_api::{
    HotLayoutError, Q4KHotCudaStagingTensorV1, QkHotCudaStagingTensorV1,
    Q4K_HOT_CUDA_V1_BLOCK_BYTES, Q4K_HOT_CUDA_V1_MIN_OFFSET, Q4K_HOT_CUDA_V1_QS_OFFSET,
    Q4K_HOT_CUDA_V1_RESERVED_OFFSET, Q4K_HOT_CUDA_V1_SCALE_OFFSET, Q5K_HOT_CUDA_V1_BLOCK_BYTES,
    Q5K_HOT_CUDA_V1_QH_OFFSET, Q5K_HOT_CUDA_V1_QS_OFFSET, Q5K_HOT_CUDA_V1_RESERVED_OFFSET,
    Q6K_HOT_CUDA_V1_BLOCK_BYTES,
};
use std::ops::Range;

const COMPACT_FACTORS_END: usize = 4;
const COMPACT_SCALES_OFFSET: usize = 4;
const COMPACT_SCALES_END: usize = 16;
const Q4K_COMPACT_QS_OFFSET: usize = 16;
const Q5K_COMPACT_QH_OFFSET: usize = 16;
const Q5K_COMPACT_QS_OFFSET: usize = 48;

pub fn repack_q4k_compact_to_cuda_v1(
    compact: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Q4KHotCudaStagingTensorV1, HotLayoutError> {
    let blocks = parse_shape(compact, rows, cols, Q4K_BLOCK_BYTES)?;
    let mut output = vec![0_u8; blocks * Q4K_HOT_CUDA_V1_BLOCK_BYTES];
    repack_q4k_compact_to_cuda_v1_into(compact, rows, cols, &mut output)?;
    Q4KHotCudaStagingTensorV1::new(rows, cols, output)
}

/// Repack compact Q4_K blocks directly into a caller-owned CUDA-hot payload.
///
/// The output is byte-identical to [`repack_q4k_compact_to_cuda_v1`] but does
/// not allocate complete CPU-hot and CUDA-hot tensors. This is intended for
/// fixed pinned staging buffers whose lifetime is managed by the caller.
pub fn repack_q4k_compact_to_cuda_v1_into(
    compact: &[u8],
    rows: usize,
    cols: usize,
    output: &mut [u8],
) -> Result<(), HotLayoutError> {
    let blocks = parse_shape(compact, rows, cols, Q4K_BLOCK_BYTES)?;
    validate_output_len(output, blocks, Q4K_HOT_CUDA_V1_BLOCK_BYTES)?;
    repack_q4k_compact_block_range_to_cuda_v1_into(compact, 0..blocks, blocks, output)
}

/// Repack one contiguous range of Q4_K blocks into a caller-owned CUDA-hot
/// extent.
///
/// `compact` and `output` must contain exactly the blocks in `block_range`;
/// `total_blocks` binds that range to the source tensor's admitted logical
/// extent. The conversion itself is allocation-free and does not require the
/// surrounding tensor to be resident in memory.
pub fn repack_q4k_compact_block_range_to_cuda_v1_into(
    compact: &[u8],
    block_range: Range<usize>,
    total_blocks: usize,
    output: &mut [u8],
) -> Result<(), HotLayoutError> {
    validate_block_range(
        compact,
        output,
        block_range,
        total_blocks,
        Q4K_BLOCK_BYTES,
        Q4K_HOT_CUDA_V1_BLOCK_BYTES,
    )?;
    for (source, destination) in compact
        .chunks_exact(Q4K_BLOCK_BYTES)
        .zip(output.chunks_exact_mut(Q4K_HOT_CUDA_V1_BLOCK_BYTES))
    {
        validate_finite_factors(source)?;
        destination[..COMPACT_FACTORS_END].copy_from_slice(&source[..COMPACT_FACTORS_END]);
        unpack_scales_into(
            &source[COMPACT_SCALES_OFFSET..COMPACT_SCALES_END],
            destination,
        );
        destination[Q4K_HOT_CUDA_V1_RESERVED_OFFSET..Q4K_HOT_CUDA_V1_QS_OFFSET].fill(0);
        destination[Q4K_HOT_CUDA_V1_QS_OFFSET..].copy_from_slice(&source[Q4K_COMPACT_QS_OFFSET..]);
    }
    Ok(())
}

pub fn repack_q5k_compact_to_cuda_v1(
    compact: &[u8],
    rows: usize,
    cols: usize,
) -> Result<QkHotCudaStagingTensorV1, HotLayoutError> {
    let blocks = parse_shape(compact, rows, cols, Q5K_BLOCK_BYTES)?;
    let mut output = vec![0_u8; blocks * Q5K_HOT_CUDA_V1_BLOCK_BYTES];
    repack_q5k_compact_to_cuda_v1_into(compact, rows, cols, &mut output)?;
    QkHotCudaStagingTensorV1::new(
        PhysicalLayoutId::Q5K_HOT_CUDA_V1,
        Q5K_HOT_CUDA_V1_BLOCK_BYTES,
        Q5K_BLOCK_BYTES,
        rows,
        cols,
        output,
    )
}

/// Repack compact Q5_K blocks directly into a caller-owned CUDA-hot payload.
///
/// Reserved bytes are canonicalized on every call, so a reused destination
/// cannot retain data from the preceding expert.
pub fn repack_q5k_compact_to_cuda_v1_into(
    compact: &[u8],
    rows: usize,
    cols: usize,
    output: &mut [u8],
) -> Result<(), HotLayoutError> {
    let blocks = parse_shape(compact, rows, cols, Q5K_BLOCK_BYTES)?;
    validate_output_len(output, blocks, Q5K_HOT_CUDA_V1_BLOCK_BYTES)?;
    repack_q5k_compact_block_range_to_cuda_v1_into(compact, 0..blocks, blocks, output)
}

/// Repack one contiguous range of Q5_K blocks into a caller-owned CUDA-hot
/// extent. See [`repack_q4k_compact_block_range_to_cuda_v1_into`] for the
/// extent and allocation contract.
pub fn repack_q5k_compact_block_range_to_cuda_v1_into(
    compact: &[u8],
    block_range: Range<usize>,
    total_blocks: usize,
    output: &mut [u8],
) -> Result<(), HotLayoutError> {
    validate_block_range(
        compact,
        output,
        block_range,
        total_blocks,
        Q5K_BLOCK_BYTES,
        Q5K_HOT_CUDA_V1_BLOCK_BYTES,
    )?;
    for (source, destination) in compact
        .chunks_exact(Q5K_BLOCK_BYTES)
        .zip(output.chunks_exact_mut(Q5K_HOT_CUDA_V1_BLOCK_BYTES))
    {
        validate_finite_factors(source)?;
        destination[..COMPACT_FACTORS_END].copy_from_slice(&source[..COMPACT_FACTORS_END]);
        unpack_scales_into(
            &source[COMPACT_SCALES_OFFSET..COMPACT_SCALES_END],
            destination,
        );
        destination[Q5K_HOT_CUDA_V1_RESERVED_OFFSET..Q5K_HOT_CUDA_V1_QH_OFFSET].fill(0);
        destination[Q5K_HOT_CUDA_V1_QH_OFFSET..Q5K_HOT_CUDA_V1_QS_OFFSET]
            .copy_from_slice(&source[Q5K_COMPACT_QH_OFFSET..Q5K_COMPACT_QS_OFFSET]);
        destination[Q5K_HOT_CUDA_V1_QS_OFFSET..Q5K_HOT_CUDA_V1_BLOCK_BYTES]
            .copy_from_slice(&source[Q5K_COMPACT_QS_OFFSET..]);
    }
    Ok(())
}

#[inline]
fn validate_finite_factors(source: &[u8]) -> Result<(), HotLayoutError> {
    debug_assert!(source.len() >= COMPACT_FACTORS_END);
    let d = u16::from_le_bytes([source[0], source[1]]);
    let dmin = u16::from_le_bytes([source[2], source[3]]);
    if d & 0x7c00 == 0x7c00 || dmin & 0x7c00 == 0x7c00 {
        return Err(HotLayoutError::BadShape);
    }
    Ok(())
}

#[inline]
fn unpack_scales_into(packed: &[u8], destination: &mut [u8]) {
    debug_assert_eq!(packed.len(), 12);
    debug_assert!(destination.len() >= Q4K_HOT_CUDA_V1_MIN_OFFSET + 8);
    for group in 0..4 {
        destination[Q4K_HOT_CUDA_V1_SCALE_OFFSET + group] = packed[group] & 0x3f;
        destination[Q4K_HOT_CUDA_V1_MIN_OFFSET + group] = packed[group + 4] & 0x3f;
        destination[Q4K_HOT_CUDA_V1_SCALE_OFFSET + group + 4] =
            (packed[group + 8] & 0x0f) | ((packed[group] >> 6) << 4);
        destination[Q4K_HOT_CUDA_V1_MIN_OFFSET + group + 4] =
            (packed[group + 8] >> 4) | ((packed[group + 4] >> 6) << 4);
    }
}

pub fn repack_q6k_compact_to_cuda_v1(
    compact: &[u8],
    rows: usize,
    cols: usize,
) -> Result<QkHotCudaStagingTensorV1, HotLayoutError> {
    repack_padded::<Q6KCompactBlock>(
        compact,
        rows,
        cols,
        Q6K_BLOCK_BYTES,
        Q6K_HOT_CUDA_V1_BLOCK_BYTES,
        PhysicalLayoutId::Q6K_HOT_CUDA_V1,
        Q6KCompactBlock::from_le_bytes,
    )
}

fn repack_padded<T>(
    compact: &[u8],
    rows: usize,
    cols: usize,
    compact_block_bytes: usize,
    hot_block_bytes: usize,
    layout: PhysicalLayoutId,
    parse: fn(&[u8]) -> Result<T, augur_formats::FormatError>,
) -> Result<QkHotCudaStagingTensorV1, HotLayoutError> {
    let blocks = parse_shape(compact, rows, cols, compact_block_bytes)?;
    let mut output = vec![0_u8; blocks * hot_block_bytes];
    for (index, block) in compact.chunks_exact(compact_block_bytes).enumerate() {
        parse(block).map_err(|_| HotLayoutError::BadShape)?;
        let start = index * hot_block_bytes;
        output[start..start + compact_block_bytes].copy_from_slice(block);
    }
    QkHotCudaStagingTensorV1::new(
        layout,
        hot_block_bytes,
        compact_block_bytes,
        rows,
        cols,
        output,
    )
}

fn parse_shape(
    compact: &[u8],
    rows: usize,
    cols: usize,
    block_bytes: usize,
) -> Result<usize, HotLayoutError> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(256) {
        return Err(HotLayoutError::BadShape);
    }
    let blocks = rows
        .checked_mul(cols / 256)
        .ok_or(HotLayoutError::BadShape)?;
    let expected = blocks
        .checked_mul(block_bytes)
        .ok_or(HotLayoutError::BadShape)?;
    if compact.len() != expected {
        return Err(HotLayoutError::BadPayloadBytes {
            expected,
            actual: compact.len(),
        });
    }
    Ok(blocks)
}

fn validate_output_len(
    output: &[u8],
    blocks: usize,
    hot_block_bytes: usize,
) -> Result<(), HotLayoutError> {
    let expected = blocks
        .checked_mul(hot_block_bytes)
        .ok_or(HotLayoutError::BadShape)?;
    if output.len() != expected {
        return Err(HotLayoutError::BadPayloadBytes {
            expected,
            actual: output.len(),
        });
    }
    Ok(())
}

fn validate_block_range(
    compact: &[u8],
    output: &[u8],
    block_range: Range<usize>,
    total_blocks: usize,
    compact_block_bytes: usize,
    hot_block_bytes: usize,
) -> Result<usize, HotLayoutError> {
    if total_blocks == 0 || block_range.start >= block_range.end || block_range.end > total_blocks {
        return Err(HotLayoutError::BadShape);
    }
    total_blocks
        .checked_mul(compact_block_bytes)
        .and_then(|_| total_blocks.checked_mul(hot_block_bytes))
        .ok_or(HotLayoutError::BadShape)?;
    let blocks = block_range
        .end
        .checked_sub(block_range.start)
        .ok_or(HotLayoutError::BadShape)?;
    let expected_compact = blocks
        .checked_mul(compact_block_bytes)
        .ok_or(HotLayoutError::BadShape)?;
    if compact.len() != expected_compact {
        return Err(HotLayoutError::BadPayloadBytes {
            expected: expected_compact,
            actual: compact.len(),
        });
    }
    let expected_hot = blocks
        .checked_mul(hot_block_bytes)
        .ok_or(HotLayoutError::BadShape)?;
    if output.len() != expected_hot {
        return Err(HotLayoutError::BadPayloadBytes {
            expected: expected_hot,
            actual: output.len(),
        });
    }
    Ok(blocks)
}
