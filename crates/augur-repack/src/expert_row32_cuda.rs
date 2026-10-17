use augur_core::PhysicalLayoutId;
use augur_kernel_api::{
    HotLayoutError, Q4KHotCudaStagingTensorV1, QkExpertRow32CudaStagingTensorV2,
    QkHotCudaStagingTensorV1, EXPERT_ROW32_CUDA_V2_ROWS_PER_GROUP,
    EXPERT_ROW32_CUDA_V2_VECTOR_BYTES, Q4K_HOT_CUDA_V1_BLOCK_BYTES, Q5K_HOT_CUDA_V1_BLOCK_BYTES,
    Q5K_HOT_CUDA_V1_COMPACT_BYTES,
};

pub fn repack_q4k_hot_cuda_v1_to_expert_row32_v2(
    source: &Q4KHotCudaStagingTensorV1,
) -> Result<QkExpertRow32CudaStagingTensorV2, HotLayoutError> {
    interleave(
        PhysicalLayoutId::Q4K_EXPERT_ROW32_CUDA_V2,
        source.rows(),
        source.cols(),
        Q4K_HOT_CUDA_V1_BLOCK_BYTES,
        source.payload_bytes(),
    )
}

pub fn repack_q5k_hot_cuda_v1_to_expert_row32_v2(
    source: &QkHotCudaStagingTensorV1,
) -> Result<QkExpertRow32CudaStagingTensorV2, HotLayoutError> {
    if source.layout() != PhysicalLayoutId::Q5K_HOT_CUDA_V1
        || source.block_bytes() != Q5K_HOT_CUDA_V1_BLOCK_BYTES
    {
        return Err(HotLayoutError::BadShape);
    }
    interleave(
        PhysicalLayoutId::Q5K_EXPERT_ROW32_CUDA_V2,
        source.rows(),
        source.cols(),
        Q5K_HOT_CUDA_V1_BLOCK_BYTES,
        source.payload_bytes(),
    )
}

pub fn deinterleave_q4k_expert_row32_v2_to_hot_cuda_v1(
    source: &QkExpertRow32CudaStagingTensorV2,
) -> Result<Q4KHotCudaStagingTensorV1, HotLayoutError> {
    if source.layout() != PhysicalLayoutId::Q4K_EXPERT_ROW32_CUDA_V2 {
        return Err(HotLayoutError::BadShape);
    }
    let bytes = deinterleave(source)?;
    Q4KHotCudaStagingTensorV1::new(source.rows(), source.cols(), bytes)
}

pub fn deinterleave_q5k_expert_row32_v2_to_hot_cuda_v1(
    source: &QkExpertRow32CudaStagingTensorV2,
) -> Result<QkHotCudaStagingTensorV1, HotLayoutError> {
    if source.layout() != PhysicalLayoutId::Q5K_EXPERT_ROW32_CUDA_V2 {
        return Err(HotLayoutError::BadShape);
    }
    let bytes = deinterleave(source)?;
    QkHotCudaStagingTensorV1::new(
        PhysicalLayoutId::Q5K_HOT_CUDA_V1,
        Q5K_HOT_CUDA_V1_BLOCK_BYTES,
        Q5K_HOT_CUDA_V1_COMPACT_BYTES,
        source.rows(),
        source.cols(),
        bytes,
    )
}

fn interleave(
    layout: PhysicalLayoutId,
    rows: usize,
    cols: usize,
    block_bytes: usize,
    source: &[u8],
) -> Result<QkExpertRow32CudaStagingTensorV2, HotLayoutError> {
    if rows == 0
        || !rows.is_multiple_of(EXPERT_ROW32_CUDA_V2_ROWS_PER_GROUP)
        || cols == 0
        || !cols.is_multiple_of(256)
        || !block_bytes.is_multiple_of(EXPERT_ROW32_CUDA_V2_VECTOR_BYTES)
    {
        return Err(HotLayoutError::BadShape);
    }
    let blocks_per_row = cols / 256;
    let vectors_per_block = block_bytes / EXPERT_ROW32_CUDA_V2_VECTOR_BYTES;
    let expected = rows
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or(HotLayoutError::BadShape)?;
    if source.len() != expected {
        return Err(HotLayoutError::BadPayloadBytes {
            expected,
            actual: source.len(),
        });
    }

    let mut output = vec![0_u8; expected];
    for row in 0..rows {
        for block_index in 0..blocks_per_row {
            for vector in 0..vectors_per_block {
                let source_offset = ((row * blocks_per_row + block_index) * block_bytes)
                    + vector * EXPERT_ROW32_CUDA_V2_VECTOR_BYTES;
                let destination_offset = ((((row / EXPERT_ROW32_CUDA_V2_ROWS_PER_GROUP)
                    * blocks_per_row
                    + block_index)
                    * vectors_per_block
                    + vector)
                    * EXPERT_ROW32_CUDA_V2_ROWS_PER_GROUP
                    + (row % EXPERT_ROW32_CUDA_V2_ROWS_PER_GROUP))
                    * EXPERT_ROW32_CUDA_V2_VECTOR_BYTES;
                output[destination_offset..destination_offset + EXPERT_ROW32_CUDA_V2_VECTOR_BYTES]
                    .copy_from_slice(
                        &source[source_offset..source_offset + EXPERT_ROW32_CUDA_V2_VECTOR_BYTES],
                    );
            }
        }
    }
    QkExpertRow32CudaStagingTensorV2::new(layout, rows, cols, output)
}

fn deinterleave(source: &QkExpertRow32CudaStagingTensorV2) -> Result<Vec<u8>, HotLayoutError> {
    let mut output = vec![0_u8; source.payload_len_bytes()];
    for row in 0..source.rows() {
        for block_index in 0..source.blocks_per_row() {
            for vector in 0..source.vectors_per_block() {
                let source_vector = source
                    .vector_bytes(row, block_index, vector)
                    .ok_or(HotLayoutError::BadShape)?;
                let destination_offset = ((row * source.blocks_per_row() + block_index)
                    * source.block_bytes())
                    + vector * EXPERT_ROW32_CUDA_V2_VECTOR_BYTES;
                output[destination_offset..destination_offset + EXPERT_ROW32_CUDA_V2_VECTOR_BYTES]
                    .copy_from_slice(source_vector);
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use augur_kernel_api::{
        Q4K_HOT_CUDA_V1_RESERVED_OFFSET, Q5K_HOT_CUDA_V1_BLOCK_BYTES, Q5K_HOT_CUDA_V1_QS_OFFSET,
    };

    use super::*;

    fn q4_source(rows: usize, cols: usize) -> Q4KHotCudaStagingTensorV1 {
        let mut bytes = fixture_bytes(rows, cols, Q4K_HOT_CUDA_V1_BLOCK_BYTES);
        canonicalize_reserved(&mut bytes, Q4K_HOT_CUDA_V1_BLOCK_BYTES);
        Q4KHotCudaStagingTensorV1::new(rows, cols, bytes).unwrap()
    }

    fn q5_source(rows: usize, cols: usize) -> QkHotCudaStagingTensorV1 {
        let mut bytes = fixture_bytes(rows, cols, Q5K_HOT_CUDA_V1_BLOCK_BYTES);
        canonicalize_reserved(&mut bytes, Q5K_HOT_CUDA_V1_BLOCK_BYTES);
        QkHotCudaStagingTensorV1::new(
            PhysicalLayoutId::Q5K_HOT_CUDA_V1,
            Q5K_HOT_CUDA_V1_BLOCK_BYTES,
            Q5K_HOT_CUDA_V1_COMPACT_BYTES,
            rows,
            cols,
            bytes,
        )
        .unwrap()
    }

    fn fixture_bytes(rows: usize, cols: usize, block_bytes: usize) -> Vec<u8> {
        let len = rows * (cols / 256) * block_bytes;
        (0..len)
            .map(|index| ((index.wrapping_mul(131).wrapping_add(17)) % 251) as u8)
            .collect()
    }

    fn canonicalize_reserved(bytes: &mut [u8], block_bytes: usize) {
        for block in bytes.chunks_exact_mut(block_bytes) {
            block[Q4K_HOT_CUDA_V1_RESERVED_OFFSET..32].fill(0);
        }
    }

    fn assert_q4_roundtrip(rows: usize, cols: usize) {
        let source = q4_source(rows, cols);
        let row32 = repack_q4k_hot_cuda_v1_to_expert_row32_v2(&source).unwrap();
        let restored = deinterleave_q4k_expert_row32_v2_to_hot_cuda_v1(&row32).unwrap();
        assert_eq!(row32.layout(), PhysicalLayoutId::Q4K_EXPERT_ROW32_CUDA_V2);
        assert_eq!(row32.payload_len_bytes(), source.payload_len_bytes());
        assert_eq!(restored.payload_bytes(), source.payload_bytes());
    }

    fn assert_q5_roundtrip(rows: usize, cols: usize) {
        let source = q5_source(rows, cols);
        let row32 = repack_q5k_hot_cuda_v1_to_expert_row32_v2(&source).unwrap();
        let restored = deinterleave_q5k_expert_row32_v2_to_hot_cuda_v1(&row32).unwrap();
        assert_eq!(row32.layout(), PhysicalLayoutId::Q5K_EXPERT_ROW32_CUDA_V2);
        assert_eq!(row32.payload_len_bytes(), source.payload_bytes().len());
        assert_eq!(restored.payload_bytes(), source.payload_bytes());
    }

    #[test]
    fn q4_roundtrips_boundary_and_real_input_shapes() {
        assert_q4_roundtrip(32, 256);
        assert_q4_roundtrip(64, 512);
        assert_q4_roundtrip(2048, 4096);
    }

    #[test]
    fn q5_roundtrips_boundary_and_real_down_shapes() {
        assert_q5_roundtrip(32, 256);
        assert_q5_roundtrip(64, 512);
        assert_q5_roundtrip(4096, 2048);
    }

    #[test]
    fn q5_trailing_vector_is_quant_payload_not_padding() {
        let source = q5_source(32, 256);
        let source_tail = &source.payload_bytes()[Q5K_HOT_CUDA_V1_BLOCK_BYTES
            - EXPERT_ROW32_CUDA_V2_VECTOR_BYTES
            ..Q5K_HOT_CUDA_V1_BLOCK_BYTES];
        assert!(source_tail.iter().any(|byte| *byte != 0));
        assert_eq!(Q5K_HOT_CUDA_V1_QS_OFFSET, 64);

        let row32 = repack_q5k_hot_cuda_v1_to_expert_row32_v2(&source).unwrap();
        assert_eq!(
            row32.vector_bytes(0, 0, row32.vectors_per_block() - 1),
            Some(source_tail),
        );
        let restored = deinterleave_q5k_expert_row32_v2_to_hot_cuda_v1(&row32).unwrap();
        assert_eq!(restored.payload_bytes(), source.payload_bytes());
    }

    #[test]
    fn offset_order_is_row_group_block_vector_lane() {
        let source = q4_source(64, 512);
        let row32 = repack_q4k_hot_cuda_v1_to_expert_row32_v2(&source).unwrap();
        for row in [0, 1, 31, 32, 63] {
            for block in 0..2 {
                for vector in [0, 4, 9] {
                    let source_offset = ((row * 2 + block) * Q4K_HOT_CUDA_V1_BLOCK_BYTES)
                        + vector * EXPERT_ROW32_CUDA_V2_VECTOR_BYTES;
                    assert_eq!(
                        row32.vector_bytes(row, block, vector).unwrap(),
                        &source.payload_bytes()
                            [source_offset..source_offset + EXPERT_ROW32_CUDA_V2_VECTOR_BYTES],
                    );
                }
            }
        }
    }

    #[test]
    fn repack_and_projection_mismatch_fail_closed() {
        let q4 = q4_source(32, 256);
        let q5 = q5_source(32, 256);
        let row32_q4 = repack_q4k_hot_cuda_v1_to_expert_row32_v2(&q4).unwrap();
        let row32_q5 = repack_q5k_hot_cuda_v1_to_expert_row32_v2(&q5).unwrap();
        assert_eq!(
            deinterleave_q4k_expert_row32_v2_to_hot_cuda_v1(&row32_q5),
            Err(HotLayoutError::BadShape),
        );
        assert_eq!(
            deinterleave_q5k_expert_row32_v2_to_hot_cuda_v1(&row32_q4),
            Err(HotLayoutError::BadShape),
        );

        let q6 = QkHotCudaStagingTensorV1::new(
            PhysicalLayoutId::Q6K_HOT_CUDA_V1,
            224,
            210,
            32,
            256,
            vec![0; 32 * 224],
        )
        .unwrap();
        assert_eq!(
            repack_q5k_hot_cuda_v1_to_expert_row32_v2(&q6),
            Err(HotLayoutError::BadShape),
        );
    }

    #[test]
    fn row_group_extent_and_reserved_drift_fail_closed() {
        let source = q4_source(32, 256);
        let row32 = repack_q4k_hot_cuda_v1_to_expert_row32_v2(&source).unwrap();
        let mut drifted = row32.payload_bytes().to_vec();
        assert_eq!(
            QkExpertRow32CudaStagingTensorV2::new(
                PhysicalLayoutId::Q4K_EXPERT_ROW32_CUDA_V2,
                64,
                256,
                drifted.clone(),
            ),
            Err(HotLayoutError::BadPayloadBytes {
                expected: 64 * Q4K_HOT_CUDA_V1_BLOCK_BYTES,
                actual: 32 * Q4K_HOT_CUDA_V1_BLOCK_BYTES,
            }),
        );

        let reserved = row32.vector_offset(1, 0, 1).unwrap() + 4;
        drifted[reserved] = 1;
        assert_eq!(
            QkExpertRow32CudaStagingTensorV2::new(
                PhysicalLayoutId::Q4K_EXPERT_ROW32_CUDA_V2,
                32,
                256,
                drifted,
            ),
            Err(HotLayoutError::BadReservedBytes { block: 1 }),
        );
    }

    #[test]
    fn q5_reserved_drift_fails_without_rejecting_trailing_payload() {
        let source = q5_source(32, 256);
        let row32 = repack_q5k_hot_cuda_v1_to_expert_row32_v2(&source).unwrap();
        let mut drifted = row32.payload_bytes().to_vec();
        let reserved = row32.vector_offset(7, 0, 1).unwrap() + 4;
        drifted[reserved] = 1;
        assert_eq!(
            QkExpertRow32CudaStagingTensorV2::new(
                PhysicalLayoutId::Q5K_EXPERT_ROW32_CUDA_V2,
                32,
                256,
                drifted,
            ),
            Err(HotLayoutError::BadReservedBytes { block: 7 }),
        );
    }
}
