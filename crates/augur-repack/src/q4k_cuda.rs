use augur_kernel_api::{
    HotLayoutError, Q4KHotCpuBlock, Q4KHotCpuTensor, Q4KHotCudaStagingTensorV1,
    Q4K_HOT_CUDA_V1_BLOCK_BYTES, Q4K_HOT_CUDA_V1_DECODED_LEN, Q4K_HOT_CUDA_V1_DMIN_OFFSET,
    Q4K_HOT_CUDA_V1_D_OFFSET, Q4K_HOT_CUDA_V1_GROUPS, Q4K_HOT_CUDA_V1_MIN_OFFSET,
    Q4K_HOT_CUDA_V1_QS_BYTES, Q4K_HOT_CUDA_V1_QS_OFFSET, Q4K_HOT_CUDA_V1_RESERVED_BYTES,
    Q4K_HOT_CUDA_V1_RESERVED_OFFSET, Q4K_HOT_CUDA_V1_SCALE_OFFSET,
};

/// Serializes an already-decoded Q4K CPU Hot tensor into the independent
/// `Q4K_HOT_CUDA_V1` byte protocol.
///
/// This is a staging transformation, not a Rust ABI cast. No bytes from the
/// memory representation of `Q4KHotCpuBlock` are copied directly.
///
/// The CPU Hot tensor is currently used as the validated semantic source
/// because it already carries unpacked scale/min metadata. A future direct
/// Compact -> CUDA repacker must produce byte-identical output.
pub fn repack_q4k_hot_cpu_to_cuda_v1(
    src: &Q4KHotCpuTensor,
) -> Result<Q4KHotCudaStagingTensorV1, HotLayoutError> {
    if src.rows() == 0
        || src.cols() == 0
        || src.cols()
            != src
                .blocks_per_row()
                .checked_mul(Q4K_HOT_CUDA_V1_DECODED_LEN)
                .ok_or(HotLayoutError::BadShape)?
    {
        return Err(HotLayoutError::BadShape);
    }

    let expected_blocks = src
        .rows()
        .checked_mul(src.blocks_per_row())
        .ok_or(HotLayoutError::BadShape)?;

    if src.blocks().len() != expected_blocks {
        return Err(HotLayoutError::BadBlockCount {
            expected: expected_blocks,
            actual: src.blocks().len(),
        });
    }

    let payload_len = expected_blocks
        .checked_mul(Q4K_HOT_CUDA_V1_BLOCK_BYTES)
        .ok_or(HotLayoutError::BadShape)?;

    let mut bytes = vec![0_u8; payload_len];

    for (block_index, block) in src.blocks().iter().enumerate() {
        let start = block_index * Q4K_HOT_CUDA_V1_BLOCK_BYTES;
        let dst = &mut bytes[start..start + Q4K_HOT_CUDA_V1_BLOCK_BYTES];
        serialize_block(block, dst);
    }

    Q4KHotCudaStagingTensorV1::new(src.rows(), src.cols(), bytes)
}

pub(crate) fn serialize_block(block: &Q4KHotCpuBlock, dst: &mut [u8]) {
    debug_assert_eq!(dst.len(), Q4K_HOT_CUDA_V1_BLOCK_BYTES);
    debug_assert_eq!(block.scale.len(), Q4K_HOT_CUDA_V1_GROUPS);
    debug_assert_eq!(block.min.len(), Q4K_HOT_CUDA_V1_GROUPS);
    debug_assert_eq!(block.qs.len(), Q4K_HOT_CUDA_V1_QS_BYTES);

    dst[Q4K_HOT_CUDA_V1_D_OFFSET..Q4K_HOT_CUDA_V1_D_OFFSET + 2]
        .copy_from_slice(&block.d.to_bits().to_le_bytes());

    dst[Q4K_HOT_CUDA_V1_DMIN_OFFSET..Q4K_HOT_CUDA_V1_DMIN_OFFSET + 2]
        .copy_from_slice(&block.dmin.to_bits().to_le_bytes());

    dst[Q4K_HOT_CUDA_V1_SCALE_OFFSET..Q4K_HOT_CUDA_V1_SCALE_OFFSET + Q4K_HOT_CUDA_V1_GROUPS]
        .copy_from_slice(&block.scale);

    dst[Q4K_HOT_CUDA_V1_MIN_OFFSET..Q4K_HOT_CUDA_V1_MIN_OFFSET + Q4K_HOT_CUDA_V1_GROUPS]
        .copy_from_slice(&block.min);

    dst[Q4K_HOT_CUDA_V1_RESERVED_OFFSET
        ..Q4K_HOT_CUDA_V1_RESERVED_OFFSET + Q4K_HOT_CUDA_V1_RESERVED_BYTES]
        .fill(0);

    dst[Q4K_HOT_CUDA_V1_QS_OFFSET..Q4K_HOT_CUDA_V1_QS_OFFSET + Q4K_HOT_CUDA_V1_QS_BYTES]
        .copy_from_slice(&block.qs);
}

#[cfg(test)]
mod tests {
    use augur_core::{F16Bits, PhysicalLayoutId};
    use augur_kernel_api::{
        Q4KHotCpuBlock, Q4KHotCpuTensor, Q4KHotCudaStagingTensorV1, Q4K_HOT_CUDA_V1_BLOCK_BYTES,
        Q4K_HOT_CUDA_V1_DMIN_OFFSET, Q4K_HOT_CUDA_V1_D_OFFSET, Q4K_HOT_CUDA_V1_MIN_OFFSET,
        Q4K_HOT_CUDA_V1_QS_OFFSET, Q4K_HOT_CUDA_V1_RESERVED_BYTES, Q4K_HOT_CUDA_V1_RESERVED_OFFSET,
        Q4K_HOT_CUDA_V1_SCALE_OFFSET,
    };

    use super::*;

    fn fixture_block(seed: u8) -> Q4KHotCpuBlock {
        let mut scale = [0_u8; 8];
        let mut min = [0_u8; 8];
        let mut qs = [0_u8; 128];

        for (index, value) in scale.iter_mut().enumerate() {
            *value = seed.wrapping_add(index as u8);
        }
        for (index, value) in min.iter_mut().enumerate() {
            *value = seed.wrapping_add(16).wrapping_add(index as u8);
        }
        for (index, value) in qs.iter_mut().enumerate() {
            *value = seed.wrapping_add(32).wrapping_add(index as u8);
        }

        Q4KHotCpuBlock {
            d: F16Bits::from_f32(1.5),
            dmin: F16Bits::from_f32(0.25),
            scale,
            min,
            qs,
        }
    }

    #[test]
    fn cuda_layout_offsets_are_exact() {
        let block = fixture_block(3);
        let tensor =
            Q4KHotCpuTensor::new(1, Q4KHotCpuBlock::DECODED_LEN, vec![block.clone()]).unwrap();

        let staging = repack_q4k_hot_cpu_to_cuda_v1(&tensor).unwrap();
        let bytes = staging.block_bytes(0).unwrap();

        assert_eq!(staging.payload_len_bytes(), Q4K_HOT_CUDA_V1_BLOCK_BYTES);
        assert_eq!(
            Q4KHotCudaStagingTensorV1::PHYSICAL_LAYOUT,
            PhysicalLayoutId::Q4K_HOT_CUDA_V1,
        );

        assert_eq!(
            &bytes[Q4K_HOT_CUDA_V1_D_OFFSET..Q4K_HOT_CUDA_V1_D_OFFSET + 2],
            &block.d.to_bits().to_le_bytes(),
        );
        assert_eq!(
            &bytes[Q4K_HOT_CUDA_V1_DMIN_OFFSET..Q4K_HOT_CUDA_V1_DMIN_OFFSET + 2],
            &block.dmin.to_bits().to_le_bytes(),
        );
        assert_eq!(
            &bytes[Q4K_HOT_CUDA_V1_SCALE_OFFSET..Q4K_HOT_CUDA_V1_SCALE_OFFSET + 8],
            &block.scale,
        );
        assert_eq!(
            &bytes[Q4K_HOT_CUDA_V1_MIN_OFFSET..Q4K_HOT_CUDA_V1_MIN_OFFSET + 8],
            &block.min,
        );
        assert_eq!(&bytes[Q4K_HOT_CUDA_V1_QS_OFFSET..], &block.qs,);
    }

    #[test]
    fn reserved_padding_is_canonical_zero() {
        let tensor = Q4KHotCpuTensor::new(1, 256, vec![fixture_block(9)]).unwrap();
        let staging = repack_q4k_hot_cpu_to_cuda_v1(&tensor).unwrap();
        let block = staging.block_bytes(0).unwrap();

        let end = Q4K_HOT_CUDA_V1_RESERVED_OFFSET + Q4K_HOT_CUDA_V1_RESERVED_BYTES;
        assert!(block[Q4K_HOT_CUDA_V1_RESERVED_OFFSET..end]
            .iter()
            .all(|&byte| byte == 0));
    }

    #[test]
    fn tensor_uses_row_major_block_order() {
        let blocks = vec![
            fixture_block(1),
            fixture_block(2),
            fixture_block(3),
            fixture_block(4),
        ];
        let tensor = Q4KHotCpuTensor::new(2, 512, blocks.clone()).unwrap();
        let staging = repack_q4k_hot_cpu_to_cuda_v1(&tensor).unwrap();

        assert_eq!(staging.rows(), 2);
        assert_eq!(staging.cols(), 512);
        assert_eq!(staging.blocks_per_row(), 2);
        assert_eq!(staging.block_count(), 4);

        for (index, source) in blocks.iter().enumerate() {
            let serialized = staging.block_bytes(index).unwrap();
            assert_eq!(&serialized[Q4K_HOT_CUDA_V1_QS_OFFSET..], &source.qs,);
        }
    }

    #[test]
    fn staging_constructor_rejects_nonzero_reserved_bytes() {
        let mut bytes = vec![0_u8; Q4K_HOT_CUDA_V1_BLOCK_BYTES];
        bytes[Q4K_HOT_CUDA_V1_RESERVED_OFFSET] = 1;

        assert_eq!(
            Q4KHotCudaStagingTensorV1::new(1, 256, bytes),
            Err(HotLayoutError::BadReservedBytes { block: 0 }),
        );
    }

    #[test]
    fn staging_constructor_rejects_wrong_payload_length() {
        let actual = Q4K_HOT_CUDA_V1_BLOCK_BYTES - 1;
        assert_eq!(
            Q4KHotCudaStagingTensorV1::new(1, 256, vec![0_u8; actual]),
            Err(HotLayoutError::BadPayloadBytes {
                expected: Q4K_HOT_CUDA_V1_BLOCK_BYTES,
                actual,
            }),
        );
    }

    #[test]
    fn payload_is_not_cpu_struct_byte_image() {
        assert_ne!(
            core::mem::size_of::<Q4KHotCpuBlock>(),
            Q4K_HOT_CUDA_V1_BLOCK_BYTES,
            "CUDA protocol must not silently become the Rust CPU struct ABI",
        );
    }
}
