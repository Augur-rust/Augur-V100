use augur_core::PhysicalLayoutId;
use augur_formats::{
    unpack_q4k_scales_ref, Q4KCompactBlock, Q5KCompactBlock, Q4K_BLOCK_BYTES, Q5K_BLOCK_BYTES,
    Q6K_BLOCK_BYTES,
};
use augur_kernel_api::{
    Q4K_HOT_CUDA_V1_BLOCK_BYTES, Q5K_HOT_CUDA_V1_BLOCK_BYTES, Q6K_HOT_CUDA_V1_BLOCK_BYTES,
};
use augur_repack::{
    repack_q4k_compact_block_range_to_cuda_v1_into, repack_q4k_compact_to_cuda_v1,
    repack_q4k_compact_to_cuda_v1_into, repack_q5k_compact_block_range_to_cuda_v1_into,
    repack_q5k_compact_to_cuda_v1, repack_q5k_compact_to_cuda_v1_into,
    repack_q6k_compact_to_cuda_v1,
};

#[test]
fn direct_q4_repack_expands_metadata_and_zeroes_padding() {
    let compact = vec![0_u8; Q4K_BLOCK_BYTES];
    let hot = repack_q4k_compact_to_cuda_v1(&compact, 1, 256).unwrap();
    assert_eq!(hot.payload_bytes().len(), Q4K_HOT_CUDA_V1_BLOCK_BYTES);
    assert!(hot.payload_bytes().iter().all(|byte| *byte == 0));
}

#[test]
fn q5_and_q6_repack_preserve_fields_and_zero_reserved_bytes() {
    let mut q5 = vec![0_u8; Q5K_BLOCK_BYTES];
    q5[..4].copy_from_slice(&[1, 2, 3, 4]);
    q5[4..16].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    q5[16] = 0x34;
    q5[48] = 0x21;
    let hot5 = repack_q5k_compact_to_cuda_v1(&q5, 1, 256).unwrap();
    assert_eq!(hot5.layout(), PhysicalLayoutId::Q5K_HOT_CUDA_V1);
    assert_eq!(&hot5.payload_bytes()[..4], &q5[..4]);
    let metadata = augur_formats::unpack_q4k_scales_ref(q5[4..16].try_into().unwrap());
    assert_eq!(&hot5.payload_bytes()[4..12], &metadata.scale);
    assert_eq!(&hot5.payload_bytes()[12..20], &metadata.min);
    assert_eq!(&hot5.payload_bytes()[32..64], &q5[16..48]);
    assert_eq!(&hot5.payload_bytes()[64..192], &q5[48..176]);
    assert!(hot5.payload_bytes()[20..32].iter().all(|byte| *byte == 0));
    assert_eq!(hot5.payload_bytes().len(), Q5K_HOT_CUDA_V1_BLOCK_BYTES);

    let mut q6 = vec![0_u8; Q6K_BLOCK_BYTES];
    q6[7] = 0x43;
    let hot6 = repack_q6k_compact_to_cuda_v1(&q6, 1, 256).unwrap();
    assert_eq!(hot6.layout(), PhysicalLayoutId::Q6K_HOT_CUDA_V1);
    assert_eq!(&hot6.payload_bytes()[..Q6K_BLOCK_BYTES], q6);
    assert!(hot6.payload_bytes()[Q6K_BLOCK_BYTES..]
        .iter()
        .all(|byte| *byte == 0));
    assert_eq!(hot6.payload_bytes().len(), Q6K_HOT_CUDA_V1_BLOCK_BYTES);
}

#[test]
fn repack_rejects_shape_and_payload_drift() {
    assert!(repack_q5k_compact_to_cuda_v1(&[0; Q5K_BLOCK_BYTES], 1, 255).is_err());
    assert!(repack_q6k_compact_to_cuda_v1(&[0; Q6K_BLOCK_BYTES - 1], 1, 256).is_err());
}

#[test]
fn direct_into_q4_and_q5_match_owned_payloads_byte_for_byte() {
    let q4 = (0..Q4K_BLOCK_BYTES)
        .map(|index| (index as u8).wrapping_mul(17))
        .collect::<Vec<_>>();
    let expected_q4 = repack_q4k_compact_to_cuda_v1(&q4, 1, 256).unwrap();
    let mut actual_q4 = vec![0xa5; Q4K_HOT_CUDA_V1_BLOCK_BYTES];
    repack_q4k_compact_to_cuda_v1_into(&q4, 1, 256, &mut actual_q4).unwrap();
    assert_eq!(actual_q4, expected_q4.payload_bytes());

    let q5 = (0..Q5K_BLOCK_BYTES)
        .map(|index| (index as u8).wrapping_mul(29))
        .collect::<Vec<_>>();
    let expected_q5 = repack_q5k_compact_to_cuda_v1(&q5, 1, 256).unwrap();
    let mut actual_q5 = vec![0xa5; Q5K_HOT_CUDA_V1_BLOCK_BYTES];
    repack_q5k_compact_to_cuda_v1_into(&q5, 1, 256, &mut actual_q5).unwrap();
    assert_eq!(actual_q5, expected_q5.payload_bytes());
}

#[test]
fn direct_into_rejects_destination_extent_drift() {
    let q4 = vec![0; Q4K_BLOCK_BYTES];
    assert!(repack_q4k_compact_to_cuda_v1_into(
        &q4,
        1,
        256,
        &mut [0; Q4K_HOT_CUDA_V1_BLOCK_BYTES - 1],
    )
    .is_err());
}

fn q4_reference_oracle(compact: &[u8]) -> Vec<u8> {
    let mut output = vec![0_u8; compact.len() / Q4K_BLOCK_BYTES * Q4K_HOT_CUDA_V1_BLOCK_BYTES];
    for (source, destination) in compact
        .chunks_exact(Q4K_BLOCK_BYTES)
        .zip(output.chunks_exact_mut(Q4K_HOT_CUDA_V1_BLOCK_BYTES))
    {
        let block = Q4KCompactBlock::from_le_bytes(source).unwrap();
        let metadata = unpack_q4k_scales_ref(&block.scales);
        destination[..2].copy_from_slice(&block.d.to_bits().to_le_bytes());
        destination[2..4].copy_from_slice(&block.dmin.to_bits().to_le_bytes());
        destination[4..12].copy_from_slice(&metadata.scale);
        destination[12..20].copy_from_slice(&metadata.min);
        destination[32..].copy_from_slice(&block.qs);
    }
    output
}

fn q5_reference_oracle(compact: &[u8]) -> Vec<u8> {
    let mut output = vec![0_u8; compact.len() / Q5K_BLOCK_BYTES * Q5K_HOT_CUDA_V1_BLOCK_BYTES];
    for (source, destination) in compact
        .chunks_exact(Q5K_BLOCK_BYTES)
        .zip(output.chunks_exact_mut(Q5K_HOT_CUDA_V1_BLOCK_BYTES))
    {
        let block = Q5KCompactBlock::from_le_bytes(source).unwrap();
        let metadata = unpack_q4k_scales_ref(&block.scales);
        destination[..2].copy_from_slice(&block.d.to_bits().to_le_bytes());
        destination[2..4].copy_from_slice(&block.dmin.to_bits().to_le_bytes());
        destination[4..12].copy_from_slice(&metadata.scale);
        destination[12..20].copy_from_slice(&metadata.min);
        destination[32..64].copy_from_slice(&block.qh);
        destination[64..].copy_from_slice(&block.qs);
    }
    output
}

fn random_finite_blocks(blocks: usize, block_bytes: usize, mut state: u64) -> Vec<u8> {
    let mut output = vec![0_u8; blocks * block_bytes];
    for byte in &mut output {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = state as u8;
    }
    for block in output.chunks_exact_mut(block_bytes) {
        // Zero is finite; all remaining bytes continue to exercise arbitrary payload bits.
        block[..4].fill(0);
    }
    output
}

#[test]
fn raw_q4_and_q5_repack_match_independent_random_multiblock_oracles() {
    let q4 = random_finite_blocks(257, Q4K_BLOCK_BYTES, 0x9e37_79b9_7f4a_7c15);
    let mut q4_hot = vec![0xa5; 257 * Q4K_HOT_CUDA_V1_BLOCK_BYTES];
    repack_q4k_compact_to_cuda_v1_into(&q4, 1, 257 * 256, &mut q4_hot).unwrap();
    assert_eq!(q4_hot, q4_reference_oracle(&q4));

    let q5 = random_finite_blocks(263, Q5K_BLOCK_BYTES, 0xd1b5_4a32_d192_ed03);
    let mut q5_hot = vec![0xa5; 263 * Q5K_HOT_CUDA_V1_BLOCK_BYTES];
    repack_q5k_compact_to_cuda_v1_into(&q5, 1, 263 * 256, &mut q5_hot).unwrap();
    assert_eq!(q5_hot, q5_reference_oracle(&q5));
}

#[test]
fn raw_repack_rejects_nonfinite_factors_and_rezeroes_reused_destinations() {
    let mut q4 = random_finite_blocks(2, Q4K_BLOCK_BYTES, 7);
    q4[Q4K_BLOCK_BYTES..Q4K_BLOCK_BYTES + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
    let mut q4_hot = vec![0xa5; 2 * Q4K_HOT_CUDA_V1_BLOCK_BYTES];
    assert!(repack_q4k_compact_to_cuda_v1_into(&q4, 1, 512, &mut q4_hot).is_err());

    let q5 = random_finite_blocks(3, Q5K_BLOCK_BYTES, 11);
    let mut q5_hot = vec![0xa5; 3 * Q5K_HOT_CUDA_V1_BLOCK_BYTES];
    repack_q5k_compact_to_cuda_v1_into(&q5, 1, 768, &mut q5_hot).unwrap();
    for block in q5_hot.chunks_exact(Q5K_HOT_CUDA_V1_BLOCK_BYTES) {
        assert!(block[20..32].iter().all(|byte| *byte == 0));
    }
}

#[test]
fn q4_and_q5_block_ranges_stitch_to_the_full_repack_byte_exactly() {
    let q4_blocks = 257;
    let q4 = random_finite_blocks(q4_blocks, Q4K_BLOCK_BYTES, 0x2f6e_2b1d_5a93_77c1);
    let full_q4 = repack_q4k_compact_to_cuda_v1(&q4, 1, q4_blocks * 256).unwrap();
    let mut stitched_q4 = vec![0xa5; full_q4.payload_bytes().len()];
    let q4_ranges = [0..3, 3..64, 64..191, 191..q4_blocks];
    for range in q4_ranges {
        let compact_start = range.start * Q4K_BLOCK_BYTES;
        let compact_end = range.end * Q4K_BLOCK_BYTES;
        let hot_start = range.start * Q4K_HOT_CUDA_V1_BLOCK_BYTES;
        let hot_end = range.end * Q4K_HOT_CUDA_V1_BLOCK_BYTES;
        repack_q4k_compact_block_range_to_cuda_v1_into(
            &q4[compact_start..compact_end],
            range,
            q4_blocks,
            &mut stitched_q4[hot_start..hot_end],
        )
        .unwrap();
    }
    assert_eq!(stitched_q4, full_q4.payload_bytes());

    let q5_blocks = 263;
    let q5 = random_finite_blocks(q5_blocks, Q5K_BLOCK_BYTES, 0x7a4c_8ef1_0e55_19d3);
    let full_q5 = repack_q5k_compact_to_cuda_v1(&q5, 1, q5_blocks * 256).unwrap();
    let mut stitched_q5 = vec![0xa5; full_q5.payload_bytes().len()];
    let q5_ranges = [0..1, 1..32, 32..128, 128..257, 257..q5_blocks];
    for range in q5_ranges {
        let compact_start = range.start * Q5K_BLOCK_BYTES;
        let compact_end = range.end * Q5K_BLOCK_BYTES;
        let hot_start = range.start * Q5K_HOT_CUDA_V1_BLOCK_BYTES;
        let hot_end = range.end * Q5K_HOT_CUDA_V1_BLOCK_BYTES;
        repack_q5k_compact_block_range_to_cuda_v1_into(
            &q5[compact_start..compact_end],
            range,
            q5_blocks,
            &mut stitched_q5[hot_start..hot_end],
        )
        .unwrap();
    }
    assert_eq!(stitched_q5, full_q5.payload_bytes());
}

#[test]
fn block_range_repack_rejects_extent_drift_and_overflow() {
    assert!(repack_q4k_compact_block_range_to_cuda_v1_into(
        &[0; Q4K_BLOCK_BYTES],
        std::ops::Range { start: 2, end: 1 },
        2,
        &mut [0; Q4K_HOT_CUDA_V1_BLOCK_BYTES],
    )
    .is_err());
    assert!(repack_q5k_compact_block_range_to_cuda_v1_into(
        &[0; Q5K_BLOCK_BYTES],
        0..2,
        1,
        &mut [0; Q5K_HOT_CUDA_V1_BLOCK_BYTES],
    )
    .is_err());
    assert!(repack_q4k_compact_block_range_to_cuda_v1_into(
        &[],
        usize::MAX..usize::MAX,
        usize::MAX,
        &mut [],
    )
    .is_err());
    assert!(repack_q4k_compact_block_range_to_cuda_v1_into(
        &[0; Q4K_BLOCK_BYTES],
        usize::MAX - 1..usize::MAX,
        usize::MAX,
        &mut [0; Q4K_HOT_CUDA_V1_BLOCK_BYTES],
    )
    .is_err());
    assert!(repack_q4k_compact_block_range_to_cuda_v1_into(
        &[0; Q4K_BLOCK_BYTES],
        1..2,
        2,
        &mut [0; Q4K_HOT_CUDA_V1_BLOCK_BYTES - 1],
    )
    .is_err());
}
