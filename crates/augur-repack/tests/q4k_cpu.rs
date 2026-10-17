use augur_core::F16Bits;
use augur_formats::{pack_q4k_scales_ref, Q4KCompactBlock, Q4KScales, ReferenceDecode};
use augur_kernel_api::Q4KHotCpuBlock;
use augur_repack::{decode_q4k_hot_cpu_block_ref, repack_q4k_block_cpu_ref};

fn sample_block() -> Q4KCompactBlock {
    let scales = Q4KScales {
        scale: [1, 2, 3, 4, 17, 18, 35, 63],
        min: [5, 6, 7, 8, 9, 26, 42, 61],
    };

    let mut qs = [0u8; 128];
    for (i, q) in qs.iter_mut().enumerate() {
        let lo = (i as u8) & 0x0f;
        let hi = ((127 - i) as u8) & 0x0f;
        *q = lo | (hi << 4);
    }

    Q4KCompactBlock {
        d: F16Bits::from_bits(0x3c00),    // 1.0
        dmin: F16Bits::from_bits(0x3800), // 0.5
        scales: pack_q4k_scales_ref(scales).unwrap(),
        qs,
    }
}

#[test]
fn repack_q4k_cpu_predecodes_scales() {
    let block = sample_block();
    let hot = repack_q4k_block_cpu_ref(&block);
    assert_eq!(hot.scale, [1, 2, 3, 4, 17, 18, 35, 63]);
    assert_eq!(hot.min, [5, 6, 7, 8, 9, 26, 42, 61]);
}

#[test]
fn repack_q4k_cpu_preserves_qs_and_factors() {
    let block = sample_block();
    let hot = repack_q4k_block_cpu_ref(&block);
    assert_eq!(hot.d, block.d);
    assert_eq!(hot.dmin, block.dmin);
    assert_eq!(hot.qs, block.qs);
}

#[test]
fn q4k_hot_cpu_decode_matches_compact_decode() {
    let block = sample_block();
    let hot = repack_q4k_block_cpu_ref(&block);

    let mut compact_out = vec![0.0f32; Q4KCompactBlock::DECODED_LEN];
    let mut hot_out = vec![0.0f32; Q4KHotCpuBlock::DECODED_LEN];

    block.decode_ref(&mut compact_out).unwrap();
    decode_q4k_hot_cpu_block_ref(&hot, &mut hot_out).unwrap();

    assert_eq!(hot_out, compact_out);
}

#[test]
fn q4k_hot_cpu_decode_rejects_short_output() {
    let block = sample_block();
    let hot = repack_q4k_block_cpu_ref(&block);
    let mut out = vec![0.0f32; Q4KHotCpuBlock::DECODED_LEN - 1];
    assert!(decode_q4k_hot_cpu_block_ref(&hot, &mut out).is_err());
}
