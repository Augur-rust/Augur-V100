use core::mem::{align_of, size_of};

use augur_core::F16Bits;
use augur_formats::ReferenceDecode;
use augur_formats::{
    pack_q4k_scales_ref, quantize_q4k_block_ref, unpack_q4k_scales_ref, FormatError,
    Q4KCompactBlock, Q4KScales, Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE, Q4K_QS_BYTES, Q4K_SCALE_BYTES,
};

#[test]
fn q4k_compact_block_layout_size() {
    assert_eq!(Q4K_SCALE_BYTES, 12);
    assert_eq!(Q4K_QS_BYTES, 128);
    assert_eq!(Q4K_BLOCK_SIZE, 256);

    assert_eq!(size_of::<Q4KCompactBlock>(), 144);
    assert_eq!(Q4K_BLOCK_BYTES, 144);
    assert_eq!(align_of::<Q4KCompactBlock>(), 2);
}

#[test]
fn llama_cpp_b10001_q4k_packing_vector_is_bit_exact() {
    let source: [f32; Q4K_BLOCK_SIZE] = core::array::from_fn(|index| (index % 16) as f32);
    let block = quantize_q4k_block_ref(&source).unwrap();
    assert_eq!(block.d.to_bits(), 0x2410);
    assert_eq!(block.dmin.to_bits(), 0);
    assert_eq!(
        block.scales,
        [0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x0f, 0x0f, 0x0f]
    );
    let expected_qs: [u8; Q4K_QS_BYTES] = core::array::from_fn(|index| ((index % 16) as u8) * 0x11);
    assert_eq!(block.qs, expected_qs);
    assert_eq!(
        Q4KCompactBlock::from_le_bytes(&block.to_le_bytes()),
        Ok(block.clone())
    );

    let mut decoded = [0.0_f32; Q4K_BLOCK_SIZE];
    block.decode_ref(&mut decoded).unwrap();
    assert_eq!(decoded[0], 0.0);
    assert_eq!(decoded[31], block.d.to_f32() * 63.0 * 15.0);
    assert_eq!(decoded[32], 0.0);
    assert_eq!(decoded[63], block.d.to_f32() * 63.0 * 15.0);
}

#[test]
fn llama_cpp_b10001_q4k_min_vector_is_bit_exact() {
    let block = quantize_q4k_block_ref(&[-1.0; Q4K_BLOCK_SIZE]).unwrap();
    assert_eq!(block.d.to_bits(), 0);
    assert_eq!(block.dmin.to_bits(), 0x2410);
    assert_eq!(
        block.scales,
        [0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xf0, 0xf0, 0xf0, 0xf0]
    );
    assert_eq!(block.qs, [0; Q4K_QS_BYTES]);
}

#[test]
fn invalid_q4k_source_is_rejected() {
    assert_eq!(
        quantize_q4k_block_ref(&[0.0; Q4K_BLOCK_SIZE - 1]),
        Err(FormatError::InputLengthMismatch {
            expected: Q4K_BLOCK_SIZE,
            actual: Q4K_BLOCK_SIZE - 1,
        })
    );
    let mut source = [0.0; Q4K_BLOCK_SIZE];
    source[17] = f32::NAN;
    assert_eq!(
        quantize_q4k_block_ref(&source),
        Err(FormatError::NonFiniteInput { index: 17 })
    );
}

#[test]
fn unpack_q4k_scales_ref_known_values() {
    let decoded = Q4KScales {
        scale: [1, 2, 3, 4, 17, 18, 35, 63],
        min: [5, 6, 7, 8, 9, 26, 42, 61],
    };
    let packed = pack_q4k_scales_ref(decoded).unwrap();
    assert_eq!(
        packed,
        [0x41, 0x42, 0x83, 0xc4, 0x05, 0x46, 0x87, 0xc8, 0x91, 0xa2, 0xa3, 0xdf,]
    );
    assert_eq!(unpack_q4k_scales_ref(&packed), decoded);
}

#[test]
fn unpack_q4k_scales_ref_roundtrip_pack() {
    let decoded = Q4KScales {
        scale: [0, 1, 2, 3, 4, 15, 32, 63],
        min: [63, 32, 15, 4, 3, 2, 1, 0],
    };
    let packed = pack_q4k_scales_ref(decoded).unwrap();
    let unpacked = unpack_q4k_scales_ref(&packed);
    assert_eq!(unpacked, decoded);
}

#[test]
fn pack_q4k_scales_rejects_out_of_range_scale() {
    let decoded = Q4KScales {
        scale: [0, 1, 2, 3, 4, 15, 32, 64],
        min: [0; 8],
    };
    assert_eq!(
        pack_q4k_scales_ref(decoded),
        Err(FormatError::ScaleOutOfRange {
            index: 7,
            value: 64
        })
    );
}

#[test]
fn pack_q4k_scales_rejects_out_of_range_min() {
    let decoded = Q4KScales {
        scale: [0; 8],
        min: [0, 1, 2, 3, 4, 15, 32, 64],
    };
    assert_eq!(
        pack_q4k_scales_ref(decoded),
        Err(FormatError::MinOutOfRange {
            index: 7,
            value: 64
        })
    );
}

#[test]
fn q4k_decode_ref_rejects_short_output() {
    let block = Q4KCompactBlock {
        d: F16Bits::ZERO,
        dmin: F16Bits::ZERO,
        scales: [0; 12],
        qs: [0; 128],
    };
    let mut out = vec![0.0f32; 255];
    assert_eq!(
        block.decode_ref(&mut out),
        Err(FormatError::OutputTooSmall {
            expected: 256,
            actual: 255
        })
    );
}

#[test]
fn q4k_decode_ref_zero_block_outputs_zero() {
    let block = Q4KCompactBlock {
        d: F16Bits::ZERO,
        dmin: F16Bits::ZERO,
        scales: [0; 12],
        qs: [0; 128],
    };
    let mut out = vec![1.0f32; 256];
    block.decode_ref(&mut out).unwrap();
    assert!(out.iter().all(|&v| v == 0.0));
}

#[test]
fn f16_bits_to_f32_basic_values() {
    assert_eq!(F16Bits::from_bits(0x0000).to_f32(), 0.0);
    assert_eq!(F16Bits::from_bits(0x3c00).to_f32(), 1.0);
    assert_eq!(F16Bits::from_bits(0xc000).to_f32(), -2.0);
}
