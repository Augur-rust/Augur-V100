use augur_core::F16Bits;
use augur_formats::{
    encode_dual_scale_block_ref, DualScaleCompactBlock, FormatError, ReferenceDecode,
    DUAL_SCALE_BLOCK_SIZE, DUAL_SCALE_GROUPS,
};

use augur_formats::dual_scale::{
    decode_signed_q4, encode_signed_q4, get_membership_bit, get_q4_code, set_membership_bit,
    set_q4_code, DUAL_SCALE_MEMBERSHIP_BYTES, DUAL_SCALE_QS_BYTES,
};

fn assert_close(a: f32, b: f32, tol: f32) {
    let diff = (a - b).abs();
    assert!(diff <= tol, "a={a} b={b} diff={diff} tol={tol}");
}

#[test]
fn dual_scale_encode_rejects_bad_length() {
    let src = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE - 1];

    assert_eq!(
        encode_dual_scale_block_ref(&src),
        Err(FormatError::InputLengthMismatch {
            expected: DUAL_SCALE_BLOCK_SIZE,
            actual: DUAL_SCALE_BLOCK_SIZE - 1,
        })
    );
}

#[test]
fn dual_scale_encode_rejects_non_finite() {
    let mut src = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];
    src[17] = f32::NAN;

    assert_eq!(
        encode_dual_scale_block_ref(&src),
        Err(FormatError::NonFiniteInput { index: 17 })
    );
}

#[test]
fn dual_scale_zero_block_decodes_zero() {
    let src = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];
    let block = encode_dual_scale_block_ref(&src).unwrap();

    let mut decoded = vec![1.0f32; DUAL_SCALE_BLOCK_SIZE];
    block.decode_ref(&mut decoded).unwrap();

    assert!(decoded.iter().all(|&v| v == 0.0));
    assert_eq!(block.outlier_count, 0);
}

#[test]
fn dual_scale_membership_bit_order_is_lsb_first() {
    let mut membership = [0u8; DUAL_SCALE_MEMBERSHIP_BYTES];

    set_membership_bit(&mut membership, 0, true);
    set_membership_bit(&mut membership, 7, true);
    set_membership_bit(&mut membership, 8, true);

    assert_eq!(membership[0], 0b1000_0001);
    assert_eq!(membership[1], 0b0000_0001);
    assert!(get_membership_bit(&membership, 0));
    assert!(get_membership_bit(&membership, 7));
    assert!(get_membership_bit(&membership, 8));
    assert!(!get_membership_bit(&membership, 1));
}

#[test]
fn dual_scale_signed_q4_roundtrip() {
    for q in -8i8..=7 {
        let code = encode_signed_q4(q);
        assert_eq!(decode_signed_q4(code), q);
    }
}

#[test]
fn dual_scale_q4_nibble_order_even_low_odd_high() {
    let mut qs = [0u8; DUAL_SCALE_QS_BYTES];

    set_q4_code(&mut qs, 0, 0x03);
    set_q4_code(&mut qs, 1, 0x0c);

    assert_eq!(qs[0], 0xc3);
    assert_eq!(get_q4_code(&qs, 0), 0x03);
    assert_eq!(get_q4_code(&qs, 1), 0x0c);
}

#[test]
fn dual_scale_decode_rejects_short_output() {
    let src = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];
    let block = encode_dual_scale_block_ref(&src).unwrap();

    let mut dst = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE - 1];

    assert_eq!(
        block.decode_ref(&mut dst),
        Err(FormatError::OutputTooSmall {
            expected: DUAL_SCALE_BLOCK_SIZE,
            actual: DUAL_SCALE_BLOCK_SIZE - 1,
        })
    );
}

#[test]
fn dual_scale_decode_rejects_bad_outlier_count() {
    let block = DualScaleCompactBlock {
        scale_lo: [F16Bits::ZERO; DUAL_SCALE_GROUPS],
        scale_hi: [F16Bits::ZERO; DUAL_SCALE_GROUPS],
        membership: [0u8; DUAL_SCALE_MEMBERSHIP_BYTES],
        qs: [0u8; DUAL_SCALE_QS_BYTES],
        outlier_count: 3,
        outlier_idx: [0, 1],
        outlier_delta: [F16Bits::ZERO; 2],
    };

    let mut dst = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];

    assert_eq!(block.decode_ref(&mut dst), Err(FormatError::BadBlock));
}

#[test]
fn dual_scale_outlier_delta_reconstructs_selected_value() {
    let mut src = vec![0.25f32; DUAL_SCALE_BLOCK_SIZE];
    src[13] = 32.0;

    let block = encode_dual_scale_block_ref(&src).unwrap();
    assert_eq!(block.outlier_count, 1);
    assert_eq!(block.outlier_idx[0], 13);

    let mut decoded = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];
    block.decode_ref(&mut decoded).unwrap();

    assert_close(decoded[13], src[13], 0.01);
}

#[test]
fn dual_scale_outlier_selection_is_deterministic_by_abs_desc_then_index() {
    let mut src = vec![1.0f32; DUAL_SCALE_BLOCK_SIZE];
    src[42] = -50.0;
    src[7] = 50.0;
    src[99] = 60.0;

    let block = encode_dual_scale_block_ref(&src).unwrap();

    assert_eq!(block.outlier_count, 2);
    assert_eq!(block.outlier_idx, [99, 7]);
}

#[test]
fn dual_scale_encode_decode_reasonable_error_without_outliers() {
    let src: Vec<f32> = (0..DUAL_SCALE_BLOCK_SIZE)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.03125)
        .collect();

    let block = encode_dual_scale_block_ref(&src).unwrap();
    let mut decoded = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];

    block.decode_ref(&mut decoded).unwrap();

    let mse = src
        .iter()
        .zip(decoded.iter())
        .map(|(&a, &b)| {
            let d = a - b;
            d * d
        })
        .sum::<f32>()
        / DUAL_SCALE_BLOCK_SIZE as f32;

    assert!(mse < 0.01, "mse={mse}");
}

#[test]
fn dual_scale_zero_scale_decodes_zero_regardless_of_q_code() {
    let mut block = DualScaleCompactBlock {
        scale_lo: [F16Bits::ZERO; DUAL_SCALE_GROUPS],
        scale_hi: [F16Bits::ZERO; DUAL_SCALE_GROUPS],
        membership: [0u8; DUAL_SCALE_MEMBERSHIP_BYTES],
        qs: [0xffu8; DUAL_SCALE_QS_BYTES],
        outlier_count: 0,
        outlier_idx: [0; 2],
        outlier_delta: [F16Bits::ZERO; 2],
    };

    let mut dst = vec![1.0f32; DUAL_SCALE_BLOCK_SIZE];
    block.decode_ref(&mut dst).unwrap();

    assert!(dst.iter().all(|&v| v == 0.0));

    set_membership_bit(&mut block.membership, 0, true);
    block.decode_ref(&mut dst).unwrap();
    assert_eq!(dst[0], 0.0);
}

#[test]
fn dual_scale_decode_bad_outlier_count_does_not_write_output() {
    let block = DualScaleCompactBlock {
        scale_lo: [F16Bits::from_f32(1.0); DUAL_SCALE_GROUPS],
        scale_hi: [F16Bits::ZERO; DUAL_SCALE_GROUPS],
        membership: [0u8; DUAL_SCALE_MEMBERSHIP_BYTES],
        qs: [0xffu8; DUAL_SCALE_QS_BYTES],
        outlier_count: 3,
        outlier_idx: [0, 1],
        outlier_delta: [F16Bits::ZERO; 2],
    };

    let mut dst = vec![123.0f32; DUAL_SCALE_BLOCK_SIZE];

    assert_eq!(block.decode_ref(&mut dst), Err(FormatError::BadBlock));
    assert!(dst.iter().all(|&v| v == 123.0));
}
