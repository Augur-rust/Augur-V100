use augur_formats::{
    Q5KCompactBlock, Q6KCompactBlock, ReferenceDecode, Q5K_BLOCK_BYTES, Q6K_BLOCK_BYTES,
};

#[test]
fn zero_blocks_decode_exactly() {
    let q5 = Q5KCompactBlock::from_le_bytes(&[0_u8; Q5K_BLOCK_BYTES]).unwrap();
    let q6 = Q6KCompactBlock::from_le_bytes(&[0_u8; Q6K_BLOCK_BYTES]).unwrap();
    let mut decoded = [1.0_f32; 256];
    q5.decode_ref(&mut decoded).unwrap();
    assert_eq!(decoded, [0.0; 256]);
    decoded.fill(1.0);
    q6.decode_ref(&mut decoded).unwrap();
    assert_eq!(decoded, [0.0; 256]);
}

#[test]
fn q6_constant_wire_matches_llama_layout() {
    let mut bytes = [0_u8; Q6K_BLOCK_BYTES];
    bytes[192..208].fill(0x80);
    bytes[208..210].copy_from_slice(&0x0c00_u16.to_le_bytes());
    let block = Q6KCompactBlock::from_le_bytes(&bytes).unwrap();
    let mut decoded = [0.0_f32; 256];
    block.decode_ref(&mut decoded).unwrap();
    assert_eq!(decoded, [1.0; 256]);
}

#[test]
fn malformed_or_nonfinite_metadata_fails_closed() {
    assert!(Q5KCompactBlock::from_le_bytes(&[0; Q5K_BLOCK_BYTES - 1]).is_err());
    assert!(Q6KCompactBlock::from_le_bytes(&[0; Q6K_BLOCK_BYTES - 1]).is_err());
    let mut q5 = [0_u8; Q5K_BLOCK_BYTES];
    q5[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
    assert!(Q5KCompactBlock::from_le_bytes(&q5).is_err());
    let mut q6 = [0_u8; Q6K_BLOCK_BYTES];
    q6[208..].copy_from_slice(&0x7c00_u16.to_le_bytes());
    assert!(Q6KCompactBlock::from_le_bytes(&q6).is_err());
}
