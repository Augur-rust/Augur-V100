fn shared_word(block: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(block[offset..offset + 4].try_into().unwrap())
}

fn shared_byte(block: &[u8], offset: usize) -> u8 {
    let aligned = offset & !3;
    ((shared_word(block, aligned) >> ((offset & 3) * 8)) & 0xff) as u8
}

fn q4_byte_pack(block: &[u8], group: usize, subgroup_lane: usize, quad: usize) -> u32 {
    let base = 32 + (group / 2) * 32 + subgroup_lane * 8 + quad * 4;
    let shift = (group & 1) * 4;
    (0..4).fold(0, |packed, byte| {
        packed | (u32::from((block[base + byte] >> shift) & 0x0f) << (byte * 8))
    })
}

fn q4_word_pack(block: &[u8], group: usize, subgroup_lane: usize, quad: usize) -> u32 {
    let base = 32 + (group / 2) * 32 + subgroup_lane * 8 + quad * 4;
    assert_eq!(base % std::mem::align_of::<u32>(), 0);
    let packed = shared_word(block, base);
    if group & 1 == 0 {
        packed & 0x0f0f_0f0f
    } else {
        (packed & 0xf0f0_f0f0) >> 4
    }
}

fn q5_byte_pack(block: &[u8], group: usize, subgroup_lane: usize, quad: usize) -> u32 {
    let q_lane = subgroup_lane * 8 + quad * 4;
    let chunk = group / 2;
    let shift = (group & 1) * 4;
    (0..4).fold(0, |packed, byte| {
        let low = (block[64 + chunk * 32 + q_lane + byte] >> shift) & 0x0f;
        let high = ((block[32 + q_lane + byte] >> group) & 1) << 4;
        packed | (u32::from(low | high) << (byte * 8))
    })
}

fn q5_word_pack(block: &[u8], group: usize, subgroup_lane: usize, quad: usize) -> u32 {
    let q_lane = subgroup_lane * 8 + quad * 4;
    let chunk = group / 2;
    let shift = (group & 1) * 4;
    let low_offset = 64 + chunk * 32 + q_lane;
    let high_offset = 32 + q_lane;
    assert_eq!(low_offset % std::mem::align_of::<u32>(), 0);
    assert_eq!(high_offset % std::mem::align_of::<u32>(), 0);
    let low_word = shared_word(block, low_offset);
    let high_word = shared_word(block, high_offset);
    let low_nibbles = if shift == 0 {
        low_word & 0x0f0f_0f0f
    } else {
        (low_word & 0xf0f0_f0f0) >> 4
    };
    let high_bits = (high_word & (0x0101_0101 << group)) >> group;
    low_nibbles | (high_bits << 4)
}

fn fixture(seed: u32) -> [u8; 256] {
    let mut state = seed;
    std::array::from_fn(|index| {
        state = state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223_u32.wrapping_add(index as u32));
        (state >> 24) as u8
    })
}

#[test]
fn aligned_word_reads_preserve_every_metadata_byte() {
    for seed in 0..64 {
        let block = fixture(seed);
        for offset in 0..20 {
            assert_eq!(shared_byte(&block, offset), block[offset]);
        }
    }
}

#[test]
fn q4_and_q5_word_packing_matches_the_bytewise_control() {
    for seed in 0..256 {
        let block = fixture(seed);
        for group in 0..8 {
            for subgroup_lane in 0..4 {
                for quad in 0..2 {
                    assert_eq!(
                        q4_word_pack(&block, group, subgroup_lane, quad),
                        q4_byte_pack(&block, group, subgroup_lane, quad)
                    );
                    assert_eq!(
                        q5_word_pack(&block, group, subgroup_lane, quad),
                        q5_byte_pack(&block, group, subgroup_lane, quad)
                    );
                }
            }
        }
    }
}

