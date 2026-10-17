#![cfg(target_arch = "x86_64")]

use augur_core::F16Bits;
use augur_kernel_api::{Q4KHotCpuBlock, Q4K_HOT_CPU_QS_BYTES};
use augur_kernels::{
    portable::{decode_q4k_hot_f32, dot_q4k_hot_f32_direct},
    x86::avx2::dot_q4k_hot_f32_avx2_prevalidated,
};

fn host_supported() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

fn block(seed: usize, scale: u8, min: u8) -> Q4KHotCpuBlock {
    let mut qs = [0_u8; Q4K_HOT_CPU_QS_BYTES];
    for (index, packed) in qs.iter_mut().enumerate() {
        let low = index.wrapping_add(seed) as u8 & 0x0f;
        let high = 15_u8.wrapping_sub(low);
        *packed = low | (high << 4);
    }
    Q4KHotCpuBlock {
        d: F16Bits::from_f32(0.75),
        dmin: F16Bits::from_f32(0.25),
        scale: [scale; 8],
        min: [min; 8],
        qs,
    }
}

fn assert_m27_close(got: f32, reference: f32, block_count: usize, fixture: &str) {
    let abs_error = (got - reference).abs();
    let tolerance = 2.0e-4 * block_count.max(1) as f32 + 2.0e-5 * reference.abs().max(1.0);
    assert!(
        abs_error <= tolerance,
        "fixture={fixture} blocks={block_count} got={got} reference={reference} \
         abs_error={abs_error} tolerance={tolerance}"
    );
}

fn run_case(blocks: Vec<Q4KHotCpuBlock>, activation: Vec<f32>, fixture: &str) {
    if !host_supported() {
        return;
    }

    let portable = dot_q4k_hot_f32_direct(&blocks, &activation).unwrap();

    let mut dense = vec![0.0_f32; activation.len()];
    decode_q4k_hot_f32(&blocks, &mut dense).unwrap();
    let dense_reference = dense
        .iter()
        .zip(&activation)
        .fold(0.0_f32, |sum, (weight, input)| sum + weight * input);

    let avx2 = unsafe { dot_q4k_hot_f32_avx2_prevalidated(&blocks, &activation) };

    assert_m27_close(avx2, portable, blocks.len(), fixture);
    assert_m27_close(avx2, dense_reference, blocks.len(), fixture);
}

#[test]
fn avx2_q4k_zero_activation_is_canonical_zero() {
    for block_count in [1, 2, 4, 16] {
        let blocks = (0..block_count)
            .map(|seed| block(seed, 63, 63))
            .collect::<Vec<_>>();
        let activation = vec![0.0_f32; block_count * 256];
        run_case(blocks, activation, "zero");
    }
}

#[test]
fn avx2_q4k_nibble_and_metadata_extremes() {
    for &(scale, min) in &[(0, 0), (63, 0), (0, 63), (63, 63)] {
        let blocks = vec![block(0, scale, min)];
        let activation = (0..256)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        run_case(blocks, activation, "metadata-extreme-alternating");

        let all_one = vec![1.0_f32; 256];
        run_case(
            vec![block(0, scale, min)],
            all_one,
            "metadata-extreme-all-one",
        );
    }
}

#[test]
fn avx2_q4k_matches_references_for_m27_shapes() {
    for block_count in [1, 2, 4, 16] {
        let blocks = (0..block_count)
            .map(|seed| block(seed, 1 + (seed as u8 % 63), 63))
            .collect::<Vec<_>>();
        let len = block_count * 256;

        let fixtures = [
            ("ones", vec![1.0_f32; len]),
            (
                "bounded",
                (0..len)
                    .map(|i| ((i * 37 + 11) % 127) as f32 / 64.0 - 63.0 / 64.0)
                    .collect(),
            ),
            (
                "positive",
                (0..len).map(|i| ((i % 31) + 1) as f32 / 32.0).collect(),
            ),
            (
                "negative",
                (0..len).map(|i| -(((i % 31) + 1) as f32) / 32.0).collect(),
            ),
            (
                "sparse",
                (0..len)
                    .map(|i| if i % 37 == 0 { 0.75 } else { 0.0 })
                    .collect(),
            ),
        ];

        for (name, activation) in fixtures {
            run_case(blocks.clone(), activation, name);
        }
    }
}
