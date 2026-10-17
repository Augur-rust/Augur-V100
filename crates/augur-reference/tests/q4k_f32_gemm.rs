use augur_core::F16Bits;
use augur_formats::{pack_q4k_scales_ref, Q4KCompactBlock, Q4KScales, Q4K_QS_BYTES};
use augur_kernel_api::Q4KHotCpuTensor;
use augur_kernels::portable::qmatmul_q4k_f32_prefill_portable;
use augur_reference::{q4k_f32_gemm_reference, ReferenceQ8Error};
use augur_repack::repack_q4k_block_cpu_ref;

fn fixture(n: usize, k: usize) -> Vec<Q4KCompactBlock> {
    let blocks_per_row = k / 256;
    (0..n * blocks_per_row)
        .map(|block_index| {
            let scales = Q4KScales {
                scale: std::array::from_fn(|group| 1 + ((block_index + group) % 7) as u8),
                min: std::array::from_fn(|group| ((block_index * 3 + group) % 5) as u8),
            };
            Q4KCompactBlock {
                d: F16Bits::from_f32(0.03125),
                dmin: F16Bits::from_f32(0.015625),
                scales: pack_q4k_scales_ref(scales).unwrap(),
                qs: std::array::from_fn(|i| {
                    let lo = ((i + block_index) % 16) as u8;
                    let hi = ((i * 3 + block_index + 1) % 16) as u8;
                    lo | (hi << 4)
                }),
            }
        })
        .collect()
}

#[test]
fn m1_small_batch_and_model_shaped_prefill_match_compact_oracle() {
    let n = 256;
    let k = 512;
    let compact = fixture(n, k);
    assert_eq!(compact[0].qs.len(), Q4K_QS_BYTES);
    let hot =
        Q4KHotCpuTensor::new(n, k, compact.iter().map(repack_q4k_block_cpu_ref).collect()).unwrap();

    for m in [1, 4, 17] {
        let input: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) / 37.0)
            .collect();
        let mut expected = vec![0.0; m * n];
        let mut actual = vec![f32::NAN; m * n];
        q4k_f32_gemm_reference(&compact, m, n, k, &input, &mut expected).unwrap();
        qmatmul_q4k_f32_prefill_portable(&hot, &input, m, &mut actual).unwrap();
        assert_eq!(actual, expected, "m={m}");
    }
}

#[test]
fn reference_rejects_zero_and_inconsistent_shapes_before_writing() {
    let compact = fixture(2, 256);
    let mut output = [7.0; 4];
    assert_eq!(
        q4k_f32_gemm_reference(&compact, 0, 2, 256, &[], &mut output),
        Err(ReferenceQ8Error::BadWeightShape)
    );
    assert_eq!(output, [7.0; 4]);
    assert!(matches!(
        q4k_f32_gemm_reference(&compact, 2, 2, 256, &[0.0; 511], &mut output),
        Err(ReferenceQ8Error::BadActivationLength { .. })
    ));
    assert_eq!(output, [7.0; 4]);
}
