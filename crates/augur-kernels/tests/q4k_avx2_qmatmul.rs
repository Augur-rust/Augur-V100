#![cfg(target_arch = "x86_64")]

use augur_core::F16Bits;
use augur_kernel_api::{Q4KHotCpuBlock, Q4KHotCpuTensor, QMatmulF32Launch, Q4K_HOT_CPU_QS_BYTES};
use augur_kernels::{
    portable::{decode_q4k_hot_f32, qmatmul_q4k_f32_portable},
    x86::avx2::{launch_qmatmul_q4k_f32_avx2, qmatmul_q4k_f32_avx2},
    KernelError,
};

fn host_supported() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

fn block(seed: usize) -> Q4KHotCpuBlock {
    let mut qs = [0_u8; Q4K_HOT_CPU_QS_BYTES];
    for (index, packed) in qs.iter_mut().enumerate() {
        let low = index.wrapping_add(seed) as u8 & 0x0f;
        let high = 15_u8.wrapping_sub(low);
        *packed = low | (high << 4);
    }
    Q4KHotCpuBlock {
        d: F16Bits::from_f32(0.25 + (seed % 5) as f32 * 0.125),
        dmin: F16Bits::from_f32(0.0625 + (seed % 3) as f32 * 0.0625),
        scale: core::array::from_fn(|group| ((seed + group * 7) % 64) as u8),
        min: core::array::from_fn(|group| ((seed * 3 + group * 11) % 64) as u8),
        qs,
    }
}

fn tensor(rows: usize, cols: usize) -> Q4KHotCpuTensor {
    assert!(rows > 0);
    assert!(cols > 0);
    assert!(cols.is_multiple_of(Q4KHotCpuBlock::DECODED_LEN));

    let blocks_per_row = cols / Q4KHotCpuBlock::DECODED_LEN;
    let blocks = (0..rows * blocks_per_row).map(block).collect();
    Q4KHotCpuTensor::new(rows, cols, blocks).unwrap()
}

fn activation(cols: usize, fixture: usize) -> Vec<f32> {
    (0..cols)
        .map(|index| {
            let value = (index * 37 + fixture * 17) % 127;
            (value as f32 - 63.0) / 64.0
        })
        .collect()
}

fn assert_m27_close(got: f32, reference: f32, blocks_per_row: usize, fixture: &str, row: usize) {
    let abs_error = (got - reference).abs();
    let tolerance = 2.0e-4 * blocks_per_row.max(1) as f32 + 2.0e-5 * reference.abs().max(1.0);
    assert!(
        abs_error <= tolerance,
        "fixture={fixture} row={row} blocks_per_row={blocks_per_row} got={got} \
         reference={reference} abs_error={abs_error} tolerance={tolerance}",
    );
}

fn run_case(rows: usize, cols: usize, input: Vec<f32>, fixture: &str) {
    if !host_supported() {
        return;
    }

    let weight = tensor(rows, cols);
    let mut portable = vec![0.0_f32; rows];
    let mut avx2 = vec![0.0_f32; rows];

    qmatmul_q4k_f32_portable(&weight, &input, &mut portable).unwrap();
    qmatmul_q4k_f32_avx2(&weight, &input, &mut avx2).unwrap();

    for row in 0..rows {
        assert_m27_close(
            avx2[row],
            portable[row],
            weight.blocks_per_row(),
            fixture,
            row,
        );

        let mut decoded = vec![0.0_f32; weight.cols()];
        decode_q4k_hot_f32(weight.row_blocks(row).unwrap(), &mut decoded).unwrap();
        let dense_reference = decoded
            .iter()
            .zip(&input)
            .fold(0.0_f32, |sum, (w, x)| sum + w * x);
        assert_m27_close(
            avx2[row],
            dense_reference,
            weight.blocks_per_row(),
            fixture,
            row,
        );
    }
}

#[test]
fn avx2_qmatmul_matches_references() {
    for &(rows, cols) in &[(1, 256), (3, 512), (16, 1024), (4, 4096)] {
        run_case(rows, cols, vec![0.0; cols], "zero");
        run_case(rows, cols, vec![1.0; cols], "ones");
        run_case(rows, cols, activation(cols, rows + cols), "bounded");
        run_case(
            rows,
            cols,
            (0..cols)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
            "alternating",
        );
    }
}

#[test]
fn avx2_qmatmul_preserves_output_on_validation_error() {
    let weight = tensor(2, 256);

    let short_input = vec![0.0_f32; 255];
    let mut output = vec![7.0_f32, 11.0_f32];
    let before = output.clone();
    assert_eq!(
        qmatmul_q4k_f32_avx2(&weight, &short_input, &mut output),
        Err(KernelError::BadActivationLength {
            expected: 256,
            actual: 255,
        }),
    );
    assert_eq!(output, before);

    let input = vec![0.0_f32; 256];
    let mut short_output = vec![13.0_f32];
    let before = short_output.clone();
    assert_eq!(
        qmatmul_q4k_f32_avx2(&weight, &input, &mut short_output),
        Err(KernelError::BadOutputLength {
            expected: 2,
            actual: 1,
        }),
    );
    assert_eq!(short_output, before);
}

#[test]
fn avx2_launch_matches_qmatmul_f32_abi() {
    let launch: QMatmulF32Launch = launch_qmatmul_q4k_f32_avx2;
    let expected: QMatmulF32Launch = launch_qmatmul_q4k_f32_avx2;
    let actual_addr = launch as *const ();
    let expected_addr = expected as *const ();
    assert!(std::ptr::addr_eq(actual_addr, expected_addr));
}

#[test]
fn avx2_launch_matches_checked_kernel() {
    if !host_supported() {
        return;
    }
    let weight = tensor(3, 1024);
    let input = activation(1024, 97);
    let mut direct = vec![0.0_f32; 3];
    let mut launch = vec![0.0_f32; 3];

    qmatmul_q4k_f32_avx2(&weight, &input, &mut direct).unwrap();
    launch_qmatmul_q4k_f32_avx2(&weight, &input, &mut launch).unwrap();
    assert_eq!(launch, direct);
}

#[test]
fn avx2_checked_kernel_reports_missing_features_when_unavailable() {
    if host_supported() {
        return;
    }
    let weight = tensor(1, 256);
    let input = vec![0.0_f32; 256];
    let mut output = vec![9.0_f32];
    let before = output.clone();

    assert_eq!(
        qmatmul_q4k_f32_avx2(&weight, &input, &mut output),
        Err(KernelError::MissingCpuFeatures),
    );
    assert_eq!(output, before);
}
