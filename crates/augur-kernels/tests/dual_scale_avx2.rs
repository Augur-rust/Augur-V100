#![cfg(target_arch = "x86_64")]

use augur_core::F16Bits;
use augur_kernel_api::{
    DualScaleHotCpuBlock, DualScaleHotCpuTensor, DualScaleQ8HotCpuLaunch, KernelLaunchError,
    OutlierCsrHotCpu, Q8KActivationHotCpuBlock, Q8KActivationHotCpuVector,
};
use augur_kernels::{
    portable::{
        dot_dual_scale_block_q8k_hot_cpu_grouped_ref, dot_dual_scale_hot_q8k_hot_cpu_grouped_ref,
        launch_qmatvec_dual_scale_q8k_hot_cpu_portable,
    },
    x86::avx2::{
        dot_dual_scale_block_q8k_hot_avx2_prevalidated,
        dot_dual_scale_hot_q8k_hot_avx2_prevalidated, launch_qmatvec_dual_scale_q8k_hot_cpu_avx2,
        qmatvec_dual_scale_q8k_hot_cpu_avx2,
    },
    KernelError,
};

const BLOCK_LEN: usize = DualScaleHotCpuBlock::DECODED_LEN;

fn host_supported() -> bool {
    std::is_x86_feature_detected!("avx2")
}

fn q4_code(q: i8) -> u8 {
    assert!((-8..=7).contains(&q));
    (q + 8) as u8
}

#[allow(clippy::needless_range_loop)]
fn packed_q4<F: FnMut(usize) -> i8>(mut value: F) -> [u8; 128] {
    let mut output = [0_u8; 128];
    for pi in 0..128 {
        let low_lane = pi * 2;
        let high_lane = low_lane + 1;
        let low = q4_code(value(low_lane));
        let high = q4_code(value(high_lane));
        output[pi] = low | (high << 4);
    }
    output
}

fn membership<F: FnMut(usize) -> bool>(mut hi: F) -> [u8; 32] {
    let mut output = [0_u8; 32];
    for lane in 0..BLOCK_LEN {
        if hi(lane) {
            output[lane / 8] |= 1_u8 << (lane % 8);
        }
    }
    output
}

fn weight_block<Q: FnMut(usize) -> i8, M: FnMut(usize) -> bool>(
    q4: Q,
    hi: M,
    scale_lo: f32,
    scale_hi: f32,
) -> DualScaleHotCpuBlock {
    DualScaleHotCpuBlock {
        scale_lo: [F16Bits::from_f32(scale_lo); 8],
        scale_hi: [F16Bits::from_f32(scale_hi); 8],
        membership: membership(hi),
        qs: packed_q4(q4),
    }
}

fn activation_block<F: FnMut(usize) -> i8>(scale: f32, q8: F) -> Q8KActivationHotCpuBlock {
    Q8KActivationHotCpuBlock {
        d: F16Bits::from_f32(scale),
        qs: core::array::from_fn(q8),
    }
}

fn assert_numerically_close(actual: f32, expected: f32, block_count: usize, fixture: &str) {
    let abs_err = (actual - expected).abs();
    let tol = 1.0e-5 * block_count.max(1) as f32 + 1.0e-6 * expected.abs().max(1.0);
    assert!(
        abs_err <= tol,
        "fixture={fixture} actual={actual} expected={expected} abs_err={abs_err} tol={tol}",
    );
}

fn run_block_case(
    weight: &DualScaleHotCpuBlock,
    activation: &Q8KActivationHotCpuBlock,
    fixture: &str,
) {
    if !host_supported() {
        return;
    }
    let expected = dot_dual_scale_block_q8k_hot_cpu_grouped_ref(weight, activation);
    let actual = unsafe { dot_dual_scale_block_q8k_hot_avx2_prevalidated(weight, activation) };
    assert_numerically_close(actual, expected, 1, fixture);
}

fn run_row_case(
    weights: Vec<DualScaleHotCpuBlock>,
    activations: Vec<Q8KActivationHotCpuBlock>,
    fixture: &str,
) {
    if !host_supported() {
        return;
    }
    assert_eq!(weights.len(), activations.len());
    let activation =
        Q8KActivationHotCpuVector::new(activations.len() * BLOCK_LEN, activations).unwrap();
    let expected = dot_dual_scale_hot_q8k_hot_cpu_grouped_ref(&weights, &activation);
    let actual = unsafe { dot_dual_scale_hot_q8k_hot_avx2_prevalidated(&weights, &activation) };
    assert_numerically_close(actual, expected, weights.len(), fixture);
}

// ── M28-C fixture helpers ────────────────────────────────────────────────

fn q8hot_vector(block_count: usize, scale: f32, salt: usize) -> Q8KActivationHotCpuVector {
    let blocks = (0..block_count)
        .map(|b| {
            activation_block(scale + b as f32 * 0.03125, |lane| {
                ((lane
                    .wrapping_mul(37)
                    .wrapping_add(b * 17)
                    .wrapping_add(salt))
                    % 256) as u8 as i8
            })
        })
        .collect::<Vec<_>>();
    Q8KActivationHotCpuVector::new(block_count * BLOCK_LEN, blocks).unwrap()
}

fn matrix_blocks(rows: usize, blocks_per_row: usize) -> Vec<DualScaleHotCpuBlock> {
    (0..rows * blocks_per_row)
        .map(|b| {
            weight_block(
                |lane| ((lane * 5 + b * 3) % 16) as i8 - 8,
                |lane| (lane + b) % 2 == 0 || (lane + b) % 7 == 0,
                0.125 + (b % 7) as f32 * 0.03125,
                0.75 + (b % 5) as f32 * 0.0625,
            )
        })
        .collect()
}

fn tensor_without_outliers(rows: usize, cols: usize) -> DualScaleHotCpuTensor {
    let outliers =
        OutlierCsrHotCpu::new(rows, cols, vec![0_u32; rows + 1], Vec::new(), Vec::new()).unwrap();
    DualScaleHotCpuTensor::new(rows, cols, matrix_blocks(rows, cols / BLOCK_LEN), outliers).unwrap()
}

fn tensor_with_one_outlier_per_row(rows: usize, cols: usize) -> DualScaleHotCpuTensor {
    let mut rp = vec![0_u32];
    let mut ci = Vec::new();
    let mut dl = Vec::new();
    for row in 0..rows {
        let col = if row == 0 {
            0
        } else if row + 1 == rows {
            cols - 1
        } else {
            (row * 97 + 11) % cols
        };
        ci.push(u32::try_from(col).unwrap());
        dl.push(F16Bits::from_f32(0.5 + row as f32 * 0.125));
        rp.push(ci.len() as u32);
    }
    let outliers = OutlierCsrHotCpu::new(rows, cols, rp, ci, dl).unwrap();
    DualScaleHotCpuTensor::new(rows, cols, matrix_blocks(rows, cols / BLOCK_LEN), outliers).unwrap()
}

fn tensor_with_two_outliers_per_row(rows: usize, cols: usize) -> DualScaleHotCpuTensor {
    let mut rp = vec![0_u32];
    let mut ci = Vec::new();
    let mut dl = Vec::new();
    for row in 0..rows {
        let first = if row == 0 { 0 } else { (row * 43 + 5) % cols };
        let mut second = if row + 1 == rows {
            cols - 1
        } else {
            (row * 71 + 97) % cols
        };
        if second == first {
            second = (second + 1) % cols;
        }
        ci.push(u32::try_from(first).unwrap());
        dl.push(F16Bits::from_f32(0.375 + row as f32 * 0.0625));
        ci.push(u32::try_from(second).unwrap());
        dl.push(F16Bits::from_f32(-0.625 - row as f32 * 0.03125));
        rp.push(ci.len() as u32);
    }
    let outliers = OutlierCsrHotCpu::new(rows, cols, rp, ci, dl).unwrap();
    DualScaleHotCpuTensor::new(rows, cols, matrix_blocks(rows, cols / BLOCK_LEN), outliers).unwrap()
}

fn assert_full_close(actual: &[f32], expected: &[f32], blocks_per_row: usize, fixture: &str) {
    assert_eq!(actual.len(), expected.len());
    for row in 0..actual.len() {
        let ae = (actual[row] - expected[row]).abs();
        let tol = 1.0e-5 * blocks_per_row.max(1) as f32 + 1.0e-6 * expected[row].abs().max(1.0);
        assert!(
            ae <= tol,
            "fixture={fixture} row={row} actual={} expected={} ae={ae} tol={tol}",
            actual[row],
            expected[row],
        );
    }
}

fn run_full_case(
    weight: &DualScaleHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    fixture: &str,
) {
    if !host_supported() {
        return;
    }
    let mut portable = vec![0.0_f32; weight.rows()];
    let mut avx2 = vec![0.0_f32; weight.rows()];
    launch_qmatvec_dual_scale_q8k_hot_cpu_portable(weight, activation, &mut portable).unwrap();
    qmatvec_dual_scale_q8k_hot_cpu_avx2(weight, activation, &mut avx2).unwrap();
    assert_full_close(&avx2, &portable, weight.blocks_per_row(), fixture);
}

// ── M28-B tests ─────────────────────────────────────────────────────────

#[test]
fn avx2_dual_scale_all_lo_matches_portable() {
    let w = weight_block(|lane| (lane % 16) as i8 - 8, |_| false, 0.5, 2.0);
    let a = activation_block(0.25, |lane| (lane % 127) as i8 - 63);
    run_block_case(&w, &a, "all-lo");
}

#[test]
fn avx2_dual_scale_all_hi_matches_portable() {
    let w = weight_block(|lane| (lane % 16) as i8 - 8, |_| true, 0.5, 2.0);
    let a = activation_block(0.25, |lane| (lane % 127) as i8 - 63);
    run_block_case(&w, &a, "all-hi");
}

#[test]
fn avx2_dual_scale_alternating_membership_matches_portable() {
    for offset in [0, 1] {
        let w = weight_block(
            |lane| (lane % 16) as i8 - 8,
            |lane| (lane + offset) % 2 == 0,
            0.5,
            2.0,
        );
        let a = activation_block(0.25, |lane| if lane % 2 == 0 { 127 } else { -128 });
        run_block_case(
            &w,
            &a,
            if offset == 0 {
                "alternating-55"
            } else {
                "alternating-aa"
            },
        );
    }
}

#[test]
fn avx2_dual_scale_single_membership_bits_match_portable() {
    for lane in [0, 7, 8, 15, 16, 23, 24, 31, 32, 63, 64, 127, 128, 191] {
        let w = weight_block(
            |index| (index % 16) as i8 - 8,
            |index| index == lane,
            0.25,
            1.5,
        );
        let a = activation_block(0.5, |index| {
            if index == lane {
                -128
            } else {
                (index % 31) as i8 - 15
            }
        });
        run_block_case(&w, &a, "single-membership-bit");
    }
}

#[test]
fn avx2_dual_scale_low_high_nibble_order_matches_portable() {
    let w = weight_block(
        |lane| if lane % 2 == 0 { -8 } else { 7 },
        |lane| lane % 3 == 0,
        0.75,
        1.25,
    );
    let a = activation_block(0.5, |lane| if lane % 2 == 0 { 127 } else { -128 });
    run_block_case(&w, &a, "low-high-nibble-order");
}

#[test]
fn avx2_dual_scale_i32_lo_accumulator_extreme_is_exact() {
    let w = weight_block(|_| -8, |_| false, 1.0, 0.0);
    let a = activation_block(1.0, |_| -128);
    let expected = 8.0 * 32_768.0;
    let portable = dot_dual_scale_block_q8k_hot_cpu_grouped_ref(&w, &a);
    assert_eq!(portable, expected);
    if host_supported() {
        let actual = unsafe { dot_dual_scale_block_q8k_hot_avx2_prevalidated(&w, &a) };
        assert_eq!(actual, expected);
    }
}

#[test]
fn avx2_dual_scale_i32_hi_accumulator_extreme_is_exact() {
    let w = weight_block(|_| -8, |_| true, 0.0, 1.0);
    let a = activation_block(1.0, |_| -128);
    let expected = 8.0 * 32_768.0;
    let portable = dot_dual_scale_block_q8k_hot_cpu_grouped_ref(&w, &a);
    assert_eq!(portable, expected);
    if host_supported() {
        let actual = unsafe { dot_dual_scale_block_q8k_hot_avx2_prevalidated(&w, &a) };
        assert_eq!(actual, expected);
    }
}

#[test]
fn avx2_dual_scale_all_q4_q8_extremes_match_portable() {
    for q4 in [-8_i8, 7] {
        for q8 in [-128_i8, 127] {
            for membership_hi in [false, true] {
                let w = weight_block(|_| q4, |_| membership_hi, 0.75, 1.25);
                let a = activation_block(0.5, |_| q8);
                run_block_case(&w, &a, "q4-q8-extreme");
            }
        }
    }
}

#[test]
fn avx2_dual_scale_zero_activation_scale_is_canonical_zero() {
    if !host_supported() {
        return;
    }
    let actual = unsafe {
        dot_dual_scale_block_q8k_hot_avx2_prevalidated(
            &weight_block(
                |lane| (lane % 16) as i8 - 8,
                |lane| lane % 2 == 0,
                0.75,
                1.25,
            ),
            &activation_block(0.0, |_| -128),
        )
    };
    assert_eq!(actual, 0.0);
    assert!(!actual.is_sign_negative());
}

#[test]
fn avx2_dual_scale_zero_weight_scales_are_canonical_zero() {
    if !host_supported() {
        return;
    }
    let actual = unsafe {
        dot_dual_scale_block_q8k_hot_avx2_prevalidated(
            &weight_block(|_| -8, |lane| lane % 2 == 0, 0.0, 0.0),
            &activation_block(1.0, |_| -128),
        )
    };
    assert_eq!(actual, 0.0);
}

#[test]
fn avx2_dual_scale_distinct_lo_hi_scales_match_portable() {
    let w = DualScaleHotCpuBlock {
        scale_lo: core::array::from_fn(|g| F16Bits::from_f32(0.125 + g as f32 * 0.0625)),
        scale_hi: core::array::from_fn(|g| F16Bits::from_f32(1.0 + g as f32 * 0.125)),
        membership: membership(|lane| lane % 5 == 0 || lane % 7 == 0),
        qs: packed_q4(|lane| (lane % 16) as i8 - 8),
    };
    let a = activation_block(0.375, |lane| ((lane * 37 + 11) % 256) as u8 as i8);
    run_block_case(&w, &a, "distinct-lo-hi-scales");
}

#[test]
fn avx2_dual_scale_matches_portable_for_m28_k_shapes() {
    for bc in [1_usize, 2, 4, 16] {
        let weights = (0..bc)
            .map(|b| {
                weight_block(
                    |lane| ((lane + b * 3) % 16) as i8 - 8,
                    |lane| (lane + b) % 3 == 0 || (lane + b) % 11 == 0,
                    0.125 + b as f32 * 0.03125,
                    0.75 + b as f32 * 0.0625,
                )
            })
            .collect::<Vec<_>>();
        let activations = (0..bc)
            .map(|b| {
                activation_block(0.25 + b as f32 * 0.03125, |lane| {
                    ((lane * 37 + b * 17 + 11) % 256) as u8 as i8
                })
            })
            .collect::<Vec<_>>();
        let name = match bc {
            1 => "k-256",
            2 => "k-512",
            4 => "k-1024",
            _ => "k-4096",
        };
        run_row_case(weights, activations, name);
    }
}

#[test]
fn avx2_dual_scale_candidate_bit_exact_matrix() {
    if !host_supported() {
        return;
    }
    for bc in [1_usize, 2, 4, 16] {
        let weights = (0..bc)
            .map(|b| {
                weight_block(
                    |lane| ((lane * 5 + b * 3) % 16) as i8 - 8,
                    |lane| (lane + b) % 2 == 0,
                    0.25,
                    0.75,
                )
            })
            .collect::<Vec<_>>();
        let activations = (0..bc)
            .map(|b| activation_block(0.5, |lane| ((lane * 37 + b * 17) % 256) as u8 as i8))
            .collect::<Vec<_>>();
        let activation = Q8KActivationHotCpuVector::new(bc * BLOCK_LEN, activations).unwrap();
        let expected = dot_dual_scale_hot_q8k_hot_cpu_grouped_ref(&weights, &activation);
        let avx2 = unsafe { dot_dual_scale_hot_q8k_hot_avx2_prevalidated(&weights, &activation) };
        assert_eq!(avx2, expected, "candidate BitExact mismatch bc={bc}");
    }
}

// ── M28-C complete qmatvec tests ─────────────────────────────────────────

#[test]
fn avx2_complete_qmatvec_zero_outliers_matches_portable() {
    for &(rows, cols) in &[(1, 256), (3, 512), (16, 1024), (4, 4096)] {
        let weight = tensor_without_outliers(rows, cols);
        let activation = q8hot_vector(cols / BLOCK_LEN, 0.5, rows + cols);
        run_full_case(&weight, &activation, "zero-outliers");
    }
}

#[test]
fn avx2_complete_qmatvec_one_outlier_per_row_matches_portable() {
    for &(rows, cols) in &[(1, 256), (3, 512), (16, 1024), (4, 4096)] {
        let weight = tensor_with_one_outlier_per_row(rows, cols);
        let activation = q8hot_vector(cols / BLOCK_LEN, 0.375, rows * 17 + cols);
        run_full_case(&weight, &activation, "one-outlier-per-row");
    }
}

#[test]
fn avx2_complete_qmatvec_two_outliers_per_row_matches_portable() {
    for &(rows, cols) in &[(1, 256), (3, 512), (16, 1024), (4, 4096)] {
        let weight = tensor_with_two_outliers_per_row(rows, cols);
        let activation = q8hot_vector(cols / BLOCK_LEN, 0.625, rows * 29 + cols);
        run_full_case(&weight, &activation, "two-outliers-per-row");
    }
}

#[test]
fn avx2_complete_qmatvec_csr_first_and_last_lanes_match_portable() {
    let (rows, cols) = (2, BLOCK_LEN * 2);
    let outliers = OutlierCsrHotCpu::new(
        rows,
        cols,
        vec![0, 1, 2],
        vec![0, u32::try_from(cols - 1).unwrap()],
        vec![F16Bits::from_f32(2.0), F16Bits::from_f32(-3.0)],
    )
    .unwrap();
    let weight =
        DualScaleHotCpuTensor::new(rows, cols, matrix_blocks(rows, cols / BLOCK_LEN), outliers)
            .unwrap();
    let mut ab = vec![Q8KActivationHotCpuBlock::ZERO; 2];
    ab[0].d = F16Bits::from_f32(1.0);
    ab[0].qs[0] = -7;
    ab[1].d = F16Bits::from_f32(1.0);
    ab[1].qs[255] = 11;
    let activation = Q8KActivationHotCpuVector::new(cols, ab).unwrap();
    run_full_case(&weight, &activation, "csr-first-last-lane");
}

#[test]
fn avx2_complete_qmatvec_preserves_output_tail() {
    if !host_supported() {
        return;
    }
    let (rows, cols) = (2, BLOCK_LEN * 2);
    let weight = tensor_with_two_outliers_per_row(rows, cols);
    let activation = q8hot_vector(cols / BLOCK_LEN, 0.5, 701);
    let mut expected = vec![0.0_f32; rows];
    launch_qmatvec_dual_scale_q8k_hot_cpu_portable(&weight, &activation, &mut expected).unwrap();
    let mut output = vec![91.0_f32, 92.0, 93.0, 94.0];
    qmatvec_dual_scale_q8k_hot_cpu_avx2(&weight, &activation, &mut output).unwrap();
    assert_full_close(
        &output[..rows],
        &expected,
        weight.blocks_per_row(),
        "output-tail",
    );
    assert_eq!(&output[rows..], &[93.0, 94.0]);
}

#[test]
fn avx2_complete_qmatvec_bad_activation_leaves_output_unchanged() {
    let (rows, cols) = (2, BLOCK_LEN * 2);
    let weight = tensor_without_outliers(rows, cols);
    let activation = q8hot_vector(1, 0.5, 703);
    let mut output = vec![123.0_f32, 456.0];
    let before = output.clone();
    let ptr = output.as_ptr();
    assert_eq!(
        qmatvec_dual_scale_q8k_hot_cpu_avx2(&weight, &activation, &mut output),
        Err(KernelError::BadActivationLength {
            expected: cols,
            actual: BLOCK_LEN,
        }),
    );
    assert_eq!(output, before);
    assert_eq!(output.as_ptr(), ptr);
}

#[test]
fn avx2_complete_qmatvec_short_output_is_unchanged() {
    let (rows, cols) = (2, BLOCK_LEN * 2);
    let weight = tensor_without_outliers(rows, cols);
    let activation = q8hot_vector(cols / BLOCK_LEN, 0.5, 709);
    let mut output = vec![321.0_f32];
    let before = output.clone();
    let ptr = output.as_ptr();
    assert_eq!(
        qmatvec_dual_scale_q8k_hot_cpu_avx2(&weight, &activation, &mut output),
        Err(KernelError::BadOutputLength {
            expected: rows,
            actual: 1,
        }),
    );
    assert_eq!(output, before);
    assert_eq!(output.as_ptr(), ptr);
}

#[test]
fn avx2_complete_qmatvec_zero_q8_scale_zeroes_main_and_csr() {
    if !host_supported() {
        return;
    }
    let (rows, cols) = (2, BLOCK_LEN * 2);
    let weight = tensor_with_two_outliers_per_row(rows, cols);
    let ab = (0..cols / BLOCK_LEN)
        .map(|_| Q8KActivationHotCpuBlock {
            d: F16Bits::ZERO,
            qs: [-128; BLOCK_LEN],
        })
        .collect::<Vec<_>>();
    let mut output = vec![99.0_f32; rows];
    qmatvec_dual_scale_q8k_hot_cpu_avx2(
        &weight,
        &Q8KActivationHotCpuVector::new(cols, ab).unwrap(),
        &mut output,
    )
    .unwrap();
    assert_eq!(output, vec![0.0_f32; rows]);
    assert!(output.iter().all(|v| !v.is_sign_negative()));
}

#[test]
fn avx2_complete_launch_matches_existing_abi() {
    let launch: DualScaleQ8HotCpuLaunch = launch_qmatvec_dual_scale_q8k_hot_cpu_avx2;
    let expected: DualScaleQ8HotCpuLaunch = launch_qmatvec_dual_scale_q8k_hot_cpu_avx2;
    assert!(std::ptr::addr_eq(
        launch as *const (),
        expected as *const (),
    ));
}

#[test]
fn avx2_complete_launch_matches_checked_kernel() {
    if !host_supported() {
        return;
    }
    let (rows, cols) = (3, BLOCK_LEN * 4);
    let weight = tensor_with_two_outliers_per_row(rows, cols);
    let activation = q8hot_vector(cols / BLOCK_LEN, 0.5, 727);
    let mut direct = vec![0.0_f32; rows];
    let mut launched = vec![0.0_f32; rows];
    qmatvec_dual_scale_q8k_hot_cpu_avx2(&weight, &activation, &mut direct).unwrap();
    launch_qmatvec_dual_scale_q8k_hot_cpu_avx2(&weight, &activation, &mut launched).unwrap();
    assert_eq!(launched, direct);
}

#[test]
fn avx2_complete_launch_maps_kernel_error() {
    let (rows, cols) = (2, BLOCK_LEN * 2);
    let weight = tensor_without_outliers(rows, cols);
    let activation = q8hot_vector(1, 0.5, 733);
    let mut output = vec![17.0_f32; rows];
    let before = output.clone();
    assert_eq!(
        launch_qmatvec_dual_scale_q8k_hot_cpu_avx2(&weight, &activation, &mut output),
        Err(KernelLaunchError::BadActivationLength {
            expected: cols,
            actual: BLOCK_LEN,
        }),
    );
    assert_eq!(output, before);
}

#[test]
fn avx2_complete_qmatvec_candidate_bit_exact_matrix() {
    if !host_supported() {
        return;
    }
    for &(rows, cols) in &[(1, 256), (3, 512), (16, 1024), (4, 4096)] {
        for fixture in 0..3 {
            let weight = match fixture {
                0 => tensor_without_outliers(rows, cols),
                1 => tensor_with_one_outlier_per_row(rows, cols),
                _ => tensor_with_two_outliers_per_row(rows, cols),
            };
            let activation = q8hot_vector(cols / BLOCK_LEN, 0.5, rows * 31 + cols + fixture);
            let mut portable = vec![0.0_f32; rows];
            let mut avx2 = vec![0.0_f32; rows];
            launch_qmatvec_dual_scale_q8k_hot_cpu_portable(&weight, &activation, &mut portable)
                .unwrap();
            qmatvec_dual_scale_q8k_hot_cpu_avx2(&weight, &activation, &mut avx2).unwrap();
            assert_eq!(
                avx2, portable,
                "candidate BitExact mismatch: rows={rows} cols={cols} fixture={fixture}",
            );
        }
    }
}
