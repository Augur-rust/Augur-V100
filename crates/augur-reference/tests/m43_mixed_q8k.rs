use augur_formats::{
    quantize_q2k_tensor_ref, quantize_q8_0_tensor_ref, quantize_q8k_activation_vector_ref,
    ReferenceDecode, Q2K_BLOCK_SIZE, Q8_0_BLOCK_SIZE,
};
use augur_kernel_api::KernelLaunchError;
use augur_kernels::portable::{
    launch_qmatvec_q2k_q8hot_cpu_portable, launch_qmatvec_q8_0_q8hot_cpu_portable,
};
use augur_reference::{qmatvec_q2k_q8k_ref, qmatvec_q8_0_q8k_ref};
use augur_repack::{
    repack_q2k_tensor_cpu_ref, repack_q8_0_tensor_cpu_ref, repack_q8k_activation_vector_cpu_ref,
};

const ROWS: usize = 3;
const COLS: usize = 512;

fn weight_source() -> Vec<f32> {
    (0..ROWS * COLS)
        .map(|index| {
            let row = index / COLS;
            let col = index % COLS;
            (((col * 17 + row * 29) % 101) as f32 - 50.0) * (0.0025 + row as f32 * 0.0005)
        })
        .collect()
}

fn activation_source() -> Vec<f32> {
    (0..COLS)
        .map(|index| (((index * 13) % 67) as f32 - 33.0) * 0.015625)
        .collect()
}

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    let difference = (actual - expected).abs();
    let denominator = actual.abs().max(expected.abs()).max(1.0);
    assert!(
        difference / denominator <= tolerance,
        "actual={actual} expected={expected} difference={difference} tolerance={tolerance}"
    );
}

fn dense_rows(weight: &[f32], activation: &[f32]) -> Vec<f32> {
    weight
        .chunks_exact(COLS)
        .map(|row| {
            row.iter()
                .zip(activation)
                .map(|(left, right)| left * right)
                .sum()
        })
        .collect()
}

#[test]
fn q8_0_reference_hot_and_production_paths_agree() {
    assert_eq!(COLS % Q8_0_BLOCK_SIZE, 0);
    let source = weight_source();
    let compact = quantize_q8_0_tensor_ref(&source, ROWS, COLS).unwrap();
    let hot = repack_q8_0_tensor_cpu_ref(&compact, ROWS, COLS).unwrap();
    let activation = quantize_q8k_activation_vector_ref(&activation_source()).unwrap();
    let activation_hot = repack_q8k_activation_vector_cpu_ref(&activation).unwrap();

    let mut reference = [0.0_f32; ROWS];
    qmatvec_q8_0_q8k_ref(&hot, &activation, &mut reference).unwrap();
    let mut production = [0.0_f32; ROWS];
    launch_qmatvec_q8_0_q8hot_cpu_portable(&hot, &activation_hot, &mut production).unwrap();
    assert_eq!(production, reference);

    let mut decoded_weight = vec![0.0_f32; ROWS * COLS];
    for (block, output) in compact
        .iter()
        .zip(decoded_weight.chunks_exact_mut(Q8_0_BLOCK_SIZE))
    {
        block.decode_ref(output).unwrap();
    }
    let mut decoded_activation = vec![0.0_f32; COLS];
    activation.decode_ref(&mut decoded_activation).unwrap();
    for (actual, expected) in reference
        .iter()
        .zip(dense_rows(&decoded_weight, &decoded_activation))
    {
        assert_close(*actual, expected, 1e-4);
    }
}

#[test]
fn q2k_reference_hot_and_production_paths_agree() {
    assert_eq!(COLS % Q2K_BLOCK_SIZE, 0);
    let source = weight_source();
    let compact = quantize_q2k_tensor_ref(&source, ROWS, COLS).unwrap();
    let hot = repack_q2k_tensor_cpu_ref(&compact, ROWS, COLS).unwrap();
    let activation = quantize_q8k_activation_vector_ref(&activation_source()).unwrap();
    let activation_hot = repack_q8k_activation_vector_cpu_ref(&activation).unwrap();

    let mut reference = [0.0_f32; ROWS];
    qmatvec_q2k_q8k_ref(&hot, &activation, &mut reference).unwrap();
    let mut production = [0.0_f32; ROWS];
    launch_qmatvec_q2k_q8hot_cpu_portable(&hot, &activation_hot, &mut production).unwrap();
    assert_eq!(production, reference);

    let mut decoded_weight = vec![0.0_f32; ROWS * COLS];
    for (block, output) in compact
        .iter()
        .zip(decoded_weight.chunks_exact_mut(Q2K_BLOCK_SIZE))
    {
        block.decode_ref(output).unwrap();
    }
    let mut decoded_activation = vec![0.0_f32; COLS];
    activation.decode_ref(&mut decoded_activation).unwrap();
    for (actual, expected) in reference
        .iter()
        .zip(dense_rows(&decoded_weight, &decoded_activation))
    {
        assert_close(*actual, expected, 1e-4);
    }
}

#[test]
fn production_preflight_preserves_short_output() {
    let source = weight_source();
    let q8 = quantize_q8_0_tensor_ref(&source, ROWS, COLS).unwrap();
    let q8_hot = repack_q8_0_tensor_cpu_ref(&q8, ROWS, COLS).unwrap();
    let q2 = quantize_q2k_tensor_ref(&source, ROWS, COLS).unwrap();
    let q2_hot = repack_q2k_tensor_cpu_ref(&q2, ROWS, COLS).unwrap();
    let activation = quantize_q8k_activation_vector_ref(&activation_source()).unwrap();
    let activation_hot = repack_q8k_activation_vector_cpu_ref(&activation).unwrap();
    let mut output = [123.0_f32; 2];

    assert_eq!(
        launch_qmatvec_q8_0_q8hot_cpu_portable(&q8_hot, &activation_hot, &mut output),
        Err(KernelLaunchError::BadOutputLength {
            expected: ROWS,
            actual: 2
        })
    );
    assert_eq!(output, [123.0; 2]);
    assert_eq!(
        launch_qmatvec_q2k_q8hot_cpu_portable(&q2_hot, &activation_hot, &mut output),
        Err(KernelLaunchError::BadOutputLength {
            expected: ROWS,
            actual: 2
        })
    );
    assert_eq!(output, [123.0; 2]);
}
