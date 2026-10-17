use augur_formats::{quantize_q8k_activation_vector_ref, Q8K_BLOCK_SIZE};
use augur_kernel_api::{
    KernelLaunchError, Q8KActivationHotCpuBlock, Q8KActivationHotCpuVector, QuantizeQ8HotCpuLaunch,
    Q8K_HOT_CPU_MAX_INPUT_ABS,
};
use augur_kernels::portable::{
    launch_quantize_q8k_hot_cpu_portable, quantize_q8k_hot_cpu_portable_into,
};
use augur_kernels::KernelError;
use augur_repack::repack_q8k_activation_vector_cpu_ref;

fn activation(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = i.wrapping_mul(37).wrapping_add(salt * 17) % 127;
            (x as f32 - 63.0) / 64.0
        })
        .collect()
}

fn assert_matches_reference(input: &[f32]) {
    let reference = quantize_q8k_activation_vector_ref(input).unwrap();
    let expected = repack_q8k_activation_vector_cpu_ref(&reference).unwrap();
    let mut actual = Q8KActivationHotCpuVector::new_zeroed(input.len()).unwrap();
    quantize_q8k_hot_cpu_portable_into(input, &mut actual).unwrap();
    assert_eq!(actual, expected);
}

fn seeded_nonzero_output(len: usize) -> Q8KActivationHotCpuVector {
    let block_count = len / Q8K_BLOCK_SIZE;
    let blocks = (0..block_count)
        .map(|i| Q8KActivationHotCpuBlock {
            d: augur_core::F16Bits::from_f32(i as f32 + 1.0),
            qs: [i as i8 + 3; Q8K_BLOCK_SIZE],
        })
        .collect();
    Q8KActivationHotCpuVector::new(len, blocks).unwrap()
}

#[test]
fn direct_q8hot_matches_reference_bit_exact() {
    for len in [256, 1024, 4096] {
        assert_matches_reference(&activation(len, 11));
    }
}

#[test]
fn zero_block_is_canonical_zero() {
    let input = vec![0.0_f32; Q8K_BLOCK_SIZE];
    let mut output = seeded_nonzero_output(Q8K_BLOCK_SIZE);
    quantize_q8k_hot_cpu_portable_into(&input, &mut output).unwrap();
    assert_eq!(output.blocks(), &[Q8KActivationHotCpuBlock::ZERO]);
}

#[test]
fn f16_scale_underflow_matches_reference() {
    let mut input = vec![0.0_f32; Q8K_BLOCK_SIZE];
    input[17] = f32::from_bits(1);
    assert_matches_reference(&input);
}

#[test]
fn bad_length_leaves_output_unchanged() {
    let input = vec![0.25_f32; Q8K_BLOCK_SIZE - 1];
    let mut output = seeded_nonzero_output(Q8K_BLOCK_SIZE);
    let before = output.clone();
    let pointer_before = output.blocks().as_ptr();

    assert_eq!(
        quantize_q8k_hot_cpu_portable_into(&input, &mut output),
        Err(KernelError::BadActivationLength {
            expected: Q8K_BLOCK_SIZE,
            actual: Q8K_BLOCK_SIZE - 1,
        }),
    );
    assert_eq!(output, before);
    assert_eq!(output.blocks().as_ptr(), pointer_before);
}

#[test]
fn non_finite_values_leave_output_unchanged() {
    for (index, invalid) in [(7, f32::NAN), (19, f32::INFINITY), (31, f32::NEG_INFINITY)] {
        let mut input = vec![0.25_f32; Q8K_BLOCK_SIZE * 2];
        input[index] = invalid;
        let mut output = seeded_nonzero_output(Q8K_BLOCK_SIZE * 2);
        let before = output.clone();
        let pointer_before = output.blocks().as_ptr();

        assert_eq!(
            quantize_q8k_hot_cpu_portable_into(&input, &mut output),
            Err(KernelError::NonFiniteActivation { index }),
        );
        assert_eq!(output, before);
        assert_eq!(output.blocks().as_ptr(), pointer_before);
    }
}

#[test]
fn later_block_error_does_not_modify_earlier_block() {
    let len = Q8K_BLOCK_SIZE * 2;
    let mut input = activation(len, 23);
    let invalid_index = Q8K_BLOCK_SIZE + 37;
    input[invalid_index] = f32::NAN;
    let mut output = seeded_nonzero_output(len);
    let before = output.clone();

    assert_eq!(
        quantize_q8k_hot_cpu_portable_into(&input, &mut output),
        Err(KernelError::NonFiniteActivation {
            index: invalid_index
        }),
    );
    assert_eq!(output, before);
}

#[test]
fn out_of_range_finite_value_leaves_output_unchanged() {
    let len = Q8K_BLOCK_SIZE * 2;
    let mut input = activation(len, 29);
    let invalid_index = Q8K_BLOCK_SIZE + 11;
    input[invalid_index] = Q8K_HOT_CPU_MAX_INPUT_ABS + 1024.0;
    assert!(input[invalid_index].is_finite());

    let mut output = seeded_nonzero_output(len);
    let before = output.clone();
    let pointer_before = output.blocks().as_ptr();

    assert_eq!(
        quantize_q8k_hot_cpu_portable_into(&input, &mut output),
        Err(KernelError::ActivationValueOutOfRange {
            index: invalid_index
        }),
    );
    assert_eq!(output, before);
    assert_eq!(output.blocks().as_ptr(), pointer_before);
}

#[test]
fn range_preflight_reports_first_offending_index() {
    let mut input = vec![0.25_f32; Q8K_BLOCK_SIZE * 2];
    let first = 17;
    let second = Q8K_BLOCK_SIZE + 9;
    input[first] = Q8K_HOT_CPU_MAX_INPUT_ABS + 1024.0;
    input[second] = -(Q8K_HOT_CPU_MAX_INPUT_ABS + 2048.0);

    let mut output = seeded_nonzero_output(Q8K_BLOCK_SIZE * 2);
    let before = output.clone();

    assert_eq!(
        quantize_q8k_hot_cpu_portable_into(&input, &mut output),
        Err(KernelError::ActivationValueOutOfRange { index: first }),
    );
    assert_eq!(output, before);
}

#[test]
fn maximum_accepted_value_produces_finite_scale() {
    let mut input = vec![0.0_f32; Q8K_BLOCK_SIZE];
    input[3] = Q8K_HOT_CPU_MAX_INPUT_ABS;
    input[9] = -Q8K_HOT_CPU_MAX_INPUT_ABS;

    let mut output = Q8KActivationHotCpuVector::new_zeroed(Q8K_BLOCK_SIZE).unwrap();
    quantize_q8k_hot_cpu_portable_into(&input, &mut output).unwrap();

    let scale = output.blocks()[0].d.to_f32();
    assert!(scale.is_finite());
    assert_eq!(scale, 65_504.0);
    assert_matches_reference(&input);
}

#[test]
fn repeated_quantization_preserves_owner_allocation() {
    let len = Q8K_BLOCK_SIZE * 4;
    let mut output = Q8KActivationHotCpuVector::new_zeroed(len).unwrap();
    let pointer_before = output.blocks().as_ptr();
    let block_count_before = output.block_count();
    let len_before = output.len_elements();

    for iteration in 0..32 {
        let input = activation(len, iteration + 41);
        quantize_q8k_hot_cpu_portable_into(&input, &mut output).unwrap();
        assert_eq!(output.blocks().as_ptr(), pointer_before);
        assert_eq!(output.block_count(), block_count_before);
        assert_eq!(output.len_elements(), len_before);
    }
}

#[test]
fn typed_launch_maps_errors_without_writing_output() {
    let launch: QuantizeQ8HotCpuLaunch = launch_quantize_q8k_hot_cpu_portable;
    let mut input = vec![0.25_f32; Q8K_BLOCK_SIZE];
    input[42] = f32::NAN;
    let mut output = seeded_nonzero_output(Q8K_BLOCK_SIZE);
    let before = output.clone();

    assert_eq!(
        launch(&input, &mut output),
        Err(KernelLaunchError::NonFiniteActivation { index: 42 }),
    );
    assert_eq!(output, before);
}

#[test]
fn typed_launch_maps_out_of_range_index() {
    let launch: QuantizeQ8HotCpuLaunch = launch_quantize_q8k_hot_cpu_portable;
    let invalid_index = 73;
    let mut input = vec![0.25_f32; Q8K_BLOCK_SIZE];
    input[invalid_index] = Q8K_HOT_CPU_MAX_INPUT_ABS + 1024.0;
    let mut output = seeded_nonzero_output(Q8K_BLOCK_SIZE);
    let before = output.clone();

    assert_eq!(
        launch(&input, &mut output),
        Err(KernelLaunchError::ActivationValueOutOfRange {
            index: invalid_index
        }),
    );
    assert_eq!(output, before);
}
