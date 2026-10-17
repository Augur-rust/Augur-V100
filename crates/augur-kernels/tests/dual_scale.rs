use augur_formats::{
    encode_dual_scale_block_ref, quantize_q8k_activation_vector_ref, DUAL_SCALE_BLOCK_SIZE,
};
use augur_kernels::portable::{
    qmatvec_dual_scale_f32_portable, qmatvec_dual_scale_q8k_hot_cpu_portable,
};
use augur_repack::{repack_dual_scale_tensor_cpu_ref, repack_q8k_activation_vector_cpu_ref};

fn fixture() -> (augur_kernel_api::DualScaleHotCpuTensor, Vec<f32>) {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;
    let blocks = (0..rows * 2)
        .map(|block| {
            let source = (0..DUAL_SCALE_BLOCK_SIZE)
                .map(|index| (((index + block) % 37) as f32 - 18.0) * 0.03125)
                .collect::<Vec<_>>();
            encode_dual_scale_block_ref(&source).expect("valid fixture block")
        })
        .collect::<Vec<_>>();
    let weight =
        repack_dual_scale_tensor_cpu_ref(rows, cols, &blocks).expect("valid fixture tensor");
    let activation = (0..cols)
        .map(|index| ((index % 29) as f32 - 14.0) * 0.0625)
        .collect();
    (weight, activation)
}

#[test]
fn dual_scale_f32_production_path_executes() {
    let (weight, activation) = fixture();
    let mut output = vec![0.0; weight.rows()];

    qmatvec_dual_scale_f32_portable(&weight, &activation, &mut output)
        .expect("valid production invocation");

    assert!(output.iter().all(|value| value.is_finite()));
}

#[test]
fn dual_scale_q8hot_production_path_executes() {
    let (weight, activation) = fixture();
    let q8 = quantize_q8k_activation_vector_ref(&activation).expect("valid activation");
    let q8_hot = repack_q8k_activation_vector_cpu_ref(&q8).expect("valid hot activation");
    let mut output = vec![0.0; weight.rows()];

    qmatvec_dual_scale_q8k_hot_cpu_portable(&weight, &q8_hot, &mut output)
        .expect("valid production invocation");

    assert!(output.iter().all(|value| value.is_finite()));
}
