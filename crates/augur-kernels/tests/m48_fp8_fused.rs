use augur_formats::Fp8E4M3ScaleGridCompactTensor;
use augur_kernel_api::{Fp8FusedMlpF32Workspace, FusedMlpF32Config};
use augur_kernels::portable::launch_fp8_e4m3_scale_grid_fused_mlp_f32_cpu_portable;
use augur_reference::qmatvec_fp8_e4m3_scale_grid_f32_ref;
use augur_repack::repack_fp8_e4m3_scale_grid_tensor_cpu;

fn tensor(rows: usize, cols: usize, seed: usize) -> Fp8E4M3ScaleGridCompactTensor {
    let codes = [0x00, 0x20, 0x30, 0x38, 0x40, 0xb0, 0xb8];
    let values = (0..rows * cols)
        .map(|index| codes[(index + seed) % codes.len()])
        .collect();
    let scales = vec![124 + seed as u8; rows.div_ceil(128) * (cols / 128)];
    Fp8E4M3ScaleGridCompactTensor::new(rows, cols, values, scales).unwrap()
}

#[test]
fn fused_launch_matches_independent_three_matvec_reference() {
    let hidden = 256;
    let intermediate = 128;
    let gate = tensor(intermediate, hidden, 0);
    let up = tensor(intermediate, hidden, 1);
    let down = tensor(hidden, intermediate, 2);
    let gate_hot = repack_fp8_e4m3_scale_grid_tensor_cpu(&gate).unwrap();
    let up_hot = repack_fp8_e4m3_scale_grid_tensor_cpu(&up).unwrap();
    let down_hot = repack_fp8_e4m3_scale_grid_tensor_cpu(&down).unwrap();
    let input = (0..hidden)
        .map(|index| ((index as f32 - 127.0) / 19.0).cos())
        .collect::<Vec<_>>();
    let config = FusedMlpF32Config {
        gate_min: None,
        gate_max: Some(10.0),
        up_min: Some(-10.0),
        up_max: Some(10.0),
    };

    let mut gate_ref = vec![0.0; intermediate];
    let mut up_ref = vec![0.0; intermediate];
    qmatvec_fp8_e4m3_scale_grid_f32_ref(&gate, &input, &mut gate_ref).unwrap();
    qmatvec_fp8_e4m3_scale_grid_f32_ref(&up, &input, &mut up_ref).unwrap();
    for (gate, up) in gate_ref.iter_mut().zip(&mut up_ref) {
        let gate_value = (*gate).min(10.0_f32);
        *gate = gate_value / (1.0_f32 + (-gate_value).exp()) * (*up).clamp(-10.0, 10.0);
    }
    let mut expected = vec![0.0; hidden];
    qmatvec_fp8_e4m3_scale_grid_f32_ref(&down, &gate_ref, &mut expected).unwrap();

    let mut actual = vec![0.0; hidden];
    let mut workspace = Fp8FusedMlpF32Workspace::new(intermediate).unwrap();
    launch_fp8_e4m3_scale_grid_fused_mlp_f32_cpu_portable(
        &gate_hot,
        &up_hot,
        &down_hot,
        &input,
        config,
        &mut workspace,
        &mut actual,
    )
    .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(workspace.payload_bytes(), intermediate * 2 * 4);
}
