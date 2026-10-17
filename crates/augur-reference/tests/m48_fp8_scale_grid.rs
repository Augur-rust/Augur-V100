use augur_formats::Fp8E4M3ScaleGridCompactTensor;
use augur_kernels::portable::qmatvec_fp8_e4m3_scale_grid_f32_cpu_portable;
use augur_reference::qmatvec_fp8_e4m3_scale_grid_f32_ref;
use augur_repack::repack_fp8_e4m3_scale_grid_tensor_cpu;

#[test]
fn scale_grid_reference_repack_and_kernel_agree_across_tile_rows() {
    let rows = 129;
    let cols = 256;
    let values = (0..rows * cols)
        .map(|index| match index % 6 {
            0 => 0x38,
            1 => 0x30,
            2 => 0xb8,
            3 => 0x40,
            4 => 0x20,
            _ => 0x00,
        })
        .collect();
    let compact =
        Fp8E4M3ScaleGridCompactTensor::new(rows, cols, values, vec![124, 126, 128, 130]).unwrap();
    let hot = repack_fp8_e4m3_scale_grid_tensor_cpu(&compact).unwrap();
    let activation = (0..cols)
        .map(|index| ((index as f32 - 127.0) / 31.0).sin())
        .collect::<Vec<_>>();
    let mut reference = vec![0.0; rows];
    let mut production = vec![0.0; rows];
    qmatvec_fp8_e4m3_scale_grid_f32_ref(&compact, &activation, &mut reference).unwrap();
    qmatvec_fp8_e4m3_scale_grid_f32_cpu_portable(&hot, &activation, &mut production).unwrap();
    assert_eq!(production, reference);
    assert_ne!(production[127], production[128]);
}

#[test]
fn scale_grid_repack_rejects_nan_code_and_nan_scale_encoding() {
    let bad_value = Fp8E4M3ScaleGridCompactTensor::new(1, 128, vec![0x7f; 128], vec![127]).unwrap();
    assert!(repack_fp8_e4m3_scale_grid_tensor_cpu(&bad_value).is_err());
    let bad_scale = Fp8E4M3ScaleGridCompactTensor::new(1, 128, vec![0x38; 128], vec![255]).unwrap();
    assert!(repack_fp8_e4m3_scale_grid_tensor_cpu(&bad_scale).is_err());
}
