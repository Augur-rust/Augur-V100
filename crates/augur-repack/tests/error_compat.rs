use augur_repack::TensorError;

#[test]
fn q4k_cpu_tensor_error_reexport_preserves_old_path() {
    let err = augur_repack::q4k_cpu::TensorError::BadShape;
    assert_eq!(err, TensorError::BadShape);
}
