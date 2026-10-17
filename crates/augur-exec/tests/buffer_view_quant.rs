#![allow(clippy::needless_range_loop)]

use augur_backend::CpuBackend;
use augur_core::{DType, PhysicalLayoutId, QuantFormatId};
use augur_exec::BufferView;

#[test]
fn buffer_view_preserves_quantized_activation_metadata() {
    let view = BufferView::<CpuBackend>::new(
        7,
        512,
        DType::I8,
        Some(QuantFormatId::Q8K),
        PhysicalLayoutId::Q8K_ACTIVATION_REF_VECTOR_V1,
    );

    assert_eq!(view.buffer_index(), 7);
    assert_eq!(view.len(), 512);
    assert_eq!(view.dtype(), DType::I8);
    assert_eq!(view.quant(), Some(QuantFormatId::Q8K));
    assert_eq!(
        view.physical_layout(),
        PhysicalLayoutId::Q8K_ACTIVATION_REF_VECTOR_V1
    );
    assert!(!view.is_plain_f32());
}

#[test]
fn buffer_view_dense_f32_has_no_quant_metadata() {
    let view = BufferView::<CpuBackend>::dense_f32(3, 256);

    assert!(view.is_plain_f32());
    assert_eq!(view.quant(), None);
}

#[test]
fn cpu_dense_f32_predicate_requires_canonical_metadata() {
    let canonical = BufferView::<CpuBackend>::dense_f32(0, 256);
    assert!(canonical.is_cpu_contiguous_f32());

    let wrong_dtype = BufferView::<CpuBackend>::new(
        0,
        256,
        DType::I8,
        None,
        PhysicalLayoutId::HOST_CONTIGUOUS_F32,
    );
    assert!(!wrong_dtype.is_cpu_contiguous_f32());

    let wrong_quant = BufferView::<CpuBackend>::new(
        0,
        256,
        DType::F32,
        Some(QuantFormatId::Q8K),
        PhysicalLayoutId::HOST_CONTIGUOUS_F32,
    );
    assert!(!wrong_quant.is_cpu_contiguous_f32());

    let wrong_layout = BufferView::<CpuBackend>::new(
        0,
        256,
        DType::F32,
        None,
        PhysicalLayoutId::Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1,
    );
    assert!(!wrong_layout.is_cpu_contiguous_f32());
}

#[test]
fn cpu_q8hot_predicate_requires_canonical_metadata() {
    let canonical = BufferView::<CpuBackend>::q8hot_cpu(1, 256);
    assert!(canonical.is_cpu_q8hot());

    let wrong_dtype = BufferView::<CpuBackend>::new(
        1,
        256,
        DType::F32,
        Some(QuantFormatId::Q8K),
        PhysicalLayoutId::Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1,
    );
    assert!(!wrong_dtype.is_cpu_q8hot());

    let wrong_quant = BufferView::<CpuBackend>::new(
        1,
        256,
        DType::I8,
        None,
        PhysicalLayoutId::Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1,
    );
    assert!(!wrong_quant.is_cpu_q8hot());

    let wrong_layout = BufferView::<CpuBackend>::new(
        1,
        256,
        DType::I8,
        Some(QuantFormatId::Q8K),
        PhysicalLayoutId::HOST_CONTIGUOUS_F32,
    );
    assert!(!wrong_layout.is_cpu_q8hot());
}

#[test]
fn buffer_view_q8hot_cpu_uses_canonical_metadata() {
    let view = BufferView::<CpuBackend>::q8hot_cpu(5, 1024);

    assert_eq!(view.buffer_index(), 5);
    assert_eq!(view.len(), 1024);
    assert_eq!(view.dtype(), DType::I8);
    assert_eq!(view.quant(), Some(QuantFormatId::Q8K));
    assert_eq!(
        view.physical_layout(),
        PhysicalLayoutId::Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1
    );
    assert!(!view.is_plain_f32());
}
