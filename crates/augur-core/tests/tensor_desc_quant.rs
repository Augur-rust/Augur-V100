use augur_core::{DType, LogicalLayout, QuantDesc, QuantFormatId, Shape, StorageRef, TensorDesc};

#[test]
fn tensor_desc_can_describe_quantized_activation_metadata() {
    let desc = TensorDesc {
        dtype: DType::I8,
        shape: Shape::from_dims(&[512]),
        layout: LogicalLayout::Contiguous,
        storage: StorageRef::None,
        quant: Some(QuantDesc {
            format: QuantFormatId::Q8K,
            block: 256,
            version: 1,
        }),
    };

    assert_eq!(desc.dtype, DType::I8);
    assert_eq!(desc.shape, Shape::from_dims(&[512]));
    assert_eq!(desc.layout, LogicalLayout::Contiguous);
    assert_eq!(
        desc.quant.as_ref().map(|q| q.format),
        Some(QuantFormatId::Q8K)
    );
    assert!(desc.is_quantized());
    assert!(!desc.is_plain_f32());
    assert!(desc.is_quant_format(QuantFormatId::Q8K));
    assert!(!desc.is_quant_format(QuantFormatId::Q4K));
}

#[test]
fn tensor_desc_plain_f32_is_not_quantized() {
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[256]));

    assert!(desc.is_plain_f32());
    assert!(!desc.is_quantized());
    assert!(!desc.is_quant_format(QuantFormatId::Q8K));
}
