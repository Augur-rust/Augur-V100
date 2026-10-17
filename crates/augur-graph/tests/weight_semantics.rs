use augur_core::{
    CompileError, DType, GgmlError, LogicalLayout, QuantDesc, QuantFormatId, Shape, StorageId,
    StorageRef, TensorDesc,
};
use augur_graph::{validate_graph, Graph};

fn weight_desc(rows: usize, cols: usize, format: QuantFormatId) -> TensorDesc {
    TensorDesc::quantized_weight(
        Shape::from_dims(&[rows, cols]),
        StorageRef::HostOwned {
            id: StorageId::new(7, 0),
        },
        format,
        256,
        1,
    )
}

#[test]
fn graph_accepts_q4k_weight_tensor() {
    let mut graph = Graph::new();

    let activation = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[256])));
    let weight = graph
        .add_weight(weight_desc(2, 256, QuantFormatId::Q4K))
        .unwrap();
    let output = graph.add_qmatmul(
        activation,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[2])),
    );

    graph.try_set_output(output).unwrap();
    assert_eq!(validate_graph(&graph), Ok(()));
}

#[test]
fn graph_accepts_dual_scale_weight_tensor() {
    let mut graph = Graph::new();

    let activation = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[512])));
    let weight = graph
        .add_weight(weight_desc(4, 512, QuantFormatId::DualScale))
        .unwrap();
    let output = graph.add_qmatmul(
        activation,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[4])),
    );

    graph.try_set_output(output).unwrap();
    assert_eq!(validate_graph(&graph), Ok(()));
}

#[test]
fn add_weight_rejects_missing_quant_metadata() {
    let mut graph = Graph::new();

    let result = graph.add_weight(TensorDesc {
        dtype: DType::PackedU8,
        shape: Shape::from_dims(&[2, 256]),
        layout: LogicalLayout::Contiguous,
        storage: StorageRef::HostOwned {
            id: StorageId::new(0, 0),
        },
        quant: None,
    });

    assert_eq!(result, Err(CompileError::InvalidInput));
}

#[test]
fn add_weight_rejects_missing_storage() {
    let mut graph = Graph::new();

    let result = graph.add_weight(TensorDesc {
        dtype: DType::PackedU8,
        shape: Shape::from_dims(&[2, 256]),
        layout: LogicalLayout::Contiguous,
        storage: StorageRef::None,
        quant: Some(QuantDesc {
            format: QuantFormatId::Q4K,
            block: 256,
            version: 1,
        }),
    });

    assert_eq!(result, Err(CompileError::InvalidInput));
}

#[test]
fn add_weight_rejects_non_matrix_shape() {
    let mut graph = Graph::new();

    let result = graph.add_weight(TensorDesc::quantized_weight(
        Shape::from_dims(&[512]),
        StorageRef::HostOwned {
            id: StorageId::new(0, 0),
        },
        QuantFormatId::Q4K,
        256,
        1,
    ));

    assert_eq!(result, Err(CompileError::InvalidInput));
}

#[test]
fn qmatmul_rejects_activation_width_mismatch() {
    let mut graph = Graph::new();

    let activation = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[128])));
    let weight = graph
        .add_weight(weight_desc(2, 256, QuantFormatId::Q4K))
        .unwrap();
    let output = graph.add_qmatmul(
        activation,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[2])),
    );

    graph.try_set_output(output).unwrap();
    assert_eq!(
        validate_graph(&graph),
        Err(CompileError::Graph(GgmlError::ShapeMismatch))
    );
}

#[test]
fn qmatmul_rejects_output_height_mismatch() {
    let mut graph = Graph::new();

    let activation = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[256])));
    let weight = graph
        .add_weight(weight_desc(2, 256, QuantFormatId::Q4K))
        .unwrap();
    let output = graph.add_qmatmul(
        activation,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[3])),
    );

    graph.try_set_output(output).unwrap();
    assert_eq!(
        validate_graph(&graph),
        Err(CompileError::Graph(GgmlError::ShapeMismatch))
    );
}

#[test]
fn qmatmul_bias_uses_third_input_without_attr_duplication() {
    let mut graph = Graph::new();

    let activation = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[256])));
    let weight = graph
        .add_weight(weight_desc(2, 256, QuantFormatId::Q4K))
        .unwrap();
    let bias = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[2])));
    let output = graph.add_qmatmul_bias(
        activation,
        weight,
        bias,
        TensorDesc::f32_contiguous(Shape::from_dims(&[2])),
    );

    graph.try_set_output(output).unwrap();
    assert_eq!(validate_graph(&graph), Ok(()));
}
