use augur_core::{CompileError, GgmlError, Shape, TensorDesc};
use augur_graph::{
    validate_graph, BoundaryId, Graph, HadamardAttrs, QdqAttrs, QdqFormat, SliceAttrs, SoftmaxAttrs,
};

fn f32(shape: &[usize]) -> TensorDesc {
    TensorDesc::f32_contiguous(Shape::from_dims(shape))
}

#[test]
fn invalid_slice_extent_fails_before_lowering() {
    let mut graph = Graph::new();
    let input = graph.add_input(f32(&[2, 4]));
    let output = graph.add_slice(
        input,
        SliceAttrs {
            axis: 1,
            start: 3,
            len: 2,
        },
        f32(&[2, 2]),
    );
    graph.try_set_output(output).unwrap();
    assert_eq!(
        validate_graph(&graph),
        Err(CompileError::Graph(GgmlError::ShapeMismatch))
    );
}

#[test]
fn qdq_group_must_partition_the_last_dimension() {
    let mut graph = Graph::new();
    let input = graph.add_input(f32(&[2, 65]));
    let output = graph.add_qdq(
        input,
        QdqAttrs {
            format: QdqFormat::Fp8E4M3,
            group_size: 64,
            boundary: BoundaryId(1),
        },
        f32(&[2, 65]),
    );
    graph.try_set_output(output).unwrap();
    assert_eq!(validate_graph(&graph), Err(CompileError::InvalidInput));
}

#[test]
fn hadamard_requires_power_of_two_width_and_softmax_axis_is_explicit() {
    let mut graph = Graph::new();
    let input = graph.add_input(f32(&[2, 8]));
    let softmax = graph.add_softmax(
        input,
        SoftmaxAttrs {
            axis: 1,
            boundary: Some(BoundaryId(2)),
        },
        f32(&[2, 8]),
    );
    let output = graph.add_hadamard(
        softmax,
        HadamardAttrs {
            normalized: true,
            boundary: BoundaryId(3),
        },
        f32(&[2, 8]),
    );
    graph.try_set_output(output).unwrap();
    validate_graph(&graph).unwrap();
}
