use augur_core::{DType, Shape, TensorDesc};
use augur_graph::{BoundaryId, Graph, PrecisionBoundaryAttrs, SoftmaxAttrs};
use augur_reference::DebugExecutor;

#[test]
fn stable_softmax_and_bf16_boundary_are_node_visible() {
    let mut graph = Graph::new();
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[2, 2]));
    let input = graph.add_input(desc.clone());
    let softmax = graph.add_softmax(
        input,
        SoftmaxAttrs {
            axis: 1,
            boundary: Some(BoundaryId(10)),
        },
        desc.clone(),
    );
    let output = graph.add_precision_boundary(
        softmax,
        PrecisionBoundaryAttrs {
            target: DType::BF16,
            boundary: BoundaryId(11),
        },
        desc,
    );
    graph.try_set_output(output).unwrap();
    let run = DebugExecutor::run_f32_ordered(&graph, &[&[1.0, 2.0, 3.0, 4.0]]).unwrap();
    let softmax_values = run.tensor_f32(softmax).unwrap();
    assert!((softmax_values[0] + softmax_values[1] - 1.0).abs() < 1.0e-6);
    assert_ne!(
        run.tensor_f32(output).unwrap().as_ptr(),
        softmax_values.as_ptr()
    );
}
