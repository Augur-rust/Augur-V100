use augur_compiler::{lower_m138_graph, M138StepKind};
use augur_core::{DType, LogicalLayout, Shape, StorageId, StorageRef, TensorDesc};
use augur_graph::{BoundaryId, Graph, PrecisionBoundaryAttrs};

fn f32(shape: &[usize]) -> TensorDesc {
    TensorDesc::f32_contiguous(Shape::from_dims(shape))
}

#[test]
fn staged_lowering_covers_every_node_and_preserves_boundaries() {
    let mut graph = Graph::new();
    let input = graph.add_input(f32(&[2, 4]));
    let weight = graph.add_tensor(TensorDesc {
        dtype: DType::BF16,
        shape: Shape::from_dims(&[4, 4]),
        layout: LogicalLayout::Contiguous,
        storage: StorageRef::HostOwned {
            id: StorageId::new(1, 0),
        },
        quant: None,
    });
    let matmul = graph.add_matmul(input, weight, f32(&[2, 4]));
    let boundary = graph.add_precision_boundary(
        matmul,
        PrecisionBoundaryAttrs {
            target: DType::BF16,
            boundary: BoundaryId(17),
        },
        f32(&[2, 4]),
    );
    graph.try_set_output(boundary).unwrap();
    let plan = lower_m138_graph(&graph).unwrap();
    assert_eq!(plan.steps().len(), graph.node_count());
    assert_eq!(plan.graph_tensor_count(), graph.tensor_count());
    assert_eq!(plan.boundaries(), &[BoundaryId(17)]);
    assert!(matches!(plan.steps()[0].kind, M138StepKind::Bf16Matmul));
    assert!(matches!(plan.steps()[1].kind, M138StepKind::Bf16Boundary));
    assert_eq!(plan.constants().len(), 1);
    assert!(plan
        .memory()
        .buffers()
        .iter()
        .any(|slot| slot.purpose == augur_exec::BufferPurpose::Output));
}

#[test]
fn unsupported_operation_fails_closed_before_plan_creation() {
    let mut graph = Graph::new();
    let input = graph.add_input(f32(&[2, 4]));
    let output = graph.add_add(input, input, f32(&[2, 4]));
    graph.try_set_output(output).unwrap();
    let error = lower_m138_graph(&graph).unwrap_err();
    assert!(matches!(
        error,
        augur_compiler::M138LoweringError::UnsupportedOp { .. }
    ));
}
