use augur_core::{Shape, TensorDesc};
use augur_graph::Graph;
use augur_reference::{DebugExecError, DebugExecutor};

#[test]
fn debug_executor_add_matches_expected() {
    let mut g = Graph::new();
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[4]));
    let a = g.add_input(desc.clone());
    let b = g.add_input(desc.clone());
    let y = g.add_add(a, b, desc);
    g.try_set_output(y).unwrap();

    let run =
        DebugExecutor::run_f32_ordered(&g, &[&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]])
            .unwrap();
    assert_eq!(run.tensor_f32(y).unwrap(), &[11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn debug_executor_rejects_bad_input_length() {
    let mut g = Graph::new();
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[4]));
    let a = g.add_input(desc.clone());
    let b = g.add_input(desc.clone());
    let y = g.add_add(a, b, desc);
    g.try_set_output(y).unwrap();

    let err =
        DebugExecutor::run_f32_ordered(&g, &[&[1.0, 2.0], &[10.0, 20.0, 30.0, 40.0]]).unwrap_err();

    assert_eq!(
        err,
        DebugExecError::BadInputLength {
            tensor: a,
            expected: 4,
            actual: 2
        }
    );
}

#[test]
fn debug_executor_rejects_input_count_mismatch() {
    let mut g = Graph::new();
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[2]));
    let a = g.add_input(desc.clone());
    let b = g.add_input(desc);
    let _y = g.add_add(a, b, TensorDesc::f32_contiguous(Shape::from_dims(&[2])));

    let err = DebugExecutor::run_f32_ordered(&g, &[&[1.0, 2.0]]).unwrap_err();
    assert_eq!(
        err,
        DebugExecError::InputCountMismatch {
            expected: 2,
            actual: 1
        }
    );
}
