use augur_backend::CpuStream;
use augur_compiler::{CpuCompiledStep, GraphCompiler};
use augur_core::{Shape, TensorDesc};
use augur_exec::PlannedOp;
use augur_graph::Graph;
use augur_reference::DebugExecutor;

#[test]
fn m48_elementwise_graph_reference_and_execution_plan_agree() {
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[6]));
    let mut graph = Graph::new();
    let gate = graph.add_input(desc.clone());
    let up = graph.add_input(desc.clone());
    let gate = graph
        .add_clamp(gate, None, Some(10.0), desc.clone())
        .unwrap();
    let gate = graph.add_silu(gate, desc.clone());
    let up = graph
        .add_clamp(up, Some(-10.0), Some(10.0), desc.clone())
        .unwrap();
    let output = graph.add_mul(gate, up, desc);
    graph.try_set_output(output).unwrap();

    let gate_values = vec![-20.0, -1.0, 0.0, 1.0, 10.0, 20.0];
    let up_values = vec![-20.0, -3.0, 0.5, 2.0, 12.0, f32::NAN];
    let reference = DebugExecutor::run_f32_ordered(&graph, &[&gate_values, &up_values]).unwrap();
    let expected = reference.tensor_f32(output).unwrap();

    let compiled = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![gate_values, up_values])
        .unwrap();
    assert_eq!(
        compiled
            .steps()
            .iter()
            .filter(|step| matches!(step, CpuCompiledStep::Clamp(_)))
            .count(),
        2
    );
    assert_eq!(
        compiled
            .steps()
            .iter()
            .filter(|step| matches!(step, CpuCompiledStep::Silu(_)))
            .count(),
        1
    );
    assert!(matches!(
        compiled.steps().last(),
        Some(CpuCompiledStep::Mul(_))
    ));
    let mut plan =
        augur_compiler::bind_cpu_plan(compiled, augur_compiler::CpuWeightResources::new()).unwrap();
    assert!(matches!(
        plan.steps().last(),
        Some(PlannedOp::MulContiguous { .. })
    ));
    let output_buffer = plan.memory().buffer_for(output).unwrap();
    plan.execute_cpu(&mut CpuStream).unwrap();
    let actual = plan.output_f32(output_buffer).unwrap();
    for (actual, expected) in actual.iter().zip(expected) {
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn clamp_rejects_nonfinite_empty_and_reversed_bounds() {
    let mut graph = Graph::new();
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[1]));
    let input = graph.add_input(desc.clone());
    assert!(graph.add_clamp(input, None, None, desc.clone()).is_err());
    assert!(graph
        .add_clamp(input, Some(f32::NEG_INFINITY), None, desc.clone())
        .is_err());
    assert!(graph.add_clamp(input, Some(2.0), Some(1.0), desc).is_err());
}
