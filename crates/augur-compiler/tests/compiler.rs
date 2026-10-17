use augur_backend::CpuStream;
use augur_compiler::GraphCompiler;
use augur_core::{Shape, TensorDesc};
use augur_graph::Graph;

#[test]
fn graph_compiler_cpu_baseline_executes_add() {
    let mut g = Graph::new();
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[4]));
    let a = g.add_input(desc.clone());
    let b = g.add_input(desc.clone());
    let y = g.add_add(a, b, desc);
    g.try_set_output(y).unwrap();

    let compiler = GraphCompiler::cpu_baseline();
    let mut plan = compiler
        .compile_cpu(
            &g,
            vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]],
        )
        .unwrap();

    let mut stream = CpuStream;
    plan.execute_cpu(&mut stream).unwrap();

    let out = plan.buffers()[2].as_f32().unwrap();
    assert_eq!(out, &[11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn graph_compiler_moves_input_allocations_into_the_compiled_plan() {
    let mut graph = Graph::new();
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[4]));
    let left_tensor = graph.add_input(desc.clone());
    let right_tensor = graph.add_input(desc.clone());
    let output = graph.add_add(left_tensor, right_tensor, desc);
    graph.try_set_output(output).unwrap();

    let left = vec![1.0, 2.0, 3.0, 4.0];
    let right = vec![10.0, 20.0, 30.0, 40.0];
    let left_address = left.as_ptr();
    let right_address = right.as_ptr();

    let plan = GraphCompiler::cpu_baseline()
        .compile_cpu(&graph, vec![left, right])
        .unwrap();
    let left_buffer = plan.memory().buffer_for(left_tensor).unwrap();
    let right_buffer = plan.memory().buffer_for(right_tensor).unwrap();

    assert_eq!(
        plan.buffers()[left_buffer].as_f32().unwrap().as_ptr(),
        left_address
    );
    assert_eq!(
        plan.buffers()[right_buffer].as_f32().unwrap().as_ptr(),
        right_address
    );
}
