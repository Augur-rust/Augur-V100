use augur_backend::CpuStream;
use augur_compiler::compile_cpu_add_only;
use augur_core::{Shape, TensorDesc};
use augur_graph::Graph;

#[test]
fn graph_add_compiles_and_executes() {
    let mut g = Graph::new();
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[4]));
    let a = g.add_input(desc.clone());
    let b = g.add_input(desc.clone());
    let y = g.add_add(a, b, desc);
    g.try_set_output(y).unwrap();

    let mut plan = compile_cpu_add_only(
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
fn graph_add_chain() {
    let mut g = Graph::new();
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[3]));
    let a = g.add_input(desc.clone());
    let b = g.add_input(desc.clone());
    let y1 = g.add_add(a, b, desc.clone());
    let y2 = g.add_add(y1, a, desc);
    g.try_set_output(y2).unwrap();

    let mut plan =
        compile_cpu_add_only(&g, vec![vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]]).unwrap();
    let mut stream = CpuStream;
    plan.execute_cpu(&mut stream).unwrap();

    let out = plan.buffers()[3].as_f32().unwrap();
    assert_eq!(out, &[12.0, 24.0, 36.0]);
}
