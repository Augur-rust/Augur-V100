use augur_compiler::GraphCompiler;
use augur_core::{Shape, TensorDesc};
use augur_exec::BufferPurpose;
use augur_graph::Graph;

#[test]
fn memory_plan_records_tensor_allocations() {
    let mut g = Graph::new();
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[3]));
    let a = g.add_input(desc.clone());
    let b = g.add_input(desc.clone());
    let y1 = g.add_add(a, b, desc.clone());
    let y2 = g.add_add(y1, a, desc);
    g.try_set_output(y2).unwrap();

    let plan = GraphCompiler::cpu_baseline()
        .compile_cpu(&g, vec![vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]])
        .unwrap();

    assert_eq!(plan.buffers().len(), 4);
    assert_eq!(plan.memory().buffers().len(), 4);

    assert_eq!(plan.memory().buffer_for(a), Some(0));
    assert_eq!(plan.memory().buffer_for(b), Some(1));
    assert_eq!(plan.memory().buffer_for(y1), Some(2));
    assert_eq!(plan.memory().buffer_for(y2), Some(3));

    assert_eq!(plan.memory().buffers()[0].purpose, BufferPurpose::Input);
    assert_eq!(plan.memory().buffers()[1].purpose, BufferPurpose::Input);
    assert_eq!(
        plan.memory().buffers()[2].purpose,
        BufferPurpose::Intermediate
    );
    assert_eq!(plan.memory().buffers()[3].purpose, BufferPurpose::Output);

    assert_eq!(plan.memory().buffers()[3].len_elements, 3);

    let y2_buf = plan.memory().buffer_for(y2).unwrap();
    assert_eq!(
        plan.memory().buffers()[y2_buf].purpose,
        BufferPurpose::Output
    );
    assert_eq!(plan.memory().buffers()[y2_buf].len_elements, 3);
}
