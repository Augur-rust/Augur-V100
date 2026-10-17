use augur_compiler::{CompilerError, CpuCompiledStep, GraphCompiler, LoweringError};
use augur_core::{QuantFormatId, Shape, StorageId, StorageRef, TensorDesc};
use augur_graph::Graph;

fn q4k_weight_desc(storage: StorageId, rows: usize, cols: usize) -> TensorDesc {
    TensorDesc::quantized_weight(
        Shape::from_dims(&[rows, cols]),
        StorageRef::HostOwned { id: storage },
        QuantFormatId::Q4K,
        256,
        1,
    )
}

#[test]
fn compile_cpu_plan_does_not_require_weight_owner() {
    let storage = StorageId::new(31, 7);

    let mut graph = Graph::new();

    let input = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[256])));

    let weight = graph.add_weight(q4k_weight_desc(storage, 2, 256)).unwrap();

    let output = graph.add_qmatmul(
        input,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[2])),
    );

    graph.try_set_output(output).unwrap();

    let compiled = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![vec![0.0_f32; 256]])
        .unwrap();

    assert_eq!(compiled.weight_requirements().len(), 1);

    let requirement = &compiled.weight_requirements()[0];

    assert_eq!(requirement.storage, storage);
    assert_eq!(requirement.rows, 2);
    assert_eq!(requirement.cols, 256);
    assert_eq!(requirement.quant_format, QuantFormatId::Q4K);

    match &compiled.steps()[0] {
        CpuCompiledStep::QMatmulF32(args) => {
            assert_eq!(args.base.weight, requirement.binding);
        }
        other => panic!("expected compiled QMatmulF32, got {other:?}"),
    }
}

#[test]
fn repeated_weight_consumer_reuses_one_requirement() {
    let storage = StorageId::new(9, 3);

    let mut graph = Graph::new();

    let input = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[256])));

    let weight = graph.add_weight(q4k_weight_desc(storage, 2, 256)).unwrap();

    let output_a = graph.add_qmatmul(
        input,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[2])),
    );
    let output_b = graph.add_qmatmul(
        input,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[2])),
    );

    graph.try_set_output(output_a).unwrap();
    graph.try_set_output(output_b).unwrap();

    let compiled = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![vec![0.0_f32; 256]])
        .unwrap();

    assert_eq!(compiled.weight_requirements().len(), 1);
    assert_eq!(compiled.steps().len(), 2);

    let first = match &compiled.steps()[0] {
        CpuCompiledStep::QMatmulF32(args) => args.base.weight,
        other => panic!("unexpected step: {other:?}"),
    };

    let second = match &compiled.steps()[1] {
        CpuCompiledStep::QMatmulF32(args) => args.base.weight,
        other => panic!("unexpected step: {other:?}"),
    };

    assert_eq!(first, second);
}

#[test]
fn conflicting_descriptors_for_same_storage_are_rejected() {
    let storage = StorageId::new(5, 1);

    let mut graph = Graph::new();

    let input_256 = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[256])));
    let input_512 = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[512])));

    let weight_256 = graph.add_weight(q4k_weight_desc(storage, 2, 256)).unwrap();
    let weight_512 = graph.add_weight(q4k_weight_desc(storage, 2, 512)).unwrap();

    let output_a = graph.add_qmatmul(
        input_256,
        weight_256,
        TensorDesc::f32_contiguous(Shape::from_dims(&[2])),
    );
    let output_b = graph.add_qmatmul(
        input_512,
        weight_512,
        TensorDesc::f32_contiguous(Shape::from_dims(&[2])),
    );

    graph.try_set_output(output_a).unwrap();
    graph.try_set_output(output_b).unwrap();

    let error = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![vec![0.0_f32; 256], vec![0.0_f32; 512]])
        .unwrap_err();

    assert_eq!(
        error,
        CompilerError::Lowering(LoweringError::ConflictingWeightRequirement { storage })
    );
}

#[test]
fn add_only_plan_has_no_weight_requirements() {
    let mut graph = Graph::new();

    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[4]));

    let a = graph.add_input(desc.clone());
    let b = graph.add_input(desc.clone());
    let output = graph.add_add(a, b, desc);

    graph.try_set_output(output).unwrap();

    let compiled = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(
            &graph,
            vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]],
        )
        .unwrap();

    assert!(compiled.weight_requirements().is_empty());
    assert!(matches!(compiled.steps()[0], CpuCompiledStep::Add(_)));
}
