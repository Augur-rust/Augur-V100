use std::sync::Arc;

use augur_backend::CpuStream;
use augur_compiler::{CpuCompiledStep, CpuWeightResources, GraphCompiler};
use augur_core::{QuantFormatId, Shape, StorageRef, TensorDesc};
use augur_exec::PlannedOp;
use augur_formats::{
    quantize_q2k_tensor_ref, quantize_q8_0_tensor_ref, quantize_q8k_activation_vector_ref,
};
use augur_graph::Graph;
use augur_reference::{qmatvec_q2k_q8k_ref, qmatvec_q8_0_q8k_ref};
use augur_repack::{repack_q2k_tensor_cpu_ref, repack_q8_0_tensor_cpu_ref};

const ROWS: usize = 4;
const COLS: usize = 512;

fn source() -> Vec<f32> {
    (0..ROWS * COLS)
        .map(|index| (((index * 19) % 113) as f32 - 56.0) * 0.00390625)
        .collect()
}

fn input() -> Vec<f32> {
    (0..COLS)
        .map(|index| (((index * 7) % 71) as f32 - 35.0) * 0.03125)
        .collect()
}

#[test]
fn graph_compile_bind_execute_consumes_both_m43_formats() {
    let q8_compact = quantize_q8_0_tensor_ref(&source(), ROWS, COLS).unwrap();
    let q2_compact = quantize_q2k_tensor_ref(&source(), ROWS, COLS).unwrap();
    let q8_hot = repack_q8_0_tensor_cpu_ref(&q8_compact, ROWS, COLS).unwrap();
    let q2_hot = repack_q2k_tensor_cpu_ref(&q2_compact, ROWS, COLS).unwrap();
    let q8_reference = q8_hot.clone();
    let q2_reference = q2_hot.clone();

    let mut resources = CpuWeightResources::new();
    let q8_storage = resources.add_q8_0_hot_cpu_portable(q8_hot).unwrap();
    let q2_storage = resources.add_q2k_hot_cpu_portable(q2_hot).unwrap();

    let mut graph = Graph::new();
    let input_tensor = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[COLS])));
    let q8_weight = graph
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[ROWS, COLS]),
            StorageRef::HostOwned { id: q8_storage },
            QuantFormatId::Q8_0,
            32,
            1,
        ))
        .unwrap();
    let q2_weight = graph
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[ROWS, COLS]),
            StorageRef::HostOwned { id: q2_storage },
            QuantFormatId::Q2K,
            256,
            1,
        ))
        .unwrap();
    let q8_output = graph.add_qmatmul(
        input_tensor,
        q8_weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[ROWS])),
    );
    let q2_output = graph.add_qmatmul(
        input_tensor,
        q2_weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[ROWS])),
    );
    graph.try_set_output(q8_output).unwrap();
    graph.try_set_output(q2_output).unwrap();

    let input = input();
    let compiled = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![input.clone()])
        .unwrap();
    assert_eq!(compiled.steps().len(), 3);
    assert_eq!(
        compiled
            .steps()
            .iter()
            .filter(|step| matches!(step, CpuCompiledStep::QuantizeQ8Hot(_)))
            .count(),
        1
    );
    assert!(matches!(compiled.steps()[1], CpuCompiledStep::Q8_0Q8Hot(_)));
    assert!(matches!(compiled.steps()[2], CpuCompiledStep::Q2KQ8Hot(_)));
    assert_eq!(compiled.weight_requirements().len(), 2);

    let mut plan = augur_compiler::bind_cpu_plan(compiled, resources).unwrap();
    assert_eq!(plan.resource_accounting().unique_weight_count, 2);
    assert_eq!(plan.resource_accounting().executable_weight_refs, 2);
    let q8_owner = match &plan.steps()[1] {
        PlannedOp::Q8_0Q8Hot(args) => Arc::clone(args.weight()),
        other => panic!("unexpected q8 step: {other:?}"),
    };
    let q2_owner = match &plan.steps()[2] {
        PlannedOp::Q2KQ8Hot(args) => Arc::clone(args.weight()),
        other => panic!("unexpected q2 step: {other:?}"),
    };
    let q8_count = Arc::strong_count(&q8_owner);
    let q2_count = Arc::strong_count(&q2_owner);

    let activation = quantize_q8k_activation_vector_ref(&input).unwrap();
    let mut expected_q8 = [0.0_f32; ROWS];
    let mut expected_q2 = [0.0_f32; ROWS];
    qmatvec_q8_0_q8k_ref(&q8_reference, &activation, &mut expected_q8).unwrap();
    qmatvec_q2k_q8k_ref(&q2_reference, &activation, &mut expected_q2).unwrap();
    let q8_output_buffer = plan.memory().buffer_for(q8_output).unwrap();
    let q2_output_buffer = plan.memory().buffer_for(q2_output).unwrap();
    let mut stream = CpuStream;
    for _ in 0..8 {
        plan.execute_cpu(&mut stream).unwrap();
        assert_eq!(plan.output_f32(q8_output_buffer).unwrap(), expected_q8);
        assert_eq!(plan.output_f32(q2_output_buffer).unwrap(), expected_q2);
        assert_eq!(Arc::strong_count(&q8_owner), q8_count);
        assert_eq!(Arc::strong_count(&q2_owner), q2_count);
    }
    let telemetry = plan.execute_cpu_profiled(&mut stream).unwrap();
    assert_eq!(telemetry.step_count, 3);
    assert_eq!(telemetry.q4k_f32_ns, 0);
    assert_eq!(telemetry.other_ns, 0);
    assert_eq!(
        telemetry.total_ns(),
        telemetry.activation_quantize_ns
            + telemetry.q8_0_q8hot_ns
            + telemetry.q4k_f32_ns
            + telemetry.q2k_q8hot_ns
    );
}
