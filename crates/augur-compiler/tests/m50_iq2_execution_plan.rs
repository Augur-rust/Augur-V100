use std::sync::Arc;

use augur_backend::CpuStream;
use augur_compiler::{CpuCompiledStep, CpuWeightResources, GraphCompiler};
use augur_core::{CompileError, QuantFormatId, Shape, StorageRef, TensorDesc};
use augur_exec::PlannedOp;
use augur_formats::{
    quantize_iq2_xxs_tensor_ref, quantize_q2k_tensor_ref, quantize_q8k_activation_vector_ref,
};
use augur_graph::Graph;
use augur_reference::{qmatvec_iq2_xxs_q8k_ref, qmatvec_q2k_q8k_ref};
use augur_repack::{repack_iq2_xxs_tensor_cpu_ref, repack_q2k_tensor_cpu_ref};

const ROWS: usize = 4;
const COLS: usize = 512;

fn source() -> Vec<f32> {
    (0..ROWS * COLS)
        .map(|index| (((index * 23) % 127) as f32 - 63.0) * 0.00390625)
        .collect()
}

fn input() -> Vec<f32> {
    (0..COLS)
        .map(|index| (((index * 11) % 79) as f32 - 39.0) * 0.03125)
        .collect()
}

#[test]
fn graph_compile_bind_execute_reuses_q8hot_for_iq2_and_q2() {
    let source = source();
    let iq2 = quantize_iq2_xxs_tensor_ref(&source, ROWS, COLS).unwrap();
    let q2 = quantize_q2k_tensor_ref(&source, ROWS, COLS).unwrap();
    let iq2_hot = repack_iq2_xxs_tensor_cpu_ref(&iq2, ROWS, COLS).unwrap();
    let q2_hot = repack_q2k_tensor_cpu_ref(&q2, ROWS, COLS).unwrap();
    let iq2_reference = iq2_hot.clone();
    let q2_reference = q2_hot.clone();

    let mut resources = CpuWeightResources::new();
    let iq2_storage = resources.add_iq2_xxs_hot_cpu_portable(iq2_hot).unwrap();
    let q2_storage = resources.add_q2k_hot_cpu_portable(q2_hot).unwrap();
    let mut graph = Graph::new();
    let input_tensor = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[COLS])));
    let iq2_weight = graph
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[ROWS, COLS]),
            StorageRef::HostOwned { id: iq2_storage },
            QuantFormatId::IQ2_XXS,
            256,
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
    let output_desc = TensorDesc::f32_contiguous(Shape::from_dims(&[ROWS]));
    let iq2_output = graph.add_qmatmul(input_tensor, iq2_weight, output_desc.clone());
    let q2_output = graph.add_qmatmul(input_tensor, q2_weight, output_desc);
    graph.try_set_output(iq2_output).unwrap();
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
    assert!(matches!(
        compiled.steps()[1],
        CpuCompiledStep::Iq2XxsQ8Hot(_)
    ));
    assert!(matches!(compiled.steps()[2], CpuCompiledStep::Q2KQ8Hot(_)));

    let mut plan = augur_compiler::bind_cpu_plan(compiled, resources).unwrap();
    assert_eq!(plan.resource_accounting().unique_weight_count, 2);
    assert_eq!(plan.resource_accounting().executable_weight_refs, 2);
    let iq2_owner = match &plan.steps()[1] {
        PlannedOp::Iq2XxsQ8Hot(args) => Arc::clone(args.weight()),
        other => panic!("unexpected IQ2 step: {other:?}"),
    };
    let owner_count = Arc::strong_count(&iq2_owner);
    let activation = quantize_q8k_activation_vector_ref(&input).unwrap();
    let mut expected_iq2 = [0.0_f32; ROWS];
    let mut expected_q2 = [0.0_f32; ROWS];
    qmatvec_iq2_xxs_q8k_ref(&iq2_reference, &activation, &mut expected_iq2).unwrap();
    qmatvec_q2k_q8k_ref(&q2_reference, &activation, &mut expected_q2).unwrap();
    let iq2_buffer = plan.memory().buffer_for(iq2_output).unwrap();
    let q2_buffer = plan.memory().buffer_for(q2_output).unwrap();
    for _ in 0..4 {
        plan.execute_cpu(&mut CpuStream).unwrap();
        for (actual, expected) in plan
            .output_f32(iq2_buffer)
            .unwrap()
            .iter()
            .zip(expected_iq2)
        {
            assert!((actual - expected).abs() <= 2.0e-5);
        }
        assert_eq!(plan.output_f32(q2_buffer).unwrap(), expected_q2);
        assert_eq!(Arc::strong_count(&iq2_owner), owner_count);
    }
    let telemetry = plan.execute_cpu_profiled(&mut CpuStream).unwrap();
    assert!(telemetry.activation_quantize_ns > 0);
    assert!(telemetry.iq2_xxs_q8hot_ns > 0);
    assert!(telemetry.q2k_q8hot_ns > 0);
    assert_eq!(telemetry.other_ns, 0);
}

#[test]
fn compiler_rejects_iq2_with_wrong_block_contract() {
    let mut graph = Graph::new();
    let input_tensor = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[COLS])));
    let weight = graph
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[ROWS, COLS]),
            StorageRef::HostOwned {
                id: augur_core::StorageId {
                    index: 0,
                    generation: 0,
                },
            },
            QuantFormatId::IQ2_XXS,
            32,
            1,
        ))
        .unwrap();
    let output = graph.add_qmatmul(
        input_tensor,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[ROWS])),
    );
    graph.try_set_output(output).unwrap();
    assert!(matches!(
        GraphCompiler::cpu_baseline().compile_cpu_plan(&graph, vec![vec![0.0; COLS]]),
        Err(augur_compiler::CompilerError::Lowering(
            augur_compiler::LoweringError::Core(CompileError::BackendUnsupported)
        ))
    ));
}
