use augur_backend::CpuStream;
use augur_compiler::{bind_cpu_plan, CpuCompiledStep, CpuWeightResources, GraphCompiler};
use augur_core::{QuantFormatId, Shape, StorageRef, TensorDesc};
use augur_exec::PlannedOp;
use augur_formats::Fp8E4M3ScaleGridCompactTensor;
use augur_graph::{ClampAttrs, FusedMlpActivation, FusedMlpAttrs, Graph};
use augur_reference::qmatvec_fp8_e4m3_scale_grid_f32_ref;
use augur_repack::repack_fp8_e4m3_scale_grid_tensor_cpu;

fn compact(rows: usize, cols: usize, seed: usize) -> Fp8E4M3ScaleGridCompactTensor {
    let codes = [0x00, 0x20, 0x30, 0x38, 0x40, 0xb0, 0xb8];
    let values = (0..rows * cols)
        .map(|index| codes[(index + seed) % codes.len()])
        .collect();
    let scales = vec![124 + seed as u8; rows.div_ceil(128) * (cols / 128)];
    Fp8E4M3ScaleGridCompactTensor::new(rows, cols, values, scales).unwrap()
}

fn weight_desc(rows: usize, cols: usize, storage: augur_core::StorageId) -> TensorDesc {
    TensorDesc::quantized_weight(
        Shape::from_dims(&[rows, cols]),
        StorageRef::HostOwned { id: storage },
        QuantFormatId::FP8E4M3ScaleGrid,
        128,
        1,
    )
}

#[test]
fn fused_graph_lower_bind_execute_matches_unfused_compact_reference() {
    let hidden = 256;
    let intermediate = 128;
    let gate_compact = compact(intermediate, hidden, 0);
    let up_compact = compact(intermediate, hidden, 1);
    let down_compact = compact(hidden, intermediate, 2);
    let mut resources = CpuWeightResources::new();
    let gate_storage = resources
        .add_fp8_scale_grid_hot_cpu_portable(
            repack_fp8_e4m3_scale_grid_tensor_cpu(&gate_compact).unwrap(),
        )
        .unwrap();
    let up_storage = resources
        .add_fp8_scale_grid_hot_cpu_portable(
            repack_fp8_e4m3_scale_grid_tensor_cpu(&up_compact).unwrap(),
        )
        .unwrap();
    let down_storage = resources
        .add_fp8_scale_grid_hot_cpu_portable(
            repack_fp8_e4m3_scale_grid_tensor_cpu(&down_compact).unwrap(),
        )
        .unwrap();

    let input_values = (0..hidden)
        .map(|index| ((index as f32 - 127.0) / 23.0).sin())
        .collect::<Vec<_>>();
    let mut graph = Graph::new();
    let input = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[hidden])));
    let gate = graph
        .add_weight(weight_desc(intermediate, hidden, gate_storage))
        .unwrap();
    let up = graph
        .add_weight(weight_desc(intermediate, hidden, up_storage))
        .unwrap();
    let down = graph
        .add_weight(weight_desc(hidden, intermediate, down_storage))
        .unwrap();
    let output = graph.add_fused_mlp(
        input,
        gate,
        up,
        down,
        FusedMlpAttrs {
            gate_clamp: ClampAttrs::try_new(None, Some(10.0)).unwrap(),
            up_clamp: ClampAttrs::try_new(Some(-10.0), Some(10.0)).unwrap(),
            activation: FusedMlpActivation::Silu,
        },
        TensorDesc::f32_contiguous(Shape::from_dims(&[hidden])),
    );
    graph.try_set_output(output).unwrap();

    let compiled = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![input_values.clone()])
        .unwrap();
    assert!(matches!(
        compiled.steps(),
        [CpuCompiledStep::Fp8FusedMlpF32(_)]
    ));
    let mut plan = bind_cpu_plan(compiled, resources).unwrap();
    let planned = match &plan.steps()[0] {
        PlannedOp::Fp8FusedMlpF32(args) => args,
        other => panic!("unexpected planned op: {other:?}"),
    };
    assert_eq!(planned.workspace_payload_bytes(), intermediate * 2 * 4);
    assert_eq!(plan.resource_accounting().unique_weight_count, 3);
    assert_eq!(plan.resource_accounting().executable_weight_refs, 3);

    let mut gate_ref = vec![0.0; intermediate];
    let mut up_ref = vec![0.0; intermediate];
    qmatvec_fp8_e4m3_scale_grid_f32_ref(&gate_compact, &input_values, &mut gate_ref).unwrap();
    qmatvec_fp8_e4m3_scale_grid_f32_ref(&up_compact, &input_values, &mut up_ref).unwrap();
    for (gate, up) in gate_ref.iter_mut().zip(&up_ref) {
        let gate_value = (*gate).min(10.0_f32);
        *gate = gate_value / (1.0_f32 + (-gate_value).exp()) * (*up).clamp(-10.0, 10.0);
    }
    let mut expected = vec![0.0; hidden];
    qmatvec_fp8_e4m3_scale_grid_f32_ref(&down_compact, &gate_ref, &mut expected).unwrap();

    let output_buffer = plan.memory().buffer_for(output).unwrap();
    let mut stream = CpuStream;
    plan.execute_cpu(&mut stream).unwrap();
    assert_eq!(plan.output_f32(output_buffer).unwrap(), expected);
    plan.execute_cpu(&mut stream).unwrap();
    assert_eq!(plan.output_f32(output_buffer).unwrap(), expected);
}

#[test]
fn standalone_fp8_qmatmul_uses_distinct_registered_step() {
    let rows = 128;
    let cols = 256;
    let compact = compact(rows, cols, 0);
    let mut resources = CpuWeightResources::new();
    let storage = resources
        .add_fp8_scale_grid_hot_cpu_portable(
            repack_fp8_e4m3_scale_grid_tensor_cpu(&compact).unwrap(),
        )
        .unwrap();
    let input_values = vec![0.25; cols];
    let mut graph = Graph::new();
    let input = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[cols])));
    let weight = graph.add_weight(weight_desc(rows, cols, storage)).unwrap();
    let output = graph.add_qmatmul(
        input,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
    );
    graph.try_set_output(output).unwrap();
    let compiled = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![input_values.clone()])
        .unwrap();
    assert!(matches!(
        compiled.steps(),
        [CpuCompiledStep::Fp8ScaleGridF32(_)]
    ));
    let mut plan = bind_cpu_plan(compiled, resources).unwrap();
    assert!(matches!(plan.steps(), [PlannedOp::Fp8ScaleGridF32(_)]));
    let mut expected = vec![0.0; rows];
    qmatvec_fp8_e4m3_scale_grid_f32_ref(&compact, &input_values, &mut expected).unwrap();
    plan.execute_cpu(&mut CpuStream).unwrap();
    let output_buffer = plan.memory().buffer_for(output).unwrap();
    assert_eq!(plan.output_f32(output_buffer).unwrap(), expected);
}
