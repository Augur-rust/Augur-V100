use std::sync::Arc;

use augur_backend::CpuStream;
use augur_compiler::{bind_cpu_plan, CpuWeightResources, GraphCompiler};
use augur_core::{F16Bits, QuantFormatId, Shape, StorageRef, TensorDesc};
use augur_exec::PlannedOp;
use augur_graph::Graph;
use augur_kernel_api::{Q4KHotCpuBlock, Q4KHotCpuTensor};
use augur_kernels::portable::qmatmul_q4k_f32_portable;

fn q4k_tensor(rows: usize, cols: usize) -> Q4KHotCpuTensor {
    assert!(cols.is_multiple_of(Q4KHotCpuBlock::DECODED_LEN));
    let blocks_per_row = cols / Q4KHotCpuBlock::DECODED_LEN;
    let block_count = rows.checked_mul(blocks_per_row).unwrap();
    let blocks = (0..block_count)
        .map(|seed| Q4KHotCpuBlock {
            d: F16Bits::from_f32(1.0),
            dmin: F16Bits::from_f32(0.5),
            scale: [1 + (seed % 3) as u8; 8],
            min: [1; 8],
            qs: [seed as u8; 128],
        })
        .collect();
    Q4KHotCpuTensor::new(rows, cols, blocks).unwrap()
}

#[test]
fn executable_plan_outlives_consumed_resource_store() {
    let rows = 2;
    let cols = 256;
    let activation: Vec<f32> = (0..cols).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect();
    let direct_weight = q4k_tensor(rows, cols);
    let bound_weight = q4k_tensor(rows, cols);
    let mut expected = vec![0.0_f32; rows];
    qmatmul_q4k_f32_portable(&direct_weight, &activation, &mut expected).unwrap();

    let mut resources = CpuWeightResources::new();
    let storage = resources.add_q4k_hot_cpu_portable(bound_weight).unwrap();

    let mut graph = Graph::new();
    let input = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[cols])));
    let weight = graph
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[rows, cols]),
            StorageRef::HostOwned { id: storage },
            QuantFormatId::Q4K,
            256,
            1,
        ))
        .unwrap();
    let output = graph.add_qmatmul(
        input,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
    );
    graph.try_set_output(output).unwrap();

    let compiled = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![activation])
        .unwrap();
    // resources is consumed by value, proving the plan doesn't borrow the store
    let mut executable = bind_cpu_plan(compiled, resources).unwrap();

    let mut stream = CpuStream;
    executable.execute_cpu(&mut stream).unwrap();
    let output_index = executable.memory().buffer_for(output).unwrap();
    let actual = executable.buffers()[output_index].as_f32().unwrap();
    assert_eq!(actual, expected.as_slice());
}

#[test]
fn two_executable_steps_share_same_bound_owner() {
    let rows = 2;
    let cols = 256;
    let mut resources = CpuWeightResources::new();
    let storage = resources
        .add_q4k_hot_cpu_portable(q4k_tensor(rows, cols))
        .unwrap();

    let mut graph = Graph::new();
    let input = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[cols])));
    let weight = graph
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[rows, cols]),
            StorageRef::HostOwned { id: storage },
            QuantFormatId::Q4K,
            256,
            1,
        ))
        .unwrap();
    let output_a = graph.add_qmatmul(
        input,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
    );
    let output_b = graph.add_qmatmul(
        input,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
    );
    graph.try_set_output(output_a).unwrap();
    graph.try_set_output(output_b).unwrap();

    let compiled = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![vec![0.0_f32; cols]])
        .unwrap();
    assert_eq!(compiled.weight_requirements().len(), 1);
    let executable = bind_cpu_plan(compiled, resources).unwrap();

    let first = match &executable.steps()[0] {
        PlannedOp::QMatmulF32(a) => &a.base().weight(),
        other => panic!("unexpected first step: {other:?}"),
    };
    let second = match &executable.steps()[1] {
        PlannedOp::QMatmulF32(a) => &a.base().weight(),
        other => panic!("unexpected second step: {other:?}"),
    };
    assert!(Arc::ptr_eq(first, second));
    assert_eq!(executable.resource_accounting().unique_weight_count, 1);
    assert_eq!(executable.resource_accounting().executable_weight_refs, 2);
}

#[test]
fn add_only_plan_binds_and_executes_with_empty_resources() {
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
    let mut executable = bind_cpu_plan(compiled, CpuWeightResources::new()).unwrap();

    let mut stream = CpuStream;
    executable.execute_cpu(&mut stream).unwrap();
    let output_index = executable.memory().buffer_for(output).unwrap();
    assert_eq!(
        executable.buffers()[output_index].as_f32().unwrap(),
        &[11.0, 22.0, 33.0, 44.0]
    );
    assert_eq!(executable.resource_accounting(), Default::default());
}
