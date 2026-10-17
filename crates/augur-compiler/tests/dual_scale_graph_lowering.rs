use std::sync::Arc;

use augur_backend::CpuStream;
use augur_compiler::{CpuCompiledStep, CpuWeightResources, GraphCompiler};
use augur_core::{F16Bits, QuantFormatId, Shape, StorageRef, TensorDesc};
use augur_exec::{BufferIdentity, BufferLifetimeEnd, BufferPurpose, PlannedOp};
use augur_graph::Graph;
use augur_kernel_api::{
    DualScaleHotCpuBlock, DualScaleHotCpuTensor, OutlierCsrHotCpu, DUAL_SCALE_HOT_CPU_GROUPS,
    DUAL_SCALE_HOT_CPU_MEMBERSHIP_BYTES, DUAL_SCALE_HOT_CPU_QS_BYTES,
};

fn dual_scale_tensor(rows: usize, cols: usize, code: u8) -> DualScaleHotCpuTensor {
    assert!(cols.is_multiple_of(DualScaleHotCpuBlock::DECODED_LEN));

    let blocks_per_row = cols / DualScaleHotCpuBlock::DECODED_LEN;
    let block_count = rows.checked_mul(blocks_per_row).unwrap();

    let block = DualScaleHotCpuBlock {
        scale_lo: [F16Bits::from_f32(0.25); DUAL_SCALE_HOT_CPU_GROUPS],
        scale_hi: [F16Bits::from_f32(0.5); DUAL_SCALE_HOT_CPU_GROUPS],
        membership: [0; DUAL_SCALE_HOT_CPU_MEMBERSHIP_BYTES],
        qs: [code; DUAL_SCALE_HOT_CPU_QS_BYTES],
    };

    let outliers =
        OutlierCsrHotCpu::new(rows, cols, vec![0; rows + 1], Vec::new(), Vec::new()).unwrap();

    DualScaleHotCpuTensor::new(rows, cols, vec![block; block_count], outliers).unwrap()
}

fn add_dual_scale_weight(
    graph: &mut Graph,
    storage: augur_core::StorageId,
    rows: usize,
    cols: usize,
) -> augur_core::TensorId {
    graph
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[rows, cols]),
            StorageRef::HostOwned { id: storage },
            QuantFormatId::DualScale,
            256,
            1,
        ))
        .unwrap()
}

#[test]
fn graph_lowering_inserts_quantize_before_dual_scale() {
    let rows = 2;
    let cols = 256;

    let mut resources = CpuWeightResources::new();
    let storage = resources
        .add_dual_scale_hot_cpu_portable(dual_scale_tensor(rows, cols, 0x99))
        .unwrap();

    let mut graph = Graph::new();
    let input = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[cols])));
    let weight = add_dual_scale_weight(&mut graph, storage, rows, cols);
    let output = graph.add_qmatmul(
        input,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
    );
    graph.try_set_output(output).unwrap();

    let compiled = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![vec![0.25; cols]])
        .unwrap();

    assert_eq!(compiled.steps().len(), 2);

    let q8hot_buffer = match &compiled.steps()[0] {
        CpuCompiledStep::QuantizeQ8Hot(args) => args.output.buffer_index(),
        other => panic!("unexpected first step: {other:?}"),
    };

    match &compiled.steps()[1] {
        CpuCompiledStep::DualScaleQ8Hot(args) => {
            assert_eq!(args.activation.buffer_index(), q8hot_buffer);
            assert_eq!(args.rows, rows);
            assert_eq!(args.cols, cols);
            assert_eq!(args.blocks_per_row, 1);
        }
        other => panic!("unexpected second step: {other:?}"),
    }
}

#[test]
fn same_activation_two_consumers_reuse_one_q8hot() {
    let rows = 1;
    let cols = 256;

    let mut resources = CpuWeightResources::new();
    let storage_a = resources
        .add_dual_scale_hot_cpu_portable(dual_scale_tensor(rows, cols, 0x99))
        .unwrap();
    let storage_b = resources
        .add_dual_scale_hot_cpu_portable(dual_scale_tensor(rows, cols, 0xaa))
        .unwrap();

    let mut graph = Graph::new();
    let input = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[cols])));
    let weight_a = add_dual_scale_weight(&mut graph, storage_a, rows, cols);
    let weight_b = add_dual_scale_weight(&mut graph, storage_b, rows, cols);
    let output_a = graph.add_qmatmul(
        input,
        weight_a,
        TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
    );
    let output_b = graph.add_qmatmul(
        input,
        weight_b,
        TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
    );
    graph.try_set_output(output_a).unwrap();
    graph.try_set_output(output_b).unwrap();

    let compiled = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![vec![0.25; cols]])
        .unwrap();

    assert_eq!(compiled.steps().len(), 3);

    let quantize_count = compiled
        .steps()
        .iter()
        .filter(|step| matches!(step, CpuCompiledStep::QuantizeQ8Hot(_)))
        .count();
    assert_eq!(quantize_count, 1);

    let activation_buffers: Vec<usize> = compiled
        .steps()
        .iter()
        .filter_map(|step| match step {
            CpuCompiledStep::DualScaleQ8Hot(args) => Some(args.activation.buffer_index()),
            _ => None,
        })
        .collect();
    assert_eq!(activation_buffers.len(), 2);
    assert_eq!(activation_buffers[0], activation_buffers[1]);

    assert_eq!(compiled.weight_requirements().len(), 2);
    assert_eq!(compiled.memory().temporary_allocations().len(), 1);

    let allocation = &compiled.memory().temporary_allocations()[0];
    let slot = &compiled.memory().buffers()[allocation.buffer_index];

    assert_eq!(
        slot.identity,
        BufferIdentity::Temporary(allocation.temporary)
    );
    assert_eq!(slot.purpose, BufferPurpose::HotActivation);
    // step 0 = Quantize, step 1 = first DualScale consumer, step 2 = second
    assert_eq!(
        slot.reusable_after,
        Some(BufferLifetimeEnd::CompiledStep(2))
    );
}

#[test]
fn graph_compile_bind_execute_reuses_stable_owners() {
    let rows = 1;
    let cols = 256;

    let mut resources = CpuWeightResources::new();
    let storage = resources
        .add_dual_scale_hot_cpu_portable(dual_scale_tensor(rows, cols, 0x99))
        .unwrap();

    let mut graph = Graph::new();
    let input = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[cols])));
    let weight = add_dual_scale_weight(&mut graph, storage, rows, cols);
    let output = graph.add_qmatmul(
        input,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
    );
    graph.try_set_output(output).unwrap();

    let mut executable = GraphCompiler::cpu_baseline()
        .compile_cpu_with_weights(&graph, vec![vec![0.25; cols]], resources)
        .unwrap();

    assert_eq!(executable.steps().len(), 2);

    let temporary_buffer = executable.memory().temporary_allocations()[0].buffer_index;
    let output_buffer = executable.memory().buffer_for(output).unwrap();

    let q8hot_pointer = executable.buffers()[temporary_buffer]
        .as_q8hot()
        .unwrap()
        .blocks()
        .as_ptr();
    let output_pointer = executable.buffers()[output_buffer]
        .as_f32()
        .unwrap()
        .as_ptr();

    let expected_weight_owner = match &executable.steps()[1] {
        PlannedOp::DualScaleQ8Hot(args) => Arc::clone(args.weight()),
        other => panic!("unexpected step: {other:?}"),
    };

    let strong_count_before = Arc::strong_count(&expected_weight_owner);

    let mut stream = CpuStream;
    for _ in 0..16 {
        executable.execute_cpu(&mut stream).unwrap();

        assert_eq!(
            executable.buffers()[temporary_buffer]
                .as_q8hot()
                .unwrap()
                .blocks()
                .as_ptr(),
            q8hot_pointer
        );
        assert_eq!(
            executable.buffers()[output_buffer]
                .as_f32()
                .unwrap()
                .as_ptr(),
            output_pointer
        );

        match &executable.steps()[1] {
            PlannedOp::DualScaleQ8Hot(args) => {
                assert!(Arc::ptr_eq(args.weight(), &expected_weight_owner));
            }
            other => panic!("unexpected step: {other:?}"),
        }

        assert_eq!(
            Arc::strong_count(&expected_weight_owner),
            strong_count_before
        );
    }
}
