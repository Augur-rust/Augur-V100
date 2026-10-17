#![cfg(target_arch = "x86_64")]

use std::sync::Arc;

use augur_backend::CpuStream;
use augur_compiler::{
    bind_cpu_plan, CpuCompiledPlan, CpuCompiledStep, CpuWeightResources, GraphCompiler,
};
use augur_core::{F16Bits, QuantFormatId, Shape, StorageRef, TensorDesc, TensorId};
use augur_exec::{ExecutionPlan, PlannedOp};
use augur_graph::Graph;
use augur_kernel_api::{Q4KHotCpuBlock, Q4KHotCpuTensor, QMatmulF32Launch};
use augur_kernels::{
    portable::{launch_qmatmul_q4k_f32_portable, qmatmul_q4k_f32_portable},
    x86::avx2::launch_qmatmul_q4k_f32_avx2,
};

const ROWS: usize = 16;
const COLS: usize = 1024;

fn host_supports_avx2_fma() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

fn q4k_hot_tensor(rows: usize, cols: usize) -> Q4KHotCpuTensor {
    assert!(rows > 0);
    assert!(cols > 0);
    assert_eq!(cols % Q4KHotCpuBlock::DECODED_LEN, 0);

    let blocks_per_row = cols / Q4KHotCpuBlock::DECODED_LEN;
    let blocks = (0..rows * blocks_per_row)
        .map(|block_index| {
            let mut qs = [0_u8; 128];
            for (index, packed) in qs.iter_mut().enumerate() {
                let low = index.wrapping_add(block_index) as u8 & 0x0f;
                let high = 15_u8.wrapping_sub(low);
                *packed = low | (high << 4);
            }
            Q4KHotCpuBlock {
                d: F16Bits::from_f32(0.25 + (block_index % 5) as f32 * 0.125),
                dmin: F16Bits::from_f32(0.0625 + (block_index % 3) as f32 * 0.0625),
                scale: core::array::from_fn(|group| ((block_index + group * 7) % 64) as u8),
                min: core::array::from_fn(|group| ((block_index * 3 + group * 11) % 64) as u8),
                qs,
            }
        })
        .collect();
    Q4KHotCpuTensor::new(rows, cols, blocks).unwrap()
}

fn activation(cols: usize, salt: usize) -> Vec<f32> {
    (0..cols)
        .map(|i| {
            let v = (i * 37 + salt * 17) % 127;
            (v as f32 - 63.0) / 64.0
        })
        .collect()
}

struct Q4KFixture {
    graph: Graph,
    resources: CpuWeightResources,
    input: Vec<f32>,
    output_tensor: TensorId,
    reference_weight: Q4KHotCpuTensor,
}

fn q4k_fixture(rows: usize, cols: usize) -> Q4KFixture {
    let resource_weight = q4k_hot_tensor(rows, cols);
    let reference_weight = q4k_hot_tensor(rows, cols);
    let input = activation(cols, 811);

    let mut resources = CpuWeightResources::new();
    let storage = resources.add_q4k_hot_cpu_portable(resource_weight).unwrap();

    let mut graph = Graph::new();
    let input_tensor = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[cols])));
    let weight_tensor = graph
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[rows, cols]),
            StorageRef::HostOwned { id: storage },
            QuantFormatId::Q4K,
            Q4KHotCpuBlock::DECODED_LEN,
            1,
        ))
        .unwrap();
    let output_tensor = graph.add_qmatmul(
        input_tensor,
        weight_tensor,
        TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
    );
    graph.try_set_output(output_tensor).unwrap();

    Q4KFixture {
        graph,
        resources,
        input,
        output_tensor,
        reference_weight,
    }
}

fn compiled_qmatmul_launch(compiled: &CpuCompiledPlan) -> QMatmulF32Launch {
    match &compiled.steps()[0] {
        CpuCompiledStep::QMatmulF32(args) => args.launch,
        other => panic!("unexpected compiled step: {other:?}"),
    }
}

fn planned_qmatmul_launch(plan: &ExecutionPlan<augur_backend::CpuBackend>) -> QMatmulF32Launch {
    match &plan.steps()[0] {
        PlannedOp::QMatmulF32(args) => args.launch(),
        other => panic!("unexpected planned op: {other:?}"),
    }
}

fn planned_qmatmul_weight(plan: &ExecutionPlan<augur_backend::CpuBackend>) -> &Arc<Q4KHotCpuTensor> {
    match &plan.steps()[0] {
        PlannedOp::QMatmulF32(args) => args.base().weight(),
        other => panic!("unexpected planned op: {other:?}"),
    }
}

fn same_launch(actual: QMatmulF32Launch, expected: QMatmulF32Launch) -> bool {
    std::ptr::addr_eq(actual as *const (), expected as *const ())
}

fn assert_m27_close(got: f32, reference: f32, blocks_per_row: usize, row: usize) {
    let abs_err = (got - reference).abs();
    let tol = 2.0e-4 * blocks_per_row.max(1) as f32 + 2.0e-5 * reference.abs().max(1.0);
    assert!(
        abs_err <= tol,
        "row={row} got={got} reference={reference} abs_err={abs_err} tol={tol}",
    );
}

#[test]
fn native_launch_identity_survives_compile_and_bind() {
    let Q4KFixture {
        graph,
        resources,
        input,
        ..
    } = q4k_fixture(ROWS, COLS);

    let compiled = GraphCompiler::cpu_native()
        .compile_cpu_plan(&graph, vec![input])
        .unwrap();
    assert_eq!(compiled.step_count(), 1);

    let expected = if host_supports_avx2_fma() {
        launch_qmatmul_q4k_f32_avx2
    } else {
        launch_qmatmul_q4k_f32_portable
    };

    let compiled_launch = compiled_qmatmul_launch(&compiled);
    assert!(
        same_launch(compiled_launch, expected),
        "compiled launch mismatch"
    );

    let plan = bind_cpu_plan(compiled, resources).unwrap();
    assert_eq!(plan.steps().len(), 1);
    assert!(
        same_launch(planned_qmatmul_launch(&plan), expected),
        "binding changed launch"
    );
    assert!(
        same_launch(planned_qmatmul_launch(&plan), compiled_launch),
        "compiled/planned address differ"
    );
}

#[test]
fn baseline_plan_remains_portable() {
    let Q4KFixture {
        graph,
        resources,
        input,
        ..
    } = q4k_fixture(ROWS, COLS);

    let compiled = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![input])
        .unwrap();
    assert!(same_launch(
        compiled_qmatmul_launch(&compiled),
        launch_qmatmul_q4k_f32_portable
    ));

    let plan = bind_cpu_plan(compiled, resources).unwrap();
    assert!(same_launch(
        planned_qmatmul_launch(&plan),
        launch_qmatmul_q4k_f32_portable
    ));
}

#[test]
fn native_execution_plan_matches_portable_reference() {
    let Q4KFixture {
        graph,
        resources,
        input,
        output_tensor,
        reference_weight,
    } = q4k_fixture(ROWS, COLS);

    let mut reference = vec![0.0_f32; ROWS];
    qmatmul_q4k_f32_portable(&reference_weight, &input, &mut reference).unwrap();

    let compiled = GraphCompiler::cpu_native()
        .compile_cpu_plan(&graph, vec![input])
        .unwrap();
    let mut plan = bind_cpu_plan(compiled, resources).unwrap();

    let mut stream = CpuStream;
    plan.execute_cpu(&mut stream).unwrap();

    let output_index = plan.memory().buffer_for(output_tensor).unwrap();
    let output = plan.buffers()[output_index].as_f32().unwrap();
    assert_eq!(output.len(), ROWS);

    for row in 0..ROWS {
        assert_m27_close(
            output[row],
            reference[row],
            reference_weight.blocks_per_row(),
            row,
        );
    }
}

#[test]
fn repeated_native_execution_preserves_all_owners() {
    let Q4KFixture {
        graph,
        resources,
        input,
        output_tensor,
        ..
    } = q4k_fixture(ROWS, COLS);

    let compiled = GraphCompiler::cpu_native()
        .compile_cpu_plan(&graph, vec![input])
        .unwrap();
    let mut plan = bind_cpu_plan(compiled, resources).unwrap();
    let mut stream = CpuStream;

    plan.execute_cpu(&mut stream).unwrap();
    let output_index = plan.memory().buffer_for(output_tensor).unwrap();

    let (input_index, planned_output_index) = match &plan.steps()[0] {
        PlannedOp::QMatmulF32(args) => (
            args.base().input().buffer_index(),
            args.base().output().buffer_index(),
        ),
        other => panic!("unexpected planned op: {other:?}"),
    };
    assert_eq!(planned_output_index, output_index);

    let weight_arc_ptr = Arc::as_ptr(planned_qmatmul_weight(&plan));
    let weight_block_ptr = planned_qmatmul_weight(&plan).blocks().as_ptr();
    let input_ptr = plan.buffers()[input_index].as_f32().unwrap().as_ptr();
    let output_ptr = plan.buffers()[output_index].as_f32().unwrap().as_ptr();
    let launch = planned_qmatmul_launch(&plan);
    let step_count = plan.steps().len();
    let buffer_count = plan.buffers().len();

    for _ in 0..16 {
        plan.execute_cpu(&mut stream).unwrap();
        assert_eq!(Arc::as_ptr(planned_qmatmul_weight(&plan)), weight_arc_ptr);
        assert_eq!(
            planned_qmatmul_weight(&plan).blocks().as_ptr(),
            weight_block_ptr
        );
        assert_eq!(
            plan.buffers()[input_index].as_f32().unwrap().as_ptr(),
            input_ptr
        );
        assert_eq!(
            plan.buffers()[output_index].as_f32().unwrap().as_ptr(),
            output_ptr
        );
        assert!(same_launch(planned_qmatmul_launch(&plan), launch));
        assert_eq!(plan.steps().len(), step_count);
        assert_eq!(plan.buffers().len(), buffer_count);
    }
}

#[test]
fn native_plan_resource_accounting_is_stable() {
    let rows = 4;
    let cols = 1024;
    let Q4KFixture {
        graph,
        resources,
        input,
        ..
    } = q4k_fixture(rows, cols);

    let compiled = GraphCompiler::cpu_native()
        .compile_cpu_plan(&graph, vec![input])
        .unwrap();

    let expected_blocks = rows * (cols / Q4KHotCpuBlock::DECODED_LEN);
    let expected_payload = expected_blocks * core::mem::size_of::<Q4KHotCpuBlock>();

    let plan = bind_cpu_plan(compiled, resources).unwrap();
    assert_eq!(plan.resource_accounting().unique_weight_count, 1);
    assert_eq!(plan.resource_accounting().executable_weight_refs, 1);
    assert_eq!(
        plan.resource_accounting().hot_weight_payload_bytes,
        expected_payload
    );
}
