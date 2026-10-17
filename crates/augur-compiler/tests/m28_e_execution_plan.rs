#![cfg(target_arch = "x86_64")]

use std::sync::Arc;

use augur_backend::CpuStream;
use augur_compiler::{
    bind_cpu_plan, CpuCompiledPlan, CpuCompiledStep, CpuWeightResources, GraphCompiler,
};
use augur_core::{QuantFormatId, Shape, StorageRef, TensorDesc, TensorId};
use augur_exec::{ExecutionPlan, PlannedOp, TemporaryBufferId};
use augur_formats::{encode_dual_scale_block_ref, DualScaleCompactBlock, DUAL_SCALE_BLOCK_SIZE};
use augur_graph::Graph;
use augur_kernel_api::{
    DualScaleHotCpuTensor, DualScaleQ8HotCpuLaunch, DualScaleQ8HotCpuPrevalidatedLaunch,
};
use augur_kernels::{
    portable::{
        launch_qmatvec_dual_scale_q8k_hot_cpu_portable,
        launch_qmatvec_dual_scale_q8k_hot_cpu_portable_prevalidated,
        launch_quantize_q8k_hot_cpu_portable,
    },
    x86::avx2::{
        launch_qmatvec_dual_scale_q8k_hot_cpu_avx2,
        launch_qmatvec_dual_scale_q8k_hot_cpu_avx2_prevalidated,
    },
};
use augur_repack::repack_dual_scale_tensor_cpu_ref;

const ROWS: usize = 16;
const COLS: usize = 1024;

// ── Self-contained fixtures (no augur-bench dependency) ─────────────────────

#[derive(Clone, Copy)]
#[allow(dead_code)]
enum RequestedOutliers {
    Zero,
    Two,
}

fn activation(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let value = index.wrapping_mul(37).wrapping_add(salt.wrapping_mul(17)) % 127;
            (value as f32 - 63.0) / 64.0
        })
        .collect()
}

fn dual_scale_source_block(seed: usize, requested: RequestedOutliers) -> Vec<f32> {
    let mut source = (0..DUAL_SCALE_BLOCK_SIZE)
        .map(|index| {
            if (index + seed).is_multiple_of(2) {
                0.25
            } else {
                -0.25
            }
        })
        .collect::<Vec<_>>();

    if matches!(requested, RequestedOutliers::Two) {
        let first = seed.wrapping_mul(29).wrapping_add(5) % DUAL_SCALE_BLOCK_SIZE;
        let mut second = seed.wrapping_mul(43).wrapping_add(97) % DUAL_SCALE_BLOCK_SIZE;
        if second == first {
            second = (second + 1) % DUAL_SCALE_BLOCK_SIZE;
        }
        source[first] = 48.0 + (seed % 7) as f32;
        source[second] = -56.0 - (seed % 11) as f32;
    }

    source
}

fn dual_scale_hot_matrix(
    rows: usize,
    cols: usize,
    requested: RequestedOutliers,
) -> (Vec<DualScaleCompactBlock>, DualScaleHotCpuTensor) {
    assert!(rows > 0);
    assert!(cols > 0);
    assert!(cols.is_multiple_of(DUAL_SCALE_BLOCK_SIZE));

    let block_count = rows.checked_mul(cols / DUAL_SCALE_BLOCK_SIZE).unwrap();

    let compact = (0..block_count)
        .map(|block| {
            encode_dual_scale_block_ref(&dual_scale_source_block(block, requested)).unwrap()
        })
        .collect::<Vec<_>>();

    let hot = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();

    (compact, hot)
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn host_supports_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}

fn same_launch(actual: DualScaleQ8HotCpuLaunch, expected: DualScaleQ8HotCpuLaunch) -> bool {
    std::ptr::addr_eq(actual as *const (), expected as *const ())
}

fn same_prevalidated_launch(
    actual: DualScaleQ8HotCpuPrevalidatedLaunch,
    expected: DualScaleQ8HotCpuPrevalidatedLaunch,
) -> bool {
    std::ptr::addr_eq(actual as *const (), expected as *const ())
}

fn assert_close(actual: f32, expected: f32, blocks_per_row: usize, row: usize) {
    let absolute_error = (actual - expected).abs();

    // M28 remains formally NumericallyClose. This direct comparison against
    // the M26 portable production path is deliberately tighter than the full
    // dense-reference admission threshold.
    let tolerance = 1.0e-5 * blocks_per_row.max(1) as f32 + 1.0e-6 * expected.abs().max(1.0);

    assert!(
        absolute_error <= tolerance,
        "row={row} actual={actual} expected={expected} \
         absolute_error={absolute_error} tolerance={tolerance}",
    );
}

struct DualScaleFixture {
    graph: Graph,
    resources: CpuWeightResources,
    input: Vec<f32>,
    output_tensor: TensorId,

    /// Independent owner used only for the portable correctness reference.
    reference_weight: DualScaleHotCpuTensor,
}

fn fixture(
    rows: usize,
    cols: usize,
    requested: RequestedOutliers,
    salt: usize,
) -> DualScaleFixture {
    let (_, resource_weight) = dual_scale_hot_matrix(rows, cols, requested);
    let (_, reference_weight) = dual_scale_hot_matrix(rows, cols, requested);

    let input = activation(cols, salt);

    let mut resources = CpuWeightResources::new();

    let storage = resources
        .add_dual_scale_hot_cpu_portable(resource_weight)
        .unwrap();

    let mut graph = Graph::new();

    let input_tensor = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[cols])));

    let weight_tensor = graph
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[rows, cols]),
            StorageRef::HostOwned { id: storage },
            QuantFormatId::DualScale,
            256,
            1,
        ))
        .unwrap();

    let output_tensor = graph.add_qmatmul(
        input_tensor,
        weight_tensor,
        TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
    );

    graph.try_set_output(output_tensor).unwrap();

    DualScaleFixture {
        graph,
        resources,
        input,
        output_tensor,
        reference_weight,
    }
}

fn compiled_dual_scale_launch(compiled: &CpuCompiledPlan) -> DualScaleQ8HotCpuLaunch {
    let consumers = compiled
        .steps()
        .iter()
        .filter_map(|step| match step {
            CpuCompiledStep::DualScaleQ8Hot(args) => Some(args.launch),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(consumers.len(), 1);

    consumers[0]
}

fn planned_dual_scale_launch(
    plan: &ExecutionPlan<augur_backend::CpuBackend>,
) -> DualScaleQ8HotCpuPrevalidatedLaunch {
    let consumers = plan
        .steps()
        .iter()
        .filter_map(|step| match step {
            PlannedOp::DualScaleQ8Hot(args) => Some(args.launch()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(consumers.len(), 1);

    consumers[0]
}

fn planned_dual_scale_weight(
    plan: &ExecutionPlan<augur_backend::CpuBackend>,
) -> &Arc<DualScaleHotCpuTensor> {
    let consumers = plan
        .steps()
        .iter()
        .filter_map(|step| match step {
            PlannedOp::DualScaleQ8Hot(args) => Some(args.weight()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(consumers.len(), 1);

    consumers[0]
}

fn q8hot_temporary_index(plan: &ExecutionPlan<augur_backend::CpuBackend>) -> usize {
    assert_eq!(plan.memory().temporary_allocations().len(), 1);

    plan.memory().temporary_allocations()[0].buffer_index
}

fn direct_portable_reference(weight: &DualScaleHotCpuTensor, input: &[f32]) -> Vec<f32> {
    let mut q8hot = augur_kernel_api::Q8KActivationHotCpuVector::new_zeroed(input.len()).unwrap();

    launch_quantize_q8k_hot_cpu_portable(input, &mut q8hot).unwrap();

    let mut output = vec![0.0_f32; weight.rows()];

    launch_qmatvec_dual_scale_q8k_hot_cpu_portable(weight, &q8hot, &mut output).unwrap();

    output
}

// ── Tests ────────────────────────────────────────────────────────────────

#[test]
fn native_launch_identity_survives_compile_and_bind() {
    let DualScaleFixture {
        graph,
        resources,
        input,
        ..
    } = fixture(ROWS, COLS, RequestedOutliers::Two, 1201);

    let compiled = GraphCompiler::cpu_native()
        .compile_cpu_plan(&graph, vec![input])
        .unwrap();

    assert_eq!(compiled.step_count(), 2);

    assert!(matches!(
        compiled.steps()[0],
        CpuCompiledStep::QuantizeQ8Hot(_),
    ));

    assert!(matches!(
        compiled.steps()[1],
        CpuCompiledStep::DualScaleQ8Hot(_),
    ));

    let expected = if host_supports_avx2() {
        launch_qmatvec_dual_scale_q8k_hot_cpu_avx2
    } else {
        launch_qmatvec_dual_scale_q8k_hot_cpu_portable
    };

    let compiled_launch = compiled_dual_scale_launch(&compiled);

    assert!(
        same_launch(compiled_launch, expected),
        "compiled launch mismatch",
    );

    let plan = bind_cpu_plan(compiled, resources).unwrap();

    assert_eq!(plan.steps().len(), 2);

    assert!(matches!(plan.steps()[0], PlannedOp::QuantizeQ8Hot(_),));

    assert!(matches!(plan.steps()[1], PlannedOp::DualScaleQ8Hot(_),));

    let planned_launch = planned_dual_scale_launch(&plan);

    let expected_prevalidated = if host_supports_avx2() {
        launch_qmatvec_dual_scale_q8k_hot_cpu_avx2_prevalidated
            as DualScaleQ8HotCpuPrevalidatedLaunch
    } else {
        launch_qmatvec_dual_scale_q8k_hot_cpu_portable_prevalidated
            as DualScaleQ8HotCpuPrevalidatedLaunch
    };
    assert!(
        same_prevalidated_launch(planned_launch, expected_prevalidated),
        "planned launch mismatch",
    );
}

#[test]
fn baseline_plan_remains_portable() {
    let DualScaleFixture {
        graph,
        resources,
        input,
        ..
    } = fixture(ROWS, COLS, RequestedOutliers::Two, 1213);

    let compiled = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![input])
        .unwrap();

    assert!(same_launch(
        compiled_dual_scale_launch(&compiled),
        launch_qmatvec_dual_scale_q8k_hot_cpu_portable,
    ));

    let plan = bind_cpu_plan(compiled, resources).unwrap();

    assert!(same_prevalidated_launch(
        planned_dual_scale_launch(&plan),
        launch_qmatvec_dual_scale_q8k_hot_cpu_portable_prevalidated,
    ));
}

#[test]
fn native_execution_plan_matches_portable_reference() {
    let DualScaleFixture {
        graph,
        resources,
        input,
        output_tensor,
        reference_weight,
    } = fixture(ROWS, COLS, RequestedOutliers::Two, 1229);

    let reference = direct_portable_reference(&reference_weight, &input);

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
        assert_close(
            output[row],
            reference[row],
            reference_weight.blocks_per_row(),
            row,
        );
    }
}

#[test]
fn repeated_native_execution_preserves_all_owners() {
    let DualScaleFixture {
        graph,
        resources,
        input,
        output_tensor,
        ..
    } = fixture(ROWS, COLS, RequestedOutliers::Two, 1237);

    let compiled = GraphCompiler::cpu_native()
        .compile_cpu_plan(&graph, vec![input])
        .unwrap();

    let mut plan = bind_cpu_plan(compiled, resources).unwrap();

    let mut stream = CpuStream;

    plan.execute_cpu(&mut stream).unwrap();

    let output_index = plan.memory().buffer_for(output_tensor).unwrap();

    let q8hot_index = q8hot_temporary_index(&plan);

    let input_index = match &plan.steps()[0] {
        PlannedOp::QuantizeQ8Hot(args) => args.input().buffer_index(),
        other => {
            panic!("unexpected first step: {other:?}")
        }
    };

    let weight_arc_pointer = Arc::as_ptr(planned_dual_scale_weight(&plan));

    let weight_block_pointer = planned_dual_scale_weight(&plan).blocks().as_ptr();

    let input_pointer = plan.buffers()[input_index].as_f32().unwrap().as_ptr();

    let q8hot_pointer = plan.buffers()[q8hot_index]
        .as_q8hot()
        .unwrap()
        .blocks()
        .as_ptr();

    let output_pointer = plan.buffers()[output_index].as_f32().unwrap().as_ptr();

    let step_count = plan.steps().len();
    let buffer_count = plan.buffers().len();

    for _ in 0..16 {
        plan.execute_cpu(&mut stream).unwrap();

        assert_eq!(
            Arc::as_ptr(planned_dual_scale_weight(&plan)),
            weight_arc_pointer,
        );

        assert_eq!(
            planned_dual_scale_weight(&plan).blocks().as_ptr(),
            weight_block_pointer,
        );

        assert_eq!(
            plan.buffers()[input_index].as_f32().unwrap().as_ptr(),
            input_pointer,
        );

        assert_eq!(
            plan.buffers()[q8hot_index]
                .as_q8hot()
                .unwrap()
                .blocks()
                .as_ptr(),
            q8hot_pointer,
        );

        assert_eq!(
            plan.buffers()[output_index].as_f32().unwrap().as_ptr(),
            output_pointer,
        );

        assert!(same_prevalidated_launch(
            planned_dual_scale_launch(&plan),
            if host_supports_avx2() {
                launch_qmatvec_dual_scale_q8k_hot_cpu_avx2_prevalidated
            } else {
                launch_qmatvec_dual_scale_q8k_hot_cpu_portable_prevalidated
            },
        ));

        assert_eq!(plan.steps().len(), step_count);
        assert_eq!(plan.buffers().len(), buffer_count);
    }
}

#[test]
fn native_plan_resource_accounting_is_stable() {
    let rows = 4;
    let cols = 1024;

    let DualScaleFixture {
        graph,
        resources,
        input,
        ..
    } = fixture(rows, cols, RequestedOutliers::Two, 1249);

    let compiled = GraphCompiler::cpu_native()
        .compile_cpu_plan(&graph, vec![input])
        .unwrap();

    let plan = bind_cpu_plan(compiled, resources).unwrap();

    assert_eq!(plan.resource_accounting().unique_weight_count, 1);

    assert_eq!(plan.resource_accounting().executable_weight_refs, 1);

    assert!(plan.resource_accounting().hot_weight_payload_bytes > 0);
}

#[test]
fn native_plan_uses_existing_step_and_temporary_types() {
    let DualScaleFixture {
        graph,
        resources,
        input,
        ..
    } = fixture(ROWS, COLS, RequestedOutliers::Two, 1259);

    let compiled = GraphCompiler::cpu_native()
        .compile_cpu_plan(&graph, vec![input])
        .unwrap();

    assert_eq!(
        compiled
            .steps()
            .iter()
            .filter(|step| { matches!(step, CpuCompiledStep::QuantizeQ8Hot(_)) })
            .count(),
        1,
    );

    assert_eq!(
        compiled
            .steps()
            .iter()
            .filter(|step| { matches!(step, CpuCompiledStep::DualScaleQ8Hot(_)) })
            .count(),
        1,
    );

    assert_eq!(compiled.memory().temporary_allocations().len(), 1);

    let temporary = compiled.memory().temporary_allocations()[0].temporary;

    assert_eq!(temporary, TemporaryBufferId::new(0));

    let plan = bind_cpu_plan(compiled, resources).unwrap();

    assert!(matches!(plan.steps()[0], PlannedOp::QuantizeQ8Hot(_),));

    assert!(matches!(plan.steps()[1], PlannedOp::DualScaleQ8Hot(_),));

    assert_eq!(plan.memory().temporary_allocations().len(), 1);
}

#[test]
fn native_two_consumers_share_one_q8hot_preparation() {
    let rows = 16;
    let cols = 1024;

    let (_, weight_a) = dual_scale_hot_matrix(rows, cols, RequestedOutliers::Two);
    let (_, weight_b) = dual_scale_hot_matrix(rows, cols, RequestedOutliers::Two);

    let input = activation(cols, 1277);

    let mut resources = CpuWeightResources::new();

    let storage_a = resources.add_dual_scale_hot_cpu_portable(weight_a).unwrap();

    let storage_b = resources.add_dual_scale_hot_cpu_portable(weight_b).unwrap();

    let mut graph = Graph::new();

    let input_tensor = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[cols])));

    let weight_tensor_a = graph
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[rows, cols]),
            StorageRef::HostOwned { id: storage_a },
            QuantFormatId::DualScale,
            256,
            1,
        ))
        .unwrap();

    let weight_tensor_b = graph
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[rows, cols]),
            StorageRef::HostOwned { id: storage_b },
            QuantFormatId::DualScale,
            256,
            1,
        ))
        .unwrap();

    let output_a = graph.add_qmatmul(
        input_tensor,
        weight_tensor_a,
        TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
    );

    let output_b = graph.add_qmatmul(
        input_tensor,
        weight_tensor_b,
        TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
    );

    graph.try_set_output(output_a).unwrap();
    graph.try_set_output(output_b).unwrap();

    let compiled = GraphCompiler::cpu_native()
        .compile_cpu_plan(&graph, vec![input])
        .unwrap();

    let quantize_steps = compiled
        .steps()
        .iter()
        .filter(|step| matches!(step, CpuCompiledStep::QuantizeQ8Hot(_)))
        .count();

    let consumers = compiled
        .steps()
        .iter()
        .filter_map(|step| match step {
            CpuCompiledStep::DualScaleQ8Hot(args) => {
                Some((args.activation.buffer_index(), args.launch))
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(quantize_steps, 1);
    assert_eq!(consumers.len(), 2);
    assert_eq!(consumers[0].0, consumers[1].0);
    assert_eq!(compiled.memory().temporary_allocations().len(), 1);

    let expected = if host_supports_avx2() {
        launch_qmatvec_dual_scale_q8k_hot_cpu_avx2
    } else {
        launch_qmatvec_dual_scale_q8k_hot_cpu_portable
    };

    assert!(same_launch(consumers[0].1, expected));
    assert!(same_launch(consumers[1].1, expected));

    let mut plan = bind_cpu_plan(compiled, resources).unwrap();

    let planned_consumers = plan
        .steps()
        .iter()
        .filter_map(|step| match step {
            PlannedOp::DualScaleQ8Hot(args) => Some((
                args.activation().buffer_index(),
                args.launch(),
                Arc::as_ptr(args.weight()),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(planned_consumers.len(), 2);
    assert_eq!(planned_consumers[0].0, planned_consumers[1].0);

    let expected_prevalidated = if host_supports_avx2() {
        launch_qmatvec_dual_scale_q8k_hot_cpu_avx2_prevalidated
            as DualScaleQ8HotCpuPrevalidatedLaunch
    } else {
        launch_qmatvec_dual_scale_q8k_hot_cpu_portable_prevalidated
            as DualScaleQ8HotCpuPrevalidatedLaunch
    };
    assert!(same_prevalidated_launch(
        planned_consumers[0].1,
        expected_prevalidated,
    ));

    assert!(same_prevalidated_launch(
        planned_consumers[1].1,
        expected_prevalidated,
    ));

    assert_ne!(planned_consumers[0].2, planned_consumers[1].2);

    assert_eq!(plan.memory().temporary_allocations().len(), 1);

    let q8hot_index = plan.memory().temporary_allocations()[0].buffer_index;

    assert_eq!(planned_consumers[0].0, q8hot_index);

    let mut stream = CpuStream;

    plan.execute_cpu(&mut stream).unwrap();

    let output_a_index = plan.memory().buffer_for(output_a).unwrap();

    let output_b_index = plan.memory().buffer_for(output_b).unwrap();

    assert_eq!(plan.buffers()[output_a_index].as_f32().unwrap().len(), rows);

    assert_eq!(plan.buffers()[output_b_index].as_f32().unwrap().len(), rows);
}
