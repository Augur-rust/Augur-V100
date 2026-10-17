use std::sync::Arc;

use augur_backend::CpuStream;
use augur_compiler::{bind_cpu_plan, CpuCompiledStep, CpuWeightResources, GraphCompiler};
use augur_core::{CompileError, QuantFormatId, Shape, StorageRef, TensorDesc};
use augur_exec::PlannedOp;
use augur_formats::Mxfp4CompactBlock;
use augur_graph::Graph;
use augur_kernel_api::Mxfp4HotCpuBlock;
use augur_kernels::portable::launch_qmatvec_mxfp4_f32_cpu_portable;
use augur_reference::qmatvec_mxfp4_f32_ref;
use augur_repack::repack_mxfp4_tensor_cpu;

const ROWS: usize = 8;
const COLS: usize = 256;

fn compact() -> Vec<Mxfp4CompactBlock> {
    (0..ROWS * (COLS / 32))
        .map(|block| Mxfp4CompactBlock {
            qs: core::array::from_fn(|lane| {
                let low = ((block * 3 + lane * 5) & 0x0f) as u8;
                let high = (15 - ((block * 7 + lane * 11) & 0x0f)) as u8;
                low | (high << 4)
            }),
            scale_e8m0: 121 + (block % 8) as u8,
        })
        .collect()
}

fn activation() -> Vec<f32> {
    (0..COLS)
        .map(|index| (((index * 17) % 97) as f32 - 48.0) / 32.0)
        .collect()
}

fn graph(storage: augur_core::StorageId, input_shape: &[usize], block: usize, version: u16) -> Graph {
    let mut graph = Graph::new();
    let input = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(input_shape)));
    let weight = graph
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[ROWS, COLS]),
            StorageRef::HostOwned { id: storage },
            QuantFormatId::MXFP4,
            block,
            version,
        ))
        .unwrap();
    let output_shape = if input_shape.len() == 1 {
        Shape::from_dims(&[ROWS])
    } else {
        Shape::from_dims(&[input_shape[0], ROWS])
    };
    let output = graph.add_qmatmul(input, weight, TensorDesc::f32_contiguous(output_shape));
    graph.try_set_output(output).unwrap();
    graph
}

#[test]
fn graph_compile_bind_execute_matches_independent_compact_reference() {
    let compact = compact();
    let hot = repack_mxfp4_tensor_cpu(&compact, ROWS, COLS).unwrap();
    let mut resources = CpuWeightResources::new();
    let storage = resources.add_mxfp4_hot_cpu_portable(hot).unwrap();
    let graph = graph(storage, &[COLS], 32, 1);
    let input = activation();

    let compiled = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![input.clone()])
        .unwrap();
    assert_eq!(compiled.step_count(), 1);
    let compiled_launch = match &compiled.steps()[0] {
        CpuCompiledStep::Mxfp4F32(args) => args.launch,
        other => panic!("unexpected compiled step: {other:?}"),
    };
    assert!(std::ptr::fn_addr_eq(
        compiled_launch,
        launch_qmatvec_mxfp4_f32_cpu_portable as augur_kernel_api::Mxfp4F32CpuLaunch
    ));
    assert_eq!(compiled.weight_requirements().len(), 1);

    let mut plan = bind_cpu_plan(compiled, resources).unwrap();
    assert_eq!(plan.resource_accounting().unique_weight_count, 1);
    assert_eq!(plan.resource_accounting().executable_weight_refs, 1);
    assert_eq!(
        plan.resource_accounting().hot_weight_payload_bytes,
        compact.len() * core::mem::size_of::<Mxfp4HotCpuBlock>()
    );
    let (weight, planned_launch, output_buffer) = match &plan.steps()[0] {
        PlannedOp::Mxfp4F32(args) => (
            Arc::clone(args.weight()),
            args.launch(),
            args.output().buffer_index(),
        ),
        other => panic!("unexpected planned step: {other:?}"),
    };
    assert!(std::ptr::fn_addr_eq(compiled_launch, planned_launch));

    let mut expected = vec![0.0_f32; ROWS];
    qmatvec_mxfp4_f32_ref(&compact, ROWS, COLS, &input, &mut expected).unwrap();
    let owner_ptr = Arc::as_ptr(&weight);
    let payload_ptr = weight.blocks().as_ptr();
    let mut stream = CpuStream;
    for _ in 0..8 {
        plan.execute_cpu(&mut stream).unwrap();
        assert_eq!(plan.output_f32(output_buffer).unwrap(), expected);
        assert_eq!(Arc::as_ptr(&weight), owner_ptr);
        assert_eq!(weight.blocks().as_ptr(), payload_ptr);
    }
    let telemetry = plan.execute_cpu_profiled(&mut stream).unwrap();
    assert_eq!(telemetry.step_count, 1);
    assert!(telemetry.mxfp4_f32_ns > 0);
    assert_eq!(telemetry.total_ns(), telemetry.mxfp4_f32_ns);
}

#[test]
fn mxfp4_production_slice_rejects_prefill_wrong_block_and_wrong_version() {
    let storage = augur_core::StorageId::new(0, 0);
    let cases = [
        (vec![2, COLS], 32, 1, vec![0.0; 2 * COLS]),
        (vec![COLS], 16, 1, vec![0.0; COLS]),
        (vec![COLS], 32, 2, vec![0.0; COLS]),
    ];
    for (shape, block, version, input) in cases {
        let graph = graph(storage, &shape, block, version);
        let error = GraphCompiler::cpu_baseline()
            .compile_cpu_plan(&graph, vec![input])
            .unwrap_err();
        assert_eq!(
            error,
            augur_compiler::CompilerError::Lowering(augur_compiler::LoweringError::Core(
                CompileError::BackendUnsupported,
            ))
        );
    }
}
