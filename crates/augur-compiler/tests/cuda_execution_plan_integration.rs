#![cfg(feature = "cuda")]

use augur_compiler::{
    bind_cuda_plan, CudaBindError, CudaCompiledQ4kF32Prefill, CudaCompiledStep, CudaGraphCompiler,
    CudaWeightBindingId, CudaWeightResources,
};
use augur_core::{F16Bits, FormatVersion, QuantFormatId, Shape, StorageId, StorageRef, TensorDesc};
use augur_exec::CudaPlanResourceAccounting;
use augur_formats::{Q4KCompactBlock, Q4K_QS_BYTES, Q4K_SCALE_BYTES};
use augur_graph::Graph;
use augur_kernel_api::{
    Q4KHotCpuBlock, Q4KHotCpuTensor, Q4KHotCudaStagingTensorV1, Q4K_HOT_CUDA_V1_BLOCK_BYTES,
};
use augur_kernels::portable::qmatmul_q4k_f32_portable;
use augur_reference::q4k_f32_gemm_reference;
use augur_repack::{repack_q4k_block_cpu_ref, repack_q4k_hot_cpu_to_cuda_v1};

const TOLERANCE_ABS: f32 = 1.0e-4;

struct Q4kCudaFixture {
    graph: Graph,
    storage: StorageId,
    hot: Q4KHotCpuTensor,
    staging: Q4KHotCudaStagingTensorV1,
    input: Vec<f32>,
}

fn compact_block(index: usize) -> Q4KCompactBlock {
    let mut scales = [0_u8; Q4K_SCALE_BYTES];
    let mut qs = [0_u8; Q4K_QS_BYTES];

    for (byte_index, value) in scales.iter_mut().enumerate() {
        *value = index.wrapping_mul(7).wrapping_add(byte_index) as u8;
    }

    for (byte_index, value) in qs.iter_mut().enumerate() {
        let low = index.wrapping_add(byte_index.wrapping_mul(3)) % 15;
        let high = index
            .wrapping_add(byte_index.wrapping_mul(5))
            .wrapping_add(7)
            % 15;
        *value = u8::try_from(low).unwrap() | (u8::try_from(high).unwrap() << 4);
    }

    Q4KCompactBlock {
        d: F16Bits::from_f32(0.5 + (index % 4) as f32 * 0.125),
        dmin: F16Bits::from_f32(0.125 + (index % 3) as f32 * 0.0625),
        scales,
        qs,
    }
}

fn fixture(rows: usize, cols: usize, storage: StorageId) -> Q4kCudaFixture {
    assert!(rows > 0);
    assert!(cols > 0);
    assert!(cols.is_multiple_of(Q4KHotCpuBlock::DECODED_LEN));

    let blocks_per_row = cols.checked_div(Q4KHotCpuBlock::DECODED_LEN).unwrap();
    let block_count = rows.checked_mul(blocks_per_row).unwrap();

    let hot_blocks = (0..block_count)
        .map(compact_block)
        .map(|compact| repack_q4k_block_cpu_ref(&compact))
        .collect::<Vec<_>>();

    let hot = Q4KHotCpuTensor::new(rows, cols, hot_blocks).unwrap();
    let staging = repack_q4k_hot_cpu_to_cuda_v1(&hot).unwrap();

    let input = (0..cols)
        .map(|index| 0.5 - (index as f32) / ((cols as f32) * 2.0))
        .collect::<Vec<_>>();

    let mut graph = Graph::new();

    let activation = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[cols])));

    let weight = graph
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[rows, cols]),
            StorageRef::HostOwned { id: storage },
            QuantFormatId::Q4K,
            Q4KHotCpuBlock::DECODED_LEN,
            1,
        ))
        .unwrap();

    let output = graph.add_qmatmul(
        activation,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
    );

    graph.try_set_output(output).unwrap();

    Q4kCudaFixture {
        graph,
        storage,
        hot,
        staging,
        input,
    }
}

fn cpu_reference(hot: &Q4KHotCpuTensor, input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0_f32; hot.rows()];
    qmatmul_q4k_f32_portable(hot, input, &mut output).unwrap();
    output
}

fn assert_numerically_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());

    for (row, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let absolute_error = (actual - expected).abs();
        let scale = actual.abs().max(expected.abs()).max(1.0);
        let tolerance = TOLERANCE_ABS * scale;
        assert!(
            absolute_error <= tolerance,
            "row={row}: actual={actual}, expected={expected}, \
             abs={absolute_error}, tolerance={tolerance}",
        );
    }
}

fn execute_planned_cuda(
    fixture: Q4kCudaFixture,
) -> (Vec<f32>, Vec<f32>, CudaPlanResourceAccounting) {
    let expected = cpu_reference(&fixture.hot, &fixture.input);

    let compiler = CudaGraphCompiler::new(0).unwrap();

    let compiled = compiler
        .compile_plan(&fixture.graph, fixture.input.clone())
        .unwrap();

    assert_eq!(compiled.device_ordinal(), 0);
    assert_eq!(compiled.steps().len(), 1);
    assert_eq!(compiled.weight_requirements().len(), 1);

    let requirement = compiled.weight_requirements()[0];

    assert_eq!(requirement.storage, fixture.storage);
    assert_eq!(requirement.rows, fixture.hot.rows());
    assert_eq!(requirement.cols, fixture.hot.cols());
    assert_eq!(
        requirement.payload_bytes,
        fixture.staging.payload_len_bytes(),
    );

    let mut resources = CudaWeightResources::new();

    resources
        .add_q4k_cuda_v1(fixture.storage, FormatVersion(1), fixture.staging)
        .unwrap();

    let plan = bind_cuda_plan(compiled, resources).unwrap();

    assert_eq!(plan.step_count(), 1);
    assert_eq!(plan.buffer_count(), 3);

    plan.execute_cuda().unwrap();

    let mut actual = vec![0.0_f32; expected.len()];
    plan.read_output_f32(&mut actual).unwrap();

    (actual, expected, plan.accounting())
}

fn expected_accounting(rows: usize, cols: usize) -> CudaPlanResourceAccounting {
    let blocks_per_row = cols.checked_div(Q4KHotCpuBlock::DECODED_LEN).unwrap();

    let weight_payload_bytes = rows
        .checked_mul(blocks_per_row)
        .and_then(|count| count.checked_mul(Q4K_HOT_CUDA_V1_BLOCK_BYTES))
        .unwrap();

    let input_payload_bytes = cols.checked_mul(core::mem::size_of::<f32>()).unwrap();

    let output_payload_bytes = rows.checked_mul(core::mem::size_of::<f32>()).unwrap();

    let device_payload_bytes = weight_payload_bytes
        .checked_add(input_payload_bytes)
        .and_then(|bytes| bytes.checked_add(output_payload_bytes))
        .unwrap();

    CudaPlanResourceAccounting {
        device_allocation_count: 3,
        device_payload_bytes,
        weight_payload_bytes,
        input_payload_bytes,
        output_payload_bytes,
        executable_step_count: 1,
        weight_allocation_count: 1,
        activation_allocation_count: 2,
        event_count: 2,
        logical_intermediate_payload_bytes: 0,
        physical_intermediate_payload_bytes: 0,
        reuse_transition_count: 0,
    }
}

#[test]
fn planned_cuda_q4k_one_by_256_matches_cpu_reference() {
    let fixture = fixture(1, 256, StorageId::new(0, 0));

    let (actual, expected, accounting) = execute_planned_cuda(fixture);

    assert_numerically_close(&actual, &expected);

    assert_eq!(accounting, expected_accounting(1, 256));
}

#[test]
fn planned_cuda_q4k_two_by_512_matches_cpu_reference() {
    let fixture = fixture(2, 512, StorageId::new(0, 0));

    let (actual, expected, accounting) = execute_planned_cuda(fixture);

    assert_numerically_close(&actual, &expected);

    assert_eq!(accounting, expected_accounting(2, 512));
}

#[test]
fn planned_cuda_rejects_stale_weight_generation() {
    let graph_storage = StorageId::new(0, 0);
    let resource_storage = StorageId::new(0, 1);

    let fixture = fixture(1, 256, graph_storage);

    let compiler = CudaGraphCompiler::new(0).unwrap();

    let compiled = compiler
        .compile_plan(&fixture.graph, fixture.input)
        .unwrap();

    assert_eq!(compiled.weight_requirements()[0].storage, graph_storage,);

    let mut resources = CudaWeightResources::new();

    resources
        .add_q4k_cuda_v1(resource_storage, FormatVersion(1), fixture.staging)
        .unwrap();

    let error = bind_cuda_plan(compiled, resources).unwrap_err();

    assert_eq!(
        error,
        CudaBindError::GenerationMismatch {
            requested: graph_storage,
            actual: resource_storage,
        },
    );
}

fn deterministic_repeated_input(len: usize, iteration: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let value = index
                .wrapping_mul(37)
                .wrapping_add(iteration.wrapping_mul(17))
                % 127;
            (value as f32 - 63.0) / 64.0
        })
        .collect()
}

#[test]
fn planned_cuda_q4k_repeated_execution_matches_cpu_reference() {
    let fixture = fixture(2, 512, StorageId::new(0, 0));

    let Q4kCudaFixture {
        graph,
        storage,
        hot,
        staging,
        input,
    } = fixture;

    let compiler = CudaGraphCompiler::new(0).unwrap();
    let compiled = compiler.compile_plan(&graph, input).unwrap();

    let mut resources = CudaWeightResources::new();

    resources
        .add_q4k_cuda_v1(storage, FormatVersion(1), staging)
        .unwrap();

    let plan = bind_cuda_plan(compiled, resources).unwrap();

    let original_accounting = plan.accounting();
    let original_host_arena = plan.host_arena_identity().unwrap();
    let host_accounting = plan.host_arena_accounting();

    assert_eq!(plan.step_count(), 1);
    assert_eq!(plan.buffer_count(), 3);
    assert_eq!(host_accounting.allocation_count, 1);
    assert_eq!(host_accounting.buffer_count, 2);
    assert_eq!(host_accounting.payload_bytes, original_host_arena.len_bytes);
    assert_eq!(host_accounting.max_alignment_bytes, 64);

    for iteration in 0..8 {
        let input = deterministic_repeated_input(hot.cols(), iteration);

        let expected = cpu_reference(&hot, &input);

        plan.set_input_f32(&input).unwrap();
        plan.execute_cuda().unwrap();

        let mut actual = vec![0.0_f32; hot.rows()];
        plan.read_output_f32(&mut actual).unwrap();

        assert_numerically_close(&actual, &expected);

        assert_eq!(plan.accounting(), original_accounting);
        assert_eq!(plan.host_arena_identity().unwrap(), original_host_arena);
        assert_eq!(plan.step_count(), 1);
        assert_eq!(plan.buffer_count(), 3);

        // The first input is submitted and executed twice on the same plan.
        // This proves deterministic repeated execution without rebinding.
        if iteration == 0 {
            plan.set_input_f32(&input).unwrap();
            plan.execute_cuda().unwrap();

            let mut repeated = vec![0.0_f32; hot.rows()];
            plan.read_output_f32(&mut repeated).unwrap();

            assert_eq!(repeated, actual);
        }

        plan.reset_host_arena().unwrap();
        assert_eq!(plan.host_arena_identity().unwrap(), original_host_arena);
    }
}

#[test]
fn planned_cuda_prefill_m1_small_batch_and_model_shape_match_compact_oracle() {
    for (m, rows, cols) in [(1, 5, 256), (4, 17, 512), (17, 256, 512)] {
        let storage = StorageId::new(0, 0);
        let blocks_per_row = cols / Q4KHotCpuBlock::DECODED_LEN;
        let compact: Vec<_> = (0..rows * blocks_per_row).map(compact_block).collect();
        let hot = Q4KHotCpuTensor::new(
            rows,
            cols,
            compact.iter().map(repack_q4k_block_cpu_ref).collect(),
        )
        .unwrap();
        let staging = repack_q4k_hot_cpu_to_cuda_v1(&hot).unwrap();
        let input: Vec<f32> = (0..m * cols)
            .map(|i| ((i * 19 % 113) as f32 - 56.0) / 61.0)
            .collect();

        let mut graph = Graph::new();
        let activation = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[m, cols])));
        let weight = graph
            .add_weight(TensorDesc::quantized_weight(
                Shape::from_dims(&[rows, cols]),
                StorageRef::HostOwned { id: storage },
                QuantFormatId::Q4K,
                Q4KHotCpuBlock::DECODED_LEN,
                1,
            ))
            .unwrap();
        let output = graph.add_qmatmul(
            activation,
            weight,
            TensorDesc::f32_contiguous(Shape::from_dims(&[m, rows])),
        );
        graph.try_set_output(output).unwrap();

        let compiled = CudaGraphCompiler::new(0)
            .unwrap()
            .compile_plan(&graph, input.clone())
            .unwrap();
        assert_eq!(
            compiled.steps()[0],
            CudaCompiledStep::Q4kF32Prefill(CudaCompiledQ4kF32Prefill {
                weight: CudaWeightBindingId::new(0),
                input: augur_compiler::CudaCompiledF32View {
                    buffer_index: 0,
                    len_elements: m * cols,
                },
                output: augur_compiler::CudaCompiledF32View {
                    buffer_index: 1,
                    len_elements: m * rows,
                },
                m,
                rows,
                cols,
            })
        );
        let mut resources = CudaWeightResources::new();
        resources
            .add_q4k_cuda_v1(storage, FormatVersion(1), staging)
            .unwrap();
        let plan = bind_cuda_plan(compiled, resources).unwrap();
        plan.execute_cuda().unwrap();
        let mut actual = vec![0.0; m * rows];
        plan.read_output_f32(&mut actual).unwrap();

        let mut expected = vec![0.0; m * rows];
        q4k_f32_gemm_reference(&compact, m, rows, cols, &input, &mut expected).unwrap();
        assert_numerically_close(&actual, &expected);
        assert_eq!(
            plan.accounting(),
            expected_accounting_prefill(m, rows, cols)
        );
    }
}

fn expected_accounting_prefill(m: usize, rows: usize, cols: usize) -> CudaPlanResourceAccounting {
    let mut accounting = expected_accounting(rows, cols);
    accounting.input_payload_bytes = m * cols * core::mem::size_of::<f32>();
    accounting.output_payload_bytes = m * rows * core::mem::size_of::<f32>();
    accounting.device_payload_bytes = accounting.weight_payload_bytes
        + accounting.input_payload_bytes
        + accounting.output_payload_bytes;
    accounting
}
