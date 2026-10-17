#![cfg(feature = "cuda")]

use augur_compiler::{
    bind_cuda_plan, preflight_cuda_plan, CudaBindError, CudaCompiledStep, CudaGraphCompiler,
    CudaWeightResources,
};
use augur_core::{F16Bits, FormatVersion, QuantFormatId, Shape, StorageId, StorageRef, TensorDesc};
use augur_formats::{pack_q4k_scales_ref, Q4KCompactBlock, Q4KScales, Q4K_QS_BYTES};
use augur_graph::Graph;
use augur_kernel_api::{Q4KHotCpuBlock, Q4KHotCpuTensor};
use augur_reference::q4k_f32_gemm_reference;
use augur_repack::{repack_q4k_block_cpu_ref, repack_q4k_hot_cpu_to_cuda_v1};

const M: usize = 4;
const WIDTH: usize = 256;
const TOLERANCE: f32 = 1.0e-4;

fn compact_block(seed: usize, index: usize) -> Q4KCompactBlock {
    let mut qs = [0_u8; Q4K_QS_BYTES];
    for (byte, value) in qs.iter_mut().enumerate() {
        let low = seed.wrapping_add(index).wrapping_add(byte * 3) & 0x0f;
        let high = seed
            .wrapping_mul(5)
            .wrapping_add(index)
            .wrapping_add(byte * 7)
            & 0x0f;
        *value = low as u8 | ((high as u8) << 4);
    }
    Q4KCompactBlock {
        d: F16Bits::from_f32(1.0 / 1024.0),
        dmin: F16Bits::from_f32(0.0),
        scales: pack_q4k_scales_ref(Q4KScales {
            scale: [1 + (seed & 1) as u8; 8],
            min: [0; 8],
        })
        .unwrap(),
        qs,
    }
}

fn compact_weight(seed: usize) -> Vec<Q4KCompactBlock> {
    (0..WIDTH).map(|index| compact_block(seed, index)).collect()
}

fn staging(seed: usize) -> augur_kernel_api::Q4KHotCudaStagingTensorV1 {
    let compact = compact_weight(seed);
    let hot = Q4KHotCpuTensor::new(
        WIDTH,
        WIDTH,
        compact.iter().map(repack_q4k_block_cpu_ref).collect(),
    )
    .unwrap();
    repack_q4k_hot_cpu_to_cuda_v1(&hot).unwrap()
}

fn graph() -> Graph {
    let mut graph = Graph::new();
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[M, WIDTH]));
    let input = graph.add_input(desc.clone());
    let weights: Vec<_> = (0..3)
        .map(|index| {
            graph
                .add_weight(TensorDesc::quantized_weight(
                    Shape::from_dims(&[WIDTH, WIDTH]),
                    StorageRef::HostOwned {
                        id: StorageId::new(index, 0),
                    },
                    QuantFormatId::Q4K,
                    Q4KHotCpuBlock::DECODED_LEN,
                    1,
                ))
                .unwrap()
        })
        .collect();
    let h0 = graph.add_qmatmul(input, weights[0], desc.clone());
    let h1 = graph.add_add(h0, input, desc.clone());
    let h2 = graph.add_qmatmul(h1, weights[1], desc.clone());
    let h3 = graph.add_add(h2, input, desc.clone());
    let output = graph.add_qmatmul(h3, weights[2], desc);
    graph.try_set_output(output).unwrap();
    graph
}

fn input(iteration: usize) -> Vec<f32> {
    (0..M * WIDTH)
        .map(|index| {
            let value = index
                .wrapping_mul(31)
                .wrapping_add(iteration.wrapping_mul(17))
                % 127;
            (value as f32 - 63.0) / 67.0
        })
        .collect()
}

fn reference(weights: &[Vec<Q4KCompactBlock>], input: &[f32]) -> Vec<f32> {
    let mut h0 = vec![0.0; M * WIDTH];
    q4k_f32_gemm_reference(&weights[0], M, WIDTH, WIDTH, input, &mut h0).unwrap();
    let h1: Vec<_> = h0.iter().zip(input).map(|(a, b)| a + b).collect();
    let mut h2 = vec![0.0; M * WIDTH];
    q4k_f32_gemm_reference(&weights[1], M, WIDTH, WIDTH, &h1, &mut h2).unwrap();
    let h3: Vec<_> = h2.iter().zip(input).map(|(a, b)| a + b).collect();
    let mut output = vec![0.0; M * WIDTH];
    q4k_f32_gemm_reference(&weights[2], M, WIDTH, WIDTH, &h3, &mut output).unwrap();
    output
}

fn resources() -> CudaWeightResources {
    let mut resources = CudaWeightResources::new();
    for seed in 0..3 {
        resources
            .add_q4k_cuda_v1(StorageId::new(seed, 0), FormatVersion(1), staging(seed))
            .unwrap();
    }
    resources
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let limit = TOLERANCE * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= limit,
            "index={index} actual={actual} expected={expected} limit={limit}"
        );
    }
}

#[test]
fn m35_multi_op_plan_reuses_intermediates_and_matches_reference() {
    let graph = graph();
    let initial = input(0);
    let compiled = CudaGraphCompiler::new(0)
        .unwrap()
        .compile_plan(&graph, initial.clone())
        .unwrap();

    assert_eq!(compiled.steps().len(), 5);
    assert!(matches!(
        compiled.steps()[0],
        CudaCompiledStep::Q4kF32Prefill(_)
    ));
    assert!(matches!(compiled.steps()[1], CudaCompiledStep::AddF32(_)));
    assert!(matches!(
        compiled.steps()[2],
        CudaCompiledStep::Q4kF32Prefill(_)
    ));
    assert!(matches!(compiled.steps()[3], CudaCompiledStep::AddF32(_)));
    assert!(matches!(
        compiled.steps()[4],
        CudaCompiledStep::Q4kF32Prefill(_)
    ));
    assert_eq!(compiled.weight_requirements().len(), 3);
    assert_eq!(compiled.memory().buffers().len(), 4);
    assert_eq!(compiled.memory().reuse_transitions().len(), 2);
    for reuse in compiled.memory().reuse_transitions() {
        let previous = match reuse.previous_reusable_after {
            augur_exec::BufferLifetimeEnd::GraphTopoStep(step) => step,
            other => panic!("unexpected previous lifetime: {other:?}"),
        };
        let next = match reuse.next_first_use {
            augur_exec::BufferLifetimeEnd::GraphTopoStep(step) => step,
            other => panic!("unexpected next lifetime: {other:?}"),
        };
        assert!(previous < next);
    }

    let preflight = preflight_cuda_plan(&compiled, &resources()).unwrap();
    let accounting = preflight.accounting;
    assert_eq!(accounting.device_allocation_count, 7);
    assert_eq!(accounting.weight_allocation_count, 3);
    assert_eq!(accounting.activation_allocation_count, 4);
    assert_eq!(accounting.event_count, 2);
    assert_eq!(accounting.executable_step_count, 5);
    assert_eq!(accounting.reuse_transition_count, 2);
    assert_eq!(
        accounting.logical_intermediate_payload_bytes,
        4 * M * WIDTH * 4
    );
    assert_eq!(
        accounting.physical_intermediate_payload_bytes,
        2 * M * WIDTH * 4
    );

    let plan = bind_cuda_plan(compiled, resources()).unwrap();
    assert_eq!(plan.step_count(), 5);
    assert_eq!(plan.buffer_count(), 7);
    let weights: Vec<_> = (0..3).map(compact_weight).collect();

    for iteration in 0..8 {
        let input = input(iteration);
        let expected = reference(&weights, &input);
        plan.set_input_f32(&input).unwrap();
        if iteration == 0 {
            let timing = plan.execute_cuda_profiled().unwrap();
            assert!(timing.kernel_event_ms > 0.0);
            assert!(timing.complete_host >= timing.launch_host);
        } else {
            plan.execute_cuda().unwrap();
        }
        let mut actual = vec![0.0; expected.len()];
        plan.read_output_f32(&mut actual).unwrap();
        assert_close(&actual, &expected);
        assert_eq!(plan.accounting(), accounting);
    }
}

#[test]
fn m35_global_preflight_failure_preserves_all_resources() {
    let compiled = CudaGraphCompiler::new(0)
        .unwrap()
        .compile_plan(&graph(), input(0))
        .unwrap();
    let mut resources = CudaWeightResources::new();
    resources
        .add_q4k_cuda_v1(StorageId::new(0, 0), FormatVersion(1), staging(0))
        .unwrap();
    resources
        .add_q4k_cuda_v1(StorageId::new(1, 0), FormatVersion(1), staging(1))
        .unwrap();
    resources
        .add_q4k_cuda_v1(StorageId::new(2, 1), FormatVersion(1), staging(2))
        .unwrap();

    assert_eq!(resources.len(), 3);
    assert_eq!(
        preflight_cuda_plan(&compiled, &resources).unwrap_err(),
        CudaBindError::GenerationMismatch {
            requested: StorageId::new(2, 0),
            actual: StorageId::new(2, 1),
        }
    );
    assert_eq!(resources.len(), 3);
    for storage in [
        StorageId::new(0, 0),
        StorageId::new(1, 0),
        StorageId::new(2, 1),
    ] {
        assert!(resources.contains_exact(storage));
    }
}
