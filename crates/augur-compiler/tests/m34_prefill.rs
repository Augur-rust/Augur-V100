use augur_backend::CpuStream;
use augur_compiler::{bind_cpu_plan, CpuCompiledStep, CpuWeightResources, GraphCompiler};
use augur_core::{F16Bits, QuantFormatId, Shape, StorageRef, TensorDesc};
use augur_formats::{pack_q4k_scales_ref, Q4KCompactBlock, Q4KScales};
use augur_graph::Graph;
use augur_reference::q4k_f32_gemm_reference;
use augur_repack::repack_q4k_block_cpu_ref;

fn compact_fixture(n: usize, k: usize) -> Vec<Q4KCompactBlock> {
    (0..n * (k / 256))
        .map(|block_index| Q4KCompactBlock {
            d: F16Bits::from_f32(0.0625),
            dmin: F16Bits::from_f32(0.0078125),
            scales: pack_q4k_scales_ref(Q4KScales {
                scale: std::array::from_fn(|g| 1 + ((block_index + g) % 9) as u8),
                min: std::array::from_fn(|g| ((block_index + g * 2) % 4) as u8),
            })
            .unwrap(),
            qs: std::array::from_fn(|i| {
                ((i + block_index) % 16) as u8 | ((((i * 5 + block_index) % 16) as u8) << 4)
            }),
        })
        .collect()
}

fn execute_case(m: usize, n: usize, k: usize) {
    let compact = compact_fixture(n, k);
    let hot = augur_kernel_api::Q4KHotCpuTensor::new(
        n,
        k,
        compact.iter().map(repack_q4k_block_cpu_ref).collect(),
    )
    .unwrap();
    let mut resources = CpuWeightResources::new();
    let storage = resources.add_q4k_hot_cpu_portable(hot).unwrap();
    let input_values: Vec<f32> = (0..m * k)
        .map(|i| ((i * 13 % 89) as f32 - 44.0) / 29.0)
        .collect();

    let mut graph = Graph::new();
    let input = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[m, k])));
    let weight = graph
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[n, k]),
            StorageRef::HostOwned { id: storage },
            QuantFormatId::Q4K,
            256,
            1,
        ))
        .unwrap();
    let output = graph.add_qmatmul(
        input,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[m, n])),
    );
    graph.try_set_output(output).unwrap();

    let compiled = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![input_values.clone()])
        .unwrap();
    assert!(matches!(
        &compiled.steps()[0],
        CpuCompiledStep::QMatmulF32Prefill(step) if step.m == m && step.rows == n && step.cols == k
    ));
    let output_index = compiled.memory().buffer_for(output).unwrap();
    let mut plan = bind_cpu_plan(compiled, resources).unwrap();
    plan.execute_cpu(&mut CpuStream).unwrap();

    let mut expected = vec![0.0; m * n];
    q4k_f32_gemm_reference(&compact, m, n, k, &input_values, &mut expected).unwrap();
    assert_eq!(plan.output_f32(output_index).unwrap(), expected);
}

#[test]
fn compile_bind_execute_covers_m1_small_batch_and_model_shape() {
    execute_case(1, 5, 256);
    execute_case(4, 17, 512);
    execute_case(16, 256, 512);
}
