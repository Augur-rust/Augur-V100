#![cfg(target_arch = "x86_64")]

use augur_compiler::{CpuCompiledStep, CpuWeightResources, GraphCompiler};
use augur_core::{F16Bits, QuantFormatId, Shape, StorageRef, TensorDesc};
use augur_graph::Graph;
use augur_kernel_api::{Q4KHotCpuBlock, Q4KHotCpuTensor};
use augur_kernels::{
    portable::launch_qmatmul_q4k_f32_portable, x86::avx2::launch_qmatmul_q4k_f32_avx2,
};

fn q4k_hot_tensor(rows: usize, cols: usize) -> Q4KHotCpuTensor {
    let blocks_per_row = cols / Q4KHotCpuBlock::DECODED_LEN;
    let block = Q4KHotCpuBlock {
        d: F16Bits::from_f32(1.0),
        dmin: F16Bits::from_f32(0.25),
        scale: [1; 8],
        min: [1; 8],
        qs: [0; 128],
    };
    Q4KHotCpuTensor::new(rows, cols, vec![block; rows * blocks_per_row]).unwrap()
}

fn selected_launch(plan: &augur_compiler::CpuCompiledPlan) -> augur_kernel_api::QMatmulF32Launch {
    match &plan.steps()[0] {
        CpuCompiledStep::QMatmulF32(args) => args.launch,
        other => panic!("unexpected step: {other:?}"),
    }
}

#[test]
fn baseline_compiler_selects_portable_identity() {
    let rows = 2;
    let cols = 256;
    let mut resources = CpuWeightResources::new();
    let storage = resources
        .add_q4k_hot_cpu_portable(q4k_hot_tensor(rows, cols))
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
    let output = graph.add_qmatmul(
        input,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
    );
    graph.try_set_output(output).unwrap();

    let plan = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![vec![0.0; cols]])
        .unwrap();

    let actual = selected_launch(&plan);
    assert!(std::ptr::addr_eq(
        actual as *const (),
        launch_qmatmul_q4k_f32_portable as *const ()
    ));
}

#[test]
fn native_compiler_selects_host_compatible_identity() {
    let rows = 2;
    let cols = 256;
    let mut resources = CpuWeightResources::new();
    let storage = resources
        .add_q4k_hot_cpu_portable(q4k_hot_tensor(rows, cols))
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
    let output = graph.add_qmatmul(
        input,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
    );
    graph.try_set_output(output).unwrap();

    let plan = GraphCompiler::cpu_native()
        .compile_cpu_plan(&graph, vec![vec![0.0; cols]])
        .unwrap();

    let actual = selected_launch(&plan);
    let expected = if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    {
        launch_qmatmul_q4k_f32_avx2
    } else {
        launch_qmatmul_q4k_f32_portable
    };
    assert!(std::ptr::addr_eq(
        actual as *const (),
        expected as *const ()
    ));
}
