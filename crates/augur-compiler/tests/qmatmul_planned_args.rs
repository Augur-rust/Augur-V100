use augur_compiler::{CpuWeightResources, GraphCompiler};
use augur_core::{F16Bits, QuantFormatId, Shape, StorageRef, TensorDesc};
use augur_exec::PlannedOp;
use augur_formats::{pack_q4k_scales_ref, Q4KCompactBlock, Q4KScales};
use augur_graph::Graph;
use augur_kernel_api::Q4KHotCpuTensor;
use augur_repack::repack_q4k_block_cpu_ref;

fn sample_compact_block(seed: u8) -> Q4KCompactBlock {
    let scales = Q4KScales {
        scale: [1 + (seed & 3), 2, 3, 4, 17, 18, 35, 63],
        min: [5, 6, 7, 8, 9, 26, 42, 61],
    };
    let mut qs = [0u8; 128];
    for (i, q) in qs.iter_mut().enumerate() {
        let lo = ((i as u8).wrapping_add(seed)) & 0x0f;
        let hi = (15u8.wrapping_sub(i as u8).wrapping_add(seed)) & 0x0f;
        *q = lo | (hi << 4);
    }
    Q4KCompactBlock {
        d: F16Bits::from_bits(0x3c00),
        dmin: F16Bits::from_bits(0x3800),
        scales: pack_q4k_scales_ref(scales).unwrap(),
        qs,
    }
}

fn add_weight_to_graph(
    g: &mut Graph,
    resources: &mut augur_compiler::CpuWeightResources,
    rows: usize,
    cols: usize,
    blocks: Vec<augur_kernel_api::Q4KHotCpuBlock>,
) -> augur_core::TensorId {
    let tensor = Q4KHotCpuTensor::new(rows, cols, blocks).unwrap();
    let storage_id = resources.add_q4k_hot_cpu_portable(tensor).unwrap();
    g.add_weight(TensorDesc::quantized_weight(
        Shape::from_dims(&[rows, cols]),
        StorageRef::HostOwned { id: storage_id },
        QuantFormatId::Q4K,
        256,
        1,
    ))
    .unwrap()
}

#[test]
fn lowering_qmatmul_bias_silu_keeps_concrete_planned_variant() {
    let blocks = vec![
        repack_q4k_block_cpu_ref(&sample_compact_block(0)),
        repack_q4k_block_cpu_ref(&sample_compact_block(1)),
        repack_q4k_block_cpu_ref(&sample_compact_block(2)),
        repack_q4k_block_cpu_ref(&sample_compact_block(3)),
    ];

    let mut resources = CpuWeightResources::new();
    let mut g = Graph::new();

    let input = g.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[512])));
    let bias = g.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[2])));

    let weight = add_weight_to_graph(&mut g, &mut resources, 2, 512, blocks);

    let y = g.add_qmatmul_bias_silu(
        input,
        weight,
        bias,
        TensorDesc::f32_contiguous(Shape::from_dims(&[2])),
    );
    g.try_set_output(y).unwrap();

    let activation: Vec<f32> = (0..512).map(|i| ((i % 19) as f32 - 9.0) * 0.0625).collect();
    let bias_vals = vec![0.125f32, -0.25f32];

    let compiler = GraphCompiler::cpu_baseline();
    let plan = compiler
        .compile_cpu_with_weights(&g, vec![activation, bias_vals], resources)
        .unwrap();

    match &plan.steps()[0] {
        PlannedOp::QMatmulBiasSiluF32(args) => {
            assert_eq!(args.base().rows(), 2);
            assert_eq!(args.base().cols(), 512);
        }
        other => panic!("expected QMatmulBiasSiluF32, got {other:?}"),
    }
}

#[test]
fn lowering_qmatmul_f32_keeps_concrete_planned_variant() {
    let blocks = vec![repack_q4k_block_cpu_ref(&sample_compact_block(5))];

    let mut resources = CpuWeightResources::new();
    let mut g = Graph::new();

    let input = g.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[256])));

    let weight = add_weight_to_graph(&mut g, &mut resources, 1, 256, blocks);

    let y = g.add_qmatmul(
        input,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[1])),
    );
    g.try_set_output(y).unwrap();

    let activation: Vec<f32> = (0..256)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.03125)
        .collect();

    let compiler = GraphCompiler::cpu_baseline();
    let plan = compiler
        .compile_cpu_with_weights(&g, vec![activation], resources)
        .unwrap();

    match &plan.steps()[0] {
        PlannedOp::QMatmulF32(args) => {
            assert_eq!(args.base().rows(), 1);
            assert_eq!(args.base().cols(), 256);
        }
        other => panic!("expected QMatmulF32, got {other:?}"),
    }
}
