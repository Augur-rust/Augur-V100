use augur_compiler::{CpuWeightResources, GraphCompiler};
use augur_core::{F16Bits, QuantFormatId, Shape, StorageRef, TensorDesc};
use augur_exec::PlannedOp;
use augur_formats::{pack_q4k_scales_ref, Q4KCompactBlock, Q4KScales};
use augur_graph::Graph;
use augur_kernel_api::{
    CpuFeatureSet, CpuKernelAbi, EpilogueKind, Q4KHotCpuTensor, QMatmulF32Launch, ShapeClass,
};
use augur_kernels::cpu_kernel_registrations;
use augur_repack::repack_q4k_block_cpu_ref;

fn compact_block(seed: u8) -> Q4KCompactBlock {
    let scales = Q4KScales {
        scale: [1 + (seed & 3), 2, 3, 4, 17, 18, 35, 63],
        min: [5, 6, 7, 8, 9, 26, 42, 61],
    };
    let mut qs = [0_u8; 128];
    for (index, value) in qs.iter_mut().enumerate() {
        let low = (index as u8).wrapping_add(seed) & 0x0f;
        let high = 15_u8.wrapping_sub(index as u8).wrapping_add(seed) & 0x0f;
        *value = low | (high << 4);
    }
    Q4KCompactBlock {
        d: F16Bits::from_bits(0x3c00),
        dmin: F16Bits::from_bits(0x3800),
        scales: pack_q4k_scales_ref(scales).unwrap(),
        qs,
    }
}

fn registered_plain_launch() -> QMatmulF32Launch {
    cpu_kernel_registrations()
        .iter()
        .filter(|registration| registration.key.epilogue == EpilogueKind::Identity)
        .find(|registration| {
            registration.required_features == CpuFeatureSet::NONE
                && registration.launch.abi() == CpuKernelAbi::QMatmulF32
                && (registration.shape_support)(&ShapeClass::DECODE_GEMV)
        })
        .and_then(|registration| registration.launch.qmatmul_f32())
        .unwrap()
}

#[test]
fn lowering_binds_identity_launch_selected_by_registry() {
    let tensor = Q4KHotCpuTensor::new(
        2,
        256,
        vec![
            repack_q4k_block_cpu_ref(&compact_block(0)),
            repack_q4k_block_cpu_ref(&compact_block(7)),
        ],
    )
    .unwrap();

    let mut resources = CpuWeightResources::new();
    let storage = resources.add_q4k_hot_cpu_portable(tensor).unwrap();

    let mut graph = Graph::new();

    let input = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[256])));

    let weight = graph
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[2, 256]),
            StorageRef::HostOwned { id: storage },
            QuantFormatId::Q4K,
            256,
            1,
        ))
        .unwrap();

    let output = graph.add_qmatmul(
        input,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[2])),
    );

    graph.try_set_output(output).unwrap();

    let plan = GraphCompiler::cpu_baseline()
        .compile_cpu_with_weights(&graph, vec![vec![0.0_f32; 256]], resources)
        .unwrap();

    let actual = match &plan.steps()[0] {
        PlannedOp::QMatmulF32(args) => args.launch(),
        other => panic!("unexpected planned operation: {other:?}"),
    };

    let expected = registered_plain_launch();
    let actual_address = actual as *const ();
    let expected_address = expected as *const ();
    assert!(std::ptr::addr_eq(actual_address, expected_address));
}
