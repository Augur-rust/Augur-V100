use augur_compiler::{CompilerError, CpuWeightResources, GraphCompiler};
use augur_core::F16Bits;
use augur_core::{
    CompileError, DType, GgmlError, QuantDesc, QuantFormatId, Shape, StorageRef, TensorDesc,
};

use augur_formats::{pack_q4k_scales_ref, Q4KCompactBlock, Q4KScales};
use augur_graph::Graph;
use augur_kernel_api::Q4KHotCpuTensor;
use augur_repack::repack_q4k_block_cpu_ref;

fn sample_block(seed: u8) -> Q4KCompactBlock {
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

#[test]
fn qmatmul_f32_lowering_rejects_quantized_activation_input() {
    let mut resources = CpuWeightResources::new();
    let hot = Q4KHotCpuTensor::new(
        2,
        256,
        vec![
            repack_q4k_block_cpu_ref(&sample_block(0)),
            repack_q4k_block_cpu_ref(&sample_block(1)),
        ],
    )
    .unwrap();
    let storage = resources.add_q4k_hot_cpu_portable(hot).unwrap();

    let mut g = Graph::new();

    let input = g.add_input(TensorDesc {
        dtype: DType::I8,
        shape: Shape::from_dims(&[256]),
        layout: augur_core::LogicalLayout::Contiguous,
        storage: StorageRef::None,
        quant: Some(QuantDesc {
            format: QuantFormatId::Q8K,
            block: 256,
            version: 1,
        }),
    });

    let out_desc = TensorDesc {
        dtype: DType::F32,
        shape: Shape::from_dims(&[2]),
        layout: augur_core::LogicalLayout::Contiguous,
        storage: StorageRef::None,
        quant: None,
    };

    let weight = g
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[2, 256]),
            StorageRef::HostOwned { id: storage },
            QuantFormatId::Q4K,
            256,
            1,
        ))
        .unwrap();

    let _out = g.add_qmatmul(input, weight, out_desc);
    g.try_set_output(_out).unwrap();

    let compiler = GraphCompiler::cpu_baseline();
    let err = compiler
        .compile_cpu_with_weights(&g, vec![vec![0.0f32; 256]], resources)
        .unwrap_err();

    assert_eq!(
        err,
        CompilerError::Graph(CompileError::Graph(GgmlError::DTypeMismatch)),
    );
}
