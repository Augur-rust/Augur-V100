use augur_backend::CpuStream;
use augur_compiler::{CpuWeightResources, GraphCompiler};
use augur_core::{F16Bits, QuantFormatId, Shape, StorageRef, TensorDesc};

use augur_formats::{pack_q4k_scales_ref, Q4KCompactBlock, Q4KScales};
use augur_graph::Graph;
use augur_kernel_api::Q4KHotCpuTensor;
use augur_repack::{decode_q4k_hot_cpu_block_ref, repack_q4k_block_cpu_ref};

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

fn dense_q4k_row(compact: &Q4KCompactBlock) -> Vec<f32> {
    let hot = repack_q4k_block_cpu_ref(compact);
    let mut out = vec![0.0f32; 256];
    decode_q4k_hot_cpu_block_ref(&hot, &mut out).unwrap();
    out
}

#[test]
fn execution_plan_qmatmul_f32_portable_matches_dense_reference() {
    // 1 row × 256 cols
    let compact = sample_compact_block(3);
    let dense_w = dense_q4k_row(&compact);

    let hot = repack_q4k_block_cpu_ref(&compact);
    let tensor = Q4KHotCpuTensor::new(1, 256, vec![hot]).unwrap();

    let mut resources = CpuWeightResources::new();
    let storage_id = resources.add_q4k_hot_cpu_portable(tensor).unwrap();

    let mut g = Graph::new();
    let act_desc = TensorDesc::f32_contiguous(Shape::from_dims(&[256]));
    let out_desc = TensorDesc::f32_contiguous(Shape::from_dims(&[1]));
    let input = g.add_input(act_desc);
    let weight = g
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[1, 256]),
            StorageRef::HostOwned { id: storage_id },
            QuantFormatId::Q4K,
            256,
            1,
        ))
        .unwrap();
    let y = g.add_qmatmul(input, weight, out_desc);
    g.try_set_output(y).unwrap();

    let activation: Vec<f32> = (0..256).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect();

    let compiler = GraphCompiler::cpu_baseline();
    let mut plan = compiler
        .compile_cpu_with_weights(&g, vec![activation.clone()], resources)
        .unwrap();
    let mut stream = CpuStream;
    plan.execute_cpu(&mut stream).unwrap();

    let out_buf_idx = plan.memory().buffer_for(y).unwrap();
    let result = plan.buffers()[out_buf_idx].as_f32().unwrap();

    let expected: f32 = dense_w
        .iter()
        .zip(activation.iter())
        .map(|(w, a)| w * a)
        .sum();
    assert!((result[0] - expected).abs() < 1e-5);
}

#[test]
fn execution_plan_qmatmul_bias_silu_matches_dense_reference() {
    // 2 rows × 256 cols
    let compact0 = sample_compact_block(0);
    let compact1 = sample_compact_block(7);
    let dense0 = dense_q4k_row(&compact0);
    let dense1 = dense_q4k_row(&compact1);

    let blocks = vec![
        repack_q4k_block_cpu_ref(&compact0),
        repack_q4k_block_cpu_ref(&compact1),
    ];
    let tensor = Q4KHotCpuTensor::new(2, 256, blocks).unwrap();

    let mut resources = CpuWeightResources::new();
    let storage_id = resources.add_q4k_hot_cpu_portable(tensor).unwrap();

    let mut g = Graph::new();
    let act_desc = TensorDesc::f32_contiguous(Shape::from_dims(&[256]));
    let bias_desc = TensorDesc::f32_contiguous(Shape::from_dims(&[2]));
    let out_desc = TensorDesc::f32_contiguous(Shape::from_dims(&[2]));

    let input = g.add_input(act_desc);
    let bias = g.add_input(bias_desc);
    let weight = g
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[2, 256]),
            StorageRef::HostOwned { id: storage_id },
            QuantFormatId::Q4K,
            256,
            1,
        ))
        .unwrap();
    let y = g.add_qmatmul_bias_silu(input, weight, bias, out_desc);
    g.try_set_output(y).unwrap();

    let activation: Vec<f32> = (0..256).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect();
    let bias_vals = vec![1.5_f32, -0.5_f32];

    let compiler = GraphCompiler::cpu_baseline();
    let mut plan = compiler
        .compile_cpu_with_weights(&g, vec![activation.clone(), bias_vals.clone()], resources)
        .unwrap();
    let mut stream = CpuStream;
    plan.execute_cpu(&mut stream).unwrap();

    let out_buf_idx = plan.memory().buffer_for(y).unwrap();
    let result = plan.buffers()[out_buf_idx].as_f32().unwrap();

    // Reference: dot + bias + silu per row
    for row in 0..2 {
        let dense = if row == 0 { &dense0 } else { &dense1 };
        let dot: f32 = dense
            .iter()
            .zip(activation.iter())
            .map(|(w, a)| w * a)
            .sum();
        let x = dot + bias_vals[row];
        let expected = x / (1.0 + (-x).exp());
        assert!(
            (result[row] - expected).abs() < 1e-5,
            "row {row}: result={}, expected={}",
            result[row],
            expected
        );
    }
}

#[test]
fn execution_plan_qmatmul_bias_matches_dense_reference() {
    // 2 rows × 256 cols — bias only, no silu
    let compact0 = sample_compact_block(0);
    let compact1 = sample_compact_block(7);
    let dense0 = dense_q4k_row(&compact0);
    let dense1 = dense_q4k_row(&compact1);

    let blocks = vec![
        repack_q4k_block_cpu_ref(&compact0),
        repack_q4k_block_cpu_ref(&compact1),
    ];
    let tensor = Q4KHotCpuTensor::new(2, 256, blocks).unwrap();

    let mut resources = CpuWeightResources::new();
    let storage_id = resources.add_q4k_hot_cpu_portable(tensor).unwrap();

    let mut g = Graph::new();
    let act_desc = TensorDesc::f32_contiguous(Shape::from_dims(&[256]));
    let bias_desc = TensorDesc::f32_contiguous(Shape::from_dims(&[2]));
    let out_desc = TensorDesc::f32_contiguous(Shape::from_dims(&[2]));

    let input = g.add_input(act_desc);
    let bias = g.add_input(bias_desc);
    let weight = g
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[2, 256]),
            StorageRef::HostOwned { id: storage_id },
            QuantFormatId::Q4K,
            256,
            1,
        ))
        .unwrap();
    let y = g.add_qmatmul_bias(input, weight, bias, out_desc);
    g.try_set_output(y).unwrap();

    let activation: Vec<f32> = (0..256).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect();
    let bias_vals = vec![10.0_f32, -3.0_f32];

    let compiler = GraphCompiler::cpu_baseline();
    let mut plan = compiler
        .compile_cpu_with_weights(&g, vec![activation.clone(), bias_vals.clone()], resources)
        .unwrap();
    let mut stream = CpuStream;
    plan.execute_cpu(&mut stream).unwrap();

    let out_buf_idx = plan.memory().buffer_for(y).unwrap();
    let result = plan.buffers()[out_buf_idx].as_f32().unwrap();

    for row in 0..2 {
        let dense = if row == 0 { &dense0 } else { &dense1 };
        let dot: f32 = dense
            .iter()
            .zip(activation.iter())
            .map(|(w, a)| w * a)
            .sum();
        let expected = dot + bias_vals[row];
        assert!(
            (result[row] - expected).abs() < 1e-5,
            "row {row}: result={}, expected={}",
            result[row],
            expected
        );
    }
}
