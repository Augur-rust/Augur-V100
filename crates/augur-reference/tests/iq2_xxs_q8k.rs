use augur_core::F16Bits;
use augur_formats::{quantize_q8k_activation_vector_ref, Iq2XxsCompactBlock, IQ2_XXS_BLOCK_SIZE};
use augur_kernel_api::KernelLaunchError;
use augur_kernels::portable::launch_qmatvec_iq2_xxs_q8hot_cpu_portable;
use augur_reference::qmatvec_iq2_xxs_q8k_ref;
use augur_repack::{repack_iq2_xxs_tensor_cpu_ref, repack_q8k_activation_vector_cpu_ref};

fn compact_block(block_index: usize) -> Iq2XxsCompactBlock {
    let mut block = Iq2XxsCompactBlock::ZERO;
    block.d = F16Bits::from_f32(0.00390625 * (block_index + 1) as f32);
    for group in 0..8 {
        let grids = [
            ((block_index * 64 + group * 4) % 256) as u8,
            ((block_index * 64 + group * 4 + 1) % 256) as u8,
            ((block_index * 64 + group * 4 + 2) % 256) as u8,
            ((block_index * 64 + group * 4 + 3) % 256) as u8,
        ];
        block.qs[group * 4] = u16::from_le_bytes([grids[0], grids[1]]);
        block.qs[group * 4 + 1] = u16::from_le_bytes([grids[2], grids[3]]);
        let signs = [
            ((group * 11 + 1) & 0x7f) as u32,
            ((group * 13 + 3) & 0x7f) as u32,
            ((group * 17 + 5) & 0x7f) as u32,
            ((group * 19 + 7) & 0x7f) as u32,
        ];
        let sign_scale = signs[0]
            | (signs[1] << 7)
            | (signs[2] << 14)
            | (signs[3] << 21)
            | (((block_index + group) as u32 & 0x0f) << 28);
        let packed = sign_scale.to_le_bytes();
        block.qs[group * 4 + 2] = u16::from_le_bytes([packed[0], packed[1]]);
        block.qs[group * 4 + 3] = u16::from_le_bytes([packed[2], packed[3]]);
    }
    block
}

#[test]
fn production_matches_independent_reference_across_all_grids() {
    const ROWS: usize = 2;
    const COLS: usize = 2 * IQ2_XXS_BLOCK_SIZE;
    let compact = (0..ROWS * 2).map(compact_block).collect::<Vec<_>>();
    let hot = repack_iq2_xxs_tensor_cpu_ref(&compact, ROWS, COLS).unwrap();
    let source = (0..COLS)
        .map(|index| {
            let x = index as f32 - 255.5;
            (x * 0.073).sin() * (1.0 + (index % 9) as f32)
        })
        .collect::<Vec<_>>();
    let activation = quantize_q8k_activation_vector_ref(&source).unwrap();
    let activation_hot = repack_q8k_activation_vector_cpu_ref(&activation).unwrap();

    let mut reference = [0.0_f32; ROWS];
    qmatvec_iq2_xxs_q8k_ref(&hot, &activation, &mut reference).unwrap();
    let mut production = [0.0_f32; ROWS];
    launch_qmatvec_iq2_xxs_q8hot_cpu_portable(&hot, &activation_hot, &mut production).unwrap();
    for (actual, expected) in production.into_iter().zip(reference) {
        assert!(
            (actual - expected).abs() <= 2.0e-5,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn production_rejects_bad_activation_and_output_shapes() {
    let hot = repack_iq2_xxs_tensor_cpu_ref(&[compact_block(0)], 1, 256).unwrap();
    let activation = quantize_q8k_activation_vector_ref(&[1.0; 512]).unwrap();
    let activation_hot = repack_q8k_activation_vector_cpu_ref(&activation).unwrap();
    assert!(matches!(
        launch_qmatvec_iq2_xxs_q8hot_cpu_portable(&hot, &activation_hot, &mut [0.0]),
        Err(KernelLaunchError::BadActivationLength { .. })
    ));
    let activation = quantize_q8k_activation_vector_ref(&[1.0; 256]).unwrap();
    let activation_hot = repack_q8k_activation_vector_cpu_ref(&activation).unwrap();
    assert!(matches!(
        launch_qmatvec_iq2_xxs_q8hot_cpu_portable(&hot, &activation_hot, &mut []),
        Err(KernelLaunchError::BadOutputLength { .. })
    ));
}
