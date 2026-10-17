use augur_formats::{Fp8E4M3CompactBlock, Mxfp4CompactBlock, Nvfp4CompactBlock};
use augur_kernels::portable::{
    qmatvec_fp8_e4m3_f32_cpu_portable, qmatvec_mxfp4_f32_cpu_portable,
    qmatvec_nvfp4_f32_cpu_portable,
};
use augur_reference::{qmatvec_fp8_e4m3_f32_ref, qmatvec_mxfp4_f32_ref, qmatvec_nvfp4_f32_ref};
use augur_repack::{repack_fp8_e4m3_tensor_cpu, repack_mxfp4_tensor_cpu, repack_nvfp4_tensor_cpu};

#[test]
fn production_mxfp4_and_nvfp4_match_independent_compact_reference() {
    let activation: Vec<f32> = (0..64).map(|index| (index as f32 - 31.0) / 17.0).collect();
    let mx = vec![
        Mxfp4CompactBlock {
            qs: [0x76; 16],
            scale_e8m0: 124,
        },
        Mxfp4CompactBlock {
            qs: [0xab; 16],
            scale_e8m0: 126,
        },
        Mxfp4CompactBlock {
            qs: [0x31; 16],
            scale_e8m0: 127,
        },
        Mxfp4CompactBlock {
            qs: [0xed; 16],
            scale_e8m0: 123,
        },
    ];
    let mx_hot = repack_mxfp4_tensor_cpu(&mx, 2, 64).unwrap();
    let mut mx_actual = [0.0; 2];
    let mut mx_expected = [0.0; 2];
    qmatvec_mxfp4_f32_cpu_portable(&mx_hot, &activation, &mut mx_actual).unwrap();
    qmatvec_mxfp4_f32_ref(&mx, 2, 64, &activation, &mut mx_expected).unwrap();
    assert_eq!(mx_actual, mx_expected);

    let nv = (0..8)
        .map(|index| Nvfp4CompactBlock {
            qs: [0x5c_u8.wrapping_add(index as u8); 8],
            scale_e4m3: 0x30 + index as u8,
            global_scale: 0.125,
        })
        .collect::<Vec<_>>();
    let nv_hot = repack_nvfp4_tensor_cpu(&nv, 2, 64).unwrap();
    let mut nv_actual = [0.0; 2];
    let mut nv_expected = [0.0; 2];
    qmatvec_nvfp4_f32_cpu_portable(&nv_hot, &activation, &mut nv_actual).unwrap();
    qmatvec_nvfp4_f32_ref(&nv, 2, 64, &activation, &mut nv_expected).unwrap();
    assert_eq!(nv_actual, nv_expected);
}

#[test]
fn invalid_scale_and_shapes_fail_closed() {
    let bad = Nvfp4CompactBlock {
        qs: [0; 8],
        scale_e4m3: 0x7f,
        global_scale: 1.0,
    };
    assert!(augur_repack::repack_nvfp4_block_cpu(&bad).is_err());
    let block = Mxfp4CompactBlock {
        qs: [0; 16],
        scale_e8m0: 127,
    };
    assert!(repack_mxfp4_tensor_cpu(&[block], 1, 31).is_err());
}

#[test]
fn shared_fp8_weight_path_matches_compact_reference() {
    let activation: Vec<f32> = (0..256)
        .map(|index| (index as f32 * 0.03125).cos())
        .collect();
    let blocks = (0..4)
        .map(|index| Fp8E4M3CompactBlock {
            qs: [0x38 + index as u8; 128],
            scale_e8m0: 124 + index as u8,
        })
        .collect::<Vec<_>>();
    let hot = repack_fp8_e4m3_tensor_cpu(&blocks, 2, 256).unwrap();
    let mut actual = [0.0; 2];
    let mut expected = [0.0; 2];
    qmatvec_fp8_e4m3_f32_cpu_portable(&hot, &activation, &mut actual).unwrap();
    qmatvec_fp8_e4m3_f32_ref(&blocks, 2, 256, &activation, &mut expected).unwrap();
    assert_eq!(actual, expected);
}
