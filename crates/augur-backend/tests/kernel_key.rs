use std::collections::HashSet;

use augur_core::PhysicalLayoutId;
use augur_kernel_api::{BatchClass, KernelKey, ShapeClass, SizeClass, TileConfig, TunedKernelKey};

#[test]
fn kernel_key_hashes_by_full_identity() {
    let key = KernelKey::q4k_f32_cpu_portable(augur_kernel_api::EpilogueKind::Identity);

    let mut set = HashSet::new();
    assert!(set.insert(key));
    assert!(!set.insert(key));
}

#[test]
fn kernel_key_distinguishes_activation_layout() {
    let a = KernelKey::q4k_f32_cpu_portable(augur_kernel_api::EpilogueKind::Identity);
    let mut b = a;

    b.activation_layout = PhysicalLayoutId::Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1;

    assert_ne!(a, b);
}

#[test]
fn tuned_kernel_key_distinguishes_tile_config() {
    let kernel = KernelKey::q4k_f32_cpu_portable(augur_kernel_api::EpilogueKind::Identity);

    let a = TunedKernelKey {
        kernel,
        shape: ShapeClass::DECODE_GEMV,
        tile: TileConfig::PORTABLE_GEMV,
    };

    let b = TunedKernelKey {
        kernel,
        shape: ShapeClass::DECODE_GEMV,
        tile: TileConfig {
            tile_k: 128,
            ..TileConfig::PORTABLE_GEMV
        },
    };

    assert_ne!(a, b);
}

#[test]
fn shape_class_classifies_decode_and_prefill() {
    let decode = ShapeClass::classify_matmul(1, 4096, 4096, 1);
    assert_eq!(decode.batch, BatchClass::Decode);

    let prefill = ShapeClass::classify_matmul(128, 4096, 4096, 128);
    assert_eq!(prefill.batch, BatchClass::Prefill);

    assert_eq!(decode.k, SizeClass::Medium);
    assert_eq!(prefill.m, SizeClass::Small);
}
