use std::collections::HashSet;

use augur_core::{DType, PhysicalLayoutId, QuantFormatId};
use augur_kernel_api::{
    ArchKind, BackendKind, CpuFeatureSet, CpuKernelAbi, EpilogueKind, Iq2XxsHotCpuTensor,
    KernelKey, OpKind, Q4KHotCpuTensor,
};
use augur_kernels::cpu_kernel_registrations;

#[test]
fn registry_contains_expected_portable_cpu_matrix() {
    let registrations = cpu_kernel_registrations();

    #[cfg(not(target_arch = "x86_64"))]
    assert_eq!(registrations.len(), 11);
    #[cfg(target_arch = "x86_64")]
    assert_eq!(registrations.len(), 13);

    let mut ids = HashSet::new();
    let mut key_priorities: HashSet<(KernelKey, u16)> = HashSet::new();

    for registration in registrations {
        assert!(ids.insert(registration.id));

        let key_priority = (registration.key, registration.priority);

        assert!(key_priorities.insert(key_priority));

        assert_eq!(registration.key.backend, BackendKind::Cpu);

        assert!(
            registration.key.arch == ArchKind::Portable || registration.key.arch == ArchKind::Avx2,
            "unexpected arch: {:?}",
            registration.key.arch,
        );

        assert!(matches!(
            registration.key.op,
            OpKind::QMatmul | OpKind::FusedMlp
        ));

        // AVX2 registrations require features; portable registrations do not.
    }

    let q4k: Vec<_> = registrations
        .iter()
        .filter(|registration| registration.key.weight_format == Some(QuantFormatId::Q4K))
        .collect();

    let dual_scale: Vec<_> = registrations
        .iter()
        .filter(|registration| registration.key.weight_format == Some(QuantFormatId::DualScale))
        .collect();

    let q8_0: Vec<_> = registrations
        .iter()
        .filter(|registration| registration.key.weight_format == Some(QuantFormatId::Q8_0))
        .collect();
    let q2k: Vec<_> = registrations
        .iter()
        .filter(|registration| registration.key.weight_format == Some(QuantFormatId::Q2K))
        .collect();
    let iq2_xxs: Vec<_> = registrations
        .iter()
        .filter(|registration| registration.key.weight_format == Some(QuantFormatId::IQ2_XXS))
        .collect();
    let mxfp4: Vec<_> = registrations
        .iter()
        .filter(|registration| registration.key.weight_format == Some(QuantFormatId::MXFP4))
        .collect();
    let fp8_scale_grid: Vec<_> = registrations
        .iter()
        .filter(|registration| {
            registration.key.weight_format == Some(QuantFormatId::FP8E4M3ScaleGrid)
        })
        .collect();

    #[cfg(not(target_arch = "x86_64"))]
    assert_eq!(q4k.len(), 4);
    #[cfg(target_arch = "x86_64")]
    assert_eq!(q4k.len(), 5);
    #[cfg(not(target_arch = "x86_64"))]
    assert_eq!(dual_scale.len(), 1);
    #[cfg(target_arch = "x86_64")]
    assert_eq!(dual_scale.len(), 2);
    assert_eq!(q8_0.len(), 1);
    assert_eq!(q2k.len(), 1);
    assert_eq!(iq2_xxs.len(), 1);
    assert_eq!(mxfp4.len(), 1);
    assert_eq!(fp8_scale_grid.len(), 2);

    let q4k_epilogues: HashSet<_> = q4k
        .iter()
        .map(|registration| registration.key.epilogue)
        .collect();

    assert_eq!(
        q4k_epilogues,
        HashSet::from([
            EpilogueKind::Identity,
            EpilogueKind::Bias,
            EpilogueKind::BiasSilu,
        ])
    );

    for registration in q4k {
        assert_eq!(registration.key.weight_dtype, DType::PackedU8);

        assert_eq!(registration.key.activation_dtype, DType::F32);

        assert_eq!(registration.key.output_dtype, DType::F32);

        assert_eq!(registration.key.activation_format, None);

        assert_eq!(
            registration.key.weight_layout,
            Q4KHotCpuTensor::PHYSICAL_LAYOUT
        );

        assert_eq!(
            registration.key.weight_layout,
            PhysicalLayoutId::Q4K_HOT_CPU_PORTABLE_V1
        );

        assert_ne!(
            registration.key.weight_layout,
            PhysicalLayoutId::Q4K_COMPACT_V1
        );

        assert_ne!(
            registration.key.weight_layout,
            PhysicalLayoutId::Q4K_HOT_AVX2_V1
        );

        assert_eq!(
            registration.key.activation_layout,
            PhysicalLayoutId::HOST_CONTIGUOUS_F32
        );

        match registration.key.epilogue {
            EpilogueKind::Identity => {
                let expected = if registration.id.0 == "cpu.portable.q4k_f32.prefill.identity.v1" {
                    CpuKernelAbi::QMatmulF32Prefill
                } else {
                    CpuKernelAbi::QMatmulF32
                };
                assert_eq!(registration.launch.abi(), expected);
            }

            EpilogueKind::Bias | EpilogueKind::BiasSilu => {
                assert_eq!(registration.launch.abi(), CpuKernelAbi::QMatmulBiasF32);
            }

            unexpected => {
                panic!(
                    "unexpected Q4K epilogue: \
                     {unexpected:?}"
                );
            }
        }
    }

    // M28-D: find the portable DualScale registration specifically, since
    // the AVX2 registration may be listed first on x86_64.
    let registration = dual_scale
        .iter()
        .find(|r| r.key.arch == ArchKind::Portable)
        .expect("portable DualScale registration not found");

    assert_eq!(registration.key, KernelKey::dual_scale_q8hot_cpu_portable());

    assert_eq!(registration.key.weight_dtype, DType::PackedU8);

    assert_eq!(registration.key.activation_dtype, DType::I8);

    assert_eq!(registration.key.output_dtype, DType::F32);

    assert_eq!(
        registration.key.weight_format,
        Some(QuantFormatId::DualScale)
    );

    assert_eq!(registration.key.activation_format, Some(QuantFormatId::Q8K));

    assert_eq!(
        registration.key.weight_layout,
        PhysicalLayoutId::DUAL_SCALE_HOT_CPU_PORTABLE_V1
    );

    assert_ne!(
        registration.key.weight_layout,
        PhysicalLayoutId::DUAL_SCALE_COMPACT_V1
    );

    assert_eq!(
        registration.key.activation_layout,
        PhysicalLayoutId::Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1
    );

    assert_ne!(
        registration.key.activation_layout,
        PhysicalLayoutId::Q8K_ACTIVATION_REF_VECTOR_V1
    );

    assert_eq!(
        registration.key.output_layout,
        PhysicalLayoutId::HOST_CONTIGUOUS_F32
    );

    assert_eq!(registration.key.epilogue, EpilogueKind::Identity);

    assert_eq!(registration.launch.abi(), CpuKernelAbi::DualScaleQ8Hot);

    assert_eq!(registration.required_features, CpuFeatureSet::NONE);

    for (registrations, key, abi, compact_layout) in [
        (
            q8_0,
            KernelKey::q8_0_q8hot_cpu_portable(),
            CpuKernelAbi::Q8_0Q8Hot,
            PhysicalLayoutId::Q8_0_COMPACT_V1,
        ),
        (
            q2k,
            KernelKey::q2k_q8hot_cpu_portable(),
            CpuKernelAbi::Q2KQ8Hot,
            PhysicalLayoutId::Q2K_COMPACT_V1,
        ),
        (
            iq2_xxs,
            KernelKey::iq2_xxs_q8hot_cpu_portable(),
            CpuKernelAbi::Iq2XxsQ8Hot,
            PhysicalLayoutId::IQ2_XXS_COMPACT_V1,
        ),
    ] {
        let registration = registrations[0];
        assert_eq!(registration.key, key);
        assert_eq!(registration.launch.abi(), abi);
        assert_eq!(registration.required_features, CpuFeatureSet::NONE);
        assert_eq!(registration.key.activation_format, Some(QuantFormatId::Q8K));
        assert_eq!(
            registration.key.activation_layout,
            PhysicalLayoutId::Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1
        );
        assert_ne!(registration.key.weight_layout, compact_layout);
        if registration.key.weight_format == Some(QuantFormatId::IQ2_XXS) {
            assert_eq!(
                registration.key.weight_layout,
                Iq2XxsHotCpuTensor::PHYSICAL_LAYOUT
            );
        }
    }

    let mxfp4 = mxfp4[0];
    assert_eq!(mxfp4.key, KernelKey::mxfp4_f32_cpu_portable());
    assert_eq!(mxfp4.launch.abi(), CpuKernelAbi::Mxfp4F32);
    assert_eq!(mxfp4.required_features, CpuFeatureSet::NONE);
    assert_eq!(mxfp4.key.activation_format, None);
    assert_eq!(
        mxfp4.key.weight_layout,
        PhysicalLayoutId::MXFP4_HOT_CPU_PORTABLE_V1
    );
    assert_eq!(
        mxfp4.key.activation_layout,
        PhysicalLayoutId::HOST_CONTIGUOUS_F32
    );

    let fp8_qmatmul = fp8_scale_grid
        .iter()
        .find(|registration| registration.key.op == OpKind::QMatmul)
        .expect("FP8 scale-grid QMatmul registration not found");
    assert_eq!(
        fp8_qmatmul.key,
        KernelKey::fp8_scale_grid_f32_cpu_portable()
    );
    assert_eq!(fp8_qmatmul.launch.abi(), CpuKernelAbi::Fp8ScaleGridF32);
    assert_eq!(fp8_qmatmul.required_features, CpuFeatureSet::NONE);

    let fp8_fused_mlp = fp8_scale_grid
        .iter()
        .find(|registration| registration.key.op == OpKind::FusedMlp)
        .expect("FP8 scale-grid fused MLP registration not found");
    assert_eq!(
        fp8_fused_mlp.key,
        KernelKey::fp8_scale_grid_fused_mlp_f32_cpu_portable()
    );
    assert_eq!(fp8_fused_mlp.launch.abi(), CpuKernelAbi::Fp8FusedMlpF32);
    assert_eq!(fp8_fused_mlp.required_features, CpuFeatureSet::NONE);
}

#[test]
fn registry_contains_no_reference_q8_or_compact_dual_scale_kernel() {
    for registration in cpu_kernel_registrations() {
        assert_ne!(
            registration.key.activation_layout,
            PhysicalLayoutId::Q8K_ACTIVATION_REF_VECTOR_V1
        );

        assert_ne!(
            registration.key.weight_layout,
            PhysicalLayoutId::DUAL_SCALE_COMPACT_V1
        );
        assert_ne!(
            registration.key.weight_layout,
            PhysicalLayoutId::Q8_0_COMPACT_V1
        );
        assert_ne!(
            registration.key.weight_layout,
            PhysicalLayoutId::Q2K_COMPACT_V1
        );
        assert_ne!(
            registration.key.weight_layout,
            PhysicalLayoutId::IQ2_XXS_COMPACT_V1
        );
    }
}

#[test]
fn dual_scale_registry_contains_identity_only() {
    let dual_scale: Vec<_> = cpu_kernel_registrations()
        .iter()
        .filter(|registration| registration.key.weight_format == Some(QuantFormatId::DualScale))
        .collect();

    #[cfg(not(target_arch = "x86_64"))]
    assert_eq!(dual_scale.len(), 1);
    #[cfg(target_arch = "x86_64")]
    assert_eq!(dual_scale.len(), 2);

    for registration in &dual_scale {
        assert_eq!(registration.key.epilogue, EpilogueKind::Identity);
        assert_eq!(registration.launch.abi(), CpuKernelAbi::DualScaleQ8Hot);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod m27_d_tests {
    use augur_core::PhysicalLayoutId;
    use augur_kernel_api::{
        ArchKind, CpuFeatureSet, CpuKernelAbi, EpilogueKind, KernelKey, KernelRegistrationId,
    };
    use augur_kernels::cpu_kernel_registrations;

    #[test]
    fn avx2_identity_registration_contract_is_exact() {
        let reg = cpu_kernel_registrations()
            .iter()
            .find(|r| {
                r.key.arch == ArchKind::Avx2
                    && r.key.weight_format == Some(augur_core::QuantFormatId::Q4K)
            })
            .expect("AVX2 Q4K registration not found");

        assert_eq!(
            reg.id,
            KernelRegistrationId("cpu.avx2_fma.q4k_f32.qmatmul.identity.v1")
        );
        assert_eq!(reg.key, KernelKey::q4k_f32_cpu_avx2_identity());
        assert_eq!(reg.key.arch, ArchKind::Avx2);
        assert_eq!(reg.key.epilogue, EpilogueKind::Identity);
        assert_eq!(
            reg.key.weight_layout,
            PhysicalLayoutId::Q4K_HOT_CPU_PORTABLE_V1
        );
        assert_eq!(
            reg.required_features,
            CpuFeatureSet::AVX2.union(CpuFeatureSet::FMA)
        );
        assert_eq!(reg.launch.abi(), CpuKernelAbi::QMatmulF32);
        assert_eq!(reg.priority, 200);
    }

    #[test]
    fn avx2_registration_preserves_launch_identity() {
        let reg = cpu_kernel_registrations()
            .iter()
            .find(|r| {
                r.key.arch == ArchKind::Avx2
                    && r.key.weight_format == Some(augur_core::QuantFormatId::Q4K)
            })
            .expect("AVX2 Q4K registration not found");
        let actual = reg.launch.qmatmul_f32().unwrap();
        let expected = augur_kernels::x86::avx2::launch_qmatmul_q4k_f32_avx2;
        let actual_addr = actual as *const ();
        let expected_addr = expected as *const ();
        assert!(std::ptr::addr_eq(actual_addr, expected_addr));
    }

    #[test]
    fn avx2_registration_does_not_replace_portable_identity() {
        let portable_count = cpu_kernel_registrations()
            .iter()
            .filter(|r| {
                r.key.arch == ArchKind::Portable && r.key.epilogue == EpilogueKind::Identity
            })
            .count();
        assert!(
            portable_count >= 1,
            "Portable Identity registration should still exist"
        );
    }
}

// ── M28-D DualScale AVX2 registration tests ─────────────────────────────

#[cfg(all(test, target_arch = "x86_64"))]
mod m28_d_tests {
    use augur_core::PhysicalLayoutId;
    use augur_kernel_api::{
        ArchKind, CpuFeatureSet, CpuKernelAbi, EpilogueKind, KernelKey, KernelRegistrationId,
    };
    use augur_kernels::cpu_kernel_registrations;

    fn find_dual_scale_avx2() -> &'static augur_kernel_api::CpuKernelRegistration {
        cpu_kernel_registrations()
            .iter()
            .find(|r| {
                r.key.arch == ArchKind::Avx2
                    && r.key.weight_format == Some(augur_core::QuantFormatId::DualScale)
            })
            .expect("DualScale AVX2 registration not found")
    }

    #[test]
    fn dual_scale_avx2_registration_contract_is_exact() {
        let reg = find_dual_scale_avx2();

        assert_eq!(
            reg.id,
            KernelRegistrationId("cpu.avx2.dual_scale_q8hot.qmatmul.identity.v1")
        );
        assert_eq!(reg.key, KernelKey::dual_scale_q8hot_cpu_avx2_identity());
        assert_eq!(reg.key.arch, ArchKind::Avx2);
        assert_eq!(reg.key.epilogue, EpilogueKind::Identity);
        assert_eq!(
            reg.key.weight_layout,
            PhysicalLayoutId::DUAL_SCALE_HOT_CPU_PORTABLE_V1
        );
        assert_eq!(
            reg.key.activation_layout,
            PhysicalLayoutId::Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1
        );
        assert_eq!(reg.required_features, CpuFeatureSet::AVX2);
        assert_eq!(reg.launch.abi(), CpuKernelAbi::DualScaleQ8Hot);
        assert_eq!(reg.priority, 200);
    }

    #[test]
    fn dual_scale_avx2_does_not_require_fma() {
        let reg = find_dual_scale_avx2();
        assert_eq!(reg.required_features, CpuFeatureSet::AVX2);
        assert_ne!(
            reg.required_features,
            CpuFeatureSet::AVX2.union(CpuFeatureSet::FMA)
        );
    }

    #[test]
    fn dual_scale_avx2_registration_preserves_launch_identity() {
        let reg = find_dual_scale_avx2();
        let actual = reg.launch.dual_scale_q8hot().unwrap();
        let expected = augur_kernels::x86::avx2::launch_qmatvec_dual_scale_q8k_hot_cpu_avx2;
        let actual_addr = actual as *const ();
        let expected_addr = expected as *const ();
        assert!(std::ptr::addr_eq(actual_addr, expected_addr));
    }

    #[test]
    fn dual_scale_avx2_portable_still_present() {
        let portable_count = cpu_kernel_registrations()
            .iter()
            .filter(|r| {
                r.key.arch == ArchKind::Portable
                    && r.key.weight_format == Some(augur_core::QuantFormatId::DualScale)
            })
            .count();
        assert!(
            portable_count >= 1,
            "Portable DualScale registration should still exist"
        );
    }
}
