use augur_kernel_api::{
    BatchClass, CpuFeatureSet, CpuKernelLaunch, CpuKernelRegistration, CpuShapeSupportId,
    EpilogueKind, KernelKey, KernelRegistrationId, ShapeClass,
};

#[cfg(target_arch = "x86_64")]
use crate::x86::avx2::{launch_qmatmul_q4k_f32_avx2, launch_qmatvec_dual_scale_q8k_hot_cpu_avx2};

use crate::portable::{
    launch_fp8_e4m3_scale_grid_fused_mlp_f32_cpu_portable, launch_qmatmul_q4k_f32_bias_portable,
    launch_qmatmul_q4k_f32_bias_silu_portable, launch_qmatmul_q4k_f32_portable,
    launch_qmatmul_q4k_f32_prefill_portable, launch_qmatvec_dual_scale_q8k_hot_cpu_portable,
    launch_qmatvec_fp8_e4m3_scale_grid_f32_cpu_portable, launch_qmatvec_iq2_xxs_q8hot_cpu_portable,
    launch_qmatvec_mxfp4_f32_cpu_portable, launch_qmatvec_q2k_q8hot_cpu_portable,
    launch_qmatvec_q8_0_q8hot_cpu_portable,
};

pub const FP8_SCALE_GRID_F32_PORTABLE_IDENTITY: CpuKernelRegistration = CpuKernelRegistration {
    id: KernelRegistrationId("cpu.portable.fp8_e4m3_scale_grid_f32.qmatmul.identity.v1"),
    key: KernelKey::fp8_scale_grid_f32_cpu_portable(),
    shape_support_id: CPU_SHAPE_DECODE_QMATMUL_V1,
    shape_support: supports_decode_qmatmul,
    priority: 100,
    required_features: CpuFeatureSet::NONE,
    launch: CpuKernelLaunch::Fp8ScaleGridF32(launch_qmatvec_fp8_e4m3_scale_grid_f32_cpu_portable),
};

pub const FP8_SCALE_GRID_FUSED_MLP_F32_PORTABLE: CpuKernelRegistration = CpuKernelRegistration {
    id: KernelRegistrationId("cpu.portable.fp8_e4m3_scale_grid_f32.fused_mlp.clamp_silu_mul.v1"),
    key: KernelKey::fp8_scale_grid_fused_mlp_f32_cpu_portable(),
    shape_support_id: CPU_SHAPE_DECODE_QMATMUL_V1,
    shape_support: supports_decode_qmatmul,
    priority: 100,
    required_features: CpuFeatureSet::NONE,
    launch: CpuKernelLaunch::Fp8FusedMlpF32(launch_fp8_e4m3_scale_grid_fused_mlp_f32_cpu_portable),
};

pub const MXFP4_F32_PORTABLE_IDENTITY: CpuKernelRegistration = CpuKernelRegistration {
    id: KernelRegistrationId("cpu.portable.mxfp4_f32.qmatmul.identity.v1"),
    key: KernelKey::mxfp4_f32_cpu_portable(),
    shape_support_id: CPU_SHAPE_DECODE_QMATMUL_V1,
    shape_support: supports_decode_qmatmul,
    priority: 100,
    required_features: CpuFeatureSet::NONE,
    launch: CpuKernelLaunch::Mxfp4F32(launch_qmatvec_mxfp4_f32_cpu_portable),
};

pub const CPU_SHAPE_DECODE_QMATMUL_V1: CpuShapeSupportId =
    CpuShapeSupportId("cpu.shape.decode_qmatmul.v1");
pub const CPU_SHAPE_PREFILL_QMATMUL_V1: CpuShapeSupportId =
    CpuShapeSupportId("cpu.shape.prefill_qmatmul.v1");

pub const Q4K_F32_PORTABLE_PREFILL_IDENTITY: CpuKernelRegistration = CpuKernelRegistration {
    id: KernelRegistrationId("cpu.portable.q4k_f32.prefill.identity.v1"),
    key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
    shape_support_id: CPU_SHAPE_PREFILL_QMATMUL_V1,
    shape_support: supports_prefill_qmatmul,
    priority: 110,
    required_features: CpuFeatureSet::NONE,
    launch: CpuKernelLaunch::QMatmulF32Prefill(launch_qmatmul_q4k_f32_prefill_portable),
};

pub const Q4K_F32_PORTABLE_IDENTITY: CpuKernelRegistration = CpuKernelRegistration {
    id: KernelRegistrationId("cpu.portable.q4k_f32.qmatmul.identity.v1"),

    key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),

    shape_support_id: CPU_SHAPE_DECODE_QMATMUL_V1,
    shape_support: supports_decode_qmatmul,

    priority: 100,

    required_features: CpuFeatureSet::NONE,

    launch: CpuKernelLaunch::QMatmulF32(launch_qmatmul_q4k_f32_portable),
};

pub const Q4K_F32_PORTABLE_BIAS: CpuKernelRegistration = CpuKernelRegistration {
    id: KernelRegistrationId("cpu.portable.q4k_f32.qmatmul.bias.v1"),

    key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Bias),

    shape_support_id: CPU_SHAPE_DECODE_QMATMUL_V1,
    shape_support: supports_decode_qmatmul,

    priority: 100,

    required_features: CpuFeatureSet::NONE,

    launch: CpuKernelLaunch::QMatmulBiasF32(launch_qmatmul_q4k_f32_bias_portable),
};

pub const Q4K_F32_PORTABLE_BIAS_SILU: CpuKernelRegistration = CpuKernelRegistration {
    id: KernelRegistrationId("cpu.portable.q4k_f32.qmatmul.bias_silu.v1"),

    key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::BiasSilu),

    shape_support_id: CPU_SHAPE_DECODE_QMATMUL_V1,
    shape_support: supports_decode_qmatmul,

    priority: 100,

    required_features: CpuFeatureSet::NONE,

    launch: CpuKernelLaunch::QMatmulBiasF32(launch_qmatmul_q4k_f32_bias_silu_portable),
};

/// CPU portable decode-only DualScale × Q8Hot Identity qmatvec.
///
/// This registration consumes only:
///
/// - `DualScaleHotCpuTensor`;
/// - `Q8KActivationHotCpuVector`;
/// - caller-preallocated f32 output.
///
/// It does not register a Compact/reference Q8 path, and it does not imply
/// support for Bias, BiasSiLU or Prefill.
pub const DUAL_SCALE_Q8HOT_PORTABLE_IDENTITY: CpuKernelRegistration = CpuKernelRegistration {
    id: KernelRegistrationId("cpu.portable.dual_scale_q8hot.qmatmul.identity.v1"),

    key: KernelKey::dual_scale_q8hot_cpu_portable(),

    shape_support_id: CPU_SHAPE_DECODE_QMATMUL_V1,
    shape_support: supports_decode_qmatmul,

    priority: 100,

    required_features: CpuFeatureSet::NONE,

    launch: CpuKernelLaunch::DualScaleQ8Hot(launch_qmatvec_dual_scale_q8k_hot_cpu_portable),
};

pub const Q8_0_Q8HOT_PORTABLE_IDENTITY: CpuKernelRegistration = CpuKernelRegistration {
    id: KernelRegistrationId("cpu.portable.q8_0_q8hot.qmatmul.identity.v1"),
    key: KernelKey::q8_0_q8hot_cpu_portable(),
    shape_support_id: CPU_SHAPE_DECODE_QMATMUL_V1,
    shape_support: supports_decode_qmatmul,
    priority: 100,
    required_features: CpuFeatureSet::NONE,
    launch: CpuKernelLaunch::Q8_0Q8Hot(launch_qmatvec_q8_0_q8hot_cpu_portable),
};

pub const Q2K_Q8HOT_PORTABLE_IDENTITY: CpuKernelRegistration = CpuKernelRegistration {
    id: KernelRegistrationId("cpu.portable.q2k_q8hot.qmatmul.identity.v1"),
    key: KernelKey::q2k_q8hot_cpu_portable(),
    shape_support_id: CPU_SHAPE_DECODE_QMATMUL_V1,
    shape_support: supports_decode_qmatmul,
    priority: 100,
    required_features: CpuFeatureSet::NONE,
    launch: CpuKernelLaunch::Q2KQ8Hot(launch_qmatvec_q2k_q8hot_cpu_portable),
};

pub const IQ2_XXS_Q8HOT_PORTABLE_IDENTITY: CpuKernelRegistration = CpuKernelRegistration {
    id: KernelRegistrationId("cpu.portable.iq2_xxs_q8hot.qmatmul.identity.v1"),
    key: KernelKey::iq2_xxs_q8hot_cpu_portable(),
    shape_support_id: CPU_SHAPE_DECODE_QMATMUL_V1,
    shape_support: supports_decode_qmatmul,
    priority: 100,
    required_features: CpuFeatureSet::NONE,
    launch: CpuKernelLaunch::Iq2XxsQ8Hot(launch_qmatvec_iq2_xxs_q8hot_cpu_portable),
};

/// CPU AVX2 decode-only DualScale × Q8Hot Identity qmatvec.
///
/// M28 reuses `DUAL_SCALE_HOT_CPU_PORTABLE_V1` and
/// `Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1`. M28 ISA audit confirmed AVX2 only
/// (no FMA required). Bias and BiasSiLU are not part of M28.
#[cfg(target_arch = "x86_64")]
pub const DUAL_SCALE_Q8HOT_AVX2_IDENTITY: CpuKernelRegistration = CpuKernelRegistration {
    id: KernelRegistrationId("cpu.avx2.dual_scale_q8hot.qmatmul.identity.v1"),

    key: KernelKey::dual_scale_q8hot_cpu_avx2_identity(),

    shape_support_id: CPU_SHAPE_DECODE_QMATMUL_V1,
    shape_support: supports_decode_qmatmul,

    priority: 200,

    required_features: CpuFeatureSet::AVX2,

    launch: CpuKernelLaunch::DualScaleQ8Hot(launch_qmatvec_dual_scale_q8k_hot_cpu_avx2),
};

/// CPU AVX2+FMA decode-only Q4K x f32 Identity QMatmul.
///
/// M27 reuses Q4K_HOT_CPU_PORTABLE_V1; this registration does not imply an
/// AVX2-specific HotWeight layout. Bias and BiasSiLU remain portable-only.
#[cfg(target_arch = "x86_64")]
pub const Q4K_F32_AVX2_IDENTITY: CpuKernelRegistration = CpuKernelRegistration {
    id: KernelRegistrationId("cpu.avx2_fma.q4k_f32.qmatmul.identity.v1"),

    key: KernelKey::q4k_f32_cpu_avx2_identity(),

    shape_support_id: CPU_SHAPE_DECODE_QMATMUL_V1,
    shape_support: supports_decode_qmatmul,

    priority: 200,

    required_features: CpuFeatureSet::AVX2.union(CpuFeatureSet::FMA),

    launch: CpuKernelLaunch::QMatmulF32(launch_qmatmul_q4k_f32_avx2),
};

#[cfg(target_arch = "x86_64")]
pub static CPU_KERNEL_REGISTRATIONS: &[CpuKernelRegistration] = &[
    DUAL_SCALE_Q8HOT_AVX2_IDENTITY,
    Q4K_F32_AVX2_IDENTITY,
    Q4K_F32_PORTABLE_PREFILL_IDENTITY,
    Q4K_F32_PORTABLE_IDENTITY,
    Q4K_F32_PORTABLE_BIAS,
    Q4K_F32_PORTABLE_BIAS_SILU,
    DUAL_SCALE_Q8HOT_PORTABLE_IDENTITY,
    Q8_0_Q8HOT_PORTABLE_IDENTITY,
    Q2K_Q8HOT_PORTABLE_IDENTITY,
    IQ2_XXS_Q8HOT_PORTABLE_IDENTITY,
    MXFP4_F32_PORTABLE_IDENTITY,
    FP8_SCALE_GRID_F32_PORTABLE_IDENTITY,
    FP8_SCALE_GRID_FUSED_MLP_F32_PORTABLE,
];

#[cfg(not(target_arch = "x86_64"))]
pub static CPU_KERNEL_REGISTRATIONS: &[CpuKernelRegistration] = &[
    Q4K_F32_PORTABLE_PREFILL_IDENTITY,
    Q4K_F32_PORTABLE_IDENTITY,
    Q4K_F32_PORTABLE_BIAS,
    Q4K_F32_PORTABLE_BIAS_SILU,
    DUAL_SCALE_Q8HOT_PORTABLE_IDENTITY,
    Q8_0_Q8HOT_PORTABLE_IDENTITY,
    Q2K_Q8HOT_PORTABLE_IDENTITY,
    IQ2_XXS_Q8HOT_PORTABLE_IDENTITY,
    MXFP4_F32_PORTABLE_IDENTITY,
    FP8_SCALE_GRID_F32_PORTABLE_IDENTITY,
    FP8_SCALE_GRID_FUSED_MLP_F32_PORTABLE,
];

pub fn cpu_kernel_registrations() -> &'static [CpuKernelRegistration] {
    CPU_KERNEL_REGISTRATIONS
}

fn supports_decode_qmatmul(shape: &ShapeClass) -> bool {
    shape.batch == BatchClass::Decode
}

fn supports_prefill_qmatmul(shape: &ShapeClass) -> bool {
    matches!(shape.batch, BatchClass::SmallBatch | BatchClass::Prefill)
}

#[cfg(test)]
mod tests {
    use super::*;
    use augur_kernel_api::BackendKind;
    use std::collections::HashSet;

    #[test]
    fn production_cpu_registry_contains_only_cpu_keys() {
        for registration in cpu_kernel_registrations() {
            assert_eq!(
                registration.key.backend,
                BackendKind::Cpu,
                "CPU registration '{}' carries a non-CPU KernelKey",
                registration.id.0,
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn production_registry_has_thirteen_unique_registration_ids() {
        let registrations = cpu_kernel_registrations();

        assert_eq!(
            registrations.len(),
            13,
            "M50 production CPU registration count"
        );

        let unique_ids: HashSet<_> = registrations
            .iter()
            .map(|registration| registration.id)
            .collect();

        assert_eq!(
            unique_ids.len(),
            registrations.len(),
            "production CPU registration IDs must remain unique"
        );

        let expected_ids = [
            "cpu.avx2.dual_scale_q8hot.qmatmul.identity.v1",
            "cpu.avx2_fma.q4k_f32.qmatmul.identity.v1",
            "cpu.portable.q4k_f32.prefill.identity.v1",
            "cpu.portable.q4k_f32.qmatmul.identity.v1",
            "cpu.portable.q4k_f32.qmatmul.bias.v1",
            "cpu.portable.q4k_f32.qmatmul.bias_silu.v1",
            "cpu.portable.dual_scale_q8hot.qmatmul.identity.v1",
            "cpu.portable.q8_0_q8hot.qmatmul.identity.v1",
            "cpu.portable.q2k_q8hot.qmatmul.identity.v1",
            "cpu.portable.iq2_xxs_q8hot.qmatmul.identity.v1",
            "cpu.portable.mxfp4_f32.qmatmul.identity.v1",
            "cpu.portable.fp8_e4m3_scale_grid_f32.qmatmul.identity.v1",
            "cpu.portable.fp8_e4m3_scale_grid_f32.fused_mlp.clamp_silu_mul.v1",
        ];

        for expected_id in expected_ids {
            assert!(
                registrations
                    .iter()
                    .any(|registration| registration.id.0 == expected_id),
                "missing production registration '{expected_id}'"
            );
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[test]
    fn production_registry_ids_are_unique() {
        let registrations = cpu_kernel_registrations();

        let unique_ids: HashSet<_> = registrations
            .iter()
            .map(|registration| registration.id)
            .collect();

        assert_eq!(
            unique_ids.len(),
            registrations.len(),
            "production CPU registration IDs must remain unique"
        );
    }
}
