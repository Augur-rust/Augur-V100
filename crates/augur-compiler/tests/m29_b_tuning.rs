use std::sync::Arc;

use augur_backend::{BackendCapabilities, CpuIdentity};
use augur_compiler::{
    fingerprint_cpu_registry, normalize_cpu_tuning_shape,
    select_cpu_kernel_with_fallback_and_tuning, select_cpu_kernel_with_tuning,
    validate_cpu_tuning_profile, CpuKernelRequest, CpuKernelSelectionSource, CpuTargetArch,
    CpuTuningEntryInput, CpuTuningEnvironment, CpuTuningProfileError, CpuTuningProfileInput,
    CpuTuningProfileVersion, CpuTuningRequestKey, CpuTuningShape, CpuTuningShapeSchemaVersion,
    RawCpuTuningShape, RegistryFingerprint, ValidatedCpuTuningProfile, CPU_TUNING_PROFILE_V1,
    CPU_TUNING_SHAPE_SCHEMA_V1,
};
use augur_core::DeviceKind;
use augur_kernel_api::{
    BatchClass, CpuFeatureSet, CpuKernelAbi, CpuKernelLaunch, CpuKernelRegistration,
    CpuShapeSupportId, EpilogueKind, KernelKey, KernelLaunchError, KernelRegistrationId,
    Q4KHotCpuTensor, ShapeClass,
};

fn decode_only(shape: &ShapeClass) -> bool {
    shape.batch == BatchClass::Decode
}

fn candidate_a(_w: &Q4KHotCpuTensor, _i: &[f32], _o: &mut [f32]) -> Result<(), KernelLaunchError> {
    Ok(())
}

fn candidate_b(_w: &Q4KHotCpuTensor, _i: &[f32], _o: &mut [f32]) -> Result<(), KernelLaunchError> {
    Ok(())
}

const KEY: KernelKey = KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity);

const SHAPE_ID: CpuShapeSupportId = CpuShapeSupportId("test.shape.decode.v1");

const CANDIDATE_A: CpuKernelRegistration = CpuKernelRegistration {
    id: KernelRegistrationId("test.candidate.a"),
    key: KEY,
    shape_support_id: SHAPE_ID,
    shape_support: decode_only,
    priority: 200,
    required_features: CpuFeatureSet::NONE,
    launch: CpuKernelLaunch::QMatmulF32(candidate_a),
};

const CANDIDATE_B: CpuKernelRegistration = CpuKernelRegistration {
    id: KernelRegistrationId("test.candidate.b"),
    key: KEY,
    shape_support_id: SHAPE_ID,
    shape_support: decode_only,
    priority: 200,
    required_features: CpuFeatureSet::NONE,
    launch: CpuKernelLaunch::QMatmulF32(candidate_b),
};

static TWO_CANDIDATES: &[CpuKernelRegistration] = &[CANDIDATE_A, CANDIDATE_B];

fn capabilities() -> BackendCapabilities {
    BackendCapabilities {
        backend: DeviceKind::Cpu,
        cpu_features: CpuFeatureSet::NONE,
    }
}

fn shape() -> CpuTuningShape {
    normalize_cpu_tuning_shape(RawCpuTuningShape {
        batch: 1,
        m: 1,
        n: 128,
        k: 1024,
    })
    .unwrap()
}

fn request() -> CpuKernelRequest {
    CpuKernelRequest {
        key: KEY,
        shape: ShapeClass::DECODE_GEMV,
        expected_abi: CpuKernelAbi::QMatmulF32,
    }
}

fn tuning_key() -> CpuTuningRequestKey {
    CpuTuningRequestKey {
        kernel: KEY,
        shape: shape(),
        expected_abi: CpuKernelAbi::QMatmulF32,
    }
}

fn x86_env(thread_count: u16) -> CpuTuningEnvironment {
    CpuTuningEnvironment {
        target_arch: CpuTargetArch::X86_64,
        identity: CpuIdentity {
            vendor: *b"GenuineIntel",
            family: 6,
            model: 143,
            stepping: 8,
        },
        available_features: CpuFeatureSet::NONE,
        thread_count,
    }
}

fn profile_selecting(winner_id: &str) -> CpuTuningProfileInput {
    CpuTuningProfileInput {
        version: CPU_TUNING_PROFILE_V1,
        shape_schema: CPU_TUNING_SHAPE_SCHEMA_V1,
        registry_fingerprint: fingerprint_cpu_registry(TWO_CANDIDATES),
        environment: x86_env(1),
        entries: Box::new([CpuTuningEntryInput {
            request: tuning_key(),
            selected_registration_id: winner_id.into(),
        }]),
    }
}

// ── Profile validation ──────────────────────────────────────────────────

#[test]
fn version_mismatch_is_rejected() {
    let base = profile_selecting("test.candidate.a");
    let input = CpuTuningProfileInput {
        version: CpuTuningProfileVersion(99),
        ..base
    };
    let err = validate_cpu_tuning_profile(input, TWO_CANDIDATES, x86_env(1)).unwrap_err();
    assert!(matches!(
        err,
        CpuTuningProfileError::UnsupportedProfileVersion { .. }
    ));
}

#[test]
fn shape_schema_mismatch_is_rejected() {
    let base = profile_selecting("test.candidate.a");
    let input = CpuTuningProfileInput {
        shape_schema: CpuTuningShapeSchemaVersion(99),
        ..base
    };
    let err = validate_cpu_tuning_profile(input, TWO_CANDIDATES, x86_env(1)).unwrap_err();
    assert!(matches!(
        err,
        CpuTuningProfileError::UnsupportedShapeSchema { .. }
    ));
}

#[test]
fn fingerprint_mismatch_is_rejected() {
    let base = profile_selecting("test.candidate.a");
    let input = CpuTuningProfileInput {
        registry_fingerprint: RegistryFingerprint(0),
        ..base
    };
    let err = validate_cpu_tuning_profile(input, TWO_CANDIDATES, x86_env(1)).unwrap_err();
    assert!(matches!(
        err,
        CpuTuningProfileError::RegistryFingerprintMismatch { .. }
    ));
}

#[test]
fn environment_mismatch_is_rejected() {
    let input = profile_selecting("test.candidate.a");
    let different_env = CpuTuningEnvironment {
        identity: CpuIdentity {
            vendor: *b"AuthenticAMD",
            family: 25,
            model: 1,
            stepping: 0,
        },
        ..x86_env(1)
    };
    let err = validate_cpu_tuning_profile(input, TWO_CANDIDATES, different_env).unwrap_err();
    assert!(matches!(
        err,
        CpuTuningProfileError::EnvironmentMismatch { .. }
    ));
}

#[test]
fn duplicate_entry_is_rejected() {
    let entry = CpuTuningEntryInput {
        request: tuning_key(),
        selected_registration_id: "test.candidate.a".into(),
    };
    let input = CpuTuningProfileInput {
        version: CPU_TUNING_PROFILE_V1,
        shape_schema: CPU_TUNING_SHAPE_SCHEMA_V1,
        registry_fingerprint: fingerprint_cpu_registry(TWO_CANDIDATES),
        environment: x86_env(1),
        entries: Box::new([entry.clone(), entry]),
    };
    let err = validate_cpu_tuning_profile(input, TWO_CANDIDATES, x86_env(1)).unwrap_err();
    assert!(matches!(err, CpuTuningProfileError::DuplicateEntry { .. }));
}

#[test]
fn registration_not_found_is_rejected() {
    let input = profile_selecting("test.nonexistent");
    let err = validate_cpu_tuning_profile(input, TWO_CANDIDATES, x86_env(1)).unwrap_err();
    assert!(matches!(
        err,
        CpuTuningProfileError::RegistrationNotFound { .. }
    ));
}

#[test]
fn registration_key_mismatch_is_rejected() {
    // Use a registration from a different exact key
    const WRONG_REG: CpuKernelRegistration = CpuKernelRegistration {
        id: KernelRegistrationId("test.wrong_key"),
        key: KernelKey::q4k_f32_cpu_avx2_identity(),
        shape_support_id: SHAPE_ID,
        shape_support: decode_only,
        priority: 200,
        required_features: CpuFeatureSet::NONE,
        launch: CpuKernelLaunch::QMatmulF32(candidate_a),
    };
    static WRONG_REGS: &[CpuKernelRegistration] = &[WRONG_REG];
    let input = CpuTuningProfileInput {
        registry_fingerprint: fingerprint_cpu_registry(WRONG_REGS),
        ..profile_selecting("test.wrong_key")
    };
    let err = validate_cpu_tuning_profile(input, WRONG_REGS, x86_env(1)).unwrap_err();
    assert!(matches!(
        err,
        CpuTuningProfileError::RegistrationKeyMismatch { .. }
    ));
}

#[test]
fn duplicate_registration_id_is_rejected() {
    const DUP1: CpuKernelRegistration = CpuKernelRegistration {
        id: KernelRegistrationId("test.dup"),
        key: KEY,
        shape_support_id: SHAPE_ID,
        shape_support: decode_only,
        priority: 100,
        required_features: CpuFeatureSet::NONE,
        launch: CpuKernelLaunch::QMatmulF32(candidate_a),
    };
    const DUP2: CpuKernelRegistration = CpuKernelRegistration {
        priority: 200,
        ..DUP1
    };
    static DUP_REGS: &[CpuKernelRegistration] = &[DUP1, DUP2];
    let input = profile_selecting("test.dup");
    let bad_input = CpuTuningProfileInput {
        registry_fingerprint: fingerprint_cpu_registry(DUP_REGS),
        ..input
    };
    let err = validate_cpu_tuning_profile(bad_input, DUP_REGS, x86_env(1)).unwrap_err();
    assert!(matches!(
        err,
        CpuTuningProfileError::DuplicateRegistrationId { .. }
    ));
}

// ── Ambiguity resolution ────────────────────────────────────────────────

#[test]
fn profile_hit_resolves_two_candidate_ambiguity() {
    let input = profile_selecting("test.candidate.a");
    let profile = validate_cpu_tuning_profile(input, TWO_CANDIDATES, x86_env(1)).unwrap();
    let profile = Arc::new(profile);

    let result = select_cpu_kernel_with_tuning(
        TWO_CANDIDATES,
        request(),
        tuning_key(),
        &capabilities(),
        Some(&profile),
    )
    .unwrap();

    assert_eq!(
        result.registration.id,
        KernelRegistrationId("test.candidate.a")
    );
    assert_eq!(result.source, CpuKernelSelectionSource::TunedProfile);
}

#[test]
fn profile_hit_selects_b_when_profile_says_b() {
    let input = profile_selecting("test.candidate.b");
    let profile = validate_cpu_tuning_profile(input, TWO_CANDIDATES, x86_env(1)).unwrap();
    let profile = Arc::new(profile);

    let result = select_cpu_kernel_with_tuning(
        TWO_CANDIDATES,
        request(),
        tuning_key(),
        &capabilities(),
        Some(&profile),
    )
    .unwrap();

    assert_eq!(
        result.registration.id,
        KernelRegistrationId("test.candidate.b")
    );
    assert_eq!(result.source, CpuKernelSelectionSource::TunedProfile);
}

#[test]
fn profile_miss_restores_ambiguity() {
    let input = profile_selecting("test.candidate.a");
    let profile = validate_cpu_tuning_profile(input, TWO_CANDIDATES, x86_env(1)).unwrap();
    let profile = Arc::new(profile);

    // Profile selects candidate A, but request is for a different shape
    let different_shape = CpuTuningRequestKey {
        shape: normalize_cpu_tuning_shape(RawCpuTuningShape {
            batch: 1,
            m: 1,
            n: 256,
            k: 256,
        })
        .unwrap(),
        ..tuning_key()
    };

    let err = select_cpu_kernel_with_tuning(
        TWO_CANDIDATES,
        request(),
        different_shape,
        &capabilities(),
        Some(&profile),
    )
    .unwrap_err();

    assert_eq!(
        err.failure,
        augur_compiler::KernelSelectionFailure::AmbiguousHighestPriority { priority: 200 },
    );
}

#[test]
fn profile_miss_without_profile_restores_ambiguity() {
    let err = select_cpu_kernel_with_tuning(
        TWO_CANDIDATES,
        request(),
        tuning_key(),
        &capabilities(),
        None,
    )
    .unwrap_err();

    assert_eq!(
        err.failure,
        augur_compiler::KernelSelectionFailure::AmbiguousHighestPriority { priority: 200 },
    );
}

// ── Single candidate ────────────────────────────────────────────────────

#[test]
fn single_candidate_always_static_priority_with_profile() {
    static ONE_CANDIDATE: &[CpuKernelRegistration] = &[CANDIDATE_A];
    let input = profile_selecting("test.candidate.a");
    let profile = validate_cpu_tuning_profile(
        CpuTuningProfileInput {
            registry_fingerprint: fingerprint_cpu_registry(ONE_CANDIDATE),
            ..input
        },
        ONE_CANDIDATE,
        x86_env(1),
    )
    .unwrap();
    let profile = Arc::new(profile);

    let result = select_cpu_kernel_with_tuning(
        ONE_CANDIDATE,
        request(),
        tuning_key(),
        &capabilities(),
        Some(&profile),
    )
    .unwrap();

    assert_eq!(
        result.registration.id,
        KernelRegistrationId("test.candidate.a")
    );
    assert_eq!(result.source, CpuKernelSelectionSource::StaticPriority);
}

#[test]
fn single_candidate_always_static_priority_without_profile() {
    static ONE_CANDIDATE: &[CpuKernelRegistration] = &[CANDIDATE_A];

    let result = select_cpu_kernel_with_tuning(
        ONE_CANDIDATE,
        request(),
        tuning_key(),
        &capabilities(),
        None,
    )
    .unwrap();

    assert_eq!(
        result.registration.id,
        KernelRegistrationId("test.candidate.a")
    );
    assert_eq!(result.source, CpuKernelSelectionSource::StaticPriority);
}

// ── Fallback + tuning ───────────────────────────────────────────────────

const AVX2_KEY: KernelKey = KernelKey::q4k_f32_cpu_avx2_identity();

const AVX2_CANDIDATE: CpuKernelRegistration = CpuKernelRegistration {
    id: KernelRegistrationId("test.avx2.tuned"),
    key: AVX2_KEY,
    shape_support_id: SHAPE_ID,
    shape_support: decode_only,
    priority: 200,
    required_features: CpuFeatureSet::AVX2.union(CpuFeatureSet::FMA),
    launch: CpuKernelLaunch::QMatmulF32(candidate_a),
};

static AVX2_AND_PORTABLE: &[CpuKernelRegistration] = &[
    CpuKernelRegistration {
        id: KernelRegistrationId("test.portable.tuned"),
        key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
        shape_support_id: SHAPE_ID,
        shape_support: decode_only,
        priority: 100,
        required_features: CpuFeatureSet::NONE,
        launch: CpuKernelLaunch::QMatmulF32(candidate_b),
    },
    AVX2_CANDIDATE,
];

fn avx2_request() -> CpuKernelRequest {
    CpuKernelRequest {
        key: AVX2_KEY,
        shape: ShapeClass::DECODE_GEMV,
        expected_abi: CpuKernelAbi::QMatmulF32,
    }
}

fn avx2_tuning_key() -> CpuTuningRequestKey {
    CpuTuningRequestKey {
        kernel: AVX2_KEY,
        shape: shape(),
        expected_abi: CpuKernelAbi::QMatmulF32,
    }
}

fn portable_fallback_request() -> CpuKernelRequest {
    CpuKernelRequest {
        key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
        shape: ShapeClass::DECODE_GEMV,
        expected_abi: CpuKernelAbi::QMatmulF32,
    }
}

fn portable_fallback_tuning_key() -> CpuTuningRequestKey {
    CpuTuningRequestKey {
        kernel: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
        shape: shape(),
        expected_abi: CpuKernelAbi::QMatmulF32,
    }
}

fn avx2_profile() -> Arc<ValidatedCpuTuningProfile> {
    let input = CpuTuningProfileInput {
        version: CPU_TUNING_PROFILE_V1,
        shape_schema: CPU_TUNING_SHAPE_SCHEMA_V1,
        registry_fingerprint: fingerprint_cpu_registry(AVX2_AND_PORTABLE),
        environment: CpuTuningEnvironment {
            available_features: CpuFeatureSet::AVX2.union(CpuFeatureSet::FMA),
            ..x86_env(1)
        },
        entries: Box::new([CpuTuningEntryInput {
            request: avx2_tuning_key(),
            selected_registration_id: "test.avx2.tuned".into(),
        }]),
    };
    let profile = validate_cpu_tuning_profile(
        input,
        AVX2_AND_PORTABLE,
        CpuTuningEnvironment {
            available_features: CpuFeatureSet::AVX2.union(CpuFeatureSet::FMA),
            ..x86_env(1)
        },
    )
    .unwrap();
    Arc::new(profile)
}

#[test]
fn tuned_first_candidate_selected_in_fallback_chain() {
    let avx2_caps = BackendCapabilities {
        backend: DeviceKind::Cpu,
        cpu_features: CpuFeatureSet::AVX2.union(CpuFeatureSet::FMA),
    };

    let profile = avx2_profile();

    let result = select_cpu_kernel_with_fallback_and_tuning(
        AVX2_AND_PORTABLE,
        avx2_request(),
        avx2_tuning_key(),
        &[(portable_fallback_request(), portable_fallback_tuning_key())],
        &avx2_caps,
        Some(&profile),
    )
    .unwrap();

    // Single candidate per exact key — always StaticPriority
    assert_eq!(
        result.registration.id,
        KernelRegistrationId("test.avx2.tuned")
    );
    assert_eq!(result.source, CpuKernelSelectionSource::StaticPriority);
}

#[test]
fn fallback_used_when_first_candidate_unavailable() {
    let portable_caps = BackendCapabilities {
        backend: DeviceKind::Cpu,
        cpu_features: CpuFeatureSet::NONE,
    };

    let result = select_cpu_kernel_with_fallback_and_tuning(
        AVX2_AND_PORTABLE,
        avx2_request(),
        avx2_tuning_key(),
        &[(portable_fallback_request(), portable_fallback_tuning_key())],
        &portable_caps,
        None,
    )
    .unwrap();

    assert_eq!(
        result.registration.id,
        KernelRegistrationId("test.portable.tuned")
    );
    assert_eq!(result.source, CpuKernelSelectionSource::StaticPriority);
}
