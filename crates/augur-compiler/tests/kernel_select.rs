use augur_backend::BackendCapabilities;
use augur_compiler::{select_cpu_kernel, CpuKernelRequest, KernelSelectionFailure};
use augur_core::{DType, DeviceKind, PhysicalLayoutId, QuantFormatId};
use augur_kernel_api::{
    BatchClass, CpuFeatureSet, CpuKernelAbi, CpuKernelLaunch, CpuKernelRegistration,
    CpuShapeSupportId, DualScaleHotCpuTensor, EpilogueKind, KernelKey, KernelLaunchError,
    KernelRegistrationId, Q4KHotCpuTensor, Q8KActivationHotCpuVector, ShapeClass,
};
use augur_kernels::{cpu_kernel_registrations, registrations::Q4K_F32_PORTABLE_IDENTITY};

const TEST_DECODE_SHAPE: CpuShapeSupportId = CpuShapeSupportId("test.shape.decode.v1");
const TEST_PREFILL_SHAPE: CpuShapeSupportId = CpuShapeSupportId("test.shape.prefill.v1");

fn dummy_plain_launch(
    _weight: &Q4KHotCpuTensor,
    _input: &[f32],
    _output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    Ok(())
}

fn dummy_bias_launch(
    _weight: &Q4KHotCpuTensor,
    _input: &[f32],
    _bias: &[f32],
    _output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    Ok(())
}

fn dummy_dual_scale_q8hot_launch(
    _weight: &DualScaleHotCpuTensor,
    _activation: &Q8KActivationHotCpuVector,
    _output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    Ok(())
}

fn decode_only(shape: &ShapeClass) -> bool {
    shape.batch == BatchClass::Decode
}

fn prefill_only(shape: &ShapeClass) -> bool {
    shape.batch == BatchClass::Prefill
}

fn decode_request(epilogue: EpilogueKind, expected_abi: CpuKernelAbi) -> CpuKernelRequest {
    CpuKernelRequest {
        key: KernelKey::q4k_f32_cpu_portable(epilogue),
        shape: ShapeClass::classify_matmul(1, 128, 1024, 1),
        expected_abi,
    }
}

fn dual_scale_decode_request() -> CpuKernelRequest {
    CpuKernelRequest {
        key: KernelKey::dual_scale_q8hot_cpu_portable(),

        shape: ShapeClass::classify_matmul(1, 128, 1024, 1),

        expected_abi: CpuKernelAbi::DualScaleQ8Hot,
    }
}

#[test]
fn selects_all_three_m25_q4k_registrations() {
    let capabilities = BackendCapabilities::cpu_baseline();

    for (epilogue, expected_abi) in [
        (EpilogueKind::Identity, CpuKernelAbi::QMatmulF32),
        (EpilogueKind::Bias, CpuKernelAbi::QMatmulBiasF32),
        (EpilogueKind::BiasSilu, CpuKernelAbi::QMatmulBiasF32),
    ] {
        let request = decode_request(epilogue, expected_abi);

        let selected =
            select_cpu_kernel(cpu_kernel_registrations(), request, &capabilities).unwrap();

        assert_eq!(selected.key, request.key);
        assert_eq!(selected.key.epilogue, epilogue);
        assert_eq!(selected.launch.abi(), expected_abi);
    }
}

#[test]
fn rejects_non_exact_weight_layout() {
    let mut request = decode_request(EpilogueKind::Identity, CpuKernelAbi::QMatmulF32);

    request.key.weight_layout = PhysicalLayoutId::Q4K_HOT_AVX2_V1;

    let error = select_cpu_kernel(
        cpu_kernel_registrations(),
        request,
        &BackendCapabilities::cpu_baseline(),
    )
    .unwrap_err();

    assert_eq!(error.failure, KernelSelectionFailure::NoExactKey);
}

#[test]
fn rejects_non_exact_activation_layout() {
    let mut request = decode_request(EpilogueKind::Identity, CpuKernelAbi::QMatmulF32);

    request.key.activation_layout = PhysicalLayoutId::Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1;

    let error = select_cpu_kernel(
        cpu_kernel_registrations(),
        request,
        &BackendCapabilities::cpu_baseline(),
    )
    .unwrap_err();

    assert_eq!(error.failure, KernelSelectionFailure::NoExactKey);
}

#[test]
fn rejects_non_exact_quant_format() {
    let mut request = decode_request(EpilogueKind::Identity, CpuKernelAbi::QMatmulF32);

    request.key.weight_format = Some(QuantFormatId::DualScale);

    let error = select_cpu_kernel(
        cpu_kernel_registrations(),
        request,
        &BackendCapabilities::cpu_baseline(),
    )
    .unwrap_err();

    assert_eq!(error.failure, KernelSelectionFailure::NoExactKey);
}

#[test]
fn rejects_prefill_shape_for_decode_only_kernel() {
    let request = CpuKernelRequest {
        key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
        shape: ShapeClass::classify_matmul(128, 128, 1024, 128),
        expected_abi: CpuKernelAbi::QMatmulF32,
    };

    let error = select_cpu_kernel(
        &[Q4K_F32_PORTABLE_IDENTITY],
        request,
        &BackendCapabilities::cpu_baseline(),
    )
    .unwrap_err();

    assert_eq!(error.failure, KernelSelectionFailure::UnsupportedShape);
}

#[test]
fn selects_explicit_prefill_abi_for_rank_two_workload_shape() {
    let request = CpuKernelRequest {
        key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
        shape: ShapeClass::classify_prefill(1, 128, 1024),
        expected_abi: CpuKernelAbi::QMatmulF32Prefill,
    };
    let selected = select_cpu_kernel(
        cpu_kernel_registrations(),
        request,
        &BackendCapabilities::cpu_baseline(),
    )
    .unwrap();
    assert_eq!(selected.launch.abi(), CpuKernelAbi::QMatmulF32Prefill);
}

#[test]
fn reports_launch_abi_mismatch() {
    let request = decode_request(EpilogueKind::Identity, CpuKernelAbi::QMatmulBiasF32);

    let error = select_cpu_kernel(
        cpu_kernel_registrations(),
        request,
        &BackendCapabilities::cpu_baseline(),
    )
    .unwrap_err();

    assert_eq!(
        error.failure,
        KernelSelectionFailure::LaunchAbiMismatch {
            expected: CpuKernelAbi::QMatmulBiasF32,
            actual: CpuKernelAbi::QMatmulF32,
        }
    );
}

#[test]
fn reports_missing_cpu_features_from_synthetic_table() {
    const TABLE: &[CpuKernelRegistration] = &[CpuKernelRegistration {
        id: KernelRegistrationId("test.requires_avx2"),
        key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
        shape_support_id: TEST_DECODE_SHAPE,
        shape_support: decode_only,
        priority: 100,
        required_features: CpuFeatureSet::AVX2,
        launch: CpuKernelLaunch::QMatmulF32(dummy_plain_launch),
    }];

    let request = decode_request(EpilogueKind::Identity, CpuKernelAbi::QMatmulF32);

    let error =
        select_cpu_kernel(TABLE, request, &BackendCapabilities::cpu_baseline()).unwrap_err();

    assert_eq!(
        error.failure,
        KernelSelectionFailure::MissingCpuFeatures {
            available: CpuFeatureSet::NONE,
            required: CpuFeatureSet::AVX2,
        }
    );
}

#[test]
fn reports_backend_mismatch() {
    let request = decode_request(EpilogueKind::Identity, CpuKernelAbi::QMatmulF32);

    let capabilities = BackendCapabilities {
        backend: DeviceKind::Cuda,
        cpu_features: CpuFeatureSet::NONE,
    };

    let error = select_cpu_kernel(cpu_kernel_registrations(), request, &capabilities).unwrap_err();

    assert_eq!(
        error.failure,
        KernelSelectionFailure::BackendMismatch {
            actual: DeviceKind::Cuda,
        }
    );
}

#[test]
fn reports_ambiguous_highest_priority() {
    const TABLE: &[CpuKernelRegistration] = &[
        CpuKernelRegistration {
            id: KernelRegistrationId("test.first"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_DECODE_SHAPE,
            shape_support: decode_only,
            priority: 100,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_plain_launch),
        },
        CpuKernelRegistration {
            id: KernelRegistrationId("test.second"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_DECODE_SHAPE,
            shape_support: decode_only,
            priority: 100,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_plain_launch),
        },
    ];

    let request = decode_request(EpilogueKind::Identity, CpuKernelAbi::QMatmulF32);

    let error =
        select_cpu_kernel(TABLE, request, &BackendCapabilities::cpu_baseline()).unwrap_err();

    assert_eq!(
        error.failure,
        KernelSelectionFailure::AmbiguousHighestPriority { priority: 100 }
    );
}

#[test]
fn higher_priority_registration_wins_without_ambiguity() {
    const TABLE: &[CpuKernelRegistration] = &[
        CpuKernelRegistration {
            id: KernelRegistrationId("test.high"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_DECODE_SHAPE,
            shape_support: decode_only,
            priority: 200,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_plain_launch),
        },
        CpuKernelRegistration {
            id: KernelRegistrationId("test.low"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_DECODE_SHAPE,
            shape_support: decode_only,
            priority: 100,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_plain_launch),
        },
    ];

    let selected = select_cpu_kernel(
        TABLE,
        decode_request(EpilogueKind::Identity, CpuKernelAbi::QMatmulF32),
        &BackendCapabilities::cpu_baseline(),
    )
    .unwrap();

    assert_eq!(selected.id, KernelRegistrationId("test.high"));
}

#[test]
fn unsupported_shape_is_filtered_before_priority() {
    const TABLE: &[CpuKernelRegistration] = &[
        CpuKernelRegistration {
            id: KernelRegistrationId("test.prefill_high"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_PREFILL_SHAPE,
            shape_support: prefill_only,
            priority: 200,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_plain_launch),
        },
        CpuKernelRegistration {
            id: KernelRegistrationId("test.decode_low"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_DECODE_SHAPE,
            shape_support: decode_only,
            priority: 100,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_plain_launch),
        },
    ];

    let selected = select_cpu_kernel(
        TABLE,
        decode_request(EpilogueKind::Identity, CpuKernelAbi::QMatmulF32),
        &BackendCapabilities::cpu_baseline(),
    )
    .unwrap();

    assert_eq!(selected.id, KernelRegistrationId("test.decode_low"));
}

#[test]
fn bias_launch_abi_is_representable_by_registry_contract() {
    let launch = CpuKernelLaunch::QMatmulBiasF32(dummy_bias_launch);

    assert_eq!(launch.abi(), CpuKernelAbi::QMatmulBiasF32);
    assert!(launch.qmatmul_f32().is_none());
    assert!(launch.qmatmul_bias_f32().is_some());
}

// ── M26-E1 DualScale Q8Hot selector tests ──────────────────────────────────

#[test]
fn selects_dual_scale_q8hot_registration() {
    let request = dual_scale_decode_request();

    let selected = select_cpu_kernel(
        cpu_kernel_registrations(),
        request,
        &BackendCapabilities::cpu_baseline(),
    )
    .unwrap();

    assert_eq!(selected.key, request.key);

    assert_eq!(selected.launch.abi(), CpuKernelAbi::DualScaleQ8Hot);

    assert!(selected.launch.dual_scale_q8hot().is_some());
}

#[test]
fn dual_scale_launch_typed_extraction_preserves_target() {
    let request = dual_scale_decode_request();

    let selected = select_cpu_kernel(
        cpu_kernel_registrations(),
        request,
        &BackendCapabilities::cpu_baseline(),
    )
    .unwrap();

    let actual = selected.launch.dual_scale_q8hot().unwrap();

    let expected: augur_kernel_api::DualScaleQ8HotCpuLaunch =
        augur_kernels::portable::launch_qmatvec_dual_scale_q8k_hot_cpu_portable;

    let actual_address = actual as *const ();

    let expected_address = expected as *const ();

    assert!(std::ptr::addr_eq(actual_address, expected_address,));
}

#[test]
fn rejects_dual_scale_reference_activation_layout() {
    let mut request = dual_scale_decode_request();

    request.key.activation_layout = PhysicalLayoutId::Q8K_ACTIVATION_REF_VECTOR_V1;

    let error = select_cpu_kernel(
        cpu_kernel_registrations(),
        request,
        &BackendCapabilities::cpu_baseline(),
    )
    .unwrap_err();

    assert_eq!(error.failure, KernelSelectionFailure::NoExactKey);
}

#[test]
fn rejects_dual_scale_f32_activation_key() {
    let mut request = dual_scale_decode_request();

    request.key.activation_dtype = DType::F32;
    request.key.activation_format = None;

    request.key.activation_layout = PhysicalLayoutId::HOST_CONTIGUOUS_F32;

    let error = select_cpu_kernel(
        cpu_kernel_registrations(),
        request,
        &BackendCapabilities::cpu_baseline(),
    )
    .unwrap_err();

    assert_eq!(error.failure, KernelSelectionFailure::NoExactKey);
}

#[test]
fn rejects_dual_scale_compact_weight_layout() {
    let mut request = dual_scale_decode_request();

    request.key.weight_layout = PhysicalLayoutId::DUAL_SCALE_COMPACT_V1;

    let error = select_cpu_kernel(
        cpu_kernel_registrations(),
        request,
        &BackendCapabilities::cpu_baseline(),
    )
    .unwrap_err();

    assert_eq!(error.failure, KernelSelectionFailure::NoExactKey);
}

#[test]
fn rejects_unregistered_dual_scale_bias_epilogue() {
    let mut request = dual_scale_decode_request();

    // The canonical constructor cannot manufacture this unsupported
    // combination. The mutation models malformed/manual selection input and
    // verifies that the registry still rejects it exactly.
    request.key.epilogue = EpilogueKind::Bias;

    let error = select_cpu_kernel(
        cpu_kernel_registrations(),
        request,
        &BackendCapabilities::cpu_baseline(),
    )
    .unwrap_err();

    assert_eq!(error.failure, KernelSelectionFailure::NoExactKey);
}

#[test]
fn rejects_dual_scale_prefill_shape() {
    let request = CpuKernelRequest {
        shape: ShapeClass::classify_matmul(128, 128, 1024, 128),

        ..dual_scale_decode_request()
    };

    let error = select_cpu_kernel(
        cpu_kernel_registrations(),
        request,
        &BackendCapabilities::cpu_baseline(),
    )
    .unwrap_err();

    assert_eq!(error.failure, KernelSelectionFailure::UnsupportedShape);
}

#[test]
fn reports_dual_scale_launch_abi_mismatch() {
    let request = CpuKernelRequest {
        expected_abi: CpuKernelAbi::QMatmulF32,

        ..dual_scale_decode_request()
    };

    let error = select_cpu_kernel(
        cpu_kernel_registrations(),
        request,
        &BackendCapabilities::cpu_baseline(),
    )
    .unwrap_err();

    assert_eq!(
        error.failure,
        KernelSelectionFailure::LaunchAbiMismatch {
            expected: CpuKernelAbi::QMatmulF32,

            actual: CpuKernelAbi::DualScaleQ8Hot,
        }
    );
}

#[test]
fn dual_scale_launch_abi_is_representable() {
    let launch = CpuKernelLaunch::DualScaleQ8Hot(dummy_dual_scale_q8hot_launch);

    assert_eq!(launch.abi(), CpuKernelAbi::DualScaleQ8Hot);

    assert!(launch.dual_scale_q8hot().is_some());

    assert!(launch.qmatmul_f32().is_none());

    assert!(launch.qmatmul_bias_f32().is_none());
}

#[test]
fn dual_scale_key_is_identity_only_by_construction() {
    let key = KernelKey::dual_scale_q8hot_cpu_portable();

    assert_eq!(key.epilogue, EpilogueKind::Identity);

    assert_eq!(key.weight_format, Some(QuantFormatId::DualScale));

    assert_eq!(key.activation_format, Some(QuantFormatId::Q8K));
}
