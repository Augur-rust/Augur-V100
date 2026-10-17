use augur_backend::BackendCapabilities;
use augur_core::DeviceKind;
use augur_kernel_api::{
    BackendKind, CpuFeatureSet, CpuKernelAbi, CpuKernelRegistration, KernelKey, ShapeClass,
};

use crate::{
    CpuKernelSelectionSource, CpuTuningRequestKey, KernelSelectionFailure, SelectedCpuKernel,
    ValidatedCpuTuningProfile,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuKernelRequest {
    pub key: KernelKey,
    pub shape: ShapeClass,
    pub expected_abi: CpuKernelAbi,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelSelectionError {
    pub request: CpuKernelRequest,
    pub failure: KernelSelectionFailure,
}

/// Diagnostic state collected while scanning one exact CPU kernel request.
///
/// This type is compiler-private. It records enough information to preserve
/// the existing structured error precedence without allocating an eligible
/// candidate list.
///
/// Candidate callbacks are emitted only after all of the following pass:
///
/// 1. exact [`KernelKey`];
/// 2. required CPU features;
/// 3. shape predicate;
/// 4. launch ABI.
///
/// Eligible callbacks are emitted in the same order as `registrations`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CpuCandidateScan {
    saw_key: bool,
    saw_capability: bool,
    saw_shape: bool,

    first_missing_features: Option<CpuFeatureSet>,
    first_actual_abi: Option<CpuKernelAbi>,

    eligible_count: usize,
}

impl CpuCandidateScan {
    /// Error construction with the actual available feature set retained for
    /// `MissingCpuFeatures` diagnostics.
    fn no_selection_error_with_available(
        self,
        request: CpuKernelRequest,
        available_features: CpuFeatureSet,
    ) -> KernelSelectionError {
        if !self.saw_key {
            return KernelSelectionFailure::NoExactKey.request(request);
        }

        if !self.saw_capability {
            return KernelSelectionFailure::MissingCpuFeatures {
                available: available_features,
                required: self.first_missing_features.unwrap_or(CpuFeatureSet::NONE),
            }
            .request(request);
        }

        if !self.saw_shape {
            return KernelSelectionFailure::UnsupportedShape.request(request);
        }

        if let Some(actual) = self.first_actual_abi {
            return KernelSelectionFailure::LaunchAbiMismatch {
                expected: request.expected_abi,
                actual,
            }
            .request(request);
        }

        KernelSelectionFailure::NoExactKey.request(request)
    }
}

/// Scans all registrations eligible for one exact CPU request.
///
/// The callback contract is frozen for M29:
///
/// - `on_eligible` is called exactly once for every eligible registration;
/// - calls occur in the same order as the `registrations` slice;
/// - no sorting, priority grouping, deduplication or reordering occurs;
/// - different `KernelKey` values never enter the same candidate scan;
/// - capability, shape and ABI-incompatible registrations are not emitted;
/// - the scan performs no heap allocation.
///
/// Preserving registry order is important even though static selection is
/// priority-based. The first registration observed at a priority is the
/// provisional winner, and a later registration at the same highest priority
/// establishes `AmbiguousHighestPriority`.
///
/// M29-B may reuse this scan for validated tuning-profile selection. It must
/// not change the ordering contract.
///
/// Backend mismatch is reported before the registration table is inspected.
fn scan_eligible_cpu_kernels(
    registrations: &'static [CpuKernelRegistration],
    request: CpuKernelRequest,
    capabilities: &BackendCapabilities,
    mut on_eligible: impl FnMut(&'static CpuKernelRegistration),
) -> Result<CpuCandidateScan, KernelSelectionError> {
    if capabilities.backend != DeviceKind::Cpu {
        return Err(KernelSelectionFailure::BackendMismatch {
            actual: capabilities.backend,
        }
        .request(request));
    }

    if request.key.backend != BackendKind::Cpu {
        return Err(KernelSelectionFailure::RequestBackendMismatch {
            actual: request.key.backend,
        }
        .request(request));
    }

    let mut scan = CpuCandidateScan::default();

    for registration in registrations {
        if registration.key != request.key {
            continue;
        }

        scan.saw_key = true;

        if !capabilities.supports_cpu_features(registration.required_features) {
            scan.first_missing_features
                .get_or_insert(registration.required_features);
            continue;
        }

        scan.saw_capability = true;

        if !(registration.shape_support)(&request.shape) {
            continue;
        }

        scan.saw_shape = true;

        let actual_abi = registration.launch.abi();

        if actual_abi != request.expected_abi {
            scan.first_actual_abi.get_or_insert(actual_abi);
            continue;
        }

        scan.eligible_count = scan.eligible_count.saturating_add(1);

        // M29-A ordering contract:
        //
        // callback order is exactly registry slice order.
        on_eligible(registration);
    }

    Ok(scan)
}

/// Public wrapper that returns the exact eligible candidate count.
///
/// This is the tooling-facing API for M29-C and future offline harnesses.
/// It exposes only the count and the registry-order callback; internal
/// diagnostic state remains private.
///
/// Return values:
///
/// - `Ok(0)` — no eligible candidates (error path, caller inspects the error);
/// - `Ok(1)` — one candidate, skip benchmark;
/// - `Ok(>=2)` — multiple candidates, autotune is meaningful.
pub fn for_each_eligible_cpu_kernel(
    registrations: &'static [CpuKernelRegistration],
    request: CpuKernelRequest,
    capabilities: &BackendCapabilities,
    on_eligible: impl FnMut(&'static CpuKernelRegistration),
) -> Result<usize, KernelSelectionError> {
    let scan = scan_eligible_cpu_kernels(registrations, request, capabilities, on_eligible)?;
    Ok(scan.eligible_count)
}

/// Select one exact static CPU kernel registration.
///
/// Selection is compile/lowering-only and performs no allocation.
///
/// Selection order:
///
/// 1. exact KernelKey;
/// 2. backend and required CPU features;
/// 3. shape support;
/// 4. launch ABI;
/// 5. highest priority;
/// 6. equal highest priority is an error.
///
/// M29-A implementation note:
///
/// Eligible registrations are obtained through `scan_eligible_cpu_kernels`.
/// The scanner preserves registry order and the static selector applies
/// priority policy without constructing an intermediate candidate collection.
pub fn select_cpu_kernel(
    registrations: &'static [CpuKernelRegistration],
    request: CpuKernelRequest,
    capabilities: &BackendCapabilities,
) -> Result<&'static CpuKernelRegistration, KernelSelectionError> {
    let mut best: Option<&'static CpuKernelRegistration> = None;

    // This records ambiguity only at the current highest observed priority.
    //
    // A later, strictly higher-priority registration replaces the provisional
    // winner and clears ambiguity at the lower priority.
    let mut ambiguous_priority: Option<u16> = None;

    let scan =
        scan_eligible_cpu_kernels(
            registrations,
            request,
            capabilities,
            |registration| match best {
                None => {
                    best = Some(registration);
                    ambiguous_priority = None;
                }

                Some(current) if registration.priority > current.priority => {
                    best = Some(registration);
                    ambiguous_priority = None;
                }

                Some(current) if registration.priority == current.priority => {
                    ambiguous_priority = Some(registration.priority);
                }

                Some(_) => {}
            },
        )?;

    if let Some(priority) = ambiguous_priority {
        return Err(KernelSelectionFailure::AmbiguousHighestPriority { priority }.request(request));
    }

    if let Some(selected) = best {
        return Ok(selected);
    }

    Err(scan.no_selection_error_with_available(request, capabilities.cpu_features))
}

/// Ordered candidate selection with controlled fallback.
///
/// Tries `first` then each `fallbacks` request in order and returns the first
/// successful selection. The API guarantees at least one candidate through the
/// `first + &[]` split, avoiding the need for an empty-list error variant.
///
/// Fallthrough is allowed only when the current candidate is unavailable:
///
/// - `NoExactKey`;
/// - `MissingCpuFeatures`;
/// - `UnsupportedShape`.
///
/// Registry/backend invariant failures are fatal and are never hidden:
///
/// - `LaunchAbiMismatch`;
/// - `AmbiguousHighestPriority`;
/// - `BackendMismatch`.
///
/// If all candidates fail non-fatally, this function returns the final
/// candidate's error. In an AVX2 -> Portable chain, that error explains why
/// the portable terminal fallback was also unavailable.
///
/// M29-A does not reinterpret architecture fallback as autotune competition.
/// Each request is scanned independently by exact `KernelKey`.
pub fn select_cpu_kernel_with_fallback(
    registrations: &'static [CpuKernelRegistration],
    first: CpuKernelRequest,
    fallbacks: &[CpuKernelRequest],
    capabilities: &BackendCapabilities,
) -> Result<&'static CpuKernelRegistration, KernelSelectionError> {
    let mut last_non_fatal_error = match select_cpu_kernel(registrations, first, capabilities) {
        Ok(selected) => return Ok(selected),

        Err(error) => {
            if is_fatal_fallback_error(error.failure) {
                return Err(error);
            }

            error
        }
    };

    for request in fallbacks.iter().copied() {
        match select_cpu_kernel(registrations, request, capabilities) {
            Ok(selected) => return Ok(selected),

            Err(error) => {
                if is_fatal_fallback_error(error.failure) {
                    return Err(error);
                }

                last_non_fatal_error = error;
            }
        }
    }

    Err(last_non_fatal_error)
}

/// Selects one exact CPU kernel request with an optional validated tuning
/// profile.
///
/// Tuning is considered only when at least two registrations are eligible
/// for the same exact request.
///
/// A valid profile winner may select any eligible same-key registration and
/// may resolve an equal-highest-priority static ambiguity.
///
/// With zero or one eligible candidate, the result preserves static-selection
/// semantics and reports `StaticPriority`.
pub fn select_cpu_kernel_with_tuning(
    registrations: &'static [CpuKernelRegistration],
    request: CpuKernelRequest,
    tuning_key: CpuTuningRequestKey,
    capabilities: &BackendCapabilities,
    profile: Option<&ValidatedCpuTuningProfile>,
) -> Result<SelectedCpuKernel, KernelSelectionError> {
    debug_assert_eq!(tuning_key.kernel, request.key);
    debug_assert_eq!(tuning_key.expected_abi, request.expected_abi);

    let tuned_registration = profile.and_then(|p| p.lookup(tuning_key));

    let mut best: Option<&'static CpuKernelRegistration> = None;
    let mut ambiguous_priority: Option<u16> = None;
    let mut eligible_tuned_registration: Option<&'static CpuKernelRegistration> = None;

    let scan = scan_eligible_cpu_kernels(registrations, request, capabilities, |registration| {
        if tuned_registration.is_some_and(|tuned| core::ptr::eq(tuned, registration)) {
            eligible_tuned_registration = Some(registration);
        }

        match best {
            None => {
                best = Some(registration);
                ambiguous_priority = None;
            }
            Some(current) if registration.priority > current.priority => {
                best = Some(registration);
                ambiguous_priority = None;
            }
            Some(current) if registration.priority == current.priority => {
                ambiguous_priority = Some(registration.priority);
            }
            Some(_) => {}
        }
    })?;

    // Tuned profile may resolve ambiguity only when ≥2 eligible candidates
    if scan.eligible_count >= 2 {
        if let Some(registration) = eligible_tuned_registration {
            return Ok(SelectedCpuKernel {
                registration,
                source: CpuKernelSelectionSource::TunedProfile,
            });
        }
    }

    if let Some(priority) = ambiguous_priority {
        return Err(KernelSelectionFailure::AmbiguousHighestPriority { priority }.request(request));
    }

    if let Some(registration) = best {
        return Ok(SelectedCpuKernel {
            registration,
            source: CpuKernelSelectionSource::StaticPriority,
        });
    }

    Err(scan.no_selection_error_with_available(request, capabilities.cpu_features))
}

/// Ordered candidate selection with controlled fallback and optional tuning.
pub fn select_cpu_kernel_with_fallback_and_tuning(
    registrations: &'static [CpuKernelRegistration],
    first_request: CpuKernelRequest,
    first_tuning_key: CpuTuningRequestKey,
    fallbacks: &[(CpuKernelRequest, CpuTuningRequestKey)],
    capabilities: &BackendCapabilities,
    profile: Option<&ValidatedCpuTuningProfile>,
) -> Result<SelectedCpuKernel, KernelSelectionError> {
    let mut last_non_fatal_error = match select_cpu_kernel_with_tuning(
        registrations,
        first_request,
        first_tuning_key,
        capabilities,
        profile,
    ) {
        Ok(selected) => return Ok(selected),
        Err(error) => {
            if is_fatal_fallback_error(error.failure) {
                return Err(error);
            }
            error
        }
    };

    for (request, tuning_key) in fallbacks.iter().copied() {
        match select_cpu_kernel_with_tuning(
            registrations,
            request,
            tuning_key,
            capabilities,
            profile,
        ) {
            Ok(selected) => return Ok(selected),
            Err(error) => {
                if is_fatal_fallback_error(error.failure) {
                    return Err(error);
                }
                last_non_fatal_error = error;
            }
        }
    }

    Err(last_non_fatal_error)
}

fn is_fatal_fallback_error(failure: KernelSelectionFailure) -> bool {
    matches!(
        failure,
        KernelSelectionFailure::LaunchAbiMismatch { .. }
            | KernelSelectionFailure::AmbiguousHighestPriority { .. }
            | KernelSelectionFailure::BackendMismatch { .. }
            | KernelSelectionFailure::RequestBackendMismatch { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use augur_core::DeviceKind;
    use augur_kernel_api::{
        ArchKind, BatchClass, CpuKernelLaunch, CpuShapeSupportId, EpilogueKind, KernelLaunchError,
        KernelRegistrationId, Q4KHotCpuTensor,
    };

    const TEST_DECODE_SHAPE: CpuShapeSupportId = CpuShapeSupportId("test.shape.decode.v1");
    const TEST_PREFILL_SHAPE: CpuShapeSupportId = CpuShapeSupportId("test.shape.prefill.v1");

    fn supports_decode(shape: &ShapeClass) -> bool {
        shape.batch == BatchClass::Decode
    }

    fn supports_prefill(shape: &ShapeClass) -> bool {
        shape.batch == BatchClass::Prefill
    }

    fn dummy_portable(
        _weight: &Q4KHotCpuTensor,
        _input: &[f32],
        _output: &mut [f32],
    ) -> Result<(), KernelLaunchError> {
        Ok(())
    }

    fn dummy_avx2(
        _weight: &Q4KHotCpuTensor,
        _input: &[f32],
        _output: &mut [f32],
    ) -> Result<(), KernelLaunchError> {
        Ok(())
    }

    fn dummy_bias(
        _weight: &Q4KHotCpuTensor,
        _input: &[f32],
        _bias: &[f32],
        _output: &mut [f32],
    ) -> Result<(), KernelLaunchError> {
        Ok(())
    }

    const PORTABLE: CpuKernelRegistration = CpuKernelRegistration {
        id: KernelRegistrationId("test.portable"),
        key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
        shape_support_id: TEST_DECODE_SHAPE,
        shape_support: supports_decode,
        priority: 100,
        required_features: CpuFeatureSet::NONE,
        launch: CpuKernelLaunch::QMatmulF32(dummy_portable),
    };

    const AVX2: CpuKernelRegistration = CpuKernelRegistration {
        id: KernelRegistrationId("test.avx2"),
        key: KernelKey::q4k_f32_cpu_avx2_identity(),
        shape_support_id: TEST_DECODE_SHAPE,
        shape_support: supports_decode,
        priority: 200,
        required_features: CpuFeatureSet::AVX2.union(CpuFeatureSet::FMA),
        launch: CpuKernelLaunch::QMatmulF32(dummy_avx2),
    };

    static REGS: &[CpuKernelRegistration] = &[AVX2, PORTABLE];

    fn request(key: KernelKey) -> CpuKernelRequest {
        CpuKernelRequest {
            key,
            shape: ShapeClass::DECODE_GEMV,
            expected_abi: CpuKernelAbi::QMatmulF32,
        }
    }

    fn capabilities(features: CpuFeatureSet) -> BackendCapabilities {
        BackendCapabilities {
            backend: DeviceKind::Cpu,
            cpu_features: features,
        }
    }

    fn select_with_arch_fallback(features: CpuFeatureSet) -> &'static CpuKernelRegistration {
        select_cpu_kernel_with_fallback(
            REGS,
            request(KernelKey::q4k_f32_cpu_avx2_identity()),
            &[request(KernelKey::q4k_f32_cpu_portable(
                EpilogueKind::Identity,
            ))],
            &capabilities(features),
        )
        .unwrap()
    }

    #[test]
    fn avx2_and_fma_select_avx2() {
        assert_eq!(
            select_with_arch_fallback(CpuFeatureSet::AVX2.union(CpuFeatureSet::FMA)).id,
            KernelRegistrationId("test.avx2"),
        );
    }

    #[test]
    fn avx2_without_fma_falls_back() {
        assert_eq!(
            select_with_arch_fallback(CpuFeatureSet::AVX2).id,
            KernelRegistrationId("test.portable"),
        );
    }

    #[test]
    fn no_features_fall_back_to_portable() {
        assert_eq!(
            select_with_arch_fallback(CpuFeatureSet::NONE).id,
            KernelRegistrationId("test.portable"),
        );
    }

    #[test]
    fn terminal_fallback_error_is_returned() {
        static AVX2_ONLY: &[CpuKernelRegistration] = &[AVX2];

        let error = select_cpu_kernel_with_fallback(
            AVX2_ONLY,
            request(KernelKey::q4k_f32_cpu_avx2_identity()),
            &[request(KernelKey::q4k_f32_cpu_portable(
                EpilogueKind::Identity,
            ))],
            &capabilities(CpuFeatureSet::NONE),
        )
        .unwrap_err();

        assert_eq!(error.failure, KernelSelectionFailure::NoExactKey);
    }

    #[test]
    fn fatal_abi_error_does_not_fallback() {
        const WRONG: CpuKernelRegistration = CpuKernelRegistration {
            id: KernelRegistrationId("test.wrong_abi"),
            key: KernelKey::q4k_f32_cpu_avx2_identity(),
            shape_support_id: TEST_DECODE_SHAPE,
            shape_support: supports_decode,
            priority: 200,
            required_features: CpuFeatureSet::AVX2.union(CpuFeatureSet::FMA),
            launch: CpuKernelLaunch::QMatmulBiasF32(dummy_bias),
        };

        static WRONG_REGS: &[CpuKernelRegistration] = &[WRONG, PORTABLE];

        let error = select_cpu_kernel_with_fallback(
            WRONG_REGS,
            request(KernelKey::q4k_f32_cpu_avx2_identity()),
            &[request(KernelKey::q4k_f32_cpu_portable(
                EpilogueKind::Identity,
            ))],
            &capabilities(CpuFeatureSet::AVX2.union(CpuFeatureSet::FMA)),
        )
        .unwrap_err();

        assert_eq!(
            error.failure,
            KernelSelectionFailure::LaunchAbiMismatch {
                expected: CpuKernelAbi::QMatmulF32,
                actual: CpuKernelAbi::QMatmulBiasF32,
            },
        );
    }

    // ── M29-A candidate scan contract ────────────────────────────────────

    const ORDER_FIRST: CpuKernelRegistration = CpuKernelRegistration {
        id: KernelRegistrationId("test.order.first"),
        key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
        shape_support_id: TEST_DECODE_SHAPE,
        shape_support: supports_decode,
        priority: 100,
        required_features: CpuFeatureSet::NONE,
        launch: CpuKernelLaunch::QMatmulF32(dummy_portable),
    };

    const ORDER_SECOND: CpuKernelRegistration = CpuKernelRegistration {
        id: KernelRegistrationId("test.order.second"),
        key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
        shape_support_id: TEST_DECODE_SHAPE,
        shape_support: supports_decode,
        priority: 300,
        required_features: CpuFeatureSet::NONE,
        launch: CpuKernelLaunch::QMatmulF32(dummy_portable),
    };

    const ORDER_THIRD: CpuKernelRegistration = CpuKernelRegistration {
        id: KernelRegistrationId("test.order.third"),
        key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
        shape_support_id: TEST_DECODE_SHAPE,
        shape_support: supports_decode,
        priority: 200,
        required_features: CpuFeatureSet::NONE,
        launch: CpuKernelLaunch::QMatmulF32(dummy_portable),
    };

    #[test]
    fn eligible_callback_order_equals_registry_order() {
        static TABLE: &[CpuKernelRegistration] = &[ORDER_THIRD, ORDER_FIRST, ORDER_SECOND];

        let mut observed = Vec::new();

        let scan = scan_eligible_cpu_kernels(
            TABLE,
            request(KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity)),
            &capabilities(CpuFeatureSet::NONE),
            |registration| observed.push(registration.id),
        )
        .unwrap();

        assert_eq!(
            observed,
            vec![
                KernelRegistrationId("test.order.third"),
                KernelRegistrationId("test.order.first"),
                KernelRegistrationId("test.order.second"),
            ],
        );

        assert_eq!(scan.eligible_count, 3);
    }

    #[test]
    fn static_priority_is_independent_of_registry_order() {
        static FIRST_ORDER: &[CpuKernelRegistration] = &[ORDER_FIRST, ORDER_SECOND, ORDER_THIRD];

        static SECOND_ORDER: &[CpuKernelRegistration] = &[ORDER_THIRD, ORDER_FIRST, ORDER_SECOND];

        let req = request(KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity));

        let first =
            select_cpu_kernel(FIRST_ORDER, req, &capabilities(CpuFeatureSet::NONE)).unwrap();

        let second =
            select_cpu_kernel(SECOND_ORDER, req, &capabilities(CpuFeatureSet::NONE)).unwrap();

        assert_eq!(first.id, KernelRegistrationId("test.order.second"));
        assert_eq!(second.id, KernelRegistrationId("test.order.second"));
    }

    #[test]
    fn same_highest_priority_remains_ambiguous() {
        const TIED_FIRST: CpuKernelRegistration = CpuKernelRegistration {
            id: KernelRegistrationId("test.tie.first"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_DECODE_SHAPE,
            shape_support: supports_decode,
            priority: 300,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_portable),
        };

        const TIED_SECOND: CpuKernelRegistration = CpuKernelRegistration {
            id: KernelRegistrationId("test.tie.second"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_DECODE_SHAPE,
            shape_support: supports_decode,
            priority: 300,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_portable),
        };

        static TABLE: &[CpuKernelRegistration] =
            &[ORDER_FIRST, TIED_FIRST, ORDER_THIRD, TIED_SECOND];

        let error = select_cpu_kernel(
            TABLE,
            request(KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity)),
            &capabilities(CpuFeatureSet::NONE),
        )
        .unwrap_err();

        assert_eq!(
            error.failure,
            KernelSelectionFailure::AmbiguousHighestPriority { priority: 300 },
        );
    }

    #[test]
    fn lower_priority_tie_is_cleared_by_unique_higher_priority() {
        const LOW_TIE_FIRST: CpuKernelRegistration = CpuKernelRegistration {
            id: KernelRegistrationId("test.low_tie.first"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_DECODE_SHAPE,
            shape_support: supports_decode,
            priority: 100,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_portable),
        };

        const LOW_TIE_SECOND: CpuKernelRegistration = CpuKernelRegistration {
            id: KernelRegistrationId("test.low_tie.second"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_DECODE_SHAPE,
            shape_support: supports_decode,
            priority: 100,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_portable),
        };

        const UNIQUE_HIGH: CpuKernelRegistration = CpuKernelRegistration {
            id: KernelRegistrationId("test.unique_high"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_DECODE_SHAPE,
            shape_support: supports_decode,
            priority: 200,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_portable),
        };

        static TABLE: &[CpuKernelRegistration] = &[LOW_TIE_FIRST, LOW_TIE_SECOND, UNIQUE_HIGH];

        let selected = select_cpu_kernel(
            TABLE,
            request(KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity)),
            &capabilities(CpuFeatureSet::NONE),
        )
        .unwrap();

        assert_eq!(selected.id, KernelRegistrationId("test.unique_high"),);
    }

    #[test]
    fn capability_filtering_occurs_before_priority() {
        const UNAVAILABLE_HIGH: CpuKernelRegistration = CpuKernelRegistration {
            id: KernelRegistrationId("test.unavailable_high"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_DECODE_SHAPE,
            shape_support: supports_decode,
            priority: 500,
            required_features: CpuFeatureSet::AVX512F,
            launch: CpuKernelLaunch::QMatmulF32(dummy_portable),
        };

        const AVAILABLE_LOW: CpuKernelRegistration = CpuKernelRegistration {
            id: KernelRegistrationId("test.available_low"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_DECODE_SHAPE,
            shape_support: supports_decode,
            priority: 100,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_portable),
        };

        static TABLE: &[CpuKernelRegistration] = &[UNAVAILABLE_HIGH, AVAILABLE_LOW];

        let selected = select_cpu_kernel(
            TABLE,
            request(KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity)),
            &capabilities(CpuFeatureSet::NONE),
        )
        .unwrap();

        assert_eq!(selected.id, KernelRegistrationId("test.available_low"),);
    }

    #[test]
    fn shape_filtering_occurs_before_priority() {
        const PREFILL_HIGH: CpuKernelRegistration = CpuKernelRegistration {
            id: KernelRegistrationId("test.prefill_high"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_PREFILL_SHAPE,
            shape_support: supports_prefill,
            priority: 500,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_portable),
        };

        const DECODE_LOW: CpuKernelRegistration = CpuKernelRegistration {
            id: KernelRegistrationId("test.decode_low"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_DECODE_SHAPE,
            shape_support: supports_decode,
            priority: 100,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_portable),
        };

        static TABLE: &[CpuKernelRegistration] = &[PREFILL_HIGH, DECODE_LOW];

        let selected = select_cpu_kernel(
            TABLE,
            request(KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity)),
            &capabilities(CpuFeatureSet::NONE),
        )
        .unwrap();

        assert_eq!(selected.id, KernelRegistrationId("test.decode_low"),);
    }

    #[test]
    fn abi_filtering_occurs_before_priority() {
        const WRONG_ABI_HIGH: CpuKernelRegistration = CpuKernelRegistration {
            id: KernelRegistrationId("test.wrong_abi_high"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_DECODE_SHAPE,
            shape_support: supports_decode,
            priority: 500,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulBiasF32(dummy_bias),
        };

        const CORRECT_ABI_LOW: CpuKernelRegistration = CpuKernelRegistration {
            id: KernelRegistrationId("test.correct_abi_low"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: TEST_DECODE_SHAPE,
            shape_support: supports_decode,
            priority: 100,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_portable),
        };

        static TABLE: &[CpuKernelRegistration] = &[WRONG_ABI_HIGH, CORRECT_ABI_LOW];

        let selected = select_cpu_kernel(
            TABLE,
            request(KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity)),
            &capabilities(CpuFeatureSet::NONE),
        )
        .unwrap();

        assert_eq!(selected.id, KernelRegistrationId("test.correct_abi_low"),);
    }

    #[test]
    fn different_kernel_keys_never_share_candidate_scan() {
        static TABLE: &[CpuKernelRegistration] = &[AVX2, PORTABLE];

        let mut observed = Vec::new();

        let scan = scan_eligible_cpu_kernels(
            TABLE,
            request(KernelKey::q4k_f32_cpu_avx2_identity()),
            &capabilities(CpuFeatureSet::AVX2.union(CpuFeatureSet::FMA)),
            |registration| observed.push(registration.id),
        )
        .unwrap();

        assert_eq!(observed, vec![KernelRegistrationId("test.avx2")],);

        assert_eq!(scan.eligible_count, 1);
    }

    #[test]
    fn no_exact_key_preserves_structured_error() {
        static TABLE: &[CpuKernelRegistration] = &[PORTABLE];

        let error = select_cpu_kernel(
            TABLE,
            request(KernelKey::q4k_f32_cpu_avx2_identity()),
            &capabilities(CpuFeatureSet::AVX2.union(CpuFeatureSet::FMA)),
        )
        .unwrap_err();

        assert_eq!(error.failure, KernelSelectionFailure::NoExactKey);
    }

    #[test]
    fn missing_features_preserves_available_feature_diagnostic() {
        static TABLE: &[CpuKernelRegistration] = &[AVX2];

        let error = select_cpu_kernel(
            TABLE,
            request(KernelKey::q4k_f32_cpu_avx2_identity()),
            &capabilities(CpuFeatureSet::AVX2),
        )
        .unwrap_err();

        assert_eq!(
            error.failure,
            KernelSelectionFailure::MissingCpuFeatures {
                available: CpuFeatureSet::AVX2,
                required: CpuFeatureSet::AVX2.union(CpuFeatureSet::FMA),
            },
        );
    }

    #[test]
    fn backend_mismatch_occurs_before_registry_scan() {
        let request = request(KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity));

        let capabilities = BackendCapabilities {
            backend: DeviceKind::Cuda,
            cpu_features: CpuFeatureSet::NONE,
        };

        let mut callback_count = 0;

        let error =
            scan_eligible_cpu_kernels(&[PORTABLE], request, &capabilities, |_| callback_count += 1)
                .unwrap_err();

        assert_eq!(callback_count, 0);

        assert_eq!(
            error.failure,
            KernelSelectionFailure::BackendMismatch {
                actual: DeviceKind::Cuda,
            },
        );
    }

    #[test]
    fn cpu_selector_rejects_non_cpu_request_before_registry_scan() {
        let mut key = KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity);
        key.backend = BackendKind::Cuda;
        key.arch = ArchKind::NvidiaSm;

        let request = CpuKernelRequest {
            key,
            shape: ShapeClass::DECODE_GEMV,
            expected_abi: CpuKernelAbi::QMatmulF32,
        };

        let mut callback_count = 0;

        let error = scan_eligible_cpu_kernels(
            &[PORTABLE],
            request,
            &capabilities(CpuFeatureSet::NONE),
            |_| callback_count += 1,
        )
        .unwrap_err();

        assert_eq!(callback_count, 0);
        assert_eq!(
            error.failure,
            KernelSelectionFailure::RequestBackendMismatch {
                actual: BackendKind::Cuda,
            },
        );
    }

    #[test]
    fn non_cpu_request_does_not_fall_back_to_cpu() {
        let mut cuda_key = KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity);
        cuda_key.backend = BackendKind::Cuda;
        cuda_key.arch = ArchKind::NvidiaSm;

        let error = select_cpu_kernel_with_fallback(
            REGS,
            request(cuda_key),
            &[request(KernelKey::q4k_f32_cpu_portable(
                EpilogueKind::Identity,
            ))],
            &capabilities(CpuFeatureSet::NONE),
        )
        .unwrap_err();

        assert_eq!(
            error.failure,
            KernelSelectionFailure::RequestBackendMismatch {
                actual: BackendKind::Cuda,
            },
        );
    }
}
