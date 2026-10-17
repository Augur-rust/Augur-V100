use augur_backend::{detect_cpu_identity, BackendCapabilities, CpuIdentity};
use augur_core::{DType, DeviceKind, QuantFormatId};
use augur_kernel_api::{
    ArchKind, BackendKind, CpuFeatureSet, CpuKernelAbi, CpuKernelRegistration, EpilogueKind,
    KernelKey, OpKind,
};

// ── Versions ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CpuTuningProfileVersion(pub u16);

pub const CPU_TUNING_PROFILE_V1: CpuTuningProfileVersion = CpuTuningProfileVersion(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CpuTuningShapeSchemaVersion(pub u16);

pub const CPU_TUNING_SHAPE_SCHEMA_V1: CpuTuningShapeSchemaVersion = CpuTuningShapeSchemaVersion(1);

const REGISTRY_FINGERPRINT_SCHEMA_V1: u16 = 1;

// ── Shape normalization ──────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RawCpuTuningShape {
    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CpuTuningBatchBucket {
    Decode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CpuTuningDimBucket(u64);

impl CpuTuningDimBucket {
    pub const ONE: Self = Self(1);

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CpuTuningShape {
    pub batch: CpuTuningBatchBucket,
    pub m: CpuTuningDimBucket,
    pub n: CpuTuningDimBucket,
    pub k: CpuTuningDimBucket,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuTuningDimension {
    N,
    K,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuTuningShapeError {
    UnsupportedBatch { actual: usize },
    UnsupportedM { actual: usize },
    ZeroDimension { dimension: CpuTuningDimension },
    DimensionTooLarge { dimension: CpuTuningDimension },
}

pub fn normalize_cpu_tuning_shape(
    raw: RawCpuTuningShape,
) -> Result<CpuTuningShape, CpuTuningShapeError> {
    if raw.batch != 1 {
        return Err(CpuTuningShapeError::UnsupportedBatch { actual: raw.batch });
    }
    if raw.m != 1 {
        return Err(CpuTuningShapeError::UnsupportedM { actual: raw.m });
    }
    Ok(CpuTuningShape {
        batch: CpuTuningBatchBucket::Decode,
        m: CpuTuningDimBucket::ONE,
        n: power_of_two_ceiling(raw.n, CpuTuningDimension::N)?,
        k: power_of_two_ceiling(raw.k, CpuTuningDimension::K)?,
    })
}

fn power_of_two_ceiling(
    value: usize,
    dimension: CpuTuningDimension,
) -> Result<CpuTuningDimBucket, CpuTuningShapeError> {
    if value == 0 {
        return Err(CpuTuningShapeError::ZeroDimension { dimension });
    }
    let value =
        u64::try_from(value).map_err(|_| CpuTuningShapeError::DimensionTooLarge { dimension })?;
    let bucket = value
        .checked_next_power_of_two()
        .ok_or(CpuTuningShapeError::DimensionTooLarge { dimension })?;
    Ok(CpuTuningDimBucket(bucket))
}

// ── Environment ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CpuTargetArch {
    X86_64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CpuTuningEnvironment {
    pub target_arch: CpuTargetArch,
    pub identity: CpuIdentity,
    pub available_features: CpuFeatureSet,
    pub thread_count: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuTuningEnvironmentError {
    BackendMismatch { actual: DeviceKind },
    CpuIdentityUnavailable,
    UnsupportedTargetArchitecture,
    InvalidThreadCount,
}

impl CpuTuningEnvironment {
    pub fn detect(
        capabilities: &BackendCapabilities,
        thread_count: u16,
    ) -> Result<Self, CpuTuningEnvironmentError> {
        if capabilities.backend != DeviceKind::Cpu {
            return Err(CpuTuningEnvironmentError::BackendMismatch {
                actual: capabilities.backend,
            });
        }
        if thread_count == 0 {
            return Err(CpuTuningEnvironmentError::InvalidThreadCount);
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = capabilities;
            let _ = thread_count;
            return Err(CpuTuningEnvironmentError::UnsupportedTargetArchitecture);
        }

        #[cfg(target_arch = "x86_64")]
        {
            let identity =
                detect_cpu_identity().ok_or(CpuTuningEnvironmentError::CpuIdentityUnavailable)?;
            Ok(Self {
                target_arch: CpuTargetArch::X86_64,
                identity,
                available_features: capabilities.cpu_features,
                thread_count,
            })
        }
    }
}

// ── Tuning request identity ──────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CpuTuningRequestKey {
    pub kernel: KernelKey,
    pub shape: CpuTuningShape,
    pub expected_abi: CpuKernelAbi,
}

// ── Registry fingerprint ─────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RegistryFingerprint(pub u64);

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

struct StableFingerprint {
    state: u64,
}

impl StableFingerprint {
    fn new() -> Self {
        Self {
            state: FNV_OFFSET_BASIS,
        }
    }

    fn write_u8(&mut self, value: u8) {
        self.state ^= u64::from(value);
        self.state = self.state.wrapping_mul(FNV_PRIME);
    }

    fn write_u16(&mut self, value: u16) {
        self.write_bytes(&value.to_le_bytes());
    }

    fn write_u32(&mut self, value: u32) {
        self.write_bytes(&value.to_le_bytes());
    }

    fn write_u64(&mut self, value: u64) {
        self.write_bytes(&value.to_le_bytes());
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= u64::from(byte);
            self.state = self.state.wrapping_mul(FNV_PRIME);
        }
    }

    fn write_str(&mut self, value: &str) {
        self.write_u64(value.len() as u64);
        self.write_bytes(value.as_bytes());
    }

    fn finish(self) -> RegistryFingerprint {
        RegistryFingerprint(self.state)
    }
}

pub fn fingerprint_cpu_registry(registrations: &[CpuKernelRegistration]) -> RegistryFingerprint {
    let mut f = StableFingerprint::new();
    f.write_u16(REGISTRY_FINGERPRINT_SCHEMA_V1);
    f.write_u64(registrations.len() as u64);

    for reg in registrations {
        f.write_str(reg.id.0);
        write_kernel_key(&mut f, reg.key);
        f.write_str(reg.shape_support_id.0);
        f.write_u16(reg.priority);
        f.write_u64(reg.required_features.bits());
        f.write_u8(cpu_kernel_abi_tag(reg.launch.abi()));
    }

    f.finish()
}

fn write_kernel_key(f: &mut StableFingerprint, key: KernelKey) {
    f.write_u8(backend_kind_tag(key.backend));
    f.write_u8(arch_kind_tag(key.arch));
    f.write_u8(op_kind_tag(key.op));
    f.write_u8(dtype_tag(key.weight_dtype));
    f.write_u8(dtype_tag(key.activation_dtype));
    f.write_u8(dtype_tag(key.output_dtype));
    write_optional_quant(f, key.weight_format);
    write_optional_quant(f, key.activation_format);
    f.write_u32(key.weight_layout.0);
    f.write_u32(key.activation_layout.0);
    f.write_u32(key.output_layout.0);
    f.write_u8(epilogue_kind_tag(key.epilogue));
}

fn write_optional_quant(f: &mut StableFingerprint, value: Option<QuantFormatId>) {
    match value {
        None => f.write_u8(0),
        Some(format) => {
            f.write_u8(1);
            f.write_u8(quant_format_tag(format));
        }
    }
}

// ── Stable tags (explicit match, no auto-derived tags) ────────────────────

fn backend_kind_tag(v: BackendKind) -> u8 {
    match v {
        BackendKind::Cpu => 1,
        BackendKind::Cuda => 2,
        BackendKind::Rocm => 3,
    }
}

fn arch_kind_tag(v: ArchKind) -> u8 {
    match v {
        ArchKind::Portable => 1,
        ArchKind::Avx2 => 2,
        ArchKind::Avx512 => 3,
        ArchKind::NvidiaSm => 4,
        ArchKind::AmdGfx => 5,
    }
}

fn op_kind_tag(v: OpKind) -> u8 {
    match v {
        OpKind::Add => 1,
        OpKind::RmsNorm => 2,
        OpKind::QMatmul => 3,
        OpKind::FusedMlp => 5,
        OpKind::Mul => 6,
        OpKind::Clamp => 7,
        OpKind::Silu => 8,
    }
}

fn epilogue_kind_tag(v: EpilogueKind) -> u8 {
    match v {
        EpilogueKind::Identity => 1,
        EpilogueKind::Bias => 2,
        EpilogueKind::BiasSilu => 3,
        EpilogueKind::BiasSiluMul => 4,
        EpilogueKind::ResidualAdd => 5,
        EpilogueKind::ClampSiluMul => 6,
    }
}

fn dtype_tag(v: DType) -> u8 {
    match v {
        DType::F32 => 1,
        DType::F16 => 2,
        DType::BF16 => 3,
        DType::I8 => 4,
        DType::I32 => 5,
        DType::PackedU8 => 6,
    }
}

fn quant_format_tag(v: QuantFormatId) -> u8 {
    match v {
        QuantFormatId::Q4K => 1,
        QuantFormatId::Q5K => 13,
        QuantFormatId::Q6K => 14,
        QuantFormatId::Q8_0 => 2,
        QuantFormatId::Q2K => 3,
        QuantFormatId::Q4X => 4,
        QuantFormatId::DualScale => 5,
        QuantFormatId::LogQ8 => 6,
        QuantFormatId::Q8K => 7,
        QuantFormatId::MXFP4 => 8,
        QuantFormatId::NVFP4 => 9,
        QuantFormatId::FP8E4M3Block => 10,
        QuantFormatId::FP8E4M3ScaleGrid => 11,
        QuantFormatId::IQ2_XXS => 12,
    }
}

fn cpu_kernel_abi_tag(v: CpuKernelAbi) -> u8 {
    match v {
        CpuKernelAbi::QMatmulF32 => 1,
        CpuKernelAbi::QMatmulBiasF32 => 2,
        CpuKernelAbi::DualScaleQ8Hot => 3,
        CpuKernelAbi::QMatmulF32Prefill => 4,
        CpuKernelAbi::Q8_0Q8Hot => 5,
        CpuKernelAbi::Q2KQ8Hot => 6,
        CpuKernelAbi::Mxfp4F32 => 7,
        CpuKernelAbi::Fp8ScaleGridF32 => 8,
        CpuKernelAbi::Fp8FusedMlpF32 => 9,
        CpuKernelAbi::Iq2XxsQ8Hot => 10,
    }
}

// ── Profile input and validated representation ───────────────────────────

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuTuningEntryInput {
    pub request: CpuTuningRequestKey,
    pub selected_registration_id: Box<str>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuTuningProfileInput {
    pub version: CpuTuningProfileVersion,
    pub shape_schema: CpuTuningShapeSchemaVersion,
    pub registry_fingerprint: RegistryFingerprint,
    pub environment: CpuTuningEnvironment,
    pub entries: Box<[CpuTuningEntryInput]>,
}

#[derive(Clone, Copy, Debug)]
struct ValidatedCpuTuningEntry {
    request: CpuTuningRequestKey,
    selected_registration: &'static CpuKernelRegistration,
}

#[derive(Clone, Debug)]
pub struct ValidatedCpuTuningProfile {
    version: CpuTuningProfileVersion,
    shape_schema: CpuTuningShapeSchemaVersion,
    registry_fingerprint: RegistryFingerprint,
    environment: CpuTuningEnvironment,
    entries: Box<[ValidatedCpuTuningEntry]>,
}

impl ValidatedCpuTuningProfile {
    pub fn version(&self) -> CpuTuningProfileVersion {
        self.version
    }

    pub fn shape_schema(&self) -> CpuTuningShapeSchemaVersion {
        self.shape_schema
    }

    pub fn registry_fingerprint(&self) -> RegistryFingerprint {
        self.registry_fingerprint
    }

    pub fn environment(&self) -> CpuTuningEnvironment {
        self.environment
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn lookup(
        &self,
        request: CpuTuningRequestKey,
    ) -> Option<&'static CpuKernelRegistration> {
        self.entries
            .iter()
            .find(|entry| entry.request == request)
            .map(|entry| entry.selected_registration)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CpuTuningProfileError {
    UnsupportedProfileVersion {
        actual: CpuTuningProfileVersion,
    },
    UnsupportedShapeSchema {
        actual: CpuTuningShapeSchemaVersion,
    },
    RegistryFingerprintMismatch {
        expected: RegistryFingerprint,
        actual: RegistryFingerprint,
    },
    EnvironmentMismatch {
        expected: CpuTuningEnvironment,
        actual: CpuTuningEnvironment,
    },
    DuplicateEntry {
        request: CpuTuningRequestKey,
    },
    DuplicateRegistrationId {
        id: Box<str>,
    },
    RegistrationBackendMismatch {
        id: Box<str>,
        actual: BackendKind,
    },
    RequestBackendMismatch {
        actual: BackendKind,
    },
    RegistrationNotFound {
        id: Box<str>,
    },
    RegistrationKeyMismatch {
        id: Box<str>,
    },
    RegistrationAbiMismatch {
        id: Box<str>,
    },
}

pub fn validate_cpu_tuning_profile(
    input: CpuTuningProfileInput,
    registrations: &'static [CpuKernelRegistration],
    current_environment: CpuTuningEnvironment,
) -> Result<ValidatedCpuTuningProfile, CpuTuningProfileError> {
    // 1. Version check
    if input.version != CPU_TUNING_PROFILE_V1 {
        return Err(CpuTuningProfileError::UnsupportedProfileVersion {
            actual: input.version,
        });
    }
    if input.shape_schema != CPU_TUNING_SHAPE_SCHEMA_V1 {
        return Err(CpuTuningProfileError::UnsupportedShapeSchema {
            actual: input.shape_schema,
        });
    }

    // 2. Fingerprint check
    let expected_fp = fingerprint_cpu_registry(registrations);
    if input.registry_fingerprint != expected_fp {
        return Err(CpuTuningProfileError::RegistryFingerprintMismatch {
            expected: expected_fp,
            actual: input.registry_fingerprint,
        });
    }

    // 3. Environment check
    if input.environment != current_environment {
        return Err(CpuTuningProfileError::EnvironmentMismatch {
            expected: current_environment,
            actual: input.environment,
        });
    }

    // 4. Validate the CPU registry boundary and unique IDs.
    for i in 0..registrations.len() {
        if registrations[i].key.backend != BackendKind::Cpu {
            return Err(CpuTuningProfileError::RegistrationBackendMismatch {
                id: registrations[i].id.0.into(),
                actual: registrations[i].key.backend,
            });
        }

        for j in 0..i {
            if registrations[i].id == registrations[j].id {
                return Err(CpuTuningProfileError::DuplicateRegistrationId {
                    id: registrations[i].id.0.into(),
                });
            }
        }
    }

    // 5. Validate entries
    let mut entries: Vec<ValidatedCpuTuningEntry> = Vec::with_capacity(input.entries.len());
    let input_entries = input.entries.into_vec();

    for entry in input_entries {
        // Request backend must be CPU
        if entry.request.kernel.backend != BackendKind::Cpu {
            return Err(CpuTuningProfileError::RequestBackendMismatch {
                actual: entry.request.kernel.backend,
            });
        }

        // Duplicate request key check
        if entries.iter().any(|e| e.request == entry.request) {
            return Err(CpuTuningProfileError::DuplicateEntry {
                request: entry.request,
            });
        }

        // Find registration by ID
        let registration = registrations
            .iter()
            .find(|r| r.id.0 == entry.selected_registration_id.as_ref())
            .ok_or_else(|| CpuTuningProfileError::RegistrationNotFound {
                id: entry.selected_registration_id.clone(),
            })?;

        // Key match
        if registration.key != entry.request.kernel {
            return Err(CpuTuningProfileError::RegistrationKeyMismatch {
                id: entry.selected_registration_id,
            });
        }

        // ABI match
        if registration.launch.abi() != entry.request.expected_abi {
            return Err(CpuTuningProfileError::RegistrationAbiMismatch {
                id: entry.selected_registration_id,
            });
        }

        entries.push(ValidatedCpuTuningEntry {
            request: entry.request,
            selected_registration: registration,
        });
    }

    Ok(ValidatedCpuTuningProfile {
        version: input.version,
        shape_schema: input.shape_schema,
        registry_fingerprint: input.registry_fingerprint,
        environment: input.environment,
        entries: entries.into_boxed_slice(),
    })
}

// ── Attachment validation ────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CpuTuningAttachError {
    Environment(CpuTuningEnvironmentError),
    RegistryFingerprintMismatch {
        expected: RegistryFingerprint,
        actual: RegistryFingerprint,
    },
    EnvironmentMismatch {
        expected: CpuTuningEnvironment,
        actual: CpuTuningEnvironment,
    },
    UnsupportedProfileVersion,
    UnsupportedShapeSchema,
}

pub fn validate_cpu_tuning_attachment(
    profile: &ValidatedCpuTuningProfile,
    registrations: &'static [CpuKernelRegistration],
    capabilities: &BackendCapabilities,
    thread_count: u16,
) -> Result<(), CpuTuningAttachError> {
    if profile.version() != CPU_TUNING_PROFILE_V1 {
        return Err(CpuTuningAttachError::UnsupportedProfileVersion);
    }
    if profile.shape_schema() != CPU_TUNING_SHAPE_SCHEMA_V1 {
        return Err(CpuTuningAttachError::UnsupportedShapeSchema);
    }

    let expected_fp = fingerprint_cpu_registry(registrations);
    if profile.registry_fingerprint() != expected_fp {
        return Err(CpuTuningAttachError::RegistryFingerprintMismatch {
            expected: expected_fp,
            actual: profile.registry_fingerprint(),
        });
    }

    let expected_env = CpuTuningEnvironment::detect(capabilities, thread_count)
        .map_err(CpuTuningAttachError::Environment)?;

    if profile.environment() != expected_env {
        return Err(CpuTuningAttachError::EnvironmentMismatch {
            expected: expected_env,
            actual: profile.environment(),
        });
    }

    Ok(())
}

// ── Selection source ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuKernelSelectionSource {
    StaticPriority,
    TunedProfile,
}

#[derive(Clone, Copy, Debug)]
pub struct SelectedCpuKernel {
    pub registration: &'static CpuKernelRegistration,
    pub source: CpuKernelSelectionSource,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Shape normalization ───────────────────────────────────────────

    #[test]
    fn power_of_two_shape_normalization_v1() {
        let cases = [
            (1, 1),
            (2, 2),
            (3, 4),
            (4, 4),
            (5, 8),
            (255, 256),
            (256, 256),
            (257, 512),
            (4096, 4096),
            (5000, 8192),
        ];
        for (raw_n, expected_n) in cases {
            let shape = normalize_cpu_tuning_shape(RawCpuTuningShape {
                batch: 1,
                m: 1,
                n: raw_n,
                k: 256,
            })
            .unwrap();
            assert_eq!(shape.n.get(), expected_n);
            assert_eq!(shape.k.get(), 256);
        }
    }

    #[test]
    fn rows_three_and_four_share_v1_bucket() {
        let a = normalize_cpu_tuning_shape(RawCpuTuningShape {
            batch: 1,
            m: 1,
            n: 3,
            k: 256,
        })
        .unwrap();
        let b = normalize_cpu_tuning_shape(RawCpuTuningShape {
            batch: 1,
            m: 1,
            n: 4,
            k: 256,
        })
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(a.n.get(), 4);
    }

    #[test]
    fn unsupported_batch_rejected() {
        assert_eq!(
            normalize_cpu_tuning_shape(RawCpuTuningShape {
                batch: 2,
                m: 1,
                n: 4,
                k: 256
            }),
            Err(CpuTuningShapeError::UnsupportedBatch { actual: 2 }),
        );
    }

    #[test]
    fn unsupported_m_rejected() {
        assert_eq!(
            normalize_cpu_tuning_shape(RawCpuTuningShape {
                batch: 1,
                m: 2,
                n: 4,
                k: 256
            }),
            Err(CpuTuningShapeError::UnsupportedM { actual: 2 }),
        );
    }

    #[test]
    fn zero_dimensions_rejected() {
        assert_eq!(
            normalize_cpu_tuning_shape(RawCpuTuningShape {
                batch: 1,
                m: 1,
                n: 0,
                k: 256
            }),
            Err(CpuTuningShapeError::ZeroDimension {
                dimension: CpuTuningDimension::N
            }),
        );
        assert_eq!(
            normalize_cpu_tuning_shape(RawCpuTuningShape {
                batch: 1,
                m: 1,
                n: 4,
                k: 0
            }),
            Err(CpuTuningShapeError::ZeroDimension {
                dimension: CpuTuningDimension::K
            }),
        );
    }

    // ── Fingerprint: every KernelKey field ────────────────────────────

    #[test]
    fn m46_quant_format_tags_are_append_only() {
        assert_eq!(quant_format_tag(QuantFormatId::MXFP4), 8);
        assert_eq!(quant_format_tag(QuantFormatId::NVFP4), 9);
        assert_eq!(quant_format_tag(QuantFormatId::FP8E4M3Block), 10);
        assert_eq!(quant_format_tag(QuantFormatId::FP8E4M3ScaleGrid), 11);
    }

    #[test]
    fn m48_elementwise_op_tags_are_append_only() {
        assert_eq!(op_kind_tag(OpKind::Add), 1);
        assert_eq!(op_kind_tag(OpKind::FusedMlp), 5);
        assert_eq!(op_kind_tag(OpKind::Mul), 6);
        assert_eq!(op_kind_tag(OpKind::Clamp), 7);
        assert_eq!(op_kind_tag(OpKind::Silu), 8);
    }

    #[test]
    fn every_kernel_key_field_affects_fingerprint() {
        let base = KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity);
        let base_fp = {
            let mut f = StableFingerprint::new();
            write_kernel_key(&mut f, base);
            f.finish()
        };

        let variants: &[(KernelKey, &str)] = &[
            (
                {
                    let mut k = base;
                    k.backend = BackendKind::Cuda;
                    k
                },
                "backend",
            ),
            (
                {
                    let mut k = base;
                    k.arch = ArchKind::Avx2;
                    k
                },
                "arch",
            ),
            (
                {
                    let mut k = base;
                    k.op = OpKind::Add;
                    k
                },
                "op",
            ),
            (
                {
                    let mut k = base;
                    k.weight_dtype = DType::F16;
                    k
                },
                "weight_dtype",
            ),
            (
                {
                    let mut k = base;
                    k.activation_dtype = DType::I8;
                    k
                },
                "activation_dtype",
            ),
            (
                {
                    let mut k = base;
                    k.output_dtype = DType::BF16;
                    k
                },
                "output_dtype",
            ),
            (
                {
                    let mut k = base;
                    k.weight_format = Some(QuantFormatId::Q4X);
                    k
                },
                "weight_format",
            ),
            (
                {
                    let mut k = base;
                    k.activation_format = Some(QuantFormatId::Q8K);
                    k
                },
                "activation_format",
            ),
            (
                {
                    let mut k = base;
                    k.weight_layout = PhysicalLayoutId::Q4K_COMPACT_V1;
                    k
                },
                "weight_layout",
            ),
            (
                {
                    let mut k = base;
                    k.activation_layout = PhysicalLayoutId::Q8K_ACTIVATION_REF_VECTOR_V1;
                    k
                },
                "activation_layout",
            ),
            (
                {
                    let mut k = base;
                    k.output_layout = PhysicalLayoutId::Q4X_HOT_AVX2_V1;
                    k
                },
                "output_layout",
            ),
            (
                {
                    let mut k = base;
                    k.epilogue = EpilogueKind::Bias;
                    k
                },
                "epilogue",
            ),
        ];

        for (variant, field_name) in variants {
            let mut f = StableFingerprint::new();
            write_kernel_key(&mut f, *variant);
            assert_ne!(
                f.finish(),
                base_fp,
                "KernelKey field '{field_name}' does not affect fingerprint",
            );
        }
    }

    // ── Fingerprint: registry properties ──────────────────────────────

    use augur_core::PhysicalLayoutId;
    use augur_kernel_api::{
        BatchClass, CpuFeatureSet, CpuKernelLaunch, CpuShapeSupportId, KernelLaunchError,
        KernelRegistrationId, Q4KHotCpuTensor, ShapeClass,
    };

    fn dummy_launch(
        _w: &Q4KHotCpuTensor,
        _i: &[f32],
        _o: &mut [f32],
    ) -> Result<(), KernelLaunchError> {
        Ok(())
    }

    fn decode_only(s: &ShapeClass) -> bool {
        s.batch == BatchClass::Decode
    }

    const SHAPE_A: CpuShapeSupportId = CpuShapeSupportId("test.shape.a");
    const SHAPE_B: CpuShapeSupportId = CpuShapeSupportId("test.shape.b");

    fn make_reg(id: &'static str, shape_id: CpuShapeSupportId, prio: u16) -> CpuKernelRegistration {
        CpuKernelRegistration {
            id: KernelRegistrationId(id),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: shape_id,
            shape_support: decode_only,
            priority: prio,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_launch),
        }
    }

    #[test]
    fn registry_order_affects_fingerprint() {
        let a = make_reg("a", SHAPE_A, 100);
        let b = make_reg("b", SHAPE_A, 100);
        let fp_ab = fingerprint_cpu_registry(&[a, b]);
        let fp_ba = fingerprint_cpu_registry(&[b, a]);
        assert_ne!(fp_ab, fp_ba);
    }

    #[test]
    fn shape_support_id_affects_fingerprint() {
        let r1 = make_reg("r", SHAPE_A, 100);
        let r2 = make_reg("r", SHAPE_B, 100);
        assert_ne!(
            fingerprint_cpu_registry(&[r1]),
            fingerprint_cpu_registry(&[r2]),
        );
    }

    #[test]
    fn launch_address_does_not_affect_fingerprint() {
        fn alt_launch(
            _w: &Q4KHotCpuTensor,
            _i: &[f32],
            _o: &mut [f32],
        ) -> Result<(), KernelLaunchError> {
            Ok(())
        }
        let r1 = CpuKernelRegistration {
            id: KernelRegistrationId("x"),
            key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
            shape_support_id: SHAPE_A,
            shape_support: decode_only,
            priority: 100,
            required_features: CpuFeatureSet::NONE,
            launch: CpuKernelLaunch::QMatmulF32(dummy_launch),
        };
        let r2 = CpuKernelRegistration {
            launch: CpuKernelLaunch::QMatmulF32(alt_launch),
            ..r1
        };
        assert_eq!(
            fingerprint_cpu_registry(&[r1]),
            fingerprint_cpu_registry(&[r2]),
        );
    }

    #[test]
    fn registration_count_affects_fingerprint() {
        let r = make_reg("r", SHAPE_A, 100);
        assert_ne!(
            fingerprint_cpu_registry(&[r]),
            fingerprint_cpu_registry(&[r, r]),
        );
    }
}
