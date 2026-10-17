//! Kernel identity and coarse tuning keys.
//!
//! These types belong to backend lowering, registry lookup, benchmark
//! attribution, and future autotune. Graph IR must not carry `KernelKey`,
//! `ShapeClass`, `TileConfig` or `TunedKernelKey`.
//!
//! `OpKind` describes the logical kernel family. Decode/prefill workload
//! classification belongs to [`ShapeClass`], while fused output behavior
//! belongs to [`EpilogueKind`].

use augur_core::{DType, PhysicalLayoutId, QuantFormatId};

// ── Backend / Arch ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Rocm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ArchKind {
    Portable,
    Avx2,
    Avx512,
    /// NVIDIA Streaming Multiprocessor architecture family.
    ///
    /// The concrete compute capability belongs to CUDA backend capability
    /// metadata, not this coarse kernel identity.
    NvidiaSm,

    /// AMD GFX architecture family.
    ///
    /// The concrete GFX version belongs to ROCm backend capability metadata,
    /// not this coarse kernel identity.
    AmdGfx,
}

// ── Op ────────────────────────────────────────────────────

/// Canonical logical kernel family.
///
/// Decode versus prefill is represented by [`ShapeClass`], not by distinct
/// `QMatVec`/`QMatMul` variants.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OpKind {
    Add,
    Mul,
    Clamp,
    Silu,
    RmsNorm,
    QMatmul,
    FusedMlp,
}

// ── Epilogue ──────────────────────────────────────────────────────────────

/// Output transformation selected before execution.
///
/// `Identity` means that no additional epilogue is applied.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EpilogueKind {
    Identity,
    Bias,
    BiasSilu,
    BiasSiluMul,
    ResidualAdd,
    ClampSiluMul,
}

// ── KernelKey ─────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct KernelKey {
    pub backend: BackendKind,
    pub arch: ArchKind,
    pub op: OpKind,
    pub weight_dtype: DType,
    pub activation_dtype: DType,
    pub output_dtype: DType,
    pub weight_format: Option<QuantFormatId>,
    pub activation_format: Option<QuantFormatId>,
    pub weight_layout: PhysicalLayoutId,
    pub activation_layout: PhysicalLayoutId,
    pub output_layout: PhysicalLayoutId,
    pub epilogue: EpilogueKind,
}

impl KernelKey {
    pub const fn fp8_scale_grid_f32_cpu_portable() -> Self {
        Self {
            backend: BackendKind::Cpu,
            arch: ArchKind::Portable,
            op: OpKind::QMatmul,
            weight_dtype: DType::PackedU8,
            activation_dtype: DType::F32,
            output_dtype: DType::F32,
            weight_format: Some(QuantFormatId::FP8E4M3ScaleGrid),
            activation_format: None,
            weight_layout: PhysicalLayoutId::FP8_E4M3_SCALE_GRID_HOT_CPU_PORTABLE_V1,
            activation_layout: PhysicalLayoutId::HOST_CONTIGUOUS_F32,
            output_layout: PhysicalLayoutId::HOST_CONTIGUOUS_F32,
            epilogue: EpilogueKind::Identity,
        }
    }

    pub const fn fp8_scale_grid_fused_mlp_f32_cpu_portable() -> Self {
        Self {
            backend: BackendKind::Cpu,
            arch: ArchKind::Portable,
            op: OpKind::FusedMlp,
            weight_dtype: DType::PackedU8,
            activation_dtype: DType::F32,
            output_dtype: DType::F32,
            weight_format: Some(QuantFormatId::FP8E4M3ScaleGrid),
            activation_format: None,
            weight_layout: PhysicalLayoutId::FP8_E4M3_SCALE_GRID_HOT_CPU_PORTABLE_V1,
            activation_layout: PhysicalLayoutId::HOST_CONTIGUOUS_F32,
            output_layout: PhysicalLayoutId::HOST_CONTIGUOUS_F32,
            epilogue: EpilogueKind::ClampSiluMul,
        }
    }

    /// Exact selection identity for the CPU portable decode-only
    /// MXFP4 x f32 Identity qmatvec kernel.
    pub const fn mxfp4_f32_cpu_portable() -> Self {
        Self {
            backend: BackendKind::Cpu,
            arch: ArchKind::Portable,
            op: OpKind::QMatmul,
            weight_dtype: DType::PackedU8,
            activation_dtype: DType::F32,
            output_dtype: DType::F32,
            weight_format: Some(QuantFormatId::MXFP4),
            activation_format: None,
            weight_layout: PhysicalLayoutId::MXFP4_HOT_CPU_PORTABLE_V1,
            activation_layout: PhysicalLayoutId::HOST_CONTIGUOUS_F32,
            output_layout: PhysicalLayoutId::HOST_CONTIGUOUS_F32,
            epilogue: EpilogueKind::Identity,
        }
    }

    /// Selection identity for the current Q4K × f32 CPU portable kernel
    /// family.
    ///
    /// Constructing a key does not prove that a registration exists for the
    /// requested epilogue.
    pub const fn q4k_f32_cpu_portable(epilogue: EpilogueKind) -> Self {
        Self {
            backend: BackendKind::Cpu,
            arch: ArchKind::Portable,
            op: OpKind::QMatmul,
            weight_dtype: DType::PackedU8,
            activation_dtype: DType::F32,
            output_dtype: DType::F32,
            weight_format: Some(QuantFormatId::Q4K),
            activation_format: None,
            weight_layout: PhysicalLayoutId::Q4K_HOT_CPU_PORTABLE_V1,
            activation_layout: PhysicalLayoutId::HOST_CONTIGUOUS_F32,
            output_layout: PhysicalLayoutId::HOST_CONTIGUOUS_F32,
            epilogue,
        }
    }

    /// Exact selection identity for the CPU portable decode-only
    /// DualScale × Q8Hot qmatvec kernel.
    ///
    /// M26 supports only the Identity epilogue for this kernel family.
    /// Bias and BiasSiLU are deliberately not representable through this
    /// canonical constructor.
    pub const fn dual_scale_q8hot_cpu_portable() -> Self {
        Self {
            backend: BackendKind::Cpu,
            arch: ArchKind::Portable,
            op: OpKind::QMatmul,

            weight_dtype: DType::PackedU8,
            activation_dtype: DType::I8,
            output_dtype: DType::F32,

            weight_format: Some(QuantFormatId::DualScale),

            activation_format: Some(QuantFormatId::Q8K),

            weight_layout: PhysicalLayoutId::DUAL_SCALE_HOT_CPU_PORTABLE_V1,

            activation_layout: PhysicalLayoutId::Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1,

            output_layout: PhysicalLayoutId::HOST_CONTIGUOUS_F32,

            epilogue: EpilogueKind::Identity,
        }
    }

    pub const fn q8_0_q8hot_cpu_portable() -> Self {
        Self::mixed_weight_q8hot_cpu_portable(
            QuantFormatId::Q8_0,
            PhysicalLayoutId::Q8_0_HOT_CPU_PORTABLE_V1,
        )
    }

    pub const fn q2k_q8hot_cpu_portable() -> Self {
        Self::mixed_weight_q8hot_cpu_portable(
            QuantFormatId::Q2K,
            PhysicalLayoutId::Q2K_HOT_CPU_PORTABLE_V1,
        )
    }

    pub const fn iq2_xxs_q8hot_cpu_portable() -> Self {
        Self::mixed_weight_q8hot_cpu_portable(
            QuantFormatId::IQ2_XXS,
            PhysicalLayoutId::IQ2_XXS_HOT_CPU_PORTABLE_V1,
        )
    }

    const fn mixed_weight_q8hot_cpu_portable(
        format: QuantFormatId,
        layout: PhysicalLayoutId,
    ) -> Self {
        Self {
            backend: BackendKind::Cpu,
            arch: ArchKind::Portable,
            op: OpKind::QMatmul,
            weight_dtype: DType::PackedU8,
            activation_dtype: DType::I8,
            output_dtype: DType::F32,
            weight_format: Some(format),
            activation_format: Some(QuantFormatId::Q8K),
            weight_layout: layout,
            activation_layout: PhysicalLayoutId::Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1,
            output_layout: PhysicalLayoutId::HOST_CONTIGUOUS_F32,
            epilogue: EpilogueKind::Identity,
        }
    }

    /// Exact selection identity for the Q4K × f32 CPU AVX2 decode kernel.
    ///
    /// This key reuses the existing `Q4K_HOT_CPU_PORTABLE_V1` layout (M27-A
    /// layout audit conclusion). M27 supports only the Identity epilogue.
    /// Bias and BiasSiLU are deliberately not representable through this
    /// canonical constructor.
    pub const fn q4k_f32_cpu_avx2_identity() -> Self {
        Self {
            backend: BackendKind::Cpu,
            arch: ArchKind::Avx2,
            op: OpKind::QMatmul,
            weight_dtype: DType::PackedU8,
            activation_dtype: DType::F32,
            output_dtype: DType::F32,
            weight_format: Some(QuantFormatId::Q4K),
            activation_format: None,
            weight_layout: PhysicalLayoutId::Q4K_HOT_CPU_PORTABLE_V1,
            activation_layout: PhysicalLayoutId::HOST_CONTIGUOUS_F32,
            output_layout: PhysicalLayoutId::HOST_CONTIGUOUS_F32,
            epilogue: EpilogueKind::Identity,
        }
    }

    /// Exact selection identity for the CPU AVX2 decode-only DualScale × Q8Hot
    /// qmatvec kernel.
    ///
    /// M28 supports only the Identity epilogue. This key reuses the existing
    /// `DUAL_SCALE_HOT_CPU_PORTABLE_V1` and `Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1`
    /// layouts (M28-A layout audit conclusion). Bias and BiasSiLU are
    /// deliberately not representable through this canonical constructor.
    pub const fn dual_scale_q8hot_cpu_avx2_identity() -> Self {
        Self {
            backend: BackendKind::Cpu,
            arch: ArchKind::Avx2,
            op: OpKind::QMatmul,

            weight_dtype: DType::PackedU8,
            activation_dtype: DType::I8,
            output_dtype: DType::F32,

            weight_format: Some(QuantFormatId::DualScale),

            activation_format: Some(QuantFormatId::Q8K),

            weight_layout: PhysicalLayoutId::DUAL_SCALE_HOT_CPU_PORTABLE_V1,

            activation_layout: PhysicalLayoutId::Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1,

            output_layout: PhysicalLayoutId::HOST_CONTIGUOUS_F32,

            epilogue: EpilogueKind::Identity,
        }
    }
}

// ── ShapeClass ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BatchClass {
    Decode,
    SmallBatch,
    Prefill,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SizeClass {
    Tiny,
    Small,
    Medium,
    Large,
    Huge,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ShapeClass {
    pub batch: BatchClass,
    pub m: SizeClass,
    pub n: SizeClass,
    pub k: SizeClass,
}

impl ShapeClass {
    pub const DECODE_GEMV: Self = Self {
        batch: BatchClass::Decode,
        m: SizeClass::Tiny,
        n: SizeClass::Small,
        k: SizeClass::Medium,
    };

    pub const PREFILL_GEMM: Self = Self {
        batch: BatchClass::Prefill,
        m: SizeClass::Small,
        n: SizeClass::Medium,
        k: SizeClass::Medium,
    };

    pub fn classify_matmul(m: usize, n: usize, k: usize, batch: usize) -> Self {
        let batch_class = if batch == 1 {
            BatchClass::Decode
        } else if batch <= 8 {
            BatchClass::SmallBatch
        } else {
            BatchClass::Prefill
        };

        Self {
            batch: batch_class,
            m: classify_size(m),
            n: classify_size(n),
            k: classify_size(k),
        }
    }

    /// Classifies an explicitly batched/prefill matrix multiplication.
    ///
    /// Rank-2 activation identity is authoritative, so the degenerate `m=1`
    /// case remains a prefill workload rather than aliasing decode GEMV.
    pub fn classify_prefill(m: usize, n: usize, k: usize) -> Self {
        Self {
            batch: if m <= 8 {
                BatchClass::SmallBatch
            } else {
                BatchClass::Prefill
            },
            m: classify_size(m),
            n: classify_size(n),
            k: classify_size(k),
        }
    }
}

fn classify_size(value: usize) -> SizeClass {
    match value {
        0..=64 => SizeClass::Tiny,
        65..=512 => SizeClass::Small,
        513..=4096 => SizeClass::Medium,
        4097..=16384 => SizeClass::Large,
        _ => SizeClass::Huge,
    }
}

// ── TileConfig ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TileConfig {
    pub tile_m: u16,
    pub tile_n: u16,
    pub tile_k: u16,
    pub split_k: u16,
    pub vector_lanes: u16,
    pub unroll_k: u16,
}

impl TileConfig {
    pub const NONE: Self = Self {
        tile_m: 0,
        tile_n: 0,
        tile_k: 0,
        split_k: 1,
        vector_lanes: 0,
        unroll_k: 1,
    };

    pub const PORTABLE_GEMV: Self = Self {
        tile_m: 1,
        tile_n: 1,
        tile_k: 256,
        split_k: 1,
        vector_lanes: 0,
        unroll_k: 1,
    };
}

// ── TunedKernelKey ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TunedKernelKey {
    pub kernel: KernelKey,
    pub shape: ShapeClass,
    pub tile: TileConfig,
}
