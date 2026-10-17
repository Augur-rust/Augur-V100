//! Backend-independent floating-point semantics for static numerical lowering.

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32ProductSemantics {
    Native,
    ExplicitRoundNearestEven,
    ErrorFreeTransform,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32SumSemantics {
    Native,
    ExplicitRoundNearestEven,
    Compensated,
}

/// The accumulator promised by a numerical contract.
///
/// `ExactI32` is restricted to integer dot products whose range proof is part
/// of the contract.  It is deliberately distinct from native FP32: an
/// implementation cannot silently replace an integer accumulation with an
/// FMA chain and retain the same identity.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StaticNumericalAccumulatorStrategy {
    /// The stage is structurally present in the model contract but performs no
    /// arithmetic in this execution topology.
    NoOp,
    /// An immutable representation boundary with no accumulator of its own.
    Boundary,
    NativeF32,
    ExplicitRoundNearestEvenF32,
    CompensatedF32,
    ExactI32,
}

/// The representation at a stage boundary.  This is metadata for lowering
/// and identity; it does not authorize a runtime conversion by itself.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StaticNumericalBoundary {
    F32,
    BF16,
    I32,
    Q8,
    Q4,
    Q5,
    None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StaticNumericalClamp {
    /// IEEE-754 bound. `-infinity` denotes no finite lower clamp.
    pub min_f32_bits: u32,
    /// IEEE-754 bound. `+infinity` denotes no finite upper clamp.
    pub max_f32_bits: u32,
}

/// Source-bound proof inputs for one signed i32 dot accumulator.
///
/// The compiler independently derives both accumulator bounds.  The auxiliary
/// operand covers affine quantizers such as Q4/Q5, where the integer dot and
/// the input sum are accumulated separately before their FP32 scale terms are
/// combined.  A zero auxiliary magnitude means that no auxiliary accumulator
/// exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StaticNumericalIntegerRangeProof {
    pub term_count: u32,
    pub primary_lhs_abs_max: u32,
    pub rhs_abs_max: u32,
    pub auxiliary_lhs_abs_max: u32,
}

/// Fixed-point, source-bound error bounds consumed by the compiler.
///
/// Absolute error is represented in `2^-32` units and relative error in parts
/// per billion.  Keeping the values integral makes the contract deterministic
/// across hosts and lets the compiler use checked `u128` arithmetic for
/// cumulative propagation.  `model_error_bound_known` is intentionally
/// explicit: quantization and active-width capacity error must come from a
/// held-out quality report, never be disguised as arithmetic error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StaticNumericalErrorBudget {
    pub local_absolute_q32: u64,
    pub local_arithmetic_relative_ppb: u64,
    pub local_model_relative_ppb: u64,
    pub propagation_factor_ppm: u32,
    pub max_cumulative_absolute_q32: u64,
    pub max_cumulative_arithmetic_relative_ppb: u64,
    pub max_cumulative_model_relative_ppb: u64,
    pub logical_width: u32,
    pub physical_width: u32,
    pub boundary: StaticNumericalBoundary,
    pub finite_required: bool,
    /// Primary-input clamp, when the operation has one.
    pub clamp: Option<StaticNumericalClamp>,
    /// Secondary-input clamp for two-input operations such as SwiGLU.
    pub secondary_clamp: Option<StaticNumericalClamp>,
    pub model_error_bound_known: bool,
}

impl StaticNumericalErrorBudget {
    pub const PROPAGATION_IDENTITY_PPM: u32 = 1_000_000;

    // The constructor mirrors the serialized contract fields one-for-one so
    // callers cannot omit a bound by relying on a default.
    #[allow(clippy::too_many_arguments)]
    pub const fn conservative(
        local_absolute_q32: u64,
        local_arithmetic_relative_ppb: u64,
        local_model_relative_ppb: u64,
        max_cumulative_absolute_q32: u64,
        max_cumulative_arithmetic_relative_ppb: u64,
        max_cumulative_model_relative_ppb: u64,
        logical_width: u32,
        physical_width: u32,
        boundary: StaticNumericalBoundary,
        clamp: Option<StaticNumericalClamp>,
        model_error_bound_known: bool,
    ) -> Self {
        Self {
            local_absolute_q32,
            local_arithmetic_relative_ppb,
            local_model_relative_ppb,
            propagation_factor_ppm: Self::PROPAGATION_IDENTITY_PPM,
            max_cumulative_absolute_q32,
            max_cumulative_arithmetic_relative_ppb,
            max_cumulative_model_relative_ppb,
            logical_width,
            physical_width,
            boundary,
            finite_required: true,
            clamp,
            secondary_clamp: None,
            model_error_bound_known,
        }
    }
}

/// One source-bound logical operation in an ordered numerical-semantics IR.
///
/// The model profile assigns the stable ordinal and source operator identity.
/// Kernel families and backend capabilities are intentionally absent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StaticNumericalOpRequest {
    pub ordinal: u16,
    pub operator_identity_sha256: [u8; 32],
    pub product: F32ProductSemantics,
    pub sum: F32SumSemantics,
}

/// A source-bound operation with an executable cumulative error contract.
/// This is separate from the historical M216 request so old recovery plans
/// remain byte-compatible while new lowering paths must opt into the stronger
/// contract explicitly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StaticNumericalContractOpRequest {
    pub ordinal: u16,
    pub operator_identity_sha256: [u8; 32],
    pub product: F32ProductSemantics,
    pub sum: F32SumSemantics,
    pub accumulator: StaticNumericalAccumulatorStrategy,
    pub integer_range_proof: Option<StaticNumericalIntegerRangeProof>,
    pub error_budget: StaticNumericalErrorBudget,
}
