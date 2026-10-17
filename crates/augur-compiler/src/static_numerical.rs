//! Compile source-bound floating-point semantics into immutable kernel families.
//!
//! This module owns no model identity and performs no runtime selection. A
//! model profile supplies an ordered list of logical operations and their
//! required product/sum semantics; compilation either resolves every entry to
//! one supported kernel family or fails closed.

use sha2::Digest;

pub use augur_graph::{
    F32ProductSemantics, F32SumSemantics, StaticNumericalAccumulatorStrategy,
    StaticNumericalBoundary, StaticNumericalClamp, StaticNumericalContractOpRequest,
    StaticNumericalErrorBudget, StaticNumericalIntegerRangeProof, StaticNumericalOpRequest,
};

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StaticNumericalKernelFamily {
    NoOp,
    Boundary,
    NativeF32,
    ExplicitRoundNearestEvenF32,
    CompensatedF32,
    ExactI32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledStaticNumericalOp {
    pub ordinal: u16,
    pub operator_identity_sha256: [u8; 32],
    pub kernel: StaticNumericalKernelFamily,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledNumericalContractOp {
    pub ordinal: u16,
    pub operator_identity_sha256: [u8; 32],
    pub kernel: StaticNumericalKernelFamily,
    pub accumulator: StaticNumericalAccumulatorStrategy,
    pub integer_range_proof: Option<CompiledNumericalIntegerRangeProof>,
    pub error_budget: StaticNumericalErrorBudget,
    pub cumulative_absolute_q32: u64,
    pub cumulative_arithmetic_relative_ppb: u64,
    pub cumulative_model_relative_ppb: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledNumericalIntegerRangeProof {
    pub source: StaticNumericalIntegerRangeProof,
    pub primary_accumulator_abs_max: u64,
    pub auxiliary_accumulator_abs_max: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledNumericalContract {
    operations: Box<[CompiledNumericalContractOp]>,
    contract_digest_sha256: [u8; 32],
    pub quality_gate_required: bool,
}

impl CompiledNumericalContract {
    pub fn operations(&self) -> &[CompiledNumericalContractOp] {
        &self.operations
    }

    pub const fn contract_digest_sha256(&self) -> [u8; 32] {
        self.contract_digest_sha256
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledStaticNumericalPlan {
    operations: Box<[CompiledStaticNumericalOp]>,
}

impl CompiledStaticNumericalPlan {
    pub fn operations(&self) -> &[CompiledStaticNumericalOp] {
        &self.operations
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StaticNumericalCompileError {
    EmptyPlan,
    OperationOrder,
    ZeroOperatorIdentity,
    DuplicateOperatorIdentity,
    UnsupportedSemantics,
    InvalidBudget,
    MissingIntegerRangeProof,
    UnexpectedIntegerRangeProof,
    IntegerRangeExceeded,
    BudgetOverflow,
    CumulativeBudgetExceeded,
}

impl std::fmt::Display for StaticNumericalCompileError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "static numerical compilation failed: {self:?}")
    }
}

impl std::error::Error for StaticNumericalCompileError {}

fn lower_kernel(
    product: F32ProductSemantics,
    sum: F32SumSemantics,
) -> Result<StaticNumericalKernelFamily, StaticNumericalCompileError> {
    match (product, sum) {
        (F32ProductSemantics::Native, F32SumSemantics::Native) => {
            Ok(StaticNumericalKernelFamily::NativeF32)
        }
        (
            F32ProductSemantics::ExplicitRoundNearestEven,
            F32SumSemantics::ExplicitRoundNearestEven,
        ) => Ok(StaticNumericalKernelFamily::ExplicitRoundNearestEvenF32),
        (F32ProductSemantics::ErrorFreeTransform, F32SumSemantics::Compensated) => {
            Ok(StaticNumericalKernelFamily::CompensatedF32)
        }
        _ => Err(StaticNumericalCompileError::UnsupportedSemantics),
    }
}

fn lower_contract_kernel(
    product: F32ProductSemantics,
    sum: F32SumSemantics,
    accumulator: StaticNumericalAccumulatorStrategy,
) -> Result<StaticNumericalKernelFamily, StaticNumericalCompileError> {
    let expected = match accumulator {
        StaticNumericalAccumulatorStrategy::NoOp | StaticNumericalAccumulatorStrategy::Boundary => {
            (F32ProductSemantics::Native, F32SumSemantics::Native)
        }
        StaticNumericalAccumulatorStrategy::NativeF32 => {
            (F32ProductSemantics::Native, F32SumSemantics::Native)
        }
        StaticNumericalAccumulatorStrategy::ExplicitRoundNearestEvenF32 => (
            F32ProductSemantics::ExplicitRoundNearestEven,
            F32SumSemantics::ExplicitRoundNearestEven,
        ),
        StaticNumericalAccumulatorStrategy::CompensatedF32 => (
            F32ProductSemantics::ErrorFreeTransform,
            F32SumSemantics::Compensated,
        ),
        StaticNumericalAccumulatorStrategy::ExactI32 => {
            (F32ProductSemantics::Native, F32SumSemantics::Native)
        }
    };
    if (product, sum) != expected {
        return Err(StaticNumericalCompileError::UnsupportedSemantics);
    }
    Ok(match accumulator {
        StaticNumericalAccumulatorStrategy::NoOp => StaticNumericalKernelFamily::NoOp,
        StaticNumericalAccumulatorStrategy::Boundary => StaticNumericalKernelFamily::Boundary,
        StaticNumericalAccumulatorStrategy::NativeF32 => StaticNumericalKernelFamily::NativeF32,
        StaticNumericalAccumulatorStrategy::ExplicitRoundNearestEvenF32 => {
            StaticNumericalKernelFamily::ExplicitRoundNearestEvenF32
        }
        StaticNumericalAccumulatorStrategy::CompensatedF32 => {
            StaticNumericalKernelFamily::CompensatedF32
        }
        StaticNumericalAccumulatorStrategy::ExactI32 => StaticNumericalKernelFamily::ExactI32,
    })
}

fn compile_integer_range_proof(
    accumulator: StaticNumericalAccumulatorStrategy,
    budget: StaticNumericalErrorBudget,
    proof: Option<StaticNumericalIntegerRangeProof>,
) -> Result<Option<CompiledNumericalIntegerRangeProof>, StaticNumericalCompileError> {
    if accumulator == StaticNumericalAccumulatorStrategy::Boundary
        && (budget.boundary == StaticNumericalBoundary::None
            || budget.local_absolute_q32 != 0
            || budget.local_arithmetic_relative_ppb != 0
            || budget.clamp.is_some()
            || budget.secondary_clamp.is_some())
    {
        return Err(StaticNumericalCompileError::InvalidBudget);
    }
    if accumulator == StaticNumericalAccumulatorStrategy::NoOp
        && (budget.boundary != StaticNumericalBoundary::None
            || budget.local_absolute_q32 != 0
            || budget.local_arithmetic_relative_ppb != 0
            || budget.local_model_relative_ppb != 0
            || budget.propagation_factor_ppm
                != StaticNumericalErrorBudget::PROPAGATION_IDENTITY_PPM
            || budget.logical_width != 1
            || budget.physical_width != 1
            || budget.finite_required
            || budget.clamp.is_some()
            || budget.secondary_clamp.is_some())
    {
        return Err(StaticNumericalCompileError::InvalidBudget);
    }
    if accumulator != StaticNumericalAccumulatorStrategy::ExactI32 {
        return match proof {
            None => Ok(None),
            Some(_) => Err(StaticNumericalCompileError::UnexpectedIntegerRangeProof),
        };
    }
    if budget.boundary != StaticNumericalBoundary::I32
        || budget.clamp.is_some()
        || budget.secondary_clamp.is_some()
    {
        return Err(StaticNumericalCompileError::InvalidBudget);
    }
    let proof = proof.ok_or(StaticNumericalCompileError::MissingIntegerRangeProof)?;
    if proof.term_count == 0 || proof.primary_lhs_abs_max == 0 || proof.rhs_abs_max == 0 {
        return Err(StaticNumericalCompileError::IntegerRangeExceeded);
    }
    let primary = u128::from(proof.term_count)
        * u128::from(proof.primary_lhs_abs_max)
        * u128::from(proof.rhs_abs_max);
    let auxiliary = u128::from(proof.term_count)
        * u128::from(proof.auxiliary_lhs_abs_max)
        * u128::from(proof.rhs_abs_max);
    if primary > i32::MAX as u128 || auxiliary > i32::MAX as u128 {
        return Err(StaticNumericalCompileError::IntegerRangeExceeded);
    }
    Ok(Some(CompiledNumericalIntegerRangeProof {
        source: proof,
        primary_accumulator_abs_max: primary as u64,
        auxiliary_accumulator_abs_max: auxiliary as u64,
    }))
}

fn checked_propagate(
    previous: u64,
    local: u64,
    factor_ppm: u32,
) -> Result<u64, StaticNumericalCompileError> {
    let scaled = (u128::from(previous) * u128::from(factor_ppm)).div_ceil(1_000_000);
    let total = scaled
        .checked_add(u128::from(local))
        .ok_or(StaticNumericalCompileError::BudgetOverflow)?;
    u64::try_from(total).map_err(|_| StaticNumericalCompileError::BudgetOverflow)
}

fn validate_clamp(clamp: StaticNumericalClamp) -> Result<(), StaticNumericalCompileError> {
    let min = f32::from_bits(clamp.min_f32_bits);
    let max = f32::from_bits(clamp.max_f32_bits);
    let valid_lower = min.is_finite() || min == f32::NEG_INFINITY;
    let valid_upper = max.is_finite() || max == f32::INFINITY;
    if !valid_lower || !valid_upper || min > max || (min.is_infinite() && max.is_infinite()) {
        return Err(StaticNumericalCompileError::InvalidBudget);
    }
    Ok(())
}

fn validate_budget(budget: StaticNumericalErrorBudget) -> Result<(), StaticNumericalCompileError> {
    if budget.logical_width == 0
        || budget.physical_width == 0
        || budget.logical_width > budget.physical_width
        || budget.propagation_factor_ppm == 0
    {
        return Err(StaticNumericalCompileError::InvalidBudget);
    }
    if let Some(clamp) = budget.clamp {
        validate_clamp(clamp)?;
    }
    if let Some(clamp) = budget.secondary_clamp {
        validate_clamp(clamp)?;
    }
    Ok(())
}

/// Compile an ordered numerical contract and fail closed on any unbounded or
/// structurally invalid cumulative error path.
pub fn compile_numerical_contract(
    requests: &[StaticNumericalContractOpRequest],
) -> Result<CompiledNumericalContract, StaticNumericalCompileError> {
    if requests.is_empty() {
        return Err(StaticNumericalCompileError::EmptyPlan);
    }
    let mut operations = Vec::with_capacity(requests.len());
    let mut identities = std::collections::BTreeSet::new();
    let mut cumulative_absolute_q32 = 0_u64;
    let mut cumulative_arithmetic_relative_ppb = 0_u64;
    let mut cumulative_model_relative_ppb = 0_u64;
    let mut quality_gate_required = false;
    let mut digest = sha2::Sha256::new();
    digest.update(b"CRYSTAL-QUANT-NUMERICAL-CONTRACT-V1\0");
    for (index, request) in requests.iter().enumerate() {
        if usize::from(request.ordinal) != index {
            return Err(StaticNumericalCompileError::OperationOrder);
        }
        if request.operator_identity_sha256 == [0; 32] {
            return Err(StaticNumericalCompileError::ZeroOperatorIdentity);
        }
        if !identities.insert(request.operator_identity_sha256) {
            return Err(StaticNumericalCompileError::DuplicateOperatorIdentity);
        }
        validate_budget(request.error_budget)?;
        let kernel = lower_contract_kernel(request.product, request.sum, request.accumulator)?;
        let integer_range_proof = compile_integer_range_proof(
            request.accumulator,
            request.error_budget,
            request.integer_range_proof,
        )?;
        cumulative_absolute_q32 = checked_propagate(
            cumulative_absolute_q32,
            request.error_budget.local_absolute_q32,
            request.error_budget.propagation_factor_ppm,
        )?;
        cumulative_arithmetic_relative_ppb = checked_propagate(
            cumulative_arithmetic_relative_ppb,
            request.error_budget.local_arithmetic_relative_ppb,
            request.error_budget.propagation_factor_ppm,
        )?;
        cumulative_model_relative_ppb = checked_propagate(
            cumulative_model_relative_ppb,
            request.error_budget.local_model_relative_ppb,
            request.error_budget.propagation_factor_ppm,
        )?;
        if cumulative_absolute_q32 > request.error_budget.max_cumulative_absolute_q32
            || cumulative_arithmetic_relative_ppb
                > request.error_budget.max_cumulative_arithmetic_relative_ppb
            || cumulative_model_relative_ppb
                > request.error_budget.max_cumulative_model_relative_ppb
        {
            return Err(StaticNumericalCompileError::CumulativeBudgetExceeded);
        }
        quality_gate_required |= !request.error_budget.model_error_bound_known;
        let operation = CompiledNumericalContractOp {
            ordinal: request.ordinal,
            operator_identity_sha256: request.operator_identity_sha256,
            kernel,
            accumulator: request.accumulator,
            integer_range_proof,
            error_budget: request.error_budget,
            cumulative_absolute_q32,
            cumulative_arithmetic_relative_ppb,
            cumulative_model_relative_ppb,
        };
        digest.update(request.ordinal.to_le_bytes());
        digest.update(request.operator_identity_sha256);
        digest.update([request.product as u8, request.sum as u8]);
        digest.update([request.accumulator as u8, kernel as u8]);
        digest.update(request.error_budget.local_absolute_q32.to_le_bytes());
        digest.update(
            request
                .error_budget
                .local_arithmetic_relative_ppb
                .to_le_bytes(),
        );
        digest.update(request.error_budget.local_model_relative_ppb.to_le_bytes());
        digest.update(request.error_budget.propagation_factor_ppm.to_le_bytes());
        digest.update(
            request
                .error_budget
                .max_cumulative_absolute_q32
                .to_le_bytes(),
        );
        digest.update(
            request
                .error_budget
                .max_cumulative_arithmetic_relative_ppb
                .to_le_bytes(),
        );
        digest.update(
            request
                .error_budget
                .max_cumulative_model_relative_ppb
                .to_le_bytes(),
        );
        digest.update(request.error_budget.logical_width.to_le_bytes());
        digest.update(request.error_budget.physical_width.to_le_bytes());
        digest.update([request.error_budget.boundary as u8]);
        digest.update([request.error_budget.finite_required as u8]);
        digest.update([request.error_budget.model_error_bound_known as u8]);
        if let Some(clamp) = request.error_budget.clamp {
            digest.update([1]);
            digest.update(clamp.min_f32_bits.to_le_bytes());
            digest.update(clamp.max_f32_bits.to_le_bytes());
        } else {
            digest.update([0]);
        }
        if let Some(clamp) = request.error_budget.secondary_clamp {
            digest.update([1]);
            digest.update(clamp.min_f32_bits.to_le_bytes());
            digest.update(clamp.max_f32_bits.to_le_bytes());
        } else {
            digest.update([0]);
        }
        if let Some(proof) = integer_range_proof {
            digest.update([1]);
            digest.update(proof.source.term_count.to_le_bytes());
            digest.update(proof.source.primary_lhs_abs_max.to_le_bytes());
            digest.update(proof.source.rhs_abs_max.to_le_bytes());
            digest.update(proof.source.auxiliary_lhs_abs_max.to_le_bytes());
            digest.update(proof.primary_accumulator_abs_max.to_le_bytes());
            digest.update(proof.auxiliary_accumulator_abs_max.to_le_bytes());
        } else {
            digest.update([0]);
        }
        digest.update(cumulative_absolute_q32.to_le_bytes());
        digest.update(cumulative_arithmetic_relative_ppb.to_le_bytes());
        digest.update(cumulative_model_relative_ppb.to_le_bytes());
        operations.push(operation);
    }
    Ok(CompiledNumericalContract {
        operations: operations.into_boxed_slice(),
        contract_digest_sha256: digest.finalize().into(),
        quality_gate_required,
    })
}

pub fn compile_static_numerical_plan(
    requests: &[StaticNumericalOpRequest],
) -> Result<CompiledStaticNumericalPlan, StaticNumericalCompileError> {
    if requests.is_empty() {
        return Err(StaticNumericalCompileError::EmptyPlan);
    }
    let mut operations = Vec::with_capacity(requests.len());
    let mut identities = std::collections::BTreeSet::new();
    for (index, request) in requests.iter().enumerate() {
        if usize::from(request.ordinal) != index {
            return Err(StaticNumericalCompileError::OperationOrder);
        }
        if request.operator_identity_sha256 == [0; 32] {
            return Err(StaticNumericalCompileError::ZeroOperatorIdentity);
        }
        if !identities.insert(request.operator_identity_sha256) {
            return Err(StaticNumericalCompileError::DuplicateOperatorIdentity);
        }
        operations.push(CompiledStaticNumericalOp {
            ordinal: request.ordinal,
            operator_identity_sha256: request.operator_identity_sha256,
            kernel: lower_kernel(request.product, request.sum)?,
        });
    }
    Ok(CompiledStaticNumericalPlan {
        operations: operations.into_boxed_slice(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(
        ordinal: u16,
        product: F32ProductSemantics,
        sum: F32SumSemantics,
    ) -> StaticNumericalOpRequest {
        StaticNumericalOpRequest {
            ordinal,
            operator_identity_sha256: [ordinal as u8 + 1; 32],
            product,
            sum,
        }
    }

    #[test]
    fn supported_semantics_lower_to_fixed_kernel_families() {
        let plan = compile_static_numerical_plan(&[
            request(0, F32ProductSemantics::Native, F32SumSemantics::Native),
            request(
                1,
                F32ProductSemantics::ExplicitRoundNearestEven,
                F32SumSemantics::ExplicitRoundNearestEven,
            ),
            request(
                2,
                F32ProductSemantics::ErrorFreeTransform,
                F32SumSemantics::Compensated,
            ),
        ])
        .unwrap();
        assert_eq!(
            plan.operations()
                .iter()
                .map(|operation| operation.kernel)
                .collect::<Vec<_>>(),
            [
                StaticNumericalKernelFamily::NativeF32,
                StaticNumericalKernelFamily::ExplicitRoundNearestEvenF32,
                StaticNumericalKernelFamily::CompensatedF32,
            ]
        );
    }

    #[test]
    fn drift_and_unrepresentable_combinations_fail_closed() {
        let mut reordered = [
            request(0, F32ProductSemantics::Native, F32SumSemantics::Native),
            request(1, F32ProductSemantics::Native, F32SumSemantics::Native),
        ];
        reordered.swap(0, 1);
        assert_eq!(
            compile_static_numerical_plan(&reordered),
            Err(StaticNumericalCompileError::OperationOrder)
        );
        assert_eq!(
            compile_static_numerical_plan(&[request(
                0,
                F32ProductSemantics::Native,
                F32SumSemantics::Compensated,
            )]),
            Err(StaticNumericalCompileError::UnsupportedSemantics)
        );
    }

    fn contract_request(
        ordinal: u16,
        accumulator: StaticNumericalAccumulatorStrategy,
        mut budget: StaticNumericalErrorBudget,
    ) -> StaticNumericalContractOpRequest {
        let (product, sum) = match accumulator {
            StaticNumericalAccumulatorStrategy::NoOp
            | StaticNumericalAccumulatorStrategy::Boundary
            | StaticNumericalAccumulatorStrategy::NativeF32
            | StaticNumericalAccumulatorStrategy::ExactI32 => {
                (F32ProductSemantics::Native, F32SumSemantics::Native)
            }
            StaticNumericalAccumulatorStrategy::ExplicitRoundNearestEvenF32 => (
                F32ProductSemantics::ExplicitRoundNearestEven,
                F32SumSemantics::ExplicitRoundNearestEven,
            ),
            StaticNumericalAccumulatorStrategy::CompensatedF32 => (
                F32ProductSemantics::ErrorFreeTransform,
                F32SumSemantics::Compensated,
            ),
        };
        let integer_range_proof = (accumulator == StaticNumericalAccumulatorStrategy::ExactI32)
            .then_some(StaticNumericalIntegerRangeProof {
                term_count: 32,
                primary_lhs_abs_max: 15,
                rhs_abs_max: 127,
                auxiliary_lhs_abs_max: 1,
            });
        if accumulator == StaticNumericalAccumulatorStrategy::ExactI32 {
            budget.boundary = StaticNumericalBoundary::I32;
            budget.clamp = None;
            budget.secondary_clamp = None;
        }
        StaticNumericalContractOpRequest {
            ordinal,
            operator_identity_sha256: [ordinal as u8 + 1; 32],
            product,
            sum,
            accumulator,
            integer_range_proof,
            error_budget: budget,
        }
    }

    fn budget(max: u64) -> StaticNumericalErrorBudget {
        StaticNumericalErrorBudget {
            local_absolute_q32: 1,
            local_arithmetic_relative_ppb: 2,
            local_model_relative_ppb: 3,
            propagation_factor_ppm: 2_000_000,
            max_cumulative_absolute_q32: max,
            max_cumulative_arithmetic_relative_ppb: max,
            max_cumulative_model_relative_ppb: max,
            logical_width: 4,
            physical_width: 8,
            boundary: StaticNumericalBoundary::F32,
            finite_required: true,
            clamp: Some(StaticNumericalClamp {
                min_f32_bits: (-10.0_f32).to_bits(),
                max_f32_bits: 10.0_f32.to_bits(),
            }),
            secondary_clamp: None,
            model_error_bound_known: false,
        }
    }

    #[test]
    fn cumulative_contract_is_deterministic_and_keeps_quality_gate_visible() {
        let plan = compile_numerical_contract(&[
            contract_request(0, StaticNumericalAccumulatorStrategy::ExactI32, budget(32)),
            contract_request(
                1,
                StaticNumericalAccumulatorStrategy::CompensatedF32,
                budget(32),
            ),
        ])
        .unwrap();
        assert_eq!(plan.operations().len(), 2);
        assert_eq!(
            plan.operations()[0].kernel,
            StaticNumericalKernelFamily::ExactI32
        );
        assert_eq!(
            plan.operations()[0]
                .integer_range_proof
                .unwrap()
                .primary_accumulator_abs_max,
            60_960
        );
        assert_eq!(
            plan.operations()[0]
                .integer_range_proof
                .unwrap()
                .auxiliary_accumulator_abs_max,
            4_064
        );
        assert_eq!(plan.operations()[1].cumulative_absolute_q32, 3);
        assert!(plan.quality_gate_required);
        assert_ne!(plan.contract_digest_sha256(), [0; 32]);
        assert_eq!(
            plan.contract_digest_sha256(),
            compile_numerical_contract(&[
                contract_request(0, StaticNumericalAccumulatorStrategy::ExactI32, budget(32),),
                contract_request(
                    1,
                    StaticNumericalAccumulatorStrategy::CompensatedF32,
                    budget(32),
                ),
            ])
            .unwrap()
            .contract_digest_sha256()
        );
    }

    #[test]
    fn cumulative_contract_rejects_width_clamp_and_budget_drift() {
        let mut invalid_width = budget(32);
        invalid_width.logical_width = 9;
        assert_eq!(
            compile_numerical_contract(&[contract_request(
                0,
                StaticNumericalAccumulatorStrategy::NativeF32,
                invalid_width,
            )]),
            Err(StaticNumericalCompileError::InvalidBudget)
        );
        let mut one_sided_clamp = budget(32);
        one_sided_clamp.clamp = Some(StaticNumericalClamp {
            min_f32_bits: f32::NEG_INFINITY.to_bits(),
            max_f32_bits: 10.0_f32.to_bits(),
        });
        assert!(compile_numerical_contract(&[contract_request(
            0,
            StaticNumericalAccumulatorStrategy::NativeF32,
            one_sided_clamp,
        )])
        .is_ok());
        let mut two_input_clamp = budget(32);
        two_input_clamp.secondary_clamp = Some(StaticNumericalClamp {
            min_f32_bits: (-10.0_f32).to_bits(),
            max_f32_bits: 10.0_f32.to_bits(),
        });
        let primary_only_digest = compile_numerical_contract(&[contract_request(
            0,
            StaticNumericalAccumulatorStrategy::NativeF32,
            budget(32),
        )])
        .unwrap()
        .contract_digest_sha256();
        assert_ne!(
            compile_numerical_contract(&[contract_request(
                0,
                StaticNumericalAccumulatorStrategy::NativeF32,
                two_input_clamp,
            )])
            .unwrap()
            .contract_digest_sha256(),
            primary_only_digest
        );
        let mut invalid_clamp = budget(32);
        invalid_clamp.clamp = Some(StaticNumericalClamp {
            min_f32_bits: f32::NAN.to_bits(),
            max_f32_bits: 10.0_f32.to_bits(),
        });
        assert_eq!(
            compile_numerical_contract(&[contract_request(
                0,
                StaticNumericalAccumulatorStrategy::NativeF32,
                invalid_clamp,
            )]),
            Err(StaticNumericalCompileError::InvalidBudget)
        );
        assert_eq!(
            compile_numerical_contract(&[
                contract_request(0, StaticNumericalAccumulatorStrategy::NativeF32, budget(1),),
                contract_request(1, StaticNumericalAccumulatorStrategy::NativeF32, budget(1),),
            ]),
            Err(StaticNumericalCompileError::CumulativeBudgetExceeded)
        );
    }

    #[test]
    fn exact_i32_requires_a_proof_and_rejects_an_overflowing_range() {
        let mut missing =
            contract_request(0, StaticNumericalAccumulatorStrategy::ExactI32, budget(32));
        missing.integer_range_proof = None;
        assert_eq!(
            compile_numerical_contract(&[missing]),
            Err(StaticNumericalCompileError::MissingIntegerRangeProof)
        );

        let mut overflowing =
            contract_request(0, StaticNumericalAccumulatorStrategy::ExactI32, budget(32));
        overflowing.integer_range_proof = Some(StaticNumericalIntegerRangeProof {
            term_count: 32,
            primary_lhs_abs_max: u32::MAX,
            rhs_abs_max: 127,
            auxiliary_lhs_abs_max: 1,
        });
        assert_eq!(
            compile_numerical_contract(&[overflowing]),
            Err(StaticNumericalCompileError::IntegerRangeExceeded)
        );
    }

    #[test]
    fn every_cumulative_ceiling_is_bound_into_the_contract_digest() {
        let base = contract_request(0, StaticNumericalAccumulatorStrategy::NativeF32, budget(64));
        let base_digest = compile_numerical_contract(&[base])
            .unwrap()
            .contract_digest_sha256();
        for mutate in [
            |budget: &mut StaticNumericalErrorBudget| budget.max_cumulative_absolute_q32 += 1,
            |budget: &mut StaticNumericalErrorBudget| {
                budget.max_cumulative_arithmetic_relative_ppb += 1
            },
            |budget: &mut StaticNumericalErrorBudget| budget.max_cumulative_model_relative_ppb += 1,
        ] {
            let mut changed = base;
            mutate(&mut changed.error_budget);
            assert_ne!(
                compile_numerical_contract(&[changed])
                    .unwrap()
                    .contract_digest_sha256(),
                base_digest
            );
        }
    }
}
