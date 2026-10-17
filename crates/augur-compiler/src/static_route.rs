//! Compile source-bound MoE route reduction semantics into immutable policies.

pub use augur_graph::{ExpertAggregationOrder, RouteDenominatorOrder, StaticRouteOpRequest};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledStaticRouteOp {
    pub ordinal: u16,
    pub operator_identity_sha256: [u8; 32],
    pub denominator_order: RouteDenominatorOrder,
    pub compensated_denominator: bool,
    pub aggregation_order: ExpertAggregationOrder,
    pub compensated_aggregation: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledStaticRoutePlan {
    operations: Box<[CompiledStaticRouteOp]>,
}

impl CompiledStaticRoutePlan {
    pub fn operations(&self) -> &[CompiledStaticRouteOp] {
        &self.operations
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StaticRouteCompileError {
    EmptyPlan,
    OperationOrder,
    ZeroOperatorIdentity,
    DuplicateOperatorIdentity,
    InvalidCompensationOrder,
}

impl std::fmt::Display for StaticRouteCompileError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "static route compilation failed: {self:?}")
    }
}

impl std::error::Error for StaticRouteCompileError {}

pub fn compile_static_route_plan(
    requests: &[StaticRouteOpRequest],
) -> Result<CompiledStaticRoutePlan, StaticRouteCompileError> {
    if requests.is_empty() {
        return Err(StaticRouteCompileError::EmptyPlan);
    }
    let mut identities = std::collections::BTreeSet::new();
    let mut operations = Vec::with_capacity(requests.len());
    for (index, request) in requests.iter().enumerate() {
        if usize::from(request.ordinal) != index {
            return Err(StaticRouteCompileError::OperationOrder);
        }
        if request.operator_identity_sha256 == [0; 32] {
            return Err(StaticRouteCompileError::ZeroOperatorIdentity);
        }
        if !identities.insert(request.operator_identity_sha256) {
            return Err(StaticRouteCompileError::DuplicateOperatorIdentity);
        }
        if request.compensated_aggregation
            && request.aggregation_order != ExpertAggregationOrder::CanonicalExpertId
        {
            return Err(StaticRouteCompileError::InvalidCompensationOrder);
        }
        operations.push(CompiledStaticRouteOp {
            ordinal: request.ordinal,
            operator_identity_sha256: request.operator_identity_sha256,
            denominator_order: request.denominator_order,
            compensated_denominator: request.compensated_denominator,
            aggregation_order: request.aggregation_order,
            compensated_aggregation: request.compensated_aggregation,
        });
    }
    Ok(CompiledStaticRoutePlan {
        operations: operations.into_boxed_slice(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(ordinal: u16, compensated: bool) -> StaticRouteOpRequest {
        StaticRouteOpRequest {
            ordinal,
            operator_identity_sha256: [ordinal as u8 + 1; 32],
            denominator_order: RouteDenominatorOrder::CanonicalExpertId,
            compensated_denominator: false,
            aggregation_order: ExpertAggregationOrder::CanonicalExpertId,
            compensated_aggregation: compensated,
        }
    }

    #[test]
    fn route_semantics_are_lowered_without_runtime_choice() {
        let plan = compile_static_route_plan(&[request(0, false), request(1, true)]).unwrap();
        assert_eq!(
            plan.operations()[1].aggregation_order,
            ExpertAggregationOrder::CanonicalExpertId
        );
        assert!(plan.operations()[1].compensated_aggregation);
    }

    #[test]
    fn invalid_compensation_and_identity_drift_fail_closed() {
        let mut invalid = request(0, true);
        invalid.aggregation_order = ExpertAggregationOrder::RouteRank;
        assert_eq!(
            compile_static_route_plan(&[invalid]),
            Err(StaticRouteCompileError::InvalidCompensationOrder)
        );
        let mut duplicate = request(0, false);
        duplicate.ordinal = 1;
        assert_eq!(
            compile_static_route_plan(&[request(0, false), duplicate]),
            Err(StaticRouteCompileError::DuplicateOperatorIdentity)
        );
    }
}
