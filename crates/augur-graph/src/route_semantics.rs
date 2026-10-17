//! Backend-independent MoE route reduction semantics.

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteDenominatorOrder {
    RouteRank,
    CanonicalExpertId,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertAggregationOrder {
    RouteRank,
    CanonicalExpertId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StaticRouteOpRequest {
    pub ordinal: u16,
    pub operator_identity_sha256: [u8; 32],
    pub denominator_order: RouteDenominatorOrder,
    pub compensated_denominator: bool,
    pub aggregation_order: ExpertAggregationOrder,
    pub compensated_aggregation: bool,
}
