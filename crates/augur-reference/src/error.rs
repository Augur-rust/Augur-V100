//! Errors produced by reference-only quantized arithmetic.
//!
//! These errors belong to the reference layer. They must not enter
//! production lowering, ExecutionPlan or kernel launch contracts.

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReferenceQ8Error {
    BadWeight,
    BadWeightShape,
    BadWeightBlockCount { expected: usize, actual: usize },
    BadActivationLength { expected: usize, actual: usize },
    BadActivationBlockCount { expected: usize, actual: usize },
    BadOutputLength { expected: usize, actual: usize },
    BadBiasLength { expected: usize, actual: usize },
}
