//! Kernel-facing contracts shared by lowering, repack, execution, and kernels.
//!
//! This crate owns:
//! - kernel identity;
//! - coarse shape/tile keys;
//! - concrete hot-layout data contracts;
//! - stable bound-launch function signatures;
//! - thin static-registration protocol types.
//!
//! It does not own kernel implementations, registry tables, selection policy,
//! Graph lowering or execution.

pub mod cpu_feature;
pub mod error;
pub mod hot;
pub mod key;
pub mod launch;
pub mod registry;

pub use cpu_feature::*;
pub use error::*;
pub use hot::*;
pub use key::*;
pub use launch::*;
pub use registry::*;
