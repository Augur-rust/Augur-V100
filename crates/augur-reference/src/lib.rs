//! Reference execution and mathematical correctness layer.
//!
//! This crate owns correctness oracles used by tests, fuzzing, Miri,
//! numerical comparisons and backend validation. It must never appear in a
//! production execution dependency path.

pub mod dual_scale_q8k;
pub mod error;
pub mod executor;
pub mod iq2_xxs_q8k;
pub mod m43_mixed_q8k;
pub mod m46_fp4;
pub mod q4k_f32_gemm;
pub mod q4k_q8k;
pub mod resident_moe;

pub use dual_scale_q8k::*;
pub use error::*;
pub use executor::*;
pub use iq2_xxs_q8k::*;
pub use m43_mixed_q8k::*;
pub use m46_fp4::*;
pub use q4k_f32_gemm::*;
pub use q4k_q8k::*;
pub use resident_moe::*;
