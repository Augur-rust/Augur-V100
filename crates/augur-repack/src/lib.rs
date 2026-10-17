//! Compact-to-hot repacking.
//!
//! This crate owns transformations from compact quantization formats into
//! concrete hot layouts defined by `augur-kernel-api`.
//!
//! It does not own the hot-layout ABI itself and does not execute kernels.

pub mod dual_scale_cpu;
pub mod error;
pub mod expert_row32_cuda;
pub mod fp4_cpu;
pub mod iq2_xxs_cpu;
pub mod q2k_cpu;
pub mod q4k_cpu;
pub mod q4k_cuda;
pub mod q8_0_cpu;
pub mod q8k_cpu;
pub mod qk_cuda;

pub use dual_scale_cpu::*;
pub use error::*;
pub use expert_row32_cuda::*;
pub use fp4_cpu::*;
pub use iq2_xxs_cpu::*;
pub use q2k_cpu::*;
pub use q4k_cpu::*;
pub use q4k_cuda::*;
pub use q8_0_cpu::*;
pub use q8k_cpu::*;
pub use qk_cuda::*;
