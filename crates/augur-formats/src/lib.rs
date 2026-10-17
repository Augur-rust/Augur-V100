//! Compact quantization format semantics.
//!
//! This crate owns compact block layouts, bit-exact reference unpacking,
//! and reference encode/decode behavior.
//!
//! It does not own backend hot layouts, SIMD lane layouts, repack,
//! ExecutionPlan lowering, or kernel execution.

#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_lifetimes,
    clippy::needless_range_loop,
    clippy::manual_div_ceil
)]

pub mod dual_scale;
pub mod error;
pub mod fp4;
pub mod iq2_xxs;
pub mod outlier_csr;
pub mod q2k;
pub mod q4k_compat;
pub mod q5k_compat;
pub mod q6k_compat;
pub mod q8_0;
pub mod q8k;
pub mod reference;

pub use dual_scale::{
    encode_dual_scale_block_ref, validate_dual_scale_compact_block, DualScaleCompactBlock,
    DUAL_SCALE_BLOCK_SIZE, DUAL_SCALE_GROUPS, DUAL_SCALE_GROUP_SIZE, DUAL_SCALE_MAX_OUTLIERS,
};
pub use error::*;
pub use fp4::*;
pub use iq2_xxs::*;
pub use outlier_csr::{
    add_outlier_csr_correction_ref, extract_dual_scale_outlier_csr_ref, OutlierCsrRef,
};
pub use q2k::*;
pub use q4k_compat::*;
pub use q5k_compat::*;
pub use q6k_compat::*;
pub use q8_0::*;
pub use q8k::*;
pub use reference::*;
