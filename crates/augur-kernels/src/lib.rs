#![allow(clippy::needless_range_loop)]
#![deny(unsafe_op_in_unsafe_fn)]

//! Portable and architecture-specific production kernels.
//!
//! Production kernels consume typed Hot contracts defined by
//! `augur-kernel-api`. Compact and reference-format arithmetic belongs to
//! `augur-formats` and `augur-reference`; this crate does not have a normal
//! dependency on either crate.
//!
//! Static registration tables bind concrete kernel functions to protocol
//! metadata. Selection remains owned by `augur-compiler`.

pub mod error;
pub mod portable;
pub mod registrations;

#[cfg(target_arch = "x86_64")]
pub mod x86;

pub use error::*;
pub use registrations::cpu_kernel_registrations;
