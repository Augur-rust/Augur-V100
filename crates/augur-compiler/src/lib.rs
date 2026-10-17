//! Graph analysis consumption, physical lowering, and plan construction.
//!
//! This crate is the only production path from Graph IR to ExecutionPlan.
//! It may understand Graph semantics, backend capabilities, physical layouts,
//! kernel identity, liveness, memory planning, registry selection, compiled
//! resource requirements and resource binding.
//!
//! It does not execute kernels or call kernel inner loops.

mod activation_reuse;

pub mod bind;
pub mod compiled_plan;
pub mod compiler;
pub mod cpu_lowering;
#[cfg(feature = "cuda")]
pub mod cuda_bind;
#[cfg(feature = "cuda")]
pub mod cuda_compiled_plan;
#[cfg(feature = "cuda")]
pub mod cuda_lowering;
#[cfg(feature = "cuda")]
pub mod cuda_resources;
pub mod error;
pub mod kernel_select;
pub mod lowering;
pub mod m138_graph_lowering;
pub mod m141_fusion_candidate;
pub mod memory_planner;
pub mod resident_moe;
pub mod resources;
pub mod static_numerical;
pub mod static_route;
pub mod tuning;

pub use bind::*;
pub use compiled_plan::*;
pub use compiler::*;
pub use cpu_lowering::*;
#[cfg(feature = "cuda")]
pub use cuda_bind::*;
#[cfg(feature = "cuda")]
pub use cuda_compiled_plan::*;
#[cfg(feature = "cuda")]
pub use cuda_lowering::*;
#[cfg(feature = "cuda")]
pub use cuda_resources::*;
pub use error::*;
pub use kernel_select::*;
pub use lowering::*;
pub use m138_graph_lowering::*;
pub use m141_fusion_candidate::*;
pub use memory_planner::*;
pub use resident_moe::*;
pub use resources::*;
pub use static_numerical::*;
pub use static_route::*;
pub use tuning::*;
