use augur_backend::CpuBackend;
use augur_exec::ExecutionPlan;
use augur_graph::Graph;

use crate::{CompilerError, GraphCompiler};

pub fn compile_cpu_add_only(
    graph: &Graph,
    input_buffers: Vec<Vec<f32>>,
) -> Result<ExecutionPlan<CpuBackend>, CompilerError> {
    GraphCompiler::cpu_baseline().compile_cpu(graph, input_buffers)
}
