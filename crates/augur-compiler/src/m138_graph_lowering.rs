//! M138 typed, staged lowering for the generic ratio-4 Graph IR.
//!
//! This module deliberately stops at an immutable plan. It does not select a
//! runtime precision path, call CUDA, mutate transaction state or fuse nodes.

use std::collections::HashSet;

use augur_core::{CompileError, DType, QuantFormatId, StorageRef, TensorId};
use augur_exec::{BufferPurpose, MemoryPlan};
use augur_graph::{analyze_graph, BoundaryId, Graph, Node, Op};

use crate::MemoryPlanner;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum M138StepKind {
    Q6KMatmul,
    Bf16Matmul,
    Reshape,
    Slice,
    Concat,
    Pad,
    BroadcastAdd,
    Softmax,
    Mul,
    ReduceSum,
    Bf16Boundary,
    RmsNorm,
    Rope,
    Hadamard,
    Fp8Qdq,
    Fp4Qdq,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M138Step {
    pub ordinal: usize,
    pub node: augur_core::NodeId,
    pub kind: M138StepKind,
    pub inputs: Vec<TensorId>,
    pub output: TensorId,
    pub boundary: Option<BoundaryId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M138ConstantRequirement {
    pub tensor: TensorId,
    pub storage: StorageRef,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub quant: Option<QuantFormatId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M138CompiledPlan {
    steps: Vec<M138Step>,
    memory: MemoryPlan,
    constants: Vec<M138ConstantRequirement>,
    boundaries: Vec<BoundaryId>,
    graph_node_count: usize,
    graph_tensor_count: usize,
}

impl M138CompiledPlan {
    pub fn steps(&self) -> &[M138Step] {
        &self.steps
    }

    pub fn memory(&self) -> &MemoryPlan {
        &self.memory
    }

    pub fn constants(&self) -> &[M138ConstantRequirement] {
        &self.constants
    }

    pub fn boundaries(&self) -> &[BoundaryId] {
        &self.boundaries
    }

    pub fn graph_node_count(&self) -> usize {
        self.graph_node_count
    }

    pub fn graph_tensor_count(&self) -> usize {
        self.graph_tensor_count
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum M138LoweringError {
    Graph(CompileError),
    UnsupportedOp {
        node: augur_core::NodeId,
        op: &'static str,
    },
    InvalidNodeShape {
        node: augur_core::NodeId,
    },
    InvalidWeight {
        tensor: TensorId,
        reason: &'static str,
    },
    InvalidStorage {
        tensor: TensorId,
    },
}

impl std::fmt::Display for M138LoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Graph(error) => write!(f, "graph analysis failed: {error:?}"),
            Self::UnsupportedOp { node, op } => write!(f, "M138 unsupported op {op} at {node:?}"),
            Self::InvalidNodeShape { node } => write!(f, "M138 node shape differs at {node:?}"),
            Self::InvalidWeight { tensor, reason } => {
                write!(f, "M138 invalid weight {tensor:?}: {reason}")
            }
            Self::InvalidStorage { tensor } => write!(f, "M138 invalid storage for {tensor:?}"),
        }
    }
}

impl std::error::Error for M138LoweringError {}

impl From<CompileError> for M138LoweringError {
    fn from(value: CompileError) -> Self {
        Self::Graph(value)
    }
}

fn op_name(op: &Op) -> &'static str {
    match op {
        Op::QMatmul(_) => "QMatmul",
        Op::Matmul => "Matmul",
        Op::Reshape(_) => "Reshape",
        Op::Slice(_) => "Slice",
        Op::Concat(_) => "Concat",
        Op::Pad(_) => "Pad",
        Op::BroadcastAdd(_) => "BroadcastAdd",
        Op::Softmax(_) => "Softmax",
        Op::Mul => "Mul",
        Op::ReduceSum(_) => "ReduceSum",
        Op::PrecisionBoundary(_) => "PrecisionBoundary",
        Op::RmsNorm(_) => "RmsNorm",
        Op::Rope(_) => "RoPE",
        Op::Hadamard(_) => "Hadamard",
        Op::Qdq(_) => "QDQ",
        Op::OrderedTopK(_) => "OrderedTopK",
        Op::Add => "Add",
        Op::Clamp(_) => "Clamp",
        Op::Silu => "Silu",
        Op::FusedMlp(_) => "FusedMlp",
    }
}

fn boundary(op: &Op) -> Option<BoundaryId> {
    match op {
        Op::Softmax(attrs) => attrs.boundary,
        Op::PrecisionBoundary(attrs) => Some(attrs.boundary),
        Op::RmsNorm(attrs) => Some(attrs.boundary),
        Op::Rope(attrs) => Some(attrs.boundary),
        Op::Hadamard(attrs) => Some(attrs.boundary),
        Op::Qdq(attrs) => Some(attrs.boundary),
        _ => None,
    }
}

fn classify(node: &Node) -> Result<M138StepKind, M138LoweringError> {
    let kind = match &node.op {
        Op::QMatmul(attrs) if attrs.epilogue == augur_graph::QMatmulEpilogue::Identity => {
            M138StepKind::Q6KMatmul
        }
        Op::Matmul => M138StepKind::Bf16Matmul,
        Op::Reshape(_) => M138StepKind::Reshape,
        Op::Slice(_) => M138StepKind::Slice,
        Op::Concat(_) => M138StepKind::Concat,
        Op::Pad(_) => M138StepKind::Pad,
        Op::BroadcastAdd(_) => M138StepKind::BroadcastAdd,
        Op::Softmax(_) => M138StepKind::Softmax,
        Op::Mul => M138StepKind::Mul,
        Op::ReduceSum(_) => M138StepKind::ReduceSum,
        Op::PrecisionBoundary(attrs) if attrs.target == DType::BF16 => M138StepKind::Bf16Boundary,
        Op::RmsNorm(_) => M138StepKind::RmsNorm,
        Op::Rope(_) => M138StepKind::Rope,
        Op::Hadamard(_) => M138StepKind::Hadamard,
        Op::Qdq(attrs) => match attrs.format {
            augur_graph::QdqFormat::Fp8E4M3 => M138StepKind::Fp8Qdq,
            augur_graph::QdqFormat::Fp4E2M1 => M138StepKind::Fp4Qdq,
        },
        _ => {
            return Err(M138LoweringError::UnsupportedOp {
                node: node.id,
                op: op_name(&node.op),
            })
        }
    };
    if node.inputs.is_empty() || node.outputs.len() != 1 {
        return Err(M138LoweringError::InvalidNodeShape { node: node.id });
    }
    Ok(kind)
}

fn require_weight(graph: &Graph, node: &Node, kind: M138StepKind) -> Result<(), M138LoweringError> {
    if !matches!(kind, M138StepKind::Q6KMatmul | M138StepKind::Bf16Matmul) {
        return Ok(());
    }
    let tensor = node
        .inputs
        .get(1)
        .copied()
        .ok_or(M138LoweringError::InvalidNodeShape { node: node.id })?;
    let desc = graph
        .tensor(tensor)
        .ok_or(M138LoweringError::InvalidWeight {
            tensor,
            reason: "weight tensor missing",
        })?;
    if desc.storage.is_none() {
        return Err(M138LoweringError::InvalidStorage { tensor });
    }
    match kind {
        M138StepKind::Q6KMatmul => {
            if !desc.is_quant_format(QuantFormatId::Q6K) {
                return Err(M138LoweringError::InvalidWeight {
                    tensor,
                    reason: "Q6_K required",
                });
            }
        }
        M138StepKind::Bf16Matmul => {
            if desc.dtype != DType::BF16 || desc.quant.is_some() {
                return Err(M138LoweringError::InvalidWeight {
                    tensor,
                    reason: "BF16 source required",
                });
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// Lower a validated graph without executing it. The resulting plan is fully
/// staged: every node is represented exactly once and every temporary has a
/// deterministic liveness-backed allocation.
pub fn lower_m138_graph(graph: &Graph) -> Result<M138CompiledPlan, M138LoweringError> {
    let analyzed = analyze_graph(graph)?;
    let mut steps = Vec::with_capacity(analyzed.topo().len());
    let mut boundaries = Vec::new();
    for (ordinal, &node_id) in analyzed.topo().iter().enumerate() {
        let slot = graph.node(node_id).ok_or(CompileError::InvalidNode)?;
        let kind = classify(&slot.node)?;
        require_weight(graph, &slot.node, kind)?;
        let output = *slot
            .node
            .outputs
            .first()
            .ok_or(M138LoweringError::InvalidNodeShape { node: node_id })?;
        if let Some(id) = boundary(&slot.node.op) {
            if !boundaries.contains(&id) {
                boundaries.push(id);
            }
        }
        steps.push(M138Step {
            ordinal,
            node: node_id,
            kind,
            inputs: slot.node.inputs.iter().copied().collect(),
            output,
            boundary: boundary(&slot.node.op),
        });
    }

    let produced: HashSet<TensorId> = steps.iter().map(|step| step.output).collect();
    let outputs: HashSet<TensorId> = graph.outputs().iter().copied().collect();
    let mut constants = Vec::new();
    for slot in graph.tensors() {
        if !slot.desc.storage.is_none() {
            constants.push(M138ConstantRequirement {
                tensor: slot.id,
                storage: slot.desc.storage.clone(),
                dtype: slot.desc.dtype,
                shape: slot.desc.shape.dims().to_vec(),
                quant: slot.desc.quant.as_ref().map(|value| value.format),
            });
        }
    }
    constants.sort_by_key(|value| value.tensor.index);

    let mut planner = MemoryPlanner::new(graph.tensor_count());
    let mut alloc_order = Vec::new();
    for slot in graph.tensors() {
        if !slot.desc.storage.is_none() {
            continue;
        }
        let range =
            analyzed
                .liveness()
                .range(slot.id)
                .ok_or(M138LoweringError::InvalidNodeShape {
                    node: augur_core::NodeId::new(slot.id.index as usize, slot.id.generation),
                })?;
        alloc_order.push((range.first_step, slot.id.index, slot.id, range.last_step));
    }
    alloc_order.sort_by_key(|(first, index, _, _)| (*first, *index));
    for (_, _, tensor, last) in alloc_order {
        let desc = graph.tensor(tensor).ok_or(CompileError::InvalidInput)?;
        let elements = desc.elem_count().ok_or(CompileError::InvalidInput)?;
        let is_input = !produced.contains(&tensor);
        let is_output = outputs.contains(&tensor);
        let purpose = if is_output {
            BufferPurpose::Output
        } else if is_input {
            BufferPurpose::Input
        } else {
            BufferPurpose::Intermediate
        };
        if purpose == BufferPurpose::Intermediate {
            let range = analyzed.liveness().range(tensor).unwrap();
            planner.assign_reusing_tensor(tensor, elements, purpose, range.first_step, last)?;
        } else {
            planner.assign(tensor, elements, purpose, Some(last))?;
        }
    }

    Ok(M138CompiledPlan {
        steps,
        memory: planner.finish(),
        constants,
        boundaries,
        graph_node_count: graph.node_count(),
        graph_tensor_count: graph.tensor_count(),
    })
}
