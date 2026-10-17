use augur_core::{CompileError, DType, LogicalLayout, NodeId, StorageRef, TensorDesc, TensorId};
use smallvec::smallvec;

use crate::{
    BroadcastAddAttrs, ClampAttrs, ConcatAttrs, FusedMlpAttrs, HadamardAttrs, Node, Op,
    OrderedTopKAttrs, PadAttrs, PrecisionBoundaryAttrs, QMatmulAttrs, QMatmulEpilogue, QdqAttrs,
    ReduceSumAttrs, ReshapeAttrs, RmsNormAttrs, RopeAttrs, SliceAttrs, SoftmaxAttrs,
};

#[derive(Clone, Debug)]
pub struct TensorSlot {
    pub id: TensorId,
    pub desc: TensorDesc,
}

#[derive(Clone, Debug)]
pub struct NodeSlot {
    pub id: NodeId,
    pub node: Node,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    tensors: Vec<TensorSlot>,
    nodes: Vec<NodeSlot>,
    inputs: Vec<TensorId>,
    outputs: Vec<TensorId>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_tensor(&mut self, desc: TensorDesc) -> TensorId {
        let id = TensorId::new(self.tensors.len(), 0);
        self.tensors.push(TensorSlot { id, desc });
        id
    }

    pub fn add_input(&mut self, desc: TensorDesc) -> TensorId {
        let id = self.add_tensor(desc);
        self.inputs.push(id);
        id
    }

    pub fn add_weight(&mut self, desc: TensorDesc) -> Result<TensorId, CompileError> {
        validate_weight_desc_for_builder(&desc)?;
        Ok(self.add_tensor(desc))
    }

    // ── Add ─────────────────────────────────────────────────────────

    pub fn add_add(&mut self, a: TensorId, b: TensorId, out_desc: TensorDesc) -> TensorId {
        let out = self.add_tensor(out_desc);
        let id = NodeId::new(self.nodes.len(), 0);
        let node = Node {
            id,
            op: Op::Add,
            inputs: smallvec![a, b],
            outputs: smallvec![out],
        };
        self.nodes.push(NodeSlot { id, node });
        out
    }

    pub fn add_mul(&mut self, a: TensorId, b: TensorId, out_desc: TensorDesc) -> TensorId {
        let out = self.add_tensor(out_desc);
        let id = NodeId::new(self.nodes.len(), 0);
        self.nodes.push(NodeSlot {
            id,
            node: Node {
                id,
                op: Op::Mul,
                inputs: smallvec![a, b],
                outputs: smallvec![out],
            },
        });
        out
    }

    pub fn add_clamp(
        &mut self,
        input: TensorId,
        min: Option<f32>,
        max: Option<f32>,
        out_desc: TensorDesc,
    ) -> Result<TensorId, CompileError> {
        let attrs = ClampAttrs::try_new(min, max)?;
        let out = self.add_tensor(out_desc);
        let id = NodeId::new(self.nodes.len(), 0);
        self.nodes.push(NodeSlot {
            id,
            node: Node {
                id,
                op: Op::Clamp(attrs),
                inputs: smallvec![input],
                outputs: smallvec![out],
            },
        });
        Ok(out)
    }

    pub fn add_silu(&mut self, input: TensorId, out_desc: TensorDesc) -> TensorId {
        let out = self.add_tensor(out_desc);
        let id = NodeId::new(self.nodes.len(), 0);
        self.nodes.push(NodeSlot {
            id,
            node: Node {
                id,
                op: Op::Silu,
                inputs: smallvec![input],
                outputs: smallvec![out],
            },
        });
        out
    }

    /// Adds a generic gated MLP with input order `x, gate, up, down`.
    pub fn add_fused_mlp(
        &mut self,
        input: TensorId,
        gate_weight: TensorId,
        up_weight: TensorId,
        down_weight: TensorId,
        attrs: FusedMlpAttrs,
        out_desc: TensorDesc,
    ) -> TensorId {
        let out = self.add_tensor(out_desc);
        let id = NodeId::new(self.nodes.len(), 0);
        self.nodes.push(NodeSlot {
            id,
            node: Node {
                id,
                op: Op::FusedMlp(attrs),
                inputs: smallvec![input, gate_weight, up_weight, down_weight],
                outputs: smallvec![out],
            },
        });
        out
    }

    // ── QMatmul (Identity) ───────────────────────────────────────────

    pub fn add_qmatmul(
        &mut self,
        activation: TensorId,
        weight: TensorId,
        out_desc: TensorDesc,
    ) -> TensorId {
        let out = self.add_tensor(out_desc);
        let id = NodeId::new(self.nodes.len(), 0);
        let node = Node {
            id,
            op: Op::QMatmul(QMatmulAttrs {
                epilogue: QMatmulEpilogue::Identity,
            }),
            inputs: smallvec![activation, weight],
            outputs: smallvec![out],
        };
        self.nodes.push(NodeSlot { id, node });
        out
    }

    // ── QMatmul + Bias ──────────────────────────────────────────────

    pub fn add_qmatmul_bias(
        &mut self,
        activation: TensorId,
        weight: TensorId,
        bias: TensorId,
        out_desc: TensorDesc,
    ) -> TensorId {
        let out = self.add_tensor(out_desc);
        let id = NodeId::new(self.nodes.len(), 0);
        let node = Node {
            id,
            op: Op::QMatmul(QMatmulAttrs {
                epilogue: QMatmulEpilogue::Bias,
            }),
            inputs: smallvec![activation, weight, bias],
            outputs: smallvec![out],
        };
        self.nodes.push(NodeSlot { id, node });
        out
    }

    // ── QMatmul + Bias + SiLU ───────────────────────────────────────

    pub fn add_qmatmul_bias_silu(
        &mut self,
        activation: TensorId,
        weight: TensorId,
        bias: TensorId,
        out_desc: TensorDesc,
    ) -> TensorId {
        let out = self.add_tensor(out_desc);
        let id = NodeId::new(self.nodes.len(), 0);
        let node = Node {
            id,
            op: Op::QMatmul(QMatmulAttrs {
                epilogue: QMatmulEpilogue::BiasSilu,
            }),
            inputs: smallvec![activation, weight, bias],
            outputs: smallvec![out],
        };
        self.nodes.push(NodeSlot { id, node });
        out
    }

    pub fn add_matmul(
        &mut self,
        activation: TensorId,
        weight: TensorId,
        out_desc: TensorDesc,
    ) -> TensorId {
        self.add_op(Op::Matmul, &[activation, weight], out_desc)
    }

    pub fn add_reshape(&mut self, input: TensorId, out_desc: TensorDesc) -> TensorId {
        self.add_op(Op::Reshape(ReshapeAttrs), &[input], out_desc)
    }

    pub fn add_slice(
        &mut self,
        input: TensorId,
        attrs: SliceAttrs,
        out_desc: TensorDesc,
    ) -> TensorId {
        self.add_op(Op::Slice(attrs), &[input], out_desc)
    }

    pub fn add_concat(
        &mut self,
        inputs: &[TensorId],
        attrs: ConcatAttrs,
        out_desc: TensorDesc,
    ) -> TensorId {
        self.add_op(Op::Concat(attrs), inputs, out_desc)
    }

    pub fn add_pad(&mut self, input: TensorId, attrs: PadAttrs, out_desc: TensorDesc) -> TensorId {
        self.add_op(Op::Pad(attrs), &[input], out_desc)
    }

    pub fn add_broadcast_add(
        &mut self,
        left: TensorId,
        right: TensorId,
        out_desc: TensorDesc,
    ) -> TensorId {
        self.add_op(
            Op::BroadcastAdd(BroadcastAddAttrs),
            &[left, right],
            out_desc,
        )
    }

    pub fn add_softmax(
        &mut self,
        input: TensorId,
        attrs: SoftmaxAttrs,
        out_desc: TensorDesc,
    ) -> TensorId {
        self.add_op(Op::Softmax(attrs), &[input], out_desc)
    }

    pub fn add_reduce_sum(
        &mut self,
        input: TensorId,
        attrs: ReduceSumAttrs,
        out_desc: TensorDesc,
    ) -> TensorId {
        self.add_op(Op::ReduceSum(attrs), &[input], out_desc)
    }

    pub fn add_precision_boundary(
        &mut self,
        input: TensorId,
        attrs: PrecisionBoundaryAttrs,
        out_desc: TensorDesc,
    ) -> TensorId {
        self.add_op(Op::PrecisionBoundary(attrs), &[input], out_desc)
    }

    pub fn add_rms_norm(
        &mut self,
        input: TensorId,
        weight: TensorId,
        attrs: RmsNormAttrs,
        out_desc: TensorDesc,
    ) -> TensorId {
        self.add_op(Op::RmsNorm(attrs), &[input, weight], out_desc)
    }

    pub fn add_rope(
        &mut self,
        input: TensorId,
        attrs: RopeAttrs,
        out_desc: TensorDesc,
    ) -> TensorId {
        self.add_op(Op::Rope(attrs), &[input], out_desc)
    }

    pub fn add_hadamard(
        &mut self,
        input: TensorId,
        attrs: HadamardAttrs,
        out_desc: TensorDesc,
    ) -> TensorId {
        self.add_op(Op::Hadamard(attrs), &[input], out_desc)
    }

    pub fn add_qdq(&mut self, input: TensorId, attrs: QdqAttrs, out_desc: TensorDesc) -> TensorId {
        self.add_op(Op::Qdq(attrs), &[input], out_desc)
    }

    pub fn add_ordered_topk(
        &mut self,
        input: TensorId,
        attrs: OrderedTopKAttrs,
        indices_desc: TensorDesc,
        scores_desc: TensorDesc,
    ) -> (TensorId, TensorId) {
        let indices = self.add_tensor(indices_desc);
        let scores = self.add_tensor(scores_desc);
        let id = NodeId::new(self.nodes.len(), 0);
        self.nodes.push(NodeSlot {
            id,
            node: Node {
                id,
                op: Op::OrderedTopK(attrs),
                inputs: smallvec![input],
                outputs: smallvec![indices, scores],
            },
        });
        (indices, scores)
    }

    fn add_op(&mut self, op: Op, inputs: &[TensorId], out_desc: TensorDesc) -> TensorId {
        let out = self.add_tensor(out_desc);
        let id = NodeId::new(self.nodes.len(), 0);
        self.nodes.push(NodeSlot {
            id,
            node: Node {
                id,
                op,
                inputs: smallvec::SmallVec::from_slice(inputs),
                outputs: smallvec![out],
            },
        });
        out
    }

    // ── Output / accessors ──────────────────────────────────────────

    pub fn try_set_output(&mut self, tensor: TensorId) -> Result<(), CompileError> {
        self.tensor(tensor).ok_or(CompileError::InvalidOutput)?;
        if !self.outputs.contains(&tensor) {
            self.outputs.push(tensor);
        }
        Ok(())
    }

    pub fn tensor(&self, id: TensorId) -> Option<&TensorDesc> {
        let slot = self.tensors.get(id.index as usize)?;
        if slot.id == id {
            Some(&slot.desc)
        } else {
            None
        }
    }

    pub fn tensor_slot(&self, id: TensorId) -> Option<&TensorSlot> {
        let slot = self.tensors.get(id.index as usize)?;
        if slot.id == id {
            Some(slot)
        } else {
            None
        }
    }

    pub fn node(&self, id: NodeId) -> Option<&NodeSlot> {
        let slot = self.nodes.get(id.index as usize)?;
        if slot.id == id && slot.node.id == id {
            Some(slot)
        } else {
            None
        }
    }

    pub fn tensors(&self) -> &[TensorSlot] {
        &self.tensors
    }
    pub fn nodes(&self) -> &[NodeSlot] {
        &self.nodes
    }
    pub fn inputs(&self) -> &[TensorId] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[TensorId] {
        &self.outputs
    }
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

fn validate_weight_desc_for_builder(desc: &TensorDesc) -> Result<(), CompileError> {
    if desc.dtype != DType::PackedU8 {
        return Err(CompileError::InvalidInput);
    }
    if desc.shape.rank() != 2 {
        return Err(CompileError::InvalidInput);
    }
    let dims = desc.shape.dims();
    if dims[0] == 0 || dims[1] == 0 {
        return Err(CompileError::InvalidInput);
    }
    if !matches!(desc.layout, LogicalLayout::Contiguous) {
        return Err(CompileError::InvalidInput);
    }
    let quant = desc.quant.as_ref().ok_or(CompileError::InvalidInput)?;
    if quant.block == 0 || quant.version == 0 {
        return Err(CompileError::InvalidInput);
    }
    if matches!(desc.storage, StorageRef::None) {
        return Err(CompileError::InvalidInput);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate_graph;
    use augur_core::{
        CompileError, GgmlError, LogicalLayout, QuantDesc, QuantFormatId, Shape, StorageId,
    };

    #[test]
    fn node_lookup_rejects_stale_generation() {
        let mut g = Graph::new();
        let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[2]));
        let a = g.add_input(desc.clone());
        let b = g.add_input(desc.clone());
        g.add_add(a, b, desc);
        let valid = NodeId::new(0, 0);
        let stale = NodeId::new(0, 1);
        assert!(g.node(valid).is_some());
        assert!(g.node(stale).is_none());
    }

    #[test]
    fn try_set_output_is_idempotent() {
        let mut g = Graph::new();
        let tensor = g.add_tensor(TensorDesc::f32_contiguous(Shape::from_dims(&[2])));
        g.try_set_output(tensor).unwrap();
        g.try_set_output(tensor).unwrap();
        assert_eq!(g.outputs(), &[tensor]);
    }

    #[test]
    fn validate_accepts_graph_input_as_output() {
        let mut graph = Graph::new();
        let input = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[2])));
        graph.try_set_output(input).unwrap();
        assert_eq!(validate_graph(&graph), Ok(()));
    }

    #[test]
    fn validate_rejects_duplicate_input_role() {
        let mut graph = Graph::new();
        let input = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[2])));
        graph.inputs.push(input);
        assert_eq!(validate_graph(&graph), Err(CompileError::InvalidInput));
    }

    #[test]
    fn validate_rejects_duplicate_output_role() {
        let mut graph = Graph::new();
        let output = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[2])));
        graph.try_set_output(output).unwrap();
        graph.outputs.push(output);
        assert_eq!(validate_graph(&graph), Err(CompileError::InvalidOutput));
    }

    #[test]
    fn try_set_output_rejects_invalid_tensor() {
        let mut g = Graph::new();
        let invalid = TensorId {
            index: 999,
            generation: 0,
        };
        assert_eq!(g.try_set_output(invalid), Err(CompileError::InvalidOutput));
    }

    #[test]
    fn try_set_output_rejects_stale_generation() {
        let mut g = Graph::new();
        let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[2]));
        let t = g.add_tensor(desc);
        let stale = TensorId {
            index: t.index,
            generation: t.generation + 1,
        };
        assert_eq!(g.try_set_output(stale), Err(CompileError::InvalidOutput));
    }

    #[test]
    fn validate_rejects_output_without_source() {
        let mut graph = Graph::new();
        let output = graph.add_tensor(TensorDesc::f32_contiguous(Shape::from_dims(&[2])));
        graph.try_set_output(output).unwrap();
        assert_eq!(validate_graph(&graph), Err(CompileError::InvalidOutput));
    }

    #[test]
    fn validate_rejects_node_input_without_source() {
        let mut graph = Graph::new();
        let left = graph.add_tensor(TensorDesc::f32_contiguous(Shape::from_dims(&[2])));
        let right = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[2])));
        let output = graph.add_add(
            left,
            right,
            TensorDesc::f32_contiguous(Shape::from_dims(&[2])),
        );
        graph.try_set_output(output).unwrap();
        assert_eq!(validate_graph(&graph), Err(CompileError::InvalidInput));
    }

    #[test]
    fn validate_accepts_nested_view_inheriting_input_source() {
        let mut graph = Graph::new();
        let base = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[8])));
        let first = graph.add_tensor(TensorDesc {
            dtype: DType::F32,
            shape: Shape::from_dims(&[6]),
            layout: LogicalLayout::View { base, offset: 4 },
            storage: StorageRef::None,
            quant: None,
        });
        let second = graph.add_tensor(TensorDesc {
            dtype: DType::F32,
            shape: Shape::from_dims(&[2]),
            layout: LogicalLayout::View {
                base: first,
                offset: 8,
            },
            storage: StorageRef::None,
            quant: None,
        });
        graph.try_set_output(second).unwrap();
        assert_eq!(validate_graph(&graph), Ok(()));
    }

    #[test]
    fn validate_rejects_view_cycle() {
        let mut graph = Graph::new();
        let first = graph.add_tensor(TensorDesc::f32_contiguous(Shape::from_dims(&[1])));
        let second = graph.add_tensor(TensorDesc {
            dtype: DType::F32,
            shape: Shape::from_dims(&[1]),
            layout: LogicalLayout::View {
                base: first,
                offset: 0,
            },
            storage: StorageRef::None,
            quant: None,
        });
        graph.tensors[first.index as usize].desc.layout = LogicalLayout::View {
            base: second,
            offset: 0,
        };
        assert_eq!(validate_graph(&graph), Err(CompileError::Cycle));
    }

    #[test]
    fn validate_rejects_view_bounds_overflow() {
        let mut graph = Graph::new();
        let base = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[2])));
        graph.add_tensor(TensorDesc {
            dtype: DType::F32,
            shape: Shape::from_dims(&[1]),
            layout: LogicalLayout::View {
                base,
                offset: usize::MAX,
            },
            storage: StorageRef::None,
            quant: None,
        });
        assert_eq!(validate_graph(&graph), Err(CompileError::InvalidInput));
    }

    #[test]
    fn validate_rejects_view_dtype_and_quant_mismatch() {
        let mut graph = Graph::new();
        let base = graph.add_tensor(TensorDesc::quantized_weight(
            Shape::from_dims(&[1]),
            StorageRef::HostOwned {
                id: StorageId::new(0, 0),
            },
            QuantFormatId::Q4K,
            256,
            1,
        ));
        graph.add_tensor(TensorDesc {
            dtype: DType::PackedU8,
            shape: Shape::from_dims(&[1]),
            layout: LogicalLayout::View { base, offset: 0 },
            storage: StorageRef::None,
            quant: Some(QuantDesc {
                format: QuantFormatId::DualScale,
                block: 256,
                version: 1,
            }),
        });
        assert_eq!(
            validate_graph(&graph),
            Err(CompileError::Graph(GgmlError::DTypeMismatch))
        );
    }

    #[test]
    fn qmatmul_with_missing_activation_is_rejected() {
        let mut graph = Graph::new();

        let weight_desc = TensorDesc::quantized_weight(
            Shape::from_dims(&[2, 256]),
            StorageRef::HostOwned {
                id: StorageId::new(0, 0),
            },
            augur_core::QuantFormatId::Q4K,
            256,
            1,
        );

        let weight = graph.add_weight(weight_desc).unwrap();
        let output = graph.add_tensor(TensorDesc::f32_contiguous(Shape::from_dims(&[2])));

        let node_id = NodeId::new(0, 0);
        let node = Node {
            id: node_id,
            op: Op::QMatmul(QMatmulAttrs {
                epilogue: QMatmulEpilogue::Identity,
            }),
            inputs: smallvec![weight],
            outputs: smallvec![output],
        };
        graph.nodes.push(NodeSlot { id: node_id, node });
        assert_eq!(validate_graph(&graph), Err(CompileError::InvalidInput));
    }

    #[test]
    fn qmatmul_bias_requires_third_input() {
        let mut graph = Graph::new();

        let activation = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[256])));

        let weight_desc = TensorDesc::quantized_weight(
            Shape::from_dims(&[2, 256]),
            StorageRef::HostOwned {
                id: StorageId::new(0, 0),
            },
            augur_core::QuantFormatId::Q4K,
            256,
            1,
        );

        let weight = graph.add_weight(weight_desc).unwrap();
        let output = graph.add_tensor(TensorDesc::f32_contiguous(Shape::from_dims(&[2])));

        let node_id = NodeId::new(0, 0);
        graph.nodes.push(NodeSlot {
            id: node_id,
            node: Node {
                id: node_id,
                op: Op::QMatmul(QMatmulAttrs {
                    epilogue: QMatmulEpilogue::Bias,
                }),
                inputs: smallvec![activation, weight],
                outputs: smallvec![output],
            },
        });
        assert_eq!(validate_graph(&graph), Err(CompileError::InvalidInput));
    }
}
