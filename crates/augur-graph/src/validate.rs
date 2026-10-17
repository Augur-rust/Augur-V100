use augur_core::{CompileError, DType, GgmlError, LogicalLayout, NodeId, TensorId};

use crate::{ClampAttrs, FusedMlpAttrs, Graph, Node, Op, QMatmulAttrs, QMatmulEpilogue};

#[derive(Clone, Copy, Debug, Default)]
struct GraphIoRole {
    is_input: bool,
    is_output: bool,
}

#[derive(Debug)]
pub(crate) struct ValidatedGraph<'g> {
    graph: &'g Graph,
    producers: Vec<Option<NodeId>>,
}

impl<'g> ValidatedGraph<'g> {
    pub(crate) fn graph(&self) -> &'g Graph {
        self.graph
    }

    pub(crate) fn producers(&self) -> &[Option<NodeId>] {
        &self.producers
    }
}

pub fn validate_graph(graph: &Graph) -> Result<(), CompileError> {
    validate_graph_for_analysis(graph).map(|_| ())
}

pub(crate) fn validate_graph_for_analysis(
    graph: &Graph,
) -> Result<ValidatedGraph<'_>, CompileError> {
    validate_tensor_slots(graph)?;
    validate_node_slots(graph)?;
    let mut io_roles = vec![GraphIoRole::default(); graph.tensor_count()];
    validate_graph_inputs(graph, &mut io_roles)?;
    validate_graph_outputs(graph, &mut io_roles)?;
    validate_node_references_and_ops(graph)?;
    let producers = validate_single_producer(graph, &io_roles)?;
    validate_tensor_sources(graph, &producers, &io_roles)?;
    Ok(ValidatedGraph { graph, producers })
}

fn validate_tensor_slots(graph: &Graph) -> Result<(), CompileError> {
    for (index, slot) in graph.tensors().iter().enumerate() {
        if slot.id.index as usize != index {
            return Err(CompileError::InvalidInput);
        }
        if let LogicalLayout::View { base, .. } = &slot.desc.layout {
            graph.tensor(*base).ok_or(CompileError::InvalidInput)?;
            if *base == slot.id {
                return Err(CompileError::InvalidInput);
            }
        }
        slot.desc.elem_count().ok_or(CompileError::InvalidInput)?;
    }
    Ok(())
}

fn validate_node_slots(graph: &Graph) -> Result<(), CompileError> {
    for (index, slot) in graph.nodes().iter().enumerate() {
        if slot.id.index as usize != index {
            return Err(CompileError::InvalidNode);
        }
        if slot.node.id != slot.id {
            return Err(CompileError::InvalidNode);
        }
        if graph.node(slot.id).is_none() {
            return Err(CompileError::InvalidNode);
        }
    }
    Ok(())
}

fn validate_graph_inputs(graph: &Graph, io_roles: &mut [GraphIoRole]) -> Result<(), CompileError> {
    for &input in graph.inputs() {
        graph.tensor(input).ok_or(CompileError::InvalidInput)?;
        let index = input.index as usize;
        let role = io_roles.get_mut(index).ok_or(CompileError::InvalidInput)?;
        if role.is_input {
            return Err(CompileError::InvalidInput);
        }
        role.is_input = true;
    }
    Ok(())
}

fn validate_graph_outputs(graph: &Graph, io_roles: &mut [GraphIoRole]) -> Result<(), CompileError> {
    for &output in graph.outputs() {
        graph.tensor(output).ok_or(CompileError::InvalidOutput)?;
        let index = output.index as usize;
        let role = io_roles.get_mut(index).ok_or(CompileError::InvalidOutput)?;
        if role.is_output {
            return Err(CompileError::InvalidOutput);
        }
        role.is_output = true;
    }
    Ok(())
}

fn validate_node_references_and_ops(graph: &Graph) -> Result<(), CompileError> {
    for slot in graph.nodes() {
        let node = &slot.node;
        for &input in &node.inputs {
            graph.tensor(input).ok_or(CompileError::InvalidInput)?;
        }
        for &output in &node.outputs {
            graph.tensor(output).ok_or(CompileError::InvalidOutput)?;
        }
        match &node.op {
            Op::Add | Op::Mul => validate_binary_elementwise(graph, node)?,
            Op::BroadcastAdd(_) => validate_broadcast_add(graph, node)?,
            Op::Clamp(attrs) => validate_unary_elementwise(graph, node, Some(attrs))?,
            Op::Silu => validate_unary_elementwise(graph, node, None)?,
            Op::QMatmul(attrs) => validate_qmatmul(graph, node, attrs)?,
            Op::Matmul => validate_matmul(graph, node)?,
            Op::Reshape(_) => validate_reshape(graph, node)?,
            Op::Slice(attrs) => validate_slice(graph, node, attrs)?,
            Op::Concat(attrs) => validate_concat(graph, node, attrs)?,
            Op::Pad(attrs) => validate_pad(graph, node, attrs)?,
            Op::Softmax(attrs) => validate_softmax(graph, node, attrs)?,
            Op::ReduceSum(attrs) => validate_reduce_sum(graph, node, attrs)?,
            Op::PrecisionBoundary(attrs) => validate_precision_boundary(graph, node, attrs)?,
            Op::RmsNorm(attrs) => validate_rms_norm(graph, node, attrs)?,
            Op::Rope(attrs) => validate_rope(graph, node, attrs)?,
            Op::Hadamard(attrs) => validate_hadamard(graph, node, attrs)?,
            Op::Qdq(attrs) => validate_qdq(graph, node, attrs)?,
            Op::OrderedTopK(attrs) => validate_ordered_topk(graph, node, attrs)?,
            Op::FusedMlp(attrs) => validate_fused_mlp(graph, node, attrs)?,
        }
    }
    Ok(())
}

fn validate_ordered_topk(
    graph: &Graph,
    node: &Node,
    attrs: &crate::OrderedTopKAttrs,
) -> Result<(), CompileError> {
    if node.inputs.len() != 1 || node.outputs.len() != 2 || attrs.top_k == 0 {
        return Err(CompileError::InvalidInput);
    }
    let input = graph
        .tensor(node.inputs[0])
        .ok_or(CompileError::InvalidInput)?;
    let indices = graph
        .tensor(node.outputs[0])
        .ok_or(CompileError::InvalidOutput)?;
    let scores = graph
        .tensor(node.outputs[1])
        .ok_or(CompileError::InvalidOutput)?;
    validate_f32_contiguous(input)?;
    if attrs.axis >= input.shape.rank() || attrs.top_k > input.shape.dims()[attrs.axis] {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    if attrs.index_base < 0
        || (attrs.padding >= attrs.index_base
            && (attrs.padding - attrs.index_base) as usize >= input.shape.dims()[attrs.axis])
    {
        return Err(CompileError::InvalidInput);
    }
    if indices.dtype != DType::I32
        || !indices.is_contiguous()
        || scores.dtype != DType::F32
        || !scores.is_contiguous()
    {
        return Err(CompileError::Graph(GgmlError::DTypeMismatch));
    }
    if !shape_matches_axis_replacement(
        input.shape.dims(),
        indices.shape.dims(),
        attrs.axis,
        attrs.top_k,
    ) || !shape_matches_axis_replacement(
        input.shape.dims(),
        scores.shape.dims(),
        attrs.axis,
        attrs.top_k,
    ) {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    Ok(())
}

fn shape_matches_axis_replacement(
    source: &[usize],
    target: &[usize],
    axis: usize,
    replacement: usize,
) -> bool {
    axis < source.len()
        && source.len() == target.len()
        && source
            .iter()
            .zip(target)
            .enumerate()
            .all(|(index, (&source_dim, &target_dim))| {
                let expected_dim = if index == axis {
                    replacement
                } else {
                    source_dim
                };
                target_dim == expected_dim
            })
}

fn validate_single_producer(
    graph: &Graph,
    io_roles: &[GraphIoRole],
) -> Result<Vec<Option<NodeId>>, CompileError> {
    let mut producer_of_tensor: Vec<Option<NodeId>> = vec![None; graph.tensor_count()];
    for slot in graph.nodes() {
        for &output in &slot.node.outputs {
            graph.tensor(output).ok_or(CompileError::InvalidOutput)?;
            let out_idx = output.index as usize;
            if io_roles
                .get(out_idx)
                .ok_or(CompileError::InvalidOutput)?
                .is_input
            {
                return Err(CompileError::DuplicateProducer);
            }
            if producer_of_tensor[out_idx].is_some() {
                return Err(CompileError::DuplicateProducer);
            }
            producer_of_tensor[out_idx] = Some(slot.id);
        }
    }
    Ok(producer_of_tensor)
}

fn validate_tensor_sources(
    graph: &Graph,
    producers: &[Option<augur_core::NodeId>],
    io_roles: &[GraphIoRole],
) -> Result<(), CompileError> {
    let mut states = vec![0u8; graph.tensor_count()];
    for slot in graph.tensors() {
        validate_tensor_source(graph, &mut states, slot.id)?;
    }
    for slot in graph.nodes() {
        for &input in &slot.node.inputs {
            if !has_source(graph, producers, io_roles, input)? {
                return Err(CompileError::InvalidInput);
            }
        }
    }
    for &output in graph.outputs() {
        if !has_source(graph, producers, io_roles, output)? {
            return Err(CompileError::InvalidOutput);
        }
    }
    Ok(())
}

fn validate_tensor_source(
    graph: &Graph,
    states: &mut [u8],
    tensor: TensorId,
) -> Result<(), CompileError> {
    let index = tensor.index as usize;
    match states[index] {
        1 => return Err(CompileError::Cycle),
        2 => return Ok(()),
        _ => states[index] = 1,
    }

    let desc = graph.tensor(tensor).ok_or(CompileError::InvalidInput)?;
    if desc.quant.is_some() != (desc.dtype == DType::PackedU8) {
        return Err(CompileError::Graph(GgmlError::DTypeMismatch));
    }
    if let LogicalLayout::View { base, offset } = desc.layout {
        let base_desc = graph.tensor(base).ok_or(CompileError::InvalidInput)?;
        if desc.dtype != base_desc.dtype || desc.quant != base_desc.quant {
            return Err(CompileError::Graph(GgmlError::DTypeMismatch));
        }
        let bytes = desc
            .elem_count()
            .and_then(|n| n.checked_mul(desc.dtype.size_bytes()?))
            .ok_or(CompileError::InvalidInput)?;
        let base_bytes = base_desc
            .elem_count()
            .and_then(|n| n.checked_mul(base_desc.dtype.size_bytes()?))
            .ok_or(CompileError::InvalidInput)?;
        if offset
            .checked_add(bytes)
            .filter(|&end| end <= base_bytes)
            .is_none()
        {
            return Err(CompileError::InvalidInput);
        }
        validate_tensor_source(graph, states, base)?;
    }
    states[index] = 2;
    Ok(())
}

fn has_source(
    graph: &Graph,
    producers: &[Option<augur_core::NodeId>],
    io_roles: &[GraphIoRole],
    tensor: TensorId,
) -> Result<bool, CompileError> {
    let index = tensor.index as usize;
    let desc = graph.tensor(tensor).ok_or(CompileError::InvalidOutput)?;
    if io_roles
        .get(index)
        .ok_or(CompileError::InvalidOutput)?
        .is_input
        || producers[index].is_some()
        || !desc.storage.is_none()
    {
        return Ok(true);
    }
    match desc.layout {
        LogicalLayout::View { base, .. } => has_source(graph, producers, io_roles, base),
        _ => Ok(false),
    }
}

fn validate_binary_elementwise(graph: &Graph, node: &Node) -> Result<(), CompileError> {
    if node.inputs.len() != 2 || node.outputs.len() != 1 {
        return Err(CompileError::InvalidInput);
    }
    let a = graph
        .tensor(node.inputs[0])
        .ok_or(CompileError::InvalidInput)?;
    let b = graph
        .tensor(node.inputs[1])
        .ok_or(CompileError::InvalidInput)?;
    let out = graph
        .tensor(node.outputs[0])
        .ok_or(CompileError::InvalidOutput)?;
    if a.dtype != DType::F32 || b.dtype != DType::F32 || out.dtype != DType::F32 {
        return Err(CompileError::Graph(GgmlError::DTypeMismatch));
    }
    if !a.shape.same_shape(&b.shape) || !a.shape.same_shape(&out.shape) {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    if !a.is_contiguous() || !b.is_contiguous() || !out.is_contiguous() {
        return Err(CompileError::UnsupportedOp);
    }
    Ok(())
}

fn validate_broadcast_add(graph: &Graph, node: &Node) -> Result<(), CompileError> {
    if node.inputs.len() != 2 || node.outputs.len() != 1 {
        return Err(CompileError::InvalidInput);
    }
    let left = graph
        .tensor(node.inputs[0])
        .ok_or(CompileError::InvalidInput)?;
    let right = graph
        .tensor(node.inputs[1])
        .ok_or(CompileError::InvalidInput)?;
    let output = graph
        .tensor(node.outputs[0])
        .ok_or(CompileError::InvalidOutput)?;
    validate_f32_contiguous(left)?;
    validate_f32_contiguous(right)?;
    validate_f32_contiguous(output)?;
    if right.shape.rank() > left.shape.rank() {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    let left_dims = left.shape.dims();
    let right_dims = right.shape.dims();
    for offset in 0..right_dims.len() {
        let left_dim = left_dims[left_dims.len() - 1 - offset];
        let right_dim = right_dims[right_dims.len() - 1 - offset];
        if right_dim != 1 && right_dim != left_dim {
            return Err(CompileError::Graph(GgmlError::ShapeMismatch));
        }
    }
    if output.shape != left.shape {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    Ok(())
}

fn validate_f32_contiguous(desc: &augur_core::TensorDesc) -> Result<(), CompileError> {
    if desc.dtype != DType::F32 || desc.quant.is_some() || !desc.is_contiguous() {
        return Err(CompileError::Graph(GgmlError::DTypeMismatch));
    }
    Ok(())
}

fn validate_unary_f32(graph: &Graph, node: &Node) -> Result<(), CompileError> {
    if node.inputs.len() != 1 || node.outputs.len() != 1 {
        return Err(CompileError::InvalidInput);
    }
    let input = graph
        .tensor(node.inputs[0])
        .ok_or(CompileError::InvalidInput)?;
    let output = graph
        .tensor(node.outputs[0])
        .ok_or(CompileError::InvalidOutput)?;
    validate_f32_contiguous(input)?;
    validate_f32_contiguous(output)?;
    Ok(())
}

fn validate_unary_shape_op(graph: &Graph, node: &Node) -> Result<(), CompileError> {
    validate_unary_f32(graph, node)?;
    let input = graph.tensor(node.inputs[0]).unwrap();
    let output = graph.tensor(node.outputs[0]).unwrap();
    if input.shape != output.shape {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    Ok(())
}

fn validate_reshape(graph: &Graph, node: &Node) -> Result<(), CompileError> {
    validate_unary_f32(graph, node)?;
    let input = graph.tensor(node.inputs[0]).unwrap();
    let output = graph.tensor(node.outputs[0]).unwrap();
    if input.elem_count() != output.elem_count() {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    Ok(())
}

fn validate_slice(
    graph: &Graph,
    node: &Node,
    attrs: &crate::SliceAttrs,
) -> Result<(), CompileError> {
    validate_unary_f32(graph, node)?;
    let input = graph.tensor(node.inputs[0]).unwrap();
    let output = graph.tensor(node.outputs[0]).unwrap();
    if attrs.axis >= input.shape.rank() || attrs.axis >= output.shape.rank() {
        return Err(CompileError::InvalidInput);
    }
    if attrs
        .start
        .checked_add(attrs.len)
        .filter(|&end| end <= input.shape.dims()[attrs.axis])
        .is_none()
        || output.shape.dims()[attrs.axis] != attrs.len
    {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    for (index, (&left, &right)) in input
        .shape
        .dims()
        .iter()
        .zip(output.shape.dims())
        .enumerate()
    {
        if index != attrs.axis && left != right {
            return Err(CompileError::Graph(GgmlError::ShapeMismatch));
        }
    }
    Ok(())
}

fn validate_concat(
    graph: &Graph,
    node: &Node,
    attrs: &crate::ConcatAttrs,
) -> Result<(), CompileError> {
    if node.inputs.len() < 2 || node.outputs.len() != 1 {
        return Err(CompileError::InvalidInput);
    }
    let output = graph
        .tensor(node.outputs[0])
        .ok_or(CompileError::InvalidOutput)?;
    validate_f32_contiguous(output)?;
    let first = graph
        .tensor(node.inputs[0])
        .ok_or(CompileError::InvalidInput)?;
    validate_f32_contiguous(first)?;
    if attrs.axis >= first.shape.rank() || attrs.axis >= output.shape.rank() {
        return Err(CompileError::InvalidInput);
    }
    let mut axis_total = 0usize;
    for &tensor in &node.inputs {
        let value = graph.tensor(tensor).ok_or(CompileError::InvalidInput)?;
        validate_f32_contiguous(value)?;
        if value.shape.rank() != first.shape.rank() {
            return Err(CompileError::Graph(GgmlError::ShapeMismatch));
        }
        axis_total = axis_total
            .checked_add(value.shape.dims()[attrs.axis])
            .ok_or(CompileError::InvalidInput)?;
        for index in 0..first.shape.rank() {
            if index != attrs.axis && value.shape.dims()[index] != first.shape.dims()[index] {
                return Err(CompileError::Graph(GgmlError::ShapeMismatch));
            }
        }
    }
    if output.shape.dims()[attrs.axis] != axis_total {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    Ok(())
}

fn validate_pad(graph: &Graph, node: &Node, attrs: &crate::PadAttrs) -> Result<(), CompileError> {
    validate_unary_f32(graph, node)?;
    let input = graph.tensor(node.inputs[0]).unwrap();
    let output = graph.tensor(node.outputs[0]).unwrap();
    if attrs.axis >= input.shape.rank() {
        return Err(CompileError::InvalidInput);
    }
    let expected = input.shape.dims()[attrs.axis]
        .checked_add(attrs.before)
        .and_then(|value| value.checked_add(attrs.after))
        .ok_or(CompileError::InvalidInput)?;
    if output.shape.dims()[attrs.axis] != expected {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    for index in 0..input.shape.rank() {
        if index != attrs.axis && input.shape.dims()[index] != output.shape.dims()[index] {
            return Err(CompileError::Graph(GgmlError::ShapeMismatch));
        }
    }
    Ok(())
}

fn validate_softmax(
    graph: &Graph,
    node: &Node,
    attrs: &crate::SoftmaxAttrs,
) -> Result<(), CompileError> {
    validate_unary_shape_op(graph, node)?;
    let input = graph.tensor(node.inputs[0]).unwrap();
    if attrs.axis >= input.shape.rank() {
        return Err(CompileError::InvalidInput);
    }
    Ok(())
}

fn validate_reduce_sum(
    graph: &Graph,
    node: &Node,
    attrs: &crate::ReduceSumAttrs,
) -> Result<(), CompileError> {
    if node.inputs.len() != 1 || node.outputs.len() != 1 {
        return Err(CompileError::InvalidInput);
    }
    let input = graph
        .tensor(node.inputs[0])
        .ok_or(CompileError::InvalidInput)?;
    let output = graph
        .tensor(node.outputs[0])
        .ok_or(CompileError::InvalidOutput)?;
    validate_f32_contiguous(input)?;
    validate_f32_contiguous(output)?;
    if attrs.axis >= input.shape.rank()
        || !shape_matches_axis_removal(input.shape.dims(), output.shape.dims(), attrs.axis)
    {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    Ok(())
}

fn shape_matches_axis_removal(source: &[usize], target: &[usize], axis: usize) -> bool {
    axis < source.len()
        && source.len().checked_sub(1) == Some(target.len())
        && source
            .iter()
            .enumerate()
            .filter_map(|(index, &dimension)| (index != axis).then_some(dimension))
            .eq(target.iter().copied())
}

fn validate_precision_boundary(
    graph: &Graph,
    node: &Node,
    attrs: &crate::PrecisionBoundaryAttrs,
) -> Result<(), CompileError> {
    validate_unary_shape_op(graph, node)?;
    if !matches!(attrs.target, DType::BF16 | DType::F16) {
        return Err(CompileError::InvalidInput);
    }
    Ok(())
}

fn validate_rms_norm(
    graph: &Graph,
    node: &Node,
    attrs: &crate::RmsNormAttrs,
) -> Result<(), CompileError> {
    if node.inputs.len() != 2 || node.outputs.len() != 1 || attrs.epsilon.value() <= 0.0 {
        return Err(CompileError::InvalidInput);
    }
    let input = graph
        .tensor(node.inputs[0])
        .ok_or(CompileError::InvalidInput)?;
    let weight = graph
        .tensor(node.inputs[1])
        .ok_or(CompileError::InvalidInput)?;
    let output = graph
        .tensor(node.outputs[0])
        .ok_or(CompileError::InvalidOutput)?;
    validate_f32_contiguous(input)?;
    validate_f32_contiguous(output)?;
    if !matches!(weight.dtype, DType::F32 | DType::BF16)
        || weight.quant.is_some()
        || !weight.is_contiguous()
        || weight.shape.rank() != 1
    {
        return Err(CompileError::Graph(GgmlError::DTypeMismatch));
    }
    if input.shape != output.shape || input.shape.dims().last() != weight.shape.dims().first() {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    Ok(())
}

fn validate_rope(graph: &Graph, node: &Node, attrs: &crate::RopeAttrs) -> Result<(), CompileError> {
    validate_unary_shape_op(graph, node)?;
    let input = graph.tensor(node.inputs[0]).unwrap();
    let width = *input
        .shape
        .dims()
        .last()
        .ok_or(CompileError::InvalidInput)?;
    if attrs.rope_dim == 0
        || attrs.rope_dim > width
        || !attrs.rope_dim.is_multiple_of(2)
        || attrs.base.value() <= 0.0
        || attrs.factor.value() <= 0.0
        || attrs.beta_fast.value() <= 0.0
        || attrs.beta_slow.value() <= 0.0
    {
        return Err(CompileError::InvalidInput);
    }
    Ok(())
}

fn validate_hadamard(
    graph: &Graph,
    node: &Node,
    _attrs: &crate::HadamardAttrs,
) -> Result<(), CompileError> {
    validate_unary_shape_op(graph, node)?;
    let input = graph.tensor(node.inputs[0]).unwrap();
    let width = *input
        .shape
        .dims()
        .last()
        .ok_or(CompileError::InvalidInput)?;
    if width == 0 || !width.is_power_of_two() {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    Ok(())
}

fn validate_qdq(graph: &Graph, node: &Node, attrs: &crate::QdqAttrs) -> Result<(), CompileError> {
    validate_unary_shape_op(graph, node)?;
    let input = graph.tensor(node.inputs[0]).unwrap();
    let width = *input
        .shape
        .dims()
        .last()
        .ok_or(CompileError::InvalidInput)?;
    if attrs.group_size == 0 || width % attrs.group_size != 0 {
        return Err(CompileError::InvalidInput);
    }
    Ok(())
}

fn validate_matmul(graph: &Graph, node: &Node) -> Result<(), CompileError> {
    if node.inputs.len() != 2 || node.outputs.len() != 1 {
        return Err(CompileError::InvalidInput);
    }
    let activation = graph
        .tensor(node.inputs[0])
        .ok_or(CompileError::InvalidInput)?;
    let weight = graph
        .tensor(node.inputs[1])
        .ok_or(CompileError::InvalidInput)?;
    let output = graph
        .tensor(node.outputs[0])
        .ok_or(CompileError::InvalidOutput)?;
    validate_f32_contiguous(activation)?;
    validate_f32_contiguous(output)?;
    if !matches!(weight.dtype, DType::F32 | DType::BF16)
        || !weight.is_contiguous()
        || weight.shape.rank() != 2
        || weight.storage.is_none()
    {
        return Err(CompileError::InvalidInput);
    }
    let [rows, cols] = weight.shape.dims() else {
        return Err(CompileError::InvalidInput);
    };
    let valid = match activation.shape.dims() {
        [width] => *width == *cols && output.shape.dims() == [*rows],
        [m, width] => *m > 0 && *width == *cols && output.shape.dims() == [*m, *rows],
        _ => false,
    };
    if !valid {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    Ok(())
}

fn validate_unary_elementwise(
    graph: &Graph,
    node: &Node,
    clamp: Option<&ClampAttrs>,
) -> Result<(), CompileError> {
    if node.inputs.len() != 1 || node.outputs.len() != 1 {
        return Err(CompileError::InvalidInput);
    }
    if clamp.is_some_and(|attrs| !attrs.is_valid()) {
        return Err(CompileError::InvalidInput);
    }
    let input = graph
        .tensor(node.inputs[0])
        .ok_or(CompileError::InvalidInput)?;
    let output = graph
        .tensor(node.outputs[0])
        .ok_or(CompileError::InvalidOutput)?;
    if input.dtype != DType::F32 || output.dtype != DType::F32 {
        return Err(CompileError::Graph(GgmlError::DTypeMismatch));
    }
    if !input.shape.same_shape(&output.shape) {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(CompileError::UnsupportedOp);
    }
    Ok(())
}

fn validate_fused_mlp(
    graph: &Graph,
    node: &Node,
    attrs: &FusedMlpAttrs,
) -> Result<(), CompileError> {
    if node.inputs.len() != 4 || node.outputs.len() != 1 {
        return Err(CompileError::InvalidInput);
    }
    if !attrs.gate_clamp.is_valid() || !attrs.up_clamp.is_valid() {
        return Err(CompileError::InvalidInput);
    }
    let input = graph
        .tensor(node.inputs[0])
        .ok_or(CompileError::InvalidInput)?;
    let gate = graph
        .tensor(node.inputs[1])
        .ok_or(CompileError::InvalidInput)?;
    let up = graph
        .tensor(node.inputs[2])
        .ok_or(CompileError::InvalidInput)?;
    let down = graph
        .tensor(node.inputs[3])
        .ok_or(CompileError::InvalidInput)?;
    let output = graph
        .tensor(node.outputs[0])
        .ok_or(CompileError::InvalidOutput)?;
    for weight in [gate, up, down] {
        validate_logical_weight(weight)?;
    }
    if input.dtype != DType::F32
        || input.quant.is_some()
        || output.dtype != DType::F32
        || output.quant.is_some()
    {
        return Err(CompileError::Graph(GgmlError::DTypeMismatch));
    }
    if !input.is_contiguous() || !output.is_contiguous() || input.shape.rank() != 1 {
        return Err(CompileError::UnsupportedOp);
    }
    let hidden = input.shape.dims()[0];
    let gate_dims = gate.shape.dims();
    let up_dims = up.shape.dims();
    let down_dims = down.shape.dims();
    let intermediate = gate_dims[0];
    if gate_dims != [intermediate, hidden]
        || up_dims != [intermediate, hidden]
        || down_dims != [hidden, intermediate]
        || output.shape.dims() != [hidden]
    {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    Ok(())
}

fn validate_qmatmul(graph: &Graph, node: &Node, attrs: &QMatmulAttrs) -> Result<(), CompileError> {
    if node.outputs.len() != 1 {
        return Err(CompileError::InvalidOutput);
    }

    let expected_inputs = match attrs.epilogue {
        QMatmulEpilogue::Identity => 2,
        QMatmulEpilogue::Bias | QMatmulEpilogue::BiasSilu => 3,
    };

    if node.inputs.len() != expected_inputs {
        return Err(CompileError::InvalidInput);
    }

    let activation = graph
        .tensor(node.inputs[0])
        .ok_or(CompileError::InvalidInput)?;

    let weight = graph
        .tensor(node.inputs[1])
        .ok_or(CompileError::InvalidInput)?;

    let output = graph
        .tensor(node.outputs[0])
        .ok_or(CompileError::InvalidOutput)?;

    validate_logical_weight(weight)?;

    if !activation.is_contiguous() || !output.is_contiguous() {
        return Err(CompileError::UnsupportedOp);
    }

    if output.dtype != DType::F32 {
        return Err(CompileError::Graph(GgmlError::DTypeMismatch));
    }

    let weight_dims = weight.shape.dims();
    let rows = weight_dims[0];
    let cols = weight_dims[1];

    if activation.dtype != DType::F32 || activation.quant.is_some() || output.quant.is_some() {
        return Err(CompileError::Graph(GgmlError::DTypeMismatch));
    }

    let valid_shape = match activation.shape.dims() {
        [activation_cols] => *activation_cols == cols && output.shape.dims() == [rows],
        [m, activation_cols] => {
            *m > 0 && *activation_cols == cols && output.shape.dims() == [*m, rows]
        }
        _ => false,
    };
    if !valid_shape {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }

    match attrs.epilogue {
        QMatmulEpilogue::Identity => {}
        QMatmulEpilogue::Bias | QMatmulEpilogue::BiasSilu => {
            let bias = graph
                .tensor(node.inputs[2])
                .ok_or(CompileError::InvalidInput)?;

            if bias.dtype != DType::F32 {
                return Err(CompileError::Graph(GgmlError::DTypeMismatch));
            }
            if !bias.is_contiguous() {
                return Err(CompileError::UnsupportedOp);
            }
            let bias_elements = bias.elem_count().ok_or(CompileError::InvalidInput)?;
            if bias_elements != rows {
                return Err(CompileError::Graph(GgmlError::ShapeMismatch));
            }
        }
    }

    Ok(())
}

fn validate_logical_weight(weight: &augur_core::TensorDesc) -> Result<(), CompileError> {
    if weight.dtype != DType::PackedU8 {
        return Err(CompileError::Graph(GgmlError::DTypeMismatch));
    }
    if weight.shape.rank() != 2 {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    let dims = weight.shape.dims();
    if dims[0] == 0 || dims[1] == 0 {
        return Err(CompileError::Graph(GgmlError::ShapeMismatch));
    }
    if !weight.is_contiguous() {
        return Err(CompileError::UnsupportedOp);
    }
    let quant = weight.quant.as_ref().ok_or(CompileError::InvalidInput)?;
    if quant.block == 0 || quant.version == 0 {
        return Err(CompileError::InvalidInput);
    }
    if matches!(weight.storage, augur_core::StorageRef::None) {
        return Err(CompileError::InvalidInput);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{shape_matches_axis_removal, shape_matches_axis_replacement};

    #[test]
    fn axis_replacement_requires_exact_rank_and_non_axis_dimensions() {
        assert!(shape_matches_axis_replacement(&[2, 8, 4], &[2, 3, 4], 1, 3));
        assert!(!shape_matches_axis_replacement(&[2, 8, 4], &[2, 3], 1, 3));
        assert!(!shape_matches_axis_replacement(
            &[2, 8, 4],
            &[7, 3, 4],
            1,
            3
        ));
        assert!(!shape_matches_axis_replacement(
            &[2, 8, 4],
            &[2, 3, 4],
            3,
            3
        ));
    }

    #[test]
    fn axis_removal_requires_exact_remaining_dimensions() {
        assert!(shape_matches_axis_removal(&[2, 8, 4], &[2, 4], 1));
        assert!(!shape_matches_axis_removal(&[2, 8, 4], &[2, 8], 1));
        assert!(!shape_matches_axis_removal(&[2, 8, 4], &[2, 4, 1], 1));
        assert!(!shape_matches_axis_removal(&[2, 8, 4], &[2, 4], 3));
    }
}
