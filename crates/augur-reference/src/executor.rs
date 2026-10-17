use augur_core::{CompileError, DType, LogicalLayout, TensorId};
use augur_graph::{analyze_graph, Graph, Op, PadValue, QdqFormat};

#[derive(Clone, Debug, PartialEq)]
pub enum DebugExecError {
    Compile(CompileError),
    InputCountMismatch {
        expected: usize,
        actual: usize,
    },
    MissingInput(TensorId),
    MissingTensor(TensorId),
    DTypeMismatch(TensorId),
    UnsupportedLayout(TensorId),
    BadInputLength {
        tensor: TensorId,
        expected: usize,
        actual: usize,
    },
    UnsupportedOp,
    BadGraph,
}

impl From<CompileError> for DebugExecError {
    fn from(value: CompileError) -> Self {
        Self::Compile(value)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DebugTensor {
    F32(Vec<f32>),
}

impl DebugTensor {
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            DebugTensor::F32(v) => Some(v),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct DebugTensorSlot {
    tensor: TensorId,
    value: DebugTensor,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DebugRun {
    tensors: Vec<Option<DebugTensorSlot>>,
}

impl DebugRun {
    pub fn tensor_f32(&self, id: TensorId) -> Option<&[f32]> {
        let slot = self.tensors.get(id.index as usize)?.as_ref()?;
        if slot.tensor != id {
            return None;
        }
        slot.value.as_f32()
    }
}

#[derive(Clone, Debug, Default)]
pub struct DebugExecutor;

impl DebugExecutor {
    pub fn run_f32_ordered(graph: &Graph, inputs: &[&[f32]]) -> Result<DebugRun, DebugExecError> {
        if inputs.len() != graph.inputs().len() {
            return Err(DebugExecError::InputCountMismatch {
                expected: graph.inputs().len(),
                actual: inputs.len(),
            });
        }
        let pairs: Vec<(TensorId, &[f32])> = graph
            .inputs()
            .iter()
            .copied()
            .zip(inputs.iter().copied())
            .collect();
        Self::run_f32(graph, &pairs)
    }

    pub fn run_f32(
        graph: &Graph,
        inputs: &[(TensorId, &[f32])],
    ) -> Result<DebugRun, DebugExecError> {
        Self::run_decoded_f32(graph, inputs)
    }

    /// Executes a graph from decoded f32 source bindings.
    ///
    /// Stored BF16 and quantized parameters remain logically typed in Graph
    /// IR; this reference-only API accepts their independently decoded values.
    pub fn run_decoded_f32(
        graph: &Graph,
        bindings: &[(TensorId, &[f32])],
    ) -> Result<DebugRun, DebugExecError> {
        let analyzed = analyze_graph(graph)?;
        let mut tensors = vec![None; graph.tensor_count()];
        for &(tensor, values) in bindings {
            load_decoded_f32(graph, &mut tensors, tensor, values)?;
        }

        for &node_id in analyzed.topo() {
            let slot = analyzed
                .graph()
                .node(node_id)
                .ok_or(DebugExecError::BadGraph)?;
            match &slot.node.op {
                Op::Add => eval_add(analyzed.graph(), &mut tensors, &slot.node)?,
                Op::Mul => {
                    eval_binary(analyzed.graph(), &mut tensors, &slot.node, |left, right| {
                        left * right
                    })?
                }
                Op::Clamp(attrs) => {
                    eval_unary(analyzed.graph(), &mut tensors, &slot.node, |value| {
                        let value =
                            attrs
                                .min()
                                .map_or(value, |bound| if value < bound { bound } else { value });
                        attrs
                            .max()
                            .map_or(value, |bound| if value > bound { bound } else { value })
                    })?
                }
                Op::Silu => eval_unary(analyzed.graph(), &mut tensors, &slot.node, |value| {
                    value / (1.0 + (-value).exp())
                })?,
                Op::Matmul | Op::QMatmul(_) => {
                    eval_matmul(analyzed.graph(), &mut tensors, &slot.node)?
                }
                Op::Reshape(_) => eval_reshape(analyzed.graph(), &mut tensors, &slot.node)?,
                Op::Slice(attrs) => eval_slice(analyzed.graph(), &mut tensors, &slot.node, attrs)?,
                Op::Concat(attrs) => {
                    eval_concat(analyzed.graph(), &mut tensors, &slot.node, attrs.axis)?
                }
                Op::Pad(attrs) => eval_pad(analyzed.graph(), &mut tensors, &slot.node, attrs)?,
                Op::BroadcastAdd(_) => {
                    eval_broadcast_add(analyzed.graph(), &mut tensors, &slot.node)?
                }
                Op::Softmax(attrs) => {
                    eval_softmax(analyzed.graph(), &mut tensors, &slot.node, attrs.axis)?
                }
                Op::ReduceSum(attrs) => {
                    eval_reduce_sum(analyzed.graph(), &mut tensors, &slot.node, attrs.axis)?
                }
                Op::PrecisionBoundary(attrs) => {
                    if attrs.target != DType::BF16 {
                        return Err(DebugExecError::UnsupportedOp);
                    }
                    eval_unary(analyzed.graph(), &mut tensors, &slot.node, bf16_round)?
                }
                Op::RmsNorm(attrs) => eval_rms_norm(
                    analyzed.graph(),
                    &mut tensors,
                    &slot.node,
                    attrs.epsilon.value(),
                )?,
                Op::Rope(attrs) => eval_rope(analyzed.graph(), &mut tensors, &slot.node, attrs)?,
                Op::Hadamard(attrs) => {
                    eval_hadamard(analyzed.graph(), &mut tensors, &slot.node, attrs.normalized)?
                }
                Op::Qdq(attrs) => eval_qdq(
                    analyzed.graph(),
                    &mut tensors,
                    &slot.node,
                    attrs.format,
                    attrs.group_size,
                )?,
                _ => return Err(DebugExecError::UnsupportedOp),
            }
        }

        for &output in graph.outputs() {
            if get_tensor(&tensors, output).is_none() {
                return Err(DebugExecError::MissingTensor(output));
            }
        }
        Ok(DebugRun { tensors })
    }
}

fn eval_binary(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    node: &augur_graph::Node,
    op: impl Fn(f32, f32) -> f32,
) -> Result<(), DebugExecError> {
    if node.inputs.len() != 2 || node.outputs.len() != 1 {
        return Err(DebugExecError::BadGraph);
    }
    let left = get_tensor_f32(tensors, node.inputs[0])?;
    let right = get_tensor_f32(tensors, node.inputs[1])?;
    if left.len() != right.len() {
        return Err(DebugExecError::BadGraph);
    }
    let output = left
        .iter()
        .zip(right)
        .map(|(&left, &right)| op(left, right))
        .collect();
    store_f32(graph, tensors, node.outputs[0], output)
}

fn eval_unary(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    node: &augur_graph::Node,
    op: impl Fn(f32) -> f32,
) -> Result<(), DebugExecError> {
    if node.inputs.len() != 1 || node.outputs.len() != 1 {
        return Err(DebugExecError::BadGraph);
    }
    let input = get_tensor_f32(tensors, node.inputs[0])?;
    let output = input.iter().copied().map(op).collect();
    store_f32(graph, tensors, node.outputs[0], output)
}

fn store_f32(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    tensor: TensorId,
    value: Vec<f32>,
) -> Result<(), DebugExecError> {
    ensure_f32_contiguous(graph, tensor)?;
    let expected = graph
        .tensor(tensor)
        .and_then(|desc| desc.elem_count())
        .ok_or(DebugExecError::BadGraph)?;
    if value.len() != expected {
        return Err(DebugExecError::BadGraph);
    }
    *tensors
        .get_mut(tensor.index as usize)
        .ok_or(DebugExecError::MissingTensor(tensor))? = Some(DebugTensorSlot {
        tensor,
        value: DebugTensor::F32(value),
    });
    Ok(())
}

fn load_decoded_f32(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    tensor: TensorId,
    values: &[f32],
) -> Result<(), DebugExecError> {
    let desc = graph
        .tensor(tensor)
        .ok_or(DebugExecError::MissingInput(tensor))?;
    if !matches!(desc.layout, LogicalLayout::Contiguous) {
        return Err(DebugExecError::UnsupportedLayout(tensor));
    }
    let expected = desc.elem_count().ok_or(DebugExecError::BadInputLength {
        tensor,
        expected: usize::MAX,
        actual: values.len(),
    })?;
    if values.len() != expected {
        return Err(DebugExecError::BadInputLength {
            tensor,
            expected,
            actual: values.len(),
        });
    }
    let idx = tensor.index as usize;
    *tensors
        .get_mut(idx)
        .ok_or(DebugExecError::MissingInput(tensor))? = Some(DebugTensorSlot {
        tensor,
        value: DebugTensor::F32(values.to_vec()),
    });
    Ok(())
}

fn eval_add(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    node: &augur_graph::Node,
) -> Result<(), DebugExecError> {
    if node.inputs.len() != 2 || node.outputs.len() != 1 {
        return Err(DebugExecError::BadGraph);
    }
    let a_id = node.inputs[0];
    let b_id = node.inputs[1];
    let out_id = node.outputs[0];
    ensure_f32_contiguous(graph, a_id)?;
    ensure_f32_contiguous(graph, b_id)?;
    ensure_f32_contiguous(graph, out_id)?;
    let a = get_tensor_f32(tensors, a_id)?;
    let b = get_tensor_f32(tensors, b_id)?;
    if a.len() != b.len() {
        return Err(DebugExecError::BadGraph);
    }
    let out_desc = graph
        .tensor(out_id)
        .ok_or(DebugExecError::MissingTensor(out_id))?;
    let n = out_desc.elem_count().ok_or(DebugExecError::BadGraph)?;
    if a.len() != n {
        return Err(DebugExecError::BadGraph);
    }
    let out: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let idx = out_id.index as usize;
    *tensors
        .get_mut(idx)
        .ok_or(DebugExecError::MissingTensor(out_id))? = Some(DebugTensorSlot {
        tensor: out_id,
        value: DebugTensor::F32(out),
    });
    Ok(())
}

fn eval_matmul(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    node: &augur_graph::Node,
) -> Result<(), DebugExecError> {
    let activation_desc = graph
        .tensor(node.inputs[0])
        .ok_or(DebugExecError::BadGraph)?;
    let weight_desc = graph
        .tensor(node.inputs[1])
        .ok_or(DebugExecError::BadGraph)?;
    let activation = get_tensor_f32(tensors, node.inputs[0])?;
    let weight = get_tensor_f32(tensors, node.inputs[1])?;
    let dims = weight_desc.shape.dims();
    let rows = dims[0];
    let cols = dims[1];
    let m = match activation_desc.shape.dims() {
        [_] => 1,
        [m, _] => *m,
        _ => return Err(DebugExecError::BadGraph),
    };
    if activation.len() != m * cols || weight.len() != rows * cols {
        return Err(DebugExecError::BadGraph);
    }
    let mut output = vec![0.0f32; m * rows];
    for batch in 0..m {
        for row in 0..rows {
            let mut sum = 0.0f32;
            for col in 0..cols {
                sum += activation[batch * cols + col] * weight[row * cols + col];
            }
            output[batch * rows + row] = sum;
        }
    }
    store_f32(graph, tensors, node.outputs[0], output)
}

fn eval_reshape(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    node: &augur_graph::Node,
) -> Result<(), DebugExecError> {
    let output = get_tensor_f32(tensors, node.inputs[0])?.to_vec();
    store_f32(graph, tensors, node.outputs[0], output)
}

fn axis_geometry(dims: &[usize], axis: usize) -> Result<(usize, usize, usize), DebugExecError> {
    if axis >= dims.len() {
        return Err(DebugExecError::BadGraph);
    }
    let outer = dims[..axis].iter().product();
    let width = dims[axis];
    let inner = dims[axis + 1..].iter().product();
    Ok((outer, width, inner))
}

fn eval_slice(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    node: &augur_graph::Node,
    attrs: &augur_graph::SliceAttrs,
) -> Result<(), DebugExecError> {
    let input_desc = graph
        .tensor(node.inputs[0])
        .ok_or(DebugExecError::BadGraph)?;
    let input = get_tensor_f32(tensors, node.inputs[0])?;
    let (outer, width, inner) = axis_geometry(input_desc.shape.dims(), attrs.axis)?;
    let mut output = Vec::with_capacity(outer * attrs.len * inner);
    for outer_index in 0..outer {
        let base = outer_index * width * inner + attrs.start * inner;
        output.extend_from_slice(&input[base..base + attrs.len * inner]);
    }
    store_f32(graph, tensors, node.outputs[0], output)
}

fn eval_concat(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    node: &augur_graph::Node,
    axis: usize,
) -> Result<(), DebugExecError> {
    let first = graph
        .tensor(node.inputs[0])
        .ok_or(DebugExecError::BadGraph)?;
    let (outer, _, inner) = axis_geometry(first.shape.dims(), axis)?;
    let values: Vec<Vec<f32>> = node
        .inputs
        .iter()
        .map(|&tensor| get_tensor_f32(tensors, tensor).map(ToOwned::to_owned))
        .collect::<Result<_, _>>()?;
    let widths: Vec<usize> = node
        .inputs
        .iter()
        .map(|&tensor| {
            graph
                .tensor(tensor)
                .map(|desc| desc.shape.dims()[axis])
                .ok_or(DebugExecError::BadGraph)
        })
        .collect::<Result<_, _>>()?;
    let mut output = Vec::with_capacity(values.iter().map(Vec::len).sum());
    for outer_index in 0..outer {
        for (value, &width) in values.iter().zip(&widths) {
            let base = outer_index * width * inner;
            output.extend_from_slice(&value[base..base + width * inner]);
        }
    }
    store_f32(graph, tensors, node.outputs[0], output)
}

fn eval_pad(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    node: &augur_graph::Node,
    attrs: &augur_graph::PadAttrs,
) -> Result<(), DebugExecError> {
    let input_desc = graph
        .tensor(node.inputs[0])
        .ok_or(DebugExecError::BadGraph)?;
    let input = get_tensor_f32(tensors, node.inputs[0])?;
    let (outer, width, inner) = axis_geometry(input_desc.shape.dims(), attrs.axis)?;
    let fill = match attrs.value {
        PadValue::Zero => 0.0,
        PadValue::NegativeInfinity => f32::NEG_INFINITY,
    };
    let padded_width = attrs.before + width + attrs.after;
    let mut output = vec![fill; outer * padded_width * inner];
    for outer_index in 0..outer {
        let source = outer_index * width * inner;
        let target = outer_index * padded_width * inner + attrs.before * inner;
        output[target..target + width * inner]
            .copy_from_slice(&input[source..source + width * inner]);
    }
    store_f32(graph, tensors, node.outputs[0], output)
}

fn eval_broadcast_add(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    node: &augur_graph::Node,
) -> Result<(), DebugExecError> {
    let left_desc = graph
        .tensor(node.inputs[0])
        .ok_or(DebugExecError::BadGraph)?;
    let right_desc = graph
        .tensor(node.inputs[1])
        .ok_or(DebugExecError::BadGraph)?;
    let left = get_tensor_f32(tensors, node.inputs[0])?;
    let right = get_tensor_f32(tensors, node.inputs[1])?;
    let left_dims = left_desc.shape.dims();
    let right_dims = right_desc.shape.dims();
    let mut right_strides = vec![0usize; right_dims.len()];
    let mut stride = 1usize;
    for index in (0..right_dims.len()).rev() {
        right_strides[index] = stride;
        stride *= right_dims[index];
    }
    let mut output = Vec::with_capacity(left.len());
    for (flat, left_value) in left.iter().enumerate() {
        let mut remaining = flat;
        let mut right_flat = 0usize;
        for left_axis in (0..left_dims.len()).rev() {
            let coordinate = remaining % left_dims[left_axis];
            remaining /= left_dims[left_axis];
            if left_axis + right_dims.len() >= left_dims.len() {
                let right_axis = left_axis + right_dims.len() - left_dims.len();
                let right_coordinate = if right_dims[right_axis] == 1 {
                    0
                } else {
                    coordinate
                };
                right_flat += right_coordinate * right_strides[right_axis];
            }
        }
        output.push(*left_value + right[right_flat]);
    }
    store_f32(graph, tensors, node.outputs[0], output)
}

fn eval_softmax(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    node: &augur_graph::Node,
    axis: usize,
) -> Result<(), DebugExecError> {
    let input_desc = graph
        .tensor(node.inputs[0])
        .ok_or(DebugExecError::BadGraph)?;
    let input = get_tensor_f32(tensors, node.inputs[0])?;
    let (outer, width, inner) = axis_geometry(input_desc.shape.dims(), axis)?;
    let mut output = vec![0.0f32; input.len()];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let mut maximum = f32::NEG_INFINITY;
            for axis_index in 0..width {
                maximum = maximum
                    .max(input[outer_index * width * inner + axis_index * inner + inner_index]);
            }
            let mut denominator = 0.0f32;
            for axis_index in 0..width {
                let index = outer_index * width * inner + axis_index * inner + inner_index;
                let value = (input[index] - maximum).exp();
                output[index] = value;
                denominator += value;
            }
            for axis_index in 0..width {
                let index = outer_index * width * inner + axis_index * inner + inner_index;
                output[index] /= denominator;
            }
        }
    }
    store_f32(graph, tensors, node.outputs[0], output)
}

fn eval_reduce_sum(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    node: &augur_graph::Node,
    axis: usize,
) -> Result<(), DebugExecError> {
    let input_desc = graph
        .tensor(node.inputs[0])
        .ok_or(DebugExecError::BadGraph)?;
    let input = get_tensor_f32(tensors, node.inputs[0])?;
    let (outer, width, inner) = axis_geometry(input_desc.shape.dims(), axis)?;
    let mut output = vec![0.0f32; outer * inner];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let mut sum = 0.0f32;
            for axis_index in 0..width {
                sum += input[outer_index * width * inner + axis_index * inner + inner_index];
            }
            output[outer_index * inner + inner_index] = sum;
        }
    }
    store_f32(graph, tensors, node.outputs[0], output)
}

fn bf16_round(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

fn eval_rms_norm(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    node: &augur_graph::Node,
    epsilon: f32,
) -> Result<(), DebugExecError> {
    let input_desc = graph
        .tensor(node.inputs[0])
        .ok_or(DebugExecError::BadGraph)?;
    let width = *input_desc
        .shape
        .dims()
        .last()
        .ok_or(DebugExecError::BadGraph)?;
    let input = get_tensor_f32(tensors, node.inputs[0])?;
    let weight = get_tensor_f32(tensors, node.inputs[1])?;
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(width) {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / width as f32;
        let scale = (mean_square + epsilon).sqrt().recip();
        output.extend(
            row.iter()
                .zip(weight)
                .map(|(&value, &weight)| bf16_round(value * scale * weight)),
        );
    }
    store_f32(graph, tensors, node.outputs[0], output)
}

fn eval_rope(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    node: &augur_graph::Node,
    attrs: &augur_graph::RopeAttrs,
) -> Result<(), DebugExecError> {
    let desc = graph
        .tensor(node.inputs[0])
        .ok_or(DebugExecError::BadGraph)?;
    let dims = desc.shape.dims();
    let width = *dims.last().ok_or(DebugExecError::BadGraph)?;
    let positions = dims[0];
    let rows_per_position = desc.elem_count().ok_or(DebugExecError::BadGraph)? / positions / width;
    let input = get_tensor_f32(tensors, node.inputs[0])?;
    let mut output = input.to_vec();
    let correction = |value: f32| {
        attrs.rope_dim as f32
            * (attrs.original_context as f32 / (value * 2.0 * core::f32::consts::PI)).ln()
            / (2.0 * attrs.base.value().ln())
    };
    let low = correction(attrs.beta_fast.value()).floor().max(0.0) as usize;
    let high = correction(attrs.beta_slow.value())
        .ceil()
        .min((attrs.rope_dim - 1) as f32) as usize;
    for row in 0..(input.len() / width) {
        let position_index = row / rows_per_position;
        let position = attrs.position_start + position_index * attrs.position_step;
        let tail = row * width + width - attrs.rope_dim;
        for pair in 0..attrs.rope_dim / 2 {
            let ramp = ((pair as f32 - low as f32) / (high.saturating_sub(low).max(1) as f32))
                .clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            let base_frequency = 1.0
                / attrs
                    .base
                    .value()
                    .powf((pair * 2) as f32 / attrs.rope_dim as f32);
            let frequency =
                base_frequency / attrs.factor.value() * (1.0 - smooth) + base_frequency * smooth;
            let angle = position as f32 * frequency;
            let left = input[tail + pair * 2];
            let right = input[tail + pair * 2 + 1];
            output[tail + pair * 2] = bf16_round(left * angle.cos() - right * angle.sin());
            output[tail + pair * 2 + 1] = bf16_round(left * angle.sin() + right * angle.cos());
        }
        for value in &mut output[row * width..tail] {
            *value = bf16_round(*value);
        }
    }
    store_f32(graph, tensors, node.outputs[0], output)
}

fn eval_hadamard(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    node: &augur_graph::Node,
    normalized: bool,
) -> Result<(), DebugExecError> {
    let desc = graph
        .tensor(node.inputs[0])
        .ok_or(DebugExecError::BadGraph)?;
    let width = *desc.shape.dims().last().ok_or(DebugExecError::BadGraph)?;
    let input = get_tensor_f32(tensors, node.inputs[0])?;
    let mut output = input.to_vec();
    for row in output.chunks_exact_mut(width) {
        let mut step = 1usize;
        while step < width {
            for block in row.chunks_exact_mut(step * 2) {
                for lane in 0..step {
                    let left = block[lane];
                    let right = block[step + lane];
                    block[lane] = left + right;
                    block[step + lane] = left - right;
                }
            }
            step *= 2;
        }
        let scale = if normalized {
            (width as f32).sqrt().recip()
        } else {
            1.0
        };
        for value in row {
            *value = bf16_round(*value * scale);
        }
    }
    store_f32(graph, tensors, node.outputs[0], output)
}

fn fp8_codebook() -> Vec<f32> {
    let mut values = Vec::new();
    for code in 0u16..=255 {
        let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
        let exponent = (code >> 3) & 0x0f;
        let mantissa = code & 0x07;
        if exponent == 15 && mantissa == 7 {
            continue;
        }
        let value = if exponent == 0 {
            sign * mantissa as f32 / 8.0 * 2.0f32.powi(-6)
        } else {
            sign * (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
        };
        if !values.contains(&value) {
            values.push(value);
        }
    }
    values.sort_by(f32::total_cmp);
    values
}

fn nearest(value: f32, codebook: &[f32]) -> f32 {
    let mut best = codebook[0];
    let mut distance = (value - best).abs();
    for &candidate in &codebook[1..] {
        let candidate_distance = (value - candidate).abs();
        if candidate_distance < distance {
            best = candidate;
            distance = candidate_distance;
        }
    }
    best
}

fn eval_qdq(
    graph: &Graph,
    tensors: &mut [Option<DebugTensorSlot>],
    node: &augur_graph::Node,
    format: QdqFormat,
    group_size: usize,
) -> Result<(), DebugExecError> {
    let input = get_tensor_f32(tensors, node.inputs[0])?;
    let fp8 = fp8_codebook();
    let fp4 = [
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    ];
    let (maximum, minimum, codebook): (f32, f32, &[f32]) = match format {
        QdqFormat::Fp8E4M3 => (448.0, 1.0e-4, &fp8),
        QdqFormat::Fp4E2M1 => (6.0, 6.0 * 2.0f32.powi(-126), &fp4),
    };
    let mut output = Vec::with_capacity(input.len());
    for group in input.chunks_exact(group_size) {
        let amax = group
            .iter()
            .fold(0.0f32, |acc, value| acc.max(value.abs()))
            .max(minimum);
        let scale = 2.0f32.powf((amax / maximum).log2().ceil());
        output.extend(group.iter().map(|&value| {
            let quantized = nearest((value / scale).clamp(-maximum, maximum), codebook);
            bf16_round(quantized * scale)
        }));
    }
    store_f32(graph, tensors, node.outputs[0], output)
}

fn ensure_f32_contiguous(graph: &Graph, tensor: TensorId) -> Result<(), DebugExecError> {
    let desc = graph
        .tensor(tensor)
        .ok_or(DebugExecError::MissingTensor(tensor))?;
    if desc.dtype != DType::F32 {
        return Err(DebugExecError::DTypeMismatch(tensor));
    }
    if !matches!(desc.layout, LogicalLayout::Contiguous) {
        return Err(DebugExecError::UnsupportedLayout(tensor));
    }
    Ok(())
}

fn get_tensor(tensors: &[Option<DebugTensorSlot>], tensor: TensorId) -> Option<&DebugTensor> {
    let slot = tensors.get(tensor.index as usize)?.as_ref()?;
    if slot.tensor != tensor {
        return None;
    }
    Some(&slot.value)
}

fn get_tensor_f32(
    tensors: &[Option<DebugTensorSlot>],
    tensor: TensorId,
) -> Result<&[f32], DebugExecError> {
    match get_tensor(tensors, tensor).ok_or(DebugExecError::MissingTensor(tensor))? {
        DebugTensor::F32(v) => Ok(v),
    }
}
