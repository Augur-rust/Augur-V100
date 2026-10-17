#![cfg(feature = "cuda")]

use augur_backend::{
    cuda_device_properties, CudaComputeCapability, CudaDeviceProperties, CudaRuntimeError,
};
use augur_core::{
    CompileError, DType, FormatVersion, LogicalLayout, PhysicalLayoutId, QuantFormatId, StorageRef,
    TensorId,
};
use augur_exec::BufferPurpose;
use augur_graph::{analyze_graph, AnalyzedGraph, Graph, Op, QMatmulEpilogue};
use augur_kernel_api::{Q4K_HOT_CUDA_V1_BLOCK_BYTES, Q4K_HOT_CUDA_V1_DECODED_LEN};

use crate::{
    CudaCompiledAddF32, CudaCompiledF32View, CudaCompiledPlan, CudaCompiledQ4kF32Decode,
    CudaCompiledQ4kF32Prefill, CudaCompiledStep, CudaQ4kWeightRequirement, CudaWeightBindingId,
    MemoryPlanner,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaCompileError {
    Graph(CompileError),

    Runtime(CudaRuntimeError),

    UnsupportedCapability {
        actual: CudaComputeCapability,
        required: CudaComputeCapability,
    },

    UnsupportedGraphTopology,

    UnsupportedOperation,

    UnsupportedEpilogue {
        actual: QMatmulEpilogue,
    },

    InvalidActivation,

    InvalidWeight,

    InvalidOutput,

    InputLengthMismatch {
        expected: usize,
        actual: usize,
    },

    ArithmeticOverflow,
}

impl From<CudaRuntimeError> for CudaCompileError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Runtime(value)
    }
}

/// Compiler entry for the M30 CUDA Q4K decode slice.
///
/// Construction performs CUDA device discovery and capability validation once.
/// The resulting compiler does not contain CPU capabilities, CPU tuning state,
/// a CUDA stream, device allocations or resolved device addresses.
#[derive(Clone, Debug)]
pub struct CudaGraphCompiler {
    properties: CudaDeviceProperties,
}

impl CudaGraphCompiler {
    pub fn new(device_ordinal: i32) -> Result<Self, CudaCompileError> {
        Self::from_properties(cuda_device_properties(device_ordinal)?)
    }

    fn from_properties(properties: CudaDeviceProperties) -> Result<Self, CudaCompileError> {
        if !properties
            .compute_capability
            .is_at_least(CudaComputeCapability::SM70)
        {
            return Err(CudaCompileError::UnsupportedCapability {
                actual: properties.compute_capability,
                required: CudaComputeCapability::SM70,
            });
        }

        Ok(Self { properties })
    }

    pub const fn device_ordinal(&self) -> i32 {
        self.properties.ordinal
    }

    pub const fn properties(&self) -> &CudaDeviceProperties {
        &self.properties
    }

    /// Lowers a bounded Q4K Identity / f32 Add CUDA plan.
    ///
    /// This method performs Graph analysis and fixes all logical shape,
    /// format, layout and resource requirements. It does not allocate CUDA
    /// resources, upload payloads, resolve device addresses or construct
    /// native launch arguments.
    pub fn compile_plan(
        &self,
        graph: &Graph,
        input_f32: Vec<f32>,
    ) -> Result<CudaCompiledPlan, CudaCompileError> {
        let analyzed = analyze_graph(graph).map_err(CudaCompileError::Graph)?;

        if graph.node_count() == 0 || graph.inputs().len() != 1 || graph.outputs().len() != 1 {
            return Err(CudaCompileError::UnsupportedGraphTopology);
        }
        let input_tensor = graph.inputs()[0];
        let input_desc = graph
            .tensor(input_tensor)
            .ok_or(CudaCompileError::InvalidActivation)?;
        validate_plain_f32(input_desc)?;
        if !matches!(input_desc.shape.rank(), 1 | 2) {
            return Err(CudaCompileError::InvalidActivation);
        }
        let input_elements = input_desc
            .elem_count()
            .ok_or(CudaCompileError::ArithmeticOverflow)?;
        if input_f32.len() != input_elements {
            return Err(CudaCompileError::InputLengthMismatch {
                expected: input_elements,
                actual: input_f32.len(),
            });
        }

        let mut memory = MemoryPlanner::new(graph.tensor_count());
        let input_range = analyzed
            .liveness()
            .range(input_tensor)
            .ok_or(CudaCompileError::InvalidActivation)?;
        let input_buffer = memory
            .assign(
                input_tensor,
                input_elements,
                BufferPurpose::Input,
                Some(input_range.last_step),
            )
            .map_err(CudaCompileError::Graph)?;
        let input_view = CudaCompiledF32View {
            buffer_index: input_buffer,
            len_elements: input_elements,
        };
        let mut requirements = Vec::new();
        let mut steps = Vec::with_capacity(graph.node_count());

        for (topo_step, &node_id) in analyzed.topo().iter().enumerate() {
            let node = &graph
                .node(node_id)
                .ok_or(CudaCompileError::UnsupportedGraphTopology)?
                .node;
            match &node.op {
                Op::Add => {
                    let elements = graph
                        .tensor(node.outputs[0])
                        .and_then(|desc| desc.elem_count())
                        .ok_or(CudaCompileError::InvalidOutput)?;
                    let a = compiled_view(&memory, node.inputs[0], elements)?;
                    let b = compiled_view(&memory, node.inputs[1], elements)?;
                    let output = assign_output(&analyzed, &mut memory, node.outputs[0], topo_step)?;
                    steps.push(CudaCompiledStep::AddF32(CudaCompiledAddF32 {
                        a,
                        b,
                        output,
                        elements,
                    }));
                }
                Op::QMatmul(attrs) => {
                    if attrs.epilogue != QMatmulEpilogue::Identity {
                        return Err(CudaCompileError::UnsupportedEpilogue {
                            actual: attrs.epilogue,
                        });
                    }
                    let (requirement, m, rows, cols, prefill) =
                        q4k_requirement(graph, node.inputs[0], node.inputs[1], &mut requirements)?;
                    let input = compiled_view(
                        &memory,
                        node.inputs[0],
                        m.checked_mul(cols)
                            .ok_or(CudaCompileError::ArithmeticOverflow)?,
                    )?;
                    let output = assign_output(&analyzed, &mut memory, node.outputs[0], topo_step)?;
                    steps.push(if prefill {
                        CudaCompiledStep::Q4kF32Prefill(CudaCompiledQ4kF32Prefill {
                            weight: requirement,
                            input,
                            output,
                            m,
                            rows,
                            cols,
                        })
                    } else {
                        CudaCompiledStep::Q4kF32Decode(CudaCompiledQ4kF32Decode {
                            weight: requirement,
                            input,
                            output,
                            rows,
                            cols,
                        })
                    });
                }
                Op::OrderedTopK(_) => return Err(CudaCompileError::UnsupportedOperation),
                _ => return Err(CudaCompileError::UnsupportedOperation),
            }
        }

        let output_tensor = graph.outputs()[0];
        let output_buffer = memory
            .buffer_for(output_tensor)
            .ok_or(CudaCompileError::InvalidOutput)?;
        let output_elements = graph
            .tensor(output_tensor)
            .and_then(|desc| desc.elem_count())
            .ok_or(CudaCompileError::InvalidOutput)?;

        Ok(CudaCompiledPlan {
            device_ordinal: self.properties.ordinal,
            input_f32: input_f32.into_boxed_slice(),
            steps: steps.into_boxed_slice(),
            weight_requirements: requirements.into_boxed_slice(),
            memory: memory.finish(),
            input: input_view,
            output: CudaCompiledF32View {
                buffer_index: output_buffer,
                len_elements: output_elements,
            },
        })
    }
}

fn validate_plain_f32(desc: &augur_core::TensorDesc) -> Result<(), CudaCompileError> {
    if desc.dtype != DType::F32 || desc.quant.is_some() || desc.layout != LogicalLayout::Contiguous
    {
        return Err(CudaCompileError::InvalidActivation);
    }
    Ok(())
}

fn compiled_view(
    memory: &MemoryPlanner,
    tensor: TensorId,
    len_elements: usize,
) -> Result<CudaCompiledF32View, CudaCompileError> {
    Ok(CudaCompiledF32View {
        buffer_index: memory
            .buffer_for(tensor)
            .ok_or(CudaCompileError::InvalidActivation)?,
        len_elements,
    })
}

fn assign_output(
    analyzed: &AnalyzedGraph<'_>,
    memory: &mut MemoryPlanner,
    tensor: TensorId,
    topo_step: usize,
) -> Result<CudaCompiledF32View, CudaCompileError> {
    let graph = analyzed.graph();
    let desc = graph
        .tensor(tensor)
        .ok_or(CudaCompileError::InvalidOutput)?;
    validate_plain_f32(desc).map_err(|_| CudaCompileError::InvalidOutput)?;
    let len_elements = desc
        .elem_count()
        .ok_or(CudaCompileError::ArithmeticOverflow)?;
    let range = analyzed
        .liveness()
        .range(tensor)
        .ok_or(CudaCompileError::InvalidOutput)?;
    if range.first_step != topo_step {
        return Err(CudaCompileError::UnsupportedGraphTopology);
    }
    let purpose = if graph.outputs().contains(&tensor) {
        BufferPurpose::Output
    } else {
        BufferPurpose::Intermediate
    };
    let buffer_index = if purpose == BufferPurpose::Intermediate {
        memory.assign_reusing_tensor(
            tensor,
            len_elements,
            purpose,
            range.first_step,
            range.last_step,
        )
    } else {
        memory.assign(tensor, len_elements, purpose, Some(range.last_step))
    }
    .map_err(CudaCompileError::Graph)?;
    Ok(CudaCompiledF32View {
        buffer_index,
        len_elements,
    })
}

fn q4k_requirement(
    graph: &Graph,
    activation: TensorId,
    weight: TensorId,
    requirements: &mut Vec<CudaQ4kWeightRequirement>,
) -> Result<(CudaWeightBindingId, usize, usize, usize, bool), CudaCompileError> {
    let activation = graph
        .tensor(activation)
        .ok_or(CudaCompileError::InvalidActivation)?;
    validate_plain_f32(activation)?;
    let weight = graph
        .tensor(weight)
        .ok_or(CudaCompileError::InvalidWeight)?;
    let quant = weight
        .quant
        .as_ref()
        .ok_or(CudaCompileError::InvalidWeight)?;
    let storage = match weight.storage {
        StorageRef::HostOwned { id } => id,
        _ => return Err(CudaCompileError::InvalidWeight),
    };
    if weight.dtype != DType::PackedU8
        || weight.layout != LogicalLayout::Contiguous
        || weight.shape.rank() != 2
        || quant.format != QuantFormatId::Q4K
        || quant.version != 1
        || quant.block != Q4K_HOT_CUDA_V1_DECODED_LEN
    {
        return Err(CudaCompileError::InvalidWeight);
    }
    let rows = weight.shape.dims()[0];
    let cols = weight.shape.dims()[1];
    if rows == 0 || cols == 0 || !cols.is_multiple_of(Q4K_HOT_CUDA_V1_DECODED_LEN) {
        return Err(CudaCompileError::InvalidWeight);
    }
    let (m, prefill) = match activation.shape.dims() {
        [activation_cols] if *activation_cols == cols => (1, false),
        [m, activation_cols] if *m > 0 && *activation_cols == cols => (*m, true),
        _ => return Err(CudaCompileError::InvalidActivation),
    };
    let payload_bytes = rows
        .checked_mul(cols / Q4K_HOT_CUDA_V1_DECODED_LEN)
        .and_then(|blocks| blocks.checked_mul(Q4K_HOT_CUDA_V1_BLOCK_BYTES))
        .ok_or(CudaCompileError::ArithmeticOverflow)?;
    let metadata = (
        storage,
        rows,
        cols,
        Q4K_HOT_CUDA_V1_DECODED_LEN,
        payload_bytes,
    );
    if let Some(existing) = requirements
        .iter()
        .find(|requirement| requirement.storage.index == storage.index)
    {
        if (
            existing.storage,
            existing.rows,
            existing.cols,
            existing.block_size,
            existing.payload_bytes,
        ) != metadata
        {
            return Err(CudaCompileError::InvalidWeight);
        }
        return Ok((existing.binding, m, rows, cols, prefill));
    }
    let binding_index =
        u32::try_from(requirements.len()).map_err(|_| CudaCompileError::ArithmeticOverflow)?;
    let binding = CudaWeightBindingId::new(binding_index);
    requirements.push(CudaQ4kWeightRequirement {
        binding,
        storage,
        quant_format: QuantFormatId::Q4K,
        format_version: FormatVersion(1),
        physical_layout: PhysicalLayoutId::Q4K_HOT_CUDA_V1,
        rows,
        cols,
        block_size: Q4K_HOT_CUDA_V1_DECODED_LEN,
        payload_bytes,
    });
    Ok((binding, m, rows, cols, prefill))
}

#[cfg(test)]
mod tests {
    use super::*;

    use augur_core::{Shape, StorageId, TensorDesc};
    use augur_graph::Graph;

    fn properties(major: u16, minor: u16) -> CudaDeviceProperties {
        CudaDeviceProperties {
            ordinal: 0,
            compute_capability: CudaComputeCapability { major, minor },
            total_global_memory_bytes: 16 * 1024 * 1024 * 1024,
            name: format!("test-sm{major}{minor}"),
        }
    }

    fn compiler() -> CudaGraphCompiler {
        CudaGraphCompiler::from_properties(properties(7, 0)).unwrap()
    }

    fn q4k_graph(rows: usize, cols: usize, block_size: usize, version: u16) -> Graph {
        let mut graph = Graph::new();

        let activation = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[cols])));

        let weight = graph
            .add_weight(TensorDesc::quantized_weight(
                Shape::from_dims(&[rows, cols]),
                StorageRef::HostOwned {
                    id: StorageId::new(0, 0),
                },
                QuantFormatId::Q4K,
                block_size,
                version,
            ))
            .unwrap();

        let output = graph.add_qmatmul(
            activation,
            weight,
            TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
        );

        graph.try_set_output(output).unwrap();
        graph
    }

    #[test]
    fn sm61_is_rejected_before_lowering() {
        let error = CudaGraphCompiler::from_properties(properties(6, 1)).unwrap_err();

        assert_eq!(
            error,
            CudaCompileError::UnsupportedCapability {
                actual: CudaComputeCapability { major: 6, minor: 1 },
                required: CudaComputeCapability::SM70,
            },
        );
    }

    #[test]
    fn sm70_is_accepted() {
        let compiler = CudaGraphCompiler::from_properties(properties(7, 0)).unwrap();

        assert_eq!(compiler.device_ordinal(), 0);
        assert_eq!(
            compiler.properties().compute_capability,
            CudaComputeCapability::SM70,
        );
    }

    #[test]
    fn newer_compute_capability_is_accepted() {
        let compiler = CudaGraphCompiler::from_properties(properties(8, 9)).unwrap();

        assert_eq!(
            compiler.properties().compute_capability,
            CudaComputeCapability { major: 8, minor: 9 },
        );
    }

    #[test]
    fn one_by_256_q4k_identity_lowers() {
        let graph = q4k_graph(1, 256, 256, 1);
        let plan = compiler().compile_plan(&graph, vec![0.25; 256]).unwrap();

        assert_eq!(plan.device_ordinal(), 0);
        assert_eq!(plan.steps().len(), 1);
        assert_eq!(plan.weight_requirements().len(), 1);

        assert_eq!(
            plan.steps()[0],
            CudaCompiledStep::Q4kF32Decode(CudaCompiledQ4kF32Decode {
                weight: CudaWeightBindingId::new(0),
                input: CudaCompiledF32View {
                    buffer_index: 0,
                    len_elements: 256,
                },
                output: CudaCompiledF32View {
                    buffer_index: 1,
                    len_elements: 1,
                },
                rows: 1,
                cols: 256,
            }),
        );

        assert_eq!(plan.weight_requirements()[0].payload_bytes, 160);
    }

    #[test]
    fn two_by_512_payload_accounting_is_exact() {
        let graph = q4k_graph(2, 512, 256, 1);
        let plan = compiler().compile_plan(&graph, vec![0.25; 512]).unwrap();

        assert_eq!(plan.weight_requirements()[0].payload_bytes, 2 * 2 * 160);
        assert_eq!(
            plan.steps()[0],
            CudaCompiledStep::Q4kF32Decode(CudaCompiledQ4kF32Decode {
                weight: CudaWeightBindingId::new(0),
                input: CudaCompiledF32View {
                    buffer_index: 0,
                    len_elements: 512,
                },
                output: CudaCompiledF32View {
                    buffer_index: 1,
                    len_elements: 2,
                },
                rows: 2,
                cols: 512,
            }),
        );
    }

    #[test]
    fn wrong_input_length_is_rejected() {
        let graph = q4k_graph(1, 256, 256, 1);

        let error = compiler()
            .compile_plan(&graph, vec![0.25; 255])
            .unwrap_err();

        assert_eq!(
            error,
            CudaCompileError::InputLengthMismatch {
                expected: 256,
                actual: 255,
            },
        );
    }

    #[test]
    fn q4k_version_two_is_rejected() {
        let graph = q4k_graph(1, 256, 256, 2);

        assert_eq!(
            compiler()
                .compile_plan(&graph, vec![0.25; 256])
                .unwrap_err(),
            CudaCompileError::InvalidWeight,
        );
    }

    #[test]
    fn q4k_block_128_is_rejected() {
        let graph = q4k_graph(1, 256, 128, 1);

        assert_eq!(
            compiler()
                .compile_plan(&graph, vec![0.25; 256])
                .unwrap_err(),
            CudaCompileError::InvalidWeight,
        );
    }

    #[test]
    fn add_lowers_to_typed_cuda_step() {
        let mut graph = Graph::new();

        let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[256]));
        let input = graph.add_input(desc.clone());
        let output = graph.add_add(input, input, desc);

        graph.try_set_output(output).unwrap();

        let plan = compiler().compile_plan(&graph, vec![0.25; 256]).unwrap();
        assert_eq!(plan.weight_requirements().len(), 0);
        assert_eq!(plan.memory().buffers().len(), 2);
        assert_eq!(
            plan.steps()[0],
            CudaCompiledStep::AddF32(CudaCompiledAddF32 {
                a: CudaCompiledF32View {
                    buffer_index: 0,
                    len_elements: 256,
                },
                b: CudaCompiledF32View {
                    buffer_index: 0,
                    len_elements: 256,
                },
                output: CudaCompiledF32View {
                    buffer_index: 1,
                    len_elements: 256,
                },
                elements: 256,
            })
        );
    }

    #[test]
    fn multiple_add_nodes_lower_in_topological_order() {
        let mut graph = Graph::new();

        let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[256]));
        let input = graph.add_input(desc.clone());

        let intermediate = graph.add_add(input, input, desc.clone());
        let output = graph.add_add(intermediate, input, desc);

        graph.try_set_output(output).unwrap();

        let plan = compiler().compile_plan(&graph, vec![0.25; 256]).unwrap();
        assert_eq!(plan.steps().len(), 2);
        assert!(matches!(plan.steps()[0], CudaCompiledStep::AddF32(_)));
        assert!(matches!(plan.steps()[1], CudaCompiledStep::AddF32(_)));
        assert_eq!(plan.memory().buffers().len(), 3);
    }

    #[test]
    fn bias_is_rejected() {
        let mut graph = Graph::new();

        let activation = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[256])));
        let bias = graph.add_tensor(TensorDesc {
            storage: StorageRef::HostOwned {
                id: StorageId::new(1, 0),
            },
            ..TensorDesc::f32_contiguous(Shape::from_dims(&[2]))
        });
        let weight = graph
            .add_weight(TensorDesc::quantized_weight(
                Shape::from_dims(&[2, 256]),
                StorageRef::HostOwned {
                    id: StorageId::new(0, 0),
                },
                QuantFormatId::Q4K,
                256,
                1,
            ))
            .unwrap();

        let output = graph.add_qmatmul_bias(
            activation,
            weight,
            bias,
            TensorDesc::f32_contiguous(Shape::from_dims(&[2])),
        );
        graph.try_set_output(output).unwrap();

        assert_eq!(
            compiler()
                .compile_plan(&graph, vec![0.25; 256])
                .unwrap_err(),
            CudaCompileError::UnsupportedEpilogue {
                actual: QMatmulEpilogue::Bias,
            },
        );
    }

    #[test]
    fn bias_silu_is_rejected() {
        let mut graph = Graph::new();

        let activation = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[256])));
        let bias = graph.add_tensor(TensorDesc {
            storage: StorageRef::HostOwned {
                id: StorageId::new(1, 0),
            },
            ..TensorDesc::f32_contiguous(Shape::from_dims(&[2]))
        });
        let weight = graph
            .add_weight(TensorDesc::quantized_weight(
                Shape::from_dims(&[2, 256]),
                StorageRef::HostOwned {
                    id: StorageId::new(0, 0),
                },
                QuantFormatId::Q4K,
                256,
                1,
            ))
            .unwrap();

        let output = graph.add_qmatmul_bias_silu(
            activation,
            weight,
            bias,
            TensorDesc::f32_contiguous(Shape::from_dims(&[2])),
        );
        graph.try_set_output(output).unwrap();

        assert_eq!(
            compiler()
                .compile_plan(&graph, vec![0.25; 256])
                .unwrap_err(),
            CudaCompileError::UnsupportedEpilogue {
                actual: QMatmulEpilogue::BiasSilu,
            },
        );
    }

    #[test]
    fn rank_two_m1_activation_lowers_to_prefill() {
        let mut graph = Graph::new();

        let activation = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[1, 256])));

        let weight = graph
            .add_weight(TensorDesc::quantized_weight(
                Shape::from_dims(&[2, 256]),
                StorageRef::HostOwned {
                    id: StorageId::new(0, 0),
                },
                QuantFormatId::Q4K,
                256,
                1,
            ))
            .unwrap();

        let output = graph.add_qmatmul(
            activation,
            weight,
            TensorDesc::f32_contiguous(Shape::from_dims(&[1, 2])),
        );

        graph.try_set_output(output).unwrap();

        let plan = compiler().compile_plan(&graph, vec![0.25; 256]).unwrap();
        assert_eq!(
            plan.steps()[0],
            CudaCompiledStep::Q4kF32Prefill(CudaCompiledQ4kF32Prefill {
                weight: CudaWeightBindingId::new(0),
                input: CudaCompiledF32View {
                    buffer_index: 0,
                    len_elements: 256,
                },
                output: CudaCompiledF32View {
                    buffer_index: 1,
                    len_elements: 2,
                },
                m: 1,
                rows: 2,
                cols: 256,
            })
        );
    }

    #[test]
    fn dual_scale_weight_is_rejected() {
        let mut graph = Graph::new();

        let activation = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[256])));

        let weight = graph
            .add_weight(TensorDesc::quantized_weight(
                Shape::from_dims(&[2, 256]),
                StorageRef::HostOwned {
                    id: StorageId::new(0, 0),
                },
                QuantFormatId::DualScale,
                256,
                1,
            ))
            .unwrap();

        let output = graph.add_qmatmul(
            activation,
            weight,
            TensorDesc::f32_contiguous(Shape::from_dims(&[2])),
        );

        graph.try_set_output(output).unwrap();

        assert_eq!(
            compiler()
                .compile_plan(&graph, vec![0.25; 256])
                .unwrap_err(),
            CudaCompileError::InvalidWeight,
        );
    }

    #[test]
    fn non_host_owned_weight_is_rejected() {
        let mut graph = Graph::new();

        let activation = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[256])));

        let weight = graph
            .add_weight(TensorDesc::quantized_weight(
                Shape::from_dims(&[2, 256]),
                StorageRef::External {
                    id: StorageId::new(0, 0),
                },
                QuantFormatId::Q4K,
                256,
                1,
            ))
            .unwrap();

        let output = graph.add_qmatmul(
            activation,
            weight,
            TensorDesc::f32_contiguous(Shape::from_dims(&[2])),
        );

        graph.try_set_output(output).unwrap();

        assert_eq!(
            compiler()
                .compile_plan(&graph, vec![0.25; 256])
                .unwrap_err(),
            CudaCompileError::InvalidWeight,
        );
    }

    // Graph::add_qmatmul validates output rows == weight rows at construction
    // time, so wrong-output-shape cannot reach D1b lowering. The Graph layer
    // is the authoritative guard for this invariant.
}
