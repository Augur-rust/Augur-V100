use std::sync::Arc;

use augur_backend::{BackendCapabilities, CapabilityQuery, CpuBackend, CpuBuffer};
use augur_core::{CompileError, DType, FormatVersion, PhysicalLayoutId, QuantFormatId, TensorId};
use augur_exec::{BufferPurpose, BufferView};
use augur_graph::{AnalyzedGraph, FusedMlpActivation, FusedMlpAttrs, Node, Op, QMatmulEpilogue};
use augur_kernel_api::{
    CpuKernelAbi, DualScaleHotCpuBlock, EpilogueKind, Iq2XxsHotCpuBlock, KernelKey, OpKind,
    Q2KHotCpuBlock, Q4KHotCpuBlock, Q8_0HotCpuBlock, QuantizeQ8HotCpuLaunch, ShapeClass,
    MXFP4_HOT_CPU_BLOCK_LEN,
};
use augur_kernels::{cpu_kernel_registrations, portable::launch_quantize_q8k_hot_cpu_portable};

use crate::activation_reuse::{CpuActivationPreparationKey, PreparedQ8HotActivation};
use crate::{
    dual_scale_requirement, fp8_scale_grid_requirement, iq2_xxs_requirement, mxfp4_requirement,
    normalize_cpu_tuning_shape, q2k_requirement, q4k_requirement, q8_0_requirement,
    select_cpu_kernel, select_cpu_kernel_with_fallback_and_tuning, select_cpu_kernel_with_tuning,
    CompiledAdd, CompiledClamp, CompiledDualScaleQ8Hot, CompiledFp8FusedMlpF32,
    CompiledFp8ScaleGridF32, CompiledIq2XxsQ8Hot, CompiledMul, CompiledMxfp4F32, CompiledQ2KQ8Hot,
    CompiledQ8_0Q8Hot, CompiledQMatmulBiasF32, CompiledQMatmulF32, CompiledQMatmulF32Base,
    CompiledQMatmulF32Prefill, CompiledQuantizeQ8Hot, CompiledSilu, CpuCompiledPlan,
    CpuCompiledStep, CpuKernelRequest, CpuTuningRequestKey, LoweringError, MemoryPlanner,
    RawCpuTuningShape, RegistryInvariantError, ValidatedCpuTuningProfile, WeightRequirementBuilder,
};

pub struct LoweringContext<'g> {
    analyzed: AnalyzedGraph<'g>,
    capabilities: BackendCapabilities,
    memory: MemoryPlanner,
    buffers: Vec<CpuBuffer>,
    steps: Vec<CpuCompiledStep>,

    weight_requirements: WeightRequirementBuilder,

    /// Compiler-only interning table for generated Q8Hot representations.
    prepared_q8hot: Vec<PreparedQ8HotActivation>,

    cpu_tuning: Option<Arc<ValidatedCpuTuningProfile>>,
}

impl<'g> LoweringContext<'g> {
    pub fn new(
        analyzed: AnalyzedGraph<'g>,
        capabilities: BackendCapabilities,
        cpu_tuning: Option<Arc<ValidatedCpuTuningProfile>>,
        input_buffers: Vec<Vec<f32>>,
    ) -> Result<Self, CompileError> {
        let graph = analyzed.graph();

        if input_buffers.len() != graph.inputs().len() {
            return Err(CompileError::InvalidInput);
        }

        let mut memory = MemoryPlanner::new(graph.tensor_count());
        let mut buffers = Vec::with_capacity(graph.tensor_count());

        for (&tensor, input_buffer) in graph.inputs().iter().zip(input_buffers) {
            let desc = graph.tensor(tensor).ok_or(CompileError::InvalidInput)?;

            let element_count = desc.elem_count().ok_or(CompileError::InvalidInput)?;

            if input_buffer.len() != element_count {
                return Err(CompileError::InvalidInput);
            }

            if memory.next_buffer_index() != buffers.len() {
                return Err(CompileError::InvalidInput);
            }

            let owner = CpuBuffer::from_f32(input_buffer);

            let reusable_after_step = analyzed
                .liveness()
                .range(tensor)
                .map(|range| range.last_step);

            let buffer_index = memory.assign(
                tensor,
                element_count,
                BufferPurpose::Input,
                reusable_after_step,
            )?;

            debug_assert_eq!(buffer_index, buffers.len());
            buffers.push(owner);
        }

        Ok(Self {
            analyzed,
            capabilities,
            memory,
            buffers,
            steps: Vec::new(),
            weight_requirements: WeightRequirementBuilder::new(),
            prepared_q8hot: Vec::new(),
            cpu_tuning,
        })
    }

    fn tensor_elem_count(&self, tensor: TensorId) -> Result<usize, CompileError> {
        let desc = self
            .analyzed
            .graph()
            .tensor(tensor)
            .ok_or(CompileError::InvalidOutput)?;
        desc.elem_count().ok_or(CompileError::InvalidOutput)
    }

    fn is_output(&self, tensor: TensorId) -> bool {
        self.analyzed.graph().outputs().contains(&tensor)
    }

    fn tensor_plain_f32(&self, tensor: TensorId) -> Result<(), CompileError> {
        let desc = self
            .analyzed
            .graph()
            .tensor(tensor)
            .ok_or(CompileError::InvalidInput)?;
        if !desc.is_plain_f32() {
            return Err(CompileError::ExpectedPlainF32QMatmulInput {
                actual_dtype: desc.dtype,
                actual_quant: desc.quant.as_ref().map(|quant| quant.format),
            });
        }
        Ok(())
    }

    fn tensor_is_decode_vector(&self, tensor: TensorId) -> Result<bool, CompileError> {
        let desc = self
            .analyzed
            .graph()
            .tensor(tensor)
            .ok_or(CompileError::InvalidInput)?;
        Ok(desc.shape.rank() == 1)
    }

    fn qmatmul_workload(&self, tensor: TensorId) -> Result<(usize, bool), CompileError> {
        let desc = self
            .analyzed
            .graph()
            .tensor(tensor)
            .ok_or(CompileError::InvalidInput)?;
        match desc.shape.dims() {
            [_] => Ok((1, false)),
            [m, _] if *m > 0 => Ok((*m, true)),
            _ => Err(CompileError::InvalidInput),
        }
    }

    fn buffer_for(&self, tensor: TensorId) -> Result<usize, CompileError> {
        self.memory
            .buffer_for(tensor)
            .ok_or(CompileError::InvalidInput)
    }

    fn ensure_buffer_tables_synchronized(&self) -> Result<(), LoweringError> {
        if self.memory.next_buffer_index() != self.buffers.len() {
            return Err(LoweringError::CompiledPlanInvariant);
        }
        Ok(())
    }

    fn allocate_tensor(
        &mut self,
        tensor: TensorId,
        purpose: BufferPurpose,
    ) -> Result<usize, LoweringError> {
        let element_count = self.tensor_elem_count(tensor)?;

        let reusable_after_step = self
            .analyzed
            .liveness()
            .range(tensor)
            .map(|range| range.last_step);

        // Allocate the concrete owner before mutating MemoryPlanner.
        let owner = CpuBuffer::zeros_f32(element_count);

        // Report synchronization failure before either table is mutated.
        self.ensure_buffer_tables_synchronized()?;

        let buffer_index =
            self.memory
                .assign(tensor, element_count, purpose, reusable_after_step)?;

        debug_assert_eq!(buffer_index, self.buffers.len());
        self.buffers.push(owner);

        Ok(buffer_index)
    }

    fn supports_elementwise_f32_contiguous(&self, op: OpKind) -> bool {
        self.capabilities.supports(CapabilityQuery {
            op,
            dtype: DType::F32,
            quant: None,
            layout: PhysicalLayoutId::HOST_CONTIGUOUS_F32,
            epilogue: EpilogueKind::Identity,
        })
    }

    fn qmatmul_weight_desc(
        &self,
        node: &Node,
    ) -> Result<
        (
            augur_core::StorageId,
            usize,
            usize,
            QuantFormatId,
            usize,
            FormatVersion,
        ),
        CompileError,
    > {
        let weight_tensor = *node.inputs.get(1).ok_or(CompileError::InvalidInput)?;
        self.weight_desc(weight_tensor)
    }

    fn weight_desc(
        &self,
        weight_tensor: TensorId,
    ) -> Result<
        (
            augur_core::StorageId,
            usize,
            usize,
            QuantFormatId,
            usize,
            FormatVersion,
        ),
        CompileError,
    > {
        let desc = self
            .analyzed
            .graph()
            .tensor(weight_tensor)
            .ok_or(CompileError::InvalidInput)?;

        if desc.dtype != DType::PackedU8 || desc.shape.rank() != 2 {
            return Err(CompileError::InvalidInput);
        }

        let quant = desc.quant.as_ref().ok_or(CompileError::InvalidInput)?;
        let storage = match desc.storage {
            augur_core::StorageRef::HostOwned { id } => id,
            _ => return Err(CompileError::InvalidInput),
        };

        let dims = desc.shape.dims();
        Ok((
            storage,
            dims[0],
            dims[1],
            quant.format,
            quant.block,
            FormatVersion(quant.version),
        ))
    }

    /// Returns one exact reusable Q8Hot representation for an immediate
    /// compiled consumer.
    ///
    /// Lifetime semantics:
    ///
    /// - a newly emitted `QuantizeQ8Hot` step is the producer;
    /// - after optional producer insertion, `self.steps.len()` is the
    ///   immediate consumer's compiled-step index;
    /// - `extend_temporary_lifetime` records that consumer index inclusively;
    /// - repeated consumers monotonically extend the same temporary.
    ///
    /// On success, the caller must immediately append the corresponding
    /// consumer. It must not perform another fallible lowering operation or
    /// emit an intervening compiled step.
    fn tuning_shape(rows: usize, cols: usize) -> Result<crate::CpuTuningShape, LoweringError> {
        normalize_cpu_tuning_shape(RawCpuTuningShape {
            batch: 1,
            m: 1,
            n: rows,
            k: cols,
        })
        .map_err(LoweringError::CpuTuningShape)
    }

    fn q8hot_for_immediate_consumer(
        &mut self,
        source_tensor: TensorId,
        source_buffer_index: usize,
        len_elements: usize,
    ) -> Result<usize, LoweringError> {
        let source_view = BufferView::<CpuBackend>::dense_f32(source_buffer_index, len_elements);

        let key = CpuActivationPreparationKey::q8k_hot_cpu_portable_v1(source_tensor, &source_view)
            .ok_or(LoweringError::CompiledPlanInvariant)?;

        let existing = self
            .prepared_q8hot
            .iter()
            .find(|prepared| prepared.key == key)
            .copied();

        let prepared = if let Some(prepared) = existing {
            prepared
        } else {
            // Construct the concrete owner before mutating either planner or
            // compiled-step state.
            let owner = CpuBuffer::new_q8hot_zeroed(len_elements)
                .map_err(|_| LoweringError::Core(CompileError::InvalidInput))?;

            // Validate the parallel planner/owner tables before either table
            // is changed.
            self.ensure_buffer_tables_synchronized()?;

            let producer_step = self.steps.len();
            let (temporary, buffer_index) = self.memory.assign_temporary(
                len_elements,
                BufferPurpose::HotActivation,
                producer_step,
            )?;

            // next_buffer_index() was checked against buffers.len() before
            // assign_temporary. This assertion documents that invariant and
            // is not a post-mutation recovery mechanism.
            debug_assert_eq!(buffer_index, self.buffers.len());
            self.buffers.push(owner);

            let quantize_launch: QuantizeQ8HotCpuLaunch = launch_quantize_q8k_hot_cpu_portable;

            self.steps
                .push(CpuCompiledStep::QuantizeQ8Hot(CompiledQuantizeQ8Hot {
                    input: source_view,
                    output: BufferView::<CpuBackend>::q8hot_cpu(buffer_index, len_elements),
                    launch: quantize_launch,
                }));

            let prepared = PreparedQ8HotActivation {
                key,
                temporary,
                buffer_index,
            };
            self.prepared_q8hot.push(prepared);
            prepared
        };

        // After optional Quantize insertion, this is exactly the inclusive
        // compiled-step index of the consumer that must be emitted next.
        let consumer_step = self.steps.len();
        self.memory
            .extend_temporary_lifetime(prepared.temporary, consumer_step)?;

        Ok(prepared.buffer_index)
    }
}

pub struct CpuLowerer<'g> {
    ctx: LoweringContext<'g>,
}

impl<'g> CpuLowerer<'g> {
    pub fn new(ctx: LoweringContext<'g>) -> Self {
        Self { ctx }
    }

    pub fn lower(mut self) -> Result<CpuCompiledPlan, LoweringError> {
        let graph = self.ctx.analyzed.graph();

        for topo_index in 0..self.ctx.analyzed.topo().len() {
            let node_id = self.ctx.analyzed.topo()[topo_index];
            let node = &graph.node(node_id).ok_or(CompileError::InvalidNode)?.node;

            match &node.op {
                Op::Add => self.lower_add(node)?,
                Op::Mul => self.lower_mul(node)?,
                Op::Clamp(attrs) => self.lower_clamp(node, attrs.min(), attrs.max())?,
                Op::Silu => self.lower_silu(node)?,
                Op::QMatmul(_) => self.lower_qmatmul(node)?,
                Op::FusedMlp(attrs) => self.lower_fused_mlp(node, attrs)?,
                Op::OrderedTopK(_) => return Err(LoweringError::Core(CompileError::UnsupportedOp)),
                _ => return Err(LoweringError::Core(CompileError::UnsupportedOp)),
            }
        }

        let LoweringContext {
            steps,
            buffers,
            memory,
            weight_requirements,
            ..
        } = self.ctx;

        Ok(CpuCompiledPlan {
            steps: steps.into_boxed_slice(),
            buffers,
            memory: memory.finish(),
            weight_requirements: weight_requirements.finish(),
        })
    }

    fn lower_add(&mut self, node: &Node) -> Result<(), LoweringError> {
        if !self.ctx.supports_elementwise_f32_contiguous(OpKind::Add) {
            return Err(LoweringError::Core(CompileError::BackendUnsupported));
        }

        let a = node.inputs[0];
        let b = node.inputs[1];
        let output = node.outputs[0];
        let n = self.ctx.tensor_elem_count(output)?;

        let purpose = if self.ctx.is_output(output) {
            BufferPurpose::Output
        } else {
            BufferPurpose::Intermediate
        };

        let a_buffer = self.ctx.buffer_for(a)?;
        let b_buffer = self.ctx.buffer_for(b)?;
        let output_buffer = self.ctx.allocate_tensor(output, purpose)?;

        self.ctx.steps.push(CpuCompiledStep::Add(CompiledAdd {
            dst: BufferView::dense_f32(output_buffer, n),
            a: BufferView::dense_f32(a_buffer, n),
            b: BufferView::dense_f32(b_buffer, n),
            n,
        }));

        Ok(())
    }

    fn lower_mul(&mut self, node: &Node) -> Result<(), LoweringError> {
        if !self.ctx.supports_elementwise_f32_contiguous(OpKind::Mul) {
            return Err(LoweringError::Core(CompileError::BackendUnsupported));
        }
        let a = node.inputs[0];
        let b = node.inputs[1];
        let output = node.outputs[0];
        let n = self.ctx.tensor_elem_count(output)?;
        let purpose = if self.ctx.is_output(output) {
            BufferPurpose::Output
        } else {
            BufferPurpose::Intermediate
        };
        let a_buffer = self.ctx.buffer_for(a)?;
        let b_buffer = self.ctx.buffer_for(b)?;
        let output_buffer = self.ctx.allocate_tensor(output, purpose)?;
        self.ctx.steps.push(CpuCompiledStep::Mul(CompiledMul {
            dst: BufferView::dense_f32(output_buffer, n),
            a: BufferView::dense_f32(a_buffer, n),
            b: BufferView::dense_f32(b_buffer, n),
            n,
        }));
        Ok(())
    }

    fn lower_clamp(
        &mut self,
        node: &Node,
        min: Option<f32>,
        max: Option<f32>,
    ) -> Result<(), LoweringError> {
        if !self.ctx.supports_elementwise_f32_contiguous(OpKind::Clamp) {
            return Err(LoweringError::Core(CompileError::BackendUnsupported));
        }
        let input = node.inputs[0];
        let output = node.outputs[0];
        let n = self.ctx.tensor_elem_count(output)?;
        let purpose = if self.ctx.is_output(output) {
            BufferPurpose::Output
        } else {
            BufferPurpose::Intermediate
        };
        let input_buffer = self.ctx.buffer_for(input)?;
        let output_buffer = self.ctx.allocate_tensor(output, purpose)?;
        self.ctx.steps.push(CpuCompiledStep::Clamp(CompiledClamp {
            dst: BufferView::dense_f32(output_buffer, n),
            input: BufferView::dense_f32(input_buffer, n),
            n,
            min,
            max,
        }));
        Ok(())
    }

    fn lower_silu(&mut self, node: &Node) -> Result<(), LoweringError> {
        if !self.ctx.supports_elementwise_f32_contiguous(OpKind::Silu) {
            return Err(LoweringError::Core(CompileError::BackendUnsupported));
        }
        let input = node.inputs[0];
        let output = node.outputs[0];
        let n = self.ctx.tensor_elem_count(output)?;
        let purpose = if self.ctx.is_output(output) {
            BufferPurpose::Output
        } else {
            BufferPurpose::Intermediate
        };
        let input_buffer = self.ctx.buffer_for(input)?;
        let output_buffer = self.ctx.allocate_tensor(output, purpose)?;
        self.ctx.steps.push(CpuCompiledStep::Silu(CompiledSilu {
            dst: BufferView::dense_f32(output_buffer, n),
            input: BufferView::dense_f32(input_buffer, n),
            n,
        }));
        Ok(())
    }

    fn lower_qmatmul(&mut self, node: &Node) -> Result<(), LoweringError> {
        let attrs = match &node.op {
            Op::QMatmul(attrs) => attrs,
            _ => return Err(LoweringError::Core(CompileError::InvalidNode)),
        };

        let input = node.inputs[0];
        self.ctx.tensor_plain_f32(input)?;
        let (m, is_prefill) = self.ctx.qmatmul_workload(input)?;

        let (weight_storage, rows, cols, weight_format, block_size, format_version) =
            self.ctx.qmatmul_weight_desc(node)?;

        match weight_format {
            QuantFormatId::Q4K => self.lower_q4k_qmatmul(
                node,
                input,
                weight_storage,
                rows,
                cols,
                block_size,
                format_version,
                attrs.epilogue,
                m,
                is_prefill,
            ),

            QuantFormatId::DualScale => self.lower_dual_scale_q8hot(
                node,
                input,
                weight_storage,
                rows,
                cols,
                block_size,
                format_version,
                attrs.epilogue,
            ),

            QuantFormatId::Q8_0 => self.lower_q8_0_q8hot(
                node,
                input,
                weight_storage,
                rows,
                cols,
                block_size,
                format_version,
                attrs.epilogue,
            ),

            QuantFormatId::Q2K => self.lower_q2k_q8hot(
                node,
                input,
                weight_storage,
                rows,
                cols,
                block_size,
                format_version,
                attrs.epilogue,
            ),

            QuantFormatId::IQ2_XXS => self.lower_iq2_xxs_q8hot(
                node,
                input,
                weight_storage,
                rows,
                cols,
                block_size,
                format_version,
                attrs.epilogue,
            ),

            QuantFormatId::MXFP4 => self.lower_mxfp4_f32(
                node,
                input,
                weight_storage,
                rows,
                cols,
                block_size,
                format_version,
                attrs.epilogue,
            ),

            QuantFormatId::FP8E4M3ScaleGrid => self.lower_fp8_scale_grid_f32(
                node,
                input,
                weight_storage,
                rows,
                cols,
                block_size,
                format_version,
                attrs.epilogue,
            ),

            _ => Err(LoweringError::Core(CompileError::BackendUnsupported)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_q4k_qmatmul(
        &mut self,
        node: &Node,
        input: TensorId,
        weight_storage: augur_core::StorageId,
        rows: usize,
        cols: usize,
        block_size: usize,
        format_version: FormatVersion,
        epilogue: QMatmulEpilogue,
        m: usize,
        is_prefill: bool,
    ) -> Result<(), LoweringError> {
        if block_size != Q4KHotCpuBlock::DECODED_LEN || format_version != FormatVersion(1) {
            return Err(LoweringError::Core(CompileError::BackendUnsupported));
        }

        let binding = self.ctx.weight_requirements.intern(q4k_requirement(
            weight_storage,
            format_version,
            block_size,
            rows,
            cols,
        ))?;

        let output = node.outputs[0];

        let purpose = if self.ctx.is_output(output) {
            BufferPurpose::Output
        } else {
            BufferPurpose::Intermediate
        };

        let input_buffer = self.ctx.buffer_for(input)?;
        let output_buffer = self.ctx.allocate_tensor(output, purpose)?;

        if is_prefill {
            if epilogue != QMatmulEpilogue::Identity {
                return Err(LoweringError::Core(CompileError::BackendUnsupported));
            }
            let input_len = m
                .checked_mul(cols)
                .ok_or(LoweringError::Core(CompileError::BackendUnsupported))?;
            let output_len = m
                .checked_mul(rows)
                .ok_or(LoweringError::Core(CompileError::BackendUnsupported))?;
            let request = CpuKernelRequest {
                key: KernelKey::q4k_f32_cpu_portable(EpilogueKind::Identity),
                shape: ShapeClass::classify_prefill(m, rows, cols),
                expected_abi: CpuKernelAbi::QMatmulF32Prefill,
            };
            let selected =
                select_cpu_kernel(cpu_kernel_registrations(), request, &self.ctx.capabilities)?;
            let launch = selected.launch.qmatmul_f32_prefill().ok_or_else(|| {
                LoweringError::RegistryInvariant(RegistryInvariantError::UnexpectedLaunchAbi {
                    registration: selected.id,
                    expected: CpuKernelAbi::QMatmulF32Prefill,
                    actual: selected.launch.abi(),
                })
            })?;
            self.ctx.steps.push(CpuCompiledStep::QMatmulF32Prefill(
                CompiledQMatmulF32Prefill {
                    weight: binding,
                    input: BufferView::dense_f32(input_buffer, input_len),
                    output: BufferView::dense_f32(output_buffer, output_len),
                    m,
                    rows,
                    cols,
                    launch,
                },
            ));
            return Ok(());
        }

        let base = CompiledQMatmulF32Base {
            weight: binding,
            input: BufferView::dense_f32(input_buffer, cols),
            output: BufferView::dense_f32(output_buffer, rows),
            rows,
            cols,
        };

        let (registry_epilogue, expected_abi) = match epilogue {
            QMatmulEpilogue::Identity => (EpilogueKind::Identity, CpuKernelAbi::QMatmulF32),
            QMatmulEpilogue::Bias => (EpilogueKind::Bias, CpuKernelAbi::QMatmulBiasF32),
            QMatmulEpilogue::BiasSilu => (EpilogueKind::BiasSilu, CpuKernelAbi::QMatmulBiasF32),
        };

        let shape = ShapeClass::classify_matmul(1, rows, cols, 1);
        let tuning_shape = LoweringContext::tuning_shape(rows, cols)?;

        let portable_request = CpuKernelRequest {
            key: KernelKey::q4k_f32_cpu_portable(registry_epilogue),
            shape,
            expected_abi,
        };

        let portable_tuning_key = CpuTuningRequestKey {
            kernel: portable_request.key,
            shape: tuning_shape,
            expected_abi: portable_request.expected_abi,
        };

        let selected = match epilogue {
            QMatmulEpilogue::Identity => {
                let avx2_request = CpuKernelRequest {
                    key: KernelKey::q4k_f32_cpu_avx2_identity(),
                    shape,
                    expected_abi: CpuKernelAbi::QMatmulF32,
                };

                let avx2_tuning_key = CpuTuningRequestKey {
                    kernel: avx2_request.key,
                    shape: tuning_shape,
                    expected_abi: avx2_request.expected_abi,
                };

                select_cpu_kernel_with_fallback_and_tuning(
                    cpu_kernel_registrations(),
                    avx2_request,
                    avx2_tuning_key,
                    &[(portable_request, portable_tuning_key)],
                    &self.ctx.capabilities,
                    self.ctx.cpu_tuning.as_deref(),
                )?
            }

            QMatmulEpilogue::Bias | QMatmulEpilogue::BiasSilu => select_cpu_kernel_with_tuning(
                cpu_kernel_registrations(),
                portable_request,
                portable_tuning_key,
                &self.ctx.capabilities,
                self.ctx.cpu_tuning.as_deref(),
            )?,
        };

        let selected = selected.registration;

        match epilogue {
            QMatmulEpilogue::Identity => {
                let launch = selected.launch.qmatmul_f32().ok_or_else(|| {
                    LoweringError::RegistryInvariant(RegistryInvariantError::UnexpectedLaunchAbi {
                        registration: selected.id,
                        expected: CpuKernelAbi::QMatmulF32,
                        actual: selected.launch.abi(),
                    })
                })?;
                self.ctx
                    .steps
                    .push(CpuCompiledStep::QMatmulF32(CompiledQMatmulF32 {
                        base,
                        launch,
                    }));
            }
            QMatmulEpilogue::Bias => {
                let launch = selected.launch.qmatmul_bias_f32().ok_or_else(|| {
                    LoweringError::RegistryInvariant(RegistryInvariantError::UnexpectedLaunchAbi {
                        registration: selected.id,
                        expected: CpuKernelAbi::QMatmulBiasF32,
                        actual: selected.launch.abi(),
                    })
                })?;
                let bias = node.inputs[2];
                let bias_buffer = self.ctx.buffer_for(bias)?;
                self.ctx
                    .steps
                    .push(CpuCompiledStep::QMatmulBiasF32(CompiledQMatmulBiasF32 {
                        base,
                        bias: BufferView::dense_f32(bias_buffer, rows),
                        launch,
                    }));
            }
            QMatmulEpilogue::BiasSilu => {
                let launch = selected.launch.qmatmul_bias_f32().ok_or_else(|| {
                    LoweringError::RegistryInvariant(RegistryInvariantError::UnexpectedLaunchAbi {
                        registration: selected.id,
                        expected: CpuKernelAbi::QMatmulBiasF32,
                        actual: selected.launch.abi(),
                    })
                })?;
                let bias = node.inputs[2];
                let bias_buffer = self.ctx.buffer_for(bias)?;
                self.ctx.steps.push(CpuCompiledStep::QMatmulBiasSiluF32(
                    CompiledQMatmulBiasF32 {
                        base,
                        bias: BufferView::dense_f32(bias_buffer, rows),
                        launch,
                    },
                ));
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_dual_scale_q8hot(
        &mut self,
        node: &Node,
        input: TensorId,
        weight_storage: augur_core::StorageId,
        rows: usize,
        cols: usize,
        block_size: usize,
        format_version: FormatVersion,
        epilogue: QMatmulEpilogue,
    ) -> Result<(), LoweringError> {
        let epilogue_kind = graph_epilogue_kind(epilogue);

        if epilogue != QMatmulEpilogue::Identity {
            return Err(LoweringError::UnsupportedDualScaleEpilogue {
                epilogue: epilogue_kind,
            });
        }

        if !self.ctx.tensor_is_decode_vector(input)? {
            return Err(LoweringError::Core(CompileError::BackendUnsupported));
        }

        if block_size != DualScaleHotCpuBlock::DECODED_LEN
            || format_version != FormatVersion(1)
            || rows == 0
            || cols == 0
            || !cols.is_multiple_of(DualScaleHotCpuBlock::DECODED_LEN)
        {
            return Err(LoweringError::Core(CompileError::BackendUnsupported));
        }

        let blocks_per_row = cols / DualScaleHotCpuBlock::DECODED_LEN;

        if blocks_per_row == 0 {
            return Err(LoweringError::Core(CompileError::BackendUnsupported));
        }

        // Complete registry selection before changing compiled-plan buffers or
        // preparation state. M28-D: explicit AVX2 → Portable fallback chain.
        // M29-B: tuned selection reuses exact-key fallback with optional profile.
        let shape = ShapeClass::classify_matmul(1, rows, cols, 1);
        let tuning_shape = LoweringContext::tuning_shape(rows, cols)?;

        let avx2_request = CpuKernelRequest {
            key: KernelKey::dual_scale_q8hot_cpu_avx2_identity(),
            shape,
            expected_abi: CpuKernelAbi::DualScaleQ8Hot,
        };

        let avx2_tuning_key = CpuTuningRequestKey {
            kernel: avx2_request.key,
            shape: tuning_shape,
            expected_abi: avx2_request.expected_abi,
        };

        let portable_request = CpuKernelRequest {
            key: KernelKey::dual_scale_q8hot_cpu_portable(),
            shape,
            expected_abi: CpuKernelAbi::DualScaleQ8Hot,
        };

        let portable_tuning_key = CpuTuningRequestKey {
            kernel: portable_request.key,
            shape: tuning_shape,
            expected_abi: portable_request.expected_abi,
        };

        let selected = select_cpu_kernel_with_fallback_and_tuning(
            cpu_kernel_registrations(),
            avx2_request,
            avx2_tuning_key,
            &[(portable_request, portable_tuning_key)],
            &self.ctx.capabilities,
            self.ctx.cpu_tuning.as_deref(),
        )?;

        let selected = selected.registration;

        let launch = selected.launch.dual_scale_q8hot().ok_or_else(|| {
            LoweringError::RegistryInvariant(RegistryInvariantError::UnexpectedLaunchAbi {
                registration: selected.id,
                expected: CpuKernelAbi::DualScaleQ8Hot,
                actual: selected.launch.abi(),
            })
        })?;

        let input_buffer = self.ctx.buffer_for(input)?;

        let binding = self.ctx.weight_requirements.intern(dual_scale_requirement(
            weight_storage,
            format_version,
            block_size,
            rows,
            cols,
        ))?;

        let output = node.outputs[0];

        let purpose = if self.ctx.is_output(output) {
            BufferPurpose::Output
        } else {
            BufferPurpose::Intermediate
        };

        // This fallible allocation must occur before preparation insertion.
        // Otherwise a failed output allocation could leave an orphan
        // QuantizeQ8Hot step in the lowering context.
        let output_buffer = self.ctx.allocate_tensor(output, purpose)?;

        // This is the final fallible lowering operation before consumer
        // emission. On success, no intervening operation may be inserted.
        let q8hot_buffer = self
            .ctx
            .q8hot_for_immediate_consumer(input, input_buffer, cols)?;

        self.ctx
            .steps
            .push(CpuCompiledStep::DualScaleQ8Hot(CompiledDualScaleQ8Hot {
                weight: binding,
                activation: BufferView::q8hot_cpu(q8hot_buffer, cols),
                output: BufferView::dense_f32(output_buffer, rows),
                rows,
                cols,
                blocks_per_row,
                launch,
            }));

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_q8_0_q8hot(
        &mut self,
        node: &Node,
        input: TensorId,
        weight_storage: augur_core::StorageId,
        rows: usize,
        cols: usize,
        block_size: usize,
        format_version: FormatVersion,
        epilogue: QMatmulEpilogue,
    ) -> Result<(), LoweringError> {
        if epilogue != QMatmulEpilogue::Identity
            || !self.ctx.tensor_is_decode_vector(input)?
            || block_size != Q8_0HotCpuBlock::DECODED_LEN
            || format_version != FormatVersion(1)
            || rows == 0
            || cols == 0
            || !cols.is_multiple_of(Q8_0HotCpuBlock::DECODED_LEN)
        {
            return Err(LoweringError::Core(CompileError::BackendUnsupported));
        }
        let selected = select_cpu_kernel(
            cpu_kernel_registrations(),
            CpuKernelRequest {
                key: KernelKey::q8_0_q8hot_cpu_portable(),
                shape: ShapeClass::classify_matmul(1, rows, cols, 1),
                expected_abi: CpuKernelAbi::Q8_0Q8Hot,
            },
            &self.ctx.capabilities,
        )?;
        let launch = selected.launch.q8_0_q8hot().ok_or_else(|| {
            LoweringError::RegistryInvariant(RegistryInvariantError::UnexpectedLaunchAbi {
                registration: selected.id,
                expected: CpuKernelAbi::Q8_0Q8Hot,
                actual: selected.launch.abi(),
            })
        })?;
        let binding = self.ctx.weight_requirements.intern(q8_0_requirement(
            weight_storage,
            format_version,
            block_size,
            rows,
            cols,
        ))?;
        let input_buffer = self.ctx.buffer_for(input)?;
        let output = node.outputs[0];
        let purpose = if self.ctx.is_output(output) {
            BufferPurpose::Output
        } else {
            BufferPurpose::Intermediate
        };
        let output_buffer = self.ctx.allocate_tensor(output, purpose)?;
        let q8hot_buffer = self
            .ctx
            .q8hot_for_immediate_consumer(input, input_buffer, cols)?;
        self.ctx
            .steps
            .push(CpuCompiledStep::Q8_0Q8Hot(CompiledQ8_0Q8Hot {
                weight: binding,
                activation: BufferView::q8hot_cpu(q8hot_buffer, cols),
                output: BufferView::dense_f32(output_buffer, rows),
                rows,
                cols,
                launch,
            }));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_q2k_q8hot(
        &mut self,
        node: &Node,
        input: TensorId,
        weight_storage: augur_core::StorageId,
        rows: usize,
        cols: usize,
        block_size: usize,
        format_version: FormatVersion,
        epilogue: QMatmulEpilogue,
    ) -> Result<(), LoweringError> {
        if epilogue != QMatmulEpilogue::Identity
            || !self.ctx.tensor_is_decode_vector(input)?
            || block_size != Q2KHotCpuBlock::DECODED_LEN
            || format_version != FormatVersion(1)
            || rows == 0
            || cols == 0
            || !cols.is_multiple_of(Q2KHotCpuBlock::DECODED_LEN)
        {
            return Err(LoweringError::Core(CompileError::BackendUnsupported));
        }
        let selected = select_cpu_kernel(
            cpu_kernel_registrations(),
            CpuKernelRequest {
                key: KernelKey::q2k_q8hot_cpu_portable(),
                shape: ShapeClass::classify_matmul(1, rows, cols, 1),
                expected_abi: CpuKernelAbi::Q2KQ8Hot,
            },
            &self.ctx.capabilities,
        )?;
        let launch = selected.launch.q2k_q8hot().ok_or_else(|| {
            LoweringError::RegistryInvariant(RegistryInvariantError::UnexpectedLaunchAbi {
                registration: selected.id,
                expected: CpuKernelAbi::Q2KQ8Hot,
                actual: selected.launch.abi(),
            })
        })?;
        let binding = self.ctx.weight_requirements.intern(q2k_requirement(
            weight_storage,
            format_version,
            block_size,
            rows,
            cols,
        ))?;
        let input_buffer = self.ctx.buffer_for(input)?;
        let output = node.outputs[0];
        let purpose = if self.ctx.is_output(output) {
            BufferPurpose::Output
        } else {
            BufferPurpose::Intermediate
        };
        let output_buffer = self.ctx.allocate_tensor(output, purpose)?;
        let q8hot_buffer = self
            .ctx
            .q8hot_for_immediate_consumer(input, input_buffer, cols)?;
        self.ctx
            .steps
            .push(CpuCompiledStep::Q2KQ8Hot(CompiledQ2KQ8Hot {
                weight: binding,
                activation: BufferView::q8hot_cpu(q8hot_buffer, cols),
                output: BufferView::dense_f32(output_buffer, rows),
                rows,
                cols,
                launch,
            }));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_iq2_xxs_q8hot(
        &mut self,
        node: &Node,
        input: TensorId,
        weight_storage: augur_core::StorageId,
        rows: usize,
        cols: usize,
        block_size: usize,
        format_version: FormatVersion,
        epilogue: QMatmulEpilogue,
    ) -> Result<(), LoweringError> {
        if epilogue != QMatmulEpilogue::Identity
            || !self.ctx.tensor_is_decode_vector(input)?
            || block_size != Iq2XxsHotCpuBlock::DECODED_LEN
            || format_version != FormatVersion(1)
            || rows == 0
            || cols == 0
            || !cols.is_multiple_of(Iq2XxsHotCpuBlock::DECODED_LEN)
        {
            return Err(LoweringError::Core(CompileError::BackendUnsupported));
        }
        let selected = select_cpu_kernel(
            cpu_kernel_registrations(),
            CpuKernelRequest {
                key: KernelKey::iq2_xxs_q8hot_cpu_portable(),
                shape: ShapeClass::classify_matmul(1, rows, cols, 1),
                expected_abi: CpuKernelAbi::Iq2XxsQ8Hot,
            },
            &self.ctx.capabilities,
        )?;
        let launch = selected.launch.iq2_xxs_q8hot().ok_or_else(|| {
            LoweringError::RegistryInvariant(RegistryInvariantError::UnexpectedLaunchAbi {
                registration: selected.id,
                expected: CpuKernelAbi::Iq2XxsQ8Hot,
                actual: selected.launch.abi(),
            })
        })?;
        let binding = self.ctx.weight_requirements.intern(iq2_xxs_requirement(
            weight_storage,
            format_version,
            block_size,
            rows,
            cols,
        ))?;
        let input_buffer = self.ctx.buffer_for(input)?;
        let output = node.outputs[0];
        let purpose = if self.ctx.is_output(output) {
            BufferPurpose::Output
        } else {
            BufferPurpose::Intermediate
        };
        let output_buffer = self.ctx.allocate_tensor(output, purpose)?;
        let q8hot_buffer = self
            .ctx
            .q8hot_for_immediate_consumer(input, input_buffer, cols)?;
        self.ctx
            .steps
            .push(CpuCompiledStep::Iq2XxsQ8Hot(CompiledIq2XxsQ8Hot {
                weight: binding,
                activation: BufferView::q8hot_cpu(q8hot_buffer, cols),
                output: BufferView::dense_f32(output_buffer, rows),
                rows,
                cols,
                launch,
            }));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_mxfp4_f32(
        &mut self,
        node: &Node,
        input: TensorId,
        weight_storage: augur_core::StorageId,
        rows: usize,
        cols: usize,
        block_size: usize,
        format_version: FormatVersion,
        epilogue: QMatmulEpilogue,
    ) -> Result<(), LoweringError> {
        if epilogue != QMatmulEpilogue::Identity
            || !self.ctx.tensor_is_decode_vector(input)?
            || block_size != MXFP4_HOT_CPU_BLOCK_LEN
            || format_version != FormatVersion(1)
            || rows == 0
            || cols == 0
            || !cols.is_multiple_of(MXFP4_HOT_CPU_BLOCK_LEN)
        {
            return Err(LoweringError::Core(CompileError::BackendUnsupported));
        }
        let selected = select_cpu_kernel(
            cpu_kernel_registrations(),
            CpuKernelRequest {
                key: KernelKey::mxfp4_f32_cpu_portable(),
                shape: ShapeClass::classify_matmul(1, rows, cols, 1),
                expected_abi: CpuKernelAbi::Mxfp4F32,
            },
            &self.ctx.capabilities,
        )?;
        let launch = selected.launch.mxfp4_f32().ok_or_else(|| {
            LoweringError::RegistryInvariant(RegistryInvariantError::UnexpectedLaunchAbi {
                registration: selected.id,
                expected: CpuKernelAbi::Mxfp4F32,
                actual: selected.launch.abi(),
            })
        })?;
        let binding = self.ctx.weight_requirements.intern(mxfp4_requirement(
            weight_storage,
            format_version,
            block_size,
            rows,
            cols,
        ))?;
        let input_buffer = self.ctx.buffer_for(input)?;
        let output = node.outputs[0];
        let purpose = if self.ctx.is_output(output) {
            BufferPurpose::Output
        } else {
            BufferPurpose::Intermediate
        };
        let output_buffer = self.ctx.allocate_tensor(output, purpose)?;
        self.ctx
            .steps
            .push(CpuCompiledStep::Mxfp4F32(CompiledMxfp4F32 {
                weight: binding,
                input: BufferView::dense_f32(input_buffer, cols),
                output: BufferView::dense_f32(output_buffer, rows),
                rows,
                cols,
                launch,
            }));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_fp8_scale_grid_f32(
        &mut self,
        node: &Node,
        input: TensorId,
        weight_storage: augur_core::StorageId,
        rows: usize,
        cols: usize,
        block_size: usize,
        format_version: FormatVersion,
        epilogue: QMatmulEpilogue,
    ) -> Result<(), LoweringError> {
        if epilogue != QMatmulEpilogue::Identity
            || !self.ctx.tensor_is_decode_vector(input)?
            || block_size != augur_kernel_api::FP8_E4M3_SCALE_TILE
            || format_version != FormatVersion(1)
            || rows == 0
            || cols == 0
            || !cols.is_multiple_of(augur_kernel_api::FP8_E4M3_SCALE_TILE)
        {
            return Err(LoweringError::Core(CompileError::BackendUnsupported));
        }
        let selected = select_cpu_kernel(
            cpu_kernel_registrations(),
            CpuKernelRequest {
                key: KernelKey::fp8_scale_grid_f32_cpu_portable(),
                shape: ShapeClass::classify_matmul(1, rows, cols, 1),
                expected_abi: CpuKernelAbi::Fp8ScaleGridF32,
            },
            &self.ctx.capabilities,
        )?;
        let launch = selected.launch.fp8_scale_grid_f32().ok_or_else(|| {
            LoweringError::RegistryInvariant(RegistryInvariantError::UnexpectedLaunchAbi {
                registration: selected.id,
                expected: CpuKernelAbi::Fp8ScaleGridF32,
                actual: selected.launch.abi(),
            })
        })?;
        let binding = self
            .ctx
            .weight_requirements
            .intern(fp8_scale_grid_requirement(
                weight_storage,
                format_version,
                block_size,
                rows,
                cols,
            ))?;
        let input_buffer = self.ctx.buffer_for(input)?;
        let output = node.outputs[0];
        let purpose = if self.ctx.is_output(output) {
            BufferPurpose::Output
        } else {
            BufferPurpose::Intermediate
        };
        let output_buffer = self.ctx.allocate_tensor(output, purpose)?;
        self.ctx
            .steps
            .push(CpuCompiledStep::Fp8ScaleGridF32(CompiledFp8ScaleGridF32 {
                weight: binding,
                input: BufferView::dense_f32(input_buffer, cols),
                output: BufferView::dense_f32(output_buffer, rows),
                rows,
                cols,
                launch,
            }));
        Ok(())
    }

    fn lower_fused_mlp(&mut self, node: &Node, attrs: &FusedMlpAttrs) -> Result<(), LoweringError> {
        if attrs.activation != FusedMlpActivation::Silu {
            return Err(LoweringError::Core(CompileError::BackendUnsupported));
        }
        let input = node.inputs[0];
        self.ctx.tensor_plain_f32(input)?;
        if !self.ctx.tensor_is_decode_vector(input)? {
            return Err(LoweringError::Core(CompileError::BackendUnsupported));
        }
        let (gate_storage, intermediate, hidden, gate_format, gate_block, gate_version) =
            self.ctx.weight_desc(node.inputs[1])?;
        let (up_storage, up_rows, up_cols, up_format, up_block, up_version) =
            self.ctx.weight_desc(node.inputs[2])?;
        let (down_storage, down_rows, down_cols, down_format, down_block, down_version) =
            self.ctx.weight_desc(node.inputs[3])?;
        let expected_format = QuantFormatId::FP8E4M3ScaleGrid;
        let expected_block = augur_kernel_api::FP8_E4M3_SCALE_TILE;
        if hidden == 0
            || intermediate == 0
            || up_rows != intermediate
            || up_cols != hidden
            || down_rows != hidden
            || down_cols != intermediate
            || [gate_format, up_format, down_format]
                .into_iter()
                .any(|format| format != expected_format)
            || [gate_block, up_block, down_block]
                .into_iter()
                .any(|block| block != expected_block)
            || [gate_version, up_version, down_version]
                .into_iter()
                .any(|version| version != FormatVersion(1))
        {
            return Err(LoweringError::Core(CompileError::BackendUnsupported));
        }
        let config = augur_kernel_api::FusedMlpF32Config {
            gate_min: attrs.gate_clamp.min(),
            gate_max: attrs.gate_clamp.max(),
            up_min: attrs.up_clamp.min(),
            up_max: attrs.up_clamp.max(),
        };
        config
            .validate()
            .map_err(|_| LoweringError::Core(CompileError::InvalidInput))?;
        let selected = select_cpu_kernel(
            cpu_kernel_registrations(),
            CpuKernelRequest {
                key: KernelKey::fp8_scale_grid_fused_mlp_f32_cpu_portable(),
                shape: ShapeClass::classify_matmul(1, hidden, hidden, 1),
                expected_abi: CpuKernelAbi::Fp8FusedMlpF32,
            },
            &self.ctx.capabilities,
        )?;
        let launch = selected.launch.fp8_fused_mlp_f32().ok_or_else(|| {
            LoweringError::RegistryInvariant(RegistryInvariantError::UnexpectedLaunchAbi {
                registration: selected.id,
                expected: CpuKernelAbi::Fp8FusedMlpF32,
                actual: selected.launch.abi(),
            })
        })?;
        let gate_weight = self
            .ctx
            .weight_requirements
            .intern(fp8_scale_grid_requirement(
                gate_storage,
                gate_version,
                gate_block,
                intermediate,
                hidden,
            ))?;
        let up_weight = self
            .ctx
            .weight_requirements
            .intern(fp8_scale_grid_requirement(
                up_storage,
                up_version,
                up_block,
                intermediate,
                hidden,
            ))?;
        let down_weight = self
            .ctx
            .weight_requirements
            .intern(fp8_scale_grid_requirement(
                down_storage,
                down_version,
                down_block,
                hidden,
                intermediate,
            ))?;
        let input_buffer = self.ctx.buffer_for(input)?;
        let output = node.outputs[0];
        let purpose = if self.ctx.is_output(output) {
            BufferPurpose::Output
        } else {
            BufferPurpose::Intermediate
        };
        let output_buffer = self.ctx.allocate_tensor(output, purpose)?;
        self.ctx
            .steps
            .push(CpuCompiledStep::Fp8FusedMlpF32(CompiledFp8FusedMlpF32 {
                gate_weight,
                up_weight,
                down_weight,
                input: BufferView::dense_f32(input_buffer, hidden),
                output: BufferView::dense_f32(output_buffer, hidden),
                hidden,
                intermediate,
                config,
                launch,
            }));
        Ok(())
    }
}

fn graph_epilogue_kind(epilogue: QMatmulEpilogue) -> EpilogueKind {
    match epilogue {
        QMatmulEpilogue::Identity => EpilogueKind::Identity,
        QMatmulEpilogue::Bias => EpilogueKind::Bias,
        QMatmulEpilogue::BiasSilu => EpilogueKind::BiasSilu,
    }
}
