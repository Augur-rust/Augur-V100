//! Original-activation scalar baseline for authenticated new checkpoints.
//! Packed NVFP4/FP8 stays packed in HBM; AUGUR48-row gates use the real NVFP4
//! scalar kernel unless the separate shape+small policy selects SplitK16 F64.
//! Those scalar paths do not pad or requantize weights. A separate phase-only
//! wide adapter may append explicit ZERO rows, retaining all original codes/scales.
use augur_backend::{
    CudaBf16F32MatvecArgsV1, CudaBf16F32MatvecRowsArgsV1, CudaBf16WeightViewV1, CudaF32View, CudaF8E4M3F32MatvecArgsV1,
    CudaF8E4M3RowScaleF32MatvecArgsV1, CudaF8E4M3WeightViewV1, CudaNvfp4Sm70F32MatvecArgsV1,
    CudaNvfp4Sm70WeightViewV1, OwnedCudaBuffer, OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;
use augur_model_qwen35::{
    CheckpointFp8Matrix, CheckpointLinearSpec, CheckpointLinearStorage, CheckpointNvfp4Matrix,
    CheckpointNvfp4Scale, LoadedCheckpointLinear,
};
use std::{error::Error, rc::Rc};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[cfg(feature = "qwen35-augur-http")]
pub(crate) mod startup_bf16;
#[cfg(feature = "qwen35-prefill-cublas")]
pub(crate) mod startup_qpn;
#[cfg(all(feature="qwen35-prefill-cublas",feature="qwen35-augur-dflash-bridge"))]
mod dflash_head;
#[cfg(all(feature="qwen35-prefill-cublas",feature="qwen35-augur-dflash-bridge"))]
pub(crate) use dflash_head::CheckpointSharedBf16HeadRows;

/// Scan every BF16 word using a branch-free reduction, permitting LLVM SIMD.
/// Exact same exponent-bit predicate: reject all NaNs/infinities, keep zeros,
/// subnormals and every finite original weight. No value conversion or copy.
pub(crate) fn bf16_payload_is_finite(bytes: &[u8]) -> bool {
    !bytes.is_empty() && bytes.len().is_multiple_of(2)
        && !bytes.chunks_exact(2).fold(false, |bad, b| {
            bad | (u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80)
        })
}


#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CheckpointProjectionPolicy {
    #[default]
    OriginalF32,
    NvidiaFp8WarpF32,
    QpnFfnF16,
    NvidiaFp8WarpQpnFfnF16,
    NvidiaFp8WarpQpnFfnF16Tail8,
    QpnFfnF16PostScale,
    NvidiaFp8WarpQpnFfnF16PostScale,
    /// Augur: QPN on all eligible large NVFP4 projections, exact48-row gates.
    AugurQpnAllF16PostScale,
    AugurQpnAllResidualF32,
    AugurQpnAllRefinedF64,
    AugurQpnAllRefinedFastF64,
    AugurQpnAllWorkspaceF64,
    AugurQpnAllWorkspaceCtaFoldF64,
    AugurQpnAllWorkspaceShapeF64,
    /// Shape QPN (profile8) plus independent real 48x5120 SplitK16 gates.
    AugurQpnAllWorkspaceShapeSmallF64,
    /// Same per-row SplitK16 math; pair schedules are separate execution IDs.
    AugurQpnAllWorkspaceShapeSmallPairF64,
    AugurQpnAllWorkspaceShapeSmallPairPrefetchF64,
}
impl CheckpointProjectionPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::AugurQpnAllWorkspaceShapeSmallPairF64 => "checkpoint_quasar_qpn_all_workspace_shape_small_pair_f64_v1",
            Self::AugurQpnAllWorkspaceShapeSmallPairPrefetchF64 => "checkpoint_quasar_qpn_all_workspace_shape_small_pair_prefetch_f64_v1",
            Self::AugurQpnAllWorkspaceShapeSmallF64 => "checkpoint_quasar_qpn_all_workspace_shape_small_f64_v1",
            Self::AugurQpnAllWorkspaceShapeF64 => "checkpoint_quasar_qpn_all_workspace_shape_f64_v1",
            Self::AugurQpnAllWorkspaceCtaFoldF64 => "checkpoint_quasar_qpn_all_workspace_cta_fold_f64_v1",
            Self::AugurQpnAllWorkspaceF64 => "checkpoint_quasar_qpn_all_workspace_f64_v2",
            Self::AugurQpnAllRefinedFastF64 => "checkpoint_quasar_qpn_all_refined_hi16_f64_v2",
            Self::AugurQpnAllRefinedF64 => "checkpoint_quasar_qpn_all_refined_f64_v2",
            Self::AugurQpnAllResidualF32 => "checkpoint_quasar_qpn_all_residual_f32_v2",
            Self::AugurQpnAllF16PostScale => "checkpoint_quasar_qpn_all_f16_postscale_v1",
            Self::OriginalF32 => "checkpoint_original_f32_scalar_v1",
            Self::NvidiaFp8WarpF32 => "checkpoint_nvidia_fp8_warp_f32_v1",
            Self::QpnFfnF16 => "checkpoint_qpn_ffn_f16_v1",
            Self::NvidiaFp8WarpQpnFfnF16 => "checkpoint_nvidia_fp8_warp_qpn_ffn_f16_v1",
            Self::NvidiaFp8WarpQpnFfnF16Tail8 => "checkpoint_nvidia_fp8_warp_qpn_ffn_f16_tail8_v1",
            Self::QpnFfnF16PostScale => "checkpoint_qpn_ffn_f16_postscale_v1",
            Self::NvidiaFp8WarpQpnFfnF16PostScale => {
                "checkpoint_nvidia_fp8_warp_qpn_ffn_f16_postscale_v1"
            }
        }
    }
    pub fn uses_fp8_warp(self) -> bool {
        matches!(
            self,
            Self::NvidiaFp8WarpF32
                | Self::NvidiaFp8WarpQpnFfnF16
                | Self::NvidiaFp8WarpQpnFfnF16Tail8
                | Self::NvidiaFp8WarpQpnFfnF16PostScale
        )
    }
    pub fn uses_qpn_projections(self) -> bool {
        matches!(
            self,
            Self::AugurQpnAllF16PostScale
                | Self::AugurQpnAllResidualF32
                | Self::AugurQpnAllRefinedF64
                | Self::AugurQpnAllRefinedFastF64
                | Self::AugurQpnAllWorkspaceF64
                | Self::AugurQpnAllWorkspaceCtaFoldF64 | Self::AugurQpnAllWorkspaceShapeF64 | Self::AugurQpnAllWorkspaceShapeSmallF64 | Self::AugurQpnAllWorkspaceShapeSmallPairF64 | Self::AugurQpnAllWorkspaceShapeSmallPairPrefetchF64
        )
    }
    pub fn qpn_residual(self) -> bool {
        matches!(
            self,
            Self::AugurQpnAllResidualF32
                | Self::AugurQpnAllRefinedF64
                | Self::AugurQpnAllRefinedFastF64
                | Self::AugurQpnAllWorkspaceF64
                | Self::AugurQpnAllWorkspaceCtaFoldF64 | Self::AugurQpnAllWorkspaceShapeF64 | Self::AugurQpnAllWorkspaceShapeSmallF64 | Self::AugurQpnAllWorkspaceShapeSmallPairF64 | Self::AugurQpnAllWorkspaceShapeSmallPairPrefetchF64
        )
    }
    pub fn qpn_workspace(self) -> bool {
        matches!(self, Self::AugurQpnAllWorkspaceF64 | Self::AugurQpnAllWorkspaceCtaFoldF64 | Self::AugurQpnAllWorkspaceShapeF64 | Self::AugurQpnAllWorkspaceShapeSmallF64 | Self::AugurQpnAllWorkspaceShapeSmallPairF64 | Self::AugurQpnAllWorkspaceShapeSmallPairPrefetchF64)
    }
    pub fn qpn_shape_split(self) -> bool {
        matches!(self, Self::AugurQpnAllWorkspaceShapeF64 | Self::AugurQpnAllWorkspaceShapeSmallF64 | Self::AugurQpnAllWorkspaceShapeSmallPairF64 | Self::AugurQpnAllWorkspaceShapeSmallPairPrefetchF64)
    }
    pub fn uses_small_gate_split16(self) -> bool {
        matches!(self, Self::AugurQpnAllWorkspaceShapeSmallF64 | Self::AugurQpnAllWorkspaceShapeSmallPairF64 | Self::AugurQpnAllWorkspaceShapeSmallPairPrefetchF64)
    }
    pub fn small_gate_pair_schedule(self) -> Option<augur_backend::CudaNvfp4SmallGatePairSchedule> {
        use augur_backend::CudaNvfp4SmallGatePairSchedule::*;
        match self {
            Self::AugurQpnAllWorkspaceShapeSmallPairF64 => Some(FixedKRowConcatenated),
            Self::AugurQpnAllWorkspaceShapeSmallPairPrefetchF64 => Some(FixedKSharedInputPrefetch),
            _ => None,
        }
    }
    pub fn admits_small_gate_split16(self, rows: usize, cols: usize) -> bool {
        self.uses_small_gate_split16() && rows == 48 && cols == 5120
    }
    pub fn admits_qpn_projection(self, rows: usize, cols: usize) -> bool {
        self.uses_qpn_projections() && rows > 0 && rows % 32 == 0 && cols > 0 && cols % 64 == 0
    }
    pub fn uses_qpn_ffn(self) -> bool {
        matches!(
            self,
            Self::QpnFfnF16
                | Self::NvidiaFp8WarpQpnFfnF16
                | Self::NvidiaFp8WarpQpnFfnF16Tail8
                | Self::QpnFfnF16PostScale
                | Self::NvidiaFp8WarpQpnFfnF16PostScale
                | Self::AugurQpnAllF16PostScale
                | Self::AugurQpnAllResidualF32
                | Self::AugurQpnAllRefinedF64
                | Self::AugurQpnAllRefinedFastF64
                | Self::AugurQpnAllWorkspaceF64
                | Self::AugurQpnAllWorkspaceCtaFoldF64 | Self::AugurQpnAllWorkspaceShapeF64 | Self::AugurQpnAllWorkspaceShapeSmallF64 | Self::AugurQpnAllWorkspaceShapeSmallPairF64 | Self::AugurQpnAllWorkspaceShapeSmallPairPrefetchF64
        )
    }
    pub fn qpn_ffn_layer(self, layer: usize) -> bool {
        layer < 64
            && self.uses_qpn_ffn()
            && (self != Self::NvidiaFp8WarpQpnFfnF16Tail8 || layer < 56)
    }
    pub fn qpn_postscale(self) -> bool {
        matches!(
            self,
            Self::QpnFfnF16PostScale
                | Self::NvidiaFp8WarpQpnFfnF16PostScale
                | Self::AugurQpnAllF16PostScale
                | Self::AugurQpnAllResidualF32
                | Self::AugurQpnAllRefinedF64
                | Self::AugurQpnAllRefinedFastF64
                | Self::AugurQpnAllWorkspaceF64
                | Self::AugurQpnAllWorkspaceCtaFoldF64 | Self::AugurQpnAllWorkspaceShapeF64 | Self::AugurQpnAllWorkspaceShapeSmallF64 | Self::AugurQpnAllWorkspaceShapeSmallPairF64 | Self::AugurQpnAllWorkspaceShapeSmallPairPrefetchF64
        )
    }
}
enum Storage {
    #[cfg(feature = "qwen35-prefill-cublas")]
    Qpn {
        weight: Rc<augur_backend::cuda_qwen35_qpn::OwnedCudaQwen35QpnWeightV1>,
        plan: Rc<augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35PrefillCublasPlanV1>,
        source_scale: CheckpointNvfp4Scale,
    },
    Nvfp4 {
        weight: OwnedCudaBuffer,
        multiplier: f32,
        source_scale: CheckpointNvfp4Scale,
    },
    Fp8 {
        weight: OwnedCudaBuffer,
        row_scales: OwnedCudaBuffer,
        input_scale: f32,
    },
    Fp8Warp {
        weight: OwnedCudaBuffer,
        block_scales: OwnedCudaBuffer,
        input_scale: f32,
    },
    Bf16(OwnedCudaBuffer),
}
pub struct CheckpointScalarCudaWeight {
    storage: Storage,
    spec: CheckpointLinearSpec,
    stream: Rc<OwnedCudaStream>,
    policy: CheckpointProjectionPolicy,
}
enum Operation {
    #[cfg(feature = "qwen35-prefill-cublas")]
    Qpn(augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnRetainedProjectionV1),
    Nvfp4(CudaNvfp4Sm70F32MatvecArgsV1),
    Fp8(CudaF8E4M3RowScaleF32MatvecArgsV1),
    Fp8Warp(CudaF8E4M3F32MatvecArgsV1),
    Bf16(CudaBf16F32MatvecArgsV1),
}
#[cfg(feature = "qwen35-prefill-cublas")]
impl CheckpointScalarCudaProjection {
    pub(crate) fn faithful_augur_qpn(&self) -> Result<(Rc<augur_backend::cuda_qwen35_qpn::OwnedCudaQwen35QpnWeightV1>, f64)> {
        match &self.weight.storage {
            Storage::Qpn { weight, source_scale: CheckpointNvfp4Scale::CompressedTensors { weight_global_scale, .. }, .. } =>
                Ok((Rc::clone(weight), *weight_global_scale as f64)),
            _ => Err("source-precision KV requires QUASAR QPN and original global divisor".into()),
        }
    }
    pub(crate) fn retained_qpn(&self) -> Result<augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnRetainedProjectionV1> {
        match &self.operation {
            Operation::Qpn(program) => Ok(program.clone()),
            _ => Err("fused input norm requires a QPN consumer".into()),
        }
    }
}
pub struct CheckpointScalarCudaProjection {
    operation: Operation,
    weight: Rc<CheckpointScalarCudaWeight>,
    _input: Rc<OwnedCudaBuffer>,
    _output: Rc<OwnedCudaBuffer>,
}
fn upload(stream: &OwnedCudaStream, bytes: &[u8]) -> Result<OwnedCudaBuffer> {
    let out = OwnedCudaBuffer::allocate(stream.device_ordinal(), bytes.len())?;
    out.copy_from_host_sync(stream, bytes)?;
    Ok(out)
}
fn f32_view(buffer: &OwnedCudaBuffer, offset: usize, count: usize) -> Result<CudaF32View> {
    Ok(CudaF32View::new(
        buffer.view(
            offset,
            count.checked_mul(4).ok_or("projection extent overflow")?,
        )?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        count,
    )?)
}
impl CheckpointScalarCudaWeight {
    #[cfg(feature = "qwen35-augur-http")]
    pub(crate) fn load_streamed_bf16(
        reader: &mut augur_model_qwen35::CheckpointArtifact,
        stream: Rc<OwnedCudaStream>,
        name: &str,
    ) -> Result<(Rc<Self>, serde_json::Value)> {
        let spec = reader.linears().find(|spec| spec.name == name)
            .ok_or("missing streamed BF16 linear")?.clone();
        if spec.storage != CheckpointLinearStorage::Bf16 {
            return Err("streamed linear requires original BF16 storage".into());
        }
        let (weight, detail) = startup_bf16::upload_authenticated(
            reader, &format!("{name}.weight"), &stream,
        )?;
        Ok((Rc::new(Self {
            storage: Storage::Bf16(weight), spec, stream,
            policy: CheckpointProjectionPolicy::OriginalF32,
        }), detail))
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    pub(crate) fn upload_with_shared_qpn(
        stream: Rc<OwnedCudaStream>,
        source: LoadedCheckpointLinear,
        policy: CheckpointProjectionPolicy,
        plan: Rc<augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35PrefillCublasPlanV1>,
    ) -> Result<Rc<Self>> {
        let spec = source.spec().clone();
        if !policy.admits_qpn_projection(spec.rows, spec.cols)
            || spec.storage != CheckpointLinearStorage::Nvfp4
        {
            return Self::upload_with_policy(stream, source, policy);
        }
        let LoadedCheckpointLinear::Nvfp4 {
            packed,
            scales,
            global,
            ..
        } = source
        else {
            return Err("QPN projection storage mismatch".into());
        };
        let weight = Rc::new(startup_qpn::upload(
            &stream, &packed, &scales, spec.rows, spec.cols, global,
        )?);
        Ok(Rc::new(Self {
            storage: Storage::Qpn {
                weight,
                plan,
                source_scale: global,
            },
            spec,
            stream,
            policy,
        }))
    }

    pub fn upload(stream: Rc<OwnedCudaStream>, source: LoadedCheckpointLinear) -> Result<Rc<Self>> {
        Self::upload_with_policy(stream, source, CheckpointProjectionPolicy::OriginalF32)
    }
    pub fn upload_with_policy(
        stream: Rc<OwnedCudaStream>,
        source: LoadedCheckpointLinear,
        policy: CheckpointProjectionPolicy,
    ) -> Result<Rc<Self>> {
        let spec = source.spec().clone();
        if policy.admits_qpn_projection(spec.rows, spec.cols)
            && spec.storage == CheckpointLinearStorage::Nvfp4
        {
            return Err(
                "expanded QPN projection requires its explicit shared-plan constructor".into(),
            );
        }
        let storage = match source {
            LoadedCheckpointLinear::Nvfp4 {
                packed,
                scales,
                global,
                ..
            } => {
                if spec.storage != CheckpointLinearStorage::Nvfp4 {
                    return Err("NVFP4 storage tag mismatch".into());
                }
                let matrix =
                    CheckpointNvfp4Matrix::new(spec.rows, spec.cols, &packed, &scales, global)?;
                let mut wire = vec![0; spec.rows * (spec.cols / 64) * 36];
                matrix.to_ggml_planes_into(&mut wire)?;
                let multiplier = match global {
                    CheckpointNvfp4Scale::CompressedTensors {
                        weight_global_scale,
                        ..
                    } => 1.0 / weight_global_scale,
                    CheckpointNvfp4Scale::ModelOpt { weight_scale_2, .. } => weight_scale_2,
                };
                if !multiplier.is_finite() || multiplier <= 0.0 {
                    return Err("NVFP4 execution scalar out of F32 range".into());
                }
                Storage::Nvfp4 {
                    weight: upload(&stream, &wire)?,
                    multiplier,
                    source_scale: global,
                }
            }
            LoadedCheckpointLinear::Fp8 {
                weights,
                weight_scale,
                input_scale,
                ..
            } => {
                if spec.storage != CheckpointLinearStorage::Fp8 {
                    return Err("FP8 storage tag mismatch".into());
                }
                CheckpointFp8Matrix::new(
                    spec.rows,
                    spec.cols,
                    &weights,
                    weight_scale,
                    input_scale,
                )?;
                let warp = policy.uses_fp8_warp();
                if warp && (spec.rows % 128 != 0 || spec.cols % 128 != 0) {
                    return Err("FP8 warp policy requires complete128x128 scale tiles".into());
                }
                let scale_count = if warp {
                    (spec.rows / 128) * (spec.cols / 128)
                } else {
                    spec.rows
                };
                // NVIDIA has one scalar for the whole matrix. Repeating that
                // exact F32 value in every tile changes neither FP8 codes nor
                // represented weights; never reinterpret row-scale bytes.
                let scales: Vec<u8> = std::iter::repeat_n(weight_scale, scale_count)
                    .flat_map(f32::to_le_bytes)
                    .collect();
                if warp {
                    Storage::Fp8Warp {
                        weight: upload(&stream, &weights)?,
                        block_scales: upload(&stream, &scales)?,
                        input_scale,
                    }
                } else {
                    Storage::Fp8 {
                        weight: upload(&stream, &weights)?,
                        row_scales: upload(&stream, &scales)?,
                        input_scale,
                    }
                }
            }
            LoadedCheckpointLinear::Bf16 { weights, .. } => {
                let expected = spec
                    .rows
                    .checked_mul(spec.cols)
                    .and_then(|v| v.checked_mul(2))
                    .ok_or("BF16 extent overflow")?;
                if spec.storage != CheckpointLinearStorage::Bf16
                    || expected == 0
                    || weights.len() != expected
                    || !bf16_payload_is_finite(&weights)
                {
                    return Err("BF16 shape/value/storage mismatch".into());
                }
                Storage::Bf16(upload(&stream, &weights)?)
            }
        };
        Ok(Rc::new(Self {
            storage,
            spec,
            stream,
            policy,
        }))
    }
    pub fn spec(&self) -> &CheckpointLinearSpec {
        &self.spec
    }
    pub fn policy(&self) -> serde_json::Value {
        let scales = match &self.storage {
            #[cfg(feature = "qwen35-prefill-cublas")]
            Storage::Qpn {
                source_scale,
                weight,
                ..
            } => serde_json::json!({
                "source_scale":format!("{source_scale:?}"),"execution_multiplier":weight.tensor_scale(),
                "operands":if self.policy.qpn_residual() {"F16_hi_plus_scaled_residual"} else {"F16"},
                "global_scale_after_f32_accumulation":true}),
            Storage::Nvfp4 {
                source_scale,
                multiplier,
                ..
            } => {
                serde_json::json!({"source_scale":format!("{source_scale:?}"),"execution_multiplier":multiplier})
            }
            Storage::Fp8 { input_scale, .. } | Storage::Fp8Warp { input_scale, .. } => {
                serde_json::json!({"input_scale_retained_not_applied":input_scale})
            }
            Storage::Bf16(_) => serde_json::Value::Null,
        };
        serde_json::json!({"profile":self.policy.label(),"storage":self.spec.storage,
            "weight_requantized":false,"activation_quantized":false,"native_w4a4":false,"scales":scales,
            "small_gate_split16": self.policy.admits_small_gate_split16(self.spec.rows,self.spec.cols)})
    }
    pub fn bind(
        self: &Rc<Self>,
        input: Rc<OwnedCudaBuffer>,
        input_offset: usize,
        output: Rc<OwnedCudaBuffer>,
        output_offset: usize,
    ) -> Result<CheckpointScalarCudaProjection> {
        let x = f32_view(&input, input_offset, self.spec.cols)?;
        let y = f32_view(&output, output_offset, self.spec.rows)?;
        let input_end = input_offset
            .checked_add(self.spec.cols * 4)
            .ok_or("input extent overflow")?;
        let output_end = output_offset
            .checked_add(self.spec.rows * 4)
            .ok_or("output extent overflow")?;
        if Rc::ptr_eq(&input, &output) && input_offset < output_end && output_offset < input_end {
            return Err("checkpoint projection input/output alias".into());
        }
        let (n, k) = (self.spec.rows, self.spec.cols);
        let operation = match &self.storage {
            #[cfg(feature = "qwen35-prefill-cublas")]
            Storage::Qpn { weight, plan, .. } => Operation::Qpn(
                augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnRetainedProjectionV1::bind(
                    Rc::clone(plan), Rc::clone(&self.stream), Rc::clone(weight),
                    Rc::clone(&input), input_offset, Rc::clone(&output), output_offset, 1)?
                .with_scale_placement(if self.policy.qpn_shape_split() {
                    augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnScalePlacement::ResidualWorkspaceShapeF64
                } else if self.policy == CheckpointProjectionPolicy::AugurQpnAllWorkspaceCtaFoldF64 {
                    augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnScalePlacement::ResidualWorkspaceCtaFoldF64
                } else if self.policy == CheckpointProjectionPolicy::AugurQpnAllWorkspaceF64 {
                    augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnScalePlacement::ResidualWorkspaceF64
                } else if self.policy == CheckpointProjectionPolicy::AugurQpnAllRefinedFastF64 {
                    augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnScalePlacement::ResidualHiChunk16F64
                } else if self.policy == CheckpointProjectionPolicy::AugurQpnAllRefinedF64 {
                    augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnScalePlacement::ResidualChunkedF64
                } else if self.policy.qpn_residual() {
                    augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnScalePlacement::ResidualF32AfterAccumulation
                } else { augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnScalePlacement::AfterF32Accumulation })?),
            Storage::Nvfp4 {
                weight, multiplier, ..
            } => Operation::Nvfp4(CudaNvfp4Sm70F32MatvecArgsV1::bind(
                &self.stream,
                CudaNvfp4Sm70WeightViewV1::new(
                    weight.view(0, weight.len_bytes())?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,
                    n,
                    k,
                )?,
                x,
                y,
                *multiplier,
            )?),
            Storage::Fp8 {
                weight, row_scales, ..
            } => Operation::Fp8(CudaF8E4M3RowScaleF32MatvecArgsV1::bind(
                &self.stream,
                CudaF8E4M3WeightViewV1::new(
                    weight.view(0, weight.len_bytes())?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F8E4M3_CUDA_V1,
                    n,
                    k,
                )?,
                f32_view(row_scales, 0, n)?,
                x,
                y,
            )?),
            Storage::Bf16(weight) => Operation::Bf16(CudaBf16F32MatvecArgsV1::bind(
                &self.stream,
                CudaBf16WeightViewV1::new(
                    weight.view(0, weight.len_bytes())?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
                    n,
                    k,
                )?,
                x,
                y,
            )?),
            Storage::Fp8Warp {
                weight,
                block_scales,
                ..
            } => Operation::Fp8Warp(CudaF8E4M3F32MatvecArgsV1::bind(
                &self.stream,
                CudaF8E4M3WeightViewV1::new(
                    weight.view(0, weight.len_bytes())?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F8E4M3_CUDA_V1,
                    n,
                    k,
                )?,
                f32_view(block_scales, 0, (n / 128) * (k / 128))?,
                x,
                y,
            )?),
        };
        Ok(CheckpointScalarCudaProjection {
            operation,
            weight: Rc::clone(self),
            _input: input,
            _output: output,
        })
    }
}
/// Retains both authenticated weights, their common input, both outputs and
/// the exact stream owner. No lifetime depends on the source binding locals.
pub(crate) struct CheckpointSmallGatePair {
    op: augur_backend::CudaNvfp4Sm70SmallGatePairArgsV1,
    left: CheckpointScalarCudaProjection,
    _right: CheckpointScalarCudaProjection,
}
impl CheckpointSmallGatePair {
    /// Phase-only exact byte repack for the real48-row gates. Existing scalar
    /// weights remain unchanged; temporary64-row QPN weights append ZERO rows.
    /// Synchronous D2H/repack/upload is setup cost, never a hot submit operation.
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub(crate) fn wide_phase_source(&self) -> Result<CheckpointWideSmallGateSource> {
        use augur_backend::cuda_qwen35_qpn::OwnedCudaQwen35QpnWeightV1;
        let pack = |p: &CheckpointScalarCudaProjection| -> Result<Rc<OwnedCudaQwen35QpnWeightV1>> {
            let w = &p.weight;
            if w.spec.rows != 48 || w.spec.cols != 5120 {
                return Err("wide gate requires exact authenticated48x5120 source".into());
            }
            let Storage::Nvfp4 { weight, multiplier, .. } = &w.storage else {
                return Err("wide small gate requires original NVFP4".into());
            };
            let source_bytes = 48 * (5120 / 64) * 36;
            if weight.len_bytes() != source_bytes { return Err("small source extent".into()); }
            let mut data = vec![0; source_bytes];
            weight.copy_to_host_sync(&w.stream, &mut data)?;
            // Zero scale and zero code represent an exact zero row. No row is
            // copied/repeated, no scale is dropped, no weight is requantized.
            data.resize(64 * (5120 / 64) * 36, 0);
            Ok(Rc::new(OwnedCudaQwen35QpnWeightV1::upload_ggml(&w.stream, &data, 64, 5120, *multiplier)?))
        };
        Ok(CheckpointWideSmallGateSource {
            padded: [pack(&self.left)?, pack(&self._right)?],
            originals: [Rc::clone(&self.left.weight), Rc::clone(&self._right.weight)],
        })
    }
    pub(crate) fn submit_for_plan(&self) -> Result<()> {
        self.op.submit_for_plan(&self.left.weight.stream)?;
        Ok(())
    }
}
/// Lifetime is deliberately owned by a prefill phase, not the model loader.
#[cfg(feature = "qwen35-prefill-cublas")]
pub(crate) struct CheckpointWideSmallGateSource {
    pub(crate) padded: [Rc<augur_backend::cuda_qwen35_qpn::OwnedCudaQwen35QpnWeightV1>; 2],
    originals: [Rc<CheckpointScalarCudaWeight>; 2],
}
#[cfg(feature = "qwen35-prefill-cublas")]
impl CheckpointWideSmallGateSource {

    #[cfg(test)]
    pub(crate) fn diagnostic_identity(&self) -> Result<serde_json::Value> {
        use sha2::{Digest,Sha256};
        let mut rows=Vec::new();
        for (p,original) in self.padded.iter().zip(&self.originals) {
            let Storage::Nvfp4{weight,multiplier,..}=&original.storage else{return Err("gate source changed".into());};
            let mut raw=vec![0;weight.len_bytes()];weight.copy_to_host_sync(&original.stream,&mut raw)?;
            let owner=p.shared_storage();let mut padded=vec![0;owner.len_bytes()];owner.copy_to_host_sync(&original.stream,&mut padded)?;
            rows.push(serde_json::json!({"name":original.spec.name,"logical_rows":48,"physical_rows":64,"columns":5120,
                "original_ggml_sha256":format!("{:x}",Sha256::digest(&raw)),"padded_qpn_sha256":format!("{:x}",Sha256::digest(&padded)),
                "tensor_scale_f32":multiplier,"padded_physical_bytes":padded.len()}));
        }
        Ok(serde_json::json!(rows))
    }
    pub(crate) const PADDED_BYTES: usize = 2 * 64 * (5120 / 64) * 36;
    pub(crate) fn same_owner(&self, pair: &CheckpointSmallGatePair) -> bool {
        Rc::ptr_eq(&self.originals[0], &pair.left.weight) && Rc::ptr_eq(&self.originals[1], &pair._right.weight)
    }
}

impl CheckpointScalarCudaProjection {

    /// Rebind only the retained authentic BF16 vocabulary head, on the exact
    /// same stream owner. No upload, quantization or additional normalization.
    #[cfg(feature = "qwen35-augur-dflash-bridge")]
    pub(crate) fn rebind_shared_bf16_head(
        &self, stream: &Rc<OwnedCudaStream>, input: Rc<OwnedCudaBuffer>,
        input_offset: usize, output: Rc<OwnedCudaBuffer>, output_offset: usize,
    ) -> Result<Self> {
        if !Rc::ptr_eq(stream, &self.weight.stream)
            || !matches!(self.weight.storage, Storage::Bf16(_))
            || self.weight.spec.rows != 248320 || self.weight.spec.cols != 5120 {
            return Err("shared draft head requires the same retained BF16 vocabulary owner".into());
        }
        self.weight.bind(input, input_offset, output, output_offset)
    }
    pub(crate) fn into_small_gate_pair(self, right: Self) -> Result<CheckpointSmallGatePair> {
        let policy = self.weight.policy;
        let schedule = policy.small_gate_pair_schedule().ok_or("small-gate pair policy required")?;
        let left_name = self.weight.spec.name.strip_suffix(".in_proj_a");
        let right_name = right.weight.spec.name.strip_suffix(".in_proj_b");
        if policy != right.weight.policy || left_name.is_none() || left_name != right_name
            || !policy.admits_small_gate_split16(self.weight.spec.rows, self.weight.spec.cols)
            || !policy.admits_small_gate_split16(right.weight.spec.rows, right.weight.spec.cols)
            || !Rc::ptr_eq(&self.weight.stream, &right.weight.stream)
            || !Rc::ptr_eq(&self._input, &right._input) {
            return Err("small-gate pair requires matching a/b layer, policy, input and stream owners".into());
        }
        let (Operation::Nvfp4(a),Operation::Nvfp4(b)) = (&self.operation,&right.operation) else {
            return Err("small-gate pair requires retained original NVFP4 storage".into());
        };
        let op = augur_backend::CudaNvfp4Sm70SmallGatePairArgsV1::bind(
            &self.weight.stream, *a, *b, schedule,
        )?;
        Ok(CheckpointSmallGatePair {op,left:self,_right:right})
    }
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub(crate) fn rebind_exact_batch(
        &self,
        stream: &Rc<OwnedCudaStream>,
        input: Rc<OwnedCudaBuffer>,
        output: Rc<OwnedCudaBuffer>,
        batch: usize,
    ) -> Result<CheckpointExactBatchProjection> {
        if !Rc::ptr_eq(stream, &self.weight.stream) || !(1..=8).contains(&batch) {
            return Err("exact batch requires same stream owner and real1..8 rows".into());
        }
        let Storage::Nvfp4 {
            weight, multiplier, ..
        } = &self.weight.storage
        else {
            return Err("exact batch only supports retained original NVFP4".into());
        };
        let (n, k) = (self.weight.spec.rows, self.weight.spec.cols);
        if self.weight.policy.admits_small_gate_split16(n, k) {
            return Err("SplitK16 small gates cannot rebind as original exact-batch math".into());
        }
        let op = augur_backend::CudaNvfp4Sm70F32BatchArgsV1::bind(
            stream,
            CudaNvfp4Sm70WeightViewV1::new(
                weight.view(0, weight.len_bytes())?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,
                n,
                k,
            )?,
            f32_view(&input, 0, batch * k)?,
            f32_view(&output, 0, batch * n)?,
            *multiplier,
            batch,
        )?;
        Ok(CheckpointExactBatchProjection {
            op,
            weight: Rc::clone(&self.weight),
            _input: input,
            _output: output,
        })
    }
    pub fn submit_for_plan(&self) -> Result<()> {
        let stream = &self.weight.stream;
        match &self.operation {
            #[cfg(feature = "qwen35-prefill-cublas")]
            Operation::Qpn(op) => op.submit_for_plan()?,
            Operation::Nvfp4(op) => {
                if self.weight.policy.admits_small_gate_split16(self.weight.spec.rows, self.weight.spec.cols) {
                    op.submit_small_gate_split16_for_plan(stream)?;
                } else {
                    op.submit_for_plan(stream)?;
                }
            },
            Operation::Fp8(op) => op.submit_for_plan(stream)?,
            Operation::Fp8Warp(op) => op.submit_warp_fp32_for_plan(stream)?,
            Operation::Bf16(op) => op.submit_for_plan(stream)?,
        }
        Ok(())
    }
}

#[cfg(feature = "qwen35-prefill-cublas")]
pub(crate) struct CheckpointExactBatchProjection {
    op: augur_backend::CudaNvfp4Sm70F32BatchArgsV1,
    weight: Rc<CheckpointScalarCudaWeight>,
    _input: Rc<OwnedCudaBuffer>,
    _output: Rc<OwnedCudaBuffer>,
}
#[cfg(feature = "qwen35-prefill-cublas")]
impl CheckpointExactBatchProjection {
    pub(crate) fn submit(&self) -> Result<()> {
        self.op.submit_for_plan(&self.weight.stream)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use augur_backend::cuda_device_allocation_snapshot;
    use augur_model_qwen35::{
        CheckpointArtifact, CheckpointExpectedIdentity, Qwen35CheckpointProfile,
    };

    #[test]
    #[ignore = "exclusive CUDA; paired checkpoint owner retention and rejection"]
    fn small_gate_pair_retains_both_weights_input_outputs_and_stream() {
        let baseline=cuda_device_allocation_snapshot();
        for policy in [CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64,
                       CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairPrefetchF64] {
            let stream=Rc::new(OwnedCudaStream::create(0).unwrap());
            let foreign=Rc::new(OwnedCudaStream::create(0).unwrap());
            let make=|stream:Rc<OwnedCudaStream>,name:&str,seed:usize,policy| {
                CheckpointScalarCudaWeight::upload_with_policy(stream,LoadedCheckpointLinear::Nvfp4 {
                    spec:CheckpointLinearSpec{name:name.into(),rows:48,cols:5120,storage:CheckpointLinearStorage::Nvfp4},
                    packed:(0..48*2560).map(|i|(i*19+seed) as u8).collect(),
                    scales:vec![0x38;48*320],
                    global:CheckpointNvfp4Scale::CompressedTensors{weight_global_scale:128.,input_global_scale:1.},
                },policy).unwrap()
            };
            let an="model.language_model.layers.0.linear_attn.in_proj_a";
            let bn="model.language_model.layers.0.linear_attn.in_proj_b";
            let wa=make(Rc::clone(&stream),an,3,policy);
            let wb=make(Rc::clone(&stream),bn,47,policy);
            let x=Rc::new(OwnedCudaBuffer::allocate(0,5120*4).unwrap());
            let other_x=Rc::new(OwnedCudaBuffer::allocate(0,5120*4).unwrap());
            let ya=Rc::new(OwnedCudaBuffer::allocate(0,48*4).unwrap());
            let yb=Rc::new(OwnedCudaBuffer::allocate(0,48*4).unwrap());
            let ra=Rc::new(OwnedCudaBuffer::allocate(0,48*4).unwrap());
            let rb=Rc::new(OwnedCudaBuffer::allocate(0,48*4).unwrap());
            let bind=|w:&Rc<CheckpointScalarCudaWeight>,x:&Rc<OwnedCudaBuffer>,y:&Rc<OwnedCudaBuffer>|
                w.bind(Rc::clone(x),0,Rc::clone(y),0).unwrap();
            assert!(bind(&wb,&x,&yb).into_small_gate_pair(bind(&wa,&x,&ya)).is_err());
            assert!(bind(&wa,&other_x,&ya).into_small_gate_pair(bind(&wb,&x,&yb)).is_err());
            for wrong in [
                make(Rc::clone(&stream),"model.language_model.layers.1.linear_attn.in_proj_b",47,policy),
                make(Rc::clone(&foreign),bn,47,policy),
                make(Rc::clone(&stream),bn,47,CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallF64),
            ] {
                assert!(bind(&wa,&x,&ya).into_small_gate_pair(bind(&wrong,&x,&yb)).is_err());
            }
            let pair=bind(&wa,&x,&ya).into_small_gate_pair(bind(&wb,&x,&yb)).unwrap();
            let ar=bind(&wa,&x,&ra);let br=bind(&wb,&x,&rb);
            let before=cuda_device_allocation_snapshot();
            let mut expected_a=vec![0u8;48*4];let mut expected_b=expected_a.clone();
            for phase in 0..3 {
                let values:Vec<u8>=(0..5120).flat_map(|i|((((i*17+phase*7)%101) as f32-50.)/23.).to_le_bytes()).collect();
                x.copy_from_host_sync(&stream,&values).unwrap();
                ar.submit_for_plan().unwrap();br.submit_for_plan().unwrap();pair.submit_for_plan().unwrap();
                ra.copy_to_host_sync(&stream,&mut expected_a).unwrap();rb.copy_to_host_sync(&stream,&mut expected_b).unwrap();
                for (actual,expected) in [(&ya,&expected_a),(&yb,&expected_b)] {
                    let mut bytes=vec![0u8;48*4];actual.copy_to_host_sync(&stream,&mut bytes).unwrap();assert_eq!(&bytes,expected);
                }
            }
            assert_eq!(before.successful_allocations,cuda_device_allocation_snapshot().successful_allocations);
            let weak_a=Rc::downgrade(&wa);let weak_b=Rc::downgrade(&wb);
            let weak_x=Rc::downgrade(&x);let weak_ya=Rc::downgrade(&ya);let weak_yb=Rc::downgrade(&yb);
            let weak_stream=Rc::downgrade(&stream);
            drop(ar);drop(br);drop(wa);drop(wb);drop(x);drop(ya);drop(yb);drop(stream);
            assert!(weak_a.upgrade().is_some()&&weak_b.upgrade().is_some()&&weak_x.upgrade().is_some());
            let retained_stream=weak_stream.upgrade().unwrap();
            weak_ya.upgrade().unwrap().copy_from_host_sync(&retained_stream,&vec![0u8;48*4]).unwrap();
            weak_yb.upgrade().unwrap().copy_from_host_sync(&retained_stream,&vec![0u8;48*4]).unwrap();
            pair.submit_for_plan().unwrap();
            for (weak,expected) in [(&weak_ya,&expected_a),(&weak_yb,&expected_b)] {
                let mut bytes=vec![0u8;48*4];weak.upgrade().unwrap().copy_to_host_sync(&retained_stream,&mut bytes).unwrap();assert_eq!(&bytes,expected);
            }
            drop(pair);drop(retained_stream);
            assert!(weak_a.upgrade().is_none()&&weak_b.upgrade().is_none()&&weak_x.upgrade().is_none());
            assert!(weak_ya.upgrade().is_none()&&weak_yb.upgrade().is_none()&&weak_stream.upgrade().is_none());
        }
        let after=cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes,baseline.live_bytes);assert_eq!(after.live_allocations,baseline.live_allocations);
        assert_eq!(after.successful_allocations-baseline.successful_allocations,after.successful_frees-baseline.successful_frees);
    }

    #[test]
    fn small_gate_pair_policies_keep_qpn_and_small_row_math() {
        let old=CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallF64;
        for policy in [CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64,
                       CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairPrefetchF64] {
            assert!(policy.qpn_shape_split()&&policy.qpn_workspace()&&policy.qpn_residual());
            assert!(policy.uses_qpn_ffn()&&policy.qpn_postscale());
            assert!(policy.uses_small_gate_split16()&&policy.small_gate_pair_schedule().is_some());
            assert!(policy.admits_small_gate_split16(48,5120));
            assert!(!policy.admits_qpn_projection(48,5120));
            assert_ne!(policy.label(),old.label());
        }
        assert!(old.small_gate_pair_schedule().is_none());
        assert!(CheckpointProjectionPolicy::default().small_gate_pair_schedule().is_none());
    }

    #[test]
    fn shape_small_policy_has_separate_identity_and_exact_geometry() {
        let candidate = CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallF64;
        let shape = CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeF64;
        assert_ne!(candidate.label(), shape.label());
        assert!(candidate.qpn_shape_split() && candidate.qpn_workspace() && candidate.qpn_residual());
        assert!(candidate.uses_qpn_ffn() && candidate.qpn_postscale());
        assert!(candidate.admits_small_gate_split16(48,5120));
        for (n,k) in [(32,5120),(64,5120),(48,5056),(0,5120),(48,0)] {
            assert!(!candidate.admits_small_gate_split16(n,k));
        }
        assert!(!candidate.admits_qpn_projection(48,5120));
        assert!(candidate.admits_qpn_projection(1024,5120));
        assert!(!shape.uses_small_gate_split16());
        assert!(!CheckpointProjectionPolicy::default().uses_small_gate_split16());
    }

    #[test]
    #[ignore = "exclusive SM70 and audited checkpoints; AUGUR_CHECKPOINT_CUDA_OUTPUT"]
    fn real_new_checkpoint_projection_reference_and_cleanup() {
        let root = std::path::PathBuf::from(
            std::env::var_os("AUGUR_CHECKPOINT_CUDA_OUTPUT").expect("output"),
        );
        std::fs::create_dir(&root).unwrap();
        let baseline = cuda_device_allocation_snapshot();
        let mut results = Vec::new();
        for (profile, label, audit) in [
            (
                Qwen35CheckpointProfile::QuasarNvfp4,
                "quasar",
                "F:/qwen35-quasar-input-audit-20261008-v2/checkpoint.json",
            ),
            (
                Qwen35CheckpointProfile::NvidiaMixed,
                "nvidia",
                "F:/qwen35-nvidia-input-audit-20261008-v1/checkpoint.json",
            ),
        ] {
            let a: serde_json::Value =
                serde_json::from_slice(&std::fs::read(audit).unwrap()).unwrap();
            let expected = CheckpointExpectedIdentity {
                config_sha256: a["metadata"]["config.json"]["sha256"]
                    .as_str()
                    .unwrap()
                    .into(),
                index_sha256: a["metadata"]["model.safetensors.index.json"]["sha256"]
                    .as_str()
                    .unwrap()
                    .into(),
                shards: a["shards"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| {
                        (
                            s["name"].as_str().unwrap().into(),
                            s["sha256"].as_str().unwrap().into(),
                        )
                    })
                    .collect(),
            };
            let mut artifact =
                CheckpointArtifact::open(a["model_dir"].as_str().unwrap(), profile, expected)
                    .unwrap();
            let names = [
                "model.language_model.layers.0.mlp.gate_proj",
                "model.language_model.layers.0.mlp.down_proj",
                "model.language_model.layers.50.mlp.gate_proj",
                "model.language_model.layers.63.mlp.down_proj",
                "model.language_model.layers.0.linear_attn.in_proj_b",
                "model.language_model.layers.0.linear_attn.in_proj_qkv",
                "model.language_model.layers.3.self_attn.q_proj",
            ];
            for name in names {
                let source = artifact.load_linear(name).unwrap();
                let spec = source.spec().clone();
                let rows = if spec.rows == 48 {
                    (0..48).collect::<Vec<_>>()
                } else {
                    vec![0, 1, 2, 3, spec.rows / 2, spec.rows - 2, spec.rows - 1]
                };
                let reference: Vec<Vec<f64>> = rows
                    .iter()
                    .map(|&r| {
                        artifact
                            .load_linear_rows(name, r, 1)
                            .unwrap()
                            .decode_f64()
                            .unwrap()
                    })
                    .collect();
                let stream = Rc::new(OwnedCudaStream::create(0).unwrap());
                let weight =
                    CheckpointScalarCudaWeight::upload(Rc::clone(&stream), source).unwrap();
                let policy = weight.policy();
                let input = Rc::new(OwnedCudaBuffer::allocate(0, spec.cols * 4).unwrap());
                let output = Rc::new(OwnedCudaBuffer::allocate(0, spec.rows * 4).unwrap());
                assert!(weight
                    .bind(Rc::clone(&input), 4, Rc::clone(&output), 0)
                    .is_err());
                let projection = weight
                    .bind(Rc::clone(&input), 0, Rc::clone(&output), 0)
                    .unwrap();
                let weak = Rc::downgrade(&weight);
                drop(weight); // the bound program retains its source/stream owners.
                let before = cuda_device_allocation_snapshot();
                for phase in 0..2 {
                    let x: Vec<f32> = (0..spec.cols)
                        .map(|i| (((i * 17 + phase * 11) % 257) as f32 - 128.0) / 128.0)
                        .collect();
                    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
                    input.copy_from_host_sync(&stream, &bytes).unwrap();
                    projection.submit_for_plan().unwrap();
                    let mut result = vec![0; spec.rows * 4];
                    output.copy_to_host_sync(&stream, &mut result).unwrap();
                    assert!(
                        result
                            .chunks_exact(4)
                            .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
                        "nonfinite projection row outside reference sample"
                    );
                    let mut error2 = 0.0;
                    let mut ref2 = 0.0;
                    let mut max_abs = 0.0_f64;
                    for (&r, w) in rows.iter().zip(&reference) {
                        let expected: f64 = w.iter().zip(&x).map(|(a, b)| a * f64::from(*b)).sum();
                        let actual =
                            f32::from_le_bytes(result[r * 4..r * 4 + 4].try_into().unwrap()) as f64;
                        assert!(actual.is_finite());
                        let d = (actual - expected).abs();
                        error2 += d * d;
                        ref2 += expected * expected;
                        max_abs = max_abs.max(d);
                    }
                    let relative_l2 = (error2 / ref2.max(1e-300)).sqrt();
                    assert!(
                        relative_l2 <= 1e-5,
                        "{label} {name} relative_l2={relative_l2}"
                    );
                    results.push(serde_json::json!({"profile":label,"name":name,"rows":spec.rows,"cols":spec.cols,
                        "phase":phase,"checked_rows":rows.len(),"relative_l2":relative_l2,"max_abs":max_abs,"policy":policy}));
                }
                let after = cuda_device_allocation_snapshot();
                assert_eq!(before.successful_allocations, after.successful_allocations);
                stream.synchronize().unwrap();
                drop(projection);
                assert!(weak.upgrade().is_none());
                println!("CHECKPOINT_CUDA_CASE={label} {name} PASS");
            }
        }
        let after = cuda_device_allocation_snapshot();
        assert_eq!(baseline.live_bytes, after.live_bytes);
        assert_eq!(baseline.live_allocations, after.live_allocations);
        assert_eq!(
            after.successful_allocations - baseline.successful_allocations,
            after.successful_frees - baseline.successful_frees
        );
        std::fs::write(root.join("result.json"),serde_json::to_vec_pretty(&serde_json::json!({
            "status":"PASS_NEW_CHECKPOINT_SCALAR_PROJECTION_REFERENCE_AND_OWNER_CLEANUP","cases":results,
            "product_ready":false,"semantic_qualified":false,"native_w4a4":false,
            "live_bytes_after":after.live_bytes,"live_allocations_after":after.live_allocations})).unwrap()).unwrap();
    }

    /// Export authenticated real-shape weights for the standalone kernel cost
    /// audit. Deterministic F32 inputs are diagnostic, not semantic evidence.
    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    #[ignore = "exclusive SM70; AUGUR_QPN_COST_FIXTURE_OUTPUT must be a fresh directory"]
    fn export_augur_qpn_compensation_cost_fixture() {
        use augur_backend::cuda_qwen35_prefill_cublas::{
            CudaQwen35PrefillCublasPlanV1, CudaQwen35QpnRetainedProjectionV1,
            CudaQwen35QpnScalePlacement,
        };
        use sha2::{Digest, Sha256};
        let root = std::path::PathBuf::from(std::env::var_os("AUGUR_QPN_COST_FIXTURE_OUTPUT").unwrap());
        std::fs::create_dir(&root).unwrap();
        let audit = std::env::var("AUGUR_QPN_COST_AUDIT").unwrap_or_else(|_| {
            "F:/qwen35-quasar-input-audit-20261008-v2/checkpoint.json".into()
        });
        let audit_bytes = std::fs::read(&audit).unwrap();
        let a: serde_json::Value = serde_json::from_slice(&audit_bytes).unwrap();
        let expected = CheckpointExpectedIdentity {
            config_sha256: a["metadata"]["config.json"]["sha256"].as_str().unwrap().into(),
            index_sha256: a["metadata"]["model.safetensors.index.json"]["sha256"].as_str().unwrap().into(),
            shards: a["shards"].as_array().unwrap().iter().map(|s| (
                s["name"].as_str().unwrap().into(), s["sha256"].as_str().unwrap().into()
            )).collect(),
        };
        let mut artifact = CheckpointArtifact::open(a["model_dir"].as_str().unwrap(),
            Qwen35CheckpointProfile::QuasarNvfp4, expected).unwrap();
        let baseline = cuda_device_allocation_snapshot();
        let mut cases = Vec::new();
        for (label, name, count) in [
            ("gate", "model.language_model.layers.0.mlp.gate_proj", 128),
            ("down", "model.language_model.layers.0.mlp.down_proj", 64),
            ("qkv", "model.language_model.layers.0.linear_attn.in_proj_qkv", 48),
            ("z", "model.language_model.layers.0.linear_attn.in_proj_z", 48),
            ("gdn_out", "model.language_model.layers.0.linear_attn.out_proj", 48),
            ("q", "model.language_model.layers.3.self_attn.q_proj", 16),
            ("k", "model.language_model.layers.3.self_attn.k_proj", 32),
            ("o", "model.language_model.layers.3.self_attn.o_proj", 16),
        ] {
            let source = artifact.load_linear(name).unwrap();
            let spec = source.spec().clone();
            let (n,k) = (spec.rows,spec.cols);
            let split_k = if n == 17408 && k == 5120 {16} else {8};
            let stream = Rc::new(OwnedCudaStream::create(0).unwrap());
            let plan = Rc::new(CudaQwen35PrefillCublasPlanV1::new(
                &stream,n * 8 * split_k * 8,k * 2,8).unwrap());
            let owner = CheckpointScalarCudaWeight::upload_with_shared_qpn(
                Rc::clone(&stream), source, CheckpointProjectionPolicy::AugurQpnAllWorkspaceF64,
                Rc::clone(&plan)).unwrap();
            let Storage::Qpn {weight,..} = &owner.storage else {panic!("QPN expected")};
            let x: Vec<f32> = (0..k*8).map(|i| {
                let h = ((i as u32).wrapping_mul(1664525).wrapping_add(1013904223)) ^ ((i as u32).rotate_left(13));
                ((h % 2000003) as f32 - 1000001.0) / 313337.0
            }).collect();
            let input = Rc::new(OwnedCudaBuffer::allocate(0,k*8*4).unwrap());
            let output = Rc::new(OwnedCudaBuffer::allocate(0,n*8*4).unwrap());
            let input_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            input.copy_from_host_sync(&stream,&input_bytes).unwrap();
            let program = CudaQwen35QpnRetainedProjectionV1::bind(Rc::clone(&plan),Rc::clone(&stream),
                Rc::clone(weight),Rc::clone(&input),0,Rc::clone(&output),0,8).unwrap()
                .with_scale_placement(CudaQwen35QpnScalePlacement::ResidualWorkspaceF64).unwrap();
            let before = cuda_device_allocation_snapshot();
            program.submit_for_plan().unwrap();
            let mut y = vec![0;n*8*4];
            output.copy_to_host_sync(&stream,&mut y).unwrap();
            assert_eq!(before.successful_allocations,cuda_device_allocation_snapshot().successful_allocations);
            let mut physical = vec![0;weight.len_bytes()];
            weight.shared_storage().copy_to_host_sync(&stream,&mut physical).unwrap();
            let write = |suffix: &str, bytes: &[u8]| {
                let filename=format!("{label}.{suffix}");
                std::fs::write(root.join(&filename),bytes).unwrap();
                serde_json::json!({"file":filename,"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes))})
            };
            let sampled_rows = [0,1,2,3,n/2,n-2,n-1];
            let mut reference=Vec::new();
            for row in sampled_rows {
                let w = artifact.load_linear_rows(name,row,1).unwrap().decode_f64().unwrap();
                reference.push(serde_json::json!({"row":row,"values":(0..8).map(|b| {
                    w.iter().zip(&x[b*k..(b+1)*k]).map(|(a,b)| a*f64::from(*b)).sum::<f64>()
                }).collect::<Vec<_>>()}));
            }
            cases.push(serde_json::json!({"label":label,"name":name,"rows":n,"cols":k,
                "split_k":split_k,"count":count,"tensor_scale":weight.tensor_scale(),
                "weight":write("qpn",&physical),"input":write("f32",&input_bytes),
                "profile6":write("baseline",&y),"f64_samples":reference}));
            eprintln!("QPN_COST_FIXTURE {label} N{n} K{k}");
        }
        let after=cuda_device_allocation_snapshot();
        assert_eq!(baseline.live_bytes,after.live_bytes);
        assert_eq!(baseline.live_allocations,after.live_allocations);
        assert_eq!(after.successful_allocations-baseline.successful_allocations,
            after.successful_frees-baseline.successful_frees);
        std::fs::write(root.join("manifest.json"),serde_json::to_vec_pretty(&serde_json::json!({
            "schema":"qpn_compensation_cost_fixture_v1","scope":"real QUASAR weights, deterministic synthetic F32 activations; NOT semantic/whole-target qualification",
            "audit_path":audit,"audit_sha256":format!("{:x}",Sha256::digest(&audit_bytes)),
            "cases":cases,"live_bytes_after":after.live_bytes,"live_allocations_after":after.live_allocations
        })).unwrap()).unwrap();
    }

}

#[cfg(all(test, feature = "qwen35-prefill-cublas"))]
mod prefill_rebind_tests;

#[cfg(test)]
mod bf16_startup_scan_tests {
    use super::bf16_payload_is_finite;
    #[test]
    fn branch_free_bf16_scan_exhaustive_word_equivalence() {
        for bits in 0..=u16::MAX {
            let expected=f32::from_bits(u32::from(bits)<<16).is_finite();
            assert_eq!(bf16_payload_is_finite(&bits.to_le_bytes()),expected);
            // Exercise every SIMD lane plus a scalar tail with each bit pattern.
            for lane in [0,1,7,8,15,16,31,32,63,64] {
                let mut payload=[0u8;130];payload[lane*2..lane*2+2].copy_from_slice(&bits.to_le_bytes());
                assert_eq!(bf16_payload_is_finite(&payload),expected);
            }
        }
        assert!(!bf16_payload_is_finite(&[]));
        assert!(!bf16_payload_is_finite(&[0]));
        assert!(!bf16_payload_is_finite(&[0,0,0]));
    }
}
