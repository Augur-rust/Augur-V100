#![cfg(feature = "cuda")]

//! Real Qwen safetensors output-head CUDA component.
//!
//! The probe authenticates the Hugging Face `lm_head.weight` and its per-row
//! `weight_scale`, applies the real final norm weight to a supplied hidden
//! vector, and evaluates all 248,320 output rows on SM70. It is intentionally
//! a target-output-head component: the hidden input may be a deterministic
//! fixture or a caller-provided f32 vector, so it does not claim a complete
//! transformer, semantic parity, decode, or product throughput.

use std::{error::Error, fmt, hint::spin_loop, path::PathBuf, time::Instant};

use augur_backend::cuda_f8::CudaF8E4M3RowScaleF32MatvecArgsV1;
use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, cuda_host_boundary_snapshot,
    CudaF32View, CudaF8E4M3WeightViewV1, CudaRmsnormF32ArgsV1, CudaRuntimeError, OwnedCudaBuffer,
    OwnedCudaEvent, OwnedCudaStream,
};
#[cfg(feature = "qwen35-output-head-acceptance")]
use augur_backend::{
    CudaQwen35DflashLogitsAcceptanceArgsV1, CudaQwen35DflashLogitsAcceptanceViewsV1,
    QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES,
};
use augur_core::PhysicalLayoutId;
use augur_model_qwen35::SafetensorsManifest;
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const QWEN35_REAL_TARGET_OUTPUT_HEAD_CUDA_SCHEMA: &str =
    "crystal-quant.qwen35.real-target-output-head-cuda.v1";

const WEIGHT_NAME: &str = "lm_head.weight";
const SCALE_NAME: &str = "lm_head.weight_scale";
const NORM_NAME: &str = "model.language_model.norm.weight";
const HIDDEN: usize = 5_120;
const VOCAB: usize = 248_320;
const MAX_POLLS: u64 = 2_000_000;

pub const QWEN35_REAL_TARGET_OUTPUT_HEAD_HIDDEN: usize = HIDDEN;
pub const QWEN35_REAL_TARGET_OUTPUT_HEAD_VOCAB: usize = VOCAB;

#[derive(Clone, Debug)]
pub struct Qwen35RealTargetOutputHeadCudaConfig {
    pub safetensors_path: PathBuf,
    pub hidden_f32_path: Option<PathBuf>,
    pub hash_file: bool,
    pub device_ordinal: i32,
    pub warmup: usize,
    pub iterations: usize,
    pub top_k: usize,
    pub source_commit: String,
}

impl Default for Qwen35RealTargetOutputHeadCudaConfig {
    fn default() -> Self {
        Self {
            safetensors_path: PathBuf::new(),
            hidden_f32_path: None,
            hash_file: false,
            device_ordinal: 0,
            warmup: 2,
            iterations: 10,
            top_k: 5,
            source_commit: "UNBOUND_COMPONENT_SOURCE".to_owned(),
        }
    }
}

#[derive(Debug)]
pub enum Qwen35RealTargetOutputHeadCudaError {
    Runtime(CudaRuntimeError),
    Model(augur_model_qwen35::Qwen35Error),
    InvalidConfiguration(&'static str),
    Io(String),
    CompletionPollLimit,
}

/// Resident real Qwen output head. The 1.27 GB F8 weight matrix, row scales,
/// final norm, one normalized-hidden scratch row, and a bounded `[K, vocab]`
/// logits buffer remain device-resident across submissions. This is a device
/// ABI component, not a complete target transformer or decode session.
pub struct Qwen35RealTargetOutputHeadCudaSession {
    device_ordinal: i32,
    capacity: usize,
    weight: OwnedCudaBuffer,
    scales: OwnedCudaBuffer,
    norm: OwnedCudaBuffer,
    normalized: OwnedCudaBuffer,
    logits: OwnedCudaBuffer,
    weight_view: CudaF8E4M3WeightViewV1,
    scale_view: CudaF32View,
    norm_view: CudaF32View,
    #[cfg(feature = "qwen35-output-head-acceptance")]
    target_ids: OwnedCudaBuffer,
    #[cfg(feature = "qwen35-output-head-acceptance")]
    receipt: OwnedCudaBuffer,
}

impl Qwen35RealTargetOutputHeadCudaSession {
    pub fn open(
        config: &Qwen35RealTargetOutputHeadCudaConfig,
        stream: &OwnedCudaStream,
        capacity: usize,
    ) -> Result<Self, Qwen35RealTargetOutputHeadCudaError> {
        if capacity == 0 || capacity > 64 || config.safetensors_path.as_os_str().is_empty() {
            return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                "resident output-head capacity and safetensors path",
            ));
        }
        if stream.device_ordinal() != config.device_ordinal {
            return Err(Qwen35RealTargetOutputHeadCudaError::Runtime(
                CudaRuntimeError::device_mismatch(),
            ));
        }
        let properties = cuda_device_properties(config.device_ordinal)?;
        if properties.compute_capability != augur_backend::CudaComputeCapability::SM70 {
            return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                "SM70 device",
            ));
        }
        let manifest = SafetensorsManifest::inspect(&config.safetensors_path, config.hash_file)?;
        let weight_record = manifest.tensor(WEIGHT_NAME).ok_or(
            Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration("lm_head.weight missing"),
        )?;
        let scale_record = manifest.tensor(SCALE_NAME).ok_or(
            Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                "lm_head.weight_scale missing",
            ),
        )?;
        let norm_record = manifest.tensor(NORM_NAME).ok_or(
            Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration("final norm missing"),
        )?;
        if weight_record.dtype != "F8_E4M3"
            || weight_record.shape != vec![VOCAB as u64, HIDDEN as u64]
            || scale_record.dtype != "BF16"
            || scale_record.shape != vec![VOCAB as u64, 1]
            || norm_record.dtype != "BF16"
            || norm_record.shape != vec![HIDDEN as u64]
        {
            return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                "Qwen safetensors output-head geometry",
            ));
        }
        let weight_payload = manifest.read_tensor(WEIGHT_NAME)?;
        let scale_payload = manifest.read_tensor(SCALE_NAME)?;
        let norm_payload = manifest.read_tensor(NORM_NAME)?;
        if weight_payload.len() != VOCAB * HIDDEN
            || scale_payload.len() != VOCAB * 2
            || norm_payload.len() != HIDDEN * 2
        {
            return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                "resident output-head payload lengths",
            ));
        }
        let row_scales = bf16_bytes_to_f32(&scale_payload)?;
        let norm = bf16_bytes_to_f32(&norm_payload)?;
        let weight = OwnedCudaBuffer::allocate(config.device_ordinal, weight_payload.len())?;
        let scales = OwnedCudaBuffer::allocate(config.device_ordinal, row_scales.len() * 4)?;
        let norm_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, norm.len() * 4)?;
        let normalized = OwnedCudaBuffer::allocate(
            config.device_ordinal,
            capacity
                .checked_mul(HIDDEN)
                .and_then(|elements| elements.checked_mul(4))
                .ok_or(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                    "resident normalized extent",
                ))?,
        )?;
        let logits = OwnedCudaBuffer::allocate(
            config.device_ordinal,
            capacity
                .checked_mul(VOCAB)
                .and_then(|elements| elements.checked_mul(4))
                .ok_or(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                    "resident logits extent",
                ))?,
        )?;
        weight.copy_from_host_sync(stream, &weight_payload)?;
        scales.copy_from_host_sync(stream, &f32_bytes(&row_scales))?;
        norm_buffer.copy_from_host_sync(stream, &f32_bytes(&norm))?;
        let weight_view = CudaF8E4M3WeightViewV1::new(
            weight.view(0, weight_payload.len())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F8E4M3_CUDA_V1,
            VOCAB,
            HIDDEN,
        )?;
        let scale_view = CudaF32View::new(
            scales.view(0, row_scales.len() * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            VOCAB,
        )?;
        let norm_view = CudaF32View::new(
            norm_buffer.view(0, norm.len() * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            HIDDEN,
        )?;
        #[cfg(feature = "qwen35-output-head-acceptance")]
        let target_ids = OwnedCudaBuffer::allocate(
            config.device_ordinal,
            capacity.checked_mul(4).ok_or(
                Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                    "resident target-id extent",
                ),
            )?,
        )?;
        #[cfg(feature = "qwen35-output-head-acceptance")]
        let receipt = OwnedCudaBuffer::allocate(
            config.device_ordinal,
            QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES,
        )?;
        Ok(Self {
            device_ordinal: config.device_ordinal,
            capacity,
            weight,
            scales,
            norm: norm_buffer,
            normalized,
            logits,
            weight_view,
            scale_view,
            norm_view,
            #[cfg(feature = "qwen35-output-head-acceptance")]
            target_ids,
            #[cfg(feature = "qwen35-output-head-acceptance")]
            receipt,
        })
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Target backbone hidden, before the target final norm.
    pub fn submit_hidden_device(
        &self,
        stream: &OwnedCudaStream,
        hidden: CudaF32View,
        slot: usize,
    ) -> Result<(), Qwen35RealTargetOutputHeadCudaError> {
        if slot >= self.capacity || stream.device_ordinal() != self.device_ordinal {
            return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                "resident output-head slot/device",
            ));
        }
        if hidden.element_count() != HIDDEN
            || hidden.buffer_view().device_ordinal() != self.device_ordinal
        {
            return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                "resident hidden geometry",
            ));
        }
        let normalized_view = CudaF32View::new(
            self.normalized.view(slot * HIDDEN * 4, HIDDEN * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            HIDDEN,
        )?;
        CudaRmsnormF32ArgsV1::bind(stream, hidden, self.norm_view, normalized_view, 1.0e-6)?
            .submit_for_plan(stream)?;
        self.submit_normalized_hidden_device(stream, normalized_view, slot)
    }

    /// Projection only: input already passed its producing model's final norm.
    /// DFlash owns its norm; applying target norm here changes the draft model.
    /// Keep the input alive until stream completion, as for the raw-hidden API.
    pub fn submit_normalized_hidden_device(
        &self,
        stream: &OwnedCudaStream,
        normalized: CudaF32View,
        slot: usize,
    ) -> Result<(), Qwen35RealTargetOutputHeadCudaError> {
        if slot >= self.capacity || stream.device_ordinal() != self.device_ordinal {
            return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                "resident post-norm output-head slot/device",
            ));
        }
        if normalized.element_count() != HIDDEN
            || normalized.buffer_view().device_ordinal() != self.device_ordinal
        {
            return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                "resident post-norm hidden geometry",
            ));
        }
        let logits_view = CudaF32View::new(
            self.logits.view(slot * VOCAB * 4, VOCAB * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            VOCAB,
        )?;
        CudaF8E4M3RowScaleF32MatvecArgsV1::bind(
            stream,
            self.weight_view,
            self.scale_view,
            normalized,
            logits_view,
        )?
        .submit_for_plan(stream)?;
        Ok(())
    }

    pub fn logits_view(
        &self,
        slot: usize,
    ) -> Result<CudaF32View, Qwen35RealTargetOutputHeadCudaError> {
        if slot >= self.capacity {
            return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                "resident logits slot",
            ));
        }
        Ok(CudaF32View::new(
            self.logits.view(slot * VOCAB * 4, VOCAB * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            VOCAB,
        )?)
    }

    /// Copy the resident logits slab after stream completion. This is a
    /// diagnostic bridge; product sampling should remain device-resident.
    pub fn copy_logits_slab_to_host(
        &self,
        stream: &OwnedCudaStream,
        output: &mut [u8],
    ) -> Result<(), Qwen35RealTargetOutputHeadCudaError> {
        let expected = self
            .capacity
            .checked_mul(VOCAB)
            .and_then(|elements| elements.checked_mul(4))
            .ok_or(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                "resident logits slab extent",
            ))?;
        if output.len() != expected {
            return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                "resident logits slab output extent",
            ));
        }
        self.logits.copy_to_host_sync(stream, output)?;
        Ok(())
    }

    #[cfg(feature = "qwen35-output-head-acceptance")]
    pub fn bind_acceptance(
        &self,
        stream: &OwnedCudaStream,
        draft_ids: augur_backend::CudaBufferView,
        draft_tokens: usize,
        step_generation: u64,
        plan_digest: [u8; 32],
    ) -> Result<CudaQwen35DflashLogitsAcceptanceArgsV1, Qwen35RealTargetOutputHeadCudaError> {
        self.bind_acceptance_at(
            stream,
            draft_ids,
            draft_tokens,
            0,
            step_generation,
            plan_digest,
        )
    }

    #[cfg(feature = "qwen35-output-head-acceptance")]
    pub fn bind_acceptance_at(
        &self,
        stream: &OwnedCudaStream,
        draft_ids: augur_backend::CudaBufferView,
        draft_tokens: usize,
        slot: usize,
        step_generation: u64,
        plan_digest: [u8; 32],
    ) -> Result<CudaQwen35DflashLogitsAcceptanceArgsV1, Qwen35RealTargetOutputHeadCudaError> {
        if draft_tokens == 0 || draft_tokens > self.capacity || slot >= self.capacity {
            return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                "resident acceptance draft extent",
            ));
        }
        Ok(CudaQwen35DflashLogitsAcceptanceArgsV1::bind(
            stream,
            CudaQwen35DflashLogitsAcceptanceViewsV1 {
                draft_ids,
                target_logits: self
                    .logits
                    .view(slot * VOCAB * 4, draft_tokens * VOCAB * 4)?,
                target_ids: self.target_ids.view(0, draft_tokens * 4)?,
                receipt: self
                    .receipt
                    .view(0, QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES)?,
            },
            draft_tokens,
            step_generation,
            plan_digest,
        )?)
    }

    #[cfg(feature = "qwen35-output-head-acceptance")]
    pub fn receipt_view(
        &self,
    ) -> Result<augur_backend::CudaBufferView, Qwen35RealTargetOutputHeadCudaError> {
        Ok(self
            .receipt
            .view(0, QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES)?)
    }
}

impl fmt::Display for Qwen35RealTargetOutputHeadCudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(formatter),
            Self::Model(error) => error.fmt(formatter),
            Self::InvalidConfiguration(field) => write!(
                formatter,
                "invalid Qwen real target output-head CUDA configuration: {field}"
            ),
            Self::Io(error) => write!(formatter, "target output-head I/O error: {error}"),
            Self::CompletionPollLimit => {
                formatter.write_str("CUDA completion polling exceeded its bound")
            }
        }
    }
}

impl Error for Qwen35RealTargetOutputHeadCudaError {}

impl From<CudaRuntimeError> for Qwen35RealTargetOutputHeadCudaError {
    fn from(error: CudaRuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<augur_model_qwen35::Qwen35Error> for Qwen35RealTargetOutputHeadCudaError {
    fn from(error: augur_model_qwen35::Qwen35Error) -> Self {
        Self::Model(error)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35RealTargetOutputHeadCudaReport {
    pub schema: String,
    pub status: String,
    pub component_only: bool,
    pub source_commit: String,
    pub device: DeviceReport,
    pub source: SourceReport,
    pub input: InputReport,
    pub shape: ShapeReport,
    pub timing: TimingReport,
    pub completion: CompletionReport,
    pub numerical: NumericalReport,
    pub top_k: Vec<LogitEntry>,
    pub resources: ResourceReport,
    pub claims: ClaimReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeviceReport {
    pub ordinal: i32,
    pub name: String,
    pub compute_capability: [u16; 2],
    pub total_global_memory_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceReport {
    pub path: String,
    pub file_size: u64,
    pub file_sha256: Option<String>,
    pub header_bytes: u64,
    pub data_offset: u64,
    pub weight_dtype: String,
    pub weight_shape: Vec<u64>,
    pub weight_offsets: [u64; 2],
    pub scale_dtype: String,
    pub scale_shape: Vec<u64>,
    pub scale_offsets: [u64; 2],
    pub norm_dtype: String,
    pub norm_shape: Vec<u64>,
    pub norm_offsets: [u64; 2],
    pub weight_payload_sha256: String,
    pub scale_payload_sha256: String,
    pub norm_payload_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct InputReport {
    pub source: String,
    pub hidden_elements: usize,
    pub hidden_f32_bits_sha256: String,
    pub normalized_f32_bits_sha256: String,
    pub norm_epsilon: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ShapeReport {
    pub rows: usize,
    pub columns: usize,
    pub output_elements: usize,
    pub weight_bytes: usize,
    pub row_scale_elements: usize,
    pub row_scale_bytes: usize,
    pub output_bytes: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct TimingReport {
    pub setup_wall_ms: f64,
    pub cuda_event_ms: f32,
    pub batch_wall_ms: f64,
    pub per_submission_event_ms: f64,
    pub per_submission_wall_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CompletionReport {
    pub poll_iterations: u64,
    pub nonblocking_poll_only: bool,
    pub completion_proof_required: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct NumericalReport {
    pub finite_count: usize,
    pub expected_count: usize,
    pub max_abs_diff: f32,
    pub relative_l2: f32,
    pub tolerance: f32,
    pub cpu_reference_sha256: String,
    pub gpu_output_sha256: String,
    pub pass: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct LogitEntry {
    pub token_id: usize,
    pub value: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResourceReport {
    pub setup_allocations: AllocationDeltaReport,
    pub steady_allocations: AllocationDeltaReport,
    pub setup_host_boundaries: HostBoundaryDeltaReport,
    pub steady_host_boundaries: HostBoundaryDeltaReport,
    pub completion_host_boundaries: HostBoundaryDeltaReport,
    pub final_allocation_delta_after_cleanup: AllocationDeltaReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct AllocationDeltaReport {
    pub successful_allocations: u64,
    pub successful_frees: u64,
    pub live_allocations_after: u64,
    pub live_bytes_after: u64,
    pub peak_live_allocations_observed: u64,
    pub peak_live_bytes_observed: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HostBoundaryDeltaReport {
    pub host_waits: u64,
    pub stream_synchronizations: u64,
    pub event_synchronizations: u64,
    pub synchronous_h2d_copies: u64,
    pub synchronous_h2d_bytes: u64,
    pub synchronous_d2h_copies: u64,
    pub synchronous_d2h_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClaimReport {
    pub real_safetensors_lm_head_loaded: bool,
    pub real_row_scale_cuda_matvec_executed: bool,
    pub target_logits_from_real_output_head: bool,
    pub complete_transformer_executed: bool,
    pub decode_executed: bool,
    pub semantic_parity_established: bool,
    pub product_tps_permitted: bool,
}

pub fn run_qwen35_real_target_output_head_cuda(
    config: &Qwen35RealTargetOutputHeadCudaConfig,
) -> Result<Qwen35RealTargetOutputHeadCudaReport, Qwen35RealTargetOutputHeadCudaError> {
    run_qwen35_real_target_output_head_cuda_impl(config, None)
}

pub fn run_qwen35_real_target_output_head_cuda_with_hidden(
    config: &Qwen35RealTargetOutputHeadCudaConfig,
    hidden: &[f32],
) -> Result<Qwen35RealTargetOutputHeadCudaReport, Qwen35RealTargetOutputHeadCudaError> {
    run_qwen35_real_target_output_head_cuda_impl(config, Some(hidden))
}

fn run_qwen35_real_target_output_head_cuda_impl(
    config: &Qwen35RealTargetOutputHeadCudaConfig,
    hidden_override: Option<&[f32]>,
) -> Result<Qwen35RealTargetOutputHeadCudaReport, Qwen35RealTargetOutputHeadCudaError> {
    if config.safetensors_path.as_os_str().is_empty()
        || config.warmup > 1_000
        || config.iterations == 0
        || config.iterations > 100_000
        || config.top_k == 0
        || config.top_k > 1_000
    {
        return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
            "paths, warmup, iterations, and top_k",
        ));
    }
    let setup_start = Instant::now();
    let manifest = SafetensorsManifest::inspect(&config.safetensors_path, config.hash_file)?;
    let weight_record = manifest.tensor(WEIGHT_NAME).ok_or(
        Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration("lm_head.weight missing"),
    )?;
    let scale_record = manifest.tensor(SCALE_NAME).ok_or(
        Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration("lm_head.weight_scale missing"),
    )?;
    let norm_record = manifest.tensor(NORM_NAME).ok_or(
        Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration("final norm missing"),
    )?;
    if weight_record.dtype != "F8_E4M3"
        || weight_record.shape != vec![VOCAB as u64, HIDDEN as u64]
        || scale_record.dtype != "BF16"
        || scale_record.shape != vec![VOCAB as u64, 1]
        || norm_record.dtype != "BF16"
        || norm_record.shape != vec![HIDDEN as u64]
    {
        return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
            "Qwen safetensors output-head geometry",
        ));
    }
    let weight_payload = manifest.read_tensor(WEIGHT_NAME)?;
    let scale_payload = manifest.read_tensor(SCALE_NAME)?;
    let norm_payload = manifest.read_tensor(NORM_NAME)?;
    if weight_payload.len() != VOCAB * HIDDEN
        || scale_payload.len() != VOCAB * 2
        || norm_payload.len() != HIDDEN * 2
    {
        return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
            "output-head payload lengths",
        ));
    }
    let row_scales = bf16_bytes_to_f32(&scale_payload)?;
    let norm = bf16_bytes_to_f32(&norm_payload)?;
    let hidden = match hidden_override {
        Some(hidden) => hidden.to_vec(),
        None => load_hidden(config.hidden_f32_path.as_ref())?,
    };
    let normalized = rms_normalize(&hidden, &norm, 1.0e-6)?;
    let expected = row_scale_f8_matvec(&weight_payload, &row_scales, &normalized, VOCAB, HIDDEN)?;
    let top_k = top_k(&expected, config.top_k);
    let properties = cuda_device_properties(config.device_ordinal)?;
    if properties.compute_capability != augur_backend::CudaComputeCapability::SM70 {
        return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
            "SM70 device",
        ));
    }
    let allocation_start = cuda_device_allocation_snapshot();
    let boundary_start = cuda_host_boundary_snapshot();
    let stream = OwnedCudaStream::create(config.device_ordinal)?;
    let weight_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, weight_payload.len())?;
    let scales_f32 = f32_bytes(&row_scales);
    let scales_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, scales_f32.len())?;
    let input_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, normalized.len() * 4)?;
    let output_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, VOCAB * 4)?;
    weight_buffer.copy_from_host_sync(&stream, &weight_payload)?;
    scales_buffer.copy_from_host_sync(&stream, &scales_f32)?;
    input_buffer.copy_from_host_sync(&stream, &f32_bytes(&normalized))?;
    let weight_view = CudaF8E4M3WeightViewV1::new(
        weight_buffer.view(0, weight_payload.len())?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F8E4M3_CUDA_V1,
        VOCAB,
        HIDDEN,
    )?;
    let input_view = CudaF32View::new(
        input_buffer.view(0, normalized.len() * 4)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        HIDDEN,
    )?;
    let scale_view = CudaF32View::new(
        scales_buffer.view(0, scales_f32.len())?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        VOCAB,
    )?;
    let output_view = CudaF32View::new(
        output_buffer.view(0, VOCAB * 4)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        VOCAB,
    )?;
    let args = CudaF8E4M3RowScaleF32MatvecArgsV1::bind(
        &stream,
        weight_view,
        scale_view,
        input_view,
        output_view,
    )?;
    let setup_allocations = delta_allocations(allocation_start, cuda_device_allocation_snapshot())?;
    let setup_host_boundaries = delta_boundaries(boundary_start, cuda_host_boundary_snapshot())?;
    for _ in 0..config.warmup {
        args.submit_for_plan(&stream)?;
    }
    stream.synchronize()?;
    let steady_alloc_start = cuda_device_allocation_snapshot();
    let steady_boundary_start = cuda_host_boundary_snapshot();
    let started = OwnedCudaEvent::create(config.device_ordinal)?;
    let stopped = OwnedCudaEvent::create(config.device_ordinal)?;
    let wall_start = Instant::now();
    started.record(&stream)?;
    let mut last_receipt = None;
    for _ in 0..config.iterations {
        args.submit_for_plan(&stream)?;
        last_receipt = Some(stream.submission_receipt()?);
    }
    let receipt = last_receipt.ok_or(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
        "steady submission receipt",
    ))?;
    stopped.record_for_submission(&stream, receipt)?;
    stopped.synchronize()?;
    let batch_wall_ms = wall_start.elapsed().as_secs_f64() * 1_000.0;
    let cuda_event_ms = started.elapsed_ms(&stopped)?;
    let steady_allocations =
        delta_allocations(steady_alloc_start, cuda_device_allocation_snapshot())?;
    let steady_host_boundaries =
        delta_boundaries(steady_boundary_start, cuda_host_boundary_snapshot())?;
    let completion_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let completion_boundary_start = cuda_host_boundary_snapshot();
    args.submit_for_plan(&stream)?;
    let completion_receipt = stream.submission_receipt()?;
    completion_event.record_for_submission(&stream, completion_receipt)?;
    let (poll_iterations, proof) = poll_completion(&stream, &completion_event, completion_receipt)?;
    if !proof {
        return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
            "completion proof binding",
        ));
    }
    let completion_host_boundaries =
        delta_boundaries(completion_boundary_start, cuda_host_boundary_snapshot())?;
    let readback_boundary_start = cuda_host_boundary_snapshot();
    let mut output_bytes = vec![0_u8; VOCAB * 4];
    output_buffer.copy_to_host_sync_with_submission_proof(
        &stream,
        &mut output_bytes,
        completion_receipt,
    )?;
    let readback_boundaries =
        delta_boundaries(readback_boundary_start, cuda_host_boundary_snapshot())?;
    let actual = bytes_to_f32(&output_bytes)?;
    let max_abs = max_abs_diff(&actual, &expected);
    let relative_l2 = relative_l2(&actual, &expected);
    let finite_count = actual.iter().filter(|value| value.is_finite()).count();
    let tolerance = 2.0e-2_f32;
    let pass = finite_count == VOCAB && max_abs <= tolerance && relative_l2 <= 1.0e-4;
    drop(output_buffer);
    drop(input_buffer);
    drop(scales_buffer);
    drop(weight_buffer);
    stream.synchronize()?;
    let final_allocation_delta_after_cleanup =
        delta_allocations(allocation_start, cuda_device_allocation_snapshot())?;
    let setup_wall_ms = setup_start.elapsed().as_secs_f64() * 1_000.0;
    Ok(Qwen35RealTargetOutputHeadCudaReport {
        schema: QWEN35_REAL_TARGET_OUTPUT_HEAD_CUDA_SCHEMA.to_owned(),
        status: if pass {
            "PASS_COMPONENT_QWEN35_REAL_TARGET_OUTPUT_HEAD_CUDA"
        } else {
            "FAIL_COMPONENT_QWEN35_REAL_TARGET_OUTPUT_HEAD_CUDA"
        }
        .to_owned(),
        component_only: true,
        source_commit: config.source_commit.clone(),
        device: DeviceReport {
            ordinal: properties.ordinal,
            name: properties.name,
            compute_capability: [
                properties.compute_capability.major,
                properties.compute_capability.minor,
            ],
            total_global_memory_bytes: properties.total_global_memory_bytes,
        },
        source: SourceReport {
            path: manifest.path().display().to_string(),
            file_size: manifest.file_size,
            file_sha256: manifest.file_sha256.clone(),
            header_bytes: manifest.header_bytes,
            data_offset: manifest.data_offset,
            weight_dtype: weight_record.dtype.clone(),
            weight_shape: weight_record.shape.clone(),
            weight_offsets: weight_record.data_offsets,
            scale_dtype: scale_record.dtype.clone(),
            scale_shape: scale_record.shape.clone(),
            scale_offsets: scale_record.data_offsets,
            norm_dtype: norm_record.dtype.clone(),
            norm_shape: norm_record.shape.clone(),
            norm_offsets: norm_record.data_offsets,
            weight_payload_sha256: sha256_hex(&weight_payload),
            scale_payload_sha256: sha256_hex(&scale_payload),
            norm_payload_sha256: sha256_hex(&norm_payload),
        },
        input: InputReport {
            source: if hidden_override.is_some() || config.hidden_f32_path.is_some() {
                "caller_hidden_f32"
            } else {
                "deterministic_hidden_fixture"
            }
            .to_owned(),
            hidden_elements: HIDDEN,
            hidden_f32_bits_sha256: sha256_f32(&hidden),
            normalized_f32_bits_sha256: sha256_f32(&normalized),
            norm_epsilon: 1.0e-6,
        },
        shape: ShapeReport {
            rows: VOCAB,
            columns: HIDDEN,
            output_elements: VOCAB,
            weight_bytes: weight_payload.len(),
            row_scale_elements: row_scales.len(),
            row_scale_bytes: scales_f32.len(),
            output_bytes: output_bytes.len(),
        },
        timing: TimingReport {
            setup_wall_ms,
            cuda_event_ms,
            batch_wall_ms,
            per_submission_event_ms: f64::from(cuda_event_ms) / config.iterations as f64,
            per_submission_wall_ms: batch_wall_ms / config.iterations as f64,
        },
        completion: CompletionReport {
            poll_iterations,
            nonblocking_poll_only: true,
            completion_proof_required: true,
        },
        numerical: NumericalReport {
            finite_count,
            expected_count: expected.len(),
            max_abs_diff: max_abs,
            relative_l2,
            tolerance,
            cpu_reference_sha256: sha256_f32(&expected),
            gpu_output_sha256: sha256_f32(&actual),
            pass,
        },
        top_k,
        resources: ResourceReport {
            setup_allocations,
            steady_allocations,
            setup_host_boundaries,
            steady_host_boundaries,
            completion_host_boundaries: merge_boundaries(
                completion_host_boundaries,
                readback_boundaries,
            ),
            final_allocation_delta_after_cleanup,
        },
        claims: ClaimReport {
            real_safetensors_lm_head_loaded: true,
            real_row_scale_cuda_matvec_executed: true,
            target_logits_from_real_output_head: true,
            complete_transformer_executed: false,
            decode_executed: false,
            semantic_parity_established: false,
            product_tps_permitted: false,
        },
    })
}

fn load_hidden(path: Option<&PathBuf>) -> Result<Vec<f32>, Qwen35RealTargetOutputHeadCudaError> {
    if let Some(path) = path {
        let bytes = std::fs::read(path)
            .map_err(|error| Qwen35RealTargetOutputHeadCudaError::Io(error.to_string()))?;
        return bytes_to_f32(&bytes);
    }
    Ok((0..HIDDEN)
        .map(|index| ((index % 97) as f32 - 48.0) / 83.0)
        .collect())
}

fn rms_normalize(
    hidden: &[f32],
    norm: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, Qwen35RealTargetOutputHeadCudaError> {
    if hidden.len() != HIDDEN || norm.len() != HIDDEN || !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
            "rms geometry",
        ));
    }
    let mean_square = hidden.iter().map(|value| value * value).sum::<f32>() / HIDDEN as f32;
    let inv = (mean_square + epsilon).sqrt().recip();
    let values = hidden
        .iter()
        .zip(norm)
        .map(|(value, weight)| value * inv * weight)
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
            "normalized hidden",
        ));
    }
    Ok(values)
}

fn row_scale_f8_matvec(
    weight: &[u8],
    scales: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, Qwen35RealTargetOutputHeadCudaError> {
    if rows != VOCAB
        || cols != HIDDEN
        || weight.len() != rows * cols
        || scales.len() != rows
        || input.len() != cols
    {
        return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
            "row-scale matvec geometry",
        ));
    }
    let mut output = vec![0.0_f32; rows];
    for row in 0..rows {
        let scale = scales[row];
        if !scale.is_finite() {
            return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                "non-finite row scale",
            ));
        }
        let row_start = row * cols;
        let mut sum = 0.0_f32;
        for col in 0..cols {
            sum += f8_e4m3_to_f32(weight[row_start + col]) * scale * input[col];
        }
        output[row] = sum;
    }
    Ok(output)
}

fn f8_e4m3_to_f32(value: u8) -> f32 {
    let sign = if value & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = (value >> 3) & 0x0f;
    let mantissa = value & 0x07;
    if exponent == 0 {
        return sign * (mantissa as f32) * 2.0_f32.powi(-9);
    }
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    sign * (1.0 + mantissa as f32 / 8.0) * 2.0_f32.powi(exponent as i32 - 7)
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, Qwen35RealTargetOutputHeadCudaError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
            "BF16 byte length",
        ));
    }
    let values = bytes
        .chunks_exact(2)
        .map(|chunk| f32::from(bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]))))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
            "non-finite BF16 payload",
        ));
    }
    Ok(values)
}

fn poll_completion(
    stream: &OwnedCudaStream,
    event: &OwnedCudaEvent,
    receipt: augur_backend::CudaStreamSubmissionReceipt,
) -> Result<(u64, bool), Qwen35RealTargetOutputHeadCudaError> {
    let mut polls = 0_u64;
    loop {
        if stream.query_event_completion(event, receipt)?.is_some() {
            return Ok((polls, true));
        }
        polls = polls.saturating_add(1);
        if polls > MAX_POLLS {
            return Err(Qwen35RealTargetOutputHeadCudaError::CompletionPollLimit);
        }
        spin_loop();
    }
}

fn top_k(values: &[f32], count: usize) -> Vec<LogitEntry> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|left, right| {
        values[*right]
            .partial_cmp(&values[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    indices
        .into_iter()
        .take(count)
        .map(|token_id| LogitEntry {
            token_id,
            value: values[token_id],
        })
        .collect()
}

fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn relative_l2(actual: &[f32], expected: &[f32]) -> f32 {
    let numerator = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f32>()
        .sqrt();
    let denominator = expected
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1.0e-12);
    numerator / denominator
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, Qwen35RealTargetOutputHeadCudaError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
            "f32 byte length",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn sha256_f32(values: &[f32]) -> String {
    sha256_hex(&f32_bytes(values))
}

fn delta_allocations(
    before: augur_backend::CudaDeviceAllocationSnapshot,
    after: augur_backend::CudaDeviceAllocationSnapshot,
) -> Result<AllocationDeltaReport, Qwen35RealTargetOutputHeadCudaError> {
    Ok(AllocationDeltaReport {
        successful_allocations: after
            .successful_allocations
            .checked_sub(before.successful_allocations)
            .ok_or(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                "allocation counters",
            ))?,
        successful_frees: after
            .successful_frees
            .checked_sub(before.successful_frees)
            .ok_or(Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration(
                "allocation counters",
            ))?,
        live_allocations_after: after.live_allocations,
        live_bytes_after: after.live_bytes,
        peak_live_allocations_observed: after.peak_live_allocations,
        peak_live_bytes_observed: after.peak_live_bytes,
    })
}

fn delta_boundaries(
    before: augur_backend::CudaHostBoundarySnapshot,
    after: augur_backend::CudaHostBoundarySnapshot,
) -> Result<HostBoundaryDeltaReport, Qwen35RealTargetOutputHeadCudaError> {
    let delta = after.checked_delta_since(before).ok_or(
        Qwen35RealTargetOutputHeadCudaError::InvalidConfiguration("host counters"),
    )?;
    Ok(HostBoundaryDeltaReport {
        host_waits: delta.host_waits,
        stream_synchronizations: delta.stream_synchronizations,
        event_synchronizations: delta.event_synchronizations,
        synchronous_h2d_copies: delta.synchronous_h2d_copies,
        synchronous_h2d_bytes: delta.synchronous_h2d_bytes,
        synchronous_d2h_copies: delta.synchronous_d2h_copies,
        synchronous_d2h_bytes: delta.synchronous_d2h_bytes,
    })
}

fn merge_boundaries(
    left: HostBoundaryDeltaReport,
    right: HostBoundaryDeltaReport,
) -> HostBoundaryDeltaReport {
    HostBoundaryDeltaReport {
        host_waits: left.host_waits + right.host_waits,
        stream_synchronizations: left.stream_synchronizations + right.stream_synchronizations,
        event_synchronizations: left.event_synchronizations + right.event_synchronizations,
        synchronous_h2d_copies: left.synchronous_h2d_copies + right.synchronous_h2d_copies,
        synchronous_h2d_bytes: left.synchronous_h2d_bytes + right.synchronous_h2d_bytes,
        synchronous_d2h_copies: left.synchronous_d2h_copies + right.synchronous_d2h_copies,
        synchronous_d2h_bytes: left.synchronous_d2h_bytes + right.synchronous_d2h_bytes,
    }
}

#[cfg(test)]
mod post_norm_tests {
    use super::*;

    #[test]
    #[ignore = "requires SM70 and AUGUR_QWEN35_SAFETENSORS pointing at real target weights"]
    fn real_weight_post_norm_projection_preserves_input_and_target_norm_is_separate(
    ) -> Result<(), Box<dyn Error>> {
        let path = PathBuf::from(std::env::var("AUGUR_QWEN35_SAFETENSORS")?);
        let manifest = SafetensorsManifest::inspect(&path, false)?;
        let weights = manifest.read_tensor(WEIGHT_NAME)?;
        let scales = bf16_bytes_to_f32(&manifest.read_tensor(SCALE_NAME)?)?;
        let norm = bf16_bytes_to_f32(&manifest.read_tensor(NORM_NAME)?)?;
        // Sparse, deliberately non-unit input makes an accidental second norm
        // observable. Reference reads only four columns, independent of CUDA.
        let columns = [0, 127, 2048, HIDDEN - 1];
        let mut hidden = vec![0.0; HIDDEN];
        for (col, value) in columns.into_iter().zip([3.0, -1.5, 0.75, 5.0]) {
            hidden[col] = value;
        }
        let normalized = rms_normalize(&hidden, &norm, 1.0e-6)?;
        let reference = |values: &[f32]| -> Vec<f32> {
            (0..VOCAB)
                .map(|row| {
                    columns
                        .iter()
                        .map(|&col| {
                            f64::from(f8_e4m3_to_f32(weights[row * HIDDEN + col]))
                                * f64::from(scales[row])
                                * f64::from(values[col])
                        })
                        .sum::<f64>() as f32
                })
                .collect()
        };
        let expected = reference(&hidden);
        let expected_target = reference(&normalized);
        assert!(max_abs_diff(&expected, &expected_target) > 0.01);
        let stream = OwnedCudaStream::create(0)?;
        let session = Qwen35RealTargetOutputHeadCudaSession::open(
            &Qwen35RealTargetOutputHeadCudaConfig {
                safetensors_path: path,
                ..Default::default()
            },
            &stream,
            2,
        )?;
        let input = OwnedCudaBuffer::allocate(0, HIDDEN * 4)?;
        input.copy_from_host_sync(&stream, &f32_bytes(&hidden))?;
        let view = CudaF32View::new(
            input.view(0, HIDDEN * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            HIDDEN,
        )?;
        assert!(session
            .submit_normalized_hidden_device(&stream, view, 2)
            .is_err());
        let short = CudaF32View::new(
            input.view(0, (HIDDEN - 1) * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            HIDDEN - 1,
        )?;
        assert!(session
            .submit_normalized_hidden_device(&stream, short, 0)
            .is_err());
        session.submit_normalized_hidden_device(&stream, view, 0)?;
        session.submit_hidden_device(&stream, view, 1)?;
        stream.synchronize()?;
        let mut bytes = vec![0u8; 2 * VOCAB * 4];
        session.logits.copy_to_host_sync(&stream, &mut bytes)?;
        let actual = bytes_to_f32(&bytes)?;
        for (slot, reference) in [&expected, &expected_target].into_iter().enumerate() {
            let row = &actual[slot * VOCAB..(slot + 1) * VOCAB];
            let peak = reference.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let max_abs = max_abs_diff(row, reference);
            let l2 = relative_l2(row, reference);
            eprintln!("POST_NORM_HEAD slot={slot} max_abs={max_abs} relative_l2={l2}");
            assert!(row.iter().all(|v| v.is_finite()));
            assert!(max_abs <= 1.0e-5 + 2.0e-5 * peak);
            assert!(l2 <= 2.0e-5);
            assert_eq!(top_k(row, 1)[0].token_id, top_k(reference, 1)[0].token_id);
        }
        // This numerical diagnostic reads logits for comparison, unlike the
        // production selection boundary's compact token-only D2H.
        drop(input);
        drop(session);
        Ok(())
    }
}
