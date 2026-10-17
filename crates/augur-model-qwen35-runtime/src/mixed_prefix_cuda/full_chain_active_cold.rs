//! Bounded teacher-forced suffix through a real 64-layer chain. Full-cache oracle
//! allocations and prefill scratch are dropped before candidate execution.
//! This is not a rolling decoder or a product/semantic/performance admission.

use super::real_layer_lifecycle::{AsyncColdPageTransport, DevicePartialMerge};
use super::*;
use crate::run_dflash_real_prefix_cuda_ab;
use crate::{
    inspect_dflash_artifact, upload_dflash_after_handoff, upload_dflash_after_handoff_from_ram,
    DFlashCompletionProof, DFlashDType, DFlashHotWindow, DFlashHotWindowConfig, DFlashLifecycle,
    DFlashLifecycleBudget, DFlashPrefillReleaseObservation, DFlashRamResident,
    DFlashRealProducerPrefix, DFlashRequestIdentity, DFlashSelectorArtifact, DFlashUploadConfig,
    DFLASH_HIDDEN_ELEMENTS,
};
use augur_backend::cuda_qwen35_kv8_segment::device_merge::LSE_FINALIZE;
use augur_backend::CudaStreamSubmissionReceipt;
#[cfg(feature = "qwen35-device-token-handoff")]
use augur_backend::{
    dflash_acceptance_cpu_v1, parse_cuda_qwen35_dflash_acceptance_receipt_v1,
    CudaQwen35DflashLogitsAcceptanceArgsV1, CudaQwen35DflashLogitsAcceptanceViewsV1,
    QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES,
};
use augur_backend::{
    CudaBf16F32MatvecArgsV1, CudaBf16WeightViewV1, CudaF32View, CudaPinnedH2dDependencyReceipt,
    CudaQ8_0F32MatvecArgsV1, CudaQ8_0WeightViewV1, CudaStreamCompletionProof, OwnedCudaBuffer,
    OwnedPinnedH2dBuffer,
};
use augur_core::PhysicalLayoutId;
use augur_model_qwen35::{
    quantized_matvec, DFlashConfig, GgufAuditOptions, GgufManifest, GGML_TYPE_Q8_0,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::OpenOptions, io::Write};
mod page_lease;
use page_lease::{PageBinding, PageLease, PageLeaseBook};

const ACTIVE: usize = 2048;
const PAGE: usize = 512;
const MODEL_CONTEXT: usize = 262_144;
// Keep a full 512-row tail available for a suffix while retaining page-aligned
// prefixes. The runner separately enforces prefix + suffix <= MODEL_CONTEXT.
const MAX_PAGE_ALIGNED_PREFIX: usize = MODEL_CONTEXT - PAGE;
const LANES: usize = 4;
const CHUNK: usize = 128;
const ROW: usize = QWEN35_KV8_CACHE_ROW_BYTES;
const Q8_SELECTOR_TENSOR: &str = "selector_hidden.weight";
const Q8_SELECTOR_ROWS: usize = 256;
const Q8_SELECTOR_COLS: usize = 5120;

fn digest_hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn write_f32_export(path: &PathBuf, values: &[f32]) -> Result<String, Qwen35MixedPrefixCudaError> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("target hidden export is empty or non-finite"));
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| invalid(format!("target hidden export open: {error}")))?;
    let mut digest = Sha256::new();
    for chunk in values.chunks(4096) {
        let bytes = f32_bytes(chunk);
        file.write_all(&bytes)
            .map_err(|error| invalid(format!("target hidden export write: {error}")))?;
        digest.update(&bytes);
    }
    file.sync_all()
        .map_err(|error| invalid(format!("target hidden export sync: {error}")))?;
    Ok(digest_hex(&digest.finalize().into()))
}
#[derive(Clone, Debug)]
pub struct Qwen35FullChainActiveColdConfig {
    pub gguf_path: PathBuf,
    pub config_path: PathBuf,
    pub prefix_positions: usize,
    /// Sequential teacher-forced inputs, with candidate-owned state carried
    /// between steps. Default one preserves the original component scope.
    pub suffix_tokens: usize,
    pub source_commit: String,
    /// Optional diagnostic export of the five target hidden rows consumed by
    /// the real DFlash target-feature producer. The default path is unchanged.
    pub target_feature_export_path: Option<PathBuf>,
    /// Optional diagnostic export of the target final-hidden rows used by the
    /// target output head. This is never a product/publication artifact.
    pub target_final_hidden_export_path: Option<PathBuf>,
    /// Capture only the bounded target source rows; skip the memory-heavy
    /// candidate comparison path. The resulting report cannot claim parity.
    pub capture_target_only: bool,
    /// Optional real safetensors DFlash artifact. When present, the loader is
    /// admitted only after the physical prefill handoff and held until the
    /// candidate terminal; decode still does not consume it in this runner.
    pub dflash_artifact_path: Option<PathBuf>,
    pub dflash_config_path: Option<PathBuf>,
    pub dflash_allow_f16: bool,
    pub dflash_chunk_bytes: usize,
    /// Keep a complete validated artifact in ordinary pageable RAM and use it
    /// as the source for the post-prefill four-lane upload.
    pub dflash_source_ram: bool,
    pub dflash_ram_budget_bytes: u64,
    /// Consume only the real selector hidden-projection tile through a bounded
    /// HBM hot window. BF16 is copied byte-for-byte; F16 is converted to BF16
    /// in pageable RAM before upload. This deliberately skips full DFlash
    /// upload and does not claim the five-layer draft backbone.
    pub dflash_hot_window_only: bool,
    /// Optional complete Q8 GGUF DFlash owner.  The file is copied once to
    /// HBM after the prefill handoff and the selector_hidden tensor is
    /// consumed by the target-bound Q8 matvec component.  This is independent
    /// from the BF16 hot-window path and does not claim the full DFlash draft.
    pub dflash_q8_gguf_path: Option<PathBuf>,
    pub inject_upload_corruption: bool,
    /// Opt-in real page-path fault. A rejection emits a separate NO_GO
    /// diagnostic, never a successful tensor/qualification report.
    pub fault_injection: Option<String>,
    pub fault_step: usize,
    pub fault_layer: usize,
    pub fault_page: usize,
    pub direct_post_attention: bool,
    /// Diagnostic only: replace the candidate input at one layer with the
    /// contiguous oracle input to isolate preceding layer drift.
    pub freeze_oracle_input_layer: Option<usize>,
    /// Diagnostic only: receipt-bound snapshots of the device partial/LSE
    /// planes at layer 3. The default execution path remains unchanged.
    pub audit_partial_state: bool,
    /// Diagnostic only: use the FP64-accumulating GDN candidate for the paged
    /// execution. The contiguous oracle remains on the established path.
    pub precise_gdn: bool,
    /// Receipt-bound observations of unmodified GDN 4--6, not a fallback.
    pub audit_gdn_stages: bool,
    /// Diagnostic-only layer-3 arithmetic A/B. Mode 2 mirrors the V1 SIMT
    /// reference ordering while retaining the V2 partial/page ABI.
    pub segment_v1_match: bool,
    /// Explicit segment producer mode. Modes 0--2 remain bounded arithmetic
    /// diagnostics; mode 3 is the semantic rolling path and carries its FP32
    /// online recurrence across pages (cumulative, not independent partials).
    /// Mode 3 is the default because it is the only mode that preserves the
    /// ordered page contract across repeated full-attention layers.
    pub segment_mode: u32,
    /// Opt-in free-running suffix. The default remains teacher-forced so
    /// existing component evidence is reproducible.
    pub free_running: bool,
}

impl Default for Qwen35FullChainActiveColdConfig {
    fn default() -> Self {
        Self {
            gguf_path: PathBuf::new(),
            config_path: PathBuf::new(),
            prefix_positions: 4096,
            suffix_tokens: 1,
            source_commit: "DIRTY_FULL_CHAIN_ACTIVE_COLD".into(),
            target_feature_export_path: None,
            target_final_hidden_export_path: None,
            capture_target_only: false,
            dflash_artifact_path: None,
            dflash_config_path: None,
            dflash_allow_f16: false,
            dflash_chunk_bytes: DFlashUploadConfig::default().chunk_bytes,
            dflash_source_ram: false,
            dflash_ram_budget_bytes: 12 * 1024 * 1024 * 1024,
            dflash_hot_window_only: false,
            dflash_q8_gguf_path: None,
            inject_upload_corruption: false,
            fault_injection: None,
            fault_step: 1,
            fault_layer: 3,
            fault_page: 0,
            direct_post_attention: true,
            freeze_oracle_input_layer: None,
            audit_partial_state: false,
            precise_gdn: false,
            audit_gdn_stages: false,
            segment_v1_match: false,
            segment_mode: 3,
            free_running: false,
        }
    }
}

/// Physical slot state for the target-bound selector projection consumer.
/// The slot may be overwritten only after the compute-stream terminal proof
/// has been observed and the pinned staging dependency has been retired.
struct DFlashHotWindowCudaSlot {
    device: OwnedCudaBuffer,
    staging: OwnedPinnedH2dBuffer,
    output: OwnedCudaBuffer,
    active: Option<DFlashHotWindowCudaTicket>,
}

struct DFlashHotWindowCudaTicket {
    step: usize,
    dependency: CudaPinnedH2dDependencyReceipt,
    terminal_event: OwnedCudaEvent,
    terminal_receipt: CudaStreamSubmissionReceipt,
    lease: crate::DFlashHotWindowLease,
    terminal_digest: [u8; 32],
}

struct DFlashHotWindowCudaConsumer {
    selector: DFlashSelectorArtifact,
    selector_tile_bf16: Vec<u8>,
    selector_source_dtype: DFlashDType,
    control: DFlashHotWindow,
    copy_stream: OwnedCudaStream,
    slots: Vec<DFlashHotWindowCudaSlot>,
    rows: usize,
    cols: usize,
    tile_bytes: usize,
    submissions: usize,
    last_expected: Option<Vec<f32>>,
    last_output_slot: Option<usize>,
}

impl DFlashHotWindowCudaConsumer {
    fn new(
        selector: DFlashSelectorArtifact,
        device: i32,
        request: DFlashRequestIdentity,
    ) -> Result<Self, Qwen35MixedPrefixCudaError> {
        let selector_tile_bf16 = selector
            .hidden_projection_tile_bf16_owned()
            .map_err(|error| invalid(format!("DFlash selector tile: {error}")))?;
        let selector_source_dtype = selector.resident().inspection().artifact.dtype;
        let tile_bytes = selector_tile_bf16.len();
        if tile_bytes == 0 || tile_bytes != selector.selector_rank() * selector.hidden_size() * 2 {
            return Err(invalid("DFlash selector hidden-projection tile geometry"));
        }
        let control = DFlashHotWindow::new(DFlashHotWindowConfig {
            capacity_positions: MODEL_CONTEXT,
            active_window_positions: ACTIVE,
            slot_count: 2,
            tile_bytes,
        })
        .map_err(|error| invalid(format!("DFlash hot-window control: {error}")))?;
        let mut control = control;
        control
            .begin_request(request, 1)
            .map_err(|error| invalid(format!("DFlash hot-window request: {error}")))?;
        let copy_stream = OwnedCudaStream::create(device)?;
        let rows = selector.selector_rank();
        let cols = selector.hidden_size();
        let slots = (0..2)
            .map(|_| {
                Ok(DFlashHotWindowCudaSlot {
                    device: OwnedCudaBuffer::allocate(device, tile_bytes)?,
                    staging: OwnedPinnedH2dBuffer::allocate_host_only(device, tile_bytes)?,
                    output: OwnedCudaBuffer::allocate(device, rows * 4)?,
                    active: None,
                })
            })
            .collect::<Result<Vec<_>, CudaRuntimeError>>()?;
        Ok(Self {
            selector,
            selector_tile_bf16,
            selector_source_dtype,
            control,
            copy_stream,
            slots,
            rows,
            cols,
            tile_bytes,
            submissions: 0,
            last_expected: None,
            last_output_slot: None,
        })
    }

    fn retire_slot(
        &mut self,
        slot_index: usize,
        compute_stream: &OwnedCudaStream,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        let slot = self
            .slots
            .get_mut(slot_index)
            .ok_or_else(|| invalid("DFlash hot-window slot index"))?;
        let Some(ticket) = slot.active.take() else {
            return Ok(());
        };
        let proof: CudaStreamCompletionProof = loop {
            if let Some(proof) = compute_stream
                .query_event_completion(&ticket.terminal_event, ticket.terminal_receipt)?
            {
                break proof;
            }
            std::hint::spin_loop();
        };
        slot.staging
            .release_after_dependent_completion(ticket.dependency, &proof)?;
        self.control
            .mark_terminal(ticket.lease, ticket.terminal_digest)
            .map_err(|error| invalid(format!("DFlash hot-window terminal: {error}")))?;
        Ok(())
    }

    fn consume(
        &mut self,
        step: usize,
        hidden: &[f32],
        hidden_view: CudaF32View,
        compute_stream: &OwnedCudaStream,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if hidden.len() != self.cols {
            return Err(invalid("DFlash hot-window hidden input geometry"));
        }
        let slot_index = step % self.slots.len();
        self.retire_slot(slot_index, compute_stream)?;
        self.slots[slot_index]
            .staging
            .writable()?
            .copy_from_slice(&self.selector_tile_bf16);
        let destination = self.slots[slot_index].device.view(0, self.tile_bytes)?;
        unsafe {
            self.slots[slot_index]
                .staging
                .submit_to_view(&self.copy_stream, destination)?;
        }
        let copy_receipt = self.slots[slot_index].staging.submission_receipt()?;
        let dependency = self.slots[slot_index]
            .staging
            .order_completion_on(compute_stream)?;
        let submission_digest = Sha256::digest(
            [
                b"QWEN35-DFLASH-HOT-WINDOW-CONSUME-V1\0".as_slice(),
                &(step as u64).to_le_bytes(),
                &copy_receipt.submission_epoch().to_le_bytes(),
            ]
            .concat(),
        );
        let lease = self
            .control
            .submit_prefetch(
                slot_index,
                0,
                self.tile_bytes,
                copy_receipt.submission_epoch(),
                submission_digest.into(),
            )
            .map_err(|error| invalid(format!("DFlash hot-window prefetch: {error}")))?;
        self.control
            .mark_ready(lease)
            .map_err(|error| invalid(format!("DFlash hot-window ready: {error}")))?;
        self.control
            .begin_compute(lease)
            .map_err(|error| invalid(format!("DFlash hot-window compute: {error}")))?;
        let input = hidden_view;
        let output = CudaF32View::new(
            self.slots[slot_index].output.view(0, self.rows * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            self.rows,
        )?;
        let weight = CudaBf16WeightViewV1::new(
            self.slots[slot_index].device.view(0, self.tile_bytes)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
            self.rows,
            self.cols,
        )?;
        let matvec = CudaBf16F32MatvecArgsV1::bind(compute_stream, weight, input, output)?;
        matvec.submit_for_plan(compute_stream)?;
        let terminal_event = OwnedCudaEvent::create(compute_stream.device_ordinal())?;
        let terminal_receipt = compute_stream.submission_receipt()?;
        terminal_event.record_for_submission(compute_stream, terminal_receipt)?;
        let terminal_digest = Sha256::digest(
            [
                b"QWEN35-DFLASH-HOT-WINDOW-TERMINAL-V1\0".as_slice(),
                &terminal_receipt.submission_epoch().to_le_bytes(),
            ]
            .concat(),
        );
        self.slots[slot_index].active = Some(DFlashHotWindowCudaTicket {
            step,
            dependency,
            terminal_event,
            terminal_receipt,
            lease,
            terminal_digest: terminal_digest.into(),
        });
        self.submissions += 1;
        self.last_output_slot = Some(slot_index);
        self.last_expected = Some(
            self.selector
                .hidden_projection_gate_f32(hidden)
                .map_err(|error| invalid(format!("DFlash selector CPU oracle: {error}")))?,
        );
        Ok(())
    }

    fn finish(
        &mut self,
        compute_stream: &OwnedCudaStream,
    ) -> Result<Value, Qwen35MixedPrefixCudaError> {
        while self.slots.iter().any(|slot| slot.active.is_some()) {
            let slot_index = self
                .slots
                .iter()
                .enumerate()
                .filter_map(|(index, slot)| slot.active.as_ref().map(|ticket| (index, ticket.step)))
                .min_by_key(|(_, step)| *step)
                .map(|(index, _)| index)
                .ok_or_else(|| invalid("DFlash hot-window active-slot inventory"))?;
            self.retire_slot(slot_index, compute_stream)?;
        }
        self.control
            .finish_request()
            .map_err(|error| invalid(format!("DFlash hot-window finish: {error}")))?;
        let mut bytes = vec![0_u8; self.rows * 4];
        let output_slot = self
            .last_output_slot
            .ok_or_else(|| invalid("DFlash hot-window output slot"))?;
        self.slots[output_slot]
            .output
            .copy_to_host_sync(compute_stream, &mut bytes)?;
        let mut max_abs = 0.0_f32;
        let mut max_relative = 0.0_f32;
        const MAX_ABS_ERROR: f32 = 1.0e-4;
        const MAX_RELATIVE_ERROR: f32 = 3.0e-5;
        let mut finite = true;
        if let Some(expected) = self.last_expected.as_ref() {
            for (index, expected_value) in expected.iter().enumerate() {
                let actual = f32::from_le_bytes(
                    bytes[index * 4..index * 4 + 4]
                        .try_into()
                        .expect("DFlash output word"),
                );
                finite &= actual.is_finite() && expected_value.is_finite();
                let abs = (actual - expected_value).abs();
                max_abs = max_abs.max(abs);
                max_relative = max_relative.max(abs / expected_value.abs().max(1.0e-12));
            }
        }
        Ok(json!({
            "enabled": true,
            "component_consumed": self.submissions != 0,
            "submissions": self.submissions,
            "h2d_bytes": self.tile_bytes.saturating_mul(self.submissions),
            "source_tier": "pageable_ram",
            "transport_path": "RAM->pinned-H2D->HBM-slot->compute-stream",
            "slot_count": self.slots.len(),
            "weight_tile_bytes": self.tile_bytes,
            "source_dtype": match self.selector_source_dtype {
                DFlashDType::Bf16 => "BF16",
                DFlashDType::F16 => "F16",
            },
            "compute_dtype": "BF16",
            "wire_conversion": self.selector_source_dtype == DFlashDType::F16,
            "selector_rows": self.rows,
            "selector_cols": self.cols,
            "numerical_ab": {
                "last_step_checked": self.last_expected.is_some(),
                "finite": finite,
                "max_abs_error": max_abs,
                "max_relative_error": max_relative,
                "max_abs_error_threshold": MAX_ABS_ERROR,
                "max_relative_error_threshold": MAX_RELATIVE_ERROR,
                "pass": self.last_expected.is_some()
                    && finite
                    && max_abs <= MAX_ABS_ERROR
                    && max_relative <= MAX_RELATIVE_ERROR,
            },
            "claim_ceiling": "component_only_target_hidden_projection_consumer; full_dflash_backbone_not_connected"
        }))
    }

    fn abort(
        &mut self,
        compute_stream: &OwnedCudaStream,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        self.copy_stream.synchronize()?;
        compute_stream.synchronize()?;
        self.control.poison();
        for slot in &mut self.slots {
            slot.active = None;
        }
        Ok(())
    }
}

/// Resident Q8_0 selector consumer used by the full-chain diagnostic.
///
/// Unlike the BF16 hot-window path, this owner copies the complete Q8 GGUF
/// once after the prefill handoff and never performs a per-token RAM upload.
/// The target hidden projection is the first real CUDA consumer; the rest of
/// the DFlash draft backbone remains intentionally outside this component.
struct DFlashQ8ResidentCudaConsumer {
    q8_owner: OwnedCudaBuffer,
    weight: CudaQ8_0WeightViewV1,
    output: OwnedCudaBuffer,
    tensor: augur_model_qwen35::GgufTensor,
    payload: Vec<u8>,
    file_bytes: usize,
    file_sha256: String,
    tensor_offset: usize,
    tensor_bytes: usize,
    rows: usize,
    cols: usize,
    submissions: usize,
    last_step: Option<usize>,
    last_expected: Option<Vec<f32>>,
    terminal: Option<(OwnedCudaEvent, CudaStreamSubmissionReceipt)>,
}

impl DFlashQ8ResidentCudaConsumer {
    fn new(
        path: &std::path::Path,
        device: i32,
        stream: &OwnedCudaStream,
    ) -> Result<Self, Qwen35MixedPrefixCudaError> {
        let manifest = GgufManifest::inspect(
            path,
            GgufAuditOptions {
                hash_file: false,
                ..Default::default()
            },
        )?;
        let tensor = manifest
            .tensor(Q8_SELECTOR_TENSOR)
            .ok_or_else(|| invalid("Q8 selector_hidden.weight missing"))?
            .clone();
        if tensor.ggml_type != GGML_TYPE_Q8_0
            || tensor.dimensions != vec![Q8_SELECTOR_COLS as u64, Q8_SELECTOR_ROWS as u64]
        {
            return Err(invalid("Q8 selector_hidden.weight must be Q8_0 [5120,256]"));
        }
        let file = std::fs::read(path).map_err(Qwen35Error::from)?;
        if file.len() as u64 != manifest.file_size {
            return Err(invalid("Q8 GGUF extent changed after inspection"));
        }
        let mut reader = manifest.open_tensor_reader(path)?;
        let payload = reader.read_tensor_bytes(Q8_SELECTOR_TENSOR)?;
        let tensor_offset = usize::try_from(tensor.absolute_offset)
            .map_err(|_| invalid("Q8 selector tensor offset overflow"))?;
        let tensor_bytes = usize::try_from(tensor.byte_len)
            .map_err(|_| invalid("Q8 selector tensor extent overflow"))?;
        if tensor_offset
            .checked_add(tensor_bytes)
            .is_none_or(|end| end > file.len())
            || payload.len() != tensor_bytes
        {
            return Err(invalid("Q8 selector tensor payload extent"));
        }
        let q8_owner = OwnedCudaBuffer::allocate(device, file.len())?;
        q8_owner.copy_from_host_sync(stream, &file)?;
        stream.synchronize()?;
        let weight = CudaQ8_0WeightViewV1::new(
            q8_owner.view(tensor_offset, tensor_bytes)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            Q8_SELECTOR_ROWS,
            Q8_SELECTOR_COLS,
        )?;
        let output = OwnedCudaBuffer::allocate(device, Q8_SELECTOR_ROWS * 4)?;
        Ok(Self {
            q8_owner,
            weight,
            output,
            tensor,
            payload,
            file_bytes: file.len(),
            file_sha256: sha256_hex(&file),
            tensor_offset,
            tensor_bytes,
            rows: Q8_SELECTOR_ROWS,
            cols: Q8_SELECTOR_COLS,
            submissions: 0,
            last_step: None,
            last_expected: None,
            terminal: None,
        })
    }

    fn retire(&mut self, stream: &OwnedCudaStream) -> Result<(), Qwen35MixedPrefixCudaError> {
        let Some((event, receipt)) = self.terminal.take() else {
            return Ok(());
        };
        loop {
            if stream.query_event_completion(&event, receipt)?.is_some() {
                break;
            }
            std::hint::spin_loop();
        }
        Ok(())
    }

    fn consume(
        &mut self,
        step: usize,
        hidden: &[f32],
        hidden_view: CudaF32View,
        stream: &OwnedCudaStream,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if hidden.len() != self.cols || hidden_view.element_count() != self.cols {
            return Err(invalid("Q8 DFlash selector hidden input geometry"));
        }
        self.retire(stream)?;
        let output = CudaF32View::new(
            self.output.view(0, self.rows * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            self.rows,
        )?;
        let matvec = CudaQ8_0F32MatvecArgsV1::bind(stream, self.weight, hidden_view, output)?;
        matvec.submit_for_plan(stream)?;
        let terminal = OwnedCudaEvent::create(stream.device_ordinal())?;
        let receipt = stream.submission_receipt()?;
        terminal.record_for_submission(stream, receipt)?;
        self.terminal = Some((terminal, receipt));
        self.last_expected = Some(
            quantized_matvec(&self.tensor, &self.payload, hidden)
                .map_err(|error| invalid(format!("Q8 selector CPU oracle: {error}")))?,
        );
        self.last_step = Some(step);
        self.submissions += 1;
        Ok(())
    }

    fn finish(&mut self, stream: &OwnedCudaStream) -> Result<Value, Qwen35MixedPrefixCudaError> {
        self.retire(stream)?;
        let mut bytes = vec![0_u8; self.rows * 4];
        self.output.copy_to_host_sync(stream, &mut bytes)?;
        let actual = bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().expect("Q8 output word")))
            .collect::<Vec<_>>();
        let expected = self
            .last_expected
            .as_ref()
            .ok_or_else(|| invalid("Q8 selector output missing"))?;
        Ok(json!({
            "enabled": true,
            "component_consumed": self.submissions != 0,
            "submissions": self.submissions,
            "last_step": self.last_step,
            "transport_path": "one-time-GGUF-H2D->resident-HBM->Q8_0-f32-matvec",
            "q8_full_file_device_resident": true,
            "q8_file_bytes": self.file_bytes,
            "q8_file_sha256": self.file_sha256,
            "selector_tensor": Q8_SELECTOR_TENSOR,
            "selector_tensor_offset": self.tensor_offset,
            "selector_tensor_bytes": self.tensor_bytes,
            "selector_rows": self.rows,
            "selector_cols": self.cols,
            "initial_h2d_bytes": self.file_bytes,
            "per_token_h2d_bytes": 0,
            "numerical_ab": numerical("q8_selector", &actual, expected),
            "claim_ceiling": "component_only_target_hidden_projection_consumer; full_dflash_backbone_not_connected"
        }))
    }

    fn abort(&mut self, stream: &OwnedCudaStream) -> Result<(), Qwen35MixedPrefixCudaError> {
        stream.synchronize()?;
        self.terminal = None;
        Ok(())
    }
}

enum PrefixState {
    Linear { history: Vec<u8>, state: Vec<u8> },
    Full { key: Vec<u8>, value: Vec<u8> },
}

struct LayerOracle {
    input_hidden: Vec<f32>,
    hidden: Vec<f32>,
    query: Option<Vec<f32>>,
    attention: Option<Vec<f32>>,
    current_key: Vec<u8>,
    current_value: Vec<u8>,
    recurrent: Option<Vec<f32>>,
    convolution: Option<Vec<f32>>,
    gdn_stages: Option<GdnStageAudit>,
}

#[derive(Default)]
struct GdnStageAudit {
    f32: BTreeMap<String, Vec<f32>>,
    f16: BTreeMap<String, Vec<u8>>,
}

impl GdnStageAudit {
    fn read(
        &mut self,
        name: &str,
        stream: &OwnedCudaStream,
        terminal: &OwnedCudaEvent,
        buffer: &OwnedCudaBuffer,
        count: usize,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        let receipt = complete(stream, terminal)?;
        self.f32
            .insert(name.into(), values(stream, buffer, count, receipt)?);
        Ok(())
    }

    fn staging(
        &mut self,
        name: &str,
        stream: &OwnedCudaStream,
        terminal: &OwnedCudaEvent,
        workspace: &PrefillWorkspace,
        count: usize,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        let receipt = complete(stream, terminal)?;
        self.f16.insert(
            name.into(),
            workspace
                .cublas
                .snapshot_input_f16_diagnostic(stream, count, receipt)?,
        );
        Ok(())
    }

    fn compare(&self, expected: &Self) -> Result<Value, Qwen35MixedPrefixCudaError> {
        if !self.f32.keys().eq(expected.f32.keys()) || !self.f16.keys().eq(expected.f16.keys()) {
            return Err(invalid("GDN stage inventory mismatch"));
        }
        let f32: BTreeMap<_, _> = self
            .f32
            .iter()
            .map(|(name, actual)| {
                let reference = &expected.f32[name];
                (
                    name,
                    json!({
                        "metrics": numerical(name, actual, reference),
                        "actual_sha256": sha256_f32_bits(actual),
                        "expected_sha256": sha256_f32_bits(reference),
                        "actual": actual, "expected": reference,
                    }),
                )
            })
            .collect();
        let f16: BTreeMap<_, _> = self
            .f16
            .iter()
            .map(|(name, actual)| {
                let reference = &expected.f16[name];
                let words = |bytes: &[u8]| {
                    bytes
                        .chunks_exact(2)
                        .map(|v| u16::from_le_bytes([v[0], v[1]]))
                        .collect::<Vec<_>>()
                };
                (
                    name,
                    json!({
                        "byte_diff": byte_diff(actual, reference),
                        "actual_u16": words(actual), "expected_u16": words(reference),
                    }),
                )
            })
            .collect();
        Ok(json!({"diagnostic_only": true, "oracle_injection": false,
            "submission_proof": "owner_epoch_bound", "f32": f32, "physical_f16_staging": f16}))
    }
}

/// Same submit order as PrefillLinearOps::submit_with_activation_policy(true),
/// with completion-bound observations before scratch is overwritten.
fn submit_gdn_stage_audit(
    ops: &PrefillLinearOps,
    stream: &OwnedCudaStream,
    terminal: &OwnedCudaEvent,
    w: &PrefillWorkspace,
) -> Result<GdnStageAudit, Qwen35MixedPrefixCudaError> {
    let mut a = GdnStageAudit::default();
    ops.attn_norm.submit(stream)?;
    a.read(
        "normalized",
        stream,
        terminal,
        &w.normalized,
        linear::HIDDEN,
    )?;
    let input = &ops.direct_input_projections;
    w.cublas.submit_f32(stream, &input.qkv)?;
    a.staging("normalized", stream, terminal, w, linear::HIDDEN)?;
    a.read("qkv", stream, terminal, &w.qkv, linear::QKV_ELEMENTS)?;
    w.cublas.submit_f32(stream, &input.z)?;
    a.read("z", stream, terminal, &w.z, linear::RECURRENT_ELEMENTS)?;
    w.cublas.submit_f32(stream, &input.alpha)?;
    a.read(
        "alpha_raw",
        stream,
        terminal,
        &w.alpha_raw,
        linear::VALUE_HEADS,
    )?;
    w.cublas.submit_f32(stream, &input.beta)?;
    a.read(
        "beta_raw",
        stream,
        terminal,
        &w.beta_raw,
        linear::VALUE_HEADS,
    )?;
    ops.submit_recurrence(stream)?;
    a.read(
        "conv_activated",
        stream,
        terminal,
        &w.activated,
        linear::QKV_ELEMENTS,
    )?;
    a.read(
        "recurrent",
        stream,
        terminal,
        &w.recurrent,
        linear::RECURRENT_ELEMENTS,
    )?;
    a.read(
        "gated_recurrent",
        stream,
        terminal,
        &w.attention_output,
        linear::RECURRENT_ELEMENTS,
    )?;
    let post = &ops.direct_post_attention;
    w.cublas.submit_f32(stream, &post.output)?;
    a.staging(
        "gated_recurrent",
        stream,
        terminal,
        w,
        linear::RECURRENT_ELEMENTS,
    )?;
    a.read(
        "output_projection",
        stream,
        terminal,
        &w.ffn_output,
        linear::HIDDEN,
    )?;
    post.attention_residual.submit_for_plan(stream)?;
    a.read(
        "attention_residual",
        stream,
        terminal,
        &w.attention_residual,
        linear::HIDDEN,
    )?;
    ops.post_norm.submit(stream)?;
    a.read("post_norm", stream, terminal, &w.post_norm, linear::HIDDEN)?;
    w.cublas.submit_f32(stream, &post.ffn_gate)?;
    a.staging("post_norm", stream, terminal, w, linear::HIDDEN)?;
    a.read(
        "ffn_gate",
        stream,
        terminal,
        &w.ffn_gate,
        linear::INTERMEDIATE,
    )?;
    w.cublas.submit_f32(stream, &post.ffn_up)?;
    a.read("ffn_up", stream, terminal, &w.ffn_up, linear::INTERMEDIATE)?;
    ops.silu_mul.submit_for_plan(stream)?;
    a.read(
        "ffn_activation",
        stream,
        terminal,
        &w.ffn_activation,
        linear::INTERMEDIATE,
    )?;
    w.cublas.submit_f32(stream, &post.ffn_down)?;
    a.staging("ffn_activation", stream, terminal, w, linear::INTERMEDIATE)?;
    a.read("ffn_down", stream, terminal, &w.ffn_output, linear::HIDDEN)?;
    ops.final_residual.submit_for_plan(stream)?;
    Ok(a)
}

fn complete(
    stream: &OwnedCudaStream,
    event: &OwnedCudaEvent,
) -> Result<CudaStreamSubmissionReceipt, Qwen35MixedPrefixCudaError> {
    let receipt = stream.submission_receipt()?;
    event.record_for_submission(stream, receipt)?;
    let (_, proof) = wait_tc_prefix_submission(stream, event, receipt)?;
    if !proof {
        return Err(invalid("full-chain owner/epoch incomplete"));
    }
    Ok(receipt)
}

fn raw(
    stream: &OwnedCudaStream,
    buffer: &OwnedCudaBuffer,
    offset: usize,
    len: usize,
    receipt: CudaStreamSubmissionReceipt,
) -> Result<Vec<u8>, Qwen35MixedPrefixCudaError> {
    let mut bytes = vec![0; len];
    let proof = buffer
        .copy_range_to_host_sync_with_submission_proof(stream, offset, &mut bytes, receipt)?;
    if !proof.covers_submission(receipt) || !proof.belongs_to_stream(stream)? {
        return Err(invalid("full-chain snapshot completion"));
    }
    Ok(bytes)
}

fn values(
    stream: &OwnedCudaStream,
    buffer: &OwnedCudaBuffer,
    count: usize,
    receipt: CudaStreamSubmissionReceipt,
) -> Result<Vec<f32>, Qwen35MixedPrefixCudaError> {
    read_f32_view(
        stream,
        prefill_f32_view(buffer, count)?.buffer_view(),
        receipt,
    )
}

fn v1_segment_probe(
    stream: &OwnedCudaStream,
    terminal: &OwnedCudaEvent,
    workspace: &PrefillWorkspace,
    plan: &FullPlan,
    prefix_positions: usize,
    capacity: usize,
    begin: usize,
    end: usize,
) -> Result<Value, Qwen35MixedPrefixCudaError> {
    if begin >= end || end > prefix_positions + 1 || end > capacity {
        return Err(invalid("V1 segment probe range"));
    }
    let query = prefill_f32_view(&workspace.activated, full::QUERY_VALUES)?;
    let attention = prefill_f32_view(&workspace.ffn_gate, full::QUERY_VALUES)?;
    let lse = prefill_f32_view(&workspace.tc_attention_lse, full::QUERY_HEADS)?;
    let views = CudaQwen35TcPrefixAttentionViewsV1 {
        query,
        key_cache_q8_0: plan.key_cache.view(0, plan.key_cache.len_bytes())?,
        value_cache_q8_0: plan.value_cache.view(0, plan.value_cache.len_bytes())?,
        output: attention,
        lse,
        workspace: workspace.tc_attention_workspace.views_for(1)?,
    };
    let operation = workspace.cublas.bind_tc_prefix_attention(
        stream,
        views,
        1,
        prefix_positions,
        capacity,
        workspace.tc_attention_workspace.layout().tile_positions(),
        begin..end,
    )?;
    // The SIMT reference kernel only initializes on key_begin == 0.  Reset
    // explicitly for every isolated segment so active/current probes do not
    // accidentally merge into the preceding diagnostic segment.
    let lse_reset = vec![f32::NEG_INFINITY; full::QUERY_HEADS];
    let output_reset = vec![0.0_f32; full::QUERY_VALUES];
    workspace
        .tc_attention_lse
        .copy_range_from_host_sync(stream, 0, &f32_bytes(&lse_reset))?;
    workspace
        .ffn_gate
        .copy_range_from_host_sync(stream, 0, &f32_bytes(&output_reset))?;
    workspace
        .cublas
        .submit_simt_prefix_attention(stream, &operation)?;
    let receipt = complete(stream, terminal)?;
    let lse_bytes = raw(
        stream,
        &workspace.tc_attention_lse,
        0,
        full::QUERY_HEADS * 4,
        receipt,
    )?;
    let output_bytes = raw(
        stream,
        &workspace.ffn_gate,
        0,
        full::QUERY_VALUES * 4,
        receipt,
    )?;
    let lse_values = lse_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("V1 tile LSE bytes")))
        .collect::<Vec<_>>();
    let output_values = output_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("V1 tile output bytes")))
        .collect::<Vec<_>>();
    Ok(json!({
        "diagnostic_only": true,
        "reference": "v1_simt_segment",
        "submission_proof": "owner_epoch_bound",
        "range_begin": begin,
        "range_end": end,
        "lse_f32": lse_values,
        "output_f32": output_values,
    }))
}

fn numerical(name: &str, actual: &[f32], expected: &[f32]) -> Value {
    let sample = trajectory_sample(name, actual, expected);
    let max = actual.iter().zip(expected).fold(0.0_f64, |m, (a, b)| {
        m.max((f64::from(*a) - f64::from(*b)).abs())
    });
    let pass = !actual.is_empty()
        && actual.len() == expected.len()
        && actual.iter().chain(expected).all(|v| v.is_finite())
        && max <= 0.02
        && sample.relative_l2_error.is_some_and(|e| e <= 0.015)
        && sample.cosine_similarity.is_some_and(|c| c >= 0.9999);
    json!({"pass": pass, "max_abs_error": max, "byte_exact": equal_f32_bits(actual, expected), "trajectory": sample})
}

fn relative_l2(value: &Value) -> Option<f64> {
    value
        .get("trajectory")
        .and_then(|trajectory| trajectory.get("relative_l2_error"))
        .and_then(Value::as_f64)
}

fn amplification(current: Option<f64>, previous: Option<f64>) -> Value {
    match (current, previous) {
        (Some(current), Some(previous)) if previous > 1.0e-12 => json!(current / previous),
        (Some(current), Some(_)) if current <= 1.0e-12 => json!(0.0),
        _ => Value::Null,
    }
}

fn byte_diff(actual: &[u8], expected: &[u8]) -> Value {
    let mismatch_count = actual.iter().zip(expected).filter(|(a, b)| a != b).count()
        + actual.len().abs_diff(expected.len());
    let first_mismatch = actual.iter().zip(expected).position(|(a, b)| a != b);
    json!({
        "byte_exact": actual == expected,
        "actual_len": actual.len(),
        "expected_len": expected.len(),
        "mismatch_count": mismatch_count,
        "first_mismatch": first_mismatch,
    })
}

fn top_ids(logits: &[f32]) -> Vec<usize> {
    let mut ids: Vec<_> = (0..logits.len()).collect();
    ids.sort_unstable_by(|a, b| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b)));
    ids.truncate(50);
    ids
}

fn admitted_prefix(prefix: usize) -> bool {
    prefix > ACTIVE && prefix <= MAX_PAGE_ALIGNED_PREFIX && prefix.is_multiple_of(PAGE)
}

fn validate_fault(
    config: &Qwen35FullChainActiveColdConfig,
    mode: u32,
) -> Result<(), Qwen35MixedPrefixCudaError> {
    let Some(kind) = config.fault_injection.as_deref() else {
        return Ok(());
    };
    if !matches!(
        kind,
        "cancel-before-submit"
            | "cancel-after-submit"
            | "missing-page"
            | "stale-generation"
            | "corrupt-page"
            | "partial-write"
    ) || mode != 3
        || config.inject_upload_corruption
        || config.fault_step >= config.suffix_tokens
        || config.fault_layer >= 64
        || config.fault_layer % 4 != 3
        || config.freeze_oracle_input_layer.is_some()
        || config.precise_gdn
    {
        return Err(invalid("full-chain fault selection invalid or conflicting"));
    }
    let layout = ring_step(config.prefix_positions, config.fault_step)?;
    if config.fault_page >= layout.cold_end.div_ceil(PAGE)
        || (kind == "stale-generation" && config.fault_step == 0)
    {
        return Err(invalid("fault page absent or no prior generation"));
    }
    Ok(())
}

fn expected_fault_rejection(kind: &str, error: &str) -> bool {
    let expected = match kind {
        "cancel-before-submit" | "cancel-after-submit" => "page lease request cancelled",
        "missing-page" => "ordered prefix missing, repeated or out-of-order page",
        "stale-generation" => "page lease stale generation, owner, identity or phase",
        "corrupt-page" | "partial-write" => "async cold page readback mismatch",
        _ => return false,
    };
    error.contains(expected)
}

/// Logical-to-physical map for one step of the fixed-size active KV ring.
/// Cold [0,cold_end), these spans, then the private current row must cover the
/// visible range exactly once, in absolute position order.
#[derive(Debug)]
struct RingStep {
    position: usize,
    cold_end: usize,
    head: usize,
    spans: Vec<(usize, usize, usize)>, // physical offset, length, absolute base
}

fn ring_step(prefix: usize, step: usize) -> Result<RingStep, Qwen35MixedPrefixCudaError> {
    let position = prefix
        .checked_add(step)
        .ok_or_else(|| invalid("ring position overflow"))?;
    let cold_end = position
        .checked_sub(ACTIVE)
        .ok_or_else(|| invalid("ring active extent"))?;
    if prefix <= ACTIVE || position >= u32::MAX as usize {
        return Err(invalid("ring absolute position admission"));
    }
    let head = step % ACTIVE;
    let mut spans = vec![(head, ACTIVE - head, cold_end)];
    if head != 0 {
        spans.push((0, head, cold_end + ACTIVE - head));
    }
    Ok(RingStep {
        position,
        cold_end,
        head,
        spans,
    })
}

fn validate_loader_options(
    config: &Qwen35FullChainActiveColdConfig,
) -> Result<(), Qwen35MixedPrefixCudaError> {
    if config.precise_gdn {
        return Err(invalid(
            "precise-GDN recurrence is not available in the current producer; no silent fallback",
        ));
    }
    if config.dflash_artifact_path.is_some() != config.dflash_config_path.is_some() {
        return Err(invalid("DFlash artifact/config must be supplied together"));
    }
    if config.dflash_q8_gguf_path.is_some()
        && (config.dflash_artifact_path.is_some()
            || config.dflash_config_path.is_some()
            || config.dflash_hot_window_only
            || config.dflash_allow_f16
            || config.dflash_source_ram)
    {
        return Err(invalid(
            "Q8 resident DFlash mode is mutually exclusive with safetensors/hot-window mode",
        ));
    }
    if config.dflash_artifact_path.is_none() && config.dflash_allow_f16 {
        return Err(invalid(
            "DFlash F16 opt-in requires an artifact/config pair",
        ));
    }
    if config.dflash_source_ram && config.dflash_artifact_path.is_none() {
        return Err(invalid(
            "DFlash RAM source requires an artifact/config pair",
        ));
    }
    if config.dflash_source_ram && config.dflash_ram_budget_bytes == 0 {
        return Err(invalid("DFlash RAM source budget must be non-zero"));
    }
    if config.dflash_hot_window_only {
        if config.dflash_artifact_path.is_none() || !config.dflash_source_ram {
            return Err(invalid(
                "DFlash hot-window mode requires an artifact/config pair and RAM source",
            ));
        }
    }
    if config.dflash_artifact_path.is_some()
        && !(1..=64 * 1024 * 1024).contains(&config.dflash_chunk_bytes)
    {
        return Err(invalid("DFlash chunk size outside bounded lane limit"));
    }
    Ok(())
}

fn validate_capture_options(
    config: &Qwen35FullChainActiveColdConfig,
) -> Result<(), Qwen35MixedPrefixCudaError> {
    if config.capture_target_only
        && (config.target_feature_export_path.is_none()
            || config.dflash_artifact_path.is_some()
            || config.dflash_config_path.is_some()
            || config.dflash_q8_gguf_path.is_some()
            || config.dflash_source_ram
            || config.dflash_hot_window_only
            || config.inject_upload_corruption
            || config.fault_injection.is_some()
            || config.freeze_oracle_input_layer.is_some()
            || config.audit_partial_state
            || config.audit_gdn_stages
            || config.precise_gdn
            || config.segment_v1_match)
    {
        return Err(invalid(
            "target-only capture requires a feature export and no candidate/fault/audit options",
        ));
    }
    Ok(())
}

pub fn run_qwen35_full_chain_active_cold_cuda(
    config: &Qwen35FullChainActiveColdConfig,
) -> Result<Value, Qwen35MixedPrefixCudaError> {
    validate_capture_options(config)?;
    validate_loader_options(config)?;
    let segment_mode = if config.segment_v1_match {
        2
    } else {
        config.segment_mode
    };
    if segment_mode > 3 || (config.segment_v1_match && config.segment_mode != 1) {
        return Err(invalid("unsupported or conflicting segment mode"));
    }
    if config.audit_gdn_stages
        && (!config.direct_post_attention
            || config.precise_gdn
            || config.freeze_oracle_input_layer.is_some())
    {
        return Err(invalid(
            "GDN stage audit requires unmodified direct activation and no oracle injection",
        ));
    }
    if !admitted_prefix(config.prefix_positions) {
        return Err(invalid(
            "full-chain prefix must be page-aligned, >2048 and <=261632",
        ));
    }
    let dflash_ram_resident = match (&config.dflash_artifact_path, &config.dflash_config_path) {
        (Some(artifact_path), Some(config_path)) if config.dflash_source_ram => {
            let inspection =
                inspect_dflash_artifact(artifact_path, config_path, config.dflash_allow_f16)
                    .map_err(|error| invalid(format!("DFlash artifact inspection: {error}")))?;
            Some(
                DFlashRamResident::load_with_budget(&inspection, config.dflash_ram_budget_bytes)
                    .map_err(|error| invalid(format!("DFlash RAM resident load: {error}")))?,
            )
        }
        _ => None,
    };
    let prefix = config.prefix_positions;
    let suffix_tokens = config.suffix_tokens;
    if !(1..=2048).contains(&suffix_tokens) {
        return Err(invalid("full-chain suffix must be in 1..=2048"));
    }
    if prefix
        .checked_add(suffix_tokens)
        .is_none_or(|capacity| capacity > MODEL_CONTEXT)
    {
        return Err(invalid(
            "full-chain prefix plus suffix exceeds model context 262144",
        ));
    }
    if suffix_tokens > 1
        && (segment_mode != 3
            || config.precise_gdn
            || config.freeze_oracle_input_layer.is_some()
            || config.audit_partial_state
            || config.audit_gdn_stages)
    {
        return Err(invalid(
            "multi-suffix requires ordered mode3 without diagnostic interventions",
        ));
    }
    let capacity = prefix + suffix_tokens;
    validate_fault(config, segment_mode)?;
    let cold_end = prefix - ACTIVE;
    let config_bytes = read_stable_regular_file(&config.config_path, "Qwen config")?;
    let model = Qwen35Config::parse(&config_bytes)?;
    model.validate_exact_profile()?;
    let manifest = GgufManifest::inspect(&config.gguf_path, GgufAuditOptions::default())?;
    let schema = Qwen35TensorSchema::validate_manifest_allowing_mtp(&model, &manifest)?;
    if schema.actual_tensor_count < 1187 || model.num_hidden_layers != 64 {
        return Err(invalid("full-chain model profile"));
    }
    let device = cuda_device_properties(0)?;
    if device.compute_capability.major != 7 || device.compute_capability.minor != 0 {
        return Err(invalid("full-chain smoke requires SM70"));
    }
    let binary_path = std::env::current_exe().map_err(Qwen35Error::from)?;
    let binary_sha256 = sha256_hex(&std::fs::read(&binary_path).map_err(Qwen35Error::from)?);
    let allocation_start = cuda_device_allocation_snapshot();
    let stream = OwnedCudaStream::create(0)?;
    let terminal = OwnedCudaEvent::create(0)?;
    let mut reader = manifest.open_tensor_reader(&config.gguf_path)?;
    let embedding = manifest
        .tensor(linear::EMBEDDING)
        .ok_or_else(|| invalid("embedding"))?
        .clone();
    let (token_ids, input) =
        read_tc_prefix_embedding_chunk(&mut reader, &embedding, 0, capacity, model.vocab_size - 1)?;
    let materials = load_layer_materials(&mut reader, &manifest, &model, 64)?;
    let zero = linear::ProjectedToken {
        qkv: vec![0.; linear::QKV_ELEMENTS],
        z: vec![0.; linear::RECURRENT_ELEMENTS],
        decay_gate: vec![0.; linear::VALUE_HEADS],
        beta: vec![0.; linear::VALUE_HEADS],
    };
    let mut plans = Vec::with_capacity(64);
    for material in &materials {
        plans.push(match material {
            LayerMaterial::Linear(m) => DevicePlan::Linear(build_linear_plan(
                &stream,
                m,
                &input[..linear::HIDDEN],
                &zero,
                true,
                None,
            )?),
            LayerMaterial::Full(m) => {
                DevicePlan::Full(build_full_plan(&stream, m, &model, capacity, 1, None)?)
            }
        });
    }
    if plans
        .iter()
        .filter(|p| matches!(p, DevicePlan::Linear(_)))
        .count()
        != 48
    {
        return Err(invalid("full-chain 48/16 partition"));
    }
    drop(materials);
    let head_material = load_output_head_material(&mut reader, &manifest, &model)?;
    let head_input = alloc_f32(0, linear::HIDDEN)?;
    let head = build_device_output_head_plan(
        &stream,
        &head_material,
        head_input.view(0, linear::HIDDEN * 4)?,
        model.hidden_size,
        model.vocab_size,
        model.rms_norm_eps as f32,
    )?;
    drop(head_material);
    eprintln!("QWEN35_FULL_CHAIN stage=prefill prefix={prefix} layers=64");
    let before_workspace = cuda_device_allocation_snapshot();
    let workspace = PrefillWorkspace::new_with_rope_capacity(&stream, CHUNK, capacity)?;
    workspace.upload_rope_tables(&stream, capacity, 0, model.rope_theta as f32)?;
    let workspace_bytes = cuda_device_allocation_snapshot()
        .live_bytes
        .checked_sub(before_workspace.live_bytes)
        .ok_or_else(|| invalid("workspace accounting"))?;
    let full_cache_capacity_bytes = (32_u64)
        .checked_mul((capacity.saturating_sub(1) * ROW) as u64)
        .ok_or_else(|| invalid("full-cache lifecycle accounting"))?;
    let mut request_digest = Sha256::new();
    request_digest.update(b"QWEN35-PREFILL-DFLASH-HANDOFF-V1\0");
    request_digest.update(config.source_commit.as_bytes());
    request_digest.update((prefix as u64).to_le_bytes());
    request_digest.update((suffix_tokens as u64).to_le_bytes());
    request_digest.update((capacity as u64).to_le_bytes());
    let request_sha256: [u8; 32] = request_digest.finalize().into();
    let mut dflash_lifecycle = DFlashLifecycle::new(DFlashLifecycleBudget {
        total_hbm_bytes: device.total_global_memory_bytes,
        safety_reserve_bytes: 256 * 1024 * 1024,
    });
    dflash_lifecycle.begin_prefill(
        DFlashRequestIdentity {
            request_sha256,
            request_generation: 1,
        },
        1,
        full_cache_capacity_bytes,
        workspace_bytes,
    )?;
    let schedule = build_prefill_chunk_schedule(
        &workspace,
        &plans,
        &stream,
        prefix,
        CHUNK,
        model.rms_norm_eps as f32,
    )?;
    let start = OwnedCudaEvent::create(0)?;
    let stop = OwnedCudaEvent::create(0)?;
    let prefill = run_prefill_chunk_schedule(
        &mut plans,
        &workspace,
        &schedule,
        false,
        config.direct_post_attention,
        &stream,
        None,
        &start,
        &stop,
        &terminal,
        &input[..prefix * linear::HIDDEN],
        false,
        false,
    )?;
    let receipt = complete(&stream, &terminal)?;
    let mut submission_digest = Sha256::new();
    submission_digest.update(b"QWEN35-PREFILL-DFLASH-SUBMISSION-V1\0");
    submission_digest.update(request_sha256);
    submission_digest.update(receipt.submission_epoch().to_le_bytes());
    let dflash_submission = dflash_lifecycle.record_prefill_submission_at_epoch(
        submission_digest.finalize().into(),
        receipt.submission_epoch(),
    )?;
    let mut terminal_digest = Sha256::new();
    terminal_digest.update(b"QWEN35-PREFILL-DFLASH-TERMINAL-V1\0");
    terminal_digest.update(dflash_submission.receipt_digest);
    terminal_digest.update(receipt.submission_epoch().to_le_bytes());
    let dflash_terminal = DFlashCompletionProof {
        owner_generation: dflash_submission.owner_generation,
        request_sha256,
        request_generation: 1,
        covering_submission_epoch: receipt.submission_epoch(),
        terminal_digest: terminal_digest.finalize().into(),
    };
    dflash_lifecycle.mark_prefill_terminal(dflash_terminal)?;
    if !prefill.owner_epoch_proof {
        return Err(invalid("prefix completion"));
    }
    let mut prefix_states = Vec::new();
    for plan in plans.iter().filter(|_| !config.capture_target_only) {
        prefix_states.push(match plan {
            DevicePlan::Linear(p) => PrefixState::Linear {
                history: raw(&stream, &p.history, 0, p.history.len_bytes(), receipt)?,
                state: raw(&stream, &p.state, 0, p.state.len_bytes(), receipt)?,
            },
            DevicePlan::Full(p) => PrefixState::Full {
                key: raw(&stream, &p.key_cache, 0, prefix * ROW, receipt)?,
                value: raw(&stream, &p.value_cache, 0, prefix * ROW, receipt)?,
            },
        });
    }
    drop(schedule);
    let before_release = cuda_device_allocation_snapshot();
    drop(workspace);
    let after_release = cuda_device_allocation_snapshot();
    let released_workspace_bytes = before_release
        .live_bytes
        .checked_sub(after_release.live_bytes)
        .ok_or_else(|| invalid("workspace release accounting"))?;
    if released_workspace_bytes != workspace_bytes
        || after_release.live_bytes != before_workspace.live_bytes
    {
        return Err(invalid("prefill workspace did not release completely"));
    }
    eprintln!(
        "QWEN35_FULL_CHAIN stage=contiguous_oracle workspace_released={released_workspace_bytes}"
    );
    let oracle_workspace = PrefillWorkspace::new_with_rope_capacity(&stream, 1, capacity)?;
    oracle_workspace.upload_rope_tables(&stream, capacity, 0, model.rope_theta as f32)?;
    let mut oracle_steps = if config.capture_target_only {
        Vec::new()
    } else {
        Vec::with_capacity(suffix_tokens)
    };
    let mut oracle_logits_steps = if config.capture_target_only {
        Vec::new()
    } else {
        Vec::with_capacity(suffix_tokens)
    };
    let mut capture_feature_values = if config.capture_target_only {
        Vec::with_capacity(suffix_tokens * 5 * DFLASH_HIDDEN_ELEMENTS)
    } else {
        Vec::new()
    };
    let mut capture_final_hidden_values = if config.capture_target_only {
        Vec::with_capacity(suffix_tokens * DFLASH_HIDDEN_ELEMENTS)
    } else {
        Vec::new()
    };
    let mut capture_oracle_top50 = if config.capture_target_only {
        Vec::with_capacity(suffix_tokens)
    } else {
        Vec::new()
    };
    let mut oracle_next_input = None::<Vec<f32>>;
    let mut oracle_next_token = None::<usize>;
    let mut oracle_input_tokens = Vec::with_capacity(suffix_tokens);
    let mut v1_segment_audit = Value::Null;
    // One sequential oracle run: never restore prefix state between tokens.
    for step in 0..suffix_tokens {
        let position = prefix + step;
        let suffix = if config.free_running && step > 0 {
            oracle_next_input
                .as_deref()
                .ok_or_else(|| invalid("free-running oracle input missing"))?
        } else {
            &input[position * linear::HIDDEN..(position + 1) * linear::HIDDEN]
        };
        oracle_input_tokens.push(if config.free_running && step > 0 {
            oracle_next_token.ok_or_else(|| invalid("free-running oracle token missing"))?
        } else {
            token_ids[position]
        });
        oracle_workspace
            .hidden_a
            .copy_from_host_sync(&stream, &f32_bytes(suffix))?;
        let mut oracle = Vec::with_capacity(64);
        let mut oracle_input_hidden = suffix.to_vec();
        for (layer, plan) in plans.iter().enumerate() {
            let mut gdn_stages = None;
            let mut query = None;
            let mut attention = None;
            let mut current_key = Vec::new();
            let mut current_value = Vec::new();
            match plan {
                DevicePlan::Linear(p) => {
                    let ops = oracle_workspace.build_linear_ops(
                        &stream,
                        p,
                        layer,
                        1,
                        model.rms_norm_eps as f32,
                    )?;
                    if config.audit_gdn_stages && (4..=6).contains(&layer) {
                        gdn_stages = Some(submit_gdn_stage_audit(
                            &ops,
                            &stream,
                            &terminal,
                            &oracle_workspace,
                        )?);
                    } else {
                        ops.submit_with_activation_policy(
                            &stream,
                            &oracle_workspace.cublas,
                            config.direct_post_attention,
                        )?;
                    }
                }
                DevicePlan::Full(p) => {
                    let ops = oracle_workspace.build_full_ops(
                        &stream,
                        p,
                        layer,
                        1,
                        position,
                        model.rms_norm_eps as f32,
                    )?;
                    ops.submit_projection_prefix(&stream, &oracle_workspace.cublas)?;
                    // Original contiguous KV8 operator is the independent attention oracle.
                    ops.attention.submit(&stream)?;
                    if !config.capture_target_only {
                        let r = complete(&stream, &terminal)?;
                        query = Some(values(
                            &stream,
                            &oracle_workspace.activated,
                            full::QUERY_VALUES,
                            r,
                        )?);
                        attention = Some(values(
                            &stream,
                            &oracle_workspace.ffn_gate,
                            full::QUERY_VALUES,
                            r,
                        )?);
                        current_key = raw(&stream, &p.key_cache, position * ROW, ROW, r)?;
                        current_value = raw(&stream, &p.value_cache, position * ROW, ROW, r)?;
                    }
                    if config.audit_partial_state && layer == 3 {
                        let cold = v1_segment_probe(
                            &stream,
                            &terminal,
                            &oracle_workspace,
                            p,
                            prefix,
                            capacity,
                            0,
                            cold_end,
                        )?;
                        let active = v1_segment_probe(
                            &stream,
                            &terminal,
                            &oracle_workspace,
                            p,
                            prefix,
                            capacity,
                            cold_end,
                            prefix,
                        )?;
                        let current = v1_segment_probe(
                            &stream,
                            &terminal,
                            &oracle_workspace,
                            p,
                            prefix,
                            capacity,
                            prefix,
                            prefix + 1,
                        )?;
                        v1_segment_audit = json!({
                            "diagnostic_only": true,
                            "layer": layer,
                            "cold_pages": cold,
                            "active_window": active,
                            "current_row": current,
                        });
                        // The segment probes reuse the oracle attention output;
                        // restore the original contiguous attention before FFN.
                        ops.attention.submit(&stream)?;
                        let _ = complete(&stream, &terminal)?;
                    }
                    ops.submit_post_attention_with_activation_policy(
                        &stream,
                        &oracle_workspace.cublas,
                        config.direct_post_attention,
                    )?;
                }
            }
            let r = complete(&stream, &terminal)?;
            let hidden = read_f32_view(
                &stream,
                oracle_workspace.hidden_output(layer, 1)?.buffer_view(),
                r,
            )?;
            let (recurrent, convolution) = match plan {
                DevicePlan::Linear(p) if !config.capture_target_only => (
                    Some(read_f32_buffer(&stream, &p.state, r)?),
                    Some(read_f32_buffer(&stream, &p.history, r)?),
                ),
                _ => (None, None),
            };
            if config.capture_target_only {
                if hidden.len() != DFLASH_HIDDEN_ELEMENTS || hidden.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("capture target hidden geometry/non-finite"));
                }
                if [5_usize, 19, 33, 47, 61].contains(&layer) {
                    if hidden.len() != DFLASH_HIDDEN_ELEMENTS {
                        return Err(invalid("capture target feature hidden geometry"));
                    }
                    capture_feature_values.extend_from_slice(&hidden);
                }
                if layer == 63 {
                    capture_final_hidden_values.extend_from_slice(&hidden);
                }
            } else {
                oracle.push(LayerOracle {
                    gdn_stages,
                    input_hidden: oracle_input_hidden,
                    hidden: hidden.clone(),
                    query,
                    attention,
                    current_key,
                    current_value,
                    recurrent,
                    convolution,
                });
            }
            oracle_input_hidden = hidden;
        }
        head_input.copy_from_host_sync(&stream, &f32_bytes(&oracle_input_hidden))?;
        head.submit(&stream)?;
        let r = complete(&stream, &terminal)?;
        let oracle_logits = read_f32_buffer(&stream, &head.logits, r)?;
        if config.capture_target_only
            && (oracle_logits.len() != model.vocab_size
                || oracle_logits.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid("capture target logits geometry/non-finite"));
        }
        if config.free_running && step + 1 < suffix_tokens {
            let next_token = top_ids(&oracle_logits)
                .first()
                .copied()
                .ok_or_else(|| invalid("free-running oracle next token"))?;
            oracle_next_token = Some(next_token);
            oracle_next_input = Some(linear::read_embedding_row(
                &mut reader,
                &embedding,
                next_token,
            )?);
        }
        if config.capture_target_only {
            capture_oracle_top50.push(top_ids(&oracle_logits));
        } else {
            oracle_logits_steps.push(oracle_logits);
            oracle_steps.push(oracle);
        }
        eprintln!("QWEN35_FULL_CHAIN oracle_step={step} position={position} terminal=true");
    }
    let capture_feature_payload_bytes = capture_feature_values.len() * 4;
    let capture_final_hidden_payload_bytes = capture_final_hidden_values.len() * 4;
    if config.capture_target_only
        && (capture_feature_values.len() != suffix_tokens * 5 * DFLASH_HIDDEN_ELEMENTS
            || capture_final_hidden_values.len() != suffix_tokens * DFLASH_HIDDEN_ELEMENTS
            || capture_oracle_top50.len() != suffix_tokens)
    {
        return Err(invalid("target capture output extent"));
    }
    let target_layer_ids = [5_usize, 19, 33, 47, 61];
    let target_feature_export = if let Some(path) = config.target_feature_export_path.as_ref() {
        let values = if config.capture_target_only {
            std::mem::take(&mut capture_feature_values)
        } else {
            let mut values = Vec::with_capacity(
                oracle_steps.len() * target_layer_ids.len() * DFLASH_HIDDEN_ELEMENTS,
            );
            for oracle in &oracle_steps {
                for layer in target_layer_ids {
                    let row = oracle
                        .get(layer)
                        .ok_or_else(|| invalid("target feature export layer"))?;
                    if row.hidden.len() != DFLASH_HIDDEN_ELEMENTS {
                        return Err(invalid("target feature export hidden geometry"));
                    }
                    values.extend_from_slice(&row.hidden);
                }
            }
            values
        };
        Some(json!({
            "path": path,
            "elements": values.len(),
            "rows_per_step": target_layer_ids.len(),
            "row_elements": DFLASH_HIDDEN_ELEMENTS,
            "layer_ids": target_layer_ids,
            "sha256": write_f32_export(path, &values)?,
            "source": "same_target_oracle_hidden_rows",
        }))
    } else {
        None
    };
    let target_final_hidden_export =
        if let Some(path) = config.target_final_hidden_export_path.as_ref() {
            let values = if config.capture_target_only {
                std::mem::take(&mut capture_final_hidden_values)
            } else {
                let mut values = Vec::with_capacity(oracle_steps.len() * DFLASH_HIDDEN_ELEMENTS);
                for oracle in &oracle_steps {
                    let row = oracle
                        .get(63)
                        .ok_or_else(|| invalid("target final hidden export layer"))?;
                    if row.hidden.len() != DFLASH_HIDDEN_ELEMENTS {
                        return Err(invalid("target final hidden export hidden geometry"));
                    }
                    values.extend_from_slice(&row.hidden);
                }
                values
            };
            Some(json!({
                "path": path,
                "elements": values.len(),
                "rows_per_step": 1,
                "row_elements": DFLASH_HIDDEN_ELEMENTS,
                "layer_id": 63,
                "sha256": write_f32_export(path, &values)?,
                "source": "same_target_oracle_final_hidden",
            }))
        } else {
            None
        };
    drop(oracle_workspace);

    if config.capture_target_only {
        drop(head);
        drop(head_input);
        drop(plans);
        let final_allocations = cuda_device_allocation_snapshot();
        if final_allocations.live_bytes != allocation_start.live_bytes
            || final_allocations.live_allocations != allocation_start.live_allocations
        {
            return Err(invalid("target capture device cleanup did not close"));
        }
        let report = json!({
            "schema": "crystal-quant.qwen35.full-chain-target-capture.component.v1",
            "status": "PASS_COMPONENT_FULL_CHAIN_TARGET_CAPTURE",
            "component_only": true, "diagnostic_only": true, "capture_only": true,
            "source_commit": config.source_commit, "source_dirty": true,
            "binary_path": binary_path, "binary_sha256": binary_sha256,
            "config_sha256": sha256_hex(&config_bytes), "gguf_path": config.gguf_path,
            "gguf_sha256": manifest.file_sha256,
            "prefix_positions": prefix, "suffix_tokens": suffix_tokens,
            "complete_64_layer_chain_executed": true,
            "target_feature_export": target_feature_export,
            "target_final_hidden_export": target_final_hidden_export,
            "oracle_input_tokens": oracle_input_tokens,
            "steps": capture_oracle_top50.iter().enumerate().map(|(step, top)| json!({
                "step": step, "absolute_position": prefix + step,
                "oracle_step_pass": true, "oracle_top50": top,
                "target_capture_only": true
            })).collect::<Vec<_>>(),
            "full_draft_window_supplied": suffix_tokens >= 2048,
            "input_mode": if config.free_running { "oracle_greedy" } else { "deterministic_teacher_forced" },
            "claims": {"target_capture": true, "candidate_executed": false,
                "rolling_decode": false, "semantic_parity": false, "product_ready": false,
                "default_cli": false, "token_publication": false, "kv_publication": false,
                "visible_tps": false, "context_32k_qualified": false, "context_256k_qualified": false},
            "resource": {
                "prefill_workspace_bytes": workspace_bytes,
                "prefill_workspace_released_bytes": released_workspace_bytes,
                "prefill_workspace_released_before_capture": true,
                "capture_feature_payload_bytes": capture_feature_payload_bytes,
                "capture_final_hidden_payload_bytes": capture_final_hidden_payload_bytes,
                "capture_oracle_receipt_count": capture_oracle_top50.len(),
                "live_allocations_after_oracle_cleanup": final_allocations.live_allocations,
                "live_bytes_after_oracle_cleanup": final_allocations.live_bytes,
                "source_capture_memory_policy": "retain_target_features_and_compact_receipts_only"
            },
            "claim_ceiling": "real_64_layer_target_oracle_capture_only; no_candidate_comparison; no_dflash_selector; no_target_verify; no_acceptance; no_product_tps"
        });
        return Ok(report);
    }

    // The target oracle has now produced the exact intermediate hidden rows
    // that llama.cpp's DFlash2 graph consumes.  Bind the real F16 producer
    // prefix against those rows before candidate execution.  This is a CPU
    // producer receipt only: it deliberately does not replace the candidate
    // input, run the five-layer draft, or authorize acceptance.
    let mut dflash_target_feature_cuda_pass = true;
    let dflash_target_feature_capture = if let Some(resident) = dflash_ram_resident.as_ref() {
        if resident.inspection().artifact.dtype == DFlashDType::F16 {
            let producer = DFlashRealProducerPrefix::bind(resident.clone())
                .map_err(|error| invalid(format!("DFlash target-feature producer: {error}")))?;
            let target_layer_ids = [5_usize, 19, 33, 47, 61];
            let mut suffix_receipts = Vec::with_capacity(oracle_steps.len());
            for (step, oracle) in oracle_steps.iter().enumerate() {
                let mut features = Vec::with_capacity(producer.input_elements());
                for layer in target_layer_ids {
                    let row = oracle
                        .get(layer)
                        .ok_or_else(|| invalid("DFlash target-feature layer capture"))?;
                    if row.hidden.len() != DFLASH_HIDDEN_ELEMENTS {
                        return Err(invalid("DFlash target-feature hidden geometry"));
                    }
                    features.extend_from_slice(&row.hidden);
                }
                let output = producer
                    .execute(&features)
                    .map_err(|error| invalid(format!("DFlash target-feature prefix: {error}")))?;
                let cuda_ab = run_dflash_real_prefix_cuda_ab(&stream, &producer, &features)
                    .map_err(|error| {
                        invalid(format!("DFlash target-feature CUDA prefix: {error}"))
                    })?;
                let cuda_pass = cuda_ab.projected_relative_l2 <= 2.0e-2
                    && cuda_ab.normalized_relative_l2 <= 2.0e-2;
                dflash_target_feature_cuda_pass &= cuda_pass;
                suffix_receipts.push(json!({
                    "step": step,
                    "target_layer_ids": target_layer_ids,
                    "input_elements": features.len(),
                    "input_sha256": digest_hex(&output.input_sha256),
                    "projected_sha256": digest_hex(&output.projected_sha256),
                    "normalized_sha256": digest_hex(&output.normalized_sha256),
                    "source": "real_64_layer_target_oracle_hidden_rows",
                    "cuda_ab": {
                        "executed": true,
                        "pass": cuda_pass,
                        "projected_relative_l2": cuda_ab.projected_relative_l2,
                        "normalized_relative_l2": cuda_ab.normalized_relative_l2,
                        "projected_max_abs": cuda_ab.projected_max_abs,
                        "normalized_max_abs": cuda_ab.normalized_max_abs,
                        "projected_sha256": digest_hex(&cuda_ab.projected_sha256),
                        "normalized_sha256": digest_hex(&cuda_ab.normalized_sha256),
                        "cpu_projected_sha256": digest_hex(&cuda_ab.cpu_projected_sha256),
                        "cpu_normalized_sha256": digest_hex(&cuda_ab.cpu_normalized_sha256),
                        "host_wall_ms": cuda_ab.host_wall_ms,
                    },
                }));
            }
            json!({
                "executed": true,
                "dtype": "F16",
                "target_layer_ids": target_layer_ids,
                "input_elements": producer.input_elements(),
                "output_elements": producer.output_elements(),
                "rms_norm_eps": producer.rms_norm_eps(),
                "suffix_receipts": suffix_receipts,
                "candidate_input_replaced": false,
                "cuda_producer_executed": true,
                "cuda_producer_pass": dflash_target_feature_cuda_pass,
                "candidate_logits_consumed": false,
                "claim_ceiling": "real_target_layer_feature_capture_plus_cuda_fc_hidden_norm_component_only",
            })
        } else {
            json!({
                "executed": false,
                "reason": "DFlash target-feature producer requires F16 safetensors artifact",
            })
        }
    } else {
        Value::Null
    };

    // The oracle is now CPU evidence only. Restore GDN prefix state and replace
    // EVERY full-capacity KV owner with a single-row writer before any candidate
    // operation is bound. No candidate can submit the old native attention_args.
    let before_detach = cuda_device_allocation_snapshot();
    for (plan, saved) in plans.iter_mut().zip(&prefix_states) {
        match (plan, saved) {
            (DevicePlan::Linear(p), PrefixState::Linear { history, state }) => {
                p.history.copy_from_host_sync(&stream, history)?;
                p.state.copy_from_host_sync(&stream, state)?;
            }
            (DevicePlan::Full(p), PrefixState::Full { .. }) => {
                p.attention_args.clear();
                p.key_cache = OwnedCudaBuffer::allocate(0, ROW)?;
                p.value_cache = OwnedCudaBuffer::allocate(0, ROW)?;
            }
            _ => return Err(invalid("prefix state inventory")),
        }
    }
    let after_detach = cuda_device_allocation_snapshot();
    let full_cache_bytes_released = before_detach
        .live_bytes
        .checked_sub(after_detach.live_bytes)
        .ok_or_else(|| invalid("KV release accounting"))?;
    if full_cache_bytes_released != (32 * (capacity - 1) * ROW) as u64 {
        return Err(invalid("full KV owners were not physically released"));
    }
    let mut release_digest = Sha256::new();
    release_digest.update(b"QWEN35-PREFILL-DFLASH-RELEASE-V1\0");
    release_digest.update(dflash_terminal.terminal_digest);
    release_digest.update(full_cache_bytes_released.to_le_bytes());
    release_digest.update(released_workspace_bytes.to_le_bytes());
    let dflash_handoff = dflash_lifecycle.release_prefill_with_observation(
        dflash_terminal,
        DFlashPrefillReleaseObservation {
            owner_generation: dflash_terminal.owner_generation,
            request_sha256,
            request_generation: 1,
            covering_submission_epoch: dflash_terminal.covering_submission_epoch,
            observed_full_cache_bytes: 0,
            observed_prefill_workspace_bytes: 0,
            stream_quiescent: true,
            release_digest: release_digest.finalize().into(),
        },
    )?;
    let mut dflash_device_artifact = None;
    let mut dflash_hot_consumer = None;
    let mut dflash_upload_report = Value::Null;
    match (&config.dflash_artifact_path, &config.dflash_config_path) {
        (Some(artifact_path), Some(config_path)) => {
            let inspection = match dflash_ram_resident.as_ref() {
                Some(resident) => resident.inspection().clone(),
                None => {
                    inspect_dflash_artifact(artifact_path, config_path, config.dflash_allow_f16)
                        .map_err(|error| invalid(format!("DFlash artifact inspection: {error}")))?
                }
            };
            if config.dflash_hot_window_only {
                let resident = dflash_ram_resident
                    .as_ref()
                    .ok_or_else(|| invalid("DFlash hot-window RAM owner missing"))?;
                let dflash_config_bytes = read_stable_regular_file(config_path, "DFlash config")?;
                let dflash_config = if config.dflash_allow_f16 {
                    DFlashConfig::parse_f16_opt_in(&dflash_config_bytes)
                } else {
                    DFlashConfig::parse(&dflash_config_bytes)
                }
                .map_err(|error| invalid(format!("DFlash config parse: {error}")))?;
                if dflash_config.hidden_size != model.hidden_size
                    || dflash_config.vocab_size != model.vocab_size
                {
                    return Err(invalid("DFlash selector/model geometry mismatch"));
                }
                let selector = match inspection.artifact.dtype {
                    DFlashDType::Bf16 => DFlashSelectorArtifact::bind(
                        resident.clone(),
                        dflash_config.vocab_size,
                        dflash_config.selector_rank,
                        dflash_config.hidden_size,
                    ),
                    DFlashDType::F16 => DFlashSelectorArtifact::bind_f16(
                        resident.clone(),
                        dflash_config.vocab_size,
                        dflash_config.selector_rank,
                        dflash_config.hidden_size,
                    ),
                }
                .map_err(|error| invalid(format!("DFlash selector bind: {error}")))?;
                dflash_hot_consumer = Some(DFlashHotWindowCudaConsumer::new(
                    selector,
                    stream.device_ordinal(),
                    DFlashRequestIdentity {
                        request_sha256,
                        request_generation: 1,
                    },
                )?);
                dflash_upload_report = json!({
                    "mode": "hot_window_only",
                    "full_artifact_uploaded": false,
                    "selector_hidden_projection_consumed": true,
                    "selector_tile_source_dtype": match inspection.artifact.dtype {
                        DFlashDType::Bf16 => "BF16",
                        DFlashDType::F16 => "F16",
                    },
                    "selector_tile_compute_dtype": "BF16",
                    "selector_tile_wire_conversion": inspection.artifact.dtype == DFlashDType::F16,
                    "source_tier": "pageable_ram",
                    "artifact_sha256": digest_hex(&inspection.artifact.artifact_sha256),
                    "config_sha256": digest_hex(&inspection.artifact.config_sha256),
                });
            } else {
                let upload_config = DFlashUploadConfig {
                    chunk_bytes: config.dflash_chunk_bytes,
                    ..DFlashUploadConfig::default()
                };
                let upload_result = match dflash_ram_resident.as_ref() {
                    Some(resident) => upload_dflash_after_handoff_from_ram(
                        &mut dflash_lifecycle,
                        dflash_handoff,
                        resident,
                        inspection.artifact.artifact_sha256,
                        inspection.artifact.config_sha256,
                        config.dflash_allow_f16,
                        &stream,
                        upload_config,
                    ),
                    None => upload_dflash_after_handoff(
                        &mut dflash_lifecycle,
                        dflash_handoff,
                        &inspection,
                        inspection.artifact.artifact_sha256,
                        inspection.artifact.config_sha256,
                        config.dflash_allow_f16,
                        &stream,
                        upload_config,
                    ),
                };
                let (device_artifact, upload) =
                    upload_result.map_err(|error| invalid(format!("DFlash upload: {error}")))?;
                dflash_upload_report = serde_json::to_value(upload)
                    .map_err(|error| invalid(format!("DFlash upload report: {error}")))?;
                dflash_device_artifact = Some(device_artifact);
                // Target-only suffix execution starts below. This lifecycle
                // transition does not imply any DFlash draft-kernel execution.
                dflash_lifecycle.begin_decode()?;
            }
        }
        (None, None) => {}
        _ => return Err(invalid("DFlash artifact/config must be supplied together")),
    }
    let mut dflash_q8_consumer = match config.dflash_q8_gguf_path.as_ref() {
        Some(path) => Some(DFlashQ8ResidentCudaConsumer::new(
            path,
            stream.device_ordinal(),
            &stream,
        )?),
        None => None,
    };
    let mut active = Vec::with_capacity(16);
    let mut cold = Vec::with_capacity(16);
    for saved in prefix_states.drain(..) {
        if let PrefixState::Full { key, value } = saved {
            let k = OwnedCudaBuffer::allocate(0, ACTIVE * ROW)?;
            let v = OwnedCudaBuffer::allocate(0, ACTIVE * ROW)?;
            k.copy_from_host_sync(&stream, &key[cold_end * ROW..])?;
            v.copy_from_host_sync(&stream, &value[cold_end * ROW..])?;
            // Only populated cold history remains in the candidate's RAM tier.
            let mut cold_k = Vec::with_capacity((cold_end + suffix_tokens) * ROW);
            let mut cold_v = Vec::with_capacity((cold_end + suffix_tokens) * ROW);
            cold_k.extend_from_slice(&key[..cold_end * ROW]);
            cold_v.extend_from_slice(&value[..cold_end * ROW]);
            cold.push((cold_k, cold_v));
            active.push((k, v));
        }
    }
    let candidate_workspace_start = cuda_device_allocation_snapshot();
    let candidate_workspace = PrefillWorkspace::new_with_rope_capacity(&stream, 1, 1)?;
    let candidate_workspace_bytes = cuda_device_allocation_snapshot()
        .live_bytes
        .checked_sub(candidate_workspace_start.live_bytes)
        .ok_or_else(|| invalid("candidate workspace accounting"))?;
    let slots = (0..LANES)
        .map(|_| {
            Ok((
                OwnedCudaBuffer::allocate(0, PAGE * ROW)?,
                OwnedCudaBuffer::allocate(0, PAGE * ROW)?,
            ))
        })
        .collect::<Result<Vec<_>, Qwen35MixedPrefixCudaError>>()?;
    let copy_stream = OwnedCudaStream::create(0)?;
    let mut transport = AsyncColdPageTransport::new_four_lane(
        &copy_stream,
        &slots,
        PAGE * ROW,
        config.inject_upload_corruption,
    )?;
    let merge = DevicePartialMerge::new_with_segment_mode(&stream, segment_mode)?;
    let candidate_allocation = cuda_device_allocation_snapshot();
    let execution_before = cuda_host_boundary_snapshot();
    let wall = Instant::now();
    let mut steps = Vec::with_capacity(suffix_tokens);
    let mut candidate_next_input = None::<Vec<f32>>;
    let mut candidate_next_token = None::<usize>;
    let mut candidate_input_tokens = Vec::with_capacity(suffix_tokens);
    let mut dflash_hot_report = Value::Null;
    let mut dflash_q8_report = Value::Null;
    let mut total_expected_pages = 0usize;
    let mut ring_updates = 0usize;
    let mut ring_eviction_d2h_bytes = 0usize;
    let mut ring_write_d2d_bytes = 0usize;
    let mut ring_proof_d2h_bytes = 0usize;
    let mut sequence_pass = true;
    let mut gpu_acceptance_steps = Vec::with_capacity(suffix_tokens);
    let mut partial_state_audit = Value::Null;
    let mut gdn_stage_reports = Vec::new();
    let mut leases = PageLeaseBook::new()?;
    let mut previous_tickets: [Option<PageLease>; LANES] = [None; LANES];
    let mut fault_reached = false;
    let mut fault_event = Value::Null;
    let mut progress = json!({"step": 0, "layer": null, "page": null});
    // Keep every GPU owner alive across exceptional exits. Streams are drained
    // below before any slot, writer, workspace or pinned buffer can be dropped.
    let candidate_result: Result<(), Qwen35MixedPrefixCudaError> = (|| {
        eprintln!("QWEN35_FULL_CHAIN stage=paged_candidate released_kv_bytes={full_cache_bytes_released} lanes=4 suffix={suffix_tokens}");
        for step in 0..suffix_tokens {
            let layout = ring_step(prefix, step)?;
            let position = layout.position;
            let cold_end = layout.cold_end;
            if cold
                .iter()
                .any(|(k, v)| k.len() != cold_end * ROW || v.len() != cold_end * ROW)
            {
                return Err(invalid("candidate cold archive cursor drift"));
            }
            let oracle = &oracle_steps[step];
            let oracle_logits = &oracle_logits_steps[step];
            let suffix = if config.free_running && step > 0 {
                candidate_next_input
                    .as_deref()
                    .ok_or_else(|| invalid("free-running candidate input missing"))?
            } else {
                &input[position * linear::HIDDEN..(position + 1) * linear::HIDDEN]
            };
            candidate_input_tokens.push(if config.free_running && step > 0 {
                candidate_next_token
                    .ok_or_else(|| invalid("free-running candidate token missing"))?
            } else {
                token_ids[position]
            });
            // Only RoPE and the teacher-forced input change here. GDN/conv and active
            // KV are this candidate's previous step output, never the oracle's output.
            let (cos, sin) =
                native_rope_tables(model.rope_theta as f32, position, full::ROTARY_DIM)?;
            candidate_workspace
                .rope_cos
                .copy_from_host_sync(&stream, &f32_bytes(&cos))?;
            candidate_workspace
                .rope_sin
                .copy_from_host_sync(&stream, &f32_bytes(&sin))?;
            candidate_workspace
                .hidden_a
                .copy_from_host_sync(&stream, &f32_bytes(suffix))?;
            let mut reports = Vec::with_capacity(64);
            let mut full_index = 0;
            let mut all_pass = true;
            let mut previous_hidden_relative_l2 = None::<f64>;
            let mut candidate_rows = Vec::with_capacity(16);
            let step_wall = Instant::now();
            for (layer, (plan, expected)) in plans.iter().zip(oracle).enumerate() {
                progress = json!({"step": step, "layer": layer, "page": null});
                let mut attention_check = Value::Null;
                let mut query_check = Value::Null;
                let mut current_kv_exact = Value::Null;
                let mut current_kv_diff = Value::Null;
                if config.freeze_oracle_input_layer == Some(layer) {
                    let input = if layer.is_multiple_of(2) {
                        &candidate_workspace.hidden_a
                    } else {
                        &candidate_workspace.hidden_b
                    };
                    input.copy_from_host_sync(&stream, &f32_bytes(&expected.input_hidden))?;
                    eprintln!("QWEN35_FULL_CHAIN oracle_input_injected layer={layer}");
                }
                match plan {
                    DevicePlan::Linear(p) => {
                        let ops = candidate_workspace.build_linear_ops(
                            &stream,
                            p,
                            layer,
                            1,
                            model.rms_norm_eps as f32,
                        )?;
                        if config.audit_gdn_stages && (4..=6).contains(&layer) {
                            let actual = submit_gdn_stage_audit(
                                &ops,
                                &stream,
                                &terminal,
                                &candidate_workspace,
                            )?;
                            let reference = expected
                                .gdn_stages
                                .as_ref()
                                .ok_or_else(|| invalid("missing GDN stage oracle"))?;
                            gdn_stage_reports.push(
                                json!({"layer": layer, "stages": actual.compare(reference)?}),
                            );
                        } else {
                            ops.submit_with_activation_policy(
                                &stream,
                                &candidate_workspace.cublas,
                                config.direct_post_attention,
                            )?;
                        }
                    }
                    DevicePlan::Full(p) => {
                        let ops = candidate_workspace.build_full_ops(
                            &stream,
                            p,
                            layer,
                            1,
                            0,
                            model.rms_norm_eps as f32,
                        )?;
                        ops.submit_tc_prepare_only(&stream, &candidate_workspace.cublas)?;
                        let r = complete(&stream, &terminal)?;
                        let q = values(
                            &stream,
                            &candidate_workspace.activated,
                            full::QUERY_VALUES,
                            r,
                        )?;
                        query_check = numerical(
                            "query",
                            &q,
                            expected
                                .query
                                .as_ref()
                                .ok_or_else(|| invalid("oracle query"))?,
                        );
                        let k = raw(&stream, &p.key_cache, 0, ROW, r)?;
                        let v = raw(&stream, &p.value_cache, 0, ROW, r)?;
                        current_kv_exact =
                            json!(k == expected.current_key && v == expected.current_value);
                        current_kv_diff = json!({
                            "key": byte_diff(&k, &expected.current_key),
                            "value": byte_diff(&v, &expected.current_value),
                        });
                        candidate_rows.push((k, v));
                        merge.begin_query(&stream, &candidate_workspace, position + 1)?;
                        for (page_index, base) in (0..cold_end).step_by(PAGE).enumerate() {
                            progress = json!({"step": step, "layer": layer, "page": page_index});
                            let slot = (full_index * cold_end.div_ceil(PAGE) + page_index) % LANES;
                            let end = (base + PAGE).min(cold_end);
                            let range = base * ROW..end * ROW;
                            let (keys, vals) = (
                                &cold[full_index].0[range.clone()],
                                &cold[full_index].1[range],
                            );
                            let requested = if config.fault_injection.is_some()
                                && step == config.fault_step
                                && layer == config.fault_layer
                                && page_index == config.fault_page
                            {
                                fault_reached = true;
                                fault_event = json!({"kind": config.fault_injection, "step": step, "layer": layer,
                                "page": page_index, "slot": slot, "absolute_begin": base, "absolute_end": end,
                                "target_page_dma_submitted": false, "target_page_consumer_submitted": false});
                                eprintln!("QWEN35_FULL_CHAIN_FAULT reached kind={} step={step} layer={layer} page={page_index}", config.fault_injection.as_deref().unwrap());
                                config.fault_injection.as_deref()
                            } else {
                                None
                            };
                            if requested == Some("cancel-before-submit") {
                                leases.cancel()?;
                            }
                            if requested == Some("missing-page") {
                                continue;
                            }
                            if matches!(requested, Some("corrupt-page" | "partial-write")) {
                                transport
                                    .arm_next_upload_fault(requested == Some("partial-write"))?;
                            }
                            leases.reusable(slot)?;
                            transport.restore_page(
                                &stream,
                                slot,
                                full_index * cold_end.div_ceil(PAGE) + page_index,
                                step,
                                keys,
                                vals,
                            )?;
                            let binding = PageBinding {
                                step,
                                layer,
                                begin: base,
                                end,
                            };
                            let ticket = leases.issue(slot, binding)?;
                            if requested.is_some() {
                                fault_event["target_page_dma_submitted"] = json!(true);
                            }
                            if requested == Some("cancel-after-submit") {
                                leases.cancel()?;
                            }
                            let offered = if requested == Some("stale-generation") {
                                let retired = previous_tickets[slot].ok_or_else(|| {
                                    invalid("stale-generation fixture lacks a real retired ticket")
                                })?;
                                let replay = ticket.replay_generation(retired)?;
                                fault_event["current_generation"] = json!(ticket.generation());
                                fault_event["offered_generation"] = json!(replay.generation());
                                fault_event["same_step_layer_position_binding"] = json!(true);
                                replay
                            } else {
                                ticket
                            };
                            leases.acquire(offered, binding)?;
                            merge.page(
                                &stream,
                                &candidate_workspace,
                                &slots[slot].0,
                                &slots[slot].1,
                                base,
                                base,
                                end,
                                PAGE,
                            )?;
                            if requested.is_some() {
                                fault_event["target_page_consumer_submitted"] = json!(true);
                            }
                            let consumer = stream.submission_receipt()?;
                            transport.mark_slot_released(&stream, slot, consumer, keys, vals)?;
                            leases.retire(ticket)?;
                            previous_tickets[slot] = Some(ticket);
                        }
                        let cold_partial_audit = if config.audit_partial_state && layer == 3 {
                            Some(
                                merge
                                    .snapshot_diagnostic(&stream, complete(&stream, &terminal)?)?,
                            )
                        } else {
                            None
                        };
                        for &(physical, length, base) in &layout.spans {
                            merge.page_views(
                                &stream,
                                &candidate_workspace,
                                active[full_index].0.view(physical * ROW, length * ROW)?,
                                active[full_index].1.view(physical * ROW, length * ROW)?,
                                base,
                                base,
                                base + length,
                                length,
                            )?;
                        }
                        let active_partial_audit = if config.audit_partial_state && layer == 3 {
                            Some(
                                merge
                                    .snapshot_diagnostic(&stream, complete(&stream, &terminal)?)?,
                            )
                        } else {
                            None
                        };
                        merge.page(
                            &stream,
                            &candidate_workspace,
                            &p.key_cache,
                            &p.value_cache,
                            position,
                            position,
                            position + 1,
                            1,
                        )?;
                        let current_partial_audit = if config.audit_partial_state && layer == 3 {
                            Some(
                                merge
                                    .snapshot_diagnostic(&stream, complete(&stream, &terminal)?)?,
                            )
                        } else {
                            None
                        };
                        merge.merge(&stream, &candidate_workspace, LSE_FINALIZE)?;
                        let r = complete(&stream, &terminal)?;
                        merge.check_terminal_status(&stream)?;
                        if config.audit_partial_state && layer == 3 {
                            partial_state_audit = json!({
                                "diagnostic_only": true,
                                "layer": layer,
                                "cold_pages": cold_partial_audit,
                                "active_window": active_partial_audit,
                                "current_row": current_partial_audit,
                                "finalized": merge.snapshot_diagnostic(&stream, r)?,
                            });
                        }
                        let att = values(
                            &stream,
                            &candidate_workspace.ffn_gate,
                            full::QUERY_VALUES,
                            r,
                        )?;
                        attention_check = numerical(
                            "attention",
                            &att,
                            expected
                                .attention
                                .as_ref()
                                .ok_or_else(|| invalid("oracle attention"))?,
                        );
                        if config.audit_partial_state && layer == 3 {
                            if let Some(audit) = partial_state_audit.as_object_mut() {
                                audit.insert("final_output_f32".into(), json!(att));
                            }
                        }
                        // Gate consumes this path's own query/gate/merged attention.
                        ops.attention.submit_gate(&stream)?;
                        ops.submit_post_attention_with_activation_policy(
                            &stream,
                            &candidate_workspace.cublas,
                            config.direct_post_attention,
                        )?;
                        full_index += 1;
                    }
                }
                let r = complete(&stream, &terminal)?;
                let hidden = read_f32_view(
                    &stream,
                    candidate_workspace.hidden_output(layer, 1)?.buffer_view(),
                    r,
                )?;
                let hidden_check = numerical("hidden", &hidden, &expected.hidden);
                let hidden_relative_l2 = relative_l2(&hidden_check);
                let hidden_amplification =
                    amplification(hidden_relative_l2, previous_hidden_relative_l2);
                let mut state_check = Value::Null;
                let mut history_check = Value::Null;
                if let DevicePlan::Linear(p) = plan {
                    state_check = numerical(
                        "recurrent_state",
                        &read_f32_buffer(&stream, &p.state, r)?,
                        expected
                            .recurrent
                            .as_ref()
                            .ok_or_else(|| invalid("oracle GDN"))?,
                    );
                    history_check = numerical(
                        "convolution_state",
                        &read_f32_buffer(&stream, &p.history, r)?,
                        expected
                            .convolution
                            .as_ref()
                            .ok_or_else(|| invalid("oracle conv"))?,
                    );
                }
                let layer_pass = [
                    &hidden_check,
                    &query_check,
                    &attention_check,
                    &state_check,
                    &history_check,
                ]
                .iter()
                .all(|c| {
                    c.is_null()
                        || (c["pass"] == true && (segment_mode != 3 || c["byte_exact"] == true))
                }) && (current_kv_exact.is_null() || current_kv_exact == true);
                all_pass &= layer_pass;
                reports.push(
            json!({"layer": layer, "pass": layer_pass, "hidden": hidden_check, "query": query_check,
            "attention": attention_check, "current_kv_byte_exact": current_kv_exact,
            "current_kv_diff": current_kv_diff,
            "recurrent_state": state_check, "convolution_state": history_check,
            "numerical_contract": {
                "input_hidden_relative_l2": previous_hidden_relative_l2,
                "output_hidden_relative_l2": hidden_relative_l2,
                "hidden_amplification": hidden_amplification,
                "recurrent_state_relative_l2": relative_l2(&state_check),
                "convolution_state_relative_l2": relative_l2(&history_check),
                "current_kv_byte_exact": current_kv_exact,
                "thresholds_changed": false,
            }}),
        );
                previous_hidden_relative_l2 = hidden_relative_l2;
                eprintln!(
                "QWEN35_FULL_CHAIN candidate_step={step} candidate_layer={layer} pass={layer_pass}"
            );
                if layer == 63 {
                    head_input.copy_from_host_sync(&stream, &f32_bytes(&hidden))?;
                    if let Some(consumer) = dflash_hot_consumer.as_mut() {
                        let hidden_view = CudaF32View::new(
                            candidate_workspace.hidden_output(layer, 1)?.buffer_view(),
                            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                            model.hidden_size,
                        )?;
                        consumer.consume(step, &hidden, hidden_view, &stream)?;
                    }
                    if let Some(consumer) = dflash_q8_consumer.as_mut() {
                        let hidden_view = CudaF32View::new(
                            candidate_workspace.hidden_output(layer, 1)?.buffer_view(),
                            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                            model.hidden_size,
                        )?;
                        consumer.consume(step, &hidden, hidden_view, &stream)?;
                    }
                }
            }
            head.submit(&stream)?;
            let r = complete(&stream, &terminal)?;
            let candidate_logits = read_f32_buffer(&stream, &head.logits, r)?;
            transport.finish_timing()?;
            if !leases.terminal() {
                return Err(invalid(
                    "page lease inventory not terminal at suffix boundary",
                ));
            }
            let page_proof = transport.readback_ledger();
            let logits_check = numerical("logits", &candidate_logits, oracle_logits);
            let oracle_top = top_ids(oracle_logits);
            let candidate_top = top_ids(&candidate_logits);
            #[cfg(feature = "qwen35-device-token-handoff")]
            {
                // Exercise the production SM70 acceptance ABI using the
                // target-derived candidate argmax as a bounded component
                // probe. This validates device verify/receipt wiring without
                // claiming that the DFlash backbone produced these logits.
                let draft_id = *candidate_top
                    .first()
                    .ok_or_else(|| invalid("GPU acceptance candidate top-1"))?
                    as u32;
                let target_id = *oracle_top
                    .first()
                    .ok_or_else(|| invalid("GPU acceptance oracle top-1"))?
                    as u32;
                let target_logits_bytes = f32_bytes(oracle_logits);
                let draft_buffer = OwnedCudaBuffer::allocate(0, 4)?;
                let target_logits_buffer = OwnedCudaBuffer::allocate(0, target_logits_bytes.len())?;
                let target_ids_buffer = OwnedCudaBuffer::allocate(0, 4)?;
                let receipt_buffer =
                    OwnedCudaBuffer::allocate(0, QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES)?;
                draft_buffer.copy_from_host_sync(&stream, &draft_id.to_le_bytes())?;
                target_logits_buffer.copy_from_host_sync(&stream, &target_logits_bytes)?;
                target_ids_buffer.copy_from_host_sync(&stream, &[0_u8; 4])?;
                receipt_buffer.copy_from_host_sync(
                    &stream,
                    &vec![0_u8; QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES],
                )?;
                let acceptance = CudaQwen35DflashLogitsAcceptanceArgsV1::bind(
                    &stream,
                    CudaQwen35DflashLogitsAcceptanceViewsV1 {
                        draft_ids: draft_buffer.view(0, 4)?,
                        target_logits: target_logits_buffer.view(0, target_logits_bytes.len())?,
                        target_ids: target_ids_buffer.view(0, 4)?,
                        receipt: receipt_buffer.view(0, QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES)?,
                    },
                    1,
                    (step + 1) as u64,
                    request_sha256,
                )?;
                acceptance.submit(&stream)?;
                let acceptance_submission = complete(&stream, &terminal)?;
                let mut acceptance_receipt = vec![0_u8; QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES];
                receipt_buffer.copy_to_host_sync(&stream, &mut acceptance_receipt)?;
                let decision = parse_cuda_qwen35_dflash_acceptance_receipt_v1(
                    &acceptance_receipt,
                    (step + 1) as u64,
                    request_sha256,
                )
                .map_err(|error| invalid(format!("GPU acceptance receipt: {error}")))?;
                let expected = dflash_acceptance_cpu_v1(&[draft_id], &[target_id])
                    .map_err(|error| invalid(format!("GPU acceptance CPU oracle: {error}")))?;
                let pass = (
                    decision.accepted_tokens,
                    decision.rejected_position,
                    decision.replacement_token,
                ) == expected;
                gpu_acceptance_steps.push(json!({
                    "step": step,
                    "executed": true,
                    "draft_source": "target_candidate_argmax_component_probe",
                    "draft_id": draft_id,
                    "target_id": target_id,
                    "accepted_tokens": decision.accepted_tokens,
                    "rejected_position": decision.rejected_position,
                    "replacement_token": decision.replacement_token,
                    "pass": pass,
                    "submission_epoch": acceptance_submission.submission_epoch(),
                    "receipt_d2h_bytes": QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES,
                    "claim_ceiling": "device_acceptance_abi_only; not_real_dflash_logits",
                }));
                all_pass &= pass;
                sequence_pass &= pass;
            }
            if config.free_running && step + 1 < suffix_tokens {
                let next_token = candidate_top
                    .first()
                    .copied()
                    .ok_or_else(|| invalid("free-running candidate next token"))?;
                candidate_next_token = Some(next_token);
                candidate_next_input = Some(linear::read_embedding_row(
                    &mut reader,
                    &embedding,
                    next_token,
                )?);
            }
            let final_token_equal = candidate_top.first() == oracle_top.first();
            let top50_set_equal = candidate_top.iter().all(|id| oracle_top.contains(id));
            let step_pages = 16 * cold_end.div_ceil(PAGE);
            total_expected_pages += step_pages;
            let page_proof_pass = page_proof["byte_exact"] == true
                && page_proof["pending_pages"] == 0
                && page_proof["submitted_pages"] == json!(total_expected_pages)
                && page_proof["completed_pages"] == json!(total_expected_pages);
            all_pass &= logits_check["pass"] == true
                && (segment_mode != 3 || logits_check["byte_exact"] == true)
                && final_token_equal
                && top50_set_equal
                && (segment_mode != 3 || candidate_top == oracle_top)
                && page_proof_pass
                && full_index == 16;
            let mut promotions = Vec::with_capacity(16);
            // A suffix-wide terminal gate precedes archive/ring mutation. Failure
            // ends this diagnostic run; it does not claim healthy rollback/retry.
            if suffix_tokens > 1 && all_pass {
                let mut full_index = 0;
                for (layer, plan) in plans.iter().enumerate() {
                    if let DevicePlan::Full(p) = plan {
                        let (key, value) = &active[full_index];
                        let (cold_k, cold_v) = &mut cold[full_index];
                        let receipt = complete(&stream, &terminal)?;
                        let evicted_k = raw(&stream, key, layout.head * ROW, ROW, receipt)?;
                        let evicted_v = raw(&stream, value, layout.head * ROW, ROW, receipt)?;
                        cold_k.extend_from_slice(&evicted_k);
                        cold_v.extend_from_slice(&evicted_v);
                        ring_eviction_d2h_bytes += 2 * ROW;
                        // SAFETY: these are disjoint owners. The current writers and
                        // fixed ring owners survive the exact terminal/readback below;
                        // no next token can overwrite a source or reuse a ring slot.
                        unsafe {
                            key.copy_from_device_async_unchecked(
                                &stream,
                                layout.head * ROW,
                                p.key_cache.view(0, ROW)?,
                            )?;
                            value.copy_from_device_async_unchecked(
                                &stream,
                                layout.head * ROW,
                                p.value_cache.view(0, ROW)?,
                            )?;
                        }
                        ring_write_d2d_bytes += 2 * ROW;
                        let receipt = complete(&stream, &terminal)?;
                        let copied_k = raw(&stream, key, layout.head * ROW, ROW, receipt)?;
                        let copied_v = raw(&stream, value, layout.head * ROW, ROW, receipt)?;
                        ring_proof_d2h_bytes += 2 * ROW;
                        if copied_k != candidate_rows[full_index].0
                            || copied_v != candidate_rows[full_index].1
                        {
                            return Err(invalid("candidate ring copy readback mismatch"));
                        }
                        promotions.push(json!({"layer": layer, "physical_slot": layout.head,
                    "evicted_absolute_position": cold_end, "inserted_absolute_position": position,
                    "cold_positions_after": cold_k.len() / ROW, "source": "candidate_current_writer",
                    "terminal_before_reuse": true, "write_readback_byte_exact": true}));
                        ring_updates += 1;
                        full_index += 1;
                    }
                }
            }
            steps.push(json!({"step": step, "absolute_position": position,
        "teacher_forced_input_token": token_ids[position],
        "input_token": candidate_input_tokens[step],
        "input_source": if config.free_running && step > 0 { "previous_candidate_argmax" } else { "seed_sequence" },
        "state_origin": if step == 0 { "shared_prefix_snapshot_once" } else { "previous_candidate_step" },
        "cold_positions_before": cold_end, "cold_pages_this_step": step_pages,
        "active_ring_head": layout.head, "active_spans": layout.spans.iter().map(|&(offset, length, base)|
            json!({"physical_offset": offset, "positions": length, "absolute_base": base})).collect::<Vec<_>>(),
        "pass": all_pass, "layers": reports, "logits": logits_check,
        "oracle_top50": oracle_top, "candidate_top50": candidate_top,
        "top50_set_equal": top50_set_equal, "final_token_equal": final_token_equal,
        "page_proof": page_proof, "ring_promotions": promotions,
        "candidate_audit_wall_ms": step_wall.elapsed().as_secs_f64() * 1000.}));
            sequence_pass &= all_pass;
            eprintln!("QWEN35_FULL_CHAIN candidate_step={step} terminal=true pass={all_pass}");
            if !all_pass {
                break;
            }
        }
        Ok(())
    })();
    if candidate_result.is_ok() {
        if let Some(consumer) = dflash_hot_consumer.as_mut() {
            dflash_hot_report = consumer.finish(&stream)?;
        }
        if let Some(consumer) = dflash_q8_consumer.as_mut() {
            dflash_q8_report = consumer.finish(&stream)?;
        }
    }
    let mut failure_drain = Value::Null;
    if candidate_result.is_err() || (fault_reached && config.fault_injection.is_some()) {
        if let Some(consumer) = dflash_hot_consumer.as_mut() {
            consumer.abort(&stream)?;
        }
        if let Some(consumer) = dflash_q8_consumer.as_mut() {
            consumer.abort(&stream)?;
        }
        if dflash_device_artifact.is_some() {
            dflash_lifecycle.poison();
        }
        leases.poison();
        // Fault cleanup may synchronize; this is explicitly outside a healthy
        // decode-performance contract. All dependencies were enqueued already.
        let copy_drain = copy_stream.synchronize();
        let compute_drain = stream.synchronize();
        let copy_tail_drain = copy_stream.synchronize();
        let retry_rejected = leases.reusable(0).is_err();
        failure_drain = json!({"copy_stream_drained": copy_drain.is_ok(),
            "compute_stream_drained": compute_drain.is_ok(), "copy_tail_drained": copy_tail_drain.is_ok(),
            "owner_poisoned": true, "reuse_rejected": retry_rejected,
            "healthy_rollback": false, "owners_held_until_drained": true});
        // If CUDA itself cannot drain, never label this an expected fault PASS.
        copy_drain?;
        compute_drain?;
        copy_tail_drain?;
        drop(dflash_device_artifact.take());
    }
    let candidate_wall_ms = wall.elapsed().as_secs_f64() * 1000.;
    let execution_boundaries = boundary_delta(execution_before, cuda_host_boundary_snapshot())?;
    let page_proof = transport.readback_ledger();
    let upload_faults = transport.upload_fault_ledger();
    let mut all_pass = sequence_pass && steps.len() == suffix_tokens;
    if suffix_tokens > 1 {
        all_pass &= ring_updates == 16 * suffix_tokens;
    }
    let ram_cold_bytes: usize = cold.iter().map(|(k, v)| k.len() + v.len()).sum();
    let ram_cold_capacity_bytes: usize =
        cold.iter().map(|(k, v)| k.capacity() + v.capacity()).sum();
    let oracle_ram_evidence_bytes: usize = oracle_steps
        .iter()
        .flatten()
        .map(|row| {
            4 * (row.input_hidden.len()
                + row.hidden.len()
                + row.query.as_ref().map_or(0, Vec::len)
                + row.attention.as_ref().map_or(0, Vec::len)
                + row.recurrent.as_ref().map_or(0, Vec::len)
                + row.convolution.as_ref().map_or(0, Vec::len))
                + row.current_key.len()
                + row.current_value.len()
        })
        .sum();
    let candidate_end = cuda_device_allocation_snapshot();
    let dflash_loader_active = dflash_device_artifact.is_some();
    let dflash_hot_window_active = config.dflash_hot_window_only;
    let dflash_q8_active = config.dflash_q8_gguf_path.is_some();
    let mut dflash_released_bytes = 0;
    if dflash_loader_active && candidate_result.is_ok() {
        // Upload readiness is event-proven and no draft kernel consumes this
        // buffer. Close it before sampling cleanup, without adding a host
        // synchronization. Propagate close failure instead of ignoring Drop.
        dflash_lifecycle.mark_decode_terminal()?;
        if let Some(mut buffer) = dflash_device_artifact.take() {
            dflash_released_bytes = buffer.len_bytes();
            buffer.close().map_err(|error| {
                dflash_lifecycle.poison();
                Qwen35MixedPrefixCudaError::Runtime(error)
            })?;
        }
        dflash_lifecycle.release_dflash()?;
        dflash_lifecycle.finish_request()?;
    }
    drop(dflash_hot_consumer.take());
    drop(dflash_q8_consumer.take());
    drop(transport);
    drop(merge);
    drop(copy_stream);
    drop(slots);
    drop(active);
    drop(candidate_workspace);
    drop(head);
    drop(head_input);
    drop(plans);
    drop(start);
    drop(stop);
    drop(terminal);
    drop(stream);
    let cleanup = allocation_cleanup_report(allocation_start, cuda_device_allocation_snapshot())?;
    if let Some(kind) = config.fault_injection.as_deref() {
        let observed_error = candidate_result.as_ref().err().map(|e| e.to_string());
        let prior_steps_exact =
            steps.len() == config.fault_step && steps.iter().all(|s| s["pass"] == true);
        let fail_closed = fault_reached
            && observed_error
                .as_deref()
                .is_some_and(|e| expected_fault_rejection(kind, e))
            && prior_steps_exact
            && ring_updates == 16 * config.fault_step
            && cleanup.pass
            && failure_drain["owner_poisoned"] == true
            && failure_drain["reuse_rejected"] == true;
        return Ok(json!({
            "schema": "crystal-quant.qwen35.full-chain-active-cold-fault.component.v1",
            "status": if fail_closed { "NO_GO_COMPONENT_FULL_CHAIN_ACTIVE_COLD_FAULT_REJECTED" }
                else { "NO_GO_COMPONENT_FULL_CHAIN_ACTIVE_COLD_FAULT_CONTRACT_FAILED" },
            "component_only": true, "diagnostic_only": true, "source_dirty": true,
            "source_commit": config.source_commit, "binary_path": binary_path, "binary_sha256": binary_sha256,
            "config_sha256": sha256_hex(&config_bytes), "gguf_path": config.gguf_path, "gguf_sha256": manifest.file_sha256,
            "prefix_positions": prefix, "suffix_tokens": suffix_tokens, "segment_mode": segment_mode,
            "fault_kind": kind, "fault_step": config.fault_step, "fault_layer": config.fault_layer, "fault_page": config.fault_page,
            "fault_reached": fault_reached, "fault_event": fault_event, "upload_faults": upload_faults,
            "fail_closed_observed": fail_closed, "observed_error": observed_error, "stopped_at": progress,
            "completed_steps": steps, "completed_steps_exact": prior_steps_exact,
            "requested_sequence_completed": steps.len() == suffix_tokens && sequence_pass,
            "new_report_is_fault_diagnostic_only": true,
            "page_lease_counts": {"issued": leases.issued, "acquired": leases.acquired, "release_submitted": leases.retired},
            "page_proof_at_failure": page_proof, "failure_drain": failure_drain,
            "resource": {"cleanup": cleanup, "ring_updates": ring_updates,
                "full_cache_net_released_bytes": full_cache_bytes_released,
                "candidate_full_cache_allocation_bytes": 0, "cold_slot_count": LANES,
                "prefill_workspace_released_bytes": released_workspace_bytes,
                "candidate_live_hbm_bytes_at_entry": candidate_allocation.live_bytes,
                "candidate_live_hbm_bytes_at_exit": candidate_end.live_bytes,
                "execution_boundaries": execution_boundaries},
            "claims": {"healthy_rollback": false, "product_ready": false, "semantic_parity": false,
                "default_cli": false, "token_publication": false, "kv_publication": false,
                "visible_tps": false, "context_32k_qualified": false, "context_256k_qualified": false}
        }));
    }
    candidate_result?;
    all_pass &= cleanup.pass;
    if dflash_target_feature_capture.is_object() {
        all_pass &= dflash_target_feature_cuda_pass;
    }
    if dflash_q8_active {
        all_pass &= dflash_q8_report["component_consumed"] == true
            && dflash_q8_report["numerical_ab"]["pass"] == true;
    }
    let last = steps.last().ok_or_else(|| invalid("no candidate steps"))?;
    let mut report = json!({
        "schema": if suffix_tokens == 1 { "crystal-quant.qwen35.full-chain-active-cold-one-suffix.component.v1" }
            else { "crystal-quant.qwen35.full-chain-active-cold-multi-suffix.component.v2" },
        "status": match (suffix_tokens == 1, all_pass) {
            (true, true) => "PASS_COMPONENT_FULL_CHAIN_ACTIVE_COLD_ONE_SUFFIX",
            (true, false) => "NO_GO_COMPONENT_FULL_CHAIN_ACTIVE_COLD_ONE_SUFFIX",
            (false, true) => "PASS_COMPONENT_FULL_CHAIN_ACTIVE_COLD_MULTI_SUFFIX",
            (false, false) => "NO_GO_COMPONENT_FULL_CHAIN_ACTIVE_COLD_MULTI_SUFFIX",
        },
        "component_only": true, "diagnostic_only": true,
        "source_commit": config.source_commit, "source_dirty": true,
        "binary_path": binary_path, "binary_sha256": binary_sha256,
        "config_sha256": sha256_hex(&config_bytes), "gguf_path": config.gguf_path,
        "gguf_sha256": manifest.file_sha256,
        "prefix_positions": prefix, "suffix_tokens": suffix_tokens, "teacher_forced_input_token": token_ids[prefix],
        "free_running_execution": config.free_running,
        "dflash_hot_window_only": config.dflash_hot_window_only,
        "dflash_q8_gguf_path": config.dflash_q8_gguf_path,
        "target_feature_export": target_feature_export,
        "target_final_hidden_export": target_final_hidden_export,
        "dflash_target_feature_capture": dflash_target_feature_capture,
        "gpu_acceptance": {
            "executed": !gpu_acceptance_steps.is_empty(),
            "steps": gpu_acceptance_steps,
            "claim_ceiling": "device_acceptance_abi_only; target-derived draft ids; no real DFlash candidate logits",
        },
        "oracle_input_tokens": oracle_input_tokens,
        "candidate_input_tokens": candidate_input_tokens,
        "input_hidden_sha256": sha256_f32_bits(&input),
        "complete_64_layer_chain_executed": true, "gdn_layers_executed": 48, "full_attention_layers_executed": 16,
        "post_attention_policy": if config.direct_post_attention { "direct_f32_activation" } else { "retained_q8_1" },
        "freeze_oracle_input_layer": config.freeze_oracle_input_layer,
        "precise_gdn": config.precise_gdn,
        "segment_v1_match": config.segment_v1_match,
        "segment_mode": segment_mode,
        "partial_state_semantics": if segment_mode == 3 { "ordered_prefix_fp32_recurrence" } else { "independent_page_partials_f64_merge" },
        "gdn_stage_audit": gdn_stage_reports,
        "partial_state_audit": partial_state_audit,
        "v1_segment_audit": v1_segment_audit,
        "oracle": "contiguous_kv8_v1_same_prefix_and_matching_post_attention_policy",
        "candidate": if segment_mode == 3 { "four_slot_ordered_prefix_fp32_no_fma_absolute_positions_no_full_cache" } else { "four_slot_v2_partial_f64_lse_absolute_positions_no_full_cache" },
        "prefill_host_wall_ms": prefill.host_wall_ms, "candidate_audit_wall_ms": candidate_wall_ms,
        "resource": {
            "prefill_workspace_bytes": workspace_bytes, "prefill_workspace_released_bytes": released_workspace_bytes,
            "prefill_workspace_released_before_suffix": true,
            "candidate_workspace_bytes": candidate_workspace_bytes,
            "full_cache_net_released_bytes": full_cache_bytes_released,
            "candidate_full_cache_allocation_bytes": 0,
            "prefill_dflash_handoff": {
                "schema": crate::DFLASH_LIFECYCLE_SCHEMA,
                "physical_receipt": true,
                "owner_generation": dflash_handoff.owner_generation,
                "request_generation": dflash_handoff.request_generation,
                "covering_submission_epoch": dflash_handoff.covering_submission_epoch,
                "full_cache_bytes_released": dflash_handoff.full_cache_bytes_released,
                "prefill_workspace_bytes_released": dflash_handoff.prefill_workspace_bytes_released,
                "release_digest": digest_hex(&dflash_handoff.release_digest),
                "dflash_admission": dflash_loader_active,
                "loader_upload": dflash_loader_active,
                "source_tier": if dflash_ram_resident.is_some() { "pageable_ram" } else { "artifact_path" },
                "ram_resident_bytes": dflash_ram_resident.as_ref().map(DFlashRamResident::retained_bytes),
                "upload_report": dflash_upload_report,
                "dflash_execution_consumed": false,
                "dflash_hot_window_component_consumed": dflash_hot_window_active
                    && dflash_hot_report["component_consumed"] == true,
                "dflash_hot_window_report": dflash_hot_report,
                "dflash_q8_resident_component_consumed": dflash_q8_active
                    && dflash_q8_report["component_consumed"] == true,
                "dflash_q8_resident_report": dflash_q8_report,
                "device_bytes_released_before_cleanup": dflash_released_bytes,
                "lifecycle_state_after_cleanup": format!("{:?}", dflash_lifecycle.state()),
                "claim_ceiling": if dflash_q8_active {
                    "physical_handoff_plus_q8_resident_target_hidden_projection_component_only"
                } else if dflash_hot_window_active {
                    "physical_handoff_plus_target_bound_selector_hot_window_component_only"
                } else if dflash_loader_active {
                    "physical_handoff_plus_loader_component_only"
                } else {
                    "physical_handoff_component_only"
                }
            },
            "active_hbm_kv_bytes": 32 * ACTIVE * ROW, "current_writer_hbm_bytes": 32 * ROW,
            "cold_slot_hbm_bytes": LANES * 2 * PAGE * ROW,
            "pinned_upload_bytes": LANES * 2 * PAGE * ROW,
            "pinned_proof_bytes": page_proof["pinned_capacity_bytes"],
            "host_input_hidden_payload_bytes": input.len() * 4,
            "host_oracle_evidence_payload_bytes": oracle_ram_evidence_bytes,
            "host_payload_accounting_is_not_process_rss": true,
            "ram_cold_bytes": ram_cold_bytes, "ram_cold_capacity_bytes": ram_cold_capacity_bytes,
            "cold_pages": total_expected_pages,
            "ring_updates": ring_updates, "ring_eviction_d2h_bytes": ring_eviction_d2h_bytes,
            "ring_write_d2d_bytes": ring_write_d2d_bytes, "ring_proof_d2h_bytes": ring_proof_d2h_bytes,
            "candidate_live_hbm_bytes_at_entry": candidate_allocation.live_bytes,
            "candidate_live_hbm_bytes_at_exit": candidate_end.live_bytes,
            "cold_slot_count": LANES, "copy_stream_count": 1,
            "execution_boundaries": execution_boundaries, "cleanup": cleanup
        },
        "page_proof": page_proof, "layers": last["layers"], "logits": last["logits"],
        "oracle_top50": last["oracle_top50"], "candidate_top50": last["candidate_top50"],
        "top50_set_equal": last["top50_set_equal"], "final_token_equal": last["final_token_equal"],
        "claims": { "rolling_decode": false, "semantic_parity": false, "product_ready": false,
            "default_cli": false, "token_publication": false, "kv_publication": false,
            "visible_tps": false, "context_32k_qualified": false, "context_256k_qualified": false }
    });
    report["page_lease_counts"] = json!({"issued": leases.issued, "acquired": leases.acquired,
        "release_submitted": leases.retired, "terminal": leases.terminal(),
        "binding": "owner_slot_generation_step_layer_absolute_range_v1",
        "release_is_submission_not_physical_completion": true});
    if suffix_tokens > 1 {
        report["teacher_forced_input_tokens"] = json!(&token_ids[prefix..capacity]);
        report["state_contract"] = json!({"prefix_restore_count": 1, "oracle_state_injection_between_steps": false,
            "teacher_forced_not_autoregressive": !config.free_running,
            "free_running_candidate_and_oracle": config.free_running,
            "terminal_before_next_step": true,
            "active_ring_positions": ACTIVE,
            "multi_suffix_is_not_product_decode": !config.free_running});
        report["steps"] = json!(steps);
        report["top_level_metrics_scope"] = json!("last_step_only_use_steps_for_sequence_contract");
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_only_capture_rejects_candidate_options_before_cuda() {
        let c = Qwen35FullChainActiveColdConfig {
            capture_target_only: true,
            target_feature_export_path: Some("capture.f32".into()),
            ..Default::default()
        };
        assert!(validate_capture_options(&c).is_ok());
        let mut bad = c.clone();
        bad.target_feature_export_path = None;
        assert!(validate_capture_options(&bad).is_err());
        let mut bad = c.clone();
        bad.fault_injection = Some("missing-page".into());
        assert!(validate_capture_options(&bad).is_err());
        let mut bad = c.clone();
        bad.dflash_q8_gguf_path = Some("draft.gguf".into());
        assert!(validate_capture_options(&bad).is_err());
        let mut bad = c;
        bad.audit_partial_state = true;
        assert!(validate_capture_options(&bad).is_err());
    }

    #[test]
    fn fault_admission_rejects_conflict_unreachable_unknown_and_stale_without_history() {
        let mut c = Qwen35FullChainActiveColdConfig {
            prefix_positions: 2560,
            suffix_tokens: 2,
            segment_mode: 3,
            fault_injection: Some("partial-write".into()),
            ..Default::default()
        };
        assert!(validate_fault(&c, 3).is_ok());
        assert!(validate_fault(&c, 1).is_err());
        c.fault_page = 2;
        assert!(validate_fault(&c, 3).is_err());
        c.fault_page = 0;
        c.fault_step = 2;
        assert!(validate_fault(&c, 3).is_err());
        c.fault_step = 1;
        c.inject_upload_corruption = true;
        assert!(validate_fault(&c, 3).is_err());
        c.inject_upload_corruption = false;
        c.fault_injection = Some("unknown".into());
        assert!(validate_fault(&c, 3).is_err());
        c.fault_injection = Some("stale-generation".into());
        c.fault_step = 0;
        assert!(validate_fault(&c, 3).is_err());
        assert!(!expected_fault_rejection("partial-write", "GPU OOM"));
        assert!(!expected_fault_rejection(
            "missing-page",
            "page lease request cancelled"
        ));
    }

    #[test]
    fn smoke_requires_cold_history_and_bounded_aligned_prefix() {
        for invalid in [0, 2048, 2049, 32769, usize::MAX] {
            assert!(!admitted_prefix(invalid));
        }
        for valid in [2560, 4096, 8192, 32768, 65536, 131072, 261632] {
            assert!(admitted_prefix(valid));
        }
        for invalid in [262144, 262656] {
            assert!(!admitted_prefix(invalid));
        }
    }

    #[test]
    fn numerical_gate_rejects_shape_nonfinite_and_large_drift() {
        assert_eq!(numerical("x", &[1., 2.], &[1., 2.])["pass"], true);
        for actual in [vec![], vec![1.], vec![1., f32::NAN], vec![1., 2.1]] {
            assert_eq!(numerical("x", &actual, &[1., 2.])["pass"], false);
        }
    }

    #[test]
    fn ring_spans_cover_absolute_history_once_including_wrap() {
        for prefix in [2560, 8192, 32768] {
            for step in [0, 1, 2, 63, ACTIVE - 1, ACTIVE, ACTIVE + 1, 2 * ACTIVE + 17] {
                let layout = ring_step(prefix, step).unwrap();
                let mut next = layout.cold_end;
                let mut total = 0;
                for (physical, length, base) in layout.spans {
                    assert_eq!(base, next);
                    assert!(length > 0 && physical + length <= ACTIVE);
                    next += length;
                    total += length;
                }
                assert_eq!(total, ACTIVE);
                assert_eq!(next, prefix + step);
                assert_eq!(layout.head, step % ACTIVE);
            }
        }
    }

    #[test]
    fn ring_reference_carries_candidate_rows_without_reordering() {
        let prefix = 2560;
        let mut ring: Vec<_> = (prefix - ACTIVE..prefix).collect();
        let mut cold: Vec<_> = (0..prefix - ACTIVE).collect();
        for step in 0..2 * ACTIVE + 3 {
            let layout = ring_step(prefix, step).unwrap();
            assert_eq!(cold.len(), layout.cold_end);
            let ordered: Vec<_> = layout
                .spans
                .iter()
                .flat_map(|&(offset, len, _)| ring[offset..offset + len].iter().copied())
                .collect();
            assert_eq!(
                ordered,
                (layout.cold_end..layout.position).collect::<Vec<_>>()
            );
            cold.push(ring[layout.head]);
            ring[layout.head] = layout.position;
        }
        assert_eq!(cold, (0..cold.len()).collect::<Vec<_>>());
    }

    #[test]
    fn ring_rejects_extent_overflow_and_counts_partial_cold_tail() {
        assert!(ring_step(ACTIVE, 0).is_err());
        assert!(ring_step(2560, usize::MAX).is_err());
        assert!(ring_step(u32::MAX as usize, 0).is_err());
        assert_eq!(ring_step(2560, 0).unwrap().cold_end.div_ceil(PAGE), 1);
        assert_eq!(ring_step(2560, 1).unwrap().cold_end.div_ceil(PAGE), 2);
    }
}
