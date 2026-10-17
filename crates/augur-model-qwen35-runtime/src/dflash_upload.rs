//! Opt-in four-lane DFlash payload upload.
//!
//! The uploader is deliberately downstream of [`crate::DFlashLifecycle`]. It
//! cannot allocate or copy a DFlash artifact until a physical prefill handoff
//! receipt proves that the prefill owner is quiescent and all full-cache and
//! workspace bytes have been released. The implementation mirrors the
//! DeepSeek DSpark owner discipline: bounded pinned lanes, exact stream
//! submission epochs, non-blocking completion polling, and fail-closed owner
//! poisoning on every error path.

use crate::{
    DFlashArtifactInspection, DFlashLifecycle, DFlashLifecycleError, DFlashPrefillHandoffReceipt,
    DFlashRamResident,
};
use augur_backend::{
    CudaRuntimeError, OwnedCudaBuffer, OwnedCudaEvent, OwnedCudaStream, PinnedH2dLane,
};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::{self, File};
use std::io::Read;
use std::time::Instant;

pub const DFLASH_UPLOAD_SCHEMA: &str = "qwen35-dflash-four-lane-upload-v1";
pub const DFLASH_UPLOAD_LANES: usize = 4;
const MAX_CHUNK_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct DFlashUploadConfig {
    pub chunk_bytes: usize,
    pub poll_budget: u64,
}

impl Default for DFlashUploadConfig {
    fn default() -> Self {
        Self {
            chunk_bytes: 16 * 1024 * 1024,
            poll_budget: 5_000_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct DFlashUploadReport {
    pub schema: &'static str,
    pub lanes: usize,
    pub chunk_bytes: usize,
    pub chunk_count: usize,
    pub payload_bytes: u64,
    pub uploaded_bytes: u64,
    pub max_inflight_lanes: usize,
    pub ready_before_return: bool,
    pub payload_sha256: [u8; 32],
    pub artifact_sha256: [u8; 32],
    pub config_sha256: [u8; 32],
    pub ready_event_submission_epoch: u64,
    pub ready_event_owner_bound: bool,
    pub ready_event_epoch_exact: bool,
    pub elapsed_ms: f64,
}

#[derive(Debug)]
pub enum DFlashUploadError {
    Invalid(&'static str),
    Io(String),
    Cuda(CudaRuntimeError),
    Lifecycle(DFlashLifecycleError),
    HashMismatch(&'static str),
    PollBudgetExceeded,
}

impl fmt::Display for DFlashUploadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "invalid DFlash upload: {reason}"),
            Self::Io(error) => write!(f, "DFlash upload I/O failed: {error}"),
            Self::Cuda(error) => write!(f, "DFlash upload CUDA failed: {error}"),
            Self::Lifecycle(error) => write!(f, "DFlash lifecycle rejected upload: {error}"),
            Self::HashMismatch(kind) => write!(f, "DFlash {kind} hash changed during upload"),
            Self::PollBudgetExceeded => write!(f, "DFlash upload completion poll budget exceeded"),
        }
    }
}

impl std::error::Error for DFlashUploadError {}

impl From<CudaRuntimeError> for DFlashUploadError {
    fn from(error: CudaRuntimeError) -> Self {
        Self::Cuda(error)
    }
}

impl From<DFlashLifecycleError> for DFlashUploadError {
    fn from(error: DFlashLifecycleError) -> Self {
        Self::Lifecycle(error)
    }
}

fn checked_chunk_count(payload_bytes: u64, chunk_bytes: usize) -> Result<usize, DFlashUploadError> {
    let payload = usize::try_from(payload_bytes)
        .map_err(|_| DFlashUploadError::Invalid("payload does not fit host usize"))?;
    Ok(payload.div_ceil(chunk_bytes))
}

fn poll_lanes(lanes: &mut [PinnedH2dLane<'_>], poll_budget: u64) -> Result<(), DFlashUploadError> {
    let mut polls = 0_u64;
    loop {
        let mut all_ready = true;
        for lane in lanes.iter_mut() {
            if !lane.try_reclaim()? {
                all_ready = false;
            }
        }
        if all_ready {
            return Ok(());
        }
        polls = polls.saturating_add(1);
        if polls > poll_budget {
            return Err(DFlashUploadError::PollBudgetExceeded);
        }
        std::hint::spin_loop();
    }
}

fn reclaim_lane(lane: &mut PinnedH2dLane<'_>, poll_budget: u64) -> Result<(), DFlashUploadError> {
    let mut polls = 0_u64;
    while !lane.try_reclaim()? {
        polls = polls.saturating_add(1);
        if polls > poll_budget {
            return Err(DFlashUploadError::PollBudgetExceeded);
        }
        std::hint::spin_loop();
    }
    Ok(())
}

trait DFlashUploadSource {
    fn inspection(&self) -> &DFlashArtifactInspection;
    fn read_header(&mut self, payload_offset: usize) -> Result<Vec<u8>, DFlashUploadError>;
    fn read_payload(
        &mut self,
        offset: usize,
        destination: &mut [u8],
    ) -> Result<(), DFlashUploadError>;
    fn verify_stable(&self) -> Result<(), DFlashUploadError>;
}

struct FileUploadSource<'a> {
    inspection: &'a DFlashArtifactInspection,
    file: File,
    metadata_before: fs::Metadata,
    payload_cursor: usize,
}

impl<'a> FileUploadSource<'a> {
    fn open(inspection: &'a DFlashArtifactInspection) -> Result<Self, DFlashUploadError> {
        let metadata_before = fs::symlink_metadata(&inspection.artifact_path)
            .map_err(|error| DFlashUploadError::Io(error.to_string()))?;
        if !metadata_before.file_type().is_file() || metadata_before.len() != inspection.file_bytes
        {
            return Err(DFlashUploadError::Invalid("artifact changed before upload"));
        }
        let file = File::open(&inspection.artifact_path)
            .map_err(|error| DFlashUploadError::Io(error.to_string()))?;
        Ok(Self {
            inspection,
            file,
            metadata_before,
            payload_cursor: 0,
        })
    }
}

impl DFlashUploadSource for FileUploadSource<'_> {
    fn inspection(&self) -> &DFlashArtifactInspection {
        self.inspection
    }

    fn read_header(&mut self, payload_offset: usize) -> Result<Vec<u8>, DFlashUploadError> {
        let mut header = vec![0_u8; payload_offset];
        self.file
            .read_exact(&mut header)
            .map_err(|error| DFlashUploadError::Io(error.to_string()))?;
        Ok(header)
    }

    fn read_payload(
        &mut self,
        offset: usize,
        destination: &mut [u8],
    ) -> Result<(), DFlashUploadError> {
        if offset != self.payload_cursor {
            return Err(DFlashUploadError::Invalid("file payload source order"));
        }
        self.file
            .read_exact(destination)
            .map_err(|error| DFlashUploadError::Io(error.to_string()))?;
        self.payload_cursor = self
            .payload_cursor
            .checked_add(destination.len())
            .ok_or(DFlashUploadError::Invalid("file payload offset"))?;
        Ok(())
    }

    fn verify_stable(&self) -> Result<(), DFlashUploadError> {
        let metadata_after = fs::symlink_metadata(&self.inspection.artifact_path)
            .map_err(|error| DFlashUploadError::Io(error.to_string()))?;
        if !metadata_after.file_type().is_file()
            || metadata_after.len() != self.metadata_before.len()
        {
            return Err(DFlashUploadError::Invalid("artifact changed after upload"));
        }
        Ok(())
    }
}

struct RamUploadSource<'a> {
    resident: &'a DFlashRamResident,
    payload_cursor: usize,
}

impl<'a> RamUploadSource<'a> {
    fn new(resident: &'a DFlashRamResident) -> Self {
        Self {
            resident,
            payload_cursor: 0,
        }
    }
}

impl DFlashUploadSource for RamUploadSource<'_> {
    fn inspection(&self) -> &DFlashArtifactInspection {
        self.resident.inspection()
    }

    fn read_header(&mut self, payload_offset: usize) -> Result<Vec<u8>, DFlashUploadError> {
        self.resident
            .file_bytes()
            .get(..payload_offset)
            .map(ToOwned::to_owned)
            .ok_or(DFlashUploadError::Invalid("RAM header extent"))
    }

    fn read_payload(
        &mut self,
        offset: usize,
        destination: &mut [u8],
    ) -> Result<(), DFlashUploadError> {
        if offset != self.payload_cursor {
            return Err(DFlashUploadError::Invalid("RAM payload source order"));
        }
        let end = offset
            .checked_add(destination.len())
            .ok_or(DFlashUploadError::Invalid("RAM payload offset"))?;
        let source = self
            .resident
            .payload()
            .get(offset..end)
            .ok_or(DFlashUploadError::Invalid("RAM payload extent"))?;
        destination.copy_from_slice(source);
        self.payload_cursor = end;
        Ok(())
    }

    fn verify_stable(&self) -> Result<(), DFlashUploadError> {
        Ok(())
    }
}

fn upload_inner<S: DFlashUploadSource>(
    lifecycle: &mut DFlashLifecycle,
    handoff: DFlashPrefillHandoffReceipt,
    source: &mut S,
    expected_artifact_sha256: [u8; 32],
    expected_config_sha256: [u8; 32],
    allow_f16: bool,
    stream: &OwnedCudaStream,
    config: DFlashUploadConfig,
) -> Result<(OwnedCudaBuffer, DFlashUploadReport), DFlashUploadError> {
    let inspection = source.inspection().clone();
    if config.chunk_bytes == 0 || config.chunk_bytes > MAX_CHUNK_BYTES {
        return Err(DFlashUploadError::Invalid(
            "chunk size outside bounded lane limit",
        ));
    }
    if inspection.artifact.bytes == 0
        || inspection.artifact.artifact_sha256 != expected_artifact_sha256
        || inspection.artifact.config_sha256 != expected_config_sha256
    {
        return Err(DFlashUploadError::HashMismatch("identity"));
    }
    lifecycle.admit_dflash_after_handoff(
        handoff,
        inspection.artifact,
        expected_artifact_sha256,
        expected_config_sha256,
        allow_f16,
    )?;

    let started = Instant::now();
    let payload_bytes = inspection.artifact.bytes;
    let chunk_count = checked_chunk_count(payload_bytes, config.chunk_bytes)?;
    let destination = OwnedCudaBuffer::allocate(
        stream.device_ordinal(),
        usize::try_from(payload_bytes).map_err(|_| DFlashUploadError::Invalid("payload extent"))?,
    )?;
    let mut lanes = (0..DFLASH_UPLOAD_LANES)
        .map(|_| PinnedH2dLane::new(&destination, stream, config.chunk_bytes))
        .collect::<Result<Vec<_>, _>>()?;
    let payload_offset = inspection
        .header_bytes
        .checked_add(8)
        .ok_or(DFlashUploadError::Invalid("header offset overflow"))?;
    let expected_file_bytes = payload_offset
        .checked_add(payload_bytes)
        .ok_or(DFlashUploadError::Invalid("artifact extent overflow"))?;
    if expected_file_bytes != inspection.file_bytes {
        return Err(DFlashUploadError::Invalid("artifact geometry changed"));
    }
    let header_len = usize::try_from(payload_offset)
        .map_err(|_| DFlashUploadError::Invalid("header does not fit host usize"))?;
    let header = source.read_header(header_len)?;
    let mut artifact_hasher = Sha256::new();
    artifact_hasher.update(&header);
    let mut payload_hasher = Sha256::new();
    let mut uploaded_bytes = 0_u64;
    for chunk_index in 0..chunk_count {
        let lane_index = chunk_index % DFLASH_UPLOAD_LANES;
        reclaim_lane(&mut lanes[lane_index], config.poll_budget)?;
        let len = (payload_bytes - uploaded_bytes).min(config.chunk_bytes as u64) as usize;
        let lane = &mut lanes[lane_index];
        let host = lane.writable()?;
        let source_offset = usize::try_from(uploaded_bytes)
            .map_err(|_| DFlashUploadError::Invalid("source offset"))?;
        source.read_payload(source_offset, &mut host[..len])?;
        artifact_hasher.update(&host[..len]);
        payload_hasher.update(&host[..len]);
        lane.submit(
            usize::try_from(uploaded_bytes)
                .map_err(|_| DFlashUploadError::Invalid("destination offset"))?,
            len,
        )?;
        uploaded_bytes = uploaded_bytes
            .checked_add(len as u64)
            .ok_or(DFlashUploadError::Invalid("uploaded byte overflow"))?;
    }
    if uploaded_bytes != payload_bytes {
        return Err(DFlashUploadError::Invalid("uploaded extent differs"));
    }
    poll_lanes(&mut lanes, config.poll_budget)?;
    let payload_sha256: [u8; 32] = payload_hasher.finalize().into();
    let artifact_sha256: [u8; 32] = artifact_hasher.finalize().into();
    if payload_sha256 != inspection.payload_sha256 {
        return Err(DFlashUploadError::HashMismatch("payload"));
    }
    if artifact_sha256 != inspection.artifact.artifact_sha256 {
        return Err(DFlashUploadError::HashMismatch("artifact"));
    }
    drop(lanes);
    let ready_receipt = stream.submission_receipt()?;
    let ready_event = OwnedCudaEvent::create(stream.device_ordinal())?;
    ready_event.record_for_submission(stream, ready_receipt)?;
    let ready_event_owner_bound = ready_receipt.belongs_to_stream(stream)?;
    if !ready_event_owner_bound {
        return Err(DFlashUploadError::Invalid("ready event stream owner"));
    }
    let mut polls = 0_u64;
    loop {
        if stream
            .query_event_completion(&ready_event, ready_receipt)?
            .is_some()
        {
            break;
        }
        polls = polls.saturating_add(1);
        if polls > config.poll_budget {
            return Err(DFlashUploadError::PollBudgetExceeded);
        }
        std::hint::spin_loop();
    }
    source.verify_stable()?;
    lifecycle.mark_dflash_ready()?;
    Ok((
        destination,
        DFlashUploadReport {
            schema: DFLASH_UPLOAD_SCHEMA,
            lanes: DFLASH_UPLOAD_LANES,
            chunk_bytes: config.chunk_bytes,
            chunk_count,
            payload_bytes,
            uploaded_bytes,
            max_inflight_lanes: DFLASH_UPLOAD_LANES,
            ready_before_return: true,
            payload_sha256,
            artifact_sha256,
            config_sha256: inspection.artifact.config_sha256,
            ready_event_submission_epoch: ready_receipt.submission_epoch(),
            ready_event_owner_bound,
            ready_event_epoch_exact: true,
            elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
        },
    ))
}

/// Upload a validated DFlash payload after the physical prefill handoff.
///
/// The returned device buffer is ready for decode. Any failure poisons the
/// lifecycle before the error is returned; callers must not publish a partial
/// DFlash destination or attempt a retry with the same owner generation.
pub fn upload_dflash_after_handoff(
    lifecycle: &mut DFlashLifecycle,
    handoff: DFlashPrefillHandoffReceipt,
    inspection: &DFlashArtifactInspection,
    expected_artifact_sha256: [u8; 32],
    expected_config_sha256: [u8; 32],
    allow_f16: bool,
    stream: &OwnedCudaStream,
    config: DFlashUploadConfig,
) -> Result<(OwnedCudaBuffer, DFlashUploadReport), DFlashUploadError> {
    let result = match FileUploadSource::open(inspection) {
        Ok(mut source) => upload_inner(
            lifecycle,
            handoff,
            &mut source,
            expected_artifact_sha256,
            expected_config_sha256,
            allow_f16,
            stream,
            config,
        ),
        Err(error) => Err(error),
    };
    match result {
        Ok(result) => Ok(result),
        Err(error) => {
            lifecycle.poison();
            Err(error)
        }
    }
}

/// Upload the same validated artifact from an ordinary pageable-RAM resident.
///
/// This path intentionally shares the exact lifecycle and failure handling of
/// [`upload_dflash_after_handoff`]. It removes the SSD read from the transfer
/// path but keeps only the four bounded pinned lanes as page-locked memory.
pub fn upload_dflash_after_handoff_from_ram(
    lifecycle: &mut DFlashLifecycle,
    handoff: DFlashPrefillHandoffReceipt,
    resident: &DFlashRamResident,
    expected_artifact_sha256: [u8; 32],
    expected_config_sha256: [u8; 32],
    allow_f16: bool,
    stream: &OwnedCudaStream,
    config: DFlashUploadConfig,
) -> Result<(OwnedCudaBuffer, DFlashUploadReport), DFlashUploadError> {
    let mut source = RamUploadSource::new(resident);
    match upload_inner(
        lifecycle,
        handoff,
        &mut source,
        expected_artifact_sha256,
        expected_config_sha256,
        allow_f16,
        stream,
        config,
    ) {
        Ok(result) => Ok(result),
        Err(error) => {
            lifecycle.poison();
            Err(error)
        }
    }
}
