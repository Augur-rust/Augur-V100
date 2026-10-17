#![cfg(feature = "cuda")]

//! CUDA owners, address descriptors and explicit completion protocols.
//!
//! `CudaBufferView` is a copyable descriptor, NOT an owning or lifetime-borrowed
//! allocation. Extent/device checks do not establish that an old view is live.
//! Graphs and some bound kernels retain addresses but not the owners they use;
//! mapped host slices also need an external CPU/GPU access protocol. These
//! obligations are not uniformly represented by the current safe Rust APIs.
//! See `docs/UNSAFE_TRUST_BOUNDARIES.md` for the reviewed scope and GAP-VIEW,
//! GAP-GRAPH, GAP-MAPPED and GAP-ERROR. Documentation is not a soundness fix.

use std::{
    cell::Cell,
    ffi::c_void,
    marker::PhantomData,
    ptr::{self, NonNull},
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

use augur_cuda_sys::{
    self, AugurCudaDevicePtr, AugurCudaEvent, AugurCudaEventHandle, AugurCudaResultV1, AugurCudaStream,
    AUGUR_CUDA_STATUS_INVALID_ARGUMENT, AUGUR_CUDA_STATUS_OK, AUGUR_CUDA_STATUS_OUT_OF_MEMORY,
    AUGUR_CUDA_STATUS_RUNTIME_FAILURE,
};
use sha2::{Digest, Sha256};

#[cfg(test)]
mod independent_copy_tests;

pub const M306_K13_STREAM_CAPABILITY_RECEIPT_ACTIVE: bool =
    false;
pub const CTX6R_PACED_CAPABILITY_RECEIPT_ACTIVE: bool = false;







extern "C" {
    fn cudaProfilerStart() -> i32;
    fn cudaProfilerStop() -> i32;
}

// ── Structured error ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaRuntimeErrorKind {
    Native,
    ClosedResource,
    InvalidDeviceOrdinal,
    DeviceMismatch,
    RangeOverflow,
    OutOfBounds,
    UnexpectedNullHandle,
    InvalidMetadata,
    UnsupportedCapability,
    InvalidLayout,
    InvalidDimensions,
    MisalignedDeviceAddress,
    OverlappingDeviceRanges,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaRuntimeError {
    pub operation: &'static str,
    pub kind: CudaRuntimeErrorKind,
    pub status: i32,
    pub raw_cuda_status: i32,
}

impl core::fmt::Display for CudaRuntimeError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "{} failed: kind={:?}, status={}, cuda_status={}",
            self.operation, self.kind, self.status, self.raw_cuda_status
        )
    }
}

impl std::error::Error for CudaRuntimeError {}

impl CudaRuntimeError {
    pub const fn from_result(operation: &'static str, result: AugurCudaResultV1) -> Result<(), Self> {
        if result.status == AUGUR_CUDA_STATUS_OK {
            Ok(())
        } else {
            Err(Self {
                operation,
                kind: CudaRuntimeErrorKind::Native,
                status: result.status,
                raw_cuda_status: result.raw_cuda_status,
            })
        }
    }

    pub const fn is_out_of_memory(self) -> bool {
        self.status == AUGUR_CUDA_STATUS_OUT_OF_MEMORY
    }

    const fn new(operation: &'static str, kind: CudaRuntimeErrorKind) -> Self {
        Self {
            operation,
            kind,
            status: AUGUR_CUDA_STATUS_INVALID_ARGUMENT,
            raw_cuda_status: 0,
        }
    }

    pub const fn invalid_argument(operation: &'static str) -> Self {
        Self::new(operation, CudaRuntimeErrorKind::InvalidDeviceOrdinal)
    }

    pub const fn closed(operation: &'static str) -> Self {
        Self::new(operation, CudaRuntimeErrorKind::ClosedResource)
    }

    pub const fn device_mismatch() -> Self {
        Self {
            operation: "device validation",
            kind: CudaRuntimeErrorKind::DeviceMismatch,
            status: AUGUR_CUDA_STATUS_INVALID_ARGUMENT,
            raw_cuda_status: 0,
        }
    }

    pub const fn range_overflow() -> Self {
        Self {
            operation: "range validation",
            kind: CudaRuntimeErrorKind::RangeOverflow,
            status: AUGUR_CUDA_STATUS_INVALID_ARGUMENT,
            raw_cuda_status: 0,
        }
    }

    pub const fn out_of_bounds() -> Self {
        Self {
            operation: "buffer view",
            kind: CudaRuntimeErrorKind::OutOfBounds,
            status: AUGUR_CUDA_STATUS_INVALID_ARGUMENT,
            raw_cuda_status: 0,
        }
    }

    pub const fn null_handle() -> Self {
        Self {
            operation: "raw handle",
            kind: CudaRuntimeErrorKind::UnexpectedNullHandle,
            status: AUGUR_CUDA_STATUS_INVALID_ARGUMENT,
            raw_cuda_status: 0,
        }
    }

    pub const fn validation(operation: &'static str, kind: CudaRuntimeErrorKind) -> Self {
        Self::new(operation, kind)
    }

    pub const fn invalid_metadata(operation: &'static str) -> Self {
        Self::new(operation, CudaRuntimeErrorKind::InvalidMetadata)
    }
}

// ── Pure validation helpers ───────────────────────────────────────────────

fn validate_same_device(
    operation: &'static str,
    stream_ordinal: i32,
    resource_ordinal: i32,
) -> Result<(), CudaRuntimeError> {
    if stream_ordinal != resource_ordinal {
        return Err(CudaRuntimeError {
            operation,
            kind: CudaRuntimeErrorKind::DeviceMismatch,
            status: AUGUR_CUDA_STATUS_INVALID_ARGUMENT,
            raw_cuda_status: 0,
        });
    }
    Ok(())
}

fn validate_device_ordinal(ordinal: i32) -> Result<(), CudaRuntimeError> {
    if ordinal < 0 {
        return Err(CudaRuntimeError::invalid_argument("device ordinal"));
    }
    Ok(())
}

fn checked_2d_footprint(
    operation: &'static str,
    width_bytes: usize,
    height: usize,
    pitch_bytes: usize,
) -> Result<usize, CudaRuntimeError> {
    if width_bytes == 0 || height == 0 || pitch_bytes < width_bytes {
        return Err(CudaRuntimeError::validation(
            operation,
            CudaRuntimeErrorKind::InvalidDimensions,
        ));
    }
    let rows_before_last = height - 1;
    rows_before_last
        .checked_mul(pitch_bytes)
        .and_then(|offset| offset.checked_add(width_bytes))
        .ok_or_else(CudaRuntimeError::range_overflow)
}

// ── CudaDevicePtr ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CudaDevicePtr {
    raw: AugurCudaDevicePtr,
    device_ordinal: i32,
}

impl CudaDevicePtr {
    pub const fn device_ordinal(self) -> i32 {
        self.device_ordinal
    }

    pub const fn raw_value(self) -> u64 {
        self.raw
    }
}

// ── CudaBufferView ────────────────────────────────────────────────────────

/// Non-owning device-address descriptor; copying it does not retain an allocation.
///
/// Owner IDs and checked ranges describe an allocation incarnation, but this
/// type has no borrow lifetime and cannot prevent freeing/reusing that storage.
/// Consumers must supply an owner/completion protocol. Safe APIs that accept
/// only such descriptors are a documented soundness-audit gap (GAP-VIEW), not
/// evidence that arbitrary descriptor reuse is safe.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct CudaBufferView {
    ptr: CudaDevicePtr,
    allocation_owner_instance_id: u64,
    allocation_len_bytes: usize,
    offset_bytes: usize,
    len_bytes: usize,
}

impl core::fmt::Debug for CudaBufferView {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CudaBufferView")
            .field("ptr", &self.ptr)
            .field("allocation_len_bytes", &self.allocation_len_bytes)
            .field("offset_bytes", &self.offset_bytes)
            .field("len_bytes", &self.len_bytes)
            .finish()
    }
}

impl CudaBufferView {
    pub const fn device_ordinal(self) -> i32 {
        self.ptr.device_ordinal
    }

    pub const fn offset_bytes(self) -> usize {
        self.offset_bytes
    }

    pub const fn allocation_len_bytes(self) -> usize {
        self.allocation_len_bytes
    }

    pub const fn len_bytes(self) -> usize {
        self.len_bytes
    }

    pub const fn device_ptr(self) -> CudaDevicePtr {
        self.ptr
    }

    #[cfg(test)]
    pub(crate) fn validate_owner(
        self,
        owner: &OwnedCudaBuffer,
        operation: &'static str,
    ) -> Result<(), CudaRuntimeError> {
        self.checked_range(operation)?;
        let owner_ptr = owner
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        if owner_ptr != self.ptr
            || owner.allocation_owner_instance_id != self.allocation_owner_instance_id
            || owner.len_bytes != self.allocation_len_bytes
        {
            return Err(CudaRuntimeError::validation(
                operation,
                CudaRuntimeErrorKind::InvalidMetadata,
            ));
        }
        Ok(())
    }

    pub(crate) fn checked_address(self, operation: &'static str) -> Result<u64, CudaRuntimeError> {
        if self.ptr.raw_value() == 0 {
            return Err(CudaRuntimeError::validation(
                operation,
                CudaRuntimeErrorKind::UnexpectedNullHandle,
            ));
        }
        let offset = u64::try_from(self.offset_bytes).map_err(|_| {
            CudaRuntimeError::validation(operation, CudaRuntimeErrorKind::RangeOverflow)
        })?;
        self.ptr.raw_value().checked_add(offset).ok_or_else(|| {
            CudaRuntimeError::validation(operation, CudaRuntimeErrorKind::RangeOverflow)
        })
    }

    pub(crate) fn checked_range(
        self,
        operation: &'static str,
    ) -> Result<(u64, u64), CudaRuntimeError> {
        let view_end = self
            .offset_bytes
            .checked_add(self.len_bytes)
            .ok_or_else(|| {
                CudaRuntimeError::validation(operation, CudaRuntimeErrorKind::RangeOverflow)
            })?;
        if view_end > self.allocation_len_bytes {
            return Err(CudaRuntimeError::validation(
                operation,
                CudaRuntimeErrorKind::OutOfBounds,
            ));
        }
        let start = self.checked_address(operation)?;
        let len = u64::try_from(self.len_bytes).map_err(|_| {
            CudaRuntimeError::validation(operation, CudaRuntimeErrorKind::RangeOverflow)
        })?;
        let end = start.checked_add(len).ok_or_else(|| {
            CudaRuntimeError::validation(operation, CudaRuntimeErrorKind::RangeOverflow)
        })?;
        self.ptr
            .raw_value()
            .checked_add(u64::try_from(self.allocation_len_bytes).map_err(|_| {
                CudaRuntimeError::validation(operation, CudaRuntimeErrorKind::RangeOverflow)
            })?)
            .ok_or_else(|| {
                CudaRuntimeError::validation(operation, CudaRuntimeErrorKind::RangeOverflow)
            })?;
        Ok((start, end))
    }

    pub(crate) fn overlaps(
        self,
        other: Self,
        operation: &'static str,
    ) -> Result<bool, CudaRuntimeError> {
        let (self_start, self_end) = self.checked_range(operation)?;
        let (other_start, other_end) = other.checked_range(operation)?;
        Ok(self_start < other_end && other_start < self_end)
    }

    pub fn subview(self, offset_bytes: usize, len_bytes: usize) -> Result<Self, CudaRuntimeError> {
        let end = offset_bytes
            .checked_add(len_bytes)
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        if end > self.len_bytes {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        let offset_bytes = self
            .offset_bytes
            .checked_add(offset_bytes)
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        Ok(Self {
            ptr: self.ptr,
            allocation_owner_instance_id: self.allocation_owner_instance_id,
            allocation_len_bytes: self.allocation_len_bytes,
            offset_bytes,
            len_bytes,
        })
    }

    /// Copies this complete device view to `destination` and synchronizes
    /// `stream` before returning.
    pub fn copy_to_host_sync(
        self,
        stream: &OwnedCudaStream,
        destination: &mut [u8],
    ) -> Result<(), CudaRuntimeError> {
        const OPERATION: &str = "cuda_buffer_view_memcpy_d2h";

        validate_same_device(OPERATION, stream.device_ordinal(), self.device_ordinal())?;
        if destination.len() != self.len_bytes {
            return Err(CudaRuntimeError::validation(
                OPERATION,
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        if destination.is_empty() {
            return Ok(());
        }
        self.checked_range(OPERATION)?;

        // SAFETY: `destination` is live and exclusively borrowed for exactly
        // the validated view extent. The sync_v2 call drains the stream before
        // returning, so neither the host borrow nor the device access escapes
        // this call. Same-device and allocation bounds were checked above.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memcpy_d2h_sync_v2(
                stream.device_ordinal(),
                stream.raw()?,
                destination.as_mut_ptr(),
                destination.len(),
                self.device_ptr().raw_value(),
                self.offset_bytes(),
                self.allocation_len_bytes(),
                self.len_bytes(),
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)?;
        record_synchronous_d2h(destination.len());
        Ok(())
    }

    /// Copies this view to host and returns proof that all work ordered before
    /// `receipt` completed at the same synchronous D2H boundary.
    pub fn copy_to_host_sync_with_submission_proof(
        self,
        stream: &OwnedCudaStream,
        destination: &mut [u8],
        receipt: CudaStreamSubmissionReceipt,
    ) -> Result<CudaStreamCompletionProof, CudaRuntimeError> {
        stream.validate_submission_receipt(receipt)?;
        self.copy_to_host_sync(stream, destination)?;
        Ok(stream.completion_proof_after_sync(receipt))
    }
}

// ── Device query ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CudaComputeCapability {
    pub major: u16,
    pub minor: u16,
}

impl CudaComputeCapability {
    pub const SM70: Self = Self { major: 7, minor: 0 };
    pub const SM120: Self = Self {
        major: 12,
        minor: 0,
    };

    pub const fn is_at_least(self, minimum: Self) -> bool {
        self.major > minimum.major || (self.major == minimum.major && self.minor >= minimum.minor)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaDeviceProperties {
    pub ordinal: i32,
    pub compute_capability: CudaComputeCapability,
    pub total_global_memory_bytes: u64,
    pub name: String,
}

/// Queries CUDA device properties for one ordinal.
///
/// This function allocates a device-name `String` once during
/// capability/binding. It does not run on the kernel execute hot path.
pub fn cuda_device_properties(ordinal: i32) -> Result<CudaDeviceProperties, CudaRuntimeError> {
    validate_device_ordinal(ordinal)?;

    let mut raw = augur_cuda_sys::AugurCudaDevicePropertiesV1::new();
    // SAFETY: `raw` is a live stack value initialized by
    // `AugurCudaDevicePropertiesV1::new()` with the abi_version/struct_size the
    // shim validates before writing; `ordinal` was checked non-negative above.
    let result = unsafe { augur_cuda_sys::augur_cuda_get_device_properties(ordinal, &mut raw) };
    CudaRuntimeError::from_result("cuda_get_device_properties", result)?;

    // SAFETY: `name` is NUL-terminated: `new()` zero-fills the array and the
    // shim memsets it then strncpy's at most capacity-1 bytes, preserving the
    // final NUL; `raw` is a live local, so the pointer outlives the `CStr` use.
    let name = unsafe { std::ffi::CStr::from_ptr(raw.name.as_ptr()) }
        .to_str()
        .map_err(|_| CudaRuntimeError::invalid_metadata("cuda_get_device_properties"))?
        .to_owned();

    Ok(CudaDeviceProperties {
        ordinal: raw.ordinal,
        compute_capability: CudaComputeCapability {
            major: raw.compute_major,
            minor: raw.compute_minor,
        },
        total_global_memory_bytes: raw.total_global_memory_bytes,
        name,
    })
}

/// Returns the CUDA runtime's canonical 16-byte UUID for one device ordinal.
pub fn cuda_device_uuid(ordinal: i32) -> Result<[u8; 16], CudaRuntimeError> {
    validate_device_ordinal(ordinal)?;
    let mut uuid = [0_u8; 16];
    // SAFETY: `uuid` is a live 16-byte stack array and its exact length is
    // passed alongside; the shim rejects any out_uuid_bytes != 16 and writes
    // at most that many bytes.
    let result = unsafe {
        augur_cuda_sys::augur_cuda_get_device_uuid_v1(
            ordinal,
            uuid.as_mut_ptr(),
            u32::try_from(uuid.len())
                .map_err(|_| CudaRuntimeError::invalid_metadata("cuda_get_device_uuid_v1"))?,
        )
    };
    CudaRuntimeError::from_result("cuda_get_device_uuid_v1", result)?;
    if uuid.iter().all(|byte| *byte == 0) {
        return Err(CudaRuntimeError::invalid_metadata(
            "cuda_get_device_uuid_v1",
        ));
    }
    Ok(uuid)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaMemoryInfo {
    pub free_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaDeviceAllocationSnapshot {
    pub successful_allocations: u64,
    pub successful_frees: u64,
    pub live_allocations: u64,
    pub live_bytes: u64,
    pub peak_live_allocations: u64,
    pub peak_live_bytes: u64,
}

/// Successful host-visible CUDA boundaries observed on the current thread.
///
/// Snapshot immediately before and after a submission-thread interval, then
/// use [`Self::checked_delta_since`] to audit that interval. Async enqueues and
/// device-side event waits are deliberately excluded; each `host_waits` entry
/// is a synchronous copy, stream synchronization, or event synchronization.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHostBoundarySnapshot {
    pub host_waits: u64,
    pub stream_synchronizations: u64,
    pub event_synchronizations: u64,
    pub synchronous_h2d_copies: u64,
    pub synchronous_h2d_bytes: u64,
    pub synchronous_d2h_copies: u64,
    pub synchronous_d2h_bytes: u64,
}

impl CudaHostBoundarySnapshot {
    const ZERO: Self = Self {
        host_waits: 0,
        stream_synchronizations: 0,
        event_synchronizations: 0,
        synchronous_h2d_copies: 0,
        synchronous_h2d_bytes: 0,
        synchronous_d2h_copies: 0,
        synchronous_d2h_bytes: 0,
    };

    pub fn checked_delta_since(self, earlier: Self) -> Option<Self> {
        Some(Self {
            host_waits: self.host_waits.checked_sub(earlier.host_waits)?,
            stream_synchronizations: self
                .stream_synchronizations
                .checked_sub(earlier.stream_synchronizations)?,
            event_synchronizations: self
                .event_synchronizations
                .checked_sub(earlier.event_synchronizations)?,
            synchronous_h2d_copies: self
                .synchronous_h2d_copies
                .checked_sub(earlier.synchronous_h2d_copies)?,
            synchronous_h2d_bytes: self
                .synchronous_h2d_bytes
                .checked_sub(earlier.synchronous_h2d_bytes)?,
            synchronous_d2h_copies: self
                .synchronous_d2h_copies
                .checked_sub(earlier.synchronous_d2h_copies)?,
            synchronous_d2h_bytes: self
                .synchronous_d2h_bytes
                .checked_sub(earlier.synchronous_d2h_bytes)?,
        })
    }
}

thread_local! {
    static CUDA_HOST_BOUNDARY_COUNTERS: Cell<CudaHostBoundarySnapshot> =
        const { Cell::new(CudaHostBoundarySnapshot::ZERO) };
}

fn update_cuda_host_boundary(update: impl FnOnce(&mut CudaHostBoundarySnapshot)) {
    CUDA_HOST_BOUNDARY_COUNTERS.with(|counters| {
        let mut snapshot = counters.get();
        update(&mut snapshot);
        counters.set(snapshot);
    });
}

pub(crate) fn record_stream_synchronization() {
    update_cuda_host_boundary(|snapshot| {
        snapshot.host_waits = snapshot.host_waits.saturating_add(1);
        snapshot.stream_synchronizations = snapshot.stream_synchronizations.saturating_add(1);
    });
}

fn record_event_synchronization() {
    update_cuda_host_boundary(|snapshot| {
        snapshot.host_waits = snapshot.host_waits.saturating_add(1);
        snapshot.event_synchronizations = snapshot.event_synchronizations.saturating_add(1);
    });
}

fn record_synchronous_h2d(len_bytes: usize) {
    update_cuda_host_boundary(|snapshot| {
        snapshot.host_waits = snapshot.host_waits.saturating_add(1);
        snapshot.synchronous_h2d_copies = snapshot.synchronous_h2d_copies.saturating_add(1);
        snapshot.synchronous_h2d_bytes = snapshot
            .synchronous_h2d_bytes
            .saturating_add(len_bytes as u64);
    });
}

fn record_synchronous_d2h(len_bytes: usize) {
    update_cuda_host_boundary(|snapshot| {
        snapshot.host_waits = snapshot.host_waits.saturating_add(1);
        snapshot.synchronous_d2h_copies = snapshot.synchronous_d2h_copies.saturating_add(1);
        snapshot.synchronous_d2h_bytes = snapshot
            .synchronous_d2h_bytes
            .saturating_add(len_bytes as u64);
    });
}

struct CudaDeviceAllocationCounters {
    successful_allocations: AtomicU64,
    successful_frees: AtomicU64,
    live_allocations: AtomicU64,
    live_bytes: AtomicU64,
    peak_live_allocations: AtomicU64,
    peak_live_bytes: AtomicU64,
}

impl CudaDeviceAllocationCounters {
    const fn new() -> Self {
        Self {
            successful_allocations: AtomicU64::new(0),
            successful_frees: AtomicU64::new(0),
            live_allocations: AtomicU64::new(0),
            live_bytes: AtomicU64::new(0),
            peak_live_allocations: AtomicU64::new(0),
            peak_live_bytes: AtomicU64::new(0),
        }
    }

    fn record_allocation(&self, len_bytes: usize) {
        let len_bytes = len_bytes as u64;
        self.successful_allocations.fetch_add(1, Ordering::Relaxed);
        let live_allocations = self.live_allocations.fetch_add(1, Ordering::Relaxed) + 1;
        let live_bytes = self.live_bytes.fetch_add(len_bytes, Ordering::Relaxed) + len_bytes;
        self.peak_live_allocations
            .fetch_max(live_allocations, Ordering::Relaxed);
        self.peak_live_bytes
            .fetch_max(live_bytes, Ordering::Relaxed);
    }

    fn record_free(&self, len_bytes: usize) {
        let len_bytes = len_bytes as u64;
        let previous_allocations = self.live_allocations.fetch_sub(1, Ordering::Relaxed);
        let previous_bytes = self.live_bytes.fetch_sub(len_bytes, Ordering::Relaxed);
        assert!(previous_allocations > 0 && previous_bytes >= len_bytes);
        self.successful_frees.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> CudaDeviceAllocationSnapshot {
        CudaDeviceAllocationSnapshot {
            successful_allocations: self.successful_allocations.load(Ordering::Relaxed),
            successful_frees: self.successful_frees.load(Ordering::Relaxed),
            live_allocations: self.live_allocations.load(Ordering::Relaxed),
            live_bytes: self.live_bytes.load(Ordering::Relaxed),
            peak_live_allocations: self.peak_live_allocations.load(Ordering::Relaxed),
            peak_live_bytes: self.peak_live_bytes.load(Ordering::Relaxed),
        }
    }
}

static CUDA_DEVICE_ALLOCATION_COUNTERS: CudaDeviceAllocationCounters =
    CudaDeviceAllocationCounters::new();

/// Returns process-local counters for successful `OwnedCudaBuffer` allocations.
/// Snapshots around a steady interval prove whether that interval allocated or
/// freed project-owned device buffers without adding work to the CUDA stream.
pub fn cuda_device_allocation_snapshot() -> CudaDeviceAllocationSnapshot {
    CUDA_DEVICE_ALLOCATION_COUNTERS.snapshot()
}

/// Returns host-boundary counters for the calling submission thread.
pub fn cuda_host_boundary_snapshot() -> CudaHostBoundarySnapshot {
    CUDA_HOST_BOUNDARY_COUNTERS.with(Cell::get)
}

fn profiler_api_result(
    operation: &'static str,
    raw_cuda_status: i32,
) -> Result<(), CudaRuntimeError> {
    if raw_cuda_status == 0 {
        Ok(())
    } else {
        Err(CudaRuntimeError {
            operation,
            kind: CudaRuntimeErrorKind::Native,
            status: AUGUR_CUDA_STATUS_RUNTIME_FAILURE,
            raw_cuda_status,
        })
    }
}

/// Starts collection for profilers configured with the CUDA profiler API range.
///
/// This calls `cudaProfilerStart` directly and does not synchronize a device or
/// stream. The external profiler remains responsible for capture configuration.
fn cuda_profiler_start() -> Result<(), CudaRuntimeError> {
    // SAFETY: the CUDA runtime function has no pointer arguments and returns a
    // status synchronously; the process already links against cudart.
    profiler_api_result("cudaProfilerStart", unsafe { cudaProfilerStart() })
}

/// Stops collection for the active CUDA profiler API range without adding a
/// device or stream synchronization.
fn cuda_profiler_stop() -> Result<(), CudaRuntimeError> {
    // SAFETY: the CUDA runtime function has no pointer arguments and returns a
    // status synchronously; the process already links against cudart.
    profiler_api_result("cudaProfilerStop", unsafe { cudaProfilerStop() })
}

pub const AUGUR_M306_STEADY_TOKEN_NVTX_RANGE_V1: &str = "AUGUR_M306_STEADY_TOKEN_V1";
pub const AUGUR_BATCH01_B4_ROW_MAIN_NVTX_RANGES_V1: [&str; 4] = [
    "CQ_BATCH01_B4_ROW0_MAIN_STREAM_START_V1",
    "CQ_BATCH01_B4_ROW1_MAIN_STREAM_START_V1",
    "CQ_BATCH01_B4_ROW2_MAIN_STREAM_START_V1",
    "CQ_BATCH01_B4_ROW3_MAIN_STREAM_START_V1",
];

/// Owns the fixed M306 steady-token NVTX thread range.
///
/// NVTX push/pop ranges are thread-local, so this owner is deliberately
/// neither `Send` nor `Sync`. An explicit `end` reports native failures;
/// dropping an active range makes only one best-effort cleanup attempt.
#[derive(Debug)]
pub struct CudaNvtxRange {
    active: bool,
    _thread_bound: PhantomData<Rc<()>>,
}

impl CudaNvtxRange {
    pub fn begin_m306_steady_token() -> Result<Self, CudaRuntimeError> {
        // SAFETY: no-argument FFI call; the shim tracks its own thread-local
        // nesting flag and reports duplicate begins through the status.
        let result = unsafe { augur_cuda_sys::augur_cuda_nvtx_m306_steady_token_begin_v1() };
        CudaRuntimeError::from_result("nvtxRangePushA(AUGUR_M306_STEADY_TOKEN_V1)", result)?;
        Ok(Self {
            active: true,
            _thread_bound: PhantomData,
        })
    }

    pub fn end(&mut self) -> Result<(), CudaRuntimeError> {
        if !self.active {
            return Err(CudaRuntimeError::invalid_metadata(
                "duplicate nvtxRangePop(AUGUR_M306_STEADY_TOKEN_V1)",
            ));
        }
        // SAFETY: no-argument FFI call; `self.active` plus this type's `!Send`
        // marker (`PhantomData<Rc<()>>`) prove the matching begin ran on this
        // thread, and the shim re-checks its thread-local nesting flag.
        let result = unsafe { augur_cuda_sys::augur_cuda_nvtx_m306_steady_token_end_v1() };
        CudaRuntimeError::from_result("nvtxRangePop(AUGUR_M306_STEADY_TOKEN_V1)", result)?;
        self.active = false;
        Ok(())
    }

    pub fn is_active(&self) -> bool {
        self.active
    }
}

impl Drop for CudaNvtxRange {
    fn drop(&mut self) {
        if self.active {
            // SAFETY: same contract as `end`: no arguments, and `self.active`
            // on this `!Send` owner implies the matching begin on this thread;
            // best-effort cleanup deliberately ignores the returned status.
            let _ = unsafe { augur_cuda_sys::augur_cuda_nvtx_m306_steady_token_end_v1() };
            self.active = false;
        }
    }
}

/// Owns one fixed BATCH-01 logical-row marker nested in the M306 range.
///
/// The marker exists only to attribute the enclosed `cudaEventRecord` runtime
/// call to one logical row. Native validation requires the outer steady-token
/// range, rejects nesting, and binds the pop to the same row index.
#[derive(Debug)]
pub struct CudaBatch01B4RowMainNvtxRange {
    row_index: u32,
    active: bool,
    _thread_bound: PhantomData<Rc<()>>,
}

impl CudaBatch01B4RowMainNvtxRange {
    pub fn begin(row_index: usize) -> Result<Self, CudaRuntimeError> {
        if row_index >= AUGUR_BATCH01_B4_ROW_MAIN_NVTX_RANGES_V1.len() {
            return Err(CudaRuntimeError::invalid_metadata(
                "BATCH-01 NVTX row index",
            ));
        }
        // SAFETY: the row index was checked against the fixed native table;
        // the shim validates outer-range ownership and thread-local nesting.
        let result =
            unsafe { augur_cuda_sys::augur_cuda_nvtx_batch01_b4_row_main_begin_v1(row_index as u32) };
        CudaRuntimeError::from_result("BATCH-01 row nvtxRangePushA", result)?;
        Ok(Self {
            row_index: row_index as u32,
            active: true,
            _thread_bound: PhantomData,
        })
    }

    pub fn end(&mut self) -> Result<(), CudaRuntimeError> {
        if !self.active {
            return Err(CudaRuntimeError::invalid_metadata(
                "duplicate BATCH-01 row nvtxRangePop",
            ));
        }
        // SAFETY: this `!Send` owner preserves the begin thread and the native
        // shim verifies that the same row is currently active.
        let result =
            unsafe { augur_cuda_sys::augur_cuda_nvtx_batch01_b4_row_main_end_v1(self.row_index) };
        CudaRuntimeError::from_result("BATCH-01 row nvtxRangePop", result)?;
        self.active = false;
        Ok(())
    }

    pub fn range_name(&self) -> &'static str {
        AUGUR_BATCH01_B4_ROW_MAIN_NVTX_RANGES_V1[self.row_index as usize]
    }
}

impl Drop for CudaBatch01B4RowMainNvtxRange {
    fn drop(&mut self) {
        if self.active {
            // SAFETY: same contract as `end`; cleanup is best effort because
            // `Drop` cannot report a native NVTX failure.
            let _ = unsafe { augur_cuda_sys::augur_cuda_nvtx_batch01_b4_row_main_end_v1(self.row_index) };
            self.active = false;
        }
    }
}

/// Owns one CUDA profiler API capture range without synchronizing CUDA work.
///
/// The narrow owner prevents a profiled executor from accepting an arbitrary
/// callback at its terminal boundary. Dropping an active range makes one
/// best-effort stop attempt; formal callers must still observe `stop` errors.
#[derive(Debug)]
pub struct CudaProfilerCapture {
    profiler_active: bool,
    nvtx_range: Option<CudaNvtxRange>,
}

impl CudaProfilerCapture {
    pub fn start() -> Result<Self, CudaRuntimeError> {
        cuda_profiler_start()?;
        Ok(Self {
            profiler_active: true,
            nvtx_range: None,
        })
    }

    pub fn start_with_nvtx_range(name: &str) -> Result<Self, CudaRuntimeError> {
        if name != AUGUR_M306_STEADY_TOKEN_NVTX_RANGE_V1 {
            return Err(CudaRuntimeError::invalid_metadata(
                "unsupported CUDA profiler NVTX range",
            ));
        }

        cuda_profiler_start()?;
        match CudaNvtxRange::begin_m306_steady_token() {
            Ok(nvtx_range) => Ok(Self {
                profiler_active: true,
                nvtx_range: Some(nvtx_range),
            }),
            Err(error) => {
                let _ = cuda_profiler_stop();
                Err(error)
            }
        }
    }

    pub fn stop(&mut self) -> Result<(), CudaRuntimeError> {
        if !self.profiler_active {
            return Err(CudaRuntimeError::invalid_metadata(
                "duplicate cudaProfilerStop",
            ));
        }

        let nvtx_result = match self.nvtx_range.as_mut() {
            Some(range) if range.is_active() => range.end(),
            _ => Ok(()),
        };
        let profiler_result = cuda_profiler_stop();
        if profiler_result.is_ok() {
            self.profiler_active = false;
        }

        nvtx_result.and(profiler_result)
    }

    pub fn is_active(&self) -> bool {
        self.profiler_active
    }
}

impl Drop for CudaProfilerCapture {
    fn drop(&mut self) {
        if let Some(range) = self.nvtx_range.take() {
            drop(range);
        }
        if self.profiler_active {
            let _ = cuda_profiler_stop();
            self.profiler_active = false;
        }
    }
}

/// Returns one read-only `cudaMemGetInfo` snapshot for the selected device.
pub fn cuda_memory_info(ordinal: i32) -> Result<CudaMemoryInfo, CudaRuntimeError> {
    validate_device_ordinal(ordinal)?;
    let mut free_bytes = 0_u64;
    let mut total_bytes = 0_u64;
    // SAFETY: both out-params point at live stack `u64`s that the shim zeroes
    // before use; `ordinal` was validated non-negative above.
    let result =
        unsafe { augur_cuda_sys::augur_cuda_mem_get_info_v1(ordinal, &mut free_bytes, &mut total_bytes) };
    CudaRuntimeError::from_result("cuda_mem_get_info", result)?;
    if total_bytes == 0 || free_bytes > total_bytes {
        return Err(CudaRuntimeError::invalid_metadata("cuda_mem_get_info"));
    }
    Ok(CudaMemoryInfo {
        free_bytes,
        total_bytes,
    })
}

// ── OwnedCudaStream ───────────────────────────────────────────────────────

static NEXT_CUDA_STREAM_OWNER_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_CUDA_BUFFER_OWNER_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_CUDA_STREAM_CLOSE_EPOCH: AtomicU64 = AtomicU64::new(1);

fn next_cuda_stream_owner_id() -> Result<u64, CudaRuntimeError> {
    let mut current = NEXT_CUDA_STREAM_OWNER_ID.load(Ordering::Relaxed);
    loop {
        if current == u64::MAX {
            return Err(CudaRuntimeError::invalid_metadata(
                "cuda_stream_owner_instance_key",
            ));
        }
        match NEXT_CUDA_STREAM_OWNER_ID.compare_exchange_weak(
            current,
            current + 1,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return Ok(current),
            Err(observed) => current = observed,
        }
    }
}

fn next_cuda_stream_close_epoch() -> Result<u64, CudaRuntimeError> {
    let mut current = NEXT_CUDA_STREAM_CLOSE_EPOCH.load(Ordering::Relaxed);
    loop {
        if current == u64::MAX {
            return Err(CudaRuntimeError::invalid_metadata(
                "cuda_stream_close_epoch",
            ));
        }
        match NEXT_CUDA_STREAM_CLOSE_EPOCH.compare_exchange_weak(
            current,
            current + 1,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return Ok(current),
            Err(observed) => current = observed,
        }
    }
}

fn next_cuda_buffer_owner_id() -> Result<u64, CudaRuntimeError> {
    let mut current = NEXT_CUDA_BUFFER_OWNER_ID.load(Ordering::Relaxed);
    loop {
        if current == u64::MAX {
            return Err(CudaRuntimeError::invalid_metadata(
                "cuda_buffer_owner_instance_key",
            ));
        }
        match NEXT_CUDA_BUFFER_OWNER_ID.compare_exchange_weak(
            current,
            current + 1,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return Ok(current),
            Err(observed) => current = observed,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaStreamSubmissionReceipt {
    device_ordinal: i32,
    stream_owner_instance_id: u64,
    submission_epoch: u64,
}

impl CudaStreamSubmissionReceipt {
    pub const fn device_ordinal(self) -> i32 {
        self.device_ordinal
    }

    pub const fn submission_epoch(self) -> u64 {
        self.submission_epoch
    }

    /// Returns whether this receipt belongs to the exact live stream owner.
    ///
    /// The owner incarnation is intentionally stronger than a native stream
    /// handle or device ordinal: a destroyed stream must not satisfy a
    /// dependency check after allocator address reuse.
    pub fn belongs_to_stream(self, stream: &OwnedCudaStream) -> Result<bool, CudaRuntimeError> {
        let _ = stream.raw()?;
        Ok(self.device_ordinal == stream.device_ordinal
            && self.stream_owner_instance_id == stream.owner_instance_id)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaStreamCloseReceipt {
    device_ordinal: i32,
    stream_owner_instance_id: u64,
    close_epoch: u64,
    receipt_sha256: [u8; 32],
}

impl CudaStreamCloseReceipt {
    pub const fn device_ordinal(self) -> i32 {
        self.device_ordinal
    }

    pub const fn stream_owner_instance_id(self) -> u64 {
        self.stream_owner_instance_id
    }

    pub const fn close_epoch(self) -> u64 {
        self.close_epoch
    }

    pub const fn receipt_sha256(self) -> [u8; 32] {
        self.receipt_sha256
    }
}

fn cuda_stream_close_receipt_sha256(
    device_ordinal: i32,
    stream_owner_instance_id: u64,
    close_epoch: u64,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"CQ-P4-WIDTH6-STREAM-CLOSE-RECEIPT-V2\0");
    digest.update((device_ordinal as u64).to_le_bytes());
    digest.update(stream_owner_instance_id.to_le_bytes());
    digest.update(close_epoch.to_le_bytes());
    digest.update(b"CLOSED");
    digest.finalize().into()
}

#[derive(Debug, Eq, PartialEq)]
pub struct CudaStreamCompletionProof {
    device_ordinal: i32,
    stream_owner_instance_id: u64,
    submission_epoch: u64,
}

impl CudaStreamCompletionProof {
    pub const fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    pub const fn submission_epoch(&self) -> u64 {
        self.submission_epoch
    }

    /// Returns whether this terminal proof covers an earlier submission point
    /// on the same stream owner.
    pub const fn covers_submission(&self, receipt: CudaStreamSubmissionReceipt) -> bool {
        self.device_ordinal == receipt.device_ordinal
            && self.stream_owner_instance_id == receipt.stream_owner_instance_id
            && self.submission_epoch >= receipt.submission_epoch
    }

    /// Returns whether this proof belongs to the exact live stream owner.
    ///
    /// This checks the process-local owner incarnation rather than a native
    /// handle address, so destroying and recreating a stream cannot satisfy
    /// the comparison through allocator address reuse.
    pub fn belongs_to_stream(&self, stream: &OwnedCudaStream) -> Result<bool, CudaRuntimeError> {
        let _ = stream.raw()?;
        Ok(self.device_ordinal == stream.device_ordinal
            && self.stream_owner_instance_id == stream.owner_instance_id)
    }
}

pub struct OwnedCudaStream {
    raw: Option<NonNull<augur_cuda_sys::AugurCudaStreamHandle>>,
    device_ordinal: i32,
    /// Monotonic process-local identity for lifetime/fence checks.
    ///
    /// The native wrapper address is deliberately not used here: CUDA may
    /// destroy a wrapper while queued work is still draining, and a later
    /// allocation can reuse that address (ABA).  This identity remains unique
    /// for the lifetime of the process even when the native address is reused.
    owner_instance_id: u64,
    next_submission_epoch: AtomicU64,
    
    
}

impl core::fmt::Debug for OwnedCudaStream {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OwnedCudaStream")
            .field("device_ordinal", &self.device_ordinal)
            .field("closed", &self.raw.is_none())
            .finish()
    }
}

impl OwnedCudaStream {
    pub fn create(device_ordinal: i32) -> Result<Self, CudaRuntimeError> {
        validate_device_ordinal(device_ordinal)?;
        let owner_instance_id = next_cuda_stream_owner_id()?;

        

        let mut raw: AugurCudaStream = ptr::null_mut();

        // SAFETY: `raw` is a live out-slot and `device_ordinal` was validated
        // non-negative above; on success the shim transfers sole ownership of
        // a freshly heap-allocated stream handle, which `Self` then owns.
        let result = unsafe { augur_cuda_sys::augur_cuda_stream_create(device_ordinal, &mut raw) };

        CudaRuntimeError::from_result("cuda_stream_create", result)?;

        let raw = NonNull::new(raw).ok_or_else(CudaRuntimeError::null_handle)?;

        Ok(Self {
            raw: Some(raw),
            device_ordinal,
            owner_instance_id,
            next_submission_epoch: AtomicU64::new(1),
            
            
        })
    }

    pub const fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    /// Process-local monotonic owner key used to prove that a replay kept the
    /// same CUDA stream owner. It is never serialized. The key is independent
    /// of the native wrapper address and therefore cannot pass an ABA check
    /// after a stream is destroyed and a new stream is allocated.
    pub fn owner_instance_key(&self) -> Result<usize, CudaRuntimeError> {
        let _ = self.raw()?;
        usize::try_from(self.owner_instance_id)
            .map_err(|_| CudaRuntimeError::invalid_metadata("cuda_stream_owner_instance_key"))
    }

    fn reserve_submission_receipt(&self) -> Result<CudaStreamSubmissionReceipt, CudaRuntimeError> {
        let _ = self.raw()?;
        let submission_epoch = self
            .next_submission_epoch
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |epoch| {
                (epoch != u64::MAX).then_some(epoch + 1)
            })
            .map_err(|_| CudaRuntimeError::invalid_metadata("cuda_stream_submission_epoch"))?;
        Ok(CudaStreamSubmissionReceipt {
            device_ordinal: self.device_ordinal,
            stream_owner_instance_id: self.owner_instance_id,
            submission_epoch,
        })
    }

    /// Captures a monotonic submission point after the caller has enqueued the
    /// work whose lifetime must be covered by a later terminal proof.
    pub fn submission_receipt(&self) -> Result<CudaStreamSubmissionReceipt, CudaRuntimeError> {
        self.reserve_submission_receipt()
    }

    fn validate_submission_receipt(
        &self,
        receipt: CudaStreamSubmissionReceipt,
    ) -> Result<(), CudaRuntimeError> {
        let _ = self.raw()?;
        if receipt.device_ordinal != self.device_ordinal
            || receipt.stream_owner_instance_id != self.owner_instance_id
            || receipt.submission_epoch == 0
        {
            return Err(CudaRuntimeError::invalid_metadata(
                "cuda_stream_submission_receipt",
            ));
        }
        Ok(())
    }

    const fn completion_proof_after_sync(
        &self,
        receipt: CudaStreamSubmissionReceipt,
    ) -> CudaStreamCompletionProof {
        CudaStreamCompletionProof {
            device_ordinal: receipt.device_ordinal,
            stream_owner_instance_id: receipt.stream_owner_instance_id,
            submission_epoch: receipt.submission_epoch,
        }
    }

    /// Waits for all work ordered before `receipt` and returns an owner/epoch
    /// proof that can release the corresponding D2D batch.
    pub fn synchronize_submission(
        &self,
        receipt: CudaStreamSubmissionReceipt,
    ) -> Result<CudaStreamCompletionProof, CudaRuntimeError> {
        self.validate_submission_receipt(receipt)?;
        self.synchronize()?;
        Ok(self.completion_proof_after_sync(receipt))
    }

    /// Poll a pre-recorded event and, when it has completed, return a
    /// non-blocking completion proof for the associated stream submission.
    ///
    /// This is the event-driven counterpart to [`Self::synchronize_submission`].
    /// The event must have been recorded for this exact receipt through
    /// [`OwnedCudaEvent::record_for_submission`]. Both the stream owner and
    /// submission epoch must match, so a completed event from an earlier
    /// submission cannot be reinterpreted as a later completion proof. A
    /// `None` result means the device is still working and does not perform a
    /// host or stream synchronization.
    pub fn query_event_completion(
        &self,
        event: &OwnedCudaEvent,
        receipt: CudaStreamSubmissionReceipt,
    ) -> Result<Option<CudaStreamCompletionProof>, CudaRuntimeError> {
        self.validate_submission_receipt(receipt)?;
        validate_same_device(
            "cuda_stream_event_completion",
            self.device_ordinal,
            event.device_ordinal(),
        )?;
        validate_event_submission_binding(
            event.recorded_stream_owner_instance_id(),
            event.recorded_submission_epoch(),
            self.owner_instance_id,
            receipt.submission_epoch,
        )?;
        if event.query_complete()? {
            Ok(Some(self.completion_proof_after_sync(receipt)))
        } else {
            Ok(None)
        }
    }

    /// Returns the immutable stream capability receipt when K13 is enabled.
    ///
    /// Token-paced builds deliberately retain the live query here. On the
    /// Windows SM70 product path it is also a measured driver-submit pacing
    /// boundary; caching it regresses steady decode even when an equivalent
    /// request-boundary reset probe is retained.
    pub fn compute_capability_receipt(&self) -> Result<CudaComputeCapability, CudaRuntimeError> {
        
        
        {
            Ok(cuda_device_properties(self.device_ordinal)?.compute_capability)
        }
    }

    /// Performs one device-reset/capability probe at a request boundary.
    ///
    /// The token-paced successor moves this live CUDA query before warm-request
    /// timing; its historical live capability receipt remains unchanged in the
    /// kernel-binding path. Other builds retain a zero-cost no-op so callers
    /// can share the recovery boundary without changing the default identity.
    pub fn paced_capability_probe(&self) -> Result<(), CudaRuntimeError> {
        
        Ok(())
    }

    pub(crate) fn raw(&self) -> Result<AugurCudaStream, CudaRuntimeError> {
        self.raw
            .map(NonNull::as_ptr)
            .ok_or_else(|| CudaRuntimeError::closed("stream"))
    }

    pub fn synchronize(&self) -> Result<(), CudaRuntimeError> {
        let raw = self.raw()?;

        // SAFETY: `raw` came from `self.raw()`, which only yields the handle
        // while the stream is open; `close`/`Drop` are the sole destroyers and
        // clear `self.raw` on success, so a destroyed handle is never reused.
        let result = unsafe { augur_cuda_sys::augur_cuda_stream_synchronize(raw) };

        CudaRuntimeError::from_result("cuda_stream_synchronize", result)?;
        record_stream_synchronization();
        Ok(())
    }

    pub fn wait_event(&self, event: &OwnedCudaEvent) -> Result<(), CudaRuntimeError> {
        validate_same_device(
            "cuda_stream_wait_event",
            self.device_ordinal,
            event.device_ordinal(),
        )?;
        // SAFETY: both handles come from their owners' `raw()` accessors, so
        // stream and event are open (created by `create`, destroyed only by
        // close/Drop which clear the slots); same-device was validated above.
        let result =
            unsafe { augur_cuda_sys::augur_cuda_stream_wait_event_v1(self.raw()?, event.raw()?) };
        CudaRuntimeError::from_result("cuda_stream_wait_event", result)
    }

    pub fn close_with_receipt(&mut self) -> Result<CudaStreamCloseReceipt, CudaRuntimeError> {
        let raw = self.raw.ok_or_else(|| CudaRuntimeError::closed("stream"))?;
        let close_epoch = next_cuda_stream_close_epoch()?;

        // SAFETY: `raw` is the open handle exclusively owned by `self`
        // (`create` is its only source). The shim deletes it only on success,
        // and we clear `self.raw` only then, so no double-destroy can occur.
        let result = unsafe { augur_cuda_sys::augur_cuda_stream_destroy(raw.as_ptr()) };

        CudaRuntimeError::from_result("cuda_stream_destroy", result)?;

        self.raw = None;
        Ok(CudaStreamCloseReceipt {
            device_ordinal: self.device_ordinal,
            stream_owner_instance_id: self.owner_instance_id,
            close_epoch,
            receipt_sha256: cuda_stream_close_receipt_sha256(
                self.device_ordinal,
                self.owner_instance_id,
                close_epoch,
            ),
        })
    }

    pub fn close(&mut self) -> Result<(), CudaRuntimeError> {
        self.close_with_receipt().map(|_| ())
    }

    /// Deliberately leak a stream whose terminal state cannot be proven.
    ///
    /// This is a last-resort failure boundary for owners that also quarantine
    /// every allocation referenced by the stream. It closes the safe Rust
    /// handle without calling `cudaStreamDestroy`, so no later code can submit
    /// through an indeterminate native stream incarnation.
    pub fn quarantine_after_unproven_completion(&mut self) {
        self.raw = None;
    }
}

impl Drop for OwnedCudaStream {
    fn drop(&mut self) {
        if self.raw.is_some() {
            let _ = self.close();
        }
    }
}

// ── OwnedCudaGraphExec ────────────────────────────────────────────────────

/// Owns a native graph-exec handle, not the buffers referenced by graph nodes.
///
/// The enclosing execution plan must retain captured addresses, weights and
/// scratch buffers through every replay and its completion. A successful capture
/// does not extend Rust borrows from the closure. `capture`/`launch` do not encode
/// that resource lifetime in their signatures (GAP-GRAPH).
pub struct OwnedCudaGraphExec {
    raw: Option<NonNull<augur_cuda_sys::AugurCudaGraphExecHandle>>,
    device_ordinal: i32,
    node_count: u64,
}

struct CudaGraphCaptureGuard<'a> {
    stream: &'a OwnedCudaStream,
    active: bool,
}

impl Drop for CudaGraphCaptureGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            // SAFETY: the guard only exists while a capture begun on
            // `self.stream` (borrowed for the guard's lifetime) is unfinished;
            // the shim rejects a null stream, so the closed-stream fallback
            // degrades to an error status instead of a dangling dereference.
            unsafe {
                let _ = augur_cuda_sys::augur_cuda_graph_capture_abort_v1(
                    self.stream.device_ordinal(),
                    self.stream.raw().unwrap_or(ptr::null_mut()),
                );
            }
        }
    }
}

impl core::fmt::Debug for OwnedCudaGraphExec {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("OwnedCudaGraphExec")
            .field("device_ordinal", &self.device_ordinal)
            .field("node_count", &self.node_count)
            .field("closed", &self.raw.is_none())
            .finish()
    }
}

impl OwnedCudaGraphExec {
    pub fn capture(
        stream: &OwnedCudaStream,
        submit: impl FnOnce() -> Result<(), CudaRuntimeError>,
    ) -> Result<Self, CudaRuntimeError> {
        // SAFETY: the handle comes from `stream.raw()` (open, owned by the
        // borrowed `OwnedCudaStream`); the shim re-validates that the ordinal
        // matches the stream's before beginning capture.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_graph_capture_begin_v1(stream.device_ordinal(), stream.raw()?)
        };
        CudaRuntimeError::from_result("cuda_graph_capture_begin", result)?;
        let mut guard = CudaGraphCaptureGuard {
            stream,
            active: true,
        };
        submit()?;
        let mut raw: augur_cuda_sys::AugurCudaGraphExec = ptr::null_mut();
        // SAFETY: `raw` is a live out-slot and the stream handle is the same
        // open handle capture began on (still borrowed for the whole capture);
        // on success the shim hands over sole ownership of the graph handle.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_graph_capture_end_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &mut raw,
            )
        };
        guard.active = false;
        CudaRuntimeError::from_result("cuda_graph_capture_end", result)?;
        let raw = NonNull::new(raw).ok_or_else(CudaRuntimeError::null_handle)?;
        let mut node_count = 0_u64;
        // SAFETY: `raw` was NonNull-checked from the successful capture_end
        // just above and has not been destroyed; `node_count` is a live
        // out-slot.
        let result =
            unsafe { augur_cuda_sys::augur_cuda_graph_node_count_v1(raw.as_ptr(), &mut node_count) };
        if let Err(error) = CudaRuntimeError::from_result("cuda_graph_node_count", result) {
            // SAFETY: `raw` is the graph handle this function exclusively owns
            // from capture_end; it has not been stored anywhere else, so
            // destroying it here cannot race another user.
            unsafe {
                let _ = augur_cuda_sys::augur_cuda_graph_destroy_v1(raw.as_ptr());
            }
            return Err(error);
        }
        if node_count == 0 {
            // SAFETY: as above — `raw` is still exclusively owned by this
            // function and escapes only via the `Self` constructed below.
            unsafe {
                let _ = augur_cuda_sys::augur_cuda_graph_destroy_v1(raw.as_ptr());
            }
            return Err(CudaRuntimeError::invalid_metadata("cuda_graph_node_count"));
        }
        Ok(Self {
            raw: Some(raw),
            device_ordinal: stream.device_ordinal(),
            node_count,
        })
    }

    pub const fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    pub const fn node_count(&self) -> u64 {
        self.node_count
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        validate_same_device(
            "cuda_graph_launch",
            stream.device_ordinal(),
            self.device_ordinal,
        )?;
        let raw = self
            .raw
            .map(NonNull::as_ptr)
            .ok_or_else(|| CudaRuntimeError::closed("graph"))?;
        // SAFETY: `raw` is open (checked Some above; only close/Drop destroy
        // it) and the stream handle comes from `stream.raw()`; same-device was
        // validated above and the shim re-checks the graph/stream ordinals.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_graph_launch_v1(self.device_ordinal, raw, stream.raw()?)
        };
        CudaRuntimeError::from_result("cuda_graph_launch", result)
    }

    pub fn close(&mut self) -> Result<(), CudaRuntimeError> {
        let raw = self.raw.ok_or_else(|| CudaRuntimeError::closed("graph"))?;
        // SAFETY: `raw` is the open handle exclusively owned by `self`; the
        // shim deletes it only on success and we clear `self.raw` only then,
        // so a failed destroy leaves the handle valid for a Drop retry.
        let result = unsafe { augur_cuda_sys::augur_cuda_graph_destroy_v1(raw.as_ptr()) };
        CudaRuntimeError::from_result("cuda_graph_destroy", result)?;
        self.raw = None;
        Ok(())
    }
}

impl Drop for OwnedCudaGraphExec {
    fn drop(&mut self) {
        if self.raw.is_some() {
            let _ = self.close();
        }
    }
}

fn validate_event_submission_binding(
    recorded_stream_owner_instance_id: Option<u64>,
    recorded_submission_epoch: Option<u64>,
    expected_stream_owner_instance_id: u64,
    expected_submission_epoch: u64,
) -> Result<(), CudaRuntimeError> {
    if recorded_stream_owner_instance_id != Some(expected_stream_owner_instance_id) {
        return Err(CudaRuntimeError::invalid_metadata(
            "cuda_stream_event_owner",
        ));
    }
    if recorded_submission_epoch != Some(expected_submission_epoch) {
        return Err(CudaRuntimeError::invalid_metadata(
            "cuda_stream_event_submission_epoch",
        ));
    }
    Ok(())
}

// ── OwnedCudaEvent ────────────────────────────────────────────────────────

pub struct OwnedCudaEvent {
    raw: Option<NonNull<AugurCudaEventHandle>>,
    device_ordinal: i32,
    /// Process-local stream owner that recorded the most recent event.  This
    /// is an ABA guard for non-blocking completion proofs; the native event
    /// handle alone is not a sufficient identity after stream recreation.
    recorded_stream_owner_instance_id: Cell<Option<u64>>,
    /// Submission epoch covered by the most recent record when that record
    /// was made through `record_for_submission`. A plain timing/event record
    /// deliberately leaves this unset and cannot mint a completion proof.
    recorded_submission_epoch: Cell<Option<u64>>,
}

impl core::fmt::Debug for OwnedCudaEvent {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("OwnedCudaEvent")
            .field("device_ordinal", &self.device_ordinal)
            .field("closed", &self.raw.is_none())
            .finish()
    }
}

impl OwnedCudaEvent {
    pub fn create(device_ordinal: i32) -> Result<Self, CudaRuntimeError> {
        validate_device_ordinal(device_ordinal)?;
        let mut raw: AugurCudaEvent = ptr::null_mut();
        // SAFETY: `raw` is a live out-slot and `device_ordinal` was validated
        // non-negative above; on success the shim transfers sole ownership of
        // the heap-allocated event handle to `Self`.
        let result = unsafe { augur_cuda_sys::augur_cuda_event_create(device_ordinal, &mut raw) };
        CudaRuntimeError::from_result("cuda_event_create", result)?;
        Ok(Self {
            raw: Some(NonNull::new(raw).ok_or_else(CudaRuntimeError::null_handle)?),
            device_ordinal,
            recorded_stream_owner_instance_id: Cell::new(None),
            recorded_submission_epoch: Cell::new(None),
        })
    }

    pub const fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    /// Process-local owner key used only for non-alias checks. Reports bind an
    /// address-neutral logical identity instead of exposing this value.
    pub fn owner_instance_key(&self) -> Result<usize, CudaRuntimeError> {
        Ok(self.raw()? as usize)
    }

    fn recorded_stream_owner_instance_id(&self) -> Option<u64> {
        self.recorded_stream_owner_instance_id.get()
    }

    fn recorded_submission_epoch(&self) -> Option<u64> {
        self.recorded_submission_epoch.get()
    }

    pub(crate) fn raw(&self) -> Result<AugurCudaEvent, CudaRuntimeError> {
        self.raw
            .map(NonNull::as_ptr)
            .ok_or_else(|| CudaRuntimeError::closed("event"))
    }

    pub fn record(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        validate_same_device(
            "cuda_event_record",
            stream.device_ordinal(),
            self.device_ordinal,
        )?;
        // A plain record is useful for timing and stream dependencies, but it
        // does not identify a submission receipt. Invalidate any old binding
        // before attempting the native overwrite so a failed record cannot
        // leave stale proof metadata usable.
        self.recorded_stream_owner_instance_id.set(None);
        self.recorded_submission_epoch.set(None);
        // SAFETY: both handles come from `raw()` accessors, so the event and
        // stream are open (close/Drop are the sole destroyers and clear the
        // slots); same-device was validated above and re-checked by the shim.
        let result = unsafe { augur_cuda_sys::augur_cuda_event_record(self.raw()?, stream.raw()?) };
        CudaRuntimeError::from_result("cuda_event_record", result)?;
        self.recorded_stream_owner_instance_id
            .set(Some(stream.owner_instance_id));
        Ok(())
    }

    /// Records this event as the terminal marker for an exact stream
    /// submission. Only an event recorded through this method may later be
    /// queried to mint a completion proof for that receipt.
    pub fn record_for_submission(
        &self,
        stream: &OwnedCudaStream,
        receipt: CudaStreamSubmissionReceipt,
    ) -> Result<(), CudaRuntimeError> {
        stream.validate_submission_receipt(receipt)?;
        validate_same_device(
            "cuda_event_record_submission",
            stream.device_ordinal(),
            self.device_ordinal,
        )?;
        // Invalidate any prior binding before attempting the native overwrite;
        // a failed record must fail closed rather than preserve stale proof
        // metadata.
        self.recorded_stream_owner_instance_id.set(None);
        self.recorded_submission_epoch.set(None);
        // SAFETY: both handles are open and same-device validation completed.
        // The receipt is validated against this exact stream owner.
        let result = unsafe { augur_cuda_sys::augur_cuda_event_record(self.raw()?, stream.raw()?) };
        CudaRuntimeError::from_result("cuda_event_record_submission", result)?;
        self.recorded_stream_owner_instance_id
            .set(Some(stream.owner_instance_id));
        self.recorded_submission_epoch
            .set(Some(receipt.submission_epoch));
        Ok(())
    }

    /// Polls the most recent event record without synchronizing the host.
    ///
    /// The native boundary translates `cudaErrorNotReady` into a successful
    /// `false` result. Every other CUDA error remains fail-closed.
    pub fn query_complete(&self) -> Result<bool, CudaRuntimeError> {
        let mut complete = 0_u8;
        // SAFETY: `self.raw()` only yields a live event handle and `complete`
        // is a writable one-byte out-slot for the duration of the call.
        let result =
            unsafe { augur_cuda_sys::augur_cuda_event_query_complete_v1(self.raw()?, &mut complete) };
        CudaRuntimeError::from_result("cuda_event_query_complete", result)?;
        match complete {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(CudaRuntimeError::invalid_metadata(
                "cuda_event_query_complete",
            )),
        }
    }

    /// Records and waits for this event after a sealed stream submission.
    ///
    /// Record and wait deliberately form one call. Splitting them would let a
    /// caller enqueue work through a previously copied non-owning view between
    /// the event record and proof publication.
    pub fn record_and_synchronize_submission(
        &self,
        stream: &OwnedCudaStream,
        receipt: CudaStreamSubmissionReceipt,
    ) -> Result<CudaStreamCompletionProof, CudaRuntimeError> {
        self.record_for_submission(stream, receipt)?;
        // `receipt` can only be obtained after its D2D batch is sealed, so the
        // event is ordered after every submission covered by that receipt.
        self.synchronize()?;
        Ok(stream.completion_proof_after_sync(receipt))
    }

    pub fn synchronize(&self) -> Result<(), CudaRuntimeError> {
        // SAFETY: `self.raw()` only yields the handle while the event is open;
        // close/Drop are the sole destroyers and clear the slot on success.
        let result = unsafe { augur_cuda_sys::augur_cuda_event_synchronize(self.raw()?) };
        CudaRuntimeError::from_result("cuda_event_synchronize", result)?;
        record_event_synchronization();
        Ok(())
    }

    pub fn elapsed_ms(&self, stopped: &Self) -> Result<f32, CudaRuntimeError> {
        validate_same_device(
            "cuda_event_elapsed_ms",
            self.device_ordinal,
            stopped.device_ordinal,
        )?;
        let mut milliseconds = 0.0_f32;
        // SAFETY: both event handles are open (via `raw()`), `milliseconds` is
        // a live stack out-slot, and same-device was validated above.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_event_elapsed_ms(self.raw()?, stopped.raw()?, &mut milliseconds)
        };
        CudaRuntimeError::from_result("cuda_event_elapsed_ms", result)?;
        Ok(milliseconds)
    }

    pub fn close(&mut self) -> Result<(), CudaRuntimeError> {
        let raw = self.raw.ok_or_else(|| CudaRuntimeError::closed("event"))?;
        // SAFETY: `raw` is the open handle exclusively owned by `self`; the
        // shim deletes it only on success and we clear `self.raw` only then,
        // so no double-destroy and no dangling reuse are possible.
        let result = unsafe { augur_cuda_sys::augur_cuda_event_destroy(raw.as_ptr()) };
        CudaRuntimeError::from_result("cuda_event_destroy", result)?;
        self.raw = None;
        self.recorded_stream_owner_instance_id.set(None);
        self.recorded_submission_epoch.set(None);
        Ok(())
    }
}

impl Drop for OwnedCudaEvent {
    fn drop(&mut self) {
        if self.raw.is_some() {
            let _ = self.close();
        }
    }
}

// ── OwnedCudaBuffer ───────────────────────────────────────────────────────

pub struct OwnedCudaHostBuffer {
    raw: Option<NonNull<u8>>,
    device_ordinal: i32,
    len_bytes: usize,
}

/// Page-locked host storage with a stable kernel-visible CUDA address.
///
/// Host mutation is only legal before submission or after a covering stream
/// completion. The owner itself does not infer that boundary; callers must
/// retain it through the enclosing request terminal proof.
/// Mapped pinned memory is addressable by both CPU and GPU.
///
/// Rust slice borrows exclude Rust-side aliases only; a previously returned
/// device view can still refer to this storage. CPU access requires completion
/// of conflicting GPU work and exclusion of a new submission during the borrow.
/// This condition is not encoded by the current safe slice/view API (GAP-MAPPED).
pub struct OwnedCudaMappedHostBuffer {
    host_raw: Option<NonNull<u8>>,
    device_raw: Option<CudaDevicePtr>,
    device_ordinal: i32,
    allocation_owner_instance_id: u64,
    len_bytes: usize,
}

impl core::fmt::Debug for OwnedCudaMappedHostBuffer {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("OwnedCudaMappedHostBuffer")
            .field("device_ordinal", &self.device_ordinal)
            .field("len_bytes", &self.len_bytes)
            .field("closed", &self.host_raw.is_none())
            .finish()
    }
}

impl OwnedCudaMappedHostBuffer {
    pub fn allocate(device_ordinal: i32, len_bytes: usize) -> Result<Self, CudaRuntimeError> {
        validate_device_ordinal(device_ordinal)?;
        if len_bytes == 0 {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        let allocation_owner_instance_id = next_cuda_buffer_owner_id()?;
        let mut host_raw: *mut c_void = ptr::null_mut();
        let mut device_raw = 0_u64;
        // SAFETY: both out-slots are live, the ordinal and non-zero extent are
        // validated, and this owner is the sole releaser of the returned host
        // allocation. The device token aliases those bytes and is never freed
        // through the device allocator.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_host_alloc_mapped_v1(
                device_ordinal,
                len_bytes,
                &mut host_raw,
                &mut device_raw,
            )
        };
        CudaRuntimeError::from_result("cuda_host_alloc_mapped", result)?;
        let host_raw =
            NonNull::new(host_raw.cast::<u8>()).ok_or_else(CudaRuntimeError::null_handle)?;
        if device_raw == 0 {
            // SAFETY: allocation succeeded and `host_raw` is still solely
            // owned here. No device submission can exist because no view was
            // published. Use the ordinary pinned release for this rollback.
            let _ = unsafe {
                augur_cuda_sys::augur_cuda_host_free_pinned_v1(device_ordinal, host_raw.as_ptr().cast())
            };
            return Err(CudaRuntimeError::null_handle());
        }
        Ok(Self {
            host_raw: Some(host_raw),
            device_raw: Some(CudaDevicePtr {
                raw: device_raw,
                device_ordinal,
            }),
            device_ordinal,
            allocation_owner_instance_id,
            len_bytes,
        })
    }

    pub const fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    pub fn as_slice(&self) -> Result<&[u8], CudaRuntimeError> {
        let raw = self
            .host_raw
            .ok_or_else(|| CudaRuntimeError::closed("mapped host buffer"))?;
        // SAFETY: `raw` spans `len_bytes` initialized bytes and `&self`
        // prevents Rust-side mutable aliasing for the returned lifetime. GPU
        // writes through a retained device view are not excluded by this
        // borrow; the owner protocol must do so (GAP-MAPPED).
        Ok(unsafe { core::slice::from_raw_parts(raw.as_ptr(), self.len_bytes) })
    }

    pub fn as_mut_slice(&mut self) -> Result<&mut [u8], CudaRuntimeError> {
        let mut raw = self
            .host_raw
            .ok_or_else(|| CudaRuntimeError::closed("mapped host buffer"))?;
        // SAFETY: `&mut self` gives exclusive access to the full live mapped
        // allocation on the Rust side. GPU access must additionally be fenced
        // and excluded for the borrow duration; the safe signature does not
        // enforce that condition (GAP-MAPPED).
        Ok(unsafe { core::slice::from_raw_parts_mut(raw.as_mut(), self.len_bytes) })
    }

    pub fn device_view(
        &self,
        offset_bytes: usize,
        len_bytes: usize,
    ) -> Result<CudaBufferView, CudaRuntimeError> {
        let end = offset_bytes
            .checked_add(len_bytes)
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        if end > self.len_bytes {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        let ptr = self
            .device_raw
            .ok_or_else(|| CudaRuntimeError::closed("mapped host buffer"))?;
        Ok(CudaBufferView {
            ptr,
            allocation_owner_instance_id: self.allocation_owner_instance_id,
            allocation_len_bytes: self.len_bytes,
            offset_bytes,
            len_bytes,
        })
    }

    pub fn close(&mut self) -> Result<(), CudaRuntimeError> {
        let raw = self
            .host_raw
            .take()
            .ok_or_else(|| CudaRuntimeError::closed("mapped host buffer"))?;
        // SAFETY: `raw` is the unique host owner returned by the mapped
        // allocator. The mapped-specific release fences every device consumer
        // before invalidating the aliased device token. On fence failure the
        // allocation remains owned and is restored into `self` below.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_host_free_mapped_sync_v1(self.device_ordinal, raw.as_ptr().cast())
        };
        if result.status != AUGUR_CUDA_STATUS_OK {
            self.host_raw = Some(raw);
            return CudaRuntimeError::from_result("cuda_host_free_mapped", result);
        }
        self.device_raw = None;
        Ok(())
    }
}

impl Drop for OwnedCudaMappedHostBuffer {
    fn drop(&mut self) {
        if self.host_raw.is_some() {
            let _ = self.close();
        }
    }
}

impl core::fmt::Debug for OwnedCudaHostBuffer {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("OwnedCudaHostBuffer")
            .field("device_ordinal", &self.device_ordinal)
            .field("len_bytes", &self.len_bytes)
            .field("closed", &self.raw.is_none())
            .finish()
    }
}

impl OwnedCudaHostBuffer {
    pub fn allocate(device_ordinal: i32, len_bytes: usize) -> Result<Self, CudaRuntimeError> {
        validate_device_ordinal(device_ordinal)?;
        if len_bytes == 0 {
            return Ok(Self {
                raw: None,
                device_ordinal,
                len_bytes,
            });
        }
        let mut raw: *mut c_void = ptr::null_mut();
        // SAFETY: `raw` is a live out-slot, `device_ordinal` was validated
        // non-negative, and `len_bytes > 0` (the zero-length case returned
        // early); on success the pinned allocation is exclusively owned by
        // `Self` and freed only in `close`.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_host_alloc_pinned_v1(device_ordinal, len_bytes, &mut raw)
        };
        CudaRuntimeError::from_result("cuda_host_alloc_pinned", result)?;
        Ok(Self {
            raw: Some(NonNull::new(raw.cast::<u8>()).ok_or_else(CudaRuntimeError::null_handle)?),
            device_ordinal,
            len_bytes,
        })
    }

    pub const fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }
    pub const fn len_bytes(&self) -> usize {
        self.len_bytes
    }
    pub const fn is_empty(&self) -> bool {
        self.len_bytes == 0
    }

    pub fn as_slice(&self) -> Result<&[u8], CudaRuntimeError> {
        if self.len_bytes == 0 {
            return Ok(&[]);
        }
        let raw = self
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?;
        // SAFETY: `raw` was allocated with exactly `self.len_bytes` by
        // augur_cuda_host_alloc_pinned_v1 in `allocate` and lives until
        // close/Drop; `u8` needs no alignment and the `&self` borrow rules out
        // mutable aliasing. The native allocation ABI zero-initializes all
        // `self.len_bytes` before publishing the pointer, so every byte is a
        // valid initialized `u8` before this shared slice can be constructed.
        Ok(unsafe { core::slice::from_raw_parts(raw.as_ptr(), self.len_bytes) })
    }

    pub fn as_mut_slice(&mut self) -> Result<&mut [u8], CudaRuntimeError> {
        if self.len_bytes == 0 {
            return Ok(&mut []);
        }
        let mut raw = self
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?;
        // SAFETY: same provenance as `as_slice` (pinned allocation of
        // `len_bytes` made in `allocate`, live until close/Drop); `&mut self`
        // gives exclusivity for the returned slice. The native allocation ABI
        // zero-initializes the full allocation before publishing its pointer,
        // so the fresh buffer already contains initialized `u8` values.
        Ok(unsafe { core::slice::from_raw_parts_mut(raw.as_mut(), self.len_bytes) })
    }

    pub fn close(&mut self) -> Result<(), CudaRuntimeError> {
        let raw = self
            .raw
            .take()
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?;
        // SAFETY: `raw` was produced by augur_cuda_host_alloc_pinned_v1 in
        // `allocate` and was just taken out of `self.raw`, so this is the sole
        // free; on a failed status we restore `self.raw` for a Drop retry.
        // Callers staging async copies (the H2D slot/lane owners) must fence
        // the stream before this drop-time free can run.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_host_free_pinned_v1(self.device_ordinal, raw.as_ptr().cast())
        };
        if result.status != AUGUR_CUDA_STATUS_OK {
            self.raw = Some(raw);
        }
        CudaRuntimeError::from_result("cuda_host_free_pinned", result)
    }
}

impl Drop for OwnedCudaHostBuffer {
    fn drop(&mut self) {
        if self.raw.is_some() {
            let _ = self.close();
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CudaD2dOwnerState {
    Idle,
    Pending {
        stream_owner_instance_id: u64,
        submission_epoch: u64,
    },
}

impl CudaD2dOwnerState {
    const fn pending(self) -> bool {
        !matches!(self, Self::Idle)
    }

    fn accepts_submission(
        self,
        receipt: CudaStreamSubmissionReceipt,
    ) -> Result<(), CudaRuntimeError> {
        match self {
            Self::Idle => Ok(()),
            Self::Pending {
                stream_owner_instance_id,
                submission_epoch,
            } if stream_owner_instance_id == receipt.stream_owner_instance_id
                && submission_epoch == receipt.submission_epoch =>
            {
                Ok(())
            }
            Self::Pending { .. } => Err(CudaRuntimeError::invalid_metadata(
                "cuda_d2d_owner_already_pending",
            )),
        }
    }

    fn mark_pending(&mut self, receipt: CudaStreamSubmissionReceipt) {
        debug_assert!(self.accepts_submission(receipt).is_ok());
        *self = Self::Pending {
            stream_owner_instance_id: receipt.stream_owner_instance_id,
            submission_epoch: receipt.submission_epoch,
        };
    }

    const fn matches_proof(self, proof: &CudaStreamCompletionProof) -> bool {
        matches!(
            self,
            Self::Pending {
                stream_owner_instance_id,
                submission_epoch,
            } if stream_owner_instance_id == proof.stream_owner_instance_id
                && submission_epoch == proof.submission_epoch
        )
    }

    const fn matches_submission(self, receipt: CudaStreamSubmissionReceipt) -> bool {
        matches!(
            self,
            Self::Pending {
                stream_owner_instance_id,
                submission_epoch,
            } if stream_owner_instance_id == receipt.stream_owner_instance_id
                && submission_epoch == receipt.submission_epoch
        )
    }

    const fn covered_by(self, proof: &CudaStreamCompletionProof) -> bool {
        matches!(
            self,
            Self::Pending {
                stream_owner_instance_id,
                submission_epoch,
            } if stream_owner_instance_id == proof.stream_owner_instance_id
                && submission_epoch <= proof.submission_epoch
        )
    }

    fn release(&mut self) {
        debug_assert!(self.pending());
        *self = Self::Idle;
    }

    fn observe_submission_result(
        self,
        operation: &'static str,
        result: AugurCudaResultV1,
    ) -> Result<(), CudaRuntimeError> {
        debug_assert!(self.pending());
        CudaRuntimeError::from_result(operation, result)
    }
}

pub struct OwnedCudaBuffer {
    raw: Option<CudaDevicePtr>,
    device_ordinal: i32,
    allocation_owner_instance_id: u64,
    len_bytes: usize,
    d2d_state: CudaD2dOwnerState,
}





impl core::fmt::Debug for OwnedCudaBuffer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OwnedCudaBuffer")
            .field("device_ordinal", &self.device_ordinal)
            .field("len_bytes", &self.len_bytes)
            .field("closed", &self.raw.is_none())
            .field("d2d_pending", &self.d2d_state.pending())
            .finish()
    }
}

impl OwnedCudaBuffer {
    pub fn allocate(device_ordinal: i32, len_bytes: usize) -> Result<Self, CudaRuntimeError> {
        validate_device_ordinal(device_ordinal)?;

        if len_bytes == 0 {
            return Ok(Self {
                raw: None,
                device_ordinal,
                allocation_owner_instance_id: 0,
                len_bytes: 0,
                d2d_state: CudaD2dOwnerState::Idle,
            });
        }

        let allocation_owner_instance_id = next_cuda_buffer_owner_id()?;
        let mut raw = 0_u64;

        // SAFETY: `raw` is a live out-slot, `device_ordinal` was validated
        // non-negative, and `len_bytes > 0` (zero-length returned early); on
        // success the returned device address (never a host pointer) is
        // exclusively owned by `Self` and freed only in `close`.
        let result =
            unsafe { augur_cuda_sys::augur_cuda_device_alloc(device_ordinal, len_bytes, &mut raw) };

        CudaRuntimeError::from_result("cuda_device_alloc", result)?;

        if raw == 0 {
            return Err(CudaRuntimeError::null_handle());
        }

        CUDA_DEVICE_ALLOCATION_COUNTERS.record_allocation(len_bytes);

        Ok(Self {
            raw: Some(CudaDevicePtr {
                raw,
                device_ordinal,
            }),
            device_ordinal,
            allocation_owner_instance_id,
            len_bytes,
            d2d_state: CudaD2dOwnerState::Idle,
        })
    }

    pub const fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    pub const fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    pub const fn is_empty(&self) -> bool {
        self.len_bytes == 0
    }

    pub const fn has_pending_d2d(&self) -> bool {
        self.d2d_state.pending()
    }

    /// Initializes the complete device allocation to canonical zero and waits
    /// for `stream` before returning.
    pub fn zero_sync(&mut self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        // SAFETY: the exclusive owner borrow prevents simultaneous Rust access.
        unsafe { self.zero_sync_unchecked(stream) }
    }

    /// Zeros exactly `len_bytes` at the beginning of the allocation, preserving
    /// the remaining bytes. Requires an exclusive owner and waits for completion.
    pub fn zero_prefix_sync(
        &mut self,
        stream: &OwnedCudaStream,
        len_bytes: usize,
    ) -> Result<(), CudaRuntimeError> {
        // SAFETY: exclusive owner borrow; helper checks the complete prefix extent.
        unsafe { self.zero_prefix_sync_unchecked(stream, len_bytes) }
    }

    /// Zeros the complete allocation and drains `stream`, without a host zero slab.
    ///
    /// # Safety
    /// All aliases, including retained graph/kernel bindings, must be idle or
    /// ordered on `stream`. No other stream may access these bytes until this
    /// call returns. The owner must remain live throughout; this does not release
    /// pending D2D ownership or make concurrent alias access safe.
    pub unsafe fn zero_sync_unchecked(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<(), CudaRuntimeError> {
        // SAFETY: the caller supplies the documented alias-ordering guarantee.
        unsafe { self.zero_prefix_sync_unchecked(stream, self.len_bytes) }
    }

    // Internal shared implementation; only full-allocation zero exposes an
    // unchecked public entrypoint. Prefix zero retains its exclusive-owner API.
    unsafe fn zero_prefix_sync_unchecked(
        &self,
        stream: &OwnedCudaStream,
        len_bytes: usize,
    ) -> Result<(), CudaRuntimeError> {
        self.ensure_d2d_idle("cuda_memset_zero")?;
        validate_same_device(
            "cuda_memset_zero",
            stream.device_ordinal(),
            self.device_ordinal(),
        )?;
        if len_bytes > self.len_bytes {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        if len_bytes == 0 {
            return Ok(());
        }
        let ptr = self.raw.ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        let raw_stream = stream.raw()?;
        // SAFETY: the caller guarantees alias ordering/exclusion. The allocation
        // token and checked prefix extent come from this live owner; the entrypoint
        // synchronizes the validated same-device stream before returning.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memset_zero_sync_v1(
                self.device_ordinal,
                raw_stream,
                ptr.raw,
                len_bytes,
            )
        };
        CudaRuntimeError::from_result("cuda_memset_zero", result)?;
        record_stream_synchronization();
        Ok(())
    }

    pub fn device_ptr(&self) -> Option<CudaDevicePtr> {
        self.raw
    }

    pub fn view(
        &self,
        offset_bytes: usize,
        len_bytes: usize,
    ) -> Result<CudaBufferView, CudaRuntimeError> {
        self.ensure_d2d_idle("cuda_buffer_view")?;
        let end = offset_bytes
            .checked_add(len_bytes)
            .ok_or_else(CudaRuntimeError::range_overflow)?;

        if end > self.len_bytes {
            return Err(CudaRuntimeError::out_of_bounds());
        }

        let ptr = self.raw.ok_or_else(|| CudaRuntimeError::closed("buffer"))?;

        Ok(CudaBufferView {
            ptr,
            allocation_owner_instance_id: self.allocation_owner_instance_id,
            allocation_len_bytes: self.len_bytes,
            offset_bytes,
            len_bytes,
        })
    }

    #[deprecated(note = "use copy_from_host_sync; this operation waits for the stream")]
    pub fn copy_from_host(
        &self,
        stream: &OwnedCudaStream,
        src: &[u8],
    ) -> Result<(), CudaRuntimeError> {
        self.copy_from_host_sync(stream, src)
    }

    /// Copies host bytes to this buffer and synchronizes `stream` before returning.
    pub fn copy_from_host_sync(
        &self,
        stream: &OwnedCudaStream,
        src: &[u8],
    ) -> Result<(), CudaRuntimeError> {
        self.ensure_d2d_idle("cuda_memcpy_h2d")?;
        validate_same_device(
            "cuda_memcpy_h2d",
            stream.device_ordinal(),
            self.device_ordinal(),
        )?;

        if src.len() > self.len_bytes {
            return Err(CudaRuntimeError::out_of_bounds());
        }

        if src.is_empty() {
            return Ok(());
        }

        let ptr = self.raw.ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        let raw_stream = stream.raw()?;

        // SAFETY: `ptr` is the device address allocated with `self.len_bytes`
        // by augur_cuda_device_alloc in `allocate`, still live (checked Some
        // above). Host `src` is a borrowed slice valid for `src.len()`, and
        // `src.len() <= len_bytes` was checked. The sync_v2 call drains the
        // stream before returning, so the host borrow outlives the DMA read;
        // same-device was validated above. Host and device pointers travel in
        // distinct parameters, so provenance is never conflated.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memcpy_h2d_sync_v2(
                self.device_ordinal,
                raw_stream,
                ptr.raw,
                0,
                self.len_bytes,
                src.as_ptr(),
                src.len(),
                src.len(),
            )
        };

        CudaRuntimeError::from_result("cuda_memcpy_h2d", result)?;
        record_synchronous_h2d(src.len());
        Ok(())
    }

    #[deprecated(note = "use copy_to_host_sync; this operation waits for the stream")]
    pub fn copy_to_host(
        &self,
        stream: &OwnedCudaStream,
        dst: &mut [u8],
    ) -> Result<(), CudaRuntimeError> {
        self.copy_to_host_sync(stream, dst)
    }

    /// Copies device bytes to `dst` and synchronizes `stream` before returning.
    pub fn copy_to_host_sync(
        &self,
        stream: &OwnedCudaStream,
        dst: &mut [u8],
    ) -> Result<(), CudaRuntimeError> {
        self.ensure_d2d_idle("cuda_memcpy_d2h")?;
        validate_same_device(
            "cuda_memcpy_d2h",
            stream.device_ordinal(),
            self.device_ordinal(),
        )?;

        if dst.len() > self.len_bytes {
            return Err(CudaRuntimeError::out_of_bounds());
        }

        if dst.is_empty() {
            return Ok(());
        }

        let ptr = self.raw.ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        let raw_stream = stream.raw()?;

        // SAFETY: device `ptr` spans `self.len_bytes` (augur_cuda_device_alloc in
        // `allocate`, still live) and `dst.len() <= len_bytes` was checked.
        // `dst` is an exclusive borrow valid for `dst.len()`, and the sync_v2
        // call drains the stream before returning, so the DMA write ends
        // before the borrow does; same-device was validated above.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memcpy_d2h_sync_v2(
                self.device_ordinal,
                raw_stream,
                dst.as_mut_ptr(),
                dst.len(),
                ptr.raw,
                0,
                self.len_bytes,
                dst.len(),
            )
        };

        CudaRuntimeError::from_result("cuda_memcpy_d2h", result)?;
        record_synchronous_d2h(dst.len());
        Ok(())
    }

    /// Copies this buffer to host and returns proof that all work ordered
    /// before `receipt` completed at the same synchronous D2H boundary.
    pub fn copy_to_host_sync_with_submission_proof(
        &self,
        stream: &OwnedCudaStream,
        dst: &mut [u8],
        receipt: CudaStreamSubmissionReceipt,
    ) -> Result<CudaStreamCompletionProof, CudaRuntimeError> {
        stream.validate_submission_receipt(receipt)?;
        self.copy_to_host_sync(stream, dst)?;
        Ok(stream.completion_proof_after_sync(receipt))
    }

    

    #[deprecated(note = "use copy_range_to_host_sync; this operation waits for the stream")]
    pub fn copy_range_to_host(
        &self,
        stream: &OwnedCudaStream,
        source_offset_bytes: usize,
        dst: &mut [u8],
    ) -> Result<(), CudaRuntimeError> {
        self.copy_range_to_host_sync(stream, source_offset_bytes, dst)
    }

    /// Copies a device range to `dst` and synchronizes `stream` before returning.
    pub fn copy_range_to_host_sync(
        &self,
        stream: &OwnedCudaStream,
        source_offset_bytes: usize,
        dst: &mut [u8],
    ) -> Result<(), CudaRuntimeError> {
        self.ensure_d2d_idle("cuda_memcpy_d2h")?;
        validate_same_device(
            "cuda_memcpy_d2h",
            stream.device_ordinal(),
            self.device_ordinal(),
        )?;
        let source_end = source_offset_bytes
            .checked_add(dst.len())
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        if source_end > self.len_bytes {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        if dst.is_empty() {
            return Ok(());
        }
        let ptr = self.raw.ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        // SAFETY: `source_offset_bytes + dst.len() <= self.len_bytes` was
        // checked overflow-free above and `ptr` spans `len_bytes` (allocated
        // in `allocate`, still live), so the offset read stays in bounds.
        // `dst` is an exclusive borrow and the sync_v2 call drains the stream
        // before returning; same-device was validated above.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memcpy_d2h_sync_v2(
                self.device_ordinal,
                stream.raw()?,
                dst.as_mut_ptr(),
                dst.len(),
                ptr.raw,
                source_offset_bytes,
                self.len_bytes,
                dst.len(),
            )
        };
        CudaRuntimeError::from_result("cuda_memcpy_d2h", result)?;
        record_synchronous_d2h(dst.len());
        Ok(())
    }

    /// Copies a host range into this device allocation and synchronizes
    /// `stream` before returning. This is intentionally a blocking boundary:
    /// prefix snapshot restore is TTFT work and must never enter steady decode.
    pub fn copy_range_from_host_sync(
        &self,
        stream: &OwnedCudaStream,
        destination_offset_bytes: usize,
        src: &[u8],
    ) -> Result<(), CudaRuntimeError> {
        self.ensure_d2d_idle("cuda_memcpy_h2d")?;
        validate_same_device(
            "cuda_memcpy_h2d",
            stream.device_ordinal(),
            self.device_ordinal,
        )?;
        let destination_end = destination_offset_bytes
            .checked_add(src.len())
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        if destination_end > self.len_bytes {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        if src.is_empty() {
            return Ok(());
        }
        let ptr = self.raw.ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        // SAFETY: `destination_offset_bytes + src.len() <= self.len_bytes`
        // was checked overflow-free above and `ptr` spans the allocation
        // retained by this owner. The synchronous shim drains the stream
        // before returning, so the borrowed host source remains live for the
        // complete DMA; same-device was validated above.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memcpy_h2d_sync_v2(
                self.device_ordinal,
                stream.raw()?,
                ptr.raw,
                destination_offset_bytes,
                self.len_bytes,
                src.as_ptr(),
                src.len(),
                src.len(),
            )
        };
        CudaRuntimeError::from_result("cuda_memcpy_h2d", result)?;
        record_synchronous_h2d(src.len());
        Ok(())
    }

    /// Copies a range to host and returns proof that all work ordered before
    /// `receipt` completed at the same synchronous D2H boundary.
    pub fn copy_range_to_host_sync_with_submission_proof(
        &self,
        stream: &OwnedCudaStream,
        source_offset_bytes: usize,
        dst: &mut [u8],
        receipt: CudaStreamSubmissionReceipt,
    ) -> Result<CudaStreamCompletionProof, CudaRuntimeError> {
        stream.validate_submission_receipt(receipt)?;
        self.copy_range_to_host_sync(stream, source_offset_bytes, dst)?;
        Ok(stream.completion_proof_after_sync(receipt))
    }

    /// Enqueues a device-to-device copy from a non-owning view.
    ///
    /// Prefer [`CudaD2dBatch`] whenever both allocation owners are available.
    /// This low-level entry point cannot retain either allocation or prove
    /// stream completion.
    ///
    /// # Safety
    ///
    /// The caller must keep the exact source allocation incarnation,
    /// destination allocation and `stream` alive and prevent either allocation
    /// from being freed, reused or mutated from another stream until all work
    /// ordered after this copy has reached a terminal completion proof. The
    /// copied `CudaBufferView` alone does not carry that lifetime.
    pub unsafe fn copy_from_device_async_unchecked(
        &self,
        stream: &OwnedCudaStream,
        destination_offset_bytes: usize,
        source: CudaBufferView,
    ) -> Result<(), CudaRuntimeError> {
        self.ensure_d2d_idle("cuda_memcpy_d2d")?;
        validate_same_device(
            "cuda_memcpy_d2d",
            stream.device_ordinal(),
            self.device_ordinal,
        )?;
        validate_same_device(
            "cuda_memcpy_d2d",
            self.device_ordinal,
            source.device_ordinal(),
        )?;
        let destination_end = destination_offset_bytes
            .checked_add(source.len_bytes())
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        if destination_end > self.len_bytes || source.len_bytes() == 0 {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        let destination = self.raw.ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        let source_address = source
            .device_ptr()
            .raw_value()
            .checked_add(
                u64::try_from(source.offset_bytes())
                    .map_err(|_| CudaRuntimeError::range_overflow())?,
            )
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        // SAFETY: the destination spans `self.len_bytes` (allocated in
        // `allocate`, checked Some above) and `destination_offset_bytes +
        // source.len_bytes() <= len_bytes` was checked; `source_address` stays
        // inside the source allocation per the view invariants upheld by
        // `OwnedCudaBuffer::view`/`subview`, and the shim re-checks both
        // extents. Both sides are device pointers on the validated same
        // device. The submit is ASYNC and nothing here fences either
        // allocation against drop before the stream drains — see findings.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memcpy_d2d_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                destination.raw,
                destination_offset_bytes,
                self.len_bytes,
                source_address,
                0,
                source.len_bytes(),
                source.len_bytes(),
            )
        };
        CudaRuntimeError::from_result("cuda_memcpy_d2d", result)
    }

    /// Submit 1..128 independent aligned copies in one bounded GPU launch.
    ///
    /// # Safety
    /// Retain every source/destination allocation incarnation and the stream
    /// through a terminal drain, also on error. All writes and consumers must
    /// be ordered on this stream. Like the scalar unchecked helper this does
    /// not retain owners or clear their pending state. Native admission rejects
    /// cross-copy dependencies; it is not memmove and cannot implement a shift.
    pub unsafe fn copy_independent_batch_async_unchecked(
        stream: &OwnedCudaStream,
        copies: &[(&OwnedCudaBuffer, usize, CudaBufferView)],
    ) -> Result<(), CudaRuntimeError> {
        const OP: &str = "cuda_d2d_independent_batch";
        if copies.is_empty() || copies.len() > 128 {
            return Err(CudaRuntimeError::invalid_metadata(OP));
        }
        let mut entries = [augur_cuda_sys::AugurCudaD2dCopyEntryV1::default(); 128];
        for (entry, &(destination, offset, source)) in entries.iter_mut().zip(copies) {
            destination.ensure_d2d_idle(OP)?;
            validate_same_device(OP, stream.device_ordinal(), destination.device_ordinal)?;
            validate_same_device(OP, destination.device_ordinal, source.device_ordinal())?;
            let end = offset.checked_add(source.len_bytes()).ok_or_else(CudaRuntimeError::range_overflow)?;
            if source.len_bytes() == 0 || end > destination.len_bytes {
                return Err(CudaRuntimeError::out_of_bounds());
            }
            let target = destination.view(offset, source.len_bytes())?;
            entry.source = source.checked_address(OP)?;
            entry.destination = target.checked_address(OP)?;
            entry.bytes = source.len_bytes() as u64;
        }
        // SAFETY: checked device identities and extents above; native checks
        // alignment, overflow and every cross alias before launching. Caller
        // retains owners and drains even after a failed asynchronous submit.
        let result = unsafe { augur_cuda_sys::augur_cuda_memcpy_d2d_independent_batch_submit_v1(
            stream.device_ordinal(), stream.raw()?, entries.as_ptr(), copies.len() as u32,
        ) };
        CudaRuntimeError::from_result(OP, result)
    }

    /// One strided D2D enqueue without changing the caller's owner state.
    ///
    /// # Safety
    /// Retain both exact allocation incarnations and the stream until a
    /// terminal drain, including after a submission error. All source writes
    /// and destination consumers must be ordered on this same stream.
    pub unsafe fn copy_from_device_2d_async_unchecked(
        &self, stream: &OwnedCudaStream, destination_offset_bytes: usize,
        destination_pitch_bytes: usize, source: CudaBufferView,
        source_pitch_bytes: usize, width_bytes: usize, height: usize,
    ) -> Result<(), CudaRuntimeError> {
        const OP: &str = "cuda_memcpy_d2d_2d";
        self.ensure_d2d_idle(OP)?;
        validate_same_device(OP, stream.device_ordinal(), self.device_ordinal)?;
        validate_same_device(OP, self.device_ordinal, source.device_ordinal())?;
        source.checked_range(OP)?;
        let dst_span = checked_2d_footprint(OP, width_bytes, height, destination_pitch_bytes)?;
        let src_span = checked_2d_footprint(OP, width_bytes, height, source_pitch_bytes)?;
        let end = destination_offset_bytes.checked_add(dst_span)
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        if end > self.len_bytes || src_span > source.len_bytes() {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        let destination = self.raw.ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        let source_address = source.checked_address(OP)?;
        if destination.raw == source.device_ptr().raw_value()
            && destination_offset_bytes < source.offset_bytes().saturating_add(src_span)
            && source.offset_bytes() < end {
            return Err(CudaRuntimeError::validation(OP, CudaRuntimeErrorKind::OverlappingDeviceRanges));
        }
        // SAFETY: pitches, extents, aliasing and device are checked above;
        // the caller supplies stream-ordered ownership through terminal drain.
        let result = unsafe { augur_cuda_sys::augur_cuda_memcpy_d2d_2d_submit_v1(
            self.device_ordinal, stream.raw()?, destination.raw,
            destination_offset_bytes, self.len_bytes, destination_pitch_bytes,
            source_address, 0, source.len_bytes(), source_pitch_bytes, width_bytes, height,
        ) };
        CudaRuntimeError::from_result(OP, result)
    }

    /// Bitwise left-shift of complete u32 rows, leaving the vacated tail intact.
    /// Does not allocate, synchronize, or convert floating-point values.
    ///
    /// # Safety
    /// Keep this exact allocation and stream alive until completion, and order
    /// every other access to the allocation before/after this stream operation.
    /// As with `copy_from_device_async_unchecked`, this does not retain owners.
    pub unsafe fn shift_rows_u32_async_unchecked(
        &self,
        stream: &OwnedCudaStream,
        row_bytes: usize,
        retained_rows: usize,
        shift_rows: usize,
    ) -> Result<(), CudaRuntimeError> {
        self.ensure_d2d_idle("cuda_shift_rows_u32")?;
        validate_same_device("cuda_shift_rows_u32", stream.device_ordinal(), self.device_ordinal)?;
        let buffer = self.raw.ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        if row_bytes == 0 || row_bytes % 4 != 0 || self.len_bytes % row_bytes != 0
            || shift_rows > self.len_bytes / row_bytes
            || retained_rows > self.len_bytes / row_bytes - shift_rows
        {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        // SAFETY: complete allocation extent and same-device live stream are
        // checked here; the caller supplies exclusive ordered lifetime.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_shift_rows_u32_submit_v1(
                self.device_ordinal, stream.raw()?, buffer.raw, self.len_bytes,
                row_bytes, retained_rows, shift_rows,
            )
        };
        CudaRuntimeError::from_result("cuda_shift_rows_u32", result)
    }

    /// Enqueues one range copy while retaining this destination until a
    /// covering same-stream completion proof is supplied.
    ///
    /// Multiple non-overlapping ranges may be submitted with the same
    /// `receipt`. The first call marks the destination pending before entering
    /// CUDA; a submit failure therefore remains fail-closed until the stream is
    /// drained. Ordinary views, copies and free operations reject the pending
    /// allocation.
    ///
    /// # Safety
    ///
    /// The caller must keep the exact source allocation incarnation alive and
    /// immutable until a proof covering `receipt` is obtained. Any destination
    /// views created before the first call may be consumed only by work ordered
    /// on `stream` after the corresponding copy and before that proof. No view
    /// may escape to another stream or survive destination release.
    pub unsafe fn copy_from_device_async_pending_unchecked(
        &mut self,
        stream: &OwnedCudaStream,
        receipt: CudaStreamSubmissionReceipt,
        destination_offset_bytes: usize,
        source: CudaBufferView,
    ) -> Result<(), CudaRuntimeError> {
        const OPERATION: &str = "cuda_memcpy_d2d_pending_owner";
        stream.validate_submission_receipt(receipt)?;
        validate_same_device(OPERATION, stream.device_ordinal(), self.device_ordinal)?;
        validate_same_device(OPERATION, self.device_ordinal, source.device_ordinal())?;
        source.checked_range(OPERATION)?;
        let destination_end = destination_offset_bytes
            .checked_add(source.len_bytes())
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        if source.len_bytes() == 0 || destination_end > self.len_bytes {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        let destination = self.raw.ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        let source_address = source
            .device_ptr()
            .raw_value()
            .checked_add(
                u64::try_from(source.offset_bytes())
                    .map_err(|_| CudaRuntimeError::range_overflow())?,
            )
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        if destination.raw == source.device_ptr().raw_value()
            && destination_offset_bytes < source.offset_bytes() + source.len_bytes()
            && source.offset_bytes() < destination_end
        {
            return Err(CudaRuntimeError::validation(
                OPERATION,
                CudaRuntimeErrorKind::OverlappingDeviceRanges,
            ));
        }
        self.d2d_state.accepts_submission(receipt)?;
        self.d2d_state.mark_pending(receipt);

        // SAFETY: destination ownership is retained by `self` and marked
        // pending above; checked ranges are inside both same-device
        // allocations. Source lifetime and any pre-created destination view
        // are retained by the caller under this method's contract.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memcpy_d2d_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                destination.raw,
                destination_offset_bytes,
                self.len_bytes,
                source_address,
                0,
                source.len_bytes(),
                source.len_bytes(),
            )
        };
        self.d2d_state.observe_submission_result(OPERATION, result)
    }

    /// Enqueues a strided device-to-device copy while retaining this
    /// destination until a covering terminal proof. The source view describes
    /// the exact source allocation window; `src_pitch_bytes` and
    /// `dst_pitch_bytes` describe the row strides, while `width_bytes` by
    /// `height` describes the copied rectangle.
    ///
    /// # Safety
    ///
    /// The source allocation incarnation must remain live and immutable until
    /// a proof covering `receipt` is obtained. Any destination view created
    /// before the first call may be consumed only by work ordered on
    /// `stream`. The caller must not use either allocation from another stream
    /// until that proof. On any native submission failure the destination stays
    /// pending and must still be released only by a covering proof.
    pub unsafe fn copy_from_device_2d_async_pending_unchecked(
        &mut self,
        stream: &OwnedCudaStream,
        receipt: CudaStreamSubmissionReceipt,
        destination_offset_bytes: usize,
        destination_pitch_bytes: usize,
        source: CudaBufferView,
        source_pitch_bytes: usize,
        width_bytes: usize,
        height: usize,
    ) -> Result<(), CudaRuntimeError> {
        const OPERATION: &str = "cuda_memcpy_d2d_2d_pending_owner";
        stream.validate_submission_receipt(receipt)?;
        validate_same_device(OPERATION, stream.device_ordinal(), self.device_ordinal)?;
        validate_same_device(OPERATION, self.device_ordinal, source.device_ordinal())?;
        source.checked_range(OPERATION)?;
        let destination_footprint =
            checked_2d_footprint(OPERATION, width_bytes, height, destination_pitch_bytes)?;
        let source_footprint =
            checked_2d_footprint(OPERATION, width_bytes, height, source_pitch_bytes)?;
        let destination_end = destination_offset_bytes
            .checked_add(destination_footprint)
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        if destination_end > self.len_bytes || source_footprint > source.len_bytes() {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        let destination = self.raw.ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        let source_address = source
            .device_ptr()
            .raw_value()
            .checked_add(
                u64::try_from(source.offset_bytes())
                    .map_err(|_| CudaRuntimeError::range_overflow())?,
            )
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        if destination.raw == source.device_ptr().raw_value()
            && destination_offset_bytes < source.offset_bytes().saturating_add(source_footprint)
            && source.offset_bytes() < destination_end
        {
            return Err(CudaRuntimeError::validation(
                OPERATION,
                CudaRuntimeErrorKind::OverlappingDeviceRanges,
            ));
        }
        self.d2d_state.accepts_submission(receipt)?;
        self.d2d_state.mark_pending(receipt);

        // SAFETY: both device ranges and pitches were checked above; the
        // destination owner is pending before the asynchronous native call.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memcpy_d2d_2d_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                destination.raw,
                destination_offset_bytes,
                self.len_bytes,
                destination_pitch_bytes,
                source_address,
                0,
                source.len_bytes(),
                source_pitch_bytes,
                width_bytes,
                height,
            )
        };
        self.d2d_state.observe_submission_result(OPERATION, result)
    }

    /// Validates that `proof` covers this allocation's pending D2D group.
    pub fn validate_pending_d2d_covering_completion(
        &self,
        proof: &CudaStreamCompletionProof,
    ) -> Result<(), CudaRuntimeError> {
        if !self.d2d_state.covered_by(proof) {
            return Err(CudaRuntimeError::invalid_metadata(
                "cuda_memcpy_d2d_pending_owner_completion",
            ));
        }
        Ok(())
    }

    /// Releases a pending D2D destination after a covering terminal proof.
    pub fn release_pending_d2d_after_covering_completion(
        &mut self,
        proof: &CudaStreamCompletionProof,
    ) -> Result<(), CudaRuntimeError> {
        self.validate_pending_d2d_covering_completion(proof)?;
        self.d2d_state.release();
        Ok(())
    }

    /// Releases a pending D2D destination after an external exact-stream drain.
    ///
    /// # Safety
    ///
    /// The caller must have successfully synchronized the stream used by every
    /// pending copy and later consumer, and no pre-created view may be used
    /// after this call.
    pub unsafe fn release_pending_d2d_after_external_stream_completion(
        &mut self,
    ) -> Result<(), CudaRuntimeError> {
        if !self.d2d_state.pending() {
            return Err(CudaRuntimeError::invalid_metadata(
                "cuda_memcpy_d2d_pending_owner_external_completion",
            ));
        }
        self.d2d_state.release();
        Ok(())
    }

    pub fn close(&mut self) -> Result<(), CudaRuntimeError> {
        self.ensure_d2d_idle("cuda_device_free_pending_d2d")?;
        let ptr = self
            .raw
            .take()
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?;

        // SAFETY: `ptr` was produced by augur_cuda_device_alloc in `allocate`
        // (same pinned ordinal) and was just taken out of `self.raw`, so this
        // is the sole free; on a failed status we restore `self.raw` so the
        // handle stays valid for a Drop retry.
        let result = unsafe { augur_cuda_sys::augur_cuda_device_free(self.device_ordinal, ptr.raw) };

        if result.status != AUGUR_CUDA_STATUS_OK {
            self.raw = Some(ptr);
        } else {
            CUDA_DEVICE_ALLOCATION_COUNTERS.record_free(self.len_bytes);
        }

        CudaRuntimeError::from_result("cuda_device_free", result)
    }

    /// Deliberately leak an allocation after a covering CUDA completion proof
    /// becomes impossible. The safe owner is closed without calling
    /// `cudaFree`; pre-created views may still exist, so the enclosing protocol
    /// must also remain poisoned and unreachable.
    pub fn quarantine_after_unproven_completion(&mut self) {
        self.raw = None;
        self.allocation_owner_instance_id = 0;
        self.len_bytes = 0;
        self.d2d_state = CudaD2dOwnerState::Idle;
    }

    fn ensure_d2d_idle(&self, operation: &'static str) -> Result<(), CudaRuntimeError> {
        if self.d2d_state.pending() {
            return Err(CudaRuntimeError::invalid_metadata(operation));
        }
        Ok(())
    }
}

impl Drop for OwnedCudaBuffer {
    fn drop(&mut self) {
        if self.raw.is_some() {
            let _ = self.close();
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaD2dOwnerHandle {
    slot: usize,
    stream_owner_instance_id: u64,
    submission_epoch: u64,
}

#[derive(Clone, Copy)]
struct CudaD2dBatchOwner {
    buffer: NonNull<OwnedCudaBuffer>,
    allocation_address: u64,
    allocation_owner_instance_id: u64,
    allocation_len_bytes: usize,
    touched: bool,
}

impl CudaD2dBatchOwner {
    const fn matches_view(self, view: CudaBufferView) -> bool {
        self.allocation_address == view.device_ptr().raw_value()
            && self.allocation_owner_instance_id == view.allocation_owner_instance_id
            && self.allocation_len_bytes == view.allocation_len_bytes()
    }
}

/// Fixed-capacity owner set for asynchronous device-to-device copies and
/// explicitly armed same-stream accesses.
///
/// The batch borrows every registered allocation exclusively, supports any
/// number of copies or external accesses among at most `N` owners, and uses
/// one terminal stream completion proof. No event or growable collection is
/// created per copy.
/// If the batch or proof is forgotten, touched owners remain pending and their
/// `Drop` path retains the device allocation rather than freeing an in-flight
/// address.
pub struct CudaD2dBatch<'a, const N: usize> {
    stream: &'a OwnedCudaStream,
    receipt: CudaStreamSubmissionReceipt,
    completion_floor: Option<CudaStreamSubmissionReceipt>,
    owners: [Option<CudaD2dBatchOwner>; N],
    owner_count: usize,
    submission_count: u64,
    sealed: bool,
    released: bool,
    _owners: PhantomData<&'a mut OwnedCudaBuffer>,
}

impl<const N: usize> core::fmt::Debug for CudaD2dBatch<'_, N> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CudaD2dBatch")
            .field("device_ordinal", &self.receipt.device_ordinal)
            .field("submission_epoch", &self.receipt.submission_epoch)
            .field(
                "completion_floor_epoch",
                &self
                    .completion_floor
                    .map(|receipt| receipt.submission_epoch),
            )
            .field("owner_capacity", &N)
            .field("owner_count", &self.owner_count)
            .field("submission_count", &self.submission_count)
            .field("sealed", &self.sealed)
            .field("released", &self.released)
            .finish()
    }
}

impl<'a, const N: usize> CudaD2dBatch<'a, N> {
    pub fn new(stream: &'a OwnedCudaStream) -> Result<Self, CudaRuntimeError> {
        if N == 0 {
            return Err(CudaRuntimeError::validation(
                "cuda_d2d_batch_capacity",
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        Ok(Self {
            stream,
            receipt: stream.reserve_submission_receipt()?,
            completion_floor: None,
            owners: [None; N],
            owner_count: 0,
            submission_count: 0,
            sealed: false,
            released: false,
            _owners: PhantomData,
        })
    }

    pub const fn owner_capacity(&self) -> usize {
        N
    }

    pub const fn owner_count(&self) -> usize {
        self.owner_count
    }

    pub const fn submission_count(&self) -> u64 {
        self.submission_count
    }

    pub fn register(
        &mut self,
        buffer: &'a mut OwnedCudaBuffer,
    ) -> Result<CudaD2dOwnerHandle, CudaRuntimeError> {
        if self.sealed || self.released {
            return Err(CudaRuntimeError::invalid_metadata("cuda_d2d_batch_sealed"));
        }
        validate_same_device(
            "cuda_d2d_batch_register",
            self.receipt.device_ordinal,
            buffer.device_ordinal,
        )?;
        buffer.ensure_d2d_idle("cuda_d2d_batch_register")?;
        let allocation_address = buffer
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?
            .raw_value();
        let allocation_len_bytes = buffer.len_bytes;
        let allocation_owner_instance_id = buffer.allocation_owner_instance_id;
        if self.owners[..self.owner_count]
            .iter()
            .flatten()
            .any(|owner| owner.allocation_address == allocation_address)
        {
            return Err(CudaRuntimeError::invalid_metadata(
                "cuda_d2d_batch_duplicate_owner",
            ));
        }
        if self.owner_count == N {
            return Err(CudaRuntimeError::validation(
                "cuda_d2d_batch_capacity",
                CudaRuntimeErrorKind::OutOfBounds,
            ));
        }
        let slot = self.owner_count;
        self.owners[slot] = Some(CudaD2dBatchOwner {
            buffer: NonNull::from(buffer),
            allocation_address,
            allocation_owner_instance_id,
            allocation_len_bytes,
            touched: false,
        });
        self.owner_count += 1;
        Ok(CudaD2dOwnerHandle {
            slot,
            stream_owner_instance_id: self.receipt.stream_owner_instance_id,
            submission_epoch: self.receipt.submission_epoch,
        })
    }

    /// Registers an owner without encoding its lifetime in the Rust borrow
    /// checker.  This is reserved for a higher-level step owner which keeps
    /// the allocation alive until the batch's terminal proof.
    ///
    /// # Safety
    ///
    /// `buffer` must remain at the same host address and retain the same device
    /// allocation incarnation, and must not be mutably accessed or dropped,
    /// until this batch has been released after a successful terminal
    /// completion proof. Owners inside a parent value that may move must be
    /// pinned or stored behind a stable indirection such as `Box`. The caller
    /// must also keep the stream and batch alive for that interval. This method
    /// exists so a batch can span a request step whose owners are fields of the
    /// same request; it must not be used as a general view-only escape hatch.
    pub unsafe fn register_external(
        &mut self,
        buffer: &OwnedCudaBuffer,
    ) -> Result<CudaD2dOwnerHandle, CudaRuntimeError> {
        if self.sealed || self.released {
            return Err(CudaRuntimeError::invalid_metadata("cuda_d2d_batch_sealed"));
        }
        validate_same_device(
            "cuda_d2d_batch_register_external",
            self.receipt.device_ordinal,
            buffer.device_ordinal,
        )?;
        buffer.ensure_d2d_idle("cuda_d2d_batch_register_external")?;
        let allocation_address = buffer
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?
            .raw_value();
        let allocation_len_bytes = buffer.len_bytes;
        let allocation_owner_instance_id = buffer.allocation_owner_instance_id;
        if self.owners[..self.owner_count]
            .iter()
            .flatten()
            .any(|owner| owner.allocation_address == allocation_address)
        {
            return Err(CudaRuntimeError::invalid_metadata(
                "cuda_d2d_batch_duplicate_owner",
            ));
        }
        if self.owner_count == N {
            return Err(CudaRuntimeError::validation(
                "cuda_d2d_batch_capacity",
                CudaRuntimeErrorKind::OutOfBounds,
            ));
        }
        let slot = self.owner_count;
        self.owners[slot] = Some(CudaD2dBatchOwner {
            buffer: NonNull::from(buffer),
            allocation_address,
            allocation_owner_instance_id,
            allocation_len_bytes,
            touched: false,
        });
        self.owner_count += 1;
        Ok(CudaD2dOwnerHandle {
            slot,
            stream_owner_instance_id: self.receipt.stream_owner_instance_id,
            submission_epoch: self.receipt.submission_epoch,
        })
    }

    pub fn copy(
        &mut self,
        destination: CudaD2dOwnerHandle,
        destination_offset_bytes: usize,
        source: CudaD2dOwnerHandle,
        source_offset_bytes: usize,
        copy_len_bytes: usize,
    ) -> Result<(), CudaRuntimeError> {
        if self.sealed || self.released {
            return Err(CudaRuntimeError::invalid_metadata("cuda_d2d_batch_sealed"));
        }
        let destination_owner = self.resolve(destination)?;
        let source_owner = self.resolve(source)?;
        // SAFETY: register accepted exclusive borrows for the batch lifetime.
        // Distinct slots have distinct live allocations, and the same slot is
        // accessed here only through shared references for validation.
        let destination_buffer = unsafe { destination_owner.buffer.as_ref() };
        // SAFETY: same registry invariant as `destination_buffer`.
        let source_buffer = unsafe { source_owner.buffer.as_ref() };
        let destination_len_bytes = destination_buffer.len_bytes;
        let source_len_bytes = source_buffer.len_bytes;
        let destination_end = destination_offset_bytes
            .checked_add(copy_len_bytes)
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        let source_end = source_offset_bytes
            .checked_add(copy_len_bytes)
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        if copy_len_bytes == 0
            || destination_end > destination_len_bytes
            || source_end > source_len_bytes
        {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        if destination_owner.allocation_address == source_owner.allocation_address
            && destination_offset_bytes < source_end
            && source_offset_bytes < destination_end
        {
            return Err(CudaRuntimeError::validation(
                "cuda_d2d_batch_copy",
                CudaRuntimeErrorKind::OverlappingDeviceRanges,
            ));
        }
        destination_buffer
            .d2d_state
            .accepts_submission(self.receipt)?;
        source_buffer.d2d_state.accepts_submission(self.receipt)?;
        let raw_stream = self.stream.raw()?;
        let destination_ptr = destination_buffer
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        let source_ptr = source_buffer
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        let next_submission_count = self
            .submission_count
            .checked_add(1)
            .ok_or_else(|| CudaRuntimeError::invalid_metadata("cuda_d2d_batch_submission_count"))?;

        self.mark_touched(destination.slot);
        if source.slot != destination.slot {
            self.mark_touched(source.slot);
        }
        self.submission_count = next_submission_count;

        // SAFETY: both owners are exclusively borrowed by this batch and were
        // marked pending before the async call. Bounds, non-overlap and device
        // identity were checked above. A matching terminal proof is the only
        // safe path that clears the pending states; otherwise buffer Drop
        // retains the allocations.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memcpy_d2d_submit_v1(
                self.receipt.device_ordinal,
                raw_stream,
                destination_ptr.raw,
                destination_offset_bytes,
                destination_len_bytes,
                source_ptr.raw,
                source_offset_bytes,
                source_len_bytes,
                copy_len_bytes,
            )
        };
        // Submission failure cannot prove that the runtime consumed no device
        // address. Keep both owners pending until an exact terminal proof is
        // obtained, just as for a successful asynchronous submission.
        let destination_state = unsafe {
            self.owners[destination.slot]
                .expect("validated D2D destination must remain registered")
                .buffer
                .as_ref()
                .d2d_state
        };
        destination_state.observe_submission_result("cuda_d2d_batch_copy", result)
    }

    /// Retains a registered owner for stream work submitted outside this
    /// batch's D2D helpers.
    ///
    /// Call this immediately before the first asynchronous producer or
    /// consumer that accesses the owner. If that submission or a later step
    /// fails before the first D2D copy, dropping the batch leaves the owner
    /// pending instead of freeing an address that may still be in flight.
    ///
    /// # Safety
    ///
    /// Every external access covered by this arm must be submitted to this
    /// batch's stream after this call and before its terminal completion
    /// proof. No copied view of the owner may be submitted after that proof or
    /// to another stream until the batch has released the owner.
    pub(crate) unsafe fn arm_owner_for_stream_access(
        &mut self,
        owner: CudaD2dOwnerHandle,
    ) -> Result<(), CudaRuntimeError> {
        if self.sealed || self.released {
            return Err(CudaRuntimeError::invalid_metadata("cuda_d2d_batch_sealed"));
        }
        self.resolve(owner)?;
        self.mark_touched(owner.slot);
        Ok(())
    }

    /// Arms a registered allocation for an external same-stream access and
    /// returns the already validated view for that access.
    ///
    /// This is the capability used by a layer-major request chunk when a
    /// projection scratch allocation is written again after an earlier D2D
    /// copy has marked it pending.  Ordinary `OwnedCudaBuffer::view` remains
    /// fail-closed while pending; callers must submit the returned view on
    /// this batch's stream before its terminal proof.
    ///
    /// # Safety
    ///
    /// The returned view must be used only for work submitted to this batch's
    /// stream, and the allocation must remain owned by this batch until a
    /// covering completion proof releases it.
    pub unsafe fn arm_registered_view_for_stream_access(
        &mut self,
        owner: CudaD2dOwnerHandle,
        view: CudaBufferView,
    ) -> Result<CudaBufferView, CudaRuntimeError> {
        let registered = self.registered_view_owner(view)?;
        if registered.slot != owner.slot
            || registered.stream_owner_instance_id != owner.stream_owner_instance_id
            || registered.submission_epoch != owner.submission_epoch
        {
            return Err(CudaRuntimeError::invalid_metadata(
                "cuda_d2d_batch_registered_view_owner",
            ));
        }
        // SAFETY: forwarded from this method's owner/lifetime contract.
        unsafe { self.arm_owner_for_stream_access(owner) }?;
        Ok(view)
    }

    /// Copies from a view whose allocation owner is already registered in
    /// this batch.
    ///
    /// The copied `CudaBufferView` supplies range and allocation-incarnation
    /// metadata. A matching live owner must be held exclusively by this batch,
    /// so the asynchronous submission cannot outlive or ABA-alias its source.
    pub fn copy_from_registered_view(
        &mut self,
        destination: CudaD2dOwnerHandle,
        destination_offset_bytes: usize,
        source: CudaBufferView,
    ) -> Result<(), CudaRuntimeError> {
        let source_owner = self.registered_view_owner(source)?;
        self.copy(
            destination,
            destination_offset_bytes,
            source_owner,
            source.offset_bytes(),
            source.len_bytes(),
        )
    }

    /// Validates that a view belongs to the exact registered allocation
    /// incarnation without submitting device work.
    pub fn validate_registered_view(&self, source: CudaBufferView) -> Result<(), CudaRuntimeError> {
        self.registered_view_owner(source).map(|_| ())
    }

    pub fn completion_receipt(&mut self) -> Result<CudaStreamSubmissionReceipt, CudaRuntimeError> {
        if !self.has_touched_owners() || self.released {
            return Err(CudaRuntimeError::invalid_metadata(
                "cuda_d2d_batch_completion_receipt",
            ));
        }
        if self.completion_floor.is_none() {
            self.completion_floor = Some(self.stream.reserve_submission_receipt()?);
        }
        self.sealed = true;
        Ok(self.receipt)
    }

    /// Copies a registered owner range to host at the batch's terminal
    /// boundary and returns the exact completion proof for this batch.
    ///
    /// This is the owner-aware form of a synchronous D2H receipt. It permits
    /// the source owner to remain pending in this batch while the native D2H
    /// drains the stream, and seals the batch before submitting that terminal
    /// copy. A validation or completion failure never releases pending owners.
    pub fn copy_owner_range_to_host_sync(
        &mut self,
        source: CudaD2dOwnerHandle,
        source_offset_bytes: usize,
        destination: &mut [u8],
    ) -> Result<CudaStreamCompletionProof, CudaRuntimeError> {
        if self.submission_count == 0 || self.sealed || self.released {
            return Err(CudaRuntimeError::invalid_metadata(
                "cuda_d2d_batch_terminal_d2h",
            ));
        }
        let source_owner = self.resolve(source)?;
        // SAFETY: `register` accepted an exclusive owner borrow for the batch
        // lifetime, and `resolve` proved that this handle belongs to the batch.
        let source_buffer = unsafe { source_owner.buffer.as_ref() };
        let source_end = source_offset_bytes
            .checked_add(destination.len())
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        if destination.is_empty() || source_end > source_buffer.len_bytes {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        let source_ptr = source_buffer
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        let raw_stream = self.stream.raw()?;

        self.sealed = true;
        // SAFETY: the source allocation is exclusively retained by this
        // batch, the checked range is inside it, and `destination` is an
        // exclusive host borrow for the complete synchronous call. The same
        // stream orders every preceding batch copy, so successful return is
        // an exact completion boundary for `self.receipt`.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memcpy_d2h_sync_v2(
                self.receipt.device_ordinal,
                raw_stream,
                destination.as_mut_ptr(),
                destination.len(),
                source_ptr.raw,
                source_offset_bytes,
                source_buffer.len_bytes,
                destination.len(),
            )
        };
        CudaRuntimeError::from_result("cuda_d2d_batch_terminal_d2h", result)?;
        record_synchronous_d2h(destination.len());
        Ok(self.stream.completion_proof_after_sync(self.receipt))
    }

    pub fn release_after_completion(
        &mut self,
        proof: &CudaStreamCompletionProof,
    ) -> Result<(), CudaRuntimeError> {
        if !self.has_touched_owners()
            || !self.sealed
            || self.released
            || proof.device_ordinal != self.receipt.device_ordinal
            || proof.stream_owner_instance_id != self.receipt.stream_owner_instance_id
            || proof.submission_epoch != self.receipt.submission_epoch
        {
            return Err(CudaRuntimeError::invalid_metadata(
                "cuda_d2d_batch_completion_proof",
            ));
        }
        for owner in self.owners[..self.owner_count].iter().flatten() {
            if owner.touched {
                // SAFETY: the batch still holds the exclusive owner borrow.
                let buffer = unsafe { owner.buffer.as_ref() };
                if !buffer.d2d_state.matches_proof(proof) {
                    return Err(CudaRuntimeError::invalid_metadata(
                        "cuda_d2d_batch_completion_proof",
                    ));
                }
            }
        }
        for owner in self.owners[..self.owner_count].iter_mut().flatten() {
            if owner.touched {
                // SAFETY: distinct registry slots came from distinct exclusive
                // borrows; validation above completed before any state change.
                unsafe { owner.buffer.as_mut() }.d2d_state.release();
            }
        }
        self.released = true;
        Ok(())
    }

    /// Releases a sealed owner batch when a later terminal operation completed
    /// the same stream after this batch's submission point.
    ///
    /// Call [`Self::completion_receipt`] after the final batch submission and
    /// before obtaining the covering proof. Requiring that ordering prevents a
    /// proof from being reused for work submitted after its completion.
    pub fn release_after_covering_completion(
        &mut self,
        proof: &CudaStreamCompletionProof,
    ) -> Result<(), CudaRuntimeError> {
        let completion_floor = self.completion_floor.ok_or_else(|| {
            CudaRuntimeError::invalid_metadata("cuda_d2d_batch_covering_completion_proof_floor")
        })?;
        if !self.has_touched_owners()
            || !self.sealed
            || self.released
            || !proof.covers_submission(completion_floor)
        {
            return Err(CudaRuntimeError::invalid_metadata(
                "cuda_d2d_batch_covering_completion_proof_boundary",
            ));
        }
        for owner in self.owners[..self.owner_count].iter().flatten() {
            if owner.touched {
                // SAFETY: the batch still holds the exclusive owner borrow.
                let buffer = unsafe { owner.buffer.as_ref() };
                if !buffer.d2d_state.matches_submission(self.receipt) {
                    return Err(CudaRuntimeError::invalid_metadata(
                        "cuda_d2d_batch_covering_completion_proof_owner",
                    ));
                }
            }
        }
        for owner in self.owners[..self.owner_count].iter_mut().flatten() {
            if owner.touched {
                // SAFETY: distinct registry slots came from distinct exclusive
                // borrows; validation above completed before any state change.
                unsafe { owner.buffer.as_mut() }.d2d_state.release();
            }
        }
        self.released = true;
        Ok(())
    }

    /// Releases owners after an external operation has already completed the
    /// batch stream.  No additional CUDA wait is issued.
    ///
    /// # Safety
    ///
    /// The caller must have successfully completed a blocking operation on
    /// this batch's exact stream after every batch submission, and must not
    /// submit further work through any registered owner before this method
    /// returns.  A mere enqueue, event record, or unrelated stream wait is
    /// not a completion proof.
    pub unsafe fn release_after_external_stream_completion(
        &mut self,
    ) -> Result<(), CudaRuntimeError> {
        if !self.has_touched_owners() || self.released {
            return Err(CudaRuntimeError::invalid_metadata(
                "cuda_d2d_batch_external_completion",
            ));
        }
        let receipt = if self.sealed {
            self.receipt
        } else {
            self.completion_receipt()?
        };
        let proof = CudaStreamCompletionProof {
            device_ordinal: receipt.device_ordinal,
            stream_owner_instance_id: receipt.stream_owner_instance_id,
            submission_epoch: receipt.submission_epoch,
        };
        self.release_after_completion(&proof)
    }

    /// Returns whether at least one registered owner has been armed or used.
    ///
    /// An external producer may be armed before its first D2D copy, so
    /// `submission_count` alone is not sufficient to decide whether an error
    /// path must drain and release the batch.
    pub const fn has_touched_owners(&self) -> bool {
        let mut index = 0;
        while index < N {
            if let Some(owner) = self.owners[index] {
                if owner.touched {
                    return true;
                }
            }
            index += 1;
        }
        false
    }

    fn resolve(&self, handle: CudaD2dOwnerHandle) -> Result<CudaD2dBatchOwner, CudaRuntimeError> {
        if handle.stream_owner_instance_id != self.receipt.stream_owner_instance_id
            || handle.submission_epoch != self.receipt.submission_epoch
        {
            return Err(CudaRuntimeError::invalid_metadata(
                "cuda_d2d_batch_owner_handle",
            ));
        }
        self.owners
            .get(handle.slot)
            .and_then(|owner| *owner)
            .ok_or_else(|| CudaRuntimeError::invalid_metadata("cuda_d2d_batch_owner_handle"))
    }

    fn registered_view_owner(
        &self,
        source: CudaBufferView,
    ) -> Result<CudaD2dOwnerHandle, CudaRuntimeError> {
        if self.sealed || self.released {
            return Err(CudaRuntimeError::invalid_metadata("cuda_d2d_batch_sealed"));
        }
        validate_same_device(
            "cuda_d2d_batch_registered_view",
            self.receipt.device_ordinal,
            source.device_ordinal(),
        )?;
        source.checked_range("cuda_d2d_batch_registered_view")?;
        let slot = self.owners[..self.owner_count]
            .iter()
            .position(|owner| owner.is_some_and(|owner| owner.matches_view(source)))
            .ok_or_else(|| {
                CudaRuntimeError::invalid_metadata("cuda_d2d_batch_unregistered_view")
            })?;
        Ok(CudaD2dOwnerHandle {
            slot,
            stream_owner_instance_id: self.receipt.stream_owner_instance_id,
            submission_epoch: self.receipt.submission_epoch,
        })
    }

    fn mark_touched(&mut self, slot: usize) {
        let owner = self.owners[slot]
            .as_mut()
            .expect("validated D2D owner slot must remain registered");
        if !owner.touched {
            // SAFETY: register stored an exclusive borrow for the batch
            // lifetime and every distinct slot refers to a distinct owner.
            unsafe { owner.buffer.as_mut() }
                .d2d_state
                .mark_pending(self.receipt);
            owner.touched = true;
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PinnedH2dSlotState {
    Idle,
    EventPending,
    Indeterminate,
}

/// Compile-time ceiling for one allocation-free segmented H2D transaction.
/// The isolated expert micro-slice plan uses twenty ranges; the extra headroom
/// keeps this owner reusable for other bounded layouts without a heap ledger.
pub const OWNED_PINNED_H2D_MAX_SEGMENTS: usize = 32;

impl PinnedH2dSlotState {
    const fn pending(self) -> bool {
        !matches!(self, Self::Idle)
    }

    fn mark_submitted(&mut self) {
        debug_assert_eq!(*self, Self::Idle);
        *self = Self::Indeterminate;
    }

    fn mark_event_recorded(&mut self) {
        debug_assert_eq!(*self, Self::Indeterminate);
        *self = Self::EventPending;
    }

    fn mark_event_unreliable(&mut self) {
        debug_assert!(self.pending());
        *self = Self::Indeterminate;
    }

    fn mark_completed(&mut self) {
        debug_assert!(self.pending());
        *self = Self::Idle;
    }

    fn observe_stream_synchronize(&mut self, completed: bool) {
        debug_assert!(self.pending());
        if completed {
            self.mark_completed();
        } else {
            *self = Self::Indeterminate;
        }
    }
}

pub struct OwnedPinnedH2dSlot<'a> {
    destination: Option<OwnedCudaBuffer>,
    stream: &'a OwnedCudaStream,
    host: Option<OwnedCudaHostBuffer>,
    completion: Option<OwnedCudaEvent>,
    capacity_bytes: usize,
    state: PinnedH2dSlotState,
}

impl core::fmt::Debug for OwnedPinnedH2dSlot<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("OwnedPinnedH2dSlot")
            .field("device_ordinal", &self.stream.device_ordinal())
            .field("capacity_bytes", &self.capacity_bytes())
            .field("state", &self.state)
            .finish()
    }
}

impl<'a> OwnedPinnedH2dSlot<'a> {
    pub fn allocate(
        stream: &'a OwnedCudaStream,
        capacity_bytes: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if capacity_bytes == 0 {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        Ok(Self {
            destination: Some(OwnedCudaBuffer::allocate(
                stream.device_ordinal(),
                capacity_bytes,
            )?),
            stream,
            host: Some(OwnedCudaHostBuffer::allocate(
                stream.device_ordinal(),
                capacity_bytes,
            )?),
            completion: Some(OwnedCudaEvent::create(stream.device_ordinal())?),
            capacity_bytes,
            state: PinnedH2dSlotState::Idle,
        })
    }

    pub const fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub const fn pending(&self) -> bool {
        self.state.pending()
    }

    pub const fn stream(&self) -> &'a OwnedCudaStream {
        self.stream
    }

    pub fn device_view(&self) -> Result<CudaBufferView, CudaRuntimeError> {
        let destination = self
            .destination
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        destination.view(0, destination.len_bytes())
    }

    pub fn writable(&mut self) -> Result<&mut [u8], CudaRuntimeError> {
        self.wait()?;
        self.host
            .as_mut()
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?
            .as_mut_slice()
    }

    /// # Safety
    ///
    /// The caller must have completed this slot's fixed stream (for example
    /// via [`OwnedCudaStream::synchronize`] or an event fence recorded after
    /// the copy) after the last `submit`. Otherwise the returned `&mut [u8]`
    /// aliases pinned memory an in-flight H2D copy is still reading, and
    /// clearing the pending state here disarms the leak-on-drop fence that
    /// normally keeps `Drop` from freeing that memory under the DMA.
    pub unsafe fn writable_after_external_completion(
        &mut self,
    ) -> Result<&mut [u8], CudaRuntimeError> {
        if self.state.pending() {
            self.state.mark_completed();
        }
        self.host
            .as_mut()
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?
            .as_mut_slice()
    }

    pub fn retain_until_external_stream_completion(&mut self) {
        self.state = PinnedH2dSlotState::Indeterminate;
    }

    /// # Safety
    ///
    /// The caller must have completed this slot's fixed stream after its last
    /// device consumer. Marking the slot idle without that fence lets `Drop`
    /// free the device and pinned buffers (instead of leaking them) while
    /// queued copies or kernels may still read or write them.
    pub unsafe fn release_after_external_stream_completion(&mut self) {
        if self.state.pending() {
            self.state.mark_completed();
        }
    }

    pub fn submit(&mut self) -> Result<(), CudaRuntimeError> {
        if self.state.pending() {
            return Err(CudaRuntimeError::invalid_metadata(
                "owned_pinned_h2d_slot_pending",
            ));
        }
        let destination = self
            .destination
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        let source = self
            .host
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?;
        let destination_len = self
            .destination
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?
            .len_bytes();
        let host_len = self
            .host
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?
            .len_bytes();
        // SAFETY: destination and pinned source were both allocated with
        // `capacity_bytes` by OwnedCudaBuffer/OwnedCudaHostBuffer in
        // `allocate` and are open (checked Some above); extents are the
        // recorded allocation lengths, re-checked by the shim. The submit is
        // async, but `state` turns pending below and `writable` waits on the
        // completion event before handing out `&mut`, while `Drop` waits and
        // leaks all owned resources if completion cannot be confirmed — so
        // the DMA never reads freed or aliased pinned memory.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memcpy_h2d_submit_v1(
                self.stream.device_ordinal(),
                self.stream.raw()?,
                destination.raw,
                0,
                destination_len,
                source.as_ptr().cast(),
                host_len,
                host_len,
            )
        };
        CudaRuntimeError::from_result("cuda_memcpy_h2d_async", result)?;
        self.state.mark_submitted();
        let record = self
            .completion
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("event"))?
            .record(self.stream);
        if let Err(error) = record {
            if self.stream.synchronize().is_ok() {
                self.state.mark_completed();
            }
            return Err(error);
        }
        self.state.mark_event_recorded();
        Ok(())
    }

    pub fn wait(&mut self) -> Result<(), CudaRuntimeError> {
        match self.state {
            PinnedH2dSlotState::Idle => Ok(()),
            PinnedH2dSlotState::EventPending => {
                let result = self
                    .completion
                    .as_ref()
                    .ok_or_else(|| CudaRuntimeError::closed("event"))?
                    .synchronize();
                match result {
                    Ok(()) => {
                        self.state.mark_completed();
                        Ok(())
                    }
                    Err(error) => {
                        self.state.mark_event_unreliable();
                        if self.stream.synchronize().is_ok() {
                            self.state.mark_completed();
                        }
                        Err(error)
                    }
                }
            }
            PinnedH2dSlotState::Indeterminate => {
                self.stream.synchronize()?;
                self.state.mark_completed();
                Ok(())
            }
        }
    }

    fn leak_owned_resources(&mut self) {
        if let Some(destination) = self.destination.take() {
            core::mem::forget(destination);
        }
        if let Some(host) = self.host.take() {
            core::mem::forget(host);
        }
        if let Some(completion) = self.completion.take() {
            core::mem::forget(completion);
        }
    }
}

impl Drop for OwnedPinnedH2dSlot<'_> {
    fn drop(&mut self) {
        let _ = self.wait();
        if self.state.pending() {
            self.leak_owned_resources();
        }
    }
}

/// Owns one pinned host staging buffer and its fixed-size device destination.
///
/// Unlike [`OwnedPinnedH2dSlot`], this owner is not tied to a borrowed stream.
/// Callers can therefore keep it inside a long-lived execution state while
/// still binding every submission to the stream that orders its consumers.
pub struct OwnedPinnedH2dBuffer {
    destination: Option<OwnedCudaBuffer>,
    host: Option<OwnedCudaHostBuffer>,
    completion: Option<OwnedCudaEvent>,
    device_ordinal: i32,
    capacity_bytes: usize,
    state: PinnedH2dSlotState,
    submission_receipt: Option<CudaStreamSubmissionReceipt>,
    segmented_range_count: usize,
    segmented_stream_owner_instance_key: Option<usize>,
}

/// Opaque proof that a dependent CUDA stream queued a wait for one pinned
/// H2D owner's private completion event.
///
/// The receipt is only useful with the exact owner that minted it. A covering
/// completion proof from the dependent stream establishes a transitive CUDA
/// happens-before edge from the pinned upload (and any same-stream work queued
/// before the owner's completion event) to the dependent terminal boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaPinnedH2dDependencyReceipt {
    source_submission: CudaStreamSubmissionReceipt,
    completion_event_owner_instance_key: usize,
    dependent_submission: CudaStreamSubmissionReceipt,
}

impl CudaPinnedH2dDependencyReceipt {
    pub const fn source_submission(self) -> CudaStreamSubmissionReceipt {
        self.source_submission
    }

    pub const fn dependent_submission(self) -> CudaStreamSubmissionReceipt {
        self.dependent_submission
    }
}

/// Allocation-free permit for adding disjoint ranges to one pinned H2D
/// transaction. The exclusive borrow keeps submitted host ranges immutable
/// until the final completion event (or a covering proof) is observed.
#[must_use = "a segmented pinned H2D transaction must be finished or dropped"]
pub struct OwnedPinnedH2dSegments<'buffer> {
    buffer: &'buffer mut OwnedPinnedH2dBuffer,
    submitted_ranges: [Option<core::ops::Range<usize>>; OWNED_PINNED_H2D_MAX_SEGMENTS],
    submitted_range_count: usize,
    stream_owner_instance_key: Option<usize>,
    last_submission_receipt: Option<CudaStreamSubmissionReceipt>,
}

impl core::fmt::Debug for OwnedPinnedH2dSegments<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("OwnedPinnedH2dSegments")
            .field(
                "submitted_ranges",
                &&self.submitted_ranges[..self.submitted_range_count],
            )
            .field("stream_owner_instance_key", &self.stream_owner_instance_key)
            .field("last_submission_receipt", &self.last_submission_receipt)
            .finish()
    }
}

impl core::fmt::Debug for OwnedPinnedH2dBuffer {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("OwnedPinnedH2dBuffer")
            .field("device_ordinal", &self.device_ordinal)
            .field("capacity_bytes", &self.capacity_bytes)
            .field("state", &self.state)
            .finish()
    }
}

impl OwnedPinnedH2dBuffer {
    pub fn allocate(device_ordinal: i32, capacity_bytes: usize) -> Result<Self, CudaRuntimeError> {
        if capacity_bytes == 0 {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        Ok(Self {
            destination: Some(OwnedCudaBuffer::allocate(device_ordinal, capacity_bytes)?),
            host: Some(OwnedCudaHostBuffer::allocate(
                device_ordinal,
                capacity_bytes,
            )?),
            completion: Some(OwnedCudaEvent::create(device_ordinal)?),
            device_ordinal,
            capacity_bytes,
            state: PinnedH2dSlotState::Idle,
            submission_receipt: None,
            segmented_range_count: 0,
            segmented_stream_owner_instance_key: None,
        })
    }

    pub const fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub const fn pending(&self) -> bool {
        self.state.pending()
    }

    /// Allocates only the pinned host source and completion event. This form
    /// is used when each submission binds a different live device view; it
    /// avoids allocating an unused fixed device staging destination.
    pub fn allocate_host_only(
        device_ordinal: i32,
        capacity_bytes: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if capacity_bytes == 0 {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        Ok(Self {
            destination: None,
            host: Some(OwnedCudaHostBuffer::allocate(
                device_ordinal,
                capacity_bytes,
            )?),
            completion: Some(OwnedCudaEvent::create(device_ordinal)?),
            device_ordinal,
            capacity_bytes,
            state: PinnedH2dSlotState::Idle,
            submission_receipt: None,
            segmented_range_count: 0,
            segmented_stream_owner_instance_key: None,
        })
    }

    pub fn device_view(&self) -> Result<CudaBufferView, CudaRuntimeError> {
        let destination = self
            .destination
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        destination.view(0, destination.len_bytes())
    }

    pub fn writable(&mut self) -> Result<&mut [u8], CudaRuntimeError> {
        self.wait()?;
        self.host
            .as_mut()
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?
            .as_mut_slice()
    }

    /// Begins one bounded, allocation-free segmented transaction. Each range
    /// is submitted asynchronously and the transaction records one completion
    /// event only after its final range.
    pub fn begin_segments(&mut self) -> Result<OwnedPinnedH2dSegments<'_>, CudaRuntimeError> {
        if self.state.pending()
            || self.submission_receipt.is_some()
            || self.segmented_range_count != 0
        {
            return Err(CudaRuntimeError::invalid_metadata(
                "owned_pinned_h2d_buffer_pending",
            ));
        }
        Ok(OwnedPinnedH2dSegments {
            buffer: self,
            submitted_ranges: core::array::from_fn(|_| None),
            submitted_range_count: 0,
            stream_owner_instance_key: None,
            last_submission_receipt: None,
        })
    }

    pub fn submit(&mut self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        validate_same_device(
            "owned_pinned_h2d_buffer",
            stream.device_ordinal(),
            self.device_ordinal,
        )?;
        if self.state.pending() || self.segmented_range_count != 0 {
            return Err(CudaRuntimeError::invalid_metadata(
                "owned_pinned_h2d_buffer_pending",
            ));
        }
        debug_assert!(self.submission_receipt.is_none());
        let destination = self
            .destination
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        let source = self
            .host
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?;
        // SAFETY: destination and pinned source were both allocated with
        // `capacity_bytes` in `allocate` and are open (checked Some above);
        // same-device between `stream` and this owner was validated. The
        // submit is async, but `state` turns pending below and `writable`/
        // `wait` gate reuse on the completion event (Indeterminate refuses to
        // clear), while `Drop` leaks all owned resources when completion is
        // unproven — the queued copy never touches freed memory.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memcpy_h2d_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                destination.raw,
                0,
                self.capacity_bytes,
                source.as_ptr().cast(),
                self.capacity_bytes,
                self.capacity_bytes,
            )
        };
        CudaRuntimeError::from_result("cuda_memcpy_h2d_async", result)?;
        self.state.mark_submitted();
        let receipt = stream.submission_receipt()?;
        if let Err(error) = self
            .completion
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("event"))?
            .record_for_submission(stream, receipt)
        {
            self.state.mark_event_unreliable();
            if stream.synchronize().is_ok() {
                self.state.mark_completed();
                self.clear_segmented_state();
            }
            return Err(error);
        }
        self.state.mark_event_recorded();
        self.submission_receipt = Some(receipt);
        Ok(())
    }

    /// Enqueues this pinned source into an externally owned device view.
    ///
    /// The owner remains pending until [`Self::wait`], a covering completion
    /// proof, or a dependent-stream proof is observed. The caller must keep
    /// `destination`'s allocation live and unchanged until that proof.
    ///
    /// # Safety
    ///
    /// `destination` must refer to a live allocation whose owner outlives the
    /// asynchronous copy and whose bytes are not reused before completion.
    pub unsafe fn submit_to_view(
        &mut self,
        stream: &OwnedCudaStream,
        destination: CudaBufferView,
    ) -> Result<(), CudaRuntimeError> {
        const OPERATION: &str = "cuda_memcpy_h2d_async_pinned_view";
        validate_same_device(OPERATION, stream.device_ordinal(), self.device_ordinal)?;
        if self.state.pending()
            || self.submission_receipt.is_some()
            || self.segmented_range_count != 0
        {
            return Err(CudaRuntimeError::invalid_metadata(
                "owned_pinned_h2d_buffer_pending",
            ));
        }
        if destination.device_ordinal() != self.device_ordinal
            || destination.len_bytes() != self.capacity_bytes
        {
            return Err(CudaRuntimeError::validation(
                OPERATION,
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        destination.checked_range(OPERATION)?;
        let source = self
            .host
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?;
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memcpy_h2d_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                destination.device_ptr().raw,
                destination.offset_bytes(),
                destination.allocation_len_bytes(),
                source.as_ptr().cast(),
                self.capacity_bytes,
                self.capacity_bytes,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)?;
        self.state.mark_submitted();
        let receipt = stream.submission_receipt()?;
        if let Err(error) = self
            .completion
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("event"))?
            .record_for_submission(stream, receipt)
        {
            self.state.mark_event_unreliable();
            if stream.synchronize().is_ok() {
                self.state.mark_completed();
                self.clear_segmented_state();
            }
            return Err(error);
        }
        self.state.mark_event_recorded();
        self.submission_receipt = Some(receipt);
        Ok(())
    }

    pub fn submission_receipt(&self) -> Result<CudaStreamSubmissionReceipt, CudaRuntimeError> {
        if !self.state.pending() {
            return Err(CudaRuntimeError::invalid_metadata(
                "owned_pinned_h2d_buffer_not_pending",
            ));
        }
        self.submission_receipt.ok_or_else(|| {
            CudaRuntimeError::invalid_metadata("owned_pinned_h2d_buffer_submission_receipt")
        })
    }

    /// Returns the event recorded for the current H2D submission. Consumers
    /// on another stream must pair this event with [`Self::submission_receipt`]
    /// and use [`OwnedCudaStream::wait_event`] before touching the destination.
    pub fn completion_event(&self) -> Result<&OwnedCudaEvent, CudaRuntimeError> {
        self.completion
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("owned pinned h2d completion event"))
    }

    /// Orders this owner's private completion event on `dependent_stream`.
    ///
    /// The owner remains pending. The returned receipt must later be paired
    /// with a covering proof from the dependent stream before the pinned host
    /// source or fixed device destination can be reused.
    pub fn order_completion_on(
        &self,
        dependent_stream: &OwnedCudaStream,
    ) -> Result<CudaPinnedH2dDependencyReceipt, CudaRuntimeError> {
        validate_same_device(
            "owned_pinned_h2d_buffer_dependency",
            dependent_stream.device_ordinal(),
            self.device_ordinal,
        )?;
        let source_submission = self.submission_receipt()?;
        let completion = self
            .completion
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("event"))?;
        dependent_stream.wait_event(completion)?;
        let dependent_submission = dependent_stream.submission_receipt()?;
        Ok(CudaPinnedH2dDependencyReceipt {
            source_submission,
            completion_event_owner_instance_key: completion.owner_instance_key()?,
            dependent_submission,
        })
    }

    pub fn wait(&mut self) -> Result<(), CudaRuntimeError> {
        match self.state {
            PinnedH2dSlotState::Idle => Ok(()),
            PinnedH2dSlotState::EventPending => {
                self.completion
                    .as_ref()
                    .ok_or_else(|| CudaRuntimeError::closed("event"))?
                    .synchronize()?;
                self.state.mark_completed();
                self.submission_receipt = None;
                self.clear_segmented_state();
                Ok(())
            }
            PinnedH2dSlotState::Indeterminate => Err(CudaRuntimeError::invalid_metadata(
                "owned_pinned_h2d_buffer_indeterminate",
            )),
        }
    }

    /// Releases the pinned source and device destination after a later
    /// terminal operation proved completion on the same stream owner.
    pub fn release_after_covering_completion(
        &mut self,
        proof: &CudaStreamCompletionProof,
    ) -> Result<(), CudaRuntimeError> {
        let receipt = self.submission_receipt()?;
        if !proof.covers_submission(receipt) {
            return Err(CudaRuntimeError::invalid_metadata(
                "owned_pinned_h2d_buffer_completion_proof",
            ));
        }
        self.state.mark_completed();
        self.submission_receipt = None;
        self.clear_segmented_state();
        Ok(())
    }

    /// Releases this owner after a dependent stream completed a wait for the
    /// owner's private completion event.
    ///
    /// Unlike [`Self::release_after_covering_completion`], the terminal proof
    /// belongs to the dependent stream rather than the upload stream. The
    /// opaque dependency receipt binds both streams through the private event,
    /// so a stale owner, foreign event, or unrelated stream proof fails closed.
    pub fn release_after_dependent_completion(
        &mut self,
        dependency: CudaPinnedH2dDependencyReceipt,
        proof: &CudaStreamCompletionProof,
    ) -> Result<(), CudaRuntimeError> {
        let source_submission = self.submission_receipt()?;
        let completion = self
            .completion
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("event"))?;
        if dependency.source_submission != source_submission
            || dependency.completion_event_owner_instance_key != completion.owner_instance_key()?
            || !proof.covers_submission(dependency.dependent_submission)
        {
            return Err(CudaRuntimeError::invalid_metadata(
                "owned_pinned_h2d_buffer_dependency_proof",
            ));
        }
        self.state.mark_completed();
        self.submission_receipt = None;
        self.clear_segmented_state();
        Ok(())
    }

    /// # Safety
    ///
    /// The caller must have completed the submission stream after the last
    /// upload and every device consumer of this buffer. Without that fence,
    /// marking the state idle lets `Drop` free the pinned and device buffers
    /// (instead of leaking them) while a queued copy or kernel still uses
    /// them.
    pub unsafe fn release_after_external_stream_completion(&mut self) {
        if self.state.pending() {
            self.state.mark_completed();
        }
        self.submission_receipt = None;
        self.clear_segmented_state();
    }

    fn clear_segmented_state(&mut self) {
        self.segmented_range_count = 0;
        self.segmented_stream_owner_instance_key = None;
    }

    fn leak_owned_resources(&mut self) {
        if let Some(destination) = self.destination.take() {
            core::mem::forget(destination);
        }
        if let Some(host) = self.host.take() {
            core::mem::forget(host);
        }
        if let Some(completion) = self.completion.take() {
            core::mem::forget(completion);
        }
    }
}

impl OwnedPinnedH2dSegments<'_> {
    

    fn validate_range(&self, range: &core::ops::Range<usize>) -> Result<(), CudaRuntimeError> {
        if range.is_empty() || range.start > range.end || range.end > self.buffer.capacity_bytes {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        if self
            .submitted_ranges
            .iter()
            .flatten()
            .any(|submitted| range.start < submitted.end && submitted.start < range.end)
        {
            return Err(CudaRuntimeError::validation(
                "owned_pinned_h2d_segments_overlap",
                CudaRuntimeErrorKind::InvalidLayout,
            ));
        }
        Ok(())
    }

    /// Borrows one host range that has not yet been submitted.
    pub fn writable_range(
        &mut self,
        range: core::ops::Range<usize>,
    ) -> Result<&mut [u8], CudaRuntimeError> {
        self.validate_range(&range)?;
        self.buffer
            .host
            .as_mut()
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?
            .as_mut_slice()
            .map(|host| &mut host[range])
    }

    /// Enqueues one disjoint range without recording a completion event. The
    /// final `finish` call records the sole completion boundary.
    pub fn submit_range(
        &mut self,
        stream: &OwnedCudaStream,
        range: core::ops::Range<usize>,
    ) -> Result<(), CudaRuntimeError> {
        self.validate_range(&range)?;
        if self.submitted_range_count == OWNED_PINNED_H2D_MAX_SEGMENTS {
            return Err(CudaRuntimeError::validation(
                "owned_pinned_h2d_segments_capacity",
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        validate_same_device(
            "owned_pinned_h2d_segments",
            stream.device_ordinal(),
            self.buffer.device_ordinal,
        )?;
        let stream_owner_instance_key = stream.owner_instance_key()?;
        if self
            .stream_owner_instance_key
            .is_some_and(|owner| owner != stream_owner_instance_key)
        {
            return Err(CudaRuntimeError::invalid_metadata(
                "owned_pinned_h2d_segments_stream",
            ));
        }
        let destination = self
            .buffer
            .destination
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        let source = self
            .buffer
            .host
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?;
        // SAFETY: `range` is bounded by both equal-sized allocations. The
        // permit's immutable submitted-range ledger prevents host mutation
        // while CUDA may still read a range.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memcpy_h2d_submit_v1(
                self.buffer.device_ordinal,
                stream.raw()?,
                destination.raw,
                range.start,
                self.buffer.capacity_bytes,
                source.as_ptr().add(range.start).cast(),
                range.len(),
                range.len(),
            )
        };
        CudaRuntimeError::from_result("cuda_memcpy_h2d_async_range", result)?;
        if !self.buffer.state.pending() {
            self.buffer.state.mark_submitted();
        } else {
            self.buffer.state.mark_event_unreliable();
        }
        self.stream_owner_instance_key = Some(stream_owner_instance_key);
        self.submitted_ranges[self.submitted_range_count] = Some(range);
        self.submitted_range_count += 1;
        self.buffer.segmented_range_count = self.submitted_range_count;
        self.buffer.segmented_stream_owner_instance_key = Some(stream_owner_instance_key);
        let receipt = stream.submission_receipt()?;
        self.last_submission_receipt = Some(receipt);
        self.buffer.submission_receipt = Some(receipt);
        Ok(())
    }

    /// Records the single completion event after the final range.
    pub fn finish(
        mut self,
        stream: &OwnedCudaStream,
    ) -> Result<CudaStreamSubmissionReceipt, CudaRuntimeError> {
        let receipt = self
            .last_submission_receipt
            .ok_or_else(|| CudaRuntimeError::invalid_metadata("owned_pinned_h2d_segments_empty"))?;
        let stream_owner_instance_key = stream.owner_instance_key()?;
        if self.stream_owner_instance_key != Some(stream_owner_instance_key)
            || self.buffer.segmented_stream_owner_instance_key != Some(stream_owner_instance_key)
        {
            return Err(CudaRuntimeError::invalid_metadata(
                "owned_pinned_h2d_segments_stream",
            ));
        }
        if let Err(error) = self
            .buffer
            .completion
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("event"))?
            .record(stream)
        {
            self.buffer.state.mark_event_unreliable();
            if stream.synchronize().is_ok() {
                self.buffer.state.mark_completed();
                self.buffer.submission_receipt = None;
                self.buffer.clear_segmented_state();
            }
            return Err(error);
        }
        self.buffer.state.mark_event_recorded();
        self.last_submission_receipt = None;
        Ok(receipt)
    }
}

impl Drop for OwnedPinnedH2dSegments<'_> {
    fn drop(&mut self) {
        if self.last_submission_receipt.is_some() && self.buffer.state.pending() {
            self.buffer.state.mark_event_unreliable();
        }
    }
}

impl Drop for OwnedPinnedH2dBuffer {
    fn drop(&mut self) {
        let _ = self.wait();
        if self.state.pending() {
            self.leak_owned_resources();
        }
    }
}

/// Owns one page-locked host destination for a bounded asynchronous D2H
/// receipt copy.
///
/// This owner deliberately has no device allocation of its own. The source is
/// a request/workspace view that must remain live until the covering proof is
/// supplied. The unsafe submit boundary is the only place that accepts that
/// non-owning source; after enqueue, the host bytes cannot be read or reused
/// until the event/stream proof is checked.
#[derive(Debug)]
pub struct OwnedPinnedD2hBuffer {
    host: Option<OwnedCudaHostBuffer>,
    completion: Option<OwnedCudaEvent>,
    device_ordinal: i32,
    capacity_bytes: usize,
    state: PinnedH2dSlotState,
    submission_receipt: Option<CudaStreamSubmissionReceipt>,
}

impl OwnedPinnedD2hBuffer {
    pub fn allocate(device_ordinal: i32, capacity_bytes: usize) -> Result<Self, CudaRuntimeError> {
        if capacity_bytes == 0 {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        Ok(Self {
            host: Some(OwnedCudaHostBuffer::allocate(
                device_ordinal,
                capacity_bytes,
            )?),
            completion: Some(OwnedCudaEvent::create(device_ordinal)?),
            device_ordinal,
            capacity_bytes,
            state: PinnedH2dSlotState::Idle,
            submission_receipt: None,
        })
    }

    pub const fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub const fn pending(&self) -> bool {
        self.state.pending()
    }

    pub fn submission_receipt(&self) -> Result<CudaStreamSubmissionReceipt, CudaRuntimeError> {
        if !self.state.pending() {
            return Err(CudaRuntimeError::invalid_metadata(
                "owned_pinned_d2h_buffer_not_pending",
            ));
        }
        self.submission_receipt.ok_or_else(|| {
            CudaRuntimeError::invalid_metadata("owned_pinned_d2h_buffer_submission_receipt")
        })
    }

    /// Returns the completion event for the most recent asynchronous D2H
    /// submission. Callers must pair it with the exact submission receipt
    /// returned by [`Self::submission_receipt`].
    pub fn completion_event(&self) -> Result<&OwnedCudaEvent, CudaRuntimeError> {
        self.completion
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("owned pinned d2h completion event"))
    }

    /// Enqueues a D2H copy without waiting on `stream`.
    ///
    /// # Safety
    ///
    /// The allocation that produced `source` must remain live and unchanged
    /// until a covering completion proof is passed to
    /// [`Self::read_after_covering_completion`]. The view's extent must be a
    /// live request/workspace allocation, not a fabricated device address.
    pub unsafe fn submit_from_view(
        &mut self,
        stream: &OwnedCudaStream,
        source: CudaBufferView,
    ) -> Result<CudaStreamSubmissionReceipt, CudaRuntimeError> {
        const OPERATION: &str = "cuda_memcpy_d2h_async_pinned_receipt";
        if self.state.pending() || self.submission_receipt.is_some() {
            return Err(CudaRuntimeError::invalid_metadata(
                "owned_pinned_d2h_buffer_pending",
            ));
        }
        validate_same_device(OPERATION, stream.device_ordinal(), self.device_ordinal)?;
        if source.device_ordinal() != self.device_ordinal
            || source.len_bytes() != self.capacity_bytes
        {
            return Err(CudaRuntimeError::validation(
                OPERATION,
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        source.checked_range(OPERATION)?;
        let destination = self
            .host
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?;
        // SAFETY: the caller upholds the source lifetime contract above;
        // destination is the live cudaHostAlloc owner and all extents/device
        // identities were validated. The native function only enqueues
        // cudaMemcpyAsync; the completion event below fences the host bytes.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memcpy_d2h_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                destination.as_ptr().cast(),
                self.capacity_bytes,
                source.device_ptr().raw,
                source.offset_bytes(),
                source.allocation_len_bytes(),
                self.capacity_bytes,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)?;
        self.state.mark_submitted();
        let receipt = stream.submission_receipt()?;
        let completion = self
            .completion
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("event"))?;
        if let Err(error) = completion.record_for_submission(stream, receipt) {
            // No stream handle is retained by this detached owner. If event
            // recording failed after enqueue, keep the allocation pending and
            // leak on drop rather than claiming a proof we do not have.
            self.state.mark_event_unreliable();
            return Err(error);
        }
        self.state.mark_event_recorded();
        self.submission_receipt = Some(receipt);
        Ok(receipt)
    }

    /// Waits on the private completion event. This is intended for a CPU
    /// consumer, never for the CUDA submission path.
    pub fn wait(&mut self) -> Result<(), CudaRuntimeError> {
        match self.state {
            PinnedH2dSlotState::Idle => Ok(()),
            PinnedH2dSlotState::EventPending => {
                let result = self
                    .completion
                    .as_ref()
                    .ok_or_else(|| CudaRuntimeError::closed("event"))?
                    .synchronize();
                match result {
                    Ok(()) => {
                        self.state.mark_completed();
                        self.submission_receipt = None;
                        Ok(())
                    }
                    Err(error) => {
                        self.state.mark_event_unreliable();
                        Err(error)
                    }
                }
            }
            PinnedH2dSlotState::Indeterminate => Err(CudaRuntimeError::invalid_metadata(
                "owned_pinned_d2h_buffer_indeterminate",
            )),
        }
    }

    /// Makes the copied bytes visible only after a proof for this exact
    /// stream submission. The returned slice borrows the owner, so it cannot
    /// overlap a subsequent submission or drop.
    pub fn read_after_covering_completion(
        &mut self,
        proof: &CudaStreamCompletionProof,
    ) -> Result<&[u8], CudaRuntimeError> {
        let receipt = self.submission_receipt()?;
        if !proof.covers_submission(receipt) {
            return Err(CudaRuntimeError::invalid_metadata(
                "owned_pinned_d2h_buffer_completion_proof",
            ));
        }
        self.state.mark_completed();
        self.submission_receipt = None;
        self.host
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?
            .as_slice()
    }

    /// CPU-consumer convenience wrapper. It performs an event wait, not a
    /// synchronous D2H copy; the CUDA submission thread remains untouched.
    pub fn wait_and_read(&mut self) -> Result<&[u8], CudaRuntimeError> {
        self.wait()?;
        self.host
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?
            .as_slice()
    }

    fn leak_owned_resources(&mut self) {
        if let Some(host) = self.host.take() {
            core::mem::forget(host);
        }
        if let Some(completion) = self.completion.take() {
            core::mem::forget(completion);
        }
    }
}

impl Drop for OwnedPinnedD2hBuffer {
    fn drop(&mut self) {
        let _ = self.wait();
        if self.state.pending() {
            self.leak_owned_resources();
        }
    }
}

pub struct PinnedH2dLane<'a> {
    destination: &'a OwnedCudaBuffer,
    stream: &'a OwnedCudaStream,
    host: Option<OwnedCudaHostBuffer>,
    completion: Option<OwnedCudaEvent>,
    completion_receipt: Option<CudaStreamSubmissionReceipt>,
    capacity_bytes: usize,
    state: PinnedH2dSlotState,
}

impl core::fmt::Debug for PinnedH2dLane<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("PinnedH2dLane")
            .field("device_ordinal", &self.stream.device_ordinal())
            .field("capacity_bytes", &self.capacity_bytes)
            .field("state", &self.state)
            .finish()
    }
}

impl<'a> PinnedH2dLane<'a> {
    pub fn new(
        destination: &'a OwnedCudaBuffer,
        stream: &'a OwnedCudaStream,
        capacity_bytes: usize,
    ) -> Result<Self, CudaRuntimeError> {
        validate_same_device(
            "pinned_h2d_lane",
            stream.device_ordinal(),
            destination.device_ordinal(),
        )?;
        Ok(Self {
            destination,
            stream,
            host: Some(OwnedCudaHostBuffer::allocate(
                stream.device_ordinal(),
                capacity_bytes,
            )?),
            completion: Some(OwnedCudaEvent::create(stream.device_ordinal())?),
            completion_receipt: None,
            capacity_bytes,
            state: PinnedH2dSlotState::Idle,
        })
    }

    pub const fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    /// Returns the event recorded after the most recent asynchronous copy.
    /// Consumers on another stream must wait on this event before touching
    /// the destination. The lane reuses the event only after `writable()`
    /// observes completion, preserving the bounded-slot lifetime contract.
    pub fn completion_event(&self) -> Result<&OwnedCudaEvent, CudaRuntimeError> {
        self.completion
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("pinned h2d completion event"))
    }

    pub fn writable(&mut self) -> Result<&mut [u8], CudaRuntimeError> {
        self.wait()?;
        self.host
            .as_mut()
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?
            .as_mut_slice()
    }

    /// Poll and reclaim a completed asynchronous copy without host or stream
    /// synchronization. `false` means the pinned bytes are still owned by DMA;
    /// callers must not inspect or overwrite them. An indeterminate event is
    /// rejected rather than guessed complete.
    pub fn try_reclaim(&mut self) -> Result<bool, CudaRuntimeError> {
        match self.state {
            PinnedH2dSlotState::Idle => Ok(true),
            PinnedH2dSlotState::Indeterminate => Err(CudaRuntimeError::invalid_metadata(
                "pinned_h2d_lane_indeterminate",
            )),
            PinnedH2dSlotState::EventPending => {
                let event = self
                    .completion
                    .as_ref()
                    .ok_or_else(|| CudaRuntimeError::closed("event"))?;
                let receipt = self
                    .completion_receipt
                    .ok_or_else(|| CudaRuntimeError::invalid_metadata("pinned_h2d_lane_receipt"))?;
                match self.stream.query_event_completion(event, receipt)? {
                    Some(proof) => {
                        if !proof.covers_submission(receipt)
                            || !proof.belongs_to_stream(self.stream)?
                        {
                            return Err(CudaRuntimeError::invalid_metadata(
                                "pinned_h2d_lane_completion_proof",
                            ));
                        }
                        self.state.mark_completed();
                        self.completion_receipt = None;
                        Ok(true)
                    }
                    None => Ok(false),
                }
            }
        }
    }

    pub fn submit(
        &mut self,
        destination_offset_bytes: usize,
        copy_len_bytes: usize,
    ) -> Result<(), CudaRuntimeError> {
        if self.state.pending() {
            return Err(CudaRuntimeError::invalid_metadata(
                "pinned_h2d_lane_pending",
            ));
        }
        let end = destination_offset_bytes
            .checked_add(copy_len_bytes)
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        if copy_len_bytes == 0
            || end > self.destination.len_bytes()
            || copy_len_bytes > self.capacity_bytes
        {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        let destination = self
            .destination
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("buffer"))?;
        let source = self
            .host
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?
            .raw
            .ok_or_else(|| CudaRuntimeError::closed("pinned host buffer"))?;
        // SAFETY: `destination` is the borrowed OwnedCudaBuffer's live device
        // allocation (spans its len_bytes; both it and the stream outlive the
        // lane via 'a) and `source` is the lane's pinned allocation spanning
        // host.len_bytes; the offset/len checks above keep the copy inside
        // both, and same-device was validated in `new`. The submit is async
        // and only the completion event recorded below fences `writable` and
        // Drop. `state` becomes Indeterminate immediately after a successful
        // enqueue; event-record failure can clear it only when a stream-wide
        // synchronization proves completion. Otherwise Drop leaks the owned
        // pinned allocation and event rather than freeing memory under DMA.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_memcpy_h2d_submit_v1(
                self.stream.device_ordinal(),
                self.stream.raw()?,
                destination.raw,
                destination_offset_bytes,
                self.destination.len_bytes(),
                source.as_ptr().cast(),
                self.capacity_bytes,
                copy_len_bytes,
            )
        };
        CudaRuntimeError::from_result("cuda_memcpy_h2d_async", result)?;
        self.state.mark_submitted();
        let receipt = self.stream.submission_receipt()?;
        if let Err(error) = self
            .completion
            .as_ref()
            .ok_or_else(|| CudaRuntimeError::closed("event"))?
            .record_for_submission(self.stream, receipt)
        {
            self.state
                .observe_stream_synchronize(self.stream.synchronize().is_ok());
            return Err(error);
        }
        self.state.mark_event_recorded();
        self.completion_receipt = Some(receipt);
        Ok(())
    }

    pub fn wait(&mut self) -> Result<(), CudaRuntimeError> {
        match self.state {
            PinnedH2dSlotState::Idle => Ok(()),
            PinnedH2dSlotState::EventPending => {
                let result = self
                    .completion
                    .as_ref()
                    .ok_or_else(|| CudaRuntimeError::closed("event"))?
                    .synchronize();
                match result {
                    Ok(()) => {
                        self.state.mark_completed();
                        self.completion_receipt = None;
                        Ok(())
                    }
                    Err(error) => {
                        self.state.mark_event_unreliable();
                        self.state
                            .observe_stream_synchronize(self.stream.synchronize().is_ok());
                        Err(error)
                    }
                }
            }
            PinnedH2dSlotState::Indeterminate => {
                let result = self.stream.synchronize();
                self.state.observe_stream_synchronize(result.is_ok());
                result
            }
        }
    }

    fn leak_owned_resources(&mut self) {
        if let Some(host) = self.host.take() {
            core::mem::forget(host);
        }
        if let Some(completion) = self.completion.take() {
            core::mem::forget(completion);
        }
    }
}

impl Drop for PinnedH2dLane<'_> {
    fn drop(&mut self) {
        let _ = self.wait();
        if self.state.pending() {
            self.leak_owned_resources();
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) const fn test_buffer_view(
    raw: u64,
    device_ordinal: i32,
    offset_bytes: usize,
    len_bytes: usize,
) -> CudaBufferView {
    test_buffer_view_with_owner(raw, device_ordinal, 0, offset_bytes, len_bytes)
}

#[cfg(test)]
pub(crate) const fn test_buffer_view_with_owner(
    raw: u64,
    device_ordinal: i32,
    allocation_owner_instance_id: u64,
    offset_bytes: usize,
    len_bytes: usize,
) -> CudaBufferView {
    CudaBufferView {
        ptr: CudaDevicePtr {
            raw,
            device_ordinal,
        },
        allocation_owner_instance_id,
        allocation_len_bytes: offset_bytes.saturating_add(len_bytes),
        offset_bytes,
        len_bytes,
    }
}

mod tests {
    #[test]
    #[ignore = "requires exclusive CUDA owner"]
    fn feature_shift_rows_bitwise_and_admission() {
        use super::*;
        let before = cuda_device_allocation_snapshot();
        let mut stream = OwnedCudaStream::create(0).unwrap();
        for rows in [1usize, 9, 33, 2048] {
            for words in [1usize, 257, 25600] {
                let bytes = rows * words * 4;
                let mut buffer = OwnedCudaBuffer::allocate(0, bytes).unwrap();
                let original: Vec<u8> = (0..rows * words)
                    .flat_map(|i| ((i as u32).wrapping_mul(2654435761)
                        ^ ((i >> 12) as u32).wrapping_mul(2246822519)).to_le_bytes())
                    .collect();
                for shift in 0..=rows.min(8) {
                    for retained in [(rows - shift) / 2, rows - shift] {
                        buffer.copy_from_host_sync(&stream, &original).unwrap();
                        let mut expected = original.clone();
                        if shift > 0 {
                            expected.copy_within(shift * words * 4..(shift + retained) * words * 4, 0);
                        }
                        unsafe { buffer.shift_rows_u32_async_unchecked(&stream, words * 4, retained, shift).unwrap(); }
                        let mut actual = vec![0; bytes];
                        buffer.copy_to_host_sync(&stream, &mut actual).unwrap();
                        assert_eq!(actual, expected, "rows={rows} words={words} shift={shift} retained={retained}");
                    }
                }
                for (pitch, retained, shift) in [(0,0,0),(3,0,0),(words*4,rows+1,0),(words*4,1,rows),(words*4,0,usize::MAX)] {
                    assert!(unsafe { buffer.shift_rows_u32_async_unchecked(&stream,pitch,retained,shift) }.is_err());
                }
                // Exercise native rejection independently of the Rust wrapper.
                let raw = buffer.raw.unwrap().raw;
                for (ptr,len,pitch,retained,shift) in [
                    (0,bytes,words*4,1,1),(raw+1,bytes,words*4,1,1),
                    (raw,bytes,0,1,1),(raw,bytes,3,1,1),(raw,bytes-1,words*4,0,0),
                    (raw,bytes,words*4,1,rows),(raw,bytes,words*4,rows+1,0),
                    (raw,bytes,words*4,0,usize::MAX),(u64::MAX-3,bytes,words*4,0,0),
                ] {
                    let r = unsafe { augur_cuda_sys::augur_cuda_shift_rows_u32_submit_v1(0,stream.raw().unwrap(),ptr,len,pitch,retained,shift) };
                    assert_eq!(r.status,AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
                }
                buffer.close().unwrap();
                assert!(unsafe { buffer.shift_rows_u32_async_unchecked(&stream,words*4,0,0) }.is_err());
            }
        }
        stream.close().unwrap();
        let after = cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes,before.live_bytes);
        assert_eq!(after.live_allocations,before.live_allocations);
        assert_eq!(after.successful_allocations-before.successful_allocations,
            after.successful_frees-before.successful_frees);
    }

    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn cuda_buffer_quarantine_closes_owner_without_freeing_unknown_allocation() {
        let mut owner = OwnedCudaBuffer {
            raw: Some(CudaDevicePtr {
                raw: 0x1000,
                device_ordinal: 0,
            }),
            device_ordinal: 0,
            allocation_owner_instance_id: 17,
            len_bytes: 4096,
            d2d_state: CudaD2dOwnerState::Idle,
        };
        owner.quarantine_after_unproven_completion();
        assert!(owner.device_ptr().is_none());
        assert_eq!(owner.len_bytes(), 0);
        assert!(!owner.has_pending_d2d());
    }

    #[test]
    fn different_device_ordinals_are_rejected() {
        let error = validate_same_device("test_device_match", 0, 1).unwrap_err();

        assert_eq!(error.operation, "test_device_match");
        assert_eq!(error.kind, CudaRuntimeErrorKind::DeviceMismatch);
        assert_eq!(error.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
        assert_eq!(error.raw_cuda_status, 0);
    }

    #[test]
    fn checked_2d_footprint_rejects_invalid_pitch_and_overflow() {
        assert_eq!(
            checked_2d_footprint("test_2d", 0, 4, 8).unwrap_err().kind,
            CudaRuntimeErrorKind::InvalidDimensions
        );
        assert_eq!(
            checked_2d_footprint("test_2d", 9, 4, 8).unwrap_err().kind,
            CudaRuntimeErrorKind::InvalidDimensions
        );
        assert_eq!(
            checked_2d_footprint("test_2d", usize::MAX, 2, usize::MAX)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::RangeOverflow
        );
        assert_eq!(checked_2d_footprint("test_2d", 4, 4, 8).unwrap(), 28);
    }

    #[test]
    fn pending_owner_2d_copy_enforces_geometry_identity_and_terminal_proof() {
        let mut stream = OwnedCudaStream::create(0).unwrap();
        let mut source = OwnedCudaBuffer::allocate(0, 32).unwrap();
        let mut destination = OwnedCudaBuffer::allocate(0, 48).unwrap();
        let source_bytes = (0_u8..32).collect::<Vec<_>>();
        source.copy_from_host_sync(&stream, &source_bytes).unwrap();
        let source_view = source.view(1, 31).unwrap();

        let invalid_pitch = stream.submission_receipt().unwrap();
        let error = unsafe {
            destination.copy_from_device_2d_async_pending_unchecked(
                &stream,
                invalid_pitch,
                0,
                8,
                source_view,
                3,
                4,
                4,
            )
        }
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::InvalidDimensions);
        assert!(!destination.has_pending_d2d());

        let out_of_bounds = stream.submission_receipt().unwrap();
        let error = unsafe {
            destination.copy_from_device_2d_async_pending_unchecked(
                &stream,
                out_of_bounds,
                0,
                12,
                source_view,
                8,
                4,
                5,
            )
        }
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::OutOfBounds);
        assert!(!destination.has_pending_d2d());

        let foreign = CudaBufferView {
            ptr: CudaDevicePtr {
                raw: source_view.device_ptr().raw_value(),
                device_ordinal: 1,
            },
            ..source_view
        };
        let foreign_receipt = stream.submission_receipt().unwrap();
        let error = unsafe {
            destination.copy_from_device_2d_async_pending_unchecked(
                &stream,
                foreign_receipt,
                0,
                12,
                foreign,
                8,
                4,
                4,
            )
        }
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::DeviceMismatch);
        assert!(!destination.has_pending_d2d());

        let destination_view = destination.view(0, 48).unwrap();
        let overlap_receipt = stream.submission_receipt().unwrap();
        let error = unsafe {
            destination.copy_from_device_2d_async_pending_unchecked(
                &stream,
                overlap_receipt,
                0,
                12,
                destination_view,
                12,
                4,
                4,
            )
        }
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::OverlappingDeviceRanges);
        assert!(!destination.has_pending_d2d());

        let receipt = stream.submission_receipt().unwrap();
        unsafe {
            destination
                .copy_from_device_2d_async_pending_unchecked(
                    &stream,
                    receipt,
                    2,
                    12,
                    source_view,
                    8,
                    4,
                    4,
                )
                .unwrap();
        }
        let stale = CudaStreamCompletionProof {
            device_ordinal: receipt.device_ordinal,
            stream_owner_instance_id: receipt.stream_owner_instance_id,
            submission_epoch: receipt.submission_epoch.saturating_sub(1),
        };
        assert!(destination
            .release_pending_d2d_after_covering_completion(&stale)
            .is_err());
        let proof = stream.synchronize_submission(receipt).unwrap();
        destination
            .release_pending_d2d_after_covering_completion(&proof)
            .unwrap();
        let mut actual = [0_u8; 48];
        destination.copy_to_host_sync(&stream, &mut actual).unwrap();
        for row in 0..4 {
            assert_eq!(
                &actual[row * 12 + 2..row * 12 + 6],
                &source_bytes[row * 8 + 1..row * 8 + 5]
            );
        }
        source.close().unwrap();
        destination.close().unwrap();
        stream.close().unwrap();
    }

    #[test]
    fn d2d_registered_view_rejects_address_and_extent_aba() {
        let owner = CudaD2dBatchOwner {
            buffer: NonNull::dangling(),
            allocation_address: 0x4000,
            allocation_owner_instance_id: 19,
            allocation_len_bytes: 4096,
            touched: false,
        };
        let current = CudaBufferView {
            ptr: CudaDevicePtr {
                raw: 0x4000,
                device_ordinal: 0,
            },
            allocation_owner_instance_id: 19,
            allocation_len_bytes: 4096,
            offset_bytes: 128,
            len_bytes: 256,
        };
        let stale = CudaBufferView {
            allocation_owner_instance_id: 18,
            ..current
        };

        assert!(owner.matches_view(current));
        assert!(!owner.matches_view(stale));
        assert_ne!(current, stale);
        assert!(!format!("{current:?}").contains("allocation_owner_instance_id"));
    }

    #[test]
    fn buffer_view_owner_validation_rejects_incarnation_aba() {
        let owner = core::mem::ManuallyDrop::new(OwnedCudaBuffer {
            raw: Some(CudaDevicePtr {
                raw: 0x8000,
                device_ordinal: 0,
            }),
            device_ordinal: 0,
            allocation_owner_instance_id: 41,
            len_bytes: 4096,
            d2d_state: CudaD2dOwnerState::Idle,
        });
        let current = CudaBufferView {
            ptr: owner.raw.unwrap(),
            allocation_owner_instance_id: 41,
            allocation_len_bytes: 4096,
            offset_bytes: 256,
            len_bytes: 512,
        };
        let stale = CudaBufferView {
            allocation_owner_instance_id: 40,
            ..current
        };

        assert_eq!(current.validate_owner(&owner, "test_owner"), Ok(()));
        let error = stale.validate_owner(&owner, "test_owner").unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::InvalidMetadata);
    }

    #[test]
    fn d2d_batch_external_stream_access_is_pending_before_the_first_copy() {
        let stream = OwnedCudaStream::create(0).unwrap();
        let mut source = OwnedCudaBuffer::allocate(0, 64).unwrap();

        {
            let mut batch = CudaD2dBatch::<1>::new(&stream).unwrap();
            let source_owner = batch.register(&mut source).unwrap();
            // SAFETY: this state-machine test intentionally models an
            // uncertain same-stream submission and obtains no proof that
            // could release the owner afterward.
            unsafe { batch.arm_owner_for_stream_access(source_owner) }.unwrap();
        }

        assert_eq!(
            source
                .copy_to_host_sync(&stream, &mut [0_u8; 64])
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::InvalidMetadata
        );
    }

    #[test]
    fn d2d_registered_view_arm_is_owner_exact_and_zero_copy_releasable() {
        let mut stream = OwnedCudaStream::create(0).unwrap();
        let mut first = OwnedCudaBuffer::allocate(0, 64).unwrap();
        let mut second = OwnedCudaBuffer::allocate(0, 64).unwrap();
        let first_view = first.view(0, 64).unwrap();
        let second_view = second.view(0, 64).unwrap();

        {
            let mut batch = CudaD2dBatch::<2>::new(&stream).unwrap();
            let first_owner = batch.register(&mut first).unwrap();
            let _second_owner = batch.register(&mut second).unwrap();
            let mismatch =
                unsafe { batch.arm_registered_view_for_stream_access(first_owner, second_view) }
                    .unwrap_err();
            assert_eq!(mismatch.kind, CudaRuntimeErrorKind::InvalidMetadata);
            assert!(batch.completion_receipt().is_err());

            let armed =
                unsafe { batch.arm_registered_view_for_stream_access(first_owner, first_view) }
                    .unwrap();
            assert_eq!(armed, first_view);
            let receipt = batch.completion_receipt().unwrap();
            let proof = stream.synchronize_submission(receipt).unwrap();
            batch.release_after_completion(&proof).unwrap();
        }

        assert!(first.view(0, 64).is_ok());
        first.close().unwrap();
        second.close().unwrap();
        stream.close().unwrap();
    }

    #[test]
    fn d2d_registered_view_can_rearm_after_an_earlier_same_batch_copy() {
        let mut stream = OwnedCudaStream::create(0).unwrap();
        let mut source = OwnedCudaBuffer::allocate(0, 64).unwrap();
        let mut destination = OwnedCudaBuffer::allocate(0, 64).unwrap();
        let source_view = source.view(0, 64).unwrap();

        {
            let mut batch = CudaD2dBatch::<2>::new(&stream).unwrap();
            let source_owner = batch.register(&mut source).unwrap();
            let destination_owner = batch.register(&mut destination).unwrap();
            unsafe { batch.arm_registered_view_for_stream_access(source_owner, source_view) }
                .unwrap();
            batch
                .copy_from_registered_view(destination_owner, 0, source_view)
                .unwrap();
            assert_eq!(
                unsafe { batch.arm_registered_view_for_stream_access(source_owner, source_view) }
                    .unwrap(),
                source_view
            );
            let receipt = batch.completion_receipt().unwrap();
            let proof = stream.synchronize_submission(receipt).unwrap();
            batch.release_after_completion(&proof).unwrap();
        }

        source.close().unwrap();
        destination.close().unwrap();
        stream.close().unwrap();
    }

    #[test]
    fn pending_owner_accepts_same_receipt_multirange_and_requires_covering_proof() {
        let mut stream = OwnedCudaStream::create(0).unwrap();
        let mut source = OwnedCudaBuffer::allocate(0, 128).unwrap();
        let mut destination = OwnedCudaBuffer::allocate(0, 128).unwrap();
        let source_bytes = (0_u8..128).collect::<Vec<_>>();
        source.copy_from_host_sync(&stream, &source_bytes).unwrap();
        let first = source.view(0, 64).unwrap();
        let second = source.view(64, 64).unwrap();
        let receipt = stream.submission_receipt().unwrap();

        // SAFETY: both owners and the stream remain live until the exact
        // receipt is synchronized below. No destination view escapes.
        unsafe {
            destination
                .copy_from_device_async_pending_unchecked(&stream, receipt, 64, first)
                .unwrap();
            destination
                .copy_from_device_async_pending_unchecked(&stream, receipt, 0, second)
                .unwrap();
        }
        assert!(destination.view(0, 128).is_err());
        assert!(destination.close().is_err());
        let stale = CudaStreamCompletionProof {
            device_ordinal: receipt.device_ordinal,
            stream_owner_instance_id: receipt.stream_owner_instance_id,
            submission_epoch: receipt.submission_epoch - 1,
        };
        assert!(destination
            .release_pending_d2d_after_covering_completion(&stale)
            .is_err());

        let proof = stream.synchronize_submission(receipt).unwrap();
        destination
            .release_pending_d2d_after_covering_completion(&proof)
            .unwrap();
        let mut actual = [0_u8; 128];
        destination.copy_to_host_sync(&stream, &mut actual).unwrap();
        assert_eq!(&actual[..64], &source_bytes[64..]);
        assert_eq!(&actual[64..], &source_bytes[..64]);

        source.close().unwrap();
        destination.close().unwrap();
        stream.close().unwrap();
    }

    #[test]
    fn negative_ordinal_is_rejected() {
        assert_eq!(
            validate_device_ordinal(-1).unwrap_err().kind,
            CudaRuntimeErrorKind::InvalidDeviceOrdinal,
        );
    }

    #[test]
    fn stream_owner_ids_are_monotonic_and_not_address_tokens() {
        let first = next_cuda_stream_owner_id().unwrap();
        let second = next_cuda_stream_owner_id().unwrap();
        assert!(second > first);
    }

    #[test]
    fn stream_close_receipt_digest_matches_v2_recipe() {
        assert_eq!(
            cuda_stream_close_receipt_sha256(0, 17, 29),
            [
                0x7b, 0x26, 0x5a, 0xe3, 0x34, 0xfc, 0x12, 0xf5, 0x3d, 0x51, 0x9b, 0xfb, 0x77, 0x04,
                0x75, 0xb5, 0x54, 0x7d, 0xb7, 0xf6, 0x23, 0x21, 0x3e, 0x61, 0x48, 0xc5, 0x51, 0xcb,
                0x0d, 0x3f, 0x87, 0x73,
            ]
        );
    }

    #[test]
    fn d2d_owner_state_requires_exact_stream_epoch_proof() {
        let receipt = CudaStreamSubmissionReceipt {
            device_ordinal: 0,
            stream_owner_instance_id: 7,
            submission_epoch: 11,
        };
        let mut state = CudaD2dOwnerState::Idle;
        state.mark_pending(receipt);

        let wrong_stream = CudaStreamCompletionProof {
            device_ordinal: 0,
            stream_owner_instance_id: 8,
            submission_epoch: 11,
        };
        let stale_epoch = CudaStreamCompletionProof {
            device_ordinal: 0,
            stream_owner_instance_id: 7,
            submission_epoch: 10,
        };
        assert!(!state.matches_proof(&wrong_stream));
        assert!(!state.matches_proof(&stale_epoch));
        assert!(state.pending());

        let exact = CudaStreamCompletionProof {
            device_ordinal: 0,
            stream_owner_instance_id: 7,
            submission_epoch: 11,
        };
        assert!(state.matches_proof(&exact));
        state.release();
        assert_eq!(state, CudaD2dOwnerState::Idle);
    }

    #[test]
    fn completion_proof_covers_only_earlier_epochs_on_the_same_stream() {
        let earlier = CudaStreamSubmissionReceipt {
            device_ordinal: 0,
            stream_owner_instance_id: 7,
            submission_epoch: 11,
        };
        let later = CudaStreamCompletionProof {
            device_ordinal: 0,
            stream_owner_instance_id: 7,
            submission_epoch: 13,
        };
        assert!(later.covers_submission(earlier));
        assert!(!later.covers_submission(CudaStreamSubmissionReceipt {
            submission_epoch: 14,
            ..earlier
        }));
        assert!(!later.covers_submission(CudaStreamSubmissionReceipt {
            stream_owner_instance_id: 8,
            ..earlier
        }));
        assert!(!later.covers_submission(CudaStreamSubmissionReceipt {
            device_ordinal: 1,
            ..earlier
        }));
    }

    #[test]
    fn event_completion_binding_requires_exact_owner_and_submission_epoch() {
        assert!(validate_event_submission_binding(Some(7), Some(11), 7, 11).is_ok());

        let unbound = validate_event_submission_binding(Some(7), None, 7, 11).unwrap_err();
        assert_eq!(unbound.operation, "cuda_stream_event_submission_epoch");
        assert_eq!(unbound.kind, CudaRuntimeErrorKind::InvalidMetadata);

        let stale = validate_event_submission_binding(Some(7), Some(10), 7, 11).unwrap_err();
        assert_eq!(stale.operation, "cuda_stream_event_submission_epoch");
        assert_eq!(stale.kind, CudaRuntimeErrorKind::InvalidMetadata);

        let newer = validate_event_submission_binding(Some(7), Some(12), 7, 11).unwrap_err();
        assert_eq!(newer.operation, "cuda_stream_event_submission_epoch");
        assert_eq!(newer.kind, CudaRuntimeErrorKind::InvalidMetadata);

        let foreign = validate_event_submission_binding(Some(8), Some(11), 7, 11).unwrap_err();
        assert_eq!(foreign.operation, "cuda_stream_event_owner");
        assert_eq!(foreign.kind, CudaRuntimeErrorKind::InvalidMetadata);
    }

    #[test]
    fn d2d_owner_state_stays_pending_after_submission_failure() {
        let receipt = CudaStreamSubmissionReceipt {
            device_ordinal: 0,
            stream_owner_instance_id: 17,
            submission_epoch: 19,
        };
        let mut state = CudaD2dOwnerState::Idle;
        state.mark_pending(receipt);

        let error = state
            .observe_submission_result(
                "cuda_d2d_batch_copy",
                AugurCudaResultV1 {
                    status: augur_cuda_sys::AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,
                    raw_cuda_status: 700,
                },
            )
            .unwrap_err();

        assert_eq!(error.kind, CudaRuntimeErrorKind::Native);
        assert!(state.pending());
        assert!(state.matches_proof(&CudaStreamCompletionProof {
            device_ordinal: 0,
            stream_owner_instance_id: 17,
            submission_epoch: 19,
        }));
    }

    #[test]
    fn subview_preserves_absolute_offset_and_fails_closed() {
        let view = test_buffer_view(0x1000, 2, 64, 128);
        let nested = view.subview(32, 48).unwrap();
        assert_eq!(nested.device_ptr().raw_value(), 0x1000);
        assert_eq!(nested.device_ordinal(), 2);
        assert_eq!(nested.offset_bytes(), 96);
        assert_eq!(nested.len_bytes(), 48);
        assert_eq!(
            view.subview(120, 16).unwrap_err().kind,
            CudaRuntimeErrorKind::OutOfBounds
        );
        assert_eq!(
            view.subview(usize::MAX, 1).unwrap_err().kind,
            CudaRuntimeErrorKind::RangeOverflow
        );
    }

    #[test]
    fn pinned_h2d_slot_state_never_clears_without_completion() {
        let mut state = PinnedH2dSlotState::Idle;
        state.mark_submitted();
        assert_eq!(state, PinnedH2dSlotState::Indeterminate);
        assert!(state.pending());

        state.mark_event_recorded();
        assert_eq!(state, PinnedH2dSlotState::EventPending);
        state.mark_event_unreliable();
        assert_eq!(state, PinnedH2dSlotState::Indeterminate);
        assert!(state.pending());

        state.mark_completed();
        assert_eq!(state, PinnedH2dSlotState::Idle);
        assert!(!state.pending());
    }

    #[test]
    fn pinned_h2d_event_record_and_stream_sync_double_failure_stays_indeterminate() {
        let mut state = PinnedH2dSlotState::Idle;
        state.mark_submitted();

        state.observe_stream_synchronize(false);

        assert_eq!(state, PinnedH2dSlotState::Indeterminate);
        assert!(state.pending());
    }

    #[test]
    fn pinned_h2d_unreliable_event_requires_stream_completion_to_clear() {
        let mut state = PinnedH2dSlotState::Idle;
        state.mark_submitted();
        state.mark_event_recorded();
        state.mark_event_unreliable();

        state.observe_stream_synchronize(false);
        assert_eq!(state, PinnedH2dSlotState::Indeterminate);
        assert!(state.pending());

        state.observe_stream_synchronize(true);
        assert_eq!(state, PinnedH2dSlotState::Idle);
        assert!(!state.pending());
    }

    #[test]
    fn fresh_pinned_host_buffer_is_zero_initialized() {
        let buffer = OwnedCudaHostBuffer::allocate(0, 257).unwrap();

        assert_eq!(buffer.as_slice().unwrap(), &[0_u8; 257]);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn segmented_pinned_h2d_owner_is_byte_exact_and_releasable_by_one_proof() {
        const RANGE_BYTES: usize = 256;
        const RANGE_COUNT: usize = 20;
        let capacity_bytes = RANGE_BYTES * RANGE_COUNT;
        let stream = OwnedCudaStream::create(0).unwrap();
        let mut owner = OwnedPinnedH2dBuffer::allocate(0, capacity_bytes).unwrap();

        let receipt = {
            let mut segments = owner.begin_segments().unwrap();
            for index in 0..RANGE_COUNT {
                let start = index * RANGE_BYTES;
                let range = start..start + RANGE_BYTES;
                let host = segments.writable_range(range.clone()).unwrap();
                host.fill((index as u8).wrapping_mul(17).wrapping_add(3));
                segments.submit_range(&stream, range).unwrap();
            }
            segments.finish(&stream).unwrap()
        };

        let completion = OwnedCudaEvent::create(0)
            .unwrap()
            .record_and_synchronize_submission(&stream, receipt)
            .unwrap();
        owner
            .release_after_covering_completion(&completion)
            .unwrap();
        assert!(!owner.pending());

        let mut actual = vec![0_u8; capacity_bytes];
        owner
            .device_view()
            .unwrap()
            .copy_to_host_sync(&stream, &mut actual)
            .unwrap();
        for index in 0..RANGE_COUNT {
            let start = index * RANGE_BYTES;
            let expected = (index as u8).wrapping_mul(17).wrapping_add(3);
            assert!(actual[start..start + RANGE_BYTES]
                .iter()
                .all(|value| *value == expected));
        }
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn pinned_h2d_dependency_release_requires_dependent_terminal_proof() {
        let source = OwnedCudaStream::create(0).unwrap();
        let dependent = OwnedCudaStream::create(0).unwrap();
        let foreign = OwnedCudaStream::create(0).unwrap();
        let mut owner = OwnedPinnedH2dBuffer::allocate(0, 256).unwrap();
        owner.writable().unwrap().fill(0x3c);
        owner.submit(&source).unwrap();

        let dependency = owner.order_completion_on(&dependent).unwrap();
        assert!(dependency
            .dependent_submission()
            .belongs_to_stream(&dependent)
            .unwrap());
        assert!(!dependency
            .dependent_submission()
            .belongs_to_stream(&foreign)
            .unwrap());
        let foreign_receipt = foreign.submission_receipt().unwrap();
        let foreign_proof = foreign.synchronize_submission(foreign_receipt).unwrap();
        assert_eq!(
            owner
                .release_after_dependent_completion(dependency, &foreign_proof)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::InvalidMetadata
        );
        assert!(owner.pending());

        let proof = dependent
            .synchronize_submission(dependency.dependent_submission())
            .unwrap();
        owner
            .release_after_dependent_completion(dependency, &proof)
            .unwrap();
        assert!(!owner.pending());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn segmented_pinned_h2d_owner_rejects_overlap_and_capacity_without_poisoning_cleanup() {
        let stream = OwnedCudaStream::create(0).unwrap();
        let mut owner = OwnedPinnedH2dBuffer::allocate(0, 1024).unwrap();
        let mut segments = owner.begin_segments().unwrap();
        segments.writable_range(0..512).unwrap().fill(0x5a);
        segments.submit_range(&stream, 0..512).unwrap();
        let overlap = segments.submit_range(&stream, 256..768).unwrap_err();
        assert_eq!(overlap.kind, CudaRuntimeErrorKind::InvalidLayout);
        let receipt = segments.finish(&stream).unwrap();
        let completion = OwnedCudaEvent::create(0)
            .unwrap()
            .record_and_synchronize_submission(&stream, receipt)
            .unwrap();
        owner
            .release_after_covering_completion(&completion)
            .unwrap();

        let range_width = 4;
        let mut capacity_owner =
            OwnedPinnedH2dBuffer::allocate(0, (OWNED_PINNED_H2D_MAX_SEGMENTS + 1) * range_width)
                .unwrap();
        let mut capacity_segments = capacity_owner.begin_segments().unwrap();
        for index in 0..OWNED_PINNED_H2D_MAX_SEGMENTS {
            let start = index * range_width;
            capacity_segments
                .submit_range(&stream, start..start + range_width)
                .unwrap();
        }
        let capacity_error = capacity_segments
            .submit_range(
                &stream,
                OWNED_PINNED_H2D_MAX_SEGMENTS * range_width
                    ..(OWNED_PINNED_H2D_MAX_SEGMENTS + 1) * range_width,
            )
            .unwrap_err();
        assert_eq!(capacity_error.kind, CudaRuntimeErrorKind::InvalidDimensions);
        let receipt = capacity_segments.finish(&stream).unwrap();
        let completion = OwnedCudaEvent::create(0)
            .unwrap()
            .record_and_synchronize_submission(&stream, receipt)
            .unwrap();
        capacity_owner
            .release_after_covering_completion(&completion)
            .unwrap();
        assert!(!capacity_owner.pending());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn segmented_pinned_h2d_owner_drop_keeps_unproven_submission_pending() {
        let stream = OwnedCudaStream::create(0).unwrap();
        let mut owner = OwnedPinnedH2dBuffer::allocate(0, 64).unwrap();
        {
            let mut segments = owner.begin_segments().unwrap();
            segments.submit_range(&stream, 0..64).unwrap();
        }
        assert!(owner.pending());
        assert_eq!(
            owner.wait().unwrap_err().kind,
            CudaRuntimeErrorKind::InvalidMetadata
        );
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn device_zero_sync_overwrites_the_complete_allocation() {
        let stream = OwnedCudaStream::create(0).unwrap();
        let mut buffer = OwnedCudaBuffer::allocate(0, 257).unwrap();
        buffer
            .copy_from_host_sync(&stream, &[0xa5_u8; 257])
            .unwrap();

        buffer.zero_sync(&stream).unwrap();

        let mut actual = [0xff_u8; 257];
        buffer.copy_to_host_sync(&stream, &mut actual).unwrap();
        assert_eq!(actual, [0_u8; 257]);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn device_zero_prefix_preserves_tail_and_rejects_oversize() {
        let stream = OwnedCudaStream::create(0).unwrap();
        let mut buffer = OwnedCudaBuffer::allocate(0, 257).unwrap();
        buffer.copy_from_host_sync(&stream, &[0xa5; 257]).unwrap();
        assert!(buffer.zero_prefix_sync(&stream, 258).is_err());
        assert!(buffer.zero_prefix_sync(&stream, usize::MAX).is_err());
        buffer.zero_prefix_sync(&stream, 0).unwrap();
        let mut actual = [0; 257];
        buffer.copy_to_host_sync(&stream, &mut actual).unwrap();
        assert_eq!(actual, [0xa5; 257]);
        buffer.zero_prefix_sync(&stream, 3).unwrap();
        buffer.copy_to_host_sync(&stream, &mut actual).unwrap();
        assert_eq!(&actual[..3], &[0; 3]);
        assert_eq!(&actual[3..], &[0xa5; 254]);
        buffer.zero_prefix_sync(&stream, 257).unwrap();
        buffer.copy_to_host_sync(&stream, &mut actual).unwrap();
        assert_eq!(actual, [0; 257]);
    }

    #[test]
    #[ignore = "exclusive CUDA device; full physical KV262144 reset component screen"]
    fn device_zero_kv_capacity_prefix_screen() {
        use std::time::Instant;
        let before = cuda_device_allocation_snapshot();
        let stream = OwnedCudaStream::create(0).unwrap();
        let capacity = 262144usize;
        let row_bytes = 1088usize;
        let mut buffers = Vec::new();
        for _ in 0..32 {
            let mut buffer = OwnedCudaBuffer::allocate(0, capacity * row_bytes).unwrap();
            // As in the model's empty(): cudaMalloc alone is not initialization.
            buffer.zero_sync(&stream).unwrap();
            buffers.push(buffer);
        }
        let allocated = cuda_device_allocation_snapshot();
        for rows in [256usize, 8448, capacity] {
            for repeat in 0..4 {
                let order = if repeat % 2 == 0 { [false, true] } else { [true, false] };
                for prefix in order {
                    let start = Instant::now();
                    for buffer in &mut buffers {
                        buffer.zero_prefix_sync(&stream, if prefix { rows * row_bytes } else { capacity * row_bytes }).unwrap();
                    }
                    println!("KV_RESET_SCREEN rows={rows} repeat={repeat} prefix={prefix} wall_ms={:.6}",
                             start.elapsed().as_secs_f64() * 1000.0);
                }
            }
        }
        let measured = cuda_device_allocation_snapshot();
        assert_eq!(allocated.successful_allocations, measured.successful_allocations);
        assert_eq!(allocated.successful_frees, measured.successful_frees);
        // Audit real nonzero written prefixes, including the largest timed prefix.
        // Startup initialized every byte beyond the prefix; inspect full capacity.
        for buffer in &mut buffers {
            buffer.copy_from_host_sync(&stream, &vec![0x5a; 8448 * row_bytes]).unwrap();
            buffer.zero_prefix_sync(&stream, 8448 * row_bytes).unwrap();
            let mut actual = vec![0xff; capacity * row_bytes];
            buffer.copy_to_host_sync(&stream, &mut actual).unwrap();
            assert!(actual.iter().all(|&v| v == 0));
        }
        drop(buffers);
        let after = cuda_device_allocation_snapshot();
        assert_eq!(before.live_bytes, after.live_bytes);
        assert_eq!(before.live_allocations, after.live_allocations);
    }

    #[test]
    #[ignore = "requires a CUDA device; isolated recurrent reset component screen"]
    fn device_zero_shared_recurrent_reset_screen() {
        use std::{rc::Rc, time::Instant};
        let before = cuda_device_allocation_snapshot();
        let stream = OwnedCudaStream::create(0).unwrap();
        let mut buffers = Vec::new();
        for _ in 0..48 {
            for bytes in [10240 * 3 * 4, 48 * 128 * 128 * 4] {
                let buffer = Rc::new(OwnedCudaBuffer::allocate(0, bytes).unwrap());
                buffer.copy_from_host_sync(&stream, &vec![0xa5; bytes]).unwrap();
                buffers.push(buffer);
            }
        }
        let aliases = buffers.clone(); // retained inactive B8 bindings may share owners
        let allocated = cuda_device_allocation_snapshot();
        for repeat in 0..8 {
            let order = if repeat % 2 == 0 { [false, true] } else { [true, false] };
            for device_zero in order {
                let started = Instant::now();
                for buffer in &buffers {
                    if device_zero {
                        // SAFETY: all retained aliases are idle, one live stream.
                        unsafe { buffer.zero_sync_unchecked(&stream).unwrap(); }
                    } else {
                        buffer.copy_from_host_sync(&stream, &vec![0; buffer.len_bytes()]).unwrap();
                    }
                }
                println!("RESET_SCREEN repeat={repeat} device_zero={device_zero} wall_ms={:.6}",
                         started.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let measured = cuda_device_allocation_snapshot();
        assert_eq!(allocated.successful_allocations, measured.successful_allocations);
        assert_eq!(allocated.successful_frees, measured.successful_frees);
        // Re-poison before the candidate-only pass: not merely clearing old zeros.
        for buffer in &aliases {
            buffer.copy_from_host_sync(&stream, &vec![0x7f; buffer.len_bytes()]).unwrap();
            unsafe { buffer.zero_sync_unchecked(&stream).unwrap(); }
            let mut actual = vec![0xff; buffer.len_bytes()];
            buffer.copy_to_host_sync(&stream, &mut actual).unwrap();
            assert!(actual.iter().all(|&byte| byte == 0));
        }
        drop(aliases);
        drop(buffers);
        let after = cuda_device_allocation_snapshot();
        assert_eq!(before.live_bytes, after.live_bytes);
        assert_eq!(before.live_allocations, after.live_allocations);
    }

    #[test]
    fn device_allocation_counters_track_live_and_peak_bytes() {
        let counters = CudaDeviceAllocationCounters::new();
        counters.record_allocation(64);
        counters.record_allocation(128);
        assert_eq!(
            counters.snapshot(),
            CudaDeviceAllocationSnapshot {
                successful_allocations: 2,
                successful_frees: 0,
                live_allocations: 2,
                live_bytes: 192,
                peak_live_allocations: 2,
                peak_live_bytes: 192,
            }
        );

        counters.record_free(64);
        counters.record_free(128);
        assert_eq!(
            counters.snapshot(),
            CudaDeviceAllocationSnapshot {
                successful_allocations: 2,
                successful_frees: 2,
                live_allocations: 0,
                live_bytes: 0,
                peak_live_allocations: 2,
                peak_live_bytes: 192,
            }
        );
    }

    #[test]
    fn host_boundary_counters_preserve_wait_and_transfer_identity() {
        let before = cuda_host_boundary_snapshot();
        record_synchronous_h2d(64);
        record_synchronous_d2h(32);
        record_stream_synchronization();
        record_event_synchronization();
        let after = cuda_host_boundary_snapshot();

        assert_eq!(
            after.checked_delta_since(before),
            Some(CudaHostBoundarySnapshot {
                host_waits: 4,
                stream_synchronizations: 1,
                event_synchronizations: 1,
                synchronous_h2d_copies: 1,
                synchronous_h2d_bytes: 64,
                synchronous_d2h_copies: 1,
                synchronous_d2h_bytes: 32,
            })
        );
        assert_eq!(before.checked_delta_since(after), None);
    }

    #[test]
    fn profiler_api_status_is_mapped_without_cuda_work() {
        assert!(profiler_api_result("test profiler", 0).is_ok());
        let error = profiler_api_result("test profiler", 17).unwrap_err();
        assert_eq!(error.operation, "test profiler");
        assert_eq!(error.kind, CudaRuntimeErrorKind::Native);
        assert_eq!(error.status, AUGUR_CUDA_STATUS_RUNTIME_FAILURE);
        assert_eq!(error.raw_cuda_status, 17);
    }

    #[test]
    fn inactive_profiler_owner_rejects_duplicate_stop_without_cuda_work() {
        let mut capture = CudaProfilerCapture {
            profiler_active: false,
            nvtx_range: None,
        };
        let error = capture.stop().unwrap_err();
        assert_eq!(error.operation, "duplicate cudaProfilerStop");
        assert_eq!(error.kind, CudaRuntimeErrorKind::InvalidMetadata);
    }

    #[test]
    fn m306_nvtx_owner_rejects_nested_and_duplicate_end() {
        let mut range = CudaNvtxRange::begin_m306_steady_token().unwrap();
        let nested = CudaNvtxRange::begin_m306_steady_token().unwrap_err();
        assert_eq!(nested.kind, CudaRuntimeErrorKind::Native);
        assert_eq!(nested.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);

        range.end().unwrap();
        let duplicate = range.end().unwrap_err();
        assert_eq!(duplicate.kind, CudaRuntimeErrorKind::InvalidMetadata);
        assert_eq!(
            duplicate.operation,
            "duplicate nvtxRangePop(AUGUR_M306_STEADY_TOKEN_V1)"
        );
    }

    #[test]
    fn batch01_row_nvtx_owner_requires_outer_range_and_rejects_nesting() {
        let outside = CudaBatch01B4RowMainNvtxRange::begin(0).unwrap_err();
        assert_eq!(outside.kind, CudaRuntimeErrorKind::Native);
        assert_eq!(outside.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);

        let invalid = CudaBatch01B4RowMainNvtxRange::begin(4).unwrap_err();
        assert_eq!(invalid.kind, CudaRuntimeErrorKind::InvalidMetadata);

        let mut outer = CudaNvtxRange::begin_m306_steady_token().unwrap();
        let mut row = CudaBatch01B4RowMainNvtxRange::begin(2).unwrap();
        assert_eq!(row.range_name(), AUGUR_BATCH01_B4_ROW_MAIN_NVTX_RANGES_V1[2]);

        let nested = CudaBatch01B4RowMainNvtxRange::begin(1).unwrap_err();
        assert_eq!(nested.kind, CudaRuntimeErrorKind::Native);
        assert_eq!(nested.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
        let premature_outer_end = outer.end().unwrap_err();
        assert_eq!(premature_outer_end.kind, CudaRuntimeErrorKind::Native);
        assert_eq!(premature_outer_end.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);

        row.end().unwrap();
        let duplicate = row.end().unwrap_err();
        assert_eq!(duplicate.kind, CudaRuntimeErrorKind::InvalidMetadata);
        outer.end().unwrap();
    }

    #[test]
    fn profiler_owner_rejects_unapproved_nvtx_name_before_cuda_work() {
        let error = CudaProfilerCapture::start_with_nvtx_range("unexpected").unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::InvalidMetadata);
        assert_eq!(error.operation, "unsupported CUDA profiler NVTX range");
    }
}
