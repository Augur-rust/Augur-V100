//! Default-off rotated KV binding. Runtime/cache persistence integration pending.
use crate::{CudaBufferView, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};
use augur_cuda_sys::{AugurCudaQwen35RotatedKvArgsV1, AugurCudaStream};

const OP: &str = "qwen35_rotated_kv_v1";
pub const QWEN35_ROTATED_KV_IDENTITY: &str = "qwen35-q8-k-hadamard256-v-hadamard64-v1";
pub const QWEN35_ROTATED_KV_PLANAR_IDENTITY: &str = "qwen35-q8-k-hadamard256-v-hadamard64-head-code-scale-planar-v2";

fn geometry(capacity: usize, position: usize, rows: usize) -> Result<usize, CudaRuntimeError> {
    if capacity == 0
        || capacity > 262144
        || !capacity.is_multiple_of(32)
        || rows == 0
        || rows > 512
        || position >= capacity
        || rows > capacity - position
    {
        return Err(CudaRuntimeError::validation(
            OP,
            CudaRuntimeErrorKind::InvalidDimensions,
        ));
    }
    Ok(rows * (6144 + 2 * 1024) * 4)
}

fn small_geometry(
    capacity: usize,
    position: usize,
    rows: usize,
) -> Result<usize, CudaRuntimeError> {
    geometry(capacity, position, rows)?;
    // Extend only B1 at the full Augur capacity; existing B2..8 admission is unchanged.
    if (capacity > 196608 && !(capacity == 262144 && rows == 1))
        || !capacity.is_multiple_of(256) || rows > 8 {
        return Err(CudaRuntimeError::validation(
            OP,
            CudaRuntimeErrorKind::InvalidDimensions,
        ));
    }
    Ok(rows * (32768 + 2 * capacity + 6144 * 32 * 4 + 24 * 32 * 8))
}

const INPUT_META_BYTES: usize = 96 * 64 * 130 * 8;
const INPUT_PREFILL_META_BYTES: usize = 576 * 64 * 130 * 8;
fn input_meta_bytes(rows: usize) -> usize {
    match rows {
        128 => 144 * 64 * 130 * 8,
        224 => 252 * 64 * 130 * 8,
        256 => 288 * 64 * 130 * 8,
        480 => 540 * 64 * 130 * 8,
        512 => INPUT_PREFILL_META_BYTES,
        _ => INPUT_META_BYTES,
    }
}
fn gqa8_geometry(capacity: usize, position: usize, rows: usize, planar: bool) -> Result<usize, CudaRuntimeError> {
    // Format2 Augur alone extends B1..B8 to the physical 256Ki owner.
    // The legacy interleaved research ABI remains bounded to 196608.
    geometry(capacity, position, rows)?;
    if !capacity.is_multiple_of(256) || rows > 8
        || capacity > if planar { 262144 } else { 196608 } {
        return Err(CudaRuntimeError::validation(OP, CudaRuntimeErrorKind::InvalidDimensions));
    }
    Ok(rows * 32768 + INPUT_META_BYTES)
}
fn input_geometry(capacity: usize, position: usize, rows: usize) -> Result<usize, CudaRuntimeError> {
    geometry(capacity, position, rows)?;
    if !capacity.is_multiple_of(256) || rows < 9 {
        return Err(CudaRuntimeError::validation(OP, CudaRuntimeErrorKind::InvalidDimensions));
    }
    Ok(rows * 32768 + input_meta_bytes(rows))
}

/// Caller owns all buffers; retain them through the covering stream completion.
#[derive(Clone, Copy, Debug)]
pub struct Qwen35RotatedKvViews {
    pub query: CudaBufferView,
    pub gate: CudaBufferView,
    pub key: CudaBufferView,
    pub value: CudaBufferView,
    pub key_cache: CudaBufferView,
    pub value_cache: CudaBufferView,
    pub attention: CudaBufferView,
    pub gated: CudaBufferView,
    pub workspace: CudaBufferView,
}

pub struct CudaQwen35RotatedKvPlanV1 {
    ordinal: i32,
    owner: AugurCudaStream,
    small_dispatch: bool,
    input_dispatch: bool,
    gqa8_dispatch: bool,
    gqa8_c4: bool,
    planar: bool,
}

impl CudaQwen35RotatedKvPlanV1 {
    pub fn new(stream: &OwnedCudaStream) -> Result<Self, CudaRuntimeError> {
        let owner = stream.raw()?;
        // SAFETY: only configures a kernel for the live stream's ordinal.
        let result =
            unsafe { augur_cuda_sys::augur_cuda_qwen35_rotated_kv_configure_v1(stream.device_ordinal()) };
        CudaRuntimeError::from_result(OP, result)?;
        Ok(Self {
            ordinal: stream.device_ordinal(),
            owner,
            small_dispatch: false,
            input_dispatch: false,
            gqa8_dispatch: false, gqa8_c4: false,
            planar: false,
        })
    }

    pub fn workspace_bytes(rows: usize) -> Result<usize, CudaRuntimeError> {
        if rows == 0 || rows > 512 {
            return Err(CudaRuntimeError::validation(
                OP,
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        Ok(rows * (6144 + 2 * 1024) * 4)
    }

    /// Cache owners must explicitly carry QWEN35_ROTATED_KV_IDENTITY. Passing
    /// old cache bytes cannot be detected from addresses or sizes alone.
    pub fn submit(
        &self,
        stream: &OwnedCudaStream,
        views: Qwen35RotatedKvViews,
        capacity: usize,
        position: usize,
        rows: usize,
    ) -> Result<(), CudaRuntimeError> {
        self.submit_impl(stream, views, capacity, position, rows, 0)
    }

    fn submit_impl(
        &self, stream: &OwnedCudaStream, views: Qwen35RotatedKvViews,
        capacity: usize, position: usize, rows: usize, row_policy: u8,
    ) -> Result<(), CudaRuntimeError> {
        if row_policy > 2 || (row_policy != 0 && (!self.small_dispatch || self.input_dispatch)) {
            return Err(CudaRuntimeError::validation(OP, CudaRuntimeErrorKind::InvalidDimensions));
        }
        let workspace_bytes = if self.gqa8_dispatch {
            if row_policy != 0 {
                return Err(CudaRuntimeError::validation(OP, CudaRuntimeErrorKind::InvalidDimensions));
            }
            gqa8_geometry(capacity, position, rows, self.planar)?
        } else if self.input_dispatch {
            input_geometry(capacity, position, rows)?
        } else if self.small_dispatch {
            small_geometry(capacity, position, rows)?
        } else {
            geometry(capacity, position, rows)?
        };
        if views.workspace.len_bytes() != workspace_bytes {
            return Err(CudaRuntimeError::validation(
                OP,
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        if stream.device_ordinal() != self.ordinal || stream.raw()? != self.owner {
            return Err(CudaRuntimeError::validation(
                OP,
                CudaRuntimeErrorKind::DeviceMismatch,
            ));
        }
        let mut raw = AugurCudaQwen35RotatedKvArgsV1::new();
        macro_rules! bind {
            ($view:ident, $len:ident) => {{
                let view = views.$view;
                if view.device_ordinal() != self.ordinal {
                    return Err(CudaRuntimeError::validation(
                        OP,
                        CudaRuntimeErrorKind::DeviceMismatch,
                    ));
                }
                view.checked_range(OP)?;
                raw.$view = view.checked_address(OP)?;
                raw.$len = view.len_bytes() as u64;
            }};
        }
        bind!(query, query_bytes);
        bind!(gate, gate_bytes);
        bind!(key, key_bytes);
        bind!(value, value_bytes);
        bind!(key_cache, key_cache_bytes);
        bind!(value_cache, value_cache_bytes);
        bind!(attention, attention_bytes);
        bind!(gated, gated_bytes);
        bind!(workspace, workspace_bytes);
        let dim = |n| {
            u32::try_from(n).map_err(|_| {
                CudaRuntimeError::validation(OP, CudaRuntimeErrorKind::InvalidDimensions)
            })
        };
        raw.capacity = dim(capacity)?;
        raw.position = dim(position)?;
        raw.rows = dim(rows)?;
        if self.planar {
            if !self.gqa8_dispatch && !self.input_dispatch {
                return Err(CudaRuntimeError::validation(OP, CudaRuntimeErrorKind::InvalidDimensions));
            }
            raw.cache_format = augur_cuda_sys::AUGUR_QWEN35_ROTATED_KV_PLANAR_FORMAT_V2;
        }
        // SAFETY: all ranges are checked, native validates exact extents and
        // disjointness before launch. The caller holds buffers until drain.
        let result = unsafe {
            if self.planar && self.gqa8_dispatch && self.gqa8_c4 {
                augur_cuda_sys::augur_cuda_qwen35_rotated_kv_planar_gqa8_c4_submit_v1(self.ordinal, self.owner, &raw)
            } else if self.planar && self.gqa8_dispatch {
                augur_cuda_sys::augur_cuda_qwen35_rotated_kv_planar_gqa8_submit_v1(self.ordinal, self.owner, &raw)
            } else if self.planar && self.input_dispatch {
                augur_cuda_sys::augur_cuda_qwen35_rotated_kv_planar_input_submit_v1(self.ordinal, self.owner, &raw)
            } else if self.gqa8_dispatch {
                augur_cuda_sys::augur_cuda_qwen35_rotated_kv_gqa8_submit_v1(self.ordinal, self.owner, &raw)
            } else if self.input_dispatch {
                augur_cuda_sys::augur_cuda_qwen35_rotated_kv_input_submit_v1(self.ordinal, self.owner, &raw)
            } else if self.small_dispatch {
                if row_policy == 2 {
                    augur_cuda_sys::augur_cuda_qwen35_rotated_kv_small_tile8_submit_v1(self.ordinal, self.owner, &raw)
                } else if row_policy == 1 {
                    augur_cuda_sys::augur_cuda_qwen35_rotated_kv_small_vector_submit_v1(self.ordinal, self.owner, &raw)
                } else {
                augur_cuda_sys::augur_cuda_qwen35_rotated_kv_small_submit_v1(
                    self.ordinal,
                    self.owner,
                    &raw,
                )
                }
            } else {
                augur_cuda_sys::augur_cuda_qwen35_rotated_kv_submit_v1(self.ordinal, self.owner, &raw)
            }
        };
        CudaRuntimeError::from_result(OP, result)
    }
}

/// Scalar/verify plan for the rotated format1 cache, with physical capacity
/// padded to256. It is not a consumer of existing unrotated V3/V4 cache bytes.
/// Configure once per owner; retain all views through stream completion.
pub struct CudaQwen35RotatedKvSmallPlanV1(CudaQwen35RotatedKvPlanV1);

impl CudaQwen35RotatedKvSmallPlanV1 {
    /// Experimental grid-batched B1 arithmetic; ownership/extent checks are
    /// identical to submit. Does not change precision or allocate scratch.
    pub fn submit_vector_rows(
        &self, stream: &OwnedCudaStream, views: Qwen35RotatedKvViews,
        capacity: usize, position: usize, rows: usize,
    ) -> Result<(), CudaRuntimeError> {
        self.0.submit_impl(stream, views, capacity, position, rows, 1)
    }

    /// Separate opt-in tile8 contract across every row count, including B1.
    /// A caller must not mix it with the scalar vec reference within a session.
    pub fn submit_tile8_rows(
        &self, stream: &OwnedCudaStream, views: Qwen35RotatedKvViews,
        capacity: usize, position: usize, rows: usize,
    ) -> Result<(), CudaRuntimeError> {
        self.0.submit_impl(stream, views, capacity, position, rows, 2)
    }

    pub fn new(stream: &OwnedCudaStream) -> Result<Self, CudaRuntimeError> {
        let owner = stream.raw()?;
        // SAFETY: calibration only touches the live stream's device ordinal.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_rotated_kv_small_configure_v1(stream.device_ordinal())
        };
        CudaRuntimeError::from_result(OP, result)?;
        Ok(Self(CudaQwen35RotatedKvPlanV1 {
            ordinal: stream.device_ordinal(),
            owner,
            small_dispatch: true,
            input_dispatch: false,
            gqa8_dispatch: false, gqa8_c4: false,
            planar: false,
        }))
    }

    pub fn workspace_bytes(capacity: usize, rows: usize) -> Result<usize, CudaRuntimeError> {
        small_geometry(capacity, 0, rows)
    }

    pub fn submit(
        &self,
        stream: &OwnedCudaStream,
        views: Qwen35RotatedKvViews,
        capacity: usize,
        position: usize,
        rows: usize,
    ) -> Result<(), CudaRuntimeError> {
        self.0.submit(stream, views, capacity, position, rows)
    }
}

/// Owner-fixed research GQA8 MMA contract. All B1..8 calls use the same
/// physical-page cohorts and reduction order. Default small plan is unchanged.
pub struct CudaQwen35RotatedKvGqa8PlanV1(CudaQwen35RotatedKvPlanV1);
impl CudaQwen35RotatedKvGqa8PlanV1 {
    pub fn new(stream: &OwnedCudaStream) -> Result<Self, CudaRuntimeError> {
        Self::new_layout(stream, false, false)
    }
    pub fn new_planar(stream: &OwnedCudaStream) -> Result<Self, CudaRuntimeError> {
        Self::new_layout(stream, true, false)
    }
    /// C4 query rows with the same planar codec; separate arithmetic profile.
    pub fn new_planar_c4(stream: &OwnedCudaStream) -> Result<Self, CudaRuntimeError> {
        Self::new_layout(stream, true, true)
    }
    fn new_layout(stream: &OwnedCudaStream, planar: bool, c4: bool) -> Result<Self, CudaRuntimeError> {
        let owner = stream.raw()?;
        // SAFETY: calibrates kernels on the live owner's device; no submission.
        let result = unsafe {
            if c4 { augur_cuda_sys::augur_cuda_qwen35_rotated_kv_planar_gqa8_c4_configure_v1(stream.device_ordinal()) }
            else if planar { augur_cuda_sys::augur_cuda_qwen35_rotated_kv_planar_gqa8_configure_v1(stream.device_ordinal()) }
            else { augur_cuda_sys::augur_cuda_qwen35_rotated_kv_gqa8_configure_v1(stream.device_ordinal()) }
        };
        CudaRuntimeError::from_result(OP, result)?;
        Ok(Self(CudaQwen35RotatedKvPlanV1 {
            ordinal: stream.device_ordinal(), owner,
            small_dispatch: false, input_dispatch: false, gqa8_dispatch: true, gqa8_c4: c4,
            planar,
        }))
    }
    pub fn workspace_bytes(capacity: usize, rows: usize) -> Result<usize, CudaRuntimeError> {
        gqa8_geometry(capacity, 0, rows, false)
    }
    pub fn workspace_bytes_planar(capacity: usize, rows: usize) -> Result<usize, CudaRuntimeError> {
        gqa8_geometry(capacity, 0, rows, true)
    }
    pub fn submit(&self, stream: &OwnedCudaStream, views: Qwen35RotatedKvViews,
        capacity: usize, position: usize, rows: usize) -> Result<(), CudaRuntimeError> {
        self.0.submit(stream, views, capacity, position, rows)
    }
}

/// Real input-only MMA dispatch. Does not replace legacy prefix or B1..8 plans.
/// Capacity256-aligned; metadata belongs to the caller's owner-stream arena.
pub struct CudaQwen35RotatedKvInputPlanV1(CudaQwen35RotatedKvPlanV1);

impl CudaQwen35RotatedKvInputPlanV1 {
    pub fn new(stream: &OwnedCudaStream) -> Result<Self, CudaRuntimeError> {
        Self::new_layout(stream, false)
    }
    pub fn new_planar(stream: &OwnedCudaStream) -> Result<Self, CudaRuntimeError> {
        Self::new_layout(stream, true)
    }
    fn new_layout(stream: &OwnedCudaStream, planar: bool) -> Result<Self, CudaRuntimeError> {
        let owner = stream.raw()?;
        // SAFETY: calibrates original kernels for the live stream's ordinal.
        let result = unsafe {
            if planar { augur_cuda_sys::augur_cuda_qwen35_rotated_kv_planar_input_configure_v1(stream.device_ordinal()) }
            else { augur_cuda_sys::augur_cuda_qwen35_rotated_kv_input_configure_v1(stream.device_ordinal()) }
        };
        CudaRuntimeError::from_result(OP, result)?;
        Ok(Self(CudaQwen35RotatedKvPlanV1 {
            ordinal: stream.device_ordinal(), owner,
            small_dispatch: false, input_dispatch: true, gqa8_dispatch: false, gqa8_c4: false,
            planar,
        }))
    }

    pub fn workspace_bytes(capacity: usize, rows: usize) -> Result<usize, CudaRuntimeError> {
        input_geometry(capacity, 0, rows)
    }

    pub fn submit(&self, stream: &OwnedCudaStream, views: Qwen35RotatedKvViews,
        capacity: usize, position: usize, rows: usize) -> Result<(), CudaRuntimeError> {
        self.0.submit(stream, views, capacity, position, rows)
    }
}

#[cfg(test)]
mod planar_augur_tests;
#[cfg(test)]
mod planar_c4_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gqa8_exact_extent_and_bounded_rows() {
        for capacity in [256, 8192, 196608] {
            for rows in 1..=8 {
                assert_eq!(gqa8_geometry(capacity, capacity-rows, rows, false).unwrap(),
                    rows*32768 + INPUT_META_BYTES);
                assert!(gqa8_geometry(capacity, capacity-rows+1, rows, false).is_err());
            }
        }
        for (capacity, rows) in [(0,1),(32,1),(196864,1),(256,0),(256,9)] {
            assert!(gqa8_geometry(capacity,0,rows,false).is_err());
        }
    }

    #[test]
    fn augur_input_preserves_full_256ki_capacity_and_real_tail_bounds() {
        for rows in [1, 7, 8, 9, 44, 512] {
            assert!(geometry(262144, 262144 - rows, rows).is_ok());
            assert!(geometry(262144, 262145 - rows, rows).is_err());
            if rows >= 9 {
                assert_eq!(input_geometry(262144, 262144 - rows, rows).unwrap(), rows*32768 + input_meta_bytes(rows));
            }
        }
        assert!(geometry(262176, 0, 1).is_err());
        assert!(input_geometry(262144, usize::MAX, 44).is_err());
        // The old small and interleaved GQA8 ABIs remain bounded; planar GQA8 is separate.
        assert!(small_geometry(262144, 262143, 1).is_ok());
        assert!(small_geometry(262144, 262144, 1).is_err());
        assert!(gqa8_geometry(262144, 0, 1, false).is_err());
        for rows in 1..=8 {
            assert_eq!(gqa8_geometry(262144, 262144-rows, rows, true).unwrap(),
                rows*32768 + INPUT_META_BYTES);
            assert!(gqa8_geometry(262144, 262145-rows, rows, true).is_err());
        }
        assert!(gqa8_geometry(262400, 0, 1, true).is_err());
        assert!(gqa8_geometry(262144, 0, 9, true).is_err());
        assert!(small_geometry(262144, 0, 8).is_err());
        assert!(gqa8_geometry(262144, 0, 8, false).is_err());
    }

    #[test]
    fn exact_input_metadata_extent_and_boundaries() {
        // Split3 covers measured long tails; the maximum remains the 512-row bound.
        assert_eq!(input_geometry(262144, 261760, 128).unwrap(), 13_778_944);
        assert_eq!(input_geometry(262144, 261632, 256).unwrap(), 27_557_888);
        assert_eq!(input_geometry(262144, 261920, 224).unwrap(), 24_113_152);
        assert_eq!(input_geometry(262144, 261664, 480).unwrap(), 51_671_040);
        assert_eq!(input_geometry(262144, 32248, 512).unwrap(), 55_115_776);
        assert_eq!(input_geometry(262144, 32249, 511).unwrap(), 23_134_208);
        for rows in [9, 15, 16, 17, 31, 32, 33, 97, 127, 128, 129, 193, 223, 224, 225, 255, 256, 257, 285, 479, 480, 481, 511, 512] {
            for position in [0, 5632, 196608 - rows] {
                assert_eq!(input_geometry(196608, position, rows).unwrap(), rows * 32768 + input_meta_bytes(rows));
            }
            assert_eq!(CudaQwen35RotatedKvInputPlanV1::workspace_bytes(196608, rows).unwrap(),
                rows * 32768 + input_meta_bytes(rows));
        }
        for rows in [0, 1, 8, 513, usize::MAX] {
            assert!(input_geometry(196608, 0, rows).is_err());
        }
        for capacity in [0, 32, 255, 196609, usize::MAX] {
            assert!(input_geometry(capacity, 0, 9).is_err());
        }
        assert!(input_geometry(256, 248, 9).is_err());
        assert!(input_geometry(256, usize::MAX, 9).is_err());
        // The original ABI still admits32-padded capacity and no metadata.
        assert_eq!(geometry(32, 0, 16).unwrap(), 16 * 32768);
    }

    #[test]
    fn rotated_geometry_accepts_scalar_verify_prefix_and_capacity_edge() {
        for rows in [1, 7, 8, 31, 32, 33, 127, 511, 512] {
            for position in [0, 5632, 196608 - rows] {
                assert_eq!(geometry(196608, position, rows).unwrap(), rows * 32768);
            }
        }
        assert_eq!(
            CudaQwen35RotatedKvPlanV1::workspace_bytes(512).unwrap(),
            16777216
        );
    }

    #[test]
    fn rotated_geometry_rejects_unpadded_overflow_and_empty() {
        for args in [
            (0, 0, 1),
            (31, 0, 1),
            (196609, 0, 1),
            (196608, 0, 0),
            (196608, 0, 513),
            (196608, 196608, 1),
            (196608, 196607, 2),
            (196608, usize::MAX, 1),
            (usize::MAX, 0, 1),
            (196608, 0, usize::MAX),
        ] {
            assert!(geometry(args.0, args.1, args.2).is_err());
        }
    }

    #[test]
    fn rotated_native_abi_layout_and_identity_are_explicit() {
        let args = AugurCudaQwen35RotatedKvArgsV1::new();
        assert_eq!(args.struct_size, 168);
        assert_eq!(args.cache_format, 1);
        assert_eq!(
            core::mem::offset_of!(AugurCudaQwen35RotatedKvArgsV1, capacity),
            152
        );
        assert!(QWEN35_ROTATED_KV_IDENTITY.contains("hadamard256"));
        assert!(QWEN35_ROTATED_KV_IDENTITY.contains("hadamard64"));
        // SAFETY: host structural validation only; no CUDA calls/allocations.
        assert!(!unsafe { augur_cuda_sys::augur_cuda_qwen35_rotated_kv_validate_v1(&args) }.is_ok());
    }

    #[test]
    fn augur_b1_native_and_rust_geometry_agree() {
        for (capacity, position, rows, valid) in [
            (262144, 0, 1, true), (262144, 8477, 1, true),
            (262144, 262143, 1, true), (262144, 262144, 1, false),
            (262144, 0, 2, false), (196864, 0, 1, false),
            (262400, 0, 1, false), (262144, usize::MAX, 1, false),
        ] {
            assert_eq!(small_geometry(capacity, position, rows).is_ok(), valid);
            let mut a = AugurCudaQwen35RotatedKvArgsV1::new();
            a.capacity = capacity as u32; a.position = position as u32; a.rows = rows as u32;
            let q = (rows*6144*4) as u64; let k = (rows*1024*4) as u64;
            a.query=0x100000000; a.query_bytes=q;
            a.gate=0x200000000; a.gate_bytes=q;
            a.key=0x300000000; a.key_bytes=k;
            a.value=0x400000000; a.value_bytes=k;
            a.key_cache=0x500000000; a.key_cache_bytes=(capacity*1088) as u64;
            a.value_cache=0x600000000; a.value_cache_bytes=a.key_cache_bytes;
            a.attention=0x700000000; a.attention_bytes=q;
            a.gated=0x800000000; a.gated_bytes=q;
            a.workspace=0x900000000;
            a.workspace_bytes=(rows*(32768+2*capacity+6144*32*4+24*32*8)) as u64;
            // SAFETY: validation inspects numeric ranges only, never dereferences addresses.
            assert_eq!(unsafe { augur_cuda_sys::augur_cuda_qwen35_rotated_kv_small_validate_v1(&a) }.is_ok(), valid);
        }
    }

    #[test]
    fn rotated_small_geometry_covers_all_verify_rows_and_capacity_edges() {
        for capacity in [256, 512, 6144, 196608] {
            for rows in 1..=8 {
                let bytes =
                    CudaQwen35RotatedKvSmallPlanV1::workspace_bytes(capacity, rows).unwrap();
                assert_eq!(
                    small_geometry(capacity, capacity - rows, rows).unwrap(),
                    bytes
                );
                assert_eq!(
                    bytes,
                    rows * (32768 + 2 * capacity + 6144 * 32 * 4 + 24 * 32 * 8)
                );
                assert!(
                    bytes <= 16777216,
                    "small scratch fits the B512 rotation arena"
                );
            }
        }
    }

    #[test]
    fn rotated_small_geometry_rejects_old_workspace_and_invalid_extent() {
        for (capacity, position, rows) in [
            (32, 0, 1),
            (196608, 0, 9),
            (256, 256, 1),
            (256, 255, 2),
            (0, 0, 1),
            (256, 0, 0),
        ] {
            assert!(small_geometry(capacity, position, rows).is_err());
        }
        let mut args = AugurCudaQwen35RotatedKvArgsV1::new();
        args.capacity = 196608;
        args.rows = 8;
        args.workspace_bytes = geometry(196608, 0, 8).unwrap() as u64;
        // SAFETY: pure host validation rejects empty addresses/old workspace.
        assert!(
            !unsafe { augur_cuda_sys::augur_cuda_qwen35_rotated_kv_small_validate_v1(&args) }.is_ok()
        );
    }
}

#[cfg(test)]
mod augur_b1_gpu_tests {
    use super::*;
    use crate::{cuda_device_allocation_snapshot, OwnedCudaBuffer};

    #[test]
    #[ignore = "SM70 GPU: bounded synthetic full-capacity B1 dispatch comparison"]
    fn augur_vector_b1_exceeds_flash32_equivalence_bound() {
        compare_flash32(false);
    }
    #[test]
    #[ignore = "SM70 GPU: bounded synthetic full-capacity tile8 B1 dispatch comparison"]
    fn augur_tile8_b1_full_capacity_matches_flash32_bounded() {
        compare_flash32(true);
    }
    fn compare_flash32(tile8: bool) {
        let before = cuda_device_allocation_snapshot();
        {
            let stream = OwnedCudaStream::create(0).unwrap();
            let upload = |n: usize, salt: usize| {
                let bytes: Vec<u8> = (0..n).flat_map(|i| {
                    (((i*37+salt)%127) as f32 / 127.0 - 0.5).to_le_bytes()
                }).collect();
                let mut b = OwnedCudaBuffer::allocate(0, bytes.len()).unwrap();
                b.copy_from_host_sync(&stream, &bytes).unwrap(); b
            };
            let q=upload(6144,3); let gate=upload(6144,11);
            let k=upload(1024,17); let v=upload(1024,23);
            let mut kc=OwnedCudaBuffer::allocate(0,262144*1088).unwrap();
            let mut vc=OwnedCudaBuffer::allocate(0,262144*1088).unwrap();
            kc.zero_sync(&stream).unwrap(); vc.zero_sync(&stream).unwrap();
            let attention=OwnedCudaBuffer::allocate(0,6144*4).unwrap();
            let gated=OwnedCudaBuffer::allocate(0,6144*4).unwrap();
            let standard=CudaQwen35RotatedKvPlanV1::new(&stream).unwrap();
            let small=CudaQwen35RotatedKvSmallPlanV1::new(&stream).unwrap();
            let sa=OwnedCudaBuffer::allocate(0,CudaQwen35RotatedKvPlanV1::workspace_bytes(1).unwrap()).unwrap();
            let sb=OwnedCudaBuffer::allocate(0,CudaQwen35RotatedKvSmallPlanV1::workspace_bytes(262144,1).unwrap()).unwrap();
            let view=|b: &OwnedCudaBuffer| b.view(0,b.len_bytes()).unwrap();
            let views=|scratch: &OwnedCudaBuffer| Qwen35RotatedKvViews {
                query:view(&q),gate:view(&gate),key:view(&k),value:view(&v),
                key_cache:view(&kc),value_cache:view(&vc),attention:view(&attention),
                gated:view(&gated),workspace:view(scratch),
            };
            let read=|b: &OwnedCudaBuffer| {
                let mut bytes=vec![0;b.len_bytes()];b.copy_to_host_sync(&stream,&mut bytes).unwrap();
                bytes.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect::<Vec<_>>()
            };
            // Sparse deterministic initialized cache, not a real long-history qualification.
            // Vector adds query-Q8 quantization and is a negative equivalence control.
            let mut equivalence_failed = false;
            for position in [0,1,31,32,254,255,256,8476,8477,262143] {
                standard.submit(&stream,views(&sa),262144,position,1).unwrap();
                let a=read(&gated);
                let mut old_row=vec![0;1088];kc.view(position*1088,1088).unwrap().copy_to_host_sync(&stream,&mut old_row).unwrap();
                let mut old_v=vec![0;1088];vc.view(position*1088,1088).unwrap().copy_to_host_sync(&stream,&mut old_v).unwrap();
                if tile8 { small.submit_tile8_rows(&stream,views(&sb),262144,position,1).unwrap(); }
                else { small.submit(&stream,views(&sb),262144,position,1).unwrap(); }
                let b=read(&gated);
                let mut new_row=vec![0;1088];kc.view(position*1088,1088).unwrap().copy_to_host_sync(&stream,&mut new_row).unwrap();
                let mut new_v=vec![0;1088];vc.view(position*1088,1088).unwrap().copy_to_host_sync(&stream,&mut new_v).unwrap();
                assert_eq!(old_row,new_row);assert_eq!(old_v,new_v);
                let mut max=0f32;
                for (x,y) in a.iter().zip(&b) {
                    assert!(x.is_finite()&&y.is_finite());max=max.max((x-y).abs());
                    equivalence_failed |= (x-y).abs() > 0.0001+0.001*x.abs();
                }
                eprintln!("AUGUR_B1 tile8={tile8} position={position} max_abs={max} cache_row_equal=true");
            }
            assert_eq!(equivalence_failed, !tile8, "tile8 must meet the unchanged bound; vector is intentionally rejected");
            stream.synchronize().unwrap();
        }
        let after=cuda_device_allocation_snapshot();
        assert_eq!(before.live_bytes,after.live_bytes);
        assert_eq!(before.live_allocations,after.live_allocations);
    }

    #[test]
    #[ignore = "SM70 GPU: compact B1 versus unchanged tile8 multirow, exact causal page boundaries"]
    fn augur_compact_b1_matches_tile8_multirow_bytes() {
        let before = cuda_device_allocation_snapshot();
        {
            let stream = OwnedCudaStream::create(0).unwrap();
            let plan = CudaQwen35RotatedKvSmallPlanV1::new(&stream).unwrap();
            let capacity = 131072usize;
            let upload = |n: usize, salt: usize| {
                let bytes: Vec<u8> = (0..n).flat_map(|i| {
                    (((i * 37 + salt) % 509) as f32 / 177.0 - 1.4).to_le_bytes()
                }).collect();
                let mut b = OwnedCudaBuffer::allocate(0, bytes.len()).unwrap();
                b.copy_from_host_sync(&stream, &bytes).unwrap(); b
            };
            let q = upload(8 * 6144, 3); let gate = upload(8 * 6144, 11);
            let k = upload(8 * 1024, 17); let v = upload(8 * 1024, 23);
            let mut kc = OwnedCudaBuffer::allocate(0, capacity * 1088).unwrap();
            let mut vc = OwnedCudaBuffer::allocate(0, capacity * 1088).unwrap();
            // Dense finite Q8 history, not a trained-model long-history proof.
            for (b, salt) in [(&mut kc, 5usize), (&mut vc, 13usize)] {
                let mut bytes = vec![0u8; capacity * 1088];
                for (i, block) in bytes.chunks_exact_mut(34).enumerate() {
                    block[..2].copy_from_slice(&0x2000u16.to_le_bytes()); // exact 1/128 scale
                    for j in 0..32 { block[j + 2] = (((i * 97 + j * 37 + salt) % 255) as i16 - 127) as i8 as u8; }
                }
                b.copy_from_host_sync(&stream, &bytes).unwrap();
            }
            let attention = OwnedCudaBuffer::allocate(0, 8 * 6144 * 4).unwrap();
            let gated = OwnedCudaBuffer::allocate(0, 8 * 6144 * 4).unwrap();
            let one_scratch = OwnedCudaBuffer::allocate(0, CudaQwen35RotatedKvSmallPlanV1::workspace_bytes(capacity, 1).unwrap()).unwrap();
            let batch_scratch = OwnedCudaBuffer::allocate(0, CudaQwen35RotatedKvSmallPlanV1::workspace_bytes(capacity, 8).unwrap()).unwrap();
            let views = |row: usize, rows: usize| Qwen35RotatedKvViews {
                query: q.view(row * 6144 * 4, rows * 6144 * 4).unwrap(),
                gate: gate.view(row * 6144 * 4, rows * 6144 * 4).unwrap(),
                key: k.view(row * 1024 * 4, rows * 1024 * 4).unwrap(),
                value: v.view(row * 1024 * 4, rows * 1024 * 4).unwrap(),
                key_cache: kc.view(0, kc.len_bytes()).unwrap(), value_cache: vc.view(0, vc.len_bytes()).unwrap(),
                attention: attention.view(row * 6144 * 4, rows * 6144 * 4).unwrap(),
                gated: gated.view(row * 6144 * 4, rows * 6144 * 4).unwrap(),
                workspace: if rows == 1 { one_scratch.view(0, one_scratch.len_bytes()).unwrap() }
                    else { batch_scratch.view(0, CudaQwen35RotatedKvSmallPlanV1::workspace_bytes(capacity, rows).unwrap()).unwrap() },
            };
            let read = |b: &OwnedCudaBuffer, bytes: usize| {
                let mut raw = vec![0; bytes]; b.view(0, bytes).unwrap().copy_to_host_sync(&stream, &mut raw).unwrap(); raw
            };
            for position in [0, 254, 255, 256, 511, 7934, 7935, 7936, 8190, 8191, 8240, 8447, 32760, 32767, 32768, 32894, 33023, 33024, 36856, 36863, 65528, 65535, 65536] {
                for rows in [2, 3, 4, 8] {
                    let allocated = cuda_device_allocation_snapshot();
                    plan.submit_tile8_rows(&stream, views(0, rows), capacity, position, rows).unwrap();
                    let expected = read(&attention, rows * 6144 * 4);
                    let expected_gated = read(&gated, rows * 6144 * 4);
                    for row in 0..rows { plan.submit_tile8_rows(&stream, views(row, 1), capacity, position + row, 1).unwrap(); }
                    assert_eq!(read(&attention, rows * 6144 * 4), expected, "attention position={position} rows={rows}");
                    assert_eq!(read(&gated, rows * 6144 * 4), expected_gated, "gate position={position} rows={rows}");
                    let after = cuda_device_allocation_snapshot();
                    assert_eq!(after.successful_allocations, allocated.successful_allocations);
                    assert_eq!(after.successful_frees, allocated.successful_frees);
                }
            }
            stream.synchronize().unwrap();
        }
        let after = cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
    }
}
