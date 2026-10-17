//! Owned QPN storage and stream-local FP16 activation staging.
//!
//! This is an opt-in numerical profile, not a transparent replacement for
//! Q8-activation DP4A. Pack once at startup; share one staged input across
//! gate/up. Bindings borrow every allocation and their original stream.
use crate::{
    CudaBufferView, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaBuffer, OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;
use augur_cuda_sys::*;
use std::rc::Rc;

const OP: &str = "cuda_qwen35_qpn_v1";
fn error(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OP, kind)
}
fn geometry(rows: usize, cols: usize) -> Result<usize, CudaRuntimeError> {
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))?;
    if rows == 0
        || cols == 0
        || !rows.is_multiple_of(32)
        || !cols.is_multiple_of(64)
        || rows > i32::MAX as usize
        || cols > i32::MAX as usize
        || elements > u32::MAX as usize
    {
        return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
    }
    Ok(elements / 64 * 36)
}
fn scalar(scale: f32) -> Result<(), CudaRuntimeError> {
    // Keep all finite E4M3/E2M1 words finite through the QPN FP16 boundaries.
    if !scale.is_finite() || !(2.0_f32.powi(-32)..=24.0).contains(&scale) {
        return Err(error(CudaRuntimeErrorKind::InvalidMetadata));
    }
    Ok(())
}
fn checked_ggml(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    scale: f32,
) -> Result<usize, CudaRuntimeError> {
    let len = geometry(rows, cols)?;
    scalar(scale)?;
    if bytes.len() != len {
        return Err(error(CudaRuntimeErrorKind::OutOfBounds));
    }
    if bytes
        .chunks_exact(36)
        .any(|b| b[..4].iter().any(|s| *s >= 127))
    {
        return Err(error(CudaRuntimeErrorKind::InvalidMetadata));
    }
    Ok(len)
}
fn checked_checkpoint(
    packed: &[u8], scales: &[u8], rows: usize, cols: usize, scale: f32,
) -> Result<usize, CudaRuntimeError> {
    let len = geometry(rows, cols)?;
    scalar(scale)?;
    if packed.len() != rows * cols / 2 || scales.len() != rows * cols / 16 {
        return Err(error(CudaRuntimeErrorKind::OutOfBounds));
    }
    if scales.iter().any(|&byte| byte >= 127) {
        return Err(error(CudaRuntimeErrorKind::InvalidMetadata));
    }
    Ok(len)
}
fn view(
    owner: &OwnedCudaBuffer,
    stream: &OwnedCudaStream,
    offset: usize,
    bytes: usize,
) -> Result<CudaBufferView, CudaRuntimeError> {
    let v = owner.view(offset, bytes)?;
    if v.device_ordinal() != stream.device_ordinal() {
        return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
    }
    v.checked_range(OP)?;
    Ok(v)
}

/// One permanent physical allocation; no retained GGML or expanded-FP16 copy.
#[derive(Debug)]
pub struct OwnedCudaQwen35QpnWeightV1 {
    storage: Rc<OwnedCudaBuffer>,
    rows: usize,
    cols: usize,
    tensor_scale: f32,
}
impl OwnedCudaQwen35QpnWeightV1 {
    /// Upload original checkpoint planes and perform the lossless QPN permutation
    /// on the device. No GGML host intermediate or numerical conversion.
    pub fn upload_checkpoint_nvfp4(
        stream: &OwnedCudaStream,
        packed: &[u8],
        scales: &[u8],
        rows: usize,
        cols: usize,
        tensor_scale: f32,
    ) -> Result<Self, CudaRuntimeError> {
        let len = checked_checkpoint(packed, scales, rows, cols, tensor_scale)?;
        let source = OwnedCudaBuffer::allocate(stream.device_ordinal(), len)?;
        let storage = Rc::new(OwnedCudaBuffer::allocate(stream.device_ordinal(), len)?);
        source.copy_range_from_host_sync(stream, 0, packed)?;
        source.copy_range_from_host_sync(stream, packed.len(), scales)?;
        let mut raw = AugurCudaQwen35QpnPackArgsV1::new();
        raw.source = view(&source, stream, 0, len)?.checked_address(OP)?;
        raw.destination = view(&storage, stream, 0, len)?.checked_address(OP)?;
        raw.source_bytes = len as u64;
        raw.destination_bytes = len as u64;
        raw.rows = rows as u32;
        raw.cols = cols as u32;
        // SAFETY: exact disjoint retained allocations and the original stream;
        // synchronous transfers finish before pack, and pack drains before drop.
        unsafe {
            CudaRuntimeError::from_result(OP, augur_cuda_qwen35_qpn_pack_checkpoint_preflight_v1(
                stream.device_ordinal(), stream.raw()?, &raw,
            ))?;
            CudaRuntimeError::from_result(OP, augur_cuda_qwen35_qpn_pack_checkpoint_prevalidated_submit_v1(
                stream.device_ordinal(), stream.raw()?, &raw,
            ))?;
        }
        stream.synchronize()?;
        Ok(Self { storage, rows, cols, tensor_scale })
    }
    /// Synchronous startup adapter. Validates scale words before allocation.
    /// Temporary GGML storage is released after stream completion. Async/pinned
    /// startup integration can later batch this without changing the format.
    pub fn upload_ggml(
        stream: &OwnedCudaStream,
        bytes: &[u8],
        rows: usize,
        cols: usize,
        tensor_scale: f32,
    ) -> Result<Self, CudaRuntimeError> {
        let len = checked_ggml(bytes, rows, cols, tensor_scale)?;
        let source = OwnedCudaBuffer::allocate(stream.device_ordinal(), len)?;
        let storage = Rc::new(OwnedCudaBuffer::allocate(stream.device_ordinal(), len)?);
        source.copy_from_host_sync(stream, bytes)?;
        let mut raw = AugurCudaQwen35QpnPackArgsV1::new();
        raw.source = view(&source, stream, 0, len)?.checked_address(OP)?;
        raw.destination = view(&storage, stream, 0, len)?.checked_address(OP)?;
        raw.source_bytes = len as u64;
        raw.destination_bytes = len as u64;
        raw.rows = rows as u32;
        raw.cols = cols as u32;
        // SAFETY: exact disjoint allocations, live stream and validated GGML
        // metadata. Both owners remain alive until the explicit completion.
        unsafe {
            CudaRuntimeError::from_result(
                OP,
                augur_cuda_qwen35_qpn_pack_preflight_v1(stream.device_ordinal(), stream.raw()?, &raw),
            )?;
            CudaRuntimeError::from_result(
                OP,
                augur_cuda_qwen35_qpn_pack_prevalidated_submit_v1(
                    stream.device_ordinal(),
                    stream.raw()?,
                    &raw,
                ),
            )?;
        }
        stream.synchronize()?;
        Ok(Self {
            storage,
            rows,
            cols,
            tensor_scale,
        })
    }
    pub const fn layout(&self) -> PhysicalLayoutId {
        PhysicalLayoutId::DEVICE_QWEN35_NVFP4_QPN_PLANES_V1
    }
    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn cols(&self) -> usize {
        self.cols
    }
    pub const fn tensor_scale(&self) -> f32 {
        self.tensor_scale
    }
    pub fn len_bytes(&self) -> usize {
        self.storage.len_bytes()
    }
    pub fn physical_view(&self) -> Result<CudaBufferView, CudaRuntimeError> {
        self.storage.view(0, self.len_bytes())
    }
    /// Share the single physical allocation with an outer immutable model
    /// owner. Its layout remains QPN118; this is never a legacy GGML view.
    pub fn shared_storage(&self) -> Rc<OwnedCudaBuffer> {
        Rc::clone(&self.storage)
    }

    /// Explicit expansion contract for wide prefill. Folded weights use GEMM
    /// alpha=1; LegacyUnscaled requires the separate tensor scalar in GEMM.
    pub fn expand_f16_sync(
        &self,
        stream: &OwnedCudaStream,
        destination: &OwnedCudaBuffer,
        profile: Qwen35QpnExpansionV1,
    ) -> Result<(), CudaRuntimeError> {
        let src = view(&self.storage, stream, 0, self.len_bytes())?;
        let dst = view(destination, stream, 0, self.rows * self.cols * 2)?;
        let mut raw = AugurCudaQwen35QpnExpandArgsV1::new();
        raw.source = src.checked_address(OP)?;
        raw.source_bytes = src.len_bytes() as u64;
        raw.destination = dst.checked_address(OP)?;
        raw.destination_bytes = dst.len_bytes() as u64;
        raw.rows = self.rows as u32;
        raw.cols = self.cols as u32;
        raw.tensor_scale = self.tensor_scale;
        raw.numerical_profile = profile as u32;
        // SAFETY: both owners and exact stream remain retained through sync.
        unsafe {
            CudaRuntimeError::from_result(
                OP,
                augur_cuda_qwen35_qpn_expand_preflight_v1(
                    stream.device_ordinal(),
                    stream.raw()?,
                    &raw,
                ),
            )?;
            CudaRuntimeError::from_result(
                OP,
                augur_cuda_qwen35_qpn_expand_prevalidated_submit_v1(
                    stream.device_ordinal(),
                    stream.raw()?,
                    &raw,
                ),
            )?;
        }
        stream.synchronize()
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Qwen35QpnExpansionV1 {
    LegacyUnscaled = 0,
    FoldedF16 = 1,
}

/// One fixed-capacity activation allocation belonging to exactly one stream.
/// Stage/consume ordering is the plan's responsibility; no global scratch.
pub struct CudaQwen35QpnWorkspaceV1<'s> {
    stream: &'s OwnedCudaStream,
    scratch: OwnedCudaBuffer,
    cols: usize,
    capacity: usize,
}
impl<'s> CudaQwen35QpnWorkspaceV1<'s> {
    pub fn new(
        stream: &'s OwnedCudaStream,
        cols: usize,
        capacity: usize,
    ) -> Result<Self, CudaRuntimeError> {
        geometry(32, cols)?;
        if !(1..=8).contains(&capacity) {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let scratch = OwnedCudaBuffer::allocate(stream.device_ordinal(), capacity * cols * 2)?;
        Ok(Self {
            stream,
            scratch,
            cols,
            capacity,
        })
    }
    pub fn capacity_bytes(&self) -> usize {
        self.scratch.len_bytes()
    }
    fn input_view(&self, tokens: usize) -> Result<CudaBufferView, CudaRuntimeError> {
        if tokens == 0 || tokens > self.capacity {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        view(&self.scratch, self.stream, 0, tokens * self.cols * 2)
    }
    pub fn bind_stage<'a>(
        &'a self,
        input: &'a OwnedCudaBuffer,
        offset: usize,
        tokens: usize,
    ) -> Result<CudaQwen35QpnStageV1<'a, 's>, CudaRuntimeError> {
        let dst = self.input_view(tokens)?;
        let src = view(input, self.stream, offset, tokens * self.cols * 4)?;
        let mut raw = AugurCudaQwen35QpnStageArgsV1::new();
        raw.source = src.checked_address(OP)?;
        raw.source_bytes = src.len_bytes() as u64;
        raw.destination = dst.checked_address(OP)?;
        raw.destination_bytes = dst.len_bytes() as u64;
        raw.elements = (tokens * self.cols) as u64;
        // SAFETY: owners retained by the returned binding; native preflight
        // independently validates alignment, extent and actual device ranges.
        unsafe {
            CudaRuntimeError::from_result(
                OP,
                augur_cuda_qwen35_qpn_stage_preflight_v1(
                    self.stream.device_ordinal(),
                    self.stream.raw()?,
                    &raw,
                ),
            )?;
        }
        Ok(CudaQwen35QpnStageV1 {
            workspace: self,
            _input: input,
            raw,
        })
    }
    /// Bind ahead of execution; submit after staging these input rows. Every
    /// width uses the same shape-selected K reduction, including rejection replay.
    pub fn bind_projection<'a>(
        &'a self,
        weight: &'a OwnedCudaQwen35QpnWeightV1,
        output: &'a OwnedCudaBuffer,
        offset: usize,
        tokens: usize,
    ) -> Result<CudaQwen35QpnProjectionV1<'a, 's>, CudaRuntimeError> {
        if weight.cols != self.cols {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let input = self.input_view(tokens)?;
        let w = view(&weight.storage, self.stream, 0, weight.len_bytes())?;
        let out = view(output, self.stream, offset, tokens * weight.rows * 4)?;
        let mut raw = AugurCudaQwen35QpnProjectArgsV1::new();
        raw.weight = w.checked_address(OP)?;
        raw.weight_bytes = w.len_bytes() as u64;
        raw.input = input.checked_address(OP)?;
        raw.input_bytes = input.len_bytes() as u64;
        raw.output = out.checked_address(OP)?;
        raw.output_bytes = out.len_bytes() as u64;
        raw.rows = weight.rows as u32;
        raw.cols = weight.cols as u32;
        raw.tokens = tokens as u32;
        raw.split_k = if weight.rows == 17408 && weight.cols == 5120 {
            16
        } else {
            8
        };
        raw.tensor_scale = weight.tensor_scale;
        raw.numerical_profile = 1;
        // SAFETY: the returned binding retains all allocation/stream owners.
        unsafe {
            CudaRuntimeError::from_result(
                OP,
                augur_cuda_qwen35_qpn_project_preflight_v1(
                    self.stream.device_ordinal(),
                    self.stream.raw()?,
                    &raw,
                ),
            )?;
        }
        Ok(CudaQwen35QpnProjectionV1 {
            workspace: self,
            _weight: weight,
            _output: output,
            raw,
        })
    }
}
pub struct CudaQwen35QpnStageV1<'a, 's> {
    workspace: &'a CudaQwen35QpnWorkspaceV1<'s>,
    _input: &'a OwnedCudaBuffer,
    raw: AugurCudaQwen35QpnStageArgsV1,
}
impl CudaQwen35QpnStageV1<'_, '_> {
    pub fn submit_for_plan(&self) -> Result<(), CudaRuntimeError> {
        let s = self.workspace.stream;
        // SAFETY: preflight checked these immutable arguments and all owners
        // are borrowed. There is no API to substitute a different stream.
        unsafe {
            CudaRuntimeError::from_result(
                OP,
                augur_cuda_qwen35_qpn_stage_prevalidated_submit_v1(
                    s.device_ordinal(),
                    s.raw()?,
                    &self.raw,
                ),
            )
        }
    }
}
pub struct CudaQwen35QpnProjectionV1<'a, 's> {
    workspace: &'a CudaQwen35QpnWorkspaceV1<'s>,
    _weight: &'a OwnedCudaQwen35QpnWeightV1,
    _output: &'a OwnedCudaBuffer,
    raw: AugurCudaQwen35QpnProjectArgsV1,
}
impl CudaQwen35QpnProjectionV1<'_, '_> {
    pub fn submit_for_plan(&self) -> Result<(), CudaRuntimeError> {
        let s = self.workspace.stream;
        // SAFETY: same immutable binding and borrowed owners as preflight.
        unsafe {
            CudaRuntimeError::from_result(
                OP,
                augur_cuda_qwen35_qpn_project_prevalidated_submit_v1(
                    s.device_ordinal(),
                    s.raw()?,
                    &self.raw,
                ),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_shape_extent_and_scale_words_are_checked_before_upload() {
        let packed = vec![0xff; 32 * 64 / 2];
        let mut scales: Vec<u8> = (0..32 * 64 / 16).map(|i| (i % 127) as u8).collect();
        assert_eq!(checked_checkpoint(&packed, &scales, 32, 64, 0.001).unwrap(), 32 * 36);
        for (rows, cols) in [(0, 64), (31, 64), (32, 63), (usize::MAX, 64)] {
            assert!(checked_checkpoint(&packed, &scales, rows, cols, 0.001).is_err());
        }
        assert!(checked_checkpoint(&packed[1..], &scales, 32, 64, 0.001).is_err());
        assert!(checked_checkpoint(&packed, &scales[1..], 32, 64, 0.001).is_err());
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY, 25.0, 1e-20] {
            assert!(checked_checkpoint(&packed, &scales, 32, 64, scale).is_err());
        }
        for byte in 127..=255 {
            scales[0] = byte;
            assert!(checked_checkpoint(&packed, &scales, 32, 64, 0.001).is_err());
        }
        scales[0] = 126;
        assert!(checked_checkpoint(&packed, &scales, 32, 64, 24.0).is_ok());
    }

    #[test]
    #[ignore = "exclusive SM70; direct pack ABI negative controls and allocation closure"]
    fn checkpoint_pack_abi_rejects_invalid_admission_without_writes() {
        let before = crate::cuda_device_allocation_snapshot();
        {
            let stream = OwnedCudaStream::create(0).unwrap();
            let source = OwnedCudaBuffer::allocate(0, 32 * 36).unwrap();
            let destination = OwnedCudaBuffer::allocate(0, 32 * 36).unwrap();
            let sentinel = vec![0xa5; 32 * 36];
            source.copy_from_host_sync(&stream, &sentinel).unwrap();
            destination.copy_from_host_sync(&stream, &sentinel).unwrap();
            let mut valid = AugurCudaQwen35QpnPackArgsV1::new();
            valid.source = view(&source, &stream, 0, sentinel.len()).unwrap().checked_address(OP).unwrap();
            valid.destination = view(&destination, &stream, 0, sentinel.len()).unwrap().checked_address(OP).unwrap();
            valid.source_bytes = sentinel.len() as u64;
            valid.destination_bytes = valid.source_bytes;
            valid.rows = 32;
            valid.cols = 64;
            let mut invalid = Vec::new();
            let mut a = valid; a.destination = a.source; invalid.push(a);
            let mut a = valid; a.source_bytes -= 1; invalid.push(a);
            let mut a = valid; a.destination_bytes -= 1; invalid.push(a);
            let mut a = valid; a.rows = 31; invalid.push(a);
            let mut a = valid; a.cols = 63; invalid.push(a);
            let mut a = valid; a.source += 1; invalid.push(a);
            let mut a = valid; a.destination += 4; invalid.push(a);
            let mut a = valid; a.abi_version = 2; invalid.push(a);
            let mut a = valid; a.struct_size -= 1; invalid.push(a);
            // SAFETY: retained live allocations and stream. Invalid structural
            // descriptors must fail before enqueue in either ABI entry.
            unsafe {
                assert!(augur_cuda_qwen35_qpn_pack_checkpoint_preflight_v1(0, stream.raw().unwrap(), &valid).is_ok());
                for args in invalid {
                    assert!(!augur_cuda_qwen35_qpn_pack_checkpoint_preflight_v1(0, stream.raw().unwrap(), &args).is_ok());
                    assert!(!augur_cuda_qwen35_qpn_pack_checkpoint_prevalidated_submit_v1(0, stream.raw().unwrap(), &args).is_ok());
                }
                assert!(!augur_cuda_qwen35_qpn_pack_checkpoint_preflight_v1(1, stream.raw().unwrap(), &valid).is_ok());
            }
            stream.synchronize().unwrap();
            let mut actual = vec![0; sentinel.len()];
            destination.copy_to_host_sync(&stream, &mut actual).unwrap();
            assert_eq!(actual, sentinel, "rejected descriptors never write destination");
        }
        let after = crate::cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
        assert_eq!(after.successful_allocations - before.successful_allocations,
            after.successful_frees - before.successful_frees);
    }

    #[test]
    fn shape_extent_and_scale_words_are_checked_before_upload() {
        let mut bytes = vec![0_u8; 32 * 36];
        assert_eq!(checked_ggml(&bytes, 32, 64, 0.001).unwrap(), bytes.len());
        assert!(checked_ggml(&bytes, 31, 64, 0.001).is_err());
        assert!(checked_ggml(&bytes, 32, 63, 0.001).is_err());
        assert!(checked_ggml(&bytes[..bytes.len() - 1], 32, 64, 0.001).is_err());
        for s in [0.0, -1.0, f32::NAN, f32::INFINITY, 25.0, 1e-20] {
            assert!(checked_ggml(&bytes, 32, 64, s).is_err());
        }
        bytes[36] = 127;
        assert!(checked_ggml(&bytes, 32, 64, 0.001).is_err());
        bytes[36] = 126;
        assert!(checked_ggml(&bytes, 32, 64, 24.0).is_ok());
    }
    #[test]
    fn legacy_nvfp4_cannot_admit_qpn() {
        let v = crate::cuda_runtime::test_buffer_view(0x1000, 0, 0, 32 * 36);
        let err = crate::CudaNvfp4Sm70WeightViewV1::new(
            v,
            PhysicalLayoutId::DEVICE_QWEN35_NVFP4_QPN_PLANES_V1,
            32,
            64,
        )
        .unwrap_err();
        assert_eq!(err.kind, CudaRuntimeErrorKind::InvalidLayout);
    }

    #[test]
    #[ignore = "requires exclusive SM70 and AUGUR_QPN_CONSUMER_FIXTURE with validated real-weight references"]
    fn real_qpn_abi_owner_roundtrip() {
        use sha2::{Digest, Sha256};
        let root = std::path::PathBuf::from(
            std::env::var_os("AUGUR_QPN_CONSUMER_FIXTURE").expect("fixture root"),
        );
        let read = |name: &str| std::fs::read(root.join(name)).unwrap();
        let metadata: serde_json::Value = serde_json::from_slice(&read("start.json")).unwrap();
        let rows = metadata["rows"].as_u64().unwrap() as usize;
        let cols = metadata["cols"].as_u64().unwrap() as usize;
        let scale = metadata["scale"].as_f64().unwrap() as f32;
        let bytes = read("weight.ggml");
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            metadata["weight_sha256"].as_str().unwrap()
        );
        let input_bytes = read("input.f32");
        assert_eq!(
            format!("{:x}", Sha256::digest(&input_bytes)),
            metadata["input_f32_sha256"].as_str().unwrap()
        );
        let before = crate::cuda_device_allocation_snapshot();
        {
            let stream = OwnedCudaStream::create(0).unwrap();
            let weight =
                OwnedCudaQwen35QpnWeightV1::upload_ggml(&stream, &bytes, rows, cols, scale)
                    .unwrap();
            let mut physical = vec![0; weight.len_bytes()];
            weight
                .storage
                .copy_to_host_sync(&stream, &mut physical)
                .unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&physical)),
                metadata["expected_physical_sha256"].as_str().unwrap()
            );
            let input = OwnedCudaBuffer::allocate(0, input_bytes.len()).unwrap();
            input.copy_from_host_sync(&stream, &input_bytes).unwrap();
            let output = OwnedCudaBuffer::allocate(0, rows * 8 * 4).unwrap();
            let workspace = CudaQwen35QpnWorkspaceV1::new(&stream, cols, 8).unwrap();
            assert_eq!(workspace.capacity_bytes(), cols * 8 * 2);
            assert!(
                workspace.bind_stage(&input, 4, 8).is_err(),
                "float2 misalignment"
            );
            assert!(workspace.bind_stage(&input, 0, 9).is_err(), "capacity");
            assert!(
                workspace.bind_projection(&weight, &output, 4, 8).is_err(),
                "output extent"
            );
            let project = workspace.bind_projection(&weight, &output, 0, 8).unwrap();
            let mut invalid = project.raw;
            invalid.numerical_profile = 0;
            // SAFETY: negative admission probes use live stream/allocation
            // owners and never submit the deliberately invalid descriptors.
            unsafe {
                assert!(!augur_cuda_qwen35_qpn_project_preflight_v1(
                    0,
                    stream.raw().unwrap(),
                    &invalid
                )
                .is_ok());
                invalid = project.raw;
                invalid.output = invalid.input;
                assert!(!augur_cuda_qwen35_qpn_project_preflight_v1(
                    0,
                    stream.raw().unwrap(),
                    &invalid
                )
                .is_ok());
                assert!(!augur_cuda_qwen35_qpn_project_preflight_v1(
                    1,
                    stream.raw().unwrap(),
                    &project.raw
                )
                .is_ok());
            }
            let expected = read("b8.f32");
            let mut output_bytes = vec![0; rows * 8 * 4];
            for width in 1_usize..=8 {
                for row in (0..8).step_by(width) {
                    let width = width.min(8 - row);
                    let stage = workspace.bind_stage(&input, row * cols * 4, width).unwrap();
                    let project = workspace
                        .bind_projection(&weight, &output, row * rows * 4, width)
                        .unwrap();
                    stage.submit_for_plan().unwrap();
                    project.submit_for_plan().unwrap();
                }
                stream.synchronize().unwrap();
                output
                    .copy_to_host_sync(&stream, &mut output_bytes)
                    .unwrap();
                assert!(
                    output_bytes == expected[..output_bytes.len()],
                    "B{width} actual ABI vs upstream"
                );
            }
            let expanded = OwnedCudaBuffer::allocate(0, rows * cols * 2).unwrap();
            for (profile, file) in [
                (Qwen35QpnExpansionV1::FoldedF16, "folded.f16"),
                (Qwen35QpnExpansionV1::LegacyUnscaled, "unscaled.f16"),
            ] {
                weight.expand_f16_sync(&stream, &expanded, profile).unwrap();
                let mut actual = vec![0; rows * cols * 2];
                expanded.copy_to_host_sync(&stream, &mut actual).unwrap();
                assert!(actual == read(file), "explicit {profile:?} expansion");
            }
            // Fixed-address graph replay must consume refreshed activations,
            // including the rare DFlash acceptance widths covered above.
            let stage = workspace.bind_stage(&input, 0, 8).unwrap();
            let project = workspace.bind_projection(&weight, &output, 0, 8).unwrap();
            let graph = crate::OwnedCudaGraphExec::capture(&stream, || {
                stage.submit_for_plan()?;
                project.submit_for_plan()
            })
            .unwrap();
            assert_eq!(graph.node_count(), 2, "one shared stage and one QPN kernel");
            let mut reversed = input_bytes.clone();
            for row in 0..8 {
                reversed[row * cols * 4..(row + 1) * cols * 4]
                    .copy_from_slice(&input_bytes[(7 - row) * cols * 4..(8 - row) * cols * 4]);
            }
            input.copy_from_host_sync(&stream, &reversed).unwrap();
            for _ in 0..3 {
                graph.launch(&stream).unwrap();
            }
            stream.synchronize().unwrap();
            output
                .copy_to_host_sync(&stream, &mut output_bytes)
                .unwrap();
            for row in 0..8 {
                assert!(
                    output_bytes[row * rows * 4..(row + 1) * rows * 4]
                        == expected[(7 - row) * rows * 4..(8 - row) * rows * 4],
                    "graph uses refreshed row{row}"
                );
            }
            input.copy_from_host_sync(&stream, &input_bytes).unwrap();
            // Exercise the real existing cuBLAS owner/workspace, not a probe's
            // private handle. Small batches and B128 consume one QPN profile.
            let plan =
                crate::CudaQwen35PrefillCublasPlanV1::new(&stream, rows * cols, cols, 128).unwrap();
            let foreign = OwnedCudaStream::create(0).unwrap();
            assert!(plan
                .bind_qpn_projection(&foreign, &weight, &input, 0, &output, 0, 8)
                .is_err());
            assert!(
                plan.bind_qpn_projection(&stream, &weight, &input, 0, &output, 0, 9)
                    .is_err(),
                "short output"
            );
            for width in 1_usize..=8 {
                for row in (0..8).step_by(width) {
                    let width = width.min(8 - row);
                    let project = plan
                        .bind_qpn_projection(
                            &stream,
                            &weight,
                            &input,
                            row * cols * 4,
                            &output,
                            row * rows * 4,
                            width,
                        )
                        .unwrap();
                    project.submit_for_plan().unwrap();
                }
                stream.synchronize().unwrap();
                output
                    .copy_to_host_sync(&stream, &mut output_bytes)
                    .unwrap();
                assert!(
                    output_bytes == expected[..output_bytes.len()],
                    "shared-plan B{width}"
                );
            }
            let wide_output = OwnedCudaBuffer::allocate(0, rows * 128 * 4).unwrap();
            let allocations = crate::cuda_device_allocation_snapshot();
            plan.bind_qpn_projection(&stream, &weight, &input, 0, &wide_output, 0, 128)
                .unwrap()
                .submit_for_plan()
                .unwrap();
            stream.synchronize().unwrap();
            let mut actual = vec![0; rows * 128 * 4];
            wide_output.copy_to_host_sync(&stream, &mut actual).unwrap();
            assert!(
                actual == read("prefill.f32"),
                "shared-plan B128 matches retained cuBLAS reference"
            );
            assert_eq!(
                crate::cuda_device_allocation_snapshot().successful_allocations,
                allocations.successful_allocations,
                "no submit-time owned allocation"
            );
            stream.synchronize().unwrap();
        }
        let after = crate::cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
        assert_eq!(
            after.successful_allocations - before.successful_allocations,
            after.successful_frees - before.successful_frees
        );
        eprintln!("QPN real ABI every width1..8/B128, refreshed-input graph, existing cuBLAS workspace, all physical/FP16 words and owner cleanup PASS: {rows}x{cols}");
    }
}
