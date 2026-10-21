#![cfg(feature = "cuda")]

//! Checked binding for the row-major GGML NVFP4 SM70 matvec component.
//!
//! The device view preserves the 36-byte GGML block wire format.  A block is
//! four UE4M3 scale bytes followed by 32 packed E2M1 bytes.  The separate
//! tensor-level scale is applied by the native kernel after the dot product.
//! This primitive is intentionally independent of the Blackwell/CUTLASS
//! NVFP4 path.

use crate::{
    CudaBufferView, CudaF32View, CudaQ8_1ActivationViewV1, CudaRuntimeError, CudaRuntimeErrorKind,
    OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;
use augur_cuda_sys::{
    AugurCudaNvfp4Sm70F32MatvecArgsV1, AugurCudaNvfp4Sm70Q8_1MatvecArgsV1,
    AUGUR_CUDA_NVFP4_SM70_F32_MATVEC_ARGS_VERSION_V1,
};

const OPERATION: &str = "cuda_nvfp4_sm70_f32_matvec_v1";
const Q8_1_OPERATION: &str = "cuda_nvfp4_sm70_q8_1_matvec_v1";
const BLOCK_ELEMENTS: usize = 64;
const BLOCK_BYTES: usize = 36;

fn validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OPERATION, kind)
}

fn q8_1_validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(Q8_1_OPERATION, kind)
}

/// A device view containing exactly `rows` GGML NVFP4 rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaNvfp4Sm70WeightViewV1 {
    view: CudaBufferView,
    rows: u32,
    cols: u32,
    blocks_per_row: u32,
}

impl CudaNvfp4Sm70WeightViewV1 {
    pub fn new(
        view: CudaBufferView,
        layout: PhysicalLayoutId,
        rows: usize,
        cols: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if layout != PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1 {
            return Err(validation(CudaRuntimeErrorKind::InvalidLayout));
        }
        if rows == 0 || cols == 0 || !cols.is_multiple_of(BLOCK_ELEMENTS) {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let blocks_per_row = cols
            .checked_div(BLOCK_ELEMENTS)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let row_bytes = blocks_per_row
            .checked_mul(BLOCK_BYTES)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let expected_bytes = rows
            .checked_mul(row_bytes)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        if view.len_bytes() != expected_bytes {
            return Err(validation(CudaRuntimeErrorKind::OutOfBounds));
        }
        Ok(Self {
            view,
            rows: rows
                .try_into()
                .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?,
            cols: cols
                .try_into()
                .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?,
            blocks_per_row: blocks_per_row
                .try_into()
                .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?,
        })
    }

    pub const fn rows(self) -> u32 {
        self.rows
    }

    pub const fn cols(self) -> u32 {
        self.cols
    }

    pub const fn blocks_per_row(self) -> u32 {
        self.blocks_per_row
    }

    pub const fn buffer_view(self) -> CudaBufferView {
        self.view
    }
}

/// Bound arguments for one asynchronous NVFP4 SM70 f32 matvec submission.
#[derive(Clone, Copy, Debug)]
pub struct CudaNvfp4Sm70F32MatvecArgsV1 {
    raw: AugurCudaNvfp4Sm70F32MatvecArgsV1,
    device_ordinal: i32,
}

impl CudaNvfp4Sm70F32MatvecArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: CudaNvfp4Sm70WeightViewV1,
        input: CudaF32View,
        output: CudaF32View,
        tensor_scale: f32,
    ) -> Result<Self, CudaRuntimeError> {
        let ordinal = stream.device_ordinal();
        let weight_view = weight.view;
        let input_view = input.buffer_view();
        let output_view = output.buffer_view();
        if !tensor_scale.is_finite() || tensor_scale <= 0.0 {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        if [weight_view, input_view, output_view]
            .iter()
            .any(|view| view.device_ordinal() != ordinal)
            || input.element_count() != weight.cols as usize
            || output.element_count() < weight.rows as usize
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if weight_view.overlaps(input_view, OPERATION)?
            || weight_view.overlaps(output_view, OPERATION)?
            || input_view.overlaps(output_view, OPERATION)?
        {
            return Err(validation(CudaRuntimeErrorKind::OverlappingDeviceRanges));
        }

        let mut raw = AugurCudaNvfp4Sm70F32MatvecArgsV1::new();
        raw.abi_version = AUGUR_CUDA_NVFP4_SM70_F32_MATVEC_ARGS_VERSION_V1;
        raw.weight = address(weight_view)?;
        raw.weight_len_bytes = u64::try_from(weight_view.len_bytes())
            .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        raw.input = address(input_view)?;
        raw.input_len_bytes = u64::try_from(input_view.len_bytes())
            .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        raw.output = address(output_view)?;
        raw.output_len_bytes = u64::try_from(output_view.len_bytes())
            .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        raw.rows = weight.rows;
        raw.cols = weight.cols;
        raw.tensor_scale = tensor_scale;
        raw.reserved = 0;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn as_raw(&self) -> &AugurCudaNvfp4Sm70F32MatvecArgsV1 {
        &self.raw
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: `raw` was populated from checked non-owning views. The
        // caller keeps the owning buffers alive until the synchronous launch
        // returns; the native shim repeats ABI and extent validation.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_nvfp4_sm70_f32_matvec_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    /// Enqueue the kernel without a host synchronization.
    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: same binding contract as `launch`; owners must outlive the
        // stream completion proof for this asynchronous submission.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_nvfp4_sm70_f32_matvec_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_nvfp4_sm70_f32_matvec_submit_v1", result)
    }

    /// Separate large-matrix research schedule. Original F32 pair products,
    /// contiguous SplitK8 partitions, ordered F64 fold; no HBM scratch.
    pub fn submit_large_split8_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        if self.raw.rows < 1024 || self.raw.rows % 32 != 0 || self.raw.cols < 5120 || self.raw.cols % 512 != 0 {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        // SAFETY: checked original descriptors and retained owners; the native
        // entry independently checks shape, exact extents and disjointness.
        let result = unsafe { augur_cuda_sys::augur_cuda_nvfp4_sm70_large_split8_submit_v1(
            stream.device_ordinal(), stream.raw()?, &self.raw) };
        CudaRuntimeError::from_result("cuda_nvfp4_sm70_large_split8_submit_v1", result)
    }

    /// Explicit 48x5120-only small-gate contract: unchanged F32 pair expression,
    /// consecutive SplitK16 F64 partials, ordered split fold, then warp fold.
    /// The F64 addition order differs from the original scalar entry point.
    pub fn submit_small_gate_split16_for_plan(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        if self.raw.rows != 48 || self.raw.cols != 5120 {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        // SAFETY: bound owners/ranges live through stream completion; the native
        // entry point rechecks shape, exact extents, alignment and disjointness.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_nvfp4_sm70_small_gate_split16_submit_v1(
                stream.device_ordinal(), stream.raw()?, &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_nvfp4_sm70_small_gate_split16_submit_v1", result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        Ok(())
    }
}

/// Launch schedules for the SAME per-row SplitK16 F32-pair/F64-fold math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum CudaNvfp4SmallGatePairSchedule {
    FixedKRowConcatenated = 1,
    FixedKSharedInputPrefetch = 2,
}

/// Prebound real 48x5120 pair, retained by the runtime's two projection owners.
/// No batch padding or input caching by address. Exact stream identity is fixed.
#[derive(Clone, Copy, Debug)]
pub struct CudaNvfp4Sm70SmallGatePairArgsV1 {
    a: CudaNvfp4Sm70F32MatvecArgsV1,
    b: CudaNvfp4Sm70F32MatvecArgsV1,
    schedule: CudaNvfp4SmallGatePairSchedule,
    stream_identity: usize,
}
impl CudaNvfp4Sm70SmallGatePairArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        a: CudaNvfp4Sm70F32MatvecArgsV1,
        b: CudaNvfp4Sm70F32MatvecArgsV1,
        schedule: CudaNvfp4SmallGatePairSchedule,
    ) -> Result<Self, CudaRuntimeError> {
        a.validate_stream(stream)?;
        b.validate_stream(stream)?;
        if a.raw.rows != 48 || b.raw.rows != 48 || a.raw.cols != 5120 || b.raw.cols != 5120
            || a.raw.input != b.raw.input || a.raw.input_len_bytes != b.raw.input_len_bytes {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let raw_stream = stream.raw()?;
        // SAFETY: native prevalidation inspects descriptors only; it never queues
        // work or dereferences device memory. All ranges were bound by the owner.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_nvfp4_sm70_small_gate_pair_prevalidate_v1(
                stream.device_ordinal(), raw_stream, &a.raw, &b.raw, schedule as u32,
            )
        };
        CudaRuntimeError::from_result("cuda_nvfp4_sm70_small_gate_pair_prevalidate_v1", result)?;
        Ok(Self {a,b,schedule,stream_identity:raw_stream as usize})
    }
    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.a.validate_stream(stream)?;
        self.b.validate_stream(stream)?;
        if stream.raw()? as usize != self.stream_identity {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        // SAFETY: the runtime retains BOTH weight/input/output owners through
        // completion on this exact stream; native rechecks all writable aliases.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_nvfp4_sm70_small_gate_pair_submit_v1(
                stream.device_ordinal(), stream.raw()?, &self.a.raw, &self.b.raw, self.schedule as u32,
            )
        };
        CudaRuntimeError::from_result("cuda_nvfp4_sm70_small_gate_pair_submit_v1", result)
    }
}

/// Bound arguments for an NVFP4-weight by Q8_1-activation SM70 DP4A matvec.
#[derive(Clone, Copy, Debug)]
pub struct CudaNvfp4Sm70Q8_1MatvecArgsV1 {
    raw: AugurCudaNvfp4Sm70Q8_1MatvecArgsV1,
    device_ordinal: i32,
}

impl CudaNvfp4Sm70Q8_1MatvecArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: CudaNvfp4Sm70WeightViewV1,
        input: CudaQ8_1ActivationViewV1,
        output: CudaF32View,
        tensor_scale: f32,
    ) -> Result<Self, CudaRuntimeError> {
        let ordinal = stream.device_ordinal();
        let weight_view = weight.buffer_view();
        let input_view = input.buffer_view();
        let output_view = output.buffer_view();
        if !tensor_scale.is_finite() || tensor_scale <= 0.0 {
            return Err(q8_1_validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        if [weight_view, input_view, output_view]
            .iter()
            .any(|view| view.device_ordinal() != ordinal)
            || input.elements() != weight.cols as usize
            || output.element_count() < weight.rows as usize
        {
            return Err(q8_1_validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if weight_view.overlaps(input_view, Q8_1_OPERATION)?
            || weight_view.overlaps(output_view, Q8_1_OPERATION)?
            || input_view.overlaps(output_view, Q8_1_OPERATION)?
        {
            return Err(q8_1_validation(
                CudaRuntimeErrorKind::OverlappingDeviceRanges,
            ));
        }
        let weight_address = address_for(Q8_1_OPERATION, weight_view)?;
        if weight_address % core::mem::align_of::<u32>() as u64 != 0 {
            return Err(q8_1_validation(
                CudaRuntimeErrorKind::MisalignedDeviceAddress,
            ));
        }

        let mut raw = AugurCudaNvfp4Sm70Q8_1MatvecArgsV1::new();
        raw.weight = weight_address;
        raw.weight_len_bytes = u64::try_from(weight_view.len_bytes())
            .map_err(|_| q8_1_validation(CudaRuntimeErrorKind::RangeOverflow))?;
        raw.input_q8_1 = address_for(Q8_1_OPERATION, input_view)?;
        raw.input_len_bytes = u64::try_from(input_view.len_bytes())
            .map_err(|_| q8_1_validation(CudaRuntimeErrorKind::RangeOverflow))?;
        raw.output = address_for(Q8_1_OPERATION, output_view)?;
        raw.output_len_bytes = u64::try_from(output_view.len_bytes())
            .map_err(|_| q8_1_validation(CudaRuntimeErrorKind::RangeOverflow))?;
        raw.rows = weight.rows;
        raw.cols = weight.cols;
        raw.tensor_scale = tensor_scale;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn as_raw(&self) -> &AugurCudaNvfp4Sm70Q8_1MatvecArgsV1 {
        &self.raw
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: the checked binding owns exact, disjoint device ranges and
        // the synchronous native entry point completes before returning.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_nvfp4_sm70_q8_1_matvec_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(Q8_1_OPERATION, result)
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: the buffers behind the checked views outlive completion.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_nvfp4_sm70_q8_1_matvec_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_nvfp4_sm70_q8_1_matvec_submit_v1", result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(q8_1_validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        Ok(())
    }
}

fn address(view: CudaBufferView) -> Result<u64, CudaRuntimeError> {
    view.device_ptr()
        .raw_value()
        .checked_add(
            u64::try_from(view.offset_bytes())
                .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?,
        )
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))
}

/// Real row-major batch1..8 with the original F32 pair/double reduction math.
/// The caller retains weights and both matrix owners through stream completion.
#[derive(Clone, Copy, Debug)]
pub struct CudaNvfp4Sm70F32BatchArgsV1 {
    scalar: CudaNvfp4Sm70F32MatvecArgsV1,
    batch_rows: u32,
}

fn batch_extent(
    rows: usize,
    cols: usize,
    batch: usize,
) -> Result<(usize, usize), CudaRuntimeError> {
    if !(1..=8).contains(&batch) || rows == 0 || cols == 0 || cols % 64 != 0 {
        return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
    }
    let extent = |n: usize| {
        n.checked_mul(batch)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))
    };
    Ok((extent(cols)?, extent(rows)?))
}

impl CudaNvfp4Sm70F32BatchArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: CudaNvfp4Sm70WeightViewV1,
        input: CudaF32View,
        output: CudaF32View,
        tensor_scale: f32,
        batch_rows: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let (x, y) = batch_extent(weight.rows as usize, weight.cols as usize, batch_rows)?;
        if input.element_count() != x / 4
            || output.element_count() != y / 4
            || input.buffer_view().len_bytes() != x
            || output.buffer_view().len_bytes() != y
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let w = weight.view;
        let iv = input.buffer_view();
        let ov = output.buffer_view();
        if [w, iv, ov]
            .iter()
            .any(|v| v.device_ordinal() != stream.device_ordinal())
        {
            return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        if w.overlaps(iv, OPERATION)? || w.overlaps(ov, OPERATION)? || iv.overlaps(ov, OPERATION)? {
            return Err(validation(CudaRuntimeErrorKind::OverlappingDeviceRanges));
        }
        if !tensor_scale.is_finite() || tensor_scale <= 0.0 {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        let mut raw = AugurCudaNvfp4Sm70F32MatvecArgsV1::new();
        raw.weight = address(w)?;
        raw.weight_len_bytes = w.len_bytes() as u64;
        raw.input = address(iv)?;
        raw.input_len_bytes = x as u64;
        raw.output = address(ov)?;
        raw.output_len_bytes = y as u64;
        raw.rows = weight.rows;
        raw.cols = weight.cols;
        raw.tensor_scale = tensor_scale;
        Ok(Self {
            scalar: CudaNvfp4Sm70F32MatvecArgsV1 {
                raw,
                device_ordinal: stream.device_ordinal(),
            },
            batch_rows: batch_rows as u32,
        })
    }
    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.scalar.validate_stream(stream)?;
        // SAFETY: checked descriptor/extents; owners must outlive completion.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_nvfp4_sm70_f32_batch_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.scalar.raw,
                self.batch_rows,
            )
        };
        CudaRuntimeError::from_result("cuda_nvfp4_sm70_f32_batch_submit_v1", result)
    }
}

fn address_for(operation: &'static str, view: CudaBufferView) -> Result<u64, CudaRuntimeError> {
    view.device_ptr()
        .raw_value()
        .checked_add(u64::try_from(view.offset_bytes()).map_err(|_| {
            CudaRuntimeError::validation(operation, CudaRuntimeErrorKind::RangeOverflow)
        })?)
        .ok_or_else(|| CudaRuntimeError::validation(operation, CudaRuntimeErrorKind::RangeOverflow))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_runtime::test_buffer_view;

    #[test]
    fn small_gate_pair_null_native_descriptors_are_rejected() {
        for schedule in 0..=3 {
            let result = unsafe {
                augur_cuda_sys::augur_cuda_nvfp4_sm70_small_gate_pair_prevalidate_v1(
                    0, std::ptr::null_mut(), std::ptr::null(), std::ptr::null(), schedule,
                )
            };
            assert_ne!(result.status,0);
            let result = unsafe {
                augur_cuda_sys::augur_cuda_nvfp4_sm70_small_gate_pair_submit_v1(
                    0, std::ptr::null_mut(), std::ptr::null(), std::ptr::null(), schedule,
                )
            };
            assert_ne!(result.status,0);
        }
    }

    #[test]
    #[ignore = "exclusive CUDA; paired real-shape ABI, stream, refresh and cleanup"]
    fn small_gate_pair_schedules_owner_rejection_and_refresh_gpu() {
        use crate::{cuda_device_allocation_snapshot, OwnedCudaBuffer};
        use CudaNvfp4SmallGatePairSchedule::*;
        let baseline = cuda_device_allocation_snapshot();
        {
            let stream = OwnedCudaStream::create(0).unwrap();
            let foreign = OwnedCudaStream::create(0).unwrap();
            let wa = OwnedCudaBuffer::allocate(0,48*80*36).unwrap();
            let wb = OwnedCudaBuffer::allocate(0,48*80*36).unwrap();
            let x = OwnedCudaBuffer::allocate(0,2*5120*4).unwrap();
            let ya = OwnedCudaBuffer::allocate(0,48*4).unwrap();
            let yb = OwnedCudaBuffer::allocate(0,48*4).unwrap();
            let ra = OwnedCudaBuffer::allocate(0,48*4).unwrap();
            let rb = OwnedCudaBuffer::allocate(0,48*4).unwrap();
            for (w,phase) in [(&wa,0usize),(&wb,23usize)] {
                let mut bytes=vec![0u8;48*80*36];
                for (r,block) in bytes.chunks_mut(36).enumerate() {
                    for i in 0..4 {block[i]=((r*13+i*7+phase)%127) as u8;}
                    for i in 4..36 {block[i]=(r*19+i*31+phase) as u8;}
                }
                w.copy_from_host_sync(&stream,&bytes).unwrap();
            }
            let f=|b:&OwnedCudaBuffer,offset,n|CudaF32View::new(
                b.view(offset,n*4).unwrap(),PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,n,
            ).unwrap();
            let w=|b:&OwnedCudaBuffer|CudaNvfp4Sm70WeightViewV1::new(
                b.view(0,b.len_bytes()).unwrap(),PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,48,5120,
            ).unwrap();
            let a=CudaNvfp4Sm70F32MatvecArgsV1::bind(&stream,w(&wa),f(&x,0,5120),f(&ya,0,48),0.03125).unwrap();
            let b=CudaNvfp4Sm70F32MatvecArgsV1::bind(&stream,w(&wb),f(&x,0,5120),f(&yb,0,48),0.015625).unwrap();
            let ar=CudaNvfp4Sm70F32MatvecArgsV1::bind(&stream,w(&wa),f(&x,0,5120),f(&ra,0,48),0.03125).unwrap();
            let br=CudaNvfp4Sm70F32MatvecArgsV1::bind(&stream,w(&wb),f(&x,0,5120),f(&rb,0,48),0.015625).unwrap();
            let before=cuda_device_allocation_snapshot();
            for schedule in [FixedKRowConcatenated,FixedKSharedInputPrefetch] {
                let pair=CudaNvfp4Sm70SmallGatePairArgsV1::bind(&stream,a,b,schedule).unwrap();
                assert!(pair.submit_for_plan(&foreign).is_err());
                for phase in 0..3 {
                    let values:Vec<u8>=(0..2*5120).flat_map(|i| {
                        let v=(((i*17+phase*11)%109) as f32-54.)/17.;
                        (if i%5==0 {v*128.} else {v}).to_le_bytes()
                    }).collect();
                    x.copy_from_host_sync(&stream,&values).unwrap();
                    ar.submit_small_gate_split16_for_plan(&stream).unwrap();
                    br.submit_small_gate_split16_for_plan(&stream).unwrap();
                    pair.submit_for_plan(&stream).unwrap();
                    for (actual,expected) in [(&ya,&ra),(&yb,&rb)] {
                        let mut av=vec![0u8;48*4];let mut ev=av.clone();
                        actual.copy_to_host_sync(&stream,&mut av).unwrap();expected.copy_to_host_sync(&stream,&mut ev).unwrap();
                        assert_eq!(av,ev,"schedule={schedule:?} phase={phase}");
                    }
                }
                for mutation in 0..20 {
                    let (mut left,mut right)=(a.raw,b.raw);
                    match mutation {
                        0=>right.input+=5120*4,
                        1=>right.output_len_bytes-=4,
                        2=>right.output=left.output,
                        3=>right.output=left.weight,
                        4=>left.output=right.weight,
                        5=>{left.input=right.weight;right.input=right.weight;},
                        6=>right.weight_len_bytes-=1,
                        7=>{right.rows=32;right.weight_len_bytes=32*80*36;right.output_len_bytes=32*4;},
                        8=>{right.cols=5056;right.weight_len_bytes=48*79*36;right.input_len_bytes=5056*4;},
                        9=>right.reserved=1,
                        10=>right.tensor_scale=f32::NAN,
                        11=>right.abi_version+=1,
                        12=>right.struct_size-=1,
                        13=>right.weight+=1,
                        14=>right.output+=1,
                        15=>right.tensor_scale=0.,
                        16=>right.output_len_bytes+=4,
                        17=>{left.input_len_bytes*=2;right.input_len_bytes*=2;},
                        18=>right.weight=u64::MAX-16,
                        _=>right.output=left.input,
                    }
                    for submit in [false,true] {
                        let result=unsafe {
                            if submit {augur_cuda_sys::augur_cuda_nvfp4_sm70_small_gate_pair_submit_v1(0,stream.raw().unwrap(),&left,&right,schedule as u32)}
                            else {augur_cuda_sys::augur_cuda_nvfp4_sm70_small_gate_pair_prevalidate_v1(0,stream.raw().unwrap(),&left,&right,schedule as u32)}
                        };
                        assert_ne!(result.status,0,"mutation={mutation} submit={submit}");
                    }
                }
                for bad_schedule in [0,3,u32::MAX] {
                    let result=unsafe {augur_cuda_sys::augur_cuda_nvfp4_sm70_small_gate_pair_submit_v1(0,stream.raw().unwrap(),&a.raw,&b.raw,bad_schedule)};
                    assert_ne!(result.status,0);
                }
                let mut other=b;other.raw.input+=5120*4;
                assert!(CudaNvfp4Sm70SmallGatePairArgsV1::bind(&stream,a,other,schedule).is_err());
            }
            stream.synchronize().unwrap();
            assert_eq!(before.successful_allocations,cuda_device_allocation_snapshot().successful_allocations);
        }
        let after=cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes,baseline.live_bytes);assert_eq!(after.live_allocations,baseline.live_allocations);
        assert_eq!(after.successful_allocations-baseline.successful_allocations,after.successful_frees-baseline.successful_frees);
    }

    #[test]
    fn small_gate_null_native_descriptor_is_rejected() {
        let result = unsafe {
            augur_cuda_sys::augur_cuda_nvfp4_sm70_small_gate_split16_submit_v1(
                0, std::ptr::null_mut(), std::ptr::null(),
            )
        };
        assert_ne!(result.status, 0);
    }

    #[test]
    #[ignore = "exclusive CUDA; independent small-gate ABI and owner cleanup"]
    fn small_gate_split16_native_owner_and_rejection_gpu() {
        use crate::{cuda_device_allocation_snapshot, OwnedCudaBuffer};
        let baseline = cuda_device_allocation_snapshot();
        {
            let stream = OwnedCudaStream::create(0).unwrap();
            let weight = OwnedCudaBuffer::allocate(0, 48 * 80 * 36).unwrap();
            let input = OwnedCudaBuffer::allocate(0, 8 * 5120 * 4).unwrap();
            let output = OwnedCudaBuffer::allocate(0, 8 * 48 * 4).unwrap();
            let reference = OwnedCudaBuffer::allocate(0, 8 * 48 * 4).unwrap();
            let mut data = vec![0u8; 48 * 80 * 36];
            for (r, block) in data.chunks_mut(36).enumerate() {
                for i in 0..4 { block[i] = ((r * 7 + i * 13) % 127) as u8; }
                for i in 4..36 { block[i] = (r * 31 + i * 19) as u8; }
            }
            weight.copy_from_host_sync(&stream, &data).unwrap();
            let values: Vec<u8> = (0..8 * 5120)
                .flat_map(|i| (((i * 17 % 83) as f32 - 41.) / 13.).to_le_bytes()).collect();
            input.copy_from_host_sync(&stream, &values).unwrap();
            let w = CudaNvfp4Sm70WeightViewV1::new(
                weight.view(0, weight.len_bytes()).unwrap(),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1, 48, 5120,
            ).unwrap();
            let f = |b: &OwnedCudaBuffer, offset, n| CudaF32View::new(
                b.view(offset, n * 4).unwrap(),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, n,
            ).unwrap();
            let before = cuda_device_allocation_snapshot();
            for t in 0..8 {
                let op = CudaNvfp4Sm70F32MatvecArgsV1::bind(
                    &stream, w, f(&input,t*5120*4,5120), f(&output,t*48*4,48), 0.03125,
                ).unwrap();
                op.submit_small_gate_split16_for_plan(&stream).unwrap();
                CudaNvfp4Sm70F32MatvecArgsV1::bind(
                    &stream, w, f(&input,t*5120*4,5120), f(&reference,t*48*4,48), 0.03125,
                ).unwrap().submit_for_plan(&stream).unwrap();
                for mutation in 0..15 {
                    let mut raw = op.raw;
                    match mutation {
                        0 => raw.input_len_bytes -= 4,
                        1 => raw.output_len_bytes -= 4,
                        2 => raw.output = raw.input,
                        3 => raw.input = raw.weight,
                        4 => raw.reserved = 1,
                        5 => raw.tensor_scale = f32::NAN,
                        6 => raw.input += 1,
                        7 => raw.weight += 1,
                        8 => raw.input_len_bytes += 4,
                        9 => raw.output_len_bytes += 4,
                        10 => {raw.rows=32; raw.weight_len_bytes=32*80*36; raw.output_len_bytes=32*4;},
                        11 => {raw.cols=5056; raw.weight_len_bytes=48*79*36; raw.input_len_bytes=5056*4;},
                        12 => raw.output = raw.weight,
                        13 => raw.abi_version += 1,
                        _ => raw.tensor_scale = 0.,
                    }
                    let status = unsafe {
                        augur_cuda_sys::augur_cuda_nvfp4_sm70_small_gate_split16_submit_v1(
                            0, stream.raw().unwrap(), &raw,
                        )
                    };
                    assert_ne!(status.status, 0, "small-gate mutation {mutation}");
                }
                let mut bad = op;
                bad.raw.rows = 32;
                assert!(bad.submit_small_gate_split16_for_plan(&stream).is_err());
            }
            let mut actual = vec![0u8; 8*48*4]; let mut expected = actual.clone();
            output.copy_to_host_sync(&stream, &mut actual).unwrap();
            reference.copy_to_host_sync(&stream, &mut expected).unwrap();
            // A fixture assertion, not a general same-bits claim for this tree.
            assert_eq!(actual, expected);
            stream.synchronize().unwrap();
            assert_eq!(before.successful_allocations, cuda_device_allocation_snapshot().successful_allocations);
        }
        let after = cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes, baseline.live_bytes);
        assert_eq!(after.live_allocations, baseline.live_allocations);
        assert_eq!(after.successful_allocations-baseline.successful_allocations,
                   after.successful_frees-baseline.successful_frees);
    }

    #[test]
    fn exact_f32_batch_geometry_is_real_and_bounded() {
        for b in 1..=8 {
            assert_eq!(
                batch_extent(17408, 5120, b).unwrap(),
                (b * 5120 * 4, b * 17408 * 4)
            );
        }
        for b in [0, 9, usize::MAX] {
            assert!(batch_extent(17408, 5120, b).is_err());
        }
        assert!(batch_extent(0, 5120, 1).is_err());
        assert!(batch_extent(1, 63, 1).is_err());
        assert!(batch_extent(usize::MAX, 64, 8).is_err());
        // Invalid/null descriptors reject before device selection or launch.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_nvfp4_sm70_f32_batch_submit_v1(
                0,
                std::ptr::null_mut(),
                std::ptr::null(),
                8,
            )
        };
        assert_ne!(result.status, 0);
    }

    #[test]
    #[ignore = "exclusive CUDA; exact batch ABI, alias and extent checks"]
    fn exact_f32_batch_native_owner_and_rejection_gpu() {
        use crate::{cuda_device_allocation_snapshot, OwnedCudaBuffer};
        let baseline = cuda_device_allocation_snapshot();
        {
            let stream = OwnedCudaStream::create(0).unwrap();
            let weight = OwnedCudaBuffer::allocate(0, 4 * 36).unwrap();
            let input = OwnedCudaBuffer::allocate(0, 8 * 64 * 4).unwrap();
            let output = OwnedCudaBuffer::allocate(0, 8 * 4 * 4).unwrap();
            let reference = OwnedCudaBuffer::allocate(0, 8 * 4 * 4).unwrap();
            let mut data = vec![0u8; 4 * 36];
            for (r, block) in data.chunks_mut(36).enumerate() {
                for i in 0..4 {
                    block[i] = 56 + i as u8;
                }
                for i in 4..36 {
                    block[i] = (r * 31 + i * 19) as u8;
                }
            }
            weight.copy_from_host_sync(&stream, &data).unwrap();
            let values: Vec<u8> = (0..8 * 64)
                .flat_map(|i| (((i * 17 % 83) as f32 - 41.) / 13.).to_le_bytes())
                .collect();
            input.copy_from_host_sync(&stream, &values).unwrap();
            let w = CudaNvfp4Sm70WeightViewV1::new(
                weight.view(0, weight.len_bytes()).unwrap(),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,
                4,
                64,
            )
            .unwrap();
            let f = |b: &OwnedCudaBuffer, offset, n| {
                CudaF32View::new(
                    b.view(offset, n * 4).unwrap(),
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    n,
                )
                .unwrap()
            };
            let before = cuda_device_allocation_snapshot();
            for b in 1..=8 {
                let batch = CudaNvfp4Sm70F32BatchArgsV1::bind(
                    &stream,
                    w,
                    f(&input, 0, b * 64),
                    f(&output, 0, b * 4),
                    0.03125,
                    b,
                )
                .unwrap();
                batch.submit_for_plan(&stream).unwrap();
                for t in 0..b {
                    CudaNvfp4Sm70F32MatvecArgsV1::bind(
                        &stream,
                        w,
                        f(&input, t * 64 * 4, 64),
                        f(&reference, t * 4 * 4, 4),
                        0.03125,
                    )
                    .unwrap()
                    .submit_for_plan(&stream)
                    .unwrap();
                }
                let mut actual = vec![0u8; b * 4 * 4];
                let mut expected = actual.clone();
                output
                    .view(0, actual.len())
                    .unwrap()
                    .copy_to_host_sync(&stream, &mut actual)
                    .unwrap();
                reference
                    .view(0, expected.len())
                    .unwrap()
                    .copy_to_host_sync(&stream, &mut expected)
                    .unwrap();
                assert_eq!(actual, expected);
                assert!(CudaNvfp4Sm70F32BatchArgsV1::bind(
                    &stream,
                    w,
                    f(&input, 0, b * 64),
                    f(&input, 0, b * 4),
                    1.,
                    b
                )
                .is_err());
                for mutation in 0..6 {
                    let mut raw = batch.scalar.raw;
                    match mutation {
                        0 => raw.input_len_bytes -= 4,
                        1 => raw.output_len_bytes -= 4,
                        2 => raw.output = raw.input,
                        3 => raw.input = raw.weight,
                        4 => raw.reserved = 1,
                        _ => raw.tensor_scale = f32::NAN,
                    }
                    let status = unsafe {
                        augur_cuda_sys::augur_cuda_nvfp4_sm70_f32_batch_submit_v1(
                            0,
                            stream.raw().unwrap(),
                            &raw,
                            b as u32,
                        )
                    };
                    assert_ne!(
                        status.status, 0,
                        "invalid raw descriptor mutation{mutation}"
                    );
                }
            }
            stream.synchronize().unwrap();
            assert_eq!(
                before.successful_allocations,
                cuda_device_allocation_snapshot().successful_allocations
            );
        }
        let after = cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes, baseline.live_bytes);
        assert_eq!(after.live_allocations, baseline.live_allocations);
        assert_eq!(
            after.successful_allocations - baseline.successful_allocations,
            after.successful_frees - baseline.successful_frees
        );
    }

    #[test]
    fn exact_nvfp4_layout_binds() {
        let view = test_buffer_view(0x1000, 0, 0, 2 * 2 * BLOCK_BYTES);
        let weight = CudaNvfp4Sm70WeightViewV1::new(
            view,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,
            2,
            128,
        )
        .unwrap();
        assert_eq!(weight.rows(), 2);
        assert_eq!(weight.cols(), 128);
        assert_eq!(weight.blocks_per_row(), 2);
    }

    #[test]
    fn compact_layout_is_rejected() {
        let error = CudaNvfp4Sm70WeightViewV1::new(
            test_buffer_view(0x1000, 0, 0, BLOCK_BYTES),
            PhysicalLayoutId::NVFP4_COMPACT_V1,
            1,
            64,
        )
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::InvalidLayout);
    }

    #[test]
    fn columns_must_be_nvfp4_block_multiple() {
        let error = CudaNvfp4Sm70WeightViewV1::new(
            test_buffer_view(0x1000, 0, 0, BLOCK_BYTES),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,
            1,
            63,
        )
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::InvalidDimensions);
    }

    #[test]
    fn weight_payload_length_must_be_exact() {
        let error = CudaNvfp4Sm70WeightViewV1::new(
            test_buffer_view(0x1000, 0, 0, BLOCK_BYTES - 1),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,
            1,
            64,
        )
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::OutOfBounds);
    }
}
