#![cfg(feature = "cuda")]

use augur_core::PhysicalLayoutId;
use augur_cuda_sys::{
    AugurCudaQ4kF32DecodeArgsV1, AugurCudaQ4kF32PrefillArgsV1, AUGUR_CUDA_Q4K_BLOCK_ALIGNMENT_V1,
    AUGUR_CUDA_Q4K_BLOCK_BYTES_V1, AUGUR_CUDA_Q4K_BLOCK_DECODED_LEN_V1,
};

use crate::{
    cuda_device_properties, CudaBufferView, CudaComputeCapability, CudaRuntimeError,
    CudaRuntimeErrorKind, OwnedCudaStream,
};
use augur_cuda_sys;

const OPERATION: &str = "cuda_q4k_f32_decode_v1";
const PREFILL_OPERATION: &str = "cuda_q4k_f32_prefill_v1";

fn validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OPERATION, kind)
}

fn prefill_validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(PREFILL_OPERATION, kind)
}

fn to_u64(value: usize) -> Result<u64, CudaRuntimeError> {
    u64::try_from(value).map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))
}

fn to_u64_prefill(value: usize) -> Result<u64, CudaRuntimeError> {
    u64::try_from(value).map_err(|_| prefill_validation(CudaRuntimeErrorKind::RangeOverflow))
}

// ── CudaQ4kWeightViewV1 ───────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaQ4kWeightViewV1 {
    view: CudaBufferView,
    rows: u32,
    cols: u32,
    blocks_per_row: u32,
}

impl CudaQ4kWeightViewV1 {
    pub fn new(
        view: CudaBufferView,
        layout: PhysicalLayoutId,
        rows: usize,
        cols: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if layout != PhysicalLayoutId::Q4K_HOT_CUDA_V1 {
            return Err(validation(CudaRuntimeErrorKind::InvalidLayout));
        }

        let decoded_len = AUGUR_CUDA_Q4K_BLOCK_DECODED_LEN_V1 as usize;

        if rows == 0 || cols == 0 || !cols.is_multiple_of(decoded_len) {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }

        let blocks_per_row = cols / decoded_len;
        let block_count = rows
            .checked_mul(blocks_per_row)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let expected_bytes = block_count
            .checked_mul(AUGUR_CUDA_Q4K_BLOCK_BYTES_V1 as usize)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;

        if view.len_bytes() != expected_bytes {
            return Err(validation(CudaRuntimeErrorKind::OutOfBounds));
        }

        let address = view
            .device_ptr()
            .raw_value()
            .checked_add(view.offset_bytes() as u64)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;

        if address % u64::from(AUGUR_CUDA_Q4K_BLOCK_ALIGNMENT_V1) != 0 {
            return Err(validation(CudaRuntimeErrorKind::MisalignedDeviceAddress));
        }

        Ok(Self {
            view,
            rows: u32::try_from(rows)
                .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?,
            cols: u32::try_from(cols)
                .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?,
            blocks_per_row: u32::try_from(blocks_per_row)
                .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?,
        })
    }

    pub const fn rows(&self) -> u32 {
        self.rows
    }

    pub const fn cols(&self) -> u32 {
        self.cols
    }

    pub const fn blocks_per_row(&self) -> u32 {
        self.blocks_per_row
    }

    /// Raw immutable storage view used by audited composite-weight assembly.
    pub const fn buffer_view(self) -> CudaBufferView {
        self.view
    }
}

// ── CudaF32View ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaF32View {
    view: CudaBufferView,
    element_count: u32,
}

impl CudaF32View {
    pub fn new(
        view: CudaBufferView,
        layout: PhysicalLayoutId,
        element_count: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if layout != PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1 {
            return Err(validation(CudaRuntimeErrorKind::InvalidLayout));
        }

        let expected_bytes = element_count
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;

        if view.len_bytes() < expected_bytes {
            return Err(validation(CudaRuntimeErrorKind::OutOfBounds));
        }

        let address = view
            .device_ptr()
            .raw_value()
            .checked_add(view.offset_bytes() as u64)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;

        if address % core::mem::align_of::<f32>() as u64 != 0 {
            return Err(validation(CudaRuntimeErrorKind::MisalignedDeviceAddress));
        }

        Ok(Self {
            view,
            element_count: u32::try_from(element_count)
                .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?,
        })
    }

    pub const fn buffer_view(self) -> CudaBufferView {
        self.view
    }

    pub const fn element_count(self) -> usize {
        self.element_count as usize
    }
}

// ── CudaQ4kF32DecodeArgsV1 ────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct CudaQ4kF32DecodeArgsV1 {
    raw: AugurCudaQ4kF32DecodeArgsV1,
    /// Device ordinal captured during binding; launches may not retarget it.
    device_ordinal: i32,
}

impl CudaQ4kF32DecodeArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: CudaQ4kWeightViewV1,
        input: CudaF32View,
        output: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        let ordinal = stream.device_ordinal();

        for view_ordinal in [
            weight.view.device_ordinal(),
            input.view.device_ordinal(),
            output.view.device_ordinal(),
        ] {
            if view_ordinal != ordinal {
                return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
            }
        }

        if input.element_count != weight.cols() || output.element_count < weight.rows() {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }

        let properties = cuda_device_properties(ordinal)?;

        if !properties
            .compute_capability
            .is_at_least(CudaComputeCapability::SM70)
        {
            return Err(validation(CudaRuntimeErrorKind::UnsupportedCapability));
        }

        let mut raw = AugurCudaQ4kF32DecodeArgsV1::new();
        raw.weight = addr_from_view(weight.view)?;
        raw.weight_len_bytes = to_u64(weight.view.len_bytes())?;
        raw.input = addr_from_view(input.view)?;
        raw.input_len_bytes = to_u64(input.view.len_bytes())?;
        raw.output = addr_from_view(output.view)?;
        raw.output_len_bytes = to_u64(output.view.len_bytes())?;
        raw.rows = weight.rows();
        raw.cols = weight.cols();
        raw.blocks_per_row = weight.blocks_per_row();
        raw.weight_block_stride_bytes = AUGUR_CUDA_Q4K_BLOCK_BYTES_V1;

        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn as_raw(&self) -> &AugurCudaQ4kF32DecodeArgsV1 {
        &self.raw
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        validate_launch_device(OPERATION, self.device_ordinal, stream.device_ordinal())?;

        // SAFETY: `&self.raw` is owned args populated by `bind`, which proved layout,
        // dimensions, and alignment via CudaQ4kWeightViewV1::new / CudaF32View::new on the
        // ordinal re-checked by validate_launch_device; `stream.raw()?` fails once closed.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_q4k_f32_decode_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_q4k_f32_decode_v1", result)
    }

    #[doc(hidden)]
    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        validate_launch_device(OPERATION, self.device_ordinal, stream.device_ordinal())?;
        // SAFETY: Same bind-established contract as `launch`; completion is asynchronous,
        // so the caller must keep the viewed OwnedCudaBuffers alive until the stream
        // drains (CudaBufferView carries no lifetime — see findings).
        let result = unsafe {
            augur_cuda_sys::augur_cuda_q4k_f32_decode_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_q4k_f32_decode_submit_v1", result)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQ4kF32PrefillArgsV1 {
    raw: AugurCudaQ4kF32PrefillArgsV1,
    device_ordinal: i32,
}

impl CudaQ4kF32PrefillArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: CudaQ4kWeightViewV1,
        input: CudaF32View,
        output: CudaF32View,
        m: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let ordinal = stream.device_ordinal();
        for view_ordinal in [
            weight.view.device_ordinal(),
            input.view.device_ordinal(),
            output.view.device_ordinal(),
        ] {
            if view_ordinal != ordinal {
                return Err(prefill_validation(CudaRuntimeErrorKind::DeviceMismatch));
            }
        }
        let input_elements = m
            .checked_mul(weight.cols() as usize)
            .ok_or_else(|| prefill_validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let output_elements = m
            .checked_mul(weight.rows() as usize)
            .ok_or_else(|| prefill_validation(CudaRuntimeErrorKind::RangeOverflow))?;
        if m == 0
            || input.element_count as usize != input_elements
            || (output.element_count as usize) < output_elements
        {
            return Err(prefill_validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let properties = cuda_device_properties(ordinal)?;
        if !properties
            .compute_capability
            .is_at_least(CudaComputeCapability::SM70)
        {
            return Err(prefill_validation(
                CudaRuntimeErrorKind::UnsupportedCapability,
            ));
        }
        let mut raw = AugurCudaQ4kF32PrefillArgsV1::new();
        raw.weight = addr_from_view_prefill(weight.view)?;
        raw.weight_len_bytes = to_u64_prefill(weight.view.len_bytes())?;
        raw.input = addr_from_view_prefill(input.view)?;
        raw.input_len_bytes = to_u64_prefill(input.view.len_bytes())?;
        raw.output = addr_from_view_prefill(output.view)?;
        raw.output_len_bytes = to_u64_prefill(output.view.len_bytes())?;
        raw.m = u32::try_from(m)
            .map_err(|_| prefill_validation(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.rows = weight.rows();
        raw.cols = weight.cols();
        raw.blocks_per_row = weight.blocks_per_row();
        raw.weight_block_stride_bytes = AUGUR_CUDA_Q4K_BLOCK_BYTES_V1;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn as_raw(&self) -> &AugurCudaQ4kF32PrefillArgsV1 {
        &self.raw
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        validate_launch_device(
            PREFILL_OPERATION,
            self.device_ordinal,
            stream.device_ordinal(),
        )?;
        // SAFETY: `&self.raw` is owned args populated by `bind`, which checked m x cols /
        // m x rows extents against the bounds-checked views (CudaQ4kWeightViewV1::new,
        // CudaF32View::new) on the ordinal re-checked by validate_launch_device.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_q4k_f32_prefill_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(PREFILL_OPERATION, result)
    }

    #[doc(hidden)]
    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        validate_launch_device(
            PREFILL_OPERATION,
            self.device_ordinal,
            stream.device_ordinal(),
        )?;
        // SAFETY: Same bind-established contract as `launch`; completion is asynchronous,
        // so the caller must keep the viewed OwnedCudaBuffers alive until the stream
        // drains (CudaBufferView carries no lifetime — see findings).
        let result = unsafe {
            augur_cuda_sys::augur_cuda_q4k_f32_prefill_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_q4k_f32_prefill_submit_v1", result)
    }
}

fn validate_launch_device(
    operation: &'static str,
    bound_ordinal: i32,
    stream_ordinal: i32,
) -> Result<(), CudaRuntimeError> {
    if bound_ordinal != stream_ordinal {
        return Err(CudaRuntimeError::validation(
            operation,
            CudaRuntimeErrorKind::DeviceMismatch,
        ));
    }
    Ok(())
}

fn addr_from_view(view: CudaBufferView) -> Result<u64, CudaRuntimeError> {
    view.device_ptr()
        .raw_value()
        .checked_add(
            u64::try_from(view.offset_bytes())
                .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?,
        )
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))
}

fn addr_from_view_prefill(view: CudaBufferView) -> Result<u64, CudaRuntimeError> {
    view.device_ptr()
        .raw_value()
        .checked_add(
            u64::try_from(view.offset_bytes())
                .map_err(|_| prefill_validation(CudaRuntimeErrorKind::RangeOverflow))?,
        )
        .ok_or_else(|| prefill_validation(CudaRuntimeErrorKind::RangeOverflow))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cuda_runtime::test_buffer_view, OwnedCudaBuffer};

    #[test]
    fn launch_device_mismatch_is_structured_without_cuda() {
        let error = validate_launch_device(OPERATION, 0, 1).unwrap_err();
        assert_eq!(error.operation, OPERATION);
        assert_eq!(error.kind, CudaRuntimeErrorKind::DeviceMismatch);
    }

    #[test]
    fn exact_q4k_layout_binds() {
        let view = test_buffer_view(0x1000, 0, 0, 160);
        let weight =
            CudaQ4kWeightViewV1::new(view, PhysicalLayoutId::Q4K_HOT_CUDA_V1, 1, 256).unwrap();
        assert_eq!(weight.rows(), 1);
        assert_eq!(weight.cols(), 256);
        assert_eq!(weight.blocks_per_row(), 1);
    }

    #[test]
    fn cpu_layout_is_rejected() {
        let error = CudaQ4kWeightViewV1::new(
            test_buffer_view(0x1000, 0, 0, 160),
            PhysicalLayoutId::Q4K_HOT_CPU_PORTABLE_V1,
            1,
            256,
        )
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::InvalidLayout);
    }

    #[test]
    fn columns_must_be_q4k_block_multiple() {
        let error = CudaQ4kWeightViewV1::new(
            test_buffer_view(0x1000, 0, 0, 160),
            PhysicalLayoutId::Q4K_HOT_CUDA_V1,
            1,
            255,
        )
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::InvalidDimensions);
    }

    #[test]
    fn weight_payload_length_must_be_exact() {
        let error = CudaQ4kWeightViewV1::new(
            test_buffer_view(0x1000, 0, 0, 159),
            PhysicalLayoutId::Q4K_HOT_CUDA_V1,
            1,
            256,
        )
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::OutOfBounds);
    }

    #[test]
    fn weight_address_must_be_32_byte_aligned() {
        let error = CudaQ4kWeightViewV1::new(
            test_buffer_view(0x1001, 0, 0, 160),
            PhysicalLayoutId::Q4K_HOT_CUDA_V1,
            1,
            256,
        )
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::MisalignedDeviceAddress);
    }

    #[test]
    fn f32_view_rejects_host_layout() {
        let error = CudaF32View::new(
            test_buffer_view(0x2000, 0, 0, 1024),
            PhysicalLayoutId::HOST_CONTIGUOUS_F32,
            256,
        )
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::InvalidLayout);
    }

    #[test]
    fn invalid_abi_version_rejected() {
        let stream = OwnedCudaStream::create(0).unwrap();
        let w = OwnedCudaBuffer::allocate(0, 160).unwrap();
        let inp = OwnedCudaBuffer::allocate(0, 256 * 4).unwrap();
        let out = OwnedCudaBuffer::allocate(0, 4).unwrap();

        let wv = CudaQ4kWeightViewV1::new(
            w.view(0, 160).unwrap(),
            PhysicalLayoutId::Q4K_HOT_CUDA_V1,
            1,
            256,
        )
        .unwrap();
        let iv = CudaF32View::new(
            inp.view(0, 256 * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            256,
        )
        .unwrap();
        let ov = CudaF32View::new(
            out.view(0, 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            1,
        )
        .unwrap();

        let args = CudaQ4kF32DecodeArgsV1::bind(&stream, wv, iv, ov).unwrap();
        let mut raw = *args.as_raw();
        raw.abi_version = 0;

        // SAFETY: All pointer/length fields still describe live buffers owned above;
        // only abi_version is corrupted, which the native entry must reject during
        // argument validation before dereferencing any device pointer.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_q4k_f32_decode_v1(
                stream.device_ordinal(),
                stream.raw().unwrap(),
                &raw,
            )
        };
        assert_ne!(result.status, 0, "invalid ABI version must be rejected");
    }
}
