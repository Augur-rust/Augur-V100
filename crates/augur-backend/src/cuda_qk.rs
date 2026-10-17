#![cfg(feature = "cuda")]

use augur_core::PhysicalLayoutId;

use augur_cuda_sys::{
    self, AugurCudaQ5kF32DecodeArgsV1, AugurCudaQ6kF32DecodeArgsV1, AUGUR_CUDA_Q5K_BLOCK_ALIGNMENT_V1,
    AUGUR_CUDA_Q5K_BLOCK_BYTES_V1, AUGUR_CUDA_Q6K_BLOCK_ALIGNMENT_V1, AUGUR_CUDA_Q6K_BLOCK_BYTES_V1,
};






use crate::{
    CudaBufferView, CudaComputeCapability, CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind,
    OwnedCudaStream,
};

fn validation(operation: &'static str, kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(operation, kind)
}

fn address(operation: &'static str, view: CudaBufferView) -> Result<u64, CudaRuntimeError> {
    view.device_ptr()
        .raw_value()
        .checked_add(
            u64::try_from(view.offset_bytes())
                .map_err(|_| validation(operation, CudaRuntimeErrorKind::RangeOverflow))?,
        )
        .ok_or_else(|| validation(operation, CudaRuntimeErrorKind::RangeOverflow))
}

macro_rules! qk_weight {
    ($name:ident, $layout:expr, $block_bytes:expr, $alignment:expr, $operation:expr) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub struct $name {
            view: CudaBufferView,
            rows: u32,
            cols: u32,
            blocks_per_row: u32,
        }
        impl $name {
            pub fn new(
                view: CudaBufferView,
                layout: PhysicalLayoutId,
                rows: usize,
                cols: usize,
            ) -> Result<Self, CudaRuntimeError> {
                if layout != $layout || rows == 0 || cols == 0 || !cols.is_multiple_of(256) {
                    return Err(validation($operation, CudaRuntimeErrorKind::InvalidLayout));
                }
                let blocks_per_row = cols / 256;
                let expected = rows
                    .checked_mul(blocks_per_row)
                    .and_then(|blocks| blocks.checked_mul($block_bytes as usize))
                    .ok_or_else(|| validation($operation, CudaRuntimeErrorKind::RangeOverflow))?;
                if view.len_bytes() != expected {
                    return Err(validation($operation, CudaRuntimeErrorKind::OutOfBounds));
                }
                if address($operation, view)? % u64::from($alignment) != 0 {
                    return Err(validation(
                        $operation,
                        CudaRuntimeErrorKind::MisalignedDeviceAddress,
                    ));
                }
                Ok(Self {
                    view,
                    rows: rows.try_into().map_err(|_| {
                        validation($operation, CudaRuntimeErrorKind::InvalidDimensions)
                    })?,
                    cols: cols.try_into().map_err(|_| {
                        validation($operation, CudaRuntimeErrorKind::InvalidDimensions)
                    })?,
                    blocks_per_row: blocks_per_row.try_into().map_err(|_| {
                        validation($operation, CudaRuntimeErrorKind::InvalidDimensions)
                    })?,
                })
            }
            pub const fn rows(self) -> u32 {
                self.rows
            }
            pub const fn cols(self) -> u32 {
                self.cols
            }
        }
    };
}

qk_weight!(
    CudaQ5kWeightViewV1,
    PhysicalLayoutId::Q5K_HOT_CUDA_V1,
    AUGUR_CUDA_Q5K_BLOCK_BYTES_V1,
    AUGUR_CUDA_Q5K_BLOCK_ALIGNMENT_V1,
    "cuda_q5k_f32_decode_v1"
);
qk_weight!(
    CudaQ6kWeightViewV1,
    PhysicalLayoutId::Q6K_HOT_CUDA_V1,
    AUGUR_CUDA_Q6K_BLOCK_BYTES_V1,
    AUGUR_CUDA_Q6K_BLOCK_ALIGNMENT_V1,
    "cuda_q6k_f32_decode_v1"
);

impl CudaQ5kWeightViewV1 {
    /// Raw immutable storage view used by audited composite-weight assembly.
    pub const fn buffer_view(self) -> CudaBufferView {
        self.view
    }
}

impl CudaQ6kWeightViewV1 {
    pub(crate) const fn buffer_view(self) -> CudaBufferView {
        self.view
    }
}





































macro_rules! qk_args {
    ($name:ident, $weight:ident, $raw:ident, $block_bytes:expr, $ffi:path, $submit:path, $operation:expr) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $name {
            raw: $raw,
            device_ordinal: i32,
        }
        impl $name {
            pub fn bind(
                stream: &OwnedCudaStream,
                weight: $weight,
                input: CudaF32View,
                output: CudaF32View,
            ) -> Result<Self, CudaRuntimeError> {
                let ordinal = stream.device_ordinal();
                for resource in [
                    weight.view.device_ordinal(),
                    input.buffer_view().device_ordinal(),
                    output.buffer_view().device_ordinal(),
                ] {
                    if resource != ordinal {
                        return Err(validation($operation, CudaRuntimeErrorKind::DeviceMismatch));
                    }
                }
                if input.element_count() != weight.cols as usize
                    || output.element_count() < weight.rows as usize
                {
                    return Err(validation(
                        $operation,
                        CudaRuntimeErrorKind::InvalidDimensions,
                    ));
                }
                if !stream
                    .compute_capability_receipt()?
                    .is_at_least(CudaComputeCapability::SM70)
                {
                    return Err(validation(
                        $operation,
                        CudaRuntimeErrorKind::UnsupportedCapability,
                    ));
                }
                let mut raw = <$raw>::new();
                raw.weight = address($operation, weight.view)?;
                raw.weight_len_bytes = weight.view.len_bytes() as u64;
                raw.input = address($operation, input.buffer_view())?;
                raw.input_len_bytes = input.buffer_view().len_bytes() as u64;
                raw.output = address($operation, output.buffer_view())?;
                raw.output_len_bytes = output.buffer_view().len_bytes() as u64;
                raw.rows = weight.rows;
                raw.cols = weight.cols;
                raw.blocks_per_row = weight.blocks_per_row;
                raw.weight_block_stride_bytes = $block_bytes;
                Ok(Self {
                    raw,
                    device_ordinal: ordinal,
                })
            }
            pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
                if stream.device_ordinal() != self.device_ordinal {
                    return Err(validation($operation, CudaRuntimeErrorKind::DeviceMismatch));
                }
                // SAFETY: `&self.raw` is owned args built by `bind` (weight extent and
                // alignment proven by the qk_weight! constructor; dims/device/capability
                // checked in `bind`); the ordinal guard re-pins the bound device.
                let result = unsafe { $ffi(stream.device_ordinal(), stream.raw()?, &self.raw) };
                CudaRuntimeError::from_result($operation, result)
            }
            #[doc(hidden)]
            pub fn submit_for_plan(
                &self,
                stream: &OwnedCudaStream,
            ) -> Result<(), CudaRuntimeError> {
                if stream.device_ordinal() != self.device_ordinal {
                    return Err(validation($operation, CudaRuntimeErrorKind::DeviceMismatch));
                }
                // SAFETY: Same bind-established contract as `launch`; completion is
                // asynchronous, so the caller must keep the viewed buffers alive until
                // the stream drains (CudaBufferView carries no lifetime — see findings).
                let result = unsafe { $submit(stream.device_ordinal(), stream.raw()?, &self.raw) };
                CudaRuntimeError::from_result($operation, result)
            }
            pub const fn as_raw(&self) -> &$raw {
                &self.raw
            }
        }
    };
}

qk_args!(
    CudaQ5kF32DecodeArgsV1,
    CudaQ5kWeightViewV1,
    AugurCudaQ5kF32DecodeArgsV1,
    AUGUR_CUDA_Q5K_BLOCK_BYTES_V1,
    augur_cuda_sys::augur_cuda_q5k_f32_decode_v1,
    augur_cuda_sys::augur_cuda_q5k_f32_decode_submit_v1,
    "cuda_q5k_f32_decode_v1"
);
qk_args!(
    CudaQ5kF32DecodeGrid32ArgsV2,
    CudaQ5kWeightViewV1,
    AugurCudaQ5kF32DecodeArgsV1,
    AUGUR_CUDA_Q5K_BLOCK_BYTES_V1,
    augur_cuda_sys::augur_cuda_q5k_f32_decode_grid32_v2,
    augur_cuda_sys::augur_cuda_q5k_f32_decode_grid32_submit_v2,
    "cuda_q5k_f32_decode_grid32_v2"
);


























qk_args!(
    CudaQ6kF32DecodeArgsV1,
    CudaQ6kWeightViewV1,
    AugurCudaQ6kF32DecodeArgsV1,
    AUGUR_CUDA_Q6K_BLOCK_BYTES_V1,
    augur_cuda_sys::augur_cuda_q6k_f32_decode_v1,
    augur_cuda_sys::augur_cuda_q6k_f32_decode_submit_v1,
    "cuda_q6k_f32_decode_v1"
);


