#![cfg(feature = "cuda")]

use augur_cuda_sys::{
    self, AugurCudaDflashGroupedConvF32ArgsV1, AugurCudaDflashGroupedConvHistoryF32ArgsV1,
    AugurCudaDflashGroupedConvHistoryRowsF32ArgsV2,
};

use crate::{CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};

const OPERATION: &str = "cuda_dflash_grouped_conv_f32_v1";

/// One-token Qwen35 DFlash grouped convolution.
///
/// The ABI carries the complete [side, tap, group]/[side, tap, hidden]
/// coefficient slabs, while this first boundary intentionally uses zero
/// previous history. It is therefore suitable for layer-0 CPU-oracle A/B,
/// not yet for rolling decode history.
#[derive(Clone, Copy, Debug)]
pub struct CudaDflashGroupedConvF32ArgsV1 {
    raw: AugurCudaDflashGroupedConvF32ArgsV1,
    device_ordinal: i32,
}

impl CudaDflashGroupedConvF32ArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        hidden: CudaF32View,
        dynamic: CudaF32View,
        base: CudaF32View,
        output: CudaF32View,
        groups: usize,
        group_size: usize,
        kernel_size: usize,
        side: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if hidden.element_count() == 0
            || hidden.element_count() != output.element_count()
            || groups == 0
            || group_size == 0
            || groups.checked_mul(group_size) != Some(hidden.element_count())
            || kernel_size == 0
            || side >= 2
            || dynamic.element_count() != 2 * kernel_size * groups
            || base.element_count() != 2 * kernel_size * hidden.element_count()
        {
            return Err(CudaRuntimeError::validation(
                OPERATION,
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        let ordinal = stream.device_ordinal();
        for view in [hidden, dynamic, base, output] {
            if view.buffer_view().device_ordinal() != ordinal {
                return Err(CudaRuntimeError::validation(
                    OPERATION,
                    CudaRuntimeErrorKind::DeviceMismatch,
                ));
            }
        }
        let views = [
            hidden.buffer_view(),
            dynamic.buffer_view(),
            base.buffer_view(),
            output.buffer_view(),
        ];
        for (index, view) in views.iter().enumerate() {
            for other in views.iter().skip(index + 1) {
                if view.overlaps(*other, OPERATION)? {
                    return Err(CudaRuntimeError::validation(
                        OPERATION,
                        CudaRuntimeErrorKind::OverlappingDeviceRanges,
                    ));
                }
            }
        }
        let mut raw = AugurCudaDflashGroupedConvF32ArgsV1::new();
        raw.hidden_f32 = address(hidden)?;
        raw.hidden_len_bytes = hidden.buffer_view().len_bytes() as u64;
        raw.dynamic_f32 = address(dynamic)?;
        raw.dynamic_len_bytes = dynamic.buffer_view().len_bytes() as u64;
        raw.base_f32 = address(base)?;
        raw.base_len_bytes = base.buffer_view().len_bytes() as u64;
        raw.output_f32 = address(output)?;
        raw.output_len_bytes = output.buffer_view().len_bytes() as u64;
        raw.hidden_elements = hidden.element_count() as u32;
        raw.groups = groups as u32;
        raw.group_size = group_size as u32;
        raw.kernel_size = kernel_size as u32;
        raw.side = side as u32;
        raw.reserved = 0;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        let result = unsafe {
            augur_cuda_sys::augur_cuda_dflash_grouped_conv_f32_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        let result = unsafe {
            augur_cuda_sys::augur_cuda_dflash_grouped_conv_f32_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_dflash_grouped_conv_f32_submit_v1", result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::validation(
                OPERATION,
                CudaRuntimeErrorKind::DeviceMismatch,
            ));
        }
        Ok(())
    }
}

fn address(view: CudaF32View) -> Result<u64, CudaRuntimeError> {
    view.buffer_view()
        .device_ptr()
        .raw_value()
        .checked_add(view.buffer_view().offset_bytes() as u64)
        .ok_or_else(|| CudaRuntimeError::validation(OPERATION, CudaRuntimeErrorKind::RangeOverflow))
}

#[derive(Clone, Copy, Debug)]
pub struct CudaDflashGroupedConvHistoryF32ArgsV1 {
    raw: AugurCudaDflashGroupedConvHistoryF32ArgsV1,
    device_ordinal: i32,
}

impl CudaDflashGroupedConvHistoryF32ArgsV1 {
    /// Submission only. The caller retains all owners through stream completion.
    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::validation(
                OPERATION,
                CudaRuntimeErrorKind::DeviceMismatch,
            ));
        }
        // SAFETY: bind checked disjoint ranges and geometry; the owning plan
        // retains hidden/history/dynamic/base/output through completion.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_dflash_grouped_conv_history_f32_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_dflash_grouped_conv_history_f32_submit_v1", result)
    }

    pub fn bind(
        stream: &OwnedCudaStream,
        hidden: CudaF32View,
        history: CudaF32View,
        dynamic: CudaF32View,
        base: CudaF32View,
        output: CudaF32View,
        groups: usize,
        group_size: usize,
        side: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let hidden_elements = hidden.element_count();
        if hidden_elements == 0
            || history.element_count() != hidden_elements
            || groups == 0
            || group_size == 0
            || groups.checked_mul(group_size) != Some(hidden_elements)
            || side >= 2
            || dynamic.element_count() != 4 * groups
            || base.element_count() != 4 * hidden_elements
        {
            return Err(CudaRuntimeError::validation(
                OPERATION,
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        let ordinal = stream.device_ordinal();
        let views = [
            hidden.buffer_view(),
            history.buffer_view(),
            dynamic.buffer_view(),
            base.buffer_view(),
            output.buffer_view(),
        ];
        for view in views {
            if view.device_ordinal() != ordinal {
                return Err(CudaRuntimeError::validation(
                    OPERATION,
                    CudaRuntimeErrorKind::DeviceMismatch,
                ));
            }
        }
        for (index, view) in views.iter().enumerate() {
            for other in views.iter().skip(index + 1) {
                if view.overlaps(*other, OPERATION)? {
                    return Err(CudaRuntimeError::validation(
                        OPERATION,
                        CudaRuntimeErrorKind::OverlappingDeviceRanges,
                    ));
                }
            }
        }
        let mut raw = AugurCudaDflashGroupedConvHistoryF32ArgsV1::new();
        raw.hidden_f32 = address(hidden)?;
        raw.hidden_len_bytes = hidden.buffer_view().len_bytes() as u64;
        raw.history_f32 = address(history)?;
        raw.history_len_bytes = history.buffer_view().len_bytes() as u64;
        raw.dynamic_f32 = address(dynamic)?;
        raw.dynamic_len_bytes = dynamic.buffer_view().len_bytes() as u64;
        raw.base_f32 = address(base)?;
        raw.base_len_bytes = base.buffer_view().len_bytes() as u64;
        raw.output_f32 = address(output)?;
        raw.output_len_bytes = output.buffer_view().len_bytes() as u64;
        raw.hidden_elements = hidden_elements as u32;
        raw.groups = groups as u32;
        raw.group_size = group_size as u32;
        raw.kernel_size = 2;
        raw.side = side as u32;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::validation(
                OPERATION,
                CudaRuntimeErrorKind::DeviceMismatch,
            ));
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_dflash_grouped_conv_history_f32_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }
}

/// Exact contiguous-row rolling convolution for a prepared DFlash block.
/// Row zero uses the explicit zero row as history; each following row reads
/// the immediately preceding input row. Dynamic coefficients are row-major,
/// while the immutable base slab is shared across rows.
#[derive(Clone, Copy, Debug)]
pub struct CudaDflashGroupedConvHistoryRowsF32ArgsV2 {
    raw: AugurCudaDflashGroupedConvHistoryRowsF32ArgsV2,
    device_ordinal: i32,
}

impl CudaDflashGroupedConvHistoryRowsF32ArgsV2 {
    pub fn bind(
        stream: &OwnedCudaStream,
        hidden: CudaF32View,
        zero: CudaF32View,
        dynamic: CudaF32View,
        base: CudaF32View,
        output: CudaF32View,
        rows: usize,
        groups: usize,
        group_size: usize,
        side: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if !(1..=8).contains(&rows)
            || groups == 0
            || group_size == 0
            || groups.checked_mul(group_size).is_none()
            || hidden.element_count() == 0
            || hidden.element_count() != output.element_count()
            || hidden.element_count() % rows != 0
            || groups.checked_mul(group_size) != Some(hidden.element_count() / rows)
            || zero.element_count() != hidden.element_count() / rows
            || dynamic.element_count() != rows * 4 * groups
            || base.element_count() != 4 * hidden.element_count() / rows
            || side >= 2
        {
            return Err(CudaRuntimeError::validation(
                OPERATION,
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        let ordinal = stream.device_ordinal();
        let views = [
            hidden.buffer_view(),
            zero.buffer_view(),
            dynamic.buffer_view(),
            base.buffer_view(),
            output.buffer_view(),
        ];
        if views.iter().any(|view| view.device_ordinal() != ordinal) {
            return Err(CudaRuntimeError::validation(
                OPERATION,
                CudaRuntimeErrorKind::DeviceMismatch,
            ));
        }
        for (index, view) in views.iter().enumerate() {
            for other in views.iter().skip(index + 1) {
                if view.overlaps(*other, OPERATION)? {
                    return Err(CudaRuntimeError::validation(
                        OPERATION,
                        CudaRuntimeErrorKind::OverlappingDeviceRanges,
                    ));
                }
            }
        }
        let row_elements = hidden.element_count() / rows;
        let mut raw = AugurCudaDflashGroupedConvHistoryRowsF32ArgsV2::new();
        raw.hidden_f32 = address(hidden)?;
        raw.hidden_len_bytes = hidden.buffer_view().len_bytes() as u64;
        raw.zero_f32 = address(zero)?;
        raw.zero_len_bytes = zero.buffer_view().len_bytes() as u64;
        raw.dynamic_f32 = address(dynamic)?;
        raw.dynamic_len_bytes = dynamic.buffer_view().len_bytes() as u64;
        raw.base_f32 = address(base)?;
        raw.base_len_bytes = base.buffer_view().len_bytes() as u64;
        raw.output_f32 = address(output)?;
        raw.output_len_bytes = output.buffer_view().len_bytes() as u64;
        raw.rows = rows as u32;
        raw.hidden_elements = row_elements as u32;
        raw.groups = groups as u32;
        raw.group_size = group_size as u32;
        raw.kernel_size = 2;
        raw.side = side as u32;
        raw.reserved = 0;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::validation(
                OPERATION,
                CudaRuntimeErrorKind::DeviceMismatch,
            ));
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_dflash_grouped_conv_history_rows_f32_submit_v2(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(
            "cuda_dflash_grouped_conv_history_rows_f32_submit_v2",
            result,
        )
    }
}
