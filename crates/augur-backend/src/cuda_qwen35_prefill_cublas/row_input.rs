//! Row-local immutable Q8 input weights, explicitly distinct from wire and head.
use super::*;
use augur_core::PhysicalLayoutId;
pub(super) const FORMAT: u32 = augur_cuda_sys::AUGUR_CUDA_QWEN35_ROW_Q8_INPUT_WEIGHT_V1;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaQwen35RowQ8InputWeightV1 {
    pub(super) view: CudaBufferView,
    pub(super) rows: usize,
}
impl CudaQwen35RowQ8InputWeightV1 {
    pub fn new(
        view: CudaBufferView,
        layout: PhysicalLayoutId,
        rows: usize,
        cols: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if layout != PhysicalLayoutId::DEVICE_QWEN35_Q8_ROW_CODE_SCALE_PLANES_V1 {
            return Err(validation(CudaRuntimeErrorKind::InvalidLayout));
        }
        if cols != 5120 || !matches!(rows, 1024 | 6144 | 10240 | 12288) {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if view.len_bytes() != rows * 5440 {
            return Err(validation(CudaRuntimeErrorKind::OutOfBounds));
        }
        if view.checked_address(OPERATION)? % 4 != 0 {
            return Err(validation(CudaRuntimeErrorKind::MisalignedDeviceAddress));
        }
        view.checked_range(OPERATION)?;
        Ok(Self { view, rows })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaQwen35RowQ8InputProjectionV1 {
    raw: AugurCudaQwen35PrefillCublasProjectionArgsV1,
    ordinal: i32,
    owner: usize,
}
impl CudaQwen35RowQ8InputProjectionV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: CudaQwen35RowQ8InputWeightV1,
        input: CudaQ8_1ActivationViewV1,
        output: CudaF32View,
        tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let raw = bind_packed_projection_raw(
            stream.device_ordinal(),
            CudaQwen35PrefillWeightV1::RowQ8Input(weight),
            input,
            output,
            tokens,
        )?;
        // SAFETY: explicitly typed disjoint views and retained owners; native checks live device ranges.
        CudaRuntimeError::from_result(OPERATION, unsafe {
            augur_cuda_sys::augur_cuda_qwen35_row_q8_input_packed_preflight_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &raw,
            )
        })?;
        Ok(Self {
            raw,
            ordinal: stream.device_ordinal(),
            owner: stream.owner_instance_key()?,
        })
    }
    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        if stream.owner_instance_key()? != self.owner {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        // SAFETY: same preflighted owner and immutable views. No allocation or conversion.
        CudaRuntimeError::from_result(OPERATION, unsafe {
            augur_cuda_sys::augur_cuda_qwen35_row_q8_input_packed_submit_v1(
                self.ordinal,
                stream.raw()?,
                &self.raw,
            )
        })
    }
}

impl CudaQwen35PrefillCublasPlanV1 {
    pub(super) fn submit_row_input(
        &self,
        stream: &OwnedCudaStream,
        p: &CudaQwen35PrefillCublasProjectionV1,
    ) -> Result<(), CudaRuntimeError> {
        if p.weight_decode_lanes != 1
            || p.canonical_tiled_rows
            || p.canonical_grid_rows
            || p.tensor_ffn_down
            || p.tensor_ffn_down_compensated
            || p.integer_mma_ffn
            || p.integer_mma_input
            || p.llama_f16_input
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        if p.llama_full_original {
            if p.quantized_small_batch || p.exact_projection_tiles || p.projection_row_reuse {
                return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
            }
            let raw = p
                .original_input
                .as_ref()
                .ok_or_else(|| validation(CudaRuntimeErrorKind::InvalidMetadata))?;
            // SAFETY: caller required exact plan/stream binding; original-input preflight checked scratch and all owners.
            return CudaRuntimeError::from_result(OPERATION, unsafe {
                augur_cuda_sys::augur_cuda_qwen35_row_q8_input_original_submit_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    self.raw,
                    raw,
                )
            });
        }
        if p.original_input.is_some()
            || !p.quantized_small_batch
            || !(1..=8).contains(&p.raw.tokens)
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        // SAFETY: initial native binding checked the immutable same-device views; submit rejects wide placeholders.
        CudaRuntimeError::from_result(OPERATION, unsafe {
            augur_cuda_sys::augur_cuda_qwen35_row_q8_input_packed_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &p.raw,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_runtime::test_buffer_view;
    #[test]
    fn row_input_layout_is_closed_and_not_wire_or_head() {
        let layout = PhysicalLayoutId::DEVICE_QWEN35_Q8_ROW_CODE_SCALE_PLANES_V1;
        for rows in [1024, 6144, 10240, 12288] {
            let view = test_buffer_view(0x1000, 0, 0, rows * 5440);
            let w = CudaQwen35RowQ8InputWeightV1::new(view, layout, rows, 5120).unwrap();
            assert_eq!(CudaQwen35PrefillWeightV1::RowQ8Input(w).format(), 5);
            for bad in [
                PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
                PhysicalLayoutId::DEVICE_QWEN35_Q8_CODE_SCALE_PLANES_V1,
            ] {
                assert!(CudaQwen35RowQ8InputWeightV1::new(view, bad, rows, 5120).is_err());
            }
            for (addr, bytes, r, c) in [
                (0x1002, rows * 5440, rows, 5120),
                (0x1000, rows * 5440 - 1, rows, 5120),
                (0x1000, rows * 5440, 17408, 5120),
                (0x1000, rows * 5440, rows, 6144),
            ] {
                assert!(CudaQwen35RowQ8InputWeightV1::new(
                    test_buffer_view(addr, 0, 0, bytes),
                    layout,
                    r,
                    c
                )
                .is_err());
            }
        }
    }
    #[test]
    fn row_input_packed_binding_rejects_alias_width_and_device_drift() {
        let rows = 1024;
        let w = CudaQwen35RowQ8InputWeightV1::new(
            test_buffer_view(0x1000, 0, 0, rows * 5440),
            PhysicalLayoutId::DEVICE_QWEN35_Q8_ROW_CODE_SCALE_PLANES_V1,
            rows,
            5120,
        )
        .unwrap();
        for tokens in 1..=8 {
            let x = CudaQ8_1ActivationViewV1::new(
                test_buffer_view(0x1000_0000, 0, 0, tokens * 5760),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,
                tokens * 5120,
            )
            .unwrap();
            let y = |address| {
                CudaF32View::new(
                    test_buffer_view(address, 0, 0, tokens * rows * 4),
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    tokens * rows,
                )
                .unwrap()
            };
            let weight = CudaQwen35PrefillWeightV1::RowQ8Input(w);
            assert_eq!(
                bind_packed_projection_raw(0, weight, x, y(0x2000_0000), tokens)
                    .unwrap()
                    .weight_format,
                5
            );
            for bad in [0, 9, tokens + 1] {
                assert!(bind_packed_projection_raw(0, weight, x, y(0x2000_0000), bad).is_err());
            }
            assert!(bind_packed_projection_raw(0, weight, x, y(0x1000), tokens).is_err());
            assert!(bind_packed_projection_raw(1, weight, x, y(0x2000_0000), tokens).is_err());
        }
    }
}
