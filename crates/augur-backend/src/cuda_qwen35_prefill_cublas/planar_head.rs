//! Explicit byte-exact code/scale-plane head. No implicit Q8_0 reinterpretation.
use super::*;
use augur_core::PhysicalLayoutId;

const ROWS: usize = 248_320;
const COLS: usize = 5_120;
const BYTES: usize = ROWS * COLS / 32 * 34;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaQwen35PlanarQ8HeadWeightV1(CudaBufferView);

impl CudaQwen35PlanarQ8HeadWeightV1 {
    pub fn new(
        view: CudaBufferView,
        layout: PhysicalLayoutId,
        rows: usize,
        cols: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if layout != PhysicalLayoutId::DEVICE_QWEN35_Q8_CODE_SCALE_PLANES_V1 {
            return Err(validation(CudaRuntimeErrorKind::InvalidLayout));
        }
        if rows != ROWS || cols != COLS {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if view.len_bytes() != BYTES {
            return Err(validation(CudaRuntimeErrorKind::OutOfBounds));
        }
        if view.checked_address(OPERATION)? % 4 != 0 {
            return Err(validation(CudaRuntimeErrorKind::MisalignedDeviceAddress));
        }
        view.checked_range(OPERATION)?;
        Ok(Self(view))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaQwen35PlanarQ8HeadProjectionV1 {
    raw: AugurCudaQwen35PrefillCublasProjectionArgsV1,
    ordinal: i32,
    stream_owner: usize,
}

fn bind_raw(
    ordinal: i32,
    weight: CudaQwen35PlanarQ8HeadWeightV1,
    input: CudaQ8_1ActivationViewV1,
    output: CudaF32View,
    tokens: usize,
) -> Result<AugurCudaQwen35PrefillCublasProjectionArgsV1, CudaRuntimeError> {
    if !(1..=8).contains(&tokens)
        || tokens.checked_mul(COLS) != Some(input.elements())
        || tokens.checked_mul(ROWS) != Some(output.element_count())
    {
        return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
    }
    validate_disjoint_views(
        ordinal,
        &[weight.0, input.buffer_view(), output.buffer_view()],
    )?;
    let mut raw = AugurCudaQwen35PrefillCublasProjectionArgsV1::new();
    raw.rows = ROWS as u32;
    raw.cols = COLS as u32;
    raw.tokens = tokens as u32;
    raw.weight_format = augur_cuda_sys::AUGUR_CUDA_QWEN35_PLANAR_Q8_HEAD_WEIGHT_V1;
    raw.tensor_scale = 1.0;
    bind_raw_view(&mut raw.weight, &mut raw.weight_len_bytes, weight.0)?;
    bind_raw_view(
        &mut raw.input_q8_1,
        &mut raw.input_len_bytes,
        input.buffer_view(),
    )?;
    bind_raw_view(
        &mut raw.output_f32,
        &mut raw.output_len_bytes,
        output.buffer_view(),
    )?;
    Ok(raw)
}

impl CudaQwen35PlanarQ8HeadProjectionV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: CudaQwen35PlanarQ8HeadWeightV1,
        input: CudaQ8_1ActivationViewV1,
        output: CudaF32View,
        tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let raw = bind_raw(stream.device_ordinal(), weight, input, output, tokens)?;
        // SAFETY: checked explicit layout, dimensions, same-device/disjoint views;
        // native preflight checks actual allocations and SM70 before any launch.
        let r = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_planar_q8_head_preflight_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, r)?;
        Ok(Self {
            raw,
            ordinal: stream.device_ordinal(),
            stream_owner: stream.owner_instance_key()?,
        })
    }

    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        if stream.owner_instance_key()? != self.stream_owner {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        // SAFETY: retained immutable owners on the exact preflighted stream.
        // Submit has no device allocation or alternate-format fallback.
        let r = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_planar_q8_head_prevalidated_submit_v1(
                self.ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_runtime::test_buffer_view;
    const LAYOUT: PhysicalLayoutId = PhysicalLayoutId::DEVICE_QWEN35_Q8_CODE_SCALE_PLANES_V1;
    fn weight(address: u64) -> CudaQwen35PlanarQ8HeadWeightV1 {
        CudaQwen35PlanarQ8HeadWeightV1::new(
            test_buffer_view(address, 0, 0, BYTES),
            LAYOUT,
            ROWS,
            COLS,
        )
        .unwrap()
    }
    #[test]
    fn planar_head_rejects_wire_layout_geometry_extent_and_alignment() {
        for layout in [
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        ] {
            assert!(CudaQwen35PlanarQ8HeadWeightV1::new(
                test_buffer_view(0x1000, 0, 0, BYTES),
                layout,
                ROWS,
                COLS
            )
            .is_err());
        }
        for (address, bytes, rows, cols) in [
            (0x1002, BYTES, ROWS, COLS),
            (0x1000, BYTES - 1, ROWS, COLS),
            (0x1000, BYTES, ROWS - 1, COLS),
            (0x1000, BYTES, ROWS, COLS - 1),
        ] {
            assert!(CudaQwen35PlanarQ8HeadWeightV1::new(
                test_buffer_view(address, 0, 0, bytes),
                LAYOUT,
                rows,
                cols
            )
            .is_err());
        }
    }
    #[test]
    fn planar_head_binds_one_to_eight_and_rejects_alias_or_device_drift() {
        for tokens in 1..=8 {
            let input = CudaQ8_1ActivationViewV1::new(
                test_buffer_view(0x9000_0000, 0, 0, tokens * COLS / 32 * 36),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,
                tokens * COLS,
            )
            .unwrap();
            let output = |address| {
                CudaF32View::new(
                    test_buffer_view(address, 0, 0, tokens * ROWS * 4),
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    tokens * ROWS,
                )
                .unwrap()
            };
            let raw = bind_raw(0, weight(0x1000), input, output(0xa000_0000), tokens).unwrap();
            assert_eq!(raw.weight_format, 4);
            for invalid in [0, 9, tokens + 1] {
                assert!(bind_raw(0, weight(0x1000), input, output(0xa000_0000), invalid).is_err());
            }
            assert!(bind_raw(0, weight(0x1000), input, output(0x1000), tokens).is_err());
            assert!(bind_raw(1, weight(0x1000), input, output(0xa000_0000), tokens).is_err());
        }
    }
    #[test]
    fn planar_head_native_null_rejects_without_cuda() {
        // SAFETY: null inputs must reject before touching CUDA.
        unsafe {
            assert!(CudaRuntimeError::from_result(
                OPERATION,
                augur_cuda_sys::augur_cuda_qwen35_planar_q8_head_preflight_v1(
                    0,
                    core::ptr::null_mut(),
                    core::ptr::null()
                )
            )
            .is_err());
            assert!(CudaRuntimeError::from_result(
                OPERATION,
                augur_cuda_sys::augur_cuda_qwen35_planar_q8_head_prevalidated_submit_v1(
                    0,
                    core::ptr::null_mut(),
                    core::ptr::null()
                )
            )
            .is_err());
        }
    }
}
