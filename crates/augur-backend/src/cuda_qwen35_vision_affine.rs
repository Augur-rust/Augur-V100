//! Default-off direct FP16 affine binding, not a complete Vision forward.
//! Inputs are already packed FP16; finite/rounding admission is a separate
//! producer contract. No implicit Q8 conversion, weight decode or FP16 output.

use super::{CudaQwen35PrefillCublasPlanV1, OwnedCudaStream};
use crate::{CudaBufferView, CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind};
use augur_cuda_sys::AugurCudaQwen35VisionF16AffineArgsV1;

const OP: &str = "qwen35_vision_f16_affine";

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35VisionAffineViewsV1 {
    /// Row-major [output columns, input columns], two bytes per element.
    pub weight_f16: CudaBufferView,
    /// Row-major [tokens, input columns], two bytes per element.
    pub input_f16: CudaBufferView,
    pub bias: CudaF32View,
    pub output: CudaF32View,
}

#[derive(Debug)]
pub struct CudaQwen35VisionF16AffineV1 {
    raw: AugurCudaQwen35VisionF16AffineArgsV1,
    ordinal: i32,
    stream_owner: usize,
    plan_owner: usize,
}

impl CudaQwen35PrefillCublasPlanV1 {
    /// Reuse the checked stream-owned handle. The two legacy workspaces are
    /// minimally sized (2 + max_tokens*128 bytes) and never touched by this
    /// operation. This is not total cuBLAS/driver workspace accounting.
    pub fn new_for_resident_vision_affine(
        stream: &OwnedCudaStream,
        max_tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        Self::new(stream, 1, 64, max_tokens)
    }

    pub fn bind_vision_f16_affine(
        &self,
        stream: &OwnedCudaStream,
        views: CudaQwen35VisionAffineViewsV1,
        tokens: usize,
        rows: usize,
        cols: usize,
    ) -> Result<CudaQwen35VisionF16AffineV1, CudaRuntimeError> {
        self.require_stream(stream)?;
        let raw = bind_raw(
            self.device_ordinal,
            self.max_tokens,
            views,
            tokens,
            rows,
            cols,
        )?;
        // SAFETY: CPU binding checks complete extents, alignment and aliases;
        // native preflight also checks actual device allocation membership.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_vision_f16_affine_preflight_v1(
                self.device_ordinal,
                stream.raw()?,
                self.raw,
                &raw,
            )
        };
        CudaRuntimeError::from_result(OP, result)?;
        Ok(CudaQwen35VisionF16AffineV1 {
            raw,
            ordinal: self.device_ordinal,
            stream_owner: self.stream_owner_instance_key,
            plan_owner: self.raw as usize,
        })
    }

    /// Enqueue only. As with other backend views, callers retain the plan,
    /// stream and all four buffer owners through completion and validate the
    /// output before it becomes an authoritative language-model input.
    pub fn submit_vision_f16_affine(
        &self,
        stream: &OwnedCudaStream,
        operation: &CudaQwen35VisionF16AffineV1,
    ) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        if operation.ordinal != self.device_ordinal
            || operation.stream_owner != self.stream_owner_instance_key
            || operation.plan_owner != self.raw as usize
        {
            return Err(error(CudaRuntimeErrorKind::InvalidMetadata));
        }
        // SAFETY: operation is immutable, preflighted and bound to this exact
        // plan/stream. Input and output owners are retained by the caller.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_vision_f16_affine_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                self.raw,
                &operation.raw,
            )
        };
        CudaRuntimeError::from_result(OP, result)
    }
}

fn error(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OP, kind)
}

fn bind_raw(
    ordinal: i32,
    max_tokens: usize,
    views: CudaQwen35VisionAffineViewsV1,
    tokens: usize,
    rows: usize,
    cols: usize,
) -> Result<AugurCudaQwen35VisionF16AffineArgsV1, CudaRuntimeError> {
    if ordinal < 0
        || tokens > max_tokens
        || [tokens, rows, cols]
            .iter()
            .any(|&n| n == 0 || n > i32::MAX as usize)
    {
        return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
    }
    let bytes = |a: usize, b: usize, width: usize| {
        a.checked_mul(b)
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))
    };
    let expected = [
        bytes(rows, cols, 2)?,
        bytes(tokens, cols, 2)?,
        bytes(rows, 1, 4)?,
        bytes(tokens, rows, 4)?,
    ];
    let buffers = [
        views.weight_f16,
        views.input_f16,
        views.bias.buffer_view(),
        views.output.buffer_view(),
    ];
    let mut addresses = [0; 4];
    for (index, view) in buffers.iter().enumerate() {
        if view.device_ordinal() != ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let alignment = if index < 2 { 2 } else { 4 };
        let address = view.checked_address(OP)?;
        if view.len_bytes() != expected[index] || address == 0 || address % alignment != 0 {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        view.checked_range(OP)?;
        for prior in &buffers[..index] {
            if view.overlaps(*prior, OP)? {
                return Err(error(CudaRuntimeErrorKind::OverlappingDeviceRanges));
            }
        }
        addresses[index] = address;
    }
    Ok(AugurCudaQwen35VisionF16AffineArgsV1 {
        abi_version: 1,
        struct_size: core::mem::size_of::<AugurCudaQwen35VisionF16AffineArgsV1>() as u32,
        tokens: tokens as u32,
        rows: rows as u32,
        cols: cols as u32,
        reserved: 0,
        weight_f16: addresses[0],
        weight_len_bytes: expected[0] as u64,
        input_f16: addresses[1],
        input_len_bytes: expected[1] as u64,
        bias_f32: addresses[2],
        bias_len_bytes: expected[2] as u64,
        output_f32: addresses[3],
        output_len_bytes: expected[3] as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_runtime::test_buffer_view;
    use augur_core::PhysicalLayoutId;

    const SHAPES: [(usize, usize); 7] = [
        (1152, 1536),
        (3456, 1152),
        (1152, 1152),
        (4304, 1152),
        (1152, 4304),
        (4608, 4608),
        (5120, 4608),
    ];

    fn f32_view(view: CudaBufferView) -> CudaF32View {
        CudaF32View::new(
            view,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            view.len_bytes() / 4,
        )
        .unwrap()
    }

    fn views(tokens: usize, rows: usize, cols: usize) -> CudaQwen35VisionAffineViewsV1 {
        CudaQwen35VisionAffineViewsV1 {
            weight_f16: test_buffer_view(0x1000_0000, 0, 0, rows * cols * 2),
            input_f16: test_buffer_view(0x2000_0000, 0, 0, tokens * cols * 2),
            bias: f32_view(test_buffer_view(0x3000_0000, 0, 0, rows * 4)),
            output: f32_view(test_buffer_view(0x4000_0000, 0, 0, tokens * rows * 4)),
        }
    }

    #[test]
    fn vision_f16_affine_abi_and_all_real_geometries() {
        type Raw = AugurCudaQwen35VisionF16AffineArgsV1;
        assert_eq!(core::mem::size_of::<Raw>(), 88);
        assert_eq!(core::mem::align_of::<Raw>(), 8);
        assert_eq!(core::mem::offset_of!(Raw, weight_f16), 24);
        assert_eq!(core::mem::offset_of!(Raw, output_f32), 72);
        for (rows, cols) in SHAPES {
            for tokens in [1, 64, 256, 1024] {
                let raw = bind_raw(0, 1024, views(tokens, rows, cols), tokens, rows, cols).unwrap();
                assert_eq!(raw.weight_len_bytes, (rows * cols * 2) as u64);
                assert_eq!(raw.output_len_bytes, (tokens * rows * 4) as u64);
                assert_eq!(raw.cols as usize, cols);
            }
        }
    }

    #[test]
    fn vision_f16_affine_rejects_shape_extent_and_scalar_type_drift() {
        let good = views(2, 1152, 4304);
        for (tokens, rows, cols) in [
            (0, 1152, 4304),
            (3, 1152, 4304),
            (2, 0, 4304),
            (2, 1152, 0),
            (2, 1152, 4305),
            (2, usize::MAX, 4304),
        ] {
            assert!(bind_raw(0, 2, good, tokens, rows, cols).is_err());
        }
        let mut bad = good;
        bad.input_f16 = test_buffer_view(0x2000_0000, 0, 0, 2 * 4304 * 4);
        assert!(bind_raw(0, 2, bad, 2, 1152, 4304).is_err());
        bad = good;
        bad.weight_f16 = test_buffer_view(0x1000_0000, 0, 0, 1152 * 4304 * 2 - 2);
        assert!(bind_raw(0, 2, bad, 2, 1152, 4304).is_err());
        bad = good;
        bad.bias = f32_view(test_buffer_view(0x3000_0000, 0, 0, 1153 * 4));
        assert!(bind_raw(0, 2, bad, 2, 1152, 4304).is_err());
    }

    #[test]
    fn vision_f16_affine_rejects_alias_device_alignment_and_pointer_overflow() {
        let good = views(2, 1152, 4304);
        for input in [
            test_buffer_view(0x1000_0000, 0, 0, 2 * 4304 * 2),
            test_buffer_view(0x2000_0000, 1, 0, 2 * 4304 * 2),
            test_buffer_view(0x2000_0001, 0, 0, 2 * 4304 * 2),
            test_buffer_view(u64::MAX - 1, 0, 0, 2 * 4304 * 2),
            test_buffer_view(0, 0, 0, 2 * 4304 * 2),
        ] {
            let mut bad = good;
            bad.input_f16 = input;
            assert!(bind_raw(0, 2, bad, 2, 1152, 4304).is_err());
        }
        assert!(bind_raw(1, 2, good, 2, 1152, 4304).is_err());
        assert!(bind_raw(-1, 2, good, 2, 1152, 4304).is_err());
    }

    #[test]
    fn vision_f16_affine_rejects_every_buffer_alias_pair() {
        let bases = [0x1000_0000, 0x2000_0000, 0x3000_0000, 0x4000_0000];
        for a in 0..4 {
            for b in 0..4 {
                if a == b {
                    continue;
                }
                let mut v = views(2, 1152, 4304);
                let original = [
                    v.weight_f16,
                    v.input_f16,
                    v.bias.buffer_view(),
                    v.output.buffer_view(),
                ][a];
                let alias = test_buffer_view(bases[b], 0, 0, original.len_bytes());
                match a {
                    0 => v.weight_f16 = alias,
                    1 => v.input_f16 = alias,
                    2 => v.bias = f32_view(alias),
                    _ => v.output = f32_view(alias),
                }
                assert!(bind_raw(0, 2, v, 2, 1152, 4304).is_err());
            }
        }
    }

    #[test]
    #[ignore = "requires an idle SM70 GPU; never run alongside the native ladder/HTTP matrix"]
    fn vision_f16_affine_gpu_all_real_shapes_and_exact_plan_binding() {
        use crate::{cuda_device_allocation_snapshot, OwnedCudaBuffer};
        let baseline = cuda_device_allocation_snapshot();
        let mut stream = OwnedCudaStream::create(0).unwrap();
        let mut foreign_stream = OwnedCudaStream::create(0).unwrap();
        let plan =
            CudaQwen35PrefillCublasPlanV1::new_for_resident_vision_affine(&stream, 4).unwrap();
        let other =
            CudaQwen35PrefillCublasPlanV1::new_for_resident_vision_affine(&stream, 4).unwrap();
        for (rows, cols) in SHAPES {
            let upload = |raw: &[u8]| {
                let b = OwnedCudaBuffer::allocate(0, raw.len()).unwrap();
                b.copy_from_host_sync(&stream, raw).unwrap();
                b
            };
            // Signed, nonuniform values vary along row, column and token axes;
            // an all-ones oracle would hide transpose/stride/output-order bugs.
            let half = |v: i32| -> [u8; 2] {
                [0xc200u16, 0xc000, 0xbc00, 0, 0x3c00, 0x4000, 0x4200][(v + 3) as usize]
                    .to_le_bytes()
            };
            let row_factor = |r: usize| (1 + (r % 3) as i32) * if r % 2 == 0 { 1 } else { -1 };
            let weight_raw = (0..rows * cols)
                .flat_map(|i| half(row_factor(i / cols) * ((i % cols % 3) as i32 - 1)))
                .collect::<Vec<_>>();
            let input_raw = (0..4 * cols)
                .flat_map(|i| half(((i / cols + i % cols) % 5) as i32 - 2))
                .collect::<Vec<_>>();
            let weight = upload(&weight_raw);
            let input = upload(&input_raw);
            let bias_values = (0..rows).map(|i| (i % 7) as f32 / 8.0).collect::<Vec<_>>();
            let bias = upload(
                &bias_values
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            let output = OwnedCudaBuffer::allocate(0, 4 * rows * 4).unwrap();
            let v = CudaQwen35VisionAffineViewsV1 {
                weight_f16: weight.view(0, weight.len_bytes()).unwrap(),
                input_f16: input.view(0, input.len_bytes()).unwrap(),
                bias: f32_view(bias.view(0, bias.len_bytes()).unwrap()),
                output: f32_view(output.view(0, output.len_bytes()).unwrap()),
            };
            let op = plan
                .bind_vision_f16_affine(&stream, v, 4, rows, cols)
                .unwrap();
            assert!(other.submit_vision_f16_affine(&stream, &op).is_err());
            assert!(plan.submit_vision_f16_affine(&foreign_stream, &op).is_err());
            plan.submit_vision_f16_affine(&stream, &op).unwrap();
            stream.synchronize().unwrap();
            let mut raw = vec![0u8; output.len_bytes()];
            output.copy_to_host_sync(&stream, &mut raw).unwrap();
            let sums = (0..4)
                .map(|t| {
                    (0..cols)
                        .map(|c| (c % 3) as i32 - 1)
                        .enumerate()
                        .map(|(c, w)| w * (((t + c) % 5) as i32 - 2))
                        .sum::<i32>()
                })
                .collect::<Vec<_>>();
            for (i, bytes) in raw.chunks_exact(4).enumerate() {
                assert_eq!(
                    f32::from_le_bytes(bytes.try_into().unwrap()),
                    (sums[i / rows] * row_factor(i % rows)) as f32 + bias_values[i % rows]
                );
            }
            let mut bad = op.raw;
            bad.reserved = 1;
            // SAFETY: valid retained owners/plan/stream, deliberately invalid ABI metadata.
            let result = unsafe {
                augur_cuda_sys::augur_cuda_qwen35_vision_f16_affine_preflight_v1(
                    0,
                    stream.raw().unwrap(),
                    plan.raw,
                    &bad,
                )
            };
            assert!(CudaRuntimeError::from_result(OP, result).is_err());
        }
        drop(other);
        drop(plan);
        foreign_stream.close().unwrap();
        stream.close().unwrap();
        let after = cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes, baseline.live_bytes);
        assert_eq!(after.live_allocations, baseline.live_allocations);
        assert_eq!(
            after.successful_allocations - baseline.successful_allocations,
            after.successful_frees - baseline.successful_frees
        );
    }
}
