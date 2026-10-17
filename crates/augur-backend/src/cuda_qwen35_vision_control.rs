//! Stream-bound Vision control stages; callers retain all owners through completion.
use crate::{CudaBufferView, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};
use augur_cuda_sys::AugurCudaQwen35VisionControlArgsV1;
const OP: &str = "qwen35_vision_control";

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaQwen35VisionControlOp {
    PositionAdd = 1,
    LayerNormToF16 = 2,
    CastToF16 = 3,
    GeluTanhToF16 = 4,
    GeluErfToF16 = 5,
    ResidualAdd = 6,
    RotaryQkInPlace = 7,
    OnlineAttention = 8,
}
#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35VisionControlViews {
    pub inputs: [Option<CudaBufferView>; 5],
    pub output: CudaBufferView,
    /// Sticky uint32 error flag, initialized by the owner and checked before publication.
    pub status: CudaBufferView,
}
#[derive(Debug)]
pub struct CudaQwen35VisionControlV1 {
    raw: AugurCudaQwen35VisionControlArgsV1,
    ordinal: i32,
    stream_owner: usize,
}
impl CudaQwen35VisionControlV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        op: CudaQwen35VisionControlOp,
        tokens: usize,
        cols: usize,
        views: CudaQwen35VisionControlViews,
    ) -> Result<Self, CudaRuntimeError> {
        let raw = bind_raw(stream.device_ordinal(), op, tokens, cols, views)?;
        // SAFETY: exact typed footprints and alias policy were checked; native preflight
        // verifies current allocation membership. Owners remain the executor's responsibility.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_vision_control_preflight_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &raw,
            )
        };
        CudaRuntimeError::from_result(OP, result)?;
        Ok(Self {
            raw,
            ordinal: stream.device_ordinal(),
            stream_owner: stream.owner_instance_key()?,
        })
    }
    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if self.ordinal != stream.device_ordinal()
            || self.stream_owner != stream.owner_instance_key()?
        {
            return Err(error(CudaRuntimeErrorKind::InvalidMetadata));
        }
        // SAFETY: immutable preflighted operation, exact stream instance and caller-retained owners.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_vision_control_submit_v1(
                self.ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OP, result)
    }
}
fn error(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OP, kind)
}
fn shape(
    op: CudaQwen35VisionControlOp,
    n: usize,
    c: usize,
) -> Result<([usize; 6], bool, bool), CudaRuntimeError> {
    use CudaQwen35VisionControlOp::*;
    if n == 0 || n > 4096 {
        return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
    }
    let result = match (op, c) {
        (PositionAdd, 1152) => (
            [n * c * 4, 2304 * c * 4, n * 16, n * 16, 0, n * c * 4],
            false,
            true,
        ),
        (LayerNormToF16, 1152) => ([n * c * 4, c * 4, c * 4, 0, 0, n * c * 2], true, false),
        (CastToF16, 1152 | 4608) | (GeluTanhToF16, 4304) | (GeluErfToF16, 4608) => {
            ([n * c * 4, 0, 0, 0, 0, n * c * 2], true, false)
        }
        (ResidualAdd, 1152) => ([n * c * 4, n * c * 4, 0, 0, 0, n * c * 4], false, true),
        (RotaryQkInPlace, 72) => (
            [n * 3456 * 4, n * 72 * 4, n * 72 * 4, 0, 0, n * 3456 * 4],
            false,
            true,
        ),
        (OnlineAttention, 72) => ([n * 3456 * 4, 0, 0, 0, 0, n * 1152 * 4], false, false),
        _ => return Err(error(CudaRuntimeErrorKind::InvalidDimensions)),
    };
    Ok(result)
}
fn bind_raw(
    ordinal: i32,
    op: CudaQwen35VisionControlOp,
    tokens: usize,
    cols: usize,
    views: CudaQwen35VisionControlViews,
) -> Result<AugurCudaQwen35VisionControlArgsV1, CudaRuntimeError> {
    if ordinal < 0 {
        return Err(error(CudaRuntimeErrorKind::InvalidMetadata));
    }
    let (required, half_output, inplace) = shape(op, tokens, cols)?;
    let buffers = [
        views.inputs[0],
        views.inputs[1],
        views.inputs[2],
        views.inputs[3],
        views.inputs[4],
        Some(views.output),
        Some(views.status),
    ];
    let mut pointers = [0u64; 7];
    let mut sizes = [0u64; 7];
    for (index, view) in buffers.iter().enumerate() {
        let expected = if index == 6 { 4 } else { required[index] };
        let Some(view) = view else {
            if expected != 0 {
                return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
            }
            continue;
        };
        if expected == 0 || view.len_bytes() != expected {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if view.device_ordinal() != ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let address = view.checked_address(OP)?;
        if address % (if index == 5 && half_output { 2 } else { 4 }) != 0 {
            return Err(error(CudaRuntimeErrorKind::MisalignedDeviceAddress));
        }
        let range = view.checked_range(OP)?;
        for (prior_index, prior) in buffers[..index].iter().enumerate() {
            if let Some(prior) = prior {
                if inplace && index == 5 && prior_index == 0 && range == prior.checked_range(OP)? {
                    continue;
                }
                if view.overlaps(*prior, OP)? {
                    return Err(error(CudaRuntimeErrorKind::OverlappingDeviceRanges));
                }
            }
        }
        pointers[index] = address;
        sizes[index] = expected as u64;
    }
    if inplace && (pointers[0] != pointers[5] || sizes[0] != sizes[5]) {
        return Err(error(CudaRuntimeErrorKind::InvalidMetadata));
    }
    Ok(AugurCudaQwen35VisionControlArgsV1 {
        abi_version: 1,
        struct_size: core::mem::size_of::<AugurCudaQwen35VisionControlArgsV1>() as u32,
        operation: op as u32,
        tokens: tokens as u32,
        cols: cols as u32,
        reserved: 0,
        input_a: pointers[0],
        input_a_bytes: sizes[0],
        input_b: pointers[1],
        input_b_bytes: sizes[1],
        input_c: pointers[2],
        input_c_bytes: sizes[2],
        input_d: pointers[3],
        input_d_bytes: sizes[3],
        input_e: pointers[4],
        input_e_bytes: sizes[4],
        output: pointers[5],
        output_bytes: sizes[5],
        status: pointers[6],
        status_bytes: sizes[6],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_runtime::test_buffer_view;
    fn fixture(op: CudaQwen35VisionControlOp, cols: usize) -> CudaQwen35VisionControlViews {
        let (sizes, _, inplace) = shape(op, 4, cols).unwrap();
        let mut inputs = [None; 5];
        for i in 0..5 {
            if sizes[i] != 0 {
                inputs[i] = Some(test_buffer_view(
                    0x1000_0000 + i as u64 * 0x1000_0000,
                    0,
                    0,
                    sizes[i],
                ));
            }
        }
        CudaQwen35VisionControlViews {
            inputs,
            output: if inplace {
                inputs[0].unwrap()
            } else {
                test_buffer_view(0x6000_0000, 0, 0, sizes[5])
            },
            status: test_buffer_view(0x7000_0000, 0, 0, 4),
        }
    }
    fn all_shapes() -> [(CudaQwen35VisionControlOp, usize); 8] {
        use CudaQwen35VisionControlOp::*;
        [
            (PositionAdd, 1152),
            (LayerNormToF16, 1152),
            (CastToF16, 1152),
            (GeluTanhToF16, 4304),
            (GeluErfToF16, 4608),
            (ResidualAdd, 1152),
            (RotaryQkInPlace, 72),
            (OnlineAttention, 72),
        ]
    }
    #[test]
    fn vision_control_abi_and_all_stages() {
        assert_eq!(core::mem::size_of::<AugurCudaQwen35VisionControlArgsV1>(), 136);
        assert_eq!(
            core::mem::offset_of!(AugurCudaQwen35VisionControlArgsV1, status),
            120
        );
        for (op, c) in all_shapes() {
            assert_eq!(
                bind_raw(0, op, 4, c, fixture(op, c)).unwrap().operation,
                op as u32
            );
        }
    }
    #[test]
    fn vision_control_rejects_extent_device_and_optional_drift() {
        for (op, c) in all_shapes() {
            let good = fixture(op, c);
            for n in [0, 5, 4097] {
                assert!(bind_raw(0, op, n, c, good).is_err());
            }
            assert!(bind_raw(0, op, 4, c + 1, good).is_err());
            assert!(bind_raw(1, op, 4, c, good).is_err());
            let mut bad = good;
            bad.status = test_buffer_view(0x7000_0000, 0, 0, 8);
            assert!(bind_raw(0, op, 4, c, bad).is_err());
            bad = good;
            bad.inputs[4] = Some(test_buffer_view(0x5000_0000, 0, 0, 4));
            assert!(bind_raw(0, op, 4, c, bad).is_err());
        }
    }
    #[test]
    fn vision_control_rejects_flag_alias_partial_alias_and_unbound_inplace() {
        for (op, c) in all_shapes() {
            let mut bad = fixture(op, c);
            bad.status = bad.inputs[0].unwrap().subview(0, 4).unwrap();
            assert!(bind_raw(0, op, 4, c, bad).is_err());
        }
        let op = CudaQwen35VisionControlOp::ResidualAdd;
        let mut bad = fixture(op, 1152);
        bad.output = test_buffer_view(0x6000_0000, 0, 0, 4 * 1152 * 4);
        assert!(bind_raw(0, op, 4, 1152, bad).is_err());
        bad = fixture(op, 1152);
        bad.inputs[1] = bad.inputs[0];
        assert!(bind_raw(0, op, 4, 1152, bad).is_err());
    }
}
