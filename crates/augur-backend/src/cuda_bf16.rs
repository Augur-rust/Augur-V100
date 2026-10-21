#![cfg(feature = "cuda")]

use crate::{CudaBufferView, CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};
use augur_core::PhysicalLayoutId;
use augur_cuda_sys::{AugurCudaBf16F32MatvecArgsV1, AugurCudaBf16TokenRowGatherArgsV1};

const OP: &str = "cuda_bf16_f32_matvec_v1";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaBf16WeightViewV1 {
    view: CudaBufferView,
    rows: u32,
    cols: u32,
}

impl CudaBf16WeightViewV1 {
    pub fn new(
        view: CudaBufferView,
        layout: PhysicalLayoutId,
        rows: usize,
        cols: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let expected = rows
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))?;
        if layout != PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1
            || rows == 0
            || cols == 0
            || view.len_bytes() != expected
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        Ok(Self {
            view,
            rows: rows
                .try_into()
                .map_err(|_| error(CudaRuntimeErrorKind::InvalidDimensions))?,
            cols: cols
                .try_into()
                .map_err(|_| error(CudaRuntimeErrorKind::InvalidDimensions))?,
        })
    }

    pub const fn buffer_view(self) -> CudaBufferView {
        self.view
    }

    pub const fn rows(self) -> u32 {
        self.rows
    }

    pub const fn cols(self) -> u32 {
        self.cols
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaBf16F32MatvecArgsV1 {
    raw: AugurCudaBf16F32MatvecArgsV1,
    ordinal: i32,
    #[cfg_attr(
        not(feature = "p0-prefill-v22-projection-resume-batch-exact"),
        allow(dead_code)
    )]
    tokens: u32,
}

impl CudaBf16F32MatvecArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: CudaBf16WeightViewV1,
        input: CudaF32View,
        output: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        if input.element_count() != weight.cols as usize
            || output.element_count() != weight.rows as usize
            || [weight.view, input.buffer_view(), output.buffer_view()]
                .iter()
                .any(|v| v.device_ordinal() != stream.device_ordinal())
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let mut raw = AugurCudaBf16F32MatvecArgsV1::new();
        raw.weight = address(weight.view)?;
        raw.weight_len_bytes = weight.view.len_bytes() as u64;
        raw.input = address(input.buffer_view())?;
        raw.input_len_bytes = input.buffer_view().len_bytes() as u64;
        raw.output = address(output.buffer_view())?;
        raw.output_len_bytes = output.buffer_view().len_bytes() as u64;
        raw.rows = weight.rows;
        raw.cols = weight.cols;
        Ok(Self {
            raw,
            ordinal: stream.device_ordinal(),
            tokens: 1,
        })
    }

    
    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_bf16_f32_matvec_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OP, result)
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_bf16_f32_matvec_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_bf16_f32_matvec_submit_v1", result)
    }

    
}

/// Explicit research pair. The caller retains the two authenticated scalar
/// owners and all buffers; binding cannot opt an existing scalar into new math.
#[derive(Clone, Copy, Debug)]
pub struct CudaNerkyorBf16GatePairF32CtaArgsV1 {
    stream_owner_instance_key: usize,
    left: CudaBf16F32MatvecArgsV1,
    right: CudaBf16F32MatvecArgsV1,
}
impl CudaNerkyorBf16GatePairF32CtaArgsV1 {
    pub fn bind(stream: &OwnedCudaStream, left: CudaBf16F32MatvecArgsV1,
        right: CudaBf16F32MatvecArgsV1) -> Result<Self, CudaRuntimeError> {
        if [left.ordinal, right.ordinal].iter().any(|&v| v != stream.device_ordinal()) {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        if left.tokens != 1 || right.tokens != 1 || left.raw.rows != 48 || right.raw.rows != 48
            || left.raw.cols != 5120 || right.raw.cols != 5120 || left.raw.input != right.raw.input {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let ranges = [(left.raw.weight,left.raw.weight_len_bytes),
            (right.raw.weight,right.raw.weight_len_bytes),(left.raw.input,left.raw.input_len_bytes),
            (left.raw.output,left.raw.output_len_bytes),(right.raw.output,right.raw.output_len_bytes)];
        for (i, &(a, an)) in ranges.iter().enumerate() {
            let ae = a.checked_add(an).ok_or_else(|| error(CudaRuntimeErrorKind::InvalidDimensions))?;
            for &(b, bn) in &ranges[i+1..] {
                let be = b.checked_add(bn).ok_or_else(|| error(CudaRuntimeErrorKind::InvalidDimensions))?;
                if a < be && b < ae { return Err(error(CudaRuntimeErrorKind::InvalidDimensions)); }
            }
        }
        if crate::cuda_device_properties(stream.device_ordinal())?.compute_capability != crate::CudaComputeCapability::SM70 {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        Ok(Self {stream_owner_instance_key: stream.owner_instance_key()?, left, right})
    }
    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.left.ordinal || stream.owner_instance_key()? != self.stream_owner_instance_key {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let result = unsafe { augur_cuda_sys::augur_cuda_nerkyor_bf16_gate_pair_f32_cta_submit_v1(
            stream.device_ordinal(), stream.raw()?, &self.left.raw, &self.right.raw) };
        CudaRuntimeError::from_result("nerkyor_bf16_gate_pair_f32_cta_submit_v1", result)
    }
}

/// Independent token-major rows with the canonical warp-row FP64 reduction.
/// The caller retains all backing buffers until stream completion.
#[derive(Clone, Copy, Debug)]
pub struct CudaBf16F32MatvecRowsArgsV1 {
    raw: AugurCudaBf16F32MatvecArgsV1,
    tokens: u32,
    ordinal: i32,
    stream_owner_instance_key: usize,
}

impl CudaBf16F32MatvecRowsArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: CudaBf16WeightViewV1,
        input: CudaF32View,
        output: CudaF32View,
        tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let inputs = tokens.checked_mul(weight.cols as usize)
            .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))?;
        let outputs = tokens.checked_mul(weight.rows as usize)
            .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))?;
        let views = [weight.view, input.buffer_view(), output.buffer_view()];
        if !cfg!(feature = "m306-k17-bf16-warp-row") || !(1..=512).contains(&tokens)
            || weight.rows > 65535 || weight.cols > 65535 || input.element_count() != inputs || output.element_count() != outputs
            || views.iter().any(|v| v.device_ordinal() != stream.device_ordinal()) {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let mut raw = AugurCudaBf16F32MatvecArgsV1::new();
        let mut addresses = [0u64; 3];
        for (i, view) in views.iter().enumerate() {
            let begin = address(*view)?;
            let end = begin.checked_add(view.len_bytes() as u64)
                .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))?;
            if begin == 0 || begin % if i == 0 { 2 } else { 4 } != 0 {
                return Err(error(CudaRuntimeErrorKind::MisalignedDeviceAddress));
            }
            for j in 0..i {
                if begin < addresses[j] + views[j].len_bytes() as u64 && addresses[j] < end {
                    return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
                }
            }
            addresses[i] = begin;
        }
        raw.weight = addresses[0]; raw.weight_len_bytes = views[0].len_bytes() as u64;
        raw.input = addresses[1]; raw.input_len_bytes = views[1].len_bytes() as u64;
        raw.output = addresses[2]; raw.output_len_bytes = views[2].len_bytes() as u64;
        raw.rows = weight.rows; raw.cols = weight.cols;
        Ok(Self { raw, tokens: tokens as u32, ordinal: stream.device_ordinal(),
            stream_owner_instance_key: stream.owner_instance_key()? })
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal
            || stream.owner_instance_key()? != self.stream_owner_instance_key {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        // SAFETY: bound exact disjoint extents, rows and stream owner above;
        // native checks sizes/ABI/ranges again without changing stream ordering.
        let result = unsafe { augur_cuda_sys::augur_cuda_bf16_f32_matvec_rows_submit_v1(
            self.ordinal, stream.raw()?, &self.raw, self.tokens) };
        CudaRuntimeError::from_result("cuda_bf16_f32_matvec_rows_submit_v1", result)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaBf16TokenRowGatherArgsV1 {
    raw: AugurCudaBf16TokenRowGatherArgsV1,
    ordinal: i32,
}

impl CudaBf16TokenRowGatherArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: CudaBf16WeightViewV1,
        token_source_u32: CudaBufferView,
        output: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        let ordinal = stream.device_ordinal();
        let output_view = output.buffer_view();
        if token_source_u32.device_ordinal() != ordinal
            || weight.view.device_ordinal() != ordinal
            || output_view.device_ordinal() != ordinal
            || token_source_u32.len_bytes() < core::mem::size_of::<u32>()
            || output.element_count() != weight.cols as usize
            || overlaps(output_view, weight.view)?
            || overlaps(output_view, token_source_u32)?
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let token_address = address(token_source_u32)?;
        if !token_address.is_multiple_of(core::mem::align_of::<u32>() as u64) {
            return Err(error(CudaRuntimeErrorKind::MisalignedDeviceAddress));
        }
        let mut raw = AugurCudaBf16TokenRowGatherArgsV1::new();
        raw.weight = address(weight.view)?;
        raw.weight_len_bytes = weight.view.len_bytes() as u64;
        raw.token_source_u32 = token_address;
        raw.token_source_len_bytes = token_source_u32.len_bytes() as u64;
        raw.output_f32 = address(output_view)?;
        raw.output_len_bytes = output_view.len_bytes() as u64;
        raw.rows = weight.rows;
        raw.cols = weight.cols;
        Ok(Self { raw, ordinal })
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        let result = unsafe {
            augur_cuda_sys::augur_cuda_bf16_token_row_gather_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_bf16_token_row_gather_v1", result)
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        let result = unsafe {
            augur_cuda_sys::augur_cuda_bf16_token_row_gather_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_bf16_token_row_gather_submit_v1", result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        Ok(())
    }
}

fn gather_elements(tokens: usize, cols: usize) -> Result<usize, CudaRuntimeError> {
    if !(1..=512).contains(&tokens) || cols == 0 || cols > u32::MAX as usize {
        return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
    }
    tokens.checked_mul(cols).and_then(|n| n.checked_mul(4)).map(|bytes| bytes / 4)
        .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))
}

/// Same BF16 bit expansion as the scalar token gather; bounded independent rows.
#[derive(Clone, Copy, Debug)]
pub struct CudaBf16TokenRowsGatherArgsV1 {
    raw: AugurCudaBf16TokenRowGatherArgsV1,
    ordinal: i32,
    stream_owner: usize,
    tokens: u32,
}

impl CudaBf16TokenRowsGatherArgsV1 {
    pub fn bind(stream: &OwnedCudaStream, weight: CudaBf16WeightViewV1,
        ids: CudaBufferView, output: CudaF32View, tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let elements = gather_elements(tokens, weight.cols as usize)?;
        let out = output.buffer_view();
        if output.element_count() != elements || out.len_bytes() != elements * 4
            || ids.len_bytes() != tokens * 4
            || [weight.view, ids, out].iter().any(|v| v.device_ordinal() != stream.device_ordinal())
            || overlaps(out, weight.view)? || overlaps(out, ids)? {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        for (v, alignment) in [(weight.view, 2), (ids, 4), (out, 4)] {
            v.checked_range("cuda_bf16_token_rows_gather")?;
            if address(v)? % alignment != 0 { return Err(error(CudaRuntimeErrorKind::MisalignedDeviceAddress)); }
        }
        let mut raw = AugurCudaBf16TokenRowGatherArgsV1::new();
        raw.weight = address(weight.view)?;
        raw.weight_len_bytes = weight.view.len_bytes() as u64;
        raw.token_source_u32 = address(ids)?;
        raw.token_source_len_bytes = ids.len_bytes() as u64;
        raw.output_f32 = address(out)?;
        raw.output_len_bytes = out.len_bytes() as u64;
        raw.rows = weight.rows;
        raw.cols = weight.cols;
        Ok(Self { raw, ordinal: stream.device_ordinal(),
            stream_owner: stream.owner_instance_key()?, tokens: tokens as u32 })
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal || stream.owner_instance_key()? != self.stream_owner {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        // SAFETY: bind checks all extents and aliasing; the plan retains the
        // weight/ID/output owners through completion on this exact stream.
        let result = unsafe { augur_cuda_sys::augur_cuda_bf16_token_rows_gather_submit_v1(
            self.ordinal, stream.raw()?, &self.raw, self.tokens) };
        CudaRuntimeError::from_result("cuda_bf16_token_rows_gather", result)
    }
}

fn error(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OP, kind)
}
fn address(view: CudaBufferView) -> Result<u64, CudaRuntimeError> {
    view.device_ptr()
        .raw_value()
        .checked_add(view.offset_bytes() as u64)
        .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))
}

fn overlaps(a: CudaBufferView, b: CudaBufferView) -> Result<bool, CudaRuntimeError> {
    if a.device_ptr() != b.device_ptr() {
        return Ok(false);
    }
    let a_end = a
        .offset_bytes()
        .checked_add(a.len_bytes())
        .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))?;
    let b_end = b
        .offset_bytes()
        .checked_add(b.len_bytes())
        .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))?;
    Ok(a.offset_bytes() < b_end && b.offset_bytes() < a_end)
}

#[cfg(test)]
mod submission_bundle_tests {
    use super::*;
    use crate::{OwnedCudaBuffer, OwnedCudaEvent, cuda_device_allocation_snapshot};

    fn f32_view(b: &OwnedCudaBuffer, offset: usize, n: usize) -> CudaF32View {
        CudaF32View::new(b.view(offset, n * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, n).unwrap()
    }

    fn measure(stream: &OwnedCudaStream, mut work: impl FnMut()) -> (f64, f32) {
        stream.synchronize().unwrap();
        let start = OwnedCudaEvent::create(0).unwrap();
        let stop = OwnedCudaEvent::create(0).unwrap();
        start.record(stream).unwrap();
        let host = std::time::Instant::now();
        for _ in 0..40 { work(); }
        let host_ms = host.elapsed().as_secs_f64() * 1000.0 / 40.0;
        stop.record(stream).unwrap();
        stop.synchronize().unwrap();
        (host_ms, start.elapsed_ms(&stop).unwrap() / 40.0)
    }

    #[test]
    fn submission_bundle_gather_geometry() {
        for rows in [1, 7, 8, 9, 31, 511, 512] {
            assert_eq!(gather_elements(rows, 5120).unwrap(), rows * 5120);
        }
        for rows in [0, 513, usize::MAX] { assert!(gather_elements(rows, 5120).is_err()); }
        for cols in [0, usize::MAX] { assert!(gather_elements(8, cols).is_err()); }
    }

    #[test]
    #[ignore = "exclusive CUDA component; exact bits and submission cost, not HTTP qualification"]
    fn submission_bundle_gather_gpu() {
        let before = cuda_device_allocation_snapshot();
        {
            let stream = OwnedCudaStream::create(0).unwrap();
            let foreign = OwnedCudaStream::create(0).unwrap();
            let vocab = 17;
            let mut cases = 0;
            for cols in [1, 255, 256, 257, 5120] {
                // Full BF16 bit-pattern cycle, including signed zero, subnormal,
                // infinities and NaN payloads. No floating point host conversion.
                let weights: Vec<u8> = (0..vocab*cols).flat_map(|i| (i as u16).to_le_bytes()).collect();
                let w = OwnedCudaBuffer::allocate(0, weights.len()).unwrap();
                w.copy_from_host_sync(&stream, &weights).unwrap();
                let weight = CudaBf16WeightViewV1::new(w.view(0, weights.len()).unwrap(),
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1, vocab, cols).unwrap();
                for rows in [1, 7, 8, 9, 31, 511, 512] {
                    let ids = OwnedCudaBuffer::allocate(0, rows * 4).unwrap();
                    let guard = vec![0xa5; rows * cols * 4 + 64];
                    let out = OwnedCudaBuffer::allocate(0, guard.len()).unwrap();
                    let oracle = OwnedCudaBuffer::allocate(0, guard.len()).unwrap();
                    let batch = CudaBf16TokenRowsGatherArgsV1::bind(&stream, weight,
                        ids.view(0, rows * 4).unwrap(), f32_view(&out, 32, rows * cols), rows).unwrap();
                    assert!(batch.submit_for_plan(&foreign).is_err());
                    let scalar: Vec<_> = (0..rows).map(|row| CudaBf16TokenRowGatherArgsV1::bind(
                        &stream, weight, ids.view(row*4,4).unwrap(),
                        f32_view(&oracle,32+row*cols*4,cols)).unwrap()).collect();
                    for generation in 0..3 {
                        let token_ids: Vec<u32> = (0..rows).map(|i| match (i+generation)%19 {
                            17 => vocab as u32, 18 => u32::MAX, n => n as u32 }).collect();
                        let id_bytes: Vec<_> = token_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
                        ids.copy_from_host_sync(&stream,&id_bytes).unwrap();
                        out.copy_from_host_sync(&stream,&guard).unwrap();
                        oracle.copy_from_host_sync(&stream,&guard).unwrap();
                        batch.submit_for_plan(&stream).unwrap();
                        for op in &scalar { op.submit_for_plan(&stream).unwrap(); }
                        let mut actual = guard.clone(); let mut reference = guard.clone();
                        out.copy_to_host_sync(&stream,&mut actual).unwrap();
                        oracle.copy_to_host_sync(&stream,&mut reference).unwrap();
                        let mut expected = guard.clone();
                        for (row, token) in token_ids.iter().enumerate() {
                            for col in 0..cols {
                                let bits = if *token as usize >= vocab { 0x7fc00000 } else {
                                    (((*token as usize * cols + col) as u16) as u32) << 16 };
                                let at = 32 + (row*cols+col)*4;
                                expected[at..at+4].copy_from_slice(&bits.to_le_bytes());
                            }
                        }
                        assert_eq!(actual,reference,"scalar rows={rows} cols={cols} generation={generation}");
                        assert_eq!(actual,expected,"CPU bits / guards");
                        let mut unchanged = id_bytes.clone();
                        ids.copy_to_host_sync(&stream,&mut unchanged).unwrap(); assert_eq!(unchanged,id_bytes);
                        cases += 1;
                    }
                    for case in 0..16 {
                        let mut raw=batch.raw; let mut n=rows as u32;
                        match case {
                            0=>n=0,1=>n=513,2=>raw.abi_version=0,3=>raw.struct_size=0,
                            4=>raw.token_source_len_bytes-=1,5=>raw.output_len_bytes-=1,
                            6=>raw.weight_len_bytes-=1,7=>raw.output_f32=raw.weight,
                            8=>raw.output_f32=raw.token_source_u32,9=>raw.output_f32+=1,
                            10=>raw.token_source_u32+=1,11=>raw.weight+=1,
                            12=>raw.weight=0,13=>raw.rows=0,14=>raw.cols=0,
                            _=>raw.weight=u64::MAX-1,
                        }
                        // SAFETY: invalid metadata rejects before device access.
                        let result=unsafe { augur_cuda_sys::augur_cuda_bf16_token_rows_gather_submit_v1(
                            0,stream.raw().unwrap(),&raw,n) };
                        assert_eq!(result.status,augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT,"case={case}");
                    }
                    if cols == 5120 && [8,512].contains(&rows) {
                        for batched in [false,true,true,false] {
                            let t=measure(&stream,|| { if batched { batch.submit_for_plan(&stream).unwrap(); }
                                else { for op in &scalar { op.submit_for_plan(&stream).unwrap(); } } });
                            println!("GATHER_COST rows={rows} batched={batched} host_enqueue_ms={} stream_envelope_ms={}",t.0,t.1);
                        }
                    }
                }
                let mut actual=weights.clone(); w.copy_to_host_sync(&stream,&mut actual).unwrap(); assert_eq!(actual,weights);
            }
            stream.synchronize().unwrap();
            println!("GATHER_EXACT_PASS cases={cases} native_rejections=560");
        }
        let after=cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes,before.live_bytes);
        assert_eq!(after.live_allocations,before.live_allocations);
        assert_eq!(after.successful_allocations-before.successful_allocations,
            after.successful_frees-before.successful_frees);
        println!("GATHER_CLEANUP_PASS live_bytes_delta=0 live_allocations_delta=0");
    }

    #[test]
    #[ignore = "exclusive CUDA component; exact bits, pitch guards and enqueue timing"]
    fn submission_bundle_feature_copies_gpu() {
        let before=cuda_device_allocation_snapshot();
        {
            let stream=OwnedCudaStream::create(0).unwrap();
            let width: usize=5120*4; let rows=8;
            let source: Vec<u8>=(0..rows*width).map(|i|
                ((i.wrapping_mul(71) ^ (i >> 11) ^ (i >> 17)) >> 3) as u8).collect();
            let x=OwnedCudaBuffer::allocate(0,source.len()).unwrap();
            x.copy_from_host_sync(&stream,&source).unwrap();
            let guard=vec![0xa5; rows*5*width+64];
            let y=OwnedCudaBuffer::allocate(0,guard.len()).unwrap();
            let old=OwnedCudaBuffer::allocate(0,guard.len()).unwrap();
            for slot in 0..5 {
                y.copy_from_host_sync(&stream,&guard).unwrap();
                old.copy_from_host_sync(&stream,&guard).unwrap();
                unsafe { y.copy_from_device_2d_async_unchecked(&stream,32+slot*width,5*width,
                    x.view(0,source.len()).unwrap(),width,width,rows).unwrap(); }
                for row in 0..rows { unsafe { old.copy_from_device_async_unchecked(&stream,
                    32+(row*5+slot)*width,x.view(row*width,width).unwrap()).unwrap(); } }
                let mut actual=guard.clone(); let mut reference=guard.clone(); let mut expected=guard.clone();
                for row in 0..rows { let at=32+(row*5+slot)*width;
                    expected[at..at+width].copy_from_slice(&source[row*width..(row+1)*width]); }
                y.copy_to_host_sync(&stream,&mut actual).unwrap(); old.copy_to_host_sync(&stream,&mut reference).unwrap();
                assert_eq!(actual,expected); assert_eq!(actual,reference);
                for batched in [false,true,true,false] {
                    let t=measure(&stream,|| unsafe { if batched {
                        y.copy_from_device_2d_async_unchecked(&stream,32+slot*width,5*width,
                            x.view(0,source.len()).unwrap(),width,width,rows).unwrap();
                    } else { for row in 0..rows { old.copy_from_device_async_unchecked(&stream,
                        32+(row*5+slot)*width,x.view(row*width,width).unwrap()).unwrap(); } } });
                    println!("FEATURE_COPY_COST slot={slot} batched={batched} host_enqueue_ms={} stream_envelope_ms={}",t.0,t.1);
                }
            }
            for (dp,sp,w,h,offset) in [(0,width,width,8,0),(width,0,width,8,0),
                (5*width,width,0,8,0),(5*width,width,width,0,0),
                (5*width,width,width,9,0),(5*width,width,width,8,usize::MAX)] {
                assert!(unsafe { y.copy_from_device_2d_async_unchecked(&stream,offset,dp,
                    x.view(0,source.len()).unwrap(),sp,w,h) }.is_err());
            }
            assert!(unsafe { y.copy_from_device_2d_async_unchecked(&stream,0,width,
                y.view(0,width*8).unwrap(),width,width,8) }.is_err());
            let mut unchanged=source.clone(); x.copy_to_host_sync(&stream,&mut unchanged).unwrap(); assert_eq!(unchanged,source);
            stream.synchronize().unwrap();
            println!("FEATURE_COPY_EXACT_PASS slots=5 invalid=7 guards_and_source_unchanged=true");
        }
        let after=cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes,before.live_bytes); assert_eq!(after.live_allocations,before.live_allocations);
        assert_eq!(after.successful_allocations-before.successful_allocations,after.successful_frees-before.successful_frees);
        println!("FEATURE_COPY_CLEANUP_PASS live_bytes_delta=0 live_allocations_delta=0");
    }

    #[test]
    #[ignore = "exclusive CUDA: wide/partial initialized-history copies, untouched rows and slots"]
    fn submission_cleanup_history_copies_gpu() {
        let before=cuda_device_allocation_snapshot();
        let mut cases=0;
        {
            let stream=OwnedCudaStream::create(0).unwrap();
            let width=5120*4;
            for (source_row,destination_row,rows) in [(0,0,1),(511,2047,1),(3,2039,8),
                (449,0,63),(227,0,285),(0,1763,285),(1,0,511),(0,1536,512)] {
                let source: Vec<u8>=(0usize..512*width).map(|i|
                    ((i.wrapping_mul(71) ^ (i >> 11) ^ (i >> 17)) >> 3) as u8).collect();
                let x=OwnedCudaBuffer::allocate(0,source.len()).unwrap();
                x.copy_from_host_sync(&stream,&source).unwrap();
                let guard=vec![0xa5;2048*5*width+64];
                let y=OwnedCudaBuffer::allocate(0,guard.len()).unwrap();
                for slot in 0..5 {
                    y.copy_from_host_sync(&stream,&guard).unwrap();
                    unsafe { y.copy_from_device_2d_async_unchecked(&stream,
                        32+(destination_row*5+slot)*width,5*width,
                        x.view(source_row*width,rows*width).unwrap(),width,width,rows).unwrap(); }
                    let mut actual=vec![0;guard.len()]; y.copy_to_host_sync(&stream,&mut actual).unwrap();
                    let mut expected=guard.clone();
                    for row in 0..rows { let at=32+((destination_row+row)*5+slot)*width;
                        expected[at..at+width].copy_from_slice(&source[(source_row+row)*width..(source_row+row+1)*width]); }
                    assert!(actual==expected,"history copy rows={rows} source={source_row} dest={destination_row} slot={slot}");
                    cases+=1;
                }
                if rows==512 {
                    for batched in [false,true,true,false] {
                        let cost=measure(&stream,|| unsafe {
                            if batched { y.copy_from_device_2d_async_unchecked(&stream,
                                32+destination_row*5*width,5*width,x.view(0,rows*width).unwrap(),width,width,rows).unwrap(); }
                            else { for row in 0..rows { y.copy_from_device_async_unchecked(&stream,
                                32+(destination_row+row)*5*width,x.view(row*width,width).unwrap()).unwrap(); } }
                        });
                        println!("HISTORY_COPY_COST rows={rows} batched={batched} host_enqueue_ms={} stream_envelope_ms={}",cost.0,cost.1);
                    }
                }
                let mut unchanged=vec![0;source.len()];x.copy_to_host_sync(&stream,&mut unchanged).unwrap();
                assert!(unchanged==source);stream.synchronize().unwrap();
            }
        }
        let after=cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes,before.live_bytes);assert_eq!(after.live_allocations,before.live_allocations);
        assert_eq!(after.successful_allocations-before.successful_allocations,after.successful_frees-before.successful_frees);
        println!("HISTORY_COPY_EXACT cases={cases} guards_and_source_unchanged=true live_bytes_delta=0 live_allocations_delta=0");
    }

    #[test]
    #[ignore = "exclusive CUDA screening; narrow draft KV shift is not automatically admitted"]
    fn submission_bundle_narrow_kv_shift_gpu() {
        let before=cuda_device_allocation_snapshot();
        {
            let stream=OwnedCudaStream::create(0).unwrap();
            let width=1024*4;
            let bytes: Vec<u8>=(0usize..2048*width).map(|i|
                ((i.wrapping_mul(73) ^ (i >> 9) ^ (i >> 19)) >> 3) as u8).collect();
            let old=OwnedCudaBuffer::allocate(0,bytes.len()).unwrap();
            let new=OwnedCudaBuffer::allocate(0,bytes.len()).unwrap();
            let scratch=OwnedCudaBuffer::allocate(0,bytes.len()).unwrap();
            for shift in [1,4,8] {
                let retained=2048-shift; let n=retained*width;
                old.copy_from_host_sync(&stream,&bytes).unwrap(); new.copy_from_host_sync(&stream,&bytes).unwrap();
                unsafe {
                    scratch.copy_from_device_async_unchecked(&stream,0,old.view(shift*width,n).unwrap()).unwrap();
                    old.copy_from_device_async_unchecked(&stream,0,scratch.view(0,n).unwrap()).unwrap();
                    new.shift_rows_u32_async_unchecked(&stream,width,retained,shift).unwrap();
                }
                let mut a=bytes.clone();let mut b=bytes.clone();let mut expected=bytes.clone();
                expected[..n].copy_from_slice(&bytes[shift*width..]);
                old.copy_to_host_sync(&stream,&mut a).unwrap();new.copy_to_host_sync(&stream,&mut b).unwrap();
                assert_eq!(a,expected);assert_eq!(b,expected);
                for inplace in [false,true,true,false] {
                    let t=measure(&stream,|| unsafe { if inplace {
                        new.shift_rows_u32_async_unchecked(&stream,width,retained,shift).unwrap();
                    } else {
                        scratch.copy_from_device_async_unchecked(&stream,0,old.view(shift*width,n).unwrap()).unwrap();
                        old.copy_from_device_async_unchecked(&stream,0,scratch.view(0,n).unwrap()).unwrap();
                    } });
                    println!("NARROW_KV_SHIFT_COST shift={shift} inplace={inplace} host_enqueue_ms={} stream_envelope_ms={}",t.0,t.1);
                }
            }
            stream.synchronize().unwrap();
        }
        let after=cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes,before.live_bytes); assert_eq!(after.live_allocations,before.live_allocations);
        assert_eq!(after.successful_allocations-before.successful_allocations,after.successful_frees-before.successful_frees);
        println!("NARROW_KV_SHIFT_EXACT_CLEANUP_PASS cases=3 live_delta=0");
    }
}

#[cfg(all(test, feature = "m306-k17-bf16-warp-row"))]
mod row_tests {
    use super::*;
    use crate::{CudaRmsnormF32ArgsV1, OwnedCudaBuffer};

    fn f32_view(b: &OwnedCudaBuffer, offset: usize, n: usize) -> CudaF32View {
        CudaF32View::new(b.view(offset * 4, n * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, n).unwrap()
    }

    fn compare(input: &[u8], weight: &[u8], tokens: usize, rows: usize, cols: usize) {
        assert_eq!(input.len(), tokens * cols * 4);
        assert_eq!(weight.len(), rows * cols * 2);
        let stream = OwnedCudaStream::create(0).unwrap();
        let foreign = OwnedCudaStream::create(0).unwrap();
        let x = OwnedCudaBuffer::allocate(0, input.len()).unwrap();
        let w = OwnedCudaBuffer::allocate(0, weight.len()).unwrap();
        let control = OwnedCudaBuffer::allocate(0, tokens * rows * 4).unwrap();
        let candidate = OwnedCudaBuffer::allocate(0, tokens * rows * 4).unwrap();
        x.copy_from_host_sync(&stream, input).unwrap();
        w.copy_from_host_sync(&stream, weight).unwrap();
        let weights = CudaBf16WeightViewV1::new(w.view(0, weight.len()).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1, rows, cols).unwrap();
        let batch = CudaBf16F32MatvecRowsArgsV1::bind(&stream, weights,
            f32_view(&x, 0, tokens * cols), f32_view(&candidate, 0, tokens * rows), tokens).unwrap();
        assert!(batch.submit_for_plan(&foreign).is_err(), "foreign stream accepted");
        let scalar: Vec<_> = (0..tokens).map(|t| CudaBf16F32MatvecArgsV1::bind(&stream,
            weights, f32_view(&x, t * cols, cols), f32_view(&control, t * rows, rows)).unwrap()).collect();
        for repeat in 0..3 {
            candidate.copy_from_host_sync(&stream, &vec![0xff; tokens * rows * 4]).unwrap();
            if repeat % 2 == 0 { batch.submit_for_plan(&stream).unwrap(); }
            for op in &scalar { op.submit_for_plan(&stream).unwrap(); }
            if repeat % 2 == 1 { batch.submit_for_plan(&stream).unwrap(); }
            let mut expected = vec![0; tokens * rows * 4];
            let mut actual = expected.clone();
            control.copy_to_host_sync(&stream, &mut expected).unwrap();
            candidate.copy_to_host_sync(&stream, &mut actual).unwrap();
            assert!(expected.chunks_exact(4).all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()));
            assert_eq!(actual, expected, "canonical GDN bits changed: {tokens}x{rows}x{cols}");
        }
        let mut unchanged = vec![0; input.len()];
        x.copy_to_host_sync(&stream, &mut unchanged).unwrap();
        assert_eq!(unchanged, input);
        let mut unchanged = vec![0; weight.len()];
        w.copy_to_host_sync(&stream, &mut unchanged).unwrap();
        assert_eq!(unchanged, weight);
        println!("canonical_gdn_exact tokens={tokens} rows={rows} cols={cols}");
    }

    #[test]
    #[ignore = "requires exclusive SM70 GPU and real target fixture"]
    fn canonical_gdn_rows_gpu_real_layer0() {
        let root = std::path::PathBuf::from(std::env::var_os("AUGUR_QWEN35_GDN_FIXTURE_DIR").unwrap());
        let input = std::fs::read(root.join("input.f32")).unwrap();
        let norm = std::fs::read(root.join("norm.f32")).unwrap();
        assert_eq!(input.len(), 512 * 5120 * 4);
        assert_eq!(norm.len(), 5120 * 4);
        // Use the actual scalar device norm to produce layer0 projection inputs.
        let normalized = {
            let stream = OwnedCudaStream::create(0).unwrap();
            let x = OwnedCudaBuffer::allocate(0, input.len()).unwrap();
            let n = OwnedCudaBuffer::allocate(0, norm.len()).unwrap();
            let y = OwnedCudaBuffer::allocate(0, input.len()).unwrap();
            x.copy_from_host_sync(&stream, &input).unwrap();
            n.copy_from_host_sync(&stream, &norm).unwrap();
            for token in 0..512 {
                CudaRmsnormF32ArgsV1::bind(&stream, f32_view(&x, token * 5120, 5120),
                    f32_view(&n, 0, 5120), f32_view(&y, token * 5120, 5120), 1e-6).unwrap()
                    .submit_for_plan(&stream).unwrap();
            }
            let mut result = vec![0; input.len()];
            y.copy_to_host_sync(&stream, &mut result).unwrap();
            result
        };
        for name in ["alpha.bf16", "beta.bf16"] {
            let weight = std::fs::read(root.join(name)).unwrap();
            for tokens in [1, 7, 8, 9, 31, 512] {
                compare(&normalized[..tokens * 5120 * 4], &weight, tokens, 48, 5120);
            }
        }
    }

    #[test]
    #[ignore = "requires exclusive SM70 GPU"]
    fn canonical_gdn_rows_gpu_stress_tail_geometry() {
        for (rows, cols) in [(1, 1), (7, 31), (48, 5120), (49, 513)] {
            let weight: Vec<_> = (0..rows * cols).flat_map(|i| {
                let v = (((i * 71) % 513) as f32 - 256.0) / 128.0;
                ((v.to_bits() >> 16) as u16).to_le_bytes()
            }).collect();
            for tokens in [1, 7, 8, 9, 31, 512] {
                let input: Vec<_> = (0..tokens * cols).flat_map(|i| {
                    let v = (((i * 97) % 1023) as f32 - 511.0) / 257.0;
                    (if i % 19 == 0 { v * 1e-20 } else { v }).to_le_bytes()
                }).collect();
                compare(&input, &weight, tokens, rows, cols);
            }
        }
    }

    #[test]
    #[ignore = "requires exclusive SM70 GPU"]
    fn canonical_gdn_rows_gpu_native_bad_arguments() {
        let stream = OwnedCudaStream::create(0).unwrap();
        let x = OwnedCudaBuffer::allocate(0, 8 * 5120 * 4).unwrap();
        let w = OwnedCudaBuffer::allocate(0, 48 * 5120 * 2).unwrap();
        let y = OwnedCudaBuffer::allocate(0, 8 * 48 * 4).unwrap();
        let weights = CudaBf16WeightViewV1::new(w.view(0, 48 * 5120 * 2).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1, 48, 5120).unwrap();
        let batch = CudaBf16F32MatvecRowsArgsV1::bind(&stream, weights,
            f32_view(&x, 0, 8 * 5120), f32_view(&y, 0, 8 * 48), 8).unwrap();
        assert!(CudaBf16F32MatvecRowsArgsV1::bind(&stream, weights,
            f32_view(&x, 0, 8 * 5120), f32_view(&x, 0, 8 * 48), 8).is_err());
        assert!(CudaBf16F32MatvecRowsArgsV1::bind(&stream, weights,
            f32_view(&x, 0, 7 * 5120), f32_view(&y, 0, 8 * 48), 8).is_err());
        for case in 0..18 {
            let mut raw = batch.raw;
            let mut tokens = 8;
            match case {
                0 => tokens = 0, 1 => tokens = 513, 2 => tokens = u32::MAX,
                3 => raw.input_len_bytes -= 4, 4 => raw.output_len_bytes += 4,
                5 => raw.weight_len_bytes += 2, 6 => raw.input = u64::MAX - 3,
                7 => raw.output = raw.input, 8 => raw.output = raw.weight,
                9 => raw.input += 1, 10 => raw.weight = 0,
                11 => raw.rows = 0, 12 => raw.rows = u32::MAX,
                13 => raw.cols = 0, 14 => raw.cols = u32::MAX,
                15 => raw.struct_size = 0, 16 => raw.abi_version = 0,
                _ => raw.weight += 1,
            }
            // SAFETY: invalid arguments must reject before a device dereference.
            let result = unsafe { augur_cuda_sys::augur_cuda_bf16_f32_matvec_rows_submit_v1(
                0, stream.raw().unwrap(), &raw, tokens) };
            assert_eq!(result.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT, "case{case}");
        }
        stream.synchronize().unwrap();
    }

}


#[cfg(test)]
mod nerkyor_gate_pair_tests {
    use super::*;
    use crate::OwnedCudaBuffer;
    fn f32_view(b: &OwnedCudaBuffer, offset: usize, n: usize) -> CudaF32View {
        CudaF32View::new(b.view(offset*4,n*4).unwrap(),PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,n).unwrap()
    }
    #[test]
    #[ignore = "requires exclusive SM70 GPU; isolated gate ABI/alias/stream contract"]
    fn nerkyor_gate_pair_gpu_negative_contracts() {
        let before = crate::cuda_device_allocation_snapshot();
        {
            let stream=OwnedCudaStream::create(0).unwrap();
            let foreign=OwnedCudaStream::create(0).unwrap();
            let w0=OwnedCudaBuffer::allocate(0,48*5120*2).unwrap();
            let w1=OwnedCudaBuffer::allocate(0,48*5120*2).unwrap();
            let x=OwnedCudaBuffer::allocate(0,5120*4).unwrap();
            let y0=OwnedCudaBuffer::allocate(0,48*4).unwrap();
            let y1=OwnedCudaBuffer::allocate(0,48*4).unwrap();
            let weight_bytes:Vec<u8>=(0..48*5120).flat_map(|_|0x3f80_u16.to_le_bytes()).collect();
            for w in [&w0,&w1] {w.copy_from_host_sync(&stream,&weight_bytes).unwrap();}
            x.copy_from_host_sync(&stream,&vec![1.0_f32;5120].iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
            let sentinel:Vec<u8>=vec![-77.0_f32;48].iter().flat_map(|v|v.to_le_bytes()).collect();
            for y in [&y0,&y1] {y.copy_from_host_sync(&stream,&sentinel).unwrap();}
            let weight=|w:&OwnedCudaBuffer| CudaBf16WeightViewV1::new(w.view(0,48*5120*2).unwrap(),PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,48,5120).unwrap();
            let a=CudaBf16F32MatvecArgsV1::bind(&stream,weight(&w0),f32_view(&x,0,5120),f32_view(&y0,0,48)).unwrap();
            let b=CudaBf16F32MatvecArgsV1::bind(&stream,weight(&w1),f32_view(&x,0,5120),f32_view(&y1,0,48)).unwrap();
            let pair=CudaNerkyorBf16GatePairF32CtaArgsV1::bind(&stream,a,b).unwrap();
            assert!(pair.submit_for_plan(&foreign).is_err());
            for case in 0..9 {
                let mut invalid=b;
                match case {
                    0=>invalid.raw.output=a.raw.output,
                    1=>invalid.raw.weight=a.raw.weight,
                    2=>invalid.raw.output=a.raw.input,
                    3=>invalid.raw.rows=47,
                    4=>invalid.raw.cols=5119,
                    5=>invalid.raw.input+=4,
                    6=>invalid.tokens=2,
                    7=>invalid.ordinal=1,
                    _=>invalid.raw.weight=u64::MAX-1,
                }
                assert!(CudaNerkyorBf16GatePairF32CtaArgsV1::bind(&stream,a,invalid).is_err(),"bind case {case}");
            }
            for case in 0..24 {
                let mut aa=a.raw;let mut bb=b.raw;
                match case {
                    0=>aa.abi_version=0,1=>bb.struct_size=0,
                    2=>aa.rows=47,3=>bb.cols=5119,
                    4=>aa.weight=0,5=>bb.input=0,6=>aa.output=0,
                    7=>aa.weight+=1,8=>aa.input+=1,9=>bb.output+=1,
                    10=>aa.weight_len_bytes-=2,11=>aa.input_len_bytes-=4,
                    12=>bb.output_len_bytes+=4,13=>aa.input=u64::MAX-3,
                    14=>bb.output=u64::MAX-3,15=>aa.weight=u64::MAX-1,
                    16=>bb.input+=4,17=>bb.output=aa.output,
                    18=>bb.weight=aa.weight,19=>bb.output=aa.input,
                    20=>aa.output=bb.weight,21=>aa.input=aa.weight,
                    22=>bb.weight=aa.weight+2,_=>bb.output=aa.output+4,
                }
                // Invalid extents/identities reject before any device access.
                let result=unsafe {augur_cuda_sys::augur_cuda_nerkyor_bf16_gate_pair_f32_cta_submit_v1(0,stream.raw().unwrap(),&aa,&bb)};
                assert_eq!(result.status,augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT,"native case {case}");
            }
            for y in [&y0,&y1] {
                let mut result=vec![0;48*4];y.copy_to_host_sync(&stream,&mut result).unwrap();assert_eq!(result,sentinel);
            }
            let allocated=crate::cuda_device_allocation_snapshot().successful_allocations;
            pair.submit_for_plan(&stream).unwrap();stream.synchronize().unwrap();
            assert_eq!(allocated,crate::cuda_device_allocation_snapshot().successful_allocations);
            for y in [&y0,&y1] {
                let mut result=vec![0;48*4];y.copy_to_host_sync(&stream,&mut result).unwrap();
                assert!(result.chunks_exact(4).all(|b|f32::from_le_bytes(b.try_into().unwrap())==5120.0));
            }
        }
        let after=crate::cuda_device_allocation_snapshot();
        assert_eq!(before.live_bytes,after.live_bytes);assert_eq!(before.live_allocations,after.live_allocations);
    }

}
