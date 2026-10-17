//! Default-off fused Qwen SiLU(gate) * up -> Q8_1 activation binding.

use augur_cuda_sys::AugurCudaSiluMulQ8_1ArgsV1;

use crate::{
    CudaBufferView, CudaF32View, CudaQ8_1ActivationViewV1, CudaRuntimeError, CudaRuntimeErrorKind,
    OwnedCudaStream,
};

const OPERATION: &str = "cuda_silu_mul_q8_1_v1";

#[derive(Clone, Copy, Debug)]
pub struct CudaSiluMulQ8_1ArgsV1 {
    raw: AugurCudaSiluMulQ8_1ArgsV1,
    device_ordinal: i32,
}

impl CudaSiluMulQ8_1ArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        gate: CudaF32View,
        up: CudaF32View,
        output: CudaQ8_1ActivationViewV1,
    ) -> Result<Self, CudaRuntimeError> {
        let ordinal = stream.device_ordinal();
        let gate_view = gate.buffer_view();
        let up_view = up.buffer_view();
        let output_view = output.buffer_view();
        if gate_view.device_ordinal() != ordinal
            || up_view.device_ordinal() != ordinal
            || output_view.device_ordinal() != ordinal
        {
            return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let elements = gate.element_count();
        if elements == 0
            || !elements.is_multiple_of(crate::Q8_1_ACTIVATION_BLOCK_ELEMENTS)
            || up.element_count() != elements
            || output.elements() != elements
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if overlaps(gate_view, up_view)?
            || overlaps(gate_view, output_view)?
            || overlaps(up_view, output_view)?
        {
            return Err(validation(CudaRuntimeErrorKind::OverlappingDeviceRanges));
        }
        let output_bytes = (elements / crate::Q8_1_ACTIVATION_BLOCK_ELEMENTS)
            .checked_mul(crate::Q8_1_ACTIVATION_BLOCK_BYTES)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        if output_view.len_bytes() != output_bytes {
            return Err(validation(CudaRuntimeErrorKind::OutOfBounds));
        }
        let mut raw = AugurCudaSiluMulQ8_1ArgsV1::new();
        raw.gate = address(gate_view)?;
        raw.gate_len_bytes = to_u64(gate_view.len_bytes())?;
        raw.up = address(up_view)?;
        raw.up_len_bytes = to_u64(up_view.len_bytes())?;
        raw.output_q8_1 = address(output_view)?;
        raw.output_len_bytes = to_u64(output_view.len_bytes())?;
        raw.elements = u32::try_from(elements)
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn as_raw(&self) -> &AugurCudaSiluMulQ8_1ArgsV1 {
        &self.raw
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        let result = unsafe {
            augur_cuda_sys::augur_cuda_silu_mul_q8_1_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        Ok(())
    }
}

fn validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OPERATION, kind)
}

fn to_u64(value: usize) -> Result<u64, CudaRuntimeError> {
    u64::try_from(value).map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))
}

fn address(view: CudaBufferView) -> Result<u64, CudaRuntimeError> {
    view.device_ptr()
        .raw_value()
        .checked_add(to_u64(view.offset_bytes())?)
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))
}

fn overlaps(a: CudaBufferView, b: CudaBufferView) -> Result<bool, CudaRuntimeError> {
    if a.device_ptr() != b.device_ptr() {
        return Ok(false);
    }
    let a_start = a.offset_bytes();
    let a_end = a_start
        .checked_add(a.len_bytes())
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    let b_start = b.offset_bytes();
    let b_end = b_start
        .checked_add(b.len_bytes())
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    Ok(a_start < b_end && b_start < a_end)
}

/// Exact full-B8 verify boundary. Both existing output buffers remain live and
/// observable; this binding adds no allocation and never caches input contents.
#[derive(Debug)]
pub struct CudaSiluQ8PreservedB8V1 {
    silu: augur_cuda_sys::AugurCudaSiluMulF32ArgsV1,
    quantize: augur_cuda_sys::AugurCudaQ8_1ActivationQuantizeArgsV1,
    device_ordinal: i32,
    stream_owner: usize,
    submissions: core::cell::Cell<u64>,
}

impl CudaSiluQ8PreservedB8V1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        silu: &crate::CudaSiluMulF32ArgsV1,
        quantize: &crate::CudaQ8_1ActivationQuantizeArgsV1,
    ) -> Result<Self, CudaRuntimeError> {
        let s = *silu.as_raw();
        let q = *quantize.as_raw();
        validate_preserved_pair(&s, &q)?;
        let owner = stream.owner_instance_key()?;
        // SAFETY: existing typed bindings describe live allocations retained by
        // the workspace. Native preflight checks their exact ranges and device.
        CudaRuntimeError::from_result(OPERATION, unsafe {
            augur_cuda_sys::augur_cuda_silu_q8_preserved_b8_preflight_v1(
                stream.device_ordinal(), stream.raw()?, &s, &q)
        })?;
        Ok(Self { silu: s, quantize: q, device_ordinal: stream.device_ordinal(),
            stream_owner: owner, submissions: core::cell::Cell::new(0) })
    }

    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal || stream.owner_instance_key()? != self.stream_owner {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        let next = self.submissions.get().checked_add(1)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        // SAFETY: preflight bound this exact owner and all four immutable ranges;
        // the caller retains the buffers until covering stream completion.
        CudaRuntimeError::from_result(OPERATION, unsafe {
            augur_cuda_sys::augur_cuda_silu_q8_preserved_b8_submit_v1(
                self.device_ordinal, stream.raw()?, &self.silu, &self.quantize)
        })?;
        self.submissions.set(next);
        Ok(())
    }

    pub fn eager_submissions(&self) -> u64 { self.submissions.get() }
}

fn validate_preserved_pair(
    s: &augur_cuda_sys::AugurCudaSiluMulF32ArgsV1,
    q: &augur_cuda_sys::AugurCudaQ8_1ActivationQuantizeArgsV1,
) -> Result<(), CudaRuntimeError> {
    const N: u32 = 8 * 17408;
    const F: u64 = N as u64 * 4;
    const Q: u64 = N as u64 / 32 * 36;
    if s.abi_version != 1 || s.struct_size as usize != core::mem::size_of_val(s)
        || q.abi_version != 1 || q.struct_size as usize != core::mem::size_of_val(q)
        || s.reserved != 0 || q.reserved != 0 || s.elements != N || q.elements != N
        || s.gate_len_bytes != F || s.up_len_bytes != F || s.output_len_bytes != F
        || q.input_f32 != s.output || q.input_len_bytes != F || q.output_len_bytes != Q {
        return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
    }
    let ranges = [(s.gate, F), (s.up, F), (s.output, F), (q.output_q8_1, Q)];
    for (i, &(ptr, len)) in ranges.iter().enumerate() {
        let end = ptr.checked_add(len).ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        if ptr == 0 || ptr % 4 != 0 { return Err(validation(CudaRuntimeErrorKind::InvalidMetadata)); }
        for &(other, bytes) in &ranges[..i] {
            if ptr < other + bytes && other < end {
                return Err(validation(CudaRuntimeErrorKind::OverlappingDeviceRanges));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod preserved_tests {
    use super::*;
    use crate::{CudaQ8_1ActivationQuantizeArgsV1, CudaSiluMulF32ArgsV1, OwnedCudaBuffer};
    use augur_core::PhysicalLayoutId as Layout;
    const N: usize = 8 * 17408;

    fn fixture() -> (augur_cuda_sys::AugurCudaSiluMulF32ArgsV1, augur_cuda_sys::AugurCudaQ8_1ActivationQuantizeArgsV1) {
        let mut s = augur_cuda_sys::AugurCudaSiluMulF32ArgsV1::new();
        s.gate = 0x100000; s.up = 0x200000; s.output = 0x300000;
        s.gate_len_bytes = (N * 4) as u64; s.up_len_bytes = s.gate_len_bytes; s.output_len_bytes = s.gate_len_bytes;
        s.elements = N as u32;
        let mut q = augur_cuda_sys::AugurCudaQ8_1ActivationQuantizeArgsV1::new();
        q.input_f32 = s.output; q.input_len_bytes = s.output_len_bytes;
        q.output_q8_1 = 0x400000; q.output_len_bytes = (N / 32 * 36) as u64; q.elements = N as u32;
        (s, q)
    }

    fn mutate(s: &mut augur_cuda_sys::AugurCudaSiluMulF32ArgsV1,
        q: &mut augur_cuda_sys::AugurCudaQ8_1ActivationQuantizeArgsV1, i: usize) {
        match i {
            0 => s.abi_version = 0, 1 => s.struct_size = 0, 2 => s.reserved = 1,
            3 => q.abi_version = 0, 4 => q.struct_size = 0, 5 => q.reserved = 1,
            6 => s.elements -= 32, 7 => q.elements -= 32, 8 => s.elements = 0,
            9 => s.gate_len_bytes -= 1, 10 => s.up_len_bytes += 1, 11 => s.output_len_bytes -= 4,
            12 => q.input_len_bytes += 4, 13 => q.output_len_bytes -= 1,
            14 => q.input_f32 += 4, 15 => s.gate = 0, 16 => s.up += 2,
            17 => { s.output += 2; q.input_f32 = s.output; }, 18 => q.output_q8_1 += 2,
            19 => s.up = s.gate, 20 => { s.output = s.gate; q.input_f32 = s.output; },
            21 => q.output_q8_1 = s.up, 22 => q.output_q8_1 = s.output,
            23 => s.gate = u64::MAX - 3, 24 => q.output_q8_1 = s.gate + 4,
            25 => { s.elements = 7 * 17408; q.elements = s.elements; },
            26 => { s.elements = 512 * 17408; q.elements = s.elements; },
            _ => unreachable!(),
        }
    }

    #[test]
    fn preserved_silu_q8_exact_extent_consumer_and_alias_contract() {
        let (s, q) = fixture(); validate_preserved_pair(&s, &q).unwrap();
        for i in 0..27 { let (mut s, mut q) = fixture(); mutate(&mut s, &mut q, i);
            assert!(validate_preserved_pair(&s, &q).is_err(), "mutation {i}"); }
    }

    #[test]
    fn preserved_silu_q8_null_native_rejects_before_cuda() {
        for preflight in [true, false] {
            // SAFETY: null stream/args reject before CUDA or dereferencing args.
            let result = unsafe {
                let call = if preflight { augur_cuda_sys::augur_cuda_silu_q8_preserved_b8_preflight_v1 }
                    else { augur_cuda_sys::augur_cuda_silu_q8_preserved_b8_submit_v1 };
                call(0, core::ptr::null_mut(), core::ptr::null(), core::ptr::null())
            };
            assert_eq!(result.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
        }
    }

    struct Guard { owner: OwnedCudaBuffer, bytes: usize }
    impl Guard {
        fn new(bytes: usize) -> Self { Self { owner: OwnedCudaBuffer::allocate(0, bytes + 512).unwrap(), bytes } }
        fn f32(&self) -> CudaF32View { CudaF32View::new(self.owner.view(256, self.bytes).unwrap(),
            Layout::DEVICE_CONTIGUOUS_F32_CUDA_V1, self.bytes / 4).unwrap() }
        fn q8(&self) -> CudaQ8_1ActivationViewV1 { CudaQ8_1ActivationViewV1::new(self.owner.view(256, self.bytes).unwrap(),
            Layout::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1, N).unwrap() }
        fn set(&self, stream: &OwnedCudaStream, bytes: &[u8]) {
            assert_eq!(bytes.len(), self.bytes); let mut all = vec![0xa5; self.bytes + 512];
            all[256..256+self.bytes].copy_from_slice(bytes); self.owner.copy_from_host_sync(stream, &all).unwrap();
        }
        fn get(&self, stream: &OwnedCudaStream) -> Vec<u8> {
            let mut all = vec![0; self.bytes + 512]; self.owner.copy_to_host_sync(stream, &mut all).unwrap();
            assert!(all[..256].iter().chain(all[256+self.bytes..].iter()).all(|x| *x == 0xa5));
            all[256..256+self.bytes].to_vec()
        }
    }

    #[test]
    #[ignore = "exclusive SM70; AUGUR_QWEN35_BOUNDARY_CAPTURE_ROOT holds bound gate/up tensors"]
    fn preserved_silu_q8_real_native_exact_owner_guards_and_cleanup() {
        let root = std::path::PathBuf::from(std::env::var_os("AUGUR_QWEN35_BOUNDARY_CAPTURE_ROOT").expect("capture root"));
        let before = crate::cuda_device_allocation_snapshot(); let mut cases = 0; let mut rejected = 0;
        {
            let stream = OwnedCudaStream::create(0).unwrap(); let foreign = OwnedCudaStream::create(0).unwrap();
            for layer in [0, 31, 50, 51, 63] {
                let read = |name: &str| { let all = std::fs::read(root.join(format!("layer{layer}-{name}-full.f32le.bin"))).unwrap();
                    assert!(all.len() >= N * 4); all[..N*4].to_vec() };
                let mut hg = read("ffn_gate_scaled"); let mut hu = read("ffn_up_scaled");
                let g = Guard::new(N*4); let u = Guard::new(N*4); let y = Guard::new(N*4); let q = Guard::new(N/32*36);
                let silu = CudaSiluMulF32ArgsV1::bind(&stream, g.f32(), u.f32(), y.f32()).unwrap();
                let quant = CudaQ8_1ActivationQuantizeArgsV1::bind(&stream, y.f32(), q.q8()).unwrap();
                let allocated = crate::cuda_device_allocation_snapshot();
                let fused = CudaSiluQ8PreservedB8V1::bind(&stream, &silu, &quant).unwrap();
                assert_eq!(fused.eager_submissions(), 0);
                assert!(fused.submit(&foreign).is_err()); rejected += 1;
                fused.submissions.set(u64::MAX); assert!(fused.submit(&stream).is_err()); rejected += 1; fused.submissions.set(0);
                for i in 0..27 {
                    let mut s = *silu.as_raw(); let mut q = *quant.as_raw(); mutate(&mut s, &mut q, i);
                    for preflight in [true, false] {
                        // SAFETY: mutated metadata must reject before any kernel access.
                        let result = unsafe { let call = if preflight { augur_cuda_sys::augur_cuda_silu_q8_preserved_b8_preflight_v1 }
                            else { augur_cuda_sys::augur_cuda_silu_q8_preserved_b8_submit_v1 };
                            call(0, stream.raw().unwrap(), &s, &q) };
                        assert_eq!(result.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT, "mutation {i}"); rejected += 1;
                    }
                }
                // Metadata-valid extent crossing its actual allocation must fail preflight.
                let mut outside = *silu.as_raw(); outside.gate += 512;
                let result = unsafe { augur_cuda_sys::augur_cuda_silu_q8_preserved_b8_preflight_v1(
                    0, stream.raw().unwrap(), &outside, quant.as_raw()) };
                assert_eq!(result.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT); rejected += 1;
                for generation in 0..3 {
                    if generation == 1 {
                        for bytes in [&mut hg, &mut hu] { for (i, c) in bytes.chunks_exact_mut(4).enumerate() {
                            let f = f32::from_le_bytes(c.try_into().unwrap()); let f = if i%11==0 { -f } else { f * 0.97 };
                            c.copy_from_slice(&f.to_le_bytes()); } }
                    } else if generation == 2 {
                        for (i, c) in hg.chunks_exact_mut(4).enumerate() {
                            let f: f32 = match i%5 { 0 => -0.0, 1 => 0.0, 2 => 80.0, 3 => -80.0, _ => 1e-30 };
                            c.copy_from_slice(&f.to_le_bytes()); }
                    }
                    g.set(&stream, &hg); u.set(&stream, &hu);
                    // Initialize the reference destination guards before its
                    // first read as well as poisoning the candidate below.
                    y.set(&stream, &vec![0xff; y.bytes]); q.set(&stream, &vec![0xff; q.bytes]);
                    silu.submit_for_plan(&stream).unwrap(); quant.submit_for_plan(&stream).unwrap();
                    let expected_y = y.get(&stream); let expected_q = q.get(&stream);
                    y.set(&stream, &vec![0xff; y.bytes]); q.set(&stream, &vec![0xff; q.bytes]);
                    fused.submit(&stream).unwrap(); stream.synchronize().unwrap();
                    assert_eq!(y.get(&stream), expected_y, "layer {layer} generation {generation} F32");
                    assert_eq!(q.get(&stream), expected_q, "layer {layer} generation {generation} Q8");
                    assert_eq!(g.get(&stream), hg); assert_eq!(u.get(&stream), hu); cases += 1;
                }
                assert_eq!(fused.eager_submissions(), 3);
                let after = crate::cuda_device_allocation_snapshot();
                assert_eq!(allocated.successful_allocations, after.successful_allocations);
                assert_eq!(allocated.successful_frees, after.successful_frees);
                assert_eq!(allocated.live_bytes, after.live_bytes);
            }
        }
        let after = crate::cuda_device_allocation_snapshot();
        assert_eq!(after.successful_allocations - before.successful_allocations, 20);
        assert_eq!(after.successful_frees - before.successful_frees, 20);
        assert_eq!(after.live_bytes, before.live_bytes); assert_eq!(after.live_allocations, before.live_allocations);
        println!("preserved_silu_q8 cases={cases} native_and_owner_rejections={rejected} allocations=20 frees=20 live_delta=0");
    }
}
