#![cfg(feature = "qwen35-fused-head-norm-rope")]

//! Checked Qwen3.8 full-attention head normalization and RoPE fusion.

use crate::{
    CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream,
    QWEN35_FULL_ATTENTION_KV_ELEMENTS, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS,
    QWEN35_FULL_ATTENTION_ROTARY_DIM,
};
use augur_cuda_sys::AugurCudaQwen35FusedHeadNormRopeArgsV1;

const OPERATION: &str = "cuda_qwen35_fused_head_norm_rope_v1";

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35FusedHeadNormRopeViewsV1 {
    pub q_input: CudaF32View,
    pub q_weight: CudaF32View,
    pub q_normed: CudaF32View,
    pub q_output: CudaF32View,
    pub k_input: CudaF32View,
    pub k_weight: CudaF32View,
    pub k_normed: CudaF32View,
    pub k_output: CudaF32View,
    pub rope_cos: CudaF32View,
    pub rope_sin: CudaF32View,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35FusedHeadNormRopeArgsV1 {
    raw: AugurCudaQwen35FusedHeadNormRopeArgsV1,
    ordinal: i32,
}

impl CudaQwen35FusedHeadNormRopeArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        views: CudaQwen35FusedHeadNormRopeViewsV1,
        epsilon: f32,
    ) -> Result<Self, CudaRuntimeError> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        let rope_elements = QWEN35_FULL_ATTENTION_ROTARY_DIM / 2;
        let expected = [
            (views.q_input, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS),
            (views.q_weight, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS),
            (views.q_normed, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS),
            (views.q_output, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS),
            (views.k_input, QWEN35_FULL_ATTENTION_KV_ELEMENTS),
            (views.k_weight, QWEN35_FULL_ATTENTION_KV_ELEMENTS),
            (views.k_normed, QWEN35_FULL_ATTENTION_KV_ELEMENTS),
            (views.k_output, QWEN35_FULL_ATTENTION_KV_ELEMENTS),
            (views.rope_cos, rope_elements),
            (views.rope_sin, rope_elements),
        ];
        let ordinal = stream.device_ordinal();
        for (view, elements) in expected {
            if view.buffer_view().device_ordinal() != ordinal
                || view.element_count() != elements
                || view.buffer_view().checked_address(OPERATION)?
                    % core::mem::align_of::<f32>() as u64
                    != 0
            {
                return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
            }
            view.buffer_view().checked_range(OPERATION)?;
        }

        let buffers = [
            views.q_input.buffer_view(),
            views.q_weight.buffer_view(),
            views.q_normed.buffer_view(),
            views.q_output.buffer_view(),
            views.k_input.buffer_view(),
            views.k_weight.buffer_view(),
            views.k_normed.buffer_view(),
            views.k_output.buffer_view(),
            views.rope_cos.buffer_view(),
            views.rope_sin.buffer_view(),
        ];
        for (index, left) in buffers.iter().enumerate() {
            for right in buffers.iter().skip(index + 1) {
                if left.overlaps(*right, OPERATION)? {
                    return Err(validation(CudaRuntimeErrorKind::OverlappingDeviceRanges));
                }
            }
        }

        let mut raw = AugurCudaQwen35FusedHeadNormRopeArgsV1::new();
        macro_rules! bind_view {
            ($ptr:ident, $len:ident, $view:expr) => {{
                let view = $view.buffer_view();
                raw.$ptr = view.checked_address(OPERATION)?;
                raw.$len = u64::try_from(view.len_bytes())
                    .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
            }};
        }
        bind_view!(q_input_f32, q_input_len_bytes, views.q_input);
        bind_view!(q_weight_f32, q_weight_len_bytes, views.q_weight);
        bind_view!(q_normed_f32, q_normed_len_bytes, views.q_normed);
        bind_view!(q_output_f32, q_output_len_bytes, views.q_output);
        bind_view!(k_input_f32, k_input_len_bytes, views.k_input);
        bind_view!(k_weight_f32, k_weight_len_bytes, views.k_weight);
        bind_view!(k_normed_f32, k_normed_len_bytes, views.k_normed);
        bind_view!(k_output_f32, k_output_len_bytes, views.k_output);
        bind_view!(rope_cos_f32, rope_cos_len_bytes, views.rope_cos);
        bind_view!(rope_sin_f32, rope_sin_len_bytes, views.rope_sin);
        raw.epsilon_bits = epsilon.to_bits();
        Ok(Self { raw, ordinal })
    }

    pub const fn as_raw(&self) -> &AugurCudaQwen35FusedHeadNormRopeArgsV1 {
        &self.raw
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: bind proves exact device extents, alignment, and disjoint
        // ownership. The plan retains every allocation until terminal proof.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_fused_head_norm_rope_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_qwen35_fused_head_norm_rope_submit_v1", result)
    }

    pub fn launch_sync(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: same bind-established contract as submit_for_plan.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_fused_head_norm_rope_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        Ok(())
    }
}

fn validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OPERATION, kind)
}

const BATCH_OPERATION: &str = "cuda_qwen35_batched_head_norm_rope_v1";

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35BatchedHeadNormRopeViewsV1 {
    pub q_gate: CudaF32View,
    pub q_weight: CudaF32View,
    pub query: CudaF32View,
    pub key_input: CudaF32View,
    pub k_weight: CudaF32View,
    pub key: CudaF32View,
    pub rope_cos: CudaF32View,
    pub rope_sin: CudaF32View,
    pub gate: CudaF32View,
    pub value_input: CudaF32View,
    pub value: CudaF32View,
}

fn batch_elements(rows: usize, epsilon: f32) -> Result<[usize; 11], CudaRuntimeError> {
    if !(1..=512).contains(&rows) || !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(CudaRuntimeError::validation(BATCH_OPERATION, CudaRuntimeErrorKind::InvalidMetadata));
    }
    let q = QWEN35_FULL_ATTENTION_QUERY_ELEMENTS;
    let k = QWEN35_FULL_ATTENTION_KV_ELEMENTS;
    let rope = rows * (QWEN35_FULL_ATTENTION_ROTARY_DIM / 2);
    Ok([2*rows*q, q, rows*q, rows*k, k, rows*k, rope, rope, rows*q, rows*k, rows*k])
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35BatchedHeadNormRopeArgsV1 {
    raw: augur_cuda_sys::AugurCudaQwen35BatchedHeadNormRopeArgsV1,
    ordinal: i32,
    stream_owner: usize,
}

impl CudaQwen35BatchedHeadNormRopeArgsV1 {
    pub fn bind(stream: &OwnedCudaStream, views: CudaQwen35BatchedHeadNormRopeViewsV1,
                rows: usize, epsilon: f32) -> Result<Self, CudaRuntimeError> {
        let expected = batch_elements(rows, epsilon)?;
        let views = [views.q_gate, views.q_weight, views.query, views.key_input, views.k_weight,
            views.key, views.rope_cos, views.rope_sin, views.gate, views.value_input, views.value];
        let ordinal = stream.device_ordinal();
        let mut raw = augur_cuda_sys::AugurCudaQwen35BatchedHeadNormRopeArgsV1 {
            abi_version: 1, struct_size: 200, rows: rows as u32, epsilon_bits: epsilon.to_bits(),
            reserved0: 0, reserved1: 0, addresses: [0;11], lengths: [0;11],
        };
        for (i, view) in views.iter().enumerate() {
            let buffer = view.buffer_view();
            if buffer.device_ordinal() != ordinal || view.element_count() != expected[i]
                || buffer.len_bytes() != expected[i] * 4
                || buffer.checked_address(BATCH_OPERATION)? % core::mem::align_of::<f32>() as u64 != 0 {
                return Err(CudaRuntimeError::validation(BATCH_OPERATION, CudaRuntimeErrorKind::InvalidDimensions));
            }
            buffer.checked_range(BATCH_OPERATION)?;
            for other in views.iter().skip(i + 1) {
                if buffer.overlaps(other.buffer_view(), BATCH_OPERATION)? {
                    return Err(CudaRuntimeError::validation(BATCH_OPERATION, CudaRuntimeErrorKind::OverlappingDeviceRanges));
                }
            }
            raw.addresses[i] = buffer.checked_address(BATCH_OPERATION)?;
            raw.lengths[i] = u64::try_from(buffer.len_bytes())
                .map_err(|_| CudaRuntimeError::validation(BATCH_OPERATION, CudaRuntimeErrorKind::RangeOverflow))?;
        }
        Ok(Self { raw, ordinal, stream_owner: stream.owner_instance_key()? })
    }

    pub const fn as_raw(&self) -> &augur_cuda_sys::AugurCudaQwen35BatchedHeadNormRopeArgsV1 { &self.raw }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal || stream.owner_instance_key()? != self.stream_owner {
            return Err(CudaRuntimeError::validation(BATCH_OPERATION, CudaRuntimeErrorKind::DeviceMismatch));
        }
        // SAFETY: bind checks exact extents, device, alignment and disjoint ranges;
        // the caller retains all allocations through completion on this stream.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_batched_head_norm_rope_submit_v1(
                self.ordinal, stream.raw()?, &self.raw)
        };
        CudaRuntimeError::from_result(BATCH_OPERATION, result)
    }
}

#[cfg(test)]
mod batched_head_rope_tests {
    use super::*;
    #[test]
    fn batched_head_rope_exact_extents_cover_tails_and_full_batches() {
        for rows in [1, 8, 9, 63, 64, 255, 256, 285, 511, 512] {
            let e = batch_elements(rows, 1e-6).unwrap();
            assert_eq!(e[0], rows * 12288);
            assert_eq!(e[1], 6144);
            assert_eq!(e[2], rows * 6144);
            assert_eq!(e[3], rows * 1024);
            assert_eq!(e[4], 1024);
            assert_eq!(e[6], rows * (QWEN35_FULL_ATTENTION_ROTARY_DIM / 2));
            assert_eq!(e[8], e[2]);
            assert_eq!(e[9], e[10]);
        }
    }
    #[test]
    fn batched_head_rope_rejects_invalid_shapes_and_epsilon() {
        for rows in [0, 513, usize::MAX] { assert!(batch_elements(rows, 1e-6).is_err()); }
        for epsilon in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(batch_elements(512, epsilon).is_err());
        }
    }

    #[test]
    #[ignore = "requires exclusive CUDA GPU; component bitwise oracle, not HTTP performance"]
    fn batched_head_rope_gpu_matches_scalar_and_copies() {
        use crate::{OwnedCudaBuffer, cuda_device_allocation_snapshot};
        use augur_core::PhysicalLayoutId;
        fn view(b: &OwnedCudaBuffer, offset: usize, n: usize) -> CudaF32View {
            CudaF32View::new(b.view(offset*4, n*4).unwrap(),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, n).unwrap()
        }
        fn bytes(n: usize, seed: u32) -> Vec<u8> {
            (0..n).flat_map(|i| {
                let u = (i as u32).wrapping_mul(1664525).wrapping_add(seed);
                let x = ((u % 4093) as i32 - 2046) as f32 * (1.0 / 512.0);
                x.to_le_bytes()
            }).collect()
        }
        let before = cuda_device_allocation_snapshot();
        {
            let stream = OwnedCudaStream::create(0).unwrap();
            let foreign = OwnedCudaStream::create(0).unwrap();
            for rows in [1, 8, 9, 63, 64, 255, 256, 285, 511, 512] {
                let counts = batch_elements(rows, 1e-6).unwrap();
                let owners: Vec<_> = counts.iter().map(|n| OwnedCudaBuffer::allocate(0, n*4).unwrap()).collect();
                let mut host: Vec<Vec<u8>> = counts.iter().enumerate().map(|(i,n)| bytes(*n, 97+i as u32*31)).collect();
                // Cover both signed zeros and tiny finite inputs without changing the reference contract.
                for i in [0, 3, 9] {
                    for (j, bits) in [0u32, 0x80000000, 1, 0x80000001, 0x00800000].iter().enumerate() {
                        host[i][j*4..j*4+4].copy_from_slice(&bits.to_le_bytes());
                    }
                }
                for (b, h) in owners.iter().zip(&host) { b.copy_from_host_sync(&stream,h).unwrap(); }
                let v: Vec<_> = owners.iter().zip(counts).map(|(b,n)| view(b,0,n)).collect();
                let batch = CudaQwen35BatchedHeadNormRopeArgsV1::bind(&stream,
                    CudaQwen35BatchedHeadNormRopeViewsV1 { q_gate:v[0], q_weight:v[1], query:v[2],
                        key_input:v[3], k_weight:v[4], key:v[5], rope_cos:v[6], rope_sin:v[7],
                        gate:v[8], value_input:v[9], value:v[10] }, rows, 1e-6).unwrap();
                assert!(batch.submit_for_plan(&foreign).is_err());
                let qn=OwnedCudaBuffer::allocate(0,6144*4).unwrap();
                let kn=OwnedCudaBuffer::allocate(0,1024*4).unwrap();
                let qo=OwnedCudaBuffer::allocate(0,rows*6144*4).unwrap();
                let ko=OwnedCudaBuffer::allocate(0,rows*1024*4).unwrap();
                let scalar: Vec<_> = (0..rows).map(|row| CudaQwen35FusedHeadNormRopeArgsV1::bind(
                    &stream,CudaQwen35FusedHeadNormRopeViewsV1 {
                        q_input:view(&owners[0],row*12288,6144), q_weight:v[1],
                        q_normed:view(&qn,0,6144), q_output:view(&qo,row*6144,6144),
                        k_input:view(&owners[3],row*1024,1024), k_weight:v[4],
                        k_normed:view(&kn,0,1024), k_output:view(&ko,row*1024,1024),
                        rope_cos:view(&owners[6],row*(QWEN35_FULL_ATTENTION_ROTARY_DIM/2),QWEN35_FULL_ATTENTION_ROTARY_DIM/2),
                        rope_sin:view(&owners[7],row*(QWEN35_FULL_ATTENTION_ROTARY_DIM/2),QWEN35_FULL_ATTENTION_ROTARY_DIM/2),
                    },1e-6).unwrap()).collect();
                for repeat in 0..3 {
                    if repeat%2==0 { batch.submit_for_plan(&stream).unwrap(); }
                    for op in &scalar { op.submit_for_plan(&stream).unwrap(); }
                    if repeat%2==1 { batch.submit_for_plan(&stream).unwrap(); }
                    for (index,reference) in [(2,&qo),(5,&ko)] {
                        let mut actual=vec![0u8;counts[index]*4];
                        let mut expected=actual.clone();
                        owners[index].copy_to_host_sync(&stream,&mut actual).unwrap();
                        reference.copy_to_host_sync(&stream,&mut expected).unwrap();
                        assert_eq!(actual,expected,"rows={rows}, slot={index}, repeat={repeat}");
                    }
                    let mut gate=vec![0u8;counts[8]*4];
                    owners[8].copy_to_host_sync(&stream,&mut gate).unwrap();
                    for row in 0..rows { assert_eq!(&gate[row*6144*4..(row+1)*6144*4],
                        &host[0][(row*12288+6144)*4..(row+1)*12288*4]); }
                    let mut value=vec![0u8;counts[10]*4];
                    owners[10].copy_to_host_sync(&stream,&mut value).unwrap();
                    assert_eq!(value,host[9]);
                }
                for index in [0,1,3,4,6,7,9] {
                    let mut actual=vec![0u8;counts[index]*4];
                    owners[index].copy_to_host_sync(&stream,&mut actual).unwrap();
                    assert_eq!(actual,host[index],"input mutated: {index}");
                }
                let raw=*batch.as_raw();
                let mut invalid=Vec::new();
                let mut bad=raw;bad.rows=0;invalid.push(bad);
                let mut bad=raw;bad.rows=513;invalid.push(bad);
                let mut bad=raw;bad.struct_size-=1;invalid.push(bad);
                let mut bad=raw;bad.reserved0=1;invalid.push(bad);
                let mut bad=raw;bad.epsilon_bits=f32::NAN.to_bits();invalid.push(bad);
                let mut bad=raw;bad.lengths[2]-=4;invalid.push(bad);
                let mut bad=raw;bad.addresses[2]+=1;invalid.push(bad);
                let mut bad=raw;bad.addresses[2]=raw.addresses[0];invalid.push(bad);
                for bad in invalid { // Native rejection must occur before any device submission.
                    let result=unsafe { augur_cuda_sys::augur_cuda_qwen35_batched_head_norm_rope_submit_v1(0,stream.raw().unwrap(),&bad) };
                    assert_ne!(result.status,0);
                }
                stream.synchronize().unwrap();
                println!("BATCHED_HEAD_ROPE_BITWISE_PASS rows={rows} repeats=3 invalid=8");
            }
        }
        let after=cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes,before.live_bytes);
        assert_eq!(after.live_allocations,before.live_allocations);
        assert_eq!(after.successful_allocations-before.successful_allocations,
            after.successful_frees-before.successful_frees);
        println!("BATCHED_HEAD_ROPE_CLEANUP_PASS allocations={} frees={} live={}",
            after.successful_allocations-before.successful_allocations,
            after.successful_frees-before.successful_frees,after.live_bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_geometry_matches_qwen_full_attention() {
        assert_eq!(crate::QWEN35_FULL_ATTENTION_QUERY_HEADS, 24);
        assert_eq!(crate::QWEN35_FULL_ATTENTION_KV_HEADS, 4);
        assert_eq!(crate::QWEN35_FULL_ATTENTION_HEAD_DIM, 256);
        assert_eq!(QWEN35_FULL_ATTENTION_ROTARY_DIM, 64);
        assert_eq!(QWEN35_FULL_ATTENTION_QUERY_ELEMENTS, 6_144);
        assert_eq!(QWEN35_FULL_ATTENTION_KV_ELEMENTS, 1_024);
    }
}
