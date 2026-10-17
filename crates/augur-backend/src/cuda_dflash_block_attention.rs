#![cfg(feature = "cuda")]
//! Eight-query noncausal DFlash2 attention over dense [history || draft] K/V.
//! The owner must initialize every row on the bound stream and retain all
//! buffers through completion. This is not the circular single-query cache.

use crate::{CudaBufferView, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};
use augur_cuda_sys::AugurCudaDflashBlockAttentionF32ArgsV1 as Raw;

const OP: &str = "cuda_dflash_block_attention_f32_submit_v1";
pub const DFLASH_BLOCK_ATTENTION_QUERY_ELEMENTS: usize = 8 * 32 * 128;
pub const DFLASH_BLOCK_ATTENTION_KV_ROW_ELEMENTS: usize = 8 * 128;

#[derive(Clone, Copy, Debug)]
pub struct CudaDflashBlockAttentionViewsV1 {
    pub query: CudaBufferView,
    pub key: CudaBufferView,
    pub value: CudaBufferView,
    pub output: CudaBufferView,
}

impl CudaDflashBlockAttentionViewsV1 {
    fn buffers(self) -> [CudaBufferView; 4] {
        [self.query, self.key, self.value, self.output]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaDflashBlockAttentionF32ArgsV1 {
    raw: Raw,
    ordinal: i32,
    stream_owner: usize,
}

impl CudaDflashBlockAttentionF32ArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        views: CudaDflashBlockAttentionViewsV1,
        context_len: usize,
        context_start: usize,
        block_start: usize,
        sliding_window: usize,
    ) -> Result<Self, CudaRuntimeError> {
        Ok(Self {
            raw: bind_raw(
                stream.device_ordinal(),
                views,
                context_len,
                context_start,
                block_start,
                sliding_window,
            )?,
            ordinal: stream.device_ordinal(),
            stream_owner: stream.owner_instance_key()?,
        })
    }

    /// Submission only: no allocation, host transfer, or host synchronization.
    /// Invalid arithmetic poisons the affected query/head with NaNs. Downstream
    /// consumers must reject nonfinite results before any token publication.
    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if self.ordinal != stream.device_ordinal() {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        if self.stream_owner != stream.owner_instance_key()? {
            return Err(error(CudaRuntimeErrorKind::InvalidMetadata));
        }
        // SAFETY: checked geometry, aligned disjoint ranges, device and stream;
        // caller owns buffer lifetime and initialization until completion.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_dflash_block_attention_f32_submit_v1(
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

fn bind_raw(
    ordinal: i32,
    views: CudaDflashBlockAttentionViewsV1,
    context_len: usize,
    context_start: usize,
    block_start: usize,
    sliding_window: usize,
) -> Result<Raw, CudaRuntimeError> {
    if ordinal < 0
        || context_len > 2048
        || !(1..=2048).contains(&sliding_window)
        || context_start.checked_add(context_len) != Some(block_start)
        || block_start.checked_add(8).is_none_or(|end| end > 262144)
    {
        return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
    }
    let kv_bytes = (context_len + 8) * DFLASH_BLOCK_ATTENTION_KV_ROW_ELEMENTS * 4;
    let lengths = [
        DFLASH_BLOCK_ATTENTION_QUERY_ELEMENTS * 4,
        kv_bytes,
        kv_bytes,
        DFLASH_BLOCK_ATTENTION_QUERY_ELEMENTS * 4,
    ];
    let buffers = views.buffers();
    let mut addresses = [0u64; 4];
    for (i, b) in buffers.iter().enumerate() {
        if b.device_ordinal() != ordinal {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        if b.len_bytes() != lengths[i] {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let p = b.checked_address(OP)?;
        if p == 0 || !p.is_multiple_of(4) {
            return Err(error(CudaRuntimeErrorKind::MisalignedDeviceAddress));
        }
        b.checked_range(OP)?;
        for other in &buffers[..i] {
            if b.overlaps(*other, OP)? {
                return Err(error(CudaRuntimeErrorKind::OverlappingDeviceRanges));
            }
        }
        addresses[i] = p;
    }
    Ok(Raw {
        abi_version: 1,
        struct_size: core::mem::size_of::<Raw>() as u32,
        query: addresses[0],
        query_len_bytes: lengths[0] as u64,
        key: addresses[1],
        key_len_bytes: lengths[1] as u64,
        value: addresses[2],
        value_len_bytes: lengths[2] as u64,
        output: addresses[3],
        output_len_bytes: lengths[3] as u64,
        context_len: context_len as u32,
        context_start: context_start as u32,
        block_start: block_start as u32,
        sliding_window: sliding_window as u32,
        block_size: 8,
        query_heads: 32,
        kv_heads: 8,
        head_dim: 128,
        scale: 128.0f32.sqrt().recip(),
        reserved: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_buffers(b: [CudaBufferView; 4]) -> CudaDflashBlockAttentionViewsV1 {
        CudaDflashBlockAttentionViewsV1 {
            query: b[0],
            key: b[1],
            value: b[2],
            output: b[3],
        }
    }
    fn fake(context: usize) -> CudaDflashBlockAttentionViewsV1 {
        let lengths = [
            8 * 32 * 128 * 4,
            (context + 8) * 8 * 128 * 4,
            (context + 8) * 8 * 128 * 4,
            8 * 32 * 128 * 4,
        ];
        from_buffers(std::array::from_fn(|i| {
            crate::cuda_runtime::test_buffer_view(
                0x1000_0000 + i as u64 * 0x1000_0000,
                0,
                0,
                lengths[i],
            )
        }))
    }

    #[test]
    fn block_attention_abi_and_geometry_fail_closed() {
        assert_eq!(core::mem::size_of::<Raw>(), 112);
        assert_eq!(core::mem::offset_of!(Raw, query), 8);
        assert_eq!(core::mem::offset_of!(Raw, context_len), 72);
        for n in [0, 1, 64, 2048] {
            assert!(bind_raw(0, fake(n), n, 4096, 4096 + n, 2048).is_ok());
        }
        for (n, s, b, w) in [
            (2049, 0, 2049, 2048),
            (1, 0, 1, 0),
            (1, 0, 1, 2049),
            (1, 4096, 4096, 2048),
            (0, 262137, 262137, 1),
            (1, usize::MAX, 0, 1),
        ] {
            assert!(bind_raw(0, fake(1), n, s, b, w).is_err());
        }
        assert!(bind_raw(1, fake(1), 1, 4096, 4097, 2048).is_err());
        for i in 0..4 {
            let mut b = fake(2048).buffers();
            b[i] = crate::cuda_runtime::test_buffer_view(
                0x1000_0000 + i as u64 * 0x1000_0000,
                0,
                0,
                b[i].len_bytes() - 4,
            );
            assert!(bind_raw(0, from_buffers(b), 2048, 4096, 6144, 2048).is_err());
            for j in 0..i {
                let mut b = fake(2048).buffers();
                b[i] = crate::cuda_runtime::test_buffer_view(
                    0x1000_0000 + j as u64 * 0x1000_0000,
                    0,
                    0,
                    b[i].len_bytes(),
                );
                assert!(bind_raw(0, from_buffers(b), 2048, 4096, 6144, 2048).is_err());
            }
        }
    }

    // Independent F64 dot/softmax/PV, not the CUDA reduction tree.
    fn reference(q: &[f32], k: &[f32], v: &[f32], context: usize, window: usize) -> Vec<f32> {
        let mut out = vec![0.; 8 * 32 * 128];
        for token in 0..8 {
            let lower = (context + token + 1).saturating_sub(window);
            for head in 0..32 {
                let qb = (token * 32 + head) * 128;
                let kh = head / 4;
                let scores: Vec<f64> = (lower..context + 8)
                    .map(|r| {
                        (0..128)
                            .map(|d| q[qb + d] as f64 * k[(r * 8 + kh) * 128 + d] as f64)
                            .sum::<f64>()
                            / 128.0f64.sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let ps: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = ps.iter().sum();
                for d in 0..128 {
                    out[qb + d] = (ps
                        .iter()
                        .enumerate()
                        .map(|(i, p)| p * v[((lower + i) * 8 + kh) * 128 + d] as f64)
                        .sum::<f64>()
                        / sum) as f32;
                }
            }
        }
        out
    }

    #[test]
    #[ignore = "requires SM70 CUDA; explicit noncausal dense block arithmetic"]
    fn block_attention_gpu_full_window_future_rows_and_faults(
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::{
            cuda_device_allocation_snapshot, cuda_host_boundary_snapshot, OwnedCudaBuffer,
        };
        let before = cuda_device_allocation_snapshot();
        let stream = OwnedCudaStream::create(0)?;
        let fbytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        for (context, window) in [
            (0, 1),
            (1, 2048),
            (64, 16),
            (511, 512),
            (512, 2048),
            (2040, 2048),
            (2048, 1),
            (2048, 2048),
        ] {
            let mut q: Vec<_> = (0..8 * 32 * 128)
                .map(|i| ((i % 83) as f32 - 41.) * 0.015)
                .collect();
            let k: Vec<_> = (0..(context + 8) * 8 * 128)
                .map(|i| ((i % 59) as f32 - 29.) * 0.019)
                .collect();
            let mut v: Vec<_> = (0..k.len())
                .map(|i| ((i % 101) as f32 - 50.) * 0.012)
                .collect();
            // With zero Q the first query must still consume future draft V.
            if context == 0 {
                q.fill(0.);
                v.fill(0.);
                v[7 * 8 * 128..].fill(8.);
            }
            let expected = reference(&q, &k, &v, context, window);
            let payloads = [
                fbytes(&q),
                fbytes(&k),
                fbytes(&v),
                fbytes(&vec![f32::NAN; q.len()]),
            ];
            let owners = payloads
                .iter()
                .map(|p| OwnedCudaBuffer::allocate(0, p.len()))
                .collect::<Result<Vec<_>, _>>()?;
            for (o, p) in owners.iter().zip(&payloads) {
                o.copy_from_host_sync(&stream, p)?;
            }
            let views = from_buffers(
                owners
                    .iter()
                    .map(|o| o.view(0, o.len_bytes()))
                    .collect::<Result<Vec<_>, _>>()?
                    .try_into()
                    .unwrap(),
            );
            let bound = CudaDflashBlockAttentionF32ArgsV1::bind(
                &stream,
                views,
                context,
                4096,
                4096 + context,
                window,
            )?;
            let boundary = cuda_host_boundary_snapshot();
            let allocation = cuda_device_allocation_snapshot();
            bound.submit(&stream)?;
            let delta = cuda_host_boundary_snapshot()
                .checked_delta_since(boundary)
                .unwrap();
            assert_eq!(
                delta.synchronous_h2d_bytes + delta.synchronous_d2h_bytes + delta.host_waits,
                0
            );
            assert_eq!(
                cuda_device_allocation_snapshot().successful_allocations,
                allocation.successful_allocations
            );
            let mut bytes = vec![0u8; q.len() * 4];
            owners[3].copy_to_host_sync(&stream, &mut bytes)?;
            let actual: Vec<_> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.is_finite() && (a - b).abs() <= 2e-5 + 2e-5 * b.abs()),
                "context={context} max={max}"
            );
            if context == 0 {
                assert!((actual[0] - 1.).abs() < 1e-6);
                assert!((actual[7 * 32 * 128] - 8.).abs() < 1e-6);
            }
            eprintln!(
                "DFLASH_BLOCK_ATTENTION context={context} window={window} max_abs={max:e} PASS"
            );
            let other = OwnedCudaStream::create(0)?;
            assert!(bound.submit(&other).is_err());
            for bad in 0..6 {
                let mut raw = bound.raw;
                match bad {
                    0 => raw.block_start += 1,
                    1 => raw.context_len = 2049,
                    2 => raw.query = raw.output,
                    3 => raw.key_len_bytes -= 4,
                    4 => raw.query = u64::MAX - 3,
                    _ => raw.reserved = 1,
                }
                let result = unsafe {
                    augur_cuda_sys::augur_cuda_dflash_block_attention_f32_submit_v1(
                        0,
                        stream.raw()?,
                        &raw,
                    )
                };
                assert_eq!(result.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
            }
            for index in 0..3 {
                let mut poison = payloads[index].clone();
                poison[..4].copy_from_slice(&f32::NAN.to_le_bytes());
                owners[index].copy_from_host_sync(&stream, &poison)?;
                bound.submit(&stream)?;
                owners[3].copy_to_host_sync(&stream, &mut bytes)?;
                assert!(bytes[..128 * 4]
                    .chunks_exact(4)
                    .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_nan()));
                owners[index].copy_from_host_sync(&stream, &payloads[index])?;
            }
            bound.submit(&stream)?;
            owners[3].copy_to_host_sync(&stream, &mut bytes)?;
            assert!(bytes
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()));
        }
        let after = cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
        Ok(())
    }
}
