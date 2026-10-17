#![cfg(feature = "cuda")]

//! Bounded DFlash cache-attention consumer.

use crate::{CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};
use augur_cuda_sys::AugurCudaDflashCacheAttentionF32ArgsV1;

const OPERATION: &str = "cuda_dflash_cache_attention_f32_v1";
const QUERY_HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const QUERY_ELEMENTS: usize = QUERY_HEADS * HEAD_DIM;
const CACHE_ROW_ELEMENTS: usize = KV_HEADS * HEAD_DIM;
const MAX_CACHE_POSITIONS: usize = 2048;

// Absolute [start, end) is mapped through position % capacity. Allocation
// extent does not imply initialized history; the owner supplies valid rows.
fn validate_window(capacity: usize, start: usize, end: usize) -> Result<(), CudaRuntimeError> {
    if capacity == 0
        || capacity > MAX_CACHE_POSITIONS
        || start >= end
        || end - start > capacity
        || end > u32::MAX as usize
    {
        return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct CudaDflashCacheAttentionF32ArgsV1 {
    raw: AugurCudaDflashCacheAttentionF32ArgsV1,
    ordinal: i32,
    stream_owner: usize,
}

impl CudaDflashCacheAttentionF32ArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        query: CudaF32View,
        key_cache: CudaF32View,
        value_cache: CudaF32View,
        output: CudaF32View,
        capacity_positions: usize,
        visible_position_start: usize,
        visible_position_end: usize,
        scale: f32,
    ) -> Result<Self, CudaRuntimeError> {
        validate_window(
            capacity_positions,
            visible_position_start,
            visible_position_end,
        )?;
        let cache_elements = capacity_positions
            .checked_mul(CACHE_ROW_ELEMENTS)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        if query.element_count() != QUERY_ELEMENTS
            || output.element_count() != QUERY_ELEMENTS
            || key_cache.element_count() != cache_elements
            || value_cache.element_count() != cache_elements
            || !scale.is_finite()
            || scale <= 0.0
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let ordinal = stream.device_ordinal();
        for view in [
            query.buffer_view(),
            key_cache.buffer_view(),
            value_cache.buffer_view(),
            output.buffer_view(),
        ] {
            if view.device_ordinal() != ordinal {
                return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
            }
            if view.checked_address(OPERATION)? % 4 != 0 {
                return Err(validation(CudaRuntimeErrorKind::MisalignedDeviceAddress));
            }
            view.checked_range(OPERATION)?;
        }
        if query
            .buffer_view()
            .overlaps(output.buffer_view(), OPERATION)?
            || key_cache
                .buffer_view()
                .overlaps(output.buffer_view(), OPERATION)?
            || value_cache
                .buffer_view()
                .overlaps(output.buffer_view(), OPERATION)?
        {
            return Err(validation(CudaRuntimeErrorKind::OverlappingDeviceRanges));
        }
        let mut raw = AugurCudaDflashCacheAttentionF32ArgsV1::new();
        macro_rules! bind_view {
            ($ptr:ident, $len:ident, $view:expr) => {{
                let view = $view.buffer_view();
                raw.$ptr = view.checked_address(OPERATION)?;
                raw.$len = u64::try_from(view.len_bytes())
                    .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
            }};
        }
        bind_view!(query, query_len_bytes, query);
        bind_view!(key_cache, key_cache_len_bytes, key_cache);
        bind_view!(value_cache, value_cache_len_bytes, value_cache);
        bind_view!(output, output_len_bytes, output);
        raw.capacity_positions = u32::try_from(capacity_positions)
            .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        raw.visible_position_start = u32::try_from(visible_position_start)
            .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        raw.visible_position_end = u32::try_from(visible_position_end)
            .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        raw.query_heads = QUERY_HEADS as u32;
        raw.kv_heads = KV_HEADS as u32;
        raw.head_dim = HEAD_DIM as u32;
        raw.scale = scale;
        Ok(Self {
            raw,
            ordinal,
            stream_owner: stream.owner_instance_key()?,
        })
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        let result = unsafe {
            augur_cuda_sys::augur_cuda_dflash_cache_attention_f32_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_dflash_cache_attention_f32_submit_v1", result)
    }

    pub fn launch_sync(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        let result = unsafe {
            augur_cuda_sys::augur_cuda_dflash_cache_attention_f32_v1(
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
        if stream.owner_instance_key()? != self.stream_owner {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        Ok(())
    }
}

fn validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OPERATION, kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OwnedCudaBuffer;
    use augur_core::PhysicalLayoutId;

    fn view(buffer: &OwnedCudaBuffer, elements: usize) -> CudaF32View {
        CudaF32View::new(
            buffer.view(0, elements * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            elements,
        )
        .unwrap()
    }
    fn upload(buffer: &mut OwnedCudaBuffer, stream: &OwnedCudaStream, values: &[f32]) {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        buffer.copy_from_host_sync(stream, &bytes).unwrap();
    }

    #[test]
    #[ignore = "requires CUDA device; bounded arithmetic/validation only"]
    fn gpu_cache_attention_ring_poison_and_f64_oracle() {
        let stream = OwnedCudaStream::create(0).unwrap();
        let mut q = OwnedCudaBuffer::allocate(0, QUERY_ELEMENTS * 4).unwrap();
        let out = OwnedCudaBuffer::allocate(0, QUERY_ELEMENTS * 4).unwrap();
        let query: Vec<f32> = (0..QUERY_ELEMENTS)
            .map(|i| ((i * 7 % 31) as f32 - 15.0) / 31.0)
            .collect();
        upload(&mut q, &stream, &query);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        for capacity in [1, 4, 2048] {
            let elements = capacity * CACHE_ROW_ELEMENTS;
            let mut k = OwnedCudaBuffer::allocate(0, elements * 4).unwrap();
            let mut v = OwnedCudaBuffer::allocate(0, elements * 4).unwrap();
            let mut keys = vec![f32::NAN; elements];
            let mut values = keys.clone();
            let first = 196607;
            // Small ring crosses its head repeatedly; full-capacity fixture
            // exercises the exact upper shared-score bound.
            let steps = if capacity == 2048 { 2048 } else { 12 };
            for step in 0..steps {
                let pos = first + step;
                let slot = pos % capacity;
                for d in 0..CACHE_ROW_ELEMENTS {
                    keys[slot * CACHE_ROW_ELEMENTS + d] = ((pos + d * 3) % 53) as f32 / 29.0 - 0.9;
                    values[slot * CACHE_ROW_ELEMENTS + d] =
                        ((pos * 7 + d) % 71) as f32 / 17.0 - 2.0;
                }
                if capacity == 2048 && step != 0 && step != steps - 1 {
                    continue;
                }
                upload(&mut k, &stream, &keys);
                upload(&mut v, &stream, &values);
                let start = pos + 1 - (step + 1).min(capacity);
                let args = CudaDflashCacheAttentionF32ArgsV1::bind(
                    &stream,
                    view(&q, QUERY_ELEMENTS),
                    view(&k, elements),
                    view(&v, elements),
                    view(&out, QUERY_ELEMENTS),
                    capacity,
                    start,
                    pos + 1,
                    scale,
                )
                .unwrap();
                args.launch_sync(&stream).unwrap();
                let mut bytes = vec![0; QUERY_ELEMENTS * 4];
                out.copy_to_host_sync(&stream, &mut bytes).unwrap();
                let actual: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                for h in 0..QUERY_HEADS {
                    let head = h / 4;
                    let scores: Vec<f64> = (start..=pos)
                        .map(|p| {
                            (0..HEAD_DIM)
                                .map(|d| {
                                    query[h * HEAD_DIM + d] as f64
                                        * keys[(p % capacity) * CACHE_ROW_ELEMENTS
                                            + head * HEAD_DIM
                                            + d] as f64
                                })
                                .sum::<f64>()
                                * scale as f64
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                    let sum: f64 = weights.iter().sum();
                    for d in 0..HEAD_DIM {
                        let expected = (start..=pos)
                            .zip(&weights)
                            .map(|(p, w)| {
                                w * values
                                    [(p % capacity) * CACHE_ROW_ELEMENTS + head * HEAD_DIM + d]
                                    as f64
                            })
                            .sum::<f64>()
                            / sum;
                        let a = actual[h * HEAD_DIM + d] as f64;
                        assert!(
                            a.is_finite() && (a - expected).abs() <= 2e-4 + 1e-4 * expected.abs(),
                            "capacity={capacity} step={step} head={h} dim={d}: {a} != {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA device; raw ABI rejects before launch"]
    fn gpu_cache_attention_rejects_invalid_native_args() {
        let stream = OwnedCudaStream::create(0).unwrap();
        let q = OwnedCudaBuffer::allocate(0, QUERY_ELEMENTS * 4).unwrap();
        let k = OwnedCudaBuffer::allocate(0, 4 * CACHE_ROW_ELEMENTS * 4).unwrap();
        let v = OwnedCudaBuffer::allocate(0, 4 * CACHE_ROW_ELEMENTS * 4).unwrap();
        let out = OwnedCudaBuffer::allocate(0, QUERY_ELEMENTS * 4).unwrap();
        let bound = CudaDflashCacheAttentionF32ArgsV1::bind(
            &stream,
            view(&q, QUERY_ELEMENTS),
            view(&k, 4 * CACHE_ROW_ELEMENTS),
            view(&v, 4 * CACHE_ROW_ELEMENTS),
            view(&out, QUERY_ELEMENTS),
            4,
            3,
            4,
            0.125,
        )
        .unwrap();
        let mutations: &[fn(&mut AugurCudaDflashCacheAttentionF32ArgsV1)] = &[
            |a| a.capacity_positions = 2049,
            |a| a.capacity_positions = 0,
            |a| a.visible_position_end = 8,
            |a| a.visible_position_end = 3,
            |a| a.reserved = 1,
            |a| a.abi_version += 1,
            |a| a.struct_size -= 1,
            |a| a.query_heads = 31,
            |a| a.kv_heads = 7,
            |a| a.head_dim = 64,
            |a| a.query_len_bytes -= 4,
            |a| a.output_len_bytes += 4,
            |a| a.key_cache_len_bytes -= 4,
            |a| a.value_cache_len_bytes += 4,
            |a| a.scale = f32::NAN,
            |a| a.scale = 0.0,
            |a| a.query += 1,
            |a| a.query = u64::MAX - 3,
            |a| a.output = a.key_cache + 4,
            |a| a.output = a.query + 4,
            |a| a.output = a.value_cache + 4,
        ];
        let other_stream = OwnedCudaStream::create(0).unwrap();
        assert!(bound.submit_for_plan(&other_stream).is_err());
        for mutate in mutations {
            let mut raw = bound.raw;
            mutate(&mut raw);
            let result = unsafe {
                augur_cuda_sys::augur_cuda_dflash_cache_attention_f32_submit_v1(
                    0,
                    stream.raw().unwrap(),
                    &raw,
                )
            };
            assert_ne!(result.status, 0, "invalid raw ABI accepted: {raw:?}");
        }
        assert!(CudaDflashCacheAttentionF32ArgsV1::bind(
            &stream,
            view(&q, QUERY_ELEMENTS - 1),
            view(&k, 4 * CACHE_ROW_ELEMENTS),
            view(&v, 4 * CACHE_ROW_ELEMENTS),
            view(&out, QUERY_ELEMENTS),
            4,
            3,
            4,
            0.125
        )
        .is_err());
        assert!(CudaDflashCacheAttentionF32ArgsV1::bind(
            &stream,
            view(&q, QUERY_ELEMENTS),
            view(&k, 4 * CACHE_ROW_ELEMENTS),
            view(&v, 4 * CACHE_ROW_ELEMENTS),
            view(&k, QUERY_ELEMENTS),
            4,
            3,
            4,
            0.125
        )
        .is_err());
    }

    #[test]
    fn geometry_constants_match_dflash_contract() {
        assert_eq!(QUERY_ELEMENTS, 4_096);
        assert_eq!(CACHE_ROW_ELEMENTS, 1_024);
        assert_eq!(QUERY_HEADS / KV_HEADS, 4);
    }

    #[test]
    fn absolute_window_allows_wrap_but_not_over_capacity_or_overflow() {
        assert!(validate_window(4, 3, 7).is_ok());
        assert!(validate_window(2048, 196607, 196608).is_ok());
        for (capacity, start, end) in [
            (0, 0, 1),
            (2049, 0, 1),
            (4, 3, 3),
            (4, 4, 3),
            (4, 0, 5),
            (4, u32::MAX as usize, u32::MAX as usize + 1),
        ] {
            assert!(validate_window(capacity, start, end).is_err());
        }
    }
}
