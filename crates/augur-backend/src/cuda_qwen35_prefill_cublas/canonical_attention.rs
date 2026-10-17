//! Input-only attention scoring; decode/verify keep their original V3/V4 ABI.
use super::*;
use crate::{
    CudaQwen35Kv8BatchedAttentionArgsV4, CudaQwen35Kv8BatchedAttentionViewsV4,
    QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY,
};
use augur_cuda_sys::{
    AugurCudaQwen35CanonicalPrefillArgsV1, AUGUR_CUDA_QWEN35_CANONICAL_PREFILL_ARGS_VERSION_V1,
    AUGUR_CUDA_QWEN35_CANONICAL_PREFILL_MAX_TOKENS_V1,
};

const OP: &str = "qwen35_canonical_prefill_scores_v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Qwen35CanonicalPrefillScorePolicy {
    /// The canonical FP64 lane accumulation and warp reduction, in parallel.
    CanonicalWarp = 0,
    /// FP64 cuBLAS changes the dot-product reduction tree, not the precision.
    Dgemm = 1,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35CanonicalPrefillV1 {
    raw: AugurCudaQwen35CanonicalPrefillArgsV1,
    ordinal: i32,
    plan: usize,
    scores: u64,
    score_bytes: u64,
    policy: Qwen35CanonicalPrefillScorePolicy,
}

fn extent(rows: usize, capacity: usize, position: usize) -> Result<(), CudaRuntimeError> {
    if rows == 0
        || rows > AUGUR_CUDA_QWEN35_CANONICAL_PREFILL_MAX_TOKENS_V1
        || capacity == 0
        || capacity > QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY
        || position.checked_add(rows).is_none_or(|end| end > capacity)
    {
        return Err(CudaRuntimeError::validation(
            OP,
            CudaRuntimeErrorKind::InvalidDimensions,
        ));
    }
    Ok(())
}

impl CudaQwen35CanonicalPrefillV1 {
    pub const MAX_ROWS: usize = AUGUR_CUDA_QWEN35_CANONICAL_PREFILL_MAX_TOKENS_V1;

    /// Bounded row128/key8192 score/FP64 packing storage, in F32-sized units.
    pub fn score_elements(rows: usize, capacity: usize) -> Result<usize, CudaRuntimeError> {
        extent(rows, capacity, 0)?;
        CudaQwen35PrefillKv8ChunkAttentionV1::dgemm_score_elements(rows, capacity)
    }

    /// Bind exactly sized, disjoint buffers to one cuBLAS/stream owner.
    /// Owners must remain alive until the covering stream drain, on errors too.
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        stream: &OwnedCudaStream,
        plan: &CudaQwen35PrefillCublasPlanV1,
        views: CudaQwen35Kv8BatchedAttentionViewsV4,
        capacity: usize,
        position: usize,
        rows: usize,
        scores: CudaBufferView,
        policy: Qwen35CanonicalPrefillScorePolicy,
    ) -> Result<Self, CudaRuntimeError> {
        extent(rows, capacity, position)?;
        plan.require_stream(stream)?;
        let bound = CudaQwen35Kv8BatchedAttentionArgsV4::bind_rows(
            stream,
            views,
            capacity,
            position,
            rows,
            Self::MAX_ROWS,
            AUGUR_CUDA_QWEN35_CANONICAL_PREFILL_ARGS_VERSION_V1,
        )?;
        let bytes = Self::score_elements(rows, capacity)? * 4;
        let address = scores.checked_address(OP)?;
        if scores.device_ordinal() != stream.device_ordinal()
            || scores.len_bytes() != bytes
            || !address.is_multiple_of(8)
        {
            return Err(CudaRuntimeError::validation(
                OP,
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        for view in [
            views.query.buffer_view(),
            views.gate.buffer_view(),
            views.key.buffer_view(),
            views.value.buffer_view(),
            views.key_cache,
            views.value_cache,
            views.attention.buffer_view(),
            views.gated.buffer_view(),
            views.workspace.buffer_view(),
        ] {
            if scores.overlaps(view, OP)? {
                return Err(CudaRuntimeError::validation(
                    OP,
                    CudaRuntimeErrorKind::OverlappingDeviceRanges,
                ));
            }
        }
        let raw = *bound.as_raw();
        // SAFETY: Rust validates all declared shapes/owners/ranges above.
        // Native preflight additionally checks live CUDA extents and disjoint
        // cuBLAS scratch. It submits no work and allocates no device memory.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_canonical_prefill_scores_preflight_v1(
                stream.device_ordinal(),
                stream.raw()?,
                plan.raw,
                &raw,
                address,
                bytes as u64,
                policy as u32,
            )
        };
        CudaRuntimeError::from_result(OP, result)?;
        Ok(Self {
            raw,
            ordinal: stream.device_ordinal(),
            plan: plan.raw as usize,
            scores: address,
            score_bytes: bytes as u64,
            policy,
        })
    }

    pub const fn as_raw(&self) -> &AugurCudaQwen35CanonicalPrefillArgsV1 {
        &self.raw
    }
    pub const fn score_policy(&self) -> Qwen35CanonicalPrefillScorePolicy {
        self.policy
    }

    pub fn submit(
        &self,
        stream: &OwnedCudaStream,
        plan: &CudaQwen35PrefillCublasPlanV1,
    ) -> Result<(), CudaRuntimeError> {
        plan.require_stream(stream)?;
        if stream.device_ordinal() != self.ordinal || plan.raw as usize != self.plan {
            return Err(CudaRuntimeError::validation(
                OP,
                CudaRuntimeErrorKind::InvalidMetadata,
            ));
        }
        // SAFETY: exact buffers and plan were preflighted; the caller retains
        // their owners until the covering drain. No submit-time allocation.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_canonical_prefill_scores_submit_v1(
                self.ordinal,
                stream.raw()?,
                plan.raw,
                &self.raw,
                self.scores,
                self.score_bytes,
                self.policy as u32,
            )
        };
        CudaRuntimeError::from_result(OP, result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_prefill_is_a_distinct_abi_not_widened_verification() {
        assert_eq!(CudaQwen35CanonicalPrefillV1::MAX_ROWS, 128);
        assert_eq!(
            augur_cuda_sys::AUGUR_CUDA_QWEN35_KV8_BATCHED_ATTENTION_MAX_TOKENS_V4,
            8
        );
        assert_ne!(
            AUGUR_CUDA_QWEN35_CANONICAL_PREFILL_ARGS_VERSION_V1,
            augur_cuda_sys::AUGUR_CUDA_QWEN35_KV8_BATCHED_ATTENTION_ARGS_VERSION_V4
        );
        assert_eq!(
            core::mem::size_of::<AugurCudaQwen35CanonicalPrefillArgsV1>(),
            192
        );
        assert_eq!(Qwen35CanonicalPrefillScorePolicy::CanonicalWarp as u32, 0);
        assert_eq!(Qwen35CanonicalPrefillScorePolicy::Dgemm as u32, 1);
    }

    #[test]
    fn rows_capacity_and_causal_endpoint_fail_closed() {
        for (rows, capacity, position) in [
            (0, 128, 0),
            (129, 1024, 0),
            (8, 0, 0),
            (8, 16, 9),
            (1, 128, usize::MAX),
            (1, usize::MAX, 0),
        ] {
            assert!(extent(rows, capacity, position).is_err());
        }
        for rows in [1, 2, 3, 8, 32, 128] {
            assert!(extent(rows, 196608, 196608 - rows).is_ok());
            assert!(extent(rows, 196608, 196609 - rows).is_err());
        }
    }

    #[test]
    fn score_workspace_is_exact_bounded_and_double_aligned() {
        for rows in [1, 3, 8, 32, 128] {
            for capacity in [128, 8191, 8192, 32768, 196608] {
                let keys = capacity.min(8192);
                let bytes =
                    CudaQwen35CanonicalPrefillV1::score_elements(rows, capacity).unwrap() * 4;
                let expected = rows * 24 * (keys + 2) * 4
                    + (rows * 6 * 256 + keys * 256 + rows * 6 * keys) * 8;
                assert_eq!(bytes, expected);
                assert_eq!(bytes % 8, 0);
                assert!(bytes <= 169369600);
            }
        }
        assert!(CudaQwen35CanonicalPrefillV1::score_elements(129, 196608).is_err());
    }

    #[test]
    fn tiled_consumer_visits_the_exact_original_72_intervals_once() {
        for rows in [1, 3, 8, 32, 128] {
            for base in [0, 1, 511, 8191, 32767, 195000] {
                for row in 0..rows {
                    let visible = base + row + 1;
                    let mut previous_end = 0;
                    for split in 0..72 {
                        let first = visible * split / 72;
                        let last = visible * (split + 1) / 72;
                        assert_eq!(first, previous_end);
                        let mut visited_end = first;
                        for tile in (0..base + rows).step_by(8192) {
                            let lo = first.max(tile);
                            let hi = last.min((tile + 8192).min(base + rows));
                            if lo < hi {
                                assert_eq!(lo, visited_end);
                                visited_end = hi;
                            }
                        }
                        assert_eq!(visited_end, last);
                        previous_end = last;
                    }
                    assert_eq!(previous_end, visible);
                }
            }
        }
    }

    #[test]
    #[ignore = "requires exclusive SM70 device; pair with the CUDA component run"]
    fn native_declared_length_drift_rejects_before_any_publication_gpu(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut stream = OwnedCudaStream::create(0)?;
        let mut owners = Vec::new();
        let lengths = [
            6144 * 4,
            6144 * 4,
            1024 * 4,
            1024 * 4,
            1088,
            1088,
            6144 * 4,
            6144 * 4,
            crate::QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2 * 4,
        ];
        let result = (|| -> Result<(), Box<dyn std::error::Error>> {
            for bytes in lengths {
                let owner = OwnedCudaBuffer::allocate(0, bytes)?;
                owner.copy_from_host_sync(&stream, &vec![0xa5; bytes])?;
                owners.push(owner);
            }
            let f = |i: usize| {
                CudaF32View::new(
                    owners[i].view(0, lengths[i])?,
                    augur_core::PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    lengths[i] / 4,
                )
            };
            let views = CudaQwen35Kv8BatchedAttentionViewsV4 {
                query: f(0)?,
                gate: f(1)?,
                key: f(2)?,
                value: f(3)?,
                key_cache: owners[4].view(0, lengths[4])?,
                value_cache: owners[5].view(0, lengths[5])?,
                attention: f(6)?,
                gated: f(7)?,
                workspace: f(8)?,
            };
            let op = CudaQwen35Kv8BatchedAttentionArgsV4::bind(&stream, views, 1, 0, 1)?;
            let boundary = crate::cuda_host_boundary_snapshot();
            for i in 0..9 {
                let mut bad = *op.as_raw();
                match i {
                    0 => bad.query_len_bytes -= 4,
                    1 => bad.gate_len_bytes -= 4,
                    2 => bad.key_len_bytes -= 4,
                    3 => bad.value_len_bytes -= 4,
                    4 => bad.key_cache_len_bytes -= 2,
                    5 => bad.value_cache_len_bytes -= 2,
                    6 => bad.attention_len_bytes -= 4,
                    7 => bad.gated_len_bytes -= 4,
                    _ => bad.workspace_len_bytes -= 4,
                }
                // SAFETY: all physical allocations/stream are live and sized
                // for the unmodified geometry. Only a reported length drifts.
                let rejected = unsafe {
                    augur_cuda_sys::augur_cuda_qwen35_kv8_batched_attention_submit_v4(
                        0,
                        stream.raw()?,
                        &bad,
                    )
                };
                if rejected.status != -1 {
                    return Err("native length drift accepted".into());
                }
            }
            if crate::cuda_host_boundary_snapshot() != boundary {
                return Err("rejected metadata performed a host copy/wait".into());
            }
            for (owner, bytes) in owners.iter().zip(lengths) {
                let mut actual = vec![0; bytes];
                owner.copy_to_host_sync(&stream, &mut actual)?;
                if actual.iter().any(|&b| b != 0xa5) {
                    return Err("rejected metadata mutated a device buffer".into());
                }
            }
            Ok(())
        })();
        let drained = stream.synchronize();
        drop(owners);
        let closed = stream.close();
        drained?;
        closed?;
        result
    }
}
