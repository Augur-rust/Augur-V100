//! Full-vocabulary (not causal-prefix) top-k for bounded DFlash candidate sets.
//! No softmax, sampling, selector or acceptance authority is implied.

use crate::{CudaBufferView, CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};
use crate::{QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS, QWEN35_DFLASH_ACCEPTANCE_VOCAB};

pub const QWEN35_VOCAB_TOP_K_MAX: usize = 20;
const OP: &str = "cuda_qwen35_vocab_topk_submit_v1";

/// Descending finite logits with lowest-id numeric ties (including signed zero).
/// Any NaN/Inf invalidates the entire row: all ids UINT32_MAX and scores NaN.
#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35VocabTopkArgsV1 {
    raw: augur_cuda_sys::AugurCudaQwen35VocabTopkArgsV1,
    ordinal: i32,
    tiled: Option<(u64, u64, u32)>,
}

impl CudaQwen35VocabTopkArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        logits: CudaF32View,
        ids: CudaBufferView,
        scores: CudaBufferView,
        rows: usize,
        top_k: usize,
    ) -> Result<Self, CudaRuntimeError> {
        Self::bind_for_ordinal(stream.device_ordinal(), logits, ids, scores, rows, top_k)
    }

    fn bind_for_ordinal(
        ordinal: i32,
        logits: CudaF32View,
        ids: CudaBufferView,
        scores: CudaBufferView,
        rows: usize,
        top_k: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if ordinal < 0
            || !(1..=QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS).contains(&rows)
            || !(1..=QWEN35_VOCAB_TOP_K_MAX).contains(&top_k)
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let input = logits.buffer_view();
        let buffers = [input, ids, scores];
        if buffers.iter().any(|b| b.device_ordinal() != ordinal) {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let count = rows * QWEN35_DFLASH_ACCEPTANCE_VOCAB;
        if logits.element_count() != count
            || input.len_bytes() != count * 4
            || ids.len_bytes() != rows * top_k * 4
            || scores.len_bytes() != rows * top_k * 4
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let mut ranges = [(0_u64, 0_u64); 3];
        for (index, buffer) in buffers.iter().enumerate() {
            let start = buffer
                .device_ptr()
                .raw_value()
                .checked_add(buffer.offset_bytes() as u64)
                .ok_or_else(CudaRuntimeError::range_overflow)?;
            let end = start
                .checked_add(buffer.len_bytes() as u64)
                .ok_or_else(CudaRuntimeError::range_overflow)?;
            if start == 0 || !start.is_multiple_of(4) {
                return Err(error(CudaRuntimeErrorKind::InvalidLayout));
            }
            for &(other_start, other_end) in &ranges[..index] {
                if start < other_end && other_start < end {
                    return Err(error(CudaRuntimeErrorKind::InvalidLayout));
                }
            }
            ranges[index] = (start, end);
        }
        Ok(Self {
            raw: augur_cuda_sys::AugurCudaQwen35VocabTopkArgsV1 {
                abi_version: 1,
                struct_size: core::mem::size_of::<augur_cuda_sys::AugurCudaQwen35VocabTopkArgsV1>()
                    as u32,
                rows: rows as u32,
                vocab: QWEN35_DFLASH_ACCEPTANCE_VOCAB as u32,
                top_k: top_k as u32,
                reserved: 0,
                logits: ranges[0].0,
                logits_len_bytes: input.len_bytes() as u64,
                ids: ranges[1].0,
                ids_len_bytes: ids.len_bytes() as u64,
                scores: ranges[2].0,
                scores_len_bytes: scores.len_bytes() as u64,
            },
            ordinal,
            tiled: None,
        })
    }

    /// Bind optional tile scratch. The caller retains this fourth owner too.
    pub fn with_tiled_scratch(mut self, scratch: CudaBufferView, parts: usize) -> Result<Self, CudaRuntimeError> {
        if !matches!(self.raw.top_k, 16 | 20) || !matches!(parts, 8 | 16 | 32 | 64) {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if scratch.device_ordinal() != self.ordinal { return Err(CudaRuntimeError::device_mismatch()); }
        let expected = self.raw.rows as usize * parts * self.raw.top_k as usize * 8;
        if scratch.len_bytes() != expected { return Err(error(CudaRuntimeErrorKind::InvalidDimensions)); }
        let start = scratch.device_ptr().raw_value().checked_add(scratch.offset_bytes() as u64)
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        let end = start.checked_add(expected as u64).ok_or_else(CudaRuntimeError::range_overflow)?;
        if start == 0 || !start.is_multiple_of(4) { return Err(error(CudaRuntimeErrorKind::InvalidLayout)); }
        for (other, len) in [(self.raw.logits,self.raw.logits_len_bytes), (self.raw.ids,self.raw.ids_len_bytes), (self.raw.scores,self.raw.scores_len_bytes)] {
            if start < other + len && other < end { return Err(error(CudaRuntimeErrorKind::InvalidLayout)); }
        }
        self.tiled = Some((start, expected as u64, parts as u32)); Ok(self)
    }

    /// Submit only. Retain inputs, outputs and any tile scratch until completion.
    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        // SAFETY: bind and with_tiled_scratch validate exact extents, alignment,
        // disjoint ranges and ordinal. The caller retains all owners, including
        // tile scratch, until this stream completes; neither stage allocates.
        let result = unsafe {
            if let Some((scratch, len, parts)) = self.tiled {
                augur_cuda_sys::augur_cuda_qwen35_vocab_topk_tiled_submit_v1(self.ordinal, stream.raw()?, &self.raw, scratch, len, parts)
            } else {
                augur_cuda_sys::augur_cuda_qwen35_vocab_topk_submit_v1(self.ordinal, stream.raw()?, &self.raw)
            }
        };
        CudaRuntimeError::from_result(OP, result)
    }
}

fn error(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OP, kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_runtime::test_buffer_view;
    use augur_core::PhysicalLayoutId;

    fn views(rows: usize, k: usize) -> (CudaF32View, CudaBufferView, CudaBufferView) {
        let elements = rows * QWEN35_DFLASH_ACCEPTANCE_VOCAB;
        (
            CudaF32View::new(
                test_buffer_view(0x100_0000, 0, 0, elements * 4),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                elements,
            )
            .unwrap(),
            test_buffer_view(0x200_0000, 0, 0, rows * k * 4),
            test_buffer_view(0x300_0000, 0, 0, rows * k * 4),
        )
    }

    #[test]
    fn vocab_topk_abi_and_exact_extents() {
        type Raw = augur_cuda_sys::AugurCudaQwen35VocabTopkArgsV1;
        assert_eq!(core::mem::size_of::<Raw>(), 72);
        assert_eq!(core::mem::align_of::<Raw>(), 8);
        assert_eq!(core::mem::offset_of!(Raw, logits), 24);
        assert_eq!(core::mem::offset_of!(Raw, scores), 56);
        for rows in [1, 4, 8] {
            for k in [1, 4, 16, 20] {
                let (logits, ids, scores) = views(rows, k);
                let bound =
                    CudaQwen35VocabTopkArgsV1::bind_for_ordinal(0, logits, ids, scores, rows, k)
                        .unwrap();
                assert_eq!(bound.raw.ids_len_bytes, (rows * k * 4) as u64);
                assert_eq!(bound.raw.logits_len_bytes, (rows * 248320 * 4) as u64);
            }
        }
    }

    #[test]
    fn tiled_topk_scratch_rejects_alias_extent_ordinal_and_policy_drift() {
        let (logits, ids, scores) = views(8,20);
        let bound = CudaQwen35VocabTopkArgsV1::bind_for_ordinal(0,logits,ids,scores,8,20).unwrap();
        let size=8*32*20*8;
        assert!(bound.with_tiled_scratch(test_buffer_view(0x400_0000,0,0,size),32).is_ok());
        for bad in [test_buffer_view(0x100_0000,0,0,size), test_buffer_view(0x200_0000,0,0,size),
            test_buffer_view(0x300_0000,0,0,size),test_buffer_view(0x400_0001,0,0,size),
            test_buffer_view(0x400_0000,1,0,size),test_buffer_view(0x400_0000,0,0,size-4),
            test_buffer_view(u64::MAX-3,0,0,size)] {
            assert!(bound.with_tiled_scratch(bad,32).is_err());
        }
        for parts in [0,1,7,31,65,usize::MAX] {
            assert!(bound.with_tiled_scratch(test_buffer_view(0x400_0000,0,0,size),parts).is_err());
        }
        let (logits,ids,scores)=views(1,4);
        let small=CudaQwen35VocabTopkArgsV1::bind_for_ordinal(0,logits,ids,scores,1,4).unwrap();
        assert!(small.with_tiled_scratch(test_buffer_view(0x400_0000,0,0,4*32*8),32).is_err());
    }

    #[test]
    fn vocab_topk_rejects_invalid_geometry_and_ordinal() {
        let (logits, ids, scores) = views(1, 16);
        for (rows, k) in [
            (0, 16),
            (9, 16),
            (usize::MAX, 16),
            (1, 0),
            (1, 17),
            (1, usize::MAX),
            (2, 16),
            (1, 4),
        ] {
            assert!(
                CudaQwen35VocabTopkArgsV1::bind_for_ordinal(0, logits, ids, scores, rows, k)
                    .is_err()
            );
        }
        assert!(
            CudaQwen35VocabTopkArgsV1::bind_for_ordinal(1, logits, ids, scores, 1, 16).is_err()
        );
    }

    #[test]
    fn vocab_topk_rejects_alias_alignment_and_pointer_overflow() {
        let (logits, ids, scores) = views(1, 16);
        for bad in [
            test_buffer_view(0x100_0004, 0, 0, 64),
            test_buffer_view(0x200_0000, 0, 0, 64),
            test_buffer_view(0x300_0001, 0, 0, 64),
            test_buffer_view(u64::MAX - 3, 0, 0, 64),
            test_buffer_view(0x300_0000, 0, 0, 60),
            test_buffer_view(0x300_0000, 1, 0, 64),
        ] {
            assert!(
                CudaQwen35VocabTopkArgsV1::bind_for_ordinal(0, logits, ids, bad, 1, 16).is_err()
            );
        }
        assert!(
            CudaQwen35VocabTopkArgsV1::bind_for_ordinal(0, logits, scores, scores, 1, 16).is_err()
        );
    }

    #[test]
    #[ignore = "requires an SM70 CUDA device"]
    fn vocab_topk_gpu_matches_independent_full_sort_and_rejects_bad_abi(
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::{
            cuda_device_allocation_snapshot, cuda_host_boundary_snapshot, OwnedCudaBuffer,
        };
        const ROWS: usize = 8;
        let vocab = QWEN35_DFLASH_ACCEPTANCE_VOCAB;
        let mut values = vec![-1.0_f32; ROWS * vocab];
        for i in 0..32 {
            values[vocab + vocab - 1 - i] = 32.0 - i as f32;
        }
        let mut seed = 0x12345678_u32;
        for value in &mut values[2 * vocab..3 * vocab] {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *value = ((seed % 65536) as i32 - 32768) as f32 * 0.03125;
        }
        // Includes FTZ-sensitive +/- subnormals and +/- zero.
        for (i, bits) in [
            0_u32, 0x80000000, 1, 2, 0x80000001, 0x80000002, 0x007fffff, 0x00800000,
        ]
        .into_iter()
        .enumerate()
        {
            values[3 * vocab + 200000 + i] = f32::from_bits(bits);
        }
        values[4 * vocab + vocab - 1] = f32::NAN;
        values[5 * vocab + 77] = f32::INFINITY;
        values[6 * vocab + 199999] = f32::NEG_INFINITY;
        values[7 * vocab + 44444] = f32::MAX;
        values[7 * vocab + 22222] = f32::MAX;
        let expected = values
            .chunks_exact(vocab)
            .map(|row| {
                if row.iter().any(|v| !v.is_finite()) {
                    return None;
                }
                let mut ids = (0..vocab).collect::<Vec<_>>();
                ids.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
                ids.truncate(20);
                Some(ids)
            })
            .collect::<Vec<_>>();
        let start_alloc = cuda_device_allocation_snapshot();
        let stream = OwnedCudaStream::create(0)?;
        let input = OwnedCudaBuffer::allocate(0, values.len() * 4)?;
        let ids_owner = OwnedCudaBuffer::allocate(0, ROWS * 20 * 4)?;
        let scores_owner = OwnedCudaBuffer::allocate(0, ROWS * 20 * 4)?;
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        input.copy_from_host_sync(&stream, &bytes)?;
        let logits = CudaF32View::new(
            input.view(0, bytes.len())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            values.len(),
        )?;
        for k in [1, 4, 16, 20] {
            let size = ROWS * k * 4;
            let bound = CudaQwen35VocabTopkArgsV1::bind(
                &stream,
                logits,
                ids_owner.view(0, size)?,
                scores_owner.view(0, size)?,
                ROWS,
                k,
            )?;
            for kind in 0..7 {
                let mut raw = bound.raw;
                match kind {
                    0 => raw.reserved = 1,
                    1 => raw.top_k = 21,
                    2 => raw.rows = 0,
                    3 => raw.ids = raw.scores,
                    4 => raw.scores = raw.logits,
                    5 => raw.scores_len_bytes -= 4,
                    _ => raw.logits = u64::MAX - 3,
                }
                // SAFETY: live owners except deliberate invalid metadata;
                // the native boundary must reject every case before a launch.
                let result = unsafe {
                    augur_cuda_sys::augur_cuda_qwen35_vocab_topk_submit_v1(0, stream.raw()?, &raw)
                };
                assert_eq!(result.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
            }
            let alloc_before = cuda_device_allocation_snapshot();
            let boundary_before = cuda_host_boundary_snapshot();
            bound.submit(&stream)?;
            stream.synchronize()?;
            let mut ids_bytes = vec![0_u8; size];
            let mut score_bytes = vec![0_u8; size];
            ids_owner
                .view(0, size)?
                .copy_to_host_sync(&stream, &mut ids_bytes)?;
            scores_owner
                .view(0, size)?
                .copy_to_host_sync(&stream, &mut score_bytes)?;
            let boundary = cuda_host_boundary_snapshot()
                .checked_delta_since(boundary_before)
                .ok_or("counter regression")?;
            assert_eq!(boundary.synchronous_d2h_bytes, (size * 2) as u64);
            assert_eq!(boundary.synchronous_h2d_bytes, 0);
            assert_eq!(
                cuda_device_allocation_snapshot().successful_allocations,
                alloc_before.successful_allocations
            );
            for (row, oracle) in expected.iter().enumerate() {
                for rank in 0..k {
                    let offset = (row * k + rank) * 4;
                    let id = u32::from_le_bytes(ids_bytes[offset..offset + 4].try_into()?);
                    let bits = u32::from_le_bytes(score_bytes[offset..offset + 4].try_into()?);
                    if let Some(oracle) = oracle {
                        assert_eq!(id as usize, oracle[rank], "row={row}, rank={rank}, k={k}");
                        assert_eq!(bits, values[row * vocab + oracle[rank]].to_bits());
                    } else {
                        assert_eq!(id, u32::MAX);
                        assert!(f32::from_bits(bits).is_nan());
                    }
                }
            }
        }
        for rows in [1usize,8] { for k in [16usize,20] {
            let size=rows*k*4;
            let bound=CudaQwen35VocabTopkArgsV1::bind(&stream,
                CudaF32View::new(input.view(0,rows*vocab*4)?, PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,rows*vocab)?,
                ids_owner.view(0,size)?,scores_owner.view(0,size)?,rows,k)?;
            for parts in [8usize,16,32,64] {
                let scratch=OwnedCudaBuffer::allocate(0,rows*parts*k*8)?;
                let scratch_view=scratch.view(0,rows*parts*k*8)?;
                let tiled=bound.with_tiled_scratch(scratch_view,parts)?;
                let mut ids_bytes=vec![0u8;size];let mut scores_bytes=vec![0u8;size];
                tiled.submit(&stream)?;
                ids_owner.view(0,size)?.copy_to_host_sync(&stream,&mut ids_bytes)?;
                scores_owner.view(0,size)?.copy_to_host_sync(&stream,&mut scores_bytes)?;
                for (row, oracle) in expected.iter().enumerate().take(rows) { for rank in 0..k {
                    let offset=(row*k+rank)*4;
                    let id=u32::from_le_bytes(ids_bytes[offset..offset+4].try_into()?);
                    let bits=u32::from_le_bytes(scores_bytes[offset..offset+4].try_into()?);
                    if let Some(oracle)=oracle {
                        assert_eq!(id as usize,oracle[rank],"tiled rows={rows} parts={parts} k={k}");
                        assert_eq!(bits,values[row*vocab+oracle[rank]].to_bits());
                    } else {assert_eq!(id,u32::MAX);assert!(f32::from_bits(bits).is_nan());}
                }}
                let (ptr,len,part_count)=tiled.tiled.unwrap();
                for (bad_ptr,bad_len,bad_parts) in [(bound.raw.logits,len,part_count),(ptr,len-4,part_count),(ptr,len,0),(u64::MAX-3,len,part_count)] {
                    // SAFETY: invalid metadata must be rejected before touching memory.
                    let result=unsafe {augur_cuda_sys::augur_cuda_qwen35_vocab_topk_tiled_submit_v1(0,stream.raw()?,&bound.raw,bad_ptr,bad_len,bad_parts)};
                    assert_eq!(result.status,augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
                }
            }
        }}
        // Input logits are immutable across all submissions.
        let mut after = vec![0; bytes.len()];
        input.copy_to_host_sync(&stream, &mut after)?;
        assert_eq!(after, bytes);
        drop(scores_owner);
        drop(ids_owner);
        drop(input);
        let end_alloc = cuda_device_allocation_snapshot();
        assert_eq!(end_alloc.live_bytes, start_alloc.live_bytes);
        assert_eq!(end_alloc.live_allocations, start_alloc.live_allocations);
        Ok(())
    }
}

#[cfg(test)]
mod bench_tests;
