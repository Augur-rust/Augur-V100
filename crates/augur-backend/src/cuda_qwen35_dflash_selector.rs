//! Device-only DFlash2 F16 selector: learned gate and predecessor-conditioned
//! greedy walk. This selects draft proposals; it does not accept target tokens.
use crate::{CudaBufferView, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};

pub const QWEN35_DFLASH_SELECTOR_BYTES: [usize; 9] = [
    8 * 5120 * 4,
    256 * 5120 * 2,
    248320 * 256 * 2,
    248320 * 256 * 2,
    8 * 16 * 4,
    8 * 16 * 4,
    8 * 256 * 4,
    8 * 16 * 4,
    8 * 4,
];

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35DflashSelectorViewsV1 {
    pub hidden_rows: CudaBufferView,
    pub gate_weight: CudaBufferView,
    pub predecessor: CudaBufferView,
    pub successor: CudaBufferView,
    pub candidate_ids: CudaBufferView,
    pub unary_scores: CudaBufferView,
    pub gates: CudaBufferView,
    pub transition_scores: CudaBufferView,
    pub path_ids: CudaBufferView,
}

impl CudaQwen35DflashSelectorViewsV1 {
    fn buffers(self) -> [CudaBufferView; 9] {
        [
            self.hidden_rows,
            self.gate_weight,
            self.predecessor,
            self.successor,
            self.candidate_ids,
            self.unary_scores,
            self.gates,
            self.transition_scores,
            self.path_ids,
        ]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35DflashSelectorArgsV1 {
    raw: augur_cuda_sys::AugurCudaQwen35DflashSelectorArgsV1,
    ordinal: i32,
}
impl CudaQwen35DflashSelectorArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        views: CudaQwen35DflashSelectorViewsV1,
        anchor: u32,
    ) -> Result<Self, CudaRuntimeError> {
        Self::bind_for_ordinal(stream.device_ordinal(), views, anchor)
    }
    fn bind_for_ordinal(
        ordinal: i32,
        views: CudaQwen35DflashSelectorViewsV1,
        anchor: u32,
    ) -> Result<Self, CudaRuntimeError> {
        if ordinal < 0 || anchor >= 248320 {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let buffers = views.buffers();
        let mut ranges = [(0_u64, 0_u64); 9];
        for (i, b) in buffers.iter().enumerate() {
            if b.device_ordinal() != ordinal {
                return Err(CudaRuntimeError::device_mismatch());
            }
            if b.len_bytes() != QWEN35_DFLASH_SELECTOR_BYTES[i] {
                return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
            }
            let start = b
                .device_ptr()
                .raw_value()
                .checked_add(b.offset_bytes() as u64)
                .ok_or_else(CudaRuntimeError::range_overflow)?;
            let end = start
                .checked_add(b.len_bytes() as u64)
                .ok_or_else(CudaRuntimeError::range_overflow)?;
            let align = if (1..=3).contains(&i) { 2 } else { 4 };
            if start == 0 || !start.is_multiple_of(align) {
                return Err(error(CudaRuntimeErrorKind::InvalidLayout));
            }
            if ranges[..i].iter().any(|&(s, e)| start < e && s < end) {
                return Err(error(CudaRuntimeErrorKind::InvalidLayout));
            }
            ranges[i] = (start, end);
        }
        let ptr = |i: usize| ranges[i].0;
        let len = |i: usize| QWEN35_DFLASH_SELECTOR_BYTES[i] as u64;
        Ok(Self {
            ordinal,
            raw: augur_cuda_sys::AugurCudaQwen35DflashSelectorArgsV1 {
                abi_version: 1,
                struct_size: 176,
                rows: 8,
                top_k: 16,
                rank: 256,
                hidden: 5120,
                vocab: 248320,
                anchor_token: anchor,
                hidden_rows: ptr(0),
                hidden_rows_len_bytes: len(0),
                gate_weight: ptr(1),
                gate_weight_len_bytes: len(1),
                predecessor: ptr(2),
                predecessor_len_bytes: len(2),
                successor: ptr(3),
                successor_len_bytes: len(3),
                candidate_ids: ptr(4),
                candidate_ids_len_bytes: len(4),
                unary_scores: ptr(5),
                unary_scores_len_bytes: len(5),
                gates: ptr(6),
                gates_len_bytes: len(6),
                transition_scores: ptr(7),
                transition_scores_len_bytes: len(7),
                path_ids: ptr(8),
                path_ids_len_bytes: len(8),
            },
        })
    }
    /// No host work, allocation or synchronization. Keep every owner live
    /// through stream completion, including the gate/transition scratch.
    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        // SAFETY: bound extents, disjoint ranges, alignment and ordinal; owners
        // are retained by the caller for both ordered kernels.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_dflash_selector_submit_v1(
                self.ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_qwen35_dflash_selector_submit_v1", result)
    }
}
fn error(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation("cuda_qwen35_dflash_selector_submit_v1", kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn views_from(b: [CudaBufferView; 9]) -> CudaQwen35DflashSelectorViewsV1 {
        CudaQwen35DflashSelectorViewsV1 {
            hidden_rows: b[0],
            gate_weight: b[1],
            predecessor: b[2],
            successor: b[3],
            candidate_ids: b[4],
            unary_scores: b[5],
            gates: b[6],
            transition_scores: b[7],
            path_ids: b[8],
        }
    }
    fn views() -> CudaQwen35DflashSelectorViewsV1 {
        views_from(std::array::from_fn(|i| {
            crate::cuda_runtime::test_buffer_view(
                0x1000_0000 + (i as u64) * 0x1000_0000,
                0,
                0,
                QWEN35_DFLASH_SELECTOR_BYTES[i],
            )
        }))
    }
    #[test]
    fn selector_abi_exact_and_anchor_bound() {
        type Raw = augur_cuda_sys::AugurCudaQwen35DflashSelectorArgsV1;
        assert_eq!(core::mem::size_of::<Raw>(), 176);
        assert_eq!(core::mem::align_of::<Raw>(), 8);
        assert_eq!(core::mem::offset_of!(Raw, hidden_rows), 32);
        assert_eq!(core::mem::offset_of!(Raw, path_ids), 160);
        assert!(CudaQwen35DflashSelectorArgsV1::bind_for_ordinal(0, views(), 248319).is_ok());
        assert!(CudaQwen35DflashSelectorArgsV1::bind_for_ordinal(0, views(), 248320).is_err());
        assert!(CudaQwen35DflashSelectorArgsV1::bind_for_ordinal(1, views(), 0).is_err());
    }
    #[test]
    fn selector_rejects_each_extent_and_all_alias_pairs() {
        for i in 0..9 {
            let mut b = views().buffers();
            b[i] = crate::cuda_runtime::test_buffer_view(
                0x1000_0000 + (i as u64) * 0x1000_0000,
                0,
                0,
                QWEN35_DFLASH_SELECTOR_BYTES[i] - 1,
            );
            assert!(CudaQwen35DflashSelectorArgsV1::bind_for_ordinal(0, views_from(b), 0).is_err());
            for j in 0..i {
                let mut b = views().buffers();
                b[i] = crate::cuda_runtime::test_buffer_view(
                    0x1000_0000 + (j as u64) * 0x1000_0000,
                    0,
                    0,
                    QWEN35_DFLASH_SELECTOR_BYTES[i],
                );
                assert!(
                    CudaQwen35DflashSelectorArgsV1::bind_for_ordinal(0, views_from(b), 0).is_err()
                );
            }
        }
    }
    #[test]
    #[ignore = "requires an SM70 CUDA device"]
    fn selector_gpu_anchor_chain_ties_and_fail_closed() -> Result<(), Box<dyn std::error::Error>> {
        use crate::{
            cuda_device_allocation_snapshot, cuda_host_boundary_snapshot, OwnedCudaBuffer,
        };
        let before = cuda_device_allocation_snapshot();
        let stream = OwnedCudaStream::create(0)?;
        let owners = QWEN35_DFLASH_SELECTOR_BYTES
            .iter()
            .map(|&size| OwnedCudaBuffer::allocate(0, size))
            .collect::<Result<Vec<_>, _>>()?;
        let b = owners
            .iter()
            .map(|o| o.view(0, o.len_bytes()))
            .collect::<Result<Vec<_>, _>>()?;
        let views = views_from(b.try_into().unwrap());
        let bound = CudaQwen35DflashSelectorArgsV1::bind(&stream, views, 200000)?;
        let fbytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        let mut hidden = vec![0.0_f32; 8 * 5120];
        for row in 0..8 {
            hidden[row * 5120] = 1.0;
        }
        let mut gate = vec![0_u8; QWEN35_DFLASH_SELECTOR_BYTES[1]];
        gate[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let mut prev = vec![0_u8; QWEN35_DFLASH_SELECTOR_BYTES[2]];
        let mut next = prev.clone();
        for (id, bits) in [(200000, 0x3c00_u16), (7, 0xbc00), (8, 0x3c00)] {
            prev[id * 512..id * 512 + 2].copy_from_slice(&bits.to_le_bytes());
        }
        for (id, bits) in [(7, 0x5000_u16), (8, 0xd000)] {
            next[id * 512..id * 512 + 2].copy_from_slice(&bits.to_le_bytes());
        } // +/-32
        let ids = (0..8).flat_map(|_| 0..16_u32).collect::<Vec<_>>();
        let unary = (0..8)
            .flat_map(|_| (0..16).map(|i| 16.0 - i as f32))
            .collect::<Vec<_>>();
        let ibytes = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        owners[0].copy_from_host_sync(&stream, &fbytes(&hidden))?;
        owners[1].copy_from_host_sync(&stream, &gate)?;
        owners[2].copy_from_host_sync(&stream, &prev)?;
        owners[3].copy_from_host_sync(&stream, &next)?;
        owners[4].copy_from_host_sync(&stream, &ibytes(&ids))?;
        owners[5].copy_from_host_sync(&stream, &fbytes(&unary))?;
        let run = || -> Result<Vec<u32>, Box<dyn std::error::Error>> {
            let a = cuda_device_allocation_snapshot();
            let h = cuda_host_boundary_snapshot();
            bound.submit(&stream)?;
            let mut bytes = [0_u8; 32];
            owners[8].copy_to_host_sync(&stream, &mut bytes)?;
            let d = cuda_host_boundary_snapshot()
                .checked_delta_since(h)
                .ok_or("counter regression")?;
            assert_eq!(d.synchronous_d2h_bytes, 32);
            assert_eq!(d.synchronous_h2d_bytes, 0);
            assert_eq!(
                cuda_device_allocation_snapshot().successful_allocations,
                a.successful_allocations
            );
            Ok(bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect())
        };
        assert_eq!(run()?, [200000, 7, 8, 7, 8, 7, 8, 7]);
        for kind in 0..6 {
            let mut raw = bound.raw;
            match kind {
                0 => raw.rank = 255,
                1 => raw.anchor_token = 248320,
                2 => raw.path_ids = raw.gates,
                3 => raw.predecessor_len_bytes -= 2,
                4 => raw.hidden_rows = u64::MAX - 3,
                _ => raw.rows = 7,
            }
            let result = unsafe {
                augur_cuda_sys::augur_cuda_qwen35_dflash_selector_submit_v1(0, stream.raw()?, &raw)
            };
            assert_eq!(result.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
        }
        for kind in 0..4 {
            let mut corrupt_ids = ids.clone();
            let mut corrupt_unary = unary.clone();
            let mut corrupt_hidden = hidden.clone();
            match kind {
                0 => corrupt_ids[127] = u32::MAX,
                1 => corrupt_ids[127] = corrupt_ids[126],
                2 => corrupt_unary[127] = f32::NAN,
                _ => corrupt_hidden[7 * 5120] = f32::INFINITY,
            }
            owners[0].copy_from_host_sync(&stream, &fbytes(&corrupt_hidden))?;
            owners[4].copy_from_host_sync(&stream, &ibytes(&corrupt_ids))?;
            owners[5].copy_from_host_sync(&stream, &fbytes(&corrupt_unary))?;
            assert_eq!(run()?, [u32::MAX; 8]);
        }
        // Valid candidates cannot hide a nonfinite selector tensor, even when
        // the poisoned successor would not otherwise win the current row.
        owners[0].copy_from_host_sync(&stream, &fbytes(&hidden))?;
        owners[4].copy_from_host_sync(&stream, &ibytes(&ids))?;
        owners[5].copy_from_host_sync(&stream, &fbytes(&unary))?;
        for kind in 0..3 {
            let (index, offset) = match kind {
                0 => (1, 0),
                1 => (2, 200000 * 512),
                _ => (3, 15 * 512 + 128 * 2),
            };
            let original = match index {
                1 => &gate,
                2 => &prev,
                _ => &next,
            };
            let mut bad = original.clone();
            bad[offset..offset + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
            owners[index].copy_from_host_sync(&stream, &bad)?;
            assert_eq!(run()?, [u32::MAX; 8]);
            owners[index].copy_from_host_sync(&stream, original)?;
        }
        // First candidate wins score ties; publication recovers after failures.
        gate.fill(0);
        owners[1].copy_from_host_sync(&stream, &gate)?;
        owners[0].copy_from_host_sync(&stream, &fbytes(&hidden))?;
        owners[4].copy_from_host_sync(&stream, &ibytes(&ids))?;
        owners[5].copy_from_host_sync(&stream, &fbytes(&vec![0.0; 128]))?;
        assert_eq!(run()?, [200000, 0, 0, 0, 0, 0, 0, 0]);
        drop(owners);
        let after = cuda_device_allocation_snapshot();
        assert_eq!(before.live_bytes, after.live_bytes);
        assert_eq!(before.live_allocations, after.live_allocations);
        Ok(())
    }
}
