#![cfg(feature = "qwen35-device-token-handoff")]

use crate::{CudaBufferView, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};
pub use augur_cuda_sys::{
    AugurCudaQwen35DflashAcceptanceArgsV1 as CudaQwen35RawDflashAcceptanceArgsV1,
    AugurCudaQwen35DflashAcceptanceReceiptV1 as CudaQwen35RawDflashAcceptanceReceiptV1,
    AugurCudaQwen35DflashLogitsAcceptanceArgsV1 as CudaQwen35RawDflashLogitsAcceptanceArgsV1,
};

pub const QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS: usize =
    augur_cuda_sys::AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS_V1 as usize;
pub const QWEN35_DFLASH_ACCEPTANCE_VOCAB: usize =
    augur_cuda_sys::AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_VOCAB_V1 as usize;
pub const QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES: usize =
    augur_cuda_sys::AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES_V1 as usize;

const OP: &str = "cuda_qwen35_dflash_acceptance_submit_v1";
const PARSE_OP: &str = "cuda_qwen35_dflash_acceptance_parse_v1";

/// Greedy full-vocabulary selection, unlike the causal-prefix M142 top-k.
/// Ties select the lowest id; any nonfinite logit invalidates the whole row
/// (UINT32_MAX). The output can feed acceptance directly on the same stream.
#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35VocabTop1ArgsV1 {
    raw: augur_cuda_sys::AugurCudaQwen35VocabTop1ArgsV1,
    ordinal: i32,
}

impl CudaQwen35VocabTop1ArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        logits: crate::CudaF32View,
        ids: CudaBufferView,
        rows: usize,
    ) -> Result<Self, CudaRuntimeError> {
        Self::bind_for_ordinal(stream.device_ordinal(), logits, ids, rows)
    }

    fn bind_for_ordinal(
        ordinal: i32,
        logits: crate::CudaF32View,
        ids: CudaBufferView,
        rows: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if ordinal < 0 || !(1..=QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS).contains(&rows) {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let elements = rows * QWEN35_DFLASH_ACCEPTANCE_VOCAB;
        let buffer = logits.buffer_view();
        if buffer.device_ordinal() != ordinal || ids.device_ordinal() != ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        if logits.element_count() != elements
            || buffer.len_bytes() != elements * 4
            || ids.len_bytes() != rows * 4
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let input = address(buffer)?;
        let output = address(ids)?;
        if input == 0
            || output == 0
            || !input.is_multiple_of(4)
            || !output.is_multiple_of(4)
            || ranges_overlap(input, buffer.len_bytes(), output, ids.len_bytes())?
        {
            return Err(error(CudaRuntimeErrorKind::InvalidLayout));
        }
        Ok(Self {
            raw: augur_cuda_sys::AugurCudaQwen35VocabTop1ArgsV1 {
                abi_version: 1,
                struct_size: core::mem::size_of::<augur_cuda_sys::AugurCudaQwen35VocabTop1ArgsV1>()
                    as u32,
                rows: rows as u32,
                vocab: QWEN35_DFLASH_ACCEPTANCE_VOCAB as u32,
                logits: input,
                logits_len_bytes: buffer.len_bytes() as u64,
                ids: output,
                ids_len_bytes: ids.len_bytes() as u64,
            },
            ordinal,
        })
    }

    /// Submit only: callers must keep both buffers live through completion.
    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        // SAFETY: bind validated ordinal, exact extents, alignment and disjoint
        // ranges. Owners remain the caller's responsibility through completion.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_vocab_top1_submit_v1(self.ordinal, stream.raw()?, &self.raw)
        };
        CudaRuntimeError::from_result("cuda_qwen35_vocab_top1_submit_v1", result)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaQwen35DflashAcceptanceViewsV1 {
    pub draft_ids: CudaBufferView,
    pub target_ids: CudaBufferView,
    pub receipt: CudaBufferView,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35DflashAcceptanceArgsV1 {
    raw: CudaQwen35RawDflashAcceptanceArgsV1,
    device_ordinal: i32,
}

impl CudaQwen35DflashAcceptanceArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        views: CudaQwen35DflashAcceptanceViewsV1,
        draft_tokens: usize,
        step_generation: u64,
        plan_digest: [u8; 32],
    ) -> Result<Self, CudaRuntimeError> {
        let CudaQwen35DflashAcceptanceViewsV1 {
            draft_ids,
            target_ids,
            receipt,
        } = views;
        let ordinal = stream.device_ordinal();
        if !(1..=QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS).contains(&draft_tokens)
            || step_generation == 0
            || plan_digest == [0; 32]
            || draft_ids.device_ordinal() != ordinal
            || target_ids.device_ordinal() != ordinal
            || receipt.device_ordinal() != ordinal
            || draft_ids.len_bytes() != draft_tokens.saturating_mul(core::mem::size_of::<u32>())
            || target_ids.len_bytes() != draft_tokens.saturating_mul(core::mem::size_of::<u32>())
            || receipt.len_bytes() != QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let draft_address = address(draft_ids)?;
        let target_address = address(target_ids)?;
        let receipt_address = address(receipt)?;
        if !draft_address.is_multiple_of(4)
            || !target_address.is_multiple_of(4)
            || !receipt_address.is_multiple_of(8)
            || ranges_overlap(
                draft_address,
                draft_ids.len_bytes(),
                target_address,
                target_ids.len_bytes(),
            )?
            || ranges_overlap(
                draft_address,
                draft_ids.len_bytes(),
                receipt_address,
                receipt.len_bytes(),
            )?
            || ranges_overlap(
                target_address,
                target_ids.len_bytes(),
                receipt_address,
                receipt.len_bytes(),
            )?
        {
            return Err(error(CudaRuntimeErrorKind::InvalidLayout));
        }
        let mut raw = CudaQwen35RawDflashAcceptanceArgsV1::new();
        raw.draft_tokens =
            u32::try_from(draft_tokens).map_err(|_| CudaRuntimeError::range_overflow())?;
        raw.draft_ids = draft_address;
        raw.draft_ids_len_bytes = draft_ids.len_bytes() as u64;
        raw.target_ids = target_address;
        raw.target_ids_len_bytes = target_ids.len_bytes() as u64;
        raw.receipt = receipt_address;
        raw.step_generation = step_generation;
        raw.plan_digest = plan_digest;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn raw(&self) -> &CudaQwen35RawDflashAcceptanceArgsV1 {
        &self.raw
    }

    pub const fn step_generation(&self) -> u64 {
        self.raw.step_generation
    }

    pub const fn plan_digest(&self) -> [u8; 32] {
        self.raw.plan_digest
    }

    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_dflash_acceptance_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OP, result)
    }
}

/// Device-side target verification views. `target_logits` is row-major
/// `[draft_tokens, vocab]`; `target_ids` is an internal device scratch buffer
/// populated by the CUDA operation and must not be read by the host.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaQwen35DflashLogitsAcceptanceViewsV1 {
    pub draft_ids: CudaBufferView,
    pub target_logits: CudaBufferView,
    pub target_ids: CudaBufferView,
    pub receipt: CudaBufferView,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35DflashLogitsAcceptanceArgsV1 {
    raw: CudaQwen35RawDflashLogitsAcceptanceArgsV1,
    device_ordinal: i32,
}

impl CudaQwen35DflashLogitsAcceptanceArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        views: CudaQwen35DflashLogitsAcceptanceViewsV1,
        draft_tokens: usize,
        step_generation: u64,
        plan_digest: [u8; 32],
    ) -> Result<Self, CudaRuntimeError> {
        let CudaQwen35DflashLogitsAcceptanceViewsV1 {
            draft_ids,
            target_logits,
            target_ids,
            receipt,
        } = views;
        let ordinal = stream.device_ordinal();
        let logits_bytes = draft_tokens
            .checked_mul(QWEN35_DFLASH_ACCEPTANCE_VOCAB)
            .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()));
        let ids_bytes = draft_tokens.checked_mul(core::mem::size_of::<u32>());
        if !(1..=QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS).contains(&draft_tokens)
            || step_generation == 0
            || plan_digest == [0; 32]
            || logits_bytes.is_none()
            || ids_bytes.is_none()
            || draft_ids.device_ordinal() != ordinal
            || target_logits.device_ordinal() != ordinal
            || target_ids.device_ordinal() != ordinal
            || receipt.device_ordinal() != ordinal
            || draft_ids.len_bytes() != ids_bytes.unwrap_or(usize::MAX)
            || target_logits.len_bytes() != logits_bytes.unwrap_or(usize::MAX)
            || target_ids.len_bytes() != ids_bytes.unwrap_or(usize::MAX)
            || receipt.len_bytes() != QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let draft_address = address(draft_ids)?;
        let logits_address = address(target_logits)?;
        let target_address = address(target_ids)?;
        let receipt_address = address(receipt)?;
        if !draft_address.is_multiple_of(4)
            || !logits_address.is_multiple_of(4)
            || !target_address.is_multiple_of(4)
            || !receipt_address.is_multiple_of(8)
            || ranges_overlap(
                draft_address,
                draft_ids.len_bytes(),
                logits_address,
                target_logits.len_bytes(),
            )?
            || ranges_overlap(
                draft_address,
                draft_ids.len_bytes(),
                target_address,
                target_ids.len_bytes(),
            )?
            || ranges_overlap(
                draft_address,
                draft_ids.len_bytes(),
                receipt_address,
                receipt.len_bytes(),
            )?
            || ranges_overlap(
                logits_address,
                target_logits.len_bytes(),
                target_address,
                target_ids.len_bytes(),
            )?
            || ranges_overlap(
                logits_address,
                target_logits.len_bytes(),
                receipt_address,
                receipt.len_bytes(),
            )?
            || ranges_overlap(
                target_address,
                target_ids.len_bytes(),
                receipt_address,
                receipt.len_bytes(),
            )?
        {
            return Err(error(CudaRuntimeErrorKind::InvalidLayout));
        }
        let mut raw = CudaQwen35RawDflashLogitsAcceptanceArgsV1::new();
        raw.draft_tokens =
            u32::try_from(draft_tokens).map_err(|_| CudaRuntimeError::range_overflow())?;
        raw.draft_ids = draft_address;
        raw.draft_ids_len_bytes = draft_ids.len_bytes() as u64;
        raw.target_logits = logits_address;
        raw.target_logits_len_bytes = target_logits.len_bytes() as u64;
        raw.target_ids = target_address;
        raw.target_ids_len_bytes = target_ids.len_bytes() as u64;
        raw.receipt = receipt_address;
        raw.receipt_len_bytes = receipt.len_bytes() as u64;
        raw.step_generation = step_generation;
        raw.vocab = QWEN35_DFLASH_ACCEPTANCE_VOCAB as u32;
        raw.plan_digest = plan_digest;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn raw(&self) -> &CudaQwen35RawDflashLogitsAcceptanceArgsV1 {
        &self.raw
    }

    pub const fn step_generation(&self) -> u64 {
        self.raw.step_generation
    }

    pub const fn plan_digest(&self) -> [u8; 32] {
        self.raw.plan_digest
    }

    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_dflash_logits_acceptance_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_qwen35_dflash_logits_acceptance_submit_v1", result)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaQwen35DflashAcceptanceDecisionV1 {
    pub draft_tokens: u32,
    pub accepted_tokens: u32,
    pub rejected_position: u32,
    pub replacement_token: u32,
    pub step_generation: u64,
    pub plan_digest: [u8; 32],
}

pub fn parse_cuda_qwen35_dflash_acceptance_receipt_v1(
    bytes: &[u8],
    expected_step_generation: u64,
    expected_plan_digest: [u8; 32],
) -> Result<CudaQwen35DflashAcceptanceDecisionV1, CudaRuntimeError> {
    if bytes.len() != QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES
        || expected_step_generation == 0
        || expected_plan_digest == [0; 32]
    {
        return Err(parse_error());
    }
    let word = |offset: usize| {
        u32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .expect("validated receipt"),
        )
    };
    let draft_tokens = word(0);
    let accepted_tokens = word(4);
    let rejected_position = word(8);
    let replacement_token = word(12);
    let status = word(16);
    let error_code = word(20);
    let step_generation = u64::from_le_bytes(bytes[24..32].try_into().expect("validated receipt"));
    let mut plan_digest = [0_u8; 32];
    plan_digest.copy_from_slice(&bytes[32..64]);
    if status != augur_cuda_sys::AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_STATUS_OK_V1
        || error_code != augur_cuda_sys::AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ERROR_NONE_V1
        || !(1..=QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS as u32).contains(&draft_tokens)
        || accepted_tokens > draft_tokens
        || rejected_position > draft_tokens
        || (accepted_tokens < draft_tokens && rejected_position != accepted_tokens)
        || (accepted_tokens == draft_tokens && rejected_position != draft_tokens)
        || (rejected_position < draft_tokens
            && replacement_token as usize >= QWEN35_DFLASH_ACCEPTANCE_VOCAB)
        || (rejected_position == draft_tokens && replacement_token != u32::MAX)
        || step_generation != expected_step_generation
        || plan_digest != expected_plan_digest
    {
        return Err(parse_error());
    }
    Ok(CudaQwen35DflashAcceptanceDecisionV1 {
        draft_tokens,
        accepted_tokens,
        rejected_position,
        replacement_token,
        step_generation,
        plan_digest,
    })
}

pub fn dflash_acceptance_cpu_v1(
    draft_ids: &[u32],
    target_ids: &[u32],
) -> Result<(u32, u32, u32), CudaRuntimeError> {
    if draft_ids.is_empty()
        || draft_ids.len() > QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS
        || draft_ids.len() != target_ids.len()
        || draft_ids
            .iter()
            .chain(target_ids)
            .any(|&id| id as usize >= QWEN35_DFLASH_ACCEPTANCE_VOCAB)
    {
        return Err(parse_error());
    }
    for (index, (&draft, &target)) in draft_ids.iter().zip(target_ids).enumerate() {
        if draft != target {
            return Ok((index as u32, index as u32, target));
        }
    }
    let count = draft_ids.len() as u32;
    Ok((count, count, u32::MAX))
}

/// CPU oracle for the device-side logits acceptance operation.  It is used
/// only for numerical tests: each row is greedily reduced with lowest-id tie
/// breaking, then the ordinary verified-prefix rule is applied.
pub fn dflash_logits_acceptance_cpu_v1(
    draft_ids: &[u32],
    target_logits: &[f32],
) -> Result<(u32, u32, u32), CudaRuntimeError> {
    if draft_ids.is_empty()
        || draft_ids.len() > QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS
        || target_logits.len()
            != draft_ids
                .len()
                .checked_mul(QWEN35_DFLASH_ACCEPTANCE_VOCAB)
                .ok_or_else(parse_error)?
        || draft_ids
            .iter()
            .any(|&id| id as usize >= QWEN35_DFLASH_ACCEPTANCE_VOCAB)
        || target_logits.iter().any(|value| !value.is_finite())
    {
        return Err(parse_error());
    }
    let mut target_ids = Vec::with_capacity(draft_ids.len());
    for row in target_logits.chunks_exact(QWEN35_DFLASH_ACCEPTANCE_VOCAB) {
        let mut best_id = 0usize;
        let mut best_value = row[0];
        for (id, &value) in row.iter().enumerate().skip(1) {
            if value > best_value || (value == best_value && id < best_id) {
                best_id = id;
                best_value = value;
            }
        }
        target_ids.push(best_id as u32);
    }
    dflash_acceptance_cpu_v1(draft_ids, &target_ids)
}

fn address(view: CudaBufferView) -> Result<u64, CudaRuntimeError> {
    view.device_ptr()
        .raw_value()
        .checked_add(
            u64::try_from(view.offset_bytes()).map_err(|_| CudaRuntimeError::range_overflow())?,
        )
        .ok_or_else(CudaRuntimeError::range_overflow)
}

fn ranges_overlap(
    left_start: u64,
    left_len: usize,
    right_start: u64,
    right_len: usize,
) -> Result<bool, CudaRuntimeError> {
    let left_end = left_start
        .checked_add(u64::try_from(left_len).map_err(|_| CudaRuntimeError::range_overflow())?)
        .ok_or_else(CudaRuntimeError::range_overflow)?;
    let right_end = right_start
        .checked_add(u64::try_from(right_len).map_err(|_| CudaRuntimeError::range_overflow())?)
        .ok_or_else(CudaRuntimeError::range_overflow)?;
    Ok(left_start < right_end && right_start < left_end)
}

fn error(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OP, kind)
}

fn parse_error() -> CudaRuntimeError {
    CudaRuntimeError::invalid_metadata(PARSE_OP)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn top1_views(rows: usize) -> (crate::CudaF32View, CudaBufferView) {
        let count = rows * QWEN35_DFLASH_ACCEPTANCE_VOCAB;
        (
            crate::CudaF32View::new(
                crate::cuda_runtime::test_buffer_view(0x100_0000, 0, 0, count * 4),
                augur_core::PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                count,
            )
            .unwrap(),
            crate::cuda_runtime::test_buffer_view(0x200_0000, 0, 0, rows * 4),
        )
    }

    #[test]
    fn vocab_top1_abi_and_complete_extent() {
        type Raw = augur_cuda_sys::AugurCudaQwen35VocabTop1ArgsV1;
        assert_eq!(core::mem::size_of::<Raw>(), 48);
        assert_eq!(core::mem::align_of::<Raw>(), 8);
        assert_eq!(core::mem::offset_of!(Raw, logits), 16);
        assert_eq!(core::mem::offset_of!(Raw, ids), 32);
        for rows in [1, 4, 8] {
            let (logits, ids) = top1_views(rows);
            let bound = CudaQwen35VocabTop1ArgsV1::bind_for_ordinal(0, logits, ids, rows).unwrap();
            assert_eq!(bound.raw.logits_len_bytes, (rows * 248_320 * 4) as u64);
            assert_eq!(bound.raw.ids_len_bytes, (rows * 4) as u64);
            assert_eq!(bound.raw.rows, rows as u32);
        }
    }

    #[test]
    fn vocab_top1_rejects_extent_ordinal_and_alias_drift() {
        let (logits, ids) = top1_views(1);
        for rows in [0, 2, usize::MAX] {
            assert!(CudaQwen35VocabTop1ArgsV1::bind_for_ordinal(0, logits, ids, rows).is_err());
        }
        assert!(CudaQwen35VocabTop1ArgsV1::bind_for_ordinal(1, logits, ids, 1).is_err());
        let alias = crate::cuda_runtime::test_buffer_view(0x100_0000, 0, 0, 4);
        assert!(CudaQwen35VocabTop1ArgsV1::bind_for_ordinal(0, logits, alias, 1).is_err());
        let unaligned = crate::cuda_runtime::test_buffer_view(0x200_0001, 0, 0, 4);
        assert!(CudaQwen35VocabTop1ArgsV1::bind_for_ordinal(0, logits, unaligned, 1).is_err());
    }

    #[test]
    #[ignore = "requires an SM70 CUDA device"]
    fn vocab_top1_gpu_full_vocab_ties_nonfinite_and_compact_readback(
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::{cuda_host_boundary_snapshot, OwnedCudaBuffer};
        const ROWS: usize = 8;
        let vocab = QWEN35_DFLASH_ACCEPTANCE_VOCAB;
        let mut values = vec![-100.0_f32; ROWS * vocab];
        values[25] = 3.0;
        values[vocab + vocab - 1] = 7.0;
        values[2 * vocab + 38317] = 9.0;
        values[2 * vocab + 214694] = 9.0; // tie must select 38317
        values[3 * vocab + 1] = f32::NAN;
        values[4 * vocab + vocab - 1] = f32::INFINITY;
        values[5 * vocab + 200_000] = f32::NEG_INFINITY;
        values[7 * vocab + 1001] = f32::MAX;
        let stream = OwnedCudaStream::create(0)?;
        let input = OwnedCudaBuffer::allocate(0, values.len() * 4)?;
        let output = OwnedCudaBuffer::allocate(0, ROWS * 4)?;
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        input.copy_from_host_sync(&stream, &bytes)?;
        let view = crate::CudaF32View::new(
            input.view(0, bytes.len())?,
            augur_core::PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            values.len(),
        )?;
        let before = cuda_host_boundary_snapshot();
        CudaQwen35VocabTop1ArgsV1::bind(&stream, view, output.view(0, ROWS * 4)?, ROWS)?
            .submit(&stream)?;
        stream.synchronize()?;
        let mut actual = [0_u8; ROWS * 4];
        output.copy_to_host_sync(&stream, &mut actual)?;
        let actual = actual
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            [25, 248319, 38317, u32::MAX, u32::MAX, u32::MAX, 0, 1001]
        );
        let delta = cuda_host_boundary_snapshot()
            .checked_delta_since(before)
            .ok_or("counter regression")?;
        assert_eq!(delta.synchronous_d2h_bytes, (ROWS * 4) as u64);
        assert_eq!(delta.synchronous_h2d_bytes, 0);
        drop(output);
        drop(input);
        Ok(())
    }

    #[test]
    #[ignore = "exclusive SM70 CUDA; hierarchical top1 widths, ties and poison lanes"]
    fn vocab_top1_gpu_all_widths_hierarchical_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
        use crate::{cuda_device_allocation_snapshot, OwnedCudaBuffer};
        let baseline = cuda_device_allocation_snapshot();
        let vocab = QWEN35_DFLASH_ACCEPTANCE_VOCAB;
        let stream = OwnedCudaStream::create(0)?;
        for rows in 1..=8 {
            let input = OwnedCudaBuffer::allocate(0, rows * vocab * 4)?;
            let output = OwnedCudaBuffer::allocate(0, (rows + 2) * 4)?;
            let view = crate::CudaF32View::new(input.view(0, rows * vocab * 4)?,
                augur_core::PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, rows * vocab)?;
            let op = CudaQwen35VocabTop1ArgsV1::bind(&stream, view, output.view(4, rows * 4)?, rows)?;
            for round in 0..8 {
                let mut values = vec![-100.0_f32; rows * vocab];
                let mut expected = Vec::new();
                for row in 0..rows {
                    let v = &mut values[row * vocab..(row + 1) * vocab];
                    let id = match (row + round) % 8 {
                        0 => { v[31] = 8.0; v[1024] = 8.0; v[vocab-1] = 8.0; 31 },
                        1 => { v[vocab-1] = f32::MAX; vocab as u32-1 },
                        2 => { v[1024] = 0.0; v[31] = -0.0; 31 },
                        3 => { v[1024] = f32::from_bits(1); v[31] = 0.0; 1024 },
                        4 => { v[1023] = f32::from_bits(0x7f80_0001); u32::MAX },
                        5 => { v[vocab-1] = f32::INFINITY; u32::MAX },
                        6 => { v[512] = f32::NEG_INFINITY; u32::MAX },
                        _ => { v.fill(-f32::MAX); 0 },
                    };
                    expected.push(id);
                }
                let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
                input.copy_from_host_sync(&stream, &bytes)?;
                output.copy_from_host_sync(&stream, &vec![0xa5; (rows + 2) * 4])?;
                let retained = cuda_device_allocation_snapshot();
                op.submit(&stream)?;
                stream.synchronize()?;
                let after = cuda_device_allocation_snapshot();
                assert_eq!(after.live_bytes, retained.live_bytes);
                assert_eq!(after.live_allocations, retained.live_allocations);
                let mut actual = vec![0_u8; (rows + 2) * 4];
                output.copy_to_host_sync(&stream, &mut actual)?;
                let actual = actual.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
                assert_eq!(actual[0], 0xa5a5_a5a5);
                assert_eq!(actual[rows+1], 0xa5a5_a5a5);
                assert_eq!(actual[1..rows+1], expected, "rows={rows} round={round}");
            }
        }
        drop(stream);
        let after = cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes, baseline.live_bytes);
        assert_eq!(after.live_allocations, baseline.live_allocations);
        Ok(())
    }

    #[test]
    #[ignore = "requires an SM70 CUDA device"]
    fn vocab_top1_gpu_feeds_k4_acceptance_without_id_or_logits_readback(
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::{cuda_host_boundary_snapshot, OwnedCudaBuffer};
        const ROWS: usize = 4;
        let vocab = QWEN35_DFLASH_ACCEPTANCE_VOCAB;
        let mut draft = vec![-10.0_f32; ROWS * vocab];
        let mut target = draft.clone();
        for (row, id) in [23379, 240000, 123, 247000].into_iter().enumerate() {
            draft[row * vocab + id] = 9.0;
        }
        for (row, id) in [23379, 240000, 999, 247000].into_iter().enumerate() {
            target[row * vocab + id] = 9.0;
        }
        let stream = OwnedCudaStream::create(0)?;
        let draft_logits = OwnedCudaBuffer::allocate(0, draft.len() * 4)?;
        let target_logits = OwnedCudaBuffer::allocate(0, target.len() * 4)?;
        let draft_ids = OwnedCudaBuffer::allocate(0, ROWS * 4)?;
        let target_ids = OwnedCudaBuffer::allocate(0, ROWS * 4)?;
        let receipt = OwnedCudaBuffer::allocate(0, QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES)?;
        draft_logits.copy_from_host_sync(
            &stream,
            &draft
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        target_logits.copy_from_host_sync(
            &stream,
            &target
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let top1 = CudaQwen35VocabTop1ArgsV1::bind(
            &stream,
            crate::CudaF32View::new(
                draft_logits.view(0, draft.len() * 4)?,
                augur_core::PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                draft.len(),
            )?,
            draft_ids.view(0, ROWS * 4)?,
            ROWS,
        )?;
        let acceptance = CudaQwen35DflashLogitsAcceptanceArgsV1::bind(
            &stream,
            CudaQwen35DflashLogitsAcceptanceViewsV1 {
                draft_ids: draft_ids.view(0, ROWS * 4)?,
                target_logits: target_logits.view(0, target.len() * 4)?,
                target_ids: target_ids.view(0, ROWS * 4)?,
                receipt: receipt.view(0, QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES)?,
            },
            ROWS,
            42,
            [7; 32],
        )?;
        let before = cuda_host_boundary_snapshot();
        top1.submit(&stream)?;
        acceptance.submit(&stream)?;
        stream.synchronize()?;
        let mut bytes = [0_u8; QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES];
        receipt.copy_to_host_sync(&stream, &mut bytes)?;
        let decision = parse_cuda_qwen35_dflash_acceptance_receipt_v1(&bytes, 42, [7; 32])?;
        assert_eq!(
            (
                decision.accepted_tokens,
                decision.rejected_position,
                decision.replacement_token
            ),
            (2, 2, 999)
        );
        let delta = cuda_host_boundary_snapshot()
            .checked_delta_since(before)
            .ok_or("counter regression")?;
        assert_eq!(delta.synchronous_d2h_bytes, 64);
        assert_eq!(delta.synchronous_h2d_bytes, 0);
        // A poisoned candidate row must not turn into an accepted token.
        draft[0] = f32::NAN;
        draft_logits.copy_from_host_sync(
            &stream,
            &draft
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        top1.submit(&stream)?;
        acceptance.submit(&stream)?;
        stream.synchronize()?;
        receipt.copy_to_host_sync(&stream, &mut bytes)?;
        assert!(parse_cuda_qwen35_dflash_acceptance_receipt_v1(&bytes, 42, [7; 32]).is_err());
        Ok(())
    }

    fn valid_receipt() -> [u8; QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES] {
        let mut bytes = [0_u8; QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES];
        bytes[0..4].copy_from_slice(&3_u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&3_u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&3_u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        bytes[16..20].copy_from_slice(
            &augur_cuda_sys::AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_STATUS_OK_V1.to_le_bytes(),
        );
        bytes[20..24].copy_from_slice(
            &augur_cuda_sys::AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ERROR_NONE_V1.to_le_bytes(),
        );
        bytes[24..32].copy_from_slice(&7_u64.to_le_bytes());
        bytes[32..64].copy_from_slice(&[1_u8; 32]);
        bytes
    }

    #[test]
    fn cpu_accepts_full_prefix() {
        assert_eq!(
            dflash_acceptance_cpu_v1(&[1, 2, 3], &[1, 2, 3]).unwrap(),
            (3, 3, u32::MAX)
        );
    }

    #[test]
    fn cpu_stops_at_first_rejection() {
        assert_eq!(
            dflash_acceptance_cpu_v1(&[1, 2, 3], &[1, 9, 8]).unwrap(),
            (1, 1, 9)
        );
    }

    #[test]
    fn cpu_logits_oracle_reduces_rows_before_acceptance() {
        let vocab = QWEN35_DFLASH_ACCEPTANCE_VOCAB;
        let mut logits = vec![-1.0_f32; 2 * vocab];
        logits[7] = 3.0;
        logits[vocab + 11] = 4.0;
        assert_eq!(
            dflash_logits_acceptance_cpu_v1(&[7, 11], &logits).unwrap(),
            (2, 2, u32::MAX)
        );
        assert_eq!(
            dflash_logits_acceptance_cpu_v1(&[7, 12], &logits).unwrap(),
            (1, 1, 11)
        );
    }

    #[test]
    fn cpu_logits_oracle_rejects_non_finite_rows() {
        let mut logits = vec![-1.0_f32; QWEN35_DFLASH_ACCEPTANCE_VOCAB];
        logits[3] = f32::NAN;
        assert!(dflash_logits_acceptance_cpu_v1(&[1], &logits).is_err());
    }

    #[test]
    fn parser_rejects_inconsistent_receipt() {
        let mut bytes = valid_receipt();
        bytes[4..8].copy_from_slice(&1_u32.to_le_bytes());
        assert!(parse_cuda_qwen35_dflash_acceptance_receipt_v1(&bytes, 7, [1; 32]).is_err());
    }

    #[test]
    fn parser_rejects_generation_and_digest_drift() {
        let bytes = valid_receipt();
        assert!(parse_cuda_qwen35_dflash_acceptance_receipt_v1(&bytes, 8, [1; 32]).is_err());
        assert!(parse_cuda_qwen35_dflash_acceptance_receipt_v1(&bytes, 7, [2; 32]).is_err());
    }

    #[test]
    fn parser_rejects_replacement_and_position_drift() {
        let mut bytes = valid_receipt();
        bytes[8..12].copy_from_slice(&2_u32.to_le_bytes());
        assert!(parse_cuda_qwen35_dflash_acceptance_receipt_v1(&bytes, 7, [1; 32]).is_err());

        let mut bytes = valid_receipt();
        bytes[4..8].copy_from_slice(&2_u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&2_u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&(QWEN35_DFLASH_ACCEPTANCE_VOCAB as u32).to_le_bytes());
        assert!(parse_cuda_qwen35_dflash_acceptance_receipt_v1(&bytes, 7, [1; 32]).is_err());

        let mut bytes = valid_receipt();
        bytes[12..16].copy_from_slice(&0_u32.to_le_bytes());
        assert!(parse_cuda_qwen35_dflash_acceptance_receipt_v1(&bytes, 7, [1; 32]).is_err());
    }

    #[test]
    fn parser_rejects_draft_count_overflow_and_bad_error_status() {
        let mut bytes = valid_receipt();
        bytes[0..4]
            .copy_from_slice(&(QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS as u32 + 1).to_le_bytes());
        assert!(parse_cuda_qwen35_dflash_acceptance_receipt_v1(&bytes, 7, [1; 32]).is_err());

        let mut bytes = valid_receipt();
        bytes[16..20].copy_from_slice(
            &augur_cuda_sys::AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_STATUS_ERROR_V1.to_le_bytes(),
        );
        assert!(parse_cuda_qwen35_dflash_acceptance_receipt_v1(&bytes, 7, [1; 32]).is_err());

        let mut bytes = valid_receipt();
        bytes[20..24].copy_from_slice(&1_u32.to_le_bytes());
        assert!(parse_cuda_qwen35_dflash_acceptance_receipt_v1(&bytes, 7, [1; 32]).is_err());
    }

    #[test]
    fn cpu_rejects_invalid_vocab_and_extent() {
        assert!(dflash_acceptance_cpu_v1(&[], &[]).is_err());
        assert!(dflash_acceptance_cpu_v1(&[1], &[]).is_err());
        assert!(dflash_acceptance_cpu_v1(&[QWEN35_DFLASH_ACCEPTANCE_VOCAB as u32], &[1],).is_err());
    }
}
