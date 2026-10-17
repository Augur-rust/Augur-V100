#![cfg(all(feature = "cuda", feature = "qwen35-device-token-handoff"))]

use crate::{CudaBufferView, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};
pub use augur_cuda_sys::{
    AugurCudaQwen35TokenHandoffArgsV1 as CudaQwen35RawTokenHandoffArgsV1,
    AugurCudaQwen35TokenHandoffReceiptV1 as CudaQwen35RawTokenHandoffReceiptV1,
};

pub const QWEN35_TOKEN_HANDOFF_VOCAB: usize =
    augur_cuda_sys::AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_VOCAB_V1 as usize;
pub const QWEN35_TOKEN_HANDOFF_LOGITS_BYTES: usize =
    augur_cuda_sys::AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_LOGITS_BYTES_V1 as usize;
pub const QWEN35_TOKEN_HANDOFF_PARTIAL_BYTES: usize =
    augur_cuda_sys::AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_PARTIAL_BYTES_V1 as usize;
pub const QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES: usize =
    augur_cuda_sys::AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES_V1 as usize;

const OP: &str = "cuda_qwen35_token_handoff_submit_v1";
const PARSE_OP: &str = "cuda_qwen35_token_handoff_parse_v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaQwen35TokenHandoffViewsV1 {
    pub logits: CudaBufferView,
    pub partials: CudaBufferView,
    pub receipt: CudaBufferView,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35TokenHandoffArgsV1 {
    raw: CudaQwen35RawTokenHandoffArgsV1,
    device_ordinal: i32,
}

impl CudaQwen35TokenHandoffArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        views: CudaQwen35TokenHandoffViewsV1,
        step_generation: u64,
        plan_digest: [u8; 32],
    ) -> Result<Self, CudaRuntimeError> {
        let CudaQwen35TokenHandoffViewsV1 {
            logits,
            partials,
            receipt,
        } = views;
        let ordinal = stream.device_ordinal();
        if step_generation == 0
            || plan_digest == [0; 32]
            || logits.device_ordinal() != ordinal
            || partials.device_ordinal() != ordinal
            || receipt.device_ordinal() != ordinal
            || logits.len_bytes() != QWEN35_TOKEN_HANDOFF_LOGITS_BYTES
            || partials.len_bytes() != QWEN35_TOKEN_HANDOFF_PARTIAL_BYTES
            || receipt.len_bytes() != QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }

        let logits_address = address(logits)?;
        let partials_address = address(partials)?;
        let receipt_address = address(receipt)?;
        if !logits_address.is_multiple_of(4)
            || !partials_address.is_multiple_of(16)
            || !receipt_address.is_multiple_of(8)
            || ranges_overlap(
                logits_address,
                QWEN35_TOKEN_HANDOFF_LOGITS_BYTES,
                partials_address,
                QWEN35_TOKEN_HANDOFF_PARTIAL_BYTES,
            )?
            || ranges_overlap(
                logits_address,
                QWEN35_TOKEN_HANDOFF_LOGITS_BYTES,
                receipt_address,
                QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES,
            )?
            || ranges_overlap(
                partials_address,
                QWEN35_TOKEN_HANDOFF_PARTIAL_BYTES,
                receipt_address,
                QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES,
            )?
        {
            return Err(error(CudaRuntimeErrorKind::InvalidLayout));
        }

        let mut raw = CudaQwen35RawTokenHandoffArgsV1::new();
        raw.logits = logits_address;
        raw.partials = partials_address;
        raw.receipt = receipt_address;
        raw.step_generation = step_generation;
        raw.plan_digest = plan_digest;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn raw(&self) -> &CudaQwen35RawTokenHandoffArgsV1 {
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
            augur_cuda_sys::augur_cuda_qwen35_token_handoff_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OP, result)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaQwen35Top2V1 {
    pub token_id: u32,
    pub top1_value_bits: u32,
    pub runner_up_id: u32,
    pub runner_up_value_bits: u32,
}

impl CudaQwen35Top2V1 {
    pub fn top1_value(self) -> f32 {
        f32::from_bits(self.top1_value_bits)
    }

    pub fn runner_up_value(self) -> f32 {
        f32::from_bits(self.runner_up_value_bits)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaQwen35TokenHandoffOutcomeV1 {
    Selected {
        top2: CudaQwen35Top2V1,
        finite_count: u32,
        step_generation: u64,
        plan_digest: [u8; 32],
    },
    NonFiniteLogit {
        finite_count: u32,
        first_non_finite_id: u32,
        step_generation: u64,
        plan_digest: [u8; 32],
    },
}

pub fn parse_cuda_qwen35_token_handoff_receipt_v1(
    bytes: &[u8],
    expected_step_generation: u64,
    expected_plan_digest: [u8; 32],
) -> Result<CudaQwen35TokenHandoffOutcomeV1, CudaRuntimeError> {
    if bytes.len() != QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES
        || expected_step_generation == 0
        || expected_plan_digest == [0; 32]
    {
        return Err(parse_error());
    }
    let word = |offset: usize| {
        u32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .expect("validated 72-byte Qwen35 token-handoff receipt"),
        )
    };
    let token_id = word(0);
    let top1_value_bits = word(4);
    let runner_up_id = word(8);
    let runner_up_value_bits = word(12);
    let finite_count = word(16);
    let first_non_finite_id = word(20);
    let status = word(24);
    let error_code = word(28);
    let step_generation = u64::from_le_bytes(
        bytes[32..40]
            .try_into()
            .expect("validated 72-byte Qwen35 token-handoff receipt"),
    );
    let mut plan_digest = [0_u8; 32];
    plan_digest.copy_from_slice(&bytes[40..72]);
    if step_generation != expected_step_generation || plan_digest != expected_plan_digest {
        return Err(parse_error());
    }

    match status {
        augur_cuda_sys::AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_STATUS_OK_V1 => {
            let top1 = f32::from_bits(top1_value_bits);
            let runner_up = f32::from_bits(runner_up_value_bits);
            if error_code != augur_cuda_sys::AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ERROR_NONE_V1
                || finite_count as usize != QWEN35_TOKEN_HANDOFF_VOCAB
                || first_non_finite_id != u32::MAX
                || token_id as usize >= QWEN35_TOKEN_HANDOFF_VOCAB
                || runner_up_id as usize >= QWEN35_TOKEN_HANDOFF_VOCAB
                || token_id == runner_up_id
                || !top1.is_finite()
                || !runner_up.is_finite()
                || !(top1 > runner_up || (top1 == runner_up && token_id < runner_up_id))
            {
                return Err(parse_error());
            }
            Ok(CudaQwen35TokenHandoffOutcomeV1::Selected {
                top2: CudaQwen35Top2V1 {
                    token_id,
                    top1_value_bits,
                    runner_up_id,
                    runner_up_value_bits,
                },
                finite_count,
                step_generation,
                plan_digest,
            })
        }
        augur_cuda_sys::AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_STATUS_ERROR_V1 => {
            if error_code != augur_cuda_sys::AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ERROR_NON_FINITE_LOGIT_V1
                || finite_count as usize >= QWEN35_TOKEN_HANDOFF_VOCAB
                || first_non_finite_id as usize >= QWEN35_TOKEN_HANDOFF_VOCAB
                || token_id != u32::MAX
                || runner_up_id != u32::MAX
                || top1_value_bits != 0
                || runner_up_value_bits != 0
            {
                return Err(parse_error());
            }
            Ok(CudaQwen35TokenHandoffOutcomeV1::NonFiniteLogit {
                finite_count,
                first_non_finite_id,
                step_generation,
                plan_digest,
            })
        }
        _ => Err(parse_error()),
    }
}

pub fn cuda_qwen35_finite_top2_cpu_v1(
    logits: &[f32],
) -> Result<CudaQwen35Top2V1, CudaRuntimeError> {
    if logits.len() != QWEN35_TOKEN_HANDOFF_VOCAB {
        return Err(parse_error());
    }
    let mut top1: Option<(usize, u32)> = None;
    let mut top2: Option<(usize, u32)> = None;
    for (id, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(parse_error());
        }
        let candidate = (id, value.to_bits());
        if top1.is_none_or(|current| candidate_better(candidate, current)) {
            top2 = top1;
            top1 = Some(candidate);
        } else if top2.is_none_or(|current| candidate_better(candidate, current)) {
            top2 = Some(candidate);
        }
    }
    let (token_id, top1_value_bits) = top1.ok_or_else(parse_error)?;
    let (runner_up_id, runner_up_value_bits) = top2.ok_or_else(parse_error)?;
    Ok(CudaQwen35Top2V1 {
        token_id: u32::try_from(token_id).map_err(|_| CudaRuntimeError::range_overflow())?,
        top1_value_bits,
        runner_up_id: u32::try_from(runner_up_id)
            .map_err(|_| CudaRuntimeError::range_overflow())?,
        runner_up_value_bits,
    })
}

fn candidate_better(left: (usize, u32), right: (usize, u32)) -> bool {
    let left_value = f32::from_bits(left.1);
    let right_value = f32::from_bits(right.1);
    left_value > right_value || (left_value == right_value && left.0 < right.0)
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
    use crate::cuda_runtime::test_buffer_view;

    const DIGEST: [u8; 32] = [0x71; 32];

    fn valid_views() -> CudaQwen35TokenHandoffViewsV1 {
        CudaQwen35TokenHandoffViewsV1 {
            logits: test_buffer_view(0x10_0000, 0, 0, QWEN35_TOKEN_HANDOFF_LOGITS_BYTES),
            partials: test_buffer_view(0x20_0000, 0, 0, QWEN35_TOKEN_HANDOFF_PARTIAL_BYTES),
            receipt: test_buffer_view(0x30_0000, 0, 0, QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES),
        }
    }

    fn receipt_bytes(raw: CudaQwen35RawTokenHandoffReceiptV1) -> [u8; 72] {
        let mut bytes = [0_u8; 72];
        bytes[0..4].copy_from_slice(&raw.token_id.to_le_bytes());
        bytes[4..8].copy_from_slice(&raw.top1_value_bits.to_le_bytes());
        bytes[8..12].copy_from_slice(&raw.runner_up_id.to_le_bytes());
        bytes[12..16].copy_from_slice(&raw.runner_up_value_bits.to_le_bytes());
        bytes[16..20].copy_from_slice(&raw.finite_count.to_le_bytes());
        bytes[20..24].copy_from_slice(&raw.first_non_finite_id.to_le_bytes());
        bytes[24..28].copy_from_slice(&raw.status.to_le_bytes());
        bytes[28..32].copy_from_slice(&raw.error_code.to_le_bytes());
        bytes[32..40].copy_from_slice(&raw.step_generation.to_le_bytes());
        bytes[40..72].copy_from_slice(&raw.plan_digest);
        bytes
    }

    fn selected_raw() -> CudaQwen35RawTokenHandoffReceiptV1 {
        CudaQwen35RawTokenHandoffReceiptV1 {
            token_id: 3,
            top1_value_bits: 9.0_f32.to_bits(),
            runner_up_id: 7,
            runner_up_value_bits: 9.0_f32.to_bits(),
            finite_count: QWEN35_TOKEN_HANDOFF_VOCAB as u32,
            first_non_finite_id: u32::MAX,
            status: augur_cuda_sys::AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_STATUS_OK_V1,
            error_code: augur_cuda_sys::AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ERROR_NONE_V1,
            step_generation: 17,
            plan_digest: DIGEST,
        }
    }

    #[test]
    fn binding_uses_absolute_view_addresses_and_frozen_extents() {
        let stream = OwnedCudaStream::create(0).unwrap();
        let mut views = valid_views();
        views.logits = test_buffer_view(0x10_0000, 0, 64, QWEN35_TOKEN_HANDOFF_LOGITS_BYTES);
        let binding = CudaQwen35TokenHandoffArgsV1::bind(&stream, views, 17, DIGEST).unwrap();
        assert_eq!(binding.raw().logits, 0x10_0040);
        assert_eq!(
            binding.raw().logits_len_bytes,
            QWEN35_TOKEN_HANDOFF_LOGITS_BYTES as u64
        );
        assert_eq!(
            binding.raw().partials_len_bytes,
            QWEN35_TOKEN_HANDOFF_PARTIAL_BYTES as u64
        );
        assert_eq!(
            binding.raw().receipt_len_bytes,
            QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES as u64
        );
    }

    #[test]
    fn binding_rejects_zero_identity_wrong_extent_alignment_alias_and_device() {
        let stream = OwnedCudaStream::create(0).unwrap();
        assert!(CudaQwen35TokenHandoffArgsV1::bind(&stream, valid_views(), 0, DIGEST).is_err());
        assert!(CudaQwen35TokenHandoffArgsV1::bind(&stream, valid_views(), 1, [0; 32]).is_err());

        let mut views = valid_views();
        views.logits = test_buffer_view(0x10_0000, 0, 0, QWEN35_TOKEN_HANDOFF_LOGITS_BYTES - 4);
        assert!(CudaQwen35TokenHandoffArgsV1::bind(&stream, views, 1, DIGEST).is_err());
        let mut views = valid_views();
        views.partials = test_buffer_view(0x20_0001, 0, 0, QWEN35_TOKEN_HANDOFF_PARTIAL_BYTES);
        assert!(CudaQwen35TokenHandoffArgsV1::bind(&stream, views, 1, DIGEST).is_err());
        let mut views = valid_views();
        views.receipt = test_buffer_view(0x20_0000, 0, 128, QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES);
        assert!(CudaQwen35TokenHandoffArgsV1::bind(&stream, views, 1, DIGEST).is_err());
        let mut views = valid_views();
        views.receipt = test_buffer_view(0x30_0000, 1, 0, QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES);
        assert!(CudaQwen35TokenHandoffArgsV1::bind(&stream, views, 1, DIGEST).is_err());
    }

    #[test]
    fn cpu_oracle_uses_first_index_ties_and_preserves_signed_zero_bits() {
        let mut logits = vec![-1.0_f32; QWEN35_TOKEN_HANDOFF_VOCAB];
        logits[7] = -0.0;
        logits[3] = 0.0;
        let top2 = cuda_qwen35_finite_top2_cpu_v1(&logits).unwrap();
        assert_eq!(top2.token_id, 3);
        assert_eq!(top2.top1_value_bits, 0.0_f32.to_bits());
        assert_eq!(top2.runner_up_id, 7);
        assert_eq!(top2.runner_up_value_bits, (-0.0_f32).to_bits());
    }

    #[test]
    fn cpu_oracle_handles_vocabulary_edges_negatives_and_non_finite_values() {
        let mut logits = vec![-9.0_f32; QWEN35_TOKEN_HANDOFF_VOCAB];
        logits[QWEN35_TOKEN_HANDOFF_VOCAB - 1] = f32::from_bits(1);
        logits[0] = 0.0;
        let top2 = cuda_qwen35_finite_top2_cpu_v1(&logits).unwrap();
        assert_eq!(top2.token_id, QWEN35_TOKEN_HANDOFF_VOCAB as u32 - 1);
        assert_eq!(top2.runner_up_id, 0);
        logits[12] = f32::NAN;
        assert!(cuda_qwen35_finite_top2_cpu_v1(&logits).is_err());
    }

    #[test]
    fn parser_accepts_selected_and_canonical_non_finite_receipts() {
        let selected =
            parse_cuda_qwen35_token_handoff_receipt_v1(&receipt_bytes(selected_raw()), 17, DIGEST)
                .unwrap();
        assert!(matches!(
            selected,
            CudaQwen35TokenHandoffOutcomeV1::Selected { top2, finite_count, .. }
                if top2.token_id == 3
                    && top2.runner_up_id == 7
                    && finite_count as usize == QWEN35_TOKEN_HANDOFF_VOCAB
        ));

        let error = CudaQwen35RawTokenHandoffReceiptV1 {
            token_id: u32::MAX,
            top1_value_bits: 0,
            runner_up_id: u32::MAX,
            runner_up_value_bits: 0,
            finite_count: QWEN35_TOKEN_HANDOFF_VOCAB as u32 - 1,
            first_non_finite_id: 12,
            status: augur_cuda_sys::AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_STATUS_ERROR_V1,
            error_code: augur_cuda_sys::AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ERROR_NON_FINITE_LOGIT_V1,
            step_generation: 17,
            plan_digest: DIGEST,
        };
        assert!(matches!(
            parse_cuda_qwen35_token_handoff_receipt_v1(&receipt_bytes(error), 17, DIGEST).unwrap(),
            CudaQwen35TokenHandoffOutcomeV1::NonFiniteLogit {
                first_non_finite_id: 12,
                ..
            }
        ));
    }

    #[test]
    fn parser_rejects_stale_unwritten_incomplete_and_malformed_receipts() {
        let bytes = receipt_bytes(selected_raw());
        assert!(parse_cuda_qwen35_token_handoff_receipt_v1(&bytes, 18, DIGEST).is_err());
        assert!(parse_cuda_qwen35_token_handoff_receipt_v1(&bytes, 17, [0x6b; 32]).is_err());
        assert!(parse_cuda_qwen35_token_handoff_receipt_v1(&[0; 72], 17, DIGEST).is_err());

        let mut malformed = selected_raw();
        malformed.finite_count -= 1;
        assert!(
            parse_cuda_qwen35_token_handoff_receipt_v1(&receipt_bytes(malformed), 17, DIGEST)
                .is_err()
        );
        let mut malformed = selected_raw();
        malformed.runner_up_id = malformed.token_id;
        assert!(
            parse_cuda_qwen35_token_handoff_receipt_v1(&receipt_bytes(malformed), 17, DIGEST)
                .is_err()
        );
        let mut malformed = selected_raw();
        malformed.top1_value_bits = f32::INFINITY.to_bits();
        assert!(
            parse_cuda_qwen35_token_handoff_receipt_v1(&receipt_bytes(malformed), 17, DIGEST)
                .is_err()
        );
        let mut incomplete = selected_raw();
        incomplete.status = augur_cuda_sys::AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_STATUS_ERROR_V1;
        incomplete.error_code =
            augur_cuda_sys::AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ERROR_INCOMPLETE_REDUCTION_V1;
        incomplete.token_id = u32::MAX;
        incomplete.runner_up_id = u32::MAX;
        incomplete.top1_value_bits = 0;
        incomplete.runner_up_value_bits = 0;
        assert!(
            parse_cuda_qwen35_token_handoff_receipt_v1(&receipt_bytes(incomplete), 17, DIGEST)
                .is_err()
        );
    }
}
