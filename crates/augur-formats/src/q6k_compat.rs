use augur_core::{F16Bits, FormatVersion, PhysicalLayoutId, QuantFormatId};

use crate::{FormatError, ReferenceDecode};

pub const Q6K_BLOCK_SIZE: usize = 256;
pub const Q6K_QL_BYTES: usize = 128;
pub const Q6K_QH_BYTES: usize = 64;
pub const Q6K_SCALE_BYTES: usize = 16;
pub const Q6K_BLOCK_BYTES: usize = 210;
pub const Q6K_FORMAT_ID: QuantFormatId = QuantFormatId::Q6K;
pub const Q6K_FORMAT_VERSION: FormatVersion = FormatVersion(1);
pub const Q6K_COMPACT_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::Q6K_COMPACT_V1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q6KCompactBlock {
    pub ql: [u8; Q6K_QL_BYTES],
    pub qh: [u8; Q6K_QH_BYTES],
    pub scales: [i8; Q6K_SCALE_BYTES],
    pub d: F16Bits,
}

impl Q6KCompactBlock {
    pub const DECODED_LEN: usize = Q6K_BLOCK_SIZE;

    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        if bytes.len() != Q6K_BLOCK_BYTES {
            return Err(FormatError::InputLengthMismatch {
                expected: Q6K_BLOCK_BYTES,
                actual: bytes.len(),
            });
        }
        let mut ql = [0; Q6K_QL_BYTES];
        ql.copy_from_slice(&bytes[..128]);
        let mut qh = [0; Q6K_QH_BYTES];
        qh.copy_from_slice(&bytes[128..192]);
        let mut scales = [0_i8; Q6K_SCALE_BYTES];
        for (dst, src) in scales.iter_mut().zip(&bytes[192..208]) {
            *dst = *src as i8;
        }
        let d = F16Bits::from_bits(u16::from_le_bytes([bytes[208], bytes[209]]));
        if !d.to_f32().is_finite() {
            return Err(FormatError::NonFiniteValue);
        }
        Ok(Self { ql, qh, scales, d })
    }

    pub fn quant_at(&self, index: usize) -> i8 {
        debug_assert!(index < Q6K_BLOCK_SIZE);
        let half = index / 128;
        let local = index % 128;
        let quarter = local / 32;
        let lane = local % 32;
        let low_base = half * 64;
        let high = self.qh[half * 32 + lane];
        let quant = match quarter {
            0 => (self.ql[low_base + lane] & 0x0f) | ((high & 0x03) << 4),
            1 => (self.ql[low_base + 32 + lane] & 0x0f) | (((high >> 2) & 0x03) << 4),
            2 => (self.ql[low_base + lane] >> 4) | (((high >> 4) & 0x03) << 4),
            3 => (self.ql[low_base + 32 + lane] >> 4) | (((high >> 6) & 0x03) << 4),
            _ => unreachable!(),
        };
        quant as i8 - 32
    }

    pub fn scale_at(&self, index: usize) -> i8 {
        let half = index / 128;
        let local = index % 128;
        let quarter = local / 32;
        let lane = local % 32;
        self.scales[half * 8 + quarter * 2 + lane / 16]
    }
}

impl ReferenceDecode for Q6KCompactBlock {
    type Elem = f32;

    fn decoded_len(&self) -> usize {
        Self::DECODED_LEN
    }

    fn decode_ref(&self, dst: &mut [f32]) -> Result<(), FormatError> {
        if dst.len() < Self::DECODED_LEN {
            return Err(FormatError::OutputTooSmall {
                expected: Self::DECODED_LEN,
                actual: dst.len(),
            });
        }
        let d = self.d.to_f32();
        for (index, output) in dst.iter_mut().take(Self::DECODED_LEN).enumerate() {
            *output = d * self.scale_at(index) as f32 * self.quant_at(index) as f32;
        }
        Ok(())
    }
}
