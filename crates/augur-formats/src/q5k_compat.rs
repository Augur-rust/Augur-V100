use augur_core::{F16Bits, FormatVersion, PhysicalLayoutId, QuantFormatId};

use crate::{unpack_q4k_scales_ref, FormatError, ReferenceDecode};

pub const Q5K_BLOCK_SIZE: usize = 256;
pub const Q5K_SCALE_BYTES: usize = 12;
pub const Q5K_QH_BYTES: usize = 32;
pub const Q5K_QS_BYTES: usize = 128;
pub const Q5K_BLOCK_BYTES: usize = 176;
pub const Q5K_FORMAT_ID: QuantFormatId = QuantFormatId::Q5K;
pub const Q5K_FORMAT_VERSION: FormatVersion = FormatVersion(1);
pub const Q5K_COMPACT_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::Q5K_COMPACT_V1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q5KCompactBlock {
    pub d: F16Bits,
    pub dmin: F16Bits,
    pub scales: [u8; Q5K_SCALE_BYTES],
    pub qh: [u8; Q5K_QH_BYTES],
    pub qs: [u8; Q5K_QS_BYTES],
}

impl Q5KCompactBlock {
    pub const DECODED_LEN: usize = Q5K_BLOCK_SIZE;

    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        if bytes.len() != Q5K_BLOCK_BYTES {
            return Err(FormatError::InputLengthMismatch {
                expected: Q5K_BLOCK_BYTES,
                actual: bytes.len(),
            });
        }
        let d = F16Bits::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]));
        let dmin = F16Bits::from_bits(u16::from_le_bytes([bytes[2], bytes[3]]));
        if !d.to_f32().is_finite() || !dmin.to_f32().is_finite() {
            return Err(FormatError::NonFiniteValue);
        }
        let mut scales = [0; Q5K_SCALE_BYTES];
        scales.copy_from_slice(&bytes[4..16]);
        let mut qh = [0; Q5K_QH_BYTES];
        qh.copy_from_slice(&bytes[16..48]);
        let mut qs = [0; Q5K_QS_BYTES];
        qs.copy_from_slice(&bytes[48..]);
        Ok(Self {
            d,
            dmin,
            scales,
            qh,
            qs,
        })
    }

    pub fn quant_at(&self, index: usize) -> u8 {
        debug_assert!(index < Q5K_BLOCK_SIZE);
        let chunk = index / 64;
        let within = index % 64;
        let half = within / 32;
        let lane = within % 32;
        let packed = self.qs[chunk * 32 + lane];
        let low = if half == 0 {
            packed & 0x0f
        } else {
            packed >> 4
        };
        let high = (self.qh[lane] >> (2 * chunk + half)) & 1;
        low | (high << 4)
    }
}

impl ReferenceDecode for Q5KCompactBlock {
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
        let metadata = unpack_q4k_scales_ref(&self.scales);
        let d = self.d.to_f32();
        let dmin = self.dmin.to_f32();
        for (index, output) in dst.iter_mut().take(Self::DECODED_LEN).enumerate() {
            let group = index / 32;
            *output = d * metadata.scale[group] as f32 * self.quant_at(index) as f32
                - dmin * metadata.min[group] as f32;
        }
        Ok(())
    }
}
