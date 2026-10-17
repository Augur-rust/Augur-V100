#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct F16Bits(pub u16);

impl F16Bits {
    pub const ZERO: Self = Self(0);

    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }
    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    pub fn to_f32(self) -> f32 {
        let h = self.0;
        let sign = ((h & 0x8000) as u32) << 16;
        let exp = (h >> 10) & 0x1f;
        let frac = h & 0x03ff;

        let bits = match exp {
            0 => {
                if frac == 0 {
                    sign
                } else {
                    let mut f = frac as u32;
                    let mut e = -14i32;
                    while (f & 0x0400) == 0 {
                        f <<= 1;
                        e -= 1;
                    }
                    f &= 0x03ff;
                    sign | (((e + 127) as u32) << 23) | (f << 13)
                }
            }
            0x1f => sign | (0xffu32 << 23) | ((frac as u32) << 13),
            _ => sign | (((exp as u32) + (127 - 15)) << 23) | ((frac as u32) << 13),
        };

        f32::from_bits(bits)
    }

    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let sign = (bits >> 16) as u16 & 0x8000;
        let exp = ((bits >> 23) & 0xff) as i32;
        let frac = bits & 0x007f_ffff;

        if exp == 0 {
            return Self(sign);
        }

        if exp == 0xff {
            return if frac == 0 {
                Self(sign | 0x7c00)
            } else {
                Self(sign | 0x7e00)
            };
        }

        let unbiased = exp - 127;
        if unbiased > 15 {
            return Self(sign | 0x7c00);
        }

        if unbiased < -14 {
            return Self(sign);
        }

        let f16_exp = ((unbiased + 15) as u16) << 10;
        let rounding = 0x0000_1000u32;
        let mantissa = (frac + rounding) >> 13;
        Self(sign | f16_exp | (mantissa as u16 & 0x03ff))
    }
}
