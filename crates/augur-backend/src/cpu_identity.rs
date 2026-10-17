//! Stable CPU identity used by compiler/tooling environment validation.
//!
//! This module detects hardware identity only. Thread-count and tuning-policy
//! metadata belong to `augur-compiler`.

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CpuIdentity {
    pub vendor: [u8; 12],
    pub family: u16,
    pub model: u16,
    pub stepping: u16,
}

/// Decodes x86 CPUID leaf-1 EAX into display family/model/stepping.
///
/// Intel and AMD use the same architectural display-family/model rules:
///
/// ```text
/// stepping    = EAX[3:0]
/// base_model  = EAX[7:4]
/// base_family = EAX[11:8]
/// ext_model   = EAX[19:16]
/// ext_family  = EAX[27:20]
///
/// display_family =
///     base_family + ext_family, if base_family == 0x0f
///     base_family,              otherwise
///
/// display_model =
///     base_model + (ext_model << 4),
///         if base_family == 0x06 or base_family == 0x0f
///     base_model,
///         otherwise
/// ```
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn decode_family_model_stepping(eax: u32) -> (u16, u16, u16) {
    let stepping = (eax & 0x0f) as u16;
    let base_model = ((eax >> 4) & 0x0f) as u16;
    let base_family = ((eax >> 8) & 0x0f) as u16;
    let extended_model = ((eax >> 16) & 0x0f) as u16;
    let extended_family = ((eax >> 20) & 0xff) as u16;

    let family = if base_family == 0x0f {
        base_family + extended_family
    } else {
        base_family
    };

    let model = if base_family == 0x06 || base_family == 0x0f {
        base_model + (extended_model << 4)
    } else {
        base_model
    };

    (family, model, stepping)
}

/// Detects the current CPU vendor/family/model/stepping.
///
/// Returns `None` outside x86/x86_64 or if the required CPUID leaves are not
/// available.
#[cfg(target_arch = "x86_64")]
pub fn detect_cpu_identity() -> Option<CpuIdentity> {
    use core::arch::x86_64::{__cpuid, __cpuid_count};

    // SAFETY:
    //
    // CPUID is available in the x86_64 architectural baseline. Leaves zero
    // and one return register values only and do not access memory.
    let leaf_zero = __cpuid(0);

    if leaf_zero.eax < 1 {
        return None;
    }

    let mut vendor = [0_u8; 12];
    vendor[0..4].copy_from_slice(&leaf_zero.ebx.to_le_bytes());
    vendor[4..8].copy_from_slice(&leaf_zero.edx.to_le_bytes());
    vendor[8..12].copy_from_slice(&leaf_zero.ecx.to_le_bytes());

    // SAFETY:
    //
    // Leaf one is supported because leaf_zero.eax >= 1. The intrinsic only
    // reads CPUID register output.
    let leaf_one = __cpuid_count(1, 0);

    let (family, model, stepping) = decode_family_model_stepping(leaf_one.eax);

    Some(CpuIdentity {
        vendor,
        family,
        model,
        stepping,
    })
}

#[cfg(target_arch = "x86")]
pub fn detect_cpu_identity() -> Option<CpuIdentity> {
    use core::arch::x86::{__cpuid, __cpuid_count};

    let leaf_zero = __cpuid(0);

    if leaf_zero.eax < 1 {
        return None;
    }

    let mut vendor = [0_u8; 12];
    vendor[0..4].copy_from_slice(&leaf_zero.ebx.to_le_bytes());
    vendor[4..8].copy_from_slice(&leaf_zero.edx.to_le_bytes());
    vendor[8..12].copy_from_slice(&leaf_zero.ecx.to_le_bytes());

    let leaf_one = __cpuid_count(1, 0);

    let (family, model, stepping) = decode_family_model_stepping(leaf_one.eax);

    Some(CpuIdentity {
        vendor,
        family,
        model,
        stepping,
    })
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn detect_cpu_identity() -> Option<CpuIdentity> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn encode_leaf_one(
        base_family: u32,
        base_model: u32,
        extended_family: u32,
        extended_model: u32,
        stepping: u32,
    ) -> u32 {
        (stepping & 0x0f)
            | ((base_model & 0x0f) << 4)
            | ((base_family & 0x0f) << 8)
            | ((extended_model & 0x0f) << 16)
            | ((extended_family & 0xff) << 20)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn decodes_family_six_extended_model() {
        let eax = encode_leaf_one(0x06, 0x0f, 0, 0x08, 0x08);
        assert_eq!(decode_family_model_stepping(eax), (6, 143, 8));
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn decodes_family_fifteen_extended_family_and_model() {
        let eax = encode_leaf_one(0x0f, 0x03, 0x02, 0x04, 0x07);
        assert_eq!(decode_family_model_stepping(eax), (0x11, 0x43, 0x07));
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn non_extended_family_uses_base_model() {
        let eax = encode_leaf_one(0x05, 0x09, 0x7f, 0x0e, 0x02);
        assert_eq!(decode_family_model_stepping(eax), (5, 9, 2));
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn host_identity_detection_is_stable() {
        let first = detect_cpu_identity().expect("x86 CPU identity");
        let second = detect_cpu_identity().expect("x86 CPU identity");
        assert_eq!(first, second);
        assert_ne!(first.vendor, [0; 12]);
    }
}
