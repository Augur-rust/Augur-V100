use augur_formats::{unpack_q4k_scales_ref, FormatError, Q4KCompactBlock, Q4K_QS_BYTES};
use augur_kernel_api::{Q4KHotCpuBlock, Q4K_HOT_CPU_BLOCK_LEN, Q4K_HOT_CPU_QS_BYTES};

pub use crate::error::TensorError;

pub fn repack_q4k_block_cpu_ref(src: &Q4KCompactBlock) -> Q4KHotCpuBlock {
    debug_assert_eq!(Q4KCompactBlock::DECODED_LEN, Q4K_HOT_CPU_BLOCK_LEN);
    debug_assert_eq!(Q4K_QS_BYTES, Q4K_HOT_CPU_QS_BYTES);

    let scales = unpack_q4k_scales_ref(&src.scales);
    Q4KHotCpuBlock {
        d: src.d,
        dmin: src.dmin,
        scale: scales.scale,
        min: scales.min,
        qs: src.qs,
    }
}

pub fn decode_q4k_hot_cpu_block_ref(
    block: &Q4KHotCpuBlock,
    dst: &mut [f32],
) -> Result<(), FormatError> {
    if dst.len() < Q4KHotCpuBlock::DECODED_LEN {
        return Err(FormatError::OutputTooSmall {
            expected: Q4KHotCpuBlock::DECODED_LEN,
            actual: dst.len(),
        });
    }

    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();

    for (i, out) in dst.iter_mut().take(Q4KHotCpuBlock::DECODED_LEN).enumerate() {
        let group = i / 32;
        let q = block.quant_at(i);
        *out = d * (block.scale[group] as f32) * (q as f32) - dmin * (block.min[group] as f32);
    }
    Ok(())
}
