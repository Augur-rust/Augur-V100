use augur_formats::{
    decode_e4m3fn, decode_e8m0, Fp8E4M3CompactBlock, Fp8E4M3ScaleGridCompactTensor,
    Mxfp4CompactBlock, Nvfp4CompactBlock,
};
use augur_kernel_api::{
    Fp8E4M3HotCpuBlock, Fp8E4M3HotCpuTensor, Fp8E4M3ScaleGridHotCpuTensor, Mxfp4HotCpuBlock,
    Mxfp4HotCpuTensor, Nvfp4HotCpuBlock, Nvfp4HotCpuTensor,
};

use crate::RepackError;

pub fn repack_mxfp4_block_cpu(source: &Mxfp4CompactBlock) -> Result<Mxfp4HotCpuBlock, RepackError> {
    Ok(Mxfp4HotCpuBlock {
        scale: decode_e8m0(source.scale_e8m0)?,
        qs: source.qs,
    })
}

pub fn repack_nvfp4_block_cpu(source: &Nvfp4CompactBlock) -> Result<Nvfp4HotCpuBlock, RepackError> {
    if !source.global_scale.is_finite() {
        return Err(augur_formats::FormatError::NonFiniteValue.into());
    }
    let scale = decode_e4m3fn(source.scale_e4m3)? * source.global_scale;
    if !scale.is_finite() {
        return Err(augur_formats::FormatError::NonFiniteValue.into());
    }
    Ok(Nvfp4HotCpuBlock {
        scale,
        qs: source.qs,
    })
}

pub fn repack_mxfp4_tensor_cpu(
    source: &[Mxfp4CompactBlock],
    rows: usize,
    cols: usize,
) -> Result<Mxfp4HotCpuTensor, RepackError> {
    let blocks = source
        .iter()
        .map(repack_mxfp4_block_cpu)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Mxfp4HotCpuTensor::new(rows, cols, blocks)?)
}

pub fn repack_nvfp4_tensor_cpu(
    source: &[Nvfp4CompactBlock],
    rows: usize,
    cols: usize,
) -> Result<Nvfp4HotCpuTensor, RepackError> {
    let blocks = source
        .iter()
        .map(repack_nvfp4_block_cpu)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Nvfp4HotCpuTensor::new(rows, cols, blocks)?)
}

pub fn repack_fp8_e4m3_block_cpu(
    source: &Fp8E4M3CompactBlock,
) -> Result<Fp8E4M3HotCpuBlock, RepackError> {
    for value in source.qs {
        decode_e4m3fn(value)?;
    }
    Ok(Fp8E4M3HotCpuBlock {
        scale: decode_e8m0(source.scale_e8m0)?,
        qs: source.qs,
    })
}

pub fn repack_fp8_e4m3_tensor_cpu(
    source: &[Fp8E4M3CompactBlock],
    rows: usize,
    cols: usize,
) -> Result<Fp8E4M3HotCpuTensor, RepackError> {
    let blocks = source
        .iter()
        .map(repack_fp8_e4m3_block_cpu)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Fp8E4M3HotCpuTensor::new(rows, cols, blocks)?)
}

pub fn repack_fp8_e4m3_scale_grid_tensor_cpu(
    source: &Fp8E4M3ScaleGridCompactTensor,
) -> Result<Fp8E4M3ScaleGridHotCpuTensor, RepackError> {
    for value in source.values() {
        decode_e4m3fn(*value)?;
    }
    let scales = source
        .scales_e8m0()
        .iter()
        .map(|value| decode_e8m0(*value))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Fp8E4M3ScaleGridHotCpuTensor::new(
        source.rows(),
        source.cols(),
        source.values().to_vec(),
        scales,
    )?)
}
