//! Compact-domain M46 FP4 correctness oracles.

use augur_formats::{
    decode_e4m3fn, decode_e8m0, Fp8E4M3CompactBlock, Fp8E4M3ScaleGridCompactTensor,
    Mxfp4CompactBlock, Nvfp4CompactBlock, ReferenceDecode, FP8_E4M3_BLOCK_SIZE,
    FP8_E4M3_SCALE_TILE, MXFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE,
};

use crate::ReferenceQ8Error;
use augur_kernel_api::FusedMlpF32Config;

pub fn qmatvec_mxfp4_f32_ref(
    blocks: &[Mxfp4CompactBlock],
    rows: usize,
    cols: usize,
    activation: &[f32],
    output: &mut [f32],
) -> Result<(), ReferenceQ8Error> {
    qmatvec_ref(blocks, rows, cols, MXFP4_BLOCK_SIZE, activation, output)
}

pub fn qmatvec_nvfp4_f32_ref(
    blocks: &[Nvfp4CompactBlock],
    rows: usize,
    cols: usize,
    activation: &[f32],
    output: &mut [f32],
) -> Result<(), ReferenceQ8Error> {
    qmatvec_ref(blocks, rows, cols, NVFP4_BLOCK_SIZE, activation, output)
}

pub fn qmatvec_fp8_e4m3_f32_ref(
    blocks: &[Fp8E4M3CompactBlock],
    rows: usize,
    cols: usize,
    activation: &[f32],
    output: &mut [f32],
) -> Result<(), ReferenceQ8Error> {
    qmatvec_ref(blocks, rows, cols, FP8_E4M3_BLOCK_SIZE, activation, output)
}

pub fn qmatvec_fp8_e4m3_scale_grid_f32_ref(
    weight: &Fp8E4M3ScaleGridCompactTensor,
    activation: &[f32],
    output: &mut [f32],
) -> Result<(), ReferenceQ8Error> {
    if activation.len() != weight.cols() {
        return Err(ReferenceQ8Error::BadActivationLength {
            expected: weight.cols(),
            actual: activation.len(),
        });
    }
    if output.len() < weight.rows() {
        return Err(ReferenceQ8Error::BadOutputLength {
            expected: weight.rows(),
            actual: output.len(),
        });
    }
    for value in weight.values() {
        decode_e4m3fn(*value).map_err(|_| ReferenceQ8Error::BadWeight)?;
    }
    for scale in weight.scales_e8m0() {
        decode_e8m0(*scale).map_err(|_| ReferenceQ8Error::BadWeight)?;
    }
    for (row, output_value) in output.iter_mut().enumerate().take(weight.rows()) {
        let values = weight.row_values(row).ok_or(ReferenceQ8Error::BadWeight)?;
        let mut sum = 0.0_f32;
        for block in 0..weight.scale_cols() {
            let scale = decode_e8m0(
                weight
                    .scale_e8m0(row, block)
                    .ok_or(ReferenceQ8Error::BadWeight)?,
            )
            .map_err(|_| ReferenceQ8Error::BadWeight)?;
            let start = block * FP8_E4M3_SCALE_TILE;
            for lane in 0..FP8_E4M3_SCALE_TILE {
                let value =
                    decode_e4m3fn(values[start + lane]).map_err(|_| ReferenceQ8Error::BadWeight)?;
                sum += value * scale * activation[start + lane];
            }
        }
        *output_value = sum;
    }
    Ok(())
}

pub fn fused_mlp_fp8_e4m3_scale_grid_f32_ref(
    gate_weight: &Fp8E4M3ScaleGridCompactTensor,
    up_weight: &Fp8E4M3ScaleGridCompactTensor,
    down_weight: &Fp8E4M3ScaleGridCompactTensor,
    input: &[f32],
    config: FusedMlpF32Config,
    output: &mut [f32],
) -> Result<(), ReferenceQ8Error> {
    config.validate().map_err(|_| ReferenceQ8Error::BadWeight)?;
    let intermediate = gate_weight.rows();
    if gate_weight.cols() != input.len()
        || up_weight.rows() != intermediate
        || up_weight.cols() != input.len()
        || down_weight.rows() != input.len()
        || down_weight.cols() != intermediate
    {
        return Err(ReferenceQ8Error::BadWeightShape);
    }
    if output.len() < input.len() {
        return Err(ReferenceQ8Error::BadOutputLength {
            expected: input.len(),
            actual: output.len(),
        });
    }
    let mut gate = vec![0.0; intermediate];
    let mut up = vec![0.0; intermediate];
    qmatvec_fp8_e4m3_scale_grid_f32_ref(gate_weight, input, &mut gate)?;
    qmatvec_fp8_e4m3_scale_grid_f32_ref(up_weight, input, &mut up)?;
    for (gate_value, up_value) in gate.iter_mut().zip(&up) {
        let gated = clamp_ref(*gate_value, config.gate_min, config.gate_max);
        let up = clamp_ref(*up_value, config.up_min, config.up_max);
        *gate_value = gated / (1.0 + (-gated).exp()) * up;
    }
    qmatvec_fp8_e4m3_scale_grid_f32_ref(down_weight, &gate, output)
}

fn clamp_ref(value: f32, min: Option<f32>, max: Option<f32>) -> f32 {
    let value = min.map_or(value, |bound| if value < bound { bound } else { value });
    max.map_or(value, |bound| if value > bound { bound } else { value })
}

fn qmatvec_ref<B: ReferenceDecode<Elem = f32>>(
    blocks: &[B],
    rows: usize,
    cols: usize,
    block_len: usize,
    activation: &[f32],
    output: &mut [f32],
) -> Result<(), ReferenceQ8Error> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(block_len) {
        return Err(ReferenceQ8Error::BadWeightShape);
    }
    if activation.len() != cols {
        return Err(ReferenceQ8Error::BadActivationLength {
            expected: cols,
            actual: activation.len(),
        });
    }
    if output.len() < rows {
        return Err(ReferenceQ8Error::BadOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }
    let blocks_per_row = cols / block_len;
    let expected = rows
        .checked_mul(blocks_per_row)
        .ok_or(ReferenceQ8Error::BadWeightShape)?;
    if blocks.len() != expected {
        return Err(ReferenceQ8Error::BadWeightBlockCount {
            expected,
            actual: blocks.len(),
        });
    }
    let mut decoded = vec![0.0_f32; block_len];
    for row in 0..rows {
        let mut sum = 0.0_f32;
        for block_index in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + block_index];
            block
                .decode_ref(&mut decoded)
                .map_err(|_| ReferenceQ8Error::BadWeight)?;
            let start = block_index * block_len;
            for lane in 0..block_len {
                sum += decoded[lane] * activation[start + lane];
            }
        }
        output[row] = sum;
    }
    Ok(())
}
