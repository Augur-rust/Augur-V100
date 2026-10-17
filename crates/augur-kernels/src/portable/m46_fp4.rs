//! M46 CPU-portable FP4 weight kernels.

use augur_kernel_api::{
    Fp8E4M3HotCpuTensor, Fp8E4M3ScaleGridHotCpuTensor, Fp8FusedMlpF32Workspace, FusedMlpF32Config,
    KernelLaunchError, Mxfp4HotCpuTensor, Nvfp4HotCpuTensor, FP8_E4M3_SCALE_TILE,
};

use crate::KernelError;

#[inline]
fn e2m1(nibble: u8) -> f32 {
    const TABLE: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    TABLE[(nibble & 0x0f) as usize]
}

macro_rules! qmatvec {
    ($name:ident, $tensor:ty, $block_len:expr) => {
        pub fn $name(
            weight: &$tensor,
            activation: &[f32],
            output: &mut [f32],
        ) -> Result<(), KernelError> {
            if activation.len() != weight.cols() {
                return Err(KernelError::BadActivationLength {
                    expected: weight.cols(),
                    actual: activation.len(),
                });
            }
            if output.len() < weight.rows() {
                return Err(KernelError::BadOutputLength {
                    expected: weight.rows(),
                    actual: output.len(),
                });
            }
            for row in 0..weight.rows() {
                let blocks = weight.row_blocks(row).ok_or(KernelError::BadWeight)?;
                let mut sum = 0.0_f32;
                for (block_index, block) in blocks.iter().enumerate() {
                    let start = block_index * $block_len;
                    for lane in 0..$block_len {
                        let byte = block.qs[lane / 2];
                        let nibble = if lane % 2 == 0 { byte } else { byte >> 4 };
                        sum += e2m1(nibble) * block.scale * activation[start + lane];
                    }
                }
                output[row] = sum;
            }
            Ok(())
        }
    };
}

qmatvec!(qmatvec_mxfp4_f32_cpu_portable, Mxfp4HotCpuTensor, 32);
qmatvec!(qmatvec_nvfp4_f32_cpu_portable, Nvfp4HotCpuTensor, 16);

pub fn launch_qmatvec_mxfp4_f32_cpu_portable(
    weight: &Mxfp4HotCpuTensor,
    activation: &[f32],
    output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    qmatvec_mxfp4_f32_cpu_portable(weight, activation, output)
        .map_err(KernelError::into_launch_error)
}

#[inline]
fn e4m3(value: u8) -> f32 {
    debug_assert!((value & 0x7f) != 0x7f);
    E4M3_TABLE[value as usize]
}

const E4M3_EXPONENT_SCALE: [f32; 16] = [
    0.015625, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
    128.0, 256.0,
];

const fn e4m3_const(value: u8) -> f32 {
    let sign = if value & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = ((value >> 3) & 0x0f) as usize;
    let mantissa = (value & 0x07) as f32;
    let magnitude = if exponent == 0 {
        E4M3_EXPONENT_SCALE[0] * (mantissa / 8.0)
    } else {
        E4M3_EXPONENT_SCALE[exponent] * (1.0 + mantissa / 8.0)
    };
    sign * magnitude
}

const fn e4m3_table() -> [f32; 256] {
    let mut table = [0.0; 256];
    let mut index = 0;
    while index < table.len() {
        table[index] = e4m3_const(index as u8);
        index += 1;
    }
    table
}

const E4M3_TABLE: [f32; 256] = e4m3_table();

pub fn qmatvec_fp8_e4m3_f32_cpu_portable(
    weight: &Fp8E4M3HotCpuTensor,
    activation: &[f32],
    output: &mut [f32],
) -> Result<(), KernelError> {
    if activation.len() != weight.cols() {
        return Err(KernelError::BadActivationLength {
            expected: weight.cols(),
            actual: activation.len(),
        });
    }
    if output.len() < weight.rows() {
        return Err(KernelError::BadOutputLength {
            expected: weight.rows(),
            actual: output.len(),
        });
    }
    for (row, output_value) in output.iter_mut().enumerate().take(weight.rows()) {
        let mut sum = 0.0_f32;
        for (block_index, block) in weight
            .row_blocks(row)
            .ok_or(KernelError::BadWeight)?
            .iter()
            .enumerate()
        {
            let start = block_index * 128;
            for lane in 0..128 {
                sum += e4m3(block.qs[lane]) * block.scale * activation[start + lane];
            }
        }
        *output_value = sum;
    }
    Ok(())
}

pub fn qmatvec_fp8_e4m3_scale_grid_f32_cpu_portable(
    weight: &Fp8E4M3ScaleGridHotCpuTensor,
    activation: &[f32],
    output: &mut [f32],
) -> Result<(), KernelError> {
    if activation.len() != weight.cols() {
        return Err(KernelError::BadActivationLength {
            expected: weight.cols(),
            actual: activation.len(),
        });
    }
    if output.len() < weight.rows() {
        return Err(KernelError::BadOutputLength {
            expected: weight.rows(),
            actual: output.len(),
        });
    }
    for row in 0..weight.rows() {
        let values = weight.row_values(row).ok_or(KernelError::BadWeight)?;
        let mut sum = 0.0_f32;
        for block in 0..weight.scale_cols() {
            let scale = weight.scale(row, block).ok_or(KernelError::BadWeight)?;
            let start = block * FP8_E4M3_SCALE_TILE;
            for lane in 0..FP8_E4M3_SCALE_TILE {
                sum += e4m3(values[start + lane]) * scale * activation[start + lane];
            }
        }
        output[row] = sum;
    }
    Ok(())
}

pub fn launch_qmatvec_fp8_e4m3_scale_grid_f32_cpu_portable(
    weight: &Fp8E4M3ScaleGridHotCpuTensor,
    activation: &[f32],
    output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    qmatvec_fp8_e4m3_scale_grid_f32_cpu_portable(weight, activation, output)
        .map_err(KernelError::into_launch_error)
}

pub fn launch_fp8_e4m3_scale_grid_fused_mlp_f32_cpu_portable(
    gate_weight: &Fp8E4M3ScaleGridHotCpuTensor,
    up_weight: &Fp8E4M3ScaleGridHotCpuTensor,
    down_weight: &Fp8E4M3ScaleGridHotCpuTensor,
    input: &[f32],
    config: FusedMlpF32Config,
    workspace: &mut Fp8FusedMlpF32Workspace,
    output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    config.validate()?;
    let hidden = input.len();
    let intermediate = gate_weight.rows();
    if hidden == 0
        || gate_weight.cols() != hidden
        || up_weight.rows() != intermediate
        || up_weight.cols() != hidden
        || down_weight.rows() != hidden
        || down_weight.cols() != intermediate
    {
        return Err(KernelLaunchError::BadWeightShape);
    }
    if workspace.intermediate() != intermediate {
        return Err(KernelLaunchError::BadWorkspace);
    }
    if output.len() < hidden {
        return Err(KernelLaunchError::BadOutputLength {
            expected: hidden,
            actual: output.len(),
        });
    }

    let (gate, up) = workspace.parts_mut();
    qmatvec_fp8_e4m3_scale_grid_f32_cpu_portable(gate_weight, input, gate)
        .map_err(KernelError::into_launch_error)?;
    qmatvec_fp8_e4m3_scale_grid_f32_cpu_portable(up_weight, input, up)
        .map_err(KernelError::into_launch_error)?;
    for (gate_value, up_value) in gate.iter_mut().zip(up.iter()) {
        let gated = clamp(*gate_value, config.gate_min, config.gate_max);
        let up = clamp(*up_value, config.up_min, config.up_max);
        *gate_value = gated / (1.0 + (-gated).exp()) * up;
    }
    qmatvec_fp8_e4m3_scale_grid_f32_cpu_portable(down_weight, gate, output)
        .map_err(KernelError::into_launch_error)
}

#[inline]
fn clamp(value: f32, min: Option<f32>, max: Option<f32>) -> f32 {
    let value = min.map_or(value, |bound| if value < bound { bound } else { value });
    max.map_or(value, |bound| if value > bound { bound } else { value })
}
