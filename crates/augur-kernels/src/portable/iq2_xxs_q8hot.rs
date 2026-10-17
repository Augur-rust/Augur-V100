//! Portable GGUF-compatible IQ2_XXS weight x Q8K activation kernel.

use augur_kernel_api::{
    Iq2XxsHotCpuBlock, Iq2XxsHotCpuTensor, KernelLaunchError, Q8KActivationHotCpuBlock,
    Q8KActivationHotCpuVector,
};

use crate::KernelError;

const GRID_CODES: [u16; 256] = [
    0, 2, 5, 8, 10, 17, 20, 32, 34, 40, 42, 65, 68, 80, 88, 97, 100, 128, 130, 138, 162, 257, 260,
    272, 277, 320, 388, 408, 512, 514, 546, 642, 1025, 1028, 1040, 1057, 1060, 1088, 1090, 1096,
    1120, 1153, 1156, 1168, 1188, 1280, 1282, 1288, 1312, 1350, 1385, 1408, 1425, 1545, 1552, 1600,
    1668, 1700, 2048, 2053, 2056, 2068, 2088, 2113, 2116, 2128, 2130, 2184, 2308, 2368, 2562, 2580,
    4097, 4100, 4112, 4129, 4160, 4192, 4228, 4240, 4245, 4352, 4360, 4384, 4432, 4442, 4480, 4644,
    4677, 5120, 5128, 5152, 5157, 5193, 5248, 5400, 5474, 5632, 5654, 6145, 6148, 6160, 6208, 6273,
    6400, 6405, 6560, 6737, 8192, 8194, 8202, 8260, 8289, 8320, 8322, 8489, 8520, 8704, 8706, 9217,
    9220, 9232, 9280, 9302, 9472, 9537, 9572, 9872, 10248, 10272, 10388, 10820, 16385, 16388,
    16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516, 16528, 16640, 16672,
    16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
    17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013,
    19088, 20480, 20488, 20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648,
    21650, 21770, 22017, 22100, 22528, 22545, 22553, 22628, 22848, 23048, 24580, 24592, 24640,
    24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874, 25988, 26690, 32768, 32770,
    32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856, 33888,
    34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928,
    37025, 37142, 37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093,
    41225, 41472, 42008, 43088, 43268,
];

pub fn qmatvec_iq2_xxs_q8hot_cpu_portable(
    weight: &Iq2XxsHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
) -> Result<(), KernelError> {
    validate(weight, activation, output)?;
    for row in 0..weight.rows() {
        output[row] = dot_unchecked(
            weight.row_blocks(row).ok_or(KernelError::BadWeight)?,
            activation.blocks(),
        );
    }
    Ok(())
}

pub fn launch_qmatvec_iq2_xxs_q8hot_cpu_portable(
    weight: &Iq2XxsHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    qmatvec_iq2_xxs_q8hot_cpu_portable(weight, activation, output)
        .map_err(KernelError::into_launch_error)
}

fn dot_unchecked(weight: &[Iq2XxsHotCpuBlock], activation: &[Q8KActivationHotCpuBlock]) -> f32 {
    let mut output = 0.0_f32;
    for (weight_block, activation_block) in weight.iter().zip(activation) {
        let mut block_sum = 0_i32;
        for group in 0..8 {
            let words = weight_block
                .group_words(group)
                .expect("validated fixed IQ2_XXS group");
            let [grid0, grid1] = words[0].to_le_bytes();
            let [grid2, grid3] = words[1].to_le_bytes();
            let grids = [grid0, grid1, grid2, grid3];
            let sign_scale = u32::from_le_bytes([
                words[2].to_le_bytes()[0],
                words[2].to_le_bytes()[1],
                words[3].to_le_bytes()[0],
                words[3].to_le_bytes()[1],
            ]);
            let group_scale = (2 * (sign_scale >> 28) + 1) as i32;
            let mut group_sum = 0_i32;
            for sub_group in 0..4 {
                let sign_index = ((sign_scale >> (7 * sub_group)) & 0x7f) as u8;
                let sign_mask = sign_index | ((((sign_index & 0x7f).count_ones() & 1) as u8) << 7);
                let code = GRID_CODES[grids[sub_group] as usize];
                for lane in 0..8 {
                    let level = (code >> (2 * lane)) & 0x03;
                    let magnitude = match level {
                        0 => 8,
                        1 => 25,
                        2 => 43,
                        _ => 0,
                    };
                    let signed = if sign_mask & (1 << lane) == 0 {
                        magnitude
                    } else {
                        -magnitude
                    };
                    group_sum +=
                        signed * (activation_block.qs[group * 32 + sub_group * 8 + lane] as i32);
                }
            }
            block_sum += group_sum * group_scale;
        }
        output += weight_block.d.to_f32() * activation_block.d.to_f32() * block_sum as f32 * 0.125;
    }
    output
}

fn validate(
    weight: &Iq2XxsHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &[f32],
) -> Result<(), KernelError> {
    if activation.len_elements() != weight.cols() {
        return Err(KernelError::BadActivationLength {
            expected: weight.cols(),
            actual: activation.len_elements(),
        });
    }
    if output.len() < weight.rows() {
        return Err(KernelError::BadOutputLength {
            expected: weight.rows(),
            actual: output.len(),
        });
    }
    if weight.blocks_per_row() != activation.block_count() {
        return Err(KernelError::BadActivationBlockCount {
            expected: weight.blocks_per_row(),
            actual: activation.block_count(),
        });
    }
    Ok(())
}
