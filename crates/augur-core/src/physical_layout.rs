#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FormatVersion(pub u16);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LayoutId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PhysicalLayoutId(pub u32);

impl PhysicalLayoutId {
    pub const HOST_CONTIGUOUS_F32: Self = Self(1);

    /// Contiguous f32 values in CUDA device memory.
    ///
    /// This is distinct from `HOST_CONTIGUOUS_F32`; identical element
    /// encoding does not imply the same address space.
    pub const DEVICE_CONTIGUOUS_F32_CUDA_V1: Self = Self(2);

    pub const Q4K_COMPACT_V1: Self = Self(100);

    /// Reserved for the future AVX2-specific Q4K Hot layout.
    ///
    /// No current portable kernel may claim this layout.
    pub const Q4K_HOT_AVX2_V1: Self = Self(101);

    /// Current Q4K CPU portable Hot layout.
    ///
    /// The layout predecodes the compact 6-bit scale/min metadata into
    /// eight-byte scale/min arrays while preserving the compact q4 payload.
    pub const Q4K_HOT_CPU_PORTABLE_V1: Self = Self(102);

    /// CUDA Q4K decode layout.
    ///
    /// This ID describes the explicit byte protocol defined by
    /// `Q4KHotCudaStagingTensorV1`. It does not describe the Rust memory
    /// layout of `Q4KHotCpuBlock`.
    pub const Q4K_HOT_CUDA_V1: Self = Self(103);

    /// llama.cpp-compatible 256-value Q5_K compact super-block.
    pub const Q5K_COMPACT_V1: Self = Self(104);
    /// SM70 CUDA Q5_K hot block with unpacked scale/min metadata.
    pub const Q5K_HOT_CUDA_V1: Self = Self(105);
    /// llama.cpp-compatible 256-value Q6_K compact super-block.
    pub const Q6K_COMPACT_V1: Self = Self(106);
    /// SM70 CUDA Q6_K hot block with canonical alignment padding.
    pub const Q6K_HOT_CUDA_V1: Self = Self(107);
    /// Direct little-endian BF16 matrix bytes on CUDA device memory.
    pub const DEVICE_CONTIGUOUS_BF16_CUDA_V1: Self = Self(108);
    pub const DEVICE_CONTIGUOUS_F8E4M3_CUDA_V1: Self = Self(109);

    /// SM70 expert Q4_K Hot bytes interleaved across groups of 32 rows.
    ///
    /// This is a distinct physical protocol from `Q4K_HOT_CUDA_V1`; equal
    /// payload extents do not make row-major and row32 views interchangeable.
    pub const Q4K_EXPERT_ROW32_CUDA_V2: Self = Self(110);

    /// SM70 expert Q5_K Hot bytes interleaved across groups of 32 rows.
    ///
    /// This is a distinct physical protocol from `Q5K_HOT_CUDA_V1`; equal
    /// payload extents do not make row-major and row32 views interchangeable.
    pub const Q5K_EXPERT_ROW32_CUDA_V2: Self = Self(111);

    /// SM70 FP8 E4M3 matrix bytes interleaved by column inside 32-row tiles.
    ///
    /// This layout preserves every logical FP8 byte while making adjacent
    /// logical rows contiguous for a warp at a fixed column.
    pub const DEVICE_ROW32_F8E4M3_CUDA_V1: Self = Self(112);

    /// Row-major GGML Q8_0 blocks resident in CUDA device memory.
    ///
    /// Each logical row contains 32-value Q8_0 blocks (two-byte FP16 scale
    /// followed by 32 signed int8 codes).  This is distinct from the CPU
    /// compact/hot layouts even though the payload bytes are unchanged.
    pub const DEVICE_CONTIGUOUS_Q8_0_CUDA_V1: Self = Self(113);

    /// Row-major GGML NVFP4 blocks resident in SM70 CUDA device memory.
    ///
    /// Each 64-value block contains four UE4M3 scale bytes followed by 32
    /// packed E2M1 bytes.  The tensor-level f32 scale is supplied separately
    /// by the matvec ABI; it is not part of this byte layout.
    pub const DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1: Self = Self(114);

    /// llama.cpp-compatible Q8_1 activation blocks in CUDA device memory.
    ///
    /// Each 32-value block contains an FP16 scale, an FP16 sum, and 32 signed
    /// int8 codes. The sum is retained for wire compatibility even though the
    /// Q8_0 and NVFP4 SM70 dot products consume only the scale and codes.
    pub const DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1: Self = Self(115);

    /// Qwen output-head Q8 codes followed by FP16 block scales, same byte count.
    /// Never compatible with interleaved GGML Q8_0 views despite equal extent.
    pub const DEVICE_QWEN35_Q8_CODE_SCALE_PLANES_V1: Self = Self(116);
    /// Input-only K5120 row-local codes followed by that row's FP16 scales.
    pub const DEVICE_QWEN35_Q8_ROW_CODE_SCALE_PLANES_V1: Self = Self(117);

    /// SM70 QPN NVFP4: code plane then scale plane, quadpair row/group order.
    /// Same extent as GGML36 but never compatible with its weight views.
    /// The per-tensor multiplier remains separate; no requantization.
    pub const DEVICE_QWEN35_NVFP4_QPN_PLANES_V1: Self = Self(118);

    pub const Q4X_COMPACT_V1: Self = Self(200);
    pub const Q4X_HOT_AVX2_V1: Self = Self(201);

    /// Reference vector layout for Q8K activation metadata.
    ///
    /// This identifies the portable/reference Q8K activation vector layout
    /// used for metadata and tests. It is not an AVX2/CUDA/HIP hot layout.
    pub const Q8K_ACTIVATION_REF_VECTOR_V1: Self = Self(300);

    /// Q8K CPU portable hot activation layout (M20).
    pub const Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1: Self = Self(301);

    /// DualScale compact reference layout (M13).
    pub const DUAL_SCALE_COMPACT_V1: Self = Self(0x3000_0001);

    /// DualScale CPU portable hot layout (M15).
    pub const DUAL_SCALE_HOT_CPU_PORTABLE_V1: Self = Self(0x3000_0101);

    /// Fully resident row-major f32 router and expert matrices (M36).
    pub const RESIDENT_MOE_F32_HOT_CPU_PORTABLE_V1: Self = Self(0x4000_0101);

    /// llama.cpp-compatible 32-value Q8_0 compact block.
    pub const Q8_0_COMPACT_V1: Self = Self(0x5000_0001);

    /// CPU portable Q8_0 weight layout consumed with Q8K activations.
    pub const Q8_0_HOT_CPU_PORTABLE_V1: Self = Self(0x5000_0101);

    /// llama.cpp-compatible 256-value Q2_K compact super-block.
    pub const Q2K_COMPACT_V1: Self = Self(0x5000_1001);

    /// CPU portable Q2_K layout with predecoded scale/min nibbles.
    pub const Q2K_HOT_CPU_PORTABLE_V1: Self = Self(0x5000_1101);

    /// GGUF-compatible 256-value IQ2_XXS compact block.
    pub const IQ2_XXS_COMPACT_V1: Self = Self(0x5000_2001);

    /// CPU portable IQ2_XXS layout consumed with Q8K activations.
    pub const IQ2_XXS_HOT_CPU_PORTABLE_V1: Self = Self(0x5000_2101);

    /// OCP MXFP4 E2M1 values with one E8M0 scale per 32 values.
    pub const MXFP4_COMPACT_V1: Self = Self(0x6000_0001);
    pub const MXFP4_HOT_CPU_PORTABLE_V1: Self = Self(0x6000_0101);

    /// NVIDIA NVFP4 E2M1 values with one E4M3 scale per 16 values.
    pub const NVFP4_COMPACT_V1: Self = Self(0x6000_1001);
    pub const NVFP4_HOT_CPU_PORTABLE_V1: Self = Self(0x6000_1101);

    /// CUTLASS SM120 NVFP4 values plus tiled E4M3 scale factors.
    ///
    /// This layout is accepted only by the finite M52 Blackwell AOT shapes.
    pub const NVFP4_HOT_CUDA_SM120_CUTLASS_V1: Self = Self(0x6000_1201);

    /// FP8 E4M3 values with one E8M0 scale for a logical 128-value row block.
    pub const FP8_E4M3_BLOCK_COMPACT_V1: Self = Self(0x6000_2001);
    pub const FP8_E4M3_BLOCK_HOT_CPU_PORTABLE_V1: Self = Self(0x6000_2101);

    /// Row-major E4M3 values plus a two-dimensional E8M0 128x128 scale grid.
    pub const FP8_E4M3_SCALE_GRID_COMPACT_V1: Self = Self(0x6000_2201);

    /// CPU portable row-major E4M3 values plus decoded f32 128x128 scales.
    pub const FP8_E4M3_SCALE_GRID_HOT_CPU_PORTABLE_V1: Self = Self(0x6000_2301);
}

#[cfg(test)]
mod tests {
    use super::PhysicalLayoutId;

    #[test]
    fn expert_row32_layout_ids_follow_the_existing_cuda_range() {
        assert_eq!(PhysicalLayoutId::DEVICE_CONTIGUOUS_F8E4M3_CUDA_V1.0, 109);
        assert_eq!(PhysicalLayoutId::Q4K_EXPERT_ROW32_CUDA_V2.0, 110);
        assert_eq!(PhysicalLayoutId::Q5K_EXPERT_ROW32_CUDA_V2.0, 111);
        assert_eq!(PhysicalLayoutId::DEVICE_ROW32_F8E4M3_CUDA_V1.0, 112);
        assert_eq!(PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1.0, 113);
        assert_eq!(
            PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1.0,
            114
        );
        assert_eq!(
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1.0,
            115
        );
        assert_ne!(
            PhysicalLayoutId::Q4K_EXPERT_ROW32_CUDA_V2,
            PhysicalLayoutId::Q4K_HOT_CUDA_V1,
        );
        assert_ne!(
            PhysicalLayoutId::Q5K_EXPERT_ROW32_CUDA_V2,
            PhysicalLayoutId::Q5K_HOT_CUDA_V1,
        );
    }

    #[test]
    fn qwen_qpn_cannot_alias_wire_or_q8_plane_layouts() {
        let qpn=PhysicalLayoutId::DEVICE_QWEN35_NVFP4_QPN_PLANES_V1;
        assert_eq!(qpn.0,118);
        for other in [
            PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,
            PhysicalLayoutId::DEVICE_QWEN35_Q8_CODE_SCALE_PLANES_V1,
            PhysicalLayoutId::DEVICE_QWEN35_Q8_ROW_CODE_SCALE_PLANES_V1,
        ] { assert_ne!(qpn,other); }
    }
}
