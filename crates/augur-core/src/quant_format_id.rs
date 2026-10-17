#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum QuantFormatId {
    Q4K,
    Q5K,
    Q6K,
    #[allow(non_camel_case_types)]
    Q8_0,
    Q2K,
    #[allow(non_camel_case_types)]
    IQ2_XXS,
    Q4X,
    DualScale,
    LogQ8,
    Q8K,
    MXFP4,
    NVFP4,
    FP8E4M3Block,
    /// FP8 E4M3 values with one E8M0 scale shared by each 128x128 tile.
    FP8E4M3ScaleGrid,
}
