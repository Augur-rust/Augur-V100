use crate::dtype::DType;
use crate::quant_format_id::FormatId;
use crate::physical_layout::PhysicalLayoutId;

pub trait QuantFormat: Copy + Send + Sync + 'static {
    const ID: FormatId;
    const BITS: u32;
    const BLOCK_SIZE: usize;
    const WEIGHT_DTYPE: DType;
    const COMPACT_LAYOUT: PhysicalLayoutId;

    fn block_bytes() -> usize;
}
