#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I32,
    PackedU8,
}

impl DType {
    pub fn size_bytes(self) -> Option<usize> {
        match self {
            DType::F32 => Some(4),
            DType::F16 | DType::BF16 => Some(2),
            DType::I8 | DType::PackedU8 => Some(1),
            DType::I32 => Some(4),
        }
    }
}
