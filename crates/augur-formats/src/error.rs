#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FormatError {
    BadLength,
    BadBlock,
    UnsupportedVersion,

    InputLengthMismatch { expected: usize, actual: usize },

    OutputTooSmall { expected: usize, actual: usize },

    NonFiniteInput { index: usize },

    NonFiniteValue,

    ScaleOutOfRange { index: usize, value: u8 },

    MinOutOfRange { index: usize, value: u8 },
}
