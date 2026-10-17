use augur_formats::FormatError;

/// Compatibility alias for repack callers.
///
/// Canonical owner: `augur_kernel_api::HotLayoutError`.
/// Do not use this alias in new cross-crate APIs.
pub type TensorError = augur_kernel_api::HotLayoutError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RepackError {
    Format(FormatError),
    Tensor(TensorError),
}

impl From<FormatError> for RepackError {
    fn from(value: FormatError) -> Self {
        Self::Format(value)
    }
}

impl From<TensorError> for RepackError {
    fn from(value: TensorError) -> Self {
        Self::Tensor(value)
    }
}
