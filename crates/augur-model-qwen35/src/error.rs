use std::fmt;

#[derive(Debug)]
pub enum Qwen35Error {
    Json(String),
    MissingField(String),
    InvalidField { field: String, detail: String },
    Io(String),
    Gguf(String),
    SizeOverflow,
}

impl Qwen35Error {
    pub(crate) fn invalid(field: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::InvalidField {
            field: field.into(),
            detail: detail.into(),
        }
    }
}

impl fmt::Display for Qwen35Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(detail) => write!(formatter, "invalid JSON: {detail}"),
            Self::MissingField(field) => write!(formatter, "missing required field: {field}"),
            Self::InvalidField { field, detail } => write!(formatter, "{field}: {detail}"),
            Self::Io(detail) => write!(formatter, "I/O error: {detail}"),
            Self::Gguf(detail) => write!(formatter, "GGUF error: {detail}"),
            Self::SizeOverflow => formatter.write_str("size arithmetic overflow"),
        }
    }
}

impl std::error::Error for Qwen35Error {}

impl From<std::io::Error> for Qwen35Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}
