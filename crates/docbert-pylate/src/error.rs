use thiserror::Error;

/// Custom error enum for all possible errors in the ColBERT library.
///
/// This enum consolidates errors from various dependencies like `candle_core`,
/// `tokenizers`, and `serde_json`, as well as custom operational errors.
#[derive(Error, Debug)]
pub enum ColbertError {
    /// Error originating from the `candle_core` library.
    #[error("Candle Error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Error originating from the `tokenizers` library.
    #[error("Tokenizer Error: {0}")]
    Tokenizer(String),

    /// Error related to JSON serialization or deserialization.
    #[error("JSON Parsing Error: {0}")]
    Json(#[from] serde_json::Error),

    /// Custom operational errors, e.g., missing configuration values.
    #[error("Operation Error: {0}")]
    Operation(String),

    /// Error originating from the `hf-hub` library for Hugging Face Hub interactions.
    #[error("Hugging Face Hub Error: {0}")]
    Hub(#[from] hf_hub::api::sync::ApiError),

    /// I/O errors, typically from reading model files.
    #[error("I/O Error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<Box<dyn std::error::Error + Send + Sync>> for ColbertError {
    fn from(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        ColbertError::Tokenizer(err.to_string())
    }
}

impl From<ColbertError> for candle_core::Error {
    fn from(err: ColbertError) -> Self {
        match err {
            // If the error is already a Candle error, we can just return it directly.
            ColbertError::Candle(e) => e,
            // For any other type of `ColbertError`, we convert it to a string
            // and wrap it in the generic `candle_core::Error::Msg` variant.
            _ => candle_core::Error::Msg(err.to_string()),
        }
    }
}
