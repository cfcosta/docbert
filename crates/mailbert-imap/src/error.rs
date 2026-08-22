//! Top-level error type for mailbert-imap.
//!
//! Each variant wraps a single source error with `#[from]`, so call sites
//! use `?` and do not use `map_err`.

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Core(#[from] mailbert_core::Error),

    #[error("the server sent a response that I cannot read: {0}")]
    Malformed(String),

    #[error("the server sent text that is not UTF-8: {0}")]
    NotText(#[from] std::string::FromUtf8Error),

    #[error("the server closed the connection")]
    Closed,

    #[error("the server refused the connection: {0}")]
    Refused(String),

    #[error("the server said no: {0}")]
    No(String),

    #[error("the server did not understand: {0}")]
    Bad(String),

    #[error("TLS error: {0}")]
    Tls(#[from] rustls::Error),

    #[error("`{0}` is not a name that TLS can check")]
    BadName(#[from] rustls_pki_types::InvalidDnsNameError),
}
