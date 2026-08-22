//! Top-level error type for mailbert-core.
//!
//! Each variant wraps a single source error via `#[from]` so call sites
//! can use `?` instead of `map_err`.

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("config parse error: {0}")]
    ConfigParse(#[from] toml::de::Error),

    #[error("two accounts are named `{0}` — account names must be unique")]
    DuplicateAccount(String),

    #[error("account names cannot be empty")]
    EmptyAccountName,

    #[error(
        "account `{0}` has no credential — set password_command, \
         password_file, or password"
    )]
    MissingCredential(String),

    #[error("account `{account}` has an empty {field}")]
    EmptyField {
        account: String,
        field: &'static str,
    },

    #[error("account `{account}` has an invalid footer pattern `{pattern}`")]
    InvalidFooter {
        account: String,
        pattern: String,
        #[source]
        source: regex::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `#[from]` wiring is what lets the whole crate use `?` instead
    /// of sprinkling `map_err`. Assert it actually converts.
    #[test]
    fn io_error_converts_through_question_mark() {
        fn inner() -> Result<()> {
            std::fs::read_to_string("/nonexistent/mailbert/probe")?;
            Ok(())
        }

        assert!(matches!(inner(), Err(Error::Io(_))));
    }
}
