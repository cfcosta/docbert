//! Top-level error type for mailbert-core.
//!
//! Each variant wraps a single source error via `#[from]` so call sites
//! can use `?` instead of `map_err`.

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
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
