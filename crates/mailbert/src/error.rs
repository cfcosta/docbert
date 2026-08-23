//! The errors that the tool shows.
//!
//! Each variant holds one source error, so the code uses `?` and never
//! `map_err`. The query error keeps its own report, because §7.2 asks
//! for a message that points at the word that is wrong.

use std::path::PathBuf;

use miette::Diagnostic;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error, Diagnostic)]
pub enum Error {
    #[error(transparent)]
    #[diagnostic(transparent)]
    Query(#[from] mailbert_core::query::QueryError),

    #[error(transparent)]
    Core(#[from] mailbert_core::Error),

    #[error("these bytes are not a message: {0}")]
    #[diagnostic(help("The store may hold a truncated download."))]
    Mime(#[from] mailbert_core::MimeError),

    #[error(transparent)]
    Imap(#[from] mailbert_imap::Error),

    #[error(transparent)]
    Model(#[from] docbert_core::Error),

    #[error(transparent)]
    Plaid(#[from] docbert_plaid::PlaidError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("cannot write the JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("I cannot find the directory that holds the data")]
    #[diagnostic(help("Set MAILBERT_DATA_DIR, or give --data-dir."))]
    NoDataDir,

    #[error("I cannot find the directory that holds the configuration")]
    #[diagnostic(help("Set MAILBERT_CONFIG, or give --config."))]
    NoConfigDir,

    #[error("there is no configuration file at `{0}`")]
    #[diagnostic(help(
        "Write the accounts into that file. See §1.2 of docs/mailbert.md."
    ))]
    NoConfig(PathBuf),

    #[error("`tag` needs a change, such as `+todo` or `-done`")]
    NoEdits,

    #[error("`tag` needs the identity of one message or more")]
    NoMessages,

    #[error("`{0}` comes after an identity")]
    #[diagnostic(help("Put every change before the identities."))]
    LateEdit(String),

    #[error("`{0}` is not ready in this build")]
    #[diagnostic(help(
        "The task list in docs/mailbert-tasks.md says when it lands."
    ))]
    NotYet(&'static str),

    #[error("the identity `{prefix}` names more than one message")]
    #[diagnostic(help("Give more characters. These match: {}", .ids.join(", ")))]
    AmbiguousMessage { prefix: String, ids: Vec<String> },

    #[error("I cannot run `gpg`: {0}")]
    #[diagnostic(help(
        "Install gpg. `mailbert get` gives the ciphertext without it."
    ))]
    NoGpg(String),

    #[error("no account is named `{0}`")]
    UnknownAccount(String),

    #[error("`{command}` stopped with the code {status}")]
    #[diagnostic(help("Run the command yourself, and read what it says."))]
    CommandFailed { command: String, status: i32 },

    #[error("the credential of account `{0}` is empty")]
    #[diagnostic(help(
        "The command, or the file, must write the password on its first line."
    ))]
    EmptySecret(String),
}

impl Error {
    /// The error for a tag that the store refuses.
    pub fn bad_tag(word: &str) -> Self {
        Self::Core(mailbert_core::Error::InvalidTag(word.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use mailbert_core::date::Clock;

    use super::*;

    /// §7.2 wants the report to point at the word that is wrong. The
    /// report only holds the span if the variant is transparent.
    #[test]
    fn a_bad_query_keeps_the_place_of_the_problem() {
        let broken = mailbert_core::query::parse("from:", Clock::new(0, 0))
            .expect_err("`from:` has no value");
        let error = Error::from(broken);

        assert!(error.source_code().is_some(), "the report lost the query");
        assert_eq!(error.labels().into_iter().flatten().count(), 1);
    }

    /// The `?` operator must carry an I/O error without a `map_err`.
    #[test]
    fn an_io_error_goes_through_the_question_mark() {
        fn inner() -> Result<()> {
            std::fs::read_to_string("/nonexistent/mailbert/probe")?;
            Ok(())
        }

        assert!(matches!(inner(), Err(Error::Io(_))));
    }
}
