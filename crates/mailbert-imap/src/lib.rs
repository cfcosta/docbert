//! mailbert-imap speaks IMAP for mailbert.
//!
//! The crate reads mail and never writes to the server. It sets no flag,
//! it moves no message, and it expunges nothing. See `docs/mailbert.md`
//! §3 for the design.

pub mod error;
pub mod fake;
pub mod sequence;
pub mod token;
pub mod wire;

pub use error::{Error, Result};
pub use sequence::UidSet;
pub use token::{Token, encode, lex};
pub use wire::{Tags, read_answer, read_line, write_line};
