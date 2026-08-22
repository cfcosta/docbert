//! mailbert-imap speaks IMAP for mailbert.
//!
//! The crate reads mail and never writes to the server. It sets no flag,
//! it moves no message, and it expunges nothing. See `docs/mailbert.md`
//! §3 for the design.

pub mod connection;
pub mod error;
pub mod fake;
pub mod pool;
pub mod sequence;
pub mod stream;
pub mod token;
pub mod wire;

pub use connection::{Answer, Batch, Connection, Fetched, Folder, State, View};
pub use error::{Error, Result};
pub use pool::{Held, Pool, Server};
pub use sequence::{UidSet, batches};
pub use stream::Stream;
pub use token::{Token, encode, lex};
pub use wire::{Tags, read_answer, read_line, write_line};
