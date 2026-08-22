//! mailbert-core is the library behind mailbert, a local mail search
//! engine that fuses BM25 with ColBERT semantic retrieval.
//!
//! See `docs/mailbert.md` in the repository root for the design document.
//! This crate is in early implementation.

pub mod error;
pub mod message_id;

pub use error::{Error, Result};
pub use message_id::MessageId;
