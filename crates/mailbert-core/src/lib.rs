//! mailbert-core is the library behind mailbert, a local mail search
//! engine that fuses BM25 with ColBERT semantic retrieval.
//!
//! See `docs/mailbert.md` in the repository root for the design document.
//! This crate is in early implementation.

pub mod address;
pub mod body;
pub mod config;
pub mod contacts;
pub mod date;
pub mod error;
pub mod message_id;
pub mod query;
pub mod threading;

pub use address::Address;
pub use body::{Stripped, strip, strip_with_footers};
pub use config::{Account, Config, Credential};
pub use contacts::{Contact, Contacts, Seen};
pub use date::{Clock, DateRange};
pub use error::{Error, Result};
pub use message_id::MessageId;
pub use query::{Field, Flag, Query, QueryError, Value};
pub use threading::{ThreadId, ThreadInput, Threading};
