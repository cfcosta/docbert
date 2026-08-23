//! mailbert-core is the library behind mailbert, a local mail search
//! engine that fuses BM25 with ColBERT semantic retrieval.
//!
//! See `docs/mailbert.md` in the repository root for the design document.
//! This crate is in early implementation.

pub mod address;
pub mod body;
pub mod compile;
pub mod config;
pub mod contacts;
pub mod date;
pub mod embed;
pub mod error;
pub mod index;
pub mod message;
pub mod message_id;
pub mod mime;
pub mod query;
pub mod rank;
pub mod store;
pub mod threading;

pub use address::Address;
pub use body::{Stripped, strip, strip_with_footers};
pub use compile::{Compiled, Vocabulary, compile, expand};
pub use config::{Account, Config, Credential};
pub use contacts::{Contact, Contacts, Seen};
pub use date::{Clock, DateRange, internal_date};
pub use error::{Error, Result};
pub use index::{Fields, FlagTerm, Hit, MailIndex, flag_term, flag_terms};
pub use message::{Location, Message};
pub use message_id::MessageId;
pub use mime::{Attachment, MimeError, Parsed, Source};
pub use query::{Field, Flag, Query, QueryError, Value};
pub use rank::{Options, Row, Sort, Threads, decay, fuse, rank};
pub use store::{Store, normalize_tag};
pub use threading::{ThreadId, ThreadInput, Threading};
