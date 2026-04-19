//! Error type for the PLAID pipeline.
//!
//! A single `PlaidError` enum covers every failure mode the crate
//! surfaces to callers. Variants use `#[from]` where a clean `?`
//! conversion from a foreign error type makes sense, so the hot
//! paths stay free of `map_err` boilerplate.
//!
//! Variants are added incrementally as each module migrates off
//! `unwrap`/`expect`. This commit introduces the enum and the
//! `Tensor` variant for `candle_core::Error` propagation; later
//! commits add `InvalidCodec`, `InvalidIndex`, and `Io`.

use thiserror::Error;

/// Every failure the PLAID pipeline can surface.
#[derive(Debug, Error)]
pub enum PlaidError {
    /// Underlying tensor operation failed (allocation, matmul,
    /// transpose, argmin, etc.). Wraps `candle_core::Error` so
    /// callers can downcast when they care about the specific
    /// cause but still propagate uniformly via `?`.
    #[error("tensor operation failed: {0}")]
    Tensor(#[from] candle_core::Error),

    /// A [`ResidualCodec`] failed its shape/invariant checks. Raised
    /// from `encode_vector`, `decode_vector`, `batch_encode_tokens`,
    /// and `build_index`'s validate step. The wrapped `String`
    /// describes the specific constraint violated (dim == 0, nbits
    /// not in `{1,2,4,8}`, non-monotonic cutoffs, etc.).
    ///
    /// [`ResidualCodec`]: crate::codec::ResidualCodec
    #[error("invalid codec: {0}")]
    InvalidCodec(String),
}

/// Crate-level `Result` alias. Every public function that can fail
/// returns this.
pub type Result<T> = std::result::Result<T, PlaidError>;
