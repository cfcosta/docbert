//! PLAID-style multi-vector index for docbert.
//!
//! This crate implements the ColBERT late-interaction indexing pipeline
//! originally described in "PLAID: An Efficient Engine for Late Interaction
//! Retrieval" (Santhanam et al., 2022), modelled on the `fast-plaid`
//! reference implementation but written directly against plain Rust
//! primitives so that it drops into docbert's candle-based stack without a
//! libtorch dependency.
//!
//! The pipeline is built bottom-up over several modules:
//!
//! - [`distance`] — vector distance primitives.
//! - k-means clustering of token embeddings (next).
//! - residual codec (coarse centroid + quantized residual).
//! - inverted file (centroid → documents).
//! - search path (query tokens → candidate docs → MaxSim).
//!
//! At the moment the crate only exposes the lowest-level primitives; more
//! layers will be added as subsequent TDD cycles come in.

pub mod codec;
pub mod distance;
pub mod index;
pub mod kmeans;
