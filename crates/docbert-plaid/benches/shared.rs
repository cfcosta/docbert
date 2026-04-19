//! Shared fixture builders for the criterion benches.
//!
//! Each bench `#[path]`-includes this file so we have one source of
//! truth for "ColBERT-shaped synthetic data" without making it part of
//! the crate's public surface.

#![allow(dead_code)]

use docbert_plaid::index::DocumentTokens;
use rand::{RngExt, SeedableRng, rngs::StdRng};

/// Generate `n_points` L2-normalized random vectors of dimension `dim`,
/// returned as a flat row-major `n_points × dim` buffer.
///
/// Uses a deterministic seed so bench runs are repeatable: changes in
/// timing reflect code changes, not data drift. The vectors aren't
/// truly Gaussian (we use uniform-then-normalize) but for benchmarking
/// inner-loop throughput the distribution doesn't matter.
pub fn random_unit_vectors(seed: u64, n_points: usize, dim: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(n_points * dim);
    for _ in 0..n_points {
        let mut row = vec![0.0f32; dim];
        for v in &mut row {
            *v = rng.random::<f32>() * 2.0 - 1.0;
        }
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for v in &mut row {
            *v /= norm;
        }
        data.extend_from_slice(&row);
    }
    data
}

/// Build a synthetic corpus: `n_docs` documents, each with
/// `tokens_per_doc` unit-norm token vectors of dimension `dim`.
pub fn random_corpus(
    seed: u64,
    n_docs: usize,
    tokens_per_doc: usize,
    dim: usize,
) -> Vec<DocumentTokens> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_docs)
        .map(|i| {
            let inner_seed = rng.random::<u64>();
            let tokens = random_unit_vectors(inner_seed, tokens_per_doc, dim);
            DocumentTokens {
                doc_id: i as u64,
                tokens,
                n_tokens: tokens_per_doc,
            }
        })
        .collect()
}
