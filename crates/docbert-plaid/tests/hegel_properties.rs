//! Hegel-based property tests for the PLAID primitives.
//!
//! Complements `tests/properties.rs` (which uses hand-rolled `StdRng`
//! seeds as smoke tests). Every test here uses hegel's shrinking to
//! surface minimal counter-examples when an invariant breaks.
//!
//! The tests are grouped by module for readability. Reusable
//! generators live at the top so each test body stays focused on the
//! invariant it encodes.
//!
//! Runner defaults are conservative per test: cheap scalar properties
//! run ~200 cases; whole-pipeline tests (`build_index`, search,
//! save/load) run 20–50 because each case is orders of magnitude more
//! expensive. Hegel auto-derandomises in CI so runs are reproducible.

#![allow(dead_code)]

use docbert_plaid::index::{DocumentTokens, IndexParams};
use hegel::{TestCase, generators as gs};

// ---------------------------------------------------------------------------
// Reusable composite generators
// ---------------------------------------------------------------------------

/// Draw an `n × dim` row-major f32 buffer with each row L2-normalised
/// to unit length. Mirrors the ColBERT invariant that token embeddings
/// live on the unit sphere, which is what makes dot-product MaxSim
/// meaningful.
#[hegel::composite]
fn unit_rows(tc: TestCase, dim: usize, n: usize) -> Vec<f32> {
    let total = n * dim;
    let mut v: Vec<f32> = tc.draw(
        gs::vecs(
            gs::floats::<f32>()
                .min_value(-1.0)
                .max_value(1.0)
                .allow_nan(false)
                .allow_infinity(false),
        )
        .min_size(total)
        .max_size(total),
    );
    for row in v.chunks_exact_mut(dim) {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in row {
            *x /= norm;
        }
    }
    v
}

/// Draw a random raw (non-normalised) f32 vector of length `n`,
/// excluding NaN/infinity. Useful for algebraic tests where we only
/// care about finite behaviour.
#[hegel::composite]
fn finite_floats(tc: TestCase, n: usize) -> Vec<f32> {
    tc.draw(
        gs::vecs(
            gs::floats::<f32>()
                .min_value(-1e3)
                .max_value(1e3)
                .allow_nan(false)
                .allow_infinity(false),
        )
        .min_size(n)
        .max_size(n),
    )
}

/// Draw a small corpus of documents whose tokens are unit-norm.
///
/// Ensures `dim > 0`, at least `min_docs` documents, and that the
/// total token count across the corpus is at least `min_total_tokens`
/// (padding the first doc if necessary). Valid-by-construction —
/// `build_index` with matching `k_centroids ≤ min_total_tokens` will
/// accept the output without panicking.
#[hegel::composite]
fn corpus(
    tc: TestCase,
    dim: usize,
    min_docs: usize,
    max_docs: usize,
    max_tokens_per_doc: usize,
    min_total_tokens: usize,
) -> Vec<DocumentTokens> {
    let n_docs: usize = tc.draw(
        gs::integers::<usize>()
            .min_value(min_docs)
            .max_value(max_docs.max(min_docs)),
    );
    let mut docs: Vec<DocumentTokens> = Vec::with_capacity(n_docs);
    let mut total_tokens = 0usize;
    for i in 0..n_docs {
        let n_tokens: usize = tc.draw(
            gs::integers::<usize>()
                .min_value(0)
                .max_value(max_tokens_per_doc),
        );
        let tokens = if n_tokens == 0 {
            Vec::new()
        } else {
            tc.draw(unit_rows(dim, n_tokens))
        };
        total_tokens += n_tokens;
        docs.push(DocumentTokens {
            doc_id: (i as u64) + 1,
            tokens,
            n_tokens,
        });
    }
    // Guarantee enough tokens for k-means by padding the first doc if
    // the draw came up short. Still valid-by-construction — we never
    // reject.
    if total_tokens < min_total_tokens {
        let missing = min_total_tokens - total_tokens;
        let extra = tc.draw(unit_rows(dim, missing));
        let first = &mut docs[0];
        first.tokens.extend_from_slice(&extra);
        first.n_tokens += missing;
    }
    docs
}

/// Draw index parameters that are always valid for the `corpus`
/// generator below. `k_centroids ≤ min_total_tokens` so `build_index`
/// can't panic on "need at least k tokens".
#[hegel::composite]
fn index_params(
    tc: TestCase,
    dim: usize,
    max_k: usize,
    max_tokens: usize,
) -> IndexParams {
    let nbits: u32 = tc.draw(gs::sampled_from(vec![1u32, 2, 4, 8]));
    let k_centroids: usize = tc.draw(
        gs::integers::<usize>()
            .min_value(1)
            .max_value(max_k.min(max_tokens).max(1)),
    );
    let max_kmeans_iters: usize =
        tc.draw(gs::integers::<usize>().min_value(1).max_value(20));
    IndexParams {
        dim,
        nbits,
        k_centroids,
        max_kmeans_iters,
    }
}

/// Pick a dim from the set commonly used by ColBERT-family models,
/// but restricted to values where `dim * nbits` is a multiple of 8
/// for every supported nbits — otherwise `pack_codes` would drop
/// trailing bits from partially-filled bytes, and every per-token
/// invariant downstream would need a bytes-equal-codes allowance.
#[hegel::composite]
fn codec_dim(tc: TestCase) -> usize {
    tc.draw(gs::sampled_from(vec![2usize, 4, 8, 16]))
}

// ---------------------------------------------------------------------------
// Smoke test: exercises every generator so compilation errors surface
// before the real property tests are layered on top.
// ---------------------------------------------------------------------------

#[hegel::test(test_cases = 20)]
fn generators_produce_valid_shapes(tc: TestCase) {
    let dim = tc.draw(codec_dim());
    let n = tc.draw(gs::integers::<usize>().min_value(1).max_value(16));

    let rows = tc.draw(unit_rows(dim, n));
    assert_eq!(rows.len(), n * dim);

    let docs = tc.draw(corpus(dim, 1, 4, 6, 4));
    assert!(!docs.is_empty());
    let total: usize = docs.iter().map(|d| d.n_tokens).sum();
    assert!(total >= 4);
    for d in &docs {
        assert_eq!(d.tokens.len(), d.n_tokens * dim);
    }

    let params = tc.draw(index_params(dim, 4, total));
    assert!(params.k_centroids <= total);
    assert!(matches!(params.nbits, 1 | 2 | 4 | 8));
}

// ---------------------------------------------------------------------------
// distance.rs
// ---------------------------------------------------------------------------

/// Algebraic: `dot(a, b) == dot(b, a)`. Summation order is the only
/// source of asymmetry and we compute both sums in the same left-to-right
/// order, so equality holds bit-for-bit rather than within an ε.
#[hegel::test(test_cases = 200)]
fn prop_dot_commutative(tc: TestCase) {
    use docbert_plaid::distance::dot;
    let n = tc.draw(gs::integers::<usize>().min_value(0).max_value(64));
    let a = tc.draw(finite_floats(n));
    let b = tc.draw(finite_floats(n));
    assert_eq!(dot(&a, &b), dot(&b, &a));
}

/// Algebraic: `squared_l2(a, b) == squared_l2(b, a)`. Each squared
/// difference is symmetric in its operands, and the summation iterates
/// in the same left-to-right order in both directions, so equality is
/// bit-exact.
#[hegel::test(test_cases = 200)]
fn prop_squared_l2_symmetric(tc: TestCase) {
    use docbert_plaid::distance::squared_l2;
    let n = tc.draw(gs::integers::<usize>().min_value(0).max_value(64));
    let a = tc.draw(finite_floats(n));
    let b = tc.draw(finite_floats(n));
    assert_eq!(squared_l2(&a, &b), squared_l2(&b, &a));
}

/// Algebraic: `squared_l2(a, b) >= 0`. A sum of squares over finite
/// inputs cannot be negative. Any regression that swaps the subtraction
/// order for a signed difference would surface here.
#[hegel::test(test_cases = 200)]
fn prop_squared_l2_non_negative(tc: TestCase) {
    use docbert_plaid::distance::squared_l2;
    let n = tc.draw(gs::integers::<usize>().min_value(0).max_value(64));
    let a = tc.draw(finite_floats(n));
    let b = tc.draw(finite_floats(n));
    assert!(squared_l2(&a, &b) >= 0.0);
}
