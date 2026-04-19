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

/// Algebraic: `‖a‖² − 2·dot(a,b) + ‖b‖² == squared_l2(a,b)` within a
/// small tolerance. This is the exact identity `assign_tensor_chunked`
/// relies on (`||c||² − 2·p·c` with the `||p||²` constant dropped for
/// argmin); the formula only sees unit-norm inputs in practice, so we
/// test that regime — it avoids the catastrophic cancellation that
/// would otherwise dominate when `a ≈ b` and both are large.
#[hegel::test(test_cases = 200)]
fn prop_squared_l2_expansion_identity(tc: TestCase) {
    use docbert_plaid::distance::{dot, squared_l2};
    let dim = tc.draw(codec_dim());
    let ab = tc.draw(unit_rows(dim, 2));
    let (a, b) = ab.split_at(dim);
    let lhs = dot(a, a) - 2.0 * dot(a, b) + dot(b, b);
    let rhs = squared_l2(a, b);
    // Unit-norm inputs keep every term ≤ 4, so 1e-5 absolute is a
    // generous tolerance against per-element f32 rounding.
    assert!(
        (lhs - rhs).abs() <= 1e-5,
        "expansion identity drifted: lhs={lhs} rhs={rhs}",
    );
}

// ---------------------------------------------------------------------------
// kmeans.rs
// ---------------------------------------------------------------------------

/// Differential: the tensor-matmul `assign_points` path must produce
/// the same assignment vector as calling the scalar `nearest_centroid`
/// per row. This is the flagship catch-all for E-step regressions —
/// any drift between the `||c||² − 2·p·c` shortcut and the direct L2
/// computation surfaces here as a mismatched assignment. Ties are
/// permitted to break either way on either side, so we compare the
/// equivalence class (same squared-distance to the returned centroid)
/// rather than exact index equality.
#[hegel::test(test_cases = 60)]
fn prop_assign_points_matches_per_point_nearest_centroid(tc: TestCase) {
    use docbert_plaid::{
        distance::squared_l2,
        kmeans::{assign_points, nearest_centroid},
    };
    let dim = tc.draw(codec_dim());
    let n = tc.draw(gs::integers::<usize>().min_value(1).max_value(32));
    let k = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let points = tc.draw(unit_rows(dim, n));
    let centroids = tc.draw(unit_rows(dim, k));

    let batched = assign_points(&points, &centroids, dim);
    assert_eq!(batched.len(), n);

    for (i, (point, &cluster)) in
        points.chunks_exact(dim).zip(batched.iter()).enumerate()
    {
        let scalar = nearest_centroid(point, &centroids, dim);
        if cluster == scalar {
            continue;
        }
        let d_batched =
            squared_l2(point, &centroids[cluster * dim..(cluster + 1) * dim]);
        let d_scalar =
            squared_l2(point, &centroids[scalar * dim..(scalar + 1) * dim]);
        // If the two paths picked different centroids, they must at
        // least be equidistant up to one f32 ULP — otherwise the
        // batched path truly picked a worse one.
        assert!(
            (d_batched - d_scalar).abs() <= 1e-5,
            "row {i}: batched picked {cluster} (d²={d_batched}) \
             but scalar picked {scalar} (d²={d_scalar})",
        );
    }
}

/// Shape: `assign_points` returns one assignment per input row, each
/// in `0..k`. Covers both the non-empty and empty-input paths so the
/// early-return branch in `assign_points` stays correct.
#[hegel::test(test_cases = 100)]
fn prop_assign_points_output_shape_and_range(tc: TestCase) {
    use docbert_plaid::kmeans::assign_points;
    let dim = tc.draw(codec_dim());
    let n = tc.draw(gs::integers::<usize>().min_value(0).max_value(32));
    let k = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let points = if n == 0 {
        Vec::new()
    } else {
        tc.draw(unit_rows(dim, n))
    };
    let centroids = tc.draw(unit_rows(dim, k));

    let out = assign_points(&points, &centroids, dim);
    assert_eq!(out.len(), n);
    for &c in &out {
        assert!(c < k, "assignment {c} out of range 0..{k}");
    }
}

/// Shape + algebraic: `top_n_centroids` has length `min(n, k)`, every
/// index is unique and in `0..k`, and the dot products at the returned
/// indices are non-increasing — i.e. the list is sorted by relevance,
/// most relevant first.
#[hegel::test(test_cases = 100)]
fn prop_top_n_centroids_shape_sort_uniqueness(tc: TestCase) {
    use docbert_plaid::{distance::dot, search::top_n_centroids};
    let dim = tc.draw(codec_dim());
    let k = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let n = tc.draw(gs::integers::<usize>().min_value(1).max_value(12));
    let point = tc.draw(unit_rows(dim, 1));
    let centroids = tc.draw(unit_rows(dim, k));

    let out = top_n_centroids(&point, &centroids, dim, n);
    assert_eq!(out.len(), n.min(k));

    let mut seen: Vec<usize> = out.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), out.len(), "duplicate centroid index in {out:?}");
    for &i in &out {
        assert!(i < k, "centroid index {i} out of range 0..{k}");
    }

    let scores: Vec<f32> = out
        .iter()
        .map(|&i| dot(&point, &centroids[i * dim..(i + 1) * dim]))
        .collect();
    for pair in scores.windows(2) {
        assert!(
            pair[0] >= pair[1],
            "top_n_centroids scores not descending: {scores:?}",
        );
    }
}

/// Shape: `update_centroids` returns a `k × dim` buffer; every cluster
/// with at least one assigned point equals the arithmetic mean of its
/// points within float ε, and empty clusters keep their previous
/// centroid byte-for-byte (preventing degenerate collapses to origin).
#[hegel::test(test_cases = 100)]
fn prop_update_centroids_shape_and_empty_cluster_preservation(tc: TestCase) {
    use docbert_plaid::kmeans::update_centroids;
    let dim = tc.draw(codec_dim());
    let k = tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
    let n = tc.draw(gs::integers::<usize>().min_value(0).max_value(24));
    let previous = tc.draw(unit_rows(dim, k));

    let (points, assignments): (Vec<f32>, Vec<usize>) = if n == 0 {
        (Vec::new(), Vec::new())
    } else {
        let points = tc.draw(unit_rows(dim, n));
        let assignments: Vec<usize> = (0..n)
            .map(|_| {
                tc.draw(gs::integers::<usize>().min_value(0).max_value(k - 1))
            })
            .collect();
        (points, assignments)
    };

    let updated = update_centroids(&points, &assignments, &previous, dim);
    assert_eq!(updated.len(), k * dim);

    for cluster in 0..k {
        let members: Vec<&[f32]> = points
            .chunks_exact(dim)
            .zip(&assignments)
            .filter_map(|(p, &a)| (a == cluster).then_some(p))
            .collect();
        let got = &updated[cluster * dim..(cluster + 1) * dim];
        if members.is_empty() {
            let prev = &previous[cluster * dim..(cluster + 1) * dim];
            assert_eq!(got, prev, "empty cluster {cluster} not preserved");
        } else {
            for d in 0..dim {
                let want = members.iter().map(|m| m[d]).sum::<f32>()
                    / members.len() as f32;
                assert!(
                    (got[d] - want).abs() <= 1e-5,
                    "cluster {cluster} dim {d}: got {} want {}",
                    got[d],
                    want,
                );
            }
        }
    }
}

/// Shape: every row of `farthest_first_init`'s output is byte-equal to
/// one of the input rows, and the first output row equals `points[..dim]`
/// (Gonzalez's algorithm picks row 0 as the first seed). Rules out the
/// kind of bug where seeding accidentally averages or perturbs inputs.
#[hegel::test(test_cases = 100)]
fn prop_farthest_first_init_rows_are_input_rows(tc: TestCase) {
    use docbert_plaid::kmeans::farthest_first_init;
    let dim = tc.draw(codec_dim());
    let k = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let n = tc.draw(gs::integers::<usize>().min_value(k).max_value(24));
    let points = tc.draw(unit_rows(dim, n));

    let seeds = farthest_first_init(&points, k, dim);
    assert_eq!(seeds.len(), k * dim);
    assert_eq!(&seeds[..dim], &points[..dim]);

    let input_rows: Vec<&[f32]> = points.chunks_exact(dim).collect();
    for seed in seeds.chunks_exact(dim) {
        assert!(
            input_rows.contains(&seed),
            "seed row {seed:?} is not any input row",
        );
    }
}

/// Algebraic / monotonicity: across Lloyd iterations (`assign →
/// update → repeat`) the within-cluster sum-of-squares is
/// non-increasing within a small f32 slack. Generalises the
/// single-seed test in `tests/properties.rs` across randomly drawn
/// dims/n/k so shrinking can surface the minimal failing
/// configuration if the invariant ever breaks.
#[hegel::test(test_cases = 50)]
fn prop_lloyd_inertia_non_increasing(tc: TestCase) {
    use docbert_plaid::{
        distance::squared_l2,
        kmeans::{assign_points, update_centroids},
    };
    let dim = tc.draw(codec_dim());
    let k = tc.draw(gs::integers::<usize>().min_value(2).max_value(6));
    let n = tc.draw(gs::integers::<usize>().min_value(k).max_value(32));
    let points = tc.draw(unit_rows(dim, n));
    let mut centroids = points[..k * dim].to_vec();

    let mut prev = f64::INFINITY;
    for iter in 0..8 {
        let assignments = assign_points(&points, &centroids, dim);
        let current: f64 = points
            .chunks_exact(dim)
            .zip(&assignments)
            .map(|(p, &c)| {
                squared_l2(p, &centroids[c * dim..(c + 1) * dim]) as f64
            })
            .sum();
        assert!(
            current <= prev + 1e-4,
            "inertia rose from {prev} to {current} at iter {iter}",
        );
        prev = current;
        centroids = update_centroids(&points, &assignments, &centroids, dim);
    }
}

/// Determinism: two `fit` calls with identical inputs produce
/// byte-identical centroids. Covers the farthest-first seeding too —
/// it must stay reproducible without an RNG so test fixtures remain
/// stable across runs.
#[hegel::test(test_cases = 40)]
fn prop_fit_is_deterministic(tc: TestCase) {
    use docbert_plaid::kmeans::fit;
    let dim = tc.draw(codec_dim());
    let k = tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
    let n = tc.draw(gs::integers::<usize>().min_value(k).max_value(24));
    let iters = tc.draw(gs::integers::<usize>().min_value(0).max_value(8));
    let points = tc.draw(unit_rows(dim, n));

    let a = fit(&points, k, dim, iters);
    let b = fit(&points, k, dim, iters);
    assert_eq!(a, b);
}

/// Round-trip: running `fit_with_init` on its own output yields the
/// same assignments — Lloyd's is idempotent once it has converged.
/// We use a generous `max_iters` in the first run to reach a fixed
/// point, then check that a second pass doesn't nudge anything.
#[hegel::test(test_cases = 40)]
fn prop_fit_with_init_idempotent_after_convergence(tc: TestCase) {
    use docbert_plaid::kmeans::{assign_points, fit_with_init};
    let dim = tc.draw(codec_dim());
    let k = tc.draw(gs::integers::<usize>().min_value(2).max_value(6));
    let n = tc.draw(gs::integers::<usize>().min_value(k).max_value(24));
    let points = tc.draw(unit_rows(dim, n));
    let initial = points[..k * dim].to_vec();

    let once = fit_with_init(&points, &initial, dim, 50);
    let twice = fit_with_init(&points, &once, dim, 50);

    let assign_once = assign_points(&points, &once, dim);
    let assign_twice = assign_points(&points, &twice, dim);
    assert_eq!(assign_once, assign_twice);
}

/// Differential: the tensor-matmul `fit_with_init` must agree with a
/// hand-rolled scalar Lloyd loop built from the public scalar
/// primitives (`nearest_centroid` + `update_centroids`) after the
/// same number of iterations, within a small float ε. If the tensor
/// path ever short-circuits an iteration or re-associates
/// accumulation, the final centroid buffers will drift and this test
/// flags it.
#[hegel::test(test_cases = 30)]
fn prop_fit_with_init_matches_scalar_lloyd(tc: TestCase) {
    use docbert_plaid::kmeans::{
        fit_with_init,
        nearest_centroid,
        update_centroids,
    };
    let dim = tc.draw(codec_dim());
    let k = tc.draw(gs::integers::<usize>().min_value(2).max_value(5));
    let n = tc.draw(gs::integers::<usize>().min_value(k).max_value(16));
    let iters = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let points = tc.draw(unit_rows(dim, n));
    let initial = points[..k * dim].to_vec();

    let tensor = fit_with_init(&points, &initial, dim, iters);

    let mut scalar = initial.clone();
    let mut prev: Option<Vec<usize>> = None;
    for _ in 0..iters {
        let assignments: Vec<usize> = points
            .chunks_exact(dim)
            .map(|p| nearest_centroid(p, &scalar, dim))
            .collect();
        if prev.as_ref() == Some(&assignments) {
            break;
        }
        scalar = update_centroids(&points, &assignments, &scalar, dim);
        prev = Some(assignments);
    }

    assert_eq!(tensor.len(), scalar.len());
    for (t, s) in tensor.iter().zip(scalar.iter()) {
        assert!(
            (t - s).abs() <= 1e-4,
            "tensor Lloyd drifted from scalar: tensor={tensor:?} scalar={scalar:?}",
        );
    }
}

// ---------------------------------------------------------------------------
// codec.rs
// ---------------------------------------------------------------------------

/// Round-trip: for every supported `nbits ∈ {1,2,4,8}` and every
/// code sequence where each element lives in `[0, 2^nbits)`, reading
/// positions back out of the packed buffer returns the original
/// sequence element-for-element. `pack_codes` is private, so we
/// exercise it through `encode_vector` on a codec whose cutoffs place
/// each integer-valued input cleanly into the target bucket — the
/// codes packed by `encode_vector` are then the target sequence.
#[hegel::test(test_cases = 200)]
fn prop_pack_read_roundtrip_all_nbits(tc: TestCase) {
    use docbert_plaid::codec::{ResidualCodec, read_code};
    let nbits: u32 = tc.draw(gs::sampled_from(vec![1u32, 2, 4, 8]));
    let num_buckets = 1u32 << nbits;
    let codes_per_byte = (8 / nbits) as usize;
    let n_bytes = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let dim = n_bytes * codes_per_byte;

    let max_code: u8 = (num_buckets - 1) as u8;
    let targets: Vec<u8> = tc.draw(
        gs::vecs(gs::integers::<u8>().min_value(0).max_value(max_code))
            .min_size(dim)
            .max_size(dim),
    );

    // Cutoffs at integer+0.5 so feeding in `b as f32` always lands in
    // bucket `b` (bucket = #cutoffs ≤ value).
    let cutoffs: Vec<f32> = (1..num_buckets).map(|i| i as f32 - 0.5).collect();
    let weights: Vec<f32> = (0..num_buckets).map(|i| i as f32).collect();
    let codec = ResidualCodec {
        nbits,
        dim,
        centroids: vec![0.0; dim],
        bucket_cutoffs: cutoffs,
        bucket_weights: weights,
    };

    let input: Vec<f32> = targets.iter().map(|&b| b as f32).collect();
    let encoded = codec.encode_vector(&input);
    for (i, &expected) in targets.iter().enumerate() {
        let got = read_code(&encoded.codes, i, nbits);
        assert_eq!(
            got, expected,
            "nbits={nbits} i={i}: got {got}, expected {expected}",
        );
    }
}

/// Shape: `packed_bytes_per_vector(dim, nbits) == (dim * nbits + 7) / 8`
/// and the codes returned by `encode_vector` have exactly that many
/// bytes. Locks in the paper's §4.5 packed-layout byte count so any
/// off-by-one in the packing path is caught at the boundary.
#[hegel::test(test_cases = 200)]
fn prop_packed_bytes_formula(tc: TestCase) {
    use docbert_plaid::codec::{ResidualCodec, packed_bytes_per_vector};
    let nbits: u32 = tc.draw(gs::sampled_from(vec![1u32, 2, 4, 8]));
    let codes_per_byte = (8 / nbits) as usize;
    let n_bytes = tc.draw(gs::integers::<usize>().min_value(1).max_value(32));
    let dim = n_bytes * codes_per_byte;

    let expected = (dim * nbits as usize).div_ceil(8);
    assert_eq!(packed_bytes_per_vector(dim, nbits), expected);

    let num_buckets = 1u32 << nbits;
    let codec = ResidualCodec {
        nbits,
        dim,
        centroids: vec![0.0; dim],
        bucket_cutoffs: (1..num_buckets).map(|i| i as f32 - 0.5).collect(),
        bucket_weights: (0..num_buckets).map(|i| i as f32).collect(),
    };
    let encoded = codec.encode_vector(&vec![0.0; dim]);
    assert_eq!(encoded.codes.len(), expected);
}

/// Algebraic: `bucket_for_value` is monotone — if `v1 ≤ v2` then the
/// bucket containing v1 is ≤ the bucket containing v2 for any
/// ascending cutoff vector. `bucket_for_value` is module-private, so
/// we test it indirectly via `encode_vector` + `read_code` on a 2-dim
/// codec fed `(v1, v2)`.
#[hegel::test(test_cases = 200)]
fn prop_bucket_for_value_monotonic(tc: TestCase) {
    use docbert_plaid::codec::{ResidualCodec, read_code};
    let nbits: u32 = tc.draw(gs::sampled_from(vec![2u32, 4]));
    let num_buckets = 1u32 << nbits;
    let codec = ResidualCodec {
        nbits,
        dim: 2,
        centroids: vec![0.0, 0.0],
        bucket_cutoffs: (1..num_buckets).map(|i| i as f32 - 0.5).collect(),
        bucket_weights: (0..num_buckets).map(|i| i as f32).collect(),
    };
    let (v1, v2): (f32, f32) = {
        let a: f32 = tc.draw(
            gs::floats::<f32>()
                .min_value(-2.0)
                .max_value((num_buckets + 2) as f32)
                .allow_nan(false)
                .allow_infinity(false),
        );
        let b: f32 = tc.draw(
            gs::floats::<f32>()
                .min_value(-2.0)
                .max_value((num_buckets + 2) as f32)
                .allow_nan(false)
                .allow_infinity(false),
        );
        if a <= b { (a, b) } else { (b, a) }
    };

    let encoded = codec.encode_vector(&[v1, v2]);
    let b1 = read_code(&encoded.codes, 0, nbits);
    let b2 = read_code(&encoded.codes, 1, nbits);
    assert!(
        b1 <= b2,
        "bucket_for_value not monotone: v1={v1} → {b1}, v2={v2} → {b2}",
    );
}

/// Shape: every code produced by `encode_vector` lies in
/// `[0, 2^nbits)`. Paired with the pack/read roundtrip this nails
/// down both ends of the packing contract — bucket range upstream
/// and lossless read downstream.
#[hegel::test(test_cases = 100)]
fn prop_bucket_for_value_range(tc: TestCase) {
    use docbert_plaid::codec::{ResidualCodec, read_code};
    let nbits: u32 = tc.draw(gs::sampled_from(vec![1u32, 2, 4, 8]));
    let num_buckets = 1u32 << nbits;
    let codes_per_byte = (8 / nbits) as usize;
    let n_bytes = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let dim = n_bytes * codes_per_byte;

    let codec = ResidualCodec {
        nbits,
        dim,
        centroids: vec![0.0; dim],
        bucket_cutoffs: (1..num_buckets).map(|i| i as f32 - 0.5).collect(),
        bucket_weights: (0..num_buckets).map(|i| i as f32).collect(),
    };
    let input = tc.draw(finite_floats(dim));

    let encoded = codec.encode_vector(&input);
    for i in 0..dim {
        let code = read_code(&encoded.codes, i, nbits);
        assert!(
            (code as u32) < num_buckets,
            "code {code} out of range 0..{num_buckets} at position {i}",
        );
    }
}

/// Algebraic: `train_quantizer` returns non-decreasing cutoffs for any
/// non-NaN residual sample. This is the quantile-picking invariant —
/// cutoffs are chosen at equally-spaced quantile positions of a
/// sorted copy, so they must come out sorted.
#[hegel::test(test_cases = 100)]
fn prop_train_quantizer_cutoffs_monotonic(tc: TestCase) {
    use docbert_plaid::codec::train_quantizer;
    let nbits: u32 = tc.draw(gs::sampled_from(vec![1u32, 2, 4, 8]));
    let n = tc.draw(gs::integers::<usize>().min_value(1).max_value(256));
    let residuals = tc.draw(finite_floats(n));
    let (cutoffs, _) = train_quantizer(&residuals, nbits);
    for pair in cutoffs.windows(2) {
        assert!(pair[0] <= pair[1], "cutoffs not monotone: {:?}", &cutoffs,);
    }
}

/// Shape: `train_quantizer(_, nbits)` returns `2^nbits - 1` cutoffs
/// and `2^nbits` weights. These two counts are load-bearing in
/// `ResidualCodec::validate` and in the persistence layer's fixed
/// header sizes, so any drift would make every downstream codec fail
/// to construct.
#[hegel::test(test_cases = 80)]
fn prop_train_quantizer_shapes(tc: TestCase) {
    use docbert_plaid::codec::train_quantizer;
    let nbits: u32 = tc.draw(gs::sampled_from(vec![1u32, 2, 4, 8]));
    let n = tc.draw(gs::integers::<usize>().min_value(1).max_value(256));
    let residuals = tc.draw(finite_floats(n));
    let num_buckets = 1usize << nbits;
    let (cutoffs, weights) = train_quantizer(&residuals, nbits);
    assert_eq!(cutoffs.len(), num_buckets - 1);
    assert_eq!(weights.len(), num_buckets);
}

/// Algebraic / monotonicity: MSE reconstruction error trained and
/// evaluated on the same pool is non-increasing as `nbits` grows from
/// 1 → 2 → 4 → 8. More buckets ⇒ narrower buckets ⇒ weights sit
/// closer to the residuals they represent. Generalises the
/// hand-rolled test in `tests/properties.rs` across random residual
/// distributions.
#[hegel::test(test_cases = 30)]
fn prop_reconstruction_error_decreases_with_nbits(tc: TestCase) {
    use docbert_plaid::codec::{ResidualCodec, train_quantizer};
    let n = tc.draw(gs::integers::<usize>().min_value(32).max_value(512));
    let pool = tc.draw(finite_floats(n));

    let mut prev = f64::INFINITY;
    for nbits in [1u32, 2, 4, 8] {
        let (cutoffs, weights) = train_quantizer(&pool, nbits);
        let codec = ResidualCodec {
            nbits,
            dim: 1,
            centroids: vec![0.0],
            bucket_cutoffs: cutoffs,
            bucket_weights: weights,
        };
        let mse: f64 = pool
            .iter()
            .map(|v| codec.reconstruction_error(&[*v]) as f64)
            .sum::<f64>()
            / pool.len() as f64;
        assert!(
            mse <= prev + 1e-6,
            "MSE rose from {prev} to {mse} at nbits={nbits}",
        );
        prev = mse;
    }
}

/// Algebraic: `reconstruction_error(v) >= 0` for any encodable input
/// and any valid codec. Trivial from squared-L2 non-negativity, but
/// locks in the invariant so anyone swapping in a signed distance for
/// "relative error" gets caught.
#[hegel::test(test_cases = 100)]
fn prop_reconstruction_error_non_negative(tc: TestCase) {
    use docbert_plaid::codec::{ResidualCodec, train_quantizer};
    let nbits: u32 = tc.draw(gs::sampled_from(vec![1u32, 2, 4, 8]));
    let n = tc.draw(gs::integers::<usize>().min_value(16).max_value(128));
    let pool = tc.draw(finite_floats(n));
    let (cutoffs, weights) = train_quantizer(&pool, nbits);
    let codec = ResidualCodec {
        nbits,
        dim: 1,
        centroids: vec![0.0],
        bucket_cutoffs: cutoffs,
        bucket_weights: weights,
    };
    let probe_n = tc.draw(gs::integers::<usize>().min_value(1).max_value(32));
    let probes = tc.draw(finite_floats(probe_n));
    for v in &probes {
        assert!(codec.reconstruction_error(&[*v]) >= 0.0);
    }
}

/// Differential: `decode_vector_with_table` must produce bit-for-bit
/// the same output as scalar `decode_vector` for every supported
/// nbits and any valid encoded vector. This is the PLAID §4.5 lookup
/// table path and the inner loop of `batch_maxsim`; any drift silently
/// corrupts MaxSim scores.
#[hegel::test(test_cases = 100)]
fn prop_decode_table_matches_scalar(tc: TestCase) {
    use docbert_plaid::codec::{DecodeTable, ResidualCodec};
    let nbits: u32 = tc.draw(gs::sampled_from(vec![1u32, 2, 4, 8]));
    let codes_per_byte = (8 / nbits) as usize;
    let n_bytes = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let dim = n_bytes * codes_per_byte;

    let num_buckets = 1u32 << nbits;
    let codec = ResidualCodec {
        nbits,
        dim,
        centroids: tc.draw(unit_rows(dim, 2)),
        bucket_cutoffs: (1..num_buckets).map(|i| i as f32 - 0.5).collect(),
        bucket_weights: (0..num_buckets).map(|i| i as f32).collect(),
    };
    let input = tc.draw(finite_floats(dim));
    let encoded = codec.encode_vector(&input);

    let scalar = codec.decode_vector(&encoded);
    let table = DecodeTable::new(&codec);
    let via_table = codec.decode_vector_with_table(&encoded, &table);
    assert_eq!(scalar, via_table);
}

/// Differential: `batch_encode_tokens` must produce the same
/// `(centroid_id, codes)` per-token as calling `encode_vector` on
/// each row independently. Catches any divergence between the
/// matmul-driven batched assignment and the scalar single-token
/// path — a drift there would poison every index built via
/// `build_index`.
#[hegel::test(test_cases = 50)]
fn prop_batch_encode_matches_per_token(tc: TestCase) {
    use docbert_plaid::codec::{ResidualCodec, train_quantizer};
    let nbits: u32 = tc.draw(gs::sampled_from(vec![1u32, 2, 4, 8]));
    let codes_per_byte = (8 / nbits) as usize;
    let n_bytes = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
    let dim = n_bytes * codes_per_byte;
    let k = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
    let n_tokens = tc.draw(gs::integers::<usize>().min_value(1).max_value(16));

    let centroids = tc.draw(unit_rows(dim, k));
    let residual_pool = tc.draw(finite_floats(64));
    let (cutoffs, weights) = train_quantizer(&residual_pool, nbits);
    let codec = ResidualCodec {
        nbits,
        dim,
        centroids,
        bucket_cutoffs: cutoffs,
        bucket_weights: weights,
    };
    let tokens = tc.draw(unit_rows(dim, n_tokens));

    let (cids, packed) = codec.batch_encode_tokens(&tokens);
    let packed_per = codec.packed_bytes();
    assert_eq!(cids.len(), n_tokens);
    assert_eq!(packed.len(), n_tokens * packed_per);

    for (i, token) in tokens.chunks_exact(dim).enumerate() {
        let expected = codec.encode_vector(token);
        assert_eq!(
            cids[i], expected.centroid_id,
            "token {i}: batched centroid_id differs",
        );
        let batch_slice = &packed[i * packed_per..(i + 1) * packed_per];
        assert_eq!(
            batch_slice,
            expected.codes.as_slice(),
            "token {i}: batched codes differ from per-token encode",
        );
    }
}

/// Round-trip: `encode(decode(encode(v))) == encode(v)` on
/// single-centroid codecs with strictly-increasing cutoffs. Once a
/// residual is snapped to its bucket weight (strictly inside the
/// bucket's open interval), decoding and re-encoding must land in
/// the same bucket — the centroid choice can't change with only one
/// centroid, so only the bucket idempotence is under test. This
/// deliberately avoids the multi-centroid Voronoi-crossing case,
/// where the decoded point can drift into a neighbour cell and flip
/// `centroid_id`.
#[hegel::test(test_cases = 100)]
fn prop_encode_decode_encode_idempotent(tc: TestCase) {
    use docbert_plaid::codec::{ResidualCodec, train_quantizer};
    let nbits: u32 = tc.draw(gs::sampled_from(vec![2u32, 4, 8]));
    let codes_per_byte = (8 / nbits) as usize;
    let n_bytes = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
    let dim = n_bytes * codes_per_byte;

    // Dense linear ramp over [-1, 1] with 4x more points than buckets
    // so every bucket is populated and cutoffs come out strictly
    // increasing.
    let num_buckets = 1usize << nbits;
    let pool_n = num_buckets * 4;
    let residual_pool: Vec<f32> = (0..pool_n)
        .map(|i| (i as f32 / pool_n as f32) * 2.0 - 1.0)
        .collect();
    let (cutoffs, weights) = train_quantizer(&residual_pool, nbits);
    let codec = ResidualCodec {
        nbits,
        dim,
        centroids: vec![0.0; dim],
        bucket_cutoffs: cutoffs,
        bucket_weights: weights,
    };

    let v = tc.draw(unit_rows(dim, 1));
    let enc1 = codec.encode_vector(&v);
    let dec1 = codec.decode_vector(&enc1);
    let enc2 = codec.encode_vector(&dec1);
    assert_eq!(enc1, enc2, "encode→decode→encode not idempotent");
}

// ---------------------------------------------------------------------------
// index.rs
// ---------------------------------------------------------------------------

/// Shape: after `build_index`, sum of encoded token lengths equals
/// sum of input `n_tokens`; `doc_ids` preserves input order; each
/// encoded doc's length equals the input doc's `n_tokens`.
#[hegel::test(test_cases = 30)]
fn prop_build_index_shape(tc: TestCase) {
    use docbert_plaid::index::build_index;
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 1, 5, 6, 4));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let expected_ids: Vec<u64> = docs.iter().map(|d| d.doc_id).collect();

    let index = build_index(&docs, params);

    assert_eq!(index.num_documents(), docs.len());
    assert_eq!(index.num_tokens(), total_tokens);
    assert_eq!(index.doc_ids, expected_ids);
    for (encoded, doc) in index.doc_tokens.iter().zip(docs.iter()) {
        assert_eq!(encoded.len(), doc.n_tokens);
    }
}

/// Shape: for every valid generated corpus, the codec `build_index`
/// trains validates. This is the entry point for both the persistence
/// layer and `apply_update` — a malformed codec here would cascade
/// into every downstream test and production save path.
#[hegel::test(test_cases = 30)]
fn prop_build_index_codec_validates(tc: TestCase) {
    use docbert_plaid::index::build_index;
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 1, 5, 6, 4));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));

    let index = build_index(&docs, params);
    assert!(
        index.codec.validate().is_ok(),
        "build_index produced a codec that fails validate(): {:?}",
        index.codec.validate(),
    );
    assert_eq!(index.codec.dim, params.dim);
    assert_eq!(index.codec.nbits, params.nbits);
    assert_eq!(index.codec.num_centroids(), params.k_centroids);
}
