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
///
/// If a drawn row's norm is too small to normalise stably it is
/// replaced by a standard basis vector (`e₀`). This keeps every row a
/// true unit vector regardless of what the drawn floats happen to be —
/// otherwise tests that rely on `‖a‖ = 1` (e.g. self-dot = 1) fail on
/// the all-near-zero corner.
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
        let norm_sq: f32 = row.iter().map(|x| x * x).sum();
        if norm_sq < 1e-6 {
            row.fill(0.0);
            row[0] = 1.0;
        } else {
            let norm = norm_sq.sqrt();
            for x in row {
                *x /= norm;
            }
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

/// Algebraic: on unit-norm vectors `a`, `b`:
///   - `dot(a, a) ≈ 1`
///   - `dot(a, -a) ≈ -1`
///   - `squared_l2(a, b) ≈ 2·(1 − dot(a, b))`
///
/// The third identity is what justifies ColBERT/PLAID using dot
/// product as the per-token similarity: on the unit sphere it's
/// monotonic with squared L2, so argmax on dot matches argmin on
/// L2. Drift here would silently break that equivalence.
#[hegel::test(test_cases = 200)]
fn prop_unit_vector_distance_identities(tc: TestCase) {
    use docbert_plaid::distance::{dot, squared_l2};
    let dim = tc.draw(codec_dim());
    let ab = tc.draw(unit_rows(dim, 2));
    let (a, b) = ab.split_at(dim);

    let self_dot = dot(a, a);
    assert!(
        (self_dot - 1.0).abs() <= 1e-5,
        "unit-norm self-dot {self_dot} != 1",
    );

    let neg_a: Vec<f32> = a.iter().map(|x| -x).collect();
    let anti_dot = dot(a, &neg_a);
    assert!(
        (anti_dot + 1.0).abs() <= 1e-5,
        "antipodal dot {anti_dot} != -1",
    );

    let cos = dot(a, b);
    let expected = 2.0 * (1.0 - cos);
    let got = squared_l2(a, b);
    assert!(
        (got - expected).abs() <= 1e-5,
        "squared_l2 {got} != 2(1 - cos) = {expected}",
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

    let batched = assign_points(&points, &centroids, dim).unwrap();
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

    let out = assign_points(&points, &centroids, dim).unwrap();
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

    let seeds = farthest_first_init(&points, k, dim).unwrap();
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
        let assignments = assign_points(&points, &centroids, dim).unwrap();
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

    let a = fit(&points, k, dim, iters).unwrap();
    let b = fit(&points, k, dim, iters).unwrap();
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

    let once = fit_with_init(&points, &initial, dim, 50).unwrap();
    let twice = fit_with_init(&points, &once, dim, 50).unwrap();

    let assign_once = assign_points(&points, &once, dim).unwrap();
    let assign_twice = assign_points(&points, &twice, dim).unwrap();
    assert_eq!(assign_once, assign_twice);
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
    let encoded = codec.encode_vector(&input).unwrap();
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
    let encoded = codec.encode_vector(&vec![0.0; dim]).unwrap();
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

    let encoded = codec.encode_vector(&[v1, v2]).unwrap();
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

    let encoded = codec.encode_vector(&input).unwrap();
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
    let (cutoffs, _) = train_quantizer(residuals, nbits);
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
    let (cutoffs, weights) = train_quantizer(residuals, nbits);
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
        let (cutoffs, weights) = train_quantizer(pool.clone(), nbits);
        let codec = ResidualCodec {
            nbits,
            dim: 1,
            centroids: vec![0.0],
            bucket_cutoffs: cutoffs,
            bucket_weights: weights,
        };
        let mse: f64 = pool
            .iter()
            .map(|v| codec.reconstruction_error(&[*v]).unwrap() as f64)
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
    let (cutoffs, weights) = train_quantizer(pool, nbits);
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
        assert!(codec.reconstruction_error(&[*v]).unwrap() >= 0.0);
    }
}

/// Differential: `encode_vector` picks the same centroid as scalar
/// `nearest_centroid` applied to the original vector. Ties may break
/// either way between the two paths, so on disagreement we verify
/// the chosen centroids are equidistant within one f32 ULP.
#[hegel::test(test_cases = 100)]
fn prop_encode_picks_nearest_centroid(tc: TestCase) {
    use docbert_plaid::{
        codec::{ResidualCodec, train_quantizer},
        distance::squared_l2,
        kmeans::nearest_centroid,
    };
    let nbits: u32 = tc.draw(gs::sampled_from(vec![2u32, 4]));
    let codes_per_byte = (8 / nbits) as usize;
    let n_bytes = tc.draw(gs::integers::<usize>().min_value(1).max_value(2));
    let dim = n_bytes * codes_per_byte;
    let k = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
    let centroids = tc.draw(unit_rows(dim, k));
    let residual_pool = tc.draw(finite_floats(64));
    let (cutoffs, weights) = train_quantizer(residual_pool, nbits);
    let codec = ResidualCodec {
        nbits,
        dim,
        centroids: centroids.clone(),
        bucket_cutoffs: cutoffs,
        bucket_weights: weights,
    };
    let v = tc.draw(unit_rows(dim, 1));

    let encoded = codec.encode_vector(&v).unwrap();
    let scalar = nearest_centroid(&v, &centroids, dim);
    if encoded.centroid_id as usize == scalar {
        return;
    }
    let encoded_idx = encoded.centroid_id as usize;
    let d_enc =
        squared_l2(&v, &centroids[encoded_idx * dim..(encoded_idx + 1) * dim]);
    let d_sca = squared_l2(&v, &centroids[scalar * dim..(scalar + 1) * dim]);
    assert!(
        (d_enc - d_sca).abs() <= 1e-5,
        "encode picked {encoded_idx} (d²={d_enc}) but scalar picked \
         {scalar} (d²={d_sca})",
    );
}

/// Algebraic: for a codec whose interior bucket weights are at
/// bucket midpoints, any input inside the outermost cutoffs
/// reconstructs within half the widest interior bucket width.
///
/// Generalises the hand-picked unit test by drawing the nbits,
/// probe position, and input value — but keeps the codec
/// construction explicit so the "weights at bucket midpoints"
/// precondition stays true by construction. Without that
/// precondition (e.g. on an arbitrarily trained codec), the bound
/// doesn't hold in general.
#[hegel::test(test_cases = 200)]
fn prop_codec_interior_error_bounded_by_half_bucket_width(tc: TestCase) {
    use docbert_plaid::codec::ResidualCodec;
    let nbits: u32 = tc.draw(gs::sampled_from(vec![2u32, 4, 8]));
    let num_buckets = 1u32 << nbits;
    // Uniform interior cutoffs over [-1, 1] and weights at bucket
    // centres — exactly the midpoint construction the original test
    // used, now fuzzed over nbits and probe value.
    let step = 2.0 / num_buckets as f32;
    let bucket_cutoffs: Vec<f32> =
        (1..num_buckets).map(|i| -1.0 + i as f32 * step).collect();
    let bucket_weights: Vec<f32> = (0..num_buckets)
        .map(|i| -1.0 + (i as f32 + 0.5) * step)
        .collect();
    let codec = ResidualCodec {
        nbits,
        dim: 1,
        centroids: vec![0.0],
        bucket_cutoffs,
        bucket_weights,
    };

    // Input strictly inside the outermost cutoffs so it always
    // lands in an interior bucket where the half-width bound holds.
    let v: f32 = tc.draw(
        gs::floats::<f32>()
            .min_value(-1.0 + step)
            .max_value(1.0 - step)
            .allow_nan(false)
            .allow_infinity(false),
    );

    let encoded = codec.encode_vector(&[v]).unwrap();
    let decoded = codec.decode_vector(&encoded).unwrap();
    let half_width = step / 2.0;
    assert!(
        (decoded[0] - v).abs() <= half_width + 1e-5,
        "per-dim error {} exceeds half bucket width {half_width}",
        (decoded[0] - v).abs(),
    );
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
    let encoded = codec.encode_vector(&input).unwrap();

    let scalar = codec.decode_vector(&encoded).unwrap();
    let table = DecodeTable::new(&codec);
    let via_table = codec.decode_vector_with_table(&encoded, &table).unwrap();
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
    let (cutoffs, weights) = train_quantizer(residual_pool, nbits);
    let codec = ResidualCodec {
        nbits,
        dim,
        centroids,
        bucket_cutoffs: cutoffs,
        bucket_weights: weights,
    };
    let tokens = tc.draw(unit_rows(dim, n_tokens));

    let (cids, packed) = codec.batch_encode_tokens(&tokens).unwrap();
    let packed_per = codec.packed_bytes();
    assert_eq!(cids.len(), n_tokens);
    assert_eq!(packed.len(), n_tokens * packed_per);

    for (i, token) in tokens.chunks_exact(dim).enumerate() {
        let expected = codec.encode_vector(token).unwrap();
        let batch_slice = &packed[i * packed_per..(i + 1) * packed_per];
        if cids[i] == expected.centroid_id {
            assert_eq!(
                batch_slice,
                expected.codes.as_slice(),
                "token {i}: batched codes differ from per-token encode",
            );
            continue;
        }
        // Tie break: scalar `nearest_centroid` takes the earlier
        // index, the batched matmul argmin can land on a later one.
        // When that happens the two paths must still agree on
        // distance, and the packed codes will differ because they
        // are residuals against different centroids — which is
        // correct, not a regression.
        let expected_idx = expected.centroid_id as usize;
        let batched_idx = cids[i] as usize;
        let centroids = &codec.centroids;
        let d_exp = docbert_plaid::distance::squared_l2(
            token,
            &centroids[expected_idx * dim..(expected_idx + 1) * dim],
        );
        let d_bat = docbert_plaid::distance::squared_l2(
            token,
            &centroids[batched_idx * dim..(batched_idx + 1) * dim],
        );
        assert!(
            (d_exp - d_bat).abs() <= 1e-5,
            "token {i}: batched picked {batched_idx} (d²={d_bat}) \
             but scalar picked {expected_idx} (d²={d_exp})",
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
    let (cutoffs, weights) = train_quantizer(residual_pool, nbits);
    let codec = ResidualCodec {
        nbits,
        dim,
        centroids: vec![0.0; dim],
        bucket_cutoffs: cutoffs,
        bucket_weights: weights,
    };

    let v = tc.draw(unit_rows(dim, 1));
    let enc1 = codec.encode_vector(&v).unwrap();
    let dec1 = codec.decode_vector(&enc1).unwrap();
    let enc2 = codec.encode_vector(&dec1).unwrap();
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

    let index = build_index(&docs, params).unwrap();

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

    let index = build_index(&docs, params).unwrap();
    assert!(
        index.codec.validate().is_ok(),
        "build_index produced a codec that fails validate(): {:?}",
        index.codec.validate(),
    );
    assert_eq!(index.codec.dim, params.dim);
    assert_eq!(index.codec.nbits, params.nbits);
    assert_eq!(index.codec.num_centroids(), params.k_centroids);
}

/// Shape: every `(doc_idx, centroid_id)` pair implied by the encoded
/// tokens appears in the IVF's posting list for that centroid; each
/// list is sorted ascending with no duplicates; and
/// `ivf.num_centroids() == params.k_centroids`. This is the paper's
/// "centroid → unique passage ids" contract and the foundation of
/// candidate generation at query time.
#[hegel::test(test_cases = 30)]
fn prop_build_inverted_file_invariants(tc: TestCase) {
    use docbert_plaid::index::build_index;
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 1, 5, 6, 4));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));

    let index = build_index(&docs, params).unwrap();
    assert_eq!(index.ivf.num_centroids(), params.k_centroids);

    for (doc_idx, encoded) in index.doc_tokens.iter().enumerate() {
        for ev in encoded {
            let list = index.ivf.docs_for_centroid(ev.centroid_id as usize);
            assert!(
                list.contains(&(doc_idx as u32)),
                "doc_idx={doc_idx} missing from centroid {} postings",
                ev.centroid_id,
            );
        }
    }

    for c in 0..index.ivf.num_centroids() {
        let postings = index.ivf.docs_for_centroid(c);
        assert!(
            postings.windows(2).all(|w| w[0] < w[1]),
            "centroid {c} postings not strictly ascending: {postings:?}",
        );
    }
}

// ---------------------------------------------------------------------------
// search.rs
// ---------------------------------------------------------------------------

/// Algebraic + shape: `search` returns at most `top_k` results, its
/// scores are non-increasing, and equal-scoring entries are ordered
/// by ascending `doc_id`. All three are load-bearing for downstream
/// ranking consumers.
#[hegel::test(test_cases = 30)]
fn prop_search_scores_non_increasing_and_capped_by_top_k(tc: TestCase) {
    use docbert_plaid::{
        index::build_index,
        search::{SearchParams, search},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 2, 6, 5, 6));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    let n_q = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
    let query = tc.draw(unit_rows(dim, n_q));
    let top_k = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let n_probe = tc.draw(
        gs::integers::<usize>()
            .min_value(1)
            .max_value(params.k_centroids),
    );

    let results = search(
        &index,
        &query,
        SearchParams {
            top_k,
            n_probe,
            n_candidate_docs: None,
            centroid_score_threshold: None,
        },
    )
    .unwrap();

    assert!(results.len() <= top_k);
    for pair in results.windows(2) {
        assert!(
            pair[0].score > pair[1].score
                || (pair[0].score == pair[1].score
                    && pair[0].doc_id < pair[1].doc_id),
            "search ordering violated: {pair:?}",
        );
    }
}

/// Determinism: two `search` calls with identical inputs produce
/// identical result vectors — doc_id-for-doc_id and score-for-score.
/// Covers both the baseline path and the full cascade (pruning +
/// centroid interaction), since all four knobs are drawn.
#[hegel::test(test_cases = 30)]
fn prop_search_deterministic(tc: TestCase) {
    use docbert_plaid::{
        index::build_index,
        search::{SearchParams, search},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 2, 6, 5, 6));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    let n_q = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
    let query = tc.draw(unit_rows(dim, n_q));
    let top_k = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let n_probe = tc.draw(
        gs::integers::<usize>()
            .min_value(1)
            .max_value(params.k_centroids),
    );
    let n_candidate_docs: Option<usize> = tc.draw(gs::optional(
        gs::integers::<usize>().min_value(1).max_value(32),
    ));
    let threshold: Option<f32> = tc.draw(gs::optional(
        gs::floats::<f32>()
            .min_value(-2.0)
            .max_value(2.0)
            .allow_nan(false)
            .allow_infinity(false),
    ));

    let sp = SearchParams {
        top_k,
        n_probe,
        n_candidate_docs,
        centroid_score_threshold: threshold,
    };
    let a = search(&index, &query, sp).unwrap();
    let b = search(&index, &query, sp).unwrap();
    assert_eq!(a, b);
}

/// Metamorphic: MaxSim is a sum-of-max over query tokens, so the
/// *set* of (doc_id, score) results must be identical under any
/// permutation of query-token order. We shuffle via a drawn
/// permutation of token indices, rerun search, and assert the two
/// result sets are equal after sorting by doc_id. Tie-break order
/// by doc_id means the returned vectors also agree element-wise.
#[hegel::test(test_cases = 30)]
fn prop_search_permutation_invariant_over_query_tokens(tc: TestCase) {
    use docbert_plaid::{
        index::build_index,
        search::{SearchParams, search},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 2, 6, 5, 6));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    let n_q = tc.draw(gs::integers::<usize>().min_value(2).max_value(5));
    let query = tc.draw(unit_rows(dim, n_q));

    // Build a drawn permutation of 0..n_q via a Fisher-Yates-ish
    // shuffle using hegel draws — valid-by-construction.
    let mut perm: Vec<usize> = (0..n_q).collect();
    for i in (1..n_q).rev() {
        let j = tc.draw(gs::integers::<usize>().min_value(0).max_value(i));
        perm.swap(i, j);
    }
    let permuted: Vec<f32> = perm
        .iter()
        .flat_map(|&i| query[i * dim..(i + 1) * dim].iter().copied())
        .collect();

    let sp = SearchParams {
        top_k: 10,
        n_probe: params.k_centroids,
        n_candidate_docs: None,
        centroid_score_threshold: None,
    };
    let a = search(&index, &query, sp).unwrap();
    let b = search(&index, &permuted, sp).unwrap();

    // Tie-break is doc_id asc so the vectors agree element-wise; scores
    // must match within a tiny ε since the only difference is the
    // order of the per-query-token partial maxes the reducer sums.
    assert_eq!(a.len(), b.len());
    for (ra, rb) in a.iter().zip(b.iter()) {
        assert_eq!(ra.doc_id, rb.doc_id);
        assert!(
            (ra.score - rb.score).abs() <= 1e-4,
            "permutation changed score: {ra:?} vs {rb:?}",
        );
    }
}

/// Metamorphic: if a document is duplicated under a new doc_id, both
/// copies must rank together with identical scores. A failure would
/// imply the scorer depends on doc position or uses non-stable
/// per-doc state.
#[hegel::test(test_cases = 20)]
fn prop_search_duplicated_doc_returns_both_with_same_score(tc: TestCase) {
    use docbert_plaid::{
        index::{DocumentTokens, build_index},
        search::{SearchParams, search},
    };
    let dim = tc.draw(codec_dim());
    let mut docs = tc.draw(corpus(dim, 2, 4, 5, 6));
    // Pick one doc with tokens to duplicate.
    let target_idx = docs.iter().position(|d| d.n_tokens > 0).unwrap_or(0);
    let clone_tokens = docs[target_idx].tokens.clone();
    let clone_n = docs[target_idx].n_tokens;
    if clone_n == 0 {
        // Ensure we have a non-empty doc to duplicate.
        docs[target_idx].tokens = tc.draw(unit_rows(dim, 1));
        docs[target_idx].n_tokens = 1;
    }
    let clone_tokens = if clone_n == 0 {
        docs[target_idx].tokens.clone()
    } else {
        clone_tokens
    };
    let clone_n = docs[target_idx].n_tokens;
    let clone_id: u64 = docs.iter().map(|d| d.doc_id).max().unwrap() + 1;
    docs.push(DocumentTokens {
        doc_id: clone_id,
        tokens: clone_tokens,
        n_tokens: clone_n,
    });
    let original_id = docs[target_idx].doc_id;

    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    let query = tc.draw(unit_rows(dim, 2));
    let sp = SearchParams {
        top_k: docs.len(),
        n_probe: params.k_centroids,
        n_candidate_docs: None,
        centroid_score_threshold: None,
    };
    let results = search(&index, &query, sp).unwrap();

    let original_score = results
        .iter()
        .find(|r| r.doc_id == original_id)
        .map(|r| r.score);
    let clone_score = results
        .iter()
        .find(|r| r.doc_id == clone_id)
        .map(|r| r.score);
    match (original_score, clone_score) {
        (Some(a), Some(b)) => assert!(
            (a - b).abs() <= 1e-4,
            "duplicated doc scored differently: {a} vs {b}",
        ),
        (None, None) => {
            // Neither surfaced — the query has no match for this
            // doc in either copy, which is fine.
        }
        other => panic!(
            "only one copy surfaced: original={:?} clone={:?}",
            other.0, other.1,
        ),
    }
}

/// Metamorphic: `n_candidate_docs = Some(huge)` must produce the same
/// result vector as `None`. A shortlist large enough to admit every
/// candidate is a no-op, so the cascade should not alter the final
/// ranking.
#[hegel::test(test_cases = 20)]
fn prop_search_large_shortlist_equals_none(tc: TestCase) {
    use docbert_plaid::{
        index::build_index,
        search::{SearchParams, search},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 2, 6, 5, 6));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    let query = tc.draw(unit_rows(dim, 2));
    let base = SearchParams {
        top_k: 10,
        n_probe: params.k_centroids,
        n_candidate_docs: None,
        centroid_score_threshold: None,
    };
    let shortlisted = SearchParams {
        n_candidate_docs: Some(100_000),
        ..base
    };
    let a = search(&index, &query, base).unwrap();
    let b = search(&index, &query, shortlisted).unwrap();
    assert_eq!(a, b);
}

/// Metamorphic: `centroid_score_threshold = Some(-very_large)` is a
/// no-op — every centroid clears a threshold below the attainable
/// dot-product floor, so pruning drops nothing. Results must match
/// `None` byte-for-byte.
#[hegel::test(test_cases = 20)]
fn prop_search_unreachable_threshold_equals_none(tc: TestCase) {
    use docbert_plaid::{
        index::build_index,
        search::{SearchParams, search},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 2, 6, 5, 6));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    let query = tc.draw(unit_rows(dim, 2));
    let base = SearchParams {
        top_k: 10,
        n_probe: params.k_centroids,
        n_candidate_docs: None,
        centroid_score_threshold: None,
    };
    // Dot product of two unit-norm vectors is bounded below by -dim,
    // well above -1e6.
    let very_negative = SearchParams {
        centroid_score_threshold: Some(-1e6),
        ..base
    };
    let a = search(&index, &query, base).unwrap();
    let b = search(&index, &query, very_negative).unwrap();
    assert_eq!(a, b);
}

/// Metamorphic (paper §4.2): centroid interaction is a *filter*, so
/// with `top_k` ≥ every candidate the doc_ids surfaced with a
/// shortlist are always a subset of those surfaced without one. The
/// existing `prop_search_large_shortlist_equals_none` only tests the
/// no-op end of the spectrum; this pins down the general filter
/// semantics.
#[hegel::test(test_cases = 20)]
fn prop_centroid_interaction_result_subset_of_unfiltered(tc: TestCase) {
    use docbert_plaid::{
        index::build_index,
        search::{SearchParams, search},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 2, 6, 5, 6));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    let query = tc.draw(unit_rows(dim, 2));
    let top_k = docs.len() + 4;
    let base = SearchParams {
        top_k,
        n_probe: params.k_centroids,
        n_candidate_docs: None,
        centroid_score_threshold: None,
    };
    let n_candidate_docs =
        tc.draw(gs::integers::<usize>().min_value(1).max_value(32));
    let filtered = SearchParams {
        n_candidate_docs: Some(n_candidate_docs),
        ..base
    };

    let unfiltered_results = search(&index, &query, base).unwrap();
    let filtered_results = search(&index, &query, filtered).unwrap();

    let unfiltered_ids: std::collections::HashSet<u64> =
        unfiltered_results.iter().map(|r| r.doc_id).collect();
    for r in &filtered_results {
        assert!(
            unfiltered_ids.contains(&r.doc_id),
            "shortlisted doc {} is not in the unfiltered result set {:?}",
            r.doc_id,
            unfiltered_ids,
        );
    }
}

/// Metamorphic (paper §4.3, Eq 5): centroid pruning is a *filter*, so
/// with `top_k` ≥ every reachable candidate the doc_ids surfaced
/// under `centroid_score_threshold=Some(_)` are always a subset of
/// those surfaced under `None`. Our existing
/// `prop_search_unreachable_threshold_equals_none` only pins down
/// the no-op floor — this covers any threshold including ones that
/// drop a real fraction of centroids.
#[hegel::test(test_cases = 20)]
fn prop_centroid_pruning_result_subset_of_unfiltered(tc: TestCase) {
    use docbert_plaid::{
        index::build_index,
        search::{SearchParams, search},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 2, 6, 5, 6));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    let query = tc.draw(unit_rows(dim, 2));
    let top_k = docs.len() + 4;
    let base = SearchParams {
        top_k,
        n_probe: params.k_centroids,
        n_candidate_docs: None,
        centroid_score_threshold: None,
    };
    // Dot products of unit-norm vectors lie in [-1, 1], so drawing
    // from that range gives us non-trivial thresholds that sometimes
    // prune and sometimes don't.
    let threshold: f32 = tc.draw(
        gs::floats::<f32>()
            .min_value(-1.0)
            .max_value(1.0)
            .allow_nan(false)
            .allow_infinity(false),
    );
    let pruned = SearchParams {
        centroid_score_threshold: Some(threshold),
        ..base
    };

    let unfiltered_results = search(&index, &query, base).unwrap();
    let pruned_results = search(&index, &query, pruned).unwrap();

    let unfiltered_ids: std::collections::HashSet<u64> =
        unfiltered_results.iter().map(|r| r.doc_id).collect();
    for r in &pruned_results {
        assert!(
            unfiltered_ids.contains(&r.doc_id),
            "pruned doc {} is not in the unfiltered result set {:?}",
            r.doc_id,
            unfiltered_ids,
        );
    }
}

/// Shape: `search(index, &[], _)` returns an empty vector for every
/// non-degenerate index. The early-return is easy to lose when
/// refactoring the cascade, so it's worth a property-level check
/// rather than a single example.
#[hegel::test(test_cases = 20)]
fn prop_search_empty_query_empty_result(tc: TestCase) {
    use docbert_plaid::{
        index::build_index,
        search::{SearchParams, search},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 1, 4, 4, 4));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    let sp = SearchParams {
        top_k: 5,
        n_probe: params.k_centroids,
        n_candidate_docs: None,
        centroid_score_threshold: None,
    };
    let results = search(&index, &[], sp).unwrap();
    assert!(results.is_empty());
}

/// Metamorphic: with `n_probe = num_centroids`, no shortlist, no
/// threshold, and `top_k` at least `num_documents`, every non-empty
/// doc must appear in the result set. This pins down the "no filter
/// path drops" contract — if any branch silently excludes a doc
/// when nothing should filter it, it surfaces here.
#[hegel::test(test_cases = 20)]
fn prop_search_full_probe_considers_every_nonempty_doc(tc: TestCase) {
    use docbert_plaid::{
        index::build_index,
        search::{SearchParams, search},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 1, 5, 4, 4));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    let query = tc.draw(unit_rows(dim, 1));
    let sp = SearchParams {
        top_k: docs.len() + 2,
        n_probe: params.k_centroids,
        n_candidate_docs: None,
        centroid_score_threshold: None,
    };
    let results = search(&index, &query, sp).unwrap();

    let expected: std::collections::HashSet<u64> = docs
        .iter()
        .filter(|d| d.n_tokens > 0)
        .map(|d| d.doc_id)
        .collect();
    let got: std::collections::HashSet<u64> =
        results.iter().map(|r| r.doc_id).collect();
    assert_eq!(
        got, expected,
        "full-probe search should see every non-empty doc",
    );
}

// ---------------------------------------------------------------------------
// persistence.rs
// ---------------------------------------------------------------------------

/// Round-trip: `load(save(index))` matches the original on params,
/// codec fields, doc_ids, doc_tokens, and every IVF posting list.
/// The IVF is rebuilt from scratch on load, so this also checks the
/// rebuild stays consistent with the on-disk encoding.
#[hegel::test(test_cases = 20)]
fn prop_save_load_roundtrip_exact(tc: TestCase) {
    use docbert_plaid::{
        index::build_index,
        persistence::{load, save},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 1, 5, 5, 4));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.plaid");
    save(&index, &path).unwrap();
    let loaded = load(&path).unwrap();

    assert_eq!(loaded.params.dim, index.params.dim);
    assert_eq!(loaded.params.nbits, index.params.nbits);
    assert_eq!(loaded.params.k_centroids, index.params.k_centroids);
    assert_eq!(loaded.doc_ids, index.doc_ids);
    assert_eq!(loaded.codec.centroids, index.codec.centroids);
    assert_eq!(loaded.codec.bucket_cutoffs, index.codec.bucket_cutoffs);
    assert_eq!(loaded.codec.bucket_weights, index.codec.bucket_weights);
    assert_eq!(loaded.doc_tokens, index.doc_tokens);
    assert_eq!(loaded.ivf.num_centroids(), index.ivf.num_centroids());
    for c in 0..index.ivf.num_centroids() {
        assert_eq!(
            loaded.ivf.docs_for_centroid(c),
            index.ivf.docs_for_centroid(c),
        );
    }
}

/// Round-trip + determinism: `search(index, q) ==
/// search(load(save(index)), q)` for any valid query and search
/// params. Combines structural persistence (task #38) with the search
/// cascade, so any bug that affects only the decoded-token path or
/// IVF traversal would still flip the result vectors here.
#[hegel::test(test_cases = 20)]
fn prop_save_load_search_identical(tc: TestCase) {
    use docbert_plaid::{
        index::build_index,
        persistence::{load, save},
        search::{SearchParams, search},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 1, 5, 5, 4));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.plaid");
    save(&index, &path).unwrap();
    let loaded = load(&path).unwrap();

    let query = tc.draw(unit_rows(dim, 2));
    let sp = SearchParams {
        top_k: docs.len(),
        n_probe: params.k_centroids,
        n_candidate_docs: None,
        centroid_score_threshold: None,
    };
    assert_eq!(
        search(&index, &query, sp).unwrap(),
        search(&loaded, &query, sp).unwrap(),
    );
}

// ---------------------------------------------------------------------------
// update.rs
// ---------------------------------------------------------------------------

/// Metamorphic: `apply_update` with empty deletions and upserts
/// preserves doc_ids, every encoded token, and each IVF posting list
/// exactly. The only legal no-op path; any drift here would imply
/// the update loop touches state unnecessarily.
#[hegel::test(test_cases = 20)]
fn prop_apply_update_empty_is_identity(tc: TestCase) {
    use docbert_plaid::{
        index::build_index,
        update::{IndexUpdate, apply_update},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 1, 5, 5, 4));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    let before_ids = index.doc_ids.clone();
    let before_tokens = index.doc_tokens.clone();
    let before_ivf: Vec<Vec<u32>> = (0..index.ivf.num_centroids())
        .map(|c| index.ivf.docs_for_centroid(c).to_vec())
        .collect();

    let updated = apply_update(
        index,
        IndexUpdate {
            deletions: &[],
            upserts: &[],
        },
    )
    .unwrap();

    assert_eq!(updated.doc_ids, before_ids);
    assert_eq!(updated.doc_tokens, before_tokens);
    for (c, expected) in before_ivf.iter().enumerate() {
        assert_eq!(updated.ivf.docs_for_centroid(c), expected.as_slice());
    }
}

/// Metamorphic: after `apply_update` deletes a subset of doc_ids,
/// those ids are gone from `updated.doc_ids`, and every surviving
/// doc retains its encoded tokens verbatim. Catches regressions that
/// accidentally drop or re-encode surviving documents.
#[hegel::test(test_cases = 20)]
fn prop_apply_update_deletions_removed(tc: TestCase) {
    use docbert_plaid::{
        index::build_index,
        update::{IndexUpdate, apply_update},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 2, 6, 5, 6));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    // Draw a boolean per existing doc_id deciding whether to delete
    // it. Ensures at least one survives so we can assert equality
    // on the preserved state.
    let survive_mask: Vec<bool> = (0..index.doc_ids.len())
        .map(|_| tc.draw(gs::booleans()))
        .collect();
    let mut deletions: Vec<u64> = Vec::new();
    let mut survivors_ids: Vec<u64> = Vec::new();
    let mut survivors_tokens: Vec<_> = Vec::new();
    for (i, &survives) in survive_mask.iter().enumerate() {
        if survives || i == 0 {
            survivors_ids.push(index.doc_ids[i]);
            survivors_tokens.push(index.doc_tokens[i].clone());
        } else {
            deletions.push(index.doc_ids[i]);
        }
    }

    let updated = apply_update(
        index,
        IndexUpdate {
            deletions: &deletions,
            upserts: &[],
        },
    )
    .unwrap();

    for id in &deletions {
        assert!(
            !updated.doc_ids.contains(id),
            "deleted doc_id {id} still present",
        );
    }
    assert_eq!(updated.doc_ids, survivors_ids);
    assert_eq!(updated.doc_tokens, survivors_tokens);
}

/// Metamorphic: `apply_update` with any mix of deletions and upserts
/// leaves the codec (centroids, cutoffs, weights, nbits, dim)
/// byte-for-byte unchanged. This is the whole point of incremental
/// updates — they never retrain the codec, which means callers can
/// reuse the original codec's decode tables across updates.
#[hegel::test(test_cases = 20)]
fn prop_apply_update_upsert_codec_unchanged(tc: TestCase) {
    use docbert_plaid::{
        index::{DocumentTokens, build_index},
        update::{IndexUpdate, apply_update},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 2, 5, 5, 6));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();
    let before_codec = index.codec.clone();

    let next_id = docs.iter().map(|d| d.doc_id).max().unwrap_or(0) + 1;
    let upsert_n = tc.draw(gs::integers::<usize>().min_value(0).max_value(2));
    let upserts: Vec<DocumentTokens> = (0..upsert_n)
        .map(|i| {
            let n = tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
            DocumentTokens {
                doc_id: next_id + i as u64,
                tokens: tc.draw(unit_rows(dim, n)),
                n_tokens: n,
            }
        })
        .collect();
    let deletions: Vec<u64> = if !docs.is_empty() && tc.draw(gs::booleans()) {
        vec![docs[0].doc_id]
    } else {
        Vec::new()
    };

    let updated = apply_update(
        index,
        IndexUpdate {
            deletions: &deletions,
            upserts: &upserts,
        },
    )
    .unwrap();

    assert_eq!(updated.codec.dim, before_codec.dim);
    assert_eq!(updated.codec.nbits, before_codec.nbits);
    assert_eq!(updated.codec.centroids, before_codec.centroids);
    assert_eq!(updated.codec.bucket_cutoffs, before_codec.bucket_cutoffs);
    assert_eq!(updated.codec.bucket_weights, before_codec.bucket_weights);
}

/// Shape: after any apply_update, the IVF invariants still hold —
/// every `(doc_idx, centroid_id)` is in the list for that centroid,
/// every list is strictly ascending (sorted + deduped), and the
/// number of lists matches `params.k_centroids`.
#[hegel::test(test_cases = 20)]
fn prop_apply_update_ivf_invariants_preserved(tc: TestCase) {
    use docbert_plaid::{
        index::{DocumentTokens, build_index},
        update::{IndexUpdate, apply_update},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 2, 5, 5, 6));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    let next_id = docs.iter().map(|d| d.doc_id).max().unwrap_or(0) + 1;
    let n_upsert = tc.draw(gs::integers::<usize>().min_value(0).max_value(3));
    let upserts: Vec<DocumentTokens> = (0..n_upsert)
        .map(|i| {
            let n = tc.draw(gs::integers::<usize>().min_value(0).max_value(3));
            let tokens = if n == 0 {
                Vec::new()
            } else {
                tc.draw(unit_rows(dim, n))
            };
            DocumentTokens {
                doc_id: next_id + i as u64,
                tokens,
                n_tokens: n,
            }
        })
        .collect();
    let deletions: Vec<u64> = docs
        .iter()
        .filter(|_| tc.draw(gs::booleans()))
        .map(|d| d.doc_id)
        .collect();

    let updated = apply_update(
        index,
        IndexUpdate {
            deletions: &deletions,
            upserts: &upserts,
        },
    )
    .unwrap();

    assert_eq!(updated.ivf.num_centroids(), params.k_centroids);
    for (doc_idx, encoded) in updated.doc_tokens.iter().enumerate() {
        for ev in encoded {
            let list = updated.ivf.docs_for_centroid(ev.centroid_id as usize);
            assert!(
                list.contains(&(doc_idx as u32)),
                "doc_idx={doc_idx} missing from centroid {} postings",
                ev.centroid_id,
            );
        }
    }
    for c in 0..updated.ivf.num_centroids() {
        let postings = updated.ivf.docs_for_centroid(c);
        assert!(
            postings.windows(2).all(|w| w[0] < w[1]),
            "centroid {c} postings not strictly ascending: {postings:?}",
        );
    }
}

/// Metamorphic: after building an index on `A` and upserting `B` via
/// `apply_update`, the final doc_id set equals `A ∪ B`, total token
/// count equals the combined sum, and every `(doc_idx, centroid_id)`
/// appears in the IVF. We don't assert byte-for-byte equality with
/// `build_index(A ∪ B)` — the rebuild would retrain on the combined
/// pool and pick a different codec — but the structural invariants
/// both paths must satisfy are checked.
#[hegel::test(test_cases = 20)]
fn prop_apply_update_full_upsert_matches_build_index(tc: TestCase) {
    use docbert_plaid::{
        index::build_index,
        update::{IndexUpdate, apply_update},
    };
    let dim = tc.draw(codec_dim());
    let a = tc.draw(corpus(dim, 1, 3, 4, 4));
    let a_total: usize = a.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, a_total));
    let index = build_index(&a, params).unwrap();

    let a_max_id = a.iter().map(|d| d.doc_id).max().unwrap();
    let mut b = tc.draw(corpus(dim, 1, 3, 4, 0));
    // Rewrite B's doc_ids to sit after A's so there's no overwrite.
    // Upserts with duplicate ids would exercise a different invariant
    // (prop_apply_update_upsert_codec_unchanged handles that cross).
    for (i, d) in b.iter_mut().enumerate() {
        d.doc_id = a_max_id + 1 + i as u64;
    }

    let updated = apply_update(
        index,
        IndexUpdate {
            deletions: &[],
            upserts: &b,
        },
    )
    .unwrap();

    let expected_ids: std::collections::HashSet<u64> =
        a.iter().chain(b.iter()).map(|d| d.doc_id).collect();
    let got_ids: std::collections::HashSet<u64> =
        updated.doc_ids.iter().copied().collect();
    assert_eq!(got_ids, expected_ids);

    let expected_tokens: usize =
        a.iter().chain(b.iter()).map(|d| d.n_tokens).sum();
    assert_eq!(updated.num_tokens(), expected_tokens);

    for (doc_idx, encoded) in updated.doc_tokens.iter().enumerate() {
        for ev in encoded {
            let list = updated.ivf.docs_for_centroid(ev.centroid_id as usize);
            assert!(list.contains(&(doc_idx as u32)));
        }
    }
}

/// Metamorphic (paper Eq 1): MaxSim `Σ_i max_j Q_i · D_j^T` has a
/// commutative `max_j` over doc tokens, so shuffling tokens within a
/// single doc must not change its search score. We run the check
/// against a *fixed* codec via `apply_update`: upsert the original
/// doc and a reordered copy side-by-side, then assert their scores
/// agree under the same query.
#[hegel::test(test_cases = 20)]
fn prop_doc_token_shuffle_preserves_score(tc: TestCase) {
    use docbert_plaid::{
        index::{DocumentTokens, build_index},
        search::{SearchParams, search},
        update::{IndexUpdate, apply_update},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 2, 5, 5, 6));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    // Pick the first non-empty doc as the shuffle target.
    let target_idx = match docs.iter().position(|d| d.n_tokens >= 2) {
        Some(i) => i,
        None => return, // Nothing to shuffle meaningfully.
    };
    let target = &docs[target_idx];
    let n = target.n_tokens;

    // Draw a permutation of 0..n via Fisher-Yates over hegel draws,
    // valid-by-construction.
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = tc.draw(gs::integers::<usize>().min_value(0).max_value(i));
        perm.swap(i, j);
    }
    let shuffled_tokens: Vec<f32> = perm
        .iter()
        .flat_map(|&i| target.tokens[i * dim..(i + 1) * dim].iter().copied())
        .collect();
    let shuffled_id = docs.iter().map(|d| d.doc_id).max().unwrap() + 1;
    let shuffled_doc = DocumentTokens {
        doc_id: shuffled_id,
        tokens: shuffled_tokens,
        n_tokens: n,
    };

    let updated = apply_update(
        index,
        IndexUpdate {
            deletions: &[],
            upserts: std::slice::from_ref(&shuffled_doc),
        },
    )
    .unwrap();

    let query = tc.draw(unit_rows(dim, 2));
    let sp = SearchParams {
        top_k: updated.num_documents(),
        n_probe: params.k_centroids,
        n_candidate_docs: None,
        centroid_score_threshold: None,
    };
    let results = search(&updated, &query, sp).unwrap();

    let original_score = results
        .iter()
        .find(|r| r.doc_id == target.doc_id)
        .map(|r| r.score);
    let shuffled_score = results
        .iter()
        .find(|r| r.doc_id == shuffled_id)
        .map(|r| r.score);
    match (original_score, shuffled_score) {
        (Some(a), Some(b)) => assert!(
            (a - b).abs() <= 1e-4,
            "shuffle changed score: original={a} shuffled={b}",
        ),
        (None, None) => {
            // Neither surfaced — the query doesn't match this doc
            // in either orientation, which is fine.
        }
        other => panic!(
            "only one orientation surfaced: original={:?} shuffled={:?}",
            other.0, other.1,
        ),
    }
}

// ---------------------------------------------------------------------------
// Paper Table 2 defaults
// ---------------------------------------------------------------------------

/// Invariant: for every top_k, `paper_defaults` returns params that are
/// valid to feed into `search` and preserve the paper's key promises —
/// pruning always on, nprobe positive, ndocs large enough for Stage 3
/// to return `top_k` results.
#[hegel::test(test_cases = 200)]
fn prop_paper_defaults_always_valid(tc: TestCase) {
    use docbert_plaid::search::SearchParams;
    let top_k: usize =
        tc.draw(gs::integers::<usize>().min_value(1).max_value(100_000));
    let p = SearchParams::paper_defaults(top_k);

    assert!(
        p.n_probe >= 1,
        "n_probe must be positive, got {}",
        p.n_probe
    );
    let t_cs = p.centroid_score_threshold.expect("pruning must be on");
    assert!(
        (0.0..=1.0).contains(&t_cs),
        "t_cs {t_cs} outside [0, 1]: unit-norm dot products can't exceed that range"
    );
    let ndocs = p.n_candidate_docs.expect("centroid interaction must be on");
    assert!(
        ndocs >= top_k.saturating_mul(4),
        "ndocs {ndocs} < 4 * top_k = {} — Stage 3 shortlist (ndocs/4) would drop below top_k",
        top_k * 4,
    );
    assert_eq!(p.top_k, top_k, "top_k not threaded through");
}

// ---------------------------------------------------------------------------
// GPU-side decode parity
// ---------------------------------------------------------------------------

/// Differential: the on-device (candle `index_select`) decode path
/// used by `search::batch_maxsim` must produce the same MaxSim scores
/// as the per-token CPU decode path (`codec.decode_vector`) followed
/// by a hand-rolled dot product. If the gather/reshape/narrow pipeline
/// diverges from the bit-unpacking reference even for one token, this
/// property catches it on the minimal corpus that triggers the drift.
///
/// This is the key correctness guard added alongside the move from a
/// CPU decode loop to candle `index_select` gathers in `batch_maxsim`.
#[hegel::test(test_cases = 30)]
fn prop_gpu_decode_scores_match_cpu_decode(tc: TestCase) {
    use docbert_plaid::{
        index::build_index,
        search::{SearchParams, search},
    };
    let dim = tc.draw(codec_dim());
    let docs = tc.draw(corpus(dim, 2, 5, 5, 8));
    let total_tokens: usize = docs.iter().map(|d| d.n_tokens).sum();
    let params = tc.draw(index_params(dim, 4, total_tokens));
    let index = build_index(&docs, params).unwrap();

    // Reach every candidate so the GPU decode is what's actually being
    // scored against the CPU reference — with no pruning or shortlist
    // filtering there's no scoring-stage divergence to hide behind.
    let query = tc.draw(unit_rows(dim, 2));
    let sp = SearchParams {
        top_k: docs.len(),
        n_probe: params.k_centroids,
        n_candidate_docs: None,
        centroid_score_threshold: None,
    };
    let got = search(&index, &query, sp).unwrap();

    for r in &got {
        let doc_idx = index
            .position_of(r.doc_id)
            .expect("returned doc must exist");
        // Reference: CPU decode every token, then compute MaxSim by hand.
        let decoded: Vec<Vec<f32>> = index.doc_tokens[doc_idx]
            .iter()
            .map(|ev| index.codec.decode_vector(ev).unwrap())
            .collect();
        let mut expected = 0.0f32;
        for q in query.chunks_exact(dim) {
            let best = decoded
                .iter()
                .map(|d| d.iter().zip(q).map(|(a, b)| a * b).sum::<f32>())
                .fold(f32::NEG_INFINITY, f32::max);
            if best.is_finite() {
                expected += best;
            }
        }
        assert!(
            (expected - r.score).abs() < 1e-4,
            "GPU decode score {} diverged from CPU-decode MaxSim {} for doc {}",
            r.score,
            expected,
            r.doc_id,
        );
    }
}

/// Monotonic: growing `top_k` should never make the algorithm
/// cheaper / more aggressive. That is, moving into a larger bucket
/// keeps `nprobe` and `ndocs` monotonically non-decreasing and
/// `t_cs` monotonically non-increasing, which is the direction the
/// paper's empirical tuning runs.
#[hegel::test(test_cases = 200)]
fn prop_paper_defaults_monotonic_in_top_k(tc: TestCase) {
    use docbert_plaid::search::SearchParams;
    let a: usize =
        tc.draw(gs::integers::<usize>().min_value(1).max_value(10_000));
    let delta: usize =
        tc.draw(gs::integers::<usize>().min_value(0).max_value(10_000));
    let b = a.saturating_add(delta);

    let pa = SearchParams::paper_defaults(a);
    let pb = SearchParams::paper_defaults(b);

    assert!(
        pb.n_probe >= pa.n_probe,
        "nprobe decreased when top_k grew: a={a} pa={} b={b} pb={}",
        pa.n_probe,
        pb.n_probe,
    );
    assert!(
        pb.n_candidate_docs >= pa.n_candidate_docs,
        "ndocs decreased when top_k grew: a={a} b={b}",
    );
    assert!(
        pb.centroid_score_threshold <= pa.centroid_score_threshold,
        "t_cs increased when top_k grew: a={a} b={b}",
    );
}
