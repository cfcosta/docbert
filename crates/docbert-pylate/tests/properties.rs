//! Hegel property tests for the public surface of `docbert-pylate`.
//!
//! These cover three tiers:
//!
//! - **A** — `normalize_l2` (row-norm bounds, scale invariance, idempotence,
//!   shape preservation).
//! - **B** — `hierarchical_pooling` (`pool_factor ≤ 1` identity, shape
//!   contract, protected-token preservation, monotonicity in `pool_factor`,
//!   rank-error on non-3D input).
//! - **D** — serde round-trips for every `pub` type in `types.rs`.
//!
//! Tests that exercise non-`pub` helpers or `ColBERT::similarity` live in
//! `#[cfg(test)] mod hegel_tests` inside `src/model.rs`, because they need
//! `pub(crate)` access.

use candle_core::{Device, Tensor};
use docbert_pylate::{
    EncodeInput,
    EncodeOutput,
    RawSimilarityOutput,
    Similarities,
    SimilarityInput,
    hierarchical_pooling,
    normalize_l2,
};
use hegel::{TestCase, generators as gs};

const DEV: Device = Device::Cpu;

// ---------------------------------------------------------------------------
// Reusable composite generators
// ---------------------------------------------------------------------------

/// Draws `(batch, seq, dim, flat_row_major_data)` within small bounds so each
/// test case is cheap. The caller assembles the `Tensor`; that keeps the
/// shrunk counter-example readable (three ints and a `Vec<f32>`, not an opaque
/// `Tensor`).
///
/// Values can be arbitrarily small — including zero — which is the right
/// default for invariants like "row norm ≤ 1 + ε". Properties that break
/// near zero (idempotence, scale invariance) should use
/// [`tensor_3d_parts_nonzero`] instead.
#[hegel::composite]
fn tensor_3d_parts(tc: TestCase) -> (usize, usize, usize, Vec<f32>) {
    let batch: usize =
        tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
    let seq: usize =
        tc.draw(gs::integers::<usize>().min_value(1).max_value(12));
    let dim: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let n = batch * seq * dim;
    let data: Vec<f32> = tc.draw(
        gs::vecs(
            gs::floats::<f32>()
                .min_value(-10.0)
                .max_value(10.0)
                .allow_nan(false)
                .allow_infinity(false),
        )
        .min_size(n)
        .max_size(n),
    );
    (batch, seq, dim, data)
}

/// Like [`tensor_3d_parts`] but every element has `|v| ≥ 1e-2`, so every row
/// has norm well above the `1e-12` epsilon inside `normalize_l2`. Use this
/// when the property only holds outside the epsilon-regularised regime
/// (scale invariance, idempotence).
#[hegel::composite]
fn tensor_3d_parts_nonzero(tc: TestCase) -> (usize, usize, usize, Vec<f32>) {
    let batch: usize =
        tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
    let seq: usize =
        tc.draw(gs::integers::<usize>().min_value(1).max_value(12));
    let dim: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let n = batch * seq * dim;
    let data: Vec<f32> = tc.draw(
        gs::vecs(hegel::one_of!(
            gs::floats::<f32>()
                .min_value(-10.0)
                .max_value(-1e-2)
                .allow_nan(false)
                .allow_infinity(false),
            gs::floats::<f32>()
                .min_value(1e-2)
                .max_value(10.0)
                .allow_nan(false)
                .allow_infinity(false),
        ))
        .min_size(n)
        .max_size(n),
    );
    (batch, seq, dim, data)
}

fn mk_tensor(b: usize, s: usize, d: usize, data: Vec<f32>) -> Tensor {
    Tensor::from_vec(data, (b, s, d), &DEV).unwrap()
}

/// Row-wise squared L2 norm over the last dim of a 3-D tensor.
fn row_sq_norms(t: &Tensor) -> Vec<f32> {
    let v: Vec<Vec<Vec<f32>>> = t.to_vec3::<f32>().unwrap();
    v.into_iter()
        .flat_map(|batch| {
            batch
                .into_iter()
                .map(|row| row.iter().map(|x| x * x).sum::<f32>())
        })
        .collect()
}

// ---------------------------------------------------------------------------
// A — normalize_l2
// ---------------------------------------------------------------------------

/// A1+A5: every row's L2 norm is at most `1 + ε`, and the shape is preserved.
#[hegel::test(test_cases = 300)]
fn normalize_l2_bounds_and_shape(tc: TestCase) {
    let (b, s, d, data) = tc.draw(tensor_3d_parts());
    let x = mk_tensor(b, s, d, data);

    let y = normalize_l2(&x).unwrap();
    assert_eq!(y.dims(), x.dims(), "shape must be preserved");

    for n2 in row_sq_norms(&y) {
        assert!(n2 <= 1.0 + 1e-4, "row squared-norm {n2} exceeds 1 + ε",);
    }
}

/// A3: `normalize_l2(k·x) ≈ normalize_l2(x)` for `k > 0`, assuming inputs
/// live outside the `1e-12` epsilon regime. Tiny inputs are tested separately
/// — near zero the epsilon-regulariser dominates and the function is not
/// strictly scale-invariant by design.
#[hegel::test(test_cases = 200)]
fn normalize_l2_positive_scale_invariance(tc: TestCase) {
    let (b, s, d, data) = tc.draw(tensor_3d_parts_nonzero());
    // Keep `v * k` well above the 1e-12 epsilon for every element.
    let k: f32 = tc.draw(
        gs::floats::<f32>()
            .min_value(1e-2)
            .max_value(1e2)
            .allow_nan(false)
            .allow_infinity(false),
    );
    let x = mk_tensor(b, s, d, data.clone());
    let scaled = mk_tensor(b, s, d, data.iter().map(|v| v * k).collect());

    let y = normalize_l2(&x).unwrap().to_vec3::<f32>().unwrap();
    let y_scaled = normalize_l2(&scaled).unwrap().to_vec3::<f32>().unwrap();

    for (ra, rb) in y.iter().flatten().zip(y_scaled.iter().flatten()) {
        for (va, vb) in ra.iter().zip(rb.iter()) {
            assert!(
                (va - vb).abs() < 5e-3,
                "scale-invariance drift: {va} vs {vb} (k={k})",
            );
        }
    }
}

/// A4: `normalize_l2` is idempotent (applied twice ≈ applied once) whenever
/// the input is already outside the epsilon regime. Near-zero inputs are
/// deliberately excluded: the `+1e-12` regulariser means a tiny input is
/// lifted to a small (non-unit) output on the first pass and then pushed to
/// unit on the second, so idempotence does not hold there.
#[hegel::test(test_cases = 200)]
fn normalize_l2_idempotent(tc: TestCase) {
    let (b, s, d, data) = tc.draw(tensor_3d_parts_nonzero());
    let x = mk_tensor(b, s, d, data);

    let once = normalize_l2(&x).unwrap();
    let twice = normalize_l2(&once).unwrap();

    let a: Vec<Vec<Vec<f32>>> = once.to_vec3::<f32>().unwrap();
    let b_: Vec<Vec<Vec<f32>>> = twice.to_vec3::<f32>().unwrap();
    for (ra, rb) in a.iter().flatten().zip(b_.iter().flatten()) {
        for (va, vb) in ra.iter().zip(rb.iter()) {
            assert!((va - vb).abs() < 5e-4, "idempotence drift: {va} vs {vb}",);
        }
    }
}

// ---------------------------------------------------------------------------
// B — hierarchical_pooling
// ---------------------------------------------------------------------------

/// B1: `pool_factor ∈ {0, 1}` returns the input clone exactly.
#[hegel::test(test_cases = 200)]
fn hierarchical_pooling_trivial_is_identity(tc: TestCase) {
    let (b, s, d, data) = tc.draw(tensor_3d_parts());
    let pf: usize = tc.draw(gs::sampled_from(vec![0usize, 1]));

    let x = mk_tensor(b, s, d, data);
    let y = hierarchical_pooling(&x, pf).unwrap();

    assert_eq!(y.dims(), x.dims());
    let xv: Vec<Vec<Vec<f32>>> = x.to_vec3::<f32>().unwrap();
    let yv: Vec<Vec<Vec<f32>>> = y.to_vec3::<f32>().unwrap();
    assert_eq!(xv, yv, "pool_factor {pf} must return the input verbatim");
}

/// B2+B3: pooling preserves batch and embedding dims and never grows the
/// sequence dim. `pool_factor` can be larger than `seq_len` (the impl clamps
/// with `max(1)`), so we generate it up to `seq + 4` to cover that path.
#[hegel::test(test_cases = 200)]
fn hierarchical_pooling_shape_contract(tc: TestCase) {
    let (b, s, d, data) = tc.draw(tensor_3d_parts());
    let pf: usize =
        tc.draw(gs::integers::<usize>().min_value(1).max_value(s + 4));

    let x = mk_tensor(b, s, d, data);
    let y = hierarchical_pooling(&x, pf).unwrap();

    assert_eq!(y.dim(0).unwrap(), b, "batch dim must match");
    assert_eq!(y.dim(2).unwrap(), d, "embedding dim must match");
    assert!(
        y.dim(1).unwrap() <= s,
        "pooled seq-len {} must not exceed input {}",
        y.dim(1).unwrap(),
        s,
    );
}

/// B4: the first token of each document ("protected" embedding, CLS-style) is
/// preserved verbatim in the pooled output. The impl splits it off via
/// `narrow(0, 0, 1)` and re-appends it, so whatever row 0 looked like going
/// in must still be somewhere in the output (at the tail) unchanged.
#[hegel::test(test_cases = 150)]
fn hierarchical_pooling_protects_token_zero(tc: TestCase) {
    let (b, s, d, data) = tc.draw(tensor_3d_parts());
    let pf: usize =
        tc.draw(gs::integers::<usize>().min_value(2).max_value(s + 2));

    let x = mk_tensor(b, s, d, data);
    let y = hierarchical_pooling(&x, pf).unwrap();

    let xv: Vec<Vec<Vec<f32>>> = x.to_vec3::<f32>().unwrap();
    let yv: Vec<Vec<Vec<f32>>> = y.to_vec3::<f32>().unwrap();
    for (i, doc) in xv.iter().enumerate() {
        let protected = &doc[0];
        assert!(
            yv[i].iter().any(|row| row == protected),
            "doc {i} lost its protected token 0 after pooling",
        );
    }
}

/// B6: output seq-len is monotonically non-increasing in `pool_factor`. Larger
/// `pool_factor` means more aggressive clustering, never less.
#[hegel::test(test_cases = 150)]
fn hierarchical_pooling_seq_len_monotone_in_pf(tc: TestCase) {
    let (b, s, d, data) = tc.draw(tensor_3d_parts());
    let pf_small: usize =
        tc.draw(gs::integers::<usize>().min_value(1).max_value(s.max(1)));
    let pf_large: usize = tc.draw(
        gs::integers::<usize>()
            .min_value(pf_small)
            .max_value(pf_small + s + 2),
    );

    let x = mk_tensor(b, s, d, data);
    let len_small = hierarchical_pooling(&x, pf_small).unwrap().dim(1).unwrap();
    let len_large = hierarchical_pooling(&x, pf_large).unwrap().dim(1).unwrap();
    assert!(
        len_large <= len_small,
        "pf={pf_large} produced more rows ({len_large}) than pf={pf_small} ({len_small})",
    );
}

/// B7: 2-D and 4-D inputs are rejected with an error rather than silently
/// reshaped. The rank guard in `pooling.rs` is a correctness latch.
#[hegel::test(test_cases = 40)]
fn hierarchical_pooling_rejects_non_3d(tc: TestCase) {
    let rank: usize = tc.draw(gs::sampled_from(vec![2usize, 4]));
    let mut dims: Vec<usize> = Vec::with_capacity(rank);
    let mut n = 1usize;
    for _ in 0..rank {
        let s: usize =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
        dims.push(s);
        n *= s;
    }
    let data: Vec<f32> = tc.draw(
        gs::vecs(
            gs::floats::<f32>()
                .min_value(-1.0)
                .max_value(1.0)
                .allow_nan(false)
                .allow_infinity(false),
        )
        .min_size(n)
        .max_size(n),
    );
    let t = Tensor::from_vec(data, dims, &DEV).unwrap();
    assert!(hierarchical_pooling(&t, 2).is_err());
}

// ---------------------------------------------------------------------------
// D — serde round-trips for the public types
// ---------------------------------------------------------------------------

#[hegel::composite]
fn small_strings(tc: TestCase, max: usize) -> Vec<String> {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(0).max_value(max));
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(tc.draw(gs::text().max_size(16)));
    }
    v
}

#[hegel::composite]
fn small_3d_f32(tc: TestCase) -> Vec<Vec<Vec<f32>>> {
    let b: usize = tc.draw(gs::integers::<usize>().min_value(0).max_value(3));
    let s: usize = tc.draw(gs::integers::<usize>().min_value(0).max_value(6));
    let d: usize = tc.draw(gs::integers::<usize>().min_value(0).max_value(4));
    let finite = || {
        gs::floats::<f32>()
            .min_value(-1e3)
            .max_value(1e3)
            .allow_nan(false)
            .allow_infinity(false)
    };
    (0..b)
        .map(|_| {
            (0..s)
                .map(|_| tc.draw(gs::vecs(finite()).min_size(d).max_size(d)))
                .collect()
        })
        .collect()
}

#[hegel::test(test_cases = 100)]
fn similarity_input_round_trip(tc: TestCase) {
    let v = SimilarityInput {
        queries: tc.draw(small_strings(4)),
        documents: tc.draw(small_strings(4)),
    };
    let s = serde_json::to_string(&v).unwrap();
    let back: SimilarityInput = serde_json::from_str(&s).unwrap();
    assert_eq!(v.queries, back.queries);
    assert_eq!(v.documents, back.documents);
}

#[hegel::test(test_cases = 100)]
fn encode_input_round_trip(tc: TestCase) {
    let v = EncodeInput {
        sentences: tc.draw(small_strings(4)),
        batch_size: tc.draw(gs::optional(
            gs::integers::<usize>().min_value(1).max_value(64),
        )),
    };
    let s = serde_json::to_string(&v).unwrap();
    let back: EncodeInput = serde_json::from_str(&s).unwrap();
    assert_eq!(v.sentences, back.sentences);
    assert_eq!(v.batch_size, back.batch_size);
}

#[hegel::test(test_cases = 100)]
fn encode_output_round_trip(tc: TestCase) {
    let v = EncodeOutput {
        embeddings: tc.draw(small_3d_f32()),
    };
    let s = serde_json::to_string(&v).unwrap();
    let back: EncodeOutput = serde_json::from_str(&s).unwrap();
    assert_eq!(v.embeddings, back.embeddings);
}

#[hegel::test(test_cases = 100)]
fn similarities_round_trip(tc: TestCase) {
    let rows: usize =
        tc.draw(gs::integers::<usize>().min_value(0).max_value(4));
    let cols: usize =
        tc.draw(gs::integers::<usize>().min_value(0).max_value(4));
    let finite = || {
        gs::floats::<f32>()
            .min_value(-1e3)
            .max_value(1e3)
            .allow_nan(false)
            .allow_infinity(false)
    };
    let mut data = Vec::with_capacity(rows);
    for _ in 0..rows {
        data.push(tc.draw(gs::vecs(finite()).min_size(cols).max_size(cols)));
    }
    let v = Similarities { data };
    let s = serde_json::to_string(&v).unwrap();
    let back: Similarities = serde_json::from_str(&s).unwrap();
    assert_eq!(v.data, back.data);
}

#[hegel::test(test_cases = 60)]
fn raw_similarity_output_round_trip(tc: TestCase) {
    let nq: usize = tc.draw(gs::integers::<usize>().min_value(0).max_value(2));
    let nd: usize = tc.draw(gs::integers::<usize>().min_value(0).max_value(2));
    let ql: usize = tc.draw(gs::integers::<usize>().min_value(0).max_value(3));
    let dl: usize = tc.draw(gs::integers::<usize>().min_value(0).max_value(3));
    let finite = || {
        gs::floats::<f32>()
            .min_value(-1e3)
            .max_value(1e3)
            .allow_nan(false)
            .allow_infinity(false)
    };

    let mut matrix = Vec::with_capacity(nq);
    for _ in 0..nq {
        let mut q_block = Vec::with_capacity(nd);
        for _ in 0..nd {
            let mut q_d_block = Vec::with_capacity(ql);
            for _ in 0..ql {
                q_d_block.push(
                    tc.draw(gs::vecs(finite()).min_size(dl).max_size(dl)),
                );
            }
            q_block.push(q_d_block);
        }
        matrix.push(q_block);
    }

    let v = RawSimilarityOutput {
        similarity_matrix: matrix,
        query_tokens: (0..nq).map(|_| tc.draw(small_strings(3))).collect(),
        document_tokens: (0..nd).map(|_| tc.draw(small_strings(3))).collect(),
    };
    let s = serde_json::to_string(&v).unwrap();
    let back: RawSimilarityOutput = serde_json::from_str(&s).unwrap();
    assert_eq!(v.similarity_matrix, back.similarity_matrix);
    assert_eq!(v.query_tokens, back.query_tokens);
    assert_eq!(v.document_tokens, back.document_tokens);
}
