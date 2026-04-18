//! K-means clustering over token embeddings.
//!
//! PLAID treats the entire set of document token embeddings as a cloud of
//! points in ℝᵈ and summarises it with `k` centroids. Every stored token is
//! then quantized relative to its nearest centroid, and queries use the
//! centroids as the first-stage filter. The quality of that summary caps
//! the quality of the whole index, so k-means lives at the base of the
//! stack.
//!
//! Points and centroids are both represented as flat row-major `&[f32]`
//! slices at the public boundary; internally the hot path
//! ([`assign_points`], [`fit_with_init`]) materialises them into
//! `candle_core::Tensor`s and computes the centroid distance matrix as
//! a matmul, split into row-chunks so the transient score tensors
//! stay within a fixed byte budget. With the `cuda` feature enabled
//! this matmul runs on the GPU; without it, candle's CPU `gemm`
//! implementation is still orders of magnitude faster than the
//! previous scalar nested loop.
//!
//! [`nearest_centroid`] (single point) and [`update_centroids`] (the
//! M-step) remain scalar — both are cheap relative to the assignment
//! step and the per-call tensor-allocation overhead would dominate.

use candle_core::Tensor;

use crate::{device::default_device, distance::squared_l2};

/// Index of the centroid nearest to `point` under squared L2 distance.
///
/// Ties are broken by preferring the earlier centroid, which keeps the
/// function deterministic regardless of input length.
///
/// # Panics
///
/// Panics if `centroids.len()` is not a positive multiple of `dim`, or if
/// `point.len() != dim`.
///
/// # Examples
///
/// ```
/// use docbert_plaid::kmeans::nearest_centroid;
///
/// // Two centroids in 2-D at (0, 0) and (10, 10). Point (9, 9) is closer
/// // to the second centroid.
/// let centroids = [0.0, 0.0, 10.0, 10.0];
/// assert_eq!(nearest_centroid(&[9.0, 9.0], &centroids, 2), 1);
/// ```
pub fn nearest_centroid(point: &[f32], centroids: &[f32], dim: usize) -> usize {
    assert_eq!(
        point.len(),
        dim,
        "nearest_centroid: point length {} does not match dim {}",
        point.len(),
        dim,
    );
    assert!(dim > 0, "nearest_centroid: dim must be positive");
    assert!(
        !centroids.is_empty() && centroids.len().is_multiple_of(dim),
        "nearest_centroid: centroids length {} is not a positive multiple of dim {}",
        centroids.len(),
        dim,
    );

    let mut best_idx = 0usize;
    let mut best_dist = f32::INFINITY;
    for (i, centroid) in centroids.chunks_exact(dim).enumerate() {
        let d = squared_l2(point, centroid);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

/// Recompute centroids as the mean of the points currently assigned to them.
///
/// This is the M-step of Lloyd's algorithm. If a cluster ends up with no
/// points assigned, its centroid is copied from `previous` unchanged —
/// this keeps empty clusters from collapsing to the origin and follows
/// what fast-plaid's index builder does.
///
/// `points` is row-major `n_points × dim`, `assignments` has length
/// `n_points` with values in `0..k`, and `previous` is row-major
/// `k × dim`. Returns a new `k × dim` buffer.
///
/// # Panics
///
/// Panics if any shape invariant is violated or if an assignment is
/// out of range.
pub fn update_centroids(
    points: &[f32],
    assignments: &[usize],
    previous: &[f32],
    dim: usize,
) -> Vec<f32> {
    assert!(dim > 0, "update_centroids: dim must be positive");
    assert!(
        points.len().is_multiple_of(dim),
        "update_centroids: points length {} is not a multiple of dim {}",
        points.len(),
        dim,
    );
    assert!(
        previous.len().is_multiple_of(dim) && !previous.is_empty(),
        "update_centroids: previous length {} is not a positive multiple of dim {}",
        previous.len(),
        dim,
    );
    let k = previous.len() / dim;
    assert_eq!(
        assignments.len(),
        points.len() / dim,
        "update_centroids: {} assignments for {} points",
        assignments.len(),
        points.len() / dim,
    );

    let mut sums = vec![0.0f32; k * dim];
    let mut counts = vec![0usize; k];

    for (point, &cluster) in points.chunks_exact(dim).zip(assignments.iter()) {
        assert!(
            cluster < k,
            "update_centroids: assignment {cluster} out of range 0..{k}"
        );
        let slot = &mut sums[cluster * dim..(cluster + 1) * dim];
        for (s, p) in slot.iter_mut().zip(point.iter()) {
            *s += *p;
        }
        counts[cluster] += 1;
    }

    let mut new_centroids = vec![0.0f32; k * dim];
    for (cluster, &count) in counts.iter().enumerate() {
        let start = cluster * dim;
        let end = start + dim;
        if count == 0 {
            new_centroids[start..end].copy_from_slice(&previous[start..end]);
            continue;
        }
        let inv = 1.0f32 / count as f32;
        for (out, s) in
            new_centroids[start..end].iter_mut().zip(&sums[start..end])
        {
            *out = s * inv;
        }
    }

    new_centroids
}

/// Run Lloyd's algorithm starting from an explicit set of initial centroids.
///
/// Iterates "assign points to nearest centroid, recompute centroids" up to
/// `max_iters` times, stopping early when no point changes cluster between
/// iterations. Returns the final centroids as a flat `k × dim` buffer.
///
/// Taking the initial centroids as an argument keeps the function
/// deterministic and trivial to test. Random initialization (e.g.
/// sampling `k` distinct points) is layered on top in [`fit`].
///
/// Internally we allocate the points tensor exactly once and reuse it
/// across iterations, so the only per-iteration cost is the centroids
/// upload (`k × dim` floats — negligible) plus the matmul itself.
///
/// # Panics
///
/// Panics on any shape violation between `points`, `initial`, and `dim`,
/// or if `dim == 0`.
pub fn fit_with_init(
    points: &[f32],
    initial: &[f32],
    dim: usize,
    max_iters: usize,
) -> Vec<f32> {
    assert!(dim > 0, "fit_with_init: dim must be positive");
    assert!(
        initial.len().is_multiple_of(dim) && !initial.is_empty(),
        "fit_with_init: initial centroids length {} is not a positive multiple of dim {}",
        initial.len(),
        dim,
    );
    assert!(
        points.len().is_multiple_of(dim),
        "fit_with_init: points length {} is not a multiple of dim {}",
        points.len(),
        dim,
    );

    let mut centroids = initial.to_vec();
    if points.is_empty() || max_iters == 0 {
        return centroids;
    }

    let n_points = points.len() / dim;
    let k = initial.len() / dim;
    let device = default_device();
    // Allocate the points tensor once and keep it on-device for the
    // entire Lloyd loop.
    let p_tensor = Tensor::from_slice(points, (n_points, dim), device)
        .expect("fit_with_init: failed to allocate points tensor");

    let mut previous_assignments: Option<Vec<usize>> = None;
    for _ in 0..max_iters {
        let c_tensor = Tensor::from_slice(&centroids, (k, dim), device)
            .expect("fit_with_init: failed to allocate centroids tensor");
        let assignments = assign_tensor(&p_tensor, &c_tensor)
            .expect("fit_with_init: assignment matmul failed");
        if previous_assignments.as_deref() == Some(assignments.as_slice()) {
            break;
        }
        centroids = update_centroids(points, &assignments, &centroids, dim);
        previous_assignments = Some(assignments);
    }
    centroids
}

/// Run k-means with deterministic farthest-first (Gonzalez) seeding.
///
/// The seeder picks the first point as the initial centroid, then for
/// each subsequent centroid picks the point whose squared distance to
/// the *nearest* already-chosen centroid is largest. This spreads the
/// initial centroids across the data cloud and avoids the failure mode
/// of the old "first k points" strategy, where a pre-sorted corpus
/// could drop every initial centroid into the same cluster and leave
/// Lloyd's M-step stuck in a degenerate local minimum.
///
/// Gonzalez's farthest-first is a deterministic approximation to
/// k-means++ (which samples proportional to `D²`). For the synthetic
/// and docbert-scale corpora this crate targets it's cheap — O(k · n · dim)
/// — and gives the bulk of the clustering-quality win without
/// depending on an RNG, which keeps fixtures reproducible.
///
/// # Panics
///
/// Panics if `k == 0`, if `dim == 0`, if `points` is empty, or if
/// `points` has fewer than `k` rows.
pub fn fit(points: &[f32], k: usize, dim: usize, max_iters: usize) -> Vec<f32> {
    assert!(k > 0, "fit: k must be positive");
    assert!(dim > 0, "fit: dim must be positive");
    assert!(
        points.len().is_multiple_of(dim),
        "fit: points length {} is not a multiple of dim {}",
        points.len(),
        dim,
    );
    let n_points = points.len() / dim;
    assert!(
        n_points >= k,
        "fit: need at least k={k} points, got {n_points}",
    );

    let initial = farthest_first_init(points, k, dim);
    fit_with_init(points, &initial, dim, max_iters)
}

/// Pick `k` initial centroids via deterministic farthest-first
/// seeding (Gonzalez's algorithm, a deterministic approximation to
/// k-means++).
///
/// The first centroid is row 0 of `points`. Each subsequent centroid
/// is the row whose squared L2 distance to the nearest already-picked
/// centroid is largest; ties break toward the earlier row index.
///
/// # Panics
///
/// Panics if `points` has fewer than `k` rows or if `dim == 0`.
pub fn farthest_first_init(points: &[f32], k: usize, dim: usize) -> Vec<f32> {
    assert!(k > 0, "farthest_first_init: k must be positive");
    assert!(dim > 0, "farthest_first_init: dim must be positive");
    assert!(
        points.len().is_multiple_of(dim) && !points.is_empty(),
        "farthest_first_init: points length {} is not a positive multiple of dim {}",
        points.len(),
        dim,
    );
    let n = points.len() / dim;
    assert!(
        n >= k,
        "farthest_first_init: need at least k={k} points, got {n}",
    );

    let mut centroids = Vec::with_capacity(k * dim);
    centroids.extend_from_slice(&points[..dim]);

    // Squared L2 to the nearest picked centroid for each point.
    let mut min_dists = vec![0.0f32; n];
    for (i, p) in points.chunks_exact(dim).enumerate() {
        min_dists[i] = squared_l2(p, &points[..dim]);
    }

    for _ in 1..k {
        let mut best_idx = 0usize;
        let mut best_dist = f32::NEG_INFINITY;
        for (i, &d) in min_dists.iter().enumerate() {
            if d > best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        let new_c = &points[best_idx * dim..(best_idx + 1) * dim];
        centroids.extend_from_slice(new_c);
        for (i, p) in points.chunks_exact(dim).enumerate() {
            let d = squared_l2(p, new_c);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }
    }

    centroids
}

/// Assign every point in `points` to the index of its nearest centroid.
///
/// `points` is a flat row-major `n_points × dim` buffer; `centroids` is a
/// flat row-major `k × dim` buffer. The returned vector has length
/// `n_points` and each entry is in `0..k`.
///
/// # Panics
///
/// Panics if `points.len() % dim != 0`, if `centroids.len() % dim != 0`,
/// if `centroids.len() == 0`, or if `dim == 0`.
///
/// # Examples
///
/// ```
/// use docbert_plaid::kmeans::assign_points;
///
/// let centroids = [0.0, 0.0, 10.0, 10.0];
/// let points = [0.1, 0.1, 9.5, 9.5, -1.0, 0.0];
/// assert_eq!(assign_points(&points, &centroids, 2), vec![0, 1, 0]);
/// ```
pub fn assign_points(
    points: &[f32],
    centroids: &[f32],
    dim: usize,
) -> Vec<usize> {
    assert!(dim > 0, "assign_points: dim must be positive");
    assert!(
        points.len().is_multiple_of(dim),
        "assign_points: points length {} is not a multiple of dim {}",
        points.len(),
        dim,
    );
    let n_points = points.len() / dim;
    if n_points == 0 {
        return Vec::new();
    }
    assert!(
        !centroids.is_empty() && centroids.len().is_multiple_of(dim),
        "assign_points: centroids length {} is not a positive multiple of dim {}",
        centroids.len(),
        dim,
    );
    let k = centroids.len() / dim;

    let device = default_device();
    let p = Tensor::from_slice(points, (n_points, dim), device)
        .expect("assign_points: failed to allocate points tensor");
    let c = Tensor::from_slice(centroids, (k, dim), device)
        .expect("assign_points: failed to allocate centroids tensor");
    assign_tensor(&p, &c).expect("assign_points: matmul failed")
}

/// Cap on the transient bytes materialised by a single assignment
/// matmul.
///
/// Each chunk produces three tensors of shape `[chunk_rows, k]`: the
/// matmul output, its affine-scaled twin, and the broadcast-added
/// score matrix. Sizing each of those to ~128 MiB keeps the working
/// set near ~384 MiB beside the resident points tensor — small enough
/// to fit on a 6 GiB GPU after the 3 GB pool upload, big enough that
/// cuBLAS kernels still amortise their launch overhead.
const ASSIGN_CHUNK_BYTES: usize = 128 * 1024 * 1024;

/// Tensor-based Lloyd E-step: returns the index of the nearest centroid
/// for every row of `p` against every row of `c`.
///
/// Delegates to [`assign_tensor_chunked`] with a chunk size derived
/// from [`ASSIGN_CHUNK_BYTES`]. Splitting this into a pair lets tests
/// drive the chunked path with tiny chunks without having to
/// synthesise gigabytes of input.
fn assign_tensor(
    p: &Tensor,
    c: &Tensor,
) -> Result<Vec<usize>, candle_core::Error> {
    let n = p.dim(0)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let k = c.dim(0)?.max(1);
    let bytes_per_row = k * std::mem::size_of::<f32>();
    let chunk_rows = (ASSIGN_CHUNK_BYTES / bytes_per_row).max(1).min(n);
    assign_tensor_chunked(p, c, chunk_rows)
}

/// Chunked implementation of the Lloyd E-step.
///
/// Argmin uses the formula `||c||² − 2·p·c` (the `||p||²` term is
/// constant for argmin per row, so we drop it). For typical ColBERT
/// centroids of ~unit norm, the dominant cost is the `[chunk, dim] ×
/// [dim, k]` matmul, which is exactly the GEMM candle is best at.
///
/// Point rows are sliced via [`Tensor::narrow`] — a zero-copy view —
/// so the caller's points tensor stays resident across chunks and
/// across Lloyd iterations. The centroid-side quantities (`c_t` and
/// `c_sq`) only depend on `c`, so they're computed once outside the
/// loop and reused.
fn assign_tensor_chunked(
    p: &Tensor,
    c: &Tensor,
    chunk_rows: usize,
) -> Result<Vec<usize>, candle_core::Error> {
    assert!(
        chunk_rows > 0,
        "assign_tensor_chunked: chunk_rows must be positive",
    );
    let n = p.dim(0)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let c_t = c.t()?;
    let c_sq = c.sqr()?.sum_keepdim(1)?.t()?; // [1, k]

    let mut out = Vec::with_capacity(n);
    let mut start = 0usize;
    while start < n {
        let len = chunk_rows.min(n - start);
        let p_chunk = p.narrow(0, start, len)?;
        let dot = p_chunk.matmul(&c_t)?; // [len, k]
        // scores[i, j] = ||c[j]||² - 2·p[i]·c[j], argmin gives nearest.
        let scores = dot.affine(-2.0, 0.0)?.broadcast_add(&c_sq)?;
        let argmin_u32: Vec<u32> = scores.argmin(1)?.to_vec1::<u32>()?;
        out.extend(argmin_u32.into_iter().map(|x| x as usize));
        start += len;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_centroid_returns_zero_for_single_centroid() {
        let centroids = [1.0, 2.0, 3.0];
        assert_eq!(nearest_centroid(&[0.0, 0.0, 0.0], &centroids, 3), 0);
    }

    #[test]
    fn nearest_centroid_picks_the_closest_of_many() {
        // Centroids at (0,0), (5,5), (10,10). Point (4,4) is closest to
        // the middle one at index 1.
        let centroids = [0.0, 0.0, 5.0, 5.0, 10.0, 10.0];
        assert_eq!(nearest_centroid(&[4.0, 4.0], &centroids, 2), 1);
    }

    #[test]
    fn nearest_centroid_breaks_ties_toward_earlier_index() {
        // Point (5,0) is exactly the same distance from (0,0) and (10,0).
        // The earlier centroid wins.
        let centroids = [0.0, 0.0, 10.0, 0.0];
        assert_eq!(nearest_centroid(&[5.0, 0.0], &centroids, 2), 0);
    }

    #[test]
    #[should_panic(expected = "point length")]
    fn nearest_centroid_panics_on_dim_mismatch() {
        let centroids = [0.0, 0.0];
        let _ = nearest_centroid(&[1.0], &centroids, 2);
    }

    #[test]
    #[should_panic(expected = "centroids length")]
    fn nearest_centroid_panics_on_ragged_centroid_block() {
        // 3 f32s with dim=2 can't split cleanly into centroids.
        let centroids = [0.0, 0.0, 1.0];
        let _ = nearest_centroid(&[0.0, 0.0], &centroids, 2);
    }

    #[test]
    fn assign_points_maps_each_point_to_its_cluster() {
        let centroids = [0.0, 0.0, 10.0, 10.0];
        let points = [0.1, 0.2, 9.0, 9.5, 0.0, -0.1, 10.1, 10.1];
        assert_eq!(assign_points(&points, &centroids, 2), vec![0, 1, 0, 1]);
    }

    #[test]
    fn assign_points_handles_empty_input() {
        let centroids = [0.0, 1.0];
        assert_eq!(assign_points(&[], &centroids, 2), Vec::<usize>::new());
    }

    #[test]
    #[should_panic(expected = "points length")]
    fn assign_points_panics_on_ragged_points() {
        let centroids = [0.0, 0.0];
        let _ = assign_points(&[1.0, 2.0, 3.0], &centroids, 2);
    }

    #[test]
    fn assign_tensor_chunked_agrees_with_unchunked() {
        // Build a dataset with >1 row and exercise chunk boundaries
        // that are 1, prime, round, off-by-one, and the full length.
        // The chunked and unchunked paths must agree row-for-row
        // regardless of chunk size — if they ever diverge, the
        // per-chunk argmin has drifted from the per-row argmin.
        let dim = 3;
        let k = 4;
        let centroids: Vec<f32> = vec![
            0.0, 0.0, 0.0, //
            10.0, 0.0, 0.0, //
            0.0, 10.0, 0.0, //
            10.0, 10.0, 0.0, //
        ];
        let n = 173;
        let mut points: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            let a = ((i * 37) % 11) as f32;
            let b = ((i * 53) % 13) as f32;
            points.extend_from_slice(&[a, b, 0.0]);
        }

        let device = default_device();
        let p = Tensor::from_slice(&points, (n, dim), device).unwrap();
        let c = Tensor::from_slice(&centroids, (k, dim), device).unwrap();

        let baseline = assign_tensor_chunked(&p, &c, n).unwrap();
        assert_eq!(baseline.len(), n);
        for chunk in [1usize, 7, 64, n - 1, n, n + 5] {
            let chunked = assign_tensor_chunked(&p, &c, chunk).unwrap();
            assert_eq!(
                chunked, baseline,
                "chunk_rows={chunk} must agree with the unchunked result",
            );
        }
    }

    // -- Lloyd M-step --

    #[test]
    fn update_centroids_averages_assigned_points() {
        // Two clusters. Cluster 0 gets (0,0) and (2,0); cluster 1 gets (10,0).
        let points = [0.0, 0.0, 2.0, 0.0, 10.0, 0.0];
        let assignments = [0, 0, 1];
        let previous = [0.0, 0.0, 0.0, 0.0];

        let updated = update_centroids(&points, &assignments, &previous, 2);

        assert_eq!(updated, vec![1.0, 0.0, 10.0, 0.0]);
    }

    #[test]
    fn update_centroids_keeps_previous_for_empty_clusters() {
        // Cluster 1 receives no points; its centroid must survive.
        let points = [0.0, 0.0, 2.0, 0.0];
        let assignments = [0, 0];
        let previous = [5.0, 5.0, 99.0, -99.0];

        let updated = update_centroids(&points, &assignments, &previous, 2);

        assert_eq!(updated[..2], [1.0, 0.0]);
        assert_eq!(updated[2..], [99.0, -99.0]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn update_centroids_panics_on_out_of_range_assignment() {
        let points = [0.0, 0.0];
        let assignments = [5];
        let previous = [0.0, 0.0];
        let _ = update_centroids(&points, &assignments, &previous, 2);
    }

    // -- Lloyd driver --

    #[test]
    fn fit_with_init_converges_on_well_separated_clusters() {
        // Two tight clusters at (0,0) and (10,10). Starting from slightly
        // off-centre initial centroids, one round of Lloyd's is enough to
        // snap them to the true cluster means.
        let points = vec![
            0.0, 0.0, 0.2, -0.1, -0.1, 0.1, // near (0,0)
            10.0, 10.0, 10.1, 9.9, 9.9, 10.1, // near (10,10)
        ];
        let initial = [0.5, 0.5, 9.5, 9.5];

        let fitted = fit_with_init(&points, &initial, 2, 20);

        assert_eq!(fitted.len(), 4);
        // Cluster 0 mean
        assert!((fitted[0] - 0.0333).abs() < 1e-3);
        assert!((fitted[1] - 0.0).abs() < 1e-3);
        // Cluster 1 mean
        assert!((fitted[2] - 10.0).abs() < 1e-3);
        assert!((fitted[3] - 10.0).abs() < 1e-3);
    }

    #[test]
    fn fit_with_init_is_idempotent_after_convergence() {
        // Running Lloyd again on an already-converged centroid set must
        // produce the same centroids (no further movement).
        let points = vec![0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 11.0, 0.0];
        let initial = [0.5, 0.0, 10.5, 0.0];

        let once = fit_with_init(&points, &initial, 2, 50);
        let twice = fit_with_init(&points, &once, 2, 50);

        assert_eq!(once, twice);
    }

    #[test]
    fn fit_with_init_returns_initial_when_no_iterations_allowed() {
        let points = vec![0.0, 0.0, 10.0, 10.0];
        let initial = [1.0, 1.0, 9.0, 9.0];
        assert_eq!(fit_with_init(&points, &initial, 2, 0), initial.to_vec());
    }

    #[test]
    fn fit_uses_farthest_first_seeding_to_spread_initial_centroids() {
        // Two tight clusters, one near origin and one near (10, 10).
        // With `max_iters = 0` no Lloyd step runs, so the returned
        // centroids ARE the initial seeds. Farthest-first must pick
        // one point from each cluster — the two returned centroids
        // should therefore sit far apart, not both near origin like
        // the old "first k points" seeding would give.
        let points = vec![
            0.0, 0.0, 0.1, 0.1, -0.1, 0.1, // cluster near origin
            10.0, 10.0, 10.1, 9.9, 9.9, 10.1, // cluster near (10, 10)
        ];
        let seeds = fit(&points, 2, 2, 0);
        let c0 = &seeds[0..2];
        let c1 = &seeds[2..4];
        let dist = squared_l2(c0, c1);
        assert!(
            dist > 100.0,
            "expected well-separated seeds, got {c0:?} and {c1:?} (dist²={dist})",
        );
    }

    #[test]
    fn fit_is_deterministic_across_identical_calls() {
        // Farthest-first seeding must still be reproducible — no RNG.
        let points = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let a = fit(&points, 3, 2, 10);
        let b = fit(&points, 3, 2, 10);
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic(expected = "need at least")]
    fn fit_panics_when_asked_for_more_clusters_than_points() {
        let points = vec![0.0, 0.0];
        let _ = fit(&points, 5, 2, 1);
    }
}
