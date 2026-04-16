//! K-means clustering over token embeddings.
//!
//! PLAID treats the entire set of document token embeddings as a cloud of
//! points in ℝᵈ and summarises it with `k` centroids. Every stored token is
//! then quantized relative to its nearest centroid, and queries use the
//! centroids as the first-stage filter. The quality of that summary caps
//! the quality of the whole index, so k-means lives at the base of the
//! stack.
//!
//! This module builds the pieces bottom-up, one TDD cycle at a time. Right
//! now it exposes only the assignment step; the Lloyd update and the
//! full `fit` routine will follow.
//!
//! Points and centroids are both represented as flat row-major `&[f32]`
//! slices. A centroid block of length `k * dim` holds centroid `i` at
//! `[i*dim .. (i+1)*dim]`. This matches the storage format fast-plaid uses
//! on disk and keeps the hot path cache-friendly.

use crate::distance::squared_l2;

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

    let mut previous_assignments: Option<Vec<usize>> = None;
    for _ in 0..max_iters {
        let assignments = assign_points(points, &centroids, dim);
        if previous_assignments.as_deref() == Some(assignments.as_slice()) {
            break;
        }
        centroids = update_centroids(points, &assignments, &centroids, dim);
        previous_assignments = Some(assignments);
    }
    centroids
}

/// Run k-means with a deterministic initial centroid selection.
///
/// This convenience picks the first `k` rows of `points` as the starting
/// centroids and then delegates to [`fit_with_init`]. It's deterministic
/// (no RNG), useful for tests, and a sensible default when callers have
/// already shuffled their input. More sophisticated initialization
/// (k-means++, sampling) can be added later without changing this API.
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

    let initial = &points[..k * dim];
    fit_with_init(points, initial, dim, max_iters)
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
    points
        .chunks_exact(dim)
        .map(|point| nearest_centroid(point, centroids, dim))
        .collect()
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
    fn fit_uses_first_k_points_as_initial_centroids() {
        // With k equal to the number of points, the centroids should be
        // the points themselves and the assignment is the identity.
        let points = vec![1.0, 2.0, 3.0, 4.0];
        let fitted = fit(&points, 2, 2, 10);
        assert_eq!(fitted, points);
    }

    #[test]
    #[should_panic(expected = "need at least")]
    fn fit_panics_when_asked_for_more_clusters_than_points() {
        let points = vec![0.0, 0.0];
        let _ = fit(&points, 5, 2, 1);
    }
}
