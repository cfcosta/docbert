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
}
