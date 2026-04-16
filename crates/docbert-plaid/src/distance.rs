//! Vector distance primitives used throughout the index.
//!
//! PLAID's hot path is "given a query token, find the centroid with the
//! smallest squared L2 distance". Every other distance-shaped operation in
//! the codec, IVF, and search stages ultimately calls [`squared_l2`].
//!
//! We operate on plain `&[f32]` slices rather than any particular tensor
//! type so the primitives stay trivial to test, branch-predict, and
//! vectorize. Higher layers bridge the gap to candle tensors.

/// Dot product between two equal-length vectors.
///
/// This is the per-token similarity ColBERT and PLAID use to build
/// MaxSim scores. For L2-normalized embeddings it equals cosine
/// similarity; for non-unit vectors it additionally encodes magnitude,
/// which is usually what you want after residual decoding can pull a
/// token slightly off the unit sphere.
///
/// # Panics
///
/// Panics when `a.len() != b.len()`.
///
/// # Examples
///
/// ```
/// use docbert_plaid::distance::dot;
///
/// assert_eq!(dot(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
/// assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
/// assert_eq!(dot(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
/// ```
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "dot requires equal-length vectors (got {} and {})",
        a.len(),
        b.len(),
    );
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

/// Squared Euclidean distance between two equal-length vectors.
///
/// This deliberately computes the *squared* distance so we never pay for a
/// `sqrt` inside tight inner loops. For "which of these vectors is closest"
/// queries the ordering is identical.
///
/// # Panics
///
/// Panics when `a.len() != b.len()`. Distance is undefined across
/// different dimensionalities and callers should treat that as a bug
/// rather than silently padding or truncating.
///
/// # Examples
///
/// ```
/// use docbert_plaid::distance::squared_l2;
///
/// assert_eq!(squared_l2(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
/// assert_eq!(squared_l2(&[0.0, 0.0], &[3.0, 4.0]), 25.0);
/// ```
pub fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "squared_l2 requires equal-length vectors (got {} and {})",
        a.len(),
        b.len(),
    );
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_l2_is_zero_for_identical_vectors() {
        assert_eq!(squared_l2(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn squared_l2_matches_manual_calculation() {
        // (0-3)^2 + (0-4)^2 = 9 + 16 = 25
        assert_eq!(squared_l2(&[0.0, 0.0], &[3.0, 4.0]), 25.0);
    }

    #[test]
    fn squared_l2_is_symmetric() {
        let a = [1.0_f32, -2.0, 0.5];
        let b = [4.0_f32, 0.0, -1.5];
        assert_eq!(squared_l2(&a, &b), squared_l2(&b, &a));
    }

    #[test]
    fn squared_l2_handles_empty_vectors() {
        assert_eq!(squared_l2(&[], &[]), 0.0);
    }

    #[test]
    #[should_panic(expected = "equal-length vectors")]
    fn squared_l2_panics_on_length_mismatch() {
        let _ = squared_l2(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn dot_is_zero_for_orthogonal_vectors() {
        assert_eq!(dot(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]), 0.0);
    }

    #[test]
    fn dot_matches_manual_calculation() {
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn dot_is_symmetric() {
        let a = [1.0_f32, -2.0, 0.5];
        let b = [4.0_f32, 0.0, -1.5];
        assert_eq!(dot(&a, &b), dot(&b, &a));
    }

    #[test]
    fn dot_of_unit_vector_with_itself_is_one() {
        // A ColBERT token embedding is L2-normalized; cosine with
        // itself is 1.0.
        let v = [0.6_f32, 0.8];
        assert!((dot(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dot_handles_empty_vectors() {
        assert_eq!(dot(&[], &[]), 0.0);
    }

    #[test]
    #[should_panic(expected = "equal-length vectors")]
    fn dot_panics_on_length_mismatch() {
        let _ = dot(&[1.0, 2.0], &[1.0]);
    }
}
