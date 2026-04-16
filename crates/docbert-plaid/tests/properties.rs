//! Implementation-independent property checks for the PLAID primitives.
//!
//! Every test in this file asserts a mathematical invariant that any
//! correct implementation should satisfy, regardless of how it's coded
//! internally:
//!
//! - K-means: assignments deterministic; inertia (within-cluster sum of
//!   squared distances) is non-increasing across Lloyd iterations;
//!   centroids equal the mean of their assigned points after the
//!   M-step.
//! - Residual codec: encode→decode preserves the centroid identity;
//!   per-dim reconstruction error is bounded by half the local bucket
//!   width; total error decreases as `nbits` grows.
//! - Distance primitives: orthogonal vectors give dot product 0;
//!   identical unit vectors give dot product 1.
//!
//! These tests live as a separate integration test crate so they only
//! depend on `docbert-plaid`'s public API. If we ever swap the inner
//! loops for vectorized / GPU / other implementations, they should
//! still pass without modification.

use docbert_plaid::{
    codec::{ResidualCodec, train_quantizer},
    distance::{dot, squared_l2},
    kmeans::{assign_points, fit_with_init, update_centroids},
};
use rand::{Rng, SeedableRng, rngs::StdRng};

const DIM: usize = 32;

/// L2-normalize each row of a flat n × dim buffer in place.
fn normalize_rows(data: &mut [f32], dim: usize) {
    for row in data.chunks_exact_mut(dim) {
        let n = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for v in row {
            *v /= n;
        }
    }
}

fn random_unit_points(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data: Vec<f32> = (0..n * dim)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();
    normalize_rows(&mut data, dim);
    data
}

/// Within-cluster sum of squared distances given current centroids and
/// assignments. This is the quantity Lloyd's algorithm minimizes.
fn inertia(
    points: &[f32],
    centroids: &[f32],
    assignments: &[usize],
    dim: usize,
) -> f64 {
    let mut total = 0.0f64;
    for (point, &cluster) in points.chunks_exact(dim).zip(assignments) {
        let centroid = &centroids[cluster * dim..(cluster + 1) * dim];
        total += squared_l2(point, centroid) as f64;
    }
    total
}

#[test]
fn kmeans_inertia_is_non_increasing_across_lloyd_iterations() {
    let dim = DIM;
    let n = 500;
    let k = 8;
    let points = random_unit_points(0xA, n, dim);

    // Take the first k points as initial centroids.
    let mut centroids = points[..k * dim].to_vec();
    let mut prev_inertia = f64::INFINITY;

    for iter in 0..20 {
        let assignments = assign_points(&points, &centroids, dim);
        let current = inertia(&points, &centroids, &assignments, dim);
        assert!(
            current <= prev_inertia + 1e-5,
            "inertia rose from {prev_inertia} to {current} at iteration {iter}",
        );
        prev_inertia = current;
        centroids = update_centroids(&points, &assignments, &centroids, dim);
    }
}

#[test]
fn kmeans_assignment_is_deterministic_for_fixed_inputs() {
    let dim = DIM;
    let points = random_unit_points(0xB, 300, dim);
    let centroids = random_unit_points(0xC, 16, dim);

    let a = assign_points(&points, &centroids, dim);
    let b = assign_points(&points, &centroids, dim);
    assert_eq!(a, b);
}

#[test]
fn kmeans_update_centroids_yields_arithmetic_mean_per_cluster() {
    // Hand-built scenario: two 1-D clusters of three points each.
    let dim = 1;
    let points = vec![1.0f32, 2.0, 3.0, 100.0, 101.0, 102.0];
    let assignments = vec![0, 0, 0, 1, 1, 1];
    let previous = vec![0.0, 0.0];

    let updated = update_centroids(&points, &assignments, &previous, dim);

    assert!((updated[0] - 2.0).abs() < 1e-6);
    assert!((updated[1] - 101.0).abs() < 1e-6);
}

#[test]
fn kmeans_fit_eventually_stops_changing_assignments() {
    let dim = DIM;
    let points = random_unit_points(0xD, 200, dim);
    let initial = points[..4 * dim].to_vec();

    let centroids_a = fit_with_init(&points, &initial, dim, 50);
    // Re-running on the converged centroids must yield the same
    // assignments (same centroids, same data → no movement).
    let centroids_b = fit_with_init(&points, &centroids_a, dim, 50);

    let assign_a = assign_points(&points, &centroids_a, dim);
    let assign_b = assign_points(&points, &centroids_b, dim);
    assert_eq!(assign_a, assign_b);
}

#[test]
fn codec_encode_then_decode_chooses_the_nearest_centroid() {
    let dim = 4;
    let centroids = vec![
        0.0, 0.0, 0.0, 0.0, //
        10.0, 10.0, 10.0, 10.0, //
        -5.0, -5.0, -5.0, -5.0,
    ];
    let bucket_cutoffs = vec![-0.5, 0.0, 0.5];
    let bucket_weights = vec![-0.75, -0.25, 0.25, 0.75];
    let codec = ResidualCodec {
        nbits: 2,
        dim,
        centroids,
        bucket_cutoffs,
        bucket_weights,
    };
    codec.validate().unwrap();

    // A point much closer to centroid 1 should pick it.
    let v = [9.5f32, 10.1, 9.9, 10.0];
    let encoded = codec.encode_vector(&v);
    assert_eq!(encoded.centroid_id, 1);
}

#[test]
fn codec_reconstruction_error_decreases_as_nbits_grows() {
    // Build a codec from a fixed pool of residuals. With more bits we
    // get more, narrower buckets → strictly less average error per
    // dimension on the same training distribution.
    let pool: Vec<f32> = (0..4096).map(|i| (i as f32 / 2048.0) - 1.0).collect();

    let dim = 1;
    let centroids = vec![0.0f32];
    let mut errors = Vec::new();
    for nbits in [2u32, 4, 6] {
        let (cutoffs, weights) = train_quantizer(&pool, nbits);
        let codec = ResidualCodec {
            nbits,
            dim,
            centroids: centroids.clone(),
            bucket_cutoffs: cutoffs,
            bucket_weights: weights,
        };
        let mse: f64 = pool
            .iter()
            .map(|v| codec.reconstruction_error(&[*v]) as f64)
            .sum::<f64>()
            / pool.len() as f64;
        errors.push(mse);
    }

    for window in errors.windows(2) {
        assert!(
            window[0] >= window[1],
            "reconstruction MSE should not grow with more bits: {errors:?}",
        );
    }
}

#[test]
fn codec_round_trip_error_per_dim_is_bounded_by_widest_bucket_half_width() {
    let dim = 1;
    let centroids = vec![0.0f32];
    let cutoffs = vec![-0.5, 0.0, 0.5];
    let weights = vec![-0.75, -0.25, 0.25, 0.75];

    // Half of the widest interior bucket is 0.25; outermost open
    // buckets can decode to values up to 0.25 away from the cutoff
    // edge, so any input in [-0.5, 0.5] should reconstruct within 0.25
    // of itself. Inputs outside that range are quantized to the
    // farthest weight and incur larger error — this test is on the
    // bounded interior.
    let codec = ResidualCodec {
        nbits: 2,
        dim,
        centroids,
        bucket_cutoffs: cutoffs,
        bucket_weights: weights,
    };
    for v in [-0.5f32, -0.4, -0.1, 0.0, 0.1, 0.4, 0.49] {
        let encoded = codec.encode_vector(&[v]);
        let decoded = codec.decode_vector(&encoded);
        assert!(
            (decoded[0] - v).abs() <= 0.25,
            "v={v} decoded={d}",
            d = decoded[0]
        );
    }
}

#[test]
fn distance_primitives_satisfy_basic_invariants() {
    // Orthogonal unit vectors → dot 0.
    assert!(dot(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]).abs() < 1e-6);
    // Identical unit vectors → dot 1.
    assert!((dot(&[0.6, 0.8], &[0.6, 0.8]) - 1.0).abs() < 1e-6);
    // Antipodal unit vectors → dot −1.
    assert!((dot(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    // Squared L2 between unit vectors equals 2(1 − cos).
    let q = [0.6_f32, 0.8];
    let d = [0.8_f32, 0.6];
    let cos = dot(&q, &d);
    assert!((squared_l2(&q, &d) - 2.0 * (1.0 - cos)).abs() < 1e-5);
}
