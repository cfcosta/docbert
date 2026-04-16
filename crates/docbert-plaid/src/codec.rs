//! Residual quantization codec for PLAID.
//!
//! Once k-means has produced a set of coarse centroids, every token
//! embedding can be represented as:
//!
//! ```text
//! token ≈ centroid[centroid_id] + decode(residual_codes)
//! ```
//!
//! The residual is the element-wise difference between the token and its
//! nearest centroid. Each residual dimension is then placed into one of
//! `2^nbits` buckets according to a precomputed set of cutoffs, and the
//! bucket index (0…2ⁿ-1) is what we store on disk. At read time the
//! bucket index is mapped back to a reconstruction value via
//! `bucket_weights` and added to the centroid, yielding an approximate
//! copy of the original token.
//!
//! This module exposes the codec state and the encode/decode operations
//! assuming the codec has already been trained. A separate routine
//! (next TDD cycle) will learn bucket cutoffs and weights from a sample
//! of residuals.
//!
//! Storage layout: one `u8` per residual dimension. `nbits` may be less
//! than 8 (2 and 4 are the values fast-plaid's reference supports), but
//! for now we don't bit-pack on write. Bit packing can be layered in
//! later without changing the semantic API.

use crate::{distance::squared_l2, kmeans::nearest_centroid};

/// A trained residual-quantization codec.
///
/// Cutoffs partition the real line into `2^nbits` buckets. `bucket_cutoffs`
/// holds the `2^nbits - 1` internal boundaries in ascending order;
/// `bucket_weights` holds the `2^nbits` reconstruction values used when
/// decoding. Both are codec-wide: the same cutoffs/weights are applied to
/// every residual dimension of every token.
#[derive(Debug, Clone)]
pub struct ResidualCodec {
    /// Number of bits per residual dimension. Typically 2 or 4.
    pub nbits: u32,
    /// Dimensionality of the (original) token embeddings.
    pub dim: usize,
    /// Flat row-major coarse centroids, `k × dim`.
    pub centroids: Vec<f32>,
    /// `(2^nbits) - 1` ascending cutoff values for bucketing residuals.
    pub bucket_cutoffs: Vec<f32>,
    /// `2^nbits` reconstruction values, one per bucket.
    pub bucket_weights: Vec<f32>,
}

/// A single encoded token: a centroid reference plus per-dim bucket codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedVector {
    /// Index of the coarse centroid this token was quantized against.
    pub centroid_id: u32,
    /// One bucket code per residual dimension. Values are in
    /// `0 .. 2^nbits`.
    pub codes: Vec<u8>,
}

impl ResidualCodec {
    /// Number of buckets this codec partitions the residual space into.
    pub fn num_buckets(&self) -> usize {
        1usize << self.nbits
    }

    /// Number of coarse centroids stored.
    pub fn num_centroids(&self) -> usize {
        self.centroids.len() / self.dim
    }

    /// Validate internal shape invariants. Called automatically by
    /// encode/decode; exposed so callers loading a codec from disk can
    /// fail fast.
    pub fn validate(&self) -> Result<(), String> {
        if self.dim == 0 {
            return Err("codec: dim must be positive".into());
        }
        if self.nbits == 0 || self.nbits > 8 {
            return Err(format!(
                "codec: nbits must be in 1..=8, got {}",
                self.nbits
            ));
        }
        if !self.centroids.len().is_multiple_of(self.dim)
            || self.centroids.is_empty()
        {
            return Err(format!(
                "codec: centroids length {} is not a positive multiple of dim {}",
                self.centroids.len(),
                self.dim,
            ));
        }
        let expected_buckets = self.num_buckets();
        if self.bucket_weights.len() != expected_buckets {
            return Err(format!(
                "codec: expected {} bucket_weights, got {}",
                expected_buckets,
                self.bucket_weights.len(),
            ));
        }
        if self.bucket_cutoffs.len() != expected_buckets - 1 {
            return Err(format!(
                "codec: expected {} bucket_cutoffs, got {}",
                expected_buckets - 1,
                self.bucket_cutoffs.len(),
            ));
        }
        for pair in self.bucket_cutoffs.windows(2) {
            if pair[0] > pair[1] || pair[0].is_nan() || pair[1].is_nan() {
                return Err(
                    "codec: bucket_cutoffs must be non-decreasing and finite"
                        .into(),
                );
            }
        }
        Ok(())
    }

    /// Encode a single token embedding.
    ///
    /// Finds the nearest centroid, computes the residual, and quantizes
    /// each dimension against `bucket_cutoffs`.
    ///
    /// # Panics
    ///
    /// Panics if the codec is malformed or if `vector.len() != dim`.
    pub fn encode_vector(&self, vector: &[f32]) -> EncodedVector {
        self.validate().unwrap();
        assert_eq!(
            vector.len(),
            self.dim,
            "encode_vector: expected {} dims, got {}",
            self.dim,
            vector.len(),
        );

        let centroid_id = nearest_centroid(vector, &self.centroids, self.dim);
        let centroid_slice = &self.centroids
            [centroid_id * self.dim..(centroid_id + 1) * self.dim];

        let codes: Vec<u8> = vector
            .iter()
            .zip(centroid_slice.iter())
            .map(|(v, c)| bucket_for_value(*v - *c, &self.bucket_cutoffs))
            .collect();

        EncodedVector {
            centroid_id: centroid_id as u32,
            codes,
        }
    }

    /// Reconstruct an approximate token embedding from its codes.
    ///
    /// # Panics
    ///
    /// Panics if the codec is malformed, if `codes.len() != dim`, or if
    /// any code is out of range.
    pub fn decode_vector(&self, encoded: &EncodedVector) -> Vec<f32> {
        self.validate().unwrap();
        assert_eq!(
            encoded.codes.len(),
            self.dim,
            "decode_vector: expected {} codes, got {}",
            self.dim,
            encoded.codes.len(),
        );
        let centroid_id = encoded.centroid_id as usize;
        assert!(
            centroid_id < self.num_centroids(),
            "decode_vector: centroid_id {} out of range 0..{}",
            centroid_id,
            self.num_centroids(),
        );

        let centroid_slice = &self.centroids
            [centroid_id * self.dim..(centroid_id + 1) * self.dim];

        encoded
            .codes
            .iter()
            .zip(centroid_slice.iter())
            .map(|(code, c)| {
                let idx = *code as usize;
                assert!(
                    idx < self.bucket_weights.len(),
                    "decode_vector: code {} out of range 0..{}",
                    idx,
                    self.bucket_weights.len(),
                );
                *c + self.bucket_weights[idx]
            })
            .collect()
    }

    /// Return the squared L2 reconstruction error for `vector` under
    /// this codec. Useful as a lightweight codec-quality probe in tests
    /// and evaluation scripts.
    pub fn reconstruction_error(&self, vector: &[f32]) -> f32 {
        let encoded = self.encode_vector(vector);
        let decoded = self.decode_vector(&encoded);
        squared_l2(vector, &decoded)
    }
}

/// Return the index of the bucket that `value` falls into given a set of
/// ascending cutoffs.
///
/// Values strictly below the first cutoff go into bucket 0; values at or
/// above the last cutoff go into the top bucket (`cutoffs.len()`). This
/// matches the "lower-inclusive" convention used throughout PLAID.
fn bucket_for_value(value: f32, cutoffs: &[f32]) -> u8 {
    let mut idx = 0u8;
    for cutoff in cutoffs {
        if value >= *cutoff {
            idx += 1;
        } else {
            break;
        }
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal 2-bit codec over 1-D residuals with symmetric
    /// cutoffs around zero, handy for checking encode/decode without
    /// worrying about centroid geometry.
    fn two_bit_1d_codec_with_centroids(centroids: Vec<f32>) -> ResidualCodec {
        ResidualCodec {
            nbits: 2,
            dim: 1,
            centroids,
            bucket_cutoffs: vec![-0.5, 0.0, 0.5],
            bucket_weights: vec![-0.75, -0.25, 0.25, 0.75],
        }
    }

    #[test]
    fn bucket_for_value_places_below_first_cutoff_in_bucket_zero() {
        let cutoffs = [-0.5, 0.0, 0.5];
        assert_eq!(bucket_for_value(-1.0, &cutoffs), 0);
    }

    #[test]
    fn bucket_for_value_places_at_or_above_last_cutoff_in_top_bucket() {
        let cutoffs = [-0.5, 0.0, 0.5];
        assert_eq!(bucket_for_value(0.5, &cutoffs), 3);
        assert_eq!(bucket_for_value(9.9, &cutoffs), 3);
    }

    #[test]
    fn bucket_for_value_picks_intermediate_buckets() {
        let cutoffs = [-0.5, 0.0, 0.5];
        assert_eq!(bucket_for_value(-0.25, &cutoffs), 1);
        assert_eq!(bucket_for_value(0.25, &cutoffs), 2);
    }

    #[test]
    fn num_buckets_is_two_to_the_nbits() {
        let codec = two_bit_1d_codec_with_centroids(vec![0.0]);
        assert_eq!(codec.num_buckets(), 4);

        let mut four_bit = codec.clone();
        four_bit.nbits = 4;
        four_bit.bucket_cutoffs = (0..15).map(|i| i as f32 / 15.0).collect();
        four_bit.bucket_weights = (0..16).map(|i| i as f32).collect();
        assert_eq!(four_bit.num_buckets(), 16);
    }

    #[test]
    fn encode_picks_nearest_centroid() {
        // Two 1-D centroids at 0 and 10. Input 9.0 should snap to
        // centroid 1 (distance 1) rather than centroid 0 (distance 9).
        let codec = two_bit_1d_codec_with_centroids(vec![0.0, 10.0]);
        let encoded = codec.encode_vector(&[9.0]);
        assert_eq!(encoded.centroid_id, 1);
    }

    #[test]
    fn decode_inverts_a_known_encoding() {
        let codec = two_bit_1d_codec_with_centroids(vec![0.0]);
        // Residual −0.3 → bucket 1 (between −0.5 and 0.0) → weight −0.25.
        let encoded = codec.encode_vector(&[-0.3]);
        assert_eq!(encoded.codes, vec![1]);
        let decoded = codec.decode_vector(&encoded);
        assert_eq!(decoded, vec![-0.25]);
    }

    #[test]
    fn encode_then_decode_stays_inside_bucket_half_width() {
        // With cutoffs [-0.5, 0, 0.5] and weights at bucket midpoints,
        // reconstruction error per dim is at most 0.25 for any value in
        // the middle buckets.
        let codec = two_bit_1d_codec_with_centroids(vec![0.0]);
        for &value in &[-0.4f32, -0.1, 0.0, 0.2, 0.4] {
            let encoded = codec.encode_vector(&[value]);
            let decoded = codec.decode_vector(&encoded);
            assert!(
                (decoded[0] - value).abs() <= 0.25,
                "value {value} -> decoded {d}",
                d = decoded[0],
            );
        }
    }

    #[test]
    fn reconstruction_error_is_zero_when_residual_exactly_matches_weight() {
        let codec = two_bit_1d_codec_with_centroids(vec![0.0]);
        // Residual −0.25 lives in bucket 1, which decodes to −0.25.
        assert_eq!(codec.reconstruction_error(&[-0.25]), 0.0);
    }

    #[test]
    fn validate_rejects_wrong_number_of_cutoffs() {
        let mut codec = two_bit_1d_codec_with_centroids(vec![0.0]);
        codec.bucket_cutoffs.push(1.0); // now 4 cutoffs, expected 3
        assert!(codec.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_monotonic_cutoffs() {
        let mut codec = two_bit_1d_codec_with_centroids(vec![0.0]);
        codec.bucket_cutoffs = vec![0.5, 0.0, 0.5];
        assert!(codec.validate().is_err());
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn decode_panics_on_code_out_of_range() {
        let codec = two_bit_1d_codec_with_centroids(vec![0.0]);
        let bad = EncodedVector {
            centroid_id: 0,
            codes: vec![99],
        };
        let _ = codec.decode_vector(&bad);
    }

    #[test]
    fn encode_and_decode_roundtrip_multi_dim_stays_close() {
        // 2-D centroid at (1, 1). For an input (1.1, 0.7), the residual
        // is (0.1, -0.3). Both land in inner buckets and decode back to
        // values within 0.25 of the truth per dimension.
        let codec = ResidualCodec {
            nbits: 2,
            dim: 2,
            centroids: vec![1.0, 1.0],
            bucket_cutoffs: vec![-0.5, 0.0, 0.5],
            bucket_weights: vec![-0.75, -0.25, 0.25, 0.75],
        };
        let input = [1.1f32, 0.7];
        let encoded = codec.encode_vector(&input);
        let decoded = codec.decode_vector(&encoded);
        for (d, i) in decoded.iter().zip(input.iter()) {
            assert!((d - i).abs() <= 0.25);
        }
    }
}
