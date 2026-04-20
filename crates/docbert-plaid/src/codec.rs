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
//! assuming the codec has already been trained. Bucket cutoffs and
//! weights are learned from a sample of residuals via
//! [`train_quantizer`].
//!
//! Storage layout: residual codes are LSB-first bit-packed at `nbits`
//! bits each. Supported widths are `{1, 2, 4, 8}` — enough to cover
//! every value the ColBERTv2/PLAID papers use in practice. For a 128-d
//! embedding at 2 bits, this is 32 bytes per token (vs. 128 bytes
//! unpacked), matching the paper's §4.5 packed-index layout.

use crate::{
    PlaidError,
    Result,
    distance::squared_l2,
    kmeans::nearest_centroid,
};

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

/// A single encoded token: a centroid reference plus a bit-packed
/// buffer of per-dim bucket codes.
///
/// The `codes` buffer holds `dim` quantization codes packed LSB-first
/// at `nbits` bits each. For the supported widths of 1, 2, 4, and 8
/// bits the buffer length is `(dim * nbits) / 8` (dim is expected to
/// be a multiple of `8/nbits` so code positions don't span bytes —
/// ColBERT dims are 128 or 96, which satisfies that constraint for
/// every supported `nbits`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedVector {
    /// Index of the coarse centroid this token was quantized against.
    pub centroid_id: u32,
    /// Bit-packed bucket codes. Use [`ResidualCodec::read_code`] or the
    /// codec's `decode_vector` to pull values out.
    pub codes: Vec<u8>,
}

/// Number of bytes required to pack `dim` codes at `nbits` bits each.
///
/// Panics if `nbits` is not one of the supported packed widths.
pub fn packed_bytes_per_vector(dim: usize, nbits: u32) -> usize {
    assert_supported_nbits(nbits);
    (dim * nbits as usize).div_ceil(8)
}

fn assert_supported_nbits(nbits: u32) {
    assert!(
        matches!(nbits, 1 | 2 | 4 | 8),
        "packed codec: nbits must be 1, 2, 4, or 8 (got {nbits})",
    );
}

/// Pack `unpacked` (one byte per code, values in `0 .. 2^nbits`) into
/// an LSB-first bit-packed buffer.
fn pack_codes(unpacked: &[u8], nbits: u32) -> Vec<u8> {
    assert_supported_nbits(nbits);
    if nbits == 8 {
        return unpacked.to_vec();
    }
    let codes_per_byte = 8 / nbits as usize;
    let mask: u8 = ((1u16 << nbits) - 1) as u8;
    let n_bytes = unpacked.len().div_ceil(codes_per_byte);
    let mut packed = vec![0u8; n_bytes];
    for (i, &code) in unpacked.iter().enumerate() {
        let byte_idx = i / codes_per_byte;
        let bit_off = (i % codes_per_byte) * nbits as usize;
        packed[byte_idx] |= (code & mask) << bit_off;
    }
    packed
}

/// Read the code at logical position `i` from a packed buffer.
pub fn read_code(packed: &[u8], i: usize, nbits: u32) -> u8 {
    assert_supported_nbits(nbits);
    if nbits == 8 {
        return packed[i];
    }
    let codes_per_byte = 8 / nbits as usize;
    let mask: u8 = ((1u16 << nbits) - 1) as u8;
    let byte_idx = i / codes_per_byte;
    let bit_off = (i % codes_per_byte) * nbits as usize;
    (packed[byte_idx] >> bit_off) & mask
}

/// Precomputed 256-entry lookup table mapping every possible packed
/// byte to the sequence of `bucket_weights` values it decodes to.
///
/// PLAID §4.5 notes that naive decompression pays a chain of
/// shift-mask-weight-lookup operations per residual dimension; a
/// one-off table that already composes the shift/mask with the weight
/// lookup reduces decoding to a single load per code position. For
/// `nbits=2` the whole table is `256 × 4` f32 = 4 KiB and easily stays
/// in L1.
pub struct DecodeTable {
    /// `weights[b * codes_per_byte + k]` = weight for the `k`-th code
    /// position inside packed byte value `b`.
    weights: Vec<f32>,
    codes_per_byte: usize,
    nbits: u32,
}

impl DecodeTable {
    /// Build the table for `codec`. Call once per search/decode batch
    /// and reuse across every encoded vector.
    pub fn new(codec: &ResidualCodec) -> Self {
        assert_supported_nbits(codec.nbits);
        let codes_per_byte = 8 / codec.nbits as usize;
        let entries = 256;
        let mut weights = vec![0.0f32; entries * codes_per_byte];
        let mask: u8 = ((1u16 << codec.nbits) - 1) as u8;
        for b in 0u16..256 {
            let byte = b as u8;
            for k in 0..codes_per_byte {
                let code = (byte >> (k * codec.nbits as usize)) & mask;
                weights[b as usize * codes_per_byte + k] =
                    codec.bucket_weights[code as usize];
            }
        }
        Self {
            weights,
            codes_per_byte,
            nbits: codec.nbits,
        }
    }

    /// Weights for the `codes_per_byte` positions inside packed byte
    /// `byte`. Length always equals `codes_per_byte`.
    pub fn weights_for(&self, byte: u8) -> &[f32] {
        let start = byte as usize * self.codes_per_byte;
        &self.weights[start..start + self.codes_per_byte]
    }

    /// Raw row-major `[256, codes_per_byte]` weights buffer.
    ///
    /// Exposed so the search path can upload the table once per query
    /// and decode residuals via batched `index_select` on the device,
    /// matching the GPU decompression kernel described in PLAID §4.5
    /// (one thread per packed byte).
    pub fn weights_flat(&self) -> &[f32] {
        &self.weights
    }

    /// Number of codes packed into one byte at this table's `nbits`.
    pub fn codes_per_byte(&self) -> usize {
        self.codes_per_byte
    }

    /// Bit-width the table was built for.
    pub fn nbits(&self) -> u32 {
        self.nbits
    }
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

    /// Number of packed bytes each encoded vector uses.
    pub fn packed_bytes(&self) -> usize {
        packed_bytes_per_vector(self.dim, self.nbits)
    }

    /// Validate internal shape invariants. Called automatically by
    /// encode/decode; exposed so callers loading a codec from disk can
    /// fail fast.
    ///
    /// # Errors
    ///
    /// Returns [`PlaidError::InvalidCodec`] with a description of the
    /// constraint that's violated.
    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 {
            return Err(PlaidError::InvalidCodec(
                "codec: dim must be positive".into(),
            ));
        }
        if !matches!(self.nbits, 1 | 2 | 4 | 8) {
            return Err(PlaidError::InvalidCodec(format!(
                "codec: nbits must be 1, 2, 4, or 8, got {}",
                self.nbits
            )));
        }
        if !self.centroids.len().is_multiple_of(self.dim)
            || self.centroids.is_empty()
        {
            return Err(PlaidError::InvalidCodec(format!(
                "codec: centroids length {} is not a positive multiple of dim {}",
                self.centroids.len(),
                self.dim,
            )));
        }
        let expected_buckets = self.num_buckets();
        if self.bucket_weights.len() != expected_buckets {
            return Err(PlaidError::InvalidCodec(format!(
                "codec: expected {} bucket_weights, got {}",
                expected_buckets,
                self.bucket_weights.len(),
            )));
        }
        if self.bucket_cutoffs.len() != expected_buckets - 1 {
            return Err(PlaidError::InvalidCodec(format!(
                "codec: expected {} bucket_cutoffs, got {}",
                expected_buckets - 1,
                self.bucket_cutoffs.len(),
            )));
        }
        for pair in self.bucket_cutoffs.windows(2) {
            if pair[0] > pair[1] || pair[0].is_nan() || pair[1].is_nan() {
                return Err(PlaidError::InvalidCodec(
                    "codec: bucket_cutoffs must be non-decreasing and finite"
                        .into(),
                ));
            }
        }
        Ok(())
    }

    /// Encode a single token embedding.
    ///
    /// Finds the nearest centroid, computes the residual, and quantizes
    /// each dimension against `bucket_cutoffs`.
    ///
    /// # Errors
    ///
    /// Returns [`PlaidError::InvalidCodec`] if this codec fails its
    /// shape invariants.
    ///
    /// # Panics
    ///
    /// Panics if `vector.len() != dim`.
    pub fn encode_vector(&self, vector: &[f32]) -> Result<EncodedVector> {
        self.validate()?;
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

        let unpacked: Vec<u8> = vector
            .iter()
            .zip(centroid_slice.iter())
            .map(|(v, c)| bucket_for_value(*v - *c, &self.bucket_cutoffs))
            .collect();
        let codes = pack_codes(&unpacked, self.nbits);

        Ok(EncodedVector {
            centroid_id: centroid_id as u32,
            codes,
        })
    }

    /// Encode every token in a flat `n × dim` buffer in one batched
    /// pass, returning the per-token centroid id and a flat `n × dim`
    /// code buffer.
    ///
    /// The expensive step — the nearest-centroid lookup — runs as a
    /// single matmul through [`crate::kmeans::assign_points`], which
    /// uses candle's GEMM (CPU or CUDA depending on build). The
    /// residual + bucket loop stays scalar because per-element
    /// `searchsorted` would otherwise require either a 3-D broadcast
    /// against the cutoffs table or a per-cutoff kernel launch — both
    /// less efficient than a tight Rust loop over the small cutoffs
    /// vector. Returning the codes flat avoids `n` `Vec<u8>`
    /// allocations; callers split into per-token slices as needed.
    ///
    /// # Errors
    ///
    /// Returns [`PlaidError::InvalidCodec`] if the codec fails its
    /// shape invariants, or [`PlaidError::Tensor`] if the
    /// matmul-driven nearest-centroid lookup fails.
    ///
    /// # Panics
    ///
    /// Panics if `tokens.len() % dim != 0` or if `tokens` is empty.
    ///
    /// [`PlaidError::InvalidCodec`]: crate::PlaidError::InvalidCodec
    /// [`PlaidError::Tensor`]: crate::PlaidError::Tensor
    pub fn batch_encode_tokens(
        &self,
        tokens: &[f32],
    ) -> Result<(Vec<u32>, Vec<u8>)> {
        self.validate()?;
        assert!(
            tokens.len().is_multiple_of(self.dim),
            "batch_encode_tokens: tokens length {} is not a multiple of dim {}",
            tokens.len(),
            self.dim,
        );
        let n = tokens.len() / self.dim;
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let assignments =
            crate::kmeans::assign_points(tokens, &self.centroids, self.dim)?;

        let packed_per_token = self.packed_bytes();
        let mut centroid_ids: Vec<u32> = Vec::with_capacity(n);
        let mut packed_codes: Vec<u8> =
            Vec::with_capacity(n * packed_per_token);
        let mut scratch: Vec<u8> = Vec::with_capacity(self.dim);
        for (token, &cluster) in
            tokens.chunks_exact(self.dim).zip(assignments.iter())
        {
            let centroid_slice =
                &self.centroids[cluster * self.dim..(cluster + 1) * self.dim];
            scratch.clear();
            for (t, c) in token.iter().zip(centroid_slice.iter()) {
                scratch.push(bucket_for_value(*t - *c, &self.bucket_cutoffs));
            }
            packed_codes.extend(pack_codes(&scratch, self.nbits));
            centroid_ids.push(cluster as u32);
        }
        Ok((centroid_ids, packed_codes))
    }

    /// Reconstruct an approximate token embedding from its codes.
    ///
    /// # Errors
    ///
    /// Returns [`PlaidError::InvalidCodec`] if this codec fails its
    /// shape invariants.
    ///
    /// # Panics
    ///
    /// Panics if `codes.len() != dim` or if any code is out of range.
    ///
    /// [`PlaidError::InvalidCodec`]: crate::PlaidError::InvalidCodec
    pub fn decode_vector(&self, encoded: &EncodedVector) -> Result<Vec<f32>> {
        let table = DecodeTable::new(self);
        self.decode_vector_with_table(encoded, &table)
    }

    /// Decode an encoded vector using a pre-built [`DecodeTable`].
    ///
    /// Callers that decode many vectors in a row (e.g., the search
    /// path's per-candidate decode loop) should build the table once
    /// outside the loop and reuse it here — each call then amounts to
    /// one table load per packed byte plus the centroid add.
    ///
    /// # Errors
    ///
    /// Returns [`PlaidError::InvalidCodec`] if this codec fails its
    /// shape invariants.
    ///
    /// [`PlaidError::InvalidCodec`]: crate::PlaidError::InvalidCodec
    pub fn decode_vector_with_table(
        &self,
        encoded: &EncodedVector,
        table: &DecodeTable,
    ) -> Result<Vec<f32>> {
        self.validate()?;
        assert_eq!(
            table.nbits, self.nbits,
            "decode_vector_with_table: table nbits {} != codec nbits {}",
            table.nbits, self.nbits,
        );
        let expected_bytes = self.packed_bytes();
        assert_eq!(
            encoded.codes.len(),
            expected_bytes,
            "decode_vector_with_table: expected {expected_bytes} packed bytes, got {}",
            encoded.codes.len(),
        );
        let centroid_id = encoded.centroid_id as usize;
        assert!(
            centroid_id < self.num_centroids(),
            "decode_vector_with_table: centroid_id {} out of range 0..{}",
            centroid_id,
            self.num_centroids(),
        );

        let centroid_slice = &self.centroids
            [centroid_id * self.dim..(centroid_id + 1) * self.dim];
        let codes_per_byte = table.codes_per_byte;

        let mut out = Vec::with_capacity(self.dim);
        for (byte_idx, &byte) in encoded.codes.iter().enumerate() {
            let weights = table.weights_for(byte);
            let base_dim = byte_idx * codes_per_byte;
            for (k, &w) in weights.iter().enumerate() {
                let dim_idx = base_dim + k;
                if dim_idx >= self.dim {
                    break;
                }
                out.push(centroid_slice[dim_idx] + w);
            }
        }
        Ok(out)
    }

    /// Return the squared L2 reconstruction error for `vector` under
    /// this codec. Useful as a lightweight codec-quality probe in tests
    /// and evaluation scripts.
    ///
    /// # Errors
    ///
    /// Returns [`PlaidError::InvalidCodec`] if this codec fails its
    /// shape invariants.
    ///
    /// [`PlaidError::InvalidCodec`]: crate::PlaidError::InvalidCodec
    pub fn reconstruction_error(&self, vector: &[f32]) -> Result<f32> {
        let encoded = self.encode_vector(vector)?;
        let decoded = self.decode_vector(&encoded)?;
        Ok(squared_l2(vector, &decoded))
    }
}

/// Learn bucket cutoffs and reconstruction weights from a sample of
/// residual values.
///
/// The returned tuple is `(bucket_cutoffs, bucket_weights)` with
/// `2^nbits - 1` cutoffs and `2^nbits` weights, ready to plug into a
/// [`ResidualCodec`]. Buckets are equal-quantile slices of the input:
/// cutoffs are picked at `i / (2^nbits)` quantile positions, and each
/// weight is the arithmetic mean of the residuals falling into that
/// bucket. This matches fast-plaid's "fit" step and keeps the codec
/// unbiased on the training distribution.
///
/// `residuals` does not need to be sorted; this function sorts a
/// locally-owned copy. NaN values are rejected up front since they
/// would poison the sort order.
///
/// # Panics
///
/// Panics if `residuals` is empty, if `nbits` is zero or exceeds 8, or
/// if `residuals` contains NaN.
pub fn train_quantizer(residuals: &[f32], nbits: u32) -> (Vec<f32>, Vec<f32>) {
    assert!(!residuals.is_empty(), "train_quantizer: empty sample");
    assert!(
        nbits > 0 && nbits <= 8,
        "train_quantizer: nbits must be in 1..=8, got {nbits}"
    );
    assert!(
        residuals.iter().all(|v| !v.is_nan()),
        "train_quantizer: residual sample contains NaN"
    );

    let num_buckets = 1usize << nbits;
    let n = residuals.len();

    let mut sorted = residuals.to_vec();
    // We already rejected NaN above, so `total_cmp` is a strict
    // ordering and doesn't need the `partial_cmp().unwrap()` dance.
    sorted.sort_by(|a, b| a.total_cmp(b));

    let bucket_bounds = |i: usize| -> (usize, usize) {
        let start = i * n / num_buckets;
        let end = if i + 1 == num_buckets {
            n
        } else {
            (i + 1) * n / num_buckets
        };
        (start, end)
    };

    let cutoffs: Vec<f32> = (1..num_buckets)
        .map(|i| sorted[i * n / num_buckets])
        .collect();

    let weights: Vec<f32> = (0..num_buckets)
        .map(|i| {
            let (start, end) = bucket_bounds(i);
            // If the bucket is empty (e.g., many duplicate values pushed
            // everyone into one slice), fall back to the nearest real
            // sample so the decoder still has a sensible value.
            if start == end {
                let idx = start.min(n - 1);
                sorted[idx]
            } else {
                let slice = &sorted[start..end];
                slice.iter().sum::<f32>() / slice.len() as f32
            }
        })
        .collect();

    (cutoffs, weights)
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
    fn decode_with_lookup_table_matches_scalar_decode() {
        // Paper §4.5: precompute the 2^8 possible unpack outputs for a
        // packed byte, decode via table lookup instead of bit ops. The
        // output must match the scalar reference bit-for-bit.
        for &nbits in &[1u32, 2, 4, 8] {
            let num_buckets = 1usize << nbits;
            let codec = ResidualCodec {
                nbits,
                dim: 16,
                centroids: (0..16).map(|i| i as f32 * 0.1).collect(),
                bucket_cutoffs: (1..num_buckets)
                    .map(|i| (i as f32 / num_buckets as f32) - 0.5)
                    .collect(),
                bucket_weights: (0..num_buckets)
                    .map(|i| (i as f32 + 0.5) / num_buckets as f32 - 0.5)
                    .collect(),
            };
            let input: Vec<f32> =
                (0..16).map(|i| i as f32 * 0.05 - 0.3).collect();
            let encoded = codec.encode_vector(&input).unwrap();

            let scalar = codec.decode_vector(&encoded).unwrap();
            let table = DecodeTable::new(&codec);
            let via_table =
                codec.decode_vector_with_table(&encoded, &table).unwrap();
            assert_eq!(scalar, via_table, "mismatch at nbits={nbits}");
        }
    }

    #[test]
    fn pack_then_read_code_recovers_every_input() {
        // Every nbits ∈ {1,2,4,8} should pack losslessly: reading each
        // position back from the packed buffer must return the
        // original value.
        for &nbits in &[1u32, 2, 4, 8] {
            let num_buckets = 1usize << nbits;
            // Cycle through 0..num_buckets so every code value lands
            // somewhere, plus a few more for good byte alignment.
            let unpacked: Vec<u8> = (0..32u8)
                .map(|i| (i as usize % num_buckets) as u8)
                .collect();
            let packed = pack_codes(&unpacked, nbits);
            for (i, &expected) in unpacked.iter().enumerate() {
                let got = read_code(&packed, i, nbits);
                assert_eq!(
                    got, expected,
                    "nbits={nbits} position {i}: got {got}, expected {expected}",
                );
            }
            // Byte count matches the advertised formula.
            assert_eq!(
                packed.len(),
                packed_bytes_per_vector(unpacked.len(), nbits),
            );
        }
    }

    #[test]
    fn encode_vector_produces_packed_codes_at_two_bits() {
        // Paper §4.5: ColBERTv2/PLAID pack `8/nbits` residual codes
        // per byte. For dim=8 at 2-bit, that's 4 codes per byte ⇒
        // 2 bytes of packed storage, not 8.
        let codec = ResidualCodec {
            nbits: 2,
            dim: 8,
            centroids: vec![0.0; 8],
            bucket_cutoffs: vec![-0.5, 0.0, 0.5],
            bucket_weights: vec![-0.75, -0.25, 0.25, 0.75],
        };
        let encoded = codec.encode_vector(&[0.1f32; 8]).unwrap();
        assert_eq!(encoded.codes.len(), 2);
    }

    #[test]
    fn encode_vector_produces_packed_codes_at_four_bits() {
        // dim=8 at 4-bit ⇒ 2 codes per byte ⇒ 4 bytes.
        let codec = ResidualCodec {
            nbits: 4,
            dim: 8,
            centroids: vec![0.0; 8],
            bucket_cutoffs: (0..15).map(|i| i as f32 / 15.0 - 0.5).collect(),
            bucket_weights: (0..16).map(|i| i as f32 / 16.0 - 0.5).collect(),
        };
        let encoded = codec.encode_vector(&[0.1f32; 8]).unwrap();
        assert_eq!(encoded.codes.len(), 4);
    }

    #[test]
    fn encode_decode_roundtrip_at_every_supported_nbits() {
        // Roundtripping through packing/unpacking must be lossless up
        // to the bucket quantisation. We exercise 1, 2, 4, and 8 bit
        // widths on a small residual so every branch of the packing
        // math gets hit.
        for nbits in [1u32, 2, 4, 8] {
            let num_buckets = 1usize << nbits;
            let bucket_cutoffs: Vec<f32> = (1..num_buckets)
                .map(|i| (i as f32 / num_buckets as f32) - 0.5)
                .collect();
            let bucket_weights: Vec<f32> = (0..num_buckets)
                .map(|i| (i as f32 + 0.5) / num_buckets as f32 - 0.5)
                .collect();
            let codec = ResidualCodec {
                nbits,
                dim: 8,
                centroids: vec![0.0; 8],
                bucket_cutoffs,
                bucket_weights,
            };
            let input = [-0.4f32, -0.1, 0.0, 0.25, 0.49, -0.25, 0.1, 0.3];
            let encoded = codec.encode_vector(&input).unwrap();
            let decoded = codec.decode_vector(&encoded).unwrap();
            let max_err = input
                .iter()
                .zip(decoded.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            // Each bucket covers at most `1 / num_buckets` of [−0.5, 0.5]
            // so reconstruction error per dim is bounded by half a
            // bucket width.
            let tolerance = 1.0 / num_buckets as f32;
            assert!(
                max_err <= tolerance,
                "nbits={nbits}: max_err={max_err}, tolerance={tolerance}",
            );
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
        let encoded = codec.encode_vector(&[9.0]).unwrap();
        assert_eq!(encoded.centroid_id, 1);
    }

    #[test]
    fn decode_inverts_a_known_encoding() {
        let codec = two_bit_1d_codec_with_centroids(vec![0.0]);
        // Residual −0.3 → bucket 1 (between −0.5 and 0.0) → weight −0.25.
        let encoded = codec.encode_vector(&[-0.3]).unwrap();
        assert_eq!(encoded.codes, vec![1]);
        let decoded = codec.decode_vector(&encoded).unwrap();
        assert_eq!(decoded, vec![-0.25]);
    }

    #[test]
    fn encode_then_decode_stays_inside_bucket_half_width() {
        // With cutoffs [-0.5, 0, 0.5] and weights at bucket midpoints,
        // reconstruction error per dim is at most 0.25 for any value in
        // the middle buckets.
        let codec = two_bit_1d_codec_with_centroids(vec![0.0]);
        for &value in &[-0.4f32, -0.1, 0.0, 0.2, 0.4] {
            let encoded = codec.encode_vector(&[value]).unwrap();
            let decoded = codec.decode_vector(&encoded).unwrap();
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
        assert_eq!(codec.reconstruction_error(&[-0.25]).unwrap(), 0.0);
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
    #[should_panic(expected = "packed bytes")]
    fn decode_panics_on_wrong_packed_code_length() {
        // With `dim=1` at 2 bits, a valid encoding is 1 packed byte.
        // Handing decode a 2-byte buffer should fail the shape check.
        let codec = two_bit_1d_codec_with_centroids(vec![0.0]);
        let bad = EncodedVector {
            centroid_id: 0,
            codes: vec![0, 0],
        };
        let _ = codec.decode_vector(&bad).unwrap();
    }

    #[test]
    fn train_quantizer_produces_right_number_of_cutoffs_and_weights() {
        let residuals: Vec<f32> =
            (0..1000).map(|i| i as f32 / 1000.0).collect();
        let (cutoffs, weights) = train_quantizer(&residuals, 2);
        assert_eq!(cutoffs.len(), 3);
        assert_eq!(weights.len(), 4);
    }

    #[test]
    fn train_quantizer_cutoffs_are_monotonic() {
        let residuals: Vec<f32> =
            (0..2048).map(|i| (i as f32 / 2048.0) - 0.5).collect();
        let (cutoffs, _) = train_quantizer(&residuals, 4);
        for pair in cutoffs.windows(2) {
            assert!(
                pair[0] <= pair[1],
                "cutoffs must be non-decreasing: {pair:?}"
            );
        }
    }

    #[test]
    fn train_quantizer_on_uniform_data_gives_quartile_cutoffs() {
        // Uniform samples in [0, 1000) with 2 bits → quartile cutoffs at
        // roughly 250, 500, 750.
        let residuals: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let (cutoffs, _) = train_quantizer(&residuals, 2);
        assert!((cutoffs[0] - 250.0).abs() < 1.0);
        assert!((cutoffs[1] - 500.0).abs() < 1.0);
        assert!((cutoffs[2] - 750.0).abs() < 1.0);
    }

    #[test]
    fn train_quantizer_weights_bracket_cutoffs() {
        // Each weight should fall within its bucket's [low, high] range.
        // For uniform data, this is straightforward to verify.
        let residuals: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let (cutoffs, weights) = train_quantizer(&residuals, 2);

        // Bucket 0: below cutoffs[0]
        assert!(weights[0] < cutoffs[0]);
        // Bucket 3: above cutoffs[2]
        assert!(weights[3] > cutoffs[2]);
        // Middle buckets fall inside their cutoff ranges.
        assert!(cutoffs[0] <= weights[1] && weights[1] < cutoffs[1]);
        assert!(cutoffs[1] <= weights[2] && weights[2] < cutoffs[2]);
    }

    #[test]
    #[should_panic(expected = "empty sample")]
    fn train_quantizer_panics_on_empty_sample() {
        let _ = train_quantizer(&[], 2);
    }

    #[test]
    #[should_panic(expected = "NaN")]
    fn train_quantizer_panics_on_nan() {
        let _ = train_quantizer(&[0.1, f32::NAN, 0.3], 2);
    }

    #[test]
    fn trained_codec_round_trips_within_reasonable_error() {
        // Train a 4-bit codec on synthetic residuals, then check the
        // reconstruction error on held-out samples is small relative to
        // the residual magnitude.
        let training: Vec<f32> =
            (0..2048).map(|i| (i as f32 / 2048.0) - 0.5).collect();
        let (cutoffs, weights) = train_quantizer(&training, 4);

        let codec = ResidualCodec {
            nbits: 4,
            dim: 1,
            centroids: vec![0.0],
            bucket_cutoffs: cutoffs,
            bucket_weights: weights,
        };
        codec.validate().unwrap();

        let mut max_err: f32 = 0.0;
        for v in &[-0.4f32, -0.1, 0.0, 0.25, 0.49] {
            let err = codec.reconstruction_error(&[*v]).unwrap().sqrt();
            max_err = max_err.max(err);
        }
        // 16 buckets spanning ~1.0 of range ⇒ each bucket ≈ 0.0625 wide,
        // so reconstruction error should sit well below 0.05.
        assert!(
            max_err < 0.05,
            "max reconstruction error {max_err} above tolerance"
        );
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
        let encoded = codec.encode_vector(&input).unwrap();
        let decoded = codec.decode_vector(&encoded).unwrap();
        for (d, i) in decoded.iter().zip(input.iter()) {
            assert!((d - i).abs() <= 0.25);
        }
    }
}
