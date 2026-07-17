use std::path::Path;

use half::bf16;
use heed::{
    Database,
    Env,
    byteorder::BigEndian,
    types::{Bytes, U64},
};

use crate::{
    error::{Error, Result},
    heed_env::{self, EMBEDDINGS_MAX_DBS},
};

/// Generous map size for the embeddings env. LMDB allocates a sparse
/// file at this size on disk; the actual on-disk usage tracks the
/// stored data, but the virtual address space stays mapped at this
/// ceiling, so picking a number large enough that operators rarely
/// hit it is the easiest path to "no surprises".
const MAP_SIZE: usize = 64 * 1024 * 1024 * 1024; // 64 GiB

const EMBEDDINGS_DB_NAME: &str = "embeddings";

/// Header size: 4 bytes token count + 4 bytes dimension.
const HEADER_SIZE: usize = 8;

/// Bytes per component in the current (bf16) value layout.
const BF16_BYTES: usize = 2;
/// Bytes per component in the legacy (f32) value layout.
const F32_BYTES: usize = 4;

/// Read the `(num_tokens, dimension, component_count)` header of a
/// stored value. Returns `None` when the blob is too short to carry
/// the 8-byte header at all.
fn parse_header(bytes: &[u8]) -> Option<(u32, u32, usize)> {
    if bytes.len() < HEADER_SIZE {
        return None;
    }
    let num_tokens =
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let dimension =
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let components = (num_tokens as usize) * (dimension as usize);
    Some((num_tokens, dimension, components))
}

fn parse_embedding_matrix(bytes: &[u8]) -> Result<Option<EmbeddingMatrix>> {
    let Some((num_tokens, dimension, components)) = parse_header(bytes) else {
        return Ok(None);
    };
    let body = &bytes[HEADER_SIZE..];

    // Byte-wise reads keep decoding independent of the mmap'd slice's
    // alignment.
    if body.len() == components * BF16_BYTES {
        let data = body
            .as_chunks::<BF16_BYTES>()
            .0
            .iter()
            .map(|c| bf16::from_bits(u16::from_le_bytes(*c)).to_f32())
            .collect();
        Ok(Some(EmbeddingMatrix {
            num_tokens,
            dimension,
            data,
        }))
    } else if body.len() == components * F32_BYTES {
        // Pre-1.0 layout: little-endian f32 components. Decoding it
        // was dropped for 1.0; `docbert clean` removes such rows.
        Err(Error::LegacyEmbeddings)
    } else {
        Ok(None)
    }
}

/// Validate a stored value's shape without decoding its payload.
fn parse_shape(bytes: &[u8]) -> Result<Option<(u32, u32)>> {
    let Some((num_tokens, dimension, components)) = parse_header(bytes) else {
        return Ok(None);
    };
    let body_len = bytes.len() - HEADER_SIZE;
    if body_len == components * BF16_BYTES {
        Ok(Some((num_tokens, dimension)))
    } else if body_len == components * F32_BYTES {
        Err(Error::LegacyEmbeddings)
    } else {
        Ok(None)
    }
}

/// `true` when a stored value uses the legacy pre-1.0 f32 layout.
fn is_legacy_f32_value(bytes: &[u8]) -> bool {
    parse_header(bytes).is_some_and(|(_, _, components)| {
        components > 0 && bytes.len() - HEADER_SIZE == components * F32_BYTES
    })
}

fn serialize_embedding_matrix(
    num_tokens: u32,
    dimension: u32,
    data: &[f32],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER_SIZE + data.len() * BF16_BYTES);
    bytes.extend_from_slice(&num_tokens.to_le_bytes());
    bytes.extend_from_slice(&dimension.to_le_bytes());
    for &value in data {
        bytes.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }
    bytes
}

/// Stores ColBERT token embedding matrices keyed by numeric document ID.
///
/// Each entry is packed like this:
/// - 4 bytes: token count `T` (`u32`, little-endian)
/// - 4 bytes: embedding dimension `D` (`u32`, little-endian)
/// - `T * D * 2` bytes: `bf16` values in row-major order
///
/// Components are stored as `bf16` because that's all the precision the
/// pipeline carries: the encoder trunk computes in bf16 on CUDA, so the
/// low 16 mantissa bits of the f32s handed to [`store`](Self::store)
/// are normalization noise below the model's own noise floor — and the
/// downstream consumer (the PLAID bridge) re-quantizes every token to
/// 2 bits per dimension anyway. Dropping them halves docbert's
/// dominant on-disk artifact.
///
/// Entries written by docbert releases before 1.0 carried `T * D * 4`
/// bytes of little-endian `f32` data instead. That layout is no longer
/// decoded: reads fail with [`Error::LegacyEmbeddings`], and `docbert
/// clean` drops such rows so the affected documents re-embed on the
/// next `docbert sync`.
///
/// Backed by an [LMDB](https://www.symas.com/lmdb) environment via the
/// [`heed`](https://docs.rs/heed) crate, which gives us multi-process
/// readers and writers on the same data dir — useful when several
/// `docbert mcp` / `docbert web` processes share a data dir. The
/// open path refuses files still in the redb format used before 1.0
/// with [`Error::LegacyDatabase`].
pub struct EmbeddingDb {
    env: Env,
    db: Database<U64<BigEndian>, Bytes>,
}

impl EmbeddingDb {
    /// Open or create an embeddings database at the given path.
    ///
    /// Fails with [`Error::LegacyDatabase`] if the file at `path` is
    /// still in the redb format written by docbert releases before
    /// 1.0; `docbert clean` resets such files.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// use docbert_core::EmbeddingDb;
    ///
    /// let db = EmbeddingDb::open(&tmp.path().join("embeddings.db")).unwrap();
    /// assert!(db.list_ids().unwrap().is_empty());
    /// ```
    pub fn open(path: &Path) -> Result<Self> {
        heed_env::ensure_not_redb(path)?;
        let env = heed_env::open_heed_env(path, MAP_SIZE, EMBEDDINGS_MAX_DBS)?;
        let mut wtxn = env.write_txn()?;
        let db: Database<U64<BigEndian>, Bytes> =
            env.create_database(&mut wtxn, Some(EMBEDDINGS_DB_NAME))?;
        wtxn.commit()?;
        Ok(Self { env, db })
    }

    /// Store an embedding matrix for a document.
    ///
    /// Overwrites any existing embedding for this `doc_id`. Components
    /// are rounded to `bf16` (round-to-nearest-even) on write, so
    /// [`load`](Self::load) returns the rounded values, not the exact
    /// input floats.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != num_tokens * dimension`.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// use docbert_core::EmbeddingDb;
    ///
    /// let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
    /// // Store 2 tokens x 3 dimensions = 6 floats
    /// db.store(42, 2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// assert!(db.load(42).unwrap().is_some());
    /// ```
    pub fn store(
        &self,
        doc_id: u64,
        num_tokens: u32,
        dimension: u32,
        data: &[f32],
    ) -> Result<()> {
        assert_eq!(
            data.len(),
            (num_tokens as usize) * (dimension as usize),
            "data length must equal num_tokens * dimension"
        );

        let bytes = serialize_embedding_matrix(num_tokens, dimension, data);

        let mut wtxn = self.env.write_txn()?;
        self.db.put(&mut wtxn, &doc_id, bytes.as_slice())?;
        wtxn.commit()?;
        Ok(())
    }

    /// Retrieve an embedding matrix for a document.
    ///
    /// Returns `None` if the document has no stored embedding or if
    /// the stored data is malformed. Fails with
    /// [`Error::LegacyEmbeddings`] if the entry is still in the
    /// pre-1.0 f32 layout.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// use docbert_core::EmbeddingDb;
    ///
    /// let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
    /// assert!(db.load(999).unwrap().is_none()); // not found
    ///
    /// db.store(42, 2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let matrix = db.load(42).unwrap().unwrap();
    /// assert_eq!(matrix.num_tokens, 2);
    /// assert_eq!(matrix.dimension, 3);
    /// assert_eq!(matrix.data.len(), 6);
    /// ```
    pub fn load(&self, doc_id: u64) -> Result<Option<EmbeddingMatrix>> {
        let rtxn = self.env.read_txn()?;
        let Some(bytes) = self.db.get(&rtxn, &doc_id)? else {
            return Ok(None);
        };
        parse_embedding_matrix(bytes)
    }

    /// Remove an embedding entry. Returns `true` if it existed.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// use docbert_core::EmbeddingDb;
    ///
    /// let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
    /// db.store(42, 1, 2, &[1.0, 2.0]).unwrap();
    /// assert!(db.remove(42).unwrap());
    /// assert!(!db.remove(42).unwrap()); // already gone
    /// ```
    pub fn remove(&self, doc_id: u64) -> Result<bool> {
        let mut wtxn = self.env.write_txn()?;
        let removed = self.db.delete(&mut wtxn, &doc_id)?;
        wtxn.commit()?;
        Ok(removed)
    }

    /// Remove multiple embedding entries in a single transaction.
    ///
    /// More efficient than calling [`remove`](Self::remove) in a loop.
    /// Silently skips IDs that do not exist.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// use docbert_core::EmbeddingDb;
    ///
    /// let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
    /// db.store(1, 1, 2, &[1.0, 2.0]).unwrap();
    /// db.store(2, 1, 2, &[3.0, 4.0]).unwrap();
    /// db.batch_remove(&[1, 2, 999]).unwrap(); // 999 is silently ignored
    /// assert!(db.list_ids().unwrap().is_empty());
    /// ```
    pub fn batch_remove(&self, doc_ids: &[u64]) -> Result<()> {
        if doc_ids.is_empty() {
            return Ok(());
        }
        let mut wtxn = self.env.write_txn()?;
        for &doc_id in doc_ids {
            self.db.delete(&mut wtxn, &doc_id)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Store multiple embedding matrices in a single transaction.
    ///
    /// Each entry is `(doc_id, num_tokens, dimension, data)`.
    /// More efficient than calling [`store`](Self::store) in a loop.
    /// Like [`store`](Self::store), components are rounded to `bf16`
    /// on write.
    ///
    /// # Panics
    ///
    /// Panics if any entry's `data.len() != num_tokens * dimension`.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// use docbert_core::EmbeddingDb;
    ///
    /// let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
    /// db.batch_store(&[
    ///     (1, 1, 2, vec![1.0, 2.0]),
    ///     (2, 1, 2, vec![3.0, 4.0]),
    /// ]).unwrap();
    /// assert_eq!(db.list_ids().unwrap().len(), 2);
    /// ```
    pub fn batch_store(
        &self,
        entries: &[(u64, u32, u32, Vec<f32>)],
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut wtxn = self.env.write_txn()?;
        for (doc_id, num_tokens, dimension, data) in entries {
            assert_eq!(
                data.len(),
                (*num_tokens as usize) * (*dimension as usize),
                "data length must equal num_tokens * dimension"
            );

            let bytes =
                serialize_embedding_matrix(*num_tokens, *dimension, data);
            self.db.put(&mut wtxn, doc_id, bytes.as_slice())?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Load multiple embedding matrices in a single transaction.
    ///
    /// Returns a vector of `(doc_id, Option<EmbeddingMatrix>)` preserving input
    /// order. Missing embeddings return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// use docbert_core::EmbeddingDb;
    ///
    /// let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
    /// db.store(10, 1, 2, &[1.0, 2.0]).unwrap();
    /// db.store(20, 1, 2, &[3.0, 4.0]).unwrap();
    ///
    /// let results = db.batch_load(&[20, 999, 10]).unwrap();
    /// assert!(results[0].1.is_some());  // doc 20
    /// assert!(results[1].1.is_none());  // doc 999 missing
    /// assert!(results[2].1.is_some());  // doc 10
    /// ```
    pub fn batch_load(
        &self,
        doc_ids: &[u64],
    ) -> Result<Vec<(u64, Option<EmbeddingMatrix>)>> {
        if doc_ids.is_empty() {
            return Ok(Vec::new());
        }

        let rtxn = self.env.read_txn()?;
        let mut results = Vec::with_capacity(doc_ids.len());
        for &doc_id in doc_ids {
            let matrix = match self.db.get(&rtxn, &doc_id)? {
                Some(bytes) => parse_embedding_matrix(bytes)?,
                None => None,
            };
            results.push((doc_id, matrix));
        }
        Ok(results)
    }

    /// List all stored document IDs.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// use docbert_core::EmbeddingDb;
    ///
    /// let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
    /// db.store(10, 1, 2, &[1.0, 2.0]).unwrap();
    /// db.store(20, 1, 2, &[3.0, 4.0]).unwrap();
    /// let mut ids = db.list_ids().unwrap();
    /// ids.sort();
    /// assert_eq!(ids, vec![10, 20]);
    /// ```
    pub fn list_ids(&self) -> Result<Vec<u64>> {
        let rtxn = self.env.read_txn()?;
        let mut result = Vec::new();
        for entry in self.db.iter(&rtxn)? {
            let (doc_id, _) = entry?;
            result.push(doc_id);
        }
        Ok(result)
    }

    /// List `(doc_id, num_tokens, dimension)` triples for every valid
    /// embedding entry.
    ///
    /// Reads only the 8-byte header of each stored value — enough to
    /// know an entry's shape without decoding its token payload into
    /// RAM. The embedding bridge uses this to size a pooled token
    /// buffer up front and then stream each matrix straight into that
    /// buffer, which keeps peak RSS near a single copy of the corpus
    /// instead of two.
    ///
    /// Malformed entries (header disagrees with the stored blob's
    /// length) are skipped silently, matching [`Self::load`]'s
    /// "return `None` on garbage" behaviour; entries still in the
    /// pre-1.0 f32 layout fail with [`Error::LegacyEmbeddings`].
    pub fn list_shapes(&self) -> Result<Vec<(u64, u32, u32)>> {
        let rtxn = self.env.read_txn()?;
        let mut result = Vec::new();
        for entry in self.db.iter(&rtxn)? {
            let (doc_id, bytes) = entry?;
            if let Some((num_tokens, dimension)) = parse_shape(bytes)? {
                result.push((doc_id, num_tokens, dimension));
            }
        }
        Ok(result)
    }

    /// List the IDs of entries still stored in the pre-1.0 f32 layout.
    ///
    /// Reads only headers and value lengths, so it stays cheap even on
    /// large stores. `docbert clean` uses this to find and drop the
    /// rows this version no longer decodes.
    pub fn list_legacy_ids(&self) -> Result<Vec<u64>> {
        let rtxn = self.env.read_txn()?;
        let mut result = Vec::new();
        for entry in self.db.iter(&rtxn)? {
            let (doc_id, bytes) = entry?;
            if is_legacy_f32_value(bytes) {
                result.push(doc_id);
            }
        }
        Ok(result)
    }

    /// Test-only escape hatch: write raw bytes for a `doc_id` so tests
    /// can exercise the read paths against malformed or legacy
    /// payloads without going through the public store/serialize path.
    /// Hidden from docs; not part of the supported API.
    #[doc(hidden)]
    pub fn insert_raw(&self, doc_id: u64, bytes: &[u8]) -> Result<()> {
        let mut wtxn = self.env.write_txn()?;
        self.db.put(&mut wtxn, &doc_id, bytes)?;
        wtxn.commit()?;
        Ok(())
    }
}

impl std::fmt::Debug for EmbeddingDb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingDb").finish_non_exhaustive()
    }
}

/// ColBERT embedding matrix loaded from the database.
///
/// The data lives in a flat `Vec<f32>` in row-major order. Use
/// [`token_embedding`](Self::token_embedding) when you want one token vector.
/// Components carry bf16 precision (see [`EmbeddingDb::store`]).
///
/// # Examples
///
/// ```
/// # let tmp = tempfile::tempdir().unwrap();
/// use docbert_core::EmbeddingDb;
///
/// let db = EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
/// db.store(1, 2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
///
/// let matrix = db.load(1).unwrap().unwrap();
/// assert_eq!(matrix.num_tokens, 2);
/// assert_eq!(matrix.dimension, 3);
/// assert_eq!(matrix.token_embedding(0), &[1.0, 2.0, 3.0]);
/// assert_eq!(matrix.token_embedding(1), &[4.0, 5.0, 6.0]);
/// ```
#[derive(Debug, Clone)]
pub struct EmbeddingMatrix {
    /// Number of tokens (rows) in the matrix.
    pub num_tokens: u32,
    /// Embedding dimension (columns) per token.
    pub dimension: u32,
    /// Flat array of f32 values in row-major order: `data[token_idx * dimension + dim_idx]`.
    pub data: Vec<f32>,
}

impl EmbeddingMatrix {
    /// Get the embedding vector for a specific token.
    ///
    /// Returns a slice of length [`dimension`](Self::dimension).
    ///
    /// # Panics
    ///
    /// Panics if `token_idx >= num_tokens`.
    pub fn token_embedding(&self, token_idx: u32) -> &[f32] {
        let start = (token_idx * self.dimension) as usize;
        let end = start + self.dimension as usize;
        &self.data[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> (tempfile::TempDir, EmbeddingDb) {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("embeddings.db")).unwrap();
        (tmp, db)
    }

    #[test]
    fn store_and_load() {
        let (_tmp, db) = test_db();

        // 3 tokens, 4 dimensions; 0.125 steps are exactly representable
        // in bf16, so the roundtrip compares bit-for-bit.
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.125).collect();
        db.store(42, 3, 4, &data).unwrap();

        let matrix = db.load(42).unwrap().unwrap();
        assert_eq!(matrix.num_tokens, 3);
        assert_eq!(matrix.dimension, 4);
        assert_eq!(matrix.data, data);
    }

    #[test]
    fn store_rounds_components_to_bf16() {
        let (_tmp, db) = test_db();

        // 0.1 is not representable in bf16; the store path must round
        // to nearest-even rather than hand back the exact input.
        let input = [0.1f32, -0.3, std::f32::consts::FRAC_1_SQRT_2];
        db.store(1, 1, 3, &input).unwrap();

        let loaded = db.load(1).unwrap().unwrap();
        for (got, want) in loaded.data.iter().zip(input.iter()) {
            assert_eq!(*got, bf16::from_f32(*want).to_f32());
            // bf16 keeps ~2^-8 relative precision: close, not exact.
            assert!((got - want).abs() <= want.abs() * 4e-3);
        }
        assert_ne!(loaded.data[0], 0.1);
    }

    /// Serialize a matrix in the pre-1.0 layout: same 8-byte header,
    /// body of little-endian f32s.
    fn legacy_f32_bytes(
        num_tokens: u32,
        dimension: u32,
        data: &[f32],
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&num_tokens.to_le_bytes());
        bytes.extend_from_slice(&dimension.to_le_bytes());
        for &v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn legacy_f32_entries_error_with_clean_instructions() {
        let (_tmp, db) = test_db();

        db.store(1, 1, 2, &[1.0, 2.0]).unwrap();
        db.insert_raw(7, &legacy_f32_bytes(2, 2, &[0.1, 0.2, 0.3, 0.4]))
            .unwrap();

        let msg = db.load(7).unwrap_err().to_string();
        assert!(msg.contains("docbert clean"), "unhelpful error: {msg}");
        assert!(db.batch_load(&[1, 7]).is_err());
        assert!(db.list_shapes().is_err());

        // Healthy rows stay readable.
        assert!(db.load(1).unwrap().is_some());
    }

    #[test]
    fn list_legacy_ids_finds_only_f32_rows() {
        let (_tmp, db) = test_db();

        db.store(1, 1, 2, &[1.0, 2.0]).unwrap(); // current layout
        db.insert_raw(2, &legacy_f32_bytes(2, 3, &[0.0; 6]))
            .unwrap();
        db.insert_raw(3, &[1, 2, 3, 4]).unwrap(); // garbage, not legacy

        assert_eq!(db.list_legacy_ids().unwrap(), vec![2]);

        // Dropping the legacy rows makes the full-scan reads work again.
        db.batch_remove(&[2]).unwrap();
        assert!(db.list_legacy_ids().unwrap().is_empty());
        assert_eq!(db.list_shapes().unwrap(), vec![(1, 1, 2)]);
    }

    #[test]
    fn list_shapes_skips_garbage() {
        let (_tmp, db) = test_db();

        db.store(1, 1, 2, &[1.0, 2.0]).unwrap(); // bf16 layout
        db.insert_raw(3, &[1, 2, 3, 4]).unwrap(); // short header
        db.insert_raw(4, &legacy_f32_bytes(5, 5, &[0.0; 3]))
            .unwrap(); // matches neither layout's length

        let shapes = db.list_shapes().unwrap();
        assert_eq!(shapes, vec![(1, 1, 2)]);
    }

    #[test]
    fn token_embedding_access() {
        let (_tmp, db) = test_db();

        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        db.store(1, 2, 3, &data).unwrap();

        let matrix = db.load(1).unwrap().unwrap();
        assert_eq!(matrix.token_embedding(0), &[0.0, 1.0, 2.0]);
        assert_eq!(matrix.token_embedding(1), &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn load_missing_returns_none() {
        let (_tmp, db) = test_db();
        assert!(db.load(999).unwrap().is_none());
    }

    #[test]
    fn load_returns_none_for_short_header() {
        let (_tmp, db) = test_db();
        db.insert_raw(7, &[1, 2, 3, 4]).unwrap();

        assert!(db.load(7).unwrap().is_none());
    }

    #[test]
    fn load_and_batch_load_match_on_length_mismatch() {
        let (_tmp, db) = test_db();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]));
        db.insert_raw(8, &bytes).unwrap();

        assert!(db.load(8).unwrap().is_none());

        let loaded = db.batch_load(&[8]).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, 8);
        assert!(loaded[0].1.is_none());
    }

    #[test]
    fn remove_entry() {
        let (_tmp, db) = test_db();

        db.store(42, 1, 2, &[1.0, 2.0]).unwrap();
        assert!(db.load(42).unwrap().is_some());

        assert!(db.remove(42).unwrap());
        assert!(db.load(42).unwrap().is_none());
        assert!(!db.remove(42).unwrap());
    }

    #[test]
    fn list_ids() {
        let (_tmp, db) = test_db();

        db.store(10, 1, 2, &[1.0, 2.0]).unwrap();
        db.store(20, 1, 2, &[3.0, 4.0]).unwrap();
        db.store(30, 1, 2, &[5.0, 6.0]).unwrap();

        let mut ids = db.list_ids().unwrap();
        ids.sort();
        assert_eq!(ids, vec![10, 20, 30]);
    }

    #[test]
    fn batch_remove_drops_only_listed_ids() {
        let (_tmp, db) = test_db();
        db.store(1, 1, 2, &[1.0, 2.0]).unwrap();
        db.store(2, 1, 2, &[3.0, 4.0]).unwrap();
        db.store(3, 1, 2, &[5.0, 6.0]).unwrap();

        db.batch_remove(&[1, 3, 999]).unwrap();

        assert!(db.load(1).unwrap().is_none());
        assert!(db.load(2).unwrap().is_some());
        assert!(db.load(3).unwrap().is_none());
    }

    #[test]
    fn batch_load_retrieves_multiple() {
        let (_tmp, db) = test_db();

        db.store(10, 1, 2, &[1.0, 2.0]).unwrap();
        db.store(20, 1, 2, &[3.0, 4.0]).unwrap();
        db.store(30, 1, 2, &[5.0, 6.0]).unwrap();

        // Load in different order, including missing ID
        let results = db.batch_load(&[30, 99, 10]).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 30);
        assert!(results[0].1.is_some());
        assert_eq!(results[0].1.as_ref().unwrap().data, vec![5.0, 6.0]);

        assert_eq!(results[1].0, 99);
        assert!(results[1].1.is_none()); // Missing

        assert_eq!(results[2].0, 10);
        assert!(results[2].1.is_some());
        assert_eq!(results[2].1.as_ref().unwrap().data, vec![1.0, 2.0]);
    }

    #[test]
    fn overwrite_entry() {
        let (_tmp, db) = test_db();

        db.store(42, 1, 2, &[1.0, 2.0]).unwrap();
        db.store(42, 2, 2, &[3.0, 4.0, 5.0, 6.0]).unwrap();

        let matrix = db.load(42).unwrap().unwrap();
        assert_eq!(matrix.num_tokens, 2);
        assert_eq!(matrix.data, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn reopen_preserves_data() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("embeddings.db");

        {
            let db = EmbeddingDb::open(&path).unwrap();
            db.store(42, 1, 2, &[1.0, 2.0]).unwrap();
        }

        {
            let db = EmbeddingDb::open(&path).unwrap();
            let matrix = db.load(42).unwrap().unwrap();
            assert_eq!(matrix.data, vec![1.0, 2.0]);
        }
    }

    #[test]
    #[should_panic(expected = "data length must equal num_tokens * dimension")]
    fn store_wrong_length_panics() {
        let (_tmp, db) = test_db();
        db.store(1, 2, 3, &[1.0, 2.0]).unwrap(); // expects 6 floats
    }
}
