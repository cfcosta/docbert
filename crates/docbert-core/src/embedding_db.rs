use std::path::Path;

use heed::{
    Database,
    Env,
    byteorder::BigEndian,
    types::{Bytes, U64},
};

use crate::{
    chunking::document_family_key,
    error::Result,
    redb_migration::{self, EMBEDDINGS_MAX_DBS},
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

fn parse_embedding_matrix(bytes: &[u8]) -> Option<EmbeddingMatrix> {
    if bytes.len() < HEADER_SIZE {
        return None;
    }

    let num_tokens =
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let dimension =
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let expected_len =
        HEADER_SIZE + (num_tokens as usize) * (dimension as usize) * 4;
    if bytes.len() != expected_len {
        return None;
    }

    Some(EmbeddingMatrix {
        num_tokens,
        dimension,
        data: bytemuck::cast_slice(&bytes[HEADER_SIZE..]).to_vec(),
    })
}

fn serialize_embedding_matrix(
    num_tokens: u32,
    dimension: u32,
    data: &[f32],
) -> Vec<u8> {
    let byte_len = HEADER_SIZE + std::mem::size_of_val(data);
    let mut bytes = vec![0; byte_len];
    bytes[0..4].copy_from_slice(&num_tokens.to_le_bytes());
    bytes[4..8].copy_from_slice(&dimension.to_le_bytes());
    bytes[HEADER_SIZE..].copy_from_slice(bytemuck::cast_slice(data));
    bytes
}

/// Stores ColBERT token embedding matrices keyed by numeric document ID.
///
/// Each entry is packed like this:
/// - 4 bytes: token count `T` (`u32`, little-endian)
/// - 4 bytes: embedding dimension `D` (`u32`, little-endian)
/// - `T * D * 4` bytes: `f32` values in row-major order
///
/// Backed by an [LMDB](https://www.symas.com/lmdb) environment via the
/// [`heed`](https://docs.rs/heed) crate, which gives us multi-process
/// readers and writers on the same data dir — useful when several
/// `docbert mcp` / `docbert web` processes share a data dir. The
/// open path transparently migrates legacy redb-format files via
/// [`crate::redb_migration`].
pub struct EmbeddingDb {
    env: Env,
    db: Database<U64<BigEndian>, Bytes>,
}

impl EmbeddingDb {
    /// Open or create an embeddings database at the given path.
    ///
    /// If the file at `path` is still in the legacy redb format, this
    /// transparently migrates it to the heed/LMDB format before
    /// returning. The original is preserved as `<path>.redb-bak`.
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
        redb_migration::ensure_embedding_db_migrated(path)?;
        let env =
            redb_migration::open_heed_env(path, MAP_SIZE, EMBEDDINGS_MAX_DBS)?;
        let mut wtxn = env.write_txn()?;
        let db: Database<U64<BigEndian>, Bytes> =
            env.create_database(&mut wtxn, Some(EMBEDDINGS_DB_NAME))?;
        wtxn.commit()?;
        Ok(Self { env, db })
    }

    /// Store an embedding matrix for a document.
    ///
    /// Overwrites any existing embedding for this `doc_id`.
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
    /// the stored data is malformed.
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
        Ok(parse_embedding_matrix(bytes))
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

    /// Remove the base embedding and all chunk embeddings for one document.
    ///
    /// Returns the number of embedding entries removed.
    pub fn remove_document_family(&self, base_doc_id: u64) -> Result<usize> {
        self.batch_remove_document_families(&[base_doc_id])
    }

    /// Remove the base embeddings and all chunk embeddings for many documents.
    ///
    /// Returns the number of embedding entries removed.
    pub fn batch_remove_document_families(
        &self,
        base_doc_ids: &[u64],
    ) -> Result<usize> {
        if base_doc_ids.is_empty() {
            return Ok(0);
        }

        let family_keys: std::collections::HashSet<u64> = base_doc_ids
            .iter()
            .map(|&doc_id| document_family_key(doc_id))
            .collect();

        let mut wtxn = self.env.write_txn()?;
        let mut doc_ids_to_remove = Vec::new();
        for entry in self.db.iter(&wtxn)? {
            let (doc_id, _) = entry?;
            if family_keys.contains(&document_family_key(doc_id)) {
                doc_ids_to_remove.push(doc_id);
            }
        }

        let removed = doc_ids_to_remove.len();
        for doc_id in doc_ids_to_remove {
            self.db.delete(&mut wtxn, &doc_id)?;
        }
        wtxn.commit()?;
        Ok(removed)
    }

    /// Store multiple embedding matrices in a single transaction.
    ///
    /// Each entry is `(doc_id, num_tokens, dimension, data)`.
    /// More efficient than calling [`store`](Self::store) in a loop.
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
            let matrix = self
                .db
                .get(&rtxn, &doc_id)?
                .and_then(parse_embedding_matrix);
            results.push((doc_id, matrix));
        }
        Ok(results)
    }

    /// Load all stored embeddings for the requested document families.
    ///
    /// Results are grouped by the order of `base_doc_ids`. Within each family,
    /// embeddings are sorted by stored `doc_id` ascending for deterministic
    /// output. Malformed embeddings are skipped.
    pub fn batch_load_document_families(
        &self,
        base_doc_ids: &[u64],
    ) -> Result<Vec<(u64, EmbeddingMatrix)>> {
        if base_doc_ids.is_empty() {
            return Ok(Vec::new());
        }

        let requested_family_keys: std::collections::HashSet<u64> =
            base_doc_ids
                .iter()
                .map(|&doc_id| document_family_key(doc_id))
                .collect();

        let rtxn = self.env.read_txn()?;
        let mut families: std::collections::HashMap<
            u64,
            Vec<(u64, EmbeddingMatrix)>,
        > = std::collections::HashMap::new();

        for entry in self.db.iter(&rtxn)? {
            let (doc_id, bytes) = entry?;
            let family_key = document_family_key(doc_id);
            if !requested_family_keys.contains(&family_key) {
                continue;
            }

            let Some(matrix) = parse_embedding_matrix(bytes) else {
                continue;
            };

            families
                .entry(family_key)
                .or_default()
                .push((doc_id, matrix));
        }

        let mut results = Vec::new();
        for &base_doc_id in base_doc_ids {
            let family_key = document_family_key(base_doc_id);
            let Some(mut family_rows) = families.remove(&family_key) else {
                continue;
            };
            family_rows.sort_by_key(|(doc_id, _)| *doc_id);
            results.extend(family_rows);
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
    /// know an entry's shape without pulling its `T × D × 4` token
    /// bytes into RAM. The embedding bridge uses this to size a
    /// pooled token buffer up front and then stream each matrix
    /// straight into that buffer, which keeps peak RSS near a single
    /// copy of the corpus instead of two.
    ///
    /// Malformed entries (header says more data than the stored blob
    /// carries) are skipped silently, matching [`Self::load`]'s
    /// "return `None` on garbage" behaviour.
    pub fn list_shapes(&self) -> Result<Vec<(u64, u32, u32)>> {
        let rtxn = self.env.read_txn()?;
        let mut result = Vec::new();
        for entry in self.db.iter(&rtxn)? {
            let (doc_id, bytes) = entry?;
            if bytes.len() < HEADER_SIZE {
                continue;
            }
            let num_tokens =
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let dimension =
                u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
            let expected_len =
                HEADER_SIZE + (num_tokens as usize) * (dimension as usize) * 4;
            if bytes.len() != expected_len {
                continue;
            }
            result.push((doc_id, num_tokens, dimension));
        }
        Ok(result)
    }

    /// Test-only escape hatch: write raw bytes for a `doc_id` so unit
    /// tests can exercise [`Self::load`]'s parser against malformed
    /// payloads without going through the public store/serialize path.
    #[cfg(test)]
    pub(crate) fn insert_raw(&self, doc_id: u64, bytes: &[u8]) -> Result<()> {
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
    use crate::{DocumentId, chunking::chunk_doc_id};

    fn test_db() -> (tempfile::TempDir, EmbeddingDb) {
        let tmp = tempfile::tempdir().unwrap();
        let db = EmbeddingDb::open(&tmp.path().join("embeddings.db")).unwrap();
        (tmp, db)
    }

    #[test]
    fn store_and_load() {
        let (_tmp, db) = test_db();

        // 3 tokens, 4 dimensions
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        db.store(42, 3, 4, &data).unwrap();

        let matrix = db.load(42).unwrap().unwrap();
        assert_eq!(matrix.num_tokens, 3);
        assert_eq!(matrix.dimension, 4);
        assert_eq!(matrix.data, data);
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
    fn remove_document_family_removes_base_and_chunk_embeddings_only() {
        let (_tmp, db) = test_db();
        let base_doc_id = DocumentId::new("notes", "hello.md").numeric;
        let first_chunk_id = chunk_doc_id(base_doc_id, 1);
        let second_chunk_id = chunk_doc_id(base_doc_id, 2);
        let unrelated_doc_id = DocumentId::new("docs", "guide.md").numeric;
        let unrelated_chunk_id = chunk_doc_id(unrelated_doc_id, 1);

        db.store(base_doc_id, 1, 2, &[1.0, 2.0]).unwrap();
        db.store(first_chunk_id, 1, 2, &[3.0, 4.0]).unwrap();
        db.store(second_chunk_id, 1, 2, &[5.0, 6.0]).unwrap();
        db.store(unrelated_doc_id, 1, 2, &[7.0, 8.0]).unwrap();
        db.store(unrelated_chunk_id, 1, 2, &[9.0, 10.0]).unwrap();

        let removed = db.remove_document_family(base_doc_id).unwrap();
        assert_eq!(removed, 3);
        assert!(db.load(base_doc_id).unwrap().is_none());
        assert!(db.load(first_chunk_id).unwrap().is_none());
        assert!(db.load(second_chunk_id).unwrap().is_none());
        assert!(db.load(unrelated_doc_id).unwrap().is_some());
        assert!(db.load(unrelated_chunk_id).unwrap().is_some());
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
    fn batch_load_document_families_returns_base_and_chunk_embeddings_for_requested_family()
     {
        let (_tmp, db) = test_db();
        let base_doc_id = DocumentId::new("notes", "hello.md").numeric;
        let first_chunk_id = chunk_doc_id(base_doc_id, 1);
        let second_chunk_id = chunk_doc_id(base_doc_id, 2);

        db.store(base_doc_id, 1, 2, &[1.0, 2.0]).unwrap();
        db.store(first_chunk_id, 1, 2, &[3.0, 4.0]).unwrap();
        db.store(second_chunk_id, 1, 2, &[5.0, 6.0]).unwrap();

        let loaded = db.batch_load_document_families(&[base_doc_id]).unwrap();
        let loaded_ids: Vec<u64> =
            loaded.iter().map(|(doc_id, _)| *doc_id).collect();
        let mut expected_ids =
            vec![base_doc_id, first_chunk_id, second_chunk_id];
        expected_ids.sort_unstable();

        assert_eq!(loaded_ids, expected_ids);

        let loaded_by_id: std::collections::HashMap<u64, Vec<f32>> = loaded
            .into_iter()
            .map(|(doc_id, matrix)| (doc_id, matrix.data))
            .collect();
        assert_eq!(loaded_by_id.get(&base_doc_id), Some(&vec![1.0, 2.0]));
        assert_eq!(loaded_by_id.get(&first_chunk_id), Some(&vec![3.0, 4.0]));
        assert_eq!(loaded_by_id.get(&second_chunk_id), Some(&vec![5.0, 6.0]));
    }

    #[test]
    fn batch_load_document_families_excludes_unrelated_families() {
        let (_tmp, db) = test_db();
        let requested_base_doc_id =
            DocumentId::new("notes", "hello.md").numeric;
        let requested_chunk_id = chunk_doc_id(requested_base_doc_id, 1);
        let unrelated_base_doc_id = DocumentId::new("docs", "guide.md").numeric;
        let unrelated_chunk_id = chunk_doc_id(unrelated_base_doc_id, 1);

        db.store(requested_base_doc_id, 1, 2, &[1.0, 2.0]).unwrap();
        db.store(requested_chunk_id, 1, 2, &[3.0, 4.0]).unwrap();
        db.store(unrelated_base_doc_id, 1, 2, &[5.0, 6.0]).unwrap();
        db.store(unrelated_chunk_id, 1, 2, &[7.0, 8.0]).unwrap();

        let loaded = db
            .batch_load_document_families(&[requested_base_doc_id])
            .unwrap();
        let loaded_ids: Vec<u64> =
            loaded.iter().map(|(doc_id, _)| *doc_id).collect();
        let mut expected_ids = vec![requested_base_doc_id, requested_chunk_id];
        expected_ids.sort_unstable();

        assert_eq!(loaded_ids, expected_ids);
    }

    #[test]
    fn batch_load_document_families_returns_base_only_documents() {
        let (_tmp, db) = test_db();
        let base_doc_id = DocumentId::new("notes", "hello.md").numeric;

        db.store(base_doc_id, 1, 2, &[1.0, 2.0]).unwrap();

        let loaded = db.batch_load_document_families(&[base_doc_id]).unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, base_doc_id);
        assert_eq!(loaded[0].1.data, vec![1.0, 2.0]);
    }

    #[test]
    fn batch_load_document_families_empty_input_returns_empty() {
        let (_tmp, db) = test_db();

        db.store(10, 1, 2, &[1.0, 2.0]).unwrap();

        let loaded = db.batch_load_document_families(&[]).unwrap();

        assert!(loaded.is_empty());
    }

    #[test]
    fn batch_load_document_families_preserves_request_order_and_sorts_within_family()
     {
        let (_tmp, db) = test_db();
        let first_base_doc_id = DocumentId::new("notes", "a.md").numeric;
        let second_base_doc_id = DocumentId::new("notes", "b.md").numeric;
        let first_family_chunk_2 = chunk_doc_id(first_base_doc_id, 2);
        let first_family_chunk_1 = chunk_doc_id(first_base_doc_id, 1);
        let second_family_chunk_2 = chunk_doc_id(second_base_doc_id, 2);
        let second_family_chunk_1 = chunk_doc_id(second_base_doc_id, 1);

        db.store(first_family_chunk_2, 1, 2, &[1.0, 2.0]).unwrap();
        db.store(second_family_chunk_2, 1, 2, &[3.0, 4.0]).unwrap();
        db.store(first_base_doc_id, 1, 2, &[5.0, 6.0]).unwrap();
        db.store(second_family_chunk_1, 1, 2, &[7.0, 8.0]).unwrap();
        db.store(second_base_doc_id, 1, 2, &[9.0, 10.0]).unwrap();
        db.store(first_family_chunk_1, 1, 2, &[11.0, 12.0]).unwrap();

        let loaded = db
            .batch_load_document_families(&[
                second_base_doc_id,
                first_base_doc_id,
            ])
            .unwrap();
        let loaded_ids: Vec<u64> =
            loaded.iter().map(|(doc_id, _)| *doc_id).collect();
        let mut second_family_expected = vec![
            second_base_doc_id,
            second_family_chunk_1,
            second_family_chunk_2,
        ];
        second_family_expected.sort_unstable();
        let mut first_family_expected = vec![
            first_base_doc_id,
            first_family_chunk_1,
            first_family_chunk_2,
        ];
        first_family_expected.sort_unstable();
        let expected_ids: Vec<u64> = second_family_expected
            .into_iter()
            .chain(first_family_expected)
            .collect();

        assert_eq!(loaded_ids, expected_ids);
    }

    #[test]
    fn batch_load_document_families_skips_malformed_entries_and_preserves_order()
     {
        let (_tmp, db) = test_db();
        let first_base_doc_id = DocumentId::new("notes", "a.md").numeric;
        let second_base_doc_id = DocumentId::new("notes", "b.md").numeric;
        let malformed_chunk_id = chunk_doc_id(first_base_doc_id, 1);
        let valid_chunk_id = chunk_doc_id(first_base_doc_id, 2);
        let other_family_chunk_id = chunk_doc_id(second_base_doc_id, 1);

        db.store(second_base_doc_id, 1, 1, &[9.0]).unwrap();
        db.store(valid_chunk_id, 1, 1, &[2.0]).unwrap();
        db.store(first_base_doc_id, 1, 1, &[1.0]).unwrap();
        db.store(other_family_chunk_id, 1, 1, &[10.0]).unwrap();

        let mut malformed_bytes = Vec::new();
        malformed_bytes.extend_from_slice(&2u32.to_le_bytes());
        malformed_bytes.extend_from_slice(&2u32.to_le_bytes());
        malformed_bytes
            .extend_from_slice(bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]));
        db.insert_raw(malformed_chunk_id, &malformed_bytes).unwrap();

        let loaded = db
            .batch_load_document_families(&[
                second_base_doc_id,
                first_base_doc_id,
            ])
            .unwrap();
        let loaded_ids: Vec<u64> =
            loaded.iter().map(|(doc_id, _)| *doc_id).collect();
        let mut first_family_expected = vec![first_base_doc_id, valid_chunk_id];
        first_family_expected.sort_unstable();
        let mut second_family_expected =
            vec![second_base_doc_id, other_family_chunk_id];
        second_family_expected.sort_unstable();
        let expected_ids: Vec<u64> = second_family_expected
            .into_iter()
            .chain(first_family_expected)
            .collect();

        assert_eq!(loaded_ids, expected_ids);
        assert!(!loaded_ids.contains(&malformed_chunk_id));
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
