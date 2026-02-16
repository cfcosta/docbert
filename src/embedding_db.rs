use std::path::Path;

use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};

use crate::error::Result;

const EMBEDDINGS: TableDefinition<u64, &[u8]> =
    TableDefinition::new("embeddings");

/// Header size: 4 bytes token count + 4 bytes dimension.
const HEADER_SIZE: usize = 8;

/// Stores ColBERT per-token embedding matrices keyed by document numeric ID.
///
/// Binary format per entry:
/// - 4 bytes: token count T (u32 LE)
/// - 4 bytes: embedding dimension D (u32 LE)
/// - T * D * 4 bytes: f32 LE values in row-major order
pub struct EmbeddingDb {
    db: Database,
}

impl EmbeddingDb {
    /// Open or create an embeddings database at the given path.
    ///
    /// # Examples
    ///
    /// ```
    /// # let tmp = tempfile::tempdir().unwrap();
    /// use docbert::EmbeddingDb;
    ///
    /// let db = EmbeddingDb::open(&tmp.path().join("embeddings.db")).unwrap();
    /// assert!(db.list_ids().unwrap().is_empty());
    /// ```
    pub fn open(path: &Path) -> Result<Self> {
        let db = Database::create(path)?;

        let txn = db.begin_write()?;
        txn.open_table(EMBEDDINGS)?;
        txn.commit()?;

        Ok(Self { db })
    }

    /// Store an embedding matrix for a document.
    ///
    /// Uses `insert_reserve` for zero-copy writes.
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

        let byte_len = HEADER_SIZE + std::mem::size_of_val(data);

        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(EMBEDDINGS)?;
            let mut guard = table.insert_reserve(doc_id, byte_len)?;
            let dest = guard.as_mut();

            dest[0..4].copy_from_slice(&num_tokens.to_le_bytes());
            dest[4..8].copy_from_slice(&dimension.to_le_bytes());
            dest[HEADER_SIZE..].copy_from_slice(bytemuck::cast_slice(data));
        }
        txn.commit()?;
        Ok(())
    }

    /// Retrieve an embedding matrix for a document.
    ///
    /// Returns (num_tokens, dimension, data) or None if not found.
    pub fn load(&self, doc_id: u64) -> Result<Option<EmbeddingMatrix>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(EMBEDDINGS)?;

        let Some(guard) = table.get(doc_id)? else {
            return Ok(None);
        };

        let bytes = guard.value();
        if bytes.len() < HEADER_SIZE {
            return Ok(None);
        }

        let num_tokens = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let dimension = u32::from_le_bytes(bytes[4..8].try_into().unwrap());

        let expected_len =
            HEADER_SIZE + (num_tokens as usize) * (dimension as usize) * 4;
        if bytes.len() != expected_len {
            return Ok(None);
        }

        let data: Vec<f32> =
            bytemuck::cast_slice(&bytes[HEADER_SIZE..]).to_vec();

        Ok(Some(EmbeddingMatrix {
            num_tokens,
            dimension,
            data,
        }))
    }

    /// Remove an embedding entry.
    pub fn remove(&self, doc_id: u64) -> Result<bool> {
        let txn = self.db.begin_write()?;
        let removed = {
            let mut table = txn.open_table(EMBEDDINGS)?;
            table.remove(doc_id)?.is_some()
        };
        txn.commit()?;
        Ok(removed)
    }

    /// Remove multiple embedding entries in a single transaction.
    pub fn batch_remove(&self, doc_ids: &[u64]) -> Result<()> {
        if doc_ids.is_empty() {
            return Ok(());
        }
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(EMBEDDINGS)?;
            for &doc_id in doc_ids {
                table.remove(doc_id)?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Store multiple embedding matrices in a single transaction.
    pub fn batch_store(
        &self,
        entries: &[(u64, u32, u32, Vec<f32>)],
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(EMBEDDINGS)?;
            for (doc_id, num_tokens, dimension, data) in entries {
                assert_eq!(
                    data.len(),
                    (*num_tokens as usize) * (*dimension as usize),
                    "data length must equal num_tokens * dimension"
                );

                let byte_len =
                    HEADER_SIZE + std::mem::size_of_val(data.as_slice());
                let mut guard = table.insert_reserve(*doc_id, byte_len)?;
                let dest = guard.as_mut();

                dest[0..4].copy_from_slice(&num_tokens.to_le_bytes());
                dest[4..8].copy_from_slice(&dimension.to_le_bytes());
                dest[HEADER_SIZE..].copy_from_slice(bytemuck::cast_slice(data));
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Load multiple embedding matrices in a single transaction.
    ///
    /// Returns a vector of `(doc_id, Option<EmbeddingMatrix>)` preserving input order.
    /// Missing embeddings return None.
    pub fn batch_load(
        &self,
        doc_ids: &[u64],
    ) -> Result<Vec<(u64, Option<EmbeddingMatrix>)>> {
        if doc_ids.is_empty() {
            return Ok(Vec::new());
        }

        let txn = self.db.begin_read()?;
        let table = txn.open_table(EMBEDDINGS)?;

        let mut results = Vec::with_capacity(doc_ids.len());
        for &doc_id in doc_ids {
            let matrix = if let Some(guard) = table.get(doc_id)? {
                let bytes = guard.value();
                if bytes.len() >= HEADER_SIZE {
                    let num_tokens =
                        u32::from_le_bytes(bytes[0..4].try_into().unwrap());
                    let dimension =
                        u32::from_le_bytes(bytes[4..8].try_into().unwrap());
                    let expected_len = HEADER_SIZE
                        + (num_tokens as usize) * (dimension as usize) * 4;

                    if bytes.len() == expected_len {
                        let data: Vec<f32> =
                            bytemuck::cast_slice(&bytes[HEADER_SIZE..])
                                .to_vec();
                        Some(EmbeddingMatrix {
                            num_tokens,
                            dimension,
                            data,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };
            results.push((doc_id, matrix));
        }

        Ok(results)
    }

    /// List all stored document IDs.
    pub fn list_ids(&self) -> Result<Vec<u64>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(EMBEDDINGS)?;
        let mut result = Vec::new();
        for entry in table.iter()? {
            let (k, _) = entry?;
            result.push(k.value());
        }
        Ok(result)
    }
}

impl std::fmt::Debug for EmbeddingDb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingDb").finish_non_exhaustive()
    }
}

/// A retrieved ColBERT embedding matrix.
#[derive(Debug, Clone)]
pub struct EmbeddingMatrix {
    pub num_tokens: u32,
    pub dimension: u32,
    /// Flat array of f32 values in row-major order: `data[token_idx * dimension + dim_idx]`.
    pub data: Vec<f32>,
}

impl EmbeddingMatrix {
    /// Get the embedding vector for a specific token.
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
