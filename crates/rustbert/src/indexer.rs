//! `Indexer` — the one place that owns the docbert-core stack
//! (`SearchIndex` / `EmbeddingDb` / `ConfigDb` / `ModelManager`).
//!
//! Every code path that wants to index, search, or evict crate items
//! goes through this struct. The JSON cache (see [`crate::cache`])
//! continues to hold the parsed `Vec<RustItem>` as the human-readable
//! source of truth; the indexer maintains the searchable artifacts on
//! top of it: Tantivy entries for BM25, ColBERT chunk embeddings, and
//! a periodically-rebuilt PLAID index.

use std::path::Path;

pub use docbert_core::search::FinalResult;
use docbert_core::{
    ConfigDb,
    DataDir,
    EmbeddingDb,
    ModelManager,
    SearchIndex,
    chunking,
    embedding,
    incremental::DocumentMetadata,
    plaid::{self, PlaidBuildParams},
    preparation::{self, SearchDocument},
    search::{self, SearchParams},
};

use crate::{
    collection::SyntheticCollection,
    error::{Error, Result},
    item::RustItem,
    lowering,
};

const TANTIVY_WRITER_BUDGET: usize = 50_000_000;

pub struct Indexer {
    data_dir: DataDir,
    config_db: ConfigDb,
    search_index: SearchIndex,
    embedding_db: EmbeddingDb,
    model: ModelManager,
}

impl Indexer {
    /// Open the docbert-core stack rooted at `data_dir`.
    pub fn open(data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir)?;
        let docbert_data_dir = DataDir::new(data_dir);
        let tantivy_path =
            docbert_data_dir.tantivy_dir().map_err(map_core_err)?;
        std::fs::create_dir_all(&tantivy_path)?;

        let config_db = ConfigDb::open(&docbert_data_dir.config_db())
            .map_err(map_core_err)?;
        let search_index =
            SearchIndex::open(&tantivy_path).map_err(map_core_err)?;
        let embedding_db = EmbeddingDb::open(&docbert_data_dir.embeddings_db())
            .map_err(map_core_err)?;
        let model = ModelManager::new();

        Ok(Self {
            data_dir: docbert_data_dir,
            config_db,
            search_index,
            embedding_db,
            model,
        })
    }

    /// Borrow the inner [`ConfigDb`] for direct queries.
    pub fn config_db(&self) -> &ConfigDb {
        &self.config_db
    }

    /// Borrow the inner [`DataDir`] for path lookups.
    pub fn data_dir(&self) -> &DataDir {
        &self.data_dir
    }

    /// Index a crate's items + embed them as a single unit. Either
    /// both succeed or the lexical entries are rolled back so the
    /// data store never holds half-indexed crates.
    ///
    /// PLAID is **not** rebuilt here — call [`Self::rebuild_plaid`]
    /// once after a batch of ingestions completes.
    pub fn index_items(
        &mut self,
        collection: &SyntheticCollection,
        items: &[RustItem],
    ) -> Result<usize> {
        let count = self.index_lexical(collection, items)?;
        if let Err(e) = self.embed_items(collection, items) {
            // Roll back the lexical entries so an embed failure
            // doesn't leave the index inconsistent.
            let _ = self.remove_collection(collection);
            return Err(e);
        }
        Ok(count)
    }

    /// Internal: lexical write only. Use [`Self::index_items`] from
    /// production paths so the lexical and semantic indexes stay
    /// coupled.
    fn index_lexical(
        &self,
        collection: &SyntheticCollection,
        items: &[RustItem],
    ) -> Result<usize> {
        let collection_name = collection.to_string();
        let documents: Vec<SearchDocument> = items
            .iter()
            .map(|i| lowering::lower(collection, i))
            .collect();

        let mut writer = self
            .search_index
            .writer(TANTIVY_WRITER_BUDGET)
            .map_err(map_core_err)?;

        // Wipe the collection's prior entries before re-indexing so a
        // re-run doesn't accumulate stale items.
        self.search_index
            .delete_collection(&writer, &collection_name)
            .map_err(map_core_err)?;

        for doc in &documents {
            self.search_index
                .add_document(
                    &writer,
                    &doc.did.full_hex(),
                    doc.did.numeric,
                    &collection_name,
                    &doc.relative_path,
                    &doc.title,
                    &doc.searchable_body,
                    doc.mtime,
                )
                .map_err(map_core_err)?;
        }
        writer.commit()?;

        // ConfigDb metadata so the semantic leg can resolve doc_ids.
        let metadata_entries: Vec<(u64, DocumentMetadata)> = documents
            .iter()
            .map(|d| {
                (
                    d.did.numeric,
                    DocumentMetadata {
                        collection: collection_name.clone(),
                        relative_path: d.relative_path.clone(),
                        mtime: d.mtime,
                    },
                )
            })
            .collect();
        self.config_db
            .batch_set_document_metadata_typed(&metadata_entries)
            .map_err(map_core_err)?;

        Ok(documents.len())
    }

    /// Internal: embed each item's `searchable_body` via the ColBERT
    /// model and store per-chunk vectors in the embedding DB.
    /// Triggers the model download on first call. Use
    /// [`Self::index_items`] from production paths.
    fn embed_items(
        &mut self,
        collection: &SyntheticCollection,
        items: &[RustItem],
    ) -> Result<usize> {
        let chunking_config = chunking::resolve_config(self.model.model_id());

        let mut chunks: Vec<(u64, String)> = Vec::new();
        for item in items {
            let doc = lowering::lower(collection, item);
            chunks
                .extend(preparation::embedding_chunks(&doc, &chunking_config));
        }

        if chunks.is_empty() {
            return Ok(0);
        }

        embedding::embed_and_store(&mut self.model, &self.embedding_db, chunks)
            .map_err(map_core_err)
    }

    /// Rebuild the PLAID index from every stored embedding. Cheap at
    /// startup, expensive on large corpora — call after a sync, not
    /// after every single ingest.
    ///
    /// Fail-soft: PLAID is a search *optimization* over the
    /// embeddings already in `EmbeddingDb`. When the corpus is too
    /// small for the configured k-means (the common "first crate
    /// indexed" case yields fewer than 256 tokens), this logs a
    /// warning and returns Ok — `search::run` falls back to a linear
    /// MaxSim scan when no PLAID index exists.
    pub fn rebuild_plaid(&self) -> Result<()> {
        let params = PlaidBuildParams::default();
        match plaid::build_index_from_embedding_db(&self.embedding_db, params) {
            Ok(index) => {
                plaid::save_index(&index, &self.data_dir)
                    .map_err(map_core_err)?;
                Ok(())
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "PLAID rebuild skipped — search falls back to linear MaxSim"
                );
                Ok(())
            }
        }
    }

    /// Hybrid (BM25 + ColBERT/PLAID) search via docbert-core. Falls
    /// back to BM25-only when no PLAID index has been built yet so
    /// search remains useful immediately after the first ingest.
    pub fn search(&mut self, params: SearchParams) -> Result<Vec<FinalResult>> {
        let plaid_path = self.data_dir.plaid_index();
        let bm25_only = params.bm25_only || !plaid_path.exists();
        let effective = SearchParams {
            bm25_only,
            ..params
        };
        search::run(
            &effective,
            &self.search_index,
            &self.config_db,
            &self.data_dir,
            &mut self.model,
        )
        .map_err(map_core_err)
    }

    /// Remove a collection's entries from Tantivy and metadata. The
    /// JSON cache is the responsibility of [`crate::cache::CrateCache`].
    pub fn remove_collection(
        &self,
        collection: &SyntheticCollection,
    ) -> Result<()> {
        let collection_name = collection.to_string();
        let mut writer = self
            .search_index
            .writer(TANTIVY_WRITER_BUDGET)
            .map_err(map_core_err)?;
        self.search_index
            .delete_collection(&writer, &collection_name)
            .map_err(map_core_err)?;
        writer.commit()?;
        Ok(())
    }
}

fn map_core_err(e: docbert_core::Error) -> Error {
    Error::Cache(format!("docbert-core: {e}"))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;
    use crate::item::{RustItemKind, Visibility};

    fn collection() -> SyntheticCollection {
        SyntheticCollection {
            crate_name: "demo".to_string(),
            version: semver::Version::new(0, 1, 0),
        }
    }

    fn sample(qpath: &str, doc: &str) -> RustItem {
        RustItem {
            kind: RustItemKind::Fn,
            crate_name: "demo".to_string(),
            crate_version: semver::Version::new(0, 1, 0),
            module_path: vec![],
            name: Some("f".to_string()),
            qualified_path: qpath.to_string(),
            signature: "pub fn f()".to_string(),
            doc_markdown: doc.to_string(),
            body: "pub fn f () { }".to_string(),
            source_file: PathBuf::from("src/lib.rs"),
            byte_start: 0,
            byte_len: 0,
            line_start: 1,
            line_end: 1,
            visibility: Visibility::Public,
            attrs: vec![],
        }
    }

    #[test]
    fn opens_against_a_fresh_data_dir() {
        let tmp = TempDir::new().unwrap();
        let _ = Indexer::open(tmp.path()).unwrap();
        // Subdirs created on demand.
        assert!(tmp.path().join("tantivy").is_dir());
    }

    #[test]
    fn index_lexical_writes_to_tantivy_and_metadata() {
        let tmp = TempDir::new().unwrap();
        let indexer = Indexer::open(tmp.path()).unwrap();
        let coll = collection();
        let items = vec![
            sample("demo::greet", "say hello"),
            sample("demo::farewell", "say bye"),
        ];
        let count = indexer.index_lexical(&coll, &items).unwrap();
        assert_eq!(count, 2);

        // ConfigDb has the metadata.
        let did =
            docbert_core::DocumentId::new(&coll.to_string(), "demo::greet");
        let meta = indexer
            .config_db
            .get_document_metadata_typed(did.numeric)
            .unwrap()
            .unwrap();
        assert_eq!(meta.collection, coll.to_string());
    }

    #[test]
    fn bm25_search_returns_lexical_hits() {
        let tmp = TempDir::new().unwrap();
        let mut indexer = Indexer::open(tmp.path()).unwrap();
        let coll = collection();
        let items = vec![
            sample("demo::greet", "say hello to a friend"),
            sample("demo::farewell", "say bye to a friend"),
        ];
        // Tests use `index_lexical` directly to avoid loading the
        // ColBERT model in CI; production paths must go through
        // `index_items` so lexical + embed stay coupled.
        indexer.index_lexical(&coll, &items).unwrap();

        let params = SearchParams {
            query: "hello".to_string(),
            count: 10,
            collection: Some(coll.to_string()),
            min_score: 0.0,
            bm25_only: true,
            no_fuzzy: false,
            all: false,
        };
        let results = indexer.search(params).unwrap();
        assert!(!results.is_empty());
        // The "hello" doc should rank above the "bye" doc.
        assert_eq!(results[0].title, "demo::greet");
    }

    #[test]
    fn re_indexing_replaces_prior_entries() {
        let tmp = TempDir::new().unwrap();
        let mut indexer = Indexer::open(tmp.path()).unwrap();
        let coll = collection();
        indexer
            .index_lexical(&coll, &[sample("demo::a", "alpha")])
            .unwrap();
        indexer
            .index_lexical(&coll, &[sample("demo::b", "beta")])
            .unwrap();

        let params = SearchParams {
            query: "alpha".to_string(),
            count: 10,
            collection: Some(coll.to_string()),
            min_score: 0.0,
            bm25_only: true,
            no_fuzzy: false,
            all: false,
        };
        let results = indexer.search(params).unwrap();
        assert!(
            results.is_empty(),
            "old `alpha` entry should have been deleted, got {results:?}",
        );
    }

    #[test]
    fn remove_collection_drops_tantivy_entries() {
        let tmp = TempDir::new().unwrap();
        let mut indexer = Indexer::open(tmp.path()).unwrap();
        let coll = collection();
        indexer
            .index_lexical(&coll, &[sample("demo::a", "alpha")])
            .unwrap();
        indexer.remove_collection(&coll).unwrap();

        let params = SearchParams {
            query: "alpha".to_string(),
            count: 10,
            collection: Some(coll.to_string()),
            min_score: 0.0,
            bm25_only: true,
            no_fuzzy: false,
            all: false,
        };
        let results = indexer.search(params).unwrap();
        assert!(results.is_empty());
    }
}
