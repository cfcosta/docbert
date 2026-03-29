use std::sync::MutexGuard;

use docbert_core::{
    chunking::document_family_key,
    document_preparation::PreparedSearchDocument,
    incremental,
};
use tantivy::IndexWriter;

use crate::{error::ApiError, state::AppState};

pub(crate) type EmbeddingEntry = (u64, u32, u32, Vec<f32>);

const API_INGEST_MTIME: u64 = 0;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PersistedIngestDocument {
    pub(crate) doc_id: String,
    pub(crate) path: String,
    pub(crate) title: String,
    pub(crate) metadata: Option<serde_json::Value>,
}

pub(crate) trait IngestionBackend {
    fn load_document_family_embeddings(
        &mut self,
        base_doc_ids: &[u64],
    ) -> Result<Vec<EmbeddingEntry>, ApiError>;

    fn stage_document(
        &mut self,
        collection: &str,
        document: &PreparedSearchDocument,
    ) -> Result<(), ApiError>;

    fn rollback_staged_documents(&mut self);

    fn store_embeddings(
        &mut self,
        entries: &[EmbeddingEntry],
    ) -> Result<(), ApiError>;

    fn remove_embeddings(&mut self, doc_ids: &[u64], context: &str);

    fn persist_document_artifacts(
        &mut self,
        collection: &str,
        document: &PreparedSearchDocument,
    ) -> Result<(), ApiError>;

    fn remove_persisted_documents(&mut self, doc_ids: &[u64], context: &str);

    fn commit_documents(&mut self) -> Result<(), ApiError>;
}

pub(crate) struct AppStateIngestionBackend<'a> {
    state: &'a AppState,
    writer: MutexGuard<'a, IndexWriter>,
}

impl<'a> AppStateIngestionBackend<'a> {
    pub(crate) fn new(state: &'a AppState) -> Result<Self, ApiError> {
        let writer = state.writer.lock().map_err(ApiError::internal)?;
        Ok(Self { state, writer })
    }
}

impl IngestionBackend for AppStateIngestionBackend<'_> {
    fn load_document_family_embeddings(
        &mut self,
        base_doc_ids: &[u64],
    ) -> Result<Vec<EmbeddingEntry>, ApiError> {
        if base_doc_ids.is_empty() {
            return Ok(Vec::new());
        }

        let family_keys: std::collections::HashSet<u64> = base_doc_ids
            .iter()
            .map(|&doc_id| document_family_key(doc_id))
            .collect();
        let doc_ids: Vec<u64> = self
            .state
            .embedding_db
            .list_ids()?
            .into_iter()
            .filter(|doc_id| {
                family_keys.contains(&document_family_key(*doc_id))
            })
            .collect();

        let loaded = self.state.embedding_db.batch_load(&doc_ids)?;
        let mut entries = Vec::with_capacity(loaded.len());
        for (doc_id, matrix) in loaded {
            let Some(matrix) = matrix else {
                continue;
            };
            entries.push((
                doc_id,
                matrix.num_tokens,
                matrix.dimension,
                matrix.data,
            ));
        }
        Ok(entries)
    }

    fn stage_document(
        &mut self,
        collection: &str,
        document: &PreparedSearchDocument,
    ) -> Result<(), ApiError> {
        self.state.search_index.add_document(
            &self.writer,
            &document.did.to_string(),
            document.did.numeric,
            collection,
            &document.relative_path,
            &document.title,
            &document.searchable_body,
            API_INGEST_MTIME,
        )?;
        Ok(())
    }

    fn rollback_staged_documents(&mut self) {
        if let Err(error) = self.writer.rollback() {
            tracing::warn!(error = %error, "failed to rollback tantivy writer");
        }
    }

    fn store_embeddings(
        &mut self,
        entries: &[EmbeddingEntry],
    ) -> Result<(), ApiError> {
        self.state
            .embedding_db
            .batch_store(entries)
            .map_err(ApiError::internal)
    }

    fn remove_embeddings(&mut self, doc_ids: &[u64], context: &str) {
        if doc_ids.is_empty() {
            return;
        }

        if let Err(error) = self.state.embedding_db.batch_remove(doc_ids) {
            tracing::warn!(error = %error, ids = doc_ids.len(), %context, "failed to cleanup embeddings");
        }
    }

    fn persist_document_artifacts(
        &mut self,
        collection: &str,
        document: &PreparedSearchDocument,
    ) -> Result<(), ApiError> {
        let metadata = document_metadata(collection, document);
        let stored_content = document
            .raw_content
            .as_deref()
            .unwrap_or(&document.searchable_body);
        self.state.config_db.put_document_artifacts(
            document.did.numeric,
            &metadata,
            stored_content,
            document.metadata.as_ref(),
        )?;
        Ok(())
    }

    fn remove_persisted_documents(&mut self, doc_ids: &[u64], context: &str) {
        for &doc_id in doc_ids {
            if let Err(error) =
                self.state.config_db.remove_document_artifacts(doc_id)
            {
                tracing::warn!(error = %error, doc_id, %context, "failed to cleanup document artifacts");
            }
        }
    }

    fn commit_documents(&mut self) -> Result<(), ApiError> {
        self.writer.commit().map(|_| ()).map_err(ApiError::internal)
    }
}

pub(crate) struct DocumentIngestCommand<'a, B> {
    pub(crate) backend: &'a mut B,
    pub(crate) collection: &'a str,
    pub(crate) documents: &'a [PreparedSearchDocument],
    pub(crate) embedding_entries: &'a [EmbeddingEntry],
}

impl<'a, B> DocumentIngestCommand<'a, B>
where
    B: IngestionBackend,
{
    pub(crate) fn execute(
        self,
    ) -> Result<Vec<PersistedIngestDocument>, ApiError> {
        let replacement_doc_ids: Vec<u64> = self
            .documents
            .iter()
            .map(|document| document.did.numeric)
            .collect();
        let current_embedding_ids: Vec<u64> = self
            .embedding_entries
            .iter()
            .map(|(doc_id, _, _, _)| *doc_id)
            .collect();
        let existing_embeddings = self
            .backend
            .load_document_family_embeddings(&replacement_doc_ids)?;
        let existing_embedding_ids: Vec<u64> = existing_embeddings
            .iter()
            .map(|(doc_id, _, _, _)| *doc_id)
            .collect();

        for document in self.documents {
            if let Err(error) =
                self.backend.stage_document(self.collection, document)
            {
                self.backend.rollback_staged_documents();
                return Err(error);
            }
        }

        if let Err(error) =
            self.backend.store_embeddings(self.embedding_entries)
        {
            self.backend.rollback_staged_documents();
            return Err(error);
        }

        let restore_previous_embeddings = |backend: &mut B, context: &str| {
            let previous_embedding_ids: std::collections::HashSet<u64> =
                existing_embedding_ids.iter().copied().collect();
            let stale_new_ids: Vec<u64> = current_embedding_ids
                .iter()
                .copied()
                .filter(|doc_id| !previous_embedding_ids.contains(doc_id))
                .collect();
            backend.remove_embeddings(&stale_new_ids, context);
            if !existing_embeddings.is_empty()
                && let Err(error) =
                    backend.store_embeddings(&existing_embeddings)
            {
                tracing::warn!(error = ?error, %context, "failed to restore previous embeddings");
            }
        };

        let mut persisted_doc_ids = Vec::with_capacity(self.documents.len());
        let mut ingested_documents = Vec::with_capacity(self.documents.len());

        for document in self.documents {
            if let Err(error) = self
                .backend
                .persist_document_artifacts(self.collection, document)
            {
                self.backend.rollback_staged_documents();
                restore_previous_embeddings(
                    self.backend,
                    "metadata persistence failed",
                );
                self.backend.remove_persisted_documents(
                    &persisted_doc_ids,
                    "metadata persistence failed",
                );
                return Err(error);
            }

            persisted_doc_ids.push(document.did.numeric);
            ingested_documents.push(PersistedIngestDocument {
                doc_id: document.did.to_string(),
                path: document.relative_path.clone(),
                title: document.title.clone(),
                metadata: document.metadata.clone(),
            });
        }

        if let Err(error) = self.backend.commit_documents() {
            restore_previous_embeddings(self.backend, "tantivy commit failed");
            self.backend.remove_persisted_documents(
                &persisted_doc_ids,
                "tantivy commit failed",
            );
            return Err(error);
        }

        let current_embedding_ids: std::collections::HashSet<u64> =
            current_embedding_ids.iter().copied().collect();
        let stale_previous_ids: Vec<u64> = existing_embedding_ids
            .into_iter()
            .filter(|doc_id| !current_embedding_ids.contains(doc_id))
            .collect();
        self.backend
            .remove_embeddings(&stale_previous_ids, "replacement committed");

        Ok(ingested_documents)
    }
}

#[allow(dead_code)]
pub(crate) fn document_metadata(
    collection: &str,
    document: &PreparedSearchDocument,
) -> incremental::DocumentMetadata {
    incremental::DocumentMetadata {
        collection: collection.to_string(),
        relative_path: document.relative_path.clone(),
        mtime: API_INGEST_MTIME,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use docbert_core::{
        ConfigDb,
        DocumentId,
        EmbeddingDb,
        ModelManager,
        SearchIndex,
        chunking::chunk_doc_id,
        incremental,
    };

    use super::*;
    use crate::state::Inner;

    struct FakeIngestionBackend {
        loaded_embeddings: Vec<EmbeddingEntry>,
        staged_doc_ids: Vec<String>,
        stored_embedding_ids: Vec<u64>,
        store_call_count: usize,
        removed_embedding_ids: Vec<u64>,
        removed_persisted_doc_ids: Vec<u64>,
        rollback_count: usize,
        fail_on_stage_doc: Option<usize>,
        fail_on_store_embeddings: bool,
        fail_on_persist_doc: Option<usize>,
        fail_on_commit: bool,
        persist_calls: usize,
        commit_count: usize,
    }

    impl FakeIngestionBackend {
        fn new() -> Self {
            Self {
                loaded_embeddings: Vec::new(),
                staged_doc_ids: Vec::new(),
                stored_embedding_ids: Vec::new(),
                store_call_count: 0,
                removed_embedding_ids: Vec::new(),
                removed_persisted_doc_ids: Vec::new(),
                rollback_count: 0,
                fail_on_stage_doc: None,
                fail_on_store_embeddings: false,
                fail_on_persist_doc: None,
                fail_on_commit: false,
                persist_calls: 0,
                commit_count: 0,
            }
        }
    }

    impl IngestionBackend for FakeIngestionBackend {
        fn load_document_family_embeddings(
            &mut self,
            _base_doc_ids: &[u64],
        ) -> Result<Vec<EmbeddingEntry>, ApiError> {
            Ok(self.loaded_embeddings.clone())
        }

        fn stage_document(
            &mut self,
            _collection: &str,
            document: &PreparedSearchDocument,
        ) -> Result<(), ApiError> {
            let next_index = self.staged_doc_ids.len();
            if self.fail_on_stage_doc == Some(next_index) {
                return Err(ApiError::internal("stage_document failed"));
            }
            self.staged_doc_ids.push(document.did.to_string());
            Ok(())
        }

        fn rollback_staged_documents(&mut self) {
            self.rollback_count += 1;
        }

        fn store_embeddings(
            &mut self,
            entries: &[EmbeddingEntry],
        ) -> Result<(), ApiError> {
            if self.fail_on_store_embeddings {
                return Err(ApiError::internal("store_embeddings failed"));
            }
            self.store_call_count += 1;
            self.stored_embedding_ids =
                entries.iter().map(|(doc_id, _, _, _)| *doc_id).collect();
            Ok(())
        }

        fn remove_embeddings(&mut self, doc_ids: &[u64], _context: &str) {
            self.removed_embedding_ids = doc_ids.to_vec();
        }

        fn persist_document_artifacts(
            &mut self,
            _collection: &str,
            document: &PreparedSearchDocument,
        ) -> Result<(), ApiError> {
            let next_index = self.persist_calls;
            self.persist_calls += 1;
            if self.fail_on_persist_doc == Some(next_index) {
                return Err(ApiError::internal(
                    "persist_document_artifacts failed",
                ));
            }
            self.removed_persisted_doc_ids
                .retain(|doc_id| *doc_id != document.did.numeric);
            Ok(())
        }

        fn remove_persisted_documents(
            &mut self,
            doc_ids: &[u64],
            _context: &str,
        ) {
            self.removed_persisted_doc_ids = doc_ids.to_vec();
        }

        fn commit_documents(&mut self) -> Result<(), ApiError> {
            self.commit_count += 1;
            if self.fail_on_commit {
                return Err(ApiError::internal("commit_documents failed"));
            }
            Ok(())
        }
    }

    fn test_state() -> (tempfile::TempDir, AppState) {
        let tmp = tempfile::tempdir().unwrap();
        let config_db = ConfigDb::open(&tmp.path().join("config.db")).unwrap();
        let search_index = SearchIndex::open_in_ram().unwrap();
        let embedding_db =
            EmbeddingDb::open(&tmp.path().join("emb.db")).unwrap();
        let writer = search_index.writer(15_000_000).unwrap();
        let state = Arc::new(Inner {
            config_db,
            search_index,
            embedding_db,
            model: Mutex::new(ModelManager::new()),
            writer: Mutex::new(writer),
        });

        (tmp, state)
    }

    fn seed_existing_document(
        state: &AppState,
        collection: &str,
        path: &str,
        title: &str,
        body: &str,
        chunk_count: usize,
    ) -> DocumentId {
        state.config_db.set_managed_collection(collection).unwrap();
        let did = DocumentId::new(collection, path);
        let metadata = incremental::DocumentMetadata {
            collection: collection.to_string(),
            relative_path: path.to_string(),
            mtime: API_INGEST_MTIME,
        };
        state
            .config_db
            .put_document_artifacts(did.numeric, &metadata, body, None)
            .unwrap();
        {
            let mut writer = state.writer.lock().unwrap();
            state
                .search_index
                .add_document(
                    &writer,
                    &did.to_string(),
                    did.numeric,
                    collection,
                    path,
                    title,
                    body,
                    API_INGEST_MTIME,
                )
                .unwrap();
            writer.commit().unwrap();
        }
        state.embedding_db.store(did.numeric, 1, 1, &[1.0]).unwrap();
        for chunk_index in 1..chunk_count {
            state
                .embedding_db
                .store(
                    chunk_doc_id(did.numeric, chunk_index),
                    1,
                    1,
                    &[chunk_index as f32],
                )
                .unwrap();
        }
        did
    }

    struct FailPersistBackend<'a> {
        inner: AppStateIngestionBackend<'a>,
    }

    impl IngestionBackend for FailPersistBackend<'_> {
        fn load_document_family_embeddings(
            &mut self,
            base_doc_ids: &[u64],
        ) -> Result<Vec<EmbeddingEntry>, ApiError> {
            self.inner.load_document_family_embeddings(base_doc_ids)
        }

        fn stage_document(
            &mut self,
            collection: &str,
            document: &PreparedSearchDocument,
        ) -> Result<(), ApiError> {
            self.inner.stage_document(collection, document)
        }

        fn rollback_staged_documents(&mut self) {
            self.inner.rollback_staged_documents();
        }

        fn store_embeddings(
            &mut self,
            entries: &[EmbeddingEntry],
        ) -> Result<(), ApiError> {
            self.inner.store_embeddings(entries)
        }

        fn remove_embeddings(&mut self, doc_ids: &[u64], context: &str) {
            self.inner.remove_embeddings(doc_ids, context);
        }

        fn persist_document_artifacts(
            &mut self,
            _collection: &str,
            _document: &PreparedSearchDocument,
        ) -> Result<(), ApiError> {
            Err(ApiError::internal("persist_document_artifacts failed"))
        }

        fn remove_persisted_documents(
            &mut self,
            doc_ids: &[u64],
            context: &str,
        ) {
            self.inner.remove_persisted_documents(doc_ids, context);
        }

        fn commit_documents(&mut self) -> Result<(), ApiError> {
            self.inner.commit_documents()
        }
    }

    fn prepared_document(
        collection: &str,
        path: &str,
    ) -> PreparedSearchDocument {
        let did = DocumentId::new(collection, path);
        PreparedSearchDocument {
            did,
            relative_path: path.to_string(),
            title: format!("title for {path}"),
            searchable_body: format!("body for {path}"),
            raw_content: Some(format!("# {path}\nbody")),
            metadata: Some(serde_json::json!({ "path": path })),
            mtime: API_INGEST_MTIME,
        }
    }

    fn embedding_entries(
        documents: &[PreparedSearchDocument],
    ) -> Vec<EmbeddingEntry> {
        documents
            .iter()
            .map(|document| (document.did.numeric, 1, 1, vec![1.0]))
            .collect()
    }

    #[test]
    fn document_metadata_uses_api_ingest_mtime() {
        let document = prepared_document("notes", "a.md");
        let metadata = document_metadata("notes", &document);

        assert_eq!(API_INGEST_MTIME, 0);
        assert_eq!(metadata.mtime, API_INGEST_MTIME);
    }

    #[test]
    fn persist_document_artifacts_uses_raw_content_when_present() {
        let (_tmp, state) = test_state();
        state.config_db.set_managed_collection("notes").unwrap();
        let document = prepared_document("notes", "a.md");
        let mut backend = AppStateIngestionBackend::new(&state).unwrap();

        backend
            .persist_document_artifacts("notes", &document)
            .unwrap();

        assert_eq!(
            state
                .config_db
                .get_document_content(document.did.numeric)
                .unwrap(),
            document.raw_content.clone()
        );
    }

    #[test]
    fn persist_document_artifacts_falls_back_to_searchable_body_when_raw_content_missing()
     {
        let (_tmp, state) = test_state();
        state.config_db.set_managed_collection("notes").unwrap();
        let mut document = prepared_document("notes", "a.md");
        document.raw_content = None;
        let mut backend = AppStateIngestionBackend::new(&state).unwrap();

        backend
            .persist_document_artifacts("notes", &document)
            .unwrap();

        assert_eq!(
            state
                .config_db
                .get_document_content(document.did.numeric)
                .unwrap(),
            Some(document.searchable_body.clone())
        );
    }

    #[test]
    fn ingest_cleans_embeddings_and_persisted_docs_when_embedding_store_fails()
    {
        let documents = vec![prepared_document("notes", "a.md")];
        let embeddings = embedding_entries(&documents);
        let mut backend = FakeIngestionBackend {
            fail_on_store_embeddings: true,
            ..FakeIngestionBackend::new()
        };

        let error = DocumentIngestCommand {
            backend: &mut backend,
            collection: "notes",
            documents: &documents,
            embedding_entries: &embeddings,
        }
        .execute()
        .unwrap_err();

        assert!(matches!(error, ApiError::Internal(_)));
        assert_eq!(backend.rollback_count, 1);
        assert!(backend.removed_embedding_ids.is_empty());
        assert!(backend.removed_persisted_doc_ids.is_empty());
    }

    #[test]
    fn ingest_cleans_embeddings_and_prior_persisted_docs_when_artifact_persist_fails_mid_batch()
     {
        let documents = vec![
            prepared_document("notes", "a.md"),
            prepared_document("notes", "b.md"),
        ];
        let embeddings = embedding_entries(&documents);
        let mut backend = FakeIngestionBackend {
            fail_on_persist_doc: Some(1),
            ..FakeIngestionBackend::new()
        };

        let error = DocumentIngestCommand {
            backend: &mut backend,
            collection: "notes",
            documents: &documents,
            embedding_entries: &embeddings,
        }
        .execute()
        .unwrap_err();

        assert!(matches!(error, ApiError::Internal(_)));
        assert_eq!(backend.rollback_count, 1);
        assert_eq!(
            backend.removed_embedding_ids,
            embeddings
                .iter()
                .map(|(doc_id, _, _, _)| *doc_id)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            backend.removed_persisted_doc_ids,
            vec![documents[0].did.numeric]
        );
    }

    #[test]
    fn ingest_cleans_other_state_when_index_commit_fails() {
        let documents = vec![
            prepared_document("notes", "a.md"),
            prepared_document("notes", "b.md"),
        ];
        let embeddings = embedding_entries(&documents);
        let mut backend = FakeIngestionBackend {
            fail_on_commit: true,
            ..FakeIngestionBackend::new()
        };

        let error = DocumentIngestCommand {
            backend: &mut backend,
            collection: "notes",
            documents: &documents,
            embedding_entries: &embeddings,
        }
        .execute()
        .unwrap_err();

        assert!(matches!(error, ApiError::Internal(_)));
        assert_eq!(backend.commit_count, 1);
        assert_eq!(
            backend.removed_embedding_ids,
            embeddings
                .iter()
                .map(|(doc_id, _, _, _)| *doc_id)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            backend.removed_persisted_doc_ids,
            documents
                .iter()
                .map(|document| document.did.numeric)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn replacement_failure_restores_previous_searchable_document_state() {
        let (_tmp, state) = test_state();
        let did = seed_existing_document(
            &state,
            "notes",
            "note.md",
            "Legacy Title",
            "legacy body",
            2,
        );
        let replacement_document = PreparedSearchDocument {
            did: did.clone(),
            relative_path: "note.md".to_string(),
            title: "Replacement Title".to_string(),
            searchable_body: "replacement body".to_string(),
            raw_content: Some("replacement body".to_string()),
            metadata: None,
            mtime: API_INGEST_MTIME,
        };
        let replacement_embeddings = vec![
            (did.numeric, 1, 1, vec![9.0]),
            (chunk_doc_id(did.numeric, 1), 1, 1, vec![8.0]),
            (chunk_doc_id(did.numeric, 2), 1, 1, vec![7.0]),
        ];
        let mut backend = FailPersistBackend {
            inner: AppStateIngestionBackend::new(&state).unwrap(),
        };

        let error = DocumentIngestCommand {
            backend: &mut backend,
            collection: "notes",
            documents: &[replacement_document],
            embedding_entries: &replacement_embeddings,
        }
        .execute()
        .unwrap_err();

        assert!(matches!(error, ApiError::Internal(_)));
        assert_eq!(
            state.config_db.get_document_content(did.numeric).unwrap(),
            Some("legacy body".to_string())
        );
        assert_eq!(state.search_index.search("legacy", 10).unwrap().len(), 1);
        assert!(
            state
                .search_index
                .search("replacement", 10)
                .unwrap()
                .is_empty()
        );
        assert!(state.embedding_db.load(did.numeric).unwrap().is_some());
        assert!(
            state
                .embedding_db
                .load(chunk_doc_id(did.numeric, 1))
                .unwrap()
                .is_some()
        );
        assert!(
            state
                .embedding_db
                .load(chunk_doc_id(did.numeric, 2))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn replacement_success_removes_stale_old_chunk_embeddings() {
        let (_tmp, state) = test_state();
        let did = seed_existing_document(
            &state,
            "notes",
            "note.md",
            "Legacy Title",
            "legacy body",
            3,
        );
        let replacement_document = PreparedSearchDocument {
            did: did.clone(),
            relative_path: "note.md".to_string(),
            title: "Replacement Title".to_string(),
            searchable_body: "replacement body".to_string(),
            raw_content: Some("replacement body".to_string()),
            metadata: None,
            mtime: API_INGEST_MTIME,
        };
        let replacement_embeddings = vec![(did.numeric, 1, 1, vec![9.0])];
        let mut backend = AppStateIngestionBackend::new(&state).unwrap();

        let ingested = DocumentIngestCommand {
            backend: &mut backend,
            collection: "notes",
            documents: &[replacement_document],
            embedding_entries: &replacement_embeddings,
        }
        .execute()
        .unwrap();

        assert_eq!(ingested.len(), 1);
        assert_eq!(
            state.config_db.get_document_content(did.numeric).unwrap(),
            Some("replacement body".to_string())
        );
        assert_eq!(
            state.search_index.search("replacement", 10).unwrap().len(),
            1
        );
        assert!(state.search_index.search("legacy", 10).unwrap().is_empty());
        assert!(state.embedding_db.load(did.numeric).unwrap().is_some());
        assert!(
            state
                .embedding_db
                .load(chunk_doc_id(did.numeric, 1))
                .unwrap()
                .is_none()
        );
        assert!(
            state
                .embedding_db
                .load(chunk_doc_id(did.numeric, 2))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn ingest_preserves_response_document_order_for_multi_document_success() {
        let documents = vec![
            prepared_document("notes", "b.md"),
            prepared_document("notes", "a.md"),
        ];
        let embeddings = embedding_entries(&documents);
        let mut backend = FakeIngestionBackend::new();

        let ingested = DocumentIngestCommand {
            backend: &mut backend,
            collection: "notes",
            documents: &documents,
            embedding_entries: &embeddings,
        }
        .execute()
        .unwrap();

        assert_eq!(
            ingested
                .iter()
                .map(|document| document.path.as_str())
                .collect::<Vec<_>>(),
            vec!["b.md", "a.md"]
        );
        assert_eq!(backend.staged_doc_ids.len(), 2);
        assert_eq!(backend.store_call_count, 1);
        assert_eq!(backend.commit_count, 1);
    }
}
