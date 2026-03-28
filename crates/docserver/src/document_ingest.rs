use std::sync::MutexGuard;

use docbert_core::{DocumentId, incremental};
use tantivy::IndexWriter;

use crate::{error::ApiError, state::AppState};

pub(crate) type EmbeddingEntry = (u64, u32, u32, Vec<f32>);

const API_INGEST_MTIME: u64 = 0;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedIngestDocument {
    pub(crate) did: DocumentId,
    pub(crate) path: String,
    pub(crate) title: String,
    pub(crate) body: String,
    pub(crate) content: String,
    pub(crate) metadata: Option<serde_json::Value>,
    pub(crate) embedding_chunks: Vec<(u64, String)>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PersistedIngestDocument {
    pub(crate) doc_id: String,
    pub(crate) path: String,
    pub(crate) title: String,
    pub(crate) metadata: Option<serde_json::Value>,
}

pub(crate) trait IngestionBackend {
    fn delete_existing_documents(
        &mut self,
        documents: &[PreparedIngestDocument],
    ) -> Result<(), ApiError>;

    fn stage_document(
        &mut self,
        collection: &str,
        document: &PreparedIngestDocument,
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
        document: &PreparedIngestDocument,
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
    fn delete_existing_documents(
        &mut self,
        documents: &[PreparedIngestDocument],
    ) -> Result<(), ApiError> {
        if documents.is_empty() {
            return Ok(());
        }

        for document in documents {
            self.state
                .search_index
                .delete_document(&self.writer, &document.did.to_string());
        }
        self.writer.commit().map_err(ApiError::internal)?;
        Ok(())
    }

    fn stage_document(
        &mut self,
        collection: &str,
        document: &PreparedIngestDocument,
    ) -> Result<(), ApiError> {
        self.state.search_index.add_document(
            &self.writer,
            &document.did.to_string(),
            document.did.numeric,
            collection,
            &document.path,
            &document.title,
            &document.body,
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
        document: &PreparedIngestDocument,
    ) -> Result<(), ApiError> {
        let metadata = document_metadata(collection, document);
        self.state.config_db.put_document_artifacts(
            document.did.numeric,
            &metadata,
            &document.content,
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
    pub(crate) documents: &'a [PreparedIngestDocument],
    pub(crate) embedding_entries: &'a [EmbeddingEntry],
}

impl<'a, B> DocumentIngestCommand<'a, B>
where
    B: IngestionBackend,
{
    pub(crate) fn execute(
        self,
    ) -> Result<Vec<PersistedIngestDocument>, ApiError> {
        let embedding_ids: Vec<u64> = self
            .embedding_entries
            .iter()
            .map(|(doc_id, _, _, _)| *doc_id)
            .collect();

        self.backend.delete_existing_documents(self.documents)?;

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

        let mut persisted_doc_ids = Vec::with_capacity(self.documents.len());
        let mut ingested_documents = Vec::with_capacity(self.documents.len());

        for document in self.documents {
            if let Err(error) = self
                .backend
                .persist_document_artifacts(self.collection, document)
            {
                self.backend.rollback_staged_documents();
                self.backend.remove_embeddings(
                    &embedding_ids,
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
                path: document.path.clone(),
                title: document.title.clone(),
                metadata: document.metadata.clone(),
            });
        }

        if let Err(error) = self.backend.commit_documents() {
            self.backend
                .remove_embeddings(&embedding_ids, "tantivy commit failed");
            self.backend.remove_persisted_documents(
                &persisted_doc_ids,
                "tantivy commit failed",
            );
            return Err(error);
        }

        Ok(ingested_documents)
    }
}

#[allow(dead_code)]
pub(crate) fn document_metadata(
    collection: &str,
    document: &PreparedIngestDocument,
) -> incremental::DocumentMetadata {
    incremental::DocumentMetadata {
        collection: collection.to_string(),
        relative_path: document.path.clone(),
        mtime: API_INGEST_MTIME,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeIngestionBackend {
        deleted_doc_ids: Vec<String>,
        staged_doc_ids: Vec<String>,
        stored_embedding_ids: Vec<u64>,
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
                deleted_doc_ids: Vec::new(),
                staged_doc_ids: Vec::new(),
                stored_embedding_ids: Vec::new(),
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
        fn delete_existing_documents(
            &mut self,
            documents: &[PreparedIngestDocument],
        ) -> Result<(), ApiError> {
            self.deleted_doc_ids = documents
                .iter()
                .map(|document| document.did.to_string())
                .collect();
            Ok(())
        }

        fn stage_document(
            &mut self,
            _collection: &str,
            document: &PreparedIngestDocument,
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
            document: &PreparedIngestDocument,
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

    fn prepared_document(
        collection: &str,
        path: &str,
    ) -> PreparedIngestDocument {
        let did = DocumentId::new(collection, path);
        let doc_numeric_id = did.numeric;
        PreparedIngestDocument {
            did,
            path: path.to_string(),
            title: format!("title for {path}"),
            body: format!("body for {path}"),
            content: format!("# {path}\nbody"),
            metadata: Some(serde_json::json!({ "path": path })),
            embedding_chunks: vec![(
                doc_numeric_id,
                format!("body for {path}"),
            )],
        }
    }

    fn embedding_entries(
        documents: &[PreparedIngestDocument],
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
        assert_eq!(backend.deleted_doc_ids.len(), 2);
        assert_eq!(backend.staged_doc_ids.len(), 2);
        assert_eq!(backend.commit_count, 1);
    }
}
