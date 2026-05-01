pub(crate) mod collections;
pub(crate) mod contexts;
pub(crate) mod indexing;
mod json_output;
pub(crate) mod model;
pub(crate) mod search;
mod style;

#[cfg(test)]
mod tests {
    use std::path::Path;

    use docbert_core::{
        ConfigDb,
        DataDir,
        DocChunkEntry,
        DocumentId,
        EmbeddingDb,
        incremental,
        model_manager::{ModelResolution, ModelSource},
    };

    use super::{
        collections,
        indexing::{
            remove_chunk_manifests_for_ids,
            remove_document_artifacts_for_ids,
        },
        json_output::{
            MultiGetJsonItem,
            collection_list_json_string,
            context_list_json_string,
            get_json_string,
            model_show_json_string,
            multi_get_json_string,
            status_json_string,
        },
        model::EMBEDDING_MODEL_KEY,
    };

    fn test_data_dir() -> (tempfile::TempDir, DataDir, ConfigDb) {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let config_db = ConfigDb::open(&data_dir.config_db()).unwrap();
        (tmp, data_dir, config_db)
    }

    fn seed_document_artifacts(
        config_db: &ConfigDb,
        collection: &str,
        path: &str,
    ) -> DocumentId {
        let doc_id = DocumentId::new(collection, path);
        let metadata = incremental::DocumentMetadata {
            collection: collection.to_string(),
            relative_path: path.to_string(),
            mtime: 0,
        };
        config_db
            .set_document_metadata_typed(doc_id.numeric, &metadata)
            .unwrap();
        config_db
            .set_document_user_metadata(
                doc_id.numeric,
                &serde_json::json!({ "topic": "rust" }),
            )
            .unwrap();
        doc_id
    }

    /// Synthesize a per-document chunk id for tests. The actual
    /// scheme isn't relevant — only that distinct documents get
    /// distinct ids and the same (doc, index) pair is stable.
    fn synthetic_chunk_id(doc_id: u64, chunk_index: usize) -> u64 {
        doc_id.wrapping_add(chunk_index as u64).wrapping_add(0xC0DE)
    }

    fn seed_document_embeddings(
        config_db: &ConfigDb,
        data_dir: &DataDir,
        doc_id: &DocumentId,
        chunk_indices: &[usize],
    ) {
        let embedding_db =
            EmbeddingDb::open(&data_dir.embeddings_db()).unwrap();
        let mut manifest = Vec::new();
        let base_chunk = synthetic_chunk_id(doc_id.numeric, 0);
        embedding_db.store(base_chunk, 1, 2, &[1.0, 2.0]).unwrap();
        manifest.push(DocChunkEntry {
            chunk_doc_id: base_chunk,
            start_byte: 0,
            byte_len: 50,
        });
        for &chunk_index in chunk_indices {
            let chunk = synthetic_chunk_id(doc_id.numeric, chunk_index);
            embedding_db.store(chunk, 1, 2, &[3.0, 4.0]).unwrap();
            manifest.push(DocChunkEntry {
                chunk_doc_id: chunk,
                start_byte: (chunk_index as u64) * 100,
                byte_len: 50,
            });
        }
        config_db.set_doc_chunks(doc_id.numeric, &manifest).unwrap();
    }

    #[test]
    fn collection_list_json_snapshot() {
        let json = collection_list_json_string(&[
            ("notes".to_string(), "/tmp/notes".to_string()),
            ("docs".to_string(), "/tmp/docs".to_string()),
        ])
        .unwrap();

        assert_eq!(
            json,
            "[{\"name\":\"notes\",\"path\":\"/tmp/notes\"},{\"name\":\"docs\",\"path\":\"/tmp/docs\"}]"
        );
    }

    #[test]
    fn context_list_json_snapshot() {
        let json = context_list_json_string(&[(
            "bert://notes".to_string(),
            "Personal notes".to_string(),
        )])
        .unwrap();

        assert_eq!(
            json,
            "[{\"uri\":\"bert://notes\",\"description\":\"Personal notes\"}]"
        );
    }

    #[test]
    fn get_json_snapshot() {
        let json = get_json_string(
            "notes",
            "hello.md",
            Path::new("/tmp/notes/hello.md"),
            Some("Hello\nWorld\n"),
        )
        .unwrap();

        assert_eq!(
            json,
            "{\"collection\":\"notes\",\"path\":\"hello.md\",\"file\":\"/tmp/notes/hello.md\",\"content\":\"Hello\\nWorld\\n\"}"
        );
    }

    #[test]
    fn multi_get_json_snapshot() {
        let json = multi_get_json_string(&[
            MultiGetJsonItem {
                collection: "notes".to_string(),
                path: "hello.md".to_string(),
                file: Some("/tmp/notes/hello.md".to_string()),
                content: Some("Hello\n".to_string()),
            },
            MultiGetJsonItem {
                collection: "docs".to_string(),
                path: "missing.md".to_string(),
                file: None,
                content: None,
            },
        ])
        .unwrap();

        assert_eq!(
            json,
            "[{\"collection\":\"notes\",\"path\":\"hello.md\",\"file\":\"/tmp/notes/hello.md\",\"content\":\"Hello\\n\"},{\"collection\":\"docs\",\"path\":\"missing.md\"}]"
        );
    }

    #[test]
    fn cli_collection_remove_drops_metadata_and_manifests_but_keeps_cache() {
        let (_tmp, data_dir, config_db) = test_data_dir();
        config_db.set_collection("notes", "/tmp/notes").unwrap();
        let doc_id = seed_document_artifacts(&config_db, "notes", "hello.md");
        seed_document_embeddings(&config_db, &data_dir, &doc_id, &[1]);

        collections::remove(&config_db, &data_dir, "notes").unwrap();

        assert!(config_db.get_collection("notes").unwrap().is_none());
        assert!(
            config_db
                .get_document_metadata_typed(doc_id.numeric)
                .unwrap()
                .is_none()
        );
        assert!(
            config_db
                .get_document_user_metadata(doc_id.numeric)
                .unwrap()
                .is_none()
        );
        assert!(
            config_db.get_doc_chunks(doc_id.numeric).unwrap().is_none(),
            "manifest should be dropped"
        );

        // Embedding cache survives — that's the immortal-cache property
        // that lets a future re-index reuse the same content.
        let embedding_db =
            EmbeddingDb::open(&data_dir.embeddings_db()).unwrap();
        let base_chunk = synthetic_chunk_id(doc_id.numeric, 0);
        let extra_chunk = synthetic_chunk_id(doc_id.numeric, 1);
        assert!(embedding_db.load(base_chunk).unwrap().is_some());
        assert!(embedding_db.load(extra_chunk).unwrap().is_some());

        // ...but those chunks no longer have any owners in the reverse
        // index, so search will skip them.
        assert!(config_db.get_chunk_owners(base_chunk).unwrap().is_empty());
        assert!(config_db.get_chunk_owners(extra_chunk).unwrap().is_empty());
    }

    #[test]
    fn cli_sync_deleted_ids_remove_document_artifacts() {
        let (_tmp, data_dir, config_db) = test_data_dir();
        let deleted =
            seed_document_artifacts(&config_db, "notes", "deleted.md");
        let retained = seed_document_artifacts(&config_db, "notes", "kept.md");
        seed_document_embeddings(&config_db, &data_dir, &deleted, &[1]);
        seed_document_embeddings(&config_db, &data_dir, &retained, &[1]);

        remove_chunk_manifests_for_ids(&config_db, &[deleted.numeric]).unwrap();
        remove_document_artifacts_for_ids(&config_db, &[deleted.numeric])
            .unwrap();

        assert!(
            config_db
                .get_document_metadata_typed(deleted.numeric)
                .unwrap()
                .is_none()
        );
        assert!(
            config_db
                .get_document_user_metadata(deleted.numeric)
                .unwrap()
                .is_none()
        );
        assert!(
            config_db
                .get_document_metadata_typed(retained.numeric)
                .unwrap()
                .is_some()
        );

        // Manifests reflect the change.
        assert!(config_db.get_doc_chunks(deleted.numeric).unwrap().is_none());
        assert!(
            config_db
                .get_doc_chunks(retained.numeric)
                .unwrap()
                .is_some()
        );

        // Embedding rows remain (immortal cache); chunk_owners
        // distinguishes which docs still reference them.
        let embedding_db =
            EmbeddingDb::open(&data_dir.embeddings_db()).unwrap();
        let retained_base = synthetic_chunk_id(retained.numeric, 0);
        let retained_extra = synthetic_chunk_id(retained.numeric, 1);
        assert!(embedding_db.load(retained_base).unwrap().is_some());
        assert!(embedding_db.load(retained_extra).unwrap().is_some());
        assert_eq!(
            config_db.get_chunk_owners(retained_base).unwrap(),
            vec![retained.numeric],
        );
    }

    #[test]
    fn cli_rebuild_collection_cleanup_removes_prior_document_artifacts() {
        let (_tmp, _data_dir, config_db) = test_data_dir();
        let first = seed_document_artifacts(&config_db, "notes", "a.md");
        let second = seed_document_artifacts(&config_db, "notes", "b.md");

        remove_document_artifacts_for_ids(
            &config_db,
            &[first.numeric, second.numeric],
        )
        .unwrap();

        for doc_id in [first.numeric, second.numeric] {
            assert!(
                config_db
                    .get_document_metadata_typed(doc_id)
                    .unwrap()
                    .is_none()
            );
            assert!(
                config_db
                    .get_document_user_metadata(doc_id)
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn status_json_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = DataDir::new(tmp.path());
        let model_resolution = ModelResolution {
            model_id: "lightonai/ColBERT-Zero".to_string(),
            source: ModelSource::Config,
            env_model: None,
            config_model: Some("lightonai/ColBERT-Zero".to_string()),
            cli_model: None,
        };

        let with_embedding = status_json_string(
            &data_dir,
            &model_resolution,
            Some("lightonai/ColBERT-Zero"),
            2,
            15,
        )
        .unwrap();
        assert_eq!(
            with_embedding,
            format!(
                "{{\"data_dir\":\"{}\",\"model\":\"lightonai/ColBERT-Zero\",\"model_source\":\"config\",\"embedding_model\":\"lightonai/ColBERT-Zero\",\"collections\":2,\"documents\":15}}",
                data_dir.root().display()
            )
        );

        let without_embedding =
            status_json_string(&data_dir, &model_resolution, None, 2, 15)
                .unwrap();
        assert_eq!(
            without_embedding,
            format!(
                "{{\"data_dir\":\"{}\",\"model\":\"lightonai/ColBERT-Zero\",\"model_source\":\"config\",\"embedding_model\":null,\"collections\":2,\"documents\":15}}",
                data_dir.root().display()
            )
        );
    }

    #[test]
    fn model_show_json_snapshot() {
        let resolution = ModelResolution {
            model_id: "cli/model".to_string(),
            source: ModelSource::Cli,
            cli_model: Some("cli/model".to_string()),
            env_model: Some("env/model".to_string()),
            config_model: None,
        };

        let json = model_show_json_string(&resolution).unwrap();
        assert_eq!(
            json,
            "{\"resolved\":\"cli/model\",\"source\":\"cli\",\"cli\":\"cli/model\",\"env\":\"env/model\",\"config\":null}"
        );
    }

    #[test]
    fn embedding_model_setting_key_is_stable() {
        assert_eq!(EMBEDDING_MODEL_KEY, "embedding_model");
    }
}
