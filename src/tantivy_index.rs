use std::path::Path;

use tantivy::{
    Index,
    IndexReader,
    IndexWriter,
    TantivyDocument,
    collector::TopDocs,
    doc,
    query::QueryParser,
    schema::*,
    tokenizer::{
        LowerCaser,
        RemoveLongFilter,
        SimpleTokenizer,
        Stemmer,
        TextAnalyzer,
    },
};

use crate::error::Result;

/// Field names used in the schema.
pub mod fields {
    pub const DOC_ID: &str = "doc_id";
    pub const DOC_NUM_ID: &str = "doc_num_id";
    pub const COLLECTION: &str = "collection";
    pub const PATH: &str = "path";
    pub const TITLE: &str = "title";
    pub const BODY: &str = "body";
    pub const MTIME: &str = "mtime";
}

/// Manages a Tantivy full-text search index for docbert documents.
pub struct SearchIndex {
    index: Index,
    reader: IndexReader,
    schema: Schema,
}

/// Resolved field handles for the schema.
#[derive(Clone, Copy)]
pub struct SchemaFields {
    pub doc_id: Field,
    pub doc_num_id: Field,
    pub collection: Field,
    pub path: Field,
    pub title: Field,
    pub body: Field,
    pub mtime: Field,
}

/// A search result from the index.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub score: f32,
    pub doc_id: String,
    pub doc_num_id: u64,
    pub collection: String,
    pub path: String,
    pub title: String,
    pub mtime: u64,
}

fn build_schema() -> (Schema, SchemaFields) {
    let mut builder = Schema::builder();

    let doc_id = builder.add_text_field(fields::DOC_ID, STRING | STORED);
    let doc_num_id = builder.add_u64_field(fields::DOC_NUM_ID, STORED | FAST);
    let collection =
        builder.add_text_field(fields::COLLECTION, STRING | STORED | FAST);
    let path = builder.add_text_field(fields::PATH, STRING | STORED);

    let title_opts = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("en_stem")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored();
    let title = builder.add_text_field(fields::TITLE, title_opts);

    let body_opts = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("en_stem")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    );
    let body = builder.add_text_field(fields::BODY, body_opts);

    let mtime = builder.add_u64_field(fields::MTIME, STORED | FAST);

    let schema = builder.build();
    let fields = SchemaFields {
        doc_id,
        doc_num_id,
        collection,
        path,
        title,
        body,
        mtime,
    };

    (schema, fields)
}

fn register_tokenizers(index: &Index) {
    let en_stem = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(RemoveLongFilter::limit(40))
        .filter(LowerCaser)
        .filter(Stemmer::new(tantivy::tokenizer::Language::English))
        .build();
    index.tokenizers().register("en_stem", en_stem);
}

impl SearchIndex {
    /// Open or create a search index at the given directory.
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let (schema, _) = build_schema();

        let mmap_dir = tantivy::directory::MmapDirectory::open(dir)
            .map_err(|e| tantivy::TantivyError::SystemError(e.to_string()))?;
        let index = if Index::exists(&mmap_dir)
            .map_err(|e| tantivy::TantivyError::SystemError(e.to_string()))?
        {
            Index::open(mmap_dir)?
        } else {
            Index::create(
                mmap_dir,
                schema.clone(),
                tantivy::IndexSettings::default(),
            )?
        };

        register_tokenizers(&index);
        let reader = index.reader()?;

        Ok(Self {
            index,
            reader,
            schema,
        })
    }

    /// Create an in-memory search index (for testing).
    pub fn open_in_ram() -> Result<Self> {
        let (schema, _) = build_schema();
        let index = Index::create_in_ram(schema.clone());
        register_tokenizers(&index);
        let reader = index.reader()?;

        Ok(Self {
            index,
            reader,
            schema,
        })
    }

    /// Get the resolved field handles.
    pub fn fields(&self) -> SchemaFields {
        let f = |name: &str| self.schema.get_field(name).unwrap();
        SchemaFields {
            doc_id: f(fields::DOC_ID),
            doc_num_id: f(fields::DOC_NUM_ID),
            collection: f(fields::COLLECTION),
            path: f(fields::PATH),
            title: f(fields::TITLE),
            body: f(fields::BODY),
            mtime: f(fields::MTIME),
        }
    }

    /// Create a writer with the given memory budget (in bytes).
    pub fn writer(&self, memory_budget: usize) -> Result<IndexWriter> {
        Ok(self.index.writer(memory_budget)?)
    }

    /// Add a document to the index via the given writer.
    #[allow(clippy::too_many_arguments)]
    pub fn add_document(
        &self,
        writer: &IndexWriter,
        doc_id: &str,
        doc_num_id: u64,
        collection: &str,
        path: &str,
        title: &str,
        body: &str,
        mtime: u64,
    ) -> Result<()> {
        let f = self.fields();

        // Delete any existing document with this ID first.
        let term = tantivy::Term::from_field_text(f.doc_id, doc_id);
        writer.delete_term(term);

        writer.add_document(doc!(
            f.doc_id => doc_id,
            f.doc_num_id => doc_num_id,
            f.collection => collection,
            f.path => path,
            f.title => title,
            f.body => body,
            f.mtime => mtime,
        ))?;

        Ok(())
    }

    /// Delete all documents belonging to a collection.
    pub fn delete_collection(&self, writer: &IndexWriter, collection: &str) {
        let f = self.fields();
        let term = tantivy::Term::from_field_text(f.collection, collection);
        writer.delete_term(term);
    }

    /// Delete a single document by its short doc_id.
    pub fn delete_document(&self, writer: &IndexWriter, doc_id: &str) {
        let f = self.fields();
        let term = tantivy::Term::from_field_text(f.doc_id, doc_id);
        writer.delete_term(term);
    }

    /// Search the index with BM25 scoring.
    ///
    /// Returns the top `limit` results. The `title` field is boosted 2x.
    pub fn search(
        &self,
        query_str: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let f = self.fields();
        self.reader.reload()?;
        let searcher = self.reader.searcher();

        let mut parser =
            QueryParser::for_index(&self.index, vec![f.title, f.body]);
        parser.set_field_boost(f.title, 2.0);

        let (query, _errors) = parser.parse_query_lenient(query_str);
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            results.push(SearchResult {
                score,
                doc_id: extract_text(&doc, f.doc_id),
                doc_num_id: extract_u64(&doc, f.doc_num_id),
                collection: extract_text(&doc, f.collection),
                path: extract_text(&doc, f.path),
                title: extract_text(&doc, f.title),
                mtime: extract_u64(&doc, f.mtime),
            });
        }

        Ok(results)
    }

    /// Search within a specific collection only.
    pub fn search_in_collection(
        &self,
        query_str: &str,
        collection: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let f = self.fields();
        self.reader.reload()?;
        let searcher = self.reader.searcher();

        let mut parser =
            QueryParser::for_index(&self.index, vec![f.title, f.body]);
        parser.set_field_boost(f.title, 2.0);

        let (user_query, _errors) = parser.parse_query_lenient(query_str);

        // Combine with collection filter.
        let collection_term =
            tantivy::Term::from_field_text(f.collection, collection);
        let collection_query = tantivy::query::TermQuery::new(
            collection_term,
            IndexRecordOption::Basic,
        );
        let combined = tantivy::query::BooleanQuery::new(vec![
            (tantivy::query::Occur::Must, user_query),
            (tantivy::query::Occur::Must, Box::new(collection_query)),
        ]);

        let top_docs =
            searcher.search(&combined, &TopDocs::with_limit(limit))?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            results.push(SearchResult {
                score,
                doc_id: extract_text(&doc, f.doc_id),
                doc_num_id: extract_u64(&doc, f.doc_num_id),
                collection: extract_text(&doc, f.collection),
                path: extract_text(&doc, f.path),
                title: extract_text(&doc, f.title),
                mtime: extract_u64(&doc, f.mtime),
            });
        }

        Ok(results)
    }

    /// Search with BM25 + fuzzy matching combined.
    ///
    /// Creates FuzzyTermQuery with Levenshtein distance 1 for each query
    /// term on the body field, then ORs them with the BM25 query.
    /// Deduplicates results by doc_id.
    pub fn search_fuzzy(
        &self,
        query_str: &str,
        collection: Option<&str>,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let f = self.fields();
        self.reader.reload()?;
        let searcher = self.reader.searcher();

        // BM25 query
        let mut parser =
            QueryParser::for_index(&self.index, vec![f.title, f.body]);
        parser.set_field_boost(f.title, 2.0);
        let (bm25_query, _errors) = parser.parse_query_lenient(query_str);

        // Build fuzzy queries for each significant term
        let terms: Vec<&str> = query_str.split_whitespace().collect();
        let mut should_clauses: Vec<(
            tantivy::query::Occur,
            Box<dyn tantivy::query::Query>,
        )> = vec![(tantivy::query::Occur::Should, bm25_query)];

        for term_str in &terms {
            if term_str.len() >= 3 {
                let term = tantivy::Term::from_field_text(
                    f.body,
                    &term_str.to_lowercase(),
                );
                let fuzzy = tantivy::query::FuzzyTermQuery::new(term, 1, true);
                should_clauses
                    .push((tantivy::query::Occur::Should, Box::new(fuzzy)));
            }
        }

        let combined_query: Box<dyn tantivy::query::Query> =
            Box::new(tantivy::query::BooleanQuery::new(should_clauses));

        // Optionally filter by collection
        let final_query: Box<dyn tantivy::query::Query> = if let Some(coll) =
            collection
        {
            let coll_term = tantivy::Term::from_field_text(f.collection, coll);
            let coll_query = tantivy::query::TermQuery::new(
                coll_term,
                IndexRecordOption::Basic,
            );
            Box::new(tantivy::query::BooleanQuery::new(vec![
                (tantivy::query::Occur::Must, combined_query),
                (tantivy::query::Occur::Must, Box::new(coll_query)),
            ]))
        } else {
            combined_query
        };

        let top_docs =
            searcher.search(&*final_query, &TopDocs::with_limit(limit))?;

        // Deduplicate by doc_id (keep highest score)
        let mut seen = std::collections::HashSet::new();
        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            let doc_id = extract_text(&doc, f.doc_id);
            if seen.insert(doc_id.clone()) {
                results.push(SearchResult {
                    score,
                    doc_id,
                    doc_num_id: extract_u64(&doc, f.doc_num_id),
                    collection: extract_text(&doc, f.collection),
                    path: extract_text(&doc, f.path),
                    title: extract_text(&doc, f.title),
                    mtime: extract_u64(&doc, f.mtime),
                });
            }
        }

        Ok(results)
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}

impl std::fmt::Debug for SearchIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchIndex").finish_non_exhaustive()
    }
}

fn extract_text(doc: &TantivyDocument, field: Field) -> String {
    doc.get_first(field)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn extract_u64(doc: &TantivyDocument, field: Field) -> u64 {
    doc.get_first(field).and_then(|v| v.as_u64()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_search() {
        let idx = SearchIndex::open_in_ram().unwrap();
        let mut writer = idx.writer(15_000_000).unwrap();

        idx.add_document(
            &writer,
            "abc123",
            1,
            "notes",
            "hello.md",
            "Hello World",
            "This is a test document about hello world",
            1000,
        )
        .unwrap();
        idx.add_document(
            &writer,
            "def456",
            2,
            "notes",
            "rust.md",
            "Rust Programming",
            "Rust is a systems programming language",
            2000,
        )
        .unwrap();

        writer.commit().unwrap();

        let results = idx.search("hello world", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "abc123");
        assert_eq!(results[0].doc_num_id, 1);
        assert_eq!(results[0].collection, "notes");
    }

    #[test]
    fn search_in_collection_filters() {
        let idx = SearchIndex::open_in_ram().unwrap();
        let mut writer = idx.writer(15_000_000).unwrap();

        idx.add_document(
            &writer,
            "a",
            1,
            "notes",
            "a.md",
            "Test",
            "hello from notes",
            1000,
        )
        .unwrap();
        idx.add_document(
            &writer,
            "b",
            2,
            "docs",
            "b.md",
            "Test",
            "hello from docs",
            2000,
        )
        .unwrap();

        writer.commit().unwrap();

        let all = idx.search("hello", 10).unwrap();
        assert_eq!(all.len(), 2);

        let notes_only =
            idx.search_in_collection("hello", "notes", 10).unwrap();
        assert_eq!(notes_only.len(), 1);
        assert_eq!(notes_only[0].collection, "notes");
    }

    #[test]
    fn delete_document() {
        let idx = SearchIndex::open_in_ram().unwrap();
        let mut writer = idx.writer(15_000_000).unwrap();

        idx.add_document(
            &writer,
            "abc",
            1,
            "notes",
            "a.md",
            "Test",
            "hello world",
            1000,
        )
        .unwrap();
        writer.commit().unwrap();

        assert_eq!(idx.search("hello", 10).unwrap().len(), 1);

        idx.delete_document(&writer, "abc");
        writer.commit().unwrap();

        assert_eq!(idx.search("hello", 10).unwrap().len(), 0);
    }

    #[test]
    fn delete_collection() {
        let idx = SearchIndex::open_in_ram().unwrap();
        let mut writer = idx.writer(15_000_000).unwrap();

        idx.add_document(&writer, "a", 1, "notes", "a.md", "A", "hello", 1000)
            .unwrap();
        idx.add_document(&writer, "b", 2, "notes", "b.md", "B", "world", 2000)
            .unwrap();
        idx.add_document(&writer, "c", 3, "docs", "c.md", "C", "hello", 3000)
            .unwrap();
        writer.commit().unwrap();

        assert_eq!(idx.search("hello", 10).unwrap().len(), 2);

        idx.delete_collection(&writer, "notes");
        writer.commit().unwrap();

        let results = idx.search("hello", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].collection, "docs");
    }

    #[test]
    fn update_document_replaces() {
        let idx = SearchIndex::open_in_ram().unwrap();
        let mut writer = idx.writer(15_000_000).unwrap();

        idx.add_document(
            &writer,
            "abc",
            1,
            "notes",
            "a.md",
            "Old Title",
            "old content",
            1000,
        )
        .unwrap();
        writer.commit().unwrap();

        // Update with same doc_id should replace.
        idx.add_document(
            &writer,
            "abc",
            1,
            "notes",
            "a.md",
            "New Title",
            "new content",
            2000,
        )
        .unwrap();
        writer.commit().unwrap();

        let results = idx.search("new content", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "New Title");
        assert_eq!(results[0].mtime, 2000);

        // Searching for a unique term from the old doc should find at most
        // one result (the replacement), not two copies.
        let all_results = idx.search("content", 10).unwrap();
        let abc_count =
            all_results.iter().filter(|r| r.doc_id == "abc").count();
        assert_eq!(abc_count, 1, "should have exactly one doc with id 'abc'");
    }

    #[test]
    fn title_boost() {
        let idx = SearchIndex::open_in_ram().unwrap();
        let mut writer = idx.writer(15_000_000).unwrap();

        // "rust" in title only
        idx.add_document(
            &writer,
            "a",
            1,
            "notes",
            "a.md",
            "Rust Guide",
            "programming language guide",
            1000,
        )
        .unwrap();
        // "rust" in body only
        idx.add_document(
            &writer,
            "b",
            2,
            "notes",
            "b.md",
            "Language Guide",
            "rust is a programming language",
            2000,
        )
        .unwrap();
        writer.commit().unwrap();

        let results = idx.search("rust", 10).unwrap();
        assert_eq!(results.len(), 2);
        // Title match should score higher due to 2x boost.
        assert_eq!(results[0].doc_id, "a");
    }

    #[test]
    fn stemming_works() {
        let idx = SearchIndex::open_in_ram().unwrap();
        let mut writer = idx.writer(15_000_000).unwrap();

        idx.add_document(
            &writer,
            "a",
            1,
            "notes",
            "a.md",
            "Running",
            "the runners were running quickly",
            1000,
        )
        .unwrap();
        writer.commit().unwrap();

        // "run" should match "running" and "runners" via stemming.
        let results = idx.search("run", 10).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn disk_persistence() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("tantivy");

        {
            let idx = SearchIndex::open(&dir).unwrap();
            let mut writer = idx.writer(15_000_000).unwrap();
            idx.add_document(
                &writer,
                "abc",
                1,
                "notes",
                "a.md",
                "Test",
                "persistent data",
                1000,
            )
            .unwrap();
            writer.commit().unwrap();
        }

        {
            let idx = SearchIndex::open(&dir).unwrap();
            let results = idx.search("persistent", 10).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "abc");
        }
    }
}
