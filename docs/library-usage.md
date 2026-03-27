# Using docbert-core as a library

If you want to embed docbert in another Rust application, the CLI is not doing anything magical. It uses the same library types exposed by the `docbert-core` crate.

## Installation

Add docbert-core to your `Cargo.toml`:

```toml
[dependencies]
docbert-core = { path = "../docbert/crates/docbert-core" }
```

## Quick start

```rust,no_run
use docbert_core::{ConfigDb, DataDir, EmbeddingDb, ModelManager, SearchIndex};
use docbert_core::search::{SearchParams, execute_search};

fn main() -> docbert_core::Result<()> {
    // Resolve the data directory from $DOCBERT_DATA_DIR or XDG defaults.
    let data_dir = DataDir::resolve(None)?;
    let config_db = ConfigDb::open(&data_dir.config_db())?;
    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let embedding_db = EmbeddingDb::open(&data_dir.embeddings_db())?;
    let mut model = ModelManager::new();

    let params = SearchParams {
        query: "rust ownership".to_string(),
        count: 10,
        collection: None,
        min_score: 0.0,
        bm25_only: false,
        no_fuzzy: false,
        all: false,
    };

    let results = execute_search(&params, &search_index, &embedding_db, &mut model)?;
    for r in &results {
        println!("{} [{:.3}] {}/{}", r.doc_id, r.score, r.collection, r.path);
    }

    Ok(())
}
```

## Core types

### `DataDir`

`DataDir` resolves the docbert data directory and gives you paths for the config database, embeddings database, and Tantivy index.

```rust,no_run
use std::path::Path;
use docbert_core::DataDir;

fn main() -> docbert_core::Result<()> {
    // Use the default XDG location (~/.local/share/docbert)
    let data_dir = DataDir::resolve(None)?;

    // Or use an explicit path
    let data_dir = DataDir::resolve(Some(Path::new("/tmp/myindex")))?;

    // Access subpaths
    let config_path = data_dir.config_db();          // config.db
    let embeddings_path = data_dir.embeddings_db();  // embeddings.db
    let tantivy_path = data_dir.tantivy_dir()?;      // tantivy/ (created if needed)

    Ok(())
}
```

### `ConfigDb`

`ConfigDb` manages collections, context strings, document metadata, and settings in a redb database.

```rust,no_run
use docbert_core::ConfigDb;

fn main() -> docbert_core::Result<()> {
    let config_db = ConfigDb::open(std::path::Path::new("config.db"))?;

    // Register a collection (name -> directory path)
    config_db.set_collection("notes", "/home/user/notes")?;

    // List collections as (name, path) pairs
    let collections = config_db.list_collections()?;

    // Attach context to a collection for MCP display
    config_db.set_context("bert://notes", "Personal notes and memos")?;

    // Store and read settings
    config_db.set_setting("model_name", "lightonai/ColBERT-Zero")?;
    let model = config_db.get_setting("model_name")?; // Option<String>

    Ok(())
}
```

### `SearchIndex`

`SearchIndex` wraps Tantivy for BM25 full-text search with English stemming.

```rust,no_run
use std::path::Path;
use docbert_core::SearchIndex;

fn main() -> docbert_core::Result<()> {
    // Open on disk (creates the directory if needed)
    let index = SearchIndex::open(Path::new("/path/to/tantivy"))?;

    // Or open an in-memory index for testing
    let index = SearchIndex::open_in_ram()?;

    // Index a document
    let mut writer = index.writer(15_000_000)?; // 15 MB memory budget
    index.add_document(
        &mut writer,
        "#a1b2c3",          // doc_id (short hex with # prefix)
        12345,              // doc_num_id (numeric ID)
        "notes",            // collection
        "hello.md",         // relative path
        "Hello World",      // title
        "body text here",   // body (indexed but not stored)
        1700000000,         // mtime (Unix timestamp)
    )?;
    writer.commit()?;

    // Search with BM25
    let results = index.search("hello", 10)?;

    // Search within one collection
    let results = index.search_in_collection("hello", "notes", 10)?;

    // Fuzzy search (BM25 + Levenshtein distance 1)
    let results = index.search_fuzzy("helo", None, 10)?;

    Ok(())
}
```

### `EmbeddingDb`

`EmbeddingDb` stores ColBERT token-level embedding matrices in redb.

Each entry is a flat `f32` array with a small header that records the dimensions.

```rust,no_run
use std::path::Path;
use docbert_core::EmbeddingDb;

fn main() -> docbert_core::Result<()> {
    let db = EmbeddingDb::open(Path::new("embeddings.db"))?;

    // Store an embedding matrix (2 tokens, 3 dimensions = 6 floats)
    db.store(42, 2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

    // Load an embedding matrix
    if let Some(matrix) = db.load(42)? {
        println!("tokens={}, dim={}", matrix.num_tokens, matrix.dimension);
        let token0 = matrix.token_embedding(0); // &[f32] of length `dimension`
    }

    // Remove an embedding (returns true if it existed)
    db.remove(42)?;

    // Batch operations are cheaper because they share a transaction
    db.batch_store(&[
        (1, 1, 2, vec![1.0, 2.0]),
        (2, 1, 2, vec![3.0, 4.0]),
    ])?;
    let results = db.batch_load(&[1, 2, 999])?; // preserves order, None for missing
    db.batch_remove(&[1, 2])?;

    // List all stored document IDs
    let ids = db.list_ids()?;

    Ok(())
}
```

### `ModelManager`

`ModelManager` lazily loads the ColBERT model. The first encode call downloads the model from HuggingFace Hub if it is not already cached.

By default, documents are encoded at 519 tokens. Call `with_document_length(...)` if you need a different value.

```rust,no_run
use docbert_core::ModelManager;

fn main() -> docbert_core::Result<()> {
    // Use the default model, or DOCBERT_MODEL if it is set
    let mut model = ModelManager::new();

    // Use a specific model
    let mut model = ModelManager::with_model_id("lightonai/ColBERT-Zero".into());

    // Override the document encoding length
    let mut model = ModelManager::new().with_document_length(512);

    // The model is loaded on first encode
    let doc_embeddings = model.encode_documents(&["document text".into()])?;
    // Returns a Tensor of shape [batch_size, num_tokens, dimension]

    let query_embedding = model.encode_query("search query")?;
    // Returns a Tensor of shape [num_tokens, dimension]

    Ok(())
}
```

### `DocumentId`

`DocumentId` generates a stable ID from the collection name and relative path. The same inputs always produce the same ID.

```rust,ignore
use docbert_core::DocumentId;

let id = DocumentId::new("notes", "hello.md");
println!("Short ID: {}", id);           // e.g. #a1b2c3
println!("Numeric ID: {}", id.numeric); // u64 for database keys
println!("Short hex: {}", id.short);    // e.g. a1b2c3 without #

assert_eq!(DocumentId::new("notes", "hello.md"), id);
```

## Search pipeline

### Hybrid search (BM25 + ColBERT)

The default search path looks like this:

1. BM25 retrieval through Tantivy, usually top 1000 candidates
2. ColBERT reranking over those candidates
3. Score filtering with `min_score`
4. Result limiting with `count`

```rust,no_run
use docbert_core::{SearchIndex, EmbeddingDb, ModelManager};
use docbert_core::search::{SearchParams, execute_search};

fn main() -> docbert_core::Result<()> {
    let search_index = SearchIndex::open_in_ram()?;
    let embedding_db = EmbeddingDb::open(std::path::Path::new("emb.db"))?;
    let mut model = ModelManager::new();

    let params = SearchParams {
        query: "rust memory safety".to_string(),
        count: 10,
        collection: Some("docs".to_string()),
        min_score: 0.0,
        bm25_only: false,
        no_fuzzy: false,
        all: false,
    };

    let results = execute_search(&params, &search_index, &embedding_db, &mut model)?;
    for r in &results {
        println!("{}: {} (score {:.3})", r.rank, r.title, r.score);
    }

    Ok(())
}
```

### Semantic-only search

Semantic-only search skips BM25 and scores every stored embedding. That can surface related documents even when the query shares few or no keywords with them.

```rust,no_run
use docbert_core::{ConfigDb, EmbeddingDb, ModelManager};
use docbert_core::search::{SemanticSearchParams, execute_semantic_search};

fn main() -> docbert_core::Result<()> {
    let config_db = ConfigDb::open(std::path::Path::new("config.db"))?;
    let embedding_db = EmbeddingDb::open(std::path::Path::new("emb.db"))?;
    let mut model = ModelManager::new();

    let params = SemanticSearchParams {
        query: "how does garbage collection work".to_string(),
        count: 10,
        min_score: 0.0,
        all: false,
    };

    let results = execute_semantic_search(&params, &config_db, &embedding_db, &mut model)?;
    Ok(())
}
```

### BM25-only search

If you only want fast keyword search, set `bm25_only` and skip model loading entirely.

```rust,no_run
use docbert_core::{SearchIndex, EmbeddingDb, ModelManager};
use docbert_core::search::{SearchParams, execute_search};

fn main() -> docbert_core::Result<()> {
    let search_index = SearchIndex::open_in_ram()?;
    let embedding_db = EmbeddingDb::open(std::path::Path::new("emb.db"))?;
    let mut model = ModelManager::new();

    let params = SearchParams {
        query: "error handling".to_string(),
        count: 10,
        collection: None,
        min_score: 0.0,
        bm25_only: true,
        no_fuzzy: false,
        all: false,
    };

    let results = execute_search(&params, &search_index, &embedding_db, &mut model)?;
    Ok(())
}
```

## Indexing pipeline

### Discovering and ingesting files

```rust,no_run
use std::path::Path;
use docbert_core::{SearchIndex, walker, ingestion};

fn main() -> docbert_core::Result<()> {
    // Discover .md and .txt files; hidden files and directories are skipped
    let files = walker::discover_files(Path::new("/home/user/notes"))?;

    // Index into Tantivy
    let search_index = SearchIndex::open_in_ram()?;
    let mut writer = search_index.writer(15_000_000)?;
    let count = ingestion::ingest_files(&search_index, &mut writer, "notes", &files)?;
    println!("Indexed {} files", count);

    Ok(())
}
```

### Chunking documents

Long documents can be split into overlapping chunks before embedding.

```rust,ignore
use docbert_core::chunking::{chunk_text, chunk_doc_id, ChunkingConfig, DEFAULT_CHUNK_SIZE};

// Split text into chunks of about 1000 characters with 200 chars of overlap
let chunks = chunk_text("long document text...", 1000, 200);
for chunk in &chunks {
    println!("Chunk {}: {} chars at offset {}", chunk.index, chunk.text.len(), chunk.start_offset);
}

// Generate unique IDs for each chunk
let base_id = 12345u64;
let chunk0_id = chunk_doc_id(base_id, 0); // same as base_id
let chunk1_id = chunk_doc_id(base_id, 1); // unique ID for chunk 1
```

### Incremental sync

Incremental sync uses modification times to find what changed since the last run.

```rust,no_run
use std::path::Path;
use docbert_core::{ConfigDb, walker};
use docbert_core::incremental::{diff_collection, store_metadata, DiffResult};

fn main() -> docbert_core::Result<()> {
    let config_db = ConfigDb::open(Path::new("config.db"))?;
    let files = walker::discover_files(Path::new("/home/user/notes"))?;

    let diff: DiffResult = diff_collection(&config_db, "notes", &files)?;
    println!(
        "New: {}, Changed: {}, Deleted: {}",
        diff.new_files.len(),
        diff.changed_files.len(),
        diff.deleted_ids.len(),
    );

    // After indexing, store metadata so future runs can diff correctly
    for file in &diff.new_files {
        store_metadata(&config_db, "notes", file)?;
    }

    Ok(())
}
```

## MCP server

The MCP server is part of the `docbert` CLI binary, not the `docbert-core` library.
Run it with `docbert mcp` or configure it as an MCP server in your editor.

## Error handling

Most fallible operations return `docbert_core::Result<T>`, which wraps `docbert_core::Error`.

```rust,ignore
use docbert_core::{Error, Result};

fn my_function() -> Result<()> {
    // Common variants:
    // - Error::Io(std::io::Error)                 // file I/O
    // - Error::Config(String)                     // configuration or validation
    // - Error::NotFound { kind, name }            // missing document or collection
    // - Error::DataDir(PathBuf)                   // data directory issues
    // - Error::Tantivy(tantivy::TantivyError)     // search index errors
    // - Error::Redb(redb::Error)                  // config or embedding database errors
    // - Error::Colbert(pylate_rs::ColbertError)   // model errors
    // - Error::Candle(candle_core::Error)         // tensor operation errors
    Ok(())
}
```

## Model resolution

The ColBERT model ID is resolved in this order:

1. `--model` CLI flag
2. `DOCBERT_MODEL` environment variable
3. `model_name` stored in `config.db`
4. compiled-in default: `lightonai/ColBERT-Zero`

```rust,no_run
use docbert_core::ConfigDb;
use docbert_core::model_manager::{resolve_model, ModelSource};

fn main() -> docbert_core::Result<()> {
    let config_db = ConfigDb::open(std::path::Path::new("config.db"))?;

    let resolution = resolve_model(&config_db, Some("my/model"))?;
    println!("Model: {} (source: {})", resolution.model_id, resolution.source.as_str());
    // Model: my/model (source: cli)

    Ok(())
}
```
