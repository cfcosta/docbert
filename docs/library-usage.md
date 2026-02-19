# Using docbert as a Library

docbert can be used as a Rust library for embedding document indexing and semantic search into your own applications. The library exposes all core components while keeping the CLI and MCP server as thin wrappers.

## Installation

Add docbert to your `Cargo.toml`:

```toml
[dependencies]
docbert = { path = "../docbert" }
```

## Quick Start

```rust,no_run
use docbert::{ConfigDb, DataDir, EmbeddingDb, ModelManager, SearchIndex};
use docbert::search::{SearchParams, execute_search};

fn main() -> docbert::Result<()> {
    // Resolve data directory (uses $DOCBERT_DATA_DIR or XDG default)
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

## Core Types

### `DataDir`

Resolves and manages the docbert data directory, which contains the config database, embeddings database, and Tantivy index.

```rust,no_run
use std::path::Path;
use docbert::DataDir;

fn main() -> docbert::Result<()> {
    // Use default XDG location (~/.local/share/docbert)
    let data_dir = DataDir::resolve(None)?;

    // Use explicit path
    let data_dir = DataDir::resolve(Some(Path::new("/tmp/myindex")))?;

    // Access subdirectories
    let config_path = data_dir.config_db();       // config.db path
    let embeddings_path = data_dir.embeddings_db(); // embeddings.db path
    let tantivy_path = data_dir.tantivy_dir()?;    // tantivy/ directory (created if needed)

    Ok(())
}
```

### `ConfigDb`

Manages collections, context strings, document metadata, and settings in a redb database.

```rust,no_run
use docbert::ConfigDb;

fn main() -> docbert::Result<()> {
    let config_db = ConfigDb::open(std::path::Path::new("config.db"))?;

    // Register a collection (name -> directory path)
    config_db.set_collection("notes", "/home/user/notes")?;

    // List collections as (name, path) pairs
    let collections = config_db.list_collections()?;

    // Attach context to a collection (for MCP display)
    config_db.set_context("bert://notes", "Personal notes and memos")?;

    // Store/retrieve settings
    config_db.set_setting("model_name", "lightonai/ColBERT-Zero")?;
    let model = config_db.get_setting("model_name")?; // Option<String>

    Ok(())
}
```

### `SearchIndex`

Wraps Tantivy for BM25 full-text search with English stemming.

```rust,no_run
use std::path::Path;
use docbert::SearchIndex;

fn main() -> docbert::Result<()> {
    // Open on disk (creates directory if needed)
    let index = SearchIndex::open(Path::new("/path/to/tantivy"))?;

    // In-memory (for testing)
    let index = SearchIndex::open_in_ram()?;

    // Index a document (requires a writer)
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

    // Search within a collection
    let results = index.search_in_collection("hello", "notes", 10)?;

    // Fuzzy search (BM25 + Levenshtein distance 1)
    let results = index.search_fuzzy("helo", None, 10)?;

    Ok(())
}
```

### `EmbeddingDb`

Stores and retrieves ColBERT token-level embedding matrices in a redb database.

Each entry stores a matrix of shape `[num_tokens, dimension]` as a flat `f32` array
with a header containing the dimensions.

```rust,no_run
use std::path::Path;
use docbert::EmbeddingDb;

fn main() -> docbert::Result<()> {
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

    // Batch operations (single transaction, more efficient)
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

Manages the ColBERT model lifecycle with lazy loading. The model is downloaded
from HuggingFace Hub on first use and cached locally.

```rust,no_run
use docbert::ModelManager;

fn main() -> docbert::Result<()> {
    // Use default model (or DOCBERT_MODEL env var)
    let mut model = ModelManager::new();

    // Use a specific model
    let mut model = ModelManager::with_model_id("lightonai/ColBERT-Zero".into());

    // Override the document encoding length
    let mut model = ModelManager::new().with_document_length(512);

    // Model downloads on first encode call
    let doc_embeddings = model.encode_documents(&["document text".into()])?;
    // Returns Tensor of shape [batch_size, num_tokens, dimension]

    let query_embedding = model.encode_query("search query")?;
    // Returns Tensor of shape [num_tokens, dimension]

    Ok(())
}
```

### `DocumentId`

Generates stable, deterministic document IDs from collection name and relative path.
The same inputs always produce the same ID.

```rust,ignore
use docbert::DocumentId;

let id = DocumentId::new("notes", "hello.md");
println!("Short ID: {}", id);           // e.g., #a1b2c3
println!("Numeric ID: {}", id.numeric); // u64 for database keys
println!("Short hex: {}", id.short);    // e.g., a1b2c3 (without #)

// Same inputs always produce the same ID
assert_eq!(DocumentId::new("notes", "hello.md"), id);
```

## Search Pipeline

### Hybrid Search (BM25 + ColBERT)

The default search pipeline combines BM25 retrieval with ColBERT reranking:

1. **BM25 retrieval** -- Tantivy keyword search (top 1000 candidates)
2. **ColBERT reranking** -- neural MaxSim scoring of candidates
3. **Score filtering** -- drop results below `min_score`
4. **Limit** -- return at most `count` results

```rust,no_run
use docbert::{SearchIndex, EmbeddingDb, ModelManager};
use docbert::search::{SearchParams, execute_search};

fn main() -> docbert::Result<()> {
    let search_index = SearchIndex::open_in_ram()?;
    let embedding_db = EmbeddingDb::open(std::path::Path::new("emb.db"))?;
    let mut model = ModelManager::new();

    let params = SearchParams {
        query: "rust memory safety".to_string(),
        count: 10,
        collection: Some("docs".to_string()), // None for all collections
        min_score: 0.0,
        bm25_only: false,  // set true to skip ColBERT
        no_fuzzy: false,   // set true to disable fuzzy matching
        all: false,        // set true to return all results (ignore count)
    };

    let results = execute_search(&params, &search_index, &embedding_db, &mut model)?;
    for r in &results {
        println!("{}: {} (score {:.3})", r.rank, r.title, r.score);
    }

    Ok(())
}
```

### Semantic-Only Search

Bypasses BM25 and searches purely with ColBERT embeddings. Scores every
document with stored embeddings, finding semantically related documents
even when they share no keywords with the query.

```rust,no_run
use docbert::{ConfigDb, EmbeddingDb, ModelManager};
use docbert::search::{SemanticSearchParams, execute_semantic_search};

fn main() -> docbert::Result<()> {
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

### BM25-Only Search

Fast keyword search without model loading:

```rust,no_run
use docbert::{SearchIndex, EmbeddingDb, ModelManager};
use docbert::search::{SearchParams, execute_search};

fn main() -> docbert::Result<()> {
    let search_index = SearchIndex::open_in_ram()?;
    let embedding_db = EmbeddingDb::open(std::path::Path::new("emb.db"))?;
    let mut model = ModelManager::new();

    let params = SearchParams {
        query: "error handling".to_string(),
        count: 10,
        collection: None,
        min_score: 0.0,
        bm25_only: true, // skips ColBERT entirely -- model is never loaded
        no_fuzzy: false,
        all: false,
    };

    let results = execute_search(&params, &search_index, &embedding_db, &mut model)?;
    Ok(())
}
```

## Indexing Pipeline

### Discovering and Ingesting Files

```rust,no_run
use std::path::Path;
use docbert::{SearchIndex, walker, ingestion};

fn main() -> docbert::Result<()> {
    // Discover .md and .txt files (skips hidden files/directories)
    let files = walker::discover_files(Path::new("/home/user/notes"))?;

    // Index into Tantivy (requires a writer)
    let search_index = SearchIndex::open_in_ram()?;
    let mut writer = search_index.writer(15_000_000)?;
    let count = ingestion::ingest_files(&search_index, &mut writer, "notes", &files)?;
    println!("Indexed {} files", count);

    Ok(())
}
```

### Chunking Documents

For long documents, split into overlapping chunks before embedding:

```rust,ignore
use docbert::chunking::{chunk_text, chunk_doc_id, ChunkingConfig, DEFAULT_CHUNK_SIZE};

// Split text into chunks of ~1000 characters with 200 char overlap
let chunks = chunk_text("long document text...", 1000, 200);
for chunk in &chunks {
    println!("Chunk {}: {} chars at offset {}", chunk.index, chunk.text.len(), chunk.start_offset);
}

// Generate unique IDs for each chunk
let base_id = 12345u64;
let chunk0_id = chunk_doc_id(base_id, 0); // same as base_id
let chunk1_id = chunk_doc_id(base_id, 1); // unique ID for chunk 1
```

### Incremental Sync

Detect changes since last sync using modification time tracking:

```rust,no_run
use std::path::Path;
use docbert::{ConfigDb, walker};
use docbert::incremental::{diff_collection, store_metadata, DiffResult};

fn main() -> docbert::Result<()> {
    let config_db = ConfigDb::open(Path::new("config.db"))?;
    let files = walker::discover_files(Path::new("/home/user/notes"))?;

    let diff: DiffResult = diff_collection(&config_db, "notes", &files)?;
    println!("New: {}, Changed: {}, Deleted: {}",
        diff.new_files.len(),
        diff.changed_files.len(),
        diff.deleted_ids.len(),
    );

    // After indexing, store metadata so future diffs detect changes
    for file in &diff.new_files {
        store_metadata(&config_db, "notes", file)?;
    }

    Ok(())
}
```

## MCP Server

To run docbert as an MCP server programmatically:

```rust,no_run
use docbert::{ConfigDb, DataDir};
use docbert::model_manager::DEFAULT_MODEL_ID;

fn main() -> docbert::Result<()> {
    let data_dir = DataDir::resolve(None)?;
    let config_db = ConfigDb::open(&data_dir.config_db())?;

    // Blocks until the client disconnects
    docbert::mcp::run_mcp(data_dir, config_db, DEFAULT_MODEL_ID.to_string())?;
    Ok(())
}
```

## Error Handling

All fallible operations return `docbert::Result<T>`, which uses `docbert::Error`:

```rust,ignore
use docbert::{Error, Result};

fn my_function() -> Result<()> {
    // Error variants:
    // - Error::Io(std::io::Error)        -- file I/O
    // - Error::Config(String)            -- configuration / validation
    // - Error::NotFound { kind, name }   -- missing document / collection
    // - Error::DataDir(PathBuf)          -- data directory issues
    // - Error::Tantivy(tantivy::TantivyError) -- search index
    // - Error::Redb(redb::Error)         -- config/embedding database
    // - Error::Colbert(pylate_rs::ColbertError) -- model errors
    // - Error::Candle(candle_core::Error) -- tensor operations
    Ok(())
}
```

## Model Resolution

The ColBERT model ID is resolved from multiple sources, in priority order:

1. `--model` CLI flag (highest priority)
2. `DOCBERT_MODEL` environment variable
3. `model_name` setting in `config.db`
4. Compiled-in default (`lightonai/ColBERT-Zero`)

```rust,no_run
use docbert::ConfigDb;
use docbert::model_manager::{resolve_model, ModelSource};

fn main() -> docbert::Result<()> {
    let config_db = ConfigDb::open(std::path::Path::new("config.db"))?;

    let resolution = resolve_model(&config_db, Some("my/model"))?;
    println!("Model: {} (source: {})", resolution.model_id, resolution.source.as_str());
    // "Model: my/model (source: cli)"

    Ok(())
}
```
