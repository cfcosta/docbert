# Using docbert as a Library

docbert can be used as a Rust library for embedding document indexing and semantic search into your own applications. The library exposes all core components while keeping the CLI and MCP server as thin wrappers.

## Installation

Add docbert to your `Cargo.toml`:

```toml
[dependencies]
docbert = { path = "../docbert" }
```

## Quick Start

```rust
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
        println!("#{} [{:.3}] {}/{}", r.doc_id, r.score, r.collection, r.path);
    }

    Ok(())
}
```

## Core Types

### `DataDir`

Resolves and manages the docbert data directory, which contains the config database, embeddings database, and Tantivy index.

```rust
// Use default XDG location (~/.local/share/docbert)
let data_dir = DataDir::resolve(None)?;

// Use explicit path
let data_dir = DataDir::resolve(Some(Path::new("/tmp/myindex")))?;

// Access subdirectories
let config_path = data_dir.config_db();
let embeddings_path = data_dir.embeddings_db();
let tantivy_path = data_dir.tantivy_dir()?;
```

### `ConfigDb`

Manages collections, context strings, document metadata, and settings in a redb database.

```rust
let config_db = ConfigDb::open(&data_dir.config_db())?;

// Register a collection (name -> directory path)
config_db.set_collection("notes", "/home/user/notes")?;

// List collections
let collections = config_db.list_collections()?;

// Attach context to a collection (for MCP display)
config_db.set_context("bert://notes", "Personal notes and memos")?;

// Store/retrieve settings
config_db.set_setting("model_name", "lightonai/GTE-ModernColBERT-v1")?;
let model = config_db.get_setting("model_name")?;
```

### `SearchIndex`

Wraps Tantivy for BM25 full-text search with English stemming.

```rust
// Open on disk
let index = SearchIndex::open(Path::new("/path/to/tantivy"))?;

// In-memory (for testing)
let index = SearchIndex::open_in_ram()?;

// Index a document
let mut writer = index.writer(15_000_000)?;
index.add_document(&writer, "doc123", 1, "notes", "hello.md", "Hello World", "body text", 1000)?;
writer.commit()?;

// Search with BM25
let results = index.search("hello", 10)?;

// Search within a collection
let results = index.search_in_collection("hello", "notes", 10)?;

// Fuzzy search (BM25 + Levenshtein distance 1)
let results = index.search_fuzzy("helo", None, 10)?;
```

### `EmbeddingDb`

Stores and retrieves ColBERT token-level embeddings in a redb database.

```rust
let db = EmbeddingDb::open(Path::new("/path/to/embeddings.db"))?;

// Store an embedding (raw bytes: u32 token count prefix + flat f32 array)
db.store(doc_num_id, &serialized_embedding)?;

// Load an embedding
let bytes = db.load(doc_num_id)?;

// Delete an embedding
db.delete(doc_num_id)?;

// Batch load
let entries = db.batch_load(&[1, 2, 3])?;
```

### `ModelManager`

Manages the ColBERT model lifecycle with lazy loading.

```rust
// Use default model (or DOCBERT_MODEL env var)
let mut model = ModelManager::new();

// Use specific model
let mut model = ModelManager::with_model_id("lightonai/GTE-ModernColBERT-v1".into());

// Configure document length
let mut model = ModelManager::new().with_document_length(512);

// Model loads on first use
let embeddings = model.encode_documents(&["document text".into()])?;
let query_embedding = model.encode_query("search query")?;
```

### `DocumentId`

Generates stable, deterministic document IDs from collection name and path.

```rust
use docbert::DocumentId;

let id = DocumentId::new("notes", "hello.md");
println!("Short ID: #{}", id.short);     // e.g., #a1b2c3
println!("Numeric ID: {}", id.numeric);   // u64 for database keys
```

## Search Pipeline

### Hybrid Search (BM25 + ColBERT)

The default search pipeline combines BM25 retrieval with ColBERT reranking:

```rust
use docbert::search::{SearchParams, execute_search};

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
```

### Semantic-Only Search

Bypasses BM25 and searches purely with ColBERT embeddings:

```rust
use docbert::search::{SemanticSearchParams, execute_semantic_search};

let params = SemanticSearchParams {
    query: "how does garbage collection work".to_string(),
    count: 10,
    min_score: 0.0,
    all: false,
};

let results = execute_semantic_search(&params, &config_db, &embedding_db, &mut model)?;
```

### BM25-Only Search

Fast keyword search without model loading:

```rust
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
// Model is never loaded when bm25_only is true
```

## Indexing Pipeline

### Discovering and Ingesting Files

```rust
use docbert::{walker, ingestion, incremental};

// Discover files in a collection directory
let files = walker::discover_files(Path::new("/home/user/notes"))?;

// Ingest into the Tantivy index
let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
ingestion::ingest_files(&search_index, "notes", &files)?;
```

### Chunking Documents

For long documents, split into chunks before embedding:

```rust
use docbert::chunking::chunk_text;

let chunks = chunk_text("long document text...", 1024);
for chunk in &chunks {
    println!("Chunk: {} chars", chunk.len());
}
```

### Incremental Sync

Detect changes since last sync:

```rust
use docbert::incremental::{diff_collection, DiffAction};

let diff = diff_collection(&config_db, "notes", &files)?;
for action in &diff {
    match action {
        DiffAction::Add { path, .. } => println!("New: {}", path),
        DiffAction::Update { path, .. } => println!("Changed: {}", path),
        DiffAction::Remove { doc_num_id, .. } => println!("Removed: {}", doc_num_id),
    }
}
```

## MCP Server

To run docbert as an MCP server programmatically:

```rust
use docbert::{ConfigDb, DataDir, mcp};

let data_dir = DataDir::resolve(None)?;
let config_db = ConfigDb::open(&data_dir.config_db())?;
let model_id = "lightonai/GTE-ModernColBERT-v1".to_string();

mcp::run_mcp(data_dir, config_db, model_id)?;
```

## Error Handling

All fallible operations return `docbert::Result<T>`, which uses `docbert::Error`:

```rust
use docbert::{Error, Result};

fn my_function() -> Result<()> {
    // Error variants include:
    // - Error::Io(std::io::Error)
    // - Error::Config(String)
    // - Error::NotFound { collection, path }
    // - Error::Tantivy(tantivy::TantivyError)
    // - Error::Redb(redb::Error)
    // - Error::Model(pylate_rs::Error)
    Ok(())
}
```
