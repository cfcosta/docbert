# Using `docbert-core` as a library

`docbert-core` is the reusable library behind docbert's local retrieval stack.

Use it when you want to embed docbert's storage, indexing, preparation, and search primitives inside your own Rust application without shelling out to the CLI.

This page focuses on the **library surface** exposed by `docbert-core`. It does **not** document application-only behavior such as:

- the `docbert` CLI command tree
- `docbert web`
- `docbert mcp`
- clap argument parsing or long-lived runtime wiring in `crates/docbert`

For those surfaces, see the other docs pages instead.

## What `docbert-core` gives you

At a high level, the public library gives you:

- local storage helpers (`DataDir`, `ConfigDb`, `EmbeddingDb`)
- lexical indexing (`SearchIndex`)
- model management (`ModelManager`)
- document discovery and preparation (`walker`, `ingestion`, `preparation`, `chunking`)
- search entrypoints (`search::run`, `search::semantic`, `search::by_mode`)
- document identifiers (`DocumentId`)
- result enrichment helpers (`results::enrich`)
- a shared error type (`docbert_core::Error` / `docbert_core::Result`)

The library is centered on **local files and local state**. Your application chooses where the data directory lives and when indexing/searching happens.

## Adding the dependency

For an in-repo consumer or local checkout:

```toml
[dependencies]
docbert-core = { path = "../docbert/crates/docbert-core" }
```

If you expose your own wrapper crate or workspace member, keep the dependency local to the parts that actually need indexing/search functionality.

## Mental model

A typical embedded setup has four storage pieces:

- `config.db`
- `embeddings.db`
- `tantivy/`
- one or more registered collection roots on disk

`docbert-core` helps you work with those layers directly, but it does **not** provide the application runtime that `docbert web` and `docbert mcp` add on top.

## Minimal search example

This is the smallest useful end-to-end library example: open the local stores, resolve a query, and run shared search logic.

```rust,no_run
use std::path::Path;

use docbert_core::{ConfigDb, DataDir, ModelManager, SearchIndex};
use docbert_core::search::{self, SearchMode, SearchQuery};

fn main() -> docbert_core::Result<()> {
    let data_dir = DataDir::new(Path::new("/home/user/.local/share/docbert"));
    let config_db = ConfigDb::open(&data_dir.config_db())?;
    let search_index = SearchIndex::open(&data_dir.tantivy_dir()?)?;
    let mut model = ModelManager::new();

    let request = SearchQuery {
        query: "rust ownership".to_string(),
        collection: None,
        count: 10,
        min_score: 0.0,
    };

    let results = search::by_mode(
        SearchMode::Hybrid,
        &request,
        &search_index,
        &config_db,
        &data_dir,
        &mut model,
    )?;

    for result in results {
        println!(
            "{}. [{}] {}:{} {}",
            result.rank,
            result.score,
            result.collection,
            result.path,
            result.doc_id,
        );
    }

    Ok(())
}
```

Note that `search::by_mode` takes `&DataDir`, not `&EmbeddingDb`. The semantic leg reads from the PLAID index file (`<data-dir>/plaid.idx`) internally, not from the embedding rows — embeddings only feed the index at build time. If `plaid.idx` is missing, the hybrid and semantic paths both fail with `Error::PlaidIndexMissing`; run `docbert sync` (or `docbert reindex` if embeddings already exist) to build it.

## Core public types

## `DataDir`

`DataDir` is a lightweight wrapper around the root directory for docbert's local state.

It gives you paths for:

- `config.db`
- `embeddings.db`
- `plaid.idx`
- `tantivy/`

It does **not** resolve XDG defaults for you. That is application behavior; as a library embedder, you choose the root yourself.

```rust,no_run
use std::path::Path;
use docbert_core::DataDir;

fn main() -> docbert_core::Result<()> {
    let data_dir = DataDir::new(Path::new("/tmp/my-docbert-state"));

    let config = data_dir.config_db();
    let embeddings = data_dir.embeddings_db();
    let plaid_index = data_dir.plaid_index();
    let tantivy = data_dir.tantivy_dir()?; // creates tantivy/ if needed

    println!("{}", config.display());
    println!("{}", embeddings.display());
    println!("{}", plaid_index.display());
    println!("{}", tantivy.display());
    Ok(())
}
```

## `ConfigDb`

`ConfigDb` is the main metadata/configuration store.

Its public helpers cover:

- collections
- contexts
- document metadata
- collection Merkle snapshots
- settings
- persisted LLM settings
- conversations
- document user metadata

If you are embedding only retrieval, the most common operations are:

- opening the DB
- registering collections
- listing collections
- storing/retrieving document metadata
- reading settings such as `model_name`

```rust,no_run
use docbert_core::ConfigDb;
use docbert_core::incremental::DocumentMetadata;

fn main() -> docbert_core::Result<()> {
    let db = ConfigDb::open(std::path::Path::new("config.db"))?;

    db.set_collection("notes", "/home/user/notes")?;
    db.set_context("bert://notes", "Personal notes")?;
    db.set_setting("model_name", "lightonai/LateOn")?;

    let doc_meta = DocumentMetadata {
        collection: "notes".to_string(),
        relative_path: "guide.md".to_string(),
        mtime: 1_700_000_000,
    };
    db.set_document_metadata_typed(42, &doc_meta)?;

    let collections = db.list_collections()?;
    let model_name = db.get_setting("model_name")?;
    let loaded_meta = db.get_document_metadata_typed(42)?;

    println!("collections: {}", collections.len());
    println!("model: {:?}", model_name);
    println!("doc meta exists: {}", loaded_meta.is_some());
    Ok(())
}
```

### Conversations and settings are library-visible too

Even if your app is not building a web UI, note that these storage APIs are part of the public library surface:

- `set_conversation_typed`
- `get_conversation_typed`
- `list_conversations_typed`
- `get_persisted_llm_settings`
- `set_persisted_llm_settings`
- `set_document_user_metadata`
- `get_document_user_metadata`

That makes `docbert-core` usable for custom apps that want docbert-compatible conversation or metadata persistence without reimplementing the storage schema.

## `SearchIndex`

`SearchIndex` wraps Tantivy.

Public capabilities include:

- opening an on-disk index
- opening an in-memory index for tests
- adding documents
- deleting documents
- deleting all documents in a collection
- plain search
- collection-scoped search
- fuzzy search
- lookup by collection/path

```rust,no_run
use docbert_core::SearchIndex;

fn main() -> docbert_core::Result<()> {
    let index = SearchIndex::open_in_ram()?;
    let mut writer = index.writer(15_000_000)?;

    index.add_document(
        &writer,
        "#abc123",
        42,
        "notes",
        "hello.md",
        "Hello",
        "hello from rust",
        1_700_000_000,
    )?;
    writer.commit()?;

    let results = index.search("hello", 10)?;
    println!("{} result(s)", results.len());
    Ok(())
}
```

### Important boundary

`SearchIndex` is the lexical index only.

It does **not** handle:

- model loading
- semantic reranking by itself
- collection discovery from the filesystem
- metadata persistence in `config.db`

You compose those layers yourself when embedding the library.

## `EmbeddingDb`

`EmbeddingDb` stores ColBERT token-level embedding matrices keyed by numeric document or chunk ID.

The public API supports:

- `store` / `load`
- `batch_store` / `batch_load` / `batch_remove`
- `list_ids`
- document-family helpers used by chunked embeddings

```rust,no_run
use docbert_core::EmbeddingDb;

fn main() -> docbert_core::Result<()> {
    let db = EmbeddingDb::open(std::path::Path::new("embeddings.db"))?;

    db.store(42, 2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

    if let Some(matrix) = db.load(42)? {
        println!("tokens={}, dim={}", matrix.num_tokens, matrix.dimension);
    }

    Ok(())
}
```

## `ModelManager`

`ModelManager` owns the ColBERT model lifecycle.

Common public entrypoints include:

- `ModelManager::new()`
- `ModelManager::with_model_id(...)`
- `with_document_length(...)`
- `runtime_config()`
- `encode_documents(...)`
- `encode_query(...)`

```rust,no_run
use docbert_core::ModelManager;

fn main() -> docbert_core::Result<()> {
    let mut model = ModelManager::with_model_id(
        "lightonai/LateOn".to_string(),
    )
    .with_document_length(512);

    let runtime = model.runtime_config()?;
    println!("device={}", runtime.device);

    let docs = vec!["document text".to_string()];
    let _doc_embeddings = model.encode_documents(&docs)?;
    let _query_embedding = model.encode_query("search query")?;
    Ok(())
}
```

### Note on model resolution

`resolve_model(...)` is also public via `docbert_core::model_manager`, but it is an **application-facing convenience** for the CLI/web runtime precedence rules (CLI override, env var, config, default).

If you are embedding the library directly, you can use it, but many applications will be better served by choosing their model explicitly and constructing `ModelManager` themselves.

## Document preparation and indexing primitives

The library exposes the same core preparation steps that the application uses under the hood.

## Discovery with `walker`

Use `walker::discover_files(...)` to recursively discover supported files.

Current behavior includes:

- supported extensions: `.md`, `.txt`, `.pdf`
- hidden files/directories are skipped
- Git ignore rules are respected only when the collection root is itself a Git repo
- returned items include relative path, absolute path, and mtime

```rust,no_run
use std::path::Path;
use docbert_core::walker;

fn main() -> docbert_core::Result<()> {
    let files = walker::discover_files(Path::new("/home/user/notes"))?;
    for file in files {
        println!("{}", file.relative_path.display());
    }
    Ok(())
}
```

## Preparing documents with `preparation`

The main shared prepared-document type is `preparation::SearchDocument`.

You typically build it through one of these helpers (note the flat,
unprefixed names — `preparation::markdown`, not `prepare_markdown`):

- `preparation::markdown(...)` — returns the lightweight `MarkdownBody` (title + searchable body); used as a building block by the other helpers
- `preparation::uploaded(...)` — builds a full `SearchDocument` and keeps the raw content for later ingest/re-embedding
- `preparation::filesystem(...)` — builds a `SearchDocument` without retaining the raw content
- `preparation::supported_filesystem(...)` — reads a supported file from disk (markdown/text/PDF) and feeds it through `filesystem`

```rust,no_run
use std::path::Path;
use docbert_core::preparation;

fn main() -> docbert_core::Result<()> {
    let prepared = preparation::markdown(
        Path::new("guide.md"),
        "---\ntitle: ignored\n---\n# Guide\n\nBody",
    );
    assert_eq!(prepared.title, "Guide");
    assert_eq!(prepared.searchable_body, "# Guide\n\nBody");

    let uploaded = preparation::uploaded(
        "notes",
        "guide.md",
        "# Guide\n\nBody",
        None,
        0,
    );
    assert!(uploaded.raw_content.is_some());

    let filesystem = preparation::filesystem(
        "notes",
        Path::new("guide.md"),
        "# Guide\n\nBody",
        1_700_000_000,
    );
    assert!(filesystem.raw_content.is_none());

    Ok(())
}
```

## Loading documents with `ingestion`

`ingestion::load_documents(...)` is the usual bridge from discovered files to prepared documents.

It returns:

- successfully prepared documents
- successfully loaded discovered files
- load failures

```rust,no_run
use std::path::Path;
use docbert_core::{ingestion, walker};

fn main() -> docbert_core::Result<()> {
    let files = walker::discover_files(Path::new("/home/user/notes"))?;
    let loaded = ingestion::load_documents("notes", &files);

    println!("documents={}", loaded.documents.len());
    println!("failures={}", loaded.failures.len());
    Ok(())
}
```

## Writing lexical documents with `ingestion`

If you already have prepared documents, use `ingest_prepared_documents(...)`.

If you want the convenience wrapper that reads discovered files and writes them immediately, use `ingest_files(...)`.

```rust,no_run
use docbert_core::{SearchIndex, ingestion, walker};

fn main() -> docbert_core::Result<()> {
    let files = walker::discover_files(std::path::Path::new("/home/user/notes"))?;
    let loaded = ingestion::load_documents("notes", &files);

    let index = SearchIndex::open_in_ram()?;
    let mut writer = index.writer(15_000_000)?;
    let count = ingestion::ingest_prepared_documents(
        &index,
        &mut writer,
        "notes",
        &loaded.documents,
    )?;

    println!("indexed={count}");
    Ok(())
}
```

## Chunking and embedding

The chunking and embedding layers are library-accessible too.

### Chunking helpers

Use `chunking::resolve_config(...)` if you want the same chunk-size selection logic the application uses for a given model path.

Use `preparation::embedding_chunks(...)` or `preparation::collect_chunks(...)` if you already have `SearchDocument` values.

```rust,no_run
use std::path::Path;
use docbert_core::chunking;
use docbert_core::preparation;

fn main() -> docbert_core::Result<()> {
    let config = chunking::resolve_config("lightonai/LateOn");

    let doc = preparation::filesystem(
        "notes",
        Path::new("long.md"),
        "Long document text...",
        0,
    );

    let chunks = preparation::collect_chunks(&[doc], config, |_| {});
    println!("chunks={}", chunks.len());
    Ok(())
}
```

### Embedding helpers

The `embedding` module gives you two common library-level workflows:

- `embed_documents(...)` if you want to generate embeddings before deciding how to persist other state
- `embed_and_store(...)` / `embed_and_store_in_batches(...)` if you want to write directly into `EmbeddingDb`

```rust,no_run
use docbert_core::{EmbeddingDb, ModelManager};
use docbert_core::embedding;

fn main() -> docbert_core::Result<()> {
    let db = EmbeddingDb::open(std::path::Path::new("embeddings.db"))?;
    let mut model = ModelManager::new();

    let docs = vec![
        (1_u64, "first document".to_string()),
        (2_u64, "second document".to_string()),
    ];

    let entries = embedding::embed_documents(&mut model, docs.clone())?;
    db.batch_store(&entries)?;

    let written = embedding::embed_and_store(&mut model, &db, docs)?;
    println!("written={written}");
    Ok(())
}
```

## Search entrypoints

The main public search APIs live in `docbert_core::search`.

## `search::run(...)`

Use this when you want the full hybrid-search parameter surface, including:

- `bm25_only`
- `no_fuzzy`
- `all`

By default, BM25 and semantic retrieval run in parallel and are fused with
Reciprocal Rank Fusion. Setting `bm25_only = true` skips the semantic leg
entirely and does not touch the PLAID index.

```rust,no_run
use docbert_core::{ConfigDb, DataDir, ModelManager, SearchIndex};
use docbert_core::search::{self, SearchParams};

fn main() -> docbert_core::Result<()> {
    let data_dir = DataDir::new(std::path::Path::new("/tmp/docbert-state"));
    let config_db = ConfigDb::open(&data_dir.config_db())?;
    let search_index = SearchIndex::open_in_ram()?;
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

    let _results = search::run(
        &params,
        &search_index,
        &config_db,
        &data_dir,
        &mut model,
    )?;
    Ok(())
}
```

## `search::semantic(...)`

Use this when you want semantic-only retrieval over the stored document set.
It requires a prebuilt PLAID index and returns `Error::PlaidIndexMissing`
otherwise.

```rust,no_run
use docbert_core::{ConfigDb, DataDir, ModelManager};
use docbert_core::search::{self, SemanticSearchParams};

fn main() -> docbert_core::Result<()> {
    let data_dir = DataDir::new(std::path::Path::new("/tmp/docbert-state"));
    let config_db = ConfigDb::open(&data_dir.config_db())?;
    let mut model = ModelManager::new();

    let params = SemanticSearchParams {
        query: "memory management".to_string(),
        collection: None,
        count: 10,
        min_score: 0.0,
        all: false,
    };

    let _results = search::semantic(&params, &config_db, &data_dir, &mut model)?;
    Ok(())
}
```

## `search::by_mode(...)`

Use this when your app wants a simpler mode-switching wrapper around `hybrid` vs `semantic` behavior. It takes the smaller `SearchQuery` shape (no `bm25_only`/`no_fuzzy`/`all`) and dispatches to `run` or `semantic` internally.

```rust,no_run
use docbert_core::{ConfigDb, DataDir, ModelManager, SearchIndex};
use docbert_core::search::{self, SearchMode, SearchQuery};

fn main() -> docbert_core::Result<()> {
    let data_dir = DataDir::new(std::path::Path::new("/tmp/docbert-state"));
    let config_db = ConfigDb::open(&data_dir.config_db())?;
    let search_index = SearchIndex::open_in_ram()?;
    let mut model = ModelManager::new();

    let query = SearchQuery {
        query: "rust".to_string(),
        collection: Some("notes".to_string()),
        count: 5,
        min_score: 0.0,
    };

    let _results = search::by_mode(
        SearchMode::Hybrid,
        &query,
        &search_index,
        &config_db,
        &data_dir,
        &mut model,
    )?;
    Ok(())
}
```

## Result shapes

The shared search functions return `Vec<search::FinalResult>`.

That type contains:

- `rank`
- `score`
- `doc_id`
- `doc_num_id`
- `collection`
- `path`
- `title`
- `best_chunk_doc_id` — `Option<u64>` carrying the chunk id of the best-scoring semantic-leg match, used to look up a chunk's byte range via `ConfigDb::get_chunk_offset`. `None` for BM25-only hits and for documents indexed before chunk offsets were tracked.

If you want to attach JSON metadata for your own API/UI surface, use `results::enrich(...)`.

```rust,no_run
use docbert_core::{ConfigDb, DataDir, ModelManager, SearchIndex};
use docbert_core::results::enrich;
use docbert_core::search::{self, SearchMode, SearchQuery};

fn main() -> docbert_core::Result<()> {
    let data_dir = DataDir::new(std::path::Path::new("/tmp/docbert-state"));
    let config_db = ConfigDb::open(&data_dir.config_db())?;
    let search_index = SearchIndex::open_in_ram()?;
    let mut model = ModelManager::new();

    let results = search::by_mode(
        SearchMode::Hybrid,
        &SearchQuery {
            query: "rust".to_string(),
            collection: None,
            count: 5,
            min_score: 0.0,
        },
        &search_index,
        &config_db,
        &data_dir,
        &mut model,
    )?;

    let hits = enrich(results, |doc_num_id| {
        config_db.get_document_user_metadata(doc_num_id).ok().flatten()
    });

    for hit in hits {
        println!("{} {:?}", hit.path, hit.metadata);
    }

    Ok(())
}
```

## Document IDs and reference helpers

`DocumentId` is the shared stable identifier for documents.

```rust,no_run
use docbert_core::DocumentId;

fn main() {
    let id = DocumentId::new("notes", "guide.md");
    println!("display={}", id);
    println!("short={}", id.short);
    println!("numeric={}", id.numeric);
}
```

The `search` module also exposes library-visible reference helpers such as:

- `resolve_by_doc_id(...)`
- `resolve_by_path(...)`
- `resolve_reference(...)`
- `short_doc_id(...)`

These are useful if your embedded app accepts docbert-style references like `#abc123` or `collection:path`.

## Error handling

Most fallible library operations return `docbert_core::Result<T>`.

The top-level error type is `docbert_core::Error`.

Common variants include:

- `Error::Io`
- `Error::Config`
- `Error::NotFound`
- `Error::DataDir`
- `Error::Tantivy`
- `Error::QueryParse`
- `Error::Redb*`
- `Error::Colbert`
- `Error::Candle`
- `Error::Json`
- `Error::Pdf`
- `Error::Plaid` — wraps `docbert_plaid::PlaidError`
- `Error::PlaidIndexMissing` — sentinel raised by `search::run` and `search::semantic` when `plaid.idx` has not been built yet; surface as a "run `docbert sync`" message
- `Error::Rkyv`

```rust,no_run
use docbert_core::{Error, Result};

fn do_work() -> Result<()> {
    let err = Error::Config("example configuration problem".to_string());
    eprintln!("{err}");
    Ok(())
}

fn main() -> Result<()> {
    do_work()
}
```

## What stays application-only

When embedding `docbert-core`, keep these boundaries in mind.

### In the library

The library owns:

- storage primitives
- indexing/discovery/preparation primitives
- model loading and embedding helpers
- search functions
- result enrichment helpers
- typed errors

### Outside the library

The application crate (`crates/docbert`) owns:

- CLI argument parsing and subcommands
- long-lived web runtime state
- long-lived MCP runtime state
- route definitions and HTTP response shapes
- higher-level sync/rebuild orchestration convenience commands
- browser UI and chat runtime logic

That distinction matters when writing examples. If you are documenting embedded usage, prefer showing direct calls to library APIs rather than invoking CLI concepts indirectly.

## Related references

- [`architecture.md`](./architecture.md)
- [`pipeline.md`](./pipeline.md)
- [`storage.md`](./storage.md)
- [`web-api.md`](./web-api.md)
- [`mcp.md`](./mcp.md)
