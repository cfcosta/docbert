# `docbert-core` as a library

`docbert-core` is the library for docbert's local retrieval system.

Use it to put docbert's storage, indexing, preparation, and search functions inside your Rust application. Then you do not use the CLI.

This page is about the public library API of `docbert-core`. It does not include the application-only parts, for example:

- the `docbert` CLI command tree
- `docbert web`
- `docbert mcp`
- clap argument parsing, or the runtime code in `crates/docbert`.

For those parts, see the other docs pages instead.

## What `docbert-core` gives you

The public library gives you:

- the local storage types (`DataDir`, `ConfigDb`, `EmbeddingDb`)
- the lexical indexing (`SearchIndex`)
- the model management (`ModelManager`)
- the functions to find and prepare documents (`walker`, `ingestion`, `preparation`, `chunking`)
- the search functions (`search::run`, `search::semantic`, `search::by_mode`)
- the document IDs (`DocumentId`)
- the result enrichment functions (`results::enrich`)
- one error type for all parts (`docbert_core::Error` / `docbert_core::Result`).

The library operates on local files and local state. Your application selects where the data directory is. It also selects when to index or search.

## Add the dependency

For a crate in the same repo, or a local checkout:

```toml
[dependencies]
docbert-core = { path = "../docbert/crates/docbert-core" }
```

You can make another crate or a workspace member. In that crate, add the dependency only where indexing or search is necessary.

## Storage model

A typical setup has these four storage pieces:

- `config.db`
- `embeddings.db`
- `tantivy/`
- one or more collection root directories on disk.

`docbert-core` helps you use those pieces directly, but it does not give the application runtime that `docbert web` and `docbert mcp` add.

## Small search example

This is the smallest library example that shows all the steps. It opens the local storage, makes a query, then does the search.

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

`search::by_mode` uses a `&DataDir`, not a `&EmbeddingDb`. The semantic search reads from the PLAID index file (`<data-dir>/plaid.idx`) internally, not from the embedding rows. The embeddings supply the index only when docbert makes it. If `plaid.idx` is missing, the hybrid and semantic paths both give `Error::PlaidIndexMissing`. To make `plaid.idx`, use the `docbert sync` command. If the embeddings exist, use `docbert reindex` instead.

## Core public types

## `DataDir`

`DataDir` is a small type for the root directory of docbert's local state.

It gives you the paths for:

- `config.db`
- `embeddings.db`
- `plaid.idx`
- `tantivy/`.

It does not find the XDG default paths for you. The application does that. As a library user, you select the root yourself.

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

`ConfigDb` is the primary metadata and configuration storage.

Its public functions include:

- collections
- contexts
- document metadata
- collection Merkle snapshots
- settings
- the LLM settings in storage
- conversations
- document user metadata.

If you use only the retrieval part, the most frequent operations are:

- To open the database
- To add collections
- To list the collections
- To write or read document metadata
- To read settings, for example `model_name`.

```rust,no_run
use docbert_core::ConfigDb;
use docbert_core::incremental::DocumentMetadata;

fn main() -> docbert_core::Result<()> {
    let db = ConfigDb::open(std::path::Path::new("config.db"))?;

    db.set_collection("notes", "/home/user/notes")?;
    db.set_context("bert://notes", "Personal notes")?;
    db.set_setting("model_name", "your-org/your-colbert-model")?;

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

### Conversations and settings are part of the library too

Even if your app does not build a web UI, these storage APIs are part of the public library API:

- `set_conversation_typed`
- `get_conversation_typed`
- `list_conversations_typed`
- `get_persisted_llm_settings`
- `set_persisted_llm_settings`
- `set_document_user_metadata`
- `get_document_user_metadata`.

So you can use `docbert-core` for custom apps. These apps want docbert-compatible conversation or metadata storage, and do not write the storage schema again.

## `SearchIndex`

`SearchIndex` uses Tantivy.

The public functions let you:

- Open an on-disk index
- Open an in-memory index for tests
- Add documents
- Remove documents
- Remove all documents in a collection
- Do a search
- Do a search in one collection
- Do a fuzzy search
- Find documents by collection or path.

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

### Important limit

`SearchIndex` is the lexical index only.

It does not do these tasks:

- the model load
- the semantic reranking
- the collection discovery on the filesystem
- the metadata storage in `config.db`.

You put those parts together yourself when you use the library inside your app.

## `EmbeddingDb`

`EmbeddingDb` keeps ColBERT token-level embedding matrices. The key is a document ID or chunk ID number.

The public API has these functions:

- `open`: opens the storage at a path, or makes a new one.
- `store` / `load` / `remove`: the write, read, and remove functions for one entry.
- `batch_store` / `batch_load` / `batch_remove`: the versions for many entries that use one transaction.
- `list_ids`: gives every stored ID.
- `list_shapes`: gives the `(id, num_tokens, dimension)` triples. It reads these from the entry headers only.
- `list_legacy_ids`: gives the IDs of entries in the pre-1.0 f32 layout, for `docbert clean`.

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

`ModelManager` controls the ColBERT model. It does the model load and the encode operations.

The primary public functions include:

- `ModelManager::new()`
- `ModelManager::with_model_id(...)`
- `with_document_length(...)`
- `runtime_config()`
- `encode_documents(...)`
- `encode_query(...)`.

```rust,no_run
use docbert_core::ModelManager;

fn main() -> docbert_core::Result<()> {
    let mut model = ModelManager::with_model_id(
        "your-org/your-colbert-model".to_string(),
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

### Note on model choice

`resolve_model(...)` is also public through `docbert_core::model_manager`. But it is an application function for the model-choice rules of the CLI and web runtime. These rules use this order: the CLI value, then the env var, then the config, then the built-in value.

If you use the library inside your app directly, you can use it too. But for many applications, we recommend that you select the model directly and make `ModelManager` yourself.

## Document preparation and indexing functions

The library gives the same preparation steps that the application uses.

## Find files with `walker`

Use `walker::discover_files(...)` to find the files it can read in a directory and its subdirectories.

`walker::discover_files` does these operations:

- It reads files with the `.md`, `.txt`, or `.pdf` extension.
- It ignores files and directories that start with a period.
- It obeys Git ignore rules only when the collection root is a Git repo itself.
- Each item it gives has the relative path, the absolute path, and the mtime.

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

## Prepare documents with `preparation`

The primary prepared-document type is `preparation::SearchDocument`.

Usually you make it through one of these functions. The function names have no prefix (`preparation::markdown`, not `prepare_markdown`):

- `preparation::markdown(...)`: gives the small `MarkdownBody` with the title and searchable body. The other functions use it.
- `preparation::uploaded(...)`: makes a full `SearchDocument`. It keeps the raw content so you can ingest or embed it again later.
- `preparation::filesystem(...)`: makes a `SearchDocument`, but does not keep the raw content.
- `preparation::supported_filesystem(...)`: reads a markdown, text, or PDF file from disk, then sends it through `filesystem`.

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

## Load documents with `ingestion`

`ingestion::load_documents(...)` is the usual step from found files to prepared documents.

It gives:

- the documents that it prepared
- the found files that it read
- the load failures.

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

## Write lexical documents with `ingestion`

If you have prepared documents, use `ingest_prepared_documents(...)`.

If you want one function that reads the found files and writes them immediately, use `ingest_files(...)`.

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

The library also includes the chunking and embedding functions.

### Chunking functions

Use `chunking::resolve_config(...)` to get the same chunk-size that the application selects for a model path.

Use `preparation::embedding_chunks(...)` or `preparation::collect_chunks(...)` if you have `SearchDocument` values.

```rust,no_run
use std::path::Path;
use docbert_core::chunking;
use docbert_core::preparation;

fn main() -> docbert_core::Result<()> {
    let config = chunking::resolve_config("lightonai/GTE-ModernColBERT-v1");

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

### Embedding functions

The `embedding` module gives you two frequent operations:

- `embed_documents(...)`: use this to make the embeddings before you select how to keep other state.
- `embed_and_store(...)` / `embed_and_store_in_batches(...)`: use these to write directly into `EmbeddingDb`.

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

## Search functions

The primary public search APIs are in `docbert_core::search`.

## `search::run(...)`

Use this when you want the full set of hybrid-search parameters:

- `bm25_only`
- `no_fuzzy`
- `all`.

Usually, BM25 and semantic retrieval operate at the same time. docbert puts the two result sets together with Reciprocal Rank Fusion. If you set `bm25_only = true`, docbert does not do the semantic search. It also does not touch the PLAID index.

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

Use this when you want semantic-only retrieval of the stored document set.
A PLAID index must exist before you use this. If the PLAID index does not
exist, `search::semantic` gives `Error::PlaidIndexMissing`.

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

Use this for an easy change between `hybrid` and `semantic` modes. It uses the smaller `SearchQuery` type (no `bm25_only`, `no_fuzzy`, or `all`). It uses `run` or `semantic` internally.

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

## Result types

The search functions give `Vec<search::FinalResult>`.

That type contains:

- `rank`
- `score`
- `doc_id`
- `doc_num_id`
- `collection`
- `path`
- `title`
- `best_chunk_doc_id`: an `Option<u64>` that holds the chunk id of the semantic-search match with the highest score. You give it with `doc_num_id` to `ConfigDb::get_chunk_offset_for_doc(doc_num_id, chunk_doc_id)` to find the byte range of the chunk in that document. The value is `None` for BM25-only hits, and for documents that docbert indexed before it recorded chunk offsets.

To attach JSON metadata for your API or UI, use `results::enrich(...)`.

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

## Document IDs and reference functions

`DocumentId` is the stable ID for a document.

```rust,no_run
use docbert_core::DocumentId;

fn main() {
    let id = DocumentId::new("notes", "guide.md");
    println!("display={}", id);
    println!("short={}", id.short);
    println!("numeric={}", id.numeric);
}
```

The `search` module also gives these public reference functions:

- `resolve_by_doc_id(...)`
- `resolve_by_path(...)`
- `resolve_reference(...)`
- `short_doc_id(...)`.

These help if your app accepts docbert-style references, for example `#abc123` or `collection:path`.

## Errors

Most library operations give a `docbert_core::Result<T>`.

The top-level error type is `docbert_core::Error`.

The error variants include:

- `Error::Io`
- `Error::Config`
- `Error::NotFound`
- `Error::DataDir`
- `Error::Tantivy`
- `Error::QueryParse`
- `Error::Heed`: holds a `heed::Error`. It includes every LMDB and heed failure from `ConfigDb` and `EmbeddingDb`.
- `Error::Colbert`
- `Error::Candle`
- `Error::Json`
- `Error::Pdf`
- `Error::Plaid`: holds a `docbert_plaid::PlaidError`.
- `Error::PlaidIndexMissing`: `search::run` and `search::semantic` give this error when `plaid.idx` does not exist yet. You must show this error as a message that tells users to use `docbert sync`.
- `Error::LegacyDatabase { path }`: docbert gives this error when you open a `config.db` or `embeddings.db` in the pre-1.0 redb format. The message tells users to use `docbert clean`, then `docbert sync`.
- `Error::LegacyEmbeddings`: docbert gives this error when an embedding row is in the pre-1.0 f32 layout. You correct it with the same procedure (`docbert clean`, then `docbert sync`).
- `Error::Rkyv`.

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

### Find legacy databases

The storage code also gives `docbert_core::is_legacy_redb_file`:

```rust,no_run
use docbert_core::is_legacy_redb_file;

fn main() -> docbert_core::Result<()> {
    let legacy = is_legacy_redb_file(std::path::Path::new("config.db"))?;
    println!("legacy: {legacy}");
    Ok(())
}
```

It examines whether a file on disk is a pre-1.0 redb database, and it uses the magic bytes for this. It does not do a full open. It gives `false` for missing files, empty files, and LMDB files. `docbert clean` uses it to find legacy `config.db` and `embeddings.db` files. You can do the same to find if you must remove the local state, and not get `Error::LegacyDatabase` from `open`.

## What stays application-only

These limits apply when you use `docbert-core` inside your app.

### In the library

The library has:

- the storage functions and types
- the functions to index, find, and prepare documents
- the functions to load the model and make embeddings
- the search functions
- the result enrichment functions
- the typed errors.

### Outside the library

The application crate (`crates/docbert`) has:

- the code that parses CLI arguments and subcommands
- the web runtime state that continues while the program operates
- the MCP runtime state that continues while the program operates
- the routing definitions and HTTP response shapes
- the higher-level commands for the sync and rebuild operations
- the browser UI and chat runtime code.

This is important when you write examples. For use inside an app, examples must use the library APIs directly, not the CLI.

## Related references

- [`architecture.md`](./architecture.md)
- [`pipeline.md`](./pipeline.md)
- [`storage.md`](./storage.md)
- [`web-api.md`](./web-api.md)
- [`mcp.md`](./mcp.md)
