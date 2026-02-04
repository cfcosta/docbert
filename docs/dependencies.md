# Dependency Reference

## Core Dependencies

### pylate-rs (v1.0.4)

ColBERT inference engine built on HuggingFace Candle.

**Crate**: `pylate-rs`
**Purpose**: Compute ColBERT per-token embeddings and MaxSim similarity scores.

Key types:
- `ColBERT` -- main model struct; holds the language model, projection layer, tokenizer, and config
- `ColbertBuilder` -- fluent builder for constructing ColBERT instances
- `Similarities` -- aggregated MaxSim scores (`data: Vec<Vec<f32>>`)
- `ColbertError` -- error enum wrapping Candle, tokenizer, I/O errors

Key operations:
- `ColBERT::from("model_id").with_device(device).try_into()` -- load a model
- `model.encode(&sentences, is_query)` -- encode text into per-token embeddings (Tensor)
- `model.similarity(&query_embs, &doc_embs)` -- compute aggregated MaxSim scores

Feature flags:
- Default: CPU with auto-vectorization
- `mkl`: Intel MKL backend (AVX/AVX-512 SIMD)
- `accelerate`: Apple Accelerate backend (NEON SIMD)
- `metal`: Apple Metal GPU
- `cuda`: NVIDIA CUDA GPU

Model files (resolved automatically from HuggingFace Hub):
- `model.safetensors` -- main model weights
- `dense.safetensors` -- linear projection layer (768 -> 128 dim)
- `tokenizer.json` -- HuggingFace tokenizer
- `config.json` -- model architecture config
- `dense_config.json` -- projection layer config

### Tantivy (v0.25.0)

Full-text search engine with BM25 scoring.

**Crate**: `tantivy`
**Purpose**: First-stage retrieval via inverted index, BM25 scoring, and fuzzy matching.

Key types:
- `Index` -- the search index (on-disk or in-RAM)
- `Schema` / `SchemaBuilder` -- field definitions
- `IndexWriter` -- write documents, commit to disk
- `IndexReader` / `Searcher` -- read and search
- `QueryParser` -- parse human-readable query strings
- `FuzzyTermQuery` -- Levenshtein distance matching
- `TopDocs` -- collector for top-K results by score

Index persistence:
- `Index::create_in_dir(path, schema)` -- create new on-disk index
- `Index::open_in_dir(path)` -- open existing index
- `MmapDirectory` -- production default, mmap-based I/O

Field options:
- `TEXT` -- tokenized + indexed (BM25)
- `STRING` -- exact match, not tokenized
- `STORED` -- retrievable after search
- `FAST` -- columnar storage for fast access by doc ID
- `INDEXED` -- searchable

### redb (v3.1.0)

Embedded ACID key-value store.

**Crate**: `redb`
**Purpose**: Store collection configuration, document metadata, and ColBERT embeddings.

Key types:
- `Database` -- database handle (create/open)
- `TableDefinition<K, V>` -- typed table definition (compile-time)
- `WriteTransaction` / `ReadTransaction` -- MVCC transactions
- `Table` / `ReadOnlyTable` -- table handles within a transaction

Key operations:
- `Database::create(path)` -- create or open database
- `db.begin_write()` / `db.begin_read()` -- start transactions
- `txn.open_table(definition)` -- open (or create) a table
- `table.insert(key, value)` -- write a key-value pair
- `table.insert_reserve(key, len)` -- pre-allocate for zero-copy write
- `table.get(key)` -- read a value
- `table.iter()` / `table.range(bounds)` -- iterate entries
- `txn.commit()` -- commit transaction

Concurrency: single writer, multiple concurrent readers (MVCC). Readers never block the writer.

Durability:
- `Durability::Immediate` -- fsync on commit (default, safe)
- `Durability::None` -- skip fsync (faster for batch operations, follow with an Immediate commit)

### clap (v4.5.57)

Command-line argument parser.

**Crate**: `clap` with `derive` feature
**Purpose**: Parse CLI arguments into typed Rust structs.

### xdg (v3.0.0)

XDG Base Directory specification.

**Crate**: `xdg`
**Purpose**: Resolve platform-appropriate data, config, and cache directories.

Key operation:
- `xdg::BaseDirectories::with_prefix("docbert")` -- creates a handle that resolves paths like `~/.local/share/docbert/`

## Additional Dependencies (to add)

These are not yet in Cargo.toml but will be needed:

| Crate | Version | Purpose |
|-------|---------|---------|
| `tantivy` | 0.25 | Full-text search (not yet in Cargo.toml) |
| `pylate-rs` | 1.0 | ColBERT embeddings (not yet in Cargo.toml) |
| `serde` | 1.x | Serialization framework |
| `serde_json` | 1.x | JSON output formatting |
| `glob` or `globset` | latest | Pattern matching for `multi-get` |
| `anyhow` or `thiserror` | latest | Error handling |
| `tracing` + `tracing-subscriber` | latest | Structured logging |

## Feature Flag Strategy

The Cargo.toml should expose feature flags that map to pylate-rs backends:

```toml
[features]
default = []
mkl = ["pylate-rs/mkl"]
accelerate = ["pylate-rs/accelerate"]
metal = ["pylate-rs/metal"]
cuda = ["pylate-rs/cuda"]
```

The default build uses CPU with compiler auto-vectorization. Users on Intel can enable `mkl` for AVX-512; macOS users can enable `accelerate` or `metal`.
