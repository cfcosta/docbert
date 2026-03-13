# Dependency reference

These are the crates that matter most when you work on indexing, search, and model loading in docbert.

## Core dependencies

### pylate-rs (`1.0.4`)

ColBERT inference engine built on HuggingFace Candle.

- Crate: `pylate-rs`
- Used for: computing ColBERT token embeddings and MaxSim similarity scores

Key types:

- `ColBERT`: main model struct; holds the language model, projection layer, tokenizer, and config
- `ColbertBuilder`: builder used to construct `ColBERT` instances
- `Similarities`: aggregated MaxSim scores (`data: Vec<Vec<f32>>`)
- `ColbertError`: error enum that wraps Candle, tokenizer, and I/O errors

Key operations:

- `ColBERT::from("model_id").with_device(device).try_into()`: load a model
- `model.encode(&sentences, is_query)`: encode text into per-token embeddings (`Tensor`)
- `model.similarity(&query_embs, &doc_embs)`: compute aggregated MaxSim scores

Feature flags:

- default: CPU with compiler auto-vectorization
- `mkl`: Intel MKL backend, useful on AVX/AVX-512 systems
- `accelerate`: Apple Accelerate backend
- `metal`: Apple Metal GPU backend
- `cuda`: NVIDIA CUDA GPU backend

Model files are resolved automatically from HuggingFace Hub or a local export:

- `model.safetensors`: main model weights
- `tokenizer.json`: HuggingFace tokenizer
- `config.json`: model architecture config
- `config_sentence_transformers.json`: ColBERT config such as query and document lengths
- `special_tokens_map.json`: mask token mapping
- `1_Dense/model.safetensors`: projection layer weights, for example `768 -> 128`
- `1_Dense/config.json`: projection layer config

### Tantivy (`0.25.0`)

Full-text search engine with BM25 scoring.

- Crate: `tantivy`
- Used for: first-stage retrieval, BM25 scoring, and fuzzy matching

Key types:

- `Index`: the search index, either on disk or in RAM
- `Schema` / `SchemaBuilder`: field definitions
- `IndexWriter`: writes documents and commits them to disk
- `IndexReader` / `Searcher`: reads and searches the index
- `QueryParser`: parses human-readable query strings
- `FuzzyTermQuery`: Levenshtein-distance matching
- `TopDocs`: collector for top-K search results

Index persistence:

- `Index::create_in_dir(path, schema)`: create a new on-disk index
- `Index::open_in_dir(path)`: open an existing on-disk index
- `MmapDirectory`: production default, based on memory-mapped I/O

Field options:

- `TEXT`: tokenized and indexed for BM25
- `STRING`: exact match, not tokenized
- `STORED`: retrievable after search
- `FAST`: columnar storage for fast access by document ID
- `INDEXED`: searchable

### redb (`3.1.0`)

Embedded ACID key-value store.

- Crate: `redb`
- Used for: collection config, document metadata, and ColBERT embeddings

Key types:

- `Database`: database handle
- `TableDefinition<K, V>`: typed table definition
- `WriteTransaction` / `ReadTransaction`: MVCC transactions
- `Table` / `ReadOnlyTable`: table handles within a transaction

Key operations:

- `Database::create(path)`: create or open a database
- `db.begin_write()` / `db.begin_read()`: start transactions
- `txn.open_table(definition)`: open or create a table
- `table.insert(key, value)`: write a key-value pair
- `table.insert_reserve(key, len)`: pre-allocate space for zero-copy writes
- `table.get(key)`: read a value
- `table.iter()` / `table.range(bounds)`: iterate over entries
- `txn.commit()`: commit the transaction

Concurrency: one writer, many readers through MVCC. Readers do not block the writer.

Durability options:

- `Durability::Immediate`: fsync on commit; default and safest
- `Durability::None`: skip fsync for faster batch work; follow it with an `Immediate` commit if you care about durability

### hf-hub (`0.4.3`)

Hugging Face Hub client.

- Crate: `hf-hub`
- Used for: fetching `config_sentence_transformers.json` for remote model IDs so docbert can resolve model prompts from Sentence Transformers metadata

### clap (`4.5.57`)

Command-line argument parser.

- Crate: `clap` with the `derive` feature
- Used for: parsing CLI arguments into typed Rust structs

### xdg (`3.0.0`)

XDG Base Directory helper.

- Crate: `xdg`
- Used for: resolving platform-appropriate data, config, and cache directories

Key operation:

- `xdg::BaseDirectories::with_prefix("docbert")`: creates a handle that resolves paths like `~/.local/share/docbert/`

## Feature flag strategy

`Cargo.toml` should expose feature flags that map to pylate-rs backends:

```toml
[features]
default = []
mkl = ["pylate-rs/mkl"]
accelerate = ["pylate-rs/accelerate"]
metal = ["pylate-rs/metal"]
cuda = ["pylate-rs/cuda"]
```

The default build uses CPU inference with compiler auto-vectorization. Intel users can enable `mkl`; macOS users can pick `accelerate` or `metal`.
