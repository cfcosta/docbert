# Indexing and Search Pipeline

## Indexing Pipeline

### Phase 1: Document Discovery

When a user adds a collection (`docbert collection add ~/notes --name notes`), the system:

1. Records the collection definition (name, directory path) in `config.db`
2. Walks the directory tree recursively, collecting all eligible files
3. For each file, computes a stable document ID by hashing the collection name + relative path
4. Compares file modification times against stored values to identify new or changed documents
5. Queues changed documents for processing

### Phase 2: Tantivy Indexing

For each new or modified document:

1. Parse the file to extract title (first heading or filename) and body text
2. If the document already exists in the Tantivy index, delete the old entry by term (using the document ID field)
3. Add a new Tantivy document with fields: doc_id, collection, relative_path, title, body, mtime
4. After processing all documents in a batch, call `commit()` to flush to disk

The Tantivy schema uses the `en_stem` tokenizer for the body field to enable English stemming. The title field also uses `en_stem` with a 2x boost factor during search.

### Phase 3: ColBERT Embedding

For each new or modified document:

1. Initialize the ColBERT model (lazy, first use only -- downloads from HuggingFace Hub)
2. Split the document body into chunks if it exceeds the model's maximum document length (typically 300 tokens after tokenization for the default model). Chunking strategy: character-based windowing with word-boundary adjustments, no overlap (to minimize chunk count)
3. Encode each chunk with `model.encode(&chunks, false)` (is_query=false)
4. Optionally apply hierarchical pooling with a configurable pool factor (default: 1, no pooling) to reduce token count
5. Serialize the resulting `[num_tokens, 128]` f32 matrix as bytes and store in `embeddings.db` keyed by the document's internal numeric ID
6. For chunked documents, store the concatenated embeddings of all chunks with chunk boundary markers

### Incremental Re-indexing

On subsequent runs of `docbert collection add` or a future `docbert sync` command:

1. Walk the directory tree again
2. Compare mtimes against stored values in config.db
3. Only re-process documents that are new or have changed
4. Remove embeddings and Tantivy entries for documents that no longer exist on disk

## Search Pipeline

### Step 1: Parse Query

1. Parse the user's query string
2. Apply collection filter if `-c <name>` is specified
3. Determine output format (human-readable or JSON)

### Step 2: First-Stage Retrieval (Tantivy)

1. Build a Tantivy query using `QueryParser` with fields `[title, body]`, title boosted 2x
2. If a collection filter is active, wrap in a BooleanQuery requiring the collection field to match
3. Use `parse_query_lenient` to handle typos gracefully (the fuzzy matching supplements this)
4. Also run a FuzzyTermQuery (distance=1) on key terms and combine results
5. Retrieve the top-1000 candidates via `TopDocs::with_limit(1000)`
6. Collect document IDs and BM25 scores

### Step 3: Query Encoding (ColBERT)

1. Load the ColBERT model (cached after first use)
2. Encode the query with `model.encode(&[query], true)` (is_query=true)
3. This produces a `[32, 128]` matrix (query is padded to 32 tokens with `[MASK]` tokens)

### Step 4: Reranking (MaxSim)

1. For each of the 1000 candidate documents, load the pre-computed embedding matrix from `embeddings.db`
2. Compute `similarity_matrix = query_embeddings @ doc_embeddings.T` yielding shape `[32, num_doc_tokens]`
3. For each query token (row), take the maximum value across all document tokens (columns)
4. Sum these 32 maximum values to get the final MaxSim score for this document
5. Sort all candidates by MaxSim score descending
6. Apply `--min-score` threshold if specified

### Step 5: Result Formatting

For human-readable output:

- Show rank, score, collection name, relative path, and a text snippet
- Display the short document ID (e.g., `#abc123`) for use with `docbert get`

For JSON output (`--json`):

- Emit a JSON array of objects with fields: rank, score, doc_id, collection, path, title, snippet

For file-list output (`--files`):

- Emit only the file paths, one per line (useful for piping to other tools)

## Performance Expectations

### Indexing

- Tantivy indexing: high throughput, limited mainly by disk I/O
- ColBERT encoding: the bottleneck. On CPU, encoding a 200-token document takes ~20-50ms. Batch encoding with rayon helps. For a 10,000 document corpus, initial indexing will be minutes, not seconds. GPU acceleration (CUDA feature flag) dramatically improves this
- redb writes: negligible overhead compared to encoding

### Search

- Tantivy BM25 retrieval: sub-10ms for corpora up to millions of documents
- ColBERT query encoding: ~5-20ms (single short query, CPU)
- MaxSim over 1000 candidates: ~5-50ms depending on average document length and SIMD backend
- Total search latency target: under 100ms for most queries on CPU

### Storage

Per document (assuming average 200 effective tokens after punctuation filtering):

| Component                         | Size per Document |
| --------------------------------- | ----------------- |
| Tantivy index entry               | ~1-2 KB           |
| ColBERT embeddings (f32, 128-dim) | ~100 KB           |
| redb overhead                     | negligible        |
| **Total**                         | **~100 KB**       |

For a 10,000 document corpus: ~1 GB total storage.
For a 100,000 document corpus: ~10 GB total storage.

Future optimization: hierarchical pooling with pool_factor=2 halves the embedding storage with negligible quality loss.
