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
2. Split the document body into chunks if it exceeds the model's maximum document length (read from `config_sentence_transformers.json` locally, or fetched via hf-hub for remote model IDs; defaults to the built-in chunk size of ~1024 tokens if unavailable). Chunking strategy: character-based windowing with word-boundary adjustments, no overlap (to minimize chunk count)
3. Encode each chunk with `model.encode(&chunks, false)` (is_query=false)
4. Serialize the resulting `[num_tokens, 128]` f32 matrix as bytes and store in `embeddings.db` keyed by the document's internal numeric ID
5. For chunked documents, store each chunk separately using a chunk-specific numeric ID (derived from the base doc ID and chunk index)
   - Note: reranking currently fetches embeddings by the base document ID only, so only the first chunk (index 0) is used during search; additional chunk embeddings are stored but not referenced.

### Incremental Re-indexing

On subsequent runs of `docbert sync`:

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
3. This produces a `[query_length, 128]` matrix (query is padded to `query_length` with the tokenizer mask token; `query_length` comes from `config_sentence_transformers.json` or defaults to 32)

### Step 4: Reranking (MaxSim)

1. For each of the 1000 candidate documents, load the pre-computed embedding matrix from `embeddings.db`
2. Compute `similarity_matrix = query_embeddings @ doc_embeddings.T` yielding shape `[query_length, num_doc_tokens]`
3. For each query token (row), take the maximum value across all document tokens (columns)
4. Sum these maximum values to get the final MaxSim score for this document
5. Sort all candidates by MaxSim score descending
6. Apply `--min-score` threshold if specified

### Step 5: Result Formatting

For human-readable output (CLI):

- Show rank, score, collection name, relative path, and document ID
- Display the short document ID (e.g., `#abc123`) for use with `docbert get`

For JSON output (`--json`):

- Emit an object with fields: query, result_count, and results (rank, score, doc_id, collection, path, title)

For MCP tool responses:

- Optional snippets are included when requested (separate from the CLI output).

For file-list output (`--files`):

- Emit only the file paths, one per line (useful for piping to other tools)

### Semantic-only Search (ssearch / semantic_search)

Semantic-only search skips Tantivy entirely and relies on ColBERT scoring:

1. Load all document IDs from `config.db`
2. Encode the query with ColBERT
3. For each stored embedding (chunk 0 only), compute MaxSim against the query
4. Sort by score descending, apply `--min-score`, and limit to `-n` (unless `--all`)
5. Format results the same way as `docbert search`

Because it scores every document embedding, semantic-only search is O(N) in the
number of documents and can be significantly slower than the two-stage pipeline.

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
- Semantic-only search: O(N) over stored embeddings; expect substantially higher latency for large corpora

### Storage

Per document (assuming average 200 effective tokens after truncation/chunking):

| Component                         | Size per Document |
| --------------------------------- | ----------------- |
| Tantivy index entry               | ~1-2 KB           |
| ColBERT embeddings (f32, 128-dim) | ~100 KB           |
| redb overhead                     | negligible        |
| **Total**                         | **~100 KB**       |

For a 10,000 document corpus: ~1 GB total storage.
For a 100,000 document corpus: ~10 GB total storage.

Future optimization: none currently implemented.
