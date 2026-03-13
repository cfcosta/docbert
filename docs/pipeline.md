# Indexing and search pipeline

## Indexing pipeline

### Phase 1: document discovery

`docbert collection add ~/notes --name notes` does not index files. It only records the collection name and directory path in `config.db`.

Actual discovery and change detection happen during `docbert sync` and `docbert rebuild`:

1. Walk the directory tree and collect eligible files
2. Compute a stable document ID from `collection + relative_path`
3. Compare file modification times against stored metadata
4. Queue new or changed documents for processing

### Phase 2: Tantivy indexing

For each new or modified document:

1. Parse the file and extract a title from the first heading or the filename
2. If the document is already in the Tantivy index, delete the old entry by document ID
3. Add a new Tantivy document with `doc_id`, `collection`, `relative_path`, `title`, `body`, and `mtime`
4. After the batch finishes, call `commit()` to flush changes to disk

The Tantivy schema uses the `en_stem` tokenizer for the body field so English stemming works out of the box. The title field uses the same tokenizer and gets a 2x boost during search.

### Phase 3: ColBERT embedding

For each new or modified document:

1. Initialize the ColBERT model if it has not been loaded yet. The first load downloads model files from HuggingFace Hub.
2. Split the document into chunks if it exceeds the model's document length. For local model directories, that length comes from `config_sentence_transformers.json`. Otherwise docbert falls back to the built-in chunk size of 519 tokens, or about 2K characters.
3. Encode each chunk with `model.encode(&chunks, false)`.
4. Serialize the resulting `[num_tokens, 128]` `f32` matrix and store it in `embeddings.db`, keyed by the document's numeric ID.
5. For chunked documents, store each chunk under a chunk-specific numeric ID derived from the base document ID.

Current limitation: reranking still fetches embeddings by the base document ID only, so search uses chunk 0 during ranking. Extra chunk embeddings are stored, but not read back during reranking yet.

### Incremental re-indexing

On later runs of `docbert sync`:

1. Walk the directory tree again
2. Compare mtimes against values stored in `config.db`
3. Re-process only new or changed files
4. Remove Tantivy entries and embeddings for files that were deleted on disk

## Search pipeline

### Step 1: parse the query

1. Read the user's query string
2. Apply a collection filter if `-c <name>` is set
3. Decide whether output should be human-readable, JSON, or file-only

### Step 2: first-stage retrieval with Tantivy

1. Build a Tantivy query over `[title, body]`, with the title field boosted 2x
2. If a collection filter is active, wrap the query in a `BooleanQuery` that also requires the collection field to match
3. Use `parse_query_lenient` so minor syntax issues do not blow up the search
4. Run `FuzzyTermQuery` matches with distance 1 on key terms and merge those results in
5. Fetch the top 1000 candidates with `TopDocs::with_limit(1000)`
6. Collect document IDs and BM25 scores

### Step 3: encode the query with ColBERT

1. Load the ColBERT model if needed
2. Encode the query with `model.encode(&[query], true)`
3. Get a `[query_length, 128]` embedding matrix

`query_length` comes from `config_sentence_transformers.json` when available, or falls back to 32.

### Step 4: rerank with MaxSim

1. Load the precomputed embedding matrix for each candidate from `embeddings.db`
2. Compute `similarity_matrix = query_embeddings @ doc_embeddings.T`, which has shape `[query_length, num_doc_tokens]`
3. For each query token, take the maximum value across all document tokens
4. Sum those per-token maxima to get the final MaxSim score
5. Sort candidates by MaxSim score in descending order
6. Apply `--min-score` if the user set it

### Step 5: format results

For human-readable CLI output:

- show rank, score, collection name, relative path, and document ID
- show the short document ID such as `#abc123`, which can be passed to `docbert get`

For JSON output (`--json`):

- emit an object with `query`, `result_count`, and `results`
- each result includes `rank`, `score`, `doc_id`, `collection`, `path`, and `title`

For MCP tool responses:

- include snippets when requested

For file-list output (`--files`):

- print only file paths, one per line

### Semantic-only search (`ssearch` / `semantic_search`)

Semantic-only search skips Tantivy and scores every stored embedding directly:

1. Load all document IDs from `config.db`
2. Encode the query with ColBERT
3. Compute MaxSim against each stored embedding, using chunk 0 only
4. Sort by score, apply `--min-score`, and limit to `-n` unless `--all` is set
5. Format the output the same way as `docbert search`

Because it scores every stored embedding, semantic-only search is `O(N)` in the number of indexed documents and can get slow on large corpora.

## Performance expectations

### Indexing

- Tantivy indexing is usually limited by disk I/O
- ColBERT encoding is the main bottleneck. On CPU, encoding a 200-token document takes about 20-50 ms. Batch encoding with rayon helps.
- For a corpus of 10,000 documents, initial indexing is measured in minutes, not seconds
- GPU acceleration with the `cuda` feature can cut embedding time substantially
- redb write overhead is small compared to embedding time

### Search

- Tantivy BM25 retrieval: usually under 10 ms for corpora up to millions of documents
- ColBERT query encoding: about 5-20 ms for a short query on CPU
- MaxSim across 1000 candidates: about 5-50 ms, depending on document length and SIMD backend
- Target total latency: under 100 ms for most queries on CPU
- Semantic-only search: `O(N)` over stored embeddings, so expect much higher latency on large corpora

### Storage

Assuming an average of 200 effective tokens per document after truncation or chunking:

| Component                         | Size per document |
| --------------------------------- | ----------------- |
| Tantivy index entry               | ~1-2 KB           |
| ColBERT embeddings (`f32`, 128-d) | ~100 KB           |
| redb overhead                     | negligible        |
| Total                             | ~100 KB           |

For a corpus of 10,000 documents, that is about 1 GB of storage.
For a corpus of 100,000 documents, it is about 10 GB.

No storage optimizations beyond that are implemented yet.
