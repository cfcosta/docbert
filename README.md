# docbert

A ColBERT-style indexer and search interface for documents.

## Plan

1. Use [PyLate-rs](https://lightonai.github.io/pylate-rs/) to compute ColBERT embeddings for all the documents in a directory.
2. Use [Tantivy](https://github.com/quickwit-oss/tantivy) to index the documents, and do both BM25 and Fuzzy matching.

The pipeline goes like this: we use Tantivy BM25 and Fuzzy to find the top 1k docs, then use ColBERT late interaction (with SIMD) to rerank the results.

