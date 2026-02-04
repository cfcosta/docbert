# docbert

A ColBERT-style indexer and search interface for documents.

## Plan

1. Use [PyLate-rs](https://lightonai.github.io/pylate-rs/) to compute ColBERT embeddings for all the documents in a directory.
2. Use [Tantivy](https://github.com/quickwit-oss/tantivy) to index the documents, and do both BM25 and Fuzzy matching.
3. Use [redb](https://github.com/cberner/redb) to save the embeddings for all documents.

The pipeline goes like this: we use Tantivy BM25 and Fuzzy to find the top 1k docs, then use ColBERT late interaction (with SIMD) to rerank the results.

## Proposed usage

For users and humans:

```
# Create collections for your notes, docs, and meeting transcripts
docbert collection add ~/notes --name notes
docbert collection add ~/Documents/meetings --name meetings
docbert collection add ~/work/docs --name docs

# Add context to help with search results
docbert context add bert://notes "Personal notes and ideas"
docbert context add bert://meetings "Meeting transcripts and notes"
docbert context add bert://docs "Work documentation"

# Search across everything
docbert search "project timeline"

# Get a specific document
docbert get "meetings/2024-01-15.md"

# Get a document by docid (shown in search results)
docbert get "#abc123"

# Get multiple documents by glob pattern
docbert multi-get "journals/2025-05*.md"

# Search within a specific collection
docbert search "API" -c notes

# Export all matches for an agent
docbert search "API" --all --files --min-score 0.3
```

For AI agents:

```
# Get structured results for an LLM
docbert search "authentication" --json -n 10

# List all relevant files above a threshold
docbert query "error handling" --all --files --min-score 0.4

# Retrieve full document content
docbert get "docs/api-reference.md" --full
```

## Notes

1. Tantivy indexes and embeddings should be saved on the correct xdg data dir.
