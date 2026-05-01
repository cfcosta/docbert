//! Document scoring helpers consumed by the search pipeline.
//!
//! The pre-PLAID reranker walked every embedding it could find,
//! computed a MaxSim score, and collapsed the per-chunk results back
//! to one row per document family using a path-derived
//! `document_family_key`. With content-derived chunk ids, that
//! collapse no longer makes sense — the search pipeline now drives
//! the ranking through PLAID and the per-document `chunk_owners`
//! reverse index, both of which produce `RankedDocument` rows
//! directly.
//!
//! What remains here is the `RankedDocument` shape itself, exposed
//! publicly so the search module can return it.

/// Candidate document after ranking.
///
/// Returned by the semantic search path. Results are sorted by
/// score, highest first.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedDocument {
    /// Numeric document identifier.
    pub doc_num_id: u64,
    /// ColBERT MaxSim similarity score (higher = more relevant).
    pub score: f32,
    /// Numeric ID of the best-scoring chunk for this document, when
    /// known. `None` for ranking flows that don't expose per-chunk
    /// scores.
    pub best_chunk_doc_id: Option<u64>,
}
