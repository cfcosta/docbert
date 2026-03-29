use crate::search::FinalResult;

#[derive(Debug, Clone, PartialEq)]
pub struct EnrichedSearchResult {
    pub rank: usize,
    pub score: f32,
    pub doc_id: String,
    pub collection: String,
    pub path: String,
    pub title: String,
    pub metadata: Option<serde_json::Value>,
}

pub fn enrich<F>(
    results: Vec<FinalResult>,
    mut load_metadata: F,
) -> Vec<EnrichedSearchResult>
where
    F: FnMut(u64) -> Option<serde_json::Value>,
{
    results
        .into_iter()
        .map(|r| EnrichedSearchResult {
            rank: r.rank,
            score: r.score,
            doc_id: r.doc_id,
            collection: r.collection,
            path: r.path,
            title: r.title,
            metadata: load_metadata(r.doc_num_id),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::FinalResult;

    fn result(rank: usize, score: f32, doc_num_id: u64) -> FinalResult {
        FinalResult {
            rank,
            score,
            doc_id: format!("#doc{doc_num_id}"),
            doc_num_id,
            collection: "notes".to_string(),
            path: format!("{doc_num_id}.md"),
            title: format!("Doc {doc_num_id}"),
        }
    }

    #[test]
    fn enrich_preserves_order_rank_and_score() {
        let enriched =
            enrich(vec![result(1, 2.0, 10), result(2, 1.0, 20)], |_| None);

        assert_eq!(enriched.len(), 2);
        assert_eq!(enriched[0].rank, 1);
        assert_eq!(enriched[0].score, 2.0);
        assert_eq!(enriched[0].doc_id, "#doc10");
        assert_eq!(enriched[1].rank, 2);
        assert_eq!(enriched[1].score, 1.0);
        assert_eq!(enriched[1].doc_id, "#doc20");
    }

    #[test]
    fn enrich_attaches_metadata_by_doc_num_id() {
        let enriched = enrich(vec![result(1, 2.0, 10)], |doc_id| {
            Some(serde_json::json!({ "doc": doc_id }))
        });

        assert_eq!(
            enriched[0].metadata,
            Some(serde_json::json!({ "doc": 10 }))
        );
    }

    #[test]
    fn enrich_keeps_none_when_loader_returns_none() {
        let enriched = enrich(vec![result(1, 2.0, 10)], |_| None);

        assert_eq!(enriched[0].metadata, None);
    }
}
