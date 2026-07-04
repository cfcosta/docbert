//! End-to-end indexing pipeline speed harness.
//!
//! Times `embed_and_store_in_batches` — GPU encode, the batched
//! dot-product matmul, host copies, Ward pooling, and LMDB writes —
//! over a deterministic varied-length corpus, the same workload the
//! `docbert index` embedding phase runs. Complements
//! docbert-pylate's `encode_speed`, which measures the encode stage
//! alone: deltas that only show up here live in the pooling/store
//! tail or in how it overlaps with encoding.
//!
//! Usage:
//!   index_speed [--docs N] [--repeats N] [--batch N]
//!
//! Each repeat writes into a fresh database inside a temp dir so LMDB
//! page reuse can't skew later repeats. Prints one JSON object.

use std::time::Instant;

use docbert_core::{EmbeddingDb, ModelManager, embedding};

/// Deterministic xorshift so corpora are identical across runs/builds.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() as usize) % (hi - lo).max(1)
    }
}

const WORDS: &[&str] = &[
    "retrieval",
    "embedding",
    "token",
    "index",
    "search",
    "ranking",
    "colbert",
    "attention",
    "kernel",
    "tensor",
    "batch",
    "layer",
    "vector",
    "corpus",
    "query",
    "document",
];

fn build_corpus(n_docs: usize) -> Vec<(u64, String)> {
    let mut rng = Rng(0xC0FFEE);
    (0..n_docs)
        .map(|i| {
            // 30–420 words: spans short chunks up to truncation-bound
            // documents, mirroring real chunked-markdown indexing.
            let n_words = rng.range(30, 420);
            let text: Vec<&str> = (0..n_words)
                .map(|_| WORDS[rng.range(0, WORDS.len())])
                .collect();
            (i as u64, text.join(" "))
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let n_docs: usize =
        flag("--docs").map(|v| v.parse().unwrap()).unwrap_or(512);
    let repeats: usize =
        flag("--repeats").map(|v| v.parse().unwrap()).unwrap_or(3);
    let submission_batch: usize = flag("--batch")
        .map(|v| v.parse().unwrap())
        .unwrap_or(embedding::EMBEDDING_SUBMISSION_BATCH_SIZE);

    let corpus = build_corpus(n_docs);
    let mut model = ModelManager::new();

    let tmp = tempfile::tempdir().expect("temp dir");

    // Warmup: loads the model and compiles/caches kernels.
    {
        let db = EmbeddingDb::open(&tmp.path().join("warmup.db"))
            .expect("warmup db");
        embedding::embed_and_store_in_batches(
            &mut model,
            &db,
            corpus[..n_docs.min(64)].to_vec(),
            submission_batch,
            |_| {},
        )
        .expect("warmup");
    }

    let mut wall_ms = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let db = EmbeddingDb::open(&tmp.path().join(format!("r{repeat}.db")))
            .expect("db");
        let docs = corpus.clone();
        let start = Instant::now();
        let stored = embedding::embed_and_store_in_batches(
            &mut model,
            &db,
            docs,
            submission_batch,
            |_| {},
        )
        .expect("embed and store");
        wall_ms.push(start.elapsed().as_secs_f64() * 1e3);
        assert_eq!(stored, n_docs);
    }

    let best = wall_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = wall_ms.iter().sum::<f64>() / wall_ms.len() as f64;
    println!(
        "{}",
        serde_json::json!({
            "docs": n_docs,
            "submission_batch": submission_batch,
            "repeats": repeats,
            "wall_ms": wall_ms,
            "docs_per_sec_best": n_docs as f64 / (best / 1e3),
            "docs_per_sec_mean": n_docs as f64 / (mean / 1e3),
        })
    );
}
