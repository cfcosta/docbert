use std::sync::Mutex;

use anyhow::Result;
use candle_core::Device;
use docbert_pylate::{hierarchical_pooling, ColBERT};

/// Serialises hf-hub downloads across the integration tests.
///
/// Cargo runs integration tests concurrently by default, and several
/// of the tests below load `lightonai/GTE-ModernColBERT-v1`. On a
/// warm local cache this is free — hf-hub's `.lock` files resolve
/// instantly because the blobs are already on disk. In CI the cache
/// is cold, two tests grab the same `<blob>.lock` simultaneously,
/// and one of them trips hf-hub's "Lock acquisition failed" guard
/// before the other finishes downloading.
///
/// Holding the mutex only across the short model-construction path
/// (download + weight map + tokeniser init) lets the expensive
/// encode / similarity work parallelise normally.
static MODEL_LOAD_LOCK: Mutex<()> = Mutex::new(());

fn load_model(repo_id: &str, device: Device) -> Result<ColBERT> {
    let _guard = MODEL_LOAD_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    Ok(ColBERT::from(repo_id).with_device(device).try_into()?)
}

/// Selects the device integration tests should run on. Prefers CUDA when
/// the `cuda` feature is enabled, then Metal when `metal` is enabled, and
/// falls back to CPU — with a runtime fall-back to CPU when the preferred
/// accelerator can't be initialised. Leaves the explicit CPU/CUDA parity
/// test below alone: that test compares *both* devices on purpose.
fn test_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(d) = Device::new_metal(0) {
            return d;
        }
    }
    Device::Cpu
}

fn assert_close(actual: f32, expected: f32, tolerance: f32, context: &str) {
    assert!(
        (actual - expected).abs() < tolerance,
        "{context}: got {actual}, expected {expected} ± {tolerance}",
    );
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

/// Tests the `GTE-ModernColBERT-v1` model from the Hugging Face Hub.
#[test]
fn gte_modern_colbert_test() -> Result<()> {
    let device = test_device();
    println!("Testing with lightonai/GTE-ModernColBERT-v1...");

    let mut model = load_model("lightonai/GTE-ModernColBERT-v1", device)?;

    let query_sentences = vec!["what is the capital of france".to_string()];
    let document_sentences = vec!["paris is the capital of france".to_string()];

    let query_embeddings = model.encode(&query_sentences, true)?;
    let document_embeddings = model.encode(&document_sentences, false)?;

    let similarities =
        model.similarity(&query_embeddings, &document_embeddings)?;
    let score = similarities.data[0][0];

    println!("GTE-ModernColBERT-v1 Similarity: {}", score);
    assert_close(
        score,
        9.50805,
        1e-2,
        "GTE-ModernColBERT-v1 score regression",
    );

    let document_sentences = vec![
        "paris is the capital of france".to_string(),
        "berlin is the capital of germany, this is a test".to_string(),
    ];

    let document_embeddings = model.encode(&document_sentences, false)?;
    let pooled_embeddings = hierarchical_pooling(&document_embeddings, 2)?;

    println!(
        "Documents embeddings shape: {:?}",
        document_embeddings.dims()
    );
    println!(
        "Pooled documents embeddings shape: {:?}",
        pooled_embeddings.dims()
    );

    assert_eq!(document_embeddings.dim(0)?, pooled_embeddings.dim(0)?);
    assert!(pooled_embeddings.dim(1)? <= document_embeddings.dim(1)?);

    Ok(())
}

#[test]
fn gte_modern_colbert_semantics_regression_test() -> Result<()> {
    let mut model =
        load_model("lightonai/GTE-ModernColBERT-v1", test_device())?;

    let query_sentences = vec![
        "what is the capital of france".to_string(),
        "who wrote pride and prejudice".to_string(),
    ];
    let document_sentences = vec![
        "paris is the capital of france".to_string(),
        "jane austen wrote pride and prejudice".to_string(),
        "berlin is the capital of germany".to_string(),
        "the pacific ocean is the largest ocean on earth".to_string(),
    ];

    let query_embeddings = model.encode(&query_sentences, true)?;
    let document_embeddings = model.encode(&document_sentences, false)?;
    let similarities =
        model.similarity(&query_embeddings, &document_embeddings)?;

    assert_eq!(
        argmax(&similarities.data[0]),
        0,
        "France query should rank Paris first"
    );
    assert_eq!(
        argmax(&similarities.data[1]),
        1,
        "Pride and Prejudice query should rank Jane Austen first"
    );

    for (query_index, query) in query_sentences.iter().enumerate() {
        for (doc_index, document) in document_sentences.iter().enumerate() {
            let single_query =
                model.encode(std::slice::from_ref(query), true)?;
            let single_document =
                model.encode(std::slice::from_ref(document), false)?;
            let single_score =
                model.similarity(&single_query, &single_document)?.data[0][0];

            assert_close(
                similarities.data[query_index][doc_index],
                single_score,
                1e-4,
                &format!(
                    "batch invariance regression for query {query_index} and document {doc_index}"
                ),
            );
        }
    }

    let pooled_document_embeddings =
        hierarchical_pooling(&document_embeddings, 2)?;
    let pooled_similarities =
        model.similarity(&query_embeddings, &pooled_document_embeddings)?;

    assert_eq!(
        argmax(&pooled_similarities.data[0]),
        0,
        "Pooling should preserve Paris as the top-ranked document"
    );
    assert_eq!(
        argmax(&pooled_similarities.data[1]),
        1,
        "Pooling should preserve Jane Austen as the top-ranked document"
    );

    Ok(())
}

/// Tests the `colbertv2.0` model from the Hugging Face Hub.
#[test]
fn colbert_v2_test() -> Result<()> {
    let device = test_device();
    println!("Testing with lightonai/colbertv2.0...");

    let mut model = load_model("lightonai/colbertv2.0", device)?;

    let query_sentences = vec!["what is the capital of france".to_string()];
    let document_sentences = vec!["paris is the capital of france".to_string()];

    let query_embeddings = model.encode(&query_sentences, true)?;
    let document_embeddings = model.encode(&document_sentences, false)?;

    let similarities =
        model.similarity(&query_embeddings, &document_embeddings)?;
    let score = similarities.data[0][0];

    println!("colbertv2.0 Similarity: {}", score);
    assert_close(score, 29.603443, 1e-2, "colbertv2.0 score regression");
    Ok(())
}

/// Tests the `answerai-colbert-small-v1` model from the Hugging Face Hub.
#[test]
fn answerai_colbert_small_v1_test() -> Result<()> {
    let device = test_device();
    println!("Testing with lightonai/answerai-colbert-small-v1...");

    let mut model = load_model("lightonai/answerai-colbert-small-v1", device)?;

    let query_sentences = vec!["what is the capital of france".to_string()];
    let document_sentences = vec!["paris is the capital of france".to_string()];

    let query_embeddings = model.encode(&query_sentences, true)?;
    let document_embeddings = model.encode(&document_sentences, false)?;

    let similarities =
        model.similarity(&query_embeddings, &document_embeddings)?;
    let score = similarities.data[0][0];

    println!("answerai-colbert-small-v1 Similarity: {}", score);
    assert_close(
        score,
        31.490696,
        1e-2,
        "answerai-colbert-small-v1 score regression",
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn gte_modern_colbert_cpu_cuda_parity_test() -> Result<()> {
    let cuda_device = match Device::new_cuda(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!(
                "Skipping CUDA parity test because CUDA device 0 is unavailable: {error}"
            );
            return Ok(());
        }
    };

    let query_sentences = vec![
        "what is the capital of france".to_string(),
        "who wrote pride and prejudice".to_string(),
    ];
    let document_sentences = vec![
        "paris is the capital of france".to_string(),
        "jane austen wrote pride and prejudice".to_string(),
        "berlin is the capital of germany".to_string(),
    ];

    let mut cpu_model =
        load_model("lightonai/GTE-ModernColBERT-v1", Device::Cpu)?;
    let cpu_query_embeddings = cpu_model.encode(&query_sentences, true)?;
    let cpu_document_embeddings =
        cpu_model.encode(&document_sentences, false)?;
    let cpu_similarities = cpu_model
        .similarity(&cpu_query_embeddings, &cpu_document_embeddings)?;

    let mut cuda_model =
        load_model("lightonai/GTE-ModernColBERT-v1", cuda_device)?;
    let cuda_query_embeddings = cuda_model.encode(&query_sentences, true)?;
    let cuda_document_embeddings =
        cuda_model.encode(&document_sentences, false)?;
    let cuda_similarities = cuda_model
        .similarity(&cuda_query_embeddings, &cuda_document_embeddings)?;

    for (query_index, (cpu_scores, cuda_scores)) in cpu_similarities
        .data
        .iter()
        .zip(cuda_similarities.data.iter())
        .enumerate()
    {
        assert_eq!(
            argmax(cpu_scores),
            argmax(cuda_scores),
            "CPU/CUDA should agree on the top-ranked document for query {query_index}"
        );

        for (doc_index, (&cpu_score, &cuda_score)) in
            cpu_scores.iter().zip(cuda_scores.iter()).enumerate()
        {
            assert_close(
                cuda_score,
                cpu_score,
                0.2,
                &format!(
                    "CPU/CUDA similarity parity regression for query {query_index} and document {doc_index}"
                ),
            );
        }
    }

    Ok(())
}
