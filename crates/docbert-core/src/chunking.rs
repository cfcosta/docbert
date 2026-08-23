//! Helpers for splitting long documents into chunks before embedding.
//!
//! When a document is longer than the configured chunk size, docbert breaks it
//! into windows that can be embedded one at a time.
//!
//! By default, the chunk size matches docbert's standard ColBERT document
//! length: 519 tokens, or about 2K characters.

use std::path::Path;

use serde::Deserialize;

use crate::model_manager::FALLBACK_DOCUMENT_LENGTH;

/// Approximate characters per token for English text.
const CHARS_PER_TOKEN: usize = 4;

/// Default chunk size in characters (roughly ~300 tokens / ~1.2K chars).
pub const DEFAULT_CHUNK_SIZE: usize =
    FALLBACK_DOCUMENT_LENGTH * CHARS_PER_TOKEN;

/// Default overlap between chunks in characters (0 to minimize chunk count).
pub const DEFAULT_CHUNK_OVERLAP: usize = 0;

/// Domain prefix for the chunk-doc-id hash. Bumping the version forces
/// every key to change so callers re-derive ids without colliding with
/// older indices.
const CHUNK_DOC_ID_DOMAIN: &[u8] = b"docbert.chunk.v1\0";

/// Chunking settings resolved from the model configuration.
///
/// [`resolve_config`] reads `config_sentence_transformers.json` and,
/// when it can, uses the model's `document_length` value. The resolved
/// `model_id` is mixed into [`chunk_doc_id`] so embeddings produced by
/// one model are not silently reused by a different one.
///
/// # Examples
///
/// ```
/// use docbert_core::chunking::{resolve_config, DEFAULT_CHUNK_SIZE};
///
/// // Remote model IDs use defaults
/// let config = resolve_config("lightonai/ColBERT-Zero");
/// assert_eq!(config.chunk_size, DEFAULT_CHUNK_SIZE);
/// assert_eq!(config.document_length, None);
/// assert_eq!(config.model_id, "lightonai/ColBERT-Zero");
/// ```
#[derive(Debug, Clone)]
pub struct Config {
    /// Maximum chunk size in characters.
    pub chunk_size: usize,
    /// Overlap between adjacent chunks in characters.
    pub overlap: usize,
    /// Token-based document length from the model config, if available.
    pub document_length: Option<usize>,
    /// Identifier of the embedding model — mixed into [`chunk_doc_id`]
    /// so embeddings can't leak across model swaps.
    pub model_id: String,
}

#[derive(Debug, Deserialize)]
struct SentenceTransformersConfig {
    document_length: Option<usize>,
}

fn chars_for_tokens(tokens: usize) -> usize {
    tokens.saturating_mul(CHARS_PER_TOKEN).max(1)
}

fn load_document_length(model_dir: &Path) -> Option<usize> {
    let config_path = model_dir.join("config_sentence_transformers.json");
    let contents = std::fs::read_to_string(config_path).ok()?;
    let config: SentenceTransformersConfig =
        serde_json::from_str(&contents).ok()?;
    config.document_length
}

/// Pick chunking settings from a model path when possible, otherwise use defaults.
///
/// If `model_id` points to a local model directory with a
/// `config_sentence_transformers.json`, docbert reads `document_length` and turns
/// it into a rough character budget with `document_length * 4`.
///
/// If `model_id` is a remote model name such as `"lightonai/ColBERT-Zero"`,
/// docbert falls back to its built-in 519-token default.
///
/// # Examples
///
/// ```
/// use docbert_core::chunking::{resolve_config, DEFAULT_CHUNK_SIZE};
///
/// let config = resolve_config("lightonai/ColBERT-Zero");
/// assert_eq!(config.chunk_size, DEFAULT_CHUNK_SIZE);
/// ```
pub fn resolve_config(model_id: &str) -> Config {
    let model_path = Path::new(model_id);
    if model_path.is_dir()
        && let Some(doc_len) = load_document_length(model_path)
    {
        return Config {
            chunk_size: chars_for_tokens(doc_len),
            overlap: DEFAULT_CHUNK_OVERLAP,
            document_length: Some(doc_len),
            model_id: model_id.to_string(),
        };
    }

    Config {
        chunk_size: DEFAULT_CHUNK_SIZE,
        overlap: DEFAULT_CHUNK_OVERLAP,
        document_length: None,
        model_id: model_id.to_string(),
    }
}

/// One chunk cut from a larger document.
///
/// Returned by [`chunk_text`]. Each chunk keeps its index and starting byte
/// offset so you can map it back to the original text.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// The chunk text content.
    pub text: String,
    /// Zero-based chunk index within the document.
    pub index: usize,
    /// Byte offset where this chunk starts in the original document.
    pub start_offset: usize,
}

/// Split text into chunks, with optional overlap.
///
/// This works in characters, not tokens. The `4 chars ~= 1 token` rule is only
/// a rough estimate, but it is good enough for chunk sizing.
///
/// If the text already fits in `chunk_size`, you get one chunk back. Empty or
/// whitespace-only text returns no chunks. UTF-8 text is handled correctly, so
/// multi-byte characters such as emoji do not break the math.
///
/// # Examples
///
/// ```
/// use docbert_core::chunking::chunk_text;
///
/// // Short text returns a single chunk
/// let chunks = chunk_text("Hello, world!", 1000, 0);
/// assert_eq!(chunks.len(), 1);
/// assert_eq!(chunks[0].text, "Hello, world!");
///
/// // Long text gets split
/// let text = "word ".repeat(500);
/// let chunks = chunk_text(&text, 1000, 200);
/// assert!(chunks.len() >= 2);
/// ```
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<Chunk> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    // A zero-width chunk would advance one character at a time and
    // emit nothing at all, dropping the whole document.
    let chunk_size = chunk_size.max(1);
    let char_count = text.chars().count();

    // Short text doesn't need chunking
    if char_count <= chunk_size {
        return vec![Chunk {
            text: text.to_string(),
            index: 0,
            start_offset: 0,
        }];
    }

    // Build a map of char index -> byte index for O(1) lookups
    let char_to_byte: Vec<usize> = text
        .char_indices()
        .map(|(byte_idx, _)| byte_idx)
        .chain(std::iter::once(text.len()))
        .collect();

    let mut chunks = Vec::new();
    let mut start_char = 0;
    let mut index = 0;

    while start_char < char_count {
        let end_char = (start_char + chunk_size).min(char_count);

        // Try to break at word boundary, but never before this chunk
        // started — a boundary behind `start_char` would invert the
        // slice below and panic.
        let chunk_end_char = if end_char < char_count {
            find_word_boundary_char(text, &char_to_byte, start_char, end_char)
        } else {
            end_char
        };

        let start_byte = char_to_byte[start_char];
        let end_byte = char_to_byte[chunk_end_char];

        let chunk_text = &text[start_byte..end_byte];
        if !chunk_text.trim().is_empty() {
            chunks.push(Chunk {
                text: chunk_text.to_string(),
                index,
                start_offset: start_byte,
            });
            index += 1;
        }

        // Resume where this chunk actually ended, not where it would
        // have ended without the boundary search. Stepping by a whole
        // `chunk_size` instead would skip everything the search backed
        // past — the split word belongs to the next chunk, not to
        // neither. `find_word_boundary_char` guarantees the end is
        // past `start_char`, so this always moves forward.
        start_char = chunk_end_char.saturating_sub(overlap).max(start_char + 1);
    }

    chunks
}

/// How far back to look for a word boundary, in characters.
const BOUNDARY_LOOKBACK: usize = 100;

/// Find a nearby word boundary so we can avoid splitting in the middle of a word.
///
/// The result is always strictly greater than `floor_char`, which is
/// where the chunk being cut starts. Without that floor a short chunk
/// could break behind its own start and produce a backwards slice.
fn find_word_boundary_char(
    text: &str,
    char_to_byte: &[usize],
    floor_char: usize,
    pos_char: usize,
) -> usize {
    // Look back up to 100 chars for a good break point
    let search_start_char =
        pos_char.saturating_sub(BOUNDARY_LOOKBACK).max(floor_char);

    if search_start_char >= pos_char {
        return pos_char;
    }

    let start_byte = char_to_byte[search_start_char];
    let end_byte = char_to_byte[pos_char];
    let search_region = &text[start_byte..end_byte];

    // Find the last whitespace in the region
    if let Some(ws_byte_offset) =
        search_region.rfind(|c: char| c.is_whitespace())
    {
        // Convert byte offset back to char position. char_to_byte is
        // sorted, so this is a binary search rather than a scan from
        // the front — the scan made chunking quadratic in document
        // length.
        let ws_byte = start_byte + ws_byte_offset;
        let char_idx = char_to_byte.partition_point(|&byte| byte <= ws_byte);

        // ws_byte >= start_byte, so char_idx > search_start_char, and
        // search_start_char >= floor_char.
        if char_idx < char_to_byte.len() {
            return char_idx;
        }
    }

    pos_char
}

/// Stable, content-derived chunk identifier.
///
/// The same chunk text under the same model produces the same id,
/// regardless of which document it lives in. This lets the embedding
/// cache dedupe identical chunks (license boilerplate, copy-pasted
/// snippets, content that moved between files) and enables migrating
/// `embeddings.db` between machines: any chunk whose text matches gets
/// a free embedding from the imported cache.
///
/// `model_id` is mixed in so cached embeddings can't be silently reused
/// across model swaps — different models hash to different ids and the
/// cache simply misses, forcing a re-embed.
///
/// The 64-bit truncation of blake3 has a birthday-collision horizon
/// near 4 × 10^9 distinct chunks; comfortably above any realistic
/// personal corpus.
///
/// # Examples
///
/// ```
/// use docbert_core::chunking::chunk_doc_id;
///
/// // Same text + same model → same id, regardless of where it appears.
/// let a = chunk_doc_id("colbert-v2", "Hello, world!");
/// let b = chunk_doc_id("colbert-v2", "Hello, world!");
/// assert_eq!(a, b);
///
/// // Different model → different id.
/// let c = chunk_doc_id("other-model", "Hello, world!");
/// assert_ne!(a, c);
/// ```
pub fn chunk_doc_id(model_id: &str, chunk_text: &str) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CHUNK_DOC_ID_DOMAIN);
    hasher.update(model_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(chunk_text.as_bytes());
    let digest = hasher.finalize();
    let bytes = digest.as_bytes();
    u64::from_be_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
        bytes[7],
    ])
}

#[cfg(test)]
mod tests {
    use hegel::{TestCase, generators as gs};
    use tempfile::tempdir;

    use super::*;

    /// The byte ranges the chunks claim to cover.
    ///
    /// Also checks that `start_offset` tells the truth: the slice it
    /// points at has to be the chunk's own text.
    fn covered(text: &str, chunks: &[Chunk]) -> Vec<bool> {
        let mut seen = vec![false; text.len()];

        for chunk in chunks {
            let end = chunk.start_offset + chunk.text.len();
            assert_eq!(
                &text[chunk.start_offset..end],
                chunk.text,
                "chunk {} points at the wrong bytes",
                chunk.index
            );
            seen[chunk.start_offset..end].fill(true);
        }

        seen
    }

    /// The first character the chunks left behind, if any.
    ///
    /// Whitespace-only gaps don't count — an all-whitespace chunk is
    /// dropped on purpose, since it carries nothing to embed.
    fn lost(text: &str, chunks: &[Chunk]) -> Option<(usize, char)> {
        let seen = covered(text, chunks);

        text.char_indices()
            .find(|(at, c)| !seen[*at] && !c.is_whitespace())
    }

    #[test]
    fn empty_text_produces_no_chunks() {
        let chunks =
            chunk_text("   \n\t", DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP);
        assert!(chunks.is_empty());
    }

    #[test]
    fn short_text_single_chunk() {
        let chunks = chunk_text(
            "Hello, world!",
            DEFAULT_CHUNK_SIZE,
            DEFAULT_CHUNK_OVERLAP,
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello, world!");
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[0].start_offset, 0);
    }

    #[test]
    fn long_text_multiple_chunks() {
        let text = "word ".repeat(500); // 2500 chars
        let chunks = chunk_text(&text, 1000, 200);

        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[1].index, 1);

        // Chunks should overlap
        let first_end = chunks[0].start_offset + chunks[0].text.len();
        let second_start = chunks[1].start_offset;
        assert!(second_start < first_end, "chunks should overlap");
    }

    #[test]
    fn chunk_doc_id_is_deterministic_per_content_and_model() {
        let id_a = chunk_doc_id("colbert-v2", "hello world");
        let id_b = chunk_doc_id("colbert-v2", "hello world");
        assert_eq!(id_a, id_b);
    }

    #[test]
    fn chunk_doc_id_changes_with_content() {
        let model = "colbert-v2";
        assert_ne!(
            chunk_doc_id(model, "hello world"),
            chunk_doc_id(model, "hello world!"),
        );
    }

    #[test]
    fn chunk_doc_id_changes_with_model() {
        let text = "shared content";
        assert_ne!(
            chunk_doc_id("colbert-v1", text),
            chunk_doc_id("colbert-v2", text),
        );
    }

    #[test]
    fn chunk_doc_id_dedups_across_documents() {
        // The whole point of content-based ids: the same chunk text in
        // different documents collapses to the same id, which is what
        // makes embedding migration work.
        let model = "colbert-v2";
        let shared = "Apache License 2.0 ...";
        assert_eq!(chunk_doc_id(model, shared), chunk_doc_id(model, shared));
    }

    #[test]
    fn chunks_cover_full_text() {
        let text = "a".repeat(3000);
        let chunks = chunk_text(&text, 1000, 200);

        // First chunk starts at 0
        assert_eq!(chunks[0].start_offset, 0);

        // The last chunk reaches the end. Anything short of that is
        // text the index will never see.
        let last = chunks.last().unwrap();
        assert_eq!(last.start_offset + last.text.len(), text.len());
    }

    #[test]
    fn the_word_at_a_boundary_survives() {
        // 40 words of 4 letters plus a space. A 100-char cut lands
        // mid-word, so the boundary search backs up — and whatever it
        // backs past has to show up in the next chunk.
        let text = "wxyz ".repeat(40);
        let chunks = chunk_text(&text, 98, 0);

        assert_eq!(lost(&text, &chunks), None, "{chunks:#?}");
    }

    #[test]
    fn a_short_tail_survives() {
        // 30 chars past the first chunk, well under chunk_size / 4.
        let text = format!("{}TAILWORD", "ab ".repeat(40));
        let chunks = chunk_text(&text, 100, 0);

        assert!(
            chunks.iter().any(|chunk| chunk.text.contains("TAILWORD")),
            "the tail is gone: {chunks:#?}"
        );
        assert_eq!(lost(&text, &chunks), None);
    }

    #[test]
    fn no_chunk_runs_past_the_size() {
        let text = "word ".repeat(400);

        for chunk in chunk_text(&text, 137, 20) {
            assert!(
                chunk.text.chars().count() <= 137,
                "chunk {} is {} chars",
                chunk.index,
                chunk.text.chars().count()
            );
        }
    }

    #[test]
    fn multibyte_text_keeps_every_character() {
        let text = "café ☕ naïve 日本語 🎉 ".repeat(50);
        let chunks = chunk_text(&text, 100, 20);

        assert_eq!(lost(&text, &chunks), None);
    }

    /// Chunking exists to feed the embedder. A character that no chunk
    /// carries is a character no search can ever match, so coverage is
    /// the one property that has to hold for every input.
    #[hegel::test(test_cases = 200)]
    fn prop_chunks_carry_every_character(tc: TestCase) {
        let text = tc
            .draw(gs::text().alphabet("ab \n\u{e9}").min_size(0).max_size(400));
        let size = tc.draw(gs::integers::<usize>().min_value(0).max_value(80));
        let overlap =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(40));

        let chunks = chunk_text(&text, size, overlap);

        assert_eq!(
            lost(&text, &chunks),
            None,
            "size {size}, overlap {overlap}"
        );
        assert!(
            chunks.is_empty()
                || chunks[0].start_offset == 0
                || text[..chunks[0].start_offset]
                    .chars()
                    .all(char::is_whitespace),
            "the front of {text:?} is gone"
        );
    }

    /// The chunk size is the model's document budget. A chunk over it
    /// gets truncated at embed time, which loses text just as surely.
    #[hegel::test(test_cases = 200)]
    fn prop_no_chunk_runs_past_the_size(tc: TestCase) {
        let text = tc
            .draw(gs::text().alphabet("ab \n\u{e9}").min_size(0).max_size(400));
        let size = tc.draw(gs::integers::<usize>().min_value(1).max_value(80));
        let overlap =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(40));

        for chunk in chunk_text(&text, size, overlap) {
            assert!(
                chunk.text.chars().count() <= size,
                "chunk {} is {} chars, size {size}",
                chunk.index,
                chunk.text.chars().count()
            );
        }
    }

    #[test]
    fn handles_emoji_and_multibyte_chars() {
        // Create text with emojis that would cause byte/char boundary issues
        let emoji_text = "Hello 👉 world 🌍 test ".repeat(100);
        let chunks = chunk_text(&emoji_text, 200, 50);

        // Should not panic and should produce valid chunks
        assert!(!chunks.is_empty());

        // Each chunk should be valid UTF-8 (implicitly tested by String)
        for chunk in &chunks {
            assert!(!chunk.text.is_empty());
            // Verify we can iterate chars (proves valid UTF-8)
            let _: usize = chunk.text.chars().count();
        }
    }

    #[test]
    fn handles_mixed_length_unicode() {
        // Mix of ASCII (1 byte), accented chars (2 bytes), and emoji (4 bytes)
        let text = "café ☕ naïve 日本語 🎉 ".repeat(50);
        let chunks = chunk_text(&text, 100, 20);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            // Should be valid UTF-8
            assert!(chunk.text.chars().count() > 0);
        }
    }

    #[test]
    fn resolve_config_from_model_dir() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config_sentence_transformers.json");
        std::fs::write(&config_path, "{\"document_length\": 512}").unwrap();

        let model_id = dir.path().to_string_lossy().to_string();
        let config = resolve_config(&model_id);
        assert_eq!(config.document_length, Some(512));
        assert_eq!(config.chunk_size, 512 * CHARS_PER_TOKEN);
        assert_eq!(config.overlap, DEFAULT_CHUNK_OVERLAP);
        assert_eq!(config.model_id, model_id);
    }

    #[test]
    fn resolve_config_defaults_without_config() {
        let dir = tempdir().unwrap();
        let model_id = dir.path().to_string_lossy().to_string();
        let config = resolve_config(&model_id);
        assert_eq!(config.document_length, None);
        assert_eq!(config.chunk_size, DEFAULT_CHUNK_SIZE);
        assert_eq!(config.overlap, DEFAULT_CHUNK_OVERLAP);
        assert_eq!(config.model_id, model_id);
    }

    #[test]
    fn resolve_config_remote_model_uses_defaults() {
        // Remote model IDs (not local directories) use defaults
        let config = resolve_config("lightonai/ColBERT-Zero");
        assert_eq!(
            DEFAULT_CHUNK_SIZE,
            FALLBACK_DOCUMENT_LENGTH * CHARS_PER_TOKEN
        );
        assert_eq!(config.document_length, None);
        assert_eq!(config.chunk_size, DEFAULT_CHUNK_SIZE);
        assert_eq!(config.overlap, DEFAULT_CHUNK_OVERLAP);
        assert_eq!(config.model_id, "lightonai/ColBERT-Zero");
    }
}
