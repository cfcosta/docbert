//! Chunking utilities for splitting long documents into overlapping segments.
//!
//! ColBERT models have a maximum document length defined in the model config.
//! Pylate-rs reads `config_sentence_transformers.json`; if unavailable, docbert
//! falls back to a conservative 4096-token document length (below the model's
//! 8K context). Longer documents are truncated, losing semantic signal.
//! Chunking splits long documents into windows (optionally overlapping)
//! that can each be embedded separately.

use std::path::Path;

use hf_hub::{Repo, RepoType, api::sync::Api};
use serde::Deserialize;

/// Approximate characters per token for English text.
const CHARS_PER_TOKEN: usize = 4;

/// Default document length in tokens (conservative fallback when config is unavailable).
const DEFAULT_DOCUMENT_TOKENS: usize = 4096;

/// Default chunk size in characters (roughly ~4096 tokens).
pub const DEFAULT_CHUNK_SIZE: usize = DEFAULT_DOCUMENT_TOKENS * CHARS_PER_TOKEN;

/// Default overlap between chunks in characters (0 to minimize chunk count).
pub const DEFAULT_CHUNK_OVERLAP: usize = 0;

/// Chunking configuration derived from model settings.
#[derive(Debug, Clone, Copy)]
pub struct ChunkingConfig {
    pub chunk_size: usize,
    pub overlap: usize,
    pub document_length: Option<usize>,
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

fn load_document_length_remote(model_id: &str) -> Option<usize> {
    let api = Api::new().ok()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        "main".to_string(),
    ));
    let config_path = repo.get("config_sentence_transformers.json").ok()?;
    let contents = std::fs::read_to_string(config_path).ok()?;
    let config: SentenceTransformersConfig =
        serde_json::from_str(&contents).ok()?;
    config.document_length
}

/// Resolve chunking settings from a model path (if local), falling back to defaults.
pub fn resolve_chunking_config(model_id: &str) -> ChunkingConfig {
    let model_path = Path::new(model_id);
    if model_path.is_dir()
        && let Some(doc_len) = load_document_length(model_path)
    {
        return ChunkingConfig {
            chunk_size: chars_for_tokens(doc_len),
            overlap: DEFAULT_CHUNK_OVERLAP,
            document_length: Some(doc_len),
        };
    }
    if let Some(doc_len) = load_document_length_remote(model_id) {
        return ChunkingConfig {
            chunk_size: chars_for_tokens(doc_len),
            overlap: DEFAULT_CHUNK_OVERLAP,
            document_length: Some(doc_len),
        };
    }

    ChunkingConfig {
        chunk_size: DEFAULT_CHUNK_SIZE,
        overlap: DEFAULT_CHUNK_OVERLAP,
        document_length: None,
    }
}

/// A chunk of text from a larger document.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// The chunk text.
    pub text: String,
    /// Zero-based chunk index within the document.
    pub index: usize,
    /// Character offset where this chunk starts in the original document.
    pub start_offset: usize,
}

/// Split text into chunks (optionally overlapping).
///
/// Uses character-based splitting as a rough approximation of token count.
/// For English text, ~4 characters ≈ 1 token on average.
///
/// If the text is shorter than `chunk_size`, returns a single chunk.
/// Properly handles UTF-8 multi-byte characters (emojis, etc.).
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<Chunk> {
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

    let step = chunk_size.saturating_sub(overlap).max(1);
    let mut chunks = Vec::new();
    let mut start_char = 0;
    let mut index = 0;

    while start_char < char_count {
        let end_char = (start_char + chunk_size).min(char_count);

        // Try to break at word boundary
        let chunk_end_char = if end_char < char_count {
            find_word_boundary_char(text, &char_to_byte, end_char)
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

        start_char += step;

        // Avoid creating a tiny final chunk
        if char_count.saturating_sub(start_char) < chunk_size / 4
            && !chunks.is_empty()
        {
            break;
        }
    }

    chunks
}

/// Find a word boundary near the given char position, preferring to break
/// at whitespace or punctuation.
fn find_word_boundary_char(
    text: &str,
    char_to_byte: &[usize],
    pos_char: usize,
) -> usize {
    // Look back up to 100 chars for a good break point
    let search_start_char = pos_char.saturating_sub(100);

    let start_byte = char_to_byte[search_start_char];
    let end_byte = char_to_byte[pos_char];
    let search_region = &text[start_byte..end_byte];

    // Find the last whitespace in the region
    if let Some(ws_byte_offset) =
        search_region.rfind(|c: char| c.is_whitespace())
    {
        // Convert byte offset back to char position
        let ws_byte = start_byte + ws_byte_offset;
        // Find the char index for this byte position
        for (char_idx, &byte_idx) in char_to_byte.iter().enumerate() {
            if byte_idx > ws_byte {
                return char_idx;
            }
        }
    }

    pos_char
}

/// Generate a chunk-specific document ID by combining the base ID with chunk index.
///
/// Format: base_id XOR (chunk_index << 48)
/// This preserves the base ID in the lower bits while encoding chunk info in upper bits.
pub fn chunk_doc_id(base_id: u64, chunk_index: usize) -> u64 {
    if chunk_index == 0 {
        base_id
    } else {
        base_id ^ ((chunk_index as u64) << 48)
    }
}

/// Extract the base document ID and chunk index from a chunk doc ID.
///
/// Note: This only works reliably for chunk_index 0 (returns the base ID unchanged).
/// For other chunks, the base ID is XOR'd, so this is a one-way operation
/// unless you know the chunk_index.
pub fn parse_chunk_doc_id(chunk_id: u64) -> (u64, usize) {
    let chunk_index = (chunk_id >> 48) as usize;
    if chunk_index == 0 {
        (chunk_id, 0)
    } else {
        let base_id = chunk_id ^ ((chunk_index as u64) << 48);
        (base_id, chunk_index)
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

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
    fn chunk_doc_id_roundtrip() {
        let base = 12345678u64;

        // Chunk 0 should be unchanged
        assert_eq!(chunk_doc_id(base, 0), base);
        let (recovered, idx) = parse_chunk_doc_id(chunk_doc_id(base, 0));
        assert_eq!(recovered, base);
        assert_eq!(idx, 0);

        // Chunk 1+
        let chunk1_id = chunk_doc_id(base, 1);
        assert_ne!(chunk1_id, base);
        let (recovered, idx) = parse_chunk_doc_id(chunk1_id);
        assert_eq!(recovered, base);
        assert_eq!(idx, 1);
    }

    #[test]
    fn chunks_cover_full_text() {
        let text = "a".repeat(3000);
        let chunks = chunk_text(&text, 1000, 200);

        // First chunk starts at 0
        assert_eq!(chunks[0].start_offset, 0);

        // Last chunk should reach near the end
        let last = chunks.last().unwrap();
        let last_end = last.start_offset + last.text.len();
        assert!(last_end >= text.len() - 250, "should cover most of text");
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
    fn resolve_chunking_config_from_model_dir() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config_sentence_transformers.json");
        std::fs::write(&config_path, "{\"document_length\": 512}").unwrap();

        let config = resolve_chunking_config(&dir.path().to_string_lossy());
        assert_eq!(config.document_length, Some(512));
        assert_eq!(config.chunk_size, 512 * CHARS_PER_TOKEN);
        assert_eq!(config.overlap, DEFAULT_CHUNK_OVERLAP);
    }

    #[test]
    fn resolve_chunking_config_defaults_without_config() {
        let dir = tempdir().unwrap();
        let config = resolve_chunking_config(&dir.path().to_string_lossy());
        assert_eq!(config.document_length, None);
        assert_eq!(config.chunk_size, DEFAULT_CHUNK_SIZE);
        assert_eq!(config.overlap, DEFAULT_CHUNK_OVERLAP);
    }
}
