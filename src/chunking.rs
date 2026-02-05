//! Chunking utilities for splitting long documents into overlapping segments.
//!
//! ColBERT models typically have a max sequence length of ~512 tokens.
//! Documents longer than this are truncated, losing semantic signal.
//! Chunking splits long documents into overlapping windows that can each
//! be embedded separately.

/// Default chunk size in characters (roughly ~400 tokens).
pub const DEFAULT_CHUNK_SIZE: usize = 1600;

/// Default overlap between chunks in characters (roughly ~50 tokens).
pub const DEFAULT_CHUNK_OVERLAP: usize = 200;

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

/// Split text into overlapping chunks.
///
/// Uses character-based splitting as a rough approximation of token count.
/// For English text, ~4 characters ≈ 1 token on average.
///
/// If the text is shorter than `chunk_size`, returns a single chunk.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<Chunk> {
    let text_len = text.len();

    // Short text doesn't need chunking
    if text_len <= chunk_size {
        return vec![Chunk {
            text: text.to_string(),
            index: 0,
            start_offset: 0,
        }];
    }

    let step = chunk_size.saturating_sub(overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut index = 0;

    while start < text_len {
        let end = (start + chunk_size).min(text_len);

        // Try to break at word boundary
        let chunk_end = if end < text_len {
            find_word_boundary(text, end)
        } else {
            end
        };

        let chunk_text = &text[start..chunk_end];
        if !chunk_text.trim().is_empty() {
            chunks.push(Chunk {
                text: chunk_text.to_string(),
                index,
                start_offset: start,
            });
            index += 1;
        }

        start += step;

        // Avoid creating a tiny final chunk
        if text_len.saturating_sub(start) < chunk_size / 4 && !chunks.is_empty()
        {
            break;
        }
    }

    chunks
}

/// Find a word boundary near the given position, preferring to break
/// at whitespace or punctuation.
fn find_word_boundary(text: &str, pos: usize) -> usize {
    // Look back up to 100 chars for a good break point
    let search_start = pos.saturating_sub(100);
    let search_region = &text[search_start..pos];

    // Find the last whitespace in the region
    if let Some(ws_offset) = search_region.rfind(|c: char| c.is_whitespace()) {
        return search_start + ws_offset + 1;
    }

    pos
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
    use super::*;

    #[test]
    fn short_text_single_chunk() {
        let chunks = chunk_text("Hello, world!", 1600, 200);
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
}
