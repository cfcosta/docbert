//! Save and load a built [`Index`] to/from disk.
//!
//! The on-disk format is a single little-endian binary file with a small
//! header followed by a sequence of plain f32/u32/u64 blobs. The layout
//! is deliberately boring — there's no compression, no extra framing,
//! and no schema evolution today; we can layer any of that on top later
//! without changing the semantic API.
//!
//! Layout, every field little-endian:
//!
//! ```text
//! magic           : 8 bytes, b"PLAIDIDX"
//! version         : u32     (currently 1)
//! dim             : u32
//! nbits           : u32
//! k_centroids    : u32
//! max_kmeans_iters: u32
//! n_documents     : u64
//! centroids       : dim * k_centroids     f32
//! bucket_cutoffs  : 2^nbits - 1            f32
//! bucket_weights  : 2^nbits                f32
//! doc_ids         : n_documents            u64
//! token_counts    : n_documents            u32  (tokens per document)
//! encoded_tokens  : for each token, u32 centroid_id then `dim` u8 codes
//! ```
//!
//! The inverted file is *not* persisted — it's derivable from the
//! encoded tokens in O(n_tokens) time on load and storing it would just
//! duplicate state that's already on disk elsewhere.

use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
    path::Path,
};

use crate::{
    codec::{EncodedVector, ResidualCodec},
    index::{Index, IndexParams, InvertedFile, TokenRef},
};

const MAGIC: &[u8; 8] = b"PLAIDIDX";
const FORMAT_VERSION: u32 = 1;

/// Write `index` to `path`, creating or truncating the file as needed.
pub fn save(index: &Index, path: &Path) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_index(index, &mut writer)?;
    writer.flush()
}

/// Read an [`Index`] back from `path`.
pub fn load(path: &Path) -> io::Result<Index> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    read_index(&mut reader)
}

fn write_index<W: Write>(index: &Index, w: &mut W) -> io::Result<()> {
    w.write_all(MAGIC)?;
    write_u32(w, FORMAT_VERSION)?;

    let params = &index.params;
    write_u32(w, params.dim as u32)?;
    write_u32(w, params.nbits)?;
    write_u32(w, params.k_centroids as u32)?;
    write_u32(w, params.max_kmeans_iters as u32)?;
    write_u64(w, index.doc_ids.len() as u64)?;

    write_f32_slice(w, &index.codec.centroids)?;
    write_f32_slice(w, &index.codec.bucket_cutoffs)?;
    write_f32_slice(w, &index.codec.bucket_weights)?;

    write_u64_slice(w, &index.doc_ids)?;

    let token_counts: Vec<u32> = index
        .doc_tokens
        .iter()
        .map(|tokens| tokens.len() as u32)
        .collect();
    write_u32_slice(w, &token_counts)?;

    for encoded_doc in &index.doc_tokens {
        for ev in encoded_doc {
            write_u32(w, ev.centroid_id)?;
            if ev.codes.len() != params.dim {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "encoded token has {} codes but dim is {}",
                        ev.codes.len(),
                        params.dim,
                    ),
                ));
            }
            w.write_all(&ev.codes)?;
        }
    }

    Ok(())
}

fn read_index<R: Read>(r: &mut R) -> io::Result<Index> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a docbert-plaid index (magic bytes mismatch)",
        ));
    }

    let version = read_u32(r)?;
    if version != FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported plaid index version {version}, expected {FORMAT_VERSION}",
            ),
        ));
    }

    let dim = read_u32(r)? as usize;
    let nbits = read_u32(r)?;
    let k_centroids = read_u32(r)? as usize;
    let max_kmeans_iters = read_u32(r)? as usize;
    let n_documents = read_u64(r)? as usize;

    if dim == 0 || k_centroids == 0 || !(1..=8).contains(&nbits) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "plaid index header has invalid dim/k_centroids/nbits",
        ));
    }

    let params = IndexParams {
        dim,
        nbits,
        k_centroids,
        max_kmeans_iters,
    };

    let centroids = read_f32_vec(r, k_centroids * dim)?;
    let num_buckets = 1usize << nbits;
    let bucket_cutoffs = read_f32_vec(r, num_buckets - 1)?;
    let bucket_weights = read_f32_vec(r, num_buckets)?;

    let doc_ids = read_u64_vec(r, n_documents)?;
    let token_counts = read_u32_vec(r, n_documents)?;

    let mut doc_tokens: Vec<Vec<EncodedVector>> =
        Vec::with_capacity(n_documents);
    let mut ivf = InvertedFile {
        lists: vec![Vec::new(); k_centroids],
    };
    for (doc_idx, count) in token_counts.iter().enumerate() {
        let mut encoded_doc = Vec::with_capacity(*count as usize);
        for token_idx in 0..*count {
            let centroid_id = read_u32(r)?;
            if (centroid_id as usize) >= k_centroids {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "centroid_id {centroid_id} out of range 0..{k_centroids}",
                    ),
                ));
            }
            let mut codes = vec![0u8; dim];
            r.read_exact(&mut codes)?;
            ivf.lists[centroid_id as usize].push(TokenRef {
                doc_idx: doc_idx as u32,
                token_idx,
            });
            encoded_doc.push(EncodedVector { centroid_id, codes });
        }
        doc_tokens.push(encoded_doc);
    }

    let codec = ResidualCodec {
        nbits,
        dim,
        centroids,
        bucket_cutoffs,
        bucket_weights,
    };
    codec.validate().map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("loaded codec is invalid: {e}"),
        )
    })?;

    Ok(Index {
        params,
        codec,
        doc_ids,
        doc_tokens,
        ivf,
    })
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_f32_slice<W: Write>(w: &mut W, slice: &[f32]) -> io::Result<()> {
    w.write_all(bytemuck::cast_slice(slice))
}

fn write_u32_slice<W: Write>(w: &mut W, slice: &[u32]) -> io::Result<()> {
    w.write_all(bytemuck::cast_slice(slice))
}

fn write_u64_slice<W: Write>(w: &mut W, slice: &[u64]) -> io::Result<()> {
    w.write_all(bytemuck::cast_slice(slice))
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32_vec<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<f32>> {
    let mut out = vec![0.0f32; n];
    r.read_exact(bytemuck::cast_slice_mut(&mut out))?;
    Ok(out)
}

fn read_u32_vec<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<u32>> {
    let mut out = vec![0u32; n];
    r.read_exact(bytemuck::cast_slice_mut(&mut out))?;
    Ok(out)
}

fn read_u64_vec<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<u64>> {
    let mut out = vec![0u64; n];
    r.read_exact(bytemuck::cast_slice_mut(&mut out))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{DocumentTokens, build_index};

    fn small_corpus() -> Vec<DocumentTokens> {
        vec![
            DocumentTokens {
                doc_id: 10,
                tokens: vec![0.0, 0.0, 0.1, 0.2, -0.1, 0.1],
                n_tokens: 3,
            },
            DocumentTokens {
                doc_id: 20,
                tokens: vec![10.0, 10.0, 10.2, 9.9, 9.8, 10.1],
                n_tokens: 3,
            },
            DocumentTokens {
                doc_id: 30,
                tokens: vec![0.3, -0.2, 9.7, 10.2],
                n_tokens: 2,
            },
        ]
    }

    fn default_params() -> IndexParams {
        IndexParams {
            dim: 2,
            nbits: 2,
            k_centroids: 2,
            max_kmeans_iters: 50,
        }
    }

    #[test]
    fn round_trip_preserves_codec_parameters_and_doc_ids() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("index.plaid");
        let index = build_index(&small_corpus(), default_params());

        save(&index, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert_eq!(loaded.params.dim, index.params.dim);
        assert_eq!(loaded.params.nbits, index.params.nbits);
        assert_eq!(loaded.params.k_centroids, index.params.k_centroids);
        assert_eq!(
            loaded.params.max_kmeans_iters,
            index.params.max_kmeans_iters
        );
        assert_eq!(loaded.doc_ids, index.doc_ids);
    }

    #[test]
    fn round_trip_preserves_codec_tables_byte_for_byte() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("index.plaid");
        let index = build_index(&small_corpus(), default_params());

        save(&index, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert_eq!(loaded.codec.centroids, index.codec.centroids);
        assert_eq!(loaded.codec.bucket_cutoffs, index.codec.bucket_cutoffs);
        assert_eq!(loaded.codec.bucket_weights, index.codec.bucket_weights);
    }

    #[test]
    fn round_trip_preserves_encoded_tokens() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("index.plaid");
        let index = build_index(&small_corpus(), default_params());

        save(&index, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert_eq!(loaded.doc_tokens.len(), index.doc_tokens.len());
        for (a, b) in loaded.doc_tokens.iter().zip(index.doc_tokens.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn round_trip_rebuilds_the_inverted_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("index.plaid");
        let index = build_index(&small_corpus(), default_params());

        save(&index, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert_eq!(loaded.ivf.num_centroids(), index.ivf.num_centroids());
        assert_eq!(loaded.ivf.total_tokens(), index.ivf.total_tokens());
        for c in 0..index.ivf.num_centroids() {
            let mut want = index.ivf.tokens_for_centroid(c).to_vec();
            let mut got = loaded.ivf.tokens_for_centroid(c).to_vec();
            want.sort_by_key(|t| (t.doc_idx, t.token_idx));
            got.sort_by_key(|t| (t.doc_idx, t.token_idx));
            assert_eq!(got, want, "IVF list for centroid {c} differs");
        }
    }

    #[test]
    fn round_trip_preserves_search_results_exactly() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("index.plaid");
        let index = build_index(&small_corpus(), default_params());
        save(&index, &path).unwrap();
        let loaded = load(&path).unwrap();

        let query = [0.05f32, 0.1, 9.9, 10.1];
        let params = crate::search::SearchParams {
            top_k: 3,
            n_probe: 2,
        };

        let a = crate::search::search(&index, &query, params);
        let b = crate::search::search(&loaded, &query, params);
        assert_eq!(a, b);
    }

    #[test]
    fn round_trip_handles_empty_documents() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("index.plaid");
        let mut docs = small_corpus();
        docs.push(DocumentTokens {
            doc_id: 999,
            tokens: vec![],
            n_tokens: 0,
        });
        let index = build_index(&docs, default_params());

        save(&index, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert_eq!(loaded.doc_ids, index.doc_ids);
        assert_eq!(loaded.doc_tokens.last().unwrap().len(), 0);
    }

    #[test]
    fn load_rejects_files_with_wrong_magic() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bogus.plaid");
        std::fs::write(&path, b"NOTPLAID").unwrap();

        let err = load(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("magic"));
    }

    #[test]
    fn load_rejects_unknown_format_version() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("future.plaid");
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&999u32.to_le_bytes());
        std::fs::write(&path, &buf).unwrap();

        let err = load(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("version"));
    }
}
