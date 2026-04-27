//! Extract a `.crate` tarball (gzip + tar) to a target directory.
//!
//! Crates.io packages every release as `<name>-<version>.crate`, a
//! gzipped tar archive whose entries all live under a single
//! `<name>-<version>/` prefix. We strip that prefix during extraction
//! so the destination directory contains the package source directly.
//!
//! Path-traversal safety: any entry whose normalized path escapes the
//! destination root (via `..` segments, absolute paths, or symlinks
//! that resolve outside) is rejected with [`Error::UnsafeTarballEntry`].

use std::{
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
};

use flate2::read::GzDecoder;

use crate::error::{Error, Result};

/// Extract the gzipped tar archive in `bytes` into `dest`.
///
/// `dest` is created if missing. The crates.io top-level
/// `<name>-<version>/` directory is stripped: a tarball entry at
/// `serde-1.0.219/src/lib.rs` lands at `dest/src/lib.rs`.
///
/// Returns the number of regular file entries extracted.
pub fn extract_crate_tarball(bytes: &[u8], dest: &Path) -> Result<usize> {
    fs::create_dir_all(dest)?;
    let canonical_dest = dest.canonicalize()?;

    let gz = GzDecoder::new(bytes);
    let mut archive = tar::Archive::new(gz);

    let mut count = 0usize;
    for entry in archive.entries()? {
        let mut entry = entry?;
        let header_path = entry.path()?.into_owned();
        let stripped = strip_top_level(&header_path);

        let Some(stripped) = stripped else {
            continue; // entry is the top-level dir itself
        };

        if !is_safe_relative(&stripped) {
            return Err(Error::UnsafeTarballEntry {
                path: header_path.display().to_string(),
            });
        }

        let target = canonical_dest.join(&stripped);

        // Defensive: post-join check. A symlink-via-..-only attack
        // can't slip past `is_safe_relative`, but a future change
        // could; assert the joined path still starts with dest.
        if !target.starts_with(&canonical_dest) {
            return Err(Error::UnsafeTarballEntry {
                path: header_path.display().to_string(),
            });
        }

        let entry_type = entry.header().entry_type();
        if entry_type.is_dir() {
            fs::create_dir_all(&target)?;
            continue;
        }

        if entry_type.is_file() {
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf)?;
            fs::write(&target, buf)?;
            count += 1;
            continue;
        }

        // Symlinks, hardlinks, character devices, etc. are not
        // expected in a published crate tarball — skip them.
    }

    if count == 0 {
        // A real .crate always contains at least Cargo.toml and
        // src/lib.rs; a 0-file extraction means we got an empty or
        // malformed archive.
        return Err(Error::EmptyTarball);
    }

    Ok(count)
}

/// Drop the leading `<name>-<version>/` component, returning the
/// remainder relative path. Returns `None` for the top-level
/// directory entry itself.
fn strip_top_level(path: &Path) -> Option<PathBuf> {
    let mut components = path.components();
    components.next()?; // discard the prefix
    let rest: PathBuf = components.collect();
    if rest.as_os_str().is_empty() {
        None
    } else {
        Some(rest)
    }
}

/// `path` is safe iff every component is a normal name (no `..`,
/// no root, no prefix, no current-dir).
fn is_safe_relative(path: &Path) -> bool {
    path.components().all(|c| matches!(c, Component::Normal(_)))
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::{Compression, write::GzEncoder};
    use tempfile::TempDir;

    use super::*;

    /// Build a synthetic `.crate` tarball wrapping a single top-level
    /// directory `<name>-<version>/`. `entries` are file paths relative
    /// to that directory and their byte contents.
    fn build_tarball(top: &str, entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut tar_bytes = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut tar_bytes);
            for (path, body) in entries {
                let full = format!("{top}/{path}");
                let mut header = tar::Header::new_gnu();
                header.set_path(&full).unwrap();
                header.set_size(body.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                builder.append(&header, *body).unwrap();
            }
            builder.finish().unwrap();
        }
        let mut gz = GzEncoder::new(Vec::new(), Compression::default());
        gz.write_all(&tar_bytes).unwrap();
        gz.finish().unwrap()
    }

    #[test]
    fn extracts_files_stripping_top_level_dir() {
        let tarball = build_tarball(
            "serde-1.0.219",
            &[
                ("Cargo.toml", b"[package]\nname=\"serde\"\n"),
                ("src/lib.rs", b"// serde root"),
            ],
        );
        let dest = TempDir::new().unwrap();
        let count = extract_crate_tarball(&tarball, dest.path()).unwrap();

        assert_eq!(count, 2);
        assert_eq!(
            fs::read_to_string(dest.path().join("Cargo.toml")).unwrap(),
            "[package]\nname=\"serde\"\n",
        );
        assert_eq!(
            fs::read_to_string(dest.path().join("src/lib.rs")).unwrap(),
            "// serde root",
        );
    }

    #[test]
    fn creates_destination_if_missing() {
        let tmp = TempDir::new().unwrap();
        let dest = tmp.path().join("nested/target");
        let tarball = build_tarball("x-0.1.0", &[("Cargo.toml", b"[package]")]);
        extract_crate_tarball(&tarball, &dest).unwrap();
        assert!(dest.join("Cargo.toml").exists());
    }

    #[test]
    fn rejects_path_traversal_via_dotdot() {
        // tar::Builder::set_path validates paths upfront, so we have to
        // construct a malicious header by writing the raw `..`-bearing
        // bytes into the name field directly and re-computing the
        // checksum. This is exactly the shape of attack the
        // `is_safe_relative` check defends against — a malicious
        // tarball wouldn't go through tar::Builder either.
        let body = b"stolen";
        let mut header = tar::Header::new_ustar();
        header.set_size(body.len() as u64);
        header.set_mode(0o644);
        let evil = b"evil-1.0.0/../escape.txt";
        let header_bytes = header.as_mut_bytes();
        header_bytes[..evil.len()].copy_from_slice(evil);
        header.set_cksum();

        let mut tar_bytes = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut tar_bytes);
            builder.append(&header, &body[..]).unwrap();
            builder.finish().unwrap();
        }
        let mut gz = GzEncoder::new(Vec::new(), Compression::default());
        gz.write_all(&tar_bytes).unwrap();
        let tarball = gz.finish().unwrap();

        let dest = TempDir::new().unwrap();
        let err = extract_crate_tarball(&tarball, dest.path()).unwrap_err();
        assert!(matches!(err, Error::UnsafeTarballEntry { .. }));
    }

    #[test]
    fn is_safe_relative_accepts_normal_components() {
        assert!(is_safe_relative(Path::new("src/lib.rs")));
        assert!(is_safe_relative(Path::new("Cargo.toml")));
        assert!(is_safe_relative(Path::new("a/b/c/d.rs")));
    }

    #[test]
    fn is_safe_relative_rejects_dotdot() {
        assert!(!is_safe_relative(Path::new("../escape.txt")));
        assert!(!is_safe_relative(Path::new("a/../b")));
        assert!(!is_safe_relative(Path::new("a/b/..")));
    }

    #[test]
    fn is_safe_relative_rejects_absolute() {
        assert!(!is_safe_relative(Path::new("/etc/passwd")));
    }

    #[test]
    fn strip_top_level_returns_remainder() {
        assert_eq!(
            strip_top_level(Path::new("serde-1.0.0/src/lib.rs")),
            Some(PathBuf::from("src/lib.rs")),
        );
    }

    #[test]
    fn strip_top_level_returns_none_for_top_dir_itself() {
        assert_eq!(strip_top_level(Path::new("serde-1.0.0")), None);
        assert_eq!(strip_top_level(Path::new("serde-1.0.0/")), None);
    }

    #[test]
    fn empty_tarball_errors() {
        let tarball = build_tarball("x-0.1.0", &[]);
        let dest = TempDir::new().unwrap();
        let err = extract_crate_tarball(&tarball, dest.path()).unwrap_err();
        assert!(matches!(err, Error::EmptyTarball));
    }

    #[test]
    fn nested_directory_entries_are_created() {
        let tarball = build_tarball(
            "x-0.1.0",
            &[("src/deep/path/file.rs", b"// deep file")],
        );
        let dest = TempDir::new().unwrap();
        extract_crate_tarball(&tarball, dest.path()).unwrap();
        assert!(dest.path().join("src/deep/path/file.rs").exists());
    }

    #[test]
    fn malformed_gzip_errors() {
        let dest = TempDir::new().unwrap();
        let err = extract_crate_tarball(b"not gzip", dest.path()).unwrap_err();
        assert!(matches!(err, Error::Io(_)));
    }

    #[hegel::test(test_cases = 20)]
    fn prop_extract_writes_every_file(tc: hegel::TestCase) {
        use hegel::generators as gs;

        let n: u8 = tc.draw(gs::integers::<u8>().min_value(1).max_value(5));
        let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
        for i in 0..n {
            let basename: String = tc.draw(
                gs::text().alphabet("abcdefghij").min_size(1).max_size(8),
            );
            let path = format!("src/{basename}_{i}.rs");
            let body: Vec<u8> =
                tc.draw(gs::vecs(gs::integers::<u8>()).max_size(64));
            entries.push((path, body));
        }

        let entry_refs: Vec<(&str, &[u8])> = entries
            .iter()
            .map(|(p, b)| (p.as_str(), b.as_slice()))
            .collect();
        let tarball = build_tarball("pkg-0.1.0", &entry_refs);

        let dest = TempDir::new().unwrap();
        let count = extract_crate_tarball(&tarball, dest.path()).unwrap();
        assert_eq!(count, entries.len());

        for (path, body) in &entries {
            let on_disk = fs::read(dest.path().join(path)).unwrap();
            assert_eq!(&on_disk, body, "content mismatch for {path}");
        }
    }
}
