use std::path::{Component, Path, PathBuf};

use crate::error::{Error, Result};

/// Validate and clean a user-supplied relative document path.
///
/// Rejects absolute paths, parent directory traversals (`..`), and empty paths.
/// Strips redundant `.` components.
///
/// # Examples
///
/// ```
/// use docbert_core::path_safety::sanitize_relative_path;
///
/// // Normal relative paths pass through
/// assert_eq!(
///     sanitize_relative_path("nested/deep/file.md").unwrap(),
///     std::path::PathBuf::from("nested/deep/file.md"),
/// );
///
/// // Dot components are stripped
/// assert_eq!(
///     sanitize_relative_path("./file.md").unwrap(),
///     std::path::PathBuf::from("file.md"),
/// );
///
/// // Parent traversal is rejected
/// assert!(sanitize_relative_path("../secret.md").is_err());
///
/// // Absolute paths are rejected
/// assert!(sanitize_relative_path("/etc/passwd").is_err());
///
/// // Empty paths are rejected
/// assert!(sanitize_relative_path("").is_err());
/// ```
pub fn sanitize_relative_path(relative_path: &str) -> Result<PathBuf> {
    if relative_path.trim().is_empty() {
        return Err(Error::Config("document path cannot be empty".to_string()));
    }

    let mut cleaned = PathBuf::new();
    for component in Path::new(relative_path).components() {
        match component {
            Component::Normal(part) => cleaned.push(part),
            Component::CurDir => {}
            Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => {
                return Err(Error::Config(format!(
                    "document path must not contain parent traversal or absolute components: {relative_path}"
                )));
            }
        }
    }

    if cleaned.as_os_str().is_empty() {
        return Err(Error::Config("document path cannot be empty".to_string()));
    }

    Ok(cleaned)
}

/// Resolve a relative document path within a collection root and verify it
/// doesn't escape.
///
/// This is the single entry point for turning `(collection_root, relative_path)`
/// into a safe filesystem path. It:
/// 1. Sanitizes the relative path (rejects `..`, `/`, etc.)
/// 2. Joins it with the root
/// 3. If the result exists, canonicalizes it and verifies it's still under root
///
/// **Note:** if the file doesn't exist yet (e.g. pre-upload), we verify the
/// nearest existing ancestor stays under root.
pub fn resolve_safe_document_path(
    collection_root: &Path,
    relative_path: &str,
) -> Result<PathBuf> {
    let sanitized = sanitize_relative_path(relative_path)?;
    let candidate = collection_root.join(&sanitized);
    ensure_path_within_root(collection_root, &candidate)?;
    Ok(candidate)
}

fn ensure_path_within_root(root: &Path, candidate: &Path) -> Result<()> {
    if candidate.exists() {
        let resolved = candidate.canonicalize()?;
        let canonical_root = root.canonicalize()?;
        if !resolved.starts_with(&canonical_root) {
            return Err(Error::Config(format!(
                "document path resolves outside the collection root: {}",
                candidate.display()
            )));
        }
        return Ok(());
    }

    let existing_ancestor = nearest_existing_ancestor(candidate)?;
    let resolved_ancestor = existing_ancestor.canonicalize()?;
    let canonical_root = root.canonicalize()?;
    if !resolved_ancestor.starts_with(&canonical_root) {
        return Err(Error::Config(format!(
            "document path resolves outside the collection root: {}",
            candidate.display()
        )));
    }

    Ok(())
}

fn nearest_existing_ancestor(path: &Path) -> Result<PathBuf> {
    let mut current = path.to_path_buf();
    loop {
        if current.exists() {
            return Ok(current);
        }
        if !current.pop() {
            return Err(Error::Config(format!(
                "document path has no existing ancestor: {}",
                path.display()
            )));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    // -- sanitize_relative_path unit tests --

    #[test]
    fn sanitize_accepts_simple_relative() {
        assert_eq!(
            sanitize_relative_path("file.md").unwrap(),
            PathBuf::from("file.md")
        );
    }

    #[test]
    fn sanitize_accepts_nested_relative() {
        assert_eq!(
            sanitize_relative_path("a/b/c.md").unwrap(),
            PathBuf::from("a/b/c.md")
        );
    }

    #[test]
    fn sanitize_strips_current_dir() {
        assert_eq!(
            sanitize_relative_path("./file.md").unwrap(),
            PathBuf::from("file.md")
        );
    }

    #[test]
    fn sanitize_rejects_parent_dir() {
        assert!(sanitize_relative_path("../secret.md").is_err());
    }

    #[test]
    fn sanitize_rejects_embedded_parent_dir() {
        assert!(sanitize_relative_path("a/../../etc/passwd").is_err());
    }

    #[test]
    fn sanitize_rejects_absolute_path() {
        assert!(sanitize_relative_path("/etc/passwd").is_err());
    }

    #[test]
    fn sanitize_rejects_empty() {
        assert!(sanitize_relative_path("").is_err());
    }

    #[test]
    fn sanitize_rejects_whitespace_only() {
        assert!(sanitize_relative_path("   ").is_err());
    }

    #[test]
    fn sanitize_rejects_dot_only() {
        assert!(sanitize_relative_path(".").is_err());
    }

    // -- resolve_safe_document_path tests --

    #[test]
    fn resolve_safe_accepts_nested_relative() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("col");
        fs::create_dir_all(&root).unwrap();

        let result =
            resolve_safe_document_path(&root, "nested/file.md").unwrap();
        assert_eq!(result, root.join("nested/file.md"));
    }

    #[test]
    fn resolve_safe_rejects_parent_traversal() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("col");
        fs::create_dir_all(&root).unwrap();

        assert!(resolve_safe_document_path(&root, "../secret.md").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn resolve_safe_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("col");
        let outside = tmp.path().join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("secret.md"), "secret").unwrap();
        symlink(outside.join("secret.md"), root.join("linked.md")).unwrap();

        assert!(resolve_safe_document_path(&root, "linked.md").is_err());
    }

    // -- Property tests --

    #[hegel::test(test_cases = 200)]
    fn prop_sanitize_never_produces_parent_component(tc: hegel::TestCase) {
        use hegel::generators as gs;

        // Generate path segments that may include "..", ".", normal names
        let segment_gen = hegel::one_of!(
            gs::text()
                .min_size(1)
                .max_size(10)
                .alphabet("abcdefghijklmnopqrstuvwxyz0123456789_-."),
            gs::sampled_from(vec![
                "..".to_string(),
                ".".to_string(),
                "".to_string()
            ])
        );
        let segments: Vec<String> =
            tc.draw(gs::vecs(segment_gen).min_size(1).max_size(6));
        let path_str = segments.join("/");

        match sanitize_relative_path(&path_str) {
            Ok(cleaned) => {
                // If it succeeds, the result must not contain ParentDir or RootDir
                for component in cleaned.components() {
                    assert!(
                        !matches!(
                            component,
                            Component::ParentDir
                                | Component::RootDir
                                | Component::Prefix(_)
                        ),
                        "sanitized path contained forbidden component: {}",
                        cleaned.display()
                    );
                }
                // And must not be empty
                assert!(!cleaned.as_os_str().is_empty());
            }
            Err(_) => {
                // Rejection is always safe
            }
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_resolve_safe_stays_under_root(tc: hegel::TestCase) {
        use hegel::generators as gs;

        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("col");
        fs::create_dir_all(&root).unwrap();

        let segment_gen = hegel::one_of!(
            gs::text()
                .min_size(1)
                .max_size(8)
                .alphabet("abcdefghijklmnopqrstuvwxyz0123456789_-"),
            gs::sampled_from(vec!["..".to_string(), ".".to_string()])
        );
        let segments: Vec<String> =
            tc.draw(gs::vecs(segment_gen).min_size(1).max_size(4));
        let path_str = segments.join("/");

        match resolve_safe_document_path(&root, &path_str) {
            Ok(resolved) => {
                // resolve_safe_document_path returns a non-canonicalized join,
                // so compare against the non-canonical root. The real safety
                // check (ensure_path_within_root) canonicalizes both sides.
                assert!(
                    resolved.starts_with(&root),
                    "resolved path {:?} escapes root {:?}",
                    resolved,
                    root,
                );
            }
            Err(_) => {
                // Rejection is always safe
            }
        }
    }

    #[hegel::test(test_cases = 100)]
    fn prop_sanitize_roundtrip_idempotent(tc: hegel::TestCase) {
        use hegel::generators as gs;

        let segment_gen = gs::text()
            .min_size(1)
            .max_size(8)
            .alphabet("abcdefghijklmnopqrstuvwxyz0123456789_-.");
        let segments: Vec<String> =
            tc.draw(gs::vecs(segment_gen).min_size(1).max_size(4));
        let path_str = segments.join("/");

        if let Ok(first) = sanitize_relative_path(&path_str) {
            let second =
                sanitize_relative_path(&first.to_string_lossy()).unwrap();
            assert_eq!(first, second, "sanitize is not idempotent");
        }
    }
}
