use std::{
    path::{Path, PathBuf},
    time::SystemTime,
};

use ignore::WalkBuilder;

use crate::error::Result;

/// Document file found on disk.
///
/// Returned by [`discover_files`]. It keeps the relative path, absolute path,
/// and modification time needed for incremental indexing.
#[derive(Debug, Clone)]
pub struct DiscoveredFile {
    /// Path relative to the collection root directory (e.g., `"subdir/note.md"`).
    pub relative_path: PathBuf,
    /// Fully resolved absolute path on disk.
    pub absolute_path: PathBuf,
    /// Last modification time as seconds since the Unix epoch.
    pub mtime: u64,
}

/// Supported file extensions for document discovery.
const SUPPORTED_EXTENSIONS: &[&str] = &["md", "txt"];

/// Walk a directory tree and return the document files docbert can index.
///
/// Hidden files and directories are skipped. Only supported extensions (`.md`
/// and `.txt`) are returned. Results come back sorted by relative path.
///
/// If the collection root is a Git repository, Git ignore rules are respected
/// as well, including nested `.gitignore` files and `.git/info/exclude`.
///
/// # Examples
///
/// ```
/// # let tmp = tempfile::tempdir().unwrap();
/// # std::fs::write(tmp.path().join("note.md"), "# Hello").unwrap();
/// # std::fs::write(tmp.path().join("readme.txt"), "Hello").unwrap();
/// # std::fs::write(tmp.path().join("image.png"), "binary").unwrap();
/// use docbert_core::walker::discover_files;
///
/// let files = discover_files(tmp.path()).unwrap();
/// assert_eq!(files.len(), 2); // .md and .txt only
/// ```
pub fn discover_files(root: &Path) -> Result<Vec<DiscoveredFile>> {
    let canonical_root = root.canonicalize()?;
    let is_git_repo = canonical_root.join(".git").exists();
    let mut results = Vec::new();

    let mut builder = WalkBuilder::new(&canonical_root);
    builder
        .standard_filters(false)
        .hidden(true)
        .ignore(false)
        .parents(false)
        .git_ignore(is_git_repo)
        .git_global(is_git_repo)
        .git_exclude(is_git_repo)
        .follow_links(false);

    for entry in builder.build() {
        let entry = entry.map_err(std::io::Error::other)?;
        let path = entry.path();

        if path == canonical_root {
            continue;
        }

        let file_type = match entry.file_type() {
            Some(file_type) => file_type,
            None => std::fs::symlink_metadata(path)?.file_type(),
        };

        if file_type.is_dir() {
            continue;
        }

        if file_type.is_symlink() {
            // Resolve symlink and check for cycles.
            let resolved = match path.canonicalize() {
                Ok(p) => p,
                Err(_) => continue, // Skip broken symlinks
            };
            // Skip if the symlink points back into or above the root
            // (cycle prevention).
            if resolved.starts_with(&canonical_root) && resolved.is_dir() {
                continue;
            }
            if resolved.is_file() && is_supported(&resolved) {
                results.push(make_discovered(
                    &canonical_root,
                    path,
                    &resolved,
                )?);
            }
            continue;
        }

        if file_type.is_file() && is_supported(path) {
            let abs = path.canonicalize()?;
            results.push(make_discovered(&canonical_root, path, &abs)?);
        }
    }

    results.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    Ok(results)
}

fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| SUPPORTED_EXTENSIONS.contains(&ext))
}

fn make_discovered(
    root: &Path,
    original_path: &Path,
    absolute_path: &Path,
) -> Result<DiscoveredFile> {
    let relative_path = original_path
        .strip_prefix(root)
        .unwrap_or(original_path)
        .to_path_buf();

    let mtime = std::fs::metadata(absolute_path)?
        .modified()
        .unwrap_or(SystemTime::UNIX_EPOCH)
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Ok(DiscoveredFile {
        relative_path,
        absolute_path: absolute_path.to_path_buf(),
        mtime,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_md_and_txt() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("note.md"), "# Hello").unwrap();
        std::fs::write(tmp.path().join("readme.txt"), "Hello").unwrap();
        std::fs::write(tmp.path().join("image.png"), "binary").unwrap();

        let files = discover_files(tmp.path()).unwrap();
        assert_eq!(files.len(), 2);

        let names: Vec<_> = files
            .iter()
            .map(|f| f.relative_path.to_string_lossy().to_string())
            .collect();
        assert!(names.contains(&"note.md".to_string()));
        assert!(names.contains(&"readme.txt".to_string()));
    }

    #[test]
    fn skips_hidden_files() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(".hidden.md"), "secret").unwrap();
        std::fs::write(tmp.path().join("visible.md"), "hello").unwrap();

        let files = discover_files(tmp.path()).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].relative_path.to_string_lossy(), "visible.md");
    }

    #[test]
    fn skips_hidden_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let hidden = tmp.path().join(".git");
        std::fs::create_dir(&hidden).unwrap();
        std::fs::write(hidden.join("config.md"), "git config").unwrap();
        std::fs::write(tmp.path().join("notes.md"), "notes").unwrap();

        let files = discover_files(tmp.path()).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].relative_path.to_string_lossy(), "notes.md");
    }

    #[test]
    fn recurses_subdirectories() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("subdir");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("deep.md"), "deep").unwrap();
        std::fs::write(tmp.path().join("top.md"), "top").unwrap();

        let files = discover_files(tmp.path()).unwrap();
        assert_eq!(files.len(), 2);

        let paths: Vec<_> = files
            .iter()
            .map(|f| f.relative_path.to_string_lossy().to_string())
            .collect();
        assert!(paths.contains(&"top.md".to_string()));
        assert!(paths.contains(&"subdir/deep.md".to_string()));
    }

    #[test]
    fn respects_gitignore_when_collection_root_is_a_git_repo() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join(".git")).unwrap();
        std::fs::write(tmp.path().join(".gitignore"), "ignored.md\nsubdir/\n")
            .unwrap();
        std::fs::write(tmp.path().join("ignored.md"), "ignore me").unwrap();
        std::fs::write(tmp.path().join("visible.md"), "keep me").unwrap();
        std::fs::create_dir(tmp.path().join("subdir")).unwrap();
        std::fs::write(tmp.path().join("subdir/nested.md"), "ignore me too")
            .unwrap();

        let files = discover_files(tmp.path()).unwrap();
        let paths: Vec<_> = files
            .iter()
            .map(|f| f.relative_path.to_string_lossy().to_string())
            .collect();

        assert_eq!(paths, vec!["visible.md"]);
    }

    #[test]
    fn ignores_gitignore_rules_when_root_is_not_a_git_repo() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(".gitignore"), "ignored.md\n").unwrap();
        std::fs::write(tmp.path().join("ignored.md"), "should still index")
            .unwrap();
        std::fs::write(tmp.path().join("visible.md"), "keep me").unwrap();

        let files = discover_files(tmp.path()).unwrap();
        let paths: Vec<_> = files
            .iter()
            .map(|f| f.relative_path.to_string_lossy().to_string())
            .collect();

        assert_eq!(paths, vec!["ignored.md", "visible.md"]);
    }

    #[test]
    fn mtime_is_nonzero() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("file.md"), "content").unwrap();

        let files = discover_files(tmp.path()).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].mtime > 0);
    }

    #[test]
    fn results_are_sorted() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("z.md"), "z").unwrap();
        std::fs::write(tmp.path().join("a.md"), "a").unwrap();
        std::fs::write(tmp.path().join("m.md"), "m").unwrap();

        let files = discover_files(tmp.path()).unwrap();
        let names: Vec<_> = files
            .iter()
            .map(|f| f.relative_path.to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["a.md", "m.md", "z.md"]);
    }

    #[test]
    fn empty_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let files = discover_files(tmp.path()).unwrap();
        assert!(files.is_empty());
    }
}
