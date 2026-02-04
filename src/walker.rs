use std::{
    path::{Path, PathBuf},
    time::SystemTime,
};

use crate::error::Result;

/// A discovered document file.
#[derive(Debug, Clone)]
pub struct DiscoveredFile {
    /// Path relative to the collection root directory.
    pub relative_path: PathBuf,
    /// Fully resolved absolute path.
    pub absolute_path: PathBuf,
    /// Last modification time as seconds since the Unix epoch.
    pub mtime: u64,
}

/// Supported file extensions for document discovery.
const SUPPORTED_EXTENSIONS: &[&str] = &["md", "txt"];

/// Recursively walk a directory and discover eligible document files.
///
/// Skips hidden files/directories (names starting with `.`) and only
/// returns files with supported extensions (.md, .txt).
pub fn discover_files(root: &Path) -> Result<Vec<DiscoveredFile>> {
    let canonical_root = root.canonicalize()?;
    let mut results = Vec::new();
    walk_dir(&canonical_root, &canonical_root, &mut results)?;
    results.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    Ok(results)
}

fn walk_dir(
    root: &Path,
    current: &Path,
    results: &mut Vec<DiscoveredFile>,
) -> Result<()> {
    let entries = std::fs::read_dir(current)?;

    for entry in entries {
        let entry = entry?;
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();

        // Skip hidden files and directories.
        if name.starts_with('.') {
            continue;
        }

        let file_type = entry.file_type()?;

        if file_type.is_dir() {
            walk_dir(root, &entry.path(), results)?;
        } else if file_type.is_symlink() {
            // Resolve symlink and check for cycles.
            let resolved = match entry.path().canonicalize() {
                Ok(p) => p,
                Err(_) => continue, // Skip broken symlinks
            };
            // Skip if the symlink points back into or above the root
            // (cycle prevention).
            if resolved.starts_with(root) && resolved.is_dir() {
                continue;
            }
            if resolved.is_file()
                && is_supported(&resolved)
                && let Some(df) =
                    make_discovered(root, &entry.path(), &resolved)?
            {
                results.push(df);
            }
        } else if file_type.is_file() && is_supported(&entry.path()) {
            let abs = entry.path().canonicalize()?;
            if let Some(df) = make_discovered(root, &entry.path(), &abs)? {
                results.push(df);
            }
        }
    }

    Ok(())
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
) -> Result<Option<DiscoveredFile>> {
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

    Ok(Some(DiscoveredFile {
        relative_path,
        absolute_path: absolute_path.to_path_buf(),
        mtime,
    }))
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
