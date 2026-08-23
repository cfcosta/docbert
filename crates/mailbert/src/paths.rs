//! Where the tool keeps its data, and where it reads its configuration.
//!
//! Each path has three sources, and §1.1 puts them in this order: the
//! flag on the command line, then the environment, then the XDG
//! default.

use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};

use crate::error::{Error, Result};

/// The environment variable that moves the data directory (§1.1).
pub const DATA_VAR: &str = "MAILBERT_DATA_DIR";

/// The environment variable that moves the configuration file (§1.2).
pub const CONFIG_VAR: &str = "MAILBERT_CONFIG";

/// The directory that mailbert adds to the XDG homes.
pub const PREFIX: &str = "mailbert";

/// The name of the configuration file inside the XDG config home.
pub const CONFIG_NAME: &str = "config.toml";

/// The files of §1.1, and the configuration file of §1.2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    /// The directory that holds the databases and the index.
    pub data: PathBuf,

    /// The TOML file that holds the accounts.
    pub config: PathBuf,
}

impl Paths {
    /// The paths that the flags, the environment, and XDG give.
    ///
    /// # Errors
    ///
    /// The function fails if no source gives a path, which happens when
    /// the machine has no home directory.
    pub fn find(data: Option<&Path>, config: Option<&Path>) -> Result<Self> {
        let seen = std::env::var_os(DATA_VAR);
        let data =
            pick(data, seen.as_deref(), data_home()).ok_or(Error::NoDataDir)?;

        let seen = std::env::var_os(CONFIG_VAR);
        let config = pick(config, seen.as_deref(), config_home())
            .ok_or(Error::NoConfigDir)?;

        Ok(Self { data, config })
    }

    /// The messages, the threads, the contacts, and the sync state.
    pub fn meta(&self) -> PathBuf {
        self.data.join("meta.db")
    }

    /// The raw bytes of each message.
    pub fn blobs(&self) -> PathBuf {
        self.data.join("blobs.db")
    }

    /// The token embeddings of each chunk.
    pub fn embeddings(&self) -> PathBuf {
        self.data.join("embeddings.db")
    }

    /// The PLAID index of the semantic leg.
    pub fn plaid(&self) -> PathBuf {
        self.data.join("plaid.idx")
    }

    /// The directory of the lexical index.
    pub fn tantivy(&self) -> PathBuf {
        self.data.join("tantivy")
    }

    /// Every file and directory of §1.1, in one list.
    pub fn files(&self) -> Vec<PathBuf> {
        vec![
            self.meta(),
            self.blobs(),
            self.embeddings(),
            self.plaid(),
            self.tantivy(),
        ]
    }

    /// Make the data directory, and the directories above it.
    pub fn make(&self) -> Result<()> {
        std::fs::create_dir_all(&self.data)?;

        Ok(())
    }
}

/// Take the first path that a source gives.
///
/// An empty environment variable gives nothing, because a shell that
/// clears a variable with `VAR=` must not move the directory.
pub fn pick(
    flag: Option<&Path>,
    env: Option<&OsStr>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    if let Some(path) = flag {
        return Some(path.to_path_buf());
    }

    if let Some(text) = env.filter(|text| !text.is_empty()) {
        return Some(PathBuf::from(text));
    }

    home
}

/// The XDG data home of mailbert.
fn data_home() -> Option<PathBuf> {
    xdg::BaseDirectories::with_prefix(PREFIX).get_data_home()
}

/// The XDG configuration file of mailbert.
fn config_home() -> Option<PathBuf> {
    let home = xdg::BaseDirectories::with_prefix(PREFIX).get_config_home()?;

    Some(home.join(CONFIG_NAME))
}

#[cfg(test)]
mod tests {
    //! # Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_pick_takes_the_first_source_that_speaks` | model-based | §1.1 gives an order to the three sources. A tool that reads the wrong one writes the mail of one machine into the directory of another. |
    //! | `prop_every_file_sits_in_the_data_directory` | invariant | `--data-dir` must move all of §1.1. A file that stays behind makes two half stores. |

    use std::ffi::OsString;

    use hegel::{TestCase, generators as gs};

    use super::*;

    fn at(text: &str) -> PathBuf {
        PathBuf::from(text)
    }

    #[test]
    fn the_flag_wins_over_the_environment() {
        let found = pick(
            Some(&at("/flag")),
            Some(OsStr::new("/env")),
            Some(at("/xdg")),
        );

        assert_eq!(found, Some(at("/flag")));
    }

    #[test]
    fn the_environment_wins_over_xdg() {
        let found = pick(None, Some(OsStr::new("/env")), Some(at("/xdg")));

        assert_eq!(found, Some(at("/env")));
    }

    #[test]
    fn xdg_speaks_last() {
        let found = pick(None, None, Some(at("/xdg")));

        assert_eq!(found, Some(at("/xdg")));
    }

    #[test]
    fn an_empty_environment_value_gives_nothing() {
        let found = pick(None, Some(OsStr::new("")), Some(at("/xdg")));

        assert_eq!(found, Some(at("/xdg")));
    }

    #[test]
    fn a_silent_machine_gives_no_path() {
        assert_eq!(pick(None, None, None), None);
    }

    #[test]
    fn a_flag_gives_the_data_directory() {
        let paths = Paths::find(Some(&at("/data")), Some(&at("/c.toml")))
            .expect("the flags give both paths");

        assert_eq!(paths.data, at("/data"));
        assert_eq!(paths.config, at("/c.toml"));
    }

    #[test]
    fn the_data_directory_holds_the_files_of_the_design() {
        let paths = Paths::find(Some(&at("/data")), Some(&at("/c.toml")))
            .expect("the flags give both paths");

        assert_eq!(paths.meta(), at("/data/meta.db"));
        assert_eq!(paths.blobs(), at("/data/blobs.db"));
        assert_eq!(paths.embeddings(), at("/data/embeddings.db"));
        assert_eq!(paths.plaid(), at("/data/plaid.idx"));
        assert_eq!(paths.tantivy(), at("/data/tantivy"));
    }

    #[test]
    fn the_default_configuration_file_is_a_toml_file() {
        let home = config_home().expect("the test machine has a home");

        assert!(home.ends_with("mailbert/config.toml"), "{home:?}");
    }

    #[test]
    fn make_builds_the_directory_that_is_not_there() {
        let temp = tempfile::tempdir().expect("a temporary directory");
        let data = temp.path().join("deep/data");
        let paths = Paths::find(Some(&data), Some(&at("/c.toml")))
            .expect("the flags give both paths");

        paths.make().expect("make builds the directory");

        assert!(data.is_dir(), "{data:?}");
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_pick_takes_the_first_source_that_speaks(tc: TestCase) {
        let names: Vec<Option<String>> = tc.draw(
            gs::vecs(gs::optional(
                gs::text().alphabet("abc/").min_size(1).max_size(5),
            ))
            .min_size(3)
            .max_size(3),
        );

        let flag = names[0].as_ref().map(PathBuf::from);
        let env = names[1].as_ref().map(OsString::from);
        let home = names[2].as_ref().map(PathBuf::from);

        let found = pick(flag.as_deref(), env.as_deref(), home.clone());
        let model = [flag, env.as_ref().map(PathBuf::from), home]
            .into_iter()
            .flatten()
            .next();

        assert_eq!(found, model);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_every_file_sits_in_the_data_directory(tc: TestCase) {
        let name: String =
            tc.draw(gs::text().alphabet("abcdef").min_size(1).max_size(8));
        let data = at("/root").join(&name);
        let paths = Paths::find(Some(&data), Some(&at("/c.toml")))
            .expect("the flags give both paths");

        for file in paths.files() {
            assert!(file.starts_with(&data), "{file:?} left {data:?}");
        }
    }
}
