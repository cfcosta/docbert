//! The mailbert configuration file.
//!
//! mailbert reads TOML from `$XDG_CONFIG_HOME/mailbert/config.toml`.
//! Accounts are an array of tables. See `docs/mailbert.md` §1.2.
//!
//! ```toml
//! [[account]]
//! name             = "work"
//! host             = "imap.fastmail.com"
//! user             = "me@work.example"
//! password_command = "pass show mail/work"
//! folders          = ["INBOX", "Archive"]
//! exclude          = ["Trash"]
//! ```

use std::{collections::HashSet, path::PathBuf};

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// The IMAPS port. mailbert does not speak cleartext IMAP.
pub const DEFAULT_PORT: u16 = 993;

/// Parallel IMAP connections for each account. Gmail permits 15, and
/// some servers permit 4, so this is configurable.
pub const DEFAULT_CONNECTIONS: usize = 8;

/// Results that `search` prints when `--count` is absent.
pub const DEFAULT_COUNT: usize = 20;

/// Half-life of the recency prior, in days. See `docs/mailbert.md` §8.3.
pub const DEFAULT_HALF_LIFE_DAYS: f64 = 180.0;

/// The name that RFC 3501 makes case-insensitive.
const INBOX: &str = "INBOX";

/// The whole configuration file.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub search: SearchConfig,

    #[serde(default)]
    pub view: ViewConfig,

    /// Written as `[[account]]`, so the TOML key is singular.
    #[serde(
        default,
        rename = "account",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub accounts: Vec<Account>,
}

/// One IMAP account.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Account {
    pub name: String,
    pub host: String,
    pub user: String,

    #[serde(default = "default_port")]
    pub port: u16,

    /// A shell command whose first line of output is the password. This
    /// is the isync convention, and it keeps the secret off disk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub password_command: Option<String>,

    /// A file whose first line is the password.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub password_file: Option<PathBuf>,

    /// The password itself. mailbert always warns about this one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub password: Option<String>,

    /// Folders to sync. Ignored when `all_folders` is set.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub folders: Vec<String>,

    /// Folders never to sync. Applied after `folders` and `all_folders`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclude: Vec<String>,

    /// Regular expressions that identify a corporate footer. A line
    /// that matches one, and each line after it, leaves the indexed
    /// text. See `docs/mailbert.md` §5.2 rule 4.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub footers: Vec<String>,

    /// Sync every folder the server lists.
    #[serde(default, skip_serializing_if = "is_false")]
    pub all_folders: bool,

    #[serde(default = "default_connections")]
    pub connections: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchConfig {
    #[serde(default = "default_count")]
    pub count: usize,

    #[serde(default = "default_half_life")]
    pub recency_half_life_days: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewConfig {
    #[serde(default = "default_theme")]
    pub theme: String,

    #[serde(default = "default_width")]
    pub width: usize,
}

fn default_port() -> u16 {
    DEFAULT_PORT
}

fn default_connections() -> usize {
    DEFAULT_CONNECTIONS
}

fn default_count() -> usize {
    DEFAULT_COUNT
}

fn default_half_life() -> f64 {
    DEFAULT_HALF_LIFE_DAYS
}

fn default_theme() -> String {
    "base16-ocean.dark".to_string()
}

fn default_width() -> usize {
    100
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            count: default_count(),
            recency_half_life_days: default_half_life(),
        }
    }
}

impl Default for ViewConfig {
    fn default() -> Self {
        Self {
            theme: default_theme(),
            width: default_width(),
        }
    }
}

/// Where an account's password comes from.
///
/// The variants are ordered by the precedence that [`Account::credential`]
/// applies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Credential {
    /// Run a shell command and read its first line.
    Command(String),
    /// Read the first line of a file.
    File(PathBuf),
    /// Use the value from the config file directly.
    Literal(String),
}

/// Compare two IMAP folder names.
///
/// RFC 3501 makes `INBOX` case-insensitive and says nothing about any
/// other name, so every other comparison is exact.
///
/// # Examples
///
/// ```
/// use mailbert_core::config::folder_eq;
///
/// assert!(folder_eq("INBOX", "inbox"));
/// assert!(!folder_eq("Archive", "archive"));
/// ```
pub fn folder_eq(a: &str, b: &str) -> bool {
    if a.eq_ignore_ascii_case(INBOX) {
        return b.eq_ignore_ascii_case(INBOX);
    }
    a == b
}

impl Config {
    /// Parse a configuration file and validate it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigParse`] when the TOML is malformed or has
    /// an unknown key, and whatever [`Config::validate`] reports.
    ///
    /// # Examples
    ///
    /// ```
    /// use mailbert_core::Config;
    ///
    /// let config = Config::parse(
    ///     r#"
    ///     [[account]]
    ///     name = "work"
    ///     host = "imap.example.com"
    ///     user = "me@example.com"
    ///     password_command = "pass show mail/work"
    /// "#,
    /// )
    /// .unwrap();
    ///
    /// assert_eq!(config.accounts.len(), 1);
    /// ```
    pub fn parse(toml_text: &str) -> Result<Self> {
        let config: Self = toml::from_str(toml_text)?;
        config.validate()?;
        Ok(config)
    }

    /// Check the invariants that parsing alone does not.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmptyAccountName`], [`Error::EmptyField`],
    /// [`Error::MissingCredential`], or [`Error::DuplicateAccount`].
    pub fn validate(&self) -> Result<()> {
        let mut seen: HashSet<&str> = HashSet::new();

        for account in &self.accounts {
            if account.name.is_empty() {
                return Err(Error::EmptyAccountName);
            }

            for (field, value) in
                [("host", &account.host), ("user", &account.user)]
            {
                if value.is_empty() {
                    return Err(Error::EmptyField {
                        account: account.name.clone(),
                        field,
                    });
                }
            }

            // Fail here rather than at sync time, when the user has
            // already waited for a connection.
            account.credential()?;
            account.footer_patterns()?;

            if !seen.insert(account.name.as_str()) {
                return Err(Error::DuplicateAccount(account.name.clone()));
            }
        }

        Ok(())
    }

    /// Find an account by name.
    pub fn account(&self, name: &str) -> Option<&Account> {
        self.accounts.iter().find(|a| a.name == name)
    }
}

impl Account {
    /// Resolve which credential source to use.
    ///
    /// Precedence is `password_command`, then `password_file`, then
    /// `password`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingCredential`] when all three are absent.
    pub fn credential(&self) -> Result<Credential> {
        if let Some(command) = &self.password_command {
            return Ok(Credential::Command(command.clone()));
        }

        if let Some(path) = &self.password_file {
            return Ok(Credential::File(path.clone()));
        }

        if let Some(password) = &self.password {
            return Ok(Credential::Literal(password.clone()));
        }

        Err(Error::MissingCredential(self.name.clone()))
    }

    /// Whether this account stores its password in the config file.
    ///
    /// The CLI warns when it does. A literal that precedence never
    /// reaches is not reported, because mailbert does not read it.
    pub fn has_plaintext_password(&self) -> bool {
        matches!(self.credential(), Ok(Credential::Literal(_)))
    }

    /// The folders to sync, given what the server listed.
    ///
    /// The result is a subset of `available` and holds the server's
    /// spelling, because that is the name mailbert must SELECT. It never
    /// contains an excluded folder and has no duplicates. `exclude`
    /// always wins, and `all_folders` ignores `folders`.
    pub fn select_folders(&self, available: &[String]) -> Vec<String> {
        let mut selected: Vec<String> = Vec::new();

        for folder in available {
            let wanted = self.all_folders
                || self.folders.iter().any(|f| folder_eq(f, folder));

            let skip = !wanted
                || self.exclude.iter().any(|e| folder_eq(e, folder))
                || selected.iter().any(|s| folder_eq(s, folder));

            if !skip {
                selected.push(folder.clone());
            }
        }

        selected
    }

    /// The compiled corporate footer patterns of this account.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFooter`] for the first pattern that
    /// does not compile.
    pub fn footer_patterns(&self) -> Result<Vec<Regex>> {
        let mut patterns = Vec::with_capacity(self.footers.len());

        for pattern in &self.footers {
            match Regex::new(pattern) {
                Ok(regex) => patterns.push(regex),
                Err(source) => {
                    return Err(Error::InvalidFooter {
                        account: self.name.clone(),
                        pattern: pattern.clone(),
                        source,
                    });
                }
            }
        }

        Ok(patterns)
    }

    /// Configured folders that the server did not list.
    ///
    /// The CLI warns about these, because a typo is otherwise silent.
    /// `all_folders` reports nothing, because it never reads `folders`.
    pub fn missing_folders(&self, available: &[String]) -> Vec<String> {
        if self.all_folders {
            return Vec::new();
        }

        let mut missing: Vec<String> = Vec::new();

        for folder in &self.folders {
            let known = available.iter().any(|a| folder_eq(a, folder));
            let seen = missing.iter().any(|m| folder_eq(m, folder));

            if !known && !seen {
                missing.push(folder.clone());
            }
        }

        missing
    }
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_config_round_trips_through_toml` | round-trip | A config mailbert writes must be one it can read. |
    //! | `prop_credential_precedence_is_total` | model-based | The precedence rule is the whole contract of the three password fields. Getting it wrong reads the wrong secret. |
    //! | `prop_selection_is_a_subset_of_available` | invariant | mailbert must never try to SELECT a folder the server did not list. |
    //! | `prop_exclude_always_wins` | invariant | A folder in both `folders` and `exclude` must not sync. This is how a user keeps Spam out. |
    //! | `prop_selection_has_no_duplicates` | invariant | A duplicate would fetch and hash the same folder twice. |
    //! | `prop_all_folders_selects_the_complement` | differential | `all_folders` must equal `available` minus `exclude`, exactly. |
    //! | `prop_folder_eq_is_an_equivalence` | algebraic | Reflexive, symmetric, transitive. Used to compare server names to config names. |
    //! | `prop_duplicate_names_always_rejected` | invariant | Two accounts with one name make `mailbert sync work` ambiguous. |

    use hegel::{TestCase, generators as gs};

    use super::*;

    // -----------------------------------------------------------------
    // Generators.
    // -----------------------------------------------------------------

    /// A folder name drawn from a small pool, so that overlap between
    /// `available`, `folders`, and `exclude` happens often enough to
    /// exercise the interesting paths.
    fn folder_name() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "INBOX".to_string(),
            "Archive".to_string(),
            "Sent".to_string(),
            "Drafts".to_string(),
            "Trash".to_string(),
            "Junk".to_string(),
            "[Gmail]/All Mail".to_string(),
        ])
    }

    fn folder_list() -> impl gs::Generator<Vec<String>> {
        gs::vecs(folder_name()).min_size(0).max_size(7)
    }

    /// An account with every folder-selection field drawn independently.
    #[hegel::composite]
    fn account_with_folders(tc: TestCase) -> (Account, Vec<String>) {
        let available: Vec<String> = tc.draw(folder_list());
        let folders: Vec<String> = tc.draw(folder_list());
        let exclude: Vec<String> = tc.draw(folder_list());
        let all_folders: bool = tc.draw(gs::booleans());

        let account = Account {
            name: "acct".to_string(),
            host: "imap.example.com".to_string(),
            user: "me@example.com".to_string(),
            port: DEFAULT_PORT,
            password_command: Some("true".to_string()),
            password_file: None,
            password: None,
            folders,
            exclude,
            footers: vec![],
            all_folders,
            connections: DEFAULT_CONNECTIONS,
        };

        (account, available)
    }

    /// A whole config with pairwise-distinct account names, so it is
    /// valid by construction.
    #[hegel::composite]
    fn valid_config(tc: TestCase) -> Config {
        let count: usize =
            tc.draw(gs::integers::<usize>().min_value(0).max_value(4));

        let accounts = (0..count)
            .map(|i| Account {
                name: format!("acct{i}"),
                host: format!("imap{i}.example.com"),
                user: format!("me{i}@example.com"),
                port: DEFAULT_PORT,
                password_command: Some(format!("pass show mail/{i}")),
                password_file: None,
                password: None,
                folders: vec!["INBOX".to_string()],
                exclude: vec![],
                footers: vec![],
                all_folders: false,
                connections: DEFAULT_CONNECTIONS,
            })
            .collect();

        Config {
            search: SearchConfig::default(),
            view: ViewConfig::default(),
            accounts,
        }
    }

    fn account_named(name: &str) -> Account {
        Account {
            name: name.to_string(),
            host: "imap.example.com".to_string(),
            user: "me@example.com".to_string(),
            port: DEFAULT_PORT,
            password_command: Some("true".to_string()),
            password_file: None,
            password: None,
            folders: vec![],
            exclude: vec![],
            footers: vec![],
            all_folders: true,
            connections: DEFAULT_CONNECTIONS,
        }
    }

    // -----------------------------------------------------------------
    // Unit tests.
    // -----------------------------------------------------------------

    const MINIMAL: &str = r#"
        [[account]]
        name = "work"
        host = "imap.fastmail.com"
        user = "me@work.example"
        password_command = "pass show mail/work"
    "#;

    #[test]
    fn parse_applies_the_documented_defaults() {
        let config = Config::parse(MINIMAL).unwrap();
        let account = &config.accounts[0];

        assert_eq!(account.port, DEFAULT_PORT);
        assert_eq!(account.connections, DEFAULT_CONNECTIONS);
        assert!(!account.all_folders);
        assert!(account.folders.is_empty());
        assert_eq!(config.search.count, DEFAULT_COUNT);
        assert_eq!(
            config.search.recency_half_life_days,
            DEFAULT_HALF_LIFE_DAYS
        );
        assert_eq!(config.view.width, 100);
    }

    #[test]
    fn parse_reads_the_documented_example() {
        let config = Config::parse(
            r#"
            [[account]]
            name             = "work"
            host             = "imap.fastmail.com"
            port             = 993
            user             = "me@work.example"
            password_command = "pass show mail/work"
            folders          = ["INBOX", "Archive", "Sent"]
            exclude          = ["Trash", "Junk"]
            connections      = 8

            [[account]]
            name          = "personal"
            host          = "imap.gmail.com"
            user          = "me@gmail.com"
            password_file = "~/.secrets/gmail"
            all_folders   = true

            [search]
            count = 20
            recency_half_life_days = 180

            [view]
            theme = "base16-ocean.dark"
            width = 100
        "#,
        )
        .unwrap();

        assert_eq!(config.accounts.len(), 2);
        assert_eq!(config.account("work").unwrap().folders.len(), 3);
        assert!(config.account("personal").unwrap().all_folders);
        assert!(config.account("nope").is_none());
    }

    #[test]
    fn parse_rejects_an_unknown_key() {
        // A typo in a key name must not be silently ignored.
        let err = Config::parse(
            r#"
            [[account]]
            name = "work"
            host = "imap.example.com"
            user = "me@example.com"
            password = "hunter2"
            fulders = ["INBOX"]
        "#,
        );

        assert!(matches!(err, Err(Error::ConfigParse(_))));
    }

    #[test]
    fn parse_rejects_duplicate_account_names() {
        let err = Config::parse(
            r#"
            [[account]]
            name = "work"
            host = "a.example.com"
            user = "a@example.com"
            password = "x"

            [[account]]
            name = "work"
            host = "b.example.com"
            user = "b@example.com"
            password = "y"
        "#,
        );

        assert!(
            matches!(err, Err(Error::DuplicateAccount(name)) if name == "work")
        );
    }

    #[test]
    fn parse_rejects_an_account_with_no_credential() {
        let err = Config::parse(
            r#"
            [[account]]
            name = "work"
            host = "imap.example.com"
            user = "me@example.com"
        "#,
        );

        assert!(
            matches!(err, Err(Error::MissingCredential(name)) if name == "work")
        );
    }

    #[test]
    fn parse_rejects_empty_required_fields() {
        assert!(matches!(
            Config::parse(
                r#"
                [[account]]
                name = ""
                host = "imap.example.com"
                user = "me@example.com"
                password = "x"
            "#
            ),
            Err(Error::EmptyAccountName)
        ));

        assert!(matches!(
            Config::parse(
                r#"
                [[account]]
                name = "work"
                host = ""
                user = "me@example.com"
                password = "x"
            "#
            ),
            Err(Error::EmptyField { field: "host", .. })
        ));
    }

    #[test]
    fn credential_precedence_is_command_then_file_then_literal() {
        let mut account = account_named("work");
        account.password_command = Some("pass show x".to_string());
        account.password_file = Some(PathBuf::from("/tmp/p"));
        account.password = Some("hunter2".to_string());

        assert_eq!(
            account.credential().unwrap(),
            Credential::Command("pass show x".to_string())
        );

        account.password_command = None;
        assert_eq!(
            account.credential().unwrap(),
            Credential::File(PathBuf::from("/tmp/p"))
        );

        account.password_file = None;
        assert_eq!(
            account.credential().unwrap(),
            Credential::Literal("hunter2".to_string())
        );

        account.password = None;
        assert!(matches!(
            account.credential(),
            Err(Error::MissingCredential(_))
        ));
    }

    #[test]
    fn plaintext_password_is_flagged_only_when_it_is_used() {
        let mut account = account_named("work");
        account.password_command = None;
        account.password = Some("hunter2".to_string());
        assert!(account.has_plaintext_password());

        // A literal that precedence never reaches is not a risk worth
        // warning about.
        account.password_command = Some("pass show x".to_string());
        assert!(!account.has_plaintext_password());
    }

    #[test]
    fn inbox_compares_case_insensitively_and_others_do_not() {
        assert!(folder_eq("INBOX", "inbox"));
        assert!(folder_eq("InBoX", "INBOX"));
        assert!(!folder_eq("Archive", "archive"));
        assert!(folder_eq("Archive", "Archive"));
    }

    #[test]
    fn select_folders_uses_the_server_spelling_of_inbox() {
        let mut account = account_named("work");
        account.all_folders = false;
        account.folders = vec!["inbox".to_string()];

        let available = vec!["INBOX".to_string()];

        // The name the server gave is the one mailbert must SELECT.
        assert_eq!(account.select_folders(&available), vec!["INBOX"]);
        assert!(account.missing_folders(&available).is_empty());
    }

    #[test]
    fn missing_folders_reports_a_typo() {
        let mut account = account_named("work");
        account.all_folders = false;
        account.folders = vec!["Archve".to_string(), "INBOX".to_string()];

        let available = vec!["INBOX".to_string(), "Archive".to_string()];

        assert_eq!(account.missing_folders(&available), vec!["Archve"]);
    }

    #[test]
    fn all_folders_ignores_the_folders_list() {
        let mut account = account_named("work");
        account.all_folders = true;
        account.folders = vec!["Sent".to_string()];
        account.exclude = vec!["Trash".to_string()];

        let available =
            vec!["INBOX".to_string(), "Sent".to_string(), "Trash".to_string()];

        assert_eq!(account.select_folders(&available), vec!["INBOX", "Sent"]);
        // `folders` is not consulted, so nothing can be missing.
        assert!(account.missing_folders(&available).is_empty());
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 100)]
    fn prop_config_round_trips_through_toml(tc: TestCase) {
        let config = tc.draw(valid_config());

        let text = toml::to_string(&config).expect("config serializes");
        let parsed = Config::parse(&text).expect("serialized config parses");

        assert_eq!(config, parsed);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_credential_precedence_is_total(tc: TestCase) {
        let command: Option<String> =
            tc.draw(gs::optional(gs::just("cmd".to_string())));
        let file: Option<String> =
            tc.draw(gs::optional(gs::just("file".to_string())));
        let literal: Option<String> =
            tc.draw(gs::optional(gs::just("lit".to_string())));

        let mut account = account_named("work");
        account.password_command = command.clone();
        account.password_file = file.clone().map(PathBuf::from);
        account.password = literal.clone();

        // The model: first present field, in declaration order, wins.
        let expected = match (&command, &file, &literal) {
            (Some(c), _, _) => Some(Credential::Command(c.clone())),
            (None, Some(f), _) => Some(Credential::File(PathBuf::from(f))),
            (None, None, Some(l)) => Some(Credential::Literal(l.clone())),
            (None, None, None) => None,
        };

        match expected {
            Some(want) => assert_eq!(account.credential().unwrap(), want),
            None => assert!(matches!(
                account.credential(),
                Err(Error::MissingCredential(_))
            )),
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_selection_is_a_subset_of_available(tc: TestCase) {
        let (account, available) = tc.draw(account_with_folders());

        for selected in account.select_folders(&available) {
            assert!(
                available.iter().any(|a| a == &selected),
                "selected {selected:?} is not in {available:?}"
            );
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_exclude_always_wins(tc: TestCase) {
        let (account, available) = tc.draw(account_with_folders());

        for selected in account.select_folders(&available) {
            assert!(
                !account.exclude.iter().any(|e| folder_eq(e, &selected)),
                "excluded folder {selected:?} was selected"
            );
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_selection_has_no_duplicates(tc: TestCase) {
        let (account, available) = tc.draw(account_with_folders());

        let selected = account.select_folders(&available);
        let mut deduped = selected.clone();
        deduped.sort();
        deduped.dedup();

        assert_eq!(
            selected.len(),
            deduped.len(),
            "{selected:?} has duplicates"
        );
    }

    #[hegel::test(test_cases = 200)]
    fn prop_all_folders_selects_the_complement(tc: TestCase) {
        let (mut account, available) = tc.draw(account_with_folders());
        account.all_folders = true;

        // The reference implementation, written the obvious slow way.
        let mut expected: Vec<String> = Vec::new();
        for folder in &available {
            let excluded = account.exclude.iter().any(|e| folder_eq(e, folder));
            let seen = expected.iter().any(|s| folder_eq(s, folder));
            if !excluded && !seen {
                expected.push(folder.clone());
            }
        }

        assert_eq!(account.select_folders(&available), expected);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_folder_eq_is_an_equivalence(tc: TestCase) {
        let a: String = tc.draw(folder_name());
        let b: String = tc.draw(folder_name());
        let c: String = tc.draw(folder_name());

        assert!(folder_eq(&a, &a), "reflexive");
        assert_eq!(folder_eq(&a, &b), folder_eq(&b, &a), "symmetric");

        if folder_eq(&a, &b) && folder_eq(&b, &c) {
            assert!(folder_eq(&a, &c), "transitive");
        }
    }

    #[hegel::test(test_cases = 100)]
    fn prop_duplicate_names_always_rejected(tc: TestCase) {
        let mut config = tc.draw(valid_config());
        tc.assume(!config.accounts.is_empty());

        // Clone an existing account under its own name.
        let clone = config.accounts[0].clone();
        let name = clone.name.clone();
        config.accounts.push(clone);

        assert!(
            matches!(config.validate(), Err(Error::DuplicateAccount(n)) if n == name)
        );
    }

    #[test]
    fn footers_default_to_none_at_all() {
        let config = Config::parse(MINIMAL).unwrap();
        let account = &config.accounts[0];

        assert!(account.footers.is_empty());
        assert!(account.footer_patterns().unwrap().is_empty());
    }

    #[test]
    fn footer_patterns_compile_in_order() {
        let config = Config::parse(
            r#"
            [[account]]
            name     = "work"
            host     = "imap.example.com"
            user     = "me@example.com"
            password = "x"
            footers  = ["^CONFIDENTIALITY NOTICE", "^Sent from my"]
        "#,
        )
        .unwrap();

        let patterns =
            config.account("work").unwrap().footer_patterns().unwrap();

        assert_eq!(patterns.len(), 2);
        assert!(patterns[0].is_match("CONFIDENTIALITY NOTICE: do not read"));
        assert!(patterns[1].is_match("Sent from my phone"));
    }

    #[test]
    fn parse_rejects_a_footer_pattern_that_does_not_compile() {
        // A bad pattern must fail here, and not at sync time, when the
        // user has already waited for a connection.
        let err = Config::parse(
            r#"
            [[account]]
            name     = "work"
            host     = "imap.example.com"
            user     = "me@example.com"
            password = "x"
            footers  = ["^Sent from ["]
        "#,
        );

        assert!(matches!(
            err,
            Err(Error::InvalidFooter { ref account, .. }) if account == "work"
        ));
    }

    #[hegel::test(test_cases = 200)]
    fn prop_configured_footers_cut_the_body(tc: TestCase) {
        let footer: String = tc.draw(gs::sampled_from(vec![
            "Sent from my iPhone".to_string(),
            "CONFIDENTIALITY NOTICE".to_string(),
            "Diese E-Mail ist vertraulich".to_string(),
        ]));
        let tail: String = tc.draw(gs::sampled_from(vec![
            "and may hold privileged material.".to_string(),
            "Delete this message if it is misdirected.".to_string(),
        ]));

        let mut account = account_named("work");
        account.footers = vec![format!("^{}", regex::escape(&footer))];

        let patterns = account.footer_patterns().unwrap();
        let body = format!("The invoice is attached.\n\n{footer}\n{tail}\n");
        let stripped = crate::body::strip_with_footers(&body, &patterns);

        assert_eq!(stripped.text, "The invoice is attached.");
    }
}
