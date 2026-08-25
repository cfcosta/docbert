//! The mailbert configuration file.
//!
//! mailbert reads TOML from `$XDG_CONFIG_HOME/mailbert/config.toml`.
//! Accounts are an array of tables. See `docs/mailbert.md` §1.2.
//!
//! An account names two servers, because mail arrives at one and
//! leaves through the other. `[account.smtp]` is optional, and an
//! account without one syncs but cannot send.
//!
//! ```toml
//! [[account]]
//! name    = "work"
//! folders = ["INBOX", "Archive"]
//! exclude = ["Trash"]
//!
//! [account.imap]
//! host             = "imap.fastmail.com"
//! user             = "me@work.example"
//! password_command = "pass show mail/work"
//!
//! [account.smtp]
//! host = "smtp.fastmail.com"
//! ```

use std::{collections::HashSet, path::PathBuf};

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// The IMAPS port. mailbert does not speak cleartext IMAP.
pub const DEFAULT_PORT: u16 = 993;

/// The submission port of RFC 8314, which wants TLS from the first
/// byte. The older port 587 wants STARTTLS, and `tls = "start"` with
/// `port = 587` is how an account says so.
pub const DEFAULT_SMTP_PORT: u16 = 465;

/// Parallel IMAP connections for each account. Gmail permits 15, and
/// some servers permit 4, so this is configurable.
pub const DEFAULT_CONNECTIONS: usize = 8;

/// Results that `search` prints when `--count` is absent.
pub const DEFAULT_COUNT: usize = 20;

/// Half-life of the recency prior, in days. See `docs/mailbert.md` §8.3.
pub const DEFAULT_HALF_LIFE_DAYS: f64 = 180.0;

/// The name that RFC 3501 makes case-insensitive.
const INBOX: &str = "INBOX";

/// The folder that a sent message is filed into. (§11.3)
const SENT: &str = "Sent";

/// The whole configuration file.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub search: SearchConfig,

    #[serde(default)]
    pub view: ViewConfig,

    #[serde(default)]
    pub pgp: PgpConfig,

    /// Written as `[[account]]`, so the TOML key is singular.
    #[serde(
        default,
        rename = "account",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub accounts: Vec<Account>,
}

/// One account: the server its mail arrives from, and the server its
/// mail leaves through.
///
/// The two servers are separate tables because they are separate
/// machines with separate ports, and often separate logins. Only
/// `[account.imap]` is required: an account without `[account.smtp]`
/// syncs like any other, and `send` refuses it by name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Account {
    pub name: String,

    /// The server that the mail arrives from.
    pub imap: ImapConfig,

    /// The server that the mail leaves through. (§11)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smtp: Option<SmtpConfig>,

    /// The `From` header that `send` writes, with a display name.
    ///
    /// Without it `send` writes the SMTP user, which is an address and
    /// carries no name. `from = "Ada Lovelace <ada@example.test>"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,

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

    /// The folder that `send` files a sent message into, locally.
    ///
    /// This never reaches the server: §11.3 writes the copy into
    /// mailbert's own store, so that a search finds what you sent
    /// before the server's own copy comes back down.
    #[serde(default = "default_sent", skip_serializing_if = "is_sent")]
    pub sent: String,
}

/// The server that an account's mail arrives from. (§3)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImapConfig {
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

    /// Parallel connections for this account. (§3.1)
    #[serde(default = "default_connections")]
    pub connections: usize,
}

/// The server that an account's mail leaves through. (§11)
///
/// Every field but `host` has a default, and the defaults are the
/// credentials of `[account.imap]`, because one provider almost always
/// takes the same login on both machines. An account that submits under
/// a different user says so, and says nothing otherwise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmtpConfig {
    pub host: String,

    #[serde(default = "default_smtp_port")]
    pub port: u16,

    /// The submission user. Defaults to the IMAP user.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// A shell command whose first line of output is the password.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub password_command: Option<String>,

    /// A file whose first line is the password.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub password_file: Option<PathBuf>,

    /// The password itself. mailbert always warns about this one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub password: Option<String>,

    /// How TLS starts on the connection.
    #[serde(default)]
    pub tls: Tls,
}

/// When TLS starts on a submission connection.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Tls {
    /// TLS from the first byte, on port 465. This is what RFC 8314
    /// wants, and it is the default.
    #[default]
    Implicit,

    /// Cleartext, and then STARTTLS, on port 587.
    Start,

    /// No TLS at all.
    ///
    /// The password crosses the wire in the clear, so mailbert warns
    /// about this the way it warns about a password in the file. It is
    /// here for a submission server on the loopback, and for tests.
    None,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchConfig {
    #[serde(default = "default_count")]
    pub count: usize,

    #[serde(default = "default_half_life")]
    pub recency_half_life_days: f64,
}

/// Where `view` looks for the keys of an encrypted message. (§5.4)
///
/// Both fields are optional, because the defaults find what GnuPG
/// installs. mailbert never holds a secret key: gpg-agent keeps every
/// secret, and these paths only name the public certificates that say
/// which key of the agent a message was encrypted to.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PgpConfig {
    /// The GnuPG home. Defaults to `$GNUPGHOME`, then `~/.gnupg`.
    ///
    /// mailbert reads the agent socket and the public certificates
    /// from it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub home: Option<PathBuf>,

    /// A file of public certificates, which overrides `home`.
    ///
    /// A keybox (`pubring.kbx`), a legacy keyring (`pubring.gpg`), and
    /// an armored export all work.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub certs: Option<PathBuf>,
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

fn default_smtp_port() -> u16 {
    DEFAULT_SMTP_PORT
}

fn default_sent() -> String {
    SENT.to_string()
}

fn is_sent(name: &str) -> bool {
    name == SENT
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

// The three impls below hold exactly what serde fills in for a field
// the file leaves out, so `..Default::default()` in Rust and an absent
// key in TOML mean the same thing. A required field defaults to empty,
// which [`Config::validate`] then rejects by name.

impl Default for Account {
    fn default() -> Self {
        Self {
            name: String::new(),
            imap: ImapConfig::default(),
            smtp: None,
            from: None,
            folders: Vec::new(),
            exclude: Vec::new(),
            footers: Vec::new(),
            all_folders: false,
            sent: default_sent(),
        }
    }
}

impl Default for ImapConfig {
    fn default() -> Self {
        Self {
            host: String::new(),
            user: String::new(),
            port: default_port(),
            password_command: None,
            password_file: None,
            password: None,
            connections: default_connections(),
        }
    }
}

impl Default for SmtpConfig {
    fn default() -> Self {
        Self {
            host: String::new(),
            port: default_smtp_port(),
            user: None,
            password_command: None,
            password_file: None,
            password: None,
            tls: Tls::default(),
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

/// The credential that three optional sources give, in precedence.
///
/// The order is the one that §1.2 promises: `password_command`, then
/// `password_file`, then `password`.
fn pick(
    command: &Option<String>,
    file: &Option<PathBuf>,
    literal: &Option<String>,
) -> Option<Credential> {
    if let Some(command) = command {
        return Some(Credential::Command(command.clone()));
    }

    if let Some(path) = file {
        return Some(Credential::File(path.clone()));
    }

    literal.clone().map(Credential::Literal)
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
/// One folder that the server named in its `LIST` answer. (§1.2)
///
/// The name alone is not enough to choose folders. Gmail translates
/// the name of `[Gmail]/All Mail` into the language of the user, and
/// the attribute `\\All` of RFC 6154 stays the same.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Listed {
    /// The name that the server gave.
    pub name: String,

    /// The attributes of the folder, such as `\\All` or `\\Trash`.
    pub attributes: Vec<String>,
}

impl Listed {
    /// A folder that the server gave no attribute for.
    pub fn named(name: &str) -> Self {
        Self {
            name: name.to_string(),
            attributes: Vec::new(),
        }
    }

    /// Whether one entry of `folders` or `exclude` names this folder.
    ///
    /// An entry that starts with a backslash names an attribute, and
    /// the comparison ignores the case. Every other entry names a
    /// folder, and [`folder_eq`] compares it. A name never reads as an
    /// attribute, and an attribute never reads as a name.
    ///
    /// ```
    /// use mailbert_core::config::Listed;
    ///
    /// let all = Listed {
    ///     name: "[Gmail]/Todos os e-mails".to_string(),
    ///     attributes: vec!["\\All".to_string()],
    /// };
    ///
    /// assert!(all.answers("\\all"));
    /// assert!(!all.answers("[Gmail]/All Mail"));
    /// ```
    pub fn answers(&self, entry: &str) -> bool {
        match entry.starts_with('\\') {
            true => self
                .attributes
                .iter()
                .any(|held| held.eq_ignore_ascii_case(entry)),
            false => folder_eq(entry, &self.name),
        }
    }
}

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
    ///
    ///     [account.imap]
    ///     host = "imap.example.com"
    ///     user = "me@example.com"
    ///     password_command = "pass show mail/work"
    ///
    ///     [account.smtp]
    ///     host = "smtp.example.com"
    /// "#,
    /// )
    /// .unwrap();
    ///
    /// assert_eq!(config.accounts.len(), 1);
    /// assert_eq!(config.accounts[0].smtp_user().unwrap(), "me@example.com");
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
    /// [`Error::MissingCredential`], [`Error::InvalidFooter`], or
    /// [`Error::DuplicateAccount`].
    pub fn validate(&self) -> Result<()> {
        let mut seen: HashSet<&str> = HashSet::new();

        for account in &self.accounts {
            if account.name.is_empty() {
                return Err(Error::EmptyAccountName);
            }

            let smtp_host = account.smtp.as_ref().map(|smtp| &smtp.host);

            for (field, value) in [
                ("imap.host", Some(&account.imap.host)),
                ("imap.user", Some(&account.imap.user)),
                ("smtp.host", smtp_host),
            ] {
                if value.is_some_and(String::is_empty) {
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

            // A `send` that only finds out at the socket that it has no
            // password has already asked the user for a message.
            if account.smtp.is_some() {
                account.smtp_credential()?;
            }

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
    /// Resolve which credential source the IMAP server takes.
    ///
    /// Precedence is `password_command`, then `password_file`, then
    /// `password`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingCredential`] when all three are absent.
    pub fn credential(&self) -> Result<Credential> {
        pick(
            &self.imap.password_command,
            &self.imap.password_file,
            &self.imap.password,
        )
        .ok_or_else(|| Error::MissingCredential(self.name.clone()))
    }

    /// Resolve which credential source the submission server takes.
    ///
    /// An `[account.smtp]` that names no password takes the IMAP one,
    /// because one provider almost always takes the same login on both
    /// machines. It is only when the SMTP table names a password source
    /// of its own that the IMAP one is left alone.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoSmtp`] when the account has no `[account.smtp]`,
    /// and [`Error::MissingCredential`] when neither table names a
    /// password.
    pub fn smtp_credential(&self) -> Result<Credential> {
        let smtp = self.smtp()?;

        pick(&smtp.password_command, &smtp.password_file, &smtp.password)
            .map_or_else(|| self.credential(), Ok)
    }

    /// The submission user: the SMTP one, or the IMAP one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoSmtp`] when the account has no `[account.smtp]`.
    pub fn smtp_user(&self) -> Result<&str> {
        Ok(self
            .smtp()?
            .user
            .as_deref()
            .unwrap_or(self.imap.user.as_str()))
    }

    /// The submission server of this account.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoSmtp`] when the account has no `[account.smtp]`.
    pub fn smtp(&self) -> Result<&SmtpConfig> {
        self.smtp
            .as_ref()
            .ok_or_else(|| Error::NoSmtp(self.name.clone()))
    }

    /// The `From` header that `send` writes for this account. (§11.2)
    ///
    /// `from` of the configuration wins, because only it can carry a
    /// display name. Without it the submission user speaks, and an
    /// account with no submission server falls back to the IMAP user so
    /// that the header is never empty.
    pub fn sender(&self) -> String {
        if let Some(from) = &self.from {
            return from.clone();
        }

        self.smtp_user().unwrap_or(&self.imap.user).to_string()
    }

    /// Whether this account stores a password in the config file.
    ///
    /// The CLI warns when it does. A literal that precedence never
    /// reaches is not reported, because mailbert does not read it.
    pub fn has_plaintext_password(&self) -> bool {
        matches!(self.credential(), Ok(Credential::Literal(_)))
            || matches!(self.smtp_credential(), Ok(Credential::Literal(_)))
    }

    /// The folders to sync, given what the server listed.
    ///
    /// The result is a subset of `available` and holds the server's
    /// spelling, because that is the name mailbert must SELECT. It never
    /// contains an excluded folder and has no duplicates. `exclude`
    /// always wins, and `all_folders` ignores `folders`.
    ///
    /// An entry of `folders` or of `exclude` that starts with a
    /// backslash names an attribute of RFC 6154. See
    /// [`Listed::answers`].
    pub fn select_folders(&self, available: &[Listed]) -> Vec<String> {
        let mut selected: Vec<String> = Vec::new();

        for folder in available {
            let wanted = self.all_folders
                || self.folders.iter().any(|f| folder.answers(f));

            let skip = !wanted
                || self.exclude.iter().any(|e| folder.answers(e))
                || selected.iter().any(|s| folder_eq(s, &folder.name));

            if !skip {
                selected.push(folder.name.clone());
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
    pub fn missing_folders(&self, available: &[Listed]) -> Vec<String> {
        if self.all_folders {
            return Vec::new();
        }

        let mut missing: Vec<String> = Vec::new();

        for folder in &self.folders {
            let known = available.iter().any(|a| a.answers(folder));
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
    //! | `prop_an_attribute_in_exclude_removes_the_folders_that_have_it` | invariant | Gmail translates its folder names, so only the attribute keeps `[Gmail]/Trash` out. |
    //! | `prop_selection_has_no_duplicates` | invariant | A duplicate would fetch and hash the same folder twice. |
    //! | `prop_all_folders_selects_the_complement` | differential | `all_folders` must equal `available` minus `exclude`, exactly. |
    //! | `prop_folder_eq_is_an_equivalence` | algebraic | Reflexive, symmetric, transitive. Used to compare server names to config names. |
    //! | `prop_duplicate_names_always_rejected` | invariant | Two accounts with one name make `mailbert sync work` ambiguous. |
    //! | `prop_submission_falls_back_to_the_imap_login` | model-based | §11.1 promises that an `[account.smtp]` that says nothing about a password takes the IMAP one. A wrong fallback submits under the wrong login, or under none. |
    //! | `prop_a_sender_is_never_empty` | invariant | Every message carries a `From`, and an account that gives an empty one writes a message that no server takes. |

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

    /// An attribute of RFC 6154, drawn from a small pool for the same
    /// reason as [`folder_name`].
    fn attribute() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "\\All".to_string(),
            "\\Trash".to_string(),
            "\\Junk".to_string(),
            "\\Sent".to_string(),
        ])
    }

    /// An entry of `folders` or of `exclude`: a name, or an attribute.
    fn entry() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "INBOX".to_string(),
            "Archive".to_string(),
            "Sent".to_string(),
            "Drafts".to_string(),
            "Trash".to_string(),
            "Junk".to_string(),
            "[Gmail]/All Mail".to_string(),
            "\\All".to_string(),
            "\\Trash".to_string(),
            "\\Junk".to_string(),
            "\\Sent".to_string(),
        ])
    }

    fn entry_list() -> impl gs::Generator<Vec<String>> {
        gs::vecs(entry()).min_size(0).max_size(7)
    }

    #[hegel::composite]
    fn listed_folder(tc: TestCase) -> Listed {
        let name: String = tc.draw(folder_name());
        let attributes: Vec<String> =
            tc.draw(gs::vecs(attribute()).min_size(0).max_size(2));

        Listed { name, attributes }
    }

    fn listed_list() -> impl gs::Generator<Vec<Listed>> {
        gs::vecs(listed_folder()).min_size(0).max_size(7)
    }

    /// An account with every folder-selection field drawn independently.
    #[hegel::composite]
    fn account_with_folders(tc: TestCase) -> (Account, Vec<Listed>) {
        let available: Vec<Listed> = tc.draw(listed_list());
        let folders: Vec<String> = tc.draw(entry_list());
        let exclude: Vec<String> = tc.draw(entry_list());
        let all_folders: bool = tc.draw(gs::booleans());

        let account = Account {
            name: "acct".to_string(),
            imap: imap("imap.example.com", "me@example.com"),
            folders,
            exclude,
            all_folders,
            ..Account::default()
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
                imap: ImapConfig {
                    password_command: Some(format!("pass show mail/{i}")),
                    ..imap(
                        &format!("imap{i}.example.com"),
                        &format!("me{i}@example.com"),
                    )
                },
                // Every other account can send, so a round trip sees
                // both an account with `[account.smtp]` and one without.
                smtp: (i % 2 == 0)
                    .then(|| smtp(&format!("smtp{i}.example.com"))),
                folders: vec!["INBOX".to_string()],
                ..Account::default()
            })
            .collect();

        Config {
            search: SearchConfig::default(),
            view: ViewConfig::default(),
            pgp: PgpConfig::default(),
            accounts,
        }
    }

    fn account_named(name: &str) -> Account {
        Account {
            name: name.to_string(),
            imap: imap("imap.example.com", "me@example.com"),
            all_folders: true,
            ..Account::default()
        }
    }

    /// A submission server that validates, and names no login of its
    /// own, so that it falls back to the IMAP one.
    fn smtp(host: &str) -> SmtpConfig {
        SmtpConfig {
            host: host.to_string(),
            ..SmtpConfig::default()
        }
    }

    /// An IMAP server that validates: a host, a user, and a password
    /// command that succeeds without prompting.
    fn imap(host: &str, user: &str) -> ImapConfig {
        ImapConfig {
            host: host.to_string(),
            user: user.to_string(),
            password_command: Some("true".to_string()),
            ..ImapConfig::default()
        }
    }

    // -----------------------------------------------------------------
    // Unit tests.
    // -----------------------------------------------------------------

    const MINIMAL: &str = r#"
        [[account]]
        name = "work"

        [account.imap]
        host = "imap.fastmail.com"
        user = "me@work.example"
        password_command = "pass show mail/work"
    "#;

    #[test]
    fn parse_applies_the_documented_defaults() {
        let config = Config::parse(MINIMAL).unwrap();
        let account = &config.accounts[0];

        assert_eq!(account.imap.port, DEFAULT_PORT);
        assert_eq!(account.imap.connections, DEFAULT_CONNECTIONS);
        assert_eq!(account.sent, SENT);
        assert!(account.smtp.is_none());
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
            folders          = ["INBOX", "Archive", "Sent"]
            exclude          = ["Trash", "Junk"]

            [account.imap]
            host             = "imap.fastmail.com"
            port             = 993
            user             = "me@work.example"
            password_command = "pass show mail/work"
            connections      = 8

            [[account]]
            name          = "personal"
            all_folders   = true

            [account.imap]
            host          = "imap.gmail.com"
            user          = "me@gmail.com"
            password_file = "~/.secrets/gmail"

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
            fulders = ["INBOX"]

            [account.imap]
            host = "imap.example.com"
            user = "me@example.com"
            password = "hunter2"
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

            [account.imap]
            host = "a.example.com"
            user = "a@example.com"
            password = "x"

            [[account]]
            name = "work"

            [account.imap]
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

            [account.imap]
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

                [account.imap]
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

                [account.imap]
                host = ""
                user = "me@example.com"
                password = "x"
            "#
            ),
            Err(Error::EmptyField {
                field: "imap.host",
                ..
            })
        ));
    }

    #[test]
    fn credential_precedence_is_command_then_file_then_literal() {
        let mut account = account_named("work");
        account.imap.password_command = Some("pass show x".to_string());
        account.imap.password_file = Some(PathBuf::from("/tmp/p"));
        account.imap.password = Some("hunter2".to_string());

        assert_eq!(
            account.credential().unwrap(),
            Credential::Command("pass show x".to_string())
        );

        account.imap.password_command = None;
        assert_eq!(
            account.credential().unwrap(),
            Credential::File(PathBuf::from("/tmp/p"))
        );

        account.imap.password_file = None;
        assert_eq!(
            account.credential().unwrap(),
            Credential::Literal("hunter2".to_string())
        );

        account.imap.password = None;
        assert!(matches!(
            account.credential(),
            Err(Error::MissingCredential(_))
        ));
    }

    #[test]
    fn plaintext_password_is_flagged_only_when_it_is_used() {
        let mut account = account_named("work");
        account.imap.password_command = None;
        account.imap.password = Some("hunter2".to_string());
        assert!(account.has_plaintext_password());

        // A literal that precedence never reaches is not a risk worth
        // warning about.
        account.imap.password_command = Some("pass show x".to_string());
        assert!(!account.has_plaintext_password());
    }

    #[test]
    fn a_submission_server_takes_the_imap_login() {
        let mut account = account_named("work");
        account.smtp = Some(smtp("smtp.example.com"));

        assert_eq!(account.smtp_user().unwrap(), "me@example.com");
        assert_eq!(
            account.smtp_credential().unwrap(),
            Credential::Command("true".to_string())
        );
    }

    #[test]
    fn a_submission_server_keeps_the_login_it_names() {
        let mut account = account_named("work");
        account.smtp = Some(SmtpConfig {
            user: Some("submit@example.com".to_string()),
            password_command: Some("pass show smtp".to_string()),
            ..smtp("smtp.example.com")
        });

        assert_eq!(account.smtp_user().unwrap(), "submit@example.com");
        assert_eq!(
            account.smtp_credential().unwrap(),
            Credential::Command("pass show smtp".to_string())
        );
    }

    #[test]
    fn a_submission_user_alone_still_takes_the_imap_password() {
        // A provider that wants a different submission name, and the
        // same secret, is the common case of a shared mailbox.
        let mut account = account_named("work");
        account.smtp = Some(SmtpConfig {
            user: Some("submit@example.com".to_string()),
            ..smtp("smtp.example.com")
        });

        assert_eq!(
            account.smtp_credential().unwrap(),
            Credential::Command("true".to_string())
        );
    }

    #[test]
    fn an_account_with_no_submission_server_says_so() {
        let account = account_named("work");

        assert!(account.smtp.is_none());
        assert!(matches!(
            account.smtp(),
            Err(Error::NoSmtp(name)) if name == "work"
        ));
        assert!(matches!(account.smtp_user(), Err(Error::NoSmtp(_))));
        assert!(matches!(account.smtp_credential(), Err(Error::NoSmtp(_))));
    }

    #[test]
    fn a_submission_password_in_the_file_is_flagged_too() {
        let mut account = account_named("work");
        account.smtp = Some(SmtpConfig {
            password: Some("hunter2".to_string()),
            ..smtp("smtp.example.com")
        });

        // The IMAP side runs a command, so only the submission side is
        // a literal, and the account is still worth a warning.
        assert!(account.has_plaintext_password());
    }

    #[test]
    fn the_sender_is_the_from_when_the_account_names_one() {
        let mut account = account_named("work");
        account.from = Some("Ada Lovelace <ada@example.com>".to_string());

        assert_eq!(account.sender(), "Ada Lovelace <ada@example.com>");
    }

    #[test]
    fn the_sender_without_a_from_is_the_login_that_submits() {
        let mut account = account_named("work");
        assert_eq!(account.sender(), "me@example.com");

        account.smtp = Some(SmtpConfig {
            user: Some("submit@example.com".to_string()),
            ..smtp("smtp.example.com")
        });
        assert_eq!(account.sender(), "submit@example.com");
    }

    #[test]
    fn parse_reads_a_submission_server_and_its_defaults() {
        let config = Config::parse(
            r#"
            [[account]]
            name = "work"
            from = "Ada <ada@work.example>"
            sent = "[Gmail]/Sent Mail"

            [account.imap]
            host = "imap.work.example"
            user = "me@work.example"
            password_command = "pass show mail/work"

            [account.smtp]
            host = "smtp.work.example"
        "#,
        )
        .unwrap();

        let account = &config.accounts[0];
        let smtp = account.smtp().unwrap();

        assert_eq!(smtp.port, DEFAULT_SMTP_PORT);
        assert_eq!(smtp.tls, Tls::Implicit);
        assert_eq!(account.sender(), "Ada <ada@work.example>");
        assert_eq!(account.sent, "[Gmail]/Sent Mail");
    }

    #[test]
    fn parse_reads_the_two_ways_that_tls_starts() {
        let config = Config::parse(
            r#"
            [[account]]
            name = "work"

            [account.imap]
            host = "imap.work.example"
            user = "me@work.example"
            password = "x"

            [account.smtp]
            host = "smtp.work.example"
            port = 587
            tls  = "start"
        "#,
        )
        .unwrap();

        let smtp = config.accounts[0].smtp().unwrap();

        assert_eq!(smtp.port, 587);
        assert_eq!(smtp.tls, Tls::Start);
    }

    #[test]
    fn parse_rejects_a_submission_server_with_no_host() {
        let err = Config::parse(
            r#"
            [[account]]
            name = "work"

            [account.imap]
            host = "imap.work.example"
            user = "me@work.example"
            password = "x"

            [account.smtp]
            host = ""
        "#,
        );

        assert!(matches!(
            err,
            Err(Error::EmptyField {
                field: "smtp.host",
                ..
            })
        ));
    }

    #[test]
    fn inbox_compares_case_insensitively_and_others_do_not() {
        assert!(folder_eq("INBOX", "inbox"));
        assert!(folder_eq("InBoX", "INBOX"));
        assert!(!folder_eq("Archive", "archive"));
        assert!(folder_eq("Archive", "Archive"));
    }

    // -----------------------------------------------------------------
    // A folder that an attribute names. (§1.2)
    // -----------------------------------------------------------------

    /// Gmail translates the name of `[Gmail]/All Mail` into the
    /// language of the user, and the attribute stays the same. (§1.2)
    #[test]
    fn an_exclude_that_starts_with_a_backslash_reads_an_attribute() {
        let mut account = account_named("work");
        account.all_folders = true;
        account.exclude = vec!["\\Trash".to_string()];

        let available = vec![
            Listed::named("INBOX"),
            Listed {
                name: "[Gmail]/Lixeira".to_string(),
                attributes: vec![
                    "\\HasNoChildren".to_string(),
                    "\\Trash".to_string(),
                ],
            },
        ];

        assert_eq!(account.select_folders(&available), vec!["INBOX"]);
    }

    #[test]
    fn an_attribute_matches_whatever_the_case_of_the_server_is() {
        let mut account = account_named("work");
        account.all_folders = true;
        account.exclude = vec!["\\trash".to_string()];

        let available = vec![Listed {
            name: "Deleted Items".to_string(),
            attributes: vec!["\\Trash".to_string()],
        }];

        assert!(account.select_folders(&available).is_empty());
    }

    #[test]
    fn a_folders_entry_may_also_name_an_attribute() {
        let mut account = account_named("work");
        account.all_folders = false;
        account.folders = vec!["\\All".to_string()];

        let available = vec![
            Listed::named("INBOX"),
            Listed {
                name: "[Gmail]/Todos os e-mails".to_string(),
                attributes: vec!["\\All".to_string()],
            },
        ];

        assert_eq!(
            account.select_folders(&available),
            vec!["[Gmail]/Todos os e-mails"]
        );
        assert!(account.missing_folders(&available).is_empty());
    }

    #[test]
    fn missing_folders_reports_an_attribute_that_no_folder_has() {
        let mut account = account_named("work");
        account.all_folders = false;
        account.folders = vec!["\\Junk".to_string()];

        let available = vec![Listed::named("INBOX")];

        assert_eq!(account.missing_folders(&available), vec!["\\Junk"]);
    }

    /// A name never reads as an attribute, and an attribute never
    /// reads as a name. A folder that a server calls `\Trash` is not
    /// the folder that the attribute `\Trash` names.
    #[test]
    fn a_name_and_an_attribute_never_cross() {
        let mut account = account_named("work");
        account.all_folders = true;
        account.exclude = vec!["Trash".to_string()];

        let available = vec![Listed {
            name: "Deleted Items".to_string(),
            attributes: vec!["\\Trash".to_string()],
        }];

        // `exclude = ["Trash"]` is a name, and this folder has
        // another name.
        assert_eq!(account.select_folders(&available), vec!["Deleted Items"]);
    }

    #[test]
    fn select_folders_uses_the_server_spelling_of_inbox() {
        let mut account = account_named("work");
        account.all_folders = false;
        account.folders = vec!["inbox".to_string()];

        let available = vec![Listed::named("INBOX")];

        // The name the server gave is the one mailbert must SELECT.
        assert_eq!(account.select_folders(&available), vec!["INBOX"]);
        assert!(account.missing_folders(&available).is_empty());
    }

    #[test]
    fn missing_folders_reports_a_typo() {
        let mut account = account_named("work");
        account.all_folders = false;
        account.folders = vec!["Archve".to_string(), "INBOX".to_string()];

        let available = vec![Listed::named("INBOX"), Listed::named("Archive")];

        assert_eq!(account.missing_folders(&available), vec!["Archve"]);
    }

    #[test]
    fn all_folders_ignores_the_folders_list() {
        let mut account = account_named("work");
        account.all_folders = true;
        account.folders = vec!["Sent".to_string()];
        account.exclude = vec!["Trash".to_string()];

        let available = vec![
            Listed::named("INBOX"),
            Listed::named("Sent"),
            Listed::named("Trash"),
        ];

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
        account.imap.password_command = command.clone();
        account.imap.password_file = file.clone().map(PathBuf::from);
        account.imap.password = literal.clone();

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
    fn prop_submission_falls_back_to_the_imap_login(tc: TestCase) {
        let command: Option<String> =
            tc.draw(gs::optional(gs::just("smtp-cmd".to_string())));
        let file: Option<String> =
            tc.draw(gs::optional(gs::just("smtp-file".to_string())));
        let literal: Option<String> =
            tc.draw(gs::optional(gs::just("smtp-lit".to_string())));
        let user: Option<String> =
            tc.draw(gs::optional(gs::just("submit@example.com".to_string())));

        let mut account = account_named("work");
        account.smtp = Some(SmtpConfig {
            user: user.clone(),
            password_command: command.clone(),
            password_file: file.clone().map(PathBuf::from),
            password: literal.clone(),
            ..smtp("smtp.example.com")
        });

        // The model: the submission table decides only when it names a
        // password at all. Otherwise the whole IMAP precedence applies,
        // and `account_named` gives that a command.
        let expected = match (&command, &file, &literal) {
            (Some(c), _, _) => Credential::Command(c.clone()),
            (None, Some(f), _) => Credential::File(PathBuf::from(f)),
            (None, None, Some(l)) => Credential::Literal(l.clone()),
            (None, None, None) => Credential::Command("true".to_string()),
        };

        assert_eq!(account.smtp_credential().unwrap(), expected);
        assert_eq!(
            account.smtp_user().unwrap(),
            user.as_deref().unwrap_or("me@example.com")
        );
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_sender_is_never_empty(tc: TestCase) {
        let from: Option<String> = tc
            .draw(gs::optional(gs::just("Ada <ada@example.com>".to_string())));
        let user: Option<String> =
            tc.draw(gs::optional(gs::just("submit@example.com".to_string())));
        let sends: bool = tc.draw(gs::booleans());

        let mut account = account_named("work");
        account.from = from.clone();
        account.smtp = sends.then(|| SmtpConfig {
            user: user.clone(),
            ..smtp("smtp.example.com")
        });

        let sender = account.sender();

        assert!(!sender.is_empty());
        assert_eq!(
            sender,
            from.or(match sends {
                true => user,
                false => None,
            })
            .unwrap_or_else(|| "me@example.com".to_string())
        );
    }

    #[hegel::test(test_cases = 200)]
    fn prop_selection_is_a_subset_of_available(tc: TestCase) {
        let (account, available) = tc.draw(account_with_folders());

        for selected in account.select_folders(&available) {
            assert!(
                available.iter().any(|a| a.name == selected),
                "selected {selected:?} is not in {available:?}"
            );
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_exclude_always_wins(tc: TestCase) {
        let (account, available) = tc.draw(account_with_folders());

        let chosen = account.select_folders(&available);

        // A server lists one name one time, but the generator can give
        // two folders one name. Each chosen name must come from a
        // folder that no `exclude` entry answers.
        for selected in &chosen {
            let clean = available.iter().any(|folder| {
                folder_eq(&folder.name, selected)
                    && !account.exclude.iter().any(|e| folder.answers(e))
            });

            assert!(clean, "excluded folder {selected:?} was selected");
        }
    }

    /// The oracle reads the attributes itself, and never calls
    /// [`Listed::answers`], so a broken `answers` cannot hide here.
    #[hegel::test(test_cases = 200)]
    fn prop_an_attribute_in_exclude_removes_the_folders_that_have_it(
        tc: TestCase,
    ) {
        let (mut account, available) = tc.draw(account_with_folders());
        let unwanted: String = tc.draw(attribute());

        account.all_folders = true;
        account.exclude = vec![unwanted.clone()];

        let chosen = account.select_folders(&available);

        for folder in &available {
            // A server lists one name one time, but the generator can
            // give two. A name stays out only when every folder that
            // carries it has the attribute.
            let all_have = available
                .iter()
                .filter(|other| folder_eq(&other.name, &folder.name))
                .all(|other| {
                    other
                        .attributes
                        .iter()
                        .any(|held| held.eq_ignore_ascii_case(&unwanted))
                });

            assert!(
                !all_have || !chosen.contains(&folder.name),
                "{folder:?} has {unwanted} and reached {chosen:?}"
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
            let excluded = account.exclude.iter().any(|e| folder.answers(e));
            let seen = expected.iter().any(|s| folder_eq(s, &folder.name));
            if !excluded && !seen {
                expected.push(folder.name.clone());
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
            footers  = ["^CONFIDENTIALITY NOTICE", "^Sent from my"]

            [account.imap]
            host     = "imap.example.com"
            user     = "me@example.com"
            password = "x"
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
            footers  = ["^Sent from ["]

            [account.imap]
            host     = "imap.example.com"
            user     = "me@example.com"
            password = "x"
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
