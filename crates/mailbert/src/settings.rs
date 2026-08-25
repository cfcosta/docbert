//! The configuration file, as the tool reads it (§1.2).
//!
//! `mailbert-core` parses the TOML and checks it. This module reads the
//! file from the disk, expands a `~`, runs `password_command`, and
//! gives a warning about a secret that other users can read.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

use mailbert_core::config::{Account, Config, Credential, Tls};

use crate::error::{Error, Result};

/// The mode bits that make a file private to its owner.
#[cfg(unix)]
const PRIVATE: u32 = 0o600;

/// Read the configuration file, and check it.
///
/// # Errors
///
/// The function fails if the file is not there, if the TOML is broken,
/// or if an account is not complete.
pub fn read(path: &Path) -> Result<Config> {
    if !path.is_file() {
        return Err(Error::NoConfig(path.to_path_buf()));
    }

    let config = Config::parse(&std::fs::read_to_string(path)?)?;
    config.validate()?;

    Ok(config)
}

/// The warnings that §1.2 asks for.
///
/// A password in the configuration file always gives a warning. A
/// password file gives a warning if its mode is not `0600`. Both
/// servers of an account are looked at, because a secret is no safer
/// for being the submission one.
pub fn warnings(config: &Config) -> Vec<String> {
    let home = home();
    let mut said = Vec::new();

    for account in &config.accounts {
        let smtp = account.smtp.is_some();

        for credential in [
            Some(account.credential()),
            smtp.then(|| account.smtp_credential()),
        ]
        .into_iter()
        .flatten()
        .flatten()
        {
            said.extend(about(&credential, account, home.as_deref()));
        }

        // A submission server without TLS reads the password off the
        // wire, so it is worth the same sentence as one in the file.
        if account
            .smtp
            .as_ref()
            .is_some_and(|smtp| smtp.tls == Tls::None)
        {
            said.push(format!(
                "account `{}` submits without TLS. Its password crosses \
                 the network in the clear.",
                account.name
            ));
        }
    }

    said
}

/// What is worth saying about where one password comes from.
fn about(
    credential: &Credential,
    account: &Account,
    home: Option<&Path>,
) -> Option<String> {
    match credential {
        Credential::Command(_) => None,

        Credential::File(path) => {
            let path = expand(path, home);

            open_wide(&path).then(|| {
                format!(
                    "the password file `{}` of account `{}` is not \
                     private. Set its mode to 0600.",
                    path.display(),
                    account.name
                )
            })
        }

        Credential::Literal(_) => Some(format!(
            "account `{}` keeps its password in the configuration \
             file. Use password_command, or password_file.",
            account.name
        )),
    }
}

/// The IMAP password of one account (§1.2).
///
/// # Errors
///
/// The function fails if the account has no credential, if the command
/// fails, if the file is not there, or if the secret is empty.
pub fn secret(account: &Account) -> Result<String> {
    resolve(account.credential()?, account)
}

/// The submission password of one account. (§11.1)
///
/// An `[account.smtp]` that names no password of its own takes the IMAP
/// one, so this usually runs the same `password_command` twice. That is
/// two `pass show` calls for one `send`, which is cheap enough.
///
/// # Errors
///
/// The function fails for the reasons [`secret`] does, and also when
/// the account has no `[account.smtp]` at all.
pub fn smtp_secret(account: &Account) -> Result<String> {
    resolve(account.smtp_credential()?, account)
}

/// Run one credential down to the password it names.
fn resolve(credential: Credential, account: &Account) -> Result<String> {
    let found = match credential {
        Credential::Command(line) => from_command(&line)?,
        Credential::File(path) => from_file(&expand(&path, home().as_deref()))?,
        Credential::Literal(text) => first_line(&text),
    };

    if found.is_empty() {
        return Err(Error::EmptySecret(account.name.clone()));
    }

    Ok(found)
}

/// Put the home directory in the place of a leading `~`.
///
/// The function only knows the home of the user who runs the tool, so
/// `~other/file` stays as it is.
pub fn expand(path: &Path, home: Option<&Path>) -> PathBuf {
    let (Some(home), Some(text)) = (home, path.to_str()) else {
        return path.to_path_buf();
    };

    if text == "~" {
        return home.to_path_buf();
    }

    match text.strip_prefix("~/") {
        Some(rest) => home.join(rest),
        None => path.to_path_buf(),
    }
}

/// The accounts that a command names.
///
/// A command with no name takes every account of the configuration.
///
/// # Errors
///
/// The function fails if the configuration has no account of that name.
pub fn accounts<'a>(
    config: &'a Config,
    name: Option<&str>,
) -> Result<Vec<&'a Account>> {
    let Some(name) = name else {
        return Ok(config.accounts.iter().collect());
    };

    let found = config
        .account(name)
        .ok_or_else(|| Error::UnknownAccount(name.to_string()))?;

    Ok(vec![found])
}

/// The home directory of the user who runs the tool.
pub fn home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|home| !home.as_os_str().is_empty())
}

/// The first line of the output of a command.
fn from_command(line: &str) -> Result<String> {
    let output = Command::new("sh").arg("-c").arg(line).output()?;

    if !output.status.success() {
        return Err(Error::CommandFailed {
            command: line.to_string(),
            status: output.status.code().unwrap_or(-1),
        });
    }

    Ok(first_line(&String::from_utf8_lossy(&output.stdout)))
}

/// The first line of a file.
fn from_file(path: &Path) -> Result<String> {
    Ok(first_line(&std::fs::read_to_string(path)?))
}

/// The first line of some text, with no carriage return at its end.
fn first_line(text: &str) -> String {
    text.lines()
        .next()
        .unwrap_or_default()
        .trim_end()
        .to_string()
}

/// True if a user other than the owner can read the file.
#[cfg(unix)]
fn open_wide(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;

    let Ok(data) = std::fs::metadata(path) else {
        return false;
    };

    data.permissions().mode() & 0o777 & !PRIVATE != 0
}

#[cfg(not(unix))]
fn open_wide(_path: &Path) -> bool {
    false
}

#[cfg(test)]
mod tests {
    //! # Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_path_with_no_tilde_never_changes` | invariant | The tool opens the path that the user wrote. A path that changes reads the wrong secret, or no secret. |
    //! | `prop_a_literal_password_always_gives_a_warning` | model-based | §1.2 says "always". A secret in a file that a backup copies must never be quiet. |
    //! | `prop_a_secret_holds_only_the_first_line` | invariant | A command that writes two lines gives a password with a newline in it, and §3 refuses that. |

    use hegel::{TestCase, generators as gs};

    use super::*;

    fn account(body: &str) -> Account {
        let text = format!(
            "[[account]]\nname = \"work\"\n[account.imap]\nhost = \"mail.example\"\n\
             user = \"me@example\"\n{body}\n"
        );
        let config = Config::parse(&text).expect("a good account");

        config.accounts.into_iter().next().expect("one account")
    }

    fn write(dir: &Path, name: &str, body: &str, mode: u32) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, body).expect("the file is writable");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            std::fs::set_permissions(
                &path,
                std::fs::Permissions::from_mode(mode),
            )
            .expect("the mode is settable");
        }

        let _ = mode;

        path
    }

    #[test]
    fn a_file_that_is_not_there_names_itself() {
        let result = read(Path::new("/nonexistent/mailbert/config.toml"));

        assert!(
            matches!(result, Err(Error::NoConfig(ref path))
                if path.ends_with("config.toml")),
            "{result:?}"
        );
    }

    #[test]
    fn a_good_file_gives_the_accounts() {
        let temp = tempfile::tempdir().expect("a temporary directory");
        let path = write(
            temp.path(),
            "config.toml",
            "[[account]]\nname = \"work\"\n[account.imap]\nhost = \"mail.example\"\n\
             user = \"me@example\"\npassword_command = \"true\"\n",
            0o600,
        );

        let config = read(&path).expect("a good file");

        assert_eq!(config.accounts.len(), 1);
        assert_eq!(config.accounts[0].name, "work");
    }

    #[test]
    fn a_broken_file_is_an_error() {
        let temp = tempfile::tempdir().expect("a temporary directory");
        let path = write(temp.path(), "config.toml", "[[account\n", 0o600);

        let result = read(&path);

        assert!(
            matches!(
                result,
                Err(Error::Core(mailbert_core::Error::ConfigParse(_)))
            ),
            "{result:?}"
        );
    }

    #[test]
    fn a_file_with_two_accounts_of_one_name_is_an_error() {
        let temp = tempfile::tempdir().expect("a temporary directory");
        let one = "[[account]]\nname = \"work\"\n[account.imap]\nhost = \"a\"\n\
                   user = \"b\"\npassword = \"c\"\n";
        let path =
            write(temp.path(), "config.toml", &format!("{one}{one}"), 0o600);

        let result = read(&path);

        assert!(
            matches!(
                result,
                Err(Error::Core(mailbert_core::Error::DuplicateAccount(_)))
            ),
            "{result:?}"
        );
    }

    #[test]
    fn a_password_in_the_file_gives_a_warning() {
        let text = "[[account]]\nname = \"work\"\n[account.imap]\nhost = \"a\"\n\
                    user = \"b\"\npassword = \"c\"\n";
        let config = Config::parse(text).expect("a good file");

        let said = warnings(&config);

        assert_eq!(said.len(), 1, "{said:?}");
        assert!(said[0].contains("work"), "{said:?}");
    }

    #[test]
    fn a_password_command_gives_no_warning() {
        let text = "[[account]]\nname = \"work\"\n[account.imap]\nhost = \"a\"\n\
                    user = \"b\"\npassword_command = \"true\"\n";
        let config = Config::parse(text).expect("a good file");

        assert!(warnings(&config).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn a_secret_file_that_others_can_read_gives_a_warning() {
        let temp = tempfile::tempdir().expect("a temporary directory");
        let path = write(temp.path(), "secret", "hunter2\n", 0o644);
        let text = format!(
            "[[account]]\nname = \"work\"\n[account.imap]\nhost = \"a\"\nuser = \"b\"\n\
             password_file = \"{}\"\n",
            path.display()
        );
        let config = Config::parse(&text).expect("a good file");

        let said = warnings(&config);

        assert_eq!(said.len(), 1, "{said:?}");
        assert!(said[0].contains("0600"), "{said:?}");
    }

    #[cfg(unix)]
    #[test]
    fn a_secret_file_of_the_right_mode_is_quiet() {
        let temp = tempfile::tempdir().expect("a temporary directory");
        let path = write(temp.path(), "secret", "hunter2\n", 0o600);
        let text = format!(
            "[[account]]\nname = \"work\"\n[account.imap]\nhost = \"a\"\nuser = \"b\"\n\
             password_file = \"{}\"\n",
            path.display()
        );
        let config = Config::parse(&text).expect("a good file");

        assert!(warnings(&config).is_empty());
    }

    #[test]
    fn a_command_gives_the_secret() {
        let account = account("password_command = \"printf 'hunter2\\n'\"");

        assert_eq!(secret(&account).expect("the command runs"), "hunter2");
    }

    #[test]
    fn a_command_that_fails_is_an_error() {
        let account = account("password_command = \"exit 3\"");
        let result = secret(&account);

        assert!(
            matches!(result, Err(Error::CommandFailed { status: 3, .. })),
            "{result:?}"
        );
    }

    #[test]
    fn a_command_that_says_nothing_is_an_error() {
        let account = account("password_command = \"true\"");
        let result = secret(&account);

        assert!(matches!(result, Err(Error::EmptySecret(_))), "{result:?}");
    }

    #[test]
    fn a_file_gives_the_secret() {
        let temp = tempfile::tempdir().expect("a temporary directory");
        let path = write(temp.path(), "secret", "hunter2\nmore\n", 0o600);
        let account =
            account(&format!("password_file = \"{}\"", path.display()));

        assert_eq!(secret(&account).expect("the file opens"), "hunter2");
    }

    #[test]
    fn a_secret_file_that_is_not_there_is_an_error() {
        let account = account("password_file = \"/nonexistent/secret\"");
        let result = secret(&account);

        assert!(matches!(result, Err(Error::Io(_))), "{result:?}");
    }

    #[test]
    fn a_password_in_the_file_is_the_last_source() {
        let account = account("password = \"hunter2\"");

        assert_eq!(secret(&account).expect("the file holds it"), "hunter2");
    }

    #[test]
    fn a_command_beats_a_file_and_a_password() {
        let account = account(
            "password_command = \"printf 'from-command\\n'\"\n\
             password_file = \"/nonexistent/secret\"\n\
             password = \"from-file\"",
        );

        assert_eq!(secret(&account).expect("the command runs"), "from-command");
        assert_eq!(
            account.credential().expect("a credential"),
            Credential::Command("printf 'from-command\n'".to_string())
        );
    }

    #[test]
    fn no_name_takes_every_account() {
        let text = "[[account]]\nname = \"work\"\n[account.imap]\nhost = \"a\"\n\
                    user = \"b\"\npassword = \"c\"\n\
                    [[account]]\nname = \"home\"\n[account.imap]\nhost = \"a\"\n\
                    user = \"b\"\npassword = \"c\"\n";
        let config = Config::parse(text).expect("a good file");

        let found = accounts(&config, None).expect("every account");

        assert_eq!(found.len(), 2);
        assert_eq!(found[0].name, "work");
    }

    #[test]
    fn a_name_takes_one_account() {
        let text = "[[account]]\nname = \"work\"\n[account.imap]\nhost = \"a\"\n\
                    user = \"b\"\npassword = \"c\"\n\
                    [[account]]\nname = \"home\"\n[account.imap]\nhost = \"a\"\n\
                    user = \"b\"\npassword = \"c\"\n";
        let config = Config::parse(text).expect("a good file");

        let found = accounts(&config, Some("home")).expect("one account");

        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "home");
    }

    #[test]
    fn a_name_that_is_not_there_is_an_error() {
        let text = "[[account]]\nname = \"work\"\n[account.imap]\nhost = \"a\"\n\
                    user = \"b\"\npassword = \"c\"\n";
        let config = Config::parse(text).expect("a good file");

        let result = accounts(&config, Some("other"));

        assert!(
            matches!(result, Err(Error::UnknownAccount(ref name))
                if name == "other"),
            "{result:?}"
        );
    }

    #[test]
    fn a_tilde_becomes_the_home() {
        let found =
            expand(Path::new("~/.secrets/gmail"), Some(Path::new("/home/me")));

        assert_eq!(found, PathBuf::from("/home/me/.secrets/gmail"));
    }

    #[test]
    fn a_tilde_alone_is_the_home() {
        let found = expand(Path::new("~"), Some(Path::new("/home/me")));

        assert_eq!(found, PathBuf::from("/home/me"));
    }

    #[test]
    fn the_home_of_another_user_stays_as_it_is() {
        let found = expand(Path::new("~bob/x"), Some(Path::new("/home/me")));

        assert_eq!(found, PathBuf::from("~bob/x"));
    }

    #[test]
    fn a_machine_with_no_home_keeps_the_tilde() {
        let found = expand(Path::new("~/x"), None);

        assert_eq!(found, PathBuf::from("~/x"));
    }

    #[test]
    fn a_secret_file_under_the_home_opens() {
        let temp = tempfile::tempdir().expect("a temporary directory");
        write(temp.path(), "secret", "hunter2\n", 0o600);
        let account = account("password_file = \"~/secret\"");
        let path = expand(
            account.imap.password_file.as_deref().expect("a file"),
            Some(temp.path()),
        );

        assert_eq!(path, temp.path().join("secret"));
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_path_with_no_tilde_never_changes(tc: TestCase) {
        let text: String =
            tc.draw(gs::text().alphabet("ab/.-_").min_size(1).max_size(12));

        if text.starts_with('~') {
            return;
        }

        let path = PathBuf::from(&text);

        assert_eq!(expand(&path, Some(Path::new("/home/me"))), path);
    }

    #[hegel::test(test_cases = 100)]
    fn prop_a_literal_password_always_gives_a_warning(tc: TestCase) {
        let names: Vec<String> = tc.draw(
            gs::vecs(gs::text().alphabet("abcdef").min_size(1).max_size(4))
                .min_size(1)
                .max_size(3),
        );
        let mut unique: Vec<String> = names.clone();
        unique.sort();
        unique.dedup();

        let mut text = String::new();
        for name in &unique {
            text.push_str(&format!(
                "[[account]]\nname = \"{name}\"\n[account.imap]\nhost = \"a\"\n\
                 user = \"b\"\npassword = \"c\"\n"
            ));
        }

        let config = Config::parse(&text).expect("a good file");
        let said = warnings(&config);

        assert_eq!(said.len(), unique.len());

        for name in &unique {
            assert!(
                said.iter().any(|line| line.contains(name.as_str())),
                "no warning names `{name}`"
            );
        }
    }

    #[hegel::test(test_cases = 100)]
    fn prop_a_secret_holds_only_the_first_line(tc: TestCase) {
        let head: String =
            tc.draw(gs::text().alphabet("abcdef").min_size(1).max_size(8));
        let tail: String =
            tc.draw(gs::text().alphabet("abcdef").min_size(0).max_size(8));
        let temp = tempfile::tempdir().expect("a temporary directory");
        let path =
            write(temp.path(), "secret", &format!("{head}\n{tail}\n"), 0o600);
        let account =
            account(&format!("password_file = \"{}\"", path.display()));

        let found = secret(&account).expect("the file opens");

        assert_eq!(found, head);
        assert!(!found.contains('\n'));
    }
}
