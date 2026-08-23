//! mailbert — hybrid search for your mail.
//!
//! The library reads the command line of §2.1, finds the paths of
//! §1.1, and gives the work to one command. It reads an IMAP server,
//! and it never writes to one.

pub mod cli;
pub mod contacts;
pub mod error;
pub mod export;
pub mod mcp;
pub mod pass;
pub mod paths;
pub mod saved;
pub mod search;
pub mod semantic;
pub mod settings;
pub mod show;
pub mod sink;
pub mod status;
pub mod sync;
pub mod tags;
pub mod thread;
pub mod trace;

use std::{
    future::Future,
    io::{self, IsTerminal, Write},
    sync::Arc,
};

use clap::CommandFactory;
use clap_complete::Shell;
use mailbert_core::{Store, config::Config, date::Clock, index::MailIndex};

use crate::{
    cli::{Cli, Command, When},
    error::Result,
    paths::Paths,
};

/// What each command needs before it starts.
#[derive(Debug)]
pub struct Tool {
    /// The files of §1.1, and the configuration file of §1.2.
    pub paths: Paths,

    /// How much the tool says about its work.
    pub verbose: u8,
}

impl Tool {
    /// Find the paths that the flags and the environment give.
    ///
    /// # Errors
    ///
    /// The function fails if no source gives a path.
    pub fn open(cli: &Cli) -> Result<Self> {
        let paths =
            Paths::find(cli.data_dir.as_deref(), cli.config.as_deref())?;

        Ok(Self {
            paths,
            verbose: cli.verbose,
        })
    }

    /// The configuration, and its warnings on the standard error.
    ///
    /// # Errors
    ///
    /// The function fails if the file is not there, or if it is broken.
    pub fn config(&self) -> Result<Config> {
        let config = settings::read(&self.paths.config)?;

        for line in settings::warnings(&config) {
            eprintln!("mailbert: warning: {line}");
        }

        Ok(config)
    }

    /// The store of §4.2, made if it is not there.
    ///
    /// The store goes behind an [`Arc`], because a sync gives it to one
    /// task for each folder.
    ///
    /// # Errors
    ///
    /// The function fails if the directory or the database is not
    /// writable.
    pub fn store(&self) -> Result<Arc<Store>> {
        self.paths.make()?;

        Ok(Arc::new(Store::open(&self.paths.data)?))
    }

    /// The lexical index of §6.1, made if it is not there.
    ///
    /// # Errors
    ///
    /// The function fails if the directory is not writable.
    pub fn index(&self) -> Result<MailIndex> {
        self.paths.make()?;

        Ok(MailIndex::open(&self.paths.tantivy())?)
    }

    /// The model, the embeddings, and the PLAID index of §6.2.
    ///
    /// The model does not load here. `DOCBERT_MODEL` names it, and the
    /// default of docbert names it when that variable is empty.
    ///
    /// # Errors
    ///
    /// The function fails if the embedding database cannot open.
    pub fn brain(&self) -> Result<semantic::Brain> {
        self.paths.make()?;

        semantic::Brain::open(
            &self.paths.embeddings(),
            &self.paths.plaid(),
            None,
        )
    }
}

/// Do the work of one command.
///
/// # Errors
///
/// The function fails if the command fails.
pub fn run(cli: Cli) -> Result<()> {
    if let Command::Completions(what) = &cli.command {
        return completions(what.shell, &mut io::stdout());
    }

    let tool = Tool::open(&cli)?;

    match &cli.command {
        Command::Sync(args) => sync::command(&tool, args),
        Command::Search(args) => {
            search::command(&tool, args, search::Legs::Both)
        }
        Command::Ksearch(args) => {
            search::command(&tool, args, search::Legs::Words)
        }
        Command::Get(args) => show::get(&tool, args),
        Command::View(args) => show::view(&tool, args),
        Command::Export(args) => export::command(&tool, args),
        Command::Thread(args) => thread::command(&tool, args),
        Command::Tag(args) => tags::command(&tool, args),
        Command::Saved { action } => saved::command(&tool, action),
        Command::Contacts(args) => contacts::command(&tool, args),
        Command::Status(args) => status::command(&tool, args),
        Command::Mcp => mcp::command(&tool),

        // `completions` left above, before the tool opened, because it
        // must run where no configuration and no store are there yet.
        Command::Completions(_) => unreachable!("handled above"),
    }
}

/// The clock of the machine, with the offset of its time zone.
///
/// §7.1 reads `date:today` against the day of the reader, and §10.1
/// writes the day of each message. Both need the local offset, which
/// only the machine knows. A machine that hides its zone gives UTC.
pub fn clock() -> Clock {
    let now = jiff::Timestamp::now();
    let offset = jiff::tz::TimeZone::try_system()
        .map(|zone| zone.to_offset(now).seconds())
        .unwrap_or(0);

    Clock::new(now.as_second(), offset)
}

/// Write the completions of one shell.
pub fn completions(shell: Shell, out: &mut dyn Write) -> Result<()> {
    clap_complete::generate(shell, &mut Cli::command(), "mailbert", out);

    Ok(())
}

/// True if the output takes color (§10.3).
pub fn wants_color(when: When) -> bool {
    cli::color(
        when,
        std::env::var_os("NO_COLOR").is_some_and(|text| !text.is_empty()),
        io::stdout().is_terminal(),
    )
}

/// Run a future that speaks to a server.
///
/// Only the commands that reach the network make a runtime, so a
/// `ksearch` in a shell loop pays nothing for it.
pub fn block_on<F: Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("the machine gives a runtime")
        .block_on(future)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use clap::Parser;

    use super::*;

    fn parse(words: &[&str]) -> Cli {
        Cli::try_parse_from(words).expect("a good command line")
    }

    #[test]
    fn the_flags_give_the_paths_of_the_tool() {
        let cli = parse(&[
            "mailbert",
            "--data-dir",
            "/tmp/m",
            "--config",
            "/tmp/m.toml",
            "--verbose",
            "status",
        ]);

        let tool = Tool::open(&cli).expect("the flags give both paths");

        assert_eq!(tool.paths.data, PathBuf::from("/tmp/m"));
        assert_eq!(tool.paths.config, PathBuf::from("/tmp/m.toml"));
        assert_eq!(tool.verbose, 1);
    }

    #[test]
    fn the_completions_of_a_shell_name_the_tool() {
        let mut out = Vec::new();
        completions(Shell::Bash, &mut out).expect("bash takes completions");

        let text = String::from_utf8(out).expect("the shell reads text");

        assert!(text.contains("mailbert"), "{text}");
        assert!(text.contains("ksearch"), "{text}");
    }

    #[test]
    fn the_configuration_of_the_tool_comes_from_its_path() {
        let temp = tempfile::tempdir().expect("a temporary directory");
        let path = temp.path().join("config.toml");
        std::fs::write(
            &path,
            "[[account]]\nname = \"work\"\nhost = \"a\"\nuser = \"b\"\n\
             password_command = \"true\"\n",
        )
        .expect("the file is writable");

        let cli = parse(&[
            "mailbert",
            "--data-dir",
            temp.path().to_str().expect("a name of text"),
            "--config",
            path.to_str().expect("a name of text"),
            "status",
        ]);
        let tool = Tool::open(&cli).expect("the flags give both paths");

        let config = tool.config().expect("a good file");

        assert_eq!(config.accounts.len(), 1);
    }
}
