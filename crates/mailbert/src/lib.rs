//! mailbert — hybrid search for your mail.
//!
//! The library reads the command line of §2.1, finds the paths of
//! §1.1, and gives the work to one command. It reads an IMAP server,
//! and it never writes to one.

pub mod cli;
pub mod error;
pub mod paths;
pub mod settings;
pub mod tags;

use std::{
    future::Future,
    io::{self, IsTerminal, Write},
};

use clap::CommandFactory;
use clap_complete::Shell;
use mailbert_core::config::Config;

use crate::{
    cli::{Cli, Command, When},
    error::{Error, Result},
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
    let _ = &tool;

    Err(Error::NotYet(cli.command.name()))
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
    fn a_command_that_is_not_ready_says_which_one() {
        let cli = parse(&[
            "mailbert",
            "--data-dir",
            "/tmp/m",
            "--config",
            "/tmp/m.toml",
            "status",
        ]);

        let result = run(cli);

        assert!(matches!(result, Err(Error::NotYet("status"))), "{result:?}");
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
