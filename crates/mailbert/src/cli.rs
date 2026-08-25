//! The command tree of §2.1.
//!
//! Each command of the design has one variant here. The variants hold
//! the words that the user typed, and nothing else. The commands
//! themselves live in their own modules.

use std::path::PathBuf;

use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use mailbert_core::rank::Sort;

/// mailbert — hybrid search for your mail.
#[derive(Debug, Clone, PartialEq, Eq, Parser)]
#[command(
    name = "mailbert",
    version,
    about = "Hybrid search for your mail",
    long_about = "mailbert mirrors an IMAP account, and searches it with \
                  a lexical leg and a semantic leg. It never writes to \
                  the server."
)]
pub struct Cli {
    /// Read the accounts from this file
    #[arg(long, global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Keep the store and the index in this directory
    #[arg(long, global = true, value_name = "DIR")]
    pub data_dir: Option<PathBuf>,

    /// Say more about the work (give it twice for more)
    #[arg(long, global = true, action = ArgAction::Count)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Command,
}

/// One command of §2.1.
#[derive(Debug, Clone, PartialEq, Eq, Subcommand)]
pub enum Command {
    /// Download the new mail of each account
    Sync(Sync),

    /// Write one message to a submission server
    Send(Send),

    /// Search the mail with both legs
    Search(Find),

    /// Search the mail with the lexical leg only, and load no model
    Ksearch(Find),

    /// Write the text of one message
    Get(One),

    /// Write one message with color
    View(Show),

    /// Write each message of one thread
    Thread(One),

    /// Put tags on messages, or take them off
    Tag(Tag),

    /// Keep a query under a name
    Saved {
        #[command(subcommand)]
        action: Saved,
    },

    /// Show the addresses that a name gives
    Contacts(Contacts),

    /// Write a maildir of symbolic links for another mail program
    Export(Export),

    /// Show the counts, the health of the index, and the last sync
    Status(Status),

    /// Speak MCP on the standard input and output
    Mcp,

    /// Write the completions of one shell
    #[command(hide = true)]
    Completions(Completions),
}

impl Command {
    /// The word that names this command on the command line.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Sync(_) => "sync",
            Self::Send(_) => "send",
            Self::Search(_) => "search",
            Self::Ksearch(_) => "ksearch",
            Self::Get(_) => "get",
            Self::View(_) => "view",
            Self::Thread(_) => "thread",
            Self::Tag(_) => "tag",
            Self::Saved { .. } => "saved",
            Self::Contacts(_) => "contacts",
            Self::Export(_) => "export",
            Self::Status(_) => "status",
            Self::Mcp => "mcp",
            Self::Completions(_) => "completions",
        }
    }

    /// True if the command writes JSON, which §10.4 keeps stable.
    pub fn wants_json(&self) -> bool {
        match self {
            Self::Sync(sync) => sync.json,
            Self::Send(send) => send.json,
            Self::Search(find) | Self::Ksearch(find) => find.json,
            Self::Get(one) | Self::Thread(one) => one.json,
            Self::Contacts(who) => who.json,
            Self::Status(status) => status.json,
            Self::Saved { action } => action.wants_json(),
            Self::View(_)
            | Self::Tag(_)
            | Self::Export(_)
            | Self::Mcp
            | Self::Completions(_) => false,
        }
    }
}

/// `mailbert sync [account] [--watch] [--full] [--dry-run]`
#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct Sync {
    /// Only this account
    pub account: Option<String>,

    /// Stay open, and wait for new mail with IMAP IDLE
    #[arg(long, conflicts_with = "dry_run")]
    pub watch: bool,

    /// Forget the sync state, and read each folder again
    #[arg(long)]
    pub full: bool,

    /// Show the plan, and download nothing
    #[arg(long)]
    pub dry_run: bool,

    /// Write JSON
    #[arg(long)]
    pub json: bool,
}

/// `mailbert search <query>` and `mailbert ksearch <query>`
#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct Find {
    /// The query, in the language of §7.1
    #[arg(required = true, num_args = 1.., value_name = "QUERY")]
    pub words: Vec<String>,

    /// How many threads to show
    #[arg(short = 'n', long, value_name = "COUNT")]
    pub count: Option<usize>,

    /// The order of the rows
    #[arg(long, value_enum, default_value_t = Order::Best)]
    pub sort: Order,

    /// Add the passage that matched
    #[arg(short = 'v', long)]
    pub snippet: bool,

    /// Write JSON
    #[arg(long)]
    pub json: bool,
}

impl Find {
    /// The query, as one line of text.
    pub fn text(&self) -> String {
        self.words.join(" ")
    }
}

/// The order of the rows of §8.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum Order {
    /// The fused score, with the recency prior of §8.3
    #[default]
    Best,

    /// The fused score alone
    Score,

    /// The date, newest first
    Date,
}

impl From<Order> for Sort {
    fn from(order: Order) -> Self {
        match order {
            Order::Best => Self::Best,
            Order::Score => Self::Score,
            Order::Date => Self::Date,
        }
    }
}

/// `mailbert get <id>` and `mailbert thread <id>`
#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct One {
    /// The identity of the message, or the start of it (§4.1)
    pub id: String,

    /// Write JSON
    #[arg(long)]
    pub json: bool,
}

/// `mailbert view <id>`
#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct Show {
    /// The identity of the message, or the start of it (§4.1)
    pub id: String,

    /// How wide the text is
    #[arg(long, value_name = "COLUMNS")]
    pub width: Option<usize>,

    /// The theme of the highlighting
    #[arg(long, value_name = "NAME")]
    pub theme: Option<String>,

    /// When to write color
    #[arg(long, value_enum, default_value_t = When::Auto)]
    pub color: When,
}

/// When a command writes color (§10.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum When {
    /// Color if the output is a terminal, and `NO_COLOR` is not set
    #[default]
    Auto,

    /// Always color
    Always,

    /// Never color
    Never,
}

/// `mailbert tag +todo -done a3f9`
#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct Tag {
    /// The changes, and then the identities
    #[arg(
        required = true,
        num_args = 1..,
        allow_hyphen_values = true,
        value_name = "WORD"
    )]
    pub words: Vec<String>,
}

/// `mailbert saved ...`
#[derive(Debug, Clone, PartialEq, Eq, Subcommand)]
pub enum Saved {
    /// Keep a query under a name
    Add {
        /// The name of the saved search
        name: String,

        /// The query, in the language of §7.1
        #[arg(required = true, num_args = 1.., value_name = "QUERY")]
        words: Vec<String>,
    },

    /// Show each saved search
    List {
        /// Write JSON
        #[arg(long)]
        json: bool,
    },

    /// Forget one saved search
    #[command(alias = "rm")]
    Remove {
        /// The name of the saved search
        name: String,
    },
}

impl Saved {
    /// True if the command writes JSON.
    pub fn wants_json(&self) -> bool {
        match self {
            Self::List { json } => *json,
            Self::Add { .. } | Self::Remove { .. } => false,
        }
    }
}

/// `mailbert send --to <address> --subject <text>` (§11)
#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct Send {
    /// A recipient. Give it once for each.
    #[arg(long, required_unless_present = "reply_to")]
    pub to: Vec<String>,

    /// A carbon copy
    #[arg(long)]
    pub cc: Vec<String>,

    /// A blind carbon copy, which no header names
    #[arg(long)]
    pub bcc: Vec<String>,

    /// The subject
    #[arg(long, required_unless_present = "reply_to")]
    pub subject: Option<String>,

    /// The body. Without it, the body comes from the standard input.
    #[arg(long)]
    pub body: Option<String>,

    /// Answer this message: its thread, its subject, its sender
    #[arg(long, value_name = "ID")]
    pub reply_to: Option<String>,

    /// Answer everyone that the message named, not only its sender
    #[arg(long, requires = "reply_to")]
    pub reply_all: bool,

    /// Send from this account
    #[arg(long)]
    pub account: Option<String>,

    /// Write the message on the standard output, and submit nothing
    #[arg(long)]
    pub dry_run: bool,

    /// Write JSON
    #[arg(long)]
    pub json: bool,
}

/// `mailbert contacts <name>`
#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct Contacts {
    /// The name, or the part of an address, to resolve
    pub name: String,

    /// Write JSON
    #[arg(long)]
    pub json: bool,
}

/// `mailbert export <query> <dir>`
#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct Export {
    /// The query, in the language of §7.1
    pub query: String,

    /// The maildir to write
    pub dir: PathBuf,
}

/// `mailbert status`
#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct Status {
    /// Write JSON
    #[arg(long)]
    pub json: bool,
}

/// `mailbert completions <shell>`
#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct Completions {
    /// The shell that reads the completions
    pub shell: Shell,
}

/// True if the output takes color.
///
/// `--color always` speaks for this one command, so it wins. `NO_COLOR`
/// speaks for the machine, so it beats the terminal (§10.3).
pub fn color(when: When, no_color: bool, terminal: bool) -> bool {
    match when {
        When::Always => true,
        When::Never => false,
        When::Auto => terminal && !no_color,
    }
}

#[cfg(test)]
mod tests {
    //! # Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_query_survives_the_command_line` | round-trip | The shell cuts a query into words. A query that comes back different searches for something that the user did not ask for. |
    //! | `prop_every_command_answers_to_its_name` | model-based | §2.1 names each command. A name that gives the wrong variant runs the wrong work. |
    //! | `prop_no_color_beats_the_terminal` | invariant | §10.3 obeys `NO_COLOR`. Escape codes in a pipe break the program that reads them. |

    use clap::CommandFactory;
    use hegel::{TestCase, generators as gs};

    use super::*;

    fn parse(words: &[&str]) -> Cli {
        Cli::try_parse_from(words).expect("a good command line")
    }

    #[test]
    fn the_command_tree_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn no_command_at_all_is_an_error() {
        assert!(Cli::try_parse_from(["mailbert"]).is_err());
    }

    #[test]
    fn each_command_of_the_design_is_there() {
        let lines: Vec<Vec<&str>> = vec![
            vec!["mailbert", "sync"],
            vec!["mailbert", "sync", "work"],
            vec!["mailbert", "sync", "--watch"],
            vec!["mailbert", "sync", "--full"],
            vec!["mailbert", "sync", "--dry-run"],
            vec!["mailbert", "search", "the deposit"],
            vec!["mailbert", "ksearch", "invoice 88213"],
            vec!["mailbert", "get", "a3f9"],
            vec!["mailbert", "view", "a3f9"],
            vec!["mailbert", "thread", "a3f9"],
            vec!["mailbert", "tag", "+todo", "a3f9"],
            vec!["mailbert", "saved", "add", "u", "account:work"],
            vec!["mailbert", "saved", "list"],
            vec!["mailbert", "contacts", "caina"],
            vec!["mailbert", "export", "tag:todo", "/mail/todo"],
            vec!["mailbert", "status"],
            vec!["mailbert", "mcp"],
        ];

        for line in lines {
            assert!(Cli::try_parse_from(&line).is_ok(), "{line:?} is in §2.1");
        }
    }

    #[test]
    fn search_joins_the_words_of_a_query() {
        let cli = parse(&["mailbert", "search", "from:bob", "and", "is:read"]);
        let Command::Search(find) = cli.command else {
            panic!("search gives a search");
        };

        assert_eq!(find.text(), "from:bob and is:read");
    }

    #[test]
    fn a_search_with_no_query_is_an_error() {
        assert!(Cli::try_parse_from(["mailbert", "search"]).is_err());
    }

    #[test]
    fn ksearch_is_a_command_of_its_own() {
        let cli = parse(&["mailbert", "ksearch", "invoice"]);

        assert_eq!(cli.command.name(), "ksearch");
    }

    #[test]
    fn a_short_v_on_a_search_asks_for_the_snippet() {
        let cli = parse(&["mailbert", "search", "rent", "-v"]);
        let Command::Search(find) = cli.command else {
            panic!("search gives a search");
        };

        assert!(find.snippet);
    }

    #[test]
    fn the_verbose_flag_of_the_tool_is_a_long_flag() {
        let cli = parse(&["mailbert", "--verbose", "--verbose", "status"]);

        assert_eq!(cli.verbose, 2);
    }

    #[test]
    fn the_default_order_is_the_best_one() {
        let cli = parse(&["mailbert", "search", "rent"]);
        let Command::Search(find) = cli.command else {
            panic!("search gives a search");
        };

        assert_eq!(find.sort, Order::Best);
        assert_eq!(Sort::from(find.sort), Sort::Best);
    }

    #[test]
    fn the_sort_flag_takes_the_names_of_the_design() {
        for (word, order) in [
            ("best", Order::Best),
            ("score", Order::Score),
            ("date", Order::Date),
        ] {
            let cli = parse(&["mailbert", "search", "rent", "--sort", word]);
            let Command::Search(find) = cli.command else {
                panic!("search gives a search");
            };

            assert_eq!(find.sort, order);
        }
    }

    #[test]
    fn a_sync_that_watches_and_shows_the_plan_is_an_error() {
        let line = ["mailbert", "sync", "--watch", "--dry-run"];

        assert!(Cli::try_parse_from(line).is_err());
    }

    #[test]
    fn a_sync_takes_the_name_of_one_account() {
        let cli = parse(&["mailbert", "sync", "work", "--full"]);
        let Command::Sync(sync) = cli.command else {
            panic!("sync gives a sync");
        };

        assert_eq!(sync.account.as_deref(), Some("work"));
        assert!(sync.full);
        assert!(!sync.watch);
    }

    #[test]
    fn a_tag_takes_a_word_that_starts_with_a_minus() {
        let cli = parse(&["mailbert", "tag", "-todo", "+done", "a3f9"]);
        let Command::Tag(tag) = cli.command else {
            panic!("tag gives a tag");
        };

        assert_eq!(tag.words, ["-todo", "+done", "a3f9"]);
    }

    #[test]
    fn a_view_takes_a_width_and_a_theme() {
        let cli = parse(&[
            "mailbert", "view", "a3f9", "--width", "100", "--theme", "dark",
        ]);
        let Command::View(show) = cli.command else {
            panic!("view gives a view");
        };

        assert_eq!(show.width, Some(100));
        assert_eq!(show.theme.as_deref(), Some("dark"));
        assert_eq!(show.color, When::Auto);
    }

    #[test]
    fn an_export_takes_a_query_and_a_directory() {
        let cli =
            parse(&["mailbert", "export", "tag:todo", "/home/me/mail/todo"]);
        let Command::Export(export) = cli.command else {
            panic!("export gives an export");
        };

        assert_eq!(export.query, "tag:todo");
        assert_eq!(export.dir, PathBuf::from("/home/me/mail/todo"));
    }

    #[test]
    fn a_saved_search_takes_a_name_and_a_query() {
        let cli = parse(&[
            "mailbert",
            "saved",
            "add",
            "unread-work",
            "account:work and not is:read",
        ]);
        let Command::Saved { action } = cli.command else {
            panic!("saved gives a saved");
        };
        let Saved::Add { name, words } = action else {
            panic!("add gives an add");
        };

        assert_eq!(name, "unread-work");
        assert_eq!(words.join(" "), "account:work and not is:read");
    }

    #[test]
    fn the_flags_of_the_tool_come_after_the_command_too() {
        let cli = parse(&["mailbert", "status", "--data-dir", "/tmp/m"]);

        assert_eq!(cli.data_dir, Some(PathBuf::from("/tmp/m")));
    }

    #[test]
    fn a_command_says_when_it_writes_json() {
        let cli = parse(&["mailbert", "search", "rent", "--json"]);
        assert!(cli.command.wants_json());

        let cli = parse(&["mailbert", "search", "rent"]);
        assert!(!cli.command.wants_json());

        let cli = parse(&["mailbert", "view", "a3f9"]);
        assert!(!cli.command.wants_json());
    }

    #[test]
    fn no_color_stops_the_color() {
        assert!(!color(When::Auto, true, true));
    }

    #[test]
    fn a_pipe_takes_no_color() {
        assert!(!color(When::Auto, false, false));
    }

    #[test]
    fn a_terminal_takes_color() {
        assert!(color(When::Auto, false, true));
    }

    #[test]
    fn the_flag_always_beats_the_pipe() {
        assert!(color(When::Always, true, false));
    }

    #[test]
    fn the_flag_never_beats_the_terminal() {
        assert!(!color(When::Never, false, true));
    }

    #[hegel::composite]
    fn a_query(tc: TestCase) -> Vec<String> {
        tc.draw(
            gs::vecs(
                gs::text()
                    .alphabet("abcdefgh:0123456789")
                    .min_size(1)
                    .max_size(8),
            )
            .min_size(1)
            .max_size(6),
        )
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_query_survives_the_command_line(tc: TestCase) {
        let words: Vec<String> = tc.draw(a_query());
        let mut line = vec!["mailbert".to_string(), "search".to_string()];
        line.extend(words.iter().cloned());

        let cli = Cli::try_parse_from(&line).expect("a good query");
        let Command::Search(find) = cli.command else {
            panic!("search gives a search");
        };

        assert_eq!(find.words, words);
        assert_eq!(find.text(), words.join(" "));
    }

    #[hegel::test(test_cases = 200)]
    fn prop_every_command_answers_to_its_name(tc: TestCase) {
        let line: Vec<&str> = tc.draw(gs::sampled_from(vec![
            vec!["mailbert", "sync"],
            vec!["mailbert", "search", "rent"],
            vec!["mailbert", "ksearch", "rent"],
            vec!["mailbert", "get", "a3f9"],
            vec!["mailbert", "view", "a3f9"],
            vec!["mailbert", "thread", "a3f9"],
            vec!["mailbert", "tag", "+todo", "a3f9"],
            vec!["mailbert", "saved", "list"],
            vec!["mailbert", "contacts", "bob"],
            vec!["mailbert", "export", "tag:todo", "/mail"],
            vec!["mailbert", "status"],
            vec!["mailbert", "mcp"],
        ]));

        let cli = Cli::try_parse_from(&line).expect("a line of §2.1");

        assert_eq!(cli.command.name(), line[1]);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_no_color_beats_the_terminal(tc: TestCase) {
        let terminal: bool = tc.draw(gs::booleans());

        assert!(!color(When::Auto, true, terminal));
        assert!(!color(When::Never, false, terminal));
    }
}
