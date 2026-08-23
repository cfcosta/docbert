//! The log of the work. (§10.5)
//!
//! Every long command writes what it does to the standard error. The
//! standard output stays for the answer alone, so `--json` and the MCP
//! server of §2.2 read the same as before.
//!
//! `--verbose` moves the level of the mailbert crates only. A reader
//! who wants more gives `MAILBERT_LOG`, which takes the syntax of
//! `RUST_LOG` and wins over the flag.

use std::io::{self, IsTerminal};

use tracing::Subscriber;
use tracing_subscriber::{EnvFilter, fmt::MakeWriter};

use crate::cli::{When, color};

/// The name of the variable that gives the filter. (§10.5)
pub const LOG: &str = "MAILBERT_LOG";

/// The filter of one count of `--verbose`.
///
/// A quiet run keeps the other crates at `warn`, because tantivy and
/// the model say a lot that a reader of mail does not want.
pub fn levels(verbose: u8) -> &'static str {
    match verbose {
        0 => "warn,mailbert=info,mailbert_core=info,mailbert_imap=info",
        1 => "warn,mailbert=debug,mailbert_core=debug,mailbert_imap=debug",
        _ => "info,mailbert=trace,mailbert_core=trace,mailbert_imap=trace",
    }
}

/// The filter that this run takes.
///
/// `MAILBERT_LOG` wins over the flag. An empty variable is not a
/// filter, and the flag gives the level in that case.
pub fn directive(verbose: u8, env: Option<&str>) -> String {
    env.map(str::trim)
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| levels(verbose))
        .to_string()
}

/// The subscriber that writes to one place.
///
/// # Panics
///
/// The function panics if the directive is not a filter.
pub fn subscriber<W>(
    directive: &str,
    writer: W,
    ansi: bool,
) -> impl Subscriber + Send + Sync + 'static
where
    W: for<'a> MakeWriter<'a> + Send + Sync + 'static,
{
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(directive))
        .with_writer(writer)
        .with_ansi(ansi)
        .without_time()
        .finish()
}

/// Start the log of this run, on the standard error.
///
/// A second call does nothing, because one log is enough.
pub fn start(verbose: u8) {
    let env = std::env::var(LOG).ok();
    let filter = directive(verbose, env.as_deref());

    let _ = tracing::subscriber::set_global_default(subscriber(
        &filter,
        io::stderr as fn() -> io::Stderr,
        colorful(),
    ));
}

/// True when the standard error takes color. (§10.3)
fn colorful() -> bool {
    color(
        When::Auto,
        std::env::var_os("NO_COLOR").is_some_and(|text| !text.is_empty()),
        io::stderr().is_terminal(),
    )
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | Every filter of every flag count parses | `EnvFilter::try_new` | A directive that does not parse loses the whole log. |
    //! | The environment always wins over the flag | the text of the variable | §10.5 gives the reader the last word. |

    use std::{
        io::Write,
        sync::{Arc, Mutex},
    };

    use hegel::{TestCase, generators as gs};

    use super::*;

    /// A writer that keeps the log in memory.
    #[derive(Clone, Default)]
    struct Pen(Arc<Mutex<Vec<u8>>>);

    impl Pen {
        fn text(&self) -> String {
            let held = self.0.lock().expect("no writer panicked");

            String::from_utf8(held.clone()).expect("the log is text")
        }
    }

    impl Write for Pen {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().expect("no writer panicked").extend(bytes);

            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for Pen {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Write the events of one closure, and give back the log.
    fn logged(verbose: u8, work: impl FnOnce()) -> String {
        let pen = Pen::default();
        let filter = directive(verbose, None);

        tracing::subscriber::with_default(
            subscriber(&filter, pen.clone(), false),
            work,
        );

        pen.text()
    }

    #[test]
    fn a_quiet_run_keeps_the_tool_at_the_level_of_information() {
        assert!(directive(0, None).contains("mailbert=info"));
    }

    #[test]
    fn a_quiet_run_keeps_the_other_crates_quiet() {
        assert!(directive(0, None).starts_with("warn"));
    }

    #[test]
    fn one_flag_gives_the_level_of_debugging() {
        assert!(directive(1, None).contains("mailbert=debug"));
    }

    #[test]
    fn two_flags_give_the_level_of_tracing() {
        assert!(directive(2, None).contains("mailbert=trace"));
    }

    #[test]
    fn more_flags_never_go_past_tracing() {
        assert_eq!(directive(9, None), directive(2, None));
    }

    /// Each of the three crates must move together. A reader who gives
    /// `--verbose` wants the IMAP conversation of §3, and that lives in
    /// `mailbert_imap`.
    #[test]
    fn a_flag_moves_every_crate_of_the_tool() {
        let filter = directive(1, None);

        for crate_name in ["mailbert", "mailbert_core", "mailbert_imap"] {
            assert!(
                filter.contains(&format!("{crate_name}=debug")),
                "{crate_name} is not in `{filter}`"
            );
        }
    }

    #[test]
    fn the_environment_wins_over_the_flag() {
        assert_eq!(directive(2, Some("mailbert=warn")), "mailbert=warn");
    }

    #[test]
    fn an_empty_environment_leaves_the_flag_alone() {
        assert_eq!(directive(1, Some("   ")), directive(1, None));
    }

    #[test]
    fn every_filter_of_every_flag_parses() {
        for verbose in 0..4 {
            let filter = directive(verbose, None);

            assert!(
                EnvFilter::try_new(&filter).is_ok(),
                "`{filter}` is not a filter"
            );
        }
    }

    #[test]
    fn the_log_carries_an_event_of_the_tool() {
        let held = logged(0, || tracing::info!(folders = 3, "listed"));

        assert!(held.contains("listed"), "{held}");
        assert!(held.contains("folders=3"), "{held}");
    }

    #[test]
    fn a_quiet_run_drops_the_events_of_debugging() {
        let held = logged(0, || tracing::debug!("the fetch went out"));

        assert!(held.is_empty(), "{held}");
    }

    #[test]
    fn one_flag_keeps_the_events_of_debugging() {
        let held = logged(1, || tracing::debug!("the fetch went out"));

        assert!(held.contains("the fetch went out"), "{held}");
    }

    /// A line starts with its level, and never with a clock. The
    /// events that take long carry their own `ms`, which tells a
    /// reader more than the time of the day does.
    #[test]
    fn the_log_never_writes_the_time_of_the_day() {
        let held = logged(0, || tracing::info!("done"));
        let first = held.lines().next().expect("the log holds one line");

        assert_eq!(first.split_whitespace().next(), Some("INFO"), "{held}");
    }

    #[hegel::test(test_cases = 200)]
    fn prop_every_filter_of_every_flag_parses(tc: TestCase) {
        let verbose = tc.draw(gs::integers::<u8>());
        let filter = directive(verbose, None);

        assert!(
            EnvFilter::try_new(&filter).is_ok(),
            "`{filter}` is not a filter"
        );
    }

    #[hegel::test(test_cases = 200)]
    fn prop_the_environment_always_wins_over_the_flag(tc: TestCase) {
        let verbose = tc.draw(gs::integers::<u8>());
        let env = tc.draw(gs::sampled_from(vec![
            "trace",
            "mailbert=warn",
            "warn,mailbert_imap=trace",
            "off",
        ]));

        assert_eq!(directive(verbose, Some(env)), env);
    }
}
