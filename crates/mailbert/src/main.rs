//! The mailbert command line tool.

use std::process::ExitCode;

use clap::Parser;
use mailbert::cli::Cli;

fn main() -> ExitCode {
    let cli = Cli::parse();

    // The log starts before the work, so a broken configuration file
    // still says what the tool looked for. (§10.5)
    mailbert::trace::start(cli.verbose);

    match mailbert::run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{:?}", miette::Report::new(error));

            ExitCode::FAILURE
        }
    }
}
