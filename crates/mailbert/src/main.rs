//! The mailbert command line tool.

use std::process::ExitCode;

use clap::Parser;
use mailbert::cli::Cli;

fn main() -> ExitCode {
    let cli = Cli::parse();

    match mailbert::run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{:?}", miette::Report::new(error));

            ExitCode::FAILURE
        }
    }
}
