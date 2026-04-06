use docbert_core::error;

use crate::cli::WebArgs;

pub(crate) fn run(_args: &WebArgs) -> error::Result<()> {
    Err(error::Error::Config(
        "docbert web is not implemented yet".to_string(),
    ))
}
