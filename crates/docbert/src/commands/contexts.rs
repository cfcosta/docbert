use docbert_core::{ConfigDb, error};

use super::json_output::context_list_json_string;

pub(crate) fn context_add(
    config_db: &ConfigDb,
    uri: &str,
    description: &str,
) -> error::Result<()> {
    config_db.set_context(uri, description)?;
    println!("Added context for '{uri}'");
    Ok(())
}

pub(crate) fn context_remove(
    config_db: &ConfigDb,
    uri: &str,
) -> error::Result<()> {
    if !config_db.remove_context(uri)? {
        return Err(error::Error::NotFound {
            kind: "context",
            name: uri.to_string(),
        });
    }
    println!("Removed context for '{uri}'");
    Ok(())
}

pub(crate) fn context_list(
    config_db: &ConfigDb,
    json: bool,
) -> error::Result<()> {
    let contexts = config_db.list_contexts()?;

    if json {
        println!("{}", context_list_json_string(&contexts)?);
    } else if contexts.is_empty() {
        println!("No contexts defined.");
    } else {
        for (uri, desc) in &contexts {
            println!("{uri}\t{desc}");
        }
    }
    Ok(())
}
