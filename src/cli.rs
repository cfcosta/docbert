use std::path::PathBuf;

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::Shell;

#[derive(Debug, Parser)]
#[command(
    name = "docbert",
    about = "A powerful semantic search CLI for your documents"
)]
pub struct Cli {
    /// Override the XDG data directory
    #[arg(long, global = true)]
    pub data_dir: Option<PathBuf>,

    /// Override the ColBERT model ID or local model path
    #[arg(long, global = true)]
    pub model: Option<String>,

    /// Increase log verbosity (can be repeated: -v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Manage document collections
    Collection {
        #[command(subcommand)]
        action: CollectionAction,
    },
    /// Manage context descriptions for collections
    Context {
        #[command(subcommand)]
        action: ContextAction,
    },
    /// Search across collections
    Search(SearchArgs),
    /// Semantic-only search across all collections
    #[command(name = "ssearch")]
    Ssearch(SemanticSearchArgs),
    /// Retrieve a document by reference
    Get(GetArgs),
    /// Retrieve multiple documents matching a glob pattern
    MultiGet(MultiGetArgs),
    /// Rebuild indexes from source files (full rebuild)
    Rebuild(RebuildArgs),
    /// Sync collections with source files (incremental)
    Sync(SyncArgs),
    /// Show system status and statistics
    Status(StatusArgs),
    /// Start MCP server for AI agent integration
    Mcp,
    /// Manage the ColBERT model configuration
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },
    /// Generate shell completions
    #[command(hide = true)]
    Completions(CompletionsArgs),
}

// -- Collection subcommands --

#[derive(Debug, Subcommand)]
pub enum CollectionAction {
    /// Register a directory as a named collection and index its contents
    Add {
        /// Path to the directory
        path: PathBuf,
        /// Human-readable collection name
        #[arg(long)]
        name: String,
    },
    /// Remove a collection and all its indexed data
    Remove {
        /// Name of the collection to remove
        name: String,
    },
    /// List all registered collections
    List {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

// -- Context subcommands --

#[derive(Debug, Subcommand)]
pub enum ContextAction {
    /// Add or update a context string for a collection
    Add {
        /// Collection URI (e.g. bert://notes)
        uri: String,
        /// Free-text description
        description: String,
    },
    /// Remove the context string for a collection
    Remove {
        /// Collection URI (e.g. bert://notes)
        uri: String,
    },
    /// List all context strings
    List {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

// -- Model --

#[derive(Debug, Subcommand)]
pub enum ModelAction {
    /// Show the currently resolved model
    Show {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Persist a default model ID or local path in config.db
    Set {
        /// Model ID (HuggingFace) or local path
        model: String,
    },
    /// Clear the stored model setting (revert to default)
    Clear,
}

// -- Search --

#[derive(Debug, Parser)]
pub struct SearchArgs {
    /// The search query
    pub query: String,

    /// Number of results to return
    #[arg(short = 'n', long, default_value = "10")]
    pub count: usize,

    /// Search only within this collection
    #[arg(short = 'c', long)]
    pub collection: Option<String>,

    /// Output results as JSON
    #[arg(long)]
    pub json: bool,

    /// Return all results above the score threshold
    #[arg(long)]
    pub all: bool,

    /// Output only file paths (one per line)
    #[arg(long)]
    pub files: bool,

    /// Minimum score threshold
    #[arg(long, default_value = "0.0")]
    pub min_score: f32,

    /// Skip ColBERT reranking, return BM25 results directly
    #[arg(long)]
    pub bm25_only: bool,

    /// Disable fuzzy matching in the first stage
    #[arg(long)]
    pub no_fuzzy: bool,
}

// -- Semantic-only Search --

#[derive(Debug, Parser)]
pub struct SemanticSearchArgs {
    /// The search query
    pub query: String,

    /// Number of results to return
    #[arg(short = 'n', long, default_value = "10")]
    pub count: usize,

    /// Output results as JSON
    #[arg(long)]
    pub json: bool,

    /// Return all results above the score threshold
    #[arg(long)]
    pub all: bool,

    /// Output only file paths (one per line)
    #[arg(long)]
    pub files: bool,

    /// Minimum score threshold
    #[arg(long, default_value = "0.0")]
    pub min_score: f32,
}

// -- Get --

#[derive(Debug, Parser)]
pub struct GetArgs {
    /// Document reference: path, #doc_id, or collection:path
    pub reference: String,

    /// Output as JSON with metadata
    #[arg(long)]
    pub json: bool,

    /// Print only metadata
    #[arg(long)]
    pub meta: bool,

    /// Print full document content (default)
    #[arg(long)]
    pub full: bool,
}

// -- Multi-Get --

#[derive(Debug, Parser)]
pub struct MultiGetArgs {
    /// Glob pattern applied to relative paths
    pub pattern: String,

    /// Restrict to a specific collection
    #[arg(short = 'c', long)]
    pub collection: Option<String>,

    /// Output as JSON array
    #[arg(long)]
    pub json: bool,

    /// Output only file paths
    #[arg(long)]
    pub files: bool,

    /// Include full document content
    #[arg(long)]
    pub full: bool,
}

// -- Rebuild --

#[derive(Debug, Parser)]
pub struct RebuildArgs {
    /// Rebuild only this collection
    #[arg(short = 'c', long)]
    pub collection: Option<String>,

    /// Only recompute ColBERT embeddings
    #[arg(long)]
    pub embeddings_only: bool,

    /// Only rebuild the Tantivy index
    #[arg(long)]
    pub index_only: bool,
}

// -- Sync --

#[derive(Debug, Parser)]
pub struct SyncArgs {
    /// Sync only this collection
    #[arg(short = 'c', long)]
    pub collection: Option<String>,
}

// -- Status --

#[derive(Debug, Parser)]
pub struct StatusArgs {
    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

// -- Completions --

#[derive(Debug, Parser)]
pub struct CompletionsArgs {
    /// Shell to generate completions for
    #[arg(value_enum)]
    pub shell: Shell,
}

impl CompletionsArgs {
    /// Generate shell completions and print to stdout.
    pub fn generate(&self) {
        let mut cmd = Cli::command();
        clap_complete::generate(
            self.shell,
            &mut cmd,
            "docbert",
            &mut std::io::stdout(),
        );
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn parse_ssearch_defaults() {
        let cli = Cli::parse_from(["docbert", "ssearch", "hello"]);
        match cli.command {
            Command::Ssearch(args) => {
                assert_eq!(args.query, "hello");
                assert_eq!(args.count, 10);
                assert!(!args.json);
                assert!(!args.all);
                assert!(!args.files);
                assert_eq!(args.min_score, 0.0);
            }
            _ => panic!("expected ssearch command"),
        }
    }
}
