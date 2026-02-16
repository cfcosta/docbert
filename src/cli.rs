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
    fn parse_search_defaults() {
        let cli = Cli::parse_from(["docbert", "search", "query"]);
        match cli.command {
            Command::Search(args) => {
                assert_eq!(args.query, "query");
                assert_eq!(args.count, 10);
                assert!(!args.json);
                assert!(!args.all);
                assert!(!args.files);
                assert_eq!(args.min_score, 0.0);
                assert!(!args.bm25_only);
                assert!(!args.no_fuzzy);
                assert!(args.collection.is_none());
            }
            _ => panic!("expected search command"),
        }
    }

    #[test]
    fn parse_search_all_flags() {
        let cli = Cli::parse_from([
            "docbert",
            "search",
            "q",
            "-n",
            "5",
            "--json",
            "--all",
            "--files",
            "--min-score",
            "0.5",
            "--bm25-only",
            "--no-fuzzy",
            "-c",
            "notes",
        ]);
        match cli.command {
            Command::Search(args) => {
                assert_eq!(args.query, "q");
                assert_eq!(args.count, 5);
                assert!(args.json);
                assert!(args.all);
                assert!(args.files);
                assert_eq!(args.min_score, 0.5);
                assert!(args.bm25_only);
                assert!(args.no_fuzzy);
                assert_eq!(args.collection.as_deref(), Some("notes"));
            }
            _ => panic!("expected search command"),
        }
    }

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

    #[test]
    fn parse_ssearch_all_flags() {
        let cli = Cli::parse_from([
            "docbert",
            "ssearch",
            "q",
            "-n",
            "20",
            "--json",
            "--all",
            "--files",
            "--min-score",
            "0.3",
        ]);
        match cli.command {
            Command::Ssearch(args) => {
                assert_eq!(args.query, "q");
                assert_eq!(args.count, 20);
                assert!(args.json);
                assert!(args.all);
                assert!(args.files);
                assert_eq!(args.min_score, 0.3);
            }
            _ => panic!("expected ssearch command"),
        }
    }

    #[test]
    fn parse_get_defaults() {
        let cli = Cli::parse_from(["docbert", "get", "notes:file.md"]);
        match cli.command {
            Command::Get(args) => {
                assert_eq!(args.reference, "notes:file.md");
                assert!(!args.json);
                assert!(!args.meta);
                assert!(!args.full);
            }
            _ => panic!("expected get command"),
        }
    }

    #[test]
    fn parse_get_with_flags() {
        let cli = Cli::parse_from([
            "docbert", "get", "#abc", "--json", "--meta", "--full",
        ]);
        match cli.command {
            Command::Get(args) => {
                assert_eq!(args.reference, "#abc");
                assert!(args.json);
                assert!(args.meta);
                assert!(args.full);
            }
            _ => panic!("expected get command"),
        }
    }

    #[test]
    fn parse_multi_get_defaults() {
        let cli = Cli::parse_from(["docbert", "multi-get", "*.md"]);
        match cli.command {
            Command::MultiGet(args) => {
                assert_eq!(args.pattern, "*.md");
                assert!(args.collection.is_none());
                assert!(!args.json);
                assert!(!args.files);
                assert!(!args.full);
            }
            _ => panic!("expected multi-get command"),
        }
    }

    #[test]
    fn parse_multi_get_with_flags() {
        let cli = Cli::parse_from([
            "docbert",
            "multi-get",
            "*.md",
            "-c",
            "notes",
            "--json",
            "--files",
            "--full",
        ]);
        match cli.command {
            Command::MultiGet(args) => {
                assert_eq!(args.pattern, "*.md");
                assert_eq!(args.collection.as_deref(), Some("notes"));
                assert!(args.json);
                assert!(args.files);
                assert!(args.full);
            }
            _ => panic!("expected multi-get command"),
        }
    }

    #[test]
    fn parse_collection_add() {
        let cli = Cli::parse_from([
            "docbert",
            "collection",
            "add",
            "/tmp/foo",
            "--name",
            "bar",
        ]);
        match cli.command {
            Command::Collection {
                action: CollectionAction::Add { path, name },
            } => {
                assert_eq!(path, PathBuf::from("/tmp/foo"));
                assert_eq!(name, "bar");
            }
            _ => panic!("expected collection add command"),
        }
    }

    #[test]
    fn parse_collection_remove() {
        let cli = Cli::parse_from(["docbert", "collection", "remove", "bar"]);
        match cli.command {
            Command::Collection {
                action: CollectionAction::Remove { name },
            } => {
                assert_eq!(name, "bar");
            }
            _ => panic!("expected collection remove command"),
        }
    }

    #[test]
    fn parse_collection_list() {
        let cli = Cli::parse_from(["docbert", "collection", "list", "--json"]);
        match cli.command {
            Command::Collection {
                action: CollectionAction::List { json },
            } => {
                assert!(json);
            }
            _ => panic!("expected collection list command"),
        }
    }

    #[test]
    fn parse_context_add() {
        let cli = Cli::parse_from([
            "docbert",
            "context",
            "add",
            "bert://notes",
            "description text",
        ]);
        match cli.command {
            Command::Context {
                action: ContextAction::Add { uri, description },
            } => {
                assert_eq!(uri, "bert://notes");
                assert_eq!(description, "description text");
            }
            _ => panic!("expected context add command"),
        }
    }

    #[test]
    fn parse_context_remove() {
        let cli =
            Cli::parse_from(["docbert", "context", "remove", "bert://notes"]);
        match cli.command {
            Command::Context {
                action: ContextAction::Remove { uri },
            } => {
                assert_eq!(uri, "bert://notes");
            }
            _ => panic!("expected context remove command"),
        }
    }

    #[test]
    fn parse_context_list() {
        let cli = Cli::parse_from(["docbert", "context", "list", "--json"]);
        match cli.command {
            Command::Context {
                action: ContextAction::List { json },
            } => {
                assert!(json);
            }
            _ => panic!("expected context list command"),
        }
    }

    #[test]
    fn parse_rebuild_defaults() {
        let cli = Cli::parse_from(["docbert", "rebuild"]);
        match cli.command {
            Command::Rebuild(args) => {
                assert!(args.collection.is_none());
                assert!(!args.embeddings_only);
                assert!(!args.index_only);
            }
            _ => panic!("expected rebuild command"),
        }
    }

    #[test]
    fn parse_rebuild_with_flags() {
        let cli = Cli::parse_from([
            "docbert",
            "rebuild",
            "-c",
            "notes",
            "--embeddings-only",
        ]);
        match cli.command {
            Command::Rebuild(args) => {
                assert_eq!(args.collection.as_deref(), Some("notes"));
                assert!(args.embeddings_only);
                assert!(!args.index_only);
            }
            _ => panic!("expected rebuild command"),
        }
    }

    #[test]
    fn parse_sync_defaults() {
        let cli = Cli::parse_from(["docbert", "sync"]);
        match cli.command {
            Command::Sync(args) => {
                assert!(args.collection.is_none());
            }
            _ => panic!("expected sync command"),
        }
    }

    #[test]
    fn parse_sync_with_collection() {
        let cli = Cli::parse_from(["docbert", "sync", "-c", "notes"]);
        match cli.command {
            Command::Sync(args) => {
                assert_eq!(args.collection.as_deref(), Some("notes"));
            }
            _ => panic!("expected sync command"),
        }
    }

    #[test]
    fn parse_status_defaults() {
        let cli = Cli::parse_from(["docbert", "status"]);
        match cli.command {
            Command::Status(args) => {
                assert!(!args.json);
            }
            _ => panic!("expected status command"),
        }
    }

    #[test]
    fn parse_status_json() {
        let cli = Cli::parse_from(["docbert", "status", "--json"]);
        match cli.command {
            Command::Status(args) => {
                assert!(args.json);
            }
            _ => panic!("expected status command"),
        }
    }

    #[test]
    fn parse_model_show() {
        let cli = Cli::parse_from(["docbert", "model", "show", "--json"]);
        match cli.command {
            Command::Model {
                action: ModelAction::Show { json },
            } => {
                assert!(json);
            }
            _ => panic!("expected model show command"),
        }
    }

    #[test]
    fn parse_model_set() {
        let cli = Cli::parse_from(["docbert", "model", "set", "custom/model"]);
        match cli.command {
            Command::Model {
                action: ModelAction::Set { model },
            } => {
                assert_eq!(model, "custom/model");
            }
            _ => panic!("expected model set command"),
        }
    }

    #[test]
    fn parse_model_clear() {
        let cli = Cli::parse_from(["docbert", "model", "clear"]);
        match cli.command {
            Command::Model {
                action: ModelAction::Clear,
            } => {}
            _ => panic!("expected model clear command"),
        }
    }

    #[test]
    fn parse_global_flags() {
        let cli = Cli::parse_from([
            "docbert",
            "-v",
            "--data-dir",
            "/tmp",
            "--model",
            "custom",
            "search",
            "q",
        ]);
        assert_eq!(cli.verbose, 1);
        assert_eq!(cli.data_dir, Some(PathBuf::from("/tmp")));
        assert_eq!(cli.model.as_deref(), Some("custom"));
    }

    #[test]
    fn parse_mcp_command() {
        let cli = Cli::parse_from(["docbert", "mcp"]);
        assert!(matches!(cli.command, Command::Mcp));
    }
}
