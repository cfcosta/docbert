//! `RustItem` — the unit of work in rustbert's data model.
//!
//! A single Rust source item (fn, struct, enum, trait, impl, mod,
//! const, static, type alias, macro) along with the metadata needed
//! to format a search result and resolve back to source.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RustItemKind {
    Fn,
    Struct,
    Enum,
    Union,
    Trait,
    Impl,
    Mod,
    Const,
    Static,
    TypeAlias,
    Macro,
}

impl RustItemKind {
    /// Stable, lowercase string used in metadata payloads and the
    /// `--kind` CLI filter.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fn => "fn",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Union => "union",
            Self::Trait => "trait",
            Self::Impl => "impl",
            Self::Mod => "mod",
            Self::Const => "const",
            Self::Static => "static",
            Self::TypeAlias => "type",
            Self::Macro => "macro",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "fn" => Self::Fn,
            "struct" => Self::Struct,
            "enum" => Self::Enum,
            "union" => Self::Union,
            "trait" => Self::Trait,
            "impl" => Self::Impl,
            "mod" => Self::Mod,
            "const" => Self::Const,
            "static" => Self::Static,
            "type" => Self::TypeAlias,
            "macro" => Self::Macro,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    Public,
    Crate,
    Restricted,
    Private,
}

impl Visibility {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Public => "pub",
            Self::Crate => "pub(crate)",
            Self::Restricted => "pub(in path)",
            Self::Private => "private",
        }
    }
}

/// One Rust source item, ready to be lowered into a search document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RustItem {
    pub kind: RustItemKind,
    pub crate_name: String,
    pub crate_version: semver::Version,
    /// Module path leading to this item, *not* including the crate
    /// root or the item itself: `["serde", "ser"]` for
    /// `serde::ser::Serializer::serialize_struct`.
    pub module_path: Vec<String>,
    /// Item name. `None` for `impl` blocks (which have no canonical name).
    pub name: Option<String>,
    /// Fully-qualified path used as the search title, e.g.
    /// `serde::ser::Serializer::serialize_struct`.
    pub qualified_path: String,
    /// Pre-rendered signature line, e.g.
    /// `pub fn serialize_struct<S: Serializer>(&self, name: &'static str)`.
    pub signature: String,
    pub doc_markdown: String,
    /// Full source rendering of the item, including its body (function
    /// bodies, struct fields, impl methods, etc.). Produced via
    /// `quote::ToTokens` so it captures every identifier and literal in
    /// the item — what ColBERT actually needs to embed.
    pub body: String,
    /// Source file path *relative* to the extracted crate root.
    pub source_file: PathBuf,
    /// Byte offset of the item in `source_file`. `byte_len == 0` means
    /// the span is unpopulated (v1 stub).
    pub byte_start: u64,
    pub byte_len: u64,
    /// 1-based start line, inclusive.
    pub line_start: u32,
    /// 1-based end line, inclusive.
    pub line_end: u32,
    pub visibility: Visibility,
    /// Pre-rendered attribute strings (e.g. "#[deprecated]",
    /// "#[cfg(unix)]"). Order matches source order.
    pub attrs: Vec<String>,
}

impl RustItem {
    /// Build a qualified path from a module path and an item name.
    /// Empty `name` is permitted (impl blocks); the path then ends at
    /// the module.
    pub fn build_qualified_path(
        crate_name: &str,
        module_path: &[String],
        name: Option<&str>,
    ) -> String {
        let mut parts: Vec<&str> = Vec::with_capacity(module_path.len() + 2);
        parts.push(crate_name);
        parts.extend(module_path.iter().map(String::as_str));
        if let Some(n) = name
            && !n.is_empty()
        {
            parts.push(n);
        }
        parts.join("::")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_round_trip() {
        for k in [
            RustItemKind::Fn,
            RustItemKind::Struct,
            RustItemKind::Enum,
            RustItemKind::Union,
            RustItemKind::Trait,
            RustItemKind::Impl,
            RustItemKind::Mod,
            RustItemKind::Const,
            RustItemKind::Static,
            RustItemKind::TypeAlias,
            RustItemKind::Macro,
        ] {
            assert_eq!(RustItemKind::parse(k.as_str()), Some(k));
        }
    }

    #[test]
    fn kind_parse_unknown_is_none() {
        assert_eq!(RustItemKind::parse("module"), None);
        assert_eq!(RustItemKind::parse(""), None);
    }

    #[test]
    fn qualified_path_with_module_and_name() {
        assert_eq!(
            RustItem::build_qualified_path(
                "serde",
                &["ser".to_string()],
                Some("Serializer"),
            ),
            "serde::ser::Serializer",
        );
    }

    #[test]
    fn qualified_path_at_crate_root() {
        assert_eq!(
            RustItem::build_qualified_path("anyhow", &[], Some("Result")),
            "anyhow::Result",
        );
    }

    #[test]
    fn qualified_path_for_impl_drops_name() {
        assert_eq!(
            RustItem::build_qualified_path("serde", &["de".to_string()], None,),
            "serde::de",
        );
    }

    #[test]
    fn qualified_path_handles_deep_module_chain() {
        let modules = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        assert_eq!(
            RustItem::build_qualified_path("crate", &modules, Some("Item")),
            "crate::a::b::c::d::Item",
        );
    }
}
