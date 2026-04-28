//! Parse a single `.rs` file into a list of [`RustItem`]s using `syn`.
//!
//! The visitor walks every top-level item, recurses into inline
//! `mod foo { … }` blocks, and emits one `RustItem` per
//! `fn`/`struct`/`enum`/`union`/`trait`/`impl`/`mod`/`const`/`static`/
//! `type`/`macro_rules!`. File-level `mod foo;` declarations surface as
//! [`PendingModule`] entries so the higher-level crate-tree walker can
//! resolve them to disk paths.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use quote::ToTokens;

use crate::{
    error::{Error, Result},
    item::{RustItem, RustItemKind, Visibility},
};

/// Per-file map from a name as it appears in source (the leaf segment
/// of a `use`, plus any `as` alias) to the fully-qualified path it
/// resolves to. Used to canonicalise the trait path on
/// `impl Trait for Type` so the implementor record can be matched
/// against trait items by their stored qualified path.
///
/// Wildcards (`use foo::*;`) are tracked separately because they
/// don't bind a single name — the resolver walks them as a fallback
/// when nothing in the explicit map matches.
///
/// Special prefixes (`crate::`, `self::`, `super::`) are kept in the
/// stored path verbatim. Resolving them to absolute paths needs the
/// caller's module context, which the parser threads in only when it
/// matters (the trait-path resolver expands them before lookup).
#[derive(Debug, Clone, Default)]
pub(crate) struct UseMap {
    aliases: HashMap<String, Vec<String>>,
    wildcards: Vec<Vec<String>>,
}

impl UseMap {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record a `use a::b::c;` (with `name = "c"`) or
    /// `use a::b::c as d;` (`name = "d"`) — the leaf name is what the
    /// surrounding code refers to.
    fn insert_alias(&mut self, name: String, path: Vec<String>) {
        self.aliases.insert(name, path);
    }

    /// Record a `use a::b::*;` so a later trait reference like `Foo`
    /// (not in the explicit alias map) can be tried under each
    /// wildcard prefix.
    fn insert_wildcard(&mut self, prefix: Vec<String>) {
        self.wildcards.push(prefix);
    }

    /// Resolve a path written in source against this file's `use`
    /// statements. Returns the fully-qualified path components when
    /// the leading segment is bound, otherwise the input path
    /// segments unchanged so the caller can fall back to its own
    /// resolution heuristics.
    pub(crate) fn resolve(&self, segments: &[String]) -> Vec<String> {
        let Some((head, tail)) = segments.split_first() else {
            return Vec::new();
        };
        if let Some(prefix) = self.aliases.get(head) {
            let mut out = prefix.clone();
            out.extend_from_slice(tail);
            return out;
        }
        // Single-segment lookup against the wildcards: any prefix
        // could supply this name, so prefer the first match. Multiple
        // wildcards on the same name resolve unambiguously in real
        // code (it's a compile error otherwise), so we don't need to
        // disambiguate.
        if tail.is_empty()
            && let Some(prefix) = self.wildcards.first()
        {
            let mut out = prefix.clone();
            out.push(head.clone());
            return out;
        }
        segments.to_vec()
    }
}

/// Walk a `use a::b::{c, d::e as f, g::*};` tree, recording every
/// alias and wildcard against `out`. `prefix` carries the path
/// accumulated so far during recursion.
pub(crate) fn collect_use_tree(
    out: &mut UseMap,
    prefix: &mut Vec<String>,
    tree: &syn::UseTree,
) {
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            collect_use_tree(out, prefix, &p.tree);
            prefix.pop();
        }
        syn::UseTree::Name(n) => {
            let ident = n.ident.to_string();
            // Skip `use self;` / `use super;` — those don't bind a
            // new name, they just carry the path forward.
            if matches!(ident.as_str(), "self" | "super" | "crate") {
                return;
            }
            let mut full = prefix.clone();
            full.push(ident.clone());
            out.insert_alias(ident, full);
        }
        syn::UseTree::Rename(r) => {
            let original = r.ident.to_string();
            let alias = r.rename.to_string();
            let mut full = prefix.clone();
            full.push(original);
            out.insert_alias(alias, full);
        }
        syn::UseTree::Glob(_) => {
            out.insert_wildcard(prefix.clone());
        }
        syn::UseTree::Group(g) => {
            for item in &g.items {
                collect_use_tree(out, prefix, item);
            }
        }
    }
}

/// A `mod foo;` declaration that needs resolving against the
/// filesystem. The crate-tree walker (Task 18) consumes these.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingModule {
    pub name: String,
    /// Optional `#[path = "..."]` override.
    pub path_attr: Option<String>,
    /// The module path of the *parent* file (e.g. `["foo", "bar"]` for
    /// a `mod baz;` decl found inside `foo::bar`).
    pub parent_module_path: Vec<String>,
    /// Source file the decl was found in, relative to the crate root.
    pub source_file: PathBuf,
}

/// One `impl Trait for Type` site captured during parse.
///
/// Carries the bits the workspace-wide implementor registry needs to
/// render a rustdoc-flavoured "Implementors" section on the trait's
/// page: the `impl` line itself, the method signatures inside, and
/// the source location for clickability. The trait's resolved path
/// (best-effort against the file's `use` statements) is the match
/// key the registry indexes against.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TraitImplementor {
    /// Fully-qualified trait path, resolved through the file's
    /// `use` statements when possible. Falls back to whatever the
    /// author wrote when no `use` statement covers the leading
    /// segment — partial paths in source land here verbatim.
    pub trait_path: String,
    /// Last identifier of the impl's self-type, generics stripped
    /// (`Tensor`, not `Tensor<T>`). Empty when the self-type isn't
    /// a path (`impl Display for [T]`, `impl Display for &str`).
    pub self_type: String,
    /// The full `impl …` line for display, e.g.
    /// `impl<T: Send> Display for Holder<T>`.
    pub impl_signature: String,
    /// Each contained `fn`'s signature, in source order. Trait-impl
    /// methods are surfaced here (rather than as their own items)
    /// because the trait already declares them.
    pub method_signatures: Vec<String>,
    /// Source path of the impl, for clickability.
    pub source_file: PathBuf,
    pub line_start: u32,
    pub line_end: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParseOutcome {
    pub items: Vec<RustItem>,
    pub pending_modules: Vec<PendingModule>,
    pub implementors: Vec<TraitImplementor>,
}

/// Parse one Rust source file and emit items.
///
/// `module_path` is the module path leading to *this* file (empty for
/// `lib.rs` / `main.rs`). `source_file` is the file's path relative to
/// the extracted crate root, used as `RustItem::source_file`.
pub fn parse_file(
    crate_name: &str,
    crate_version: &semver::Version,
    source_file: &Path,
    module_path: &[String],
    source_text: &str,
) -> Result<ParseOutcome> {
    let file = syn::parse_file(source_text).map_err(|e| Error::Syn {
        path: source_file.display().to_string(),
        source: e,
    })?;

    let mut ctx = ParseCtx {
        crate_name: crate_name.to_string(),
        crate_version: crate_version.clone(),
        source_file: source_file.to_path_buf(),
        out: ParseOutcome::default(),
        use_map: UseMap::new(),
    };
    ctx.visit_items(&file.items, module_path);
    Ok(ctx.out)
}

struct ParseCtx {
    crate_name: String,
    crate_version: semver::Version,
    source_file: PathBuf,
    out: ParseOutcome,
    use_map: UseMap,
}

impl ParseCtx {
    fn visit_items(&mut self, items: &[syn::Item], module_path: &[String]) {
        for item in items {
            self.visit_item(item, module_path);
        }
    }

    fn visit_item(&mut self, item: &syn::Item, module_path: &[String]) {
        match item {
            syn::Item::Fn(it) if is_visible(&it.vis) => self.emit(
                RustItemKind::Fn,
                Some(it.sig.ident.to_string()),
                module_path,
                &it.attrs,
                &it.vis,
                fn_signature(it),
                item,
            ),
            syn::Item::Struct(it) if is_visible(&it.vis) => self.emit(
                RustItemKind::Struct,
                Some(it.ident.to_string()),
                module_path,
                &it.attrs,
                &it.vis,
                struct_signature(it),
                item,
            ),
            syn::Item::Enum(it) if is_visible(&it.vis) => self.emit(
                RustItemKind::Enum,
                Some(it.ident.to_string()),
                module_path,
                &it.attrs,
                &it.vis,
                enum_signature(it),
                item,
            ),
            syn::Item::Union(it) if is_visible(&it.vis) => self.emit(
                RustItemKind::Union,
                Some(it.ident.to_string()),
                module_path,
                &it.attrs,
                &it.vis,
                union_signature(it),
                item,
            ),
            syn::Item::Trait(it) if is_visible(&it.vis) => self.emit(
                RustItemKind::Trait,
                Some(it.ident.to_string()),
                module_path,
                &it.attrs,
                &it.vis,
                trait_signature(it),
                item,
            ),
            // impls describe public API surface even though they don't
            // have a visibility keyword; always emit.
            syn::Item::Impl(it) => {
                self.emit(
                    RustItemKind::Impl,
                    None,
                    module_path,
                    &it.attrs,
                    &syn::Visibility::Inherited,
                    impl_signature(it),
                    item,
                );
                // Inherent impls (no `for` clause) carry the type's
                // public methods / consts / assoc types. Re-emit each
                // pub item under `module::SelfType::name` so they're
                // directly addressable — what rustdoc and docs.rs do
                // for their per-method pages.
                //
                // Trait impls are intentionally not unrolled here: the
                // methods live on the trait already, and the
                // implementor record (collected separately) attaches
                // back to the trait's page.
                if it.trait_.is_none() {
                    if let Some(self_segment) = self_type_segment(&it.self_ty) {
                        let mut inner_path = module_path.to_vec();
                        inner_path.push(self_segment);
                        self.visit_inherent_impl_items(&it.items, &inner_path);
                    }
                } else {
                    self.collect_trait_implementor(it);
                }
            }
            syn::Item::Const(it) if is_visible(&it.vis) => self.emit(
                RustItemKind::Const,
                Some(it.ident.to_string()),
                module_path,
                &it.attrs,
                &it.vis,
                const_signature(it),
                item,
            ),
            syn::Item::Static(it) if is_visible(&it.vis) => self.emit(
                RustItemKind::Static,
                Some(it.ident.to_string()),
                module_path,
                &it.attrs,
                &it.vis,
                static_signature(it),
                item,
            ),
            syn::Item::Type(it) if is_visible(&it.vis) => self.emit(
                RustItemKind::TypeAlias,
                Some(it.ident.to_string()),
                module_path,
                &it.attrs,
                &it.vis,
                type_alias_signature(it),
                item,
            ),
            syn::Item::Macro(it) => {
                // macro_rules! is exported via #[macro_export] attribute
                // rather than `pub`; check for that attr instead.
                let exported =
                    it.attrs.iter().any(|a| a.path().is_ident("macro_export"));
                if exported && let Some(name) = it.ident.as_ref() {
                    self.emit(
                        RustItemKind::Macro,
                        Some(name.to_string()),
                        module_path,
                        &it.attrs,
                        &syn::Visibility::Public(syn::token::Pub::default()),
                        format!("macro_rules! {name}"),
                        item,
                    );
                }
            }
            syn::Item::Mod(m) => self.visit_mod(m, module_path),
            syn::Item::Use(u) => {
                // `use a::b::c;` etc. don't surface as items, but
                // their alias bindings are needed by the trait-path
                // resolver in the impl-record collector.
                let mut prefix = Vec::new();
                collect_use_tree(&mut self.use_map, &mut prefix, &u.tree);
            }
            // Private items, ExternCrate, ForeignMod, TraitAlias,
            // Verbatim — skip.
            _ => {}
        }
    }

    fn visit_mod(&mut self, m: &syn::ItemMod, module_path: &[String]) {
        let name = m.ident.to_string();
        // Many crates declare a private container module
        // (`mod tensor;` in candle-core, `mod sealed;` in serde, etc.)
        // and then `pub use container::Item;` at the crate root. Skipping
        // private modules outright makes those items invisible to the
        // index even though they're reachable from outside through the
        // re-export. We still recurse, but only emit a `Mod` item for
        // public modules so the listing surface matches what callers can
        // address by module path directly.
        if is_visible(&m.vis) {
            self.emit(
                RustItemKind::Mod,
                Some(name.clone()),
                module_path,
                &m.attrs,
                &m.vis,
                format!("mod {name}"),
                &syn::Item::Mod(m.clone()),
            );
        }

        match &m.content {
            Some((_brace, items)) => {
                let mut inner = module_path.to_vec();
                inner.push(name);
                self.visit_items(items, &inner);
            }
            None => {
                self.out.pending_modules.push(PendingModule {
                    name,
                    path_attr: extract_path_attr(&m.attrs),
                    parent_module_path: module_path.to_vec(),
                    source_file: self.source_file.clone(),
                });
            }
        }
    }

    /// Stash a record for each `impl Trait for Type` so the
    /// workspace-wide implementor registry can render a rustdoc-style
    /// "Implementors" section on the trait's page later. Trait path
    /// is resolved through the file's `use` map; if no alias covers
    /// the leading segment, the path the author wrote lands in the
    /// record verbatim.
    fn collect_trait_implementor(&mut self, it: &syn::ItemImpl) {
        let Some((_, trait_path, _)) = &it.trait_ else {
            return;
        };
        let written: Vec<String> = trait_path
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect();
        let resolved = self.use_map.resolve(&written);
        let trait_path_str = resolved.join("::");

        let self_type = self_type_segment(&it.self_ty).unwrap_or_default();
        let impl_signature = impl_signature(it);
        let method_signatures = it
            .items
            .iter()
            .filter_map(|impl_item| match impl_item {
                syn::ImplItem::Fn(m) => Some(impl_method_signature(m)),
                _ => None,
            })
            .collect();

        let (line_start, line_end) = it.line_span();
        self.out.implementors.push(TraitImplementor {
            trait_path: trait_path_str,
            self_type,
            impl_signature,
            method_signatures,
            source_file: self.source_file.clone(),
            line_start,
            line_end,
        });
    }

    /// Emit each `pub` item from an inherent `impl Foo { ... }` block
    /// under `<module_path>::Foo`. Methods become `Fn`, associated
    /// consts become `Const`, and associated types become `TypeAlias`.
    /// `Macro` / `Verbatim` items are skipped — they're rare and
    /// shaping a useful signature for them isn't worth the complexity.
    fn visit_inherent_impl_items(
        &mut self,
        items: &[syn::ImplItem],
        inner_path: &[String],
    ) {
        for item in items {
            match item {
                syn::ImplItem::Fn(m) if is_visible(&m.vis) => self.emit(
                    RustItemKind::Fn,
                    Some(m.sig.ident.to_string()),
                    inner_path,
                    &m.attrs,
                    &m.vis,
                    impl_method_signature(m),
                    m,
                ),
                syn::ImplItem::Const(c) if is_visible(&c.vis) => self.emit(
                    RustItemKind::Const,
                    Some(c.ident.to_string()),
                    inner_path,
                    &c.attrs,
                    &c.vis,
                    impl_const_signature(c),
                    c,
                ),
                syn::ImplItem::Type(t) if is_visible(&t.vis) => self.emit(
                    RustItemKind::TypeAlias,
                    Some(t.ident.to_string()),
                    inner_path,
                    &t.attrs,
                    &t.vis,
                    impl_type_signature(t),
                    t,
                ),
                _ => {}
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit(
        &mut self,
        kind: RustItemKind,
        name: Option<String>,
        module_path: &[String],
        attrs: &[syn::Attribute],
        vis: &syn::Visibility,
        signature: String,
        span_node: &dyn ToTokensSpan,
    ) {
        let qualified_path = RustItem::build_qualified_path(
            &self.crate_name,
            module_path,
            name.as_deref(),
        );
        let (line_start, line_end) = span_node.line_span();
        let body = span_node.full_token_stream();
        let item = RustItem {
            kind,
            crate_name: self.crate_name.clone(),
            crate_version: self.crate_version.clone(),
            module_path: module_path.to_vec(),
            name,
            qualified_path,
            signature,
            doc_markdown: extract_doc_string(attrs),
            body,
            source_file: self.source_file.clone(),
            byte_start: 0, // populated by a future enrichment pass
            byte_len: 0,
            line_start,
            line_end,
            visibility: from_syn_vis(vis),
            attrs: rendered_attrs(attrs),
        };
        self.out.items.push(item);
    }
}

trait ToTokensSpan {
    fn line_span(&self) -> (u32, u32);
    fn full_token_stream(&self) -> String;
}

impl<T: ToTokens> ToTokensSpan for T {
    fn line_span(&self) -> (u32, u32) {
        let tokens = self.to_token_stream();
        let mut min_line = u32::MAX;
        let mut max_line = 0u32;
        for tt in tokens {
            let span = tt.span();
            let start = span.start().line as u32;
            let end = span.end().line as u32;
            if start < min_line {
                min_line = start;
            }
            if end > max_line {
                max_line = end;
            }
        }
        if min_line == u32::MAX {
            return (0, 0);
        }
        (min_line, max_line.max(min_line))
    }

    fn full_token_stream(&self) -> String {
        self.to_token_stream().to_string()
    }
}

fn is_visible(vis: &syn::Visibility) -> bool {
    !matches!(vis, syn::Visibility::Inherited)
}

fn extract_doc_string(attrs: &[syn::Attribute]) -> String {
    let mut lines = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        if let syn::Meta::NameValue(nv) = &attr.meta
            && let syn::Expr::Lit(lit) = &nv.value
            && let syn::Lit::Str(s) = &lit.lit
        {
            let raw = s.value();
            let trimmed = raw.strip_prefix(' ').unwrap_or(&raw);
            lines.push(trimmed.to_string());
        }
    }
    lines.join("\n")
}

fn rendered_attrs(attrs: &[syn::Attribute]) -> Vec<String> {
    attrs
        .iter()
        .filter(|a| !a.path().is_ident("doc"))
        .map(|a| a.to_token_stream().to_string())
        .collect()
}

fn extract_path_attr(attrs: &[syn::Attribute]) -> Option<String> {
    for attr in attrs {
        if !attr.path().is_ident("path") {
            continue;
        }
        if let syn::Meta::NameValue(nv) = &attr.meta
            && let syn::Expr::Lit(lit) = &nv.value
            && let syn::Lit::Str(s) = &lit.lit
        {
            return Some(s.value());
        }
    }
    None
}

fn from_syn_vis(vis: &syn::Visibility) -> Visibility {
    match vis {
        syn::Visibility::Public(_) => Visibility::Public,
        syn::Visibility::Restricted(r) => {
            if r.path.is_ident("crate") {
                Visibility::Crate
            } else {
                Visibility::Restricted
            }
        }
        syn::Visibility::Inherited => Visibility::Private,
    }
}

fn fn_signature(it: &syn::ItemFn) -> String {
    let mut sig = it.sig.to_token_stream().to_string();
    sig = format!("{} {sig}", it.vis.to_token_stream());
    normalize_whitespace(sig.trim())
}

fn struct_signature(it: &syn::ItemStruct) -> String {
    let mut s = String::new();
    s.push_str(&it.vis.to_token_stream().to_string());
    s.push_str(" struct ");
    s.push_str(&it.ident.to_string());
    if !it.generics.params.is_empty() {
        s.push_str(&it.generics.to_token_stream().to_string());
    }
    normalize_whitespace(s.trim())
}

fn enum_signature(it: &syn::ItemEnum) -> String {
    format!(
        "{} enum {}{}",
        it.vis.to_token_stream(),
        it.ident,
        it.generics.to_token_stream()
    )
    .trim()
    .to_string()
}

fn union_signature(it: &syn::ItemUnion) -> String {
    format!(
        "{} union {}{}",
        it.vis.to_token_stream(),
        it.ident,
        it.generics.to_token_stream()
    )
    .trim()
    .to_string()
}

fn trait_signature(it: &syn::ItemTrait) -> String {
    let unsafe_kw = if it.unsafety.is_some() { "unsafe " } else { "" };
    format!(
        "{} {unsafe_kw}trait {}{}",
        it.vis.to_token_stream(),
        it.ident,
        it.generics.to_token_stream()
    )
    .trim()
    .to_string()
}

/// Strip an `impl <Self>` block's self-type down to the last identifier
/// segment, suitable for use as a Rust path component.
///
/// `Tensor`, `Holder<T>`, and `path::to::Tensor` all collapse to
/// `Tensor`. Returns `None` for primitive impls (`impl i32`), slice
/// impls (`impl [T]`), reference impls (`impl &mut Foo`), tuple
/// impls, and anything else where the self-type isn't a path —
/// those don't have a clean segment we can splice into a qualified
/// path, and the existing `Impl` item still surfaces them.
fn self_type_segment(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(tp) => {
            tp.path.segments.last().map(|s| s.ident.to_string())
        }
        // Borrowed self types like `&mut Foo` carry the ident one
        // level deeper; recurse so `impl &mut Foo` still yields `Foo`.
        syn::Type::Reference(r) => self_type_segment(&r.elem),
        _ => None,
    }
}

fn impl_method_signature(m: &syn::ImplItemFn) -> String {
    let mut sig = m.sig.to_token_stream().to_string();
    let vis = m.vis.to_token_stream().to_string();
    if !vis.is_empty() {
        sig = format!("{vis} {sig}");
    }
    normalize_whitespace(sig.trim())
}

fn impl_const_signature(c: &syn::ImplItemConst) -> String {
    format!(
        "{} const {}: {}",
        c.vis.to_token_stream(),
        c.ident,
        c.ty.to_token_stream(),
    )
    .trim()
    .to_string()
}

fn impl_type_signature(t: &syn::ImplItemType) -> String {
    format!(
        "{} type {}{}",
        t.vis.to_token_stream(),
        t.ident,
        t.generics.to_token_stream(),
    )
    .trim()
    .to_string()
}

fn impl_signature(it: &syn::ItemImpl) -> String {
    let mut s = String::from("impl");
    if !it.generics.params.is_empty() {
        s.push_str(&it.generics.to_token_stream().to_string());
    }
    if let Some((_bang, trait_path, _for)) = &it.trait_ {
        s.push(' ');
        s.push_str(&trait_path.to_token_stream().to_string());
        s.push_str(" for");
    }
    s.push(' ');
    s.push_str(&it.self_ty.to_token_stream().to_string());
    normalize_whitespace(&s)
}

fn const_signature(it: &syn::ItemConst) -> String {
    format!(
        "{} const {}: {}",
        it.vis.to_token_stream(),
        it.ident,
        it.ty.to_token_stream()
    )
    .trim()
    .to_string()
}

fn static_signature(it: &syn::ItemStatic) -> String {
    let mut_kw = match it.mutability {
        syn::StaticMutability::Mut(_) => "mut ",
        _ => "",
    };
    format!(
        "{} static {mut_kw}{}: {}",
        it.vis.to_token_stream(),
        it.ident,
        it.ty.to_token_stream()
    )
    .trim()
    .to_string()
}

fn type_alias_signature(it: &syn::ItemType) -> String {
    format!(
        "{} type {}{} = {}",
        it.vis.to_token_stream(),
        it.ident,
        it.generics.to_token_stream(),
        it.ty.to_token_stream()
    )
    .trim()
    .to_string()
}

fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn version() -> semver::Version {
        semver::Version::new(0, 1, 0)
    }

    fn parse(src: &str) -> ParseOutcome {
        parse_file("x", &version(), Path::new("src/lib.rs"), &[], src).unwrap()
    }

    #[test]
    fn parses_a_public_function() {
        let out = parse(
            "/// adds two numbers\npub fn add(a: i32, b: i32) -> i32 { a + b }",
        );
        assert_eq!(out.items.len(), 1);
        let item = &out.items[0];
        assert_eq!(item.kind, RustItemKind::Fn);
        assert_eq!(item.name.as_deref(), Some("add"));
        assert_eq!(item.qualified_path, "x::add");
        assert!(item.signature.contains("fn add"));
        assert_eq!(item.doc_markdown, "adds two numbers");
        assert_eq!(item.visibility, Visibility::Public);
    }

    #[test]
    fn parses_a_struct_with_generics() {
        let out = parse("pub struct Holder<T: Clone> { inner: T }");
        let item = &out.items[0];
        assert_eq!(item.kind, RustItemKind::Struct);
        assert_eq!(item.name.as_deref(), Some("Holder"));
        assert!(item.signature.contains("struct Holder"));
        assert!(
            item.signature.contains("T : Clone")
                || item.signature.contains("T: Clone")
        );
    }

    #[test]
    fn parses_an_enum() {
        let out = parse("pub enum Color { Red, Green, Blue }");
        assert_eq!(out.items[0].kind, RustItemKind::Enum);
        assert_eq!(out.items[0].name.as_deref(), Some("Color"));
    }

    #[test]
    fn parses_a_trait() {
        let out = parse("pub trait Frobnicate { fn frob(&self); }");
        assert_eq!(out.items[0].kind, RustItemKind::Trait);
        assert_eq!(out.items[0].name.as_deref(), Some("Frobnicate"));
    }

    #[test]
    fn parses_an_impl_block_without_a_name() {
        let out = parse("pub struct S; impl S { pub fn new() -> Self { S } }");
        let kinds: Vec<_> = out.items.iter().map(|i| i.kind).collect();
        assert!(kinds.contains(&RustItemKind::Struct));
        assert!(kinds.contains(&RustItemKind::Impl));
        let impl_item = out
            .items
            .iter()
            .find(|i| i.kind == RustItemKind::Impl)
            .unwrap();
        assert_eq!(impl_item.name, None);
    }

    #[test]
    fn parses_const_static_typealias() {
        let out = parse(
            "pub const MAX: u32 = 100;\n\
             pub static GREETING: &str = \"hi\";\n\
             pub type Result<T> = std::result::Result<T, ()>;",
        );
        let kinds: Vec<_> = out.items.iter().map(|i| i.kind).collect();
        assert!(kinds.contains(&RustItemKind::Const));
        assert!(kinds.contains(&RustItemKind::Static));
        assert!(kinds.contains(&RustItemKind::TypeAlias));
    }

    #[test]
    fn parses_exported_macro_rules() {
        let out = parse(
            "#[macro_export]\nmacro_rules! greet { () => { println!(\"hi\") } }",
        );
        assert_eq!(out.items[0].kind, RustItemKind::Macro);
        assert_eq!(out.items[0].name.as_deref(), Some("greet"));
    }

    #[test]
    fn unexported_macro_rules_is_skipped() {
        let out = parse("macro_rules! greet { () => { println!(\"hi\") } }");
        assert!(out.items.is_empty());
    }

    #[test]
    fn inline_module_recurses_and_emits_mod_item() {
        let out = parse("pub mod inner { pub fn ping() {} }");
        let kinds: Vec<_> = out.items.iter().map(|i| i.kind).collect();
        assert!(kinds.contains(&RustItemKind::Mod));
        let inner_fn = out
            .items
            .iter()
            .find(|i| i.kind == RustItemKind::Fn)
            .unwrap();
        assert_eq!(inner_fn.qualified_path, "x::inner::ping");
        assert_eq!(inner_fn.module_path, vec!["inner".to_string()]);
    }

    #[test]
    fn private_inline_module_emits_pub_items_but_no_mod_item() {
        // Crates routinely declare a private container module and
        // re-export its public items at the root via `pub use`. The
        // parser doesn't track those re-exports, but it does need to
        // surface the contents — otherwise large swathes of the
        // popular ecosystem (serde, candle-core, reqwest, …) end up
        // looking empty.
        //
        // The container module itself is private, so we don't emit a
        // `Mod` item for it; only the `pub fn ping` inside surfaces.
        let out = parse("mod inner { pub fn ping() {} }");
        assert_eq!(out.items.len(), 1);
        assert_eq!(out.items[0].kind, RustItemKind::Fn);
        assert_eq!(out.items[0].qualified_path, "x::inner::ping");
    }

    #[test]
    fn external_module_decl_becomes_pending() {
        let out = parse("pub mod inner;");
        assert_eq!(out.items.len(), 1);
        assert_eq!(out.items[0].kind, RustItemKind::Mod);
        assert_eq!(out.pending_modules.len(), 1);
        assert_eq!(out.pending_modules[0].name, "inner");
        assert!(out.pending_modules[0].path_attr.is_none());
    }

    #[test]
    fn path_attr_on_external_pub_mod_is_captured() {
        let out = parse("#[path = \"renamed.rs\"]\npub mod inner;");
        assert_eq!(out.pending_modules.len(), 1);
        assert_eq!(
            out.pending_modules[0].path_attr.as_deref(),
            Some("renamed.rs")
        );
    }

    #[test]
    fn doc_comments_are_joined_across_lines() {
        let out = parse(
            "/// first line\n\
             /// second line\n\
             pub fn doc_me() {}",
        );
        assert_eq!(out.items[0].doc_markdown, "first line\nsecond line");
    }

    #[test]
    fn doc_attribute_form_is_picked_up() {
        let out = parse("#[doc = \" attr form\"]\npub fn attr_doc() {}");
        assert_eq!(out.items[0].doc_markdown, "attr form");
    }

    #[test]
    fn non_doc_attrs_are_kept_separately() {
        let out = parse(
            "/// real doc\n\
             #[deprecated]\n\
             pub fn warn_me() {}",
        );
        assert_eq!(out.items[0].doc_markdown, "real doc");
        assert!(
            out.items[0].attrs.iter().any(|a| a.contains("deprecated")),
            "attrs were {:?}",
            out.items[0].attrs,
        );
    }

    #[test]
    fn pub_crate_visibility_is_classified() {
        let out = parse("pub(crate) fn internal() {}");
        assert_eq!(out.items[0].visibility, Visibility::Crate);
    }

    #[test]
    fn private_items_are_skipped() {
        let out = parse("fn private() {}");
        assert!(
            out.items.is_empty(),
            "private items should not enter the index, got {:?}",
            out.items
                .iter()
                .map(|i| &i.qualified_path)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn syntax_error_returns_syn_error() {
        let err = parse_file(
            "x",
            &version(),
            Path::new("src/lib.rs"),
            &[],
            "fn busted(",
        )
        .unwrap_err();
        assert!(matches!(err, Error::Syn { .. }));
    }

    #[test]
    fn module_path_is_threaded_into_qualified_paths() {
        let out = parse_file(
            "x",
            &version(),
            Path::new("src/foo/bar.rs"),
            &["foo".to_string(), "bar".to_string()],
            "pub fn deep() {}",
        )
        .unwrap();
        assert_eq!(out.items[0].qualified_path, "x::foo::bar::deep");
    }

    #[test]
    fn inherent_impl_emits_per_method_items() {
        let out = parse(
            "pub struct Tensor;\n\
             impl Tensor {\n\
                 pub fn matmul(&self, rhs: &Self) -> Self { *self }\n\
                 pub const RANK: usize = 2;\n\
                 fn private() {}\n\
             }",
        );
        // Struct + impl block + pub fn + pub const = 4 items.
        // Private method is filtered out by `is_visible`.
        let paths: Vec<&str> = out
            .items
            .iter()
            .map(|i| i.qualified_path.as_str())
            .collect();
        assert!(paths.contains(&"x::Tensor"), "struct missing in {paths:?}");
        assert!(
            paths.contains(&"x::Tensor::matmul"),
            "method missing in {paths:?}"
        );
        assert!(
            paths.contains(&"x::Tensor::RANK"),
            "assoc const missing in {paths:?}"
        );
        assert!(
            !paths.iter().any(|p| p.contains("::private")),
            "private method leaked in {paths:?}"
        );
        // The inherent impl block itself is still emitted as before.
        assert!(out.items.iter().any(|i| i.kind == RustItemKind::Impl));
    }

    #[test]
    fn inherent_impl_strips_generics_from_self_type_segment() {
        let out = parse(
            "pub struct Holder<T> { inner: T }\n\
             impl<T> Holder<T> { pub fn new(inner: T) -> Self { Holder { inner } } }",
        );
        let paths: Vec<&str> = out
            .items
            .iter()
            .map(|i| i.qualified_path.as_str())
            .collect();
        assert!(
            paths.contains(&"x::Holder::new"),
            "expected `Holder::new` (generics stripped), got {paths:?}",
        );
    }

    #[test]
    fn trait_impl_does_not_unroll_methods() {
        // Methods on trait impls live on the trait itself; emitting
        // them as inherent-style items would just multiply the index
        // by every implementor.
        let out = parse(
            "pub trait Greet { fn hello(&self); }\n\
             pub struct Foo;\n\
             impl Greet for Foo { fn hello(&self) {} }",
        );
        let paths: Vec<&str> = out
            .items
            .iter()
            .map(|i| i.qualified_path.as_str())
            .collect();
        assert!(
            !paths.contains(&"x::Foo::hello"),
            "trait-impl method should not be unrolled, got {paths:?}",
        );
    }

    /// Build a `UseMap` from a list of `use ...;` statements so the
    /// resolver tests stay focused on lookups rather than syn parsing.
    fn use_map(srcs: &[&str]) -> UseMap {
        let mut map = UseMap::new();
        for src in srcs {
            let file = syn::parse_file(src).unwrap();
            for item in file.items {
                if let syn::Item::Use(u) = item {
                    let mut prefix = Vec::new();
                    collect_use_tree(&mut map, &mut prefix, &u.tree);
                }
            }
        }
        map
    }

    fn segs(s: &str) -> Vec<String> {
        s.split("::").map(String::from).collect()
    }

    #[test]
    fn use_map_resolves_simple_alias() {
        let m = use_map(&["use std::fmt::Display;"]);
        assert_eq!(m.resolve(&segs("Display")), vec!["std", "fmt", "Display"],);
    }

    #[test]
    fn use_map_resolves_renamed_alias() {
        let m = use_map(&["use std::fmt::Display as Disp;"]);
        assert_eq!(m.resolve(&segs("Disp")), vec!["std", "fmt", "Display"],);
        // The original name is *not* bound when an alias is given.
        assert_eq!(m.resolve(&segs("Display")), vec!["Display"]);
    }

    #[test]
    fn use_map_handles_grouped_imports() {
        let m = use_map(&["use std::fmt::{Display, Debug};"]);
        assert_eq!(m.resolve(&segs("Display")), vec!["std", "fmt", "Display"],);
        assert_eq!(m.resolve(&segs("Debug")), vec!["std", "fmt", "Debug"],);
    }

    #[test]
    fn use_map_handles_nested_groups() {
        let m = use_map(&[
            "use serde::{de::{Deserializer, Visitor}, ser::Serializer};",
        ]);
        assert_eq!(m.resolve(&segs("Visitor")), vec!["serde", "de", "Visitor"],);
        assert_eq!(
            m.resolve(&segs("Serializer")),
            vec!["serde", "ser", "Serializer"],
        );
    }

    #[test]
    fn use_map_resolves_partial_path_via_first_segment() {
        // `use std::fmt;` then writing `fmt::Display` in source: the
        // resolver looks up `fmt` and joins the remainder.
        let m = use_map(&["use std::fmt;"]);
        assert_eq!(
            m.resolve(&segs("fmt::Display")),
            vec!["std", "fmt", "Display"],
        );
    }

    #[test]
    fn use_map_wildcard_provides_fallback_for_unbound_names() {
        let m = use_map(&["use std::fmt::*;"]);
        // `Display` isn't explicitly aliased, but the wildcard
        // `std::fmt::*` covers it.
        assert_eq!(m.resolve(&segs("Display")), vec!["std", "fmt", "Display"],);
    }

    #[test]
    fn use_map_returns_input_unchanged_for_unknown_path() {
        let m = use_map(&["use std::fmt::Display;"]);
        assert_eq!(
            m.resolve(&segs("Unrelated::Path")),
            vec!["Unrelated", "Path"],
        );
    }

    #[test]
    fn trait_impl_records_implementor_with_resolved_path() {
        let out = parse(
            "use std::fmt::Display;\n\
             pub struct Foo;\n\
             impl Display for Foo { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { Ok(()) } }",
        );
        assert_eq!(out.implementors.len(), 1);
        let imp = &out.implementors[0];
        assert_eq!(imp.trait_path, "std::fmt::Display");
        assert_eq!(imp.self_type, "Foo");
        assert!(
            imp.impl_signature.contains("impl Display for Foo"),
            "got {}",
            imp.impl_signature
        );
        assert_eq!(imp.method_signatures.len(), 1);
        assert!(
            imp.method_signatures[0].contains("fn fmt"),
            "got {}",
            imp.method_signatures[0]
        );
    }

    #[test]
    fn trait_impl_falls_back_to_authored_path_when_unresolved() {
        // No `use` statement covers the trait; the path written
        // in source is what lands in the record. Cross-crate
        // resolution at registry-merge time is what compensates.
        let out = parse(
            "pub struct Foo;\n\
             impl serde::Serialize for Foo { fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error> where S: serde::Serializer { unimplemented!() } }",
        );
        assert_eq!(out.implementors.len(), 1);
        assert_eq!(out.implementors[0].trait_path, "serde::Serialize");
    }

    #[test]
    fn inherent_impl_does_not_produce_implementor_record() {
        let out = parse("pub struct Foo;\nimpl Foo { pub fn bar() {} }");
        assert!(
            out.implementors.is_empty(),
            "inherent impls don't belong on the trait page",
        );
    }

    #[test]
    fn primitive_self_type_impl_is_not_unrolled() {
        // `impl i32 { ... }` isn't a thing crates write, but `impl
        // [T] { ... }` and `impl &mut Foo` are. The self-type isn't a
        // clean Rust path segment, so we leave the impl block as the
        // only surfaced item rather than inventing a synthetic name.
        let out = parse("impl<T> [T] { fn first_or_none(&self) {} }");
        let paths: Vec<&str> = out
            .items
            .iter()
            .map(|i| i.qualified_path.as_str())
            .collect();
        assert!(
            !paths.iter().any(|p| p.contains("::first_or_none")),
            "slice-impl method should not be re-emitted, got {paths:?}",
        );
    }
}
