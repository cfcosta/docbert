//! Stable message identity.
//!
//! The identity of a mail message is `blake3` over its normalized
//! `Message-ID` header. It is deliberately **not** derived from a path,
//! a folder, or an IMAP UID, because the same message appears in many
//! folders with a different UID in each one. See `docs/mailbert.md` §4.1.
//!
//! Messages that carry no `Message-ID` fall back to a hash over
//! `(date, from, subject, body)`. The two derivations use different
//! domain separators, so a crafted `Message-ID` cannot produce the
//! identity of a content-derived message.
//!
//! The CLI accepts any unique prefix of the short form, like git.

use std::fmt;

/// Length of the short display form, in hex characters.
pub const SHORT_LEN: usize = 16;

/// Length of the full hex digest.
const FULL_LEN: usize = 64;

/// Domain separator for identities derived from a `Message-ID`.
const MESSAGE_ID_DOMAIN: &[u8] = b"mailbert.mid.v1\0";

/// Domain separator for identities derived from message content.
const CONTENT_DOMAIN: &[u8] = b"mailbert.content.v1\0";

/// Hash a sequence of fields under a domain separator.
///
/// Each field is length-prefixed rather than delimited, so no field
/// content can forge a boundary. A `\0` delimiter would not be enough:
/// a subject may legally contain a NUL byte.
fn hash_fields(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);

    for field in fields {
        hasher.update(&(field.len() as u64).to_be_bytes());
        hasher.update(field);
    }

    *hasher.finalize().as_bytes()
}

/// Whether `prefix` is a usable hex prefix: non-empty, all hex digits,
/// and no longer than a full digest.
fn is_hex_prefix(prefix: &str) -> bool {
    !prefix.is_empty()
        && prefix.len() <= FULL_LEN
        && prefix.chars().all(|c| c.is_ascii_hexdigit())
}

/// Normalize a raw `Message-ID` header value.
///
/// Removes surrounding angle brackets and all whitespace (headers fold
/// across lines), then lowercases the domain. Per RFC 5322 the local
/// part is case-sensitive and the domain is not.
///
/// Returns `None` when nothing remains, which is the signal to fall back
/// to a content hash.
///
/// # Examples
///
/// ```
/// use mailbert_core::message_id::normalize_message_id;
///
/// assert_eq!(
///     normalize_message_id("  <CAF7@Mail.Example.COM> "),
///     Some("CAF7@mail.example.com".to_string())
/// );
/// // No angle brackets, already lowercase: unchanged.
/// assert_eq!(normalize_message_id("a@b.com"), Some("a@b.com".to_string()));
/// // Nothing left to hash.
/// assert_eq!(normalize_message_id("<>"), None);
/// ```
pub fn normalize_message_id(raw: &str) -> Option<String> {
    let compact: String = raw.chars().filter(|c| !c.is_whitespace()).collect();

    let body = compact.strip_prefix('<').unwrap_or(&compact);
    let body = body.strip_suffix('>').unwrap_or(body);

    if body.is_empty() {
        return None;
    }

    // The domain is whatever follows the final `@`. A local part may
    // legally contain `@` inside a quoted string.
    match body.rsplit_once('@') {
        Some((local, domain)) => {
            Some(format!("{local}@{}", domain.to_lowercase()))
        }
        None => Some(body.to_string()),
    }
}

/// A stable message identity.
///
/// # Examples
///
/// ```
/// use mailbert_core::MessageId;
///
/// let id = MessageId::from_message_id("<abc@example.com>").unwrap();
///
/// // The same message seen in another folder hashes the same.
/// assert_eq!(id, MessageId::from_message_id("<abc@EXAMPLE.com>").unwrap());
/// assert!(id.full_hex().starts_with(&id.short()));
/// ```
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct MessageId {
    hash: [u8; 32],
}

impl MessageId {
    /// Derive an identity from a raw `Message-ID` header value.
    ///
    /// Returns `None` when the header normalizes to nothing.
    pub fn from_message_id(raw: &str) -> Option<Self> {
        let normalized = normalize_message_id(raw)?;

        Some(Self {
            hash: hash_fields(MESSAGE_ID_DOMAIN, &[normalized.as_bytes()]),
        })
    }

    /// Derive an identity from message content.
    ///
    /// Used only when a message carries no usable `Message-ID`.
    pub fn from_content(
        date: i64,
        from: &str,
        subject: &str,
        body: &str,
    ) -> Self {
        Self {
            hash: hash_fields(
                CONTENT_DOMAIN,
                &[
                    &date.to_be_bytes(),
                    from.as_bytes(),
                    subject.as_bytes(),
                    body.as_bytes(),
                ],
            ),
        }
    }

    /// Derive an identity, preferring the `Message-ID` header.
    pub fn derive(
        raw_message_id: Option<&str>,
        date: i64,
        from: &str,
        subject: &str,
        body: &str,
    ) -> Self {
        raw_message_id
            .and_then(Self::from_message_id)
            .unwrap_or_else(|| Self::from_content(date, from, subject, body))
    }

    /// The full 64-character hex digest.
    pub fn full_hex(&self) -> String {
        blake3::Hash::from_bytes(self.hash).to_hex().to_string()
    }

    /// The short display form, [`SHORT_LEN`] hex characters.
    pub fn short(&self) -> String {
        self.full_hex()[..SHORT_LEN].to_string()
    }

    /// Read an identity back from its full hex form.
    ///
    /// The store keys its entries by [`MessageId::full_hex`], so this is
    /// the way back from a key to an identity. Returns `None` when the
    /// text is not 64 hex characters.
    ///
    /// # Examples
    ///
    /// ```
    /// use mailbert_core::MessageId;
    ///
    /// let id = MessageId::from_message_id("<abc@example.com>").unwrap();
    ///
    /// assert_eq!(MessageId::from_hex(&id.full_hex()), Some(id));
    /// assert_eq!(MessageId::from_hex(&id.short()), None);
    /// ```
    pub fn from_hex(text: &str) -> Option<Self> {
        if text.len() != FULL_LEN {
            return None;
        }

        let mut hash = [0u8; 32];
        for (byte, pair) in hash.iter_mut().zip(text.as_bytes().chunks(2)) {
            let pair = std::str::from_utf8(pair).ok()?;
            *byte = u8::from_str_radix(pair, 16).ok()?;
        }

        Some(Self { hash })
    }

    /// The numeric key used by the embedding database.
    pub fn numeric(&self) -> u64 {
        u64::from_be_bytes([
            self.hash[0],
            self.hash[1],
            self.hash[2],
            self.hash[3],
            self.hash[4],
            self.hash[5],
            self.hash[6],
            self.hash[7],
        ])
    }

    /// Whether `prefix` is a case-insensitive hex prefix of this identity.
    pub fn matches_prefix(&self, prefix: &str) -> bool {
        is_hex_prefix(prefix)
            && self.full_hex().starts_with(&prefix.to_ascii_lowercase())
    }
}

impl fmt::Display for MessageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.short())
    }
}

/// The outcome of resolving a user-typed prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixMatch {
    /// Exactly one identity matched.
    Unique(MessageId),
    /// More than one identity matched. Sorted, for a stable error message.
    Ambiguous(Vec<MessageId>),
    /// Nothing matched, or the prefix was not hex.
    NotFound,
}

/// Resolve a user-typed prefix against a set of known identities.
///
/// An empty or non-hex prefix is [`PrefixMatch::NotFound`].
///
/// # Examples
///
/// ```
/// use mailbert_core::MessageId;
/// use mailbert_core::message_id::{PrefixMatch, resolve_prefix};
///
/// let id = MessageId::from_message_id("<abc@example.com>").unwrap();
///
/// assert_eq!(resolve_prefix(&id.short(), [id]), PrefixMatch::Unique(id));
/// assert_eq!(resolve_prefix("nothex", [id]), PrefixMatch::NotFound);
/// ```
pub fn resolve_prefix<I>(prefix: &str, candidates: I) -> PrefixMatch
where
    I: IntoIterator<Item = MessageId>,
{
    if !is_hex_prefix(prefix) {
        return PrefixMatch::NotFound;
    }

    let mut hits: Vec<MessageId> = candidates
        .into_iter()
        .filter(|id| id.matches_prefix(prefix))
        .collect();

    match hits.len() {
        0 => PrefixMatch::NotFound,
        1 => PrefixMatch::Unique(hits[0]),
        _ => {
            hits.sort();
            PrefixMatch::Ambiguous(hits)
        }
    }
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_derivation_is_deterministic` | algebraic (idempotence) | Identity must survive a re-sync, or tags and saved ids rot. |
    //! | `prop_normalization_is_metamorphic` | metamorphic | Angle brackets, folded whitespace, and domain case are all no-ops on the wire. The same message seen in two folders must hash the same. |
    //! | `prop_local_part_stays_case_sensitive` | metamorphic (negative) | RFC 5322 makes the local part case-sensitive. Getting it backwards merges distinct messages. |
    //! | `prop_distinct_ids_do_not_collide` | algebraic (injectivity) | A collision silently merges two messages into one entry. |
    //! | `prop_short_is_a_prefix_of_full` | structural | git-style prefix resolution depends on it. |
    //! | `prop_content_fallback_is_field_sensitive` | metamorphic | Without this, every message that lacks a `Message-ID` collapses together. |
    //! | `prop_domains_are_separated` | differential | A crafted `Message-ID` must not be able to produce a content-derived identity. |
    //! | `prop_full_short_resolves_uniquely` | round-trip | The id printed by `search` must work in `get`. |
    //! | `prop_member_prefix_is_never_not_found` | round-trip | Any prefix of a known id is Unique or Ambiguous, never missing. |

    use hegel::{TestCase, generators as gs};

    use super::*;

    // -----------------------------------------------------------------
    // Generators. Each builds valid input by construction, so no test
    // case is ever rejected.
    // -----------------------------------------------------------------

    /// A `(local, domain)` pair. The local part always starts with a
    /// letter so that uppercasing it is a real change, and the domain is
    /// always lowercase so that uppercasing it is also a real change.
    #[hegel::composite]
    fn message_id_parts(tc: TestCase) -> (String, String) {
        let head: String =
            tc.draw(gs::text().alphabet("abcdefgh").min_size(1).max_size(1));
        let tail: String = tc.draw(
            gs::text()
                .alphabet("abcdefgh0123456789.-_+")
                .min_size(0)
                .max_size(20),
        );
        let domain: String = tc.draw(
            gs::text()
                .alphabet("abcdefgh0123456789-")
                .min_size(1)
                .max_size(12),
        );

        (format!("{head}{tail}"), format!("{domain}.example"))
    }

    /// A set of pairwise-distinct identities, distinct by construction.
    #[hegel::composite]
    fn distinct_ids(tc: TestCase) -> Vec<MessageId> {
        let mut locals: Vec<String> = tc.draw(
            gs::vecs(
                gs::text()
                    .alphabet("abcdef0123456789")
                    .min_size(1)
                    .max_size(8),
            )
            .min_size(1)
            .max_size(8),
        );
        locals.sort();
        locals.dedup();

        locals
            .iter()
            .map(|local| {
                MessageId::from_message_id(&format!("<{local}@example.com>"))
                    .expect("non-empty local part yields an identity")
            })
            .collect()
    }

    // -----------------------------------------------------------------
    // Unit tests: the exact cases the docs promise.
    // -----------------------------------------------------------------

    #[test]
    fn normalize_strips_brackets_and_lowercases_only_the_domain() {
        assert_eq!(
            normalize_message_id("<CAF7abc@Mail.Example.COM>"),
            Some("CAF7abc@mail.example.com".to_string())
        );
    }

    #[test]
    fn normalize_removes_folded_whitespace() {
        assert_eq!(
            normalize_message_id("<abc\r\n  def@example.com>"),
            Some("abcdef@example.com".to_string())
        );
    }

    #[test]
    fn normalize_splits_on_the_last_at_sign() {
        // A local part may legally contain `@` inside a quoted string.
        // The domain is whatever follows the final `@`.
        assert_eq!(
            normalize_message_id("<a@b@EXAMPLE.COM>"),
            Some("a@b@example.com".to_string())
        );
    }

    #[test]
    fn normalize_handles_a_message_id_with_no_domain() {
        assert_eq!(
            normalize_message_id("<local-only>"),
            Some("local-only".to_string())
        );
    }

    #[test]
    fn normalize_rejects_an_empty_header() {
        assert_eq!(normalize_message_id("<>"), None);
        assert_eq!(normalize_message_id("   "), None);
        assert_eq!(normalize_message_id(""), None);
    }

    #[test]
    fn derive_falls_back_to_content_when_the_header_is_missing() {
        let from_header =
            MessageId::derive(Some("<a@b.com>"), 1, "x", "y", "z");
        let from_content = MessageId::derive(None, 1, "x", "y", "z");

        assert_eq!(
            from_header,
            MessageId::from_message_id("<a@b.com>").unwrap()
        );
        assert_eq!(from_content, MessageId::from_content(1, "x", "y", "z"));
        assert_ne!(from_header, from_content);
    }

    #[test]
    fn derive_falls_back_when_the_header_is_present_but_empty() {
        assert_eq!(
            MessageId::derive(Some("<>"), 1, "x", "y", "z"),
            MessageId::from_content(1, "x", "y", "z")
        );
    }

    #[test]
    fn short_is_sixteen_hex_chars() {
        let id = MessageId::from_message_id("<a@b.com>").unwrap();
        assert_eq!(id.short().len(), SHORT_LEN);
        assert_eq!(id.full_hex().len(), 64);
        assert!(id.short().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn resolve_prefix_rejects_a_non_hex_prefix() {
        let id = MessageId::from_message_id("<a@b.com>").unwrap();
        assert_eq!(
            resolve_prefix("zzzz", [id]),
            PrefixMatch::NotFound,
            "`z` is not a hex digit"
        );
    }

    #[test]
    fn resolve_prefix_rejects_an_empty_prefix() {
        let id = MessageId::from_message_id("<a@b.com>").unwrap();
        assert_eq!(resolve_prefix("", [id]), PrefixMatch::NotFound);
    }

    #[test]
    fn resolve_prefix_reports_ambiguity() {
        // 40 identities over 16 possible leading hex digits: the
        // pigeonhole principle guarantees a shared first character.
        let ids: Vec<MessageId> = (0..40)
            .map(|i| {
                MessageId::from_message_id(&format!("<m{i}@example.com>"))
                    .unwrap()
            })
            .collect();

        let (prefix, expected) = ids
            .iter()
            .map(|id| {
                let p = id.short()[..1].to_string();
                let hits: Vec<MessageId> = ids
                    .iter()
                    .copied()
                    .filter(|c| c.matches_prefix(&p))
                    .collect();
                (p, hits)
            })
            .find(|(_, hits)| hits.len() > 1)
            .expect("40 ids must share a leading hex digit");

        let mut expected = expected;
        expected.sort();

        assert_eq!(
            resolve_prefix(&prefix, ids),
            PrefixMatch::Ambiguous(expected)
        );
    }

    #[test]
    fn resolve_prefix_is_case_insensitive() {
        let id = MessageId::from_message_id("<a@b.com>").unwrap();
        let upper = id.short().to_uppercase();

        assert_eq!(resolve_prefix(&upper, [id]), PrefixMatch::Unique(id));
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 200)]
    fn prop_derivation_is_deterministic(tc: TestCase) {
        let (local, domain) = tc.draw(message_id_parts());
        let raw = format!("<{local}@{domain}>");

        assert_eq!(
            MessageId::from_message_id(&raw),
            MessageId::from_message_id(&raw)
        );
    }

    #[hegel::test(test_cases = 200)]
    fn prop_normalization_is_metamorphic(tc: TestCase) {
        let (local, domain) = tc.draw(message_id_parts());

        let plain = MessageId::from_message_id(&format!("{local}@{domain}"));
        let bracketed =
            MessageId::from_message_id(&format!("<{local}@{domain}>"));
        let padded =
            MessageId::from_message_id(&format!("  <{local}@{domain}>  "));
        let folded =
            MessageId::from_message_id(&format!("<{local}@{domain}\r\n >"));
        let shouty = MessageId::from_message_id(&format!(
            "<{local}@{}>",
            domain.to_uppercase()
        ));

        assert_eq!(plain, bracketed, "angle brackets must not change identity");
        assert_eq!(plain, padded, "padding must not change identity");
        assert_eq!(plain, folded, "header folding must not change identity");
        assert_eq!(plain, shouty, "domain case must not change identity");
    }

    #[hegel::test(test_cases = 200)]
    fn prop_local_part_stays_case_sensitive(tc: TestCase) {
        let (local, domain) = tc.draw(message_id_parts());
        // The generator guarantees a leading ASCII letter.
        let shouty_local = local.to_uppercase();
        tc.assume(shouty_local != local);

        assert_ne!(
            MessageId::from_message_id(&format!("<{local}@{domain}>")),
            MessageId::from_message_id(&format!("<{shouty_local}@{domain}>")),
            "RFC 5322 makes the local part case-sensitive"
        );
    }

    #[hegel::test(test_cases = 200)]
    fn prop_distinct_ids_do_not_collide(tc: TestCase) {
        let (local_a, domain_a) = tc.draw(message_id_parts());
        let (local_b, domain_b) = tc.draw(message_id_parts());
        tc.assume(local_a != local_b || domain_a != domain_b);

        assert_ne!(
            MessageId::from_message_id(&format!("<{local_a}@{domain_a}>")),
            MessageId::from_message_id(&format!("<{local_b}@{domain_b}>")),
        );
    }

    #[hegel::test(test_cases = 200)]
    fn prop_short_is_a_prefix_of_full(tc: TestCase) {
        let (local, domain) = tc.draw(message_id_parts());
        let id = MessageId::from_message_id(&format!("<{local}@{domain}>"))
            .expect("generated local parts are never empty");

        assert_eq!(id.full_hex().len(), 64);
        assert_eq!(id.short().len(), SHORT_LEN);
        assert!(id.full_hex().starts_with(&id.short()));
        assert!(id.matches_prefix(&id.short()));
        assert!(id.matches_prefix(&id.full_hex()));
    }

    #[hegel::test(test_cases = 200)]
    fn prop_content_fallback_is_field_sensitive(tc: TestCase) {
        let date: i64 = tc
            .draw(gs::integers::<i64>().min_value(0).max_value(2_000_000_000));
        let from: String = tc.draw(gs::text().min_size(0).max_size(20));
        let subject: String = tc.draw(gs::text().min_size(0).max_size(30));
        let body: String = tc.draw(gs::text().min_size(0).max_size(50));

        let base = MessageId::from_content(date, &from, &subject, &body);

        assert_ne!(
            base,
            MessageId::from_content(date + 1, &from, &subject, &body),
            "date must participate"
        );
        assert_ne!(
            base,
            MessageId::from_content(date, &format!("{from}x"), &subject, &body),
            "from must participate"
        );
        assert_ne!(
            base,
            MessageId::from_content(date, &from, &format!("{subject}x"), &body),
            "subject must participate"
        );
        assert_ne!(
            base,
            MessageId::from_content(date, &from, &subject, &format!("{body}x")),
            "body must participate"
        );
    }

    #[hegel::test(test_cases = 200)]
    fn prop_content_fields_are_unambiguous(tc: TestCase) {
        // Without a separator between fields, ("ab", "c") and ("a", "bc")
        // would hash identically and two different messages would merge.
        let a: String = tc.draw(gs::text().min_size(1).max_size(12));
        let b: String = tc.draw(gs::text().min_size(1).max_size(12));
        let joined = format!("{a}{b}");

        assert_ne!(
            MessageId::from_content(0, &a, &b, ""),
            MessageId::from_content(0, &joined, "", ""),
            "field boundaries must be unambiguous"
        );
    }

    #[hegel::test(test_cases = 200)]
    fn prop_domains_are_separated(tc: TestCase) {
        let (local, domain) = tc.draw(message_id_parts());
        let raw = format!("{local}@{domain}");

        // Feed the identical string through both derivations. Different
        // domain separators must keep them apart.
        assert_ne!(
            MessageId::from_message_id(&raw).unwrap(),
            MessageId::from_content(0, &raw, "", ""),
        );
    }

    #[hegel::test(test_cases = 100)]
    fn prop_full_short_resolves_uniquely(tc: TestCase) {
        let ids = tc.draw(distinct_ids());

        for id in &ids {
            assert_eq!(
                resolve_prefix(&id.short(), ids.clone()),
                PrefixMatch::Unique(*id),
                "the full short form must always resolve to itself"
            );
        }
    }

    #[hegel::test(test_cases = 100)]
    fn prop_member_prefix_is_never_not_found(tc: TestCase) {
        let ids = tc.draw(distinct_ids());
        let idx = tc.draw(
            gs::integers::<usize>()
                .min_value(0)
                .max_value(ids.len() - 1),
        );
        let len =
            tc.draw(gs::integers::<usize>().min_value(1).max_value(SHORT_LEN));

        let prefix = ids[idx].short()[..len].to_string();

        assert_ne!(
            resolve_prefix(&prefix, ids),
            PrefixMatch::NotFound,
            "a prefix of a known id must resolve or be ambiguous"
        );
    }
}
