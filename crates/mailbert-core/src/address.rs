//! Mail addresses, and the folding that contact resolution uses.
//!
//! The index stores an address in a `STRING` field, so every address
//! that reaches it must be normalized the same way. Parsing therefore
//! lowercases the address and keeps the display name as the header
//! spelled it, because only the name is worth showing back to a reader.
//!
//! See `docs/mailbert.md` §5.6.

use std::fmt;

use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

/// The characters that split a local part into words.
///
/// `alice.example`, `alice-example`, and `alice+mailbert` all name the
/// same person in three ways, so contact resolution reads each part.
const LOCAL_PART_BREAKS: [char; 4] = ['.', '-', '_', '+'];

/// One mailbox from a header.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct Address {
    /// The display name, with its whitespace collapsed and its quoting
    /// removed. `None` when the header carried no name.
    pub name: Option<String>,

    /// The address, lowercased. This is what the index stores.
    pub address: String,
}

/// The characters that force a display name to be quoted.
const SPECIALS: [char; 12] =
    ['(', ')', '<', '>', '[', ']', ':', ';', '@', '\\', ',', '"'];

/// Lowercase `text` and remove its canonical combining marks.
///
/// `Cainã` folds to `caina`, so `mailbert contacts caina` finds the
/// contact without the reader typing the tilde.
///
/// # Examples
///
/// ```
/// use mailbert_core::address::fold;
///
/// assert_eq!(fold("Cainã Costa"), "caina costa");
/// ```
pub fn fold(text: &str) -> String {
    text.nfd()
        .filter(|c| !is_combining_mark(*c))
        .flat_map(char::to_lowercase)
        .collect()
}

/// The folded words of `text`.
///
/// Whitespace and the local-part separators both break a word, so
/// `Alice Example` and `alice.example` give the same two words.
///
/// # Examples
///
/// ```
/// use mailbert_core::address::words;
///
/// assert_eq!(words("bob.smith"), vec!["bob", "smith"]);
/// ```
pub fn words(text: &str) -> Vec<String> {
    fold(text)
        .split(|c: char| c.is_whitespace() || LOCAL_PART_BREAKS.contains(&c))
        .filter(|word| !word.is_empty())
        .map(str::to_string)
        .collect()
}

/// Parse one address from a header value.
///
/// # Examples
///
/// ```
/// use mailbert_core::address;
///
/// let parsed = address::parse("Alice Example <Alice@EXAMPLE.COM>").unwrap();
///
/// assert_eq!(parsed.name.as_deref(), Some("Alice Example"));
/// assert_eq!(parsed.address, "alice@example.com");
/// ```
pub fn parse(raw: &str) -> Option<Address> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }

    let (name, addr) = split(raw);

    Some(Address {
        name: clean_name(name),
        address: normalize(addr)?,
    })
}

/// Parse a comma-separated address list, as `To` and `Cc` carry it.
///
/// A group (`Team: a@x.example, b@y.example;`) contributes its members
/// and not its label. An unparsable entry is dropped, because one bad
/// address must not cost the message its other recipients.
///
/// # Examples
///
/// ```
/// use mailbert_core::address;
///
/// let list = address::parse_list(
///     r#"Alice <a@x.example>, "Example, Bob" <b@y.example>"#,
/// );
///
/// assert_eq!(list.len(), 2);
/// assert_eq!(list[1].name.as_deref(), Some("Example, Bob"));
/// ```
pub fn parse_list(raw: &str) -> Vec<Address> {
    let mut found = Vec::new();
    let mut entry = String::new();
    let mut quoted = false;
    let mut escaped = false;
    let mut angle = 0usize;
    let mut paren = 0usize;

    for c in raw.chars() {
        if escaped {
            entry.push(c);
            escaped = false;
            continue;
        }

        match c {
            '\\' if quoted => {
                entry.push(c);
                escaped = true;
            }
            '"' => {
                quoted = !quoted;
                entry.push(c);
            }
            _ if quoted => entry.push(c),
            '<' => {
                angle += 1;
                entry.push(c);
            }
            '>' => {
                angle = angle.saturating_sub(1);
                entry.push(c);
            }
            '(' => {
                paren += 1;
                entry.push(c);
            }
            ')' => {
                paren = paren.saturating_sub(1);
                entry.push(c);
            }
            // A group label. Everything before the colon names the
            // group, and the members follow it.
            ':' if angle == 0 && paren == 0 => entry.clear(),
            ',' | ';' if angle == 0 && paren == 0 => {
                found.extend(parse(&entry));
                entry.clear();
            }
            _ => entry.push(c),
        }
    }

    found.extend(parse(&entry));
    found
}

/// Split a header value into its display name and its address.
fn split(raw: &str) -> (Option<&str>, &str) {
    if let Some(open) = raw.rfind('<')
        && let Some(close) = raw.rfind('>')
        && close > open
    {
        return (Some(&raw[..open]), &raw[open + 1..close]);
    }

    // The RFC 822 comment form: `alice@example.com (Alice Example)`.
    if let Some(open) = raw.find('(')
        && raw.ends_with(')')
    {
        return (Some(&raw[open + 1..raw.len() - 1]), &raw[..open]);
    }

    (None, raw)
}

/// Lowercase an address, and reject anything that is not one.
fn normalize(addr: &str) -> Option<String> {
    let addr = addr.trim();
    if addr.chars().any(char::is_whitespace) {
        return None;
    }

    let (local, domain) = addr.split_once('@')?;
    if local.is_empty() || domain.is_empty() || domain.contains('@') {
        return None;
    }

    Some(format!(
        "{}@{}",
        local.to_lowercase(),
        domain.to_lowercase()
    ))
}

/// Remove the quoting of a display name and collapse its whitespace.
///
/// A header folds over several lines, so a name arrives with newlines
/// and indentation that are not part of it.
fn clean_name(name: Option<&str>) -> Option<String> {
    let raw = name?.trim();

    let unquoted = match raw.strip_prefix('"').and_then(|r| r.strip_suffix('"'))
    {
        Some(inner) => unescape(inner),
        None => raw.to_string(),
    };

    let collapsed = unquoted.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return None;
    }

    Some(collapsed)
}

fn unescape(quoted: &str) -> String {
    let mut out = String::with_capacity(quoted.len());
    let mut escaped = false;

    for c in quoted.chars() {
        match c {
            _ if escaped => {
                out.push(c);
                escaped = false;
            }
            '\\' => escaped = true,
            _ => out.push(c),
        }
    }

    out
}

/// Whether a display name has to be quoted to survive a round trip.
fn needs_quoting(name: &str) -> bool {
    name.is_empty()
        || name.trim() != name
        || name.chars().any(|c| SPECIALS.contains(&c))
}

impl Address {
    /// The part before the `@`.
    /// Build an address from parts that a parser already decoded.
    ///
    /// Returns `None` when the address is not one that mailbert can
    /// index, which is the same rule that [`parse`] applies.
    ///
    /// # Examples
    ///
    /// ```
    /// use mailbert_core::Address;
    ///
    /// let found = Address::new(Some("  Alice   Smith "), "Alice@X.TEST")
    ///     .unwrap();
    ///
    /// assert_eq!(found.name.as_deref(), Some("Alice Smith"));
    /// assert_eq!(found.address, "alice@x.test");
    /// assert_eq!(Address::new(None, "not an address"), None);
    /// ```
    pub fn new(name: Option<&str>, address: &str) -> Option<Self> {
        Some(Self {
            name: clean_name(name),
            address: normalize(address)?,
        })
    }

    pub fn local_part(&self) -> &str {
        self.address
            .split_once('@')
            .map_or(self.address.as_str(), |(local, _)| local)
    }

    /// The part after the `@`.
    pub fn domain(&self) -> &str {
        self.address
            .split_once('@')
            .map_or("", |(_, domain)| domain)
    }

    /// The folded words of the local part and the display name.
    ///
    /// Contact resolution matches a needle against these. The domain is
    /// not among them, or `from:com` would match everyone.
    pub fn words(&self) -> Vec<String> {
        let mut found = words(self.local_part());

        for word in self.name.iter().flat_map(|name| words(name)) {
            if !found.contains(&word) {
                found.push(word);
            }
        }

        found
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(name) = &self.name else {
            return write!(f, "{}", self.address);
        };

        if !needs_quoting(name) {
            return write!(f, "{name} <{}>", self.address);
        }

        let escaped = name.replace('\\', r"\\").replace('"', "\\\"");
        write!(f, "\"{escaped}\" <{}>", self.address)
    }
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_display_round_trips` | round-trip | An address mailbert writes into a header must be one it can read back. |
    //! | `prop_addresses_are_always_normalized` | invariant | The index compares addresses exactly. One unnormalized address is one address the reader can never find. |
    //! | `prop_a_list_keeps_every_address` | model-based | Losing a recipient loses each `to:` query about them. |
    //! | `prop_a_comma_in_a_name_never_splits` | invariant | `"Example, Alice"` is one recipient, and splitting it invents a second. |
    //! | `prop_folding_is_idempotent` | algebraic | Folding runs on both the needle and the contact. If it were not stable, the two sides could disagree. |
    //! | `prop_folding_erases_case_and_accents` | differential | This is what makes `caina` find `Cainã`. |
    //! | `prop_words_are_never_blank` | invariant | A blank word would match every needle. |
    //! | `prop_an_address_needs_an_at_sign` | invariant | Cheap guard: prose must never become an address. |

    use hegel::{TestCase, generators as gs};

    use super::*;

    // -----------------------------------------------------------------
    // Generators.
    // -----------------------------------------------------------------

    /// Already-normalized addresses, so a round trip can compare them.
    fn address_text() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "alice@example.com".to_string(),
            "bob.smith@work.example".to_string(),
            "me+mailbert@cfcosta.com".to_string(),
            "no-reply@lists.example.org".to_string(),
            "sam@x.example".to_string(),
        ])
    }

    /// Display names, including the ones that force quoting.
    fn display_name() -> impl gs::Generator<Option<String>> {
        gs::sampled_from(vec![
            None,
            Some("Alice Example".to_string()),
            Some("Example, Alice".to_string()),
            Some("Cainã Costa".to_string()),
            Some("Zoë O'Brien".to_string()),
            Some("bob".to_string()),
        ])
    }

    #[hegel::composite]
    fn an_address(tc: TestCase) -> Address {
        Address {
            name: tc.draw(display_name()),
            address: tc.draw(address_text()),
        }
    }

    // -----------------------------------------------------------------
    // Unit tests.
    // -----------------------------------------------------------------

    #[test]
    fn a_bare_address_parses_without_a_name() {
        let address = parse("alice@example.com").unwrap();

        assert_eq!(address.name, None);
        assert_eq!(address.address, "alice@example.com");
        assert_eq!(address.local_part(), "alice");
        assert_eq!(address.domain(), "example.com");
    }

    #[test]
    fn angle_brackets_carry_the_address() {
        assert_eq!(
            parse("<alice@example.com>").unwrap().address,
            "alice@example.com"
        );

        let named = parse("Alice Example <alice@example.com>").unwrap();
        assert_eq!(named.name.as_deref(), Some("Alice Example"));
        assert_eq!(named.address, "alice@example.com");

        // Some clients omit the space before the bracket.
        let tight = parse("Alice Example<alice@example.com>").unwrap();
        assert_eq!(tight.name.as_deref(), Some("Alice Example"));
    }

    #[test]
    fn a_quoted_name_loses_its_quoting() {
        let address = parse(r#""Example, Alice" <alice@example.com>"#).unwrap();

        assert_eq!(address.name.as_deref(), Some("Example, Alice"));
        assert_eq!(address.address, "alice@example.com");
    }

    #[test]
    fn an_escaped_quote_survives_the_unquoting() {
        let address =
            parse(r#""Alice \"Ada\" Example" <alice@example.com>"#).unwrap();

        assert_eq!(address.name.as_deref(), Some(r#"Alice "Ada" Example"#));
    }

    #[test]
    fn the_comment_form_carries_the_name() {
        let address = parse("alice@example.com (Alice Example)").unwrap();

        assert_eq!(address.name.as_deref(), Some("Alice Example"));
        assert_eq!(address.address, "alice@example.com");
    }

    #[test]
    fn the_address_is_lowercased_and_the_name_is_not() {
        let address =
            parse("Alice Example <Alice.Example@EXAMPLE.COM>").unwrap();

        assert_eq!(address.address, "alice.example@example.com");
        assert_eq!(address.name.as_deref(), Some("Alice Example"));
    }

    #[test]
    fn folded_header_whitespace_collapses_in_the_name() {
        let address = parse("Alice\n   Example <alice@example.com>").unwrap();

        assert_eq!(address.name.as_deref(), Some("Alice Example"));
    }

    #[test]
    fn text_that_is_not_an_address_is_rejected() {
        assert!(parse("").is_none());
        assert!(parse("   ").is_none());
        assert!(parse("Alice Example").is_none());
        assert!(parse("@example.com").is_none());
        assert!(parse("alice@").is_none());
        assert!(parse("alice example@x.com").is_none());
    }

    #[test]
    fn a_list_splits_on_top_level_commas_only() {
        let list = parse_list(
            r#"Alice <alice@example.com>, "Example, Bob" <bob@example.com>, carol@example.com"#,
        );

        assert_eq!(list.len(), 3);
        assert_eq!(list[1].name.as_deref(), Some("Example, Bob"));
        assert_eq!(list[2].address, "carol@example.com");
    }

    #[test]
    fn a_group_contributes_its_members_and_not_its_label() {
        let list = parse_list("Team: alice@example.com, bob@example.com;");

        assert_eq!(list.len(), 2);
        assert_eq!(list[0].address, "alice@example.com");
        assert_eq!(list[1].address, "bob@example.com");
    }

    #[test]
    fn one_bad_entry_does_not_cost_the_others() {
        let list =
            parse_list("alice@example.com, not an address, bob@example.com");

        assert_eq!(list.len(), 2);
    }

    #[test]
    fn folding_erases_case_and_accents() {
        assert_eq!(fold("Cainã Costa"), "caina costa");
        assert_eq!(fold("Zoë"), "zoe");
        assert_eq!(fold("ALICE"), "alice");
    }

    #[test]
    fn words_break_on_whitespace_and_local_part_separators() {
        assert_eq!(words("Alice Example"), vec!["alice", "example"]);
        assert_eq!(words("bob.smith"), vec!["bob", "smith"]);
        assert_eq!(words("me+mailbert"), vec!["me", "mailbert"]);
        assert_eq!(words("no-reply"), vec!["no", "reply"]);
        assert!(words("   ").is_empty());
    }

    #[test]
    fn address_words_join_the_name_and_the_local_part() {
        let address = parse("Cainã Costa <me+mailbert@cfcosta.com>").unwrap();

        let found = address.words();
        for want in ["caina", "costa", "me", "mailbert"] {
            assert!(
                found.iter().any(|w| w == want),
                "{want} missing from {found:?}"
            );
        }
        // The domain is not a word, or `from:com` would match everyone.
        assert!(!found.iter().any(|w| w == "cfcosta" || w == "com"));
    }

    #[test]
    fn display_quotes_a_name_only_when_it_has_to() {
        let plain = Address {
            name: Some("Alice Example".to_string()),
            address: "alice@example.com".to_string(),
        };
        assert_eq!(plain.to_string(), "Alice Example <alice@example.com>");

        let comma = Address {
            name: Some("Example, Alice".to_string()),
            address: "alice@example.com".to_string(),
        };
        assert_eq!(
            comma.to_string(),
            r#""Example, Alice" <alice@example.com>"#
        );

        let bare = Address {
            name: None,
            address: "alice@example.com".to_string(),
        };
        assert_eq!(bare.to_string(), "alice@example.com");
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 200)]
    fn prop_display_round_trips(tc: TestCase) {
        let address = tc.draw(an_address());

        let text = address.to_string();
        let parsed = parse(&text).expect("mailbert can read what it writes");

        assert_eq!(parsed, address, "round trip of {text:?}");
    }

    #[hegel::test(test_cases = 200)]
    fn prop_addresses_are_always_normalized(tc: TestCase) {
        let address = tc.draw(an_address());

        let parsed = parse(&address.to_string()).unwrap();

        assert_eq!(parsed.address, parsed.address.to_lowercase());
        assert_eq!(parsed.address.matches('@').count(), 1);
        assert!(!parsed.address.chars().any(char::is_whitespace));
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_list_keeps_every_address(tc: TestCase) {
        let addresses: Vec<Address> =
            tc.draw(gs::vecs(an_address()).min_size(1).max_size(6));

        let text = addresses
            .iter()
            .map(Address::to_string)
            .collect::<Vec<_>>()
            .join(", ");

        assert_eq!(parse_list(&text), addresses, "list {text:?}");
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_comma_in_a_name_never_splits(tc: TestCase) {
        let address: String = tc.draw(address_text());
        let name: String = tc.draw(gs::sampled_from(vec![
            "Example, Alice".to_string(),
            "Costa, Cainã".to_string(),
            "Smith, Bob, Jr.".to_string(),
        ]));

        let text = format!(r#""{name}" <{address}>"#);
        let list = parse_list(&text);

        assert_eq!(list.len(), 1, "{text:?} split into {list:?}");
        assert_eq!(list[0].name.as_deref(), Some(name.as_str()));
    }

    #[hegel::test(test_cases = 200)]
    fn prop_folding_is_idempotent(tc: TestCase) {
        let text: String = tc.draw(gs::text().min_size(0).max_size(40));

        assert_eq!(fold(&fold(&text)), fold(&text), "{text:?}");
    }

    #[hegel::test(test_cases = 200)]
    fn prop_folding_erases_case_and_accents(tc: TestCase) {
        let (accented, plain): (String, String) =
            tc.draw(gs::sampled_from(vec![
                ("Cainã".to_string(), "Caina".to_string()),
                ("Zoë".to_string(), "Zoe".to_string()),
                ("José".to_string(), "Jose".to_string()),
                ("Ångström".to_string(), "Angstrom".to_string()),
            ]));

        assert_eq!(fold(&accented), fold(&plain));
        assert_eq!(fold(&accented), fold(&accented).to_lowercase());
    }

    #[hegel::test(test_cases = 200)]
    fn prop_words_are_never_blank(tc: TestCase) {
        let text: String = tc.draw(gs::text().min_size(0).max_size(40));

        for word in words(&text) {
            assert!(!word.is_empty(), "blank word from {text:?}");
            assert!(!word.chars().any(char::is_whitespace));
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_an_address_needs_an_at_sign(tc: TestCase) {
        let text: String = tc.draw(gs::text().min_size(0).max_size(40));
        tc.assume(!text.contains('@'));

        assert!(parse(&text).is_none(), "{text:?} parsed as an address");
    }
}
