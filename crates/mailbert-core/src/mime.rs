//! Turn a raw message into the parts that mailbert indexes.
//!
//! The pipeline has three jobs. It selects one body from the parts of a
//! MIME message (§5.1). It records the name of each attachment, and not
//! the bytes (§5.3). It finds encrypted mail, and keeps the ciphertext
//! out of the index (§5.4).
//!
//! The last job is a security rule, and not an optimization. The index
//! is a plaintext file, and so is every backup of it. A message that
//! the sender encrypted must stay encrypted at rest, so this module
//! never decrypts. `mailbert view` calls gpg on demand instead.

use mail_parser::{
    HeaderValue,
    Message as MailMessage,
    MessageParser,
    MimeHeaders,
};
use regex::Regex;
use thiserror::Error;

use crate::{
    address::Address,
    body,
    message_id::{self, MessageId},
};

/// The column that the HTML fallback wraps to.
pub const HTML_WIDTH: usize = 100;

/// The first line of an inline PGP message. (RFC 4880)
const PGP_ARMOR: &str = "-----BEGIN PGP MESSAGE-----";

/// The subtypes of `application` that hold S/MIME ciphertext.
const SMIME_SUBTYPES: [&str; 2] = ["pkcs7-mime", "x-pkcs7-mime"];

/// The values of `Precedence` that mark mail as bulk.
const BULK_PRECEDENCE: [&str; 3] = ["bulk", "list", "junk"];

/// Where the indexed text of a message came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// A `text/plain` part.
    Plain,

    /// A `text/html` part, which the pipeline made into text.
    Html,

    /// The message is encrypted, so there is no text to index.
    Encrypted,

    /// The message carries no body that the pipeline can read.
    Empty,
}

/// One attachment. mailbert indexes the name, and not the bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attachment {
    /// The filename, when the message gave one.
    pub name: Option<String>,

    /// The media type, such as `application/pdf`.
    pub content_type: String,

    /// The size in bytes, after the transfer encoding is removed.
    pub size: usize,
}

/// A message, after the pipeline reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parsed {
    /// The `Message-ID`, normalized as in §4.1.
    pub message_id: Option<String>,

    /// The `In-Reply-To`, normalized.
    pub in_reply_to: Option<String>,

    /// The `References`, normalized, in the order of the header.
    pub references: Vec<String>,

    /// Seconds since the Unix epoch, when the message gave a `Date`.
    pub date: Option<i64>,

    /// The `From` addresses.
    pub from: Vec<Address>,

    /// The `To` addresses.
    pub to: Vec<Address>,

    /// The `Cc` addresses.
    pub cc: Vec<Address>,

    /// The subject, as the reader sees it.
    pub subject: String,

    /// The `List-Id`, reduced to its identifier.
    pub list_id: Option<String>,

    /// Which part gave the text.
    pub source: Source,

    /// The selected body, before the quotes go.
    pub full: String,

    /// The text to index, after §5.2 removes the quotes.
    pub text: String,

    /// True when the body held no original text.
    pub quote_only: bool,

    /// True when the message is mail to a list, or is automatic.
    pub is_bulk: bool,

    /// One entry for each attachment.
    pub attachments: Vec<Attachment>,
}

/// What can go wrong when the pipeline reads a message.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MimeError {
    #[error("this is not a message that I can read")]
    NotAMessage,
}

/// Read a raw message.
pub fn parse(raw: &[u8]) -> Result<Parsed, MimeError> {
    parse_with_footers(raw, &[])
}

/// Read a raw message, and remove the footers of the account.
pub fn parse_with_footers(
    raw: &[u8],
    footers: &[Regex],
) -> Result<Parsed, MimeError> {
    if raw.iter().all(u8::is_ascii_whitespace) {
        return Err(MimeError::NotAMessage);
    }

    let message = MessageParser::default()
        .parse(raw)
        .ok_or(MimeError::NotAMessage)?;

    let list_id = header_text(&message, "List-Id").and_then(list_identifier);

    // The ciphertext is chosen before the body is, because the choice
    // is what keeps the ciphertext out of `full` and out of the index.
    let (source, full) = match is_encrypted(&message) {
        true => (Source::Encrypted, String::new()),
        false => select_body(&message),
    };

    let stripped = body::strip_with_footers(&full, footers);

    Ok(Parsed {
        message_id: message
            .message_id()
            .and_then(message_id::normalize_message_id),
        in_reply_to: identifiers(message.in_reply_to()).pop(),
        references: identifiers(message.references()),
        date: message.date().map(|date| date.to_timestamp()),
        from: addresses(message.from()),
        to: addresses(message.to()),
        cc: addresses(message.cc()),
        subject: message.subject().unwrap_or_default().trim().to_string(),
        is_bulk: is_bulk(&message, list_id.is_some()),
        list_id,
        source,
        full,
        text: stripped.text,
        quote_only: stripped.quote_only,
        attachments: attachments(&message),
    })
}

/// Make text out of HTML.
///
/// A tag never reaches the index, and the text of the page does.
pub fn html_to_text(html: &str) -> String {
    html2text::from_read(html.as_bytes(), HTML_WIDTH).unwrap_or_default()
}

impl Parsed {
    /// Whether the message is encrypted.
    pub fn is_encrypted(&self) -> bool {
        self.source == Source::Encrypted
    }

    /// The identity of the message, as in §4.1.
    pub fn identity(&self) -> MessageId {
        let from = match self.from.first() {
            Some(address) => address.address.as_str(),
            None => "",
        };

        MessageId::derive(
            self.message_id.as_deref(),
            self.date.unwrap_or(0),
            from,
            &self.subject,
            &self.text,
        )
    }

    /// The addresses that received the message, To before Cc.
    pub fn recipients(&self) -> impl Iterator<Item = &Address> {
        self.to.iter().chain(self.cc.iter())
    }

    /// The names of the attachments that carry one.
    pub fn attachment_names(&self) -> impl Iterator<Item = &str> {
        self.attachments
            .iter()
            .filter_map(|attachment| attachment.name.as_deref())
    }
}

/// Reduce a `List-Id` header to its identifier.
///
/// `Rust Users <users.rust-lang.org>` becomes `users.rust-lang.org`.
fn list_identifier(raw: String) -> Option<String> {
    let inside = match (raw.rfind('<'), raw.rfind('>')) {
        (Some(open), Some(close)) if open < close => &raw[open + 1..close],
        _ => raw.as_str(),
    };

    let found = inside.trim().to_lowercase();
    if found.is_empty() {
        return None;
    }

    Some(found)
}

/// Whether the message carries ciphertext instead of a body.
///
/// A signed message is not an encrypted one. Its text is readable, and
/// the index must hold it.
fn is_encrypted(message: &MailMessage<'_>) -> bool {
    if let Some(content_type) = message.content_type() {
        let kind = content_type.ctype();
        let subtype = content_type.subtype().unwrap_or_default();

        // PGP/MIME, as RFC 3156 defines it.
        if kind.eq_ignore_ascii_case("multipart")
            && subtype.eq_ignore_ascii_case("encrypted")
        {
            return true;
        }

        // S/MIME.
        if kind.eq_ignore_ascii_case("application")
            && SMIME_SUBTYPES
                .iter()
                .any(|known| subtype.eq_ignore_ascii_case(known))
        {
            return true;
        }
    }

    // Inline PGP, which carries no content type of its own. The armor
    // must open the body, so that a message about PGP stays readable.
    message
        .body_text(0)
        .and_then(|text| {
            text.lines()
                .map(str::trim)
                .find(|line| !line.is_empty())
                .map(|line| line == PGP_ARMOR)
        })
        .unwrap_or(false)
}

/// Choose the one part that the index reads. See §5.1.
fn select_body(message: &MailMessage<'_>) -> (Source, String) {
    let plain = message
        .text_part(0)
        .filter(|part| !part.is_text_html())
        .and_then(|part| part.text_contents())
        .filter(|text| !text.trim().is_empty());

    if let Some(text) = plain {
        return (Source::Plain, text.to_string());
    }

    let html = message
        .html_part(0)
        .and_then(|part| part.text_contents())
        .map(html_to_text)
        .filter(|text| !text.trim().is_empty());

    match html {
        Some(text) => (Source::Html, text),
        None => (Source::Empty, String::new()),
    }
}

/// The name, the type, and the size of each attachment. See §5.3.
fn attachments(message: &MailMessage<'_>) -> Vec<Attachment> {
    message
        .attachments()
        .map(|part| Attachment {
            name: part.attachment_name().map(str::to_string),
            content_type: media_type(part.content_type()),
            size: part.len(),
        })
        .collect()
}

/// The media type of a part, lowercased, with a safe default.
fn media_type(content_type: Option<&mail_parser::ContentType<'_>>) -> String {
    let Some(content_type) = content_type else {
        return "application/octet-stream".to_string();
    };

    match content_type.subtype() {
        Some(subtype) => format!(
            "{}/{}",
            content_type.ctype().to_lowercase(),
            subtype.to_lowercase()
        ),
        None => content_type.ctype().to_lowercase(),
    }
}

/// Whether the message is mail to a list, or is automatic.
fn is_bulk(message: &MailMessage<'_>, has_list_id: bool) -> bool {
    if has_list_id
        || !message.list_unsubscribe().is_empty()
        || !message.list_post().is_empty()
    {
        return true;
    }

    if header_text(message, "Precedence").is_some_and(|value| {
        BULK_PRECEDENCE
            .iter()
            .any(|known| value.eq_ignore_ascii_case(known))
    }) {
        return true;
    }

    // RFC 3834 says that `no` is the value of ordinary mail.
    header_text(message, "Auto-Submitted")
        .is_some_and(|value| !value.eq_ignore_ascii_case("no"))
}

/// A raw header, unfolded, with its whitespace collapsed.
fn header_text(message: &MailMessage<'_>, name: &str) -> Option<String> {
    let raw = message.header_raw(name)?;
    let found = raw.split_whitespace().collect::<Vec<&str>>().join(" ");

    match found.is_empty() {
        true => None,
        false => Some(found),
    }
}

/// The normalized identifiers of a `References` or `In-Reply-To`.
fn identifiers(value: &HeaderValue<'_>) -> Vec<String> {
    match value {
        HeaderValue::Text(one) => {
            message_id::normalize_message_id(one).into_iter().collect()
        }
        HeaderValue::TextList(many) => many
            .iter()
            .filter_map(|one| message_id::normalize_message_id(one))
            .collect(),
        _ => Vec::new(),
    }
}

/// The addresses of one header, with the groups made flat.
fn addresses(header: Option<&mail_parser::Address<'_>>) -> Vec<Address> {
    let Some(header) = header else {
        return Vec::new();
    };

    header
        .iter()
        .filter_map(|found| Address::new(found.name(), found.address()?))
        .collect()
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_parse_never_panics` | invariant | A hostile message must not stop a sync. mailbert reads mail that strangers wrote. |
    //! | `prop_ciphertext_never_reaches_the_text` | invariant | This is §5.4. The index is a plaintext file, so ciphertext in it defeats the encryption. |
    //! | `prop_attachment_bytes_never_reach_the_text` | invariant | §5.3 indexes the names. A 4 MB PDF in the body field wastes the index and pollutes the ranking. |
    //! | `prop_the_identity_is_stable` | algebraic | The same bytes must give the same identity, or a re-sync loses every tag. |
    //! | `prop_the_message_id_decides_the_identity` | model-based | §4.1. Two copies of one message differ in their bytes and must still be one entry. |
    //! | `prop_addresses_are_lowercased` | invariant | `from:BOB@x` and `from:bob@x` are one person. |
    //! | `prop_the_subject_survives_the_round_trip` | round-trip | The subject carries a 2x boost in the index, so losing it costs the most. |
    //! | `prop_html_holds_no_tag` | invariant | A tag in the index matches queries that no reader means. |

    use hegel::{TestCase, generators as gs};

    use super::*;

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    /// Make the line endings of a fixture into the ones that mail uses.
    fn raw(text: &str) -> Vec<u8> {
        text.replace('\n', "\r\n").into_bytes()
    }

    fn read(text: &str) -> Parsed {
        parse(&raw(text)).expect("a message that parses")
    }

    /// A plain message, with the headers that a fixture always needs.
    fn plain(body: &str) -> String {
        format!(
            "From: Alice Smith <alice@example.test>\n\
             To: bob@example.test\n\
             Subject: Deposit for the apartment\n\
             Date: Fri, 14 Aug 2026 09:30:00 +0000\n\
             Message-ID: <deposit-1@example.test>\n\
             \n\
             {body}"
        )
    }

    const BOUNDARY: &str = "----boundary";

    /// A `multipart/alternative` message of a plain part and an HTML one.
    fn alternative(plain_part: &str, html_part: &str) -> String {
        format!(
            "From: alice@example.test\n\
             Subject: Two parts\n\
             Message-ID: <two@example.test>\n\
             Content-Type: multipart/alternative; boundary=\"{BOUNDARY}\"\n\
             \n\
             --{BOUNDARY}\n\
             Content-Type: text/plain; charset=utf-8\n\
             \n\
             {plain_part}\n\
             --{BOUNDARY}\n\
             Content-Type: text/html; charset=utf-8\n\
             \n\
             {html_part}\n\
             --{BOUNDARY}--\n"
        )
    }

    // -----------------------------------------------------------------
    // Headers.
    // -----------------------------------------------------------------

    #[test]
    fn reads_the_headers_of_a_plain_message() {
        let parsed = read(&plain("The deposit is due on Friday.\n"));

        assert_eq!(parsed.subject, "Deposit for the apartment");
        assert_eq!(parsed.from.len(), 1);
        assert_eq!(parsed.from[0].address, "alice@example.test");
        assert_eq!(parsed.from[0].name.as_deref(), Some("Alice Smith"));
        assert_eq!(parsed.to.len(), 1);
        assert_eq!(parsed.to[0].address, "bob@example.test");
        assert_eq!(parsed.source, Source::Plain);
    }

    #[test]
    fn reads_the_date_as_seconds_since_the_epoch() {
        let parsed = read(&plain("Body.\n"));

        // 2026-08-14T09:30:00Z
        assert_eq!(parsed.date, Some(1_786_699_800));
    }

    #[test]
    fn a_message_with_no_date_has_no_date() {
        let parsed = read("Subject: No date\n\nBody.\n");

        assert_eq!(parsed.date, None);
    }

    #[test]
    fn normalizes_the_message_id() {
        let parsed =
            read("Subject: Case\nMessage-ID: <Abc@Example.COM>\n\nBody.\n");

        assert_eq!(parsed.message_id.as_deref(), Some("Abc@example.com"));
    }

    #[test]
    fn reads_the_reply_chain() {
        let parsed = read(
            "Subject: Re: Deposit\n\
             Message-ID: <c@x.test>\n\
             In-Reply-To: <b@x.test>\n\
             References: <a@x.test> <b@x.test>\n\
             \n\
             Body.\n",
        );

        assert_eq!(parsed.in_reply_to.as_deref(), Some("b@x.test"));
        assert_eq!(parsed.references, vec!["a@x.test", "b@x.test"]);
    }

    #[test]
    fn keeps_cc_apart_from_to() {
        let parsed = read(
            "Subject: Copies\n\
             To: bob@x.test\n\
             Cc: carol@x.test, dave@x.test\n\
             \n\
             Body.\n",
        );

        assert_eq!(parsed.to.len(), 1);
        assert_eq!(parsed.cc.len(), 2);
        assert_eq!(parsed.recipients().count(), 3);
    }

    #[test]
    fn reads_a_folded_header() {
        let parsed = read(
            "Subject: A subject that the sender\n \
             wrote over two lines\n\
             \n\
             Body.\n",
        );

        assert!(parsed.subject.contains("over two lines"));
    }

    #[test]
    fn decodes_an_encoded_word_subject() {
        let parsed = read("Subject: =?utf-8?B?Q2FpbsOjIENvc3Rh?=\n\nBody.\n");

        assert_eq!(parsed.subject, "Cainã Costa");
    }

    #[test]
    fn an_empty_input_is_not_a_message() {
        assert_eq!(parse(b""), Err(MimeError::NotAMessage));
    }

    // -----------------------------------------------------------------
    // Bodies.
    // -----------------------------------------------------------------

    #[test]
    fn removes_the_quotes_from_the_body() {
        let parsed = read(&plain(
            "Yes, Friday works.\n\
             \n\
             On Wed, Alice wrote:\n\
             > Is Friday fine?\n",
        ));

        assert_eq!(parsed.text.trim(), "Yes, Friday works.");
        assert!(parsed.full.contains("Is Friday fine?"));
        assert!(!parsed.quote_only);
    }

    #[test]
    fn decodes_a_quoted_printable_body() {
        let parsed = read(
            "Subject: Encoded\n\
             Content-Type: text/plain; charset=utf-8\n\
             Content-Transfer-Encoding: quoted-printable\n\
             \n\
             The caf=C3=A9 is open.\n",
        );

        assert!(parsed.text.contains("café"));
    }

    #[test]
    fn decodes_a_base64_body() {
        let parsed = read(
            "Subject: Encoded\n\
             Content-Type: text/plain; charset=utf-8\n\
             Content-Transfer-Encoding: base64\n\
             \n\
             VGhlIHJlbnQgaXMgcGFpZC4=\n",
        );

        assert!(parsed.text.contains("The rent is paid."));
    }

    #[test]
    fn prefers_the_plain_part_over_the_html_one() {
        let parsed = read(&alternative(
            "The plain words.",
            "<html><body><p>The HTML words.</p></body></html>",
        ));

        assert_eq!(parsed.source, Source::Plain);
        assert!(parsed.text.contains("The plain words."));
        assert!(!parsed.text.contains("The HTML words."));
    }

    #[test]
    fn falls_back_to_the_html_part() {
        let parsed = read(
            "Subject: Only HTML\n\
             Content-Type: text/html; charset=utf-8\n\
             \n\
             <html><body><h1>Your receipt</h1>\
             <p>The total is <b>12.50</b>.</p></body></html>\n",
        );

        assert_eq!(parsed.source, Source::Html);
        assert!(parsed.text.contains("Your receipt"));
        assert!(parsed.text.contains("12.50"));
        assert!(!parsed.text.contains("<p>"));
        assert!(!parsed.text.contains("body"));
    }

    #[test]
    fn a_message_with_no_body_is_empty() {
        let parsed = read("Subject: Nothing\n\n");

        assert_eq!(parsed.source, Source::Empty);
        assert_eq!(parsed.text, "");
    }

    #[test]
    fn removes_the_footer_of_the_account() {
        let footers =
            vec![Regex::new(r"(?m)^Sent from my phone$").expect("a pattern")];
        let message = raw(&plain("The answer is yes.\n\nSent from my phone\n"));

        let parsed = parse_with_footers(&message, &footers).expect("a message");

        assert!(parsed.text.contains("The answer is yes."));
        assert!(!parsed.text.contains("Sent from my phone"));
    }

    #[test]
    fn html_to_text_drops_the_tags() {
        let text = html_to_text("<p>Hello <b>world</b></p>");

        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
        assert!(!text.contains('<'));
    }

    // -----------------------------------------------------------------
    // Attachments.
    // -----------------------------------------------------------------

    /// A message of one text part and one attachment.
    fn with_attachment(disposition: &str, payload: &str) -> String {
        format!(
            "From: alice@example.test\n\
             Subject: The invoice\n\
             Message-ID: <inv@example.test>\n\
             Content-Type: multipart/mixed; boundary=\"{BOUNDARY}\"\n\
             \n\
             --{BOUNDARY}\n\
             Content-Type: text/plain\n\
             \n\
             The invoice is attached.\n\
             --{BOUNDARY}\n\
             Content-Type: application/pdf\n\
             {disposition}\n\
             \n\
             {payload}\n\
             --{BOUNDARY}--\n"
        )
    }

    #[test]
    fn records_the_name_and_the_type_of_an_attachment() {
        let parsed = read(&with_attachment(
            "Content-Disposition: attachment; filename=\"invoice.pdf\"",
            "PDFBYTES",
        ));

        assert_eq!(parsed.attachments.len(), 1);
        assert_eq!(parsed.attachments[0].name.as_deref(), Some("invoice.pdf"));
        assert_eq!(parsed.attachments[0].content_type, "application/pdf");
        assert_eq!(
            parsed.attachment_names().collect::<Vec<&str>>(),
            vec!["invoice.pdf"]
        );
    }

    #[test]
    fn an_attachment_with_no_filename_has_no_name() {
        let parsed = read(&with_attachment(
            "Content-Disposition: attachment",
            "PDFBYTES",
        ));

        assert_eq!(parsed.attachments.len(), 1);
        assert_eq!(parsed.attachments[0].name, None);
        assert_eq!(parsed.attachment_names().count(), 0);
    }

    #[test]
    fn the_bytes_of_an_attachment_stay_out_of_the_text() {
        let parsed = read(&with_attachment(
            "Content-Disposition: attachment; filename=\"invoice.pdf\"",
            "SECRETPAYLOADBYTES",
        ));

        assert!(parsed.text.contains("The invoice is attached."));
        assert!(!parsed.text.contains("SECRETPAYLOADBYTES"));
        assert!(!parsed.full.contains("SECRETPAYLOADBYTES"));
    }

    // -----------------------------------------------------------------
    // Encrypted mail. See §5.4.
    // -----------------------------------------------------------------

    /// A PGP/MIME message, as RFC 3156 builds one.
    fn pgp_mime(ciphertext: &str) -> String {
        format!(
            "From: alice@example.test\n\
             Subject: Private\n\
             Message-ID: <secret@example.test>\n\
             Content-Type: multipart/encrypted; \
             protocol=\"application/pgp-encrypted\"; \
             boundary=\"{BOUNDARY}\"\n\
             \n\
             --{BOUNDARY}\n\
             Content-Type: application/pgp-encrypted\n\
             \n\
             Version: 1\n\
             --{BOUNDARY}\n\
             Content-Type: application/octet-stream\n\
             \n\
             {PGP_ARMOR}\n\
             {ciphertext}\n\
             -----END PGP MESSAGE-----\n\
             --{BOUNDARY}--\n"
        )
    }

    #[test]
    fn marks_a_pgp_mime_message_as_encrypted() {
        let parsed = read(&pgp_mime("hQIMA0abcdef"));

        assert!(parsed.is_encrypted());
        assert_eq!(parsed.source, Source::Encrypted);
    }

    #[test]
    fn marks_an_inline_pgp_message_as_encrypted() {
        let parsed = read(&plain(&format!(
            "{PGP_ARMOR}\n\nhQIMA0abcdef\n-----END PGP MESSAGE-----\n"
        )));

        assert!(parsed.is_encrypted());
    }

    #[test]
    fn marks_an_smime_message_as_encrypted() {
        let parsed = read(
            "Subject: Private\n\
             Content-Type: application/pkcs7-mime; smime-type=enveloped-data\n\
             Content-Transfer-Encoding: base64\n\
             \n\
             MIAGCSqGSIb3DQEHA6CA\n",
        );

        assert!(parsed.is_encrypted());
    }

    #[test]
    fn a_signed_message_is_not_encrypted() {
        let message = format!(
            "From: alice@example.test\n\
             Subject: Signed, and readable\n\
             Content-Type: multipart/signed; \
             protocol=\"application/pgp-signature\"; \
             boundary=\"{BOUNDARY}\"\n\
             \n\
             --{BOUNDARY}\n\
             Content-Type: text/plain\n\
             \n\
             The meeting moved to Tuesday.\n\
             --{BOUNDARY}\n\
             Content-Type: application/pgp-signature\n\
             \n\
             -----BEGIN PGP SIGNATURE-----\n\
             abc\n\
             -----END PGP SIGNATURE-----\n\
             --{BOUNDARY}--\n"
        );

        let parsed = read(&message);

        assert!(!parsed.is_encrypted());
        assert!(parsed.text.contains("The meeting moved to Tuesday."));
    }

    #[test]
    fn an_encrypted_message_indexes_no_text() {
        let parsed = read(&pgp_mime("hQIMA0SECRETCIPHERTEXT"));

        assert_eq!(parsed.text, "");
        assert_eq!(parsed.full, "");
        assert!(!parsed.text.contains("SECRETCIPHERTEXT"));
    }

    #[test]
    fn an_encrypted_message_keeps_its_headers() {
        let parsed = read(&pgp_mime("hQIMA0abcdef"));

        assert_eq!(parsed.subject, "Private");
        assert_eq!(parsed.from[0].address, "alice@example.test");
        assert_eq!(parsed.message_id.as_deref(), Some("secret@example.test"));
    }

    // -----------------------------------------------------------------
    // Lists and bulk mail.
    // -----------------------------------------------------------------

    #[test]
    fn reads_the_list_identifier() {
        let parsed = read(
            "Subject: A thread\n\
             List-Id: Rust Users <users.rust-lang.org>\n\
             \n\
             Body.\n",
        );

        assert_eq!(parsed.list_id.as_deref(), Some("users.rust-lang.org"));
        assert!(parsed.is_bulk);
    }

    #[test]
    fn a_list_id_with_no_brackets_is_the_whole_value() {
        let parsed =
            read("Subject: A thread\nList-Id: users.rust-lang.org\n\nBody.\n");

        assert_eq!(parsed.list_id.as_deref(), Some("users.rust-lang.org"));
    }

    #[test]
    fn precedence_bulk_marks_bulk_mail() {
        let parsed =
            read("Subject: Sale\nPrecedence: bulk\n\nBuy something.\n");

        assert!(parsed.is_bulk);
        assert_eq!(parsed.list_id, None);
    }

    #[test]
    fn an_unsubscribe_header_marks_bulk_mail() {
        let parsed = read(
            "Subject: Sale\n\
             List-Unsubscribe: <https://x.test/u>\n\
             \n\
             Buy something.\n",
        );

        assert!(parsed.is_bulk);
    }

    #[test]
    fn ordinary_mail_is_not_bulk() {
        let parsed = read(&plain("Are you free on Friday?\n"));

        assert!(!parsed.is_bulk);
    }

    // -----------------------------------------------------------------
    // Identity. See §4.1.
    // -----------------------------------------------------------------

    #[test]
    fn the_identity_prefers_the_message_id() {
        let parsed = read(&plain("Body.\n"));
        let expected = MessageId::from_message_id("<deposit-1@example.test>")
            .expect("a valid Message-ID");

        assert_eq!(parsed.identity(), expected);
    }

    #[test]
    fn the_identity_falls_back_to_the_content() {
        let one = read(
            "Subject: No id\nDate: Fri, 14 Aug 2026 09:30:00 \
                        +0000\n\nSame body.\n",
        );
        let two = read(
            "Subject: No id\nDate: Fri, 14 Aug 2026 09:30:00 \
                        +0000\n\nSame body.\n",
        );
        let other = read(
            "Subject: No id\nDate: Fri, 14 Aug 2026 09:30:00 \
                          +0000\n\nOther body.\n",
        );

        assert_eq!(one.identity(), two.identity());
        assert_ne!(one.identity(), other.identity());
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    /// Text that a header or a body can hold. Every value is printable
    /// and on one line, because a header that folds is a different test.
    fn phrase() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "The deposit is due on Friday".to_string(),
            "Can you look at the invoice?".to_string(),
            "Cainã sent the report".to_string(),
            "Move-out inspection".to_string(),
            "quarterly numbers 2026".to_string(),
            "Re: the meeting".to_string(),
        ])
    }

    /// A payload that must never appear in the indexed text.
    fn secret() -> impl gs::Generator<String> {
        gs::sampled_from(vec![
            "hQIMA0SECRET".to_string(),
            "wcBMA1PAYLOAD".to_string(),
            "jA0ECQMCzzTOPSECRET".to_string(),
            "hF4Dxyz9abcdef".to_string(),
        ])
    }

    #[hegel::test(test_cases = 400)]
    fn prop_parse_never_panics(tc: TestCase) {
        let bytes = tc.draw(
            gs::vecs(gs::integers::<u8>().min_value(0).max_value(255))
                .min_size(0)
                .max_size(400),
        );

        // The result may be an error. It must never be a panic.
        let _ = parse(&bytes);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_parse_never_panics_on_text(tc: TestCase) {
        let text = tc.draw(gs::text().min_size(0).max_size(300));

        let _ = parse(text.as_bytes());
    }

    #[hegel::test(test_cases = 200)]
    fn prop_ciphertext_never_reaches_the_text(tc: TestCase) {
        let hidden = tc.draw(secret());

        let parsed = parse(&raw(&pgp_mime(&hidden))).expect("a message");

        assert!(parsed.is_encrypted());
        assert!(!parsed.text.contains(&hidden));
        assert!(!parsed.full.contains(&hidden));
    }

    #[hegel::test(test_cases = 200)]
    fn prop_attachment_bytes_never_reach_the_text(tc: TestCase) {
        let payload = tc.draw(secret());

        let message = with_attachment(
            "Content-Disposition: attachment; filename=\"f.bin\"",
            &payload,
        );
        let parsed = parse(&raw(&message)).expect("a message");

        assert!(!parsed.text.contains(&payload));
    }

    #[hegel::test(test_cases = 200)]
    fn prop_the_identity_is_stable(tc: TestCase) {
        let body = tc.draw(phrase());
        let message = raw(&plain(&body));

        let one = parse(&message).expect("a message");
        let two = parse(&message).expect("a message");

        assert_eq!(one.identity(), two.identity());
    }

    #[hegel::test(test_cases = 200)]
    fn prop_the_message_id_decides_the_identity(tc: TestCase) {
        let one = tc.draw(phrase());
        let two = tc.draw(phrase());

        let first = read(&plain(&one)).identity();
        let second = read(&plain(&two)).identity();

        // `plain` gives both bodies the same `Message-ID`, so the two
        // are one message however far the bodies drift apart.
        assert_eq!(first, second);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_addresses_are_lowercased(tc: TestCase) {
        let local = tc.draw(gs::sampled_from(vec![
            "Alice",
            "BOB",
            "Carol.Smith",
            "DAVE",
        ]));
        let domain =
            tc.draw(gs::sampled_from(vec!["Example.Test", "X.TEST", "y.test"]));

        let parsed =
            read(&format!("Subject: Case\nFrom: {local}@{domain}\n\nBody.\n"));

        let found = &parsed.from[0].address;

        assert_eq!(*found, found.to_lowercase());
        assert_eq!(*found, format!("{local}@{domain}").to_lowercase());
    }

    #[hegel::test(test_cases = 200)]
    fn prop_the_subject_survives_the_round_trip(tc: TestCase) {
        let subject = tc.draw(phrase());

        let parsed =
            read(&format!("Subject: {subject}\nFrom: a@x.test\n\nBody.\n"));

        assert_eq!(parsed.subject, subject);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_html_holds_no_tag(tc: TestCase) {
        let word = tc.draw(gs::sampled_from(vec![
            "receipt", "invoice", "deposit", "meeting",
        ]));
        let tag =
            tc.draw(gs::sampled_from(vec!["p", "b", "i", "span", "div", "h1"]));

        let text = html_to_text(&format!("<{tag}>{word}</{tag}>"));

        assert!(text.contains(word));
        assert!(!text.contains('<'));
        assert!(!text.contains('>'));
    }
}
