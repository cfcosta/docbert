//! Open an encrypted body through gpg-agent. (§5.4)
//!
//! mailbert never holds a secret key. Every secret stays where GnuPG
//! put it, and the agent does the one operation that needs it: it
//! unwraps the session key of the message. Sequoia does the rest in
//! this process, so no `gpg` runs and nothing of the plaintext reaches
//! the store, the index, or a temporary file.
//!
//! The agent addresses a key by its *keygrip*, and a message names its
//! recipient by *key ID*. Nothing in the agent maps one to the other,
//! so the public certificates make the map. [`certs`] reads them from
//! the GnuPG home, which holds them in one of two shapes: the keybox
//! that GnuPG 2.1 introduced, or the keyring that came before it.

use std::path::{Path, PathBuf};

use mailbert_core::{
    config::PgpConfig,
    mime::{Ciphertext, ciphertext},
};
use sequoia_gpg_agent::{Agent, gnupg};
use sequoia_openpgp::{
    Cert,
    KeyHandle,
    crypto::SessionKey,
    packet::{
        Key,
        PKESK,
        SKESK,
        key::{PublicParts, UnspecifiedRole},
    },
    parse::{
        Parse,
        stream::{
            DecryptionHelper,
            DecryptorBuilder,
            MessageStructure,
            VerificationHelper,
        },
    },
    policy::StandardPolicy,
    types::SymmetricAlgorithm,
};

use crate::error::{Error, Result};

/// The magic that opens a keybox file. (GnuPG `kbx/keybox-blob.c`)
const KEYBOX_MAGIC: &[u8; 4] = b"KBXf";

/// The blob of a keybox that holds an OpenPGP keyblock.
const KEYBOX_OPENPGP: u8 = 2;

/// The fixed part of a keybox blob: the length, the type, the version,
/// the flags, and the two words that place the keyblock.
const KEYBOX_HEADER: usize = 16;

/// The keyring files of a GnuPG home, in the order that GnuPG made them.
const KEYRINGS: [&str; 2] = ["pubring.kbx", "pubring.gpg"];

/// The home that GnuPG uses when the environment names none.
const GNUPG_HOME: &str = ".gnupg";

/// The plaintext of an encrypted body. (§5.4)
///
/// The bytes are a whole message, as the store holds it. This function
/// finds the ciphertext inside it, asks gpg-agent for the session key,
/// and gives back the text. Nothing it reads reaches the disk.
///
/// # Errors
///
/// The function fails if the message carries no OpenPGP ciphertext, if
/// no certificate names a key that the agent holds, if the agent is not
/// running, or if the agent refuses.
pub fn decrypt(config: &PgpConfig, raw: &[u8]) -> Result<String> {
    let found = match ciphertext(raw) {
        Some(Ciphertext::OpenPgp(bytes)) => bytes,
        Some(Ciphertext::SMime) => return Err(Error::SMime),
        None => return Err(Error::NoCiphertext),
    };

    let home = home(config)?;
    let certs = certs(config, &home)?;

    if certs.is_empty() {
        return Err(Error::NoCerts(keyring(&home).unwrap_or(home)));
    }

    // The agent runs on a socket, and the socket is async. `view` is
    // not, so the runtime lives for this one call and ends with it.
    crate::block_on(open(&home, &certs, &found))
}

/// Ask the agent for the session key, and read the message with it.
async fn open(home: &Path, certs: &[Cert], message: &[u8]) -> Result<String> {
    let policy = StandardPolicy::new();
    let context =
        gnupg::Context::with_homedir(home).map_err(Error::agent_gone)?;

    // Fail here rather than inside the helper, where sequoia would turn
    // "the agent is not running" into "no key to decrypt message".
    Agent::connect(&context).await.map_err(Error::agent_gone)?;

    let helper = Helper {
        certs,
        context: &context,
    };

    let mut reader = DecryptorBuilder::from_bytes(message)
        .map_err(Error::not_openpgp)?
        .with_policy(&policy, None, helper)
        .map_err(Error::refused)?;

    let mut plain = Vec::new();
    std::io::copy(&mut reader, &mut plain)?;

    Ok(String::from_utf8_lossy(&plain).into_owned())
}

/// The certificates that say which key the agent must use.
///
/// `certs` of the configuration wins. Without it the function reads the
/// keyring of the GnuPG home, and it prefers the keybox because that is
/// what GnuPG writes today.
///
/// # Errors
///
/// The function fails if a named file is unreadable, or if its bytes
/// are neither a keybox nor a keyring.
pub fn certs(config: &PgpConfig, home: &Path) -> Result<Vec<Cert>> {
    let Some(path) = config
        .certs
        .clone()
        .map(|path| {
            crate::settings::expand(&path, crate::settings::home().as_deref())
        })
        .or_else(|| keyring(home))
    else {
        return Ok(Vec::new());
    };

    read_certs(&path)
}

/// The certificates of one file, whatever shape it has.
fn read_certs(path: &Path) -> Result<Vec<Cert>> {
    let bytes = std::fs::read(path)?;

    // A keybox wraps each keyblock in a blob. Anything else is already
    // a stream of OpenPGP packets, armored or not.
    let blocks = match is_keybox(&bytes) {
        true => keyblocks(&bytes),
        false => vec![bytes.as_slice()],
    };

    let mut found = Vec::new();

    for block in blocks {
        // One unreadable certificate must not hide the others. A
        // keyring holds every correspondent, and a single bad entry in
        // it would otherwise make every message unreadable.
        let Ok(parser) = sequoia_openpgp::cert::CertParser::from_bytes(block)
        else {
            continue;
        };

        found.extend(parser.flatten());
    }

    Ok(found)
}

/// Whether the bytes open a keybox file.
fn is_keybox(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[8..12] == KEYBOX_MAGIC
}

/// The OpenPGP keyblocks of a keybox file.
///
/// A keybox is a run of blobs. Each blob opens with its own length, so
/// the walk needs no index, and a blob of type 2 places its keyblock
/// with two words of its header. The first blob is the header of the
/// file and carries no key.
///
/// A blob whose length is zero, or whose keyblock runs past its end,
/// ends the walk. A truncated file gives the keys that came before the
/// damage rather than an error, because a reader who can open some of
/// the mail is better off than one who can open none.
fn keyblocks(bytes: &[u8]) -> Vec<&[u8]> {
    let mut found = Vec::new();
    let mut at = 0usize;

    while at + KEYBOX_HEADER <= bytes.len() {
        let blob = &bytes[at..];
        let length = u32::from_be_bytes([blob[0], blob[1], blob[2], blob[3]]);
        let Ok(length) = usize::try_from(length) else {
            break;
        };

        if length < KEYBOX_HEADER || length > blob.len() {
            break;
        }

        if blob[4] == KEYBOX_OPENPGP {
            let offset =
                u32::from_be_bytes([blob[8], blob[9], blob[10], blob[11]]);
            let size =
                u32::from_be_bytes([blob[12], blob[13], blob[14], blob[15]]);

            if let (Ok(offset), Ok(size)) =
                (usize::try_from(offset), usize::try_from(size))
                && let Some(end) = offset.checked_add(size)
                && end <= length
            {
                found.push(&blob[offset..end]);
            }
        }

        at += length;
    }

    found
}

/// The GnuPG home that holds the agent socket and the certificates.
///
/// # Errors
///
/// The function fails if no path is configured and the environment
/// names no home directory.
pub fn home(config: &PgpConfig) -> Result<PathBuf> {
    if let Some(home) = &config.home {
        return Ok(crate::settings::expand(
            home,
            crate::settings::home().as_deref(),
        ));
    }

    if let Some(home) = std::env::var_os("GNUPGHOME")
        .map(PathBuf::from)
        .filter(|home| !home.as_os_str().is_empty())
    {
        return Ok(home);
    }

    crate::settings::home()
        .map(|home| home.join(GNUPG_HOME))
        .ok_or(Error::NoGnupgHome)
}

/// The keyring file of a GnuPG home, if it has one.
fn keyring(home: &Path) -> Option<PathBuf> {
    KEYRINGS
        .iter()
        .map(|name| home.join(name))
        .find(|path| path.is_file())
}

/// What sequoia asks while it reads the message.
struct Helper<'a> {
    /// The certificates that map a key ID to a key of the agent.
    certs: &'a [Cert],

    /// The GnuPG home whose agent holds the secrets.
    context: &'a gnupg::Context,
}

impl VerificationHelper for Helper<'_> {
    /// §5.4 opens a message, and it does not judge a signature, so the
    /// helper offers no certificate to check one with.
    fn get_certs(
        &mut self,
        _ids: &[KeyHandle],
    ) -> sequoia_openpgp::Result<Vec<Cert>> {
        Ok(Vec::new())
    }

    /// A message that mailbert can read is a message it shows. Whether
    /// a signature on it holds is a separate question, and §5.4 does
    /// not ask it, so every structure passes.
    fn check(
        &mut self,
        _structure: MessageStructure,
    ) -> sequoia_openpgp::Result<()> {
        Ok(())
    }
}

impl DecryptionHelper for Helper<'_> {
    fn decrypt(
        &mut self,
        pkesks: &[PKESK],
        _skesks: &[SKESK],
        algorithm: Option<SymmetricAlgorithm>,
        decrypt: &mut dyn FnMut(
            Option<SymmetricAlgorithm>,
            &SessionKey,
        ) -> bool,
    ) -> sequoia_openpgp::Result<Option<Cert>> {
        for pkesk in pkesks {
            // A message with a hidden recipient names no key, and every
            // key of the agent is then a candidate for it.
            let wanted = pkesk.recipient();

            for (cert, key) in self.keys() {
                if let Some(wanted) = &wanted
                    && !matches(wanted, &key)
                {
                    continue;
                }

                if self.unwrap(pkesk, &key, algorithm, decrypt) {
                    return Ok(Some(cert));
                }
            }
        }

        Err(sequoia_openpgp::anyhow::anyhow!(
            "no key of the agent opens this message"
        ))
    }
}

impl Helper<'_> {
    /// Every encryption key of every certificate, with its owner.
    ///
    /// The policy rejects a key that is revoked or expired, so a key
    /// that the agent still holds but the certificate retired is never
    /// offered to the agent.
    fn keys(&self) -> Vec<(Cert, Key<PublicParts, UnspecifiedRole>)> {
        let policy = StandardPolicy::new();
        let mut found = Vec::new();

        for cert in self.certs {
            let usable = cert
                .keys()
                .with_policy(&policy, None)
                .supported()
                .for_storage_encryption()
                .for_transport_encryption();

            for key in usable {
                found.push((
                    cert.clone(),
                    key.key().clone().role_into_unspecified(),
                ));
            }
        }

        found
    }

    /// Ask the agent to unwrap one session key.
    ///
    /// The answer says whether the session key opened the message. A
    /// refusal is not an error here, because the next key of the next
    /// certificate may be the right one.
    fn unwrap(
        &self,
        pkesk: &PKESK,
        key: &Key<PublicParts, UnspecifiedRole>,
        algorithm: Option<SymmetricAlgorithm>,
        decrypt: &mut dyn FnMut(
            Option<SymmetricAlgorithm>,
            &SessionKey,
        ) -> bool,
    ) -> bool {
        let Ok(mut pair) = sequoia_gpg_agent::KeyPair::new_for_gnupg_context(
            self.context,
            key,
        ) else {
            return false;
        };

        pkesk
            .decrypt(&mut pair, algorithm)
            .map(|(algorithm, session)| decrypt(algorithm, &session))
            .unwrap_or(false)
    }
}

/// Whether a key is the one that a message names.
///
/// A message names its recipient by key ID or by fingerprint, and
/// `aliases` compares the two shapes without caring which it got.
fn matches(
    wanted: &KeyHandle,
    key: &Key<PublicParts, UnspecifiedRole>,
) -> bool {
    wanted.aliases(KeyHandle::from(key.fingerprint()))
}

#[cfg(test)]
mod tests {
    //! # Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_keybox_walk_never_panics` | invariant | The bytes come off the disk, and a file that another program wrote is not a file mailbert can trust. A panic here is `view` dying on the mail it was asked to open. |
    //! | `prop_a_keyblock_stays_inside_its_blob` | invariant | A blob places its keyblock with numbers of its own. Numbers that reach past the blob would read the next key, or the next blob's header, as a certificate. |

    use hegel::{TestCase, generators as gs};

    use super::*;

    /// One blob of a keybox, holding `body` as its keyblock.
    ///
    /// The layout is GnuPG's: the length, the type, the version, the
    /// flags, and the two words that place the keyblock inside the blob.
    fn blob(kind: u8, body: &[u8]) -> Vec<u8> {
        let offset = u32::try_from(KEYBOX_HEADER).expect("a small header");
        let size = u32::try_from(body.len()).expect("a small keyblock");
        let length = offset + size;

        let mut bytes = Vec::new();

        bytes.extend_from_slice(&length.to_be_bytes());
        bytes.push(kind);
        bytes.push(1);
        bytes.extend_from_slice(&0u16.to_be_bytes());
        bytes.extend_from_slice(&offset.to_be_bytes());
        bytes.extend_from_slice(&size.to_be_bytes());
        bytes.extend_from_slice(body);

        bytes
    }

    /// The header blob that opens every keybox file.
    fn header() -> Vec<u8> {
        let mut bytes = 32u32.to_be_bytes().to_vec();

        bytes.push(1);
        bytes.push(1);
        bytes.extend_from_slice(&0u16.to_be_bytes());
        bytes.extend_from_slice(KEYBOX_MAGIC);
        bytes.resize(32, 0);

        bytes
    }

    /// A whole keybox file holding one blob per keyblock.
    fn keybox(keyblocks: &[&[u8]]) -> Vec<u8> {
        let mut bytes = header();

        for block in keyblocks {
            bytes.extend_from_slice(&blob(KEYBOX_OPENPGP, block));
        }

        bytes
    }

    #[test]
    fn a_keybox_names_itself() {
        assert!(is_keybox(&keybox(&[])));
    }

    /// A keyring is a stream of packets, and its first bytes are a
    /// packet tag. Reading one as a keybox would find no key at all.
    #[test]
    fn a_keyring_is_not_a_keybox() {
        assert!(!is_keybox(b"\x99\x01\x0d\x04this is a public key packet"));
        assert!(!is_keybox(b"KBXf"));
        assert!(!is_keybox(b""));
    }

    #[test]
    fn takes_every_keyblock_of_a_keybox() {
        let bytes = keybox(&[b"first key", b"second key"]);

        assert_eq!(
            keyblocks(&bytes),
            vec![b"first key".as_slice(), b"second key".as_slice()]
        );
    }

    /// The first blob of a keybox is the header of the file. A walk
    /// that took it would hand the magic to the OpenPGP parser.
    #[test]
    fn skips_the_header_blob_of_a_keybox() {
        let bytes = keybox(&[]);

        assert!(keyblocks(&bytes).is_empty());
    }

    /// A keybox holds more than keys: GnuPG writes X.509 blobs into the
    /// same file, and those are not OpenPGP certificates.
    #[test]
    fn skips_a_blob_that_holds_no_openpgp_keyblock() {
        let mut bytes = header();

        bytes.extend_from_slice(&blob(3, b"an X.509 certificate"));
        bytes.extend_from_slice(&blob(KEYBOX_OPENPGP, b"a key"));

        assert_eq!(keyblocks(&bytes), vec![b"a key".as_slice()]);
    }

    /// A file that a crash cut short still holds the keys that came
    /// before the cut, and `view` opens the mail addressed to them.
    #[test]
    fn a_truncated_keybox_gives_the_keys_that_survived_it() {
        let bytes = keybox(&[b"first key", b"second key"]);
        let cut = &bytes[..bytes.len() - 4];

        assert_eq!(keyblocks(cut), vec![b"first key".as_slice()]);
    }

    /// A blob whose length is zero would leave the walk where it stands.
    #[test]
    fn a_blob_of_no_length_ends_the_walk() {
        let mut bytes = header();

        bytes.extend_from_slice(&[0u8; KEYBOX_HEADER]);
        bytes.extend_from_slice(&blob(KEYBOX_OPENPGP, b"a key"));

        assert!(keyblocks(&bytes).is_empty());
    }

    #[test]
    fn the_configured_home_wins() {
        let config = PgpConfig {
            home: Some(PathBuf::from("/keys")),
            certs: None,
        };

        assert_eq!(home(&config).expect("a home"), PathBuf::from("/keys"));
    }

    #[test]
    fn the_keyring_of_a_home_with_none_is_nothing() {
        let dir = tempfile::tempdir().expect("a temporary directory");

        assert_eq!(keyring(dir.path()), None);
    }

    /// GnuPG 2.1 writes a keybox, and a home that has both kept the
    /// old keyring only for the programs that never learned the new one.
    #[test]
    fn the_keybox_of_a_home_wins_over_the_keyring() {
        let dir = tempfile::tempdir().expect("a temporary directory");

        for name in KEYRINGS {
            std::fs::write(dir.path().join(name), b"").expect("a writable dir");
        }

        assert_eq!(keyring(dir.path()), Some(dir.path().join("pubring.kbx")));
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_keybox_walk_never_panics(tc: TestCase) {
        let bytes: Vec<u8> =
            tc.draw(gs::vecs(gs::integers::<u8>()).max_size(128));

        let _ = keyblocks(&bytes);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_keyblock_stays_inside_its_blob(tc: TestCase) {
        let offset: u32 =
            tc.draw(gs::integers::<u32>().min_value(0).max_value(64));
        let size: u32 =
            tc.draw(gs::integers::<u32>().min_value(0).max_value(64));
        let body: Vec<u8> =
            tc.draw(gs::vecs(gs::integers::<u8>()).max_size(32));

        // A blob that says its keyblock is somewhere other than where
        // the bytes are. Nothing it claims may reach past its length.
        let length =
            u32::try_from(KEYBOX_HEADER + body.len()).expect("a small blob");

        let mut bytes = header();

        bytes.extend_from_slice(&length.to_be_bytes());
        bytes.push(KEYBOX_OPENPGP);
        bytes.push(1);
        bytes.extend_from_slice(&0u16.to_be_bytes());
        bytes.extend_from_slice(&offset.to_be_bytes());
        bytes.extend_from_slice(&size.to_be_bytes());
        bytes.extend_from_slice(&body);

        for block in keyblocks(&bytes) {
            assert!(
                block.len() <= usize::try_from(length).expect("a small blob"),
                "a keyblock of {} bytes came out of a blob of {length}",
                block.len()
            );
        }
    }
}
