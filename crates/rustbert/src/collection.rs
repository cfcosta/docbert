//! Synthetic collection naming.
//!
//! Each cached `(crate, version)` is stored as a hidden collection in
//! rustbert's data dir under the canonical name
//! `rustbert:<crate>@<resolved_version>`. This module owns the parse /
//! format pair and never accepts non-concrete versions — the resolution
//! step happens before storage.

use std::fmt;

use crate::error::{Error, Result};

const PREFIX: &str = "rustbert:";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticCollection {
    pub crate_name: String,
    pub version: semver::Version,
}

impl SyntheticCollection {
    pub fn parse(input: &str) -> Result<Self> {
        let body = input
            .strip_prefix(PREFIX)
            .ok_or_else(|| Error::InvalidCollectionName(input.to_string()))?;
        let (crate_name, version_str) = body
            .split_once('@')
            .ok_or_else(|| Error::InvalidCollectionName(input.to_string()))?;

        if crate_name.is_empty() || version_str.is_empty() {
            return Err(Error::InvalidCollectionName(input.to_string()));
        }

        // Resolved versions are always concrete X.Y.Z — anything else
        // means the caller skipped the resolution step.
        let version = semver::Version::parse(version_str)
            .map_err(|_| Error::InvalidCollectionName(input.to_string()))?;

        Ok(SyntheticCollection {
            crate_name: crate_name.to_string(),
            version,
        })
    }
}

impl fmt::Display for SyntheticCollection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{PREFIX}{}@{}", self.crate_name, self.version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn formats_with_rustbert_prefix() {
        let c = SyntheticCollection {
            crate_name: "serde".to_string(),
            version: semver::Version::new(1, 0, 219),
        };
        assert_eq!(c.to_string(), "rustbert:serde@1.0.219");
    }

    #[test]
    fn parses_canonical_form() {
        let c = SyntheticCollection::parse("rustbert:serde@1.0.219").unwrap();
        assert_eq!(c.crate_name, "serde");
        assert_eq!(c.version, semver::Version::new(1, 0, 219));
    }

    #[test]
    fn parse_rejects_missing_prefix() {
        assert!(matches!(
            SyntheticCollection::parse("serde@1.0.219"),
            Err(Error::InvalidCollectionName(_))
        ));
    }

    #[test]
    fn parse_rejects_missing_at() {
        assert!(matches!(
            SyntheticCollection::parse("rustbert:serde"),
            Err(Error::InvalidCollectionName(_))
        ));
    }

    #[test]
    fn parse_rejects_empty_name() {
        assert!(matches!(
            SyntheticCollection::parse("rustbert:@1.0.0"),
            Err(Error::InvalidCollectionName(_))
        ));
    }

    #[test]
    fn parse_rejects_empty_version() {
        assert!(matches!(
            SyntheticCollection::parse("rustbert:serde@"),
            Err(Error::InvalidCollectionName(_))
        ));
    }

    #[test]
    fn parse_rejects_non_semver_version() {
        // Non-strict shapes (e.g. "1.0", "^1") are not valid resolved
        // versions — the cache only stores concrete `X.Y.Z` outputs.
        assert!(SyntheticCollection::parse("rustbert:serde@1.0").is_err());
    }

    #[hegel::test(test_cases = 100)]
    fn prop_parse_display_round_trip(tc: hegel::TestCase) {
        use hegel::generators as gs;

        let crate_name: String = tc.draw(
            gs::text()
                .alphabet("abcdefghijklmnopqrstuvwxyz0123456789_-")
                .min_size(1)
                .max_size(20),
        );
        let major: u64 =
            tc.draw(gs::integers::<u64>().min_value(0).max_value(50));
        let minor: u64 =
            tc.draw(gs::integers::<u64>().min_value(0).max_value(50));
        let patch: u64 =
            tc.draw(gs::integers::<u64>().min_value(0).max_value(50));

        let original = SyntheticCollection {
            crate_name,
            version: semver::Version::new(major, minor, patch),
        };
        let rendered = original.to_string();
        let parsed = SyntheticCollection::parse(&rendered)
            .unwrap_or_else(|e| panic!("re-parse of {rendered:?} failed: {e}"));
        assert_eq!(parsed, original, "round-trip mismatch for {rendered:?}");
    }
}
