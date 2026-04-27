//! Parsing for user-facing crate references like `serde`, `serde@1.0.219`,
//! `serde@^1.0`, or `serde@latest`.

use std::fmt;

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrateRef {
    pub name: String,
    pub version: VersionSpec,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionSpec {
    /// `name`, `name@latest`, or `name@*` — defer to upstream's max stable.
    Latest,
    /// A concrete `X.Y.Z` (or pre/build-tagged) version.
    Concrete(semver::Version),
    /// A semver requirement like `^1.0` or `>=1, <2`.
    Req(semver::VersionReq),
}

impl CrateRef {
    pub fn parse(input: &str) -> Result<Self> {
        if input.is_empty() {
            return Err(Error::InvalidCrateRef(input.to_string()));
        }

        // Reject `serde@1.0@2.0` etc. — at most one `@` separator.
        if input.matches('@').count() > 1 {
            return Err(Error::InvalidCrateRef(input.to_string()));
        }

        let (name, version_str) = match input.split_once('@') {
            None => (input, None),
            Some((n, v)) => (n, Some(v)),
        };

        if name.is_empty() || name.chars().any(char::is_whitespace) {
            return Err(Error::InvalidCrateRef(input.to_string()));
        }

        let version = match version_str {
            None => VersionSpec::Latest,
            Some("") => return Err(Error::InvalidCrateRef(input.to_string())),
            Some("latest") | Some("*") => VersionSpec::Latest,
            Some(v) => match semver::Version::parse(v) {
                Ok(version) => VersionSpec::Concrete(version),
                Err(_) => VersionSpec::Req(v.parse()?),
            },
        };

        Ok(CrateRef {
            name: name.to_string(),
            version,
        })
    }
}

impl fmt::Display for CrateRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.version {
            VersionSpec::Latest => write!(f, "{}", self.name),
            VersionSpec::Concrete(v) => write!(f, "{}@{}", self.name, v),
            VersionSpec::Req(r) => write!(f, "{}@{}", self.name, r),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_name_is_latest() {
        let r = CrateRef::parse("serde").unwrap();
        assert_eq!(r.name, "serde");
        assert_eq!(r.version, VersionSpec::Latest);
    }

    #[test]
    fn name_at_concrete_version() {
        let r = CrateRef::parse("serde@1.0.219").unwrap();
        assert_eq!(r.name, "serde");
        assert_eq!(
            r.version,
            VersionSpec::Concrete(semver::Version::new(1, 0, 219))
        );
    }

    #[test]
    fn name_at_semver_req() {
        let r = CrateRef::parse("serde@^1.0").unwrap();
        assert_eq!(r.name, "serde");
        let req: semver::VersionReq = "^1.0".parse().unwrap();
        assert_eq!(r.version, VersionSpec::Req(req));
    }

    #[test]
    fn latest_sentinel_is_latest() {
        assert_eq!(
            CrateRef::parse("serde@latest").unwrap().version,
            VersionSpec::Latest
        );
    }

    #[test]
    fn star_sentinel_is_latest() {
        assert_eq!(
            CrateRef::parse("serde@*").unwrap().version,
            VersionSpec::Latest
        );
    }

    #[test]
    fn empty_input_errors() {
        assert!(matches!(
            CrateRef::parse(""),
            Err(Error::InvalidCrateRef(_))
        ));
    }

    #[test]
    fn empty_name_errors() {
        assert!(matches!(
            CrateRef::parse("@1.0.0"),
            Err(Error::InvalidCrateRef(_))
        ));
    }

    #[test]
    fn empty_version_errors() {
        assert!(matches!(
            CrateRef::parse("serde@"),
            Err(Error::InvalidCrateRef(_))
        ));
    }

    #[test]
    fn whitespace_in_name_errors() {
        assert!(matches!(
            CrateRef::parse("se rde@1.0.0"),
            Err(Error::InvalidCrateRef(_))
        ));
    }

    #[test]
    fn double_at_errors() {
        assert!(matches!(
            CrateRef::parse("serde@1.0.0@2.0.0"),
            Err(Error::InvalidCrateRef(_))
        ));
    }

    #[test]
    fn display_bare_for_latest() {
        let r = CrateRef::parse("serde").unwrap();
        assert_eq!(r.to_string(), "serde");
    }

    #[test]
    fn display_concrete_uses_at_separator() {
        let r = CrateRef::parse("serde@1.0.219").unwrap();
        assert_eq!(r.to_string(), "serde@1.0.219");
    }

    #[test]
    fn display_req_uses_at_separator() {
        let r = CrateRef::parse("serde@^1.0").unwrap();
        assert_eq!(r.to_string(), "serde@^1.0");
    }

    /// Generator: a syntactically-valid `CrateRef` covering Latest /
    /// Concrete / Req variants. Names use a Cargo-like alphabet so the
    /// `@` and whitespace rules in [`CrateRef::parse`] never reject the
    /// generated input.
    #[hegel::composite]
    fn arb_crate_ref(tc: hegel::TestCase) -> CrateRef {
        use hegel::generators as gs;

        let name: String = tc.draw(
            gs::text()
                .alphabet("abcdefghijklmnopqrstuvwxyz0123456789_-")
                .min_size(1)
                .max_size(20),
        );

        let kind: u8 = tc.draw(gs::integers::<u8>().min_value(0).max_value(2));
        let version = match kind {
            0 => VersionSpec::Latest,
            1 => {
                let major: u64 =
                    tc.draw(gs::integers::<u64>().min_value(0).max_value(50));
                let minor: u64 =
                    tc.draw(gs::integers::<u64>().min_value(0).max_value(50));
                let patch: u64 =
                    tc.draw(gs::integers::<u64>().min_value(0).max_value(50));
                VersionSpec::Concrete(semver::Version::new(major, minor, patch))
            }
            _ => {
                let major: u64 =
                    tc.draw(gs::integers::<u64>().min_value(0).max_value(50));
                let minor: u64 =
                    tc.draw(gs::integers::<u64>().min_value(0).max_value(50));
                let req_str = format!("^{major}.{minor}");
                VersionSpec::Req(req_str.parse().unwrap())
            }
        };

        CrateRef { name, version }
    }

    #[hegel::test(test_cases = 100)]
    fn prop_parse_display_round_trip(tc: hegel::TestCase) {
        let original: CrateRef = tc.draw(arb_crate_ref());
        let rendered = original.to_string();
        let parsed = CrateRef::parse(&rendered)
            .unwrap_or_else(|e| panic!("re-parse of {rendered:?} failed: {e}"));
        assert_eq!(parsed, original, "round-trip mismatch for {rendered:?}");
    }
}
