//! Resolve a [`VersionSpec`] against a crate's published versions.
//!
//! Three input shapes, one output: a concrete `semver::Version`.
//!
//! - `Latest` — max non-yanked stable version (skips pre-releases).
//! - `Concrete(v)` — must exist in metadata; yanked is allowed but
//!   surfaces in [`Resolution::yanked`] so callers can warn.
//! - `Req(req)` — max non-yanked matching version; pre-releases are
//!   only considered when the req explicitly matches them (Cargo
//!   semantics inherited from the `semver` crate).

use crate::{
    crate_ref::VersionSpec,
    crates_io::CrateMetadata,
    error::{Error, Result},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolution {
    pub version: semver::Version,
    pub yanked: bool,
    pub checksum: String,
}

pub fn resolve(
    spec: &VersionSpec,
    metadata: &CrateMetadata,
) -> Result<Resolution> {
    if metadata.versions.is_empty() {
        return Err(Error::NoMatchingVersion {
            name: metadata.name.clone(),
            spec: format!("{spec:?}"),
        });
    }

    match spec {
        VersionSpec::Latest => resolve_latest(metadata),
        VersionSpec::Concrete(v) => resolve_concrete(v, metadata),
        VersionSpec::Req(req) => resolve_req(req, metadata),
    }
}

fn resolve_latest(metadata: &CrateMetadata) -> Result<Resolution> {
    metadata
        .versions
        .iter()
        .filter(|v| !v.yanked && v.num.pre.is_empty())
        .max_by(|a, b| a.num.cmp(&b.num))
        .map(|v| Resolution {
            version: v.num.clone(),
            yanked: v.yanked,
            checksum: v.checksum.clone(),
        })
        .ok_or_else(|| Error::NoMatchingVersion {
            name: metadata.name.clone(),
            spec: "latest (no stable, non-yanked versions)".to_string(),
        })
}

fn resolve_concrete(
    target: &semver::Version,
    metadata: &CrateMetadata,
) -> Result<Resolution> {
    metadata
        .versions
        .iter()
        .find(|v| &v.num == target)
        .map(|v| Resolution {
            version: v.num.clone(),
            yanked: v.yanked,
            checksum: v.checksum.clone(),
        })
        .ok_or_else(|| Error::NoMatchingVersion {
            name: metadata.name.clone(),
            spec: target.to_string(),
        })
}

fn resolve_req(
    req: &semver::VersionReq,
    metadata: &CrateMetadata,
) -> Result<Resolution> {
    metadata
        .versions
        .iter()
        .filter(|v| !v.yanked && req.matches(&v.num))
        .max_by(|a, b| a.num.cmp(&b.num))
        .map(|v| Resolution {
            version: v.num.clone(),
            yanked: v.yanked,
            checksum: v.checksum.clone(),
        })
        .ok_or_else(|| Error::NoMatchingVersion {
            name: metadata.name.clone(),
            spec: req.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crates_io::PublishedVersion;

    fn pv(num: &str, yanked: bool) -> PublishedVersion {
        PublishedVersion {
            num: semver::Version::parse(num).unwrap(),
            yanked,
            checksum: format!("checksum-{num}"),
        }
    }

    fn metadata(versions: Vec<PublishedVersion>) -> CrateMetadata {
        CrateMetadata {
            name: "serde".to_string(),
            versions,
        }
    }

    #[test]
    fn latest_returns_max_stable_non_yanked() {
        let md = metadata(vec![
            pv("1.0.218", false),
            pv("1.0.219", false),
            pv("2.0.0-rc.1", false),
        ]);
        let r = resolve(&VersionSpec::Latest, &md).unwrap();
        assert_eq!(r.version, semver::Version::new(1, 0, 219));
        assert!(!r.yanked);
    }

    #[test]
    fn latest_skips_yanked() {
        let md = metadata(vec![pv("1.0.218", false), pv("1.0.219", true)]);
        let r = resolve(&VersionSpec::Latest, &md).unwrap();
        assert_eq!(r.version, semver::Version::new(1, 0, 218));
    }

    #[test]
    fn latest_skips_prerelease() {
        let md = metadata(vec![pv("1.0.0", false), pv("2.0.0-alpha.1", false)]);
        let r = resolve(&VersionSpec::Latest, &md).unwrap();
        assert_eq!(r.version, semver::Version::new(1, 0, 0));
    }

    #[test]
    fn latest_errors_when_only_yanked_or_prerelease() {
        let md = metadata(vec![pv("1.0.0", true), pv("2.0.0-rc.1", false)]);
        let err = resolve(&VersionSpec::Latest, &md).unwrap_err();
        assert!(matches!(err, Error::NoMatchingVersion { .. }));
    }

    #[test]
    fn concrete_returns_named_version() {
        let md = metadata(vec![pv("1.0.218", false), pv("1.0.219", false)]);
        let target = semver::Version::new(1, 0, 219);
        let r = resolve(&VersionSpec::Concrete(target.clone()), &md).unwrap();
        assert_eq!(r.version, target);
    }

    #[test]
    fn concrete_allows_yanked_but_flags_it() {
        let md = metadata(vec![pv("1.0.219", true)]);
        let target = semver::Version::new(1, 0, 219);
        let r = resolve(&VersionSpec::Concrete(target.clone()), &md).unwrap();
        assert_eq!(r.version, target);
        assert!(r.yanked);
    }

    #[test]
    fn concrete_missing_errors() {
        let md = metadata(vec![pv("1.0.218", false)]);
        let target = semver::Version::new(1, 0, 219);
        let err = resolve(&VersionSpec::Concrete(target), &md).unwrap_err();
        assert!(matches!(err, Error::NoMatchingVersion { .. }));
    }

    #[test]
    fn req_returns_max_matching_non_yanked() {
        let md = metadata(vec![
            pv("1.0.218", false),
            pv("1.0.219", false),
            pv("2.0.0", false),
        ]);
        let req: semver::VersionReq = "^1.0".parse().unwrap();
        let r = resolve(&VersionSpec::Req(req), &md).unwrap();
        assert_eq!(r.version, semver::Version::new(1, 0, 219));
    }

    #[test]
    fn req_skips_yanked() {
        let md = metadata(vec![pv("1.0.218", false), pv("1.0.219", true)]);
        let req: semver::VersionReq = "^1.0".parse().unwrap();
        let r = resolve(&VersionSpec::Req(req), &md).unwrap();
        assert_eq!(r.version, semver::Version::new(1, 0, 218));
    }

    #[test]
    fn req_with_no_match_errors() {
        let md = metadata(vec![pv("1.0.0", false)]);
        let req: semver::VersionReq = "^2.0".parse().unwrap();
        let err = resolve(&VersionSpec::Req(req), &md).unwrap_err();
        assert!(matches!(err, Error::NoMatchingVersion { .. }));
    }

    #[test]
    fn empty_metadata_errors() {
        let md = metadata(vec![]);
        let err = resolve(&VersionSpec::Latest, &md).unwrap_err();
        assert!(matches!(err, Error::NoMatchingVersion { .. }));
    }

    /// For any non-empty metadata, `Latest` resolution always returns
    /// a version that is (1) present in the metadata, (2) non-yanked,
    /// (3) without a pre-release tag, and (4) at least as large as
    /// every other version satisfying the same conditions.
    #[hegel::test(test_cases = 50)]
    fn prop_latest_picks_correct_version(tc: hegel::TestCase) {
        use hegel::generators as gs;

        let n: u8 = tc.draw(gs::integers::<u8>().min_value(1).max_value(8));
        let mut versions = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let major: u64 =
                tc.draw(gs::integers::<u64>().min_value(0).max_value(5));
            let minor: u64 =
                tc.draw(gs::integers::<u64>().min_value(0).max_value(5));
            let patch: u64 =
                tc.draw(gs::integers::<u64>().min_value(0).max_value(20));
            let yanked: bool = tc.draw(gs::booleans());
            let prerelease: bool = tc.draw(gs::booleans());

            let num = if prerelease {
                semver::Version::parse(&format!("{major}.{minor}.{patch}-rc.1"))
                    .unwrap()
            } else {
                semver::Version::new(major, minor, patch)
            };
            versions.push(PublishedVersion {
                num,
                yanked,
                checksum: String::new(),
            });
        }

        let md = metadata(versions.clone());

        let any_eligible =
            versions.iter().any(|v| !v.yanked && v.num.pre.is_empty());

        match resolve(&VersionSpec::Latest, &md) {
            Ok(r) => {
                assert!(any_eligible);
                let chosen =
                    versions.iter().find(|v| v.num == r.version).unwrap();
                assert!(!chosen.yanked, "chose yanked");
                assert!(chosen.num.pre.is_empty(), "chose pre-release");
                let max_eligible = versions
                    .iter()
                    .filter(|v| !v.yanked && v.num.pre.is_empty())
                    .map(|v| &v.num)
                    .max()
                    .unwrap();
                assert_eq!(
                    &r.version, max_eligible,
                    "did not pick the maximum eligible version"
                );
            }
            Err(_) => {
                assert!(
                    !any_eligible,
                    "rejected even though an eligible version exists"
                );
            }
        }
    }
}
