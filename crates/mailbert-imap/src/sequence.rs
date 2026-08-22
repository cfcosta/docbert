//! The set of UIDs that a command names. (RFC 3501 §9, `sequence-set`)
//!
//! §3.1 fetches in batches of a few hundred UIDs. A batch goes out as
//! one set, such as `1:200,305,410:*`. The fake server reads the same
//! text, so one type does both jobs.

use std::fmt;

use crate::error::{Error, Result};

/// `*` in a set. It names the largest UID of the folder.
pub const LAST: u32 = u32::MAX;

/// A set of UIDs. The ranges are sorted, and no two of them touch.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UidSet {
    ranges: Vec<(u32, u32)>,
}

impl UidSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Read a set, as `1:5,9,12:*`.
    pub fn parse(text: &str) -> Result<Self> {
        let text = text.trim();
        if text.is_empty() {
            return Err(Error::Malformed("a set of UIDs is empty".to_string()));
        }

        let mut ranges = Vec::new();
        for part in text.split(',') {
            let mut ends = part.split(':');
            let low = number(ends.next().unwrap_or_default())?;
            let high = match ends.next() {
                None => low,
                Some(end) => number(end)?,
            };
            if ends.next().is_some() {
                return Err(Error::Malformed(format!(
                    "`{part}` names two colons, and a range takes one"
                )));
            }

            ranges.push((low.min(high), low.max(high)));
        }

        Ok(Self {
            ranges: merged(ranges),
        })
    }

    /// The smallest set that holds each of these UIDs.
    pub fn of(uids: &[u32]) -> Self {
        Self {
            ranges: merged(uids.iter().map(|uid| (*uid, *uid)).collect()),
        }
    }

    /// Is this UID in the set?
    ///
    /// A `*` counts as the largest UID that a folder can hold. Call
    /// [`UidSet::resolve`] first to make it exact.
    pub fn holds(&self, uid: u32) -> bool {
        self.ranges
            .iter()
            .any(|(low, high)| uid >= *low && uid <= *high)
    }

    /// The same set, with the largest UID of the folder in each `*`.
    pub fn resolve(&self, last: u32) -> Self {
        let ranges = self
            .ranges
            .iter()
            .map(|(low, high)| {
                let low = if *low == LAST { last } else { *low };
                let high = if *high == LAST { last } else { *high };

                (low.min(high), low.max(high))
            })
            .collect();

        Self {
            ranges: merged(ranges),
        }
    }

    /// The ranges of the set, sorted.
    pub fn ranges(&self) -> &[(u32, u32)] {
        &self.ranges
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

impl fmt::Display for UidSet {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (at, (low, high)) in self.ranges.iter().enumerate() {
            if at > 0 {
                write!(out, ",")?;
            }
            if low == high {
                write!(out, "{}", end(*low))?;
            } else {
                write!(out, "{}:{}", end(*low), end(*high))?;
            }
        }

        Ok(())
    }
}

/// One end of a range, as text.
fn end(uid: u32) -> String {
    if uid == LAST {
        "*".to_string()
    } else {
        uid.to_string()
    }
}

/// Read one end of a range. A UID starts at 1, and `*` is the last one.
fn number(text: &str) -> Result<u32> {
    if text == "*" {
        return Ok(LAST);
    }

    match text.parse::<u32>() {
        Ok(0) | Err(_) => {
            Err(Error::Malformed(format!("`{text}` is not a UID")))
        }
        Ok(uid) => Ok(uid),
    }
}

/// Sort the ranges, and join the ones that touch.
fn merged(mut ranges: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    ranges.sort_unstable();

    let mut joined: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
    for (low, high) in ranges {
        match joined.last_mut() {
            Some(last) if low <= last.1.saturating_add(1) => {
                last.1 = last.1.max(high);
            }
            _ => joined.push((low, high)),
        }
    }

    joined
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_set_survives_the_round_trip` | round-trip | The client writes a set and the server reads it. A set that changes shape fetches the wrong mail. |
    //! | `prop_a_set_holds_the_uids_that_made_it` | model-based | §3.2 fetches every new message. A lost UID leaves a hole that no later sync fills. |
    //! | `prop_a_set_holds_nothing_else` | model-based | A set that is too wide fetches mail twice, and §3.1 counts every byte. |
    //! | `prop_the_ranges_of_a_set_never_touch` | algebraic | §3.1 keeps each command short. Ranges that touch make the command longer for no gain. |
    //! | `prop_a_reversed_range_is_the_same_set` | metamorphic | RFC 3501 §9 says that `10:1` and `1:10` name one set. A server writes either one. |

    use std::collections::BTreeSet;

    use hegel::{TestCase, generators as gs};

    use super::*;

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    /// A list of UIDs, sorted, with no repeat.
    #[hegel::composite]
    fn some_uids(tc: TestCase) -> Vec<u32> {
        let drawn: Vec<u32> = tc.draw(
            gs::vecs(gs::integers::<u32>().min_value(1).max_value(200))
                .min_size(0)
                .max_size(30),
        );

        drawn
            .into_iter()
            .collect::<BTreeSet<u32>>()
            .into_iter()
            .collect()
    }

    // -----------------------------------------------------------------
    // Unit tests: reading a set.
    // -----------------------------------------------------------------

    #[test]
    fn one_uid_reads_as_one_range() {
        let set = UidSet::parse("5").unwrap();

        assert_eq!(set.ranges(), [(5, 5)]);
        assert_eq!(set.to_string(), "5");
    }

    #[test]
    fn a_range_reads_as_a_range() {
        let set = UidSet::parse("1:10").unwrap();

        assert_eq!(set.ranges(), [(1, 10)]);
        assert_eq!(set.to_string(), "1:10");
    }

    #[test]
    fn a_star_reads_as_the_last_uid() {
        let set = UidSet::parse("*").unwrap();

        assert_eq!(set.ranges(), [(LAST, LAST)]);
        assert_eq!(set.to_string(), "*");
    }

    #[test]
    fn a_range_to_the_star_reads_as_an_open_range() {
        let set = UidSet::parse("1:*").unwrap();

        assert_eq!(set.ranges(), [(1, LAST)]);
        assert_eq!(set.to_string(), "1:*");
    }

    #[test]
    fn a_reversed_range_reads_the_same_way() {
        assert_eq!(
            UidSet::parse("10:1").unwrap(),
            UidSet::parse("1:10").unwrap()
        );
        assert_eq!(
            UidSet::parse("*:4").unwrap(),
            UidSet::parse("4:*").unwrap()
        );
    }

    #[test]
    fn a_list_of_ranges_reads_in_order() {
        let set = UidSet::parse("9,1:3").unwrap();

        assert_eq!(set.ranges(), [(1, 3), (9, 9)]);
        assert_eq!(set.to_string(), "1:3,9");
    }

    #[test]
    fn ranges_that_touch_become_one_range() {
        assert_eq!(UidSet::parse("1:3,4:6").unwrap().ranges(), [(1, 6)]);
        assert_eq!(UidSet::parse("1:5,3:9").unwrap().ranges(), [(1, 9)]);
    }

    // -----------------------------------------------------------------
    // Unit tests: text that names no set.
    // -----------------------------------------------------------------

    #[test]
    fn an_empty_set_is_an_error() {
        assert!(UidSet::parse("").is_err());
        assert!(UidSet::parse("   ").is_err());
    }

    #[test]
    fn a_uid_of_zero_is_an_error() {
        assert!(UidSet::parse("0").is_err());
        assert!(UidSet::parse("0:4").is_err());
    }

    #[test]
    fn a_word_that_is_not_a_number_is_an_error() {
        assert!(UidSet::parse("a").is_err());
        assert!(UidSet::parse("1:b").is_err());
    }

    #[test]
    fn two_colons_in_one_range_are_an_error() {
        assert!(UidSet::parse("1:2:3").is_err());
    }

    #[test]
    fn a_part_that_holds_nothing_is_an_error() {
        assert!(UidSet::parse("1,,2").is_err());
        assert!(UidSet::parse("1,").is_err());
    }

    // -----------------------------------------------------------------
    // Unit tests: building a set.
    // -----------------------------------------------------------------

    #[test]
    fn a_set_of_uids_takes_the_smallest_shape() {
        assert_eq!(UidSet::of(&[1, 2, 3, 7, 8, 20]).to_string(), "1:3,7:8,20");
    }

    #[test]
    fn a_set_of_uids_sorts_them_and_drops_a_repeat() {
        assert_eq!(UidSet::of(&[3, 1, 2, 2]).to_string(), "1:3");
    }

    #[test]
    fn a_set_of_no_uid_is_empty() {
        let set = UidSet::of(&[]);

        assert!(set.is_empty());
        assert_eq!(set.to_string(), "");
    }

    // -----------------------------------------------------------------
    // Unit tests: what a set holds.
    // -----------------------------------------------------------------

    #[test]
    fn a_set_holds_the_uids_of_its_ranges() {
        let set = UidSet::parse("1:3,9").unwrap();

        assert!(set.holds(1) && set.holds(2) && set.holds(3) && set.holds(9));
        assert!(!set.holds(4) && !set.holds(8) && !set.holds(10));
    }

    // -----------------------------------------------------------------
    // Unit tests: the star, once the folder is known.
    // -----------------------------------------------------------------

    #[test]
    fn the_last_uid_takes_the_place_of_the_star() {
        assert_eq!(
            UidSet::parse("1:*").unwrap().resolve(7),
            UidSet::parse("1:7").unwrap()
        );
        assert_eq!(
            UidSet::parse("*").unwrap().resolve(7),
            UidSet::parse("7").unwrap()
        );
    }

    #[test]
    fn a_range_that_starts_past_the_last_uid_keeps_the_last_message() {
        assert_eq!(
            UidSet::parse("100:*").unwrap().resolve(7),
            UidSet::parse("7:100").unwrap()
        );
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 200)]
    fn prop_a_set_survives_the_round_trip(tc: TestCase) {
        let uids: Vec<u32> = tc.draw(some_uids());
        if uids.is_empty() {
            return;
        }

        let set = UidSet::of(&uids);

        assert_eq!(UidSet::parse(&set.to_string()).unwrap(), set);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_set_holds_the_uids_that_made_it(tc: TestCase) {
        let uids: Vec<u32> = tc.draw(some_uids());
        let set = UidSet::of(&uids);

        for uid in &uids {
            assert!(set.holds(*uid), "the set lost {uid}");
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_set_holds_nothing_else(tc: TestCase) {
        let uids: Vec<u32> = tc.draw(some_uids());
        let set = UidSet::of(&uids);
        let held: BTreeSet<u32> = uids.iter().copied().collect();

        for uid in 1..=220u32 {
            assert_eq!(set.holds(uid), held.contains(&uid), "at {uid}");
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_the_ranges_of_a_set_never_touch(tc: TestCase) {
        let uids: Vec<u32> = tc.draw(some_uids());
        let set = UidSet::of(&uids);

        for pair in set.ranges().windows(2) {
            let (_, before) = pair[0];
            let (after, _) = pair[1];

            assert!(after > before + 1, "{before} and {after} touch");
        }
    }

    #[hegel::test(test_cases = 100)]
    fn prop_a_reversed_range_is_the_same_set(tc: TestCase) {
        let low: u32 =
            tc.draw(gs::integers::<u32>().min_value(1).max_value(99));
        let high: u32 =
            tc.draw(gs::integers::<u32>().min_value(100).max_value(200));

        assert_eq!(
            UidSet::parse(&format!("{high}:{low}")).unwrap(),
            UidSet::parse(&format!("{low}:{high}")).unwrap()
        );
    }
}
