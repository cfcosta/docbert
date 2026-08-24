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

    /// The set of one range of UIDs.
    pub fn range(low: u32, high: u32) -> Self {
        let low = low.max(1);
        let high = high.max(1);

        Self {
            ranges: vec![(low.min(high), low.max(high))],
        }
    }

    /// How many UIDs the set holds.
    pub fn count(&self) -> u64 {
        self.ranges
            .iter()
            .map(|(low, high)| u64::from(*high) - u64::from(*low) + 1)
            .sum()
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
    /// Each UID that is in this set, or in the other one.
    pub fn union(&self, other: &Self) -> Self {
        let mut ranges = self.ranges.clone();
        ranges.extend_from_slice(&other.ranges);

        Self {
            ranges: merged(ranges),
        }
    }

    /// Each UID that is in this set, and in the other one. (§3.2)
    ///
    /// A plan asks for the UIDs that the store owes. The server says
    /// which UIDs it holds. What the two share is the mail that a
    /// fetch can bring, and every other UID went away long ago.
    ///
    /// Both sets hold their ranges sorted, and no two ranges of one
    /// set touch, so one walk down the two lists gives the answer.
    pub fn and(&self, other: &Self) -> Self {
        let mut ranges = Vec::new();
        let (mut here, mut there) = (0, 0);

        while here < self.ranges.len() && there < other.ranges.len() {
            let (low, high) = self.ranges[here];
            let (their_low, their_high) = other.ranges[there];

            let start = low.max(their_low);
            let end = high.min(their_high);

            if start <= end {
                ranges.push((start, end));
            }

            // The range that ends first can share nothing more, so it
            // steps and the other one waits for the next range.
            if high < their_high {
                here += 1;
            } else {
                there += 1;
            }
        }

        Self { ranges }
    }

    /// Each UID that is in this set, and not in the other one.
    pub fn without(&self, other: &Self) -> Self {
        let mut ranges = Vec::new();

        for (low, high) in &self.ranges {
            let mut pieces = vec![(*low, *high)];

            for (cut_low, cut_high) in &other.ranges {
                let mut left = Vec::new();

                for (low, high) in pieces {
                    if *cut_high < low || *cut_low > high {
                        left.push((low, high));
                        continue;
                    }
                    if *cut_low > low {
                        left.push((low, cut_low - 1));
                    }
                    if *cut_high < high {
                        left.push((cut_high + 1, high));
                    }
                }

                pieces = left;
            }

            ranges.extend(pieces);
        }

        Self {
            ranges: merged(ranges),
        }
    }

    /// The batches of this set, newest first. (§3.2)
    ///
    /// Each batch holds no more UIDs than the size. A batch can hold
    /// more than one range, because a fetch of `1:2,10` costs the same
    /// as a fetch of `1:3`.
    pub fn split(&self, size: u32) -> Vec<Self> {
        let size = u64::from(size.max(1));
        let mut out = Vec::new();
        let mut taken: Vec<(u32, u32)> = Vec::new();
        let mut room = size;

        for (low, high) in self.ranges.iter().rev() {
            let (low, mut top) = (*low, *high);

            loop {
                let span = u64::from(top) - u64::from(low) + 1;
                let take = span.min(room) as u32;
                let bottom = top - (take - 1);

                taken.push((bottom, top));
                room -= u64::from(take);

                if room == 0 {
                    out.push(Self {
                        ranges: merged(std::mem::take(&mut taken)),
                    });
                    room = size;
                }
                if bottom == low {
                    break;
                }

                top = bottom - 1;
            }
        }

        if !taken.is_empty() {
            out.push(Self {
                ranges: merged(taken),
            });
        }

        out
    }

    /// The largest UID of the set.
    pub fn last(&self) -> Option<u32> {
        self.ranges.last().map(|(_, high)| *high)
    }

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

/// The batches of one fetch, newest first. (§3.1, §3.2)
///
/// §3.1 fetches in batches of a few hundred UIDs, and §3.2 wants the
/// newest mail first. The list is therefore in falling order, and the
/// batch of the largest UIDs comes first.
pub fn batches(low: u32, high: u32, size: u32) -> Vec<UidSet> {
    let mut out = Vec::new();
    if low > high {
        return out;
    }

    let size = size.max(1);
    let mut top = high;

    loop {
        let bottom = top.saturating_sub(size - 1).max(low);
        out.push(UidSet::range(bottom, top));

        if bottom <= low {
            return out;
        }
        top = bottom - 1;
    }
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
    //! | `prop_the_batches_cover_the_range` | model-based | §3.2 fetches every message once. A gap between two batches leaves mail that no later sync finds. |
    //! | `prop_a_batch_never_holds_more_than_its_size` | algebraic | §3.1 keeps each command short, so one slow batch never blocks the rest. |
    //! | `prop_the_batches_run_newest_first` | algebraic | §3.2 wants the newest mail first, so a sync that stops early still gives the mail that matters. |
    //! | `prop_two_sets_share_the_uids_that_both_hold` | model-based | §3.2 cuts the plan down to the UIDs that the server holds. An answer that is too wide fetches a UID that is not there, and one that is too narrow leaves mail behind. |
    //! | `prop_the_order_of_two_sets_never_changes_what_they_share` | algebraic | The plan and the answer of the server go in either order. A result that depends on the order is a result that nobody can read. |
    //! | `prop_what_two_sets_share_is_what_neither_cuts_away` | differential | Two ways to cut a set must give one answer, or one of them is wrong. |

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

    /// The text of each batch.
    fn text(batches: &[UidSet]) -> Vec<String> {
        batches.iter().map(UidSet::to_string).collect()
    }

    /// A range of UIDs, and the size of a batch of it.
    fn a_plan(tc: &TestCase) -> (u32, u32, u32) {
        let low: u32 =
            tc.draw(gs::integers::<u32>().min_value(1).max_value(50));
        let span: u32 =
            tc.draw(gs::integers::<u32>().min_value(0).max_value(300));
        let size: u32 =
            tc.draw(gs::integers::<u32>().min_value(0).max_value(40));

        (low, low + span, size)
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

    // -----------------------------------------------------------------
    // Unit tests: the batches of a fetch.
    // -----------------------------------------------------------------

    #[test]
    fn a_short_range_makes_one_batch() {
        assert_eq!(text(&batches(1, 10, 300)), ["1:10"]);
    }

    #[test]
    fn a_long_range_makes_batches_of_the_size() {
        assert_eq!(
            text(&batches(1, 1000, 300)),
            ["701:1000", "401:700", "101:400", "1:100"]
        );
    }

    #[test]
    fn a_range_that_holds_nothing_makes_no_batch() {
        assert!(batches(5, 4, 10).is_empty());
    }

    #[test]
    fn a_size_of_one_makes_one_batch_for_each_uid() {
        assert_eq!(text(&batches(1, 3, 1)), ["3", "2", "1"]);
    }

    #[test]
    fn a_size_of_nothing_counts_as_one() {
        assert_eq!(text(&batches(1, 3, 0)), text(&batches(1, 3, 1)));
    }

    #[test]
    fn a_range_that_starts_high_still_makes_batches() {
        assert_eq!(
            text(&batches(900, 1000, 40)),
            ["961:1000", "921:960", "900:920"]
        );
    }

    #[test]
    fn a_range_holds_the_count_of_its_uids() {
        assert_eq!(UidSet::range(3, 7).count(), 5);
        assert_eq!(UidSet::range(7, 3).count(), 5);
        assert_eq!(UidSet::parse("1:3,9").unwrap().count(), 4);
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 200)]
    fn prop_the_batches_cover_the_range(tc: TestCase) {
        let (low, high, size) = a_plan(&tc);
        let made = batches(low, high, size);

        for uid in low..=high {
            assert!(
                made.iter().any(|batch| batch.holds(uid)),
                "no batch holds {uid}"
            );
        }
        for batch in &made {
            for (start, end) in batch.ranges() {
                assert!(*start >= low && *end <= high, "a batch goes outside");
            }
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_a_batch_never_holds_more_than_its_size(tc: TestCase) {
        let (low, high, size) = a_plan(&tc);

        for batch in batches(low, high, size) {
            assert!(batch.count() <= u64::from(size.max(1)), "{batch} is long");
        }
    }

    #[hegel::test(test_cases = 200)]
    fn prop_the_batches_run_newest_first(tc: TestCase) {
        let (low, high, size) = a_plan(&tc);
        let made = batches(low, high, size);

        for pair in made.windows(2) {
            let (younger, _) = pair[0].ranges()[0];
            let (_, older) = pair[1].ranges()[0];

            assert!(older < younger, "{} comes before {}", pair[0], pair[1]);
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
    // -----------------------------------------------------------------
    // Union, difference, and split. (§3.3)
    // -----------------------------------------------------------------

    #[test]
    fn a_union_holds_both_sets() {
        let one = UidSet::parse("1:3,10").unwrap();
        let two = UidSet::parse("4:5,20:21").unwrap();

        assert_eq!(one.union(&two).to_string(), "1:5,10,20:21");
    }

    #[test]
    fn a_union_with_nothing_changes_nothing() {
        let one = UidSet::parse("1:3").unwrap();

        assert_eq!(one.union(&UidSet::new()), one);
        assert_eq!(UidSet::new().union(&one), one);
    }

    #[test]
    fn a_difference_takes_a_piece_out_of_the_middle() {
        let one = UidSet::parse("1:10").unwrap();
        let two = UidSet::parse("4:6").unwrap();

        assert_eq!(one.without(&two).to_string(), "1:3,7:10");
    }

    #[test]
    fn a_difference_takes_a_piece_off_each_end() {
        let one = UidSet::parse("5:10").unwrap();

        assert_eq!(
            one.without(&UidSet::parse("1:6").unwrap()).to_string(),
            "7:10"
        );
        assert_eq!(
            one.without(&UidSet::parse("9:20").unwrap()).to_string(),
            "5:8"
        );
    }

    #[test]
    fn a_difference_of_everything_is_nothing() {
        let one = UidSet::parse("1:10,20").unwrap();

        assert!(one.without(&UidSet::parse("1:30").unwrap()).is_empty());
    }

    #[test]
    fn a_difference_of_nothing_changes_nothing() {
        let one = UidSet::parse("1:10,20").unwrap();

        assert_eq!(one.without(&UidSet::new()), one);
    }

    // -----------------------------------------------------------------
    // The UIDs that two sets share. (§3.2)
    // -----------------------------------------------------------------

    #[test]
    fn two_sets_share_the_uids_that_are_in_both() {
        let one = UidSet::parse("1:10,20:30").unwrap();
        let two = UidSet::parse("5:25").unwrap();

        assert_eq!(one.and(&two).to_string(), "5:10,20:25");
    }

    #[test]
    fn two_sets_that_share_nothing_give_nothing() {
        let one = UidSet::parse("1:5").unwrap();
        let two = UidSet::parse("6:10").unwrap();

        assert!(one.and(&two).is_empty());
    }

    /// A server that holds no mail must leave no batch to fetch.
    #[test]
    fn a_set_shares_nothing_with_nothing() {
        let one = UidSet::parse("1:10,20").unwrap();

        assert!(one.and(&UidSet::new()).is_empty());
        assert!(UidSet::new().and(&one).is_empty());
    }

    #[test]
    fn a_set_shares_the_whole_of_itself() {
        let one = UidSet::parse("1:10,20").unwrap();

        assert_eq!(one.and(&one), one);
    }

    /// One range of a set can cut another range into two pieces.
    #[test]
    fn a_hole_in_one_set_makes_a_hole_in_the_answer() {
        let one = UidSet::parse("1:20").unwrap();
        let two = UidSet::parse("1:5,15:20").unwrap();

        assert_eq!(one.and(&two).to_string(), "1:5,15:20");
    }

    #[test]
    fn a_split_gives_the_batches_newest_first() {
        let set = UidSet::parse("1:10").unwrap();

        assert_eq!(text(&set.split(4)), vec!["7:10", "3:6", "1:2"]);
    }

    #[test]
    fn a_split_fills_a_batch_out_of_more_than_one_range() {
        let set = UidSet::parse("1:2,10:11,20").unwrap();

        assert_eq!(text(&set.split(3)), vec!["10:11,20", "1:2"]);
    }

    #[test]
    fn a_split_of_nothing_gives_no_batch() {
        assert!(UidSet::new().split(10).is_empty());
    }

    #[test]
    fn a_split_of_a_size_of_nothing_counts_as_one() {
        let set = UidSet::parse("1:3").unwrap();

        assert_eq!(text(&set.split(0)), vec!["3", "2", "1"]);
    }

    #[test]
    fn the_last_uid_of_a_set_is_the_largest_one() {
        assert_eq!(UidSet::parse("1:3,20,7").unwrap().last(), Some(20));
        assert_eq!(UidSet::new().last(), None);
    }

    #[hegel::test(test_cases = 150)]
    fn prop_a_union_holds_every_uid_of_both_sets(tc: TestCase) {
        let one = tc.draw(some_uids());
        let two = tc.draw(some_uids());
        let both = UidSet::of(&one).union(&UidSet::of(&two));

        for uid in one.iter().chain(&two) {
            assert!(both.holds(*uid), "the union lost {uid}");
        }
        for (low, high) in both.ranges() {
            for uid in [*low, *high] {
                assert!(one.contains(&uid) || two.contains(&uid));
            }
        }
    }

    #[hegel::test(test_cases = 150)]
    fn prop_a_difference_holds_what_the_other_set_does_not(tc: TestCase) {
        let one = tc.draw(some_uids());
        let two = tc.draw(some_uids());
        let left = UidSet::of(&one).without(&UidSet::of(&two));

        for uid in &one {
            assert_eq!(
                left.holds(*uid),
                !two.contains(uid),
                "the difference is wrong at {uid}"
            );
        }
        for uid in &two {
            assert!(!left.holds(*uid), "the difference kept {uid}");
        }
    }

    #[hegel::test(test_cases = 150)]
    fn prop_two_sets_share_the_uids_that_both_hold(tc: TestCase) {
        let one = tc.draw(some_uids());
        let two = tc.draw(some_uids());
        let both = UidSet::of(&one).and(&UidSet::of(&two));

        for uid in one.iter().chain(two.iter()) {
            assert_eq!(
                both.holds(*uid),
                one.contains(uid) && two.contains(uid),
                "the answer is wrong at {uid}"
            );
        }
    }

    /// Which set comes first must not change the answer.
    #[hegel::test(test_cases = 100)]
    fn prop_the_order_of_two_sets_never_changes_what_they_share(tc: TestCase) {
        let one = UidSet::of(&tc.draw(some_uids()));
        let two = UidSet::of(&tc.draw(some_uids()));

        assert_eq!(one.and(&two), two.and(&one));
    }

    /// The two ways to cut a set must agree. `and` exists because it
    /// says what it means, and not because it can do more.
    #[hegel::test(test_cases = 100)]
    fn prop_what_two_sets_share_is_what_neither_cuts_away(tc: TestCase) {
        let one = UidSet::of(&tc.draw(some_uids()));
        let two = UidSet::of(&tc.draw(some_uids()));

        assert_eq!(one.and(&two), one.without(&one.without(&two)));
    }

    #[hegel::test(test_cases = 100)]
    fn prop_a_set_less_itself_is_nothing(tc: TestCase) {
        let uids = tc.draw(some_uids());
        let set = UidSet::of(&uids);

        assert!(set.without(&set).is_empty());
        assert_eq!(set.union(&set), set);
    }

    #[hegel::test(test_cases = 150)]
    fn prop_the_batches_of_a_split_cover_the_set(tc: TestCase) {
        let uids = tc.draw(some_uids());
        let size = tc.draw(gs::integers::<u32>().min_value(0).max_value(20));
        let set = UidSet::of(&uids);
        let parts = set.split(size);

        let mut whole = UidSet::new();
        let mut total = 0;
        for part in &parts {
            assert!(part.count() <= u64::from(size.max(1)));
            total += part.count();
            whole = whole.union(part);
        }

        assert_eq!(whole, set);
        assert_eq!(total, set.count(), "a batch holds a UID twice");
    }

    #[hegel::test(test_cases = 100)]
    fn prop_a_split_runs_newest_first(tc: TestCase) {
        let uids = tc.draw(some_uids());
        let size = tc.draw(gs::integers::<u32>().min_value(1).max_value(20));
        let parts = UidSet::of(&uids).split(size);

        for pair in parts.windows(2) {
            let earlier = pair[0].ranges().first().map(|(low, _)| *low);
            let later = pair[1].last();

            assert!(later < earlier, "{later:?} is not below {earlier:?}");
        }
    }
}
