//! Date terms: what `date:` accepts and what it means.
//!
//! The first version accepts absolute dates (`2026-08-14`), ranges with
//! `..` (`2026-01-01..2026-06-30`), open ranges (`..2026-01-01`), the
//! keywords `today`, `yesterday`, and `now`, and simple relative
//! offsets (`7d`, `3w`, `6m`, `2y`).
//!
//! notmuch's full natural-language parser ("last friday", "two weeks
//! ago") is a separate project, and mailbert does not have it yet.
//!
//! Two rules make the relative offsets predictable:
//!
//! - Alone, `7d` means "in the last 7 days". The range has no end,
//!   because a mail clock that runs fast is common, and a message from
//!   this morning must not disappear because its `Date` says 20:00.
//! - Inside a range, `7d` is the instant 7 days ago, so that `7d..2d`
//!   is the window between them.
//!
//! A [`Clock`] holds the current time and the offset from UTC, so that
//! `today` is the day of the user and every test is deterministic.
//!
//! See `docs/mailbert.md` §7.3.

use std::ops::RangeInclusive;

use thiserror::Error;

/// Seconds in a day.
pub const DAY: i64 = 86_400;

/// The years that a date may name.
const YEARS: RangeInclusive<i64> = 1..=9_999;

/// The units of a relative offset: day, week, month, year.
const UNITS: [char; 4] = ['d', 'w', 'm', 'y'];

/// The names that IMAP gives the months, in order (RFC 3501).
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
    "Nov", "Dec",
];

/// The current time, and the offset that turns it into a local time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Clock {
    now: i64,
    utc_offset: i32,
}

/// A range of instants: `start` is included, `end` is not.
///
/// `None` on a side means that the side is open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DateRange {
    pub start: Option<i64>,
    pub end: Option<i64>,
}

/// Why a date term did not parse.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DateError {
    #[error("`{0}` is not a date that I know")]
    Unknown(String),

    #[error("a range needs a date on at least one side")]
    EmptyRange,

    #[error("`{0}` is too far away to be a date")]
    OutOfRange(String),

    #[error("the range starts after it ends")]
    Backwards,
}

/// One instant, and the range that the instant stands for.
struct Moment {
    start: i64,
    end: i64,

    /// Whether the term, used alone, means "from then until now".
    open_ended: bool,
}

/// Parse one `date:` value against `clock`.
///
/// ```
/// use mailbert_core::date::{Clock, parse};
///
/// // 2026-08-22 12:00:00 UTC.
/// let clock = Clock::utc(1_787_400_000);
///
/// let today = parse("today", clock).unwrap();
/// assert!(today.contains(clock.now()));
///
/// let week = parse("7d", clock).unwrap();
/// assert_eq!(week.start, Some(clock.now() - 7 * 86_400));
/// assert_eq!(week.end, None);
/// ```
pub fn parse(text: &str, clock: Clock) -> Result<DateRange, DateError> {
    let text = text.trim();

    let Some((left, right)) = text.split_once("..") else {
        let moment = moment(text, clock)?;

        return Ok(DateRange {
            start: Some(moment.start),
            end: (!moment.open_ended).then_some(moment.end),
        });
    };

    if left.is_empty() && right.is_empty() {
        return Err(DateError::EmptyRange);
    }

    let mut range = DateRange::everything();
    if !left.is_empty() {
        range.start = Some(moment(left, clock)?.start);
    }
    if !right.is_empty() {
        range.end = Some(moment(right, clock)?.end);
    }

    if range.is_empty() {
        return Err(DateError::Backwards);
    }

    Ok(range)
}

impl Clock {
    /// A clock in UTC.
    pub fn utc(now: i64) -> Self {
        Self::new(now, 0)
    }

    /// A clock `utc_offset` seconds east of UTC.
    pub fn new(now: i64, utc_offset: i32) -> Self {
        Self { now, utc_offset }
    }

    pub fn now(self) -> i64 {
        self.now
    }

    pub fn utc_offset(self) -> i32 {
        self.utc_offset
    }

    /// The first instant of the day that holds `at`.
    fn day_start(self, at: i64) -> i64 {
        let local = at + i64::from(self.utc_offset);

        local.div_euclid(DAY) * DAY - i64::from(self.utc_offset)
    }

    /// The year, month, and day that hold `at`.
    fn civil(self, at: i64) -> (i64, u32, u32) {
        let local = at + i64::from(self.utc_offset);

        civil_from_days(local.div_euclid(DAY))
    }

    /// The first instant of one civil date.
    fn civil_start(self, year: i64, month: u32, day: u32) -> i64 {
        days_from_civil(year, month, day) * DAY - i64::from(self.utc_offset)
    }

    /// `at`, moved back by whole calendar months.
    ///
    /// The day of the month clamps, so one month before 2026-03-31 is
    /// 2026-02-28 and not a date in March. The time of day stays.
    fn shift_months(self, at: i64, months: i64) -> Option<i64> {
        let (year, month, day) = self.civil(at);

        let total = year
            .checked_mul(12)?
            .checked_add(i64::from(month) - 1)?
            .checked_sub(months)?;

        let year = total.div_euclid(12);
        let month = u32::try_from(total.rem_euclid(12)).ok()? + 1;
        if !YEARS.contains(&year) {
            return None;
        }

        let day = day.min(days_in_month(year, month));
        let time_of_day = at - self.day_start(at);

        Some(self.civil_start(year, month, day) + time_of_day)
    }
}

impl DateRange {
    /// The range that holds every instant.
    pub fn everything() -> Self {
        Self::default()
    }

    /// Whether `at` is inside the range.
    pub fn contains(self, at: i64) -> bool {
        self.start.is_none_or(|start| at >= start)
            && self.end.is_none_or(|end| at < end)
    }

    /// Whether no instant at all is inside the range.
    pub fn is_empty(self) -> bool {
        match (self.start, self.end) {
            (Some(start), Some(end)) => start >= end,
            _ => false,
        }
    }
}

impl Moment {
    fn closed(start: i64, end: i64) -> Self {
        Self {
            start,
            end,
            open_ended: false,
        }
    }

    /// An instant that, used alone, means "from then until now".
    fn open(at: i64) -> Self {
        Self {
            start: at,
            end: at,
            open_ended: true,
        }
    }
}

/// Read one side of a range, or a term that stands alone.
fn moment(text: &str, clock: Clock) -> Result<Moment, DateError> {
    let day = clock.day_start(clock.now());

    match text.to_ascii_lowercase().as_str() {
        "now" => return Ok(Moment::closed(clock.now(), clock.now() + 1)),
        "today" => return Ok(Moment::closed(day, day + DAY)),
        "yesterday" => return Ok(Moment::closed(day - DAY, day)),
        _ => {}
    }

    if let Some(moment) = absolute(text, clock)? {
        return Ok(moment);
    }

    if let Some(moment) = relative(text, clock)? {
        return Ok(moment);
    }

    Err(DateError::Unknown(text.to_string()))
}

/// Read `YYYY`, `YYYY-MM`, or `YYYY-MM-DD`.
///
/// `None` means that the text is not an absolute date at all, and an
/// error means that it is one but names a day that does not exist.
fn absolute(text: &str, clock: Clock) -> Result<Option<Moment>, DateError> {
    let mut parts = text.split('-');

    let Some(year) = parts.next().and_then(|part| digits(part, 4)) else {
        return Ok(None);
    };
    let (month, day) = (parts.next(), parts.next());
    if parts.next().is_some() {
        return Ok(None);
    }

    let year = i64::from(year);
    if !YEARS.contains(&year) {
        return Err(DateError::OutOfRange(text.to_string()));
    }

    let Some(month) = month else {
        return Ok(Some(Moment::closed(
            clock.civil_start(year, 1, 1),
            clock.civil_start(year + 1, 1, 1),
        )));
    };

    let Some(month) = digits(month, 2) else {
        return Ok(None);
    };
    if !(1..=12).contains(&month) {
        return Err(DateError::Unknown(text.to_string()));
    }

    let Some(day) = day else {
        let next = match month {
            12 => (year + 1, 1),
            _ => (year, month + 1),
        };

        return Ok(Some(Moment::closed(
            clock.civil_start(year, month, 1),
            clock.civil_start(next.0, next.1, 1),
        )));
    };

    let Some(day) = digits(day, 2) else {
        return Ok(None);
    };
    if !(1..=days_in_month(year, month)).contains(&day) {
        return Err(DateError::Unknown(text.to_string()));
    }

    let start = clock.civil_start(year, month, day);

    Ok(Some(Moment::closed(start, start + DAY)))
}

/// Read the INTERNALDATE that IMAP gives a message (§3.3).
///
/// The text is the RFC 3501 form, such as `14-Aug-2026 09:30:00 +0000`.
/// The day is two characters, and a space stands for a leading zero.
/// `None` means that the server sent text that is not a date-time.
///
/// # Examples
///
/// ```
/// use mailbert_core::date::internal_date;
///
/// assert_eq!(internal_date("01-Jan-2020 00:00:00 +0000"), Some(1_577_836_800));
/// assert_eq!(internal_date("yesterday"), None);
/// ```
pub fn internal_date(text: &str) -> Option<i64> {
    let text = text.trim_start();
    let (date, rest) = text.split_once(' ')?;
    let (time, zone) = rest.trim_start().split_once(' ')?;

    let mut parts = date.split('-');
    let day = parts.next().and_then(day_of)?;
    let month = parts.next().and_then(month_of)?;
    let year = i64::from(parts.next().and_then(|part| digits(part, 4))?);
    if parts.next().is_some() || !YEARS.contains(&year) {
        return None;
    }
    if !(1..=days_in_month(year, month)).contains(&day) {
        return None;
    }

    let mut clock = time.split(':');
    let hour = clock.next().and_then(|part| digits(part, 2))?;
    let minute = clock.next().and_then(|part| digits(part, 2))?;
    let second = clock.next().and_then(|part| digits(part, 2))?;
    if clock.next().is_some() || hour > 23 || minute > 59 || second > 60 {
        return None;
    }

    let seconds = days_from_civil(year, month, day) * DAY
        + i64::from(hour) * 3_600
        + i64::from(minute) * 60
        + i64::from(second);

    Some(seconds - i64::from(zone_of(zone.trim_end())?))
}

/// The day of the month, which IMAP writes as one or two digits.
///
/// RFC 3501 pads a single digit with a space, and the caller removed
/// that space before the split.
fn day_of(text: &str) -> Option<u32> {
    match text.len() {
        1 => digits(text, 1),
        _ => digits(text, 2),
    }
}

/// The number of a month that IMAP names, such as `Aug`.
fn month_of(name: &str) -> Option<u32> {
    let found = MONTHS
        .iter()
        .position(|month| month.eq_ignore_ascii_case(name))?;

    u32::try_from(found + 1).ok()
}

/// The seconds that a zone such as `+0530` is ahead of UTC.
fn zone_of(text: &str) -> Option<i32> {
    let (sign, digits_of) = text.split_at_checked(1)?;
    let sign = match sign {
        "+" => 1,
        "-" => -1,
        _ => return None,
    };

    let hours = i32::try_from(digits(digits_of.get(..2)?, 2)?).ok()?;
    let minutes = i32::try_from(digits(digits_of.get(2..)?, 2)?).ok()?;
    if minutes > 59 {
        return None;
    }

    Some(sign * (hours * 3_600 + minutes * 60))
}

/// Read a relative offset, such as `7d` or `6m`.
fn relative(text: &str, clock: Clock) -> Result<Option<Moment>, DateError> {
    let Some(unit) = text.chars().last() else {
        return Ok(None);
    };

    let unit = unit.to_ascii_lowercase();
    if !UNITS.contains(&unit) {
        return Ok(None);
    }

    let count = &text[..text.len() - unit.len_utf8()];
    if count.is_empty() || !count.bytes().all(|byte| byte.is_ascii_digit()) {
        return Ok(None);
    }

    let too_far = || DateError::OutOfRange(text.to_string());
    let Ok(count) = count.parse::<i64>() else {
        return Err(too_far());
    };

    let at = match unit {
        'd' => seconds_before(clock.now(), count, DAY),
        'w' => seconds_before(clock.now(), count, 7 * DAY),
        'm' => clock.shift_months(clock.now(), count),
        'y' => count
            .checked_mul(12)
            .and_then(|months| clock.shift_months(clock.now(), months)),
        _ => unreachable!("the unit is one of {UNITS:?}"),
    };

    let Some(at) = at else {
        return Err(too_far());
    };

    Ok(Some(Moment::open(at)))
}

/// `count` units of `seconds` before `from`.
fn seconds_before(from: i64, count: i64, seconds: i64) -> Option<i64> {
    from.checked_sub(count.checked_mul(seconds)?)
}

/// Read exactly `len` ASCII digits.
fn digits(text: &str, len: usize) -> Option<u32> {
    if text.len() != len || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }

    text.parse().ok()
}

/// Whether `year` has a 29th of February.
fn is_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// How many days `month` has. A month outside 1 to 12 has none.
fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Days from 1970-01-01 to one civil date.
///
/// This is the algorithm of Howard Hinnant. It puts March first, so
/// that the leap day falls at the end of the year and no month after it
/// moves.
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;

    let year_of_era = year - era * 400;
    let month_of_year = i64::from((month + 9) % 12);
    let day_of_year = (153 * month_of_year + 2) / 5 + i64::from(day) - 1;
    let day_of_era =
        year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;

    era * 146_097 + day_of_era - 719_468
}

/// The civil date of a day count from 1970-01-01. The other direction
/// of [`days_from_civil`].
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let days = days + 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;

    let day_of_era = days - era * 146_097;
    let year_of_era = (day_of_era - day_of_era / 1_460 + day_of_era / 36_524
        - day_of_era / 146_096)
        / 365;
    let year = year_of_era + era * 400;

    let day_of_year =
        day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_of_year = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_of_year + 2) / 5 + 1;

    let month = match month_of_year < 10 {
        true => month_of_year + 3,
        false => month_of_year - 9,
    };

    (
        if month <= 2 { year + 1 } else { year },
        month as u32,
        day as u32,
    )
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_civil_date_round_trips` | round-trip | The whole module stands on the day arithmetic. One wrong day moves every result of a date filter. |
    //! | `prop_a_valid_date_always_parses` | invariant | A date the user can read off a message must not be rejected. |
    //! | `prop_an_impossible_day_always_fails` | invariant | `2026-02-30` is a typo, and a silent reading of it gives the wrong mail. |
    //! | `prop_a_day_lasts_one_day` | metamorphic | A bare date must cover its day, no more and no less. |
    //! | `prop_a_range_holds_its_start_but_not_its_end` | invariant | The range is half-open, so two next-door ranges neither overlap nor leave a hole. |
    //! | `prop_one_day_to_itself_is_that_day` | metamorphic | `date:X..X` and `date:X` must agree, or the two forms mean different things. |
    //! | `prop_a_longer_offset_reaches_further_back` | metamorphic | `30d` must hold everything `7d` holds. |
    //! | `prop_the_offset_unit_orders_the_result` | metamorphic | A day is shorter than a week, a week than a month, a month than a year. |
    //! | `prop_parse_never_panics` | invariant | The text comes from a command line, so any bytes can arrive. |
    //! | `prop_the_offset_follows_the_clock` | metamorphic | Two clocks one hour apart must put the day boundary one hour apart. |
    //! | `prop_an_internal_date_reads_back_the_moment_that_made_it` | round-trip | The INTERNALDATE dates a message that carries no `Date` header, and it orders every sync. One wrong hour sorts the mailbox wrong. |

    use hegel::{TestCase, generators as gs};

    use super::*;

    /// 2026-08-22 12:00:00 UTC.
    const NOON: i64 = 1_787_400_000;

    /// 2026-08-22 00:00:00 UTC.
    const TODAY: i64 = 1_787_356_800;

    /// 2026-08-23 00:00:00 UTC.
    const TOMORROW: i64 = 1_787_443_200;

    /// 2026-08-21 00:00:00 UTC.
    const YESTERDAY: i64 = 1_787_270_400;

    fn clock() -> Clock {
        Clock::utc(NOON)
    }

    fn range(start: Option<i64>, end: Option<i64>) -> DateRange {
        DateRange { start, end }
    }

    // -----------------------------------------------------------------
    // Generators.
    // -----------------------------------------------------------------

    /// A year, a month, and a day that name a day that exists.
    #[hegel::composite]
    fn a_civil_date(tc: TestCase) -> (i64, u32, u32) {
        let year: i64 =
            tc.draw(gs::integers::<i64>().min_value(1970).max_value(2200));
        let month: u32 =
            tc.draw(gs::integers::<u32>().min_value(1).max_value(12));
        let day: u32 = tc.draw(
            gs::integers::<u32>()
                .min_value(1)
                .max_value(days_in_month(year, month)),
        );

        (year, month, day)
    }

    fn format(date: (i64, u32, u32)) -> String {
        let (year, month, day) = date;
        format!("{year:04}-{month:02}-{day:02}")
    }

    // -----------------------------------------------------------------
    // Unit tests: the INTERNALDATE of a message (§3.3).
    // -----------------------------------------------------------------

    /// 2020-01-01 00:00:00 UTC.
    const Y2020: i64 = 1_577_836_800;

    #[test]
    fn an_internal_date_reads_as_seconds() {
        assert_eq!(
            internal_date("14-Aug-2026 09:30:00 +0000"),
            Some(1_786_699_800)
        );
    }

    #[test]
    fn a_day_with_a_space_in_front_reads() {
        assert_eq!(internal_date(" 1-Jan-2020 00:00:00 +0000"), Some(Y2020));
    }

    #[test]
    fn a_zone_ahead_of_utc_moves_the_moment_back() {
        assert_eq!(
            internal_date("01-Jan-2020 00:00:00 +0100"),
            Some(Y2020 - 3_600)
        );
    }

    #[test]
    fn a_zone_behind_utc_moves_the_moment_forward() {
        assert_eq!(
            internal_date("01-Jan-2020 00:00:00 -0500"),
            Some(Y2020 + 5 * 3_600)
        );
    }

    #[test]
    fn a_zone_with_minutes_in_it_reads() {
        assert_eq!(
            internal_date("01-Jan-2020 00:00:00 +0530"),
            Some(Y2020 - 19_800)
        );
    }

    #[test]
    fn the_name_of_the_month_is_not_case_sensitive() {
        assert_eq!(internal_date("01-JAN-2020 00:00:00 +0000"), Some(Y2020));
    }

    #[test]
    fn a_month_that_is_not_a_month_reads_as_nothing() {
        assert_eq!(internal_date("01-Xxx-2020 00:00:00 +0000"), None);
    }

    #[test]
    fn a_day_that_the_month_does_not_have_reads_as_nothing() {
        assert_eq!(internal_date("31-Feb-2020 00:00:00 +0000"), None);
    }

    #[test]
    fn an_hour_that_the_day_does_not_have_reads_as_nothing() {
        assert_eq!(internal_date("01-Jan-2020 24:00:00 +0000"), None);
    }

    #[test]
    fn a_date_with_no_zone_reads_as_nothing() {
        assert_eq!(internal_date("01-Jan-2020 00:00:00"), None);
    }

    #[test]
    fn text_that_is_not_a_date_reads_as_nothing() {
        assert_eq!(internal_date("yesterday"), None);
        assert_eq!(internal_date(""), None);
    }

    #[hegel::test(test_cases = 60)]
    fn prop_an_internal_date_reads_back_the_moment_that_made_it(tc: TestCase) {
        let (year, month, day) = tc.draw(a_civil_date());
        let hour: u32 =
            tc.draw(gs::integers::<u32>().min_value(0).max_value(23));
        let minute: u32 =
            tc.draw(gs::integers::<u32>().min_value(0).max_value(59));
        let second: u32 =
            tc.draw(gs::integers::<u32>().min_value(0).max_value(59));
        let zone: i32 =
            tc.draw(gs::integers::<i32>().min_value(-720).max_value(840));

        let sign = if zone < 0 { '-' } else { '+' };
        let (hours, minutes) = (zone.abs() / 60, zone.abs() % 60);
        let text = format!(
            "{day:02}-{}-{year:04} {hour:02}:{minute:02}:{second:02} \
             {sign}{hours:02}{minutes:02}",
            MONTHS[month as usize - 1]
        );

        let want = days_from_civil(year, month, day) * DAY
            + i64::from(hour) * 3_600
            + i64::from(minute) * 60
            + i64::from(second)
            - i64::from(zone) * 60;

        assert_eq!(internal_date(&text), Some(want), "`{text}` moved");
    }

    // -----------------------------------------------------------------
    // Unit tests: absolute dates.
    // -----------------------------------------------------------------

    #[test]
    fn a_day_covers_that_day() {
        assert_eq!(
            parse("2026-08-22", clock()).unwrap(),
            range(Some(TODAY), Some(TOMORROW))
        );
    }

    #[test]
    fn a_month_covers_that_month() {
        // 2026-01-01 .. 2026-02-01.
        assert_eq!(
            parse("2026-01", clock()).unwrap(),
            range(Some(1_767_225_600), Some(1_769_904_000))
        );
    }

    #[test]
    fn a_year_covers_that_year() {
        assert_eq!(
            parse("2026", clock()).unwrap(),
            range(Some(1_767_225_600), Some(1_798_761_600))
        );
    }

    #[test]
    fn the_epoch_is_zero() {
        assert_eq!(
            parse("1970-01-01", clock()).unwrap(),
            range(Some(0), Some(DAY))
        );
    }

    #[test]
    fn a_leap_day_parses() {
        assert_eq!(
            parse("2024-02-29", clock()).unwrap().start,
            Some(1_709_164_800)
        );
        assert!(parse("2023-02-29", clock()).is_err());
    }

    #[test]
    fn an_impossible_date_fails() {
        for text in ["2026-02-30", "2026-13-01", "2026-00-01", "2026-01-00"] {
            assert!(parse(text, clock()).is_err(), "{text} parsed");
        }
    }

    #[test]
    fn a_malformed_date_fails() {
        for text in ["", "26-08-22", "2026-8-22", "banana", "2026-08-22x"] {
            assert!(parse(text, clock()).is_err(), "{text} parsed");
        }
    }

    // -----------------------------------------------------------------
    // Unit tests: keywords.
    // -----------------------------------------------------------------

    #[test]
    fn today_is_the_current_day() {
        assert_eq!(
            parse("today", clock()).unwrap(),
            range(Some(TODAY), Some(TOMORROW))
        );
    }

    #[test]
    fn yesterday_is_the_day_before() {
        assert_eq!(
            parse("yesterday", clock()).unwrap(),
            range(Some(YESTERDAY), Some(TODAY))
        );
    }

    #[test]
    fn now_is_the_current_second() {
        assert_eq!(
            parse("now", clock()).unwrap(),
            range(Some(NOON), Some(NOON + 1))
        );
    }

    #[test]
    fn a_keyword_ignores_case() {
        assert_eq!(parse("TODAY", clock()), parse("today", clock()));
    }

    // -----------------------------------------------------------------
    // Unit tests: relative offsets.
    // -----------------------------------------------------------------

    #[test]
    fn a_bare_offset_reaches_from_then_until_now() {
        assert_eq!(
            parse("7d", clock()).unwrap(),
            range(Some(NOON - 7 * DAY), None)
        );
        assert_eq!(
            parse("3w", clock()).unwrap(),
            range(Some(NOON - 21 * DAY), None)
        );
    }

    #[test]
    fn a_month_offset_counts_calendar_months() {
        // 2026-08-22 12:00 back six months is 2026-02-22 12:00.
        assert_eq!(parse("6m", clock()).unwrap().start, Some(1_771_761_600));
    }

    #[test]
    fn a_year_offset_counts_calendar_years() {
        // 2026-08-22 12:00 back two years is 2024-08-22 12:00.
        assert_eq!(parse("2y", clock()).unwrap().start, Some(1_724_328_000));
    }

    #[test]
    fn a_month_offset_clamps_a_short_month() {
        // 2026-03-31 back one month is 2026-02-28, not 2026-03-03.
        let end_of_march = Clock::utc(1_774_915_200);

        assert_eq!(
            parse("1m", end_of_march).unwrap().start,
            Some(1_772_236_800)
        );
    }

    #[test]
    fn a_zero_offset_starts_now() {
        assert_eq!(parse("0d", clock()).unwrap(), range(Some(NOON), None));
    }

    #[test]
    fn an_unknown_unit_fails() {
        for text in ["7x", "d", "7", "-3d", "7dd", "7 d"] {
            assert!(parse(text, clock()).is_err(), "{text} parsed");
        }
    }

    #[test]
    fn an_enormous_offset_fails_instead_of_wrapping() {
        assert_eq!(
            parse("999999999999y", clock()),
            Err(DateError::OutOfRange("999999999999y".to_string()))
        );
    }

    // -----------------------------------------------------------------
    // Unit tests: ranges.
    // -----------------------------------------------------------------

    #[test]
    fn a_range_covers_both_ends_whole() {
        // The first half of 2026: 2026-01-01 .. 2026-07-01.
        assert_eq!(
            parse("2026-01-01..2026-06-30", clock()).unwrap(),
            range(Some(1_767_225_600), Some(1_782_864_000))
        );
    }

    #[test]
    fn an_open_range_leaves_the_side_open() {
        assert_eq!(
            parse("..2026-08-22", clock()).unwrap(),
            range(None, Some(TOMORROW))
        );
        assert_eq!(
            parse("2026-08-22..", clock()).unwrap(),
            range(Some(TODAY), None)
        );
    }

    #[test]
    fn a_range_of_offsets_is_a_window() {
        assert_eq!(
            parse("7d..2d", clock()).unwrap(),
            range(Some(NOON - 7 * DAY), Some(NOON - 2 * DAY))
        );
    }

    #[test]
    fn a_range_ending_now_includes_now() {
        assert_eq!(
            parse("7d..now", clock()).unwrap(),
            range(Some(NOON - 7 * DAY), Some(NOON + 1))
        );
    }

    #[test]
    fn an_empty_range_fails() {
        assert_eq!(parse("..", clock()), Err(DateError::EmptyRange));
    }

    #[test]
    fn a_backwards_range_fails() {
        assert_eq!(
            parse("2026-06-30..2026-01-01", clock()),
            Err(DateError::Backwards)
        );
    }

    // -----------------------------------------------------------------
    // Unit tests: the range itself.
    // -----------------------------------------------------------------

    #[test]
    fn an_open_range_holds_everything() {
        let all = DateRange::everything();

        assert!(all.contains(i64::MIN));
        assert!(all.contains(0));
        assert!(all.contains(i64::MAX));
        assert!(!all.is_empty());
    }

    #[test]
    fn a_day_holds_its_own_instants_only() {
        let day = parse("2026-08-22", clock()).unwrap();

        assert!(!day.contains(TODAY - 1));
        assert!(day.contains(TODAY));
        assert!(day.contains(NOON));
        assert!(day.contains(TOMORROW - 1));
        assert!(!day.contains(TOMORROW));
    }

    // -----------------------------------------------------------------
    // Unit tests: the clock.
    // -----------------------------------------------------------------

    #[test]
    fn the_clock_offset_moves_the_day_boundary() {
        // One hour east of UTC, the day starts one hour earlier.
        let east = Clock::new(NOON, 3_600);

        assert_eq!(
            parse("today", east).unwrap(),
            range(Some(TODAY - 3_600), Some(TOMORROW - 3_600))
        );
    }

    #[test]
    fn the_clock_offset_can_change_which_day_today_is() {
        // 2026-08-22 23:30 UTC is already 2026-08-23 in Tokyo.
        let late = 1_787_441_400;
        let tokyo = Clock::new(late, 9 * 3_600);

        assert_eq!(
            parse("today", tokyo).unwrap().start,
            Some(TOMORROW - 9 * 3_600)
        );
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 400)]
    fn prop_a_civil_date_round_trips(tc: TestCase) {
        let date: (i64, u32, u32) = tc.draw(a_civil_date());

        let days = days_from_civil(date.0, date.1, date.2);
        assert_eq!(civil_from_days(days), date);
    }

    #[hegel::test(test_cases = 400)]
    fn prop_a_valid_date_always_parses(tc: TestCase) {
        let date: (i64, u32, u32) = tc.draw(a_civil_date());
        let text = format(date);

        let parsed = parse(&text, clock())
            .unwrap_or_else(|error| panic!("{text} failed: {error}"));

        assert_eq!(
            parsed.start,
            Some(days_from_civil(date.0, date.1, date.2) * DAY)
        );
    }

    #[hegel::test(test_cases = 300)]
    fn prop_an_impossible_day_always_fails(tc: TestCase) {
        let year: i64 =
            tc.draw(gs::integers::<i64>().min_value(1970).max_value(2200));
        let month: u32 =
            tc.draw(gs::integers::<u32>().min_value(1).max_value(12));
        let day: u32 = tc.draw(
            gs::integers::<u32>()
                .min_value(days_in_month(year, month) + 1)
                .max_value(99),
        );

        let text = format((year, month, day));
        assert!(parse(&text, clock()).is_err(), "{text} parsed");
    }

    #[hegel::test(test_cases = 400)]
    fn prop_a_day_lasts_one_day(tc: TestCase) {
        let date: (i64, u32, u32) = tc.draw(a_civil_date());

        let parsed = parse(&format(date), clock()).unwrap();
        let (Some(start), Some(end)) = (parsed.start, parsed.end) else {
            panic!("a bare date is closed on both sides");
        };

        assert_eq!(end - start, DAY);
    }

    #[hegel::test(test_cases = 400)]
    fn prop_a_range_holds_its_start_but_not_its_end(tc: TestCase) {
        let date: (i64, u32, u32) = tc.draw(a_civil_date());

        let parsed = parse(&format(date), clock()).unwrap();
        let (start, end) = (parsed.start.unwrap(), parsed.end.unwrap());

        assert!(!parsed.contains(start - 1));
        assert!(parsed.contains(start));
        assert!(parsed.contains(end - 1));
        assert!(!parsed.contains(end));
        assert!(!parsed.is_empty());
    }

    #[hegel::test(test_cases = 300)]
    fn prop_one_day_to_itself_is_that_day(tc: TestCase) {
        let date: (i64, u32, u32) = tc.draw(a_civil_date());
        let text = format(date);

        assert_eq!(
            parse(&format!("{text}..{text}"), clock()),
            parse(&text, clock())
        );
    }

    #[hegel::test(test_cases = 300)]
    fn prop_a_longer_offset_reaches_further_back(tc: TestCase) {
        let unit: String = tc.draw(gs::sampled_from(vec![
            "d".to_string(),
            "w".to_string(),
            "m".to_string(),
            "y".to_string(),
        ]));
        let short: i64 =
            tc.draw(gs::integers::<i64>().min_value(0).max_value(40));
        let extra: i64 =
            tc.draw(gs::integers::<i64>().min_value(0).max_value(40));

        let near = parse(&format!("{short}{unit}"), clock()).unwrap();
        let far = parse(&format!("{}{unit}", short + extra), clock()).unwrap();

        assert!(far.start <= near.start, "{far:?} is not before {near:?}");
        assert_eq!(far.end, None);
    }

    #[hegel::test(test_cases = 200)]
    fn prop_the_offset_unit_orders_the_result(tc: TestCase) {
        let count: i64 =
            tc.draw(gs::integers::<i64>().min_value(1).max_value(40));

        let reach = |unit: &str| {
            parse(&format!("{count}{unit}"), clock())
                .unwrap()
                .start
                .unwrap()
        };

        assert!(reach("w") < reach("d"));
        assert!(reach("m") < reach("w"));
        assert!(reach("y") < reach("m"));
    }

    #[hegel::test(test_cases = 400)]
    fn prop_parse_never_panics(tc: TestCase) {
        let text: String = tc.draw(gs::text().min_size(0).max_size(24));
        let now: i64 = tc.draw(
            gs::integers::<i64>()
                .min_value(-2_000_000_000)
                .max_value(4_000_000_000),
        );
        let offset: i32 =
            tc.draw(gs::integers::<i32>().min_value(-50_400).max_value(50_400));

        let parsed = parse(&text, Clock::new(now, offset));

        if let Ok(parsed) = parsed
            && let (Some(start), Some(end)) = (parsed.start, parsed.end)
        {
            assert!(start <= end, "{text:?} gave {parsed:?}");
        }
    }

    #[hegel::test(test_cases = 300)]
    fn prop_the_offset_follows_the_clock(tc: TestCase) {
        let offset: i32 =
            tc.draw(gs::integers::<i32>().min_value(-50_400).max_value(50_400));

        let here = parse("today", Clock::new(NOON, offset)).unwrap();
        let utc = parse("today", Clock::utc(NOON)).unwrap();

        // A day is still a day, wherever it is read.
        assert_eq!(here.end.unwrap() - here.start.unwrap(), DAY);

        // And its boundary sits where the offset puts it, to the day.
        let moved = utc.start.unwrap() - here.start.unwrap();
        assert_eq!(moved.rem_euclid(DAY), (offset as i64).rem_euclid(DAY));
    }
}
