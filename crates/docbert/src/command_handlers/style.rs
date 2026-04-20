//! Minimal ANSI styling and duration formatting for CLI command output.
//!
//! Styling honours the community `NO_COLOR` convention and auto-disables
//! whenever stderr is not a terminal, so piping `docbert sync > log`
//! stays escape-free. Set `CLICOLOR_FORCE=1` to force colors on (useful
//! for CI jobs that capture coloured logs).

use std::{fmt, io::IsTerminal, sync::OnceLock, time::Duration};

const DIM: &str = "\x1b[2m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const RESET: &str = "\x1b[0m";
const BOLD_GREEN: &str = "\x1b[1;32m";
const BOLD_CYAN: &str = "\x1b[1;36m";

/// Whether stderr should receive ANSI colour codes.
///
/// Decided once per process. Rules, in order:
/// 1. `NO_COLOR` set (any value) → off.
/// 2. `CLICOLOR_FORCE` set to a non-empty, non-`0` value → on.
/// 3. Otherwise on iff stderr is a terminal.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        if std::env::var_os("NO_COLOR").is_some() {
            return false;
        }
        if let Some(force) = std::env::var_os("CLICOLOR_FORCE") {
            let s = force.to_string_lossy();
            if !s.is_empty() && s != "0" {
                return true;
            }
        }
        std::io::stderr().is_terminal()
    })
}

/// Wrap `value` with an ANSI code and the reset sequence when colours
/// are enabled; emit it verbatim otherwise. The wrapper is cheap to
/// construct and only materialises escapes on demand through `Display`,
/// so `format!("{}", styled(...))` stays allocation-free on the
/// no-colour path.
struct Styled<'a, T: fmt::Display> {
    code: &'static str,
    value: &'a T,
}

impl<T: fmt::Display> fmt::Display for Styled<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if enabled() {
            write!(f, "{}{}{}", self.code, self.value, RESET)
        } else {
            write!(f, "{}", self.value)
        }
    }
}

fn styled<'a, T: fmt::Display>(
    code: &'static str,
    value: &'a T,
) -> Styled<'a, T> {
    Styled { code, value }
}

/// A section header: bold + bright green. Use for top-level phases
/// ("Rebuild complete", "Sync complete").
pub fn header<'a, T: fmt::Display>(value: &'a T) -> impl fmt::Display + 'a {
    styled(BOLD_GREEN, value)
}

/// A subsection header: bold + cyan. Use for per-collection banners.
pub fn subheader<'a, T: fmt::Display>(value: &'a T) -> impl fmt::Display + 'a {
    styled(BOLD_CYAN, value)
}

/// Warning: yellow.
pub fn warn<'a, T: fmt::Display>(value: &'a T) -> impl fmt::Display + 'a {
    styled(YELLOW, value)
}

/// Dim/greyed-out secondary text (counts, paths).
pub fn dim<'a, T: fmt::Display>(value: &'a T) -> impl fmt::Display + 'a {
    styled(DIM, value)
}

/// Accent: cyan. Use for durations so they pop out of surrounding text.
pub fn accent<'a, T: fmt::Display>(value: &'a T) -> impl fmt::Display + 'a {
    styled(CYAN, value)
}

/// Human-friendly duration: hours/minutes/seconds/milliseconds, kept
/// short enough to trail other log text.
///
/// Examples:
/// - `123 ms`
/// - `2.345s`
/// - `1m 23.4s`
/// - `1h 02m 03s`
pub fn format_duration(d: Duration) -> String {
    let total_secs = d.as_secs_f64();
    if total_secs < 1.0 {
        return format!("{} ms", d.as_millis());
    }
    if total_secs < 60.0 {
        return format!("{total_secs:.3}s");
    }
    let secs_u = d.as_secs();
    let hours = secs_u / 3600;
    let minutes = (secs_u % 3600) / 60;
    let secs_f = total_secs - (hours * 3600 + minutes * 60) as f64;
    if hours > 0 {
        format!("{hours}h {minutes:02}m {secs_f:02.0}s")
    } else {
        format!("{minutes}m {secs_f:04.1}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_duration_sub_second_uses_milliseconds() {
        assert_eq!(format_duration(Duration::from_millis(250)), "250 ms");
    }

    #[test]
    fn format_duration_sub_minute_uses_seconds_with_three_decimals() {
        assert_eq!(format_duration(Duration::from_millis(2_345)), "2.345s");
    }

    #[test]
    fn format_duration_multi_minute_uses_minutes_and_seconds() {
        let d = Duration::from_secs(83) + Duration::from_millis(400);
        assert_eq!(format_duration(d), "1m 23.4s");
    }

    #[test]
    fn format_duration_multi_hour_uses_hms() {
        let d = Duration::from_secs(3_723);
        assert_eq!(format_duration(d), "1h 02m 03s");
    }
}
