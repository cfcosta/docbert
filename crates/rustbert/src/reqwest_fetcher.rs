//! Real [`Fetcher`] backed by `reqwest` with rustls.
//!
//! Used by the production binary; unit tests use [`crate::FakeFetcher`]
//! instead — `ReqwestFetcher` is intentionally untestable in pure unit
//! tests (it would need a live HTTP server). Integration tests against
//! crates.io are out of scope for v1.
//!
//! Behavior:
//!
//! - User-Agent: `rustbert/<crate version> (https://github.com/cfcosta/rustbert)`
//!   so crates.io can identify the client per their TOS.
//! - 30-second per-request timeout.
//! - Retries on 429 / 503 with exponential backoff (1s, 2s, 4s) up to 3
//!   times, honoring `Retry-After` when present.

use std::time::Duration;

use crate::{
    error::{Error, Result},
    fetcher::Fetcher,
};

const USER_AGENT: &str = concat!(
    "rustbert/",
    env!("CARGO_PKG_VERSION"),
    " (https://github.com/cfcosta/rustbert)",
);

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_RETRIES: u32 = 3;

#[derive(Clone)]
pub struct ReqwestFetcher {
    client: reqwest::Client,
}

impl ReqwestFetcher {
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|e| Error::HttpTransport {
                url: "<client construction>".to_string(),
                message: e.to_string(),
            })?;
        Ok(Self { client })
    }
}

impl Default for ReqwestFetcher {
    fn default() -> Self {
        Self::new().expect("default reqwest client should build")
    }
}

impl Fetcher for ReqwestFetcher {
    async fn get_bytes(&self, url: &str) -> Result<Vec<u8>> {
        let mut attempt = 0u32;
        let mut backoff = Duration::from_secs(1);
        loop {
            let response = self.client.get(url).send().await.map_err(|e| {
                Error::HttpTransport {
                    url: url.to_string(),
                    message: e.to_string(),
                }
            })?;

            let status = response.status();

            if status.is_success() {
                return response.bytes().await.map(|b| b.to_vec()).map_err(
                    |e| Error::HttpTransport {
                        url: url.to_string(),
                        message: e.to_string(),
                    },
                );
            }

            let retryable = status.as_u16() == 429
                || status.as_u16() == 503
                || status.is_server_error();
            if retryable && attempt < MAX_RETRIES {
                let retry_after = parse_retry_after(&response);
                let delay = retry_after.unwrap_or(backoff);
                drop(response);
                tokio::time::sleep(delay).await;
                attempt += 1;
                backoff = (backoff * 2).min(Duration::from_secs(16));
                continue;
            }

            return Err(Error::HttpStatus {
                url: url.to_string(),
                status: status.as_u16(),
            });
        }
    }
}

fn parse_retry_after(response: &reqwest::Response) -> Option<Duration> {
    let value = response.headers().get(reqwest::header::RETRY_AFTER)?;
    let s = value.to_str().ok()?;
    s.parse::<u64>().ok().map(Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_agent_includes_crate_version() {
        // The package version is set in Cargo.toml; the build-time
        // env! macro picks it up. We can't verify the exact version in
        // a test (it changes), but we can verify the shape.
        assert!(USER_AGENT.starts_with("rustbert/"));
        assert!(USER_AGENT.contains("("));
    }

    #[test]
    fn fetcher_constructs_without_panicking() {
        let _ = ReqwestFetcher::new().expect("client should build");
    }
}
