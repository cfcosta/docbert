//! HTTP fetch abstraction.
//!
//! `Fetcher` is the async trait every code path that touches the
//! network goes through. Layers above (the crates.io API client, the
//! tarball downloader) are generic over `F: Fetcher`, so tests can
//! supply a deterministic in-memory fake without spinning up a real
//! HTTP server.

use std::{collections::HashMap, sync::Mutex};

use crate::error::{Error, Result};

/// Generic GET-only HTTP fetcher.
///
/// Trait bound `Send + Sync` lets `Fetcher` impls flow through tokio
/// tasks. Concrete impls (one in-memory fake here, one reqwest-backed
/// in `crate::reqwest_fetcher`) decide how the bytes are produced.
pub trait Fetcher: Send + Sync {
    /// Fetch the response body for `url` as raw bytes.
    ///
    /// Returns [`Error::HttpStatus`] for any non-2xx response and
    /// [`Error::HttpTransport`] for transport-level failures.
    fn get_bytes(
        &self,
        url: &str,
    ) -> impl std::future::Future<Output = Result<Vec<u8>>> + Send;
}

/// In-memory fake: returns canned bytes for pre-registered URLs and
/// records every request for later assertion.
///
/// `Clone` shares state via `Arc`s, so cloning the fake gives every
/// holder the same response table and the same request log.
#[derive(Default, Clone)]
pub struct FakeFetcher {
    responses: std::sync::Arc<Mutex<HashMap<String, FakeResponse>>>,
    requests: std::sync::Arc<Mutex<Vec<String>>>,
}

#[derive(Debug, Clone)]
enum FakeResponse {
    Ok(Vec<u8>),
    Status(u16),
}

impl FakeFetcher {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a 200-OK response for `url`.
    pub fn with_bytes(self, url: impl Into<String>, body: Vec<u8>) -> Self {
        self.responses
            .lock()
            .unwrap()
            .insert(url.into(), FakeResponse::Ok(body));
        self
    }

    /// Register a non-2xx response for `url` (e.g. 404, 503).
    pub fn with_status(self, url: impl Into<String>, status: u16) -> Self {
        self.responses
            .lock()
            .unwrap()
            .insert(url.into(), FakeResponse::Status(status));
        self
    }

    /// All URLs the fake has been asked for, in call order.
    pub fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

impl Fetcher for FakeFetcher {
    async fn get_bytes(&self, url: &str) -> Result<Vec<u8>> {
        self.requests.lock().unwrap().push(url.to_string());
        match self.responses.lock().unwrap().get(url).cloned() {
            Some(FakeResponse::Ok(bytes)) => Ok(bytes),
            Some(FakeResponse::Status(status)) => Err(Error::HttpStatus {
                url: url.to_string(),
                status,
            }),
            None => Err(Error::HttpTransport {
                url: url.to_string(),
                message: "no canned response registered".to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fake_returns_registered_bytes() {
        let fetcher = FakeFetcher::new()
            .with_bytes("https://example.com/a", b"hello".to_vec());
        let body = fetcher.get_bytes("https://example.com/a").await.unwrap();
        assert_eq!(body, b"hello");
    }

    #[tokio::test]
    async fn fake_returns_http_status_error() {
        let fetcher =
            FakeFetcher::new().with_status("https://example.com/x", 503);
        let err = fetcher
            .get_bytes("https://example.com/x")
            .await
            .unwrap_err();
        assert!(matches!(err, Error::HttpStatus { status: 503, .. }));
    }

    #[tokio::test]
    async fn fake_unregistered_url_is_transport_error() {
        let fetcher = FakeFetcher::new();
        let err = fetcher
            .get_bytes("https://example.com/missing")
            .await
            .unwrap_err();
        assert!(matches!(err, Error::HttpTransport { .. }));
    }

    #[tokio::test]
    async fn fake_records_requests_in_order() {
        let fetcher = FakeFetcher::new()
            .with_bytes("https://example.com/a", b"a".to_vec())
            .with_bytes("https://example.com/b", b"b".to_vec());
        let _ = fetcher.get_bytes("https://example.com/b").await;
        let _ = fetcher.get_bytes("https://example.com/a").await;
        let _ = fetcher.get_bytes("https://example.com/missing").await;
        assert_eq!(
            fetcher.requests(),
            vec![
                "https://example.com/b",
                "https://example.com/a",
                "https://example.com/missing",
            ]
        );
    }
}
