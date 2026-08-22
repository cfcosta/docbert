//! The socket under one connection: TLS, or plain for a test.

use std::{
    io,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use rustls::{ClientConfig, RootCertStore};
use rustls_pki_types::ServerName;
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    net::TcpStream,
};
use tokio_rustls::{TlsConnector, client::TlsStream};

use crate::error::Result;

/// The port of IMAP over TLS.
pub const TLS_PORT: u16 = 993;

/// The port of IMAP without TLS.
pub const PLAIN_PORT: u16 = 143;

/// The socket that carries one IMAP connection.
pub enum Stream {
    /// A socket with TLS on it. A real server needs this.
    Tls(Box<TlsStream<TcpStream>>),
    /// A socket without TLS. A test uses this.
    Plain(TcpStream),
}

impl Stream {
    /// Open a socket to this host and port.
    ///
    /// §3.1 pipelines its commands, so the socket sends at once and
    /// waits for no other command.
    pub async fn open(host: &str, port: u16, tls: bool) -> Result<Self> {
        let socket = TcpStream::connect((host, port)).await?;
        socket.set_nodelay(true)?;

        if !tls {
            return Ok(Self::Plain(socket));
        }

        let name = ServerName::try_from(host.to_string())?;
        let connector = TlsConnector::from(Arc::new(safely()?));

        Ok(Self::Tls(Box::new(connector.connect(name, socket).await?)))
    }

    pub fn is_tls(&self) -> bool {
        matches!(self, Self::Tls(_))
    }
}

/// The TLS rules: the roots of the machine, and no client certificate.
fn safely() -> Result<ClientConfig> {
    let roots = RootCertStore {
        roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
    };

    Ok(ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()?
    .with_root_certificates(roots)
    .with_no_client_auth())
}

impl AsyncRead for Stream {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Tls(stream) => {
                Pin::new(stream.as_mut()).poll_read(context, buffer)
            }
            Self::Plain(stream) => Pin::new(stream).poll_read(context, buffer),
        }
    }
}

impl AsyncWrite for Stream {
    fn poll_write(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut() {
            Self::Tls(stream) => {
                Pin::new(stream.as_mut()).poll_write(context, bytes)
            }
            Self::Plain(stream) => Pin::new(stream).poll_write(context, bytes),
        }
    }

    fn poll_flush(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Tls(stream) => Pin::new(stream.as_mut()).poll_flush(context),
            Self::Plain(stream) => Pin::new(stream).poll_flush(context),
        }
    }

    fn poll_shutdown(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Tls(stream) => {
                Pin::new(stream.as_mut()).poll_shutdown(context)
            }
            Self::Plain(stream) => Pin::new(stream).poll_shutdown(context),
        }
    }
}

#[cfg(test)]
mod tests {
    use tokio::{io::AsyncWriteExt, net::TcpListener};

    use super::*;

    #[tokio::test]
    async fn a_plain_socket_carries_bytes() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            stream.write_all(b"* OK ready\r\n").await.unwrap();
        });

        let mut stream = Stream::open("127.0.0.1", address.port(), false)
            .await
            .unwrap();

        assert!(!stream.is_tls());

        let mut reader = tokio::io::BufReader::new(&mut stream);
        let raw = crate::wire::read_line(&mut reader).await.unwrap();

        assert_eq!(raw, b"* OK ready\r\n".to_vec());
    }

    #[tokio::test]
    async fn a_socket_that_no_one_listens_on_is_an_error() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        assert!(Stream::open("127.0.0.1", port, false).await.is_err());
    }

    #[test]
    fn the_tls_rules_hold_the_roots_of_the_machine() {
        assert!(safely().is_ok());
    }
}
