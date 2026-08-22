//! The pool of connections to one server. (§3.1)
//!
//! A sync opens more than one connection, because one connection can
//! only wait for one answer. The pool opens no more connections than
//! the config names, and fewer when the server refuses one.

use std::sync::{
    Mutex,
    MutexGuard,
    atomic::{AtomicUsize, Ordering},
};

use tokio::sync::{Semaphore, SemaphorePermit};

use crate::{
    connection::Connection,
    error::{Error, Result},
    stream::TLS_PORT,
};

/// How many connections a sync opens, when the config names none.
pub const CONNECTIONS: usize = 8;

/// How to reach one server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Server {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub password: String,
    pub tls: bool,
    /// How many connections at once. (§3.1)
    pub connections: usize,
}

impl Server {
    /// A server with TLS on the usual port.
    pub fn new(host: &str) -> Self {
        Self::at(host, TLS_PORT, true)
    }

    /// A server on this port, with TLS or without it.
    pub fn at(host: &str, port: u16, tls: bool) -> Self {
        Self {
            host: host.to_string(),
            port,
            user: String::new(),
            password: String::new(),
            tls,
            connections: CONNECTIONS,
        }
    }

    pub fn with_login(mut self, user: &str, password: &str) -> Self {
        self.user = user.to_string();
        self.password = password.to_string();
        self
    }

    pub fn with_connections(mut self, count: usize) -> Self {
        self.connections = count;
        self
    }
}

/// The connections to one server.
#[derive(Debug)]
pub struct Pool {
    server: Server,
    idle: Mutex<Vec<Connection>>,
    room: Semaphore,
    limit: AtomicUsize,
    live: AtomicUsize,
}

impl Pool {
    pub fn new(server: Server) -> Self {
        // A pool of no connections can do no work, so keep one.
        let limit = server.connections.max(1);

        Self {
            server,
            idle: Mutex::new(Vec::new()),
            room: Semaphore::new(limit),
            limit: AtomicUsize::new(limit),
            live: AtomicUsize::new(0),
        }
    }

    pub fn server(&self) -> &Server {
        &self.server
    }

    /// How many connections the pool may open now. (§3.1)
    pub fn limit(&self) -> usize {
        self.limit.load(Ordering::Relaxed)
    }

    /// How many connections are open.
    pub fn live(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }

    /// How many connections wait for work.
    pub fn idle(&self) -> usize {
        hold(&self.idle).len()
    }

    /// Take a connection. Wait when every connection is busy.
    pub async fn take(&self) -> Result<Held<'_>> {
        loop {
            let Ok(permit) = self.room.acquire().await else {
                return Err(Error::Closed);
            };

            if let Some(connection) = hold(&self.idle).pop() {
                return Ok(self.give(connection, permit));
            }

            let error = match self.open().await {
                Ok(connection) => return Ok(self.give(connection, permit)),
                Err(error) => error,
            };

            // §3.1: a server that refuses a connection wants fewer of
            // them. Give this permit up, and try again with one less.
            // A pool with no connection at all can only give the error.
            if !crowded(&error) || self.live() == 0 || !self.narrow() {
                return Err(error);
            }
            permit.forget();
        }
    }

    /// Say goodbye on each connection that waits.
    pub async fn close(&self) {
        let waiting: Vec<Connection> = hold(&self.idle).drain(..).collect();

        for mut connection in waiting {
            let _ = connection.logout().await;
            self.live.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Open a connection, and log in on it.
    async fn open(&self) -> Result<Connection> {
        let mut connection = Connection::open(
            &self.server.host,
            self.server.port,
            self.server.tls,
        )
        .await?;
        connection
            .login(&self.server.user, &self.server.password)
            .await?;
        // §3.3 needs these two to sync only what changed.
        connection.enable(&["CONDSTORE", "QRESYNC"]).await?;
        self.live.fetch_add(1, Ordering::Relaxed);

        Ok(connection)
    }

    /// Use one connection fewer. False when only one is left. (§3.1)
    fn narrow(&self) -> bool {
        let mut limit = self.limit.load(Ordering::Relaxed);

        loop {
            if limit <= 1 {
                return false;
            }

            match self.limit.compare_exchange(
                limit,
                limit - 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(seen) => limit = seen,
            }
        }
    }

    fn give<'a>(
        &'a self,
        connection: Connection,
        permit: SemaphorePermit<'a>,
    ) -> Held<'a> {
        Held {
            pool: self,
            connection: Some(connection),
            permit: Some(permit),
            keep: true,
        }
    }
}

/// A connection that a task holds, and gives back when it ends.
pub struct Held<'a> {
    pool: &'a Pool,
    connection: Option<Connection>,
    permit: Option<SemaphorePermit<'a>>,
    keep: bool,
}

impl Held<'_> {
    /// Do not give this connection back. Use it after an error.
    pub fn retire(&mut self) {
        self.keep = false;
    }
}

impl std::ops::Deref for Held<'_> {
    type Target = Connection;

    fn deref(&self) -> &Connection {
        self.connection.as_ref().expect("a connection that is held")
    }
}

impl std::ops::DerefMut for Held<'_> {
    fn deref_mut(&mut self) -> &mut Connection {
        self.connection.as_mut().expect("a connection that is held")
    }
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        if let Some(connection) = self.connection.take() {
            if self.keep {
                hold(&self.pool.idle).push(connection);
            } else {
                self.pool.live.fetch_sub(1, Ordering::Relaxed);
            }
        }

        // The permit goes back here, and another task can start.
        self.permit.take();
    }
}

impl std::fmt::Debug for Held<'_> {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("Held")
            .field("connection", &self.connection)
            .field("keep", &self.keep)
            .finish()
    }
}

/// True when the error says the server has no room. (§3.1)
///
/// A bad password is not one of these. A pool that narrows for it
/// would hide the real reason from the user.
fn crowded(error: &Error) -> bool {
    matches!(error, Error::Refused(_) | Error::Io(_) | Error::Closed)
}

/// Hold a lock, and go on after a task stopped while it held one.
fn hold<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(|held| held.into_inner())
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_a_pool_never_opens_more_than_its_count` | model-based | §3.1 lets the user name a limit. A pool above it makes a server refuse the account. |
    //! | `prop_a_pool_gives_a_connection_to_every_task` | model-based | A pool that loses a permit stops a sync in the middle, and the mail never arrives. |

    use std::{sync::Arc, time::Duration};

    use hegel::{TestCase, generators as gs};

    use super::*;
    use crate::{
        fake::{FakeFolder, FakeMessage, FakeServer, Plan},
        stream::PLAIN_PORT,
        token::Token,
    };

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    fn a_plan() -> Plan {
        Plan::new().with(
            FakeFolder::new("INBOX")
                .with(FakeMessage::new(1, "Subject: one\r\n\r\nbody\r\n")),
        )
    }

    async fn a_server() -> FakeServer {
        FakeServer::start(a_plan()).await.unwrap()
    }

    fn a_pool(server: &FakeServer, connections: usize) -> Pool {
        Pool::new(
            Server::at("127.0.0.1", server.port(), false)
                .with_login("me", "secret")
                .with_connections(connections),
        )
    }

    fn noop() -> Vec<Token> {
        vec![Token::Atom("NOOP".into())]
    }

    fn run<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    // -----------------------------------------------------------------
    // The server.
    // -----------------------------------------------------------------

    #[test]
    fn a_server_has_tls_and_the_usual_port() {
        let server = Server::new("mail.example.com");

        assert_eq!(server.port, TLS_PORT);
        assert!(server.tls);
        assert_eq!(server.connections, CONNECTIONS);
    }

    #[test]
    fn a_server_without_tls_has_the_other_port() {
        let server = Server::at("mail.example.com", PLAIN_PORT, false);

        assert_eq!(server.port, PLAIN_PORT);
        assert!(!server.tls);
    }

    #[test]
    fn a_pool_of_no_connections_still_holds_one() {
        let pool =
            Pool::new(Server::new("mail.example.com").with_connections(0));

        assert_eq!(pool.limit(), 1);
    }

    // -----------------------------------------------------------------
    // Take, and give back.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn a_pool_opens_a_connection_and_logs_in() {
        let server = a_server().await;
        let pool = a_pool(&server, 2);
        let mut held = pool.take().await.unwrap();
        let answer = held.run(&noop()).await.unwrap();

        assert_eq!(answer.state, crate::connection::State::Ok);
        assert_eq!(pool.live(), 1);
        assert!(held.can("IMAP4rev1"));
    }

    #[tokio::test]
    async fn a_pool_gives_the_same_connection_back() {
        let server = a_server().await;
        let pool = a_pool(&server, 4);

        for _ in 0..3 {
            let mut held = pool.take().await.unwrap();
            held.run(&noop()).await.unwrap();
        }

        assert_eq!(pool.idle(), 1);
        assert_eq!(pool.live(), 1);
        assert_eq!(server.seen().connections, 1);
    }

    #[tokio::test]
    async fn a_connection_that_is_retired_never_comes_back() {
        let server = a_server().await;
        let pool = a_pool(&server, 4);
        {
            let mut held = pool.take().await.unwrap();
            held.retire();
        }

        assert_eq!(pool.idle(), 0);
        assert_eq!(pool.live(), 0);
    }

    #[tokio::test]
    async fn a_pool_that_closes_says_goodbye() {
        let server = a_server().await;
        let pool = a_pool(&server, 2);
        drop(pool.take().await.unwrap());
        pool.close().await;

        assert_eq!(pool.idle(), 0);
        assert_eq!(pool.live(), 0);
        assert!(
            server
                .seen()
                .commands
                .iter()
                .any(|line| line.contains("LOGOUT"))
        );
    }

    // -----------------------------------------------------------------
    // The count of connections. (§3.1)
    // -----------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_pool_never_opens_more_than_the_count() {
        let server = a_server().await;
        let pool = Arc::new(a_pool(&server, 2));
        let mut tasks = Vec::new();

        for _ in 0..8 {
            let pool = Arc::clone(&pool);
            tasks.push(tokio::spawn(async move {
                let mut held = pool.take().await.unwrap();
                held.run(&noop()).await.unwrap();
                tokio::time::sleep(Duration::from_millis(15)).await;
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }

        assert!(server.seen().most_open <= 2, "{:?}", server.seen());
        assert_eq!(pool.live(), 2);
    }

    #[tokio::test]
    async fn a_pool_uses_fewer_connections_when_the_server_refuses() {
        let server = FakeServer::start(a_plan().max_connections(1))
            .await
            .unwrap();
        let pool = a_pool(&server, 3);
        let first = pool.take().await.unwrap();

        let waited =
            tokio::time::timeout(Duration::from_millis(250), pool.take()).await;

        assert!(waited.is_err(), "the pool must wait, not open a third");
        assert_eq!(pool.limit(), 1);

        drop(first);
        assert!(pool.take().await.is_ok());
        assert_eq!(server.seen().connections, 3);
    }

    #[tokio::test]
    async fn a_pool_that_cannot_open_one_connection_gives_the_error() {
        let server = FakeServer::start(a_plan().max_connections(0))
            .await
            .unwrap();
        let pool = a_pool(&server, 2);
        let error = pool.take().await;

        assert!(matches!(error, Err(Error::Refused(_))), "{error:?}");
        assert_eq!(pool.limit(), 2);
    }

    #[tokio::test]
    async fn a_bad_password_never_makes_the_pool_narrow() {
        let server = a_server().await;
        let pool = Pool::new(
            Server::at("127.0.0.1", server.port(), false)
                .with_login("me", "wrong")
                .with_connections(3),
        );
        let error = pool.take().await;

        assert!(matches!(error, Err(Error::No(_))), "{error:?}");
        assert_eq!(pool.limit(), 3);
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    #[hegel::test(test_cases = 25)]
    fn prop_a_pool_never_opens_more_than_its_count(tc: TestCase) {
        let count = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
        let tasks = tc.draw(gs::integers::<usize>().min_value(1).max_value(10));

        let (most, live) = run(async move {
            let server = a_server().await;
            let pool = Arc::new(a_pool(&server, count));
            let mut held = Vec::new();

            for _ in 0..tasks {
                let pool = Arc::clone(&pool);
                held.push(tokio::spawn(async move {
                    let mut one = pool.take().await.unwrap();
                    one.run(&noop()).await.unwrap();
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }));
            }
            for task in held {
                task.await.unwrap();
            }

            (server.seen().most_open, pool.live())
        });

        assert!(most <= count, "{most} connections, {count} allowed");
        assert!(live <= count);
    }

    #[hegel::test(test_cases = 25)]
    fn prop_a_pool_gives_a_connection_to_every_task(tc: TestCase) {
        let count = tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
        let tasks = tc.draw(gs::integers::<usize>().min_value(1).max_value(12));

        let done = run(async move {
            let server = a_server().await;
            let pool = Arc::new(a_pool(&server, count));
            let mut held = Vec::new();

            for _ in 0..tasks {
                let pool = Arc::clone(&pool);
                held.push(tokio::spawn(async move {
                    let mut one = pool.take().await.unwrap();

                    one.run(&noop()).await.is_ok()
                }));
            }

            let mut done = 0;
            for task in held {
                if task.await.unwrap() {
                    done += 1;
                }
            }

            done
        });

        assert_eq!(done, tasks);
    }
}
