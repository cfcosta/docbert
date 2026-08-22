# mailbert — the task list

This file records the work that builds the tool in
[mailbert.md](mailbert.md). Each task is one commit. Each task starts
with its tests, and the tests fail before the code exists.

A task is complete when all of these are true:

- The tests pass, and the property tests are in `hegeltest`.
- `cargo clippy --all-targets` gives no warning.
- `nix fmt` made no change.
- `jj describe` recorded the change.

## Phase 1 — the core vocabulary

The types that every later phase speaks in. This phase is complete.

- [x] **T1** Scaffold `mailbert-core` and its error type. (§11)
- [x] **T2** Message identity, and git-style prefixes. (§4.1)
- [x] **T3** The configuration file, and the credential order. (§1.2)
- [x] **T4** Remove the quotes and the signatures from a body. (§5.2)
- [x] **T5** Addresses, and the contact book behind `from:`. (§5.6)
- [x] **T6** Threading with JWZ, and a constrained subject merge. (§5.5)
- [x] **T7** The date terms that the `date:` filter accepts. (§7.3)
- [x] **T8** The query language, and its diagnostics. (§7)

## Phase 2 — the store and the index

The parts that hold a message and find it again.

- [x] **T9** The MIME pipeline: bodies, attachments, and ciphertext. (§5.1, §5.3, §5.4)
- [x] **T10** The message record, and one message in many locations. (§4.2)
- [x] **T11** The store, the tags, and the saved searches. (§9)
  - [x] Two LMDB files: `meta.db` and `blobs.db`. (§1.1)
  - [x] Write a message, and absorb a copy in another folder. (§4.2)
  - [x] Keep the raw bytes unchanged, for gpg and for export. (§4.3, §5.4)
  - [x] Resolve a git-style prefix with a key scan. (§4.1)
  - [x] Tags on the identity, so a re-sync keeps them. (§9)
  - [x] Saved searches, by name. (§9)
- [x] **T12** The Tantivy schema, and the writer. (§6.1)
  - [x] The 14 fields, and a test that each filter field is `FAST`.
  - [x] The terms of the `flags` field: IMAP flags, tags, and states.
  - [x] Write a message, replace it, and remove it.
  - [x] Read a hit back, and run a query that a caller built. (§8.2)
- [x] **T13** Compile a query into a filter that gates both legs. (§8.2)
  - [x] Expand `saved:` into the query that it names, and refuse a cycle. (§9)
  - [x] Each filter becomes a Tantivy query on its own field. (§7.1)
  - [x] The free text becomes a query over the word fields, with the boost. (§6.1)
  - [x] `and`, `or`, and `not` become the occurrences of a `BooleanQuery`.
  - [x] The filter alone, and the allowlist that gates the semantic leg. (§8.2)
- [x] **T14** Fusion, the recency prior, and thread grouping. (§8.1, §8.3, §8.4)
  - [x] Reciprocal Rank Fusion over the two ranked legs. (§8.1)
  - [x] The recency prior, with the half-life of the config. (§8.3)
  - [x] `--sort date` and `--sort score`. (§8.3)
  - [x] One row for each thread, and the best message of it. (§8.4)
  - [x] The position of that message in its thread, for `[3/7]`. (§10.1)
  - [x] Read the messages of one thread from the index, in date order.

## Phase 3 — the downloader

- [x] **T15** Scaffold `mailbert-imap`, and a fake server to test against.
  - [x] The crate, its manifest, and its error type.
  - [x] The token tree of a response, and the encoder for it.
  - [x] Read one response from a stream, with the literals in it.
  - [x] Write a tagged command, with a tag that never repeats.
  - [x] The set of UIDs that a fetch names, and the text of it.
  - [x] A fake server that holds folders and messages, so a test needs no
        network.
  - [x] The fake server refuses each command that would write.
- [x] **T16** The connection pool, and the parallel fetch. (§3.1, §3.2)
  - [x] The batches of a fetch, newest first. (§3.2)
  - [x] The transport: TLS, and a plain socket for a test.
  - [x] One connection: greet, capability, login, examine, fetch, logout.
  - [x] Read a `FETCH` answer into a message.
  - [x] Pipelining: send the next command before the last answer arrives.
  - [x] The pool, and the count of connections that the config names.
  - [x] Fewer connections when the server refuses one. (§3.1)
- [ ] **T17** Incremental sync with UIDVALIDITY and CONDSTORE. (§3.3)
- [ ] **T18** Failure, and the resume after it. (§3.4)

## Phase 4 — the tool

- [ ] **T19** Scaffold the `mailbert` binary, and its CLI. (§2.1)
- [ ] **T20** `sync`, and the `--watch` mode.
- [ ] **T21** The embeddings, and the semantic leg. (§6.2)
- [ ] **T22** `search`, `ksearch`, and the output. (§10.1)
- [ ] **T23** `get`, and `view` with ANSI. (§10.2, §10.3)
- [ ] **T24** `export` to a maildir. (§4.3)
- [ ] **T25** The MCP server. (§2.2)
