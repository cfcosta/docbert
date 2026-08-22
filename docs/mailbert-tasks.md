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
- [ ] **T13** Compile a query into a filter that gates both legs. (§8.2)
- [ ] **T14** Fusion, the recency prior, and thread grouping. (§8.1, §8.3, §8.4)

## Phase 3 — the downloader

- [ ] **T15** Scaffold `mailbert-imap`, and a fake server to test against.
- [ ] **T16** The connection pool, and the parallel fetch. (§3.1, §3.2)
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
