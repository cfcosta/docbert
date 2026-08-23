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
- [x] **T17** Incremental sync with UIDVALIDITY and CONDSTORE. (§3.3)
  - [x] Union, difference, and the batches of a set of UIDs.
  - [x] The state of a folder: UIDVALIDITY, UIDNEXT, HIGHESTMODSEQ, and
        the UIDs that never arrived.
  - [x] The plan of one folder sync, out of the state and the view.
  - [x] A UIDVALIDITY that changed starts the folder again. (§3.3)
  - [x] CONDSTORE: ask only about the messages that changed. (§3.3)
  - [x] Run a plan on a connection, and keep each batch.
- [x] **T18** Failure, and the resume after it. (§3.4)
  - [x] The wait after a failure, and the longest wait.
  - [x] The errors that are worth another try.
  - [x] A connection that broke never goes back to the pool.
  - [x] Read the state out of the store, and continue from it.
  - [x] A sync that stops anywhere still reads every message.

## Phase 4 — the tool

- [x] **T19** Scaffold the `mailbert` binary, and its CLI. (§2.1)
  - [x] `crates/mailbert` joins the workspace, with its manifest.
  - [x] The command tree of §2.1, and the options that all commands take.
  - [x] The paths: `--data-dir`, `MAILBERT_DATA_DIR`, and the XDG defaults.
  - [x] The configuration file, a `~` in a path, and `password_command`. (§1.2)
  - [x] The words of `tag`: `+todo`, `-done`, and the identities after them.
  - [x] The error type, and the miette report for a bad query.
  - [x] Property tests for the command tree, and for the words of `tag`.
- [x] **T20** `sync`, and the `--watch` mode.
  - [x] The sync state in the store, for each account and each folder. (§1.1, §3.3)
  - [x] The sink that keeps a batch: the raw bytes, the message, and the flags. (§4.2)
  - [x] The threading pass, and the index write that follows a sync. (§5.5, §6.1)
  - [x] `sync`, `sync <account>`, and the folders that the config names. (§2.1)
  - [x] `--full`, `--dry-run`, and the counts that the command writes.
  - [x] IDLE, and the `--watch` loop that reads only what changed. (§3.1)
  - [x] Property tests: a second sync asks for nothing, and a stop resumes.
- [x] **T21** The embeddings, and the semantic leg. (§6.2)
  - [x] The passage: a preamble of `From`, `Subject`, and `Date`, and a chunk. (§6.2)
  - [x] The key of a passage, and the way back from a passage to its message.
  - [x] An encrypted message gives its preamble, and no ciphertext. (§5.4)
  - [x] The fingerprint that says which messages a pass must read again.
  - [x] The embedding pass: what the model reads, and what leaves the index.
  - [x] The PLAID index, whole on a first pass and by parts after it.
  - [x] The leg: the passages that a query finds, and the messages that own them.
  - [x] A filter gates the leg before it ranks. (§8.2)
  - [x] `sync` sweeps, and says how many messages it embedded.
- [x] **T22** `search`, `ksearch`, and the output. (§10.1)
  - [x] `search::find` runs both legs, and gives the rows of §8.4.
  - [x] `search::leg` gates the semantic leg with the filter. (§8.2)
  - [x] `ksearch` opens no brain, and loads no model. (§2.1)
  - [x] A query with no free text never asks the model.
  - [x] A search over mail that no pass embedded gives one leg, and says so.
  - [x] `search::lines` gives the identity, the day, the place in the thread,
        the sender, the subject, the folders, and the tags.
  - [x] `--snippet` gives the part of the body that holds the word.
  - [x] The columns of the text share one width. (§10.1)
  - [x] `--json` writes the query, the legs, and the rows. (§10.4)
  - [x] `crate::clock` gives the local offset that §7.1 reads dates against.
- [x] **T23** `get`, and `view` with ANSI. (§10.2, §10.3)
  - [x] `show::resolve` reads a git-style prefix of the identity. (§4.1)
  - [x] A prefix that names nothing, or names two messages, is an error.
  - [x] `show::read` gives the headers of §10.2, and the body with its
        quotes.
  - [x] The date of the header carries the hour, in the zone of the reader.
  - [x] `get` writes no escape, and never runs gpg. (§5.4)
  - [x] `get` gives the armor of an encrypted body, and never a plaintext.
  - [x] `view` colors the header names, the subject, and the quotes by depth.
  - [x] `view` colors a fenced block with syntect. (§10.3)
  - [x] `view` runs gpg on demand, and only for an encrypted body. (§5.4)
  - [x] `--json` writes the headers and the body together. (§10.4)
- [x] **T24** `export` to a maildir. (§4.3)
  - [x] `export::make` makes `cur`, `new`, and `tmp`.
  - [x] `export::name` gives the identity of §4.1, and the flags after it.
  - [x] The letters of a name are in the order of their bytes.
  - [x] A tag of §9 never becomes a letter of a maildir.
  - [x] The bytes go to `tmp` first, and then move to `cur`.
  - [x] The export writes the bytes of the server, and never one byte more.
  - [x] The export writes every match, and never a page of them.
  - [x] A second export leaves one copy of a message whose flags changed.
  - [x] The export leaves a file that another tool wrote.
  - [x] A message with no bytes counts, and does not stop the export.
  - [x] A query that is bad leaves no maildir behind.
- [ ] **T25** The MCP server. (§2.2)
