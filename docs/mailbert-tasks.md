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
- [x] **T25** `thread`, `tag`, and `saved`. (§8.4, §9)
  - [x] `thread::of` gives every message of the thread that one message is in.
  - [x] The rows come earliest first, and carry a place and a total. (§8.4)
  - [x] A message that the index does not hold is an error.
  - [x] `--json` writes the identity, the thread, and the rows. (§10.4)
  - [x] `tags::retag` puts the changes of a plan on the store, in order.
  - [x] `tags::targets` reads every identity before the first change lands.
  - [x] A plan that holds one bad identity changes nothing.
  - [x] A tag reaches the `flags` field, so `tag:` finds the message. (§6.1)
  - [x] A message that the index lost still takes a tag in the store.
  - [x] `tag` never writes to the IMAP server. (§3.3)
  - [x] `saved::add` reads the query before the store keeps it. (§9)
  - [x] A name that is already there takes the new query.
  - [x] `saved list` gives the searches by name, and `saved rm` forgets one.
  - [x] A name that `add` writes is a name that `saved:` expands. (§9)
- [x] **T26** `contacts`, and `status`. (§5.6, §10.4)
  - [x] `contacts::mine` reads the addresses of the accounts. (§1.2)
  - [x] A login that holds no `@` says nothing about direction.
  - [x] `contacts::book` reads the addresses from the store. (§5.6)
  - [x] A message that you sent counts as outbound for every address.
  - [x] The order of the store never changes the book.
  - [x] `contacts <name>` gives the addresses that the name resolves to.
  - [x] The address that you write to most comes first. (§5.6)
  - [x] `status` counts the store, the index, the vectors, and the tags.
  - [x] `status` says how far the index is behind the store. (§3.2)
  - [x] `status` names each folder, its UIDVALIDITY, and its UIDNEXT.
  - [x] `status` never speaks to the IMAP server. (§3.3)
  - [x] `status` gives the time of the last sync of each folder. (§10.4)
  - [x] `--json` writes the counts and the accounts. (§10.4)
- [x] **T27** The MCP server. (§2.2)
  - [x] `Desk` holds the store, the index, the configuration, and the model.
  - [x] The model loads on the first hybrid search, and then stays. (§2.2)
  - [x] `search` and `bm25_search` take the full query language. (§7.1)
  - [x] `bm25_search` never loads the model.
  - [x] A count gives 10 rows by default, and 100 at the most.
  - [x] `get` and `thread` take a git-style prefix. (§4.1)
  - [x] `get` never runs `gpg`, so an encrypted body stays ciphertext. (§5.4)
  - [x] `contacts` resolves a name to addresses. (§5.6)
  - [x] `tag` is the only tool that writes, and it writes to mailbert alone.
  - [x] `status` gives the counts of the store and the index. (§10.4)
  - [x] Each answer carries the text and the fields together.
  - [x] A bad query reads as the fault of the caller, and not of the server.
  - [x] Each tool schema drops the `$schema` keyword.
- [x] **T28** The log of the work. (§10.5)
  - [x] `trace::directive` turns `--verbose` into a filter.
  - [x] `MAILBERT_LOG` wins over the flag, and an empty one does not.
  - [x] The log goes to the standard error, so `--json` and MCP stay clean.
  - [x] Each IMAP command reaches the log, and no password does. (§3)
  - [x] A sync gives one span for each account and each folder. (§3.3)
  - [x] The slow steps say how many milliseconds they took.

- [x] **T29** The index catches up with the store. (§6.1)
  - [x] `MailIndex::held` gives every identity that the index holds.
  - [x] A pass writes the mail that the store holds and the index lacks.
  - [x] A pass that finds nothing behind writes nothing.
  - [x] The log says how much mail the index was behind.

- [x] **T30** Exclude a folder by the attribute that `LIST` gave. (§1.2)
  - [x] An `exclude` entry that starts with a backslash matches an attribute.
  - [x] A name in `exclude` keeps the meaning that it has now.
  - [x] `look` carries the attributes of each folder to the choice.
  - [x] The gmail account of the user leaves `\Trash` out.

## Phase 8: the pipeline does more at one time

- [x] **T31** A benchmark of the pipeline, against the fake server. (§10.5)
  - [x] The fixture builds a server with many small messages.
  - [x] The bench measures the download, the index pass, and the plan.
  - [x] The embedding seam takes a stub, so no bench needs a model.
  - [x] Each run starts from an empty store, so a number means one pass.
  - [x] Each folder holds its own mail, so no two folders make one entry.
  - [x] Two shapes run: one folder, and eight folders.

  The first numbers, from 500 messages of 4096 bytes on 8 connections.
  Run the bench again with `cargo bench -p mailbert --bench sync_pipeline`.

  | Stage                   | one folder | eight folders |
  | ----------------------- | ---------- | ------------- |
  | download                | 6.26 s     | 4.61 s        |
  | index pass              | 26.5 ms    | 27.0 ms       |
  | plan                    | 7.1 ms     | 6.8 ms        |
  | embed walk, with a stub | 2.09 s     | 2.33 s        |
  | the whole pipeline      | 8.31 s     | 6.75 s        |

  The numbers say where the time goes. The download takes 75 percent of
  the whole on one folder. Eight connections make the download only 1.36
  times as fast, and not 8 times, because each of them waits for the one
  writer of the store. The embed walk holds no model, so its 2.1 seconds
  are the store alone, at 4.2 ms for each message. The index pass and the
  plan together take less than one percent.

- [x] **T32** One write for each batch, and not one for each message. (§4.2)
  - [x] `Store::put_all` writes a batch in one transaction of each database.
  - [x] A batch that holds one message two times keeps one entry.
  - [x] A batch write leaves the store as many single writes do.
  - [x] The sink gives the whole batch to the store.

  The same run, after the batch write. The download of one folder is now
  396 times as fast.

  | Stage                   | one folder | eight folders |
  | ----------------------- | ---------- | ------------- |
  | download                | 15.8 ms    | 170 ms        |
  | index pass              | 25.0 ms    | 25.5 ms       |
  | plan                    | 6.1 ms     | 6.4 ms        |
  | embed walk, with a stub | 2.17 s     | 2.09 s        |
  | the whole pipeline      | 2.24 s     | 2.57 s        |

  The download no longer holds the pipeline. The store commits two
  transactions for each batch, and not two for each message, so 1000
  commits became 4. The embed walk is now 93 percent of the whole,
  because `semantic::record` still writes one message at a time. Eight
  folders now take more time than one, because eight tasks wait for the
  one writer. T35 and T37 take these two.

- [x] **T32b** One write for each group of the embed walk. (§6.2)
  - [x] `Store::mark_all_embedded` marks a batch in one transaction.
  - [x] A batch that marks one message two times keeps the last record.
  - [x] A batch mark leaves the store as many single marks do.
  - [x] The walk of the plan marks the whole group one time.
  - [x] The bench shares one database of embeddings, and leaks no environment.

  T32 found this one. The bench said the embed walk takes 2.1 s of the
  2.24 s of the pipeline, and it holds no model, so the time was the
  store alone.

  | Stage                   | one folder | eight folders |
  | ----------------------- | ---------- | ------------- |
  | download                | 37.3 ms    | 166 ms        |
  | index pass              | 28.4 ms    | 28.7 ms       |
  | plan                    | 6.0 ms     | 6.9 ms        |
  | embed walk, with a stub | 15.7 ms    | 17.1 ms       |
  | the whole pipeline      | 115 ms     | 211 ms        |

  The embed walk is 138 times as fast. The pipeline of 500 messages now
  takes 115 ms, and not 8.31 s. No one stage holds it now. The download,
  the index pass, and the embed walk each take about a quarter of it.
  Eight folders still take two times the time of one folder, because
  eight tasks each commit their own transactions. T35 took that.

  The bench opened one database of embeddings for each iteration, and
  docbert keeps every environment that it opens. The run then had no
  address space left, and it stopped. The stub of the model writes no
  vector, so one database for the whole run measures the same work.

- [x] **T33** The server says which UIDs it holds. (§3.2)
  - [x] `Connection::uids` asks the server for the UIDs of a folder.
  - [x] A plan holds no batch that the server has no mail for.
  - [x] A server with no answer falls back to the ranges of today.
  - [x] `UidSet::and` gives the UIDs that two sets share.
  - [x] `Job::only` cuts a plan down to the mail that the server holds.
  - [x] `Job::mostly_holes` says when the search pays for itself.
  - [x] The fake server answers a `UID SEARCH`, and can refuse one.

  A plan asks for every UID between the last sync and `UIDNEXT`. A
  folder that lost mail long ago spans a wide range of UIDs and holds
  little mail, so most of that range is holes. Each batch of holes
  costs a round trip and brings no mail.

  `UID SEARCH ALL` names the mail that the folder has. `Job::only`
  keeps the UIDs that the plan and the answer share, and drops the
  rest. The place of the folder does not move, because `done` still
  names the `UIDNEXT` that the server showed. The `CHANGEDSINCE`
  fetch stays whole, because it asks about the mail that the store
  holds, and not about what a fetch can bring. (§3.3)

  The search costs a round trip of its own, and its answer names every
  UID of the folder, so it must not run for the common sync of a few
  messages. `Job::mostly_holes` compares what the plan asks for with
  what `EXAMINE` said the folder holds, and asks for the search only
  when it saves a batch or more.

  A folder of 100 messages over a range of 60000 UIDs, on the fake
  server:

  |                                | round trips to fetch | time     |
  | ------------------------------ | -------------------- | -------- |
  | the ranges of today            | 120                  | 111.3 ms |
  | the mail that the server names | 1                    | 14.8 ms  |

  The fake server answers on loopback, where a round trip costs
  microseconds. A real server answers in tens of milliseconds, so the
  119 round trips that this drops are seconds of a sync.

  A server that refuses the search leaves the plan whole. The ranges
  still hold every UID that the folder has, so the sync reads the same
  mail, and only pays for the holes.

- [x] **T34** The socket reads while the store writes. (§3.4)
  - [x] A fetch of the next batch starts before the sink takes the last.
  - [x] The mark of a folder still follows the write of its batch.
  - [x] A sync that stops loses only the batches that are in the air.
  - [x] The fake server can answer a fetch slowly, as a real one does.

  `run` read a batch, gave it to the sink, and waited. The socket then
  did nothing while the store spoke to the disk, and the disk did
  nothing while the socket read the batch that followed.

  `run` now holds one batch back. It reads the next batch and writes
  the one that waits at the same time, and both run to the end. A write
  that fails therefore leaves no bytes of a half-read answer on the
  socket, and the connection stays whole for the pool.

  The state of a batch travels with that batch. A read that runs ahead
  must not carry the state ahead with it, or the store would say that
  it holds mail that is still in the air. (§3.4)

  This server listens on a local socket, and it answers in
  microseconds. A real server answers over a network. `Plan::slow`
  gives a fetch the cost of a round trip, because the pipeline exists
  to hide that cost. 5000 messages of 2 KiB in one folder is ten
  batches of 500:

  | round trip | one after the other | read while writing | saved |
  | ---------- | ------------------- | ------------------ | ----- |
  | 5 ms       | 172.9 ms            | 116.9 ms           | 32%   |
  | 20 ms      | 319.8 ms            | 247.1 ms           | 23%   |
  | 50 ms      | 624.3 ms            | 548.6 ms           | 12%   |

  The time that this saves settles at about 75 ms, which is what the
  store takes to write nine batches. The pipeline hides the smaller of
  the two costs behind the larger one. A fast link hides the read
  behind the write, and a slow link hides the write behind the read.

- [x] **T35** One writer takes the mail of every folder. (§4.2)

  The bench of T31 says why. `MAILBERT_BENCH_FOLDERS` sweeps the fan-out
  across folders, with 500 messages every time.

  | Folders | with `fsync` | without `fsync` |
  | ------- | ------------ | --------------- |
  | 1       | 18.9 ms      | 11.3 ms         |
  | 8       | 168.7 ms     | 4.6 ms          |

  Without `fsync`, eight folders are 2.4 times as fast as one folder,
  so the work across folders already runs at the same time. With
  `fsync`, eight folders take nine times the time. Each folder commits
  its own transactions, and the disk does the flushes one after the
  other. A sink writes three transactions for each batch: the bytes,
  the mail, and then the state of the folder.

  One writer must take the batch of every folder, merge what waits, and
  commit one time. The mail and the state of a folder must go in one
  transaction, because §3.4 asks that the state never runs ahead of the
  mail. One transaction gives that, and it costs one flush and not two.
  - [x] `Store::apply` writes the change of every folder in one transaction.
  - [x] The mail and the state of a folder go in that same transaction.
  - [x] The folders give their batches to one writer.
  - [x] The writer takes every change that already waits behind it.
  - [x] The writer holds no lock against another folder.
  - [x] Each folder still reports what it kept.
  - [x] A folder learns when its mail did not land.

  `Store::apply` takes a group of changes. It writes the bytes of the
  whole group in one transaction, and then the mail, the copies that
  went away, and the state of each folder in one more. Two flushes
  serve the group, and not two for each folder.

  `writer::Writer` gives every folder one handle. It takes the change
  that waits, and every change behind it, up to 64 of them, and gives
  the group to `Store::apply`. A folder that arrives while a commit
  runs joins the next group. Nothing waits for a timer, and the queue
  never grows without bound.

  The sink now sends one change for each batch. It sent three writes
  before: the mail, then one for each copy that went away, then the
  state of the folder.

  The bench ran both, one after the other, on one machine.

  | Folders | a transaction for each folder | one writer |
  | ------- | ----------------------------- | ---------- |
  | 1       | 34.0 ms                       | 17.4 ms    |
  | 4       | 150.3 ms                      | 40.5 ms    |
  | 8       | 237.3 ms                      | 50.9 ms    |
  | 16      | 318.7 ms                      | 89.4 ms    |

  Eight folders are 4.7 times as fast. The cost for each folder is
  gone: eight folders took seven times one folder before, and they now
  take three times. One folder is two times as fast, because the mail
  and the state of a batch now go in one transaction.

- [x] **T36** Every connection reads the same folder. (§3.1)
  - [x] A plan gives its batches to a queue that each connection drains.
  - [x] A folder of many batches uses every connection of the pool.
  - [x] The state of the folder moves only when each batch arrived.
  - [x] A folder takes no connection that it has no batch for.
  - [x] The pool gives a connection that waits, and never waits itself.
  - [x] A connection that broke never goes back to the pool.

  One folder of a mailbox can hold most of its mail. Gmail keeps every
  message in `All Mail`, so a sync of eight folders on eight
  connections leaves seven of them idle. One connection then reads
  60000 messages, and the other seven wait.

  The batches of a plan are independent, so any connection can read any
  of them. `spread` puts the batches in one queue, and every connection
  takes from it. The connections send what they read to one task. That
  task owns the state of the folder, and it is the only one that
  writes.

  The batches therefore land in the order that the server answers, and
  not in the order of the plan. `Job::after` takes the UIDs of a batch
  out of the debt, and that does not depend on the order. A property
  test mixes the order of the batches and shows that the state is the
  same.

  `hands` is a wish. The folder takes one connection and waits for it,
  because a folder with no connection can do nothing. It asks for the
  rest with `Pool::try_take`, which gives only a connection that waits
  now. A connection that another folder needs is therefore safe.

  A folder asks for no more connections than it has batches. A folder
  that is not there has one batch, and it must not open the whole pool
  before it fails.

  One connection is enough to keep the pipeline of T34. That connection
  reads the next batch while the task writes the last one.

  A store that gives an error takes no more batches. The task then
  closes the queue, so every connection stops. A batch that nobody
  takes is a round trip that brings nothing.

  The bench uses `Plan::slow`, as T34 does. 5000 messages of 2 KiB in
  one folder is ten batches of 500, and the pool holds eight:

  | round trip | 1 hand | 2 hands | 4 hands | 8 hands |
  | ---------- | ------ | ------- | ------- | ------- |
  | 5 ms       | 128 ms | 82 ms   | 72 ms   | 72 ms   |
  | 20 ms      | 283 ms | 158 ms  | 113 ms  | 87 ms   |
  | 50 ms      | 586 ms | 307 ms  | 204 ms  | 141 ms  |

  A slow link gains the most. At 50 ms, eight connections are 4.2 times
  as fast, because ten batches take two rounds and not ten. At 5 ms the
  gain stops at 1.8 times, because the work is then the parse of 10 MB
  and not the wait.

- [x] **T37** The model reads the mail as it arrives. (§6.2)
  - [x] The writer names the messages that a batch added.
  - [x] The model embeds those messages while the sync goes on.
  - [x] A sync that embeds nothing still ends well.
  - [x] A scoped plan asks for the same work as a plan of the whole store.
  - [x] A model that falls behind never holds up the download.
  - [x] The report adds what the model read to what the sweep read.

  Every sink holds a `Feed`. The sink names the messages of each batch
  to the model, and the model reads them while the connections read the
  batch that follows. `Store::embedding` reads the fingerprint of one
  message, so `semantic::plan_for` plans a batch without a walk over the
  whole store.

  The feed drops a name when the model falls behind. That costs nothing,
  because the sweep at the end of the sync walks the whole store and
  finds every message that the model did not read. The two counts go
  into one report.

  The `apart` and `along` groups of the bench measure what this saves.
  Both do the same work, on 4000 messages of 2 KiB. `apart` gives the
  whole mailbox to the model after the last folder. `along` gives the
  model each batch as it lands. The stub of the model sleeps, because
  no bench loads a model.

  | folders | download | model  | apart  | along  | gain  |
  | ------- | -------- | ------ | ------ | ------ | ----- |
  | 1       | 135 ms   | 60 ms  | 199 ms | 201 ms | 1.00x |
  | 1       | 135 ms   | 200 ms | 354 ms | 236 ms | 1.50x |
  | 1       | 135 ms   | 600 ms | 769 ms | 621 ms | 1.24x |
  | 8       | 65 ms    | 200 ms | 279 ms | 260 ms | 1.07x |

  A sync hides at most the smaller of the two stages, so the gain rises
  as the model cost meets the download cost, and falls again after it.
  A model that costs less than the download hides behind nothing.

  A fake server on the same machine answers about 100 times faster than
  a real one. On a real mailbox the download and the model take a
  similar time, which is the middle row of the table.

- [x] **T38** The accounts sync at the same time. (§2.1)
  - [x] Each account takes its own pool, and they run together.
  - [x] One account that fails does not stop another.
  - [x] The report names the accounts in the order of the configuration.
  - [x] The log says which account each line is in.

  A pass spawns one task for each account, and it joins them all. Each
  account holds a pool of its own, so no account waits for the
  connections of another one. A mailbox of four accounts costs what the
  slowest of the four costs, and not the sum of all four.

  Every account runs to its end, even after another one failed. A
  server that refuses one account then never holds back the mail of
  the accounts behind it. The pass gives the first error of the
  configuration, and not the first error that landed, so a run says
  the same thing each time.

  The reports come back in the order of the configuration, because a
  reader must see the same lines whichever server answered first.

  Two accounts on two servers that wait 120 ms on each fetch:

  | accounts | one after another | at the same time |
  | -------- | ----------------- | ---------------- |
  | 1        | 133 ms            | 137 ms           |
  | 2        | 312 ms            | 139 ms           |

- [x] **T39** A folder takes the connections that another folder gave
      back. (§3.1)
  - [x] A folder asks for a connection again while it still has batches.
  - [x] A folder that ends gives its connections to the folders that run.
  - [x] A folder that failed reads no more batches.

  T36 fixes the count of connections of a folder when that folder
  starts. Every folder asks at the same time, so each one takes a
  single connection, and none of them can take a second.

  `spread` now keeps a set of readers that grows. Each reader owns its
  connection and gives it back when the queue is empty. A folder that
  still owes batches waits for a reader to end, or for the pool to
  spare a connection, and it takes the first of the two. A folder that
  reached the count that it wants only waits for its readers.

- [ ] **T40** The model keeps up with the batches that land. (§6.2)
  - [ ] The bench says where the time of a round goes.
  - [ ] A round costs no more than the model that it holds.

  T37 hides less than the download that it runs behind. The table of
  T37 shows 1.50x where the shape of the work allows 1.68x, and the
  first row shows no gain at all. A round of the model costs more than
  the model alone, and the bench must say what that cost is.

  The small folders of a mailbox end first. Their connections go back
  to the pool, and the big folder does not ask again. That folder reads
  the rest of its mail on one connection.
