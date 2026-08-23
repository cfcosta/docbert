# mailbert — hybrid search for your mail

## Why this file exists

docbert searches _local collections_: a directory that you add, sync, and search. rustbert searches _published Rust crates_ that it fetches on demand. Neither model is correct for mail.

Mail is different in four ways. It arrives from a remote server that speaks IMAP. It has strong structure (sender, recipients, date, folder, flags) that the user wants to filter on. It groups into threads. And a reply contains a copy of everything before it, which destroys naive full-text ranking.

`mailbert` is the third tool in this workspace. It downloads mail from IMAP, keeps it in a local store, and gives hybrid search over it with a notmuch-like query language. It uses `docbert-core` as a library for the embedding stack, but it owns its own index schema.

The primary use is: **"find that message, when I remember what it was about but not what it said."**

## Primary functions

1. mailbert downloads mail from one or more IMAP accounts. Speed is a design goal, not an afterthought. offlineimap is slow, and mailbert must not be.
2. mailbert gives hybrid search (BM25 with Tantivy, ColBERT with docbert-core) over the full text of the mail.
3. mailbert gives a query language with the operators that notmuch users expect: `from:`, `to:`, `subject:`, `tag:`, `date:`, and boolean operators with parentheses.
4. mailbert resolves a sender name to an address with a contacts table that it builds during sync. `from:caina` finds `Cainã Costa <me@cfcosta.com>`.
5. mailbert groups the results into threads, but it ranks messages.
6. mailbert gives local tags and saved searches. IMAP folders cannot do this.
7. mailbert is one binary with a CLI and an MCP server.

## Functions not included

1. **mailbert does not send mail.** It is a search tool and a downloader. Use your MUA to write mail.
2. **mailbert does not write to the IMAP server.** The sync is a download-only mirror. mailbert never sets a flag, never moves a message, and never expunges. As a result, a defect in mailbert cannot delete your mail.
3. **mailbert does not support OAuth2.** Accounts that refuse a password are not supported. This keeps the initial version small.
4. **mailbert does not keep a maildir.** The local store is its own. The `export` command makes a maildir when an MUA must read the mail. See §4.3.
5. **mailbert does not decrypt encrypted mail during indexing.** See §5.4.
6. **mailbert does not read notmuch's database.** The two tools are independent.

## 1. The mailbert tool

mailbert is a different binary in this workspace, with its own CLI, MCP server, and data directory. It uses `docbert-core` for the model, the embeddings, and the PLAID index only. It does **not** use `docbert_core::SearchIndex`, because the docbert schema has no field for a sender or a date. See §6.1.

```text
mailbert (binary)
   ├──► docbert-core (ModelManager, EmbeddingDb, plaid, chunking)
   │       └──► docbert-plaid, docbert-pylate
   └──► its own Tantivy schema, IMAP client, and MIME pipeline
```

### 1.1 Data directory

```text
~/.local/share/mailbert/        # or $XDG_DATA_HOME/mailbert/
├── meta.db                     # messages, threads, contacts, tags, sync state
├── blobs.db                    # raw RFC822 bytes, one entry for each message
├── embeddings.db               # ColBERT token embeddings for each chunk
├── plaid.idx                   # PLAID multi-vector index
└── tantivy/                    # lexical index with the mail schema
```

`MAILBERT_DATA_DIR` or `--data-dir` changes the location.

### 1.2 Configuration file

The configuration is a TOML file at `$XDG_CONFIG_HOME/mailbert/config.toml`. Accounts are an array of tables.

```toml
[[account]]
name             = "work"
host             = "imap.fastmail.com"
port             = 993                       # default 993
user             = "me@work.example"
password_command = "pass show mail/work"     # or password_file, or password
folders          = ["INBOX", "Archive", "Sent"]
exclude          = ["Trash", "Junk"]
connections      = 8                         # parallel IMAP connections

[[account]]
name          = "personal"
host          = "imap.gmail.com"
user          = "me@gmail.com"
password_file = "~/.secrets/gmail"
all_folders   = true

[search]
count         = 20
recency_half_life_days = 180

[view]
theme  = "base16-ocean.dark"
width  = 100
```

Three credential fields are available, and mailbert reads them in this order: `password_command`, then `password_file`, then `password`. `password_command` runs a shell command and reads the first line of its output, the same as isync. `password_file` reads a file, and mailbert gives a warning if the mode is not `0600`. `password` holds the value directly, and mailbert always gives a warning.

## 2. User-visible surface

### 2.1 CLI

```bash
# Sync
mailbert sync                        # all accounts, newest messages first
mailbert sync work                   # one account
mailbert sync --watch                # stay open, IMAP IDLE for new mail
mailbert sync --full                 # ignore the sync state and re-scan
mailbert sync --dry-run              # show the plan

# Search
mailbert search "the deposit for the apartment"        # hybrid
mailbert search "from:bob and date:2026-01-01..now"
mailbert search "tag:todo and not is:read" --sort date
mailbert ksearch "invoice 88213"                       # BM25 only, no model load
mailbert search "..." -v                               # add the matched snippet
mailbert search "..." --json

# Read
mailbert get a3f9                    # decoded text of one message
mailbert view a3f9                   # rendered, with color
mailbert thread a3f9                 # each message of the thread

# Tags and saved searches
mailbert tag +todo a3f9
mailbert tag -todo +done a3f9 b721
mailbert saved add unread-work "account:work and not is:read"
mailbert saved list
mailbert search saved:unread-work

# Other
mailbert contacts caina              # what `from:caina` resolves to
mailbert export "tag:todo" ~/mail/todo     # a maildir for your MUA
mailbert status                      # the counts of the store and the index
mailbert mcp                         # stdio MCP server
```

`search` and `ksearch` are two commands, and not one command with a flag, because `ksearch` never loads the ColBERT model. This makes it fast enough for a shell loop or a fuzzy-finder.

### 2.2 MCP tools

The MCP server holds the model in memory, because the process is long-lived. This removes the cold-start cost that the CLI pays for each hybrid search.

| Tool          | Function                                    |
| ------------- | ------------------------------------------- |
| `search`      | Hybrid search with the full query language. |
| `bm25_search` | Lexical search only.                        |
| `get`         | The text of one message.                    |
| `thread`      | Each message of one thread, in order.       |
| `contacts`    | Address resolution for a name.              |
| `tag`         | Add or remove tags on a message.            |
| `status`      | Index health.                               |

`tag` is the only tool that writes. It writes to mailbert's own tag table, and never to the IMAP server.

## 3. The IMAP downloader

Speed is the design goal for this component. The first sync of a 60,000-message account must complete in minutes, and not hours.

### 3.1 Connection strategy

1. **Parallel connections.** Each account opens `connections` sockets (default 8). Each connection takes a folder from a queue. Servers have limits (Gmail permits 15, some servers permit 4), so the value is configurable and mailbert reduces it when the server refuses a connection.
2. **`COMPRESS=DEFLATE`** when the server announces it. Mail is text and compresses well.
3. **Pipelining.** mailbert sends the next command before the previous response arrives. This is where offlineimap loses most of its time.
4. **Batched `UID FETCH`.** Messages come in ranges of a few hundred, and not one at a time.
5. **`CONDSTORE` and `QRESYNC`** when the server announces them. These make the second sync a small delta and not a full UID scan.

### 3.2 Fetch order

mailbert fetches complete messages, ordered newest first across each folder of each account. Recent mail is what you search, so it becomes usable early. The 2019 archive fills in last.

The alternative (headers first, bodies later) makes the mailbox searchable in seconds. But it also makes search results change while the sync runs, and that is confusing. mailbert does not do this.

### 3.3 Incremental sync

For each (account, folder), mailbert keeps `UIDVALIDITY`, `UIDNEXT`, and the `HIGHESTMODSEQ` when `CONDSTORE` is available. A second sync fetches only the UIDs above `UIDNEXT`. A change of `UIDVALIDITY` causes a full re-scan of that folder, but the local blobs stay, because the identity of a message does not come from its UID. See §4.1.

A message that is no longer on the server stays in the local store, and mailbert marks it `is:gone`. Mail that you deleted is still searchable. This is a feature of a mirror, and not a defect.

### 3.4 Failure

The sync is resumable. mailbert commits its state for each batch, so an interrupted sync continues from the last batch and does not start again.

## 4. Storage and identity

### 4.1 Message identity

The identity of a message is `blake3(normalized Message-ID)`, truncated to 16 hex characters. Normalization removes the angle brackets and makes the domain lowercase. When a message has no `Message-ID`, mailbert uses `blake3(Date + From + Subject + body)` instead.

The identity does **not** come from the UID, the folder, or a file path. This is the important difference from docbert, where `DocumentId::new(collection, relative_path)` derives the identity from the path. A path-derived identity is wrong for mail, because the same message is in many folders and its UID is different in each one.

The CLI accepts a unique prefix of the identity, the same as git. `mailbert get a3f9` is sufficient when no other message starts with `a3f9`.

### 4.2 One message, many locations

Each message is one entry in the store. The `account` and `folder` fields are multi-valued. A Gmail message that is in `INBOX`, `[Gmail]/All Mail`, and two label folders is one entry with four folder values.

As a result:

- The store does not hold the same text 4 times.
- mailbert pays for the embedding of that message one time.
- Search gives one result for the message, and not four.
- `folder:INBOX` matches the entry, because one of its folder values is `INBOX`.

The same rule applies across accounts. A message sent to your work address and your personal address is one entry with two account values. This also means that threads span accounts, which is the correct behavior when you reply from a different address than the one that received the mail.

### 4.3 Export on demand

The local store is not a maildir, so an MUA cannot read it. `mailbert export <query> <dir>` writes a maildir for the messages that a query matches. This lets you open a result set in mutt, aerc, or neomutt.

The export writes the bytes, and not a symlink. §4.2 keeps the bytes in LMDB, so there is no file behind a message to point a link at.

The name of each message is its identity from §4.1, and then `.mailbert:2,` and the flags. A second export of the same query removes the copy that the first one wrote, so the maildir holds one copy of each message. A file that has no `.mailbert:2,` in its name is the mail of another tool, and the export leaves it alone.

The export writes every message that the query matches, and never a page of them. A reader who asks for 300 messages and receives 100 has no way to see that 200 are away.

## 5. The content pipeline

### 5.1 MIME and body selection

`mail-parser` decodes the MIME structure, the transfer encoding, and the character set. For a `multipart/alternative` message, mailbert uses the `text/plain` part. When there is no plain part, mailbert converts the `text/html` part to text. Approximately 80% of mail has a plain part, so the converter runs on a minority of messages.

The raw bytes always stay in `blobs.db`, so `view` can render the HTML even when the index holds the converted text.

### 5.2 Quote and signature removal

This is the most important step for search quality. Message 40 of a thread contains 39 copies of the text before it. If mailbert indexes that text:

- The IDF of the BM25 index collapses, because each term looks common.
- Each message of the thread matches each query about the thread.
- The embeddings of the messages of a thread converge, and the semantic leg loses its precision.

mailbert removes quoted blocks and signatures from the **indexed** text, and keeps the raw bytes for display. The rules are:

1. Lines that start with `>`, and the block that follows an attribution line (`On <date>, <person> wrote:`).
2. The Outlook separator `-----Original Message-----` and the text after it.
3. The signature separator `-- ` (with the space) and the text after it.
4. Corporate footers that a per-account regular expression identifies.

A message that is only a quote (a forward with no comment) keeps its quoted text, because an empty document is worse than a noisy one.

### 5.3 Attachments

mailbert keeps each attachment, because an offline mirror that is missing its attachments is not a mirror. The index holds the filename and the MIME type from `BODYSTRUCTURE`, and not the contents. `has:attachment` and `attachment:invoice*.pdf` work. The text inside a PDF does not.

### 5.4 Encrypted mail

mailbert never decrypts during indexing. An encrypted body stays as its ciphertext, mailbert marks the message `is:encrypted`, and only the headers are searchable. `mailbert view` runs `gpg` when you open the message.

This is deliberate. The index is a plaintext file, and its backup is also a plaintext file. If mailbert indexed the decrypted text, it would defeat the encryption for each message that your correspondents chose to encrypt.

### 5.5 Threading

mailbert uses the JWZ algorithm on `References` and `In-Reply-To`. When that chain breaks, mailbert merges two threads on a normalized subject **only** if two more conditions are true:

1. The participants of the two threads overlap.
2. The messages are inside a time window (default 30 days).

The constraints prevent the failure mode of subject-only threading, where two unrelated "Re: quick question" threads become one.

### 5.6 Contacts

mailbert knows each address that its mail carries, each display name for that address, and how often you and that address write to each other. `mailbert contacts caina` shows what a name resolves to.

The book comes from the store, and not from the sync. §4.2 keeps every message in the store, so one pass over the store gives the book. A book that the store makes never falls behind the mail that the store holds.

A message that one of your addresses sent counts as outbound for each address on it. Every other message counts as inbound. mailbert reads your addresses from the `user` of each account of §1.2. A login that holds no `@` is a name, and not an address, so a configuration that gives only such logins makes every message inbound.

At query time, `from:caina` becomes a set of addresses, ordered by that frequency. The filter that reaches the index is an exact match on an address set, so it stays fast. This is better than fuzzy matching on a text field, because you can see what the expansion did, and because `from:sam` cannot quietly include `samsung`.

## 6. The index

### 6.1 The schema

mailbert owns its Tantivy schema. This is the primary architectural difference from rustbert, which uses `docbert_core::SearchIndex` and its fixed schema of `doc_id`, `collection`, `path`, `title`, `body`, and `mtime`.

| Field        | Type                 | Function                                |
| ------------ | -------------------- | --------------------------------------- |
| `mid_hash`   | STRING, STORED       | The message identity from §4.1.         |
| `num_id`     | u64, STORED, FAST    | The key into the embedding database.    |
| `account`    | STRING, FAST, multi  | Account names.                          |
| `folder`     | STRING, FAST, multi  | Folder names.                           |
| `from_addr`  | STRING, FAST, multi  | Sender addresses.                       |
| `from_name`  | TEXT                 | Sender display names.                   |
| `to_addr`    | STRING, FAST, multi  | Recipient addresses (To and Cc).        |
| `subject`    | TEXT, STORED         | The subject, with a 2x boost.           |
| `body`       | TEXT                 | The text after §5.2 removes the quotes. |
| `list_id`    | STRING, FAST         | The `List-Id` header.                   |
| `date`       | u64, FAST, STORED    | Seconds since the Unix epoch.           |
| `thread_id`  | STRING, FAST, STORED | The thread from §5.5.                   |
| `flags`      | STRING, FAST, multi  | IMAP flags, tags, and states.           |
| `attachment` | TEXT, STORED         | Attachment filenames.                   |

Each field that a filter uses is `FAST`, because the filter must become a fast-field predicate and not a post-filter. See §8.2.

The `flags` field holds three kinds of term:

- The IMAP flags that the server gives (`\seen`, `\answered`, `\flagged`, `\draft`).
- The mailbert tags from §9.
- The states that no header carries: `\encrypted`, `\gone`, `\bulk`, and `\attachment`.

The states are in this field because `is:encrypted`, `is:gone`, `is:bulk`, and `has:attachment` must be fast-field predicates. A tag can never start with `\`, so a tag and a state never collide.

`is:unread` is the one question that no term answers. The field holds `\seen` or does not hold it, and `is:unread` is the negation.

### 6.2 Embeddings

mailbert embeds each message, and this includes bulk mail. At 10,000 to 100,000 messages the cost is acceptable, and nothing must be invisible to the semantic leg.

The chunking uses `docbert_core::chunking`. Each chunk gets a header preamble (`From: X | Subject: Y | Date: Z`) before the text of the chunk. This lets a query such as "the invoice from my landlord" match on the sender, and not only on the body.

## 7. The query language

### 7.1 Grammar

The parser is `chumsky`. The grammar is a pragmatic subset of notmuch:

```text
query    := or_expr
or_expr  := and_expr ("or" and_expr)*
and_expr := unary (("and")? unary)*        # adjacency means "and"
unary    := "not" unary | primary
primary  := "(" query ")" | term
term     := field ":" value | phrase | word
field    := from | to | cc | subject | body | folder | account
          | tag | is | has | date | mid | thread | list | attachment | saved
value    := word | phrase | glob | range
```

Each term that has no field prefix is free text. Free text goes to the hybrid leg. Each term that has a field prefix is a filter, and filters never go to the hybrid leg.

`is:` accepts `read`, `unread`, `flagged`, `replied`, `draft`, `encrypted`, `gone`, and `bulk`. `has:` accepts `attachment`. `saved:` expands to a saved search from §9.

A value that holds `*` or `?` is a glob. A glob matches the whole term, and `*` stands for any run of characters. A `tag:` glob matches only the tags that the store knows. This keeps a `*` away from the states of §6.1, because the same field holds the tags and the states.

A value of `from:`, `to:`, or `cc:` that holds an `@` is a whole address, and mailbert matches it exactly. A value that does not hold an `@` is a part of an address. For `from:` it is also a part of the name of the sender.

A value of `mid:` or `thread:` is a prefix of an identity from §4.1. An identity is hexadecimal, so a value that has other characters finds no message.

**Known limit.** §6.1 keeps one field for the To line and the Cc line together. `cc:` therefore finds the two lines, and it gives the same answer as `to:`. To tell the two lines apart, the schema needs a second field, and that is a change of the index format.

Deferred to a later version: `lastmod:`, `property:`, `path:` with a regular expression, and `#raw:` passthrough.

### 7.2 Errors

The parser gives its errors with `miette`. A query is short and the user retypes it immediately, so a good error message has a high value for each character of it.

```text
  × unknown filter `sender`
   ╭─[query:1:1]
 1 │ sender:bob and date:last-tuesday
   ·  ───┬──                 ───┬───
   ·     │                      ╰── this date is not a format that I know
   ·     ╰── did you mean `from`?
   ╰────
  help: the filters are: from, to, cc, subject, body, folder, account,
        tag, is, has, date, mid, thread, list, attachment, saved
```

### 7.3 Dates

The first version accepts absolute dates (`2026-08-14`), ranges with `..` (`2026-01-01..2026-06-30`), open ranges (`..2026-01-01`), the keywords `today`, `yesterday`, and `now`, and simple relative offsets (`7d`, `3w`, `6m`, `2y`). notmuch's full natural-language parser ("last friday", "two weeks ago") is a separate project, and mailbert does not have it yet.

## 8. Search and ranking

### 8.1 Two legs and fusion

The structure is the same as docbert. Tantivy gives the BM25 candidates, docbert-core's PLAID index gives the ColBERT candidates, and Reciprocal Rank Fusion combines the two ranked lists. `ksearch` runs the first leg only.

### 8.2 Filters gate both legs

This is the part that a naive implementation gets wrong. A filter is **not** a post-filter on the fused results.

If mailbert ranked first and then removed the results that do not match, then `from:bob invoice` would give nothing when Bob's invoice is at rank 300. The failure is silent, and it looks the same as a message that does not exist.

Instead, the filter becomes two things:

1. A Tantivy `BooleanQuery` on the fast fields, which the BM25 leg uses directly.
2. A document-identity allowlist, which gates the PLAID leg before it ranks.

Both legs then return 100 candidates that already match the filter, and the fusion is correct.

### 8.3 The recency prior

Mail search is recency-biased. A strong match from last week is almost always better than an equally strong match from 2017. mailbert multiplies the fused score by a time-decay factor with a configurable half-life (default 180 days).

`--sort date` removes the score order and uses the date. `--sort score` removes the decay.

### 8.4 Thread grouping

mailbert ranks messages, and then groups them. The output has one row for each thread, which shows the message that matched best and the position of that message in its thread (`[3/7]`). This prevents a 40-message thread from filling the first page with near-duplicates, and it keeps the precision of message-level ranking.

## 9. Tags and saved searches

Tags are local, mutable, and mailbert's own. They live in `meta.db`, keyed by the message identity from §4.1. A re-sync does not lose them, and a full re-index does not lose them, because the identity does not come from a path or a UID.

Tags never go to the IMAP server. Your MUA does not see them. This is the cost of a download-only mirror, and `mailbert export` is the way to get a tagged set into your MUA.

A saved search is a name for a query. `saved:name` expands inside a larger query, so a saved search composes:

```bash
mailbert saved add work-todo "account:work and tag:todo"
mailbert search "saved:work-todo and date:7d..now"
```

## 10. Output

### 10.1 `search` and `ksearch`

The default is one line for each thread:

```text
a3f9c2e1  2026-08-14  [3/7]  Alice Smith      Deposit for the apartment    (inbox todo)
b7210dd4  2026-08-02  [1/1]  landlord@ex.com  Re: Move-out inspection      (archive)
```

`-v` or `--snippet` adds a second line with the passage that matched, and the terms of the query highlighted.

### 10.2 `get`

`get` prints the decoded text of one message with minimal headers, and no color. This is the machine-readable form. It is what an agent reads, and what you pipe into another program.

### 10.3 `view`

`view` writes ANSI to stdout. It obeys `NO_COLOR`. It does not page, so you pipe it to `less -R` when you want that.

The rendering has four parts:

1. Headers with color, and the addresses resolved to the display names from §5.6.
2. Quoted blocks colored by their depth.
3. Code blocks inside the body highlighted with `syntect`.
4. HTML rendered as styled text, and not as tags.

### 10.4 JSON

Each command accepts `--json`. The shape is stable, and it is the same shape that the MCP tools return.

## 11. Crate layout

```text
crates/mailbert/          # the binary: CLI, MCP server, rendering
crates/mailbert-imap/     # the IMAP client: parallel fetch, IDLE, CONDSTORE
crates/mailbert-core/     # store, MIME pipeline, threading, index, query language
```

The IMAP client is a separate crate because it is the component that has the most risk, and it must be testable against a fake server without the rest of the tool.

New dependencies: `mail-parser` (MIME), `chumsky` (parser), `miette` (errors), `syntect` (highlighting), `html2text` (conversion), and an IMAP client. `blake3` for the identities.

## 12. Open questions

1. **Which IMAP crate?** The `imap` crate is synchronous and its maintenance is intermittent. `async-imap` is more recent. Neither has the pipelining that §3.1 needs. It is possible that mailbert must build on `imap-codec` or `imap-proto` and write its own connection layer. This is the largest unknown in the schedule, and it must be a prototype before the rest of the work starts.
2. **What is the true speed target?** "Faster than offlineimap" is not a number. A measurement of the current tools on a real account gives the baseline.
3. **How large is `blobs.db`?** 60,000 messages with attachments is 20 GB to 50 GB. LMDB has a map size that mailbert must set, and the value must grow.
4. **Does `sync --watch` do a full sync?** IDLE tells you that a folder changed, not what changed. The watch loop must do a small `CONDSTORE` delta for that folder only.
5. **Gmail's `All Mail` costs double.** Gmail shows each message twice or more. §4.2 removes the duplicate texts, but mailbert still downloads the bytes each time unless it does a `UID FETCH` of the `Message-ID` header first and then skips what it has. This is worth doing, and it conflicts with §3.2's "complete messages" rule.
