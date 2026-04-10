import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import Markdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import SearchResults from "../components/SearchResults";
import { api, type DocumentListItem, type SearchExcerpt, type SearchResult } from "../lib/api";
import DocumentPreview, { type ResolvedDocumentTarget } from "./document-preview";
import type { SelectedDocumentSummary } from "./documents-tree";
import type { ToolCallInfo, Message } from "./chat-message-codec";
import type { DisplayMessageGroup, SubagentMessage } from "./chat-message-groups";
import { buildTranscriptRenderItems } from "./chat-transcript-model";

const CHAT_MARKDOWN_REMARK_PLUGINS = [remarkGfm, remarkMath];
const CHAT_MARKDOWN_REHYPE_PLUGINS = [rehypeKatex];

type ChatTranscriptProps = {
  displayMessageGroups: DisplayMessageGroup[];
  loading: boolean;
  lastMessageRole?: Message["role"];
  bottomRef: RefObject<HTMLDivElement | null>;
  starterPrompts?: string[];
  onPickStarter?: (prompt: string) => void;
};

function isSearchExcerpt(value: unknown): value is SearchExcerpt {
  return (
    !!value &&
    typeof value === "object" &&
    typeof (value as SearchExcerpt).text === "string" &&
    typeof (value as SearchExcerpt).start_line === "number" &&
    typeof (value as SearchExcerpt).end_line === "number"
  );
}

function isSearchResult(value: unknown): value is SearchResult {
  if (!value || typeof value !== "object") {
    return false;
  }

  const result = value as SearchResult;
  return (
    typeof result.collection === "string" &&
    typeof result.path === "string" &&
    typeof result.title === "string" &&
    typeof result.rank === "number" &&
    typeof result.score === "number" &&
    typeof result.doc_id === "string" &&
    (result.excerpts === undefined ||
      (Array.isArray(result.excerpts) && result.excerpts.every(isSearchExcerpt)))
  );
}

function parseToolSearchResults(call: ToolCallInfo): SearchResult[] | null {
  if (call.name !== "search_hybrid" && call.name !== "search_semantic") {
    return null;
  }

  if (!call.result) {
    return null;
  }

  try {
    const parsed = JSON.parse(call.result);
    if (!Array.isArray(parsed)) {
      return null;
    }

    return parsed.filter(isSearchResult);
  } catch {
    return null;
  }
}

function searchDocumentKey(value: { collection: string; path: string }) {
  return `${value.collection}:${value.path}`;
}

function SearchToolResultsInline({ results }: { results: SearchResult[] }) {
  const [selectedDoc, setSelectedDoc] = useState<SelectedDocumentSummary | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [activeFragment, setActiveFragment] = useState<string | null>(null);
  const [resolverDocuments, setResolverDocuments] = useState<Record<string, DocumentListItem[]>>(
    {},
  );
  const [resolverFailures, setResolverFailures] = useState<Record<string, true>>({});
  const requestSeqRef = useRef(0);
  const pendingResolverLoadsRef = useRef<Record<string, Promise<DocumentListItem[]> | null>>({});

  useEffect(() => {
    requestSeqRef.current += 1;
    setSelectedDoc(null);
    setPreview(null);
    setActiveFragment(null);
  }, [results]);

  const ensureResolverDocuments = useCallback(
    async (collection: string) => {
      if (resolverDocuments[collection] || resolverFailures[collection]) {
        return resolverDocuments[collection] ?? null;
      }

      const pending = pendingResolverLoadsRef.current[collection];
      if (pending) {
        return pending.catch(() => null);
      }

      const promise = api.listDocuments(collection);
      pendingResolverLoadsRef.current[collection] = promise;

      try {
        const docs = await promise;
        setResolverDocuments((previous) => ({ ...previous, [collection]: docs }));
        return docs;
      } catch {
        setResolverFailures((previous) => ({ ...previous, [collection]: true }));
        return null;
      } finally {
        pendingResolverLoadsRef.current[collection] = null;
      }
    },
    [resolverDocuments, resolverFailures],
  );

  const openPreview = useCallback(
    async (result: SearchResult) => {
      const requestId = requestSeqRef.current + 1;
      requestSeqRef.current = requestId;

      setSelectedDoc({
        collection: result.collection,
        path: result.path,
        title: result.title || result.path,
        doc_id: result.doc_id,
      });
      setPreview(null);
      setActiveFragment(null);
      void ensureResolverDocuments(result.collection);

      try {
        const full = await api.getDocument(result.collection, result.path);
        if (requestSeqRef.current !== requestId) {
          return;
        }

        setSelectedDoc({
          collection: full.collection,
          path: full.path,
          title: full.title,
          doc_id: full.doc_id,
        });
        setPreview(full.content || "_No content stored._");
      } catch (error) {
        if (requestSeqRef.current !== requestId) {
          return;
        }

        setPreview(
          error instanceof Error
            ? `_Failed to load document: ${error.message}_`
            : "_Failed to load document._",
        );
      }
    },
    [ensureResolverDocuments],
  );

  const openResolvedDocument = useCallback(
    async (target: ResolvedDocumentTarget) => {
      const listedDoc = resolverDocuments[target.collection]?.find(
        (document) => document.path === target.path,
      );
      setActiveFragment(target.fragment);
      await openPreview({
        rank: 0,
        score: 0,
        doc_id: listedDoc?.doc_id ?? target.path,
        collection: target.collection,
        path: target.path,
        title: listedDoc?.title ?? target.path,
      });
      setActiveFragment(target.fragment);
    },
    [openPreview, resolverDocuments],
  );

  return (
    <div className={`chat-tool-search-preview-layout${selectedDoc ? " has-preview" : ""}`}>
      <SearchResults
        results={results}
        onOpenDocument={openPreview}
        activeDocumentKey={selectedDoc ? searchDocumentKey(selectedDoc) : null}
        variant="tool-inline"
      />
      {selectedDoc && (
        <div className="chat-tool-search-preview-shell">
          <DocumentPreview
            selectedDoc={selectedDoc}
            preview={preview}
            resolverDocuments={resolverDocuments[selectedDoc.collection] ?? []}
            activeFragment={activeFragment}
            onOpenResolvedDocument={openResolvedDocument}
          />
        </div>
      )}
    </div>
  );
}

function ThinkingInline({ text }: { text: string }) {
  if (text.trim().length === 0) {
    return null;
  }

  return (
    <div className="chat-thinking" aria-label="Assistant reasoning">
      <Markdown
        remarkPlugins={CHAT_MARKDOWN_REMARK_PLUGINS}
        rehypePlugins={CHAT_MARKDOWN_REHYPE_PLUGINS}
      >
        {text}
      </Markdown>
    </div>
  );
}

function ToolCallInline({
  call,
  tone = "root",
}: {
  call: ToolCallInfo;
  tone?: "root" | "subagent";
}) {
  const [expanded, setExpanded] = useState(false);
  const argsStr = Object.entries(call.args)
    .map(([key, value]) => `${key}: ${typeof value === "string" ? value : JSON.stringify(value)}`)
    .join(", ");
  const searchResults = parseToolSearchResults(call);
  const isSearchTool = call.name === "search_hybrid" || call.name === "search_semantic";
  const query = typeof call.args.query === "string" ? call.args.query.trim() : "";
  const searchSummary = query || argsStr || call.name;
  const searchStatus = call.result == null ? "running" : call.isError ? "error" : "done";

  if (isSearchTool) {
    return (
      <div
        className={`chat-subagent-inline chat-tool-call-search${expanded ? " expanded" : ""}${call.isError ? " error" : ""}`}
      >
        <button
          type="button"
          className="chat-subagent-header chat-subagent-toggle"
          onClick={() => setExpanded(!expanded)}
          aria-expanded={expanded}
          aria-label={`${call.name} ${searchSummary}`}
        >
          <span className="chat-subagent-icon">⌕</span>
          <span className="chat-subagent-label">Search</span>
          <code className="chat-subagent-path">{searchSummary}</code>
          <span className={`chat-subagent-status chat-subagent-status-${searchStatus}`}>
            {searchStatus}
          </span>
          <span className={`chat-subagent-chevron${expanded ? " open" : ""}`}>{"\u25B8"}</span>
        </button>
        {expanded && call.result && searchResults && (
          <SearchToolResultsInline results={searchResults} />
        )}
        {expanded && call.result && !searchResults && (
          <div className="chat-subagent-body">
            <pre className="chat-tool-call-result">{call.result}</pre>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={`chat-tool-call chat-tool-call-${tone}${call.isError ? " error" : ""}`}>
      <button
        type="button"
        className="chat-tool-call-header"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
      >
        <span className="chat-tool-call-icon">
          {call.result == null ? "\u2026" : call.isError ? "!" : "\u2713"}
        </span>
        <span className="chat-tool-call-name">{call.name}</span>
        <span className="chat-tool-call-args">{argsStr}</span>
        <span className={`chat-tool-call-chevron${expanded ? " open" : ""}`}>{"\u25B8"}</span>
      </button>
      {expanded && call.result && searchResults && (
        <SearchToolResultsInline results={searchResults} />
      )}
      {expanded && call.result && !searchResults && (
        <pre className="chat-tool-call-result">{call.result}</pre>
      )}
    </div>
  );
}

function renderMessageContent(message: Message, nestedSubagents: SubagentMessage[] = []) {
  const items = buildTranscriptRenderItems(message, nestedSubagents);
  const toolTone = message.actor?.type === "subagent" ? "subagent" : "root";

  return (
    <>
      {items.map((item) => {
        if (item.kind === "text") {
          return (
            <div key={item.key} className="chat-msg-content">
              <Markdown
                remarkPlugins={CHAT_MARKDOWN_REMARK_PLUGINS}
                rehypePlugins={CHAT_MARKDOWN_REHYPE_PLUGINS}
              >
                {item.text}
              </Markdown>
            </div>
          );
        }

        if (item.kind === "thinking") {
          return <ThinkingInline key={item.key} text={item.text} />;
        }

        if (item.kind === "subagent") {
          return <SubagentInline key={item.key} message={item.message} />;
        }

        return <ToolCallInline key={item.key} call={item.call.call} tone={toolTone} />;
      })}
    </>
  );
}

function renderMessageBody(message: Message, nestedSubagents: SubagentMessage[] = []) {
  return <>{renderMessageContent(message, nestedSubagents)}</>;
}

function SubagentInline({ message }: { message: SubagentMessage }) {
  const [expanded, setExpanded] = useState(false);
  const actor = message.actor;

  return (
    <div className="chat-subagent-inline">
      <button
        type="button"
        className="chat-subagent-header chat-subagent-toggle"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
      >
        <span className="chat-subagent-icon">◎</span>
        <span className="chat-subagent-label">File analysis</span>
        <code className="chat-subagent-path">
          {actor.collection}/{actor.path}
        </code>
        <span className={`chat-subagent-status chat-subagent-status-${actor.status}`}>
          {actor.status}
        </span>
        <span className={`chat-subagent-chevron${expanded ? " open" : ""}`}>{"\u25B8"}</span>
      </button>
      {expanded && <div className="chat-subagent-body">{renderMessageBody(message)}</div>}
    </div>
  );
}

export default function ChatTranscript({
  displayMessageGroups,
  loading,
  lastMessageRole,
  bottomRef,
  starterPrompts = [],
  onPickStarter,
}: ChatTranscriptProps) {
  return (
    <div className="chat-messages">
      {displayMessageGroups.length === 0 && (
        <div className="chat-empty">
          <div className="chat-empty-icon">
            <svg
              width="48"
              height="48"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
          </div>
          <h3>Start a conversation</h3>
          <p>
            Ask across your notes, docs, and PDFs. docbert will search first, then answer inside the
            same thread.
          </p>
          {starterPrompts.length > 0 && onPickStarter && (
            <div className="chat-empty-starters" aria-label="Suggested prompts">
              {starterPrompts.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  className="chat-starter-chip"
                  onClick={() => onPickStarter(prompt)}
                >
                  {prompt}
                  <span aria-hidden="true">…</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {displayMessageGroups.map(({ message, nestedSubagents }) => {
        const isSubagent = message.actor?.type === "subagent";
        return (
          <div
            key={message.id}
            className={`chat-msg chat-msg-${message.role}${isSubagent ? " chat-msg-subagent" : ""}`}
          >
            <div className="chat-msg-bubble">
              {isSubagent ? (
                <SubagentInline message={message as SubagentMessage} />
              ) : (
                renderMessageBody(message, nestedSubagents)
              )}
            </div>
          </div>
        );
      })}

      {loading && lastMessageRole !== "assistant" && (
        <div className="chat-msg chat-msg-assistant">
          <div className="chat-msg-bubble">
            <div className="chat-typing" role="status" aria-live="polite">
              <span className="sr-only">Assistant is thinking</span>
              <span className="chat-typing-dot chat-typing-dot-1" aria-hidden="true" />
              <span className="chat-typing-dot chat-typing-dot-2" aria-hidden="true" />
              <span className="chat-typing-dot chat-typing-dot-3" aria-hidden="true" />
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
