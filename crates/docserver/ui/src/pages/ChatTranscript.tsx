import { useState, type RefObject } from "react";
import { Link } from "react-router";
import Markdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import { buildDocumentTabHref, type SearchExcerpt, type SearchResult } from "../lib/api";
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

function formatExcerptRange(excerpt: SearchExcerpt): string {
  if (excerpt.start_line === excerpt.end_line) {
    return `${excerpt.start_line}`;
  }
  return `${excerpt.start_line}–${excerpt.end_line}`;
}

function ThinkingInline({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);

  if (text.trim().length === 0) {
    return null;
  }

  return (
    <div className="chat-thinking">
      <button
        type="button"
        className="chat-thinking-header"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
      >
        <span className="chat-thinking-icon">◎</span>
        <span className="chat-thinking-title">Reasoning</span>
        <span className={`chat-thinking-chevron${expanded ? " open" : ""}`}>{"\u25B8"}</span>
      </button>
      {expanded && (
        <div className="chat-thinking-body">
          <Markdown
            remarkPlugins={CHAT_MARKDOWN_REMARK_PLUGINS}
            rehypePlugins={CHAT_MARKDOWN_REHYPE_PLUGINS}
          >
            {text}
          </Markdown>
        </div>
      )}
    </div>
  );
}

function SearchResultTree({ results }: { results: SearchResult[] }) {
  if (results.length === 0) {
    return <div className="chat-tool-search-empty">No results</div>;
  }

  return (
    <div className="chat-tool-search-results">
      {results.map((result) => (
        <div key={`${result.collection}:${result.path}`} className="chat-tool-search-result">
          <div className="chat-tool-search-result-node">
            <div className="chat-tool-search-result-top">
              <div>
                <Link
                  className="chat-tool-search-result-title chat-tool-search-result-title-link"
                  to={buildDocumentTabHref(result.collection, result.path)}
                >
                  {result.title || result.path}
                </Link>
                <div className="chat-tool-search-result-path">
                  {result.collection}/{result.path}
                </div>
              </div>
            </div>
            <div className="chat-tool-search-result-meta">
              <span>#{result.rank}</span>
              <span>{result.score.toFixed(3)}</span>
            </div>
          </div>

          {result.excerpts && result.excerpts.length > 0 && (
            <div className="chat-tool-search-result-children">
              {result.excerpts.map((excerpt) => (
                <Link
                  key={`${result.collection}:${result.path}:${excerpt.start_line}:${excerpt.end_line}`}
                  className="chat-tool-search-excerpt chat-tool-search-excerpt-link"
                  to={buildDocumentTabHref(result.collection, result.path)}
                >
                  <span className="chat-tool-search-excerpt-range">
                    {formatExcerptRange(excerpt)}
                  </span>
                  <span className="chat-tool-search-excerpt-text">{excerpt.text}</span>
                </Link>
              ))}
            </div>
          )}
        </div>
      ))}
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

  return (
    <div
      className={`chat-tool-call chat-tool-call-${tone}${call.isError ? " error" : ""}`}
    >
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
      {expanded && call.result && searchResults && <SearchResultTree results={searchResults} />}
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
          <p>Ask a question and the assistant will search your documents for context.</p>
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
            <div className="chat-typing">
              <span />
              <span />
              <span />
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
