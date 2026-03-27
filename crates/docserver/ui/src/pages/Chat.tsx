import { useState, useRef, useEffect, useCallback } from "react";
import { Type, getModel, stream } from "@mariozechner/pi-ai";
import type { Context, Tool, UserMessage, ToolResultMessage, Message as PiMessage } from "@mariozechner/pi-ai";
import { api } from "../lib/api";
import type { SearchResult } from "../lib/api";
import "./Chat.css";

interface ToolCallInfo {
  name: string;
  args: Record<string, unknown>;
  result?: string;
  isError?: boolean;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SearchResult[];
  toolCalls?: ToolCallInfo[];
}

// ── Tool definitions for pi-ai ──

const tools: Tool[] = [
  {
    name: "search_semantic",
    description:
      "Search the document store using semantic (ColBERT) search. Best for meaning-based queries where wording may differ from the target documents.",
    parameters: Type.Object({
      query: Type.String({ description: "The search query" }),
      count: Type.Optional(
        Type.Number({ description: "Number of results to return (default 5)" }),
      ),
    }),
  },
  {
    name: "search_hybrid",
    description:
      "Search the document store using hybrid BM25 + semantic search. Best when the query shares keywords with the target documents. Faster on large collections.",
    parameters: Type.Object({
      query: Type.String({ description: "The search query" }),
      collection: Type.Optional(
        Type.String({ description: "Restrict search to this collection" }),
      ),
      count: Type.Optional(
        Type.Number({ description: "Number of results to return (default 5)" }),
      ),
    }),
  },
  {
    name: "document_get",
    description:
      "Retrieve the full content of a specific document by collection and path.",
    parameters: Type.Object({
      collection: Type.String({ description: "The collection name" }),
      path: Type.String({
        description: "The document path within the collection",
      }),
    }),
  },
];

// ── Tool execution ──

async function executeTool(
  name: string,
  args: Record<string, unknown>,
): Promise<{ text: string; sources?: SearchResult[] }> {
  switch (name) {
    case "search_semantic": {
      const res = await api.search({
        query: args.query as string,
        mode: "semantic",
        count: (args.count as number) ?? 5,
      });
      return {
        text: JSON.stringify(res.results, null, 2),
        sources: res.results,
      };
    }
    case "search_hybrid": {
      const res = await api.search({
        query: args.query as string,
        mode: "hybrid",
        collection: args.collection as string | undefined,
        count: (args.count as number) ?? 5,
      });
      return {
        text: JSON.stringify(res.results, null, 2),
        sources: res.results,
      };
    }
    case "document_get": {
      const doc = await api.getDocument(
        args.collection as string,
        args.path as string,
      );
      return {
        text: `# ${doc.title}\n\nCollection: ${doc.collection}\nPath: ${doc.path}\nDoc ID: ${doc.doc_id}\n\n${doc.content}`,
      };
    }
    default:
      return { text: `Unknown tool: ${name}` };
  }
}

const SYSTEM_PROMPT = `You are a helpful assistant with access to a document store. You can search for documents and retrieve their content using the provided tools.

When the user asks a question:
1. Use search_hybrid or search_semantic to find relevant documents
2. Use document_get to retrieve the full content of promising results
3. Answer the user's question based on the document content

If no relevant documents are found, say so and suggest what the user might want to ingest.`;

const MAX_TOOL_ROUNDS = 10;

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const settings = await api.getLlmSettings();

      if (!settings.provider || !settings.model || !settings.api_key) {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content:
              "No LLM provider configured. Go to **Settings** to select a provider, model, and API key.",
          },
        ]);
        return;
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const model = getModel(settings.provider as any, settings.model as any);
      const controller = new AbortController();
      abortRef.current = controller;

      const userPiMsg: UserMessage = {
        role: "user",
        content: text,
        timestamp: Date.now(),
      };

      const piContext: Context = {
        systemPrompt: SYSTEM_PROMPT,
        messages: [userPiMsg],
        tools,
      };

      const assistantId = crypto.randomUUID();
      const allSources: SearchResult[] = [];
      const allToolCalls: ToolCallInfo[] = [];

      // Add empty assistant message to stream into.
      setMessages((prev) => [
        ...prev,
        { id: assistantId, role: "assistant", content: "" },
      ]);

      // Agentic tool loop: stream, handle tool calls, continue.
      for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
        const s = stream(model, piContext, {
          apiKey: settings.api_key,
          abortSignal: controller.signal,
        });

        for await (const event of s) {
          if (event.type === "text_delta") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content + event.delta }
                  : m,
              ),
            );
          }
          if (event.type === "error") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      content:
                        m.content ||
                        `Error from ${settings.provider}: ${event.error}`,
                    }
                  : m,
              ),
            );
          }
        }

        // Get the final assistant message.
        const result = await s.result();
        piContext.messages.push(result);

        // Check for tool calls.
        const toolCalls = result.content.filter(
          (b) => b.type === "toolCall",
        );

        if (toolCalls.length === 0) {
          // No tool calls — done.
          break;
        }

        // Execute tool calls and add results.
        for (const call of toolCalls) {
          const callArgs = call.arguments as Record<string, unknown>;
          const callInfo: ToolCallInfo = { name: call.name, args: callArgs };

          try {
            const toolResult = await executeTool(call.name, callArgs);

            if (toolResult.sources) {
              allSources.push(...toolResult.sources);
            }

            callInfo.result = toolResult.text;

            const resultMsg: ToolResultMessage = {
              role: "toolResult",
              toolCallId: call.id,
              toolName: call.name,
              content: [{ type: "text", text: toolResult.text }],
              isError: false,
              timestamp: Date.now(),
            };
            piContext.messages.push(resultMsg as PiMessage);
          } catch (err) {
            const errText = err instanceof Error ? err.message : "unknown error";
            callInfo.result = `Error: ${errText}`;
            callInfo.isError = true;

            const resultMsg: ToolResultMessage = {
              role: "toolResult",
              toolCallId: call.id,
              toolName: call.name,
              content: [{ type: "text", text: `Error: ${errText}` }],
              isError: true,
              timestamp: Date.now(),
            };
            piContext.messages.push(resultMsg as PiMessage);
          }

          allToolCalls.push(callInfo);
        }

        // Update sources and tool calls on the message after each round.
        const uniqueSources = deduplicateSources(allSources);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, sources: uniqueSources.length > 0 ? uniqueSources : m.sources, toolCalls: [...allToolCalls] }
              : m,
          ),
        );

        // Continue to next round — the model will see tool results and respond.
      }

      abortRef.current = null;
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `Something went wrong: ${err instanceof Error ? err.message : "unknown error"}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, messages]);

  return (
    <div className="chat-page">
      <div className="chat-header-wrap">
        <header className="chat-header">
          <h2>Chat</h2>
          <p className="chat-subtitle">Ask questions about your documents</p>
        </header>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
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
              Ask a question and the assistant will search your documents for
              context.
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`chat-msg chat-msg-${msg.role}`}>
            <div className="chat-msg-bubble">
              <div className="chat-msg-content">
                {renderContent(msg.content)}
              </div>
              {msg.toolCalls && msg.toolCalls.length > 0 && (
                <ToolCallsDisplay calls={msg.toolCalls} />
              )}
              {msg.sources && msg.sources.length > 0 && (
                <div className="chat-sources">
                  <span className="chat-sources-label">Sources:</span>
                  {msg.sources.map((s) => (
                    <span
                      key={`${s.collection}:${s.path}`}
                      className="chat-source-tag"
                    >
                      {s.collection}/{s.path}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && messages[messages.length - 1]?.role !== "assistant" && (
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

      <div className="chat-input-wrap">
        <form
          className="chat-input-bar"
          onSubmit={(e) => {
            e.preventDefault();
            void sendMessage();
          }}
        >
          <input
            type="text"
            placeholder="Ask a question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="chat-input"
            disabled={loading}
          />
          <button
            type="submit"
            className="chat-send"
            disabled={loading || !input.trim()}
            aria-label="Send message"
          >
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </form>
      </div>
    </div>
  );
}

function ToolCallsDisplay({ calls }: { calls: ToolCallInfo[] }) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  return (
    <div className="chat-tool-calls">
      {calls.map((call, i) => {
        const isExpanded = expandedIdx === i;
        const argsStr = Object.entries(call.args)
          .map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`)
          .join(", ");

        return (
          <div key={i} className={`chat-tool-call${call.isError ? " error" : ""}`}>
            <button
              type="button"
              className="chat-tool-call-header"
              onClick={() => setExpandedIdx(isExpanded ? null : i)}
              aria-expanded={isExpanded}
            >
              <span className="chat-tool-call-icon">
                {call.isError ? "!" : "\u2713"}
              </span>
              <span className="chat-tool-call-name">{call.name}</span>
              <span className="chat-tool-call-args">{argsStr}</span>
              <span className={`chat-tool-call-chevron${isExpanded ? " open" : ""}`}>
                {"\u25B8"}
              </span>
            </button>
            {isExpanded && call.result && (
              <pre className="chat-tool-call-result">{call.result}</pre>
            )}
          </div>
        );
      })}
    </div>
  );
}

function deduplicateSources(sources: SearchResult[]): SearchResult[] {
  const seen = new Set<string>();
  return sources.filter((s) => {
    const key = `${s.collection}:${s.path}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function renderContent(text: string) {
  return text.split("\n").map((line, i) => (
    <p key={i}>
      {line
        .split(/(\*\*[^*]+\*\*|_[^_]+_)/)
        .map((part, j) =>
          part.startsWith("**") && part.endsWith("**") ? (
            <strong key={j}>{part.slice(2, -2)}</strong>
          ) : part.startsWith("_") && part.endsWith("_") ? (
            <em key={j}>{part.slice(1, -1)}</em>
          ) : (
            <span key={j}>{part}</span>
          ),
        )}
    </p>
  ));
}
