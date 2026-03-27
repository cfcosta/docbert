import { useState, useRef, useEffect, useCallback } from "react";
import { useParams, useNavigate } from "react-router";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Type, getModel, stream } from "@mariozechner/pi-ai";
import type {
  Context,
  Tool,
  UserMessage,
  ToolResultMessage,
  Message as PiMessage,
} from "@mariozechner/pi-ai";
import { api } from "../lib/api";
import type { SearchResult, ConversationSummary, ConversationFull } from "../lib/api";
import "./Chat.css";

function uuid(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

interface ToolCallInfo {
  name: string;
  args: Record<string, unknown>;
  result?: string;
  isError?: boolean;
}

type ContentPart = { type: "text"; text: string } | { type: "tool_call"; call: ToolCallInfo };

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  parts?: ContentPart[];
  sources?: SearchResult[];
}

// ── Tool definitions for pi-ai ──

const tools: Tool[] = [
  {
    name: "search_semantic",
    description:
      "Search the document store using semantic (ColBERT) search. Best for meaning-based queries where wording may differ from the target documents. Searches all collections by default.",
    parameters: Type.Object({
      query: Type.String({ description: "The search query" }),
      collection: Type.Optional(
        Type.String({
          description: "Restrict search to this collection. Omit to search all collections.",
        }),
      ),
      count: Type.Optional(Type.Number({ description: "Number of results to return (default 5)" })),
    }),
  },
  {
    name: "search_hybrid",
    description:
      "Search the document store using hybrid BM25 + semantic search. Best when the query shares keywords with the target documents. Faster on large collections.",
    parameters: Type.Object({
      query: Type.String({ description: "The search query" }),
      collection: Type.Optional(Type.String({ description: "Restrict search to this collection" })),
      count: Type.Optional(Type.Number({ description: "Number of results to return (default 5)" })),
    }),
  },
  {
    name: "document_get",
    description: "Retrieve the full content of a specific document by collection and path.",
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
        collection: args.collection as string | undefined,
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
      const doc = await api.getDocument(args.collection as string, args.path as string);
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
1. Use search_hybrid or search_semantic to find relevant documents. Both tools accept an optional "collection" parameter to restrict results to a single collection. Omit it to search across all collections at once — this is usually the best default.
2. Review the search results — if multiple documents look relevant, retrieve ALL of them with document_get, not just the top result. Answers often span several documents.
3. Synthesize your answer from all retrieved documents, citing which documents contributed to the answer.

If no relevant documents are found, say so and suggest what the user might want to ingest.`;

const MAX_TOOL_ROUNDS = 10;

function messagesToApi(messages: Message[]): ConversationFull["messages"] {
  return messages.map((m) => {
    const toolCalls = m.parts
      ?.filter((p): p is Extract<ContentPart, { type: "tool_call" }> => p.type === "tool_call")
      .map((p) => ({
        name: p.call.name,
        args: p.call.args,
        result: p.call.result,
        is_error: p.call.isError,
      }));
    return {
      id: m.id,
      role: m.role,
      content: m.content,
      sources: m.sources?.map((s) => ({
        collection: s.collection,
        path: s.path,
        title: s.title,
      })),
      tool_calls: toolCalls && toolCalls.length > 0 ? toolCalls : undefined,
    };
  });
}

function apiToMessages(msgs: ConversationFull["messages"]): Message[] {
  return msgs.map((m) => {
    const parts: ContentPart[] = [];
    if (m.tool_calls && m.tool_calls.length > 0) {
      for (const tc of m.tool_calls) {
        parts.push({
          type: "tool_call",
          call: { name: tc.name, args: tc.args, result: tc.result, isError: tc.is_error },
        });
      }
    }
    if (m.content) {
      parts.push({ type: "text", text: m.content });
    }
    return {
      id: m.id,
      role: m.role,
      content: m.content,
      parts: parts.length > 0 ? parts : undefined,
      sources: m.sources?.map((s) => ({
        rank: 0,
        score: 0,
        doc_id: "",
        collection: s.collection,
        path: s.path,
        title: s.title,
      })),
    };
  });
}

function formatRelativeTime(ms: number): string {
  const diff = Date.now() - ms;
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function Chat() {
  const { conversationId } = useParams<{ conversationId?: string }>();
  const navigate = useNavigate();
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [activeConv, setActiveConv] = useState<ConversationFull | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const loadConversations = useCallback(async () => {
    try {
      const list = await api.listConversations();
      setConversations(list);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    void loadConversations();
  }, [loadConversations]);

  // Load conversation from URL param on mount or when param changes.
  useEffect(() => {
    if (conversationId && conversationId !== activeId) {
      void (async () => {
        try {
          const conv = await api.getConversation(conversationId);
          setActiveId(conversationId);
          setActiveConv(conv);
          setMessages(apiToMessages(conv.messages));
        } catch {
          navigate("/chat", { replace: true });
        }
      })();
    } else if (!conversationId && activeId) {
      setActiveId(null);
      setActiveConv(null);
      setMessages([]);
    }
  }, [conversationId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const selectConversation = useCallback(
    async (id: string) => {
      try {
        const conv = await api.getConversation(id);
        setActiveId(id);
        setActiveConv(conv);
        setMessages(apiToMessages(conv.messages));
        navigate(`/chat/${id}`);
      } catch {
        /* ignore */
      }
    },
    [navigate],
  );

  const startNewChat = useCallback(() => {
    setActiveId(null);
    setActiveConv(null);
    setMessages([]);
    setInput("");
    navigate("/chat");
  }, [navigate]);

  const deleteConversation = useCallback(
    async (id: string) => {
      try {
        await api.deleteConversation(id);
        setConfirmDelete(null);
        if (activeId === id) {
          startNewChat();
        }
        await loadConversations();
      } catch {
        /* ignore */
      }
    },
    [activeId, startNewChat, loadConversations],
  );

  const saveConversation = useCallback(
    async (convId: string, conv: ConversationFull | null, msgs: Message[]) => {
      try {
        const apiMsgs = messagesToApi(msgs);
        if (conv) {
          await api.updateConversation(convId, { ...conv, messages: apiMsgs });
        }
        await loadConversations();
      } catch {
        /* ignore */
      }
    },
    [loadConversations],
  );

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg: Message = {
      id: uuid(),
      role: "user",
      content: text,
    };
    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);

    // Create conversation on first message if needed.
    let convId = activeId;
    let conv = activeConv;
    if (!convId) {
      const id = uuid();
      const title = text.length > 80 ? text.slice(0, 80) + "..." : text;
      try {
        conv = await api.createConversation(id, title);
        convId = id;
        setActiveId(id);
        setActiveConv(conv);
        navigate(`/chat/${id}`, { replace: true });
      } catch {
        /* ignore -- we'll still show the chat, just won't persist */
      }
    }

    try {
      const settings = await api.getLlmSettings();

      if (!settings.provider || !settings.model || !settings.api_key) {
        const errMsgs = [
          ...nextMessages,
          {
            id: uuid(),
            role: "assistant" as const,
            content:
              "No LLM provider configured. Go to **Settings** to select a provider, model, and API key.",
          },
        ];
        setMessages(errMsgs);
        if (convId && conv) {
          void saveConversation(convId, conv, errMsgs);
        }
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

      const assistantId = uuid();
      const allSources: SearchResult[] = [];

      // Helper to update the assistant message in-place.
      const updateMsg = (fn: (m: Message) => Message) =>
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? fn(m) : m)));

      // Add empty assistant message to stream into.
      setMessages((prev) => [
        ...prev,
        { id: assistantId, role: "assistant", content: "", parts: [] },
      ]);

      // Track the current streaming text so we can finalize it as a part.
      let streamedText = "";

      // Agentic tool loop: stream, handle tool calls, continue.
      for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
        streamedText = "";

        const s = stream(model, piContext, {
          apiKey: settings.api_key,
          abortSignal: controller.signal,
        });

        for await (const event of s) {
          if (event.type === "text_delta") {
            streamedText += event.delta;
            const captured = streamedText;
            updateMsg((m) => {
              // Update or append the trailing text part.
              const parts = [...(m.parts ?? [])];
              const last = parts[parts.length - 1];
              if (last && last.type === "text") {
                parts[parts.length - 1] = { type: "text", text: captured };
              } else {
                parts.push({ type: "text", text: captured });
              }
              return { ...m, content: m.content + event.delta, parts };
            });
          }
          if (event.type === "error") {
            updateMsg((m) => ({
              ...m,
              content: m.content || `Error from ${settings.provider}: ${event.error}`,
            }));
          }
        }

        // Get the final assistant message.
        const result = await s.result();
        piContext.messages.push(result);

        // Check for tool calls.
        const toolCalls = result.content.filter((b) => b.type === "toolCall");

        if (toolCalls.length === 0) {
          // No tool calls — done.
          break;
        }

        // Execute tool calls and append each inline as a part.
        for (const call of toolCalls) {
          const callArgs = call.arguments as Record<string, unknown>;
          const callInfo: ToolCallInfo = { name: call.name, args: callArgs };

          // Show tool call immediately (before result).
          updateMsg((m) => ({
            ...m,
            parts: [...(m.parts ?? []), { type: "tool_call" as const, call: { ...callInfo } }],
          }));

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

          // Update the tool call part with the result.
          updateMsg((m) => {
            const parts = [...(m.parts ?? [])];
            // Find the last tool_call part matching this call name.
            for (let i = parts.length - 1; i >= 0; i--) {
              const p = parts[i];
              if (p.type === "tool_call" && p.call.name === call.name && !p.call.result) {
                parts[i] = { type: "tool_call", call: { ...callInfo } };
                break;
              }
            }
            const uniqueSources = deduplicateSources(allSources);
            return {
              ...m,
              parts,
              sources: uniqueSources.length > 0 ? uniqueSources : m.sources,
            };
          });
        }

        // Continue to next round — the model will see tool results and respond.
      }

      abortRef.current = null;
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: uuid(),
          role: "assistant",
          content: `Something went wrong: ${err instanceof Error ? err.message : "unknown error"}`,
        },
      ]);
    } finally {
      setLoading(false);

      // Persist conversation after the response completes.
      if (convId && conv) {
        // We need the latest messages from state, so use a small trick:
        // read from the setter callback.
        setMessages((latest) => {
          void saveConversation(convId, conv, latest);
          return latest;
        });
      }
    }
  }, [input, loading, messages, activeId, activeConv, saveConversation, navigate]);

  return (
    <div className="chat-page">
      <aside className="chat-sidebar">
        <div className="chat-sidebar-header">
          <button type="button" className="chat-new-btn" onClick={startNewChat}>
            + New chat
          </button>
        </div>
        <div className="chat-conv-list">
          {conversations.length === 0 && (
            <div className="chat-conv-empty">No conversations yet.</div>
          )}
          {conversations.map((c) => (
            <div key={c.id} className={`chat-conv-item${activeId === c.id ? " active" : ""}`}>
              <button
                type="button"
                className="chat-conv-btn"
                onClick={() => void selectConversation(c.id)}
                title={c.title}
              >
                <span className="chat-conv-title">{c.title}</span>
                <span className="chat-conv-time">{formatRelativeTime(c.updated_at)}</span>
              </button>
              {confirmDelete === c.id ? (
                <div className="chat-conv-confirm">
                  <button
                    type="button"
                    className="chat-conv-confirm-yes"
                    onClick={() => void deleteConversation(c.id)}
                  >
                    Delete
                  </button>
                  <button
                    type="button"
                    className="chat-conv-confirm-no"
                    onClick={() => setConfirmDelete(null)}
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  className="chat-conv-delete"
                  onClick={() => setConfirmDelete(c.id)}
                  aria-label={`Delete ${c.title}`}
                  title="Delete conversation"
                >
                  <TrashIcon />
                </button>
              )}
            </div>
          ))}
        </div>
      </aside>

      <div className="chat-main">
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
              <p>Ask a question and the assistant will search your documents for context.</p>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`chat-msg chat-msg-${msg.role}`}>
              <div className="chat-msg-bubble">
                {msg.parts && msg.parts.length > 0 ? (
                  <>
                    {msg.parts.map((part, i) =>
                      part.type === "text" ? (
                        <div key={i} className="chat-msg-content">
                          <Markdown remarkPlugins={[remarkGfm]}>{part.text}</Markdown>
                        </div>
                      ) : (
                        <ToolCallInline key={i} call={part.call} />
                      ),
                    )}
                  </>
                ) : (
                  <div className="chat-msg-content">
                    <Markdown remarkPlugins={[remarkGfm]}>{msg.content}</Markdown>
                  </div>
                )}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="chat-sources">
                    <span className="chat-sources-label">Sources:</span>
                    {msg.sources.map((s) => (
                      <span key={`${s.collection}:${s.path}`} className="chat-source-tag">
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
    </div>
  );
}

function ToolCallInline({ call }: { call: ToolCallInfo }) {
  const [expanded, setExpanded] = useState(false);
  const argsStr = Object.entries(call.args)
    .map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join(", ");

  return (
    <div className={`chat-tool-call${call.isError ? " error" : ""}`}>
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
      {expanded && call.result && <pre className="chat-tool-call-result">{call.result}</pre>}
    </div>
  );
}

function TrashIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
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
