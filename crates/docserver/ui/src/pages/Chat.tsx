import { useState, useRef, useEffect, useCallback } from "react";
import { getModel, stream } from "@mariozechner/pi-ai";
import type { Context, UserMessage } from "@mariozechner/pi-ai";
import { api } from "../lib/api";
import type { SearchResult } from "../lib/api";
import "./Chat.css";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SearchResult[];
}

function buildSystemPrompt(
  docs: { title: string; collection: string; path: string; content: string }[],
): string {
  if (docs.length === 0) {
    return "You are a helpful assistant. The user has a document collection but no relevant documents were found for this query. Answer as best you can and suggest they might want to ingest relevant documents.";
  }

  let prompt =
    "You are a helpful assistant that answers questions based on the user's document collection.\n\n";
  prompt +=
    "Use the following documents as context to answer the user's question. If the documents don't contain relevant information, say so.\n\n";

  for (const doc of docs) {
    prompt += `---\nDocument: ${doc.title} (${doc.collection}/${doc.path})\n${doc.content}\n---\n\n`;
  }

  return prompt;
}

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
      // 1. Search for relevant documents.
      const searchRes = await api.search({
        query: text,
        mode: "hybrid",
        count: 5,
      });
      const sources = searchRes.results;

      // 2. Load LLM settings.
      const settings = await api.getLlmSettings();

      if (!settings.provider || !settings.model || !settings.api_key) {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content:
              "No LLM provider configured. Go to **Settings** to select a provider, model, and API key.",
            sources,
          },
        ]);
        return;
      }

      // 3. Fetch document content for RAG context.
      const docContents = await Promise.all(
        sources.slice(0, 3).map(async (s) => {
          try {
            const doc = await api.getDocument(s.collection, s.path);
            return {
              title: s.title,
              collection: s.collection,
              path: s.path,
              content: doc.content,
            };
          } catch {
            return {
              title: s.title,
              collection: s.collection,
              path: s.path,
              content: "(content unavailable)",
            };
          }
        }),
      );

      // 4. Build pi-ai context with user message.
      const userPiMsg: UserMessage = {
        role: "user",
        content: text,
        timestamp: Date.now(),
      };
      const piContext: Context = {
        systemPrompt: buildSystemPrompt(docContents),
        messages: [userPiMsg],
      };

      // 5. Stream the response.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const model = getModel(settings.provider as any, settings.model as any);
      const assistantId = crypto.randomUUID();

      // Add empty assistant message that we'll stream into.
      setMessages((prev) => [
        ...prev,
        { id: assistantId, role: "assistant", content: "", sources },
      ]);

      const controller = new AbortController();
      abortRef.current = controller;

      const s = stream(model, piContext, {
        apiKey: settings.api_key,
        abortSignal: controller.signal,
      });

      for await (const event of s) {
        if (event.type === "text_delta") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + event.delta } : m,
            ),
          );
        }
        if (event.type === "error") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    content: m.content || `Error from ${settings.provider}: ${event.error}`,
                  }
                : m,
            ),
          );
        }
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
            <p>Ask a question and your documents will be searched for relevant context.</p>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`chat-msg chat-msg-${msg.role}`}>
            <div className="chat-msg-bubble">
              <div className="chat-msg-content">{renderContent(msg.content)}</div>
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
  );
}

function renderContent(text: string) {
  return text
    .split("\n")
    .map((line, i) => (
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
