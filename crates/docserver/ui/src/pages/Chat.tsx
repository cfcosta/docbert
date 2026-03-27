import { useState, useRef, useEffect } from "react";
import { api } from "../lib/api";
import type { SearchResult } from "../lib/api";
import "./Chat.css";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SearchResult[];
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
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
      // Search the document store for relevant context.
      const searchRes = await api.search({
        query: text,
        mode: "hybrid",
        count: 5,
      });

      // For now, show search results as the assistant response.
      // A real LLM integration would send these as RAG context to a model.
      const sources = searchRes.results;
      const assistantContent =
        sources.length > 0
          ? `Found ${sources.length} relevant document${sources.length === 1 ? "" : "s"}:\n\n${sources
              .map(
                (s, i) =>
                  `**${i + 1}. ${s.title}** (${s.collection}/${s.path})\nRelevance: ${s.score.toFixed(3)}`,
              )
              .join(
                "\n\n",
              )}\n\n_LLM integration pending. These results would be sent as context to a language model to generate a response._`
          : "No relevant documents found for your query. Try ingesting some documents first.";

      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: assistantContent,
        sources,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      const errorMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "Something went wrong while searching. Please try again.",
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-page">
      <header className="chat-header">
        <h2>Chat</h2>
        <p className="chat-subtitle">Ask questions about your documents</p>
      </header>

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
              <div className="chat-msg-content">
                {msg.content.split("\n").map((line, i) => (
                  <p key={i}>
                    {line
                      .split(/(\*\*[^*]+\*\*)/)
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
                ))}
              </div>
            </div>
          </div>
        ))}

        {loading && (
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

      <form
        className="chat-input-bar"
        onSubmit={(e) => {
          e.preventDefault();
          sendMessage();
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
          >
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </form>
    </div>
  );
}
