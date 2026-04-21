import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import { useNavigate, useParams } from "react-router";
import { getModel } from "@mariozechner/pi-ai";
import "katex/dist/katex.min.css";

import type { ConversationFull, ConversationSummary } from "../lib/api";
import { createPiContext } from "./chat-context";
import { groupMessagesForDisplay } from "./chat-message-groups";
import {
  CHAT_SYSTEM_PROMPT,
  MAX_TOOL_ROUNDS,
  chatAgentTools,
  createAssistantPlaceholder,
  createConversationTitle,
  createMissingConfigMessage,
  createRuntimeErrorMessage,
  createUserMessage,
  queueSubagentResult,
  resolveReadyLlmSettings,
  runParentAgentRound,
  updateMessageById,
  updateSubagentMessages,
} from "./chat-agent-runtime";
import type { Message } from "./chat-message-codec";
import type { ChatToolRuntimeState, QueuedAnalysisFile } from "./chat-subagents";
import ChatTranscript from "./ChatTranscript";
import {
  createConversationRecord,
  deleteConversationRecord,
  loadConversationById,
  loadConversationSummaries,
  loadLlmSettings,
  persistConversationMessages,
} from "./chat-session";
import "./Chat.css";

const STARTER_PROMPTS = [
  "Summarize the documents about ",
  "Find the files that explain ",
  "Compare what my notes say about ",
] as const;

const MIN_COMPOSER_HEIGHT = 56;
const MAX_COMPOSER_HEIGHT = 220;
const AUTO_SCROLL_BOTTOM_THRESHOLD = 120;
const CONVERSATION_PREFETCH_TTL_MS = 10_000;

type LoadedConversation = Awaited<ReturnType<typeof loadConversationById>>;

type ConversationPrefetchEntry = {
  startedAt: number;
  promise: Promise<LoadedConversation>;
};

type GetModelProvider = Parameters<typeof getModel>[0];
type GetModelId = Parameters<typeof getModel>[1];

function uuid(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (char) => {
    const random = (Math.random() * 16) | 0;
    return (char === "x" ? random : (random & 0x3) | 0x8).toString(16);
  });
}

function formatRelativeTime(ms: number, now: number = Date.now()): string {
  const diff = now - ms;
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function formatAbsoluteTime(ms: number): string {
  return new Date(ms).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function useTickingNow(intervalMs: number): number {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs]);

  return now;
}

function chatScrollBehavior(): ScrollBehavior {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return "smooth";
  }

  return window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
}

function resizeComposer(node: HTMLTextAreaElement | null) {
  if (!node) {
    return;
  }

  node.style.height = "0px";
  const nextHeight = Math.min(node.scrollHeight, MAX_COMPOSER_HEIGHT);
  node.style.height = `${Math.max(nextHeight, MIN_COMPOSER_HEIGHT)}px`;
}

export default function Chat() {
  const { conversationId } = useParams<{ conversationId?: string }>();
  const navigate = useNavigate();
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [activeConversation, setActiveConversation] = useState<ConversationFull | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [isComposing, setIsComposing] = useState(false);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const pendingLocalConversationIdRef = useRef<string | null>(null);
  const atTranscriptBottomRef = useRef(true);
  const conversationPrefetchRef = useRef(new Map<string, ConversationPrefetchEntry>());
  const now = useTickingNow(30_000);

  const prefetchConversation = useCallback((id: string) => {
    const cache = conversationPrefetchRef.current;
    const existing = cache.get(id);
    if (existing && Date.now() - existing.startedAt < CONVERSATION_PREFETCH_TTL_MS) {
      return;
    }
    cache.set(id, { startedAt: Date.now(), promise: loadConversationById(id) });
  }, []);

  const consumePrefetchedConversation = useCallback(
    (id: string): Promise<LoadedConversation> | null => {
      const cache = conversationPrefetchRef.current;
      const existing = cache.get(id);
      cache.delete(id);
      if (existing && Date.now() - existing.startedAt < CONVERSATION_PREFETCH_TTL_MS) {
        return existing.promise;
      }
      return null;
    },
    [],
  );

  const reloadConversations = useCallback(async () => {
    const nextConversations = await loadConversationSummaries();
    setConversations(nextConversations);
  }, []);

  useEffect(() => {
    void reloadConversations();
  }, [reloadConversations]);

  useEffect(() => {
    let cancelled = false;

    const syncConversation = async () => {
      if (!conversationId) {
        if (pendingLocalConversationIdRef.current || activeId) {
          setConfirmDelete(null);
          return;
        }

        setActiveId(null);
        setActiveConversation(null);
        setMessages([]);
        setConfirmDelete(null);
        return;
      }

      if (conversationId === activeId) {
        setConfirmDelete(null);
        return;
      }

      const loadedConversation = await loadConversationById(conversationId);
      if (cancelled) {
        return;
      }

      if (!loadedConversation) {
        navigate("/chat", { replace: true });
        return;
      }

      setActiveId(conversationId);
      setActiveConversation(loadedConversation.conversation);
      setMessages((current) => {
        if (
          pendingLocalConversationIdRef.current === conversationId &&
          current.length > loadedConversation.messages.length
        ) {
          pendingLocalConversationIdRef.current = null;
          return current;
        }

        if (pendingLocalConversationIdRef.current === conversationId) {
          pendingLocalConversationIdRef.current = null;
        }
        return loadedConversation.messages;
      });
      setConfirmDelete(null);
    };

    void syncConversation();

    return () => {
      cancelled = true;
    };
  }, [conversationId, activeId, navigate]);

  useEffect(() => {
    const container = bottomRef.current?.parentElement;
    if (!container) {
      return;
    }

    const handleScroll = () => {
      const distanceFromBottom =
        container.scrollHeight - container.scrollTop - container.clientHeight;
      atTranscriptBottomRef.current = distanceFromBottom < AUTO_SCROLL_BOTTOM_THRESHOLD;
    };

    container.addEventListener("scroll", handleScroll, { passive: true });
    return () => container.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    atTranscriptBottomRef.current = true;
  }, [activeId]);

  useEffect(() => {
    const lastRole = messages[messages.length - 1]?.role;
    const shouldFollow = atTranscriptBottomRef.current || lastRole === "user";
    if (!shouldFollow) {
      return;
    }
    bottomRef.current?.scrollIntoView({ behavior: chatScrollBehavior() });
  }, [messages]);

  useEffect(() => {
    resizeComposer(composerRef.current);
  }, [input]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const selectConversation = useCallback(
    async (id: string) => {
      const prefetched = consumePrefetchedConversation(id);
      const loadedConversation = await (prefetched ?? loadConversationById(id));
      if (!loadedConversation) {
        return;
      }

      setActiveId(id);
      setActiveConversation(loadedConversation.conversation);
      setMessages(loadedConversation.messages);
      setInput("");
      setConfirmDelete(null);
      navigate(`/chat/${id}`);
    },
    [consumePrefetchedConversation, navigate],
  );

  const startNewChat = useCallback(() => {
    setActiveId(null);
    setActiveConversation(null);
    setMessages([]);
    setInput("");
    setConfirmDelete(null);
    navigate("/chat");
  }, [navigate]);

  const deleteConversation = useCallback(
    async (id: string) => {
      const deleted = await deleteConversationRecord(id);
      if (!deleted) {
        return;
      }

      conversationPrefetchRef.current.delete(id);
      setConfirmDelete(null);
      if (activeId === id) {
        startNewChat();
      }
      await reloadConversations();
    },
    [activeId, reloadConversations, startNewChat],
  );

  const saveConversation = useCallback(
    async (
      conversationIdToSave: string,
      conversation: ConversationFull | null,
      nextMessages: Message[],
    ) => {
      await persistConversationMessages(
        conversationIdToSave,
        conversation,
        nextMessages,
        reloadConversations,
      );
    },
    [reloadConversations],
  );

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) {
      return;
    }

    const userMessage = createUserMessage(uuid(), text);
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);

    let conversationIdToSave = activeId;
    let conversation = activeConversation;
    if (!conversationIdToSave) {
      const id = uuid();
      const createdConversation = await createConversationRecord(id, createConversationTitle(text));
      if (createdConversation) {
        conversation = createdConversation;
        conversationIdToSave = id;
        await saveConversation(id, createdConversation, nextMessages);
        pendingLocalConversationIdRef.current = id;
        setActiveId(id);
        setActiveConversation(createdConversation);
        navigate(`/chat/${id}`, { replace: true });
      }
    }

    try {
      const maybeReadySettings = resolveReadyLlmSettings(await loadLlmSettings());

      if (!maybeReadySettings) {
        const errorMessages = [...nextMessages, createMissingConfigMessage(uuid())];
        setMessages(errorMessages);
        if (conversationIdToSave && conversation) {
          void saveConversation(conversationIdToSave, conversation, errorMessages);
        }
        return;
      }

      const settings = maybeReadySettings;
      const model = getModel(settings.provider as GetModelProvider, settings.model as GetModelId);
      const controller = new AbortController();
      abortRef.current = controller;

      const piContext = createPiContext(nextMessages, CHAT_SYSTEM_PROMPT, chatAgentTools);
      const assistantId = uuid();
      const runtimeState: ChatToolRuntimeState = {
        currentTurnSearchResults: [],
        queuedAnalysisFiles: [],
      };
      const updateAssistantMessage = (updater: (message: Message) => Message) =>
        setMessages((previous) => updateMessageById(previous, assistantId, updater));
      const updateSubagentMessage = (messageId: string, updater: (message: Message) => Message) =>
        setMessages((previous) => updateSubagentMessages(previous, messageId, updater));
      const queueSubagentMessage = (file: QueuedAnalysisFile) => {
        runtimeState.queuedAnalysisFiles = [...runtimeState.queuedAnalysisFiles, file];
        setMessages((previous) => queueSubagentResult(previous, file));
      };

      setMessages((previous) => [...previous, createAssistantPlaceholder(assistantId)]);

      for (let round = 0; round < MAX_TOOL_ROUNDS; round += 1) {
        const shouldContinue = await runParentAgentRound({
          model,
          settings,
          controller,
          userQuestion: text,
          piContext,
          updateAssistantMessage,
          runtimeState,
          queueSubagentMessage,
          updateSubagentMessage,
          createId: uuid,
        });
        if (!shouldContinue) {
          break;
        }
      }
    } catch (error) {
      setMessages((previous) => [...previous, createRuntimeErrorMessage(uuid(), error)]);
    } finally {
      setLoading(false);
      abortRef.current = null;

      if (conversationIdToSave && conversation) {
        setMessages((latest) => {
          void saveConversation(conversationIdToSave, conversation, latest);
          return latest;
        });
      }
    }
  }, [input, loading, messages, activeId, activeConversation, saveConversation, navigate]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const handleStarterPrompt = useCallback((prompt: string) => {
    setInput(prompt);
    composerRef.current?.focus();
  }, []);

  const handleComposerKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key !== "Enter" || event.shiftKey || isComposing || event.nativeEvent.isComposing) {
        return;
      }

      event.preventDefault();
      void sendMessage();
    },
    [isComposing, sendMessage],
  );

  const displayMessageGroups = groupMessagesForDisplay(messages);
  const conversationCountLabel =
    conversations.length === 0
      ? "No saved threads yet"
      : `${conversations.length} saved ${conversations.length === 1 ? "thread" : "threads"}`;
  const hasDraft = input.trim().length > 0;

  return (
    <div className="chat-page">
      <aside className="chat-sidebar" aria-label="Conversations">
        <div className="chat-conv-list-header">
          <div>
            <p className="chat-sidebar-kicker">Recent conversations</p>
            <p className="chat-sidebar-summary">{conversationCountLabel}</p>
          </div>
        </div>

        <div className="chat-conv-list" aria-label="Conversation history">
          {conversations.length === 0 && (
            <div className="chat-conv-empty">
              Start a thread to build up a searchable conversation history.
            </div>
          )}
          {conversations.map((conversation) => (
            <div
              key={conversation.id}
              className={`chat-conv-item${activeId === conversation.id ? " active" : ""}`}
            >
              <button
                type="button"
                className="chat-conv-btn"
                onClick={() => void selectConversation(conversation.id)}
                onMouseEnter={() => prefetchConversation(conversation.id)}
                onFocus={() => prefetchConversation(conversation.id)}
                title={conversation.title}
              >
                <span className="chat-conv-title">{conversation.title}</span>
                <time
                  className="chat-conv-time"
                  dateTime={new Date(conversation.updated_at).toISOString()}
                  title={formatAbsoluteTime(conversation.updated_at)}
                >
                  {formatRelativeTime(conversation.updated_at, now)}
                </time>
              </button>
              {confirmDelete === conversation.id ? (
                <div className="chat-conv-confirm">
                  <button
                    type="button"
                    className="chat-conv-confirm-yes"
                    onClick={() => void deleteConversation(conversation.id)}
                  >
                    Delete
                  </button>
                  <button
                    type="button"
                    className="chat-conv-confirm-no"
                    onClick={() => setConfirmDelete(null)}
                  >
                    Keep
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  className="chat-conv-delete"
                  onClick={() => setConfirmDelete(conversation.id)}
                  aria-label={`Delete ${conversation.title}`}
                  title="Delete conversation"
                >
                  <TrashIcon />
                </button>
              )}
            </div>
          ))}
        </div>
        <div className="chat-sidebar-footer">
          <button type="button" className="chat-new-btn" onClick={startNewChat}>
            <PlusIcon />
            <span>New chat</span>
          </button>
        </div>
      </aside>

      <div className="chat-main" aria-busy={loading}>
        <ChatTranscript
          displayMessageGroups={displayMessageGroups}
          loading={loading}
          lastMessageRole={messages[messages.length - 1]?.role}
          bottomRef={bottomRef}
          starterPrompts={[...STARTER_PROMPTS]}
          onPickStarter={handleStarterPrompt}
        />

        <div className="chat-input-wrap">
          <form
            className="chat-input-bar"
            onSubmit={(event) => {
              event.preventDefault();
              void sendMessage();
            }}
          >
            <div
              className={`chat-input-shell${hasDraft ? " has-draft" : ""}${loading ? " is-loading" : ""}`}
            >
              <label className="sr-only" htmlFor="chat-composer">
                Message
              </label>
              <textarea
                id="chat-composer"
                ref={composerRef}
                placeholder="Ask about your notes, docs, or PDFs…"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                onCompositionStart={() => setIsComposing(true)}
                onCompositionEnd={() => setIsComposing(false)}
                className="chat-input"
                rows={1}
                disabled={loading}
              />
            </div>
            {loading ? (
              <button type="button" className="chat-stop" onClick={handleStop}>
                Stop
              </button>
            ) : (
              <button
                type="submit"
                className={`chat-send${hasDraft ? " ready" : ""}`}
                disabled={!hasDraft}
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
            )}
          </form>
        </div>
      </div>
    </div>
  );
}

function PlusIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 5v14" />
      <path d="M5 12h14" />
    </svg>
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
