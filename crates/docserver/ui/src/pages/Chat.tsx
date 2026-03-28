import { useState, useRef, useEffect, useCallback } from "react";
import { Link, useParams, useNavigate } from "react-router";
import Markdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { Type, getModel, streamSimple } from "@mariozechner/pi-ai";
import "katex/dist/katex.min.css";
import type { Context, Tool, UserMessage, Message as PiMessage } from "@mariozechner/pi-ai";
import { api, buildDocumentTabHref } from "../lib/api";
import type {
  ChatActor,
  ConversationFull,
  ConversationSummary,
  LlmSettings,
  SearchResult,
} from "../lib/api";
import {
  apiToMessages,
  messagesToApi,
  type Message,
  type ToolCallInfo,
} from "./chat-message-codec";
import {
  applyInterruptedStopReason,
  createPiContext,
  createToolResultMessage,
} from "./chat-context";
import {
  insertOrUpdateSubagentMessage,
  mergeCurrentTurnSearchResults,
  setSubagentStatus,
  updateSubagentMessageById,
  upsertSubagentPart,
  type AnalyzeFilesAcceptedItem,
  type ChatToolRuntimeState,
  type QueuedAnalysisFile,
  type SubagentAnalysisResult,
} from "./chat-subagents";
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

interface ReadyLlmSettings {
  provider: string;
  model: string;
  api_key: string;
}

type UpdateAssistantMessage = (fn: (message: Message) => Message) => void;
type SubagentMessage = Message & { actor: Extract<ChatActor, { type: "subagent" }> };

type DisplayMessageGroup = {
  message: Message;
  nestedSubagents: SubagentMessage[];
};

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
    name: "analyze_document",
    description:
      "Run a focused file analysis subagent for one document and return a concise summary relevant to the user question.",
    parameters: Type.Object({
      collection: Type.String({ description: "The collection name" }),
      path: Type.String({
        description: "The document path within the collection",
      }),
      focus: Type.Optional(
        Type.String({
          description: "Optional extra focus for this file analysis.",
        }),
      ),
    }),
  },
];

// ── Tool execution ──

async function executeTool(
  name: string,
  args: Record<string, unknown>,
  runtimeState: ChatToolRuntimeState,
): Promise<{ text: string; sources?: SearchResult[] }> {
  switch (name) {
    case "search_semantic": {
      const res = await api.search({
        query: args.query as string,
        mode: "semantic",
        collection: args.collection as string | undefined,
        count: (args.count as number) ?? 5,
      });
      runtimeState.currentTurnSearchResults = mergeCurrentTurnSearchResults(
        runtimeState.currentTurnSearchResults,
        res.results,
      );
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
      runtimeState.currentTurnSearchResults = mergeCurrentTurnSearchResults(
        runtimeState.currentTurnSearchResults,
        res.results,
      );
      return {
        text: JSON.stringify(res.results, null, 2),
        sources: res.results,
      };
    }
    default:
      return { text: `Unknown tool: ${name}` };
  }
}

const SYSTEM_PROMPT = `You are a helpful assistant with access to a document store.

When the user asks a question:
1. Use search_hybrid or search_semantic to find relevant documents. Both tools accept an optional "collection" parameter to restrict results to a single collection. Omit it to search across all collections at once — this is usually the best default.
2. When a specific document looks relevant, call analyze_document for that file. You may call analyze_document multiple times for different files.
3. Use the tool results from those file analyses to answer the user.
4. Prefer focused per-document analysis over making unsupported claims from titles or snippets alone.

If no relevant documents are found, say so and suggest what the user might want to ingest.`;

const MAX_TOOL_ROUNDS = 10;
const CHAT_MARKDOWN_REMARK_PLUGINS = [remarkGfm, remarkMath];
const CHAT_MARKDOWN_REHYPE_PLUGINS = [rehypeKatex];

function resolveReadyLlmSettings(settings: LlmSettings): ReadyLlmSettings | null {
  if (!settings.provider || !settings.model || !settings.api_key) {
    return null;
  }

  return {
    provider: settings.provider,
    model: settings.model,
    api_key: settings.api_key,
  };
}

function createUserMessage(text: string): Message {
  return {
    id: uuid(),
    role: "user",
    content: text,
    parts: [{ type: "text", text }],
  };
}

function createAssistantPlaceholder(id: string): Message {
  return { id, role: "assistant", content: "", parts: [] };
}

function createQueuedSubagentMessage(messageId: string, file: AnalyzeFilesAcceptedItem): Message {
  return {
    id: messageId,
    role: "assistant",
    content: "Queued for file analysis.",
    parts: [{ type: "text", text: "Queued for file analysis." }],
    actor: {
      type: "subagent",
      id: messageId,
      collection: file.collection,
      path: file.path,
      status: "queued",
    },
  };
}

function createMissingConfigMessage(): Message {
  return {
    id: uuid(),
    role: "assistant",
    content:
      "No LLM provider configured. Go to **Settings** to select a provider, model, and API key.",
  };
}

function createRuntimeErrorMessage(error: unknown): Message {
  return {
    id: uuid(),
    role: "assistant",
    content: `Something went wrong: ${error instanceof Error ? error.message : "unknown error"}`,
  };
}

function createConversationTitle(text: string): string {
  return text.length > 80 ? text.slice(0, 80) + "..." : text;
}

function createSubagentContext(
  userQuestion: string,
  fileContent: string,
  file: QueuedAnalysisFile,
  focus?: string,
): Context {
  const userPiMsg: UserMessage = {
    role: "user",
    content: [
      `User question: ${userQuestion}`,
      `Analyze exactly this file: ${file.collection}/${file.path}`,
      focus ? `Extra focus: ${focus}` : null,
      "Focus on the most relevant facts, note uncertainty, and do not answer beyond this file.",
      "Return a concise summary that the parent agent can use as tool output.",
      "",
      fileContent,
    ]
      .filter((part): part is string => Boolean(part))
      .join("\n"),
    timestamp: Date.now(),
  };

  return {
    systemPrompt:
      "You are a file-analysis subagent. Read one file and report the important findings relevant to the user question. Do not call tools. Do not synthesize across files.",
    messages: [userPiMsg],
    tools: [],
  };
}

function updateMessageById(
  messages: Message[],
  id: string,
  updater: (message: Message) => Message,
): Message[] {
  return messages.map((message) => (message.id === id ? updater(message) : message));
}

function applyTextDelta(message: Message, captured: string, delta: string): Message {
  const parts = [...(message.parts ?? [])];
  const last = parts[parts.length - 1];
  if (last && last.type === "text") {
    parts[parts.length - 1] = { type: "text", text: captured };
  } else {
    parts.push({ type: "text", text: captured });
  }

  return { ...message, content: message.content + delta, parts };
}

function applyThinkingDelta(message: Message, captured: string): Message {
  const parts = [...(message.parts ?? [])];
  const last = parts[parts.length - 1];
  if (last && last.type === "thinking") {
    parts[parts.length - 1] = { type: "thinking", text: captured };
  } else {
    parts.push({ type: "thinking", text: captured });
  }

  return { ...message, parts };
}

function applyStreamError(message: Message, provider: string, error: unknown): Message {
  const rendered = typeof error === "string" ? error : JSON.stringify(error);
  return {
    ...message,
    content: message.content || `Error from ${provider}: ${rendered}`,
  };
}

function startSubagentMessage(message: Message): Message {
  return {
    ...setSubagentStatus(message, "running"),
    content: "",
    parts: [],
  };
}

function applySubagentTextDelta(message: Message, captured: string): Message {
  return upsertSubagentPart(startSubagentMessageIfQueued(message), {
    type: "text",
    text: captured,
  });
}

function applySubagentThinkingDelta(message: Message, captured: string): Message {
  return upsertSubagentPart(startSubagentMessageIfQueued(message), {
    type: "thinking",
    text: captured,
  });
}

function startSubagentMessageIfQueued(message: Message): Message {
  if (message.actor?.type === "subagent" && message.actor.status === "queued") {
    return startSubagentMessage(message);
  }
  return message;
}

function finalizeSubagentMessage(message: Message, status: "done" | "error"): Message {
  return setSubagentStatus(message, status);
}

function appendPendingToolCall(message: Message, callInfo: ToolCallInfo): Message {
  return {
    ...message,
    parts: [...(message.parts ?? []), { type: "tool_call", call: { ...callInfo } }],
  };
}

function applyToolCallResult(
  message: Message,
  callInfo: ToolCallInfo,
  _allSources: SearchResult[],
): Message {
  const parts = [...(message.parts ?? [])];
  for (let i = parts.length - 1; i >= 0; i--) {
    const part = parts[i];
    if (part.type === "tool_call" && part.call.name === callInfo.name && !part.call.result) {
      parts[i] = { type: "tool_call", call: { ...callInfo } };
      break;
    }
  }

  return {
    ...message,
    parts,
  };
}

function createSearchResultSource(collection: string, path: string, title?: string): SearchResult {
  return {
    rank: 0,
    score: 0,
    doc_id: `${collection}:${path}`,
    collection,
    path,
    title: title ?? path,
  };
}

async function executeToolCallAndRecord(
  call: { id: string; name: string; arguments: Record<string, unknown> },
  piContext: Context,
  allSources: SearchResult[],
  runtimeState: ChatToolRuntimeState,
): Promise<ToolCallInfo> {
  const callInfo: ToolCallInfo = { name: call.name, args: call.arguments };

  try {
    const toolResult = await executeTool(call.name, call.arguments, runtimeState);

    if (toolResult.sources) {
      allSources.push(...toolResult.sources);
    }

    callInfo.result = toolResult.text;
    piContext.messages.push(
      createToolResultMessage(call.id, call.name, toolResult.text, false) as PiMessage,
    );
  } catch (error) {
    const errText = error instanceof Error ? error.message : "unknown error";
    callInfo.result = `Error: ${errText}`;
    callInfo.isError = true;
    piContext.messages.push(
      createToolResultMessage(call.id, call.name, `Error: ${errText}`, true) as PiMessage,
    );
  }

  return callInfo;
}

async function runFileSubagent({
  model,
  settings,
  controller,
  file,
  userQuestion,
  focus,
  updateMessage,
}: {
  model: ReturnType<typeof getModel>;
  settings: ReadyLlmSettings;
  controller: AbortController;
  file: QueuedAnalysisFile;
  userQuestion: string;
  focus?: string;
  updateMessage: (updater: (message: Message) => Message) => void;
}): Promise<SubagentAnalysisResult> {
  try {
    const document = await api.getDocument(file.collection, file.path);
    const context = createSubagentContext(userQuestion, document.content, file, focus);
    let streamedText = "";
    let streamedThinking = "";
    let streamError: string | undefined;

    updateMessage((message) => startSubagentMessage(message));

    const stream = streamSimple(model, context, {
      apiKey: settings.api_key,
      signal: controller.signal,
      reasoning: model.reasoning ? "medium" : undefined,
    });

    for await (const event of stream) {
      if (event.type === "text_delta") {
        streamedText += event.delta;
        const captured = streamedText;
        updateMessage((message) => applySubagentTextDelta(message, captured));
      }
      if (event.type === "thinking_delta") {
        streamedThinking += event.delta;
        const captured = streamedThinking;
        updateMessage((message) => applySubagentThinkingDelta(message, captured));
      }
      if (event.type === "error") {
        streamError = typeof event.error === "string" ? event.error : JSON.stringify(event.error);
        updateMessage((message) => {
          const next = applyStreamError(message, settings.provider, event.error);
          return finalizeSubagentMessage(next, "error");
        });
      }
    }

    await stream.result();
    if (streamError) {
      updateMessage((message) => finalizeSubagentMessage(message, "error"));
      return {
        collection: file.collection,
        path: file.path,
        reason: file.reason,
        title: file.title,
        error: streamError,
      };
    }

    updateMessage((message) => finalizeSubagentMessage(message, "done"));
    return {
      collection: file.collection,
      path: file.path,
      reason: file.reason,
      title: file.title,
      text: streamedText,
    };
  } catch (error) {
    const rendered = error instanceof Error ? error.message : "unknown error";
    updateMessage((message) => {
      const next = applyStreamError(message, settings.provider, rendered);
      return finalizeSubagentMessage(next, "error");
    });
    return {
      collection: file.collection,
      path: file.path,
      reason: file.reason,
      title: file.title,
      error: rendered,
    };
  }
}

async function runAnalyzeDocumentTool({
  call,
  model,
  settings,
  controller,
  userQuestion,
  piContext,
  allSources,
  runtimeState,
  queueSubagentMessage,
  updateSubagentMessage,
}: {
  call: { id: string; name: string; arguments: Record<string, unknown> };
  model: ReturnType<typeof getModel>;
  settings: ReadyLlmSettings;
  controller: AbortController;
  userQuestion: string;
  piContext: Context;
  allSources: SearchResult[];
  runtimeState: ChatToolRuntimeState;
  queueSubagentMessage: (file: QueuedAnalysisFile) => void;
  updateSubagentMessage: (messageId: string, updater: (message: Message) => Message) => void;
}): Promise<ToolCallInfo> {
  const collection = String(call.arguments.collection ?? "").trim();
  const path = String(call.arguments.path ?? "").trim();
  const focus =
    typeof call.arguments.focus === "string" && call.arguments.focus.trim().length > 0
      ? call.arguments.focus.trim()
      : undefined;
  const messageId = uuid();
  const matchedResult = runtimeState.currentTurnSearchResults.find(
    (result) => result.collection === collection && result.path === path,
  );
  const file: QueuedAnalysisFile = {
    collection,
    path,
    reason: focus ?? userQuestion,
    title: matchedResult?.title,
    messageId,
  };

  queueSubagentMessage(file);

  const result = await runFileSubagent({
    model,
    settings,
    controller,
    file,
    userQuestion,
    focus,
    updateMessage: (updater) => {
      updateSubagentMessage(messageId, updater);
    },
  });

  const callInfo: ToolCallInfo = { name: call.name, args: call.arguments };

  if (result.error) {
    callInfo.result = `Error: ${result.error}`;
    callInfo.isError = true;
    piContext.messages.push(
      createToolResultMessage(call.id, call.name, `Error: ${result.error}`, true) as PiMessage,
    );
    return callInfo;
  }

  const toolText = result.text?.trim() || "No relevant findings from this file.";
  callInfo.result = toolText;
  piContext.messages.push(
    createToolResultMessage(call.id, call.name, toolText, false) as PiMessage,
  );
  allSources.push(createSearchResultSource(collection, path, file.title));

  return callInfo;
}

async function runParentAgentRound({
  model,
  settings,
  controller,
  userQuestion,
  piContext,
  updateAssistantMessage,
  allSources,
  runtimeState,
  queueSubagentMessage,
  updateSubagentMessage,
}: {
  model: ReturnType<typeof getModel>;
  settings: ReadyLlmSettings;
  controller: AbortController;
  userQuestion: string;
  piContext: Context;
  updateAssistantMessage: UpdateAssistantMessage;
  allSources: SearchResult[];
  runtimeState: ChatToolRuntimeState;
  queueSubagentMessage: (file: QueuedAnalysisFile) => void;
  updateSubagentMessage: (messageId: string, updater: (message: Message) => Message) => void;
}): Promise<boolean> {
  let streamedText = "";
  let streamedThinking = "";

  const stream = streamSimple(model, piContext, {
    apiKey: settings.api_key,
    signal: controller.signal,
    reasoning: model.reasoning ? "medium" : undefined,
  });

  for await (const event of stream) {
    if (event.type === "text_delta") {
      streamedText += event.delta;
      const captured = streamedText;
      updateAssistantMessage((message) => applyTextDelta(message, captured, event.delta));
    }
    if (event.type === "thinking_delta") {
      streamedThinking += event.delta;
      const captured = streamedThinking;
      updateAssistantMessage((message) => applyThinkingDelta(message, captured));
    }
    if (event.type === "error") {
      updateAssistantMessage((message) =>
        applyStreamError(message, settings.provider, event.error),
      );
    }
  }

  const result = await stream.result();
  piContext.messages.push(result);

  const stopReason = result.stopReason;
  if (stopReason === "aborted" || stopReason === "error") {
    updateAssistantMessage((message) =>
      applyInterruptedStopReason(message, stopReason, result.errorMessage),
    );
    return false;
  }

  const toolCalls = result.content.filter((block) => block.type === "toolCall");
  if (toolCalls.length === 0) {
    return false;
  }

  for (const call of toolCalls) {
    const callArgs = call.arguments as Record<string, unknown>;
    const pendingCallInfo: ToolCallInfo = { name: call.name, args: callArgs };

    updateAssistantMessage((message) => appendPendingToolCall(message, pendingCallInfo));

    const resolvedCallInfo =
      call.name === "analyze_document"
        ? await runAnalyzeDocumentTool({
            call: { id: call.id, name: call.name, arguments: callArgs },
            model,
            settings,
            controller,
            userQuestion,
            piContext,
            allSources,
            runtimeState,
            queueSubagentMessage,
            updateSubagentMessage,
          })
        : await executeToolCallAndRecord(
            { id: call.id, name: call.name, arguments: callArgs },
            piContext,
            allSources,
            runtimeState,
          );

    updateAssistantMessage((message) => applyToolCallResult(message, resolvedCallInfo, allSources));
  }

  return true;
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

function renderMessageContent(message: Message, nestedSubagents: SubagentMessage[] = []) {
  if (!message.parts || message.parts.length === 0) {
    return (
      <div className="chat-msg-content">
        <Markdown
          remarkPlugins={CHAT_MARKDOWN_REMARK_PLUGINS}
          rehypePlugins={CHAT_MARKDOWN_REHYPE_PLUGINS}
        >
          {message.content}
        </Markdown>
      </div>
    );
  }

  let nextSubagentIndex = 0;

  return (
    <>
      {message.parts.map((part, i) => {
        if (part.type === "text") {
          return (
            <div key={`text-${i}`} className="chat-msg-content">
              <Markdown
                remarkPlugins={CHAT_MARKDOWN_REMARK_PLUGINS}
                rehypePlugins={CHAT_MARKDOWN_REHYPE_PLUGINS}
              >
                {part.text}
              </Markdown>
            </div>
          );
        }

        if (part.type === "thinking") {
          return <ThinkingInline key={`thinking-${i}`} text={part.text} />;
        }

        const subagent =
          part.call.name === "analyze_document" ? nestedSubagents[nextSubagentIndex] : undefined;
        if (subagent) {
          nextSubagentIndex += 1;
        }

        if (part.call.name === "analyze_document" && subagent) {
          return <SubagentInline key={subagent.id} message={subagent} />;
        }

        return (
          <div key={`tool-group-${i}`}>
            <ToolCallInline call={part.call} />
            {subagent ? <SubagentInline message={subagent} /> : null}
          </div>
        );
      })}
      {nestedSubagents.slice(nextSubagentIndex).map((subagent) => (
        <SubagentInline key={subagent.id} message={subagent} />
      ))}
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

function groupMessagesForDisplay(messages: Message[]): DisplayMessageGroup[] {
  const groups: DisplayMessageGroup[] = [];
  let lastParentAssistantGroup: DisplayMessageGroup | null = null;

  for (const message of messages) {
    if (message.actor?.type === "subagent") {
      const subagentMessage = message as SubagentMessage;
      if (lastParentAssistantGroup) {
        lastParentAssistantGroup.nestedSubagents.push(subagentMessage);
        continue;
      }

      groups.push({ message: subagentMessage, nestedSubagents: [] });
      lastParentAssistantGroup = null;
      continue;
    }

    const group: DisplayMessageGroup = { message, nestedSubagents: [] };
    groups.push(group);

    if (message.role === "assistant" && (!message.actor || message.actor.type === "parent")) {
      lastParentAssistantGroup = group;
    } else {
      lastParentAssistantGroup = null;
    }
  }

  return groups;
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

    const userMsg = createUserMessage(text);
    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);

    let convId = activeId;
    let conv = activeConv;
    if (!convId) {
      const id = uuid();
      try {
        conv = await api.createConversation(id, createConversationTitle(text));
        convId = id;
        setActiveId(id);
        setActiveConv(conv);
        navigate(`/chat/${id}`, { replace: true });
      } catch {
        /* ignore -- we'll still show the chat, just won't persist */
      }
    }

    try {
      const maybeReadySettings = resolveReadyLlmSettings(await api.getLlmSettings());

      if (!maybeReadySettings) {
        const errMsgs = [...nextMessages, createMissingConfigMessage()];
        setMessages(errMsgs);
        if (convId && conv) {
          void saveConversation(convId, conv, errMsgs);
        }
        return;
      }

      const settings = maybeReadySettings;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const model = getModel(settings.provider as any, settings.model as any);
      const controller = new AbortController();
      abortRef.current = controller;

      const piContext = createPiContext(nextMessages, SYSTEM_PROMPT, tools);
      const assistantId = uuid();
      const allSources: SearchResult[] = [];
      const runtimeState: ChatToolRuntimeState = {
        currentTurnSearchResults: [],
        acceptedAnalysisFiles: [],
        queuedAnalysisFiles: [],
      };
      const updateAssistantMessage: UpdateAssistantMessage = (fn) =>
        setMessages((prev) => updateMessageById(prev, assistantId, fn));
      const updateSubagentMessage = (messageId: string, updater: (message: Message) => Message) =>
        setMessages((prev) => updateSubagentMessageById(prev, messageId, updater));
      const queueSubagentMessage = (file: QueuedAnalysisFile) => {
        runtimeState.queuedAnalysisFiles = [...runtimeState.queuedAnalysisFiles, file];
        setMessages((prev) =>
          insertOrUpdateSubagentMessage(prev, createQueuedSubagentMessage(file.messageId, file)),
        );
      };

      setMessages((prev) => [...prev, createAssistantPlaceholder(assistantId)]);

      for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
        const shouldContinue = await runParentAgentRound({
          model,
          settings,
          controller,
          userQuestion: text,
          piContext,
          updateAssistantMessage,
          allSources,
          runtimeState,
          queueSubagentMessage,
          updateSubagentMessage,
        });
        if (!shouldContinue) {
          break;
        }
      }

      abortRef.current = null;
    } catch (err) {
      setMessages((prev) => [...prev, createRuntimeErrorMessage(err)]);
    } finally {
      setLoading(false);

      if (convId && conv) {
        setMessages((latest) => {
          void saveConversation(convId, conv, latest);
          return latest;
        });
      }
    }
  }, [input, loading, messages, activeId, activeConv, saveConversation, navigate]);

  const displayMessageGroups = groupMessagesForDisplay(messages);

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

    return parsed.filter(
      (item): item is SearchResult =>
        item &&
        typeof item === "object" &&
        typeof item.collection === "string" &&
        typeof item.path === "string" &&
        typeof item.title === "string",
    );
  } catch {
    return null;
  }
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

function ToolCallInline({ call }: { call: ToolCallInfo }) {
  const [expanded, setExpanded] = useState(false);
  const argsStr = Object.entries(call.args)
    .map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join(", ");
  const searchResults = parseToolSearchResults(call);

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
      {expanded && call.result && searchResults && (
        <div className="chat-tool-search-results">
          {searchResults.length === 0 ? (
            <div className="chat-tool-search-empty">No results</div>
          ) : (
            searchResults.map((result) => (
              <div key={`${result.collection}:${result.path}`} className="chat-tool-search-result">
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
            ))
          )}
        </div>
      )}
      {expanded && call.result && !searchResults && (
        <pre className="chat-tool-call-result">{call.result}</pre>
      )}
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
