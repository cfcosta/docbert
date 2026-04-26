import { Type, getModel, streamSimple } from "@mariozechner/pi-ai";
import type {
  AssistantMessage,
  Context,
  Message as PiMessage,
  Tool,
  UserMessage,
} from "@mariozechner/pi-ai";

import { api } from "../lib/api";
import type { DocumentRange, DocumentResponse, LlmSettings } from "../lib/api";
import { applyInterruptedStopReason, createToolResultMessage } from "./chat-context";
import {
  contentFromParts,
  type ContentPart,
  type Message,
  type ToolCallInfo,
} from "./chat-message-codec";
import {
  insertOrUpdateSubagentMessage,
  setSubagentStatus,
  updateSubagentMessageById,
  upsertSubagentPart,
  type AnalyzeFilesAcceptedItem,
  type ChatToolRuntimeState,
  type QueuedAnalysisFile,
  type SubagentAnalysisResult,
} from "./chat-subagents";
import {
  assistantToolCalls,
  consumeAssistantStream,
  isInterruptedAssistantResult,
  shouldContinueAssistantToolRound,
} from "./chat-stream";
import { executeSearchToolCall, searchChatTools } from "./chat-tools";

export interface ReadyLlmSettings {
  provider: string;
  model: string;
  api_key: string;
}

export type UpdateAssistantMessage = (fn: (message: Message) => Message) => void;

const analyzeDocumentTool: Tool = {
  name: "analyze_document",
  description:
    "Run a focused file analysis subagent for one document and return an evidence-rich analysis the parent agent can synthesize into the final answer. Use line_count/byte_count from search results to pass start_line/end_line (or start_byte/end_byte) when only a slice of a large file is relevant — line and byte ranges are mutually exclusive.",
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
    start_line: Type.Optional(
      Type.Number({
        description:
          "First line to read (1-indexed, inclusive). Pair with end_line to read a line range.",
      }),
    ),
    end_line: Type.Optional(
      Type.Number({
        description: "Last line to read (1-indexed, inclusive). Pair with start_line.",
      }),
    ),
    start_byte: Type.Optional(
      Type.Number({
        description:
          "First byte to read (0-indexed, inclusive). Use for transcript-style files that fit on one giant line.",
      }),
    ),
    end_byte: Type.Optional(
      Type.Number({
        description: "Last byte to read (0-indexed, inclusive). Pair with start_byte.",
      }),
    ),
  }),
};

export const chatAgentTools: Tool[] = [...searchChatTools, analyzeDocumentTool];

export const CHAT_SYSTEM_PROMPT = `You are a helpful assistant with access to a document store.

Your job is to gather enough evidence from the indexed documents before answering.

When the user asks a question:
1. Start with search_hybrid or search_semantic to find relevant documents. Both tools accept an optional "collection" parameter to restrict results to a single collection. Omit it to search across all collections at once unless the user clearly wants a specific collection.
2. Do not stop after a single search or a single file when the question could require synthesis. If the answer may be spread across multiple documents, run additional searches with alternate phrasings and analyze multiple relevant documents.
3. Use analyze_document on each promising file. Prefer reading several relevant files over guessing from one strong-looking result.
4. Each search result includes line_count and byte_count, plus a match_chunk: { start_byte, end_byte } for semantic hits that pinpoints the matching chunk in the file. Prefer match_chunk's start_byte/end_byte over a manually chosen range when it is present — it tells you exactly where the embedding model thought the answer lives. Fall back to start_line/end_line (or your own start_byte/end_byte) when you need a different region or when match_chunk is absent. Line and byte ranges are mutually exclusive. Omit ranges entirely to read the whole file.
5. Build the final answer from the combined findings of those file analyses. Reconcile overlaps, note disagreements or uncertainty, and make it clear when different files contribute different parts of the answer.
6. Prefer focused per-document analysis over making unsupported claims from titles or snippets alone.
7. If the first results are weak, incomplete, or too narrow, search again before answering.

Answering policy:
- For broad questions, compare and combine evidence from multiple sources.
- For factual questions, verify with more than one file when possible.
- For questions about a person, project, or topic, look for complementary files such as project docs, notes, memory files, specs, or related references.
- Only give a single-file answer when the evidence really appears to live in one file.

If no relevant documents are found, say so and suggest what the user might want to ingest.`;

export const MAX_TOOL_ROUNDS = 10;

export function resolveReadyLlmSettings(settings: LlmSettings): ReadyLlmSettings | null {
  if (!settings.provider || !settings.model || !settings.api_key) {
    return null;
  }

  return {
    provider: settings.provider,
    model: settings.model,
    api_key: settings.api_key,
  };
}

export function createUserMessage(id: string, text: string): Message {
  return {
    id,
    role: "user",
    content: text,
    parts: [{ type: "text", text }],
  };
}

export function createAssistantPlaceholder(id: string): Message {
  return { id, role: "assistant", content: "", parts: [] };
}

export function createQueuedSubagentMessage(
  messageId: string,
  file: AnalyzeFilesAcceptedItem,
): Message {
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

export function createMissingConfigMessage(id: string): Message {
  return {
    id,
    role: "assistant",
    content:
      "No LLM provider configured. Go to **Settings** to select a provider, model, and API key or complete the required OAuth sign-in.",
  };
}

export function createRuntimeErrorMessage(id: string, error: unknown): Message {
  return {
    id,
    role: "assistant",
    content: `Something went wrong: ${error instanceof Error ? error.message : "unknown error"}`,
  };
}

export function createConversationTitle(text: string): string {
  return text.length > 80 ? text.slice(0, 80) + "..." : text;
}

export function updateMessageById(
  messages: Message[],
  id: string,
  updater: (message: Message) => Message,
): Message[] {
  return messages.map((message) => (message.id === id ? updater(message) : message));
}

export const FILE_ANALYSIS_SYSTEM_PROMPT = `You are a file-analysis subagent.

Read one document carefully and produce a compact but evidence-rich analysis for the parent agent.
Stay strictly within this file. Do not call tools. Do not synthesize across files. Do not use outside knowledge.
Do not optimize for brevity if it would drop useful detail.

Extract the strongest concrete material that helps answer the user question, including:
- key claims, facts, definitions, decisions, examples, steps, dates, numbers, and caveats
- short quoted phrases when they carry important meaning
- uncertainty, ambiguity, or missing information in the file

If the file is only weakly relevant, say so clearly and explain why.`;

function formatSubagentMetadata(metadata?: Record<string, unknown>): string | null {
  if (!metadata || Object.keys(metadata).length === 0) {
    return null;
  }

  return JSON.stringify(metadata, null, 2);
}

function formatSubagentRange(range?: DocumentRange): string | null {
  if (!range) return null;

  const lineRange = range.startLine !== undefined || range.endLine !== undefined;
  const byteRange = range.startByte !== undefined || range.endByte !== undefined;
  if (!lineRange && !byteRange) return null;

  if (lineRange) {
    const start = range.startLine ?? 1;
    const end = range.endLine !== undefined ? String(range.endLine) : "EOF";
    return `Slice: lines ${start}-${end} (1-indexed, inclusive). Anything outside this range is not present in the excerpt below.`;
  }

  const start = range.startByte ?? 0;
  const end = range.endByte !== undefined ? String(range.endByte) : "EOF";
  return `Slice: bytes ${start}-${end} (0-indexed, inclusive). Anything outside this range is not present in the excerpt below.`;
}

export function createSubagentContext(
  userQuestion: string,
  document: Pick<DocumentResponse, "collection" | "path" | "title" | "content" | "metadata">,
  focus?: string,
  range?: DocumentRange,
): Context {
  const metadata = formatSubagentMetadata(document.metadata);
  const sliceNote = formatSubagentRange(range);
  const userPiMessage: UserMessage = {
    role: "user",
    content: [
      `User question: ${userQuestion}`,
      `Analyze exactly this file: ${document.collection}/${document.path}`,
      `Document title: ${document.title}`,
      metadata ? `Document metadata:\n${metadata}` : null,
      focus ? `Extra focus: ${focus}` : null,
      sliceNote,
      "Return markdown with these sections:",
      "## Relevance to the question",
      "State exactly what this file contributes and how strong the relevance is.",
      "## Key findings",
      "List the most useful details from the file as concrete bullet points.",
      "## Supporting evidence",
      "Include short quotes or very specific references to the text that support the findings.",
      "## Gaps or uncertainty",
      "Note limits, ambiguities, or anything the file does not establish.",
      "Be specific and information-dense so the parent agent can compose a strong final answer.",
      "",
      document.content,
    ]
      .filter((part): part is string => Boolean(part))
      .join("\n"),
    timestamp: Date.now(),
  };

  return {
    systemPrompt: FILE_ANALYSIS_SYSTEM_PROMPT,
    messages: [userPiMessage],
    tools: [],
  };
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

function resultContentParts(result: Pick<AssistantMessage, "content">): ContentPart[] {
  const parts: ContentPart[] = [];

  for (const block of result.content) {
    if (block.type === "text") {
      parts.push({ type: "text", text: block.text });
      continue;
    }

    if (block.type === "thinking") {
      parts.push({ type: "thinking", text: block.thinking });
    }
  }

  return parts;
}

function partsEndWith(parts: ContentPart[], suffix: ContentPart[]): boolean {
  if (suffix.length === 0 || suffix.length > parts.length) {
    return false;
  }

  const offset = parts.length - suffix.length;
  for (let index = 0; index < suffix.length; index += 1) {
    const left = parts[offset + index];
    const right = suffix[index];
    if (left.type !== right.type) {
      return false;
    }

    if (left.type === "tool_call" || right.type === "tool_call") {
      return false;
    }

    if (left.text !== right.text) {
      return false;
    }
  }

  return true;
}

function appendResultContentIfMissing(
  message: Message,
  result: Pick<AssistantMessage, "content">,
): Message {
  const suffix = resultContentParts(result);
  if (suffix.length === 0) {
    return message;
  }

  const parts = [...(message.parts ?? [])];
  if (partsEndWith(parts, suffix)) {
    return {
      ...message,
      content: contentFromParts(parts),
      parts,
    };
  }

  const nextParts = [...parts, ...suffix];
  return {
    ...message,
    content: contentFromParts(nextParts),
    parts: nextParts,
  };
}

function startSubagentMessage(message: Message): Message {
  return {
    ...setSubagentStatus(message, "running"),
    content: "",
    parts: [],
  };
}

function startSubagentMessageIfQueued(message: Message): Message {
  if (message.actor?.type === "subagent" && message.actor.status === "queued") {
    return startSubagentMessage(message);
  }
  return message;
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

function finalizeSubagentMessage(message: Message, status: "done" | "error"): Message {
  return setSubagentStatus(message, status);
}

function appendPendingToolCall(message: Message, callInfo: ToolCallInfo): Message {
  return {
    ...message,
    parts: [...(message.parts ?? []), { type: "tool_call", call: { ...callInfo } }],
  };
}

function applyToolCallResult(message: Message, callInfo: ToolCallInfo): Message {
  const parts = [...(message.parts ?? [])];
  for (let i = parts.length - 1; i >= 0; i -= 1) {
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

async function runFileSubagent({
  model,
  settings,
  controller,
  file,
  userQuestion,
  focus,
  range,
  updateMessage,
}: {
  model: ReturnType<typeof getModel>;
  settings: ReadyLlmSettings;
  controller: AbortController;
  file: QueuedAnalysisFile;
  userQuestion: string;
  focus?: string;
  range?: DocumentRange;
  updateMessage: (updater: (message: Message) => Message) => void;
}): Promise<SubagentAnalysisResult> {
  try {
    const document = await api.getDocument(file.collection, file.path, range);
    const context = createSubagentContext(userQuestion, document, focus, range);
    let streamError: string | undefined;

    updateMessage((message) => startSubagentMessage(message));

    const stream = streamSimple(model, context, {
      apiKey: settings.api_key,
      signal: controller.signal,
      reasoning: model.reasoning ? "medium" : undefined,
    });

    const consumed = await consumeAssistantStream(stream, {
      onTextDelta: (text) => {
        updateMessage((message) => applySubagentTextDelta(message, text));
      },
      onThinkingDelta: (thinking) => {
        updateMessage((message) => applySubagentThinkingDelta(message, thinking));
      },
      onError: (error) => {
        streamError = typeof error === "string" ? error : JSON.stringify(error);
        updateMessage((message) => {
          const next = applyStreamError(message, settings.provider, error);
          return finalizeSubagentMessage(next, "error");
        });
      },
    });

    updateMessage((message) => appendResultContentIfMissing(message, consumed.result));

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

    if (consumed.interrupted) {
      const interruptedError =
        consumed.result.errorMessage ?? "Response interrupted before completion.";
      updateMessage((message) => finalizeSubagentMessage(message, "error"));
      return {
        collection: file.collection,
        path: file.path,
        reason: file.reason,
        title: file.title,
        error: interruptedError,
      };
    }

    updateMessage((message) => finalizeSubagentMessage(message, "done"));
    return {
      collection: file.collection,
      path: file.path,
      reason: file.reason,
      title: file.title,
      text: consumed.text,
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

function parseRangeArg(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return undefined;
  }
  return Math.trunc(value);
}

function rangeFromToolArgs(args: Record<string, unknown>): DocumentRange | undefined {
  const range: DocumentRange = {
    startLine: parseRangeArg(args.start_line),
    endLine: parseRangeArg(args.end_line),
    startByte: parseRangeArg(args.start_byte),
    endByte: parseRangeArg(args.end_byte),
  };
  const hasAny =
    range.startLine !== undefined ||
    range.endLine !== undefined ||
    range.startByte !== undefined ||
    range.endByte !== undefined;
  return hasAny ? range : undefined;
}

async function runAnalyzeDocumentTool({
  call,
  model,
  settings,
  controller,
  userQuestion,
  piContext,
  runtimeState,
  queueSubagentMessage,
  updateSubagentMessage,
  createId,
}: {
  call: { id: string; name: string; arguments: Record<string, unknown> };
  model: ReturnType<typeof getModel>;
  settings: ReadyLlmSettings;
  controller: AbortController;
  userQuestion: string;
  piContext: Context;
  runtimeState: ChatToolRuntimeState;
  queueSubagentMessage: (file: QueuedAnalysisFile) => void;
  updateSubagentMessage: (messageId: string, updater: (message: Message) => Message) => void;
  createId: () => string;
}): Promise<ToolCallInfo> {
  const collection = String(call.arguments.collection ?? "").trim();
  const path = String(call.arguments.path ?? "").trim();
  const focus =
    typeof call.arguments.focus === "string" && call.arguments.focus.trim().length > 0
      ? call.arguments.focus.trim()
      : undefined;
  const range = rangeFromToolArgs(call.arguments);
  const messageId = createId();
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
    range,
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

  return callInfo;
}

export async function runParentAgentRound({
  model,
  settings,
  controller,
  userQuestion,
  piContext,
  updateAssistantMessage,
  runtimeState,
  queueSubagentMessage,
  updateSubagentMessage,
  createId,
}: {
  model: ReturnType<typeof getModel>;
  settings: ReadyLlmSettings;
  controller: AbortController;
  userQuestion: string;
  piContext: Context;
  updateAssistantMessage: UpdateAssistantMessage;
  runtimeState: ChatToolRuntimeState;
  queueSubagentMessage: (file: QueuedAnalysisFile) => void;
  updateSubagentMessage: (messageId: string, updater: (message: Message) => Message) => void;
  createId: () => string;
}): Promise<boolean> {
  const stream = streamSimple(model, piContext, {
    apiKey: settings.api_key,
    signal: controller.signal,
    reasoning: model.reasoning ? "medium" : undefined,
  });

  const consumed = await consumeAssistantStream(stream, {
    onTextDelta: (text, delta) => {
      updateAssistantMessage((message) => applyTextDelta(message, text, delta));
    },
    onThinkingDelta: (thinking) => {
      updateAssistantMessage((message) => applyThinkingDelta(message, thinking));
    },
    onError: (error) => {
      updateAssistantMessage((message) => applyStreamError(message, settings.provider, error));
    },
  });

  const result = consumed.result;
  updateAssistantMessage((message) => appendResultContentIfMissing(message, result));
  piContext.messages.push(result);

  if (isInterruptedAssistantResult(result)) {
    updateAssistantMessage((message) =>
      applyInterruptedStopReason(message, result.stopReason, result.errorMessage),
    );
    return false;
  }

  if (!shouldContinueAssistantToolRound(result)) {
    return false;
  }

  const toolCalls = assistantToolCalls(result);

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
            runtimeState,
            queueSubagentMessage,
            updateSubagentMessage,
            createId,
          })
        : await executeSearchToolCall({
            call: { id: call.id, name: call.name, arguments: callArgs },
            piContext,
            runtimeState,
          });

    updateAssistantMessage((message) => applyToolCallResult(message, resolvedCallInfo));
  }

  return true;
}

export function updateSubagentMessages(
  messages: Message[],
  messageId: string,
  updater: (message: Message) => Message,
): Message[] {
  return updateSubagentMessageById(messages, messageId, updater);
}

export function queueSubagentResult(messages: Message[], file: QueuedAnalysisFile): Message[] {
  return insertOrUpdateSubagentMessage(messages, createQueuedSubagentMessage(file.messageId, file));
}
