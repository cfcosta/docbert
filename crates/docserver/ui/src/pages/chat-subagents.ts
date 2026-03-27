import type { ChatActor, ChatSubagentStatus, SearchResult } from "../lib/api";

export interface AnalyzeFilesRequestItem {
  collection: string;
  path: string;
  reason: string;
}

export interface AnalyzeFilesAcceptedItem extends AnalyzeFilesRequestItem {
  title?: string;
}

export interface AnalyzeFilesRejectedItem {
  collection?: string;
  path?: string;
  reason?: string;
  rejection_reason: string;
}

export interface AnalyzeFilesDecision {
  accepted: AnalyzeFilesAcceptedItem[];
  rejected: AnalyzeFilesRejectedItem[];
  capped: boolean;
}

export interface AutoAnalyzeTopKDecision {
  accepted: AnalyzeFilesAcceptedItem[];
  available: number;
}

export interface QueuedAnalysisFile extends AnalyzeFilesAcceptedItem {
  messageId: string;
}

export interface SubagentAnalysisResult extends AnalyzeFilesAcceptedItem {
  text?: string;
  error?: string;
}

export interface SynthesisPayloadFile extends AnalyzeFilesAcceptedItem {
  analysis?: string;
  error?: string;
}

export interface SynthesisPayload {
  userQuestion: string;
  files: SynthesisPayloadFile[];
  sourceFiles: Array<Pick<AnalyzeFilesAcceptedItem, "collection" | "path" | "title">>;
}

export interface ChatToolRuntimeState {
  currentTurnSearchResults: SearchResult[];
  acceptedAnalysisFiles: AnalyzeFilesAcceptedItem[];
  queuedAnalysisFiles: QueuedAnalysisFile[];
}

export interface SubagentTranscriptPart {
  type: "text" | "thinking";
  text: string;
}

export interface SubagentTranscriptMessage<TPart = unknown> {
  id: string;
  role: "user" | "assistant";
  content: string;
  actor?: ChatActor;
  parts?: TPart[];
}

const MAX_ANALYZE_FILES = 3;

function normalizeFile(item: unknown): AnalyzeFilesRequestItem | null {
  if (!item || typeof item !== "object") {
    return null;
  }

  const record = item as Record<string, unknown>;
  if (
    typeof record.collection !== "string" ||
    typeof record.path !== "string" ||
    typeof record.reason !== "string"
  ) {
    return null;
  }

  const collection = record.collection.trim();
  const path = record.path.trim();
  const reason = record.reason.trim();

  if (!collection || !path || !reason) {
    return null;
  }

  return { collection, path, reason };
}

function uniqueSearchResults(results: SearchResult[]): SearchResult[] {
  const seen = new Set<string>();
  return results.filter((result) => {
    const key = `${result.collection}:${result.path}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

export function mergeCurrentTurnSearchResults(
  current: SearchResult[],
  incoming: SearchResult[],
): SearchResult[] {
  return uniqueSearchResults([...current, ...incoming]);
}

export function decideAnalyzeFiles(
  args: Record<string, unknown>,
  availableResults: SearchResult[],
): AnalyzeFilesDecision {
  const files = Array.isArray(args.files) ? args.files : [];
  const availableLookup = new Map(
    availableResults.map((result) => [`${result.collection}:${result.path}`, result]),
  );
  const accepted: AnalyzeFilesAcceptedItem[] = [];
  const rejected: AnalyzeFilesRejectedItem[] = [];
  const seen = new Set<string>();
  let capped = false;

  for (const rawFile of files) {
    const normalized = normalizeFile(rawFile);
    if (!normalized) {
      rejected.push({ rejection_reason: "invalid_file_entry" });
      continue;
    }

    const key = `${normalized.collection}:${normalized.path}`;
    if (seen.has(key)) {
      rejected.push({ ...normalized, rejection_reason: "duplicate_file" });
      continue;
    }
    seen.add(key);

    const matched = availableLookup.get(key);
    if (!matched) {
      rejected.push({ ...normalized, rejection_reason: "not_in_current_turn_search_results" });
      continue;
    }

    if (accepted.length >= MAX_ANALYZE_FILES) {
      capped = true;
      rejected.push({ ...normalized, rejection_reason: "max_files_exceeded" });
      continue;
    }

    accepted.push({ ...normalized, title: matched.title });
  }

  if (accepted.length === 0 && files.length === 0) {
    rejected.push({ rejection_reason: "missing_files" });
  }

  return { accepted, rejected, capped };
}

export function formatAnalyzeFilesAcknowledgement(decision: AnalyzeFilesDecision): string {
  return JSON.stringify(
    {
      accepted: decision.accepted,
      rejected: decision.rejected,
      capped: decision.capped,
    },
    null,
    2,
  );
}

export function selectTopSearchResultsForAnalysis(
  availableResults: SearchResult[],
  maxFiles: number,
): AutoAnalyzeTopKDecision {
  const accepted = uniqueSearchResults(availableResults)
    .slice(0, Math.max(0, maxFiles))
    .map((result, index) => ({
      collection: result.collection,
      path: result.path,
      title: result.title,
      reason: `Auto-selected from top search results (rank ${index + 1}).`,
    }));

  return {
    accepted,
    available: uniqueSearchResults(availableResults).length,
  };
}

export function insertOrUpdateSubagentMessage<T extends SubagentTranscriptMessage>(
  messages: T[],
  nextMessage: T,
): T[] {
  const index = messages.findIndex((message) => message.id === nextMessage.id);
  if (index === -1) {
    return [...messages, nextMessage];
  }

  const updated = [...messages];
  updated[index] = nextMessage;
  return updated;
}

export function setSubagentStatus<T extends SubagentTranscriptMessage>(
  message: T,
  status: ChatSubagentStatus,
): T {
  if (message.actor?.type !== "subagent") {
    return message;
  }

  return {
    ...message,
    actor: {
      ...message.actor,
      status,
    },
  };
}

export function upsertSubagentPart<T extends SubagentTranscriptMessage>(
  message: T,
  part: SubagentTranscriptPart,
): T {
  const parts = [...((message.parts ?? []) as SubagentTranscriptPart[])];
  const last = parts[parts.length - 1];
  if (last && last.type === part.type) {
    parts[parts.length - 1] = part;
  } else {
    parts.push(part);
  }

  const content = parts
    .filter((entry) => entry.type === "text")
    .map((entry) => entry.text)
    .join("");

  return {
    ...message,
    content,
    parts: parts as T["parts"],
  };
}

export function updateSubagentMessageById<T extends SubagentTranscriptMessage>(
  messages: T[],
  messageId: string,
  updater: (message: T) => T,
): T[] {
  return messages.map((message) => (message.id === messageId ? updater(message) : message));
}

export function buildSynthesisPayload({
  userQuestion,
  acceptedFiles,
  subagentResults,
}: {
  userQuestion: string;
  acceptedFiles: AnalyzeFilesAcceptedItem[];
  subagentResults: SubagentAnalysisResult[];
}): SynthesisPayload {
  const resultLookup = new Map(
    subagentResults.map((result) => [`${result.collection}:${result.path}`, result]),
  );
  const files: SynthesisPayloadFile[] = [];
  const sourceFiles: Array<Pick<AnalyzeFilesAcceptedItem, "collection" | "path" | "title">> = [];

  for (const file of acceptedFiles) {
    const result = resultLookup.get(`${file.collection}:${file.path}`);
    const payloadFile: SynthesisPayloadFile = {
      collection: file.collection,
      path: file.path,
      reason: file.reason,
      title: file.title,
    };

    if (result?.error) {
      payloadFile.error = result.error;
    } else if (result?.text && result.text.trim().length > 0) {
      payloadFile.analysis = result.text;
      sourceFiles.push({
        collection: file.collection,
        path: file.path,
        title: file.title,
      });
    } else {
      payloadFile.error = "missing_subagent_result";
    }

    files.push(payloadFile);
  }

  return {
    userQuestion,
    files,
    sourceFiles,
  };
}

export function queueAcceptedSubagentMessages<T extends SubagentTranscriptMessage>({
  messages,
  acceptedFiles,
  queuedFiles,
  createMessageId,
  createMessage,
}: {
  messages: T[];
  acceptedFiles: AnalyzeFilesAcceptedItem[];
  queuedFiles: QueuedAnalysisFile[];
  createMessageId: () => string;
  createMessage: (messageId: string, file: AnalyzeFilesAcceptedItem) => T;
}): { messages: T[]; queuedFiles: QueuedAnalysisFile[] } {
  let nextMessages = messages;
  const nextQueuedFiles = [...queuedFiles];

  for (const file of acceptedFiles) {
    const existing = nextQueuedFiles.find(
      (queued) => queued.collection === file.collection && queued.path === file.path,
    );
    if (existing) {
      continue;
    }

    const messageId = createMessageId();
    nextQueuedFiles.push({ ...file, messageId });
    nextMessages = insertOrUpdateSubagentMessage(nextMessages, createMessage(messageId, file));
  }

  return { messages: nextMessages, queuedFiles: nextQueuedFiles };
}
