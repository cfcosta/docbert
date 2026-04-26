const BASE = "/v1";

function encodePathSegments(path: string): string {
  return path
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

export function encodeDocumentPath(path: string) {
  return encodePathSegments(path);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `HTTP ${res.status}`);
  }
  if (res.status === 204) return undefined as unknown as T;
  return res.json();
}

export interface Collection {
  name: string;
}

export interface IngestDoc {
  path: string;
  content: string;
  content_type: string;
  metadata?: Record<string, unknown>;
}

export interface IngestedDoc {
  doc_id: string;
  path: string;
  title: string;
  metadata?: Record<string, unknown>;
}

export interface IngestResponse {
  ingested: number;
  documents: IngestedDoc[];
}

export type SearchMode = "semantic" | "hybrid";

export interface SearchExcerpt {
  text: string;
  start_line: number;
  end_line: number;
}

/// Inclusive byte range of the chunk that scored highest for one search
/// hit. Surfaced when the result came from the semantic leg so the
/// chat agent can pass the same range to `analyze_document` and read
/// only the matching slice instead of the whole file.
export interface ChunkMatch {
  start_byte: number;
  end_byte: number;
}

export interface SearchResult {
  rank: number;
  score: number;
  doc_id: string;
  collection: string;
  path: string;
  title: string;
  metadata?: Record<string, unknown>;
  excerpts?: SearchExcerpt[];
  /// Total line count of the matching document, when readable. Lets the
  /// chat agent pick a follow-up start_line/end_line range for a partial
  /// read instead of always loading the whole file.
  line_count?: number;
  /// Total byte count of the matching document, when readable.
  byte_count?: number;
  /// Byte range of the matching chunk for semantic hits. Omitted for
  /// BM25-only matches and for documents indexed before chunk offsets
  /// were tracked.
  match_chunk?: ChunkMatch;
}

export interface SearchResponse {
  query: string;
  mode: SearchMode;
  result_count: number;
  results: SearchResult[];
}

export interface DocumentResponse {
  doc_id: string;
  collection: string;
  path: string;
  title: string;
  content: string;
  metadata?: Record<string, unknown>;
  /// Total line count of the un-sliced document. Always populated; sized
  /// off the full on-disk file regardless of any range slice that was
  /// applied to `content`.
  line_count?: number;
  /// Total byte count of the un-sliced document.
  byte_count?: number;
}

/// Optional inclusive range for `getDocument`. Line and byte ranges are
/// mutually exclusive — passing both makes the server respond `400`.
export interface DocumentRange {
  startLine?: number;
  endLine?: number;
  startByte?: number;
  endByte?: number;
}

export interface DocumentListItem {
  doc_id: string;
  path: string;
  title: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
}

export interface ChatSource {
  collection: string;
  path: string;
  title: string;
}

export type ChatPart =
  | { type: "text"; text: string }
  | { type: "thinking"; text: string }
  | {
      type: "tool_call";
      name: string;
      args: Record<string, unknown>;
      result?: string;
      is_error?: boolean;
    };

export type ChatSubagentStatus = "queued" | "running" | "done" | "error";

export type ChatActor =
  | { type: "parent" }
  | {
      type: "subagent";
      id: string;
      collection: string;
      path: string;
      status: ChatSubagentStatus;
    };

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  parts?: ChatPart[];
  actor?: ChatActor;
  content: string;
  sources?: ChatSource[];
}

export interface ConversationFull {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  messages: ChatMessage[];
}

export interface LlmSettings {
  provider: string | null;
  model: string | null;
  api_key: string | null;
  oauth_connected?: boolean;
  oauth_expires_at?: number | null;
}

export interface LlmSettingsUpdate {
  provider: string | null;
  model: string | null;
  api_key: string | null;
}

export interface LlmOauthStartResponse {
  authorization_url: string;
}

function buildDocumentRangeQuery(range?: DocumentRange): string {
  if (!range) return "";

  const params = new URLSearchParams();
  if (range.startLine !== undefined) params.set("startLine", String(range.startLine));
  if (range.endLine !== undefined) params.set("endLine", String(range.endLine));
  if (range.startByte !== undefined) params.set("startByte", String(range.startByte));
  if (range.endByte !== undefined) params.set("endByte", String(range.endByte));

  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

export function buildDocumentTabHref(collection: string, path: string): string {
  return `/documents/${encodeURIComponent(collection)}/${encodePathSegments(path)}`;
}

export function buildDocumentTabHrefWithFragment(
  collection: string,
  path: string,
  fragment: string,
): string {
  const normalizedFragment = fragment.startsWith("#") ? fragment.slice(1) : fragment;
  return `${buildDocumentTabHref(collection, path)}#${encodeURIComponent(normalizedFragment)}`;
}

export const api = {
  getLlmSettings: () => request<LlmSettings>("/settings/llm"),

  updateLlmSettings: (settings: LlmSettingsUpdate) =>
    request<LlmSettings>("/settings/llm", {
      method: "PUT",
      body: JSON.stringify(settings),
    }),

  startOpenAICodexOAuth: () =>
    request<LlmOauthStartResponse>("/settings/llm/oauth/openai-codex/start", {
      method: "POST",
    }),

  logoutOpenAICodexOAuth: () =>
    request<void>("/settings/llm/oauth/openai-codex/logout", {
      method: "POST",
    }),

  listCollections: () => request<Collection[]>("/collections"),

  listDocuments: (collection: string) =>
    request<DocumentListItem[]>(`/collections/${encodeURIComponent(collection)}/documents`),

  createCollection: (name: string) =>
    request<Collection>("/collections", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),

  deleteCollection: (name: string) =>
    request<void>(`/collections/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }),

  ingestDocuments: (collection: string, documents: IngestDoc[]) =>
    request<IngestResponse>("/documents", {
      method: "POST",
      body: JSON.stringify({ collection, documents }),
    }),

  getDocument: (collection: string, path: string, range?: DocumentRange) =>
    request<DocumentResponse>(
      `/documents/${encodeURIComponent(collection)}/${encodeDocumentPath(path)}${buildDocumentRangeQuery(range)}`,
    ),

  deleteDocument: (collection: string, path: string) =>
    request<void>(`/documents/${encodeURIComponent(collection)}/${encodeDocumentPath(path)}`, {
      method: "DELETE",
    }),

  listConversations: () => request<ConversationSummary[]>("/conversations"),

  createConversation: (id: string, title?: string) =>
    request<ConversationFull>("/conversations", {
      method: "POST",
      body: JSON.stringify({ id, title }),
    }),

  getConversation: (id: string) =>
    request<ConversationFull>(`/conversations/${encodeURIComponent(id)}`),

  updateConversation: (id: string, conv: ConversationFull) =>
    request<ConversationFull>(`/conversations/${encodeURIComponent(id)}`, {
      method: "PUT",
      body: JSON.stringify(conv),
    }),

  deleteConversation: (id: string) =>
    request<void>(`/conversations/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),

  search: (params: {
    query: string;
    mode?: SearchMode;
    collection?: string;
    count?: number;
    min_score?: number;
  }) =>
    request<SearchResponse>("/search", {
      method: "POST",
      body: JSON.stringify(params),
    }),
};
