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

export interface SearchResult {
  rank: number;
  score: number;
  doc_id: string;
  collection: string;
  path: string;
  title: string;
  metadata?: Record<string, unknown>;
  excerpts?: SearchExcerpt[];
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

  updateLlmSettings: (settings: LlmSettings) =>
    request<LlmSettings>("/settings/llm", {
      method: "PUT",
      body: JSON.stringify(settings),
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

  getDocument: (collection: string, path: string) =>
    request<DocumentResponse>(
      `/documents/${encodeURIComponent(collection)}/${encodeDocumentPath(path)}`,
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
