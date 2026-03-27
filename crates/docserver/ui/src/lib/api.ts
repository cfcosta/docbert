const BASE = "/v1";

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

export interface SearchResult {
  rank: number;
  score: number;
  doc_id: string;
  collection: string;
  path: string;
  title: string;
  metadata?: Record<string, unknown>;
}

export interface SearchResponse {
  query: string;
  mode: string;
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

export interface LlmSettings {
  provider: string | null;
  model: string | null;
  api_key: string | null;
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
    request<DocumentResponse>(`/documents/${encodeURIComponent(collection)}/${path}`),

  deleteDocument: (collection: string, path: string) =>
    request<void>(`/documents/${encodeURIComponent(collection)}/${path}`, { method: "DELETE" }),

  search: (params: {
    query: string;
    mode?: string;
    collection?: string;
    count?: number;
    min_score?: number;
  }) =>
    request<SearchResponse>("/search", {
      method: "POST",
      body: JSON.stringify(params),
    }),
};
