import type { SearchResult } from "../lib/api";

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

export interface ChatToolRuntimeState {
  currentTurnSearchResults: SearchResult[];
  acceptedAnalysisFiles: AnalyzeFilesAcceptedItem[];
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

export function formatAnalyzeFilesAcknowledgement(
  decision: AnalyzeFilesDecision,
): string {
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
