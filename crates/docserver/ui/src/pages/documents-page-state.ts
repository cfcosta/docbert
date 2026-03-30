import type { DocumentListItem } from "../lib/api";
import type { SelectedDocumentSummary } from "./documents-tree";

export function toggleExpandedKey(expanded: Set<string>, key: string): Set<string> {
  const next = new Set(expanded);
  if (next.has(key)) {
    next.delete(key);
  } else {
    next.add(key);
  }
  return next;
}

export function removeDocumentFromDocs(
  docs: Record<string, DocumentListItem[]>,
  collection: string,
  path: string,
): Record<string, DocumentListItem[]> {
  return {
    ...docs,
    [collection]: (docs[collection] ?? []).filter((item) => item.path !== path),
  };
}

export function removeCollectionFromDocs(
  docs: Record<string, DocumentListItem[]>,
  collection: string,
): Record<string, DocumentListItem[]> {
  const next = { ...docs };
  delete next[collection];
  return next;
}

export function removeExpandedKey(expanded: Set<string>, key: string): Set<string> {
  const next = new Set(expanded);
  next.delete(key);
  return next;
}

export function clearDeletedDocumentSelection(
  selectedDoc: SelectedDocumentSummary | null,
  preview: string | null,
  collection: string,
  path: string,
): { selectedDoc: SelectedDocumentSummary | null; preview: string | null } {
  if (selectedDoc?.collection === collection && selectedDoc.path === path) {
    return { selectedDoc: null, preview: null };
  }

  return { selectedDoc, preview };
}

export function clearDeletedCollectionSelection(
  selectedDoc: SelectedDocumentSummary | null,
  preview: string | null,
  collection: string,
): { selectedDoc: SelectedDocumentSummary | null; preview: string | null } {
  if (selectedDoc?.collection === collection) {
    return { selectedDoc: null, preview: null };
  }

  return { selectedDoc, preview };
}
