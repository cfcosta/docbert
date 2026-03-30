import { describe, expect, test } from "bun:test";

import {
  clearDeletedDocumentSelection,
  toggleExpandedKey,
} from "./documents-page-state";

describe("documents-page-state", () => {
  test("toggle_collection_and_directory_keys_remain_independent", () => {
    const expandedCollection = toggleExpandedKey(new Set<string>(), "notes");
    expect([...expandedCollection]).toEqual(["notes"]);

    const expandedDirectory = toggleExpandedKey(expandedCollection, "notes/nested");
    expect([...expandedDirectory].sort()).toEqual(["notes", "notes/nested"]);

    const collapsedCollection = toggleExpandedKey(expandedDirectory, "notes");
    expect([...collapsedCollection]).toEqual(["notes/nested"]);
  });

  test("deleting_active_document_clears_selection_only_for_the_target_document", () => {
    const selectedDoc = {
      collection: "notes",
      path: "hello.md",
      title: "Hello",
      doc_id: "#abc123",
    };

    expect(
      clearDeletedDocumentSelection(selectedDoc, "preview body", "notes", "hello.md"),
    ).toEqual({ selectedDoc: null, preview: null });

    expect(
      clearDeletedDocumentSelection(selectedDoc, "preview body", "notes", "other.md"),
    ).toEqual({ selectedDoc, preview: "preview body" });

    expect(
      clearDeletedDocumentSelection(selectedDoc, "preview body", "docs", "hello.md"),
    ).toEqual({ selectedDoc, preview: "preview body" });
  });
});
