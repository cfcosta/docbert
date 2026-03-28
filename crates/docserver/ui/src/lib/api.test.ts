import { describe, expect, test } from "bun:test";

import { buildDocumentTabHref, encodeDocumentPath } from "./api";

describe("document path encoding", () => {
  test("encode_document_path_preserves_nested_segments", () => {
    expect(encodeDocumentPath("notes/a folder/file.md")).toBe("notes/a%20folder/file.md");
  });

  test("encode_document_path_encodes_spaces_unicode_hash_and_query_chars", () => {
    expect(encodeDocumentPath("café notes/file #1?.md")).toBe(
      "caf%C3%A9%20notes/file%20%231%3F.md",
    );
  });

  test("build_document_tab_href_matches_endpoint_encoding_for_same_path", () => {
    const collection = "my notes";
    const path = "nested folder/file #1?.md";
    const encodedPath = encodeDocumentPath(path);

    expect(buildDocumentTabHref(collection, path)).toBe(
      `/documents/${encodeURIComponent(collection)}/${encodedPath}`,
    );
  });

  test("build_document_tab_href_does_not_encode_path_separators", () => {
    expect(buildDocumentTabHref("notes", "nested/path/file.md")).toBe(
      "/documents/notes/nested/path/file.md",
    );
  });
});
