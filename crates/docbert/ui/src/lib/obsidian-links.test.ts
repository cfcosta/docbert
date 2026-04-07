import { describe, expect, test } from "bun:test";

import { parseObsidianLink, resolveObsidianLink } from "./obsidian-links";

describe("obsidian link parsing", () => {
  test("parse_alias_heading_and_block_ref_forms", () => {
    expect(parseObsidianLink("[[Note|Alias]]")).toEqual({
      raw: "[[Note|Alias]]",
      target: "Note",
      fragment: null,
      alias: "Alias",
    });

    expect(parseObsidianLink("[[Note#Section]]")).toEqual({
      raw: "[[Note#Section]]",
      target: "Note",
      fragment: "Section",
      alias: null,
    });

    expect(parseObsidianLink("[[Note#^block-id]]")).toEqual({
      raw: "[[Note#^block-id]]",
      target: "Note",
      fragment: "^block-id",
      alias: null,
    });
  });

  test("rejects_non_wikilinks_and_empty_destinations", () => {
    expect(parseObsidianLink("Note")).toBeNull();
    expect(parseObsidianLink("[[]]")).toBeNull();
    expect(parseObsidianLink("[[|Alias]]")).toBeNull();
  });
});

describe("obsidian link resolution", () => {
  const documents = [
    { path: "Note.md" },
    { path: "nested/Guide.md" },
    { path: "folder/Match.md" },
    { path: "other/Match.md" },
  ];

  test("resolves_file_links_by_path_and_preserves_alias", () => {
    expect(
      resolveObsidianLink("[[nested/Guide.md|Read guide]]", {
        collection: "notes",
        currentPath: "today.md",
        documents,
      }),
    ).toEqual({
      collection: "notes",
      path: "nested/Guide.md",
      fragment: null,
      alias: "Read guide",
    });
  });

  test("resolves_same_note_heading_and_block_refs_without_lookup", () => {
    expect(
      resolveObsidianLink("[[#Section]]", {
        collection: "notes",
        currentPath: "today.md",
        documents,
      }),
    ).toEqual({
      collection: "notes",
      path: "today.md",
      fragment: "Section",
      alias: null,
    });

    expect(
      resolveObsidianLink("[[#^block-id]]", {
        collection: "notes",
        currentPath: "today.md",
        documents,
      }),
    ).toEqual({
      collection: "notes",
      path: "today.md",
      fragment: "^block-id",
      alias: null,
    });
  });

  test("resolves_bare_links_by_unique_path_stem_first", () => {
    expect(
      resolveObsidianLink("[[Guide#Overview]]", {
        collection: "notes",
        currentPath: "today.md",
        documents,
      }),
    ).toEqual({
      collection: "notes",
      path: "nested/Guide.md",
      fragment: "Overview",
      alias: null,
    });
  });

  test("returns_null_for_ambiguous_or_unresolved_targets", () => {
    expect(
      resolveObsidianLink("[[Match]]", {
        collection: "notes",
        currentPath: "today.md",
        documents,
      }),
    ).toBeNull();

    expect(
      resolveObsidianLink("[[Missing]]", {
        collection: "notes",
        currentPath: "today.md",
        documents,
      }),
    ).toBeNull();
  });

  test("stays_within_the_current_collection_lookup", () => {
    expect(
      resolveObsidianLink("[[External]]", {
        collection: "notes",
        currentPath: "today.md",
        documents: [{ path: "Local.md" }],
      }),
    ).toBeNull();
  });
});
