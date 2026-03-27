import { describe, expect, test } from "bun:test";

import type { SearchResult } from "../lib/api";
import {
  decideAnalyzeFiles,
  formatAnalyzeFilesAcknowledgement,
  mergeCurrentTurnSearchResults,
} from "./chat-subagents";

function result(
  collection: string,
  path: string,
  title: string,
): SearchResult {
  return {
    rank: 1,
    score: 1,
    doc_id: `${collection}:${path}`,
    collection,
    path,
    title,
  };
}

describe("mergeCurrentTurnSearchResults", () => {
  test("deduplicates by collection and path", () => {
    const merged = mergeCurrentTurnSearchResults(
      [result("notes", "a.md", "A")],
      [result("notes", "a.md", "A2"), result("notes", "b.md", "B")],
    );

    expect(merged).toHaveLength(2);
    expect(merged.map((entry) => `${entry.collection}:${entry.path}`)).toEqual([
      "notes:a.md",
      "notes:b.md",
    ]);
  });
});

describe("decideAnalyzeFiles", () => {
  const available = [
    result("notes", "a.md", "A"),
    result("notes", "b.md", "B"),
    result("notes", "c.md", "C"),
    result("notes", "d.md", "D"),
  ];

  test("rejects duplicate files", () => {
    const decision = decideAnalyzeFiles(
      {
        files: [
          { collection: "notes", path: "a.md", reason: "first" },
          { collection: "notes", path: "a.md", reason: "duplicate" },
        ],
      },
      available,
    );

    expect(decision.accepted).toHaveLength(1);
    expect(decision.rejected).toEqual([
      {
        collection: "notes",
        path: "a.md",
        reason: "duplicate",
        rejection_reason: "duplicate_file",
      },
    ]);
  });

  test("caps accepted files at three", () => {
    const decision = decideAnalyzeFiles(
      {
        files: [
          { collection: "notes", path: "a.md", reason: "one" },
          { collection: "notes", path: "b.md", reason: "two" },
          { collection: "notes", path: "c.md", reason: "three" },
          { collection: "notes", path: "d.md", reason: "four" },
        ],
      },
      available,
    );

    expect(decision.accepted).toHaveLength(3);
    expect(decision.capped).toBe(true);
    expect(decision.rejected).toEqual([
      {
        collection: "notes",
        path: "d.md",
        reason: "four",
        rejection_reason: "max_files_exceeded",
      },
    ]);
  });

  test("rejects files missing from current-turn search results", () => {
    const decision = decideAnalyzeFiles(
      {
        files: [{ collection: "notes", path: "missing.md", reason: "check this" }],
      },
      available,
    );

    expect(decision.accepted).toHaveLength(0);
    expect(decision.rejected).toEqual([
      {
        collection: "notes",
        path: "missing.md",
        reason: "check this",
        rejection_reason: "not_in_current_turn_search_results",
      },
    ]);
  });

  test("acknowledges valid payloads", () => {
    const decision = decideAnalyzeFiles(
      {
        files: [
          { collection: "notes", path: "a.md", reason: "important" },
          { collection: "notes", path: "b.md", reason: "also relevant" },
        ],
      },
      available,
    );

    expect(decision.accepted).toEqual([
      {
        collection: "notes",
        path: "a.md",
        reason: "important",
        title: "A",
      },
      {
        collection: "notes",
        path: "b.md",
        reason: "also relevant",
        title: "B",
      },
    ]);
    expect(JSON.parse(formatAnalyzeFilesAcknowledgement(decision))).toEqual({
      accepted: decision.accepted,
      rejected: [],
      capped: false,
    });
  });
});
