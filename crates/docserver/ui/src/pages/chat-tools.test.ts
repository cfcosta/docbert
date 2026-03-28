import { describe, expect, test } from "bun:test";
import type { Context } from "@mariozechner/pi-ai";

import type { SearchResult } from "../lib/api";
import { executeSearchToolCall, searchChatTools } from "./chat-tools";
import type { ChatToolRuntimeState } from "./chat-subagents";

function runtimeState(results: SearchResult[] = []): ChatToolRuntimeState {
  return {
    currentTurnSearchResults: results,
    queuedAnalysisFiles: [],
  };
}

function emptyContext(): Context {
  return {
    messages: [],
    tools: [],
  };
}

function result(collection: string, path: string, title: string): SearchResult {
  return {
    rank: 1,
    score: 1,
    doc_id: `${collection}:${path}`,
    collection,
    path,
    title,
  };
}

describe("chat-tools", () => {
  test("executeSearchToolCall_returns_json_text_and_merges_current_turn_results", async () => {
    const state = runtimeState([result("notes", "a.md", "A")]);
    const context = emptyContext();
    const callInfo = await executeSearchToolCall({
      call: {
        id: "tool-1",
        name: "search_hybrid",
        arguments: { query: "rust" },
      },
      piContext: context,
      runtimeState: state,
      search: async () => ({
        query: "rust",
        mode: "hybrid",
        result_count: 2,
        results: [result("notes", "a.md", "A duplicate"), result("notes", "b.md", "B")],
      }),
    });

    expect(callInfo.result).toBe(
      JSON.stringify(
        [result("notes", "a.md", "A duplicate"), result("notes", "b.md", "B")],
        null,
        2,
      ),
    );
    expect(
      state.currentTurnSearchResults.map((entry) => `${entry.collection}:${entry.path}`),
    ).toEqual(["notes:a.md", "notes:b.md"]);
  });

  test("executeSearchToolCall_records_tool_results_for_success_and_error", async () => {
    const successContext = emptyContext();
    const successState = runtimeState();
    const success = await executeSearchToolCall({
      call: {
        id: "tool-1",
        name: "search_semantic",
        arguments: { query: "rust" },
      },
      piContext: successContext,
      runtimeState: successState,
      search: async () => ({
        query: "rust",
        mode: "semantic",
        result_count: 1,
        results: [result("notes", "a.md", "A")],
      }),
    });

    expect(success.isError).toBeUndefined();
    expect(successContext.messages).toHaveLength(1);
    expect(successContext.messages[0]).toMatchObject({
      role: "toolResult",
      toolCallId: "tool-1",
      toolName: "search_semantic",
      isError: false,
      content: [{ type: "text", text: success.result }],
    });

    const errorContext = emptyContext();
    const errorState = runtimeState();
    const failure = await executeSearchToolCall({
      call: {
        id: "tool-2",
        name: "search_hybrid",
        arguments: { query: "rust" },
      },
      piContext: errorContext,
      runtimeState: errorState,
      search: async () => {
        throw new Error("search failed");
      },
    });

    expect(failure.isError).toBe(true);
    expect(failure.result).toBe("Error: search failed");
    expect(errorContext.messages).toHaveLength(1);
    expect(errorContext.messages[0]).toMatchObject({
      role: "toolResult",
      toolCallId: "tool-2",
      toolName: "search_hybrid",
      isError: true,
      content: [{ type: "text", text: "Error: search failed" }],
    });
  });

  test("searchChatTools_keeps_only_search_tool_definitions", () => {
    expect(searchChatTools.map((tool) => tool.name)).toEqual(["search_semantic", "search_hybrid"]);
  });
});
