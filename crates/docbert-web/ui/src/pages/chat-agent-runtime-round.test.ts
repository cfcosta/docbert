import { describe, expect, test, mock, beforeEach } from "bun:test";
import type { AssistantMessage, AssistantMessageEventStream, Context } from "@mariozechner/pi-ai";

import type { SearchResult } from "../lib/api";
import type { Message } from "./chat-message-codec";
import type { ChatToolRuntimeState, QueuedAnalysisFile } from "./chat-subagents";
import { runParentAgentRound, createAssistantPlaceholder } from "./chat-agent-runtime";
import type { ReadyLlmSettings } from "./chat-agent-runtime";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function assistantResult(
  stopReason: AssistantMessage["stopReason"],
  content: AssistantMessage["content"] = [],
): AssistantMessage {
  return {
    role: "assistant",
    api: "openai-responses",
    provider: "test",
    model: "test-model",
    content,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason,
    timestamp: Date.now(),
  } as AssistantMessage;
}

function makeStream(
  events: Array<Record<string, unknown>>,
  result: AssistantMessage,
): AssistantMessageEventStream {
  return {
    async *[Symbol.asyncIterator]() {
      for (const event of events) {
        yield event;
      }
    },
    result: async () => result,
  } as AssistantMessageEventStream;
}

function emptyContext(): Context {
  return { messages: [], tools: [] };
}

function runtimeState(results: SearchResult[] = []): ChatToolRuntimeState {
  return { currentTurnSearchResults: results, queuedAnalysisFiles: [] };
}

const testSettings: ReadyLlmSettings = {
  provider: "test",
  model: "test-model",
  api_key: "test-key",
};

// Runtime-mocked `getModel` only exposes `{ reasoning: false }`, which
// is a narrower shape than pi-ai's real `ModelMetadata`. We cast
// through the `runParentAgentRound` parameter type so test call sites
// don't have to repeat `as any` per invocation.
type TestModel = Parameters<typeof runParentAgentRound>[0]["model"];
const testModel = { reasoning: false } as unknown as TestModel;

function searchResult(collection: string, path: string, title: string): SearchResult {
  return {
    rank: 1,
    score: 0.9,
    doc_id: "#abc123",
    collection,
    path,
    title,
    excerpts: [],
  };
}

// ---------------------------------------------------------------------------
// Module-level mock for streamSimple
// ---------------------------------------------------------------------------

let nextStream: AssistantMessageEventStream;

mock.module("@mariozechner/pi-ai", () => ({
  Type: {
    Object: (schema: unknown) => schema,
    String: (meta?: unknown) => meta ?? {},
    Optional: (t: unknown) => t,
    Number: (meta?: unknown) => meta ?? {},
  },
  getModel: () => ({ reasoning: false }),
  streamSimple: () => nextStream,
}));

// Mock the search API so executeSearchToolCall doesn't hit the network
mock.module("../lib/api", () => ({
  api: {
    search: async ({ query }: { query: string }) => ({
      query,
      mode: "hybrid",
      result_count: 1,
      results: [searchResult("notes", "result.md", "Search Result")],
    }),
    getDocument: async () => ({
      doc_id: "#abc",
      collection: "notes",
      path: "result.md",
      title: "Result",
      content: "# Result\n\nBody",
    }),
  },
}));

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("runParentAgentRound", () => {
  let assistantMessage: Message;
  let piContext: Context;
  let state: ChatToolRuntimeState;
  let queuedFiles: QueuedAnalysisFile[];
  let idCounter: number;

  function updateAssistant(fn: (m: Message) => Message) {
    assistantMessage = fn(assistantMessage);
  }

  function queueSubagent(file: QueuedAnalysisFile) {
    queuedFiles.push(file);
  }

  function updateSubagent() {
    // no-op for parent round tests
  }

  function createId() {
    return `id-${idCounter++}`;
  }

  beforeEach(() => {
    assistantMessage = createAssistantPlaceholder("asst-1");
    piContext = emptyContext();
    state = runtimeState();
    queuedFiles = [];
    idCounter = 0;
  });

  test("returns false when the assistant stops without tool calls", async () => {
    const result = assistantResult("stop", [{ type: "text", text: "Here is the answer." }]);
    nextStream = makeStream([{ type: "text_delta", delta: "Here is the answer." }], result);

    const shouldContinue = await runParentAgentRound({
      model: testModel,
      settings: testSettings,
      controller: new AbortController(),
      userQuestion: "What is X?",
      piContext,
      updateAssistantMessage: updateAssistant,
      runtimeState: state,
      queueSubagentMessage: queueSubagent,
      updateSubagentMessage: updateSubagent,
      createId,
    });

    expect(shouldContinue).toBe(false);
    expect(assistantMessage.content).toContain("Here is the answer.");
    // The result should be appended to piContext.messages
    expect(piContext.messages).toHaveLength(1);
    expect(piContext.messages[0].role).toBe("assistant");
  });

  test("returns false on interrupted (aborted) response", async () => {
    const result = assistantResult("aborted", []);
    nextStream = makeStream([], result);

    const shouldContinue = await runParentAgentRound({
      model: testModel,
      settings: testSettings,
      controller: new AbortController(),
      userQuestion: "What is X?",
      piContext,
      updateAssistantMessage: updateAssistant,
      runtimeState: state,
      queueSubagentMessage: queueSubagent,
      updateSubagentMessage: updateSubagent,
      createId,
    });

    expect(shouldContinue).toBe(false);
  });

  test("returns true and executes search tool when assistant makes a tool call", async () => {
    const toolCallContent: AssistantMessage["content"] = [
      {
        type: "toolCall",
        id: "call-1",
        name: "search_hybrid",
        arguments: { query: "deployment steps" },
      },
    ];
    const result = assistantResult("toolCall", toolCallContent);
    nextStream = makeStream([], result);

    const shouldContinue = await runParentAgentRound({
      model: testModel,
      settings: testSettings,
      controller: new AbortController(),
      userQuestion: "How to deploy?",
      piContext,
      updateAssistantMessage: updateAssistant,
      runtimeState: state,
      queueSubagentMessage: queueSubagent,
      updateSubagentMessage: updateSubagent,
      createId,
    });

    // Should continue for another round after tool execution
    expect(shouldContinue).toBe(true);

    // piContext should have the assistant result + tool result
    expect(piContext.messages.length).toBeGreaterThanOrEqual(2);
    expect(piContext.messages[0].role).toBe("assistant");
    expect(piContext.messages[1].role).toBe("toolResult");

    // runtimeState should have search results populated
    expect(state.currentTurnSearchResults.length).toBeGreaterThan(0);
    expect(state.currentTurnSearchResults[0].collection).toBe("notes");

    // The assistant message should have the pending tool call part
    const toolParts = (assistantMessage.parts ?? []).filter((p) => p.type === "tool_call");
    expect(toolParts).toHaveLength(1);
    expect(toolParts[0].type).toBe("tool_call");
    if (toolParts[0].type === "tool_call") {
      expect(toolParts[0].call.name).toBe("search_hybrid");
      // After execution, result should be populated
      expect(toolParts[0].call.result).toBeDefined();
    }
  });

  test("handles multiple tool calls in a single round", async () => {
    const toolCallContent: AssistantMessage["content"] = [
      {
        type: "toolCall",
        id: "call-1",
        name: "search_hybrid",
        arguments: { query: "architecture" },
      },
      {
        type: "toolCall",
        id: "call-2",
        name: "search_semantic",
        arguments: { query: "design patterns" },
      },
    ];
    const result = assistantResult("toolCall", toolCallContent);
    nextStream = makeStream([], result);

    const shouldContinue = await runParentAgentRound({
      model: testModel,
      settings: testSettings,
      controller: new AbortController(),
      userQuestion: "Tell me about the architecture",
      piContext,
      updateAssistantMessage: updateAssistant,
      runtimeState: state,
      queueSubagentMessage: queueSubagent,
      updateSubagentMessage: updateSubagent,
      createId,
    });

    expect(shouldContinue).toBe(true);

    // assistant result + 2 tool results
    expect(piContext.messages).toHaveLength(3);
    expect(piContext.messages[1].role).toBe("toolResult");
    expect(piContext.messages[2].role).toBe("toolResult");

    // Both tool calls should appear in assistant message parts
    const toolParts = (assistantMessage.parts ?? []).filter((p) => p.type === "tool_call");
    expect(toolParts).toHaveLength(2);
  });

  test("text content from stream is captured before tool calls execute", async () => {
    const toolCallContent: AssistantMessage["content"] = [
      { type: "text", text: "Let me search for that." },
      {
        type: "toolCall",
        id: "call-1",
        name: "search_hybrid",
        arguments: { query: "foo" },
      },
    ];
    const result = assistantResult("toolCall", toolCallContent);
    nextStream = makeStream([{ type: "text_delta", delta: "Let me search for that." }], result);

    await runParentAgentRound({
      model: testModel,
      settings: testSettings,
      controller: new AbortController(),
      userQuestion: "Find foo",
      piContext,
      updateAssistantMessage: updateAssistant,
      runtimeState: state,
      queueSubagentMessage: queueSubagent,
      updateSubagentMessage: updateSubagent,
      createId,
    });

    // The text should be in the message content
    expect(assistantMessage.content).toContain("Let me search for that.");
    // And the tool call part should follow
    const parts = assistantMessage.parts ?? [];
    const textParts = parts.filter((p) => p.type === "text");
    const toolParts = parts.filter((p) => p.type === "tool_call");
    expect(textParts.length).toBeGreaterThan(0);
    expect(toolParts).toHaveLength(1);
  });
});
