import { describe, expect, test } from "bun:test";
import type { Tool } from "@mariozechner/pi-ai";

import { applyInterruptedStopReason, createPiContext } from "./chat-context";
import type { Message } from "./chat-message-codec";

const NO_TOOLS: Tool[] = [];

describe("chat-context", () => {
  test("createPiContext_skips_subagents_and_preserves_tool_turns", () => {
    const history: Message[] = [
      {
        id: "user-1",
        role: "user",
        content: "What is Rust?",
        parts: [{ type: "text", text: "What is Rust?" }],
      },
      {
        id: "assistant-1",
        role: "assistant",
        content: "Working...Done.",
        actor: { type: "parent" },
        parts: [
          { type: "text", text: "Working..." },
          {
            type: "tool_call",
            call: {
              name: "search_hybrid",
              args: { query: "rust" },
              result: "[]",
              isError: false,
            },
          },
          { type: "text", text: "Done." },
        ],
      },
      {
        id: "subagent-1",
        role: "assistant",
        content: "Subagent detail",
        actor: {
          type: "subagent",
          id: "subagent-1",
          collection: "notes",
          path: "rust.md",
          status: "done",
        },
        parts: [{ type: "text", text: "Subagent detail" }],
      },
    ];

    const context = createPiContext(history, "System prompt", NO_TOOLS);
    const messages = context.messages as Array<Record<string, unknown>>;

    expect(context.systemPrompt).toBe("System prompt");
    expect(messages).toHaveLength(4);
    expect(messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
      "toolResult",
      "assistant",
    ]);

    const firstAssistantContent = messages[1].content as Array<Record<string, unknown>>;
    expect(firstAssistantContent).toEqual([
      { type: "text", text: "Working..." },
      {
        type: "toolCall",
        id: "assistant-1:tool:0",
        name: "search_hybrid",
        arguments: { query: "rust" },
      },
    ]);

    const toolResultContent = messages[2].content as Array<Record<string, unknown>>;
    expect(toolResultContent).toEqual([{ type: "text", text: "[]" }]);

    const finalAssistantContent = messages[3].content as Array<Record<string, unknown>>;
    expect(finalAssistantContent).toEqual([{ type: "text", text: "Done." }]);
  });

  test("applyInterruptedStopReason_appends_note_once", () => {
    const message: Message = {
      id: "assistant-1",
      role: "assistant",
      content: "Partial answer",
      parts: [{ type: "text", text: "Partial answer" }],
    };

    const once = applyInterruptedStopReason(message, "aborted");
    const twice = applyInterruptedStopReason(once, "aborted");
    const expected = "Partial answer\n\nResponse interrupted before completion.";

    expect(once.content).toBe(expected);
    expect(once.parts).toEqual([{ type: "text", text: expected }]);
    expect(twice.content).toBe(expected);
    expect(twice.parts).toEqual([{ type: "text", text: expected }]);
  });
});
