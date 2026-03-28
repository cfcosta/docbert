import { describe, expect, test } from "bun:test";

import type { Message } from "./chat-message-codec";
import { buildTranscriptRenderItems } from "./chat-transcript-model";
import type { SubagentMessage } from "./chat-message-groups";

function subagentMessage(id: string): SubagentMessage {
  return {
    id,
    role: "assistant",
    content: `subagent ${id}`,
    actor: {
      type: "subagent",
      id,
      collection: "notes",
      path: `${id}.md`,
      status: "done",
    },
    parts: [{ type: "text", text: `subagent ${id}` }],
  };
}

describe("chat-transcript-model", () => {
  test("pairs_one_analyze_tool_call_with_one_nested_subagent", () => {
    const items = buildTranscriptRenderItems(
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        parts: [
          {
            type: "tool_call",
            call: {
              name: "analyze_document",
              args: { collection: "notes", path: "a.md" },
            },
          },
        ],
      },
      [subagentMessage("sub-1")],
    );

    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({ kind: "subagent", message: { id: "sub-1" } });
  });

  test("pairs_multiple_analyze_tool_calls_in_order", () => {
    const items = buildTranscriptRenderItems(
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        parts: [
          {
            type: "tool_call",
            call: {
              name: "analyze_document",
              args: { collection: "notes", path: "a.md" },
            },
          },
          { type: "text", text: "between" },
          {
            type: "tool_call",
            call: {
              name: "analyze_document",
              args: { collection: "notes", path: "b.md" },
            },
          },
        ],
      },
      [subagentMessage("sub-1"), subagentMessage("sub-2")],
    );

    expect(items.map((item) => item.kind)).toEqual(["subagent", "text", "subagent"]);
    expect(items[0]).toMatchObject({ kind: "subagent", message: { id: "sub-1" } });
    expect(items[2]).toMatchObject({ kind: "subagent", message: { id: "sub-2" } });
  });

  test("appends_extra_trailing_subagents_after_message_parts", () => {
    const items = buildTranscriptRenderItems(
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        parts: [{ type: "text", text: "answer" }],
      },
      [subagentMessage("sub-1")],
    );

    expect(items.map((item) => item.kind)).toEqual(["text", "subagent"]);
    expect(items[1]).toMatchObject({ kind: "subagent", message: { id: "sub-1" } });
  });

  test("non_analyze_tool_calls_do_not_consume_subagent_positions", () => {
    const items = buildTranscriptRenderItems(
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        parts: [
          {
            type: "tool_call",
            call: {
              name: "search_hybrid",
              args: { query: "rust" },
              result: "[]",
            },
          },
          {
            type: "tool_call",
            call: {
              name: "analyze_document",
              args: { collection: "notes", path: "a.md" },
            },
          },
        ],
      },
      [subagentMessage("sub-1")],
    );

    expect(items.map((item) => item.kind)).toEqual(["tool_call", "subagent"]);
    expect(items[0]).toMatchObject({
      kind: "tool_call",
      call: { call: { name: "search_hybrid" } },
    });
    expect(items[1]).toMatchObject({ kind: "subagent", message: { id: "sub-1" } });
  });
});
