import { describe, expect, test } from "bun:test";

import type { ConversationFull } from "../lib/api";
import { apiToMessages, contentFromParts, messagesToApi, type Message } from "./chat-message-codec";

describe("chat-message-codec", () => {
  test("content_from_parts_joins_only_text_parts", () => {
    expect(
      contentFromParts([
        { type: "thinking", text: "plan" },
        { type: "text", text: "A" },
        {
          type: "tool_call",
          call: {
            name: "search_hybrid",
            args: { query: "rust" },
          },
        },
        { type: "text", text: "B" },
      ]),
    ).toBe("AB");
  });

  test("messagesToApi_sends_parts_and_defaults_actor_to_parent", () => {
    const messages: Message[] = [
      {
        id: "assistant-with-parts",
        role: "assistant",
        content: "answer from parts",
        parts: [
          { type: "thinking", text: "plan" },
          { type: "text", text: "answer from parts" },
        ],
      },
    ];

    const apiMessages = messagesToApi(messages);

    expect(apiMessages[0].parts).toEqual([
      { type: "thinking", text: "plan" },
      { type: "text", text: "answer from parts" },
    ]);
    expect(apiMessages[0].actor).toEqual({ type: "parent" });
    expect(apiMessages[0]).not.toHaveProperty("content");
  });

  test("apiToMessages_consumes_normalized_message_parts", () => {
    const normalizedMessages: ConversationFull["messages"] = [
      {
        id: "assistant-normalized",
        role: "assistant",
        parts: [
          { type: "thinking", text: "Planning" },
          { type: "text", text: "Answer" },
          {
            type: "tool_call",
            name: "search_hybrid",
            args: { query: "rust" },
            result: "[]",
            is_error: false,
          },
        ],
        actor: { type: "parent" },
      },
    ];

    const messages = apiToMessages(normalizedMessages);

    expect(messages).toHaveLength(1);
    expect(messages[0].content).toBe("Answer");
    expect(messages[0].parts).toEqual([
      { type: "thinking", text: "Planning" },
      { type: "text", text: "Answer" },
      {
        type: "tool_call",
        call: {
          name: "search_hybrid",
          args: { query: "rust" },
          result: "[]",
          isError: false,
        },
      },
    ]);
  });

  test("message sources roundtrip without synthetic search fields", () => {
    const messages: Message[] = [
      {
        id: "assistant-with-sources",
        role: "assistant",
        content: "Answer",
        parts: [{ type: "text", text: "Answer" }],
        sources: [
          {
            collection: "notes",
            path: "rust.md",
            title: "Rust",
          },
        ],
      },
    ];

    const apiMessages = messagesToApi(messages);
    expect(apiMessages[0].sources).toEqual([
      {
        collection: "notes",
        path: "rust.md",
        title: "Rust",
      },
    ]);

    const roundTripped = apiToMessages(apiMessages);
    expect(roundTripped[0].sources).toEqual([
      {
        collection: "notes",
        path: "rust.md",
        title: "Rust",
      },
    ]);
  });
});
