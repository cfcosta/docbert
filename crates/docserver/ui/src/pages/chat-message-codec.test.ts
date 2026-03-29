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

  test("messagesToApi_uses_parts_text_or_content_fallback", () => {
    const messages: Message[] = [
      {
        id: "assistant-with-parts",
        role: "assistant",
        content: "legacy fallback",
        parts: [
          { type: "thinking", text: "plan" },
          { type: "text", text: "answer from parts" },
        ],
      },
      {
        id: "user-without-parts",
        role: "user",
        content: "content only",
      },
    ];

    const apiMessages = messagesToApi(messages);

    expect(apiMessages[0].content).toBe("answer from parts");
    expect(apiMessages[1].content).toBe("content only");
    expect(apiMessages[1].actor).toEqual({ type: "parent" });
  });

  test("apiToMessages_rehydrates_legacy_messages_without_parts", () => {
    const legacyMessages: ConversationFull["messages"] = [
      {
        id: "assistant-legacy",
        role: "assistant",
        content: "",
        content_parts: [
          { type: "thinking", text: "Planning" },
          { type: "text", text: "Answer" },
        ],
        tool_calls: [
          {
            name: "search_hybrid",
            args: { query: "rust" },
            result: "[]",
            is_error: false,
          },
        ],
        actor: { type: "parent" },
      },
    ];

    const messages = apiToMessages(legacyMessages);

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
