import { describe, expect, test } from "bun:test";
import type { AssistantMessage, AssistantMessageEventStream } from "@mariozechner/pi-ai";

import { consumeAssistantStream } from "./chat-stream";

function assistantMessage(stopReason: AssistantMessage["stopReason"]): AssistantMessage {
  return {
    role: "assistant",
    api: "openai-responses",
    provider: "openai",
    model: "gpt-4.1",
    content: [],
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        total: 0,
      },
    },
    stopReason,
    timestamp: Date.now(),
  };
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

describe("chat-stream", () => {
  test("consumeAssistantStream_preserves_text_and_thinking_order", async () => {
    const seenText: string[] = [];
    const seenThinking: string[] = [];
    const consumed = await consumeAssistantStream(
      makeStream(
        [
          { type: "text_delta", delta: "Hello" },
          { type: "thinking_delta", delta: "Plan" },
          { type: "text_delta", delta: " world" },
          { type: "thinking_delta", delta: " more" },
        ],
        assistantMessage("stop"),
      ),
      {
        onTextDelta: (text) => seenText.push(text),
        onThinkingDelta: (thinking) => seenThinking.push(thinking),
      },
    );

    expect(consumed.text).toBe("Hello world");
    expect(consumed.thinking).toBe("Plan more");
    expect(consumed.interrupted).toBe(false);
    expect(seenText).toEqual(["Hello", "Hello world"]);
    expect(seenThinking).toEqual(["Plan", "Plan more"]);
  });

  test("consumeAssistantStream_marks_aborted_and_error_results_as_interrupted", async () => {
    const aborted = await consumeAssistantStream(makeStream([], assistantMessage("aborted")));
    const failed = await consumeAssistantStream(makeStream([], assistantMessage("error")));

    expect(aborted.interrupted).toBe(true);
    expect(aborted.result.stopReason).toBe("aborted");
    expect(failed.interrupted).toBe(true);
    expect(failed.result.stopReason).toBe("error");
  });
});
