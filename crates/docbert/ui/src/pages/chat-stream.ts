import type { AssistantMessage, AssistantMessageEventStream } from "@mariozechner/pi-ai";

export interface AssistantStreamCallbacks {
  onTextDelta?: (text: string, delta: string) => void;
  onThinkingDelta?: (thinking: string, delta: string) => void;
  onError?: (error: unknown) => void;
}

export interface ConsumedAssistantStream {
  text: string;
  thinking: string;
  result: AssistantMessage;
  interrupted: boolean;
  lastError?: unknown;
}

export function isInterruptedAssistantResult(
  result: AssistantMessage,
): result is AssistantMessage & { stopReason: "aborted" | "error" } {
  return result.stopReason === "aborted" || result.stopReason === "error";
}

export function assistantToolCalls(result: AssistantMessage) {
  return result.content.filter(
    (block): block is Extract<AssistantMessage["content"][number], { type: "toolCall" }> =>
      block.type === "toolCall",
  );
}

export function shouldContinueAssistantToolRound(result: AssistantMessage): boolean {
  return !isInterruptedAssistantResult(result) && assistantToolCalls(result).length > 0;
}

export async function consumeAssistantStream(
  stream: AssistantMessageEventStream,
  callbacks: AssistantStreamCallbacks = {},
): Promise<ConsumedAssistantStream> {
  let text = "";
  let thinking = "";
  let lastError: unknown;

  for await (const event of stream) {
    if (event.type === "text_delta") {
      text += event.delta;
      callbacks.onTextDelta?.(text, event.delta);
    }

    if (event.type === "thinking_delta") {
      thinking += event.delta;
      callbacks.onThinkingDelta?.(thinking, event.delta);
    }

    if (event.type === "error") {
      lastError = event.error;
      callbacks.onError?.(event.error);
    }
  }

  const result = await stream.result();

  return {
    text,
    thinking,
    result,
    interrupted: isInterruptedAssistantResult(result),
    lastError,
  };
}
