import type {
  Context,
  Message as PiMessage,
  Tool,
  ToolResultMessage,
  UserMessage,
} from "@mariozechner/pi-ai";

import { contentFromParts, type ContentPart, type Message } from "./chat-message-codec";

export function messageTextContent(message: Message): string {
  return contentFromParts(message.parts ?? []) || message.content;
}

export function createToolCallId(messageId: string, index: number): string {
  return `${messageId}:tool:${index}`;
}

export function createToolResultMessage(
  callId: string,
  toolName: string,
  text: string,
  isError: boolean,
): ToolResultMessage {
  return {
    role: "toolResult",
    toolCallId: callId,
    toolName,
    content: [{ type: "text", text }],
    isError,
    timestamp: Date.now(),
  };
}

export function createPiContext(history: Message[], systemPrompt: string, tools: Tool[]): Context {
  const piMessages: PiMessage[] = [];

  for (const message of history) {
    if (message.actor?.type === "subagent") {
      continue;
    }

    if (message.role === "user") {
      const text = messageTextContent(message).trim();
      if (!text) {
        continue;
      }

      const userPiMsg: UserMessage = {
        role: "user",
        content: text,
        timestamp: Date.now(),
      };
      piMessages.push(userPiMsg as PiMessage);
      continue;
    }

    const parts =
      message.parts && message.parts.length > 0
        ? message.parts
        : message.content
          ? [{ type: "text", text: message.content } satisfies ContentPart]
          : [];

    let assistantContent: Array<
      | { type: "text"; text: string }
      | { type: "thinking"; thinking: string }
      | { type: "toolCall"; id: string; name: string; arguments: Record<string, unknown> }
    > = [];
    let pendingToolResults: ToolResultMessage[] = [];
    let toolCallIndex = 0;

    const flushAssistantTurn = () => {
      if (assistantContent.length > 0) {
        piMessages.push({
          role: "assistant",
          content: assistantContent,
          timestamp: Date.now(),
        } as PiMessage);
        assistantContent = [];
      }

      if (pendingToolResults.length > 0) {
        piMessages.push(...pendingToolResults.map((toolResult) => toolResult as PiMessage));
        pendingToolResults = [];
      }
    };

    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      if (part.type === "text") {
        assistantContent.push({ type: "text", text: part.text });
        continue;
      }

      if (part.type === "thinking") {
        assistantContent.push({ type: "thinking", thinking: part.text });
        continue;
      }

      const toolCallId = createToolCallId(message.id, toolCallIndex++);
      assistantContent.push({
        type: "toolCall",
        id: toolCallId,
        name: part.call.name,
        arguments: part.call.args,
      });

      if (part.call.result) {
        pendingToolResults.push(
          createToolResultMessage(
            toolCallId,
            part.call.name,
            part.call.result,
            Boolean(part.call.isError),
          ),
        );
      }

      const nextPart = parts[i + 1];
      if (!nextPart || nextPart.type !== "tool_call") {
        flushAssistantTurn();
      }
    }

    flushAssistantTurn();
  }

  return {
    systemPrompt,
    messages: piMessages,
    tools,
  };
}

export function applyInterruptedStopReason(
  message: Message,
  stopReason: "aborted" | "error",
  errorMessage?: string,
): Message {
  const note =
    stopReason === "aborted"
      ? "Response interrupted before completion."
      : `Response interrupted due to an error${errorMessage ? `: ${errorMessage}` : "."}`;

  if ((message.content || "").includes(note)) {
    return message;
  }

  const parts = [...(message.parts ?? [])];
  const last = parts[parts.length - 1];
  if (last && last.type === "text") {
    parts[parts.length - 1] = {
      type: "text",
      text: `${last.text}${last.text ? "\n\n" : ""}${note}`,
    };
  } else {
    parts.push({ type: "text", text: note });
  }

  return {
    ...message,
    content: `${message.content}${message.content ? "\n\n" : ""}${note}`,
    parts,
  };
}
