import type { Message } from "./chat-message-codec";
import type { SubagentMessage } from "./chat-message-groups";

export type TranscriptRenderItem =
  | { kind: "text"; key: string; text: string }
  | { kind: "thinking"; key: string; text: string }
  | {
      kind: "tool_call";
      key: string;
      call: Extract<NonNullable<Message["parts"]>[number], { type: "tool_call" }>;
    }
  | { kind: "subagent"; key: string; message: SubagentMessage };

export function buildTranscriptRenderItems(
  message: Message,
  nestedSubagents: SubagentMessage[] = [],
): TranscriptRenderItem[] {
  if (!message.parts || message.parts.length === 0) {
    return [{ kind: "text", key: "text-0", text: message.content }];
  }

  const items: TranscriptRenderItem[] = [];
  let nextSubagentIndex = 0;

  for (let index = 0; index < message.parts.length; index += 1) {
    const part = message.parts[index];

    if (part.type === "text") {
      items.push({ kind: "text", key: `text-${index}`, text: part.text });
      continue;
    }

    if (part.type === "thinking") {
      items.push({ kind: "thinking", key: `thinking-${index}`, text: part.text });
      continue;
    }

    if (part.call.name === "analyze_document" && nextSubagentIndex < nestedSubagents.length) {
      const subagent = nestedSubagents[nextSubagentIndex++];
      items.push({ kind: "subagent", key: subagent.id, message: subagent });
      continue;
    }

    items.push({ kind: "tool_call", key: `tool-group-${index}`, call: part });
  }

  for (const subagent of nestedSubagents.slice(nextSubagentIndex)) {
    items.push({ kind: "subagent", key: subagent.id, message: subagent });
  }

  return items;
}
