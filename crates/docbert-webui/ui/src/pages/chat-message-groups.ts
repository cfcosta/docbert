import type { ChatActor } from "../lib/api";
import type { Message } from "./chat-message-codec";

export type SubagentMessage = Message & {
  actor: Extract<ChatActor, { type: "subagent" }>;
};

export type DisplayMessageGroup = {
  message: Message;
  nestedSubagents: SubagentMessage[];
};

export function groupMessagesForDisplay(messages: Message[]): DisplayMessageGroup[] {
  const groups: DisplayMessageGroup[] = [];
  let lastParentAssistantGroup: DisplayMessageGroup | null = null;

  for (const message of messages) {
    if (message.actor?.type === "subagent") {
      const subagentMessage = message as SubagentMessage;
      if (lastParentAssistantGroup) {
        lastParentAssistantGroup.nestedSubagents.push(subagentMessage);
        continue;
      }

      groups.push({ message: subagentMessage, nestedSubagents: [] });
      lastParentAssistantGroup = null;
      continue;
    }

    const group: DisplayMessageGroup = { message, nestedSubagents: [] };
    groups.push(group);

    if (message.role === "assistant" && (!message.actor || message.actor.type === "parent")) {
      lastParentAssistantGroup = group;
    } else {
      lastParentAssistantGroup = null;
    }
  }

  return groups;
}
