import type {
  ChatActor,
  ChatPart as ApiChatPart,
  ConversationFull,
  SearchResult,
} from "../lib/api";

export interface ToolCallInfo {
  name: string;
  args: Record<string, unknown>;
  result?: string;
  isError?: boolean;
}

export type ContentPart =
  | { type: "text"; text: string }
  | { type: "thinking"; text: string }
  | { type: "tool_call"; call: ToolCallInfo };

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  parts?: ContentPart[];
  actor?: ChatActor;
  sources?: SearchResult[];
}

export function toApiPart(part: ContentPart): ApiChatPart {
  switch (part.type) {
    case "text":
      return { type: "text", text: part.text };
    case "thinking":
      return { type: "thinking", text: part.text };
    case "tool_call":
      return {
        type: "tool_call",
        name: part.call.name,
        args: part.call.args,
        result: part.call.result,
        is_error: part.call.isError,
      };
  }
}

export function fromApiPart(part: ApiChatPart): ContentPart {
  switch (part.type) {
    case "text":
      return { type: "text", text: part.text };
    case "thinking":
      return { type: "thinking", text: part.text };
    case "tool_call":
      return {
        type: "tool_call",
        call: {
          name: part.name,
          args: part.args,
          result: part.result,
          isError: part.is_error,
        },
      };
  }
}

export function contentFromParts(parts: ContentPart[]): string {
  return parts
    .filter((part): part is Extract<ContentPart, { type: "text" }> => part.type === "text")
    .map((part) => part.text)
    .join("");
}

export function legacyParts(message: ConversationFull["messages"][number]): ContentPart[] {
  const parts: ContentPart[] = [];
  if (message.content_parts && message.content_parts.length > 0) {
    for (const part of message.content_parts) {
      parts.push({ type: part.type, text: part.text });
    }
  } else if (message.content) {
    parts.push({ type: "text", text: message.content });
  }

  if (message.tool_calls && message.tool_calls.length > 0) {
    for (const toolCall of message.tool_calls) {
      parts.push({
        type: "tool_call",
        call: {
          name: toolCall.name,
          args: toolCall.args,
          result: toolCall.result,
          isError: toolCall.is_error,
        },
      });
    }
  }

  return parts;
}

export function messagesToApi(messages: Message[]): ConversationFull["messages"] {
  return messages.map((message) => {
    const parts = (message.parts ?? []).map(toApiPart);
    return {
      id: message.id,
      role: message.role,
      actor: message.actor ?? { type: "parent" },
      parts,
      content: contentFromParts(message.parts ?? []) || message.content,
      sources: message.sources?.map((source) => ({
        collection: source.collection,
        path: source.path,
        title: source.title,
      })),
    };
  });
}

export function apiToMessages(msgs: ConversationFull["messages"]): Message[] {
  return msgs.map((message) => {
    const parts =
      (message.parts && message.parts.length > 0
        ? message.parts.map(fromApiPart)
        : legacyParts(message)) || [];

    return {
      id: message.id,
      role: message.role,
      content: contentFromParts(parts),
      parts: parts.length > 0 ? parts : undefined,
      actor: message.actor,
      sources: message.sources?.map((source) => ({
        rank: 0,
        score: 0,
        doc_id: "",
        collection: source.collection,
        path: source.path,
        title: source.title,
      })),
    };
  });
}
