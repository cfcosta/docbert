import { api } from "../lib/api";
import type { ConversationFull, ConversationSummary } from "../lib/api";
import { apiToMessages, messagesToApi, type Message } from "./chat-message-codec";

export async function loadConversationSummaries(): Promise<ConversationSummary[]> {
  try {
    return await api.listConversations();
  } catch {
    return [];
  }
}

export async function loadConversationById(
  id: string,
): Promise<{ conversation: ConversationFull; messages: Message[] } | null> {
  try {
    const conversation = await api.getConversation(id);
    return {
      conversation,
      messages: apiToMessages(conversation.messages),
    };
  } catch {
    return null;
  }
}

export async function createConversationRecord(
  id: string,
  title: string,
): Promise<ConversationFull | null> {
  try {
    return await api.createConversation(id, title);
  } catch {
    return null;
  }
}

export async function loadLlmSettings() {
  return api.getLlmSettings();
}

export async function deleteConversationRecord(id: string): Promise<boolean> {
  try {
    await api.deleteConversation(id);
    return true;
  } catch {
    return false;
  }
}

export async function persistConversationMessages(
  conversationId: string,
  conversation: ConversationFull | null,
  messages: Message[],
  reloadConversations: () => Promise<void>,
): Promise<void> {
  try {
    const apiMessages = messagesToApi(messages);
    if (conversation) {
      await api.updateConversation(conversationId, {
        ...conversation,
        messages: apiMessages,
      });
    }
    await reloadConversations();
  } catch {
    /* ignore */
  }
}
