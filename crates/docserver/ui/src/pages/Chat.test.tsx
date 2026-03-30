import "../test/setup";

import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router";

import { api, type ConversationFull, type ConversationSummary, type LlmSettings } from "../lib/api";

const toolType = {
  Object: <T,>(shape: T) => shape,
  String: <T,>(options?: T) => options ?? {},
  Number: <T,>(options?: T) => options ?? {},
  Optional: <T,>(value: T) => value,
};

let streamCallCount = 0;
let getModelCallCount = 0;
let getLlmSettingsCallCount = 0;
let nextStream: (() => unknown) | null = null;

mock.module("@mariozechner/pi-ai", () => ({
  Type: toolType,
  getModel: (provider: unknown, model: unknown) => {
    getModelCallCount += 1;
    return { provider, model, reasoning: false };
  },
  streamSimple: () => {
    streamCallCount += 1;
    if (!nextStream) {
      throw new Error("streamSimple called without a configured test stream");
    }
    return nextStream();
  },
}));

const { default: Chat } = await import("./Chat");

const originalApi = { ...api };
const NO_SETTINGS: LlmSettings = { provider: null, model: null, api_key: null };

type ApiTrackers = {
  getConversationIds: string[];
  updatedConversations: ConversationFull[];
};

function assistantMessage(
  stopReason: "stop" | "aborted",
  content: Array<Record<string, unknown>> = [],
): Record<string, unknown> {
  return {
    role: "assistant",
    api: "openai-responses",
    provider: "openai",
    model: "gpt-4.1",
    content,
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

function makeStream(events: Array<Record<string, unknown>>, result: Record<string, unknown>) {
  return {
    async *[Symbol.asyncIterator]() {
      for (const event of events) {
        yield event;
      }
    },
    result: async () => result,
  };
}

function summaryFromConversation(conversation: ConversationFull): ConversationSummary {
  return {
    id: conversation.id,
    title: conversation.title,
    created_at: conversation.created_at,
    updated_at: conversation.updated_at,
    message_count: conversation.messages.length,
  };
}

function makeConversation(
  id: string,
  messages: ConversationFull["messages"],
  title = "Loaded conversation",
): ConversationFull {
  return {
    id,
    title,
    created_at: 1,
    updated_at: 2,
    messages,
  };
}

function installApiStubs({
  conversation,
  settings = NO_SETTINGS,
}: {
  conversation: ConversationFull;
  settings?: LlmSettings;
}): ApiTrackers {
  const trackers: ApiTrackers = {
    getConversationIds: [],
    updatedConversations: [],
  };

  api.listConversations = async () => [summaryFromConversation(conversation)];
  api.getConversation = async (id: string) => {
    trackers.getConversationIds.push(id);
    return { ...conversation, messages: [...conversation.messages] };
  };
  api.getLlmSettings = async () => {
    getLlmSettingsCallCount += 1;
    return settings;
  };
  api.updateConversation = async (_id: string, nextConversation: ConversationFull) => {
    trackers.updatedConversations.push(nextConversation);
    return nextConversation;
  };
  api.createConversation = async () => {
    throw new Error("createConversation should not be called in these tests");
  };
  api.deleteConversation = async () => {};
  api.listCollections = async () => [];
  api.listDocuments = async () => [];
  api.createCollection = async () => ({ name: "notes" });
  api.deleteCollection = async () => {};
  api.ingestDocuments = async () => ({ ingested: 0, documents: [] });
  api.getDocument = async () => ({
    doc_id: "#abc123",
    collection: "notes",
    path: "hello.md",
    title: "Hello",
    content: "# Hello",
  });
  api.deleteDocument = async () => {};
  api.search = async () => ({
    query: "",
    mode: "semantic",
    result_count: 0,
    results: [],
  });
  api.updateLlmSettings = async (nextSettings) => nextSettings;

  return trackers;
}

function renderChat(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <Routes>
        <Route path="/chat" element={<Chat />} />
        <Route path="/chat/:conversationId" element={<Chat />} />
      </Routes>
    </MemoryRouter>,
  );
}

function chatInput(container: HTMLElement): HTMLInputElement {
  const input = container.getElementsByTagName("input")[0];
  if (!(input instanceof HTMLInputElement)) {
    throw new Error("chat input not found");
  }
  return input;
}

function sendButton(container: HTMLElement): HTMLButtonElement {
  const buttons = Array.from(container.getElementsByTagName("button"));
  const button = buttons.find((candidate) => candidate.className.includes("chat-send"));
  if (!(button instanceof HTMLButtonElement)) {
    throw new Error("chat send button not found");
  }
  return button;
}

async function waitForCondition(
  condition: () => boolean,
  message: () => string,
  timeoutMs = 1000,
) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (condition()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }

  throw new Error(message());
}

beforeEach(() => {
  streamCallCount = 0;
  getModelCallCount = 0;
  getLlmSettingsCallCount = 0;
  nextStream = null;
});

afterEach(() => {
  Object.assign(api, originalApi);
});

describe("Chat page", () => {
  test("loads_route_conversation_and_renders_existing_messages", async () => {
    const conversation = makeConversation("loaded-1", [
      {
        id: "assistant-1",
        role: "assistant",
        content: "Loaded answer",
        actor: { type: "parent" },
        parts: [{ type: "text", text: "Loaded answer" }],
      },
    ]);
    const trackers = installApiStubs({ conversation });

    const view = renderChat("/chat/loaded-1");

    await waitForCondition(
      () => view.container.textContent?.includes("Loaded answer") ?? false,
      () => `loaded conversation never rendered: ${JSON.stringify(view.container.textContent)}`,
    );
    expect(trackers.getConversationIds).toEqual(["loaded-1"]);
    expect(view.container.textContent).toContain("Loaded conversation");
  });

  test("missing_llm_settings_appends_configuration_message_without_starting_stream", async () => {
    const conversation = makeConversation("missing-config", [
      {
        id: "user-1",
        role: "user",
        content: "Earlier question",
        parts: [{ type: "text", text: "Earlier question" }],
      },
    ]);
    installApiStubs({ conversation, settings: NO_SETTINGS });

    const user = userEvent.setup({ pointerEventsCheck: 0 });
    const view = renderChat("/chat/missing-config");

    await waitForCondition(
      () => view.container.textContent?.includes("Earlier question") ?? false,
      () => `existing question never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    await user.type(chatInput(view.container), "What is Rust?");
    await waitForCondition(
      () => sendButton(view.container).disabled === false,
      () => `send button stayed disabled after typing: ${JSON.stringify(view.container.textContent)}`,
    );
    await user.click(sendButton(view.container));
    await waitForCondition(
      () => getLlmSettingsCallCount > 0,
      () => `submit never reached settings lookup; text=${JSON.stringify(view.container.textContent)}`,
      300,
    );

    await waitForCondition(
      () =>
        view.container.textContent?.includes(
          "No LLM provider configured. Go to Settings to select a provider, model, and API key.",
        ) ?? false,
      () => `missing config message not rendered; text=${JSON.stringify(view.container.textContent)}`,
    );
    expect(streamCallCount).toBe(0);
    expect(getModelCallCount).toBe(0);
  });

  test("send_message_persists_latest_messages_after_successful_round", async () => {
    const conversation = makeConversation("success-1", [
      {
        id: "assistant-1",
        role: "assistant",
        content: "Loaded answer",
        actor: { type: "parent" },
        parts: [{ type: "text", text: "Loaded answer" }],
      },
    ]);
    const trackers = installApiStubs({
      conversation,
      settings: {
        provider: "openai",
        model: "gpt-4.1",
        api_key: "test-key",
      },
    });
    nextStream = () =>
      makeStream(
        [
          { type: "text_delta", delta: "Assistant" },
          { type: "text_delta", delta: " reply" },
        ],
        assistantMessage("stop", [{ type: "text", text: "Assistant reply" }]),
      );

    const user = userEvent.setup({ pointerEventsCheck: 0 });
    const view = renderChat("/chat/success-1");

    await waitForCondition(
      () => view.container.textContent?.includes("Loaded answer") ?? false,
      () => `loaded answer never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    await user.type(chatInput(view.container), "Summarize the notes");
    await waitForCondition(
      () => sendButton(view.container).disabled === false,
      () => `send button stayed disabled after typing: ${JSON.stringify(view.container.textContent)}`,
    );
    await user.click(sendButton(view.container));
    await waitForCondition(
      () => view.container.textContent?.includes("Summarize the notes") ?? false,
      () => `user message never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    await waitForCondition(
      () => view.container.textContent?.includes("Assistant reply") ?? false,
      () => `assistant reply never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    await waitForCondition(
      () => trackers.updatedConversations.length > 0,
      () => `conversation was never persisted; text=${JSON.stringify(view.container.textContent)}`,
    );

    const persisted = trackers.updatedConversations.at(-1)!;
    expect(persisted.messages.map((message) => message.content)).toEqual([
      "Loaded answer",
      "Summarize the notes",
      "Assistant reply",
    ]);
    expect(streamCallCount).toBe(1);
    expect(getModelCallCount).toBe(1);
  });

  test("interrupted_stream_appends_stop_note_once_and_still_persists", async () => {
    const conversation = makeConversation("aborted-1", [
      {
        id: "assistant-1",
        role: "assistant",
        content: "Loaded answer",
        actor: { type: "parent" },
        parts: [{ type: "text", text: "Loaded answer" }],
      },
    ]);
    const trackers = installApiStubs({
      conversation,
      settings: {
        provider: "openai",
        model: "gpt-4.1",
        api_key: "test-key",
      },
    });
    nextStream = () => makeStream([], assistantMessage("aborted"));

    const user = userEvent.setup({ pointerEventsCheck: 0 });
    const view = renderChat("/chat/aborted-1");

    await waitForCondition(
      () => view.container.textContent?.includes("Loaded answer") ?? false,
      () => `loaded answer never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    await user.type(chatInput(view.container), "What changed?");
    await waitForCondition(
      () => sendButton(view.container).disabled === false,
      () => `send button stayed disabled after typing: ${JSON.stringify(view.container.textContent)}`,
    );
    await user.click(sendButton(view.container));
    await waitForCondition(
      () => view.container.textContent?.includes("What changed?") ?? false,
      () => `user message never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    const interruptionNote = "Response interrupted before completion.";
    await waitForCondition(
      () => view.container.textContent?.includes(interruptionNote) ?? false,
      () => `interruption note never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    await waitForCondition(
      () => trackers.updatedConversations.length > 0,
      () => `conversation was never persisted; text=${JSON.stringify(view.container.textContent)}`,
    );

    const persisted = trackers.updatedConversations.at(-1)!;
    const assistantMessages = persisted.messages.filter((message) => message.role === "assistant");
    const finalAssistant = assistantMessages.at(-1)!;
    expect(finalAssistant.content).toBe(interruptionNote);
    expect(finalAssistant.content.split(interruptionNote)).toHaveLength(2);
    expect(streamCallCount).toBe(1);
  });
});
