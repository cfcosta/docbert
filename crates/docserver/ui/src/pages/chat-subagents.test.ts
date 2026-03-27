import { describe, expect, test } from "bun:test";

import type { SearchResult } from "../lib/api";
import {
  buildSynthesisPayload,
  decideAnalyzeFiles,
  formatAnalyzeFilesAcknowledgement,
  insertOrUpdateSubagentMessage,
  mergeCurrentTurnSearchResults,
  queueAcceptedSubagentMessages,
  setSubagentStatus,
  updateSubagentMessageById,
  upsertSubagentPart,
} from "./chat-subagents";

function result(collection: string, path: string, title: string): SearchResult {
  return {
    rank: 1,
    score: 1,
    doc_id: `${collection}:${path}`,
    collection,
    path,
    title,
  };
}

describe("mergeCurrentTurnSearchResults", () => {
  test("deduplicates by collection and path", () => {
    const merged = mergeCurrentTurnSearchResults(
      [result("notes", "a.md", "A")],
      [result("notes", "a.md", "A2"), result("notes", "b.md", "B")],
    );

    expect(merged).toHaveLength(2);
    expect(merged.map((entry) => `${entry.collection}:${entry.path}`)).toEqual([
      "notes:a.md",
      "notes:b.md",
    ]);
  });
});

describe("decideAnalyzeFiles", () => {
  const available = [
    result("notes", "a.md", "A"),
    result("notes", "b.md", "B"),
    result("notes", "c.md", "C"),
    result("notes", "d.md", "D"),
  ];

  test("rejects duplicate files", () => {
    const decision = decideAnalyzeFiles(
      {
        files: [
          { collection: "notes", path: "a.md", reason: "first" },
          { collection: "notes", path: "a.md", reason: "duplicate" },
        ],
      },
      available,
    );

    expect(decision.accepted).toHaveLength(1);
    expect(decision.rejected).toEqual([
      {
        collection: "notes",
        path: "a.md",
        reason: "duplicate",
        rejection_reason: "duplicate_file",
      },
    ]);
  });

  test("caps accepted files at three", () => {
    const decision = decideAnalyzeFiles(
      {
        files: [
          { collection: "notes", path: "a.md", reason: "one" },
          { collection: "notes", path: "b.md", reason: "two" },
          { collection: "notes", path: "c.md", reason: "three" },
          { collection: "notes", path: "d.md", reason: "four" },
        ],
      },
      available,
    );

    expect(decision.accepted).toHaveLength(3);
    expect(decision.capped).toBe(true);
    expect(decision.rejected).toEqual([
      {
        collection: "notes",
        path: "d.md",
        reason: "four",
        rejection_reason: "max_files_exceeded",
      },
    ]);
  });

  test("rejects files missing from current-turn search results", () => {
    const decision = decideAnalyzeFiles(
      {
        files: [{ collection: "notes", path: "missing.md", reason: "check this" }],
      },
      available,
    );

    expect(decision.accepted).toHaveLength(0);
    expect(decision.rejected).toEqual([
      {
        collection: "notes",
        path: "missing.md",
        reason: "check this",
        rejection_reason: "not_in_current_turn_search_results",
      },
    ]);
  });

  test("acknowledges valid payloads", () => {
    const decision = decideAnalyzeFiles(
      {
        files: [
          { collection: "notes", path: "a.md", reason: "important" },
          { collection: "notes", path: "b.md", reason: "also relevant" },
        ],
      },
      available,
    );

    expect(decision.accepted).toEqual([
      {
        collection: "notes",
        path: "a.md",
        reason: "important",
        title: "A",
      },
      {
        collection: "notes",
        path: "b.md",
        reason: "also relevant",
        title: "B",
      },
    ]);
    expect(JSON.parse(formatAnalyzeFilesAcknowledgement(decision))).toEqual({
      accepted: decision.accepted,
      rejected: [],
      capped: false,
    });
  });
});

describe("insertOrUpdateSubagentMessage", () => {
  test("creates a missing message", () => {
    const messages = insertOrUpdateSubagentMessage([], {
      id: "sub-1",
      role: "assistant",
      content: "Queued for analysis.",
      actor: {
        type: "subagent",
        id: "sub-1",
        collection: "notes",
        path: "a.md",
        status: "queued",
      },
    });

    expect(messages).toHaveLength(1);
    expect(messages[0].id).toBe("sub-1");
  });

  test("updates only the targeted message id", () => {
    const messages = insertOrUpdateSubagentMessage(
      [
        { id: "user-1", role: "user", content: "Question" },
        {
          id: "sub-1",
          role: "assistant",
          content: "Queued for analysis.",
          actor: {
            type: "subagent",
            id: "sub-1",
            collection: "notes",
            path: "a.md",
            status: "queued",
          },
        },
      ],
      {
        id: "sub-1",
        role: "assistant",
        content: "Still queued.",
        actor: {
          type: "subagent",
          id: "sub-1",
          collection: "notes",
          path: "a.md",
          status: "queued",
        },
      },
    );

    expect(messages[0]).toEqual({ id: "user-1", role: "user", content: "Question" });
    expect(messages[1].content).toBe("Still queued.");
  });

  test("preserves surrounding message order", () => {
    const messages = insertOrUpdateSubagentMessage(
      [
        { id: "user-1", role: "user", content: "Question" },
        { id: "assistant-1", role: "assistant", content: "Answer" },
      ],
      {
        id: "sub-1",
        role: "assistant",
        content: "Queued for analysis.",
        actor: {
          type: "subagent",
          id: "sub-1",
          collection: "notes",
          path: "a.md",
          status: "queued",
        },
      },
    );

    expect(messages.map((message) => message.id)).toEqual(["user-1", "assistant-1", "sub-1"]);
  });
});

describe("queueAcceptedSubagentMessages", () => {
  test("queues one message per accepted file in order", () => {
    let nextId = 0;
    const queued = queueAcceptedSubagentMessages({
      messages: [{ id: "assistant-1", role: "assistant", content: "Ack" }],
      acceptedFiles: [
        { collection: "notes", path: "a.md", reason: "first", title: "A" },
        { collection: "notes", path: "b.md", reason: "second", title: "B" },
      ],
      queuedFiles: [],
      createMessageId: () => `sub-${++nextId}`,
      createMessage: (messageId, file) => ({
        id: messageId,
        role: "assistant",
        content: `Queued ${file.path}`,
        actor: {
          type: "subagent",
          id: messageId,
          collection: file.collection,
          path: file.path,
          status: "queued",
        },
      }),
    });

    expect(queued.messages.map((message) => message.id)).toEqual(["assistant-1", "sub-1", "sub-2"]);
    expect(queued.queuedFiles.map((file) => `${file.messageId}:${file.path}`)).toEqual([
      "sub-1:a.md",
      "sub-2:b.md",
    ]);
  });
});

describe("updateSubagentMessageById", () => {
  test("updates only the targeted message", () => {
    const messages = updateSubagentMessageById(
      [
        { id: "sub-1", role: "assistant", content: "one" },
        { id: "sub-2", role: "assistant", content: "two" },
      ],
      "sub-2",
      (message) => ({ ...message, content: "updated" }),
    );

    expect(messages).toEqual([
      { id: "sub-1", role: "assistant", content: "one" },
      { id: "sub-2", role: "assistant", content: "updated" },
    ]);
  });

  test("preserves stable order while updating by id", () => {
    const messages = updateSubagentMessageById(
      [
        { id: "assistant-1", role: "assistant", content: "Ack" },
        { id: "sub-1", role: "assistant", content: "first" },
        { id: "sub-2", role: "assistant", content: "second" },
      ],
      "sub-1",
      (message) => ({ ...message, content: "still first" }),
    );

    expect(messages.map((message) => message.id)).toEqual(["assistant-1", "sub-1", "sub-2"]);
    expect(messages[1].content).toBe("still first");
  });
});

describe("buildSynthesisPayload", () => {
  test("includes failed subagent markers", () => {
    const payload = buildSynthesisPayload({
      userQuestion: "What matters?",
      acceptedFiles: [
        { collection: "notes", path: "a.md", reason: "first", title: "A" },
        { collection: "notes", path: "b.md", reason: "second", title: "B" },
      ],
      subagentResults: [
        { collection: "notes", path: "a.md", reason: "first", title: "A", text: "Useful" },
        { collection: "notes", path: "b.md", reason: "second", title: "B", error: "timeout" },
      ],
    });

    expect(payload.files).toEqual([
      {
        collection: "notes",
        path: "a.md",
        reason: "first",
        title: "A",
        analysis: "Useful",
      },
      {
        collection: "notes",
        path: "b.md",
        reason: "second",
        title: "B",
        error: "timeout",
      },
    ]);
  });

  test("only successful files contribute final sources", () => {
    const payload = buildSynthesisPayload({
      userQuestion: "What matters?",
      acceptedFiles: [
        { collection: "notes", path: "a.md", reason: "first", title: "A" },
        { collection: "notes", path: "b.md", reason: "second", title: "B" },
      ],
      subagentResults: [
        { collection: "notes", path: "a.md", reason: "first", title: "A", text: "Useful" },
        { collection: "notes", path: "b.md", reason: "second", title: "B", error: "timeout" },
      ],
    });

    expect(payload.sourceFiles).toEqual([{ collection: "notes", path: "a.md", title: "A" }]);
  });

  test("preserves accepted-file order in the payload", () => {
    const payload = buildSynthesisPayload({
      userQuestion: "What matters?",
      acceptedFiles: [
        { collection: "notes", path: "b.md", reason: "second", title: "B" },
        { collection: "notes", path: "a.md", reason: "first", title: "A" },
      ],
      subagentResults: [
        { collection: "notes", path: "a.md", reason: "first", title: "A", text: "Useful A" },
        { collection: "notes", path: "b.md", reason: "second", title: "B", text: "Useful B" },
      ],
    });

    expect(payload.files.map((file) => file.path)).toEqual(["b.md", "a.md"]);
  });
});

describe("setSubagentStatus", () => {
  test("updates queued to running", () => {
    const updated = setSubagentStatus(
      {
        id: "sub-1",
        role: "assistant",
        content: "Queued",
        actor: {
          type: "subagent",
          id: "sub-1",
          collection: "notes",
          path: "a.md",
          status: "queued",
        },
      },
      "running",
    );

    expect(updated.actor?.type).toBe("subagent");
    if (updated.actor?.type !== "subagent") throw new Error("expected subagent actor");
    expect(updated.actor.status).toBe("running");
  });

  test("updates running to done", () => {
    const updated = setSubagentStatus(
      {
        id: "sub-1",
        role: "assistant",
        content: "Running",
        actor: {
          type: "subagent",
          id: "sub-1",
          collection: "notes",
          path: "a.md",
          status: "running",
        },
      },
      "done",
    );

    if (updated.actor?.type !== "subagent") throw new Error("expected subagent actor");
    expect(updated.actor.status).toBe("done");
  });

  test("updates running to error", () => {
    const updated = setSubagentStatus(
      {
        id: "sub-1",
        role: "assistant",
        content: "Running",
        actor: {
          type: "subagent",
          id: "sub-1",
          collection: "notes",
          path: "a.md",
          status: "running",
        },
      },
      "error",
    );

    if (updated.actor?.type !== "subagent") throw new Error("expected subagent actor");
    expect(updated.actor.status).toBe("error");
  });
});

describe("upsertSubagentPart", () => {
  test("appends text and thinking parts while keeping text-derived content", () => {
    const withText = upsertSubagentPart(
      {
        id: "sub-1",
        role: "assistant",
        content: "",
        parts: [],
        actor: {
          type: "subagent",
          id: "sub-1",
          collection: "notes",
          path: "a.md",
          status: "running",
        },
      },
      { type: "text", text: "Hello" },
    );
    const withThinking = upsertSubagentPart(withText, {
      type: "thinking",
      text: "Considering",
    });

    expect(withThinking.content).toBe("Hello");
    expect(withThinking.parts).toEqual([
      { type: "text", text: "Hello" },
      { type: "thinking", text: "Considering" },
    ]);
  });
});
