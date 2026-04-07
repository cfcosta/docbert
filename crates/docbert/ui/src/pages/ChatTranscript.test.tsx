import "../test/setup";

import { createRef } from "react";
import { beforeEach, describe, expect, test } from "bun:test";
import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router";

import ChatTranscript from "./ChatTranscript";
import type { DisplayMessageGroup } from "./chat-message-groups";
import type { Message } from "./chat-message-codec";

function messageWithToolResult(result: string, name = "search_hybrid"): Message {
  return {
    id: "assistant-1",
    role: "assistant",
    content: "",
    parts: [
      {
        type: "tool_call",
        call: {
          name,
          args: { query: "rust" },
          result,
        },
      },
    ],
  };
}

function displayGroups(message: Message): DisplayMessageGroup[] {
  return [{ message, nestedSubagents: [] }];
}

function renderTranscript(groups: DisplayMessageGroup[]) {
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <Routes>
        <Route
          path="/"
          element={
            <ChatTranscript
              displayMessageGroups={groups}
              loading={false}
              bottomRef={createRef<HTMLDivElement>()}
            />
          }
        />
        <Route path="/documents/:collection/*" element={<div>Document route</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

function subagentMessageWithToolResult(result: string, name = "search_hybrid"): Message {
  return {
    id: "subagent-1",
    role: "assistant",
    actor: {
      type: "subagent",
      id: "subagent-1",
      collection: "notes",
      path: "rust.md",
      status: "done",
    },
    content: "",
    parts: [
      {
        type: "tool_call",
        call: {
          name,
          args: { query: "rust" },
          result,
        },
      },
    ],
  };
}

beforeEach(() => {
  document.body.innerHTML = "";
});

describe("ChatTranscript", () => {
  test("expands parsed search tool results inside the transcript", async () => {
    const user = userEvent.setup();
    const results = JSON.stringify([
      {
        rank: 1,
        score: 0.914,
        doc_id: "#abc123",
        collection: "notes",
        path: "rust.md",
        title: "Rust Guide",
        excerpts: [
          {
            text: "Rust ownership keeps memory safe.\nBorrow checking prevents aliasing bugs.",
            start_line: 10,
            end_line: 11,
          },
        ],
      },
    ]);

    const view = renderTranscript(displayGroups(messageWithToolResult(results)));

    expect(view.queryByRole("link", { name: "Rust Guide" })).toBeNull();

    await user.click(view.getByRole("button", { name: /search_hybrid/i }));

    const documentLink = view.getByRole("link", { name: "Rust Guide" });
    expect(documentLink.getAttribute("href")).toBe("/documents/notes/rust.md");
    expect(view.getByText("notes/rust.md")).toBeTruthy();
  });

  test("clicking an excerpt opens the document page route", async () => {
    const user = userEvent.setup();
    const results = JSON.stringify([
      {
        rank: 1,
        score: 0.914,
        doc_id: "#abc123",
        collection: "notes",
        path: "rust.md",
        title: "Rust Guide",
        excerpts: [
          {
            text: "Rust ownership keeps memory safe.",
            start_line: 10,
            end_line: 10,
          },
        ],
      },
    ]);

    const view = renderTranscript(displayGroups(messageWithToolResult(results)));

    await user.click(view.getByRole("button", { name: /search_hybrid/i }));
    await user.click(view.getByRole("link", { name: /10\s*Rust ownership keeps memory safe\./i }));

    expect(view.getByText("Document route")).toBeTruthy();
  });

  test("renders no results for empty search payloads", async () => {
    const user = userEvent.setup();
    const view = renderTranscript(displayGroups(messageWithToolResult("[]")));

    await user.click(view.getByRole("button", { name: /search_hybrid/i }));

    expect(view.getByText("No results")).toBeTruthy();
  });

  test("falls back to preformatted output for non-search tools", async () => {
    const user = userEvent.setup();
    const view = renderTranscript(
      displayGroups(messageWithToolResult("plain output", "analyze_document")),
    );

    await user.click(view.getByRole("button", { name: /analyze_document/i }));

    expect(view.getByText("plain output").tagName).toBe("PRE");
  });

  test("uses a distinct tool-call style for subagent file analysis tools", async () => {
    const user = userEvent.setup();
    const view = renderTranscript([
      {
        message: messageWithToolResult("root output", "analyze_document"),
        nestedSubagents: [],
      },
      {
        message: subagentMessageWithToolResult("inner output", "analyze_document"),
        nestedSubagents: [],
      },
    ]);

    await user.click(view.getByRole("button", { name: /analyze_document/i }));
    await user.click(view.getByRole("button", { name: /file analysis/i }));
    const toolButtons = view.getAllByRole("button", { name: /analyze_document/i });
    await user.click(toolButtons[1]);

    const rootTool = view.getByText("root output").closest(".chat-tool-call");
    const subagentTool = view.getByText("inner output").closest(".chat-tool-call");

    expect(rootTool?.className).toContain("chat-tool-call-root");
    expect(rootTool?.className).not.toContain("chat-tool-call-subagent");
    expect(subagentTool?.className).toContain("chat-tool-call-subagent");
  });
});
