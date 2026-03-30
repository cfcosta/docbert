import "../test/setup";

import { createRef } from "react";
import { describe, expect, test } from "bun:test";
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

describe("ChatTranscript", () => {
  test("renders document nodes with excerpt children for search tool results", async () => {
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
          {
            text: "Traits enable polymorphism.",
            start_line: 20,
            end_line: 20,
          },
        ],
      },
    ]);

    const view = renderTranscript(displayGroups(messageWithToolResult(results)));

    await user.click(view.getByRole("button", { name: /search_hybrid/i }));

    const documentLink = view.getByRole("link", { name: "Rust Guide" });
    expect(documentLink.getAttribute("href")).toBe("/documents/notes/rust.md");
    expect(view.getByText("notes/rust.md")).toBeTruthy();
    expect(view.getByText("#1")).toBeTruthy();
    expect(view.getByText("0.914")).toBeTruthy();
    expect(view.getByText("10–11")).toBeTruthy();
    expect(view.getByText(/Rust ownership keeps memory safe\./)).toBeTruthy();
    expect(view.getByText("20")).toBeTruthy();
    expect(view.getByText("Traits enable polymorphism.")).toBeTruthy();
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
    const view = renderTranscript(displayGroups(messageWithToolResult("plain output", "analyze_document")));

    await user.click(view.getByRole("button", { name: /analyze_document/i }));

    expect(view.getByText("plain output").tagName).toBe("PRE");
  });
});
