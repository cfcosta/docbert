import "../test/setup";

import { createRef } from "react";
import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { cleanup, render, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes, useLocation } from "react-router";

import { api } from "../lib/api";
import ChatTranscript from "./ChatTranscript";
import type { DisplayMessageGroup } from "./chat-message-groups";
import type { Message } from "./chat-message-codec";

const originalApi = { ...api };

function LocationObserver() {
  const location = useLocation();
  return <div data-testid="location-path">{`${location.pathname}${location.hash}`}</div>;
}

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

function messageWithThinking(text: string): Message {
  return {
    id: "assistant-thinking-1",
    role: "assistant",
    content: "",
    parts: [{ type: "thinking", text }],
  };
}

function displayGroups(message: Message): DisplayMessageGroup[] {
  return [{ message, nestedSubagents: [] }];
}

function renderTranscript(groups: DisplayMessageGroup[]) {
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <LocationObserver />
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

function captureScrollIntoViewTargets() {
  const targets: string[] = [];
  const original = window.HTMLElement.prototype.scrollIntoView;

  Object.defineProperty(window.HTMLElement.prototype, "scrollIntoView", {
    value: function () {
      targets.push((this as HTMLElement).id || (this as HTMLElement).getAttribute("id") || "");
    },
    configurable: true,
    writable: true,
  });

  return {
    targets,
    restore() {
      Object.defineProperty(window.HTMLElement.prototype, "scrollIntoView", {
        value: original,
        configurable: true,
        writable: true,
      });
    },
  };
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
  api.listDocuments = async (collection) => {
    if (collection !== "notes") {
      return [];
    }

    return [
      { doc_id: "#abc123", path: "rust.md", title: "Rust Guide" },
      { doc_id: "#def456", path: "linked.md", title: "Linked Guide" },
    ];
  };
  api.getDocument = async (collection, path) => ({
    doc_id: path === "linked.md" ? "#def456" : "#abc123",
    collection,
    path,
    title: path === "rust.md" ? "Rust Guide" : path === "linked.md" ? "Linked Guide" : path,
    content:
      path === "linked.md"
        ? "# Linked Guide\n\n## target heading\n\nLinked body"
        : "# Rust Guide\n\nRust ownership keeps memory safe.",
  });
});

afterEach(() => {
  cleanup();
  document.body.innerHTML = "";
  Object.assign(api, originalApi);
  mock.restore();
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

    expect(view.queryByRole("button", { name: "Rust Guide" })).toBeNull();

    await user.click(view.getByRole("button", { name: /search_hybrid/i }));

    expect(view.getByRole("button", { name: "Rust Guide" })).toBeTruthy();
    expect(view.getByText("notes/rust.md")).toBeTruthy();
  });

  test("clicking a search result opens an inline preview and keeps the chat route", async () => {
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
    await user.click(view.getByRole("button", { name: "Rust Guide" }));

    await waitFor(() => {
      expect(view.getByText("#abc123")).toBeTruthy();
    });

    expect(view.getByRole("link", { name: "Permalink" }).getAttribute("href")).toBe(
      "/documents/notes/rust.md",
    );
    expect(view.getByTestId("location-path").textContent).toBe("/");
  });

  test("clicking an excerpt opens the inline preview instead of navigating away", async () => {
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
    await user.click(
      view.getByRole("button", { name: /10\s*Rust ownership keeps memory safe\./i }),
    );

    await waitFor(() => {
      expect(view.getByRole("link", { name: "Permalink" })).toBeTruthy();
    });

    expect(view.getByTestId("location-path").textContent).toBe("/");
    expect(view.queryByText("Document route")).toBeNull();
  });

  test("successful_cross_note_wiki_link_click_keeps_chat_route_and_swaps_inline_preview", async () => {
    const user = userEvent.setup();
    api.getDocument = async (collection, path) => ({
      doc_id: path === "linked.md" ? "#def456" : "#abc123",
      collection,
      path,
      title: path === "linked.md" ? "Linked Guide" : "Rust Guide",
      content:
        path === "linked.md"
          ? "# Linked Guide\n\nLinked body"
          : "[[linked#target heading]]\n\n# Rust Guide\n\nRust ownership keeps memory safe.",
    });

    const results = JSON.stringify([
      {
        rank: 1,
        score: 0.914,
        doc_id: "#abc123",
        collection: "notes",
        path: "rust.md",
        title: "Rust Guide",
      },
    ]);

    const view = renderTranscript(displayGroups(messageWithToolResult(results)));

    await user.click(view.getByRole("button", { name: /search_hybrid/i }));
    await user.click(view.getByRole("button", { name: "Rust Guide" }));

    await waitFor(() => {
      expect(view.getByRole("link", { name: "linked" })).toBeTruthy();
    });

    await user.click(view.getByRole("link", { name: "linked" }));

    await waitFor(() => {
      expect(view.getByText("Linked body")).toBeTruthy();
    });

    expect(view.getByTestId("location-path").textContent).toBe("/");
    expect(view.queryByText("Rust ownership keeps memory safe.")).toBeNull();
  });

  test("same_note_heading_and_block_links_trigger_scroll_into_view", async () => {
    const user = userEvent.setup();
    const scrollSpy = captureScrollIntoViewTargets();
    api.getDocument = async (collection, path) => ({
      doc_id: "#abc123",
      collection,
      path,
      title: "Rust Guide",
      content: "[[#target heading]]\n\n[[#^block-id]]\n\n# target heading\n\nParagraph ^block-id",
    });

    const results = JSON.stringify([
      {
        rank: 1,
        score: 0.914,
        doc_id: "#abc123",
        collection: "notes",
        path: "rust.md",
        title: "Rust Guide",
      },
    ]);

    try {
      const view = renderTranscript(displayGroups(messageWithToolResult(results)));
      await user.click(view.getByRole("button", { name: /search_hybrid/i }));
      await user.click(view.getByRole("button", { name: "Rust Guide" }));

      await waitFor(() => {
        expect(view.getByRole("link", { name: "target heading" })).toBeTruthy();
      });

      await user.click(view.getByRole("link", { name: "target heading" }));
      await user.click(view.getByRole("link", { name: "^block-id" }));

      expect(scrollSpy.targets).toContain("preview-heading-target-heading");
      expect(scrollSpy.targets).toContain("preview-block-block-id");
    } finally {
      scrollSpy.restore();
    }
  });

  test("failed_list_documents_leaves_cross_note_links_as_plain_text_and_preserves_preview", async () => {
    const user = userEvent.setup();
    api.listDocuments = async () => {
      throw new Error("boom");
    };
    api.getDocument = async (collection, path) => ({
      doc_id: "#abc123",
      collection,
      path,
      title: "Rust Guide",
      content: "[[linked]]\n\n# Rust Guide\n\nRust ownership keeps memory safe.",
    });

    const results = JSON.stringify([
      {
        rank: 1,
        score: 0.914,
        doc_id: "#abc123",
        collection: "notes",
        path: "rust.md",
        title: "Rust Guide",
      },
    ]);

    const view = renderTranscript(displayGroups(messageWithToolResult(results)));
    await user.click(view.getByRole("button", { name: /search_hybrid/i }));
    await user.click(view.getByRole("button", { name: "Rust Guide" }));

    await waitFor(() => {
      expect(view.getByText("Rust ownership keeps memory safe.")).toBeTruthy();
    });

    expect(view.queryByRole("link", { name: "linked" })).toBeNull();
    expect(view.getByText("linked")).toBeTruthy();
    expect(view.getByText("Rust ownership keeps memory safe.")).toBeTruthy();
    expect(view.getByTestId("location-path").textContent).toBe("/");
  });

  test("list_documents_is_called_once_per_collection_after_caching", async () => {
    const user = userEvent.setup();
    const listDocumentsCalls: string[] = [];
    api.listDocuments = async (collection) => {
      listDocumentsCalls.push(collection);
      return [
        { doc_id: "#abc123", path: "rust.md", title: "Rust Guide" },
        { doc_id: "#def456", path: "linked.md", title: "Linked Guide" },
      ];
    };
    api.getDocument = async (collection, path) => ({
      doc_id: path === "linked.md" ? "#def456" : "#abc123",
      collection,
      path,
      title: path === "linked.md" ? "Linked Guide" : "Rust Guide",
      content:
        path === "linked.md"
          ? "# Linked Guide\n\nLinked body"
          : "[[linked]]\n\n# Rust Guide\n\nRust ownership keeps memory safe.",
    });

    const results = JSON.stringify([
      {
        rank: 1,
        score: 0.914,
        doc_id: "#abc123",
        collection: "notes",
        path: "rust.md",
        title: "Rust Guide",
      },
    ]);

    const view = renderTranscript(displayGroups(messageWithToolResult(results)));
    await user.click(view.getByRole("button", { name: /search_hybrid/i }));
    await user.click(view.getByRole("button", { name: "Rust Guide" }));

    await waitFor(() => {
      expect(view.getByRole("link", { name: "linked" })).toBeTruthy();
    });

    await user.click(view.getByRole("link", { name: "linked" }));
    await waitFor(() => {
      expect(view.getByText("Linked body")).toBeTruthy();
    });

    expect(listDocumentsCalls).toEqual(["notes"]);
  });

  test("renders no results for empty search payloads", async () => {
    const user = userEvent.setup();
    const view = renderTranscript(displayGroups(messageWithToolResult("[]")));

    await user.click(view.getByRole("button", { name: /search_hybrid/i }));

    expect(view.getByText("No results")).toBeTruthy();
  });

  test("search tool calls reuse the file-analysis chrome", async () => {
    const user = userEvent.setup();
    const results = JSON.stringify([]);
    const view = renderTranscript(displayGroups(messageWithToolResult(results)));

    const shell = view.container.querySelector(".chat-tool-call-search.chat-subagent-inline");
    const button = shell?.querySelector(".chat-subagent-header");
    expect(shell).toBeTruthy();
    expect(button).toBeTruthy();
    expect(button?.textContent).toContain("Search");
    expect(button?.textContent).toContain("rust");

    await user.click(view.getByRole("button", { name: /search_hybrid/i }));
    expect(shell?.querySelector(".chat-subagent-body")).toBeNull();
    const layout = shell?.querySelector(".chat-tool-search-preview-layout");
    expect(layout).toBeTruthy();
    expect(layout?.className).not.toContain("has-preview");
  });

  test("search tool calls use the full width until a preview opens", async () => {
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

    const layout = view.container.querySelector(".chat-tool-search-preview-layout");
    expect(layout).toBeTruthy();
    expect(layout?.className).not.toContain("has-preview");

    await user.click(view.getByRole("button", { name: "Rust Guide" }));

    await waitFor(() => {
      expect(
        view.container.querySelector(".chat-tool-search-preview-layout.has-preview"),
      ).toBeTruthy();
    });
  });

  test("reasoning uses the same expanded card treatment as search", async () => {
    const user = userEvent.setup();
    const view = renderTranscript(displayGroups(messageWithThinking("Plan first")));

    const shell = view.container.querySelector(".chat-reasoning-inline.chat-subagent-inline");
    const header = shell?.querySelector(".chat-subagent-header");
    const body = shell?.querySelector(".chat-reasoning-body");

    expect(shell?.className).toContain("expanded");
    expect(header).toBeTruthy();
    expect(body).toBeTruthy();
    expect(view.getByText("Plan first")).toBeTruthy();

    await user.click(view.getByRole("button", { name: /toggle reasoning/i }));
    expect(view.queryByText("Plan first")).toBeNull();
    expect(shell?.className).not.toContain("expanded");

    await user.click(view.getByRole("button", { name: /toggle reasoning/i }));
    expect(view.getByText("Plan first")).toBeTruthy();
    expect(shell?.className).toContain("expanded");
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
