import "../test/setup";

import { describe, expect, mock, test } from "bun:test";
import { render, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";

import DocumentPreview, { type ResolvedDocumentTarget } from "./document-preview";
import type { SelectedDocumentSummary } from "./documents-tree";

const selectedDoc: SelectedDocumentSummary = {
  collection: "notes",
  path: "current.md",
  title: "Current",
  doc_id: "#cur001",
};

function renderPreview(
  preview: string,
  options: {
    resolverDocuments?: Array<{ path: string }>;
    onOpenResolvedDocument?: (target: ResolvedDocumentTarget) => void;
  } = {},
) {
  return render(
    <MemoryRouter>
      <DocumentPreview
        selectedDoc={selectedDoc}
        preview={preview}
        resolverDocuments={options.resolverDocuments}
        onOpenResolvedDocument={options.onOpenResolvedDocument}
      />
    </MemoryRouter>,
  );
}

describe("DocumentPreview", () => {
  test("renders deterministic heading ids including duplicates", () => {
    const view = renderPreview("# Overview\n\n## Overview\n\n### Deep Dive\n\n## Overview");
    const headings = within(view.container).getAllByRole("heading");
    const markdownHeadings = headings.filter(
      (heading) => heading.tagName !== "H2" || heading.textContent !== "Current",
    );

    expect(markdownHeadings.map((heading) => heading.getAttribute("id"))).toEqual([
      "preview-heading-overview",
      "preview-heading-overview-2",
      "preview-heading-deep-dive",
      "preview-heading-overview-3",
    ]);
  });

  test("renders deterministic block-ref target ids", () => {
    const view = renderPreview("Paragraph body ^block-id");
    const paragraph = view.getByText("Paragraph body");

    expect(paragraph.getAttribute("id")).toBe("preview-block-block-id");
  });

  test("same_note_heading_links_scroll_to_the_expected_target", async () => {
    const user = userEvent.setup();
    const calls: string[] = [];
    const original = window.HTMLElement.prototype.scrollIntoView;
    Object.defineProperty(window.HTMLElement.prototype, "scrollIntoView", {
      value: function () {
        calls.push((this as HTMLElement).id);
      },
      configurable: true,
      writable: true,
    });

    try {
      const view = renderPreview("[[#Section]]\n\n# Section");
      await user.click(view.getByRole("link", { name: "Section" }));
      expect(calls).toEqual(["preview-heading-section"]);
    } finally {
      Object.defineProperty(window.HTMLElement.prototype, "scrollIntoView", {
        value: original,
        configurable: true,
        writable: true,
      });
    }
  });

  test("same_note_block_links_scroll_to_the_expected_target", async () => {
    const user = userEvent.setup();
    const calls: string[] = [];
    const original = window.HTMLElement.prototype.scrollIntoView;
    Object.defineProperty(window.HTMLElement.prototype, "scrollIntoView", {
      value: function () {
        calls.push((this as HTMLElement).id);
      },
      configurable: true,
      writable: true,
    });

    try {
      const view = renderPreview("[[#^block-id]]\n\nParagraph body ^block-id");
      await user.click(view.getByRole("link", { name: "^block-id" }));
      expect(calls).toEqual(["preview-block-block-id"]);
    } finally {
      Object.defineProperty(window.HTMLElement.prototype, "scrollIntoView", {
        value: original,
        configurable: true,
        writable: true,
      });
    }
  });

  test("unresolved_links_render_as_plain_text", () => {
    const view = renderPreview("[[Missing]]", { resolverDocuments: [{ path: "Other.md" }] });

    expect(view.getByText("Missing")).toBeTruthy();
    expect(view.queryByRole("link", { name: "Missing" })).toBeNull();
  });

  test("resolved_cross_note_links_invoke_the_callback_handoff", async () => {
    const user = userEvent.setup();
    const onOpenResolvedDocument = mock(() => undefined);
    const view = renderPreview("[[Other#Part]]", {
      resolverDocuments: [{ path: "Other.md" }],
      onOpenResolvedDocument,
    });

    await user.click(view.getByRole("link", { name: "Other" }));

    expect(onOpenResolvedDocument).toHaveBeenCalledTimes(1);
    expect(onOpenResolvedDocument).toHaveBeenCalledWith({
      collection: "notes",
      path: "Other.md",
      fragment: "Part",
    });
  });
});
