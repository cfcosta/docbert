import "../test/setup";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes, useLocation } from "react-router";

import {
  api,
  type Collection,
  type DocumentListItem,
  type DocumentResponse,
  type IngestResponse,
} from "../lib/api";
import Documents from "./Documents";

const originalApi = { ...api };

type ApiTrackers = {
  listDocumentsCalls: Record<string, number>;
  ingested: Array<{ collection: string; paths: string[] }>;
  deleted: Array<{ collection: string; path: string }>;
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function LocationObserver() {
  const location = useLocation();
  return <div data-testid="location-path">{`${location.pathname}${location.hash}`}</div>;
}

function renderDocuments(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <LocationObserver />
      <Routes>
        <Route path="/documents" element={<Documents />} />
        <Route path="/documents/:collection/*" element={<Documents />} />
      </Routes>
    </MemoryRouter>,
  );
}

async function waitForCondition(condition: () => boolean, message: () => string, timeoutMs = 1000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (condition()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }

  throw new Error(message());
}

function collectionToggle(container: HTMLElement, name: string): HTMLButtonElement {
  const buttons = Array.from(container.getElementsByTagName("button"));
  const button = buttons.find(
    (candidate) =>
      candidate.className.includes("tree-collection-header") &&
      candidate.textContent?.includes(name),
  );
  if (!(button instanceof HTMLButtonElement)) {
    throw new Error(`collection toggle not found for ${name}`);
  }
  return button;
}

function uploadButton(container: HTMLElement, collection: string): HTMLButtonElement {
  const buttons = Array.from(container.getElementsByTagName("button"));
  const button = buttons.find(
    (candidate) =>
      candidate.getAttribute("aria-label") === `Upload Markdown files to ${collection}`,
  );
  if (!(button instanceof HTMLButtonElement)) {
    throw new Error(`upload button not found for ${collection}`);
  }
  return button;
}

function fileInput(container: HTMLElement): HTMLInputElement {
  const inputs = Array.from(container.getElementsByTagName("input"));
  const input = inputs.find((candidate) => candidate.type === "file");
  if (!(input instanceof HTMLInputElement)) {
    throw new Error("file input not found");
  }
  return input;
}

function fileRowDeleteButton(container: HTMLElement, fileName: string): HTMLButtonElement {
  const buttons = Array.from(container.getElementsByTagName("button"));
  const button = buttons.find(
    (candidate) => candidate.getAttribute("aria-label") === `Delete ${fileName}`,
  );
  if (!(button instanceof HTMLButtonElement)) {
    throw new Error(`delete button not found for ${fileName}`);
  }
  return button;
}

function treeConfirmDeleteButton(container: HTMLElement): HTMLButtonElement {
  const buttons = Array.from(container.getElementsByTagName("button"));
  const button = buttons.find(
    (candidate) =>
      candidate.className.includes("tree-confirm-yes") && candidate.textContent === "Delete",
  );
  if (!(button instanceof HTMLButtonElement)) {
    throw new Error("tree confirm delete button not found");
  }
  return button;
}

function treeFileButton(container: HTMLElement, fileName: string): HTMLButtonElement {
  const buttons = Array.from(container.getElementsByTagName("button"));
  const button = buttons.find(
    (candidate) =>
      candidate.className.includes("tree-file") && candidate.textContent?.includes(fileName),
  );
  if (!(button instanceof HTMLButtonElement)) {
    throw new Error(`document tree button not found for ${fileName}`);
  }
  return button;
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

function installDocumentsApiStubs({
  collections,
  docsByCollection,
  documentBodies,
  onIngest,
  onDelete,
  getDocument,
}: {
  collections: Collection[];
  docsByCollection: Record<string, DocumentListItem[]>;
  documentBodies: Record<string, DocumentResponse>;
  onIngest?: (collection: string, response: IngestResponse) => void;
  onDelete?: (collection: string, path: string) => void;
  getDocument?: (collection: string, path: string) => Promise<DocumentResponse>;
}): ApiTrackers {
  const trackers: ApiTrackers = {
    listDocumentsCalls: {},
    ingested: [],
    deleted: [],
  };

  api.listCollections = async () => collections;
  api.listDocuments = async (collection: string) => {
    trackers.listDocumentsCalls[collection] = (trackers.listDocumentsCalls[collection] ?? 0) + 1;
    return docsByCollection[collection] ?? [];
  };
  api.getDocument = async (collection: string, path: string) => {
    if (getDocument) {
      return getDocument(collection, path);
    }

    const key = `${collection}:${path}`;
    const document = documentBodies[key];
    if (!document) {
      throw new Error(`missing document ${key}`);
    }
    return document;
  };
  api.ingestDocuments = async (collection: string, documents) => {
    const response: IngestResponse = {
      ingested: documents.length,
      documents: documents.map((document) => ({
        doc_id: `#${document.path.replace(/[^a-z0-9]/gi, "").slice(0, 6) || "doc001"}`,
        path: document.path,
        title: document.path.replace(/\.[^.]+$/, ""),
      })),
    };
    trackers.ingested.push({ collection, paths: documents.map((document) => document.path) });
    onIngest?.(collection, response);
    return response;
  };
  api.deleteDocument = async (collection: string, path: string) => {
    trackers.deleted.push({ collection, path });
    onDelete?.(collection, path);
  };
  api.createCollection = async (name: string) => ({ name });
  api.deleteCollection = async () => {};
  api.getLlmSettings = async () => ({ provider: null, model: null, api_key: null });
  api.updateLlmSettings = async (settings) => settings;
  api.listConversations = async () => [];
  api.createConversation = async () => {
    throw new Error("unexpected conversation creation");
  };
  api.getConversation = async () => {
    throw new Error("unexpected conversation lookup");
  };
  api.updateConversation = async (_id, conversation) => conversation;
  api.deleteConversation = async () => {};
  api.search = async () => ({ query: "", mode: "semantic", result_count: 0, results: [] });

  return trackers;
}

beforeEach(() => {
  document.body.innerHTML = "";
});

afterEach(() => {
  Object.assign(api, originalApi);
});

describe("Documents page", () => {
  test("route_param_opens_selected_document_preview", async () => {
    installDocumentsApiStubs({
      collections: [{ name: "notes" }],
      docsByCollection: {
        notes: [{ doc_id: "#abc123", path: "hello.md", title: "Hello note" }],
      },
      documentBodies: {
        "notes:hello.md": {
          doc_id: "#abc123",
          collection: "notes",
          path: "hello.md",
          title: "Hello note",
          content: "# Hello note\n\nBody paragraph",
        },
      },
    });

    const view = renderDocuments("/documents/notes/hello.md");

    await waitForCondition(
      () => view.container.textContent?.includes("Hello note") ?? false,
      () => `preview title never rendered: ${JSON.stringify(view.container.textContent)}`,
    );
    await waitForCondition(
      () => view.container.textContent?.includes("Body paragraph") ?? false,
      () => `preview body never rendered: ${JSON.stringify(view.container.textContent)}`,
    );
  });

  test("cross_note_wiki_link_click_updates_pathname_and_hash", async () => {
    installDocumentsApiStubs({
      collections: [{ name: "notes" }],
      docsByCollection: {
        notes: [
          { doc_id: "#aaa111", path: "alpha.md", title: "Alpha note" },
          { doc_id: "#bbb222", path: "beta.md", title: "Beta note" },
        ],
      },
      documentBodies: {
        "notes:alpha.md": {
          doc_id: "#aaa111",
          collection: "notes",
          path: "alpha.md",
          title: "Alpha note",
          content: "[[beta#Section Two]]\n\n# Alpha",
        },
        "notes:beta.md": {
          doc_id: "#bbb222",
          collection: "notes",
          path: "beta.md",
          title: "Beta note",
          content: "# Section Two\n\nBeta body",
        },
      },
    });

    const user = userEvent.setup({ pointerEventsCheck: 0 });
    const view = renderDocuments("/documents/notes/alpha.md");

    await waitForCondition(
      () => view.container.textContent?.includes("beta") ?? false,
      () => `beta wiki link never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    await user.click(view.getByRole("link", { name: "beta" }));

    await waitForCondition(
      () =>
        view.getByTestId("location-path").textContent === "/documents/notes/beta.md#Section%20Two",
      () => `beta route+hash never updated: ${view.getByTestId("location-path").textContent}`,
    );
    await waitForCondition(
      () => view.container.textContent?.includes("Beta body") ?? false,
      () => `beta preview body never rendered: ${JSON.stringify(view.container.textContent)}`,
    );
  });

  test("fragment_bearing_document_opens_scroll_to_the_expected_target", async () => {
    installDocumentsApiStubs({
      collections: [{ name: "notes" }],
      docsByCollection: {
        notes: [{ doc_id: "#bbb222", path: "beta.md", title: "Beta note" }],
      },
      documentBodies: {
        "notes:beta.md": {
          doc_id: "#bbb222",
          collection: "notes",
          path: "beta.md",
          title: "Beta note",
          content: "# Section Two\n\nBeta body\n\nParagraph target ^block-id",
        },
      },
    });

    const scrollSpy = captureScrollIntoViewTargets();
    try {
      const view = renderDocuments("/documents/notes/beta.md#^block-id");

      await waitForCondition(
        () => view.container.textContent?.includes("Beta body") ?? false,
        () => `beta preview body never rendered: ${JSON.stringify(view.container.textContent)}`,
      );
      await waitForCondition(
        () => scrollSpy.targets.includes("preview-block-block-id"),
        () => `block target was never scrolled into view: ${JSON.stringify(scrollSpy.targets)}`,
      );
    } finally {
      scrollSpy.restore();
    }
  });

  test("collection_management_controls_are_hidden", async () => {
    installDocumentsApiStubs({
      collections: [{ name: "notes" }],
      docsByCollection: { notes: [] },
      documentBodies: {},
    });

    const view = renderDocuments("/documents");

    await waitForCondition(
      () => view.container.textContent?.includes("notes") ?? false,
      () => `collection never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    expect(view.queryByRole("button", { name: "Create collection" })).toBeNull();
    expect(view.queryByLabelText("Delete collection notes")).toBeNull();
  });

  test("uploading_markdown_files_refreshes_collection_and_shows_success_status", async () => {
    const docsByCollection: Record<string, DocumentListItem[]> = { notes: [] };
    installDocumentsApiStubs({
      collections: [{ name: "notes" }],
      docsByCollection,
      documentBodies: {},
      onIngest: (collection) => {
        docsByCollection[collection] = [
          { doc_id: "#upl001", path: "uploaded.md", title: "uploaded" },
        ];
      },
    });
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    const view = renderDocuments("/documents");

    await waitForCondition(
      () => view.container.textContent?.includes("notes") ?? false,
      () => `collection never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    await user.click(uploadButton(view.container, "notes"));
    await user.upload(
      fileInput(view.container),
      new File(["# Uploaded\n\nBody"], "uploaded.md", { type: "text/markdown" }),
    );

    await waitForCondition(
      () => view.container.textContent?.includes("Ingested 1 file into notes.") ?? false,
      () => `success status never rendered: ${JSON.stringify(view.container.textContent)}`,
    );
    await waitForCondition(
      () => view.container.textContent?.includes("uploaded.md") ?? false,
      () => `uploaded file never appeared in tree: ${JSON.stringify(view.container.textContent)}`,
    );
  });

  test("switching_from_one_selected_document_to_another_updates_preview_and_route", async () => {
    installDocumentsApiStubs({
      collections: [{ name: "notes" }],
      docsByCollection: {
        notes: [
          { doc_id: "#aaa111", path: "alpha.md", title: "Alpha note" },
          { doc_id: "#bbb222", path: "beta.md", title: "Beta note" },
        ],
      },
      documentBodies: {
        "notes:alpha.md": {
          doc_id: "#aaa111",
          collection: "notes",
          path: "alpha.md",
          title: "Alpha note",
          content: "# Alpha note\n\nAlpha body",
        },
        "notes:beta.md": {
          doc_id: "#bbb222",
          collection: "notes",
          path: "beta.md",
          title: "Beta note",
          content: "# Beta note\n\nBeta body",
        },
      },
    });

    const user = userEvent.setup({ pointerEventsCheck: 0 });
    const view = renderDocuments("/documents");

    await waitForCondition(
      () => view.container.textContent?.includes("notes") ?? false,
      () => `collection never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    await user.click(collectionToggle(view.container, "notes"));
    await waitForCondition(
      () => view.container.textContent?.includes("alpha.md") ?? false,
      () => `alpha document never rendered in tree: ${JSON.stringify(view.container.textContent)}`,
    );
    await waitForCondition(
      () => view.container.textContent?.includes("beta.md") ?? false,
      () => `beta document never rendered in tree: ${JSON.stringify(view.container.textContent)}`,
    );

    await user.click(treeFileButton(view.container, "alpha.md"));
    await waitForCondition(
      () => view.container.textContent?.includes("Alpha body") ?? false,
      () => `alpha preview body never rendered: ${JSON.stringify(view.container.textContent)}`,
    );
    await waitForCondition(
      () => view.getByTestId("location-path").textContent === "/documents/notes/alpha.md",
      () => `alpha route never updated: ${view.getByTestId("location-path").textContent}`,
    );

    await user.click(treeFileButton(view.container, "beta.md"));
    await waitForCondition(
      () => view.container.textContent?.includes("Beta note") ?? false,
      () => `beta preview title never rendered: ${JSON.stringify(view.container.textContent)}`,
    );
    await waitForCondition(
      () => view.container.textContent?.includes("Beta body") ?? false,
      () => `beta preview body never rendered: ${JSON.stringify(view.container.textContent)}`,
    );
    await waitForCondition(
      () => view.getByTestId("location-path").textContent === "/documents/notes/beta.md",
      () => `beta route never updated: ${view.getByTestId("location-path").textContent}`,
    );

    expect(view.container.textContent).not.toContain("Alpha body");
  });

  test("stale_document_response_does_not_override_latest_selection", async () => {
    const alphaResponse = deferred<DocumentResponse>();
    const betaResponse = deferred<DocumentResponse>();

    installDocumentsApiStubs({
      collections: [{ name: "notes" }],
      docsByCollection: {
        notes: [
          { doc_id: "#aaa111", path: "alpha.md", title: "Alpha note" },
          { doc_id: "#bbb222", path: "beta.md", title: "Beta note" },
        ],
      },
      documentBodies: {},
      getDocument: async (_collection, path) => {
        if (path === "alpha.md") {
          return alphaResponse.promise;
        }
        if (path === "beta.md") {
          return betaResponse.promise;
        }
        throw new Error(`unexpected document path ${path}`);
      },
    });

    const user = userEvent.setup({ pointerEventsCheck: 0 });
    const view = renderDocuments("/documents");

    await waitForCondition(
      () => view.container.textContent?.includes("notes") ?? false,
      () => `collection never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    await user.click(collectionToggle(view.container, "notes"));
    await waitForCondition(
      () => view.container.textContent?.includes("alpha.md") ?? false,
      () => `alpha document never rendered in tree: ${JSON.stringify(view.container.textContent)}`,
    );
    await waitForCondition(
      () => view.container.textContent?.includes("beta.md") ?? false,
      () => `beta document never rendered in tree: ${JSON.stringify(view.container.textContent)}`,
    );

    await user.click(treeFileButton(view.container, "alpha.md"));
    await waitForCondition(
      () => view.getByTestId("location-path").textContent === "/documents/notes/alpha.md",
      () => `alpha route never updated: ${view.getByTestId("location-path").textContent}`,
    );

    await user.click(treeFileButton(view.container, "beta.md"));
    await waitForCondition(
      () => view.getByTestId("location-path").textContent === "/documents/notes/beta.md",
      () => `beta route never updated: ${view.getByTestId("location-path").textContent}`,
    );

    betaResponse.resolve({
      doc_id: "#bbb222",
      collection: "notes",
      path: "beta.md",
      title: "Beta note",
      content: "# Beta note\n\nBeta body",
    });
    await waitForCondition(
      () => view.container.textContent?.includes("Beta body") ?? false,
      () => `beta preview body never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    alphaResponse.resolve({
      doc_id: "#aaa111",
      collection: "notes",
      path: "alpha.md",
      title: "Alpha note",
      content: "# Alpha note\n\nAlpha body",
    });

    await waitForCondition(
      () => view.getByTestId("location-path").textContent === "/documents/notes/beta.md",
      () =>
        `route regressed after stale alpha response: ${view.getByTestId("location-path").textContent}`,
    );
    await waitForCondition(
      () => view.container.textContent?.includes("Beta body") ?? false,
      () =>
        `beta preview body disappeared after stale alpha response: ${JSON.stringify(view.container.textContent)}`,
    );

    expect(view.container.textContent).not.toContain("Alpha body");
  });

  test("deleting_selected_document_clears_preview_and_selection", async () => {
    const docsByCollection: Record<string, DocumentListItem[]> = {
      notes: [{ doc_id: "#abc123", path: "hello.md", title: "Hello note" }],
    };
    const trackers = installDocumentsApiStubs({
      collections: [{ name: "notes" }],
      docsByCollection,
      documentBodies: {
        "notes:hello.md": {
          doc_id: "#abc123",
          collection: "notes",
          path: "hello.md",
          title: "Hello note",
          content: "# Hello note\n\nBody paragraph",
        },
      },
      onDelete: (collection, path) => {
        docsByCollection[collection] = docsByCollection[collection].filter(
          (doc) => doc.path !== path,
        );
      },
    });
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    const view = renderDocuments("/documents");

    await waitForCondition(
      () => view.container.textContent?.includes("notes") ?? false,
      () => `collection never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    await user.click(collectionToggle(view.container, "notes"));
    await waitForCondition(
      () => view.container.textContent?.includes("hello.md") ?? false,
      () => `document never rendered in tree: ${JSON.stringify(view.container.textContent)}`,
    );

    const documentButton = Array.from(view.container.getElementsByTagName("button")).find(
      (candidate) =>
        candidate.className.includes("tree-file") && candidate.textContent?.includes("hello.md"),
    );
    if (!(documentButton instanceof HTMLButtonElement)) {
      throw new Error("document tree button not found");
    }

    await user.click(documentButton);
    await waitForCondition(
      () => view.container.textContent?.includes("Body paragraph") ?? false,
      () => `preview body never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    await user.click(fileRowDeleteButton(view.container, "hello.md"));
    await user.click(treeConfirmDeleteButton(view.container));

    await waitForCondition(
      () => trackers.deleted.length > 0,
      () => `delete request never fired: ${JSON.stringify(view.container.textContent)}`,
    );

    await waitForCondition(
      () => view.container.textContent?.includes("No document selected") ?? false,
      () => `preview was not cleared after delete: ${JSON.stringify(view.container.textContent)}`,
    );
  });

  test("expanding_collection_loads_documents_once_and_renders_sorted_tree", async () => {
    const trackers = installDocumentsApiStubs({
      collections: [{ name: "notes" }],
      docsByCollection: {
        notes: [
          { doc_id: "#z00001", path: "zeta.md", title: "Zeta" },
          { doc_id: "#a00001", path: "alpha.md", title: "Alpha" },
          { doc_id: "#n00001", path: "nested/beta.md", title: "Beta" },
        ],
      },
      documentBodies: {},
    });
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    const view = renderDocuments("/documents");

    await waitForCondition(
      () => view.container.textContent?.includes("notes") ?? false,
      () => `collection never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    const header = collectionToggle(view.container, "notes");
    await user.click(header);
    await waitForCondition(
      () => view.container.textContent?.includes("nested") ?? false,
      () => `tree never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    const treeText = view.container.textContent ?? "";
    expect(treeText.indexOf("nested")).toBeLessThan(treeText.indexOf("alpha.md"));
    expect(treeText.indexOf("alpha.md")).toBeLessThan(treeText.indexOf("zeta.md"));
    expect(trackers.listDocumentsCalls.notes).toBe(1);

    await user.click(header);
    await user.click(header);
    expect(trackers.listDocumentsCalls.notes).toBe(1);
  });
});
