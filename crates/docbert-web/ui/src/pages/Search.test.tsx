import "../test/setup";

import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { cleanup, render, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes, useLocation } from "react-router";

import App from "../App";
import { api, type SearchResponse, type SearchResult } from "../lib/api";
import Search from "./Search";

const originalApi = { ...api };

function LocationObserver() {
  const location = useLocation();
  return <div data-testid="location-path">{location.pathname}</div>;
}

function renderSearchRoute(route = "/search") {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <LocationObserver />
      <Routes>
        <Route element={<App />}>
          <Route path="/search" element={<Search />} />
          <Route path="/documents" element={<div>Documents page</div>} />
          <Route path="/documents/:collection/*" element={<div>Document route</div>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function result(overrides: Partial<SearchResult> = {}): SearchResult {
  return {
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
    ...overrides,
  };
}

function searchResponse(result_count: number, results: SearchResult[] = []): SearchResponse {
  return {
    query: "rust",
    mode: "hybrid",
    result_count,
    results,
  };
}

beforeEach(() => {
  document.body.innerHTML = "";
  api.listCollections = async () => [{ name: "notes" }, { name: "docs" }];
  api.search = async () =>
    searchResponse(2, [
      result(),
      result({ rank: 2, doc_id: "#def456", path: "async.md", title: "Async Rust" }),
    ]);
});

afterEach(() => {
  cleanup();
  document.body.innerHTML = "";
  Object.assign(api, originalApi);
  mock.restore();
});

describe("Search page", () => {
  test("renders controls with hybrid default mode and loaded collection options", async () => {
    const view = renderSearchRoute();

    await waitFor(() => {
      expect(view.getByLabelText("Query")).toBeTruthy();
    });

    expect(view.getByLabelText("Query")).toBeTruthy();

    const mode = view.getByLabelText("Mode") as HTMLSelectElement;
    expect(mode.value).toBe("hybrid");

    const collection = view.getByLabelText("Collection") as HTMLSelectElement;
    const optionLabels = Array.from(collection.options).map((option) => option.textContent);
    expect(optionLabels).toEqual(["All collections", "notes", "docs"]);
    expect(view.getByRole("button", { name: "Run search" })).toBeTruthy();
  });

  test("shows collection loading state before collections are available", async () => {
    api.listCollections = () => new Promise<Array<{ name: string }>>(() => undefined);

    const view = renderSearchRoute();

    expect(view.getByText("Loading collections…")).toBeTruthy();
    expect(view.getByText("Fetching available collections before search is enabled.")).toBeTruthy();
    expect((view.getByLabelText("Collection") as HTMLSelectElement).disabled).toBe(true);
  });

  test("shows collection load failure state", async () => {
    api.listCollections = async () => {
      throw new Error("config db unavailable");
    };

    const view = renderSearchRoute();

    await waitFor(() => {
      expect(view.getByRole("alert")).toBeTruthy();
    });

    expect(view.getByRole("alert").textContent).toContain("Could not load collections");
    expect(view.getByRole("alert").textContent).toContain("config db unavailable");
    expect((view.getByLabelText("Collection") as HTMLSelectElement).disabled).toBe(true);
  });

  test("submits api calls explicitly and shows a minimal success summary", async () => {
    const user = userEvent.setup();
    const searchCalls: Array<{ query: string; mode?: string; collection?: string }> = [];
    api.search = async (params) => {
      searchCalls.push(params);
      return searchResponse(2, [
        result(),
        result({ rank: 2, doc_id: "#def456", path: "async.md", title: "Async Rust" }),
      ]);
    };

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByLabelText("Query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");
    expect(searchCalls).toHaveLength(0);

    await user.click(view.getByRole("button", { name: "Run search" }));

    await waitFor(() => {
      expect(searchCalls).toHaveLength(1);
    });

    expect(searchCalls[0]).toMatchObject({ query: "rust", mode: "hybrid", collection: undefined });
    expect(view.getByText("Found 2 results")).toBeTruthy();
  });

  test("renders full search results through the shared component and supports document navigation", async () => {
    const user = userEvent.setup();
    api.search = async () =>
      searchResponse(2, [
        result(),
        result({ rank: 2, doc_id: "#def456", path: "nested/async.md", title: "Async Rust" }),
      ]);

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByLabelText("Query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");
    await user.click(view.getByRole("button", { name: "Run search" }));

    await waitFor(() => {
      expect(view.getByText("Rust Guide")).toBeTruthy();
    });

    expect(view.container.querySelector(".chat-tool-search-results")).toBeTruthy();
    expect(view.getByText("notes/rust.md")).toBeTruthy();
    expect(view.getByText("#1")).toBeTruthy();
    expect(view.getAllByText("0.914")).toHaveLength(2);
    expect(view.getAllByText("10–11")).toHaveLength(2);
    expect(view.getAllByText("Traits enable polymorphism.")).toHaveLength(2);

    await user.click(view.getByRole("link", { name: "Rust Guide" }));
    expect(view.getByText("Document route")).toBeTruthy();
    expect(view.getByTestId("location-path").textContent).toBe("/documents/notes/rust.md");
  });

  test("excerpt links navigate to the existing documents route", async () => {
    const user = userEvent.setup();
    api.search = async () => searchResponse(1, [result({ path: "nested/rust.md" })]);

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByLabelText("Query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");
    await user.click(view.getByRole("button", { name: "Run search" }));

    await waitFor(() => {
      expect(
        view.getByRole("link", { name: /10–11\s*Rust ownership keeps memory safe\./i }),
      ).toBeTruthy();
    });

    await user.click(
      view.getByRole("link", { name: /10–11\s*Rust ownership keeps memory safe\./i }),
    );
    expect(view.getByText("Document route")).toBeTruthy();
    expect(view.getByTestId("location-path").textContent).toBe("/documents/notes/nested/rust.md");
  });

  test("blank query suppresses search calls and preserves the pre-search empty state", async () => {
    const user = userEvent.setup();
    const searchSpy = mock(async () =>
      searchResponse(2, [
        result(),
        result({ rank: 2, doc_id: "#def456", path: "async.md", title: "Async Rust" }),
      ]),
    );
    api.search = searchSpy;

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByLabelText("Query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "   ");

    expect(searchSpy).toHaveBeenCalledTimes(0);
    expect(view.getByRole("button", { name: "Run search" })).toBeTruthy();
  });

  test("resubmits search when the collection changes", async () => {
    const user = userEvent.setup();
    const searchCalls: Array<{ query: string; mode?: string; collection?: string }> = [];
    api.search = async (params) => {
      searchCalls.push(params);
      return searchResponse(1, [result()]);
    };

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByLabelText("Query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");
    await user.click(view.getByRole("button", { name: "Run search" }));
    await waitFor(() => expect(searchCalls).toHaveLength(1));

    await user.selectOptions(view.getByLabelText("Collection"), "notes");
    await user.click(view.getByRole("button", { name: "Run search" }));
    await waitFor(() => expect(searchCalls).toHaveLength(2));

    expect(searchCalls[1]).toMatchObject({ query: "rust", collection: "notes", mode: "hybrid" });
  });

  test("resubmits search when the mode changes", async () => {
    const user = userEvent.setup();
    const searchCalls: Array<{ query: string; mode?: string; collection?: string }> = [];
    api.search = async (params) => {
      searchCalls.push(params);
      return searchResponse(1, [result()]);
    };

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByLabelText("Query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");
    await user.click(view.getByRole("button", { name: "Run search" }));
    await waitFor(() => expect(searchCalls).toHaveLength(1));

    await user.selectOptions(view.getByLabelText("Mode"), "semantic");
    await user.click(view.getByRole("button", { name: "Run search" }));
    await waitFor(() => expect(searchCalls).toHaveLength(2));

    expect(searchCalls[1]).toMatchObject({
      query: "rust",
      mode: "semantic",
      collection: undefined,
    });
  });

  test("shows loading state while a search is in flight", async () => {
    const user = userEvent.setup();
    const pending = deferred<SearchResponse>();
    api.search = () => pending.promise;

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByLabelText("Query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");
    await user.click(view.getByRole("button", { name: "Run search" }));

    await waitFor(() => {
      expect(view.getAllByText("Searching…")).toHaveLength(2);
    });

    pending.resolve(
      searchResponse(2, [
        result(),
        result({ rank: 2, doc_id: "#def456", path: "async.md", title: "Async Rust" }),
      ]),
    );
    await waitFor(() => expect(view.getByText("Found 2 results")).toBeTruthy());
  });

  test("shows error state when a search fails", async () => {
    const user = userEvent.setup();
    api.search = async () => {
      throw new Error("search backend unavailable");
    };

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByLabelText("Query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");
    await user.click(view.getByRole("button", { name: "Run search" }));

    await waitFor(() => {
      expect(view.getByRole("alert").textContent).toContain("Search failed");
    });
    expect(view.getByRole("alert").textContent).toContain("search backend unavailable");
  });

  test("shows a no-results state when the search returns no matches", async () => {
    const user = userEvent.setup();
    api.search = async () => searchResponse(0);

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByLabelText("Query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");
    await user.click(view.getByRole("button", { name: "Run search" }));

    await waitFor(() => {
      expect(view.getByText("No results")).toBeTruthy();
    });
    expect(view.getByText("Try a different query, mode, or collection filter.")).toBeTruthy();
  });

  test("ignores stale older responses when a newer search completes later", async () => {
    const user = userEvent.setup();
    const first = deferred<SearchResponse>();
    const second = deferred<SearchResponse>();
    const calls: string[] = [];

    api.search = ({ query }) => {
      calls.push(query);
      return calls.length === 1 ? first.promise : second.promise;
    };

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByLabelText("Query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");
    await user.click(view.getByRole("button", { name: "Run search" }));
    await waitFor(() => expect(calls).toEqual(["rust"]));

    await user.clear(view.getByLabelText("Query"));
    await user.type(view.getByLabelText("Query"), "rust async");
    await user.click(view.getByRole("button", { name: "Run search" }));
    await waitFor(() => expect(calls).toEqual(["rust", "rust async"]));

    second.resolve({
      ...searchResponse(5, [
        result({ rank: 1, title: "Async One", path: "async-one.md" }),
        result({ rank: 2, doc_id: "#def456", title: "Async Two", path: "async-two.md" }),
      ]),
      query: "rust async",
      result_count: 5,
    });
    await waitFor(() => expect(view.getByText("Found 5 results")).toBeTruthy());

    first.resolve({
      ...searchResponse(1, [result({ title: "Old Rust Result" })]),
      query: "rust",
      result_count: 1,
    });
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(view.getByText("Found 5 results")).toBeTruthy();
    expect(view.queryByText("Found 1 results")).toBeNull();
  });

  test("preserves search state when navigating away and back", async () => {
    const user = userEvent.setup();
    const searchCalls: Array<{ query: string; mode?: string; collection?: string }> = [];
    api.search = async (params) => {
      searchCalls.push(params);
      return searchResponse(2, [
        result(),
        result({ rank: 2, doc_id: "#def456", path: "async.md", title: "Async Rust" }),
      ]);
    };

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByLabelText("Query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");
    await user.click(view.getByRole("button", { name: "Run search" }));
    await waitFor(() => expect(searchCalls).toHaveLength(1));

    await user.selectOptions(view.getByLabelText("Mode"), "semantic");
    await user.click(view.getByRole("button", { name: "Run search" }));
    await waitFor(() => expect(searchCalls).toHaveLength(2));

    await user.selectOptions(view.getByLabelText("Collection"), "notes");
    await user.click(view.getByRole("button", { name: "Run search" }));
    await waitFor(() => expect(searchCalls).toHaveLength(3));
    await waitFor(() => expect(view.getByText("Rust Guide")).toBeTruthy());

    await user.click(view.getByRole("link", { name: /^Documents$/i }));
    expect(view.getByText("Documents page")).toBeTruthy();
    expect(view.getByTestId("location-path").textContent).toBe("/documents");

    await user.click(view.getByRole("link", { name: /^Search$/i }));

    await waitFor(() => expect(view.getByLabelText("Query")).toBeTruthy());
    expect(view.getByTestId("location-path").textContent).toBe("/search");
    expect((view.getByLabelText("Query") as HTMLInputElement).value).toBe("rust");
    expect((view.getByLabelText("Mode") as HTMLSelectElement).value).toBe("semantic");
    expect((view.getByLabelText("Collection") as HTMLSelectElement).value).toBe("notes");
    expect(view.getByText("Found 2 results")).toBeTruthy();
    expect(view.getByText("Rust Guide")).toBeTruthy();
    expect(searchCalls).toHaveLength(3);
  });
});
