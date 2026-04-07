import "../test/setup";

import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { render, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router";

import App from "../App";
import { api, type SearchResponse } from "../lib/api";
import Search from "./Search";

const originalApi = { ...api };

function renderSearchRoute(route = "/search") {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <Routes>
        <Route element={<App />}>
          <Route path="/search" element={<Search />} />
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

function searchResponse(result_count: number): SearchResponse {
  return {
    query: "rust",
    mode: "hybrid",
    result_count,
    results: [],
  };
}

beforeEach(() => {
  api.listCollections = async () => [{ name: "notes" }, { name: "docs" }];
  api.search = async () => searchResponse(2);
});

afterEach(() => {
  Object.assign(api, originalApi);
  mock.restore();
});

describe("Search page", () => {
  test("renders controls with hybrid default mode and loaded collection options", async () => {
    const view = renderSearchRoute();

    await waitFor(() => {
      expect(view.getByText("Start with a search query")).toBeTruthy();
    });

    expect(view.getByRole("heading", { name: "Search" })).toBeTruthy();
    expect(view.getByLabelText("Query")).toBeTruthy();

    const mode = view.getByLabelText("Mode") as HTMLSelectElement;
    expect(mode.value).toBe("hybrid");

    const collection = view.getByLabelText("Collection") as HTMLSelectElement;
    const optionLabels = Array.from(collection.options).map((option) => option.textContent);
    expect(optionLabels).toEqual(["All collections", "notes", "docs"]);
    expect(
      view.getByText(
        "Enter a query above to search across all collections or narrow the scope with the collection filter.",
      ),
    ).toBeTruthy();
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

  test("debounces api calls while typing and shows a minimal success summary", async () => {
    const user = userEvent.setup();
    const searchCalls: Array<{ query: string; mode?: string; collection?: string }> = [];
    api.search = async (params) => {
      searchCalls.push(params);
      return searchResponse(2);
    };

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByText("Start with a search query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");
    expect(searchCalls).toHaveLength(0);

    await waitFor(() => {
      expect(searchCalls).toHaveLength(1);
    });

    expect(searchCalls[0]).toMatchObject({ query: "rust", mode: "hybrid", collection: undefined });
    expect(view.getByText("Found 2 results")).toBeTruthy();
  });

  test("blank query suppresses search calls and preserves the pre-search empty state", async () => {
    const user = userEvent.setup();
    const searchSpy = mock(async () => searchResponse(2));
    api.search = searchSpy;

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByText("Start with a search query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "   ");
    await new Promise((resolve) => setTimeout(resolve, 260));

    expect(searchSpy).toHaveBeenCalledTimes(0);
    expect(view.getByText("Start with a search query")).toBeTruthy();
  });

  test("reruns search when the collection changes", async () => {
    const user = userEvent.setup();
    const searchCalls: Array<{ query: string; mode?: string; collection?: string }> = [];
    api.search = async (params) => {
      searchCalls.push(params);
      return searchResponse(1);
    };

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByText("Start with a search query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");
    await waitFor(() => expect(searchCalls).toHaveLength(1));

    await user.selectOptions(view.getByLabelText("Collection"), "notes");
    await waitFor(() => expect(searchCalls).toHaveLength(2));

    expect(searchCalls[1]).toMatchObject({ query: "rust", collection: "notes", mode: "hybrid" });
  });

  test("reruns search when the mode changes", async () => {
    const user = userEvent.setup();
    const searchCalls: Array<{ query: string; mode?: string; collection?: string }> = [];
    api.search = async (params) => {
      searchCalls.push(params);
      return searchResponse(1);
    };

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByText("Start with a search query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");
    await waitFor(() => expect(searchCalls).toHaveLength(1));

    await user.selectOptions(view.getByLabelText("Mode"), "semantic");
    await waitFor(() => expect(searchCalls).toHaveLength(2));

    expect(searchCalls[1]).toMatchObject({ query: "rust", mode: "semantic", collection: undefined });
  });

  test("shows loading state while a search is in flight", async () => {
    const user = userEvent.setup();
    const pending = deferred<SearchResponse>();
    api.search = () => pending.promise;

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByText("Start with a search query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");

    await waitFor(() => {
      expect(view.getByText("Searching…")).toBeTruthy();
    });

    pending.resolve(searchResponse(2));
    await waitFor(() => expect(view.getByText("Found 2 results")).toBeTruthy());
  });

  test("shows error state when a search fails", async () => {
    const user = userEvent.setup();
    api.search = async () => {
      throw new Error("search backend unavailable");
    };

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByText("Start with a search query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");

    await waitFor(() => {
      expect(view.getByRole("alert").textContent).toContain("Search failed");
    });
    expect(view.getByRole("alert").textContent).toContain("search backend unavailable");
  });

  test("shows a no-results state when the search returns no matches", async () => {
    const user = userEvent.setup();
    api.search = async () => searchResponse(0);

    const view = renderSearchRoute();
    await waitFor(() => expect(view.getByText("Start with a search query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");

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
    await waitFor(() => expect(view.getByText("Start with a search query")).toBeTruthy());

    await user.type(view.getByLabelText("Query"), "rust");
    await waitFor(() => expect(calls).toEqual(["rust"]));

    await user.clear(view.getByLabelText("Query"));
    await user.type(view.getByLabelText("Query"), "rust async");
    await waitFor(() => expect(calls).toEqual(["rust", "rust async"]));

    second.resolve({ ...searchResponse(2), query: "rust async", result_count: 5 });
    await waitFor(() => expect(view.getByText("Found 5 results")).toBeTruthy());

    first.resolve({ ...searchResponse(2), query: "rust", result_count: 1 });
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(view.getByText("Found 5 results")).toBeTruthy();
    expect(view.queryByText("Found 1 results")).toBeNull();
  });
});
