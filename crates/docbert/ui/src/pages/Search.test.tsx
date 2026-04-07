import "../test/setup";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { render, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";

import App from "../App";
import { api } from "../lib/api";
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

beforeEach(() => {
  api.listCollections = async () => [{ name: "notes" }, { name: "docs" }];
});

afterEach(() => {
  Object.assign(api, originalApi);
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
    expect(view.getByText("Enter a query above to search across all collections or narrow the scope with the collection filter.")).toBeTruthy();
  });

  test("shows collection loading state before collections are available", async () => {
    api.listCollections =
      () => new Promise<Array<{ name: string }>>(() => undefined);

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
});
