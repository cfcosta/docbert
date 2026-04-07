import "../test/setup";

import { describe, expect, test } from "bun:test";
import { render } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";

import App from "../App";
import Search from "./Search";

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

describe("Search page", () => {
  test("renders under the main layout with placeholder pre-search text", () => {
    const view = renderSearchRoute();

    expect(view.getByRole("heading", { name: "Search" })).toBeTruthy();
    expect(view.getByText("Find documents across your indexed collections.")).toBeTruthy();
    expect(
      view.getByText("Search will appear here once query, collection, and mode controls are connected."),
    ).toBeTruthy();
    expect(view.container.querySelector("main.main-content")).toBeTruthy();
  });
});
