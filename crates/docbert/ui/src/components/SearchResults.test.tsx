import "../test/setup";

import { describe, expect, test } from "bun:test";
import { render } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";

import SearchResults from "./SearchResults";
import type { SearchResult } from "../lib/api";

function renderSearchResults(results: SearchResult[]) {
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <Routes>
        <Route path="/" element={<SearchResults results={results} />} />
        <Route path="/documents/:collection/*" element={<div>Document route</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("SearchResults", () => {
  test("renders standalone result details and excerpt ranges", () => {
    const view = renderSearchResults([
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

  test("renders no results for an empty result list", () => {
    const view = renderSearchResults([]);

    expect(view.getByText("No results")).toBeTruthy();
  });

  test("generates document hrefs for excerpt links", () => {
    const view = renderSearchResults([
      {
        rank: 1,
        score: 0.914,
        doc_id: "#abc123",
        collection: "notes",
        path: "nested/rust.md",
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

    const excerptLink = view.getByRole("link", { name: /10\s*Rust ownership keeps memory safe\./i });
    expect(excerptLink.getAttribute("href")).toBe("/documents/notes/nested/rust.md");
  });
});
