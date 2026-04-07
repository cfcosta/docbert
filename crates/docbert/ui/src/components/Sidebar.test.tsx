import "../test/setup";

import { describe, expect, test } from "bun:test";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router";

import Sidebar from "./Sidebar";

describe("Sidebar", () => {
  test("shows_docbert_branding_and_primary_navigation", () => {
    const view = render(
      <MemoryRouter initialEntries={["/documents"]}>
        <Sidebar />
      </MemoryRouter>,
    );

    expect(view.getByRole("heading", { name: "docbert" })).toBeTruthy();
    expect(view.getByRole("link", { name: /documents/i })).toBeTruthy();
    expect(view.getByRole("link", { name: /chat/i })).toBeTruthy();
    expect(view.getByRole("link", { name: /search/i })).toBeTruthy();
    expect(view.getByRole("link", { name: /settings/i })).toBeTruthy();
  });

  test("marks_search_link_active_on_search_route", () => {
    const view = render(
      <MemoryRouter initialEntries={["/search"]}>
        <Sidebar />
      </MemoryRouter>,
    );

    expect(view.getByRole("link", { name: /search/i }).className).toContain("active");
  });
});
