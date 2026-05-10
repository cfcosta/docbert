import { describe, expect, test } from "bun:test";

import { parseDocumentFrontmatter } from "./document-frontmatter";

describe("document-frontmatter", () => {
  test("parse_frontmatter_returns_plain_body_when_opening_fence_is_missing", () => {
    expect(parseDocumentFrontmatter("# Hello\n\nBody")).toEqual({
      frontmatter: null,
      body: "# Hello\n\nBody",
    });
  });

  test("parse_frontmatter_extracts_fields_and_strips_one_leading_newline_from_body", () => {
    expect(
      parseDocumentFrontmatter("---\ntitle: Hello\nauthor: Casey\n---\n\n# Heading\nBody"),
    ).toEqual({
      frontmatter: {
        title: "Hello",
        author: "Casey",
      },
      body: "\n# Heading\nBody",
    });
  });

  test("parse_frontmatter_uses_first_colon_and_ignores_lines_without_colons", () => {
    expect(
      parseDocumentFrontmatter("---\ntitle: a:b\nignored line\nsummary: keeps:tail\n---\nBody"),
    ).toEqual({
      frontmatter: {
        title: "a:b",
        summary: "keeps:tail",
      },
      body: "Body",
    });
  });

  test("parse_frontmatter_treats_missing_closing_delimiter_as_plain_body", () => {
    const content = "---\ntitle: Hello\n# Heading\nBody";

    expect(parseDocumentFrontmatter(content)).toEqual({
      frontmatter: null,
      body: content,
    });
  });
});
