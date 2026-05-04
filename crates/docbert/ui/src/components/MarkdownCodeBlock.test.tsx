import "../test/setup";

import { describe, expect, test } from "bun:test";
import { render } from "@testing-library/react";

import MarkdownCodeBlock from "./MarkdownCodeBlock";

describe("MarkdownCodeBlock", () => {
  test("renders an inline code element when no language is set", () => {
    const view = render(<MarkdownCodeBlock>{"const x = 1;"}</MarkdownCodeBlock>);
    const codeNode = view.container.querySelector("code");

    expect(codeNode).not.toBeNull();
    expect(codeNode?.textContent).toBe("const x = 1;");
  });

  test("routes language-mermaid to the MermaidDiagram component", () => {
    const view = render(
      <MarkdownCodeBlock className="language-mermaid">
        {"graph TD;\nA-->B;\n"}
      </MarkdownCodeBlock>,
    );

    const mermaidContainer = view.container.querySelector(".mermaid-diagram");
    expect(mermaidContainer).not.toBeNull();
  });

  test("routes other languages to the syntax highlighter", () => {
    const view = render(
      <MarkdownCodeBlock className="language-typescript">
        {"const x: number = 1;\n"}
      </MarkdownCodeBlock>,
    );

    const mermaidContainer = view.container.querySelector(".mermaid-diagram");
    expect(mermaidContainer).toBeNull();

    const highlightedRoot = view.container.querySelector('[class*="language-"]');
    expect(highlightedRoot).not.toBeNull();
  });
});
