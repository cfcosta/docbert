import type { ComponentProps } from "react";
import { PrismAsyncLight as SyntaxHighlighter } from "react-syntax-highlighter";

import { darkPrismTheme, lightPrismTheme } from "../lib/prism-theme";
import { usePrefersDarkMode } from "../lib/use-prefers-dark";
import MermaidDiagram from "./MermaidDiagram";

export default function MarkdownCodeBlock({
  className,
  children,
  ...props
}: ComponentProps<"code">) {
  const prefersDark = usePrefersDarkMode();
  const match = /language-(\w+)/.exec(className ?? "");
  const code = String(children).replace(/\n$/, "");

  if (!match) {
    return (
      <code className={className} {...props}>
        {children}
      </code>
    );
  }

  const language = match[1];

  if (language === "mermaid") {
    return <MermaidDiagram code={code} />;
  }

  return (
    <SyntaxHighlighter
      style={prefersDark ? darkPrismTheme : lightPrismTheme}
      language={language}
      PreTag="div"
      customStyle={{ margin: 0, borderRadius: "6px", fontSize: "0.85em" }}
    >
      {code}
    </SyntaxHighlighter>
  );
}
