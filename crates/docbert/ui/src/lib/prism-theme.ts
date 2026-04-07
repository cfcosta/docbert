import type { CSSProperties } from "react";

const createPrismTheme = ({
  foreground,
  background,
  comment,
  red,
  orange,
  yellow,
  green,
  cyan,
  purple,
  pink,
}: {
  foreground: string;
  background: string;
  comment: string;
  red: string;
  orange: string;
  yellow: string;
  green: string;
  cyan: string;
  purple: string;
  pink: string;
}): Record<string, CSSProperties> => ({
  'code[class*="language-"]': {
    color: foreground,
    background,
    fontFamily: "var(--mono)",
    fontSize: "0.85em",
    textAlign: "left",
    whiteSpace: "pre",
    wordSpacing: "normal",
    wordBreak: "normal",
    overflowWrap: "normal",
    lineHeight: "1.5",
    tabSize: "2",
    hyphens: "none",
  },
  'pre[class*="language-"]': {
    color: foreground,
    background,
    fontFamily: "var(--mono)",
    fontSize: "0.85em",
    textAlign: "left",
    whiteSpace: "pre",
    wordSpacing: "normal",
    wordBreak: "normal",
    overflowWrap: "normal",
    lineHeight: "1.5",
    tabSize: "2",
    hyphens: "none",
    padding: "0",
    margin: "0",
    overflow: "auto",
  },
  comment: {
    color: comment,
    fontStyle: "italic",
  },
  prolog: {
    color: comment,
  },
  doctype: {
    color: comment,
  },
  cdata: {
    color: comment,
  },
  punctuation: {
    color: comment,
  },
  namespace: {
    opacity: 0.7,
  },
  property: {
    color: pink,
  },
  tag: {
    color: pink,
  },
  boolean: {
    color: orange,
  },
  number: {
    color: orange,
  },
  constant: {
    color: purple,
  },
  symbol: {
    color: purple,
  },
  deleted: {
    color: red,
  },
  selector: {
    color: green,
  },
  "attr-name": {
    color: cyan,
  },
  string: {
    color: yellow,
  },
  char: {
    color: yellow,
  },
  builtin: {
    color: cyan,
  },
  inserted: {
    color: green,
  },
  operator: {
    color: comment,
  },
  entity: {
    color: cyan,
    cursor: "help",
  },
  url: {
    color: cyan,
  },
  variable: {
    color: foreground,
  },
  atrule: {
    color: pink,
  },
  "attr-value": {
    color: yellow,
  },
  function: {
    color: green,
  },
  "class-name": {
    color: cyan,
  },
  keyword: {
    color: pink,
    fontWeight: 600,
  },
  regex: {
    color: cyan,
  },
  important: {
    color: pink,
    fontWeight: 700,
  },
  bold: {
    fontWeight: 700,
  },
  italic: {
    fontStyle: "italic",
  },
});

export const darkPrismTheme = createPrismTheme({
  foreground: "#f8f8f2",
  background: "#282a36",
  comment: "#6272a4",
  red: "#ff5555",
  orange: "#ffb86c",
  yellow: "#f1fa8c",
  green: "#50fa7b",
  cyan: "#8be9fd",
  purple: "#bd93f9",
  pink: "#ff79c6",
});

export const lightPrismTheme = createPrismTheme({
  foreground: "#1f1f1f",
  background: "#fffbeb",
  comment: "#6c664b",
  red: "#cb3a2a",
  orange: "#a34d14",
  yellow: "#846e15",
  green: "#14710a",
  cyan: "#036a96",
  purple: "#644ac9",
  pink: "#a3144d",
});
