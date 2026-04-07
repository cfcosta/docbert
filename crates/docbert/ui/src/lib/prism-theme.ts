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
  foreground: "#cdd6f4",
  background: "#181825",
  comment: "#6c7086",
  red: "#f38ba8",
  orange: "#fab387",
  yellow: "#f9e2af",
  green: "#a6e3a1",
  cyan: "#74c7ec",
  purple: "#cba6f7",
  pink: "#f5c2e7",
});

export const lightPrismTheme = createPrismTheme({
  foreground: "#4c4f69",
  background: "#e6e9ef",
  comment: "#9ca0b0",
  red: "#d20f39",
  orange: "#fe640b",
  yellow: "#df8e1d",
  green: "#40a02b",
  cyan: "#209fb5",
  purple: "#8839ef",
  pink: "#ea76cb",
});
