import type { CSSProperties } from "react";
import { dracula } from "react-syntax-highlighter/dist/esm/styles/prism";

export const darkPrismTheme = dracula;

export const lightPrismTheme: Record<string, CSSProperties> = {
  'code[class*="language-"]': {
    color: "#1f1f1f",
    background: "#efeddc",
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
    color: "#1f1f1f",
    background: "#efeddc",
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
    color: "#6c664b",
    fontStyle: "italic",
  },
  prolog: {
    color: "#6c664b",
  },
  doctype: {
    color: "#6c664b",
  },
  cdata: {
    color: "#6c664b",
  },
  punctuation: {
    color: "#6c664b",
  },
  namespace: {
    opacity: 0.7,
  },
  property: {
    color: "#a3144d",
  },
  tag: {
    color: "#a3144d",
  },
  boolean: {
    color: "#a34d14",
  },
  number: {
    color: "#a34d14",
  },
  constant: {
    color: "#644ac9",
  },
  symbol: {
    color: "#644ac9",
  },
  deleted: {
    color: "#cb3a2a",
  },
  selector: {
    color: "#14710a",
  },
  "attr-name": {
    color: "#036a96",
  },
  string: {
    color: "#846e15",
  },
  char: {
    color: "#846e15",
  },
  builtin: {
    color: "#036a96",
  },
  inserted: {
    color: "#14710a",
  },
  operator: {
    color: "#6c664b",
  },
  entity: {
    color: "#036a96",
    cursor: "help",
  },
  url: {
    color: "#036a96",
  },
  variable: {
    color: "#1f1f1f",
  },
  atrule: {
    color: "#a3144d",
  },
  "attr-value": {
    color: "#846e15",
  },
  function: {
    color: "#14710a",
  },
  "class-name": {
    color: "#036a96",
  },
  keyword: {
    color: "#a3144d",
    fontWeight: 600,
  },
  regex: {
    color: "#036a96",
  },
  important: {
    color: "#a3144d",
    fontWeight: 700,
  },
  bold: {
    fontWeight: 700,
  },
  italic: {
    fontStyle: "italic",
  },
};
