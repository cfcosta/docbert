import type { ComponentProps } from "react";
import { Link } from "react-router";
import Markdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

import { buildDocumentTabHref } from "../lib/api";
import type { SelectedDocumentSummary } from "./documents-tree";

interface ParsedDocument {
  frontmatter: Record<string, string> | null;
  body: string;
}

const DOCUMENT_MARKDOWN_REMARK_PLUGINS = [remarkGfm, remarkMath];
const DOCUMENT_MARKDOWN_REHYPE_PLUGINS = [rehypeKatex];

function parseFrontmatter(content: string): ParsedDocument {
  if (!content.startsWith("---\n") && !content.startsWith("---\r\n")) {
    return { frontmatter: null, body: content };
  }

  const endIndex = content.indexOf("\n---", 4);
  if (endIndex === -1) {
    return { frontmatter: null, body: content };
  }

  const raw = content.slice(4, endIndex);
  const body = content.slice(endIndex + 4).replace(/^\r?\n/, "");

  const fields: Record<string, string> = {};
  for (const line of raw.split("\n")) {
    const colonIndex = line.indexOf(":");
    if (colonIndex === -1) {
      continue;
    }
    const key = line.slice(0, colonIndex).trim();
    const value = line.slice(colonIndex + 1).trim();
    if (key) {
      fields[key] = value;
    }
  }

  return {
    frontmatter: Object.keys(fields).length > 0 ? fields : null,
    body,
  };
}

export default function DocumentPreview({
  selectedDoc,
  preview,
}: {
  selectedDoc: SelectedDocumentSummary | null;
  preview: string | null;
}) {
  if (!selectedDoc) {
    return (
      <div className="preview-empty">
        <div className="preview-empty-icon">
          <FileIcon size={48} />
        </div>
        <h3>No document selected</h3>
        <p>
          Select a file from the tree to preview it, or upload Markdown files into any collection.
        </p>
      </div>
    );
  }

  const permalink = buildDocumentTabHref(selectedDoc.collection, selectedDoc.path);
  const parsed = preview ? parseFrontmatter(preview) : null;

  return (
    <>
      <div className="preview-header">
        <div className="preview-title-row">
          <div>
            <p className="preview-kicker">{selectedDoc.collection}</p>
            <h2 className="preview-title">{selectedDoc.title}</h2>
          </div>
          <Link className="preview-permalink" to={permalink}>
            Permalink
          </Link>
        </div>
        <div className="preview-meta">
          <code>{selectedDoc.doc_id}</code>
          <span className="preview-path" title={selectedDoc.path}>
            {selectedDoc.path}
          </span>
        </div>
        {parsed?.frontmatter && (
          <dl className="preview-frontmatter">
            {Object.entries(parsed.frontmatter)
              .filter(([, value]) => value.length > 0)
              .map(([key, value]) => (
                <div key={key} className="preview-fm-field">
                  <dt>{key}</dt>
                  <dd>{value}</dd>
                </div>
              ))}
          </dl>
        )}
      </div>

      <div className="preview-body">
        {preview === null ? (
          <div className="preview-loading">Loading document…</div>
        ) : (
          <div className="preview-content">
            <Markdown
              remarkPlugins={DOCUMENT_MARKDOWN_REMARK_PLUGINS}
              rehypePlugins={DOCUMENT_MARKDOWN_REHYPE_PLUGINS}
              components={{ code: CodeBlock }}
            >
              {parsed?.body ?? preview}
            </Markdown>
          </div>
        )}
      </div>
    </>
  );
}

function CodeBlock({ className, children, ...props }: ComponentProps<"code">) {
  const match = /language-(\w+)/.exec(className ?? "");
  const code = String(children).replace(/\n$/, "");

  if (!match) {
    return (
      <code className={className} {...props}>
        {children}
      </code>
    );
  }

  return (
    <SyntaxHighlighter
      style={oneDark}
      language={match[1]}
      PreTag="div"
      customStyle={{ margin: 0, borderRadius: "6px", fontSize: "0.85em" }}
    >
      {code}
    </SyntaxHighlighter>
  );
}

function FileIcon({ size = 16 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  );
}
