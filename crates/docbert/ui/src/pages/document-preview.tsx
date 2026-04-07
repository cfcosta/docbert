import {
  Children,
  useEffect,
  useMemo,
  useState,
  type ComponentProps,
  type ReactNode,
} from "react";
import { Link } from "react-router";
import Markdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";

import { buildDocumentTabHref, type DocumentListItem } from "../lib/api";
import { resolveObsidianLink, type ResolvedObsidianLinkTarget } from "../lib/obsidian-links";
import {
  buildBlockTargetId,
  buildHeadingTargetIds,
  extractTrailingBlockReference,
  previewTargetIdFromFragment,
} from "../lib/document-preview-targets";
import { darkPrismTheme, lightPrismTheme } from "../lib/prism-theme";
import { parseDocumentFrontmatter } from "./document-frontmatter";
import type { SelectedDocumentSummary } from "./documents-tree";

const DOCUMENT_MARKDOWN_REMARK_PLUGINS = [remarkGfm, remarkMath];
const DOCUMENT_MARKDOWN_REHYPE_PLUGINS = [rehypeKatex];
const RESOLVED_LINK_PREFIX = "/__docbert_resolved__/";

export interface ResolvedDocumentTarget {
  collection: string;
  path: string;
  fragment: string | null;
}

export default function DocumentPreview({
  selectedDoc,
  preview,
  resolverDocuments = [],
  activeFragment = null,
  onOpenResolvedDocument,
}: {
  selectedDoc: SelectedDocumentSummary | null;
  preview: string | null;
  resolverDocuments?: Pick<DocumentListItem, "path">[];
  activeFragment?: string | null;
  onOpenResolvedDocument?: (target: ResolvedDocumentTarget) => void;
}) {
  if (!selectedDoc) {
    return (
      <div className="preview-empty">
        <div className="preview-empty-icon">
          <FileIcon size={48} />
        </div>
        <h3>No document selected</h3>
        <p>
          Select a file from the tree to preview it, or upload Markdown files into any docbert
          collection.
        </p>
      </div>
    );
  }

  const permalink = buildDocumentTabHref(selectedDoc.collection, selectedDoc.path);
  const parsed = preview ? parseDocumentFrontmatter(preview) : null;
  const body = parsed?.body ?? preview ?? "";
  const headingIds = useMemo(() => buildHeadingTargetIds(extractHeadingTexts(body)), [body]);
  let headingIndex = 0;
  const markdownSource = useMemo(
    () =>
      rewriteObsidianLinksToMarkdown(body, {
        currentDoc: selectedDoc,
        documents: resolverDocuments,
      }),
    [body, resolverDocuments, selectedDoc],
  );

  useEffect(() => {
    if (!preview || !activeFragment) {
      return;
    }

    const targetId = previewTargetIdFromFragment(activeFragment);
    if (!targetId) {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      document.getElementById(targetId)?.scrollIntoView();
    }, 0);

    return () => window.clearTimeout(timeoutId);
  }, [activeFragment, preview, selectedDoc.path]);

  const renderHeading = (Tag: "h1" | "h2" | "h3" | "h4" | "h5" | "h6") => {
    return function Heading({ children, ...props }: ComponentProps<typeof Tag>) {
      const id = headingIds[headingIndex];
      headingIndex += 1;
      return <Tag id={id} {...props}>{children}</Tag>;
    };
  };

  const components = {
    code: CodeBlock,
    a: ({ href, children, ...props }: ComponentProps<"a">) => {
      const resolvedTarget = href ? parseResolvedDocumentHref(href) : null;
      const targetId = href?.startsWith("#") ? previewTargetIdFromFragment(href) : null;

      if (resolvedTarget && onOpenResolvedDocument) {
        return (
          <a
            href={href}
            {...props}
            onClick={(event) => {
              event.preventDefault();
              onOpenResolvedDocument(resolvedTarget);
            }}
          >
            {children}
          </a>
        );
      }

      if (targetId) {
        return (
          <a
            href={href}
            {...props}
            onClick={(event) => {
              event.preventDefault();
              document.getElementById(targetId)?.scrollIntoView();
            }}
          >
            {children}
          </a>
        );
      }

      return (
        <a href={href} {...props}>
          {children}
        </a>
      );
    },
    p: ({ children, ...props }: ComponentProps<"p">) => {
      const text = textContent(children);
      const blockRef = extractTrailingBlockReference(text);
      const content =
        typeof children === "string" && blockRef
          ? children.replace(new RegExp(`(?:^|\\s)\\^${escapeRegExp(blockRef)}\\s*$`), "").trimEnd()
          : children;

      return (
        <p id={blockRef ? buildBlockTargetId(blockRef) : undefined} {...props}>
          {content}
        </p>
      );
    },
    h1: renderHeading("h1"),
    h2: renderHeading("h2"),
    h3: renderHeading("h3"),
    h4: renderHeading("h4"),
    h5: renderHeading("h5"),
    h6: renderHeading("h6"),
  } satisfies ComponentProps<typeof Markdown>["components"];

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
              components={components}
            >
              {markdownSource}
            </Markdown>
          </div>
        )}
      </div>
    </>
  );
}

function rewriteObsidianLinksToMarkdown(
  source: string,
  options: {
    currentDoc: SelectedDocumentSummary;
    documents: Pick<DocumentListItem, "path">[];
  },
) {
  return source.replace(/\[\[[^\]]+\]\]/g, (raw) => {
    const resolved = resolveObsidianLink(raw, {
      collection: options.currentDoc.collection,
      currentPath: options.currentDoc.path,
      documents: options.documents,
    });
    const label = linkLabel(raw);

    if (!resolved) {
      return label;
    }

    if (resolved.path === options.currentDoc.path && resolved.fragment) {
      return `[${escapeMarkdownLabel(label)}](#${encodeURIComponent(resolved.fragment)})`;
    }

    if (resolved.path !== options.currentDoc.path) {
      return `[${escapeMarkdownLabel(label)}](${buildResolvedDocumentHref(resolved)})`;
    }

    return label;
  });
}

function linkLabel(raw: string) {
  const resolved = resolveObsidianLabel(raw);
  return resolved || raw;
}

function resolveObsidianLabel(raw: string) {
  const inner = raw.slice(2, -2);
  const [destination, alias] = inner.split("|");
  if (alias?.trim()) {
    return alias.trim();
  }

  const target = destination?.trim() ?? "";
  if (!target) {
    return raw;
  }

  if (target.startsWith("#")) {
    return target.slice(1);
  }

  const hashIndex = target.indexOf("#");
  return hashIndex === -1 ? target : target.slice(0, hashIndex);
}

function buildResolvedDocumentHref(target: ResolvedObsidianLinkTarget) {
  return `${RESOLVED_LINK_PREFIX}${encodeURIComponent(
    JSON.stringify({
      collection: target.collection,
      path: target.path,
      fragment: target.fragment,
    } satisfies ResolvedDocumentTarget),
  )}`;
}

function parseResolvedDocumentHref(href: string): ResolvedDocumentTarget | null {
  if (!href.startsWith(RESOLVED_LINK_PREFIX)) {
    return null;
  }

  try {
    return JSON.parse(decodeURIComponent(href.slice(RESOLVED_LINK_PREFIX.length)));
  } catch {
    return null;
  }
}

function extractHeadingTexts(markdown: string) {
  return markdown
    .split(/\r?\n/)
    .map((line) => line.match(/^#{1,6}\s+(.*)$/)?.[1]?.trim() ?? null)
    .filter((value): value is string => !!value);
}

function textContent(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") {
    return String(node);
  }

  if (Array.isArray(node)) {
    return node.map((child) => textContent(child)).join("");
  }

  if (!node || typeof node !== "object") {
    return "";
  }

  return textContent((node as { props?: { children?: ReactNode } }).props?.children ?? Children.toArray(node));
}

function escapeMarkdownLabel(text: string) {
  return text.replace(/[\[\]\\]/g, "\\$&");
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function CodeBlock({ className, children, ...props }: ComponentProps<"code">) {
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

  return (
    <SyntaxHighlighter
      style={prefersDark ? darkPrismTheme : lightPrismTheme}
      language={match[1]}
      PreTag="div"
      customStyle={{ margin: 0, borderRadius: "6px", fontSize: "0.85em" }}
    >
      {code}
    </SyntaxHighlighter>
  );
}

function usePrefersDarkMode() {
  const [prefersDark, setPrefersDark] = useState(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return false;
    }

    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return undefined;
    }

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const updatePreference = (event?: MediaQueryListEvent) => {
      setPrefersDark(event?.matches ?? mediaQuery.matches);
    };

    updatePreference();
    mediaQuery.addEventListener("change", updatePreference);
    return () => mediaQuery.removeEventListener("change", updatePreference);
  }, []);

  return prefersDark;
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
