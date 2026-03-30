import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type ComponentProps,
} from "react";
import { Link, useNavigate, useParams } from "react-router";
import Markdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import "katex/dist/katex.min.css";
import { api, buildDocumentTabHref } from "../lib/api";
import type { Collection, DocumentListItem } from "../lib/api";
import "./Documents.css";

interface ParsedDocument {
  frontmatter: Record<string, string> | null;
  body: string;
}

function parseFrontmatter(content: string): ParsedDocument {
  if (!content.startsWith("---\n") && !content.startsWith("---\r\n")) {
    return { frontmatter: null, body: content };
  }

  const endIdx = content.indexOf("\n---", 4);
  if (endIdx === -1) {
    return { frontmatter: null, body: content };
  }

  const raw = content.slice(4, endIdx);
  const body = content.slice(endIdx + 4).replace(/^\r?\n/, "");

  const fields: Record<string, string> = {};
  for (const line of raw.split("\n")) {
    const colonIdx = line.indexOf(":");
    if (colonIdx === -1) continue;
    const key = line.slice(0, colonIdx).trim();
    const value = line.slice(colonIdx + 1).trim();
    if (key) fields[key] = value;
  }

  return {
    frontmatter: Object.keys(fields).length > 0 ? fields : null,
    body,
  };
}

interface TreeNode {
  name: string;
  path: string;
  isDir: boolean;
  children: TreeNode[];
  doc?: DocumentListItem;
}

interface StatusMessage {
  tone: "loading" | "success" | "error";
  text: string;
}

const ACCEPTED_MARKDOWN = ".md,.markdown,.mdown,.mkd";
const MARKDOWN_FILE_RE = /\.(md|markdown|mdown|mkd)$/i;
const DOCUMENT_MARKDOWN_REMARK_PLUGINS = [remarkGfm, remarkMath];
const DOCUMENT_MARKDOWN_REHYPE_PLUGINS = [rehypeKatex];

function buildTree(docs: DocumentListItem[]): TreeNode[] {
  const root: TreeNode[] = [];

  for (const doc of docs) {
    const parts = doc.path.split("/");
    let level = root;
    let pathSoFar = "";

    for (let i = 0; i < parts.length; i += 1) {
      pathSoFar += `${i > 0 ? "/" : ""}${parts[i]}`;
      const isLast = i === parts.length - 1;
      let existing = level.find((node) => node.name === parts[i]);

      if (!existing) {
        existing = {
          name: parts[i],
          path: pathSoFar,
          isDir: !isLast,
          children: [],
          doc: isLast ? doc : undefined,
        };
        level.push(existing);
      }

      level = existing.children;
    }
  }

  return root;
}

function formatFileCount(count: number) {
  return `${count} file${count === 1 ? "" : "s"}`;
}

function isMarkdownFile(file: File) {
  return MARKDOWN_FILE_RE.test(file.name) || file.type === "text/markdown";
}

export default function Documents() {
  const navigate = useNavigate();
  const params = useParams<{ collection?: string; "*"?: string }>();
  const [collections, setCollections] = useState<Collection[]>([]);
  const [docs, setDocs] = useState<Record<string, DocumentListItem[]>>({});
  const [loadingColls, setLoadingColls] = useState<Set<string>>(new Set());
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [selectedDoc, setSelectedDoc] = useState<{
    collection: string;
    path: string;
    title: string;
    doc_id: string;
  } | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [newCollName, setNewCollName] = useState("");
  const [dragOver, setDragOver] = useState<string | null>(null);
  const [ingesting, setIngesting] = useState(false);
  const [deletingDoc, setDeletingDoc] = useState(false);
  const [status, setStatus] = useState<StatusMessage | null>(null);
  const [uploadCollection, setUploadCollection] = useState<string | null>(null);
  const [confirmDeleteDoc, setConfirmDeleteDoc] = useState<string | null>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);

  const loadCollections = useCallback(async () => {
    try {
      const colls = await api.listCollections();
      setCollections(colls);
    } catch (error) {
      setStatus({
        tone: "error",
        text: error instanceof Error ? error.message : "Failed to load collections.",
      });
    }
  }, []);

  useEffect(() => {
    void loadCollections();
  }, [loadCollections]);

  useEffect(() => {
    if (!status || status.tone === "loading") {
      return undefined;
    }

    const timeoutId = window.setTimeout(() => setStatus(null), 4000);
    return () => window.clearTimeout(timeoutId);
  }, [status]);

  const loadDocs = useCallback(async (collection: string) => {
    setLoadingColls((prev) => new Set(prev).add(collection));

    try {
      const items = await api.listDocuments(collection);
      setDocs((prev) => ({ ...prev, [collection]: items }));
    } catch (error) {
      setStatus({
        tone: "error",
        text:
          error instanceof Error ? error.message : `Failed to load documents for ${collection}.`,
      });
    } finally {
      setLoadingColls((prev) => {
        const next = new Set(prev);
        next.delete(collection);
        return next;
      });
    }
  }, []);

  const toggleCollection = (name: string) => {
    const opening = !expanded.has(name);

    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });

    if (opening && !docs[name]) {
      void loadDocs(name);
    }
  };

  const toggleDir = (key: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const openDocument = useCallback(
    async (
      collection: string,
      path: string,
      fallback?: Pick<DocumentListItem, "title" | "doc_id">,
    ) => {
      setSelectedDoc({
        collection,
        path,
        title: fallback?.title ?? path,
        doc_id: fallback?.doc_id ?? "Loading…",
      });
      setConfirmDeleteDoc(null);
      setPreview(null);
      navigate(buildDocumentTabHref(collection, path), { replace: true });

      try {
        const full = await api.getDocument(collection, path);
        setSelectedDoc({
          collection: full.collection,
          path: full.path,
          title: full.title,
          doc_id: full.doc_id,
        });
        setPreview(full.content || "_No content stored._");
      } catch (error) {
        setPreview(
          error instanceof Error
            ? `_Failed to load document: ${error.message}_`
            : "_Failed to load document._",
        );
      }
    },
    [navigate],
  );

  const selectFile = useCallback(
    async (collection: string, doc: DocumentListItem) => {
      await openDocument(collection, doc.path, doc);
    },
    [openDocument],
  );

  useEffect(() => {
    const collection = params.collection?.trim();
    const path = params["*"]?.trim();

    if (!collection || !path) {
      return;
    }

    setExpanded((prev) => new Set(prev).add(collection));
    if (!docs[collection]) {
      void loadDocs(collection);
    }

    if (selectedDoc?.collection === collection && selectedDoc.path === path) {
      return;
    }

    const listedDoc = docs[collection]?.find((doc) => doc.path === path);
    if (collection in docs && !listedDoc) {
      setSelectedDoc(null);
      setPreview(null);
      return;
    }

    void openDocument(collection, path, listedDoc);
  }, [docs, loadDocs, openDocument, params, selectedDoc]);

  const ingestFiles = useCallback(
    async (collection: string, files: File[]) => {
      if (files.length === 0) {
        return;
      }

      const unsupported = files.filter((file) => !isMarkdownFile(file));
      if (unsupported.length > 0) {
        setStatus({
          tone: "error",
          text: "Only Markdown files are supported right now.",
        });
        return;
      }

      setIngesting(true);
      setStatus({
        tone: "loading",
        text: `Ingesting ${formatFileCount(files.length)} into ${collection}…`,
      });

      try {
        const toIngest = await Promise.all(
          files.map(async (file) => ({
            path: file.webkitRelativePath || file.name,
            content: await file.text(),
            content_type: "text/markdown" as const,
          })),
        );

        const response = await api.ingestDocuments(collection, toIngest);
        setExpanded((prev) => new Set(prev).add(collection));
        await loadDocs(collection);

        setStatus({
          tone: "success",
          text: `Ingested ${formatFileCount(response.ingested)} into ${collection}.`,
        });
      } catch (error) {
        setStatus({
          tone: "error",
          text:
            error instanceof Error ? error.message : "Something went wrong while ingesting files.",
        });
      } finally {
        setIngesting(false);
      }
    },
    [loadDocs],
  );

  const handleCreateCollection = async () => {
    const name = newCollName.trim();
    if (!name) {
      return;
    }

    try {
      await api.createCollection(name);
      setNewCollName("");
      setExpanded((prev) => new Set(prev).add(name));
      setDocs((prev) => ({ ...prev, [name]: [] }));
      setStatus({ tone: "success", text: `Created collection ${name}.` });
      await loadCollections();
    } catch (error) {
      setStatus({
        tone: "error",
        text: error instanceof Error ? error.message : "Could not create collection.",
      });
    }
  };

  const openUploadPicker = (collection: string) => {
    if (ingesting) {
      return;
    }

    setUploadCollection(collection);
    uploadInputRef.current?.click();
  };

  const handleUploadInputChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const collection = uploadCollection;
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";

    if (!collection || files.length === 0) {
      return;
    }

    await ingestFiles(collection, files);
  };

  const onDragOver = (event: DragEvent<HTMLDivElement>, collection: string) => {
    event.preventDefault();
    event.stopPropagation();
    setDragOver(collection);
  };

  const onDragLeave = (event: DragEvent<HTMLDivElement>, collection: string) => {
    event.preventDefault();
    event.stopPropagation();

    const nextTarget = event.relatedTarget;
    if (nextTarget instanceof Node && event.currentTarget.contains(nextTarget)) {
      return;
    }

    setDragOver((current) => (current === collection ? null : current));
  };

  const onDrop = async (event: DragEvent<HTMLDivElement>, collection: string) => {
    event.preventDefault();
    event.stopPropagation();
    setDragOver(null);

    await ingestFiles(collection, Array.from(event.dataTransfer.files));
  };

  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  const handleDeleteDocument = useCallback(
    async (collection: string, doc: DocumentListItem) => {
      setDeletingDoc(true);

      try {
        await api.deleteDocument(collection, doc.path);
        setConfirmDeleteDoc(null);
        setDocs((prev) => ({
          ...prev,
          [collection]: (prev[collection] ?? []).filter((item) => item.path !== doc.path),
        }));
        if (selectedDoc?.collection === collection && selectedDoc.path === doc.path) {
          setSelectedDoc(null);
          setPreview(null);
          navigate(`/documents/${encodeURIComponent(collection)}`, { replace: true });
        }
        setStatus({ tone: "success", text: `Deleted ${doc.title}.` });
        await loadDocs(collection);
      } catch (error) {
        setStatus({
          tone: "error",
          text: error instanceof Error ? error.message : "Could not delete document.",
        });
      } finally {
        setDeletingDoc(false);
      }
    },
    [loadDocs, navigate, selectedDoc],
  );

  const handleDeleteCollection = async (name: string) => {
    try {
      await api.deleteCollection(name);
      setConfirmDelete(null);

      // Clean up local state.
      setDocs((prev) => {
        const next = { ...prev };
        delete next[name];
        return next;
      });
      setExpanded((prev) => {
        const next = new Set(prev);
        next.delete(name);
        return next;
      });
      if (selectedDoc?.collection === name) {
        setSelectedDoc(null);
        setPreview(null);
        setConfirmDeleteDoc(null);
      }

      setStatus({ tone: "success", text: `Deleted collection ${name}.` });
      await loadCollections();
    } catch (error) {
      setStatus({
        tone: "error",
        text: error instanceof Error ? error.message : "Could not delete collection.",
      });
    }
  };

  const renderTree = (nodes: TreeNode[], collection: string, depth: number) => {
    const dirs = nodes.filter((node) => node.isDir).sort((a, b) => a.name.localeCompare(b.name));
    const files = nodes.filter((node) => !node.isDir).sort((a, b) => a.name.localeCompare(b.name));

    return (
      <>
        {dirs.map((node) => {
          const key = `${collection}/${node.path}`;
          const isExpanded = expanded.has(key);

          return (
            <div key={key}>
              <button
                type="button"
                className="tree-item tree-dir"
                style={{ paddingLeft: `${12 + depth * 16}px` }}
                onClick={() => toggleDir(key)}
                aria-expanded={isExpanded}
                title={node.path}
              >
                <span className={`tree-chevron${isExpanded ? " open" : ""}`}>
                  <ChevronIcon />
                </span>
                <FolderIcon />
                <span className="tree-name">{node.name}</span>
              </button>

              {isExpanded && <div>{renderTree(node.children, collection, depth + 1)}</div>}
            </div>
          );
        })}

        {files.map((node) => {
          const isSelected =
            selectedDoc?.collection === collection && selectedDoc.path === node.path;
          const deleteKey = `${collection}/${node.path}`;
          const isConfirmingDelete = confirmDeleteDoc === deleteKey;

          return (
            <div
              key={`${collection}/${node.path}`}
              className={`tree-file-row${isSelected ? " selected" : ""}`}
            >
              <button
                type="button"
                className={`tree-item tree-file${isSelected ? " selected" : ""}`}
                style={{ paddingLeft: `${12 + depth * 16}px` }}
                onClick={() => node.doc && void selectFile(collection, node.doc)}
                title={node.path}
              >
                <FileIcon />
                <span className="tree-name">{node.name}</span>
              </button>

              {isConfirmingDelete ? (
                <div className="tree-file-confirm">
                  <button
                    type="button"
                    className="tree-confirm-yes"
                    onClick={() => node.doc && void handleDeleteDocument(collection, node.doc)}
                    disabled={deletingDoc}
                    title="Confirm delete"
                  >
                    {deletingDoc ? "Deleting…" : "Delete"}
                  </button>
                  <button
                    type="button"
                    className="tree-confirm-no"
                    onClick={() => setConfirmDeleteDoc(null)}
                    disabled={deletingDoc}
                    title="Cancel"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  className="tree-file-delete-btn"
                  onClick={(event) => {
                    event.stopPropagation();
                    setConfirmDeleteDoc(deleteKey);
                  }}
                  aria-label={`Delete ${node.name}`}
                  title="Delete file"
                >
                  <TrashIcon />
                </button>
              )}
            </div>
          );
        })}
      </>
    );
  };

  return (
    <div className="documents-page">
      <input
        ref={uploadInputRef}
        className="sr-only"
        type="file"
        multiple
        accept={ACCEPTED_MARKDOWN}
        onChange={(event) => void handleUploadInputChange(event)}
      />

      {status && (
        <div
          className={`ingest-toast ${status.tone}`}
          role={status.tone === "error" ? "alert" : "status"}
          aria-live="polite"
        >
          {status.text}
        </div>
      )}

      <div className="file-manager">
        <section className="file-tree-panel" aria-labelledby="collections-heading">
          <div className="file-tree-header">
            <span id="collections-heading" className="file-tree-title">
              Collections
            </span>
            <form
              className="collection-add-form"
              onSubmit={(event) => {
                event.preventDefault();
                void handleCreateCollection();
              }}
            >
              <label className="sr-only" htmlFor="collection-name">
                New collection name
              </label>
              <input
                id="collection-name"
                type="text"
                placeholder="New collection"
                value={newCollName}
                onChange={(event) => setNewCollName(event.target.value)}
                className="collection-add-input"
              />
              <button type="submit" className="collection-add-btn" disabled={!newCollName.trim()}>
                +
              </button>
            </form>
          </div>

          <div className="file-tree" aria-label="Document collections">
            {collections.length === 0 && (
              <div className="tree-empty">
                <p>No collections yet.</p>
                <p>Create one to start indexing Markdown files.</p>
              </div>
            )}

            {collections.map((collection) => {
              const isExpanded = expanded.has(collection.name);
              const collDocs = docs[collection.name] ?? [];
              const tree = buildTree(collDocs);
              const isDragTarget = dragOver === collection.name;
              const isLoading = loadingColls.has(collection.name);
              const countLabel = collection.name in docs ? formatFileCount(collDocs.length) : null;

              return (
                <div
                  key={collection.name}
                  className={`tree-collection${isDragTarget ? " drag-over" : ""}`}
                  onDragOver={(event) => onDragOver(event, collection.name)}
                  onDragLeave={(event) => onDragLeave(event, collection.name)}
                  onDrop={(event) => void onDrop(event, collection.name)}
                >
                  <div className="tree-collection-head">
                    <button
                      type="button"
                      className="tree-item tree-collection-header"
                      onClick={() => toggleCollection(collection.name)}
                      aria-expanded={isExpanded}
                      title={collection.name}
                    >
                      <span className={`tree-chevron${isExpanded ? " open" : ""}`}>
                        <ChevronIcon />
                      </span>
                      <FolderIcon />
                      <span className="tree-name">{collection.name}</span>
                      {(isLoading || countLabel) && (
                        <span className="tree-count">{isLoading ? "Loading…" : countLabel}</span>
                      )}
                    </button>

                    <button
                      type="button"
                      className="tree-upload-btn"
                      onClick={() => openUploadPicker(collection.name)}
                      aria-label={`Upload Markdown files to ${collection.name}`}
                      title={`Upload Markdown files to ${collection.name}`}
                      disabled={ingesting}
                    >
                      <UploadIcon />
                    </button>
                    {confirmDelete === collection.name ? (
                      <div className="tree-confirm-delete">
                        <button
                          type="button"
                          className="tree-confirm-yes"
                          onClick={() => void handleDeleteCollection(collection.name)}
                          title="Confirm delete"
                        >
                          Delete
                        </button>
                        <button
                          type="button"
                          className="tree-confirm-no"
                          onClick={() => setConfirmDelete(null)}
                          title="Cancel"
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button
                        type="button"
                        className="tree-delete-btn"
                        onClick={() => setConfirmDelete(collection.name)}
                        aria-label={`Delete collection ${collection.name}`}
                        title={`Delete collection ${collection.name}`}
                      >
                        <TrashIcon />
                      </button>
                    )}
                  </div>

                  {isExpanded && (
                    <div className="tree-children">
                      {isLoading ? (
                        <div className="tree-loading">Loading documents…</div>
                      ) : collDocs.length === 0 ? (
                        <div className="tree-empty-hint">
                          Drop Markdown files here or use upload.
                        </div>
                      ) : (
                        renderTree(tree, collection.name, 1)
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        <section className="preview-panel" aria-live="polite">
          {selectedDoc ? (
            <PreviewContent selectedDoc={selectedDoc} preview={preview} />
          ) : (
            <div className="preview-empty">
              <div className="preview-empty-icon">
                <FileIcon size={48} />
              </div>
              <h3>No document selected</h3>
              <p>
                Select a file from the tree to preview it, or upload Markdown files into any
                collection.
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function PreviewContent({
  selectedDoc,
  preview,
}: {
  selectedDoc: { collection: string; path: string; title: string; doc_id: string };
  preview: string | null;
}) {
  const permalink = buildDocumentTabHref(selectedDoc.collection, selectedDoc.path);
  const parsed = useMemo(() => (preview ? parseFrontmatter(preview) : null), [preview]);

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

function ChevronIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}

function FolderIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    </svg>
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

function UploadIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
  );
}
