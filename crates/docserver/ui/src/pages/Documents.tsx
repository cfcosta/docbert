import { useEffect, useState, useCallback, type DragEvent } from "react";
import { api } from "../lib/api";
import type { Collection, DocumentListItem } from "../lib/api";
import "./Documents.css";

interface TreeNode {
  name: string;
  path: string;
  isDir: boolean;
  children: TreeNode[];
  doc?: DocumentListItem;
}

function buildTree(docs: DocumentListItem[]): TreeNode[] {
  const root: TreeNode[] = [];
  for (const doc of docs) {
    const parts = doc.path.split("/");
    let level = root;
    let pathSoFar = "";
    for (let i = 0; i < parts.length; i++) {
      pathSoFar += (i > 0 ? "/" : "") + parts[i];
      const isLast = i === parts.length - 1;
      let existing = level.find((n) => n.name === parts[i]);
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

export default function Documents() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [docs, setDocs] = useState<Record<string, DocumentListItem[]>>({});
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [selectedDoc, setSelectedDoc] = useState<{
    collection: string;
    path: string;
    title: string;
    doc_id: string;
  } | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [newCollName, setNewCollName] = useState("");

  // Drag-and-drop state
  const [dragOver, setDragOver] = useState<string | null>(null);
  const [ingesting, setIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState<string | null>(null);

  const loadCollections = useCallback(async () => {
    try {
      const colls = await api.listCollections();
      setCollections(colls);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    loadCollections();
  }, [loadCollections]);

  const loadDocs = useCallback(async (collection: string) => {
    try {
      const items = await api.listDocuments(collection);
      setDocs((prev) => ({ ...prev, [collection]: items }));
    } catch {
      /* ignore */
    }
  }, []);

  const toggleCollection = (name: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
        if (!docs[name]) loadDocs(name);
      }
      return next;
    });
  };

  const toggleDir = (key: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const selectFile = async (collection: string, doc: DocumentListItem) => {
    setSelectedDoc({ collection, path: doc.path, title: doc.title, doc_id: doc.doc_id });
    setPreview(null);
    try {
      const full = await api.getDocument(collection, doc.path);
      setPreview(full.content || "_No content stored._");
    } catch {
      setPreview("_Failed to load document._");
    }
  };

  const handleCreateCollection = async () => {
    const name = newCollName.trim();
    if (!name) return;
    try {
      await api.createCollection(name);
      setNewCollName("");
      await loadCollections();
      setExpanded((prev) => new Set(prev).add(name));
    } catch {
      /* ignore */
    }
  };

  // Drag-and-drop handlers
  const onDragOver = (e: DragEvent, collection: string) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(collection);
  };

  const onDragLeave = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(null);
  };

  const onDrop = async (e: DragEvent, collection: string) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(null);

    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;

    setIngesting(true);
    setIngestResult(null);

    try {
      const toIngest = await Promise.all(
        files.map(async (f) => ({
          path: f.name,
          content: await f.text(),
          content_type: "text/markdown" as const,
        })),
      );

      const res = await api.ingestDocuments(collection, toIngest);
      setIngestResult(
        `Ingested ${res.ingested} file${res.ingested === 1 ? "" : "s"} into ${collection}`,
      );
      await loadDocs(collection);

      // Auto-clear success message
      setTimeout(() => setIngestResult(null), 3000);
    } catch (err) {
      setIngestResult(`Error: ${err instanceof Error ? err.message : "ingestion failed"}`);
    } finally {
      setIngesting(false);
    }
  };

  const renderTree = (nodes: TreeNode[], collection: string, depth: number) => {
    const dirs = nodes.filter((n) => n.isDir).sort((a, b) => a.name.localeCompare(b.name));
    const files = nodes.filter((n) => !n.isDir).sort((a, b) => a.name.localeCompare(b.name));
    return (
      <>
        {dirs.map((node) => {
          const key = `${collection}/${node.path}`;
          const isExpanded = expanded.has(key);
          return (
            <div key={key}>
              <button
                className="tree-item tree-dir"
                style={{ paddingLeft: `${12 + depth * 16}px` }}
                onClick={() => toggleDir(key)}
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
            selectedDoc?.collection === collection && selectedDoc?.path === node.path;
          return (
            <button
              key={`${collection}/${node.path}`}
              className={`tree-item tree-file${isSelected ? " selected" : ""}`}
              style={{ paddingLeft: `${12 + depth * 16}px` }}
              onClick={() => node.doc && selectFile(collection, node.doc)}
            >
              <FileIcon />
              <span className="tree-name">{node.name}</span>
            </button>
          );
        })}
      </>
    );
  };

  return (
    <div className="documents-page">
      {/* Status bar */}
      {(ingesting || ingestResult) && (
        <div className={`ingest-toast${ingestResult?.startsWith("Error") ? " error" : ""}`}>
          {ingesting ? "Ingesting files..." : ingestResult}
        </div>
      )}

      <div className="file-manager">
        {/* File tree panel */}
        <div className="file-tree-panel">
          <div className="file-tree-header">
            <span className="file-tree-title">Files</span>
            <form
              className="collection-add-form"
              onSubmit={(e) => {
                e.preventDefault();
                handleCreateCollection();
              }}
            >
              <input
                type="text"
                placeholder="+ collection"
                value={newCollName}
                onChange={(e) => setNewCollName(e.target.value)}
                className="collection-add-input"
              />
            </form>
          </div>
          <div className="file-tree">
            {collections.length === 0 && (
              <div className="tree-empty">Create a collection to get started.</div>
            )}
            {collections.map((coll) => {
              const isExpanded = expanded.has(coll.name);
              const collDocs = docs[coll.name] ?? [];
              const tree = buildTree(collDocs);
              const isDragTarget = dragOver === coll.name;
              return (
                <div
                  key={coll.name}
                  className={`tree-collection${isDragTarget ? " drag-over" : ""}`}
                  onDragOver={(e) => onDragOver(e, coll.name)}
                  onDragLeave={onDragLeave}
                  onDrop={(e) => onDrop(e, coll.name)}
                >
                  <button
                    className="tree-item tree-collection-header"
                    onClick={() => toggleCollection(coll.name)}
                  >
                    <span className={`tree-chevron${isExpanded ? " open" : ""}`}>
                      <ChevronIcon />
                    </span>
                    <FolderIcon />
                    <span className="tree-name">{coll.name}</span>
                    {isDragTarget && <span className="drop-badge">Drop here</span>}
                  </button>
                  {isExpanded && (
                    <div className="tree-children">
                      {collDocs.length === 0 ? (
                        <div className="tree-empty-hint">Drop files here to ingest</div>
                      ) : (
                        renderTree(tree, coll.name, 1)
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Preview panel */}
        <div className="preview-panel">
          {selectedDoc ? (
            <>
              <div className="preview-header">
                <h3 className="preview-title">{selectedDoc.title}</h3>
                <div className="preview-meta">
                  <code>{selectedDoc.doc_id}</code>
                  <span>
                    {selectedDoc.collection}/{selectedDoc.path}
                  </span>
                </div>
              </div>
              <div className="preview-body">
                {preview === null ? (
                  <div className="preview-loading">Loading...</div>
                ) : (
                  <pre className="preview-content">{preview}</pre>
                )}
              </div>
            </>
          ) : (
            <div className="preview-empty">
              <div className="preview-empty-icon">
                <FileIcon size={48} />
              </div>
              <h3>No document selected</h3>
              <p>
                Select a file from the tree, or drag and drop files onto a collection to ingest
                them.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
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
