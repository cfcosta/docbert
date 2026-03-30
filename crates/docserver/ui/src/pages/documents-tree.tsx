import type { DragEvent } from "react";

import type { Collection, DocumentListItem } from "../lib/api";
import { buildDocumentTree, type DocumentTreeNode } from "./document-tree";

export interface SelectedDocumentSummary {
  collection: string;
  path: string;
  title: string;
  doc_id: string;
}

function formatFileCount(count: number) {
  return `${count} file${count === 1 ? "" : "s"}`;
}

function renderTree({
  nodes,
  collection,
  depth,
  expanded,
  selectedDoc,
  confirmDeleteDoc,
  deletingDoc,
  onToggleDir,
  onSelectFile,
  onDeleteDocument,
  onSetConfirmDeleteDoc,
}: {
  nodes: DocumentTreeNode[];
  collection: string;
  depth: number;
  expanded: Set<string>;
  selectedDoc: SelectedDocumentSummary | null;
  confirmDeleteDoc: string | null;
  deletingDoc: boolean;
  onToggleDir: (key: string) => void;
  onSelectFile: (collection: string, doc: DocumentListItem) => void;
  onDeleteDocument: (collection: string, doc: DocumentListItem) => void;
  onSetConfirmDeleteDoc: (key: string | null) => void;
}) {
  const directories = nodes.filter((node) => node.isDir).sort((a, b) => a.name.localeCompare(b.name));
  const files = nodes.filter((node) => !node.isDir).sort((a, b) => a.name.localeCompare(b.name));

  return (
    <>
      {directories.map((node) => {
        const key = `${collection}/${node.path}`;
        const isExpanded = expanded.has(key);

        return (
          <div key={key}>
            <button
              type="button"
              className="tree-item tree-dir"
              style={{ paddingLeft: `${12 + depth * 16}px` }}
              onClick={() => onToggleDir(key)}
              aria-expanded={isExpanded}
              title={node.path}
            >
              <span className={`tree-chevron${isExpanded ? " open" : ""}`}>
                <ChevronIcon />
              </span>
              <FolderIcon />
              <span className="tree-name">{node.name}</span>
            </button>

            {isExpanded && (
              <div>
                {renderTree({
                  nodes: node.children,
                  collection,
                  depth: depth + 1,
                  expanded,
                  selectedDoc,
                  confirmDeleteDoc,
                  deletingDoc,
                  onToggleDir,
                  onSelectFile,
                  onDeleteDocument,
                  onSetConfirmDeleteDoc,
                })}
              </div>
            )}
          </div>
        );
      })}

      {files.map((node) => {
        const isSelected = selectedDoc?.collection === collection && selectedDoc.path === node.path;
        const deleteKey = `${collection}/${node.path}`;
        const isConfirmingDelete = confirmDeleteDoc === deleteKey;

        return (
          <div key={`${collection}/${node.path}`} className={`tree-file-row${isSelected ? " selected" : ""}`}>
            <button
              type="button"
              className={`tree-item tree-file${isSelected ? " selected" : ""}`}
              style={{ paddingLeft: `${12 + depth * 16}px` }}
              onClick={() => node.doc && onSelectFile(collection, node.doc)}
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
                  onClick={() => node.doc && onDeleteDocument(collection, node.doc)}
                  disabled={deletingDoc}
                  title="Confirm delete"
                >
                  {deletingDoc ? "Deleting…" : "Delete"}
                </button>
                <button
                  type="button"
                  className="tree-confirm-no"
                  onClick={() => onSetConfirmDeleteDoc(null)}
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
                  onSetConfirmDeleteDoc(deleteKey);
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
}

export default function DocumentsTree({
  collections,
  docs,
  loadingCollections,
  expanded,
  dragOver,
  ingesting,
  deletingDoc,
  confirmDelete,
  confirmDeleteDoc,
  selectedDoc,
  onToggleCollection,
  onToggleDir,
  onOpenUploadPicker,
  onSelectFile,
  onDeleteDocument,
  onSetConfirmDelete,
  onSetConfirmDeleteDoc,
  onDeleteCollection,
  onDragOver,
  onDragLeave,
  onDrop,
}: {
  collections: Collection[];
  docs: Record<string, DocumentListItem[]>;
  loadingCollections: Set<string>;
  expanded: Set<string>;
  dragOver: string | null;
  ingesting: boolean;
  deletingDoc: boolean;
  confirmDelete: string | null;
  confirmDeleteDoc: string | null;
  selectedDoc: SelectedDocumentSummary | null;
  onToggleCollection: (name: string) => void;
  onToggleDir: (key: string) => void;
  onOpenUploadPicker: (collection: string) => void;
  onSelectFile: (collection: string, doc: DocumentListItem) => void;
  onDeleteDocument: (collection: string, doc: DocumentListItem) => void;
  onSetConfirmDelete: (collection: string | null) => void;
  onSetConfirmDeleteDoc: (path: string | null) => void;
  onDeleteCollection: (name: string) => void;
  onDragOver: (event: DragEvent<HTMLDivElement>, collection: string) => void;
  onDragLeave: (event: DragEvent<HTMLDivElement>, collection: string) => void;
  onDrop: (event: DragEvent<HTMLDivElement>, collection: string) => void;
}) {
  return (
    <div className="file-tree" aria-label="Document collections">
      {collections.length === 0 && (
        <div className="tree-empty">
          <p>No collections yet.</p>
          <p>Create one to start indexing Markdown files.</p>
        </div>
      )}

      {collections.map((collection) => {
        const isExpanded = expanded.has(collection.name);
        const collectionDocs = docs[collection.name] ?? [];
        const tree = buildDocumentTree(collectionDocs);
        const isDragTarget = dragOver === collection.name;
        const isLoading = loadingCollections.has(collection.name);
        const countLabel = collection.name in docs ? formatFileCount(collectionDocs.length) : null;

        return (
          <div
            key={collection.name}
            className={`tree-collection${isDragTarget ? " drag-over" : ""}`}
            onDragOver={(event) => onDragOver(event, collection.name)}
            onDragLeave={(event) => onDragLeave(event, collection.name)}
            onDrop={(event) => onDrop(event, collection.name)}
          >
            <div className="tree-collection-head">
              <button
                type="button"
                className="tree-item tree-collection-header"
                onClick={() => onToggleCollection(collection.name)}
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
                onClick={() => onOpenUploadPicker(collection.name)}
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
                    onClick={() => onDeleteCollection(collection.name)}
                    title="Confirm delete"
                  >
                    Delete
                  </button>
                  <button
                    type="button"
                    className="tree-confirm-no"
                    onClick={() => onSetConfirmDelete(null)}
                    title="Cancel"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  className="tree-delete-btn"
                  onClick={() => onSetConfirmDelete(collection.name)}
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
                ) : collectionDocs.length === 0 ? (
                  <div className="tree-empty-hint">Drop Markdown files here or use upload.</div>
                ) : (
                  renderTree({
                    nodes: tree,
                    collection: collection.name,
                    depth: 1,
                    expanded,
                    selectedDoc,
                    confirmDeleteDoc,
                    deletingDoc,
                    onToggleDir,
                    onSelectFile,
                    onDeleteDocument,
                    onSetConfirmDeleteDoc,
                  })
                )}
              </div>
            )}
          </div>
        );
      })}
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
