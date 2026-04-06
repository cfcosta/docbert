import { useCallback, useEffect, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { useNavigate, useParams } from "react-router";
import "katex/dist/katex.min.css";

import { api, buildDocumentTabHref } from "../lib/api";
import type { Collection, DocumentListItem } from "../lib/api";
import DocumentPreview from "./document-preview";
import {
  clearDeletedDocumentSelection,
  removeDocumentFromDocs,
  toggleExpandedKey,
} from "./documents-page-state";
import DocumentsTree, { type SelectedDocumentSummary } from "./documents-tree";
import "./Documents.css";

interface StatusMessage {
  tone: "loading" | "success" | "error";
  text: string;
}

const ACCEPTED_MARKDOWN = ".md,.markdown,.mdown,.mkd";
const MARKDOWN_FILE_RE = /\.(md|markdown|mdown|mkd)$/i;

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
  const [loadingCollections, setLoadingCollections] = useState<Set<string>>(new Set());
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [selectedDoc, setSelectedDoc] = useState<SelectedDocumentSummary | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState<string | null>(null);
  const [ingesting, setIngesting] = useState(false);
  const [deletingDoc, setDeletingDoc] = useState(false);
  const [status, setStatus] = useState<StatusMessage | null>(null);
  const [uploadCollection, setUploadCollection] = useState<string | null>(null);
  const [confirmDeleteDoc, setConfirmDeleteDoc] = useState<string | null>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);

  const loadCollections = useCallback(async () => {
    try {
      const nextCollections = await api.listCollections();
      setCollections(nextCollections);
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
    setLoadingCollections((previous) => new Set(previous).add(collection));

    try {
      const items = await api.listDocuments(collection);
      setDocs((previous) => ({ ...previous, [collection]: items }));
    } catch (error) {
      setStatus({
        tone: "error",
        text:
          error instanceof Error ? error.message : `Failed to load documents for ${collection}.`,
      });
    } finally {
      setLoadingCollections((previous) => {
        const next = new Set(previous);
        next.delete(collection);
        return next;
      });
    }
  }, []);

  const toggleCollection = useCallback(
    (name: string) => {
      const opening = !expanded.has(name);

      setExpanded((previous) => toggleExpandedKey(previous, name));

      if (opening && !docs[name]) {
        void loadDocs(name);
      }
    },
    [docs, expanded, loadDocs],
  );

  const toggleDir = useCallback((key: string) => {
    setExpanded((previous) => toggleExpandedKey(previous, key));
  }, []);

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

    setExpanded((previous) => new Set(previous).add(collection));
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
        const documents = await Promise.all(
          files.map(async (file) => ({
            path: file.webkitRelativePath || file.name,
            content: await file.text(),
            content_type: "text/markdown" as const,
          })),
        );

        const response = await api.ingestDocuments(collection, documents);
        setExpanded((previous) => new Set(previous).add(collection));
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

  const openUploadPicker = useCallback(
    (collection: string) => {
      if (ingesting) {
        return;
      }

      setUploadCollection(collection);
      uploadInputRef.current?.click();
    },
    [ingesting],
  );

  const handleUploadInputChange = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const collection = uploadCollection;
      const files = Array.from(event.target.files ?? []);
      event.target.value = "";

      if (!collection || files.length === 0) {
        return;
      }

      await ingestFiles(collection, files);
    },
    [ingestFiles, uploadCollection],
  );

  const onDragOver = useCallback((event: DragEvent<HTMLDivElement>, collection: string) => {
    event.preventDefault();
    event.stopPropagation();
    setDragOver(collection);
  }, []);

  const onDragLeave = useCallback((event: DragEvent<HTMLDivElement>, collection: string) => {
    event.preventDefault();
    event.stopPropagation();

    const nextTarget = event.relatedTarget;
    if (nextTarget instanceof Node && event.currentTarget.contains(nextTarget)) {
      return;
    }

    setDragOver((current) => (current === collection ? null : current));
  }, []);

  const onDrop = useCallback(
    async (event: DragEvent<HTMLDivElement>, collection: string) => {
      event.preventDefault();
      event.stopPropagation();
      setDragOver(null);

      await ingestFiles(collection, Array.from(event.dataTransfer.files));
    },
    [ingestFiles],
  );

  const handleDeleteDocument = useCallback(
    async (collection: string, doc: DocumentListItem) => {
      setDeletingDoc(true);

      try {
        await api.deleteDocument(collection, doc.path);
        setConfirmDeleteDoc(null);
        setDocs((previous) => removeDocumentFromDocs(previous, collection, doc.path));
        const nextSelection = clearDeletedDocumentSelection(
          selectedDoc,
          preview,
          collection,
          doc.path,
        );
        setSelectedDoc(nextSelection.selectedDoc);
        setPreview(nextSelection.preview);
        if (
          !nextSelection.selectedDoc &&
          selectedDoc?.collection === collection &&
          selectedDoc.path === doc.path
        ) {
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
    [loadDocs, navigate, preview, selectedDoc],
  );

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
          </div>

          <DocumentsTree
            collections={collections}
            docs={docs}
            loadingCollections={loadingCollections}
            expanded={expanded}
            dragOver={dragOver}
            ingesting={ingesting}
            deletingDoc={deletingDoc}
            confirmDeleteDoc={confirmDeleteDoc}
            selectedDoc={selectedDoc}
            onToggleCollection={toggleCollection}
            onToggleDir={toggleDir}
            onOpenUploadPicker={openUploadPicker}
            onSelectFile={(collection, doc) => void selectFile(collection, doc)}
            onDeleteDocument={(collection, doc) => void handleDeleteDocument(collection, doc)}
            onSetConfirmDeleteDoc={setConfirmDeleteDoc}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={(event, collection) => void onDrop(event, collection)}
          />
        </section>

        <section className="preview-panel" aria-live="polite">
          <DocumentPreview selectedDoc={selectedDoc} preview={preview} />
        </section>
      </div>
    </div>
  );
}
