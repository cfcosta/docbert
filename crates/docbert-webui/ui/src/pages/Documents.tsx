import { useCallback, useEffect, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { useLocation, useNavigate, useParams } from "react-router";
import "katex/dist/katex.min.css";

import { api, buildDocumentTabHref, buildDocumentTabHrefWithFragment } from "../lib/api";
import type { Collection, DocumentListItem } from "../lib/api";
import DocumentPreview, { type ResolvedDocumentTarget } from "./document-preview";
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

interface UploadCandidate {
  file: File;
  path: string;
}

interface FileSystemEntryLike {
  readonly isFile: boolean;
  readonly isDirectory: boolean;
  readonly fullPath?: string;
  readonly name: string;
}

interface FileSystemFileEntryLike extends FileSystemEntryLike {
  file(successCallback: (file: File) => void, errorCallback?: (error: DOMException) => void): void;
}

interface FileSystemDirectoryReaderLike {
  readEntries(
    successCallback: (entries: FileSystemEntryLike[]) => void,
    errorCallback?: (error: DOMException) => void,
  ): void;
}

interface FileSystemDirectoryEntryLike extends FileSystemEntryLike {
  createReader(): FileSystemDirectoryReaderLike;
}

interface DataTransferItemWithEntry {
  webkitGetAsEntry?: () => FileSystemEntryLike | null;
}

const ACCEPTED_UPLOADS = ".md,.markdown,.mdown,.mkd,.pdf";
const MARKDOWN_FILE_RE = /\.(md|markdown|mdown|mkd)$/i;
const PDF_FILE_RE = /\.pdf$/i;

function formatFileCount(count: number) {
  return `${count} file${count === 1 ? "" : "s"}`;
}

function isMarkdownFile(file: File) {
  return MARKDOWN_FILE_RE.test(file.name) || file.type === "text/markdown";
}

function isPdfFile(file: File) {
  return PDF_FILE_RE.test(file.name) || file.type === "application/pdf";
}

function isSupportedUploadFile(file: File) {
  return isMarkdownFile(file) || isPdfFile(file);
}

function uploadCandidateFromFile(file: File): UploadCandidate {
  return {
    file,
    path: file.webkitRelativePath || file.name,
  };
}

async function fileToBase64(file: File) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  const chunkSize = 0x8000;

  for (let index = 0; index < bytes.length; index += chunkSize) {
    const chunk = bytes.subarray(index, index + chunkSize);
    binary += String.fromCharCode(...chunk);
  }

  return btoa(binary);
}

async function buildIngestDocument(candidate: UploadCandidate) {
  const { file, path } = candidate;
  if (isPdfFile(file)) {
    return {
      path,
      content: await fileToBase64(file),
      content_type: "application/pdf" as const,
    };
  }

  return {
    path,
    content: await file.text(),
    content_type: "text/markdown" as const,
  };
}

async function extractDroppedFiles(dataTransfer: DataTransfer): Promise<UploadCandidate[]> {
  const entryItems = Array.from(dataTransfer.items ?? [])
    .map((item) => (item as DataTransferItemWithEntry).webkitGetAsEntry?.() ?? null)
    .filter((entry): entry is FileSystemEntryLike => entry !== null);

  if (entryItems.length === 0) {
    return Array.from(dataTransfer.files).map(uploadCandidateFromFile);
  }

  const nestedFiles = await Promise.all(entryItems.map((entry) => readDroppedEntry(entry)));
  return dedupeUploadCandidates(nestedFiles.flat());
}

async function readDroppedEntry(entry: FileSystemEntryLike): Promise<UploadCandidate[]> {
  if (entry.isFile) {
    const file = await getDroppedFile(entry as FileSystemFileEntryLike);
    const path = normalizeDroppedPath(entry.fullPath, file.name);
    return [{ file, path }];
  }

  if (!entry.isDirectory) {
    return [];
  }

  const reader = (entry as FileSystemDirectoryEntryLike).createReader();
  const children = await readAllDirectoryEntries(reader);
  const nestedFiles = await Promise.all(children.map((child) => readDroppedEntry(child)));
  return nestedFiles.flat();
}

function getDroppedFile(entry: FileSystemFileEntryLike): Promise<File> {
  return new Promise((resolve, reject) => {
    entry.file(resolve, reject);
  });
}

async function readAllDirectoryEntries(
  reader: FileSystemDirectoryReaderLike,
): Promise<FileSystemEntryLike[]> {
  const entries: FileSystemEntryLike[] = [];

  while (true) {
    const chunk = await new Promise<FileSystemEntryLike[]>((resolve, reject) => {
      reader.readEntries(resolve, reject);
    });
    if (chunk.length === 0) {
      return entries;
    }
    entries.push(...chunk);
  }
}

function normalizeDroppedPath(fullPath: string | undefined, fallbackName: string) {
  const normalized = fullPath?.replace(/^\/+/, "").trim();
  return normalized && normalized.length > 0 ? normalized : fallbackName;
}

function dedupeUploadCandidates(candidates: UploadCandidate[]) {
  return Array.from(new Map(candidates.map((candidate) => [candidate.path, candidate])).values());
}

export default function Documents() {
  const navigate = useNavigate();
  const location = useLocation();
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
  const selectedDocRef = useRef<SelectedDocumentSummary | null>(null);
  const documentRequestSeqRef = useRef(0);

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

  // Mount-time data fetch. `loadCollections` resolves to `setCollections`
  // after an async network call; the `set-state-in-effect` rule can't
  // see that the setState hop is behind a promise and flags the call
  // as a synchronous-in-effect update. Effects _are_ the documented tool
  // for "fetch data on mount" (see
  // https://react.dev/learn/synchronizing-with-effects#fetching-data),
  // so silence the rule with a targeted disable rather than contorting
  // the code around it.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- async data fetch
    void loadCollections();
  }, [loadCollections]);

  useEffect(() => {
    if (!status || status.tone === "loading" || status.tone === "error") {
      return undefined;
    }

    const timeoutId = window.setTimeout(() => setStatus(null), 4000);
    return () => window.clearTimeout(timeoutId);
  }, [status]);

  useEffect(() => {
    selectedDocRef.current = selectedDoc;
  }, [selectedDoc]);

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
      fragment?: string | null,
    ) => {
      const requestId = documentRequestSeqRef.current + 1;
      documentRequestSeqRef.current = requestId;

      setSelectedDoc({
        collection,
        path,
        title: fallback?.title ?? path,
        doc_id: fallback?.doc_id ?? "Loading…",
      });
      setConfirmDeleteDoc(null);
      setPreview(null);
      navigate(
        fragment
          ? buildDocumentTabHrefWithFragment(collection, path, fragment)
          : buildDocumentTabHref(collection, path),
        { replace: true },
      );

      try {
        const full = await api.getDocument(collection, path);
        if (documentRequestSeqRef.current !== requestId) {
          return;
        }

        setSelectedDoc({
          collection: full.collection,
          path: full.path,
          title: full.title,
          doc_id: full.doc_id,
        });
        setPreview(full.content || "_No content stored._");
      } catch (error) {
        if (documentRequestSeqRef.current !== requestId) {
          return;
        }

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

  const openResolvedDocument = useCallback(
    ({ collection, path, fragment }: ResolvedDocumentTarget) => {
      const listedDoc = docs[collection]?.find((document) => document.path === path);
      void openDocument(collection, path, listedDoc, fragment);
    },
    [docs, openDocument],
  );

  const routeCollection = params.collection?.trim();
  const routePath = params["*"]?.trim();
  const routeFragment = location.hash ? decodeURIComponent(location.hash.slice(1)) : null;

  // Route-driven sync: when the URL points at a specific document we
  // expand its collection (if the user hasn't already) and kick off
  // the document list + document body fetch. Both the `setExpanded`
  // call and the downstream `void loadDocs` eventually hit setState,
  // which the rule flags — but this is the canonical "URL route is
  // the source of truth, reflect it into component state" pattern and
  // can't be done during render because expanded-ness is mutable local
  // UI state the user can override with a later collapse click.
  useEffect(() => {
    if (!routeCollection || !routePath) {
      return;
    }

    // eslint-disable-next-line react-hooks/set-state-in-effect -- route -> UI sync
    setExpanded((previous) => new Set(previous).add(routeCollection));
    if (!docs[routeCollection]) {
      void loadDocs(routeCollection);
    }

    const currentSelection = selectedDocRef.current;
    if (currentSelection?.collection === routeCollection && currentSelection.path === routePath) {
      return;
    }

    const listedDoc = docs[routeCollection]?.find((doc) => doc.path === routePath);
    if (routeCollection in docs && !listedDoc) {
      setSelectedDoc(null);
      setPreview(null);
      return;
    }

    void openDocument(routeCollection, routePath, listedDoc, routeFragment);
  }, [docs, loadDocs, openDocument, routeCollection, routeFragment, routePath]);

  const ingestFiles = useCallback(
    async (collection: string, candidates: UploadCandidate[]) => {
      if (candidates.length === 0) {
        return;
      }

      const unsupported = candidates.filter(({ file }) => !isSupportedUploadFile(file));
      if (unsupported.length > 0) {
        setStatus({
          tone: "error",
          text: "Only Markdown and PDF files are supported right now.",
        });
        return;
      }

      setIngesting(true);
      setStatus({
        tone: "loading",
        text: `Ingesting ${formatFileCount(candidates.length)} into ${collection}…`,
      });

      try {
        const documents = await Promise.all(
          candidates.map((candidate) => buildIngestDocument(candidate)),
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

      await ingestFiles(collection, files.map(uploadCandidateFromFile));
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

      const droppedFiles = await extractDroppedFiles(event.dataTransfer);
      await ingestFiles(collection, droppedFiles);
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
        accept={ACCEPTED_UPLOADS}
        onChange={(event) => void handleUploadInputChange(event)}
      />

      {status && (
        <div
          className={`ingest-toast ${status.tone}`}
          role={status.tone === "error" ? "alert" : "status"}
          aria-live="polite"
        >
          <span className="ingest-toast-text">{status.text}</span>
          {status.tone === "error" && (
            <button
              type="button"
              className="ingest-toast-dismiss"
              onClick={() => setStatus(null)}
              aria-label="Dismiss error"
            >
              ×
            </button>
          )}
        </div>
      )}

      <div className="file-manager">
        <section className="file-tree-panel" aria-label="Collections">
          <p className="sr-only">Drop files or folders onto a collection.</p>
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
          <DocumentPreview
            selectedDoc={selectedDoc}
            preview={preview}
            resolverDocuments={selectedDoc ? (docs[selectedDoc.collection] ?? []) : []}
            activeFragment={routeFragment}
            onOpenResolvedDocument={openResolvedDocument}
          />
        </section>
      </div>
    </div>
  );
}
