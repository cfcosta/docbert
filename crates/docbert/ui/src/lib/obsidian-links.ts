import type { DocumentListItem } from "./api";

export interface ParsedObsidianLink {
  raw: string;
  target: string | null;
  fragment: string | null;
  alias: string | null;
}

export interface ResolvedObsidianLinkTarget {
  collection: string;
  path: string;
  fragment: string | null;
  alias: string | null;
}

export interface ResolveObsidianLinkOptions {
  collection: string;
  currentPath: string;
  documents: Pick<DocumentListItem, "path">[];
}

const MARKDOWN_EXTENSIONS = [".md", ".markdown", ".mdown", ".mkd"];

export function parseObsidianLink(raw: string): ParsedObsidianLink | null {
  if (!raw.startsWith("[[") || !raw.endsWith("]]")) {
    return null;
  }

  const inner = raw.slice(2, -2).trim();
  if (!inner) {
    return null;
  }

  const pipeIndex = inner.indexOf("|");
  const destination = pipeIndex === -1 ? inner : inner.slice(0, pipeIndex).trim();
  const alias = pipeIndex === -1 ? null : inner.slice(pipeIndex + 1).trim() || null;

  if (!destination) {
    return null;
  }

  const hashIndex = destination.indexOf("#");
  const target = hashIndex === -1 ? destination : destination.slice(0, hashIndex).trim() || null;
  const fragment = hashIndex === -1 ? null : destination.slice(hashIndex + 1).trim() || null;

  return {
    raw,
    target,
    fragment,
    alias,
  };
}

export function resolveObsidianLink(
  input: string | ParsedObsidianLink,
  options: ResolveObsidianLinkOptions,
): ResolvedObsidianLinkTarget | null {
  const parsed = typeof input === "string" ? parseObsidianLink(input) : input;
  if (!parsed) {
    return null;
  }

  if (!parsed.target) {
    return {
      collection: options.collection,
      path: options.currentPath,
      fragment: parsed.fragment,
      alias: parsed.alias,
    };
  }

  const explicit = resolveByPath(options.documents, parsed.target);
  if (explicit) {
    return {
      collection: options.collection,
      path: explicit.path,
      fragment: parsed.fragment,
      alias: parsed.alias,
    };
  }

  const stemMatches = resolveByStem(options.documents, parsed.target);
  if (!stemMatches) {
    return null;
  }

  return {
    collection: options.collection,
    path: stemMatches.path,
    fragment: parsed.fragment,
    alias: parsed.alias,
  };
}

function resolveByPath(documents: Pick<DocumentListItem, "path">[], target: string) {
  const normalizedTarget = normalizePathInput(target);
  return documents.find((document) => normalizePathInput(document.path) === normalizedTarget) ?? null;
}

function resolveByStem(documents: Pick<DocumentListItem, "path">[], target: string) {
  const targetStem = pathStem(target);
  const matches = documents.filter((document) => {
    const documentPathStem = pathStem(document.path);
    const documentBaseStem = basenameStem(document.path);
    return documentPathStem === targetStem || documentBaseStem === targetStem;
  });

  if (matches.length !== 1) {
    return null;
  }

  return matches[0];
}

function normalizePathInput(path: string) {
  const trimmed = path.trim();
  return hasMarkdownExtension(trimmed) ? trimmed : `${trimmed}.md`;
}

function pathStem(path: string) {
  const normalized = path.trim().replace(/\\/g, "/");
  for (const extension of MARKDOWN_EXTENSIONS) {
    if (normalized.endsWith(extension)) {
      return normalized.slice(0, -extension.length);
    }
  }
  return normalized;
}

function basenameStem(path: string) {
  const stem = pathStem(path);
  const segments = stem.split("/");
  return segments[segments.length - 1] ?? stem;
}

function hasMarkdownExtension(path: string) {
  return MARKDOWN_EXTENSIONS.some((extension) => path.endsWith(extension));
}
