export interface ParsedDocumentFrontmatter {
  frontmatter: Record<string, string> | null;
  body: string;
}

export function parseDocumentFrontmatter(content: string): ParsedDocumentFrontmatter {
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
