const HEADING_TARGET_PREFIX = "preview-heading-";
const BLOCK_TARGET_PREFIX = "preview-block-";

export function slugifyPreviewHeading(text: string): string {
  const normalized = text
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .trim()
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");

  return normalized || "section";
}

export function buildHeadingTargetIds(headings: string[]): string[] {
  const counts = new Map<string, number>();

  return headings.map((heading) => {
    const slug = slugifyPreviewHeading(heading);
    const count = counts.get(slug) ?? 0;
    counts.set(slug, count + 1);
    return count === 0
      ? `${HEADING_TARGET_PREFIX}${slug}`
      : `${HEADING_TARGET_PREFIX}${slug}-${count + 1}`;
  });
}

export function extractTrailingBlockReference(text: string): string | null {
  const match = text.match(/(?:^|\s)\^([A-Za-z0-9-]+)\s*$/);
  return match ? match[1] : null;
}

export function normalizePreviewFragment(fragment: string): string | null {
  const stripped = fragment.trim().replace(/^#+/, "");
  if (!stripped) {
    return null;
  }

  try {
    return decodeURIComponent(stripped);
  } catch {
    return stripped;
  }
}

export function previewTargetIdFromFragment(fragment: string): string | null {
  const normalized = normalizePreviewFragment(fragment);
  if (!normalized) {
    return null;
  }

  if (normalized.startsWith("^")) {
    return `${BLOCK_TARGET_PREFIX}${normalized.slice(1)}`;
  }

  return `${HEADING_TARGET_PREFIX}${slugifyPreviewHeading(normalized)}`;
}

export function buildBlockTargetId(blockId: string): string {
  return `${BLOCK_TARGET_PREFIX}${blockId}`;
}
