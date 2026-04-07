import { describe, expect, test } from "bun:test";

import {
  buildBlockTargetId,
  buildHeadingTargetIds,
  extractTrailingBlockReference,
  normalizePreviewFragment,
  previewTargetIdFromFragment,
  slugifyPreviewHeading,
} from "./document-preview-targets";

describe("document preview targets", () => {
  test("slugify_preview_heading_generates_stable_heading_slugs", () => {
    expect(slugifyPreviewHeading("Hello, World!")).toBe("hello-world");
    expect(slugifyPreviewHeading("Café guide")).toBe("cafe-guide");
    expect(slugifyPreviewHeading("   ***   ")).toBe("section");
  });

  test("build_heading_target_ids_disambiguates_duplicate_headings", () => {
    expect(buildHeadingTargetIds(["Overview", "Overview", "Deep Dive", "Overview"])).toEqual([
      "preview-heading-overview",
      "preview-heading-overview-2",
      "preview-heading-deep-dive",
      "preview-heading-overview-3",
    ]);
  });

  test("extract_trailing_block_reference_reads_only_trailing_block_refs", () => {
    expect(extractTrailingBlockReference("Paragraph text ^block-id")).toBe("block-id");
    expect(extractTrailingBlockReference("^top-level")).toBe("top-level");
    expect(extractTrailingBlockReference("^inline block ref in the middle text")).toBeNull();
    expect(extractTrailingBlockReference("Paragraph text")).toBeNull();
  });

  test("normalize_preview_fragment_strips_hashes_and_decodes_text", () => {
    expect(normalizePreviewFragment("#Heading%20One")).toBe("Heading One");
    expect(normalizePreviewFragment("##^block-id")).toBe("^block-id");
    expect(normalizePreviewFragment("   ")).toBeNull();
  });

  test("preview_target_id_from_fragment_maps_headings_and_block_refs", () => {
    expect(previewTargetIdFromFragment("#Heading One")).toBe("preview-heading-heading-one");
    expect(previewTargetIdFromFragment("#^block-id")).toBe("preview-block-block-id");
    expect(previewTargetIdFromFragment("   ")).toBeNull();
  });

  test("build_block_target_id_uses_the_shared_prefix", () => {
    expect(buildBlockTargetId("block-id")).toBe("preview-block-block-id");
  });
});
