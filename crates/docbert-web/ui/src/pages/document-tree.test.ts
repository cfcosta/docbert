import { describe, expect, test } from "bun:test";

import type { DocumentListItem } from "../lib/api";
import { buildDocumentTree } from "./document-tree";

describe("document-tree", () => {
  test("build_document_tree_groups_nested_segments_and_preserves_leaf_paths", () => {
    const docs: DocumentListItem[] = [
      { doc_id: "#top001", path: "top.md", title: "Top" },
      { doc_id: "#beta01", path: "nested/beta.md", title: "Beta" },
      { doc_id: "#gamma1", path: "nested/deeper/gamma.md", title: "Gamma" },
    ];

    const tree = buildDocumentTree(docs);

    expect(tree.map((node) => node.name)).toEqual(["nested", "top.md"]);
    expect(tree[0]).toMatchObject({
      name: "nested",
      path: "nested",
      isDir: true,
      children: [
        {
          name: "deeper",
          path: "nested/deeper",
          isDir: true,
          children: [
            {
              name: "gamma.md",
              path: "nested/deeper/gamma.md",
              isDir: false,
            },
          ],
        },
        {
          name: "beta.md",
          path: "nested/beta.md",
          isDir: false,
        },
      ],
    });
    expect(tree[1]).toMatchObject({
      name: "top.md",
      path: "top.md",
      isDir: false,
    });
  });

  test("document_tree_orders_directories_before_files_and_sorts_lexically", () => {
    const docs: DocumentListItem[] = [
      { doc_id: "#zeta01", path: "zeta.md", title: "Zeta" },
      { doc_id: "#alpha1", path: "alpha.md", title: "Alpha" },
      { doc_id: "#beta01", path: "nested/beta.md", title: "Beta" },
      { doc_id: "#gamma1", path: "aardvark/gamma.md", title: "Gamma" },
      { doc_id: "#delta1", path: "nested/alpha.md", title: "Alpha nested" },
    ];

    const tree = buildDocumentTree(docs);

    expect(tree.map((node) => node.name)).toEqual(["aardvark", "nested", "alpha.md", "zeta.md"]);
    expect(tree[1].children.map((node) => node.name)).toEqual(["alpha.md", "beta.md"]);
  });

  test("document_tree_keeps_same_leaf_name_distinct_by_full_path", () => {
    const first: DocumentListItem = { doc_id: "#readm1", path: "a/readme.md", title: "Readme A" };
    const second: DocumentListItem = { doc_id: "#readm2", path: "b/readme.md", title: "Readme B" };

    const tree = buildDocumentTree([second, first]);

    expect(tree.map((node) => node.path)).toEqual(["a", "b"]);
    expect(tree[0].children[0]).toMatchObject({ name: "readme.md", path: "a/readme.md" });
    expect(tree[1].children[0]).toMatchObject({ name: "readme.md", path: "b/readme.md" });
    expect(tree[0].children[0].doc).toBe(first);
    expect(tree[1].children[0].doc).toBe(second);
  });
});
