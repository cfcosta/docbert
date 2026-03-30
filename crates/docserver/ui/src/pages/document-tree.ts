import type { DocumentListItem } from "../lib/api";

export interface DocumentTreeNode {
  name: string;
  path: string;
  isDir: boolean;
  children: DocumentTreeNode[];
  doc?: DocumentListItem;
}

export function buildDocumentTree(docs: DocumentListItem[]): DocumentTreeNode[] {
  const root: DocumentTreeNode[] = [];

  for (const doc of docs) {
    const parts = doc.path.split("/");
    let level = root;
    let pathSoFar = "";

    for (let index = 0; index < parts.length; index += 1) {
      pathSoFar += `${index > 0 ? "/" : ""}${parts[index]}`;
      const isLast = index === parts.length - 1;
      let existing = level.find((node) => node.name === parts[index]);

      if (!existing) {
        existing = {
          name: parts[index],
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

  return sortDocumentTree(root);
}

function sortDocumentTree(nodes: DocumentTreeNode[]): DocumentTreeNode[] {
  const directories = nodes
    .filter((node) => node.isDir)
    .sort((left, right) => left.name.localeCompare(right.name))
    .map((node) => ({ ...node, children: sortDocumentTree(node.children) }));
  const files = nodes
    .filter((node) => !node.isDir)
    .sort((left, right) => left.name.localeCompare(right.name));

  return [...directories, ...files];
}
