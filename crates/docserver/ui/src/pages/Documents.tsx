import { useEffect, useState, useCallback } from "react";
import { api } from "../lib/api";
import type { Collection, SearchResult } from "../lib/api";
import "./Documents.css";

export default function Documents() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [newCollName, setNewCollName] = useState("");
  const [showIngest, setShowIngest] = useState(false);
  const [ingestPath, setIngestPath] = useState("");
  const [ingestContent, setIngestContent] = useState("");
  const [ingesting, setIngesting] = useState(false);

  const loadCollections = useCallback(async () => {
    try {
      const colls = await api.listCollections();
      setCollections(colls);
      if (colls.length > 0 && !selected) setSelected(colls[0].name);
    } catch {
      /* ignore */
    }
  }, [selected]);

  useEffect(() => {
    loadCollections();
  }, [loadCollections]);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setSearching(true);
    try {
      const res = await api.search({
        query: query.trim(),
        mode: "hybrid",
        collection: selected ?? undefined,
        count: 20,
      });
      setResults(res.results);
    } catch {
      /* ignore */
    } finally {
      setSearching(false);
    }
  };

  const handleCreateCollection = async () => {
    const name = newCollName.trim();
    if (!name) return;
    try {
      await api.createCollection(name);
      setNewCollName("");
      setSelected(name);
      await loadCollections();
    } catch {
      /* ignore */
    }
  };

  const handleIngest = async () => {
    if (!selected || !ingestPath.trim() || !ingestContent.trim()) return;
    setIngesting(true);
    try {
      await api.ingestDocuments(selected, [
        {
          path: ingestPath.trim(),
          content: ingestContent,
          content_type: "text/markdown",
        },
      ]);
      setIngestPath("");
      setIngestContent("");
      setShowIngest(false);
    } catch {
      /* ignore */
    } finally {
      setIngesting(false);
    }
  };

  return (
    <div className="documents-page">
      <header className="documents-header">
        <div className="header-top">
          <h2>Documents</h2>
          <div className="header-actions">
            <button className="btn btn-secondary" onClick={() => setShowIngest(!showIngest)}>
              {showIngest ? "Cancel" : "+ Ingest"}
            </button>
          </div>
        </div>

        <div className="collection-bar">
          <div className="collection-tabs">
            {collections.map((c) => (
              <button
                key={c.name}
                className={`collection-tab${selected === c.name ? " active" : ""}`}
                onClick={() => setSelected(c.name)}
              >
                {c.name}
              </button>
            ))}
          </div>
          <form
            className="collection-add"
            onSubmit={(e) => {
              e.preventDefault();
              handleCreateCollection();
            }}
          >
            <input
              type="text"
              placeholder="New collection..."
              value={newCollName}
              onChange={(e) => setNewCollName(e.target.value)}
              className="input-sm"
            />
          </form>
        </div>

        {showIngest && (
          <div className="ingest-panel">
            <input
              type="text"
              placeholder="Document path (e.g. notes/meeting.md)"
              value={ingestPath}
              onChange={(e) => setIngestPath(e.target.value)}
              className="input"
            />
            <textarea
              placeholder="Paste markdown content here..."
              value={ingestContent}
              onChange={(e) => setIngestContent(e.target.value)}
              className="textarea"
              rows={6}
            />
            <button
              className="btn btn-primary"
              onClick={handleIngest}
              disabled={ingesting || !ingestPath.trim() || !ingestContent.trim()}
            >
              {ingesting ? "Ingesting..." : "Ingest Document"}
            </button>
          </div>
        )}

        <form
          className="search-bar"
          onSubmit={(e) => {
            e.preventDefault();
            handleSearch();
          }}
        >
          <input
            type="text"
            placeholder="Search documents..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="search-input"
          />
          <button type="submit" className="btn btn-primary" disabled={searching || !query.trim()}>
            {searching ? "Searching..." : "Search"}
          </button>
        </form>
      </header>

      <div className="documents-results">
        {results.length === 0 ? (
          <div className="empty-state">
            <p>Search for documents or ingest new ones to get started.</p>
          </div>
        ) : (
          <ul className="result-list">
            {results.map((r) => (
              <li key={`${r.collection}:${r.path}`} className="result-item">
                <div className="result-header">
                  <span className="result-title">{r.title}</span>
                  <span className="result-score">{r.score.toFixed(3)}</span>
                </div>
                <div className="result-meta">
                  <code className="result-id">{r.doc_id}</code>
                  <span className="result-path">
                    {r.collection}/{r.path}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
