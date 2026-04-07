import { useEffect, useState } from "react";

import { api, type Collection, type SearchMode } from "../lib/api";
import "./Search.css";

const ALL_COLLECTIONS_VALUE = "";

export default function Search() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [selectedCollection, setSelectedCollection] = useState(ALL_COLLECTIONS_VALUE);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [loadingCollections, setLoadingCollections] = useState(true);
  const [collectionsError, setCollectionsError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    api
      .listCollections()
      .then((nextCollections) => {
        if (!active) {
          return;
        }
        setCollections(nextCollections);
        setCollectionsError(null);
      })
      .catch((error) => {
        if (!active) {
          return;
        }
        setCollectionsError(error instanceof Error ? error.message : "Could not load collections.");
      })
      .finally(() => {
        if (active) {
          setLoadingCollections(false);
        }
      });

    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="search-page">
      <header className="search-header">
        <h2>Search</h2>
        <p className="search-subtitle">Find documents across your indexed collections.</p>
      </header>

      <div className="search-body">
        <section className="search-controls" aria-label="Search controls">
          <div className="search-field search-field-query">
            <label className="search-label" htmlFor="search-query">
              Query
            </label>
            <input
              id="search-query"
              className="search-input"
              type="search"
              placeholder="Search your documents..."
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>

          <div className="search-field-grid">
            <div className="search-field">
              <label className="search-label" htmlFor="search-mode">
                Mode
              </label>
              <select
                id="search-mode"
                className="search-select"
                value={mode}
                onChange={(event) => setMode(event.target.value as SearchMode)}
              >
                <option value="hybrid">Hybrid</option>
                <option value="semantic">Semantic</option>
              </select>
            </div>

            <div className="search-field">
              <label className="search-label" htmlFor="search-collection">
                Collection
              </label>
              <select
                id="search-collection"
                className="search-select"
                value={selectedCollection}
                onChange={(event) => setSelectedCollection(event.target.value)}
                disabled={loadingCollections || collectionsError !== null}
              >
                <option value={ALL_COLLECTIONS_VALUE}>All collections</option>
                {collections.map((collection) => (
                  <option key={collection.name} value={collection.name}>
                    {collection.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </section>

        {loadingCollections ? (
          <div className="search-state-card">
            <p className="search-state-title">Loading collections…</p>
            <p className="search-state-text">Fetching available collections before search is enabled.</p>
          </div>
        ) : collectionsError ? (
          <div className="search-state-card search-state-card-error" role="alert">
            <p className="search-state-title">Could not load collections</p>
            <p className="search-state-text">{collectionsError}</p>
          </div>
        ) : (
          <div className="search-state-card">
            <p className="search-state-title">Start with a search query</p>
            <p className="search-state-text">
              Enter a query above to search across all collections or narrow the scope with the
              collection filter.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
