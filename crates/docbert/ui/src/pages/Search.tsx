import { useEffect, useRef, useState } from "react";

import SearchResults from "../components/SearchResults";
import { api, type Collection, type SearchMode, type SearchResult } from "../lib/api";
import "./Search.css";

const ALL_COLLECTIONS_VALUE = "";
const SEARCH_DEBOUNCE_MS = 200;

export default function Search() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [selectedCollection, setSelectedCollection] = useState(ALL_COLLECTIONS_VALUE);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [loadingCollections, setLoadingCollections] = useState(true);
  const [collectionsError, setCollectionsError] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [resultCount, setResultCount] = useState<number | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const latestSearchRequestRef = useRef(0);

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

  useEffect(() => {
    if (loadingCollections || collectionsError) {
      return;
    }

    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      latestSearchRequestRef.current += 1;
      setSearching(false);
      setSearchError(null);
      setResultCount(null);
      setResults([]);
      return;
    }

    const timeoutId = window.setTimeout(() => {
      const requestId = latestSearchRequestRef.current + 1;
      latestSearchRequestRef.current = requestId;
      setSearching(true);
      setSearchError(null);

      api
        .search({
          query: trimmedQuery,
          mode,
          collection: selectedCollection || undefined,
        })
        .then((response) => {
          if (latestSearchRequestRef.current !== requestId) {
            return;
          }
          setResultCount(response.result_count);
          setResults(response.results);
          setSearchError(null);
        })
        .catch((error) => {
          if (latestSearchRequestRef.current !== requestId) {
            return;
          }
          setSearchError(error instanceof Error ? error.message : "Search failed.");
          setResultCount(null);
          setResults([]);
        })
        .finally(() => {
          if (latestSearchRequestRef.current === requestId) {
            setSearching(false);
          }
        });
    }, SEARCH_DEBOUNCE_MS);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [collectionsError, loadingCollections, mode, query, selectedCollection]);

  const trimmedQuery = query.trim();

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
        ) : !trimmedQuery ? (
          <div className="search-state-card">
            <p className="search-state-title">Start with a search query</p>
            <p className="search-state-text">
              Enter a query above to search across all collections or narrow the scope with the
              collection filter.
            </p>
          </div>
        ) : searching ? (
          <div className="search-state-card">
            <p className="search-state-title">Searching…</p>
            <p className="search-state-text">Looking for matching documents.</p>
          </div>
        ) : searchError ? (
          <div className="search-state-card search-state-card-error" role="alert">
            <p className="search-state-title">Search failed</p>
            <p className="search-state-text">{searchError}</p>
          </div>
        ) : resultCount === 0 || results.length === 0 ? (
          <div className="search-state-card">
            <p className="search-state-title">No results</p>
            <p className="search-state-text">Try a different query, mode, or collection filter.</p>
          </div>
        ) : (
          <div className="search-results-panel">
            <div className="search-results-summary">
              <p className="search-results-title">Found {resultCount ?? 0} results</p>
            </div>
            <SearchResults results={results} />
          </div>
        )}
      </div>
    </div>
  );
}
