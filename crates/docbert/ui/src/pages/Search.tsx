import { useEffect } from "react";

import SearchResults from "../components/SearchResults";
import { api } from "../lib/api";
import { useSearchSession } from "./search-session";
import "./Search.css";

const ALL_COLLECTIONS_VALUE = "";
const SEARCH_DEBOUNCE_MS = 200;

export default function Search() {
  const { searchSession, setSearchSession } = useSearchSession();
  const {
    query,
    mode,
    selectedCollection,
    collections,
    loadingCollections,
    collectionsError,
    searching,
    searchError,
    resultCount,
    results,
    latestSearchRequestId,
    lastIssuedQuery,
    lastIssuedMode,
    lastIssuedCollection,
  } = searchSession;

  useEffect(() => {
    if (!loadingCollections) {
      return;
    }

    let active = true;

    api
      .listCollections()
      .then((nextCollections) => {
        if (!active) {
          return;
        }
        setSearchSession((previous) => ({
          ...previous,
          collections: nextCollections,
          collectionsError: null,
        }));
      })
      .catch((error) => {
        if (!active) {
          return;
        }
        setSearchSession((previous) => ({
          ...previous,
          collectionsError: error instanceof Error ? error.message : "Could not load collections.",
        }));
      })
      .finally(() => {
        if (active) {
          setSearchSession((previous) => ({
            ...previous,
            loadingCollections: false,
          }));
        }
      });

    return () => {
      active = false;
    };
  }, [loadingCollections, setSearchSession]);

  useEffect(() => {
    if (loadingCollections || collectionsError) {
      return;
    }

    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      const shouldResetBlankState =
        searching ||
        searchError !== null ||
        resultCount !== null ||
        results.length > 0 ||
        lastIssuedQuery !== null ||
        lastIssuedMode !== null ||
        lastIssuedCollection !== null;

      if (shouldResetBlankState) {
        setSearchSession((previous) => ({
          ...previous,
          latestSearchRequestId: previous.latestSearchRequestId + 1,
          searching: false,
          searchError: null,
          resultCount: null,
          results: [],
          lastIssuedQuery: null,
          lastIssuedMode: null,
          lastIssuedCollection: null,
        }));
      }
      return;
    }

    const selectedCollectionValue = selectedCollection || null;
    const shouldSkipSearch =
      lastIssuedQuery === trimmedQuery &&
      lastIssuedMode === mode &&
      lastIssuedCollection === selectedCollectionValue;

    if (shouldSkipSearch) {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      const requestId = latestSearchRequestId + 1;
      setSearchSession((previous) => ({
        ...previous,
        latestSearchRequestId: requestId,
        searching: true,
        searchError: null,
        lastIssuedQuery: trimmedQuery,
        lastIssuedMode: previous.mode,
        lastIssuedCollection: previous.selectedCollection || null,
      }));

      api
        .search({
          query: trimmedQuery,
          mode,
          collection: selectedCollection || undefined,
        })
        .then((response) => {
          let shouldApply = false;
          setSearchSession((previous) => {
            if (previous.latestSearchRequestId !== requestId) {
              return previous;
            }
            shouldApply = true;
            return {
              ...previous,
              resultCount: response.result_count,
              results: response.results,
              searchError: null,
            };
          });
          if (!shouldApply) {
            return;
          }
        })
        .catch((error) => {
          setSearchSession((previous) => {
            if (previous.latestSearchRequestId !== requestId) {
              return previous;
            }
            return {
              ...previous,
              searchError: error instanceof Error ? error.message : "Search failed.",
              resultCount: null,
              results: [],
            };
          });
        })
        .finally(() => {
          setSearchSession((previous) => {
            if (previous.latestSearchRequestId !== requestId) {
              return previous;
            }
            return {
              ...previous,
              searching: false,
            };
          });
        });
    }, SEARCH_DEBOUNCE_MS);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [
    collectionsError,
    lastIssuedCollection,
    lastIssuedMode,
    lastIssuedQuery,
    latestSearchRequestId,
    loadingCollections,
    mode,
    query,
    resultCount,
    results,
    searchError,
    searching,
    selectedCollection,
    setSearchSession,
  ]);

  const trimmedQuery = query.trim();

  return (
    <div className={`search-page${trimmedQuery ? " search-page-has-query" : ""}`}>
      <header className="search-header">
        <div className="search-header-inner">
          <h2>Search</h2>
          <p className="search-subtitle">Search across your indexed collections.</p>
        </div>
      </header>

      <div className="search-body">
        <section className="search-controls" aria-label="Search controls">
          <div className="search-query-shell">
            <label className="search-label sr-only" htmlFor="search-query">
              Query
            </label>
            <span className="search-query-icon" aria-hidden="true">
              ⌕
            </span>
            <input
              id="search-query"
              className="search-input search-input-query"
              type="search"
              placeholder="Search your documents..."
              value={query}
              onChange={(event) =>
                setSearchSession((previous) => ({ ...previous, query: event.target.value }))
              }
            />
          </div>

          <div className="search-toolbar">
            <div className="search-toolbar-group">
              <div className="search-field search-field-inline">
                <label className="search-label sr-only" htmlFor="search-mode">
                  Mode
                </label>
                <select
                  id="search-mode"
                  className="search-select search-filter"
                  value={mode}
                  onChange={(event) =>
                    setSearchSession((previous) => ({
                      ...previous,
                      mode: event.target.value as typeof previous.mode,
                    }))
                  }
                >
                  <option value="hybrid">Hybrid</option>
                  <option value="semantic">Semantic</option>
                </select>
              </div>

              <div className="search-field search-field-inline">
                <label className="search-label sr-only" htmlFor="search-collection">
                  Collection
                </label>
                <select
                  id="search-collection"
                  className="search-select search-filter search-filter-collection"
                  value={selectedCollection}
                  onChange={(event) =>
                    setSearchSession((previous) => ({
                      ...previous,
                      selectedCollection: event.target.value,
                    }))
                  }
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
          </div>
        </section>

        {loadingCollections ? (
          <div className="search-state-card">
            <p className="search-state-title">Loading collections…</p>
            <p className="search-state-text">
              Fetching available collections before search is enabled.
            </p>
          </div>
        ) : collectionsError ? (
          <div className="search-state-card search-state-card-error" role="alert">
            <p className="search-state-title">Could not load collections</p>
            <p className="search-state-text">{collectionsError}</p>
          </div>
        ) : !trimmedQuery ? (
          <div className="search-state-card search-state-card-blank">
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
              <p className="search-results-meta">
                Showing matches for “{trimmedQuery}” in {selectedCollection || "all collections"}.
              </p>
            </div>
            <SearchResults results={results} />
          </div>
        )}
      </div>
    </div>
  );
}
