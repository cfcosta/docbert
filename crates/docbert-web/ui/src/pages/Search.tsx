import { useEffect, type FormEvent } from "react";

import SearchResults from "../components/SearchResults";
import { api } from "../lib/api";
import { useSearchSession } from "./search-session";
import "./Search.css";

const ALL_COLLECTIONS_VALUE = "";

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
    if (trimmedQuery) {
      return;
    }

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
  }, [
    collectionsError,
    lastIssuedCollection,
    lastIssuedMode,
    lastIssuedQuery,
    loadingCollections,
    query,
    resultCount,
    results,
    searchError,
    searching,
    setSearchSession,
  ]);

  const trimmedQuery = query.trim();
  const selectedCollectionValue = selectedCollection || null;
  const hasPendingChanges =
    Boolean(trimmedQuery) &&
    (lastIssuedQuery !== trimmedQuery ||
      lastIssuedMode !== mode ||
      lastIssuedCollection !== selectedCollectionValue);

  const submitBlocked =
    loadingCollections || searching || collectionsError !== null || !trimmedQuery;

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (submitBlocked) {
      return;
    }

    const requestId = latestSearchRequestId + 1;

    setSearchSession((previous) => ({
      ...previous,
      latestSearchRequestId: requestId,
      searching: true,
      searchError: null,
      resultCount: null,
      results: [],
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
        setSearchSession((previous) => {
          if (previous.latestSearchRequestId !== requestId) {
            return previous;
          }
          return {
            ...previous,
            resultCount: response.result_count,
            results: response.results,
            searchError: null,
          };
        });
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
  };

  return (
    <div className={`search-page${trimmedQuery ? " search-page-has-query" : ""}`}>
      <header className="search-header">Search</header>
      <div className="search-body">
        <form className="search-controls" aria-label="Search controls" onSubmit={handleSubmit}>
          <div className="search-query-shell">
            <div
              className="search-query-row"
              onClick={(event) => {
                if (event.target === event.currentTarget) {
                  event.currentTarget.querySelector("input")?.focus();
                }
              }}
            >
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

            <div className="search-query-filters">
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

              <button type="submit" className="search-submit-button" disabled={submitBlocked}>
                {searching ? "Searching…" : "Run search"}
              </button>
            </div>
          </div>
        </form>

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
        ) : !trimmedQuery || hasPendingChanges ? null : searching ? (
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
