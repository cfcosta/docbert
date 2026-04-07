import { Link } from "react-router";

import { buildDocumentTabHref, type SearchExcerpt, type SearchResult } from "../lib/api";
import "./SearchResults.css";

type SearchResultsProps = {
  results: SearchResult[];
  onOpenDocument?: (result: SearchResult) => void;
  activeDocumentKey?: string | null;
};

function formatExcerptRange(excerpt: SearchExcerpt): string {
  if (excerpt.start_line === excerpt.end_line) {
    return `${excerpt.start_line}`;
  }
  return `${excerpt.start_line}–${excerpt.end_line}`;
}

function documentKey(result: Pick<SearchResult, "collection" | "path">): string {
  return `${result.collection}:${result.path}`;
}

export default function SearchResults({
  results,
  onOpenDocument,
  activeDocumentKey = null,
}: SearchResultsProps) {
  if (results.length === 0) {
    return <div className="chat-tool-search-empty">No results</div>;
  }

  return (
    <div className="chat-tool-search-results">
      {results.map((result) => {
        const href = buildDocumentTabHref(result.collection, result.path);
        const isActive = activeDocumentKey === documentKey(result);

        return (
          <div key={`${result.collection}:${result.path}`} className="chat-tool-search-result">
            <div className={`chat-tool-search-result-node${isActive ? " active" : ""}`}>
              <div className="chat-tool-search-result-top">
                <div>
                  {onOpenDocument ? (
                    <button
                      type="button"
                      className="chat-tool-search-result-title chat-tool-search-result-title-button"
                      onClick={() => onOpenDocument(result)}
                    >
                      {result.title || result.path}
                    </button>
                  ) : (
                    <Link
                      className="chat-tool-search-result-title chat-tool-search-result-title-link"
                      to={href}
                    >
                      {result.title || result.path}
                    </Link>
                  )}
                  <div className="chat-tool-search-result-path">
                    {result.collection}/{result.path}
                  </div>
                </div>
              </div>
              <div className="chat-tool-search-result-meta">
                <span>#{result.rank}</span>
                <span>{result.score.toFixed(3)}</span>
              </div>
            </div>

            {result.excerpts && result.excerpts.length > 0 && (
              <div className="chat-tool-search-result-children">
                {result.excerpts.map((excerpt) => {
                  const excerptKey = `${result.collection}:${result.path}:${excerpt.start_line}:${excerpt.end_line}`;
                  const excerptClassName = `chat-tool-search-excerpt${isActive ? " active" : ""}`;

                  return onOpenDocument ? (
                    <button
                      key={excerptKey}
                      type="button"
                      className={`${excerptClassName} chat-tool-search-excerpt-button`}
                      onClick={() => onOpenDocument(result)}
                    >
                      <span className="chat-tool-search-excerpt-range">
                        {formatExcerptRange(excerpt)}
                      </span>
                      <span className="chat-tool-search-excerpt-text">{excerpt.text}</span>
                    </button>
                  ) : (
                    <Link
                      key={excerptKey}
                      className={`${excerptClassName} chat-tool-search-excerpt-link`}
                      to={href}
                    >
                      <span className="chat-tool-search-excerpt-range">
                        {formatExcerptRange(excerpt)}
                      </span>
                      <span className="chat-tool-search-excerpt-text">{excerpt.text}</span>
                    </Link>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
