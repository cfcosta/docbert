import { Link } from "react-router";

import { buildDocumentTabHref, type SearchExcerpt, type SearchResult } from "../lib/api";
import "./SearchResults.css";

function formatExcerptRange(excerpt: SearchExcerpt): string {
  if (excerpt.start_line === excerpt.end_line) {
    return `${excerpt.start_line}`;
  }
  return `${excerpt.start_line}–${excerpt.end_line}`;
}

export default function SearchResults({ results }: { results: SearchResult[] }) {
  if (results.length === 0) {
    return <div className="chat-tool-search-empty">No results</div>;
  }

  return (
    <div className="chat-tool-search-results">
      {results.map((result) => (
        <div key={`${result.collection}:${result.path}`} className="chat-tool-search-result">
          <div className="chat-tool-search-result-node">
            <div className="chat-tool-search-result-top">
              <div>
                <Link
                  className="chat-tool-search-result-title chat-tool-search-result-title-link"
                  to={buildDocumentTabHref(result.collection, result.path)}
                >
                  {result.title || result.path}
                </Link>
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
              {result.excerpts.map((excerpt) => (
                <Link
                  key={`${result.collection}:${result.path}:${excerpt.start_line}:${excerpt.end_line}`}
                  className="chat-tool-search-excerpt chat-tool-search-excerpt-link"
                  to={buildDocumentTabHref(result.collection, result.path)}
                >
                  <span className="chat-tool-search-excerpt-range">
                    {formatExcerptRange(excerpt)}
                  </span>
                  <span className="chat-tool-search-excerpt-text">{excerpt.text}</span>
                </Link>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
