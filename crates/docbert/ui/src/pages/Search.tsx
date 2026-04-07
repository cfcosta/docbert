import "./Search.css";

export default function Search() {
  return (
    <div className="search-page">
      <header className="search-header">
        <h2>Search</h2>
        <p className="search-subtitle">Find documents across your indexed collections.</p>
      </header>

      <div className="search-body">
        <div className="search-empty-state">
          <p>Search will appear here once query, collection, and mode controls are connected.</p>
        </div>
      </div>
    </div>
  );
}
