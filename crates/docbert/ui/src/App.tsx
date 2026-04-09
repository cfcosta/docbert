import { Suspense } from "react";
import { Outlet } from "react-router";
import Sidebar from "./components/Sidebar";
import { SearchSessionProvider } from "./pages/search-session";
import "./App.css";

export default function App() {
  return (
    <SearchSessionProvider>
      <Sidebar />
      <main className="main-content">
        <Suspense
          fallback={
            <div className="main-loading" role="status" aria-live="polite">
              <div className="main-loading-panel">
                <p className="main-loading-kicker">Loading workspace</p>
                <h2 className="main-loading-title">Preparing the next surface</h2>
                <div className="main-loading-dots" aria-hidden="true">
                  <span className="main-loading-dot main-loading-dot-1" />
                  <span className="main-loading-dot main-loading-dot-2" />
                  <span className="main-loading-dot main-loading-dot-3" />
                </div>
                <p className="main-loading-copy">
                  docbert keeps the shell responsive while chat, search, and document tools load.
                </p>
              </div>
            </div>
          }
        >
          <Outlet />
        </Suspense>
      </main>
    </SearchSessionProvider>
  );
}
