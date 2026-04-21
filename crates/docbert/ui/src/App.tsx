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
              <div className="main-loading-dots" aria-hidden="true">
                <span className="main-loading-dot main-loading-dot-1" />
                <span className="main-loading-dot main-loading-dot-2" />
                <span className="main-loading-dot main-loading-dot-3" />
              </div>
              <span className="sr-only">Loading</span>
            </div>
          }
        >
          <Outlet />
        </Suspense>
      </main>
    </SearchSessionProvider>
  );
}
