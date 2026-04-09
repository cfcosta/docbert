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
        <Suspense fallback={<div className="main-loading">Loading workspace…</div>}>
          <Outlet />
        </Suspense>
      </main>
    </SearchSessionProvider>
  );
}
