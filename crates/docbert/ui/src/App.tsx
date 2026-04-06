import { Outlet } from "react-router";
import Sidebar from "./components/Sidebar";
import "./App.css";

export default function App() {
  return (
    <>
      <Sidebar />
      <main className="main-content">
        <Outlet />
      </main>
    </>
  );
}
