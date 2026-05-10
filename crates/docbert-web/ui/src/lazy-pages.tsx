import { lazy } from "react";

export const DocumentsPage = lazy(() => import("./pages/Documents"));
export const ChatPage = lazy(() => import("./pages/Chat"));
export const SearchPage = lazy(() => import("./pages/Search"));
export const SettingsPage = lazy(() => import("./pages/Settings"));
