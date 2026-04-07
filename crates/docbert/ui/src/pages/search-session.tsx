import { createContext, useContext, useMemo, useState, type ReactNode } from "react";

import type { Collection, SearchMode, SearchResult } from "../lib/api";

export interface SearchSessionState {
  query: string;
  mode: SearchMode;
  selectedCollection: string;
  collections: Collection[];
  loadingCollections: boolean;
  collectionsError: string | null;
  searching: boolean;
  searchError: string | null;
  resultCount: number | null;
  results: SearchResult[];
  latestSearchRequestId: number;
}

export const initialSearchSessionState: SearchSessionState = {
  query: "",
  mode: "hybrid",
  selectedCollection: "",
  collections: [],
  loadingCollections: true,
  collectionsError: null,
  searching: false,
  searchError: null,
  resultCount: null,
  results: [],
  latestSearchRequestId: 0,
};

interface SearchSessionContextValue {
  searchSession: SearchSessionState;
  setSearchSession: React.Dispatch<React.SetStateAction<SearchSessionState>>;
}

const SearchSessionContext = createContext<SearchSessionContextValue | null>(null);

export function SearchSessionProvider({ children }: { children: ReactNode }) {
  const [searchSession, setSearchSession] = useState(initialSearchSessionState);

  const value = useMemo(
    () => ({ searchSession, setSearchSession }),
    [searchSession, setSearchSession],
  );

  return <SearchSessionContext.Provider value={value}>{children}</SearchSessionContext.Provider>;
}

export function useSearchSession() {
  const context = useContext(SearchSessionContext);
  if (!context) {
    throw new Error("useSearchSession must be used within SearchSessionProvider");
  }
  return context;
}
