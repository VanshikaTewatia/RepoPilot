"use client";

import React, { createContext, useContext } from "react";
import { Repository } from "@/lib/api";

const RepositoryContext = createContext<Repository | null | undefined>(undefined);

export function RepositoryProvider({
  repository,
  children,
}: {
  repository: Repository;
  children: React.ReactNode;
}) {
  return (
    <RepositoryContext.Provider value={repository}>
      {children}
    </RepositoryContext.Provider>
  );
}

/** Must be called within a `RepositoryProvider` (i.e. under
 * `app/repositories/[id]/layout.tsx`). Throws otherwise so a missing
 * provider fails loudly instead of silently rendering with `null`. */
export function useCurrentRepository(): Repository {
  const repo = useContext(RepositoryContext);
  if (repo === undefined) {
    throw new Error("useCurrentRepository must be used within a RepositoryProvider");
  }
  if (repo === null) {
    throw new Error("useCurrentRepository called with no repository loaded");
  }
  return repo;
}
