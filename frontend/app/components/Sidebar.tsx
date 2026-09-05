"use client";

import React, { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useParams } from "next/navigation";
import { FolderGit2, LayoutGrid, Menu, Plug, X } from "lucide-react";
import { Repository, isApiError, listRepositories } from "@/lib/api";

function repoStatusDot(status: string): string {
  return status === "indexed" ? "bg-emerald-500" : "bg-zinc-600";
}

function SidebarNav({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  const params = useParams();
  const activeRepoId = params?.id ? Number(params.id) : null;

  const [repos, setRepos] = useState<Repository[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadRepositories = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      setRepos(await listRepositories());
    } catch (err) {
      setError(isApiError(err) ? err.detail : "Failed to load repositories.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadRepositories();
  }, [loadRepositories]);

  const isDashboard = pathname === "/";

  return (
    <>
      <div className="h-14 flex items-center gap-2.5 px-5 border-b border-zinc-800/80 shrink-0">
        <div className="w-6 h-6 rounded-md bg-indigo-500 flex items-center justify-center text-white text-[13px] font-bold">
          R
        </div>
        <span className="text-sm font-semibold tracking-tight text-zinc-100">
          RepoPilot
        </span>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-4">
        <Link
          href="/"
          onClick={onNavigate}
          className={`flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm font-medium transition ${
            isDashboard
              ? "bg-indigo-500/10 text-indigo-300"
              : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900"
          }`}
        >
          <LayoutGrid className="w-4 h-4" />
          Dashboard
        </Link>

        <div>
          <div className="flex items-center justify-between px-2.5 mb-1.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-600">
              Repositories
            </span>
          </div>

          {isLoading && (
            <p className="px-2.5 py-1.5 text-xs text-zinc-600">Loading...</p>
          )}
          {error && (
            <p className="px-2.5 py-1.5 text-xs text-red-400">{error}</p>
          )}
          {!isLoading && !error && repos.length === 0 && (
            <p className="px-2.5 py-1.5 text-xs text-zinc-600">
              No repositories yet
            </p>
          )}

          <ul className="space-y-0.5">
            {repos.map((repo) => {
              const active = activeRepoId === repo.id;
              return (
                <li key={repo.id}>
                  <Link
                    href={`/repositories/${repo.id}`}
                    onClick={onNavigate}
                    className={`flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-sm transition ${
                      active
                        ? "bg-indigo-500/10 text-indigo-300 font-medium"
                        : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900"
                    }`}
                  >
                    <FolderGit2 className="w-3.5 h-3.5 shrink-0" />
                    <span className="truncate flex-1">{repo.name}</span>
                    <span
                      className={`w-1.5 h-1.5 rounded-full shrink-0 ${repoStatusDot(repo.status)}`}
                      title={repo.status}
                    />
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
      </nav>

      <div className="p-3 border-t border-zinc-800/80 shrink-0">
        <Link
          href="/"
          onClick={onNavigate}
          className="flex items-center justify-center gap-1.5 border border-zinc-700 hover:border-zinc-500 hover:bg-zinc-900 text-zinc-300 text-xs font-medium px-3 py-2 rounded-lg transition w-full"
        >
          <Plug className="w-3.5 h-3.5" />
          Connect Repository
        </Link>
      </div>
    </>
  );
}

export default function Sidebar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  // Close the mobile drawer whenever the route changes.
  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  return (
    <>
      {/* Desktop: persistent sidebar */}
      <aside className="hidden md:flex w-64 shrink-0 border-r border-zinc-800/80 bg-zinc-950 flex-col h-screen sticky top-0">
        <SidebarNav />
      </aside>

      {/* Mobile: compact top bar with a menu toggle */}
      <div className="md:hidden sticky top-0 z-30 h-14 shrink-0 flex items-center justify-between px-4 border-b border-zinc-800/80 bg-zinc-950/95 backdrop-blur">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-md bg-indigo-500 flex items-center justify-center text-white text-[13px] font-bold">
            R
          </div>
          <span className="text-sm font-semibold tracking-tight text-zinc-100">
            RepoPilot
          </span>
        </div>
        <button
          onClick={() => setMobileOpen(true)}
          aria-label="Open navigation"
          className="p-2 -mr-2 text-zinc-400 hover:text-zinc-100"
        >
          <Menu className="w-5 h-5" />
        </button>
      </div>

      {/* Mobile: slide-over drawer */}
      {mobileOpen && (
        <div className="md:hidden fixed inset-0 z-40 flex">
          <div
            className="absolute inset-0 bg-black/60"
            onClick={() => setMobileOpen(false)}
          />
          <aside className="relative w-72 max-w-[80vw] bg-zinc-950 border-r border-zinc-800/80 flex flex-col h-full">
            <button
              onClick={() => setMobileOpen(false)}
              aria-label="Close navigation"
              className="absolute top-3 right-3 p-2 text-zinc-400 hover:text-zinc-100"
            >
              <X className="w-4 h-4" />
            </button>
            <SidebarNav onNavigate={() => setMobileOpen(false)} />
          </aside>
        </div>
      )}
    </>
  );
}
