"use client";

import React, { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { FolderGit2, Plug, X } from "lucide-react";
import { Repository, isApiError, listRepositories } from "@/lib/api";
import ConnectRepositoryPanel from "./components/ConnectRepositoryPanel";
import StatusBadge from "./components/StatusBadge";

export default function Dashboard() {
  const [repos, setRepos] = useState<Repository[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showConnect, setShowConnect] = useState(false);

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

  const handleConnected = (repo: Repository) => {
    setRepos((prev) => [repo, ...prev]);
    setShowConnect(false);
  };

  return (
    <main className="flex-1 w-full max-w-5xl mx-auto px-6 py-10 space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4 pb-2">
        <div>
          <h1 className="text-xl font-semibold text-zinc-100">
            Your development workspace
          </h1>
          <p className="text-sm text-zinc-400 mt-1 max-w-2xl">
            Connect a repository to index it, ask evidence-based questions
            about the codebase, and let RepoPilot investigate, plan, and
            verify fixes inside an isolated workspace before you approve them.
          </p>
        </div>
        {repos.length > 0 && (
          <button
            onClick={() => setShowConnect((v) => !v)}
            className="inline-flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium px-4 py-2 rounded-lg transition whitespace-nowrap"
          >
            {showConnect ? (
              <>
                <X className="w-3.5 h-3.5" />
                Cancel
              </>
            ) : (
              <>
                <Plug className="w-3.5 h-3.5" />
                Connect Repository
              </>
            )}
          </button>
        )}
      </div>

      {showConnect && <ConnectRepositoryPanel onConnected={handleConnected} />}

      {error && (
        <p className="text-sm text-red-400 bg-red-950/40 border border-red-900 rounded-lg p-3">
          {error}{" "}
          <button
            onClick={() => void loadRepositories()}
            className="underline hover:text-red-300"
          >
            Retry
          </button>
        </p>
      )}

      {isLoading && (
        <p className="text-sm text-zinc-500">Loading repositories...</p>
      )}

      {!isLoading && !error && repos.length === 0 && !showConnect && (
        <div className="border border-dashed border-zinc-800 rounded-2xl p-12 flex flex-col items-center text-center gap-3">
          <span className="flex items-center justify-center w-11 h-11 rounded-xl bg-indigo-500/10 text-indigo-400">
            <FolderGit2 className="w-5 h-5" />
          </span>
          <h2 className="text-base font-semibold text-zinc-100">
            Connect your first repository
          </h2>
          <p className="text-sm text-zinc-500 max-w-sm">
            Register a local path or clone from GitHub to start indexing,
            asking questions, and running the coding agent against it.
          </p>
          <button
            onClick={() => setShowConnect(true)}
            className="mt-2 inline-flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium px-5 py-2 rounded-lg transition"
          >
            <Plug className="w-3.5 h-3.5" />
            Connect Repository
          </button>
        </div>
      )}

      {repos.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2">
          {repos.map((repo) => (
            <Link
              key={repo.id}
              href={`/repositories/${repo.id}`}
              className="bg-zinc-900/60 border border-zinc-800 hover:border-zinc-700 rounded-2xl p-5 space-y-3 transition shadow-sm shadow-black/20"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-2.5 min-w-0">
                  <span className="flex items-center justify-center w-8 h-8 rounded-lg bg-indigo-500/10 text-indigo-400 shrink-0">
                    <FolderGit2 className="w-4 h-4" />
                  </span>
                  <div className="min-w-0">
                    <h3 className="text-sm font-semibold text-zinc-100 truncate">
                      {repo.name}
                    </h3>
                    <p className="text-xs text-zinc-500 truncate">
                      {repo.remote_url ?? repo.local_path}
                    </p>
                  </div>
                </div>
                <StatusBadge status={repo.status} size="sm" />
              </div>
              <p className="text-[11px] font-mono text-zinc-600">
                #{repo.id} · {repo.default_branch}
              </p>
            </Link>
          ))}
        </div>
      )}
    </main>
  );
}
