"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useParams } from "next/navigation";
import { FolderGit2 } from "lucide-react";
import { Repository, isApiError, listRepositories } from "@/lib/api";
import { RepositoryProvider } from "@/lib/repository-context";
import StatusBadge from "../../components/StatusBadge";

const TABS = [
  { label: "Overview", suffix: "" },
  { label: "Q&A", suffix: "/qa" },
  { label: "Tasks", suffix: "/tasks" },
];

export default function RepositoryLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const params = useParams();
  const pathname = usePathname();
  const repoId = Number(params?.id);

  const [repo, setRepo] = useState<Repository | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setError(null);
    listRepositories()
      .then((repos) => {
        if (cancelled) return;
        const match = repos.find((r) => r.id === repoId) ?? null;
        setRepo(match);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(isApiError(err) ? err.detail : "Failed to load repository.");
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [repoId]);

  if (isLoading) {
    return (
      <main className="flex-1 w-full max-w-5xl mx-auto px-6 py-10">
        <p className="text-sm text-zinc-500">Loading repository...</p>
      </main>
    );
  }

  if (error || !repo) {
    return (
      <main className="flex-1 w-full max-w-5xl mx-auto px-6 py-10 space-y-3">
        <p className="text-sm text-red-400 bg-red-950/40 border border-red-900 rounded-lg p-3">
          {error ?? `Repository #${repoId} was not found.`}
        </p>
        <Link href="/" className="text-sm text-indigo-400 hover:text-indigo-300 underline">
          Back to Dashboard
        </Link>
      </main>
    );
  }

  const basePath = `/repositories/${repo.id}`;

  return (
    <RepositoryProvider repository={repo}>
      <div className="flex-1 flex flex-col">
        <div className="border-b border-zinc-800/80 bg-zinc-950/85 backdrop-blur supports-[backdrop-filter]:bg-zinc-950/70 sticky top-0 z-10">
          <div className="max-w-5xl mx-auto px-6 pt-6">
            <Link
              href="/"
              className="text-xs text-zinc-500 hover:text-zinc-300 transition"
            >
              Dashboard
            </Link>
            <div className="flex items-center gap-3 mt-1.5 pb-4">
              <span className="flex items-center justify-center w-8 h-8 rounded-lg bg-indigo-500/10 text-indigo-400 shrink-0">
                <FolderGit2 className="w-4 h-4" />
              </span>
              <h1 className="text-lg font-semibold text-zinc-100 truncate">
                {repo.name}
              </h1>
              <StatusBadge status={repo.status} size="sm" />
              <span className="text-xs font-mono text-zinc-600 truncate hidden sm:inline">
                {repo.remote_url ?? repo.local_path}
              </span>
            </div>
            <nav className="flex gap-1 -mb-px">
              {TABS.map((tab) => {
                const href = `${basePath}${tab.suffix}`;
                const active =
                  tab.suffix === ""
                    ? pathname === basePath
                    : pathname.startsWith(href);
                return (
                  <Link
                    key={tab.label}
                    href={href}
                    className={`px-3.5 py-2 text-sm font-medium border-b-2 transition ${
                      active
                        ? "border-indigo-500 text-zinc-100"
                        : "border-transparent text-zinc-500 hover:text-zinc-200"
                    }`}
                  >
                    {tab.label}
                  </Link>
                );
              })}
            </nav>
          </div>
        </div>
        <div className="flex-1 flex flex-col">{children}</div>
      </div>
    </RepositoryProvider>
  );
}
