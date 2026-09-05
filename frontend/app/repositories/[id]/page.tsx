"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Bot, Sparkles } from "lucide-react";
import { ApiError, IndexResult, indexRepository } from "@/lib/api";
import { useCurrentRepository } from "@/lib/repository-context";

export default function RepositoryOverviewPage() {
  const repo = useCurrentRepository();

  const [isIndexing, setIsIndexing] = useState(false);
  const [indexMessage, setIndexMessage] = useState<string | null>(null);
  const [indexIsError, setIndexIsError] = useState(false);

  const handleIndex = async () => {
    setIsIndexing(true);
    setIndexMessage(`Indexing "${repo.name}"...`);
    setIndexIsError(false);
    try {
      const result: IndexResult = await indexRepository(repo.id);
      setIndexMessage(
        `Indexed ${result.total_chunks} syntax chunks ` +
          `(${result.new_chunks} new, ${result.reused_chunks} reused).`
      );
    } catch (err) {
      let detail = err instanceof ApiError ? err.detail : "Indexing failed.";
      if (err instanceof ApiError && err.status === 429) {
        detail = `Gemini embedding rate limit reached: ${detail}`;
      }
      setIndexMessage(detail);
      setIndexIsError(true);
    } finally {
      setIsIndexing(false);
    }
  };

  return (
    <main className="flex-1 w-full max-w-5xl mx-auto px-6 py-8 space-y-6">
      <section className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6 space-y-4 shadow-sm shadow-black/20">
        <h2 className="text-base font-semibold text-zinc-100">
          Repository
        </h2>
        <dl className="grid gap-3 sm:grid-cols-2 text-sm">
          <div className="min-w-0">
            <dt className="text-xs text-zinc-500 uppercase tracking-wide">Name</dt>
            <dd className="text-zinc-200 mt-0.5">{repo.name}</dd>
          </div>
          <div className="min-w-0">
            <dt className="text-xs text-zinc-500 uppercase tracking-wide">ID</dt>
            <dd className="text-zinc-200 mt-0.5 font-mono">#{repo.id}</dd>
          </div>
          <div className="min-w-0">
            <dt className="text-xs text-zinc-500 uppercase tracking-wide">
              {repo.remote_url ? "Remote" : "Local path"}
            </dt>
            <dd className="text-zinc-200 mt-0.5 font-mono truncate">
              {repo.remote_url ?? repo.local_path}
            </dd>
          </div>
          <div className="min-w-0">
            <dt className="text-xs text-zinc-500 uppercase tracking-wide">Default branch</dt>
            <dd className="text-zinc-200 mt-0.5 font-mono">{repo.default_branch}</dd>
          </div>
          <div className="min-w-0">
            <dt className="text-xs text-zinc-500 uppercase tracking-wide">Status</dt>
            <dd className="text-zinc-200 mt-0.5">{repo.status}</dd>
          </div>
          {repo.indexed_at && (
            <div className="min-w-0">
              <dt className="text-xs text-zinc-500 uppercase tracking-wide">Indexed at</dt>
              <dd className="text-zinc-200 mt-0.5">
                {new Date(repo.indexed_at).toLocaleString()}
              </dd>
            </div>
          )}
        </dl>

        <div className="pt-2 flex flex-wrap items-center gap-3">
          <button
            onClick={handleIndex}
            disabled={isIndexing}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium px-5 py-2 rounded-lg transition whitespace-nowrap shadow-sm shadow-indigo-950/50"
          >
            {isIndexing ? "Indexing..." : "Index Repository"}
          </button>
        </div>

        {indexMessage && (
          <p
            className={`text-sm font-mono p-3 rounded-lg border ${
              indexIsError
                ? "text-red-300 bg-red-950/40 border-red-900"
                : "text-indigo-300 bg-zinc-950 border-zinc-800"
            }`}
          >
            {indexMessage}
          </p>
        )}
      </section>

      <div className="grid gap-4 sm:grid-cols-2">
        <Link
          href={`/repositories/${repo.id}/qa`}
          className="bg-zinc-900/60 border border-zinc-800 hover:border-zinc-700 rounded-2xl p-5 flex items-center gap-3 transition shadow-sm shadow-black/20"
        >
          <span className="flex items-center justify-center w-9 h-9 rounded-lg bg-indigo-500/10 text-indigo-400 shrink-0">
            <Sparkles className="w-4 h-4" />
          </span>
          <div>
            <h3 className="text-sm font-semibold text-zinc-100">Deep Codebase Q&A</h3>
            <p className="text-xs text-zinc-500 mt-0.5">
              Ask evidence-based questions about this repository.
            </p>
          </div>
        </Link>
        <Link
          href={`/repositories/${repo.id}/tasks`}
          className="bg-zinc-900/60 border border-zinc-800 hover:border-zinc-700 rounded-2xl p-5 flex items-center gap-3 transition shadow-sm shadow-black/20"
        >
          <span className="flex items-center justify-center w-9 h-9 rounded-lg bg-indigo-500/10 text-indigo-400 shrink-0">
            <Bot className="w-4 h-4" />
          </span>
          <div>
            <h3 className="text-sm font-semibold text-zinc-100">Start a Coding Task</h3>
            <p className="text-xs text-zinc-500 mt-0.5">
              Describe a bug or change for the agent to investigate and fix.
            </p>
          </div>
        </Link>
      </div>
    </main>
  );
}
