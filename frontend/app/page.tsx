"use client";

import React, { useCallback, useState } from "react";
import {
  RagAnswer,
  Repository,
  Task,
  askCodebase,
  isApiError,
  submitTask,
} from "@/lib/api";
import RepositoryPicker from "./components/RepositoryPicker";
import TaskCreateForm from "./components/TaskCreateForm";
import TaskProgress from "./components/TaskProgress";
import ReviewPanel from "./components/ReviewPanel";
import StatusBadge from "./components/StatusBadge";

const TERMINAL_STATUSES = new Set(["approved", "rejected", "failed"]);

export default function Home() {
  // Repository selection (Section 1)
  const [selectedRepo, setSelectedRepo] = useState<Repository | null>(null);

  // Code-aware RAG (Section 2)
  const [query, setQuery] = useState("");
  const [ragAnswer, setRagAnswer] = useState("");
  const [citations, setCitations] = useState<string[]>([]);
  const [isAsking, setIsAsking] = useState(false);
  const [ragError, setRagError] = useState<string | null>(null);

  // AI Coding Agent (Section 3)
  const [task, setTask] = useState<Task | null>(null);
  const [creating, setCreating] = useState(false);
  const [agentError, setAgentError] = useState<string | null>(null);

  const handleTaskChange = useCallback((updated: Task) => {
    setTask(updated);
  }, []);

  const handleCreateTask = useCallback(
    async (payload: {
      description: string;
      title?: string;
      test_target?: string;
      max_attempts?: number;
    }) => {
      if (!selectedRepo) return;

      setCreating(true);
      setAgentError(null);
      try {
        const created = await submitTask({
          repository_id: selectedRepo.id,
          ...payload,
        });
        setTask(created);
      } catch (err) {
        setAgentError(
          isApiError(err) ? err.detail : "Failed to create the coding task."
        );
      } finally {
        setCreating(false);
      }
    },
    [selectedRepo]
  );

  const handleReviewed = useCallback(
    (status: string) => {
      setTask((prev) => (prev ? { ...prev, status } : prev));
    },
    []
  );

  const handleAskQuestion = async () => {
    if (!selectedRepo) {
      setRagError("Please select a repository first.");
      return;
    }
    setIsAsking(true);
    setRagError(null);
    setRagAnswer("");
    setCitations([]);
    try {
      const data: RagAnswer = await askCodebase({
        query,
        repository_id: selectedRepo.id,
        top_k: 5,
      });
      setRagAnswer(data.answer);
      setCitations(data.citations || []);
    } catch (err) {
      setRagError(
        isApiError(err) ? err.detail : "Failed to query the codebase."
      );
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <main className="max-w-6xl mx-auto p-8 space-y-8">
      {/* Header */}
      <header className="border-b border-slate-800 pb-6">
        <h1 className="text-3xl font-bold tracking-tight text-cyan-400">
          RepoPilot
        </h1>
        <p className="text-slate-400 mt-1">
          Autonomous AI Software Engineer — isolated workspaces, sandboxed
          testing & human-approved patches.
        </p>
      </header>

      {/* Section 1: Repository selection / registration / indexing */}
      <RepositoryPicker selectedRepo={selectedRepo} onSelect={setSelectedRepo} />

      {/* Section 2: Code-Aware RAG */}
      <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
        <h2 className="text-xl font-semibold text-slate-200">
          2. Codebase Q&amp;A (Semantic RAG)
        </h2>
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            type="text"
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-2 text-slate-200 focus:outline-none focus:border-indigo-500"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question (e.g. 'How does OrderService calculate discounts and taxes?')"
          />
          <button
            onClick={handleAskQuestion}
            disabled={isAsking || !query.trim()}
            title={!selectedRepo ? "Select a repository first" : undefined}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium px-6 py-2 rounded-lg transition whitespace-nowrap"
          >
            {isAsking ? "Searching..." : "Ask Codebase"}
          </button>
        </div>

        {ragError && (
          <p className="text-sm text-red-300 bg-red-950/40 border border-red-900 rounded-lg p-3">
            {ragError}
          </p>
        )}

        {ragAnswer && (
          <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 space-y-3">
            <h3 className="font-semibold text-slate-300">Answer:</h3>
            <p className="text-slate-200 text-sm whitespace-pre-wrap leading-relaxed">
              {ragAnswer}
            </p>
            {citations.length > 0 && (
              <div className="pt-2 border-t border-slate-800">
                <span className="text-xs font-semibold text-slate-400">
                  Citations:{" "}
                </span>
                {citations.map((c, i) => (
                  <span
                    key={i}
                    className="inline-block bg-slate-800 text-cyan-300 text-xs px-2 py-1 rounded mr-2 mt-1"
                  >
                    {c}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </section>

      {/* Section 3: AI Coding Agent workflow */}
      <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-5">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <h2 className="text-xl font-semibold text-slate-200">
            3. AI Coding Agent
          </h2>
          {selectedRepo && (
            <span className="text-xs text-slate-500">
              Target:{" "}
              <span className="text-slate-300 font-mono">
                {selectedRepo.name}
              </span>
            </span>
          )}
        </div>

        {/* Step A: describe the task */}
        {!task && !creating && (
          <TaskCreateForm
            repo={selectedRepo}
            onSubmit={handleCreateTask}
          />
        )}

        {/* Step B: agent progress (live or post-run) */}
        {(creating || task) && (
          <TaskProgress
            task={task}
            creating={creating}
            onTaskChange={handleTaskChange}
          />
        )}

        {agentError && (
          <p className="text-sm text-red-300 bg-red-950/40 border border-red-900 rounded-lg p-3">
            {agentError}
          </p>
        )}

        {/* Step C: human review of generated changes */}
        {task && task.status === "human_approval_required" && (
          <ReviewPanel task={task} onReviewed={handleReviewed} />
        )}

        {/* Step D: final outcome */}
        {task && TERMINAL_STATUSES.has(task.status) && (
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pt-1">
            <StatusBadge status={task.status} size="lg" />
            <button
              onClick={() => {
                setTask(null);
                setAgentError(null);
              }}
              className="border border-slate-700 hover:border-slate-500 text-slate-300 text-sm font-medium px-5 py-2 rounded-lg transition"
            >
              Start Another Task
            </button>
          </div>
        )}
      </section>

      <footer className="text-center text-xs text-slate-600 pb-4">
        RepoPilot runs every task inside an isolated copy of your repository —
        the original code is only modified when you approve a fix.
      </footer>
    </main>
  );
}
