"use client";

import React, { useCallback, useState } from "react";
import {
  Repository,
  Task,
  isApiError,
  submitTask,
} from "@/lib/api";
import RepositoryPicker from "./components/RepositoryPicker";
import DeepQAPanel from "./components/DeepQAPanel";
import TaskCreateForm from "./components/TaskCreateForm";
import TaskProgress from "./components/TaskProgress";
import ReviewPanel from "./components/ReviewPanel";
import StatusBadge from "./components/StatusBadge";
import { isFinalTaskStatus, needsHumanReview } from "@/lib/taskStatus";

export default function Home() {
  // Repository selection (Section 1)
  const [selectedRepo, setSelectedRepo] = useState<Repository | null>(null);

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

  return (
    <main className="max-w-6xl mx-auto p-8 space-y-8">
      {/* Header */}
      <header className="border-b border-slate-800 pb-6">
        <h1 className="text-3xl font-bold tracking-tight text-cyan-400">
          RepoPilot
        </h1>
        <p className="text-slate-400 mt-1">
          Autonomous AI Software Engineer — isolated workspaces, sandboxed
          testing, human-approved patches, and automatic Pull Requests for
          GitHub-connected repositories.
        </p>
      </header>

      {/* Section 1: Repository selection / registration / indexing */}
      <RepositoryPicker selectedRepo={selectedRepo} onSelect={setSelectedRepo} />

      {/* Section 2: Deep Codebase Q&A */}
      <DeepQAPanel selectedRepo={selectedRepo} />

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

        {/* Step C: human review of generated changes (also re-shown after a
            push/PR failure so the verified fix can be retried without
            re-running the agent) */}
        {task && needsHumanReview(task.status) && (
          <ReviewPanel task={task} onReviewed={handleReviewed} />
        )}

        {/* Step D: final outcome */}
        {task && isFinalTaskStatus(task.status) && (
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
        the original code is only modified (and, for GitHub repositories,
        only branched, pushed, and opened as a Pull Request) once you approve
        a fix.
      </footer>
    </main>
  );
}
