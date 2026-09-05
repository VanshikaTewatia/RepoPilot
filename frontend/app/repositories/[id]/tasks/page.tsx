"use client";

import React, { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { Bot } from "lucide-react";
import { isApiError, submitTask } from "@/lib/api";
import { useCurrentRepository } from "@/lib/repository-context";
import TaskCreateForm, { TaskFormPayload } from "../../../components/TaskCreateForm";

export default function CreateTaskPage() {
  const repo = useCurrentRepository();
  const router = useRouter();

  const [creating, setCreating] = useState(false);
  const [agentError, setAgentError] = useState<string | null>(null);

  const handleCreateTask = useCallback(
    async (payload: TaskFormPayload) => {
      setCreating(true);
      setAgentError(null);
      try {
        const created = await submitTask({
          repository_id: repo.id,
          ...payload,
        });
        router.push(`/repositories/${repo.id}/tasks/${created.id}`);
      } catch (err) {
        setAgentError(
          isApiError(err) ? err.detail : "Failed to create the coding task."
        );
        setCreating(false);
      }
    },
    [repo.id, router]
  );

  return (
    <main className="flex-1 w-full max-w-5xl mx-auto px-6 py-8 space-y-6">
      <div>
        <h1 className="text-lg font-semibold text-zinc-100">Start a Coding Task</h1>
        <p className="text-sm text-zinc-400 mt-1 max-w-2xl">
          Describe a bug or a change. RepoPilot runs inside an isolated copy
          of this repository — investigating, planning, editing, and
          verifying with tests — and nothing touches your original code until
          you approve the result.
        </p>
      </div>

      <section className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6 space-y-5 shadow-sm shadow-black/20">
        <div className="flex items-center gap-2.5">
          <span className="flex items-center justify-center w-7 h-7 rounded-lg bg-indigo-500/10 text-indigo-400">
            <Bot className="w-4 h-4" />
          </span>
          <h2 className="text-base font-semibold text-zinc-100">
            AI Coding Agent
          </h2>
        </div>

        {creating ? (
          <p className="text-sm text-indigo-300 bg-zinc-950 border border-zinc-800 rounded-lg p-4 inline-flex items-center gap-2">
            <span className="w-3 h-3 rounded-full border-2 border-indigo-400 border-t-transparent animate-spin" />
            Agent is working on the repository copy — this can take up to a
            few minutes. You&apos;ll be taken to the task once it starts.
          </p>
        ) : (
          <TaskCreateForm repo={repo} onSubmit={handleCreateTask} />
        )}

        {agentError && (
          <p className="text-sm text-red-300 bg-red-950/40 border border-red-900 rounded-lg p-3">
            {agentError}
          </p>
        )}
      </section>
    </main>
  );
}
