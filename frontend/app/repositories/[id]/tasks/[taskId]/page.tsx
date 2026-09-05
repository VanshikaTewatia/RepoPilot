"use client";

import React, { useCallback, useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Bot, RotateCcw } from "lucide-react";
import { Task, isApiError, getTask } from "@/lib/api";
import { useCurrentRepository } from "@/lib/repository-context";
import { isFinalTaskStatus, needsHumanReview } from "@/lib/taskStatus";
import TaskPipeline from "../../../../components/TaskPipeline";
import ReviewPanel from "../../../../components/ReviewPanel";
import StatusBadge from "../../../../components/StatusBadge";

export default function TaskDetailPage() {
  const repo = useCurrentRepository();
  const router = useRouter();
  const params = useParams();
  const taskId = Number(params?.taskId);

  const [task, setTask] = useState<Task | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setLoadError(null);
    getTask(taskId)
      .then((fresh) => {
        if (!cancelled) setTask(fresh);
      })
      .catch((err) => {
        if (!cancelled) {
          setLoadError(isApiError(err) ? err.detail : "Failed to load task.");
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [taskId]);

  const handleTaskChange = useCallback((updated: Task) => {
    setTask(updated);
  }, []);

  const handleReviewed = useCallback((status: string) => {
    setTask((prev) => (prev ? { ...prev, status } : prev));
  }, []);

  if (isLoading) {
    return (
      <main className="flex-1 w-full max-w-5xl mx-auto px-6 py-8">
        <p className="text-sm text-zinc-500">Loading task...</p>
      </main>
    );
  }

  if (loadError || !task) {
    return (
      <main className="flex-1 w-full max-w-5xl mx-auto px-6 py-8">
        <p className="text-sm text-red-400 bg-red-950/40 border border-red-900 rounded-lg p-3">
          {loadError ?? `Task #${taskId} was not found.`}
        </p>
      </main>
    );
  }

  return (
    <main className="flex-1 w-full max-w-5xl mx-auto px-6 py-8 space-y-6">
      <div>
        <h1 className="text-lg font-semibold text-zinc-100 truncate">
          {task.title}
        </h1>
        <p className="text-sm text-zinc-400 mt-1 max-w-2xl whitespace-pre-wrap">
          {task.description}
        </p>
      </div>

      <section className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6 space-y-5 shadow-sm shadow-black/20">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2.5">
            <span className="flex items-center justify-center w-7 h-7 rounded-lg bg-indigo-500/10 text-indigo-400">
              <Bot className="w-4 h-4" />
            </span>
            <h2 className="text-base font-semibold text-zinc-100">
              AI Coding Agent
            </h2>
          </div>
          <span className="text-xs text-zinc-500">
            Target: <span className="text-zinc-300 font-mono">{repo.name}</span>
          </span>
        </div>

        <TaskPipeline task={task} creating={false} onTaskChange={handleTaskChange} />

        {needsHumanReview(task.status) && (
          <ReviewPanel task={task} onReviewed={handleReviewed} />
        )}

        {isFinalTaskStatus(task.status) && (
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pt-1">
            <StatusBadge status={task.status} size="lg" />
            <button
              onClick={() => router.push(`/repositories/${repo.id}/tasks`)}
              className="inline-flex items-center gap-1.5 border border-zinc-700 hover:border-zinc-500 hover:bg-zinc-800/50 text-zinc-300 text-sm font-medium px-4 py-2 rounded-lg transition"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Start Another Task
            </button>
          </div>
        )}
      </section>
    </main>
  );
}
