"use client";

import React, { useCallback, useEffect, useState } from "react";
import { Check, ExternalLink, X } from "lucide-react";
import {
  ApiError,
  Task,
  TaskDiff,
  approveTask,
  getTask,
  getTaskDiff,
  isApiError,
  rejectTask,
} from "@/lib/api";
import { needsHumanReview } from "@/lib/taskStatus";
import StatusBadge from "./StatusBadge";

interface ReviewPanelProps {
  task: Task;
  /** Called after a successful approve/reject with the new task status. */
  onReviewed: (status: string) => void;
}

/** Classify a raw unified-diff line for coloring. Content is never modified. */
function lineClass(line: string): string {
  if (line.startsWith("diff --git") || line.startsWith("index ")) {
    return "text-zinc-500";
  }
  if (line.startsWith("--- ") || line.startsWith("+++ ")) {
    return "text-zinc-400";
  }
  if (line.startsWith("@@")) {
    return "text-indigo-400 bg-indigo-950/30";
  }
  if (line.startsWith("+")) {
    return "text-emerald-300 bg-emerald-950/40";
  }
  if (line.startsWith("-")) {
    return "text-red-300 bg-red-950/40";
  }
  return "text-zinc-300";
}

export default function ReviewPanel({ task, onReviewed }: ReviewPanelProps) {
  const [diffData, setDiffData] = useState<TaskDiff | null>(null);
  const [isLoadingDiff, setIsLoadingDiff] = useState(true);
  const [diffError, setDiffError] = useState<string | null>(null);

  // Authoritative task refresh: the prop snapshot can be stale (e.g. missing
  // test_output), so the task is re-fetched from the server on mount.
  const [refreshedTask, setRefreshedTask] = useState<Task | null>(null);
  const [isLoadingTask, setIsLoadingTask] = useState(true);
  const [taskLoadError, setTaskLoadError] = useState<string | null>(null);

  const [conflictMessage, setConflictMessage] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [acting, setActing] = useState<"approve" | "reject" | null>(null);
  const [prUrl, setPrUrl] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setIsLoadingTask(true);
    setTaskLoadError(null);
    getTask(task.id)
      .then((fresh) => {
        if (!cancelled) setRefreshedTask(fresh);
      })
      .catch((err) => {
        if (!cancelled) {
          setTaskLoadError(
            isApiError(err) ? err.detail : "Failed to load task details."
          );
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoadingTask(false);
      });
    return () => {
      cancelled = true;
    };
  }, [task.id]);


  const loadDiff = useCallback(async () => {
    setIsLoadingDiff(true);
    setDiffError(null);
    try {
      const data = await getTaskDiff(task.id);
      setDiffData(data);
    } catch (err) {
      setDiffError(
        isApiError(err) ? err.detail : "Failed to load the generated diff."
      );
    } finally {
      setIsLoadingDiff(false);
    }
  }, [task.id]);

  useEffect(() => {
    void loadDiff();
  }, [loadDiff]);

  const handleApprove = async () => {
    setActing("approve");
    setConflictMessage(null);
    setActionError(null);
    setPrUrl(null);
    try {
      const result = await approveTask(task.id);
      if (result.pr_url) setPrUrl(result.pr_url);
      onReviewed(result.status);
    } catch (err) {
      if (err instanceof ApiError && err.isConflict) {
        // Patch no longer applies cleanly to the original repository.
        // Stay in review state and surface the conflict details.
        setConflictMessage(err.detail);
      } else {
        setActionError(
          isApiError(err) ? err.detail : "Approval failed unexpectedly."
        );
        // A push/PR failure (GitHub flow) moves the task to "approval_failed"
        // server-side even though this request threw -- refresh so the badge
        // and retry affordance reflect that instead of a stale status.
        try {
          const fresh = await getTask(task.id);
          setRefreshedTask(fresh);
        } catch {
          // Best-effort only; the actionError message above already explains what happened.
        }
      }
    } finally {
      setActing(null);
    }
  };

  const handleReject = async () => {
    if (
      !window.confirm(
        "Reject this fix?\n\nThe isolated workspace will be deleted and the original repository will remain untouched."
      )
    ) {
      return;
    }

    setActing("reject");
    setConflictMessage(null);
    setActionError(null);
    try {
      const result = await rejectTask(task.id);
      onReviewed(result.status);
    } catch (err) {
      setActionError(
        isApiError(err) ? err.detail : "Rejection failed unexpectedly."
      );
    } finally {
      setActing(null);
    }
  };

  const diffLines = diffData?.diff ? diffData.diff.split("\n") : [];
  // Drop only the trailing empty element produced by the final newline.
  while (diffLines.length > 0 && diffLines[diffLines.length - 1] === "") {
    diffLines.pop();
  }

  // Server-fetched task data wins once loaded; fall back to the prop snapshot
  // while loading or if the refresh failed.
  const currentTask = refreshedTask ?? task;
  const isPendingReview = needsHumanReview(currentTask.status);
  const resolvedPrUrl = prUrl ?? currentTask.pr_url ?? null;

  return (
    <div className="space-y-4 border border-amber-900/50 rounded-2xl p-5 bg-amber-500/[0.04] shadow-sm shadow-black/20">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-semibold text-amber-200">
          Review Generated Changes
        </h3>
        <StatusBadge status={currentTask.status} size="sm" />
      </div>

      <p className="text-sm text-zinc-300">
        The agent verified this fix against the full test suite in an isolated
        workspace. Your original repository has not been modified yet — approve
        to apply the patch (pushing a branch and opening a Pull Request for
        GitHub-connected repositories), or reject to discard it.
      </p>

      {currentTask.status === "approval_failed" && (
        <div className="border border-orange-800 bg-orange-950/30 rounded-xl p-4 space-y-1">
          <p className="text-sm font-semibold text-orange-300">
            The verified fix could not be pushed / opened as a Pull Request.
          </p>
          <p className="text-xs text-zinc-400">
            The patch itself was verified and is unchanged — the isolated
            workspace and stored diff are preserved. Your repository was
            rolled back to its original branch with nothing partially
            applied. Fix the underlying issue (e.g. configure{" "}
            <code className="text-zinc-300">GITHUB_TOKEN</code> on the
            server) and press Approve again to retry.
          </p>
        </div>
      )}

      {resolvedPrUrl && (
        <div className="border border-emerald-800 bg-emerald-950/30 rounded-xl p-4 flex items-center justify-between gap-3 flex-wrap">
          <p className="text-sm text-emerald-300">
            Pull Request created successfully.
          </p>
          <a
            href={resolvedPrUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-sm font-semibold text-emerald-200 underline hover:text-emerald-100 whitespace-nowrap"
          >
            View Pull Request
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      )}

      {isLoadingTask && (
        <p className="text-xs text-zinc-500 animate-pulse">
          Refreshing task details...
        </p>
      )}
      {taskLoadError && (
        <p className="text-xs text-amber-400/80">
          Could not refresh task details ({taskLoadError}). Showing cached
          details instead.
        </p>
      )}

      {/* Conflict banner (HTTP 409 from approve) */}
      {conflictMessage && (
        <div className="border border-red-800 bg-red-950/40 rounded-xl p-4 space-y-1">
          <p className="text-sm font-semibold text-red-300">
            Patch conflict — the repository changed since this fix was
            generated. Nothing was applied.
          </p>
          <p className="text-xs font-mono text-red-200/80 break-all">
            {conflictMessage}
          </p>
          <p className="text-xs text-zinc-400 pt-1">
            Resolve the conflict in the repository, then retry approval below.
          </p>
        </div>
      )}

      {actionError && (
        <p className="text-sm text-red-300 bg-red-950/40 border border-red-900 rounded-lg p-3">
          {actionError}
        </p>
      )}

      {/* Changed files */}
      {isLoadingDiff ? (
        <p className="text-sm text-zinc-400 animate-pulse">Loading diff...</p>
      ) : diffError ? (
        <div className="text-sm text-red-300 bg-red-950/40 border border-red-900 rounded-lg p-3 flex items-center justify-between gap-3">
          <span>{diffError}</span>
          <button
            onClick={() => void loadDiff()}
            className="underline hover:text-red-200 whitespace-nowrap"
          >
            Retry
          </button>
        </div>
      ) : (
        <>
          {diffData && diffData.changed_files.length > 0 && (
            <div className="space-y-2">
              <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wide">
                Changed Files ({diffData.changed_files.length})
              </span>
              <div className="flex flex-wrap gap-2">
                {diffData.changed_files.map((file) => (
                  <span
                    key={file}
                    className="inline-block bg-zinc-950 border border-zinc-700 text-indigo-300 text-xs font-mono px-2.5 py-1 rounded"
                  >
                    {file}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Unified diff */}
          <div className="space-y-2">
            <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wide">
              Unified Diff
            </span>
            {diffLines.length === 0 ? (
              <p className="text-sm text-zinc-500 bg-zinc-950 border border-zinc-800 rounded-xl p-4">
                No textual changes were produced for this task.
              </p>
            ) : (
              <pre className="bg-zinc-950 border border-zinc-800 rounded-xl p-0 overflow-x-auto max-h-96 text-xs font-mono leading-5">
                {diffLines.map((line, i) => (
                  <div
                    key={i}
                    className={`px-4 whitespace-pre ${lineClass(line)}`}
                  >
                    {line || " "}
                  </div>
                ))}
              </pre>
            )}
          </div>

          {/* Sandbox test output (collapsible) */}
          {currentTask.test_output && (
            <details className="bg-zinc-950 border border-zinc-800 rounded-xl overflow-hidden">
              <summary className="cursor-pointer select-none px-4 py-3 text-xs font-semibold text-zinc-400 uppercase tracking-wide hover:text-zinc-200">
                Sandbox Test Output
              </summary>
              <pre className="px-4 pb-4 text-xs text-zinc-300 overflow-x-auto max-h-64 font-mono whitespace-pre-wrap">
                {currentTask.test_output}
              </pre>
            </details>
          )}

          {/* Actions */}
          {isPendingReview ? (
            <div className="flex flex-col sm:flex-row gap-3 pt-1">
              <button
                onClick={handleApprove}
                disabled={acting !== null}
                className="flex-1 inline-flex items-center justify-center gap-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-semibold px-6 py-2.5 rounded-lg transition shadow-sm shadow-emerald-950/50"
              >
                <Check className="w-4 h-4" />
                {acting === "approve" ? "Applying patch..." : "Approve & Apply"}
              </button>
              <button
                onClick={handleReject}
                disabled={acting !== null}
                className="flex-1 inline-flex items-center justify-center gap-1.5 bg-rose-700 hover:bg-rose-600 disabled:opacity-50 text-white font-semibold px-6 py-2.5 rounded-lg transition"
              >
                <X className="w-4 h-4" />
                {acting === "reject" ? "Rejecting..." : "Reject Fix"}
              </button>
            </div>
          ) : (
            <p className="text-sm text-zinc-400 pt-1">
              This task is no longer pending review (status:{" "}
              <span className="font-mono">{currentTask.status}</span>).
            </p>
          )}
        </>
      )}
    </div>
  );
}
