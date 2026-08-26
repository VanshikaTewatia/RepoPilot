"use client";

import React, { useEffect, useRef, useState } from "react";
import { Task, getTask } from "@/lib/api";
import { isFinalTaskStatus, isTerminalTaskStatus } from "@/lib/taskStatus";
import StatusBadge from "./StatusBadge";

/** Ordered workflow steps shown in the stepper. */
const STEPS = [
  "Investigating",
  "Retrieving",
  "Planning",
  "Editing",
  "Testing",
  "Review",
] as const;

/**
 * Map backend statuses to a step index. The backend may emit intermediate
 * statuses from the LangGraph loop; unknown/transitional statuses are mapped
 * to -1 (indeterminate) rather than guessed.
 */
const STATUS_TO_STEP: Record<string, number> = {
  pending: 0,
  investigating: 0,
  retrieving: 1,
  retrieved: 1,
  planning: 2,
  analyzing_failure: 2,
  editing: 3,
  edited: 3,
  testing: 4,
  tested: 4,
  verifying: 4,
  verified: 5,
  human_approval_required: 5,
  approval_failed: 5,
};

/** Final-outcome display copy. Every status in `taskStatus.ts`'s "final"
 * category MUST have an entry here -- this is what renders the terminal
 * outcome box instead of the in-progress stepper/notes. */
const OUTCOME_STYLES: Record<string, string> = {
  approved: "bg-emerald-950/40 border-emerald-800",
  rejected: "bg-rose-950/40 border-rose-800",
  failed: "bg-red-950/40 border-red-900",
  unable_to_verify: "bg-amber-950/40 border-amber-800",
  no_change_needed: "bg-slate-800/40 border-slate-600",
};

const OUTCOME_MESSAGES: Record<string, string> = {
  rejected: "Fix rejected — workspace discarded, repository untouched.",
  failed: "The agent could not produce a verified fix within the retry budget.",
  unable_to_verify:
    "Verification could not be run in this environment (missing tooling or an unsupported project type) — this is not evidence that the reported issue does or does not exist.",
  no_change_needed:
    "No code changes were needed — the reported behavior was already correct, or the claimed issue could not be substantiated against the repository.",
};

interface TaskProgressProps {
  task: Task | null;
  /** True while the synchronous POST /tasks/fix request is still in flight. */
  creating: boolean;
  onTaskChange: (task: Task) => void;
}

export default function TaskProgress({
  task,
  creating,
  onTaskChange,
}: TaskProgressProps) {
  const [pollError, setPollError] = useState<string | null>(null);
  const pollingRef = useRef(false);

  const isTerminal = isTerminalTaskStatus(task?.status);

  // Poll GET /tasks/{id} every 2s while the task is in a non-terminal state.
  useEffect(() => {
    if (!task || isTerminal) return;

    const taskId = task.id;
    const interval = setInterval(async () => {
      if (pollingRef.current) return;
      pollingRef.current = true;
      try {
        const fresh = await getTask(taskId);
        setPollError(null);
        onTaskChange(fresh);
      } catch {
        // Transient errors (e.g. backend restarting): keep polling silently,
        // but surface a soft note after failures persist via error state.
        setPollError("Last status refresh failed — retrying...");
      } finally {
        pollingRef.current = false;
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [task, isTerminal, onTaskChange]);

  // Resolve current step; unknown statuses render as indeterminate.
  let currentStep: number;
  if (!task) {
    currentStep = -1;
  } else if (isFinalTaskStatus(task.status)) {
    currentStep = STEPS.length; // everything completed
  } else {
    currentStep = STATUS_TO_STEP[task.status] ?? -1;
  }

  return (
    <div className="space-y-4">
      {/* Header row */}
      <div className="flex flex-wrap items-center gap-3">
        {creating ? (
          <span className="inline-flex items-center gap-2 text-sm text-cyan-300">
            <span className="w-3 h-3 rounded-full border-2 border-cyan-400 border-t-transparent animate-spin" />
            Agent is working on the repository copy...
          </span>
        ) : (
          task && <StatusBadge status={task.status} />
        )}
        {task && (
          <span className="text-xs font-mono text-slate-500">
            Task #{task.id} · attempt {task.attempts}
          </span>
        )}
      </div>

      {/* Stepper */}
      <ol className="flex items-start">
        {STEPS.map((label, idx) => {
          const done = currentStep > idx;
          const active = currentStep === idx;
          return (
            <li key={label} className="flex-1 flex flex-col items-center relative">
              {idx > 0 && (
                <span
                  className={`absolute top-4 right-1/2 w-full h-0.5 ${
                    done || active ? "bg-cyan-700" : "bg-slate-800"
                  }`}
                  aria-hidden
                />
              )}
              <span
                className={`relative z-10 w-8 h-8 rounded-full border-2 flex items-center justify-center text-[10px] font-bold transition-colors ${
                  done
                    ? "bg-cyan-600 border-cyan-500 text-white"
                    : active
                      ? "border-cyan-400 text-cyan-300 bg-slate-950 animate-pulse"
                      : "border-slate-700 text-slate-600 bg-slate-950"
                }`}
              >
                {done ? "✓" : idx + 1}
              </span>
              <span
                className={`mt-2 text-[10px] uppercase tracking-wide text-center ${
                  active ? "text-cyan-300" : done ? "text-slate-400" : "text-slate-600"
                }`}
              >
                {label}
              </span>
            </li>
          );
        })}
      </ol>

      {/* Indeterminate / informational notes */}
      {creating && (
        <p className="text-xs text-slate-500 bg-slate-950 border border-slate-800 rounded-lg p-3">
          Task execution is synchronous — this can take up to a few minutes.
          Progress will appear here as soon as the run completes.
        </p>
      )}
      {!creating && task && currentStep === -1 && !isTerminal && (
        <p className="text-xs text-slate-500 bg-slate-950 border border-slate-800 rounded-lg p-3">
          Current status: <span className="font-mono">{task.status}</span> — waiting for the next agent stage...
        </p>
      )}
      {pollError && (
        <p className="text-xs text-amber-400">{pollError}</p>
      )}

      {/* Terminal outcome */}
      {task && isFinalTaskStatus(task.status) && (
        <div
          className={`flex items-center justify-between p-4 rounded-lg border ${
            OUTCOME_STYLES[task.status] ?? "bg-slate-800/40 border-slate-600"
          }`}
        >
          <div className="flex items-center gap-3">
            <StatusBadge status={task.status} size="lg" />
            <span className="text-sm text-slate-300">
              {task.status === "approved"
                ? task.pr_url
                  ? "Patch pushed to a new branch and a Pull Request was opened."
                  : "Patch applied to the original repository."
                : OUTCOME_MESSAGES[task.status]}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
