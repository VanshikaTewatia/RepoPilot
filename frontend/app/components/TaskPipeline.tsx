"use client";

import React, { useEffect, useRef, useState } from "react";
import { CheckCircle2, CircleDot, Search, Lightbulb, FileEdit, FlaskConical, ClipboardCheck } from "lucide-react";
import { Task, getTask } from "@/lib/api";
import { isFinalTaskStatus, isTerminalTaskStatus } from "@/lib/taskStatus";
import StatusBadge from "./StatusBadge";

/** Ordered pipeline stages shown in the vertical timeline. This is a
 * presentation layer over the backend's single coarse `status` field --
 * the backend does not stream six independent stage states. */
const STAGES = [
  {
    label: "Investigation",
    description: "Inspecting the repository and locating relevant code",
    icon: Search,
  },
  {
    label: "Evidence",
    description: "Gathering code and repository evidence",
    icon: Lightbulb,
  },
  {
    label: "Plan",
    description: "Determining the safest change",
    icon: ClipboardCheck,
  },
  {
    label: "Changes",
    description: "Applying the proposed patch in an isolated workspace",
    icon: FileEdit,
  },
  {
    label: "Tests",
    description: "Running project verification",
    icon: FlaskConical,
  },
  {
    label: "Review",
    description: "Review the resulting changes before approval",
    icon: CheckCircle2,
  },
] as const;

/**
 * Map backend statuses to a stage index. The backend may emit intermediate
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
 * outcome box instead of the in-progress timeline/notes. */
const OUTCOME_STYLES: Record<string, string> = {
  approved: "bg-emerald-950/40 border-emerald-800",
  rejected: "bg-rose-950/40 border-rose-800",
  failed: "bg-red-950/40 border-red-900",
  unable_to_verify: "bg-amber-950/40 border-amber-800",
  no_change_needed: "bg-zinc-800/40 border-zinc-600",
};

const OUTCOME_MESSAGES: Record<string, string> = {
  rejected: "Fix rejected — workspace discarded, repository untouched.",
  failed: "The agent could not produce a verified fix within the retry budget.",
  unable_to_verify:
    "Verification could not be run in this environment (missing tooling or an unsupported project type) — this is not evidence that the reported issue does or does not exist.",
  no_change_needed:
    "No code changes were needed — the reported behavior was already correct, or the claimed issue could not be substantiated against the repository.",
};

/** Statuses whose generic outcome message hides real, useful detail
 * (the exact test/verification output, or why verification couldn't run) --
 * `approved`/`rejected` need no further detail beyond their own message. */
const SHOW_DETAIL_STATUSES = new Set(["failed", "unable_to_verify", "no_change_needed"]);

interface TaskPipelineProps {
  task: Task | null;
  /** True while the synchronous POST /tasks/fix request is still in flight. */
  creating: boolean;
  onTaskChange: (task: Task) => void;
}

export default function TaskPipeline({
  task,
  creating,
  onTaskChange,
}: TaskPipelineProps) {
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

  // Resolve current stage; unknown statuses render as indeterminate.
  let currentStep: number;
  if (!task) {
    currentStep = -1;
  } else if (isFinalTaskStatus(task.status)) {
    currentStep = STAGES.length; // everything completed
  } else {
    currentStep = STATUS_TO_STEP[task.status] ?? -1;
  }

  return (
    <div className="space-y-5">
      {/* Header row */}
      <div className="flex flex-wrap items-center gap-3">
        {creating ? (
          <span className="inline-flex items-center gap-2 text-sm text-indigo-300">
            <span className="w-3 h-3 rounded-full border-2 border-indigo-400 border-t-transparent animate-spin" />
            Agent is working on the repository copy...
          </span>
        ) : (
          task && <StatusBadge status={task.status} />
        )}
        {task && (
          <span className="text-xs font-mono text-zinc-500">
            Task #{task.id} · attempt {task.attempts}
          </span>
        )}
      </div>

      {/* Vertical timeline */}
      <ol className="space-y-0">
        {STAGES.map((stage, idx) => {
          const done = currentStep > idx;
          const active = currentStep === idx;
          const Icon = stage.icon;
          const isLast = idx === STAGES.length - 1;
          return (
            <li key={stage.label} className="flex gap-3.5">
              <div className="flex flex-col items-center">
                <span
                  className={`flex items-center justify-center w-8 h-8 rounded-full border-2 shrink-0 transition-colors ${
                    done
                      ? "bg-indigo-600 border-indigo-500 text-white"
                      : active
                        ? "border-indigo-400 text-indigo-300 bg-zinc-950 animate-pulse"
                        : "border-zinc-700 text-zinc-600 bg-zinc-950"
                  }`}
                >
                  {done ? (
                    <CheckCircle2 className="w-4 h-4" />
                  ) : active ? (
                    <CircleDot className="w-4 h-4" />
                  ) : (
                    <Icon className="w-3.5 h-3.5" />
                  )}
                </span>
                {!isLast && (
                  <span
                    className={`w-0.5 flex-1 min-h-[1.5rem] ${
                      done ? "bg-indigo-700" : "bg-zinc-800"
                    }`}
                    aria-hidden
                  />
                )}
              </div>
              <div className={`pb-5 ${isLast ? "pb-0" : ""}`}>
                <p
                  className={`text-sm font-semibold ${
                    active ? "text-indigo-300" : done ? "text-zinc-200" : "text-zinc-500"
                  }`}
                >
                  {stage.label}
                </p>
                <p className="text-xs text-zinc-500 mt-0.5">{stage.description}</p>
              </div>
            </li>
          );
        })}
      </ol>

      {/* Indeterminate / informational notes */}
      {creating && (
        <p className="text-xs text-zinc-500 bg-zinc-950 border border-zinc-800 rounded-lg p-3">
          Task execution is synchronous — this can take up to a few minutes.
          Progress will appear here as soon as the run completes.
        </p>
      )}
      {!creating && task && currentStep === -1 && !isTerminal && (
        <p className="text-xs text-zinc-500 bg-zinc-950 border border-zinc-800 rounded-lg p-3">
          Current status: <span className="font-mono">{task.status}</span> — waiting for the next agent stage...
        </p>
      )}
      {pollError && (
        <p className="text-xs text-amber-400">{pollError}</p>
      )}

      {/* Terminal outcome */}
      {task && isFinalTaskStatus(task.status) && (
        <div className="space-y-2">
          <div
            className={`flex items-center justify-between p-4 rounded-xl border ${
              OUTCOME_STYLES[task.status] ?? "bg-zinc-800/40 border-zinc-600"
            }`}
          >
            <div className="flex items-center gap-3">
              <StatusBadge status={task.status} size="lg" />
              <span className="text-sm text-zinc-300">
                {task.status === "approved"
                  ? task.pr_url
                    ? "Patch pushed to a new branch and a Pull Request was opened."
                    : "Patch applied to the original repository."
                  : OUTCOME_MESSAGES[task.status]}
              </span>
            </div>
          </div>

          {SHOW_DETAIL_STATUSES.has(task.status) && task.test_output && (
            <details className="bg-zinc-950 border border-zinc-800 rounded-xl overflow-hidden">
              <summary className="cursor-pointer select-none px-4 py-3 text-xs font-semibold text-zinc-400 uppercase tracking-wide hover:text-zinc-200">
                Details
              </summary>
              <pre className="px-4 pb-4 text-xs text-zinc-300 overflow-x-auto max-h-64 font-mono whitespace-pre-wrap">
                {task.test_output}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
