"use client";

import React, { useState } from "react";
import { Repository } from "@/lib/api";

export interface TaskFormPayload {
  description: string;
  title?: string;
  test_target?: string;
  max_attempts?: number;
}

interface TaskCreateFormProps {
  repo: Repository | null;
  /** Performs POST /tasks/fix (or /tasks). Errors are surfaced by the parent. */
  onSubmit: (payload: TaskFormPayload) => Promise<void>;
}

export default function TaskCreateForm({
  repo,
  onSubmit,
}: TaskCreateFormProps) {
  const [description, setDescription] = useState("");
  const [title, setTitle] = useState("");
  const [testTarget, setTestTarget] = useState("");
  const [maxAttempts, setMaxAttempts] = useState(3);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [isSubmitting, setIsSubmitting] = useState(false);

  const canSubmit =
    !!repo && description.trim().length > 0 && !isSubmitting;

  const handleSubmit = async () => {
    if (!repo || isSubmitting) return;

    setIsSubmitting(true);
    try {
      await onSubmit({
        description: description.trim(),
        title: title.trim() || undefined,
        test_target: testTarget.trim() || undefined,
        max_attempts: maxAttempts,
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-4">
      <label className="block space-y-1.5">
        <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wide">
          Issue Description
        </span>
        <textarea
          className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-4 py-3 text-zinc-100 text-sm placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500 resize-y min-h-[80px]"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe the bug or the change the agent should make, e.g. 'Fix the off-by-one error in the pagination helper' or 'Add input validation to the signup form'..."
          rows={3}
        />
      </label>

      <div className="flex flex-col sm:flex-row sm:items-center gap-3">
        <button
          onClick={() => setShowAdvanced((v) => !v)}
          className="text-xs text-zinc-400 hover:text-zinc-200 underline text-left"
        >
          {showAdvanced
            ? "Hide advanced options"
            : "Advanced options (title, test target, retry budget)"}
        </button>

        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          title={!repo ? "Select a repository first" : undefined}
          className="sm:ml-auto bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium px-6 py-2 rounded-lg transition whitespace-nowrap shadow-sm shadow-indigo-950/50"
        >
          {isSubmitting ? "Agent Working..." : "Start Coding Agent"}
        </button>
      </div>

      {showAdvanced && (
        <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-4 grid gap-3 sm:grid-cols-3">
          <label className="block space-y-1.5">
            <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wide">
              Title (optional)
            </span>
            <input
              type="text"
              className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Defaults to issue description"
            />
          </label>
          <label className="block space-y-1.5">
            <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wide">
              Test Target (optional)
            </span>
            <input
              type="text"
              className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500"
              value={testTarget}
              onChange={(e) => setTestTarget(e.target.value)}
              placeholder="tests/test_module.py::test_case"
            />
          </label>
          <label className="block space-y-1.5">
            <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wide">
              Max Attempts
            </span>
            <select
              className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500"
              value={maxAttempts}
              onChange={(e) => setMaxAttempts(Number(e.target.value))}
            >
              {[1, 2, 3, 4, 5].map((n) => (
                <option key={n} value={n}>
                  {n} attempt{n > 1 ? "s" : ""}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}
    </div>
  );
}
