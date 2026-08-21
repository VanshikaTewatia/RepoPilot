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
  const [description, setDescription] = useState(
    "Fix VIP discount to apply to order subtotal in OrderService"
  );
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
      <label className="block space-y-1">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
          Issue Description
        </span>
        <textarea
          className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-3 text-slate-200 text-sm focus:outline-none focus:border-emerald-500 resize-y min-h-[80px]"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe the bug or the change the agent should make..."
          rows={3}
        />
      </label>

      <div className="flex flex-col sm:flex-row sm:items-center gap-3">
        <button
          onClick={() => setShowAdvanced((v) => !v)}
          className="text-xs text-slate-400 hover:text-slate-200 underline text-left"
        >
          {showAdvanced
            ? "Hide advanced options"
            : "Advanced options (title, test target, retry budget)"}
        </button>

        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          title={!repo ? "Select a repository first" : undefined}
          className="sm:ml-auto bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium px-6 py-2 rounded-lg transition whitespace-nowrap"
        >
          {isSubmitting ? "Agent Working..." : "Start Coding Agent"}
        </button>
      </div>

      {showAdvanced && (
        <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 grid gap-3 sm:grid-cols-3">
          <label className="block space-y-1">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
              Title (optional)
            </span>
            <input
              type="text"
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Defaults to issue description"
            />
          </label>
          <label className="block space-y-1">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
              Test Target (optional)
            </span>
            <input
              type="text"
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500"
              value={testTarget}
              onChange={(e) => setTestTarget(e.target.value)}
              placeholder="tests/test_order_service.py::test_vip_discount"
            />
          </label>
          <label className="block space-y-1">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
              Max Attempts
            </span>
            <select
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500"
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
