"use client";

import React, { useEffect, useRef, useState } from "react";
import { Search } from "lucide-react";
import {
  CitationRef,
  QAAnswer,
  QAConfidence,
  Repository,
  askDeepQA,
  isApiError,
} from "@/lib/api";

interface DeepQAPanelProps {
  selectedRepo: Repository | null;
}

const CONFIDENCE_STYLES: Record<QAConfidence, string> = {
  direct_evidence: "bg-emerald-950 text-emerald-300 border-emerald-700",
  inferred: "bg-amber-950 text-amber-300 border-amber-700",
  no_evidence: "bg-zinc-800 text-zinc-300 border-zinc-600",
};

const CONFIDENCE_LABELS: Record<QAConfidence, string> = {
  direct_evidence: "Direct Evidence",
  inferred: "Inferred",
  no_evidence: "No Evidence",
};

/** Informational `file:start-end` label only -- never a link/button (no
 * source-code fetching or click-to-open behavior in this phase). */
function citationLabel(c: CitationRef): string {
  return `${c.file_path}:${c.start_line}-${c.end_line}`;
}

export default function DeepQAPanel({ selectedRepo }: DeepQAPanelProps) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<QAAnswer | null>(null);
  const [isAsking, setIsAsking] = useState(false);
  const [qaError, setQaError] = useState<string | null>(null);

  // Tracks the currently-selected repository id on every render, so an
  // in-flight request's response can tell whether the user has since
  // switched repositories (a plain closure variable captured at request
  // time would always see the *old* selectedRepo it started with).
  const latestRepoIdRef = useRef<number | null>(selectedRepo?.id ?? null);
  latestRepoIdRef.current = selectedRepo?.id ?? null;

  // Changing the selected repository alone (without asking a new question)
  // must not leave the previous repository's answer/error visible.
  useEffect(() => {
    setAnswer(null);
    setQaError(null);
  }, [selectedRepo?.id]);

  const canAsk = !!selectedRepo && question.trim().length > 0 && !isAsking;

  const handleAsk = async () => {
    if (!selectedRepo || !question.trim()) return;

    const requestRepoId = selectedRepo.id;
    setIsAsking(true);
    setQaError(null);
    setAnswer(null);
    try {
      const data = await askDeepQA({
        question: question.trim(),
        repository_id: requestRepoId,
      });
      // A stale response for a repository the user has since switched away
      // from must never overwrite the newly-selected repository's state.
      if (latestRepoIdRef.current === requestRepoId) {
        setAnswer(data);
      }
    } catch (err) {
      if (latestRepoIdRef.current === requestRepoId) {
        setQaError(
          isApiError(err) ? err.detail : "Failed to query the codebase."
        );
      }
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <section className="space-y-4">
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-600" />
          <input
            type="text"
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg pl-10 pr-4 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question (e.g. 'How does OrderService calculate discounts and taxes?')"
          />
        </div>
        <button
          onClick={handleAsk}
          disabled={!canAsk}
          title={!selectedRepo ? "Select a repository first" : undefined}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium px-6 py-2 rounded-lg transition whitespace-nowrap shadow-sm shadow-indigo-950/50"
        >
          {isAsking ? "Investigating..." : "Ask Codebase"}
        </button>
      </div>

      {qaError && (
        <p className="text-sm text-red-300 bg-red-950/40 border border-red-900 rounded-lg p-3">
          {qaError}
        </p>
      )}

      {answer && (
        <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800 space-y-3">
          <span
            className={`inline-flex items-center rounded-full border font-semibold tracking-wide text-xs px-3 py-1 ${CONFIDENCE_STYLES[answer.confidence]}`}
          >
            {CONFIDENCE_LABELS[answer.confidence]}
          </span>

          <p className="text-zinc-100 text-sm whitespace-pre-wrap leading-relaxed">
            {answer.summary}
          </p>

          {answer.corrected_premise && (
            <div className="border border-amber-800 bg-amber-950/30 rounded-lg p-3 text-sm text-amber-200">
              <span className="font-semibold">Note: </span>
              {answer.corrected_premise}
            </div>
          )}

          {answer.details && (
            <p className="text-zinc-300 text-sm whitespace-pre-wrap leading-relaxed">
              {answer.details}
            </p>
          )}

          {answer.flow_trace && answer.flow_trace.length > 0 && (
            <div className="pt-2 border-t border-zinc-800 space-y-2">
              <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wide">
                Flow
              </span>
              <ol className="space-y-1.5 list-decimal list-inside">
                {answer.flow_trace.map((step) => (
                  <li key={step.order} className="text-sm text-zinc-300">
                    {step.description}
                    {step.file_path && (
                      <span className="ml-2 inline-block bg-zinc-800 text-indigo-300 text-xs px-2 py-0.5 rounded font-mono">
                        {step.citation ? citationLabel(step.citation) : step.file_path}
                      </span>
                    )}
                  </li>
                ))}
              </ol>
            </div>
          )}

          {answer.evidence.length > 0 && (
            <div className="pt-2 border-t border-zinc-800">
              <span className="text-xs font-semibold text-zinc-400">
                Citations:{" "}
              </span>
              {answer.evidence.map((c, i) => (
                <span
                  key={i}
                  className="inline-block bg-zinc-800 text-indigo-300 text-xs px-2 py-1 rounded mr-2 mt-1 font-mono"
                >
                  {citationLabel(c)}
                </span>
              ))}
            </div>
          )}

          {answer.projects_considered.length > 1 && (
            <p className="text-xs text-zinc-500">
              Projects considered: {answer.projects_considered.join(", ")}
            </p>
          )}
        </div>
      )}
    </section>
  );
}
