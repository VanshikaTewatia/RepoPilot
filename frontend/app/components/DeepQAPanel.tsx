"use client";

import React, { useState } from "react";
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
  no_evidence: "bg-slate-800 text-slate-300 border-slate-600",
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

  const canAsk = !!selectedRepo && question.trim().length > 0 && !isAsking;

  const handleAsk = async () => {
    if (!selectedRepo || !question.trim()) return;

    setIsAsking(true);
    setQaError(null);
    setAnswer(null);
    try {
      const data = await askDeepQA({
        question: question.trim(),
        repository_id: selectedRepo.id,
      });
      setAnswer(data);
    } catch (err) {
      setQaError(
        isApiError(err) ? err.detail : "Failed to query the codebase."
      );
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
      <h2 className="text-xl font-semibold text-slate-200">
        2. Deep Codebase Q&amp;A
      </h2>
      <div className="flex flex-col sm:flex-row gap-3">
        <input
          type="text"
          className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-2 text-slate-200 focus:outline-none focus:border-indigo-500"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question (e.g. 'How does OrderService calculate discounts and taxes?')"
        />
        <button
          onClick={handleAsk}
          disabled={!canAsk}
          title={!selectedRepo ? "Select a repository first" : undefined}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium px-6 py-2 rounded-lg transition whitespace-nowrap"
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
        <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 space-y-3">
          <span
            className={`inline-flex items-center rounded-full border font-semibold tracking-wide text-xs px-3 py-1 ${CONFIDENCE_STYLES[answer.confidence]}`}
          >
            {CONFIDENCE_LABELS[answer.confidence]}
          </span>

          <p className="text-slate-200 text-sm whitespace-pre-wrap leading-relaxed">
            {answer.summary}
          </p>

          {answer.corrected_premise && (
            <div className="border border-amber-800 bg-amber-950/30 rounded-lg p-3 text-sm text-amber-200">
              <span className="font-semibold">Note: </span>
              {answer.corrected_premise}
            </div>
          )}

          {answer.details && (
            <p className="text-slate-300 text-sm whitespace-pre-wrap leading-relaxed">
              {answer.details}
            </p>
          )}

          {answer.flow_trace && answer.flow_trace.length > 0 && (
            <div className="pt-2 border-t border-slate-800 space-y-2">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
                Flow
              </span>
              <ol className="space-y-1.5 list-decimal list-inside">
                {answer.flow_trace.map((step) => (
                  <li key={step.order} className="text-sm text-slate-300">
                    {step.description}
                    {step.file_path && (
                      <span className="ml-2 inline-block bg-slate-800 text-cyan-300 text-xs px-2 py-0.5 rounded">
                        {step.citation ? citationLabel(step.citation) : step.file_path}
                      </span>
                    )}
                  </li>
                ))}
              </ol>
            </div>
          )}

          {answer.evidence.length > 0 && (
            <div className="pt-2 border-t border-slate-800">
              <span className="text-xs font-semibold text-slate-400">
                Citations:{" "}
              </span>
              {answer.evidence.map((c, i) => (
                <span
                  key={i}
                  className="inline-block bg-slate-800 text-cyan-300 text-xs px-2 py-1 rounded mr-2 mt-1"
                >
                  {citationLabel(c)}
                </span>
              ))}
            </div>
          )}

          {answer.projects_considered.length > 1 && (
            <p className="text-xs text-slate-500">
              Projects considered: {answer.projects_considered.join(", ")}
            </p>
          )}
        </div>
      )}
    </section>
  );
}
