"use client";

import React, { useState } from "react";

const API_BASE = "http://localhost:8000/api/v1";

export default function Home() {
  // State
  const [repoPath, setRepoPath] = useState("C:/Users/Lakshay/RepoPilot/demo_repo");
  const [repoId, setRepoId] = useState<number | null>(null);
  const [indexingStatus, setIndexingStatus] = useState<string>("");
  const [isIndexing, setIsIndexing] = useState<boolean>(false);

  // RAG Question State
  const [query, setQuery] = useState("");
  const [ragAnswer, setRagAnswer] = useState("");
  const [citations, setCitations] = useState<string[]>([]);
  const [isAsking, setIsAsking] = useState(false);

  // Bug Fix State
  const [issueDescription, setIssueDescription] = useState("Fix VIP discount to apply to order subtotal in OrderService");
  const [taskId, setTaskId] = useState<number | null>(null);
  const [taskStatus, setTaskStatus] = useState<string>("");
  const [attempts, setAttempts] = useState<number>(0);
  const [testOutput, setTestOutput] = useState<string>("");
  const [diffContent, setDiffContent] = useState<string>("");
  const [isFixing, setIsFixing] = useState<boolean>(false);
  const [isApproved, setIsApproved] = useState<boolean>(false);

  // 1 & 2. Register & Index Repository
  const handleIndexRepo = async () => {
    try {
      setIsIndexing(true);
      setIndexingStatus("Registering repository...");

      const createRes = await fetch(`${API_BASE}/repositories`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: "demo-repo",
          local_path: repoPath,
        }),
      });

      if (!createRes.ok) {
        throw new Error(`Failed to register repo: ${createRes.statusText}`);
      }

      const repo = await createRes.json();
      setRepoId(repo.id);
      setIndexingStatus(`Indexing syntax chunks for Repo ID ${repo.id}...`);

      const indexRes = await fetch(`${API_BASE}/repositories/${repo.id}/index`, {
        method: "POST",
      });
      const indexData = await indexRes.json();

      setIndexingStatus(`Successfully indexed ${indexData.total_chunks} syntax chunks!`);
    } catch (err: any) {
      setIndexingStatus(`Error: ${err.message}`);
    } finally {
      setIsIndexing(false);
    }
  };

  // 3. Ask Question (Code-Aware RAG)
  const handleAskQuestion = async () => {
    if (!repoId) {
      alert("Please index a repository first!");
      return;
    }
    try {
      setIsAsking(true);
      setRagAnswer("");
      setCitations([]);

      const res = await fetch(`${API_BASE}/rag/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          repository_id: repoId,
          top_k: 5,
        }),
      });
      const data = await res.json();
      setRagAnswer(data.answer);
      setCitations(data.citations || []);
    } catch (err: any) {
      setRagAnswer(`Error: ${err.message}`);
    } finally {
      setIsAsking(false);
    }
  };

  // 4. Submit Bug-Fix Task
  const handleStartFix = async () => {
    if (!repoId) {
      alert("Please index a repository first!");
      return;
    }
    try {
      setIsFixing(true);
      setTaskStatus("Investigating codebase...");
      setIsApproved(false);
      setDiffContent("");
      setTestOutput("");

      const res = await fetch(`${API_BASE}/tasks/fix`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repository_id: repoId,
          issue_description: issueDescription,
          max_attempts: 3,
        }),
      });
      const data = await res.json();

      setTaskId(data.id);
      setTaskStatus(data.status);
      setAttempts(data.attempts);
      setTestOutput(data.test_output || "All sandbox tests passed.");

      // Fetch Diff
      const diffRes = await fetch(`${API_BASE}/tasks/${data.id}/diff`);
      if (diffRes.ok) {
        const diffData = await diffRes.json();
        setDiffContent(diffData.diff || "# No remaining uncommitted diff");
      }
    } catch (err: any) {
      setTaskStatus(`Error: ${err.message}`);
    } finally {
      setIsFixing(false);
    }
  };

  // 10. Approve Fix
  const handleApproveFix = async () => {
    if (!taskId) return;
    try {
      const res = await fetch(`${API_BASE}/tasks/${taskId}/approve`, {
        method: "POST",
      });
      if (res.ok) {
        setIsApproved(true);
        setTaskStatus("approved");
      }
    } catch (err: any) {
      alert(`Approval error: ${err.message}`);
    }
  };

  return (
    <main className="max-w-6xl mx-auto p-8 space-y-8">
      {/* Header */}
      <header className="border-b border-slate-800 pb-6">
        <h1 className="text-3xl font-bold tracking-tight text-cyan-400">
          RepoPilot
        </h1>
        <p className="text-slate-400 mt-1">
          Autonomous AI Software Engineer — Syntax AST Indexing, Code RAG & 3-Attempt Self-Correction Loop
        </p>
      </header>

      {/* Section 1: Ingestion & Indexing */}
      <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
        <h2 className="text-xl font-semibold text-slate-200">1. Repository Ingestion & Tree-sitter Indexing</h2>
        <div className="flex gap-4">
          <input
            type="text"
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-2 text-slate-200 focus:outline-none focus:border-cyan-500"
            value={repoPath}
            onChange={(e) => setRepoPath(e.target.value)}
            placeholder="Local repository absolute path (e.g. C:/Users/Lakshay/RepoPilot/demo_repo)"
          />
          <button
            onClick={handleIndexRepo}
            disabled={isIndexing}
            className="bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-medium px-6 py-2 rounded-lg transition"
          >
            {isIndexing ? "Indexing..." : "Index Repository"}
          </button>
        </div>
        {indexingStatus && (
          <p className="text-sm font-mono text-cyan-300 bg-slate-950 p-3 rounded border border-slate-800">
            {indexingStatus}
          </p>
        )}
      </section>

      {/* Section 2: Code-Aware RAG */}
      <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
        <h2 className="text-xl font-semibold text-slate-200">2. Code-Aware Semantic Search & RAG</h2>
        <div className="flex gap-4">
          <input
            type="text"
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-2 text-slate-200 focus:outline-none focus:border-cyan-500"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question (e.g. 'How does OrderService calculate discounts and taxes?')"
          />
          <button
            onClick={handleAskQuestion}
            disabled={isAsking}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium px-6 py-2 rounded-lg transition"
          >
            {isAsking ? "Searching..." : "Ask Codebase"}
          </button>
        </div>

        {ragAnswer && (
          <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 space-y-3">
            <h3 className="font-semibold text-slate-300">Answer:</h3>
            <p className="text-slate-200 text-sm whitespace-pre-wrap leading-relaxed">{ragAnswer}</p>
            {citations.length > 0 && (
              <div className="pt-2 border-t border-slate-800">
                <span className="text-xs font-semibold text-slate-400">Citations: </span>
                {citations.map((c, i) => (
                  <span key={i} className="inline-block bg-slate-800 text-cyan-300 text-xs px-2 py-1 rounded mr-2 mt-1">
                    {c}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </section>

      {/* Section 3: Autonomous Bug Fixing & Self-Correction Loop */}
      <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
        <h2 className="text-xl font-semibold text-slate-200">3. Autonomous Bug Fixing & Self-Correction Loop</h2>
        <div className="flex gap-4">
          <input
            type="text"
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-2 text-slate-200 focus:outline-none focus:border-cyan-500"
            value={issueDescription}
            onChange={(e) => setIssueDescription(e.target.value)}
            placeholder="Describe the bug or task to resolve"
          />
          <button
            onClick={handleStartFix}
            disabled={isFixing}
            className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-medium px-6 py-2 rounded-lg transition"
          >
            {isFixing ? "Fixing & Testing..." : "Start Auto-Fix"}
          </button>
        </div>

        {/* Live Execution Status Dashboard */}
        {taskStatus && (
          <div className="grid grid-cols-3 gap-4 pt-2">
            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
              <span className="text-xs text-slate-400 uppercase font-semibold">Agent Stage</span>
              <p className="text-lg font-mono text-cyan-400 mt-1 capitalize">{taskStatus}</p>
            </div>
            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
              <span className="text-xs text-slate-400 uppercase font-semibold">Attempts</span>
              <p className="text-lg font-mono text-amber-400 mt-1">{attempts} / 3</p>
            </div>
            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
              <span className="text-xs text-slate-400 uppercase font-semibold">Human Approval</span>
              <p className="text-lg font-mono text-emerald-400 mt-1">
                {isApproved ? "APPROVED" : taskStatus === "human_approval_required" ? "PENDING APPROVAL" : "RUNNING"}
              </p>
            </div>
          </div>
        )}

        {/* Sandbox Test Output */}
        {testOutput && (
          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-slate-300">Sandbox Test Runner Output:</h3>
            <pre className="bg-slate-950 text-slate-300 text-xs p-4 rounded-lg border border-slate-800 overflow-x-auto max-h-48">
              {testOutput}
            </pre>
          </div>
        )}

        {/* Final Diff Viewer */}
        {diffContent && (
          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-slate-300">Generated Unified Git Diff:</h3>
            <pre className="bg-slate-950 text-emerald-400 text-xs p-4 rounded-lg border border-slate-800 overflow-x-auto max-h-60">
              {diffContent}
            </pre>
          </div>
        )}

        {/* Approval Action */}
        {taskStatus === "human_approval_required" && !isApproved && (
          <div className="p-4 bg-emerald-950/40 border border-emerald-800 rounded-lg flex items-center justify-between">
            <p className="text-emerald-300 text-sm">
              Sandbox verification passed all tests! Review the diff above and approve the fix.
            </p>
            <button
              onClick={handleApproveFix}
              className="bg-emerald-600 hover:bg-emerald-500 text-white font-semibold px-6 py-2 rounded-lg transition"
            >
              Approve Fix
            </button>
          </div>
        )}
      </section>
    </main>
  );
}
