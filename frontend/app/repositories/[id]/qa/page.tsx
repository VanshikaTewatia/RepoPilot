"use client";

import React from "react";
import { useCurrentRepository } from "@/lib/repository-context";
import DeepQAPanel from "../../../components/DeepQAPanel";

export default function RepositoryQAPage() {
  const repo = useCurrentRepository();

  return (
    <main className="flex-1 w-full max-w-5xl mx-auto px-6 py-8 space-y-6">
      <div>
        <h1 className="text-lg font-semibold text-zinc-100">Deep Codebase Q&A</h1>
        <p className="text-sm text-zinc-400 mt-1 max-w-2xl">
          Ask a question about how this codebase actually works. Every answer
          is grounded in retrieved code — RepoPilot cites the exact files and
          lines it used, and says so plainly when it finds no evidence.
        </p>
      </div>
      <DeepQAPanel selectedRepo={repo} />
    </main>
  );
}
