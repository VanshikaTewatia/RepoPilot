"use client";

import React, { useState } from "react";
import { Plug } from "lucide-react";
import {
  Repository,
  createRepository,
  createRepositoryFromGitHub,
  isApiError,
} from "@/lib/api";

const DEFAULT_NEW_REPO_PATH = "C:/Users/Lakshay/RepoPilot/demo_repo";
type RegisterMode = "github" | "local";

interface ConnectRepositoryPanelProps {
  onConnected: (repo: Repository) => void;
}

export default function ConnectRepositoryPanel({
  onConnected,
}: ConnectRepositoryPanelProps) {
  const [registerMode, setRegisterMode] = useState<RegisterMode>("github");
  const [newPath, setNewPath] = useState(DEFAULT_NEW_REPO_PATH);
  const [newName, setNewName] = useState("");
  const [githubUrl, setGithubUrl] = useState("");
  const [githubBranch, setGithubBranch] = useState("");
  const [isRegistering, setIsRegistering] = useState(false);
  const [registerError, setRegisterError] = useState<string | null>(null);

  const handleRegisterGithub = async () => {
    const trimmedUrl = githubUrl.trim();
    if (!trimmedUrl) {
      setRegisterError("Please enter a GitHub repository URL.");
      return;
    }

    setIsRegistering(true);
    setRegisterError(null);
    try {
      const repo = await createRepositoryFromGitHub({
        url: trimmedUrl,
        name: newName.trim() || undefined,
        default_branch: githubBranch.trim() || undefined,
      });
      setGithubUrl("");
      setGithubBranch("");
      setNewName("");
      onConnected(repo);
    } catch (err) {
      setRegisterError(
        isApiError(err) ? err.detail : "Failed to connect the GitHub repository."
      );
    } finally {
      setIsRegistering(false);
    }
  };

  const handleRegister = async () => {
    const trimmedPath = newPath.trim();
    if (!trimmedPath) {
      setRegisterError("Please enter a local repository path.");
      return;
    }

    setIsRegistering(true);
    setRegisterError(null);
    try {
      const name =
        newName.trim() ||
        trimmedPath.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ||
        "repository";
      const repo = await createRepository({ name, local_path: trimmedPath });
      setNewName("");
      onConnected(repo);
    } catch (err) {
      setRegisterError(
        isApiError(err) ? err.detail : "Failed to register repository."
      );
    } finally {
      setIsRegistering(false);
    }
  };

  return (
    <section className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6 space-y-4 shadow-sm shadow-black/20">
      <div className="flex items-center gap-2.5">
        <span className="flex items-center justify-center w-7 h-7 rounded-lg bg-indigo-500/10 text-indigo-400">
          <Plug className="w-4 h-4" />
        </span>
        <h2 className="text-base font-semibold text-zinc-100">
          Connect Repository
        </h2>
      </div>

      {/* Mode toggle */}
      <div className="inline-flex rounded-lg border border-zinc-800 overflow-hidden text-xs font-medium">
        <button
          onClick={() => {
            setRegisterMode("github");
            setRegisterError(null);
          }}
          className={`px-3 py-1.5 transition ${
            registerMode === "github"
              ? "bg-indigo-600 text-white"
              : "bg-zinc-900 text-zinc-400 hover:text-zinc-200"
          }`}
        >
          GitHub URL
        </button>
        <button
          onClick={() => {
            setRegisterMode("local");
            setRegisterError(null);
          }}
          className={`px-3 py-1.5 transition ${
            registerMode === "local"
              ? "bg-indigo-600 text-white"
              : "bg-zinc-900 text-zinc-400 hover:text-zinc-200"
          }`}
        >
          Local Path
        </button>
      </div>

      {registerMode === "github" ? (
        <div className="space-y-2">
          <div className="flex flex-col sm:flex-row gap-3">
            <input
              type="text"
              className="flex-[2] bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500"
              value={githubUrl}
              onChange={(e) => setGithubUrl(e.target.value)}
              placeholder="https://github.com/owner/repository"
            />
            <input
              type="text"
              className="flex-1 bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500"
              value={githubBranch}
              onChange={(e) => setGithubBranch(e.target.value)}
              placeholder="Base branch (optional, defaults to repo default)"
            />
            <button
              onClick={handleRegisterGithub}
              disabled={isRegistering}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm font-medium px-5 py-2 rounded-lg transition whitespace-nowrap"
            >
              {isRegistering ? "Connecting..." : "Connect"}
            </button>
          </div>
          <p className="text-[11px] text-zinc-500">
            Public repositories work with no configuration. Private repositories
            require <code className="text-zinc-400">GITHUB_TOKEN</code> configured
            on the RepoPilot server — there is no token field here, and RepoPilot
            never asks for or stores one.
          </p>
        </div>
      ) : (
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            type="text"
            className="flex-[2] bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500"
            value={newPath}
            onChange={(e) => setNewPath(e.target.value)}
            placeholder="Absolute local path (e.g. C:/Users/Lakshay/RepoPilot/demo_repo)"
          />
          <input
            type="text"
            className="flex-1 bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="Display name (optional)"
          />
          <button
            onClick={handleRegister}
            disabled={isRegistering}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm font-medium px-5 py-2 rounded-lg transition whitespace-nowrap"
          >
            {isRegistering ? "Registering..." : "Register"}
          </button>
        </div>
      )}

      {registerError && (
        <p className="text-xs text-red-400">{registerError}</p>
      )}
    </section>
  );
}
