"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  ApiError,
  IndexResult,
  Repository,
  createRepository,
  createRepositoryFromGitHub,
  indexRepository,
  isApiError,
  listRepositories,
} from "@/lib/api";

const DEFAULT_NEW_REPO_PATH = "C:/Users/Lakshay/RepoPilot/demo_repo";
type RegisterMode = "github" | "local";

interface RepositoryPickerProps {
  selectedRepo: Repository | null;
  onSelect: (repo: Repository) => void;
}

function repoLabel(repo: Repository): string {
  const indexed = repo.status === "indexed" ? "indexed" : repo.status;
  return `#${repo.id} · ${repo.name} (${indexed})`;
}

export default function RepositoryPicker({
  selectedRepo,
  onSelect,
}: RepositoryPickerProps) {
  const [repos, setRepos] = useState<Repository[]>([]);
  const [isLoadingList, setIsLoadingList] = useState(true);
  const [listError, setListError] = useState<string | null>(null);

  const [showRegister, setShowRegister] = useState(false);
  const [registerMode, setRegisterMode] = useState<RegisterMode>("github");
  const [newPath, setNewPath] = useState(DEFAULT_NEW_REPO_PATH);
  const [newName, setNewName] = useState("");
  const [githubUrl, setGithubUrl] = useState("");
  const [githubBranch, setGithubBranch] = useState("");
  const [isRegistering, setIsRegistering] = useState(false);
  const [registerError, setRegisterError] = useState<string | null>(null);

  const [isIndexing, setIsIndexing] = useState(false);
  const [indexMessage, setIndexMessage] = useState<string | null>(null);
  const [indexIsError, setIndexIsError] = useState(false);

  const loadRepositories = useCallback(async () => {
    setIsLoadingList(true);
    setListError(null);
    try {
      const repos = await listRepositories();
      setRepos(repos);
    } catch (err) {
      setListError(
        isApiError(err) ? err.detail : "Failed to load repositories."
      );
    } finally {
      setIsLoadingList(false);
    }
  }, []);

  useEffect(() => {
    void loadRepositories();
  }, [loadRepositories]);

  // Auto-select the most recent repository on first load.
  useEffect(() => {
    if (!selectedRepo && repos.length > 0) {
      onSelect(repos[0]);
    }
  }, [repos, selectedRepo, onSelect]);

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
      setRepos((prev) => [repo, ...prev]);
      onSelect(repo);
      setShowRegister(false);
      setGithubUrl("");
      setGithubBranch("");
      setNewName("");
      setIndexMessage(`Cloned "${repo.name}" (#${repo.id}) from GitHub. Ready to index.`);
      setIndexIsError(false);
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
      setRepos((prev) => [repo, ...prev]);
      onSelect(repo);
      setShowRegister(false);
      setNewName("");
      setIndexMessage(`Registered "${repo.name}" (#${repo.id}). Ready to index.`);
      setIndexIsError(false);
    } catch (err) {
      setRegisterError(
        isApiError(err)
          ? err.detail
          : "Failed to register repository."
      );
    } finally {
      setIsRegistering(false);
    }
  };

  const handleIndex = async () => {
    if (!selectedRepo) return;

    setIsIndexing(true);
    setIndexMessage(`Indexing "${selectedRepo.name}"...`);
    setIndexIsError(false);
    try {
      const result: IndexResult = await indexRepository(selectedRepo.id);
      setIndexMessage(
        `Indexed ${result.total_chunks} syntax chunks ` +
          `(${result.new_chunks} new, ${result.reused_chunks} reused).`
      );
      setRepos((prev) =>
        prev.map((r) =>
          r.id === selectedRepo.id
            ? {
                ...r,
                status: "indexed",
                indexed_at: new Date().toISOString(),
              }
            : r
        )
      );
      onSelect({ ...selectedRepo, status: "indexed" });
    } catch (err) {
      const detail =
        err instanceof ApiError ? err.detail : "Indexing failed.";
      setIndexMessage(detail);
      setIndexIsError(true);
    } finally {
      setIsIndexing(false);
    }
  };

  return (
    <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-slate-200">
          1. Repository
        </h2>
        {selectedRepo && (
          <span className="text-xs font-mono text-slate-500 truncate max-w-[50%]">
            {selectedRepo.local_path}
          </span>
        )}
      </div>

      {/* Selection row */}
      <div className="flex flex-col sm:flex-row gap-3">
        <select
          className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-2 text-slate-200 focus:outline-none focus:border-cyan-500 disabled:opacity-50"
          value={selectedRepo?.id ?? ""}
          onChange={(e) => {
            const repo = repos.find((r) => r.id === Number(e.target.value));
            if (repo) onSelect(repo);
          }}
          disabled={isLoadingList || repos.length === 0}
        >
          {isLoadingList && <option>Loading repositories...</option>}
          {!isLoadingList && repos.length === 0 && (
            <option value="">No repositories registered yet</option>
          )}
          {repos.map((repo) => (
            <option key={repo.id} value={repo.id}>
              {repoLabel(repo)}
            </option>
          ))}
        </select>

        <button
          onClick={handleIndex}
          disabled={!selectedRepo || isIndexing}
          className="bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium px-6 py-2 rounded-lg transition whitespace-nowrap"
        >
          {isIndexing ? "Indexing..." : "Index Repository"}
        </button>

        <button
          onClick={() => setShowRegister((v) => !v)}
          className="border border-slate-700 hover:border-slate-500 text-slate-300 font-medium px-4 py-2 rounded-lg transition whitespace-nowrap"
        >
          {showRegister ? "Cancel" : "+ Connect Repository"}
        </button>
      </div>

      {listError && (
        <p className="text-sm text-red-400 bg-red-950/40 border border-red-900 rounded-lg p-3">
          {listError}{" "}
          <button
            onClick={() => void loadRepositories()}
            className="underline hover:text-red-300"
          >
            Retry
          </button>
        </p>
      )}

      {/* Register new repository */}
      {showRegister && (
        <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 space-y-3">
          {/* Mode toggle */}
          <div className="inline-flex rounded-lg border border-slate-700 overflow-hidden text-xs font-medium">
            <button
              onClick={() => {
                setRegisterMode("github");
                setRegisterError(null);
              }}
              className={`px-3 py-1.5 transition ${
                registerMode === "github"
                  ? "bg-cyan-600 text-white"
                  : "bg-slate-900 text-slate-400 hover:text-slate-200"
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
                  ? "bg-cyan-600 text-white"
                  : "bg-slate-900 text-slate-400 hover:text-slate-200"
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
                  className="flex-[2] bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-cyan-500"
                  value={githubUrl}
                  onChange={(e) => setGithubUrl(e.target.value)}
                  placeholder="https://github.com/owner/repository"
                />
                <input
                  type="text"
                  className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-cyan-500"
                  value={githubBranch}
                  onChange={(e) => setGithubBranch(e.target.value)}
                  placeholder="Base branch (optional, defaults to repo default)"
                />
                <button
                  onClick={handleRegisterGithub}
                  disabled={isRegistering}
                  className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-sm font-medium px-5 py-2 rounded-lg transition whitespace-nowrap"
                >
                  {isRegistering ? "Connecting..." : "Connect"}
                </button>
              </div>
              <p className="text-[11px] text-slate-500">
                Public repositories work with no configuration. Private repositories
                require <code className="text-slate-400">GITHUB_TOKEN</code> configured
                on the RepoPilot server — there is no token field here, and RepoPilot
                never asks for or stores one.
              </p>
            </div>
          ) : (
            <div className="flex flex-col sm:flex-row gap-3">
              <input
                type="text"
                className="flex-[2] bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-cyan-500"
                value={newPath}
                onChange={(e) => setNewPath(e.target.value)}
                placeholder="Absolute local path (e.g. C:/Users/Lakshay/RepoPilot/demo_repo)"
              />
              <input
                type="text"
                className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-cyan-500"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Display name (optional)"
              />
              <button
                onClick={handleRegister}
                disabled={isRegistering}
                className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-sm font-medium px-5 py-2 rounded-lg transition whitespace-nowrap"
              >
                {isRegistering ? "Registering..." : "Register"}
              </button>
            </div>
          )}

          {registerError && (
            <p className="text-xs text-red-400">{registerError}</p>
          )}
        </div>
      )}

      {/* Indexing result / error */}
      {indexMessage && (
        <p
          className={`text-sm font-mono p-3 rounded border ${
            indexIsError
              ? "text-red-300 bg-red-950/40 border-red-900"
              : "text-cyan-300 bg-slate-950 border-slate-800"
          }`}
        >
          {indexMessage}
        </p>
      )}
    </section>
  );
}
