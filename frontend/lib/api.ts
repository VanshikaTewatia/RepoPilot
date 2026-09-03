/**
 * Typed API client for the RepoPilot backend.
 *
 * Base URL comes from NEXT_PUBLIC_API_BASE_URL (inlined at build time) with a
 * local-development fallback matching the backend default port.
 */

export const API_BASE: string =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Repository {
  id: number;
  name: string;
  local_path: string;
  remote_url: string | null;
  default_branch: string;
  status: string;
  indexed_at: string | null;
}

export interface Task {
  id: number;
  repository_id: number;
  title: string;
  description: string;
  status: string;
  attempts: number;
  patch_content: string | null;
  test_output: string | null;
  pr_url: string | null;
}

export interface TaskDiff {
  task_id: number;
  status: string;
  diff: string;
  changed_files: string[];
}

export interface IndexResult {
  success: boolean;
  repository_id: number;
  total_chunks: number;
  reused_chunks: number;
  new_chunks: number;
}

export interface RagChunk {
  file_path: string;
  symbol_name: string | null;
  symbol_type: string;
  start_line: number;
  end_line: number;
  source_code: string;
  similarity_score: number;
  citation: string;
}

export interface RagAnswer {
  answer: string;
  retrieved_chunks: RagChunk[];
  citations: string[];
}

export interface CitationRef {
  file_path: string;
  start_line: number;
  end_line: number;
  symbol_name: string | null;
}

export interface FlowStep {
  order: number;
  description: string;
  file_path: string;
  citation: CitationRef | null;
}

export type QAConfidence = "direct_evidence" | "inferred" | "no_evidence";

export interface QAAnswer {
  summary: string;
  details: string | null;
  flow_trace: FlowStep[] | null;
  evidence: CitationRef[];
  corrected_premise: string | null;
  confidence: QAConfidence;
  projects_considered: string[];
}

export interface TaskReviewResult {
  task_id: number;
  status: string;
  message: string;
  pr_url?: string | null;
}

/**
 * Normalized API error. `detail` preserves the backend's error message,
 * including HTTP 409 patch-conflict details from the approve endpoint.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail || `Request failed with status ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }

  get isConflict(): boolean {
    return this.status === 409;
  }
}

export function isApiError(err: unknown): err is ApiError {
  return err instanceof ApiError;
}

// ---------------------------------------------------------------------------
// Core request helper
// ---------------------------------------------------------------------------

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(
      0,
      "Cannot reach the RepoPilot backend. Make sure it is running on port 8000."
    );
  }

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") {
        detail = body.detail;
      } else if (body?.detail !== undefined) {
        detail = JSON.stringify(body.detail);
      }
    } catch {
      // Non-JSON error body; keep the status-text fallback.
    }
    throw new ApiError(res.status, detail);
  }

  return (await res.json()) as T;
}

function jsonInit(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

// ---------------------------------------------------------------------------
// Repositories
// ---------------------------------------------------------------------------

export function listRepositories(): Promise<Repository[]> {
  return request<Repository[]>("/repositories");
}

export function createRepository(payload: {
  name: string;
  local_path: string;
}): Promise<Repository> {
  return request<Repository>("/repositories", jsonInit("POST", payload));
}

export function indexRepository(repoId: number): Promise<IndexResult> {
  return request<IndexResult>(`/repositories/${repoId}/index`, { method: "POST" });
}

/**
 * Register a repository by cloning it from a public or (server-token-
 * enabled) private GitHub URL. There is no token parameter here by design:
 * GITHUB_TOKEN lives only in the backend's environment and is never accepted
 * from the frontend.
 */
export function createRepositoryFromGitHub(payload: {
  url: string;
  name?: string;
  default_branch?: string;
}): Promise<Repository> {
  return request<Repository>("/repositories/github", jsonInit("POST", payload));
}

// ---------------------------------------------------------------------------
// Code-aware RAG
// ---------------------------------------------------------------------------

export function askCodebase(payload: {
  query: string;
  repository_id: number;
  top_k?: number;
}): Promise<RagAnswer> {
  return request<RagAnswer>(
    "/rag/ask",
    jsonInit("POST", { top_k: 5, ...payload })
  );
}

// ---------------------------------------------------------------------------
// Deep Codebase Q&A
// ---------------------------------------------------------------------------

/**
 * Request contains ONLY `question` and `repository_id` -- no workspace/local
 * path field exists on this type, matching the backend's QAAskRequest. The
 * server resolves the repository's filesystem location itself.
 */
export function askDeepQA(payload: {
  question: string;
  repository_id: number;
}): Promise<QAAnswer> {
  return request<QAAnswer>("/qa/ask", jsonInit("POST", payload));
}

// ---------------------------------------------------------------------------
// Tasks
// ---------------------------------------------------------------------------

export interface SubmitTaskPayload {
  repository_id: number;
  description: string;
  title?: string;
  test_target?: string;
  max_attempts?: number;
}

/**
 * Create an agent task. Uses POST /tasks when a custom title is provided and
 * the convenience POST /tasks/fix otherwise (the backend derives the title
 * from the description on /fix).
 */
export function submitTask(payload: SubmitTaskPayload): Promise<Task> {
  const { repository_id, description, title, test_target, max_attempts } = payload;

  if (title && title.trim()) {
    return request<Task>(
      "/tasks",
      jsonInit("POST", {
        repository_id,
        title: title.trim(),
        description,
        ...(test_target ? { test_target } : {}),
        ...(max_attempts ? { max_attempts } : {}),
      })
    );
  }

  return request<Task>(
    "/tasks/fix",
    jsonInit("POST", {
      repository_id,
      issue_description: description,
      ...(test_target ? { test_target } : {}),
      ...(max_attempts ? { max_attempts } : {}),
    })
  );
}

export function getTask(taskId: number): Promise<Task> {
  return request<Task>(`/tasks/${taskId}`);
}

export function getTaskDiff(taskId: number): Promise<TaskDiff> {
  return request<TaskDiff>(`/tasks/${taskId}/diff`);
}

export function approveTask(taskId: number): Promise<TaskReviewResult> {
  return request<TaskReviewResult>(`/tasks/${taskId}/approve`, { method: "POST" });
}

export function rejectTask(taskId: number): Promise<TaskReviewResult> {
  return request<TaskReviewResult>(`/tasks/${taskId}/reject`, { method: "POST" });
}
