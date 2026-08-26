/**
 * Single source of truth for classifying backend `Task.status` values.
 *
 * The backend's exact snake_case status strings are used as-is everywhere
 * below (`unable_to_verify`, `no_change_needed`, `approval_failed`, ...) --
 * this module classifies them, it never renames or remaps them. Any future
 * display-label mapping belongs in StatusBadge, not here.
 *
 * A task status falls into exactly one category:
 *
 *  - "active":       the agent loop is still running (investigating,
 *                     retrieving, planning, editing, testing, ...). Keep
 *                     polling; the stepper shows progress.
 *  - "needs_review": the agent loop has stopped and is waiting on a human
 *                     decision via ReviewPanel (`human_approval_required`,
 *                     `approval_failed`). Polling stops, but the workflow
 *                     is not finished -- `approval_failed` in particular
 *                     can still be retried from here.
 *  - "final":        the workflow is completely finished; nothing further
 *                     will happen on its own or via review
 *                     (`approved`, `rejected`, `failed`, `unable_to_verify`,
 *                     `no_change_needed`). "Start Another Task" applies here.
 *
 * "Terminal" (see `isTerminalTaskStatus`) means `needs_review` OR `final`:
 * anything the agent loop will never advance further on its own, so polling
 * must stop and the "waiting for the next agent stage" message must never
 * be shown.
 */

export type TaskStatusCategory = "active" | "needs_review" | "final";

/** Agent loop stopped, waiting on a human decision (ReviewPanel). */
const NEEDS_REVIEW_STATUSES: ReadonlySet<string> = new Set([
  "human_approval_required",
  "approval_failed",
]);

/** Workflow fully finished; no further action is possible. */
const FINAL_STATUSES: ReadonlySet<string> = new Set([
  "approved",
  "rejected",
  "failed",
  "unable_to_verify",
  "no_change_needed",
]);

export function classifyTaskStatus(
  status: string | null | undefined
): TaskStatusCategory {
  if (status && NEEDS_REVIEW_STATUSES.has(status)) return "needs_review";
  if (status && FINAL_STATUSES.has(status)) return "final";
  return "active";
}

/** True for any status the agent loop will never advance further on its
 * own -- polling must stop and no "waiting for next stage" message should
 * be shown. */
export function isTerminalTaskStatus(status: string | null | undefined): boolean {
  const category = classifyTaskStatus(status);
  return category === "needs_review" || category === "final";
}

/** True only once the workflow is completely finished (no pending human
 * review step remains) -- e.g. safe to show "Start Another Task". */
export function isFinalTaskStatus(status: string | null | undefined): boolean {
  return classifyTaskStatus(status) === "final";
}

/** True while a human decision (approve/reject, or retrying a push/PR
 * failure) is pending -- e.g. render ReviewPanel. */
export function needsHumanReview(status: string | null | undefined): boolean {
  return classifyTaskStatus(status) === "needs_review";
}
