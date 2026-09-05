import React from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import TaskDetailPage from "./page";
import type { Repository, Task } from "@/lib/api";

const { getTaskMock, getTaskDiffMock, pushMock } = vi.hoisted(() => ({
  getTaskMock: vi.fn(),
  getTaskDiffMock: vi.fn(),
  pushMock: vi.fn(),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    getTask: getTaskMock,
    getTaskDiff: getTaskDiffMock,
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock }),
  useParams: () => ({ id: "1", taskId: "12" }),
}));

function repo(overrides: Partial<Repository> = {}): Repository {
  return {
    id: 1,
    name: "demo_repo",
    local_path: "/demo",
    remote_url: null,
    default_branch: "main",
    status: "indexed",
    indexed_at: null,
    ...overrides,
  };
}

vi.mock("@/lib/repository-context", () => ({
  useCurrentRepository: () => repo(),
}));

function baseTask(overrides: Partial<Task> = {}): Task {
  return {
    id: 12,
    repository_id: 1,
    title: "Fix the bug",
    description: "The pagination helper has an off-by-one error.",
    status: "testing",
    attempts: 1,
    patch_content: null,
    test_output: null,
    pr_url: null,
    ...overrides,
  };
}

describe("TaskDetailPage: terminal-status handling", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    getTaskMock.mockReset();
    getTaskDiffMock.mockReset();
    pushMock.mockReset();
    getTaskDiffMock.mockResolvedValue({
      task_id: 12,
      status: "approval_failed",
      diff: "",
      changed_files: [],
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("running -> unable_to_verify: stops polling, shows terminal UI and Start Another Task, never shows the waiting message", async () => {
    getTaskMock.mockResolvedValueOnce(baseTask({ status: "testing", attempts: 1 }));
    getTaskMock.mockResolvedValue(baseTask({ status: "unable_to_verify", attempts: 3 }));

    render(<TaskDetailPage />);
    await screen.findByText("Fix the bug");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });

    expect((await screen.findAllByText(/unable to verify/i)).length).toBeGreaterThan(0);
    expect(
      screen.queryByText(/waiting for the next agent stage/i)
    ).not.toBeInTheDocument();
    expect(screen.getByText("Start Another Task")).toBeInTheDocument();

    const callsAfterTerminal = getTaskMock.mock.calls.length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000);
    });
    expect(getTaskMock.mock.calls.length).toBe(callsAfterTerminal);
  });

  it("running -> no_change_needed: stops polling, shows terminal UI and Start Another Task, never shows the waiting message", async () => {
    getTaskMock.mockResolvedValueOnce(baseTask({ status: "testing", attempts: 1 }));
    getTaskMock.mockResolvedValue(baseTask({ status: "no_change_needed", attempts: 1 }));

    render(<TaskDetailPage />);
    await screen.findByText("Fix the bug");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });

    expect((await screen.findAllByText(/no change needed/i)).length).toBeGreaterThan(0);
    expect(
      screen.queryByText(/waiting for the next agent stage/i)
    ).not.toBeInTheDocument();
    expect(screen.getByText("Start Another Task")).toBeInTheDocument();

    const callsAfterTerminal = getTaskMock.mock.calls.length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000);
    });
    expect(getTaskMock.mock.calls.length).toBe(callsAfterTerminal);
  });

  it("Start Another Task navigates back to the create-task route for this repository", async () => {
    const user = (await import("@testing-library/user-event")).default.setup({
      advanceTimers: vi.advanceTimersByTime,
    });
    getTaskMock.mockResolvedValueOnce(baseTask({ status: "testing", attempts: 1 }));
    getTaskMock.mockResolvedValue(baseTask({ status: "unable_to_verify", attempts: 3 }));

    render(<TaskDetailPage />);
    await screen.findByText("Fix the bug");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });

    const startAnotherButton = await screen.findByText("Start Another Task");
    await user.click(startAnotherButton);

    expect(pushMock).toHaveBeenCalledWith("/repositories/1/tasks");
  });

  it("approval_failed remains non-terminal for Start Another Task and still renders as needing review", async () => {
    getTaskMock.mockResolvedValueOnce(baseTask({ status: "testing", attempts: 1 }));
    getTaskMock.mockResolvedValue(baseTask({ status: "approval_failed", attempts: 1 }));

    render(<TaskDetailPage />);
    await screen.findByText("Fix the bug");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });

    await waitFor(() =>
      expect(screen.getAllByText(/push\/pr failed/i).length).toBeGreaterThan(0)
    );
    // approval_failed keeps its existing retry behavior: no "Start Another
    // Task" affordance is shown while it's still awaiting a retry.
    expect(screen.queryByText("Start Another Task")).not.toBeInTheDocument();
  });
});
