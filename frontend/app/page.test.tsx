import React from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Home from "./page";
import type { Repository, Task } from "@/lib/api";

/**
 * RepositoryPicker/TaskCreateForm are unrelated to the terminal-status bug
 * under test here -- they're replaced with trivial stand-ins that let the
 * test drive the real create-task -> poll -> terminal-outcome lifecycle
 * through the actual Home/TaskProgress/StatusBadge components.
 */
vi.mock("./components/RepositoryPicker", () => ({
  default: ({ onSelect }: { onSelect: (repo: Repository) => void }) => (
    <button
      onClick={() =>
        onSelect({
          id: 1,
          name: "demo_repo",
          local_path: "/demo",
          remote_url: null,
          default_branch: "main",
          status: "indexed",
          indexed_at: null,
        })
      }
    >
      select-repo
    </button>
  ),
}));

vi.mock("./components/TaskCreateForm", () => ({
  default: ({
    onSubmit,
  }: {
    onSubmit: (payload: { description: string }) => Promise<void>;
  }) => (
    <button onClick={() => onSubmit({ description: "Fix the bug" })}>
      create-task
    </button>
  ),
}));

const { submitTaskMock, getTaskMock, getTaskDiffMock } = vi.hoisted(() => ({
  submitTaskMock: vi.fn(),
  getTaskMock: vi.fn(),
  getTaskDiffMock: vi.fn(),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    submitTask: submitTaskMock,
    getTask: getTaskMock,
    getTaskDiff: getTaskDiffMock,
  };
});

function baseTask(overrides: Partial<Task>): Task {
  return {
    id: 12,
    repository_id: 1,
    title: "Fix the bug",
    description: "Fix the bug",
    status: "testing",
    attempts: 3,
    patch_content: null,
    test_output: null,
    pr_url: null,
    ...overrides,
  };
}

async function createRunningTask(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByText("select-repo"));
  await user.click(screen.getByText("create-task"));
  await waitFor(() => expect(submitTaskMock).toHaveBeenCalled());
}

describe("Home task lifecycle: terminal-status handling", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    submitTaskMock.mockReset();
    getTaskMock.mockReset();
    getTaskDiffMock.mockReset();
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
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    submitTaskMock.mockResolvedValue(baseTask({ status: "testing", attempts: 1 }));
    getTaskMock.mockResolvedValue(baseTask({ status: "unable_to_verify", attempts: 3 }));

    render(<Home />);
    await createRunningTask(user);

    // Poll tick: TaskProgress fetches the fresh (terminal) task.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });

    expect((await screen.findAllByText(/unable to verify/i)).length).toBeGreaterThan(0);
    expect(
      screen.queryByText(/waiting for the next agent stage/i)
    ).not.toBeInTheDocument();
    expect(screen.getByText("Start Another Task")).toBeInTheDocument();

    // Polling must have stopped: further ticks must not call getTask again.
    const callsAfterTerminal = getTaskMock.mock.calls.length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000);
    });
    expect(getTaskMock.mock.calls.length).toBe(callsAfterTerminal);
  });

  it("running -> no_change_needed: stops polling, shows terminal UI and Start Another Task, never shows the waiting message", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    submitTaskMock.mockResolvedValue(baseTask({ status: "testing", attempts: 1 }));
    getTaskMock.mockResolvedValue(baseTask({ status: "no_change_needed", attempts: 1 }));

    render(<Home />);
    await createRunningTask(user);

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

  it("Start Another Task resets the view immediately without a page reload", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    submitTaskMock.mockResolvedValue(baseTask({ status: "testing", attempts: 1 }));
    getTaskMock.mockResolvedValue(baseTask({ status: "unable_to_verify", attempts: 3 }));

    render(<Home />);
    await createRunningTask(user);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });

    const startAnotherButton = await screen.findByText("Start Another Task");
    await user.click(startAnotherButton);

    // Back to the create-task form; no terminal outcome UI remains.
    expect(screen.getByText("create-task")).toBeInTheDocument();
    expect(screen.queryByText("Start Another Task")).not.toBeInTheDocument();
    expect(screen.queryAllByText(/unable to verify/i).length).toBe(0);
  });

  it("approval_failed remains non-terminal for Start Another Task and still renders as needing review", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    submitTaskMock.mockResolvedValue(baseTask({ status: "testing", attempts: 1 }));
    getTaskMock.mockResolvedValue(baseTask({ status: "approval_failed", attempts: 1 }));

    render(<Home />);
    await createRunningTask(user);
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
