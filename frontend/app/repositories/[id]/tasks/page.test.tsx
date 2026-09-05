import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import CreateTaskPage from "./page";
import type { Repository, Task } from "@/lib/api";

const { submitTaskMock, pushMock } = vi.hoisted(() => ({
  submitTaskMock: vi.fn(),
  pushMock: vi.fn(),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    submitTask: submitTaskMock,
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock }),
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
    id: 42,
    repository_id: 1,
    title: "Fix the bug",
    description: "Fix the bug",
    status: "investigating",
    attempts: 0,
    patch_content: null,
    test_output: null,
    pr_url: null,
    ...overrides,
  };
}

describe("CreateTaskPage", () => {
  beforeEach(() => {
    submitTaskMock.mockReset();
    pushMock.mockReset();
  });

  it("submits the task and navigates to the new task's detail route", async () => {
    submitTaskMock.mockResolvedValue(baseTask({ id: 42 }));
    const user = userEvent.setup();
    render(<CreateTaskPage />);

    await user.type(
      screen.getByPlaceholderText(/describe the bug/i),
      "Fix the off-by-one error"
    );
    await user.click(screen.getByRole("button", { name: /start coding agent/i }));

    await waitFor(() =>
      expect(submitTaskMock).toHaveBeenCalledWith(
        expect.objectContaining({ repository_id: 1, description: "Fix the off-by-one error" })
      )
    );
    await waitFor(() => expect(pushMock).toHaveBeenCalledWith("/repositories/1/tasks/42"));
  });

  it("shows the working state while submission is in flight", async () => {
    let resolveSubmit: (value: Task) => void = () => {};
    submitTaskMock.mockReturnValue(
      new Promise<Task>((resolve) => {
        resolveSubmit = resolve;
      })
    );
    const user = userEvent.setup();
    render(<CreateTaskPage />);

    await user.type(screen.getByPlaceholderText(/describe the bug/i), "Fix it");
    await user.click(screen.getByRole("button", { name: /start coding agent/i }));

    expect(await screen.findByText(/agent is working on the repository copy/i)).toBeInTheDocument();
    resolveSubmit(baseTask());
    await waitFor(() => expect(pushMock).toHaveBeenCalled());
  });

  it("surfaces a submission error without navigating", async () => {
    const { ApiError } = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
    submitTaskMock.mockRejectedValue(new ApiError(500, "Agent execution failed"));
    const user = userEvent.setup();
    render(<CreateTaskPage />);

    await user.type(screen.getByPlaceholderText(/describe the bug/i), "Fix it");
    await user.click(screen.getByRole("button", { name: /start coding agent/i }));

    expect(await screen.findByText("Agent execution failed")).toBeInTheDocument();
    expect(pushMock).not.toHaveBeenCalled();
  });
});
