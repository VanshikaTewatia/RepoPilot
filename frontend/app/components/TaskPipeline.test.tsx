import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import TaskPipeline from "./TaskPipeline";
import type { Task } from "@/lib/api";

function baseTask(overrides: Partial<Task> = {}): Task {
  return {
    id: 12,
    repository_id: 1,
    title: "Fix the bug",
    description: "Fix the bug",
    status: "failed",
    attempts: 3,
    patch_content: null,
    test_output: null,
    pr_url: null,
    ...overrides,
  };
}

// Phase 3C fix #6, ported to TaskPipeline: FAILED/UNABLE_TO_VERIFY/NO_CHANGE_NEEDED
// must surface the persisted test_output/detail instead of only a generic
// message, while FIXED (the approved path) stays unchanged.
describe("TaskPipeline terminal-outcome detail", () => {
  it("renders test_output for a FAILED outcome inside a collapsible Details section", () => {
    render(
      <TaskPipeline
        task={baseTask({
          status: "failed",
          test_output: "1 failed, 0 passed\nAssertionError: expected 5 got 4",
        })}
        creating={false}
        onTaskChange={() => {}}
      />
    );

    expect(screen.getByText("Details")).toBeInTheDocument();
    expect(screen.getByText(/AssertionError/)).toBeInTheDocument();
  });

  it("renders test_output for an UNABLE_TO_VERIFY outcome", () => {
    render(
      <TaskPipeline
        task={baseTask({
          status: "unable_to_verify",
          test_output: "Required tool 'gradle' is not available",
        })}
        creating={false}
        onTaskChange={() => {}}
      />
    );

    expect(screen.getByText("Details")).toBeInTheDocument();
    expect(screen.getByText(/gradle/)).toBeInTheDocument();
  });

  it("renders test_output for a NO_CHANGE_NEEDED outcome", () => {
    render(
      <TaskPipeline
        task={baseTask({
          status: "no_change_needed",
          test_output: "1 passed\nVerification passed without any code changes",
        })}
        creating={false}
        onTaskChange={() => {}}
      />
    );

    expect(screen.getByText("Details")).toBeInTheDocument();
    expect(
      screen.getByText(/Verification passed without any code changes/)
    ).toBeInTheDocument();
  });

  it("does not render a Details section when test_output is absent", () => {
    render(
      <TaskPipeline
        task={baseTask({ status: "failed", test_output: null })}
        creating={false}
        onTaskChange={() => {}}
      />
    );

    expect(screen.queryByText("Details")).not.toBeInTheDocument();
  });

  it("does not render a Details section for the approved (FIXED) path -- unchanged behavior", () => {
    render(
      <TaskPipeline
        task={baseTask({ status: "approved", test_output: "1 passed", pr_url: null })}
        creating={false}
        onTaskChange={() => {}}
      />
    );

    expect(screen.queryByText("Details")).not.toBeInTheDocument();
    expect(
      screen.getByText("Patch applied to the original repository.")
    ).toBeInTheDocument();
  });
});

describe("TaskPipeline stage timeline", () => {
  it("renders all six pipeline stage labels", () => {
    render(
      <TaskPipeline
        task={baseTask({ status: "testing" })}
        creating={false}
        onTaskChange={() => {}}
      />
    );

    for (const label of ["Investigation", "Evidence", "Plan", "Changes", "Tests", "Review"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
  });
});
