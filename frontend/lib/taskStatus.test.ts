import { describe, expect, it } from "vitest";
import {
  classifyTaskStatus,
  isFinalTaskStatus,
  isTerminalTaskStatus,
  needsHumanReview,
} from "./taskStatus";

const ACTIVE_STATUSES = [
  "pending",
  "investigating",
  "retrieving",
  "planning",
  "editing",
  "testing",
  "analyzing_failure",
  undefined,
  null,
  "",
];

const NEEDS_REVIEW_STATUSES = ["human_approval_required", "approval_failed"];

const FINAL_STATUSES = [
  "approved",
  "rejected",
  "failed",
  "unable_to_verify",
  "no_change_needed",
];

describe("classifyTaskStatus", () => {
  it.each(ACTIVE_STATUSES)("classifies %s as active", (status) => {
    expect(classifyTaskStatus(status as string)).toBe("active");
  });

  it.each(NEEDS_REVIEW_STATUSES)("classifies %s as needs_review", (status) => {
    expect(classifyTaskStatus(status)).toBe("needs_review");
  });

  it.each(FINAL_STATUSES)("classifies %s as final", (status) => {
    expect(classifyTaskStatus(status)).toBe("final");
  });
});

describe("isTerminalTaskStatus", () => {
  it.each(ACTIVE_STATUSES)("is false for active status %s", (status) => {
    expect(isTerminalTaskStatus(status as string)).toBe(false);
  });

  it.each([...NEEDS_REVIEW_STATUSES, ...FINAL_STATUSES])(
    "is true for terminal status %s",
    (status) => {
      expect(isTerminalTaskStatus(status)).toBe(true);
    }
  );

  // Explicit regression coverage for the exact bug reported: these two
  // outcome statuses must stop polling / render as terminal.
  it("is true for unable_to_verify", () => {
    expect(isTerminalTaskStatus("unable_to_verify")).toBe(true);
  });

  it("is true for no_change_needed", () => {
    expect(isTerminalTaskStatus("no_change_needed")).toBe(true);
  });
});

describe("isFinalTaskStatus", () => {
  it.each(FINAL_STATUSES)("is true for final status %s", (status) => {
    expect(isFinalTaskStatus(status)).toBe(true);
  });

  it.each(NEEDS_REVIEW_STATUSES)(
    "is false for needs_review status %s (approval_failed keeps its retry behavior)",
    (status) => {
      expect(isFinalTaskStatus(status)).toBe(false);
    }
  );

  it.each(ACTIVE_STATUSES)("is false for active status %s", (status) => {
    expect(isFinalTaskStatus(status as string)).toBe(false);
  });
});

describe("needsHumanReview", () => {
  it.each(NEEDS_REVIEW_STATUSES)("is true for %s", (status) => {
    expect(needsHumanReview(status)).toBe(true);
  });

  it.each(FINAL_STATUSES)("is false for final status %s", (status) => {
    expect(needsHumanReview(status)).toBe(false);
  });

  it.each(ACTIVE_STATUSES)("is false for active status %s", (status) => {
    expect(needsHumanReview(status as string)).toBe(false);
  });
});
