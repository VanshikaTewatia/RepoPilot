import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import DeepQAPanel from "./DeepQAPanel";
import { ApiError } from "@/lib/api";
import type { QAAnswer, Repository } from "@/lib/api";

const { askDeepQAMock } = vi.hoisted(() => ({
  askDeepQAMock: vi.fn(),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    askDeepQA: askDeepQAMock,
  };
});

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

function baseAnswer(overrides: Partial<QAAnswer> = {}): QAAnswer {
  return {
    summary: "subtotal() sums item prices.",
    details: null,
    flow_trace: null,
    evidence: [],
    corrected_premise: null,
    confidence: "inferred",
    projects_considered: ["."],
    ...overrides,
  };
}

const askButton = () => screen.getByRole("button", { name: /ask codebase/i });
const questionInput = () => screen.getByPlaceholderText(/ask a question/i);

describe("DeepQAPanel", () => {
  beforeEach(() => {
    askDeepQAMock.mockReset();
  });

  // A. no repository / disabled submit
  it("disables submission when no repository is selected, even with a question typed", async () => {
    const user = userEvent.setup();
    render(<DeepQAPanel selectedRepo={null} />);

    expect(askButton()).toBeDisabled();

    await user.type(questionInput(), "Where is the cart subtotal calculated?");
    expect(askButton()).toBeDisabled();
    expect(askDeepQAMock).not.toHaveBeenCalled();
  });

  it("disables submission when the question is blank, even with a repository selected", () => {
    render(<DeepQAPanel selectedRepo={repo()} />);
    expect(askButton()).toBeDisabled();
  });

  // B. successful simple answer
  it("renders a successful simple answer with its confidence badge", async () => {
    askDeepQAMock.mockResolvedValue(
      baseAnswer({ summary: "It's calculated in OrderService.calculate_order_total." })
    );
    const user = userEvent.setup();
    render(<DeepQAPanel selectedRepo={repo({ id: 7 })} />);

    await user.type(questionInput(), "Where is the total calculated?");
    await user.click(askButton());

    await waitFor(() =>
      expect(askDeepQAMock).toHaveBeenCalledWith({
        question: "Where is the total calculated?",
        repository_id: 7,
      })
    );

    expect(
      await screen.findByText(/calculate_order_total/)
    ).toBeInTheDocument();
    expect(screen.getByText("Inferred")).toBeInTheDocument();
  });

  // C. full structured answer
  it("renders every field of a full structured answer", async () => {
    const answer: QAAnswer = {
      summary: "Cart flow spans two steps.",
      details: "The extra detail paragraph.",
      flow_trace: [
        {
          order: 1,
          description: "Add item to cart",
          file_path: "src/cart.py",
          citation: { file_path: "src/cart.py", start_line: 1, end_line: 5, symbol_name: "add_item" },
        },
        {
          order: 2,
          description: "Compute subtotal",
          file_path: "src/cart.py",
          citation: null,
        },
      ],
      evidence: [{ file_path: "src/cart.py", start_line: 1, end_line: 5, symbol_name: "add_item" }],
      corrected_premise: "The question mentions Django, but this repository actually uses Flask.",
      confidence: "direct_evidence",
      projects_considered: ["backend", "frontend"],
    };
    askDeepQAMock.mockResolvedValue(answer);

    const user = userEvent.setup();
    render(<DeepQAPanel selectedRepo={repo()} />);
    await user.type(questionInput(), "Explain the cart flow");
    await user.click(askButton());

    expect(await screen.findByText("Cart flow spans two steps.")).toBeInTheDocument();
    expect(screen.getByText("The extra detail paragraph.")).toBeInTheDocument();
    expect(screen.getByText("Direct Evidence")).toBeInTheDocument();
    expect(
      screen.getByText(/mentions Django, but this repository actually uses Flask/)
    ).toBeInTheDocument();
    expect(screen.getByText("Add item to cart")).toBeInTheDocument();
    expect(screen.getByText("Compute subtotal")).toBeInTheDocument();
    // The citation "src/cart.py:1-5" appears twice: once as the flow step 1
    // chip, once as the evidence chip.
    expect(screen.getAllByText("src/cart.py:1-5")).toHaveLength(2);
    // Flow step 2 has no citation, so its chip falls back to the bare file_path.
    expect(screen.getByText("src/cart.py")).toBeInTheDocument();
    expect(
      screen.getByText(/Projects considered: backend, frontend/)
    ).toBeInTheDocument();
  });

  // D. no_evidence answer
  it("renders a no_evidence answer with only the badge and summary", async () => {
    askDeepQAMock.mockResolvedValue(
      baseAnswer({
        summary: "There is no evidence of the requested component/feature in this repository.",
        confidence: "no_evidence",
        evidence: [],
        projects_considered: ["."],
      })
    );
    const user = userEvent.setup();
    render(<DeepQAPanel selectedRepo={repo()} />);
    await user.type(questionInput(), "Where is the blockchain integration?");
    await user.click(askButton());

    expect(await screen.findByText("No Evidence")).toBeInTheDocument();
    expect(
      screen.getByText(/no evidence of the requested component\/feature/i)
    ).toBeInTheDocument();
    expect(screen.queryByText(/Citations:/)).not.toBeInTheDocument();
    expect(screen.queryByText("Flow")).not.toBeInTheDocument();
    expect(screen.queryByText(/Projects considered/)).not.toBeInTheDocument();
  });

  // E. API error
  it("surfaces an API error in the existing error-banner style", async () => {
    askDeepQAMock.mockRejectedValue(new ApiError(404, "Repository not found"));
    const user = userEvent.setup();
    render(<DeepQAPanel selectedRepo={repo()} />);
    await user.type(questionInput(), "Where is the cart subtotal calculated?");
    await user.click(askButton());

    expect(await screen.findByText("Repository not found")).toBeInTheDocument();
  });

  it("falls back to a generic message for a non-ApiError failure", async () => {
    askDeepQAMock.mockRejectedValue(new Error("boom"));
    const user = userEvent.setup();
    render(<DeepQAPanel selectedRepo={repo()} />);
    await user.type(questionInput(), "Where is the cart subtotal calculated?");
    await user.click(askButton());

    expect(
      await screen.findByText("Failed to query the codebase.")
    ).toBeInTheDocument();
  });

  // F. submitting/loading state
  it("shows a disabled loading state while the request is in flight", async () => {
    let resolvePromise: (value: QAAnswer) => void = () => {};
    askDeepQAMock.mockReturnValue(
      new Promise<QAAnswer>((resolve) => {
        resolvePromise = resolve;
      })
    );
    const user = userEvent.setup();
    render(<DeepQAPanel selectedRepo={repo()} />);
    await user.type(questionInput(), "Where is X?");
    await user.click(askButton());

    const loadingButton = screen.getByRole("button", { name: /investigating/i });
    expect(loadingButton).toBeDisabled();

    resolvePromise(baseAnswer());
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /ask codebase/i })).toBeInTheDocument()
    );
  });

  // G. repository switching does not display stale answer
  it("replaces the previous answer after switching repositories and asking again", async () => {
    const user = userEvent.setup();
    askDeepQAMock.mockResolvedValueOnce(baseAnswer({ summary: "Answer from repo A" }));
    const { rerender } = render(<DeepQAPanel selectedRepo={repo({ id: 1 })} />);

    await user.type(questionInput(), "Question one");
    await user.click(askButton());
    expect(await screen.findByText("Answer from repo A")).toBeInTheDocument();

    askDeepQAMock.mockResolvedValueOnce(baseAnswer({ summary: "Answer from repo B" }));
    rerender(<DeepQAPanel selectedRepo={repo({ id: 2 })} />);

    const input = questionInput();
    await user.clear(input);
    await user.type(input, "Question two");
    await user.click(askButton());

    expect(await screen.findByText("Answer from repo B")).toBeInTheDocument();
    expect(screen.queryByText("Answer from repo A")).not.toBeInTheDocument();
    expect(askDeepQAMock).toHaveBeenLastCalledWith({
      question: "Question two",
      repository_id: 2,
    });
  });
});
