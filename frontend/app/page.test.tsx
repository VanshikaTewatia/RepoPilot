import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import Dashboard from "./page";
import type { Repository } from "@/lib/api";

const { listRepositoriesMock } = vi.hoisted(() => ({
  listRepositoriesMock: vi.fn(),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    listRepositories: listRepositoriesMock,
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

describe("Dashboard", () => {
  beforeEach(() => {
    listRepositoriesMock.mockReset();
  });

  it("shows a loading message while repositories are being fetched", () => {
    listRepositoriesMock.mockReturnValue(new Promise(() => {}));
    render(<Dashboard />);
    expect(screen.getByText(/loading repositories/i)).toBeInTheDocument();
  });

  it("renders a repository card for each registered repository, with no fabricated stats", async () => {
    listRepositoriesMock.mockResolvedValue([
      repo({ id: 1, name: "alpha", status: "indexed" }),
      repo({ id: 2, name: "beta", status: "pending" }),
    ]);
    render(<Dashboard />);

    expect(await screen.findByText("alpha")).toBeInTheDocument();
    expect(screen.getByText("beta")).toBeInTheDocument();
    expect(screen.queryByText(/loading repositories/i)).not.toBeInTheDocument();

    // Every repository link points at its own workspace route.
    expect(screen.getByRole("link", { name: /alpha/ })).toHaveAttribute(
      "href",
      "/repositories/1"
    );
  });

  it("shows the empty state with a Connect Repository CTA when there are no repositories", async () => {
    listRepositoriesMock.mockResolvedValue([]);
    render(<Dashboard />);

    expect(await screen.findByText(/connect your first repository/i)).toBeInTheDocument();
  });

  it("toggles the Connect Repository panel open and closed", async () => {
    listRepositoriesMock.mockResolvedValue([repo()]);
    const user = userEvent.setup();
    render(<Dashboard />);

    await screen.findByText("demo_repo");
    const toggle = screen.getByRole("button", { name: /connect repository/i });

    expect(screen.queryByText(/github url/i)).not.toBeInTheDocument();
    await user.click(toggle);
    expect(screen.getByText(/github url/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /cancel/i }));
    expect(screen.queryByText(/github url/i)).not.toBeInTheDocument();
  });

  it("surfaces a load error with a retry action", async () => {
    const { ApiError } = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
    listRepositoriesMock.mockRejectedValueOnce(new ApiError(500, "Backend unreachable"));
    listRepositoriesMock.mockResolvedValueOnce([repo()]);
    const user = userEvent.setup();
    render(<Dashboard />);

    expect(await screen.findByText("Backend unreachable")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /retry/i }));

    await waitFor(() => expect(listRepositoriesMock).toHaveBeenCalledTimes(2));
    expect(await screen.findByText("demo_repo")).toBeInTheDocument();
  });
});
