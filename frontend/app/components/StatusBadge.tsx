import React from "react";
import { needsHumanReview } from "@/lib/taskStatus";

const STATUS_STYLES: Record<string, string> = {
  pending: "bg-zinc-800 text-zinc-300 border-zinc-600",
  investigating: "bg-indigo-950 text-indigo-300 border-indigo-700",
  retrieving: "bg-indigo-950 text-indigo-300 border-indigo-700",
  planning: "bg-sky-950 text-sky-300 border-sky-700",
  editing: "bg-violet-950 text-violet-300 border-violet-700",
  testing: "bg-fuchsia-950 text-fuchsia-300 border-fuchsia-700",
  human_approval_required: "bg-amber-950 text-amber-300 border-amber-700",
  approval_failed: "bg-orange-950 text-orange-300 border-orange-700",
  approved: "bg-emerald-950 text-emerald-300 border-emerald-700",
  rejected: "bg-rose-950 text-rose-300 border-rose-700",
  failed: "bg-red-950 text-red-300 border-red-700",
  unable_to_verify: "bg-amber-950 text-amber-300 border-amber-700",
  no_change_needed: "bg-zinc-800 text-zinc-300 border-zinc-600",
};

const STATUS_LABELS: Record<string, string> = {
  human_approval_required: "Review Required",
  approval_failed: "Push/PR Failed",
  unable_to_verify: "Unable To Verify",
  no_change_needed: "No Change Needed",
};

function statusLabel(status: string): string {
  return (
    STATUS_LABELS[status] ??
    (status || "unknown").replace(/_/g, " ").toUpperCase()
  );
}

interface StatusBadgeProps {
  status: string;
  size?: "sm" | "md" | "lg";
  className?: string;
}

export default function StatusBadge({
  status,
  size = "md",
  className = "",
}: StatusBadgeProps) {
  const style =
    STATUS_STYLES[status] ?? "bg-zinc-800 text-zinc-300 border-zinc-600";

  const sizeClass =
    size === "lg"
      ? "text-sm px-4 py-1.5"
      : size === "sm"
        ? "text-[10px] px-2 py-0.5"
        : "text-xs px-3 py-1";

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-semibold tracking-wide ${style} ${sizeClass} ${className}`}
    >
      {needsHumanReview(status) && (
        <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
      )}
      {statusLabel(status)}
    </span>
  );
}
