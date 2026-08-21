import React from "react";

const STATUS_STYLES: Record<string, string> = {
  pending: "bg-slate-800 text-slate-300 border-slate-600",
  investigating: "bg-cyan-950 text-cyan-300 border-cyan-700",
  retrieving: "bg-cyan-950 text-cyan-300 border-cyan-700",
  planning: "bg-sky-950 text-sky-300 border-sky-700",
  editing: "bg-indigo-950 text-indigo-300 border-indigo-700",
  testing: "bg-violet-950 text-violet-300 border-violet-700",
  human_approval_required: "bg-amber-950 text-amber-300 border-amber-700",
  approved: "bg-emerald-950 text-emerald-300 border-emerald-700",
  rejected: "bg-rose-950 text-rose-300 border-rose-700",
  failed: "bg-red-950 text-red-300 border-red-700",
};

const STATUS_LABELS: Record<string, string> = {
  human_approval_required: "Review Required",
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
    STATUS_STYLES[status] ?? "bg-slate-800 text-slate-300 border-slate-600";

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
      {status === "human_approval_required" && (
        <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
      )}
      {statusLabel(status)}
    </span>
  );
}
