import React from "react";
import "./globals.css";

export const metadata = {
  title: "RepoPilot — Autonomous AI Software Engineer",
  description: "AI Agent for repository indexing, code-aware RAG, and self-correcting bug repair.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
