import React from "react";
import { Inter } from "next/font/google";
import "./globals.css";
import Sidebar from "./components/Sidebar";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

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
    <html lang="en" className={inter.variable}>
      <body className="bg-zinc-950 text-zinc-100 h-screen overflow-hidden font-sans antialiased">
        <div className="h-screen flex flex-col md:flex-row">
          <Sidebar />
          <div className="flex-1 flex flex-col min-w-0 min-h-0 overflow-y-auto">{children}</div>
        </div>
      </body>
    </html>
  );
}
